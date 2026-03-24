# ExtractTensorOp Lowering Notes

This note explains how `ExtractTensorOpConversion` in
`lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp` lowers `ttg.extract_tensor`
to a pure static register remap, using the test cases in
`python/test/unit/language/test_extract_tensor.py`.

The current test file covers these representative cases:

1. `coords=[1,2]`, `dst=32x32`, same blocked layout
2. `coords=[1,2]`, `dst=64x32`, same blocked layout
3. `coords=[1,4]`, `dst=32x16`, same thread/warp layout with smaller `sizePerThread`
4. `coords=[1,4]`, `dst=64x16`, same thread/warp layout with smaller `sizePerThread`
5. `coords=[1,4]`, `dst=32x16`, `blocked2` reinterpretation
6. `order=[0,1]`, `coords=[1,2]`, `dst=32x32`, same blocked layout with `sizePerThread=[4,1]`
7. `order=[0,1]`, `coords=[1,1]`, `dst=32x64`, same blocked layout with `sizePerThread=[4,1]`
8. `order=[0,1]`, `coords=[2,2]`, `dst=16x32`, same blocked layout with `sizePerThread=[2,1]`
9. `order=[0,1]`, `coords=[2,1]`, `dst=16x64`, same blocked layout with `sizePerThread=[2,1]`

This document explains the common lowering algorithm first, then uses case 2,
case 3, and case 5 as concrete examples. Case 1 is the degenerate single-tile
form of case 2, and case 4 is the multi-tile form of case 3.

The important constraint is that this lowering does not generate extra LLVM IR
to compute indices at runtime. After the source tensor has been unpacked to
`srcVals`, the lowering only does:

1. Build a few `LinearLayout` objects at compile time.
2. Compute a static table `dst register -> src register`.
3. Materialize `resultVals[reg] = srcVals[srcRegForDstReg[reg]]`.
4. Re-pack `resultVals` with the destination layout.

Because of that, `ExtractTensorOp` here is not a general slice. It only works
for cases where the source register for each destination register is a static
function of the destination register id.

## Lowering Algorithm

The lowering in `ExtractTensorOpConversion::matchAndRewrite` does the
following.

### 1. Build three layouts

- `srcLL = toLinearLayout(srcTy)`
  This is the full source tensor layout.
- `dstLL = toLinearLayout(dstTy).transposeOuts(outDimNames)`
  This is the destination tensor layout.
- `srcMappingLL = getExtractTensorLinearLayout(srcTy, coverageShape).transposeOuts(outDimNames)`
  This is the source-side extraction layout. It is the source layout after
  clipping to a source span large enough to cover:
  - one full source replica when `dstShape <= replicaShape`
  - multiple replicas when `dstShape > replicaShape`

Here:

- `replicaShape = getShapePerCTATile(srcTy)`
- `coverageShape[dim] = max(replicaShape[dim], dstShape[dim])`
- `offsets[dim] = coords[dim] * dstShape[dim]`

The final output shape is still `dstShape`. `coverageShape` is only the
intermediate source span used to compute the static source register mapping.

### 2. Compute `srcRegForDstReg`

For each destination register id `regId`:

1. Apply `srcMappingLL` at hardware point `(register = regId, lane = 0, warp = 0, block = 0)`.
2. Add `offsets`.
3. Ask `srcLL` which source register owns that logical element.

That gives a compile-time array:

```text
srcRegForDstReg[regId] = source register index
```

### 2.1 How `srcMappingLL -> srcRegForDstReg -> ret` fits together

This is the core dataflow of the lowering.

`srcMappingLL` does not describe the final result layout. It describes the
logical coordinates exposed by the extracted source span. For a destination
register id `regId`, the lowering first asks:

```text
srcElemCoords = srcMappingLL(register=regId, lane=0, warp=0, block=0)
```

Then it shifts those coordinates by the tile offsets:

```text
srcElemCoords[dim] += coords[dim] * dstShape[dim]
```

At that point `srcElemCoords` is a logical coordinate inside the original
source tensor. The lowering then asks `srcLL` which source register owns that
logical element:

```text
srcRegForDstReg[regId] = getRegisterIdFromCoordinates(srcLL, srcElemCoords)
```

This produces a compile-time table from destination registers to source
registers.

Finally, lowering materializes the LLVM value for the result in two steps:

```text
resultVals[regId] = srcVals[srcRegForDstReg[regId]]
ret = packLLElements(resultVals, dstTy)
```

So the chain is:

```text
srcMappingLL
  -> logical source coordinates for each destination register
  -> srcRegForDstReg
  -> resultVals
  -> ret
```

This is why the LLVM lowering stays static. `srcMappingLL` and
`srcRegForDstReg` are both compile-time objects, and `ret` is built only by
reordering unpacked source registers and repacking them under `dstTy`.

### 2.2 Two End-to-End Examples

The easiest way to read the lowering is to follow the same three objects in
two different test cases:

- Example A: `64x32` same-layout extraction
- Example C: `32x16 -> blocked2` reinterpretation

They share the same pipeline:

```text
srcMappingLL
  -> which logical source element does dst register regId want?
  -> srcRegForDstReg
  -> which physical source register already holds that element?
  -> ret
  -> pack those chosen source registers under dstLL
```

#### Example A: `64x32` same-layout extraction

From Example A below:

- `coverageShape = [64, 32]`
- `offsets = [64, 64]`
- `dstRegCount = 8`

`srcMappingLL` is the clipped source extraction layout for `[64,32]`. For a
few destination registers:

```text
regId = 0: srcMappingLL(reg=0,lane=0,warp=0) = [0, 0]
regId = 1: srcMappingLL(reg=1,lane=0,warp=0) = [0, 1]
regId = 4: srcMappingLL(reg=4,lane=0,warp=0) = [32, 0]
```

After adding tile offsets:

```text
regId = 0: [0, 0]   + [64, 64] = [64, 64]
regId = 1: [0, 1]   + [64, 64] = [64, 65]
regId = 4: [32, 0]  + [64, 64] = [96, 64]
```

Then `srcLL` resolves those logical coordinates back to concrete source
register ids:

```text
regId = 0 -> srcRegForDstReg[0] = 40
regId = 1 -> srcRegForDstReg[1] = 41
regId = 4 -> srcRegForDstReg[4] = 56
```

So the full mapping is:

```text
srcRegForDstReg = [40, 41, 42, 43, 56, 57, 58, 59]
```

At LLVM lowering time, `srcVals` is the unpacked source register vector. The
result is materialized as:

```text
resultVals = [
  srcVals[40], srcVals[41], srcVals[42], srcVals[43],
  srcVals[56], srcVals[57], srcVals[58], srcVals[59],
]
ret = packLLElements(resultVals, dstTy)
```

Because `dstLL` matches the extracted source layout in this example, `ret`
represents a straightforward two-replica extraction along `dim0`.

#### Example C: `32x16 -> blocked2` reinterpretation

From Example C below:

- `coverageShape = [32, 32]`
- `offsets = [32, 64]`
- `dstRegCount = 2`

Here `srcMappingLL` is still the full single-replica source extraction layout,
so for the two destination registers:

```text
regId = 0: srcMappingLL(reg=0,lane=0,warp=0) = [0, 0]
regId = 1: srcMappingLL(reg=1,lane=0,warp=0) = [0, 1]
```

After adding offsets:

```text
regId = 0: [0, 0] + [32, 64] = [32, 64]
regId = 1: [0, 1] + [32, 64] = [32, 65]
```

Then `srcLL` resolves those coordinates:

```text
srcRegForDstReg = [24, 25]
```

So lowering again builds:

```text
resultVals = [srcVals[24], srcVals[25]]
ret = packLLElements(resultVals, dstTy)
```

The key difference from Example A is the last line. In Example C, `dstTy` uses
the `blocked2` destination layout, so `packLLElements` reinterprets the same
two source registers under a different `dstLL`. In other words:

- `srcMappingLL` decides which source elements are requested
- `srcRegForDstReg` decides which source registers contain them
- `ret` gets its final logical coordinates only when those registers are packed
  with the destination layout

That is why Example C changes the visible logical placement of values without
changing which source registers are read.

### 3. Emit only static register moves

The lowering never builds extra arithmetic/select IR for the remap. It only
reorders unpacked source values:

```text
resultVals[regId] = srcVals[srcRegForDstReg[regId]]
```

Then it calls `packLLElements(..., dstTy)` so the same local register vector is
reinterpreted under the destination layout.

The important detail is that `ret` is not created directly from
`srcMappingLL`. `srcMappingLL` only tells us which logical source element each
destination register wants. `srcRegForDstReg` converts that logical request
into a concrete source register index, and `packLLElements(..., dstTy)` is what
finally gives those reordered registers the destination layout meaning.

## GF(2) Table Convention

All `LinearLayout` diagrams below use the same convention:

- Columns are input-dimension basis bits.
- Rows are output-dimension basis bits.
- Both axes are powers of two.
- A cell contains `1` when the input basis bit contributes that output basis
  bit in GF(2).

For example, a column `reg4` means the `2^2` bit of the `register` input
dimension. A row `dim1:32` means the `2^5` bit of output dimension `dim1`.

## Example A: `128x128 -> 64x32`, Same Layout

This is the test case:

```text
src: tensor<128x128xf16, blocked<[1,4],[4,8],[8,1]>>
dst: tensor<64x32xf16,  blocked<[1,4],[4,8],[8,1]>>
coords = [1, 2]
```

So:

- `replicaShape = [32, 32]`
- `coverageShape = [64, 32]`
- `offsets = [64, 64]`

### A.1 Full Source Layout: `srcLL`

The full source layout for `blocked<[1,4],[4,8],[8,1]>` on `128x128` has these
bases:

```text
register: [0,1] [0,2] [0,32] [0,64] [32,0] [64,0]
lane:     [0,4] [0,8] [0,16] [1,0] [2,0]
warp:     [4,0] [8,0] [16,0]
```

GF(2) table:

| out \ in | reg1 | reg2 | reg4 | reg8 | reg16 | reg32 | lane1 | lane2 | lane4 | lane8 | lane16 | warp1 | warp2 | warp4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dim0:1  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| dim0:2  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| dim0:4  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| dim0:8  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| dim0:16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| dim0:32 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim0:64 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:1  | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:2  | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:4  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:8  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| dim1:32 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:64 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### A.2 Source Extraction Layout: `srcMappingLL` for `coverageShape = [64, 32]`

Clipping the full source layout to `[64, 32]` removes dead register bases
`[0,32]`, `[0,64]`, and `[64,0]`, but preserves lane/warp participation:

```text
register: [0,1] [0,2] [32,0]
lane:     [0,4] [0,8] [0,16] [1,0] [2,0]
warp:     [4,0] [8,0] [16,0]
```

GF(2) table:

| out \ in | reg1 | reg2 | reg4 | lane1 | lane2 | lane4 | lane8 | lane16 | warp1 | warp2 | warp4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dim0:1  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| dim0:2  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| dim0:4  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| dim0:8  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| dim0:16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| dim0:32 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:1  | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:2  | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:4  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:8  | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:16 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

### A.3 Destination Layout: `dstLL`

For this case, `dstLL` is exactly the same as `srcMappingLL`, because the
destination layout is also `blocked<[1,4],[4,8],[8,1]>` on `64x32`.

So:

```text
dstRegCount = 8
```

### A.4 Static Register Mapping

For `coords = [1, 2]`, the lowering adds `offsets = [64, 64]` after
applying `srcMappingLL`.

The way to read the GF(2) tables is:

- First use `A.3` to know how many destination registers exist.
  `dstLL` has three register input bits (`reg1`, `reg2`, `reg4`), so
  `dstRegCount = 2^3 = 8`.
- For each `regId`, apply `A.2` at `(lane=0, warp=0, block=0)`.
  Since all lane/warp inputs are zero, only the register columns in `A.2`
  contribute.
- Then add offsets `[64,64]`.
- Finally use the full source table in `A.1` to convert the absolute logical
  source coordinate back into a source register id.

Representative examples:

- `regId = 0 = 000b`
  - In `A.2`, no register column is enabled, so
    `srcMappingLL(reg=0,lane=0,warp=0) = [0,0]`
  - Add offsets -> `[64,64]`
  - In `A.1`, `dim0:64` is driven by `reg32` and `dim1:64` is driven by `reg8`
  - So the source register id is `32 + 8 = 40`
- `regId = 1 = 001b`
  - In `A.2`, only `reg1` is enabled, and it contributes `dim1:1`
  - So `srcMappingLL(reg=1,lane=0,warp=0) = [0,1]`
  - Add offsets -> `[64,65]`
  - In `A.1`, `[64,65]` means `reg32 + reg8 + reg1`
  - So the source register id is `32 + 8 + 1 = 41`
- `regId = 4 = 100b`
  - In `A.2`, only `reg4` is enabled, and it contributes `dim0:32`
  - So `srcMappingLL(reg=4,lane=0,warp=0) = [32,0]`
  - Add offsets -> `[96,64]`
  - In `A.1`, `[96,64]` means `dim0:64 + dim0:32 + dim1:64`
  - Those come from `reg32 + reg16 + reg8`
  - So the source register id is `32 + 16 + 8 = 56`

So the final table is:

```text
srcRegForDstReg = [40, 41, 42, 43, 56, 57, 58, 59]
```

From there the LLVM-side result is:

```text
resultVals = [
  srcVals[40], srcVals[41], srcVals[42], srcVals[43],
  srcVals[56], srcVals[57], srcVals[58], srcVals[59],
]
ret = packLLElements(resultVals, dstTy)
```

Because `dstTy` uses the same layout family as the extracted source span, this
is a direct two-replica extraction along `dim0` instead of duplicating the
first `32x32` tile.

## Example A': `128x128 -> 64x16`, Same Layout Family with Smaller `sizePerThread`

This is test case 4:

```text
src: tensor<128x128xf16, blocked<[1,4],[4,8],[8,1]>>
dst: tensor<64x16xf16,  blocked<[1,2],[4,8],[8,1]>>
coords = [1, 4]
```

So:

- `replicaShape = [32, 32]`
- `coverageShape = [64, 32]`
- `offsets = [64, 64]`

This case combines the two mechanisms that matter in this lowering:

- `64` rows means the source-side extraction span must cover two source replicas
  along `dim0`
- `16` columns with `sizePerThread=[1,2]` means the destination consumes fewer
  destination registers than the source-side extraction span exposes

The lowering still follows the same steps:

1. Build `srcLL`
2. Build `srcMappingLL` for `coverageShape=[64,32]`
3. Build `dstLL` for `64x16`
4. Compute a static `srcRegForDstReg`
5. Re-pack those source registers using the destination layout

The key point is that `coverageShape` is driven by the source span needed to
derive the mapping, while `dstRegCount` is driven by the actual destination
layout. So this case is "multi-replica" on the source side, but still "partial
register-vector" on the destination side.

## Example B: `128x128 -> 32x16`, Same Layout but Smaller `sizePerThread`

This is the test case:

```text
src: tensor<128x128xf16, blocked<[1,4],[4,8],[8,1]>>
dst: tensor<32x16xf16,  blocked<[1,2],[4,8],[8,1]>>
coords = [1, 4]
```

So:

- `replicaShape = [32, 32]`
- `coverageShape = [32, 32]`
- `offsets = [32, 64]`

The important point is that `coverageShape` is still `[32,32]`, not `[32,16]`.
The result shape is `32x16`, but the source-side extraction span stays a full
replica. The lowering then only consumes the first `dstRegCount` destination
registers from that source span.

### B.1 Source Extraction Layout: `srcMappingLL` for `[32, 32]`

The extracted source layout is a full source replica:

```text
register: [0,1] [0,2]
lane:     [0,4] [0,8] [0,16] [1,0] [2,0]
warp:     [4,0] [8,0] [16,0]
```

GF(2) table:

| out \ in | reg1 | reg2 | lane1 | lane2 | lane4 | lane8 | lane16 | warp1 | warp2 | warp4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dim0:1  | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| dim0:2  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| dim0:4  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| dim0:8  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| dim0:16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| dim1:1  | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:2  | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:4  | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:8  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:16 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

### B.2 Destination Layout: `dstLL` on `32x16`

The destination layout now has fewer register bits:

```text
register: [0,1]
lane:     [0,2] [0,4] [0,8] [1,0] [2,0]
warp:     [4,0] [8,0] [16,0]
```

GF(2) table:

| out \ in | reg1 | lane1 | lane2 | lane4 | lane8 | lane16 | warp1 | warp2 | warp4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dim0:1  | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| dim0:2  | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| dim0:4  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| dim0:8  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| dim0:16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| dim1:1  | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:2  | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:4  | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:8  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

### B.3 Static Register Mapping

Here the same reasoning becomes smaller.

- From `B.2`, `dstLL` has only one register input bit (`reg1`), so
  `dstRegCount = 2`.
- For each destination register id, apply `B.1` at `(lane=0, warp=0, block=0)`.
  Only the register columns in `B.1` contribute.

So:

- `regId = 0 = 0b0`
  - no register bit is enabled in `B.1`
  - `srcMappingLL(reg=0,lane=0,warp=0) = [0,0]`
  - add offsets `[32,64]` -> `[32,64]`
  - in `A.1`, `dim0:32` comes from `reg16` and `dim1:64` comes from `reg8`
  - source register id = `16 + 8 = 24`
- `regId = 1 = 0b1`
  - `reg1` is enabled in `B.1`, so it contributes `dim1:1`
  - `srcMappingLL(reg=1,lane=0,warp=0) = [0,1]`
  - add offsets `[32,64]` -> `[32,65]`
  - in `A.1`, `[32,65]` means `reg16 + reg8 + reg1`
  - source register id = `16 + 8 + 1 = 25`

So:

```text
dstRegCount = 2
srcRegForDstReg = [24, 25]
```

Then lowering materializes:

```text
resultVals = [srcVals[24], srcVals[25]]
ret = packLLElements(resultVals, dstTy)
```

So the lowering takes only the first two destination registers from the
selected source replica and then packs them with the smaller destination
layout.

## Example C: `128x128 -> 32x16`, `blocked2` Reinterpretation

This is the test case:

```text
src: tensor<128x128xf16, blocked<[1,4],[4,8],[8,1]>>
dst: tensor<32x16xf16,  blocked<[1,2],[8,4],[8,1]>>
coords = [1, 4]
```

Again:

- `replicaShape = [32, 32]`
- `coverageShape = [32, 32]`
- `offsets = [32, 64]`

So `srcMappingLL` is still the same full-replica layout from Example B.

What changes is only `dstLL`.

### C.1 Destination Layout: `dstLL` for `blocked2`

`blocked2` on `32x16` has these bases:

```text
register: [0,1] [0,8]
lane:     [0,2] [0,4] [1,0] [2,0] [4,0]
warp:     [8,0] [16,0] [0,0]
```

GF(2) table:

| out \ in | reg1 | reg2 | lane1 | lane2 | lane4 | lane8 | lane16 | warp1 | warp2 | warp4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dim0:1  | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| dim0:2  | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| dim0:4  | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| dim0:8  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| dim0:16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| dim1:1  | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:2  | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:4  | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| dim1:8  | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The `warp4` column is all zeros. That zero basis is still preserved to keep the
same warp participation count even though it no longer contributes to the
logical coordinates.

### C.2 Static Register Mapping

The source-side reasoning is exactly the same as Example B, because `srcMappingLL`
is unchanged.

- `dstLL` still has one low register bit that determines `dstRegCount = 2`
- `srcMappingLL(reg=0,lane=0,warp=0) = [0,0]`
- `srcMappingLL(reg=1,lane=0,warp=0) = [0,1]`
- after adding offsets `[32,64]`, the absolute source coordinates are still
  `[32,64]` and `[32,65]`
- `A.1` still maps those to source register ids `24` and `25`

So:

```text
srcRegForDstReg = [24, 25]
```

Lowering therefore builds the same source-side vector:

```text
resultVals = [srcVals[24], srcVals[25]]
ret = packLLElements(resultVals, dstTy)
```

The difference from Example B is not which source registers are read. The
difference is that `packLLElements(..., dstTy)` now interprets those two
registers under the `blocked2` destination layout from `C.1`, so the same
local register values land at different logical `(dim0, dim1)` coordinates.

This is why the `blocked2` case is a register reinterpretation, not a
contiguous logical slice.

## Why `coverageShape` Exists

The result type is always `dstTy`. But the source span used to build
`srcMappingLL` is:

```text
coverageShape = max(replicaShape, dstShape)
```

That is required for two reasons:

1. `dstShape > replicaShape`
   The source span must expose multiple source replicas so the static register
   mapping can reach them.
2. `dstShape < replicaShape`
   The source span may still need the full source replica register stream before
   the destination layout reinterprets or truncates it.

So:

- `dstShape` says what the final result looks like.
- `coverageShape` says how much of the source local register space must be
  visible while deriving the static mapping.

## What LLVM IR Is Actually Emitted

For all supported cases, the lowering does not synthesize extra arithmetic or
selection IR in `ViewOpToLLVM.cpp`.

It only emits:

- `llvm.extractvalue` to unpack the source tensor struct
- a static reorder through `srcRegForDstReg`
- `llvm.insertvalue` to build the destination tensor struct

This is exactly why the verifier requires the mapping to be static.
