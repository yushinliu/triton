# ExtractTensorOp Lowering Notes

This note explains how `ExtractTensorOpConversion` in
`lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp` lowers `ttg.extract_tensor`
to a pure static register remap, using the test cases in
`python/test/unit/language/test_extract_tensor.py`.

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

### 3. Emit only static register moves

The lowering never builds extra arithmetic/select IR for the remap. It only
reorders unpacked source values:

```text
resultVals[regId] = srcVals[srcRegForDstReg[regId]]
```

Then it calls `packLLElements(..., dstTy)` so the same local register vector is
reinterpreted under the destination layout.

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

Representative examples:

- `regId = 0`
  - `srcMappingLL(reg=0,lane=0,warp=0) = [0,0]`
  - add offsets -> `[64,64]`
  - `getRegisterIdFromCoordinates(srcLL, [64,64]) = 40`
- `regId = 4`
  - `srcMappingLL(reg=4,lane=0,warp=0) = [32,0]`
  - add offsets -> `[96,64]`
  - `getRegisterIdFromCoordinates(srcLL, [96,64]) = 56`

So the final table is:

```text
srcRegForDstReg = [40, 41, 42, 43, 56, 57, 58, 59]
```

This is why `64x32` really spans two source replicas along `dim0` instead of
duplicating the first `32x32` tile.

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

Here:

```text
dstRegCount = 2
srcRegForDstReg = [24, 25]
```

So the lowering takes only the first two source registers from the selected
source replica and then packs them with the smaller destination layout.

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

The source side is still the same source replica, so:

```text
srcRegForDstReg = [24, 25]
```

The difference from Example B is not which source registers are read. The
difference is that `packLLElements(..., dstTy)` interprets those two registers
under the `blocked2` destination layout, so the same local register values land
at different logical `(dim0, dim1)` coordinates.

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
