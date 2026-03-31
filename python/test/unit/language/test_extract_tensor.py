import ast
import functools
import pathlib

import pytest
import torch

import triton
from triton._C.libtriton import gluon_ir, ir
from triton.experimental.gluon import language as ttgl
from triton.tools import LinearLayout


def _blocked_layout_for_warp_size(threads_per_warp: int, order):
    if order == (1, 0):
        if threads_per_warp == 32:
            return [1, 4], [4, 8], [8, 1], 8
        if threads_per_warp == 64:
            return [1, 4], [8, 8], [4, 1], 4
    if order == (0, 1):
        if threads_per_warp == 32:
            return [4, 1], [8, 4], [1, 8], 8
        if threads_per_warp == 64:
            return [4, 1], [8, 8], [1, 4], 4
    pytest.skip(f"unsupported warp size {threads_per_warp}")


def _blocked2_threads_per_warp_layout(threads_per_warp: int):
    if threads_per_warp == 32:
        return [8, 4]
    if threads_per_warp == 64:
        return [16, 4]
    pytest.skip(f"unsupported warp size {threads_per_warp}")


def _expected_blocked2(x: torch.Tensor, threads_per_warp: int, coords):
    if threads_per_warp == 32:
        rows_per_unique_warp = 8
        src_rows_per_warp = 4
    elif threads_per_warp == 64:
        rows_per_unique_warp = 16
        src_rows_per_warp = 8
    else:
        pytest.skip(f"unsupported warp size {threads_per_warp}")

    base_row = coords[0] * 32
    base_col = coords[1] * 16
    expected = torch.empty((32, 16), device=x.device, dtype=x.dtype)
    for dst_row in range(32):
        row_in_warp = dst_row % rows_per_unique_warp
        src_row = (base_row + src_rows_per_warp * (dst_row // rows_per_unique_warp) +
                   row_in_warp // 2)
        for dst_col in range(16):
            group = dst_col // 8
            within_group = dst_col % 8
            src_col = (base_col + 16 * (row_in_warp % 2) +
                       4 * (within_group // 2) + 2 * group +
                       (within_group % 2))
            expected[dst_row, dst_col] = x[src_row, src_col]
    return expected


def _format_layout(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def _format_order(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


@functools.lru_cache(maxsize=None)
def _linear_layout(shape, size_per_thread, threads_per_warp, warps_per_cta,
                   order):
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = gluon_ir.GluonOpBuilder(ctx)
    layout = ttgl.BlockedLayout(list(size_per_thread), list(threads_per_warp),
                                list(warps_per_cta), list(order))
    linear = builder.to_linear_layout(layout._to_ir(builder), list(shape))
    return LinearLayout.from_bases(
        [
            ("register", linear.reg_bases),
            ("lane", linear.lane_bases),
            ("warp", linear.warp_bases),
            ("block", linear.block_bases),
        ],
        ["dim0", "dim1"],
        list(shape),
        require_surjective=True,
    )


def _extract_linear_layout(layout, extracted_shape):
    updated_bases = []
    for in_dim, bases in layout.bases:
        truncated_bases = []
        for basis in bases:
            truncated = [
                0 if basis[dim] >= extracted_shape[dim] else basis[dim]
                for dim in range(len(extracted_shape))
            ]
            if in_dim == "register" and all(value == 0 for value in truncated):
                continue
            truncated_bases.append(truncated)
        updated_bases.append((in_dim, truncated_bases))
    return LinearLayout.from_bases(
        updated_bases,
        layout.get_out_dim_names(),
        list(extracted_shape),
        require_surjective=False,
    )


def _in_dim_size(layout, dim_name):
    bases = dict(layout.bases)
    return 1 << len(bases.get(dim_name, []))


def _get_register_id_from_coords(layout, coords):
    hardware = layout.pseudoinvert().apply({"dim0": coords[0], "dim1": coords[1]})
    reg = hardware.get("register")
    for dim_name, value in hardware.items():
        if dim_name != "register" and value != 0:
            raise AssertionError(f"non-static source mapping for {coords}: {hardware}")
    return reg


def _expected_extract_tensor_for_layout(
        x: torch.Tensor, src_shape, src_size_per_thread, src_threads_per_warp,
        warps_per_cta, src_order, result_shape, coords, result_size_per_thread,
        result_threads_per_warp_layout, result_order):
    src_size_per_thread = tuple(src_size_per_thread)
    src_threads_per_warp = tuple(src_threads_per_warp)
    warps_per_cta = tuple(warps_per_cta)
    replica_shape = tuple(
        size * threads * warps for size, threads, warps in zip(
            src_size_per_thread, src_threads_per_warp, warps_per_cta))
    coverage_shape = tuple(
        max(replica_dim, result_dim)
        for replica_dim, result_dim in zip(replica_shape, result_shape))
    src_start = tuple(
        coord * size for coord, size in zip(coords, result_shape))

    src_layout = _linear_layout(src_shape, src_size_per_thread,
                                src_threads_per_warp, warps_per_cta,
                                tuple(src_order))
    dst_layout = _linear_layout(result_shape, result_size_per_thread,
                                tuple(result_threads_per_warp_layout),
                                tuple(warps_per_cta), tuple(result_order))
    extracted_layout = _extract_linear_layout(src_layout, coverage_shape)
    src_reg_for_dst_reg = []
    for reg in range(_in_dim_size(dst_layout, "register")):
        extracted_coords = extracted_layout.apply({
            "register": reg,
            "lane": 0,
            "warp": 0,
            "block": 0,
        })
        src_reg_for_dst_reg.append(
            _get_register_id_from_coords(
                src_layout,
                (extracted_coords["dim0"] + src_start[0],
                 extracted_coords["dim1"] + src_start[1]),
            ))

    expected = torch.empty(result_shape, device=x.device, dtype=x.dtype)
    visited = torch.zeros(result_shape, device=x.device, dtype=torch.bool)
    for warp in range(_in_dim_size(dst_layout, "warp")):
        for lane in range(_in_dim_size(dst_layout, "lane")):
            for reg in range(_in_dim_size(dst_layout, "register")):
                hardware_point = {
                    "register": reg,
                    "lane": lane,
                    "warp": warp,
                    "block": 0,
                }
                dst_coords = dst_layout.apply(hardware_point)
                source_coords = src_layout.apply({
                    "register": src_reg_for_dst_reg[reg],
                    "lane": lane,
                    "warp": warp,
                    "block": 0,
                })
                dst_row, dst_col = dst_coords["dim0"], dst_coords["dim1"]
                src_row, src_col = source_coords["dim0"], source_coords["dim1"]

                value = x[src_row, src_col]
                if visited[dst_row, dst_col].item():
                    assert torch.equal(expected[dst_row, dst_col], value)
                    continue
                expected[dst_row, dst_col] = value
                visited[dst_row, dst_col] = True

    assert torch.all(visited).item()
    return expected


def _expected_extract_tensor(x: torch.Tensor, threads_per_warp: int, result_shape,
                             coords, src_order, result_size_per_thread,
                             result_threads_per_warp_layout):
    src_shape = (128, 128)
    src_size_per_thread, src_threads_per_warp, warps_per_cta, _ = \
        _blocked_layout_for_warp_size(threads_per_warp, src_order)
    return _expected_extract_tensor_for_layout(
        x,
        src_shape,
        src_size_per_thread,
        src_threads_per_warp,
        warps_per_cta,
        src_order,
        result_shape,
        coords,
        result_size_per_thread,
        result_threads_per_warp_layout,
        src_order,
    )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "coords, result_shape, result_size_per_thread, result_layout_kind, order",
    [
        ("[1, 2]", (32, 32), "[1, 4]", "same", "[1, 0]"),
        ("[1, 2]", (64, 32), "[1, 4]", "same", "[1, 0]"),
        ("[1, 4]", (32, 16), "[1, 2]", "same", "[1, 0]"),
        ("[1, 4]", (64, 16), "[1, 2]", "same", "[1, 0]"),
        ("[1, 4]", (32, 16), "[1, 2]", "blocked2", "[1, 0]"),
        ("[1, 2]", (32, 32), "[4, 1]", "same", "[0, 1]"),
        ("[1, 1]", (32, 64), "[4, 1]", "same", "[0, 1]"),
        ("[2, 2]", (16, 32), "[2, 1]", "same", "[0, 1]"),
        ("[2, 1]", (16, 64), "[2, 1]", "same", "[0, 1]"),
    ],
)
def test_extract_tensor_ttgir(dtype, coords, result_shape, result_size_per_thread,
                              result_layout_kind, order, tmp_path: pathlib.Path,
                              device):
    current_target = triton.runtime.driver.active.get_current_target()
    threads_per_warp = current_target.warp_size
    parsed_order = tuple(ast.literal_eval(order))
    src_size_per_thread, threads_per_warp_layout, warps_per_cta, num_warps = \
        _blocked_layout_for_warp_size(threads_per_warp, parsed_order)
    result_m, result_n = result_shape
    if result_layout_kind == "blocked2":
        result_tpw_layout = _blocked2_threads_per_warp_layout(threads_per_warp)
        result_order = [1, 0]
    else:
        result_tpw_layout = threads_per_warp_layout
        result_order = list(parsed_order)

    ir = f"""
    #src_blocked = #ttg.blocked<{{sizePerThread = {_format_layout(src_size_per_thread)}, threadsPerWarp = {_format_layout(threads_per_warp_layout)}, warpsPerCTA = {_format_layout(warps_per_cta)}, order = {_format_order(parsed_order)}}}>
    #dst_blocked = #ttg.blocked<{{sizePerThread = {result_size_per_thread}, threadsPerWarp = {_format_layout(result_tpw_layout)}, warpsPerCTA = {_format_layout(warps_per_cta)}, order = {_format_order(result_order)}}}>

    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
      tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst128 = arith.constant dense<128> : tensor<128x1xi32, #src_blocked>
        %cst_dst = arith.constant dense<{result_n}> : tensor<{result_m}x1xi32, #dst_blocked>

        %row = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #src_blocked}}>>
        %col = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 0, parent = #src_blocked}}>>
        %src_ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #src_blocked>
        %row2d = tt.expand_dims %row {{axis = 1 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 1, parent = #src_blocked}}>> -> tensor<128x1xi32, #src_blocked>
        %col2d = tt.expand_dims %col {{axis = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 0, parent = #src_blocked}}>> -> tensor<1x128xi32, #src_blocked>
        %row_ofs = arith.muli %row2d, %cst128 : tensor<128x1xi32, #src_blocked>
        %row_bcast = tt.broadcast %row_ofs : tensor<128x1xi32, #src_blocked> -> tensor<128x128xi32, #src_blocked>
        %col_bcast = tt.broadcast %col2d : tensor<1x128xi32, #src_blocked> -> tensor<128x128xi32, #src_blocked>
        %src_ofs = arith.addi %row_bcast, %col_bcast : tensor<128x128xi32, #src_blocked>
        %src_ptrs = tt.addptr %src_ptr, %src_ofs : tensor<128x128x!tt.ptr<f16>, #src_blocked>, tensor<128x128xi32, #src_blocked>
        %src = tt.load %src_ptrs : tensor<128x128x!tt.ptr<f16>, #src_blocked>

        %tile = ttg.extract_tensor %src {coords} : tensor<128x128xf16, #src_blocked> -> tensor<{result_m}x{result_n}xf16, #dst_blocked>

        %dst_row = tt.make_range {{end = {result_m} : i32, start = 0 : i32}} : tensor<{result_m}xi32, #ttg.slice<{{dim = 1, parent = #dst_blocked}}>>
        %dst_col = tt.make_range {{end = {result_n} : i32, start = 0 : i32}} : tensor<{result_n}xi32, #ttg.slice<{{dim = 0, parent = #dst_blocked}}>>
        %dst_ptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{result_m}x{result_n}x!tt.ptr<f16>, #dst_blocked>
        %dst_row2d = tt.expand_dims %dst_row {{axis = 1 : i32}} : tensor<{result_m}xi32, #ttg.slice<{{dim = 1, parent = #dst_blocked}}>> -> tensor<{result_m}x1xi32, #dst_blocked>
        %dst_col2d = tt.expand_dims %dst_col {{axis = 0 : i32}} : tensor<{result_n}xi32, #ttg.slice<{{dim = 0, parent = #dst_blocked}}>> -> tensor<1x{result_n}xi32, #dst_blocked>
        %dst_row_ofs = arith.muli %dst_row2d, %cst_dst : tensor<{result_m}x1xi32, #dst_blocked>
        %dst_row_bcast = tt.broadcast %dst_row_ofs : tensor<{result_m}x1xi32, #dst_blocked> -> tensor<{result_m}x{result_n}xi32, #dst_blocked>
        %dst_col_bcast = tt.broadcast %dst_col2d : tensor<1x{result_n}xi32, #dst_blocked> -> tensor<{result_m}x{result_n}xi32, #dst_blocked>
        %dst_ofs = arith.addi %dst_row_bcast, %dst_col_bcast : tensor<{result_m}x{result_n}xi32, #dst_blocked>
        %dst_ptrs = tt.addptr %dst_ptr, %dst_ofs : tensor<{result_m}x{result_n}x!tt.ptr<f16>, #dst_blocked>, tensor<{result_m}x{result_n}xi32, #dst_blocked>
        tt.store %dst_ptrs, %tile : tensor<{result_m}x{result_n}x!tt.ptr<f16>, #dst_blocked>
        tt.return
      }}
    }}
    """

    temp_file = tmp_path / "test_extract_tensor.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file), target=current_target)

    x = torch.randn((128, 128), device=device, dtype=dtype)
    y = torch.empty(result_shape, device=device, dtype=dtype)

    kernel[(1, 1, 1)](x.data_ptr(), y.data_ptr())
    parsed_coords = tuple(ast.literal_eval(coords))
    if result_layout_kind == "blocked2":
        expected = _expected_blocked2(x, threads_per_warp, parsed_coords)
    else:
        expected = _expected_extract_tensor(
            x,
            threads_per_warp,
            result_shape,
            parsed_coords,
            parsed_order,
            tuple(ast.literal_eval(result_size_per_thread)),
            tuple(result_tpw_layout),
        )
    assert torch.equal(y, expected)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_extract_tensor_ttgir_large_size_per_thread_order01(
        dtype, tmp_path: pathlib.Path, device):
    current_target = triton.runtime.driver.active.get_current_target()
    threads_per_warp = current_target.warp_size
    if threads_per_warp != 32:
        pytest.skip("the new layout is only defined for warp_size=32")

    src_shape = (64, 128)
    src_size_per_thread = (8, 4)
    src_threads_per_warp = (8, 4)
    warps_per_cta = (1, 4)
    src_order = (0, 1)
    result_shape = (64, 16)
    result_size_per_thread = (8, 1)
    result_threads_per_warp = src_threads_per_warp
    coords = (0, 4)

    ir_text = f"""
    #src_blocked = #ttg.blocked<{{sizePerThread = {_format_layout(src_size_per_thread)}, threadsPerWarp = {_format_layout(src_threads_per_warp)}, warpsPerCTA = {_format_layout(warps_per_cta)}, order = {_format_order(src_order)}}}>
    #dst_blocked = #ttg.blocked<{{sizePerThread = {_format_layout(result_size_per_thread)}, threadsPerWarp = {_format_layout(result_threads_per_warp)}, warpsPerCTA = {_format_layout(warps_per_cta)}, order = {_format_order(src_order)}}}>

    module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32}} {{
      tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst128 = arith.constant dense<128> : tensor<64x1xi32, #src_blocked>
        %cst_dst = arith.constant dense<16> : tensor<64x1xi32, #dst_blocked>

        %row = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #src_blocked}}>>
        %col = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 0, parent = #src_blocked}}>>
        %src_ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #src_blocked>
        %row2d = tt.expand_dims %row {{axis = 1 : i32}} : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #src_blocked}}>> -> tensor<64x1xi32, #src_blocked>
        %col2d = tt.expand_dims %col {{axis = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 0, parent = #src_blocked}}>> -> tensor<1x128xi32, #src_blocked>
        %row_ofs = arith.muli %row2d, %cst128 : tensor<64x1xi32, #src_blocked>
        %row_bcast = tt.broadcast %row_ofs : tensor<64x1xi32, #src_blocked> -> tensor<64x128xi32, #src_blocked>
        %col_bcast = tt.broadcast %col2d : tensor<1x128xi32, #src_blocked> -> tensor<64x128xi32, #src_blocked>
        %src_ofs = arith.addi %row_bcast, %col_bcast : tensor<64x128xi32, #src_blocked>
        %src_ptrs = tt.addptr %src_ptr, %src_ofs : tensor<64x128x!tt.ptr<f16>, #src_blocked>, tensor<64x128xi32, #src_blocked>
        %src = tt.load %src_ptrs : tensor<64x128x!tt.ptr<f16>, #src_blocked>

        %tile = ttg.extract_tensor %src [0, 4] : tensor<64x128xf16, #src_blocked> -> tensor<64x16xf16, #dst_blocked>

        %dst_row = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #dst_blocked}}>>
        %dst_col = tt.make_range {{end = 16 : i32, start = 0 : i32}} : tensor<16xi32, #ttg.slice<{{dim = 0, parent = #dst_blocked}}>>
        %dst_ptr = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x16x!tt.ptr<f16>, #dst_blocked>
        %dst_row2d = tt.expand_dims %dst_row {{axis = 1 : i32}} : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #dst_blocked}}>> -> tensor<64x1xi32, #dst_blocked>
        %dst_col2d = tt.expand_dims %dst_col {{axis = 0 : i32}} : tensor<16xi32, #ttg.slice<{{dim = 0, parent = #dst_blocked}}>> -> tensor<1x16xi32, #dst_blocked>
        %dst_row_ofs = arith.muli %dst_row2d, %cst_dst : tensor<64x1xi32, #dst_blocked>
        %dst_row_bcast = tt.broadcast %dst_row_ofs : tensor<64x1xi32, #dst_blocked> -> tensor<64x16xi32, #dst_blocked>
        %dst_col_bcast = tt.broadcast %dst_col2d : tensor<1x16xi32, #dst_blocked> -> tensor<64x16xi32, #dst_blocked>
        %dst_ofs = arith.addi %dst_row_bcast, %dst_col_bcast : tensor<64x16xi32, #dst_blocked>
        %dst_ptrs = tt.addptr %dst_ptr, %dst_ofs : tensor<64x16x!tt.ptr<f16>, #dst_blocked>, tensor<64x16xi32, #dst_blocked>
        tt.store %dst_ptrs, %tile : tensor<64x16x!tt.ptr<f16>, #dst_blocked>
        tt.return
      }}
    }}
    """

    temp_file = tmp_path / "test_extract_tensor_large_spt_order01.ttgir"
    temp_file.write_text(ir_text)
    kernel = triton.compile(str(temp_file), target=current_target)

    x = torch.randn(src_shape, device=device, dtype=dtype)
    y = torch.empty(result_shape, device=device, dtype=dtype)

    kernel[(1, 1, 1)](x.data_ptr(), y.data_ptr())
    expected = _expected_extract_tensor_for_layout(
        x,
        src_shape,
        src_size_per_thread,
        src_threads_per_warp,
        warps_per_cta,
        src_order,
        result_shape,
        coords,
        result_size_per_thread,
        result_threads_per_warp,
        src_order,
    )
    assert torch.equal(y, expected)
