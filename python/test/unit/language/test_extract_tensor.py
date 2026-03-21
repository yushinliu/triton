import pathlib

import pytest
import torch

import triton


def _blocked_layout_for_warp_size(threads_per_warp: int):
    if threads_per_warp == 32:
        return "[4, 8]", "[8, 1]", 8
    if threads_per_warp == 64:
        return "[8, 8]", "[4, 1]", 4
    pytest.skip(f"unsupported warp size {threads_per_warp}")


def _expected_contiguous(x: torch.Tensor, expected_slice):
    return x[expected_slice].clone()


def _expected_blocked2_warp32(x: torch.Tensor):
    expected = torch.empty((32, 16), device=x.device, dtype=x.dtype)
    for dst_row in range(32):
        row_in_warp = dst_row % 8
        src_row = 32 + 4 * (dst_row // 8) + row_in_warp // 2
        for dst_col in range(16):
            group = dst_col // 8
            within_group = dst_col % 8
            src_col = (64 + 16 * (row_in_warp % 2) +
                       4 * (within_group // 2) + 2 * group +
                       (within_group % 2))
            expected[dst_row, dst_col] = x[src_row, src_col]
    return expected


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "result_shape, result_size_per_thread, result_threads_per_warp_layout, "
    "expected_fn, expected_slice",
    [
        ((32, 32), "[1, 4]", None, _expected_contiguous,
         (slice(32, 64), slice(64, 96))),
        ((32, 16), "[1, 2]", "[8, 4]", _expected_blocked2_warp32, None),
    ],
)
def test_extract_tensor_ttgir(dtype, result_shape, result_size_per_thread,
                              result_threads_per_warp_layout, expected_fn,
                              expected_slice, tmp_path: pathlib.Path, device):
    current_target = triton.runtime.driver.active.get_current_target()
    threads_per_warp = current_target.warp_size
    threads_per_warp_layout, warps_per_cta, num_warps = \
        _blocked_layout_for_warp_size(threads_per_warp)
    if expected_fn is _expected_blocked2_warp32 and threads_per_warp != 32:
        pytest.skip("32x16 blocked2 extract test currently covers warp_size=32")
    result_m, result_n = result_shape
    result_tpw_layout = result_threads_per_warp_layout or \
        threads_per_warp_layout

    ir = f"""
    #src_blocked = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = {threads_per_warp_layout}, warpsPerCTA = {warps_per_cta}, order = [1, 0]}}>
    #dst_blocked = #ttg.blocked<{{sizePerThread = {result_size_per_thread}, threadsPerWarp = {result_tpw_layout}, warpsPerCTA = {warps_per_cta}, order = [1, 0]}}>

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

        %tile = ttg.extract_tensor %src [1, 2] : tensor<128x128xf16, #src_blocked> -> tensor<{result_m}x{result_n}xf16, #dst_blocked>

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
    if expected_slice is not None:
        expected = expected_fn(x, expected_slice)
    else:
        expected = expected_fn(x)
    assert torch.equal(y, expected)
