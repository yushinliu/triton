import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_cuda


@triton.jit
def _constancy_load_kernel(x, y, BLOCK: tl.constexpr, GROUP: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    grouped_offsets = offsets // GROUP
    values = tl.load(x + grouped_offsets)
    tl.store(y + offsets, values)


@pytest.mark.skipif(not is_cuda(), reason="NVIDIA constancy load lowering only")
def test_constancy_load_layout_and_correctness(device):
    block = 128
    group = 4
    x = torch.randn((block // group, ), device=device, dtype=torch.float32)
    y = torch.empty((block, ), device=device, dtype=torch.float32)

    kernel = _constancy_load_kernel[(1, )](x, y, BLOCK=block, GROUP=group,
                                           num_warps=1)

    expected = x[torch.arange(block, device=device) // group]
    torch.testing.assert_close(y, expected)

    ttgir = kernel.asm["ttgir"]
    assert re.search(r"#ttg\.blocked<\{sizePerThread = \[4\]", ttgir), ttgir
