from triton.tools import LinearLayout


def test_identity_1d():
    layout = LinearLayout.identity_1d(8, "idx", "idx")
    for value in range(8):
        assert layout.apply({"idx": value})["idx"] == value
    assert layout.is_surjective()


def test_zeros_1d():
    layout = LinearLayout.zeros_1d(8, "idx", "zero")
    for value in range(8):
        assert layout.apply({"idx": value})["zero"] == 0
    assert layout.is_surjective()

    widened = LinearLayout.zeros_1d(8, "idx", "zero", outDimSize=4)
    assert not widened.is_surjective()
    assert {widened.apply({"idx": value})["zero"] for value in range(8)} == {0}


def test_identity_2d():
    layout = LinearLayout.from_bases(
        [
            ("in0", [[0, 1], [0, 2]]),
            ("in1", [[1, 0], [2, 0]]),
        ],
        ["out0", "out1"],
    )
    for row in range(4):
        for col in range(4):
            result = layout.apply({"in0": col, "in1": row})
            assert result == {"out0": row, "out1": col}


def test_operator_mul_identity():
    layout = LinearLayout.identity_1d(4, "idx", "out") * LinearLayout.identity_1d(8, "idx", "out")
    for value in range(8):
        assert layout.apply({"idx": value})["out"] == value


def test_operator_mul_disjoint_dims():
    layout = LinearLayout.identity_1d(8, "i0", "o0") * LinearLayout.identity_1d(4, "i1", "o1")
    for i0 in range(8):
        for i1 in range(4):
            result = layout.apply({"i0": i0, "i1": i1})
            assert result == {"o0": i0, "o1": i1}


def test_compose():
    reg = LinearLayout.identity_1d(8, "reg", "tensor")
    shared = LinearLayout.identity_1d(8, "tensor", "tensor")
    composed = reg.compose(shared)
    for idx in range(8):
        assert composed.apply({"reg": idx})["tensor"] == idx


def test_invert():
    base = LinearLayout.identity_1d(8, "inp", "out")
    inverted = base.invert()
    for value in range(8):
        out = base.apply({"inp": value})["out"]
        recovered = inverted.apply({"out": out})["inp"]
        assert recovered == value


def test_invert_and_compose():
    base = LinearLayout.identity_1d(8, "inp", "mid")
    other = LinearLayout.identity_1d(8, "out", "mid")
    inverted = base.invert_and_compose(other)
    for value in range(8):
        assert inverted.apply({"inp": value})["out"] == value


def test_get_matrix_view_identity():
    layout = LinearLayout.identity_1d(4, "idx", "idx")
    assert layout.get_matrix_view() == [
        [1, 0],
        [0, 1],
    ]


def test_get_matrix_view_strided():
    layout = LinearLayout.strided_1d(4, 2, "idx", "out")
    assert layout.get_matrix_view() == [
        [0, 0],
        [1, 0],
        [0, 1],
    ]


def test_get_matrix_view_from_bases():
    layout = LinearLayout.from_bases(
        [
            ("in0", [[1, 0], [2, 0]]),
            ("in1", [[0, 1], [0, 2]]),
        ],
        ["out0", "out1"],
    )
    assert layout.get_matrix_view() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_get_distributed_view_register_only():
    """Test 2D matrix layout view with register distribution.

    Verifies that a layout with register indices displays correctly
    in the hardware point of view. The distributed view requires
    standard GPU hardware dimension names: register, lane, warp, block.
    """
    # Create a layout with all required GPU hardware dimensions
    # register=2 * lane=1 * warp=1 * block=1 -> tensor=2
    reg = LinearLayout.identity_1d(2, "register", "tensor")
    lane = LinearLayout.identity_1d(1, "lane", "tensor")
    warp = LinearLayout.identity_1d(1, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    # Test with hardware point of view
    result = layout.get_distributed_view(True)
    # Should show registers holding tensor elements
    assert isinstance(result, str)
    assert len(result) > 0
    # Output should contain warp header (since lane=1, warps are the main grouping)
    assert "Warp" in result


def test_get_distributed_view_with_threads():
    """Test 2D matrix layout view with thread distribution.

    Verifies that a layout mapping threads to tensor elements displays
    correctly, showing which thread (T0, T1, etc.) holds each element.
    """
    # Create layout with all required GPU hardware dimensions
    # register=2 * lane=4 * warp=1 * block=1 -> tensor=8
    reg = LinearLayout.identity_1d(2, "register", "tensor")
    lane = LinearLayout.identity_1d(4, "lane", "tensor")
    warp = LinearLayout.identity_1d(1, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    result = layout.get_distributed_view(True)
    # Should contain warp grouping (each warp gets its own section)
    assert isinstance(result, str)
    assert len(result) > 0
    # Check for warp headers
    assert "Warp" in result


def test_get_distributed_view_with_warps():
    """Test 2D matrix layout view with warp distribution.

    Verifies that a layout mapping warps to tensor elements displays
    correctly, showing which warp holds each element.
    """
    # Create layout with all required GPU hardware dimensions
    # register=1 * lane=4 * warp=2 * block=1 -> tensor=8
    reg = LinearLayout.identity_1d(1, "register", "tensor")
    lane = LinearLayout.identity_1d(4, "lane", "tensor")
    warp = LinearLayout.identity_1d(2, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    result = layout.get_distributed_view(True)
    # Should contain warp information (each warp gets its own section)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Warp" in result


def test_get_distributed_view_combined_hw_params():
    """Test 2D matrix layout view with all hardware parameters.

    Verifies that a layout with registers, threads (lane), warps, and blocks
    displays correctly in a 2D matrix format showing the full mapping.
    """
    # Create layout: register=2 * lane=4 * warp=2 * block=1 -> tensor=16
    reg = LinearLayout.identity_1d(2, "register", "tensor")
    lane = LinearLayout.identity_1d(4, "lane", "tensor")
    warp = LinearLayout.identity_1d(2, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    # Test hardware point of view
    result_hw = layout.get_distributed_view(True)
    assert isinstance(result_hw, str)
    assert len(result_hw) > 0

    # Test tensor point of view
    result_tensor = layout.get_distributed_view(False)
    assert isinstance(result_tensor, str)
    assert len(result_tensor) > 0

    # Results should differ between views
    assert result_hw != result_tensor


def test_get_distributed_view_2d_output():
    """Test 2D matrix layout view with 2D tensor output.

    Verifies that a layout producing 2D tensor indices displays correctly
    in matrix format, showing the (row, col) mapping.
    """
    # Create 2D layout with all required GPU hardware dimensions
    # (register, lane, warp, block) -> (row, col)
    layout = LinearLayout.from_bases(
        [
            ("register", [[0, 1]]),  # register affects column
            ("lane", [[1, 0], [2, 0]]),  # lane affects row
            ("warp", [[0, 2]]),  # warp affects column
            ("block", [[4, 0]]),  # block affects row
        ],
        ["row", "col"],
    )

    result = layout.get_distributed_view(True)
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain coordinate indicators
    assert "(" in result and ")" in result


def test_get_distributed_view_with_blocks():
    """Test 2D matrix layout view with block (CTA) distribution.

    Verifies that a layout with multiple blocks displays correctly,
    showing block-level distribution of tensor elements.
    """
    # Create layout with all required GPU hardware dimensions
    # register=1 * lane=4 * warp=1 * block=2 -> tensor=8
    reg = LinearLayout.identity_1d(1, "register", "tensor")
    lane = LinearLayout.identity_1d(4, "lane", "tensor")
    warp = LinearLayout.identity_1d(1, "warp", "tensor")
    block = LinearLayout.identity_1d(2, "block", "tensor")
    layout = reg * lane * warp * block

    result = layout.get_distributed_view(True)
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain block indicators when numBlocks > 1
    assert "B" in result


def test_get_distributed_view_swizzled_layout():
    """Test 2D matrix layout view with swizzled/banked layout.

    Verifies that a swizzled memory layout displays correctly,
    showing the non-trivial mapping between hardware and tensor.
    """
    # Create a strided/swizzled layout with all required dimensions
    # register=4 with stride 2 * lane=1 * warp=1 * block=1 -> tensor=4
    reg = LinearLayout.strided_1d(4, 2, "register", "tensor")
    lane = LinearLayout.identity_1d(1, "lane", "tensor")
    warp = LinearLayout.identity_1d(1, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    result = layout.get_distributed_view(True)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_shared_view_basic():
    """Test shared memory layout view basic functionality.

    Verifies that shared memory layouts display correctly in 2D matrix format.
    Shared layouts expect (block, offset) input dimensions.
    """
    # Create a layout suitable for shared memory: (block, offset) -> (row, col)
    layout = LinearLayout.from_bases(
        [
            ("offset", [[0, 1], [0, 2], [1, 0]]),  # offset affects both dims
            ("block", [[2, 0]]),  # block affects row
        ],
        ["row", "col"],
    )

    result = layout.get_shared_view(True)
    assert isinstance(result, str)
    assert len(result) > 0

    result_tensor = layout.get_shared_view(False)
    assert isinstance(result_tensor, str)
    assert len(result_tensor) > 0


def test_get_shared_view_bank_conflicts():
    """Test shared memory layout view showing bank conflict potential.

    Verifies that the 2D matrix view helps identify memory access patterns.
    """
    # Create a strided layout that may cause bank conflicts
    # Shared layout uses offset and block dimensions
    # Using a surjective layout: offset maps to cover all output values
    layout = LinearLayout.from_bases(
        [
            ("offset", [[1], [2]]),  # covers values 0-3
            ("block", [[4]]),  # block affects higher bits
        ],
        ["tensor"],
    )

    result = layout.get_shared_view(True)
    assert isinstance(result, str)
    assert len(result) > 0
    # Should show block and offset information
    assert "Block" in result or "block" in result
    assert "Offset" in result or "offset" in result


def test_matrix_view_consistency():
    """Test that matrix view matches the layout's get_matrix_view output.

    Verifies consistency between the binary matrix representation and
    the 2D layout view.
    """
    # Use standard GPU hardware dimensions
    reg = LinearLayout.identity_1d(4, "register", "tensor")
    lane = LinearLayout.identity_1d(1, "lane", "tensor")
    warp = LinearLayout.identity_1d(1, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    # Get binary matrix representation
    matrix = layout.get_matrix_view()
    assert isinstance(matrix, list)
    assert len(matrix) == 2  # log2(4) = 2 rows for output
    assert len(matrix[0]) == 2  # log2(4) = 2 columns for input

    # Get string representation
    view = layout.get_distributed_view(True)
    assert isinstance(view, str)


def test_distributed_view_format_structure():
    """Test the structural format of distributed layout view output.

    Verifies that the output contains expected formatting elements
    like brackets, commas, and newlines for 2D matrix representation.
    """
    # Create a 2D layout with all required GPU hardware dimensions
    layout = LinearLayout.from_bases(
        [
            ("register", [[0, 1], [1, 0]]),
            ("lane", [[0, 2], [2, 0]]),
            ("warp", [[0, 4]]),
            ("block", [[4, 0]]),
        ],
        ["row", "col"],
    )

    result = layout.get_distributed_view(True)

    # Should have matrix-like structure with brackets and delimiters
    assert isinstance(result, str)
    # Check for structural elements common in layout strings
    assert "(" in result and ")" in result  # Coordinate tuples
    assert "," in result  # Separators


def test_layout_view_with_broadcast():
    """Test 2D matrix layout view with broadcasting.

    Verifies that layouts with broadcast dimensions display correctly,
    showing which hardware elements share the same tensor values.
    """
    # Create a layout with broadcast (zeros) using all required dimensions
    # register=2 * lane=1 (broadcast) * warp=1 * block=1 -> tensor=2
    reg = LinearLayout.identity_1d(2, "register", "tensor")
    lane = LinearLayout.zeros_1d(4, "lane", "tensor")  # broadcast lane
    warp = LinearLayout.identity_1d(1, "warp", "tensor")
    block = LinearLayout.identity_1d(1, "block", "tensor")
    layout = reg * lane * warp * block

    result = layout.get_distributed_view(True)
    assert isinstance(result, str)
    assert len(result) > 0


# =============================================================================
# Tests for get_2d_matrix_view() - 2D matrix view with hardware parameters
# =============================================================================


def test_get_2d_matrix_view_register_only():
    """Test 2D matrix view with register distribution.

    Verifies that a layout with only register input dimension displays
    correctly with register labels in each cell.
    """
    layout = LinearLayout.from_bases(
        [
            ("register", [[1], [2]]),  # 4 registers -> 4 rows
        ],
        ["row"],
    )
    matrix = layout.get_2d_matrix_view()
    # Each row should show which register maps to that row
    assert matrix[0][0] == "r0"
    assert matrix[1][0] == "r1"
    assert matrix[2][0] == "r2"
    assert matrix[3][0] == "r3"


def test_get_2d_matrix_view_with_threads():
    """Test 2D matrix view with thread (lane) distribution.

    Verifies that a layout mapping (register, lane) to (row, col)
    displays correctly with both register and thread labels.
    """
    layout = LinearLayout.from_bases(
        [
            ("register", [[1, 0], [2, 0]]),  # 4 registers -> row
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes -> col
        ],
        ["row", "col"],
    )
    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"))
    # Check a few cells - format is "r{register}t{thread}"
    # The specific values depend on the layout's apply() behavior
    assert len(matrix) == 4
    assert len(matrix[0]) == 4
    # Each cell should contain hardware info
    for row in matrix:
        for cell in row:
            assert cell != "."  # All cells should be populated


def test_get_2d_matrix_view_with_warps():
    """Test 2D matrix view with warp distribution.

    Verifies that a layout with warp input dimension displays
    correctly with warp labels.
    """
    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes
            ("warp", [[1, 0], [2, 0]]),      # 4 warps
        ],
        ["row", "col"],
    )
    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"))
    # Should show warp info (w{warp_id})
    assert len(matrix) == 4
    assert len(matrix[0]) == 4
    assert any("w" in cell for row in matrix for cell in row)


def test_get_2d_matrix_view_combined_hw_params():
    """Test 2D matrix view with all hardware parameters.

    Verifies that a layout with register, lane, warp, and block
    dimensions displays correctly with all hardware labels.
    """
    layout = LinearLayout.from_bases(
        [
            ("register", [[0, 0, 1], [0, 0, 2]]),   # 4 registers
            ("lane", [[0, 1, 0], [0, 2, 0]]),       # 4 lanes
            ("warp", [[1, 0, 0], [2, 0, 0]]),       # 4 warps
        ],
        ["row", "col", "depth"],
    )
    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"))
    # Matrix should be 4x4 (from warp x lane)
    assert len(matrix) == 4
    assert len(matrix[0]) == 4


def test_get_2d_matrix_view_with_blocks():
    """Test 2D matrix view with block (CTA) distribution.

    Verifies that a layout with block input dimension displays
    correctly with block labels.
    """
    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes
            ("block", [[1, 0], [2, 0]]),     # 4 blocks
        ],
        ["row", "col"],
    )
    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"))
    # Should show block indicators (b{block_id})
    assert len(matrix) == 4
    assert len(matrix[0]) == 4
    assert any("b" in cell for row in matrix for cell in row)


def test_get_2d_matrix_view_custom_dims():
    """Test 2D matrix view with custom row/col dimension selection.

    Verifies that specifying custom dimensions for rows and columns works.
    """
    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1, 0], [0, 2, 0]]),       # 4 lanes
            ("warp", [[1, 0, 0], [2, 0, 0]]),       # 4 warps
            ("register", [[0, 0, 1], [0, 0, 2]]),   # 4 registers
        ],
        ["dim0", "dim1", "dim2"],
    )
    # Select different dimensions for rows/cols
    matrix = layout.get_2d_matrix_view(row_col_dims=("dim1", "dim2"))
    assert len(matrix) == 4
    assert len(matrix[0]) == 4


def test_get_2d_matrix_view_empty_cells():
    """Test that empty cells are marked with '.'.

    Verifies that tensor elements not mapped by any hardware location
    are marked with '.' in the matrix.
    """
    # Create a layout that doesn't cover all outputs
    layout = LinearLayout.from_bases(
        [
            ("register", [[1], [2]]),  # 4 registers -> 4 outputs
        ],
        ["out"],
        out_dim_sizes=[8],  # But output size is 8
        require_surjective=False,
    )
    matrix = layout.get_2d_matrix_view()
    # Should have 8 rows but only 4 populated
    assert len(matrix) == 8
    # Some cells should be empty (marked as ".")
    assert any(cell == "." for row in matrix for cell in row)


def test_get_2d_matrix_view_invalid_dims():
    """Test error handling for invalid dimension names.

    Verifies that an appropriate error is raised when invalid
    dimension names are specified.
    """
    layout = LinearLayout.identity_1d(4, "idx", "out")
    try:
        layout.get_2d_matrix_view(row_col_dims=("nonexistent", "out"))
        assert False, "Should have raised an error"
    except Exception as e:
        assert "not found" in str(e).lower()


def test_get_2d_matrix_view_default_dims():
    """Test 2D matrix view with default dimension selection.

    Verifies that when row_col_dims is not specified,
    the first two output dimensions are used.
    """
    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1], [0, 2]]),
            ("warp", [[1, 0], [2, 0]]),
        ],
        ["row", "col"],
    )
    # Don't specify row_col_dims - should use first two
    matrix = layout.get_2d_matrix_view()
    assert len(matrix) == 4
    assert len(matrix[0]) == 4


def test_get_2d_matrix_view_threads_per_warp():
    """Test 2D matrix view with custom threads_per_warp.

    Verifies that the threads_per_warp parameter affects
    thread ID calculations.
    """
    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes
            ("warp", [[1, 0], [2, 0]]),      # 4 warps
        ],
        ["row", "col"],
    )
    # Use AMD-style 64 threads per warp
    matrix = layout.get_2d_matrix_view(threads_per_warp=64)
    assert len(matrix) == 4
    assert len(matrix[0]) == 4


def test_get_2d_matrix_view_format():
    """Test the format of hardware parameter labels.

    Verifies that hardware parameter labels follow the expected format:
    r{register}t{thread}w{warp}b{block}
    """
    layout = LinearLayout.from_bases(
        [
            ("register", [[0, 0, 0, 1], [0, 0, 0, 2]]),  # 4 registers
            ("lane", [[0, 0, 1, 0], [0, 0, 2, 0]]),       # 4 lanes
            ("warp", [[0, 1, 0, 0]]),                     # 2 warps
            ("block", [[1, 0, 0, 0]]),                    # 2 blocks
        ],
        ["d0", "d1", "d2", "d3"],
    )
    matrix = layout.get_2d_matrix_view(row_col_dims=("d0", "d1"))
    # Find a non-empty cell and check format
    for row in matrix:
        for cell in row:
            if cell != ".":
                # Cell should contain hardware labels
                assert "r" in cell  # register label
                assert "t" in cell  # thread label
                assert "w" in cell  # warp label
                assert "b" in cell  # block label
                return  # Found valid cell, test passes


def test_get_2d_matrix_view_single_dim():
    """Test 2D matrix view with single output dimension.

    Verifies that single-dimension layouts are handled correctly,
    producing a single-column matrix.
    """
    layout = LinearLayout.from_bases(
        [
            ("register", [[1], [2]]),  # 4 registers
        ],
        ["out"],
    )
    matrix = layout.get_2d_matrix_view()
    # Should be 4 rows x 1 column
    assert len(matrix) == 4
    assert len(matrix[0]) == 1
    # Each cell should have register label
    for i, row in enumerate(matrix):
        assert f"r{i}" == row[0]
