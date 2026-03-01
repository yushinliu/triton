#!/usr/bin/env python3
"""
示例脚本：展示 LinearLayout 的 get_2d_matrix_view() 功能

此脚本演示如何使用新的 get_2d_matrix_view() 方法将线性布局
打印为2维矩阵，矩阵中显示硬件参数分布。
"""

from triton.tools import LinearLayout


def print_matrix(matrix, title=""):
    """Helper function to print a 2D matrix nicely."""
    if title:
        print(f"\n{title}")
        print("=" * len(title))

    if not matrix:
        print("(empty matrix)")
        return

    # Calculate column widths for alignment
    col_widths = []
    for col_idx in range(len(matrix[0])):
        max_width = max(len(str(row[col_idx])) for row in matrix)
        col_widths.append(max_width + 2)

    # Print header
    header = "     " + " ".join(f"Col{i:2d}".center(col_widths[i]) for i in range(len(matrix[0])))
    print(header)
    print("     " + "-" * (sum(col_widths) + len(matrix[0]) - 1))

    # Print each row
    for row_idx, row in enumerate(matrix):
        cells = [str(cell).center(col_widths[i]) for i, cell in enumerate(row)]
        print(f"Row{row_idx:2d} |" + "|".join(cells) + "|")

    print()


def example_1_register_only():
    """示例1: 纯寄存器分布"""
    print("\n" + "=" * 60)
    print("示例 1: 纯寄存器分布 (register -> row)")
    print("=" * 60)

    layout = LinearLayout.from_bases(
        [
            ("register", [[1], [2]]),  # 4 registers -> 4 rows
        ],
        ["row"],
    )

    matrix = layout.get_2d_matrix_view()
    print_matrix(matrix, "寄存器到行的映射")

    print("说明: 每个单元格显示哪个寄存器映射到对应的行")
    print("      r0 = register 0, r1 = register 1, ...")


def example_2_thread_distribution():
    """示例2: 线程分布 (register, lane -> row, col)"""
    print("\n" + "=" * 60)
    print("示例 2: 线程分布 (register, lane -> row, col)")
    print("=" * 60)

    layout = LinearLayout.from_bases(
        [
            ("register", [[1, 0], [2, 0]]),  # 4 registers -> row
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes (threads) -> col
        ],
        ["row", "col"],
    )

    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"))
    print_matrix(matrix, "寄存器×线程 -> 行×列 映射")

    print("说明: r{register}t{thread} 格式")
    print("      r0t0 = register 0, thread 0")
    print("      r1t2 = register 1, thread 2")


def example_3_warp_distribution():
    """示例3: Warp分布"""
    print("\n" + "=" * 60)
    print("示例 3: Warp 分布 (lane, warp -> row, col)")
    print("=" * 60)

    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes per warp
            ("warp", [[1, 0], [2, 0]]),      # 4 warps
        ],
        ["row", "col"],
    )

    matrix = layout.get_2d_matrix_view(row_col_dims=("row", "col"))
    print_matrix(matrix, "线程×Warp -> 行×列 映射")

    print("说明: t{thread}w{warp} 格式")
    print("      全局线程ID = lane + warp * threads_per_warp (默认32)")


def example_4_full_hardware_params():
    """示例4: 完整硬件参数"""
    print("\n" + "=" * 60)
    print("示例 4: 完整硬件参数 (register, lane, warp, block)")
    print("=" * 60)

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
    print_matrix(matrix, "完整硬件参数映射")

    print("说明: r{register}t{thread}w{warp}b{block} 格式")
    print("      r = register (寄存器)")
    print("      t = thread/lane (线程)")
    print("      w = warp (线程束)")
    print("      b = block/CTA (线程块)")


def example_5_custom_threads_per_warp():
    """示例5: 自定义 threads_per_warp (AMD GPU)"""
    print("\n" + "=" * 60)
    print("示例 5: AMD GPU (64 threads per warp)")
    print("=" * 60)

    layout = LinearLayout.from_bases(
        [
            ("lane", [[0, 1], [0, 2]]),      # 4 lanes
            ("warp", [[1, 0], [2, 0]]),      # 4 warps
        ],
        ["row", "col"],
    )

    # 使用AMD风格的64线程 per warp
    matrix = layout.get_2d_matrix_view(threads_per_warp=64)
    print_matrix(matrix, "AMD GPU 风格映射 (64 threads/warp)")

    print("说明: 使用 threads_per_warp=64 计算全局线程ID")
    print("      全局线程ID = lane + warp * 64")


def example_6_single_dimension():
    """示例6: 单维输出"""
    print("\n" + "=" * 60)
    print("示例 6: 单维输出 (1D layout)")
    print("=" * 60)

    layout = LinearLayout.from_bases(
        [
            ("register", [[1], [2]]),  # 4 registers
        ],
        ["out"],
    )

    matrix = layout.get_2d_matrix_view()
    print_matrix(matrix, "单维输出")

    print("说明: 单维输出产生单列矩阵")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Triton LinearLayout 2D矩阵视图示例")
    print("=" * 60)
    print("\n本脚本演示 get_2d_matrix_view() 方法的功能:")
    print("- 将线性布局显示为2维矩阵")
    print("- 矩阵单元格显示硬件参数 (寄存器、线程、warp、block)")
    print("- 支持自定义维度选择和 threads_per_warp")

    try:
        example_1_register_only()
        example_2_thread_distribution()
        example_3_warp_distribution()
        example_4_full_hardware_params()
        example_5_custom_threads_per_warp()
        example_6_single_dimension()

        print("\n" + "=" * 60)
        print("所有示例完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n注意: 此脚本需要在编译后的 Triton 环境中运行")
        print("      请确保已执行 'python setup.py build_ext --inplace'")
