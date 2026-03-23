// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 -verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_spans_source(%arg0: tensor<128x128xi32, #blocked>) {
    // expected-error @+1 {{invalid replica coordinate 1 at dimension 0}}
    %0 = ttg.extract_tensor %arg0 [1, 0] : tensor<128x128xi32, #blocked> -> tensor<128x32xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_coords_rank(%arg0: tensor<128x128xi32, #blocked>) {
    // expected-error @+1 {{replica coordinates must have the same rank as input}}
    %0 = ttg.extract_tensor %arg0 [0] : tensor<128x128xi32, #blocked> -> tensor<32x32xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_coords_value(%arg0: tensor<128x128xi32, #blocked>) {
    // expected-error @+1 {{invalid replica coordinate 4 at dimension 0}}
    %0 = ttg.extract_tensor %arg0 [4, 0] : tensor<128x128xi32, #blocked> -> tensor<32x32xi32, #blocked>
    tt.return
  }
}

// -----
