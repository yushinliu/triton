// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_half = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @extract_blocked_replica(%arg0: tensor<128x128xi32, #blocked>) {
    // CHECK-LABEL: llvm.func @extract_blocked_replica
    // CHECK-NOT: llvm.add
    // CHECK-NOT: llvm.select
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-4: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %0 = ttg.extract_tensor %arg0 [1, 2] : tensor<128x128xi32, #blocked> -> tensor<32x32xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @extract_multi_replica(%arg0: tensor<128x128xi32, #blocked>) {
    // CHECK-LABEL: llvm.func @extract_multi_replica
    // CHECK-NOT: llvm.add
    // CHECK-NOT: llvm.select
    %0 = ttg.extract_tensor %arg0 [1, 2] : tensor<128x128xi32, #blocked> -> tensor<64x32xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_partial = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @extract_replica_partial(%arg0: tensor<128x128xi32, #blocked>) {
    // CHECK-LABEL: llvm.func @extract_replica_partial
    // CHECK-NOT: llvm.add
    // CHECK-NOT: llvm.select
    %0 = ttg.extract_tensor %arg0 [1, 4] : tensor<128x128xi32, #blocked> -> tensor<32x16xi32, #blocked_partial>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_partial = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @extract_multi_replica_partial(%arg0: tensor<128x128xi32, #blocked>) {
    // CHECK-LABEL: llvm.func @extract_multi_replica_partial
    // CHECK-NOT: llvm.add
    // CHECK-NOT: llvm.select
    %0 = ttg.extract_tensor %arg0 [1, 4] : tensor<128x128xi32, #blocked> -> tensor<64x16xi32, #blocked_partial>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @extract_blocked_relayout(%arg0: tensor<128x128xi32, #blocked>) {
    // CHECK-LABEL: llvm.func @extract_blocked_relayout
    // CHECK-NOT: llvm.add
    // CHECK-NOT: llvm.select
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-4: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %0 = ttg.extract_tensor %arg0 [1, 4] : tensor<128x128xi32, #blocked> -> tensor<32x16xi32, #blocked2>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.compute-capability" = 90 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @extract_blocked_relayout_warp64(%arg0: tensor<128x128xi32, #blocked>) {
    // CHECK-LABEL: llvm.func @extract_blocked_relayout_warp64
    // CHECK-NOT: llvm.add
    // CHECK-NOT: llvm.select
    // CHECK-COUNT-64: %{{.*}} = llvm.extractvalue %arg0[{{.*}}] : !llvm.struct
    // CHECK-COUNT-4: %{{.*}} = llvm.insertvalue %{{.*}} : !llvm.struct
    %0 = ttg.extract_tensor %arg0 [1, 4] : tensor<128x128xi32, #blocked> -> tensor<32x16xi32, #blocked2>
    tt.return
  }
}
