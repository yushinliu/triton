#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

LogicalResult decomposeFp4ToFpScaledOp(triton::Fp4ToFpScaledOp op,
                                       IRRewriter &rewriter) {
  auto loc = op.getLoc();
  auto *ctx = op.getContext();
  auto axis = op.getAxis();
  auto elemType = op.getType().getElementType();
  auto f16 = rewriter.getF16Type();
  auto f32 = rewriter.getF32Type();

  auto input = op.getInput();
  auto inputTy = input.getType();
  if (!inputTy.getEncoding())
    return op.emitError() << "expected fp4_to_fp_scaled input to have a "
                             "TritonGPU encoding before LLVM conversion";

  auto scale = op.getScale();
  auto scaleTy = scale.getType();
  if (!scaleTy.getEncoding())
    return op.emitError() << "expected fp4_to_fp_scaled scale to have a "
                             "TritonGPU encoding before LLVM conversion";

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return op.emitError() << "expected fp4_to_fp_scaled to be nested in a "
                             "module before LLVM conversion";

  auto fp4F16 =
      triton::gpu::Fp4ToFpOp::create(rewriter, loc, input, f16, axis);
  auto fp4F16Ty = fp4F16.getType();

  auto scaleFp8Ty = scaleTy.clone(Float8E4M3FNType::get(ctx));
  auto scaleFp8 =
      triton::BitcastOp::create(rewriter, loc, scaleFp8Ty, scale);
  auto scaleF16Ty = scaleTy.clone(f16);
  auto scaleF16 = cast<TypedValue<RankedTensorType>>(
      triton::FpToFpOp::create(rewriter, loc, scaleF16Ty,
                               scaleFp8.getResult(), RoundingModeAttr())
          .getResult());

  auto rank = scaleTy.getRank();
  SmallVector<int64_t> expandedScaleShape(scaleTy.getShape());
  expandedScaleShape.push_back(1);
  auto blockedEnc = triton::gpu::getDefaultBlockedEncoding(
      ctx, expandedScaleShape, triton::gpu::lookupNumWarps(op),
      triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp),
      triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp));
  auto sliceEnc = triton::gpu::SliceEncodingAttr::get(ctx, rank, blockedEnc);
  auto sliceTy = scaleF16Ty.cloneWithEncoding(sliceEnc);
  scaleF16 =
      triton::gpu::ConvertLayoutOp::create(rewriter, loc, sliceTy, scaleF16);

  auto expandedScale =
      triton::ExpandDimsOp::create(rewriter, loc, scaleF16, rank);

  SmallVector<int64_t> broadcastShape(scaleTy.getShape());
  broadcastShape.push_back(32);
  auto broadcastScale = triton::BroadcastOp::create(
      rewriter, loc, expandedScale.getType().clone(broadcastShape),
      expandedScale);

  auto transposeOrder = llvm::to_vector(llvm::seq<int32_t>(rank));
  transposeOrder.insert(transposeOrder.begin() + axis + 1, rank);
  auto transposedScale =
      triton::TransOp::create(rewriter, loc, broadcastScale, transposeOrder);

  SmallVector<int64_t> outputShape(scaleTy.getShape());
  outputShape[axis] *= 32;
  auto reshapedScale =
      triton::ReshapeOp::create(rewriter, loc, outputShape, transposedScale);

  auto scaleF16ForMul = triton::gpu::ConvertLayoutOp::create(
      rewriter, loc, fp4F16Ty, reshapedScale);

  auto fp4F32Ty = fp4F16Ty.clone(f32);
  auto fp4F32 =
      arith::ExtFOp::create(rewriter, loc, fp4F32Ty, fp4F16.getResult(),
                            arith::FastMathFlagsAttr());
  auto scaleF32 = arith::ExtFOp::create(
      rewriter, loc, fp4F32Ty, scaleF16ForMul.getResult(),
      arith::FastMathFlagsAttr());
  auto scaledF32 = arith::MulFOp::create(rewriter, loc, fp4F32, scaleF32);

  auto outputTy = fp4F16Ty.clone(elemType);
  Value output =
      arith::TruncFOp::create(rewriter, loc, outputTy, scaledF32).getResult();
  if (output.getType() != op.getType())
    output = triton::gpu::ConvertLayoutOp::create(rewriter, loc, op.getType(),
                                                  output);

  rewriter.replaceOp(op, output);
  return success();
}

} // namespace

LogicalResult mlir::triton::decomposeFp4ToFpScaledOps(ModuleOp module) {
  SmallVector<triton::Fp4ToFpScaledOp> ops;
  module.walk([&](triton::Fp4ToFpScaledOp op) { ops.push_back(op); });

  IRRewriter rewriter(module.getContext());
  for (auto op : ops) {
    rewriter.setInsertionPoint(op);
    if (failed(decomposeFp4ToFpScaledOp(op, rewriter)))
      return failure();
  }
  return success();
}
