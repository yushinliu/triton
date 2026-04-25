#include <cstdlib>
#include <numeric>
#include <optional>

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-optimize-constancy-load"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZECONSTANCYLOAD
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static unsigned floorPowerOfTwo(unsigned value) {
  unsigned result = 1;
  while (result <= value / 2)
    result *= 2;
  return result;
}

static unsigned productExcept(ArrayRef<unsigned> values, unsigned dim) {
  unsigned result = 1;
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != dim)
      result *= value;
  }
  return result;
}

static bool isPointerTensor(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  return tensorTy && isa<PointerType>(tensorTy.getElementType());
}

static std::optional<BlockedEncodingAttr>
getConstancyLoadEncoding(triton::LoadOp loadOp,
                         ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  Value ptr = loadOp.getPtr();
  if (!isPointerTensor(ptr) || loadOp.getOther())
    return std::nullopt;

  auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
  auto layout = dyn_cast<BlockedEncodingAttr>(ptrTy.getEncoding());
  if (!layout)
    return std::nullopt;

  SmallVector<unsigned> order(layout.getOrder().begin(),
                              layout.getOrder().end());
  if (order.empty())
    return std::nullopt;

  SmallVector<int64_t> shapePerCTA = getShapePerCTA(ptrTy);
  unsigned dim = order[0];
  if (dim >= ptrTy.getRank() || dim >= shapePerCTA.size())
    return std::nullopt;

  unsigned targetPerThread = getNumElementsPerThreadForConstancyLoad(
      loadOp, order, axisInfoAnalysis, shapePerCTA);
  if (targetPerThread <= 1)
    return std::nullopt;

  SmallVector<unsigned> sizePerThread(layout.getSizePerThread().begin(),
                                      layout.getSizePerThread().end());
  if (dim >= sizePerThread.size())
    return std::nullopt;

  unsigned numElems = product<int64_t>(shapePerCTA);
  unsigned numThreads = product<unsigned>(layout.getWarpsPerCTA()) *
                        product<unsigned>(layout.getThreadsPerWarp());
  unsigned maxElemsPerThread = std::max(numElems / numThreads, 1u);
  unsigned otherDimsPerThread = productExcept(sizePerThread, dim);
  unsigned maxDimPerThread =
      std::max(maxElemsPerThread / otherDimsPerThread, 1u);

  targetPerThread =
      floorPowerOfTwo(std::min(targetPerThread, maxDimPerThread));
  if (targetPerThread <= sizePerThread[dim])
    return std::nullopt;

  sizePerThread[dim] = targetPerThread;
  unsigned numWarps = product<unsigned>(layout.getWarpsPerCTA());
  unsigned threadsPerWarp = product<unsigned>(layout.getThreadsPerWarp());
  auto newLayout = BlockedEncodingAttr::get(
      ptrTy.getContext(), ptrTy.getShape(), sizePerThread, order, numWarps,
      threadsPerWarp, layout.getCTALayout());

  LDBG("change constancy load layout from " << layout << " to " << newLayout);
  return newLayout;
}

struct OptimizeConstancyLoadPass
    : public impl::TritonGPUOptimizeConstancyLoadBase<
          OptimizeConstancyLoadPass> {
  void runOnOperation() override {
    if (std::getenv("TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT"))
      return;

    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    SmallVector<std::pair<triton::LoadOp, Attribute>> rewrites;
    moduleOp.walk([&](triton::LoadOp loadOp) {
      if (auto layout = getConstancyLoadEncoding(loadOp, axisInfoAnalysis))
        rewrites.push_back({loadOp, *layout});
    });

    for (auto &[loadOp, layout] : rewrites)
      convertDistributedOpEncoding(layout, loadOp.getOperation());
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
