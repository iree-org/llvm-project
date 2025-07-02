//===- ReifyResultShapes.cpp - Reify result shapes ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform reifies result shapes of `ReifyRankedShapedTypeOpInterface`
// operations with ranked `memref` and `tensor` results.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "reify-result-shapes"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_REIFYRESULTSHAPESPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

/// Reifies the results of `op`, potentially replacing `op` with a reified
/// version. Returns `failure` if `mlir::reifyResultShapes` returned failure,
/// otherwise it always succeeds. Users of this transform should always expect
/// it to modify the IR, even when it fails. If any of the result types changes,
/// the transform will insert cast operations to the old type to keep the IR
/// consistent.
static LogicalResult reifyOpResultShapes(RewriterBase &rewriter,
                                         ReifyRankedShapedTypeOpInterface op) {
  LLVM_DEBUG({ DBGS() << " reifying op: " << op << "\n"; });
  // Get the reified out shapes.
  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(mlir::reifyResultShapes(rewriter, op, reifiedResultShapes)) ||
      reifiedResultShapes.empty()) {
    return op->emitWarning() << "failed to get the reified shapes";
  }

  for (auto [idx, reifiedShape] : llvm::enumerate(reifiedResultShapes)) {
    SmallVector<Value> vals =
        getValueOrCreateConstantIndexOp(rewriter, op->getLoc(), reifiedShape);
    vals.insert(vals.begin(), op->getResult(idx));
    OperationState state(op->getLoc(), "transform.materialize_shape");
    state.addOperands(vals);
    rewriter.create(state);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
struct ReifyResultShapesPass final
    : public memref::impl::ReifyResultShapesPassBase<ReifyResultShapesPass> {
  void runOnOperation() override;
};
} // namespace

void ReifyResultShapesPass::runOnOperation() {
  // 1. Select ops that are not DPS and that do not carry an tied operand
  // shapes. For now, limit to tensor::PadOp and tensor::ConcatOp.
  SmallVector<ReifyRankedShapedTypeOpInterface> ops;
  getOperation()->walk([&](ReifyRankedShapedTypeOpInterface op) {
    if (!isa<tensor::PadOp, tensor::ConcatOp>(op.getOperation()))
      return;
    ops.push_back(op);
  });

  // 2. Insert materialization points to tie the result tensor to its shape
  // components as SSA values.
  IRRewriter rewriter(&getContext());
  for (ReifyRankedShapedTypeOpInterface op : ops) {
    rewriter.setInsertionPoint(op);
    (void)reifyOpResultShapes(rewriter, op);
  }

  // 3. Resolve ranked shapes greedily for all other ops that implement
  // ReifyRankedShapedTypeOpInterface, achieving propagation of information.
  RewritePatternSet patterns(&getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();

  // 4. Process the information in the materialization points if more static
  // information is now available.
  getOperation()->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "transform.materialize_shape")
      return;
    auto resultShapedVal = cast<OpResult>(op->getOperands().front());

    // 4.a. Fold information propagated to AffineApplyOp.
    SmallVector<OpFoldResult> ofrs =
        getAsOpFoldResult(op->getOperands().drop_front());
    for (auto &ofr : ofrs) {
      if (isa<Attribute>(ofr))
        continue;
      if (auto affineApplyOp =
              (cast<Value>(ofr).getDefiningOp<affine::AffineApplyOp>())) {
        OpFoldResult o = affine::makeComposedFoldedAffineApply(
            rewriter, affineApplyOp->getLoc(), affineApplyOp.getAffineMap(),
            getAsOpFoldResult(affineApplyOp->getOperands()),
            /*composeAffineMin=*/true);
        if (isa<Attribute>(o))
          ofr = o;
      }
    }

    // 4.b. Erase the materialization point.
    rewriter.eraseOp(op);

    // 4.c. Clone the op and insert a better ShapeCastOp if the shape becomes
    // strictly more static.
    auto nst = cast<ShapedType>(resultShapedVal.getType());
    nst = nst.cloneWith(getInducedShape(ofrs), getElementTypeOrSelf(nst));
    Operation *oldOp = resultShapedVal.getDefiningOp();
    assert(llvm::isa_and_nonnull<ReifyRankedShapedTypeOpInterface>(oldOp));
    // 4.c.i. If the shape did not change, bail.
    auto onst = cast<ShapedType>(
        oldOp->getResultTypes()[resultShapedVal.getResultNumber()]);
    if (onst == nst)
      return;

    // 4.c.ii. If any shape dimension becomes less static, bail.
    for (auto [ns, os] : llvm::zip_equal(nst.getShape(), onst.getShape())) {
      if (ShapedType::isDynamic(ns) && !ShapedType::isDynamic(os))
        return;
    }

    // 4.c.ii. RAUW
    Operation *newOp = rewriter.clone(*oldOp);
    OpResult newRes = newOp->getResult(resultShapedVal.getResultNumber());
    newRes.setType(nst);
    rewriter.replaceAllUsesWith(resultShapedVal, newRes);
  });
}
