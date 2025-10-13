//===- MaskedloadToLoad.cpp - Lowers maskedload to load -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPUMASKEDLOADTOLOADPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"
} // namespace mlir::amdgpu

using namespace mlir;
using namespace mlir::amdgpu;

/// This pattern supports lowering of: `vector.maskedload` to `vector.load`
/// and `arith.select` if the memref is in buffer address space.
static LogicalResult baseInBufferAddrSpace(PatternRewriter &rewriter,
                                           vector::MaskedLoadOp maskedOp) {
  auto memRefType = dyn_cast<MemRefType>(maskedOp.getBase().getType());
  if (!memRefType)
    return rewriter.notifyMatchFailure(maskedOp, "not a memref source");

  Attribute addrSpace = memRefType.getMemorySpace();
  if (!isa_and_nonnull<amdgpu::AddressSpaceAttr>(addrSpace))
    return rewriter.notifyMatchFailure(maskedOp, "no address space");

  if (dyn_cast<amdgpu::AddressSpaceAttr>(addrSpace).getValue() !=
      amdgpu::AddressSpace::FatRawBuffer)
    return rewriter.notifyMatchFailure(maskedOp, "not in buffer address space");

  return success();
}

static Value createVectorLoadForMaskedLoad(OpBuilder &builder, Location loc,
                                           vector::MaskedLoadOp maskedOp,
                                           bool passthru) {
  VectorType vectorType = maskedOp.getVectorType();
  Value load = vector::LoadOp::create(
      builder, loc, vectorType, maskedOp.getBase(), maskedOp.getIndices());
  if (passthru)
    load = arith::SelectOp::create(builder, loc, vectorType, maskedOp.getMask(),
                                   load, maskedOp.getPassThru());
  return load;
}

/// Check if the given value comes from a broadcasted i1 condition.
static FailureOr<Value> matchFullMask(OpBuilder &b, Value val) {
  auto broadcastOp = val.getDefiningOp<vector::BroadcastOp>();
  if (!broadcastOp)
    return failure();
  if (isa<VectorType>(broadcastOp.getSourceType()))
    return failure();
  return broadcastOp.getSource();
}

static constexpr char kMaskedloadNeedsMask[] =
    "amdgpu.buffer_maskedload_needs_mask";

namespace {

struct MaskedLoadLowering final : OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp maskedOp,
                                PatternRewriter &rewriter) const override {
    if (maskedOp->hasAttr(kMaskedloadNeedsMask))
      return failure();

    if (failed(baseInBufferAddrSpace(rewriter, maskedOp))) {
      return failure();
    }

      Value load = createVectorLoadForMaskedLoad(rewriter, maskedOp.getLoc(),
                                                 maskedOp, /*passthru=*/true);
      rewriter.replaceOp(maskedOp, load);
      return success();
  }
};

struct FullMaskedLoadToConditionalLoad
    : OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> maybeCond = matchFullMask(rewriter, loadOp.getMask());
    if (failed(maybeCond)) {
      return failure();
    }

    Value cond = maybeCond.value();
    auto trueBuilder = [&](OpBuilder &builder, Location loc) {
      Value res = createVectorLoadForMaskedLoad(builder, loc, loadOp,
                                                /*passthru=*/false);
      scf::YieldOp::create(rewriter, loc, res);
    };
    auto falseBuilder = [&](OpBuilder &builder, Location loc) {
      scf::YieldOp::create(rewriter, loc, loadOp.getPassThru());
    };
    auto ifOp = scf::IfOp::create(rewriter, loadOp.getLoc(), cond, trueBuilder,
                                  falseBuilder);
    rewriter.replaceOp(loadOp, ifOp);
    return success();
  }
};

struct FullMaskedStoreToConditionalStore
    : OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> maybeCond = matchFullMask(rewriter, storeOp.getMask());
    if (failed(maybeCond)) {
      return failure();
    }
    Value cond = maybeCond.value();

    auto trueBuilder = [&](OpBuilder &builder, Location loc) {
      vector::StoreOp::create(rewriter, loc, storeOp.getValueToStore(),
                              storeOp.getBase(), storeOp.getIndices());
      scf::YieldOp::create(rewriter, loc);
    };
    auto ifOp =
        scf::IfOp::create(rewriter, storeOp.getLoc(), cond, trueBuilder);
    rewriter.replaceOp(storeOp, ifOp);
    return success();
  }
};

} // namespace

void mlir::amdgpu::populateAmdgpuMaskedloadToLoadPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MaskedLoadLowering, FullMaskedLoadToConditionalLoad,
               FullMaskedStoreToConditionalStore>(patterns.getContext(),
                                                  benefit);
}

struct AmdgpuMaskedloadToLoadPass final
    : amdgpu::impl::AmdgpuMaskedloadToLoadPassBase<AmdgpuMaskedloadToLoadPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdgpuMaskedloadToLoadPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
