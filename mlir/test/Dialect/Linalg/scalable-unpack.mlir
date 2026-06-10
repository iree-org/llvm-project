// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file --verify-diagnostics | FileCheck %s

// The transform ops below carry an optional `inner_tile_alignments` hint,
// written as a per-dimension keyword list:
//   - `Equal`:    the loop tile size equals the pack/unpack inner tile size.
//   - `Multiple`: the loop tile size is an integer multiple of the inner tile.
//   - `Unknown`:  the default; nothing is asserted for that dimension.

// Perfect scalable tiling - scalable tile sizes equal scalable inner
// tiles. Outer sizes of the tiled unpack should be 1's.

// CHECK-LABEL: func.func @perfect_CKkc_to_KC_scalable
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @perfect_CKkc_to_KC_scalable(%source: tensor<32x4x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c2_vscale = arith.muli %c2, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
      inner_tiles = [%c2_vscale, %c4_vscale] into %dest
      : tensor<32x4x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // The `Equal` hints assert the loop tile sizes (2 * vscale, 4 * vscale) equal the
      // inner tile sizes (2 * vscale, 4 * vscale).
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[2], [4]]
          inner_tile_alignments = [Equal, Equal]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Aligned scalable tiling - scalable tile sizes ([16] = 16*vscale, [8] =
// 8*vscale) that are integer multiples of the scalable inner tiles (2x 8*vscale
// and 2x 4*vscale), so tiling is aligned and the tiled unpack needs no trailing
// remainder slice.

// CHECK:       #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
// CHECK-LABEL: func.func @NCnc_to_NC_scalable_aligned
//  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
//  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//  CHECK-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
//      CHECK:    %[[RES:.*]] = scf.for
//      CHECK:      %[[OUTER0:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C8_VSCALE]]]
//      CHECK:      %[[OUTER1:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C4_VSCALE]]]
//      CHECK:      %[[SRC:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [%[[OUTER0]], %[[OUTER1]], %[[C8_VSCALE]], %[[C4_VSCALE]]]
//      CHECK:      %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
// CHECK-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @NCnc_to_NC_scalable_aligned(%source: tensor<4x8x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // The `Multiple` hints are passed to hint the alignment between the loop tile sizes
      // (16 * vscale, 8 * vscale) and the inner tile sizes (8 * vscale, 4 * vscale).
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [[16], [8]]
          inner_tile_alignments = [Multiple, Multiple]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Unaligned scalable tiling - static tile sizes not aligned to scalable
// inner tiles.

// CHECK-LABEL: func.func @NCnc_to_NC_scalable_unaligned
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
// CHECK:             %[[EXTRACT:.*]] = tensor.extract_slice %[[UNPACK]]
// CHECK:             tensor.insert_slice %[[EXTRACT]]
// CHECK:         return %[[RES]]
func.func @NCnc_to_NC_scalable_unaligned(%source: tensor<4x8x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %0 = linalg.unpack %source inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %dest
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      // No hint is passed: the static tile sizes (7, 5) are not aligned to the scalable
      // inner tiles (8 * vscale, 4 * vscale), so the tiled unpack keeps a remainder slice.
      %1, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [7, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Producer fusion - linalg.unpack with scalable inner tiles fused as a producer
// into an elementwise consumer that is tiled with scalable tile
// sizes that are an integer multiple of the inner tiles (16*vscale of 8*vscale,
// 8*vscale of 4*vscale). `Multiple` hint is passed accordingly.

// CHECK:       #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
// CHECK-LABEL: func.func @unpack_elemwise_scalable_multiple
//  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
//  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//  CHECK-DAG:    %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[OUTER0:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C8_VSCALE]]]
// CHECK:             %[[OUTER1:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C4_VSCALE]]]
// CHECK:             %[[SRC:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [%[[OUTER0]], %[[OUTER1]], %[[C8_VSCALE]], %[[C4_VSCALE]]]
// CHECK:             %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
// CHECK-SAME:            tensor<?x?x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.exp ins(%[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_elemwise_scalable_multiple(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c16, %vscale : index
  %t1 = arith.muli %c8, %vscale : index
  %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %1 = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.exp ins(%1: tensor<?x?xf32>)
                       outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %t0h, %t1h = transform.split_handle %mulis : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // The `Multiple` hints are passed to hint the alignment between the loop tile sizes
    // (16 * vscale, 8 * vscale) and the inner tile sizes (8 * vscale, 4 * vscale).
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [%t0h, %t1h] interchange [0, 1]
      inner_tile_alignments = [Multiple, Multiple]
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Producer fusion - same as above but the consumer is tiled with scalable tile
// sizes equal to the unpack inner tiles (8*vscale, 4*vscale). The `Equal` hint
// collapses the fused unpack's outer dims to 1 even though the tile-size and
// inner-tile SSA values are not provably equal.

// CHECK-LABEL: func.func @unpack_elemwise_scalable_equal
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[UNPACK:.*]] = linalg.unpack
// CHECK-SAME:            tensor<1x1x?x?xf32> -> tensor<?x?xf32>
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.exp ins(%[[UNPACK]]
// CHECK:         return %[[RES]]
func.func @unpack_elemwise_scalable_equal(%arg0: tensor<4x8x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c8, %vscale : index
  %t1 = arith.muli %c4, %vscale : index
  %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
  %1 = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<4x8x?x?xf32> -> tensor<?x?xf32>
  %2 = linalg.exp ins(%1: tensor<?x?xf32>)
                       outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %exp = transform.structured.match ops{["linalg.exp"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %t0h, %t1h = transform.split_handle %mulis : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // The `Equal` hints assert the loop tile sizes equal the inner tile sizes
    // (8 * vscale, 4 * vscale).
    %tiled, %loops:2 = transform.structured.fuse %exp tile_sizes [%t0h, %t1h] interchange [0, 1]
      inner_tile_alignments = [Equal, Equal]
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Producer fusion with a transposing consumer. `inner_tile_alignments` is a 
// caller-asserted hint indexed in the unpack's dest-dim order, so the caller
// must arrange it for the transpose.

// CHECK:       #[[$MAP_CEILDIV:.+]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
// CHECK-LABEL: func.func @unpack_transposed_consumer_scalable
//  CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG:    %[[VSCALE:.*]] = vector.vscale
//  CHECK-DAG:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
// Source dim 0 is `Multiple` -> outer size is `ceilDiv(loop tile, 8*vscale)` (dynamic);
// source dim 1 is `Equal` -> outer size collapses to 1 (the `1` in tensor<?x1x?x?xf32>).
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             %[[OUTER0:.*]] = affine.apply #[[$MAP_CEILDIV]](%{{.*}})[%[[C8_VSCALE]]]
// CHECK:             %[[SRC:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [%[[OUTER0]], 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1]
// CHECK-SAME:            tensor<2x4x?x?xf32> to tensor<?x1x?x?xf32>
// CHECK:             %[[UNPACK:.*]] = linalg.unpack %[[SRC]]
// CHECK-NOT:         tensor.extract_slice %[[UNPACK]]
// CHECK:             linalg.generic
// CHECK:         return %[[RES]]
func.func @unpack_transposed_consumer_scalable(%arg0: tensor<2x4x?x?xf32>, %out: tensor<?x?xf32>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %t0 = arith.muli %c4, %vscale : index
  %t1 = arith.muli %c16, %vscale : index
  %0 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %0
      : tensor<2x4x?x?xf32> -> tensor<?x?xf32>
  // Consumer reads %unpack transposed.
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%unpack : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%in: f32, %o: f32):
      linalg.yield %in : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %mulis = transform.structured.match ops{["arith.muli"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %i0, %i1, %h0, %h1 = transform.split_handle %mulis
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                  !transform.any_op, !transform.any_op)
    // Hint indexed in the unpack's dest-dim order: dim0 Multiple, dim1 Equal.
    %tiled, %loops:2 = transform.structured.fuse %gen tile_sizes [%h0, %h1]
        inner_tile_alignments = [Multiple, Equal]
        : (!transform.any_op, !transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion - linalg.unpack with scalable inner tiles fused as a consumer
// into an scf.for loop. The loop tile size on the tiled (inner-tile) dimension
// equals the unpack inner tile size (8*vscale), so fusion succeeds. This
// information is passed as an inner tile alignment hint `Equal`.

#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @fuse_scalable_unpack_consumer
// CHECK-SAME:      %[[ARG0:.+]]: tensor<32x?xf32>, %[[ARG1:.+]]: tensor<32x?xf32>, %[[ARG2:.+]]: tensor<32x?xf32>
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
//      CHECK:    %[[RES:.*]]:2 = scf.for {{.*}} step %[[C8_VSCALE]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ARG2]], %{{.*}} = %{{.*}})
//      CHECK:      %[[GENERIC:.*]] = linalg.generic
//      CHECK:      %[[UNPACK:.*]] = linalg.unpack %[[GENERIC]]
// CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]]]
//      CHECK:      scf.yield {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#1
func.func @fuse_scalable_unpack_consumer(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  // Loop tile size is equal to the inner tile size of the consumer `linalg.unpack` (8 * vscale).
  %0 = scf.for %iv = %c0 to %dim1 step %c8_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %sz = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%iv)[%c8_vscale, %dim1]
    %ext_a = tensor.extract_slice %arg0[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_b = tensor.extract_slice %arg1[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_out = tensor.extract_slice %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%ext_out : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // The `Equal` hint is passed to hint the equality between the loop tile size 8 * vscale
    // and the inner tile size 8 * vscale.
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) inner_tile_alignments = [Equal]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative) - linalg.unpack with scalable inner tiles and no
// alignment hint. The relationship between the loop tile size (4 * vscale) and the
// inner tile size (8 * vscale) cannot be decided statically. Without a user hint
// that asserts this, fusion fails.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @negative_fuse_scalable_unpack_consumer_no_hint(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %c4, %vscale : index
  %c8_vscale = arith.muli %c8, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  // Loop tile size (4 * vscale) is not equal to the inner tile size of the consumer
  // `linalg.unpack` (8 * vscale).
  %0 = scf.for %iv = %c0 to %dim1 step %c4_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %sz = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%iv)[%c4_vscale, %dim1]
    %ext_a = tensor.extract_slice %arg0[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_b = tensor.extract_slice %arg1[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_out = tensor.extract_slice %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%ext_out : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // No inner_tile_alignments hint is passed; the relation between the loop tile size
    // 4 * vscale and the inner tile size 8 * vscale cannot be decided statically.
    %a, %b = transform.test.fuse_consumer %unpack into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Consumer fusion (negative) - the loop tile size (16 * vscale) is a non-unit
// multiple of the unpack inner tile (8 * vscale), so the inner dim is genuinely
// tiled. Only `Equal` is meaningful for an unpack consumer's inner dim; a
// `Multiple` hint is not sufficient, so fusion must still fail.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @negative_fuse_scalable_unpack_consumer_multiple_hint(
    %arg0: tensor<32x?xf32>, %arg1: tensor<32x?xf32>,
    %arg2: tensor<32x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c16_vscale = arith.muli %c16, %vscale : index
  %dim1 = tensor.dim %arg2, %c1 : tensor<32x?xf32>

  // Loop tile size (16 * vscale) is a non-unit multiple of the inner tile size of the
  // consumer `linalg.unpack` (8 * vscale), so the inner dim is genuinely tiled.
  %0 = scf.for %iv = %c0 to %dim1 step %c16_vscale iter_args(%out = %arg2) -> (tensor<32x?xf32>) {
    %sz = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%iv)[%c16_vscale, %dim1]
    %ext_a = tensor.extract_slice %arg0[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_b = tensor.extract_slice %arg1[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %ext_out = tensor.extract_slice %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> to tensor<32x?xf32>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ext_a, %ext_b : tensor<32x?xf32>, tensor<32x?xf32>)
        outs(%ext_out : tensor<32x?xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out_elem: f32):
        %mul = arith.mulf %in0, %in1 : f32
        linalg.yield %mul : f32
    } -> tensor<32x?xf32>
    %inserted = tensor.insert_slice %computed into %out[0, %iv] [32, %sz] [1, 1]
        : tensor<32x?xf32> into tensor<32x?xf32>
    scf.yield %inserted : tensor<32x?xf32>
  }

  %output = tensor.empty(%dim1) : tensor<?xf32>
  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %0 outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [%c8_vscale]
      into %output : tensor<32x?xf32> -> tensor<?xf32>
  return %unpack : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // The `Multiple` hint is passed, but for an unpack consumer's inner dim only `Equal`
    // enables fusion; a non-unit `Multiple` (16 * vscale vs inner 8 * vscale) is insufficient.
    %a, %b = transform.test.fuse_consumer %unpack into (%loop) inner_tile_alignments = [Multiple]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Fusing a scalable `linalg.unpack` producer into a containing `scf.forall`
// with equal loop and inner tile sizes.

// CHECK-LABEL: func.func @fuse_unpack_into_containing_aligned
//       CHECK:   scf.forall
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack
//  CHECK-SAME:         : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
//   CHECK-NOT:     tensor.extract_slice %[[UNPACK]]
//       CHECK:     linalg.exp ins(%[[UNPACK]]
func.func @fuse_unpack_into_containing_aligned(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %out: tensor<?x?xf32>, %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  // Loop tile sizes (8 * vscale, 4 * vscale) are equal to the inner tile size of the consumer
  // `linalg.unpack`.
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %out) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %unpack[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %oslice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%oslice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // The `Equal` hints assert the containing loop's tile sizes equal the unpack inner
    // tile sizes (8 * vscale, 4 * vscale).
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Same fusion WITHOUT the hint: the producer falls back to the general (unaligned)
// tiling, which over-allocates the unpacked tile (`tensor<?x?x?x?xf32>` source) and
// recovers the needed slice with a trailing `tensor.extract_slice` on the result.

// CHECK-LABEL: func.func @fuse_unpack_into_containing_unaligned
//       CHECK:   scf.forall
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack
//  CHECK-SAME:         : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
//       CHECK:     %[[EXTRACTED:.+]] = tensor.extract_slice %[[UNPACK]]
//   CHECK-NOT:     linalg.exp ins(%[[UNPACK]]
//       CHECK:     linalg.exp ins(%[[EXTRACTED]]
func.func @fuse_unpack_into_containing_unaligned(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %out: tensor<?x?xf32>, %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %out) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %unpack[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %oslice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%oslice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The hint also reaches the block-argument fusion path: when the producer is the
// `scf.forall` init (used through the block argument rather than via a direct
// extract use), `inner_tile_alignments` still yields the aligned tiling.

// CHECK-LABEL: func.func @fuse_unpack_through_block_arg_aligned
//       CHECK:   scf.forall
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack
//  CHECK-SAME:         : tensor<1x1x?x?xf32> -> tensor<?x?xf32>
//   CHECK-NOT:     tensor.extract_slice %[[UNPACK]]
//       CHECK:     linalg.exp ins(%[[UNPACK]]
func.func @fuse_unpack_through_block_arg_aligned(
    %src: tensor<?x?x?x?xf32>, %unpack_empty: tensor<?x?xf32>,
    %ub0: index, %ub1: index) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index
  %c4_vscale = arith.muli %c4, %vscale : index
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, %c4_vscale] into %unpack_empty
      : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %res = scf.forall (%i, %j) = (0, 0) to (%ub0, %ub1) step (%c8_vscale, %c4_vscale)
      shared_outs(%o = %unpack) -> (tensor<?x?xf32>) {
    %slice = tensor.extract_slice %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %0 = linalg.exp ins(%slice : tensor<?x?xf32>) outs(%slice : tensor<?x?xf32>) -> tensor<?x?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j] [%c8_vscale, %c4_vscale] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  return %res : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // The `Equal` hints assert the containing loop's tile sizes equal the unpack inner
    // tile sizes (8 * vscale, 4 * vscale).
    %fused, %newc = transform.structured.fuse_into_containing_op %0 into %1
        inner_tile_alignments = [Equal, Equal]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// A 3-op dispatch tiled from the mmt4d root, fusing its consumers, with an
// inner-tile alignment hint driving the scalable unpack fusion:
//
//   linalg.mmt4d              (root, produces a packed [M, N, M0, N0] layout)
//   -> linalg.generic         (transposing bias add: [M, N, M0, N0] -> [N, M, N0, M0])
//   -> linalg.unpack          ([N, M, N0, M0] -> [N*N0, M*M0])
//
// The mmt4d is tiled along its scalable inner dim N0 (iteration dim 4) by
// 8*vscale; the generic and unpack are fused as consumers. 

#id4 = affine_map<(m, n, m0, n0) -> (m, n, m0, n0)>
#tr4 = affine_map<(m, n, m0, n0) -> (n, m, n0, m0)>

// CHECK: #[[$TR:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-LABEL: func.func @mmt4d_transpose_unpack
// CHECK-SAME:      %[[LHS:.+]]: tensor<2x2x4x2xf32>, %[[RHS:.+]]: tensor<2x2x?x2xf32>, %[[ACC:.+]]: tensor<2x2x4x?xf32>, %[[BIAS:.+]]: tensor<2x2x?x4xf32>, %[[TRINIT:.+]]: tensor<2x2x?x4xf32>, %[[OUT:.+]]: tensor<?x8xf32>
//      CHECK:    %[[VSCALE:.*]] = vector.vscale
//      CHECK:    %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
//      CHECK:    %[[RES:.*]]:3 = scf.for {{.*}} step %[[C8_VSCALE]]
// CHECK-SAME:        iter_args(%{{.*}} = %[[ACC]], %{{.*}} = %[[TRINIT]], %{{.*}} = %[[OUT]])
//      CHECK:      %[[MM:.*]] = linalg.mmt4d
//      CHECK:      %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:          indexing_maps = [#{{.+}}, #[[$TR]], #[[$TR]]]
// CHECK-SAME:          ins(%[[MM]],
//      CHECK:      %[[UNPACK:.*]] = linalg.unpack %[[GENERIC]]
// CHECK-SAME:          outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
// CHECK-SAME:          inner_tiles = [%[[C8_VSCALE]], 4]
//      CHECK:      scf.yield {{.*}}, {{.*}}, %{{.*}} :
//      CHECK:    return %[[RES]]#2
func.func @mmt4d_transpose_unpack(
    %lhs: tensor<2x2x4x2xf32>, %rhs: tensor<2x2x?x2xf32>,
    %acc: tensor<2x2x4x?xf32>, %bias: tensor<2x2x?x4xf32>,
    %trinit: tensor<2x2x?x4xf32>, %out: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index

  // 1. mmt4d root: lhs[M,K,M0,K0] x rhs[N,K,N0,K0] -> out[M,N,M0,N0].
  %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x2xf32>, tensor<2x2x?x2xf32>)
      outs(%acc : tensor<2x2x4x?xf32>) -> tensor<2x2x4x?xf32>

  // 2. transposing bias add: [M,N,M0,N0] -> [N,M,N0,M0].
  %tr = linalg.generic {
      indexing_maps = [#id4, #tr4, #tr4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm, %bias : tensor<2x2x4x?xf32>, tensor<2x2x?x4xf32>)
      outs(%trinit : tensor<2x2x?x4xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<2x2x?x4xf32>

  // 3. unpack: [N,M,N0,M0] -> [N*N0, M*M0] = [?, 8].
  %unpack = linalg.unpack %tr outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, 4] into %out
      : tensor<2x2x?x4xf32> -> tensor<?x8xf32>
  return %unpack : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    // Tile the mmt4d root along its scalable inner dim N0 (iteration dim 4).
    %tiled, %loop = transform.structured.tile_using_for %mmt4d tile_sizes [0, 0, 0, 0, [8], 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse the transposing bias add (ignores the hint).
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fg, %loop2 = transform.test.fuse_consumer %gen into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse the unpack. The transpose puts the tiled scalable N0 inner tile on
    // dest dim 0, so Equal sits at index 0.
    %unp = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fu, %loop3 = transform.test.fuse_consumer %unp into (%loop2) inner_tile_alignments = [Equal, Unknown]
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative (hint is load-bearing): same chain fused without any alignment hint.
// The loop tile (8*vscale) and the unpack inner tile (8*vscale) are both
// scalable, so the relationship is not statically decidable and fusion of the
// unpack fails.

#id4 = affine_map<(m, n, m0, n0) -> (m, n, m0, n0)>
#tr4 = affine_map<(m, n, m0, n0) -> (n, m, n0, m0)>

func.func @negative_mmt4d_transpose_unpack_no_hint(
    %lhs: tensor<2x2x4x2xf32>, %rhs: tensor<2x2x?x2xf32>,
    %acc: tensor<2x2x4x?xf32>, %bias: tensor<2x2x?x4xf32>,
    %trinit: tensor<2x2x?x4xf32>, %out: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %c8, %vscale : index

  %mm = linalg.mmt4d ins(%lhs, %rhs : tensor<2x2x4x2xf32>, tensor<2x2x?x2xf32>)
      outs(%acc : tensor<2x2x4x?xf32>) -> tensor<2x2x4x?xf32>

  %tr = linalg.generic {
      indexing_maps = [#id4, #tr4, #tr4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%mm, %bias : tensor<2x2x4x?xf32>, tensor<2x2x?x4xf32>)
      outs(%trinit : tensor<2x2x?x4xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<2x2x?x4xf32>

  // expected-error @below {{'linalg.unpack' op failed to fuse consumer of slice}}
  %unpack = linalg.unpack %tr outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
      inner_tiles = [%c8_vscale, 4] into %out
      : tensor<2x2x?x4xf32> -> tensor<?x8xf32>
  return %unpack : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %mmt4d tile_sizes [0, 0, 0, 0, [8], 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fg, %loop2 = transform.test.fuse_consumer %gen into (%loop)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %unp = transform.structured.match ops{["linalg.unpack"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fu, %loop3 = transform.test.fuse_consumer %unp into (%loop2)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
