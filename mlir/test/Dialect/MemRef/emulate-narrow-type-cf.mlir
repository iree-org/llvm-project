// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8 enable-cf-conversion=true" --cse --verify-diagnostics --split-input-file %s | FileCheck %s

// Sub-byte memref type carried through cf.br block args. The
// BranchOpInterface type-conversion pattern must rewrite both the cf.br
// operand type and the successor block-arg type to the i8 container, so the
// downstream uses in the successor block see an i8 source.

// CHECK-LABEL: func.func @cf_br_block_arg_narrow_type
// CHECK-SAME:    %[[ARG:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>
// CHECK:         cf.br ^[[BB1:.+]](%[[ARG]] : memref<{{[0-9]+}}xi8>)
// CHECK:       ^[[BB1]](%[[BARG:[A-Za-z0-9_]+]]: memref<{{[0-9]+}}xi8>):
// CHECK:         return %[[BARG]]
// CHECK-NOT:     memref<{{[0-9]+}}xi4>
func.func @cf_br_block_arg_narrow_type(%arg: memref<8xi4>) -> memref<8xi4> {
  cf.br ^bb1(%arg : memref<8xi4>)
^bb1(%a: memref<8xi4>):
  return %a : memref<8xi4>
}
