set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

set(LIBS
  MLIRIR
  MLIRPDLLAST
  MLIRPDLLCodeGen
  MLIRPDLLODS
  MLIRPDLLParser
  )

add_tablegen(mlir-pdll MLIR_PDLL
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  mlir-pdll.cpp

  DEPENDS
  ${LIBS}

  DISABLE_PCH_REUSE  # Executable built with -fPIE incompatible with library PCH
  )

target_link_libraries(mlir-pdll PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-pdll)
