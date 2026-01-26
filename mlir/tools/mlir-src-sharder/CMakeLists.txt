set(LLVM_LINK_COMPONENTS Support)
set(LIBS MLIRSupport)

add_tablegen(mlir-src-sharder MLIR_SRC_SHARDER
  mlir-src-sharder.cpp

  DEPENDS
  ${LIBS}

  DISABLE_PCH_REUSE  # Executable built with -fPIE incompatible with library PCH
  )

set_target_properties(mlir-src-sharder PROPERTIES FOLDER "MLIR/Tablegenning")
target_link_libraries(mlir-src-sharder PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-src-sharder)
