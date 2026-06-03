#===============================================================================
# OPS Dialect TableGen Setup
#===============================================================================

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Dialect/OPS/OPSOps.td)

mlir_tablegen(OPSDialect.h.inc -gen-dialect-decls)
mlir_tablegen(OPSDialect.cpp.inc -gen-dialect-defs)

mlir_tablegen(OPSOps.h.inc -gen-op-decls)
mlir_tablegen(OPSOps.cpp.inc -gen-op-defs)

mlir_tablegen(OPSAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(OPSAttr.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(OPSDialectTdGen)

#===============================================================================
# OPS Dialect Library
#===============================================================================

add_mlir_dialect_library(
  MLIROPSDialect
  OPSOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Dialect/OPS

  DEPENDS
  OPSDialectTdGen

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS
  public
  MLIRIR
  MLIRDialectUtils
  MLIRPass
  MLIRSupport
)
