#===============================================================================
# OPS Frontend Library
#===============================================================================

add_library(
  MLIROPSFrontend
  OPSCapture.cpp
  OPSBuilder.cpp
)

target_include_directories(
  MLIROPSFrontend
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/OPS
)

target_link_libraries(
  MLIROPSFrontend
  PUBLIC
  MLIRIR
  MLIRDialectUtils
  MLIRSupport
  MLIRPass
  MLIROPSDialect
)
