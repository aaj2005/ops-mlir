#===============================================================================
# Laplace 2D Application
#===============================================================================

# OPS sequential backend library (optional, for Phase 1 execution fallback)
set(OPS_SEQ_LIB /home/pt/warwick/OPS/ops/c/lib/libops_seq.a)
set(OPS_INCLUDE_DIR /home/pt/warwick/OPS/ops/c/include)

add_executable(
  laplace_2d
  laplace_2d.cpp
)

target_include_directories(
  laplace_2d
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OPS_INCLUDE_DIR}
)

target_link_libraries(
  laplace_2d
  PRIVATE
  MLIROPSFrontend
  MLIROPSDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)

# Link OPS library if available (for Phase 1 fallback execution)
if(EXISTS "${OPS_SEQ_LIB}")
  target_link_libraries(laplace_2d PRIVATE ${OPS_SEQ_LIB})
else()
  message(STATUS "OPS sequential library not found - JIT execution only")
endif()

target_link_libraries(laplace_2d PRIVATE m)  # math library
target_compile_options(laplace_2d PRIVATE -fPIC)

