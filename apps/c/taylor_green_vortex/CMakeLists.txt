#===============================================================================
# Taylor Green Vortex Application

if(NOT DEFINED ENV{HDF5_INSTALL_PATH})
  message(FATAL_ERROR "Environment variable HDF5_INSTALL_PATH is not set. See README.md.")
endif()
set(HDF5_INSTALL_PATH $ENV{HDF5_INSTALL_PATH})

# Collect OPS sequential backend sources from the OPS source tree
file(GLOB OPS_CORE_SRCS   "${OPS_ROOT}/src/core/*.cpp")
file(GLOB OPS_SEQ_SRCS    "${OPS_ROOT}/src/sequential/*.cpp")
file(GLOB OPS_EXTERN_SRCS "${OPS_ROOT}/src/externlib/*.cpp")

# Exclude device-specific files that require CUDA/MPI (keep HDF5 sources - this app needs them)
list(FILTER OPS_CORE_SRCS   EXCLUDE REGEX "device")

add_library(ops_seq_hdf5_local STATIC
  ${OPS_CORE_SRCS}
  ${OPS_SEQ_SRCS}
  ${OPS_EXTERN_SRCS}
)
target_include_directories(ops_seq_hdf5_local PUBLIC
  ${OPS_ROOT}/include
  ${HDF5_INSTALL_PATH}/include
)
target_compile_options(ops_seq_hdf5_local PRIVATE -fno-PIE)

#===============================================================================

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(opensbli opensbli.cpp)

target_include_directories(opensbli
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OPS_ROOT}/include
  ${HDF5_INSTALL_PATH}/include
)

target_link_directories(opensbli PRIVATE ${HDF5_INSTALL_PATH}/lib)

target_link_libraries(opensbli
  PRIVATE
  OPSRuntime
  MLIROPSDialect
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
  ${mlir_extension_libs}
  MLIRRegisterAllPasses
  MLIRIR
  MLIRPass
  MLIRSupport
  ops_seq_hdf5_local
  hdf5_hl
  hdf5
  m
)

target_link_options(opensbli PRIVATE -no-pie)
