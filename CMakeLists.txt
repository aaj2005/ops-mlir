cmake_minimum_required(VERSION 3.20)
project(ops-mlir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===============================================================================
# MLIR/LLVM Setup
#===============================================================================

# Paths are read from environment variables. See README.md for setup
# instructions (source env_setup or export these yourself before running cmake).
if(NOT DEFINED ENV{MLIR_BUILD_DIR})
  message(FATAL_ERROR "Environment variable MLIR_BUILD_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{LLVM_BUILD_DIR})
  message(FATAL_ERROR "Environment variable LLVM_BUILD_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{LLVM_SOURCE_DIR})
  message(FATAL_ERROR "Environment variable LLVM_SOURCE_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{MLIR_SOURCE_DIR})
  message(FATAL_ERROR "Environment variable MLIR_SOURCE_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{OPS_ROOT})
  message(FATAL_ERROR "Environment variable OPS_ROOT is not set. See README.md.")
endif()

set(MLIR_BUILD_DIR $ENV{MLIR_BUILD_DIR})
set(LLVM_BUILD_DIR $ENV{LLVM_BUILD_DIR})
set(LLVM_SOURCE_DIR $ENV{LLVM_SOURCE_DIR})
set(MLIR_SOURCE_DIR $ENV{MLIR_SOURCE_DIR})
set(OPS_ROOT $ENV{OPS_ROOT})
set(OPS_INCLUDE_DIR ${OPS_ROOT}/include)

# Find MLIR and LLVM packages
find_package(MLIR REQUIRED CONFIG
  HINTS "${MLIR_BUILD_DIR}/lib/cmake/mlir"
)
find_package(LLVM REQUIRED CONFIG
  HINTS "${LLVM_BUILD_DIR}/lib/cmake/llvm"
)

# Include MLIR cmake utilities
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_BUILD_DIR}/cmake/modules")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

#===============================================================================
# Project Structure
#===============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OPS_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${MLIR_BUILD_DIR}/include)
include_directories(${LLVM_BUILD_DIR}/include)
include_directories(${MLIR_SOURCE_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/include)

add_subdirectory(lib)
add_subdirectory(apps)

#===============================================================================
# Compilation Database for Language Server
#===============================================================================
# Automatically generated at ${CMAKE_BINARY_DIR}/compile_commands.json
