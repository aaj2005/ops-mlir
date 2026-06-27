cmake_minimum_required(VERSION 3.20)
project(ops-mlir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===============================================================================
# MLIR/LLVM Setup
#===============================================================================

# TODO: Set using config file or environment variables
# Use the cs325_demo MLIR build
set(MLIR_BUILD_DIR /home/pt/warwick/cs325_demo/llvm-project/build)
set(LLVM_BUILD_DIR /home/pt/warwick/cs325_demo/llvm-project/build)
set(LLVM_SOURCE_DIR /home/pt/warwick/cs325_demo/llvm-project/llvm)
set(MLIR_SOURCE_DIR /home/pt/warwick/cs325_demo/llvm-project/mlir)

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
# OPS Library Setup
#===============================================================================
set(OPS_INCLUDE_DIR /home/pt/warwick/OPS/ops/c/include)

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
