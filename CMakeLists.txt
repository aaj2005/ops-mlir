cmake_minimum_required(VERSION 3.20)
project(ops-mlir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#===============================================================================
# MLIR/LLVM Setup
#===============================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Include MLIR cmake utilities
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

#===============================================================================
# Project Structure
#===============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(apps)
