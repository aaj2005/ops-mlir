# ops-mlir
MLIR Compiler Implementation for OPS DSL

## Dependencies

- CMake >= 3.20
- A C++17 compiler
- uv
- [LLVM/MLIR](https://github.com/llvm/llvm-project) built from source, with the `mlir` project enabled
- [OPS](https://github.com/OP-DSL/OPS) (the OPS DSL C library)
- OpenMP

## Python Dependencies

- xDSL (for generating OPS DSL code from Python)

## Installation

### 1. Build LLVM/MLIR

Clone and build `llvm-project` with the MLIR project enabled, e.g.:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="clang;mlir" \
   -DLLVM_ENABLE_RUNTIMES="openmp" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ 

ninja # optional: -jN where N is the number of threads you want to use, default is nproc
```

### 2. Clone OPS

Clone the [OPS](https://github.com/OP-DSL/OPS) repository — `ops-mlir`
builds the OPS sequential backend (`ops_seq`) from its source tree and
links against it.

### 3. Set environment variables

`ops-mlir`'s CMake configuration reads the LLVM/MLIR build/source
locations and the OPS checkout location from environment variables — it
does not hardcode any paths. Set the following before configuring the
project:

```bash
export MLIR_BUILD_DIR=/path/to/llvm-project/build
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export LLVM_SOURCE_DIR=/path/to/llvm-project/llvm
export MLIR_SOURCE_DIR=/path/to/llvm-project/mlir
export OPS_ROOT=/path/to/OPS/ops/c
export HDF5_INSTALL_PATH=/path/to/HDF5_INSTALL

export PATH=${MLIR_BUILD_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${MLIR_BUILD_DIR}:$LD_LIBRARY_PATH
```

A template is provided in [`env_setup_template`](env_setup_template) — copy/edit it for your
own paths and `source` it before building:

```bash
source env_setup
```

### 4. Build ops-mlir

```bash
cmake -S . -B build \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_C_COMPILER=$LLVM_BUILD_DIR/bin/clang \
   -DCMAKE_CXX_COMPILER=$LLVM_BUILD_DIR/bin/clang++ \
   -DPython3_EXECUTABLE=$OPS_MLIR_ROOT/.venv/bin/python3

cmake --build build -j$(nproc)
```

The project must be compiled with the clang++ bundled in the LLVM build
(`$MLIR_BUILD_DIR/bin/clang++`, added to `PATH` by `env_setup`). GCC 12 is
incompatible with the generated MLIR headers in LLVM 23. The OpenMP
headers and library are located automatically from `MLIR_BUILD_DIR` — no
extra env vars or `-D` flags are needed.

CMake configuration will fail with a clear error if any of the five
environment variables above are not set.

## 5. Use xDSL Fork
Needed for OPS reductions.

```bash
# Remove standard xDSL (if needed)
# inside the root of the repo
git clone https://github.com:Archii0/xdsl.git --branch stencil-reduce

uv sync

```

TLDR - only partial reduction support is upstreamed into xDSL, waiting on stencil dialect stakeholders to validate my full changes still...

## Usage

Run the 2D Laplace example:

```bash
# this just ensures that we use the correct version of python and xdsl which are installed in the venv
uv run ./build/apps/c/laplace_2d/laplace_2d
```
