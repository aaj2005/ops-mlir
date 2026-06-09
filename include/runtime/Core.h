
#ifndef OPS_MLIR_RUNTIME_CORE_H
#define OPS_MLIR_RUNTIME_CORE_H

#include "ops_lib_core.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ops_mlir {

enum class ArgKind { Dat, Gbl, Idx, Reduce, Unknown };

struct DatDesc {
  std::uintptr_t handle = 0;
  std::uintptr_t block = 0;

  std::string name;
  std::string type;

  int dim = 0;
  int elem_size = 0;
  int type_size = 0;

  std::vector<int64_t> size;
  std::vector<int64_t> base;
  std::vector<int64_t> d_m;
  std::vector<int64_t> d_p;
  std::vector<int64_t> stride;
};

struct StencilDesc {
  std::uintptr_t handle = 0;

  std::string name;

  int dims = 0;
  int points = 0;
  int type = 0;

  std::vector<int64_t> offsets;
  std::vector<int64_t> stride;
};

struct ArgDesc {
  ArgKind kind = ArgKind::Unknown;

  int dim = 0;
  int elem_size = 0;
  int access = OPS_READ;
  int optional = 1;

  std::uintptr_t dat_handle = 0;
  std::uintptr_t stencil_handle = 0;
  std::uintptr_t host_ptr = 0;
  std::uintptr_t device_ptr = 0;

  DatDesc dat;
  StencilDesc stencil;
};

struct LoopDesc {
  std::string kernel_name;
  std::uintptr_t kernel_token = 0;

  std::uintptr_t block = 0;
  int dims = 0;

  std::vector<int64_t> range;
  std::vector<ArgDesc> args;
};

}

#endif // OPS_MLIR_RUNTIME_CORE_H
