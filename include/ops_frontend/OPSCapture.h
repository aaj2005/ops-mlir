#ifndef OPS_CAPTURE_H
#define OPS_CAPTURE_H

#include "ops_lib_core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>


namespace ops_mlir {

enum class ArgKind {
  Dat,
  Gbl,
  Idx,
  Reduce,
  Unknown
};

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

class CaptureRuntime {
public:
  using FlushCallback = std::function<void(const std::vector<LoopDesc> &)>;

  static CaptureRuntime &instance();

  void setFlushCallback(FlushCallback callback);

  void enqueueParLoop(std::uintptr_t kernelToken,
                      const char *kernelName,
                      ops_block block,
                      int dims,
                      const int *range,
                      const ops_arg *args,
                      std::size_t nargs);

  void flush();

  const std::vector<LoopDesc> &queue() const { return queue_; }

private:
  CaptureRuntime() = default;

  LoopDesc buildLoopDesc(std::uintptr_t kernelToken,
                         const char *kernelName,
                         ops_block block,
                         int dims,
                         const int *range,
                         const ops_arg *args,
                         std::size_t nargs);

  ArgDesc buildArgDesc(const ops_arg &arg);

  DatDesc describeDat(ops_dat dat);
  StencilDesc describeStencil(ops_stencil stencil);

private:
  std::mutex mutex_;
  std::vector<LoopDesc> queue_;
  FlushCallback flushCallback_;
};

const char *accessToString(int access);
const char *argKindToString(ArgKind kind);

} // namespace ops_mlir

#endif // OPS_CAPTURE_H
