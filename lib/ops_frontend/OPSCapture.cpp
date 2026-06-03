#include "ops_frontend/OPSCapture.h"

namespace ops_mlir {

CaptureRuntime &CaptureRuntime::instance() {
  static CaptureRuntime rt;
  return rt;
}

void CaptureRuntime::setFlushCallback(FlushCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  flushCallback_ = std::move(callback);
}

void CaptureRuntime::enqueueParLoop(std::uintptr_t kernelToken,
                                    const char *kernelName, ops_block block,
                                    int dims, const int *range,
                                    const ops_arg *args, std::size_t nargs) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push_back(
      buildLoopDesc(kernelToken, kernelName, block, dims, range, args, nargs));
}

void CaptureRuntime::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (flushCallback_ && !queue_.empty()) {
    flushCallback_(queue_);
  }
  queue_.clear();
}

LoopDesc CaptureRuntime::buildLoopDesc(std::uintptr_t kernelToken,
                                       const char *kernelName, ops_block block,
                                       int dims, const int *range,
                                       const ops_arg *args, std::size_t nargs) {
  LoopDesc loop;
  loop.kernel_name = kernelName;
  loop.kernel_token = kernelToken;
  loop.block = reinterpret_cast<std::uintptr_t>(block);
  loop.dims = dims;

  loop.range.assign(range, range + 2 * dims);

  loop.args.reserve(nargs);
  for (std::size_t i = 0; i < nargs; ++i) {
    loop.args.push_back(buildArgDesc(args[i]));
  }

  return loop;
}

ArgDesc CaptureRuntime::buildArgDesc(const ops_arg &arg) {
  ArgDesc desc;
  desc.dim = arg.dim;
  desc.elem_size = arg.elem_size;
  desc.access = arg.acc;
  desc.optional = arg.opt;
  desc.dat_handle = reinterpret_cast<std::uintptr_t>(arg.dat);
  desc.stencil_handle = reinterpret_cast<std::uintptr_t>(arg.stencil);
  desc.host_ptr = reinterpret_cast<std::uintptr_t>(arg.data);
  desc.device_ptr = reinterpret_cast<std::uintptr_t>(arg.data_d);

  switch (arg.argtype) {
  case OPS_ARG_DAT:
    desc.kind = ArgKind::Dat;
    if (arg.dat)
      desc.dat = describeDat(arg.dat);
    if (arg.stencil)
      desc.stencil = describeStencil(arg.stencil);
    break;
  case OPS_ARG_IDX:
    desc.kind = ArgKind::Idx;
    break;
  case OPS_ARG_GBL:
    // Distinguish globals from reductions by access type
    desc.kind = (arg.acc != OPS_READ) ? ArgKind::Reduce : ArgKind::Gbl;
    break;
  default:
    desc.kind = ArgKind::Unknown;
    break;
  }

  return desc;
}

DatDesc CaptureRuntime::describeDat(ops_dat dat) {
  DatDesc d;
  d.handle = reinterpret_cast<std::uintptr_t>(dat);
  d.block = reinterpret_cast<std::uintptr_t>(dat->block);
  d.name = dat->name ? dat->name : "";
  d.type = dat->type ? dat->type : "";
  d.dim = dat->dim;
  d.elem_size = dat->elem_size;
  d.type_size = dat->type_size;

  int ndim = dat->block ? dat->block->dims : 1;
  for (int i = 0; i < ndim; ++i) {
    d.size.push_back(dat->size[i]);
    d.base.push_back(dat->base[i]);
    d.d_m.push_back(dat->d_m[i]);
    d.d_p.push_back(dat->d_p[i]);
    d.stride.push_back(dat->stride[i]);
  }
  return d;
}

StencilDesc CaptureRuntime::describeStencil(ops_stencil stencil) {
  StencilDesc s;
  s.handle = reinterpret_cast<std::uintptr_t>(stencil);
  s.name = stencil->name ? stencil->name : "";
  s.dims = stencil->dims;
  s.points = stencil->points;
  s.type = stencil->type;

  int n = stencil->dims * stencil->points;
  s.offsets.assign(stencil->stencil, stencil->stencil + n);
  if (stencil->stride)
    s.stride.assign(stencil->stride, stencil->stride + stencil->dims);

  return s;
}

const char *accessToString(int access) {
  switch (access) {
  case OPS_READ:
    return "READ";
  case OPS_WRITE:
    return "WRITE";
  case OPS_RW:
    return "RW";
  case OPS_INC:
    return "INC";
  case OPS_MIN:
    return "MIN";
  case OPS_MAX:
    return "MAX";
  default:
    return "UNKNOWN";
  }
}

const char *argKindToString(ArgKind kind) {
  switch (kind) {
  case ArgKind::Dat:
    return "Dat";
  case ArgKind::Gbl:
    return "Gbl";
  case ArgKind::Idx:
    return "Idx";
  case ArgKind::Reduce:
    return "Reduce";
  default:
    return "Unknown";
  }
}

} // namespace ops_mlir
