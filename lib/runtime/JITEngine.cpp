#include "runtime/JITEngine.h"
#include "runtime/BackendPipeline.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Python.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/TargetSelect.h"

#include <algorithm>
#include <memory>

#ifdef OPS_ENABLE_CUDA
#include <cuda.h>
#endif

namespace ops_mlir {

JITEngine &JITEngine::instance() {
  static JITEngine rt;
  return rt;
}

JITEngine::JITEngine() {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }

  // Make xdsl_impl/ops_to_xdsl.py importable. OPS_XDSL_DIR is injected by
  // lib/runtime/CMakeLists.txt as the absolute path to xdsl_impl/.
  std::string setup = "import sys\nsys.path.insert(0, '" OPS_XDSL_DIR "')\n";
  PyRun_SimpleString(setup.c_str());

  // Register needed dialects - will need to add more later
  mlir::registerAllPasses();
  // TODO: add all required dialects
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  ctx.getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();

  // Needed by ExecutionEngine::create to translate the lowered module to
  // LLVM IR; without these, translation fails with "missing
  // LLVMTranslationDialectInterface registration".
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);

  // Needed so the JIT's target machine can be created for the host triple;
  // without this, ExecutionEngine::create fails with "Unable to find target
  // for this triple (no targets are registered)".
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Resolve backend (without working CLI flags for now)
  resolveBackend(0, nullptr);
}

JITEngine::~JITEngine() {
  if (Py_IsInitialized()) {
    Py_FinalizeEx();
  }
}

void JITEngine::setFlushCallback(FlushCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  flushCallback_ = std::move(callback);
}

void JITEngine::enqueueParLoop(std::uintptr_t kernelToken,
                                    const char *kernelName, ops_block block,
                                    int dims, const int *range,
                                    const ops_arg *args, std::size_t nargs) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.push_back(
      buildLoopDesc(kernelToken, kernelName, block, dims, range, args, nargs));
}

void JITEngine::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (flushCallback_ && !queue_.empty()) {
    flushCallback_(queue_);
  }
  queue_.clear();
}

std::string JITEngine::detectNVGpuSm() {
  if (const char *env = std::getenv("OPS_GPU_SM"))
    return env;

#ifdef OPS_ENABLE_CUDA
  if (cuInit(0) != CUDA_SUCCESS) {
    throw std::runtime_error(
        "cuInit failed -- no NVIDIA driver found. Set OPS_GPU_SM manually.");
  }
  CUdevice device;
  cuDeviceGet(&device, 0);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  return std::to_string(major) + std::to_string(minor);
#else
  throw std::runtime_error(
      "This build was compiled without CUDA support (OPS_ENABLE_CUDA=OFF). "
      "Reconfigure with -DOPS_ENABLE_CUDA=ON to use the CUDA backend, or "
      "set OPS_GPU_SM manually if targeting a remote/precompiled binary.");
#endif
}

LoopDesc JITEngine::buildLoopDesc(std::uintptr_t kernelToken,
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

ArgDesc JITEngine::buildArgDesc(const ops_arg &arg) {
  ArgDesc desc;
  desc.dim = arg.dim;
  desc.elem_size = arg.elem_size;
  desc.data = reinterpret_cast<std::uintptr_t>(arg.data);
  desc.data_d = reinterpret_cast<std::uintptr_t>(arg.data_d);
  desc.acc = arg.acc;
  desc.argtype = arg.argtype;
  desc.opt = arg.opt;

  if (arg.argtype == OPS_ARG_DAT) {
    if (arg.dat)
      desc.dat = describeDat(arg.dat);
    if (arg.stencil)
      desc.stencil = describeStencil(arg.stencil);
  }

  return desc;
}

DatDesc JITEngine::describeDat(ops_dat dat) {
  DatDesc d;
  d.handle = reinterpret_cast<std::uintptr_t>(dat); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  d.index = dat->index;
  d.block = reinterpret_cast<std::uintptr_t>(dat->block); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  
  d.dim = dat->dim;
  d.type_size = dat->type_size;
  d.elem_size = dat->elem_size;

  int ndim = dat->block ? dat->block->dims : 1;
  for (int i = 0; i < ndim; ++i) {
    d.size.push_back(dat->size[i]);
    d.base.push_back(dat->base[i]);
    d.d_m.push_back(dat->d_m[i]);
    d.d_p.push_back(dat->d_p[i]);
    d.stride.push_back(dat->stride[i]);
  }

  d.name = dat->name ? dat->name : "";
  d.type = dat->type ? dat->type : "";

  d.data = reinterpret_cast<std::uintptr_t>(dat->data);
  d.data_d = reinterpret_cast<std::uintptr_t>(dat->data_d);

  return d;
}

StencilDesc JITEngine::describeStencil(ops_stencil stencil) {
  StencilDesc s;
  s.index = stencil->index;
  s.dims = stencil->dims;
  s.points = stencil->points;
  s.name = stencil->name ? stencil->name : "";

  s.stencil = reinterpret_cast<std::uintptr_t>(stencil->stencil);
  s.stride = reinterpret_cast<std::uintptr_t>(stencil->stride);
  s.mgrid_stride = reinterpret_cast<std::uintptr_t>(stencil->mgrid_stride);
  s.type = stencil->type;

  return s;
}

static std::string fetchPyError() {
  if (!PyErr_Occurred())
    return "unknown Python error";
  PyObject *type, *value, *tb;
  PyErr_Fetch(&type, &value, &tb);
  PyErr_NormalizeException(&type, &value, &tb);
  std::string msg = "unknown Python error";
  if (value) {
    PyObject *str = PyObject_Str(value);
    if (str) {
      if (const char *s = PyUnicode_AsUTF8(str)) msg = s;
      Py_DECREF(str);
    }
  }
  Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(tb);
  return msg;
}

XdslResult JITEngine::runXdslLowering(const std::string &ir) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  XdslResult result;

  PyObject *mod = PyImport_ImportModule("ops_to_xdsl");
  if (!mod) {
    result.error = fetchPyError();
    PyGILState_Release(gstate);
    return result;
  }

  PyObject *func = PyObject_GetAttrString(mod, "convert_ir_text");
  Py_DECREF(mod);
  if (!func || !PyCallable_Check(func)) {
    result.error = fetchPyError();
    Py_XDECREF(func);
    PyGILState_Release(gstate);
    return result;
  }

  PyObject *args = Py_BuildValue("(s)", ir.c_str());
  if (!args) {
    result.error = fetchPyError();
    Py_DECREF(func);
    PyGILState_Release(gstate);
    return result;
  }
  PyObject *pyResult = PyObject_CallObject(func, args);
  Py_DECREF(args);
  Py_DECREF(func);

  if (!pyResult) {
    result.error = fetchPyError();
    PyGILState_Release(gstate);
    return result;
  }

  if (const char *text = PyUnicode_AsUTF8(pyResult)) {
    result.ir = text;
    result.success = true;
  } else {
    result.error = fetchPyError();
  }
  Py_DECREF(pyResult);

  PyGILState_Release(gstate);
  return result;
}

void JITEngine::runBackendLowering(mlir::ModuleOp module, Backend backend) {
  std::unique_ptr<BackendPipeline> pipeline;

  switch (backend) {
  case Backend::Sequential:
    pipeline = std::make_unique<CPUSequentialPipeline>();
    break;
  case Backend::OpenMP:
    pipeline = std::make_unique<OpenMPPipeline>();
    break;
  case Backend::CUDA:
#ifdef OPS_ENABLE_CUDA
    pipeline = std::make_unique<CudaPipeline>(detectNVGpuSm());
#else
    throw std::runtime_error(
        "CUDA backend requested but this build was compiled without "
        "CUDA support (OPS_ENABLE_CUDA=OFF).");
#endif
    break;
  }

  if (!pipeline) {
    llvm::errs() << "no lowering pipeline for requested backend\n";
    return;
  }

  if (mlir::failed(pipeline->run(module, ctx))) {
    llvm::errs() << "backend lowering failed for module\n"; 
    return;
  }

  llvm::outs() << "=== BACKEND-LOWERED MLIR IR ===\n\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";
}

void JITEngine::compile() {
  module = builder.buildModule(queue_);

  std::string ir = builder.moduleToString(module);
  llvm::outs() << "=== OPS.PAR_LOOP MLIR IR ===\n\n" << ir << "\n";

  XdslResult lowered = runXdslLowering(ir);
  if (!lowered.success) {
    llvm::errs() << "xDSL lowering failed: " <<  lowered.error << "\n";
    return;
  }
  
  llvm::outs() << "=== LOWERED STENCIL IR (xDSL, in-process) ===\n\n"
            << lowered.ir << "\n";

  mlir::OwningOpRef<mlir::ModuleOp> loweredModule = 
    mlir::parseSourceString<mlir::ModuleOp>(lowered.ir, &ctx);
  
  if (!loweredModule) {
    llvm::errs() << "Failed to parse xDSL output as MLIR\n";
    return;
  }

  runBackendLowering(*loweredModule,  backend_);
  module = *loweredModule;

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

  auto engineOrErr = mlir::ExecutionEngine::create(module, engineOptions);
  if (!engineOrErr) {
    llvm::errs() << "Failed to create ExecutionEngine: "
                  << llvm::toString(engineOrErr.takeError()) << "\n";
    return;
  }

  engine = std::move(*engineOrErr);
}

void JITEngine::execute() {
  if (!engine) {
    llvm::errs() << "JITEngine::execute: no compiled module (call compile() "
                    "first)\n";
    return;
  }

  // Each ops.par_loop was lowered to a standalone function named
  // "ops_par_loop_<kernel_name>_<queue_index>" taking one bare pointer per
  // ops_dat argument, in the order the loops were enqueued. We invoke them
  // one by one with the live data pointers.
  for (std::size_t i = 0; i < queue_.size(); ++i) {
    const LoopDesc &loop = queue_[i];
    std::string funcName =
        "ops_par_loop_" + loop.kernel_name + "_" + std::to_string(i);

    std::vector<void *> datPtrs;
    for (const ArgDesc &arg : loop.args) {
      if (arg.argtype == OPS_ARG_DAT) {
        datPtrs.push_back(reinterpret_cast<void *>(arg.data));
      }
    }

    llvm::SmallVector<void *> packedArgs;
    packedArgs.reserve(datPtrs.size());
    for (void *&ptr : datPtrs) {
      packedArgs.push_back(&ptr);
    }

    if (auto err = engine->invokePacked(funcName, packedArgs)) {
      llvm::errs() << "Failed to invoke '" << funcName
                   << "': " << llvm::toString(std::move(err)) << "\n";
      return;
    }
  }
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

std::optional<Backend> parseBackendName(const std::string &name) {
  if (name == "seq" || name == "sequential") return Backend::Sequential;
  if (name == "openmp" || name == "omp") return Backend::OpenMP;
  if (name == "cuda" || name == "nvgpu") return Backend::CUDA; 
  return std::nullopt;
}


Backend JITEngine ::resolveBackend(int argc, char **argv) {
  // Explicit CLI flag takes precendence
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind(kBackendFlagPrefix, 0) == 0) {
      std::string value = arg.substr(std::string(kBackendFlagPrefix).size());
      if (auto b = parseBackendName(value)) return *b;
      throw std::runtime_error("Unknown --backend value: '" + value + "' (expected seq|openmp|cuda)");
    }
  }

  // Fall back to env variable
  if (const char *env = std::getenv(kBackendEnvVar)) {
    if (auto b = parseBackendName(env)) return *b;
    throw std::runtime_error("Unknown " + std::string(kBackendEnvVar) + " value: '" + env + "' (expected seq|openmp|cuda)");
  }

  // Default to sequential if neither cli flag or env var set
  return kDefaultBackend;
}

} // namespace ops_mlir
