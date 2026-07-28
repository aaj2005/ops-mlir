//===- OPSWrapper.h - OPS JIT capture wrapper ----------------------*- C++
//-*-===//
//
// Intercepts OPS API calls to capture program structure for JIT compilation.
// Forwards to the real OPS library for correctness during development.
//
// Users only need: #include "ops/OPSWrapper.h"
//
//===----------------------------------------------------------------------===//

#ifndef OPS_WRAPPER_H
#define OPS_WRAPPER_H

#include "runtime/JITEngine.h"
#include "ops_lib_core.h"

#include <array>
#include <type_traits>

//===----------------------------------------------------------------------===//
// ops_par_loop interception
//===----------------------------------------------------------------------===//
// This template overrides the ops_par_loop to capture loop metadata.
// The captured loops are queued for JIT compilation.
//===----------------------------------------------------------------------===//

template <typename KernelFn, typename... Args>
void ops_par_loop(KernelFn kernel, const char *name, ops_block block, int dims,
                  int *range, Args... opsArgs) {
  static_assert((std::is_same_v<std::decay_t<Args>, ops_arg> && ...),
                "All ops_par_loop variadic arguments must be ops_arg values");

  std::array<ops_arg, sizeof...(Args)> packedArgs{opsArgs...};

  auto token =
      static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(kernel));

  ops_mlir::JITEngine::instance().enqueueParLoop(
      token, name, block, dims, range, packedArgs.data(), packedArgs.size());
}

void compile_and_execute() {
  ops_mlir::JITEngine::instance().compile_and_execute();
}

#endif // OPS_WRAPPER_H
