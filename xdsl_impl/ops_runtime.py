"""Case B: true in-process JIT bridge from a captured ops.dat's raw pointer
to a real numpy buffer, for use as live input/output to a lowered
stencil.apply -- as opposed to the synthetic stencil.alloc placeholders used
by the structural-only experiment in ops_to_xdsl.py.

SAFETY: the `data`/`data_d` integers inside #ops.dat<...> are raw virtual
addresses captured by lib/runtime/JITEngine.cpp::describeDat in *that*
process. They are only valid:
  1. within the same process that produced them (same address space), and
  2. for as long as the originating ops_dat buffer is still alive (not
     freed/realloc'd).
Dereferencing them from a different process (e.g. `python3 foo.py
captured.mlir` after the C++ program exited) is undefined behaviour. Every
entry point here therefore requires `allow_unsafe=True` to make that
opt-in explicit, and is meant to be called from the same process/extension
module that did the OPS capture (e.g. via a pybind11 binding embedding this
interpreter, not via a standalone CLI on a dumped .mlir file).
"""

import ctypes

import numpy as np

from ops_dialect import ArgAttr, DatAttr, StencilAttr

# ops_dat_core::type strings -> numpy dtype (see ops_lib_core.h ops_decl_dat).
OPS_TYPE_TO_NUMPY = {
    "double": np.float64,
    "float": np.float32,
    "int": np.int32,
    "long": np.int64,
    "char": np.int8,
}


class UnsafePointerError(RuntimeError):
    pass


def _require_unsafe(allow_unsafe: bool) -> None:
    if not allow_unsafe:
        raise UnsafePointerError(
            "Dereferencing a captured ops.dat pointer is only valid in the "
            "same process that captured it. Pass allow_unsafe=True only "
            "when you know that's the case."
        )


def dat_buffer_shape(dat: DatAttr) -> tuple[int, ...]:
    """Full allocated shape (including halo), matching ops_dat_core::size."""
    return tuple(dat.size_list)


def bind_dat_buffer(dat: DatAttr, *, device: bool = False, allow_unsafe: bool = False) -> np.ndarray:
    """Return a numpy array that is a *live view* over the real ops_dat
    buffer behind `dat.data` (host) or `dat.data_d` (device), via ctypes.

    No copy is made: writes through the returned array are writes into the
    same memory OPS itself is using. Shape is taken from `dat.size`
    (allocated size including halo, row-major as stored by OPS).
    """
    _require_unsafe(allow_unsafe)

    ptr = (dat.data_d if device else dat.data).data
    if ptr == 0:
        raise UnsafePointerError("dat has a null data pointer; nothing to bind")

    dtype_name = dat.dat_type.data
    dtype = OPS_TYPE_TO_NUMPY.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported ops dat type {dtype_name!r}")

    shape = dat_buffer_shape(dat)
    count = 1
    for s in shape:
        count *= s
    count *= dat.dim.data  # multi-component dats interleave `dim` values per grid point

    c_array_type = np.ctypeslib.as_ctypes_type(dtype) * count
    c_array = ctypes.cast(ptr, ctypes.POINTER(c_array_type)).contents
    flat = np.ctypeslib.as_array(c_array)

    if dat.dim.data > 1:
        return flat.reshape((*shape, dat.dim.data))
    return flat.reshape(shape)


def bind_arg_buffer(arg: ArgAttr, *, device: bool = False, allow_unsafe: bool = False) -> np.ndarray:
    """Convenience wrapper for an ops.arg's nested dat."""
    return bind_dat_buffer(arg.dat, device=device, allow_unsafe=allow_unsafe)


def bind_stencil_offsets(stencil: StencilAttr, *, allow_unsafe: bool = False) -> list[tuple[int, ...]]:
    """Dereference `stencil.stencil` (the raw `int*` from ops_stencil_core)
    to recover the actual per-point relative offsets.

    Unlike OPS_DatAttr's size/d_m/d_p (embedded as real DenseI64ArrayAttr
    values), OPS_StencilAttr only carries the *pointer* to the stencil's
    point-offset array -- mirroring `ops_stencil_core::stencil` exactly. So,
    unlike the dat shape, the actual stencil access pattern is only
    recoverable by dereferencing this pointer in-process; the structural-only
    converter in ops_to_xdsl.py has no other source of truth for it and
    falls back to a single (0, ..., 0) point.
    """
    _require_unsafe(allow_unsafe)

    dims = stencil.dims.data
    points = stencil.points.data
    ptr = stencil.stencil.data
    if ptr == 0 or points == 0:
        return [tuple(0 for _ in range(dims))]

    count = dims * points
    c_array_type = ctypes.c_int * count
    ints = list(ctypes.cast(ptr, ctypes.POINTER(c_array_type)).contents)
    return [tuple(ints[i:i + dims]) for i in range(0, count, dims)]
