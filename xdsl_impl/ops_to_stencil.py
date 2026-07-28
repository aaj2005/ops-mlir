"""Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design.
"""

import ctypes
from dataclasses import dataclass
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, stencil
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType, ModuleOp, i32, f64
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass

from ops_dialect import ArgType, Access, DatAttr, ParLoopOp, StencilAttr

def field_bounds(dat: DatAttr) -> list[tuple[int, int]]:
    """Normalized 0-based field bounds: lb=0, ub=full allocated size per dim."""
    return [(0, size) for size in dat.size_list]

def halo_offsets(dat_args) -> list[int]:
    """Per-dim d_m values (negative) from the first dat — shared across all dats on a block."""
    return list(dat_args[0].dat.d_m_list)

def stencil_offsets(stencil_attr: StencilAttr) -> list[tuple[int, ...]]:
    """Per-point access offsets for an arg's stencil.

    Extract the offsets from the StencilAttr's raw pointer to the underlying C array.
    """
    dims = stencil_attr.dims.data
    points = stencil_attr.points.data
    addr = stencil_attr.stencil.data
    if addr == 0 or points == 0:
        return [tuple(0 for _ in range(dims))]

    flat = (ctypes.c_int32 * (dims * points)).from_address(addr)
    return [tuple(flat[p * dims + d] for d in range(dims)) for p in range(points)]

def range_bounds(rng: list[int], ndim: int) -> list[tuple[int, int]]:
    return [(rng[2 * i], rng[2 * i + 1]) for i in range(ndim)]

def normalized_range_bounds(rng: list[int], d_m: list[int], ndim: int) -> list[tuple[int, int]]:
    """Shift OPS iteration range by -d_m so bounds are relative to normalized (0-based) field."""
    return [(lb - dm, ub - dm) for (lb, ub), dm in zip(range_bounds(rng, ndim), d_m)]

def declare_kernel(
    module: ModuleOp, name: str, num_f64_args: int, num_idx_args: int, ndim: int,
    num_results: int
) -> None:
    if any(
        isinstance(o, func.FuncOp) and o.sym_name.data == name
        for o in module.body.block.ops
    ):
        return

    # Idx arguments are passed as MemRefType(i32, [ndim]) to match the OPS kernel ABI 
    param_types = (f64,) * num_f64_args + (MemRefType(i32, [ndim]),) * num_idx_args
    decl = func.FuncOp(
        name,
        (param_types, (f64,) * max(num_results, 1)),
        region=Region(),
        visibility="private",
    )
    module.body.block.add_op(decl)

def convert_par_loop(op: ParLoopOp, index: int, module: ModuleOp) -> func.FuncOp:
    ndim = op.dims.value.data
    args = op.arg_list()
    dat_args = [a for a in args if a.argtype.data == ArgType.DAT]
    num_idx_args = sum(1 for a in args if a.argtype.data == ArgType.IDX)

    d_m = halo_offsets(dat_args)
    apply_bounds = stencil.StencilBoundsAttr(
        normalized_range_bounds(list(op.range.get_values()), d_m, ndim)
    )

    field_types = [stencil.FieldType(field_bounds(arg.dat), f64) for arg in dat_args]
    
    # Outer function: ops_par_loop_<kernel>_<index>(fields...) -> ()
    kernel_name = op.kernel_name.data
    fn_name = f"ops_par_loop_{kernel_name}_{index}"
    fn = func.FuncOp(fn_name, (tuple(field_types), ()), visibility="private")
    block = fn.body.block
    fn_builder = Builder(InsertPoint.at_end(block))

    # Partition dat args (by access mode) into stencil.apply's reads/writes
    reads: list[SSAValue] = []
    read_types = []
    read_stencils: list[StencilAttr] = []
    writes: list[SSAValue] = []
    for arg, field in zip(dat_args, block.args):
        access = arg.acc.data
        if access in (Access.READ, Access.RW):
            reads.append(field)
            read_types.append(field.type)
            read_stencils.append(arg.stencil)
        if access in (Access.WRITE, Access.RW, Access.INC):
            writes.append(field)

    # Apply block body: access reads, compute indices, call kernel
    apply_block = Block(arg_types=read_types)
    block_builder = Builder(InsertPoint.at_end(apply_block))

    # One stencil.access per real stencil point (from stencil_offsets), not
    # just a placeholder (0, ..., 0) -- so multi-point stencils (e.g. a 5pt
    # Laplacian) actually read all their neighbors.
    access_results = []
    for block_arg, stencil_attr in zip(apply_block.args, read_stencils):
        for point in stencil_offsets(stencil_attr):
            access = block_builder.insert(stencil.AccessOp(block_arg, point))
            access_results.append(access.res)

    # Compute per-dim index buffers for the kernel call, which are passed as MemRefType(i32, [ndim])
    idx_buffers = []
    # Offset d_m[dim] un-normalizes the loop index back to the OPS global index:
    # ops_global_idx = normalized_loop_idx + d_m  (d_m is negative, e.g. -1)
    for _ in range(num_idx_args):
        idx_buffer = block_builder.insert(
            memref.AllocaOp.get(i32, shape=[ndim])
        )
        for d in range(ndim):
            idx_op = block_builder.insert(
                stencil.IndexOp.build(
                    attributes={
                        "dim": IntegerAttr(d, IndexType()),
                        "offset": stencil.IndexAttr.from_indices(*[
                            d_m[i] if i == d else 0 for i in range(ndim)
                        ]),
                    },
                    result_types=[IndexType()],
                )
            )
            idx_i32 = block_builder.insert(
                arith.IndexCastOp(idx_op.idx, i32)
            )
            dim_const = block_builder.insert(
                arith.ConstantOp(IntegerAttr(d, IndexType()))
            )
            block_builder.insert(
                memref.StoreOp.get(idx_i32.result, idx_buffer.memref, [dim_const.result])
            )
        idx_buffers.append(idx_buffer.memref)

    call_args = access_results + idx_buffers
    num_results = len(writes) or 1
    declare_kernel(
        module, kernel_name, len(access_results), len(idx_buffers), ndim, num_results
    )

    kernel = block_builder.insert(
        func.CallOp(kernel_name, call_args, [f64] * num_results)
    )
    block_builder.insert(stencil.ReturnOp.get(list(kernel.results)))

    # Create ApplyOp in buffer semantic form
    fn_builder.insert(
        stencil.ApplyOp.build(
            operands=[reads, writes, []],  # reduction operands empty for now
            regions=[Region([apply_block])],
            result_types=[[]],  # buffer semantic does not return results
            properties={"bounds": apply_bounds},
        )
    )

    fn_builder.insert(func.ReturnOp())
    return fn

@dataclass(frozen=True)
class OPSToStencilPass(ModulePass):
    """Lower ops.par_loop -> func.func + stencil.* (see module docstring)."""
    name = "ops-to-stencil"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        loops = [o for o in op.body.block.ops if isinstance(o, ParLoopOp)]
        for index, loop_op in enumerate(loops):
            self.lower_par_loop(loop_op, index, op)

    def lower_par_loop(self, loop_op: ParLoopOp, index: int, module: ModuleOp) -> None:
        fn = convert_par_loop(loop_op, index, module)
        module.body.block.add_op(fn)
        loop_op.detach()
        loop_op.erase()
