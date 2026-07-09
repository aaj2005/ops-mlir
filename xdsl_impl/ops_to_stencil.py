"""Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design (see xdsl_impl/laplace_stencil_1.mlir
for the target IR shape).
...
[full module docstring, unchanged]
"""

from dataclasses import dataclass
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import func, stencil
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp, f64
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass

from ops_dialect import ArgType, Access, DatAttr, ParLoopOp


def field_bounds(dat: DatAttr) -> list[tuple[int, int]]:
    """Per-dim (lb, ub) bounds of the full allocated (halo-included) field."""
    return [
        (d_m, size + d_p)
        for d_m, size, d_p in zip(dat.d_m_list, dat.size_list, dat.d_p_list)
    ]

def stencil_points(ndim: int) -> list[tuple[int, ...]]:
    """Defaults to a single (0, ..., 0) access point -- see module docstring."""
    return [tuple(0 for _ in range(ndim))]

def range_bounds(rng: list[int], ndim: int) -> list[tuple[int, int]]:
    return [(rng[2 * i], rng[2 * i + 1]) for i in range(ndim)]


def declare_kernel(
    module: ModuleOp, name: str, num_f64_args: int, num_idx_args: int, num_results: int
) -> None:
    if any(
        isinstance(o, func.FuncOp) and o.sym_name.data == name
        for o in module.body.block.ops
    ):
        return
    param_types = (f64,) * num_f64_args + (IndexType(),) * num_idx_args
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

    apply_bounds = stencil.StencilBoundsAttr(
        range_bounds(list(op.range.get_values()), ndim)
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
    writes: list[SSAValue] = []
    for arg, field in zip(dat_args, block.args):
        access = arg.acc.data
        if access in (Access.READ, Access.RW):
            reads.append(field)
            read_types.append(field.type)
        if access in (Access.WRITE, Access.RW, Access.INC):
            writes.append(field)

    # Apply block body: access reads, compute indices, call kernel
    apply_block = Block(arg_types=read_types)
    block_builder = Builder(InsertPoint.at_end(apply_block))

    access_results = []
    for block_arg in apply_block.args:
        for point in stencil_points(ndim):
            access = block_builder.insert(stencil.AccessOp(block_arg, point))
            access_results.append(access.res)

    # ops_arg_idx(): one stencil.index per dimension, per idx argument, no
    # static shift -- mirrors lib/passes/OPSToStencil.cpp.
    idx_results = []
    zero_offset = stencil.IndexAttr.from_indices(*([0] * ndim))
    for _ in range(num_idx_args):
        for d in range(ndim):
            idx_op = block_builder.insert(
                stencil.IndexOp.build(
                    attributes={
                        "dim": IntegerAttr(d, IndexType()),
                        "offset": zero_offset,
                    },
                    result_types=[IndexType()],
                )
            )
            idx_results.append(idx_op.idx)

    call_args = access_results + idx_results
    num_results = len(writes) or 1
    declare_kernel(
        module, kernel_name, len(access_results), len(idx_results), num_results
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
        