"""Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design (see xdsl_impl/laplace_stencil_1.mlir
for the target IR shape).

Usage:
    python xdsl_impl/ops_to_xdsl.py xdsl_impl/laplace_sample.mlir

Each ops.par_loop becomes its own standalone func.func, with one
!stencil.field<...> *parameter* per Dat argument -- not a pointer baked
into the IR via arith.constant/llvm.inttoptr/stencil.external_load. The
real ops_dat buffer is a call-time concern (supplied by whoever invokes
the compiled function, e.g. JITEngine, once a real execution pipeline
exists), not something to encode as an IR literal: that approach doesn't
generalize across repeated calls (the same compiled loop runs once per
OPS timestep against the same buffer) or across hardware targets (a
device pointer is not a host pointer). See the discussion that motivated
this in lib/passes/OPSToStencil.cpp's header comment.

Inside the function body: stencil.load the read/read-write fields, a
single stencil.apply (one stencil.access per dat, defaulting to a single
(0, ..., 0) point -- see below -- plus one stencil.index per dimension
per Idx argument), a placeholder func.call to the kernel (the captured
ops.par_loop only carries the kernel's name/pointer, not its source),
stencil.store the results back into the write/read-write/inc fields, then
func.return.

Stencil access pattern: OPS_StencilAttr only carries a *pointer* to the
point-offset array (mirroring ops_stencil_core::stencil exactly), not the
offsets themselves, so this structural converter cannot recover the real
per-point pattern from the textual IR alone and defaults every dat to a
single (0, ..., 0) access point.

Gbl/Reduce arguments are still not modeled (no field/temp backs them) and
are simply omitted from the generated kernel call's argument list.
"""

import sys
from pathlib import Path

from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import func, stencil
from xdsl.dialects.builtin import Builtin, IndexType, IntegerAttr, ModuleOp, f64
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer

sys.path.insert(0, str(Path(__file__).parent))
from ops_dialect import OPS, ArgAttr, ArgType, Access, DatAttr, ParLoopOp  # noqa: E402


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


def declare_kernel(module: ModuleOp, name: str, num_f64_args: int, num_idx_args: int,
                    num_results: int) -> None:
    if any(isinstance(o, func.FuncOp) and o.sym_name.data == name for o in module.body.block.ops):
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

    apply_bounds = stencil.StencilBoundsAttr(range_bounds(list(op.range.get_values()), ndim))

    field_types = [stencil.FieldType(field_bounds(arg.dat), f64) for arg in dat_args]
    kernel_name = op.kernel_name.data
    fn_name = f"ops_par_loop_{kernel_name}_{index}"
    fn = func.FuncOp(fn_name, (tuple(field_types), ()), visibility="private")
    block = fn.body.block
    builder = Builder(InsertPoint.at_end(block))

    loads: list[tuple[ArgAttr, stencil.LoadOp]] = []
    writes: list[tuple[ArgAttr, object]] = []  # (arg, field block-arg)
    for arg, field in zip(dat_args, block.args):
        access = arg.acc.data
        if access in (Access.READ, Access.RW):
            bounds = field_bounds(arg.dat)
            ix = stencil.IndexAttr.get(*(b[0] for b in bounds))
            ux = stencil.IndexAttr.get(*(b[1] for b in bounds))
            load = builder.insert(stencil.LoadOp.get(field, lb=ix, ub=ux))
            loads.append((arg, load))
        if access in (Access.WRITE, Access.RW, Access.INC):
            writes.append((arg, field))

    body_args = [load.res.type for _, load in loads]
    apply_block = Block(arg_types=body_args)
    block_builder = Builder(InsertPoint.at_end(apply_block))

    access_results = []
    for (arg, _load), block_arg in zip(loads, apply_block.args):
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
                    attributes={"dim": IntegerAttr(d, IndexType()), "offset": zero_offset},
                    result_types=[IndexType()],
                )
            )
            idx_results.append(idx_op.idx)

    call_args = access_results + idx_results
    num_results = len(writes) or 1
    declare_kernel(module, kernel_name, len(access_results), len(idx_results), num_results)

    kernel = block_builder.insert(func.CallOp(kernel_name, call_args, [f64] * num_results))
    block_builder.insert(stencil.ReturnOp.get(list(kernel.results)))

    apply = builder.insert(
        stencil.ApplyOp(
            args=[load for _, load in loads],
            body=Region(apply_block),
            result_types=[stencil.TempType(field_bounds(w.dat), f64) for w, _f in writes]
            or [stencil.TempType(apply_bounds.bounds(), f64)],
            bounds=apply_bounds,
        )
    )

    for (w, field), res in zip(writes, apply.res):
        store_bounds = stencil.StencilBoundsAttr(field_bounds(w.dat))
        builder.insert(stencil.StoreOp(res, field, store_bounds))

    builder.insert(func.ReturnOp())
    return fn


def convert_module(module: ModuleOp) -> None:
    loops = [op for op in module.body.block.ops if isinstance(op, ParLoopOp)]
    for index, op in enumerate(loops):
        fn = convert_par_loop(op, index, module)
        module.body.block.add_op(fn)
        op.detach()
        op.erase()


def convert_ir_text(text: str) -> str:
    """String-in/string-out entry point for embedding (see
    lib/runtime/JITEngine.cpp::runXdslLowering). Unlike main(), this never
    touches the filesystem or stdout/stderr -- the caller owns the IR
    string and the result string.
    """
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(OPS)
    ctx.load_dialect(stencil.Stencil)
    ctx.load_dialect(func.Func)

    module = Parser(ctx, text).parse_module()
    convert_module(module)
    module.verify()

    from io import StringIO

    out = StringIO()
    Printer(stream=out).print_op(module)
    return out.getvalue()


def main() -> None:
    args = sys.argv[1:]

    if len(args) != 1:
        print(f"usage: {sys.argv[0]} <input.mlir>", file=sys.stderr)
        sys.exit(1)

    text = Path(args[0]).read_text()
    print(convert_ir_text(text))


if __name__ == "__main__":
    main()
