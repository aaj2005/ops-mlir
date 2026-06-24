"""Experiment: lower ops.par_loop -> stencil.* using xDSL, ahead of writing
the real lib/passes/OPSToStencil.cpp pass.

Usage:
    python xdsl_impl/ops_to_xdsl.py xdsl_impl/laplace_sample.mlir
    python xdsl_impl/ops_to_xdsl.py xdsl_impl/laplace_sample.mlir --unsafe-dereference-pointers

Each ops.par_loop is rewritten into, per Dat argument:

    %p = arith.constant <dat.data> : i64
    %ptr = llvm.inttoptr %p : i64 to !llvm.ptr
    %field = stencil.external_load %ptr : !llvm.ptr -> !stencil.field<...>

instead of a synthetic stencil.alloc -- the dat's real host pointer
(captured byte-for-byte from ops_dat::data, see OPSOps.td's OPS_DatAttr)
flows into the IR itself as an inttoptr'd value, so `%field` is genuinely
backed by that address as far as the IR is concerned. This is pure value
construction (materializing an integer, then a pointer type around it) --
it does not dereference anything, so it is safe to do unconditionally,
even when lowering a `.mlir` file dumped by a since-exited process: the
*pointer value* round-trips faithfully, only *reading through* it would
be unsafe (and nothing here does that for the dat buffers).

%field then feeds stencil.load (for READ/RW dats) into a single
stencil.apply, whose body issues a stencil.access per stencil-point
offset and a placeholder call to the original kernel (declared as an
external func.func, since the captured ops.par_loop carries only a
kernel name/pointer, not the kernel body), then stencil.store of the
apply's results back into the WRITE/RW/INC dats' (same, externally
loaded) fields.

Gbl/Idx/Reduce arguments (argtype != DAT) are not modeled (no field/temp
backs them) and are simply omitted from the generated kernel call's
argument list.

Stencil access pattern: unlike the dat pointer, OPS_StencilAttr's point
offsets are a *compile-time* shape decision (which/how-many stencil.access
ops to emit, and at which literal relative offsets) baked into the IR as
immediates, not a runtime value flowing through it -- mirroring how a real
stencil compiler treats a statically-declared ops_decl_stencil pattern.
Recovering it does require *reading* the pattern, via the raw `int*`
OPS_StencilAttr carries (mirroring ops_stencil_core::stencil exactly), so
that part genuinely dereferences memory and is gated behind
--unsafe-dereference-pointers / ops_runtime.bind_stencil_offsets -- valid
ONLY when this script runs in the same process that captured the IR (see
ops_runtime.py's SAFETY note). Without it, every dat defaults to a single
(0, ..., 0) access point.
"""

import sys
from pathlib import Path

from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, stencil
from xdsl.dialects.builtin import Builtin, IntegerAttr, ModuleOp, f64, i64
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


def stencil_points(arg: ArgAttr, ndim: int, *, unsafe_dereference: bool) -> list[tuple[int, ...]]:
    if unsafe_dereference:
        from ops_runtime import bind_stencil_offsets

        return bind_stencil_offsets(arg.stencil, allow_unsafe=True)
    return [tuple(0 for _ in range(ndim))]


def range_bounds(rng: list[int], ndim: int) -> list[tuple[int, int]]:
    return [(rng[2 * i], rng[2 * i + 1]) for i in range(ndim)]


def external_field(builder: Builder, dat: DatAttr, field_type: stencil.FieldType) -> stencil.ExternalLoadOp:
    """Materialize dat.data (a raw host pointer, captured verbatim from
    ops_dat::data) as IR: arith.constant -> llvm.inttoptr -> stencil's
    external_load, so the field's provenance is a real, printable,
    re-parseable part of the IR instead of a Python-side side-table.
    """
    ptr_int = builder.insert(arith.ConstantOp(IntegerAttr(dat.data.data, i64)))
    ptr = builder.insert(llvm.IntToPtrOp(ptr_int))
    return builder.insert(stencil.ExternalLoadOp(ptr, field_type))


def convert_par_loop(
    op: ParLoopOp, builder: Builder, module: ModuleOp, *, unsafe_dereference: bool
) -> None:
    ndim = op.dims.value.data
    dat_args = [a for a in op.arg_list() if a.argtype.data == ArgType.DAT]

    apply_bounds = stencil.StencilBoundsAttr(range_bounds(list(op.range.get_values()), ndim))

    loads: list[tuple[ArgAttr, stencil.LoadOp]] = []
    writes: list[ArgAttr] = []
    fields: dict[int, stencil.ExternalLoadOp] = {}
    for arg in dat_args:
        dat = arg.dat
        bounds = field_bounds(dat)
        field_type = stencil.FieldType(bounds, f64)
        field = external_field(builder, dat, field_type)
        fields[id(arg)] = field

        access = arg.acc.data
        if access in (Access.READ, Access.RW):
            ix = stencil.IndexAttr.get(*(b[0] for b in bounds))
            ux = stencil.IndexAttr.get(*(b[1] for b in bounds))
            load = builder.insert(stencil.LoadOp.get(field, lb=ix, ub=ux))
            loads.append((arg, load))
        if access in (Access.WRITE, Access.RW, Access.INC):
            writes.append(arg)

    body_args = [load.res.type for _, load in loads]
    block = Block(arg_types=body_args)
    block_builder = Builder(InsertPoint.at_end(block))

    access_results = []
    for (arg, _load), block_arg in zip(loads, block.args):
        for point in stencil_points(arg, ndim, unsafe_dereference=unsafe_dereference):
            access = block_builder.insert(stencil.AccessOp(block_arg, point))
            access_results.append(access.res)

    num_results = len(writes) or 1
    name = op.kernel_name.data
    if not any(isinstance(o, func.FuncOp) and o.sym_name.data == name for o in module.body.block.ops):
        decl = func.FuncOp(
            name,
            ((f64,) * len(access_results), (f64,) * num_results),
            region=Region(),
            visibility="private",
        )
        module.body.block.add_op(decl)

    kernel = block_builder.insert(
        func.CallOp(name, access_results, [f64] * num_results)
    )
    block_builder.insert(stencil.ReturnOp.get(list(kernel.results)))

    apply = builder.insert(
        stencil.ApplyOp(
            args=[load for _, load in loads],
            body=Region(block),
            result_types=[stencil.TempType(field_bounds(w.dat), f64) for w in writes]
            or [stencil.TempType(apply_bounds.bounds(), f64)],
            bounds=apply_bounds,
        )
    )

    for w, res in zip(writes, apply.res):
        store_bounds = stencil.StencilBoundsAttr(field_bounds(w.dat))
        builder.insert(stencil.StoreOp(res, fields[id(w)], store_bounds))


def convert_module(module: ModuleOp, *, unsafe_dereference: bool = False) -> None:
    for op in list(module.body.block.ops):
        if isinstance(op, ParLoopOp):
            builder = Builder(InsertPoint.before(op))
            convert_par_loop(op, builder, module, unsafe_dereference=unsafe_dereference)
            op.detach()
            op.erase()


def convert_ir_text(text: str, unsafe_dereference: bool = False) -> str:
    """String-in/string-out entry point for embedding (see
    lib/runtime/JITEngine.cpp::runXdslLowering). Unlike main(), this never
    touches the filesystem or stdout/stderr -- the caller owns the IR
    string and the result string.

    `unsafe_dereference=True` only affects whether the real stencil access
    pattern is recovered by dereferencing OPS_StencilAttr's offsets
    pointer (see module docstring) -- it is only safe when called from the
    same process that captured the IR (see ops_runtime.py's SAFETY note);
    JITEngine calls this from inside compile(), before any captured
    ops_dat/ops_stencil buffer can go out of scope, so that precondition
    holds there. The dat buffer pointers themselves are always safe to
    materialize into the IR regardless of this flag (see external_field).
    """
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(OPS)
    ctx.load_dialect(stencil.Stencil)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(llvm.LLVM)

    module = Parser(ctx, text).parse_module()
    convert_module(module, unsafe_dereference=unsafe_dereference)
    module.verify()

    from io import StringIO

    out = StringIO()
    Printer(stream=out).print_op(module)
    return out.getvalue()


def main() -> None:
    args = sys.argv[1:]
    unsafe_dereference = "--unsafe-dereference-pointers" in args
    args = [a for a in args if a != "--unsafe-dereference-pointers"]

    if len(args) != 1:
        print(f"usage: {sys.argv[0]} <input.mlir> [--unsafe-dereference-pointers]", file=sys.stderr)
        sys.exit(1)

    text = Path(args[0]).read_text()
    print(convert_ir_text(text, unsafe_dereference=unsafe_dereference))


if __name__ == "__main__":
    main()
