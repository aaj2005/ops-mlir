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

from xdsl.context import Context
from xdsl.dialects import func, stencil
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
    ConvertStencilToLLMLIRPass,
)
from xdsl.passes import PassPipeline


sys.path.insert(0, str(Path(__file__).parent))
from ops_dialect import OPS
from ops_to_stencil import OPSToStencilPass

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
    pipeline = PassPipeline([
        OPSToStencilPass(),
        ConvertStencilToLLMLIRPass(),
    ])

    pipeline.apply(ctx, module)
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
