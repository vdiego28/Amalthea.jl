using TestItems

# Smoke-run a representative subset of examples/low_level_interface/ so an
# API change that breaks the documented low-level entry point fails CI
# instead of going unnoticed (docs/dev/BACKLOG.md "Distribution &
# example-code maintenance" item 2 — as of 2026-07-22, nothing under
# test/, .github/, or docs/ exercised these files).
#
# What this does and does NOT check:
#   - It `include`-executes each chosen example's *physics* statements
#     (grid/mode/field setup through `Amalthea.run`) in a fresh module and
#     asserts only that nothing throws — the cheapest honest signal that the
#     low-level API surface these examples exercise still works.
#   - It does NOT execute the plotting tail. Every example in this
#     directory does `using Amalthea, PyPlot` and ends with `Plotting.*`
#     calls (several call `Plotting.pygui(true)`, which switches to an
#     interactive matplotlib backend and would need a display / GUI
#     toolkit + PyPlot installed — a heavy, flaky dependency this project
#     deliberately avoids installing eagerly, see the PyCall/Conda note at
#     the top of deps/build.jl). The harness below strips `PyPlot` out of
#     each example's `using` line and stops at the first statement that
#     mentions `Plotting`/`plt.`/`pygui`/`PyPlot`, so PyPlot need not be
#     installed at all for this test to run.
#   - It does NOT run the full-length demo propagation. Every chosen
#     example hardcodes a multi-cm/multi-m fibre length meant for a
#     realistic demo run (e.g. `basic_modal.jl`'s 15 cm modal case takes
#     ~4 minutes even with the native backend); the harness rewrites the
#     `flength`/`L` assignment to `SMOKE_LENGTH` (a few mm) so each example
#     still exercises the real grid/mode/field/RHS/RK45 setup and a handful
#     of accepted steps, in seconds rather than minutes.
#
# Why only 8 of the 44 files under examples/low_level_interface/, not all:
# a full run (even at SMOKE_LENGTH) of every file is not a "cheap" smoke
# test — some examples carry grid-resolution costs (large radial/hankel
# matrices, dense post-processing) that don't shrink with `flength`, and
# several are independently broken today regardless of this harness (see
# below) — running those would make this CI group fail on unrelated,
# pre-existing bugs rather than on genuine regressions. The 8 chosen files
# were individually verified (2026-07-22) to run to completion under this
# harness and span the main low-level code paths: mode-averaged field/env,
# modal field/env, GNLSE, gas Raman, gas mixtures, and step-index fibre.
#
# Known-broken examples found while building this test (NOT fixed here —
# out of this change's scope, see docs/dev/native-port/portlog-inbox/hygiene.md):
#   - full_modal/basic_modal_full.jl,
#     full_modal/basic_modal_full_bothpolarisations.jl,
#     polarisation/modal_vector_plasma.jl,
#     polarisation/modal_nonvector_plasma.jl,
#     polarisation/modal_vector_plasma_CP.jl,
#     polarisation/modal_vector_plasma_45deg.jl
#     all reference `linop` in a `Stats.default(...)` call that appears
#     BEFORE the `linop = LinearOps.make_const_linop(...)` assignment in
#     the same file — `UndefVarError: linop not defined`.
#   - full_modal/basic_modal_full.jl,
#     full_modal/basic_modal_full_bothpolarisations.jl,
#     polarisation/elliptical_env.jl
#     call `NonlinearRHS.norm_modal(grid.ω)` — `norm_modal` takes the whole
#     grid object (it reads `grid.referenceλ` internally, see
#     src/NonlinearRHS.jl:476-477), not `grid.ω` — `FieldError`.
@testitem "Low-level example smoke run" tags=[:examples] begin
    import Test: @test, @testset
    using Amalthea
    import Logging: with_logger, NullLogger

    # A few mm is enough for RK45 to take several real accepted steps
    # through the nonlinear RHS without the multi-minute cost of the
    # examples' original demo-scale fibre lengths.
    const SMOKE_LENGTH = 5e-3

    examples_dir = joinpath(@__DIR__, "..", "examples", "low_level_interface")

    # (relative path, human label) — see the file-level comment above for
    # why these 8 and not the full 44.
    examples = [
        ("basic_modeAvg.jl", "mode-averaged, field-resolved, plasma"),
        ("basic_modeAvg_env.jl", "mode-averaged, envelope"),
        ("basic_modal.jl", "modal, field-resolved, PPT ionisation"),
        ("basic_modal_env.jl", "modal, envelope"),
        (joinpath("gnlse", "simplescg_modeAvg.jl"), "GNLSE (SimpleFibre + Raman)"),
        (joinpath("Raman", "Raman_modeAvg.jl"), "gas Raman response"),
        (joinpath("mixtures", "mixture_modeAvg.jl"), "gas mixture"),
        (joinpath("stepindex", "stepscg_modeAvg.jl"), "step-index fibre"),
    ]

    # `using A, PyPlot` -> `using A` (drops PyPlot only); returns `nothing`
    # if PyPlot was the only import (so the statement is skipped entirely).
    function _strip_pyplot_using(e::Expr)
        kept = filter(e.args) do a
            !(a isa Expr && a.head === :(.) && length(a.args) == 1 && a.args[1] === :PyPlot)
        end
        isempty(kept) && return nothing
        return Expr(:using, kept...)
    end

    # Evaluate an example's physics-setup statements (skipping/stopping
    # before anything PyPlot-related) in a fresh module. Throws on the
    # first statement that errors — callers wrap this in `@test`. Returns
    # whether an `Amalthea.run(...)` statement was actually reached and
    # evaluated — since this harness truncates each example before its
    # plotting tail, "didn't throw" is only a meaningful smoke signal if
    # the executed prefix actually included the propagation call.
    function run_example_physics_only(path)
        src = read(path, String)
        parsed = Meta.parseall(src; filename=path)
        mod = Module(:ExampleSmoke)
        Base.eval(mod, :(using Amalthea))
        reached_run = false
        for e in parsed.args
            e isa LineNumberNode && continue
            if e isa Expr && e.head === :using
                e2 = _strip_pyplot_using(e)
                e2 === nothing && continue
                e = e2
            end
            if e isa Expr && e.head === :(=) && e.args[1] in (:flength, :L)
                e = Expr(:(=), e.args[1], SMOKE_LENGTH)
            end
            s = string(e)
            (occursin("Plotting", s) || occursin("plt.", s) ||
             occursin("pygui", s) || occursin("PyPlot", s)) && break
            with_logger(NullLogger()) do
                Base.eval(mod, e)
            end
            occursin("Amalthea.run", s) && (reached_run = true)
        end
        return reached_run
    end

    @testset "$label ($relpath)" for (relpath, label) in examples
        path = joinpath(examples_dir, relpath)
        @test isfile(path)
        reached_run = try
            run_example_physics_only(path)
        catch e
            @error "Example smoke run threw" relpath exception=(e, catch_backtrace())
            false
        end
        @test reached_run
    end
end
