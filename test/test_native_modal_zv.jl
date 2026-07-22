using TestItems

@testitem "Native-Rust Phase I.5a (modal, multi-mode ZeisbergerMode)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary,
                      Modes, Antiresonant
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # Mirrors test_native_modal.jl's Phase 5 gate exactly (RealGrid,
        # constant-radius HE1m modes, scalar (npol=1) Kerr-only, two modes so
        # the to_space!/back-projection matmuls are genuinely multi-mode) but
        # with the modes wrapped as `ZeisbergerMode` instead of bare
        # `Capillary.MarcatiliMode` — this is the actual Phase I.5a scope:
        # the guard at RK45.jl (~line 1657) previously rejected any non-bare-
        # Marcatili mode outright, so this config always fell back to
        # `PreconStepper` (Julia) before this change. `ZeisbergerMode`
        # delegates `field`/`N`/`dimlimits` to its wrapped `MarcatiliMode`
        # verbatim (Antiresonant.jl's `@eval` loop), so the transverse mode
        # profile the native RHS synthesizes is bit-identical to plain
        # Marcatili — only `neff`/β differ, and that's baked into `linop`
        # once by Julia, not read by the native modal RHS at all.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.1
        wallthickness = 700e-9

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)

        modes = (Antiresonant.ZeisbergerMode(a, gas, pres; m=1, wallthickness, loss=false),
                 Antiresonant.ZeisbergerMode(a, gas, pres; m=2, wallthickness, loss=false))
        @assert Modes.orthonormal(modes) "test modes must be orthonormal"
        # Non-vacuousness precondition: the two modes must actually be
        # ZeisbergerMode, not something the guard would treat as bare
        # MarcatiliMode by accident.
        @assert all(m -> m isa Antiresonant.ZeisbergerMode, modes)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
        input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, responses, input, modes, :y)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransModal "Expected TransModal"
        @assert !transform.full "Expected full=false (radial modal integral)"

        t0 = 0.0
        dt = 0.001

        @testset "Native path genuinely taken (not a silent fallback)" begin
            # Drive the actual `Luna.run` integration point (`solve_precon`),
            # not a direct `RustNativeStepper(...)` construction — before this
            # change, the guard threw `NativeIneligible` here, which
            # `solve_precon` catches silently and falls back to
            # `PreconStepper`. A single tiny step is enough to prove the
            # branch taken; the equivalence testsets below do the real work.
            RK45.solve_precon(transform, linop, copy(Eω), t0, dt, t0 + dt;
                               rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            @test RK45._LAST_STEPPER_TYPE[] <: RK45.RustNativeStepper
        end

        @testset "Single-step equivalence (modal, Zeisberger, ~1e-10)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal (Zeisberger) single-step rel: ", rel_step)
            # Same tolerance tier as plain-Marcatili Phase 5 (test_native_modal.jl):
            # node placement is bit-identical (same libcubature binary), mode-field
            # synthesis reuses the same Bessel evaluation — the wrapping adds no
            # extra numerical operation to the RHS itself.
            @test rel_step < 1e-10
        end

        @testset "Full-solve equivalence (modal, Zeisberger, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            # Non-vacuousness of the two-mode physics itself (mirrors the
            # Phase 5 gate's HE11->HE12 sanity check): confirm genuine energy
            # transfer into HE12 before trusting the native-vs-Julia
            # agreement below.
            he12_frac = sum(abs2, s_jl.yn[:, 2]) / sum(abs2, s_jl.yn)
            println("Fraction of energy transferred to HE12 (Julia, Zeisberger, sanity check): ",
                    he12_frac)
            @test he12_frac > 1e-6

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal (Zeisberger) full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end

@testitem "Native-Rust Phase I.5a (modal, multi-mode VincettiMode)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary,
                      Modes, Antiresonant
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # Same shape as the ZeisbergerMode testitem above, but with
        # `VincettiMode` — the other wrapper the guard now accepts.
        # `VincettiMode` does NOT delegate `Aeff` (Antiresonant.jl:523 defines
        # its own, using `neff_real`/`Rco_eff`/`Δneff`), unlike `field`/`N`/
        # `dimlimits` which it does delegate (Antiresonant.jl:381-383) — but
        # `Aeff` never enters the modal RHS (only `TransModeAvg` uses it), so
        # that divergence from plain Marcatili is inert here. `neff` (used to
        # build `linop`) is Vincetti's own model, correctly picked up by
        # `LinearOps.make_const_linop`'s per-mode `Modes.neff` dispatch,
        # independent of this guard.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        energy = 5e-6; L = 0.1
        wallthickness = 700e-9
        # Physically sane tube-lattice geometry (positive gap between
        # resonators), derived the same way test_antiresonant.jl's "Vincetti
        # Model" testset does — a bare `tube_radius=a` (as in the ZeisbergerMode
        # testitem above) gives a negative gap and a `@warn`; that doesn't
        # break equivalence but a reviewer shouldn't have to wonder about it.
        r_ext = 10e-6; Ntubes = 6; δ = 5e-6
        a = Antiresonant.getRco(r_ext, Ntubes, δ)

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)

        modes = (Antiresonant.VincettiMode(a, gas, pres; m=1, wallthickness,
                                            tube_radius=r_ext, Ntubes, loss=false),
                 Antiresonant.VincettiMode(a, gas, pres; m=2, wallthickness,
                                            tube_radius=r_ext, Ntubes, loss=false))
        @assert Modes.orthonormal(modes) "test modes must be orthonormal"
        @assert all(m -> m isa Antiresonant.VincettiMode, modes)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
        input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, responses, input, modes, :y)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransModal "Expected TransModal"
        @assert !transform.full "Expected full=false (radial modal integral)"

        t0 = 0.0
        dt = 0.001

        @testset "Native path genuinely taken (not a silent fallback)" begin
            RK45.solve_precon(transform, linop, copy(Eω), t0, dt, t0 + dt;
                               rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            @test RK45._LAST_STEPPER_TYPE[] <: RK45.RustNativeStepper
        end

        @testset "Single-step equivalence (modal, Vincetti, ~1e-10)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal (Vincetti) single-step rel: ", rel_step)
            @test rel_step < 1e-10
        end

        @testset "Full-solve equivalence (modal, Vincetti, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            he12_frac = sum(abs2, s_jl.yn[:, 2]) / sum(abs2, s_jl.yn)
            println("Fraction of energy transferred to HE12 (Julia, Vincetti, sanity check): ",
                    he12_frac)
            @test he12_frac > 1e-6

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal (Vincetti) full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
