using TestItems

@testitem "Native-Rust Phase D.2 (plasma, radial)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Ionisation
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid + Hankel QDHT, Kerr + plasma (PPT) — BACKLOG.md Phase D.2 /
        # MATH.md §3.2's deferred plasma-radial follow-up. Same geometry as
        # test_native_radial.jl, plus a PlasmaCumtrapz response wired through
        # the Rust-backed IonRatePPTAccel LUT (test_native_phase2.jl's Phase 2b
        # pattern, applied here per r-column instead of mode-averaged).
        # w0 chosen so the beam waist actually overlaps the QDHT's first
        # radial sample point (~93.5 μm for R=4e-3, N=32) — a much smaller
        # w0 (as in test_native_radial.jl's Kerr-only geometry) puts the
        # entire beam inside the first grid cell, so the sampled field is a
        # deep, near-invisible exponential tail regardless of pulse energy,
        # and the extremely intensity-sensitive PPT ionisation rate needs
        # unreasonably large energies to show any non-vacuous effect at all.
        gas = :Ar; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 6e-5; L = 0.02; R = 4e-3; N = 32

        grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        ionpot = PhysData.ionisation_potential(gas)
        # The native plasma path needs a Rust-backed ionisation-rate handle,
        # which is only wired up if LUNA_USE_RUST_IONISATION=1 is set BEFORE
        # the LUT is built (see test_native_phase2.jl's Phase 2b comment) —
        # not just around the later RustNativeStepper construction below.
        ionrate = withenv("LUNA_USE_RUST_IONISATION" => "1") do
            Ionisation.IonRatePPTCached(gas, λ0)
        end
        plasma_resp = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), plasma_resp)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Luna.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (radial + plasma, ~1e-12)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = withenv("LUNA_USE_RUST_NATIVE" => "1",
                           "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            end

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial+plasma single-step rel: ", rel_step)
            @test rel_step < 1e-12
        end

        @testset "Full-solve equivalence (radial + plasma, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = withenv("LUNA_USE_RUST_NATIVE" => "1",
                           "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            end

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial+plasma full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: plasma actually changes the result vs plasma-off" begin
            # Uses a much stronger input field than the equivalence tests above
            # (same rationale as test_native_radial_env.jl's Phase D.1
            # non-vacuousness testset): the PPT ionisation rate is an
            # extremely sharp (near-exponential) function of |E|, so the
            # single-step Rust-vs-Julia tolerance (driven by the same
            # sensitivity amplifying the QDHT's ~1e-13 summation-order floor)
            # and this testset's "does plasma matter at all" check pull in
            # opposite directions at the SAME field strength — confirmed
            # empirically (energy=1e-4 J at the equivalence-test geometry
            # pushes non-vacuousness to only ~6.8e-10 while already breaking
            # the 1e-12 single-step tolerance). Using an independent,
            # Julia-only-compared field sidesteps this: energy=3e-4 J (5x the
            # equivalence energy) clears the 1e-6 threshold comfortably
            # without touching the Rust-vs-Julia tolerance at all — energies
            # well above this (tried: 6e-3 J) instead saturate the PPT LUT
            # and produce a NaN in the adaptive step controller.
            energy_strong = 3e-4
            inputs_strong = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy_strong,
                                                    w0=w0, propz=-0.1)
            Eω_strong, transform_strong, _ = with_logger(NullLogger()) do
                Luna.setup(grid, q, densityfun, normfun, responses, inputs_strong)
            end
            no_plasma_transform = NonlinearRHS.TransRadial(grid, q, transform_strong.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),), densityfun, normfun)
            s_jl_plasma = PreconStepper(transform_strong, linop, copy(Eω_strong), t0, dt,
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_noplasma = PreconStepper(no_plasma_transform, linop, copy(Eω_strong), t0, dt,
                                           rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_plasma, L)
            solve(s_jl_noplasma, L)
            rel_plasma_effect = norm(s_jl_plasma.yn - s_jl_noplasma.yn) / norm(s_jl_noplasma.yn)
            println("Radial plasma-on vs plasma-off rel (Julia, non-vacuousness): ", rel_plasma_effect)
            @test rel_plasma_effect > 1e-6

            # BACKLOG.md Phase I item 3 postmortem (2026-07-08): the
            # equivalence tests above run at a much weaker energy (6e-5 J)
            # than this non-vacuousness check (3e-4 J) — exactly the blind
            # spot that let a real bug (native's plasma polarization
            # missing a density factor entirely, silently contributing ~0)
            # go undetected: at the weak equivalence energy, native
            # "matching Julia" was vacuously true (both near-zero plasma
            # contribution), and this non-vacuousness check only ever
            # compared Julia against itself, never against native, at the
            # energy where the bug would actually show. Close that gap
            # directly: native vs Julia AT the plasma-matters energy, and
            # native vs Julia's plasma-OFF result (must NOT match).
            s_ru_strong = withenv("LUNA_USE_RUST_NATIVE" => "1",
                           "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform_strong, linop, copy(Eω_strong), t0, dt,
                                   rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            end
            solve(s_ru_strong, L)
            rel_native_vs_jl = norm(s_ru_strong.yn - s_jl_plasma.yn) / norm(s_jl_plasma.yn)
            rel_native_vs_noplasma = norm(s_ru_strong.yn - s_jl_noplasma.yn) / norm(s_jl_noplasma.yn)
            println("Radial native vs Julia (both plasma-on, strong energy): ", rel_native_vs_jl)
            println("Radial native vs Julia no-plasma (must NOT match — regression guard): ",
                     rel_native_vs_noplasma)
            @test rel_native_vs_jl < 1e-6
            @test rel_native_vs_noplasma > 1e-6
        end

        @testset "n_threads=1 vs n_threads=4 bit-identical (BACKLOG.md S2)" begin
            # Both the FFT-per-column loops AND apply_plasma_radial's
            # per-column loop are parallelized under S2 — this is the case
            # that actually exercises the plasma seam (test_native_radial.jl's
            # equivalent testset only covers the Kerr-only FFT seam). Same
            # disjoint-writes argument: must be exact, not within tolerance.
            s1 = withenv("LUNA_USE_RUST_NATIVE" => "1",
                         "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                   max_dt=dt, min_dt=dt, native_threads=1)
            end
            s4 = withenv("LUNA_USE_RUST_NATIVE" => "1",
                         "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                   max_dt=dt, min_dt=dt, native_threads=4)
            end
            solve(s1, L)
            solve(s4, L)
            @test s1.yn == s4.yn
        end
    end
end
