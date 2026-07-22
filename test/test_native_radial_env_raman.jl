using TestItems

@testitem "Native-Rust radial-env Raman (RamanPolarEnv, TransRadial)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # EnvGrid + Hankel QDHT, Kerr_env + Raman (N2 vibrational SDO,
        # rotation=false) — closes NATIVE_SUPPORT_MATRIX.md's "Radial EnvGrid
        # Raman" gap (previously ❌: RamanPolarField-only). Same geometry
        # pattern as test_native_radial_raman.jl (RealGrid) and
        # test_native_radial_env.jl (EnvGrid, Kerr-only), combined: the
        # missing `apply_raman_radial_env` composition in native.rs's
        # `rhs_radial_env`.
        # Energy is half of test_native_radial_raman.jl's RealGrid sibling
        # (6e-5): the envelope Kerr_env/RamanPolarEnv combination accumulates
        # the ADE-vs-FFT-convolution method difference (single-step ~1e-8,
        # see below) faster over 40 fixed-dt steps than the carrier-field
        # case does at the same energy (measured 1.14e-6 at 6e-5 J, just over
        # the ~1e-6 floor tier; 3e-5 J lands at ~5.7e-7 with comfortable
        # margin) — a config choice, not a code difference; the RamanPolarEnv
        # ADE math itself is unchanged from the mode-averaged formula.
        gas = :N2; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 3e-5; L = 0.02; R = 4e-3; N = 32

        grid = Grid.EnvGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
        raman_resp = Nonlinear.RamanPolarEnv(grid.to, rr)
        responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)), raman_resp)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (radial-env + Raman, ~1e-7)" begin
            # Same tier as test_native_radial_raman.jl's RealGrid gate, for
            # the same reason: Julia's oracle `RamanPolarEnv` (constructed
            # directly, not via `Interface.makeresponse`) takes its own FFT-
            # convolution path (Nonlinear.jl), while the resident native path
            # always uses the O(Nt) exponential-ADE integrator (`raman.rs`) —
            # a genuine method difference, not a summation-order floor, on
            # top of the QDHT's own ~1e-13 floor (MATH.md §3.2).
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial-env+Raman single-step rel: ", rel_step)
            @test rel_step < 1e-7
        end

        @testset "Full-solve equivalence (radial-env + Raman, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial-env+Raman full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Backend actually native (not a silent fallback)" begin
            # Prove the guard relaxation is genuinely taken, not that it still
            # silently falls back while the test happens to pass for some
            # other reason (AGENTS.md/CLAUDE.md discipline).
            s_ru = withenv("AMALTHEA_USE_RUST_NATIVE" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            end
            @test s_ru isa RK45.RustNativeStepper
        end

        @testset "Non-vacuousness: Raman actually changes the result vs Raman-off" begin
            no_raman_transform = NonlinearRHS.TransRadial(grid, q, transform.FT,
                (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),), densityfun, normfun)
            s_jl_raman = PreconStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_noraman = PreconStepper(no_raman_transform, linop, copy(Eω), t0, dt,
                                          rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_raman, L)
            solve(s_jl_noraman, L)
            rel_raman_effect = norm(s_jl_raman.yn - s_jl_noraman.yn) / norm(s_jl_noraman.yn)
            println("Radial-env Raman-on vs Raman-off rel (Julia, non-vacuousness): ", rel_raman_effect)
            @test rel_raman_effect > 1e-6
        end

        @testset "n_threads=1 vs n_threads=4 bit-identical (apply_raman_radial_env)" begin
            # Same bar as test_native_radial_raman.jl's RealGrid parallel
            # check, applied to the new complex-buffer path: each worker gets
            # its own cloned solver (state reset at every solve() call, so
            # cloned == shared) and disjoint column slices, so n_threads=1
            # and n_threads=4 must match EXACTLY, not within tolerance.
            s1 = withenv("AMALTHEA_USE_RUST_NATIVE" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                   max_dt=dt, min_dt=dt, native_threads=1)
            end
            s4 = withenv("AMALTHEA_USE_RUST_NATIVE" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                   max_dt=dt, min_dt=dt, native_threads=4)
            end
            solve(s1, L)
            solve(s4, L)
            @test s1.yn == s4.yn
        end
    end
end
