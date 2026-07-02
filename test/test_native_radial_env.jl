using TestItems

@testitem "Native-Rust Phase D.1 (EnvGrid radial, resident QDHT)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # EnvGrid counterpart of test_native_radial.jl (Phase 3, RealGrid) —
        # BACKLOG.md Phase D.1 / MATH.md §3.2's deferred EnvGrid-radial
        # follow-up. Same geometry/materials, only the grid type and Kerr
        # response change (Kerr_env's 3/4 SVEA factor, c2c FFT convention).
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.05; R = 4e-3; N = 32

        grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Luna.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.001

        @testset "Single-step equivalence (EnvGrid radial, ~1e-12)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("EnvGrid radial single-step rel: ", rel_step)
            # Same ~1e-13-to-~1e-12 QDHT-floor tier as the RealGrid radial
            # test (MATH.md §3.2), applied twice per RHS via apply_cplx here.
            @test rel_step < 1e-12
        end

        @testset "Full-solve equivalence (EnvGrid radial, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("EnvGrid radial full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: Kerr actually changes the result vs kerr-off" begin
            # Guard against a passing-by-coincidence comparison where the Kerr
            # term silently didn't fire (e.g. a c2c convention bug that zeroes
            # the field) — confirm Kerr meaningfully changes the propagation
            # result, per the Phase 4/5 lesson. The native path requires
            # exactly one Kerr response (RK45.jl's radial gate), so the
            # no-Kerr control run uses the (independently-trusted) Julia
            # PreconStepper on both sides — this isolates "does Kerr matter
            # here at all" from the Rust-vs-Julia correctness check above.
            # Direct TransRadial construction (not Luna.setup) — an empty
            # `responses` tuple is ambiguous between `setup`'s radial and
            # modal methods (both accept an empty-tuple-compatible arg in
            # that position), so build the no-Kerr transform directly instead.
            #
            # Uses a much stronger input field than the equivalence tests
            # above (1e-12 J there is intentionally weak, to keep Julia/Rust
            # summation-order noise well above any physical signal it might
            # be masking — but at that energy Kerr's own contribution here
            # turned out to be only ~7e-10 relative, indistinguishable from
            # that same FP-noise floor, i.e. this check's threshold would be
            # meaningless at the equivalence-test energy). Kerr's relative
            # effect scales linearly with energy at this field strength
            # (confirmed empirically: 1e-9 J gave ~6.8e-7, right at the
            # 1e-6 threshold); 1e-7 J (five orders above the equivalence
            # tests, still far below typical HCF pulse energies) clears it
            # comfortably.
            inputs_strong = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=1e-7, w0=w0, propz=-0.15)
            Eω_strong, transform_strong, _ = with_logger(NullLogger()) do
                Luna.setup(grid, q, densityfun, normfun, responses, inputs_strong)
            end
            no_kerr_transform = NonlinearRHS.TransRadial(grid, q, transform_strong.FT, (),
                                                          densityfun, normfun)
            s_jl_kerr = PreconStepper(transform_strong, linop, copy(Eω_strong), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_nokerr = PreconStepper(no_kerr_transform, linop, copy(Eω_strong), t0, dt,
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_kerr, L)
            solve(s_jl_nokerr, L)
            rel_kerr_effect = norm(s_jl_kerr.yn - s_jl_nokerr.yn) / norm(s_jl_nokerr.yn)
            println("EnvGrid radial Kerr-on vs Kerr-off rel (Julia, non-vacuousness): ", rel_kerr_effect)
            @test rel_kerr_effect > 1e-6
        end
    end
end
