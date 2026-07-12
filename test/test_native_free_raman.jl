using TestItems

@testitem "Native-Rust Phase I item 6 (Raman, free-space)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid + joint 3-D FFT, Kerr + Raman (N2 vibrational SDO,
        # thg=true, no plasma) — docs/dev/BACKLOG.md Phase I item 6's free-space-Raman
        # half (`apply_raman_free` in native.rs mirrors `apply_raman_radial`
        # per (y,x) column). Same geometry as test_native_free_plasma.jl;
        # :N2 vibration=true, rotation=false gives a single-SDO
        # CombinedRamanResponse eligible for the native path.
        gas = :N2; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 6e-5; L = 0.02; R = 1.5e-3; Nx = 10; Ny = 8

        grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
        raman_resp = Nonlinear.RamanPolarField(grid.to, rr)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), raman_resp)
        linop   = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, xygrid, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransFree "Expected TransFree"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (free-space + Raman, ~1e-7)" begin
            # Not 1e-12/1e-13 like the plasma/Kerr-only free-space gates —
            # same ADE-vs-FFT-convolution method-difference floor documented
            # in test_native_radial_raman.jl (Julia's oracle `RamanPolarField`
            # here has no `rust_handle`, so it takes the FFT-convolution
            # path; the resident native path always uses the O(Nt) ADE
            # integrator).
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space+Raman single-step rel: ", rel_step)
            @test rel_step < 1e-7
        end

        @testset "Full-solve equivalence (free-space + Raman, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space+Raman full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: Raman actually changes the result vs Raman-off" begin
            no_raman_transform = NonlinearRHS.TransFree(grid, xygrid, transform.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),), densityfun, normfun)
            s_jl_raman = PreconStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_noraman = PreconStepper(no_raman_transform, linop, copy(Eω), t0, dt,
                                          rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_raman, L)
            solve(s_jl_noraman, L)
            rel_raman_effect = norm(s_jl_raman.yn - s_jl_noraman.yn) / norm(s_jl_noraman.yn)
            println("Free-space Raman-on vs Raman-off rel (Julia, non-vacuousness): ", rel_raman_effect)
            @test rel_raman_effect > 1e-6

            # Triangulate: native vs Julia (both Raman-on) and native vs
            # Julia's Raman-off result (must NOT match) — same rationale as
            # the plasma test's regression guard.
            s_ru_raman = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                            rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_ru_raman, L)
            rel_native_vs_jl = norm(s_ru_raman.yn - s_jl_raman.yn) / norm(s_jl_raman.yn)
            rel_native_vs_noraman = norm(s_ru_raman.yn - s_jl_noraman.yn) / norm(s_jl_noraman.yn)
            println("Free-space native vs Julia (both Raman-on): ", rel_native_vs_jl)
            println("Free-space native vs Julia no-Raman (must NOT match — regression guard): ",
                     rel_native_vs_noraman)
            @test rel_native_vs_jl < 1e-6
            @test rel_native_vs_noraman > 1e-6
        end
    end
end
