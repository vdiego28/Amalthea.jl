using TestItems

@testitem "Native-Rust Phase 6 (free-space, 3-D FFT)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid + joint 3-D FFT, scalar Kerr only (no shot-noise) —
        # Phase 6's first gate. See docs/native-port/MATH.md §3.4. Small
        # transverse grid keeps the equivalence test cheap; the
        # dimension-order/normalization correctness itself is verified at
        # the FFT-primitive level in fftw.rs, not here. Nx != Ny deliberately
        # (a square Nx=Ny grid with a radially-symmetric input is invariant
        # under a y<->x transpose, so it can't actually detect a swapped-axis
        # bug in the M-array layout or RealFft3d's dimension order — only the
        # standalone fftw.rs unit test would catch that. A rectangular grid
        # makes this equivalence test a genuine RHS-level backstop too).
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        w0 = 60e-6; energy = 1e-9; L = 0.01; R = 300e-6; Nx = 8; Ny = 6

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Luna.NonlinearRHS.TransFree "Expected TransFree"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (free-space, ~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space single-step rel: ", rel_step)
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (free-space, fixed dt)" begin
            # Fixed step size (max_dt=min_dt=dt): see TESTING.md §3 — isolates
            # state-accumulation error from adaptive-path agreement.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
