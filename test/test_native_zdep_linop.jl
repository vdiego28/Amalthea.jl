using TestItems

@testitem "Native-Rust Phase 7 (z-dependent linop, pressure-gradient capillary)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm
    import Amalthea.Capillary: gradient, MarcatiliMode
    import Amalthea.PhysData: wlfreq, c
    import Amalthea: Maths

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # ── Test Geometry & Setup ─────────────────────────────────────────────
        # A pressure-gradient (tapered-fill) capillary: `pressure` as a 2-tuple
        # triggers Interface.jl's `const_linop(radius,pressure) = Val(false)`
        # branch, i.e. LinearOps.make_linop (z-dependent), which for a
        # constant-radius MarcatiliMode with a graded core dispatches to the
        # Capillary.jl specialization that returns a `ZDepLinopMarcatili`
        # wrapper (see MATH.md §3.5).
        radius = 125e-6
        flength = 0.5
        gas = :Ar
        pressure = (0.5, 5.0)  # p0, p1 -- a real pressure gradient, not a vacuous no-op
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true, shotnoise=false,
                energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        @test linop isa Amalthea.Capillary.ZDepLinopMarcatili

        t0 = 0.0
        dt = 0.01 # 10 mm step size to get error far above machine precision floor

        @testset "β1(z) analytic closed form vs BigFloat ground truth" begin
            # Two-tier check (see docs/dev/native-port/BETA1_ANALYTIC.md): this
            # tier proves Rust's resident β1(z) is exact, independent of
            # Julia's own `dispersion` (an adaptive finite difference with a
            # real, repeatable ~1e-12 relative error against the true
            # derivative) -- so this test does NOT compare against
            # `dispersion`, it compares against a BigFloat-precision
            # derivative of the exact same formula.
            coren, densf_check = gradient(gas, flength, pressure...)
            mode = MarcatiliMode(radius, coren)
            ω0 = wlfreq(λ0)

            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      flength=flength)
            dens_r = Ref{Float64}(0.0)
            beta1_r = Ref{Float64}(0.0)
            for z in (0.0, 0.0025, 0.01, 0.25, 0.49, 0.5)
                ccall((:native_debug_beta1_at, RK45._LIBLUNA_RUST_RK45), Cint,
                      (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
                      s_ru._handle.ptr, z, dens_r, beta1_r)
                β_of_ω(ω) = ω/c*real(Amalthea.Capillary.neff(mode, ω; z=z))
                β1_bigfloat = Float64(Maths.derivative(β_of_ω, BigFloat(ω0), 1))
                rel = abs(beta1_r[] - β1_bigfloat) / abs(β1_bigfloat)
                @test rel < 1e-9
            end
        end

        @testset "Single-step equivalence" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      flength=flength)

            ok_jl = step!(s_jl)
            ok_ru = step!(s_ru)

            @test ok_jl == ok_ru
            @test s_jl.tn ≈ s_ru.tn
            @test s_jl.dtn ≈ s_ru.dtn rtol=1e-8
            @test s_jl.err ≈ s_ru.err rtol=1e-8
            @test s_jl.errlast ≈ s_ru.errlast rtol=1e-8

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Phase 7 single-step rel_step: ", rel_step)
            @test rel_step < 1e-10
        end

        @testset "Full-solve equivalence (fixed step size)" begin
            # Fixed step size (max_dt=min_dt=dt) isolates genuine multi-step
            # state-accumulation error from adaptive-path divergence -- see
            # docs/dev/native-port/TESTING.md §3 (established Phase 1/2, applied
            # to every phase since).
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt, flength=flength)

            solve(s_jl, flength)
            solve(s_ru, flength)

            # Self-validating: check the pressure gradient actually changes the
            # Julia oracle's result relative to a constant-pressure run of the
            # same fibre, before trusting the Rust-vs-Julia comparison on it
            # (Phase 4/5 discipline -- a passing comparison where both paths
            # silently ignore the z-dependence would prove nothing).
            args_const = (radius, flength, gas, pressure[2])
            Eω_c, grid_c, linop_c, transform_c, FT_c, _ = with_logger(NullLogger()) do
                Interface.prop_capillary_args(args_const...; kw...)
            end
            s_const = PreconStepper(transform_c, linop_c, copy(Eω_c), t0, dt, rtol=1e-6,
                                     atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_const, flength)
            rel_gradient_effect = norm(s_jl.yn - s_const.yn) / norm(s_jl.yn)
            println("Phase 7 pressure-gradient effect (vs constant p1 fill): ", rel_gradient_effect)
            @test rel_gradient_effect > 1e-6

            # Tolerance floor here is NOT ~1e-10 like every other native-port
            # phase: Rust's β1(z) is an exact analytic closed form, while
            # Julia's own `Modes.dispersion` is an adaptive finite difference
            # with a small (~1e-12 relative) but *systematic* (not random)
            # error against the true derivative. Over this test's broad
            # bandwidth (λlims=200nm-4000nm) and 0.5m gradient length, that
            # systematic offset accumulates coherently into a full-solve
            # difference around ~1e-4 -- confirmed (via a kerr=false control
            # run showing the identical magnitude) to be entirely this β1
            # method difference, not a bug in the RHS or linop assembly. See
            # docs/dev/native-port/BETA1_ANALYTIC.md for the full derivation and
            # verification. Do not tighten this by reverting β1 to a
            # Julia-value-reproducing LUT.
            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Phase 7 full-solve rel_solve: ", rel_solve)
            @test rel_solve < 1e-3
        end
    end
end
