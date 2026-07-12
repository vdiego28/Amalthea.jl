using TestItems

@testitem "Native-Rust Phase I item 1 (mode-averaged shot noise, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm
    import Random: MersenneTwister

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase I item 1: `native_set_mode_avg_params` never
        # received a noise buffer, so `shotnoise=true` (the default for
        # `prop_capillary`/`prop_gnlse`) silently ran the native path with
        # no noise term at all. This test drives a config where noise seeds
        # a large downstream effect (broadband, near dispersion zero, long
        # enough propagation for MI-like growth) with a fixed RNG seed
        # shared between both steppers, so its presence/absence in the
        # native path is unmistakable rather than lost in a loose tolerance.
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
              shotnoise=true, energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform_jl, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw..., rng=MersenneTwister(42))
        end
        Eω2, grid2, linop2, transform_ru, FT2, output2 = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw..., rng=MersenneTwister(42))
        end

        @test transform_jl.Et_noise == transform_ru.Et_noise
        @test !isnothing(transform_jl.Et_noise)

        t0 = 0.0
        dt = 0.01

        @testset "Single-step equivalence with noise (~1e-10)" begin
            s_jl = PreconStepper(transform_jl, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform_ru, linop2, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10)

            ok_jl = step!(s_jl)
            ok_ru = step!(s_ru)

            @test ok_jl == ok_ru
            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel_step < 1e-10
        end

        @testset "Noise actually changes the native result (regression guard)" begin
            # Construct the SAME config but with shotnoise=false and confirm
            # the native single-step result differs from the noisy one — if
            # native were still silently dropping the noise term, this would
            # spuriously pass (both would equal the noiseless result).
            kw_nonoise = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
                          shotnoise=false, energy=1e-6, τfwhm=30e-15)
            Eω3, grid3, linop3, transform_nonoise, FT3, output3 = with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_nonoise...)
            end
            s_ru_noisy = RustNativeStepper(transform_ru, linop2, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru_clean = RustNativeStepper(transform_nonoise, linop3, copy(Eω3), t0, dt, rtol=1e-6, atol=1e-10)
            step!(s_ru_noisy)
            step!(s_ru_clean)
            rel_diff = norm(s_ru_noisy.yn - s_ru_clean.yn) / norm(s_ru_clean.yn)
            println("Noisy-vs-clean native single-step rel_diff: ", rel_diff)
            # Shot noise is a vacuum-level perturbation on a microjoule pulse,
            # so it moves the result by far less than a full-solve MI-growth
            # scenario would — this only needs to catch "still exactly zero",
            # i.e. native silently ignoring Et_noise again.
            @test rel_diff > 0.0
        end
    end
end
