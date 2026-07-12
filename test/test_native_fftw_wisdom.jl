using TestItems

@testitem "Native FFTW wisdom persistence toggle (LUNA_NATIVE_FFTW_WISDOM)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: RustNativeStepper
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true, shotnoise=false,
              energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        t0 = 0.0
        dt = 0.01

        wisdom_path = joinpath(Luna.Utils.cachedir(),
                               "native_fftw_wisdom_$(Luna.Utils.FFTWthreads())threads")

        # docs/dev/BACKLOG.md S1 item 1 / docs/dev/native-port/PLAN_FFTW_WISDOM_FIX.md.
        # T1: default (env var unset) must not touch the on-disk wisdom file
        # at all — this is the whole point of making persistence opt-in.
        @testset "T1: default is off, no disk writes" begin
            withenv("LUNA_NATIVE_FFTW_WISDOM" => nothing) do
                isfile(wisdom_path) && rm(wisdom_path)
                @test !RK45._native_wisdom_enabled()

                s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
                s2 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

                @test !isfile(wisdom_path)
            end
        end

        # T2: opt-in (env var = "1") writes a non-empty wisdom file and does
        # not error across repeated constructions in the same process.
        @testset "T2: opt-in exports and imports without error" begin
            withenv("LUNA_NATIVE_FFTW_WISDOM" => "1") do
                isfile(wisdom_path) && rm(wisdom_path)
                @test RK45._native_wisdom_enabled()

                s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
                @test isfile(wisdom_path)
                @test filesize(wisdom_path) > 0

                # Second construction imports what the first exported; must
                # not throw and must still produce a working stepper.
                s2 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
                @test isfile(wisdom_path)
            end
            isfile(wisdom_path) && rm(wisdom_path)
        end
    end
end
