using TestItems

@testitem "Native FFTW wisdom persistence toggle (AMALTHEA_NATIVE_FFTW_WISDOM)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: RustNativeStepper
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
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

        wisdom_path = joinpath(Amalthea.Utils.cachedir(),
                               "native_fftw_wisdom_$(Amalthea.Utils.FFTWthreads())threads")

        # docs/dev/BACKLOG.md resume-queue item 9: this config (mode-averaged,
        # RealGrid, Kerr-only, constant linop) is GPU-eligible under
        # `RK45._gpu_kernel_supports`, and the FFTW-wisdom persistence this
        # file is testing is a CPU-resident-stepper-only concept (the GPU
        # backend uses cuFFT plans, not FFTW). Under a process-wide
        # `AMALTHEA_NATIVE_GPU=on`, every `RustNativeStepper` below would
        # silently take the GPU path and never touch `wisdom_path` at all —
        # pin the backend explicitly so this test means what it says
        # regardless of the ambient environment.
        withenv("AMALTHEA_NATIVE_GPU" => "off") do

        @assert !RK45._gpu_native_eligible(transform, linop, length(Eω)) "backend guard regression: this config must be CPU-native under AMALTHEA_NATIVE_GPU=off regardless of AMALTHEA_USE_RUST_CUDA_NATIVE"

        # docs/dev/BACKLOG.md S1 item 1 / docs/dev/native-port/PLANS.md §1.
        # T1: default (env var unset) must not touch the on-disk wisdom file
        # at all — this is the whole point of making persistence opt-in.
        @testset "T1: default is off, no disk writes" begin
            withenv("AMALTHEA_NATIVE_FFTW_WISDOM" => nothing) do
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
            withenv("AMALTHEA_NATIVE_FFTW_WISDOM" => "1") do
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

        end # withenv("AMALTHEA_NATIVE_GPU" => "off")
    end
end
