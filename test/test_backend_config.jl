using TestItems

@testitem "BackendConfig / backend_report()" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Logging: with_logger, NullLogger

    # docs/dev/BACKLOG.md S4 item 1 (suggestion 6) — Config.BackendConfig/backend_config()
    # and Amalthea.backend_report() were implemented but never gated by a test,
    # the one piece of S4's own stated gate ("a backend_report() test asserting
    # RustNativeStepper for default prop_capillary and Julia stepper under
    # native=:off") left undone. Fills that gap.

    @testset "backend_config() defaults and truthiness" begin
        withenv("LUNA_USE_RUST_NATIVE" => nothing, "LUNA_USE_RUST_STEPPER" => nothing,
                "LUNA_USE_RUST_IONISATION" => nothing, "LUNA_USE_RUST_RAMAN" => nothing,
                "LUNA_USE_RUST_DISPERSION" => nothing, "LUNA_USE_RUST_QDHT" => nothing,
                "LUNA_QDHT_BLAS" => nothing, "LUNA_USE_RUST_CUDA_NATIVE" => nothing,
                "LUNA_NATIVE_FFTW_WISDOM" => nothing) do
            cfg = Amalthea.Config.backend_config()
            @test cfg.native == true   # only toggle defaulting on (Phase 8)
            @test cfg.stepper == false
            @test cfg.ionisation == false
            @test cfg.raman == false
            @test cfg.dispersion == false
            @test cfg.qdht == false
            @test cfg.qdht_blas == false
            @test cfg.cuda_native == false
            @test cfg.native_wisdom == false
        end

        # re-reads ENV every call (not a cached singleton) — flipping a
        # toggle mid-session must be visible immediately.
        withenv("LUNA_USE_RUST_NATIVE" => "0") do
            @test Amalthea.Config.backend_config().native == false
        end
    end

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        radius = 125e-6
        flength = 0.02
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        @testset "backend_report() reflects RustNativeStepper by default" begin
            withenv("LUNA_USE_RUST_NATIVE" => nothing) do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure;
                                    λ0, λlims, trange, energy=1e-7, τfwhm=30e-15, saveN=2)
                end
                report = Amalthea.backend_report()
                @test report.config.native == true
                @test report.last_stepper_type <: RK45.RustNativeStepper
            end
        end

        @testset "backend_report() reflects the Julia stepper under LUNA_USE_RUST_NATIVE=0" begin
            withenv("LUNA_USE_RUST_NATIVE" => "0") do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure;
                                    λ0, λlims, trange, energy=1e-7, τfwhm=30e-15, saveN=2)
                end
                report = Amalthea.backend_report()
                @test report.config.native == false
                @test report.last_stepper_type <: RK45.PreconStepper
            end
        end
    end
end
