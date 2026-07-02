using TestItems

@testitem "Native default workload actually runs natively" tags=[:rust] begin
    import Test: @test, @test_skip, @testset

    # BACKLOG.md Phase C.2 — the regression test that would have caught
    # REVIEW.md §3.2: the fork's flagship default field-resolved workload
    # (`prop_capillary` with `plasma = !envelope`, i.e. plasma ON by default)
    # silently fell back to the Julia stepper because the native plasma
    # wiring needed a Rust ionisation LUT handle that only got built behind
    # the separate, opt-in `LUNA_USE_RUST_IONISATION` toggle. Phase C.1
    # decoupled that handle's construction from the toggle (built whenever
    # the native stepper is enabled, which is the default since Phase 8).
    # This test calls `prop_capillary` with NO environment toggles set at
    # all — the exact out-of-the-box configuration a new user gets — and
    # asserts the stepper `solve_precon` actually picked is
    # `RK45.RustNativeStepper`, not a silent fallback.
    using Luna
    import Logging: with_logger, NullLogger

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

        @testset "Default field-resolved prop_capillary (plasma on) uses RustNativeStepper" begin
            withenv("LUNA_USE_RUST_NATIVE" => nothing, "LUNA_USE_RUST_IONISATION" => nothing,
                    "LUNA_USE_RUST_STEPPER" => nothing) do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure;
                                    λ0, λlims, trange, energy=1e-7, τfwhm=30e-15, saveN=2)
                end
            end
            @test RK45._LAST_STEPPER_TYPE[] <: RK45.RustNativeStepper
        end
    end
end
