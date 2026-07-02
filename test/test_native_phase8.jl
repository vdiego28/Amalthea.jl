using TestItems

@testitem "Native-Rust Phase 8 (default-flip + NativeIneligible fallback)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # This file is the first to exercise `RK45.solve_precon`'s own
        # dispatch branch end-to-end (via `prop_capillary`) rather than
        # constructing `RustNativeStepper`/`PreconStepper` directly — every
        # other native test (Phases 0-7) bypasses `solve_precon` entirely.
        radius = 125e-6
        flength = 0.05
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12
        common = (; λ0, λlims, trange, energy=1e-7, τfwhm=30e-15, saveN=2)

        @testset "Default (env unset) picks native for an eligible config" begin
            # Kerr-only, mode-averaged, RealGrid — eligible since Phase 1.
            kw = (; common..., raman=false, plasma=false, kerr=true, shotnoise=false)

            out_native_explicit = withenv("LUNA_USE_RUST_NATIVE" => "1") do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure; kw...)
                end
            end
            out_default = withenv("LUNA_USE_RUST_NATIVE" => nothing) do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure; kw...)
                end
            end
            out_julia_explicit = withenv("LUNA_USE_RUST_NATIVE" => "0") do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure; kw...)
                end
            end

            Eω_native = out_native_explicit["Eω"][:, end]
            Eω_default = out_default["Eω"][:, end]
            Eω_julia = out_julia_explicit["Eω"][:, end]

            # Same code path (native) taken either way -> exact bit-for-bit match.
            @test Eω_default == Eω_native

            # Confirms the default run did NOT silently take the Julia path:
            # native vs Julia agree only to the established Phase-1 method
            # tolerance (~1e-13), not bit-for-bit, so an exact match here
            # would be suspicious rather than reassuring.
            rel_default_vs_julia = norm(Eω_default - Eω_julia) / norm(Eω_julia)
            println("Phase 8 default-vs-explicit-native (eligible config): ",
                    Eω_default == Eω_native ? "bit-identical" : "MISMATCH")
            println("Phase 8 default-vs-Julia rel (eligible config, expect ~1e-13): ",
                    rel_default_vs_julia)
            @test 0 < rel_default_vs_julia < 1e-8
        end

        @testset "NativeIneligible config falls back to Julia under default, no crash" begin
            # RamanPolarField with thg=false is a documented native-ineligible
            # case (RK45.jl, "native Raman path only supports thg=true").
            # Before Phase 8 this was merely a silently-wrong-physics @warn
            # inside an opt-in path; after Phase 8, native is the default, so
            # this exact config is now reachable by any ordinary user and
            # MUST fall back to the Julia stepper instead of throwing.
            # Raman needs a molecular gas (He has no Raman response at all —
            # `Interface.makeresponse` would error before ever reaching the
            # native-eligibility question), so this testset uses :N2
            # locally instead of the shared `gas = :He`.
            gas_raman = :N2
            kw = (; common..., raman=true, thg=false, rotation=false, vibration=true,
                  plasma=false, kerr=true, shotnoise=false)

            out_default = withenv("LUNA_USE_RUST_NATIVE" => nothing) do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas_raman, pressure; kw...)
                end
            end
            out_julia_explicit = withenv("LUNA_USE_RUST_NATIVE" => "0") do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas_raman, pressure; kw...)
                end
            end

            Eω_default = out_default["Eω"][:, end]
            Eω_julia = out_julia_explicit["Eω"][:, end]

            # Both runs take the identical Julia code path (the ineligible
            # native attempt in the "default" run is caught and discarded
            # before any stepping happens), so agreement should be at the
            # Julia-vs-Julia reproducibility floor established by
            # test_stepper_rust.jl (~1e-6, FFTW summation-order), not exact
            # equality and not the ~1e-13 native-vs-Julia method tolerance.
            rel = norm(Eω_default - Eω_julia) / norm(Eω_julia)
            println("Phase 8 NativeIneligible fallback rel (expect Julia-vs-Julia floor): ", rel)
            @test rel < 1e-6
        end

        @testset "Dense-output (quartic) interpolation matches Julia, not just endpoints" begin
            # Regression test for a real bug found while verifying Phase 8:
            # `RustNativeStepper`'s dense output between accepted steps was
            # LINEAR (a documented stopgap since Phase 0), while
            # `PreconStepper`'s is the full quartic `interpC` fit from all 7
            # RK stages. The final field always matched Julia to ~1e-15 —
            # only values *resampled between* actual adaptive steps
            # (exactly what `MemoryOutput`/every `saveN>2` config does)
            # differed, by ~1-2% for a representative config, which is what
            # broke nearly every general-purpose test once Phase 8 made
            # native the default (they all sample far more densely than 2
            # points). Fixed by exporting the 7 stages via `get_ks_stage`
            # and porting the same `interpC` formula
            # (`interpolate(s::RustNativeStepper, ti)`), plus a new
            # `native_apply_prop` FFI call to re-express the polynomial
            # correction at the query time — see PORT_LOG Phase 8.
            kw = (; common..., saveN=50, raman=false, plasma=false, kerr=true, shotnoise=false)

            out_native = withenv("LUNA_USE_RUST_NATIVE" => "1") do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure; kw...)
                end
            end
            out_julia = withenv("LUNA_USE_RUST_NATIVE" => "0") do
                with_logger(NullLogger()) do
                    prop_capillary(radius, flength, gas, pressure; kw...)
                end
            end

            Eω_native = out_native["Eω"]
            Eω_julia = out_julia["Eω"]
            rel = norm(Eω_native - Eω_julia) / norm(Eω_julia)
            println("Phase 8 dense-output (saveN=50) native-vs-Julia rel: ", rel)
            @test rel < 1e-8
        end
    end
end
