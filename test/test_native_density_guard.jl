using TestItems

@testitem "Native density z-independence guard" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: RustNativeStepper, NativeIneligible
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # Same mode-averaged setup as test_native_phase1.jl, but exercised
        # through the low-level API: a plain constant Array{ComplexF64} linop
        # paired with a z-varying densityfun. Julia's own TransModeAvg
        # re-evaluates densityfun(z) every stage and handles this correctly;
        # only the dedicated ZDepLinopMarcatili path re-evaluates it on the
        # native side too (Phase 7). Everything else bakes density(0) into a
        # constant and must now refuse eligibility (BACKLOG Phase B.3 /
        # REVIEW.md §3.4) rather than silently propagating the wrong physics.
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
              shotnoise=false, energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        t0 = 0.0
        dt = 0.01

        @testset "z-varying densityfun is rejected" begin
            dens0 = transform.densityfun(0.0)
            zvarying_densityfun(z) = dens0 * (1 + z / flength)

            f_zvarying = NonlinearRHS.TransModeAvg(
                transform.Pto, transform.Eto, transform.Eωo, transform.Pωo,
                transform.FT, transform.resp, transform.grid, zvarying_densityfun,
                transform.norm!, transform.aeff, transform.Et_noise, transform.Et_nl)

            @test_throws NativeIneligible RustNativeStepper(
                f_zvarying, linop, copy(Eω), t0, dt;
                rtol=1e-6, atol=1e-10, flength=flength)
        end

        @testset "constant densityfun is still eligible" begin
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      rtol=1e-6, atol=1e-10, flength=flength)
            @test s_ru isa RustNativeStepper
        end
    end
end
