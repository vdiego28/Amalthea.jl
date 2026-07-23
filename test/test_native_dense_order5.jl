using TestItems

@testitem "Native-Rust dense output: order-5 continuous extension (BACKLOG S5 item 3)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, interpolate
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # ── Test Geometry & Setup (mode-averaged Kerr, RealGrid — same
        # scope as test_native_phase1.jl, and the only geometry
        # `compute_extra_stages`'s CPU dispatch is verified against here) ──
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

        """
        Near-exact reference value of the dense output at absolute position
        `ti`, obtained by stepping a *fresh*, much finer fixed-step stepper
        forward from `(t0, Eω)` and reading its own (order-5) `interpolate`
        once `ti` falls inside its last accepted step. `dt_fine` is chosen
        far smaller than any `h` under test, so this reference's own local
        interpolation defect (`O(dt_fine^6)`) is many orders of magnitude
        below the `O(h^6)` / `O(h^5)` effects being measured below — see the
        design record for the exact margin.
        """
        function reference_at(ti, dt_fine)
            s = RustNativeStepper(transform, linop, copy(Eω), t0, dt_fine;
                                   rtol=1e-6, atol=1e-10, max_dt=dt_fine, min_dt=dt_fine)
            while s.tn < ti
                ok = step!(s)
                @assert ok "fine reference stepper rejected a fixed-size step"
            end
            return interpolate(s, ti)
        end

        # Base step and a few halvings. `h0` is deliberately coarse — a
        # sizeable fraction of `flength` — because the *interpolation* defect
        # is what is being measured, and this propagator handles the linear
        # (dispersive) part exactly via the integrating factor, so the defect
        # comes only from the comparatively weak Kerr nonlinearity. At the
        # obvious "physically sensible" step sizes (h ~ 2e-3) the order-5
        # defect is already ~6e-15 relative, i.e. at the double-precision
        # noise floor, and every measured ratio degenerates to ~1 for both
        # interpolants — a regime in which this test proves nothing. See
        # `docs/dev/native-port/portlog-inbox/dense-order5.md`. Steps are
        # accepted regardless of the error estimate because `min_dt == max_dt`
        # forces `stepcontrol_pi` to accept (native.rs), which is what makes
        # such a coarse fixed step usable here at all.
        h0 = 4e-2
        hs = [h0, h0/2, h0/4, h0/8]
        θs = [0.15, 0.35, 0.55, 0.75, 0.9]  # avoid the (trivially exact) endpoints

        @testset "Order-5 interpolant: single-step local defect scales ~h^6" begin
            errs5 = Float64[]
            for h in hs
                s = RustNativeStepper(transform, linop, copy(Eω), t0, h;
                                       rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)
                ok = step!(s)
                @assert ok "test stepper rejected a fixed-size step"

                dt_fine = h / 32
                max_err = 0.0
                for θ in θs
                    ti = t0 + θ*h
                    yi = interpolate(s, ti)
                    yref = reference_at(ti, dt_fine)
                    err = norm(yi .- yref) / norm(yref)
                    max_err = max(max_err, err)
                end
                push!(errs5, max_err)
                println("order-5 dense output: h=$h  max local err=$max_err")
            end

            ratios5 = [errs5[i-1]/errs5[i] for i in 2:length(errs5)]
            println("order-5 ratios (expect ~2^6=64): ", ratios5)

            # Loose bracket around the theoretical 2^6=64: confirms genuine
            # order-5 (local defect O(h^6)) behaviour without being so tight
            # that ordinary FP/summation-order noise (this problem's
            # nonlinear RHS, not the toy scalar ODE used to first verify
            # these coefficients — see the design record) makes the test
            # flaky. Order-4 behaviour (ratio ~32) or lower is what a
            # regression (wrong coefficients, wrong node, wrong stage
            # combination) would show up as, and is well outside this
            # bracket.
            for r in ratios5
                @test 40 < r < 100
            end
        end

        @testset "Order-4 interpolant (unchanged interpC): single-step local defect scales ~h^5" begin
            # Same measurement, but reconstructed from the *original* quartic
            # `interpC` table directly (still exported, untouched by this
            # item) instead of `interpolate`, which now prefers order-5
            # whenever `native_compute_extra_stages` succeeds. This is the
            # baseline the order-5 result above must beat, confirming the
            # comparison is apples-to-apples on the *same* real simulation
            # RHS/propagator, not just the toy ODE used during coefficient
            # sourcing (docs/dev/native-port/portlog-inbox/dense-order5.md).
            errs4 = Float64[]
            for h in hs
                s = RustNativeStepper(transform, linop, copy(Eω), t0, h;
                                       rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)
                ok = step!(s)
                @assert ok "test stepper rejected a fixed-size step"

                n = length(s.yn)
                kbuf = similar(s.yn)
                dt_fine = h / 32
                max_err = 0.0
                for θ in θs
                    ti = t0 + θ*h
                    σ = θ
                    σ2 = σ^2; σ3 = σ2*σ; σ4 = σ3*σ
                    b4 = ntuple(ii -> σ*RK45.interpC[1,ii] + σ2*RK45.interpC[2,ii] +
                                       σ3*RK45.interpC[3,ii] + σ4*RK45.interpC[4,ii], Val(7))
                    yi = zero(s.yn)
                    for ii = 1:7
                        rc = ccall((:get_ks_stage, RK45._LIBAMALTHEA_RK45), Cint,
                              (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                              s._handle.ptr, Csize_t(ii - 1), kbuf, Csize_t(n))
                        @assert rc == 0
                        yi .+= kbuf .* b4[ii]
                    end
                    out = @. s.y + s.dt * yi
                    rc = ccall((:native_apply_prop, RK45._LIBAMALTHEA_RK45), Cint,
                          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t, Float64, Float64),
                          s._handle.ptr, out, Csize_t(n), s.t, ti)
                    @assert rc == 0

                    yref = reference_at(ti, dt_fine)
                    err = norm(out .- yref) / norm(yref)
                    max_err = max(max_err, err)
                end
                push!(errs4, max_err)
                println("order-4 dense output: h=$h  max local err=$max_err")
            end

            ratios4 = [errs4[i-1]/errs4[i] for i in 2:length(errs4)]
            println("order-4 ratios (expect ~2^5=32): ", ratios4)
            for r in ratios4
                @test 16 < r < 64
            end
        end

        @testset "Julia PreconStepper order-5 interpolant: local defect scales ~h^6" begin
            # The pure-Julia fallback stepper gained the same order-5
            # continuous extension (`_dp5_extra_stages!` + `interpC5_weights`
            # in RK45.jl), computing its 2 extra stages through `s.fbar!`
            # instead of the `native_compute_extra_stages` FFI. Exercised here
            # because it is the *oracle* the native path is validated against
            # everywhere else in this suite — if only the native side were
            # order-5 the two would silently disagree on every dense-output
            # point. Also the only coverage `_dp5_extra_stages!` has.
            errsjl = Float64[]
            for h in hs
                s = PreconStepper(transform, linop, copy(Eω), t0, h;
                                  rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)
                ok = step!(s)
                @assert ok "Julia test stepper rejected a fixed-size step"

                dt_fine = h / 32
                max_err = 0.0
                for θ in θs
                    ti = t0 + θ*h
                    yi = interpolate(s, ti)
                    yref = reference_at(ti, dt_fine)
                    max_err = max(max_err, norm(yi .- yref) / norm(yref))
                end
                push!(errsjl, max_err)
                println("Julia order-5 dense output: h=$h  max local err=$max_err")
            end

            ratiosjl = [errsjl[i-1]/errsjl[i] for i in 2:length(errsjl)]
            println("Julia order-5 ratios (expect ~2^6=64): ", ratiosjl)
            for r in ratiosjl
                @test 40 < r < 100
            end
        end

        @testset "Native and Julia dense output agree (two-tier §4 single-step check)" begin
            # Both steppers now build the *same* order-5 polynomial from the
            # same 9 stages, so their dense output must agree far more tightly
            # than either agrees with the true solution — this is what pins
            # the Rust `DP5_EXTRA_*`/`eval_extra_stage` implementation to
            # Julia's `dp5_extra_*`/`_dp5_extra_stages!`, independently of how
            # accurate the interpolant itself is.
            h = hs[end]
            s_rs = RustNativeStepper(transform, linop, copy(Eω), t0, h;
                                     rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, h;
                                 rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)
            @assert step!(s_rs) && step!(s_jl)
            for θ in θs
                ti = t0 + θ*h
                a = interpolate(s_rs, ti)
                b = interpolate(s_jl, ti)
                rel = norm(a .- b) / norm(b)
                println("native-vs-Julia dense output at θ=$θ: rel=$rel")
                @test rel < 1e-12
            end
        end
    end
end
