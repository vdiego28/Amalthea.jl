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

@testitem "Native-Rust dense output: order-5 extra stages in every geometry (BACKLOG S5 item 3)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, interpolate
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # `CpuNativeSim::eval_extra_stage` reproduces `step`'s per-geometry
        # dispatch (free / modal / radial / mode-averaged) for the 2 extra
        # order-5 stages. The testsets above pin only the mode-averaged
        # branch; a wiring slip in any of the other three — wrong `rhs_*`,
        # a missing `ensure_*_at`, a dropped propagate/unpropagate pair —
        # would be invisible there, and *also* invisible to the existing
        # per-geometry equivalence tests, which compare accepted-step values
        # (`s.yn`) and never call `interpolate`. Rather than re-measure
        # convergence order in each geometry (expensive, and the order is a
        # property of the shared tableau, already established above), this
        # asserts the cheaper and more targeted thing: the native and Julia
        # dense output must agree, which they can only do if Rust's extra
        # stages match Julia's `_dp5_extra_stages!` for that geometry.
        θs = (0.25, 0.6, 0.85)

        cases = Tuple{String, Any, Any, Any, Float64}[]

        # ── Radial (TransRadial + resident QDHT) ─────────────────────────
        let gas = :Ar, pres = 1.2, τ = 20e-15, λ0 = 800e-9,
            w0 = 40e-6, energy = 1e-12, L = 0.05, R = 4e-3, N = 32
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
            q = Hankel.QDHT(R, N, dim=2)
            dens0 = PhysData.density(gas, pres)
            linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
            normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, q, z -> dens0, normfun,
                               (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                               Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy,
                                                      w0=w0, propz=-0.15))
            end
            @assert transform isa NonlinearRHS.TransRadial
            push!(cases, ("radial", Eω, linop, transform, 0.01))
        end

        # ── Modal (TransModal + resident libcubature) ────────────────────
        let gas = :Ar, pres = 1.0, τ = 20e-15, λ0 = 800e-9,
            a = 125e-6, energy = 5e-6, L = 0.1
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            modes = (Capillary.MarcatiliMode(a, gas, pres; m=1),
                     Capillary.MarcatiliMode(a, gas, pres; m=2))
            dens0 = PhysData.density(gas, pres)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, z -> dens0,
                               (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                               Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy), modes, :y)
            end
            @assert transform isa NonlinearRHS.TransModal
            push!(cases, ("modal", Eω, linop, transform, 0.02))
        end

        # ── Free-space (TransFree + joint 3-D FFT) ───────────────────────
        let gas = :Ar, pres = 1.0, τ = 20e-15, λ0 = 800e-9,
            w0 = 60e-6, energy = 1e-9, L = 0.01, R = 300e-6, Nx = 8, Ny = 6
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            xygrid = Grid.FreeGrid(R, Nx, R, Ny)
            dens0 = PhysData.density(gas, pres)
            linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
            normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, xygrid, z -> dens0, normfun,
                               (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                               Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0))
            end
            @assert transform isa NonlinearRHS.TransFree
            push!(cases, ("free-space", Eω, linop, transform, 0.002))
        end

        for (name, Eω, linop, transform, dt) in cases
            @testset "$name: native and Julia dense output agree" begin
                t0 = 0.0
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt;
                                     rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                @assert step!(s_ru) && step!(s_jl)
                # Confirm the native stepper really is resident here rather
                # than having silently fallen back — otherwise this testset
                # would compare Julia against Julia and prove nothing.
                @test s_ru isa RustNativeStepper
                for θ in θs
                    ti = t0 + θ*dt
                    rel = norm(interpolate(s_ru, ti) .- interpolate(s_jl, ti)) /
                          norm(interpolate(s_jl, ti))
                    println("$name dense output native-vs-Julia at θ=$θ: rel=$rel")
                    # Same tier as each geometry's own single-step
                    # equivalence test (~1e-13 documented QDHT/cubature
                    # floor), loosened one decade because the order-5
                    # interpolant sums 9 stages instead of comparing one
                    # accepted-step value.
                    @test rel < 1e-12
                end
            end
        end
    end
end

@testitem "Native-Rust dense output: GPU-resident backend (BACKLOG S5 item 3)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: RustNativeStepper, step!, interpolate
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # `CudaNativeSim` doesn't implement `compute_extra_stages` (returns
        # -1), so `interpolate` takes the order-4 `interpC` branch there. Two
        # things that branch needs on the GPU backend, neither of which had
        # any coverage before 2026-07-23 because every GPU test drives the
        # stepper through raw `solve()` and never asks for dense output:
        #   1. `native_apply_prop` must work at all. `CudaNativeSim::apply_prop`
        #      returned a bare -1, so `check_ffi` threw and *every* dense-output
        #      query on the GPU path was a hard error — i.e. any `Luna.run`
        #      with `saveN` under `AMALTHEA_USE_RUST_CUDA_NATIVE=1` crashed.
        #   2. Its result must actually equal `exp(linop*(t2-t1))*y`, which is
        #      checked below against the CPU backend's own `apply_prop` on
        #      identical input — a tight (~1e-15) equivalence, since both are
        #      the same closed-form elementwise complex exponential.
        radius = 125e-6; flength = 0.15; gas = :He; pressure = 1.0
        args = (radius, flength, gas, pressure)
        kw = (; λ0=800e-9, λlims=(200e-9, 4e-6), trange=1e-12, raman=false,
              plasma=false, kerr=true, shotnoise=false, energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        t0 = 0.0; h = 0.01
        s_cpu = RustNativeStepper(transform, linop, copy(Eω), t0, h;
                                  rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)

        gpu_available = true
        gpu_error = nothing
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1", "AMALTHEA_NATIVE_GPU" => "on") do
            local s_gpu
            try
                s_gpu = RustNativeStepper(transform, linop, copy(Eω), t0, h;
                                          rtol=1e-6, atol=1e-10, max_dt=h, min_dt=h)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end
            @assert step!(s_gpu)

            @testset "dense output does not error on the GPU backend" begin
                # Regression for `CudaNativeSim::apply_prop` returning -1.
                y = interpolate(s_gpu, t0 + 0.5h)
                @test length(y) == length(Eω)
                @test all(isfinite, y)
                @test norm(y) > 0
            end

            @testset "GPU apply_prop matches the CPU backend (~1e-15)" begin
                n = length(Eω)
                for frac in (0.25, 0.6, 1.0)
                    a = copy(Eω); b = copy(Eω)
                    for (s, buf) in ((s_gpu, a), (s_cpu, b))
                        rc = ccall((:native_apply_prop, RK45._LIBAMALTHEA_RK45), Cint,
                              (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t, Float64, Float64),
                              s._handle.ptr, buf, Csize_t(n), 0.0, frac*h)
                        @test rc == 0
                    end
                    rel = norm(a .- b) / norm(b)
                    println("GPU-vs-CPU apply_prop at dt=$(frac*h): rel=$rel")
                    @test rel < 1e-14
                end
            end

            # NOT asserted here: the dense-output *convergence order* on the
            # GPU path, which is what would empirically confirm the FSAL fix
            # (`cuda_native.rs` step 0) for this backend the way the CPU
            # testsets above do. It cannot be measured while the GPU-resident
            # RHS contributes no nonlinearity at all — measured `max|kᵢ|` is
            # 3.5e-13 against the CPU backend's 12225 for this exact config,
            # and the GPU's accepted step equals pure linear propagation to
            # 15 digits. With `kᵢ ≈ 0` the interpolant is exact whichever
            # stage sits in slot 0, so the measurement is blind. Tracked as an
            # open item in `docs/dev/BACKLOG.md` / `VANILLA_LUNA_ISSUES.md` §5;
            # re-enable an order check here once it is fixed.
            @test_skip "GPU dense-output convergence order — blocked on the missing GPU-resident nonlinearity (BACKLOG S3)"
        end

        gpu_available || @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
    end
end
