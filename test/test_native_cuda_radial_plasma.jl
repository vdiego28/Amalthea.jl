using TestItems

@testitem "Native-Rust GPU-resident stepper (CUDA, radial RealGrid PPT plasma)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Ionisation
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import LinearAlgebra: I, norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA radial plasma tests require the Rust library")
        @test_skip "Rust library not found"
    else
        gas = :Ar; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 6e-5; L = 0.02; R = 4e-3; N = 32
        grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        q = Hankel.QDHT(R, N, dim=2)
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        ionpot = PhysData.ionisation_potential(gas)
        ionrate = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            Ionisation.IonRatePPTCached(gas, λ0)
        end
        plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), plasma)
        linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0,
                                        propz=-0.1)
        Eω, transform, _ = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end
        @assert transform isa Amalthea.NonlinearRHS.TransRadial

        t0 = 0.0
        dt = 0.0005
        n = length(Eω)
        n_time = length(grid.t)
        n_time_over = length(grid.to)
        n_r = q.N
        n_spec = n ÷ n_r
        T_normal = Matrix{Float64}(q.T)
        M_normal = (grid.ωwin .* (-im .* grid.ω)) ./ (2 .* normfun(0.0))
        m_re_normal = real.(M_normal)
        m_im_normal = imag.(M_normal)
        ion_ptr = ionrate.rust_handle.ptr

        make_cpu(field=Eω, time_step=dt; max_dt=time_step, min_dt=time_step) =
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                    "AMALTHEA_NATIVE_GPU" => "off",
                    "AMALTHEA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(field), t0, time_step;
                                  rtol=1e-6, atol=1e-10, max_dt, min_dt)
            end
        make_gpu(field=Eω, time_step=dt, rhs=transform;
                 max_dt=time_step, min_dt=time_step) =
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on",
                    "AMALTHEA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(rhs, linop, copy(field), t0, time_step;
                                  rtol=1e-6, atol=1e-10, max_dt, min_dt)
            end
        getk(s, i=0) = begin
            out = zeros(ComplexF64, n)
            rc = ccall((:get_ks_stage, libpath), Cint,
                       (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                       s._handle.ptr, Csize_t(i), out, Csize_t(n))
            rc == 0 || error("get_ks_stage failed with rc=$rc")
            out
        end
        set_field(ptr, field) = ccall(
            (:set_field, libpath), Cint,
            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t), ptr, field, Csize_t(length(field)))
        set_radial(ptr, T, sf, si, win, kfac, m) = ccall(
            (:native_set_radial_params, libpath), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Ptr{Float64}, Float64,
             Float64, Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
            ptr, Csize_t(n_time), Csize_t(n_time_over), Csize_t(n_r), T,
            sf, si, win, kfac, real.(m), imag.(m))
        set_plasma(ptr) = ccall(
            (:native_set_plasma_params, libpath), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
            ptr, ion_ptr, ionpot, PhysData.e_ratio, plasma.preionfrac,
            plasma.δt, dens0)
        get_ks_stage(s) = getk(s, 0)

        # The dispatch policy is testable without touching CUDA.  Radial
        # plasma is explicit-only, while EnvGrid plasma remains rejected by
        # this GPU slice.
        @test RK45._gpu_kernel_supports(transform, linop)
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                "AMALTHEA_NATIVE_GPU" => "auto") do
            @test !RK45._gpu_native_eligible(transform, linop, n)
        end

        local gpu_error
        local gpu_available = true
        local s_gpu
        try
            s_gpu = make_gpu()
        catch e
            gpu_available = false
            gpu_error = e
        end
        if !gpu_available
            require_cuda && error("CUDA radial plasma setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        else
            @test RK45._native_backend(s_gpu) === :cuda
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_kernel_supports(transform, linop)
                @test RK45._gpu_native_eligible(transform, linop, n)
            end

            @testset "Segmented radial direct stage" begin
                s_cpu = make_cpu()
                k_cpu = get_ks_stage(s_cpu)
                k_gpu = get_ks_stage(s_gpu)
                @test maximum(abs.(k_cpu)) > 100 * 1e-12
                rel_stage = norm(k_gpu - k_cpu) / norm(k_cpu)
                println("CUDA radial PPT direct stage rel: ", rel_stage)
                @test rel_stage < 1e-8
            end

            @testset "Column isolation sentinel" begin
                # Identity QDHT and zero Kerr isolate the radial columns.  A
                # scan that accidentally carries a prefix from one column to
                # the next produces a nonzero spectrum in the zero columns.
                T_identity = Matrix{Float64}(I, n_r, n_r)
                ones_t = ones(Float64, n_time_over)
                ones_m = ones(ComplexF64, n)
                Eω_strong = let input = Fields.GaussGaussField(
                        λ0=λ0, τfwhm=τ, energy=3e-4, w0=w0, propz=-0.1)
                    first(Amalthea.setup(grid, q, densityfun, normfun, responses, input))
                end
                sentinel = zeros(ComplexF64, n)
                # A DC spectral sentinel gives a known nonzero time-domain
                # field in column 1 while every other column remains exactly
                # zero, independent of the physical beam's radial tail.
                sentinel[1] = 1e15
                s_cpu = make_cpu()
                @test set_radial(s_gpu._handle.ptr, T_identity, 1.0, 1.0,
                                 ones_t, 0.0, ones_m) == 0
                @test set_radial(s_cpu._handle.ptr, T_identity, 1.0, 1.0,
                                 ones_t, 0.0, ones_m) == 0
                @test set_plasma(s_gpu._handle.ptr) == 0
                @test set_plasma(s_cpu._handle.ptr) == 0
                @test set_field(s_gpu._handle.ptr, sentinel) == 0
                @test set_field(s_cpu._handle.ptr, sentinel) == 0
                k_gpu = get_ks_stage(s_gpu)
                k_cpu = get_ks_stage(s_cpu)
                @test norm(k_gpu - k_cpu) / max(norm(k_cpu), 1e-30) < 1e-12
                @test maximum(abs.(k_gpu[1:n_spec])) > 1e-20
                @test maximum(abs.(k_gpu[n_spec+1:end])) < 1e-12
            end

            @testset "Failed plasma setup is transactional" begin
                # Restore a fresh physical setup, then reject a null ion
                # handle.  The already-active plasma configuration must remain
                # usable; an initial failed setter likewise leaves Kerr-only.
                s_gpu_invalid = make_gpu()
                s_cpu = make_cpu()
                bad_rc = ccall(
                    (:native_set_plasma_params, libpath), Cint,
                    (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                    s_gpu_invalid._handle.ptr, Ptr{Cvoid}(C_NULL), ionpot,
                    PhysData.e_ratio, plasma.preionfrac, plasma.δt, dens0)
                @test bad_rc != 0
                @test norm(get_ks_stage(s_gpu_invalid) - get_ks_stage(s_cpu)) /
                      norm(get_ks_stage(s_cpu)) < 1e-8
            end

            @testset "Fixed solve and adaptive rejection" begin
                s_cpu = make_cpu()
                s_gpu_fixed = make_gpu()
                solve(s_cpu, L)
                solve(s_gpu_fixed, L)
                rel_fixed = norm(s_gpu_fixed.yn - s_cpu.yn) / norm(s_cpu.yn)
                println("CUDA radial PPT fixed solve rel: ", rel_fixed)
                @test rel_fixed < 1e-6

                Eω_adapt = 1e4 .* Eω
                s_cpu_reject = make_cpu(Eω_adapt, 0.1; max_dt=0.2, min_dt=0.0)
                s_gpu_reject = make_gpu(Eω_adapt, 0.1; max_dt=0.2, min_dt=0.0)
                before = copy(s_gpu_reject.yn)
                @test !step!(s_cpu_reject)
                @test !step!(s_gpu_reject)
                @test s_gpu_reject.yn == before
                @test (isnan(s_gpu_reject.err) && isnan(s_cpu_reject.err)) ||
                      isapprox(s_gpu_reject.err, s_cpu_reject.err; rtol=1e-10)
                @test (isnan(s_gpu_reject.dtn) && isnan(s_cpu_reject.dtn)) ||
                      isapprox(s_gpu_reject.dtn, s_cpu_reject.dtn; rtol=1e-10)
            end

            @testset "Plasma is non-vacuous at strong field" begin
                energy_strong = 3e-4
                inputs_strong = Fields.GaussGaussField(
                    λ0=λ0, τfwhm=τ, energy=energy_strong, w0=w0, propz=-0.1)
                Eω_strong, transform_strong, _ = with_logger(NullLogger()) do
                    Amalthea.setup(grid, q, densityfun, normfun, responses, inputs_strong)
                end
                no_plasma_transform = NonlinearRHS.TransRadial(
                    grid, q, transform_strong.FT,
                    (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                    densityfun, normfun)
                s_on = PreconStepper(transform_strong, linop, copy(Eω_strong), t0, dt;
                                     rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s_off = PreconStepper(no_plasma_transform, linop, copy(Eω_strong), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s_on, L)
                solve(s_off, L)
                plasma_effect = norm(s_on.yn - s_off.yn) / norm(s_off.yn)
                println("Radial PPT plasma-on/off rel: ", plasma_effect)
                @test plasma_effect > 1e-6

                s_gpu_strong = make_gpu(Eω_strong, dt, transform_strong)
                solve(s_gpu_strong, L)
                rel_native = norm(s_gpu_strong.yn - s_on.yn) / norm(s_on.yn)
                println("Radial GPU strong plasma-on/off rel: ",
                        norm(s_gpu_strong.yn - s_off.yn) / norm(s_off.yn),
                        ", native-vs-Julia rel: ", rel_native)
                @test rel_native < 1e-6
                @test norm(s_gpu_strong.yn - s_off.yn) / norm(s_off.yn) > 1e-6
            end
        end
    end
end
