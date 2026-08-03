using TestItems

@testitem "Native-Rust GPU-resident stepper (CUDA, radial RealGrid Kerr)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import LinearAlgebra: I, norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA tests are required, but the Rust library was not found")
        @test_skip "Rust library not found"
    else
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.05; R = 4e-3; N = 32
        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q = Hankel.QDHT(R, N, dim=2)
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        γ3 = PhysData.γ3_gas(gas)
        responses = (Nonlinear.Kerr_field(γ3),)
        linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0,
                                        propz=-0.15)
        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end
        @assert transform isa Amalthea.NonlinearRHS.TransRadial
        t0 = 0.0
        dt = 0.001
        n = length(Eω)
        n_time = length(grid.t)
        n_time_over = length(grid.to)
        n_r = q.N
        T_normal = Matrix{Float64}(q.T)
        M_normal = (grid.ωwin .* (-im .* grid.ω)) ./
                   (2 .* normfun(0.0))
        m_re_normal = real.(M_normal)
        m_im_normal = imag.(M_normal)

        # The Julia oracle's Kerr-on/Kerr-off separation is the required
        # non-vacuity check: a zero RHS must not satisfy the CUDA equivalence
        # tolerance by accident.
        Eω_off, transform_off, FT_off = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun,
                           (Nonlinear.Kerr_field(0.0),), inputs)
        end
        linop_off = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        s_on = PreconStepper(transform, linop, copy(Eω), t0, dt;
                             rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        s_off = PreconStepper(transform_off, linop_off, copy(Eω_off), t0, dt;
                              rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        solve(s_on, L)
        solve(s_off, L)
        nonlinear_share = norm(s_on.yn - s_off.yn) / norm(s_on.yn)
        println("Radial Kerr nonlinear share: ", nonlinear_share)
        @test nonlinear_share > 100 * 1e-12

        local gpu_error
        gpu_available = true
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                "AMALTHEA_NATIVE_GPU" => "on") do
            local s_gpu
            try
                s_gpu = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                          rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end
            @test RK45._native_backend(s_gpu) === :cuda
            @test RK45._gpu_kernel_supports(transform, linop)
            @test RK45._gpu_native_eligible(transform, linop, n)

            getk(s, i) = begin
                k = zeros(ComplexF64, n)
                rc = ccall((:get_ks_stage, libpath), Cint,
                           (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                           s._handle.ptr, Csize_t(i), k, Csize_t(n))
                rc == 0 || error("get_ks_stage failed rc=$rc")
                k
            end

            @testset "Resident radial stage agrees with CPU-native oracle" begin
                s_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                end
                k_cpu = getk(s_cpu, 0)
                k_gpu = getk(s_gpu, 0)
                @test maximum(abs.(k_cpu)) > 100 * 1e-12
                @test norm(k_gpu - k_cpu) / norm(k_cpu) < 1e-12
            end

            @testset "Nonsymmetric QDHT direction and normalization" begin
                # Hankel.QDHT's physical T is symmetric, so use a second raw
                # resident setup with a deliberately nonsymmetric matrix. The
                # CPU and CUDA handles receive the same Julia column-major
                # matrix; agreement catches transpose and scale mistakes in
                # both QDHT directions while the Kerr stage remains nonzero.
                T_probe = Matrix{Float64}(I, n_r, n_r)
                for r in 1:n_r-1
                    T_probe[r, r+1] = 0.03 * (r + 1)
                end
                towin_probe = ones(Float64, n_time_over)
                m_probe = ones(Float64, n)
                set_radial(ptr, T, sf, si, win, kfac, m) = ccall(
                    (:native_set_radial_params, libpath), Cint,
                    (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Ptr{Float64}, Float64,
                     Float64, Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
                    ptr, Csize_t(n_time), Csize_t(n_time_over), Csize_t(n_r), T,
                    sf, si, win, kfac, m, m)
                @test set_radial(s_gpu._handle.ptr, T_probe, 1.7, 0.41,
                                 towin_probe, 1.0, m_probe) == 0
                s_cpu_probe = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                     "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                end
                @test set_radial(s_cpu_probe._handle.ptr, T_probe, 1.7, 0.41,
                                 towin_probe, 1.0, m_probe) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, Eω, Csize_t(n)) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_cpu_probe._handle.ptr, Eω, Csize_t(n)) == 0
                k_gpu = getk(s_gpu, 0)
                k_cpu = getk(s_cpu_probe, 0)
                @test norm(k_gpu - k_cpu) / norm(k_cpu) < 1e-12
            end

            @testset "Invalid replacement leaves active radial setup usable" begin
                # Restore the physical setup after the nonsymmetric primitive
                # probe before testing transactional rejection.
                @test ccall((:native_set_radial_params, libpath), Cint,
                            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Ptr{Float64}, Float64,
                             Float64, Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
                            s_gpu._handle.ptr, Csize_t(n_time), Csize_t(n_time_over),
                            Csize_t(n_r), T_normal, Float64(q.scaleRK),
                            1.0 / Float64(q.scaleRK), grid.towin, dens0 *
                            PhysData.ε_0 * γ3, m_re_normal, m_im_normal) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, Eω, Csize_t(n)) == 0
                @test ccall((:native_set_radial_params, libpath), Cint,
                            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Ptr{Float64}, Float64,
                             Float64, Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
                            s_gpu._handle.ptr, Csize_t(n_time), Csize_t(n_time_over),
                            Csize_t(0), Ptr{Float64}(C_NULL), 1.0, 1.0,
                            Ptr{Float64}(C_NULL), 1.0, Ptr{Float64}(C_NULL),
                            Ptr{Float64}(C_NULL)) != 0
                s_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                end
                @test step!(s_gpu)
                @test step!(s_cpu)
                @test norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn) < 1e-12
            end

            @testset "Fixed solve and adaptive rejection" begin
                s_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                end
                s_gpu_fixed = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=dt, min_dt=dt)
                solve(s_cpu, L)
                solve(s_gpu_fixed, L)
                rel_fixed = norm(s_gpu_fixed.yn - s_cpu.yn) / norm(s_cpu.yn)
                println("CUDA radial fixed solve rel: ", rel_fixed)
                @test rel_fixed < 1e-12
                @test norm(s_gpu_fixed.yn - s_on.yn) / norm(s_on.yn) < 1e-12
                @test nonlinear_share > 100 * 1e-12

                reject_dt = 0.1
                # The production-shaped radial example is intentionally a
                # weakly nonlinear equivalence case, so its embedded error is
                # below the rejection threshold even at a large dt. Scale the
                # initial field only for this controller test; CPU and CUDA
                # receive the same resident state and the physics remains the
                # same Kerr RHS.
                Eω_adapt = 1e4 .* Eω
                s_cpu_reject = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                       "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, Eω_adapt, t0, reject_dt;
                                      rtol=1e-6, atol=1e-10, max_dt=0.2, min_dt=0.0)
                end
                s_gpu_reject = RustNativeStepper(transform, linop, Eω_adapt, t0, reject_dt;
                                                 rtol=1e-6, atol=1e-10,
                                                 max_dt=0.2, min_dt=0.0)
                before = copy(s_gpu_reject.yn)
                @test !step!(s_cpu_reject)
                @test !step!(s_gpu_reject)
                @test s_gpu_reject.yn == before
                @test isapprox(s_gpu_reject.err, s_cpu_reject.err; rtol=1e-10)
                @test isapprox(s_gpu_reject.dtn, s_cpu_reject.dtn; rtol=1e-10)
            end
        end
        if !gpu_available
            require_cuda && error("CUDA tests are required, but GPU setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        end
    end
end
