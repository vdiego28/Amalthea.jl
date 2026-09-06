using TestItems

@testitem "dopri" tags=[:physics] begin
    import Amalthea: RK45
    import Test: @test, @testset

    # Consistency properties of coefficients
    for i in 1:6
        @test isapprox(RK45.nodes[i], sum(RK45.B[i]))
    end

    @test isapprox(sum(RK45.b5), 1.0)
    @test isapprox(sum(RK45.b4), 1.0)
    @test RK45.errest == RK45.b4 .- RK45.b5

    @testset "DP5(4) propagated and embedded weights" begin
        # Keep this oracle independent of the Float64 tableau in the solver.
        c = Rational{Int}[0, 1//5, 3//10, 4//5, 8//9, 1, 1]
        b5 = Rational{Int}[35//384, 0, 500//1113, 125//192,
                            -2187//6784, 11//84, 0]
        b4 = Rational{Int}[5179//57600, 0, 7571//16695, 393//640,
                            -92097//339200, 187//2100, 1//40]

        @test RK45.b5 == Float64.(b5)
        @test RK45.b4 == Float64.(b4)
        @test RK45.errest ≈ Float64.(b4 .- b5) rtol=0 atol=2eps()
        @test RK45.b5[1:6] == RK45.B[6]
        @test RK45.b5[7] == 0
        @test vec(sum(RK45.interpC, dims=1)) ≈ RK45.b5 rtol=0 atol=8eps()

        # A Runge--Kutta rule of order p integrates c^q exactly through q=p-1.
        for q in 0:4
            @test sum(b5 .* c.^q) == 1//(q + 1)
        end
        for q in 0:3
            @test sum(b4 .* c.^q) == 1//(q + 1)
        end
        @test sum(b4 .* c.^4) != 1//5
    end

    @testset "propagated order, FSAL, and false-mode trial" begin
        f_exp! = (out, y, _) -> (out .= y)

        function fixed_exp(dt; locextrap)
            s = RK45.Stepper(f_exp!, [1.0], 0.0, dt;
                              rtol=1e6, atol=0.0, min_dt=dt, max_dt=dt,
                              locextrap=locextrap)
            for _ in 1:round(Int, 1 / dt)
                @test RK45.step!(s)
            end
            return abs(s.yn[1] - exp(1)), s
        end

        e5_h, s5 = fixed_exp(1/8; locextrap=true)
        e5_h2, _ = fixed_exp(1/16; locextrap=true)
        e4_h, s4 = fixed_exp(1/8; locextrap=false)
        e4_h2, _ = fixed_exp(1/16; locextrap=false)
        @test 4.7 < log2(e5_h / e5_h2) < 5.3
        @test 3.7 < log2(e4_h / e4_h2) < 4.3

        # The final stage is FSAL only for the propagated fifth-order state.
        endpoint_rhs = similar(s5.yn)
        f_exp!(endpoint_rhs, s5.yn, s5.tn)
        @test s5.ks[end] ≈ endpoint_rhs rtol=1e-13 atol=1e-14

        # The free quartic extension must reach the same DP5 endpoint at σ=1.
        b_endpoint = vec(sum(RK45.interpC, dims=1))
        quartic_endpoint = copy(s5.y)
        for i in 1:7
            quartic_endpoint .+= s5.dt * b_endpoint[i] .* s5.ks[i]
        end
        @test quartic_endpoint ≈ s5.yn rtol=1e-13 atol=1e-14

        # With a zero linear operator, the preconditioned formulation must
        # choose the same explicitly formed embedded fourth-order candidate.
        plain4 = RK45.Stepper(f_exp!, [1.0], 0.0, 1/8;
                                rtol=1e6, atol=0.0, min_dt=1/8, max_dt=1/8,
                                locextrap=false)
        @test RK45.step!(plain4)
        p4 = RK45.PreconStepper(f_exp!, zeros(1), [1.0], 0.0, 1/8;
                                 rtol=1e6, atol=0.0, min_dt=1/8, max_dt=1/8,
                                 locextrap=false)
        @test RK45.step!(p4)
        @test p4.yn ≈ plain4.yn rtol=1e-13 atol=1e-14
    end

    @testset "Fourth-order endpoint, restart, and dense convergence" begin
        # Independent nonlinear solution: y' = a*y + y², y(0)=1.
        # Nonzero a exposes accidental re-propagation of a freshly computed k1.
        for precon in (false, true), a in (0.0, -0.7)
            rhs! = (out, y, _) -> (out .= y.^2)
            total! = (out, y, _) -> (out .= a.*y .+ y.^2)
            exact(t) = a == 0 ? inv(1-t) : exp(a*t)/(1-expm1(a*t)/a)
            function make(y, t, h; locextrap=false, rtol=1e6, fixed=true)
                kw = (; locextrap, rtol, atol=0.0,
                       min_dt=fixed ? h : 0.0, max_dt=h)
                precon ? RK45.PreconStepper(rhs!, [a], copy(y), t, h; kw...) :
                         RK45.Stepper(total!, copy(y), t, h; kw...)
            end

            s = make([1.0], 0.0, 0.25)
            @test RK45.step!(s)
            # Reconstruct y5 independently to prove this fixture would expose
            # interpolation ending at the wrong candidate.
            y5 = s.y + s.dt * sum(RK45.b5[i]*s.ks[i] for i in 1:7)
            precon && (y5 .*= exp(a*s.dt))
            @test abs(y5[1]-s.yn[1]) > 1e-10
            stages = deepcopy(s.ks)
            for _ in 1:2
                @test RK45.interpolate(s, prevfloat(s.tn)) ≈ s.yn rtol=1e-13
                @test RK45.interpolate(s, s.t + 1e-9*s.dt) ≈ s.y rtol=1e-8
                RK45.interpolate(s, s.t + 0.37*s.dt)
            end
            @test s.ks == stages
            endpoint = copy(s.yn)
            fresh = make(endpoint, s.tn, s.dtn)
            @test RK45.step!(s)
            @test RK45.step!(fresh)
            @test s.yn ≈ fresh.yn rtol=1e-13
            expected_k1 = precon ? endpoint.^2 : a.*endpoint .+ endpoint.^2
            @test s.ks[1] ≈ expected_k1 rtol=1e-13

            # Rejection after accepted history, followed by controller retries.
            s = make([1.0], 0.0, 0.125)
            @test RK45.step!(s)
            endpoint = copy(s.yn)
            s.rtol = 1e-12; s.min_dt = 0; s.max_dt = 0.25; s.dtn = 0.25
            fresh = make(endpoint, s.tn, 0.25; rtol=1e-12, fixed=false)
            fresh.errlast = s.errlast
            @test !RK45.step!(s)
            @test !RK45.step!(fresh)
            @test s.yn == fresh.yn == endpoint
            for _ in 1:20
                @test RK45.step!(s) == RK45.step!(fresh)
                @test s.yn ≈ fresh.yn rtol=1e-13
                s.ok && break
            end
            @test s.ok

            function errors(h; locextrap)
                s = make([1.0], 0.0, h; locextrap)
                dense_error = 0.0
                for _ in 1:round(Int, 0.5/h)
                    @test RK45.step!(s)
                    ti = s.t + 0.37*s.dt
                    dense_error = max(dense_error,
                        abs(RK45.interpolate(s, ti)[1]-exact(ti)))
                end
                abs(s.yn[1]-exact(0.5)), dense_error
            end
            e4, d4 = errors(1/32; locextrap=false)
            e4h, d4h = errors(1/64; locextrap=false)
            @test 3.6 < log2(e4/e4h) < 4.4
            @test 3.6 < log2(d4/d4h) < 4.4
            # The exponential test above supplies the fifth-order control.
            # For this Riccati problem, fifth-order error terms cancel near
            # these step sizes, so their ratio is not a stable order estimate.
        end
    end

    # Analytic tests
    # 1. Exponential decay: y' = -y => y(t) = y(0)*exp(-t)
    f_decay! = function(out, y, t)
        @. out = -y
    end

    y0_decay = [1.0]
    t0 = 0.0
    dt = 0.1
    tmax = 2.0

    tout, yout, steps = RK45.solve(f_decay!, y0_decay, t0, dt, tmax, output=true, outputN=21, rtol=1e-10, atol=1e-12)

    for i in eachindex(tout)
        @test isapprox(yout[1, i], exp(-tout[i]), atol=1e-2, rtol=1e-2)
    end

    # 2. Simple harmonic oscillator: y'' = -y
    # Let y1 = y, y2 = y'
    # y1' = y2
    # y2' = -y1
    f_sho! = function(out, y, t)
        out[1] = y[2]
        out[2] = -y[1]
    end

    y0_sho = [1.0, 0.0] # cos(t), -sin(t)
    t0_sho = 0.0
    dt_sho = 0.1
    tmax_sho = 2*pi

    tout_sho, yout_sho, steps_sho = RK45.solve(f_sho!, y0_sho, t0_sho, dt_sho, tmax_sho, output=true, outputN=101, rtol=1e-10, atol=1e-12)

    for i in eachindex(tout_sho)
        @test isapprox(yout_sho[1, i], cos(tout_sho[i]), atol=1e-2, rtol=1e-2)
        @test isapprox(yout_sho[2, i], -sin(tout_sho[i]), atol=1e-2, rtol=1e-2)
    end
end
