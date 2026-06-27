using TestItems

@testitem "dopri" tags=[:physics] begin
    import Luna: RK45
    import Test: @test

    # Consistency properties of coefficients
    for i in 1:6
        @test isapprox(RK45.nodes[i], sum(RK45.B[i]))
    end

    @test isapprox(sum(RK45.b5), 1.0)
    @test isapprox(sum(RK45.b4), 1.0)
    @test RK45.errest == RK45.b5 .- RK45.b4

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
