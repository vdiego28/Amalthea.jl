using TestItems

@testitem "Julia TransModal callback threading and safe fallback" tags=[:sim_multimode] begin
    import Test: @test, @testset, @test_skip
    using Amalthea
    import Amalthea: Capillary, Fields, Grid, Nonlinear, NonlinearRHS, PhysData
    import Logging: NullLogger, with_logger

    if Threads.nthreads() < 4
        @test_skip "requires JULIA_NUM_THREADS=4"
    else

    Amalthea.set_fftw_mode(:estimate)
    a = 40e-6
    gas = :Ar
    pressure = 1.0
    λ0 = 800e-9
    grid = Grid.RealGrid(1e-3, λ0, (400e-9, 2000e-9), 0.2e-12)
    modes = (Capillary.MarcatiliMode(a, gas, pressure; m=1),
             Capillary.MarcatiliMode(a, gas, pressure; m=2))
    density = PhysData.density(gas, pressure)
    densityfun(z) = density
    input = Fields.GaussField(λ0=λ0, τfwhm=25e-15, energy=2e-7)
    kerr = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

    Eω, transform, _ = withenv("AMALTHEA_USE_RUST_NATIVE" => "0") do
        with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, kerr, input, modes, :y; mfcn=128)
        end
    end
    @test transform.modal_threaded
    @test length(transform.modal_scratch) == Threads.nthreads()

    xs = reshape(collect(range(0.08a, 0.92a; length=17)), 1, :)
    fdim = 2length(Eω)
    sequential = zeros(fdim, size(xs, 2))
    threaded = similar(sequential)
    NonlinearRHS.reset!(transform, Eω, 0.0)
    transform.modal_threaded = false
    NonlinearRHS.pointcalc!(sequential, xs, transform)
    seq_calls = transform.ncalls
    NonlinearRHS.reset!(transform, Eω, 0.0)
    transform.modal_threaded = true
    NonlinearRHS.pointcalc!(threaded, xs, transform)
    @test threaded == sequential
    @test transform.ncalls == seq_calls == size(xs, 2)

    @testset "forced GC preserves exact callback output" begin
        for _ in 1:4
            GC.gc()
            out = similar(threaded)
            NonlinearRHS.reset!(transform, Eω, 0.0)
            NonlinearRHS.pointcalc!(out, xs, transform)
            @test out == sequential
        end
    end

    @testset "stateful user closure remains sequential" begin
        calls = Ref(0)
        stateful = let calls=calls
            (out, E, ρ) -> begin
                calls[] += 1
                fill!(out, 0)
            end
        end
        _, unsafe_transform, _ = with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, (stateful,), input, modes, :y; mfcn=32)
        end
        @test !unsafe_transform.modal_threaded
        @test isempty(unsafe_transform.modal_scratch)
    end

    @testset "Julia-only plasma response gets independent scratch" begin
        ratefunc = (rate, E) -> fill!(rate, 0.0)
        plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ratefunc, 1.0)
        _, plasma_transform, _ = withenv("AMALTHEA_USE_RUST_NATIVE" => "0") do
            with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, (kerr[1], plasma), input, modes, :y; mfcn=32)
            end
        end
        @test plasma_transform.modal_threaded
        @test all(s -> s.resp[2] !== plasma, plasma_transform.modal_scratch)
    end
    end # thread-count guard (top-level return does not skip later expressions)
end
