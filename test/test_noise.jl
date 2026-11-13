using TestItems

@testitem "Noise" tags=[:fields] begin
using Amalthea
import Amalthea: Interface, NonlinearRHS, Fields, Grid
import Test: @test, @testset, @test_throws
import Random: MersenneTwister
import Logging: with_logger, NullLogger

args = (100e-6, 0.01, :He, 1)
kwargs = (; λ0=800e-9, τfwhm=10e-15, energy=1e-12,
           trange=400e-15, λlims=(200e-9, 4e-6), plasma=false, raman=false)

@testset "generate_noise_field" begin
    @testset "RealGrid" begin
        grid = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nf = Fields.generate_noise_field(grid)
        @test size(nf) == size(grid.ω)
        @test eltype(nf) == ComplexF64
        @test any(!iszero, nf)
        # Different seeds produce different noise
        rng1 = MersenneTwister(42)
        rng2 = MersenneTwister(123)
        nf1 = Fields.generate_noise_field(grid; rng=rng1)
        nf2 = Fields.generate_noise_field(grid; rng=rng2)
        @test nf1 != nf2
        # Same seed reproduces identical noise
        rng3 = MersenneTwister(42)
        nf3 = Fields.generate_noise_field(grid; rng=rng3)
        @test nf1 == nf3
    end
    @testset "EnvGrid" begin
        grid = Grid.EnvGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nf = Fields.generate_noise_field(grid)
        @test size(nf) == size(grid.ω)
        @test eltype(nf) == ComplexF64
        # Non-signal bins should be zero
        @test all(iszero, nf[.!grid.sidx])
        @test any(!iszero, nf[grid.sidx])
    end
    @testset "Amplitude matches ShotNoise (one-photon-per-mode)" begin
        # One-photon-per-mode amplitude: |A_noise(ω)| = √(hν·Δν)
        # generate_noise_field and ShotNoise use the same formula, so their
        # amplitudes must be identical (phases differ only by RNG seed).
        grid_r = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        sn = Fields.ShotNoise(MersenneTwister(42))
        snf = sn(grid_r)
        nf = Fields.generate_noise_field(grid_r; rng=MersenneTwister(42))
        @test abs.(nf) ≈ abs.(snf)

        grid_e = Grid.EnvGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        sn_e = Fields.ShotNoise(MersenneTwister(99))
        snf_e = sn_e(grid_e)
        nf_e = Fields.generate_noise_field(grid_e; rng=MersenneTwister(99))
        @test abs.(nf_e) ≈ abs.(snf_e)
    end
    @testset "Multimode RealGrid" begin
        grid = Grid.RealGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nm = 3
        nf = Fields.generate_noise_field(grid; nmodes=nm)
        # Shape: (nω, nmodes)
        @test size(nf) == (length(grid.ω), nm)
        @test eltype(nf) == ComplexF64
        # All columns non-zero
        @test all(any(!iszero, nf[:, j]) for j in 1:nm)
        # Columns are independent (different random phases)
        @test nf[:, 1] != nf[:, 2]
        @test nf[:, 2] != nf[:, 3]
        # Amplitude identical across modes (same one-photon-per-mode formula)
        @test abs.(nf[:, 1]) ≈ abs.(nf[:, 2])
        @test abs.(nf[:, 2]) ≈ abs.(nf[:, 3])
        # Reproducible with same seed
        rng1 = MersenneTwister(42)
        rng2 = MersenneTwister(42)
        @test Fields.generate_noise_field(grid; rng=rng1, nmodes=nm) ==
              Fields.generate_noise_field(grid; rng=rng2, nmodes=nm)
        # nmodes=1 still returns 1D array (no regression)
        nf1 = Fields.generate_noise_field(grid; nmodes=1)
        @test ndims(nf1) == 1
    end
    @testset "Multimode EnvGrid" begin
        grid = Grid.EnvGrid(0.1, 800e-9, (200e-9, 4e-6), 400e-15)
        nm = 3
        nf = Fields.generate_noise_field(grid; nmodes=nm)
        @test size(nf) == (length(grid.ω), nm)
        @test eltype(nf) == ComplexF64
        # Non-signal bins should be zero in all modes
        for j in 1:nm
            @test all(iszero, nf[.!grid.sidx, j])
            @test any(!iszero, nf[grid.sidx, j])
        end
        # Columns are independent
        @test nf[:, 1] != nf[:, 2]
        # Amplitude identical across modes
        @test abs.(nf[:, 1]) ≈ abs.(nf[:, 2])
    end
end

@testset "Seeded input noise, envelope=$envelope modes=$modes" for envelope in (false, true), modes in (:HE11, 2)
    setup(noise, seed) = with_logger(NullLogger()) do
        Interface.prop_capillary_args(args...; kwargs..., envelope, modes,
            shotnoise=noise, rng=MersenneTwister(seed))
    end
    noisy = setup(true, 42)
    explicit = setup(:input, 42)
    another = setup(true, 43)
    clean = setup(false, 42)
    clean2 = setup(false, 43)
    @test noisy[1] == explicit[1]
    @test noisy[1] != another[1]
    @test noisy[1] != clean[1]
    @test clean[1] == clean2[1]
    noise_buffer = modes == 2 ? :Emω_noise : :Et_noise
    @test isnothing(getproperty(noisy[4], noise_buffer))
    @test isnothing(getproperty(clean[4], noise_buffer))
    if modes == 2
        delta = noisy[1] - clean[1]
        @test any(!iszero, delta[:, 1])
        @test any(!iszero, delta[:, 2])
        @test delta[:, 1] != delta[:, 2]
    end
    # Guard the low-level path too: a nonlinear-only source must not be usable.
    grid, FT = noisy[2], noisy[5]
    @test_throws ArgumentError NonlinearRHS.TransModeAvg(
        grid, FT, (), z -> 1.0, identity, z -> 1.0; noise_field=zero(grid.ω))
end

@testset "Withdrawn and invalid noise models" begin
    for noise in (:modified, :invalid)
        @test_throws DomainError Interface.prop_capillary_args(args...;
            kwargs..., shotnoise=noise)
    end
end

@testset "GNLSE defaults use seeded input noise" begin
    setup(noise, seed) = with_logger(NullLogger()) do
        Interface.prop_gnlse_args(1e-3, 0.01, [0.0, 0.0, -20e-27];
            λ0=800e-9, τfwhm=10e-15, energy=1e-12,
            trange=400e-15, λlims=(200e-9, 4e-6), raman=false,
            shotnoise=noise, rng=MersenneTwister(seed))
    end
    noisy = setup(true, 42)
    @test noisy[1] == setup(:input, 42)[1]
    @test noisy[1] != setup(false, 42)[1]
    @test noisy[1] != setup(true, 43)[1]
    @test isnothing(noisy[4].Et_noise)
end
end
