using TestItems

@testitem "Grid" tags=[:physics] begin
import Test: @test, @testset, @test_throws
import Luna: Grid, PhysData

@testset "RealGrid" begin
    # Test RealGrid creation
    zmax = 1.0
    refλ = 800e-9
    λlims = (160e-9, 3000e-9)
    trange = 10e-12
    grid = Grid.RealGrid(zmax, refλ, λlims, trange)

    @test grid.zmax == 1.0
    @test grid.referenceλ == 800e-9
    @test length(grid.t) == length(grid.ω) * 2 - 2
    @test length(grid.to) == length(grid.ωo) * 2 - 2

    # Test dictionary serialization
    d = Grid.to_dict(grid)
    grid2 = Grid.from_dict(Grid.RealGrid, d)
    @test grid.zmax == grid2.zmax
    @test grid.referenceλ == grid2.referenceλ
    @test all(grid.t .== grid2.t)
    @test all(grid.ω .== grid2.ω)
    @test all(grid.to .== grid2.to)
    @test all(grid.ωo .== grid2.ωo)
    @test all(grid.sidx .== grid2.sidx)

    # Test validation
    Grid.validate(grid)

    # Validation failure case
    grid_bad = Grid.RealGrid(zmax, refλ, grid.t, grid.ω, grid.to, grid.ωo, grid.sidx, grid.ωwin, grid.twin, grid.towin[1:end-1])
    @test_throws AssertionError Grid.validate(grid_bad)
end

@testset "EnvGrid" begin
    # Test EnvGrid creation
    zmax = 1.0
    refλ = 800e-9
    λlims = (160e-9, 3000e-9)
    trange = 10e-12
    grid = Grid.EnvGrid(zmax, refλ, λlims, trange)

    @test grid.zmax == 1.0
    @test grid.referenceλ == 800e-9
    @test length(grid.t) == length(grid.ω)
    @test length(grid.to) == length(grid.ωo)

    # Test thg=true
    grid_thg = Grid.EnvGrid(zmax, refλ, λlims, trange, thg=true)
    @test length(grid_thg.t) == length(grid_thg.ω)
    @test length(grid_thg.to) == length(grid_thg.ωo)

    # Test dictionary serialization
    d = Grid.to_dict(grid)
    grid2 = Grid.from_dict(Grid.EnvGrid, d)
    @test grid.zmax == grid2.zmax
    @test grid.referenceλ == grid2.referenceλ
    @test grid.ω0 == grid2.ω0
    @test all(grid.t .== grid2.t)
    @test all(grid.ω .== grid2.ω)
    @test all(grid.to .== grid2.to)
    @test all(grid.ωo .== grid2.ωo)
    @test all(grid.sidx .== grid2.sidx)

    # Test validation
    Grid.validate(grid)

    # Validation failure case
    grid_bad = Grid.EnvGrid(zmax, refλ, grid.ω0, grid.t, grid.ω, grid.to, grid.ωo, grid.sidx, grid.ωwin, grid.twin, grid.towin[1:end-1])
    @test_throws AssertionError Grid.validate(grid_bad)
end

@testset "FreeGrid" begin
    # Test FreeGrid creation
    Rx = 1e-3
    Nx = 64
    Ry = 2e-3
    Ny = 32
    grid = Grid.FreeGrid(Rx, Nx, Ry, Ny)

    @test length(grid.x) == Nx
    @test length(grid.y) == Ny
    @test length(grid.kx) == Nx
    @test length(grid.ky) == Ny
    @test size(grid.r) == (1, Ny, Nx)
    @test size(grid.xywin) == (1, Ny, Nx)

    # Test symmetric FreeGrid creation
    grid_sym = Grid.FreeGrid(Rx, Nx)
    @test length(grid_sym.x) == Nx
    @test length(grid_sym.y) == Nx
    @test length(grid_sym.kx) == Nx
    @test length(grid_sym.ky) == Nx
end

end
