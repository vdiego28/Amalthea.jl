import Test: @test, @test_throws, @testset
import Logging: @info

testdir = dirname(@__FILE__)

import Luna: set_fftw_mode, set_fftw_threads
set_fftw_mode(:estimate)

# On Windows, FFTW's internal thread pool is unstable when Julia uses many threads
# simultaneously, leading to EXCEPTION_ACCESS_VIOLATION crashes in libfftw3-3.dll.
# Restrict FFTW to a single thread to avoid this.
if Sys.iswindows()
    set_fftw_threads(1)
end

group = get(ENV, "LUNA_TEST_GROUP", "All")
@info "Running test group: $group"

@testset "All" begin

if group == "All" || group == "physics"
    @testset "Maths" begin
        @info("================= test_maths.jl")
        include(joinpath(testdir, "test_maths.jl"))
    end

    @testset "PhysData" begin
        @info("================= test_physdata.jl")
        include(joinpath(testdir, "test_physdata.jl"))
        @info("================= test_dynamic_materials.jl")
        include(joinpath(testdir, "test_dynamic_materials.jl"))
    end

    @testset "Capillary" begin
        @info("================= test_capillary.jl")
        include(joinpath(testdir, "test_capillary.jl"))
    end

    @testset "Rectangular Modes" begin
        @info("================= test_rect_modes.jl")
        include(joinpath(testdir, "test_rect_modes.jl"))
    end

    @testset "ODE Solver" begin
        @info("================= test_rk45.jl")
        include(joinpath(testdir, "test_rk45.jl"))
    end

    @testset "Ionisation" begin
        @info("================= test_ionisation.jl")
        include(joinpath(testdir, "test_ionisation.jl"))
    end

    @testset "LinearOps" begin
        @info("================= test_linops.jl")
        include(joinpath(testdir, "test_linops.jl"))
    end

    @testset "Modes" begin
        @info("================= test_modes.jl")
        include(joinpath(testdir, "test_modes.jl"))
    end

    @testset "Tapers" begin
        @info("================= test_tapers.jl")
        include(joinpath(testdir, "test_tapers.jl"))
    end

    @testset "Antiresonant modes" begin
        @info("================= test_antiresonant.jl")
        include(joinpath(testdir, "test_antiresonant.jl"))
    end
end

# sim-interface: the heaviest group — prop_capillary interface tests (~56 simulations)
if group == "All" || group == "simulation" || group == "sim-interface"
    @testset "Interface" begin
        @info("================= test_interface.jl")
        include(joinpath(testdir, "test_interface.jl"))
        @info("================= test_greek_aliases.jl")
        include(joinpath(testdir, "test_greek_aliases.jl"))
    end
end

# sim-multimode: multi-mode and polarisation propagation tests
if group == "All" || group == "simulation" || group == "sim-multimode"
    @testset "Multimode" begin
        @info("================= test_multimode.jl")
        include(joinpath(testdir, "test_multimode.jl"))
    end

    @testset "Polarisation" begin
        @info("================= test_polarisation.jl")
        include(joinpath(testdir, "test_polarisation.jl"))
        @info("================= test_polarisation_field.jl")
        include(joinpath(testdir, "test_polarisation_field.jl"))
        @info("================= test_polarisation_env.jl")
        include(joinpath(testdir, "test_polarisation_env.jl"))
    end
end

# sim-propagation: lighter propagation tests (radial, freespace, GNLSE, Kerr, Raman)
if group == "All" || group == "simulation" || group == "sim-propagation"
    @testset "Radial Propagation" begin
        @info("================= test_radial.jl")
        include(joinpath(testdir, "test_radial.jl"))
    end

    @testset "Full 3D Propagation" begin
        @info("================= test_full_freespace.jl")
        include(joinpath(testdir, "test_full_freespace.jl"))
    end

    @testset "Linear propagation" begin
        @info("================= test_linearprop.jl")
        include(joinpath(testdir, "test_linearprop.jl"))
    end

    @testset "GNLSE interface" begin
        @info("================= test_gnlse.jl")
        include(joinpath(testdir, "test_gnlse.jl"))
    end

    @testset "Kerr" begin
        @info("================= test_kerr.jl")
        include(joinpath(testdir, "test_kerr.jl"))
    end

    @testset "Raman" begin
        @info("================= test_raman.jl")
        include(joinpath(testdir, "test_raman.jl"))
    end
end

if group == "All" || group == "io"
    @testset "Output" begin
        @info("================= test_output.jl")
        include(joinpath(testdir, "test_output.jl"))
    end

    @testset "Scans" begin
        @info("================= test_scans.jl")
        include(joinpath(testdir, "test_scans.jl"))
    end

    @testset "Statistics" begin
        @info("================= test_stats.jl")
        include(joinpath(testdir, "test_stats.jl"))
    end

    @testset "Gas mixtures" begin
        @info("================= test_mixtures.jl")
        include(joinpath(testdir, "test_mixtures.jl"))
    end

    @testset "Vector plasma" begin
        @info("================= test_vectorplasma.jl")
        include(joinpath(testdir, "test_vectorplasma.jl"))
    end
end

if group == "All" || group == "fields"
    @testset "Fields" begin
        @info("================= test_fields.jl")
        include(joinpath(testdir, "test_fields.jl"))
    end

    @testset "Noise model" begin
        @info("================= test_noise.jl")
        include(joinpath(testdir, "test_noise.jl"))
    end

    @testset "Processing" begin
        @info("================= test_processing.jl")
        include(joinpath(testdir, "test_processing.jl"))
    end

    @testset "Tools" begin
        @info("================= test_tools.jl")
        include(joinpath(testdir, "test_tools.jl"))
    end

    @testset "Utils" begin
        @info("================= test_utils.jl")
        include(joinpath(testdir, "test_utils.jl"))
    end

    @testset "Gradients" begin
        @info("================= test_gradient.jl")
        include(joinpath(testdir, "test_gradient.jl"))
    end
end

if group == "All" || group == "rust"
    @testset "Rust FFI Integration" begin
        @info("================= Rust FFI tests")
        rust_test_dir = joinpath(dirname(testdir), "luna-rust", "tests")
        include(joinpath(rust_test_dir, "test_julia_ffi.jl"))
        include(joinpath(rust_test_dir, "test_stepper_dispatch.jl"))
        include(joinpath(rust_test_dir, "test_scans_io.jl"))
    end
end

end