@testitem "Native scan-point HDF5 writer" tags=[:rust] begin
    using Amalthea
    import Amalthea: Output
    import HDF5

    # Skip if the native library isn't built (mirrors test_rust_ffi.jl).
    libname = if Sys.iswindows()
        "amalthea.dll"
    elseif Sys.isapple()
        "libamalthea.dylib"
    else
        "libamalthea.so"
    end
    libpath = joinpath(@__DIR__, "..", "amalthea", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping native scan-writer test: shared library not found at $libpath."
        return
    end

    mktempdir() do dir
        # A genuinely multi-dimensional, non-square field array so the
        # column-major ⇄ HDF5 row-major dim-reversal convention is actually
        # exercised (a square array would hide a transpose bug).
        nω, nmodes, nz = 5, 3, 4
        y = ComplexF64[complex(iω + 10 * imode + 100 * iz, -iω - 3 * imode)
                       for iω in 1:nω, imode in 1:nmodes, iz in 1:nz]
        z = collect(range(0.0, 1.0; length=nz))

        @testset "round-trips through HDF5.jl untransposed" begin
            fpath = joinpath(dir, "point_0001.h5")
            Output.write_scan_point_native(fpath, y, z)
            @test isfile(fpath)

            HDF5.h5open(fpath, "r") do file
                # Structure matches HDF5Output's skeleton.
                @test HDF5.haskey(file, "Eω")
                @test HDF5.haskey(file, "z")
                @test HDF5.haskey(file, "stats")
                @test HDF5.haskey(file, "meta")

                yr = read(file["Eω"])
                zr = read(file["z"])
                @test size(yr) == size(y)          # no transpose / dim-reversal leak
                @test yr == y                      # bit-identical complex payload
                @test zr == z
            end
        end

        @testset "readable via HDF5Output getindex" begin
            fpath = joinpath(dir, "point_0002.h5")
            Output.write_scan_point_native(fpath, y, z; yname="Eω", tname="z")
            # Opened with cache=false: the native writer produces a *finished*
            # result, not a resumable one, so it intentionally omits the
            # meta/cache group HDF5Output's default constructor uses only to
            # continue an interrupted run. Data-access getindex needs no cache.
            o = Output.HDF5Output(fpath, 0, 0, 1; readonly=true, cache=false)
            @test o["Eω"] == y
            @test o["z"] == z
        end

        @testset "custom dataset names" begin
            fpath = joinpath(dir, "point_named.h5")
            Output.write_scan_point_native(fpath, y, z; yname="field", tname="dist")
            HDF5.h5open(fpath, "r") do file
                @test HDF5.haskey(file, "field")
                @test HDF5.haskey(file, "dist")
                @test read(file["field"]) == y
            end
        end

        @testset "write under scan-queue file lock" begin
            fpath = joinpath(dir, "point_locked.h5")
            lockpath = joinpath(dir, "scan_lock")
            Output.write_scan_point_native(fpath, y, z; lockpath=lockpath)
            @test isfile(fpath)
            @test isfile(lockpath)              # FlockLock created the lock file
            HDF5.h5open(fpath, "r") do file
                @test read(file["Eω"]) == y
            end
        end

        @testset "dimension-mismatch errors cleanly" begin
            fpath = joinpath(dir, "point_bad.h5")
            badz = collect(1.0:(nz + 1))        # z length ≠ last field dim
            @test_throws ErrorException Output.write_scan_point_native(fpath, y, badz)
        end
    end
end
