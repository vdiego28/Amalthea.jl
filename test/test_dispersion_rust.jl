using TestItems

# Julia ↔ Rust Zeisberger dispersion equivalence test.
#
# Tagged :rust so it runs in the `rust` CI group and is automatically skipped
# (not failed) when the Rust shared library has not been built — matching the
# skip-guard pattern used in test_rust_ffi.jl, test_ionisation_rust.jl, and
# test_raman_rust.jl.
@testitem "Rust Zeisberger dispersion equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Luna
    import Luna: Antiresonant, Capillary, Grid, LinearOps
    import Luna.PhysData: wlfreq, c
    import Logging: with_logger, NullLogger

    # ── locate the shared library ──────────────────────────────────────────────
    libname = if Sys.iswindows()
        "luna_rust.dll"
    elseif Sys.isapple()
        "libluna_rust.dylib"
    else
        "libluna_rust.so"
    end
    libpath = joinpath(@__DIR__, "..", "luna-rust", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust dispersion equivalence test: shared library not found at $libpath. " *
              "Build it with `cargo build --release` in luna-rust/ (or run `]build Luna`)."
        return
    end

    # ── Fibre parameters (realistic hollow-core fibre) ────────────────────────
    a         = 150e-6          # core radius [m]
    thickness = 550e-9          # strut wall thickness [m]
    gas       = :He             # fill gas
    pressure  = 1.0             # bar
    λ0        = 800e-9          # centre wavelength [m]

    # Frequency grid (RealGrid over 512 points, 1024 fs window)
    dt   = 1e-15
    Nt   = 512
    grid = Grid.RealGrid(Nt * dt, λ0, (200e-9, 3000e-9), 1e-12)
    sidcs = (1:length(grid.ω))[grid.sidx]

    # ── Helper: build ZeisbergerMode and compute reference neff grid ──────────
    function julia_neff_grid(kind)
        m = Antiresonant.ZeisbergerMode(a, gas, pressure;
                                        kind=kind, n=1, m=1,
                                        wallthickness=thickness,
                                        loss=true)
        neff_ref = Vector{ComplexF64}(undef, length(sidcs))
        for (i, iω) in enumerate(sidcs)
            neff_ref[i] = Antiresonant.neff(m, grid.ω[iω])
        end
        m, neff_ref
    end

    # ── Helper: compute Rust-path neff via neff_β_grid ────────────────────────
    function rust_neff_grid(mode)
        _neff_cl, _β_cl = withenv("LUNA_USE_RUST_DISPERSION" => "1") do
            with_logger(NullLogger()) do
                LinearOps.neff_β_grid(grid, mode, λ0)
            end
        end
        # Evaluate at z=0
        neff_rust = [_neff_cl(iω; z=0.0) for iω in sidcs]
        neff_rust
    end

    # ── HE11 equivalence ──────────────────────────────────────────────────────
    @testset "HE11 — Val{true} loss" begin
        m_he, neff_ref = julia_neff_grid(:HE)
        neff_rust = rust_neff_grid(m_he)

        @test length(neff_rust) == length(neff_ref)
        for i in eachindex(neff_ref)
            nr = neff_ref[i]
            rr = neff_rust[i]
            # Same formula + same nco/ncl inputs → agreement at near machine epsilon
            abs_err = abs(rr - nr)
            ref_mag = abs(nr)
            rel_err = ref_mag > 0 ? abs_err / ref_mag : abs_err
            @test rel_err < 1e-12
        end
    end

    # Note: EH modes are handled by the Zeisberger formula but MarcatiliMode does
    # not accept kind=:EH via its constructor (only :HE, :TE, :TM are valid).
    # The EH branch (kind=1 in Rust, s=+1) is covered by the Rust unit test
    # (test_zeisberger_neff_ffi in lib.rs) where the FFI is called directly.

    # ── TE01 equivalence ──────────────────────────────────────────────────────
    @testset "TE01 — Val{true} loss" begin
        m_te = Antiresonant.ZeisbergerMode(a, gas, pressure;
                                            kind=:TE, n=0, m=1,
                                            wallthickness=thickness, loss=true)
        neff_ref = [Antiresonant.neff(m_te, grid.ω[iω]) for iω in sidcs]
        neff_rust = rust_neff_grid(m_te)
        for i in eachindex(neff_ref)
            nr = neff_ref[i]; rr = neff_rust[i]
            rel_err = abs(nr) > 0 ? abs(rr - nr) / abs(nr) : abs(rr - nr)
            @test rel_err < 1e-12
        end
    end

    # ── TM01 equivalence ──────────────────────────────────────────────────────
    @testset "TM01 — Val{true} loss" begin
        m_tm = Antiresonant.ZeisbergerMode(a, gas, pressure;
                                            kind=:TM, n=0, m=1,
                                            wallthickness=thickness, loss=true)
        neff_ref = [Antiresonant.neff(m_tm, grid.ω[iω]) for iω in sidcs]
        neff_rust = rust_neff_grid(m_tm)
        for i in eachindex(neff_ref)
            nr = neff_ref[i]; rr = neff_rust[i]
            rel_err = abs(nr) > 0 ? abs(rr - nr) / abs(nr) : abs(rr - nr)
            @test rel_err < 1e-12
        end
    end

    # ── Val{false} loss (pure real neff) ──────────────────────────────────────
    @testset "HE11 — Val{false} loss (real neff)" begin
        m_noloss = Antiresonant.ZeisbergerMode(a, gas, pressure;
                                               kind=:HE, n=1, m=1,
                                               wallthickness=thickness, loss=false)
        neff_ref = [Antiresonant.neff(m_noloss, grid.ω[iω]) for iω in sidcs]
        neff_rust = rust_neff_grid(m_noloss)
        for i in eachindex(neff_ref)
            nr = neff_ref[i]; rr = neff_rust[i]
            # With loss=false Julia returns a real (imag=0); Rust also zeroes imag
            @test abs(imag(rr)) < 1e-20
            rel_err = abs(real(nr)) > 0 ? abs(rr - nr) / abs(nr) : abs(rr - nr)
            @test rel_err < 1e-12
        end
    end

    # ── Number loss (scale factor = 0.5) ──────────────────────────────────────
    @testset "HE11 — Number(0.5) loss" begin
        m_half = Antiresonant.ZeisbergerMode(a, gas, pressure;
                                              kind=:HE, n=1, m=1,
                                              wallthickness=thickness, loss=0.5)
        neff_ref = [Antiresonant.neff(m_half, grid.ω[iω]) for iω in sidcs]
        neff_rust = rust_neff_grid(m_half)
        for i in eachindex(neff_ref)
            nr = neff_ref[i]; rr = neff_rust[i]
            rel_err = abs(nr) > 0 ? abs(rr - nr) / abs(nr) : abs(rr - nr)
            @test rel_err < 1e-12
        end
    end

    # ── Toggle off: default (no LUNA_USE_RUST_DISPERSION) returns Julia neff ──
    @testset "Toggle off: neff_β_grid returns Julia values by default" begin
        m_he, neff_ref = julia_neff_grid(:HE)
        # No env toggle → neff_β_grid should use Julia
        _neff_cl, _β_cl = with_logger(NullLogger()) do
            LinearOps.neff_β_grid(grid, m_he, λ0)
        end
        neff_default = [_neff_cl(iω; z=0.0) for iω in sidcs]
        for i in eachindex(neff_ref)
            @test neff_default[i] ≈ neff_ref[i]
        end
    end

    # ── Caching: repeated calls at the same z return identical values ─────────
    @testset "Caching: repeated z=0 queries are idempotent" begin
        m_he, _ = julia_neff_grid(:HE)
        _neff_cl, _ = withenv("LUNA_USE_RUST_DISPERSION" => "1") do
            with_logger(NullLogger()) do
                LinearOps.neff_β_grid(grid, m_he, λ0)
            end
        end
        v1 = [_neff_cl(iω; z=0.0) for iω in sidcs]
        v2 = [_neff_cl(iω; z=0.0) for iω in sidcs]  # second call, same z
        @test v1 == v2
    end

    # ── β closure is consistent with Re(neff) ─────────────────────────────────
    @testset "β closure consistent with Re(neff)" begin
        m_he, _ = julia_neff_grid(:HE)
        _neff_cl, _β_cl = withenv("LUNA_USE_RUST_DISPERSION" => "1") do
            with_logger(NullLogger()) do
                LinearOps.neff_β_grid(grid, m_he, λ0)
            end
        end
        for iω in sidcs[1:10:end]  # spot-check every 10th
            ne = _neff_cl(iω; z=0.0)
            β  = _β_cl(iω; z=0.0)
            @test β ≈ grid.ω[iω] / c * real(ne)
        end
    end
end
