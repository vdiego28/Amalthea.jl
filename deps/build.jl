import Pkg, Downloads, SHA, TOML
@info "building!"

# Plotting (PyPlot/PyCall) is a package extension (see [weakdeps]/[extensions] in
# Project.toml) — it is intentionally NOT built here. Building PyCall eagerly would
# force-install a Conda-bundled libpython even for users who never plot, and that
# bundled libpython crashes if Amalthea is ever loaded from Python via
# juliacall/PythonCall (two different libpython runtimes in one process). Julia
# users who want plotting should `Pkg.add("PyPlot")` in their own environment and
# `using PyPlot` before/alongside `using Amalthea`; PyPlot's own build step handles
# PyCall/matplotlib installation.

# amalthea: try a prebuilt binary first (docs/dev/BACKLOG.md S6 item 13), fall back
# to `cargo build --release` from source — the from-source path is kept as
# the canonical dev path and as the fallback for platforms/versions with no
# published release asset.
const _AMALTHEA_RELEASE_REPO = "vdiego28/Amalthea.jl"

_libamalthea_name() = Sys.iswindows() ? "amalthea.dll" :
                        Sys.isapple()  ? "libamalthea.dylib" :
                                         "libamalthea.so"

# Only the three triples actually built by .github/workflows/release.yml
# (matching run_tests.yml's CI matrix hosts) — anything else falls back to
# source. Not a general cross-compilation target list.
_target_triple() = Sys.iswindows() ? "x86_64-pc-windows-msvc" :
                    Sys.isapple()  ? "aarch64-apple-darwin" :
                    Sys.islinux() && Sys.ARCH === :x86_64 ? "x86_64-unknown-linux-gnu" :
                                     nothing

"""
    try_download_prebuilt(rust_dir) -> Bool

Download the release asset matching this package's version (`Project.toml`)
and the running platform's target triple, verify it against the release's
`SHA256SUMS.txt` manifest, and place it at the same
`amalthea/target/release/<libname>` path `cargo build --release` would
have produced. Returns `false` (never throws) on any failure — missing
release, unsupported platform, network error, checksum mismatch — so the
caller can fall back to building from source.
"""
function try_download_prebuilt(rust_dir)
    get(ENV, "AMALTHEA_RUST_SKIP_DOWNLOAD", "") == "1" && return false
    triple = _target_triple()
    triple === nothing && return false

    version = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]
    libname = _libamalthea_name()
    ext = splitext(libname)[2]
    asset = "libamalthea-$(triple)$(ext)"
    base_url = "https://github.com/$(_AMALTHEA_RELEASE_REPO)/releases/download/v$(version)"

    dest_dir = joinpath(rust_dir, "target", "release")
    mkpath(dest_dir)
    try
        return mktempdir() do tmp_dir
            tmp_lib = joinpath(tmp_dir, asset * ".download")
            tmp_sums = joinpath(tmp_dir, "SHA256SUMS.txt")

            Downloads.download("$base_url/$asset", tmp_lib)
            Downloads.download("$base_url/SHA256SUMS.txt", tmp_sums)

            expected = nothing
            for line in eachline(tmp_sums)
                parts = split(line)
                if length(parts) == 2 && parts[2] == asset
                    expected = parts[1]
                    break
                end
            end
            expected === nothing && error("no checksum entry for $asset in SHA256SUMS.txt")

            actual = bytes2hex(open(SHA.sha256, tmp_lib))
            actual == expected || error("checksum mismatch for $asset " *
                                         "(expected $expected, got $actual)")

            mv(tmp_lib, joinpath(dest_dir, libname); force=true)
            @info "Downloaded prebuilt amalthea library ($triple, v$version), skipping cargo build."
            true
        end
    catch e
        @info "No usable prebuilt amalthea binary (falling back to source build): $e"
        return false
    end
end

rust_dir = joinpath(@__DIR__, "..", "amalthea")
if isdir(rust_dir)
    if !try_download_prebuilt(rust_dir)
        @info "Building Rust library amalthea from source..."
        try
            run(addenv(Cmd(`cargo build --release`, dir=rust_dir),
                       "RUSTFLAGS" => get(ENV, "RUSTFLAGS", "")))
            @info "Successfully compiled Rust library amalthea."
        catch e
            @error "Failed to compile Rust library amalthea: make sure Rust/Cargo (version >= 1.85) is installed on your system." e
            rethrow(e)
        end
    end
else
    @warn "Rust directory not found at $rust_dir."
end
