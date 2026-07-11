# Build PyCall and install matplotlib
import Pkg, Conda, Downloads, SHA, TOML
@info "building!"
Conda.add("matplotlib")
ENV["PYTHON"] = joinpath(Conda.ROOTENV, "bin", "python")
Pkg.build("PyCall")
@info "built PyCall!"

# luna-rust: try a prebuilt binary first (BACKLOG.md S6 item 13), fall back
# to `cargo build --release` from source — the from-source path is kept as
# the canonical dev path and as the fallback for platforms/versions with no
# published release asset.
const _LUNA_RUST_RELEASE_REPO = "vdiego28/Luna-Rust.jl"

_libluna_rust_name() = Sys.iswindows() ? "luna_rust.dll" :
                        Sys.isapple()  ? "libluna_rust.dylib" :
                                         "libluna_rust.so"

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
`luna-rust/target/release/<libname>` path `cargo build --release` would
have produced. Returns `false` (never throws) on any failure — missing
release, unsupported platform, network error, checksum mismatch — so the
caller can fall back to building from source.
"""
function try_download_prebuilt(rust_dir)
    get(ENV, "LUNA_RUST_SKIP_DOWNLOAD", "") == "1" && return false
    triple = _target_triple()
    triple === nothing && return false

    version = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]
    libname = _libluna_rust_name()
    ext = splitext(libname)[2]
    asset = "libluna_rust-$(triple)$(ext)"
    base_url = "https://github.com/$(_LUNA_RUST_RELEASE_REPO)/releases/download/v$(version)"

    dest_dir = joinpath(rust_dir, "target", "release")
    mkpath(dest_dir)
    tmp_lib = joinpath(dest_dir, asset * ".download")
    tmp_sums = joinpath(dest_dir, "SHA256SUMS.txt")
    try
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
        @info "Downloaded prebuilt luna-rust library ($triple, v$version), skipping cargo build."
        return true
    catch e
        @info "No usable prebuilt luna-rust binary (falling back to source build): $e"
        return false
    finally
        isfile(tmp_lib) && rm(tmp_lib; force=true)
        isfile(tmp_sums) && rm(tmp_sums; force=true)
    end
end

rust_dir = joinpath(@__DIR__, "..", "luna-rust")
if isdir(rust_dir)
    if !try_download_prebuilt(rust_dir)
        @info "Building Rust library luna-rust from source..."
        try
            run(addenv(Cmd(`cargo build --release`, dir=rust_dir),
                       "RUSTFLAGS" => get(ENV, "RUSTFLAGS", "")))
            @info "Successfully compiled Rust library luna-rust."
        catch e
            @error "Failed to compile Rust library luna-rust: make sure Rust/Cargo (version >= 1.85) is installed on your system." e
            rethrow(e)
        end
    end
else
    @warn "Rust directory not found at $rust_dir."
end
