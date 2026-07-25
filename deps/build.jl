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

# The repo was renamed `luna_rust` -> `amalthea` mid-development, but GitHub
# release v1.0.0 was published *before* `.github/workflows/release.yml` was
# updated to stage the new `libamalthea-<triple>` names, so its three binaries
# (and SHA256SUMS.txt manifest entries) are still named `libluna_rust-<triple>`.
# v1.0.0 is the *only* release with this problem — release.yml stages the
# canonical `libamalthea-*` names for every tag from here on — so the legacy
# fallback below is bounded to versions at or before it, not open-ended: an
# unbounded fallback would let a genuinely broken *future* release (e.g. a
# release.yml regression that silently reverts the asset name, or an asset
# upload that failed) masquerade as a working legacy-named install rather
# than surfacing as a download miss. See
# docs/dev/native-port/portlog-inbox/prebuilt-asset-compat.md for the record.
const _LAST_LEGACY_NAMED_VERSION = v"1.0.0"
const _LEGACY_LIBNAME_PREFIX = "libluna_rust"

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
    _prebuilt_asset_candidates(triple, ext, version) -> Vector{String}

Asset names to try, in priority order: the canonical `libamalthea-<triple>`
name always first (so a correctly-named future release takes the fast path
without ever touching the legacy branch), then — only for versions at or
before `_LAST_LEGACY_NAMED_VERSION` — the legacy `libluna_rust-<triple>` name
that release v1.0.0 actually published under.
"""
function _prebuilt_asset_candidates(triple, ext, version::VersionNumber)
    candidates = String["libamalthea-$(triple)$(ext)"]
    if version <= _LAST_LEGACY_NAMED_VERSION
        push!(candidates, "$(_LEGACY_LIBNAME_PREFIX)-$(triple)$(ext)")
    end
    candidates
end

"""
    try_download_prebuilt(rust_dir; base_url=nothing) -> Bool

Download the release asset matching this package's version (`Project.toml`)
and the running platform's target triple, verify it against the release's
`SHA256SUMS.txt` manifest, and place it at the same
`amalthea/target/release/<libname>` path `cargo build --release` would
have produced. Tries the canonical `libamalthea-<triple>` asset name first,
falling back to the legacy pre-rename `libluna_rust-<triple>` name (bounded
to `_LAST_LEGACY_NAMED_VERSION`, see its docstring/comment above) if the
canonical name isn't in the manifest. Returns `false` (never throws) on any
failure — missing release, unsupported platform, network error, checksum
mismatch — so the caller can fall back to building from source.

`base_url` overrides the GitHub releases URL (production default computed
from `_AMALTHEA_RELEASE_REPO` + the package version); this exists solely so
tests can point the function at a local HTTP server serving a fake release
layout — the production call site never passes it.
"""
function try_download_prebuilt(rust_dir; base_url::Union{Nothing,AbstractString}=nothing)
    get(ENV, "AMALTHEA_RUST_SKIP_DOWNLOAD", "") == "1" && return false
    triple = _target_triple()
    triple === nothing && return false

    version_str = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]
    version = VersionNumber(version_str)
    libname = _libamalthea_name()
    ext = splitext(libname)[2]
    resolved_base_url = base_url === nothing ?
        "https://github.com/$(_AMALTHEA_RELEASE_REPO)/releases/download/v$(version_str)" :
        String(base_url)
    candidates = _prebuilt_asset_candidates(triple, ext, version)

    dest_dir = joinpath(rust_dir, "target", "release")
    mkpath(dest_dir)
    try
        return mktempdir() do tmp_dir
            tmp_sums = joinpath(tmp_dir, "SHA256SUMS.txt")
            Downloads.download("$resolved_base_url/SHA256SUMS.txt", tmp_sums)
            sums_lines = collect(eachline(tmp_sums))

            for asset in candidates
                expected = nothing
                for line in sums_lines
                    parts = split(line)
                    if length(parts) == 2 && parts[2] == asset
                        expected = parts[1]
                        break
                    end
                end
                if expected === nothing
                    @info "No checksum entry for $asset in SHA256SUMS.txt; trying next candidate."
                    continue
                end

                tmp_lib = joinpath(tmp_dir, asset * ".download")
                try
                    Downloads.download("$resolved_base_url/$asset", tmp_lib)
                catch e
                    @info "Could not download prebuilt asset $asset: $e"
                    continue
                end

                actual = bytes2hex(open(SHA.sha256, tmp_lib))
                if actual != expected
                    # A checksum mismatch means the manifest *does* list this
                    # asset but the downloaded bytes don't match it — a
                    # tamper/corruption signal, not "this release doesn't have
                    # that name." Unlike the "not in manifest" and "download
                    # failed" cases above, don't treat this as "try the next
                    # candidate": if the canonical asset is present but
                    # corrupt, silently trying the legacy name next could mask
                    # a broken release behind an unrelated fallback. Fail the
                    # whole attempt so the caller falls back to source.
                    @info "Checksum mismatch for $asset (expected $expected, got $actual); " *
                          "not installing, falling back to source build."
                    return false
                end

                mv(tmp_lib, joinpath(dest_dir, libname); force=true)
                @info "Downloaded prebuilt amalthea library ($asset, v$version_str), skipping cargo build."
                return true
            end
            return false
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
            # No prebuilt binary was available for this platform/package-version (see
            # try_download_prebuilt above), so we fell back to `cargo build --release`
            # from source — which needs an actual Rust toolchain. Give an actionable
            # message instead of letting a raw cargo/process error surface, since a
            # missing `cargo` on PATH is the most common cause here (docs/dev/BACKLOG.md
            # "Distribution & example-code maintenance" item 1).
            no_cargo = e isa Base.IOError
            @error """
            Failed to $(no_cargo ? "find" : "compile") the Rust library `amalthea` from source.

            Amalthea.jl offloads its numerical kernels to a native Rust backend. A
            prebuilt binary is downloaded automatically for common platforms
            (Linux x86_64, macOS aarch64, Windows x86_64); this system either isn't
            one of those, has no matching release asset for this package version, or
            the download was skipped/failed, so `Pkg.build` fell back to compiling
            from source — which requires a working Rust toolchain (cargo >= 1.85).

            To fix this: install Rust via https://rustup.rs/ (or your system package
            manager), make sure `cargo` is on your PATH in a NEW shell/Julia session,
            then re-run `Pkg.build("Amalthea")`.

            If you expected the prebuilt-binary download to work instead, check that
            the `AMALTHEA_RUST_SKIP_DOWNLOAD` environment variable is not set to "1"
            and that your network can reach github.com.
            """ exception=e
            rethrow(e)
        end
    end
else
    @warn "Rust directory not found at $rust_dir."
end
