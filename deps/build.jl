# Build PyCall and install matplotlib
import Pkg, Conda
@info "building!"
Conda.add("matplotlib")
ENV["PYTHON"] = joinpath(Conda.ROOTENV, "bin", "python")
Pkg.build("PyCall")
@info "built PyCall!"

# Build the Rust shared library luna-rust
@info "Building Rust library luna-rust..."
rust_dir = joinpath(@__DIR__, "..", "luna-rust")
if isdir(rust_dir)
    try
        run(Cmd(`cargo build --release`, dir=rust_dir,
                env=Dict("CARGO_BUILD_RUSTFLAGS" => get(ENV, "CARGO_BUILD_RUSTFLAGS", ""))))
        @info "Successfully compiled Rust library luna-rust."
    catch e
        @error "Failed to compile Rust library luna-rust: make sure Rust/Cargo (version >= 1.85) is installed on your system." e
        rethrow(e)
    end
else
    @warn "Rust directory not found at $rust_dir."
end
