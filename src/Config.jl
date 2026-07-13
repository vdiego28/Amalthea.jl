module Config

"""
    BackendConfig

Snapshot of every `AMALTHEA_USE_RUST_*`/`AMALTHEA_*` Rust-backend toggle, resolved
from `ENV` by [`backend_config`](@ref). docs/dev/BACKLOG.md S4 item 1 (suggestion 6):
before this, each toggle was read by its own `get(ENV, "AMALTHEA_USE_RUST_...",
"default") == "1"` one-liner, duplicated at its call site with no single
place to see the whole toggle surface — and, found while consolidating
them, two of those one-liners actually used *different* truthiness checks
for the same env var (`Ionisation.jl`'s `AMALTHEA_USE_RUST_NATIVE` read used
`!= "0"` while `RK45.jl`'s used `== "1"`; a value like `"true"` would have
been read as ON in one file and OFF in the other — a real, if narrow,
inconsistency this centralizes away by construction).

Fields (each `Bool`, default `false` unless noted):
- `native`: `AMALTHEA_USE_RUST_NATIVE`, default `true` — the native-port
  resident stepper (Phase 8 default-flip).
- `cuda_native`: `AMALTHEA_USE_RUST_CUDA_NATIVE` — GPU-resident stepper opt-in.
- `stepper`: `AMALTHEA_USE_RUST_STEPPER` — the older per-kernel callback stepper.
- `ionisation`: `AMALTHEA_USE_RUST_IONISATION` — PPT ionization LUT kernel.
- `raman`: `AMALTHEA_USE_RUST_RAMAN` — time-domain Raman SDO ADE kernel.
- `dispersion`: `AMALTHEA_USE_RUST_DISPERSION` — Zeisberger/Marcatili neff/β.
- `qdht`: `AMALTHEA_USE_RUST_QDHT` — batch QDHT transform kernel.
- `qdht_blas`: `AMALTHEA_QDHT_BLAS` — BLAS-3 dgemm path for QDHT (opt-in on top
  of `qdht`; see docs/dev/BACKLOG.md S1 item 5).
- `native_wisdom`: `AMALTHEA_NATIVE_FFTW_WISDOM` — on-disk FFTW planner-wisdom
  persistence for the native path (default off; see
  `docs/dev/native-port/PLAN_FFTW_WISDOM_FIX.md`).
- `deterministic`: `AMALTHEA_NATIVE_DETERMINISTIC` — docs/dev/BACKLOG.md S5.2. Forces
  both Rust-side QDHT handles — the native-port radial-geometry one
  (`RK45.jl`'s `native_set_deterministic`) and the older per-kernel
  `AMALTHEA_USE_RUST_QDHT` one (`NonlinearRHS._make_rust_qdht_handle`) — to
  skip the BLAS-3 `dgemm` path (`qdht_blas`) even when it's enabled, using
  the row-parallel Rayon fallback instead. Wiring both matters because
  `BLAS_API` is a process-global `OnceLock` on the Rust side: only the
  per-kernel path ever populates it, but once populated it silently makes
  every later native-path QDHT call in that process BLAS-eligible too —
  see `native_set_deterministic`'s doc in `RK45.jl` for what this does and
  does not guarantee.
"""
Base.@kwdef struct BackendConfig
    native::Bool = true
    cuda_native::Bool = false
    stepper::Bool = false
    ionisation::Bool = false
    raman::Bool = false
    dispersion::Bool = false
    qdht::Bool = false
    qdht_blas::Bool = false
    native_wisdom::Bool = false
    deterministic::Bool = false
end

_truthy(var, default) = get(ENV, var, default) == "1"

"""
    backend_config() -> BackendConfig

Resolve the current `BackendConfig` from `ENV`. **Deliberately re-reads
`ENV` on every call rather than caching** — the test suite pervasively
flips these toggles mid-session with `withenv` (e.g.
`test/test_native_fftw_wisdom.jl`, `test/test_qdht_rust.jl`'s "toggle-off"
case), and a cached/one-time-read singleton would make `backend_config()`
return stale values inside those tests. This is the one place any new
`AMALTHEA_*` backend toggle should be added, so its default and truthiness
convention are visible in one spot instead of scattered across call sites.
"""
function backend_config()
    BackendConfig(
        native       = _truthy("AMALTHEA_USE_RUST_NATIVE", "1"),
        cuda_native  = _truthy("AMALTHEA_USE_RUST_CUDA_NATIVE", "0"),
        stepper      = _truthy("AMALTHEA_USE_RUST_STEPPER", "0"),
        ionisation   = _truthy("AMALTHEA_USE_RUST_IONISATION", "0"),
        raman        = _truthy("AMALTHEA_USE_RUST_RAMAN", "0"),
        dispersion   = _truthy("AMALTHEA_USE_RUST_DISPERSION", "0"),
        qdht         = _truthy("AMALTHEA_USE_RUST_QDHT", "0"),
        qdht_blas    = _truthy("AMALTHEA_QDHT_BLAS", "0"),
        native_wisdom= _truthy("AMALTHEA_NATIVE_FFTW_WISDOM", "0"),
        deterministic= _truthy("AMALTHEA_NATIVE_DETERMINISTIC", "0"),
    )
end

end # module
