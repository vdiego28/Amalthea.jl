# AGENTS.md — working agreement for agents on Luna-Rust.jl

This file lets any agent pick up work on the **native-Rust backend port** when
the lead is unavailable, and lets the lead resume afterward. Read it fully before
touching code.

> **Human reader:** `CLAUDE.md` is the project guide (build, architecture, test
> groups). This file is the *agent workflow* on top of it. The native port has
> its own docs under [`docs/native-port/`](docs/native-port/).

## 0. The one rule that is never overridden
- **Never add a `Co-Authored-By` trailer to commits.** The lead does not want it.
- **Never commit or push unless the lead explicitly asks.** Branch first if asked
  to commit while on `main`.
- **Do not modify source while documentation for a phase is unfinished.** The
  port is doc-driven: a phase's design must be written down before its code.

## 1. Orientation — read these, in order
1. This file.
2. [`docs/native-port/ARCHITECTURE.md`](docs/native-port/ARCHITECTURE.md) — why
   the current backend is *not* "exclusively Rust" and the target `NativeSim`
   design + decisions.
3. [`BACKLOG.md`](BACKLOG.md) → the **"Native-Rust backend port (phased)"**
   section — the phase checklist and which phase is next.
4. [`docs/native-port/MATH.md`](docs/native-port/MATH.md) — the loop-level math
   your phase must reproduce.
5. [`docs/native-port/TESTING.md`](docs/native-port/TESTING.md) — the tolerance
   tier your phase must hit and how to write the test.
6. [`docs/native-port/PORT_LOG.md`](docs/native-port/PORT_LOG.md) — read the
   **latest** entries: what the previous agent actually did, and any gotcha.

## 2. The established migration pattern (reuse it, don't reinvent)
Every Rust offload in this repo follows one pattern (see `CLAUDE.md` "Kernel
wiring pattern" for the canonical description):
1. **Rust (`luna-rust/src/ffi.rs`):** export `#[unsafe(no_mangle)] pub unsafe
   extern "C"` lifecycle functions — `init_*` → `*mut Handle`, `free_*`, and the
   work function — using the opaque-handle pattern.
2. **Julia (`src/*.jl`):** a `mutable struct Rust*Handle` wrapping `Ptr{Cvoid}`
   with a GC finalizer calling `free_*`; the parent struct stores it as a type
   parameter (`Nothing` = Julia path, handle = Rust path); an env toggle
   (`LUNA_USE_RUST_*`) gates construction, **Julia path is the default**.
3. **Test (`test/test_*_rust.jl`):** `@testitem tags=[:rust]` with the skip-guard
   from `test/test_stepper_rust.jl`.

For the native port specifically, the new toggle is **`LUNA_USE_RUST_NATIVE`**
and the new handle is **`NativeSim`** (ARCHITECTURE §3). The integration point is
`RK45.solve_precon` (`src/RK45.jl:19`), branched once per `solve`.

### Patterns that have already bitten us (do not repeat)
- `@cfunction` pointers as module-level `const` → **segfault** (stale precompile
  image). Use `Ref{Ptr{Cvoid}}` populated in `__init__`.
- `Threads.@threads` on `TransModal`'s integration loop → **data race**. Keep
  sequential.
- Recompute QDHT `T` from the Rust convention → silent normalization mismatch.
  Pass **Julia's** `T` in.
- Test against an installed package copy of the `.so` → `undefined symbol` for
  new exports. Use `luna-rust/target/release/libluna_rust.so`.

## 3. How to execute a phase
1. Confirm the phase's design is fully written in the docs above. If a design
   detail is missing, **add it to the docs first**, then code.
2. Build: `cd luna-rust && cargo build --release`.
3. Implement Rust side, then Julia wiring, then the test (per §2).
4. Validate against the Julia oracle at the tier in TESTING.md §4 — **two**
   tests: a single-step (~1e-13) check that proves the math, and a full-`solve`
   (~1e-6 floor) check that proves the trajectory.
5. Run `LUNA_TEST_GROUP=rust julia --project test/runtests.jl` and confirm no
   regression in the pre-existing groups you might affect.
6. **Append a `PORT_LOG.md` entry** (§4). This is mandatory, not optional.
7. Do **not** commit unless the lead asked. Leave `git status` clean of anything
   but the intended changes.

## 4. Work-log requirement (mandatory)
On finishing any unit of work — a phase, a partial phase, or a blocked attempt —
**append an entry to [`docs/native-port/PORT_LOG.md`](docs/native-port/PORT_LOG.md)**
using the template at the top of that file. The lead resumes from this log plus
`git log`, so it must state: what you did, how (with `file:line` and FFI symbols),
every decision and its reason, gotchas for the next person, the exact tests run
and tolerances achieved, and the immediate next step. An undocumented change is
treated as not done.

## 5. Conventions
- **Unicode math vars** in Julia (`ω`, `λ`, `β`, `τ`, `ε`): enter with
  `\omega<tab>`, etc. Match the surrounding code.
- Build the Rust lib with `cargo build --release` in `luna-rust/`; the package
  build (`deps/build.jl`) forces `RUSTFLAGS=""` for portability — respect it.
- Keep the FFI safety-net tests green: `test/test_rust_ffi.jl` and
  `luna-rust/tests/*.jl`.
- Prefer reusing existing Rust modules (`stepper.rs`, `diffraction.rs`,
  `integrator.rs`, `raman.rs`) over new code — several are written but not yet
  FFI-exported (ARCHITECTURE / BACKLOG note them).

## 6. How the lead resumes
Read the latest `PORT_LOG.md` entries (newest at the bottom), then `git log` and
`git status`. The "Next" field of the last entry is the resume point. Any phase
marked `blocked` states its blocker in the entry.
