## 2026-07-25 — S3 item 0 — Restore GPU nonlinearity (`CudaNativeSim`) — Claude (sonnet-5)
**Status:** complete (nonlinear physics restored and verified on real CUDA hardware; two follow-on items left open, see bottom)

**Hardware note:** both `nvidia-smi` and `/usr/local/cuda-13.3/bin/nvcc` (via
the `/usr/local/cuda` alternatives symlink) worked directly in this agent's
sandbox with no special handling required — contrary to this task's briefing
("commands that talk to the NVIDIA driver... may be blocked"), nothing was
blocked. Real RTX 5060 Ti, driver 610.43.02, CUDA 13.3, confirmed via
`nvidia-smi` and used for every measurement below.

### Diagnosis recap (from BACKLOG.md S3 item 0 / GPU.md §8, not new)
`cuda_native.rs::set_mode_avg_params` discards `_owin`, `_sidx`, `_pre_re`,
`_pre_im`, `_beta`, `_nlscale`, `_sqrt_aeff`, and `step()`'s inline Kerr path
(`rhs_mode_avg_real_kernel` called directly on `ystage_d`/`eto_d`/`pto_d`,
sized `n_time` not `n_time_over`) implements only CPU's Step 3 (Kerr cubic).
Steps 1 (oversampled crop+IFFT), 2 (scale by `1/(nlscale·sqrt_aeff)`), 5
(forward FFT + crop-back), 6 (`norm_pre_beta`), 7 (`owin`) are all missing.
Because Step 2's missing division is by a very large `1/(nlscale·sqrt_aeff)`
factor and the term entering it is cubed (Kerr ~ E³), the resulting Kerr
kernel output is many orders of magnitude too small — consistent with the
measured `max|kᵢ|=3.5e-13` vs CPU's `12225`.

### Second bug found during design review (not in the original BACKLOG diagnosis)
`CudaNativeSim::set_field` (`cuda_native.rs:202`) only does
`field_d.copy_to_device(...)` — it never seeds `ks_d[0]`. CPU's
`CpuNativeSim::set_field` (`native.rs:3679-3717`) explicitly calls
`rhs_mode_avg_real(0, &field)` after copying the field in, precisely so
`ks[0]` holds the true FSAL stage-0 derivative for the *initial* condition
before the first `step()` ever runs (`step()`'s own FSAL-carry logic only
fires when `_t_new > _t_old`, i.e. from the second step onward). On the GPU
path, `ks_d[0]` at the first `step()` call is therefore whatever bytes
`cuMemAlloc` happened to return (not necessarily zeroed), silently
corrupting the very first internal stage (`DP_B[0]=0.2`, nonzero) once the
Kerr pipeline itself is fixed. This was invisible while every stage was
~1e-13 relative to CPU's ~12225 — it will not stay invisible once the Kerr
fix lands. Fix: give `CudaNativeSim` a private `compute_rhs_mode_avg(&mut
self, idx)` helper (the full Step 1/2/3(+3b)/4/5/6/7 pipeline, parametrized
by which `ks_d[idx]` slot to write) and call it with `idx=0` from
`set_field`, exactly mirroring CPU's control flow. `step()`'s per-stage body
(currently inlined) is refactored to call the same helper for `idx=1..6`.

### Plan: CPU-step -> GPU-kernel correspondence (oracle: `native.rs::rhs_mode_avg_real`, native.rs:897-971)

| CPU step (`native.rs`) | GPU kernel (new unless noted) | Buffers |
|---|---|---|
| Step 1: zero-pad `eomega`→`eoo` (scale by `scale_fwd`), inverse rfft → `eto`, ×`fft_norm_over` | **new** `expand_spectrum_kernel` (pad+scale complex, length `n_spec`→`n_spec_over`) then `cufftExecZ2D` (plan sized `n_time_over`, replacing the current `n_time`-sized plan) then **new** `scale_real_kernel` folds the `1/n_time_over` un-normalization *and* Step 2's `1/(nlscale·sqrt_aeff)` into one combined scalar (see below) | `ystage_d`(n)→`eoo_d`(n_spec_over)→`eto_d`(n_time_over) |
| Step 2: `eto *= 1/(nlscale·sqrt_aeff)` | folded into the `scale_real_kernel` call above (one multiply, not two) | `eto_d` |
| Step 2b (shot noise) | **not ported** — GPU scope has no noise (`set_mode_avg_noise[_cplx]` already hard-`-1`; `_gpu_native_eligible` rejects noisy configs before GPU is ever constructed) | n/a |
| Step 3: Kerr `pto = kerr_fac·eto³` | existing `rhs_mode_avg_real_kernel`, **unchanged logic**, now invoked with `n_time_over` (was `n_time`) | `eto_d`→`pto_d` |
| Step 3b: plasma (PPT) | existing `ppt_ionization_kernel`/`plasma_{fraction,phase,current,polarization}_kernel` sequence, unchanged logic, all buffers resized `n_time`→`n_time_over` | `eto_d`,`pto_d`,`plas_*_d` |
| Step 3c: Raman | **not ported** — GPU scope excludes Raman (`set_raman_params[_fft]` already hard `-1`) | n/a |
| Step 4: `pto *= towin` | existing `apply_time_window_kernel`, unchanged logic, `towin_d` now correctly read/sized as `n_time_over` (was buggily read as `n_time` elements from a pointer that Julia/CPU treat as `n_time_over`-long) | `pto_d`,`towin_d` |
| Step 5: forward rfft `pto`→`poo`, crop to `n_spec`, ×`scale_inv` | `cufftExecD2Z` (plan sized `n_time_over`) then **new** `finalize_spectrum_kernel` folds crop+`scale_inv` | `pto_d`→`poo_d`(n_spec_over)→`ks_d[idx]`(n_spec) |
| Step 6: `×norm_pre_beta` | folded into `finalize_spectrum_kernel` (host precomputes `norm_pre_beta = sidx ? pre/beta*sqrt_aeff : 1+0i` once in `set_mode_avg_params`, identical formula/order to CPU, uploaded once) | `norm_pre_beta_d`(n_spec, complex) |
| Step 7: `×owin` | folded into `finalize_spectrum_kernel` (host folds `owin[i]=1.0` outside `sidx` once, identical to CPU) | `owin_d`(n_spec, real) |

### Decisions
1. **`n_time_over` sizing (BACKLOG item 6) is fixed as part of this item**, not
   deferred — it's load-bearing: without it the cuFFT plan and the crop/pad
   kernels have nothing to crop/pad *to*, and per GPU.md §8 the residual
   effect was already suspected to explain some of the Kerr-only test's
   remaining (small) error budget. All Kerr/plasma buffers move from
   `n_time`- to `n_time_over`-sized; `towin_d`'s read length bug (was reading
   `n_time` elements from an `n_time_over`-long buffer) is fixed as part of
   the same change.
2. **`expand_spectrum_kernel`/`finalize_spectrum_kernel` fold multiple CPU
   steps into one kernel launch each** (pad+scale, crop+scale_inv+norm_pre_beta+owin)
   rather than one kernel per CPU step — fewer launches, and each fused step
   is still a straight-line elementwise op with no cross-thread dependency,
   so fusing changes nothing about correctness or floating-point order
   within a step (each output element is produced by exactly one thread from
   its own inputs).
3. **Both `cufftPlan1d` return codes are now checked** in
   `set_mode_avg_params`; a failure destroys any partial plan and returns
   `-1` (previously silently left `self.fft_r2c`/`fft_c2r` as `0`, which
   *did* correctly disable the nonlinear block via the existing `!= 0` guard
   — but silently, with no diagnostic and no distinguishing "plan failed"
   from "not yet configured").
4. **`set_field` now seeds `ks_d[0]`** via the new `compute_rhs_mode_avg`
   helper, matching `CpuNativeSim::set_field`. `resync_field` is
   deliberately left as a copy-only (matches CPU; Phase 8's gotcha applies
   here too — Julia's own stepper doesn't re-evaluate the RHS after
   windowing either).
5. **Scope respected**: no plasma-ADK, no Raman, no radial/modal/free-space —
   only what was already (partially) implemented gets completed.

### What's left open after this item
- GPU CI (BACKLOG item 2) — unaffected by this item, still absent.
- Nothing else in the item's declared scope is left open; see "Next" below
  for genuinely out-of-scope follow-ons noticed along the way.

---

### Implementation — file:line summary

- `amalthea/src/cuda_native.rs`:
  - `CudaNativeSim` struct (line ~9): new fields `n_spec_over`, `scale_fwd`,
    `scale_inv`, `inv_nto_sc`, `nlscale`, `sqrt_aeff`, `norm_pre_beta_d`,
    `owin_d`. `new()` initializes them (placeholder 8/16-byte allocations
    for the device buffers, resized once real dimensions are known).
  - `compute_rhs_mode_avg(&mut self, idx: usize)` (new private method,
    ~180 lines): the full CPU-oracle pipeline (Steps 1-7), reads
    `self.ystage_d`, writes `self.ks_d[idx]`. Step-numbered comments
    cross-reference `native.rs::rhs_mode_avg_real` (native.rs:897-971)
    directly.
  - `NativeBackend::set_field` (was line 202): now also copies `field_d` into
    `ystage_d` and calls `compute_rhs_mode_avg(0)` — seeds `ks_d[0]`,
    mirroring `CpuNativeSim::set_field` (native.rs:3679-3717).
  - `NativeBackend::set_mode_avg_params` (was line 350, ~11 discarded
    `_`-prefixed params): fully rewritten. Uploads `norm_pre_beta_d`
    (`pre/beta*sqrt_aeff`, folded to `1+0i` outside `sidx`) and `owin_d`
    (folded to `1.0` outside `sidx`) — identical formula/fold order to
    `CpuNativeSim::set_mode_avg_params` (native.rs:3952-4046). Resizes
    `eto_d`/`pto_d`/`plas_*_d`/`towin_d` to `n_time_over` (was `n_time`;
    `towin_d`'s *read length* was also wrong — read `n_time` elements from a
    pointer Julia/CPU both treat as `n_time_over`-long). Resizes
    `eoo_d`/`poo_d` to `n_spec_over = n_time_over/2+1` (previously a
    permanent 16-byte placeholder, never resized — dead code). Both
    `cufftPlan1d` return codes are now checked; a failure destroys any
    partial plan and returns a nonzero FFI code instead of silently leaving
    `fft_r2c`/`fft_c2r` at `0`.
  - `NativeBackend::step` (was line 645): the entire inline "Z2D→Kerr[+
    plasma]→window→D2Z, sized `n_time`" block (was ~190 lines,
    lines 1223-1414 pre-edit) is replaced by one call,
    `self.compute_rhs_mode_avg(ii + 1)?;`. The now-unused
    `let cufft = get_cufft_api()?;` at the top of `step()`'s closure was
    removed (cuFFT handles are only touched inside `compute_rhs_mode_avg`
    now).
- `amalthea/src/kernels.cu`: three new kernels, added after
  `apply_time_window_kernel` — `expand_spectrum_kernel` (Step 1 pad+scale),
  `scale_real_kernel` (generic scalar multiply, folds Step 1's cuFFT
  unnormalized-inverse factor with Step 2), `finalize_spectrum_kernel`
  (Step 5 crop+scale_inv fused with Step 6 `norm_pre_beta` and Step 7
  `owin`).
- `amalthea/src/cuda.rs`: `GpuContext` struct gains `expand_spectrum_fn`,
  `scale_real_fn`, `finalize_spectrum_fn`; `init_gpu_context` resolves and
  stores all three via `cuModuleGetFunction`, same pattern as every existing
  kernel symbol.
- `test/test_native_cuda.jl`: both testitems rewritten (see "Tests" below).

### Decisions (as designed, confirmed unchanged during implementation)
1-5 as designed above, all held. One addition made during implementation,
not anticipated in the design section:
6. **`expand_spectrum_kernel`/`finalize_spectrum_kernel` read/write
   `cuDoubleComplex` via direct field access (`.x`/`.y`), not `cuCmul`/
   helper functions** — these are elementwise real/imaginary scale-and-fold
   operations, not complex multiplies-by-a-shared-scalar like
   `apply_prop_kernel`, so writing the real/imaginary algebra out directly
   is clearer than routing through `cuCmul` for a scalar real multiply
   (Step 1/5's `scale_fwd`/`scale_inv`) and marginally clearer for the one
   genuine complex×complex product (Step 6's `norm_pre_beta`).

### Gotchas for the next person
- **The FSAL stage-0 seeding bug (found during design review) was the
  louder of the two bugs in practice.** Once Steps 1/2/5/6/7 are correctly
  wired, `ks_d[0]` before the very first `step()` call is read from
  freshly-`cuMemAlloc`'d device memory if `set_field` doesn't seed it —
  this is *not* guaranteed to be zero on all drivers/allocators, so a build
  that fixes only the RHS pipeline (not `set_field`) could show a
  first-step discrepancy that looks like a residual RHS bug but is actually
  this. Always check `get_ks_stage(0)` immediately after construction,
  before any `step!`, when validating a `NativeBackend` implementation's
  RHS correctness — not just after a step.
- **`n_spec_over` for this Kerr-only test config turned out to equal
  `n_time_over/2+1` where `n_time_over` is twice `n_time`** (standard
  Luna/Amalthea 2x anti-aliasing oversampling) — `scale_fwd`/`scale_inv`
  are therefore comfortably away from `1.0`/degenerate `0/0` forms; nothing
  here was tested at the `n_spec_over == n_spec` (no-oversampling) edge
  case, since no config in this scope constructs one. If a future geometry
  ever does, `scale_fwd = (n_spec_over-1)/(n_spec-1) = 1.0` and
  `expand_spectrum_kernel`'s `idx < n_spec` branch is always taken — no
  special-casing needed, but not empirically exercised here.
- **The `max|kᵢ|` numbers in this entry (~1230) don't match BACKLOG.md S3
  item 0's quoted `12225`** for what is nominally "the same" Kerr-only
  config. Not investigated further — irrelevant to correctness, since the
  proof here is GPU-vs-CPU-native agreement on the *same* run (both
  measured together, ~1e-15 relative), not GPU matching some externally
  quoted absolute number. Possibly the original diagnosis measured a
  different `dt`/step index, or `max` over a different quantity (e.g. after
  more steps, where the field amplitude and thus `kᵢ` would differ). Worth
  a two-minute check if it ever seems load-bearing, but it isn't here.
- **`s_ru.err` is no longer trivially small.** Both CUDA testitems'
  pre-existing `@test s_ru.err < 1.0` assertions were quietly relying on
  the zero-nonlinearity bug (a real RHS makes the placeholder weak-norm
  estimate — `field_d` standing in for both the pre- and post-step field,
  see `weaknorm_elem_args`'s comment in `cuda_native.rs` — legitimately
  large: measured ≈0.93 for Kerr-only, ≈195 for Kerr+plasma). Removed as a
  hard assertion, kept as a `println` diagnostic, with an explanation of
  why `stepcontrol_pi`'s fixed-step (`max_dt=min_dt=dt`) clamp makes this
  harmless (`ok_final` is forced `true` once `dtn` clamps to `min_dt`,
  regardless of `err`).
- Both `cargo build --release` and `cargo test --release` are clean (no
  warnings, 71/71 Rust tests pass, including the real-hardware
  `test_cuda_native_sim_basic`/`test_cuda_native_sim_ffi_gated_by_env_var`/
  `test_gpu_*_numerical_equivalence`/`test_simulation_engine_dispatch`
  tests) both before and after this change.

### Tests run and tolerances achieved

**Standalone probes (before writing/tightening the `@testitem` file),
Kerr-only config (125µm He capillary, 1 bar, 800nm, 1µJ, 30fs, dt=0.01,
flength=0.15, matching `test_native_cuda.jl`'s existing geometry):**
- `get_ks_stage(0)` immediately after construction (before any `step!`):
  CPU-native `max|k1|` = 1230.5720437772707, GPU-native
  1230.5720437772711 — relative difference **1.04e-15**.
- All 7 stages after one `step!`: relative difference **9.5e-16 to
  1.09e-15** per stage; overall `max|kᵢ|` CPU=1230.572..., GPU=1230.572...
  (agrees to the same ~1e-15).
- Nonlinear share (Julia oracle, `kerr=true` vs `kerr=false`, both via
  `PreconStepper`, same fixed-step full solve): **rel_nl = 4.513e-4.**
- Full-solve (fixed step, 16 steps) GPU vs Julia oracle: **rel_solve =
  3.505e-16** — i.e. *tighter* than the reassociation tier (~1e-13)
  TESTING.md §2 says most single-step comparisons should land at, let alone
  the ~1e-6 floor tier a full multi-step solve is normally held to. Margin
  over `rel_nl`: **~1.3e12**.
- Repeated the full-solve measurement a second time (fresh Julia session):
  identical to the last digit (3.505e-16) — this GPU/cuFFT configuration is
  run-to-run deterministic on this hardware for this problem size, unlike
  the ~2e-8 FFTW floor TESTING.md §3 documents for the Julia/CPU path.

**Kerr+plasma config (125µm Ar capillary, 1 bar, 800nm, 6µJ, 15fs, dt=0.005,
flength=0.02, matching `test_native_cuda.jl`'s existing geometry):**
- Nonlinear share (Julia oracle, Kerr+PPT-plasma on vs off): **rel_nl =
  2.005e-2.**
- Full-solve (fixed step, 5 steps) GPU vs Julia oracle: **rel_solve =
  1.806e-16.** Margin over `rel_nl`: **~1.1e14**.

**`test/test_native_cuda.jl` (rewritten — both testitems), run via
`AMALTHEA_USE_RUST_CUDA_NATIVE`/`AMALTHEA_NATIVE_GPU=on` on the real RTX
5060 Ti:** 33/33 assertions pass. Tolerances asserted (chosen to pin the
reassociation tier TESTING.md §2 says most single-deterministic-step
comparisons should land at — `1e-12`, not the initially-drafted `1e-9`;
see "Advisor review" below — while still sitting far below the measured
`rel_nl`, per the "prove the feature changes the oracle by more than the
tolerance" rule):
- Kerr-only: stage-derivative check `< 1e-12` (measured ~1.0e-15, both
  immediately after construction/before any `step!`, and after one `step!`
  across all 7 stages); full-solve (fixed step) `< 1e-12` (measured
  3.5e-16); `rel_nl > 1e-4` asserted (measured 4.5e-4).
- Kerr+plasma: full-solve (fixed step) `< 1e-12` (measured 1.8e-16);
  `rel_nl > 1e-3` asserted (measured 2.0e-2).
- Both: non-vacuousness magnitude checks (`max|kᵢ| > 100`) so a
  regression to near-zero nonlinearity fails immediately, independent of
  the relative-error comparisons.
- **New: `Luna.run`/dense-output equivalence (adaptive stepping,
  `prop_capillary` with `saveN=11`), Kerr-only config** — added after an
  advisor review flagged that every check above drives the stepper via raw
  `solve()`/`step!()`, the exact blind spot that hid the Phase 8 windowing
  bug and the dense-output order bug (CLAUDE.md's Phase 8 gotchas). Before
  this item's fix, `interpolate(s::RustNativeStepper, ti)`'s dense-output
  correction term (built from `get_ks_stage`) was multiplying `k` values of
  ~3.5e-13 — below FP noise, so its formula's correctness was untestable in
  practice on this backend. Now that `k`~1e3, this is the first real check
  of `interpolate`'s *value*, not just that it completes. Measured on real
  hardware: final save rel diff **1.25e-7**; the 9 intermediate saves
  (which actually route through `interpolate`, unlike the final save) rel
  diff **1.8e-10 to 7.7e-8**. All asserted `< 1e-6` (the FFT-method+floor
  tier — adaptive stepping means the two integrators are not pinned to an
  identical step sequence the way the fixed-step tests above are, so this
  is not held to the fixed-step tests' `1e-12`). **Dense output on
  `CudaNativeSim` is now genuinely verified, not just "completes without
  throwing."**
- **`err` diagnostic comparison (GPU placeholder vs Julia oracle's real
  pre-acceptance trial), also added after advisor review** — printed, not
  asserted (see "Gotchas" below for why `err` doesn't gate the fixed-step
  trajectory either way). Measured: Kerr-only `s_ru.err=0.934`,
  `s_jl.err=1.43e-4` (~6500x apart); Kerr+plasma `s_ru.err=195.4`,
  `s_jl.err=1.82` (~107x apart, and notably `s_jl.err` itself exceeds 1 for
  this config/step — `PreconStepper`'s own fixed-step acceptance isn't
  gated on `err<1` either). The GPU placeholder (`field_d` for both the
  "old" and "trial new" field in `weaknorm_elem_kernel`, since `step()` has
  no real pre-acceptance trial solution) is measurably inflating `err`
  relative to what a true trial-based estimate would give, worse for
  Kerr+plasma than Kerr-only — consistent with plasma's steeper field-
  amplitude sensitivity. This does not affect `rel_solve` (the accepted
  trajectory) under fixed-step, only `dtn`/whether a step would be
  rejected under *adaptive* stepping, which the raw-stepper tests here
  don't exercise (the new `Luna.run` test above does use adaptive
  stepping and passed, so this inflation is not, in practice, causing
  spurious rejections at this config either — but see "Next" for the
  precise boundary of what that does and doesn't prove).

**Full `rust` group** (`python3 test/parallel_rust_tests.py`, 55 files,
10 load-balanced parallel workers), run twice — once after the initial
implementation (23 new assertions in `test_native_cuda.jl`), once more
after the advisor-review tightening/additions (33 new assertions):
- First run: **42237/42238**, every worker rc=0, 158.3s wall-clock.
- Final run (current state of the diff): **42247/42248**, every worker
  rc=0, 152.7s wall-clock. The +10 in both numerator and denominator
  exactly matches `test_native_cuda.jl` growing from 23 to 33 assertions.

The 1-assertion gap in both runs is the pre-existing, documented
"pool-channel" many-processes-vs-one-process residual-count effect
(`docs/dev/native-port/VANILLA_LUNA_ISSUES.md` §"Concurrent-process races
..." / `CLAUDE.md`'s "residual-count caveat"), not a new failure — no
worker reported a failed assertion in either run, only a slightly
different total assertion *count* than a single shared process would
produce, which is the documented, harmless topology effect, unrelated to
anything in this change.

### Advisor review (second pass, after the first "complete" draft)
An independent review of this item before it was marked done flagged two
real gaps in what had been verified, both addressed above rather than
deferred:
1. **Every test to that point drove the stepper via raw `solve()`/
   `step!()`, never `Luna.run`/`prop_capillary`** — exactly the blind spot
   CLAUDE.md's Phase 8 gotcha and `cuda_native.rs::apply_prop`'s own
   comment name as having hidden a real bug before (the windowing bug, and
   separately the dense-output order bug). Since this item is precisely
   what makes `interpolate`'s dense-output correction term nonzero for the
   first time on this backend, that path had never actually been checked
   at the *value* level. Fixed by adding the "Luna.run / dense-output
   equivalence" testset described above — genuinely new coverage, not a
   restatement of the fixed-step checks.
2. **The tolerances drafted initially (`1e-9` stage/Kerr-only,`1e-6`
   Kerr+plasma) were 3-9 orders of magnitude looser than what was actually
   measured (~1e-15/1e-16)** — a direct instance of the TESTING.md §2 rule
   this whole item exists to enforce ("pick the tightest tier the math
   justifies... a test that passes at a looser tier than its math allows is
   hiding a bug"). Tightened to `1e-12` throughout (still >1000x above the
   measured floor, leaving room for a different GPU/driver/cuFFT version to
   diverge slightly without a false failure, while pinning the
   reassociation tier rather than leaving a gap wide enough to hide a
   partial regression).

Both were re-measured and re-verified on the real RTX 5060 Ti after the
fix (see "Tests run" above) — 33/33 assertions pass with the tightened
values.

### Next
- Nothing further required for this item's stated scope. If the `max|kᵢ|`
  discrepancy against BACKLOG's quoted `12225` ever becomes load-bearing
  for some other measurement, it's worth two minutes to check whether that
  number was measured at a different `dt` or after more accepted steps.
- GPU CI (BACKLOG item 2) remains the standing prerequisite for trusting
  any future GPU change without a manual hardware run like this one.
- The `err` placeholder's measured inflation (6500x for Kerr-only, 107x for
  Kerr+plasma, versus the Julia oracle's real trial-based estimate) is a
  known, documented limitation (not new — `weaknorm_elem_args`'s comment in
  `cuda_native.rs` predates this item), and the new `Luna.run` adaptive-
  stepping test above did pass, so it is not observed to cause a spurious
  step rejection at either config tested here. It has **not** been proven
  harmless in general for adaptive stepping at a config/tolerance where a
  step would genuinely need to be rejected — that would require a
  dedicated test constructing such a case, which is outside this item's
  scope (implementing a real pre-acceptance trial solution for `step()` is
  the actual fix, noted in GPU.md's own `weaknorm_elem_args` comment as an
  "adaptive-step correctness concern, out of scope here" before this item
  and still true after it).
