# Luna feature plan 10 — CUDA radial RealGrid PPT plasma

Status: complete (2026-08-02; depends on plan 08).

## Outcome

RealGrid radial Kerr + one PPT `PlasmaCumtrapz` runs resident on CUDA and
matches CPU native across every radial column.

## Design

Generalize the completed mode-averaged PPT pipeline into a segmented scan:
each radial column is an independent time series. Block offsets and finalizers
must never cross column boundaries. Reuse the uploaded PPT spline format and
existing field/current formulas.

## Implementation

1. Size plasma rate, fraction, current, polarization, scan scratch, and block
   sums for `n_time_over*n_r` during radial/plasma setup.
2. Extend scan kernels with a series/segment dimension or launch one bounded
   independent scan per column. Preserve deterministic summation order within
   each series.
3. Apply rates and all three cumtrapz-derived finalizers to the same radial
   time-domain field used by Kerr; accumulate plasma additively before the
   time window.
4. Make setter order and rollback explicit: radial setup precedes plasma; a
   failed plasma setup leaves a valid Kerr-only radial backend.
5. Broaden eligibility only for RealGrid radial, one Kerr plus one PPT plasma,
   scalar density, constant linop/norm, no Raman/noise/mixture.
6. Keep `:auto` false until radial-specific performance data exists.

## Tests and acceptance

- Rust/CUDA segmented-scan primitive with at least two columns, more than one
  block per column, and a partial final block.
- Non-vacuous Julia plasma-on/off control and comparable nonzero stage scale.
- Direct stage, fixed-step `<1e-6`, reject/retry field preservation, and
  adaptive trajectory against CPU native.
- A column-isolation sentinel proving no prefix state leaks between columns.
- Strict CUDA, CPU radial plasma, FFI lifecycle, Rust group, and diff check.

Update docs and append `PORT_LOG.md` with scan layout and measured tolerances.

## Non-goals

ADK, EnvGrid plasma, Raman, radial auto threshold, or scan reassociation for
the existing mode-averaged path.

## Completion (2026-08-02)

**Status:** complete.

The CUDA implementation uses a flat column-major radial layout
`i = column*n_time_over + t`. `plasma_scan_radial_blocks_kernel` performs a
deterministic 256-thread Blelloch scan for every `(column, block)` and writes a
block total. The fraction, current, and polarization finalizers sum preceding
block totals only within the same column; therefore columns are independent,
more-than-one-block columns and a partial final block are handled without an
offset or prefix leak. The recurrence is the CPU contract
`q[0]=0; q[t]=q[t-1]+0.5*(x[t-1]+x[t])*dt`.

The radial RHS in `amalthea/src/cuda_native.rs` evaluates PPT rates and phase
on the post-QDHT RealGrid field (`radial_qdht_d`), then applies the three
segmented scans and accumulates `density*P` into `radial_pto_d` before the
existing time window. Using the pre-QDHT `radial_eto_d` would silently omit
the radial transform; the first hardware diagnostic caught and corrected that
mistake. Plasma setup sizes all flattened radial scratch and block totals,
stages the spline and device allocations in locals, and commits them only as
one valid replacement, so a failed setter preserves radial Kerr-only state.
For the plasma extension, the Julia predicate admits only RealGrid radial
scalar density, constant linop/norm, one plain Kerr, and one
`IonRatePPTAccel`; the completed Plan 09 EnvGrid radial scalar-Kerr predicate
remains admitted separately. Explicit `AMALTHEA_NATIVE_GPU=on` is required
for both radial CUDA slices and radial `:auto` remains false.

The focused strict CUDA item passed **27/27** on an RTX 5060 Ti (CUDA 13.3,
driver 610.43.02): direct stage relative error
`1.5647312256418479e-15`, fixed-solve error `4.756600300395168e-16`, CUDA
strong plasma-on/off effect `1.7924786820029344e-5`, Julia control effect
`1.7924786820007026e-5`, and strong native-vs-Julia error
`5.848007396073851e-16`. The item includes non-vacuity controls, a
multi-block/partial-column isolation sentinel, transactional null/invalid
rollback, fixed solve, and adaptive rejection/retry. `cargo build --release`
and the strict CUDA Rust suite passed; CPU radial plasma coverage passed
6/6, including the native-vs-Julia strong-field check. EnvGrid plasma, ADK,
Raman, noise, mixtures, radial automatic dispatch, and mode-averaged scan
changes are not part of this plan.
