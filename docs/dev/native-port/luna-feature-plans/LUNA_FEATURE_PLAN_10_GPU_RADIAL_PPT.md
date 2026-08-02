# Luna feature plan 10 — CUDA radial RealGrid PPT plasma

Status: depends on plan 08.

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
