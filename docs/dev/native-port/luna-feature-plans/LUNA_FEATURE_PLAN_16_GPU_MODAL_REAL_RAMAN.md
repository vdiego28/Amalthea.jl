# Luna feature plan 16 — CUDA modal RealGrid scalar SDO Raman

Status: depends on plans 02 and 14.

## Outcome

RealGrid modal `npol=1` Kerr + supported `RamanPolarField` runs on CUDA inside
each cubature-node evaluation and agrees with CPU native.

## Implementation

1. Reuse plan 14's device point batches and plan 02's oscillator capacity.
2. Allocate per-node/per-series Raman intensity, ADE polarization, and Hilbert
   scratch. Each cubature node is an independent time series; state must reset
   for every RHS evaluation exactly as CPU native does.
3. Support both THG values: direct `E^2` or the exact batched Hilbert analytic
   signal. Accumulate Raman additively before time window/projection.
4. Prevent scratch reuse races when a callback batch evaluates several nodes.
5. Broaden eligibility only for RealGrid modal `npol=1` supported SDO Raman.
   Keep `npol=2`, EnvGrid Raman, `:SiO2`, plasma, noise, and mixtures excluded.
6. Keep `:auto` false.

## Acceptance

Use multiple cubature nodes with distinct fields and N2 vibrational plus
rotational cases. Assert Julia Raman-on/off non-vacuity, direct point/stage
agreement, fixed-step `<1e-6`, rejected-state preservation/retry, and adaptive
trajectory. Run strict CUDA, CPU modal Raman/threading tests, mode-averaged
Raman regressions, Rust group, and `git diff --check`.

Update docs and append series ownership, oscillator counts, and tolerances to
`PORT_LOG.md`.

## Non-goals

Modal `npol=2` Raman, EnvGrid Raman, intermediate broadening, plasma, or auto
dispatch.
