# Luna feature plan 07 — CUDA mode-averaged EnvGrid `:SiO2` Raman

Status: complete (2026-08-02); standing CUDA CI remains strongly preferred.

## Outcome

The resident CUDA mode-averaged EnvGrid path supports
`RamanRespIntermediateBroadening`/`:SiO2` through the existing
`native_set_raman_fft_params` contract, with no per-stage host field transfer.

## Design

Mirror `CpuNativeSim::set_raman_fft_params` and the r2c-halved native
FFT-convolution kernel. Store the transferred oscillator arrays and response
spectrum in device memory, own transactional convolution plans/scratch, form
the real envelope intensity, convolve, and add `pto += E*(density*P)` before
the shared window/forward FFT. Reuse current FFT conventions; do not restore
the rejected short-kernel optimization.

## Implementation

1. Implement transactional CUDA `set_raman_fft_params` validation/allocation.
2. Add the minimum new CUDA kernels and cuFFT plans needed for the existing
   r2c convolution algorithm; no host copies inside an RHS evaluation.
3. Add an EnvGrid `has_raman_fft` branch to the mode-averaged RHS and ensure it
   composes additively with Kerr.
4. Broaden Julia eligibility only for matching EnvGrid intermediate-broadening
   responses. Keep RealGrid, radial, modal, free-space, mixtures, and
   z-dependent cases excluded.
5. Keep `:auto` off; benchmarking is a later policy task.
6. Add explicit backend-kind fallback tests outside the hardware gate and
   strict numerical tests inside it.

## Acceptance

Prove `:SiO2` changes the Julia oracle well above tolerance; compare direct
CUDA/CPU-native stages; run a fixed-step trajectory `<1e-6` (tighten to the
measured method floor if stable); and exercise rejected-step retry/adaptive
trajectory. Setter allocation/plan failure must leave prior CUDA setup valid.
Run strict Rust CUDA, focused mode-averaged Raman, CPU `:SiO2` regressions, Rust
group, and `git diff --check`. Update all support docs and append `PORT_LOG.md`.

## Non-goals

No short-kernel convolution, radial/modal/free-space support, or `:auto`
threshold.

## Completion evidence

`amalthea/src/cuda_native.rs` now stages and commits resident r2c/c2r buffers
and cuFFT plans transactionally; `amalthea/src/kernels.cu` supplies the
envelope-intensity pack and resident spectrum multiply kernels. The Julia
eligibility gate admits only the matching mode-averaged EnvGrid response, and
`:auto` remains disabled for Raman. Strict hardware verification passed the
focused CUDA bucket at **157/157**, including direct stage agreement
(`5.74e-16`), a six-step CPU/GPU trajectory (`1.46e-16`), adaptive rejection
and rollback, and the Rust failpoint test for allocation, copy, and plan
failure. The CPU `:SiO2` regression passed **5/5** with native-vs-Julia
full-solve agreement `5.37e-13`.
