# Standalone Python post-repair performance snapshot

This is a new snapshot harness. The completed CPU audit under
`test/performance_audit/` and its frozen results remain unchanged. Design and
acceptance live in `docs/dev/native-port/PYTHON_NATIVE_PLAN.md` under
“Post-repair Python performance snapshot”; current status lives in BACKLOG.

The initial runner supports Linux x86_64, with an installed CPython wheel,
NumPy and the HDF5 extra, plus the current Julia project and local Rust library.
It requires the maintained wheel build manifest and complete independent Julia
reference manifest to match the checkout. It verifies the installed package
bytes against the source-rebuilt wheel. All execution is CPU-only.

```bash
INSTALLED_PYTHON -I test/python_performance/run.py \
  --python INSTALLED_PYTHON \
  --manifest BUILD_DIRECTORY/build.json \
  --oracles ORACLE_DIRECTORY \
  --output NEW_DIRECTORY_OUTSIDE_CHECKOUT --smoke
```

`--cases gnlse-kerr capillary-envelope` selects a bounded subset. Omit `--smoke`
only after heavy validation has finished. The runner refuses accepted timing
runs while it sees active pytest/reference-export processes. It pins processes
to one allowed CPU and sets Julia, FFTW, BLAS, OMP and Rayon to one thread.
A smoke run checks actual physics and transport but never claims accepted
performance, regardless of its timing statistics.

Each frontend/backend has a separate persistent process for each workload.
The worker measures its package import and first public entrypoint before
running matched setup/solve checks. The timed matched workload uses production
setup and stepping with accepted spectral/time windows, dense saved fields,
rtol=1e-9, atol=1e-12 and the same bounded adaptive step controls. Julia's window
uses the same preallocated FFT operations as `Amalthea.run`. A counter records
accepted steps; attempts minus accepted gives rejections. Python obtains these
counts from the Rust driver's execution metadata. Every sample constructs a
fresh model and checks the complete field against independently prepared Julia.

Two unmeasured warmups precede 10–30 randomized round-robin samples. Acceptance
requires median relative MAD <=3% and bootstrap 95% CI half-width <=5% for the
complete setup-plus-solve workload. Raw setup, solve, copying, HDF5 output,
counts and peak RSS remain available in each sample. Import/startup/first-public
metrics are single-process observations, distinct from warmed sample statistics.
Peak RSS is a process lifetime high-water mark, including the cold run and
correctness checks. Optional file-writing timings have different metadata
payloads (recorded as `hdf5_scope`); do not compare them as equal work.

The first coarse SDO GNLSE case exposes the existing Julia-to-Rust ADE Raman
method difference from the Julia/Python FFT convolution. That Rust comparison
is explicitly excluded when it fails 1e-6. The original failed results remain
retained; two finer temporal cases test convergence, and the finest requires
all four paths to pass the unchanged gate. An excluded backend contributes no
accepted timing samples or speedup claim. The pressure-gradient case explicitly
records existing Julia-native ineligibility. Other unexpected fallbacks fail.
Quantum noise is disabled in every workload.

The custom GNLSE response reproduces the physical Kerr coefficient and crosses
the complete-array public callback interface. It must agree with built-in
Python evaluation at 1e-13 and produce a nonzero callback count. Julia nonlinear
controls must change the field by more than 1e-5 before any timing is admitted.
Scalar same-input RHS checks remain at 1e-13; modal integrals use independent
converged global-error budgets and the separately accepted fixed-node suite.
No test tolerance or production physics changes to accommodate a benchmark.

All commands and fields are retained under the requested output directory;
`snapshot.json` records source/artifact hashes, hardware and runtime details,
actual backend, correctness, exclusions and raw/statistical timing results.
Failures preserve their state and worker logs. Do not change package, engine,
harness or shared-library bytes during a run. Tooling checks:

```bash
INSTALLED_PYTHON -I test/test_python_performance.py -v
```
