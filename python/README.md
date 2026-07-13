# amalthea (Python)

Python interface to [Amalthea.jl](../README.md) (formerly Luna-Rust.jl), a
performance-focused fork of Luna.jl with a native Rust backend for
ultrafast-fibre pulse propagation. This package embeds Julia via
[`juliacall`](https://github.com/JuliaPy/PythonCall.jl) and calls straight
into `Amalthea.prop_capillary` / `Amalthea.prop_gnlse`, returning numpy-backed
results.

## Install

```bash
pip install -e ./python
```

This installs the Python package plus `juliacall`/`juliapkg`. **No manual
Julia install is required** — `juliapkg` provisions a compatible Julia
toolchain automatically (on this machine, via `juliaup`) and resolves/
precompiles the `Amalthea` package and its
dependencies (including the native `amalthea` Rust shared library, built by
`Amalthea`'s `deps/build.jl`) the first time `amalthea.prop_capillary`/
`prop_gnlse` is called — not at `pip install` time.

That first call is slow: expect Julia package resolution, precompilation,
and a large amount of `Pkg`/precompile log noise on stdout/stderr,
**2-4 minutes** on a cold environment. This is normal, not an error — look
for your own printed output at the end of the log. Every call after that
(and every call in a fresh process reusing the same environment) is fast,
since Julia and Amalthea stay loaded for the life of the process.

`amalthea` requires Julia `1.10`+ (matching `Amalthea.jl`'s
`Project.toml`); `juliapkg` will fetch a compatible version automatically —
you don't need to install Julia yourself, but if you already have one on
`PATH` or set via the `JULIA` environment variable, `juliapkg` will consider
it before provisioning its own.

## Quickstart

```python
import amalthea

out = amalthea.prop_capillary(
    75e-6, 0.001, "Ne", 1.5,
    trange=1e-12, lambda_lims=(150e-9, 4e-6),
    lambda0=800e-9, tfwhm=6e-15, energy=100e-6,
    saveN=5,
)

print(out.keys())
print(out["Eω"].shape, out["Eω"].dtype)
print(out["z"])
```

This was run against this repo (capillary radius 75 µm, 1 mm fibre length,
Ne at 1.5 bar, 800 nm / 6 fs / 100 µJ input pulse, 5 saved propagation
steps) and produced:

```
KEYS: ['simulation_type', 'meta', 'Eω', 'prop_capillary_args', 'grid', 'stats', 'z']
Eω shape/dtype: (4097, 5) complex128
z: [0.      0.00025 0.0005  0.00075 0.001  ]
```

`Eω` is the frequency-domain field, shape `(n_frequencies, n_saved_z)` —
here 4097 frequency bins by the 5 requested `z` save-points (`saveN=5`).
`z` is the array of fibre positions (metres) at which the field was saved.

`prop_capillary(radius, flength, gas, pressure, **kwargs)` and
`prop_gnlse(gamma, flength, betas, **kwargs)` mirror
`Amalthea.prop_capillary`/`Amalthea.prop_gnlse`'s Julia positional
arguments; everything else is passed as keyword arguments (see below).
`gas` may be a plain string (`"Ne"`, `"He"`, ...) — it's converted to a
Julia `Symbol` for you. `pressure` and `betas` may be a Python list/tuple
(e.g. a two-point pressure gradient) and are converted to Julia tuples.

## Keyword argument translation

Amalthea.jl's Julia API uses Unicode keyword names (`λ0`, `τfwhm`, `ϕ`, ...).
`amalthea` accepts either the Unicode form directly, or an ASCII alias that
gets translated before the call reaches Julia. The full table
(`amalthea._kwargs._ASCII_TO_UNICODE`):

| ASCII kwarg       | Unicode (Julia) kwarg |
|--------------------|------------------------|
| `lambda0`          | `λ0`                   |
| `lam0`              | `λ0`                   |
| `wavelength`        | `λ0`                   |
| `lambda_lims`       | `λlims`                |
| `lam_lims`          | `λlims`                |
| `wavelength_lims`   | `λlims`                |
| `tau_fwhm`          | `τfwhm`                |
| `tfwhm`             | `τfwhm`                |
| `duration`          | `τfwhm`                |
| `tau_w`             | `τw`                   |
| `tw`                | `τw`                   |
| `phi`               | `ϕ`                    |
| `phase`             | `ϕ`                    |
| `delta_t`           | `δt`                   |
| `dt`                | `δt`                   |
| `delta_lambda`      | `Δλ`                   |
| `dlambda`           | `Δλ`                   |

Any keyword not in this table (e.g. `trange`, `energy`, `saveN`,
`shotnoise`, `raman`, `shock`) is passed through unchanged. Both
`lambda0=800e-9` and `λ0=800e-9` are valid and equivalent — but supplying
both (or two aliases of the same canonical name) raises `ValueError:
Duplicate kwarg: ...` rather than silently picking one.

A few kwargs also get their *values* coerced to Julia types
(`_kwargs._to_julia_type`): string values for `gas`, `pulseshape`,
`polarisation`, `model`, `plasma`, `ramanmodel`, or `modes` become Julia
`Symbol`s (e.g. `pulseshape="gauss"` → `Symbol("gauss")`); a list/tuple
passed as `modes` has its string elements converted to `Symbol`s too
(non-string elements, e.g. mode-order integers, pass through as-is).

## `LunaOutput`

`prop_capillary`/`prop_gnlse` return a `amalthea.output.LunaOutput` — a
dict-like, lazy wrapper around whatever Julia returned (a `Dict`,
`MemoryOutput`, `HDF5Output`, or an HDF5 `Group`/`File` reached by indexing
into one). It supports:

- `out[key]` — indexes into the underlying Julia value. Julia
  `AbstractArray`s are converted to numpy arrays; Julia `Dict`s, HDF5
  `Group`/`File`s, and any `Amalthea.Output.AbstractOutput` are wrapped in
  another `LunaOutput` recursively (rather than a raw `juliacall` object);
  anything else (a scalar, a string) is returned via `juliacall`'s default
  conversion. A failed index raises `KeyError`, not a raw Julia exception.
- `key in out` — checks key membership; supported for `Dict`, any
  `AbstractOutput`, and HDF5 `Group`/`File`.
- `out.keys()` — lists top-level keys. Dispatches on the underlying type:
  a plain `Dict`'s keys; a `MemoryOutput`'s `.data` keys; an `HDF5Output`
  re-opens its file read-only and lists top-level dataset/group names; an
  HDF5 `Group`/`File` handle's own `.keys()`; any other `AbstractOutput`
  with a `.data` field falls back to that field's keys. Returns `[]` if
  none of these match.

Example of nested access, run against the same `out` as above:

```python
meta = out["meta"]
print(meta.keys())
print("Eω" in out)
```

produced:

```
type(meta): <class 'amalthea.output.LunaOutput'>
meta.keys(): ['git_commit', 'sourcecode']
"Eω" in out -> True
```

`out["meta"]` is a `Dict` in Julia, so it comes back wrapped as another
`LunaOutput` (per `_to_python`'s `Dict`/`AbstractOutput`/HDF5-handle check);
`out["grid"]` and `out["stats"]` behave the same way. `"Eω" in out`
resolves via `LunaOutput.__contains__`'s `AbstractOutput` branch
(`haskey` on the underlying `MemoryOutput`), which does work — this isn't
guaranteed just because `__getitem__` works, so it's worth confirming
explicitly if you rely on `in` for a different output type (e.g. an
`HDF5Output` after closing/reopening).

## Error propagation

`prop_capillary`/`prop_gnlse` do not wrap or translate Julia-side errors.
An invalid argument (an unrecognised gas symbol, an inconsistent grid
parameter, a missing required keyword, ...) raises whatever `juliacall`
raises for a Julia exception — typically `juliacall.JuliaError` — with the
underlying Julia backtrace attached to the exception message. This is
legible enough to debug directly from Python; there is no separate
Python-side exception hierarchy to learn.

## Examples

Two runnable scripts under `python/examples/`:

- `capillary_selfcompression.py` — hollow-core capillary self-compression,
  the same parameters as the Quickstart above.
- `gnlse_supercontinuum.py` — supercontinuum generation, a Python
  translation of `examples/simple_interface/gnlse_scg.jl` (Dudley et al.,
  RMP 78, 1135 (2006), Fig. 3) — see that file for the Julia/PyPlot version
  with plots.

Run either directly:

```bash
python python/examples/capillary_selfcompression.py
python python/examples/gnlse_supercontinuum.py
```

Both print the output shape/dtype/keys and save the `Eω`/`z` arrays to an
`.npz` file, matching the no-plotting limitation described next — plot the
saved arrays yourself with `matplotlib` afterward.

## Known limitation: no plotting through the Python wrapper

`Amalthea.Plotting` (the Julia-side plotting helpers, backed by PyPlot) is
implemented as a **weak-dependency package extension**
(`AmaltheaPyPlotExt`) precisely so that `using Amalthea` — what
`amalthea.get_julia()` does under the hood — never touches PyPlot/PyCall.
That separation exists for a real reason, not just packaging hygiene:
PyCall embeds its own Conda-bundled `libpython`, and loading it inside the
same process where `juliacall`/PythonCall.jl has already embedded Julia
into the host CPython causes a `libpython` collision that crashes the
process (this is exactly what used to make `prop_capillary` segfault on
its very first call, before the extension split).

**Do not** call `Amalthea.Plotting.*` or `Main.seval("using PyPlot")`
through the `juliacall` bridge from Python — doing so activates the
extension and reintroduces the same double-`libpython` crash, extension or
not. This is an architectural limitation of embedding Julia (with PyCall
anywhere in its dependency graph) inside an already-running CPython
process, not a bug that a future patch will remove.

If you need plots, save results with `Amalthea.Output.HDF5Output` (or read
`Eω`/`grid`/`stats`/`z` back into Python via `LunaOutput`, as above) and
plot with `matplotlib`/whatever you like on the Python side — or run the
plotting Julia-side, in a separate, plain Julia process/script that never
also embeds Python.

## Tests

```bash
pip install -e "./python[test]"
pytest python/tests/
```

Most of the test suite (`test_init.py`, `test_kwargs.py`, `test_output.py`,
`test_python_api.py`) mocks `get_julia()` via `conftest.py`'s
`mock_julia_backend` fixture — it exercises kwarg translation,
`LunaOutput`'s interface, and duplicate-kwarg detection without calling
real Julia/Amalthea, so it runs in seconds.

`test_integration.py` is real: it's marked `@pytest.mark.integration`,
which `conftest.py` recognises and skips mocking for, so it boots actual
Julia, loads the real `Amalthea` package, and calls into the real
Rust-accelerated backend (the same capillary case as the Quickstart above,
plus a GNLSE case, an invalid-argument error-propagation check, and an
`HDF5Output` round-trip). It's skip-guarded, not a hard failure, if the
Rust shared library hasn't been built (`cargo build --release` in
`amalthea/`) or Julia/Amalthea can't load.

```bash
pytest python/tests/                        # everything, including the real test (~2-4 min cold)
pytest -m "not integration" python/tests/    # fast loop, skips the real test
pytest -m integration python/tests/          # only the real test
```
