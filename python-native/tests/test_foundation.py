import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from amalthea_native import EnvGrid, RealGrid
from amalthea_native._native import ComplexFft, RealFft
from amalthea_native.grid import planck_taper


@pytest.mark.parametrize("n", [1, 2, 7, 8, 31, 64, 257, 1024])
def test_portable_transforms(n):
    rng = np.random.default_rng(812)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    original = x.copy()
    plan = ComplexFft(n)
    expected = np.fft.fft(x)
    for _ in range(3):
        actual = np.asarray(plan.transform(x.tolist()))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=2e-13)
        np.testing.assert_allclose(np.asarray(plan.transform(actual.tolist(), True)) / n,
                                   x, rtol=1e-13, atol=2e-13)
    np.testing.assert_array_equal(x, original)
    real = RealFft(n)
    spectrum = np.asarray(real.forward(x.real.tolist()))
    np.testing.assert_allclose(spectrum, np.fft.rfft(x.real), rtol=1e-13, atol=2e-13)
    # Deliberately nonzero imaginary DC/Nyquist bins must be ignored, as FFTW does.
    spectrum[0] += 3j
    if n % 2 == 0:
        spectrum[-1] += 5j
    before = spectrum.copy()
    restored = np.asarray(real.inverse(spectrum.tolist())) / n
    np.testing.assert_allclose(restored, np.fft.irfft(spectrum, n=n), rtol=1e-13, atol=2e-13)
    np.testing.assert_array_equal(spectrum, before)


def test_fft_validation_and_reuse_after_error():
    for cls in (ComplexFft, RealFft):
        with pytest.raises(ValueError):
            cls(0)
        with pytest.raises(ValueError):
            cls(2**24 + 1)
    plan = ComplexFft(2)
    for bad in ([1j], [float("nan"), 1], [1, float("inf")]):
        with pytest.raises(ValueError):
            plan.transform(bad)
    assert plan.transform([1, 0]) == [1, 1]
    real = RealFft(4)
    with pytest.raises(ValueError):
        real.forward([1])
    with pytest.raises(ValueError):
        real.inverse([0j, complex(0, float("nan")), 0j])
    assert real.forward([1, 0, 0, 0]) == [1, 1, 1]


@pytest.mark.parametrize("n", [31, 32])
def test_hilbert_and_convolution_fft_conventions(n):
    # Analytic signal has a known phase, independent of an FFT implementation.
    t = np.arange(n) * (2 * np.pi / n)
    x = 0.5 + np.cos(3 * t)
    fft = ComplexFft(n)
    spectrum = np.asarray(fft.transform(x.astype(complex).tolist()))
    multiplier = np.zeros(n)
    multiplier[0] = 1
    multiplier[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        multiplier[n // 2] = 1
    analytic = np.asarray(fft.transform((spectrum * multiplier).tolist(), True)) / n
    np.testing.assert_allclose(analytic, 0.5 + np.exp(3j * t), rtol=1e-13, atol=1e-13)
    # Raman-style padded causal convolution compared with direct time-domain sum.
    response = np.exp(-np.arange(n) / 4) * np.sin(np.arange(n) / 3)
    intensity = x**2
    real = RealFft(2 * n)
    a = np.asarray(real.forward(np.pad(response, (0, n)).tolist()))
    b = np.asarray(real.forward(np.pad(intensity, (0, n)).tolist()))
    convolution = np.asarray(real.inverse((a * b).tolist()))[:n] / (2 * n)
    reference = np.convolve(response, intensity)[:n]
    assert np.linalg.norm(reference) > 1
    np.testing.assert_allclose(convolution, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("cls", [RealGrid, EnvGrid])
def test_grid_axes(cls):
    grid = cls(1, 800e-9, (400e-9, 1600e-9), 300e-15)
    assert grid.to.size >= grid.t.size
    assert grid.to.size & (grid.to.size - 1) == 0
    assert 0 in grid.t
    assert grid.omega.shape == grid.omega_win.shape == grid.sidx.shape
    assert np.all((grid.omega_win >= 0) & (grid.omega_win <= 1))
    assert np.any((grid.omega_win > 0) & (grid.omega_win < 1))
    assert np.any(grid.twin < 1) and np.any(grid.twin == 1)
    assert not np.shares_memory(grid.omega, grid.omega_over)
    assert grid.ω is grid.omega


@pytest.mark.parametrize("cls", [RealGrid, EnvGrid])
@pytest.mark.parametrize("kw", [dict(zmax=0), dict(reference_lambda=float("nan")),
    dict(lambda_lims=(1600e-9, 400e-9)), dict(trange=-1),
    dict(delta_t=0), dict(max_samples=2), dict(trange=1e300)])
def test_invalid_grid(cls, kw):
    args = dict(zmax=1, reference_lambda=800e-9, lambda_lims=(400e-9, 1600e-9), trange=300e-15)
    args.update(kw)
    with pytest.raises(ValueError):
        cls(**args)


def test_taper_edges():
    x = np.array([-2, -1, 0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(planck_taper(x, -1, 1, 2, 4), [0, 0, .5, 1, 1, .5, 0, 0])


def test_import_does_not_load_julia():
    assert not any(name.startswith(("juliacall", "juliapkg")) for name in sys.modules)


def test_independent_julia_grid_oracle():
    location = os.environ.get("AMALTHEA_GRID_ORACLE")
    if location is None:
        pytest.skip("Set AMALTHEA_GRID_ORACLE to independently exported Julia fixtures")
    root = Path(location)
    cases = tomllib.loads((root / "cases.toml").read_text())["cases"]
    errors = {}
    for case in cases:
        grid = (RealGrid if case["kind"] == "real" else EnvGrid)(**case["kwargs"])
        for pyname, jlname in [("t", "t"), ("to", "to"), ("omega", "omega"),
                               ("omega_over", "omega_over"), ("omega_win", "omega_win"),
                               ("twin", "twin"), ("towin", "towin"), ("sidx", "sidx")]:
            reference = np.atleast_1d(np.loadtxt(root / f'{case["name"]}-{jlname}.txt'))
            actual = getattr(grid, pyname)
            assert actual.shape == reference.shape
            if pyname == "sidx":
                np.testing.assert_array_equal(actual, reference.astype(bool))
            else:
                scale = np.max(np.abs(reference))
                error = np.max(np.abs(actual - reference)) / scale
                errors[f'{case["name"]}/{pyname}'] = float(error)
                assert error < 1e-13, (case["name"], pyname, error)
    print("Grid oracle relative errors:", json.dumps(errors, sort_keys=True))
