# Supercontinuum generation from simple GNLSE parameters -- Python
# translation of examples/simple_interface/gnlse_scg.jl (see that file for
# the Julia/PyPlot version with plots).
#
# Fig. 3 of Dudley et al., RMP 78, 1135 (2006).

import amalthea

gamma = 0.11
flength = 15e-2
betas = (
    0.0, 0.0, -1.1830e-26, 8.1038e-41, -9.5205e-56, 2.0737e-70,
    -5.3943e-85, 1.3486e-99, -2.5495e-114, 3.0524e-129, -1.7140e-144,
)

tau_fwhm = 50e-15
lambda0 = 835e-9
power = 10000.0

out = amalthea.prop_gnlse(
    gamma, flength, betas,
    lambda0=lambda0, tau_fwhm=tau_fwhm, power=power,
    pulseshape="sech", lambda_lims=(400e-9, 2400e-9), trange=12.5e-12,
    ramanmodel="sdo",
    # τ1/τ2 have no ASCII alias (neither in amalthea's kwarg table nor
    # Julia's own Fields.resolve_greek_aliases) -- pass the Unicode names
    # directly, as Python 3 allows Unicode identifiers.
    **{"τ1": 12.2e-15, "τ2": 32e-15},
)

print("keys:", out.keys())
print("Eω shape/dtype:", out["Eω"].shape, out["Eω"].dtype)
print("z:", out["z"])

# amalthea can't plot through the Python bridge (see python/README.md's
# "Known limitation" section) -- save the arrays instead and plot with
# matplotlib/whatever you like on the Python side.
import numpy as np
np.savez("gnlse_supercontinuum.npz", Eω=out["Eω"], z=out["z"])
print("Saved gnlse_supercontinuum.npz")
