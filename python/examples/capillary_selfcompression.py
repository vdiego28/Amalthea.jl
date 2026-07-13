# Hollow-core capillary self-compression -- same parameters as the
# Amalthea Python README quickstart, run through the amalthea package.
#
# 75 um capillary radius, 1 mm fibre length, Ne at 1.5 bar,
# 800 nm / 6 fs / 100 uJ input pulse, 5 saved propagation steps.

import amalthea

out = amalthea.prop_capillary(
    75e-6, 0.001, "Ne", 1.5,
    trange=1e-12, lambda_lims=(150e-9, 4e-6),
    lambda0=800e-9, tfwhm=6e-15, energy=100e-6,
    saveN=5,
)

print("keys:", out.keys())
print("Eω shape/dtype:", out["Eω"].shape, out["Eω"].dtype)
print("z:", out["z"])

# amalthea can't plot through the Python bridge (see python/README.md's
# "Known limitation" section) -- save the arrays instead and plot with
# matplotlib/whatever you like on the Python side.
import numpy as np
np.savez("capillary_selfcompression.npz", Eω=out["Eω"], z=out["z"])
print("Saved capillary_selfcompression.npz")
