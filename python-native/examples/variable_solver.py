"""Julia-compatible variable linear propagation, with independent refinement."""
import numpy as np
from amalthea_native import solve_precon

errors = []
for count in (8, 16, 32):
    step = 1/count
    result = solve_precon(lambda z, field: field*0, lambda z: [1j*z], [1], 1.,
                          dt=step, min_dt=step, max_dt=step, saveN=2*count+1)
    # Sum completed endpoint intervals, then apply L(z) on the current one.
    completed = np.floor(result.z/step)
    phase = step**2*completed*(completed+1)/2 + result.z*(result.z-completed*step)
    discrepancy = np.max(abs(result.field[0]-np.exp(1j*phase)))
    assert discrepancy < 1e-13
    # The ODE's exact solution is exp(i*z²/2); the endpoint rule converges at
    # first order for this variable coefficient, independent of DOPRI order.
    error = abs(result.field[0, -1]-np.exp(.5j))
    errors.append(error)
    print(f'Variable L, max_dz={step:g}: formula error={discrepancy:.3e}; ODE error={error:.3e}')
assert all(1.9 < a/b < 2.1 for a, b in zip(errors, errors[1:]))
