# Ball motion source

Adapted with the authors' permission (confirmed by the user, 27 October 2026)
from https://github.com/andrewkang12345/monteCarloPassSearch at revision
`cc6af7b00769fc780d890eca49f530c2dce97e4b`.

Original files: `monteCarloPassSearch/initParamVar/sim_core.py`, function
`_simulate_ball_with_spin`, and `sim_viz_runtime.py`, physical constants.
The published repository has no license file; this attribution records the
specific permission supplied for this project, not a general license grant.

The adaptation removes spin (fixed at zero) and yields double-precision states
so the caller can stop at the pitch-range limit or a bounded remaining distance.
Gravity, damping, bouncing, rolling conditions and update order are preserved.
