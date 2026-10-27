"""Zero-spin adaptation of Kang et al.'s ball simulator, reused with permission.

Source: andrewkang12345/monteCarloPassSearch, revision
cc6af7b00769fc780d890eca49f530c2dce97e4b, initParamVar/sim_core.py
(_simulate_ball_with_spin) and sim_viz_runtime.py (constants).
The user confirmed the authors' reuse approval on 27 October 2026.
Changes: omit the zero-valued spin branch; yield states for adaptive termination.
The remaining arithmetic and update order follow the source implementation.
"""
import numpy as np

FPS = 25.0
GRAVITY = 9.81
AIR_RESISTANCE = 0.00648
RESTITUTION = 0.109
GROUND_FRICTION = 0.0
ROLLING_FRICTION = 0.02731
ROLLING_Z_THRESH = 0.054


def states(start_pos, start_vel):
    dt = 1.0 / FPS
    pos = np.asarray(start_pos, dtype=np.float64).copy()
    vel = np.asarray(start_vel, dtype=np.float64).copy()
    while True:
        yield pos.copy(), vel.copy()
        vel *= (1.0 - AIR_RESISTANCE)
        if pos[2] < ROLLING_Z_THRESH and abs(vel[2]) < 1.0:
            vel[0] *= (1.0 - ROLLING_FRICTION)
            vel[1] *= (1.0 - ROLLING_FRICTION)
        vel[2] -= GRAVITY * dt
        pos += vel * dt
        if pos[2] < 0:
            pos[2] = 0.0
            vel[2] = -RESTITUTION * vel[2]
            vel[0] *= (1.0 - GROUND_FRICTION)
            vel[1] *= (1.0 - GROUND_FRICTION)
