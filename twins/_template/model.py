"""Model: a mug of water cooling towards room temperature, with a heater (Newton's law of cooling).

    dT/dt = -(T - T_room) / TAU + GAIN * heater_on

The model is linear, so one step of length dt is computed exactly (zero-order hold on the heater).
"""
import math

TAU = 600.0     # s, cooling time constant
GAIN = 0.1      # K/s added by the heater when it is on (steady state with heater: T_room + 60 K)


def step(T, T_room, heater_on, dt):
    """Temperature after dt seconds, starting from T, with the heater on (1) or off (0) the whole time."""
    T_ss = T_room + float(heater_on) * GAIN * TAU
    return T_ss + (T - T_ss) * math.exp(-dt / TAU)
