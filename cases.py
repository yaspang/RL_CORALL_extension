import numpy as np


def nm(v: float) -> float:
    return v * 1852.0


COMPRESSION = 0.35

obstacle_cases = {
    "Case 1":  [[[nm(6),    nm( 0.00)],  180]],
    "Case 2":  [[[nm(3.5),  nm(-1.50)],  135]],
    "Case 3":  [[[nm(3.5),  nm( 0.00)],  180]],
    "Case 4":  [[[nm(3.0),  nm( 1.00)],  225]],
    "Case 5":  [[[nm(5),    nm(-2.14)],   90], [[nm(7.00), nm( 0.00)], 180]],
    "Case 6":  [[[nm(2.5),  nm(-0.80)],   90], [[nm(2.80), nm(-0.20)], 135]],
    "Case 7":  [[[nm(2.8),  nm( 0.50)],  180], [[nm(2.50), nm(-1.00)],  90]],
    "Case 8":  [[[nm(5),    nm(-2.13)],   90], [[nm(7.00), nm( 0.00)], 180]],
    "Case 9":  [[[nm(2.8),  nm(-0.80)],   90], [[nm(2.50), nm(-1.00)], 135]],
    "Case 10": [[[nm(2.5),  nm( 0.00)],  180], [[nm(2.80), nm(-1.00)],  90]],
    "Case 11": [[[nm(2.5),  nm( 1.00)], -135], [[nm(2.80), nm(-0.80)],  90]],
    "Case 12": [[[nm(7),    nm( 0.00)],  180], [[nm(3.00), nm( 0.35)],  -10], [[nm(3.44), nm(-1.50)],  45]],
    "Case 13": [[[nm(6),    nm( 0.00)],  180], [[nm(3.00), nm( 0.35)],  350], [[nm(3.40), nm( 1.55)], 295]],
    "Case 14": [[[nm(2.5),  nm(-0.80)],   90], [[nm(2.80), nm(-0.30)], 135], [[nm(2.50), nm(-1.20)], 115]],
    "Case 15": [[[nm(2.8),  nm( 0.20)],  180], [[nm(2.50), nm(-0.80)],  90], [[nm(2.50), nm(-1.00)], 135]],
    "Case 16": [[[nm(2.8),  nm( 0.80)],  -90], [[nm(2.50), nm( 1.00)], -135], [[nm(2.50), nm(-1.00)],  90]],
    "Case 17": [[[nm(2.8),  nm( 0.00)],  180], [[nm(2.50), nm( 0.30)], -135], [[nm(2.50), nm(-0.80)],  90]],
    "Case 18": [[[nm(2.5),  nm(-0.30)],  135], [[nm(2.80), nm(-1.00)],  90], [[nm(2.00), nm(-0.80)], 120]],
    "Case 19": [[[nm(2.5),  nm(-0.20)],  135], [[nm(2.50), nm( 0.20)], -135], [[nm(2.00), nm(-0.80)], 120]],
    "Case 20": [[[nm(2.8),  nm( 0.00)],  180], [[nm(2.50), nm(-0.30)], 135], [[nm(2.50), nm(-1.00)],  90]],
    "Case 21": [[[nm(2.5),  nm(-0.20)],  135], [[nm(2.50), nm( 0.20)], -135], [[nm(2.50), nm(-0.50)],  90]],
    "Case 22": [[[nm(2.8),  nm( 0.00)],  180], [[nm(2.50), nm(-0.80)],  90], [[nm(2.50), nm(-1.00)], 135]],
    "Case 23": [[[nm(3.0),  nm( 1.00)],  -90]],
}

MAX_INTRUDERS = 20


def get_case(case_number: int) -> list:
    return obstacle_cases[f"Case {case_number}"]


def parse_case(case_number: int, intr_speed: float = 5.0) -> list[dict]:
    """
    Returns a list of intruder dicts with keys: x, y, psi, u.

    Coordinate convention (own vessel at origin, heading north):
      x_fwd  — ahead distance (metres) → ENU y
      x_lat  — lateral offset (metres, starboard +) → ENU x
      heading — compass degrees → ENU radians: psi = pi/2 - deg*pi/180
    """
    intruders = []
    for (x_fwd, x_lat), hdg_deg in get_case(case_number):
        intruders.append({
            "x":   x_lat  * COMPRESSION,
            "y":   x_fwd  * COMPRESSION,
            "psi": np.pi / 2 - np.deg2rad(hdg_deg),
            "u":   intr_speed,
        })
    return intruders
