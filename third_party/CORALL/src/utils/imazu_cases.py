import numpy as np

NMI = 1852.0

def nautical_to_meters(nm_value):
    return nm_value * NMI

# ---- ORIGINAL CORALL CASES ONLY ----
obstacle_cases = {
    "Case 1": [[[nautical_to_meters(6), nautical_to_meters(0)], 180]],
    "Case 2": [[[nautical_to_meters(5), nautical_to_meters(-2.14)], 90]],
    "Case 3": [[[nautical_to_meters(3), nautical_to_meters(0)], 0]],
    "Case 4": [[[nautical_to_meters(3.44), nautical_to_meters(1.55 + 0.08)], 295]],
    "Case 5": [[[nautical_to_meters(5), nautical_to_meters(-2.0 - 0.14)], 90],
               [[nautical_to_meters(7 - 0.05), nautical_to_meters(0)], 180]],
    "Case 6": [[[nautical_to_meters(3.4), nautical_to_meters(-1.5 + 0.03)], 45],
               [[nautical_to_meters(3), nautical_to_meters(-0.35 - 0.04)], 10]],
    "Case 7": [[[nautical_to_meters(3), nautical_to_meters(0)], 0],
               [[nautical_to_meters(3.4), nautical_to_meters(-1.5 + 0.01)], 45]],
    "Case 8": [[[nautical_to_meters(5), nautical_to_meters(-2.13)], 90],
               [[nautical_to_meters(7), nautical_to_meters(0)], 180]],
    "Case 9": [[[nautical_to_meters(3.4), nautical_to_meters(-1.5 + 0.03)], 45],
               [[nautical_to_meters(5), nautical_to_meters(-2.1 - 0.05)], 90]],
    "Case 10": [[[nautical_to_meters(3), nautical_to_meters(0.35)], 350],
                [[nautical_to_meters(4.4), nautical_to_meters(-2.1 + 0.20)], 90]],
    "Case 11": [[[nautical_to_meters(5), nautical_to_meters(2.1)], -90],
                [[nautical_to_meters(3.4), nautical_to_meters(-1.5)], 45]],
    "Case 12": [[[nautical_to_meters(7), nautical_to_meters(0)], 180],
                [[nautical_to_meters(3), nautical_to_meters(0.3 + 0.05)], -10],
                [[nautical_to_meters(3.44), nautical_to_meters(-1.55 + 0.05)], 45]],
    "Case 13": [[[nautical_to_meters(6), nautical_to_meters(0)], 180],
                [[nautical_to_meters(3), nautical_to_meters(0.3 + 0.05)], 350],
                [[nautical_to_meters(3.4), nautical_to_meters(1.5 + 0.05)], 295]],
    "Case 14": [[[nautical_to_meters(3.4), nautical_to_meters(-1.5)], 45],
                [[nautical_to_meters(3), nautical_to_meters(-0.4)], 10],
                [[nautical_to_meters(5), nautical_to_meters(-2.1 - 0.05)], 90]],
    "Case 15": [[[nautical_to_meters(3), nautical_to_meters(0)], 0],
                [[nautical_to_meters(3.4), nautical_to_meters(-1.5)], 45],
                [[nautical_to_meters(5), nautical_to_meters(-2.1 - 0.05)], 90]],
    "Case 16": [[[nautical_to_meters(3.4), nautical_to_meters(1.5 - 0.03)], -45],
                [[nautical_to_meters(5), nautical_to_meters(2.1 + 0.04)], -90],
                [[nautical_to_meters(5), nautical_to_meters(-2.1 - 0.05)], 90]],
    "Case 17": [[[nautical_to_meters(3), nautical_to_meters(0)], 0],
                [[nautical_to_meters(3), nautical_to_meters(0.3 + 0.05)], -10],
                [[nautical_to_meters(3.4), nautical_to_meters(-1.5)], 45]],
    "Case 18": [[[nautical_to_meters(3.3), nautical_to_meters(-0.3 - 0.1)], 10],
                [[nautical_to_meters(3.4), nautical_to_meters(-1.5 + 0.05)], 45],
                [[nautical_to_meters(6.5), nautical_to_meters(-1.5)], 135]],
    "Case 19": [[[nautical_to_meters(3), nautical_to_meters(-0.3 - 0.07)], 10],
                [[nautical_to_meters(3), nautical_to_meters(0.3 + 0.05)], -10],
                [[nautical_to_meters(6.5), nautical_to_meters(-1.5 - 0.03)], 135]],
    "Case 20": [[[nautical_to_meters(3), nautical_to_meters(0)], 0],
                [[nautical_to_meters(3), nautical_to_meters(-0.3 - 0.05)], 10],
                [[nautical_to_meters(4.4), nautical_to_meters(-2.1 + 0.25)], 90]],
    "Case 21": [[[nautical_to_meters(3 - 0.3), nautical_to_meters(-0.3 - 0.05)], 10],
                [[nautical_to_meters(3 - 0.3), nautical_to_meters(0.3 + 0.02)], -10],
                [[nautical_to_meters(4.4), nautical_to_meters(-1.9)], 90]],
    "Case 22": [[[nautical_to_meters(3), nautical_to_meters(0)], 0],
                [[nautical_to_meters(3.94), nautical_to_meters(-1.6 - 0.13)], 45],
                [[nautical_to_meters(5), nautical_to_meters(-2.01 - 0.15)], 90]],
    "Case 23": [[[nautical_to_meters(4.243), nautical_to_meters(2.243)], -75]],
}

def get_obstacles(case_number):
    return obstacle_cases.get(f"Case {case_number}", [])

def line_cross_x_on_ownship(x0_m, y0_m, psi_rad):
    """
    Compute x-location where a target's infinite straight-line path intersects
    ownship centerline y=0. Returns np.nan if there is no useful crossing.
    """
    s = np.sin(psi_rad)
    c = np.cos(psi_rad)

    # Parallel to x-axis
    if abs(s) < 1e-10:
        # If already on centerline, treat current x as representative
        if abs(y0_m) < 1e-6:
            return x0_m
        return np.nan

    tau = -y0_m / s  # path parameter to y=0 crossing
    x_cross = x0_m + tau * c

    # We only want future/meaningful crossings ahead of ownship
    if x_cross <= 0.0:
        return np.nan
    return x_cross

def compute_case_scale(obstacles, desired_cross_x_nmi=1.0, min_scale=0.18, max_scale=0.45):
    """
    Choose ONE scale factor per case so the encounter cluster lands near the middle
    of a 2 nmi route.
    """
    xs = []
    for obs in obstacles:
        (x0_m, y0_m), ang_deg = obs
        psi = np.radians(ang_deg)
        x_cross = line_cross_x_on_ownship(x0_m, y0_m, psi)
        if np.isfinite(x_cross):
            xs.append(x_cross / NMI)

    # Fallback: use median radial distance if no line-crossing is found
    if not xs:
        rs = [np.hypot(obs[0][0], obs[0][1]) / NMI for obs in obstacles]
        rep = np.median(rs) if rs else 4.0
    else:
        rep = np.median(xs)

    scale = desired_cross_x_nmi / max(rep, 1e-6)
    return float(np.clip(scale, min_scale, max_scale))

def get_obstacle_data(case_number, desired_cross_x_nmi=1.0, target_speed_mps=10.0):
    """
    Return geometry-preserving scaled obstacle data for short-route training.
    """
    obstacles = get_obstacles(case_number)

    Xob, Yob, psiob = [], [], []
    scale = compute_case_scale(obstacles, desired_cross_x_nmi=desired_cross_x_nmi)

    for obs in obstacles:
        (x0_m, y0_m), ang_deg = obs
        Xob.append(scale * x0_m)
        Yob.append(scale * y0_m)
        psiob.append(np.radians(ang_deg))  # KEEP ORIGINAL HEADING

    Vob = [target_speed_mps] * len(obstacles)
    return Xob, Yob, Vob, np.array(psiob)