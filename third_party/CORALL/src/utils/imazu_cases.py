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

    tau = -y0_m / s
    x_cross = x0_m + tau * c

    # only crossings ahead of ownship
    if x_cross <= 0.0:
        return np.nan
    return x_cross


def line_cross_tau_on_ownship(x0_m, y0_m, psi_rad):
    """
    Return the path parameter tau such that:
        [x_cross, 0] = [x0, y0] + tau * [cos(psi), sin(psi)]
    Returns np.nan if no useful crossing exists.
    """
    s = np.sin(psi_rad)

    if abs(s) < 1e-10:
        if abs(y0_m) < 1e-6:
            return 0.0
        return np.nan

    tau = -y0_m / s
    x_cross = x0_m + tau * np.cos(psi_rad)

    if x_cross <= 0.0:
        return np.nan
    return tau


def compute_case_scale(obstacles, desired_cross_x_nmi=1.0, min_scale=0.18, max_scale=0.45):
    """
    Choose ONE scale factor per case so the encounter cluster lands near the middle
    of a short route while preserving geometry.
    """
    xs = []
    for obs in obstacles:
        (x0_m, y0_m), ang_deg = obs
        psi = np.radians(ang_deg)
        x_cross = line_cross_x_on_ownship(x0_m, y0_m, psi)
        if np.isfinite(x_cross):
            xs.append(x_cross / NMI)

    # fallback if no centerline crossing exists
    if not xs:
        rs = [np.hypot(obs[0][0], obs[0][1]) / NMI for obs in obstacles]
        rep = np.median(rs) if rs else 4.0
    else:
        rep = np.median(xs)

    scale = desired_cross_x_nmi / max(rep, 1e-6)
    return float(np.clip(scale, min_scale, max_scale))


def compute_synchronized_obstacle_speeds(
    obstacles,
    scale,
    ownship_speed_mps=9.5,
    desired_cross_x_nmi=1.0,
    default_speed_mps=9.5,
    min_speed_mps=6.0,
    max_speed_mps=14.0,
):
    """
    Compute per-obstacle speeds so target ships arrive near the ownship crossing time.

    ownship starts at x=0 on y=0 and nominally reaches x = desired_cross_x_nmi at:
        t_own = desired_cross_x_m / ownship_speed_mps

    each obstacle speed is then:
        v_j = distance_along_track_to_crossing_j / t_own

    If a case/obstacle has no useful crossing, fall back to default_speed_mps.
    """
    desired_cross_x_m = desired_cross_x_nmi * NMI
    t_own = desired_cross_x_m / max(float(ownship_speed_mps), 1e-6)

    speeds = []
    for obs in obstacles:
        (x0_m_raw, y0_m_raw), ang_deg = obs
        psi = np.radians(ang_deg)

        # apply same geometry scale used for positions
        x0_m = scale * x0_m_raw
        y0_m = scale * y0_m_raw

        tau = line_cross_tau_on_ownship(x0_m, y0_m, psi)

        if np.isfinite(tau):
            dist_to_cross_m = abs(float(tau))  # since direction vector is unit length
            v = dist_to_cross_m / max(t_own, 1e-6)
        else:
            v = float(default_speed_mps)

        v = float(np.clip(v, min_speed_mps, max_speed_mps))
        speeds.append(v)

    return speeds


def get_obstacle_data(
    case_number,
    desired_cross_x_nmi=1.0,
    target_speed_mps=10.0,
    ownship_speed_mps=None,
    synchronize_arrivals=True,
    min_speed_mps=6.0,
    max_speed_mps=14.0,
):
    """
    Return geometry-preserving scaled obstacle data for short-route training.

    If synchronize_arrivals=True, compute per-obstacle target speeds so that each
    target reaches its centerline crossing at roughly the same time as ownship.
    """
    obstacles = get_obstacles(case_number)

    Xob, Yob, psiob = [], [], []
    scale = compute_case_scale(
        obstacles,
        desired_cross_x_nmi=desired_cross_x_nmi
    )

    for obs in obstacles:
        (x0_m, y0_m), ang_deg = obs
        Xob.append(scale * x0_m)
        Yob.append(scale * y0_m)
        psiob.append(np.radians(ang_deg))  # KEEP ORIGINAL HEADING

    if ownship_speed_mps is None:
        ownship_speed_mps = float(target_speed_mps)

    if synchronize_arrivals:
        Vob = compute_synchronized_obstacle_speeds(
            obstacles=obstacles,
            scale=scale,
            ownship_speed_mps=ownship_speed_mps,
            desired_cross_x_nmi=desired_cross_x_nmi,
            default_speed_mps=target_speed_mps,
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
        )
    else:
        Vob = [float(target_speed_mps)] * len(obstacles)

    return Xob, Yob, Vob, np.array(psiob)