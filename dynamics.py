import numpy as np
from dataclasses import dataclass, field, astuple


@dataclass
class VesselState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0    # [rad]
    r: float = 0.0      # yaw rate [rad/s]
    b: float = 0.0      # yaw bias [rad/s]
    u: float = 0.0      # surge speed [m/s]

    def to_array(self) -> np.ndarray:
        return np.array(astuple(self))

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VesselState":
        return cls(*arr)


@dataclass
class Vessel:
    state: VesselState = field(default_factory=VesselState)
    k_psi: float = 0.001    # heading gain  
    t_psi: float = 150.0    # yaw time constant [s] 
    k_v: float = 1.0        # speed gain
    t_v: float = 300.0      # speed time constant [s]  
    sigma_b: float = 0.001  # bias noise intensity [rad/s / sqrt(s)]

    @property
    def t_b(self) -> float:
        return 20.0 * self.t_psi

    def _drift(self, tau_c: float, u_c: float) -> np.ndarray:
        s = self.state
        return np.array([
            s.u * np.cos(s.psi),
            s.u * np.sin(s.psi),
            s.r,
            -(1 / self.t_psi) * s.r + (self.k_psi / self.t_psi) * (tau_c - s.b),
            -(1 / self.t_b) * s.b,
            -(1 / self.t_v) * s.u + (self.k_v / self.t_v) * u_c,
        ])

    def step(self, tau_c: float, u_c: float, dt: float, rng: np.random.Generator) -> None:
        y = self.state.to_array()
        y_next = y + self._drift(tau_c, u_c) * dt
        y_next[4] += self.sigma_b * rng.standard_normal() * np.sqrt(dt)
        y_next[2] %= 2 * np.pi
        self.state = VesselState.from_array(y_next)

    def obs(self) -> np.ndarray:
        s = self.state
        return np.array(
            [np.cos(s.psi), np.sin(s.psi), s.r, s.u],
            dtype=np.float32,
        )

    def relative_obs(self, intruder: "Vessel") -> np.ndarray:
        own, tgt = self.state, intruder.state

        # Relative position in world frame
        dx = tgt.x - own.x
        dy = tgt.y - own.y

        # Rotate world-frame vectors into own-ship body frame by -own.psi
        c, s = np.cos(own.psi), np.sin(own.psi)
        x_rel = c * dx + s * dy      # along own bow (surge axis)
        y_rel = -s * dx + c * dy     # along own beam (sway axis)

        # Vessel velocity in world frame 
        vx_world = tgt.u * np.cos(tgt.psi)
        vy_world = tgt.u * np.sin(tgt.psi)

        # Own velocity in world frame
        ux_world = own.u * np.cos(own.psi)
        uy_world = own.u * np.sin(own.psi)

        # Relative velocity in world frame, then rotated into own body frame
        dvx_world = vx_world - ux_world
        dvy_world = vy_world - uy_world
        vx_rel = c * dvx_world + s * dvy_world
        vy_rel = -s * dvx_world + c * dvy_world

        # Relative heading, encoded as sin/cos to avoid wraparound discontinuity
        dpsi = tgt.psi - own.psi
        sin_dpsi = np.sin(dpsi)
        cos_dpsi = np.cos(dpsi)

        # Relative yaw rate
        dr = tgt.r - own.r

        return np.array([
            x_rel,       # relative position along own bow
            y_rel,       # relative position along own beam
            vx_rel,      # relative velocity along own bow
            vy_rel,      # relative velocity along own beam
            sin_dpsi,    # relative heading (sin)
            cos_dpsi,    # relative heading (cos)
            dr,          # relative yaw rate
        ], dtype=np.float32)

    def distance_to(self, other: "Vessel") -> float:
        own, tgt = self.state, other.state
        return float(np.hypot(tgt.x - own.x, tgt.y - own.y))
