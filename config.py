"""Central configuration for the AE 695 vehicle dynamics project."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VehicleParameters:
    m: float = 1500.0
    Iz: float = 2500.0
    a: float = 1.2
    b: float = 1.6
    vx: float = 20.0
    mu: float = 0.9
    g: float = 9.81
    Cf: float = 80000.0
    Cr: float = 80000.0
    pacejka_C: float = 1.3
    pacejka_E: float = 0.97

    @property
    def L(self) -> float:
        return self.a + self.b

    @property
    def Fzf(self) -> float:
        return self.m * self.g * self.b / self.L

    @property
    def Fzr(self) -> float:
        return self.m * self.g * self.a / self.L

    @property
    def Df(self) -> float:
        return self.mu * self.Fzf

    @property
    def Dr(self) -> float:
        return self.mu * self.Fzr

    @property
    def Bf(self) -> float:
        return self.Cf / (self.pacejka_C * self.Df)

    @property
    def Br(self) -> float:
        return self.Cr / (self.pacejka_C * self.Dr)

    def validate(self) -> None:
        positive = ("m", "Iz", "a", "b", "vx", "mu", "g", "Cf", "Cr",
                    "pacejka_C", "pacejka_E")
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.mu <= 1.5:
            raise ValueError("mu must lie in (0, 1.5]")


DEFAULT_PARAMS = VehicleParameters()
DEFAULT_PARAMS.validate()

