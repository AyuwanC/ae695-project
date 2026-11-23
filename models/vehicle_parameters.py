"""Vehicle parameter calculations and tables."""
import pandas as pd
from config import VehicleParameters


def parameter_dataframe(params: VehicleParameters) -> pd.DataFrame:
    """Create a complete baseline parameter table."""
    rows = [
        ("m", params.m, "kg"), ("Iz", params.Iz, "kg m^2"),
        ("a", params.a, "m"), ("b", params.b, "m"),
        ("L", params.L, "m"), ("vx", params.vx, "m/s"),
        ("mu", params.mu, "-"), ("g", params.g, "m/s^2"),
        ("Cf", params.Cf, "N/rad"), ("Cr", params.Cr, "N/rad"),
        ("Fzf", params.Fzf, "N"), ("Fzr", params.Fzr, "N"),
        ("Df", params.Df, "N"), ("Dr", params.Dr, "N"),
        ("Bf", params.Bf, "-"), ("Br", params.Br, "-"),
        ("C", params.pacejka_C, "-"), ("E", params.pacejka_E, "-"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"])

