"""Pole-placement state-feedback design."""
import numpy as np
from scipy.signal import place_poles


def pole_placement_gain(A, B, poles):
    result = place_poles(A, B, poles, method="YT")
    K = result.gain_matrix
    achieved = np.linalg.eigvals(A-B@K)
    return K, achieved
