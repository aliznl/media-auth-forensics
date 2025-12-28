from typing import List, Dict


class Kalman1D:
    """
    1D Kalman filter for smoothing per-frame anomaly scores.

    Purpose:
      Produce stable, interpretable 0/1 patterns for video streams by reducing score flicker.
    """

    def __init__(self, process_var: float = 1e-3, meas_var: float = 1e-2, init_state: float = 0.0):
        self.Q = float(process_var)
        self.R = float(meas_var)
        self.x = float(init_state)
        self.P = 1.0

    def update(self, z: float) -> float:
        """
        Update the filter given a measurement z (0..1).

        Returns:
          filtered estimate
        """
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (float(z) - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x


def kalman_binary_pattern(
    scores: List[float],
    params: Dict[str, float] = None,
    threshold: float = 0.5
) -> List[int]:
    """
    Convert per-frame float scores into a smoothed 0/1 pattern.

    Args:
      scores: list of per-frame scores (0..1)
      params: {process_var, meas_var}
      threshold: classify filtered score into 0/1

    Returns:
      list[int] 0/1 per frame
    """
    if not scores:
        return []

    p = params or {"process_var": 1e-3, "meas_var": 1e-2}
    kf = Kalman1D(p["process_var"], p["meas_var"], init_state=float(scores[0]))

    filtered = [kf.update(s) for s in scores]
    return [1 if f >= threshold else 0 for f in filtered]
