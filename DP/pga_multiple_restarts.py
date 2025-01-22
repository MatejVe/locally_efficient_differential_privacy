import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))

from DP.projected_gradient_ascent import PGA
from DP.utils import fisher_information_privatized
import numpy as np


class PGAMultipleRestarts:
    name = "PGAMultipleRestarts"

    def __init__(self, n_restarts: int = 10):
        self.n_restarts = n_restarts

    def __call__(
            self,
            p_theta,
            p_theta_dot,
            epsilon,
            k,
            tol=1e-3,
            max_iter=300,
    ):
        best_fish = -np.inf
        best_q = None
        stat = None
        history = None

        for _ in range(self.n_restarts):
            pga = PGA()
            results = pga(p_theta, p_theta_dot, epsilon, k, tol, max_iter)
            q = results["Q_matrix"]
            status = results["status"]
            hist = results["history"]
            fish_value = fisher_information_privatized(q, p_theta, p_theta_dot)

            if fish_value > best_fish:
                best_fish = fish_value
                best_q = q
                stat = status
                history = hist

        return {"Q_matrix": best_q, "status": stat, "history": history}