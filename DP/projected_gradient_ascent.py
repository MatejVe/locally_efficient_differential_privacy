from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy.stats import binom

from DP.utils import (
    fisher_gradient,
    fisher_information_privatized,
    is_epsilon_private,
)


def initialize_projection_solver(
    k: int, epsilon: float
) -> Tuple[cp.Problem, cp.Variable, cp.Parameter]:
    Q_var = cp.Variable((k, k))
    Q_param = cp.Parameter((k, k))

    objective = cp.Minimize(cp.sum_squares(Q_var - Q_param))

    constraints = []
    constraints += [Q_var >= 0]

    for j in range(k):
        constraints += [cp.sum(Q_var[:, j]) == 1]

    exp_eps = np.exp(epsilon)
    exp_neg_eps = np.exp(-epsilon)
    for i in range(k):
        Q_i = Q_var[i, :]
        constraints += [
            Q_i[j] - exp_neg_eps * Q_i[j_prime] >= 0
            for j in range(k)
            for j_prime in range(k)
            if j < j_prime
        ]
        constraints += [
            exp_eps * Q_i[j_prime] - Q_i[j] >= 0
            for j in range(k)
            for j_prime in range(k)
            if j < j_prime
        ]

    prob = cp.Problem(objective, constraints)

    return prob, Q_var, Q_param


class PGA:
    name = "PGA"

    def __call__(self, p_theta, p_theta_dot, epsilon, k, tol=1e-6, max_iter=1000):
        Q_init = np.ones((k, k)) / (k) + np.random.normal(size=(k, k), scale=0.1)

        projection_problem, Q_var, Q_param = initialize_projection_solver(k, epsilon)
        Q_param.value = Q_init
        Q_var.value = Q_init
        projection_problem.solve(solver=cp.SCS)
        Q_init = Q_var.value

        q = Q_init
        fish = fisher_information_privatized(q, p_theta, p_theta_dot)
        history = [Q_init]

        for i in range(max_iter):
            grad_I = fisher_gradient(p_theta, p_theta_dot, q)

            # grad_I = grad_I / np.max([1, np.linalg.norm(grad_I, ord="fro") / 1])
            grad_I[-1, :] = 0

            q_next = q + 0.001 * grad_I  # / np.sqrt(200 * (i + 1))
            q_next = np.vstack([q_next[:-1, :], 1 - np.sum(q_next[:-1, :], axis=0)])

            if not is_epsilon_private(q_next, epsilon):
                Q_param.value = q_next
                Q_var.value = q_next
                projection_problem.solve(solver=cp.SCS)
                q_next = Q_var.value

            fish_next = fisher_information_privatized(q_next, p_theta, p_theta_dot)

            if np.allclose(q, q_next, rtol=tol, atol=tol):
                status = f"Converged after {i+1} iterations."
                break

            if abs(fish - fish_next) < 1e-6:
                status = f"Converged after {i+1} iteratons."
                break

            q = q_next
            fish = fish_next
            history.append(q)
            status = "Max iterations reached without convergence"

        return {"Q_matrix": q, "status": status, "history": history}


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
