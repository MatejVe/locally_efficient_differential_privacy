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
    n_trials: int, epsilon: float
) -> Tuple[cp.Problem, cp.Variable, cp.Parameter]:
    n_plus_1 = n_trials + 1
    Q_var = cp.Variable((n_plus_1, n_plus_1))
    Q_param = cp.Parameter((n_plus_1, n_plus_1))

    objective = cp.Minimize(cp.sum_squares(Q_var - Q_param))

    constraints = []
    constraints += [Q_var >= 0]

    for j in range(n_plus_1):
        constraints += [cp.sum(Q_var[:, j]) == 1]

    exp_eps = np.exp(epsilon)
    exp_neg_eps = np.exp(-epsilon)
    for i in range(n_plus_1):
        Q_i = Q_var[i, :]
        constraints += [
            Q_i[j] - exp_neg_eps * Q_i[j_prime] >= 0
            for j in range(n_plus_1)
            for j_prime in range(n_plus_1)
            if j < j_prime
        ]
        constraints += [
            exp_eps * Q_i[j_prime] - Q_i[j] >= 0
            for j in range(n_plus_1)
            for j_prime in range(n_plus_1)
            if j < j_prime
        ]

    prob = cp.Problem(objective, constraints)

    return prob, Q_var, Q_param


class PGA:
    name = "PGA"

    def __call__(
        self, p_theta, p_theta_dot, theta, epsilon, n_trials, tol=1e-6, max_iter=1000
    ):
        Q_init = np.ones((n_trials + 1, n_trials + 1)) / (
            n_trials + 1
        ) + np.random.normal(size=(n_trials + 1, n_trials + 1), scale=0.1)

        projection_problem, Q_var, Q_param = initialize_projection_solver(
            n_trials, epsilon
        )
        Q_param.value = Q_init
        Q_var.value = Q_init
        projection_problem.solve(solver=cp.SCS)
        Q_init = Q_var.value

        q = Q_init
        fish = fisher_information_privatized(q, n_trials, theta)
        history = [Q_init]

        for i in range(max_iter):
            grad_I = fisher_gradient(p_theta, p_theta_dot, q)

            #grad_I = grad_I / np.max([1, np.linalg.norm(grad_I, ord="fro") / 1])
            grad_I[-1, :] = 0

            q_next = q + grad_I / np.sqrt(200 * (i + 1))
            q_next = np.vstack([q_next[:-1,:], 1 - np.sum(q_next[:-1,:], axis=0)])

            if not is_epsilon_private(q_next, epsilon):
                Q_param.value = q_next
                Q_var.value = q_next
                projection_problem.solve(solver=cp.SCS)
                q_next = Q_var.value

            fish_next = fisher_information_privatized(q_next, n_trials, theta)

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
