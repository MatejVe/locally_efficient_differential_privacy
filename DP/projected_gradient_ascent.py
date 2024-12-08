from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy.stats import binom

from DP.utils import (binom_derivative, binom_optimal_privacy, fisher_gradient,
                      fisher_information_privatized, is_epsilon_private)


def fisher_gradient_modified(p_theta, p_theta_dot, q_mat, epsilon):
    grad = np.zeros(shape=q_mat.shape)

    long_coef = np.exp(-epsilon) + np.exp(epsilon) + 1
    for i in range(q_mat.shape[0]):
        for j in range(q_mat.shape[1]):
            qpdot = q_mat[i] @ p_theta_dot
            qp = q_mat[i] @ p_theta
            qpdot2 = qpdot**2
            qp2 = qp**2

            first = 2 * p_theta_dot[j] * qpdot / qp
            second = p_theta[j] * qpdot2 / qp2

            boundary_terms = 0
            for j_prime in range(q_mat.shape[1]):
                if j != j_prime:
                    y = q_mat[i, j]
                    x = q_mat[i, j_prime]
                    boundary_terms += (
                        3 * y**2 / x**3 - long_coef * 2 * y / x**2 + long_coef / x
                    )
            grad[i, j] = first - second + 0.001 ** q_mat.shape[0] * boundary_terms
    return grad


def fisher_information_privatized_modified(Q, n, theta, eps):
    p_theta = binom.pmf(np.arange(n + 1), n, theta)
    p_theta_dot = [binom_derivative(i, n, theta) for i in range(n + 1)]

    numerator = np.power(Q @ p_theta_dot, 2)
    denominator = Q @ p_theta
    boundary_terms = 0
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            for j_prime in range(Q.shape[1]):
                if j != j_prime:
                    y = Q[i, j]
                    x = Q[i, j_prime]
                    boundary_terms += (
                        (y / x - np.exp(-eps)) * (y / x - 1) * (y / x - np.exp(eps))
                    )
    return np.sum(numerator / denominator) + 0.001**n * boundary_terms


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


class projected_gradient_ascent:
    name = "PGA"

    def __call__(
        self, p_theta, p_theta_dot, theta, epsilon, n_trials, tol=1e-6, max_iter=1000
    ):
        Q_init = np.ones((n_trials + 1, n_trials + 1)) / (
            n_trials + 1
        ) + np.random.normal(size=(n_trials + 1, n_trials + 1), scale=0.01)
        # Q_init = project_onto_feasible_set(Q_init, epsilon)

        projection_problem, Q_var, Q_param = initialize_projection_solver(
            n_trials, epsilon
        )

        q = Q_init
        fish = fisher_information_privatized(q, n_trials, theta)
        history = [Q_init]

        for i in range(max_iter):
            grad_I = fisher_gradient(p_theta, p_theta_dot, q)
            # grad_I[-1, :] = np.zeros_like(grad_I[-1, :])

            # We need to clip the gradient since for very small rows
            # that tend to 0 we will get bonkers gradients
            # On the other hand, at extreme values of theta
            # we need these bonkers gradients
            # This is why we cannot simply bound the gradients
            # to say [-1, 1]
            grad_I[(grad_I > 10000) | (grad_I < -10000)] = (
                grad_I[(grad_I > 10000) | (grad_I < -10000)] / 2
            )
            grad_I[(grad_I > 1e7) | (grad_I < -1e7)] = 0

            q_next = q + grad_I / np.sqrt(100 * (i + 1))

            if not is_epsilon_private(q_next, epsilon):
                Q_param.value = q_next
                Q_var.value = q_next
                projection_problem.solve(solver=cp.SCS)
                q_next = Q_var.value

            fish_next = fisher_information_privatized(q_next, n_trials, theta)

            if np.allclose(q, q_next, rtol=tol, atol=tol):
                status = f"Converged after {i+1} iterations."
                break

            if abs(fish - fish_next) < 1e-8:
                status = f"Converged after {i+1} iteratons."
                break

            q = q_next
            fish = fish_next
            history.append(q)
            status = "Max iterations reached without convergence"

        return {"Q_matrix": q, "status": status, "history": history}


if __name__ == "__main__":
    solver = projected_gradient_ascent()

    q, status, history = binom_optimal_privacy(solver, 1, 1.0, 0.5)

    print(q)
    print(status)
    print(history)
