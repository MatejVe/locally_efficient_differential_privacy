from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy.stats import binom

from DP.utils import (binom_derivative, fisher_gradient,
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


def linesearch(
    q_initial: np.ndarray,
    direction: np.ndarray,
    n_trials: int,
    theta: float,
    alpha_max=0.1,
):

    q_new1 = q_initial + alpha_max * direction
    q_new2 = q_initial - alpha_max * direction

    fish1 = fisher_information_privatized(q_new1, n_trials, theta)
    fish2 = fisher_information_privatized(q_new2, n_trials, theta)

    if fish1 >= fish2:
        return q_new1
    else:
        return q_new2


class PGAModifiedEdgeTraversal:
    """
    A class implementing the Projected Gradient Ascent (PGA) algorithm with edge traversal
    to find a Q matrix that maximizes Fisher information subject to epsilon-privacy constraints.
    """

    name = "PGA"

    def __call__(
        self,
        p_theta,
        p_theta_dot,
        theta,
        epsilon,
        n_trials,
        tol=1e-6,
        max_iter=2000,
        step_size=0.05,
    ):
        """
        Execute the PGA algorithm to optimize Q matrix.

        Parameters
        ----------
        p_theta : function or array-like
            Parameterized distribution p(theta).
        p_theta_dot : function or array-like
            Derivative of p(theta) with respect to theta.
        theta : float or array-like
            The parameter value(s) at which we evaluate Fisher information.
        epsilon : float
            The privacy parameter for the feasible set constraints.
        n_trials : int
            The number of trials.
        tol : float, optional
            Convergence tolerance for the Q matrix changes.
        max_iter : int, optional
            Maximum number of iterations to run.
        step_size : float, optional
            Step size for gradient ascent updates.

        Returns
        -------
        dict
            A dictionary containing:
            - "Q_matrix": Final Q matrix.
            - "status": String describing convergence status.
            - "history": List of Q matrices visited over iterations.
        """

        # Initialize Q with random perturbation around a uniform matrix.
        Q_init = np.ones((n_trials + 1, n_trials + 1)) / (
            n_trials + 1
        ) + np.random.normal(size=(n_trials + 1, n_trials + 1), scale=0.1)

        # If needed, you can project initial Q onto the feasible set.
        # Q_init = project_onto_feasible_set(Q_init, epsilon)

        projection_problem, Q_var, Q_param = initialize_projection_solver(
            n_trials, epsilon
        )

        q = Q_init
        current_fish = fisher_information_privatized_modified(
            q, n_trials, theta, epsilon
        )
        history = [q.copy()]

        # Track intermediate projections for line search logic
        first_projection = None
        second_projection = None

        for i in range(max_iter):
            if first_projection is not None and second_projection is not None:
                # If we have two projections, use them to perform a line search step
                diff = second_projection - first_projection
                q_next = linesearch(q, diff, n_trials, theta)
                # Reset projections after line search
                first_projection = None
                second_projection = None
            else:
                # Compute gradient of Fisher information
                grad_I = fisher_gradient_modified(p_theta, p_theta_dot, q, epsilon)

                # Optional: gradient clipping or scaling if needed
                # For example:
                # grad_I = np.clip(grad_I, -1e5, 1e5)

                # Perform the gradient ascent step
                q_next = q + grad_I / np.sqrt(100 * (i + 1))

                # Check feasibility; if not private, project onto feasible region
                if not is_epsilon_private(q_next, epsilon):
                    history.append(q_next.copy())
                    Q_param.value = q_next
                    Q_var.value = q_next
                    projection_problem.solve(solver=cp.SCS)
                    q_projected = Q_var.value

                    history.append(q_projected.copy())
                    if first_projection is not None:
                        second_projection = q_projected
                    else:
                        first_projection = q_projected

                    q_next = q_projected

            # Evaluate Fisher information at the candidate Q
            next_fish = fisher_information_privatized_modified(
                q_next, n_trials, theta, epsilon
            )

            # Check for convergence
            # 1. If Q hasn't moved significantly
            # 2. If Fisher information improvement is below threshold
            if (
                np.allclose(q, q_next, rtol=tol, atol=tol)
                or abs(current_fish - next_fish) < 1e-5
            ):
                status = f"Converged after {i+1} iterations."
                q = q_next
                history.append(q.copy())
                break

            # Update for next iteration
            q = q_next
            current_fish = next_fish
            history.append(q.copy())
            status = "Max iterations reached without convergence"

        return {"Q_matrix": q, "status": status, "history": history}
