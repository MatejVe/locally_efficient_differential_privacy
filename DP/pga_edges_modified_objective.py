from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy.stats import binom

from DP.utils import (
    fisher_information_privatized,
    is_epsilon_private,
)


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
            grad[i, j] = first - second - 0.001 ** q_mat.shape[0] * boundary_terms
    return grad


def fisher_information_privatized_modified(Q, p_theta, p_theta_dot, eps):
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
    return np.sum(numerator / denominator) - 0.001 ** len(p_theta) * boundary_terms


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


def linesearch(
    q_initial: np.ndarray,
    direction: np.ndarray,
    p_theta,
    p_theta_dot,
    alpha_max=0.1,
):

    q_new1 = q_initial + alpha_max * direction
    q_new2 = q_initial - alpha_max * direction

    fish1 = fisher_information_privatized(q_new1, p_theta, p_theta_dot)
    fish2 = fisher_information_privatized(q_new2, p_theta, p_theta_dot)

    if fish1 >= fish2:
        return q_new1
    else:
        return q_new2


class PGAModifiedEdgeTraversal:
    """
    A class implementing the Projected Gradient Ascent (PGA) algorithm with edge traversal
    to find a Q matrix that maximizes Fisher information subject to epsilon-privacy constraints.
    """

    name = "PGAMET"

    def __call__(self, p_theta, p_theta_dot, epsilon, k, tol=1e-6, max_iter=2000):
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
        Q_init = np.ones((k, k)) / (k) + np.random.normal(size=(k, k), scale=0.1)

        # If needed, you can project initial Q onto the feasible set.
        # Q_init = project_onto_feasible_set(Q_init, epsilon)

        projection_problem, Q_var, Q_param = initialize_projection_solver(k, epsilon)

        Q_param.value = Q_init
        Q_var.value = Q_init
        projection_problem.solve(solver=cp.SCS)
        q_projected = Q_var.value

        q = q_projected
        current_fish = fisher_information_privatized_modified(
            q, p_theta, p_theta_dot, epsilon
        )
        history = [q.copy()]

        # Track intermediate projections for line search logic
        first_projection = None
        second_projection = None

        for i in range(max_iter):
            if first_projection is not None and second_projection is not None:
                # If we have two projections, use them to perform a line search step
                diff = second_projection - first_projection
                q_next = linesearch(q, diff, p_theta, p_theta_dot)
                # Reset projections after line search
                first_projection = None
                second_projection = None
            else:
                # Compute gradient of Fisher information
                grad_I = fisher_gradient_modified(p_theta, p_theta_dot, q, epsilon)

                # Optional: gradient clipping or scaling if needed
                # For example:
                grad_I = np.clip(grad_I, -1e5, 1e5)

                # Perform the gradient ascent step
                q_next = q + grad_I / np.sqrt(10 * k * (i + 1))

                # Check feasibility; if not private, project onto feasible region
                if not is_epsilon_private(q_next, epsilon):
                    history.append(q_next.copy())
                    Q_param.value = q_next
                    Q_var.value = q_next
                    projection_problem.solve(solver=cp.SCS)
                    q_projected = Q_var.value

                    if q_projected is None:
                        print("q_next")
                        print(q_next)
                        print("grad_I")
                        print(grad_I)
                        print(grad_I / np.sqrt(10 * k * (i + 1)))
                    history.append(q_projected.copy())
                    if first_projection is not None:
                        second_projection = q_projected
                    else:
                        first_projection = q_projected

                    q_next = q_projected

            # Evaluate Fisher information at the candidate Q
            next_fish = fisher_information_privatized_modified(
                q_next, p_theta, p_theta_dot, epsilon
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


class PGAMETMultipleRestarts:
    name = "PGAETMultipleRestarts"

    def __init__(self, n_restarts: int = 10):
        self.n_restarts = n_restarts

    def __call__(
        self,
        p_theta,
        p_theta_dot,
        epsilon,
        k,
        tol=1e-5,
        max_iter=300,
    ):
        best_fish = -np.inf
        best_q = None
        stat = None
        history = None

        for _ in range(self.n_restarts):
            pga = PGAModifiedEdgeTraversal()
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
