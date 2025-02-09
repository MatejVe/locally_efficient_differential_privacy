from typing import Tuple

import cvxpy as cp
import numpy as np

from DP.utils import fisher_gradient, fisher_information_privatized, is_epsilon_private, epsilon_privacy_violation


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


def project_onto_feasible_set(
    prob: cp.Problem,
    Q_var: cp.Variable,
    Q_param: cp.Parameter,
    q_to_project: np.ndarray,
):
    Q_var.value = q_to_project
    Q_param.value = q_to_project
    prob.solve(solver=cp.SCS)
    return Q_var.value


def linesearch(
    q_initial: np.ndarray,
    direction: np.ndarray,
    p_theta,
    p_theta_dot,
    epsilon: float,
    alpha_min=1.0,
    alpha_max=100.0,
    num_steps=200,
):
    """
    Perform a constrained line search to find the optimal alpha that maximizes
    fisher_information_privatized while satisfying epsilon-privacy constraints.

    Parameters
    ----------
    q_initial : np.ndarray
        Initial Q matrix.
    direction : np.ndarray
        Direction for line search.
    n_trials : int
        Number of trials for the Q matrix.
    theta : float
        Parameter value(s) for Fisher information evaluation.
    epsilon : float
        Privacy parameter.
    alpha_max : float, optional
        Maximum value for alpha.
    num_steps : int, optional
        Number of alpha steps to evaluate in the line search.

    Returns
    -------
    np.ndarray
        The updated Q matrix corresponding to the optimal alpha.
    """
    # Generate candidate alpha values
    alphas = np.linspace(alpha_min, alpha_max, num_steps)
    best_fish = -np.inf
    best_q = None

    for alpha in alphas:
        # Compute candidate Q matrix
        q_candidate = q_initial + alpha * direction
        q_candidate = np.vstack(
            [q_candidate[:-1, :], 1 - np.sum(q_candidate[:-1, :], axis=0)]
        )

        # Check feasibility
        if is_epsilon_private(q_candidate, epsilon, tol=1e-3):
            # Compute Fisher information
            fish_value = fisher_information_privatized(
                q_candidate, p_theta, p_theta_dot
            )

            # Update best solution if the Fisher information is higher
            if fish_value > best_fish:
                best_fish = fish_value
                best_q = q_candidate

    # Return the best feasible Q matrix found
    if best_q is None:
        # print(q_initial)
        # print(direction)
        # print(epsilon)
        # raise ValueError("No feasible solution found during line search.")
        # print("No feasible solution found during line search, returning the original value.")
        return q_initial

    return best_q


class PGAWithEdgeTraversal:
    """
    A class implementing the Projected Gradient Ascent (PGA) algorithm with edge traversal
    to find a Q matrix that maximizes Fisher information subject to epsilon-privacy constraints.
    """

    name = "PGAET"

    def __call__(
        self,
        p_theta,
        p_theta_dot,
        epsilon,
        k,
        tol=1e-5,
        max_iter=1000,
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
        Q_init = np.ones((k, k)) / (k) + np.random.normal(size=(k, k), scale=0.5)

        projection_problem, Q_var, Q_param = initialize_projection_solver(k, epsilon)
        # inital projection
        Q_init = project_onto_feasible_set(projection_problem, Q_var, Q_param, Q_init)

        q = Q_init
        current_fish = fisher_information_privatized(q, p_theta, p_theta_dot)
        history = [q.copy()]

        # Track intermediate projections for line search logic
        first_projection = None
        second_projection = None

        for i in range(max_iter):
            q_linesearch = None

            if first_projection is not None and second_projection is not None:
                # If we have two projections, use them to perform a line search step
                diff = second_projection - first_projection
                diff[-1, :] = 0
                q_linesearch = linesearch(q, diff, p_theta, p_theta_dot, epsilon)
                # Reset projections after line search
                first_projection = None
                second_projection = None
                # if linesearch didn't find a feasible alpha value,
                # it simply returned the same value
                # we have to be careful not to trigger algorithm termination
                q_next = q_linesearch
            else:
                # Compute gradient of Fisher information
                grad_I = fisher_gradient(p_theta, p_theta_dot, q)

                # Optional: gradient clipping or scaling if needed
                # For example:
                # grad_I = np.clip(grad_I, -1e5, 1e5)
                #grad_I = grad_I / np.max([1, np.linalg.norm(grad_I, ord="fro") / 0.1])
                grad_I[-1, :] = 0

                # Perform the gradient ascent step
                q_next = q + grad_I / np.sqrt(i + 1)
                # fix the last row (column stochasticity)
                #q_next = np.vstack([q_next[:-1, :], 1 - np.sum(q_next[:-1, :], axis=0)])

                # Check feasibility; if not private, project onto feasible region
                if not is_epsilon_private(q_next, epsilon, tol=1e-10):
                    history.append(q_next.copy())
                    q_projected = project_onto_feasible_set(
                        projection_problem, Q_var, Q_param, q_next
                    )

                    history.append(q_projected.copy())
                    if first_projection is not None:
                        second_projection = q_projected
                    else:
                        first_projection = q_projected

                    q_next = q_projected

            # Evaluate Fisher information at the candidate Q
            next_fish = fisher_information_privatized(q_next, p_theta, p_theta_dot)

            # Check for convergence
            # 1. If Q hasn't moved significantly
            # 2. If Fisher information improvement is below threshold
            # 3. q and q_linesearch can't be equal
            if (
                np.allclose(q, q_next, rtol=tol, atol=tol)
                and abs(current_fish - next_fish) < 1e-3
                and not np.array_equal(q, q_linesearch)
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


class PGAETMultipleRestarts:
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
        max_iter=500,
    ):
        best_fish = -np.inf
        best_q = None
        stat = None
        history = None

        for _ in range(self.n_restarts):
            pga = PGAWithEdgeTraversal()
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
