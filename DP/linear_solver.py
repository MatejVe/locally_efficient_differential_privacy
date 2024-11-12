from scipy.optimize import linprog
import numpy as np
import itertools
from typing import Tuple, Dict


def generate_staircase_matrix(k: int, epsilon: float) -> np.ndarray:
    """
    Generates the staircase matrix of size k x 2^k.
    E.g. for k = 2 we get
    1      1      e^epsilon      e^epsilon
    1  e^epsilon      1          e^epsilon

    Parameters
    ----------
    k : int
        problem dimension of the mapping
    epsilon : float
        privacy parameter

    Returns
    -------
    np.ndarray
        staircase matrix
    """
    elements = [1, np.exp(epsilon)]
    columns = list(itertools.product(elements, repeat=k))
    S = np.array(columns).T
    return S


def compute_mu(s: np.ndarray, p_theta: np.ndarray, p_theta_dot: np.ndarray) -> float:
    """
    Computes mu(S_i^(k)), that is the "weight" for our linear program.

    Parameters
    ----------
    s : np.ndarray
        a single column of the staircase matrix
    p_theta : np.ndarray
        vector of private probabilities
    p_theta_dot : np.ndarray
        derivative of the vector above with respect to theta

    Returns
    -------
    float
        weight
    """
    # s: Column vector from the staircase matrix S
    # p_theta: Probability distribution of the private data
    # p_theta_dot: Derivative of the probability distribution
    numerator = sum([s[i] * p_theta_dot[i] for i in range(len(s))]) ** 2
    denominator = sum([s[i] * p_theta[i] for i in range(len(s))])
    return numerator / denominator


def solve_linear_program(
    S: np.ndarray, k: int, mu_values: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Given the staircase matrix S, k the cardinality of the private
    alphabet, and mu_values the precomputed weights this function
    calculates the optimal beta values that maximize the
    fisher information of the sanitized data.
    The output includes both the optimal theta values
    and the maximum value of the fisher information.

    Parameters
    ----------
    S : np.ndarray
        staircase matrix
    k : int
        alphabet dimension
    mu_values : np.ndarray
        precomputed weights

    Returns
    -------
    Tuple[np.ndarray, float]
        optimal beta values, max fisher
    """
    c = -mu_values  # Negate for maximization
    bounds = [(0, None) for _ in range(len(mu_values))]
    A_eq = S
    b_eq = np.ones(k)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if result.success:
        beta_opt = result.x
        return beta_opt
    else:
        raise ValueError("Linear program did not converge")


class linear_solver:
    name = "Linear solver"

    def __call__(self, p_theta, p_theta_dot, theta, epsilon, n_trials) -> Dict:
        S = generate_staircase_matrix(n_trials + 1, epsilon)
        mus = compute_mu(S, p_theta, p_theta_dot)
        opt_theta = solve_linear_program(S, n_trials + 1, mus)
        Q_matrix = S @ np.diag(opt_theta)

        return {"Q_matrix": Q_matrix, "status": "Converged"}

