from typing import Callable, List, Tuple

import numpy as np
from scipy.special import comb
from scipy.stats import binom


# utility functions
def print_matrix(matrix: np.ndarray):
    """
    Formats and prints a matrix.
    """
    for i, row in enumerate(matrix):
        print(f"row {i}: " + "\t".join(map(str, row)))


def reduce_optimal_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Removes rows which are almost 0.
    Epsilon privacy criterion ensures that if
    one entry in a column is 0, all entries must be 0.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    np.ndarray
        reduced matrix
    """
    nonzero_rows = list()
    for i in range(matrix.shape[0]):
        if np.any(matrix[i] >= 1e-6):
            nonzero_rows.append(matrix[i])
    return np.array(nonzero_rows)


def binom_derivative(k: int, n: int, theta: float) -> float:
    """
    Derivative of n choose k * p^k * (1-p)^(n-k)
    with respect to p at theta.

    Parameters
    ----------
    k : int
        number of successes
    n : int
        total number of trials
    theta : float
        sucess probability, parameter to be inferred

    Returns
    -------
    float
        calculated derivative

    Raises
    ------
    Exception
        If you input k > n
    """
    if k > n:
        raise Exception("Can't have k > n")
    first = k * np.power(theta, k - 1) * np.power(1 - theta, n - k)
    second = (n - k) * np.power(1 - theta, n - k - 1) * np.power(theta, k)
    return comb(n, k) * (first - second)


def fisher_gradient(
    p_theta: np.ndarray, p_theta_dot: np.ndarray, Q: np.ndarray
) -> np.ndarray:
    """
    Calculates the fisher information gradient at Q,
    given p_theta a vector of probabilities for a specific theta,
    and p_theta_dot a vector of derivatives wrt theta.

    Parameters
    ----------
    p_theta : np.ndarray
        vector of n probabilities for a specific theta
    p_theta_dot : np.ndarray
        vector of n derivatives of probability wrt theta
    Q : np.ndarray
        privacy matrix

    Returns
    -------
    np.ndarray
        gradient matrix pointing in the direction of greatest increase
    """

    # Compute Q @ p_theta and Q @ p_theta_dot for all rows
    Q_ptheta = Q @ p_theta  # Shape: (nrows,)
    Q_ptheta_dot = Q @ p_theta_dot  # Shape: (nrows,)

    # Precompute terms to avoid redundant calculations
    Q_ptheta_squared = np.power(Q_ptheta, 2)  # Element-wise squared values
    Q_ptheta_dot_squared = np.power(Q_ptheta_dot, 2)

    # Compute the numerator and denominator for all elements
    gradient_matrix = (
        2 * np.outer(p_theta_dot, Q_ptheta_dot) / Q_ptheta
        - np.outer(p_theta, Q_ptheta_dot_squared) / Q_ptheta_squared
    )

    return gradient_matrix.T


def is_epsilon_private(Q: np.ndarray, epsilon: float, tol=1e-6) -> bool:
    """
    Checks whether the matrix Q is epsilon private.

    Parameters
    ----------
    Q : np.ndarray
        privacy matrix
    epsilon : float
        privacy parameter

    Returns
    -------
    bool
        True if the matrix is epsilon private
    """
    if np.any(Q < 0):
        return False

    n_rows, n_cols = Q.shape
    for i in range(n_rows):
        row = Q[i]

        for j in range(n_cols):
            for j_prime in range(n_cols):
                if j != j_prime:
                    if np.exp(-epsilon) * row[j_prime] > row[j] + tol:
                        return False
                    if row[j] > np.exp(epsilon) * row[j_prime] + tol:
                        return False
    return True


def is_column_stochastic(Q: np.ndarray) -> bool:
    ncols = Q.shape[1]
    for i in range(ncols):
        col = Q[:, i]
        if np.sum(col) != 1.0:
            return False

    return True


def fisher_information_privatized(
    Q: np.ndarray, p_theta: np.ndarray, p_theta_dot: np.ndarray
) -> float:
    """
    Calculates the fisher information of the sanitazed data.

    Parameters
    ----------
    Q : np.ndarray
        privacy matrix
    p_theta : np.ndarray
        vector of probabilities
    p_theta_dot : np.ndarray
        vector of derivatives of probabilities

    Returns
    -------
    float
        fisher information
    """
    numerator = np.power(Q @ p_theta_dot, 2)
    denominator = Q @ p_theta
    # safeguard against very small denominators
    denominator[denominator < 1e-12] = 1e-12
    return np.sum(numerator / denominator)


def fisher_information_binom(n: int, theta: float) -> float:
    """
    Calculates fisher information of a binomial distribution with
    a single parameter theta.

    Parameters
    ----------
    n : int
        number of trials
    theta : float
        success probability parameter

    Returns
    -------
    float
        fisher information
    """
    return n / (theta * (1 - theta))


def binom_optimal_privacy(
    solver: Callable, n_trials: int, epsilon: float, theta: float
) -> Tuple[np.ndarray, float, List]:
    """
    Parameters
    ----------
    n_trials : int
        number of trials in binomial
    epsilon : float
        privacy param
    theta : float
        succes probability

    Returns
    -------
    Tuple[np.ndarray, str, np.ndarray]
        optimal Q matrix, convergence status, history
    """
    # input space is n_trials since we can have at most n
    # successes
    k = n_trials + 1  # output space includes 0
    p_theta = binom.pmf(np.arange(k), n_trials, theta)
    p_theta_dot = np.array([binom_derivative(i, n_trials, theta) for i in range(k)])

    result = solver(p_theta, p_theta_dot, epsilon, k)

    Q_matrix = result["Q_matrix"]
    Q_matrix = reduce_optimal_matrix(Q_matrix)
    # status and history are not necessarily implemented
    status = result.get("status", None)
    history = result.get("history", None)
    best_fisher = fisher_information_privatized(Q_matrix, p_theta, p_theta_dot)

    return (Q_matrix, status, history, best_fisher)
