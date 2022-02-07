import numpy as np


def empirical_fisher(p, dp):
    """Calculates the empirical fisher informatiom matrix of some distribution.

    For some distribution p, some d-dimensional vector of distribution
    parameters theta, this function calculates the empirical fisher information
    matrix over k datapoints (x_i, y_i) using the pdf value at theta
    p(x_i, y_i; theta) and it's derivate d/dtheta p(x_i, y_i; theta).

    Args:
        p: np.ndarray of shape (k,) with entries [p_1, p_2, ..., p_k] where
            p_i=p(x_i, y_i; theta).
        dp: np.ndarray of shape (k, d) with entries
            [[J_11, J_12, ..., J_1d],
             [J_21, ...,  ..., J_2d],
               ..., ...,  ..., ....
             [J_k1, J_k2, ..., J_kd]]        
            where J_ji = d/dtheta_j p(x_i, y_i; theta)
    
    Returns:
        Empirical Fisher information matrix, np.ndarray of shape (d, d).

    Remarks:
        Bad convergence behaviour, expect high uncertainty.
    """
    k = np.size(p)

    try:
        d = np.shape(dp)[1]
    except IndexError:
        sum = 0
        for i in range(k):
            sum += 1 / p[i]**2 * dp[i]**2
        return np.array([sum / k])

    sum = np.zeros((d, d))
    for i in range(k):
        sum += 1 / p[i]**2 * np.outer(dp[i], dp[i])
    return sum / k
