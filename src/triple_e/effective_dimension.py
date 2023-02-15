import numpy as np


def empirical_fisher(p, dp):
    """Calculates the empirical fisher informatiom matrix of some distribution.

    For some distribution p, some M different d-dimensional vector of distribution
    parameters theta, this function calculates the empirical fisher information
    matrix over k datapoints (x_i, y_i) using the pdf value at theta
    p(x_i, y_i; theta) and it's derivate d/dtheta p(x_i, y_i; theta).

    Args:
        p: np.ndarray of shape (m, k) with entries
            [[p_11, p_12, ..., p_1k],
             [p_21, p_22, ..., p_2k],
              ... , ... , ..., ...  ,
             [p_M1, p_M2, ..., p_Mk]]
             where p_mi=p(x_i, y_i; theta_m).

        dp: np.ndarray of shape (M, k, d) with entries
            [J_1, J_2, ..., J_M]
            where each J_m is equal to a jacobian of shape (k, d) with entries
            [[J_11, J_12, ..., J_1d],
             [J_21, ...,  ..., J_2d],
               ..., ...,  ..., ....
             [J_k1, J_k2, ..., J_kd]]
            where J_ji = d/dtheta_j p(x_i, y_i; theta_m)

    Returns:
        np.ndarray of shape (M, d, d), where the m-th entry is the (d, d)
        empirical fisher information matrix associated with theta_m.

    Remarks:
        Bad convergence behaviour, expect high uncertainty.
        This is a wrapper around empirical_fisher_ to allow for calculation
        over different parameter sets.
    """
    M = p.shape[0]
    res = []
    for m in range(M):
        res.append(empirical_fisher_(p[m], dp[m]))
    return np.array(res)


def empirical_fisher_(p, dp):
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
    k = p.size

    try:
        d = dp.shape[1]
    except IndexError:
        sum = 0
        for i in range(k):
            sum += 1 / p[i] ** 2 * dp[i] ** 2
        return np.array([sum / k])

    sum = np.zeros((d, d))
    for i in range(k):
        sum += 1 / p[i] ** 2 * np.outer(dp[i], dp[i])
    return sum / k


def normalised_fisher(p, dp):
    """Calculates the empirical normalised fisher informatiom matrix of some
    distribution.

    For some distribution p, some M different d-dimensional vector of distribution
    parameters theta, this function calculates the empirical fisher information
    matrix over k datapoints (x_i, y_i) using the pdf value at theta
    p(x_i, y_i; theta) and it's derivate d/dtheta p(x_i, y_i; theta).

    Args:
        p: np.ndarray of shape (m, k) with entries
            [[p_11, p_12, ..., p_1k],
             [p_21, p_22, ..., p_2k],
              ... , ... , ..., ...  ,
             [p_M1, p_M2, ..., p_Mk]]
             where p_mi=p(x_i, y_i; theta_m).

        dp: np.ndarray of shape (M, k, d) with entries
            [J_1, J_2, ..., J_M]
            where each J_m is equal to a jacobian of shape (k, d) with entries
            [[J_11, J_12, ..., J_1d],
             [J_21, ...,  ..., J_2d],
               ..., ...,  ..., ....
             [J_k1, J_k2, ..., J_kd]]
            where J_ji = d/dtheta_j p(x_i, y_i; theta_m)

    Returns:
        np.ndarray of shape (M, d, d), where the m-th entry is the (d, d)
        normalised empirical fisher information matrix associated with theta_m.
    """
    M = p.shape[0]
    d = dp.shape[2]

    all_fishers = empirical_fisher(p, dp)
    fisher_trace = np.trace(np.sum(all_fishers, axis=0))
    normalisation = d * M / fisher_trace
    return normalisation * all_fishers


def effective_dimension_(p, dp, gamma):
    r"""Calculates the effective dimension of some distribution.

    For some distribution p, some M different d-dimensional vector of distribution
    parameters theta, this function calculates the effective dimension over k
    datapoints (x_i, y_i) using the pdf value at theta p(x_i, y_i; theta) and
    it's derivate d/dtheta p(x_i, y_i; theta).

    Args:
        p: np.ndarray of shape (m, k) with entries
            [[p_11, p_12, ..., p_1k],
             [p_21, p_22, ..., p_2k],
              ... , ... , ..., ...  ,
             [p_M1, p_M2, ..., p_Mk]]
             where p_mi=p(x_i, y_i; theta_m).

        dp: np.ndarray of shape (M, k, d) with entries
            [J_1, J_2, ..., J_M]
            where each J_m is equal to a jacobian of shape (k, d) with entries
            [[J_11, J_12, ..., J_1d],
             [J_21, ...,  ..., J_2d],
               ..., ...,  ..., ....
             [J_k1, J_k2, ..., J_kd]]
            where J_ji = d/dtheta_j p(x_i, y_i; theta_m)

    Returns:
        A float, the effective dimension.

    Remarks:
        Mathematically speaking, this function calculates
        .. math::
            E = 2 \frac{\log \left( \sum_{m=1}^M \sqrt{D_m} \right) - \log M}{\log \left[ \frac{\gamma k}{2 \pi \log k} \right]}
        where D_m
        .. math::
            D_m = \det \left[\mathbb{I}_d + \frac{\gamma k}{2 \pi \log k} \hat{F}(m) \right]
        which is a discretized version of Eq. (2) in Abbas et al.
    """
    M = p.shape[0]
    k = p.shape[1]
    d = dp.shape[2]

    f_hat = normalised_fisher(p, dp)
    mat = np.eye(d) + gamma * k / (2 * np.pi * np.log(k)) * f_hat

    # log(sqrt(det)) == log(det) / 2
    rootdet = np.linalg.slogdet(mat)[1] / 2  # slogdet is more stable than det
    add = np.log(np.sum(np.exp(rootdet))) - np.log(
        M
    )  # normalized sum over parameter space
    result = 2 * add / np.log(gamma * k / (2 * np.pi * np.log(k)))
    return result
