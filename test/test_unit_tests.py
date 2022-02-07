from triple_e import empirical_fisher, empirical_fisher_

import numpy as np
from math import comb

import pytest

from triple_e.effective_dimension import effective_dimension_, normalised_fisher


def test_single_binomial():
    # p(k, theta) = binom(n, k) * theta^k * (1-theta)^(n-k)
    # d/dtheta p(k, theta) = binom(n, K) * theta^(k-1) * (1-theta)^(n-k-1)*(k-n*theta)
    n_sample = 1000
    n = 10
    np.random.seed(42)
    theta = np.random.rand()

    x = np.random.binomial(n, theta, n_sample)
    p_func = lambda x_i: comb(n, x_i) * theta**x_i * (1 - theta)**(n - x_i)
    dp_factor = lambda x_i: (x_i - n * theta) / (theta * (1 - theta))

    p = np.array([p_func(x_i) for x_i in x])
    dp = p * np.array([dp_factor(x_i) for x_i in x])

    fisher_info_binomial = np.array([n / (theta * (1 - theta))])  # analytical

    # Requires high abstol
    assert empirical_fisher_(p, dp) == pytest.approx(fisher_info_binomial, 0.1)


def test_single_gaussian_multiple_parameters():
    n_sample = 1000
    np.random.seed(42)
    mu = np.random.rand() * 5
    nu = np.random.rand()

    x = np.random.normal(mu, nu, n_sample)
    p = 1 / (2 * np.pi * nu) * np.exp(-(x - mu)**2 / (2 * nu))
    dpdmu = p * (x - mu) / nu
    dpdnu = p * ((x - mu)**2 / (2 * nu**2) - 1 / (2 * nu))

    dp = np.vstack([dpdmu, dpdnu]).T

    # analytical
    fisher_info_gaussian = np.array([[1 / nu, 0], [0, 1 / (2 * nu**2)]])

    # Requires high abstol
    assert empirical_fisher_(p, dp) == pytest.approx(fisher_info_gaussian,
                                                     abs=0.1)


def test_one_parameter_set():
    my_p = np.array([[0.2, 0.3, 0.1, 0.2, 0.3]])
    my_dp = np.array([[0.2, 0.3, 0.1, 0.2, 0.3]])

    assert empirical_fisher(my_p,
                            my_dp)[0] == empirical_fisher_(my_p[0], my_dp[0])


def test_multiple_parameter_sets():
    M = 10

    n_sample = 1000
    np.random.seed(42)

    p_all = []
    dp_all = []
    for i in range(10):
        mu = np.random.rand() * 5
        nu = np.random.rand()

        x = np.random.normal(mu, nu, n_sample)
        p = 1 / (2 * np.pi * nu) * np.exp(-(x - mu)**2 / (2 * nu))
        dpdmu = p * (x - mu) / nu
        dpdnu = p * ((x - mu)**2 / (2 * nu**2) - 1 / (2 * nu))

        dp = np.vstack([dpdmu, dpdnu]).T

        p_all.append(p)
        dp_all.append(dp)

    assert empirical_fisher(np.array(p_all),
                            np.array(dp_all)).shape == (10, 2, 2)


def test_normalisation():
    M = 10

    n_sample = 1000
    np.random.seed(42)

    p_all = []
    dp_all = []
    for i in range(10):
        mu = np.random.rand() * 5
        nu = np.random.rand()

        x = np.random.normal(mu, nu, n_sample)
        p = 1 / (2 * np.pi * nu) * np.exp(-(x - mu)**2 / (2 * nu))
        dpdmu = p * (x - mu) / nu
        dpdnu = p * ((x - mu)**2 / (2 * nu**2) - 1 / (2 * nu))

        dp = np.vstack([dpdmu, dpdnu]).T

        p_all.append(p)
        dp_all.append(dp)

    result = normalised_fisher(np.array(p_all), np.array(dp_all))
    result = 1 / M * np.trace(np.sum(result, axis=0))

    # if the normalisation works, result = d
    assert result == pytest.approx(2)


def test_effective_dimension():
    M = 10

    n_sample = 1000
    np.random.seed(42)

    p_all = []
    dp_all = []
    for i in range(10):
        mu = np.random.rand() * 5
        nu = np.random.rand()

        x = np.random.normal(mu, nu, n_sample)
        p = 1 / (2 * np.pi * nu) * np.exp(-(x - mu)**2 / (2 * nu))
        dpdmu = p * (x - mu) / nu
        dpdnu = p * ((x - mu)**2 / (2 * nu**2) - 1 / (2 * nu))

        dp = np.vstack([dpdmu, dpdnu]).T

        p_all.append(p)
        dp_all.append(dp)

    result = effective_dimension_(np.array(p_all), np.array(dp_all), 1)
    assert isinstance(result, float)

# TODO: Test against something where we know the value.