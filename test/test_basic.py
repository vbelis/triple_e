from triple_e import empirical_fisher

import numpy as np
from math import comb

import pytest


def test_fisher_binomial():
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
    assert empirical_fisher(p, dp) == pytest.approx(fisher_info_binomial, 0.1)


def test_fisher_gaussian():
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
    res = empirical_fisher(p, dp)

    # Requires high abstol
    assert res == pytest.approx(fisher_info_gaussian, abs=0.1)
