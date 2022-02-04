# Computes expressbility of circuits
# based on: https://arxiv.org/abs/1905.10876

import numpy as np
from qiskit.quantum_info import state_fidelity

# number of threads to use
#import os
#os.environ['OMP_NUM_THREADS'] = '2'


def expressibility(circuit_simulator,
                   n_params,
                   n_qubits,
                   n_shots=1000,
                   seed=None,
                   n_bins=None,
                   epstol=1e-18):
    """Computes expressibility for a circuit.

    Args:
        circuit_simulator: A function that takes n_params and returns a qiskit
            Statevector or DensityMatrix.
        n_params: The number of parameters circuit_simulator accepts. Presumed
            to be uniformly distributed in [0, 2pi]
        n_qubits: Number of qubits of the circuit.
        n_shots: How often the simulation should be repeated.

    Returns:
        The expressiblity of the circuit.
    """
    if seed is not None:
        np.random.seed(seed)

    if n_bins is None:
        n_bins = 75  # as used in the paper

    # estimate fidelities
    fidelities = []
    for _ in range(n_shots):
        params = np.random.rand(2, n_params) * 2 * np.pi
        rho1 = circuit_simulator(params[0])
        rho2 = circuit_simulator(params[1])

        fidelities.append(state_fidelity(rho1, rho2))

    # Convert fidelities to a histogram
    binning = np.linspace(0, 1, n_bins + 1)
    bin_centers = np.array([(binning[i + 1] + binning[i]) / 2
                            for i in range(n_bins)])
    fids, _ = np.histogram(fidelities, bins=binning)
    fids = (fids / n_shots)  # normalize the histogram

    # Compute P_haar(F)
    P_haar = (2**n_qubits - 1) * (1 - bin_centers)**(2**n_qubits - 2)
    P_haar = P_haar / sum(P_haar)  # normalize

    # Compute Kullback-Leibler (KL) Divergence
    D_kl = 0
    for i in range(n_bins):
        value = fids[i]
        if (value > epstol) and (P_haar[i] > epstol):
            D_kl += value * np.log(value / P_haar[i])

    return D_kl
