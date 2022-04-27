# Computes expressbility of circuits
# based on: https://arxiv.org/abs/1905.10876

import numpy as np
from qiskit.quantum_info import state_fidelity


def expressibility(circuit_simulator,
                   n_params,
                   n_qubits,
                   method="pairwise",
                   n_shots=1000,
                   seed=None,
                   n_bins=75,
                   return_histogram=False,
                   epstol=1e-18):
    """Computes expressibility for a circuit.

    Args:
        circuit_simulator: A function that takes n_params and returns a qiskit
            Statevector or DensityMatrix.
        n_params: The number of parameters circuit_simulator accepts. Presumed
            to be uniformly distributed in [0, 2pi]
        n_qubits: Number of qubits of the circuit.
        method: Method to use to estimate fidelity:
            "pairwise": Generates two samples, calculates their fidelity.
                Computationally expensive, but memory efficient.
            "full": Generates approximately sqrt(n_shots) samples, calculates
                their fidelity pairwise. Computationally efficient, but memory
                expensive.
        n_shots: How many fidelity samples to generate.
        n_bins: Number of equal-width bins.
        return_histogram: If `True`, additionally returns a tuple
            `(p_haar, p_circuit)`containing the normalized histogram data of the fidelity
            distributions.

    Returns:
        The expressiblity of the circuit.
    """
    if seed is not None:
        np.random.seed(seed)

    if n_bins is None:
        n_bins = 75  # as used in the paper
    n_shots = int(n_shots)
    n_bins = int(n_bins)

    # estimate fidelities
    fidelities = []

    if method == "pairwise":
        for _ in range(n_shots):
            params = np.random.rand(2, n_params) * 2 * np.pi
            rho1 = circuit_simulator(params[0])
            rho2 = circuit_simulator(params[1])

            fidelities.append(state_fidelity(rho1, rho2))
    elif method == "full":
        samples = []
        while len(fidelities) < n_shots:
            params = np.random.rand(n_params) * 2 * np.pi
            rho = circuit_simulator(params)

            for smpl in samples:
                fidelities.append(state_fidelity(rho, smpl))
                if len(fidelities) >= n_shots:
                    break
            samples.append(rho)
    else:
        raise ValueError("Invalid argument for method provided.")

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

    if return_histogram:
        return D_kl, (P_haar, fids)

    return D_kl
