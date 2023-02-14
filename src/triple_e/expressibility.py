# Computes expressbility of circuits
# based on: https://arxiv.org/abs/1905.10876

import numpy as np
from qiskit.quantum_info import state_fidelity, DensityMatrix
from warnings import warn


def expressibility(circuit_simulator,
                   n_params,
                   n_qubits=None,
                   n_shots=1000,
                   method="pairwise",
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
        n_qubits: *DEPRECATED* Number of qubits of the circuit. 
        n_shots: How many fidelity samples to generate.
        method: Method to use to estimate fidelity:
            "pairwise": Generates two samples, calculates their fidelity.
                Computationally expensive, but memory efficient.
            "full": Generates approximately sqrt(n_shots) samples, calculates
                their fidelity pairwise. Computationally efficient, but memory
                expensive.
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
            rho1 = circuit_simulator(params)

            for smpl in samples:
                fidelities.append(state_fidelity(rho1, smpl))
                if len(fidelities) >= n_shots:
                    break
            samples.append(rho1)
    else:
        raise ValueError("Invalid argument for method provided.")

    if n_qubits is not None: #TODO: Actually remove this in the future ;)
        warn(
            "Supplying `n_qubits` manually is deprecated and will be removed" +
            " in the future. The supplied value is ignored, and the correct" +
            " value is inferred from the state returned by `circuit_simulator`"
            + " instead.", DeprecationWarning)
    n_qubits = DensityMatrix(rho1).num_qubits

    # Convert fidelities to a histogram
    fids, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1))
    fids = (fids / sum(fids))  # normalize the histogram

    # Compute P_haar(F)
    CDF_haar = -(1 - bin_edges)**(2**n_qubits - 1)
    P_haar = np.array([CDF_haar[i + 1] - CDF_haar[i] for i in range(n_bins)])
    P_haar = P_haar / sum(P_haar)  # normalize

    # Compute Kullback-Leibler (KL) Divergence
    D_kl = 0
    for i in range(n_bins):
        value = fids[i]
        if (value > epstol) and (P_haar[i] > 0):
            D_kl += value * np.log(value / P_haar[i])
        elif value > 0:
            warn(
                "Dropping bin from calculation due to floating point accuracy."
                + " KL-divergence may be underestimated.")

    if return_histogram:
        return D_kl, (P_haar, fids)

    return D_kl
