# Computes entanglement capability of circuits
# based on: https://arxiv.org/abs/1905.10876

import numpy as np
from qiskit.quantum_info import partial_trace, DensityMatrix, Statevector

from icecream import ic

def entanglement_capability(circuit_simulator,
                   n_params,
                   n_shots=1000,
                   seed=None):
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

    # estimate fidelities
    entanglements = []
    for _ in range(n_shots):
        params = np.random.rand(n_params) * 2 * np.pi
        rho = circuit_simulator(params)

        entanglements.append(entanglement_measure(rho))
    return np.mean(np.array(entanglements))


def entanglement_measure(density_matrix):
    """Calculates Meyer and Wallach's entanglement measure Q.

    See https://arxiv.org/abs/quant-ph/0108104 for more.

    Args:
        density_matrix: qiskit DensityMatrix or Statevecrepresentation of the state to
            calculate Q of.
    
    Returns:
        Q_value: The Meyer-Wallach entanglement measure of density_matrix.
    """
    if not isinstance(density_matrix, DensityMatrix):
        try:
            density_matrix = DensityMatrix(density_matrix)
        except Exception:
            raise TypeError(f"Expected density_matrix of type {DensityMatrix} or convertable to {DensityMatrix}, but received {type(density_matrix)} instead.")

    n_qubits = density_matrix.num_qubits

    entanglement_sum = 0

    # Using Brennen's form of the MW-measure
    # https://arxiv.org/abs/quant-ph/0305094
    for k in range(n_qubits):
        rho_k = partial_trace(density_matrix, [k]).data
        rho_k_squared = rho_k**2
        entanglement_sum += rho_k_squared.trace()
    Q_value = 2 * (1 - entanglement_sum.real / n_qubits)

    # numerically a bit troublesome so we must clamp
    if Q_value < 0:
        return 0
    elif Q_value > 1:
        return 1
    return Q_value