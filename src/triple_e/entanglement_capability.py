# Computes entanglement capability of circuits
# based on: https://arxiv.org/abs/1905.10876

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector


def entanglement_capability(circuit_simulator,
                            n_params,
                            n_shots=1000,
                            data=None,
                            seed=None):
    """Computes entanglement capability for a circuit.

    Args:
        circuit_simulator: A function that takes n_params and returns a qiskit
            Statevector or DensityMatrix.
        n_params: The number of parameters circuit_simulator accepts. Presumed
            to be uniformly distributed in [0, 2pi]
        n_shots: How often the simulation should be repeated.
        data: Array of data for the case of data-based expressibility computation. The 
              values of the circuit parameters are sampled from the data distribution
              instaed of uniformly from [0, 2pi].

    Returns:
        The expressiblity of the circuit.
    """
    if seed is not None:
        np.random.seed(seed)
    n_shots = int(n_shots)

    # estimate fidelities
    entanglements = []
    for _ in range(n_shots):
        if data is not None:
            params = data[np.random.choice(data.shape[0], size=1)].flatten()
        else:
            params = np.random.rand(n_params) * 2 * np.pi
        #params = np.random.rand(n_params) * 2 * np.pi
        rho = circuit_simulator(params)

        entanglements.append(entanglement_measure(rho))
    return np.mean(np.array(entanglements))


def cast_to_statevector(state):
    if isinstance(state, Statevector):
        return state
    try:
        state = Statevector(state)
        return state
    except Exception:
        try:
            state = DensityMatrix(state)
            return state.to_statevector()
        except Exception:
            raise TypeError(
                f"Expected state of type {Statevector} or convertable to {Statevector}, but received {type(state)} instead."
            )


def entanglement_measure(rho):
    """Calculates Meyer and Wallach's entanglement measure Q.

    See https://arxiv.org/abs/quant-ph/0108104 for more.

    Args:
        rho: qiskit Statevector (or convertable) representation of the state to
            calculate Q of.

    Returns:
        Q_value: The Meyer-Wallach entanglement measure of density_matrix.
    """

    rho = cast_to_statevector(rho)
    n_qubits = rho.num_qubits
    entanglement_sum = 0

    rho_data = rho.data
    for k in range(n_qubits):
        # Elements of the statevector for which the kth qubit is 0/1 respectively
        k_zero_mask = (0 == np.arange(2**n_qubits) // 2**k % 2)
        k_one_mask = (1 == np.arange(2**n_qubits) // 2**k % 2)

        rho_k_zero = rho_data[k_zero_mask]
        rho_k_one = rho_data[k_one_mask]

        entanglement_sum += wedge_distance(rho_k_zero, rho_k_one)

    return 4 / n_qubits * entanglement_sum


def wedge_distance(u, v):
    """Calculates the wedge distance between input vectors u and v.

    Args:
        u: Vector 1
        v: Vector 2

    Returns:
        Wedge product of u and v.
    
    Remarks:
        Could be more efficient, but realistically speaking this function is
        not the bottleneck of the entanglement capability calculation.
    """
    n_it = np.size(u)
    sum = 0
    for i in range(1, n_it):
        for j in range(i):
            sum += np.abs(u[i] * v[j] - u[j] * v[i])**2
    return sum
