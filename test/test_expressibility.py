from triple_e import expressibility

import pennylane as qml
from qiskit.quantum_info import DensityMatrix
import numpy as np

import pytest


def u2_reuploading(weights, n_qubits=8):
    for feature, qubit in zip(range(0, 2 * n_qubits, 2), range(n_qubits)):
        #Pennylane decomposes to Ry,Rz or Rφ(phase) gates:
        qml.U3(np.pi / 2, weights[feature], weights[feature + 1], wires=qubit)
    for qubit in range(n_qubits):
        if qubit == n_qubits - 1:
            break
        qml.CNOT(wires=[qubit, qubit + 1])

    for feature, qubit in zip(range(0, 2 * n_qubits, 2), range(n_qubits)):
        #Pennylane decomposes to Ry,Rz or Rφ(phase) gates:
        qml.U3(weights[feature], weights[feature + 1], 0, wires=qubit)

    return qml.density_matrix(np.arange(0, n_qubits))


def test_expressibility_u2reuploading():
    n_qubits = 8
    n_params = 16

    dev = qml.device('default.qubit', wires=n_qubits)
    qnode = qml.QNode(u2_reuploading, dev)

    circuit_simulator = lambda x: DensityMatrix(qnode(x))
    expressibility(circuit_simulator, n_params, n_qubits, n_shots=10)

    assert 1 == 1


# Check against 1-qubit examples from paper
def one_qubit_expr(circuit, n_params, n_shots):
    dev = qml.device('default.qubit', wires=1)
    qnode = qml.QNode(circuit, dev)
    return expressibility(qnode, n_params, 1, n_shots=n_shots)


def test_expressibility_single_qubit_idle():

    def idle_circuit(_):
        qml.Identity(wires=0)
        return qml.density_matrix(0)

    assert one_qubit_expr(idle_circuit, 1, 10) == pytest.approx(4.30, abs=0.1)


def test_expressibility_single_qubit_a():

    def circuit_a(weights):
        qml.Hadamard(wires=0)
        qml.RZ(weights[0], wires=0)
        return qml.density_matrix(0)

    assert one_qubit_expr(circuit_a, 1, 1000) == pytest.approx(0.22, abs=0.1)


def test_expressibility_single_qubit_b():

    def circuit_b(weights):
        qml.Hadamard(wires=0)
        qml.RZ(weights[0], wires=0)
        qml.RX(weights[1], wires=0)
        return qml.density_matrix(0)

    assert one_qubit_expr(circuit_b, 2, 1000) == pytest.approx(0.02, abs=0.1)


def test_expressibility_single_qubit_c():

    def circuit_c(weights):
        qml.Hadamard(wires=0)
        qml.U3(weights[0], weights[1], weights[2], wires=0)
        return qml.density_matrix(0)

    assert one_qubit_expr(circuit_c, 3, 1000) == pytest.approx(0.007, abs=0.1)
