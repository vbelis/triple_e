from triple_e import entanglement_capability

import pennylane as qml
from qiskit.quantum_info import DensityMatrix
import numpy as np

import pytest


# Check against 1-qubit examples from paper
def two_qubit_entangling(circuit, n_params, n_shots):
    dev = qml.device('default.qubit', wires=2)
    qnode = qml.QNode(circuit, dev)
    return entanglement_capability(qnode, n_params, n_shots=n_shots)


def test_entangling_2_separable():

    def separable_circuit(params):
        qml.U3(*params[:3], wires=0)
        qml.U3(*params[3:], wires=1)
        return qml.density_matrix([0, 1])

    assert two_qubit_entangling(separable_circuit, 6, 10) == pytest.approx(0)


def test_entangling_2_maximal():

    def bell_circuit(_):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.density_matrix([0, 1])

    assert two_qubit_entangling(bell_circuit, 1, 10) == pytest.approx(1)


def test_entangling_2_inbetween():

    def RY_CNOT_circuit(params):
        qml.RY(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.density_matrix([0, 1])

    ent = two_qubit_entangling(RY_CNOT_circuit, 1, 1000)
    assert 0 < ent
    assert ent < 1


def test_entangling_4_GHZ():

    def GHZ4_circuit(_):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[0, 3])
        return qml.density_matrix([0, 1, 2, 3])

    dev = qml.device('default.qubit', wires=4)
    qnode = qml.QNode(GHZ4_circuit, dev)
    ent = entanglement_capability(qnode, 1, n_shots=10)

    assert ent == pytest.approx(1)
