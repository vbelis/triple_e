from triple_e import expressibility

import pennylane as qml

import pytest


# Check against 1-qubit examples from paper
def one_qubit_expr(circuit, n_params, n_shots):
    dev = qml.device('lightning.qubit', wires=1)
    qnode = qml.QNode(circuit, dev)
    return expressibility(qnode, n_params, 1, n_shots=n_shots)


def test_expressibility_single_qubit_idle():

    def idle_circuit(_):
        qml.Identity(wires=0)
        return qml.density_matrix(0)

    assert one_qubit_expr(idle_circuit, 1, 1000) == pytest.approx(4.30,
                                                                  abs=0.1)


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


def test_expressibility_full_method():

    def circuit_c(weights):
        qml.Hadamard(wires=0)
        qml.U3(weights[0], weights[1], weights[2], wires=0)
        return qml.density_matrix(0)

    dev = qml.device('lightning.qubit', wires=1)
    qnode = qml.QNode(circuit_c, dev)

    assert expressibility(qnode, 3, 1, n_shots=1000,
                          method="pairwise") == pytest.approx(expressibility(qnode, 3, 1, n_shots=1000,
                          method="full"), abs=0.1)
