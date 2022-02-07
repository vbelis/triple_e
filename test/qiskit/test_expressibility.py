from triple_e import expressibility

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import pytest

# Check against 1-qubit examples from paper


def test_expressibility_single_qubit_idle():

    def idle_circuit(_):
        qc = QuantumCircuit(1)
        qc.id(0)
        return Statevector.from_instruction(qc)

    assert expressibility(idle_circuit, 1, 1, 1000) == pytest.approx(4.30,
                                                                     abs=0.1)


def test_expressibility_single_qubit_a():

    def circuit_a(weights):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(weights[0], 0)
        return Statevector.from_instruction(qc)

    assert expressibility(circuit_a, 1, 1, 1000) == pytest.approx(0.22,
                                                                  abs=0.1)


def test_expressibility_single_qubit_b():

    def circuit_b(weights):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(weights[0], 0)
        qc.rx(weights[1], 0)
        return Statevector.from_instruction(qc)

    assert expressibility(circuit_b, 2, 1, 1000) == pytest.approx(0.02,
                                                                  abs=0.1)


def test_expressibility_single_qubit_c():

    def circuit_c(weights):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(*weights, 0)
        return Statevector.from_instruction(qc)

    assert expressibility(circuit_c, 3, 1, 1000) == pytest.approx(0.007,
                                                                  abs=0.1)
