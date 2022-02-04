from triple_e import entanglement_capability

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import pytest

# Check against 1-qubit examples from paper


def test_entangling_2_separable():

    def separable_circuit(params):
        qc = QuantumCircuit(2)
        qc.u(*params[:3], 0)
        qc.u(*params[3:], 1)
        return Statevector.from_instruction(qc)

    assert entanglement_capability(separable_circuit, 6,
                                   10) == pytest.approx(0)


def test_entangling_2_maximal():

    def bell_circuit(_):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        return Statevector.from_instruction(qc)

    assert entanglement_capability(bell_circuit, 1, 10) == pytest.approx(1)


def test_entangling_2_inbetween():

    def RY_CNOT_circuit(params):
        qc = QuantumCircuit(2)
        qc.ry(params[0], 0)
        qc.cnot(0, 1)
        return Statevector.from_instruction(qc)

    ent = entanglement_capability(RY_CNOT_circuit, 1, 1000)
    assert 0 < ent
    assert ent < 1


def test_entangling_4_GHZ():

    def GHZ4_circuit(_):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cnot(0, 1)
        qc.cnot(0, 2)
        qc.cnot(0, 3)
        return Statevector.from_instruction(qc)

    assert entanglement_capability(GHZ4_circuit, 1, 10) == pytest.approx(1)