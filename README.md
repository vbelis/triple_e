# triple_e

[![DOI](https://zenodo.org/badge/{github_id}.svg)](https://zenodo.org/badge/latestdoi/{455096142})

Metrics for Variational Quantum Circuits/Parameterized Quantum Circuits/Quantum Neural Networks.
- Expressibility [[1]](#1)
- Entanglement Capability [[1]](#1)
- Effective Dimension [[2]](#2)

## Installation

Simply install this package from GitHub/`master` by running:

```pip install https://github.com/QML-HEP/triple_e/archive/master.zip```

## Usage
This package aims to support both [Qiskit](https://qiskit.org/) and [PennyLane](https://pennylane.ai/).

### Expressibility
PennyLane:

```python
from triple_e import expressibility
import pennylane as qml

# define a parameterized circuit, returning a DensityMatrix/Statevector
def circuit_a(params):
    qml.Hadamard(wires=0)
    qml.RZ(params[0], wires=0)
    return qml.density_matrix(0)

n_params = 1
n_shots = 1000

# set up a quantum device with the appropriate amount of qubits/wires
dev = qml.device('default.qubit', wires=1)
qnode = qml.QNode(circuit_a, dev)

expressibility(qnode, n_params, n_shots)
```

Qiskit:

```python
from triple_e import expressibility
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# define a parameterized circuit, returning a DensityMatrix/Statevector
def circuit_b(weights):
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(weights[0], 0)
    qc.rx(weights[1], 0)
    return Statevector.from_instruction(qc)

n_params = 2
n_shots = 1000

expressibility(circuit_b, n_params, n_shots)
```

### Entanglement Capability
PennyLane:

```python
from triple_e import entanglement_capability
import pennylane as qml

# define a parameterized circuit, returning a DensityMatrix/Statevector
def separable_circuit(params):
    qml.U3(*params[:3], wires=0)
    qml.U3(*params[3:], wires=1)
    return qml.density_matrix([0, 1])

n_params = 6
n_shots = 10

# set up a quantum device with the appropriate amount of qubits/wires
dev = qml.device('default.qubit', wires=2)
qnode = qml.QNode(separable_circuit, dev)

entanglement_capability(qnode, n_params, n_shots)
```

Qiskit:

```python
from triple_e import entanglement_capability
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# define a parameterized circuit, returning a DensityMatrix/Statevector
def GHZ4_circuit(_):
    # no parameters are used, but we need to accept an argument nonetheless
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(0, 2)
    qc.cnot(0, 3)
    return Statevector.from_instruction(qc)

n_params = 0
n_shots = 10

entanglement_capability(GHZ4_circuit, n_params, n_shots)
```

## Credits
Originally developed by Till Muser at the Trapped Ion Quantum Information Group of ETH Zürich. Expressibility and entanglement capability calculation is partially based on code by Cenk Tüysüz.


## References
<a id="1">[1]</a> 
S. Sim, P.D. Johnson and A. Aspuru-Guzik;
*Expressibility and Entangling Capability of Parameterized Quantum Circuits for Hybrid Quantum-Classical Algorithms*;
Adv. Quantum Technol., 2 (2019): 1900070. https://doi.org/10.1002/qute.201900070

<a id="2">[2]</a> 
A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli and S. Woerner;
*The power of quantum neural networks*;
Nat Comput Sci 1, 403–409 (2021). https://doi.org/10.1038/s43588-021-00084-1
