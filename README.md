# Quantum Machine Learning with Qiskit

A collection of Jupyter notebooks exploring quantum computing fundamentals and quantum machine learning using [Qiskit](https://qiskit.org/).

## Overview

This project covers the basics of quantum circuits and applies quantum computing to machine learning classification tasks, including a full implementation of a **Variational Quantum Classifier (VQC)**.

## Notebooks

| Notebook | Description |
|---|---|
| [`q.ipynb`](q.ipynb) | Introduction to single-qubit circuits: Hadamard gate, measurement, and histogram visualization using the Qasm simulator. |
| [`q1.ipynb`](q1.ipynb) | Two-qubit quantum circuits: Bell state preparation using a Hadamard gate and a CNOT gate, with measurement and result visualization. |
| [`qML.ipynb`](qML.ipynb) | Quantum machine learning fundamentals using Qiskit — covers quantum circuit construction for ML workflows. |
| [`qlerning.ipynb`](qlerning.ipynb) | Connecting to IBM Quantum hardware via IBMQ and running circuits on real quantum devices. |
| [`Variational_Quantum_Classifier_Solutions_Starter.ipynb`](Variational_Quantum_Classifier_Solutions_Starter.ipynb) | End-to-end VQC implementation covering a synthetic 2D dataset and the Iris dataset classification using `ZZFeatureMap`, `RealAmplitudes`, COBYLA optimizer, and `qiskit_machine_learning`. |

## Requirements

- Python 3.8+
- [Qiskit](https://qiskit.org/documentation/getting_started.html)
- [Qiskit Aer](https://qiskit.org/ecosystem/aer/) (local simulator)
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/)
- [scikit-learn](https://scikit-learn.org/)
- NumPy, Matplotlib, Jupyter

Install all dependencies with:

```bash
pip install qiskit qiskit-aer qiskit_machine_learning scikit-learn numpy matplotlib pylatexenc jupyter
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Mohamed-ahmed-shokry/quantum-machine-learning-with-qiskit.git
   cd quantum-machine-learning-with-qiskit
   ```

2. Install dependencies (see [Requirements](#requirements) above).

3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open any of the notebooks and run the cells.

## Key Concepts

- **Quantum Circuits** — building blocks of quantum computation using gates such as Hadamard (H) and CNOT.
- **Bell States** — entangled two-qubit states demonstrated in `q1.ipynb`.
- **Quantum Simulation** — using Qiskit Aer's `qasm_simulator` to run circuits locally.
- **Variational Quantum Classifier (VQC)** — a hybrid classical-quantum ML algorithm that uses:
  - A **feature map** (ZZFeatureMap) to encode classical data into quantum states.
  - A **variational ansatz** (RealAmplitudes) as the trainable quantum circuit.
  - A classical **optimizer** (COBYLA) to minimize the cross-entropy loss.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
