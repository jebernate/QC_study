#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    np.random.seed(123)
    num_qubits = len(H.wires)
    
    # Define the ansatz
    def variational_ansatz(params):
        qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
    
    # Define the device
    dev = qml.device("lightning.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def penalty_circuit(params0, params1):
        variational_ansatz(params0)
        qml.adjoint(variational_ansatz)(params1)
        return qml.probs(wires=range(num_qubits))
    
    var_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=num_qubits)
    shape = (3, var_shape[0], var_shape[1], var_shape[2])

    final_params = np.zeros(shape, requires_grad = False)

    @qml.qnode(dev)
    def expval_H(params):
        variational_ansatz(params)
        return qml.expval(H)
    
    # Here we choose alpha_i such that it is greater than the next excited state energy.
    # We make the assumption alpha_i=2 for all i, and update it if neccessary
    alpha = [1, 1]
    def cost_fn(params, final_params, energies, state):
        penalty = 0
        for i in range(state):
            penalty += (alpha[i]-energies[i])*penalty_circuit(params, final_params[i])[0]
        return expval_H(params) + np.abs(penalty)
    
    # Optmization loop
    
    opt = qml.NesterovMomentumOptimizer(0.08)
    num_iter = 200

    for state in [0, 1, 2]:
        params = np.zeros(var_shape) + np.random.standard_normal(var_shape)*0.1
        for iteration in range(num_iter):
            params,_, _, _ = opt.step(cost_fn, params, final_params, energies, state)
            print(iteration, expval_H(params))
        final_params[state] = params
        energies[state] = expval_H(params)
    
    energies = np.sort(energies) # In case another eigenvalue is found first
    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
