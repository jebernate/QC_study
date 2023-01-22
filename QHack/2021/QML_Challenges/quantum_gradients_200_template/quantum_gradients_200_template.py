#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    
    # Calculate the gradients. We store the forward and backward computations with a shift of pi/2.
    # In this way we are able to reuse them for the diagonal elements in the hessian matrix.
    
    f_forward = np.zeros([5], dtype=np.float64)
    f_backward = np.zeros([5], dtype=np.float64)
    
    shift = np.pi/2
    
    for i in range(5): 
        basis_vectors = np.eye(5)
        f_forward[i] = circuit(weights + basis_vectors[i]*shift)
        f_backward[i] = circuit(weights - basis_vectors[i]*shift)
        gradient[i] = f_forward[i]-f_backward[i]
    
    gradient = gradient/2
 
    # Calculate the hessian.
    
    shift = np.pi/4 # This value of the shifts allows us to calculate the diagonal elements as follows.
    
    # Diagonal-elements
    
    f_central = circuit(weights)
    
    for i in range(5):
        hessian[i,i] = (f_forward[i]- 2*f_central + f_backward[i])/2
    
    # Off-diagonal elements. Here we use the fact that the hessian is symmetric
    # (assuming continuous second partial derivatives.)
    
    for i in range(1,5):
        for j in range(i):
            hessian[i,j] = ((circuit(weights + basis_vectors[i]*shift + basis_vectors[j]*shift)-
                            circuit(weights - basis_vectors[i]*shift + basis_vectors[j]*shift))-
                            (circuit(weights + basis_vectors[i]*shift - basis_vectors[j]*shift)-
                             circuit(weights - basis_vectors[i]*shift - basis_vectors[j]*shift)))/2
            hessian[j,i] = hessian[i,j]
        
    # QHACK #

    return gradient, hessian


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        # diff_method,
        sep=","
    )
