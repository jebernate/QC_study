#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    np.random.seed(1111)

    # Define the circuit
    def embedding_layer(feature_vector):
        qml.AngleEmbedding(feature_vector, wires=range(3), rotation='X')
    
    def classifier(params):
        qml.BasicEntanglerLayers(params, wires=range(3))
        
    dev = qml.device('default.qubit', wires=3, shots=1024)
    @qml.qnode(dev)
    def circuit(params, x):
        embedding_layer(x)
        classifier(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]
    
    # Define the variational classifer
    def softmax(x):
        sum = np.sum(np.exp(x))
        return np.exp(x)/sum
    
    def variational_classifier(weights, bias, x):
        return softmax(circuit(weights, x) + bias)
              
    def cross_entropy_loss(label, pred):
        return  -1*np.sum(label*np.log(pred+1e-5))
    
    # Define the cost function    
    def cost_fn(weights, bias, X, Y):
        cost = 0
        for i in range(Y.shape[0]):
            pred = variational_classifier(weights, bias, X[i])
            cost += cross_entropy_loss(Y[i], pred)
        return cost/Y.shape[0]
    
    # Map the labels to one-hot vectors
    mapping = {-1:0, 0:1, 1:2}
    def one_hot_encoding(Y):
        Y_OH = np.zeros((Y.shape[0], 3),)
        for i in range(Y.shape[0]):
            Y_OH[i, mapping[int(Y[i])]] = 1
        return Y_OH
    
    # Optimization loop
    
    # Initialize the weights and bias
    shape = qml.BasicEntanglerLayers.shape(n_layers=3, n_wires=3)
    weights = np.zeros(shape) + np.random.standard_normal(shape)*0.1
    bias = 0
    
    batch_size = 5
    Y_train_OH = one_hot_encoding(Y_train)

    num_iter = 80
    optimizer =  qml.AdamOptimizer(0.02)
    
    for i in range(num_iter):
        batch_index = np.random.randint(0, len(Y_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = Y_train_OH[batch_index]
        weights, bias, _, _ = optimizer.step(cost_fn, weights, bias, X_batch, Y_batch)      
    
    def predict(weights, bias, X):
        return np.argmax(variational_classifier(weights, bias, X) ,axis=0) - 1

    predictions = predict(weights, bias, X_test)    
    # QHACK #
    
    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
