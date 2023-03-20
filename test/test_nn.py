# TODO: import dependencies and write unit tests below

import numpy as np
import pytest

from nn import nn
from nn import preprocess


import numpy as np
import pytest
from nn import nn  # Assuming the class is saved in a file called neural_network.py

# Test configuration
nn_arch = [
    {'input_dim': 2, 'output_dim': 3, 'activation': 'relu'},
    {'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}
]
lr = 0.01
seed = 42
batch_size = 32
epochs = 10
loss_function = "binary_cross_entropy"

nn = nn.NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

def test_single_forward():
    # Test the _single_forward method to ensure it computes the correct output shape
    A_prev = np.array([[1, 2], [3, 4]])
    W_curr = nn._param_dict["W1"]
    b_curr = nn._param_dict["b1"]
    activation = "relu"

    A_curr, Z_curr = nn._single_forward(A_prev, W_curr, b_curr, activation)
    assert A_curr.shape == (3, 2)  # Expected output shape for this layer
    assert Z_curr.shape == (3, 2)  # Expected pre-activation shape for this layer

def test_forward():
    # Test the forward method to ensure it computes the correct output shape
    X = np.array([[1, 2], [3, 4]])
    y_hat, cache = nn.forward(X)
    assert y_hat.shape == (2, 1)  # Expected output shape

def test_single_backprop():
    # Test the _single_backprop method to ensure it computes the correct gradient shapes
    dA_curr = np.array([[1, 2], [3, 4], [5, 6]])
    W_curr = nn._param_dict["W1"]
    Z_curr = np.array([[1, 2], [3, 4], [5, 6]])
    A_prev = np.array([[1, 2], [3, 4]])
    activation_curr = "relu"

    dA_prev, dW_curr, db_curr = nn._single_backprop(dA_curr, W_curr, Z_curr, A_prev, activation_curr)
    assert dA_prev.shape == (2, 2)  # Expected shape for the gradient dA
    assert dW_curr.shape == (3, 2)  # Expected shape for the gradient dW
    assert db_curr.shape == (3, 1)  # Expected shape for the gradient db

def test_predict():
    # Test the predict method to ensure it computes the correct output shape
    X = np.array([[1, 2], [3, 4]])
    y_hat = nn.predict(X)
    assert y_hat.shape == (2, 1)  # Expected output shape

def test_binary_cross_entropy():
    # Test the _binary_cross_entropy method to ensure it computes the correct loss value
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    loss = nn._binary_cross_entropy(y, y_hat)
    assert np.isclose(loss, 0.5798184952529422)  # Expected loss value

def test_binary_cross_entropy_backprop():
    # Test the _binary_cross_entropy_backprop method to ensure it computes the correct gradient shape
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    assert dA.shape == (2, 2)  # Expected shape for the gradient dA

def test_mean_squared_error():
    # Test the _mean_squared_error method to ensure it computes the correct loss value
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    loss = nn._mean_squared_error(y, y_hat)
    expected_loss = 0.065  # Expected loss value
    assert np.isclose(loss, expected_loss)  # Check if the computed loss is close to the expected loss

def test_mean_squared_error_backprop():
    # Test the _mean_squared_error_backprop method to ensure it computes the correct gradient values
    y = np.array([[1, 0], [0, 1]])
    y_hat = np.array([[0.7, 0.2], [0.3, 0.8]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    expected_dA = np.array([[-0.3, 0.3], [0.2, -0.2]])  # Expected gradient values
    assert np.allclose(dA, expected_dA)  # Check if the computed gradient is close to the expected gradient


def test_sample_seqs():
    # Define test data with 4 sequences from rap1 positive set 
    seqs = ['ACATCCGTGCACCTCCG', 'ACACCCAGACATCGGGC', 'CCACCCGTACCCATGAC', 'GCACCCATACATTACAT']
    labels = [True, False, True, False]
    
    # Call the sample_seqs function
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    
    # Check if the output lengths are equal
    assert len(sampled_seqs) == len(sampled_labels)
    
    # Check if the number of True and False labels in the output is the same
    assert sampled_labels.count(True) == sampled_labels.count(False)
    
    # Check if the number of unique sequences in the output is less than or equal to the original number of sequences
    assert len(set(sampled_seqs)) <= len(seqs)

def test_one_hot_encode_seqs():
    # Define test data with 4 sequences from rap1 positive set 
    seqs = ['ACATCCGTGCACCTCCG', 'ACACCCAGACATCGGGC', 'CCACCCGTACCCATGAC', 'GCACCCATACATTACAT']

    # Call the one_hot_encode_seqs function
    encodings = preprocess.one_hot_encode_seqs(seqs)
    
    # Check if the output shape is as expected
    assert encodings.shape == (len(seqs), 17 * 4)

    # Check if the sum of each encoding equals the length of the input sequence
    for encoding in encodings:
        assert sum(encoding) == 17

    # Check if the number of rows in the output is equal to the number of input sequences
    assert encodings.shape[0] == len(seqs)
    

def test_one_hot_encode_accuracy():
    # Define more manageable test data with 4-base long sequences
    seqs = ['ATCG', 'GCTA', 'AAAT']

    # Call the one_hot_encode_seqs function
    encoded_seqs = preprocess.one_hot_encode_seqs(seqs)

    # Check if the one-hot encoding is correct for all sequences
    expected_encodings = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    ])

    assert np.array_equal(encoded_seqs, expected_encodings)