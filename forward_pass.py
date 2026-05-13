"""
Deep Neural Network Forward Pass From Scratch Using NumPy

What this demonstrates:
- Creating a multi-layer neural network manually
- Weight and bias initialization
- Matrix multiplication between layers
- ReLU activation
- Shape tracking through the network

Architecture:
Input (5)
    ↓
Hidden Layer 1 (6)
    ↓
Hidden Layer 2 (8)
    ↓
Hidden Layer 3 (8)
    ↓
Output Layer (3)

"""

import numpy as np

# Setting a random seed ensures the same random
np.random.seed(42)

# Network Configuration

batch_size = 5      # Number of samples processed together
input_size = 5      # Number of input features
hidden1_size = 6    # Neurons in hidden layer 1
hidden2_size = 8    # Neurons in hidden layer 2
hidden3_size = 8    # Neurons in hidden layer 3
output_size = 3     # Number of output neurons

# Simulated Input Data

# Shape: (batch_size, input_size)
# Here we generate random input data for testing.
X = np.random.randn(batch_size, input_size)

print("Input Shape:", X.shape)

# Weight Initialization

# We initialize small random weights to prevent
# activations from becoming too large early on.

W1 = np.random.randn(input_size, hidden1_size) * 0.1
b1 = np.zeros(hidden1_size)

W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
b2 = np.zeros(hidden2_size)

W3 = np.random.randn(hidden2_size, hidden3_size) * 0.1
b3 = np.zeros(hidden3_size)

W4 = np.random.randn(hidden3_size, output_size) * 0.1
b4 = np.zeros(output_size)

# Print Weight Shapes

print("\nWeight Matrix Shapes:")
print("W1:", W1.shape)
print("W2:", W2.shape)
print("W3:", W3.shape)
print("W4:", W4.shape)

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# FORWARD PASS

# Layer 1

hidden1 = X @ W1 + b1
hidden1_activated = relu(hidden1)

print("\nHidden Layer 1 Shape:", hidden1_activated.shape)

# Layer 2

hidden2 = hidden1_activated @ W2 + b2
hidden2_activated = relu(hidden2)

print("Hidden Layer 2 Shape:", hidden2_activated.shape)

# Layer 3

hidden3 = hidden2_activated @ W3 + b3
hidden3_activated = relu(hidden3)

print("Hidden Layer 3 Shape:", hidden3_activated.shape)

# Output Layer

output = hidden3_activated @ W4 + b4

print("Output Shape:", output.shape)

# Final Output
# These are raw output scores (logits).
# In real AI systems:
# - Softmax is often used for classification
# - Sigmoid for binary classification
# - No activation for regression

print("\nFinal Output (Logits):")
print(output)
