# CH02 Code Summary

This directory contains implementations of fundamental neural network concepts:

## Files

- **MCP.py**: McCulloch-Pitts neuron implementation
- **SOFTMAX.py**: Softmax normalization and one-hot encoding

## Key Components

### McCulloch-Pitts Neuron (MCP.py)
- Implements a basic artificial neuron using TensorFlow
- Key components:
  - Input vector (x)
  - Weight vector (W)
  - Bias (b)
  - Logistic sigmoid activation function
- Demonstrates:
  - Neuron computation: y = σ(xW + b)
  - Availability calculation: 1 - y
- Example output shows high activation (0.99999) and low availability (0.00001)

### Softmax Function (SOFTMAX.py)
- Implements softmax normalization in multiple versions:
  1. Explicit step-by-step implementation
  2. Compact implementation
  3. Numpy-based function implementation
- Key features:
  - Converts raw scores to probability distribution
  - Ensures output values sum to 1
  - Includes one-hot encoding demonstration
- Demonstrates:
  - Softmax calculation: softmax(x) = exp(x) / sum(exp(x))
  - One-hot encoding of highest probability value

## Detailed Explanations

### McCulloch-Pitts Neuron
- **Input Vector (x)**: Contains input features
  - Example: [10, 2, 1., 6., 2.]
- **Weight Vector (W)**: Contains connection weights
  - Example: [.1, .7, .75, .60, .20]
- **Bias (b)**: Additional parameter for model flexibility
  - Example: 1.0
- **Activation Function**: Logistic sigmoid
  - σ(z) = 1 / (1 + exp(-z))
  - Maps output to range (0,1)
- **Neuron Output**: Represents activation level
  - High value (close to 1) indicates strong activation
  - Low value (close to 0) indicates weak activation

### Softmax Function
- **Purpose**: Normalizes input vector to probability distribution
- **Key Properties**:
  - Output values range (0,1)
  - Sum of outputs equals 1
  - Amplifies differences between values
- **Applications**:
  - Multi-class classification
  - Probability estimation
  - Neural network output layers
- **One-Hot Encoding**:
  - Converts highest probability to 1
  - Sets all other values to 0
  - Useful for classification tasks

## Mathematical Foundations

### McCulloch-Pitts Neuron
- Neuron output calculation:
  z = x₁w₁ + x₂w₂ + ... + xₙwₙ + b
  y = σ(z) = 1 / (1 + exp(-z))

### Softmax Function
- For vector x = [x₁, x₂, ..., xₙ]:
  softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
- Properties:
  - 0 ≤ softmax(xᵢ) ≤ 1
  - Σ softmax(xᵢ) = 1
  - Preserves relative ordering of inputs

## Implementation Notes

### MCP.py
- Uses TensorFlow for variable management
- Implements logistic sigmoid manually
- Demonstrates basic neuron computation
- Shows availability calculation (1 - activation)

### SOFTMAX.py
- Provides multiple implementation styles:
  1. Explicit step-by-step
  2. Compact version
  3. Numpy-based function
- Includes one-hot encoding demonstration
- Shows intermediate calculations for clarity
