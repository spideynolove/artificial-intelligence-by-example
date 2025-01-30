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

## Deep Dive

### Chapter Topics Explained  
This chapter focuses on **preprocessing raw data into a reward matrix (R)** for Markov Decision Processes (MDPs) using neural network components. The pipeline transforms warehouse AGV (Automated Guided Vehicle) sensor data into a structured decision-making framework.  

---

#### **1. McCulloch-Pitts Neuron**  
- **Role**: Processes raw input data (e.g., AGV sensor readings) into actionable signals.  
- **Implementation** (`MCP.py`):  
  - Inputs: Feature vector (e.g., `[10, 2, 1., 6., 2.]` for AGV speed, location, etc.).  
  - Weights/Bias: Adjustable parameters to prioritize inputs.  
  - **Logistic Sigmoid**: Squashes output into `[0,1]` to represent activation probability.  
  - Example: High activation (`0.99999`) indicates AGV availability for task assignment.  

---

#### **2. Logistic Classifiers**  
- **Purpose**: Begins neural processing by mapping inputs to probabilistic outputs.  
- **Key Tool**: Logistic sigmoid ensures non-linearity and interpretability.  

---

#### **3. Softmax Function**  
- **Role** (`SOFTMAX.py`): Normalizes raw scores (e.g., AGV route options) into a probability distribution.  
  ```python
  softmax(x) = exp(x) / Σ exp(x)  # Ensures outputs sum to 1
  ```  
- **Application**: Prioritizes AGV actions (e.g., choosing between storage zones).  

---

#### **4. One-Hot Encoding**  
- **Purpose**: Selects the **highest-probability action** from softmax outputs.  
  - Converts the maximum value to `1` and others to `0`.  
  - Example: AGV picks the optimal path (e.g., `[0, 0, 1, 0, 0]`).  

---

#### **5. Reward Matrix (R) Construction**  
- **Goal**: Maps valid AGV transitions (e.g., allowed paths between warehouse zones).  
- **Process**:  
  1. Neuron processes raw data into activations.  
  2. Softmax normalizes activations into probabilities.  
  3. One-hot encoding selects the best action.  
  4. Actions populate R (e.g., `R[A][B] = 1` if AGV can move from A to B).  

---

#### **6. AGV Warehouse Example**  
- **Challenge**: Preprocess sensor data (e.g., location, inventory levels) into R.  
- **Outcome**: R defines valid AGV movements (e.g., `C→D` allowed, `C→A` blocked).  

---

### **Key Workflow**  
**Raw Data** → **McCulloch-Pitts Neuron** → **Logistic Sigmoid** → **Softmax** → **One-Hot** → **Reward Matrix**  

This pipeline ensures data standardization, critical for reliable MDP-based reinforcement learning in warehouse automation.