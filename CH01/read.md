# CH01 Code Summary

This directory contains implementations of Q-learning using the Bellman equation for reinforcement learning.

## Files

- **MDP01.ipynb** / **mdp01.py**: Basic Q-learning implementation
- **MDP02.ipynb** / **mdp02.py**: Enhanced Q-learning with decision-making process

## Key Components

### Q-Learning Implementation
- Uses a 6x6 reward matrix (R) and Q-learning matrix (Q)
- Implements the Bellman equation:  
  Q[current_state, action] = R[current_state, action] + gamma * MaxValue
- Gamma (γ) = 0.8 (learning rate/penalty factor)
- Performs 50,000 training iterations with random state transitions
- Includes functions for:
  - possible_actions(): Gets available actions from current state
  - ActionChoice(): Randomly selects next action
  - reward(): Updates Q-values using Bellman equation

### Enhanced Version (MDP02)
Adds a decision-making process that:
1. Takes user input for starting state (0-5)
2. Maps states to concept codes (A-F) 
3. Follows the optimal path by selecting actions with highest Q-values
4. Prints the concept path from starting state to terminal state

## Key Concepts
- Markov Decision Process (MDP)
- Reinforcement Learning
- Q-Learning
- Bellman Equation
- Reward Matrix
- Learning Rate (Gamma)
- State Transitions
- Decision Making Process

## Detailed Explanations

### Q-Learning Components
- **Reward Matrix (R)**: A predefined matrix that specifies immediate rewards for state transitions
  - Rows represent current states
  - Columns represent possible actions
  - Values indicate rewards for taking specific actions from specific states
  - Example: R[2,2] = 100 means moving from state 2 to state 2 yields reward 100

- **Q-Learning Matrix (Q)**: A matrix that stores learned values through training
  - Initially zeroed
  - Updated through Bellman equation
  - Represents expected future rewards for state-action pairs
  - Used to determine optimal policy

### Bellman Equation
- Core equation for Q-learning:
  Q(s,a) = R(s,a) + γ * max(Q(s',a'))
  - s: current state
  - a: current action
  - s': next state
  - a': next action
  - γ: discount factor (gamma)
- Balances immediate reward (R) with future rewards (Q)
- Enables learning optimal policy through iterative updates

### States and Actions
- **State**: Current situation/environment of the agent
  - Represented numerically (0-5 in this implementation)
  - Can be random or chosen initially
- **State Transition**: Movement from one state to another
  - Determined by action taken
  - May be stochastic (random)
- **Possible Actions**: Available moves from current state
  - Determined by non-zero values in reward matrix row
- **Action**: Specific move chosen by agent
  - Selected randomly during training
  - Selected based on Q-values during decision making

### Learning Parameters
- **Gamma (γ)**: Discount factor (0.8 in this case)
  - Controls importance of future rewards
  - Lower values favor immediate rewards
  - Must be between 0 and 1
- **Learning Rate/Penalty**:
  - Controls how quickly Q-values are updated
  - Acts as penalty to prevent overestimation of rewards
  - Higher values lead to faster learning but less stability

### Markov Decision Process (MDP)
- Mathematical framework for decision making
- Key properties:
  - Markov Property: Future depends only on current state
  - States: Discrete set of possible situations
  - Actions: Possible moves at each state
  - Transitions: Probabilistic state changes
  - Rewards: Immediate feedback for actions
- Used as foundation for Q-learning algorithm

## Q&A

1. **Markov Decision Process (MDP)**: A mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. It involves states, actions, rewards, and transition probabilities.

2. **The Bellman Equations**: A set of equations used in dynamic programming and reinforcement learning to solve MDPs. They break down the decision problem into smaller subproblems, optimizing the value function or Q-function.

3. **Q Learning**: A model-free reinforcement learning algorithm that learns the value of actions in states (Q-values) to maximize cumulative rewards. It uses the Bellman equation to update Q-values iteratively.

4. **The Bellman equations adapted to Q Learning**: The Bellman equation is adapted to update Q-values in Q Learning. It is used because it provides a recursive way to estimate the optimal action-value function. Pros: Simple, effective for model-free learning. Cons: Can be slow to converge, requires exploration. Alternatives: SARSA, Policy Gradient methods.

5. **Q action-value(reward) function**: It estimates the expected cumulative reward of taking an action in a state and following the optimal policy thereafter. Used in RL to guide decision-making. Other functions include:
   - **Value Function (V)**: Estimates the expected cumulative reward from a state.
   - **Advantage Function (A)**: Measures the advantage of taking a specific action over the average action.

   | Function Type       | Pros                                      | Cons                                      |
   |---------------------|-------------------------------------------|-------------------------------------------|
   | Q-function          | Directly estimates action values          | Can be computationally expensive          |
   | Value Function (V)  | Simpler, less computation                 | Doesn't provide action-specific guidance  |
   | Advantage Function  | Combines benefits of Q and V functions    | More complex to implement                 |

6. **Reward Matrix, Learning Matrix, Gamma**:
   - **Reward Matrix (R)**: Defines immediate rewards for state-action pairs.
   - **Learning Matrix (Q)**: Stores learned Q-values.
   - **Gamma**: Discount factor, balancing immediate and future rewards.
   - The 6x6 matrix represents 6 states and 6 possible actions in each state.

7. **Other Parts**:
   - **agent_s_state**: Current state of the agent.
   - **possible_actions**: Finds available actions in the current state.
   - **ActionChoice**: Randomly selects an action from available actions.
   - **action**: The chosen action.
   - **reward**: Updates the Q-value using the Bellman equation.

8. **50000 Iterations**: These iterations train the Q-learning algorithm to converge to optimal Q-values. It helps achieve the purpose of learning the best actions for each state.

9. **Improvement in mdp02**: The decision-making process in mdp02 improves mdp01 by introducing a path-finding mechanism based on the learned Q-values, allowing the agent to follow a sequence of states to maximize rewards.

## Q&A thinking

- **Markov Decision Process (MDP):** A framework for modeling decision-making in environments with states, actions, transitions, and rewards. It follows the Markov property (future depends only on the current state/action).  
- **Bellman Equations:** Equations used to compute optimal policies by breaking state/action values into immediate rewards + discounted future values.  
- **Q Learning:** A model-free RL algorithm that learns action-values (Q-values) to maximize cumulative rewards via trial-and-error.  

**Bellman Equations in Q Learning:**  
- **Adaptation:** Modified to update Q-values:  
  `Q(s,a) = R(s,a) + γ * max(Q(s',a'))`.  
- **Pros:** No environment model needed.  
- **Cons:** Slow in large state spaces.  
- **Alternatives:** SARSA, Policy Gradients.  

**Q Action-Value Function:**  
- **Purpose:** Evaluates actions in states for optimal policy.  
- **Why RL?** Directly links actions to rewards.  
- **Alternatives:**  
  | Function       | Pros                     | Cons                      |  
  |----------------|--------------------------|---------------------------|  
  | Q-value        | Action-specific          | High memory (Q-table)     |  
  | State-value (V)| Simpler                  | No action guidance        |  

**Key Components:**  
- **Reward Matrix (R):** Immediate rewards for state-action pairs.  
- **Learning Matrix (Q):** Stores learned Q-values.  
- **Gamma (γ):** Discount factor (0.8) balances immediate vs. future rewards.  
- **6x6 Matrix:** Represents 6 states (e.g., A-F in the code).  

**Other Parts:**  
- **agent_s_state:** Current state of the agent.  
- **possible_actions:** Lists valid actions from a state.  
- **ActionChoice:** Randomly selects an action (exploration).  
- **reward():** Updates Q-values using Bellman equation.  

**50000 Iterations:** Trains the Q-matrix to converge to optimal values. Achieves the goal of learning the best policy.  

**Improvement in mdp02:** Adds a decision-making loop to follow the highest Q-values (exploitation), creating an optimal path from user input. Enhances mdp01 by replacing random exploration with informed choices.

## Deep Dive

### **1. Objective**  
The explicit goal is to **learn an optimal policy** that enables an agent to navigate a 6-state environment (A-F) to maximize cumulative rewards, specifically reaching the high-reward state (C, reward=100) efficiently.  

---

### **2. Success Criteria**  
- **Quantitative Metrics**:  
  - Q-matrix convergence (stable values after training).  
  - Shortest path from any origin to state C (e.g., F → D → C in 2 steps).  
  - Normed Q-values ≥ 90% of maximum Q-value for critical transitions (e.g., D→C).  
- **Qualitative Indicators**:  
  - No loops in the decision-making path (e.g., A → E → D → C, not A → E → A).  
  - Consistent prioritization of high-reward actions during exploitation.  

---

### **3. Validation Process**  
**Step 1: Q-Matrix Stability Check**  
- Compare Q-matrices across training epochs (e.g., after 50k vs. 100k iterations).  
- Declare convergence if `mean(Q_new - Q_old) < 1e-5`.  

**Step 2: Path Efficiency Testing**  
- Initialize agent from all 6 states and track paths using `mdp02`’s decision-making loop.  
- Validate against ground-truth optimal paths (e.g., F→D→C, B→C).  

**Step 3: Reward Consistency**  
- Ensure transitions leading to state C have the highest Q-values (e.g., `Q[D,C] > Q[D,B]`).  

**Example Framework**:  
```python
def validate_optimal_path(origin, expected_path):
    agent_path = run_decision_loop(origin)  # From mdp02
    assert agent_path == expected_path, f"Path {agent_path} suboptimal."

validate_optimal_path(origin=5, expected_path=["F", "D", "C"])
```

---

### **4. Iterative Improvement**  
**Adjustments (Priority Order)**:  
1. **Gamma (γ)**: Increase to 0.9 if agent undervalues future rewards (e.g., avoids D→C).  
2. **Training Iterations**: Extend beyond 50k if Q-values fluctuate.  
3. **Exploration Strategy**: Replace purely random `ActionChoice` with ε-greedy (e.g., 20% random actions during training).  
4. **Reward Matrix Tweaks**: Add small rewards to critical transitions (e.g., D→C → +10 bonus).  

**Example Refinement**:  
```python
# Modified ActionChoice with ε-greedy
def ActionChoice(state, ε=0.2):
    if ql.random.uniform() < ε:
        return random_action(state)  # Explore
    else:
        return argmax(Q[state, :])   # Exploit
```

---

### **Validation Framework Example**  
```python
# Test Suite
test_cases = {
    0: ["A", "E", "D", "C"],  
    5: ["F", "D", "C"],        
    3: ["D", "C"]              
}

for origin, expected_path in test_cases.items():
    actual_path = simulate_agent(origin)
    assert actual_path == expected_path, f"Failed for origin {origin}."
```  

This ensures alignment with the problem’s goal of **reward-maximizing navigation** while enabling systematic refinement.