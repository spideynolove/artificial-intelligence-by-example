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
