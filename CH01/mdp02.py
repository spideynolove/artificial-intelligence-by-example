# -*- coding: utf-8 -*-
# Markov Decision Process (MDP) - The Bellman equations adapted to Q Learning.
# Reinforcement Learning with the Q action-value(reward) function.

import numpy as ql

# R is The Reward Matrix for each state
R = ql.matrix([ [0,0,0,0,1,0],
		            [0,0,0,1,0,1],
		            [0,0,100,1,0,0],
	             	[0,1,1,0,1,0],
		            [1,0,0,1,0,0],
		            [0,1,0,0,0,0] ])

# Q is the Learning Matrix in which rewards will be learned/stored
Q = ql.matrix(ql.zeros([6,6]))

"""##  The Learning rate or training penalty"""
# Gamma : It's a form of penalty or uncertainty for learning
# If the value is 1 , the rewards would be too high.
# This way the system knows it is learning.
gamma = 0.8

"""## Initial State"""
# agent_s_state. The agent the name of the system calculating
# s is the state the agent is going from and s' the state it's going to
# this state can be random or it can be chosen as long as the rest of the choices
# are not determined. Randomness is part of this stochastic process
agent_s_state = 5


"""## The random choice of the next state"""
def possible_actions(state):
    """
    The possible "a" actions when the agent is in a given state
    """
    current_state_row = R[state,]
    possible_act = ql.where(current_state_row >0)[1]
    return possible_act

# Get available actions in the current state
PossibleAction = possible_actions(agent_s_state)


def ActionChoice(available_actions_range):
    """
    This function chooses at random which action to be performed within the range of all the available actions.
    """
    if(sum(PossibleAction)>0):
        next_action = int(ql.random.choice(PossibleAction,1))
    if(sum(PossibleAction)<=0):
        next_action = int(ql.random.choice(5,1))
    return next_action

# Sample next action to be performed
action = ActionChoice(PossibleAction)


"""## The Bellman Equation"""
def reward(current_state, action, gamma):
    """
    A version of the Bellman equation for reinforcement learning using the Q function
    This reinforcement algorithm is a memoryless process
    The transition function T from one state to another
    is not in the equation below.  T is done by the random choice above
    """
    Max_State = ql.where(Q[action,] == ql.max(Q[action,]))[1]

    if Max_State.shape[0] > 1:
        Max_State = int(ql.random.choice(Max_State, size = 1))
    else:
        Max_State = int(Max_State)
    MaxValue = Q[action, Max_State]
    
    # The Bellman MDP based Q function
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue

# Rewarding Q matrix
reward(agent_s_state,action,gamma)


"""## Running the training episodes randomly"""
for i in range(50000):
    """
    Learning over n iterations depending on the convergence of the system
    A convergence function can replace the systematic repeating of the process
    by comparing the sum of the Q matrix to that of Q matrix n-1 in the
    previous episode
    """
    current_state = ql.random.randint(0, int(Q.shape[0]))
    PossibleAction = possible_actions(current_state)
    action = ActionChoice(PossibleAction)
    reward(current_state,action,gamma)
    
# Displaying Q before the norm of Q phase
print("Q  :")
print(Q)

# Norm of Q
print("Normed Q :")
print(Q/ql.max(Q)*100)

"""# Improving the program by introducing a decision-making process"""
nextc=-1
nextci=-1
conceptcode=["A","B","C","D","E","F"]
origin=int(input("index number origin(A=0,B=1,C=2,D=3,E=4,F=5): "))
print("Concept Path")
print("->",conceptcode[int(origin)])
for se in range(0,6):
    if(se==0):
        po=origin
    if(se>0):
        po=nextci
        #print("se:",se,"po:",po)
    for ci in range(0,6):
        maxc=Q[po,ci]
        #print(maxc,nextc)
        if(maxc>=nextc):
            nextc=maxc
            nextci=ci
            #print("next c",nextc)
    if(nextci==po):
        break;
    #print("present origin",po,"next c",nextci," ",nextc," ",conceptcode[int(nextci)])
    print("->",conceptcode[int(nextci)])
