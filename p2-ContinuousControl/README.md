# p2-ContinuousControl

### Introduction
An agent is trained to move a double-jointed arm to target locations. Considering the first version, the primary objective is to get an average score of +30 over 100 consecutive episodes. The Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic (TD3) are utilized for the task.

### Project details
An agent is trained to move a double-jointed arm to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. The observation space consists of $33$ variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and +1.

### Getting Started
The project is completed in the Udacity Workspaces. However in order to run on local machine, the following packages are required:
```
matplotlib
numpy
unityagents==0.4.0
torch
``` 

## Files
- `Navigation.ipynb`: Notebook used to train and test the agent 
- `dqn_agent.py`: Agent class for Deep Q-network and Double Deep Q-network
- `model.py`: Model architecture
- `checkpoint.pth`: (Checkpoint) Model Weights 
- `modelweights.pth`: Saved Model Weights of DDQN
- `Report.pdf`: Report

## Instructions
Run the `Navigation.ipynb` to train the agent using DQN and DDQN. The average rewards over 100 consecutive episodes will be printed.
A plot showing the agent progress will be shown, and the model weights will be saved in files `checkpoint.pth`. Nevertheless, the repository contains weights `modelweights.pth` trained using DDQN.

## Report
The detailed description of the solution is summarised in `Report.pdf`.
