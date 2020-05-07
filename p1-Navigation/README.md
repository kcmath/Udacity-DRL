# p1-Navigation

### Introduction
An agent is trained to navigate and collect bananas in a large, square world. The primary objective is to get an average score of +13 over 100 consecutive episodes. The Deep Q-network and Double Q-network are utilized for the task. 

### Project details
The environment is simulated by Unity application _Banana_. An agent is trained to navigate and collect bananas in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

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
- `checkpoint.pth`: Saved Model Weights
- `Report.pdf`: Report

## Instructions
Run the `Navigation.ipynb` to train the agent using DQN and DDQN. The average rewards over 100 consecutive episodes will be printed.
A plot showing the agent progress will be shown, and the model weights will be saved in files `checkpoint.pth`. Nevertheless, the repository contains weights trained using DDQN.

## Report
The detailed description of the solution is summarised in `Report.pdf`
