# p2-ContinuousControl

### Introduction
An agent is trained to move a double-jointed arm to target locations. Considering the first version, the primary objective is to get an average score of +30 over 100 consecutive episodes. The Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic (TD3) are utilized for the task.

### Project details
An agent is trained to move a double-jointed arm to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and +1.

### Getting Started
The project is completed in the Udacity Workspaces. However in order to run on local machine, the following packages are required:
```
matplotlib
numpy
unityagents==0.4.0
torch
``` 

## Files
- `Continuous_Control.ipynb`: Notebook used to train and test the agent 
- `agent.py`: Agent class for DDPG and TD3
- `model.py`: Model architecture
- `DDPG_checkpoint_actor.pth`: DDPG actor weights 
- `DDPG_checkpoint_critic.pth`: DDPG critic weight
- `TD3_checkpoint_actor.pth`: TD3 actor weight
- `TD3_checkpoint_critic1.pth`: TD3 critic 1 weight
- `TD3_checkpoint_critic2.pth`: TD3 critic 2 weight
- `README.md`: ReadMe
- `Report.pdf`: Report

## Instructions
Run the `Continuous_Control.ipynb` to train the agent using DDPG and TD3. The average rewards over 100 consecutive episodes will be printed. A plot showing the agent scores will be shown, and the model weights will be saved in files. Nevertheless, the repository contains weights `DDPG_checkpoint_actor.pth` and `DDPG_checkpoint_critic.pth` trained using DDPG, and `TD3_checkpoint_actor.pth`, `TD3_checkpoint_critic1.pth`, `TD3_checkpoint_critic2.pth` trained using TD3.

## Report
The detailed description of the solution is summarised in `Report.pdf`.
