# p2-ContinuousControl

### Introduction
Two agents is trained in the Tennis environment to control rackets to bounce a ball over a net. The primary objective is to get an average score of +0.5 over 100 consecutive episodes after taking the maximum over both agents. The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) and Multi-Agent Twin Delayed Deep Deterministic (MATD3) are utilized for the task. The project is completed in the \textit{Udacity Workspaces}.

\section{Environment}



### Project details
Two agents is trained in the \textit{Tennis} environment to control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of $+0.10$. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of $-0.01$. Thus, the goal of each agent is to keep the ball in play. The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, the rewards that each agent received (without discounting) are added up, to get a score for each agent. This yields 2 (potentially different) scores. By taking the maximum of these 2 scores, yields a single score for each episode.
- The environment is considered solved, when the average (over $100$ episodes) of those scores is at least +0.5


### Getting Started
The project is completed in the Udacity Workspaces. However in order to run on local machine, the following packages are required:
```
matplotlib
numpy
unityagents==0.4.0
torch
```

## Files
- `Tennis.ipynb`: Notebook used to train and test the agent
- `multiagent.py`: Wrapper to handle multiple agents
- `agent.py`: Agent class for DDPG and TD3
- `model.py`: Model architecture
- `MADDPG_actor_0.pth`: DDPG actor weights (agent 0)
- `MADDPG_critic_0.pth`: DDPG critic weight (agent 0)
- `MADDPG_actor_1.pth`: DDPG actor weights (agent 1)
- `MADDPG_critic_1.pth`: DDPG critic weight (agent 1)
- `MATD3_actor_0.pth`: TD3 actor weight (agent 0)
- `MATD3_critic1_0.pth`: TD3 critic 1 weight (agent 0)
- `MATD3_critic2_0.pth`: TD3 critic 2 weight (agent 0)
- `MATD3_actor_1.pth`: TD3 actor weight (agent 1)
- `MATD3_critic1_1.pth`: TD3 critic 1 weight (agent 1)
- `MATD3_critic2_1.pth`: TD3 critic 2 weight (agent 1)
- `README.md`: ReadMe
- `Report.pdf`: Report

## Instructions
Run the `Tennis.ipynb` to train the multiple agents using MADDPG and MATD3. The average rewards over 100 consecutive episodes, after taking the maximum over both agents, will be printed. A plot showing the two agents scores will be shown, and the model weights will be saved in files. Nevertheless, the repository contains weights `MADDPG_actor_[i].pth` and ``MADDPG_critic_[i].pth`` trained using DDPG for each ``[i]`` agent, and `MATD3_actor_[i].pth`, `MATD3_critic1_[i].pth`, `MATD3_critic2_[i].pth` trained using TD3 for each ``[i]`` agent.

## Report
The detailed description of the solution is summarised in `Report.pdf`.
