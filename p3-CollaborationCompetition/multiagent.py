import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import DDPG_Agent, TD3_Agent, device

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99

class Multi_Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, num_agents=2, state_size=24, action_size=2, random_seed=0, TD3=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        if(TD3):
            self.agents = [TD3_Agent(state_size, action_size, i+1, random_seed) for i in range(num_agents)]
        else:
            self.agents = [DDPG_Agent(state_size, action_size, i+1, random_seed) for i in range(num_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=0)

    def step(self, states, actions, rewards, next_states, dones, freq):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)

        self.memory.add(states, actions, rewards, next_states, dones)
#        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for idx, agent in enumerate(self.agents):
                experiences = self.memory.sample()
                self.learn(experiences, idx, freq)

    def act(self, states, add_noise=True, nu = 1.0):
        """Picks an action for each agent given."""
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise=add_noise,nu=nu)
            actions.append(action)
        return np.array(actions)

    def reset(self):
        """Resets OU Noise for each agent."""
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, idx, freq):
        """ observations and actions from all agents. Collect actions from each agent. """
        actions_next = []
        actions_pred = []
        states, _, _, next_states, _ = experiences

        next_states = next_states.reshape(-1, self.num_agents, self.state_size)
        states = states.reshape(-1, self.num_agents, self.state_size)

        for idxx, agent in enumerate(self.agents):
            idt = torch.tensor([idxx]).to(device)

            state = states.index_select(1, idt).squeeze(1)
            next_state = next_states.index_select(1, idt).squeeze(1)

            actions_next.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))

        actions_next = torch.cat(actions_next, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)

        agent = self.agents[idx]
        agent.learn(experiences, actions_next, actions_pred, freq)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
