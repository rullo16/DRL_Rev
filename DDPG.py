#Deep Deterministic Policy Gradient (DDPG)
'''
Off-policy, model-free RL algorithm designed for continuous action spaces.
Builds on two key ideas:
    - Deterministic Policy Gradient (DPG): Instead of learning a stochastic policy, DDPG learns a deterministic policy.
    - Deep Q-Network (DQN): Uses neural networks to approximate the Q-function.
DDPG falls into the category of actor-critic methods.
It combines several important RL techniques:
    -Actor-Critic Architecture:
        -Actor(pi_theta(s)): NN parameterized by theta outputs a deterministic action for a given state.
        -Critic(Q_phi(s,a)): NN parameterized by phi predicts the expected return (Q-value) for a given state-action pair.
    -Deterministic Policy Gradient:
        Extend the Policy Gradient Theorem from stochastic policies to deterministic policies.
        Policy Gradient Theorem:
            In policy gradient methods, like REINFORCE, we optimize a stochastic policy pi(a|s) using:
                ∇_theta J = E[∇_theta log pi_theta(a|s) * Q^pi(s,a)]
        Deterministic Policy Gradient:
            Instead of sampling actions, we directly use a deterministic function a = pi_theta(s).
                ∇_theta J = E_s~p^pi[∇_theta pi_theta(s) * ∇_a Q^pi(s,a) |_{a=pi_theta(s)}]
            where:
                - ∇_a Q^pi(s,a) is the gradient of the Q-function w.r.t. the action.
                - ∇_theta pi_theta(s) is the gradient that propagates back through the policy.
    -Target Network:
        One major issue with Q-learning is that it can diverge when using function approximators. DDPG stabilizes training using target networks:
            -Target Actor: pi_theta'
            -Target Critic: Q_phi'
        These are slowly updated using Polyak averaging:
            theta' = tau * theta + (1 - tau) * theta'
            phi' = tau * phi + (1 - tau) * phi'
        where tau is a small parameter (e.g. 0.001) to ensure smooth updates.
    -Experience Replay:
        Instead of learning from recent experiences only, DDPG stores transitions in a replay buffer:
            D = {(s,a,r,s')}
        At each update step, we sample a minibatch from D to:
            1. Break correlation between consecutive samples.
            2. Improve data efficiency by reusing past experiences.
    -Exploration (Adding Noise):
        Since DDPG uses a deterministic policy, it requires external exploration. This is done using Ornstein-Uhlenbeck (OU) noise, a correlated noise process:
            N(t+1) = theta * (mu - N(t)) + sigma * epsilon_t
        where:
            - theta controls how quickly noise returns to the mean.
            - sigma determines noise intensity.
            - epsilon_t ~ N(0,1) is a Gaussian white noise.
Training Process:
    -Critic Update:
        The critic is trained using Bellman Equation:
            y = r + gamma * Q'_phi(s', pi_theta'(s'))
        where:
            - y is the target Q-value.
            - Q'_phi(s', pi_theta'(s')) is the target Q-value from the target critic.
        The critic loss is:
            L(phi) = E[(Q_phi(s,a) - y)^2]
    -Actor Update:
        The actor is updated by maximizing Q(s,a):
            J(theta) = E[Q_phi(s, pi_theta(s))]
        The gradient is:
            ∇_theta J = E[∇_theta pi_theta(s) * ∇_a Q_phi(s,a) |_{a=pi_theta(s)}]
'''

##Implementation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import gymnasium as gym
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x)) # Scale output to [-max_action, max_action]
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state,action], dim=1) # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DDPGAgent:

    def __init__(self, state_dim, action_dim, max_action):
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_critic = Critic(state_dim, action_dim)

        #Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        #Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()

        #Experience Replay
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64

        #Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.normal(0,1, size=action.shape) # Add noise for exploration
        return action.clip(-self.max_action, self.max_action)
    
    def store_transition(self, state,action, reward, next_state,done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_Q = self.target_critic(next_states, next_actions)
            y = rewards + self.gamma * (1-dones)*target_Q
        
        q_values = self.critic(states, actions)
        critic_loss = self.loss_fn(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# agent = DDPGAgent(state_dim, action_dim, max_action)

# num_episodes = 200
# for episode in range(num_episodes):
#     state, _ = env.reset()
#     episode_reward = 0

#     for t in range(200):
#         action = agent.select_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         agent.store_transition(state, action, reward, next_state, done)
#         state = next_state
#         episode_reward += reward

#         agent.train()

#         if done:
#             break

#     print(f"Episode: {episode}, Reward: {episode_reward}")

# env.close()

#Add Ornstein-Uhlenbeck noise
class OrnsteinUhlenbeck:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu # Mean of the noise
        self.theta = theta # Rate of mean reversion
        self.sigma = sigma # Standard deviation of the noise
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
    
#Update select_action method
class DDPGAgent_OK(DDPGAgent):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGAgent_OK, self).__init__(state_dim, action_dim, max_action)
        self.noise = OrnsteinUhlenbeck(action_dim)

    def select_action(self, state, noise=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if noise:
            action += self.noise.sample()
        return action.clip(-self.max_action, self.max_action)
    
    def reset_noise(self):
        self.noise.reset()


# agent = DDPGAgent_OK(state_dim, action_dim, max_action)

# num_episodes = 200
# for episode in range(num_episodes):
#     state, _ = env.reset()
#     episode_reward = 0

#     for t in range(200):
#         action = agent.select_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         agent.store_transition(state, action, reward, next_state, done)
#         state = next_state
#         episode_reward += reward

#         agent.train()

#         if done:
#             break

#     agent.reset_noise()
#     print(f"Episode: {episode}, Reward: {episode_reward}")


env = gym.make('Ant-v5')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent_OK(state_dim, action_dim, max_action)

num_episodes = 200
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        agent.train()

        if done:
            break

    agent.reset_noise()
    print(f"Episode: {episode}, Reward: {episode_reward}")
