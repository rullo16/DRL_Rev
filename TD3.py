#Twin Delayed Deep Deterministic Policy Gradient (TD3)
'''
Advanced RL algorithm designed to address key issues with DDPG. TD3 improves stability and learning efficiency by introducing three critical changes:
- Double Q-learning: Reduces overestimation bias
- Delayed Policy updates: Updates the actor less frequently than the critic, reducing variance in policy updates
- Target Policy Smoothing: Adds noise to the target actions to mitigate harmful overestimation in Q-values.

TD3 is effective for continuous action space envs, making it a preferred choice in robotics and control tasks.
Challenges in DDPG:
DDPG struggles with:
    - Overestimation bias: Q-values are overestimated, which means that the values are higher than the true values. This can lead to suboptimal policies.
    - High variance in actor updates: Updating the actor too frequently with unstable Q-function leads to poor policy updates.
    - Sharp Q-function changes: Leading to inefficient learning.
TD3 Enhancements:
    1. Double Q-learning:
    TD3 maintains two separate critic networks, Q1 and Q2 to estimate state-action values.
    The target Q-value is computed as:
        y = r + γ * min(Q1(s', π'(s')), Q2(s', π'(s')))
    Using the minimum Q-value mitigates overestimation bias, leading to more stable learning.
    2. Delayed Policy Updates:
        TD3 updates the actor less frequently than the critics:
            θ ← θ - α * ∇θ J(πθ)
        The actor is updated only every d iterations, reducing variance in policy updates.
    3. Target Policy Smoothing:
        TD3 adds noise to the target action during critic updates:
            â = πθ'(s') + ε, where ε ~ clip(N(0, σ), -c, c)
        This prevents the Q-function from overfitting to specific sharp action values, improving stability.
'''

#Implementation
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        x = self.max_action * torch.tanh(self.fc3(x))
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x1 = torch.relu(self.fc1(x))
        x1 = torch.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x = torch.cat([state, action], dim=1)
        x2 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x2))
        x2 = self.fc3(x2)
        return x1, x2
    
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.replay_buffer = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_delay = 2
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().numpy().flatten()
        action += noise * np.random.normal(0, 1, size=len(action))
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size, update_step):
        if len(self.replay_buffer) < batch_size:
            return
        
        sample = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = zip(*sample)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) #0.2 is policy smoothing noise, 0.5 is noise clip
            next_action = torch.clip(self.actor_target(next_state) + noise, -self.max_action, self.max_action) #Use target actor

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward + self.gamma*(1-done)* target_q

        #Critic loss
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, y) + nn.MSELoss()(current_q2, y)

        #Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Delayed policy updates
        if update_step % self.policy_delay == 0:
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3Agent(state_dim, action_dim, max_action)

num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    tot_reward = 0

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.append((state, action, next_state, reward, done))
        agent.train(64, t)
        state = next_state
        tot_reward += reward

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {tot_reward:.2f}")
env.close()

        
# Improve exploration with Ornstein-Uhlenbeck noise
class OrnsteinUhlbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
    
# Update select_action method

class TD3OH(TD3Agent):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__(state_dim, action_dim, max_action)
        self.noise = OrnsteinUhlbeckNoise(action_dim)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().numpy().flatten()
        action += self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)
    
# Update training loop
agent = TD3OH(state_dim, action_dim, max_action)

num_episodes = 200

for episode in range(num_episodes):
    state, _ = env.reset()
    agent.noise.reset()
    tot_reward = 0

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.append((state, action, next_state, reward, done))
        agent.train(64, t)
        state = next_state
        tot_reward += reward

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {tot_reward:.2f}")

# More complex envs

env = gym.make('HalfCheetah-v4')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3OH(state_dim, action_dim, max_action)

num_episodes = 200

for episode in range(num_episodes):
    state, _ = env.reset()
    agent.noise.reset()
    tot_reward = 0

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.append((state, action, next_state, reward, done))
        agent.train(64, t)
        state = next_state
        tot_reward += reward

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {tot_reward:.2f}")

        
