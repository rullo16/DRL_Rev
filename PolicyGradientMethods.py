#Policy Gradient Methods (REINFORCE)
'''
Class of RL algorithms that directly optimize the policy function pi_theta(s,a) by adjusting its parameters theta in the direction of better performance.
Instead of learning a value function and deriving a policy from it, as in Q-learning or SARSA, poligy gradient methods work directly with the policy.
Key Concepts:
-Policy Objective Function:
    -The goal is to maximize the expected return:
        J(theta) = E_pi[sum^T R_t]
        This measures the total reward the agent expects to accumulate when following the policy pi_theta.
- Policy Gradient Theorem:
    The gradient of J(theta) can be expressed as:
        Delta_theta J(theta) = E_pi[Delta_theta log pi_theta(a_t|s_t) * G_t]
        where G_t is the return (or related measure such as advantage).
        This means that we can sample trajectories from the environment, compute the returns for those trahectories, and use them to estimate the gradient.
-REINFORCE Algorithm:
    Simplest policy gradient methods:
        -sample a trajectory using current policy
        -compute return G_t for each time step
        -Update the policy parameters theta as:
            theta <- theta + alpha * Delta_theta log pi_theta(a_t|s_t) * G_t
            This pushes the policy to increase the probability of actions that led to higher returns.
- Baseline for Variance Reduction:
    To reduce the variance of gradient estimate, a baseline b(s) can be subtracted from the return:
        Delta_theta J(theta) = E_pi[Delta_theta log pi_theta(a_t|s_t) * (G_t - b(s_t))]
    Common baselines include the state-value function V(s) or an average return.
'''

#Implementation of REINFORCE algorithm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Discrete Action Space Policy Network and REINFORCE Training

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))
    
def train_reinforce(env, policy_net, optimizer, num_episodes=500, gamma=0.99):

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()
            log_prob = torch.log(action_probs[0, action])
            log_probs.append(log_prob)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        returns = returns.unsqueeze(1)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f'Episode {episode}\tPolicy Loss: {policy_loss.item():.2f}, Total Reward: {total_reward}')

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
policy_net = PolicyNetwork(state_dim, num_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

train_reinforce(env, policy_net, optimizer)
env.close()

# REINFORCE with Baseline

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
def train_reinforce_baseline(env, policy_net, value_net, policy_optimizer, value_optimizer, num_episodes=500, gamma=0.99):

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        values = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()
            log_prob = torch.log(action_probs[0, action])
            log_probs.append(log_prob)

            value = value_net(state_tensor)
            values.append(value)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        values_tensor = torch.cat(values)
        value_loss = nn.MSELoss()(values_tensor, returns.unsqueeze(1))
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        advantages = returns.unsqueeze(1) - values_tensor.detach()

        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f'Episode {episode}\tPolicy Loss: {policy_loss.item():.2f}, Value Loss: {value_loss.item():.2f}, Total Reward: {total_reward}')

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
policy_net = PolicyNetwork(state_dim, num_actions)
value_net = ValueNetwork(state_dim)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

train_reinforce_baseline(env, policy_net, value_net, policy_optimizer, value_optimizer)
env.close()

# REINFORCE for Continuous Action Spaces

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std
    
def train_reinforce_continuous(env, policy_net, optimizer, num_episodes=500, gamma=0.99):
    epsilon = 1e-8
    # Precompute constant term
    log_2pi = torch.log(torch.tensor(2 * np.pi))

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = policy_net(state_tensor)
            std = torch.exp(log_std)

            # Sample action from Gaussian
            action = torch.normal(mean, std)
            # Compute log probability
            log_prob = -0.5 * (((action - mean) / (std + epsilon))**2 + 2 * log_std + log_2pi)
            log_prob = log_prob.sum(dim=-1)  # sum over action dimensions
            log_probs.append(log_prob)

            # env.step requires numpy array for continuous actions
            action_np = action.detach().numpy().flatten()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f'Episode {episode}\tPolicy Loss: {policy_loss.item():.2f}, Total Reward: {total_reward}')

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
policy_net = ContinuousPolicyNetwork(state_dim, num_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

train_reinforce_continuous(env, policy_net, optimizer)
env.close()


#REINFORCE with Advantage Estimation

def train_reinforce_advantage(env, policy_net, value_net, policy_optimizer, value_optimizer, num_episodes=500, gamma=0.99):
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs, rewards, values = [], [], []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()
            log_prob = torch.log(action_probs[0, action])
            log_probs.append(log_prob)

            value = value_net(state_tensor)
            values.append(value)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        values_tensor = torch.cat(values)
        value_loss = nn.MSELoss()(values_tensor, returns.unsqueeze(1))

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        advantages = returns.unsqueeze(1) - values_tensor.detach()
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f'Episode {episode}\tPolicy Loss: {policy_loss.item():.2f}, Value Loss: {value_loss.item():.2f}, Total Reward: {total_reward}')

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

policy_net = PolicyNetwork(state_dim, num_actions)
value_net = ValueNetwork(state_dim)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

train_reinforce_advantage(env, policy_net, value_net, policy_optimizer, value_optimizer)
env.close()

