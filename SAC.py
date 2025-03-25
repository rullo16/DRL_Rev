#Soft Actor Critic
'''
Soft Actor-Critic (SAC)is a model-free, off-policy, actor-critic RL algorithm that improves upon traditional actor-critic methods like DDPG and TD3.
SAC optimizes both:
    1. Expected Return
    2. Entropy
The goal of SAC is to maximize the cumulative discounted return while adding an entropy term to encourage exploration:
    J(π) = E[Σγ^t(r_t + αH(π(·|s_t)))]
where H(π(·|s_t)) is the entropy of the policy π, and α is a temperature parameter that controls the relative importance of the entropy term.
Key differences between SAC and TD3/DDPG
    1.SAC:
        -Stochastic policy
        -Maximizes entropy
        -More stable due to entropy regularization
        -Off-policy learning(Replay Buffer)
        -Higher sample efficiency

    2.TD3/DDPG:
        -Deterministic policy
        -No entropy maximization (use noise for exploration (Gaussian or OU))
        -Prone to policy collapse
        -Off-policy Learning
        -Lower sample efficiency

SAC Architecture:

1. Soft Q-value Function Q_theta(s,a):
    -Estimates the expected return for taking action a in state s.
    -Trained using Bellman equation:
        J(Q) = E[(Q_theta(s,a)-(r+gamma*E_s'[V_theta'(s')]))^2]
2.Soft Value Function V_theta(s):
    -Approximates the expected return from state s.
    -Uses a soft Q-function update:
        V(s) = E_a[Q_theta(s,a) - alpha * log(pi(a|s))]
3. Policy Network pi_phi(s):
    -Uses Gaussian distribution (instead of TD3)
    -Trained to maximize expected Q-value while considering entropy:
        J(pi) = E_st-D,a_t-pi[alpha log pi_phi(a_t|s_t)-Q_theta(s_t,a_t)]
    -Action is reparameterized using a normal distribution:
        a = tanh(mu_pji(s)+sigma_phi(s)*epsilon)
        where epsilon ~ N(0,1)
4. Automatic Temperature Tuning alpha
    -Adjusts entropy weight dynamically using:
        J(alpha) = E[-alpha log pi_phi(a_t|s_t) - alpha H_target]
    -prevents over-exploration or under-exploration
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
        self.mu = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        mean = self.mu(x)
        log_std = self.log_std(x).clamp(-20,2)
        std = torch.exp(log_std)
        return mean, std
    
    def sample_action(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        action = torch.tanh(action)* self.max_action
        log_prob = normal.log_prob(action)
        return action, log_prob
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)

        self.fc3 = nn.Linear(state_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        x1 = torch.cat([state, action], dim=1)
        x1 = torch.relu(self.fc1(x1))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.q1(x1)

        x2 = torch.cat([state, action], dim=1)
        x2 = torch.relu(self.fc3(x2))
        x2 = torch.relu(self.fc4(x2))
        q2 = self.q2(x2)
        return q1, q2
    
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.alpha = 0.2
        self.gamma = 0.99
        self.tau = 0.005
        self.replay_buffer = deque(maxlen=100000)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        sample = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = zip(*sample)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        with torch.no_grad():
            next_action, log_prob = self.actor.sample_action(next_state)
            q1_target, q2_target = self.target_critic(next_state, next_action)
            q_target = torch.min(q1_target, q2_target)
            y = reward + self.gamma * (1-done)*q_target

        q1,q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, y) + nn.MSELoss()(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Policy Update
        action, log_prob = self.actor.sample_action(state)
        q1, q2 = self.critic(state, action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Soft update target net
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
    

env = gym.make('Pendulum-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = SACAgent(state_dim, action_dim, max_action)

for episode in range(200):
    state, _ = env.reset()
    total_reward = 0
    for t in range(200):
        action, _ = agent.actor.sample_action(torch.FloatTensor(state).unsqueeze(0))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.append((state, action, reward, next_state, done))
        agent.train(batch_size=64)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Episode: {episode}, Total Reward: {total_reward}')
env.close()
        
         
#Discrete SAC
'''
SAC is designed for continuous action spaces, but it can be adapted for discrete action spaces.
Instead of sampling continuous actions, we compute a soft categorical policy over discrete actions.

Modified Objective for Discrete Actions:
For discrete actions SAC uses Q-values to derive a soft policy.
We define the policy distribution using a softmax over Q-values:
    pi(a|s) = exp(Q(s,a)/alpha)/ sum_alpha'(exp(Q(s,a')/alpha))    
'''

class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscreteActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.policy_logits = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        policy_logits = self.policy_logits(x)
        return policy_logits
    
    def sample_action(self, state):
        logits = self.forward(state)
        action_probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
class DiscreteSAC(SACAgent):
    def __init__(self, state_dim, action_dim):
        super(DiscreteSAC, self).__init__(state_dim, action_dim)
        self.actor = DiscreteActor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)


#Implementation
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DiscreteSAC(state_dim, action_dim)

for episode in range(200):
    state = env.reset()
    total_reward = 0
    for t in range(200):
        action, log_prob = agent.actor.sample_action(torch.FloatTensor(state).unsqueeze(0))
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        agent.train(batch_size=64)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Episode: {episode}, Total Reward: {total_reward}')
env.close()

#Automatic Temperature Adjustment
'''
Fixed entropy weight alpha may be suboptimal in different environments
Solution is to implemt automatic temperature adjustment
Instead of keeping alpha constant, we optimize it using:
    J(alpha) = E[-alpha (log pi_phi(a_t|s_t) + H_target)]
This ensures that the agent maintains the correct level of exploration dynamically.
'''

class AlphaTuner:
    def __init__(self, init_alpha = 0.2, target_entropy=1.0):
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True)
        self.target_entropy = target_entropy
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

    def update(self, log_prob):
        alpha = self.log_alpha.exp()
        loss = - (alpha * (log_prob + self.target_entropy).detatch()).mean()
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        return self.log_alpha.exp().item()
    
class SACAlpha(SACAgent):
    def __init__(self, state_dim, action_dim, max_action):
        super(SACAlpha, self).__init__(state_dim, action_dim, max_action)
        self.alpha = AlphaTuner()

#Implementation
env = gym.make('Pendulum-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = SACAlpha(state_dim, action_dim, max_action)

for episode in range(200):
    state, _ = env.reset()
    total_reward = 0
    for t in range(200):
        action, log_prob = agent.actor.sample_action(torch.FloatTensor(state).unsqueeze(0))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.append((state, action, reward, next_state, done))
        agent.train(batch_size=64)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Episode: {episode}, Total Reward: {total_reward}')
env.close()

#SAC with latent representation
'''
SAC struggles with high-dimensional image inputs
The Solution is to use an autoencoder to extract useful latent feature begore feeding them to the policy and value networks.
Instead of using raw pixels s, we learn a latent representation z:
    z = f_encoder(s)
SAc is then trained on the latent representation z instead of the raw pixels s.
'''

#SAC with Multi-Agent Learning(MASAC)
'''
SAC assumes a single-agent setting, but what if we need multiple agents learning together?
We use multi-agent SAC:
    - Each agent has its own policy and critics
    - Critics use centralized training while policies act decentralized.
the critic minimize:
    J(Q_i)=E[(Q_phi_i(s, a_1, ..., a_n)-y)^2]
where y is the multi-agent Bellman Equation
'''


