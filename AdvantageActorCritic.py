#AdvantageActorCritic
'''
A2C is a type of actor-critic algorithm that combines two key components:
    -Actor (Policy Network): Determines which actions to take, updating policy pi(a!s;theta) to maximize expected rewards.
    -Critic (Value Network): Estimates the value of states V(s;phi), providing a baseline for evaluating the actor's choices and improving the stability of policy updates.
A2C uses the advantage function:
    A(s,a) = Q(s,a) - V(s)
This advantage measures how much better an action is compared to the expected value of the state, guiding the policy updates more effectively than raw returns alone.
How A2C differs from A3C:
    -A3C uses multiple asynchronous workers that update a shared model independently.
    -A2C Synchronizes the updates from all workers at the same time, avoiding stale gradients and reducing variability. Each worker collects a batch of experience and then all workers'
    updates are aggregated before the model parameters are updated. This synchronous approah improves training stability and simplifies implementations.
Benefits of A2C:
    -Stabilyty from Synchronous Updates
    -Better utilization of GPU
    -Improved Preformance and Simplicity

Algorithm:
    1. Initialize actor-critic network:
        Start with a shared NN that has two heads:
            -Policy head: Outputs a probability distribution over actions
            -Value head: Outputs the value of the state V(s)
    2. Run Multiple Environments in Parallel:
        Use several environments to collect experience in parallel. Each environment runs the policy for a fixed number of steps
        and collects state,actions, rewards, and done flags.
    3. Calculate Returns and Advantages:
        -Compute the total rewards R = sum_k=0^inf gamma^k * r_{t+k} for each step in the trajectory.
        -Compute the advantage as A_t = R_t - V(s_t), usiing the value estimate as baseline to reduce variance
    4. Compute Losses:
        -Policy Loss: L_policy = -log(pi(a_t|s_t;theta)) * A_t
        -Value Loss: L_value = (R_t - V(s_t))^2
        -Entropy Loss: L_entropy = -betha * H(pi) where beta is a hyperparameter that controls the strength of the entropy regularization and H(pi) is the entropy of the policy distribution.
    5. Update the Actor-Critic Network:
        -Combine the policy,value and entropy losses:
            L = L_policy +c*L_value + L_entropy
            where c is a hyperparameter that controls the weight of the value loss.
        -Compute gradients for total loss and update the network params.
    6. Repeat steps 2-5 until convergence.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import gymnasium as gym

# Implementation

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        return self.policy(x), self.value(x)
    
#Hyperparameters
gamma = 0.99
rollout_length = 5
entropy_beta = 0.01
value_loss_coef = 0.5

#Environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

#Training Loop

for update in range(1000):
    states, actions,rewards, dones, log_probs, values, entropies = [],[],[],[],[],[],[]
    state, _ = env.reset()

    for _ in range(rollout_length):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = model(state_tensor)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        log_probs.append(dist.log_prob(action))
        values.append(value)
        entropies.append(dist.entropy())

        state = next_state

        if terminated or truncated:
            state, _ = env.reset()

    #Compute Returns and Advantages
    returns, advs = [],[]
    G = 0
    for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
        G = reward + gamma * G * (1-done)
        returns.insert(0, G)
        advs.insert(0, G-value.item())
    returns = torch.tensor(returns)
    advs = torch.tensor(advs)

    #Loss computation
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    entropies = torch.stack(entropies)
    policy_loss = -(log_probs * advs).mean()
    value_loss = value_loss_coef * (returns-values).pow(2).mean()
    entropy_loss = -entropy_beta * entropies.mean()
    total_loss = policy_loss + value_loss + entropy_loss

    #Gradient Update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if update % 50 == 0:
        print(f'Update: {update}, Total Loss: {total_loss.item()}, Reward: {sum(rewards)}')

# Entropy Regularization
'''
Entropy regularization encourages exploration by penalizing low-entropy policies (those that are too deterministic).
The entropy term is added to the toal loss to prevent premature convergence to a single strategy. 
Increasign this coefficient leads to more diverse actions being tried, while a smaller coefficient focuses the policy on a narrower set of choices.

Implemented above

If increase entropy_beta, the model will explore more random actions because the entropy term will have larger influence on the total loss.
If you decrease it to a very small value, the policy becomes more deterministic earlier in training.
'''

# Parallel Environments
'''
Run multiple environments in parallel to increase the diversity of collected experiences.
This can speed up training by allowing the agent to learn from multiple trajectories simultaneously.
'''

# Implementation

envs = gym.make_vec('CartPole-v1', num_envs=4, vectorization_mode='async')

current_states, _ = envs.reset()

for update in range(1000):
    batch_states, actions, rewards, dones, log_probs, values, entropies = [],[],[],[],[],[],[]
    state = current_states

    for _ in range(rollout_length):
        state_tensor = torch.FloatTensor(state)
        logits, value = model(state_tensor)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample()
        next_state, reward, terminated, truncated, _ = envs.step(action.numpy())

        batch_states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(np.logical_or(terminated, truncated))
        log_probs.append(dist.log_prob(action))
        values.append(value)
        entropies.append(dist.entropy())

        state = next_state

    #Compute Returns and Advantages
    returns, advs = [],[]
    G = torch.zeros_like(values[-1])
    for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
        reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done_tensor = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        G = reward_tensor + gamma * G * (1 - done_tensor)
        returns.insert(0, G)
        advs.insert(0, G - value)
    returns = torch.stack(returns).squeeze(-1)
    advs = torch.stack(advs).squeeze(-1)

    #Loss computation
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    entropies = torch.stack(entropies)
    policy_loss = -(log_probs * advs).mean()
    value_loss = value_loss_coef * (returns - values).pow(2).mean()
    entropy_loss = -entropy_beta * entropies.mean()
    total_loss = policy_loss + value_loss + entropy_loss

    #Gradient Update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if update % 50 == 0:
        print(f'Update: {update}, Total Loss: {total_loss.item()}, Reward: {sum(rewards)}')

#Continuous Action Spaces

class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        value = self.value(x)
        return mean, std, value


env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
model = ContinuousActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for update in range(1000):
    states, actions,rewards, dones, log_probs, values, entropies = [],[],[],[],[],[],[]
    state, _ = env.reset()

    for _ in range(rollout_length):
        state_tensor = torch.FloatTensor(state).view(1, -1)
        mean, std, value = model(state_tensor)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)

        state = next_state

        if done:
            state, _ = env.reset()

    #Compute Returns and Advantages
    next_value = 0.0 if done else model(torch.FloatTensor(next_state).view(1, -1))[2].item()
    returns = []
    G = next_value
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0.0
        G = r+gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values).squeeze()
    entropies = torch.stack(entropies)

    #Loss computation
    advs = returns - values.detach()
    policy_loss = -(log_probs * advs).mean()
    value_loss = value_loss_coef * (returns-values).pow(2).mean()
    entropy_loss = -entropy_beta * entropies.mean()
    total_loss = policy_loss + value_loss + entropy_loss

    #Gradient Update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if update % 50 == 0:
        print(f'Update: {update}, Total Loss: {total_loss.item()}, Reward: {sum(rewards)}')