#Deep Q-Learning (DQN)
'''
Extends traditional Q-Learning algorithm by using neural networks to approximate the Q-function. 
This allows to scale to environments with large or continuous state spaces, where maintaining a Q-table is infeasible.
Key Components:
-Q-Network:
    Neural network Q(s,a; theta) that takes a state s and action a as input and predicts the Q-value.
    Trained using a modified version of the Q-Learning update rule:
        L(theta) = E[(R_t+1 + gamma*max_a'Q(s_t+1,a'; theta^-) - Q(s_t,a_t; theta))^2]
        - theta^-: represents the parameters of a target network, which is updated less frequently to stabilize training.
- Replay Buffer:
    Stores past experiences (s_t, a_t, R_t+1, s_t+1), allows sampling of random minibatches to break correlations in the data and improve stability.
- Target Network:
    Separate network Q(s,a; theta^-) used to compute target values, updated periodically to reduce instability caused by rapid changes in the Q-values.
- Exploration-Exploitation Tradeoff:
    Uses an epsilon-greedy policy that decays epsilon overtime, enabling exploration in the early stages and exploitation as the policy improves.

DQN scales to high-dimensional spaces, is model free(learns directly from raw environment interactions without needing a model of the environment),
and is off policy, allowing to learn the optimal policy while explorng suboptimal actions.
'''

#DQN implementation
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

#Hyperparameters
gamma = 0.99
learning_rate = 0.001
batch_size = 64
buffer_size = 10000
num_episodes = 1000
target_update = 10
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

#Q-network
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
#Init Environment and networks
env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
q_net = QNet(num_states, num_actions)
target_net = QNet(num_states, num_actions)
target_net.load_state_dict(q_net.state_dict())

#Init optimizer and loss
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

#Init replay buffer
memory = deque(maxlen=buffer_size)

def select_action(state, epsilon):
    if np.random.rand()<epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32)
        return torch.argmax(q_net(state)).item()
    
def train():
    if len(memory)<batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + (1-dones)*gamma*next_q_values
    
    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    tot_reward = 0

    while not done:
        action=select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        tot_reward += reward

        train()

    if episode%target_update==0:
        target_net.load_state_dict(q_net.state_dict())

    #Decay epsilon
    epsilon = max(epsilon*epsilon_decay, epsilon_min)

    if episode%10==0:
        print("Episode: ", episode, "Reward: ", tot_reward)

env.close()
print("Training complete")

#Extract policy from Q-network
policy = []
for _ in range(5):  # sample 5 states instead of iterating over range(num_states)
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    action = torch.argmax(q_net(state_tensor)).item()
    policy.append(action)
print("Policy: ")
print(policy)

#Display Q-values
print("Q-values: ")
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    q_values = q_net(state_tensor).detach().numpy()
    print(q_values)


#Double DQN
'''
Extension of DQN that addresses overestimation bias by using two networks to decouple action selection and evaluation.
Q1 is used to select the best action, while Q2 is used to evaluate it.
The update rule is:
    Q1(s_t, a_t) += alpha*(r_t+1 + gamma*Q2(s_t+1, argmax_a'Q1(s_t+1, a')) - Q1(s_t, a_t))
    Q2(s_t, a_t) += alpha*(r_t+1 + gamma*Q1(s_t+1, argmax_a'Q2(s_t+1, a')) - Q2(s_t, a_t))
'''

#Double DQN implementation
Q1 = QNet(num_states, num_actions)
Q2 = QNet(num_states, num_actions)
Q2.load_state_dict(Q1.state_dict())

#Hyperparameters
gamma = 0.99
learning_rate = 0.001
batch_size = 64
buffer_size = 10000
num_episodes = 1000
target_update = 10
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

#Init optimizer and loss
optimizer1 = torch.optim.Adam(Q1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(Q2.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

#Init replay buffer
memory = deque(maxlen=buffer_size)

def select_action(state, epsilon):
    if np.random.rand()<epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32)
        return torch.argmax(Q1(state)).item()
    
def train():
    if len(memory)<batch_size:
        return
    
    batch=random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = Q1(states).gather(1, actions)

    with torch.no_grad():
        next_actions = torch.argmax(Q1(next_states), dim=1, keepdim=True)
        next_q_values = Q2(next_states).gather(1, next_actions)

        targets = rewards + (1-dones)*gamma*next_q_values

    loss = loss_fn(q_values, targets)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

    
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    tot_reward = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        tot_reward += reward

        train()

    if episode%target_update==0:
        Q2.load_state_dict(Q1.state_dict())

    epsilon = max(epsilon*epsilon_decay, epsilon_min)

    if episode%10==0:
        print("Episode: ", episode, "Reward: ", tot_reward)

env.close()
print("Training complete")

#Extract policy from Q-network
policy = []
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    action = torch.argmax(Q1(state_tensor)).item()
    policy.append(action)
print("Policy: ")
print(policy)

#Display Q-values
print("Q-values: ")
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    q_values = Q1(state_tensor).detach().numpy()
    print(q_values)

#Dueling DQN
'''
Enhanced version of DQN that separates estimation of state value and action advantages.
In Dueling network the neural network is split into two streams:
    - Value stream estimates value of V(s) 
    - Advantage stream estimates the advantage of each action A(s,a)
    These streams are combined to produce the Q-values.

A common aggregation approach normalizes the advantage stream by subtracting the mean, ensuring the relative advantages of actions are preserved.
Q(s,a) = V(s) + (A(s,a)-1/|A|*sum_q'A(s,a'))
this helps avoid instability when advantage stream can dominate the value stream.
'''

#Dueling DQN implementation
class DuelingQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)

        #Value stream
        self.fc_value = nn.Linear(128, 64)
        self.fc_value_out = nn.Linear(64, 1)

        #Advantage stream
        self.fc_adv = nn.Linear(128, 64)
        self.fc_adv_out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))

        value = torch.relu(self.fc_value(x))
        value = self.fc_value_out(value)

        adv = torch.relu(self.fc_adv(x))
        adv = self.fc_adv_out(adv)

        q_values = value + (adv - adv.mean(dim=1, keepdim=True))
        return q_values
    

def select_action(state, epsilon):
    if np.random.rand()<epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32)
        return torch.argmax(q_net(state)).item()
    
def train():
    if len(memory)<batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + (1-dones)*gamma*next_q_values
    
    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    tot_reward = 0

    while not done:
        action=select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        tot_reward += reward

        train()

    if episode%target_update==0:
        target_net.load_state_dict(q_net.state_dict())

    #Decay epsilon
    epsilon = max(epsilon*epsilon_decay, epsilon_min)

    if episode%10==0:
        print("Episode: ", episode, "Reward: ", tot_reward)

env.close()
print("Training complete")

#Extract policy from Q-network
policy = []
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    action = torch.argmax(q_net(state_tensor)).item()
    policy.append(action)
print("Policy: ")
print(policy)

#Display Q-values
print("Q-values: ")
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    q_values = q_net(state_tensor).detach().numpy()
    print(q_values)

#Prioritized Experience Replay
'''
Priority mechanism to replay buffer sampling, focusing more on transitions with higher TD errors.
It gives higher sampling priority to transitions with greater learning potential, the intuition is that some experiences are more "important" because the agent's current estimate of
their value differ from the actual return.
Key Concepts:
-importance-based sampling:
    Transition sampled with a probability proportional to their Temporal Difference error:
        Priority(i) = |delta_i|^alpha
    where delta_i is the TD error for transition i, and alpha is the strenght og prioritization. (alpha=0 correpsonds to uniform sampling)

-Bias compensation:
    Prioritized sampling introduces bias, as it changes the distribution of the training data. To correct this, importance sampling weights are used to scale the loss:
        w_i = (1/N*1/P(i))^beta
    where P(i) is the probability of sampling transition i, and beta is the strenght of the importance sampling correction, and N is the size of the replay buffer.
-Stochastic Prioritization:
    Instead of directly sampling the highest-priority experiences, stochastic prioritization smooths the selection process to avoid focusing too narrowly on certain transitions.
    For example, P(i)=p_i^alpha / sum_k p_k^alpha, where p_i = |delta_i|+epsilon ensures all transitions have a non-zero probability of being sampled.

Benefits:
-Faster convergence
-Better Utilization of experience
-Scalability
'''

#Prioritized Experience Replay implementation

# Hyperparameters
gamma = 0.99
learning_rate = 1e-3
batch_size = 64
buffer_size = 100000
num_episodes = 1000
target_update_freq = 10
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
alpha = 0.6
beta_start = 0.4
beta_end = 1.0

#Init replay buffer

#Sum Tree data structure to store priorities
'''
Each node's value is the sum of its child nodes. Leaf nodes hold the priorities, and the internal nodes store cumulative priorities.
'''

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.write_idx = 0

    def add(self, priority, data):
        idx = self.write_idx+self.capacity-1
        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx-1)//2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def sample(self, value):
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
    def _retrieve(self, idx, value):
        left = 2*idx+1
        right = left+1

        if left>=len(self.tree):
            return idx
        
        if value<=self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value-self.tree[left])
        
    def total_priority(self):
        return self.tree[0]
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5
    def __len__(self):
        return self.tree.size

    def add(self, experience, td_error):
        priority = (np.abs(td_error)+self.epsilon)**self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size, beta):
        total_priority = self.tree.total_priority()
        priorities = []
        samples = []
        idxs = []
        weights = []

        for _ in range(batch_size):
            value = np.random.uniform(0, total_priority)
            idx, priority, sample = self.tree.sample(value)
            priorities.append(priority)
            samples.append(sample)
            idxs.append(idx)

        min_prob = min(priorities)/total_priority
        for p in priorities:
            prob = p/ total_priority
            weight = (prob/min_prob)**-beta
            weights.append(weight)
        
        weights = np.array(weights, dtype=np.float32) /max(weights)
        return samples, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (np.abs(td_error)+self.epsilon)**self.alpha
            self.tree.update(idx, priority)

#Init replay buffer
buffer = PrioritizedReplayBuffer(buffer_size, alpha)

def select_action(state, epsilon):
    if np.random.rand()<epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32)
        return torch.argmax(q_net(state)).item()
    
def train():
    if len(buffer)<batch_size:
        return
    
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + (1-dones)*gamma*next_q_values
    
    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


q_network = QNet(num_states, num_actions)
target_network = QNet(num_states, num_actions)
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)  # added optimizer for PER
# Training Loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    beta = beta_start + (beta_end - beta_start) * (episode / num_episodes)

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(q_network(state_tensor)).item()  # Changed: use q_network to select action

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Compute initial TD error
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        q_values = q_network(state_tensor)
        next_q_values = target_network(next_state_tensor)
        max_next_q = torch.max(next_q_values).item()
        td_error = reward + gamma * max_next_q - q_values[0][action].item()

        # Add experience to buffer
        buffer.add((state, action, reward, next_state, done), td_error)

        state = next_state

        # Train network if enough samples
        if len(buffer) > batch_size:  # Changed: use len(buffer) instead of len(buffer.tree.data)
            samples, indices, weights = buffer.sample(batch_size, beta)
            states, actions, rewards, next_states, dones = zip(*samples)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

            # Compute Q-values and targets
            q_values = q_network(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]
                targets = rewards + (1 - dones) * gamma * next_q_values

            # Compute loss and apply importance sampling weights
            loss = (weights * (q_values - targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update priorities in buffer
            td_errors = (targets - q_values).detach().numpy().squeeze()
            buffer.update_priorities(indices, td_errors)

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Print episode information
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

# Extract policy from Q-network
policy = []
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    action = torch.argmax(q_network(state_tensor)).item()
    policy.append(action)

print("Policy: ")
print(policy)

# Display Q-values
print("Q-values: ")
for _ in range(5):
    sample_state = env.observation_space.sample()
    state_tensor = torch.tensor(sample_state, dtype=torch.float32)
    q_values = q_network(state_tensor).detach().numpy()
    print(q_values)

