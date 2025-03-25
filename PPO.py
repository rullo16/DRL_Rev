# Proximal Policy Optimization (PPO) algorithm
'''
Proximal Policy Optimization is a standard reinforcement learning algorithm due to its balance of simplicity, stability and performance.
Developed by OpenAI, PPO was introduced as a more accessible alternative to Trust Region Policy Optimization (TRPO).
Using a straightforward clipped objective to limit policy updates and maintain stability. Its adaptability has made it a go-to approach in various domains, including robotic control,
videogames playing, and even fine-tuning large language models.

Theoretical Background:
PPO builds on the policy gradient framework. Policy gradient methods aim to directly optimize the policy's parameters to maximize expected reward.
The key idea is to increase the probability of actions that yield positive advantages and decrease the probability of actions that yield negative advantages.
PPo simplifies this process by introducing a clipped surrogate objective that constrains the policy updates, ensuring that the policy does not shift too drastivally from the previous iteration.
Key Concepts:
    - Policy Gradient Theorem: Provides the foundation for PPO's objective. It defines the gradient of expected rewards as a function of the log-probability of actions weighted by their advantages.
    - Advantage Function: Measures how much better a particular action is compared to the baseline expectation. In PPO, this is typically calculated as the difference between the return 
                          (total discounted reward) and the value function's prediction.
    - Clipped Surrogate Objective: Instead of a hard constraint on how much the policy can change (as in TRPO), PPO uses a 'soft' constraint by penalizing updates that push
                                    the new policy probability ratio outside a defined range. This clipped objective ensures stability and prevents performance collapse.

Mathematical Objective:
Policy Gradient Theorem:
    Goal of RL is to maximize expected cumulative reward:
        J(theta) = E_t-pi_theta[Sum_T gamma^t * r_t]

    The policy gradient theorem states that the gradient of the expected reward can be computed as:
        Grad_theta J(theta) = E_t[Sum_t grad_theta log(pi_theta(a_t|s_t)) * A_t]

    Where:
        - pi_theta(a_t|s_t): Policy function
        - A_t: Advantage function
        - r_t: Reward at time t
        - gamma: Discount factor

PPO Maximizes:
    L_CLIP(theta) = E_t[min(r_t(theta)* A_t, clip(r_t(theta), 1-epsilon, 1+epsilon)*A_t)]

Where:
    - L_CLIP(theta): Clipped surrogate objective
    - r_t(theta): Probability ratio of the new policy to the old policy r(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
    - A_t: Advantage function
    - epsilon: Clipping parameter
The objective uses the minimum of the unclipped and clipped terms. This yields a lower-bound estimate of the true objective meaning the objective will pnlu get larger if the policy improvement
is sure not to violate the trust region.
This clipping has the effect of limiting the incentive for the policy to change too much on any given update. If an advantage is positive, teh unconstrained term encourages increaseing
pi_theta(a_t|s_t)  (to exploit a good action). But PPO clips this increase to be not too large. If the advantage is negative (action worse than expected), the term encourages 
pi_theta(a_t|s_t) to decrease, but again, the clipping prevents the policy from changing too much.
PPO also includes a value function loss and entropy bonnus in its overall loss function.
- The critic is updated by minimizing a value loss. This trains the critic to better estimate future rewards, which in turn leads to better advantage estimates. A good advantage estimator is crutcial
    for low-variance policy gradients. L_value = 1/2 (V_theta(s_t)- G_t)^2 where G_t is the estimated return. A_t = G_t - V_theta(s_t)
- The entropy bonus L_H = -beta * R[H(pi(.|s_t))] with beta as entropy coefficient, is sometime added to encourage exploration by penalizing under-confident policies. Maximizing entropy
prevents premature convergence to a suboptimal policy.

The final loss is typically:
    L(theta) = L_CLIP(theta) + c1*L_value + c2*L_H
'''

# PPO Implementation
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
env_name = 'CartPole-v1'
learning_rate = 3e-4
gamma = 0.99 # Discount factor
lam = 0.95 # GAE lambda parameter
clip_epsilon = 0.2 #PPO clip parameter
k_epochs = 4 # Number of optimization epochs per batch
batch_size = 2048 # Number of batches (experiences) sampled from the replay buffer
entropy_coef = 0.01 # Entropy bonus coefficient
value_coef = 0.5 # Value loss coefficient
max_grad_norm = 0.5 #Gradient clipping parameter
num_iterations = 1000 # Number of training iterations

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, action_dim)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
     
        logits = self.policy_head(x)
        action_probs = torch.softmax(logits, dim=1)
     
        state_value = self.value_head(x)
     
        return action_probs, state_value
    
env = gym.make(env_name)
obs_shape = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = ActorCritic(obs_shape, n_actions)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)


# for iter in range(num_iterations):
#     # Rollout function

#     states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

#     state,_ = env.reset(seed=42)
#     state = torch.tensor(state, dtype=torch.float32)

#     timesteps = 0

#     while timesteps < batch_size:

#         with torch.no_grad():
#             actions_probs, state_value = agent(state)

#         dist = torch.distributions.Categorical(actions_probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         next_state, reward, terminated, truncated, info = env.step(action.item())

#         done = terminated or truncated

#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         dones.append(done)
#         log_probs.append(log_prob)
#         values.append(state_value)

#         state = torch.tensor(next_state, dtype=torch.float32)
#         timesteps += 1

#         if done:
#             state, _ = env.reset(seed=42)
#             state = torch.tensor(state, dtype=torch.float32)

#     # Compute returns and advantages
#     returns = []
#     G = 0
#     for reward, done in zip(reversed(rewards), reversed(dones)):
#         if done:
#             G=0
#         G = reward + gamma * G
#         returns.insert(0, G)
#     returns = torch.tensor(returns, dtype=torch.float32)

#     states = torch.stack(states)
#     actions = torch.stack(actions)
#     log_probs = torch.stack(log_probs)
#     values = torch.tensor(values, dtype=torch.float32)

#     advantages = returns - values
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#     # PPO optimization
#     for epoch in range(k_epochs):
#         new_action_probs, new_values = agent(states)
#         new_values = new_values.squeeze(1)

#         new_dist = torch.distributions.Categorical(new_action_probs)
#         new_log_probs = new_dist.log_prob(actions)
#         entropy = new_dist.entropy().mean()

#         #Compute ratio r_t(theta)
#         ratios = torch.exp(new_log_probs - log_probs)
#         #Surrogate losses
#         surr1 = ratios * advantages
#         surr2 = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages

#         policy_loss = -torch.min(surr1, surr2).mean()
#         value_loss = nn.functional.mse_loss(new_values, returns)
#         loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

#         optimizer.zero_grad()
#         loss.backward()

#         nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
#         optimizer.step()

#     if iter % 50 == 0:
#         print(f'Iteration {iter+1}, Loss: {loss.item()}, Reward: {sum(rewards)}')

#General Advantage Estimation (GAE)
'''
Reduce variance in the advantage estimates by mixing multi-step returns with a parameter lambda, yielding more stable updates.
Instead of computing advantages as A_t = G_t - V_theta(s_t), use GAE:
    A_t = delta_t + (gamma * lambda) * delta_t+1 + (gamma * lambda)^2 * delta_t+2 + ... + (gamma * lambda)^(T-t-1) * delta_T-1
Where:
    - delta_t = r_t + gamma * V_theta(s_t+1) - V_theta(s_t) is the temporal difference error
    - lambda is the GAE parameter, balancing bias and variance
Adjust lambda to control the bias-variance trade-off. Smaller lambda reduces variance but increases bias, while larger lambda increases variance but reduces bias.
'''

for iter in range(num_iterations):
    # Rollout function

    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    state,_ = env.reset(seed=42)
    state = torch.tensor(state, dtype=torch.float32)

    timesteps = 0

    while timesteps < batch_size:

        with torch.no_grad():
            actions_probs, state_value = agent(state)

        dist = torch.distributions.Categorical(actions_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(state_value)

        state = torch.tensor(next_state, dtype=torch.float32)
        timesteps += 1

        if done:
            state, _ = env.reset(seed=42)
            state = torch.tensor(state, dtype=torch.float32)

    #Compute advantages using GAE
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)

    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.tensor(values, dtype=torch.float32)
    returns = values + advantages

    # PPO optimization
    for epoch in range(k_epochs):
        new_action_probs, new_values = agent(states)
        new_values = new_values.squeeze(1)

        new_dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy().mean()

        #Compute ratio r_t(theta)
        ratios = torch.exp(new_log_probs - log_probs)
        #Surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(new_values, returns)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

    if iter % 50 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}, Reward: {sum(rewards)}')


#Value function clipping
'''
Stabilize value function updated by preventing large changes in the value predictions.
Instructions:
    -Introduce a clipping mechanism to value loss. Instead of optimizing directly for (V_new(s)-G_t)^2, clip the value predictions to be within a certain range.
    -this ensures updates to value function are small and gradual, which helps maintain stable advantage calculations over time.
'''

for iter in range(num_iterations):
    # Rollout function

    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    state,_ = env.reset(seed=42)
    state = torch.tensor(state, dtype=torch.float32)

    timesteps = 0

    while timesteps < batch_size:

        with torch.no_grad():
            actions_probs, state_value = agent(state)

        dist = torch.distributions.Categorical(actions_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(state_value)

        state = torch.tensor(next_state, dtype=torch.float32)
        timesteps += 1

        if done:
            state, _ = env.reset(seed=42)
            state = torch.tensor(state, dtype=torch.float32)

    #Compute advantages using GAE
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.tensor(values, dtype=torch.float32)
    returns = torch.tensor(values + advantages, dtype=torch.float32)

    # PPO optimization
    for epoch in range(k_epochs):
        new_action_probs, new_values = agent(states)
        new_values = new_values.squeeze(1)

        new_dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy().mean()

        #Compute ratio r_t(theta)
        ratios = torch.exp(new_log_probs - log_probs)
        #Surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        # Clip value predictions
        value_pred_clipped =values + torch.clamp(new_values - values, -clip_epsilon, clip_epsilon)
        value_loss1 = (new_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)

        value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

    if iter % 50 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}, Reward: {sum(rewards)}')

# Entropy Regularization and reward normalization
'''
Entropy Reg:
Encourage exploration by adding an entropy bonus to the loss function, which penalizes overly confident policies.
Instructions:
    - Conpute entropy of policy's action distribution
    -Add a term to the total loss proportional to the negative entropy
    -Adjust the entropy coefficient (beta) to balance exploration and exploitation
Reward Normalization:
Improve training stability by normalizing rewards to have zero mean and unit variance.
Instructions:
    -Normalize raw rewards before computing returns and advantages
    -Helps prevent very large or very small reward values from dominating the training process leading to more consistent updated
'''

for iter in range(num_iterations):
    # Rollout function

    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    state,_ = env.reset(seed=42)
    state = torch.tensor(state, dtype=torch.float32)

    timesteps = 0

    while timesteps < batch_size:

        with torch.no_grad():
            actions_probs, state_value = agent(state)

        dist = torch.distributions.Categorical(actions_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(state_value)

        state = torch.tensor(next_state, dtype=torch.float32)
        timesteps += 1

        if done:
            state, _ = env.reset(seed=42)
            state = torch.tensor(state, dtype=torch.float32)
    #Normalize rewards
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    #Compute advantages using GAE
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.tensor(values, dtype=torch.float32)
    returns = values + advantages

    # PPO optimization
    for epoch in range(k_epochs):
        new_action_probs, new_values = agent(states)
        new_values = new_values.squeeze(1)

        new_dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy().mean()

        #Compute ratio r_t(theta)
        ratios = torch.exp(new_log_probs - log_probs)
        #Surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        # Clip value predictions
        value_pred_clipped =values + torch.clamp(new_values - values, -clip_epsilon, clip_epsilon)
        value_loss1 = (new_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)

        value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

    if iter % 50 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}, Reward: {sum(rewards)}')

# Learning Rate annealing
'''
Improve convergence by gradually reducing the learning rate as training progresses.
Instructions:
    - Start with higher learning rate and linearly decay it overtime
    - allows for larger, more exploratory steps in training and smaller, more stable updates later.
'''
def linear_schedyle(start_lr, end_lr, current_step, total_steps):
    return start_lr - (end_lr -start_lr)*(current_step/total_steps)

for iter in range(num_iterations):
    # Rollout function

    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    state,_ = env.reset(seed=42)
    state = torch.tensor(state, dtype=torch.float32)

    timesteps = 0

    while timesteps < batch_size:

        with torch.no_grad():
            actions_probs, state_value = agent(state)

        dist = torch.distributions.Categorical(actions_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(state_value)

        state = torch.tensor(next_state, dtype=torch.float32)
        timesteps += 1

        if done:
            state, _ = env.reset(seed=42)
            state = torch.tensor(state, dtype=torch.float32)
    #Normalize rewards
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    #Compute advantages using GAE
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * (1-dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.tensor(values, dtype=torch.float32)
    returns = values + advantages

    # PPO optimization
    for epoch in range(k_epochs):
        new_action_probs, new_values = agent(states)
        new_values = new_values.squeeze(1)

        new_dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)
        entropy = new_dist.entropy().mean()

        #Compute ratio r_t(theta)
        ratios = torch.exp(new_log_probs - log_probs)
        #Surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        # Clip value predictions
        value_pred_clipped =values + torch.clamp(new_values - values, -clip_epsilon, clip_epsilon)
        value_loss1 = (new_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)

        value_loss = 0.5 * torch.mean(torch.max(value_loss1, value_loss2))
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        current_lr = linear_schedyle(learning_rate, 0, iter, num_iterations)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    if iter % 50 == 0:
        print(f'Iteration {iter}, Loss: {loss.item()}, Reward: {sum(rewards)}')