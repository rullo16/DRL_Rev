#Temporal difference learning
'''
TD learning is model free RL approach updating value estimates based on partial feedback from the environment. TD methods update estimates after each time step.
Key difference between TD and Monte Carlo methods:
- TD methods use bootstrapping (update estimates based on other estimates):
    - V(s_t) = V(s_t) + alpha*(r_t+1 + gamma*V(s_t+1) - V(s_t))
    - r_t+1 + gamma*V(s_t+1) - V(s_t) is the TD error
- TD updates after each step:
    -Reduces memory requirements and allows learning in continuing tasks
- Covergence properties:
    - TD converges to the correct value function under certain conditions, without needing a model of the environment
'''

#SARSA (On-policy TD control)
'''
TD extended to control by learning action-value function Q(s,a)
The SARSA update rule is:
Q(s_t, a_t) = Q(s_t, a_t) + alpha*(r_t+1 +gamma*Q(s_t+1, a_t+1) - Q(s_t, a_t))
'''

#TD(0) for value estimation

import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery = False)

alpha = 0.1
gamma = 1.0
num_episodes = 10000
V = np.zeros(env.observation_space.n)

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample() # Random policy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #TD(0) update
        V[state] = V[state] + alpha*(reward+gamma*V[next_state]-V[state])
        state = next_state

print("Value function: ")
print(V.reshape((4,4)))

#SARSA for control
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1
gamma = 1.0
num_episodes = 10000
epsilon = 0.1

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[state])

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[next_state])

        #SARSA update
        Q[state, action] = Q[state, action] + alpha*(reward+gamma*Q[next_state, next_action]-Q[state, action])
        state = next_state
        action = next_action

print("Action-value function: ")
print(Q)

#Add Stochasticity to the environment
env = gym.make('FrozenLake-v1', is_slippery = True)

#TD(0)
alpha = 0.1
gamma = 1.0
num_episodes = 10000
V = np.zeros(env.observation_space.n)

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample() # Random policy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #TD(0) update
        V[state] = V[state] + alpha*(reward+gamma*V[next_state]-V[state])
        state = next_state

print("Value function: ")
print(V.reshape((4,4)))

#SARSA for control
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1
gamma = 1.0
num_episodes = 10000
epsilon = 0.1

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[state])

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[next_state])

        #SARSA update
        Q[state, action] = Q[state, action] + alpha*(reward+gamma*Q[next_state, next_action]-Q[state, action])
        state = next_state
        action = next_action

print("Action-value function: ")
print(Q)

#TD(lambda) (eligibility traces) generalizes TD(0) by combining multiple updates
alpha = 0.1
gamma = 1.0
lambda_ = 0.9
num_episodes = 10000
V = np.zeros(env.observation_space.n)
eligibility = np.zeros(env.observation_space.n)

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #TD(lambda) update
        td_error = reward + gamma*V[next_state] - V[state]
        eligibility[state] += 1
        V += alpha*td_error*eligibility
        eligibility *= gamma*lambda_
        state = next_state

print("Value function: ")
print(V.reshape((4,4)))

#Q(lambda) for control
Q = np.zeros((env.observation_space.n, env.action_space.n))
eligibility = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[state])

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[next_state])

        #Q(lambda) update
        td_error = reward + gamma*Q[next_state, next_action] - Q[state, action]
        eligibility[state, action] += 1
        Q += alpha*td_error*eligibility
        eligibility *= gamma*lambda_
        state = next_state
        action = next_action

print("Action-value function: ")
print(Q)

# Apply to continuous state spaces

env = gym.make('MountainCar-v0')

alpha = 0.1
gamma = 1.0
lambda_ = 0.9
num_episodes = 10000
# Discretize continuous state space for MountainCar
def discretize_state(state, bins=(20, 20)):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    ratios = (state - env_low) / (env_high - env_low)
    new_state = (ratios * np.array(bins)).astype(int)
    new_state = np.clip(new_state, 0, np.array(bins) - 1)
    return tuple(new_state)

n_bins = (20, 20)
V = np.zeros(n_bins)
eligibility = np.zeros(n_bins)  #eligibility traces

for episode in range(num_episodes):
    state, _ = env.reset()
    state_disc = discretize_state(state, bins=n_bins)
    done = False

    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state_disc = discretize_state(next_state, bins=n_bins)
        done = terminated or truncated

        #TD(lambda) update with discretized state indices
        td_error = reward + gamma * V[next_state_disc] - V[state_disc]
        eligibility[state_disc] += 1
        V += alpha * td_error * eligibility
        eligibility *= gamma * lambda_
        state_disc = next_state_disc

print("Value function: ")
print(V.reshape(n_bins))

#Q(lambda) for control in continuous state spaces with discretization
# Adjust Q to handle discretized state space: dimensions (n_bins[0], n_bins[1], number of actions)
Q = np.zeros((n_bins[0], n_bins[1], env.action_space.n))
eligibility = np.zeros((n_bins[0], n_bins[1], env.action_space.n))

for episode in range(num_episodes):
    state, _ = env.reset()
    state_disc = discretize_state(state, bins=n_bins)
    done = False
    if np.random.rand() < epsilon:
        action = np.random.choice(env.action_space.n)
    else:
        action = np.argmax(Q[state_disc])
    
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state_disc = discretize_state(next_state, bins=n_bins)
        done = terminated or truncated
        if np.random.rand() < epsilon:
            next_action = np.random.choice(env.action_space.n)
        else:
            next_action = np.argmax(Q[next_state_disc])
    
        #Q(lambda) update using tuple indexing for discretized state-action pairs
        td_error = reward + gamma * Q[next_state_disc + (next_action,)] - Q[state_disc + (action,)]
        eligibility[state_disc + (action,)] += 1
        Q += alpha * td_error * eligibility
        eligibility *= gamma * lambda_
        state_disc = next_state_disc
        action = next_action

print("Action-value function: ")
print(Q)