#Q-Learning (off-policy control)
'''
Q-learning is an off-policy TD control algorithm that learns the optimal action-value function:
Q*(s,a)
without following the policy it's evaluating.
Key characteristics are:
    -Off-policy nature:
        -Evaluates optimal policy while following a different, exploratory policy (epsilon-greedy)
        -Update rule uses the maximum action-value from the next state, regardless of which action the agent actually takes
    - Update rule:
        Q(s_t, a_t) = Q(s_t,a_t) + alpha*(R_t+1 + gamma*max_a'Q(s_t+1, a') - Q(s_t, a_t))
        - alpha: learning rate
        - gamma: discount factor
        - R_t+1: reward received after taking action a_t in state s_t
        - max_a'Q(s_t+1, a'): maximum action-value from the next state
    -Convergence:
        - Under certain conditions (sufficient exploration, decaying learning rate), Q-learning converges to the optimal action-value function Q*(s,a)
'''

#Q-Learning implementation
import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', is_slippery = False)

#Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

#Hyperparameters
alpha = 0.1
gamma = 1.0
epsilon = 0.1
epsilon_decay = 0.999
num_episodes = 10000
epsilon_min = 0.01

#Q-learning loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #Q-learning update
        best_next_action = np.argmax(Q[next_state])
        Q[state, action] += alpha*(reward+gamma*Q[next_state, best_next_action]-Q[state, action])
        state = next_state

    epsilon = max(epsilon*epsilon_decay, epsilon_min)

#Extract policy from Q-table
policy = np.argmax(Q, axis=1)
print("Policy: ")
print(policy.reshape((4,4)))
#Display optimal learned Q-values
print("Q-values: ")
print(Q.reshape((16,4)))

#Double Q-Learning
'''
Double Q-learning is an extension addressing overestimation bias
maintains two Q-tables, Q1 and Q2, and uses one to select the best action and the other to evaluate it
The update rule is:
    Q1(s_t, a_t) += Q1(s_t, a_t) + alpha*(r_t+1 + gamma*Q2(s_t+1, argmax_a'Q1(s_t+1, a')) - Q1(s_t, a_t))
    Q2(s_t, a_t) += Q2(s_t, a_t) + alpha*(r_t+1 + gamma*Q1(s_t+1, argmax_a'Q2(s_t+1, a')) - Q2(s_t, a_t))
'''

#Double Q-Learning implementation
Q1 = np.zeros((num_states, num_actions))
Q2 = np.zeros((num_states, num_actions))

#Hyperparameters
alpha = 0.1
gamma = 1.0
epsilon = 0.1
epsilon_decay = 0.999
num_episodes = 10000
epsilon_min = 0.01

#Double Q-learning loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q1[state]+Q2[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #Double Q-learning update
        if np.random.rand() < 0.5:
            best_next_action = np.argmax(Q1[next_state])
            Q1[state, action] += alpha*(reward+gamma*Q2[next_state, best_next_action]-Q1[state, action])
        else:
            best_next_action = np.argmax(Q2[next_state])
            Q2[state, action] += alpha*(reward+gamma*Q1[next_state, best_next_action]-Q2[state, action])

        state = next_state

    epsilon = max(epsilon*epsilon_decay, epsilon_min)

#Extract policy from Q-table
policy = np.argmax(Q1+Q2, axis=1)
print("Policy: ")
print(policy.reshape((4,4)))
#Display optimal learned Q-values
print("Q-values: ")
print((Q1+Q2).reshape((16,4)))

#Dynamic exploration
'''
Implementation of different exploration strategies:
    -Epsilon-greedy: Random action with probability epsilon, otherwise best action
    -Boltzmann exploration: Action probabilities proportional to exponentiated action-values
    -Softmax: Action probabilities proportional to exponentiated action-values divided by temperature

Boltzmann (Softmax) exploration:
    - Action probabilities:
        P(a_t = a|s_t) = exp(Q(s_t, a)/tau) / sum_a'(exp(Q(s_t, a')/tau))
    - Update rule (expected SARSA style):
        Q(s_t, a_t) = Q(s_t, a_t) + alpha*(r_t+1 + gamma*sum_a'(P(a'|s_t+1)*Q(s_t+1, a')) - Q(s_t, a_t))
    - Convergence:
        Under certain conditions (sufficient exploration and decaying learning rate), this method converges to the optimal action-value function Q*(s,a)
'''

#Boltzmann exploration implementation
Q = np.zeros((num_states, num_actions))

#Hyperparameters
alpha = 0.1
gamma = 1.0
tau = 1.0
tau_decay = 0.999
num_episodes = 10000
tau_min = 0.01

#Boltzmann exploration loop

def boltzmann_exploration(Q, state, tau):
    action_values = Q[state]/tau
    action_probs = np.exp(action_values)/np.sum(np.exp(action_values))
    return action_probs

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action_probs = boltzmann_exploration(Q, state, tau)
        action = np.random.choice(np.arange(num_actions), p=action_probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #Boltzmann update
        next_action_probs = boltzmann_exploration(Q, next_state, tau)
        Q[state, action] += alpha*(reward+gamma*np.dot(next_action_probs, Q[next_state])-Q[state, action])
        state = next_state

    tau = max(tau*tau_decay, tau_min)

#Extract policy from Q-table
policy = np.argmax(Q, axis=1)
print("Policy: ")
print(policy.reshape((4,4)))
#Display optimal learned Q-values
print("Q-values: ")
print(Q.reshape((16,4)))