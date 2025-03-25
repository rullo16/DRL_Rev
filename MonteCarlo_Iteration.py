import gymnasium as gym
import numpy as np

env = gym.make('Blackjack-v1')

#Init Q table
Q = {}
returns = {}
gamma = 1.0
num_episodes = 100000

for state in range(32):  # Possible states in Blackjack (sum of cards)
    for dealer in range(11):  # Dealer's showing card (1-10)
        for ace in [True, False]:  # Usable ace or not
            for action in range(env.action_space.n):
                Q[((state, dealer, ace), action)] = 0
                returns[((state, dealer, ace), action)] = []

# First-visit Monte Carlo policy evaluation

for eps in range(num_episodes):
    state = env.reset()[0]
    episode_data = []
    done = False

    while not done:
        action = np.random.choice(env.action_space.n)
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state

    #Compute returns and update Q-values 
    G = 0
    visited = set()
    for state, action, reward in reversed(episode_data):
        G = reward+gamma*G
        if (state, action) not in visited:
            visited.add((state, action))
            returns[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns[(state, action)])

print("Sample Q-values: ")
for key, value in list(Q.items())[:10]:
    print(f"State: {key[0]}, Action: {key[1]}, Value: {value:.2f}")


# Epsilon-greedy MonteCarlo Control
#Init Q table
Q = {}
returns = {}
gamma = 1.0
num_episodes = 100000
epsilon = 0.1

for state in range(32):  # Possible states in Blackjack (sum of cards)
    for dealer in range(11):  # Dealer's showing card (1-10)
        for ace in [True, False]:  # Usable ace or not
            for action in range(env.action_space.n):
                Q[((state, dealer, ace), action)] = 0
                returns[((state, dealer, ace), action)] = []

# First-visit Monte Carlo policy evaluation
for eps in range(num_episodes):
    state = env.reset()[0]
    episode_data = []
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax([Q[(state, action)] for action in range(env.action_space.n)])
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state

    #Compute returns and update Q-values
    G = 0
    visited = set()
    for state, action, reward in reversed(episode_data):
        G = reward+gamma*G
        if (state, action) not in visited:
            visited.add((state, action))
            returns[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns[(state, action)])


print("Sample epsilon-greedy Q-values: ")
for key, value in list(Q.items())[:10]:
    print(f"State: {key[0]}, Action: {key[1]}, Value: {value:.2f}")


#Off-policy Monte Carlo Control

#Init Q table
Q = {}
C = {}
gamma = 1.0
num_episodes = 100000
epsilon = 0.1

for state in range(32):  # Possible states in Blackjack (sum of cards)
    for dealer in range(11):  # Dealer's showing card (1-10)
        for ace in [True, False]:  # Usable ace or not
            for action in range(env.action_space.n):
                Q[((state, dealer, ace), action)] = 0
                C[((state, dealer, ace), action)] = 0

# First-visit Monte Carlo policy evaluation
for eps in range(num_episodes):
    state = env.reset()[0]
    episode_data = []
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax([Q[(state, action)] for action in range(env.action_space.n)])
            
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state

    #Compute returns and update Q-values
    G = 0
    W = 1
    for state, action, reward in reversed(episode_data):
        G = reward+gamma*G
        C[(state, action)] += W
        Q[(state, action)] += (W/C[(state, action)] * (G-Q[(state, action)]))
        if action != np.argmax([Q[(state, action)] for action in range(env.action_space.n)]):
            break
        W = W/epsilon

print("Sample off-policy Q-values: ")

for key, value in list(Q.items())[:10]:
    print(f"State: {key[0]}, Action: {key[1]}, Value: {value:.2f}")
