import numpy as np

#Markov Decision Process parameters
gamma = 0.9
theta = 1e-6

#(4x4) Grid World
states = np.arange(16)
actions = [0,1,2,3,4]
rewards = np.full(16, -1)
rewards[0] = rewards[15] = 0
terminal_states = [0,15]

#Transition probabilities
def next_state(s,a):
    if s in terminal_states:
        return s
    row, col = divmod(s,4)
    if a == 0:
        col = max(col-1,0)
    elif a == 1:
        row = max(row-1,0)
    elif a == 2:
        col = min(col+1,3)
    elif a == 3:
        row = min(row+1,3)
    return row*4 + col

#initialize policy
policy = np.random.choice(actions, len(states))
V = np.zeros(len(states))


#Policy iteration
while True:
    #Policy evaluation
    while True:
        delta = 0
        for s in states:
            if s in terminal_states:
                continue
            v_old = V[s]
            a = policy[s]
            s_prime = next_state(s,a)
            V[s] = rewards[s]+gamma*V[s_prime]
            delta = max(delta, abs(v_old-V[s]))
        if delta < theta:
            break

    #Policy improvement
    policy_stable = True
    for s in states:
        if s in terminal_states:
            continue
        old_action = policy[s]
        #find action maximises return
        policy[s] = max(actions, key=lambda a: rewards[s] + gamma * V[next_state(s,a)])
        if old_action != policy[s]:
            policy_stable = False

    if policy_stable:
        break


print("Optimal Value Function:\n", V.reshape(4,4))
print("Optimal Policy:", policy.reshape(4,4))


#Stochastic Policy Iteration
tau = 0.5 #temperature

#Initialize stochastic policy
policy = np.full((len(states), len(actions)), 1/len(actions))
V = np.zeros(len(states))

while True:
    #Policy evaluation
    while True:
        delta = 0
        new_v = np.copy(V)
        for s in states:
            if s in terminal_states:
                continue
            new_v[s] = sum(policy[s,a]*(rewards[s]+gamma*V[next_state(s,a)])for a in actions)
            delta = max(delta, abs(new_v[s]-V[s]))
        V = new_v
        if delta < theta:
            break

    #Policy improvement
    Q = np.zeros((len(states), len(actions)))
    for s in states:
        for a in actions:
            Q[s,a] = rewards[s] + gamma * V[next_state(s,a)]

    new_policy = np.zeros_like(policy)
    for s in states:
        if s in terminal_states:
            continue
        #Compute softmax probabilities
        exp_values = np.exp((Q[s]/tau))
        new_policy[s]= exp_values/np.sum(exp_values)

    if np.allclose(policy, new_policy, atol=1e-6):
        break

    policy = new_policy


print("Optimal Value Function:\n", V.reshape(4,4))
print("Optimal Policy:", np.argmax(policy, axis=1).reshape(4,4))


#Large Grid World
states = np.arange(120)
actions = [0,1,2,3,4]
rewards = np.full(120, -1)
rewards[0] = rewards[119] = 0
terminal_states = [0,119]

#Transition probabilities
def next_state(s,a):
    if s in terminal_states:
        return s
    row, col = divmod(s,10)
    if a == 0:
        col = max(col-1,0)
    elif a == 1:
        row = max(row-1,0)
    elif a == 2:
        col = min(col+1,9)
    elif a == 3:
        row = min(row+1,9)
    return row*10 + col

#Stochastic Policy Iteration
tau = 0.5 #temperature

#Initialize stochastic policy
policy = np.full((len(states), len(actions)), 1/len(actions))
V = np.zeros(len(states))

while True:
    #Policy evaluation
    while True:
        delta = 0
        new_v = np.copy(V)
        for s in states:
            if s in terminal_states:
                continue
            new_v[s] = sum(policy[s,a]*(rewards[s]+gamma*V[next_state(s,a)])for a in actions)
            delta = max(delta, abs(new_v[s]-V[s]))
        V = new_v
        if delta < theta:
            break

    #Policy improvement
    Q = np.zeros((len(states), len(actions)))
    for s in states:
        for a in actions:
            Q[s,a] = rewards[s] + gamma * V[next_state(s,a)]

    new_policy = np.zeros_like(policy)
    for s in states:
        if s in terminal_states:
            continue
        #Compute softmax probabilities
        exp_values = np.exp((Q[s]/tau))
        new_policy[s]= exp_values/np.sum(exp_values)

    if np.allclose(policy, new_policy, atol=1e-6):
        break

    policy = new_policy

print("Optimal Value Function:\n", V.reshape(10,12))
print("Optimal Policy:", np.argmax(policy, axis=1).reshape(10,12))

