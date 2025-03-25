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
    

#Value iteration
V = np.zeros(16)

while True:
    delta = 0
    V_new = np.copy(V)
    for s in states:
        if s in [0,15]:
            continue
        V_new[s] = max([rewards[s] + gamma * V[next_state(s,a)] for a in actions])
        delta = max(delta, abs(V_new[s]-V[s]))
    V = V_new
    if delta < theta:
        break

policy = {s: max(actions, key=lambda a: rewards[s] + gamma * V[next_state(s,a)]) for s in states}
print("Optimal Value Function:\n", V.reshape(4,4))
print("Optimal Policy:", policy)


#Action-Value Function iteration
Q = np.zeros((16,5))

while True:
    delta = 0
    Q_new = np.copy(Q)
    for s in states:
        for a in actions:
            if s in [0,15]:
                Q_new[s,a] = rewards[s]
            else:
                s_prime = next_state(s,a)
                Q_new[s,a] = max([rewards[s]+gamma*max(Q[s_prime])])
            delta = max(delta, abs(Q_new[s,a]-Q[s,a]))
    Q = Q_new
    if delta < theta:
        break

optimal_policy = {s: np.argmax(Q[s]) for s in states}

print("Optimal Action-Value Function:\n", Q.reshape(16,5))
print("Optimal Policy:", optimal_policy)