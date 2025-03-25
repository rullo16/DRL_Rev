import torch

#Value Iteration
#MDP params
gamma = 0.9
theta = 1e-5

#State space
states = torch.arange(120)
actions = torch.tensor([0,1,2,3,4])
rewards = torch.full((120,), -1)
rewards[0] = rewards[119] = 0
terminal_states = torch.tensor([0,119])

#Transition model (transition probabilities)
def next_state(s,a):
    if s in terminal_states:
        return s
    row, col = divmod(s.item(), 10)
    if a == 0:
        col = max(col-1,0)
    elif a == 1:
        row = max(row-1,0)
    elif a == 2:
        col = min(col+1,9)
    elif a == 3:
        row = min(row+1,9)
    return row*10 + col

#Value Iteration
V = torch.zeros(len(states))
while True:
    delta = 0
    new_v = torch.zeros_like(V)
    for s in states:
        if s in terminal_states:
            continue
        new_v[s] = max([rewards[s]+ gamma * V[next_state(s,a)] for a in actions])
        delta = max(delta, abs(new_v[s]-V[s]))
    V = new_v
    if delta < theta:
        break

#Extract policy
policy = {s: max(actions, key=lambda a: rewards[s]+gamma*V[next_state(s,a)]) for s in states if s not in terminal_states}

#Print Optimal Value Function
print("Optimal Value Function:\n", V.reshape(10,12))
print("Optimal Policy:", policy)

#Action Value Iteration
Q = torch.zeros((120,5))

while True:
    delta = 0
    new_q = torch.zeros_like(Q)
    for s in states:
        for a in actions:
            if s in terminal_states:
                new_q[s,a] = rewards[s]
            else:
                s_prime = next_state(s,a)
                new_q[s,a] = rewards[s] + gamma * max(Q[s_prime])
            delta = max(delta, abs(new_q[s,a]-Q[s,a]))
    Q = new_q
    if delta < theta:
        break

optimal_policy = {s: torch.argmax(Q[s]) for s in states if s not in terminal_states}

print("Optimal Action-Value Function:\n", Q.reshape(10,12,5))
print("Optimal Policy:", optimal_policy)


#Policy Iteration
policy = torch.multinomial(torch.ones(len(states), len(actions)), 1).squeeze()
V = torch.zeros(len(states))

while True:
    while True:
        delta = 0
        for s in states:
            if s in terminal_states:
                continue
            old_v = V[s]
            a = policy[s]
            s_prime = next_state(s,a)
            V[s] = rewards[s] + gamma * V[s_prime]
            delta = max(delta, abs(old_v-V[s]))
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

print("Optimal Value Function:\n", V.reshape(10,12))
print("Optimal Policy:", policy.tolist())


#Stochastic Policy Iteration
tau = 0.5 #temperature

policy = torch.full((len(states), len(actions)), 1/len(actions))
V = torch.zeros(len(states))

while True:
    while True:
        delta = 0
        new_v = torch.zeros_like(V)
        for s in states:
            if s in terminal_states:
                continue
            new_v[s] = sum(policy[s,a]*(rewards[s]+gamma*V[next_state(s,a)])for a in actions)
            delta = max(delta, abs(new_v[s]-V[s]))
        V = new_v
        if delta < theta:
            break

    Q = torch.zeros((len(states), len(actions)))
    for s in states:
        for a in actions:
            Q[s,a] = rewards[s] + gamma * V[next_state(s,a)]

    new_policy = torch.zeros_like(policy)
    for s in states:
        new_policy[s] = torch.nn.functional.softmax(Q[s]/tau, dim=0)
    
    if torch.allclose(new_policy, policy):
        break
    policy = new_policy

print("Optimal Value Function:\n", V.reshape(10,12))
print("Optimal Policy:", torch.argmax(policy))