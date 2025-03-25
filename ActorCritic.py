#Actor Critic Methods
'''
Combine the strenghts of policy gradients (actor) and value function approximation (critic).
Instead of using only returns or TD errors, the actor relies on feedvack from the critic to improve the policy.
Key Components:
    - Actor and Critic:
        -Actor: A neural network that outputs actions or action probabilities. It's trained to improve the policy pi(a|s;theta)
        -Critic: A neural network that estimates a value function (state value V(s) or action value Q(s,a)). It provides a learned baseline for the actor's updates.
    - Advantage Function:
        -The advantage measures how much better a chosen action is compared to the average action:
            A(s,a) = Q(s,a) - V(s)
        For simplicity, the advantage can be approximated using the TD error: A(s,a) = r + gamma*V(s') - V(s)
    - Actor-Critic Update steps:
        -Critic Update: Minimizes the error between its value estimate and the observed return or TD target: Loss_critic = (r + gamma*V(s') - V(s))^2
        -Actor Update: Uses critic's feedback (advantage) to adjust the policy: Loss_actor = -log(pi(a|s;theta)A(s,a))
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

#Implementation Actor-Critic

class Actor(nn.Module):
    def __init__(self, input_dim, out_dim, continuous=False):
        super(Actor, self).__init__()
        self.continuous = continuous
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)
        if not continuous:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if self.continuous:
            return x
        else:
            return self.softmax(x)
    
class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def training_actor_critic(env, actor, critic, actor_optim, critic_optim, num_episodes=500, gamma=0.99):
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            #Get action probs and value estim
            action_probs = actor(state_tensor)
            value = critic(state_tensor)

            #Sample action from policy
            action = torch.multinomial(action_probs, num_samples=1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state=next_state

        #Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma*G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()

        #Actor Loss
        actor_loss = (-torch.stack(log_probs)*advantages).sum()

        #Critic Loss
        critic_loss = ((returns-values)**2).mean()

        #Update networks
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Tot Reward: {sum(rewards)}")

# env = gym.make('CartPole-v1')
# actor = Actor(env.observation_space.shape[0], env.action_space.n)
# critic = Critic(env.observation_space.shape[0])
# actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
# critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

# training_actor_critic(env, actor, critic, actor_optim, critic_optim, num_episodes=500)
# env.close()

'''
-Actor-Critic updates occur at each time step
-Critic provides a baseline (value estimate), which reduces variance in the policy gradient updates
-Advantage term A(s,a) guides the actor, making the learning process more stable and efficient.
'''

# Continuous Action Spaces

env = gym.make('MountainCarContinuous-v0')
actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], continuous=True)
critic = Critic(env.observation_space.shape[0])
actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

def training_actor_critic_continuous(env, actor, critic, actor_optim, critic_optim, num_episodes=500, gamma=0.99, std=0.1):
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # The actor now outputs a mean value for the action.
            mean = actor(state_tensor)
            value = critic(state_tensor)
            # Use a Gaussian distribution with mean from the actor and fixed std.
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            # Clip action to the environment's bounds.
            action_clipped = action.clamp(torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))
            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy()[0])
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state

        # Compute returns.
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()

        # Actor Loss: policy gradient weighted by the advantage.
        actor_loss = (-torch.stack(log_probs) * advantages).sum()
        # Critic Loss: mean squared error.
        critic_loss = ((returns - values) ** 2).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Reward: {sum(rewards)}")

# training_actor_critic_continuous(env, actor, critic, actor_optim, critic_optim, num_episodes=500)
# env.close()

# TD(lambda) as Critic

def train_actor_critic_td_lambda(env, actor, critic, actor_optim, critic_optim, num_episodes=500, gamma=0.99, lambda_=0.9, alpha=0.01):

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False
        
        eligibility_traces = torch.zeros_like(torch.cat([p.view(-1) for p in critic.parameters()]))

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_mean = actor(state_tensor)
            value = critic(state_tensor)

            dist = torch.distributions.Normal(action_mean, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            action_clipped = action.clamp(torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))
            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy()[0])
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            #Compute TD error

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                next_value = critic(next_state_tensor) if not done else torch.tensor([0.0])
            td_error = reward+gamma*next_value-value

            #Update eligibility traces safely by detaching gradients
            gradients = torch.autograd.grad(value, critic.parameters(), retain_graph=True)
            flattened_grads = torch.cat([g.view(-1).detach() for g in gradients])
            eligibility_traces = gamma * lambda_ * eligibility_traces + flattened_grads

            #Update critic
            critic_params = torch.cat([p.view(-1) for p in critic.parameters()])
            critic_params += alpha * td_error.item() * eligibility_traces
            idx = 0

            for p in critic.parameters():
                param_shape = p.data.shape
                param_size = p.numel()
                p.data = critic_params[idx:idx+param_size].view(param_shape)
                idx += param_size

            state = next_state

        #Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r+gamma*G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()
        advantages = returns-values.detach()

        actor_loss = (-torch.stack(log_probs)*advantages).sum()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Actor Loss: {actor_loss.item():.4f}, Total Reward: {sum(rewards)}")

# train_actor_critic_td_lambda(env, actor, critic, actor_optim, critic_optim)
# env.close()

# env=gym.make('LunarLanderContinuous-v3')

# actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], continuous=True)
# critic = Critic(env.observation_space.shape[0])
# actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
# critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

# train_actor_critic_td_lambda(env, actor, critic, actor_optim, critic_optim)
# env.close()

# Asynchronous Actor-Critic(A3C)
'''
Reinforcement Learning algorithm that trains multiple agents(workers) in parallal. These agetns interact independently with the environment, 
allowing for faster exploration and more diverse experiences. The gradients from these agents are asynchronously aggregated to update a shared global model.
This reduces training time and stabilizes learning compared to single-agent methods.
Key Components:
    - Global Model: Shared NN storing params for policy and value function. Workers use this global model to make decisions and periodically push gradients to it.
    - Workers: Each worker has its own copy of the model. The local model interacts with its environment, collects trajectories, computes gradients, and then sends these gradients to update the global mdoels.
    - Advantage Estimation: Workers compute the advantage using their local value estimates: A(s,a) = sum^T(gammma^t*r_t)-V(s)
    - Asynchronous Updates: Workers run independently, sending gradients back to the global model. The global model updates its parameters asynchronously and then sends them back to workers.
Advantages:
    -Faster exploration
    -Increased stability
    -Scalability

Overall Loss : L = L_actor + c1*L_critic + c2*L_entropy. c1 and c2 are constants that control the balance between the actor and critic losses and the entropy bonus.
'''

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.actor(x), self.critic(x)
    

def worker(global_model, optimizer, env_name, worker_id, num_steps, gamma):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    local_model = ActorCritic(state_dim, action_dim)
    local_model.load_state_dict(global_model.state_dict())

    for episode in range(1000):
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []

        for step in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, value = local_model(state_tensor)
            # Optimize: use Categorical distribution for discrete env
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done:
                break

        returns = []
        G = 0 if done else local_model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))[1].item()
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        # Fix typo: use detach() not detatch()
        advantage = returns - values.detach()

        actor_loss = (-log_probs * advantage).mean()
        critic_loss = ((returns - values) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()

        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param.grad = local_param.grad

        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

        print(f"Worker {worker_id}, Episode {episode}, Total Reward: {sum(rewards)}")

import torch.multiprocessing as mp

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
global_model = ActorCritic(state_dim, action_dim)
global_model.share_memory()
optimizer = optim.Adam(global_model.parameters(), lr=1e-3)

num_workers = 4
workers = []

for worker_id in range(num_workers):
    worker_process = mp.Process(target=worker, args=(global_model, optimizer, env_name, worker_id, 20, 0.99))
    workers.append(worker_process)
    worker_process.start()

for worker_process in workers:
    worker_process.join()

env.close()

