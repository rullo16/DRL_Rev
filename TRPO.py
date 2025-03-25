# Trust Region Policy Optimization (TRPO) implementation
'''
Trust Region Policy Optimization (TRPO) is a reinforcement learning algorithm belonging to the family of policy gradient methods. Introduced by Schulman, it aims to improve the stability and reliability of policy gradient training. The key idea is to restrict policy updates to a trust region, ensuring that the new policy is not too far from the old policy in probability space. By limiting the size of policy changes using Kullback-Leibler (KL) divergence, TRPO avoids the large destructive updates that can plague vanilla policy gradient methods. This results in more stable learning and theoretically guaranteed monotonic improvements in performance. In practice, TRPO has demonstrated robust performance across various tasks. Its ability to handle both discrete and continuous action spaces and to yield reliable progress with minimal hyperparameter tuning made it an important milestone in deep reinforcement learning (DRL).

Theoretical Background:

Policy Gradient Basics:
In a reinforcement learning setting modeled by a Markov Decision Process (MDP), the goal is to learn a policy, π(a|s), that maximizes the expected cumulative reward. We define the policy's performance as:

    J(θ) = E[Σ_t γ^t * r_t],

the expected discounted reward. Policy gradient algorithms seek to improve J(θ) by updating θ in the direction of:

    ∇_θ J(θ) = E_{s_t,a_t ~ π_θ} [∇_θ log(π_θ(a_t|s_t)) * A^{π(θ)}(s_t,a_t)].

Here, A^{π(θ)}(s_t,a_t) is the advantage function, which measures how much better an action, a_t, is at state, s_t, compared to the policy's average performance at s_t. Intuitively, the policy gradient is an expectation of the policy's log-likelihood weighted by advantage: actions with positive advantage are up-weighted (chosen more), and negative advantage actions are down-weighted (chosen less).

Need for Trust Region:
Vanilla policy gradient methods (REINFORCE) often use simple update rules such as:

    θ <- θ + α * ∇_θ J(θ),

with a step size α. However, because the policy's behavior can change dramatically with a large parameter step, too large an update can collapse performance. For example, a slight change in neural network weights might cause the agent to choose completely different (and worse) actions. To mitigate this, TRPO imposes a constraint on how far the new policy can deviate from the old one in terms of KL divergence (a measure of difference between probability distributions). Instead of keeping updates small in the parameter space, TRPO keeps updates small in the policy distribution space. This means the behavior of the policy won't shift too abruptly, providing a sort of guarantee that each update will not drastically worsen performance. TRPO was derived with the theoretical goal of guaranteeing monotonic improvement—each policy update should never decrease the expected reward.

Constrained Optimization Formulation:
Formally, the idealized TRPO update at iteration k solves the following constrained optimization problem:
- Maximize the surrogate objective L(θ_k, θ), which is the expected advantage of the new policy π_θ relative to the old policy π_{θ_k}:
- Subject to a constraint that the average KL divergence between π_{θ_k} and π_θ is below a small threshold δ.

In notation:

    θ_{k+1} = argmax_θ L(θ_k, θ) s.t. D_{KL}(π_{θ_k} || π_θ) < δ,

where the surrogate objective is:

    L(θ_k, θ) = E_{s_t,a_t ~ π_{θ_k}}[π_θ(a_t|s_t) / π_{θ_k}(a_t|s_t) * A^{π(θ_k)}(s_t,a_t)],

and the constraint uses the average KL divergence under the old policy state distribution:

    D_{KL}(π_{θ_k} || π_θ) = E_{s_t ~ π_{θ_k}}[D_{KL}(π_{θ_k}(.|s_t) || π_θ(.|s_t))] <= δ.

We look for new parameters θ that would improve performance on the data sampled from the old policy while ensuring the new policy π_θ does not stray too far from π_{θ_k}. The ratio π_θ(a_t|s_t) / π_{θ_k}(a_t|s_t) in L is essentially an importance sampling term that corrects for the fact we use data from the old policy to evaluate the new policy. If this ratio is greater than 1, the new policy is assigning higher probability to that (s, a) than the old policy, and vice versa.

Deriving the TRPO Update (Trust Region Step):
Solving the constrained optimization exactly for neural networks is difficult. TRPO makes a series of approximations to derive a practical update. First, assume θ is close to θ_k so that we can linearize the objective and quadratic-approximate the constraint. Using first-order Taylor expansion for L and a second-order expansion for KL around θ_k:

    - L(θ_k, θ) ≈ g^T(θ - θ_k) where g = ∇_θ L(θ_k, θ) which is the policy gradient.
    - D_{KL}(π_{θ_k} || π_θ) ≈ 1/2 (θ - θ_k)^T H (θ - θ_k), where H is the Hessian (second derivative matrix) of the KL divergence at θ_k, also known as the Fisher information matrix measuring the curvature of the policy.

With these approximations, the constrained problem becomes a tractable quadratic program: maximize g^T(θ - θ_k) subject to 1/2 (θ - θ_k)^T H (θ - θ_k) <= δ. This is now the classic form of a trust-region constrained optimization (linear objective with a quadratic constraint). Using Lagrange multipliers, one can solve this analytically:

    θ_{k+1} = θ_k + √(2δ / g^T H^-1 g) * H^-1 g,

which says the update direction is proportional to H^-1 g, scaled such that the quadratic form (θ_{k+1} - θ_k)^T H (θ_{k+1} - θ_k) = δ.
In simpler terms, we take a step in the direction of the natural policy gradient (scale the gradient by the inverse Fisher information matrix, accounting for the geometry of the policy space)
and adjust the step size to exactly satisfy the KL divergence limit.
This natural gradient step is crucial as it effectively chooses a step size that is as large as possible for improvement while still honoring the trust region. If we stopped here,
this would be equivalent to the Natural Policy Gradient algorithm. However, due to the approximations we made, the resulting step might not strictly satisfy the original constraint or might not actually improve L
when applied to the non-linear, non-quadratic real objective. TRPO therefore introduces a backtracking line search: start with the full step Δθ = √(2δ / g^T H^-1 g) * H^-1 g
and check the actual new policy. If it violates the KL constraint or fails to improve the surrogate objective, scale the step down until the constraint is satisfied. In practice, one uses a backtracking coefficient (α=0.8)
and tries a few reductions (α^1, α^2,...) until the new policy π_{θ_k + Δθ} has D_{KL} < δ and does not decrease L. This ensures the theoretical guarantee.

Efficiently Computing the Update:
A major challenge is computing H^-1*g. H is as large as the number of policy parameters, so inverting it explicitly is infeasible. TRPO leverages an iterative technique called conjugate gradient (CG)
to find x = H^-1*g without forming H explicitly. CG only requires being able to compute the matrix-vector product H*v for arbitrary vectors v. Fortunately, for the KL's Hessian, one can compute H*v by a double differentiation trick:

    H*v = ∇_θ(∇_θ D_{KL}(θ_k || θ)^T*v),

which can be implemented by automatic differentiation without ever forming H. 
In code, we can obtain the gradient of KL with respect to parameters, take a directional dot product with some vector v, and then differentiate that scalar again to get H*v. 
This procedure, combined with conjugate gradient iteration, efficiently yields the search direction H^-1*g. Typically, a small damping term ω*I is added to H to improve numerical stability
and ensure H is positive definite. This means we actually solve (H + ω*I)^-1*g, to avoid issues if H is singular or nearly so. Once the direction is obtained and scaled, the line search is performed as described.

Additionally, TRPO uses a baseline (often a learned value function V_φ(s)) to compute advantages A(s,a)=Q(s,a)-V_φ(s) reducing variance in the policy gradient estimate.
An advanced option is GAE.
''' 

# Implementation TRPO

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

#Hyperparameters
gamma = 0.99
lam = 0.97 #GAE lambda
delta =  0.01 #KL constraint
cg_iters = 10 #Conjugate gradient iterations
cg_damping = 1e-2 #Conjugate gradient damping, damping fator for Hessian
backtrack_iters = 10 #Backtrack iterations, maxi iterations for line search
backtrack_coeff = 0.8 #Backtrack coefficient, step reduction for line search
vf_iters = 5 #Value function iterations
vf_lr = 1e-3 #Value function learning rate

#Policy network
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.logits = nn.Linear(64, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.logits(x)
        return x
    
#Value network
class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.v(x)
        return x
    
policy = Policy(obs_dim, n_acts)
vf = ValueNet(obs_dim)
optimizer_vf = optim.Adam(vf.parameters(), lr=vf_lr)

def collect_trajectories(policy, env, batch_size =1000):
    trajectories = []
    obs = env.reset()[0]
    done = False
    steps = 0

    while steps < batch_size:
        state = torch.tensor(obs, dtype=torch.float32)
        logits = policy(state)
        dist = distributions.Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(torch.tensor(act)).item()

        next_obs, rew, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        if len(trajectories) == 0 or 'obs' not in trajectories[-1]:
            trajectories.append({'obs': obs, 'act': act, 'rew': rew, 'logp': logp})
        trajectories[-1]['obs'].append(obs)
        trajectories[-1]['act'].append(act)
        trajectories[-1]['rew'].append(rew)
        trajectories[-1]['logp'].append(logp)

        obs = next_obs
        steps += 1
        if done:
            obs = env.reset()[0]
            done = False
    return trajectories

batch = collect_trajectories(policy, env, batch_size=1000)
print(f"Collected {len(batch)} trajectories, total steps: {sum(len(t['rew']) for t in batch)}")

def compute_gae(batch, policy, value_fn):
    all_states = []
    all_actions = []
    all_advs = []
    all_rets = []
    all_logps = []

    for episode in batch:
        states = torch.tensor(episode['obs'], dtype=torch.float32)
        actions = torch.tensor(episode['act'], dtype=torch.int32)
        rewards = episode['rew']
        logps = episode['logp']

        vals = value_fn(states).detach().numpy()
        vals_next = np.append(vals, 0.0)

        rewards = np.array(rewards, dtype=np.float32)
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        advs = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            returns[t] = rewards[t] + (gamma * returns[t+1] if t+1 < T else 0.0)
            delta = rewards[t] + gamma * vals_next[t+1] - vals_next[t]
            advs[t] = last_gae = delta + gamma * lam * last_gae

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        all_states.append(states)
        all_actions.append(actions)
        all_advs.append(torch.tensor(advs), dtype=torch.float32)
        all_rets.append(torch.tensor(returns), dtype=torch.float32)
        all_logps.append(torch.tensor(logps), dtype=torch.float32)

    all_states = torch.cat(all_states)
    all_actions = torch.cat(all_actions)
    all_advs = torch.cat(all_advs)
    all_rets = torch.cat(all_rets)
    all_logps = torch.cat(all_logps)
    return all_states, all_actions, all_advs, all_rets, all_logps

states_b, actions_b, advs_b, returns_b, logps_b = compute_gae(batch, policy, vf)

for _ in range(vf_iters):
    optimizer_vf.zero_grad()
    loss = nn.functional.mse_loss(vf(states_b), returns_b)
    loss.backward()
    optimizer_vf.step()

states_b = states_b.detach()
actions_b = actions_b.detach()
advs_b = advs_b.detach()
returns_b = returns_b.detach()
logps_b = logps_b.detach()

logits = policy(states_b)
dist = distributions.Categorical(logits=logits)
logps = dist.log_prob(actions_b)

ratio = torch.exp(logps - logps_b)
surrogate_loss = -torch.mean(ratio * advs_b)
policy_grad = torch.autograd.grad(surrogate_loss, policy.parameters())

g = torch.cat([grad.view(-1) for grad in policy_grad]).detach()

def hessian_vector_product(vec):
    logp_old = logps_b.detach()
    logp_new = dist.log_prob(actions_b)
    kl = torch.mean(logp_old - logp_new)

    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    prod = (flat_grad_kl * vec).sum()

    grads2 = torch.autograd.grad(prod, policy.parameters())
    flat_grad_kl2 = torch.cat([grad.contiguous().view(-1) + cg_damping * vec for grad in grads2])

    return flat_grad_kl2+cg_damping*vec

def conjugate_gradient(b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone() # residual = b-A*x
    p = b.clone() # search direction
    rd_old = torch.dot(r, r)
    for i in range(iters):
        Avp = hessian_vector_product(p) # A*p ( where A = Hessian)
        alpha = rd_old / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        rs_new = torch.dot(r, r)
        if rs_new < tol:
            break
        p = r+ (rs_new / rd_old) * p
        rd_old = rs_new
    return x

step_direction = conjugate_gradient(g, iters=cg_iters)

# Compute expected improvement
shs = 0.5 * torch.dot(step_direction, hessian_vector_product(step_direction))
if shs.item() < 1e-8:
    lm = 1e-8
else:
    lm = torch.sqrt(2*delta / shs)

full_step = lm * step_direction

old_params = [params.data.clone() for params in policy.parameters()]

def set_policy_params(flat_params):
    idx = 0
    for param in policy.parameters():
        param_length = param.numel()
        param.data = flat_params[idx: idx+param_length].view(param.shape)
        idx += param_length

expected_improve = -surrogate_loss.item()
step_coeff = 1.0

for i in range(backtrack_iters):
    new_params = torch.cat([p.view(-1) for p in policy.parameters()]) + step_coeff * full_step
    set_policy_params(new_params)

    logits_new = policy(states_b)
    dist_new = distributions.Categorical(logits=logits_new)
    logps_new = dist_new.log_prob(actions_b)

    kl_new = torch.mean(logps_b - logps_new).item()

    new_loss = -torch.mean(torch.exp(logps_new - logps_b) * advs_b).item()
    if kl_new <= delta and new_loss < surrogate_loss.item():
        print("Line search succeeded at step size ", step_coeff, "kl: ", kl_new, "surrogate loss: ", new_loss)
        break
    set_policy_params(torch.cat([p.view(-1) for p in old_params]))
    step_coeff *= backtrack_coeff
else:
    print("Line search failed!")
    set_policy_params(torch.cat([p.view(-1) for p in old_params]))

#Training loop
max_epochs = 50
steps_per_batch = 4000

for epoch in range(max_epochs):
    batch = collect_trajectories(policy, env, batch_size=steps_per_batch)
    states_b, actions_b, advs_b, returns_b, logps_b = compute_gae(batch, policy, vf)
    for _ in range(vf_iters):
        optimizer_vf.zero_grad()
        loss = nn.functional.mse_loss(vf(states_b), returns_b)
        loss.backward()
        optimizer_vf.step()

    states_b = states_b.detach()
    actions_b = actions_b.detach()
    advs_b = advs_b.detach()
    returns_b = returns_b.detach()
    logps_b = logps_b.detach()

    logits = policy(states_b)
    dist = distributions.Categorical(logits=logits)
    logps = dist.log_prob(actions_b)

    ratio = torch.exp(logps - logps_b)
    surrogate_loss = -torch.mean(ratio * advs_b)
    policy_grad = torch.autograd.grad(surrogate_loss, policy.parameters())

    g = torch.cat([grad.view(-1) for grad in policy_grad]).detach()

    step_direction = conjugate_gradient(g, iters=cg_iters)

    shs = 0.5 * torch.dot(step_direction, hessian_vector_product(step_direction))

    if shs.item() < 1e-8:
        lm = 1e-8
    else:
        lm = torch.sqrt(2*delta / shs)

    full_step = lm * step_direction

    old_params = [params.data.clone() for params in policy.parameters()]
    expected_improve = -surrogate_loss.item()
    step_coeff = 1.0

    for i in range(backtrack_iters):

        new_params = torch.cat([p.view(-1) for p in policy.parameters()]) + step_coeff * full_step
        set_policy_params(new_params)

        logits_new = policy(states_b)
        dist_new = distributions.Categorical(logits=logits_new)
        logps_new = dist_new.log_prob(actions_b)

        kl_new = torch.mean(logps_b - logps_new).item()

        new_loss = -torch.mean(torch.exp(logps_new - logps_b) * advs_b).item()
        if kl_new <= delta and new_loss < surrogate_loss.item():
            print("Line search succeeded at step size ", step_coeff, "kl: ", kl_new, "surrogate loss: ", new_loss)
            break
        set_policy_params(torch.cat([p.view(-1) for p in old_params]))
        step_coeff *= backtrack_coeff
    else:
        print("Line search failed!")
        set_policy_params(torch.cat([p.view(-1) for p in old_params]))
    print(f"Epoch {epoch} completed,, tot_reward = {sum(sum(t['rew']) for t in batch)}")
env.close()



    