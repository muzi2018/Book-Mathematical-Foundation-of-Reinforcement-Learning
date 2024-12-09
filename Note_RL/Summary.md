# Bellman equation

$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\ldots$

$v_{\pi}(s)=\mathbb{E}\left[R_{t+1} \mid S_t=s\right]+\gamma \mathbb{E}\left[G_{t+1} \mid S_t=s\right]$

$\mathbb{E}\left[R_{t+1} \mid S_t=s\right]=\sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s, a) r$

$\mathbb{E}\left[G_{t+1} \mid S_t=s\right]=\sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)$



**From state value to action value**  

$q_\pi(s, a) \doteq \mathbb{E}\left[G_t \mid S_t=s, A_t=a\right]$.

$q_\pi(s, a)=\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in \mathcal{A}\left(s^{\prime}\right)} \pi\left(a^{\prime} \mid s^{\prime}\right) q_\pi\left(s^{\prime}, a^{\prime}\right)$,



# Bellman optimal equation 

Bellman optimal equation 

$v^*=\max _{\pi \in \Pi}\left(r_\pi+\gamma P_\pi v^*\right)$



# Model-based Solver

**Value Iteration and Policy Iteration**

# Model-free Solver

**Monte Carlo**

**TD learning**: SARSA, n-step SARSA, Q-learning

# Off-policy

behavior policy (generate experience samples)= target policy (converge to an optimal policy)

Q-learning

# On-policy



# Tabular-base



# Function-base

Value Function: state value, action value, Deep Q-learning

Policy Gradient Method: From metric - Average state value, Average reward

# Actor-Critic( policy gradient algorithm)

“actor” refers to a policy update step

“critic” refers to a value update step
