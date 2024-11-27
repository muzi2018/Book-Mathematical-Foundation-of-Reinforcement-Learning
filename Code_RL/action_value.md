# Derivation

From state value to action value

The action value of a state-action pair $(s, a)$ is defined as

$$
q_\pi(s, a) \doteq \mathbb{E}\left[G_t \mid S_t=s, A_t=a\right]
$$

the relationship between action values and state values

1.state value is the expectation of the action values associated with that state, it show how to get state values from action values

$\underbrace{\mathbb{E}\left[G_t \mid S_t=s\right]}_{v_\pi(s)}=\sum_{a \in \mathcal{A}} \underbrace{\mathbb{E}\left[G_t \mid S_t=s, A_t=a\right]}_{q_\pi(s, a)} \pi(a \mid s)$.

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$

2.it show how to get action values from state values, action values consists of two terms. The first term is the mean

of the immediate rewards, and the second term is the mean of the future rewards.

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left[\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)\right]$

$q_\pi(s, a)=\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)$
