An MDP is a general framework for describing stochastic dynamical systems. The key ingredients of an MDP are listed below.

Sets:

- State space: the set of all states, denoted as $\mathcal{S}$.
- Action space: a set of actions, denoted as $\mathcal{A}(s)$, associated with each state $s \in \mathcal{S}$.
- Reward set: a set of rewards, denoted as $\mathcal{R}(s, a)$, associated with each state-action pair $(s, a)$.

Model:

- State transition probability: In state $s$, when taking action $a$, the probability of transitioning to state $s^{\prime}$ is $p\left(s^{\prime} \mid s, a\right)$. It holds that $\sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right)=1$ for any $(s, a)$.
- Reward probability: In state $s$, when taking action $a$, the probability of obtaining reward $r$ is $p(r \mid s, a)$. It holds that $\sum_{r \in \mathcal{R}(s, a)} p(r \mid s, a)=1$ for any $(s, a)$.

Policy: In state $s$, the probability of choosing action $a$ is $\pi(a \mid s)$. It holds that $\sum_{a \in \mathcal{A}(s)} \pi(a \mid s)=1$ for any $s \in \mathcal{S}$.
Markov property: The Markov property refers to the memoryless property of a stochastic process. Mathematically, it means that
$$
\begin{aligned}
& p\left(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0\right)=p\left(s_{t+1} \mid s_t, a_t\right), \\
& p\left(r_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0\right)=p\left(r_{t+1} \mid s_t, a_t\right)
\end{aligned}
$$

where $t$ represents the current time step and $t+1$ represents the next time step. Equation (1.4) indicates that the next state or reward depends merely on the current state and action and is independent of the previous ones. The Markov property is important for deriving the fundamental Bellman equation of MDPs, as shown in the next chapter.

Here, $p\left(s^{\prime} \mid s, a\right)$ and $p(r \mid s, a)$ for all $(s, a)$ are called the model or dynamics. The model can be either stationary or nonstationary (or in other words, time-invariant or time-variant). A stationary model does not change over time; a nonstationary model may vary over time. For instance, in the grid world example, if a forbidden area may pop up or disappear sometimes, the model is nonstationary. In this book, we only consider stationary models.





SAP-RM (State, Action, Policy, Reward, Model) to recall the key elements of an MDP.
- S : State space $(\mathcal{S})$.
- A: Action space $(\mathcal{A})$.
- P: Policy $(\pi(a \mid s)$ ).
- R: Reward $(\mathcal{R}(s, a), p(r \mid s, a)$ ).
- M: Model (state transition $p\left(s^{\prime} \mid s, a\right)$, reward probabilities).