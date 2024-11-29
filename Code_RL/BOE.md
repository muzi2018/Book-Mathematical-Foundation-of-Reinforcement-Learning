# State value and action value

$v_\pi(s) \doteq \mathbb{E}\left[G_t \mid S_t=s\right]$.

$q_\pi(s, a) \doteq \mathbb{E}\left[G_t \mid S_t=s, A_t=a\right]$.

the action value represents the expected return that can be obtained by taking action a in state s and **then** following a certain policy thereafter.

if the $q_\pi(s, a)$ calculated from the above has a better action than the previous of state $s$, we will use the action to update policy.

# Optimal state values and optimal policies

Definition 3.1 (Optimal policy and optimal state value). A policy $\pi^*$ is optimal if $v_{\pi^*}(s) \geq v_\pi(s)$ for all $s \in \mathcal{S}$ and for any other policy $\pi$. The state values of $\pi^*$ are the optimal state values.





$\diamond$ Existence: Does the optimal policy exist?
$\diamond$ Uniqueness: Is the optimal policy unique?
$\diamond$ Stochasticity: Is the optimal policy stochastic or deterministic?
$\diamond$ Algorithm: How to obtain the optimal policy and the optimal state values?



# Bellman optimality equation

By solving this equation, we can obtain optimal policies and optimal
state values  

$\diamond$ Existence: Does this equation have a solution?
$\diamond$ Uniqueness: Is the solution unique?
$\diamond$ Algorithm: How to solve this equation?
$\diamond$ Optimality: How is the solution related to optimal policies?

**Bellman equation**
$$
\begin{aligned}
v_\pi(s) & =\mathbb{E}\left[R_{t+1} \mid S_t=s\right]+\gamma \mathbb{E}\left[G_{t+1} \mid S_t=s\right], \\
& =\underbrace{\sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s, a) r}_{\text {mean of immediate rewards }}+\underbrace{\gamma \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)}_{\text {mean of future rewards }} \\
& =\sum_{a \in \mathcal{A}} \pi(a \mid s)\left[\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)\right], \quad \text { for all } s \in \mathcal{S} .
\end{aligned}
$$
**Bellman optimality equation**
$$
\begin{aligned}
&\text { For every } s \in \mathcal{S} \text {, the elementwise expression of the BOE is }\\
&\begin{aligned}
v(s) & =\max _{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a \mid s)\left(\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v\left(s^{\prime}\right)\right) \\
& =\max _{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a \mid s) q(s, a),
\end{aligned}
\end{aligned}
$$
**Solve Bellman optimality equation** 

1.Maximization of the right-hand side of the BOE  **(solving $v$ and $\pi$ one by one)**
$$
v(s)=\max _{\pi(s) \in \Pi(s)} \sum_{a \in \mathcal{A}} \pi(a \mid s) q(s, a), \quad s \in \mathcal{S} .
$$
if we want to maximize a weighted sum $\sum_{i=1}^3 c_i q_i$ with the constraint $c_1+c_2+c_3=1$, the best strategy is to allocate all the weight to the term with the largest value. In other words:
$$
\text { Maximize } \sum_{a \in \mathcal{A}} \pi(a \mid s) q(s, a) \text { subject to } \sum_{a \in \mathcal{A}} \pi(a \mid s)=1
$$


The optimal solution is to set:

$$
\pi(a \mid s)= \begin{cases}1, & \text { if } a=a^* \text { where } a^*=\arg \max _{a \in \mathcal{A}} q(s, a) \\ 0, & \text { otherwise }\end{cases}
$$


This means the optimal policy will select the action $a^*$ that maximizes $q(s, a)$ with probability 1 and assign probability 0 to all other actions. This allocation achieves the highest possible value for $v(s)$ because it chooses the best action for each state.

Here, $a^*=\arg \max _a q(s, a)$. **In summary, the optimal policy $\pi(s)$ is the one that selects the action that has the greatest value of $q(s, a)$.**

2.Matrix-vector form of the BOE  

The matrix-vector form of the BOE is

$$
v=\max _{\pi \in \Pi}\left(r_\pi+\gamma P_\pi v\right)
$$
The matrix-vector form of the Bellman optimal equation is like

$$
\underbrace{\left[\begin{array}{l}
v_\pi\left(s_1\right) \\
v_\pi\left(s_2\right) \\
v_\pi\left(s_3\right) \\
v_\pi\left(s_4\right)
\end{array}\right]}_{v_\pi}=\underbrace{\left[\begin{array}{l}
r_\pi\left(s_1\right) \\
r_\pi\left(s_2\right) \\
r_\pi\left(s_3\right) \\
r_\pi\left(s_4\right)
\end{array}\right]}_{r_\pi}+\gamma \underbrace{\left[\begin{array}{llll}
p_\pi\left(s_1 \mid s_1\right) & p_\pi\left(s_2 \mid s_1\right) & p_\pi\left(s_3 \mid s_1\right) & p_\pi\left(s_4 \mid s_1\right) \\
p_\pi\left(s_1 \mid s_2\right) & p_\pi\left(s_2 \mid s_2\right) & p_\pi\left(s_3 \mid s_2\right) & p_\pi\left(s_4 \mid s_2\right) \\
p_\pi\left(s_1 \mid s_3\right) & p_\pi\left(s_2 \mid s_3\right) & p_\pi\left(s_3 \mid s_3\right) & p_\pi\left(s_4 \mid s_3\right) \\
p_\pi\left(s_1 \mid s_4\right) & p_\pi\left(s_2 \mid s_4\right) & p_\pi\left(s_3 \mid s_4\right) & p_\pi\left(s_4 \mid s_4\right)
\end{array}\right]}_{P_\pi} \underbrace{\left[\begin{array}{l}
v_\pi\left(s_1\right) \\
v_\pi\left(s_2\right) \\
v_\pi\left(s_3\right) \\
v_\pi\left(s_4\right)
\end{array}\right]}_{v_\pi} .
$$
$\left[r_\pi\right]_s \doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s, a) r, \quad\left[P_\pi\right]_{s, s^{\prime}}=p\left(s^{\prime} \mid s\right) \doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) p\left(s^{\prime} \mid s, a\right)$

Since the optimal value of $\pi$ is determined by $v$, so we could create a function of $v$
$f(v) \doteq \max _{\pi \in \Pi}\left(r_\pi+\gamma P_\pi v\right)$

or $v=f(v)$

3.Contraction mapping theorem  (fixed-point theorem)

Consider a function $f(x)$, where $x \in \mathbb{R}^d$ and $f: \mathbb{R}^d \rightarrow \mathbb{R}^d$. A point $x^*$ is called a fixed point if

$$
f\left(x^*\right)=x^* .
$$


The interpretation of the above equation is that the map of $x^*$ is itself. This is the reason why $x^*$ is called "fixed". The function $f$ is a contraction mapping (or contractive function) if there exists $\gamma \in(0,1)$ such that

$$
\left\|f\left(x_1\right)-f\left(x_2\right)\right\| \leq \gamma\left\|x_1-x_2\right\|
$$

for any $x_1, x_2 \in \mathbb{R}^d$. In this book, $\|\cdot\|$ denotes a vector or matrix norm.





Theorem 3.1 (Contraction mapping theorem). For any equation that has the form $x=$ $f(x)$ where $x$ and $f(x)$ are real vectors, if $f$ is a contraction mapping, then the following properties hold.
$\diamond$ Existence: There exists a fixed point $x^*$ satisfying $f\left(x^*\right)=x^*$.
$\diamond$ Uniqueness: The fixed point $x^*$ is unique.
$\diamond$ Algorithm: Consider the iterative process:

$$
x_{k+1}=f\left(x_k\right),
$$

where $k=0,1,2, \ldots$. Then, $x_k \rightarrow x^*$ as $k \rightarrow \infty$ for any initial guess $x_0$. Moreover, the convergence rate is exponentially fast.

4.Contraction property of the right-hand side of the BOE  

We next show that $f(v)$ in the BOE is a contraction mapping. Thus, the contraction mapping theorem introduced in the previous subsection can be applied.

Theorem 3.2 (Contraction property of $f(v)$ ). The function $f(v)$ on the right-hand side of the BOE in (3.3) is a contraction mapping. In particular, for any $v_1, v_2 \in \mathbb{R}^{|\mathcal{S}|}$, it holds that

$$
\left\|f\left(v_1\right)-f\left(v_2\right)\right\|_{\infty} \leq \gamma\left\|v_1-v_2\right\|_{\infty},
$$

where $\gamma \in(0,1)$ is the discount rate, and $\|\cdot\|_{\infty}$ is the maximum norm, which is the maximum absolute value of the elements of a vector.

5.Solving an optimal policy from the BOE  
