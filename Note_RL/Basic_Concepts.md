**Concept**

1. State Space ( $\mathcal{S}$ )

- What it is: The collection of all possible situations the agent can encounter.
- Analogy: Imagine a chessboard. Each configuration of pieces is a unique state.
- Tip to remember: Think of states as "where you are" in the world of the problem.

2. Action Space $(\mathcal{A})$

- What it is: The set of all actions the agent can take.
- Analogy: On a chessboard, moving a piece (e.g., rook to A5) is an action.
- Tip to remember: Actions are "what you can do" at any given state.

3. State Transitions ( $s \xrightarrow{a} s^{\prime}$ )

- What it is: The process of moving from one state to another by taking an action.
- Deterministic: The next state is exactly predictable.
- Stochastic: The next state depends on probabilities.
- Analogy: Pressing a button in a vending machine (deterministic) vs. spinning a roulette wheel (stochastic).
- Tip to remember: Think of transitions as "what happens after you act."

4. Policies ( $\pi(a \mid s)$ )

- What it is: A strategy that tells the agent which action to take in a given state.
- Deterministic: Always pick the same action for a state.
- Stochastic: Use probabilities to decide.
- Analogy: A GPS system providing fixed routes (deterministic) vs. recommending routes based on traffic (stochastic).
- Tip to remember: Policies are "the agent's playbook."

5. Reward $(r(s, a)$ )

- What it is: A numerical signal given to the agent after taking an action in a state, guiding it towards a goal.
- Analogy: Positive feedback for correct answers and penalties for mistakes in a quiz.
- Tip to remember: Rewards are the "score" the agent earns.

6. Trajectory

- What it is: A sequence of states, actions, and rewards over time.
- Example: $s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=1]{a_3} s_3$.
- Analogy: A travel diary recording each location, action, and experience.
- Tip to remember: Trajectories are the "path" the agent takes.

7. Return

- What it is: The cumulative rewards collected over a trajectory.
- If discounted: $G=r_1+\gamma r_2+\gamma^2 r_3+\ldots$.
- Analogy: The total points accumulated in a game.
- Tip to remember: Returns are the "final score."

8. Discount Rate ( $\gamma$ )

- What it is: A factor that determines how much future rewards are valued compared to immediate rewards.
- $\gamma \approx 1$ : Values long-term rewards more.
- $\gamma \approx 0$ : Focuses on immediate rewards.
- Analogy: Choosing between a small prize today or a bigger prize next week.
- Tip to remember: Discount rate is the "patience factor."

9. Episode and Tasks

- What it is: A finite sequence of interactions, stopping at a terminal state (episodic task) or continuing indefinitely (continuing task).
- Terminal states: Define when the task ends.
- Absorbing states: Terminal states where the agent stays forever.
- Analogy: Episodic tasks are like solving a maze, while continuing tasks are like keeping a robot running indefinitely.
- Tip to remember: Think of an episode as a "game round."

State space, denoted as $\mathcal{S}=\left\{s_1, \ldots, s_9\right\}$

Action space, denoted as $\mathcal{A}=\left\{a_1, \ldots, a_5\right\}$

State transition $s_1 \xrightarrow{a_2} s_2$, $p\left(s_1 \mid s_1, a_2\right)$, deterministic state transitions, stochastic state transitions  

Policies can be described by conditional probabilities, $\pi(a \mid s)$, deterministic, stochastic Policies

Reward $r(s, a)$, A reward can be interpreted as a human-machine interface, with which we can guide
the agent to behave as we expect.  (immediate reward)

Trajectory is a state-action-reward chain, $s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9$. maybe without stop criterion and allows for infinitely long trajectories.

Return is defined as the sum of all the rewards collected along the trajectory. (cumulative rewards). Returns can be used to evaluate policies  

Discount rate  $\gamma \in(0,1)$

Episode is the finite trajectory following a policy to **complete an episodic task** may **stop** at some terminal states. sometimes if episodic task may not stop when it reaches terminal states, meaning **continuing task**, in this case, we should  well define the process after the agent reaches the terminal state. For example1, we could design terminal state as **absorbing states** , which always stays in this state forever. 



**Scenario**
Imagine a robot navigating a $3 \times 3$ grid (a simple environment). The robot's goal is to move from the start state to a goal state while maximizing its rewards.

Step 1: Define the Concepts in the Scenario

1. State Space $(\mathcal{S})$ :

- Each cell in the grid represents a unique state.
- For a $3 \times 3$ grid, $\mathcal{S}=\left\{s_1, s_2, \ldots, s_9\right\}$.
- Example: $s_1$ is the top-left corner, $s_9$ is the bottom-right corner.

2. Action Space $(\mathcal{A})$ :

- The robot can move: $\mathcal{A}=\{$ up, down, left, right $\}$.

3. State Transition ( $s \xrightarrow{a} s^{\prime}$ ):

- If the robot takes an action, it moves to the next state:
- $s_1 \xrightarrow{\text { right }} s_2$,
- $s_2 \xrightarrow{\text { down }} s_5$.
- Deterministic: The robot always moves as expected.
- Stochastic: There's a chance the robot may move incorrectly (e.g., slip).

4. Reward $(r(s, a))$ :

- Moving to the goal state $\left(s_9\right)$ gives a reward of +10 .
- Entering a danger zone $\left(s_5\right)$ gives a penalty of -5 .
- Moving elsewhere gives 0 reward.

5. Policy $(\pi(a \mid s)$ ):
- The robot needs a strategy to decide actions.
- Deterministic: Always chooses the optimal action at every state.
- Stochastic: Picks actions with probabilities, e.g., $70 \%$ chance of moving toward the goal and 30\% random.
6. Trajectory:
- Example trajectory: $s_1 \xrightarrow[r=0]{\text { right }} s_2 \xrightarrow[r=0]{\text { down }} s_5 \xrightarrow[r=-5]{\text { down }} s_8 \xrightarrow[r=+10]{\text { right }} s_9$.
7. Return:
- The sum of rewards along a trajectory.
- For the above trajectory: $0+0-5+10=5$.
8. Discount Rate ( $\gamma$ ):
- $\gamma=0.9$ : Future rewards are worth $90 \%$ of their original value.
- Discounted return: $G=-5+0.9 \cdot 10=4$.

9. Episode:
- Task ends when the robot reaches the goal $\left(s_9\right)$.









