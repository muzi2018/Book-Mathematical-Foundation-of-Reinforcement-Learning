Robot Model
$$
\mathbf{S} \boldsymbol{\tau}=\boldsymbol{M}(\boldsymbol{q}) \dot{\boldsymbol{v}}+\boldsymbol{h}(\boldsymbol{q}, \boldsymbol{v})-\mathbf{J}_c^T(\boldsymbol{q}) \boldsymbol{f}_c
$$

$$
\mathbf{x}=\left[\begin{array}{l}
\boldsymbol{q} \\
\boldsymbol{v}
\end{array}\right]
$$

$$
\mathbf{u}=\left[\begin{array}{c}
\dot{\boldsymbol{v}} \\
\boldsymbol{f}_c 
\end{array}\right]
$$

**1.1 A grid world example**  

1. **What is the role of the agent in the grid world example?**
   - Describe what the agent does and how it interacts with the environment.
     - ​	move
     - A reward can be interpreted as a human-machine interface, with which we can guide
       the agent to behave as we expect.  
2. **How are states defined in the grid world?**
   - Explain what constitutes a state and how many states are present in the example.
     - cell index constitutes a state;
     - nine
3. **What actions can the agent take in the grid world?**
   - List the possible actions available to the agent and discuss any restrictions based on the state.
     - five possible actions: moving upward, moving, rightward, moving downward, moving leftward, and staying still.  
     - Different states can have different action spaces.
4. **What is the significance of the target cell in the grid world?**
   -  Why is the target cell important for the agent, and what does it represent in the context of reinforcement learning?
5. **How do forbidden cells affect the agent's movement?**
   - Discuss the implications of forbidden cells on the agent's actions and decision-making process.
     - the forbidden cells are accessible, although stepping into them may get punished.  
6. **What challenges does the agent face when trying to reach the target cell?**
   - Identify potential obstacles or constraints that the agent must navigate.
7. **How does the concept of rewards apply in the grid world example?**
   - Explain how rewards are structured in this scenario and their impact on the agent's learning.
     - exit the boundary -1; enter a forbidden cell -1; reach target cell +1; Otherwise 0
8. **What does it mean for the agent to learn a "good" policy in this context?**
   - Define what constitutes a good policy and how the agent can determine it through interaction with the environment.
     - The idea is that the agent should reach the target without entering any forbidden cells, taking unnecessary detours, or colliding with the boundary of the grid.  
9. **How does the grid world example illustrate the principles of reinforcement learning?**
   - Reflect on how this example encapsulates the key ideas of states, actions, rewards, and learning.
10. **What are the limitations of using a grid world example for reinforcement learning?**
    - Consider what aspects of real-world scenarios might not be captured by this simplified model.

# Basic Concepts

state and action; 

deterministic state transitions, stochastic state transition; 

deterministic policy, stochastic policy, A tabular representation of a policy ;  

The reward is a function of the state s and action a.

Hence, it is also denoted as r(s; a). 

 A trajectory is a state-action-reward chain.  

Returns are also called total rewards or cumulative rewards.  A return consists of an immediate reward and future rewards.   

When interacting with the environment by following a policy, the agent may stop at some terminal states. The resulting trajectory is called an episode (or a trial).  

MDP
