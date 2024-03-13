This #project aims to replicate some behaviour found in nature.
The idea is to have a set of learners that interact with each other and with the environment, and see what group behaviour emerges from that.

An easy start is to optimize for food and minimize predation. The environment is a scalar field (food) $F(\mathbfit{x,t})$, that regenerates over time (up to a cap). The agents can perform an action to eat, decreasing the value of food on their tile. 

Predation can be modelled simply as a probability of being eaten. It increases the further away the agent is from the "center of mass" of the entire group. Thus the group will be rewarded to stay together but also to explore for forage for food.

One way to model predation in a RL framework is at each step to sample from a distribution $p(r)$, with $r$ as the distance to the center of mass. This distribution need not be a probability distribution. The point is that the predation penalty should not be a straight-forward function, but a random process (chance of being eaten) that increses as the distance of the group center of mass increases. 
It should also be dependent on the density of the group $\rho$. The denser the group is (taking only into account the total surface the group is covering), the lower the chance of getting punished. If you're in a dense environment, you're not very likely to get eaten.

The **total reward** should then be a trade-off between the food and predation risk:
$$
R = R_\mathrm{food}-R_\mathrm{predation}
$$
Then with a hyperparameter $\alpha>0$
$$
R_\mathrm{pred}=-\alpha \frac{x}{\rho},\quad x\sim p(r)
$$

For $R_\mathrm{food}$, we can model that a given agent consumes a given percentage of the value of the food field. So, the less there is, the less you get.  Therefore we get
$$
R_\mathrm{food}=\beta\cdot F(\mathbfit{x,t})
$$

The food function $F(\mathbfit{x},t)$ should be sigmoidal with respect to its previous step. It could be basically a grid, and since we're already using a grid for space, it might be reasonable to grid the state space for the agent as well. This simplifies things significantly for a start. We can use a Deep Q Agent where the agents just move in one of 8 neighbouring grid points or stay in place to eat more. After performing an action, the agent eats, and recieves its reward according to the food value of where it currently is and the predation risk it has taken to get there.

The question is then how to model the **agent-agent interaction**. There are some intriguing possibilities. Consider the case where two agents want to move to the same grid point. Some possibilities to resolve this might be:
- It's a random chance who gets the space, simplest would be 50/50, but this could be shifted depending on some factors (like for future: age, size, or other characteristics of the agent that makes it more or less probable to get a space in a grid).
- Allowing sharing: Both agents can be on the same field, but then they get less food. The intake is halved. This makes it less rewarding to go where others might go since you will have to share the food. This is very interesting. In that case, the modified food reward would be $R_\mathrm{food}=\beta\cdot F(\mathbfit{x,t})/N_A$, with $N_A$ the amount of agents sharing the grid point.

So now we can start thinking about the **state space** of the Agent. It should know the direction or even the entire vector to the center of mass. It should also know the density of neighbours it is currently in. Or maybe just the local density, i.e. only taking into account its own grid point and all its nearest neighbouring grid points. (this is getting computationally out of hand xD).
Lastly, it must have some information related to food. This could simply just be the food value of its current grid point.
Curiously, I don't think it needs to know its location in the map, only its relative position to the center of mass (for a big enough environment, that is). Because what happens at the edge of the environment? The simplest again would just make it invalid or to have it loop around.

So the state space, for an environment with $N\times N$ grid points, corresponds to:
$$
\mathcal{S}=\{0,\dots,N-1\}\times\{0,\dots,N-1\}\times [0,1]\times [-N,N]^2
$$
and corresponds to $x$ position, $y$ position (which are discrete), Food value and vector from agent position to center of mass.
# Specific Implementations
The **Food Function** is defined by the recursive relation:
$$
F(\mathbfit{x},t+dt)= F(\mathbfit{x},t) + rF(\mathbfit{x},t)(1-\frac{F(\mathbfit{x},t)}{k})
$$
The domain of the function is $[0,1]$

The **predation reward** is as follows. Let the distance to the geometric center be $d$, and the normalized distance to the geometric center $\hat{d}=d/N$ and $x \sim \mathcal{N}(0.5, 0.1)$. Then  
$$
\text{if: } x<\hat{d}\Rightarrow R_\mathrm{pred}=-\hat{d}\\
$$
$$
\text{else: } R_\mathrm{pred}=0
$$
Which is cool because since the range of the reward is $[-\infty, 1]$, the further away from the geometric center an agent is, the more likely it is to get punish **and** the worse the punish, scaling linearly. This should more or less work because the food reward is capped between $[0,1]$ and this reward is capped at $[-1,0]$.
**Notes:** In the current implementation, I use a **hard boundary condition**, not periodic.



# Possible Issues
There is no end condition. 


# Vorticity
The vorticity calculation could be done by using the centered difference method. because otherwise if they move only in one plane the vorticity will always be zero for forward/backwards. Thus I think the goal is to eventually have trained agents and then have them run for some time and collect all the data and from the data calculate the vorticity.
