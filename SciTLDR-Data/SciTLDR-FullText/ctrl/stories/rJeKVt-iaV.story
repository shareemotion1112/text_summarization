Partially observable Markov decision processes (POMDPs) are a natural model for scenarios where one has to deal with incomplete knowledge and random events.

Applications include, but are not limited to, robotics and motion planning.

However, many relevant properties of POMDPs are either undecidable or very expensive to compute in terms of both runtime and memory consumption.

In our work, we develop a game-based abstraction method that is able to deliver safe bounds and tight   approximations for important sub-classes of such properties.

We discuss the theoretical implications and showcase the applicability of our results on a broad spectrum of benchmarks.

In offline motion planning, we aim to find a strategy for an agent that ensures certain desired behavior, even in the presence of dynamical obstacles and uncertainties BID0 .

If random elementslike uncertainty in the outcome of an action or in the movement of dynamic obstacles -need to be taken into account, the natural model for such scenarios are Markov decision processes (MDPs).

MDPs are non-deterministic models which allow the agent to perform actions under full knowledge of the current state of the agent its surrounding environment.

In many applications, though, full knowledge cannot be assumed, and we have to deal with partial observability BID1 .

For such scenarios, MDPs are generalized to partially observable MDPs (POMDPs).

In a POMDP, the agent does not know the exact state of the environment, but only an observation that can be shared between multiple states.

Additional information about the likelihood of being in a certain state can be gained by tracking the observations over time.

This likelihood is called the belief state.

Using an update function mapping a belief state and an action as well as the newly obtained observation to a new belief state, one can construct a (typically infinite) MDP, commonly known as the belief MDP.While model checking and strategy synthesis for MDPs are, in general, well-manageable problems, POMDPs are much harder to handle and, due to the potentially infinite belief space, many problems are actually undecidable BID2 .

Our aim is to apply abstraction and abstraction refinement techniques to POMDPs in order to get good and safe approximative results for different types of properties.

As a case study, we work with a scenario featuring a controllable agent.

Within a certain area, the agent needs to traverse a room while avoiding both static obstacles and randomly moving opponents.

The area is modeled as a grid, the static obstacles as grid cells that may not be entered.

Our assumption for this scenario is that the agent always knows its own position, but the positions of an opponent is only known if its distance from the agent is below a given threshold and if the opponent is not hidden behind a static obstacle.

We assume that the opponents move probabilistically.

This directly leads to a POMDP model for our case study.

For simplification purposes, we only deal with one opponent, although our approach supports an arbitrary number of opponents.

We assume the observation function of our POMDPs to be deterministic, but more general POMDPs can easily be simplified to this case.

The goal is to find a strategy which maximizes the probability to navigate through the grid from an initial to a target location without collision.

For a grid size of n ?? n cells and one opponent, the number of states in the POMDP is in O(n 4 ), i. e., the state space grows rapidly with increasing grid size.

In order to handle non-trivial grids, we propose an approach using game-based abstraction BID3 .Intuitively, we lump together all states that induce the same observation; for each position of the agent, we can distinguish between all states in which the opponent's position is known, but states in which the position is unknown are merged into one far away state BID4 .

In order to get a safe approximation

We show that any strategy computed with our abstraction that guarantees a certain level of safety can be mapped to a strategy for the original POMDP guarantiing at least the same level of safety.

In particular, we establish a simulation relation between paths in the probabilistic game and paths in the POMDP.

Intuitively, each path in the POMDP can be reproduced in the probabilistic game if the second player resolves the nondeterminism in a certain way.

Game-based model checking assumes the non-determinism to be resolved in the worst way possible, so it will provide a lower bound on the level of safety achievable in the actual POMDP.

For full proof see BID4 .

We analyzed the game-based models using the PRISM-games model checker and compared the obtained results with the stateof-the-art POMDP model checker PRISM-pomdp BID5 , showing that we can handle grids that are considerably larger than what PRISM-pomdp can handle, while still getting schedulers that induce values which are close to optimal.

TAB0 shows a few of our experiments for verifying a reach-avoid property on a grid without obstacles.

The result colums show the probability (computed by the respective method) to reach a goal state without a collision.

As one can see, the abstraction approach is faster by orders of magnitude than solving the POMDP directly, and the game model also is much smaller for large grids while still getting very good approximations for the actual probabilities.

The strategies induce even better values when they are mapped back to the original POMDP.

While being provably sound, our approach is still targeting an undecidable problem and as such not complete in the sense that in general no strategy with maximum probability for success can be deduced.

In particular for cases with few paths to the goal location, the gap between the obtained bounds and the actual maximum can become large.

For those cases, we define a scheme to refine the abstraction by encoding one or several steps of history into the current state, which leads to larger games and accordingly longer computation times, but also to better results.

TAB0 showcases an implementation of this one-step history refinement.

We use a benchmark representing a long, narrow tunnel, in which the agent has to pass the opponent once, but, due to the abstraction, can actually run into the it repeatedly if the abstraction-player has the opponent re-appear in front of the agent.

With longer tunnels, the probability to safely arrive in a goal state diminishes.

Adding a refinement which remembers the last known position of the opponent and thus restricting the non-deterministic movement keeps the probability constant for arbitrary length.

We developed a game-based abstraction technique to synthesize strategies for a class of POMDPs.

This class encompasses typical grid-based motion planning problems under restricted observability of the environment.

For these scenarios, we efficiently compute strategies that allow the agent to maneuver the grid in order to reach a given goal state while at the same time avoiding collisions with faster moving obstacles.

Experiments show that our approach can handle state spaces up to three orders of magnitude larger than general-purpose state-of-the-art POMDP solvers in less time, while at the same time using fewer states to represent the same grid sizes.

<|TLDR|>

@highlight

This paper provides a game-based abstraction scheme to compute provably sound policies for POMDPs.