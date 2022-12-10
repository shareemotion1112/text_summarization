A central challenge in reinforcement learning is discovering effective policies for tasks where rewards are sparsely distributed.

We postulate that in the absence of useful reward signals, an effective exploration strategy should seek out {\it decision states}.

These states lie at critical junctions in the state space from where the agent can transition to new, potentially unexplored regions.

We propose to learn about decision states from prior experience.

By training a goal-conditioned model with an information bottleneck, we can identify decision states by examining where the model accesses the goal state through the bottleneck.

We find that this simple mechanism effectively identifies decision states, even in partially observed settings.

In effect, the model learns the sensory cues that correlate with potential subgoals.

In new environments, this model can then identify novel subgoals for further exploration, guiding the agent through a sequence of potential decision  states and through new regions of the state space.

@highlight

Training agents with goal-policy information bottlenecks promotes transfer and yields a powerful exploration bonus

@highlight

Proposes regularizing standard RL losses with the negative conditional mutual information for policy search in a multi-goal RL setting.

@highlight

This paper proposes the concept of decision state and proposes a KL divergence regularization to learn the structure of the tasks to use this information to encourage the policy to visit the decision states.

@highlight

The paper proposes a method of regularising goal-conditioned policies with a mutual information term. 