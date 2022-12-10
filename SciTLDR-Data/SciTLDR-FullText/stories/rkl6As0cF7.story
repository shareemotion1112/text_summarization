Humans are capable of attributing latent mental contents such as beliefs, or intentions to others.

The social skill is critical in everyday life to reason about the potential consequences of their behaviors so as to plan ahead.

It is known that humans use this reasoning ability recursively, i.e. considering what others believe about their own beliefs.

In this paper, we start  from level-$1$ recursion and introduce a probabilistic recursive reasoning (PR2) framework for multi-agent reinforcement learning.

Our hypothesis is that it is beneficial for each agent to account for how the opponents would react to its future behaviors.

Under the PR2 framework, we adopt variational Bayes methods to approximate the opponents' conditional policy, to which each agent finds the  best response and then improve their own policy.

We develop  decentralized-training-decentralized-execution  algorithms, PR2-Q and PR2-Actor-Critic, that are proved to converge in the self-play scenario when there is one Nash equilibrium.

Our methods are tested on both the matrix game and the differential game, which have a non-trivial equilibrium where common gradient-based methods fail to converge.

Our experiments show that it is critical to reason about how the opponents believe about what the agent believes.

We expect our work to contribute a new idea of modeling the opponents to the multi-agent reinforcement learning community.

@highlight

We proposed a novel probabilisitic recursive reasoning (PR2) framework for multi-agent deep reinforcement learning tasks.

@highlight

Proposes a new approach for fully decentralized training in multi-agent reinforcement learning

@highlight

Tackles the problem of endowing RL agents with recursive reasoning capabilities in a multi-agent setting based on the hypothesis that recursive reasoning is beneficial for them to converge to non-trival equilibria

@highlight

The paper introduces a decentralized training method for multi-agent reinforcement learning, where the agents infer the policies of other agents and use the inferred models for decision making. 