Imitation learning from human-expert demonstrations has been shown to be greatly helpful for challenging reinforcement learning problems with sparse environment rewards.

However, it is very difficult to achieve similar success without relying on expert demonstrations.

Recent works on self-imitation learning showed that imitating the agent's own past good experience could indirectly drive exploration in some environments, but these methods often lead to sub-optimal and myopic behavior.

To address this issue, we argue that exploration in diverse directions by imitating diverse trajectories, instead of focusing on limited good trajectories, is more desirable for the hard-exploration tasks.

We propose a new method of learning a trajectory-conditioned policy to imitate diverse trajectories from the agent's own past experiences and show that such self-imitation helps avoid myopic behavior and increases the chance of finding a globally optimal solution for hard-exploration tasks, especially when there are misleading rewards.

Our method significantly outperforms existing self-imitation learning and count-based exploration methods on various hard-exploration tasks with local optima.

In particular, we report a state-of-the-art score of more than 20,000 points on Montezumas Revenge without using expert demonstrations or resetting to arbitrary states.

@highlight

Self-imitation learning of diverse trajectories with trajectory-conditioned policy

@highlight

This paper addresses hard exploration tasks by applying self-imitation to a diverse selection of trajectories from past experience, to drive more efficient exploration in sparse-reward problems, achieving SOTA results.