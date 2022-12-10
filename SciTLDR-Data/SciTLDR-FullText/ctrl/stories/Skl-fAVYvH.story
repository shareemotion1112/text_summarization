We develop a new algorithm for imitation learning from a single expert demonstration.

In contrast to many previous one-shot imitation learning approaches, our algorithm does not assume access to more than one expert demonstration during the training phase.

Instead, we leverage an exploration policy to acquire unsupervised trajectories, which are then used to train both an encoder and a context-aware imitation policy.

The optimization procedures for the encoder, imitation learner, and exploration policy are all tightly linked.

This linking creates a feedback loop wherein the exploration policy collects new demonstrations that challenge the imitation learner, while the encoder attempts to help the imitation policy to the best of its abilities.

We evaluate our algorithm on 6 MujoCo robotics tasks.

<|TLDR|>

@highlight

Unsupervised self-imitation algorithm capable of inference from a single expert demonstration.