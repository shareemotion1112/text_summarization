Distributed optimization is essential for training large models on large datasets.

Multiple approaches have been proposed to reduce the communication overhead in distributed training, such as synchronizing only after performing multiple local SGD steps, and decentralized methods (e.g., using gossip algorithms) to decouple communications among workers.

Although these methods run faster than AllReduce-based methods, which use blocking communication before every update, the resulting models may be less accurate after the same number of updates.

Inspired by the BMUF method of Chen & Huo (2016), we propose a slow momentum (SloMo) framework, where workers periodically synchronize and perform a momentum update, after multiple iterations of a base optimization algorithm.

Experiments on image classification and machine translation tasks demonstrate that SloMo consistently yields improvements in optimization and generalization performance relative to the base optimizer, even when the additional overhead is amortized over many updates so that the SloMo runtime is on par with that of the base optimizer.

We provide theoretical convergence guarantees showing that SloMo converges to a stationary point of smooth non-convex losses.

Since BMUF is a particular instance of the SloMo framework, our results also correspond to the first theoretical convergence guarantees for BMUF.

<|TLDR|>

@highlight

SlowMo improves the optimization and generalization performance of communication-efficient decentralized algorithms without sacrificing speed.