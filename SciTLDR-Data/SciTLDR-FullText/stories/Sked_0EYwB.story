Model-based reinforcement learning (MBRL) has been shown to be a powerful framework for data-efficiently learning control of continuous tasks.

Recent work in MBRL has mostly focused on using more advanced function approximators and planning schemes, leaving the general framework virtually unchanged since its conception.

In this paper, we identify a fundamental issue of the standard MBRL framework -- what we call the objective mismatch issue.

Objective mismatch arises when one objective is optimized in the hope that a second, often uncorrelated, metric will also be optimized.

In the context of MBRL, we characterize the objective mismatch between training the forward dynamics model w.r.t.

the likelihood of the one-step ahead prediction, and the overall goal of improving performance on a downstream control task.

For example, this issue can emerge with the realization that dynamics models effective for a specific task do not necessarily need to be globally accurate, and vice versa globally accurate models might not be sufficiently accurate locally to obtain good control performance on a specific task.

In our experiments, we study this objective mismatch issue and demonstrate that the likelihood of the one-step ahead prediction is not always correlated with downstream control performance.

This observation highlights a critical flaw in the current MBRL framework which will require further research to be fully understood and addressed.

We propose an initial method to mitigate the mismatch issue by re-weighting dynamics model training.

Building on it, we conclude with a discussion about other potential directions of future research for addressing this issue.

@highlight

We define, explore, and begin to address the objective mismatch issue in model-based reinforcement learning.