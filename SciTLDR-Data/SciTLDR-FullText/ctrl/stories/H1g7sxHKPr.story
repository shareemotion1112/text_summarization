Gradient-based meta-learning algorithms require several steps of gradient descent to adapt to newly incoming tasks.

This process becomes more costly as the number  of  samples  increases.

Moreover,  the  gradient  updates  suffer  from  several sources of noise leading to a degraded performance.

In this work,  we propose a meta-learning algorithm equipped with the GradiEnt Component COrrections, aGECCO cell for short, which generates a multiplicative corrective low-rank matrix which (after vectorization) corrects the estimated gradients.

GECCO contains a simple decoder-like network with learnable parameters, an attention module and a so-called context input parameter.

The context parameter of GECCO is updated to  generate  a  low-rank  corrective  term  for  the  network  gradients.

As  a  result, meta-learning requires only a few of gradient updates to absorb new task (often, a single update is sufficient in the few-shot scenario).

While previous approaches address this problem by altering the learning rates, factorising network parameters or directly learning feature corrections from features and/or gradients, GECCO is an off-the-shelf generator-like unit that performs element-wise gradient corrections without the need to ‘observe’ the features and/or the gradients directly.

We show that our GECCO (i) accelerates learning, (ii) performs robust corrections of the gradients corrupted by a noise, and (iii) leads to notable improvements over existing gradient-based meta-learning algorithms.

<|TLDR|>

@highlight

We propose a meta-learner to adapt quickly on multiple tasks even one step in a few-shot setting.

@highlight

This paper proposes a method to meta-learn a gradient correction module in which preconditioning is parameterized by a neural network, and builds in a two-stage gradient update process during adaptation. 