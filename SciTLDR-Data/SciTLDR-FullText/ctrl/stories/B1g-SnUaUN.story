We propose a study of the stability of several few-shot learning algorithms subject to variations in the hyper-parameters and optimization schemes while controlling the random seed.

We propose a methodology for testing for statistical differences in model performances under several replications.

To study this specific design, we attempt to reproduce results from three prominent papers: Matching Nets, Prototypical Networks, and TADAM.

We analyze on the miniImagenet dataset on the standard classification task in the 5-ways, 5-shots learning setting at test time.

We find that the selected implementations exhibit stability across random seed, and repeats.

b ∼ Normal(0, σ 2 I) (1) y = Xβ + Zb + α + .(2)Where β ∈ R P is our slope vector, α ∈ R is the intercept, and ∼ Normal(0, I) is the random 127 noise vector.

To model the clusters, we introduce Zb,where Z is the n × q model matrix for the q- and the dependent variable mean is captured by Xβ + α when we marginalize over all the samples.

The random effects component Zb captures variations in the data, it can be interpreted as an individual 134 deviation from the group-level fixed effect.

In our context, we can write the model as follows:136 metric ijk = (A + α 0j + α 1k ) + βExperiment i + i (3) metric ijk = A + βExperiment i + (α 0j + α 1k + i ) (4) DISPLAYFORM0 Where A is the intercept, β is a vector of parameters and Experiment i is a one hot vector of 137 experiments for the observation i.

We can regroup all the random effects, where alpha 0j is a 138 random effect associated with an observation from a random seed j, and alpha 1k is associated to

an observation from a repeat k. Finally, it is possible to regroup all the nuisance parameters in seeds for a given implementation, except for inherent differences due to parallelism on CPU and 250 GPU.

DISPLAYFORM0

Some implementations:

• call a random number generator at an execution point placed before the episodes data 253 generation, hence changing the state of the random number generator,

• generate the episodes data in advance, others generate it for each episode on the fly: different

states of random number generator are involved in the data generation process,

• start training at different states of the random number generator: random number sequences The trend of large-scale compute-intensive ML experiments has caused concern in the community

<|TLDR|>

@highlight

We propose a study of the stability of several few-shot learning algorithms subject to variations in the hyper-parameters and optimization schemes while controlling the random seed.

@highlight

This paper studies reproducibility for few-shot learning.