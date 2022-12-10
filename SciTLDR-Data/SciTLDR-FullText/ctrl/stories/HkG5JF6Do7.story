There exists a plethora of techniques for inducing structured sparsity in parametric models during the optimization process, with the final goal of resource-efficient inference.

However, to the best of our knowledge, none target a specific number of floating-point operations (FLOPs) as part of a single end-to-end optimization objective, despite reporting FLOPs as part of the results.

Furthermore, a one-size-fits-all approach ignores realistic system constraints, which differ significantly between, say, a GPU and a mobile phone -- FLOPs on the former incur less latency than on the latter; thus, it is important for practitioners to be able to specify a target number of FLOPs during model compression.

In this work, we extend a state-of-the-art technique to directly incorporate FLOPs as part of the optimization objective and show that, given a desired FLOPs requirement, different neural networks can be successfully trained for image classification.

Neural networks are a class of parametric models that achieve the state of the art across a broad range of tasks, but their heavy computational requirements hinder practical deployment on resourceconstrained devices, such as mobile phones, Internet-of-things (IoT) devices, and offline embedded systems.

Many recent works focus on alleviating these computational burdens, mainly falling under two non-mutually exclusive categories: manually designing resource-efficient models, and automatically compressing popular architectures.

In the latter, increasingly sophisticated techniques have emerged BID3 BID4 BID5 , which have achieved respectable accuracy-efficiency operating points, some even Pareto-better than that of the original network; for example, network slimming BID3 reaches an error rate of 6.20% on CIFAR-10 using VGGNet BID9 with a 51% FLOPs reduction-an error decrease of 0.14% over the original.

However, to the best of our knowledge, none of the methods impose a FLOPs constraint as part of a single end-to-end optimization objective.

MorphNets BID0 apply an L 1 norm, shrinkage-based relaxation of a FLOPs objective, but for the purpose of searching and training multiple models to find good network architectures; in this work, we learn a sparse neural network in a single training run.

Other papers directly target device-specific metrics, such as energy usage BID15 , but the pruning procedure does not explicitly include the metrics of interest as part of the optimization objective, instead using them as heuristics.

Falling short of continuously deploying a model candidate and measuring actual inference time, as in time-consuming neural architectural search BID11 , we believe that the number of FLOPs is reasonable to use as a proxy measure for actual latency and energy usage; across variants of the same architecture, Tang et al. suggest that the number of FLOPs is a stronger predictor of energy usage and latency than the number of parameters BID12 .Indeed, there are compelling reasons to optimize for the number of FLOPs as part of the training objective: First, it would permit FLOPs-guided compression in a more principled manner.

Second, practitioners can directly specify a desired target of FLOPs, which is important in deployment.

Thus, our main contribution is to present a novel extension of the prior state of the art BID6 to incorporate the number of FLOPs as part of the optimization objective, furthermore allowing practitioners to set and meet a desired compression target.

Formally, we define the FLOPs objective L f lops : f × R m → N 0 as follows: DISPLAYFORM0 where L f lops is the FLOPs associated with hypothesis h(·; θ θ θ) := p(·|θ θ θ), g(·) is a function with the explicit dependencies, and I is the indicator function.

We assume L f lops to depend only on whether parameters are non-zero, such as the number of neurons in a neural network.

For a dataset D, our empirical risk thus becomes DISPLAYFORM1 Hyperparameters λ f ∈ R + 0 and T ∈ N 0 control the strength of the FLOPs objective and the target, respectively.

The second term is a black-box function, whose combinatorial nature prevents gradient-based optimization; thus, using the same procedure in prior art BID6 , we relax the objective to a surrogate of the evidence lower bound with a fully-factorized spike-and-slab posterior as the variational distribution, where the addition of the clipped FLOPs objective can be interpreted as a sparsity-inducing prior p(θ θ θ) DISPLAYFORM2 where denotes the Hadamard product.

To allow for efficient reparameterization and exact zeros, Louizos et al. BID6 propose to use a hard concrete distribution as the approximation, which is a stretched and clipped version of the binary Concrete distribution BID7 : ifẑ ∼ BinaryConcrete(α, β), thenz := max(0, min(1, (ζ − γ)ẑ + γ)) is said to be a hard concrete r.v., given ζ > 1 and γ < 0.

Define φ φ φ := (α α α, β), and let ψ(φ φ φ) = Sigmoid(log α α α − β log DISPLAYFORM3 ψ(·) is the probability of a gate being non-zero under the hard concrete distribution.

It is more efficient in the second expectation to sample from the equivalent Bernoulli parameterization compared to hard concrete, which is more computationally expensive to sample multiple times.

The first term now allows for efficient optimization via the reparameterization trick BID2 ; for the second, we apply the score function estimator (REINFORCE) BID14 , since the FLOPs objective is, in general, nondifferentiable and thus precludes the reparameterization trick.

High variance is a non-issue because the number of FLOPs is fast to compute, hence letting many samples to be drawn.

At inference time, the deterministic estimator isθ θ θ := θ θ θ max(0, min(1, Sigmoid(log α α α)(ζ − γ) + γ)) for the final parametersθ θ θ.

In practice, computational savings are achieved only if the model is sparse across "regular" groups of parameters, e.g., each filter in a convolutional layer.

Thus, each computational group uses one hard concrete r.v.

BID6 -in fully-connected layers, one per input neuron; in 2D convolution layers, one per output filter.

Under convention in the literature where one addition and one multiplication each count as a FLOP, the FLOPs for a 2D convolution layer h conv (·; θ θ θ) given a random draw z is then defined as L f lops (h conv , z) = (K w K h C in + 1)(I w − K w + P w + 1)(I h − K h + P h + 1) z 0 for kernel width and height (K w , K h ), input width and height (I w , I h ), padding width and height (P w , P h ), and number of input channels C in .

The number of FLOPs for a fully-connected layer h f c (·; θ θ θ) is L f lops (h f c , z) = (I n + 1) z 0 , where I n is the number of input neurons.

Note that these are conventional definitions in neural network compression papers-the objective can easily use instead a number of FLOPs incurred by other device-specific algorithms.

Thus, at each training step, we compute the FLOPs objective by sampling from the Bernoulli r.v.

's and using the aforementioned definitions, e.g., L f lops (h conv , ·) for convolution layers.

Then, we apply the score function estimator to the FLOPs objective as a black-box estimator.

We report results on MNIST, CIFAR-10, and CIFAR-100, training multiple models on each dataset corresponding to different FLOPs targets.

We follow the same initialization and hyperparameters as Louizos et al. BID6 , using Adam BID1 with temporal averaging for optimization, a weight decay of 5 × 10 −4 , and an initial α that corresponds to the original dropout rate of that layer.

We similarly choose β = 2/3, γ = −0.1, and ζ = 1.1.

For brevity, we direct the interested reader to their repository BID0 for specifics.

In all of our experiments, we replace the original L 0 penalty with our FLOPs objective, and we train all models to 200 epochs; at epoch 190, we prune the network by weights associated with zeroed gates and replace the r.v.

's with their deterministic estimators, then finetune for 10 more epochs.

For the score function estimator, we draw 1000 samples at each optimization step-this procedure is fast and has no visible effect on training time.

BID10 7-13-208-16 1.1% 254K SBP BID8 3-18-284-283 0.9% 217K BC-GNJ BID5 8-13-88-13 1.0% 290K BC-GHS BID5 5-10-76-16 1.0% 158K L 0 BID6 20-25-45-462 0.9% 1.3M L 0 -sep BID6 9-18-65-25 1.0% 403K DISPLAYFORM0 We choose λ f = 10 −6 in all of the experiments for LeNet-5-Caffe, the Caffe variant of LeNet-5.

BID0 We observe that our methods TAB0 , bottom three rows) achieve accuracy comparable to those from previous approaches while using fewer FLOPs, with the added benefit of providing a tunable "knob" for adjusting the FLOPs.

Note that the convolution layers are the most aggressively compressed, since they are responsible for most of the FLOPs in this model.

Orig.

in TAB1 denotes the original WRN-28-10 model BID16 , and L 0 -* refers to the L 0 -regularized models BID6 ; likewise, we augment CIFAR-10 and CIFAR-100 with standard random cropping and horizontal flipping.

For each of our results (last two rows), we report the median error rate of five different runs, executing a total of 20 runs across two models for each of the two datasets; we use λ f = 3 × 10 −9 in all of these experiments.

We also report both the expected FLOPs and actual FLOPs, the former denoting the number of FLOPs, on average, at training time under stochastic gates and the latter denoting the number of FLOPs at inference time.

We restrict the FLOPs calculations to the penalized non-residual convolution layers only.

For CIFAR-10, our approaches result in Pareto-better models with decreases in both error rate and the actual number of inference-time FLOPs.

For CIFAR-100, we do not achieve a Pareto-better model, since our approach trades accuracy for improved efficiency.

The acceptability of the tradeoff depends on the end application.

<|TLDR|>

@highlight

We extend a state-of-the-art technique to directly incorporate FLOPs as part of the optimization objective, and we show that, given a desired FLOPs requirement, different neural networks are successfully trained.