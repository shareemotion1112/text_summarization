We introduce a more efficient neural architecture for amortized inference, which combines continuous and conditional normalizing flows using a principled choice of structure.

Our gradient flow derives its sparsity pattern from the minimally faithful inverse of its underlying graphical model.

We find that this factorization reduces the necessary numbers both of parameters in the neural network and of adaptive integration steps in the ODE solver.

Consequently, the throughput at training time and inference time is increased, without decreasing performance in comparison to unconstrained flows.

By expressing the structural inversion and the flow construction as compilation passes of a probabilistic programming language, we demonstrate their applicability to the stochastic inversion of realistic models such as convolutional neural networks (CNN).

We introduce a more efficient neural architecture for amortized inference (Gershman, 2014; Ritchie et al., 2016) , which combines continuous (Grathwohl et al., 2018) and conditional normalizing flows using a principled choice of structure.

Our flow derives its sparsity pattern from the minimally faithful inverse of its underlying graphical model (Webb et al., 2018) .

We find that this factorization reduces the necessary numbers both of parameters in the neural network and of adaptive integration steps in the ODE solver.

Consequently, the throughput at training time and inference time is increased, without decreasing performance in comparison to unconstrained flows.

By expressing the structural inversion and the flow construction as compilation passes of a probabilistic programming language, we demonstrate their applicability to the stochastic inversion of realistic models such as convolutional neural networks (CNN).

Our automated pipeline consists of three program transformations, as illustrated in Figure 1 : First, a formal specification of a generative process is translated into a graphical model, and its minimally faithful inverse structure is computed as described in Section 2.

Subsequently, the latter acts as the sparsity pattern for the novel neural network architecture introduced in Section 3.

Finally, the resulting flow is trained with a novel symmetrized KL loss, as summarized in Section 4.

Given a static graphical model, we apply the faithful inversion algorithm of Webb et al. (2018) , and obtain a correct dependence structure for the inverse model p(z|x), which maps from observations x to latents z. In particular, this algorithm computes a structure with minimal number of moralizing edges, which are required to capture all possible correlations in the posterior.

As an example, Panel (3) in Figure 1 shows the minimally faithful inverse of the graphical model in Panel (2).

(1) is compiled into a graphical model (2) and stochastically inverted (3).

This structure is translated into the sparsity pattern of the neural network (4), which approximates the posterior p(z 0,...,4 |x 0,1 ) as a continuous normalizing flow under the control input x. The flow network's architecture is depicted using Hinton diagrams (Hinton and Shallice, 1991) of its layer-wise weight matrices -with color and size denoting sign and magnitude, and columns and rows corresponding to inputs and outputs of layers.

For clarity, augmenting dimensions are not shown.

Amortized inference techniques (Gershman, 2014; Ritchie et al., 2016) yield efficient posterior approximations as a result of training function approximators on losses defined using the generative model and training data, e.g., the variational evidence lower bound (Blei et al., 2016; Kingma and Welling, 2013) .

The general framework used here for inference amortization is a neural ordinary differential equation (ODE) system , a differentiable deterministic transformation from a reference distribution q 0 to the desired target density q ?? (?? | x).

This transformation is parametrized as a on latent particles z,

where conditioning is achieved by providing x as a constant control input to the neural network f ?? .

The numerical computation at inference time is then performed by a standard ODE solver, integrating independent particle trajectories along the dynamics in Equation (1), from initial conditions z 0 ??? q 0 to approximate posterior samples z 1 ??? q ?? (?? | x).

In order to obtain a normalized distribution at the end of the flow, the log-probability of each particle must also be integrated alongside the particle dynamics as

where ??? z denotes the gradient operator in the latent space.

This divergence term is equivalent to the trace of the Jacobian of f ?? .

There are two main algorithmic advantages of this approach: its intrinsic parallelism between independent particles, and the trivial reversibility of the flow transformation using the same integrator in opposite direction.

Recently, such flows have also been applied to graph neural networks as a form of continuous message passing (Deng et al., 2019; Liu et al., 2019) .

Our work differs from such literature chiefly in two ways: we target inference amortization instead of density estimation, and our flows represent a global continuous message passing dynamics on the sparse inverse model structure.

In order to constrain the architecture of the flow network f ?? to respect the necessary statistical independence structure, the weight matrix of each layer h ?? l is masked with the adjacency H of the minimally faithful inverted graphical model, i.e., the output reads

Here the column (h ?? l (???, ??)) i l of activations across layers l corresponds to a node i in the graphical model, ?? is the activation function tanh, b is a bias, and ?? l,?? are time dependent linear gating functions modelling time dependencies of the flow as in .

While our architecture captures the global statistical structure, we have not yet explored inversion of individual link functions as in Tavares and Lezama (2016) .

Optionally, we introduce local nuisance variables to increase the latent space dimension of each random variable, following similar reasoning to Dupont et al. (2019) .

In our experiments, we found L = 3 hidden layers to provide enough over-parametrization.

A simplified version of this architecture is shown in Panel (4) of Figure 1 .

Our optimization objective is the symmetrized Kullback-Leibler divergence in expectation over training data,

While the forward KL term measures the quality of density estimation on the support of the true posterior, the reverse KL term incentivizes samples from q ?? to behave similarly to the latter.

Efficient estimation of this objective is possible in this setting, because the joint model is available and the variational posterior q ?? is reparametrized.

Importantly, in contrast to expected forward or reverse KL alone, L [q ?? ] does not contain the unknown constant factor E x ??? X ln p(x).

In the experiments described below, we uniformly find a significant performance improvement over using only the forward or reverse KL for training.

A quantitative comparison of the minimally faithful inversion structure against three baselines is shown in Figure 2 , measuring the objective in Equation (4) on the arithmetic circuit example from Figure 1 .

Aside from using the same algorithm and hyperparameters, the different architectures are made comparable by choosing similar numbers of dimensions for the latent spaces of the flows: FFJORD (Grathwohl et al., 2018) transforms the original 6D latent space into the flow space using a fully connected layer, while for the other architectures we augment each original latent dimension by 10 additional dimensions.

As a result, FFJORD (64 dimensions, 17801 parameters) and the flows with full connectivity (66 dimensions, 18679 parameters) and minimally faithful inverse structure (66 dimensions, 7725 parameters) achieve competitive performances.

In addition to using significantly fewer parameters, our sparsity structure trains faster in the beginning, suggesting a more appropriate inductive bias.

The importance of faithful inversion is corroborated by a control experiment, which differs only in its randomized sparsity structure (66 dimensions, 7725 parameters), and performs poorly, suffering from early saturation and high variance.

Figure 1 .

By accounting for both divergences, and thereby combining mode-seeking and mass-seeking behaviour, the symmetrized loss provides a stronger learning signal in general.

In this example, the validation loss improves by more than an order of magnitude.

We plot the median and a band of one standard deviation over 10 runs.

Figure 3 shows a comparison of the different losses described in Section 4.

The reverse KL-based loss was found to be capable of training simpler models, such as small Gaussian state space models.

However, it had consistently higher variance than the forward KL and was not at all sufficient for training on the arithmetic circuit we consider, as Figure 3 shows.

The forward KL, the standard loss introduced with CNFs (Grathwohl et al., 2018) , provides a learning signal on the task, but quickly saturates with a reversed KL of about 100 nats.

The symmetrized KL, on the other hand, learns faster from the start and keeps improving to below 10 nats.

This is a crucial improvement, since the forward KL only optimizes q to be a density estimator for p(z|x), while the reverse KL optimizes the sampling behavior of q as well.

Our experiment shows that such a CNF can only be trained with the symmetrized KL.

For this run we have used an augmentation of 5 dimensions for each latent variable, all the other parameters were the same as in the previous result.

The benefits of the symmetrizeded loss wereconsistent over all our experiments.

Figure 4: (a) Adjacency matrix of the minimally faithful inverse structure for a 2D convolution, using the dimension convention of Figure 1 and black/white for 0/1.

(b) Examples of stochastic deconvolution, trained as a flow with the sparsity pattern in (4a).

Each row conditions on an output (4 ?? 4) and a filter (3 ?? 3) to infer corresponding inputs (9 ?? 9).

Posteriors are visualized using 5 samples of the input and the reconstructed outputs.

As an example for a more challenging application, Figure 4 portrays 2D deconvolution, interpreted as amortized inference for the generative process of image convolution.

To obtain output pixels (4 ?? 4), the generative model samples each of the filter weights (3 ?? 3) from a standard normal prior and calculates the forward convolution on an image patch (9 ?? 9) with stride 2 and no padding.

The minimally faithful inversion structure in Figure 4a indicates all statistical dependencies: across (inferred) input pixels, of inputs on filter weights, and of input pixels on their outputs.

For example, pixels in the middle of the input patch visibly depend on all output values.

The inference artifact is trained on randomly cropped real image patches from the MNIST digit classification dataset, and amortizes over all possible convolutional filters of this shape.

It should be noted that in contrast to usual deconvolutional architectures, this stochastic inverse function is trained without explicit weight sharing.

Finally, we perform a qualitative consistency check in Figure 4b , by reconstructing outputs from samples of the approximate posterior.

<|TLDR|>

@highlight

We introduce a more efficient neural architecture for amortized inference, which combines continuous and conditional normalizing flows using a principled choice of sparsity structure.