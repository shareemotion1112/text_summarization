Comparing the inferences of diverse candidate models is an essential part of model checking and escaping local optima.

To enable efficient comparison, we introduce an amortized variational inference framework that can perform fast and reliable posterior estimation across models of the same architecture.

Our Any Parameter Encoder (APE) extends the encoder neural network common in amortized inference to take both a data feature vector and a model parameter vector as input.

APE thus reduces posterior inference across unseen data and models to a single forward pass.

In experiments comparing candidate topic models for synthetic data and product reviews, our Any Parameter Encoder yields comparable posteriors to more expensive methods in far less time, especially when the encoder architecture is designed in model-aware fashion.

We consider the problem of approximate Bayesian inference for latent variable models, such as topic models (Blei et al., 2003) , embedding models (Mohamed et al., 2009) , and dynamical systems models (Shumway and Stoffer, 1991 ).

An important step in using such probabilistic models to extract insight from large datasets is model checking and comparison.

While many types of comparison are possible (Gelman et al., 2013) , we focus on a problem that we call within-model comparison.

Given several candidate parameter vectors θ 1 , θ 2 , . . .

, all from the same space Θ ⊆ R D , our goal is to efficiently determine which parameter θ m is best at explaining a given dataset of N examples {x n } N n=1 .

Multiple ways exist to rank candidate parameters, including performance on heldout data or human-in-the-loop inspection.

A principled choice is to select the parameter that maximizes the data's marginal likelihood: N n=1 log p(x n |θ m ).

For our latent variable models of interest, computing this likelihood requires marginalizing over a hidden variable h n : p(x n |θ m ) = p(x n |h n , θ m )p(h n |θ m )dh n .

This integral is challenging even for a single example n and model m. One promising solution is variational inference (VI).

Using VI, we can estimate an approximate posterior q(h n |x n , θ m ) over hidden variables.

Approximate posteriors q can be used to compute lower bounds on marginal likelihood, and can also be helpful for human inspection of model insights and uncertainties.

However, it is expensive to estimate a separate q at each example n and model m. In this paper, we develop new VI tools 1 that enable rapid-yet-effective within-model comparisons for large datasets.

The need for within-model comparison (and our methods) is present in many practical modeling tasks.

Here we discuss two possible scenarios, with some details specialized to our intended topic modeling applications (Blei, 2012) .

First, in human-in-the-loop scenarios, a domain expert may inspect some estimated parameter θ and then suggest an alternative parameter θ that improves interpretability.

In topic modeling, this may mean removing "intruder words" to make topics more coherent (Chang et al., 2009) .

Second, in automated parameter learning scenarios, many algorithms propose data-driven transformations of the current solution θ into a new candidate θ , in order to escape the local optima common in non-convex optimization objectives for latent variable models (Roberts et al., 2016) , Examples include split-merge proposal moves (Ueda and Ghahramani, 2002; Jain and Neal, 2004) or evolutionary algorithms (Sundararajan and Mengshoel, 2016) .

Across both these scenarios, new candidates θ arise repeatedly over time, and estimating approximate posteriors for each is essential to assess fitness yet expensive to perform for large datasets.

Our contribution is the Any Parameter Encoder (APE), which amortizes posterior inference across models θ m and data x n .

We are inspired by efforts to scale a single model to large datasets by using an encoder neural network (NN) to amortize posterior inference across data examples (Rezende et al., 2014; Kingma and Welling, 2014) .

Our key idea is that to additionally generalize across models, we feed model parameter vector θ m and data feature vector x n as input to the encoder.

APE is applicable to any model with continuous hidden variables for which amortized inference is possible via the reparameterization trick.

We consider a general family of probabilistic models that use parameter vector θ m to generate a dataset of N continuous hidden variables h n and observations x n via a factorized distribution: N n=1 p(h n |θ m )p(x n |h n , θ m ).

Our goal is fast-yet-accurate estimation of each example's local posterior p(h n |x n , θ m ) for a range of model parameters θ 1 , θ 2 , . . .

∈ Θ.

Topic Models.

As a sample application, we focus on the Logistic Normal topic model from Srivastava and Sutton (2017) .

Given known vocabulary size V , we observe N documents represented by count vectors x n (vector of size V counting the types of all T n words in document n).

We model each x n as a mixture of K possible topics.

Let hidden variable h nk be the probability that a word in document n is produced by topic k. Thus, h n ∈ ∆ K is a non-negative vector of size K that sums to one.

We model h n with a Logistic Normal prior, with mean and covariance set to be similar to a sparse Dirichlet(0.01) prior (Hennig et al., 2012) for interpretability.

Given h n , we model the observed word-count vector for document n with a Multinomial likelihood:

.

This is a document-specific mixture of topics, where each topic k is defined by a word probability vector θ k ∈ ∆ V .

Our parameter of interest is the topic-word probability vector θ = {θ k } K k=1 .

VI Approximations for the Single Example Posterior.

While the true posterior p(h n |x n , θ m ) is usually intractable, we use variational inference (VI) (Wainwright and Jordan, 2008 ) to approximate it.

We choose a simpler density q(h n |λ n ) and optimize parameter λ n to minimize KL divergence from the true posterior.

Inference reduces to the well-known evidence lower bound (ELBO) optimization problem given example x n and model θ m :

Inference:

Given several parameters of interest, we can perform model comparison by solving the above optimization problem separately for each θ m .

However, this is expensive.

Solving Eq. (1) for a model θ m requires dozens of iterative updates of gradient ascent for each example.

VI Amortized across Data Examples.

Previously, Rezende et al. (2014) and Kingma and Welling (2014) have sped up inference by setting per-example variational parameters λ n to the output of an encoder neural network (NN) instead of an iterative optimization procedure.

The "Standard" encoder, with weights parameters φ, takes input data x n and produces variational parameters λ NN φ (x n ).

Inference for example n reduces to one fast forward pass: λ n ← λ NN φ (x n ).

While encoders often produce λ n with worse ELBO scores than optimal solutions to Eq. (1) (Krishnan et al., 2018) , they are preferred for their speed.

However, for our model comparison goals the standard encoder is expensive, because for each parameter θ m of interest we must train separate specialized NN weights φ m .

Contribution: VI Amortized over Model Parameters.

Our goal is to enable rapid estimation of posteriors p(h n |x n , θ m ) for many possible parameters θ 1 , θ 2 , . . .

∈ Θ (not all known in advance).

We thus consider a family of approximating densities q that explicitly conditions on both a given data vector x n and the query parameter vector θ m .

Again, we use a neural network to transform these inputs into the variational parameters, λ n ← λ NN φ (x n , θ m ).

We call this the Any Parameter Encoder.

Unlike the earlier Standard Encoder, which trains φ for one specific θ, our approach can directly generalize to many θ.

Encoder Architecture Design for Topic Models.

Given the difficulty of posterior inference even for a single parameter θ, developing an effective Any Parameter Encoder requires careful selection of a NN architecture that can transform its two inputs, data x n and model θ, to produce accurate approximate posteriors.

Following previous work (Kingma and Welling, 2014), we use multi-layer perceptrons.

We further suggest that an architecture designed to capture structure in the generative model should improve results further.

Our baseline "naive" architecture defines the input of the neural net as simply the concatenation of vector x n and vector θ.

While simple, we suggest this will be difficult to train effectively given the size of the input ((K + 1)V for the topic model) and lack of inductive bias to prioritize the model's needed interactions between entries of x n and θ.

As an improvement, we consider a model-aware encoder architecture.

Our design is motivated by a view of posterior inference as roughly moment-matching when data is plentiful.

For our topic model, each document's Multinomial likelihood has a mean vector equal to k h nk θ k = θh n , writing θ as a V × K matrix.

This mean vector should be (roughly) equal to the observed word-frequency vector 1 Tn x n .

If µ n is the mean of q(h n ) and used as a plug-in estimate for h n , then we want to satisfy 1 Tn x n ≈ θµ n .

Solving for µ n via least squares, we get µ n ≈ 1 Tn (θ T θ) −1 θ T x n , which we might simplify to a non-linear function of θ T x n .

Thus, we suggest using the following model-aware encoder architecture:

This model-aware architecture has encoder input dimension K, which is much smaller than (K + 1)V for the naive approach (and thus hopefully easier to train).

Furthermore, this should provide desirable inductive bias to produce useful mean and covariance estimates.

We emphasize that this design is specialized to the topic model, and further work is needed to develop model-aware architecture design strategies for general latent variable models.

Training the Encoder.

Training our encoder parameters φ requires an available set of M parameter vectors {θ m } M m=1 of interest.

We choose these to be representative of the subset of Θ we wish to generalize well to.

We then maximize ELBO across all M models:

We use stochastic gradient ascent to solve for φ, using the reparameterization trick to estimate gradients for a minibatch of examples and models at each step.

We can interpret this objective as an expectation over samples θ m from a target distribution over parameters.

We compare our proposed Any Parameter Encoder (APE) to several other inference methods on two topic modeling tasks.

For all VI methods, we choose q to be a Logistic Normal parameterized by a mean and a diagonal covariance.

The appendix has complete details.

APE.

We consider both naive and model-aware encoder architectures described above.

Both use MLPs with 2 layers with 100 units per hidden layer, selected via grid search.

Baselines.

We consider three baselines implemented in Pyro (Bingham et al., 2018) and PyTorch (Paszke et al., 2017) .

First, Variational Inference (VI) uses gradient ascent to optimize Eq. (1).

Second, we use Standard encoder VAEs for topic models (Srivastava and Sutton, 2017) .

This encoder is specialized to a single parameter θ, with architecture size selected via grid search (similar to APE).

Finally, we run Pyro's off-the-shelf implementation of Hamiltonian Monte Carlo with the No U-Turn Sampler (NUTS) (Hoffman and Gelman, 2014), though we expect specialized implementations to be more performant.

Synthetic Data Experiments.

We consider a V = 100 vocabulary dataset inspired by the "toy bars" of Griffiths and Steyvers (2004) .

Using K = 20 true topics θ * , we sample Error from VI ELBO Product Reviews Figure 1 : Left: ELBO vs. elapsed time for VI on a test set of 300 document-θ m combinations on synthetic data.

We show a randomly-initialized run (black) and a warm-started run (green) initialized via our Any-Parameter Encoder (red "X").

The randomly-initialized VI would require over 400 milliseconds (vertical line) to reach the quality our APE achieved in <20 ms.

Right: Kernel Density Estimation of absolute difference between encoder ELBO and VI ELBO across different topics.

APE results (red) are closer to VI (i.e. less error).

a 500-document dataset.

We consider M = 50, 000 possible model parameters {θ m } M m=1 , sampled from a symmetric, sparse Dirichlet prior over the vocabulary.

Typical θ m look unlike the true topics θ * , as shown in the supplement, so inference must handle diversity well.

We train our APE on 25 million possible document-θ pairs for two epochs, then evaluate on unseen document-θ pairs drawn from the same generative process.

Product Reviews.

We model 6,343 text documents of consumer product reviews (Blitzer et al., 2007) .

We use the V = 3000 most frequent vocabulary terms and K = 30 topics.

We generate training topics in the same way as in the synthetic data experiments, and we evaluate on test topics found via Gibbs sampling with several separately initialized runs.

Results: Encoder Design.

Results comparing naive and model-aware encoder architectures are in Table 1 .

Our proposed model-aware input layer yields better heldout likelihoods than the naive alternative, which we suggest is due to its more effective inductive bias.

Results: Quality-vs-Time Tradeoff.

Comparing results across Table 1 and Fig. 1 , we see that while the Standard Encoder understandably fails to generalize across models, our Any Parameter Encoder achieves quality close to non-amortized VI and NUTS with a speed up factor of over 100-1000x.

APE can also provide a useful warm start initialization to VI.

Results: Agreement in model comparison.

Motivated by the need to rapidly assess proposal moves that escape local optima, we gather 10 different models and measure whether each encoder's ranking of a pair θ, θ on the test set agrees with VI's ranking.

Table 1 shows that APE agrees with VI in 75% of 45 cases in the real data scenario, while Standard Encoder agrees just 29% of the time.

This suggests APE may be trustworthy for accept/reject decisions, though further work is needed to improve this number further.

Across two datasets and many model parameters, our Any Parameter Encoder produces posterior approximations that are nearly as good as expensive VI, but over 100x faster.

Future opportunities include simultaneous training of parameters and encoders, and handling Bayesian nonparametric models where θ changes size during training (Hughes et al., 2015 n )) (6) For encoder methods, the parameters {µ n , log σ 2 n } are the output of a shared encoder NN.

For VI, these are free parameters of the optimization problem.

Variational Inference (VI).

We perform using gradient ascent to maximize the objective in Eq. (1), learning a per-example mean and variance variational parameter.

We run gradient updates until our moving average loss (window of 10 steps) has improved by less than 0.001% of its previous value.

For our VI runs from random initializations, we use the Adam optimizer with an initial learning rate of .01, decaying the rate by 50% every 5000 steps.

For our warm-started runs, we use an initial learning rate of 0.0005.

In practice, we ran VI multiple times with different learning rate parameters and took the best one.

Table  1 only reports the time to run the best setting, not the total time which includes various restarts.

Standard encoder.

We use a standard encoder that closely matches the VAE for topic models in Srivastava and Sutton (2017) .

The only architectural difference is the addition of a temperature parameter on the µ n vector before applying the softmax to ensure the means lie on the simplex.

We found that the additional parameter sped up training by allowing the peakiness of the posterior to be directly tuned by a single parameter.

We use a feedforward encoder with two hidden layers, each 100 units.

We chose the architecture via hyperparameter sweeps.

The total number of trainable parameters in the model is 24,721 on the synthetic data and 316,781 on the real data; this is compared to 16,721 and 19,781 parameters for model-aware APE.

NUTS.

For the Hamiltonian Monte Carlo (HMC) with the No U-Turn Sampler (NUTS) (Hoffman and Gelman, 2014) , we use a step size of 1 adapted during the warmup phase using Dual Averaging scheme.

Upon inspection, we find that the method's slightly lower posterior predictive log likelihood relative to VI is due to its wider posteriors.

We also find that the Pyro implementation is (understandably) quite slow and consequently warm-start the NUTS sampler using VI to encourage rapid mixing.

We are aware that there exist faster, more specialized implementations, but we decided to keep our tooling consistent for scientific purposes.

We generate a set of different models {θ 0 , θ 1 , ...θ M } from a symmetric Dirichlet prior with α = 0.1.

We train our Any-Parameter Encoder in random batches of document-topic combinations.

With 500 documents and 50,000 topics (i.e. D = 500, M = 50, 000), we have 25 million combinations in total.

The topics used to generate the synthetic data represent "toy bars", inspired by (Griffiths and Steyvers, 2004) .

See Figure 2 for a visualization.

We use this same toy bars-biased prior to generate all our topics in the holdout set, though the order of the topics is random.

See

For training both APE and the Standard encoder on the synthetic data, we use Adam with an exponential decay learning schedule, a starting learning rate of 0.01, and a decay rate of .8 every 50,000 steps.

We find that this schedule tends to be fairly robust; these hyperparameters were used for both APE and the Standard encoder on both the synthetic and real data.

We chose our initial learning rate via a learning rate finder posed in Smith (2017), and we train for 2 epochs with a batch size of 100.

We train our standard VAE encoder on a single model with parameters θ drawn randomly from a symmetric Dirichlet prior with α = 0.1.

To train the standard encoder, we pass in our model of interest to the decoder, holding its weights fixed as we perform stochastic backpropagation to update the encoder weights.

The same thing happens for APE, though the same topics are additionally included as part of the input into the encoder.

We develop VAEs where the encoder takes a model parameter vector as additional input, so we can do rapid inference for many models

@highlight

We develop VAEs where the encoder takes a model parameter vector as input, so we can do rapid inference for many models