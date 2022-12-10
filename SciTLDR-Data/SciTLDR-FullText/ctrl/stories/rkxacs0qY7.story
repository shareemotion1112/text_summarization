Variational Bayesian neural networks (BNN) perform variational inference over weights, but it is difficult to specify meaningful priors and approximating posteriors in a high-dimensional weight space.

We introduce functional variational Bayesian neural networks (fBNNs), which maximize an Evidence Lower BOund (ELBO) defined directly on stochastic processes, i.e. distributions over functions.

We prove that the KL divergence between stochastic processes is equal to the supremum of marginal KL divergences over all finite sets of inputs.

Based on this, we introduce a practical training objective which approximates the functional ELBO using finite measurement sets and the spectral Stein gradient estimator.

With fBNNs, we can specify priors which entail rich structure, including Gaussian processes and implicit stochastic processes.

Empirically, we find that fBNNs extrapolate well using various structured priors, provide reliable uncertainty estimates, and can scale to large datasets.

Bayesian neural networks (BNNs) BID22 BID40 have the potential to combine the scalability, flexibility, and predictive performance of neural networks with principled Bayesian uncertainty modelling.

However, the practical effectiveness of BNNs is limited by our ability to specify meaningful prior distributions and by the intractability of posterior inference.

Choosing a meaningful prior distribution over network weights is difficult because the weights have a complicated relationship to the function computed by the network.

Stochastic variational inference is appealing because the update rules resemble ordinary backprop BID15 BID4 , but fitting accurate posterior distributions is difficult due to strong and complicated posterior dependencies BID34 BID51 .In a classic result, BID40 showed that under certain assumptions, as the width of a shallow BNN was increased, the limiting distribution is a Gaussian process (GP).

BID31 recently extended this result to deep BNNs.

Deep Gaussian Processes (DGP) BID5 BID49 have close connections to BNNs due to similar deep structures.

However, the relationship of finite BNNs to GPs is unclear, and practical variational BNN approximations fail to match the predictions of the corresponding GP.

Furthermore, because the previous analyses related specific BNN architectures to specific GP kernels, it's not clear how to design BNN architectures for a given kernel.

Given the rich variety of structural assumptions that GP kernels can represent BID45 BID33 , there remains a significant gap in expressive power between BNNs and GPs (not to mention stochastic processes more broadly).In this paper, we perform variational inference directly on the distribution of functions.

Specifically, we introduce functional variational BNNs (fBNNs), where a BNN is trained to produce a distribution of functions with small KL divergence to the true posterior over functions.

We prove that the KL divergence between stochastic processes can be expressed as the supremum of marginal KL divergences at finite sets of points.

Based on this, we present functional ELBO (fELBO) training objective.

Then we introduce a GAN-like minimax formulation and a sampling-based approximation for functional variational inference.

To approximate the marginal KL divergence gradients, we adopt the recently proposed spectral Stein gradient estimator (SSGE) BID52 .

Here a × b represents a hidden layers of b units.

Red dots are 20 training points.

The blue curve is the mean of final prediction, and the shaded areas represent standard derivations.

We compare fBNNs and Bayes-by-Backprop (BBB).

For BBB, which performs weight-space inference, varying the network size leads to drastically different predictions.

For fBNNs, which perform functionspace inference, we observe consistent predictions for the larger networks.

Note that the 1 × 100 factorized Gaussian fBNNs network is not expressive enough to generate diverse predictions.

Our fBNNs make it possible to specify stochastic process priors which encode richly structured dependencies between function values.

This includes stochastic processes with explicit densities, such as GPs which can model various structures like smoothness and periodicity BID33 .

We can also use stochastic processes with implicit densities, such as distributions over piecewise linear or piecewise constant functions.

Furthermore, in contrast with GPs, fBNNs efficiently yield explicit posterior samples of the function.

This enables fBNNs to be used in settings that require explicit minimization of sampled functions, such as Thompson sampling BID57 BID48 or predictive entropy search BID21 BID59 .One desideratum of Bayesian models is that they behave gracefully as their capacity is increased BID44 .

Unfortunately, ordinary BNNs don't meet this basic requirement: unless the asymptotic regime is chosen very carefully (e.g. BID40 ), BNN priors may have undesirable behaviors as more units or layers are added.

Furthermore, larger BNNs entail more difficult posterior inference and larger description length for the posterior, causing degeneracy for large networks, as shown in FIG0 .

In contrast, the prior of fBNNs is defined directly over the space of functions, thus the BNN can be made arbitrarily large without changing the functional variational inference problem.

Hence, the predictions behave well as the capacity increases.

Empirically, we demonstrate that fBNNs generate sensible extrapolations for both explicit periodic priors and implicit piecewise priors.

We show fBNNs outperform competing approaches on both small scale and large scale regression datasets.

fBNNs' reliable uncertainty estimates enable state-of-art performance on the contextual bandits benchmark of BID46 .

, a Bayesian neural network (BNN) is defined in terms of a prior p(w) on the weights, as well as the likelihood p(D|w).

Variational Bayesian methods BID22 BID15 BID4 attempt to fit an approximate posterior q(w) to maximize the evidence lower bound (ELBO): DISPLAYFORM0 (1) The most commonly used variational BNN training method is Bayes By Backprop (BBB) BID4 , which uses a fully factorized Gaussian approximation to the posterior, i.e. q(w) = N (w; µ, diag(σ 2 )).

Using the reparameterization trick BID25 , the gradients of ELBO towards µ, σ can be computed by backpropagation, and then be used for updates.

Most commonly, the prior p(w) is chosen for computational convenience; for instance, independent Gaussian or Gaussian mixture distributions.

Other priors, including log-uniform priors BID26 and horseshoe priors BID13 , were proposed for specific purposes such as model compression and model selection.

But the relationships of weight-space priors to the functions computed by networks are difficult to characterize.

A stochastic process BID28 F is typically defined as a collection of random variables, on a probability space (Ω, F, P ).

The random variables, indexed by some set X , all take values in the same mathematical space Y .

In other words, given a probability space (Ω, Σ, P ), a stochastic process can be simply written as {F (x) : x ∈ X }.

For any point ω ∈ Ω, F (·, ω) is a sample function mapping index space X to space Y, which we denote as f for notational simplicity.

For any finite index set x 1:n = {x 1 , ..., x n }, we can define the finite-dimensional marginal joint distribution over function values {F (x 1 ), · · · , F (x n )}.

For example, Gaussian Processes have marginal distributions as multivariate Gaussians.

The Kolmogorov Extension Theorem (Øksendal, 2003) shows that a stochastic process can be characterized by marginals over all finite index sets.

Specifically, for a collection of joint distributions ρ x1:n , we can define a stochastic process F such that for all x 1:n , ρ x1:n is the marginal joint distribution of F at x 1:n , as long as ρ satisfies the following two conditions:Exchangeability.

For any permutation π of {1, · · · , n}, ρ π(x1:n) (π(y 1:n )) = ρ x1:n (y 1:n ).Consistency.

For any 1 ≤ m ≤ n, ρ x1:m (y 1:m ) = ρ x1:n (y 1:n )dy m+1:n .

When applying Bayesian methods to modern probabilistic models, especially those with neural networks as components (e.g., BNNs and deep generative models), it is often the case that we have to deal with intractable densities.

Examples include the marginal distribution of a non-conjugate model (e.g., the output distribution of a BNN), and neural samplers such as GANs BID14 .

A shared property of these distributions is that they are defined through a tractable sampling process, despite the intractable density.

Such distributions are called implicit distributions BID23 .The Spectral Stein Gradient Estimator (SSGE) BID52 ) is a recently proposed method for estimating the log density derivative function of an implicit distribution, only requiring samples from the distribution.

Specifically, given a continuous differentiable density q(x), and a positive definite kernel k(x, x ) in the Stein class BID32 of q, they show DISPLAYFORM0 where {ψ j } j≥1 is a series of eigenfunctions of k given by Mercer's theorem: DISPLAYFORM1 The Nyström method BID3 BID61 ) is used to approximate the eigenfunctions ψ j (x) and their derivatives.

The final estimator is given by truncating the sum in Equation (2) and replacing the expectation by Monte Carlo estimates.

We introduce function space variational inference analogously to weight space variational inference (see Section 2.1), except that the distributions are over functions rather than weights.

We assume a stochastic process prior p over functions f : X → Y. This could be a GP, but we also allow stochastic processes without closed-form marginal densities, such as distributions over piecewise linear functions.

For the variational posterior q φ ∈ Q, we consider a neural network architecture with stochastic weights and/or stochastic inputs.

Specifically, we sample a function from q by sampling a random noise vector ξ and defining f (x) = g φ (x, ξ) for some function g φ .

For example, standard weight space BNNs with factorial Gaussian posteriors can be viewed this way using the reparameterization trick BID25 BID4 .

(In this case, φ corresponds to the means and variances of all the weights.)

Note that because a single vector ξ is shared among all input locations, it corresponds to randomness in the function, rather than observation noise; hence, the sampling of ξ corresponds to epistemic, rather than aleatoric, uncertainty BID6 .Functional variational inference maximizes the functional ELBO (fELBO), akin to the weight space ELBO in Equation (1), except that the distributions are over functions rather than weights.

DISPLAYFORM0 Here KL[q p] is the KL divergence between two stochastic processes.

As pointed out in BID39 , it does not have a convenient form as log q(f ) p(f ) q(f )df due to there is no infinitedimensional Lebesgue measure BID7 .

Since the KL divergence between stochastic processes is difficult to work with, we reduce it to a more familiar object: KL divergence between the marginal distributions of function values at finite sets of points, which we term measurement sets.

Specifically, let X ∈ X n denote a finite measurement set and P X the marginal distribution of function values at X. We equate the function space KL divergence to the supremum of marginal KL divergences over all finite measurement sets: Theorem 1.

For two stochastic processes P and Q, DISPLAYFORM1 Roughly speaking, this result follows because the σ-algebra constructed with the Kolmogorov Extension Theorem (Section 2.2) is generated by cylinder sets which depend only on finite sets of points.

A full proof is given in Appendix A.fELBO.

Using this characterization of the functional KL divergence, we rewrite the fELBO: DISPLAYFORM2 We also denote L n (q) := inf X∈X n L X (q) for the restriction to sets of n points.

This casts maximizing the fELBO as a two-player zero-sum game analogous to a generative adversarial network (GAN) BID14 : one player chooses the stochastic network, and the adversary chooses the measurement set.

Note that the infimum may not be attainable, because the size of the measurement sets is unbounded.

In fact, the function space KL divergence may be infinite, for instance if the prior assigns measure zero to the set of functions representable by a neural network BID0 .

Observe that GANs face the same issue: because a generator network is typically limited to a submanifold of the input domain, an ideal discriminator could discriminate real and fake images perfectly.

However, by limiting the capacity of the discriminator, one obtains a useful training objective.

By analogy, we obtain a well-defined and practically useful training objective by restricting the measurement sets to a fixed finite size.

This is discussed further in the next section.

As discussed above, we approximate the fELBO using finite measurement sets to have a well-defined and practical optimization objective.

We now discuss how to choose the measurement sets.

Adversarial Measurement Sets.

The minimax formulation of the fELBO naturally suggests a two-player zero-sum game, analogous to GANs, whereby one player chooses the stochastic network representing the posterior, and the adversary chooses the measurement set.

max DISPLAYFORM0 We adopt concurrent optimization akin to GANs BID14 .

In the inner loop, we minimize L X (q) with respect to X; in the outer loop, we maximize L X (q) with respect to q.

Unfortunately, this approach did not perform well in terms of generalization.

The measurement set which maximizes the KL term is likely to be close to the training data, since these are the points where one has the most information about the function.

But the KL term is the only part of the fELBO encouraging the network to match the prior structure.

Hence, if the measurement set is close to the training data, then nothing will encourage the network to exploit the structured prior for extrapolation.

Sampling-Based Measurement Sets.

Instead, we adopt a sampling-based approach.

In order to use a structured prior for extrapolation, the network needs to match the prior structure both near the training data and in the regions where it must make predictions.

Therefore, we sample measurement sets which include both (a) random training inputs, and (b) random points from the domain where one is interested in making predictions.

We replace the minimization in Equation (6) with a sampling distribution c, and then maximize the expected L X (q): max DISPLAYFORM1 where X M are M points independently drawn from c.

Consistency.

With the restriction to finite measurement sets, one only has an upper bound on the true fELBO.

Unfortunately, this means the approximation is not a lower bound on the log marginal likelihood (log-ML) log p(D).

Interestingly, if the measurement set is chosen to include all of the training inputs, then L(q) is in fact a log-ML lower bound: DISPLAYFORM2 The proof is given in Appendix B.1.To better understand the relationship between adversarial and sampling-based inference, we consider the idealized scenario where the measurement points in both approaches include all training locations, i.e., DISPLAYFORM3 is equivalent to minimizing the KL divergence from the true posterior on points X M , X D .

Based on this, we have the following consistency theorem that helps justify the use of adversarial and sampling-based objectives with finite measurement points.

Corollary 3 (Consistency under finite measurement points).

Assume that the true posterior p(f |D) is a Gaussian process and the variational family Q is all Gaussian processes.

We have the following results if M > 1 and supp(c) = X : DISPLAYFORM4 (10)The proof is given in Appendix B.2.

While it is usually impractical for the measurement set to contain all the training inputs, it is still reassuring that a proper lower bound can be obtained with a finite measurement set.

While the likelihood term of the fELBO is tractable, the KL divergence term remains intractable because we don't have an explicit formula for the variational posterior density q φ (f X ).

(Even if q φ is chosen to have a tractable density in weight space , the marginal distribution over f X is likely intractable.)

To derive an approximation, we first observe that DISPLAYFORM0 The first term (expected score function) in Equation FORMULA12 is zero, so we discard it.1 The Jacobian ∇ φ f X can be exactly multiplied by a vector using backpropagation.

Therefore, it remains to estimate the log-density derivatives ∇ f log q(f X ) and ∇ f log p(f X ).The entropy derivative ∇ f log q(f X ) is generally intractable.

For priors with tractable marginal densities such as GPs BID45 2 and Student-t Processes BID50 , DISPLAYFORM1 is tractable.

However, we are also interested in implicit stochastic process priors, i.e. ∇ f log p(f X ) is also intractable.

Because the SSGE (see Section 2.3) can estimate score functions for both in-distribution and out-of-distribution samples, we use it to estimate both derivative terms in all our experiments. (We compute ∇ f log p(f X ) exactly whenever it is tractable.)

Require: Dataset D, variational posterior g(·), prior p (explicit or implicit), KL weight λ.

Require: Sampling distribution c for random measurement points.

1: while φ not converged do 2: DISPLAYFORM0 sample k function values 4: DISPLAYFORM1 compute log likelihood gradients 5: DISPLAYFORM2 φ ← Optimizer(φ, ∆ 1 − λ∆ 2 ) update the parameters 7: end while 3.4 THE ALGORITHM Now we present the whole algorithm for fBNNs in Algorithm 1.

In each iteration, our measurement points include a mini-batch D s from the training data and random points X M from a distribution c. We forward X Ds and X M together through the network g(·; φ) which defines the variational posterior q φ .

Then we try to maximize the following objective corresponding to fELBO: DISPLAYFORM3 Here λ is a regularization hyperparameter.

In principle, λ should be set as 1 |D| to match fELBO in Equation (5).

However, because the KL in Equation FORMULA2 uses a restricted number of measurement points, it only terms a lower bound of the functional KL divergence KL[q(f ) p(f )], thus bigger λ is favored to control overfitting.

We used λ = 1 |Ds| in practice, in which case Equation FORMULA2 is a proper lower bound of log p(D s ), as shown in Theorem 2.

Moreover, when using GP priors, we injected Gaussian noise on the function outputs for stability consideration (see Appendix D.1 for details).

Bayesian neural networks.

Variational inference was first applied to neural networks by BID42 and BID22 .

More recently, BID15 proposed a practical method for variational inference with fully factorized Gaussian posteriors which used a simple (but biased) gradient estimator.

Improving on that work, BID4 proposed an unbiased gradient estimator using the reparameterization trick of BID25 .

There has also been much work BID34 BID55 BID2 on modelling the correlations between weights using more complex Gaussian variational posteriors.

Some non-Gaussian variational posteriors have been proposed, such as multiplicative normalizing flows and implicit distributions BID51 .

Neural networks with dropout were also interpreted as BNNs BID10 BID11 .

Local reparameterization trick BID26 and Flipout BID60 try to decorrelate the gradients within a mini-batch for reducing variances during training.

However, all these methods place priors over the network parameters.

Often, spherical Gaussian priors are placed over the weights for convenience.

Other priors, including log-uniform priors BID26 and horseshoe priors BID13 , were proposed for specific purposes such as model compression and model selection.

But the relationships of weight-space priors to the functions computed by networks are difficult to characterize.

Functional Priors.

There have been other recent attempts to train BNNs in the spirit of functional priors.

BID8 trained a BNN prior to mimic a GP prior, but they still required variational inference in weight space.

Noise Contrastive Priors BID17 are somewhat similar in spirit to our work in that they use a random noise prior in the function space.

The prior is incorporated by adding a regularization term to the weight-space ELBO, and is not rich enough to encourage extrapolation and pattern discovery.

Neural Processes (NP) BID12 try to model any conditional distribution given arbitrary data points, whose prior is specified implicitly by prior samples.

However, in high dimensional spaces, conditional distributions become increasingly The green and blue lines represent ground truth and mean prediction, respectively.

Shaded areas correspond to standard deviations.

We considered GP priors with two kernels: RBF (which does not model the periodic structure), and PER + RBF (which does).

In each case, the fBNN makes similar predictions to the exact GP.

In contrast, the standard BBB (BBB-1) cannot even fit the training data, while BBB with scaling down KL by 0.001 (BBB-0.001) manages to fit training data, but fails to provide sensible extrapolations.complicated to model.

Variational Implicit Processes (VIP) BID37 are, in a sense, the reverse of fBNNs: they specify BNN priors and use GPs to approximate the posterior.

But the use of BNN priors means they can't exploit richly structured GP priors or other stochastic processes.

Scalable Gaussian Processes.

Gaussian processes are difficult to apply exactly to large datasets since the computational requirements scale as O(N 3 ) time, and as O(N 2 ) memory, where N is the number of training cases.

Multiple approaches have been proposed to reduce the computational complexity.

However, sparse GP methods BID29 BID53 BID58 BID18 BID27 still suffer for very large dataset, while random feature methods BID43 BID30 and KISS-GP BID62 BID24 ) must be hand-tailored to a given kernel.

Our experiments had two main aims: (1) to test the ability of fBNNs to extrapolate using various structural motifs, including both implicit and explicit priors, and (2) to test if they perform competitively with other BNNs on standard benchmark tasks such as regression and contextual bandits.

In all of our experiments, the variational posterior is represented as a stochastic neural network with independent Gaussian distributions over the weights, i.e. q(w) = N (w; µ, diag(σ 2 )).

3 We always used the ReLU activation function unless otherwise specified.

Measurement points were sampled uniformly from a rectangle containing the training inputs.

More precisely, each coordinate was sampled from the interval DISPLAYFORM0 , where x min and x max are the minimum and maximum input values along that coordinate, and d = x max − x min .

For experiments where we used GP priors, we first fit the GP hyperparameters to maximize the marginal likelihood on subsets of the training examples, and then fixed those hyperparameters to obtain the prior for the fBNNs.

Making sensible predictions outside the range of the observed data requires exploiting the underlying structure.

In this section, we consider some illustrative examples where fBNNs are able to use structured priors to make sensible extrapolations.

Appendix C.2 also shows the extrapolation of fBNNs for a time-series problem.

Gaussian processes can model periodic structure using a periodic kernel plus a RBF kernel: DISPLAYFORM0 where p is the period.

In this experiment, we consider 20 inputs randomly sampled from the interval [−2, −0.5] ∪ [0.5, 2], and targets y which are noisy observations of a periodic function: y = 2 * sin(4x) + with ∼ N (0, 0.04).

We compared our method with Bayes By Backprop (BBB) BID4 (with a spherical Gaussian prior on w) and Gaussian Processes.

For fBNNs and GPs, we considered both a single RBF kernel (which does not capture the periodic structure) and PER + RBF as in eq. (13) (which does).

As shown in FIG1 , BBB failed to fit the training data, let alone recover the periodic pattern (since its prior does not encode any periodic structure).

For this example, we view the GP with PER + RBF as the gold standard, since its kernel structure is designed to model periodic functions.

Reassuringly, the fBNNs made very similar predictions to the GPs with the corresponding kernels, though they predicted slightly smaller uncertainty.

We emphasize that the extrapolation results from the functional prior, rather than the network architecture, which does not encode periodicity, and which is not well suited to model smooth functions due to the ReLU activation function.

Because the KL term in the fELBO is estimated using the SSGE, an implicit variational inference algorithm (as discussed in Section 2.3), the functional prior need not have a tractable marginal density.

In this section, we examine approximate posterior samples and marginals for two implicit priors: a distribution over piecewise constant functions, and a distribution over piecewise linear functions.

Prior samples are shown in FIG2 ; see Appendix D.2 for the precise definitions.

In each run of the experiment, we first sampled a random function from the prior, and then sampled 20 points from [0, 0.2] and another 20 points from [0.8, 1], giving a training set of 40 data points.

To make the task more difficult for the fBNN, we used the tanh activation function, which is not well suited for piecewise constant or piecewise linear functions.

Posterior predictive samples and marginals are shown for three different runs in FIG2 .

We observe that fBNNs made predictions with roughly piecewise constant or piecewise linear structure, although their posterior samples did not seem to capture the full diversity of possible explanations of the data.

Even though the tanh activation function encourages smoothness, the network learned to generate functions with sharp transitions.

Following previous work (Hernández-Lobato & Adams, 2015), we then experimented with standard regression benchmark datasets from the UCI collection BID1 .

In particular, we only used the datasets with less than 2000 data points so that we could fit GP hyperparameters by 4 Details: we used a BNN with five hidden layers, each with 500 units.

The inputs and targets were normalized to have zero mean and unit variance.

For all methods, the observation noise variance was set to the true value.

We used the trained GP as the prior of our fBNNs.

In each iteration, measurement points included all training examples, plus 40 points randomly sampled from [−5, 5] .

We used a training budget of 80,000 iterations, and annealed the weighting factor of the KL term linearly from 0 to 1 for the first 50,000 iterations.5 Details: the standard deviation of observation noise was chosen to be 0.02.

In each iteration, we took all training examples, together with 40 points randomly sampled from [0, 1]].

We used a fully connected network with 2 hidden layers of 100 units, and tanh activations.

The network was trained for 20,000 iterations.

maximizing marginal likelihood exactly.

Each dataset was randomly split into training and test sets, comprising 90% and 10% of the data respectively.

This splitting process was repeated 10 times to reduce variability.

6 We compared our fBNNs with Bayes By Backprop (BBB) BID4 and Noisy K-FAC .

In accordance with Zhang et al. FORMULA2 , we report root mean square error (RMSE) and test log-likelihood.

The results are shown in TAB0 .

On most datasets, our fBNNs outperformed both BBB and NNG, sometimes by a significant margin.

Observe that fBNNs are naturally scalable to large datasets because they access the data only through the expected log-likelihood term, which can be estimated stochastically.

In this section, we verify this experimentally.

We compared fBNNs and BBB with large scale UCI datasets, including Naval, Protein Structures, Video Transcoding (Memory, Time) and GPU kernel performance.

We randomly split the datasets into 80% training, 10% validation, and 10% test.

We used the validating set to select the hyperparameters and performed early stopping.

Both methods were trained for 80,000 iterations.

7 We used 1 hidden layer with 100 hidden units for all datasets.

For the prior of fBNNs, we used a GP with Neural Kernel Network (NKN) kernels as used in .

We note that GP hyperparameters were fit using mini-batches of size 1000 with 10000 iterations.

In each iteration, measurement sets consist of 500 training samples and 5 or 50 points from the sampling distribution c, tuned by validation performance.

We ran each experiment 5 times, and report the mean and standard deviation in TAB1 .

More large scale regression results with bigger networks can be found at Appendix C.4 and Appendix C.5.

One of the most important applications of uncertainty modelling is to guide exploration in settings such as bandits, Bayesian optimization (BO), and reinforcement learning.

In this section, we evaluate fBNNs on a recently introduced contextual bandits benchmark BID46 .

In contextual bandits problems, the agent tries to select the action with highest reward given some input context.

Because the agent learns about the model gradually, it should balance between exploration 6 Details:

For all datasets, we used networks with one hidden layer of 50 hidden units.

We first fit GP hyper-parameters using marginal likelihood with a budget of 10,000 iterations.

We then trained the observation variance and kept it lower bounded by GP observation variance.

FBNNs were trained for 2,000 epochs.

And in each iteration, measurement points included 20 training examples, plus 5 points randomly sampled.

7 We tune the learning rate from [0.001, 0.01].

We tuned between not annealing the learning rate or annealing it by 0.1 at 40000 iterations.

We evaluated the validating set in each epoch, and selected the epoch for testing based on the validation performance.

To control overfitting, we used Gamma(6., 6.) prior following (Hernández-Lobato & Adams, 2015) for modelling observation precision and perform inference.

We compared our fBNNs with the algorithms benchmarked in BID46 .

We ran the experiments for all algorithms and tasks using the default settings open sourced by BID46 .

For fBNNs, we kept the same settings, including batchsize (512), training epochs (100) and training frequency (50).

For the prior, we use the multi-task GP of BID46 .

Measurement sets consisted of training batches, combined with 10 points sampled from data regions.

We ran each experiment 10 times; the mean and standard derivation are reported in TAB2 (Appendix C.1 has the full results for all experiments.).

Similarly to BID46 , we also report the mean rank and mean regret.

As shown in TAB2 , fBNNs outperformed other methods by a wide margin.

Additionally, fBNNs maintained consistent performance even with deeper and wider networks.

By comparison, BBB suffered significant performance degradation when the hidden size was increased from 50 to 500.

This is consistent with our hypothesis that functional variational inference can gracefully handle networks with high capacity.

Another domain where efficient exploration requires accurate uncertainty modeling is Bayesian optimization.

Our experiments with Bayesian optimization are described in App C.3.

We compared BBB, RBF Random Feature BID43 and our fBNNs in the context of Max-value Entropy Search (MES) BID59 , which requires explicit function samples for Bayesian Optimization.

We performed BO over functions sampled from Gaussian Processes corresponding to RBF, Matern12 and ArcCosine kernels, and found our fBNNs achieved comparable or better performance than RBF Random Feature.

In this paper we investigated variational inference between stochastic processes.

We proved that the KL divergence between stochastic processes equals the supremum of KL divergence for marginal distributions over all finite measurement sets.

Then we presented two practical functional variational inference approaches: adversarial and sampling-based.

Adopting BNNs as the variational posterior yields our functional variational Bayesian neural networks.

Empirically, we demonstrated that fBNNs extrapolate well over various structures, estimate reliable uncertainties, and scale to large datasets.

We begin with some basic terminology and classical results.

See BID16 BID9 for more details.

Definition 1 (KL divergence).

Given a probability measure space (Ω, F, P ) and another probability measure M on the smae space, the KL divergence of P with respect to M is defined as DISPLAYFORM0 where the supremum is taken over all finite measurable partitions DISPLAYFORM1 of Ω, and P Q , M Q represent the discrete measures over the partition Q, respectively.

Definition 2 (Pushforward measure).

Given probability spaces (X, F X , µ) and (Y, F Y , ν), we say that measure ν is a pushforward of µ if ν(A) = µ(f −1 (A)) for a measurable f : X → Y and any A ∈ F Y .

This relationship is denoted by ν = µ • f −1 .Definition 3 (Canonical projection map).

Let T be an arbitrary index set, and {(Ω t , F t )} t∈T be some collection of measurable spaces.

For each subset J ⊂ I ⊂ T , define Ω J = t∈J Ω t .

We call π I→J the canonical projection map from I to J if DISPLAYFORM2 Where w| J is defined as, if w = (w i ) i∈I , then w| J = (w i ) i∈J .

Definition 4 (Cylindrical σ-algebra).

Let T be an arbitrary index set, (Ω, F) be a measurable space.

Suppose DISPLAYFORM3 is the set of Ω-valued functions.

A cylinder subset is a finitely restricted set defined as DISPLAYFORM4 We call the σ-algebra F T := σ(G Ω T ) as the cylindrical σ-algebra of Ω T , and (Ω T , F T ) the cylindrical measurable space.

The Kolmogorov Extension Theorem is the foundational result used to construct many stochastic processes, such as Gaussian processes.

A particularly relevant fact for our purposes is that this theorem defines a measure on a cylindrical measurable space, using only canonical projection measures on finite sets of points.

Theorem 4 (Kolmogorov extension theorem (Øksendal, 2003) ).

Let T be an arbitrary index set.(Ω, F) is a standard measurable space, whose cylindrical measurable space on T is (Ω T , F T ).

Suppose that for each finite subset I ⊂ T , we have a probability measure µ I on Ω I , and these measures satisfy the following compatibility relationship: for each subset J ⊂ I, we have DISPLAYFORM5 Then there exists a unique probability measure µ on Ω T such that for all finite subsets I ⊂ T , DISPLAYFORM6 In the context of Gaussian processes, µ is a Gaussian measure on a separable Banach space, and the µ I are marginal Gaussian measures at finite sets of input positions BID38 .

Theorem 5.

Suppose that M and P are measures on the sequence space corresponding to outcomes of a sequence of random variables X 0 , X 1 , · · · with alphabet A. Let F n = σ(X 0 , · · · , X n−1 ), which asymptotically generates the σ-algebra σ(X 0 , X 1 , · · · ).

Then DISPLAYFORM7 Where P Fn , M Fn denote the pushforward measures with f : DISPLAYFORM8 Consider the canonical projection mapping π T →Tc , which induces a partition on Ω Tc , denoted by Q Ω Tc : DISPLAYFORM9 The pushforward measure defined by this mapping is DISPLAYFORM10 Step 2.

Then we have DISPLAYFORM11 = sup DISPLAYFORM12 Step 3.

Denote D(T c ) as the collection of all finite subsets of T c .

For any finite set T d ∈ D(T c ), we denote P T d as the pushforward measure of P Tc on Ω T d .

From the Kolmogorov Extension Theorem (Theorem 4), we know that P T d corresponds to the finite marginals of P at Ω T d .

Because T c is countable, based on Theorem 5, we have, DISPLAYFORM13 We are left with the last question: whether each T d is contained in some D(T c ) ?

For any finite indices set T d , we build a finite measureable partition Q. DISPLAYFORM14 2 K to be all K-length binary vectors.

We define the partition, DISPLAYFORM15 DISPLAYFORM16 Through this settting, Q is a finite parition of Ω T , and T c (Q) = T d .

Therefore T d in Equation FORMULA2 can range over all finite index sets, and we have proven the theorem.

DISPLAYFORM17 A.3 KL DIVERGENCE BETWEEN CONDITIONAL STOCHASTIC PROCESSESIn this section, we give an example of computing the KL divergence between two conditional stochastic processes.

Consider two datasets D 1 , D 2 , the KL divergence between two conditional stochastic processes is DISPLAYFORM18 Therefore, the KL divergence between these two stochastic processes equals to the marginal KL divergence on the observed locations.

When D 2 = ∅, p(f |D 2 ) = p(f ), this shows the KL divergence between posterior process and prior process are the marginal KL divergence on observed locations.

This also justifies our usage of M measurement points in the adversarial functional VI and samplingbased functional VI of Section 3.

This section provides proof for Theorem 2.Proof of Theorem 2.

Let X M = X\X D be measurement points which aren't in the training data.

Here DISPLAYFORM0 DISPLAYFORM1 Remember that in sampling-based functional variational inference, X M are randomly sampled from c(x), and supp(c) = X .

Thus when it reaches optimum, we have DISPLAYFORM2 For adversarial functional variational inference, this is also obvious due to sup DISPLAYFORM3 So we have that Equation FORMULA4 holds for any DISPLAYFORM4 Because GPs are uniquely determined by their mean and covariance functions, we arrive at the conclusion.

C ADDITIONAL EXPERIMENTS

Here we present the full table for the contextual bandits experiment.

Besides the toy experiments, we would like to examine the extrapolation behavior of our method on real-world datasets.

Here we consider a classic time-series prediction problem concerning the concentration of CO 2 in the atmosphere at the Mauna Loa Observatory, Hawaii BID45 .

The training data is given from 1958 to 2003 (with some missing values).

Our goal is to model the prediction for an equally long period after 2003 (2004-2048) .

In FIG4 we draw the prediction results given by BBB, fBNN, and GP.

We used the same BNN architecture for BBB and fBNN: a ReLU network with 2 hidden layers, each with 100 units, and the input is a normalized year number augmented by its sin transformation, whose period is set to be one year.

This special design allows both BBB and fBNN to fit the periodic structure more easily.

Both models are trained for 30k iterations by the Adam optimizer, with learning rate 0.01 and batch size 20.

For fBNN the prior is the same as the GP experiment, whose kernel is a combination of RBF, RBF×PER (period set to one year), and RQ kernels, as suggested in BID45 .

Measurement points include 20 training samples and 10 points sampled from U [1958, 2048] , and we jointly train the prior GP hyperparameters with fBNN.In FIG4 we could see that the performance of fBNN closely matches the exact prediction by GP.

Both of them give visually good extrapolation results that successfully model the long-term trend, local variations, and periodic structures.

In contrast, weight-space prior and inference (BBB) neither captures the right periodic structure, nor does it give meaningful uncertainty estimates.

DISPLAYFORM0 Figure 5: Bayesian Optimization.

We plot the minimal value found along iterations.

We compare fBNN, BBB and Random Feature methods for three kinds of functions corresponding to RBF, Order-1 ArcCosine and Matern12 GP kernels.

We plot mean and 0.2 standard derivation over 10 independent runs.

In this section, we adopt Bayesian Optimization to explore the advantage of coherent posteriors.

Specifically, we use Max Value Entropy Search (MES) BID59 , which tries to maximize the information gain about the minimum value y , DISPLAYFORM1 Where φ and Ψ are probability density function and cumulative density function of a standard normal distribution, respectively.

The y is the minimum of a random function from the posterior, and γ y (x) = µt(x)−y σt(x) .

With a probabilistic model, we can compute or estimate the mean µ t (x) and the standard deviation σ t (x).

However, to compute the MES acquisition function, samples y of function minima are required as well, which leads to difficulties.

Typically when we model the data with a GP, we can get the posterior on a specific set of points but we don't have access to the extremes of the underlying function.

In comparison, if the function posterior is represented in a parametric form, we can perform gradient decent easily and search for the minima.

We use 3-dim functions sampled from some Gaussian process prior for Bayesian optimization.

Concretely, we experiment with samples from RBF, Order-1 ArcCosine and Matern12 kernels.

We compare three parametric approaches: fBNN, BBB and Random Feature BID43 .

For fBNN, we use the true kernel as functional priors.

In contrast, ArcCosine and Matern12 kernels do not have simple explicit random feature expressions, therefore we use RBF random features for all three kernels.

When looking for minima, we sample 10 y .

For each y , we perform gradient descent along the sampled parametric function posterior with 30 different starting points.

We use 500 dimensions for random feature.

We use network with 5 × 100 for fBNN.

For BBB, we select the network within 1 × 100, 3 × 100.

Because of the similar issue in FIG0 , using larger networks won't help for BBB.

We use batch size 30 for both fBNN and BBB.

The measurement points contain 30 training points and 30 points uniformly sampled from the known input domain of functions.

For training, we rescale the inputs to [0, 1], and we normalize outputs to have zero mean and unit variance.

We train fBNN and BBB for 20000 iterations and anneal the coefficient of log likelihood term linearly from 0 to 1 for the first 10000 iterations.

The results with 10 runs are shown in Figure 5 .As seen from Figure 5 , fBNN and Random feature outperform BBB by a large margin on all three functions.

We also observe fBNN performs slightly worse than random feature in terms of RBF priors.

Because random feature method is exactly a GP with RBF kernel asymptotically, it sets a high standard for the parametric approaches.

In contrast, fBNN outperforms random feature for both ArcCosine and Matern12 functions.

This is because of the big discrepancy between such kernels and RBF random features.

Because fBNN use true kernels, it models the function structures better.

This experiment highlights a key advantage of fBNN, that fBNN can learn parametric function posteriors for various priors.

To compare with Variational Free Energy (VFE) BID58 ), we experimented with two mediumsize datasets so that we can afford to use VFE with full batch.

For VFE, we used 1000 inducing points initialized by k-means of training point.

For BBB and FBNNs, we used batch size 500 with a budget of 2000 epochs.

As shown in TAB4 , FBNNs performed slightly worse than VFE, but the gap became smaller as we used larger networks.

By contrast, BBB totally failed with large networks (5 hidden layers with 500 hidden units each layer).

Finally, we note that the gap between FBNNs and VFE diminishes if we use fewer inducing points (e.g., 300 inducing points).

In this section we experimented on large scale regression datasets with deeper networks.

For BBB and fBNNs, we used a network with 5 hidden layers of 100 units, and kept all other settings the same as Section 5.2.2.

We also compared with the stochastic variational Gaussian processes (SVGP) BID18 , which provides a principled mini-batch training for sparse GP methods, thus enabling GP to scale up to large scale datasets.

For SVGP, we used 1000 inducing points initialized by k-means of training points (Note we cannot afford larger size of inducing points because of the cubic computational cost).

We used batch size 2000 and iterations 60000 to match the training time with fBNNs.

Likewise for BNNs, we used validation set to tune the learning rate from {0.01, 0.001}. We also tuned between not annealing the learning rate or annealing it by 0.1 at 30000 iterations.

We evaluated the validating set in each epoch, and selected the epoch for testing based on the validation performance.

The averaged results over 5 runs are shown in TAB5 .

As shown in TAB5 , SVGP performs better than BBB and fBNNs in terms of the smallest naval dataset.

However, with dataset size increasing, SVGP performs worse than BBB and fBNNs by a large margin.

This stems from the limited capacity of 1000 inducing points, which fails to act as sufficient statistics for large datasets.

In contrast, BNNs including BBB and fBNNs can use larger networks freely without the intractable computational cost.

For Gaussian process priors, p(f X ) is a multivariate Gaussian distribution, which has an explicit density.

Therefore, we can compute the gradients ∇ f log p φ (f X ) analytically.

In practice, we found that the GP kernel matrix suffers from stability issues.

To stabilize the gradient computation, we propose to inject a small amount of Gaussian noise on the function values, i.e., to instead estimate the gradients of ∇ φ KL[q φ * p γ p * p γ ], where p γ = N (0, γ 2 ) is the noise distribution.

This is like the instance-noise trick that is commonly used for stabilizing GAN training BID54 .

Note that injecting the noise on the GP prior is equivalent to have a kernel matrix K + γ 2 I. Beyond that, injecting the noise on the parametric variational posterior does not affect the reparameterization trick either.

Therefore all the previous estimation formulas still apply.

Our method is applicable to implicit priors.

We experiment with piecewise constant prior and piecewise linear prior.

Concretely, we randomly generate a function f : [0, 1] → R with the specific structure.

To sample piecewise functions, we first sample n ∼ Poisson(3.), then we have n + 1 pieces within [0, 1].

We uniformly sample n locations from [0, 1] as the changing points.

For piecewise constant functions, we uniformly sample n + 1 values from [0, 1] as the function values in each piece; For piecewise linear functions, we uniformly sample n + 1 values for the values at first n + 1 locations, we force f (1) = 0..

Then we connect together each piece by a straight line.

<|TLDR|>

@highlight

We perform functional variational inference on the stochastic processes defined by Bayesian neural networks.

@highlight

Fitting of variational Bayesian Neural Network approximations in functional form and considering matching to a stochastic process prior implicitly via samples.

@highlight

Presents a novel ELBO objective for training BNNs which allows for more meaningful priors to be encoded in the model rather than the less informative weight priors features in the literature.

@highlight

Presents a new variational inference algorithm for Bayesian neural network models where the prior is specified functionally rather than via a prior over weights. 