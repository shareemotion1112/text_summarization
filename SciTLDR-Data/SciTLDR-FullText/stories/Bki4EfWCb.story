Amortized inference has led to efficient approximate inference for large datasets.

The quality of posterior inference is largely determined by two factors: a) the ability of the variational distribution to model the true posterior and b) the capacity of the recognition network to generalize inference over all datapoints.

We analyze approximate inference in variational autoencoders in terms of these factors.

We find that suboptimal inference is often due to amortizing inference rather than the limited complexity of the approximating distribution.

We show that this is due partly to the generator learning to accommodate the choice of approximation.

Furthermore, we show that the parameters used to increase the expressiveness of the approximation play a role in generalizing inference rather than simply improving the complexity of the approximation.

There has been significant work on improving inference in variational autoencoders (VAEs) BID13 BID22 through the development of expressive approximate posteriors BID21 BID14 BID20 BID27 .

These works have shown that with more expressive approximate posteriors, the model learns a better distribution over the data.

In this paper, we analyze inference suboptimality in VAEs: the mismatch between the true and approximate posterior.

In other words, we are interested in understanding what factors cause the gap between the marginal log-likelihood and the evidence lower bound (ELBO).

We refer to this as the inference gap.

Moreover, we break down the inference gap into two components: the approximation gap and the amortization gap.

The approximation gap comes from the inability of the approximate distribution family to exactly match the true posterior.

The amortization gap refers to the difference caused by amortizing the variational parameters over the entire training set, instead of optimizing for each datapoint independently.

We refer the reader to Table 1 for detailed definitions and FIG0 for a simple illustration of the gaps.

In FIG0 , L[q] refers to the ELBO using an amortized distribution q, whereas q * is the optimal q within its variational family.

Our experiments investigate how the choice of encoder, posterior approximation, decoder, and model optimization affect the approximation and amortization gaps.

We train VAE models in a number of settings on the MNIST, Fashion-MNIST BID30 , and CIFAR-10 datasets.

Our contributions are: a) we investigate inference suboptimality in terms of the approximation and amortization gaps, providing insight to guide future improvements in VAE inference, b) we quantitatively demonstrate that the learned true posterior accommodates the choice of approximation, and c) we demonstrate that using parameterized functions to improve the expressiveness of the approximation plays a large role in reducing error caused by amortization.

Table 1 : Summary of Gap Terms.

The middle column refers to the general case where our variational objective is a lower bound on the marginal log-likelihood (not necessarily the ELBO).

The right most column demonstrates the specific case in VAEs.

q * (z|x) refers to the optimal approximation within a family Q, i.e. q * (z|x) = arg min q∈Q KL (q(z|x)||p(z|x)).

Let x be the observed variable, z the latent variable, and p(x, z) be their joint distribution.

Given a dataset X = {x 1 , x 2 , ..., x N }, we would like to maximize the marginal log-likelihood: DISPLAYFORM0 In practice, the marginal log-likelihood is computationally intractable due to the integration over the latent variable z. Instead, VAEs optimize the ELBO of the marginal log-likelihood BID13 BID22 : DISPLAYFORM1 ≥ E z∼q(z|x) log p(x, z) q(z|x) = L VAE [q] .From the above we can see that the lower bound is tight if q(z|x) = p(z|x).

The choice of q(z|x) is often a factorized Gaussian distribution for its simplicity and efficiency.

VAEs perform amortized inference by utilizing a recognition network (encoder), resulting in efficient approximate inference for large datasets.

The overall model is trained by stochastically optimizing the ELBO using the reparametrization trick BID13 .

There are a number of strategies for increasing the expressiveness of approximate posteriors, going beyond the original factorized-Gaussian.

We briefly summarize normalizing flows and auxiliary variables.

Normalizing flow BID21 ) is a change of variables procedure for constructing complex distributions by transforming probability densities through a series of invertible mappings.

Specifically, if we transform a random variable z 0 with distribution q 0 (z), the resulting random variable z T = T (z 0 ) has a distribution: DISPLAYFORM0 By successively applying these transformations, we can build arbitrarily complex distributions.

Stacking these transformations remains tractable due to the determinant being decomposable: det(AB) = det(A)det(B).

An important property of these transformations is that we can take expectations with respect to the transformed density q T (z T ) without explicitly knowing its formula known as the law of the unconscious statistician (LOTUS): DISPLAYFORM1 Using the change of variable and LOTUS, the lower bound can be written as: DISPLAYFORM2 The main constraint on these transformations is that the determinant of their Jacobian needs to be easily computable.

Deep generative models can be extended with auxiliary variables which leave the generative model unchanged but make the variational distribution more expressive.

Just as hierarchical Bayesian models induce dependencies between data, hierarchical variational models can induce dependencies between latent variables.

The addition of the auxiliary variable changes the lower bound to: DISPLAYFORM0 where r(v|x, z) is called the reverse model.

From Eqn.

8, we see that this bound is looser than the regular ELBO, however the extra flexibility provided by the auxiliary variable can result in a higher lower bound.

This idea has been employed in works such as auxiliary deep generative models (ADGM, ), hierarchical variational models (HVM, BID20 ) and Hamiltonian variational inference (HVI, BID25 ).

We use two bounds to estimate the marginal log-likelihood of a model: IWAE BID2 and AIS BID19 ).

Here we describe the IWAE bound.

See Section 6.5 in the appendix for a description of AIS.The IWAE bound is a tighter lower bound than the VAE bound.

More specifically, if we take multiple samples from the q distribution, we can compute a tighter lower bound on the marginal log-likelihood: DISPLAYFORM0 As the number of importance samples approaches infinity, the bound approaches the marginal loglikelihood.

This importance weighted bound was introduced along with the Importance Weighted Autoencoder BID2 , thus we refer to it as the IWAE bound.

It is often used as an evaluation metric for generative models BID2 BID14 .

As shown by BID0 and BID4 , the IWAE bound can be seen as using the VAE bound but with an importance weighted q distribution.

The inference gap G is the difference between the marginal log-likelihood log p(x) and a lower bound L [q] .

Given the distribution in the family that maximizes the bound, q * (z|x) = arg max q∈Q L[q], the inference gap decomposes as the sum of approximation and amortization gaps: DISPLAYFORM0 For VAEs, we can translate the gaps to KL divergences by rearranging (2): DISPLAYFORM1 3.2 FLEXIBLE APPROXIMATE POSTERIOR Our experimentation compares two families of approximate posteriors: the fully-factorized Gaussian (FFG) and a flexible flow (Flow).

Our choice of flow is a combination of the Real NVP BID5 and auxiliary variables BID20 .

Our model also resembles leap-frog dynamics applied in Hamiltonian Monte Carlo (HMC, BID18 ).Let z ∈ R n be the variable of interest and v ∈ R n the auxiliary variable.

Each flow step involves: DISPLAYFORM2 where σ 1 , σ 2 , µ 1 , µ 2 : R n → R n are differentiable mappings parameterized by neural nets and • takes the Hadamard or element-wise product.

The determinant of the combined transformation's Jacobian, |det(Df )|, can be easily evaluated.

See section 6.2 in the Appendix for a detailed derivation.

Thus, we can jointly train the generative and flow-based inference model by optimizing the bound: DISPLAYFORM3 Additionally, multiple such type of transformations can be stacked to improve expressiveness.

We refer readers to section 6.1.2 in the Appendix for details of our flow configuration adopted in the experimentation.

We use several bounds to compute the inference gaps.

To estimate the marginal log-likelihood, logp(x), we take the maximum of our tightest lower bounds, specifically the maximum between the IWAE and AIS bounds.

To compute the AIS bound, we use 100 chains, each with 500 intermediate distributions, where each transition consists of one HMC trajectory with 10 leapfrog steps.

The initial distribution for AIS is the prior, so that it is encoder-independent.

For our experiments, we test two different variational distributions: the fully-factorized Gaussian q F F G and the flexible approximation q F low as described in section 3.2.

When computing DISPLAYFORM0 , we use 5000 samples.

To compute L VAE [q * ], we optimize the parameters of the variational distribution for every datapoint.

See Section 6.4 for details of the local optimization and stopping criteria.

Much of the earlier work on variational inference focused on optimizing the variational parameters locally for each datapoint, e.g. the original Stochastic Variational Inference scheme (SVI, Hoffman et al. (2013) ) specifies the variational parameters to be optimized locally in the inner loop.

BID24 perform such local optimization when learning deep Boltzmann machines.

More recent work has applied this idea to improve approximate inference in directed Belief networks BID9 .Most relevant to our work is the recent work of BID15 .

They explicitly remark on two sources of error in variational learning with inference networks, and propose to optimize approximate inference locally from an initialization output by the inference network.

They show improved training on high-dimensional, sparse data with the hybrid method, claiming that local optimization reduces the negative effects of random initialization in the inference network early on in training.

Yet, their work only dwells on reducing the amortization gap and does analyze the error arising from the use of limited approximating distributions.

Even though it is clear that failed inference would lead to a failed generative model, little quantitative assessment has been done showing the effect of the approximate posterior on the true posterior.

BID2 visually demonstrate that when trained with an importance-weighted approximate posterior, the resulting true posterior is more complex than those trained with fully-factorized Gaussian approximations.

We extend this observation quantitatively in the setting of flow-based approximate inference.

To begin, we would like to gain some insight into the properties of inference in VAEs by visualizing different distributions in the latent space.

To this end, we trained a VAE with a two-dimensional latent space on MNIST.

We show contour plots of various distributions in the latent space in FIG1 .

The first row contains contour plots of the true posteriors p(z|x) for four different training datapoints (columns).

We have selected these four examples to highlight different inference phenomena.

The amortized FFG row refers to the output of the recognition net, in this case, a fully-factorized Gaussian (FFG) approximation.

Optimal FFG is the FFG that best fits the posterior of the datapoint.

Optimal Flow is the optimal fit of a flexible distribution to the same posterior, where the flexible distribution we use is described in Section 3.2.Posterior A is an example of a distribution where FFG can fit well.

Posterior B is an example of dependence between dimensions, demonstrating the limitation of having a factorized approximation.

Posterior C highlights a shortcoming of performing amortization with a limited-capacity recognition network, where the amortized FFG shares little support with the true posterior.

Posterior D is a bimodal distribution which demonstrates the ability of the flexible approximation to fit to complex distributions, in contrast to the simple FFG approximation.

These observations raise the following question: in more typical VAEs, is the amortization of inference the leading cause of the distribution mismatch, or is it the choice of approximation?

Table 2 : Inference Gaps.

The columns q F F G and q F low refer to the variational distribution used for training the model.

All numbers are in nats.

DISPLAYFORM0

Here we will compare the influence that the approximation and amortization errors have on the total inference gap.

Table 2 are results from training on MNIST, Fashion-MNIST and CIFAR-10.For each dataset, we trained two different approximate posterior distributions: a fully-factorized Gaussian, q F F G , and a flexible distribution, q F low .

Due to the computational cost of optimizing the local parameters for each datapoint, our evaluation is performed on a subset of 1000 datapoints for MNIST and Fashion-MNIST and a subset of 100 datapoints for CIFAR-10.For MNIST, we see that the amortization and approximation gaps each account for nearly half of the inference gap.

On Fashion-MNIST, which is a more difficult dataset to model, the amortization gap becomes larger than the approximation gap.

Similarly for CIFAR-10, we see that the amortization gap is much more significant than the approximation gap.

Thus, for the three datasets and model architectures that we tested, the amortization gap seems to be the prominent cause of inference suboptimality, especially when the difficulty of the dataset increases.

This analysis indicates that improvements in inference will likely be a result of reducing amortization error, rather than approximation errors.

With these results in mind, would simply increasing the capacity of the encoder improve the amortization gap?

We examined this by training the MNIST and Fashion-MNIST models from above but with larger encoders.

See Section 6.1.2 for implementation details.

Table 3 are the results of this experiment.

Comparing to Table 2 , we see that for both datasets and both variational distributions, the inference gap decreases and the decrease is mainly due to a reduction in the amortization gap.

Table 3 : Larger Encoder.

The columns q F F G and q F low refer to the variational distribution used for training the model.

All numbers are in nats.

DISPLAYFORM0

The common reasoning for increasing the expressiveness of the approximate posterior is to minimize the difference between the true and approximate, i.e. reduce the approximation gap.

However, given that the expressive approximation is often accompanied by many additional parameters, we would like to know if it has an influence on the amortization error.

To investigate this, we trained a VAE in the same manner as Section 5.2.

After training, we kept the generator fixed and trained new encoders to fit to the fixed posterior.

Specifically, we trained a small encoder with a factorized Gaussian q distribution to obtain a large amortization gap.

We then trained a small encoder with a flow distribution.

See Section 6.2 for the details of the experiment.

The results are shown in TAB3 .

As expected, we observe that the small encoder has a very large amortization gap.

However, when we use q F low as the approximate distribution, we see the approximation gap decrease, but more importantly, there is a significant decrease in the amortization gap.

This indicates that the parameters used for increasing the complexity of the approximation also play a large role in diminishing the amortization error.

These results are expected given that the parameterization of the Flow distribution can be interpreted as an instance of the RevNet BID7 which has demonstrated that Real-NVP like transformations BID5 can model complex functions similar to typical MLPs.

Thus the flow transformations we employ should also be expected to increase the expressiveness while also increasing the capacity of the encoder.

The implication of this observation is that models which improve the flexibility of their variational approximation, and attribute their improved results to the increased expressiveness, may have actually been due to the reduction in amortization error.

We have seen that increasing the expressiveness of the approximation improves the marginal likelihood of the trained model, but to what amount does it alter the true posterior?

Will a factorized Gaussian approximation cause the true posterior to be more like a factorized Gaussian or is the true posterior mostly fixed?

Just as it is hard to evaluate a generative model by visually inspecting samples from the model, its hard to say how Gaussian the true posterior is by visual inspection.

We can quantitatively determine how close the posterior is to a fully factorized Gaussian (FFG) distribution by comparing the marginal log-likelihood estimate, logp(x), and the Optimal FFG bound, DISPLAYFORM0 In other words, we are estimating the KL divergence between the optimal Gaussian and the true posterior, KL (q * (z|x)||p(z|x)).In Table 2 on MNIST, the Optimal Flow improves upon the Optimal FFG for the FFG trained model by 0.4 nats.

In contrast, on the Flow trained model, the difference increases to 12.5 nats.

This suggests that the true posterior of a FFG-trained model is closer to FFG than the true posterior of the Flow-trained model.

The same observation can be made on the Fashion-MNIST dataset.

This implies that the decoder can learn to have a true posterior that fits better to the approximation.

Although the generative model can learn to have a posterior that fits to the approximation, it seems that not having this constraint, ie.

using a flexible approximate, results in better generative models.

We can use these observations to help justify our approximation and amortization gap results of Section 5.2.

Those results showed that the amortization error is often the main cause of inference suboptimality.

One reason for this is that the generator accommodates to the choice of approximation, as shown above, thus reducing the approximation error.

Given that we have seen that the generator could accommodate to the choice of approximation, our next question is whether a generator with more capacity can accommodate more.

To this end, we trained VAEs with decoders of different sizes and measured the approximation gaps.

Specifically, we trained decoders with 0, 2, and 4 hidden layers on MNIST.

See Table 5 for the results.

We see that as the capacity of the decoder increases, the approximation gap decreases.

This result implies that the more flexible the generator, the less flexible the approximate distribution needs to be.

Table 5 : Increased decoder capacity reduces approximation gap.

All numbers are in nats.

Typical warm-up BID1 refers to annealing KL (q(z|x)||p(z)) during training.

This can also be interpreted as performing maximum likelihood estimation (MLE) early on during training.

This optimization technique is known to help prevent the latent variable from degrading to the prior BID2 .

We employ a similar annealing scheme during training.

Rather than annealing the KL divergence, we anneal the entropy of the approximate distribution q: DISPLAYFORM0 where λ is annealed from 0 to 1 over training.

This can be interpreted as maximum a posteriori (MAP) in the initial phase.

Due to its similarity, we will also refer to this technique as warm-up.

We find that warm-up techniques, such as annealing the entropy, are important for allowing the true posterior to be more complex.

Table 6 are results from a model trained without the entropy annealing schedule.

Comparing these results to Table 2 , we observe that the difference between DISPLAYFORM1 F low ] is significantly smaller without entropy annealing.

This indicates that the true posterior is more Gaussian when entropy annealing is not used.

This suggests that, in addition to preventing the latent variable from degrading to the prior, entropy annealing allows the true posterior to better utilize the flexibility of the expressive approximation, resulting in a better trained model.

Table 6 : Models trained without entropy annealing.

The columns q F F G and q F low refer to the variational distribution used for training the model.

All numbers are in nats.

DISPLAYFORM2

In this paper, we investigated how encoder capacity, approximation choice, decoder capacity, and model optimization influence inference suboptimality in terms of the approximation and amortization gaps.

We found that the amortization gap is often the leading source of inference suboptimality and that the generator reduces the approximation gap by learning a true posterior that fits to the choice of approximate distribution.

We showed that the parameters used to increase the expressiveness of the approximation play a role in generalizing inference rather than simply improving the complexity of the approximation.

We confirmed that increasing the capacity of the encoder reduces the amortization error.

We also showed that optimization techniques, such as entropy annealing, help the generative model to better utilize the flexibility of the expressive variational distribution.

Computing these gaps can be useful for guiding improvements to inference in VAEs.

Future work includes evaluating other types of expressive approximations and more complex likelihood functions.

The VAE model of FIG1 uses a decoder p(x|z) with architecture: 2 − 100 − 784, and an encoder q(z|x) with architecture: 784 − 100 − 4.

We use tanh activations and a batch size of 50.

The model is trained for 3000 epochs with a learning rate of 10 −4 using the ADAM optimizer BID12 .

Both MNIST and Fashion-MNIST consist of a training and test set with 60k and 10k datapoints respectively, where each datapoint is a 28x28 grey-scale image.

We rescale the original images so that pixel values are within the range [0, 1].

For MNIST, We use the statically binarized version described by BID16 .

We also binarize Fashion-MINST statically.

For both datasets, we adopt the Bernoulli likelihood for the generator.

The VAE models for MNIST and Fashion-MNIST experiments have the same architecture given in table 7.

The flow configuration is given in table 8.

Generator Input ∈ R Table 7 : Neural net architecture for MNIST/Fashion-MNIST experiments.

In the large encoder setting, we change the number of hidden units for the inference network to be 500, instead of 200.

The warm-up models are trained with a linear schedule over the first 400 epochs according to Section 5.3.1.The activation function is chosen to be the exponential linear unit (ELU, BID3 ), as we observe improved performance compared to tanh.

We follow the same learning rate schedule and train for the same amount of epochs as described by BID2 .

All models are trained with the a batch-size of 100 with ADAM.

CIFAR-10 consists of a training and test dataset with 50k and 10k datapoints respectively, where each datapoint is a 32 × 32 color image.

We rescale individual pixel values to be in the range [0, 1].

We follow the discretized logistic likelihood model adopted by BID14 , where each input channel has its own scale learned by an MLP.

For the latent variable, we use a 32-dimensional factorized Gaussian for q(z|x) following BID14 .

For all neural networks, ELU is chosen to be the activation function.

The specific network architecture is shown in Table 9 .We adopt a gradually decreasing learning rate with an initialize value of 10 −3 .

Warm-up is applied with a linear schedule over the first 20 epochs.

All models are trained with a batch-size of 100 with ADAM.

Early-stopping is applied based on the performance on the held-out set.

For the model with expressive inference, we use four flow steps as opposed to only two in MNIST/Fashion-MNIST experiments.

The aim of this experiment is to show that the parameters used for increasing the expressiveness of the approximation also contribute to reducing the amortization error.

To show this, we train a VAE on MNIST, discard the encoder, then retrain two encoders on the fixed decoder: one with a factorized Gaussian distribution and the other with a parameterized 'flow' distribution.

We use fixed decoder so that the true posterior is constant for both encoders.

Next, we describe the encoders which were trained on the fixed trained decoder.

In order to highlight a large amortization gap, we employed a very small encoder architecture: D X − 2D Z .

This encoder has no hidden layers, which greatly impoverishes its ability and results in a large amortization gap.

We compare two approximate distributions q(z|x).

Firstly, we experiment with the typical fully factorized Gaussian (FFG).

The second is what we call a flow distribution.

Specifically, we use the transformations of BID5 .

We also include an auxiliary variable so we don't need to select how to divide the latent space for the transformations.

The approximate distribution over the latent z and auxiliary variable v factorizes as: q(z, v|x) = q(z|x)q(v).

The q(v) distribution is simply a N(0,1) distribution.

Since we're using a auxiliary variable, we also require the r(v|z) distribution which we parameterize as r(v|z): [D Z ] − 50 − 50 − 2D Z .

The flow transformation is the same as in Section 3.2, which we apply twice.

DISPLAYFORM0 FC.

100-ELU-FC.

100-ELU-FC. 50+50 FC.

100-ELU-FC.

100-ELU-FC. 50+50 DISPLAYFORM1 FC.

100-ELU-FC.

100-ELU-FC.

50 FC.

100-ELU-FC.

100-ELU-FC.

50 Table 9 : Network architecture for CIFAR-10 experiments.

For the generator, one of the MLPs immediately after the input layer of the generator outputs channel-wise scales for the discretized logistic likelihood model.

BN stands for batch-normalization.

The overall mapping f that performs (z, v) → (z , v ) is the composition of two sheer mappings f 1 and f 2 that respectively perform (z, v) → (z, v ) and (z, v ) → (z , v ).

Since the Jacobian of either one of the sheer mappings is diagonal, the determinant of the composed transformation's Jacobian Df can be easily computed: DISPLAYFORM0

For the local FFG optimization, we initialize the mean and variance as the prior, i.e. N (0, I).

We optimize the mean and variance using the Adam optimizer with a learning rate of 10 −3 .

To determine convergence, after every 100 optimization steps, we compute the average of the previous 100 ELBO values and compare it to the best achieved average.

If it does not improve for 10 consecutive iterations then the optimization is terminated.

For the Flow model, the same process is used to optimize all of its parameters.

All neural nets for the flow were initialized with a variant of the Xavier initilization BID6 .

We use 100 Monte Carlo samples to compute the ELBO to reduce variance.

Annealed importance sampling (AIS, Neal (2001); BID11 ) is a means of computing a lower bound to the marginal log-likelihood.

Similarly to the importance weighted bound, AIS must sample a proposal distribution f 1 (z) and compute the density of these samples, however, AIS then transforms the samples through a sequence of reversible transitions T t (z |z).

The transitions anneal the proposal distribution to the desired distribution f T (z).Specifically, AIS samples an initial state z 1 ∼ f 1 (z) and sets an initial weight w 1 = 1.

For the following annealing steps, z t is sampled from T t (z |z) and the weight is updated according to: DISPLAYFORM0 .This procedure produces weight w T such that E [w T ] = Z T /Z 1 , where Z T and Z 1 are the normalizing constants of f T (z) and f 1 (z) respectively.

This pertains to estimating the marginal likelihood when the target distribution is p(x, z) when we integrate with respect to z.

Typically, the intermediate distributions are simply defined to be geometric averages: DISPLAYFORM1 βt , where β t is monotonically increasing with β 1 = 0 and β T = 1.

When f 1 (z) = p(z) and f T (z) = p(x, z), the intermediate distributions are: DISPLAYFORM2 Model evaluation with AIS appears early on in the setting of deep belief networks BID23 .

AIS for decoder-based models was also used by BID29 .

They validated the accuracy of the approach with Bidirectional Monte Carlo (BDMC, BID8 ) and demonstrated the advantage of using AIS over the IWAE bound for evaluation when the inference network overfits to the training data.

How well is inference done in VAEs during training?

Are we close to doing the optimal or is there much room for improvement?

To answer this question, we quantitatively measure the inference gap: the gap between the true marginal log-likelihood and the lower bound.

This amounts to measuring how well inference is being done during training.

Since we cannot compute the exact marginal log-likelihood, we estimate it using the maximum of any of its lower bounds, described in 3.3.

Fig. 3a shows training curves for a FFG and Flow inference network as measured by the VAE, IWAE, and AIS bounds on the training and test set.

The inference gap on the training set with the FFG model is 3.01 nats, whereas the Flow model is 2.71 nats.

Accordingly, Fig. 3a shows that the training IWAE bound is slightly tighter for the Flow model compared to the FFG.

Due to this lower inference gap during training, the Flow model achieves a higher AIS bound on the test set than the FFG model.

To demonstrate that a very small inference gap can be achieved, even with a limited approximation such as a factorized Gaussian, we train the model on a small dataset.

In this experiment, our training set consists of 1000 datapoints randomly chosen from the original MNIST training set.

The training curves on this small datatset are show in Fig. 3b .

Even with a factorized Gaussian distribution, the inference gap is very small: the AIS and IWAE bounds are overlapping and the VAE is just slightly below.

Yet, the model is overfitting as seen by the decreasing test set bounds.

Figure 3 : Training curves for a FFG and a Flow inference model on MNIST.

AIS provides the tightest lower bound and is independent of encoder overfitting.

There is little difference between FFG and Flow models trained on the 1000 datapoints since inference is nearly equivalent.

We will begin by explaining how we separate encoder from decoder overfitting.

Decoder overfitting is the same as in the regular supervised learning scenario, where we compare the train and test error.

To measure decoder overfitting independently from encoder overfitting, we use the AIS bound since it is encoder-independent.

Thus we can observe decoder overfitting through the AIS test training curve.

In contrast, the encoder can only overfit in the sense that the recognition network becomes unsuitable for computing the marginal likelihood on the test set.

Thus, encoder overfitting is computed by: L AIS − L IW on the test set.

For the small dataset of Fig. 3b , it clear that there is significant encoder and decoder overfitting.

A model trained in this setting would benefit from regularization.

For Fig. 3a , the model is not overfit and would benefit from more training.

However, there is some encoder overfitting due to the gap between the AIS and IWAE bounds on the test set.

Comparing the FFG and Flow models, it appears that the Flow does not have a large effect on encoder or decoder overfitting.

The flexiblity of the Gaussian family with arbitrary covariance lies between that of FFG and Flow.

With covariance, the Gaussian distribution can model interactions between different latent dimensions.

Yet, compared to Flow, its expressiveness is limited due to its inability to model higher order interactions and its unimodal nature.

To apply the reparameterization trick, we perform the Cholesky decomposition on the covariance matrix: Σ = LL , where L is lower triangular.

A sample from N (µ, Σ) could be obtained by first sampling from a unit Gaussian ∼ N (0, I), then computing z = µ + L .To analyze the capability of the Gaussian family, we train several VAEs on MNIST and Fashion-MNIST with the approximate posterior q(z|x) being a Gaussian with full covariance.

To inspect how well inference is done, we perform the local optimizations described in Section 5.

Table 10 : Gaussian latents trained with full covariance.

We can see from table 10 that local optimization with FFG on a model trained with full covariance inference produces a bad lower bound.

This resonates with the argument that the approximation has a significant influence on the true posterior as described in section 5.3.Comparing to numbers in table 2, we can see that the full-covariance VAE trained on MNIST is nearly on par with that trained with Flow (-89.28 vs -88.94 ).

For Fashion-MNIST, the fullcovariance VAE even performs better by a large margin in terms of the estimated log-likelihood (-96.46 vs -97.41 ).

@highlight

We decompose the gap between the marginal log-likelihood and the evidence lower bound and study the effect of the approximate posterior on the true posterior distribution in VAEs.