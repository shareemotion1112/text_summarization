In this paper, we present a new generative model for learning latent embeddings.

Compared to the classical generative process, where each observed data point is generated from an individual latent variable, our approach assumes a global latent variable to generate the whole set of observed data points.

We then propose a learning objective that is derived as an approximation to a lower bound to the data log likelihood, leading to our algorithm, WiSE-ALE.

Compared to the standard ELBO objective, where the variational posterior for each data point is encouraged to match the prior distribution, the WiSE-ALE objective matches the averaged posterior, over all samples, with the prior, allowing the sample-wise posterior distributions to have a wider range of acceptable embedding mean and variance and leading to better reconstruction quality in the auto-encoding process.

Through various examples and comparison to other state-of-the-art VAE models, we demonstrate that WiSE-ALE has excellent information embedding properties, whilst still retaining the ability to learn a smooth, compact representation.

Unsupervised learning is a central task in machine learning.

Its objective can be informally described as learning a representation of some observed forms of information in a way that the representation summarizes the overall statistical regularities of the data BID0 .

Deep generative models are a popular choice for unsupervised learning, as they marry deep learning with probabilistic models to estimate a joint probability between high dimensional input variables x and unobserved latent variables z. Early successes of deep generative models came from Restricted Boltzmann Machines BID7 and Deep Boltzmann Machines BID15 , which aim to learn a compact representation of data.

However, the fully stochastic nature of the network requires layer-by-layer pre-training using MCMC-based sampling algorithms, resulting in heavy computation cost.

BID9 consider the objective of optimizing the parameters in an auto-encoder network by deriving an analytic solution to a variational lower bound of the log likelihood of the data, leading to the Auto-Encoding Variational Bayes (AEVB) algorithm.

They apply a reparameterization trick to maximally utilize deterministic mappings in the network, significantly simplifying the training procedure and reducing instability.

Furthermore, a regularization term naturally occurs in their model, allowing a prior p(z) to be placed over every sample embedding q(z|x).

As a result, the learned representation becomes compact and smooth; see e.g. FIG0 where we learn a 2D embedding of MNIST digits using 4 different methods and visualize the aggregate posterior distribution of 64 random samples in the learnt 2D embedding space.

However, because the choice of the prior is often uninformative, the smoothness constraint imposed by this regularization term can cause information loss between the input samples and the latent embeddings, as shown by the merging of individual embedding distributions in FIG0 (d) (especially in the outer areas away from zero code).

Extreme effects of such behaviours can be noticed from β-VAE BID6 , a derivative algorithm of AEVB which further increases the weighting on the regularizing term with the aim of learning an even smoother, disentangled representation of the data.

As shown in FIG0 (e), the individual embedding distributions are almost indistinguishable, leading to an overly severe information bottleneck which can cause high rates of distortion BID16 .

In contrast, perfect reconstruction can be achieved using WAE (Tolstikhin et al., 2017) , but the learnt embedding distributions appear to severely non-smooth ( FIG0 ), indicating a small amount of noise in the latent space would cause generation process to fail.

In this paper, we propose WiSE-ALE (a wide sample estimator), which imposes a prior on the bulk statistics of a mini-batch of latent embeddings.

Learning under our WiSE-ALE objective does not penalize individual embeddings lying away from the zero code, so long as the aggregate distribution (the average of all individual embedding distributions) does not violate the prior significantly.

Hence, our approach mitigates the distortion caused by the current form of the prior constraint in the AEVB objective.

Furthermore, the objective of our WiSE-ALE algorithm is derived by applying variational inference in a simple latent variable model (Section 2) and with further approximation, we derive an analytic form of the learning objective, resulting in efficient learning algorithm.

In general, the latent representation learned using our algorithm enjoys the following properties: 1) smoothness, as indicated in FIG0 , the probability density for each individual embedding distribution decays smoothly from the peak value; 2) compactness, as individual embeddings tend to occupy a maximal local area in the latent space with minimal gaps in between; and 3) separation, indicated by the narrow, but clear borders between neighbouring embedding distributions as opposed to the merging seen in AEVB.

In summary, our contributions are:• proposing a new latent variable model that uses a single global latent variable to generate the entire dataset,• deriving a variational lower bound to the data log likelihood in our latent variable model, which allows us to impose prior constraint on the bulk statistics of a mini-batch embedding distributions,• and deriving analytic approximations to the lower bound, leading to our efficient WiSE-ALE learning algorithm.

In the rest of the paper, we first review directed graphical models in Section 2.

We then derive our variational lower bound and its analytic approximations in Section 3.

Related work is discussed in Section 4.

Experiment results are analyzed in Section 5, leading to conclusions in Section 6.

Here we introduce the latent variable model used in our WiSE-ALE algorithm and compare with the latent variable model used in the AEVB algorithm BID9 .

DISPLAYFORM0 , we assume x is generated from a latent variable z ∈ R dz of a much lower dimension.

Here we denote x and z as random variables, DISPLAYFORM1 as the i-th input or latent code sample (i.e. a vector), and x i and z i as the random variable for x (i) and z (i) .

As shown in FIG1 , this generative process can be modelled by a simple directed graphical model BID8 , which models the joint probability distribution DISPLAYFORM2 is the data distribution for D N and p θ (x|z) and p θ (z|x) denote the complex transformation from the latent to the input space and reverse, where the transformation mapping is parameterised by θ.

The learning task is to estimate the optimal set of θ so that this latent variable model can explain the data D N well.

As the inference of the latent variable z given x (i.e. p θ (z|x)) cannot be directly estimated because p(x|D N ) is unknown, both AEVB ( FIG1 ) and our WiSE-ALE FIG1 ) resort to variational method to approximate the target distribution p θ (z|x) by a proposal distribution q φ (z|x) with the modified learning objective that both θ and φ are optimised so that the model can explain the data well and q φ (z|x) approaches p θ (z|x).

The primary difference between the AEVB model and our WiSE-ALE model lies in how the joint probability p θ (x, z|D N ) is modelled and specifically whether we assume an individual random variable for each latent code z (i) .

The AEVB model assumes a pair of random variables (x i , z i ) for each x (i) and estimates the joint probability as DISPLAYFORM3 The equality between Eq. 2 and Eq. 3 can only be made with the assumption that the generation process for each x i is independent (first product in Eq. 3) and each z i is also independent (second product in Eq. 3).

Such interpretation of the joint probability leads to the latent variable model in FIG1 (b) and the prior constraint (often taken as N (0, I) to encourage shrinkage when no data is observed) is imposed on every z i .In contrast, our WiSE-ALE model takes a single random variable to estimate the latent distribution for the entire dataset D N .

Hence, the joint probability in our model can be broken down as DISPLAYFORM4 leading to the latent variable model illustrated in FIG1 .

The only assumption we make in our model is assuming the generative process of different input samples given the latent distribution of the current dataset as independent, which we consider as a sensible assumption.

More significantly, we do not require independence between different z i as opposed to the AEVB model, leading to a more flexible model.

Furthermore, the prior constraint in our model is naturally imposed on the aggregate posterior p(z|D N ) for the entire dataset, leading to more flexibility for each individual sample latent code to shape an embedding distribution to preserve a better quality of information about the corresponding input sample.

Neural networks can be used to parameterize p θ (x i |z i ) in the generative model and q φ (z i |x i ) in the inference model from the AEVB latent variable model or p θ (x i |z) and q φ (z|x i ) correspondingly from our WiSE-ALE latent variable model.

Both networks can be implemented through an auto-encoder network illustrated in FIG1 (d).

In this section, we first define the aggregate posterior distribution p(z|D N ) which serves as a core concept in our WiSE-ALE proposal.

We then derive a variational lower bound to the data log likelihood log p(D N ) using p(z|D N ).

Further, analytic approximation to the lower bound is derived, allowing efficient optimization of the model parameters and leading to our WiSE-ALE learning algorithm.

Intuition of our proposal is also discussed.

Here we formally define the aggregate posterior distribution p(z|D N ), i.e. the latent distribution given the entire dataset D N .

Considering DISPLAYFORM0 we have the aggregate posterior distribution for the entire dataset as the average of all the individual sample posteriors.

The second equality in Eq. 9 is made by approximating the integral through summation.

The third equality is obtained following the conventional assumption in the VAE literature that each input sample, DISPLAYFORM1 , is drawn from the dataset D N with equal probability, i.e. DISPLAYFORM2 Similarly, for the estimated aggregate posterior distribution q(z|D N ), we have DISPLAYFORM3

To carry out variational inference, we minimize the KL divergence between the estimated and the true aggregate posterior distributions q φ (z|D N ) and p θ (z|D N ), i.e. DISPLAYFORM0 in Eq. 11 and breaking down the products and fractions inside the log, we have DISPLAYFORM1 Re-arranging the above equation, we have DISPLAYFORM2 There are two terms in the derived lower bound: 1 a reconstruction likelihood term that indicates how likely the current dataset D N are generated by the aggregate latent posterior distribution q φ (z|D N ) and 2 a prior constraint that penalizes severe deviation of the aggregate latent posterior distribution q φ (z|D N ) from the preferred prior p(z), acting naturally as a regularizer.

By maximizing the lower bound L WiSE-ALE (φ, θ; D N ) defined in Eq. 12, we are approaching to log p(D N ) and, hence, obtaining a set of parameters θ and φ that find a natural balance between a good reconstruction likelihood (good explanation of the observed data) and a reasonable level of compliance to the prior assumption (achieving some preferable properties of the posterior distribution, such as smoothness and compactness).

To allow fast and efficient optimization of the model parameters θ and φ, we derive analytic approximations for the two terms in our proposed lower bound (Eq. 12).

To approximate 1 reconstruction likelihood term in Eq. 12, we first substitute the definition of the approximate aggregate posterior given in Eq. 10 in the expectation operation in DISPLAYFORM0 Now we can decompose the p θ (D N |z) as a product of individual sample likelihood, due to the conditional independence, i.e. DISPLAYFORM1 Substituting this into Eq. 13, we have DISPLAYFORM2 Eq. 15 can be used to evaluate the reconstruction likelihood for D N .

However, learning directly with this reconstruction estimate does not lead to convergence in our experiments.

We choose to simplify the reconstruction likelihood further to be able to reach convergence during learning at the cost of losing the lower bound property of the objective function L WiSE-ALE (φ, θ; D N ).

Firstly, we apply Jensen inequality to the term inside the expectation in Eq. 15, leading to an upper bound of the reconstruction likelihood term as DISPLAYFORM3 Now (N − 1) sample-wise likelihood distributions in the summation inside the log can be dropped with the assumption that the p θ (x (j) |z) will only be non-zero if z is sampled from the posterior distribution of the same sample x (j) at the encoder, i.e. i = j. Therefore, the approximation becomes DISPLAYFORM4 Using the approximation of the reconstruction likelihood term given by Eq. 17 rather than Eq. 15, we are able to reach convergence efficiently during learning at the cost of the estimated objective no longer remaining a lower bound to log p(D N ).

Details of deriving the above approximation are given in Appendix A.

The 2 prior constraint term D KL q φ (z|D N ) p(z) in our objective function (Eq. 12) evaluates the KL divergence between the approximate aggregate posterior distribution q φ (z|D N ) and a zero-mean, unit-variance Gaussian distribution p(z).

Here we assume that each sample-wise posterior distribution can be modelled by a factorial Gaussian distribution, i.e. q φ (z|x DISPLAYFORM0 , where k indicates the k-th dimension of the latent variable z and µ k (x (i) ) and σ 2 k (x (i) ) are the mean and variance of the k-th dimension embedding distribution for the input x (i) .

Therefore, D KL q φ (z|D N ) p(z) computes the KL divergence between a mixture of Gaussians (as Eq. 10) and N (0, I).

There is no analytical solution for such KL divergences.

Hence, we derive an analytic upper bound allowing for efficient evaluation.

DISPLAYFORM1 Applying Jensen inequality, i.e. E x log f (x) ≤ log E x f (x) , to the first term inside the summation in Eq. 18, we have DISPLAYFORM2 Taking advantage of the Gaussian assumption for q φ (z|x (i) ) and p(z), we can compute the expectations in Eq. 20 analytically with the result quoted below and the full derivation given in Appendix B. DISPLAYFORM3 where A = 2π (σ DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 When the overall objective function L WiSE-ALE (φ, θ; D N ) in Eq. 12 is maximised, this upper bound approximation will approach the true KL divergence D KL q φ (z|D N ) p(z) , which ensures that the prior constraint on the overall aggregate posterior distribution takes effects.

Combining results from Section 3.3.1 and 3.3.2, we obtain an analytic approximation L WiSE-ALE approx DISPLAYFORM0 Eq. 12, as -4 -3 -2 -1 0 1 2 3 4 5 -4 -3 -2 -1 0 1 2 3 4 5

Ours: Figure 3 : Comparison between our WiSE-ALE learning scheme and the AEVB estimator.

AEVB imposes the prior constraint on every sample embedding distribution, whereas our WiSE-ALE imposes the constraint to the overall aggregate embedding distribution over the entire dataset (over a mini-batch as an approximation for efficient learning).shown below: DISPLAYFORM0 where we use L φ, θ | x DISPLAYFORM1 to denote the sample-wise reconstruction likelihood (φ, θ; D N ) w.r.t the model parameters φ and θ, we are able to learn a model that naturally balances between a good embedding of the observed data and some preferred properties of the latent embedding distributions, such as smoothness and compactness.

DISPLAYFORM2

Comparing the objective function in our WiSE-ALE algorithm and that proposed in AEVB algorithm BID9 DISPLAYFORM0 we notice that the difference lies in the form of prior constraint and the difference is illustrated in Fig. 3 .

AEVB learning algorithm imposes the prior constraint on every sample embedding distribution and any deviation away from the zero code or the unit variance will incur penalty.

This will cause problems, as different samples cannot be simultaneously embedded to the zero code.

Furthermore, when the model becomes more certain about the embedding of a specific sample as the learning continues, it will naturally favour a posterior distribution of small variance (e.g. less than 1).

In contrast, our WiSE-ALE learning objective imposes the prior constraint on the aggregate posterior distribution, i.e. the average of all the sample embeddings.

Such prior constraint will allow more flexibility for each sample posterior to settle at a mean and variance value in favour for good reconstruction quality, while preventing too large mean values (acting as a regulariser) or too small variance values (ensuring smoothness of the learnt latent representation).To investigate the different behaviours of the two prior constraints more concretely, we consider only two embedding distributions q(z|x (1) ) and q(z|x (2) ) (red dashed lines) in a 1D latent space, as shown in FIG2 .

The mean values of the two embedding distributions are fixed to make the analysis simple and their variances are allowed to change.

When the variances of the two embedding distributions into the latent space (the more separable q(z|x (1) ) and q(z|x (2) ) are in the latent space, the easier it is to distinguish x are large, such as FIG2 , q(z|x (1) ) and q(z|x (2) ) have a large area of overlap and it is difficult to distinguish the input samples x in the latent space, indicating the embedding only introduces a small level of information loss.

Overall, the prior constraint in the AEVB objective favours the embedding distributions much closer to the uninformative N (0, I) prior, leading to large area of overlap between the individual posteriors, whereas our WiSE-ALE objective allows a wide range of acceptable embedding mean and variance, which will then offer more flexibility in the learnt posteriors to maintain a good reconstruction quality.

So far our derivation has been for the entire dataset D N .

Given a small subset B M with M samples randomly drawn from D N , we can obtain a variational lower bound for a mini-batch as: DISPLAYFORM0 When B M is reasonably large, then DISPLAYFORM1 Given the expressions for the objective functions derived in Section 3.3, we can compute the gradient for an approximation to the lower bound of a mini-batch B M and apply stochastic gradient ascent algorithm to iteratively optimize the parameters φ and θ.

We can thus apply our WiSE-ALE algorithm efficiently to a mini-batch and learn a meaningful internal representation of the entire dataset.

Algorithmically, WiSE-ALE is similar to AEVB, save for an alternate objective function as per Section 3.3.3.

The procedural details of the algorithm are presented in Appendix C.4 RELATED WORK BID1 proposes that a learned representation of data should exhibit some generally preferable features, such as smoothness, sparsity and simplicity.

However, these attributes are not tailored to any specific downstream tasks.

Bayesian decision making (see e.g. Lacoste-Julien et al.(2011); BID4 ) requires consideration of a target task and proposes that any involved latent distribution approximations should be optimised for the performance over the target task, as well as conforming to the more general properties.

The AEVB algorithm BID9 learns the latent posterior distribution under a reconstruction task, while simultaneously satisfying a prior constraint, which ensures the representation is smooth and compact.

However, the prior constraint of the AEVB algorithm imposes significant influence on the solution space (as discussed in Section 3.4), and leads to a sacrifice of reconstruction quality.

Our WiSE-ALE algorithm, however, prioritises the reconstruction task yet still enables globally desirable properties.

WiSE-ALE is not the only algorithm that considers an alternate prior form to mitigate its impact on the reconstruction quality.

The Gaussian Mixture VAE BID5 uses a Gaussian mixture model to parameterise p(z), encouraging more flexible sample posteriors.

The Adversarial Auto-Encoder BID13 ) matches the aggregate posterior over the latent variables with a prior distribution through adversarial training.

The WAE (Tolstikhin et al., 2017) minimises a penalised form of the Wasserstein distance between the aggregate posterior distribution and the prior, claiming a generalisation of the AAE algorithm under the theory of optimal transport (Villani, 2008) .

More recently, the Sinkhorn Auto-Encoder BID14 ) builds a formal analysis of auto-encoders using an optimal transport based prior and uses the Sinkhorn algorithm as an alternative to estimate the Wasserstein distance in WAE.Our work differs from these in two main aspects.

Firstly, our objective function can be evaluated analytically, leading to an efficient optimization process.

In many of the above work, the optimization involves adversarial training and some hyper-parameter tuning, which leading to less efficient learning or even no convergence.

Secondly, our WiSE-ALE algorithm naturally finds a balance between good reconstruction quality and preferred latent representation properties, such as smoothness and compactness, as shown in FIG0 .

In contrast, some other work sacrifice the properties of smoothness and compactness severely for improved reconstruction quality, as shown in FIG0 .

Many works have indicated that those properties of the learnt latent representation are essential for tasks that require optimisation over the latent space.

We evaluate our WiSE-ALE algorithm in comparison with AEVB, β-VAE and WAE on the following 3 datasets.

The implementation details for all experiments are given in Appendix E.1.

Sine Wave.

We generated 200,000 sine waves with small random noise: x(t) = A sin(2πf t + ϕ) + , each containing 256 samples, with independently sampled frequency f ∼ Unif(0, 20Hz), phase angle ϕ ∼ Unif(0, 2π) and amplitude A ∼ Unif(0, 2).

2. MNIST (LeCun, 1998) .

70,000 28 × 28 binary images that contain hand-written digits.3.

CelebA BID12 .

202,599 RGB images of aligned celebrity faces of 218 × 178 are cropped to square images of 178 × 178 and resized to 64 × 64.

Throughout all experiments, our method has shown consistently superior reconstruction quality compared to AEVB, β-VAE and WAE.

FIG6 offers a graphical comparison across the reconstructed samples given by different methods for the sine wave and CelebA datasets.

For the sine wave dataset, our WiSE-ALE algorithms achieves almost perfect reconstruction, whereas AEVB and β-VAE often struggle with low-frequency signals and have difficulty predicting the amplitude correctly.

For the CelebA dataset, our WiSE-ALE manages to predict much sharper human faces, whereas the AEVB predictions are often blurry and personal characteristics are often ignored.

WAE reaches a similar level of reconstruction quality to ours in some images, but it sometimes struggles with discovering the right elevation and azimuth angles, as shown in the second to the right column in FIG6 .

We understand that a good latent representation should not only reconstruct well, but also preserve some preferable qualities, such as smoothness, compactness and possibly meaningful interpretation of the original data.

FIG0 indicates that our WiSE-ALE automatically learns a latent representation that finds a good tradeoff between minimizing the information loss and maintaining a smooth and compact aggregate posterior distribution.

Furthermore, as shown in FIG7 , we compare the ELBO values given by AEVB, β-VAE and our WiSE-ALE over training for the Sine dataset.

Our WiSE-ALE manages to report the highest ELBO with a significantly lower reconstruction error and a fairly good performance in the KL divergence loss.

This indicates that our WiSE-ALE is able to learn an overall good quality representation that is closest to the true latent distribution which gives rise to the data observation.

In this paper, we propose a new latent variable model where a global latent variable is used to generate the entire dataset.

We then derive a variational lower bound to the data log likelihood, which allows us to impose a prior constraint on the bulk statistics of the aggregate posterior distribution for the entire dataset.

Using an analytic approximation to this lower bound as our learning objective, we propose WiSE-ALE algorithm.

We have demonstrated its ability to achieve excellent reconstruction quality, as well as forming a smooth, compact and meaningful latent representation.

In the future, we would like to understand the properties of the latent embeddings learnt through our method and apply it for suitable applications.

In this appendix, we omit the trainable parameters φ and θ in the expressions of distributions for simplicity.

For example, q(z|x) is equivalent to q φ (z|x) and p(x|z) represents p θ (x|z).

Here we demonstration that the reconstruction term E q(z|D N ) log p(D N |z) in our lower bound can be computed with individual sample likelihood log p(x (i) |z) and how our reconstruction error term becomes the same as the reconstruction term in the AEVB objective.

Firstly, we can substitute DISPLAYFORM0 into the reconstruction term DISPLAYFORM1 Now we can decompose the the marginal likelihood of the entire dataset as a product of individual samples, due to the conditional independence, i.e. DISPLAYFORM2 Substituting this into the reconstruction term, we have: DISPLAYFORM3 To evaluate the reconstruction term in our lower bound, we need to do the following: 1) draw a sample x (i) from the dataset D N ; 2) evaluate the latent code distribution q(z|x (i) ) through the encoder function q(·|x (i) ); 3) draw samples of z according to q(z|x (i) ); 4) reconstruct input samples using the sampled latent codes z (l) ; 5) compute the reconstruction error w.r.t to every single input sample and sum this error.

We can simplify the above evaluation.

Firstly, the sampling process in Step 3 can be replaced to a sampling process at the input using the reparameterisation trick.

Besides, the sum of reconstruction errors w.r.t.

all the input samples can be further simplified.

To do this, we need to re-arrange the above expression as DISPLAYFORM4 log a i to the terms inside the expectation.

As a result, we have obtain an upper bound of the reconstruction error term as DISPLAYFORM5 This upper bound can be evaluated more efficiently with the assumption that the likelihood p(x (j) |z) representing the probability of a reconstructed sample from a latent code z imitating the sample x (j) will only be non-zero if z is sampled from the embedding prediction distribution with the same sample x (j) at the encoder input.

With this assumption, N − 1 posterior distributions in the inner summation will be dropped as zeros and the only non-zero term is p(x (i) |z).

Therefore, the upper bound becomes DISPLAYFORM6 The constant can be omitted, because it will not affect the gradient updates of the parameters.

Applying Jensen inequality, i.e. DISPLAYFORM0 to the first term of above equation, we have DISPLAYFORM1 We will look at the two summation individually.

The expectation w.r.t.

the aggregate posterior can be expanded as DISPLAYFORM2 We assume the posterior distribution of the latent code z given a specific input sample DISPLAYFORM3 Similarly, DISPLAYFORM4 Therefore, DISPLAYFORM5 Substituting the exponential form for Gaussian distribution, i.e. DISPLAYFORM6 to the above equation, we have DISPLAYFORM7 The exponent of the above equation can be simplified to DISPLAYFORM8 Using the following properties, i.e. DISPLAYFORM9 we can evaluate the integral needed for DISPLAYFORM10 Therefore, we have obtained the expression for the first term in our upper bound, i.e. DISPLAYFORM11 DISPLAYFORM12 To find out the expression for the second term DISPLAYFORM13 , we first examine the prior distribution p(z) which is chosen to be a zero-mean unit-variance Gaussian across all latent code dimensions, i.e. DISPLAYFORM14 Therefore, DISPLAYFORM15 Substituting this expression for log p(z) into DISPLAYFORM16 ) log p(z) and examining the expectation term for now, we have DISPLAYFORM17 The first integral q(z k |x (i) ) dz k = 1.

To evaluate the second integral, we substitute Equation (4) and use the following properties, i.e. DISPLAYFORM18 As a result, we have DISPLAYFORM19 Therefore, DISPLAYFORM20 Combining the first term defined in Equation (6) and the second term defined in Equation FORMULA12 , we have obtained the expression for the overall upper bound as DISPLAYFORM21 + log 2π .

We carry out experiments on four datasets (Sine wave, MNIST, Teapot and CelebA) to examine different properties of the latent representation learnt from the proposed WiSE algorithm.

Specifically, we compare with β-VAE on the smoothness and disentanglement of the learnt representation and compare with WAE and AEVB on the reconstruction quality.

In addition, by learning a 2D embedding of the MNIST dataset, we are able to visualise the latent embedding distributions learnt from AEVB, β-VAE, WAE and our WiSE and compare the compactness and smoothness of the learnt latent space across these methods.

Here we give the implementation details for each dataset.

We aim to learn a latent representation in R 4 for a one second long sine wave with sampling rate of 256Hz.

The network architecture for the Sine wave dataset is shown below.

x is an input sample, µ and σ are the latent code mean and latent code standard deviation to define the embedding distribution q(z|x) andx is the reconstructed input sample.

is an auxiliary variable drawn from unit Gaussian at the input of the encoder network so that an estimate of a sample from the embedding distribution q(z|x) can be computed.

Conv We aim to learn a 2D embedding of the MNIST dataset.

The network architecture is shown below.

Encoder network:x ∈ R 28×28×1 → Conv Decoder network: DISPLAYFORM0 ⇒x ∈ R

We use the following hyper-parameters to train the network:

<|TLDR|>

@highlight

We propose a new latent variable model to learn latent embeddings for some high-dimensional data. 