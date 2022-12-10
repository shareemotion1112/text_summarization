Normalising Flows (NFs) are a class of likelihood-based generative models that have recently gained popularity.

They are based on the idea of transforming a simple density into that of the data.

We seek to better understand this class of models, and how they compare to previously proposed techniques for generative modeling and unsupervised representation learning.

For this purpose we reinterpret NFs in the framework of Variational Autoencoders (VAEs), and present a new form of VAE that generalises normalising flows.

The new generalised model also reveals a close connection to denoising autoencoders, and we therefore call our model the Variational Denoising Autoencoder (VDAE).

Using our unified model, we systematically examine the model space between flows, variational autoencoders, and denoising autoencoders, in a set of preliminary experiments on the MNIST handwritten digits.

The experiments shed light on the modeling assumptions implicit in these models, and they suggest multiple new directions for future research in this space.

Unsupervised learning offers the promise of leveraging unlabeled data to learn representations useful for downstream tasks when labeled data is scarce BID47 , or even to generate novel data in domains where it is costly to obtain BID15 .

Generative models are particularly appealing for this as they provide a statistical model of the data, x, usually in the form of a joint probability density p (x).

The model's density function, its samples and representations can then be leveraged in applications ranging from semi-supervised learning and speech and (conditional) image synthesis BID44 BID30 BID14 BID26 to gene expression analysis BID13 and molecule design BID10 .In practice, data x is often high-dimensional and the optimization associated with learning p (x) can be challenging due to an abundance of local minima BID39 and difficulty in sampling from rich high-dimensional distributions BID34 .

Despite this, generative modelling has undergone a surge of advancements with recent developments in likelihood-based models BID25 BID8 BID44 and Generative Adversarial Networks (GANs; BID11 ).

The former class is particularly attractive, as it offers (approximate) likelihood evaluation and the ability to train models using likelihood maximisation, as well as interpretable latent representations.

Autoencoders have a rich history in the unsupervised learning literature owing to their intuitive and simple construction for learning complex latent representations of data.

Through fitting a parameterised mapping from the data through a lower dimensional or otherwise constrained layer back to the same data, the model learns to summarise the data in a compact latent representation.

Many variants of autoencoders have been proposed to encourage the model to better encode the underlying structure of the data though regularising or otherwise constraining the model (e.g., BID38 BID1 .Denoising Autoencoders (DAEs) are a variant of the autoencoder under which noise is added to the input data that the model must then output noise-free, i.e. x = f θ (x + ) where is sampled from a, possibly structured BID48 BID49 , noise distribution ∼ q( ).

They are inspired by the idea that a good representation z would be robust to noise corrupting the data x and that adding noise would discourage the model from simply learning the identity mapping.

Although DAEs have been cast as generative models , sampling and computing likelihoods under the model remains challenging.

Variational Autoencoders (VAEs) instead assume a probabilistic latent variable model, in which n-dimensional data x correspond to m-dimensional latent representations z following some tractable prior distribution, i.e. x ∼ p φ (x|z) with z ∼ p (z) BID25 .

The task is then to learn parameters φ, which requires maximising the log marginal likelihood DISPLAYFORM0 In the majority of practical cases (e.g. p φ (x|z) taken to be a flexible neural network-conditional distribution) the above integral is intractable.

A variational lower bound on the marginal likelihood is constructed using a variational approximation q θ (z|x) to the unknown posterior p (z|x): DISPLAYFORM1 The right-hand side of (2), denoted L (θ, φ), is known as the evidence lower bound (ELBO).

It can be jointly optimised with stochastic optimisation w.r.t.

parameters θ and φ in place of (1).Conditionals q θ (z|x) and p φ (x|z) can be viewed respectively as probabilistically encoding data x in the latent space, and reconstructing it from samples of this encoding.

The first term of the ELBO encourages good reconstructions, whereas the second term encourages the model's latent variables to be distributed according to the prior p (z).

Generating new data using this model is accomplished by reconstructing samples from the prior.

Normalising Flows (NFs) suppose that the sought distribution p (x) can be obtained by warping a simple base density p (z), e.g. a normal distribution BID36 .

They make use of the change of variables formula to obtain p (x) through a learned invertible transformation z = f θ (x) as DISPLAYFORM2 Typically, f θ : R n → R n is obtained by stacking several simpler mappings, i.e. DISPLAYFORM3 and the log-determinant obtained as the sum of log-determinants of these mappings.

This formulation allows for exact maximum likelihood learning, but requires f θ to be invertible and to have a tractable inverse and Jacobian determinant.

This restricts the flexibility of known transformations that can be used in NFs BID8 BID3 and leads to large and computationally intensive models in practice BID26 .NFs can also be thought of as VAEs with encoder and decoder modelled as Dirac deltas p θ (x|z) = δ (f θ (z)) and q θ (z|x) = δ f −1 θ (x) , constructed using a restricted set of transformations.

Furthermore, because NFs model continuous density, to prevent trivial solutions with infinite point densities discrete data must be dequantised by adding random noise BID42 BID40 .The contribution of this work is two-fold.

First, we shed new light on the relationship between DAEs, VAEs and NFs, and discuss the pros and cons of these model classes.

Then, we also introduce several extensions of these models, which we collectively refer to as the Variational Denoising Autoencoders (VDAEs).In the most general form VDAEs generalise NFs and DAEs to discrete data and learned noise distributions.

However, when the amount of injected noise is small, VDAE attains a form that allows for using non-invertible transformations (e.g. f θ : R n → R m , with m n).

We demonstrate these theoretical advantages through preliminary experimental results on the binary and continuous versions of the MNIST dataset.

We model data x, a n dimensional vector that can have either continuous or discrete support.

As is customary for VAEs, our model for x is hierarchical and assumes a set of latent variables z with tractable prior distribution p(z), and a flexible neural-network conditional distribution p(x|z).

On top of this standard VAE setup, we specify the dimension of z to equal the dimension of the data x. In order to form the variational lower bound to train this model, we need an approximate inference model, or encoder, q θ (z|x).

Here, we will use an encoder that samples the latents z as DISPLAYFORM0 where q ( ) is a tractable noise distribution and f θ (x) is a one-to-one transformation with tractable Jacobian-determinant.

In order to use the encoder q θ (z|x) implied by this procedure, we not only need to sample from it, but we must also evaluate its entropy for the KL-term in (2).

To do this we make use of the fact that z is a one-to-one transformation of the noise , given the training data x. Using the standard formulas for a change of variables, we thus get the following expression for the entropy of q θ (z|x): DISPLAYFORM1 where q (x|x) is a distribution whose sampling process is described in (4).

Our variational lower bound (2) on the data log marginal likelihood then becomes DISPLAYFORM2 where againx = x + and z = f θ (x).This is similar to a denoising autoencoder in that we try to reconstruct the original data x from the corrupted datax through the conditional model p(x|z).

The difference with classical denoising autoencoders is that our objective has additional terms that regularise our latent representations z to be distributed according to a prior distribution p(z).

In addition, the proposed setup allows us to learn the noise distribution q( ), where this is treated as a fixed hyperparameter in the literature on denoising autoencoders.

This model is also a generalisation of normalising flows.

Specifically, consider the special case where we take DISPLAYFORM3 then the lower bound in (5) becomes the standard normalising flow log-likelihood (3).

We provide a detailed derivation in Appendix A.The advantage of our generalised model over standard normalising flows is that our model allows for non-zero noise level σ 2 .

Interestingly, successful applications of normalising flows in the literature often already add a significant amount of noise in order to dequantise the data, and empirical results suggest higher amounts of noise lead to models that produce better-looking samples (e.g. BID26 model only the top 5 bits of the data).In addition, our model does not require tying the parameters of the encoder and decoder.

Although we are still using a flow-based encoder q θ (z|x), our decoder is not restricted to a specific functional form.

The conditional distribution p(x|z) can e.g. have discrete support if the data x is discrete, making our model naturally applicable to data such as text or other highly-structured data, without requiring an explicit dequantisation step.

When adding a significant amount of noise, a decoupled decoder will generally be able to achieve a higher variational lower bound compared to using the tied-parameter decoder (6).

The VAE we proposed in Section 2 is more general than NFs, but it still requires an invertible one-to-one encoder with tractable Jacobian-determinant.

This restricts our modeling choices since all transformations used in the encoder can only be chosen from a small set of transformations for which we know how to compute inverses and Jacobian-determinants.

Additionally, the representation given by our encoder will be of the same dimension as our data x, which may not be optimal for all applications (e.g. model-based reinforcement learning BID15 BID17 or compression BID2 ).

To relax these restrictions further we generalise our model to allow non-invertible encoders as well.

We proceed by taking our model from Section 2, withx = x + and z = f θ (x), and performing a Taylor expansion of the resulting latent variables z(x, ) around = 0 (see Appendix B).

This gives DISPLAYFORM0 where DISPLAYFORM1 is the Jacobian of f θ .For small noise levels, as used in Section 2, the O( 2 ) term becomes negligible.

If the noise distribution is Gaussian, i.e. q ( ) = N 0, σ 2 I n , this means that for small σ we get DISPLAYFORM2 Using this form of encoder q θ (z|x), together with general prior p(z) and conditional distribution p(x|z), we get a VAE that still generalises NFs but now also allows us to choose non-invertible non-one-to-one transformations f θ .

We refer to this even broader class of VAE as L-VDAE, for Linearised-VDAE.

Evaluating the entropy H [q θ (z|x)] in this case requires computing the log-determinant of the covariance matrix C = JJ T for the data x: DISPLAYFORM0 where m is the dimensionality of z. When using transformations f θ without a tractable Jacobian (e.g. the general Residual Network (ResNet; BID20 ) blocks), we explicitly evaluate C and compute log det C = m i log λ i , where the eigenvalues λ i are obtained using the eigenvalue decomposition C = QΛQ T with Λ = diag (λ i |i = 1, . . .

, m).

The decomposition is further re-used in the backward pass when evaluating the derivative of the log-determinant using Jacobi's formula: DISPLAYFORM1 Evaluation of the Jacobian J can be done by performing reverse-mode automatic differentiation with respect to each element of z, thus incurs a factor of m additional computational cost.

Covariance matrix C is obtained using a single matrix multiplication and takes O m 2 n operations with the eigenvalue decomposition taking another O m 3 operations.

Taken together, evaluation of FORMULA11 takes O m 2 n operations, which is comparable to the O d 3 cost of Glow's 1x1 invertible convolutions in later layers (i.e. after repeating the use of the multi-scale architecture from BID9 that trades spatial dimensions for channels), where d refers to the number of channels used in the convolution.

This computational cost is permissive for small latent space dimensionalities m. However, scaling up L-VDAE to larger latent spaces would require stochastic approximations of the log-determinant BID18 .

These approximations can be implemented efficiently through Jacobian-vector and vector-Jacobian products, without evaluating C or J explicitly, and can be optimised directly by backpropagating through the approximation.

With this approach computational complexity will be linear in n subject to some regularity conditions.

Sampling from the Gaussian variational posterior q φ is necessary for training and inference in L-VDAE.

It can be accomplished using the standard reparameterisation trick BID25 , where random normal noise ω ∼ N (0, I n ) is transformed into the posterior sample as z = f θ (x) + Jω.

We implement this as a Jacobian-vector product, which enables efficient sampling for cases when the Jacobian log-determinant of f θ is cheaper to evaluate than the Jacobian itself (e.g. when f θ is a flow).

VDAE blends ideas from the VAE and NFs literature and is closely related to both model families.

It is most similar to methods that combined variational inference and NFs by using the latter as part of the approximate variational posterior BID36 BID29 BID43 .

These methods use a strategy in which samples from the (Gaussian) posterior are further transformed with a NF, whereas in VDAE the posterior distribution is implicitly defined using a sampling procedure inspired by DAE, where posterior samples are obtained by transforming data with added noise using NFs.

VDAE is a natural formulation of DAEs as probabilistic models.

It is conceptually similar to the Denoising VAEs BID23 , which propose an alternative probabilistic formulation of DAEs as VAEs.

The method of Im et al., however, does not generalise NFs and, in contrast to VDAE, it requires explicitly choosing the type and amount of corruption noise.

The idea of challenging the default choice of using uniform noise for dequantisation in NFs was also explored in Flow++ BID21 , where the authors learned a flexible conditional noise model q ( |x) as NF itself.

Our sampling procedure (4) is similar to dequantisation in Flow++, as it can be viewed as a result of applying a NF to dequantisation given by an implicitly conditioned noise model.

The main differences, however, are that in VDAE the decoder reconstructs the original (quantised) data, which is also what makes our model applicable to highly-structured data; and, in contrast to Flow++, VDAE can inject substancially more noise than a single dequantisation bin.

In relation to VAEs, the linearised form of VDAE can be viewed as an extension of the vanilla VAE BID25 ) that replaces the diagonal Gaussian posterior with a Jacobian-based full covariance posterior.

It is thus similar to methods that extend VAE with more flexible prior (e.g. autoregressive BID7 or mixture (Tomczak & Welling, 2017)) or variational posterior (e.g. full covariance Gaussian (Kingma et al., 2016a) or mixture BID35 BID32 ) distributions.

Notably, unlike some of these methods, L-VDAE does not increase the number of parameters of the inference or generative networks.

As a method that increases flexibility of transformations in NFs, L-VDAE with non-invertible encoders can be compared to Invertible Residual Networks (i-ResNets; BID3 ) and FFJORD BID12 .

These methods too depart from the requirement of restricting the form of the Jacobian of the resulting transformation.

Both, i-ResNets and FFJORD also drop the requirement of having an analytical inverse, which is similar to how VDAE seeks to learn an approximate inverse using its decoder network.

However, unlike VDAE, these methods guarantee invertibility and provide ways of computing the exact inverse.

Notably, the methods differ considerably in how they achieve the above generalisations.

In i-ResNets Behrmann et al. make use of the ResNet network architecture BID20 and identify conditions on the eigenvalues of the residual blocks, under which they parameterise invertible mappings.

They then make use of spectral normalisation BID33 to guarantee that the condition is satisfied throughout training; and employ fixed point iteration to invert the residual blocks for generation.

i-ResNets further lift the restriction on the form of the Jacobian in a computationally tractable way by using Taylor series expansion in conjunction with stochastic trace estimation BID22 .FFJORD BID12 ) is inspired by the re-interpretation of ResNets and NFs as discrete approximations of solutions to the initial value problem of some underlying ODE continuously transforming data x (from data-space to the latent z-space; ).

BID12 ; parameterise this ODE as a neural network f θ (z (t) , t) to obtain Continuous-time Normalising Flows (CNFs), in which the change in log-density at time t is given by the instantaneous change of variables formula DISPLAYFORM0 The right-hand side of FORMULA13 is given by the trace of the Jacobian of transformation f θ instead of the log-determinant as in NFs.

Combined with the use of stochastic trace estimation BID22 , this difference alleviates the need to restrict transformations f θ to those with a tractable Jacobian log-determinant.

However, the use of ODEs also necessitates employing an ODE solver to integrate (9) for every evaluation of, and backpropagation through log p θ (z (t)).

The number of function evaluations required for this increases with training and may become prohibitively large BID12 .Finally, VDAE is loosely related to autoregressive generative models, as they both fall into the class of likelihood-based generative models.

Autoregressive models factorise likelihood of highdimensional data as a product of simple per-dimension conditional distributions, i.e. p (x) = i p (x i |x 0 , . . .

, x i−1 ) (van den BID44 b) .

Factorised structure of these models necessitates sequential sampling, and a good choice of the ordering of dimensions of x. Overcoming these challenges in practice often requires highly engineered solutions, for example as in BID46 or BID31 .

Furthermore, data representations formed by hidden layers of autoregressive models appear to be more challenging to manipulate than in VAEs or NFs BID26 .

We performed empirical studies of the performance of VDAE on the image generation task on the MNIST dataset BID30 , comparing it to a VAE implementation with a fully factorised Gaussian posterior and to the NICE BID8 normalising flow as baselines.

For the VDAE encoder we used additive couplings to construct f θ from the implicit variational posterior; and, unless otherwise specified, fully-connected ResNet blocks followed by a sigmoid transformation to obtain the decoder parameters µ φ and p φ .

A Gaussian distribution N (µ φ (z) , λI n ) with a learned parameter λ was used for the continuous MNIST decoder; and Bernoulli (p φ (z)) for binary MNIST.

Similarly, unless otherwise specified, ResNet blocks with linear projection layers to change dimensionality were used for the L-VDAE encoder and decoder.

Details of the chosen architectures can be found in Appendix D.

To model discrete 8-bit pixel values with continuous density models, we followed the procedure of BID42 to dequantise the data, and added noise u ∼ U (0, 1) to the pixel values prior to normalising them to [0, 1] .

Note that for VDAE this was done prior and in addition to adding noise from the posterior sampling procedure (4) to the inputs.

Decoupled encoder and decoder We start by confirming for a range of noise levels and architectures that de-coupling the encoder and decoder networks in VDAE allows for achieving higher ELBOs.

FIG0 compares ELBO attained by VDAE with Gaussian noise q ( ) = N 0, σ 2 I for a range of fixed σ.

The results show that any decoupling of the weights improves over the coupled network, in which the NICE flow is used in the encoder and its inverse -in the decoder.

Specifically, we observe that for architectures with a sigmoid activation in the last layer of the decoder, the ELBO rapidly improves with decreasing noise levels.

Based on these results, in the following experiments we only consider the more general ResNet architecture in the VDAE decoder.

We report average test set performance over 10 training runs; when sufficiently large, standard deviations are also given.

Qualitative samples are drawn from models with the best test ELBO among the training runs.

NLL was estimated via 5000 importance samples as in .Quantitative results We now consider the cases when i) noise variance σ 2 , in case of VDAE; or ii) in case of L-VDAE, the covariance scale σ from FORMULA10 ; are optimised together with the model.

Results of these experiments are shown in TAB0 .

For ease of presentation, we also include evaluation results for existing flow models (reproduced from BID3 ).We first note that for cases when the latent dimensionality is smaller than the input space (i.e. m < n), L-VDAE consistently outperforms the VAE baseline in terms of the achieved ELBO, albeit by a small margin.

This is consistent with L-VDAE having a more powerful variational posterior.

Moreover, for L-VDAE increasing the dimensionality of the latent space consistently improves the variational lower bound.

Surprisingly, L-VDAE with n = m and VDAE break this trend and do not improve on the ELBOs obtained for m = 128.

We also note that neither of our proposed extensions manage to achieve likelihoods comparable to NFs, including the NICE baseline.

Both shortcomings could be explained by the difference in architectures between the methods.

In contrast to the L-VDAE with m = n, which employs a NICE flow in the encoder, L-VDAE with m < n makes use of the more expressive ResNet blocks.

Similarly, the flexibility of the NICE flow used in VDAE for the implicit posterior may be insufficient for a denoising VAE.

We also observe that when using a NICE flow in the decoder, VDAE outperforms L-VDAE in terms of likelihood, signalling that the VDAE approach can further improve on the linearised models, if combined with a more powerful flow.

Qualitative results We found that without additional regularisation, such as fixing the decoder variance λ 2 or the noise variance σ 2 to values larger than what would have been learned by the model, or assigning a higher weight to the KL-term in the optimisation objective, our models would not produce high-quality samples for the continuous MNIST dataset.

We thus omit continuous MNIST model samples from the main text, but explore the effect of fixing the noise variance on sample quality in Appendix E.

To explore the applicability of VDAE to structured data, we applied it to the binarised version of the MNIST dataset.

As is customary for dynamic MNIST, digits were binarised by sampling from a Bernoulli distribution given by the pixel intensities.

Results in TAB1 mirror those we observed on the continuous MNIST, namely L-VDAE consistently achieves higher ELBO than the VAE baseline, which tends to improve as the latent dimensionality grows; and L-VDAE and VDAE, which make use of NICE in the decoder, attain significantly worse likelihood despite the increased dimensionality.

Finally, VDAE also improves on L-VDAE with a NICE encoder.

However, as shown in FIG1 , and in contrast to the continuous MNIST results, all our models produce plausible handwritten digit samples.

We introduced Variational Denoising Autoencoders (VDAEs), a family of models the bridges the gap between VAEs, NFs and DAEs.

Our model extends NFs to discrete data and non-invertible encoders that use lower-dimensional latent representations.

Preliminary experiments on the MNIST handwritten digits demonstrate that our model can be successfully applied to data with discrete support, attaining competitive likelihoods and generating plausible digit samples.

We also identified a failure mode of our models, in which their performance does not scale well to cases when latent and input dimensionalities are the same (i.e. when a flow-based encoder is used).Future work should address limitations of the method identified in our experiments.

In particular, replacing additive coupling blocks with the more powerful invertible convolutions, affine coupling blocks and invertible residual blocks BID9 BID26 BID3 can significantly improve the variational posterior for high dimensions.

It can also be interesting to explicitly condition the transformation f θ used for defining the posterior sampling procedure on the data x, for example by defining f θ (x, ) ≡ f x,θ ( ) using a hyper-network BID16 .

For convenience we start by repeating the variational lower bound for our VDAE model, as presented in Section 2: DISPLAYFORM0 n×n we zero-out all elements of W L .

The same scheme was employed for initialising additive coupling blocks, which can be viewed as residual blocks of a restricted form.

Projection layers reduce dimensionality of their inputs using a linear map y = xW with x ∈ R 1×n , y ∈ R 1×m and W ∈ R n×m .

This generally leads to loss of information and makes model training harder.

To mitigate this effect we initialise the rows of W using a set of m random orthogonal vectors.

The decoder projection layers, mapping data to higher dimensions, are then initialised to W T .

All models were trained for 1000 epochs using the ADAM optimiser BID24 ) with a batch size of 1000 samples.

To improve stability of the training, the learning rate was warmed up from 10 −5 to the chosen learning rate (see below) over the first 10 epochs.

Further, the KL term was warmed up by linearly annealing its weight β from 0 to 1 over the first 100 epochs BID5 .For each experiment, the learning rate schedule S ∈ {linear, none}, learning rate α ∈ 10 −5 , 10 DISPLAYFORM0 and ADAM optimiser parameters β 2 ∈ {0.9, 0.99, 0.999, 0.9999} and ∈ 10 −4 , 10 −5 , 10 DISPLAYFORM1 were determined by using Bayesian optimisation of the ELBO on the validation set.

NICE When implementing the model (standalone, or part of VDAE), we closely followed the architecture and hyper-parameters described in BID8 .

Namely, the network consisted of 4 additive coupling blocks, each with 5 fully-connected hidden layers of size 1024 with ReLU activations, followed by a linear layer (see Appendix C).

Dimension partitioning was alternated between even and odd dimensions after every block.

When used as a standalone model, a L 2 regularisation with weight λ = 0.01 was used to improve sample quality.

L-VDAE and vanilla VAE When not used in conjunction with a NICE model in the encoder, the L-VDAE and VAE models employed a fully-connected ResNet architecture with B consecutive residual blocks followed by a linear projection layer to higher or lower dimensions.

In the encoder, the last projection layer parameterised the means of the Gaussian variational posterior (and, in case of VAE, a parallel projection layer parameterised the log-variances).

A sequence of 4 residual-projection "blocks" was used with the last block i = 4 projecting to m dimensions (dimensionality of the latents) and the blocks before it, respectively to min 2 i · m, 28 × 28 dimensions.

Each residual block consisted of 2 hidden layers with ReLU activations followed by a linear layer (see Appendix C).

The residual block hidden size H ∈ {32, 64, 128, 256, 1024} and the block multiplicity B ∈ {1, 2, 3} were chosen through Bayesian optimisation as described above.

Unless otherwise specified, when used together with a NICE model, the VDAE and L-VDAE models employed a ResNet architecture in the decoder.

In this case, the ResNet architecture was chosen to closely resemble that of the NICE model.

Specifically, hyper-parameter values B = 1 and H = 1024 were used, and no projection layers were employed.

Priors We employed a logistic prior with s = 1 and µ = 0 (as in BID8 ) for models that made use of the NICE flow (even if it was only used in the encoder network); and a factorised normal prior otherwise.

Sample quality deteriorates at the extremes of the noise level spectrum: at high noise levels the model appears to be unable to learn the distribution, whereas at low noise levels the model appears to focus too much on the reconstruction error instead of organising the latent space.

Just as in DAEs, the noise variance σ 2 in VDAEs can be used as a regulariser; and can be tuned for sample quality.

<|TLDR|>

@highlight

We explore the relationship between Normalising Flows and Variational- and Denoising Autoencoders, and propose a novel model that generalises them.