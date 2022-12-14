Variational Autoencoders (VAEs) have proven to be powerful latent variable models.

How- ever, the form of the approximate posterior can limit the expressiveness of the model.

Categorical distributions are flexible and useful building blocks for example in neural memory layers.

We introduce the Hierarchical Discrete Variational Autoencoder (HD-VAE): a hi- erarchy of variational memory layers.

The Concrete/Gumbel-Softmax relaxation allows maximizing a surrogate of the Evidence Lower Bound by stochastic gradient ascent.

We show that, when using a limited number of latent variables, HD-VAE outperforms the Gaussian baseline on modelling multiple binary image datasets.

Training very deep HD-VAE remains a challenge due to the relaxation bias that is induced by the use of a surrogate objective.

We introduce a formal definition and conduct a preliminary theoretical and empirical study of the bias.

Unsupervised learning has proven powerful at leveraging vast amounts of raw unstructured data (Kingma et al., 2014; Radford et al., 2017; Peters et al., 2018; Devlin et al., 2018) .

Through unsupervised learning, latent variable models learn the explicit likelihood over an unlabeled dataset with an aim to discover hidden factors of variation as well as a generative process.

An example hereof, is the Variational Autoencoder (VAE) (Kingma and Welling, 2013; Rezende et al., 2014 ) that exploits neural networks to perform amortized approximate inference over the latent variables.

This approximation comes with limitations, both in terms of the latent prior and the amortized inference network (Burda et al., 2015; Hoffman and Johnson, 2016) .

It has been proposed to go beyond Gaussian priors and approximate posterior using, for instance, autoregressive flows (Chen et al., 2016; Kingma et al., 2016) , a hierarchy of latent variables (Sønderby et al., 2016; Maaløe et al., 2016 Maaløe et al., , 2019 , a mixture of priors (Tomczak and Welling, 2017) or discrete distributions (van den Oord et al., 2017; Razavi et al., 2019; Rolfe, 2016; Vahdat et al., 2018b,a; Sadeghi et al., 2019) .

Current state-of-the-art deep learning models are trained on web-scaled datasets and increasing the number of parameters has proven to be a way to yield remarkable results (Radford et al., 2019) .

Nonetheless, time complexity and GPU memory are scarce resources, and the need for both resources increases linearly with the depth of neural network.

Li et al. (2016) and Lample et al. (2019) showed that large memory layers are an effective way to increase the capacity of a model while reducing the computation time.

Bornschein et al. (2017) showed that discrete variational distributions are analogous to neural memory (Graves et al., 2016) , which can be used to improve generative models (Li et al., 2016; Lample et al., 2019) .

Also, memory values are yet another way to embed data, allowing for applications such as one-shot transfer learning (Rezende et al., 2016) and semi-supervised learning that scales (Jang et al., 2016) .

Depth promises to bring VAEs to the next frontier (Maaløe et al., 2019) .

However, the available computing resources may shorten that course.

Motivated by the versatility and the scalability of discrete distributions, we introduce the Hierarchical Discrete Variational Autoencoder.

HD-VAE is a VAE with a hierarchy of factorized categorical latent variables.

In contrast to the existing discrete latent variable methods, our model (a) is hierarchical, (b) trained using Concrete/Gumbel-Softmax, (c) relies on a conditional prior that is learned end-to-end and (d) uses a variational distribution that is parameterized as a large stochastic memory layer.

Despite being optimized for a biased surrogate objective we show that a shallow HD-VAE outperforms the baseline Gaussian-based models on multiple binary images datasets in terms of test log-likelihood.

This motivates us to introduce a definition of the relaxation bias and to measure how it is affected by the configuration of latent variables.

Hierarchical VAE Hierarchical VAEs define a model p θ (x, z) = p θ (x|z)p θ (z) where x is an observed variable and z = {z 1 , . . .

, z L } is a hierarchy of latent variables so that p θ (z) is factorized into L layers.

The inference model q φ (z|x) usually exploits the inverse dependency structure.

A vanilla hierarchical VAE results in the following model:

The choice of the VAE architecture is independent of the choice of the variational family and deeper models can easily be defined (see appendix F).

Variational Neural Memory Each stochastic layer consists of N categorical random variables with K class probabilities π = {π 1 , . . .

, π K } and can be parametrized as a memory layer.

Lample et al. (2019) recently proposed a scalable approach to attention-based memory layers that can be directly translated to the stochastic setting: Each categorical distribution is parametrized by factored keys {k 1 , . . .

, k K }, k i ∈ R d 1 and a parametric query model Q(h).

If {v 1 , ..., v K }, v i ∈ R d 2 are the memory values, for c ∈ R and i = 1, . . .

, K, then the output of the memory layer for one variable is

Optimization We wish to maximize the Evidence Lower Bound (ELBO):

The subscript of L denotes the number of importance weighted samples.

Guided by the analysis of Sønderby et al. (2017) , we chose to use the Concrete/GumbelSoftmax relaxation (Jang et al., 2016; Maddison et al., 2016) for differentiable, approximate sampling of categorical variables.

A relaxed categorical sample can be obtained as

where {g i } are i.i.d.

samples drawn from Gumbel(0,1), and τ ∈ R * + is a temperature parameter.

As in the categorical case, the output of the memory layer is a convex combination of the memory values weighted by the entries ofz:

The relaxed samplesz follow a Concrete/Gumbel-Softmax distribution q τ φ which depends on τ and converges to the categorical distribution q τ =0 φ = q φ as τ → 0 which is equivalent to applying the Gumbel-Max trick to soft samples, meaning z = H(z), H = one hot • arg max.

When we extend the definition of f θ,φ to the domain of the relaxed samples, as in appendix D, the surrogate objective that is maximized becomes

which is not guaranteed to be a lower bound of log p θ (x).

Hence, we are interested in the relaxation bias that we define as:

where

is the original ELBO.

If f θ,φ is a κ-Lipschitz for z, we can derive an upper bound for the relaxation bias as well as a new log-likelihood bound (relaxed ELBO) by adding a corrective term to the surrogate objective (derivation in appendix C).

For a one layer Ladder Variational Autoencoder (LVAE), it results in the following bounds:

This new bound shows that, if the model is unconstrained, the relaxation bias is free to grow and that it grows with the number of discrete variables.

In section 4.2, we provide empirical results supporting the monotonically increasing property of the relaxation bias with regards to the number of stochastic units.

Table 1 : Sample estimates of the KL(q φ,θ (z|x)||p θ (z)) and the ELBO for 1000 importance weighted hard samples (τ = 0) using the same LVAE architecture and hyperparameters across all datasets.

discrete normal discrete normal discrete normal discrete normal discrete normal discrete normal

discrete normal discrete normal discrete normal discrete normal discrete normal discrete normal (2017) .

To the best of our knowledge, HD-VAE is the only work that attempts to transform memory layers into a general purpose variational distribution.

We trained HD-VAE for different number of layers of latent variables using the surrogate objective defined in the equation 5.

In this experiment, we observe that HD-VAE consistently outperforms the baseline Gaussian model for multiple datasets and different number of latent layers (table 1) .

This shows that using variational memory layers yields a more flexible model than for the VAE with a Gaussian prior and the same number of latent variables.

Furthermore, optimizing latent variable models is challenging (Sønderby et al., 2016; Chen et al., 2016) .

In this experiment, the measured KL is higher for the discrete model, suggesting a well-tempered optimization behavior.

Finally, we observe that increasing the depth of HD-VAE consistently improves on the log-likelihood, with a limit of three layers latent layers.

The relaxation bias (section 2) may increase with the number of discrete latent variables.

We trained HD-VAE for different numbers of stochastic units and different depths on Binarized MNIST using the surrogate objective defined in the equation 5.

We measured the relaxation bias δ τ =0.1 on the test set ( figure 1, table 4) .

The relaxation bias monotonically increases with the total number of discrete latent variables for different numbers of latent variables.

This may explain why we found that HD-VAE with a large number of latent variables is not yet competitive with the Gaussian counterparts.

In this preliminary research, we have introduced a design for variational memory layers and shown that it can be exploited to build hierarchical discrete VAEs, that outperform Gaussian prior VAEs.

However, without explicitly constraining the model, the relaxation bias grows with the number of latent layers, which prevents us from building deep hierarchical models that are competitive with state-of-the-art methods.

In future work we will attempt to harness the relaxed-ELBO to improve the performance of the HD-VAE further.

Optimization During training, we mitigate the posterior collapse using the freebits (Kingma et al., 2016) strategy with λ = 2 for each stochastic layer.

A dropout of 0.5 is used to avoid overfitting.

We linearly decrease the temperature τ from 0.8 to 0.3 during the first 2 · 10 5 steps and from 0.3 to 0.1 during the next 2 · 10 5 steps.

We use the Adamax optimizer (Kingma and Ba, 2014) with initial learning rate of 2 · 10 −3 for all parameters except for the memory values that are trained using a learning rate of 2 · 10 −2 to compensate for sparsity.

We use a batch size of 128.

All models are trained until they overfit and we evaluate the log-likelihood using 1000 importance weighted samples (Burda et al., 2015) .

Despite its large number of parameters, HD-VAE seems to be more robust to overfitting, which may be explained by the sparse update of the memory values.

Runtime Sparse CUDA operations are currently not used, which means there is room to make HD-VAE more memory efficient.

Even during training, one may truncate the relaxed samples to benefit from the sparse optimizations.

The table 3 shows the average elapsed time training iteration as well as the memory usage for a 6 layers LVAE with 6 × 16 stochastic units and K = 16 2 and batch size of 128.

Table 4 : Measured one-importance-weighted ELBO on binarized MNIST for a LVAE model with different number of layers and different numbers of stochastic units using relaxed (τ = 0.1) and hard samples (τ = 0).

We report N = L l=1 n l , where n l relates to the number of latent variables at the layer l and we set K = 256 for all the variables.

Let x be an observed variable, and consider a VAE model with one layer of N categorical latent variables z = {z 1 , . . .

, z N } each with K classes.

The generative model is p θ (x, z) and the inference model is q φ (z|x).

For a temperature parameter τ > 0, the equivalent relaxed concrete variables are denotedẑ = {ẑ 1 , . . .

,ẑ N },ẑ i ∈ [0, 1] K .

We define H = one hot • arg max and

Following Tucker et al. (2017), using the Gumbel-Max trick, one can notice that

We now assume that f θ,φ,x is κ-Lipschitz for L 2 .

Then, by definition,

The relaxation bias can therefore be bounded as follows:

Furthermore, we can define the adjusted Evidence Lower Bound for relaxed categorical variables (relaxed-ELBO):

As shown by the experiment presented in the section 4.2, the quantity L τ >0

1 (θ, φ) appears to be a positive quantity.

Furthermore, as the model attempts to exploit the relaxation of z to maximize the surrogate objective, one may consider that

is a tight bound of δ τ (θ, φ), meaning that the relaxed-ELBO is a tight lower bound of the ELBO.

The relaxed-ELBO is differentiable and may enable automatic control of the temperature as left and right terms of the relaxed-ELBO seek respectively seek for high and low temperature.

κ-Lipschitz neural networks can be designed using Weight Normalization (Salimans and Kingma, 2016) or Spectral Normalization (Miyato et al., 2018) .

Nevertheless handling residual connections and multiple layers of latent variables is not trivial.

We note however that in the case of a one layer VAE, one only needs to constrain the VAE decoder to be κ-Lispchitz as the surrogate objective is computed as

In the appendix E, we show how the relaxed-ELBO can be extended to multiple layers of latent variables in the LVAE setting.

Appendix D. Defining f θ,φ on the domain of the relaxed Categorical Variablesz f θ,φ is only defined for categorical samples.

For relaxed samplesz, we define f θ,φ as:

.

The introduction of the function H is necessary as the terms (b) and (c) are only defined for categorical samples.

This expression remains valid for hard samplesz.

During training, relaxing the expressions (b) and (c) can potentially yield gradients of lower variance.

In the case of a single categorical variable z described by the set of K class probabilities π = {π 1 , ...π K }.

One can define:

Alternatively, asides from being a relaxed Categorical distribution, the Concrete/GumbelSoftmax also defines a proper continuous distribution.

When treated as such, this results in a proper probabilistic model with continuous latent variables, and the objective is unbiased.

In that case, the density is given by

We consider now an LVAE model:

In the following, we will leave the conditioning on x implicit for convenience.

The ELBO estimated with relaxed samples (relaxed-ELBO) is:

The correct ELBO can be rewritten as follows:

In this section we define the VAE (Rezende et al., 2014; Kingma et al., 2016; Dieng et al., 2018) , the LVAE (Sønderby et al., 2016) and BIVA (Maaløe et al., 2019) .

All models are characterized by a generative model p θ (x, z) = p θ (x|z)p θ (z) and can be coupled with any variational distribution.

Variational Autoencoder (VAE)

Variational Autoencoder with Skip-Connections (Skip-VAE)

Ladder Variational Autoencoder (LVAE)

Ladder Variational Autoencoder with Skip-Connections (Skip-LVAE)

Bidirectional Variational Autoencoder (BIVA)

<|TLDR|>

@highlight

In this paper, we introduce a discrete hierarchy of categorical latent variables that we train using the Concrete/Gumbel-Softmax relaxation and we derive an upper bound for the absolute difference between the unbiased and the biased objective.