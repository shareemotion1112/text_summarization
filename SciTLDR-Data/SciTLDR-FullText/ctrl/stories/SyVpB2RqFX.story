We propose the Information Maximization Autoencoder (IMAE), an information theoretic approach to simultaneously learn continuous and discrete representations in an unsupervised setting.

Unlike the Variational Autoencoder framework, IMAE starts from a stochastic encoder that seeks to map each input data to a hybrid discrete and continuous representation with the objective of maximizing the mutual information between the data and their representations.

A decoder is included to approximate the posterior distribution of the data given their representations, where a high fidelity approximation can be achieved by leveraging the informative representations.

We show that the proposed objective is theoretically valid and provides a principled framework for understanding the tradeoffs regarding informativeness of each representation factor, disentanglement of representations, and decoding quality.

A central tenet for designing and learning a model for data is that the resulting representation should be compact yet informative.

Therefore, the goal of learning can be formulated as finding informative representations about the data under proper constraints.

Generative latent variable models are a popular approach to this problem, where a model parameterized by θ of the form p θ (x) = p θ (x|z)p(z)dz is used to represent the relationship between the data x and the low dimensional latent variable z. The model is optimized by fitting the generative data distribution p θ (x) to the training data distribution p(x), which involves maximizing the likelihood for θ.

Typically, this model is intractable even for moderately complicated functions p θ (x|z) with continuous z. To remedy this issue, variational autoencoder (VAE) BID13 BID19 proposes to maximize the evidence lower bound (ELBO) of the marginal likelihood objective.

However, as was initially pointed out in BID10 , maximizing ELBO also penalizes the mutual information between data and their representations.

This in turn makes the representation learning even harder.

Many recent efforts have focused on resolving this problem by revising ELBO.

Generally speaking, these works fall into two lines.

One of them targets "disentangled representations" by encouraging the statistical independence between representation components BID9 BID12 BID8 BID4 BID7 , while the other line of work seeks to control or encourage the mutual information between data and their representations BID16 BID3 BID1 BID6 Zhao et al., 2017) .

However, these approaches either result in an invalid lower bound for the VAE objective or cannot avoid sacrificing the mutual information.

Instead of building upon the generative latent variable model, we start with a stochastic encoder p θ (z|x) and aim at maximizing the mutual information between the data x and its representations z. In this setting, a reconstruction or generating phase can be obtained as the variational inference of the true posterior p θ (x|z).

By explicitly seeking for informative representations, the proposed model yields better decoding quality.

Moreover, we show that the information maximization objective naturally induces a balance between the informativeness of each latent factor and the statistical independence between them, which gives a more principled way to learn semantically meaningful representations without invalidating ELBO or removing individual terms from it.

Another contribution of this work is proposing a framework for simultaneously learning continuous and discrete representations for categorical data.

Categorical data are ubiquitous in real-world tasks, where using a hybrid discrete and continuous representation to capture both categorical information and continuous variation in data is more consistent with the natural generation process.

In this work, we focus on categorical data that are similar in nature, i.e., where different categories still share similar variations (features).

We seek to learn semantically meaningful discrete representations while maintaining disentanglement of the continuous representations that capture the variations shared across categories.

We show that, compared to the VAE based approaches, our proposed objective gives a more natural yet effective way for learning these hybrid representations.

Recently, there has been a surge of interest in learning interpretable representations.

Among them, β-VAE BID9 ) is a popular method for learning disentangled representations, which modifies ELBO by increasing the penalty on the KL divergence between the variational posterior and the factorized prior.

However, by using large weight for the KL divergence term, β-VAE also penalizes the mutual information between the data and the latent representations more than a standard VAE does, resulting in more severe under utilization of the latent representation space.

Several follow up works propose different approaches to address the limitations of β-VAE.

BID6 BID1 BID3 BID16 propose to constrain the mutual information between the representations and the data by pushing its upper bound, i.e., the KL divergence term in ELBO, towards a progressively increased target value.

However, specifying and tuning this target value can itself be very challenging, which makes this method less practical.

Moreover, this extra constraint results in an invalid lower bound for the VAE objective.

Alternatively, (Zhao et al., 2017) drops the mutual information term in ELBO.

By pushing only the aggregated posterior towards a factorial prior, they implicitly encourage independence across the dimensions of latent representations without sacrificing the informativeness of the representations.

However, simply removing the mutual information term also violates the lower bound of the VAE objective.

Another relevant line of work BID8 BID12 BID4 BID7 seek to learn disentangled representations by explicitly encouraging statistical independence between latent factors.

They all propose to minimize the total correlation term of the latent representations, either augmented as an extra term to ELBO or obtained by reinterpreting or re-weighting the terms in the VAE objective, as a way to encourage statistical independence between the representation components.

In contrast, we show that our information maximization objective inherently contains the total correlation term while simultaneously seeking to maximize the informativeness of each representation factor.

In this paper, we introduce a different perspective to the growing body of the VAE based approaches for unsupervised representation learning.

Starting by seeking informative representations for the data, we follow a more intuitive way to maximize the mutual information between the data and the representations.

Moreover, we augment the continuous representation with a discrete one, which allows more flexibilities to model real world data that are generated from different categories.

We invoke the information maximization principle BID15 BID2 with proper constraints implied by the objective itself to avoid degenerate solutions.

The proposed objective gives a theoretically elegant yet effective way to learn semantically meaningful representations.

Given data x ∈ R d , we consider learning a hybrid continuous-discrete representation, denoted respectively with variables z ∈ R K1 and y ∈ {1, . . .

, K 2 }, using a stochastic encoder parameterized by θ, i.e., p θ (y, z|x).

We seek to learn compact yet semantically meaningful representations in the sense that they should be low dimensional but informative enough about the data.

A natural approach is to maximize the mutual information BID5 ) I θ (x; y, z) between the data and its representations under the constraint K 1 , K 2 d. Here the mutual information between two random variables, e.g., x and z, is defined as DISPLAYFORM0 is the entropy of z and H θ (z|x) = −E p θ (x,z) [log p θ (z|x)] is the conditional entropy of z given x. The mutual information can be interpreted as the decrease in uncertainty of one random variable given another random variable.

In other words, it quantifies how much information one random variable has about the other.

A probabilistic decoder q φ (x|y, z) is adopted to approximate the true posterior p θ (x|y, z), which can be hard to estimate or even intractable.

The dissimilarity between them is optimized by minimizing the KL divergence D KL (p θ (x|y, z)||q φ (x|y, z)).

In summary, IMAE considers the following, DISPLAYFORM1 Given that H(x) is independent of the optimization procedure, we can show that optimizing (1) is equivalent to optimize the following 1 , DISPLAYFORM2 We set β > 0 to balance between maximizing the informativeness of latent representations and maintaining the decoding quality.

The second term is often interpreted as the "reconstruction error" which can be optimized using the reparameterization tricks proposed by BID13 and BID11 for continuous representation z and discrete representation y respectively.

Now we introduce proper method to optimize the first term I θ (x; y, z) in (2).

We first show that I θ (x; y, z) inherently involves two keys terms that quantify the informativeness of each representation factor and the statistical dependence between these factors.

Assuming the conditional distribution of the representation (y, z) given x is factorial, we also assume the marginal distribution of y and z are independent, i.e., DISPLAYFORM0 The first two terms of the RHS quantify how much information each latent factor, i.e., y or z k , carry about the data.

The last term is known as the total correlation of z BID21 , which quantifies the statistical independence between the continuous latent factors and achieves the minimum if and only if they are independent of each other.

As is implied by (3), maximizing I θ (x; y, z) can be conducted by maximizing informativeness of each latent factor while simultaneously promoting statistical independence between the continuous factors.

Various Monte Carlo based sampling strategies have been proposed to optimize the total correlation term BID4 BID7 ; in this work we follow this line (see Appendix B).

Next we proceed by constructing tractable approximations for I θ (x; z k ) and I θ (x; y) respectively.

Without any constraints, the mutual information I θ (x; z k ) between a continuous latent factor and data can be trivially maximized by severely fragmenting the latent space.

To be more precise, consider the following proposition.

While similar results have likely been established in the information theory literature, we include this proposition to motivate our objective design.

DISPLAYFORM0 The equality in (4) is attained if and only if z k is Gaussian distributed, given which we have DISPLAYFORM1 Note here both µ k (x) and σ k (x) are random variables.

The above result implies that z k is more informative about x if it has less uncertainty given x yet captures more variance in data, i.e., σ k (x) is small while µ k (x) disperses within a large space.

However, this can result in discontinuity of z k , where in the extreme case each data sample is associated with a delta distribution in the latent space.

In light of this, we can make what we described above more precise.

A vanishing variance of the conditional distribution p(z k |x) leads to a plain autoencoder that maps each data sample to a deterministic latent point, which can fragment the latent space in a way that each data sample corresponds with a delta distribution in the latent space DISPLAYFORM2 On the other hand, Proposition 1 also implies that controlling the variance σ k (x) to be finite, I θ (x; z k ) will be maximized by pushing µ k (x) towards two extremes (±∞).

To remedy this issue while achieving the upper bound, a natural resolution is to squeeze z k within the domain of a Gaussian distribution with finite mean and variance.

By doing so, we can avoid the degenerate solution while achieving a more reasonable trade-off between enlarging the spread of µ k (x) and maintaining the continuity of z. Therefore, we consider the following as the surrogate for maximizing I θ (x; z k ), DISPLAYFORM3 (6) Here r(z k ) are i.i.d scaled normal distribution with finite variance.

That is, we push each p θ (z k ) towards a Gaussian distribution r(z k ) by minimizing the KL divergence between them.

Unlike the continuous representation, the mutual information I θ (x; y) between a discrete representation and data can be well approximated, given the fact that the cardinality of the space of y is typically low.

To be more specific, given N i.i.d samples {x n } N n=1 of the data, the empirical estimation of I θ (x; y) under the conditional distribution p θ (y|x n ) follows as DISPLAYFORM0 As shown in Proposition 2, with a suitably large batch of samples, the empirical mutual information I θ (x; y) is a good approximation to I θ (x; y).

This enables us to optimize I θ (x; y) in a theoretically justifiable way that is amenable to stochastic gradient descent with minibatches of data.

Proposition 2.

Let y be a discrete random variable that belongs to some categorical class C. Assume the marginal probabilities of the true and the predicted labels are bounded below, i.e. p θ (y), p θ (y) ∈ [1/(CK 2 ), 1] for all y ∈ C with some constant C > 1.

Then for any δ ∈ (0, 1), DISPLAYFORM1 Here N denotes the number of samples used to establish I θ (x; y) according to Eq (7).Therefore, to maximize the mutual information I θ (x; y), we consider the following: max L θ (y) := I θ (x; y).

(9) Maximizing the the mutual information I θ (x; y) provides a natural way to learn discrete categorical representations.

To see this, notice that I θ (x; y) contains two fundamental quantities, the category balance term H θ (y) and the category separation term H θ (y|x).

In other words, maximizing I θ (x; y) trades off uniformly assigning data over categories and seeking highly confident categorical identity for each sample x.

The maximum is achieved if p θ (y|x) is deterministic while the marginal distribution p θ (y) is uniform, that is H θ (y|x) = 0 and H θ (y) = log K 2 .Overall Objective As a summary of (3) (6) and (9), our overall objective is DISPLAYFORM2 The first three terms associate with our information maximization objective, while the last one aims at better approximation of the posterior p θ (x|y, z).

A better balance between these two targets can be achieved by weighting them differently.

One the other hand, the informativeness of each latent factor can be optimized through L θ (z) and L θ (y), while statistically independent latent continuous factors can be promoted by minimizing the total correlation term D KL p(z)||Π K1 k=1 p(z k ) .

Therefore, trade-offs can be formalized regarding the informativeness of each latent factor, disentanglement of the representation, and better decoding quality.

This motivates us to consider the following objective, let β, γ > 0, DISPLAYFORM3

We compare IMAE against various VAE based approaches that are summarized in Figure 1 .

We would like to demonstrate that IMAE can (i) successfully learn a hybrid of continuous and discrete representations, with y matching the intrinsic categorical information y true well and z capturing the disentangled feature information shared across categories; (ii) outperform the VAE based models by achieving a better trade-off between representation interpretability and decoding quality.

We choose the priors r(z) and r(y) to be the isotropic Gaussian distribution and uniform distribution respectively.

Detailed experimental settings are provided in Appendix G. Figure 1 : Summarization of relevant work.

β-VAE modifies ELBO by increasing the penalty on the KL divergence terms.

InfoVAE drops the mutual information terms from ELBO.

JointVAE seeks to control the mutual information by pushing the their upper bounds (the associated KL divergence terms) towards progressively increased values, C y &C z .

We drop the subscripts θ and φ hereafter.

DISPLAYFORM0

We first qualitatively demonstrate that informative representations can yield better interpretability.

For the continuous representation, FIG0 validates Proposition 1 by showing that, with roughly same amount of variance for each latent variable z k , those achieving high mutual information with the data have mean values µ k (x) of the conditional probability p(z k |x) disperse across data samples and variances σ k (x) decrease to small values for all data samples.

As a qualitative evaluation, we traverse latent dimensions corresponding with different levels of I(x, z k ).

As seen in FIG0 (b)-(d), informative variables in the continuous representation have uncovered intuitive continuous factors of the variation in the data, while the factor z 8 has no mutual information with the data and shows no variation.

We observe the same phenomenon for the discrete representation y in FIG0 (e)&(f), which were obtained with two different values of β and γ, where the more informative one discovers matches the natural labels better.

This provides further evidence for that interpretable latent factors can be attained by maximizing the mutual information between the representations and the data.

We set γ = 2β for IMAE.

For each β, we run each method over 10 random initializations.

In this section, we perform quantitative evaluations on MNIST (LeCun and Cortes, 2010), Fashion MNIST (Xiao et al., 2017) and dSprites BID17 .

We show that IMAE achieves better interpretability vs. decoding quality trade-off.

Unsupervised learning of discrete latent factor Before we present our main results, we first describe an assumption that we make on the discrete representations.

For the discrete representation, a reasonable assumption is that the conditional distribution p(y|x) should be locally smooth so that the data samples that are close on their manifold should have high probability of being assigned to the same category BID0 .

This assumption is crucial for using neural networks to learn discrete representations, since it's easy for a high capacity model to learn a non-smooth function p(y|x) that can abruptly change its predictions without guaranteeing similar data samples will be mapped to similar y. To remedy this issue, we adopt the virtual adversarial training (VAT) trick proposed by BID18 and augment L θ (y) as follows: DISPLAYFORM0 The second term of RHS regularizes p θ (y|x) to be consistent within the norm ball of each data sample so as to maintain the local smoothness of the prediction model.

For fair comparison, we augment all four methods with VAT.

As demonstrated in Appendix D, using VAT is essential for all of them except β-VAE to learn interpretable discrete representations.

We start by evaluating different methods on MNIST and Fashion MNIST, for which we train over a range of β values (we set γ = 2β for IMAE).Discrete representations For the discrete representations, by simply pushing the conditional distribution p(y|x) towards the uniform distribution r(y), β-VAE sacrifices the mutual information I(x; y) and hence struggles in learning interpretable discrete representation even with VAT.

As a comparison, InfoVAE performs much better by dropping I(x; y) from ELBO.

For data that are distinctive enough between categories (MNIST), with large β values InfoVAE performs well by uniformly distributing the whole data over categories through minimizing D KL (p(y)||r(y)) while simultaneously encouraging local smoothness with VAT.

However, InfoVAE struggles with less distinctive data (Fashion-MNIST), where it cannot give fairly confident category separation by only DISPLAYFORM0 Figure 4:

For each image, the first row is the digit type learnt by the model, where each entry is obtained by feeding the decoder with the averaged z values corresponding with the learnt y. The second row is obtained by traversing the "angle" latent factor within [−2, 2] on digit 6.

IMAE is capable of uncovering the underlying discrete factor over a wide range of β values.

More interpretable continuous representations can be obtained when the method is capable of learning discrete representations, since less overlap between the mainfolds of each category is induced.requiring local smoothness.

In contrast, IMAE achieves much better performance by explicitly encouraging confident category separation via minimizing the conditional entropy H(y|x), while using VAT to maintain local smoothness so as to prevent overfitting of neural network.

Although JointVAE performs much better than β-VAE by pushing the upper bound of I(x; y) towards a progressively increasing target value C y , we found it can easily get stuck at some bad local optima where I(x; y) is comparatively large while the accuracy is poor.

A heuristic is that once JointVAE enters the local region of a local optima, progressively increasing C y only induces oscillation within that region.

Informativeness, interpretability and decoding quality As illustrated in Figure 1 , by using large β values, β-VAE sacrifices more mutual information between the data and its representations, which in turn (see FIG1 ) results in less informative representations followed by poor decoding quality.

In contrast, the other three methods can remedy this issue to different degrees, and hence attains better trade-off regarding informativeness of latent representations and decoding quality.

Compared to JointVAE and InfoVAE, IMAE is more capable of learning discrete presentations over a wide range of β, γ values, which implies less overlap between the manifolds of different categories is induced.

As a result, IMAE is expected to yield better decoding quality for each category.

Although InfoVAE and JointVAE can also learn comparatively good discrete representations when using large and small β values respectively, the corresponding results of these two regions associate with either poor decoding quality or much lower disentanglement score (see section 4.2.2).

In contrast, IMAE consistently performs well with different hyperparameters, especially in the region of interest where the decoding quality as well as the informativeness of latent representations are good enough.

In this section, we quantitatively evaluate the disentanglement capability of IMAE on dSprites where the ground truth factors of both continuous and discrete representaions are available.

We use the disentanglement metric proposed by BID4 , which is defined in terms of the gap between the top two empirical mutual information of each latent representation factor and a ground truth factor.

The disentanglement score is defined as the weighted average of the gaps.

A high disentanglement score implies that each ground truth factor associates with one single representation factor that is more informative than the others, i.e., the learnt representation factors are more disentangled.4 FIG3 shows that, with large β values, β-VAE penalizes the mutual information too much and this degrades the usefulness of representations.

while all other three methods achieve higher disentanglement score with better decoding quality.

For JointVAE, higher β values push the upper bound of mutual information converges to the prefixed target value, it therefore can maintain more mutual (a) IMAE performs well regarding the disentanglement score vs. decoding quality trade-off, especially in the region of interest where both decoding quality and informativeness of representations are fairly good.(b) Negative correlation between total correlation and disentanglement score.

It also implies that the disentanglement score tends to decrease along with the total correlation if using even larger β, due to the diminishing informativeness of representation factors.

In the extreme case, both total correlation and disentanglement score can degrade to zero.

information between the data and the whole latent representations and give better decoding quality.

However, the disentanglement quality is poor in this region, which implies that simply restricting the overall capacity of the latent representations is not enough for learning disentangled representations.

While InfoVAE yields comparatively better disentanglement score by pushing the marginal joint distribution of the representations towards a factorial distribution harder with large values of β, the associated decoding quality and informativeness of latent representations are both poor.

In contrast, IMAE is capable of achieving better trade-off between the disentanglement score and the decoding quality in the region of interest where the decoding quality as well as the informativeness are fairly good.

We attribute this to the effect of explicitly seeking for statistically independent latent factors by minimizing the total correlation term in our objective.

We have proposed IMAE, a novel approach for simultaneously learning the categorical information of data while uncovering latent continuous features shared across categories.

Different from VAE, IMAE starts with a stochastic encoder that seeks to maximize the mutual information between data and their representations, where a decoder is used to approximate the true posterior distribution of the data given the representations.

This model targets at informative representations directly, which in turn naturally yields an objective that is capable of simultaneously inducing semantically meaningful representations and maintaining good decoding quality, which is further demonstrated by the numerical results.

Unsupervised joint learning of disentangled continuous and discrete representations is a challenging problem due to the lack of prior for semantic awareness and other inherent difficulties that arise in learning discrete representations.

This work takes a step towards achieving this goal.

A limitation of our model is that it pursues disentanglement by assuming or trying to encourage independent scalar latent factors, which may not always be sufficient for representing the real data.

For example, data may exhibit category specific variation, or a subset of latent factors might be correlated.

This motivates us to explore more structured disentangled representations; one possible direction is to encourage group independence.

We leave this for future work.

H. Xiao, K. Rasul, and R. Vollgraf.

Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.S. Zhao, J. Song, and S. Ermon.

Infovae: Information maximizing variational autoencoders.

arXiv preprint arXiv:1706.02262, 2017.

Balance between posterior inference fidelity and information maximization Notice that we can rewrite the mutual information between the data x and its representations as the following, DISPLAYFORM0 It then follows that, DISPLAYFORM1 Since H(x) is independent of the optimization procedure, we have the following, DISPLAYFORM2 where β trade-off the informativeness of the latent representation and generation fidelity.

Decomposition of I θ (x; y, z) Let b = (z, y) denote the joint random variable consisting of the continuous random variable b and discrete random variable y.

Note that I θ (x; y, z) = I θ (x; b) can be written as: DISPLAYFORM3 The second term in Eq (15) has the form: DISPLAYFORM4 where ϑ 1 follows by the assumption that p θ (b|x) is factorial.

For the first term in Eq (15), we have: DISPLAYFORM5 Substituting Eqs FORMULA1 & FORMULA1 into Eq (15) yields the result: DISPLAYFORM6 Since y and z are assumed to be marginally independent, i.e., p θ (y; z) = p θ (y)p θ (z), then DISPLAYFORM7 1 N N n=1 p θ (y|x n ) denote the Monte Carlo estimator of the true probability DISPLAYFORM8 for all x ∈ X , then applying the Hoeffding's inequality for bounded random variables [Theorem 2.2.6, BID20 ] yields, DISPLAYFORM9 Given Eq (26), we first establish the concentration results of the entropy H p θ (y) with respect to the empirical distribution p θ (y).

Assume For all y ∈ C, we have p θ (y), p θ (y) bounded below by 1/(CK 2 ) for some fixed constant C > 1.

This assumption is practical since the distributions of true data and predicted data are approximately uniform and therefore p θ (y), p θ (y) ≈ 1/K 2 for all y ∈ C. Consider the function t log t, with derivative 1 + log t DISPLAYFORM10 (1 + log t)dt DISPLAYFORM11 Summing over C gives DISPLAYFORM12 Let δ = K 2 δ , then Eq (26) together with Eq (28) yield the following, DISPLAYFORM13 Next we are going to bound the divergence between H θ (y|x) and H θ (y|x) which are defined as, DISPLAYFORM14 Note that h log h ∈ [−1/e, 0] for all h ∈ [0, 1], then again applying [Theorem 2.2.6, BID20 ] yields, DISPLAYFORM15 Following the similar arguments as before, let δ = 2 exp −2t 2 e 2 N , then DISPLAYFORM16 Now let δ = K 2 δ , then applying the union bound we have DISPLAYFORM17 hold with probability 1 − δ.

Conclude from Eqs (29) & (32), we have DISPLAYFORM18 hold with probability at least 1 − 2δ.

Computing the marginal distributions of the continuous representations z and z k requires the entire dataset, e.g., DISPLAYFORM0 To scale up our method to large datasets, we propose to estimate based on the minibatch data, e.g., DISPLAYFORM1 Now consider the entropy H(z) of z, which we approximate in the following way, DISPLAYFORM2 We estimate the integral of z by sampling z ∼ p θ (z|x i ) and perform the Monte Carlo approximation.

Although we minimize the unbiased estimator of the lower bound of the KL divergence, the term inside the logarithm is a summation of probability densities of Gaussians.

In particular, we record the distribution of the variances output by our encoder and observe that the mean of the variances of the Gaussians is bounded between 0.2 and 2, which implies that the values of probability densities do not range in a large scale.

Since logarithm is locally affine, we argue that our bound in (34) is tight.

Other quantities involved in our objective function (10) are estimated in a similar fashion.

In VAE, they assume a generative model specified by a stochastic decoder p θ (x|z), taking the continuous representation as an example, and seek an encoder q φ (z|x) as a variational approximation of the true posterior p θ (z|x).

The model is fitted by maximizing the evidence lower bound (ELBO) of the marginal likelihood, DISPLAYFORM0 Here the KL divergence term can be further decomposed as BID10 , DISPLAYFORM1 That is, minimizing the KL divergence also penalizes the mutual information I θ (x; z), thus reduces the amount of information z has about x. This can make the inference task q φ (z|x) hard and lead to poor reconstructions of x as well.

Many recent efforts have been focused on resolving this problem by revising ELBO.

Although approaches differ, it can be summarized as either dropping the mutual information term in Eq (36), or encouraging statistical independence across the dimensions of z by increasing the penalty on the total correlation term extracted from the KL divergence D KL (q φ (z)||r(z)) with respect to q φ (z).

However, these approaches either result in an invalid lower bound for the VAE objective, or cannot avoid minimizing the mutual information I θ (x; z) between the representation and the data.

In contrast, IMAE starts with a stochastic encoder p θ (z|x) and aims at maximizing the mutual information between the data x and the representations z from the very beginning.

By following the constraints which are naturally implied by the objective in order to avoid degenerated solutions, IMAE targets at both informative and statistical independent representations.

On the other hand, in IMAE the decoder q φ (x|z) serves as a variational approximation to the true posterior p θ (x|z).

As we will show in Section 4, being able to learn more interpretable representations allows IMAE to reconstruct and generate data with better quality.

(a) IMAE performs well regarding the disentanglement score vs. decoding quality trade-off, especially in the region of interest where both decoding quality and informativeness of representations are fairly good.(b) Negative correlation between total correlation and disentanglement score.

It also implies that the disentanglement score tends to decrease along with the total correlation if using even larger β, due to the diminishing informativeness of representation factors.

In the extreme case, both total correlation and disentanglement score can degrade to zero.

Training procedure:

• MNIST & Fashion MNIST: We use momentum to train all models.

The initial learning rate is set as 1e-3, and we decay the learning rate by 0.98 every epoch.• dSprites: We use Adam to train all models.

The learning rate is set as 1e-3.

<|TLDR|>

@highlight

Information theoretical approach for unsupervised learning of unsupervised learning of a hybrid of discrete and continuous representations, 