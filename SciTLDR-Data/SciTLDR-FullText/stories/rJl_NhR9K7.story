Recent work has shown increased interest in using the Variational Autoencoder (VAE) framework to discover interpretable representations of data in an unsupervised way.

These methods have focussed largely on modifying the variational cost function to achieve this goal.

However, we show that methods like beta-VAE simplify the tendency of variational inference to underfit causing pathological over-pruning and over-orthogonalization of learned components.

In this paper we take a complementary approach: to modify the probabilistic model to encourage structured latent variable representations to be discovered.

Specifically, the standard VAE probabilistic model is unidentifiable: the likelihood of the parameters is invariant under rotations of the latent space.

This means there is no pressure to identify each true factor of variation with a latent variable.

We therefore employ a rich prior distribution, akin to the ICA model, that breaks the rotational symmetry.

Extensive quantitative and qualitative experiments demonstrate that the proposed prior mitigates the trade-off introduced by modified cost functions like beta-VAE and TCVAE between reconstruction loss and disentanglement.

The proposed prior allows to improve these approaches with respect to both disentanglement and reconstruction quality significantly over the state of the art.

Recently there has been an increased interest in unsupervised learning of disentangled representations.

The term disentangled usually describes two main objectives: First, to identify each true factor of variation with a latent variable, and second, interpretability of these latent factors (Schmidhuber, 1992; Ridgeway, 2016; BID0 .

Most of this recent work is inspired by the β-VAE concept introduced in BID11 , which proposes to re-weight the terms in the evidence lower bound (ELBO) objective.

In BID11 a higher weight for the Kullback-Leibler divergence (KL) between approximate posterior and prior is proposed, and putative mechanistic explanations for the effects of this modification are studied in BID4 ; BID5 .

An alternative decomposition of the ELBO leads to the recent variant of β-VAE called β-TCVAE BID5 , which shows the highest scores on recent disentanglement benchmarks.

These modifications of the evidence lower bound however lead to a trade-off between disentanglement and reconstruction loss and therefore the quality of the learned model.

This trade-off is directly encoded in the modified objective: by increasing the β-weight of the KL-term, the relative weight of the reconstruction loss term is more and more decreased.

Therefore, optimization of the modified ELBO will lead to latent encodings which have a lower KL-divergence from the prior, but at the same time lead to a higher reconstruction loss.

Furthermore, we discuss in section 2.4 that using a higher weight for the KL-term amplifies existing biases of variational inference, potentially to a catastrophic extent.

There is a foundational contradiction in many approaches to disentangling deep generative models (DGMs): the standard model employed is not identifiable as it employs a standard normal prior which then undergoes a linear transformation.

Any rotation of the latent space can be absorbed into the linear transform and is therefore statistically indistinguishable.

If interpretability is desired, the modelling choices are setting us up to fail.

We make the following contributions:• We show that the current state of the art approaches employ a trade-off between reconstruction loss and disentanglement of the latent representation.• In section 2.3 we show that variational inference techniques are biased: the estimated components are biased towards having orthogonal effects on the data and the number of components is underestimated.• We provide a novel description of the origin of disentanglement in β-VAE and demonstrate in section 2.4 that increasing the weight of the KL term increases the over-pruning bias of variational inference.• To mitigate these drawbacks of existing approaches, we propose a family of rotationally asymmetric distributions for the latent prior, which removes the rotational ambiguity from the model.

This approach resembles independent component analysis (ICA) for variational autoencoders.• We propose to use a prior which allows a decomposition of the latent space using independent subspace analysis (ISA) and demonstrate that this prior leads to disentangled representations even for the unmodified ELBO objective.

This removes the trade-off between disentanglement and reconstruction loss of existing approaches.• An even higher disentanglement of the latent space can be achieved by incorporating the proposed prior distribution into the existing approaches β-VAE and β-TCVAE.

Since the prior distribution already favours a disentangled representation, the new method dominates previous in terms of the trade-off between disentanglement and model quality.

We briefly discuss previous work on variational inference in deep generative models and two modifications of the learning objective that have been proposed to learn a disentangled representation.

We discuss characteristic biases of variational inference and how the modifications of the learning objective actually accentuate these biases.

Variational Autoencoder Kingma & Welling (2014) introduce a latent variable model that combines a generative model, the decoder, with an inference network, the encoder.

Training is performed by optimizing the evidence lower bound (ELBO) averaged over the empirical distribution: DISPLAYFORM0 where the decoder p θ (x|z) is a deep learning model with parameters θ and each z l is sampled from the encoder z l ∼ q φ (z|x) with variational parameters φ.

When choosing appropriate families of distributions, gradients through the samples z l can be estimated using the reparameterization trick.

The approximate posterior q φ (z|x) is usually modelled as a multivariate Gaussian with diagonal covariance matrix and the prior p(z) is typically the standard normal distribution.β- VAE Higgins et al. (2016) propose to modify the evidence lower bound objective and penalize the KL-divergence of the ELBO: DISPLAYFORM1 where β > 1 is a free parameter that should encourage a disentangled representation.

In BID4 the authors provide further thoughts on the mechanism that leads to these disentangled representations.

However we will show in the following that this parameter introduces a trade-off between reconstruction loss and disentanglement.

Furthermore, we show in section 2.4 that this parameter amplifies biases of variational inference towards orthogonalization and pruning.β-TCVAE In BID5 the authors propose an alternative decomposition of the ELBO, that leads to the recent variant of β-VAE called β-TCVAE.

They demonstrate that β-TCVAE allows to learn representations with higher MIG score than β-VAE BID11 , InfoGAN BID6 and FactorVAE BID17 .

The authors propose to decompose the KL-term in the ELBO objective into three parts and to weight them independently: DISPLAYFORM2 The first term is the index-code mutual information, the second term is the total correlation and the third term the dimension-wise KL-divergence.

Because the index-code mutual information can be viewed as an estimator for the mutual information between p θ (x) and q φ (z), the authors propose to exclude this term when reweighting the KL-term with the β weight.

In addition to the improved objective, the authors propose a quantitative evaluation score for disentanglement, the mutual information gap (MIG).

They propose to first estimate the mutual information between a latent factor and an underlying generative factor of the dataset.

The mutual information gap is then defined as the difference of the mutual information between the highest and second highest correlated underlying factor.

Recent work has shown an increased interest into learning of interpretable representations.

In addition to the work mentioned already, we briefly review some of the influential papers: Chen et al.(2016) present a variant of a GAN that encourages an interpretable latent representation by maximizing the mutual information between the observation and a small subset of latent variables.

The approach relies on optimizing a lower bound of the intractable mutual information.

BID17 propose a learning objective equivalent to β-TCVAE, and train it with the density ratio trick (Sugiyama et al., 2012) .

Kumar et al. (2017) introduce a regularizer of the KL-divergence between the approximate posterior and the prior distribution.

A parallel line of research proposes not to train a perfect generative model but instead to find a simpler representation of the data (Vedantam et al., 2017; BID13 .

A similar strategy is followed in semi-supervised approaches that require implicit or explicit knowledge about the true underlying factors of the data (Kulkarni et al., 2015; Kingma et al., 2014; Reed et al., 2014; BID1 BID12 Zhu et al., 2017; BID10 BID14 BID8 .

There have been several interpretations of the behaviour of the β-VAE BID5 BID4 .

Here we provide a complementary perspective: that it enhances well known statistical biases in VI (Turner & Sahani, 2011) to produce disentangled, but not necessarily useful, representations.

The form of these biases can be understood by considering the variational objective when written as an explicit lower-bound: the log-likelihood of the parameters minus the KL divergence between the approximate posterior and the true posterior DISPLAYFORM0 From this form it is clear that VI's estimates of the parameters θ will be biased away from the maximum likelihood solution (the maximizer of the first term) in a direction that reduces the KL between the approximate and true posteriors.

When factorized approximating distributions are used, VI will therefore be biased towards settings of the parameters that reduce the statistical dependence between the latent variables in the posterior.

For example, this will bias learned components towards orthogonal directions in the output space as this reduces explaining away (e.g. in the factor analysis model, VI breaks the degeneracy of the maximum-likelihood solution finding the orthogonal PCA directions, see appendix B.8).

Moreover, these biases often cause components to be pruned out (in the sense that they have no effect on the observed variables) since then their posterior sits at the prior, which is typically factorized (e.g. in an over-complete factor analysis model VI prunes out components to return a complete model, see appendix B.8).

For simple linear models these effects are not pathological: indeed VI is arguably selecting from amongst the degenerate maximum likelihood solutions in a sensible way.

However, for more complex models the biases are more severe: often the true posterior of the underlying model has significant dependencies (e.g. due to explaining away) and the biases can prevent the discovery of some components.

For example, VAEs are known to over-prune BID3 BID7 .

What happens to these biases in the β-VAE generalization when β > 1?

The short answer is that they grow.

This can be understood by considering coordinate ascent of the modified objective.

With θ fixed, optimising q finds a solution that is closer to the prior distribution than VI due to the upweighting of the KL term in 2.

With q fixed, optimization over θ returns the same solution as VI (since the prior does not depend on the parameters θ and so the value of β is irrelevant).

However, since q is now closer to the prior than before, the KL bias in equation 2 will be greater.

These effects are shown in the ICA example in appendix B.8.

VI (β = 1) learns components that are more orthogonal than the underlying ones, but β = 5 prunes out one component entirely and sets the other two to be orthogonal.

This is disentangled, but arguably leads to incorrect interpretation of the data.

This happens even though both methods are initialised at the true model.

Arguably, the β-VAE is enhancing a statistical bug in VI and leveraging this as a feature.

We believe that this can be dangerous, preventing the discovery of the underlying model.

In this section we describe an approach for unsupervised learning of disentangled representations.

Instead of modifying the ELBO-objective, we propose to use certain families of prior distributions p(z), that lead to identifiable and interpretable models.

In contrast to the standard normal distribution, the proposed priors are not rotationally invariant, and therefore allow interpretability of the latent space.

Independent Component Analysis (ICA) seeks to factorize a distribution into non-Gaussian factors.

In order avoid the ambiguities of latent space rotations, a non-Gaussian distribution (e.g. Laplace or Student-t distribution) is used as prior for the latent variables.

Generalized Gaussian Distribution A generalized version of ICA (Lee & Lewicki, 2000; Zhang et al., 2004; Lewicki, 2002; Sinz & Bethge, 2010 ) uses a prior from the family of exponential power distributions of the form DISPLAYFORM0 also called generalized Gaussian, generalized Laplacian or p-generalized normal distribution.

Using p = 2/(1 + κ) the parameter κ is a measure of kurtosis BID2 .

This family of distributions generalizes the normal (κ = 0) and the Laplacian (κ = 1) distribution.

In general we get for κ > 0 leptokurtic and for κ < 0 platykurtic distributions.

The choice of a leptokurtic or platykurtic distribution has a strong influence on how a generative factor of the data is represented by a latent dimension.

Figure 2: Leptokurtic and platykurtic priors encourage different orientations of the encoding of the (x,y) location of a sprite in the dSprites dataset.

A leptokurtic distribution (here the Laplace distribution) has, in two dimensions, contour lines along diagonal directions and expects most of the probability mass around 0.

Because the (x,y) locations in dSprites are distributed in a square, the projection of the coordinates onto the diagonal fits better to the Laplace prior.

A platykurtic distribution however is more similar to a uniform distribution, with axis aligned contour lines in two dimensions.

This fits better to an orthogonal projection of the (x,y) location.

The red and blue colour coding denotes the value of the latent variable for the respective (x,y) location of a sprite.(x,y) spatial location of a sprite in the dSprites dataset BID4 .

The leptokurtic distribution expects most of the probability mass around 0 and therefore favours a projection of the x and y coordinates, which are distributed in a square, onto the diagonal.

The platykurtic prior is closer to a uniform distribution and therefore encourages an axis-aligned representation.

This example shows how the choice of the prior will effect the latent representation.

Obviously the normal distribution is a special instance of the class of L p -spherically symmetric distributions, and the normal distribution is the only L 2 -spherically symmetric distribution with independent marginals.

Equivalently (Sinz et al., 2009a) showed that this also generalizes to arbitrary values of p. The marginals of the p-generalized normal distribution are independent, and it is the only factorial model in the class of L p -spherically symmetric distributions.

We investigate the behaviour of L p -spherically symmetric distributions as prior distributions for p(z) in the experiments in section 4.

ICA can be further generalized to include independence between subspaces, but dependencies within them, by using a more general prior, the family of L p -nested symmetric distributions BID15 BID16 Sinz et al., 2009b; Sinz & Bethge, 2010) .

DISPLAYFORM0 To start, let's consider functions of the form DISPLAYFORM1 with p 0 , p 1 ∈ R. This function is a cascade of two L p -norms.

To aid intuition we provide a visualization of this distribution in figure 1a, which depicts (6) as a tree that visualizes the nested structure of the norms.

We call the class of functions which employ this structure L p -nested.

BID9 as DISPLAYFORM2 DISPLAYFORM3 where S f (1) is the surface area of the L p -nested sphere.

This surface area can be obtained by using the gamma function: DISPLAYFORM4 where l i is the number of children of a node i, n i is the number of leaves in a subtree under the node i, and n i,k is the number of leaves in the subtree of the k-th children of node i.

For further details we refer the reader to the excellent work of Sinz & Bethge (2010) .

The family of L p -nested distributions allows a generalization of ICA called independent subspace analysis (ISA).

ISA uses a subclass of L p -nested distributions, which are defined by functions of the form DISPLAYFORM0 and correspond to trees of depth two.

The tree structure of this subclass of functions is visualized in FIG1 where each v i , i = 1, . . .

, l 0 denotes the function value of the L p -norm evaluated over a node's children.

The components z j of z that contribute to each v i form a subspace DISPLAYFORM1 Sinz & Bethge FORMULA0 showed that the subspaces V 1 , . . .

, V l0 become independent when using the radial distribution DISPLAYFORM2 which we can interpret as a generalization of the Chi-distribution.

ISA-VAE We propose to choose the latent prior p ISA (z) (Eq. 7) with f (z) from the family of ISA models of the form of Eq. 9, which allows us to define independent subspaces in the latent space.

The Kulback-Leibler divergence of the ELBO-objective can be estimated by Monte-Carlo sampling.

This leads to an ELBO-objective of the form DISPLAYFORM3 which only requires to compute the log-density of the prior that is readily accessible from the density defined in Eq. 7.

As discussed in Roeder et al. (2017) this form of the ELBO even has potential advantages (variance reduction) in comparison to a closed form KL-divergence.

ISA-TCVAE The proposed latent prior can also be combined with the β-TCVAE approach and we get the objective DISPLAYFORM4 where I q denotes the index code mutual information.

To compute the terms in Eq. 13, BID5 propose a Monte-Carlo sampling approach called minibatch-weighted sampling, which also only requires to compute the log density of the prior.

If we want to sample from the generative model we have to be able to sample from the prior distribution.

Sinz & Bethge (2010) describe an exact sampling approach to sample from an L p -nested distribution, which we reproduce as Algorithm 1 in the appendix.

Note that during training we only have to sample from the approximate posterior q φ , which we do not have to modify and which can remain a multivariate Gaussian distribution following the original VAE approach.

As a consequence, the reparameterization trick can be applied (Kingma & Welling, 2014) .Experiments in the following section demonstrate that the proposed prior supports unsupervised learning of disentangled representation even for the unmodified ELBO objective (β = 1).

In our experiments, we evaluate the influence of the proposed prior distribution on disentanglement and on the quality of the reconstruction on the dSprites dataset BID4 , which contains images of three different shapes undergoing transformations of their position, scale and rotation.

Figure 3: Disentangled representations for ISA-VAE and ISA-TCVAE on the dSprites dataset.

We follow standard practice established in BID5 for visualizing latent representations and additionally show images generated by traversals of the latent along the respective axis.

The red and blue colour coding in the first column denotes the value of the latent variable for the respective x,ycoordinate of the sprite in the image.

Coloured lines indicate the object shape with red for ellipse, green for square, and blue for heart.

(a) Even without a modification of the ELBO (β = 1.0) the proposed ISA prior leads to a disentangled representation.

(b) When combining the ISA-model with β-TCVAE, a model with a high disentanglement score of MIG = 0.54 can be reached.

This is the highest score reported for dSprites in the literature.

ISA-layouts: (a) l 0 = 5, l 1,...,5 = 5, p 0 = 2.1, DISPLAYFORM0 Disentanglement Metrics To provide a quantitative evaluation of the disentanglement we compute the disentanglement metric Mutual Information Gap (MIG) that was proposed in BID5 .

The MIG score measures how much mutual information a latent dimension shares with the underlying factor, and how well this latent dimension is separated from the other latent factors.

Therefore the MIG measures the two desired properties usually referred to with the term disentanglement: a factorized latent representation, and interpretability of the latent factors.

BID5 compare the MIG metric to existing disentanglement metrics BID11 BID17 and demonstrate that the MIG is more effective and that the other metrics do not allow to capture both properties in a desirable way.

Reconstruction Quality To quantify the reconstruction quality, we report the expected (log-)likelihood of the reconstructed data E q φ (z|x) [log p θ (x|z)].

In our opinion this measure is more informative than the ELBO, frequently reported in existing work, e.g. BID5 , especially when varying the β parameter, the weighting of the KL term, which is part of the ELBO.

BID5 demonstrate that β-TCVAE, a modification of the β-VAE, enables learning of representations with higher MIG score than β-VAE BID11 , InfoGAN BID6 , and FactorVAE BID17 .

Therefore we choose to compare against β-TCVAE and β-VAE in our experiments.

To allow an accurate comparison we use the same architecture for the decoder and encoder as presented in BID5 .

We reproduce the description of the encoder and decoder in appendix A.5Choosing the ISA-layout We follow the practice of Sinz et al. (2009b)

First, we investigate the ability of the prior to support unsupervised learning of disentangled representations for the unmodified ELBO-objective.

Figure 3a depicts the structure of the latent representation after training for ISA-VAE, a combination of the L p -nested ISA prior with the standard VAE approach.

Because our prior allows independent subspaces the latent space becomes interpretable even when using the unmodified ELBO objective with β = 1.

The plots were produced with the reference implementation of BID5 for visualizing latent representations for the dSprites dataset.

When combining the ISA-model with β-TCVAE and varying the β parameter, a model with a high disentanglement score of MIG = 0.54 can be reached.

This is the so far highest score reported for dSprites in the literature.

This benefit of the proposed prior, that it encourages disentangled representations becomes even more obvious in our quantitative comparison.

We compare the approaches ISA-VAE and ISA-TCVAE, that use the proposed L p -nested prior p ISA with their respective counterpart β-VAE (Higgins et al., 2016) and β-TCVAE BID5 that use the standard normal prior p N .

Because the amount of disentanglement depends on the choice of the parameter β, we vary β in the interval between 1 and 6 with a stepsize of 0.5.

We compare the performance of the four different approaches in FIG4 with 16 experiments for each value of β.

Clearly in both cases a higher disentanglement score can be achieved already for smaller values of β with ISA-β-VAE and ISA-β-TCVAE in comparison to the original approaches.

Even when choosing the individually best value of β, that reaches the highest MIG score for each method, the poposed approaches reach a higher MIG score than their respective counterparts.

of β.

ISA-β-VAE and ISA-β-TCVAE reach high values of the disentanglement score for smaller values of β which and at the same time preserves a higher quality of the reconstruction than the respective original approaches.

At the same time we observe that with the proposed prior the quality of the reconstruction decreases at a smaller rate than for the original approaches.

This improvement of the trade-off between disentanglement and the reconstruction loss becomes also obvious in figure 5b where we plot the MIG score with respect to the reconstruction loss for the individually best value of β.

The proposed approaches ISA-VAE and ISA-TCVAE allow higher MIG scores than their respective base lines while at the same time providing a better reconstruction quality, almost reaching the reconstruction quality of the standard, non-disentangling VAE.

Interestingly the plot also shows that the increase of the MIG score for the baseline method β-TCVAE comes at the cost of a much lower reconstruction quality.

This difference in the reconstruction quality becomes obvious in the quality of the reconstructed images.

Please refer to the appendix where we present latent traversals in appendix A.3 and image reconstruction experiments in appendix A.4.

With the proposed approach ISA-TCVAE the reconstruction quality can be increased significantly while at the same time providing a higher disentanglement.

We presented a structured prior for unsupervised learning of disentangled representations in deep generative models.

We choose the prior from the family of L p -nested symmetric distributions which enables the definition of a hierarchy of independent subspaces in the latent space.

In contrast to the standard normal prior that is often used in training of deep generative models the proposed prior is not rotationally invariant and therefore enhances the interpretability of the latent space.

We demonstrate in our experiments, that a combination of the proposed prior with existing approaches for unsupervised learning of disentangled representations allows a significant improvement of the trade-off between disentanglement and reconstruction loss.

We vary l 0 between 4 and 10 and choose the same value for l 1 = l 2,...,l0 between 2 and 10.

We set the parameter range of the exponents p i to p i ∈ [0.9, 2.4] with a discretization step size of 0.1, which includes lepto-and platokyrtic distributions.

Fig. 2 depicts how lepto-and platykurtic distributions at the child subspaces lead to different representations of the x and y coordinate.

Because the MIG metric evaluates axis-alignment of the latent dimensions to the underlying factors, here the x and y coordinate, platykurtic priors in general achieve a higher MIG score.

The child subspaces share the same parameter p 1 = p 2,...,l0 and we choose the exponent of the root node as p 0 = p 1 to ensure independence of the subspaces.

To study the influence of the layout on the reconstruction quality and MIG score we compare the results for different values of p 0 , p 1,...,5 and l 1 , and vary the value of β in the interval β ∈ [1, 4] with a step size of 0.5 and repeat each experiment four times.

We compare four layouts with the highest MIG score for each subspace layout in FIG6 where we plot the mean and standard error of MIG score and reconstruction loss.

For this dataset, the confguration p 0 = 2.1, p 1,...,5 = 2.2 and l 1 = 4 (denoted in black) is most appropriate as it achieves high MIG scores while maintaining a good reconstruction quality, both for the ISA-VAE and the ISA-TCVAE model.

DISPLAYFORM0 2.

For each inner node i of the tree associated with f , sample the auxiliary variable s i from a Dirichlet distribution Dir 5.

Sample a new radiusṽ 0 from the radial distribution of the target radial distribution ψ 0 and obtain the sample viax =ṽ 0 · u 6.

Multiply each entry x i ofx by and independent sample z i from the uniform distribution over {−1, 1}.

We follow standard practice established in BID5 for visualizing latent representations and additionally show images generated by traversals of the latent along the respective axis.

The red and blue color coding in the first column denotes the value of the latent variable for the respective x,y-coordinate of the sprite in the image.

Colored lines indicate the object shape with red for ellipse, green for square, and blue for heart. : Reconstructed images for β-VAE, β-TCVAE, ISA-VAE and ISA-TCVAE using the models from figure 5 and appendix A.3.

Note that because of the trade-off between disentanglement and reconstruction loss the images reconstructed with β-VAE and β-TCVAE appear much noisier than the models with the ISA prior.

Further, the ISA prior allows to reconstruct more details of the heart shape than β-VAE and β-TCVAE.

The models were trained with the optimization algorithm Adam (Kingma & Ba, 2015) using a learning rate parameter of 0.01All unmentioned hyperparameters are PyTorch v0.41 defaults.

i n i t ( ) s e l f .

n e t = nn .

S e q u e n t i a l ( nn .

L i n e a r ( i n p u t d i m , 1 2 0 0 ) , nn .

Tanh ( ) , nn .

L i n e a r ( 1 2 0 0 , 1 2 0 0 ) , nn .

Tanh ( ) , nn .

L i n e a r ( 1 2 0 0 , 1 2 0 0 ) , nn .

Tanh ( ) , nn .

L i n e a r ( 1 2 0 0 , 4 0 9 6 ) ) d e f f o r w a r d ( s e l f , z ) : h = z .

view ( z . s i z e ( 0 ) , −1) h = s e l f .

n e t ( h ) mu img = h .

view ( z . s i z e ( 0 ) , 1 , 6 4 , 6 4 ) r e t u r n mu imgArchitecture of the encoder and decoder which is identical to the architecture in BID5 .

This section provides the details of the toy examples that reveal the biases in variational methods.

First we will consider the factor analysis model showing that VI breaks the degeneracy of the maximum-likelihood solution to 1) discover orthogonal weights that lie in the PCA directions, 2) prune out extra components in over-complete factor analysis models, even though there are solutions with the same likelihood that preserve all components.

We also show that in these examples the β-VI returns identical model fits to VI regardless of the setting of β.

Second, we consider an over-complete ICA model and initialize using the true model.

We show that 1) VI is biased away from the true component directions towards more orthogonal directions, and 2) β-VI with a modest setting of β = 5 prunes away one of the components and finds orthogonal directions for the other two.

That is, it finds a disentangled representation, but one which does not reflect the underlying components.

The β-VAE optimizes the modified free-energy, F β (q(z 1:N ), θ), with respect to the parameters θ and the variational approximation q(z 1:N ), DISPLAYFORM0 Consider the case where M = 1 β is a positive integer, M ∈ N, we then have DISPLAYFORM1 In this case, the β-VAE can be thought of as attaching M replicated observations to each latent variable z n and then running standard variational inference on the new replicated dataset.

This can equivalently be thought of as raising each likelihood p(x n |z n , θ) to the power M .

Now in real applications β will be set to a value that is greater than one.

In this case, the effect of β is the opposite: it is to reduce the number of effective data points per latent variable to be less than one M < 1.

Or equivalently we raise each likelihood term to a power M that is less than one.

Standard VI is then run on these modified data (e.g. via joint optimization of q and θ).Although this view is mathematically straightforward, the perspective of the β-VAE i) modifying the dataset, and ii) applying standard VI, is useful as it will allow us to derive optimal solutions for the variational distribution q(z) in simple cases like the factor analysis model considered next.

Consider the factor analysis generative model.

Let x ∈ R L and z ∈ R K .for n = 1...

N z n ∼ N (0, I), DISPLAYFORM0 The true posterior is a Gaussian p(z n |x n , θ) = N (z; µ z|x , Σ z|x ) where DISPLAYFORM1 The true log-likelihood of the parameters is DISPLAYFORM2 Here we have defined the empirical mean and covariance of the observations µ x = 1 N N n=1 x n and The posterior is again Gaussian p(z n |x n , θ, M (β)) = N (z n ;μ z|x (β, n),Σ z|x (β)) wherẽ DISPLAYFORM3 DISPLAYFORM4 Here we have taken care to explicitly reveal all of the direct dependencies on β.

Mean-field variational inference, q(z n ) = k q n,k (z k,d ), will return a diagonal Gaussian approximation to the true posterior with the same mean and matching diagonal precision, DISPLAYFORM5 We notice that the posterior mean is a linear combination of the observationsμ z|x (β, n) = R(β)x n where R(β) =Σ z|x (β)M (β)W D −1 are recognition weights.

Notice that the recognition weights and the posterior variances are the same for all data points: they do not depend on n.

The free-energy is then DISPLAYFORM6 with the reconstruction term being DISPLAYFORM7 and the KL or regularization term being DISPLAYFORM8 We will now consider the objective functions and the posterior distributions in several cases to reason about the parameter estimates arising from the methods above.

Consider the situation where we know a maximum likelihood solution of the weights W ML .

For simplicity we select the solution W ML which has orthogonal weights in the observation space.

We then rotate this solution by an amount θ so that W ML = R(θ)W ML .

The resulting weights are no longer orthogonal (assuming the rotation is not an integer multiple of π/2).

We compute the loglikelihood (which will not change) and the free-energy (which will change) and plot the true and approximate posterior covariance (which does not depend on the datapoint value x n ).First here are the weights are aligned with the true ones.

The log-likelihood and the free-energy take the same value of -17.82 nats.

When varying the rotation away fom the orthogonal setting, θ, the plots above indicate that orthogonal settings of the weights (θ = mπ/2 where m = 0, 1, 2, ...) lead to factorized posteriors.

In these cases the KL between the approximate posterior and the true posterior is zero and the free-energy is equal to the log-likelihood.

This will be the optimal free-energy for any weight setting (due to the fact that it is equal to the true log-likelihood which is maximal, and the free-energy is a lower bound of this quantity.)

For intermediate values of θ the posterior is correlated and the free-energy is not tight to the log likelihood.

Now let's plot the free-energy and the log-likelihood as θ is varied.

This shows that the free-energy prefers orthogonal settings of the weights as this leads to factorized posteriors, even though the loglikelihood is insensitive to θ.

So, variational inference recovers the same weight directions as the PCA solution.

The above shows that the bias inherent in variational methods will cause them to break the symmetry in the log-likelihood and find orthorgonal latent components.

This occurs because orthoginal components result in posterior distributions that are factorized.

These are then well-modelled by the variational approximation and result in a small KL between the approximate and true posteriors.

A similar effect occurs if we model 2D data with a 3D latent space.

Many settings of the weights attain the maximum of the likelihood, including solutions which use all three latent variables.

However, the optimal solution for VI is to retain two orthogonal components and to set the magnitude of the third component to zero.

This solution a) returns weights that maximise the likelihood, and b) has a factorised posterior distribution (the pruned component having a posterior equal to its prior) that therefore incurs no cost KL(q(z)||p(z|x, θ)) = 0.

In this way the bound becomes tight.

Here's an example of this effect.

We consider a model of the form: DISPLAYFORM0 We set α 2 + β 2 = 1 so that all models imply the same covariance and set this to be the maximum likelihood covariance by construction.

We then consider varying α from 0 to 1/2.

The setting equal to 0 attains the maximum of the free-energy, even though it has the same likelihood as any other setting.

In this example, changing β in this example just reduces the amplitude of the fluctuations in the free-energy, but it does not change the directions found.

A similar observation applies to the pruning experiment.

Increasing β will increase the uncertainty in the posterior as it is like reducing the number of observations (or increasing the observation noise, from the perspective of q).

The behaviours introduced by the β-VAE appear relatively benign, and perhaps even helpful, in the linear case: VI is breaking the degeneracy of the maximum likelihood solution in a sensible way: selecting amongst the maximum likelihood solutions to find those that have orthogonal components and removing spurious latent dimensions.

This should be tempered by the fact that the β generalization recovered precisely the same solutions and so it was necessary to obtain the desired behaviour in the PCA case.

Similar effects will occur in deep generative models, not least since these typically also have a Gaussian prior over latent variables, and these latents are initially linearly transformed, thereby resulting in a similar degeneracy to factor analysis.

However, the behaviours above benefited from the fact that maximum-likelihood solutions could be found in which the posterior distribution over latent variables factorized.

In real world examples, for example in deep generative models, this will not be case.

In such cases, these same effects will cause the variational free-energy and its β-generalization to bias the estimated parameters far away from maximum-likelihood settings, toward those settings that imply factorized Gaussian posteriors over the latent variables.

We now apply VI and the β free-energy method to ICA.

We're interested the properties of the variational objective and the β-VI objective and so we 1. fit the data using the true generative model to investigate the biases in VI and β-VI 2.

do not use amortized inference, just optimizing the approximating distributions for each data point (this is possible for these small examples).The linear independent component analysis generative model we use is defined as follows.

Let x ∈ R L and z ∈ R K .Under review as a conference paper at ICLR 2019 for n = 1...

N for k = 1...

K z n,k ∼ Student-t(0, σ, v), We apply mean-field variational inference, q(z n ) = k q n,k (z k,d ), and use Gaussian distributions for each factor q n,k (z n,k ) = N (z n,k ; µ n,k , σ 2 n,k ).

The free-energy is computed as follows: The reconstruction term is identical to PCA: an avergage of a quadratic form wrt to a Gaussian, which is analytic.

The KL is broken down into the differential entropy of q which is also analytic and the cross-entropy with the prior which we evaluate by numerical integration (finite differences).

There is a cross-entropy term for each latent variable which is one reason why the code is slow (requiring N 1D numerical integrations).

The gradient of the free-energy wrt the parameters W and the means and variances of the Gaussian q distributions are computed using autograd.

In order to be as certain as possible that we are finding a global maximum of the free-energies, all experiments initialise at the true value of the parameters and then ensure that each gradient step improves the free-energy.

Stochastic optimization or a procedure that accepted all steps regardless of the change in the objective would be faster, but they might also move us into the basis of attraction of a worse (local) optima.

Now we define the dataset.

We use a very sparse Student's t-distribtion with v = 3.5.

For v < 4 the the kurtosis is undefined so the model is fairly simple to estimate model (it's a long way away from the degenerate factor analysis case which is recovered in the limit v → ∞).We use three latent components and a two dimensional observed space.

The directions of the three weights are shown in blue below with data as blue circles.

First we run variational inference finding components (shown in red below) which are more orthogonal than the true directions.

This bias is in this directions as this reduces the dependencies (explaining away) in the underlying posterior.

In this case the bias is so great that the true component directions are not discovered.

Instead the components are forced into the orthogonal setting regardless of the structure in the data.

The ICA example illustrates that this approach -of relying on a bias inherent in VI to discover meaningful components -will sometimes return meaningful structure (e.g. in the PCA experiments above).

However it does not seem to be a sensible way of doing so in general.

For example, explaining away often means that the true components will be entangled in the posterior, as is the case in the ICA example, and the variational bias will then move us away from this solution.

The β-VI generalisation only enhances this undesirable bias.

@highlight

We present structured priors for unsupervised learning of disentangled representations in VAEs that significantly mitigate the trade-off between disentanglement and reconstruction loss.

@highlight

A general framework to use the family of L^p-nested distributions as the prior for the code vector of VAE, demonstrating a higher MIG.

@highlight

The authors point out issues in current VAE approaches and provide a new perspective on the tradeoff between reconstruction and orthogonalization for VAE, beta-VAE, and beta-TCVAE.