Posterior collapse in Variational Autoencoders (VAEs) arises when the variational distribution closely matches the uninformative prior for a subset of latent variables.

This paper presents a simple and intuitive explanation for posterior collapse through the analysis of linear VAEs and their direct correspondence with Probabilistic PCA (pPCA).

We identify how local maxima can emerge from the marginal log-likelihood of pPCA, which yields similar local maxima for the evidence lower bound (ELBO).

We show that training a linear VAE with variational inference recovers a uniquely identifiable global maximum corresponding to the principal component directions.

We provide empirical evidence that the presence of local maxima causes posterior collapse in deep non-linear VAEs.

Our findings help to explain a wide range of heuristic approaches in the literature that attempt to diminish the effect of the KL term in the ELBO to reduce posterior collapse.

The generative process of a deep latent variable model entails drawing a number of latent factors from an uninformative prior and using a neural network to convert such factors to real data points.

Maximum likelihood estimation of the parameters requires marginalizing out the latent factors, which is intractable for deep latent variable models.

The influential work of BID21 and BID28 on Variational Autoencoders (VAEs) enables optimization of a tractable lower bound on the likelihood via a reparameterization of the Evidence Lower Bound (ELBO) BID18 BID4 .

This has created a surge of recent interest in automatic discovery of the latent factors of variation for a data distribution based on VAEs and principled probabilistic modeling BID15 BID5 BID8 BID13 .Unfortunately, the quality and the number of the latent factors learned is directly controlled by the extent of a phenomenon known as posterior collapse, where the generative model learns to ignore a subset of the latent variables.

Most existing work suggests that posterior collapse is caused by the KL-divergence term in the ELBO objective, which directly encourages the variational distribution to match the prior.

Thus, a wide range of heuristic approaches in the literature have attempted to diminish the effect of the KL term in the ELBO to alleviate posterior collapse.

By contrast, we hypothesize that posterior collapse arises due to spurious local maxima in the training objective.

Surprisingly, we show that these local maxima may arise even when training with exact marginal log-likelihood.

While linear autoencoders BID30 have been studied extensively BID2 BID23 , little attention has been given to their variational counterpart.

A well-known relationship exists between linear autoencoders and PCAthe optimal solution to the linear autoencoder problem has decoder weight columns which span the subspace defined by the principal components.

The Probabilistic PCA (pPCA) model BID32 recovers the principal component subspace as the maximum likelihood solution of a Gaussian latent variable model.

In this work, we show that pPCA is recovered exactly using linear variational autoencoders.

Moreover, by specifying a diagonal covariance structure on the variational distribution we recover an identifiable model which at the global maximum has the principal components as the columns of the decoder.

The study of linear VAEs gives us new insights into the cause of posterior collapse.

Following the analysis of BID32 , we characterize the stationary points of pPCA and show that the variance of the observation model directly impacts the stability of local stationary points -if the variance is too large then the pPCA objective has spurious local maxima, which correspond to a collapsed posterior.

Our contributions include:• We prove that linear VAEs can recover the true posterior of pPCA and using ELBO to train linear VAEs does not add any additional spurious local maxima.

Further, we prove that at its global optimum, the linear VAE recovers the principal components.• We shows that posterior collapse may occur in optimization of marginal log-likelihood, without powerful decoders.

Our experiments verify the analysis of the linear setting and show that these insights extend even to high-capacity, deep, non-linear VAEs.• By learning the observation noise carefully, we are able to reduce posterior collapse.

We present evidence that the success of existing approaches in alleviating posterior collapse depends on their ability to reduce the stability of spurious local maxima.

Probabilistic PCA.

We define the probabilitic PCA (pPCA) model as follows.

Suppose latent variables z ∈ R k generate data x ∈ R n .

A standard Gaussian prior is used for z and a linear generative model with a spherical Gaussian observation model for x: DISPLAYFORM0 The pPCA model is a special case of factor analysis BID3 , which replaces the spherical covariance σ 2 I with a full covariance matrix.

As pPCA is fully Gaussian, both the marginal distribution for x and the posterior p(z|x) are Gaussian and, unlike factor analysis, the maximum likelihood estimates of W and σ 2 are tractable BID32 .Variational Autoencoders.

Recently, amortized variational inference has gained popularity as a means to learn complicated latent variable models.

In these models, the marginal log-likelihood, log p(x), is intractable but a variational distribution, q(z|x), is used to approximate the posterior, p(z|x), allowing tractable approximate inference.

To do so we typically make use of the Evidence Lower Bound (ELBO): DISPLAYFORM1 The ELBO consists of two terms, the KL divergence between the variational distribution, q(z|x), and prior, p(z), and the expected conditional log-likelihood.

The KL divergence forces the variational distribution towards the prior and so has reasonably been the focus of many attempts to alleviate posterior collapse.

We hypothesize that in fact the marginal log-likelihood itself often encourages posterior collapse.

In Variational Autoencoders (VAEs), two neural networks are used to parameterize q φ (z|x) and p θ (x|z), where φ and θ denote two sets of neural network weights.

The encoder maps an input x to the parameters of the variational distribution, and then the decoder maps a sample from the variational distribution back to the inputs.

Posterior collapse.

The most consistent issue with VAE optimization is posterior collapse, in which the variational distribution collapses towards the prior: ∃i s.t.

∀x q φ (z i |x) ≈ p(z i ).

This reduces the capacity of the generative model, making it impossible for the decoder network to make use of the information content of all of the latent dimensions.

While posterior collapse is typically described using the variational distribution as above, one can also define it in terms of the true posterior p(z|x) as: ∃i s.t.

∀x p(z i |x) ≈ p(z i ).3 Related Work BID12 discuss the relationship between robust PCA methods BID6 and VAEs.

In particular, they show that at stationary points the VAE objective locally aligns with pPCA under certain assumptions.

We study the pPCA objective explicitly and show a direct correspondence with linear VAEs.

BID12 show that the covariance structure of the variational distribution may help smooth out the loss landscape.

This is an interesting result whose interactions with ours is an exciting direction for future research.

BID14 motivate posterior collapse through an investigation of the learning dynamics of deep VAEs.

They suggest that posterior collapse is caused by the inference network lagging behind the true posterior during the early stages of training.

A related line of research studies issues arising from approximate inference causing mismatch between the variational distribution and true posterior BID10 BID19 BID16 .

By contrast, we show that local maxima may exist even when the variational distribution matches the true posterior exactly.

Alemi et al. FORMULA0 use an information theoretic framework to study the representational properties of VAEs.

They show that with infinite model capacity there are solutions with equal ELBO and marginal log-likelihood which span a range of representations, including posterior collapse.

We find that even with weak (linear) decoders, posterior collapse may occur.

Moreover, we show that in the linear case this posterior collapse is due entirely to the marginal log-likelihood.

The most common approach for dealing with posterior collapse is to anneal a weight on the KL term during training from 0 to 1 BID5 BID31 BID24 BID15 BID17 .

Unfortunately, this means that during the annealing process, one is no longer optimizing a bound on the log-likelihood.

In addition, it is difficult to design these annealing schedules and we have found that once regular ELBO training resumes the posterior will typically collapse again (Section 5.2).

propose a constraint on the KL term, which they called "free-bits", where the gradient of the KL term per dimension is ignored if the KL is below a given threshold.

Unfortunately, this method reportedly has some negative effects on training stability BID26 .

Delta-VAEs BID26 instead choose prior and variational distributions carefully such that the variational distribution can never exactly recover the prior, allocating free-bits implicitly.

Several other papers have studied alternative formulations of the VAE objective BID27 BID11 BID0 .

BID11 analyze the VAE objective with the goal of improving image fidelity under Gaussian observation models.

Through this lens they discuss the importance of the observation noise.

BID29 point out that due to the diagonal covariance used in the variational distribution of VAEs they are encouraged to pursue orthogonal representations.

They use linearizations of deep networks to prove their results under a modification of the objective function by explicitly ignoring latent dimensions with posterior collapse.

Our formulation is distinct in focusing on linear VAEs without modifying the objective function and proving an exact correspondence between the global solution of linear VAEs and principal components.

BID23 studies the optimization challenges in the linear autoencoder setting.

They expose an equivalence between pPCA and Bayesian autoencoders and point out that when σ 2 is too large information about the latent code is lost.

A similar phenomenon is discussed in the supervised learning setting by BID7 .

BID23 also show that suitable regularization allows the linear autoencoder to exactly recover the principal components.

We show that the same can be achieved using linear variational autoencoders with a diagonal covariance structure.

In this section we compare and analyze the optimal solutions to both pPCA and linear variational autoencoders.

DISPLAYFORM0 Figure 1: Stationary points of pPCA.

A zero-column of W is perturbed in the directions of two orthogonal principal components (µ5 and µ7) and the loss surface (marginal log-likelihood) is shown.

The stability of the stationary points depends critically on σ 2 .

Left: σ 2 is able to capture both principal components.

Middle: σ 2 is too large to capture one of the principal components.

Right: σ 2 is too large to capture either principal component.

We first discuss the maximum likelihood estimates of pPCA and then show that a simple linear VAE is able to recover the global optimum.

Moreover, the same linear VAE recovers identifiability of the principle components (unlike pPCA which only spans the PCA subspace).

Finally, we analyze the loss landscape of the linear VAE showing that ELBO does not introduce any additional spurious maxima.

The pPCA model (Eq. FORMULA0 ) is a fully Gaussian linear model and thus we can compute both the marginal distribution for x and the posterior p(z | x) in closed form: DISPLAYFORM0 DISPLAYFORM1 where DISPLAYFORM2 This model is particularly interesting to analyze in the setting of variational inference as the ELBO can also be computed in closed form (see Appendix C).Stationary points of pPCA We now characterize the stationary points of pPCA, largely repeating the thorough analysis of Tipping & Bishop (1999) (see Appendix A of their paper).The maximum likelihood estimate of µ is the mean of the data.

We can compute W MLE and σ MLE as follows: DISPLAYFORM3 DISPLAYFORM4 Here U k corresponds to the first k principal components of the data with the corresponding eigenvalues λ 1 , . . .

, λ k stored in the k × k diagonal matrix Λ k .

The matrix R is an arbitrary rotation matrix which accounts for weak identifiability in the model.

We can interpret σ DISPLAYFORM5 as the average variance lost in the projection.

The MLE solution is the global optima.

Stability of W MLE One surprising observation is that σ 2 directly controls the stability of the stationary points of the marginal log-likelihood (see Appendix A).

In Figure 1 , we illustrate one such stationary point of pPCA under different values of σ 2 .

We computed this stationary point by taking W to have three principal components columns and zeros elsewhere.

Each plot shows the same stationary point perturbed by two orthogonal eigenvectors corresponding to other principal components.

The stability of the stationary points depends on the size of σ 2 -as σ 2 increases the stationary point tends towards a stable local maxima.

While this example is much simpler than a non-linear VAE, we find in practice that the same principle applies.

Moreover, we observed that the non-linear dynamics make it difficult to learn a smaller value of σ 2 automatically FIG5 ).

We now show that linear VAEs are able to recover the globally optimal solution to Probabilistic PCA.

We will consider the following VAE model, DISPLAYFORM0 where D is a diagonal covariance matrix which is used globally for all data points.

While this is a significant restriction compared to typical VAE architectures, which define an amortized variance for each input point, this is sufficient to recover the global optimum of the probabilistic model.

Lemma 1.

The global maximum of the ELBO objective (Eq. (4)) for the linear VAE (Eq. FORMULA9 ) is identical to the global maximum for the marginal log-likelihood of pPCA (Eq. FORMULA3 ).Proof.

The global optimum of pPCA is obtained at the maximum likelihood estimate of W and σ 2 , which are specified only up to an orthogonal transformation of the columns of W, i.e., any rotation R in Eq. FORMULA7 FORMULA9 is able to recover the global optimum of pPCA only when DISPLAYFORM1 k (which is diagonal) recovers the true posterior at the global optimum.

In this case, the ELBO equals the marginal log-likelihood and is maximized when the decoder has weights W = W MLE .

Since, ELBO lower bounds log-likelihood, then the global maximum of ELBO for the linear VAE is the same as the global maximum of marginal likelihood for pPCA.Full details are given in Appendix C. In fact, the diagonal covariance of the variational distribution allows us to identify the principal components at the global optimum.

Corollary 1.

The global optimum to the VAE solution has the scaled principal components as the columns of the decoder network.

Proof.

Follows directly from the proof of Lemma 1 and Equation 8.Finally, we can recover full identifiability by requiring D = I. We discuss this in Appendix B.We have shown that at its global optimum the linear VAE is able to recover the pPCA solution and additionally enforces orthogonality of the decoder weight columns.

However, the VAE is trained with the ELBO rather than the marginal log-likelihood.

The majority of existing work suggests that the KL term in the ELBO objective is responsible for posterior collapse and so we should ask whether this term introduces additional spurious local maxima.

Surprisingly, for the linear VAE model the ELBO objective does not introduce any additional spurious local maxima.

We provide a sketch of the proof here with full details in Appendix C. Theorem 1.

The ELBO objective does not introduce any additional local maxima to the pPCA model.

Proof. (Sketch) If the decoder network has orthogonal columns then the variational distribution can capture the true posterior and thus the variational objective exactly recovers the marginal log-likelihood at stationary points.

If the decoder network does not have orthogonal columns then the variational distribution is no longer tight.

However, the ELBO can always be increased by rotating the columns of the decoder towards orthogonality.

This is because the variational distribution fits the true posterior more closely while the marginal log-likelihood is invariant to rotations of the weight columns.

Thus, any additional stationary points in the ELBO objective must necessarily be saddle points.

The theoretical results presented in this section provide new intuition for posterior collapse in general VAEs.

Our results suggest that the ELBO objective, in particular the KL between the variational distribution and the prior, is not entirely responsible for posterior collapse -even exact marginal log-likelihood may suffer.

The evidence for this is two-fold.

We have shown that marginal log-likelihood may have spurious local maxima but also that in the linear case the ELBO objective does not add any additional spurious local maxima.

Rephrased, in the linear setting the problem lies entirely with the probabilistic model.

FIG0 The marginal log-likelihood and optimal ELBO of MNIST pPCA solutions over increasing hidden dimension.

Green represents the MLE solution (global maximum), the red dashed line is the optimal ELBO solution which matches the global optimum.

The blue line shows the marginal log-likelihood of the solutions using the full decoder weights when σ 2 is fixed to its MLE solution for 50 hidden dimensions.

In this section we present empirical evidence found from studying two distinct claims.

First, we verified our theoretical analysis of the linear VAE model.

Second, we explored to what extent these insights apply to deep non-linear VAEs.

In FIG0 we display the likelihood values for various optimal solutions to the pPCA model trained on the MNIST dataset.

We plot the maximum log-likelihood and numerically verify that the optimal ELBO solution is able to exactly match this (Lemma 1).

We also evaluated the model with all principal components used but with a fixed value of σ 2 corresponding to the MLE solution for 50 hidden dimensions.

This is equivalent to σ 2 ≈ λ 222 .

Here the log-likelihood is optimal at σ 2 = 50 as expected, but interestingly the likelihood decreases for 300 hidden dimensions -including the additional principal components has made the solution worse under marginal log-likelihood.

We explored how well the analysis of the linear VAEs extends to deep non-linear models.

To do so, we trained VAEs with Gaussian observation models on the MNIST dataset.

This is a fairly uncommon choice of model for this dataset, which is nearly binary, but it provides a good setting for us to investigate our theoretical findings.

FIG3 shows the cumulative distribution of the per-dimension KL divergence between the variational distribution and the prior at the end of training.

We observe that using a smaller value of σ 2 prevents the posterior from collapsing and allows the model to achieve a substantially higher ELBO.

It is possible that the difference in ELBO is due entirely to the change of scale introduced by σ 2 and not because of differences in the learned representations.

To test this hypothesis we took each of the trained models and optimized for σ 2 while keeping all other parameters fixed ( TAB1 .

As expected, the ELBO increased but the relative ordering remained the same with a significant gap still present.

The final model is evaluated on the training set.

We also tuned σ 2 to the trained model and reevaluated to confirm that the difference in loss is due to differences in latent representations.

The role of KL-annealing An alternative approach to tuning σ 2 is to scale the KL term directly by a coefficient, β.

For β < 1 this provides a loose lowerbound on the ELBO but for appropriate choices of β and learning rate, this scheme can be made equivalent to tuning σ 2 .

In this section we explore this technique.

We found that KL-annealing may provide temporary relief from posterior collapse but that if σ 2 is not appropriately tuned then ultimately ELBO training will recover the default solution.

In FIG4 we show the proportion of units collapsed by threshold for several fixed choices of σ 2 when β is annealed from 0 to 1 over the first 100 epochs.

The solid lines correspond to the final model while the dashed line corresponds to the model at 80 epochs of training.

Early on, KL-annealing is able to reduce posterior collapse but ultimately we recover the ELBO solution from FIG3 .After finding that KL-annealing alone was insufficient to prevent posterior collapse we explored KL annealing while learning σ 2 .

Based on our analysis in the linear case we expect that this should work well: while β is small the model should be able to learn to reduce σ 2 .

To test this, we trained the same VAE as above on MNIST data but this time we allowed σ 2 to be learned.

The results are presented in FIG5 .

We trained first using the standard ELBO objective and then again using KL-annealing.

The ELBO objective learns to reduce σ 2 but ultimately learns a solution with a large degree of posterior collapse.

Using KL-annealing, the VAE is able to learn a much smaller σ 2 value and ultimately reduces posterior collapse.

Interestingly, despite significantly differing representations, these two models have approximately the same final training ELBO.

This is consistent with the analysis of BID0 , who showed that there can exist solutions equal under ELBO with differing posterior distributions.

We trained deep convolutional VAEs with 500 hidden dimensions on images from the CelebA dataset (resized to 64x64).

In FIG6 we show the training ELBO for the standard ELBO objective and training with KL-annealing.

In each case, σ 2 is learned online.

As in FIG5 , KL-Annealing enabled the VAE to learn a smaller value of σ 2 which corresponded to a better final ELBO value and reduced posterior collapse FIG7 ).

By analyzing the correspondence between linear VAEs and pPCA we have made significant progress towards understanding the causes of posterior collapse.

We have shown that for simple linear VAEs posterior collapse is caused by spurious local maxima in the marginal log-likelihood and we demonstrated empirically that the same local maxima seem to play a role when optimizing deep non-linear VAEs.

In future work, we hope to extend this analysis to other observation models and provide theoretical support for the non-linear case.

Here we briefly summarize the analysis of BID32 with some simple additional observations.

We recommend that interested readers study Appendix A of BID32 for the full details.

We begin by formulating the conditions for stationary points of xi log p(x i ): DISPLAYFORM0 Where S denotes the sample covariance matrix (assuming we set µ = µ M LE , which we do throughout), and C = WW T + σ 2 I (note that the dimensionality is different to M).

There are three possible solutions to this equation, (1) W = 0, (2) C = S, or (3) the more general solutions.

FORMULA0 and (2) are not particularly interesting to us, so we focus herein on (3).We can write W = ULV T using its singular value decomposition.

Substituting back into the stationary points equation, we recover the following: DISPLAYFORM1 Noting that L is diagonal, if the j th singular value (l j ) is non-zero, this gives Su j = (σ 2 +l 2 j )u j , where u j is the j th column of U. Thus, u j is an eigenvector of S with eigenvalue DISPLAYFORM2 Thus, all potential solutions can be written as, W = U q (K q − σ 2 I) 1/2 R, with singular values written as k j = σ 2 or σ 2 + l 2 j and with R representing an arbitrary orthogonal matrix.

From this formulation, one can show that the global optimum is attained with σ 2 = σ 2 M LE and U q and K q chosen to match the leading singular vectors and values of S.

Consider stationary points of the form, W = U q (K q − σ 2 I) 1/2 where U q contains arbitrary eigenvectors of S. In the original pPCA paper they show that all solutions except the leading principal components correspond to saddle points in the optimization landscape.

However, this analysis depends critically on σ 2 being set to the true maximum likelihood estimate.

Here we repeat their analysis, considering other (fixed) values of σ 2 .We consider a small perturbation to a column of W, of the form u j .

To analyze the stability of the perturbed solution, we check the sign of the dot-product of the perturbation with the likelihood gradient at w i + u j .

Ignoring terms in 2 we can write the dot-product as, DISPLAYFORM0 Now, C −1 is positive definite and so the sign depends only on λ j /k i − 1.

The stationary point is stable (local maxima) only if the sign is negative.

If k i = λ i then the maxima is stable only when λ i > λ j , in words, the top q principal components are stable.

However, we must also consider the case k = σ 2 .

BID32 show that if σ 2 = σ 2 M LE , then this also corresponds to a saddle point as σ 2 is the average of the smallest eigenvalues meaning some perturbation will be unstable (except in a special case which is handled separately).However, what happens if σ 2 is not set to be the maximum likelihood estimate?

In this case, it is possible that there are no unstable perturbation directions (that is, λ j < σ 2 for too many j).

In this case when σ 2 is fixed, there are local optima where W has zero-columnsthe same solutions that we observe in non-linear VAEs corresponding to posterior collapse.

Note that when σ 2 is learned in non-degenerate cases the local maxima presented above become saddle points where σ 2 is made smaller by its gradient.

In practice, we find that even when σ 2 is learned in the non-linear case local maxima exist.

Linear autoencoders suffer from a lack of identifiability which causes the decoder columns to span the principal component subspace instead of recovering it.

Here we show that linear VAEs are able to recover the principal components up to scaling.

We once again consider the linear VAE from Eq. FORMULA9 : DISPLAYFORM0 The output of the VAE,x is distributed as, DISPLAYFORM1 Therefore, the linear VAE is invariant to the following transformation: DISPLAYFORM2 where A is a diagonal matrix with non-zero entries so that D is well-defined.

We see that the direction of the columns of W are always identifiable, and thus the principal components can be exactly recovered.

Moreover, we can recover complete identifiability by fixing D = I, so that there is a unique global maximum.

Here we present details on the analysis of the stationary points of the ELBO objective.

To begin, we first derive closed form solutions to the components of the marginal log-likelihood (including the ELBO).

The VAE we focus on is the one presented in Eq. FORMULA9 , with a linear encoder, linear decoder, Gaussian prior, and Gaussian observation model.

Remember that one can express the marginal log-likelihood as: DISPLAYFORM0 Each of the terms (A-C) can be expressed in closed form for the linear VAE.

Note that the KL term (A) is minimized when the variational distribution is exactly the true posterior distribution.

This is possible when the columns of the decoder are orthogonal.

The term (B) can be expressed as, DISPLAYFORM1 The term (C) can be expressed as, DISPLAYFORM2 Noting that Wz ∼ N WV(x − µ), WDW T , we can compute the expectation analytically and obtain, DISPLAYFORM3 To compute the stationary points we must take derivatives with respect to µ, D, W, V, σ 2 .

As before, we have µ = µ M LE at the global maximum and for simplicity we fix µ here for the remainder of the analysis.

Taking the marginal likelihood over the whole dataset, at the stationary points we have, DISPLAYFORM4 The above are computed using standard matrix derivative identities (Petersen et al.) .

These equations yield the expected solution for the variational distribution directly.

From Eq. (20) we compute DISPLAYFORM5 recovering the true posterior mean in all cases and getting the correct posterior covariance when the columns of W are orthogonal.

We will now proceed with the proof of Theorem 1.

Theorem 1.

The ELBO objective does not introduce any additional local maxima to the pPCA model.

Proof.

If the columns of W are orthogonal then the marginal log-likelihood is recovered exactly at all stationary points.

This is a direct consequence of the posterior mean and covariance being recovered exactly at all stationary points so that (1) is zero.

We must give separate treatment to the case where there is a stationary point without orthogonal columns of W. Suppose we have such a stationary point, using the singular value decomposition we can write W = ULR T , where U and R are orthogonal matrices.

Note that log p(x) is invariant to the choice of R BID32 .

However, the choice of R does have an effect on the first term (1) of Eq. (14): this term is minimized when R = I, and thus the ELBO must increase.

To formalize this argument, we compute (1) at a stationary point.

From above, at every stationary point the mean of the variational distribution exactly matches the true posterior.

Thus the KL simplifies to: DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 where M = diag(W T W) + σ 2 I. Now consider applying a small rotation to W: W → WR .

As the optimal D and V are continuous functions of W, this corresponds to a small perturbation of these parameters too for a sufficiently small rotation.

Importantly, log det M remains fixed for any orthogonal choice of R but log det M does not.

Thus, we choose R to minimize this term.

In this manner, (1) shrinks meaning that the ELBO (-2)+(3) must increase.

Thus if the stationary point existed, it must have been a saddle point.

We now describe how to construct such a small rotation matrix.

First note that without loss of generality we can assume that det(R) = 1. (Otherwise, we can flip the sign of a column of R and the corresponding column of U.) And additionally, we have WR = UL, which is orthogonal.

The Special Orthogonal group of determinant 1 orthogonal matrices is a compact, connected Lie group and therefore the exponential map from its Lie algebra is surjective.

This means that we can find an upper-triangular matrix B, such that DISPLAYFORM10 T )}, where n( ) is an integer chosen to ensure that the elements of B are within > 0 of zero.

This matrix is a rotation in the direction of R which we can make arbitrarily close to the identity by a suitable choice of .

This is verified through the Taylor series expansion of DISPLAYFORM11 .

Thus, we have identified a small perturbation to W (and D and V) which decreases the posterior KL (A) but keeps the marginal log-likelihood constant.

Thus, the ELBO increases and the stationary point must be a saddle point.

We would like to extend our linear analysis to the case where we have a Bernoulli observation model, as this setting also suffers severely from posterior collapse.

The analysis may also shed light on more general categorical observation models which have also been used.

Typically, in these settings a continuous latent space is still used (for example, Bowman et al. FORMULA0 ).We will consider the following model, p(z) = N (0, I), p(x|z) = Bernoulli(y), y = σ(Wz + µ)where σ denotes the sigmoid function, σ(y) = 1/(1 + exp(−y)) and we assume an independent Bernoulli observation model over x.

Unfortunately, under this model it is difficult to reason about the stationary points.

There is no closed form solution for the marginal likelihood p(x) or the posterior distribution p(z|x).Numerical integration methods exist which may make it easy to evaluate this quantity in practice but they will not immediately provide us a good gradient signal.

We can compute the density function for y using the change of variables formula.

Noting that Wz + µ ∼ N (µ, WW T ), we recover the following logit-Normal distribution: DISPLAYFORM0 We can write the marginal likelihood as, DISPLAYFORM1 = E z y(z) DISPLAYFORM2 where (·) x is taken to be elementwise.

Unfortunately, the expectation of a logit-normal distribution has no closed form BID1 ) and so we cannot tractably compute the marginal likelihood.

Similarly, under ELBO we need to compute the expected reconstruction error.

This can be written as, E q(z|x) [log p(x|z)] = y(z)x (1 − y(z)) 1−x N (z; V(x − µ), D)dz,another intractable integral.

Visualizing stationary points of pPCA For this experiment we computed the pPCA MLE using a subset of 10000 random training images from the MNIST dataset.

We evaluate and plot the marginal log-likelihood in closed form on this same subset.

MNIST VAE The VAEs we trained on MNIST all had the same architecture: 784-1024-512-k-512-1024-784.

The VAE parameters were optimized jointly using the Adam optimizer BID20 .

We trained the VAE for 1000 epochs total, keeping the learning rate fixed throughout.

We performed a grid search over values for the learning rate and reported results for the model which achieved the best training ELBO.CelebA VAE We used the convolutional architecture proposed by BID15 .

Otherwise, the experimental procedure followed that of the MNIST VAEs.

We also trained convolutional VAEs on the CelebA dataset using fixed choices of σ 2 .

As expected, the same general pattern emerged as in FIG1 .

Reconstructions from the KL-Annealed model are shown in FIG9 .

We also show the output of interpolating in the latent space in FIG10 .

To produce the latter plot, we compute the variational mean of 3 input points (top left, top right, bottom left) and interpolate linearly between.

We also extrapolate out to a fourth point (bottom right), which lies on the plane defined by the other points.

@highlight

We show that posterior collapse in linear VAEs is caused entirely by marginal log-likelihood (not ELBO). Experiments on deep VAEs suggest a similar phenomenon is at play.