Variational Auto-encoders (VAEs) are deep generative latent variable models consisting of two components: a generative model that captures a data distribution p(x) by transforming a distribution p(z) over latent space, and an inference model that infers likely latent codes for each data point (Kingma and Welling, 2013).

Recent work shows that traditional training methods tend to yield solutions that violate modeling desiderata: (1) the learned generative model captures the observed data distribution but does so while ignoring the latent codes, resulting in codes that do not represent the data (e.g. van den Oord et al. (2017); Kim et al. (2018)); (2) the aggregate of the learned latent codes does not match the prior p(z).

This mismatch means that the learned generative model will be unable to generate realistic data with samples from p(z)(e.g.

Makhzani et al. (2015); Tomczak and Welling (2017)).



In this paper, we demonstrate that both issues stem from the fact that the global optima of the VAE training objective often correspond to undesirable solutions.

Our analysis builds on two observations: (1) the generative model is unidentifiable – there exist many generative models that explain the data equally well, each with different (and potentially unwanted) properties and (2) bias in the VAE objective – the VAE objective may prefer generative models that explain the data poorly but have posteriors that are easy to approximate.

We present a novel inference method, LiBI, mitigating the problems identified in our analysis.

On synthetic datasets, we show that LiBI can learn generative models that capture the data distribution and inference models that better satisfy modeling assumptions when traditional methods struggle to do so.

Introduction Variational Auto-encoders (VAEs) are deep generative latent variable models consisting of two components: a generative model that captures a data distribution p(x) by transforming a distribution p(z) over latent space, and an inference model that infers likely latent codes for each data point (Kingma and Welling, 2013) .

Recent work shows that traditional training methods tend to yield solutions that violate modeling desiderata: (1) the learned generative model captures the observed data distribution but does so while ignoring the latent codes, resulting in codes that do not represent the data (e.g. van den Oord et al. (2017) ; ); (2) the aggregate of the learned latent codes does not match the prior p(z).

This mismatch means that the learned generative model will be unable to generate realistic data with samples from p(z)(e.g.

Makhzani et al. (2015) ; Tomczak and Welling (2017) ).

In this paper, we demonstrate that both issues stem from the fact that the global optima of the VAE training objective often correspond to undesirable solutions.

Our analysis builds on two observations: (1) the generative model is unidentifiable -there exist many generative models that explain the data equally well, each with different (and potentially unwanted) properties and (2) bias in the VAE objective -the VAE objective may prefer generative models that explain the data poorly but have posteriors that are easy to approximate.

We present a novel inference method, LiBI, mitigating the problems identified in our analysis.

On synthetic datasets, we show that LiBI can learn generative models that capture the data distribution and inference models that better satisfy modeling assumptions when traditional methods struggle to do so.

Background A VAE is comprised of a generative model and an inference model.

Under the generative model, we posit that the observed data and the latent codes are jointly distributed as p θ (x, z) = p θ (x|z)p(z).

The likelihood p θ (x|z) is defined by a neural network f with parameters θ and an output noise model ∼ p( ) such that x|z = f θ (z) + .

Direct maximization of the expected observed data log-likelihood E p(x) log Z p θ (x, z)dz over θ is intractable.

Instead, we maximize the variational lower bound (ELBO),

where q η(x) ∈ Q is a variational distribution with parameters η(x).

Since the bound is tight when q η(x) (z) = p θ (z|x), we aim to infer p θ (z|x).

To speed up finding the variational parameters η for some new input x, we train a neural inference model g with parameters

We demonstrate two general ways wherein global optima of the ELBO correspond to undesirable models.

In the following, we fix our variational family to be mean-field Gaussian.

Case 1:

Learning the Inference Model Compromises the Quality of the Generative Model.

Suppose that the variational family does not contain the posteriors of the data-generating-model.

Then, often, inference must trade-off between learning a generative model that explains the data well and one that has posteriors that are easy for the inference network to approximate.

Thus, the global minima of the VAE objective can specify models that both fail to capture the data distribution and whose aggregated posterior fails to match the prior.

As demonstration, consider the following model (described fully in Appendix C.2):

with σ 2 = 0.01, B = [ 0.006 0 0 0.006 ] and A = 0.75 0.25 1.5 −1.0 as the data generating model.

Here, we fix B (which also fixes the covariance of the observation noise) and learn the parameter θ = A. In this example, the ground-truth posteriors are non-diagonal Gaussians.

Here, the VAE objective can achieve a lower loss by compromising the MLE objective in order to better satisfy the PM objective -i.e.

the VAE objective will prefer a model that fails to capture the data distribution but has a diagonal Gaussian posterior over the ground-truth model.

Figure 1C shows the data distribution of the ground truth model θ GT , φ GT (with L(θ GT , φ GT ) = 0.532) differs from the distribution of the learned model θ * , φ * in Figure  1D (with L(θ * , φ * ) = 0.196).

Moreover, since the learned model fails to capture the data distribution, its aggregated posterior fails to match the prior (see Figures 1E vs. 1F):

Even when we restrict the class of generative models to ones that fit the data well, the posterior matching objective will still select a model with a simple posterior.

Unfortunately, the selected generative model may have undesirable properties like uninformative latent codes.

As demonstration, consider the model from Equation 3 with the data generating model specified by: σ 2 = 0.01, A = 0.75 0.25 1.5 −1.0 , and B is some diagonal matrix with values in [0, σ 2 ].

In this case, we fix A and and learn the parameter θ = B. Since the observation noise covariance I ·σ 2 −B changes with B, the data marginal is fixed at p θ (x) = N 0, AA + I · σ 2 for every B. Thus, for every θ, the MLE objective is 0.

However, although every choice of θ explain the data equally well, the posterior matching objective (and hence the VAE objective) is minimized by θ's whose posteriors have the least amount of correlation.

Figure 1A shows that L(θ, φ) prefers high value in the upper diagonal of B and low value in the lower diagonal.

Figure 1B shows the informativeness of the latent codes for the corresponding θ.

We see that the data to latent code mutual information I(X; Z) corresponding to the θ selected by L(θ, φ) is not optimal.

That is, even if the true data generating model produces highly informative latent codes, the VAE objective may select a model that produces uninformative latent codes.

Discussion The principles of our analysis extend to non-linear VAEs and complex variational families.

In the VAE objective, the posterior matching objective acts like a regularizing term, biasing the learned generative models towards simple models with posteriors that are easy to approximate (with respect to the choice of variational family).

Thus, joint training of the inference and generative models introduces unintended and undesirable optima, which would not appear when these models are learned separately.

Case 2:

Learning the Inference Model Selects an Undesirable Generative Model.

Even if the variational family is rich, the inference for the posterior can nonetheless bias the learning for the generative model.

It is well known that the generative model is nonidentifiable under the MLE objective -there are many models that minimize the MLE objective.

To focus on the effects of non-identifiability, let us assume that the variational family is expressive enough that it contains the posteriors of multiple models that could have generated the data.

Then the posterior matching objective is 0 since we can find parameters φ such that q φ (z|x) = p θ (z|x) for any such θ.

Consequently, L(θ, φ) has multiple global minima corresponding to the multiple generative models that maximizes the date likelihood.

Some of these models may not satisfy our desiderata; e.g., the latent codes have low mutual information with the data.

As demonstration, consider the following model (fully described in Appendix C.1):

In this case, the mean-field variational family includes the posterior p θ (z|x) for all θ, i.e. the posterior matching objective can be fully minimized.

Furthermore, every θ ∈ [0, σ 2 ] yields the same data marginal, p θ (x) = N 0, σ 2 , and thus minimizes the MLE objective.

However, not all choice of θ are equivalent.

Given θ, the mutual information between the learned latent codes and the data is I θ (X; Z) = Const − 1 2 log(σ 2 − θ 2 ).

Thus, the set of global minima of L(θ, φ) contain many models that produce uninformative latent codes.

Discussion We've shown that posterior collapse can happen at global optima of the VAE objective and that, in these cases, collapse cannot always be mitigated by improving the inference model (as in He et al. (2019) ) or by limiting the capacity of the generative model (as in Bowman et al. (2015) ; Gulrajani et al. (2016) ; Yang et al. (2017)).

In Section 1, we showed that common problems with traditional VAE training stem from the non-identifiability of the likelihood and the bias of the VAE objective towards models with simple posteriors, even if such models cannot capture the data distribution.

We propose a novel inference method to specifically target these problems.

To avoid the biasing effect of the PM objective on learning the generative model, we decouple the training of the generative and inference models -first we learn a generative model, then we learn an inference model while fixing the learned generative model (note that amortization allows for efficient posterior inference).

To avoid undesirable global optima of the likelihood, we learn a generative model constrained by task-specific modeling desiderata.

For instance, if informative latent codes are necessary for the task, the likelihood can be constrained so that the mutual information between the data and latent codes under θ is at least δ.

While there are a number of works in literature that incorporate task-specific constraints to VAE training (e.g. Chen et al. (2016); Zhao et al. (2017 Zhao et al. ( , 2018 ; Liu et al. (2018) ), adding these constraints to the VAE objective directly affects both the generative and the inference models, and, consequently, may introduce additional undesirable global optima.

In our approach, added constraints only directly affects the generative model -i.e.

the quality of inference cannot be compromised by the added constraints.

We call our training framework Likelihood Before Inference (LiBI), and propose one possible instantiation of this framework here.

Step 1:

Learning the Generative Model We compute a tractable approximation to the MLE objective, constrained so that the likelihood satisfies task-specific modeling desiderata (such as high I(X; Z)) as needed.

:

where each c i is a constraint applied to the likelihood.

We do this by computing joint maximum likelihood estimates for θ and z n while additionally constraining the z n 's to have come from our assumed model (see Appendix D for a formal derivation of this approximation):

where HZ(·) is the Henze-Zirkler test statistic for Gaussianity, µ(·), Σ(·) represent the empirical mean and covariance, and the z n 's are amortized using a neural network z n = h(x n ; ϕ) parametrized by ϕ. These constraints encourage the generative model to capture p(x) given p(z), i.e. the aggregated posterior under this model will match the prior p(z).

Step 2:

Learning the Inference Model Given the θ learned in Step 1, we learn φ to compute approximate posteriors q φ (z|x):

.

We note that φ, too, will satisfy our modeling assumptions, since with a fixed θ, the model nonidentifiability we describe in Section 1 is no longer present.

Step 3: Reinitialize Inference for the Generative Model We repeat the process, initializing h(x n ; ϕ) = µ(x n ; φ), where µ(x n ; φ)

is the mean of q φ (z n |x n ).

This steps provides an intelligent random initialization allowing step 1 to learn a better quality model.

In theory, if the generative model and the inference models are learned perfectly in Steps 1 and 2, then Step 3 is obviated.

In practice, we find that Step 3 improves the quality of the generative model and only a very small number of iterations is actually needed.

Discussion Using LiBI, we can now evaluate the quality of the generative model and the inference models independently.

This is in contrast to traditional VAE inference, in which the ELBO entangles issues of modeling and issues of inference.

On 4 synthetic data sets for which we know the data generating model, we compare LiBI with existing inference methods: VAE (Kingma and Welling, 2013), β-VAE (Higgins et al., 2017) , β-VAE with annealing, Lagging inference networks (He et al., 2019) .

Across all datasets, LiBI learns generative models that better capture p(x) (as quantified by loglikelihood and the Smooth k-NN test statistic (Djolonga and Krause, 2017) ) and for which the aggregated posterior better matches the prior (see Appendix B).

Conclusion In this paper, we show that commonly noted issues with VAE training are attributable to the fact that global optima of the VAE training objective often includes undesirable solutions.

Based on our analysis, we propose a novel training procedure, LiBI, that avoid these undesirable optima while retaining the tractability of traditional VAE inference.

On synthetic datasets, we show that LiBI able to learn generative models that capture the data distribution and inference models whose aggregated posterior matches the prior while traditional methods struggle to do so.

Two common issues noted in VAE literature are posterior collapse and the mismatch between aggregated posterior and prior.

Posterior collapse occurs when the posterior under both the generative model and approximate posterior learned by the inference model are equal the prior p(z) (He et al., 2019) .

Surprisingly, under posterior collapse, the model is still able to generate samples from p data (x)(e.g.

Chen et al. (2016) ; Zhao et al. (2017) ).

This is often attributed to the fact the generative model is very powerful and is therefore able to maximize the log data marginal likelihood without the help of the auxiliary latent codes (van den Oord et al., 2017) .

Existing literature focuses on mitigating model collapse in one of the three ways: 1. modifying the optimization procedure to bias training way from collapse (He et al., 2019) ; 2. choosing variational families that make collapse less likely to occur (Razavi et al., 2019) ; 3. modifying the generative and inference model architecture to encourage more information sharing between the x's and the z's (Dieng et al., 2018).

Although much of existing literature describes issue of posterior collapse and proposes methods to avoid it, less attention has been given to explaining why it occurs.

He et al. (2019) conjecture that it occurs as a result of the joint training: since the likelihood changes over the course of training, it is incentivized to ignore the output of the inference network whose output in the early stages of training is not yet meaningful.

Mismatch between aggregated posterior and prior refers to the case when q φ (z) = p(z), where

One might expect the two distributions to match because for any given likelihood θ, one should be able to recover the prior from the true posterior p(z|x) as follows:

An x produced by the generated model from a z that is likely under the prior but unlikely under the aggregate posterior may have "poor sample quality", since the the generative model is unlikely to have encountered such a z during training (Makhzani et al., 2015; Tomczak and Welling, 2017) .

Existing literature mitigate this issue by either increasing the flexibility of the prior to better fit the aggregate posterior (Tomczak and Welling, 2017; Bauer and Mnih, 2018) or developing a method to sample more robustly from the latent space (Zhao et al., 2017) .

Examples of the latter include training a second VAE to be able to generate z from u and then sampling from p θ (x, z) using a Gibbs sampler (Zhao et al., 2017) .

In this work, we provide a unifying analysis of both posterior collapse and mismatch, showing that both can occur as global optima of the VAE objective.

Through our analysis, we also show that at these optima, neither issue can be reliably resolved by existing methods.

In Figures 2 and 3 , we compare the posteriors learned by traditional VAE inference and by LiBI, respectively, on the synthetic dataset LinearJTEx.

Here we demonstrate that traditional inference learns a generative model θ under which it is easy to approximate the corresponding posteriors.

However, this comes at the cost of θ being unable to capture the data distribution.

Assume the following generative process for the data:

For this generative process, p θ (x) = N 0, σ 2 for any value of θ such that 0 ≤ θ ≤ σ 2 .

Additionally, θ directly controls I(X; Z) -when θ = 0, we have that I(X; Z); when θ = σ 2 , we have that I(X; Z) = ∞. To see this, we will compute I θ (X; Z) directly (by computing p θ (x, z) and p(x)p(z)):

As such, we can compute the mutual information between x and z as follows:

For this model, the posterior p θ (z|x), is:

Since this example is univariate, the mean-field Gaussian variational family will include the true posterior for any θ.

Assume the following generative process for the data:

where B is a diagonal matrix with diagonal elements between 0 and σ 2 .

For this generative process, p B (x) = N 0, AA + I · σ 2 for all valid values of B. For this model, the complete data likelihood and marginals are,

Therefore, I B (X; Z) can be computed as follows:

Lastly, the posterior for this model, p B (z|x), is a Gaussian with mean and covariance,

For our choice of A, the mean-field Gaussian will not include the true posterior for this model.

The best-fitting mean-field approximation to the true posterior can be computed as in Appendix C.3.

Let B be a diagonal matrix and let Σ be a full-covariance matrix.

where each element in the above sum is independent and is minimized when B ii = 1 Σ

, and where Σ −1

ii is the ith diagonal entry of Σ −1 .

The LiBI Framework The LiBI framework is composed of two steps: (1) learning a high-quality likelihood capable of generating the observed data distribution, and (2) fixing the likelihood learned in Step 1, performing inference to learn the latent codes given the data.

We emphasize that our framework is general, so one can use various existing methods for either step.

For example, one can use a GAN for Step 1, and MCMC sampling for Step 2.

In this section, we derive a tractable approximation to Step 1 that can be easily enhanced to include constraints for task-specific desiderata, and that is amenable to gradient-based optimization methods.

wherein Equation 31, we approximate E p(z) [p θ (x n |z)] with a single sample, z n , that makes its corresponding x n most likely (this is analogous to the Empirical Bayes EB MAP Type II estimates often used to tune prior hyper-parameters).

This step, however, has a problem: it is biased towards learning z n 's close to 0.

We will now demonstrate that this issue exists and is a result of non-identifiability in the MLE estimate with respect to θ, {z n } N n=1 .

We then provide a solution to this problem.

Characterization of Non-Identifiability in Tractable Approximation Consider the following: let Z = {z n } N n=1 be the true z's and θ used to generate the observed data, X = {x n } N n=1 in the following generative process:

Now, consider, an alternative Z = {ẑ n } N n=1 andθ such that,

yielding the following alternative generative process:

Under these generative processes, both the data marginals and the likelihoods are equal:

However, since in our model we assumed the prior is fixed p(z) = N (0, I), the alternate parameters Z,θ are preferred by the joint log-likelihood when c > 1,

since log pθ(x n |ẑ n ) = log p θ (x n |z n ) by construction and log N (ẑ n |0, I) > log N (z n |0, I) since theẑ n 's are closer to 0 when c > 1.

This will cause our approximation from Equation 31 to prefer the modelθ, which generates a different data distribution that the true data distribution:

Identifying the Tractable Approximation using the Henze-Zirkler Test Statistic Returning to our approximation of the MLE objective in Equation 31, we can avoid this issue by constraining the z n 's to have come from the prior:

We do this by constraining the z n 's to be Gaussian using the Henze-Zirkler test for Gaussianity and by constraining the empirical mean and covariance of the z n 's to be that of the standard normal:

We hypothesize that if the likelihood function, f θ , is "smooth" and well-behaved (that is, that it maps nearby z's to nearby x's), that our approximation of the likelihood will come close to the true one.

Using this framework, we first recover a high-quality likelihood (a likelihood that, unlike in the traditional VAE objective, is not compromised to match the approximate posterior).

Our framework therefore naturally encourages this likelihood to satisfy modeling assumptions; that is, if we find a θ for which the x's are reconstructed accurately given Gaussian z's, the aggregated posterior under θ, p θ (z), will match the prior p(z).

Given this likelihood, we can then learn a posterior that accurately approximates p θ (z|x).

We note that φ, too, will satisfy our modeling assumptions, since with a fixed θ, the model non-identifiability we describe is no longer present.

The LiBI Inference Method We incorporate the constraints in Equation 43 as smooth penalties into the Lagrangian in Equation 44.

We additionally define h(x n ; ϕ) to be a neural network parameterized by ϕ that, given x n , returns the specific z n that generated it.

ϕ allows us to amortize Equation 44.

We repeat the following steps R times:

1.

Step 1: θ t , ϕ t = argmin θ,ϕ − 1 N n log p θ (x n |h(x n ; ϕ)) + HZ exp HZ {h(x n ; ϕ)} N n=1

+ exp Σ {h(x n ; ϕ)} N n=1 − I

2.

Step 2:

= argmin φ 1 N n −ELBO(θ t , φ)

<|TLDR|>

@highlight

We characterize problematic global optima of the VAE objective and present a novel inference method to avoid such optima.