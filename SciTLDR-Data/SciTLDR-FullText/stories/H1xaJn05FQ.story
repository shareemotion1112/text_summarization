In this paper we use the geometric properties of the optimal transport (OT) problem and the Wasserstein distances to define a prior distribution for the latent space of an auto-encoder.

We introduce Sliced-Wasserstein Auto-Encoders (SWAE), that enable one to shape the distribution of the latent space into any samplable probability distribution without the need for training an adversarial network or having a likelihood function specified.

In short, we regularize the auto-encoder loss with the sliced-Wasserstein distance between the distribution of the encoded training samples and a samplable prior distribution.

We show that the proposed formulation has an efficient numerical solution that provides similar capabilities to Wasserstein Auto-Encoders (WAE) and Variational Auto-Encoders (VAE), while benefiting from an embarrassingly simple implementation.

We provide extensive error analysis for our algorithm, and show its merits on three benchmark datasets.

Learning such generative models boils down to minimizing a dissimilarity measure between the data distribution and the output distribution of the generative model.

To this end, and following the work of and , we approach the problem of generative modeling from the optimal transport point of view.

The optimal transport problem BID38 ; BID22 provides a way to measure the distances between probability distributions by transporting (i.e., morphing) one distribution into another.

Moreover, and as opposed to the common information theoretic dissimilarity measures (e.g., f -divergences), the p-Wasserstein dissimilarity measures that arise from the optimal transport problem: 1) are true distances, and 2) metrize a weak convergence of probability measures (at least on compact spaces).

Wasserstein distances have recently attracted a lot of interest in the learning community BID11 ; BID14 ; ; ; BID22 due to their exquisite geometric characteristics BID34 .

See the supplementary material for an intuitive example showing the benefit of the Wasserstein distance over commonly used f -divergences.

In this paper, we introduce a new type of auto-encoders for generative modeling (Algorithm 1), which we call Sliced-Wasserstein auto-encoders (SWAE) , that minimize the sliced-Wasserstein distance between the distribution of the encoded samples and a samplable prior distribution.

Our work is most closely related to the recent work by and more specifically the follow-up work by .

However, our approach avoids the need to perform adversarial training in the encoding space and is not restricted to closed-form distributions, while still benefiting from a Wasserstein-like distance measure in the latent space.

Calculating the Wasserstein distance can be computationally expensive, but our approach permits a simple numerical solution to the problem.

Finally, we note that there has been several concurrent papers, including the work by BID10 Deshpande et al. ( ) andŞimşekli et al. (2018 , that also looked into the application of sliced-Wasserstein distance in generative modeling.

Regardless of the concurrent nature of these papers, our work remains novel and is distinguished from these methods.

BID10 use the sliced-Wasserstein distance to match the distributions of high-dimensional reconstructed images, which require large number of slices, O(10 4 ), while in our method and due to the distribution matching in the latent space we only need O(10) slices.

We also note that BID10 proposed to learn discriminative slices to mitigate the need for a very large number of random projections that is in essence similar to the adversarial training used in GANs, which contradicts with our goal of not using adversarial training.

Simşekli et al. (2018) , on the other hand, take an interesting but different approach of parameter-free generative modeling via sliced-Wasserstein flows.

Let X denote the compact domain of a manifold in Euclidean space and let x n ∈ X denote an individual input data point.

Furthermore, let ρ X be a Borel probability measure defined on X. We define the probability density function p X (x) for input data x to be: DISPLAYFORM0 Let φ : X → Z denote a deterministic parametric mapping from the input space to a latent space Z (e.g., a neural network encoder).

To obtain the density of the push forward of ρ X with respect to φ, i.e., ρ Z = φ * (ρ X ), we use Random Variable Transformation (RVT) BID12 ).

In short, the probability density function of the encoded samples z can be expressed in terms of φ and p X by: DISPLAYFORM1 where δ denotes the Dirac distribution function.

Similar to variational Auto-Encoders (VAEs) BID18 and the Wasserstein Auto-Encoders (WAE) , our main objective is to encode the input data points x ∈ X into latent codes z ∈ Z such that: 1) x can be recovered/approximated from z, and 2) the probability density function of the encoded samples, p Z , follows a prior distribution q Z .

Let ψ : Z → X be the decoder that maps the latent codes back to the original space such that DISPLAYFORM2 where y denotes the decoded samples.

It is straightforward to see that when ψ = φ −1 (i.e. ψ(φ(·)) = id(·)), the distribution of the decoder p Y and the input distribution p X are identical.

Hence, in its most general form, the objective of such auto-encoders simplifies to learning φ and ψ, so that they minimize a dissimilarity measure between p Y and p X , and between p Z and q Z .

In what follows, we briefly review the existing dissimilarity measures for these distributions.1.1 MINIMIZING DISSIMILARITY BETWEEN p X AND p Y We first emphasize that the VAE often assumes stochastic encoders and decoders BID18 , while we consider the case of only deterministic mappings.

Although, we note that, similar to WAE, SWAE can also be formulated with stochastic encoders.

Different measures have been used previously to compute the dissimilarity between p X and p Y .

Most notably, BID29 showed that for the general family of f -divergences, D f (p X , p Y ), (including the KL-divergence, JensenShannon, etc.), using the Fenchel conjugate of the convex function f and minimizing D f (p X , p Y ) leads to a min-max problem that is equivalent to the adversarial training widely used in the generative modeling literature BID13 ; BID26 ; BID27 .Others have utilized the rich mathematical foundation of the OT problem and Wasserstein distances ; BID14 ; ; to define a distance between p X and p Y .

In Wasserstein-GAN, utilized the Kantorovich-Rubinstein duality for the 1-Wasserstein distance, W 1 (p X , p Y ), and reformulated the problem as a min-max optimization that is solved through an adversarial training scheme.

Inspired by the work of and , it can be shown that (see supplementary material for a proof): DISPLAYFORM3 Furthermore, the r.h.s.

of equation 3 supports a simple implementation where for i.i.d samples of the input distribution, {x n } N n=1 , the upper bound can be approximated as: DISPLAYFORM4 The r.h.s of equation 3 and equation 4 take advantage of the existence of pairs x n and y n = ψ(φ(x n )), which make f (·) = ψ(φ(·)) a transport map between p X and p Y (but not necessarily the optimal transport map).

In this paper, we minimize W ‡ c (p X , p Y ) following equation 4 to minimize the discrepancy between p X and p Y .

Next, we focus on the discrepancy measures between p Z and q Z .1.2 MINIMIZING DISSIMILARITY BETWEEN p Z AND q Z If q Z is a known distribution with an explicit formulation (e.g. Normal distribution) the most straightforward approach for measuring the (dis)similarity between p Z and q Z is the log-likelihood of z = φ(x) with respect to q Z , formally: DISPLAYFORM5 maximizing the log-likelihood is equivalent to minimizing the KL-divergence between p Z and q Z , D KL (p Z , q Z ) (see supplementary material for more details and derivation of Equation equation 5).

This approach has two major limitations: 1) The KL-Divergence and in general f -divergences do not provide meaningful dissimilarity measures for distributions supported on non-overlapping lowdimensional manifolds ; (see supplementary material), which is common in hidden layers of neural networks, and therefore they do not provide informative gradients for training φ, and 2) we are limited to distributions q Z that have known explicit formulations, which is restrictive as it eliminates the ability to use the much broader class of samplable distributions.

Various alternatives exist in the literature to address the above-mentioned limitations.

These methods often sampleZ = {z j } N j=1 from q Z and Z = {z n = φ(x n )} N n=1 from p X and measure the discrepancy between these sets (i.e. point clouds).

Note that there are no one-to-one correspondences betweenz j s and z n s. In their influential WAE paper, proposed two different approaches for measuring the discrepancy betweenZ and Z, namely the GAN-based and the maximum mean discrepancy (MMD)-based approaches.

The GAN-based approach proposed in defines a discriminator network, D Z (p Z , q Z ), to classifyz j s and z n s as coming from 'true' and 'fake' distributions correspondingly, and proposes a min-max adversarial optimization for learning φ and D Z .

The MMD-based approach, utilizes a positive-definite reproducing kernel k : Z × Z → R to measure the discrepancy betweenZ and Z. The choice of the kernel and its parameterization, however, remain a data-dependent design parameter.

An interesting alternative approach is to use the Wasserstein distance between p Z and q Z .

Following the work of , this can be accomplished utilizing the Kantorovich-Rubinstein duality and through introducing a min-max problem, which leads to yet another adversarial training scheme similar to the GAN-based method in .

Note that, since elements ofZ and Z are not paired, an approach similar to equation 4 could not be used to minimize the discrepancy.

In this paper, we propose to use the sliced-Wasserstein metric, BID3 ; BID21 ; BID6 ; , to measure the discrepancy between p Z and q Z .

We show that using the sliced-Wasserstein distance ameliorates the need for training an adversary network or choosing a data-dependent kernel (as in WAE-MMD), and provides an efficient, stable, and simple numerical implementation.

Before explaining our proposed approach, it is worthwhile to point out the major difference between learning auto-encoders as generative models and GANs.

In GANs, one needs to minimize a distance between {ψ(z j )|z j ∼ q Z } M j=1 and {x n } M n=1 , which are high-dimensional point clouds for which there are no correspondences between ψ(z j )s and x n s.

For the auto-encoders, on the other hand, there exists correspondences between the high-dimensional point clouds {x n } M n=1 and {y n = ψ(φ(x n ))} M n=1 , and the problem simplifies to matching the lower-dimensional point clouds {φ( DISPLAYFORM6 .

In other words, the encoder performs a nonlinear dimensionality reduction, that enables us to solve a simpler problem compared to GANs.

Next we introduce the details of our approach.

In what follows we first provide a brief review of the necessary equations to understand the Wasserstein and sliced-Wasserstein distances and then present our Sliced Wasserstein auto-encoder (SWAE).

The Wasserstein distance between probability measures ρ X and ρ Y , with corresponding densities dρ X = p X (x)dx and dρ Y = p Y (y)dy is defined as: DISPLAYFORM0 where DISPLAYFORM1 is the set of all transportation plans (i.e. joint measures) with marginal densities p X and p Y , and c : X × Y → R + is the transportation cost.

equation 6 is known as the Kantorovich formulation of the optimal mass transportation problem, which seeks the optimal transportation plan between p X and p Y .

If there exist diffeomorphic mappings, f : X → Y (i.e. transport maps) such that y = f (x) and consequently, DISPLAYFORM2 where det(D·) is the determinant of the Jacobian, then the Wasserstein distance could be defined based on the Monge formulation of the problem (see BID38 and BID22 ) as: DISPLAYFORM3 where M P is the set of all diffeomorphisms that satisfy equation 7.

As can be seen from equation 6 and equation 8, obtaining the Wasserstein distance requires solving an optimization problem.

We note that various efficient optimization techniques have been proposed in the past (e.g. Cuturi FORMULA1 ; BID36 ; Oberman & Ruan FORMULA1 ) to solve this optimization.

For one-dimensional probability densities, p X and p Y , however, the Wasserstein distance has a closed-form solution.

Let P X and P Y be the cumulative distributions of one-dimensional probability distributions p X and p Y , correspondingly.

The Wassertein distance can then be calculated as below (see BID22 for more details): DISPLAYFORM4 This closed-form solution motivates the definition of sliced-Wasserstein distances.

Sliced-Wasserstein distance has similar qualitative properties to the Wasserstein distance, but it is much easier to compute.

The sliced-Wasserstein distance was used in to calculate barycenter of distributions and point clouds.

BID3 provided a nice theoretical overview of barycenteric calculations using the sliced-Wasserstein distance.

BID21 used it to define positive definite kernels for distributions BID6 to define a kernel for persistence diagrams.

Sliced-Wasserstein was recently used for learning Gaussian mixture models in , and it was also used as a measure of goodness of fit for GANs in BID17 .The main idea behind the sliced-Wasserstein distance is to slice (i.e., project) higher-dimensional probability densities into sets of one-dimensional marginal distributions and compare these marginal distributions via the Wasserstein distance.

The slicing/projection process is related to the field of Integral Geometry and specifically the Radon transform (see BID15 ).

The relevant result to our discussion is that a d-dimensional probability density p X can be uniquely represented as the set of its one-dimensional marginal distributions following the Radon transform and the Fourier slice theorem BID15 .

These one dimensional marginal distributions of p X are defined as: DISPLAYFORM0 where S d−1 is the d-dimensional unit sphere.

Note that for any fixed θ ∈ S d−1 , Rp X (·; θ) is a one-dimensional slice of distribution p X .

In other words, Rp X (·; θ) is a marginal distribution of p X that is obtained from integrating p X over the hyperplane orthogonal to θ.

Utilizing these marginal distributions in equation 10, the sliced Wasserstein distance could be defined as: DISPLAYFORM1 Given that Rp X (·; θ) and Rp Y (·; θ) are one-dimensional, the Wasserstein distance in the integrand has a closed-form solution (see equation 9).

Moreover, it can be shown that SW c is a true metric BID4 and BID20 ), and it induces the same topology as W c , at least on compact sets BID34 .

A natural transportation cost that has extensively studied in the past is the 2 2 , c(x, y) = x − y 2 2 , for which there are theoretical guarantees on existence and uniqueness of transportation plans and maps (see BID34 and BID38 ).

When c(x, y) = x − y p p for p ≥ 2, the following upper bound hold for the SW distance: DISPLAYFORM2 where, DISPLAYFORM3 Chapter 5 in BID4 proves this inequality.

In our paper, we are interested in p = 2, for which α p,d = 1 d , and we have: DISPLAYFORM4 In the Numerical Implementation Section, we provide a numerical experiment to compare W 2 and SW 2 , that confirms the above equation.

Our proposed formulation for the SWAE is as follows: DISPLAYFORM0 where φ is the encoder, ψ is the decoder, p X is the data distribution, p Y is the data distribution after encoding and decoding ( equation 2), p Z is the distribution of the encoded data ( equation 1), q Z is a predefined samplable distribution, and λ indicates the relative importance of the loss functions.

To further clarify why we use the sliced-Wasserstein distance to measure the difference between p Z and q Z , we reiterate that due to the lack of correspondences betweenz i s and z j s, one cannot minimize the upper-bound in equation 4, and calculation of the Wasserstein distance requires an additional optimization step to obtain the optimal coupling between p Z and q Z .

To avoid this additional optimization, while maintaining the favorable characteristics of the Wasserstein distance, we use the sliced-Wasserstein distance to measure the discrepancy between p Z and q Z .

We now describe the numerical details of our approach.

The Wasserstein distance between two one-dimensional probability densities p X and p Y is obtained from equation 9.

The integral in equation 9 can be numerically estimated using the midpoint Riemann sum, DISPLAYFORM0 2M (see FIG0 ).

In scenarios where only samples from the distributions are available, x m ∼ p X and y m ∼ p Y , the empirical densities can be estimated as DISPLAYFORM1 δ ym , where δ xm is the Dirac delta function centered at x m .

Therefore the corresponding empirical distribution function of p X is P X (t) ≈ P X,M (t) = 1 M M m=1 u(t−x m ) where u(.) is the step function (P Y,M (t) is defined similarly).

From Glivenko-Cantelli Theorem we have that sup t |P X,M (t) − P X (t)| a.s.− − → 0, where the convergence behavior is achieved via Dvoretzky-Kiefer-Wolfowitz inequality bound: DISPLAYFORM2 2 ).

Calculating the Wasserstein distance with the empirical distribution function is computationally attractive.

Sorting x m s in an ascending order, such that DISPLAYFORM3 and where i[m] is the index of the sorted x m s, it is straightforward to see that P −1 FIG0 .

The Wasserstein distance can be approximated by first sorting x m s and y m s and then calculating: We need to address one final question here.

How well does equation 15 approximate the Wasserstein distance, W c (p X , p Y )?

We first note that the rates of convergence of empirical distributions, for the p-Wasserstein metric (i.e., c(x, y) = |x − y| p ) of order p ≥ 1, have been extensively studied in the mathematics and statistics communities (see for instance BID2 and BID9 ).

A detailed description of these rates is, however, beyond the scope of this paper, especially since these rates are dependent on the choice of p. In short, for p = 1 it can be shown that DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where C is an absolute constant.

Similar results are achieved for E(Wp(pX,M , p X )) and (E(W p p (p X,M , p X ))) 1 p , although under more strict assumptions on p X (i.e., slightly stronger assumptions than having a finite second moment).

Using the triangle inequality together with the convergence rates of empirical distributions with respect to the p-Wasserstein distance, see BID2 , for W 1 (p X,M , p X ) (or more generally W p (p X,M , p X )) we can show that (see supplementary material): DISPLAYFORM7 for some absolute constant, C. We reiterate that similar bounds could be found for W p although with slightly more strict assumptions on p X and p Y .

In scenarios where only samples from the d-dimensional distribution, p X , are available, x m ∼ p X , the empirical density can be estimated as p X,M = 1 M M m=1 δ xm .

Following equation 10 it is straightforward to show that the marginal densities (i.e. slices) are obtained from: DISPLAYFORM0 see the supplementary material for a proof.

The Dvoretzky-Kiefer-Wolfowitz upper bound holds for Rp X (t, θ) and Rp X,M (t, θ).

Minimizing the sliced-Wasserstein distance (i.e., as in the second term of 14) requires an integration over the unit sphere in R d , i.e., S d−1 .

In practice, this integration is approximated by using a simple Monte Carlo scheme that draws uniform samples from S d−1 and replaces the integral with a finite-sample average, DISPLAYFORM0 Figure 2: SW approximations (scaled by 1.22 DISPLAYFORM1 , and different number of random slices, L. Moreover, the global minimum for SW c (p Z , q Z ) is also a global minimum for each DISPLAYFORM2 A fine sampling of S d−1 , however, is required for a good approximation of SW c (p Z , q Z ).

Intuitively, if p Z and q Z are similar, then their projections with respect to any finite subset of S d−1 would also be similar.

This leads to a stochastic gradient descent scheme where in addition to the random sampling of the input data, we also random sample the projection angles from S d−1 .A natural question arises on the effect of the number of random slices, L = |Θ|, on the approximation of the SW distance.

Here, we devised a simple experiment that demonstrates the effect of L on approximating the SW distance.

We generated two random multi-variate Gaussian distributions in a d-dimensional space, where d ∈ {2 n } 10 n=1 , to serve as p X = N (µ X , Σ X ) and p X = N (µ Y , Σ Y ).

The Wasserstein distance for the two Gaussian distributions has a closed form solution, DISPLAYFORM3 which served as the ground-truth distance between the distributions.

We then measured the SW distance between M = 1000 samples generated from the two Gaussian distributions using L ∈ {1, 10, 50, 100, 500, 1000} random slices.

We repeated the experiment for each L and d, a thousand times and report the means and standard deviations in Figure 2 .

Following equation 13 we scaled the SW distance by √ d.

Moreover we found out empirically that 1.22 DISPLAYFORM4 It can be seen from Figure 2 that the expected value of the scaled SW -distance closely follows the true Wasserstein distance.

A more interesting observation is that the variance of estimation increases for higher dimensions d and decreases as the number of random projections, L, increases.

Hence, calculating the SW distance in the image space, as in BID10 , requires a very large number of projections L to get a less variant approximation of the distance.

be randomly sampled from a uniform distribution on S d−1 .

Then using the numerical approximations described in this section, the loss function in equation 14 can be rewritten as: The steps of our proposed method are presented in Algorithm 1.

It is worth pointing out that sorting is by itself an optimization problem (which can be solved very efficiently), and therefore the sorting followed by the gradient descent update on φ and ψ is in essence a min-max problem, which is being solved in an alternating fashion.

Finally, we point out that each iteration of SWAE costs O(LM log(M )) operations.

DISPLAYFORM5

Require: Regularization coefficient λ, and number of random projections, L. Initialize the parameters of the encoder, φ, and decoder, ψ while φ and ψ have not converged do Sample {x1, ..., xM } from training set (i.e. pX ) Sample {z1, ...,zM } from qZ Sample {θ1, ..., θL} from S DISPLAYFORM0 Update φ and ψ by descending: DISPLAYFORM1

In our experiments we used three image datasets, namely the MNIST dataset by BID24 , the CelebFaces Attributes Dataset (CelebA) by BID25 , and the LSUN Bedroom Dataset by BID41 .

For the MNIST dataset we used a simple auto-encoder with mirrored classic deep convolutional neural networks with 2D average poolings, leaky rectified linear units (Leaky-ReLu) as the activation functions, and upsampling layers in the decoder.

For the CelebA and LSUN datasets we used the DCGAN Radford et al. (2015) architecture similar to .To test the capability of our proposed algorithm in shaping the latent space of the encoder, we started with the MNIST dataset and trained SWAE to encode this dataset to a two-dimensional latent space (for the sake of visualization) while enforcing a match between p X and p Y and p Z and q Z .

We chose four different samplable distributions as shown in FIG5 .

It can bee seen that SWAE can successfully embed the dataset into the latent space while enforcing p Z to closely follow q Z .

In addition, we sample the two-dimensional latent spaces on a 25 × 25 grid in [−1, 1] 2 and decode these points to visualize their corresponding images in the digit/image space.

To get a sense of the convergence behavior of SWAE, and similar to the work of BID17 , we calculate the Sliced Wasserstein distance between p Z and q Z as well as p X and p Y at each batch iteration where we used p-LDA BID39 to calculate projections (See supplementary material).

We compared the convergence behavior of SWAE with the closest related work, WAE Tolstikhin et al. (2017) (specifically WAE-GAN) where an adversarial training is used to match p Z to q Z , while the loss function for p X and p Y remains exactly the same between the two methods.

We repeated the experiments 100 times and report the summary of results in FIG2 .

We mention that the exact same models and optimizers were used for both methods in this experiment.

An interesting observation, here is that while WAE-GAN provides good or even slightly better generated random samples for MNIST (lower sliced-Wasserstein distance between p X and p Y ), it fails to provide a good match between p Z and q Z for the choice of the prior distribution reported in FIG2 .

This phenomenon seems to be related to the mode-collapse problem of GANs, where the adversary fails to sense that the distribution is not fully covered.

Finally, in our experiments we did not notice a significant difference between the computational time for SWAE and WAE-GAN.

For the MNIST experiment and on a single NVIDIA Tesla P 100 GPU, each batch iteration (batchsize=500) of WAE-GAN took 0.2571 ± 0.0435(sec) while SWAE (with L = 50 projections) took 0.2437 ± 0.0391(sec).

DISPLAYFORM0 The distribution in the 64-dimensional latent space, q Z , was set to Normal.

We also report the negative log-likelihood of {z i = φ(x i )} with repect to q Z for 1000 testing samples for both datasets.

We did not use Nowizin's trick for the GAN models.

DISPLAYFORM1 True Data 2 3 Table 2 : FID score statistics (N = 5) at final iteration of training.

Lower is better.

Scores were computed with 10 4 random samples from the testing set against an equivalent amount of generated samples.

The CelebA face and the LSUN bedroom datasets contain higher degrees of variations compared to the MNIST dataset and therefore a two-dimensional latent-space does not suffice to capture the variations in these datasets (See supplementary material for more details on the dimensionality of the latent space).

We used a K = 64 dimensional latent spaces for both the CelebA and the LSUN Bedroom datasets, and also used a larger auto-encoder (i.e., DCGAN, following the work of ).

For these datasets SWAE was trained with q Z being the Normal distribution to enable the calculation of the negative log likelihood (NLL).

TAB1 shows the comparison between SWAE and WAE for these two datasets.

We note that all experimental parameters were kept the same to enable an apples to apples comparison.

Finally, FIG4 demonstrates the interpolation between two sample points in the latent space, i.e. ψ(tφ(I 0 ) + (1 − t)φ(I 1 )) for t ∈ [0, 1], for all three datasets.

We introduced Sliced Wasserstein auto-encoders (SWAE), which enable one to shape the distribution of the encoded samples to any samplable distribution without the need for adversarial training or having a likelihood function specified.

In addition, we provided a simple and efficient numerical scheme for this problem, which only relies on few inner products and sorting operations in each SGD iteration.

We further demonstrated the capability of our method on three image datasets, namely the MNIST, the CelebA face, and the LSUN Bedroom datasets, and showed competitive performance, in the sense of matching distributions p Z and q Z , to the techniques that rely on additional adversarial trainings.

Finally, we envision SWAE could be effectively used in transfer learning and domain adaptation algorithms where q Z comes from a source domain and the task is to encode the target domain p X in a latent space such that the distribution follows the distribution of the target domain.

Figure 4: Sample convergence behavior for our method compared to the WAE-GAN, where q Z is set to a ring distribution FIG5 , top left).

The columns represent batch iterations (batchsize= 500).The top half of the table shows results of ψ(z) for z ∼ q Z , and the bottom half shows z ∼ q Z and φ(x) for x ∼ p X .

It can be seen that the adversarial loss in the latent space does not provide a full coverage of the distribution, which is a similar problem to the well-known 'mode collapse' problem in the GANs.

It can be seen that SWAE provides a superior match between p Z and q Z while it does not require adversarial training.

Following the example by and later here we show a simple example comparing the Jensen-Shannon divergence with the Wasserstein distance.

First note that the Jensen-Shannon divergence is defined as, DISPLAYFORM0 where DISPLAYFORM1 q(x) )dx is the Kullback-Leibler divergence.

Now consider the following densities, p(x) be a uniform distribution around zero and let q τ (x) = p(x − τ ) be a shifted version of the p. FIG6 show W 1 (p, q τ ) and JS(p, q τ ) as a function of τ .

As can be seen the JS divergence fails to provide a useful gradient when the distributions are supported on non-overlapping domains.

To maximize (minimize) the similarity (dissimilarity) between p Z and q Z , we can write : DISPLAYFORM0 where we replaced p Z with equation 1.

Furthermore, it is straightforward to show: DISPLAYFORM1 The Wasserstein distance between the two probability measures ρ X and ρ Y with respective densities p X and p Y , can be measured via the Kantorovich formulation of the optimal mass transport problem: DISPLAYFORM2 c(x, y)γ(x, y)dxdy DISPLAYFORM3 } is the set of all transportation plans (i.e., couplings or joint distributions) over p X and p Y .

Now, note that the two step process of encoding p X into the latent space Z and decoding it to p Y , provides a unique decomposition of γ as γ 0 (x, y) = δ(y − ψ(φ(x)))p X (x) ∈ Γ. The optimal coupling (i.e., transport plan) between p X and p Y could be equal or different from γ(x, y) = δ(y −ψ(φ(x)))p X (x).

This leads to the scenario on the right where DISPLAYFORM4 Therefore we can write: DISPLAYFORM5 which proves equation 3.

Finally, taking the infimum of the two sides of the inequality, with respect to φ and ψ, we have: DISPLAYFORM6 is non-zero.

Finally, we note that ψ(φ(·)) = id(·) is a global optima for both W c (p X , p Y ) and W ‡ c (p X , p Y ).

Following equation 10 a distribution can be sliced via: Figure 9 demonstrates the outputs of trained SWAEs with K = 2 and K = 128 for sample input images.

The input images were resized to 64 × 64 and then fed to our auto-encoder structure.

This effect can also be seen for the MNIST dataset as shown in FIG0 .

When the dimensionality of the latent-space (i.e. information bottleneck) is too low the latent space will not contain enough information to reconstruct crisp images.

Increasing the dimensionality of the latent space leads to crisper images.

DISPLAYFORM0

In this paper we also used the sliced Wasserstein distance as a measure of goodness of fit (for convergence analysis).

To provide a fair comparison between different methods, we avoided random projections for this comparison.

Instead, we calculated a discriminant subspace to separate ψ(z) from ψ(φ(x)) for z ∼ q Z and x ∼ p X , and set the projection parameters θs to the calculated discriminant components.

This will lead to only slices that contain discriminant information.

We point out that the linear discriminant analysis (LDA) is not a good choice for this task as it only leads to one discriminant component (because we only have two classes).

We used the penalized linear discriminant analysis (p-LDA) that utilizes a combination of LDA and PCA.

In short, p-LDA solves the following objective function:argmax θ θ T S T θ θ T (S W + αI)θ s.t.

θ = 1 where S W is the within class covariance matrix, S T is the data covariance matrix, I is the identity matrix, and α identifies the interpolation between PCA and LDA (i.e. α = 0 leads to LDA and α → ∞ leads to PCA).

For p ≥ 1 we can use the triangle inequality and write DISPLAYFORM0 which leads to DISPLAYFORM1 Taking the expectation of both sides of the inequality and using the empirical convergence bounds of W p (in this case W 1 ) we have, DISPLAYFORM2 for some absolute constant C, where the last line comes from the empirical convergence bounds of distributions with respect to the Wasserstein distance, see BID2 .

@highlight

In this paper we use the sliced-Wasserstein distance to shape the latent distribution of an auto-encoder into any samplable prior distribution. 