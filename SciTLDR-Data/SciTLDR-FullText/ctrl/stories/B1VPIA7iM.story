In this paper we study generative modeling via autoencoders while using the elegant geometric properties of the optimal transport (OT) problem and the Wasserstein distances.

We introduce Sliced-Wasserstein Autoencoders (SWAE), which are generative models that enable one to shape the distribution of the latent space into any samplable probability distribution without the need for training an adversarial network or defining a closed-form for the distribution.

In short, we regularize the autoencoder loss with the sliced-Wasserstein distance between the distribution of the encoded training samples and a predefined samplable distribution.

We show that the proposed formulation has an efficient numerical solution that provides similar capabilities to Wasserstein Autoencoders (WAE) and Variational Autoencoders (VAE), while benefiting from an embarrassingly simple implementation.

Scalable generative models that capture the rich and often nonlinear distribution of highdimensional data, (i.e., image, video, and audio), play a central role in various applications of machine learning, including transfer learning BID13 BID24 , super-resolution BID15 BID20 , image inpainting and completion BID34 , and image retrieval BID6 , among many others.

The recent generative models, including Generative Adversarial Networks (GANs) BID0 BID1 BID10 BID29 and Variational Autoencoders (VAE) BID4 BID14 BID23 enable an unsupervised and end-to-end modeling of the high-dimensional distribution of the training data.

Learning such generative models boils down to minimizing a dissimilarity measure between the data distribution and the output distribution of the generative model.

To this end, and following the work of Arjovsky et al. BID0 and Bousquet et al. BID4 we approach the problem of generative modeling from the optimal transport point of view.

The optimal transport problem BID17 BID33 provides a way to measure the distances between probability distributions by transporting (i.e., morphing) one distribution into another.

Moreover, and as opposed to the common information theoretic dissimilarity measures (e.g., f -divergences), the p-Wasserstein dissimilarity measures that arise from the optimal transport problem: 1) are true distances, and 2) metrize a weak convergence of probability measures (at least on compact spaces).

Wasserstein distances have recently attracted a lot of interest in the learning community BID0 BID4 BID8 BID11 BID17 due to their exquisite geometric characteristics BID30 .

See the supplementary material for an intuitive example showing the benefit of the Wasserstein distance over commonly used f -divergences.

In this paper, we introduce a new type of autoencoders for generative modeling (Algorithm 1), which we call Sliced-Wasserstein Autoencoders (SWAE), that minimize the sliced-Wasserstein distance between the distribution of the encoded samples and a predefined samplable distribution.

Our work is most closely related to the recent work by Bousquet et al. BID4 and the followup work by Tolstikhin et al. BID32 .

However, our approach avoids the need to perform costly adversarial training in the encoding space and is not restricted to closed-form distributions, while still benefiting from a Wasserstein-like distance measure in the encoding space that permits a simple numerical solution to the problem.

In what follows we first provide an extensive review of the preliminary concepts that are needed for our formulation.

In Section 3 we formulate our proposed method.

The proposed numerical scheme to solve the problem is presented in Section 4.

Our experiments are summarized in Section 5.

Finally, our work is Concluded in Section 6.

Let X denote the compact domain of a manifold in Euclidean space and let x n ∈ X denote an individual input data point.

Furthermore, let ρ X be a Borel probability measure defined on X. We define the probability density function p X (x) for input data x to be: DISPLAYFORM0 Let φ : X → Z denote a deterministic parametric mapping from the input space to a latent space Z (e.g., a neural network encoder).

Utilizing a technique often used in the theoretical physics community (See BID9 ), known as Random Variable Transformation (RVT), the probability density function of the encoded samples z can be expressed in terms of φ and p X by: DISPLAYFORM1 where δ denotes the Dirac distribution function.

The main objective of Variational Auto-Encoders (VAEs) is to encode the input data points x ∈ X into latent codes z ∈ Z such that: 1) x can be recovered/approximated from z, and 2) the probability density function of the encoded samples, p Z , follows a prior distribution q Z .

Similar to classic auto-encoders, a decoder ψ : Z → X is required to map the latent codes back to the original space such that DISPLAYFORM2 where y denotes the decoded samples.

It is straightforward to see that when ψ = φ −1 (i.e. ψ(φ(·)) = id(·)), the distribution of the decoder p Y and the input distribution p X are identical.

Hence, the objective of a variational auto-encoder simplifies to learning φ and ψ such that they minimize a dissimilarity measure between p Y and p X , and between p Z and q Z .

Defining and implementing the dissimilarity measure is a key design decision, and is one of the main contributions of this work, and thus we dedicate the next section to describing existing methods for measuring these dissimilarities.

We first emphasize that the VAE work in the literature often assumes stochastic encoders and decoders BID14 , while we consider the case of only deterministic mappings.

Different dissimilarity measures have been used between p X and p Y in various work in the literature.

Most notably, Nowozin et al. BID25 showed that for the general family of f -divergences, D f (p X , p Y ), (including the KL-divergence, Jensen-Shannon, etc.), using the Fenchel conjugate of the convex function f and minimizing D f (p X , p Y ) leads to a min-max problem that is equivalent to the adversarial training widely used in the generative modeling literature BID10 BID22 BID23 .Others have utilized the rich mathematical foundation of the OT problem and Wasserstein distances BID0 BID4 BID11 BID32 .

In Wasserstein-GAN, BID0 utilized the Kantorovich-Rubinstein duality for the 1-Wasserstein distance, W 1 (p X , p Y ), and reformulated the problem as a min-max optimization that is solved through an adversarial training scheme.

In a different approach, BID4 utilized the autoencoding nature of the problem and showed that W c (p X , p Y ) could be simplified as: DISPLAYFORM0 Note that Eq. (3) is equivalent to Theorem 1 in BID4 for deterministic encoder-decoder pair, and also note that φ and ψ are parametric differentiable models (e.g. neural networks).

Furthermore, Eq. (3) supports a simple implementation where for i.i.d samples of the input distribution {x n } N n=1 the minimization can be written as: DISPLAYFORM1 We emphasize that Eq. (3) (and consequently Eq. FORMULA4 ) takes advantage of the fact that the pairs x n and y n = ψ(φ(x n )) are available, hence calculating the transport distance coincides with summing the transportation costs between all pairs (x n , y n ).

For example, the total transport distance may be defined as the sum of Euclidean distances between all pairs of points.

In this paper, we also use W c (p X , p Y ) following Eq. (4) to measure the discrepancy between p X and p Y .

Next, we review the methods used for measuring the discrepancy between p Z and q Z .

If q Z is a known distribution with an explicit formulation (e.g. Normal distribution) the most straightforward approach for measuring the (dis)similarity between p Z and q Z is the loglikelihood of z = φ(x) with respect to q Z , formally: DISPLAYFORM0 maximizing the log-likelihood is equivalent to minimizing the KL-divergence between p Z and q Z , D KL (p Z , q Z ) (see supplementary material for more details and derivation of Equation FORMULA5 ).

This approach has two major limitations: 1) The KL-Divergence and in general f -divergences do not provide meaningful dissimilarity measures for distributions supported on non-overlapping low-dimensional manifolds BID0 BID18 (see supplementary material), which is common in hidden layers of neural networks, and therefore they do not provide informative gradients for training φ, and 2) we are limited to distributions q Z that have known explicit formulations, which is very restrictive because it eliminates the ability to use the much broader class of distributions were we know how to sample from them, but do not know their explicit form.

Various alternatives exist in the literature to address the above-mentioned limitations.

These methods often sampleZ = {z j } N j=1 from q Z and Z = {z n = φ(x n )} N n=1 from p X and measure the discrepancy between these sets (i.e. point clouds).

Note that there are no one-to-one correspondences betweenz j s and z n s. Tolstikhin et al. BID32 for instance, proposed two different approaches for measuring the discrepancy betweenZ and Z, namely the GAN-based and the maximum mean discrepancy (MMD)-based approaches.

The GAN-based approach proposed in BID32 defines a discriminator network, D Z (p Z , q Z ), to classifyz j s and z n s as coming from 'true' and 'fake' distributions correspondingly and proposes a min-max adversarial optimization for learning φ and D Z .

This approach could be thought as a Fenchel conjugate of some f -divergence between p Z and q Z .

The MMD-based approach, on the other hand, utilizes a positive-definite reproducing kernel k : Z × Z → R to measure the discrepancy betweenZ and Z, however, the choice of the kernel remain a data-dependent design parameter.

An interesting alternative approach is to use the Wasserstein distance between p Z and q Z .

The reason being that Wasserstein metrics have been shown to be particularly beneficial for measuring the distance between distributions supported on non-overlapping low-dimensional manifolds.

Following the work of Arjovsky et al. BID0 , this can be accomplished utilizing the Kantorovich-Rubinstein duality and through introducing a min-max problem, which leads to yet another adversarial training scheme similar the GAN-based method in BID32 .

Note that, since elements ofZ and Z are not paired an approach similar to Eq. (4) could not be used to calculate the Wasserstein distance.

In this paper, we propose to use the sliced-Wasserstein metric, [3, BID5 BID16 BID18 BID27 BID28 , to measure the discrepancy between p Z and q Z .

We show that using the slicedWasserstein distance ameliorates the need for training an adversary network, and provides an efficient but yet simple numerical implementation.

Before explaining our proposed approach, it is worthwhile to point out the benefits of learning autoencoders as generative models over GANs.

In GANs, one needs to minimize a distance between {ψ(z j )|z j ∼ q Z } M j=1 and {x n } M n=1 which are high-dimensional point clouds for which there are no correspondences between ψ(z j )s and x n s.

For the autoencoders, on the other hand, there exists correspondences between the high-dimensional point clouds {x n } M n=1 and {y n = ψ(φ(x n ))} M n=1 , and the problem simplifies to matching the lower-dimensional point clouds {φ(x n )} M n=1 and {z j ∼ q Z } M j=1 .

In other words, the encoder performs a nonlinear dimensionality reduction, that enables us to solve a much simpler problem compared to GANs.

Next we introduce the details of our approach.

In what follows we first provide a brief review of the necessary equations to understand the Wasserstein and sliced-Wasserstein distances and then present our Sliced Wassersten Autoencoders (SWAE).

The Wasserstein distance between probability measures ρ X and ρ Y , with corresponding densities dρ X = p X (x)dx and dρ Y = p Y (y)dy is defined as: DISPLAYFORM0 where Γ(ρ X , ρ Y ) is the set of all transportation plans (i.e. joint measures) with marginal densities p X and p Y , and c : X × Y → R + is the transportation cost.

Eq. FORMULA6 is known as the Kantorovich formulation of the optimal mass transportation problem, which seeks the optimal transportation plan between p X and p Y .

If there exist diffeomorphic mappings, f : X → Y (i.e. transport maps) such that y = f (x) and consequently, DISPLAYFORM1 where det(D·) is the determinant of the Jacobian, then the Wasserstein distance could be defined based on the Monge formulation of the problem (see BID33 and BID17 ) as: DISPLAYFORM2 where MP is the set of all diffeomorphisms that satisfy Eq. (7).

As can be seen from Eqs. FORMULA6 and FORMULA8 , obtaining the Wasserstein distance requires solving an optimization problem.

Various efficient optimization techniques have been proposed in the past (e.g. BID7 BID26 BID31 ).

The case of one dimensional probability densities, p X and p Y , is specifically interesting as the Wasserstein distance has a closed-form solution.

Let P X and P Y be the cumulative distributions of one-dimensional probability distributions p X and p Y , correspondingly.

The Wassertein distance can then be calculated as: DISPLAYFORM3 The closed-form solution of Wasserstein distance for one-dimensional probability densities motivates the definition of sliced-Wasserstein distances.

The interest in the sliced-Wasserstein distance is due to the fact that it has very similar qualitative properties as the Wasserstein distance, but it is much easier to compute, since it only depends on one-dimensional computations.

The sliced-Wasserstein distance was used in BID27 BID28 to calculate barycenter of distributions and point clouds.

Bonneel et al. [3] provided a nice theoretical overview of barycenteric calculations using the sliced-Wasserstein distance.

Kolouri et al. BID16 used this distance to define positive definite kernels for distributions and Carriere et al. BID5 used it as a distance for persistence diagrams.

Sliced-Wasserstein was also recently used for learning Gaussian mixture models BID18 .The main idea behind the sliced-Wasserstein distance is to slice (i.e. project) higherdimensional probability densities into sets of one-dimensional distributions and compare their one-dimensional representations via Wasserstein distance.

The slicing/projection process is related to the field of Integral Geometry and specifically the Radon transform BID12 .

The relevant result to our discussion is that a d-dimensional probability density p X could be uniquely represented as the set of its one-dimensional marginal distributions following the Radon transform and the Fourier slice theorem BID12 .

These one dimensional marginal distributions of p X are defined as: DISPLAYFORM0 where S d−1 is the d-dimensional unit sphere.

Note that for any fixed θ ∈ S d−1 , Rp X (·; θ) is a one-dimensional slice of distribution p X .

In other words, Rp X (·; θ) is a marginal distribution of p X that is obtained from integrating p X over the hyperplane orthogonal to θ (See FIG0 .

Utilizing the one-dimensional marginal distributions in Eq. (10), the sliced Wasserstein distance could be defined as: DISPLAYFORM1 Given that Rp X (·; θ) and Rp Y (·; θ) are one-dimensional the Wasserstein distance in the integrand has a closed-form solution as demonstrated in BID8 .

The fact that SW c is a distance comes from W c being a distance.

Moreover, the two distances also induce the same topology, at least on compact sets BID30 .A natural transportation cost that has extensively studied in the past is the 2 2 , c(x, y) = x − y 2 2 , for which there are theoretical guarantees on existence and uniqueness of transportation plans and maps (see BID30 and BID33 ).

When c(x, y) = x − y 2 2 the following inequality bounds hold for the SW distance: DISPLAYFORM2 where α is a constant.

Chapter 5 in BID3 proves this inequality with β = (2(d + 1)) −1 (See BID30 for more details).

The inequalities in FORMULA1 is the main reason we can use the sliced Wasserstein distance, SW 2 , as an approximation for W 2 .

Our proposed formulation for the SWAE is as follows: DISPLAYFORM0 where φ is the encoder, ψ is the decoder, p X is the data distribution, p Y is the data distribution after encoding and decoding (Eq. FORMULA2 ), p Z is the distribution of the encoded data (Eq. FORMULA1 ), q Z is the predefined distribution (or a distribution we know how to sample from), and λ is a hyperparameter that identifies the relative importance of the loss functions.

To further clarify why we use the Wasserstein distance to measure the difference between p X and p Y , but the sliced-Wasserstein distance to measure the difference between p Z and q Z , we reiterate that the Wasserstein distance for the first term can be solved via Eq. (4) due to the existence of correspondences between y n and x n (i.e., we desire x n = y n ), however, for p Z and q Z , analogous correspondences between thez i s and z j s do not exist and therefore calculation of the Wasserstein distance requires an additional optimization step (e.g., in the form of an adversarial network).

To avoid this additional optimization, while maintaining the favorable characteristics of the Wasserstein distance, we use the sliced-Wasserstein distance to measure the discrepancy between p Z and q Z .

The Wasserstein distance between two one-dimensional distributions p X and p Y is obtained from Eq. BID8 .

The integral in Eq. (9) could be numerically calculated using FIG2 .

Therefore, the Wasserstein distance can be approximated by first sorting x m s and y m s and then calculating: DISPLAYFORM0 DISPLAYFORM1 Eq. FORMULA1 turns the problem of calculating the Wasserstein distance for two one-dimensional probability densities from their samples into a sorting problem that can be solved efficiently (O(M) best case and O(Mlog(M)) worst case).

In scenarios where only samples from the d-dimensional distribution, p X , are available, x m ∼ p X , the empirical distribution can be estimated as FORMULA1 it is straightforward to show that the marginal distributions (i.e. slices) of the empirical distribution, p X , are obtained from: DISPLAYFORM0 DISPLAYFORM1 , and ∀t ∈ R BID14 see the supplementary material for a proof.

Minimizing the sliced-Wasserstein distance (i.e. as in the second term of Eq. 13) requires an integration over the unit sphere in R d , i.e., S d−1 .

In practice, this integration is substituted by a summation over a finite set DISPLAYFORM0 .

A fine sampling of S d−1 is required for a good approximation of SW c (p Z , q Z ).

Such sampling, however, becomes prohibitively expensive as the dimension of the embedding space grows.

Alternatively, following the approach presented by Rabin and Peyré BID27 , and later by Bonneel et al. [3] and subsequently by Kolouri et al. BID18 , we utilize random samples of S d−1 at each minimization step to approximate the sliced-Wasserstein distance.

Intuitively, if p Z and q Z are similar, then their projections with respect to any finite subset of S d−1 would also be similar.

This leads to a stochastic gradient descent scheme where in addition to the random sampling of the input data, we also random sample the projection angles from S d−1 .

Require: Regularization coefficient λ, and number of random projections, L. Initialize the parameters of the encoder, φ, and decoder, ψ while φ and ψ have not converged do Sample {x 1 , ..., x M } from training set (i.e. p X ) Sample {z 1 , ..., DISPLAYFORM0 end while

To optimize the proposed SWAE objective function in Eq. FORMULA1 we use a stochastic gradient descent scheme as described here.

In each iteration, let {x m ∼ p X } M m=1 and {z m ∼ q Z } M m=1 be i.i.d random samples from the input data and the predefined distribution, q Z , correspondingly.

Let {θ l } L l=1 be randomly sampled from a uniform distribution on S d−1 .

Then using the numerical approximations described in this section, the loss function in Eq. FORMULA1 can be rewritten as: It is worth pointing out that sorting is by itself an optimization problem (which can be solved very efficiently), and therefore the sorting followed by the gradient descent update on φ and ψ is in essence a min-max problem, which is being solved in an alternating fashion.

DISPLAYFORM0

Here we show the results of SWAE for two mid-size image datasets, namely the MNIST dataset BID19 , and the CelebFaces Attributes Dataset (CelebA) BID21 .

For the encoder and the decoder we used mirrored classic deep convolutional neural networks with 2D average poolings and leaky rectified linear units (Leaky-ReLu) as the activation functions.

The implementation details are included in the Supplementary material.

For the MNIST dataset, we designed a deep convolutional encoder that embeds the handwritten digits into a two-dimensional embedding space (for visualization).

To demonstrate the capability of SWAE on matching distributions p Z and q Z in the embedding/encoder space we chose four different q Z s, namely the ring distribution, the uniform distribution, a circle distribution, and a bowl distribution.

FIG4 shows the results of our experiment on the MNIST dataset.

The left column shows samples from q Z , the middle column shows φ(x n )s for the trained φ and the color represent the labels (note that the labels were only used for visualization).

Finally, the right column depicts a 25 × 25 grid in [−1, 1] 2 through the trained decoder ψ.

As can be seen, the embedding/encoder space closely follows the predefined q Z , while the space remains decodable.

The implementation details are included in the supplementary material.

The CelebA face dataset contains a higher degree of variations compared to the MNIST dataset and therefore a two-dimensional embedding space does not suffice to capture the variations in this dataset.

Therefore, while the SWAE loss function still goes down and the network achieves a good match between p Z and q Z the decoder is unable to match p X and p Y .

Therefore, a higher-dimensional embedding/encoder space is needed.

In our experiments for this dataset we chose a (K = 128)−dimensional embedding space.

FIG5 demonstrates the outputs of trained SWAEs with K = 2 and K = 128 for sample input images.

The input images were resized to 64 × 64 and then fed to our autoencoder structure.

For CelebA dataset we set q Z to be a (K = 128)-dimensional uniform distribution and trained our SWAE on the CelebA dataset.

Given the convex nature of q Z , any linear combination of the encoded faces should also result in a new face.

Having that in mind, we ran two experiments in the embedding space to check that in fact the embedding space satisfies this convexity assumption.

First we calculated linear interpolations of sampled pairs of faces in the embedding space and fed the interpolations to the decoder network to visualize the corresponding faces.

FIG6 , left column, shows the interpolation results for random pairs of encoded faces.

It is clear that the interpolations remain faithful as expected from a uniform q Z .

Finally, we performed Principle Component Analysis (PCA) of the encoded faces and visualized the faces corresponding to these principle components via ψ.

The PCA components are shown on the left column of FIG6 .

Various interesting modes including, hair color, skin color, gender, pose, etc. can be observed in the PC components.

We introduced Sliced Wasserstein Autoencoders (SWAE), which enable one to shape the distribution of the encoded samples to any samplable distribution.

We theoretically showed that utilizing the sliced Wasserstein distance as a dissimilarity measure between the distribution of the encoded samples and a predefined distribution ameliorates the need for training an adversarial network in the embedding space.

In addition, we provided a simple and efficient numerical scheme for this problem, which only relies on few inner products and sorting operations in each SGD iteration.

We further demonstrated the capability of our method on two mid-size image datasets, namely the MNIST dataset and the CelebA face dataset and showed results comparable to the techniques that rely on additional adversarial trainings.

Our implementation is publicly available BID0 .

This work was partially supported by NSF (CCF 1421502).

The authors would like to thank Drs.

Dejan Slepćev, and Heiko Hoffmann for their invaluable inputs and many hours of constructive conversations.

FIG0 and JS(p, q τ ) where p is a uniform distribution around zero and q τ (x) = p(x − τ).

It is clear that JS divergence does not provide a usable gradient when distributions are supported on non-overlapping domains.

Following the example by Arjovsky et al. BID0 and later Kolouri et al. BID18 here we show a simple example comparing the Jensen-Shannon divergence with the Wasserstein distance.

First note that the Jensen-Shannon divergence is defined as, DISPLAYFORM0 where KL(p, q) = X p(x)log( DISPLAYFORM1 )dx is the Kullback-Leibler divergence.

Now consider the following densities, p(x) be a uniform distribution around zero and let q τ (x) = p(x − τ) be a shifted version of the p. FIG7 show W 1 (p, q τ ) and JS(p, q τ ) as a function of τ.

As can be seen the JS divergence fails to provide a useful gradient when the distributions are supported on non-overlapping domains.

To maximize (minimize) the similarity (dissimilarity) between p Z and q Z , we can write : argmax φ Z p Z (z)log(q Z (z))dz = Z X p X (x)δ(z − φ(x))log(q Z (z))dxdz = X p X (x)log(q Z (φ(x)))dx where we replaced p Z with Eq. (1).

Furthermore, it is straightforward to show: DISPLAYFORM0

Here we calculate a Radon slice of the empirical distribution p X (x) = we have: DISPLAYFORM0 Simple manifold learning experiment Figure 7 demonstrates the results of SWAE with random initializations to embed a 2D manifold in R 3 to a 2D uniform distribution.

The following text walks you through the implementation of our Sliced Wasserstein Autoencoders (SWAE).To run this notebook you'll require the following packages:In BID20 : #Visualize the z samples plt.

FIG0

<|TLDR|>

@highlight

"Generative modeling with no need for adversarial training"