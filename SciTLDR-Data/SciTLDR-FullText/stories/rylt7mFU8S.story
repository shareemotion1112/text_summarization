Many biological learning systems such as the mushroom body, hippocampus, and cerebellum are built from sparsely connected networks of neurons.

For a new understanding of such networks, we study the function spaces induced by sparse random features and characterize what functions may and may not be learned.

A network with d inputs per neuron is found to be equivalent to an additive model of order d, whereas with a degree distribution the network combines additive terms of different orders.

We identify three specific advantages of sparsity: additive function approximation is a powerful inductive bias that limits the curse of dimensionality, sparse networks are stable to outlier noise in the inputs, and sparse random features are scalable.

Thus, even simple brain architectures can be powerful function approximators.

Finally, we hope that this work helps popularize kernel theories of networks among computational neuroscientists.

Kernel function spaces are popular among machine learning researchers as a potentially tractable framework for understanding artificial neural networks trained via gradient descent [e.g. 1, 2, 3, 4, 5, 6].

Artificial neural networks are an area of intense interest due to their often surprising empirical performance on a number of challenging problems and our still incomplete theoretical understanding.

Yet computational neuroscientists have not widely applied these new theoretical tools to describe the ability of biological networks to perform function approximation.

The idea of using fixed random weights in a neural network is primordial, and was a part of Rosenblatt's perceptron model of the retina [7] .

Random features have then resurfaced under many guises: random centers in radial basis function networks [8] , functional link networks [9] , Gaussian processes (GPs) [10, 11] , and so-called extreme learning machines [12] ; see [13] for a review.

Random feature networks, where the neurons are initialized with random weights and only the readout layer is trained, were proposed by Rahimi and Recht in order to improve the performance of kernel methods [14, 15] and can perform well for many problems [13] .

In parallel to these developments in machine learning, computational neuroscientists have also studied the properties of random networks with a goal towards understanding neurons in real brains.

To a first approximation, many neuronal circuits seem to be randomly organized [16, 17, 18, 19, 20] .

However, the recent theory of random features appears to be mostly unknown to the greater computational neuroscience community.

Here, we study random feature networks with sparse connectivity: the hidden neurons each receive input from a random, sparse subset of input neurons.

This is inspired by the observation that the connectivity in a variety of predominantly feedforward brain networks is approximately random and sparse.

These brain areas include the cerebellar cortex, invertebrate mushroom body, and dentate gyrus of the hippocampus [21] .

All of these areas perform pattern separation and associative learning.

The cerebellum is important for motor control, while the mushroom body and dentate gyrus are The function shown is the sparse random feature approximation to an additive sum of sines, learned from poorly distributed samples (red crosses).

Additivity offers structure which may be leveraged for fast and efficient learning.

general learning and memory areas for invertebrates and vertebrates, respectively, and may have evolved from a similar structure in the ancient bilaterian ancestor [22] .

Recent work has argued that the sparsity observed in these areas may be optimized to balance the dimensionality of representation with wiring cost [20] .

Sparse connectivity has been used to compress artificial networks and speed up computation [23, 24, 25] , whereas convolutions are a kind of structured sparsity [26, 27] .

We show that sparse random features approximate additive kernels [28, 29, 30, 31] with arbitrary orders of interaction.

The in-degree of the hidden neurons d sets the order of interaction.

When the degrees of the neurons are drawn from a distribution, the resulting kernel contains a weighted mixture of interactions.

These sparse features offer advantages of generalization in high-dimensions, stability under perturbations of their input, and computational and biological efficiency.

Now we will introduce the mathematical setting and review how random features give rise to kernels.

The simplest artificial neural network contains a single hidden layer, of size m, receiving input from a layer of size l (Figure 1 ).

The activity in the hidden layer is given by, for i ∈ [m],

Here each φ i is a feature in the hidden layer, h is the nonlinearity, W = (w 1 , w 2 , . . .

, w m ) ∈ R l×m are the input to mixed weights, and b ∈ R m are their biases.

We can write this in vector notation as

Random features networks draw their input-hidden layer weights at random.

Let the weights w i and biases b i in the feature expansion (1) be sampled i.i.d.

from a distribution µ on R l+1 .

Under mild assumptions, the inner product of the feature vectors for two inputs converges to its expectation

We identify the limit (2) with a reproducing kernel k(x, x ) induced by the random features, since the limiting function is an inner product and thus always positive semidefinite [14] .

The kernel defines an associated reproducing kernel Hilbert space (RKHS) of functions.

For a finite network of width m, the inner product 1 m φ(x) φ(x ) is a randomized approximation to the kernel k(x, x ).

We now turn to our main result: the general form of the random feature kernels with sparse, independent weights.

For simplicity, we start with a regular model and then generalize the result to networks with varying in-degree.

Two kernels that can be computed in closed form are highlighted.

Fix an in-degree d, where 1 ≤ d ≤ l, and let µ|d be a distribution on R d which induce, together with some nonlinearity h, the kernel

denote this set of neighbors.

Second, sample w ji ∼ µ|d if j ∈ N i and otherwise set w ji = 0.

We find that the resulting kernel

Here x N denotes the length d vector of x restricted to the neighborhood N , with the other l − d entries in x ignored.

More generally, the in-degrees may be chosen independently according to a degree distribution, so that d becomes a random variable.

Let D(d) be the probability mass function of the hidden node in-degrees.

Conditional on node i having degree d i , the in-neighborhood N i is chosen uniformly at random among the l di possible sets.

Then the induced kernel becomes

For example, if every layer-two node chooses its inputs independently with probability p, the

is the probability mass function of the binomial distribution Bin(l, p).

The regular model (3) is a special case of (4) with

Extending the proof techniques in [14] yields:

C ) many features (the proof is contained in Appendix C).

Two simple examples With Gaussian weights and regular d = 1, we find that (see Appendix B)

4 Advantages of sparse connectivity

The regular degree kernel (3) is a sum of kernels that only depend on combinations of d inputs, making it an additive kernel of order d. The general expression for the degree distribution kernel (4) illustrates that sparsity leads to a mixture of additive kernels of different orders.

These have been referred to as additive GPs [30] , but these kind of models have a long history as generalized additive models [e.g. 28, 32] .

For the regular degree model with d = 1, the sum in (3) is over neighborhoods of size one, simply the individual indices of the input space.

Thus, for any two input neighborhoods N 1 and N 2 , we have |N 1 ∩ N 2 | = ∅, and the RKHS corresponding to k reg 1 (x, x ) is the direct sum of the subspaces H = H 1 ⊕ . . .

⊕ H l .

Thus regular d = 1 defines a first-order additive model, where

, all pairwise terms.

These interactions are defined by the structure of the terms k d (x N , x N ).

Finally, the degree distribution D(d) determines how much weight to place on different degrees of interaction.

Generalization from fewer examples in high dimensions Stone proved that first-order additive models do not suffer from the curse of dimensionality [33, 34] , as the excess risk does not depend on the dimension l. Kandasamy and Yu [31] extended this result to dth-order additive models and found a bound on the excess risk of O(l 2d n −2s

for kernels with polynomial or exponential eigenvalue decay rates (n is the number of samples and the constants s and C parametrize rates).

Without additivity, these weaken to O(n −2s 2s+l ) and O(C l /n), much worse when l d.

Similarity to dropout Dropout regularization [35, 36] in deep networks has been analyzed in a kernel/GP framework [37] , leading to (4) with D = Bin(l, p) for a particular base kernel.

Dropout may thus improve generalization by enforcing approximate additivity, for the reasons above.

Equations (5) and (6) are similar: They differ only by the presence of an 0 -"norm" versus an 1 -norm and the presence of the sign function.

Both norms are stable to outlying coordinates in an input x. This property also holds for different nonlinearities and 1 < d l, since every feature φ i (x) only depends on d inputs, and therefore only a minority of the m features will be affected by the few outliers.

1 Sufficiently sparse features will then be less affected by sparse noise than a fully-connected network, offering denoising advantages [e.g .

20] .

A regressor f (x) = α φ(x) built from these features is stable so long as α p is small, since |f (x) − f (x )| ≤ α p φ(x) − φ(x ) q for any Hölder conjugates 1/p + 1/q = 1.

Thus if x = x + e where e contains a small number of nonzero entries, then f (x ) ≈ f (x) since φ(x) ≈ φ(x ).

Stability also may guarantee the robustness of these networks to sparse adversarial attacks [38, 39, 40] , although exactly the conditions under which these approximations hold (p = ∞, q = 1 is an interesting case)

we leave for future work.

Computational Sparse random features give potentially huge improvements in scaling.

Direct implementations of additive models incur a large cost for d > 1, since (3) requires a sum over

time to compute the Gram matrix of n examples and O(nl d ) operations to evaluate f (x).

In our case, since the random features method is primal, we need to perform O(nmd) computations to evaluate the feature matrix and the cost of evaluating f (x) remains O(md).

3 Sparse matrix-vector multiplication makes evaluation faster than the O(ml) time it takes when connectivity is dense.

For ridge regression, we have the usual advantages that computing an estimator takes O(nm 2 + nmd) time and O(nm + md) memory, rather than O(n 3 ) time and O(n 2 ) memory for a naïve kernel ridge method.

Biological In a small animal such as a flying insect, space is extremely limited.

Sparsity offers a huge advantage in terms of wiring cost [20] .

Additive approximation also means that such animals can learn much more quickly, as seen in the mushroom body [41, 42, 43] .

While the previous computational points do not apply as well to biology, since real neurons operate in parallel, fewer operations translate into lower metabolic cost for the animal.

Inspired by their ubiquity in biology, we have studied sparse random networks of neurons using the theory of random features, finding the advantages of additivity, stability, and scalability.

This theory shows that sparse networks such as those found in the mushroom body, cerebellum, and hippocampus can be powerful function approximators.

Kernel theories of neural circuits may be more broadly applicable in the field of computational neuroscience.

Expanding the theory of dimensionality in neuroscience Learning is easier in additive function spaces because they are low-dimensional, a possible explanation for few-shot learning in biological systems.

Our theory is complementary to existing theories of dimensionality in neural systems [16, 44, 45, 46, 47, 20, 48, 49, 50] , which defined dimensionality using a skewness measure of covariance eigenvalues.

Kernel theory extends this concept, measuring dimensionality similarly [51] in the space of nonlinear functions spanned by the kernel.

Limitations We model biological neurons as simple scalar functions, completely ignoring time and neuromodulatory context.

It seems possible that a kernel theory could be developed for timeand context-dependent features.

Our networks suppose i.i.d.

weights, but weights that follow Dale's law should also be considered.

We have not studied the sparsity of activity, postulated to be relevant in cerebellum.

It remains to be demonstrated how the theory can make concrete, testable predictions, e.g. whether this theory may explain identity versus concentration encoding of odors or the discrimination/generalization tradeoff under experimental conditions.

Appendices: Additive function approximation in the brain As said in the main text, Kandasamy and Yu [1] created a theory of the generalization properties of higher-order additive models.

They supplemented this with an empirical study of a number of datasets using their Shrunk Additive Least Squares Approximation (SALSA) implementation of the additive kernel ridge regression (KRR).

Their data and code were obtained from https: //github.com/kirthevasank/salsa.

We compared the performance of SALSA to the sparse random feature approximation of the same kernel.

We employ random sparse Fourier features with Gaussian weights N (0, σ 2 I) with σ = 0.05 · √ dn 1/5 in order to match the Gaussian radial basis function used by Kandasamy and Yu [1] .

We use m = 300l features for every problem, with regular degree d selected equal to the one chosen by SALSA.

The regressor on the features is cross-validated ridge regression (RidgeCV from scikit-learn) with ridge penalty selected from 5 logarithmically spaced points between 10 −4 · n and 10 2 · n.

In Figure 2 , we compare the performance of sparse random features to SALSA.

Generally, the training and testing errors of the sparse model are slightly higher than for the kernel method, except for the forestfires dataset.

We studied the speed of learning for a test function as well.

The function to be learned f (x) was a sparse polynomial plus a linear term:

The linear term took a ∼ N (0, I), the polynomial p was chosen to have 3 terms of degree 3 with In Figure 3 , we show the test error as well as the selected ridge penalty for different values of d and n. With a small amount of data (n < 250), the model with d = 1 has the lowest test error, since this "simplest" model is less likely to overfit.

On the other hand, in the intermediate data regime (250 < n < 400), the model with d = 3 does best.

For large amounts of data (n > 400), all of the models with interactions d ≥ 3 do roughly the same.

Note that with the RBF kernel the RKHS

can still capture the degree 3 polynomial model.

However, we see that the more complex models have a higher ridge penalty selected.

The penalty is able to adaptively control this complexity given enough data.

Here we show that sparse random features are stable for spike-and-slab input noise.

In this example, the truth follows a linear model, where we have random input points x i ∼ N (0, I) and linear observations y i = x i β for i = 1, . . .

, n and β ∼ N (0, I).

However, we only have access to sparsely corruputed inputs w i = x i + e i , where e i = 0 with probability 1 − p and e i = x − x i with probability p, x ∼ N (0, σ 2 I).

That is, the corrupted inputs are replaced with pure noise.

We use p = 0.03 1 and σ = 6 1 so that the noise is sparse but large when it occurs.

In Table 1 we show the performance of various methods on this regression problem given the corrupted data (W, y).

Note that if the practitioner has access to the uncorrupted data X, linear regression succeeds with a perfect score of 1.

Using kernel ridge regression with k(x, x ) = 1 − Table 1 : Scores (R 2 coefficient) of various regression models on linear data with corrupted inputs.

In the presence of these errors, linear regression fails to acheive as good a test score as the kernel method, which is almost as good as trimming before performing regression and better than the robust Huber estimator.

Figure 4 : Kernel eigenvalue amplification while (left) varying p with σ = 6 fixed, and (right) varying σ with p = 0.03 fixed.

Plotted is the ratio of eigenvalues of the kernel matrix corrupted by noise to those without any corruption, ordered from largest to smallest in magnitude.

We see that the sparse feature kernel shows little noise amplification when it is sparse (right), even for large amplitude.

On the other hand, less sparse noise does get amplified (left).

best performance is attained by trimming the outliers and then performing linear regression.

However, this is meant to illustrate our point that sparse random features and their corresponding kernels may be useful when dealing with noisy inputs in a learning problem.

In Figure 4 we show another way of measuring this stability property.

We compute the eigenvalues of the kernel matrix on a fixed dataset of size n = 800 points both with noise and without noise.

Plotted are the ratio of the noisy to noiseless eigenvalues, in decibels, which we call the amplification and is a measure of how corrupted the kernel matrix is by this noise.

The main trend that we see is, for fixed p = 3, changing the amplitude of the noise σ does not lead to significant amplification, especially of the early eigenvalues which are of largest magnitude.

On the other hand, making the outliers denser does lead to more amplification of all the eigenvalues.

The eigenspace spanned by the largest eigenvalues is the most "important" for any learning problem.

We will now describe a number of common random features and the kernels they generate with fully-connected weights.

Later on, we will see how these change as sparsity is introduced in the input-hidden connections.

Translation invariant kernels The classical random features [2] sample Gaussian weights w ∼ N (0, σ −2 I), uniform biases b ∼ U [−a, a], and employ the Fourier nonlinearity h(·) = cos(·).

This leads to the Gaussian radial basis function kernel

In fact, every translation-invariant kernel arises from Fourier nonlinearities for some distributions of weights and biases (Bôchner's theorem).

Moment generating function kernels The exponential function is more similar to the kinds of monotone firing rate curves found in biological neurons.

In this case, we have k(x, x ) = E exp(w (x + x ) + 2b).

We can often evaluate this expectation using moment generating functions.

For example, if w and b are independent, which is a common assumption, then

where E (exp(w (x + x )) is the moment generating function for the marginal distribution of w, and E exp(2b) is just a constant that scales the kernel.

For multivariate Gaussian weights w ∼ N (m, Σ) this becomes

This equation becomes more interpretable if m = 0 and Σ = σ −2 I and the input data are normalized:

This result highlights that dot product kernels k(x, x ) = v(x x ) , where v : R → R, are radial basis functions on the sphere S l−1 = {x ∈ R l : x 2 = 1}. The eigenbasis of these kernels are the spherical harmonics [3, 4] .

Arc-cosine kernels This class of kernels is also induced by monotone "neuronal" nonlinearities and leads to different radial basis functions on the sphere [3, 5, 6] .

Consider standard normal weights w ∼ N (0, I) and nonlinearities which are threshold polynomial functions

+ , where Θ(·) is the Heaviside step function.

The kernel in this case is given by

for a known function J p (θ) where θ = arccos

.

Note that arc-cosine kernels are also dot product kernels.

Also, if the weights are drawn as w ∼ N (0, σ −2 I), the terms x are replaced by x/σ, but this does not affect θ.

With p = 0, corresponding to the step function nonlinearity, we have J 0 (θ) = π − θ, and the resulting kernel does not depend on x or x :

Sign nonlinearity We also consider a shifted version of the step function nonlinearity, the sign function sgn(z), equal to +1 when z > 0, −1 when z < 0, and zero when

and w ∼ P , where P is any spherically symmetric distribution, such as a Gaussian.

Then,

where e = (x − x )/ x − x 2 .

The factor E(|w e|) in front of the norm is just a function of the radial part of the distribution P , which we should set inversely proportional to

The sparsest networks possible have d = 1, leading to first-order additive kernels.

Here we look at two simple nonlinearities where we can perform the sum and obtain an explicit formula for the additive kernel.

In both cases, the kernels are simply related to a robust distance metric.

This suggests that such kernels may be useful in cases where there are outlier coordinates in the input data.

Step function nonlinearity We again consider the step function nonlinearity h(·) = Θ(·), which in the case of fully-connected Gaussian weights leads to the degree p = 0 arc-cosine kernel k(x,

For a scalar a, normalization leads to a/ a = sgn(a).

Therefore, θ = arccos (sgn(x i ) sgn(x i )) = 0 if sgn(x i ) = sgn(x i ) and π otherwise.

Performing the sum in (3), we find that the kernel becomes

This kernel is equal to one minus the normalized Hamming distance of vectors sgn(x) and sgn(x ).

The fully-connected kernel, on the other hand, uses the full angle between the vectors x and x .

The sparsity can be seen as inducing a "quantization," via the sign function, on these vectors.

Finally, if the data are in the binary hypercube, with x and x ∈ {−1, +1} l , then the kernel is exactly one minus the normalized Hamming distance.

Sign nonlinearity We now consider a slightly different nonlinearity, the sign function.

It will turn out that the kernel is quite different than for the step function.

This has h(·) = sgn(·) = 2Θ(·) − 1.

Choosing P (w) = 1 2 δ(w + 1) + 1 2 δ(w − 1) and a 2 = −a 1 = a recovers the "random stump" result of Rahimi and Recht [2] .

Despite the fact that sign is just a shifted version of the step function, the kernels are quite different: the sign nonlinearity does not exhibit the quantization effect and depends on the 1 -norm rather than the 0 -"norm".

We now show a basic uniform convergence result for any random features, not necessarily sparse, that use Lipschitz continuous nonlinearities.

Recall the definition of a Lipschitz function:

holds for all x, y ∈ X .

Here, · is a norm on X (the 2 -norm unless otherwise specified).

Assuming that h is Lipschitz and some regularity assumptions on the distribution µ, the random feature expansion approximates the kernel uniformly over X .

As far as we are aware, this result has not been stated previously, although it appears to be known (see Bach [7] ) and is very similar to Claim 1 in Rahimi and Recht [2] which holds only for random Fourier features (see also Sutherland and Schneider [8] and Sriperumbudur and Szabo [9] for improved results in this case).

The rates we obtain for Lipschitz nonlinearities are not essentially different than those obtained in the Fourier features case.

As for the examples we have given, the only ones which are not Lipschitz are the step function (order 0 arc-cosine kernel) and sign nonlinearities.

Since these functions are discontinuous, their convergence to the kernel occurs in a weaker than uniform sense.

However, our result does apply to the rectified linear nonlinearity (order 1 arc-cosine kernel), which is non-differentiable at zero but 1-Lipschitz and widely applied in artificial neural networks.

The proof of the following Theorem appears at the end of this section.

Assume that x ∈ X ⊂ R l and that X is compact, ∆ = diam(X ), and the null vector 0 ∈ X .

Let the weights and biases (w, b) follow the distribution µ on R l+1 with finite second moments.

Let h(·) be a nonlinearity which is L-Lipschitz continuous and define the random feature φ : R l → R by φ(x) = h(w x − b).

We assume that the following hold for all x ∈ X : |φ(x)| ≤ κ almost surely, E |φ(x)| 2 < ∞, and

≤ with probability at least

Sample complexity Theorem 1 guarantees uniform approximation up to error using m = O features.

This is precisely the same dependence on l and as for random Fourier features.

Note that [10] also found that m should scale linearly with l to minimize error in a particular classification task.

A limitation of Theorem 1 is that it only shows approximation of the limiting kernel rather than direct approximation of functions in the RKHS.

A more detailed analysis of the convergence to RKHS is contained in the work of Bach [7] , whereas Rudi and Rosasco [11] directly analyze the generalization ability of these approximations.

Sun et al. [12] show even faster rates which also apply to SVMs, assuming that the features are compatible ("optimized") for the learning problem.

Also, the techniques of Sutherland and Schneider [8] and Sriperumbudur and Szabo [9] could be used to improve our constants and prove convergence in other L p norms.

In the sparse case, we must extend our probability space to capture the randomness of (1) the degrees, (2) the neighborhoods conditional on the degree, and (3) the weight vectors conditional on the degree and neighborhood.

The degrees are distributed independently according to d i ∼ D, with some abuse of notation since we also use D(d) to represent the probability mass function.

We shall always think of the neighborhoods N ∼ ν|d as chosen uniformly among all d element subsets, where ν|d represents this conditional distribution.

Finally, given a neighborhood of some degree, the nonzero weights and bias are drawn from a distribution (w, b) ∼ µ|d on R d+1 .

For simpler notation, we do not show any dependence on the neighborhood here, since we will always take the actual weight values to not depend on the particular neighborhood N .

However, strictly speaking, the weights do depend on N because that determines their support.

Finally, we use E to denote expectation over all variables (degree, neighborhood, and weights), whereas we use E µ|d for the expectation under µ|d for a given degree.

Corollary 2 (Kernel approximation with sparse features).

Assume that x ∈ X ⊂ R l and that X is compact, ∆ = diam(X ), and the null vector 0 ∈ X .

Let the degrees d follow the degree distribution D on [l].

For every d ∈ [l], let µ|d denote the conditional distributions for (w, b) on R d+1 and assume that these have finite second moments.

Let h(·) be a nonlinearity which is L-Lipschitz continuous, and define the random feature φ : R l → R by φ(x) = h(w x − b), where w follows the degree distribution model.

We assume that the following hold for all x N ∈ X N with |N | = d, and for all 1 ≤ d ≤ l: |φ(x N )| 2 ≤ κ almost surely under µ|d, E |φ(x N )| 2 |d < ∞, and

, with probability at least

The kernels k

are given by equations (3) and (4).

Proof.

It suffices to show that conditions (1-3) on the conditional distributions µ|d, d ∈ [l], imply conditions (1-3) in Theorem 1.

Conditions (1) and (2) clearly hold, since the distribution D has finite support.

By construction,

, which concludes the proof.

Differences of sparsity The only difference we find with sparse random features is in the terms E w 2 and E w , since sparsity adds variance to the weights.

This suggests that scaling the weights so that E µ|d w 2 is constant for all d is a good idea.

For example, setting

With this choice, the number of sparse features needed to achieve an error is the same as in the dense case, up to a small constant factor.

This is perhaps remarkable since there could be as many as 2 l terms in the expression of k dist D (x, x ).

However, the random feature expansion does not need to approximate all of these terms well, just their average.

Proof of Theorem 1.

We follow the approach of Claim 1 in [2] , a similar result for random Fourier features but which crucially uses the fact that the trigonometric functions are differentiable and bounded.

For simplicity of notation, let ξ = (x, x ) and define the direct sum norm on X + = X ⊕ X as ξ + = x + x .

Under this norm X + is a Banach space but not a Hilbert space, however this will not matter.

For i = 1, . . .

, m, let f i (ξ) = φ i (x)φ i (x ),

and note that these g i are i.i.d., centered random variables.

By assumptions (1) and (2), f i and g i are absolutely integrable and k(x, x ) = E φ i (x)φ i (x ).

Denote their mean bȳ

Our goal is to show that |ḡ(ξ)| ≤ for all ξ ∈ X + with sufficiently high probability.

The space X + is compact and 2l-dimensional, and it has diameter at most twice the diameter of X under the sum norm.

Thus we can cover X + with an -net using at most T = (4∆/R) 2l balls of radius R [13] .

Call the centers of these balls ξ i for i = 1, . . .

, T , and letL denote the Lipschitz constant ofḡ with respect to the sum norm.

Then we can show that |ḡ(ξ)| ≤ for all ξ ∈ X + if we show that we have that f i has Lipschitz constant κL w i .

This implies that g i has Lipschitz constant ≤ κL( w i + E w ).

LetL denote the Lipschitz constant ofḡ.

Note that EL ≤ 2κLE w .

Also,

Markov

Now we would like to show that |ḡ(ξ i )| ≤ /2 for all i = 1, . . .

, T anchors in the -net.

A straightforward application of Hoeffding's inequality and a union bound shows that

since |f i (ξ)| ≤ κ 2 .

Combining equations (11) and (12) results in a probability of failure Pr sup

Set R = (a/b) 1 2l+2 , so that the probability (13) has the form, 2a for all l ∈ N, assuming ∆κL E w 2 + 3(E w ) 2 > .

Considering the complementary event concludes the proof.

@highlight

We advocate for random features as a theory of biological neural networks, focusing on sparsely connected networks