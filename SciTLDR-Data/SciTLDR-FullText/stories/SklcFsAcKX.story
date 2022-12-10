Deep neural networks provide state-of-the-art performance for image denoising, where the goal is to recover a near noise-free image from a noisy image.

The underlying principle is that neural networks trained on large datasets have empirically been shown to be able to generate natural images well from a low-dimensional latent representation of the image.

Given such a generator network, or prior, a noisy image can be denoised by finding the closest image in the range of the prior.

However, there is little theory to justify this success, let alone to predict the denoising performance as a function of the networks parameters.

In this paper we consider the problem of denoising an image from additive Gaussian noise, assuming the image is well described by a deep neural network with ReLu activations functions, mapping a k-dimensional latent space to an n-dimensional image.

We state and analyze a simple gradient-descent-like iterative algorithm that minimizes a non-convex loss function, and provably removes a fraction of (1 - O(k/n)) of the noise energy.

We also demonstrate in numerical experiments that this denoising performance is, indeed, achieved by generative priors learned from data.

We consider the image or signal denoising problem, where the goal is to remove noise from an unknown image or signal.

In more detail, our goal is to obtain an estimate of an image or signal y˚P R n from y " y˚`η, where η is unknown noise, often modeled as a zero-mean white Gaussian random variable with covariance matrix σ 2 {nI.Image denoising relies on modeling or prior assumptions on the image y˚.

For example, suppose that the image y˚lies in a k-dimensional subspace of R n denoted by Y. Then we can estimate the original image by finding the closest point in 2 -distance to the noisy observation y on the subspace Y. The corresponding estimate, denoted byŷ, obeys DISPLAYFORM0 with high probability (throughout, }¨} denotes the 2 -norm).

Thus, the noise energy is reduced by a factor of k{n over the trivial estimateŷ " y which does not use any prior knowledge of the signal.

The denoising rate (1) shows that the more concise the image prior or image representation (i.e., the smaller k), the more noise can be removed.

If on the other hand the prior (the subspace, in this example) does not include the original image y˚, then the error bound (1) increases as we would remove a significant part of the signal along with noise when projecting onto the range of the signal prior.

Thus a concise and accurate prior is crucial for denoising.

Real world signals rarely lie in a priori known subspaces, and the last few decades of image denoising research have developed sophisticated and accurate image models or priors and algorithms.

Examples include models based on sparse representations in overcomplete dictionaries such as wavelets (Donoho, 1995) and curvelets (Starck et al., 2002) , and algorithms based on exploiting self-similarity within images BID4 .

A prominent example of the former class of algorithms is the BM3D BID4 algorithm, which achieves state-of-the-art performance for certain denoising problems.

However, the nuances of real world images are difficult to describe with handcrafted models.

Thus, starting with the paper (Elad & Aharon, 2006 ) that proposes to learn sparse representation based on training data, it has become common to learn concise representation for denoising (and other inverse problems) from a set of training images.

In 2012, Burger et al. BID2 applied deep networks to the denoising problem, by training a deep network on a large set of images.

Since then, deep learning based denoisers (Zhang et al., 2017) have set the standard for denoising.

The success of deep network priors can be attributed to their ability to efficiently represent and learn realistic image priors, for example via autodecoders (Hinton & Salakhutdinov, 2006) and generative adversarial models (Goodfellow et al., 2014) .

Over the last few years, the quality of deep priors has significantly improved (Karras et al., 2017; Ulyanov et al., 2017) .

As this field matures, priors will be developed with even smaller latent code dimensionality and more accurate approximation of natural signal manifolds.

Consequently, the representation error from deep priors will decrease, and thereby enable even more powerful denoisers.

As the influence of deep networks in inverse problems grows, it becomes increasingly important to understand their performance at a theoretical level.

Given that most optimization approaches for deep learning are first order gradient methods, a justification is needed for why they do not get stuck in local minima.

The closest theoretical work to this question is BID1 , which solves a noisy compressive sensing problem with generative priors by minimizing empirical risk.

Under the assumption that the network is Lipschitz, they show that if the global optimizer can be found, which is in principle NP-hard, then a signal estimate is recovered to within the noise level.

While the Lipschitzness assumption is quite mild, the resulting theory does not provide justification for why global optimality can be reached.

The most related work that establishes theoretical reasons for why gradient methods would not get stuck in local minima, when using deep generative priors for solving inverse problems, is Hand & Voroninski (2018) .

In it, the authors establish global favorability for optimization of the noiseless empirical risk function.

Specifically, they show existence of a descent direction outside a ball around the global optimizer and a negative multiple of it in the latent space of the generative model.

This work does not provide a specific algorithm which provably estimates the global minimizer, nor does it provide an analysis of the robustness of the problem with respect to noise.

In this paper, we propose the first algorithm for solving denoising with deep generative priors that provably finds an approximation of the underlying image.

Following the lead of Hand & Voroninski (2018), we assume an expansive Gaussian model for the deep generative network in order to establish this result.

Contributions: The goal of this paper is to analytically quantify the denoising performance of deep-prior based denoisers.

Specifically, we characterize the performance of a simple and efficient algorithm for denoising based on a d-layer generative neural network G : R k Ñ R n , with k ă n, and random weights.

In more detail, we propose a gradient method with a tweak that attempts to minimize the least-squares loss f pxq " 1 2 }Gpxq´y} 2 between the noisy image y and an image in the range of the prior, Gpxq.

While f is non-convex, we show that the gradient method yields an estimatex obeying DISPLAYFORM1 with high probability, where the notation À absorbs a constant factor depending on the number of layers of the network, and its expansitivity, as discussed in more detail later.

Our result shows that the denoising rate of a deep prior based denoiser is determined by the dimension of the latent representation.

We also show in numerical experiments, that this rate-shown to be analytically achieved for random priors-is also experimentally achieved for priors learned from real imaging data.

Loss surface f pxq " }Gpxq´Gpx˚q}, x˚" r1, 0s, of an expansive network G with ReLu activation functions with k " 2 nodes in the input layer and n 2 " 300 and n 3 " 784 nodes in the hidden and output layers, respectively, with random Gaussian weights in each layer.

The surface has a critical point near´x˚, a global minimum at x˚, and a local maximum at 0.

We consider the problem of estimating a vector y˚P R n from a noisy observation y " y˚`η.

We assume that the vector y˚belongs to the range of a d-layer generative neural network G : R k Ñ R n , with k ă n. That is, y˚" Gpx˚q for some x˚P R k .

We consider a generative network of the form DISPLAYFORM0 where relupxq " maxpx, 0q applies entrywise, W i P R niˆni´1 , are the weights in the i-th layer, n i is the number of neurons in the ith layer, and the network is expansive in the sense that k " n 0 ă n 1 ă¨¨¨ă n d " n. The problem at hand is: Given the weights of the network W 1 . . .

W d and a noisy observation y, obtain an estimateŷ of the original image y˚such that }ŷ´y˚} is small andŷ is in the range of G.

As a way to solve the above problem, we first obtain an estimate of x˚, denoted byx, and then estimate y˚as Gpxq.

In order to estimate x˚, we minimize the empirical risk objective DISPLAYFORM0 Since this objective is nonconvex, there is no a priori guarantee of efficiently finding the global minimum.

Approaches such as gradient methods could in principle get stuck in local minima, instead of finding a global minimizer that is close to x˚.However, as we show in this paper, under appropriate conditions, a gradient method with a tweakintroduced next-finds a point that is very close to the original latent parameter x˚, with the distance to the parameter x˚controlled by the noise.

In order to state the algorithm, we first introduce a useful quantity.

For analyzing which rows of a matrix W are active when computing relupW xq, we let DISPLAYFORM1 For a fixed weight matrix W , the matrix W`, x zeros out the rows of W that do not have a positive dot product with x. Alternatively put, W`, x contains weights from only the neurons that are active for the input x. We also define W 1,`,x " pW 1 q`, x " diagpW 1 x ą 0qW 1 and DISPLAYFORM2 The matrix W i,`,x consists only of the weights of the neurons in the ith layer that are active if the input to the first layer is x.

We are now ready to state our algorithm: a gradient method with a tweak informed by the loss surface of the function to be minimized.

Given a noisy observation y, the algorithm starts with an arbitrary initial point x 0 ‰ 0.

At each iteration i " 0, 1, . . ., the algorithm computes the step directionṽ DISPLAYFORM3 which is equal to the gradient of f if f is differentiable at x i .

It then takes a small step opposite tõ v xi .

The tweak is that before each iteration, the algorithm checks whether f p´x i q is smaller than f px i q, and if so, negates the sign of the current iterate x i .This tweak is informed by the loss surface.

To understand this step, it is instructive to examine the loss surface for the noiseless case in FIG0 .

It can be seen that while the loss function has a global minimum at x˚, it is relatively flat close to´x˚.

In expectation, there is a critical point that is a negative multiple of x˚with the property that the curvature in the˘x˚direction is positive, and the curvature in the orthogonal directions is zero.

Further, around approximately´x˚, the loss function is larger than around the optimum x˚. As a simple gradient descent method (without the tweak) could potentially get stuck in this region, the negation check provides a way to avoid converging to this region.

Our algorithm is formally summarized as Algorithm 1 below.

Require: Weights of the network W i , noisy observation y, and step size α ą 0 1: Choose an arbitrary initial point DISPLAYFORM0 if f p´x i q ă f px i q then 4: DISPLAYFORM1 7: DISPLAYFORM2 Other variations of the tweak are also possible.

For example, the negation check in Step 2 could be performed after a convergence criterion is satisfied, and if a lower objective is achieved by negating the latent code, then the gradient descent can be continued again until a convergence criterion is again satisfied.

For our analysis, we consider a fully-connected generative network G : R k Ñ R n with Gaussian weights and no bias terms.

Specifically, we assume that the weights W i are independently and identically distributed as N p0, 2{n i q, but do not require them to be independent across layers.

Moreover, we assume that the network is sufficiently expansive: Expansivity condition.

We say that the expansivity condition with constant ą 0 holds if DISPLAYFORM0 where c is a particular numerical constant.

In a real-world generative network the weights are learned from training data, and are not drawn from a Gaussian distribution.

Nonetheless, the motivation for selecting Gaussian weights for our analysis is as follows:1.

The empirical distribution of weights from deep neural networks often have statistics consistent with Gaussians.

AlexNet is a concrete example BID0 .2.

The field of theoretical analysis of recovery guarantees for deep learning is nascent, and Gaussian networks can permit theoretical results because of well developed theories for random matrices.3.

It is not clear which non-Gaussian distribution for weights is superior from the joint perspective of realism and analytical tractability.4.

Truly random nets, such as in the Deep Image Prior (Ulyanov et al., 2017) , are increasingly becoming of practical relevance.

Thus, theoretical advances on random nets is of independent interest.

We are now ready to state our main result.

Theorem 1.

Consider a network with the weights in the i-th layer, W i P R niˆni´1 , i.i.d.

N p0, 2{n i q distributed, and suppose that the network satisfies the expansivity condition for some ď K{d 90 .

Also, suppose that the noise variance obeys DISPLAYFORM1 Consider the iterates of Algorithm 1 with stepsize α " K 4 1 d 2 .

Then, there exists a number of steps N upper bounded by DISPLAYFORM2 f px 0 q }x˚} such that after N steps, the iterates of Algorithm 1 obey DISPLAYFORM3 with probability at least 1´2e´2 DISPLAYFORM4 .

are numerical constants, and x 0 is the initial point in the optimization.

The error term in the bound (2) consists of two terms-the first is controlled by , and the second depends on the noise.

The first term is negligible if is chosen sufficiently small, but that comes at the expense of the expansivity condition being more stringent.

The second term in the bound FORMULA13 is more interesting and controls the effect of noise.

Specifically, for sufficiently small, our result guarantees that after sufficiently many iterations, DISPLAYFORM5 where the notation À absorbs a factor logarithmic in n and polynomial in d. One can show that G is Lipschitz in a region around x˚1, DISPLAYFORM6 Thus, the theorem guarantees that our algorithm yields the denoising rate of σ 2 k{n, and, as a consequence, denoising based on a generative deep prior provably reduces the energy of the noise in the original image by a factor of k{n.

We note that the intention of this paper is to show rate-optimality of recovery with respect to the noise power, the latent code dimensionality, and the signal dimensionality.

As a result, no attempt was made to establish optimal bounds with respect to the scaling of constants or to powers of d. The bounds provided in the theorem are highly conservative in the constants and dependency on the number of layers, d, in order to keep the proof as simple as possible.

Numerical experiments shown later reveal that the parameter range for successful denoising are much broader than the constants suggest.

As this result is the first of its kind for rigorous analysis of denoising performance by deep generative networks, we anticipate the results can be improved in future research, as has happened for other problems, such as sparsity-based compressed sensing and phase retrieval.

To prove our main result, we make use of a deterministic condition on G, called the Weight Distribution Condition (WDC), and then show that Gaussian W i , as given by the statement of Theorem 1 are such that W i { ?

2 satisfies the WDC with the appropriate probability for all i, provided the expansivity condition holds.

Our main result, Theorem 1, continues to hold for any weight matrices such that W i { ?

2 satisfy the WDC.The condition is on the spatial arrangement of the network weights within each layer.

We say that the matrix W P R nˆk satisfies the Weight Distribution Condition with constant if for all nonzero DISPLAYFORM0 where w i P R k is the ith row of W ; Mx Øŷ P R kˆk is the matrix 2 such thatx Þ Ñŷ,ŷ Þ Ñx, and z Þ Ñ 0 for all z P spanptx, yuq K ;x " x{}x} 2 andŷ " y{}y} 2 ; θ 0 " =px, yq; and 1 S is the indicator function on S. The norm in the left hand side of FORMULA17 is the spectral norm.

Note that an elementary calculation 3 gives that Q x,y " Er ř n i"1 1 wi,x ą0 1 wi,y ą0¨wi w t i s for w i " N p0, I k {nq.

As the rows w i correspond to the neural network weights of the ith neuron in a layer given by W , the WDC provides a deterministic property under which the set of neuron weights within the layer given by W are distributed approximately like a Gaussian.

The WDC could also be interpreted as a deterministic property under which the neuron weights are distributed approximately like a uniform random variable on a sphere of a particular radius.

Note that if x " y, Q x,y is an isometry up to a factor of 1{2.

In this section we briefly discuss another important scenario to which our results apply to, namely regularizing inverse problems using deep generative priors.

Approaches that regularize inverse problems using deep generative models BID1 have empirically been shown to improve over sparsity-based approaches, see (Lucas et al., 2018) for a review for applications in imaging, and (Mardani et al., 2017) for an application in Magnetic Resonance Imaging showing a significant performance improvement over conventional methods.

Consider an inverse problem, where the goal is to reconstruct an unknown vector y˚P R n from m ă n noisy linear measurements: DISPLAYFORM0 where A P R mˆn is called the measurement matrix and η is zero mean Gaussian noise with covariance matrix σ 2 {nI, as before.

As before, assume that y˚lies in the range of a generative prior G, i.e., y˚" Gpx˚q for some x˚. As a way to recover x˚, consider minimizing the empirical risk objective f pxq " 1 2 }AGpxq´z}, using Algorithm 1, with Step 6 substituted bỹ v xi " pAΠ DISPLAYFORM1 t pAGpx i q´yq, to account for the fact that measurements were taken with the matrix A.Suppose that A is a random projection matrix, for concreteness assume that A has i.i.d.

Gaussian entries with variance 1{m.

One could prove an analogous result as Theorem 1, but with ω " DISPLAYFORM2 . . .

n d q, (note that n has been replaced by m).

This extension shows that, provided is chosen sufficiently small, that our algorithm yields an iterate x i obeying DISPLAYFORM3 where again À absorbs factors logarithmic in the n i 's, and polynomial in d. Proving this result would be analogous to the proof of Theorem 1, but with the additional assumption that the sensing matrix A acts like an isometry on the union of the ranges of DISPLAYFORM4 ,xi , analogous to the proof in (Hand & Voroninski, 2018) .

This extension of our result shows that Algorithm 1 enables solving inverse problems under noise efficiently, and quantifies the effect of the noise.2 A formula for Mx Øŷ is as follows.

If θ0 " =px,ŷq P p0, πq and R is a rotation matrix such thatx andŷ map to e1 and cos θ0¨e1`sin θ0¨e2 respectively, then Mx Øŷ " R t¨c os θ0 sin θ0 0 sin θ0´cos θ0 0 0 0 0 k´2‚ R, where 0 k´2 is a k´2ˆk´2 matrix of zeros.

If θ0 " 0 or π, then Mx Øŷ "xx t or´xx t , respectively.

3 To do this calculation, take x " e1 and y " cos θ0¨e1`sin θ0¨e2 without loss of generality.

Then each entry of the matrix can be determined analytically by an integral that factors in polar coordinates.

We hasten to add that the paper BID1 ) also derived an error bound for minimizing empirical loss.

However, the corresponding result (for example Lemma 4.3) differs in two important aspects to our result.

First, the result in BID1 only makes a statement about the minimizer of the empirical loss and does not provide justification that an algorithm can efficiently find a point near the global minimizer.

As the program is non-convex, and as non-convex optimization is NP-hard in general, the empirical loss could have local minima at which algorithms get stuck.

In contrast, the present paper presents a specific practical algorithm and proves that it finds a solution near the global optimizer regardless of initialization.

Second, the result in BID1 considers arbitrary noise η and thus can not assert denoising performance.

In contrast, we consider a random model for the noise, and show the denoising behavior that the resulting error is no more than Opk{nq, as opposed to }η} 2 « Op1q, which is what we would get from direct application of the result in BID1 .

In this section we provide experimental evidence that corroborates our theoretical claims that denoising with deep priors achieves a denoising rate proportional to σ 2 k{n.

We consider both a synthetic, random prior, as studied theoretically in the paper, as well as a prior learned from data.

All our results are reproducible with the code provided in the supplement.

We start with a synthetic generative network prior with ReLu-activation functions, and draw its weights independently from a Gaussian distribution.

We consider a two-layer network with n " 1500 neurons in the output layer, 500 in the middle layer, and vary the number of input neurons, k, and the noise level, σ.

We next present simulations showing that if k is sufficiently small, our algorithm achieves a denoising rate proportional to σk{n as guaranteed by our theory.

Towards this goal, we generate Gaussian inputs x˚to the network and observe the noisy image y " Gpx˚q`η, η " N p0, σ 2 {nIq.

From the noisy image, we first obtain an estimatex of the latent representation by running Algorithm 1 until convergence, and second we obtain an estimate of the image asŷ " Gpxq.

In the left and middle panel of Figure 3 , we depict the normalized mean squared error of the latent representation, MSEpx, x˚q, and the mean squared error in the image domain, MSEpGpxq, Gpx˚qq, where we defined MSEpz, z 1 q " }z´z 1 } 2 .

For the left panel, we fix the noise variance to σ 2 " 0.25, and vary k, and for the middle panel we fix k " 50 and vary the noise variance.

The results show that, if the network is sufficiently expansive, guaranteed by k being sufficiently small, then in the noiseless case (σ 2 " 0), the latent representation and image are perfectly recovered.

In the noisy case, we achieve a MSE proportional to σ 2 k{n, both in the representation and image domains.

We also observed that for the problem instances considered here, the negation trick in step 3-4 of Algorithm 1 is often not necessary, in that even without that step the algorithm typically converges to the global minimum.

Having said this, in general the negation step is necessary, since there exist problem instances that have a local minimum opposite of x˚.

We next consider a prior learned from data.

Technically, for such a prior our theory does not apply since we assume the weights to be chosen at random.

However, the numerical results presented in this section show that even for the learned prior we achieve the rate predicted by our theory pertaining to a random prior.

Towards this goal, we consider a fully-connected autoencoder parameterized by k, consisting of an decoder and encoder with ReLu activation functions and fully connected layers.

We choose the number of neurons in the three layers of the encoder as 784, 400, k, and those of the decoder as k, 400, 784.

We set k " 10 and k " 20 to obtain two different autoencoders.

We train both autoencoders on the MNIST (Lecun et al., 1998) training set.

We then take an image y˚from the MNIST test set, add Gaussian noise to it, and denoise it using our method based on the learned decoder-network G for k " 10 and k " 20.

Specifically, we estimate As suggested by the theory pertaining to decoders with random weights, if k is sufficiently small, and thus the network is sufficiently expansive, the denoising rate is proportional to σ 2 k{n.

Right panel: Denoising of handwritten digits based on a learned decoder with k " 10 and k " 20, along with the least-squares fit as dotted lines.

The learned decoder with k " 20 has more parameters and thus represents the images with a smaller error; therefore the MSE at σ " 0 is smaller.

However, the denoising rate for the decoder with k " 20, which is the slope of the curve is larger as well, as suggested by our theory.

the latent representationx by running Algorithm 1, and then setŷ " Gpxq.

See Figure 2 for a few examples demonstrating the performance of our approach for different noise levels.

We next show that this achieves a mean squared error (MSE) proportional to σ 2 k{n, as suggested by our theory which applies for decoders with random weights.

We add noise to the images with noise variance ranging from σ 2 " 0 to σ 2 " 6.

In the right panel of Figure 3 we show the MSE in the image domain, MSEpGpxq, Gpx˚qq, averaged over a number of images for the learned decoders with k " 10 and k " 20.

We observe an interesting tradeoff: The decoder with k " 10 has fewer parameters, and thus does not represent the digits as well, therefore the MSE is larger than that for k " 20 for the noiseless case (i.e., for σ " 0).

On the other hand, the smaller number of parameters results in a better denoising rate (by about a factor of two), corresponding to the steeper slope of the MSE as a function of the noise variance, σ 2 .

Theorem 2.

Consider a network with the weights in the i-th layer, W i P R niˆni´1 , i.i.d.

N p0, 1{n i q distributed, and suppose that the network satisfies the expansivity condition for some ď K{d 90 .

Also, suppose that the noise variance obeys DISPLAYFORM0 Consider the iterates of Algorithm 1 with stepsize α " K 4 DISPLAYFORM1 Then, there exists a number of steps N upper bounded by DISPLAYFORM2 d }x˚} such that after N steps, the iterates of Algorithm 1 obey DISPLAYFORM3 with probability at least 1´2e´2 DISPLAYFORM4 .

are numerical constants, and x 0 is the initial point in the optimization.

As mentioned in Section 4.1, our proof makes use of a deterministic condition, called the Weight Distribution Condition (WDC), formally defined in Section 4.1.

The following proposition establishes that the expansivity condition ensures that the WDC holds: Lemma 3 (Lemma 9 in (Hand & Voroninski, 2018) ).

Fix P p0, 1q.

If the entires of W i P R niˆni´1 are i.i.d.

N p0, 1{n i q and the expansivity condition n i ą c ´2 logp1{ qn i´1 log n i´1 holds, then W i satisfies the WDC with constant with probability at least 1´8n i e´K 2 ni´1 .

Here, c and K are numerical constants.

We note that the form of dependence of n i on can be read off the proofs of Lemma 10 in (Hand & Voroninski, 2018) .

It follows from Lemma 3, that the WDC holds for all W i with probability at least 1´ř DISPLAYFORM5 In the remainder of the proof we work on the event that the WDC holds for all W i .

Recall that the goal of our algorithm is to minimize the empirical risk objective DISPLAYFORM0 where y :" Gpx˚q`η, with η " N p0, σ 2 {nIq.

Our results rely on the fact that outside of two balls around x " x˚and x "´ρ d x˚, with ρ d a constant defined below, the direction chosen by the algorithm is a descent direction, with high probability.

Towards this goal, we use a concentration argument, similar to the arguments used in (Hand & Voroninski, 2018) .

First, define Λ x :" Π 1 i"d W i,`,x (with W i,`,x defined in Section 3) for notational convenience, and note that the step direction of our algorithm can be written as DISPLAYFORM1 Note that at points x where G (and hence f ) is differentiable, we have thatṽ x " ∇f pxq.

The proof is based on showing thatṽ x concentrates around a particular h x P R k , defined below, that is a continuous function of nonzero x, x˚and is zero only at x " x˚and x "´ρ d x˚. The definition of h x depends on a function that is helpful for controlling how the operator x Þ Ñ W`, x x distorts angles, defined as: DISPLAYFORM2 π´θq cos θ`sin θ π¯.With this notation, we define DISPLAYFORM3 where θ 0 " =px, x˚q and θ i " gpθ i´1 q. Note that h x is deterministic and only depends on x, x˚, and the number of layers, d.

In order to bound the deviation ofṽ x from h x we use the following two lemmas, bounding the deviation controlled by the WDC and the deviation from the noise: Lemma 4 (Lemma 6 in (Hand & Voroninski, 2018) ).

Suppose that the WDC holds with ă 1{p16πd 2 q 2 .

Then, for all nonzero x, x˚P R k , DISPLAYFORM4 @ pΠ DISPLAYFORM5 DISPLAYFORM6 Proof.

Equation FORMULA34 and FORMULA35 are Lemma 6 in (Hand & Voroninski, 2018) .

Regarding FORMULA36 , note that the WDC implies that }W i,`,x } 2 ď 1{2` .

It follows that DISPLAYFORM7 where the last inequalities follow by our assumption on .Lemma 5.

Suppose the WDC holds with ă 1{p16πd 2 q 2 , that any subset of n i´1 rows of W i are linearly independent for each i, and that η " N p0, σ 2 {nIq.

Then the event DISPLAYFORM8 , ω :" DISPLAYFORM9 holds with probability at least 1´2e´2 k log n .As the cost function f is not differentiable everywhere, we will make use of the generalized subdifferential in order to reference the subgradients at nondifferentiable points.

For a Lipschitz functionf defined from a Hilbert space X to R, the Clarke generalized directional derivative off at the point x P X in the direction u, denoted byf o px; uq, is defined byf o px; uq " lim sup yÑx,tÓ0fpy`tuq´f pyq t , and the generalized subdifferential off at x, denoted by Bf pxq, is defined byBf pxq " tv P R k | xv, uy ďf o px; uq, for all u P X u.

Since f pxq is a piecewise quadratic function, we have DISPLAYFORM10 where conv denotes the convex hull of the vectors v 1 , . . .

, v t , t is the number of quadratic functions adjoint to x, and v i is the gradient of the i-th quadratic function at x. Lemma 6.

Under the assumption of Lemma 5, and assuming that E noise holds, we have that, for any x ‰ 0 and any v x P Bf pxq, DISPLAYFORM11 .

In particular, this holds for the subgradient v x "ṽ x .Proof.

By (11), Bf pxq " convpv 1 , . . .

v t q for some finite t, and thus v x " a 1 v 1`. . .

a t v t for some a 1 , . . .

, a t ě 0, ř i a i " 1.

For each v i , there exists a w such that v i " lim tÓ0ṽx`tw .

On the event E noise , we have that for any x ‰ 0, for anyṽ x P Bf pxq DISPLAYFORM12 , where the last inequality follows from Lemmas 4 and 5 above.

The proof is concluded by appealing to the continuity of h x with respect to nonzero x, and by noting that DISPLAYFORM13 where we used the inequality above and that ř i a i " 1.We will also need an upper bound on the norm of the step direction of our algorithm: Lemma 7.

Suppose that the WDC holds with ă 1{p16πd 2 q 2 and that the event E noise holds with DISPLAYFORM14 .

Then, for all x, DISPLAYFORM15 where K is a numerical constant.

Proof.

Define for convenience ζ j " DISPLAYFORM16 .

We have DISPLAYFORM17 ď dK 2 d maxp}x}, }x˚}q, where the second inequality follows from the definition of h x and Lemma 6, the third inequality uses |ζ j | ď 1, and the last inequality uses the assumption ω ď 2´d {2 }x˚} 8π .

We are now ready to prove Theorem 2.

The logic of the proof is illustrated in Figure 4 .

Recall that x i is the ith iterate of x as per Algorithm 1.

We first ensure that we can assume throughout that x i is bounded away from zero: Lemma 8.

Suppose that WDC holds with ă 1{p16πd 2 q 2 and that E noise holds with ω in (10) DISPLAYFORM0 .

Moreover, suppose that the step size in Algorithm 1 satisfies 0 ă α ă DISPLAYFORM1 In particular, if α " K2 d {d 2 , then N is bounded by a constant times d 4 .We can therefore assume throughout this proof that x i R Bp0, K 0 }x˚}q, K 0 " 1 32π .

We prove Theorem 2 by showing that if }h x } is sufficiently large, i.e., if the iterate x i is outside of set DISPLAYFORM2 Sβ Sβ Figure 4 : Logic of the proof: Starting at an arbitrary point, Algorithm 1 moves away from 0, at least till its iterates are outside the gray ring, as 0 is a local maximum; and once an iterate x i leaves the gray ring around 0, all subsequent iterates will never be in the white circle around 0 again (see Lemma 8).

Then the algorithm might move towards´ρ d x˚, but once it enters the dashed ball around´ρ d x˚, it enters a region where the function value is strictly larger than that of the dashed ball around x˚, by Lemma 10.

Thus steps 3-5 of the algorithm will ensure that the next iterate x i is in the dashed ball around x˚. From there, the iterates will move into the region Sβ , since outside of Sβ Y Sβ the algorithm chooses a descent direction in each step (see the argument around equation FORMULA0 ).

The region Sβ is covered by a ball of radius r, by Lemma 9, determined by the noise and .

DISPLAYFORM3 then the algorithm makes progress in the sense that f px i`1 q´f px i q is smaller than a certain negative value.

The set S β is contained in two balls around x˚and´ρx˚, whose radius is controlled by β:Lemma 9.

For any β ď 1 64 2 d 12 , DISPLAYFORM4 Here, ρ d ą 0 is defined in the proof and obeys ρ d Ñ 1 as d Ñ 8.Note that by the assumption ω ď DISPLAYFORM5 and Kd

?

ď 1, our choice of β in (13) obeys β ď 1 64 2 d 12 for sufficiently small K 1 , K, and thus Lemma 9 yields: DISPLAYFORM0 were we define the radius r " DISPLAYFORM1 d{2 , where K 2 , K 3 are numerical constants.

Note that hat the radius r is equal to the right hand side in the error bound (4) in our theorem.

In order to guarantee that the algorithm converges to a ball around x˚, and not to that around´ρ d x˚, we use the following lemma:Lemma 10.

Suppose that the WDC holds with ă 1{p16πd 2 q 2 .

Moreover suppose that E noise holds, and that ω in the event E noise obeys ω 2´d {2 }x˚}2 ď K 9 {d 2 , where K 9 ă 1 is a universal constant.

Then for any φ d P rρ d , 1s, it holds that DISPLAYFORM2 for all x P Bpφ d x˚, K 3 d´1 0 }x˚}q and y P Bp´φ d x˚, K 3 d´1 0 }x˚}q, where K 3 ă 1 is a universal constant.

In order to apply Lemma 10, define for convenience the two sets:Sβ :"S β X Bpx˚, rq, and DISPLAYFORM3 By the assumption that Kd

?

ď 1 and ω ď K 1 d´1 6 2´d {2 }x˚}, we have that for sufficiently small K 1 , K, DISPLAYFORM0 Thus, the assumptions of Lemma 10 are met, and the lemma implies that for any x P Sβ and y P Sβ , it holds that f pxq ą f pyq.

We now show that the algorithm converges to a point in Sβ .

This fact and the negation step in our algorithm (line 3-5) establish that the algorithm converges to a point in Sβ if we prove that the objective is nonincreasing with iteration number, which will form the remainder of this proof.

Consider i such that x i R S β .

By the mean value theorem (Clason, 2017, Theorem 8.13) , there is a t P r0, 1s such that forx i " x i´t αṽ xi there is a vx i P Bf px i q, where Bf is the generalized subdifferential of f , obeying DISPLAYFORM1 In the next subsection, we guarantee that for any t P r0, 1s, vx i withx i " x i´t αṽ xi is close toṽ xi : DISPLAYFORM2 Applying FORMULA0 to FORMULA0 yields DISPLAYFORM3 where we used that αK 7 DISPLAYFORM4 12 , by our assumption on the stepsize α being sufficiently small.

Thus, the maximum number of iterations for which x i R S β is f px 0 q12{pα min i }ṽ xi } 2 q. We next lower-bound }ṽ xi }.

We have that on E noise , for all x R S β , with β given by (13).

DISPLAYFORM5 where the second inequality follows by the definition of S β and Lemma 6, and the third inequality follows from our definition of β in (13).

Thus, In order to conclude our proof, we remark that once x i is inside a ball of radius r around x˚, the iterates do not leave a ball of radius 2r around x˚. To see this, note that by (12) and our choice of stepsize, DISPLAYFORM6 This concludes our proof.

The remainder of the proof is devoted to prove the lemmas used in this section.

We next prove the implication (22).

Consider x i R Bp0, 2K 0 }x˚}q, and note that DISPLAYFORM7 where the second inequality follows from (12), the third inequality from }x i } ě 2K 0 }x˚}, and finally the last inequality from our assumption on the stepsize α.

This concludes the proof of (22).Proof of (21): It remains to prove (21).

We start with proving x,ṽ x ă 0.

For brevity of notation, let Λ z " DISPLAYFORM8 The first inequality follows from FORMULA35 and FORMULA36 , and the second inequality follows from our assumption on ω.

Therefore, for any x P Bp0, 1 16π }x˚}q, x,ṽ x ă 0, as desired.

We next show that, for any x P Bp0, DISPLAYFORM9 where the second inequality is from FORMULA35 and FORMULA36 .

This concludes the proof of (21).A.5 PROOF OF LEMMA 5 DISPLAYFORM10 where P Λx is a projector onto the span of Λ x .

As a consequence, }P Λx η} 2 is χ 2 -distributed random variable with k-degrees of freedom scaled by σ{n.

A standard tail bound (see (?, p. 43)) yields that, for any β ě k, DISPLAYFORM11 Next, we note that by applying Lemmas 13-14 from (Hand & Voroninski, 2018, Proof of Lem.

15)) 4 , with probability one, that the number of different matrices Λ x can be bounded as DISPLAYFORM12 where the second inequality holds for logp10q ď k{4 logpn 1 q. To see this, note that pn DISPLAYFORM13 is implied by kpd logpn 1 q`pd´1q logpn 2 q`. . .

logpn d qq ě kd 2 {4 logpn 1 q ě d 2 logp10q.

Thus, by the union bound, DISPLAYFORM14 where n " n d .

Recall from (9) that }Λ x } ď 13 12 .

Combining this inequality with }q x } 2 ď }Λ x } 2 }P Λx η} 2 concludes the proof.

A.6 PROOF OF LEMMA 9We now show that h x is away from zero outside of a neighborhood of x˚and´ρ d x˚. We prove Lemma 9 by establishing the following: Lemma 12.

Suppose 64d 6 ?

β ď 1.

Define DISPLAYFORM15 where q θ 0 " π and q DISPLAYFORM16 DISPLAYFORM17 Additionally, DISPLAYFORM18 Proof.

Without loss of generality, let }x˚} " 1, x˚" e 1 andx " r cos θ 0¨e1`r sin θ 0¨e2 for θ 0 P r0, πs.

Let x P S β .First we introduce some notation for convenience.

Let DISPLAYFORM19 π´θ j π , r " }x} 2 , M " maxpr, 1q.

DISPLAYFORM20 By inspecting the components of h x , we have that x P S β implies |´ξ`cos θ 0 pr´ζq| ď βM (24) DISPLAYFORM21 Now, we record several properties.

We have:θ i P r0, π{2s for i ě 1 DISPLAYFORM22 DISPLAYFORM23 θ 0 " π`O 1 pδq ñ θ i " q θ i`O1 piδq (31) θ 0 " π`O 1 pδq ñ |ξ| ď δ π (32) DISPLAYFORM24 We now establish (28).

Observe 0 ă gpθq ď`1 3π`1 θ˘´1 ":gpθq for θ P p0, πs.

As g andg are monotonic increasing, we have q θ i " g˝ip q θ 0 q " g˝ipπq ďg˝ipπq "`i 3π`1 π˘´1 " 3π i`3 .

Similarly, gpθq ě p 1 π`1 θ q´1 implies that q θ i ě π i`1 , establishing (29).We now establish (30).

Using (28) and θ i ď q θ i , we have DISPLAYFORM25 where the last inequality can be established by showing that the ratio of consecutive terms with respect to d is greater for the product in the middle expression than for d´3.We establish (31) by using the fact that |g 1 pθq| ď 1 for all θ P r0, πs and using the same logic as for (Hand & Voroninski, 2018, Eq. 17) .We now establish (33).

As θ 0 " π`O 1 pδq, we have θ i " q θ i`O1 piδq.

Thus (33) holds.

Next, we establish that x P S β ñ r ď 4d, and thus M ď 4d.

ď 4d.

Thus, we have x P S β ñ r ď 4d ñ M ď 4d.

Next, we establish that we only need to consider the small angle case (θ 0 « 0) and the large angle case (θ 0 « π), by considering the following three cases:(Case I) sin θ 0 ď 16d 4 β: We have θ 0 " O 1 p32d 4 βq or θ 0 " π`O 1 p32d 4 βq, as 32d 4 β ă 1. .

Using this inequality in (24), we have |ξ| ď βM`β ?

βq, as θ 0 ě π{2 and as β ă 1.At least one of the Cases I,II, or III hold.

Thus, we see that it suffices to consider the small angle case θ 0 " O 1 p32d 4 βq or the large angle case θ 0 " π`O 1 p8πd 4 ? βq.

Small Angle Case.

Assume θ 0 " O 1 pδq with δ " 32d 4 β.

As θ i ď θ 0 ď δ for all i, we have 1 ě ξ ě p1´δ π q d " 1`O 1 p 2δd π q provided δd{π ď 1{2 (which holds by our choice δ " 32d 4 β by A.8 PROOF OF LEMMA 13It holds that }x´y} ě 2 sinpθ x,y {2q minp}x}, }y}q, @x, y (46) sinpθ{2q ě θ{4, @θ P r0, πs (47) d dθ gpθq P r0, 1s @θ P r0, πslogp1`xq ď x @x P r´0.5, 1s (49) logp1´xq ě´2x @x P r0, 0.75s (50) where θ x,y " =px, yq.

We recall the results (36), (37), and FORMULA30

@highlight

By analyzing an algorithms minimizing a non-convex loss, we show that all but a small fraction of noise can be removed from an image using a deep neural network based generative prior.