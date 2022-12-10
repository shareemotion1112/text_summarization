Nonlinearity is crucial to the performance of a deep (neural) network (DN).

To date there has been little progress understanding the menagerie of available  nonlinearities, but recently progress has been made on understanding the r\^{o}le played by piecewise affine and convex nonlinearities like the ReLU and absolute value activation functions and max-pooling.

In particular, DN layers constructed from these operations can be interpreted as {\em max-affine spline operators} (MASOs) that have an elegant link to vector quantization (VQ) and $K$-means.

While this is good theoretical progress, the entire MASO approach is predicated on the requirement that the nonlinearities be piecewise affine and convex, which precludes important activation functions like the sigmoid, hyperbolic tangent, and softmax.

{\em This paper extends the MASO framework to these and an infinitely large class of new nonlinearities by linking deterministic MASOs with probabilistic Gaussian Mixture Models (GMMs).}

We show that, under a GMM, piecewise affine, convex nonlinearities like ReLU, absolute value, and max-pooling can be interpreted as solutions to certain natural ``hard'' VQ inference problems, while sigmoid, hyperbolic tangent, and softmax can be interpreted as solutions to corresponding ``soft'' VQ inference problems.

We further extend the framework by hybridizing the hard and soft VQ optimizations to create a $\beta$-VQ inference that interpolates between hard, soft, and linear VQ inference.

A prime example of a $\beta$-VQ DN nonlinearity is the {\em swish} nonlinearity, which offers state-of-the-art performance in a range of computer vision tasks but was developed ad hoc by experimentation.

Finally, we validate with experiments an important assertion of our theory, namely that DN performance can be significantly improved by enforcing orthogonality in its linear filters.

Deep (neural) networks (DNs) have recently come to the fore in a wide range of machine learning tasks, from regression to classification and beyond.

A DN is typically constructed by composing a large number of linear/affine transformations interspersed with up/down-sampling operations and simple scalar nonlinearities such as the ReLU, absolute value, sigmoid, hyperbolic tangent, etc.

BID13 .

Scalar nonlinearities are crucial to a DN's performance.

Indeed, without nonlinearity, the entire network would collapse to a simple affine transformation.

But to date there has been little progress understanding and unifying the menagerie of nonlinearities, with few reasons to choose one over another other than intuition or experimentation.

Recently, progress has been made on understanding the rôle played by piecewise affine and convex nonlinearities like the ReLU, leaky ReLU, and absolute value activations and downsampling operations like max-, average-, and channel-pooling BID1 .

In particular, these operations can be interpreted as max-affine spline operators (MASOs) BID16 ; BID14 that enable a DN to find a locally optimized piecewise affine approximation to the prediction operator given training data.

A spline-based prediction is made in two steps.

First, given an input signal x, we determine which region of the spline's partition of the domain (the input signal space) it falls into.

Second, we apply to x the fixed (in this case affine) function that is assigned to that partition region to obtain the prediction y = f (x).The key result of BID1 is any DN layer constructed from a combination of linear and piecewise affine and convex is a MASO, and hence the entire DN is merely a composition of MASOs.

MASOs have the attractive property that their partition of the signal space (the collection of multidimensional "knots") is completely determined by their affine parameters (slopes and offsets).

This provides an elegant link to vector quantization (VQ) and K-means clustering.

That is, during learning, a DN implicitly constructs a hierarchical VQ of the training data that is then used for splinebased prediction.

This is good progress for DNs based on ReLU, absolute value, and max-pooling, but what about DNs based on classical, high-performing nonlinearities that are neither piecewise affine nor convex like the sigmoid, hyperbolic tangent, and softmax or fresh nonlinearities like the swish BID20 that has been shown to outperform others on a range of tasks?Contributions.

In this paper, we address this gap in the DN theory by developing a new framework that unifies a wide range of DN nonlinearities and inspires and supports the development of new ones.

The key idea is to leverage the yinyang relationship between deterministic VQ/K-means and probabilistic Gaussian Mixture Models (GMMs) BID3 .

Under a GMM, piecewise affine, convex nonlinearities like ReLU and absolute value can be interpreted as solutions to certain natural hard inference problems, while sigmoid and hyperbolic tangent can be interpreted as solutions to corresponding soft inference problems.

We summarize our primary contributions as follows:Contribution 1: We leverage the well-understood relationship between VQ, K-means, and GMMs to propose the Soft MASO (SMASO) model, a probabilistic GMM that extends the concept of a deterministic MASO DN layer.

Under the SMASO model, hard maximum a posteriori (MAP) inference of the VQ parameters corresponds to conventional deterministic MASO DN operations that involve piecewise affine and convex functions, such as fully connected and convolution matrix multiplication; ReLU, leaky-ReLU, and absolute value activation; and max-, average-, and channelpooling.

These operations assign the layer's input signal (feature map) to the VQ partition region corresponding to the closest centroid in terms of the Euclidean distance, Contribution 2: A hard VQ inference contains no information regarding the confidence of the VQ region selection, which is related to the distance from the input signal to the region boundary.

In response, we develop a method for soft MAP inference of the VQ parameters based on the probability that the layer input belongs to a given VQ region.

Switching from hard to soft VQ inference recovers several classical and powerful nonlinearities and provides an avenue to derive completely new ones.

We illustrate by showing that the soft versions of ReLU and max-pooling are the sigmoid gated linear unit and softmax pooling, respectively.

We also find a home for the sigmoid, hyperbolic tangent, and softmax in the framework as a new kind of DN layer where the MASO output is the VQ probability.

Contribution 3: We generalize hard and soft VQ to what we call β-VQ inference, where β ∈ (0, 1) is a free and learnable parameter.

This parameter interpolates the VQ from linear (β → 0), to probabilistic SMASO (β = 0.5), to deterministic MASO (β → 1).

We show that the β-VQ version of the hard ReLU activation is the swish nonlinearity, which offers state-of-the-art performance in a range of computer vision tasks but was developed ad hoc through experimentation BID20 .Contribution 4: Seen through the MASO lens, current DNs solve a simplistic per-unit (per-neuron), independent VQ optimization problem at each layer.

In response, we extend the SMASO GMM to a factorial GMM that that supports jointly optimal VQ across all units in a layer.

Since the factorial aspect of the new model would make naïve VQ inference exponentially computationally complex, we develop a simple sufficient condition under which a we can achieve efficient, tractable, jointly optimal VQ inference.

The condition is that the linear "filters" feeding into any nonlinearity should be orthogonal.

We propose two simple strategies to learn approximately and truly orthogonal weights and show on three different datasets that both offer significant improvements in classification per-formance.

Since orthogonalization can be applied to an arbitrary DN, this result and our theoretical understanding are of independent interest.

This paper is organized as follows.

After reviewing the theory of MASOs and VQ for DNs in Section 2, we formulate the GMM-based extension to SMASOs in Section 3.

Section 4 develops the hybrid β-VQ inference with a special case study on the swish nonlinearity.

Section 5 extends the SMASO to a factorial GMM and shows the power of DN orthogonalization.

We wrap up in Section 6 with directions for future research.

Proofs of the various results appear in several appendices in the Supplementary Material.

We first briefly review max-affine spline operators (MASOs) in the context of understanding the inner workings of DNs BID1 BID14 , with each spline formed from R piecewise affine and convex mappings.

The MASO parameters consist of the "slopes" A ∈ R K×R×D and the "offsets/biases" B ∈ R K×R .

See Appendix A for the precise definition.

Given the input x ∈ R D and parameters A, B, a MASO produces the output z ∈ R K via DISPLAYFORM0 where [z] k denotes the k th dimension of z. The three subscripts of the slopes tensor [A] k,r,d correspond to output k, partition region r, and input signal index d. The two subscripts of the offsets/biases tensor [B] k,r correspond to output k and partition region r.

An important consequence of FORMULA0 is that a MASO is completely determined by its slope and offset parameters without needing to specify the partition of the input space (the "knots" when D = 1).

Indeed, solving (1) automatically computes an optimized partition of the input space R D that is equivalent to a vector quantization (VQ) BID19 ; BID9 .

We can make the VQ aspect explicit by rewriting (1) in terms of the Hard-VQ (HVQ) matrix T H ∈ R K×R .

that contains K stacked one-hot row vectors, each with the one-hot position at index [t] k ∈ {1, . . .

, R} corresponding to the arg max over r = 1, . . .

, R of (1).

Given the HVQ matrix, (or equivalently, a region of the input space), the input-output mapping is affine and fully determined by DISPLAYFORM1 We retrieve (1) from (2) by noting that DISPLAYFORM2 The key background result for this paper is that the layers of a very large class of DN are MASOs.

Hence, such a DN is a composition of MASOs, where each layer MASO has as input the feature map DISPLAYFORM3 and produces DISPLAYFORM4 , with corresponding to the layer.

Each MASO has thus specific parameters A ( ) , B ( ) .Theorem 1.

Any DN layer comprising a linear operator (e.g., fully connected or convolution) composed with a convex and piecewise affine operator (such as a ReLU, leaky-ReLU, or absolute value activation; max/average/channel-pooling; maxout; all with or without skip connections) is a MASO Balestriero & Baraniuk (2018a; .Appendix A provides the parameters A ( ) , B ( ) for the MASO corresponding to the th layer of any DN constructed from linear plus piecewise affine and convex components.

Given this connection, we will identify z ( −1) above as the input (feature map) to the MASO DN layer and z ( ) as the output (feature map).

We also identify [z ( ) ] k in FORMULA0 and FORMULA1 as the output of the k th unit (aka neuron) of the th layer.

MASOs for higher-dimensional tensor inputs/outputs are easily developed by flattening.

The MASO/HVQ connection provides deep insights into how a DN clusters and organizes signals layer by layer in a hierarchical fashion BID1 .

However, the entire ap-proach requires that the nonlinearities be piecewise affine and convex, which precludes important activation functions like the sigmoid, hyperbolic tangent, and softmax.

The goal of this paper is to extend the MASO analysis framework of Section 2 to these and an infinitely large class of other nonlinearities by linking deterministic MASOs with probabilistic Gaussian Mixture Models (GMMs).

For now, we focus on a single unit k from layer of a MASO DN, which contains both linear and nonlinear operators; we generalize below in Section 5.

The key to the MASO mechanism lies in the VQ variables [t ( ) ] k ∀k, since they fully determine the output via (2).

For a special choice of bias, the VQ variable computation is equivalent to the K-means algorithm BID1 .

DISPLAYFORM0 , the MASO VQ partition corresponds to a K- DISPLAYFORM1 For example, consider a layer using a ReLU activation function.

Unit k of that layer partitions its input space using a K-means model with R ( ) = 2 centroids: the origin of the input space and the unit layer parameter [A ( ) ] k,1,· .

The input is mapped to the partition region corresponding to the closest centroid in terms of the Euclidean distance, and the corresponding affine mapping for that region is used to project the input and produce the layer output as in FORMULA1 .We now leverage the well-known relationship between K-means and Gaussian Mixture Models (GMMs) BID4 to GMM-ize the deterministic VQ process of max-affine splines.

As we will see, the constraint on the value of B ( ) k,rin Proposition 1 will be relaxed thanks to the GMM's ability to work with a nonuniform prior over the regions (in contrast to K-means).To move from a deterministic MASO model to a probabilistic GMM, we reformulate the HVQ selection variable [t DISPLAYFORM2 Armed with this, we define the following generative model for the layer input z ( −1) as a mixture of R ( ) Gaussians with mean DISPLAYFORM3 and identical isotropic covariance with parameter σ DISPLAYFORM4 with ∼ N(0, Iσ 2 ).

Note that this GMM generates an independent vector input z ( −1) for every unit k = 1, . . .

, D ( ) in layer .

For reasons that will become clear below in Section 3.3, we will refer to the GMM model (3) as the Soft MASO (SMASO) model.

We develop a joint, factorial model for the entire MASO layer (and not just one unit) in Section 5.

Given the GMM (3) and an input z ( −1) , we can compute a hard inference of the optimal VQ selection variable [t ( ) ] k via the maximum a posteriori (MAP) principle DISPLAYFORM0 The following result is proved in Appendix E.1.Theorem 2.

Given a GMM with parameters σ 2 = 1 and DISPLAYFORM1 , t = 1, . . . , R ( ) , the MAP inference of the latent selection variable [ t ( ) ] k given in (4) can be computed via the MASO HVQ (1) DISPLAYFORM2 Note in Theorem 2 that the bias constraint of Proposition 1 (which can be interpreted as imposing a uniform prior [π ( ) ] k,· ) is completely relaxed.

HVQ inference of the selection matrix sheds light on some of the drawbacks that affect any DN employing piecewise affine, convex activation functions.

First, during gradient-based learning, the gradient will propagate back only through the activated VQ regions that correspond to the few 1-hot entries in T ( )H .

The parameters of other regions will not be updated; this is known as the "dying neurons phenomenon" BID22 ; BID0 .

Second, the overall MASO mapping is continuous but not differentiable, which leads to unexpected gradient jumps during learning.

Third, the HVQ inference contains no information regarding the confidence of the VQ region selection, which is related to the distance of the query point to the region boundary.

As we will now see, this extra information can be very useful and gives rise to a range of classical and new activation functions.

We can overcome many of the limitations of HVQ inference in DNs by replacing the 1-hot entries of the HVQ selection matrix with the probability that the layer input belongs to a given VQ region DISPLAYFORM0 which follows from the simple structure of the GMM.

This corresponds to a soft inference of the categorical variable DISPLAYFORM1 H as the noise variance in (3) → 0.

Given the SVQ selection matrix, the MASO output is still computed via (2).

The SVQ matrix can be computed indirectly from an entropy-penalized MASO optimization; the following is reproved in Appendix E.2 for completeness.

DISPLAYFORM2 Proposition 2, which was first established in BID17 ; BID18 , unifies HVQ and SVQ in a single optimization problem.

The transition from HVQ (5) to SVQ FORMULA15 is obtained simply by adding the entropy regularization H(t).

Notice that removing the Entropy regularization from (7) leads to the same VQ as (5).

We summarize this finding in Table.

1.

Remarkably, switching from HVQ to SVQ MASO inference recovers several classical and powerful nonlinearities and provides an avenue to derive completely new ones.

Given a set of MASO parameters A ( ) , B ( ) for calculating the layer-output of a DN via (1), we can derive two distinctly different DNs: one based on the HVQ inference of (5) and one based on the SVQ inference of (6).

The following results are proved in Appendix E.5.

Appendix C discusses how the GMM and SVQ formulations shed new light on the impact of parameter initialization in DC learning plus how these formulations can be extended further.

Value for DISPLAYFORM0 DISPLAYFORM1

Changing viewpoint slightly, we can also derive classical nonlinearities like the sigmoid, tanh, and softmax BID13 from the soft inference perspective.

Consider a new soft DN layer whose unit output [z ( ) ] k is not the piecewise affine spline of (2) but rather the probability [z DISPLAYFORM0 ) that the input z ( ) falls into each VQ region.

The following propositions are proved in Appendix E.6.

Combining (5) and (6) yields a hybrid optimization for a new β-VQ that recovers hard, soft, and linear VQ inference as special cases DISPLAYFORM0 with the new hyper-parameter [β ( ) ] k ∈ (0, 1).

The β-VQ obtained from the above optimization problem utilizes [β ( ) ] k to balance the impact of the regularization term (introduced in the SVQ derivation (7)), allowing to recover and interpolate the VQ between linear, soft and hard (see Table.

1).

The following is proved in Appendix E.3.

Theorem 3.

The unique global optimum of FORMULA19 is given by DISPLAYFORM1 The β-VQ covers all of the theory developed above as special cases: β = 1 yields HVQ, β = TAB1 summarizes some of the many nonlinearities that are within reach of the β-VQ.

DISPLAYFORM2

The GMM (3) models the impact of only a single layer unit on the layer-input z ( −1) .

We can easily extend this model to a factorial model for z ( −1) that enables all D ( ) units at layer to combine their syntheses: DISPLAYFORM0 with ∼ N(0, Iσ 2 ).

This new model is a mixture of R ( ) Gaussians with means DISPLAYFORM1 and identical isotropic covariances with variance σ 2 .

The factorial aspect of the model means that the number of possible combinations of the t ( ) values grow exponentially with the number of units.

Hence, inferring the latent variables t ( ) quickly becomes intractable.

However, we can break this combinatorial barrier and achieve efficient, tractable VQ inference by constraining the MASO slope parameters A ( ) to be orthogonal DISPLAYFORM2 Orthogonality is achieved in a fully connected layer (multiplication by the dense matrix W ( ) composed with activation or pooling) when the rows of W ( ) are orthogonal.

Orthogonality is achieved in a convolution layer (multiplication by the convolution matrix C ( ) composed with activation or pooling) when the rows of C ( ) are either non-overlapping or properly apodized; see Appendix E.4 for the details plus the proof of the following result.

Theorem 4.

If the slope parameters A ( ) of a MASO are orthogonal in the sense of FORMULA0 DISPLAYFORM3 In an orthogonal, factorial MASO, optimal inference can be performed independently per factor, as opposed to jointly over all of the factors.

Orthogonality renders the joint MAP inference of the factorial model's VQs tractable.

The following result is proved in Appendix E.4.Practically, this not only lowers the computational complexity tremendously but also imparts the benefit of "uncorrelated unit firing," which has been shown to be advantageous in DNs BID21 .

Beyond the scope of this paper, such an orthogonalization strategy can also be applied to more general factorial models such as factorial GMMs Zemel (1994); BID10 and factorial HMMs BID11 .

Table 2 : Classification experiment to demonstrate the utility of orthogonal DN layers.

For three datasets and the same largeCNN architecture (detailed in Appendix D), we tabulate the classification accuracy (larger is better) and its standard deviation averaged over 5 runs with different Adam learning rates.

In each case, orthogonal fully-connected and convolution matrices improve the classification accuracy over the baseline.

Corollary 1.

When the conditions of Theorem 4 are fulfilled, the joint MAP estimate for the VQs of the factorial model (10) DISPLAYFORM4 and thus can be computed with linear complexity in the number of units.

The advantages of orthogonal or near-orthogonal filters have been explored empirically in various settings, from GANs Brock et al. (2016) to RNNs Huang et al. (2017) , typically demonstrating improved performance.

Table 2 tabulates the results of a simple confirmation experiment with the largeCNN architecture described in Appendix D.

We added to the standard cross-entropy loss a term DISPLAYFORM5 2 that penalizes non-orthogonality (recall (11)).

We did not cross-validate the penalty coefficient λ but instead set it equal to 1.

The tabulated results show clearly that favoring orthogonal filters improves accuracy across both different datasets and different learning settings.

Since the orthogonality penalty does not guarantee true orthogonality but simply favors it, we performed one additional experiment where we reparametrized the fully-connected and convolution matrices using the Gram-Schmidt (GS) process BID7 so that they were truly orthogonal.

Thanks to the differentiability of all of the operations involved in the GS process, we can backpropagate the loss to the orthogonalized filters in order to update them in learning.

We also used the swish activation, which we showed to be a β-VQ nonlinearity in Section 4.

Since the GS process adds significant computational overhead to the learning algorithm, we conducted only one experiment on the largest dataset (CIFAR100).

The exactly orthogonalized largeCNN achieved a classification accuracy of 61.2%, which is a major improvement over all of the results in the bottom (CIFAR100) cell of Table 2 .

This indicates that there are good reasons to try to improve on the simple orthogonality-penalty-based approach.

Our development of the SMASO model opens the door to several new research questions.

First, we have merely scratched the surface in the exploration of new nonlinear activation functions and pooling operators based on the SVQ and β-VQ.

For example, the soft-or β-VQ versions of leakyReLU, absolute value, and other piecewise affine and convex nonlinearities could outperform the new swish nonlinearity.

Second, replacing the entropy penalty in the (7) and (8) with a different penalty will create entirely new classes of nonlinearities that inherit the rich analytical properties of MASO DNs.

Third, orthogonal DN filters will enable new analysis techniques and DN probing methods, since from a signal processing point of view problems such as denoising, reconstruction, compression have been extensively studied in terms of orthogonal filters.

This work was partially supported by NSF grants IIS-17-30574 and IIS-18-38177, AFOSR grant FA9550-18-1-0478, ARO grant W911NF-15-1-0316, ONR grants N00014-17-1-2551 and N00014-18-12571, DARPA grant G001534-7500, and a DOD Vannevar Bush Faculty Fellowship (NSSEFF) grant N00014-18-1-2047.

propose a brief approximation result.

Due to the specificity of the convolution operator we are able to provide a tractable inference coupled with an apodization scheme.

To demonstrate this, we first highlight that any input can be represented as a direct sum of its apodized patches.

Then, we see that filtering apodized patches with a filter is equivalent to convolving the input with apodized filters.

We first need to introduce the patch notation.

We define a patch P[z ( −1) ](pi, pj) ∈ {1, . . .

, I ( ) } × {1, . . . , J ( ) } as the slice of the input with indices c = 1, . . . , K ( ) , i = (all channels) and (i, j) ∈ {pi, . . .

, pi + I The above highlights the ability to treat an input via its collection of patches with the condition to apply the defined apodization function.

With the above, we can demonstrate how minimizing the per patch reconstruction loss leads to minimizing the overall input modeling DISPLAYFORM0 which represents the internal modeling of the factorial model applied across filters and patches.

As a result, when performing the per position minimization one minimizes an upper bound which ultimately reaches the global minimum as

We first present the topologies used in the experiments except for the notation ResNetD-W which is the standard wide ResNet based topology with depth D and width W .

We thus have the following network architectures for smallCNN and largeCNN: largeCNN

@highlight

Reformulate deep networks nonlinearities from a vector quantization scope and bridge most known nonlinearities together.