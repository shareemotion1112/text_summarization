Designing a convolution for a spherical neural network requires a delicate tradeoff between efficiency and rotation equivariance.

DeepSphere, a method based on a graph representation of the discretized sphere, strikes a controllable balance between these two desiderata.

This contribution is twofold.

First, we study both theoretically and empirically how equivariance is affected by the underlying graph with respect to the number of pixels and neighbors.

Second, we evaluate DeepSphere on relevant problems.

Experiments show state-of-the-art performance and demonstrates the efficiency and flexibility of this formulation.

Perhaps surprisingly, comparison with previous work suggests that anisotropic filters might be an unnecessary price to pay.

Spherical data is found in many applications (figure 1).

Planetary data (such as meteorological or geological measurements) and brain activity are example of intrinsically spherical data.

The observation of the universe, LIDAR scans, and the digitalization of 3D objects are examples of projections due to observation.

Labels or variables are often to be inferred from them.

Examples are the inference of cosmological parameters from the distribution of mass in the universe , the segmentation of omnidirectional images (Khasanova & Frossard, 2017) , and the segmentation of cyclones from Earth observation (Mudigonda et al., 2017) .

2 A rigid full-sphere sampling is not ideal: brain activity is only measured on the scalp, the Milky Way's galactic plane masks observations, climate scientists desire a variable resolution, and the position of weather stations is arbitrary and changes over time.

(e) Graphs can faithfully and efficiently represent sampled spherical data by placing vertices where it matters.

As neural networks (NNs) have proved to be great tools for inference, variants have been developed to handle spherical data.

Exploiting the locally Euclidean property of the sphere, early attempts used standard 2D convolutions on a grid sampling of the sphere (Boomsma & Frellsen, 2017; Su & Grauman, 2017; Coors et al., 2018) .

While simple and efficient, those convolutions are not equivariant to rotations.

On the other side of this tradeoff, Cohen et al. (2018) and Esteves et al. (2018) proposed to 2 METHOD DeepSphere leverages graph convolutions to achieve the following properties: (i) computational efficiency, (ii) sampling flexibility, and (iii) rotation equivariance (section 3).

The main idea is to model the sampled sphere as a graph of connected pixels: the length of the shortest path between two pixels is an approximation of the geodesic distance between them.

We use the graph CNN formulation introduced in (Defferrard et al., 2016 ) and a pooling strategy that exploits hierarchical samplings of the sphere.

A sampling scheme V = {x i ∈ S 2 } n i=1 is defined to be the discrete subset of the sphere containing the n points where the values of the signals that we want to analyse are known.

For a given continuous signal f , we represent such values in a vector f ∈ R n .

As there is no analogue of uniform sampling on the sphere, many samplings have been proposed with different tradeoffs.

In this work, depending on the considered application, we will use the equiangular (Driscoll & Healy, 1994) , HEALPix (Gorski et al., 2005) , and icosahedral (Baumgardner & Frederickson, 1985) samplings.

Graph.

From V, we construct a weighted undirected graph G = (V, w), where the elements of V are the vertices and the weight w ij = w ji is a similarity measure between vertices x i and x j .

The combinatorial graph Laplacian L ∈ R n×n is defined as L = D − A, where A = (w ij ) is the weighted adjacency matrix, D = (d ii ) is the diagonal degree matrix, and d ii = j w ij is the weighted degree of vertex x i .

Given a sampling V, usually fixed by the application or the available measurements, the freedom in constructing G is in setting w. Section 3 shows how to set w to minimize the equivariance error.

Convolution.

On Euclidean domains, convolutions are efficiently implemented by sliding a window in the signal domain.

On the sphere however, there is no straightforward way to implement a convolution in the signal domain due to non-uniform samplings.

Convolutions are most often performed in the spectral domain through a spherical harmonic transform (SHT).

That is the approach taken by Cohen et al. (2018) and Esteves et al. (2018) , which has a computational cost of O(n 3/2 ) on isolatitude samplings (such as the HEALPix and equiangular samplings) and O(n 2 ) in general.

On the other hand, following Defferrard et al. (2016) , graph convolutions can be defined as

where P is the polynomial order (which corresponds to the filter's size) and α i are the coefficients to be optimized during training.

3 Those convolutions are used by Khasanova & Frossard (2017) and and cost O(n) operations through a recursive application of L.

Pooling.

Down-and up-sampling is natural for hierarchical samplings, 5 where each subdivision divides a pixel in (an equal number of) child sub-pixels.

To pool (down-sample), the data supported on the sub-pixels is summarized by a permutation invariant function such as the maximum or the average.

To unpool (up-sample), the data supported on a pixel is copied to all its sub-pixels.

Architecture.

All our NNs are fully convolutional, and employ a global average pooling (GAP) for rotation invariant tasks.

Graph convolutional layers are always followed by batch normalization and ReLU activation, except in the last layer.

Note that batch normalization and activation act on the elements of f independently, and hence don't depend on the domain of f .

While the graph framework offers great flexibility, its ability to faithfully represent the underlying sphere -for graph convolutions to be rotation equivariant -highly depends on the sampling locations and the graph construction.

A continuous function f :

We require F V to be a suitable subspace of continuous functions such that T V is invertible, i.e., the function f ∈ F V can be unambiguously reconstructed from its sampled values f .

The existence of such a subspace depends on the sampling V, and its characterization is a common problem in signal processing (Driscoll & Healy, 1994) .

For most samplings, it is not known if F V exists and hence if T V is invertible.

A special case is the equiangular sampling where a sampling theorem holds, and thus a closed-form of T −1 V is known.

For samplings where no such sampling formula is available, we leverage the discrete SHT to reconstruct f from f = T V f , thus approximating T −1 V .

For all theoretical considerations, we assume that F V exists and f ∈ F V .

By definition, the (spherical) graph convolution is rotation equivariant if and only if it commutes with the rotation operator defined as R(g), g ∈ SO(3): R(g)f (x) = f g −1 x .

In the context of this work, graph convolution is performed by recursive applications of the graph Laplacian ((1)).

Hence, if R(g) commutes with L, then, by recursion, it will also commute with the convolution h(L).

As a result, h(L) is rotation equivariant if and only if

∀f ∈ F V and ∀g ∈ SO(3),

V .

For an empirical evaluation of equivariance, we define the normalized equivariance error for a signal f and a rotation g as

More generally for a class of signals f ∈ C ⊂ F V , the mean equivariance error defined as

represents the overall equivariance error.

The expected value is obtained by averaging over a finite number of random functions and random rotations.

Figure 2: Mean equivariance error (3).

There is a clear tradeoff between equivariance and computational cost, governed by the number of vertices n and edges kn.

Considering the equiangular sampling and graphs where each vertex is connected to 4 neighbors (north, south, east, west), Khasanova & Frossard (2017) designed a weighting scheme to minimize (3) for longitudinal and latitudinal rotations 6 .

Their solution gives weights inversely proportional to Euclidean distances:

While the resulting convolution is not equivariant to the whole of SO(3) (figure 2), it is enough for omnidirectional imaging because, as gravity consistently orients the sphere, objects only rotate longitudinally or latitudinally.

To achieve equivariance to all rotations, we take inspiration from Belkin & Niyogi (2008) .

They prove that for a random uniform sampling, the graph Laplacian L built from weights

converges to the Laplace-Beltrami operator ∆ S 2 as the number of samples grows to infinity.

This result is a good starting point as ∆ S 2 commutes with rotation, i.e., ∆ S 2 R(g) = R(g)∆ S 2 .

While the weighting scheme is full (i.e., every vertex is connected to every other vertex), most weights are small due to the exponential.

We hence make an approximation to limit the cost of the convolution (1) by only considering the k nearest neighbors (k-NN) of each vertex.

Given k, the optimal kernel width t is found by searching for the minimizer of (3).

Figure 3 shows the optimal kernel widths found for various resolutions of the HEALPix sampling.

As predicted by the theory, t n ∝ n β , β ∈ R. Importantly however, the optimal t also depends on the number of neighbors k.

Considering the HEALPix sampling, connected each vertex to their 8 adjacent vertices in the tiling of the sphere, computed the weights with (5), and heuristically set t to half the average squared Euclidean distance between connected vertices.

This heuristic however overestimates t (figure 3) and leads to an increased equivariance error (figure 2).

We analyze the proposed weighting scheme both theoretically and empirically.

Theoretical convergence.

We extend the work of (Belkin & Niyogi, 2008) to a sufficiently regular, deterministic sampling.

Following their setting, we work with the extended graph Laplacian operator as the linear op-

This operator extends the graph Laplacian with the weighting scheme (5) to each point of the sphere (i.e.,

As the radius of the kernel t will be adapted to the number of samples, we scale the operator aŝ L

Given a sampling V, we define σ i to be the patch of the surface of the sphere corresponding to x i , A i its corresponding area, and d i the largest distance between the center x i and any point on the surface

Theorem 3.1.

For a sampling V of the sphere that is equi-area and such that

This is a major step towards equivariance, as the Laplace-Beltrami operator commutes with rotation.

Based on this property, we show the equivariance of the scaled extended graph Laplacian.

Theorem 3.2.

Under the hypothesis of theorem 3.1, the scaled graph Laplacian commutes with any rotation, in the limit of infinite sampling, i.e.,

From this theorem, it follows that the discrete graph Laplacian will be equivariant in the limit of n → ∞ as by construction

n f and as the scaling does not affect the equivariance property of L t n .

Importantly, the proof of Theorem 3.1 (in Appendix A) inspires our construction of the graph Laplacian.

In particular, it tells us that t should scale as n β , which has been empirically verified (figure 3).

Nevertheless, it is important to keep in mind the limits of Theorem 3.1 and 3.2.

Both theorems present asymptotic results, but in practice we will always work with finite samplings.

Furthermore, since this method is based on the capability of the eigenvectors of the graph Laplacian to approximate the spherical harmonics, a stronger type of convergence of the graph Laplacian would be preferable, i.e., spectral convergence (that is proved for a full graph in the case of random sampling for a class of Lipschitz functions in (Belkin & Niyogi, 2007) ).

Finally, while we do not have a formal proof for it, we strongly believe that the HEALPix sampling does satisfy the hypothesis

, with α very close or equal to 1/2.

The empirical results discussed in the next paragraph also points in this direction.

This is further discussed in Appendix A.

Empirical convergence.

Figure 2 shows the equivariance error (3) for different parameter sets of DeepSphere for the HEALPix sampling as well as for the graph construction of Khasanova & Frossard (2017) for the equiangular sampling.

The error is estimated as a function of the sampling resolution and signal frequency.

The resolution is controlled by the number of pixels n = 12N 2 side for HEALPix and n = 4b 2 for the equiangular sampling.

The frequency is controlled by setting the set C to functions f made of spherical harmonics of a single degree .

To allow for an almost perfect implementation (up to numerical errors) of the operator R V , the degree was chosen in the range (0, 3N side − 1) for HEALPix and (0, b) for the equiangular sampling (Gorski et al., 1999) .

Using these parameters, the measured error is mostly due to imperfections in the empirical approximation of the Laplace-Beltrami operator and not to the sampling.

Figure 2 shows that the weighting scheme (4) from (Khasanova & Frossard, 2017) does indeed not lead to a convolution that is equivariant to all rotations g ∈ SO(3).

7 For k = 8 neighbors, selecting the optimal kernel width t improves on at no cost, highlighting the importance of this parameter.

Increasing the resolution decreases the equivariance error in the high frequencies, an effect most probably due to the sampling.

Most importantly, the equivariance error decreases when connecting more neighbors.

Hence, the number of neighbors k gives us a precise control of the tradeoff between cost and equivariance.

The recognition of 3D shapes is a rotation invariant task: rotating an object doesn't change its nature.

While 3D shapes are usually represented as meshes or point clouds, representing them as spherical maps (figure 4) naturally allows a rotation invariant treatment.

The SHREC'17 shape retrieval contest (Savva et al., 2017) contains 51,300 randomly oriented 3D models from ShapeNet (Chang et al., 2015) , to be classified in 55 categories (tables, lamps, airplanes, etc.).

As in (Cohen et al., 2018) , objects are represented by 6 spherical maps.

At each pixel, a ray is traced towards the center of the sphere.

The distance from the sphere to the object forms a depth map.

The cos and sin of the surface angle forms two normal maps.

The same is done for the object's convex hull.

8 The maps are sampled by an equiangular sampling with bandwidth b = 64 (n = 4b 2 = 16, 384 pixels) or an HEALPix sampling with N side = 32 (n = 12N 2 side = 12, 288 pixels).

The equiangular graph is built with (4) and k = 4 neighbors (following Khasanova & Frossard, 2017) .

The HEALPix graph is built with (5), k = 8, and a kernel width t set to the average of the distances (following .

The NN is made of 5 graph convolutional layers, each followed by a max pooling layer which down-samples by 4.

A GAP and a fully connected layer with softmax follow.

The polynomials are all of order P = 3 and the number of channels per layer is 16, 32, 64, 128, 256, respectively.

Following Esteves et al. (2018) , the cross-entropy plus a triplet loss is optimized with Adam for 30 epochs on the dataset augmented by 3 random translations.

The learning rate is 5 · 10 −2 and the batch size is 32.

Results are shown in table 1.

As the network is trained for shape classification rather than retrieval, we report the classification F1 alongside the mAP used in the retrieval contest.

10 DeepSphere achieves the same performance as Cohen et al. (2018) and Esteves et al. (2018) at a much lower cost, suggesting that anisotropic filters are an unnecessary price to pay.

As the information in those spherical maps resides in the low frequencies (figure 5), reducing the equivariance error didn't translate into improved performance.

For the same reason, using the more uniform HEALPix sampling or lowering the resolution down to N side = 8 (n = 768 pixels) didn't impact performance either.

7 We however verified that the convolution is equivariant to longitudinal and latitudinal rotations, as intended.

8 Albeit we didn't observe much improvement by using the convex hull.

7 As implemented in https://github.com/jonas-koehler/s2cnn.

Figure 7: Tradeoff between cost and accuracy.

Given observations, cosmologists estimate the posterior probability of cosmological parameters, such as the matter density Ω m and the normalization of the matter power spectrum σ 8 .

Those parameters are typically estimated by likelihood-free inference, which requires a function to predict the parameters from simulations.

As that is complicated to setup, prediction methods are typically benchmarked on the classification of spherical maps instead (Schmelzle et al., 2017) .

We used the same task, data, and setup as : the classification of 720 partial convergence maps made of n ≈ 10 6 pixels (1/12 ≈ 8% of a sphere at N side = 1024) from two ΛCDM cosmological models, (Ω m = 0.31, σ 8 = 0.82) and (Ω m = 0.26, σ 8 = 0.91), at a relative noise level of 3.5 (i.e., the signal is hidden in noise of 3.5 times higher standard deviation).

Convergence maps represent the distribution of over-and under-densities of mass in the universe (see Bartelmann, 2010, for a review of gravitational lensing).

Graphs are built with (5), k = 8, 20, 40 neighbors, and the corresponding optimal kernel widths t given in section 3.2.

Following , the NN is made of 5 graph convolutional layers, each followed by a max pooling layer which down-samples by 4.

A GAP and a fully connected layer with softmax follow.

The polynomials are all of order P = 4 and the number of channels per layer is 16, 32, 64, 64, 64, respectively.

The cross-entropy loss is optimized with Adam for 80 epochs.

The learning rate is 2 · 10 −4 · 0.999 step and the batch size is 8.

Unlike on SHREC'17, results (table 2) show that a lower equivariance error on the convolutions translates to higher performance.

That is probably due to the high frequency content of those maps ( figure 5 ).

There is a clear cost-accuracy tradeoff, controlled by the number of neighbors k (figure 7).

This experiment moreover demonstrates DeepSphere's flexibility (using partial spherical maps) and scalability (competing spherical CNNs were tested on maps of at most 10, 000 pixels).

We evaluate our method on a task proposed by (Mudigonda et al., 2017) : the segmentation of extreme climate events, Tropical Cyclones (TC) and Atmospheric Rivers (AR), in global climate simulations (figure 1c).

The data was produced by a 20-year run of the Community Atmospheric Model v5 (CAM5) and consists of 16 channels such as temperature, wind, humidity, and pressure at multiple altitudes.

We used the pre-processed dataset from (Jiang et al., 2019).

11 There is 1,072,805 spherical maps, down-sampled to a level-5 icosahedral sampling (n = 10 · 4 l + 2 = 10, 242 pixels).

The labels are heavily unbalanced with 0.1% TC, 2.2% AR, and 97.7% background (BG) pixels.

The graph is built with (5), k = 6 neighbors, and a kernel width t set to the average of the distances.

Following Jiang et al. (2019) , the NN is an encoder-decoder with skip connections.

Details in section C.3.

The polynomials are all of order P = 3.

The cross-entropy loss (weighted or nonweighted) is optimized with Adam for 30 epochs.

The learning rate is 1 · 10−3 and the batch size is 64.

Results are shown in table 3 (details in tables 6, 7 and 8).

The mean and standard deviation are computed over 5 runs.

Note that while Jiang et al. (2019) Table 4 : Prediction results on data from weather stations.

Structure always improves performance.

the-art performance, suggesting again that anisotropic filters are unnecessary.

Note that results from Mudigonda et al. (2017) cannot be directly compared as they don't use the same input channels.

(ii) that a larger architecture can compensate for the lack of generality.

We indeed observed that more feature maps and depth led to higher performance (section C.3).

To demonstrate the flexibility of modeling the sampled sphere by a graph, we collected historical measurements from n ≈ 10, 000 weather stations scattered across the Earth.

12 The spherical data is heavily non-uniformly sampled, with a much higher density of weather stations over North America than the Pacific (figure 1d).

For illustration, we devised two artificial tasks.

A dense regression: predict the temperature on a given day knowing the temperature on the previous 5 days.

A global regression: predict the day (represented as one period of a sine over the year) from temperature or precipitations.

Predicting from temperature is much easier as it has a clear yearly pattern.

The graph is built with (5), k = 5 neighbors, and a kernel width t set to the average of the distances.

The equivariance property of the resulting graph has not been tested, and we don't expect it to be good due to the heavily non-uniform sampling.

The NN is made of 3 graph convolutional layers.

The polynomials are all of order P = 0 or 4 and the number of channels per layer is 50, 100, 100, respectively.

For the global regression, a GAP and a fully connected layer follow.

For the dense regression, a graph convolutional layer follows instead.

The MSE loss is optimized with RMSprop for 250 epochs.

The learning rate is 1 · 10 −3 and the batch size is 64.

Results are shown in table 4.

While using a polynomial order P = 0 is like modeling each time series independently with an MLP, orders P > 0 integrate neighborhood information.

Results show that using the structure induced by the spherical geometry always yields better performance.

This work showed that DeepSphere strikes an interesting, and we think currently optimal, balance between desiderata for a spherical CNN.

A single parameter, the number of neighbors k a pixel is connected to in the graph, controls the tradeoff between cost and equivariance (which is linked to performance).

As computational cost and memory consumption scales linearly with the number of pixels, DeepSphere scales to spherical maps made of millions of pixels, a required resolution to faithfully represent cosmological and climate data.

Also relevant in scientific applications is the flexibility offered by a graph representation (for partial coverage, missing data, and non-uniform samplings).

Finally, the implementation of the graph convolution is straightforward, and the ubiquity of graph neural networks -pushing for their first-class support in DL frameworks -will make implementations even easier and more efficient.

A potential drawback of graph Laplacian-based approaches is the isotropy of graph filters, reducing in principle the expressive power of the NN.

Experiments from Cohen et al. (2019) and Boscaini et al. (2016) indeed suggest that more general convolutions achieve better performance.

Our experiments on 3D shapes (section 4.1) and climate (section 4.3) however show that DeepSphere's isotropic filters do not hinder performance.

Possible explanations for this discrepancy are that NNs somehow compensate for the lack of anisotropic filters, or that some tasks can be solved with isotropic filters.

The distortions induced by the icosahedral projection in (Cohen et al., 2019) or the leakage of curvature information in (Boscaini et al., 2016) might also alter performance.

Developing graph convolutions on irregular samplings that respect the geometry of the sphere is another research direction of importance.

Practitioners currently interpolate their measurements (coming from arbitrarily positioned weather stations, satellites or telescopes) to regular samplings.

This practice either results in a waste of resolution or computational and storage resources.

Our ultimate goal is for practitioners to be able to work directly on their measurements, however distributed.

Left blank for anonymity reason.

Left blank for anonymity reason.

A PROOF OF THEOREM 3.1

Preliminaries.

The proof of theorem 3.1 is inspired from the work of Belkin & Niyogi (2008) .

As a result, we start by restating some of their results.

Given a sampling V = {x i ∈ M} n i=1 of a closed, compact and infinitely differentiable manifold M, a smooth (∈ C ∞ (M)) function f : M → R, and defined the vector f of samples of f as follows:

The proof is constructed by leveraging 3 different operators:

•

The extended graph Laplacian operator, already presented in (6), is a linear operator L t n :

Note that we have the following relation

where µ is the uniform probability measure on the manifold M, and vol(M) is the volume of M.

• The Laplace-Beltrami operator ∆ M is defined as the divergence of the gradient

of a differentiable function f : M → R. The gradient ∇f : M → T p M is a vector field defined on the manifold pointing towards the direction of steepest ascent of f , where T p M is the affine space of all vectors tangent to M at p.

Leveraging these three operators, Belkin & Niyogi (2008; 2007) have build proofs of both pointwise and spectral convergence of the extended graph Laplacian towards the Laplace-Beltrami operator in the general setting of any compact, closed and infinitely differentiable manifold M, where the sampling V is drawn randomly on the manifold.

For this reason, their results are all to be interpreted in a probabilistic sense.

Their proofs consist in establishing that (6) converges in probability towards (8) as n → ∞ and (8) converges towards (9) as t → 0.

In particular, this second step is given by the following: Proposition 1 (Belkin & Niyogi (2008) , Proposition 4.4).

Let M be a k-dimensional compact smooth manifold embedded in some Euclidean space R N , and fix y ∈ M. Let f ∈ C ∞ (M).

Then

Building the proof.

As the sphere is a compact smooth manifold embedded in R 3 , we can reuse proposition 1.

Thus, our strategy to prove Theorem 3.1 is to (i) show that

for a particular class of deterministic samplings, and (ii) apply Proposition 1.

We start by proving that for smooth functions, for any fixed t, the extended graph Laplacian L t n converges towards its continuous counterpart L t as the sampling increases in size.

Proposition 2.

For an equal area sampling {x i ∈ S 2 } n i=1 :

A i = A j ∀i, j of the sphere it is true that for all f : S 2 → R Lipschitz with respect to the Euclidean distance · with Lipschitz constant

Furthermore, for all y ∈ S 2 the Heat Kernel Graph Laplacian operator L t n converges pointwise to the functional approximation of the Laplace Beltrami operator

Proof.

Assuming f : S 2 → R is Lipschitz with Lipschitz constant C f , we have

where σ i ⊂ S 2 is the subset of the sphere corresponding to the patch around x i .

Remember that the sampling is equal area.

Hence, using the triangular inequality and summing all the contributions of the n patches, we obtain

A direct application of this result leads to the following pointwise convergences

Definitions 6 and 8 end the proof.

The last proposition show that for a fixed t,

To utilize Proposition 1 and complete the proof, we need to find a sequence of t n for which this holds as t n → 0.

Furthermore this should hold with a faster decay than

Proposition 3.

Given a sampling regular enough, i.e., for which we assume A i = A j ∀i, j and d (n) ≤ C n α , α ∈ (0, 1/2], a Lipschitz function f and a point y ∈ S 2 there exists a sequence t n = n β , β < 0 such that

Proof.

To ease the notation, we define

We start with the following inequality

where C φ t y is the Lipschitz constant of x → φ t (x, y) and the last inequality follows from Proposition 2.

Using the assumption

We now find the explicit dependence between t and C φ t y

is the Lipschitz constant of the function x → K t (x; y).

We note that this constant does not depend on y:

Hence we have

Inculding this result in (14) and rescaling by 1/4πt 2 , we obtain

In order for

and n α t 2 = n 2β+α n→∞

As a result, for t = n β with β ∈ (− 1 5 , 0) we have

which concludes the proof.

Theorem 3.1, is then an immediate consequence of Proposition 3 and 1.

Proof of Theorem 3.1.

Thanks to Proposition 3 and Proposition 1 we conclude that ∀y ∈ S

In (Belkin & Niyogi, 2008) , the sampling is drawn from a uniform random distribution on the sphere, and their proof heavily relies on the uniformity properties of the distribution from which the sampling is drawn.

In our case the sampling is deterministic, and this is indeed a problem that we need to overcome by imposing the regularity conditions above.

Table 5 : Official metrics from the SHREC'17 object retrieval competition.

To conclude, we see that the result obtained is of similar form than the result obtained in (Belkin & Niyogi, 2008) .

Given the kernel density t(n) = n β , Belkin & Niyogi (2008) proved convergence in the random case for β ∈ (− 1 4 , 0) and we proved convergence in the deterministic case for β ∈ (− 2α 5 , 0), where α ∈ (0, 1/2] (for the spherical manifold).

Proof.

Fix x ∈ S 2 .

Since any rotation R(g) is an isometry, and the Laplacian ∆ commutes with all isometries of a Riemanniann manifold, and defining R(g)f =: f for ease of notation, we can write that

Since g −1 (x) ∈ S 2 and f still satisfies hypothesis, we can apply theorem 3.1 to say that Table 7 : Results on climate event segmentation: average precision.

Tropical cyclones (TC) and atmospheric rivers (AR) are the two positive classes.

Note that a weighted cross-entropy loss is not optimal for the average precision metric.

C.3 CLIMATE EVENT SEGMENTATION Table 6 , 7, and 8 show the accuracy, mAP, and efficiency of all the NNs we ran.

The experiment with the model from Jiang et al. (2019) was rerun in order to obtain the AP metrics, but with a batch size of 64 instead of 256 due to GPU memory limit.

Several experiments were run with different architectures for DeepSphere (DS).

Jiang architecture use a similar one as Jiang et al. (2019), with only the convolutional operators replaced.

DeepSphere only is the original architecture giving the best results, deeper and with four times more feature maps than Jiang architecture.

And the wider architecture is the same as the previous one with two times the number of feature maps.

Regarding the weighted loss, the weights are chosen with scikit-learn function compute class weight on the training set.

<|TLDR|>

@highlight

A graph-based spherical CNN that strikes an interesting balance of trade-offs for a wide variety of applications.

@highlight

Combines existing CNN frameworks based on the discretization of a sphere as a graph to show a convergence result which is related to the rotation equivalence on a sphere.

@highlight

The authors use the existing graph CNN formulation and a pooling strategy that exploits hierarchical pixelations of the sphere to learn from the discretized sphere.