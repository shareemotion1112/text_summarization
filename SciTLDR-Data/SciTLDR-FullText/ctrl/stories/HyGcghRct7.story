We propose a new learning-based approach to solve ill-posed inverse problems in imaging.

We address the case where ground truth training samples are rare and the problem is severely ill-posed---both because of the underlying physics and because we can only get few measurements.

This setting is common in geophysical imaging and remote sensing.

We show that in this case the common approach to directly learn the mapping from the measured data to the reconstruction becomes unstable.

Instead, we propose to first learn an ensemble of simpler mappings from the data to projections of the unknown image into random piecewise-constant subspaces.

We then combine the projections to form a final reconstruction by solving a deconvolution-like problem.

We show experimentally that the proposed method is more robust to measurement noise and corruptions not seen during training than a directly learned inverse.

A variety of imaging inverse problems can be discretized to a linear system y = Ax + η where y ∈ R M is the measured data, A ∈ R M ×N is the imaging or forward operator, x ∈ X ⊂ R N is the object being probed by applying A (often called the model), and η is the noise.

Depending on the application, the set of plausible reconstructions X could model natural, seismic, or biomedical images.

In many cases the resulting inverse problem is ill-posed, either because of the poor conditioning of A (a consequence of the underlying physics) or because M N .A classical approach to solve ill-posed inverse problems is to minimize an objective functional regularized via a certain norm (e.g. 1 , 2 , total variation (TV) seminorm) of the model.

These methods promote general properties such as sparsity or smoothness of reconstructions, sometimes in combination with learned synthesis or analysis operators, or dictionaries BID44 ).In this paper, we address situations with very sparse measurement data (M N ) so that even a coarse reconstruction of the unknown model is hard to get with traditional regularization schemes.

Unlike artifact-removal scenarios where applying a regularized pseudoinverse of the imaging operator already brings out considerable structure, we look at applications where standard techniques cannot produce a reasonable image (Figure 1 ).

This highly unresolved regime is common in geophysics and requires alternative, more involved strategies BID12 ).An appealing alternative to classical regularizers is to use deep neural networks.

For example, generative models (GANs) based on neural networks have recently achieved impressive results in regularization of inverse problems BID7 , BID29 ).

However, a difficulty in geophysical applications is that there are very few examples of ground truth models available for training (sometimes none at all).

Since GANs require many, they cannot be applied to such problems.

This suggests to look for methods that are not very sensitive to the training dataset.

Conversely, it means that the sought reconstructions are less detailed than what is expected in data-rich settings; for Figure 1 : We reconstruct an image x from its tomographic measurements.

In moderately ill-posed problems, conventional methods based on the pseudoinverse and regularized non-negative least squares (x ∈ [0, 1] N , N is image dimension) give correct structural information.

In fact, total variation (TV) approaches give very good results.

A neural network BID23 ) can be trained to directly invert and remove the artifacts (NN).

In a severely ill-posed problem on the other hand (explained in FIG2 ) with insufficient ground truth training data, neither the classical techniques nor a neural network recover salient geometric features.an example, see the reconstructions of the Tibetan plateau BID51 ).In this paper, we propose a two-stage method to solve ill-posed inverse problems using random low-dimensional projections and convolutional neural networks.

We first decompose the inverse problem into a collection of simpler learning problems of estimating projections into random (but structured) low-dimensional subspaces of piecewise-constant images.

Each projection is easier to learn in terms of generalization error BID10 ) thanks to its lower Lipschitz constant.

In the second stage, we solve a new linear inverse problem that combines the estimates from the different subspaces.

We show that this converts the original problem with possibly non-local (often tomographic) measurements into an inverse problem with localized measurements, and that in fact, in expectation over random subspaces the problem becomes a deconvolution.

Intuitively, projecting into piecewise-constant subspaces is equivalent to estimating local averages-a simpler problem than estimating individual pixel values.

Combining the local estimates lets us recover the underlying structure.

We believe that this technique is of independent interest in addressing inverse problems.

We test our method on linearized seismic traveltime tomography BID8 BID20 ) with sparse measurements and show that it outperforms learned direct inversion in quality of achieved reconstructions, robustness to measurement errors, and (in)sensitivity to the training data.

The latter is essential in domains with insufficient ground truth images.

Although neural networks have long been used to address inverse problems BID32 BID21 ; BID42 ), the past few years have seen the number of related deep learning papers grow exponentially.

The majority address biomedical imaging BID16 ; BID22 ) with several special issues 1 and review papers BID28 BID30 ) dedicated to the topic.

All these papers address reconstruction from subsampled or low-quality data, often motivated by reduced scanning time or lower radiation doses.

Beyond biomedical imaging, machine learning techniques are emerging in geophysical imaging BID3 ; BID25 ; BID5 ), though at a slower pace, perhaps partly due to the lack of standard open datasets.

Existing methods can be grouped into non-iterative methods that learn a feed-forward mapping from the measured data y (or some standard manipulation such as adjoint or a pseudoinverse) to the model Figure 2: Regularization by Λ random projections: 1) each orthogonal projection is approximated by a convolutional neural network which maps from a non-negative least squares reconstruction of an image to its projection onto a lower dimension subspace of Delaunay triangulations; 2) projections are combined to estimate the original image using regularized least squares.either the regularizer being a neural network BID26 ), or neural networks replacing various iteration components such as gradients, projectors, or proximal mappings BID24 ; BID1 a) ; BID9 ).

These are further related to the notion of plug-and-play regularization BID47 ), as well as early uses of neural nets to unroll and adapt standard sparse reconstruction algorithms BID14 ; BID50 ).

An advantage of the first group of methods is that they are fast; an advantage of the second group is that they are better at enforcing data consistency.

Generative models A rather different take was proposed in the context of compressed sensing where the reconstruction is constrained to lie in the range of a pretrained generative network BID6 ).

Their scheme achieves impressive results on random sensing operators and comes with theoretical guarantees.

However, training generative networks requires many examples of ground truth and the method is inherently subject to dataset bias.

Here, we focus on a setting where ground-truth samples are very few or impossible to obtain.

There are connections between our work and sketching BID15 ; BID35 ) where the learning problem is also simplified by random low-dimensional projections of some object-either the data or the unknown reconstruction itself BID54 ).

This also exposes natural connections with learning via random features BID39 ).

The two stages of our method are (i) decomposing a "hard" learning task of directly learning an unstable operator into an ensemble of "easy" tasks of estimating projections of the unknown model into low-dimensional subspaces; and (ii) combining these projection estimates to solve a reformulated inverse problem for x. The two stages are summarized in Figure 2 .

While our method is applicable to continuous and non-linear settings, we focus on linear finite-dimensional inverse problems.

Statistical learning theory tells us that the number of samples required to learn an M -variate LLipschitz function to a given sup-norm accuracy is O(L M ) BID10 ).

While this result is proved for scalar-valued multivariate maps, it is reasonable to expect the same scaling in L to hold for vector-valued maps.

This motivates us to study Lipschitz properties of the projected inverse maps.

We wish to reconstruct x, an N -pixel image from X ⊂ R N where N is large (we think of x as an √ N × √ N discrete image).

We assume that the map from x ∈ X to y ∈ R M is injective so that it is invertible on its range, and that there exists an L-Lipschitz (generally non-linear) inverse G, DISPLAYFORM0 In order for the injectivity assumption to be reasonable, we assume that X is a low-dimensional manifold embedded in R N of dimension at most M , where M is the number of measurements.

Since we are in finite dimension, injectivity implies the existence of L BID45 ).

Due to ill-posedness, L is typically large.

Consider now the map from the data y to a projection of the model x into some K-dimensional subspace S, where K N .

Note that this map exists by construction (since A is injective on X ), and that it must be non-linear.

To see this, note that the only consistent 2 linear map acting on y is an oblique, rather than an orthogonal projection on S (cf.

Section 2.4 in BID48 ).

We explain this in more detail in Appendix A.Denote the projection by P S x and assume S ⊂ R N is chosen uniformly at random.

3 We want to evaluate the expected Lipschitz constant of the map from y to P S x, noting that it can be written as P S • G: DISPLAYFORM1 where the first inequality is Jensen's inequality, and the second one follows from DISPLAYFORM2 and the observation that E P S P S = K N I N .

In other words, random projections reduce the Lipschitz constant by a factor of K/N on average.

Since learning requires O(L K ) samples, this allows us to work with exponentially fewer samples and makes the learning task easier.

Conversely, given a fixed training dataset, it gives more accurate estimates.

The above example uses unstructured random subspaces.

In many inverse problems, such as inverse scattering BID4 ; Di Cristo and Rondi FORMULA8 ), a judicious choice of subspace family can give exponential improvements in Lipschitz stability.

Particularly, it is favorable to use piecewiseconstant images: x = K k=1 x k χ k , with χ k being indicator functions of some domain subset.

Motivated by this observation, we use piecewise-constant subspaces over random Delaunay triangle meshes.

The Delaunay triangulations enjoy a number of desirable learning-theoretic properties.

For function learning it was shown that given a set of vertices, piecewise linear functions on Delaunay triangulations achieve the smallest sup-norm error among all triangulations (Omohundro (1989)).We sample Λ sets of points in the image domain from a uniform-density Poisson process and construct Λ (discrete) Delaunay triangulations with those points as vertices.

Let S = {S λ | 1 ≤ λ ≤ Λ} be the collection of Λ subspaces of piecewise-constant functions on these triangulations.

Let further G λ be the map from y to the projection of the model into subspace S λ , G λ y = P S λ x. Instead of learning the "hard" inverse mapping G, we propose to learn an ensemble of simpler mappings {G λ } Λ λ=1 .

We approximate each G λ by a convolutional neural network, Γ θ(λ) ( y) : R N → R N , parameterized by a set of trained weights θ(λ).

Similar to Jin et al. FORMULA3 , we do not use the measured data y ∈ R M directly as this would require the network to first learn to map y back to the image domain; we rather warm-start the reconstruction by a non-negative least squares reconstruction, y ∈ R N , computed from y. The weights are chosen by minimizing empirical risk: DISPLAYFORM0 where DISPLAYFORM1 is a set of J training models and non-negative least squares measurements.

By learning projections onto random subspaces, we transform our original problem into that of estimating DISPLAYFORM0 .

To see how this can be done, ascribe to the columns of B λ ∈ 2 Consistent meaning that if x already lives in S, then the map should return x.3 One way to construct the corresponding projection matrix is as P S = W W † , where W ∈ R N ×K is a matrix with standard iid Gaussian entries.

N ×K a natural orthogonal basis for the subspace S λ , B λ = [χ λ,1 , . . .

, χ λ,K ], with χ λ,k being the indicator function of the kth triangle in mesh λ.

Denote by q λ def = q λ (y) the mapping from the data y to an estimate of the expansion coefficients of x in the basis for S λ : DISPLAYFORM0 . .

, q Λ ∈ R KΛ ; then we can estimate x using the following reformulated problem: DISPLAYFORM1 and the corresponding regularized reconstruction: DISPLAYFORM2 with ϕ(x) chosen as the TV-seminorm x TV .

The regularization is not essential.

As we show experimentally, if KΛ is sufficiently large, ϕ(x) is not required.

Note that solving the original problem directly using x TV regularizer fails to recover the structure of the model (Figure 1 ).

Since the true inverse map G has a large Lipschitz constant, it would seem reasonable that as the number of mesh subspaces Λ grows large (and their direct sum approaches the whole ambient space R N ), the Lipschitz properties of G should deteriorate as well.

Denote the unregularized inverse mapping in y → x (2) by G. Then we have the following estimate: DISPLAYFORM0 with σ min (B) the smallest (non-zero) singular value of B and L K the Lipschitz constant of the stable projection mappings q λ .

Indeed, we observe empirically that σ min (B) −1 grows large as the number of subspaces increases which reflects the fact that although individual projections are easier to learn, the full-resolution reconstruction remains ill-posed.

Estimates of individual subspace projections give correct local information.

They convert possibly non-local measurements (e.g. integrals along curves in tomography) into local ones.

The key is that these local averages (subspace projection coefficients) can be estimated accurately (see Section 4).To further illustrate what we mean by correct local information, consider a simple numerical experiment with our reformulated problem, q = B T x, where x is an all-zero image with a few pixels "on".

For the sake of clarity we assume the coefficients q are perfect.

Recall that B is a block matrix comprising Λ subspace bases stacked side by side.

It is a random matrix because the subspaces are generated at random, and therefore the reconstruction x = (B ) † q is also random.

We approximate E x by simulating a large number of Λ-tuples of meshes and averaging the obtained reconstructions.

Results are shown in FIG1 for different numbers of triangles per subspace, K, and subspaces per reconstruction, Λ. As Λ or K increase, the expected reconstruction becomes increasingly localized around non-zero pixels.

The following proposition (proved in Appendix B) tells us that this phenomenon can be modeled by convolution.

4 Proposition 1.

Let x be the solution to q = B x given as (B ) † q. Then there exists a kernel κ(u), with u a discrete index, such that E x = x * κ.

Furthermore, κ(u) is isotropic.

While FIG1 suggests that more triangles are better, we note that this increases the subspace dimension which makes getting correct projection estimates harder.

Instead we choose to stack more meshes with a smaller number of triangles.

Intuitively, since every triangle average depends on many measurements, estimating each average is more robust to measurement corruptions as evidenced in Section 4.

Accurate estimates of local averages enable us to recover the geometric structure while being more robust to data errors.

4 NUMERICAL RESULTS

To demonstrate our method's benefits we consider linearized traveltime tomography BID20 ; BID8 ), but we note that the method applies to any inverse problem with scarce data.

In traveltime tomography, we measure N 2 wave travel times between N sensors as in FIG2 .

Travel times depend on the medium property called slowness (inverse of speed) and the task is to reconstruct the spatial slowness map.

Image intensities are a proxy for slowness maps-the lower the image intensity the higher the slowness.

In the straight-ray approximation, the problem data is modeled as integral along line segments: DISPLAYFORM0 where x : R 2 → R + is the continuous slowness map and s i , s j are sensor locations.

In our experiments, we use a 128 × 128 pixel grid with 25 sensors (300 measurements) placed uniformly in an inscribed circle, and corrupt the measurements with zero-mean iid Gaussian noise.

We generate random Delaunay meshes each with 50 triangles.

The corresponding projector matrices compute average intensity over triangles to yield a piecewise constant approximation P S λ x of x. We test two distinct architectures: (i) ProjNet, tasked with estimating the projection into a single subspace; and (ii) SubNet, tasked with estimating the projection over multiple subspaces.

The ProjNet architecture is inspired by the FBPConvNet BID23 ) and the U-Net BID41 ) as shown in Figure 11a in the appendix.

Crucially, we constrain the network output to live in S λ by fixing the last layer of the network to be a projector, P S λ (Figure 11a) .

A similar trick in a different context was proposed in BID43 ).We combine projection estimates from many ProjNets by regularized linear least-squares (2) to get the reconstructed model (cf.

Figure 2) with the regularization parameter λ determined on five held-out images.

A drawback of this approach is that a separate ProjNet must be trained for each subspace.

This motivates the SubNet (shown in Figure 11b ).

Each input to SubNet is the concatenation of a non-negative least squares reconstruction and 50 basis functions, one for each triangle forming a 51-channel input.

This approach scales to any number of subspaces which allows us to get visually smoother reconstructions without any further regularization as in (2).

On the other hand, the projections are less precise which can lead to slightly degraded performance.

As a quantitative figure of merit we use the signal-to-noise ratio (SNR).

The input SNR is defined as 10 log 10 (σ 2 signal /σ 2 noise ) where σ 2 signal and σ 2 noise are the signal and noise variance; the output SNR is defined as sup a,b 20 log 10 ( x 2 / x − ax − b 2 ) with x the ground truth andx the reconstruction.130 ProjNets are trained for 130 different meshes with measurements at various SNRs.

Similarly, a single SubNet is trained with 350 different meshes and the same noise levels.

We compare the ProjNet and SubNet reconstructions with a direct U-net baseline convolutional neural network that reconstructs images from their non-negative least squares reconstructions.

The direct baseline has the same architecture as SubNet except the input is a single channel non-negative least squares reconstruction like in ProjNet and the output is the target reconstruction.

Such an architecture was proposed by BID23 ) and is used as a baseline in recent learning-based inverse problem works BID29 ; Ye et al. FORMULA3 ) and is inspiring other architectures for inverse problems BID2 ).

We pick the best performing baseline network from multiple networks which have a comparable number of trainable parameters to SubNet.

We simulate the lack of training data by testing on a dataset that is different than that used for training.

Robustness to corruption To demonstrate that our method is robust against arbitrary assumptions made at training time, we consider two experiments.

First, we corrupt the data with zero-mean iid Gaussian noise and reconstruct with networks trained at different input noise levels.

In FIG3 and Table 1 , we summarize the results with reconstructions of geo images taken from the BP2004 dataset 6 and x-ray images of metal castings BID31 ).

The direct baseline and SubNet are trained on a set of 20,000 images from the arbitrarily chosen LSUN bridges dataset BID53 ) and tested with the geophysics and x-ray images.

ProjNets are trained with 10,000 images from the LSUN dataset.

Our method reports better SNRs compared to the baseline.

We note that direct reconstruction is unstable when trained on clean and tested on noisy measurements as it often hallucinates details that are artifacts of the training data.

For applications in geophysics it is important that our method correctly captures the shape of the cavities unlike the direct inversion which can produce sharp but wrong geometries (see outlines in FIG3 ).

with Gaussian noise FIG3 ) the direct method completely fails to recover coarse geometry in all test cases.

In our entire test dataset of 102 x-ray images there is not a single example where the direct network captures a geometric feature that our method misses.

This demonstrates the strengths of our approach.

For more examples of x-ray images please see Appendix E. FIG4 illustrates the influence of the training data on reconstructions.

Training with LSUN, CelebA BID27 ) and a synthetic dataset of random overlapping shapes (see FIG3 in Appendix for examples) all give comparable reconstructions-a desirable property in applications where real ground truth is unavailable.

We complement our results with reconstructions of checkerboard phantoms (standard resolution tests) and x-rays of metal castings in Figure 7 .

We note that in addition to better SNR, our method produces more accurate geometry estimates, as per the annotations in the figure.

We proposed a new approach to regularize ill-posed inverse problems in imaging, the key idea being to decompose an unstable inverse mapping into a collection of stable mappings which only estimate Figure 7 : Reconstructions on checkerboards and x-rays with 10dB measurement SNR tested on 10dB trained networks.

Red annotations highlight where the direct net fails to reconstruct correct geometry.

low-dimensional projections of the model.

By using piecewise-constant Delaunay subspaces, we showed that the projections can indeed be accurately estimated.

Combining the projections leads to a deconvolution-like problem.

Compared to directly learning the inverse map, our method is more robust against noise and corruptions.

We also showed that regularizing via projections allows our method to generalize across training datasets.

Our reconstructions are better both quantitatively in terms of SNR and qualitatively in the sense that they estimate correct geometric features even when measurements are corrupted in ways not seen at training time.

Future work involves getting precise estimates of Lipschitz constants for various inverse problems, regularizing the reformulated problem using modern regularizers BID46 ), studying extensions to non-linear problems and developing concentration bounds for the equivalent convolution kernel.

This work utilizes resources supported by the National Science Foundation's Major Research Instrumentation program, grant #1725729, as well as the University of Illinois at Urbana-Champaign.

We gratefully acknowledge the support of NVIDIA Corporation with the donation of one of the GPUs used for this research.

We explain the need for non-linear operators even in the absence of noise with reference to FIG5 .

Projecting x into a given known subspace is a simple linear operation, so it may not be a priori clear why we use non-linear neural networks to estimate the projections.

Alas, we do not know x and only have access to y. Suppose that there exists a linear operator (a matrix) F ∈ R N ×M which acts on y and computes the projection of x on S λ .

A natural requirement on F is consistency: if x already lives in S λ , then we would like to have F Ax = x. This implies that for any x, not necessarily in S λ , we require F AF Ax = F Ax which implies that F A = (F A)2 is an idempotent operator.

Letting the columns of B λ be a basis for S λ , it is easy to see that the least squares minimizer for F is B λ (AB λ ) † .

However, because R(F ) = S λ = R(A * ) (A * is the adjoint of A, simply a transpose for real matrices), in general it will not hold that (F A) * = F A. Thus, F A is an oblique, rather than orthogonal projection into S. In FIG5 this corresponds to the point P oblique S λ x which can be arbitrarily far from the orthogonal projection P ortho S λ x. The nullspace of the oblique projection is precisely N (A) = R(A * ) ⊥ .Thus consistent linear operators can at best yield oblique projections which can be far from the orthogonal one.

One could also see this geometrically from FIG5 .

As the angle between S λ and R(A * ) increases to π/2 the oblique projection point travels to infinity (note that the oblique projection always happens along the nullspace of A, which is the line orthogonal to P R(A * ) .

Since our subspaces are chosen at random, in general they are not aligned with R(A * ).

The only subspace on which we can linearly compute an orthogonal projection from y is R(A * ); this is given by the Moore-Penrose pseudoinverse.

Therefore, to get the orthogonal projection onto random subspaces, we must use non-linear operators.

More generally, for any other ad hoc linear reconstruction operator W , W y = W Ax always lives in the column space of W A which is a subspace whose dimension is at most the number of rows of A. However, we do not have any linear subspace model for x.

As shown in the right half of FIG5 , as soon as A is injective on X , the existence of this non-linear map is guaranteed by construction: since y determines x, it also determines P S λ x.

We show the results of numerical experiments in Figures 9 and 10 which further illustrate the performance difference between linear oblique projectors and our non-linear learned operator when estimating the projection of an image into a random subspace.

We refer the reader to the captions below each figure for more details.

Figure 10: We try hard to get the best reconstruction from the linear approach.

SNRs are indicated in the bottom-left of each reconstruction.

In the linear approach, coefficients are obtained using the linear oblique projection method.

Once coefficients are obtained, they are non-linearly reconstructed according to (2).

Both linear approach reconstructions use the box-constraint (BC) mentioned in (2).

For the 130 subspace reconstruction total-variation (TV) regularization is also used.

Therefore, once the coefficients are obtained using the linear approach, the reconstruction of the final image is done in an identical manner as ProjNet for 130 subspaces and SubNet for 350 subspaces.

To give the linear approach the best chance we also optimized hyperparameters such as the regularization parameter to give the highest SNR.Using the definition of the inner product and rearranging, we get DISPLAYFORM0 .

Now, the probability distribution of triangles around any point u is both shift-and rotation-invariant because a Poisson process in the plane is shift-and rotation-invariant.

It follows that E κ(u, v) = κ( u − v ) for some κ, meaning that DISPLAYFORM1 which is a convolution of the original model with a rotationally invariant (isotropic) kernel.

Figure 11 explains the network architecture used for ProjNet and SubNet.

The network consists of a sequence of downsampling layers followed by upsampling layers, with skip connections BID19 a) ) between the downsampling and upsampling layers.

Each ProjNet output is constrained to a single subspace by applying a subspace projection operator, P S λ .

We train 130 such networks and reconstruct from the projection estimate using (2).

SubNet is a single network that is trained over multiple subspaces.

To do this, we change its input to be [ y B λ ].

Moreover, we apply the same projection operator as ProjNet to the output of the SubNet.

Each SubNet is trained to give projection estimates over 350 random subspaces.

This approach allows us to scale to any number of subspaces without training new networks for each.

Moreover, this allows us to build an over-constrained system q = Bx to solve.

Even though SubNet has almost as many parameters as the direct net, reconstructing via the projection estimates allows SubNet to get higher SNR and more importantly, get better estimates of the coarse geometry than the direct inversion.

All networks are trained with the Adam optimizer.

projection into input subspace Figure 11 : a) ProjNet architecture; b) SubNet architecture.

In both cases, the input is a non-negative least squares reconstruction and the network is trained to reconstruct a projection into one subspace.

In SubNet, the subspace basis is concatenated to the non-negative least squares reconstruction.

We showcase more reconstructions on actual geophysics images taken from the BP2004 dataset in Figure 12 .

Note that all networks were trained on the LSUN bridges dataset.

We show additional reconstructions for the largest corruption case, p = 1 8 , for x-ray images ( FIG1 ) and geo images FIG2 .

Our method consistently has better SNR.

More importantly we note that there is not a single instance where the direct reconstruction gets a feature that our methods do not.

In a majority of instances, the direct network misses a feature of the image.

This is highly undesirable in settings such as geophysical imaging.

The shapes dataset was generated using random ellipses, circle and rectangle patches.

See FIG3 for examples.

This dataset was used in FIG4 .

In Section 4 we train multiple ProjNets, each focusing on a different low-dimensional subspace.

Here we train an ensemble of direct networks where each network is as described in Section 4.1.1 and evaluate the robustness of a method where the outputs of these networks are averaged to give a final reconstruction.

Once again, we consider scenarios where the model is trained with data at a particular noise level and then tested with data at a different noise level and with erasures that were unseen during training time.

We show that our proposed method is more robust to changes in the test scenario.

In FIG4 , we consider the erasure model with p = 1 8 (described in FIG3 ).

9 out of 10 randomly chosen direct network reconstructions fail to capture the key structure of the original image under this corruption mechanism.

In TAB8 , we summarize this with the SNRs of reconstructions from the erasure corruption mechanism.

In that table we also report SNRs when reconstructing from measurements at different noise levels.

The ensemble of direct networks performs well when the training and test data have the same measurement noise level.

However, our method is more robust to changes in the test noise level.

This further illustrates that direct networks are highly tuned to the training scenario and therefore not as stable as our proposed method (cf.

Section 3).

Original FIG4 : Reconstructions of the original image from 10 individually trained direct inversion networks for 10dB noise under the p = 1 8 erasure corruptions model (described in FIG3 ).

9 out of the 10 reconstructions fail to capture the key structure of the original image.

Single

<|TLDR|>

@highlight

We solve ill-posed inverse problems with scarce ground truth examples by estimating an ensemble of random projections of the model instead of the model itself.