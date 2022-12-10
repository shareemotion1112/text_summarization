Convolutional Neural Networks (CNNs) have become the method of choice for learning problems involving 2D planar images.

However, a number of problems of recent interest have created a demand for models that can analyze spherical images.

Examples include omnidirectional vision for drones, robots, and autonomous cars, molecular regression problems, and global weather and climate modelling.

A naive application of convolutional networks to a planar projection of the spherical signal is destined to fail, because the space-varying distortions introduced by such a projection will make translational weight sharing ineffective.



In this paper we introduce the building blocks for constructing spherical CNNs.

We propose a definition for the spherical cross-correlation that is both expressive and rotation-equivariant.

The spherical correlation satisfies a generalized Fourier theorem, which allows us to compute it efficiently using a generalized (non-commutative) Fast Fourier Transform (FFT) algorithm.

We demonstrate the computational efficiency, numerical accuracy, and effectiveness of spherical CNNs applied to 3D model recognition and atomization energy regression.

Figure 1: Any planar projection of a spherical signal will result in distortions.

Rotation of a spherical signal cannot be emulated by translation of its planar projection.

Convolutional networks are able to detect local patterns regardless of their position in the image.

Like patterns in a planar image, patterns on the sphere can move around, but in this case the "move" is a 3D rotation instead of a translation.

In analogy to the planar CNN, we would like to build a network that can detect patterns regardless of how they are rotated over the sphere.

As shown in Figure 1 , there is no good way to use translational convolution or cross-correlation 1 to analyze spherical signals.

The most obvious approach, then, is to change the definition of crosscorrelation by replacing filter translations by rotations.

Doing so, we run into a subtle but important difference between the plane and the sphere: whereas the space of moves for the plane (2D translations) is itself isomorphic to the plane, the space of moves for the sphere (3D rotations) is a different, three-dimensional manifold called SO(3) 2 .

It follows that the result of a spherical correlation (the output feature map) is to be considered a signal on SO(3), not a signal on the sphere, S 2 .

For this reason, we deploy SO(3) group correlation in the higher layers of a spherical CNN BID4 .The implementation of a spherical CNN (S 2 -CNN) involves two major challenges.

Whereas a square grid of pixels has discrete translation symmetries, no perfectly symmetrical grids for the sphere exist.

This means that there is no simple way to define the rotation of a spherical filter by one pixel.

Instead, in order to rotate a filter we would need to perform some kind of interpolation.

The other challenge is computational efficiency; SO(3) is a three-dimensional manifold, so a naive implementation of SO(3) correlation is O(n 6 ).We address both of these problems using techniques from non-commutative harmonic analysis BID3 BID11 .

This field presents us with a far-reaching generalization of the Fourier transform, which is applicable to signals on the sphere as well as the rotation group.

It is known that the SO(3) correlation satisfies a Fourier theorem with respect to the SO(3) Fourier transform, and the same is true for our definition of S 2 correlation.

Hence, the S 2 and SO(3) correlation can be implemented efficiently using generalized FFT algorithms.

Because we are the first to use cross-correlation on a continuous group inside a multi-layer neural network, we rigorously evaluate the degree to which the mathematical properties predicted by the continuous theory hold in practice for our discretized implementation.

Furthermore, we demonstrate the utility of spherical CNNs for rotation invariant classification and regression problems by experiments on three datasets.

First, we show that spherical CNNs are much better at rotation invariant classification of Spherical MNIST images than planar CNNs.

Second, we use the CNN for classifying 3D shapes.

In a third experiment we use the model for molecular energy regression, an important problem in computational chemistry.

The main contributions of this work are the following:1.

The theory of spherical CNNs.2.

The first automatically differentiable implementation of the generalized Fourier transform for S 2 and SO(3).

Our PyTorch code is easy to use, fast, and memory efficient.3.

The first empirical support for the utility of spherical CNNs for rotation-invariant learning problems.

It is well understood that the power of CNNs stems in large part from their ability to exploit (translational) symmetries though a combination of weight sharing and translation equivariance.

It thus becomes natural to consider generalizations that exploit larger groups of symmetries, and indeed this has been the subject of several recent papers by BID12 BID24 ; BID7 ; BID4 ; BID28 BID38 ; BID14 ; .

With the exception of SO(2)-steerable networks BID36 BID35 , these networks are all limited to discrete groups, such as discrete rotations acting on planar images or permutations acting on point clouds.

Other very recent work is concerned with the analysis of spherical images, but does not define an equivariant architecture BID32 BID1 .

Our work is the first to achieve equivariance to a continuous, non-commutative group (SO(3)), and the first to use the generalized Fourier transform for fast group correlation.

A preliminary version of this work appeared as .To efficiently perform cross-correlations on the sphere and rotation group, we use generalized FFT algorithms.

Generalized Fourier analysis, sometimes called abstract-or noncommutative harmonic analysis, has a long history in mathematics and many books have been written on the subject BID33 BID34 BID11 .

For a good engineering-oriented treatment which covers generalized FFT algorithms, see BID3 .

Other important works include BID10 BID15 BID25 BID18 BID9 BID20 BID29 BID16 BID26 BID19 BID13 .

We will explain the S 2 and SO(3) correlation by analogy to the classical planar Z 2 correlation.

The planar correlation can be understood as follows:The value of the output feature map at translation x ∈ Z 2 is computed as an inner product between the input feature map and a filter, shifted by x.

Similarly, the spherical correlation can be understood as follows:The value of the output feature map evaluated at rotation R ∈ SO(3) is computed as an inner product between the input feature map and a filter, rotated by R.Because the output feature map is indexed by a rotation, it is modelled as a function on SO(3).

We will discuss this issue in more detail shortly.

The above definition refers to various concepts that we have not yet defined mathematically.

In what follows, we will go through the required concepts one by one and provide a precise definition.

Our goal for this section is only to present a mathematical model of spherical CNNs.

Generalized Fourier theory and implementation details will be treated later.

The Unit Sphere S 2 can be defined as the set of points x ∈ R 3 with norm 1.

It is a two-dimensional manifold, which can be parameterized by spherical coordinates α ∈ [0, 2π] and β ∈ [0, π].

We model spherical images and filters as continuous functions f : S 2 → R K , where K is the number of channels.

Rotations The set of rotations in three dimensions is called SO(3), the "special orthogonal group".

Rotations can be represented by 3 × 3 matrices that preserve distance (i.e. ||Rx|| = ||x||) and orientation (det(R) = +1).

If we represent points on the sphere as 3D unit vectors x, we can perform a rotation using the matrix-vector product Rx.

The rotation group SO(3) is a three-dimensional manifold, and can be parameterized by ZYZ-Euler angles α ∈ [0, 2π], β ∈ [0, π], and γ ∈ [0, 2π].

In order to define the spherical correlation, we need to know not only how to rotate points x ∈ S 2 but also how to rotate filters (i.e. functions) on the sphere.

To this end, we introduce the rotation operator L R that takes a function f and produces a rotated function L R f by composing f with the rotation R −1 : DISPLAYFORM0 Inner products The inner product on the vector space of spherical signals is defined as: DISPLAYFORM1 The integration measure dx denotes the standard rotation invariant integration measure on the sphere, which can be expressed as dα sin(β)dβ/4π in spherical coordinates (see Appendix A).

The invariance of the measure ensures that S 2 f (Rx)dx = S 2 f (x)dx, for any rotation R ∈ SO(3).

That is, the volume under a spherical heightmap does not change when rotated.

Using this fact, we can show that DISPLAYFORM2 Spherical Correlation With these ingredients in place, we are now ready to state mathematically what was stated in words before.

For spherical signals f and ψ, we define the correlation as: DISPLAYFORM3 As mentioned before, the output of the spherical correlation is a function on SO(3).

This is perhaps somewhat counterintuitive, and indeed the conventional definition of spherical convolution gives as output a function on the sphere.

However, as shown in Appendix B, the conventional definition effectively restricts the filter to be circularly symmetric about the Z axis, which would greatly limit the expressive capacity of the network.

We defined the rotation operator L R for spherical signals (eq. 1), and used it to define spherical cross-correlation (eq. 4).

To define the SO(3) correlation, we need to generalize the rotation operator so that it can act on signals defined on SO(3).

As we will show, naively reusing eq. 1 is the way to go.

That is, for f : SO(3) → R K , and R, Q ∈ SO(3): DISPLAYFORM0 Note that while the argument R −1 x in Eq. 1 denotes the rotation of x ∈ S 2 by R −1 ∈ SO(3), the analogous term R −1 Q in Eq. 5 denotes to the composition of rotations (i.e. matrix multiplication).Rotation Group Correlation Using the same analogy as before, we can define the correlation of two signals on the rotation group, f, ψ : SO(3) → R K , as follows: DISPLAYFORM1 The integration measure dQ is the invariant measure on SO(3), which may be expressed in ZYZ-Euler angles as dα sin(β)dβdγ/(8π 2 ) (see Appendix A).Equivariance As we have seen, correlation is defined in terms of the rotation operator L R .

This operator acts naturally on the input space of the network, but what justification do we have for using it in the second layer and beyond?The justification is provided by an important property, shared by all kinds of convolution and correlation, called equivariance.

DISPLAYFORM2 Using the definition of correlation and the unitarity of L R , showing equivariance is a one liner: DISPLAYFORM3 The derivation is valid for spherical correlation as well as rotation group correlation.

It is well known that correlations and convolutions can be computed efficiently using the Fast Fourier Transform (FFT).

This is a result of the Fourier theorem, which states that f * ψ =f ·ψ.

Since the FFT can be computed in O(n log n) time and the product · has linear complexity, implementing the correlation using FFTs is asymptotically faster than the naive O(n 2 ) spatial implementation.

For functions on the sphere and rotation group, there is an analogous transform, which we will refer to as the generalized Fourier transform (GFT) and a corresponding fast algorithm (GFFT).

This transform finds it roots in the representation theory of groups, but due to space constraints we will not go into details here and instead refer the interested reader to BID33 and BID11 .Conceptually, the GFT is nothing more than the linear projection of a function onto a set of orthogonal basis functions called "matrix element of irreducible unitary representations".

For the circle (S 1 ) or line (R), these are the familiar complex exponentials exp(inθ).

For SO FORMULA2 FORMULA2 ) by X and the corresponding basis functions by U l (which is either vector-valued (Y l ) or matrix-valued (D l )), we can write the GFT of a function f : X → R aŝ DISPLAYFORM0 3 Technically, S 2 is not a group and therefore does not have irreducible representations, but it is a quotient of groups SO(3)/ SO(2) and we have the relation DISPLAYFORM1 This integral can be computed efficiently using a GFFT algorithm (see Section 4.1).The inverse SO(3) Fourier transform is defined as: DISPLAYFORM2 and similarly for S 2 .

The maximum frequency b is known as the bandwidth, and is related to the resolution of the spatial grid BID16 .Using the well-known (in fact, defining) property of the Wigner D-functions that Figure 2: Spherical correlation in the spectrum.

The signal f and the locally-supported filter ψ are Fourier transformed, block-wise tensored, summed over input channels, and finally inverse transformed.

Note that because the filter is locally supported, it is faster to use a matrix multiplication (DFT) than an FFT algorithm for it.

We parameterize the sphere using spherical coordinates α, β, and SO(3) with ZYZ-Euler angles α, β, γ.

DISPLAYFORM3

Here we sketch the implementation of GFFTs.

For details, see BID16 .The input of the SO(3) FFT is a spatial signal f on SO(3), sampled on a discrete grid and stored as a 3D array.

The axes correspond to the ZYZ-Euler angles α, β, γ.

The first step of the SO(3)-FFT is to perform a standard 2D translational FFT over the α and γ axes.

The FFT'ed axes correspond to the m, n axes of the result.

The second and last step is a linear contraction of the β axis of the FFT'ed array with a precomputed array of samples from the Wigner- DISPLAYFORM0 Because the shape of d l depends on l (it is (2l + 1) × (2l + 1)), this linear contraction is implemented as a custom GPU kernel.

The output is a set of Fourier coefficientsf l mn for l ≥ n, m ≥ −l and l = 0, . . .

, L max .The algorithm for the S 2 -FFTs is very similar, only in this case we FFT over the α axis only, and do a linear contraction with precomputed Legendre functions over the β axis.

Our code is available at https://github.com/jonas-koehler/s2cnn.

In a first sequence of experiments, we evaluate the numerical stability and accuracy of our algorithm.

In a second sequence of experiments, we showcase that the new cross-correlation layers we have In this paper we have presented the first instance of a group equivariant CNN for a continuous, non-commutative group.

In the discrete case, one can prove that the network is exactly equivariant, but although we can prove DISPLAYFORM0 for continuous functions f and ψ on the sphere or rotation group, this is not exactly true for the discretized version that we actually compute.

Hence, it is reasonable to ask if there are any significant discretization artifacts and whether they affect the equivariance properties of the network.

If equivariance can not be maintained for many layers, one may expect the weight sharing scheme to become much less effective.

We first tested the equivariance of the SO(3) correlation at various resolutions b.

We do this by first sampling n = 500 random rotations R i as well as n feature maps f i with K = 10 channels.

Then we DISPLAYFORM1 , where Φ is a composition of SO(3) correlation layers with randomly initialized filters.

In case of perfect equivariance, we expect this quantity to be zero.

The results ( FIG2 ), show that although the approximation error ∆ grows with the resolution and the number of layers, it stays manageable for the range of resolutions of interest.

We repeat the experiment with ReLU activation function after each correlation operation.

As shown in figure 3 (bottom), the error is higher but stays flat.

This indicates that the error is not due to the network layers, but due to the feature map rotation, which is exact only for bandlimited functions.

In this experiment we evaluate the generalization performance with respect to rotations of the input.

For testing we propose a version MNIST dataset projected on the sphere (see FIG3 ).

We created two instances of this dataset: one in which each digit is projected on the northern hemisphere and one in which each projected digit is additionally randomly rotated.

Architecture and Hyperparameters As a baseline model, we use a simple CNN with layers conv-ReLU-conv-ReLU-FC-softmax, with filters of size 5 × 5, k = 32, 64, 10 channels, and stride 3 in both layers (≈ 68K parameters).

We compare to a spherical CNN with layers S 2 conv-ReLU-SO(3)conv-ReLU-FC-softmax, bandwidth b = 30, 10, 6 and k = 20, 40, 10 channels (≈ 58K parameters).Results We trained each model on the nonrotated (NR) and the rotated (R) training set and evaluated it on the non-rotated and rotated test set.

See table 1.

While the planar CNN achieves high accuracy in the NR / NR regime, its performance in the R / R regime is much worse, while the spherical CNN is unaffected.

When trained on the Table 1 : Test accuracy for the networks evaluated on the spherical MNIST dataset.

Here R = rotated, NR = non-rotated and X / Y denotes, that the network was trained on X and evaluated on Y.

Next, we applied S 2 CNN to 3D shape classification.

The SHREC17 task BID31 contains 51300 3D models taken from the ShapeNet dataset BID2 which have to be classified into 55 common categories (tables, airplanes, persons, etc.) .

There is a consistently aligned regular dataset and a version in which all models are randomly perturbed by rotations.

We concentrate on the latter to test the quality of our rotation equivariant representations learned by S 2 CNN.Representation We project the 3D meshes onto an enclosing sphere using a straightforward ray casting scheme (see FIG5 ).

For each point on the sphere we send a ray towards the origin and collect 3 types of information from the intersection: ray length and cos / sin of the surface angle.

We further augment this information with ray casting information for the convex hull of the model, which in total gives us 6 channels for the signal.

This signal is discretized using a Driscoll-Healy grid BID10 with bandwidth b = 128.

Ignoring non-convexity of surfaces we assume this projection captures enough information of the shape to be useful for the recognition task.

Architecture and Hyperparameters Our network consists of an initial S 2 conv-BN-ReLU block followed by two SO(3)conv-BN-ReLU blocks.

The resulting filters are pooled using a max pooling layer followed by a last batch normalization and then fed into a linear layer for the final classification.

It is important to note that the the max pooling happens over the group SO(3): if f k is the k-th filter in the final layer (a function on SO(3)) the result of the pooling is max x∈SO(3) f k (x).

We used 50, 70, and 350 features for the S 2 and the two SO(3) layers, respectively.

Further, in each layer we reduce the resolution b, from 128, 32, 22 to 7 in the final layer.

Each filter kernel ψ on SO(3) has non-local support, where ψ(α, β, γ) = 0 iff β = π 2 and γ = 0 and the number of points of the discretization is proportional to the bandwidth in each layer.

The final network contains ≈ 1.4M parameters, takes 8GB of memory at batch size 16, and takes 50 hours to train.

Results We evaluated our trained model using the official metrics and compared to the top three competitors in each category (see TAB1 for results).

Except for precision and F1@N, in which our model ranks third, it is the runner up on each other metric.

The main competitors, Tatsuma_ReVGG and Furuya_DLAN use input representations and network architectures that are highly specialized to the SHREC17 task.

Given the rather task agnostic architecture of our model and the lossy input representation we use, we interpret our models performance as strong empirical support for the effectiveness of Spherical CNNs.

Finally, we apply S 2 CNN on molecular energy regression.

In the QM7 task BID0 BID30 ) the atomization energy of molecules has to be predicted from geometry and charges.

Molecules contain up to N = 23 atoms of T = 5 types (H, C, N, O, S).

They are given as a list of positions p i and charges z i for each atom i. BID30 propose a rotation and translation invariant representation of molecules by defining the Coulomb matrix C ∈ R N ×N (CM).

For each pair of atoms i = j they set C ij = (z i z j )/(|p i − p j |) and C ii = 0.5z

i .

Diagonal elements encode the atomic energy by nuclear charge, while other elements encode Coulomb repulsion between atoms.

This representation is not permutation invariant.

To this end BID30 propose a distance measure between Coulomb matrices used within Gaussian kernels whereas Montavon et al. (2012) propose sorting C or random sampling index permutations.

Representation as a spherical signal We utilize spherical symmetries in the geometry by defining a sphere S i around around p i for each atom i.

The radius is kept uniform across atoms and molecules and chosen minimal such that no intersections among spheres in the training set happen.

Generalizing the Coulomb matrix approach we define for each possible z and for each point x on S i potential functions U z (x) = j =i,zj =z zi·z |x−pi| producing a T channel spherical signal for each atom in the molecule (see FIG6 ).

This representation is invariant with respect to translations and equivariant with respect to rotations.

However, it is still not permutation invariant.

The signal is discretized using a Driscoll-Healy BID10 grid with bandwidth b = 10 representing the molecule as a sparse N × T × 2b × 2b tensor.

We use a deep ResNet style S 2 CNN.

Each ResNet block is made of S 2 /SO(3)conv-BN-ReLU-SO(3)conv-BN after which the input is added to the result.

We share weights among atoms making filters permutation invariant, by pushing the atom dimension into the batch dimension.

In each layer we downsample the bandwidth, while increasing the number of features F .

After integrating the signal over SO(3) each molecule becomes a N × F tensor.

For permutation invariance over atoms we follow BID37 and embed each resulting feature vector of an atom into a latent space using a MLP φ.

Then we sum these latent representations over the atom dimension and get our final regression value for the molecule by mapping with another MLP ψ.

Both φ and ψ are jointly optimized.

Training a simple MLP only on the 5 frequencies of atom types in a molecule already gives a RMSE of ∼ 19.

Thus, we train the S 2 CNN on the residual only, which improved convergence speed and stability over direct training.

The final architecture is sketched in table 3.

It has about 1.4M parameters, consumes 7GB of memory at batch size 20, and takes 3 hours to train.

Results We evaluate by RMSE and compare our results to Montavon et al. (2012) and BID27 (see table 3 ).

Our learned representation outperforms all kernel-based approaches and a MLP trained on sorted Coulomb matrices.

Superior performance could only be achieved for an MLP trained on randomly permuted Coulomb matrices.

However, sufficient sampling of random permutations grows exponentially with N , so this method is unlikely to scale to large molecules.

In this paper we have presented the theory of Spherical CNNs and evaluated them on two important learning problems.

We have defined S 2 and SO(3) cross-correlations, analyzed their properties, and implemented a Generalized FFT-based correlation algorithm.

Our numerical results confirm the stability and accuracy of this algorithm, even for deep networks.

Furthermore, we have shown that Spherical CNNs can effectively generalize across rotations, and achieve near state-of-the-art results on competitive 3D Model Recognition and Molecular Energy Regression challenges, without excessive feature engineering and task-tuning.

For intrinsically volumetric tasks like 3D model recognition, we believe that further improvements can be attained by generalizing further beyond SO(3) to the roto-translation group SE(3).

The development of Spherical CNNs is an important first step in this direction.

Another interesting generalization is the development of a Steerable CNN for the sphere , which would make it possible to analyze vector fields such as global wind directions, as well as other sections of vector bundles over the sphere.

Perhaps the most exciting future application of the Spherical CNN is in omnidirectional vision.

Although very little omnidirectional image data is currently available in public repositories, the increasing prevalence of omnidirectional sensors in drones, robots, and autonomous cars makes this a very compelling application of our work.

We use the ZYZ Euler parameterization for SO(3).

An element R ∈ SO(3) is written as DISPLAYFORM0 where α ∈ [0, 2π], β ∈ [0, π] and γ ∈ [0, 2π], and Z resp.

Y are rotations around the Z and Y axes.

Using this parameterization, the normalized Haar measure is DISPLAYFORM1 We have SO(3) dR = 1.

The Haar measure BID23 BID3 ) is sometimes called the invariant measure because it has the property that SO(3) f (R R)dR = SO FORMULA2 f (R)dR (this is analogous to the more familiar property DISPLAYFORM2 f (x)dx for functions on the line).

This invariance property allows us to do many useful substitutions.

We have a related parameterization for the sphere.

An element x ∈ S 2 is written DISPLAYFORM3 where n is the north pole.

This parameterization makes explicit the fact that the sphere is a quotient S 2 = SO(3)/ SO(2), where H = SO(2) is the subgroup of rotations around the Z axis.

Elements of this subgroup H leave the north pole invariant, and have the form Z(γ).

The point x(α, β) ∈ S 2 is associated with the coset representativex = R(α, β, 0) ∈ SO(3).

This element represents the cosetxH = {R(α, β, γ)|γ DISPLAYFORM4 The normalized Haar measure for the sphere is DISPLAYFORM5 The normalized Haar measure for SO(2) is DISPLAYFORM6 So we have dR = dx dh, again reflecting the quotient structure.

We can think of a function on S 2 as a γ-invariant function on SO(3).

Given a function f : S 2 → C we associate the functionf (α, β, γ) = f (α, β).

When using normalized Haar measures, we have: DISPLAYFORM7 This will allow us to define the Fourier transform on S 2 from the Fourier transform on SO(3), by viewing a function on S 2 as a γ-invariant function on SO(3) and taking its SO(3)-Fourier transform.

We have defined the S 2 correlation as DISPLAYFORM0 Without loss of generality, we will analyze here the single-channel case K = 1.This operation is equivariant: DISPLAYFORM1 A similar derivation can be made for the SO(3) correlation.

The spherical convolution defined by BID10 is: DISPLAYFORM2 where n is the north pole.

Note that in this definition, the output of the spherical convolution is a function on the sphere, not a function on SO(3) as in our definition of cross-correlation.

Note further that unlike our definition, this definition involves an integral over SO(3).If we write out the integral in terms of Euler angles, noting that the north-pole n is invariant to Z-axis rotations by γ, i.e. R(α, β, γ)n = Z(α)Y (β)Z(γ)n = Z(α)Y (β)n, we see that this definition implicitly integrates over γ in only one of the factors (namely ψ), making it invariant wrt γ rotation.

In other words, the filter is first "averaged" (making it circularly symmetric) before it is combined with f (This was observed before by BID19 ).

We consider this to be much too limited for the purpose of pattern matching in spherical CNNs.

With each compact topological group (like SO(3)) is associated a discrete set of orthogonal functions that arise as matrix elements of irreducible unitary representations of these groups.

For the circle (the group SO(2)) these are the complex exponentials (in the complex case) or sinusoids (for real functions).

For SO(3), these functions are known as the Wigner D-functions.

As discussed in the paper, the Wigner D-functions are parameterized by a degree parameter l ≥ 0 and order parameters m, n ∈ [−l, . . .

, l].

In other words, we have a set of matrix-valued functions The S 2 convolution of f 1 and f 2 is equivalent to the SO(3) convolution of the associated rightinvariant functionsf 1 ,f 2 (see Appendix A): DISPLAYFORM0 The Fourier transform of a right invariant function on SO(3) equals So we can think of the S 2 Fourier transform of a function on S 2 as the n = 0 column of the SO(3) Fourier transform of the associated right-invariant function.

This is a beautiful result that we have not been able to find a reference for, though it seems likely that it has been observed before.

@highlight

We introduce Spherical CNNs, a convolutional network for spherical signals, and apply it to 3D model recognition and molecular energy regression.

@highlight

The paper proposes a framework for constructing spherical convolutional networks based on a novel synthesis of several existing concepts

@highlight

This paper focuses on how to extend convolutional neural networks to have built-in spherical invariance, and adapts tools from non-Abelian harmonic analysis to achieve this goal.

@highlight

The authors develop a novel scheme for representing spherical data from the ground up