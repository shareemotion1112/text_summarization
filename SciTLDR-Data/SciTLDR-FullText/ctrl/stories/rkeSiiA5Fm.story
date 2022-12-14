The ground-breaking performance obtained by deep convolutional neural networks (CNNs) for image processing tasks is inspiring research efforts attempting to extend it for 3D geometric tasks.

One of the main challenge in applying CNNs to 3D shape analysis is how to define a natural convolution operator on non-euclidean surfaces.

In this paper, we present a method for applying deep learning to 3D surfaces using their spherical descriptors and alt-az anisotropic convolution on 2-sphere.

A cascade set of geodesic disk filters rotate on the 2-sphere and collect spherical patterns and so to extract geometric features for various 3D shape analysis tasks.

We demonstrate theoretically and experimentally that our proposed method has the possibility to bridge the gap between 2D images and 3D shapes with the desired rotation equivariance/invariance, and its effectiveness is evaluated in applications of non-rigid/ rigid shape classification and shape retrieval.

A recent research effort in computer vision and geometric processing communities is towards replicating the incredible success of deep convolutional neural networks (CNNs) from the image analysis to 3D shape analysis.

A straightforward extension is to treat a 3D shape as a voxel grid BID38 ; BID16 ; BID30 ; BID36 ; BID22 .)

Alternative methods include encoding a 3D shape as a collection of 2D renderings from multiple cameras BID20 ; BID31 ; ,) or projecting a 3D object onto geometric entities which can be flattened as 2D images BID27 ; BID4 ; BID25 .)

All these methods convert a 3D shape into an Euclidean grid structure which supports shift (translational) equivariance/invariance, such that conventional CNNs can work out-of-the box.

Although embedded in R 3 , 3D shapes are typically represented as manifold surfaces.

Recent research has particularly focused on convolutional networks for non-Euclidean domains such as manifolds or graphs.

One of the main difficulties of adopting CNNs and similar methods in these nonEuclidean domains is the lack of shift-invariance on surfaces or graphs BID15 .)

Our motivation comes from the representation of 3D shapes as functions on spheres.

We transfer the problem of manifold surface convolution into spherical convolution with the primary benefit of rotation invariance.

Although shift-invariance is hard to achieve on general surfaces, by replacing filter translations with filter rotations, rotation equivariance/invariance can be obtained on the 2-sphere.

Furthermore, spherical descriptors of 3D shapes are compact and require a network of lower capacity, compared to voxel or multi-view representations.

In this work, we are primarily interested in analyzing 3D geometric data using a specific type of spherical convolution either for classification or retrieval tasks.

Surface convolution One approach to shift-invariance on surfaces is using re-parameterization methods.

Geodesic CNN BID15 ; ; BID17 ) uses local geodesic polar coordinates to parameterize a surface patch locally around a point.

An angular max-pooling layer or local coordinate frame alignment are proposed to account for the filter's local rotational degree of freedom.

Spherical parameterization methods BID18 ; BID19 ; BID10 ) map a genus-0 3D shape onto a sphere bijectively which provides a global framework for the spherical convolution.

BID28 transfer a genus-0 3D shape into a parameterized spherical image, and then flatten it into a planar geometry image.

Data augmentation is necessary for geometry images in order to account for inconsistent cut positions and orientations.

Toric covers BID14 ) is a seamless representation which stitches four copies of a genus-0 surface and globally maps them onto a planar flat torus.

Spectral methods perform convolution on the spectral domain using the graph Laplacian and its eigen space decomposition BID39 ; BID3 ).

This method can efficiently address the shift-invariance problem, however, it suffers the difficulty with cross-shape learning since the spectral decomposition of each shape can be inconsistent.

Spherical convolution Spherical representation of 3D shapes have been used for shape matching BID11 ; BID8 ; BID13 ), remeshing BID19 ), medical imaging BID26 ) and other tasks before the deep learning era.

Recently, researchers have started to explore deep spherical convolutional neural networks for tasks such as molecular modeling BID1 ), ominidirectional vision BID32 ) and 3D shape recognition BID5 ; BID7 ).

BID32 discretized a spherical image using a lat-lon grid (see FIG2 ) and flattened it through equirectangular projection.

A variable filter size is proposed to compensate for the imbalanced sampling along longitudinal direction.

In BID1 , a cubed-sphere grid (see FIG2 ) is investigated in addition to a lat-lon grid, to achieve relatively more uniform grid on spheres.

The work of BID5 generalizes the spherical convolution with the full three rotational degrees of freedom in the 3D space, and it maps a spherical image to features on SO(3) using generalized Fourier transform.

A similar work is done in BID7 with azimuthally symmetric filters.

In this paper, we propose an alt-az anisotropic spherical convolutional neural network (or a 3 SCNN for short) for various rigid and non-rigid shape analysis tasks.

FIG0 gives an overview of our method.

A 3D shape is represented as a set of spherical images using spherical parameterization (for non-rigid shapes) or spherical projection (for rigid shapes, not shown in the figure).

An icosahedron based spherical grid is used as the discrete representation of the spherical images.

The convolution is applied directly on the spherical representation of the shape using a geodesic disc shape of filter.

The proposed deep a 3 SCNN has multiple sequential convolutions followed by a nonlinearity such as ReLU and Spherical (max or average) Pooling, all conducted on the spherical domain.

Output is a set of spherical images which capture high-level shape feature descriptors.

Following are the main contributions of our paper:(1) theoretical analysis of the relationship between various definition of convolutions for functions defined on the 2-sphere and a novel convolutional neural network using alt-az anisotropic spherical convolutions that emulates most aspects of standard convolutional networks in R 2 ;(2) an efficient geodesic grid data structure to support fast computation of the spherical convolution with locally-supported geodesic disc filters;(3) an empirical demonstration of the utility of a 3 SCNN with 3D shape learning problems.3 ALT-AZ CONVOLUTION ON 2-SPHERE

2-sphere or unit sphere S 2 can be regarded as the set of points u ??? R 3 with norm one.

The 2-sphere is a 2-manifold on which any point?? is a unit vector.

The?? can be parametrized by spherical DISPLAYFORM0 2 has a positive area A = r ds(??), where ds(??) = sin ??d??d??.

A special region on the 2-sphere is called polar cap region R ??0 , around the north pole,??(0, 0, 1), which is azimuthally symmetric and is parameterized by a maximum colatitude angle ?? 0 : DISPLAYFORM1 3D Rotations The set of rotations in three dimensions is called "special orthogonal group" SO(3).

SO(3) is a 3-manifold on which any rotation R ??? SO(3) can be represented as a 3 ?? 3 matrix.

Each rotation R is associated with three independent parameters, we use the right hand rule zyz-Euler DISPLAYFORM2 If we fix the third rotation angle ?? to zero, SO(3) is reduced into a subset A with two independent parameters.

Any rotation R ??? A can be described as an alt-az rotation 1 : DISPLAYFORM3 An alt-az rotation can be considered as a composition of an altitude rotation R (y) ?? ??? SO(2) and a azimuth rotation R (z) ?? ??? SO(2).

Rotation operator We define the effect of general rotation on spherical functions as an operator D R (??, ??, ??) which corresponds to the rotation matrix R defined in Eqn.

(2).

The effect of D R (??, ??, ??) on the spherical image f can be realized through an inverse rotation R ???1 of the coordinate system.

That is, DISPLAYFORM4 ).3.2 CONVOLUTION ON THE 2-SPHEREThe convolution operator in n dimensional Euclidean space R n is given by: DISPLAYFORM5 The above equation is used as a reference to develop different notions of convolution on the 2-sphere.

Unlike conventional Euclidean domain signal, for spherical functions there is no standard convolution operators defined.

Two competing definitions exist in literature:Type I: General anisotropic convolution: This convolution operator on 2-sphere tries to emulate the convolution in Euclidean spaces by replacing translations with full rotation in SO(3) and integrating over all possible rotations.

This gives the most general definition of spherical convolution.

Given a spherical filter h and spherical image f evaluated at a point?? ??? S 2 , general anisotropic convolution on S 2 is defined: DISPLAYFORM6 Note that the output function g is not defined on the original S 2 .

Instead, it is a function of three Euler angles (??, ??, ??) and is therefore defined on the 3-manifold SO(3) (please see BID5 for detail.)Type II: Azimuthally isotropic convolution: This spherical convolution outputs a function defined on S 2 using an azimuthally symmetric filter h 0 (??) BID7 ; BID6 ): DISPLAYFORM7 Referring to Eqn.

FORMULA10 , we see that an arbitrary filter h is essentially transformed into a rotationally symmetric filter h 0 through circular "averaging".

Type II spherical convolution zeros the contribution of angular variations from a filter, and hence, is considered restrictive for pattern matching purpose in spherical image processing.

Towards developing a spherical convolution which respects some important properties of standard convolutions defined in R 2 , we propose to use alt-az spherical convolution.

In R 2 , the two spatial translations are isometric mappings and are directly convolved, whereas the isometry corresponding to a rotation in SO(2) is generally not convolved.

Several works on the rotation equivariant/invariant CNNs have been proposed BID37 ; BID21 ; BID12 ,) and are proven to be effective; but they typically incur a significant increase in the number of parameters and computational load.

Similarly, in the spherical domain, the two degrees of freedom in altaz rotation are the direct analogs of two spatial translations in R 2 ("shifting on the sphere"), and DISPLAYFORM8 ?? , emulating the non-rotatable filters in R 2 , can be fixed and treated with data augmentation.

Intuitively, we want to shift a spherical disc filter on the 2-sphere without self rotating the filter.

We now formally define our alt-az spherical convolutional operator.

Type III: alt-az anisotropic spherical convolution (a 3 SConv): Constraining the rotation of filter within alt-az rotation set A, a filter h spans the altitude change by ?? and azimuth change by ??, and is convolved with the spherical signal f .

Mathematically, a 3 SConv is defined as: DISPLAYFORM9 a 3 SConv operator has the following desirable properties:??? Domain consistency: It takes two functions in L 2 (S 2 ) and generates a function back in L 2 (S 2 ), such that cascaded layers of spherical convolutions can be utilized to extract hierarchical spherical patterns;??? Azimuth rotation equivariance: An map is rotation equivariant if DISPLAYFORM10 In general cases, a 3 SConv is not equivariant to an arbitrary rotation in SO(3).

If Q is an azimuth rotation, a 3 SConv has the equivariance property 2 .

I.e. for an azimuth rotation Convolution with locally-supported filters Traditional CNNs are efficient due to the use of locally-supported filters and weight sharing.

On the 2-sphere, we propose to use locally-supported geodesic disc filters in the form of polar caps.

Mathematically, a locally-supported filter is defined as a space limited spherical function belonging to the follow subspace: DISPLAYFORM11 DISPLAYFORM12 where R r0 is the polar cap region on which the geodesic disc filter is defined, and r0 defines the size of a filter.

FIG1 shows a locally-supported geodesic disc filter undergoing different types of rotation.

Our a 3 SCNN consists of several layers that are applied subsequently (see FIG0 ).

Besides the a 3 SConv layer described above, we further discuss the following two specific types of layers defined on the 2-sphere.

A local max pooling (LMP) layer replaces a spherical image f in at any point?? 0 (?? 0 , ?? 0 ) with the maximum function value in its geodesic disc neighborhood, i.e., DISPLAYFORM0 where??(??, ??) is a neighboring point of?? 0 , and |.| denotes the geodesic distance between them.

A global spherical max pooling (GMP) layer operates on a spherical image f in with k channels and outputs a k dimensional vector in R k .

For each channel f in i (i = 1, 2, ..., k), a GMP layer outputs a single value represent the most salient feature.

I.e., DISPLAYFORM1 Notice a GMP layer is invariant to any rotation D R (??, ??, ??) of the input spherical image f : DISPLAYFORM2 Data augmentation After going through a set of a 3 SConv layers, the global azimuth rotation of an input spherical image f will be transformed into the same rotation of the extracted spherical descriptors (see Eqn.

FORMULA1 ).

With a GMP layer followed, the extract feature vector will be invariant to the azimuth rotation of f .

For arbitrary rotation of f in SO(3), our a 3 SConv layer does not have the equivariance property.

This means data rotation augmentation is theoretically required to recognize f in random orientations.

In appendix B, we show that, an a 3 SCNN network constructed by several a 3 SConv layers together with a GMP layer can generalize to arbitrary unseen orientations with SO(2) rotation augmentation about any axis which is not parallel to y or z axis.

From a computational point of view, implementing the spherical convolution defined above in Eqns. (8-10) is difficult because it is not possible to uniformly discretize the surface of the sphere such that each sample point shares the same neighborhood.

Therefore, a popular method of performing spherical convolution is to project the discretized spherical functions and filters onto the span of Wigner D functions for type I spherical convolution (see BID5 ,) or spherical harmonics for type II spherical convolution (see BID7 .)

They then perform the convolution in the Fourier domain via point-wise multiplications.

The lack of locality support in the spherical Fourier transform inhibits us from using this method.

Locally-supported filters belong to a subspace of space limited signals which is by nature infinite-dimensional.

No non-trivial local filters can have a finite representation in the spectral domain.

BID7 use spectral smoothness to enforce a spatial decay in filters, and hence, achieve locality.

However, the filters are still defined on the whole spherical domain which is memory inefficient.

In this paper, we propose an alternate method which performs direct spherical convolution using geodesic grid discretization.

Geodesic grid discretization Uniform geodesic grid cannot be achieved on the sphere except for the projections of five platonic polyhedra -tetrahedron, cube, octahedron, dodecahedon and icosahedron.

These polyhedra can be further subdivided into different frequencies to obtain finer approximation of a sphere FIG2 shows a subdivision of the projection of a cube on a sphere.)

Among the five platonic polyhedra, the icosahedron is most similar to the sphere BID24 ).

After the subdivision, the resulting triangulation has the least imbalance in area between its constituent triangle FIG2 ).

Most of the vertices have six direct neighbors except for the original 12 vertices of the icosahedron.

This makes the icosahedron-based geodesic grid discretization most suitable for the discrete spherical convolution.

We call this type of geodesic grid an icosahedronsphere grid.

The total number of grid vertices are N = f 2 ?? 10 + 2, where f is the subdivision frequency.

Considering the structure of the icosahedron-sphere grid, in order to obtain a multi-level spherical feature map, the stride of a convolution or pooling layer has to be a multiple of 2 n .

The stride is applied accordingly to the subdivision frequency f. FIG2 shows the icosahedron in different subdivision frequencies 1, 2, 4 and 8.

A natural shape of the locally-supported filter correlating with the icosahedron-sphere grid, is a geodesic disc which can be discretized as a hexagonal grid of different ring sizes.

FIG3 shows two examples of such filters.

The same shape of geodesic disc and discretized hexagonal grid is used for local spherical max pooling LMP layers.

Efficient data structure The icosahedron-sphere grid data structure is self-sufficient to support spherical convolution, pooling and other CNN operators.

However the linked data structure (vertexedge-face and link data for topology) is not space efficient and is time consuming, in order to find the neighbors of a vertex and shift a filter on the sphere during the convolution.

In this work, we use a rectilinear data structure to enable efficient spherical convolution and pooling.

The icosahedronbased spherical mesh can be opened into 2D plane and represented as a grid structure as shown in FIG3 .

The cut is along eleven edges of the icosahedron as shown in FIG3 .

By rotating the u and v axes in FIG3 into orthogonal axes, we obtain five rectangular 2D patches to store all the vertices of the icosahedron-sphere grid, as illustrated in FIG3 .

This construction has two main advances: (1) within each patch, shifting filters on the 2-sphere is approximately equivariant to translation in u and v and (2) features on the geodesic grid can naturally be expressed using tensors, which means that the spherical convolution can be efficiently implemented on a GPU.

When implementing spherical convolutions and pooling operations for the icosahedron-sphere grid, one has to be careful in padding each patch with the contents of the other two neighboring patches.

If a point is on the colored cut-lines as shown in FIG3 , then its k-ring hexagon neighbors are retrieved across the boundary of matrix.

Notice here that by using the cross-boundary neighborhood padding strategy, the rectilinear data structure realizes a seamless geodesic grid representation of the 2-sphere.

As a pre-processing step for all experiments, we first need to convert 3D shapes to functions on the 2-sphere.

Two different methods were employed to do this conversion: spherical projection for rigid shapes and spherical parameterization for non-rigid shapes.

Spherical projection For rigid shapes, we project a 3D shape onto an enclosing sphere using a straightforward ray casting scheme and we collect the following two types of spherical descriptors.

non-convexity of surfaces, we assume the projections capture sufficient information of the shape to be useful for rigid shape analysis.

Spherical parametrization Spherical projection produces extrinsic shape descriptors which are not suitable for non-rigid shape analysis.

To handle deformable shapes, we use the authalic spherical parametrization method BID28 ) to obtain an area-preserving bijective spherical map and use the following intrinsic shape descriptors: (1) Principal Curvatures: the two principal curvature k min and k max measure the degree to which the surface bends in orthogonal directions at a point.(2) Average Geodesic Distance (AGD): this measures the centerness of a point on the surface.

FORMULA3 Heat kernel signature (HKS) BID33 ): this measures the amount of untransferred heat after time t, assuming an unit heat source is added on each point of the surface (see FIG4 right for example).In all of the following experiments, spherical functions are discretized using icosahedron-sphere grid with subdivision frequency f = 32.

This will generate five patches of size 33 ?? 65, by stacking them one above the other.

The input size to all networks are 165 ?? 65 ?? K, where K is the number of input channels.

For the 12 valence-5 vertices in the icosahedron, we apply the shared hexagon filter by computing the center point twice.

Since it affects a small number of vertices, we empirically validate that the effect can be ignored.

We first conduct experiments on SHREC'11 non-rigid shape classification, and we compare three types of spherical functions: (i) Intrinsic-2 contains the two principal curvatures, (ii) Intrinsic-3 adds AGD to intrinsic-2, and (iii) Intrinsic-8 adds five HKS sampled at 5 logarithmic time scales on top of Intrinsic-3.

And we compare five modes of experiments: (a) trained with original data without data augmentation (NA), (b) trained with 36 azimuth rotation augmentation (AZ) by sampling ?? per 10 degrees, (c) trained with 36 rotation augmentation (SO(2)(x)) by rotating about x-axis per 10 degrees, (d) trained with 72 alt-az rotation augmentation (Alt-AZ) by sampling ?? and ?? per 30 degrees, and (e) trained with arbitrary 128 rotations (SO FORMULA3 ).

In each category, 16 objects are used for training and 4 objects are used for testing.

Architecture and hyper parameters Our network contains five a 3 SConv-dropout-ReLU-LMP blocks.

A 20% dropout is added right after each spherical convolution layer for regularization.

The resulting spherical functions are pooled using a global max pooling (GMP) layer followed by two fully connected layers for the final classification.

A 50% dropout layer is inserted in between the last two fully connected layers.

We use 32, 64, 64, 128, 128 features for the a 3 SConv layers, and 512 features are output from the GMP and fed into the first fully connected layer.

Each filter on S 2 has kernel size ring-2, stride 1 and each LMP layer has size ring-2 and stride 2.Results TAB0 shows the performance of these intrinsic descriptors for non-rigid shape classification under different augmentation modes.

Notice that the original testing data are randomly posed.

In spite the small training data, our network achieves good classification accuracy for Intrinsic-8 even without data augmentation.

We attribute the capability to generalize to random perturbed data to the use of LMP layers, which allows a certain amount SO(3) rotation invariance.

Comparing the four types of data augmentation strategies, our experimental result confirms that SO(2)(x) aug- mentation performs better than the original training data, AZ type augmentation and SO(3) random augmentation.

It is predictable that alt-az augmentation performs even better with more augmented data.

Theoretically, our network is invariant to azimuth rotation except for the two poles.

Due to the approximation error introduced in the implementation, we see AZ type data augmentation can compensate those equivariance errors.

Compare to other deep learning based non-rigid shape analysis, the geometry image method BID28 ) is most similar to ours.

Their reported classification accuracy of 96.6% is also based on an alt-az rotation augmentation.

Our method outperforms the state-of-the-art approach by about 3% margin even by using two principal curvature (Intrinsic-2) as inputs.

We further experiment on ModelNet10 and ModelNet40 rigid shape databases, and we use the model trained and tested on aligned data as baseline, and we explicitly test the equivariance/invariance property of our learned shape representations by perturbing the testing data in different modes.

Architecture and experiment setup We experiment with four types of perturbations: (a) test with original aligned data (NR), (b) test with azimuthal rotation perturbations (AZ), (c) test with alt-az field rotation perturbations (Alt-AZ) and (d) test with random SO(3) rotations.

We also randomly perturb the training data and test it with randomly perturbed testing data.

In these experiments, two channels of spherical functions: SEF and NDF are used as the input and we use the same network structure as we use in SHREC'11, except that in the five cascaded a 3 Sconv layers, 32,64,128,256,512 filters are used, and in the first fully connected layer, 1024 features are generated for classification.

Results TAB1 summarizes the performance of a 3 SCNN for classifying rigid objects for unseen orientations.

The column NR/NR shows the classification accuracy for aligned training and testing data.

As expected, aligned data gives the best classification performance and it serves as the baseline for evaluating the rotation equivariance/invariance of the learned models.

Comparing the four types of perturbing modes, the learned representation has the best classification accuracy for testing data perturbed with alt-az rotation.

For random SO(3) rotation, whether it is with perturbed testing data only (see column NR/ SO(3)) or with both testing and training data perturbed (column SO(3) / SO(3)), our network still performs well, showing that a 3 SCNN may generalize to unseen orientations even without data augmentation.

It is counter-intuitive that azimuthal type of perturbation gives slightly worse performance (see column NR/AZ) since in theory, our network is azimuthal rotation invariant.

We interpret this as the result of the equivariance error coming from icosahedron-sphere grid tessellation and singularities at poles, while alt-az and SO(3) perturbation might compensate this with perturbations which can be better treated by the LMP and GMP layers.

Discussion It is difficult to provide quantitative comparison because little research has been done on learning rotation invariant shape descriptors.

The state-of-the-art methods such as, volumetric methods and point cloud based method report very high classification accuracies on ModelNet, but none of them can generalize to unseen orientations.

Given the rather task agnostic architecture of our model and the lossy but compact input representation we use, we interpret our models performance as strong empirical support of the effectiveness of learning alt-az rotation invariant shape descriptors using alt-az spherical convolution operators.

We evaluate shape retrieval performance on the challenging SHREC'11 non-rigid dataset.

Intrinsic-8 is used as our input spherical images and we extract the output 512 features of the fully connected layer as the shape descriptors.

Our approach significantly outperforms all other methods with 0.82 mAP retrieval performance.

Fig.6 uses the dimensionality reduction method t-SNE (van der Maaten & Hinton FORMULA2 ) to plot the rotation invariant feature descriptors extracted.

It shows that our learned descriptors successfully disentangle the original 3D object space and exhibit a clustered behavior in the feature vector space.

Figure 6: The shape descriptor of SHREC'11 original models (training set) extracted by our a 3 SCNN network, rendered with t-SNE.Finally, we run shape retrieval experiments on ShapeNet Core55, following rules of the SHREC'17 3D shape contest BID23 .)

There is a aligned regular dataset and a version in which all models are perturbed by rotations.

We concentrate on the perturbed version to test the quality of our learned shape descriptors to unseen orientations for large scale rigid 3D object retrieval task.

An SO(2) rotation augmentation is performed by rotating each training data per 60 degrees about a random axis.

The same architecture and same type and size of input that we used for ModelNet classification problem is used in this experiment.

The learned model from ModelNet40 was transferred and finetuned for SHREC'17 feature extraction.

Our trained model obtains a 83% classification accuracy after about 24 hours of training and we use the 1024 features extracted from the first fully connected layer as the invariant shape descriptors to perform the shape similarity calculation using cosine distance.

We then evaluated our trained descriptors using the official metrics and compared to the top four competitors, which includes the other two spherical convolution based methods.

As shown in Table.

3, all three spherical convolution based methods BID5 BID7 and ours) preform slightly below the current best, we believe that this is due to the information loss caused by projecting 3D shapes onto the 2-sphere.

To our surprise, all the three spherical convolution based methods report very similar performance, ours is slightly below BID5 and slightly above BID7 .

Both of the other two spherical convolution methods utilize Fast Fourier Transform (FFT) to compute the convolution which does not support local filters.

Ours offers an alternative method which complete the current work while offers multi-level feature extraction capabilities and GPU based fast computation.

In this paper, we presented and analyzed a convolutional neural network based on alt-az anisotropic spherical convolution operator which is different from the existing types of networks.

Numerically, we implemented an efficient algorithm for computing spherical convolution with locally-supported geodesic filters using icosahedron-sphere grid.

We demonstrated the efficacy of our approach for non-rigid/ rigid shape classification and retrieval and showed that it compares favorably to competing methods.

Furthermore, we have shown that the proposed method can effectively generalize across rotations, and achieve state-of-the-art results on competitive 3D shape recognition tasks, without excessive data augmentation, feature engineering and task-tuning.

Under the definition of alt-azimuth anisotropic convolution, for an alt-az rotation D R (??, ??, 0), and a general rotation D Q (?? 1 , ?? 1 , ?? 1 ), (assume the number of channels K = 1 for simplicity,) we have: DISPLAYFORM0 )ds(??) = Q ???1 R is in general a rotation in SO(3), but when Q is an azimuth rotation, DISPLAYFORM1 ?? is an alt-az rotation such that, DISPLAYFORM2 B SO(2) ROTATION AUGMENTATION FOR GENERAL ORIENTATIONS Since composite a 3 SConv layers will not affect rotation equivariance property, without losing generality, we assume our network consists of one a 3 SConv and one GMP layer.

For any feature output from a GMP layer, suppose it is activated at point?? 0 (?? 0 , ?? 0 ).

It corresponds to a maximum correlation between the learned filter h and the input image f (when h is "alt-az" rotated onto?? o ).

Any SO(3) rotation of f which first rotates the point?? o back to the north pole??(0, 0, 1), followed by an arbitrary alt-az rotation D(?? , ?? , 0) to new point?? 1 , will be invariant to the network.

When h is convolved at point?? 1 .

We define this set of rotation as "alt-az shift rotation"(see FIG7 ).

Alt-az shift rotation rotation causes no relative angular change of the geodesic disc centered at?? o with respect to the h's convolving.

For a directionn (n is not along z-axis or y-axis), if one rotates f aboutn an arbitrary angle which move the original salient point?? o to?? 1 , this rotation is in general not an alt-az shift rotation and will change the correlation between h and f when h is convolved at?? 1 (See FIG7 ).

If one rotates f aboutn a full round, the geodesic disc centered at?? o , will go through a relative self rotation from 0 to 360 degrees, using the alt-az rotation of h to each corresponding point as the direction references.

Therefore, by augmenting f using SO(2) rotation about an arbitrary axisn, any feature of f , after SO(3) random rotation, can always alt-az shift rotate to the same feature in one of f 's augmented copies.

<|TLDR|>

@highlight

A method for applying deep learning to 3D surfaces using their spherical descriptors and alt-az anisotropic convolution on 2-sphere.

@highlight

Presents a polar anisotropic convolution scheme on a unit sphere by replacing filter translation with filter rotation.

@highlight

This paper explores deep learning of 3D shapes using alt-az anisotropic 2-sphere convolution