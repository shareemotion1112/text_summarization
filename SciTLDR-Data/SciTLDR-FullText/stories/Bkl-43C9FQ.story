We present an efficient convolution kernel for Convolutional Neural Networks (CNNs) on unstructured grids using parameterized differential operators while focusing on spherical signals such as panorama images or planetary signals.

To this end, we replace conventional convolution kernels with linear combinations of differential operators that are weighted by learnable parameters.

Differential operators can be efficiently estimated on unstructured grids using one-ring neighbors, and learnable parameters can be optimized through standard back-propagation.

As a result, we obtain extremely efficient neural networks that match or outperform state-of-the-art network architectures in terms of performance but with a significantly lower number of network parameters.

We evaluate our algorithm in an extensive series of experiments on a variety of computer vision and climate science tasks, including shape classification, climate pattern segmentation, and omnidirectional image semantic segmentation.

Overall, we present (1) a novel CNN approach on unstructured grids using parameterized differential operators for spherical signals, and (2) we show that our unique kernel parameterization allows our model to achieve the same or higher accuracy with significantly fewer network parameters.

A wide range of machine learning problems in computer vision and related areas require processing signals in the spherical domain; for instance, omnidirectional RGBD images from commercially available panorama cameras, such as Matterport , panaramic videos coupled with LIDAR scans from self-driving cars BID17 , or planetary signals in scientific domains such as climate science BID31 .

Unfortunately, naively mapping spherical signals to planar domains results in undesirable distortions.

Specifically, projection artifacts near polar regions and handling of boundaries makes learning with 2D convolutional neural networks (CNNs) particularly challenging and inefficient.

Very recent work, such as BID10 and BID16 , propose network architectures that operate natively in the spherical domain, and are invariant to rotations in the SO(3) group.

Such invariances are desirable in a set of problems -e.g., machine learning problems of molecules -where gravitational effects are negligible and orientation is arbitrary.

However, for other different classes of problems at large, assumed orientation information is crucial to the predictive capability of the network.

A good example of such problems is the MNIST digit recognition problem, where orientation plays an important role in distinguishing digits "6" and "9".

Other examples include omnidirectional images, where images are naturally oriented by gravity; and planetary signals, where planets are naturally oriented by their axis of rotation.

In this work, we present a new convolution kernel for CNNs on arbitrary manifolds and topologies, discretized by an unstructured grid (i.e., mesh), and focus on its applications in the spherical domain approximated by an icosahedral spherical mesh.

We propose and evaluate the use of a new parameterization scheme for CNN convolution kernels, which we call Parameterized Differential Operators (PDOs), which is easy to implement on unstructured grids.

We call the resulting convolution operator that operates on the mesh using such kernels the MeshConv operator.

This parameterization scheme utilizes only 4 parameters for each kernel, and achieves significantly better performance Illustration for the MeshConv operator using parameterized differential operators to replace conventional learnable convolutional kernels.

Similar to classic convolution kernels that establish patterns between neighboring values, differential operators computes "differences", and a linear combination of differential operators establishes similar patterns.than competing methods, with much fewer parameters.

In particular, we illustrate its use in various machine learning problems in computer vision and climate science.

In summary, our contributions are as follows:??? We present a general approach for orientable CNNs on unstructured grids using parameterized differential operators.???

We show that our spherical model achieves significantly higher parameter efficiency compared to state-of-the-art network architectures for 3D classification tasks and spherical image semantic segmentation.??? We release and open-source the codes developed and used in this study for other potential extended applications 1 .We organize the structure of the paper as follows.

We first provide an overview of related studies in the literature in Sec. 2; we then introduce details of our methodology in Sec. 3, followed by an empirical assessment of the effectiveness of our model in Sec. 4.

Finally, we evaluate the design choices of our kernel parameterization scheme in Sec. 5.

Spherical CNNs The first and foremost concern for processing spherical signals is distortions introduced by projecting signals on curved surfaces to flat surfaces.

BID35 process equirectangular images with regular convolutions with increased kernel sizes near polar regions where greater distortions are introduced by the planar mapping.

BID11 and BID41 use a constant kernel that samples points on the tangent plane of the spherical image to reduce distortions.

A slightly different line of literature explores rotational-equivariant implementations of spherical CNNs.

BID10 proposed spherical convolutions with intermediate feature maps in SO(3) that are rotational-equivariant.

BID16 used spherical harmonic basis to achieve similar results.

Reparameterized Convolutional Kernel Related to our approach in using parameterized differential operators, several works utilize the diffusion kernel for efficient Machine Learning and CNNs.

BID20 was among the first to suggest the use of diffusion kernel on graphs.

BID1 propose Diffusion-Convolutional Neural Networks (DCNN) for efficient convolution on graph structured data.

BID4 introduce a generalization of classic CNNs to non-Euclidean domains by using a set of oriented anisotropic diffusion kernels.

BID9 utilized a linear combination of filter banks to acquire equivariant convolution filters.

BID33 explore the reparameterization of convolutional kernels using parabolic and hyperbolic differential basis with regular grid images.

Non-Euclidean Convolutions Related to our work on performing convolutions on manifolds represented by an unstructured grid (i.e., mesh), works in geometric deep learning address similar problems BID6 .

Other methods perform graph convolution by parameterizing the convolution kernels in the spectral domain, thus converting the convolution step into a spectral dot product BID7 BID15 BID19 BID40 .

BID24 perform convolutions directly on manifolds using cross-correlation based on geodesic distances and BID23 use an optimal surface parameterization method (seamless toric covers) to parameterize genus-zero shapes into 2D signals for analysis using conventional planar CNNs.

Image Semantic Segmentation Image semantic segmentation is a classic problem in computer vision, and there has been an impressive body of literature studying semantic segmentation of planar images BID32 BID2 BID22 BID18 BID37 .

study semantic segmentation of equirectangular omnidirectional images, but in the context of image inpainting, where only a partial view is given as input. and provide benchmarks for semantic segmentation of 360 panorama images.

In the 3D learning literature, researchers have looked at 3D semantic segmentation on point clouds or voxels BID13 BID29 BID38 BID36 BID14 .

Our method also targets the application domain of image segmentation by providing a more efficient convolutional operator for spherical domains, for instance, focusing on panoramic images .

We present a novel scheme for efficiently performing convolutions on manifolds approximated by a given underlying mesh, using what we call Parameterized Differential Operators.

To this end, we reparameterize the learnable convolution kernel as a linear combination of differential operators.

Such reparameterization provides two distinct advantages: first, we can drastically reduce the number of parameters per given convolution kernel, allowing for an efficient and lean learning space; second, as opposed to the cross-correlation type convolution on mesh surfaces BID24 , which requires large amounts of geodesic computations and interpolations, first and second order differential operators can be efficiently estimated using only the one-ring neighborhood.

In order to illustrate the concept of PDOs, we draw comparisons to the conventional 3??3 convolution kernel in the regular grid domain.

The 3 ?? 3 kernel parameterized by parameters ??: G 3??3 ?? can be written as a linear combination of basis kernels which can be viewed as delta functions at constant offsets: DISPLAYFORM0 where x and y refer to the spatial coordinates that correspond to the two spatial dimensions over which the convolution is performed.

Due to the linearity of the cross-correlation operator ( * ), the output feature map can be expressed as a linear combination of the input function cross-correlated with different basis functions.

Defining the linear operator ??? ij to be the cross-correlation with a basis delta function, we have: DISPLAYFORM1 DISPLAYFORM2 Published as a conference paper at ICLR 2019 In our formulation of PDOs, we replace the cross-correlation linear operators ??? ij with differential operators of varying orders.

Similar to the linear operators resulting from cross-correlation with basis functions, differential operators are linear, and approximate local features.

In contrast to crosscorrelations on manifolds, differential operators on meshes can be efficiently computed using Finite Element basis, or derived by Discrete Exterior Calculus.

In the actual implementation below, we choose the identity (I, 0th order differential, same as ??? 00 ), derivatives in two orthogonal spatial dimensions (??? x , ??? y , 1st order differential), and the Laplacian operator (??? 2 , 2nd order differential): DISPLAYFORM3 The identity (I) of the input function is trivial to obtain.

The first derivative (??? x , ??? y ) can be obtained by first computing the per-face gradients, and then using area-weighted average to obtain per-vertex gradient.

The dot product between the per-vertex gradient value and the corresponding x and y vector fields are then computed to acquire ??? x F and ??? y F.

For the sphere, we choose the eastwest and north-south directions to be the x and y components, since the poles naturally orient the spherical signal.

The Laplacian operator on the mesh can be discretized using the cotangent formula: DISPLAYFORM4 where N (i) is the nodes in the neighboring one-ring of i, A i is the area of the dual face corresponding to node i, and ?? ij and ?? ij are the two angles opposing edge ij.

With this parameterization of the convolution kernel, the parameters can be similarly optimized via backpropagation using standard stochastic optimization routines.

The icosahedral spherical mesh BID3 is among the most uniform and accurate discretizations of the sphere.

A spherical mesh can be obtained by progressively subdividing each face of the unit icosahedron into four equal triangles and reprojecting each node to unit distance from the origin.

Apart from the uniformity and accuracy of the icosahedral sphere, the subdivision scheme for the triangles provides a natural coarsening and refinement scheme for the Model Accuracy(%) Number of Parameters S2CNN BID10 96.00 58k SphereNet BID11 94.41 196kOurs 99.23 62k For the ease of discussion, we adopt the following naming convention for mesh resolution: starting with the unit icosahedron as the level-0 mesh, each progressive mesh resolution is one level above the previous.

Hence, for a level-l mesh: DISPLAYFORM0 where n f , n e , n v stands for the number of faces, edges, and vertices of the spherical mesh.

A detailed schematic for the neural architectures in this study is presented in FIG1 .

The schematic includes architectures for both the classification and regression network, which share a common encoder architecture.

The segmentation network consists of an additional decoder which features transpose convolutions and skip layers, inspired by the U-Net architecture BID32 .

Minor adjustments are made for different tasks, mainly surrounding adjusting the number of input and output layers to process signals at varied resolutions.

A detailed breakdown for model architectures, as well as training details for each task in the Experiment section (Sec. 4), is provided in the appendix (Appendix Sec. B).

To validate the use of parameterized differential operators to replace conventional convolution operators, we implemented such neural networks towards solving the classic computer vision benchmark task: the MNIST digit recognition problem BID21 .Experiment Setup We follow BID10 by projecting the pixelated digits onto the surface of the unit sphere.

We further move the digits to the equator to prevent coordinate singularity at the poles.

We benchmark our model against two other implementations of spherical CNNs: a rotational-invariant model by BID10 and an orientable model by BID11 .All models are trained and tested with non-rotated digits to illustrate the performance gain from orientation information.

Results and Discussion Our model outperforms its counterparts by a significant margin, achieving the best performance among comparable algorithms, with comparable number of parameters.

We attribute the success in our model to the gain in orientation information, which is indispensable for many vision tasks.

In contrast, S2CNN (Cohen et al., 2018 ) is rotational-invariant, and thus has difficulties distinguishing digits "6" and "9".

We use the ModelNet40 benchmark BID39 , a 40-class 3D classification problem, to illustrate the applicability of our spherical method to a wider set of problems in 3D learning.

For this study, we look into two aspects of our model: peak performance and parameter efficiency.

Experiment Setup To use our spherical CNN model for the object classification task, we preprocess the 3D geometries into spherical signals.

We follow BID10 for preprocessing the 3D CAD models.

First, we normalize and translate each mesh to the coordinate origin.

We then encapsulate each mesh with a bounding level-5 unit sphere and perform ray-tracing from each point to the origin.

We record the distance from the spherical surface to the mesh, as well as the sin, cos of the incident angle.

The data is further augmented with the 3 channels corresponding to the convex hull of the input mesh, forming a total of 6 input channels.

An illustration of the data preprocessing process is presented in Fig. 5 .

For peak performance, we compare the best performance achievable by our model with other 3D learning algorithms.

For the parameter efficiency study, we progressively reduce the number of feature layers in all models without changing the overall model architecture.

Then, we evaluate the models after convergence in 250 epochs.

We benchmark our results against PointNet++ BID29 ), VoxNet (Qi et al., 2016 , and S2CNN 2 .Results and Discussion FIG2 shows a comparison of model performance versus number of parameters.

Our model achieves the best performance across all parameter ranges.

In the lowparameter range, our model is able to achieve approximately 60% accuracy for the 40-class 3D classification task with a mere 2000+ parameters.

Table 2 shows a comparison of peak performance between models.

At peak performance, our model is on-par with comparable state-of-the-art models, and achieves the best performance among models consuming spherical input signals.

We illustrate the semantic segmentation capability of our network on the omnidirectional image segmentation task.

We use the Stanford 2D3DS dataset for this task.

The 2D3DS dataset consists of 1,413 equirectangular images with RGB+depth channels, as well as semantic labels across 13 different classes.

The panoramic images are taken in 6 different areas, and the dataset is officially split for a 3-fold cross validation.

While we are unable to find reported results on the semantic segmentation of these omnidirectional images, we benchmark our spherical segmentation algorithm against classic 2D image semantic segmentation networks as well as a 3D point-based model, trained and evaluated on the same data.

Figure 5: Illustration of spherical signal rendering process for a given 3D CAD model.

Experiment Setup First, we preprocess the data into a spherical signal by sampling the original rectangular images at the latitude-longitudes of the spherical mesh vertex positions.

Input RGB-D channels are interpolated using bilinear interpolation, while semantic labels are acquired using nearest-neighbor interpolation.

We input and output spherical signals at the level-5 resolution.

We use the official 3-fold cross validation to train and evaluate our results.

We benchmark our semantic segmentation results against two classic semantic segmentation networks: the U-Net BID32 and FCN8s BID22 .

We also compared our results with a modified version of spherical S2CNN, and 3D point-based method, PointNet++ (Qi et al., 2017b) using (x, y, z,r,g,b) inputs reconstructed from panoramic RGBD images.

We provide additional details toward the implementation of these models in Appendix E. We evaluate the network performance under two standard metrics: mean Intersection-over-Union (mIoU), and pixel-accuracy.

Similar to Sec. 4.2, we evaluate the models under two settings: peak performance and a parameter efficiency study by varying model parameters.

We progressively decimate the number of feature layers uniformly for all models to study the effect of model complexity on performance.

Results and Discussion FIG3 compares our model against state-of-the-art baselines.

Our spherical segmentation outperforms the planar baselines for all parameter ranges, and more significantly so compared to the 3D PointNet++.

We attribute PointNet++'s performance to the small amount of training data.

Fig. 6 shows a visualization of our semantic segmentation performance compared to the ground truth and the planar baselines.

To further illustrate the capabilities of our model, we evaluate our model on the climate pattern segmentation task.

We follow BID26 for preprocessing the data and acquiring the ground-truth labels for this task.

Figure 6 : Visualization of semantic segmentation results on test set.

Our results are generated on a level-5 spherical mesh and mapped to the equirectangular grid for visualization.

Model underperforms in complex environments, and fails to predict ceiling lights due to incomplete RGB inputs.

Results and Discussion Segmentation accuracy is presented in TAB3 .

Our model achieves better segmentation accuracy as compared to the baseline models.

The baseline model BID26 trains and tests on random crops of the global data, whereas our model inputs the entire global data and predicts at the same output resolution as the input.

Processing full global data allows the network to acquire better holistic understanding of the information, resulting in better overall performance.

We further perform an ablation study for justifying the choice of differential operators for our convolution kernel (as in Eqn.

4).

We use the ModelNet40 classification problem as a toy example and use a 250k parameter model for evaluation.

We choose various combinations of differential operators, and record the final classification accuracy.

Results for the ablation study is presented in Table 4 Table 4 : Results for the ablation study.

The choice of kernel that includes all differential operator components achieve the best accuracy, validating our choice of kernel in Eqn.

4. among other choices, and the network performance improves with increased differential operators, thus allowing for more degrees of freedom for the kernel.

We have presented a novel method for performing convolution on unstructured grids using parameterized differential operators as convolution kernels.

Our results demonstrate its applicability to machine learning problems with spherical signals and show significant improvements in terms of overall performance and parameter efficiency.

We believe that these advances are particularly valuable with the increasing relevance of omnidirectional signals, for instance, as captured by real-world 3D or LIDAR panorama sensors.

In this section we provide more mathematical details for the implementation of the MeshConv Operator as described in Sec 3.1.

In particular, we will describe in details the implementation of the various differential operators in Eqn.

4.

The identity operator as suggested by its definition is just the identical replica of the original signal, hence no additional computation is required other than using the original signal as is.

Gradient Operator Using a triangular mesh for discretizing the surface manifold, scalar functions on the surface can be discritized as a piecewise linear function, where values are defined on each vertex.

Denoting the spatial coodinate vector as x, the scalar function as f (x), the scalar function values stored on vertex i as f i , and the piecewise linear "hat" functions as ?? i (x), we have: DISPLAYFORM0 the piecewise linear basis function ?? i (x) takes the value of 1 on vertex i and takes the value of 0 on all other vertices.

Hence, the gradient of this piecewise linear function can be computed as: DISPLAYFORM1 Due to the linearity of the basis functions ?? i , the gradient is constant within each individual triangle.

The per-face gradient value can be computed with a single linear operator G on the per-vertex scalar function f : DISPLAYFORM2 where the resulting per-face gradient is a 3-dimensional vector.

We use the superscript (f ), (v) to distinguish per-face and per-vertex quantities.

We refer the reader to BID5 for detailed derivations for the gradient operator G. Denoting the area of each face as a (f ) (which can be easily computed using Heron's formula given coordinates of three points), the resulting per-vertex gradient vector can be computed as an average of per-face gradients, weighted by face area: DISPLAYFORM3 denote the per-vertex polar (east-west) and azimuthal (north-south) direction fields asx (v) and?? (v) .

They can be easily computed using the gradient operators detailed above, with the longitudinal and latitudinal values as the scalar function, followed by normalizing each vector to unit length.

Hence two per-vertex gradient components can be computed as a dot-product against the unit directional fields: DISPLAYFORM4 Laplacian Operator The mesh Laplacian operator is a standard operator in computational and differential geometry.

We consider its derivation beyond the scope of this study.

We provide the cotangent formula for computing the mesh Laplacian in Eqn.

5.

We refer the reader to Chapters 6.2 and 6.3 of BID12 for details of the derivation.

MeshConv(a, b) Mesh convolution layer with a input channels and producing b output channels MeshConv(a, b) T Mesh transpose convolution layer with a input channels and producing b output channels.

BN Batch Normalization.

ReLURectified Linear Unit activation function DownSamp Downsampling spherical signal at the next resolution level.

ResBlock(a, b, c) As illustrated in FIG0 , c stands for input channels, bottle neck channels, and output channels.

DISPLAYFORM0 The layers therein is at a mesh resolution of Li.

Concatenate skip layers of same resolution.

Total number of parameters: 61658Training details We train our network with a batch size of 16, initial learning rate of 1 ?? 10 ???2 , step decay of 0.5 per 10 epochs, and use the Adam optimizer.

We use the cross-entropy loss for training the classification network.

Architecture The input signal is at a level-5 resolution.

The network architecture closely follows that in the schematics in FIG1 .

We present two network architectures, one that corresponds to the network architecture with the highest accuracy score (the full model), and another that scales well with low parameter counts (the lean model).

The full model: [MeshConv(6, 32) Training details We train our network with a batch size of 16, initial learning rate of 5 ?? 10 ???3 , step decay of 0.7 per 25 epochs, and use the Adam optimizer.

We use the cross-entropy loss for training the classification network.

Architecture Input signal is sampled at a level-5 resolution.

The network architecture is identical to the segmentation network in FIG1 Total number of parameters: 5180239Training details Note that the number of output channels is 15, since the 2D3DS dataset has two additional classes (invalid and unknown) that are not evaluated for performance.

We train our network with a batch size of 16, initial learning rate of 1 ?? 10 ???2 , step decay of 0.7 per 20 epochs, and use the Adam optimizer.

We use the weighted cross-entropy loss for training.

We weight the loss for each class using the following weighting scheme: DISPLAYFORM0 where w c is the weight corresponding to class c, and f c is the frequency by which class c appears in the training set.

We use zero weight for the two dropped classes (invalid and unknown).

Architecture We use the same network architecture as the Omnidirectional Image Segmentation task in Sec. B.3.

Minor difference being that all feature layers are cut by 1/4.

Training details We train our network with a batch size of 256, initial learning rate of 1 ?? 10 ???2 , step decay of 0.4 per 20 epochs, and use the Adam optimizer.

We train using weighted cross-entropy loss, using the same weighting scheme as in Eqn.

13.

We provide detailed statistics for the 2D3DS semantic segmentation task.

We evaluate our model's per-class performance against the benchmark models.

All statistics are mean over 3-fold cross validation.

To further evaluate our model's runtime gain, we record the runtime of our model in inference mode.

We tabulate the runtime of our model (across a range of model sizes) in Table 8 .

We also compare our runtime with PointNet++, whose best performing model achieves a comparable accuracy.

Inference is performed on a single NVIDIA GTX 1080 Ti GPU.

We use a batch size of 8, and take the average runtime in 64 batches.

Runtime for the first batch is disregarded due to extra initialization time.

We observe that our model achieves fast and stable runtime, even with increased parameters, possibly limited by the serial computations due to network depth.

We achieve a significant speedup compared to our baseline (PointNet++) of nearly 5x, particularly closer to the high-accuracy regime.

A frame rate of over 339 fps is sufficient for real-time applications.

We provide further details for implementing the baseline models in the semantic segmentation task.

The two planar models are slightly modified by changing the first convolution to take in 4 channels (RGBD) instead of 3.

No additional changes are made to the model, and the models are trained from scratch.

We use the available open source implementation 3 for the two models.

PointNet++ We use the official implementation of PointNet++ 4 , and utilize the same code for the examplar ScanNet task.

The number of points we use is 8192, same as the number of points used for the ScanNet task.

We perform data-augmentation by rotating around the z-axis and take sub-regions for training.

S2CNN S2CNN was not initially designed and evaluated for semantic segmentation tasks.

However, we provide a modified version of the S2CNN model for comparison.

To produce scalar fields as outputs, we perform average pooling of the output signal only in the gamma dimension.

Also since no transpose convolution operator is defined for S2CNN, we maintain its bandwidth of 64 throughout the network.

The current implementations are not particularly memory efficient, hence we were only able to fit in low-resolution images of tiny batch sizes of 2 per GPU.

Architecture overview:[S2Conv(4, 64) +BN+ReLU] b64 + [SO3Conv(64, 15)] b64 + AvgPoolGamma

@highlight

We present a new CNN kernel for unstructured grids for spherical signals, and show significant accuracy and parameter efficiency gain on tasks such as 3D classfication and omnidirectional image segmentation.

@highlight

An efficient method enabling deep learning on spherical data that reaches competitive/state-of-the-art numbers with much less parameters than popular approaches.

@highlight

The paper proposes a novel convolutional kernel for CNN on the unstructured grids and formulates the convolution by a linear combination of differential operators.