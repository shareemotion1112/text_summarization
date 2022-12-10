Convolutional Neural Networks (CNN) have been successful in processing data signals that are uniformly sampled in the spatial domain (e.g., images).

However, most data signals do not natively exist on a grid, and in the process of being sampled onto a uniform physical grid suffer significant aliasing error and information loss.

Moreover, signals can exist in different topological structures as, for example, points, lines, surfaces and volumes.

It has been challenging to analyze signals with mixed topologies (for example, point cloud with surface mesh).

To this end, we develop mathematical formulations for Non-Uniform Fourier Transforms (NUFT) to directly, and optimally, sample nonuniform data signals of different topologies defined on a simplex mesh into the spectral domain with no spatial sampling error.

The spectral transform is performed in the Euclidean space, which removes the translation ambiguity from works on the graph spectrum.

Our representation has four distinct advantages: (1) the process causes no spatial sampling error during initial sampling, (2) the generality of this approach provides a unified framework for using CNNs to analyze signals of mixed topologies, (3) it allows us to leverage state-of-the-art backbone CNN architectures for effective learning without having to design a particular architecture for a particular data structure in an ad-hoc fashion, and (4) the representation allows weighted meshes where each element has a different weight (i.e., texture) indicating local properties.

We achieve good results on-par with state-of-the-art for 3D shape retrieval task, and new state-of-the-art for point cloud to surface reconstruction task.

We present a unifying and novel geometry representation for utilizing Convolutional Neural Networks (CNNs) on geometries represented on weighted simplex meshes (including textured point clouds, line meshes, polygonal meshes, and tetrahedral meshes) which preserve maximal shape information based on the Fourier transformation.

Most methods that leverage CNNs for shape learning preprocess these shapes into uniform-grid based 2D images (rendered multiview images) or 3D images (binary voxel or Signed Distance Function (SDF)).

However, rendered 2D images do not preserve the 3D topologies of the original shapes due to occlusions and the loss of the third spatial dimension.

Binary voxels and SDF representations under low resolution suffer big aliasing errors and under high resolution become memory inefficient.

Loss of information in the input bottlenecks the effectiveness of the downstream learning process.

Moreover, it is not clear how a weighted mesh where each element is weighted by a different scalar or vector (i.e., texture) can be represented by binary voxels and SDF.

Mesh and graph based CNNs perform learning on the manifold physical space or graph spectrum, but generality across topologies remains challenging.

In contrast to methods that operate on uniform sampling based representations such as voxel-based and view-based models, which suffer significant representational errors, we use analytical integration to precisely sample in the spectral domain to avoid sample aliasing errors.

Unlike graph spectrum based methods, our method naturally generalize across input data structures of varied topologies.

Using our representation, CNNs can be directly applied in the corresponding physical domain obtainable by inverse Fast Fourier Transform (FFT) due to the equivalence of the spectral and physical domains.

This allows for the use of powerful uniform Cartesian grid based CNN backbone architectures (such as DLA BID40 , ResNet (He et al., 2016) ) for the learning task on arbitrary geometrical signals.

Although the signal is defined on a simplex mesh, it is treated as a signal in the Euclidean space instead of on a graph, differentiating our framework from graph-based spectral learning techniques which have significant difficulties generalizing across topologies and unable to utilize state-of-the-art Cartesian CNNs.

We evaluate the effectiveness of our shape representation for deep learning tasks with three experiments: a controlled MNIST toy example, the 3D shape retrieval task, and a more challenging 3D point cloud to surface reconstruction task.

In a series of evaluations on different tasks, we show the unique advantages of this representation, and good potential for its application in a wider range of shape learning problems.

We achieve state-of-the-art performance among non-pre-trained models for the shape retrieval task, and beat state-of-the-art models for the surface reconstruction task.

The key contributions of our work are as follows:• We develop mathematical formulations for performing Fourier Transforms of signals defined on a simplex mesh, which generalizes and extends to all geometries in all dimensions.

(Sec.3)• We analytically show that our approach computes the frequency domain representation precisely, leading to much lower overall representational errors. (Sec. 3)• We empirically show that our representation preserves maximal shape information compared to commonly used binary voxel and SDF representations. (Sec. 4.1)• We show that deep learning models using CNNs in conjunction with our shape representation achieves state-of-the-art performance across a range of shape-learning tasks including shape retrieval (Sec. 4.2) and point to surface reconstruction (Sec. 4.3) DISPLAYFORM0 Index of the n-th element among a total of N elements Ω j n Domain of n-th element of order j x Cartesian space coordinate vector.

DISPLAYFORM1 Imaginary number unit Shape learning involves the learning of a mapping from input geometrical signals to desired output quantities.

The representation of geometrical signals is key to the learning process, since on the one hand the representation determines the learning architectures, and, on the other hand, the richness of information preserved by the representation acts as a bottleneck to the downstream learning process.

While data representation has not been an open issue for 2D image learning, it is far from being agreed upon in the existing literature for 3D shape learning.

The varied shape representations used in 3D machine learning are generally classified as multiview images BID28 BID26 BID2 , volumetric voxels BID38 BID14 BID37 BID0 , point clouds BID19 BID24 BID36 , polygonal meshes BID3 BID34 BID15 BID12 , shape primitives BID42 BID39 , and hybrid representations (Dai & Nießner, 2018) .Our proposed representation is closest to volumetric voxel representation, since the inverse Fourier Transform of the spectral signal in physical domain is a uniform grid implicit representation of the shape.

However, binary voxel representation suffers from significant aliasing errors during the uniform sampling step in the Cartesian space BID16 .

Using boolean values for de facto floating point numbers during CNN training is a waste of information processing power.

Also, the primitive-in-cell test for binarization requires arbitrary grouping in cases such as having multiple points or planes in the same cell BID32 .

Signed Distance Function (SDF) or Truncated Signed Distance Function (TSDF) Canelhas, 2017) provides localization for the shape boundary, but is still constrained to linear surface localization due to the linear interpolation process for recovering surfaces from grids.

Our proposed representation under Fourier basis can find nonlinear surface boundaries, achieving subgrid-scale accuracy (See FIG1 ).

Cartesian CNNs are the most ubiquitous and mature type of learning architecture in Computer Vision.

It has been thoroughly studied in a range of problems, including image recognition BID5 BID27 He et al., 2016) , object detection BID21 BID21 , and image segmentation BID11 He et al., 2017) .

In the spirit of 2D image-based deep learning, Cartesian CNNs have been widely used in shape learning models that adopted multiview shape representation BID28 BID26 BID2 BID29 BID17 BID33 .

Also, due to its straightforward and analogous extension to 3D by swapping 2D convolutional kernels with 3D counterparts, Cartesian CNNs have also been widely adopted in shape learning models using volumetric representations BID38 BID14 BID37 BID0 .

However, the dense nature of the operations makes it inefficient for sparse 3D shape signals.

To this end, improvements to Cartesian CNNs have been made using space partitioning tree structures, such as Quadtree in 2D and Octree in 3D BID2 BID31 .

These Cartesian CNNs can leverage backbone CNN architectures being developed in related computer vision problems and thus achieve good performance.

Since the physical domain representation in this study is based on Cartesian uniform grids, we directly use Cartesian CNNs.

Graph CNNs utilize input graph structure for performing graph convolutions.

They have been developed to engage with general graph structured data BID1 Henaff et al., 2015; Defferrard et al., 2016) .

BID39 used spectral CNNs with the eigenfunctions of the graph Laplacian as a basis.

However, the generality of this approach across topologies and geometris is still challenging since consistency in eigenfunction basis is implied.

Specially Designed Neural Networks have been used to perform learning on unconventional data structures.

For example, Qi et al. (2017a) designed a Neural Network architecture for points that achieves invariances using global pooling, with follow-up work (Qi et al., 2017b) using CNNsinspired hiearchical structures for more efficient learning.

BID13 performed convolution directly on the shape manifold and Cohen et al. FORMULA2 designed CNNs for the spherical domain and used it for 3D shapes by projecting the shapes onto the bounding sphere.

The original work on analytical expressions for Fourier transforms of 2D polygonal shape functions is given by BID6 .

Improved and simpler calculation methods have been suggested in Chu & Huang (1989) .

A 3D formulation is proposed by BID41 .

Theoretical analyses have been performed for the Fourier analysis of simplex domains BID30 BID7 ) and BID23 designed approximation methods for Fast Fourier Transform of polynomial functions defined on simplices.

Prisacariu & Reid (2011) describe shape with elliptic Fourier descriptors for level set-based segmentation and tracking.

There has also been a substantial literature on fast non-uniform Fourier transform methods for discretely sampled signal (Greengard & Lee, 2004 ).

However we are the first to provide a simple general expression for a j-simplex mesh, an algorithm to perform the transformation, and illustrations of their applicability to deep learning problems.

Almost all discrete geometric signals can be abstracted into weighted simplicial complexes.

A simplicial complex is a set composed of points, line segments, triangles, and their d-dimensional counterparts.

We call a simplicial complex consisting solely of j-simplices as a homogeneous simplicial j-complex, or a j-simplex mesh.

Most popular geometric representations that the research community is familiar with are simplex meshes.

For example, the point cloud is a 0-simplex mesh, the triangular mesh is a 2-simplex mesh, and the tetrahedral mesh is a 3-simplex mesh.

A j-simplex mesh consists of a set of individual elements, each being a j-simplex.

If signal is non-uniformly distributed over the simplex, we can define a piecewise constant j-simplex function over the j-simplex mesh.

We call this a weighted simplex mesh.

Each element has a distinct signal density.

J-simplex function For the n-th j-simplex with domain Ω j n , we define a density function f j n (x).

For example, for some Computer Vision and Graphics applications, a three-component density value can be defined on each element of a triangular mesh for its RGB color content.

For scientific applications, signal density can be viewed as mass or charge or other physical quantity.

DISPLAYFORM0 The piecewise-constant j-simplex function consisting of N simplices is therefore defined as the superposition of the element-wise simplex function.

Using the linearity of the integral in the Fourier transform, we can decompose the Fourier transform of the density function on the j-simplex mesh to be a weighted sum of the Fourier transform on individual j-simplices.

DISPLAYFORM1

We present a general formula for performing the Fourier transform of signal over a single j-simplex.

We provide detailed derivation and proof for j = 0, 1, 2, 3 in the supplemental material.

DISPLAYFORM0 We define γ j n to be the content distortion factor, which is the ratio of content between the simplex over the domain Ω j n and the unit orthogonal j-simplex.

Content is the j-dimensional analogy of the 3-dimensional volume.

The unit orthogonal j-simplex is defined as a j-simplex with one vertex at the Cartesian origin and all edges adjacent to the origin vertex to be pairwise orthogonal and to have unit length.

Therefore from Equation (2) the final general expression for computing the Fourier transform of a signal defined on a weighted simplex mesh is: DISPLAYFORM1 For computing the simplex content, we use the Cayley-Menger Determinant for a general expression: DISPLAYFORM2 For the matrixB j n , each entry d 2 mn represents the squared distance between nodes m and n. The matrix is of size (j + 2) × (j + 2) and is symmetrical.

Since the unit orthogonal simplex has content of C j I = 1/j!, the content distortion factor γ j n can be calculated by: DISPLAYFORM3 Auxiliary Node Method: Equation (3) provides a mean of computing the Fourier transform of a simplex with uniform signal density.

However, how do we compute the Fourier transform of polytopes (i.e., polygons in 2D, polyhedra in 3D) with uniform signal density efficiently?

Here, we introduce the auxiliary node method (AuxNode) that utilizes signed content for efficient computing.

We show that for a solid j-polytope represented by a watertight (j − 1)-simplex mesh, we can compute the Fourier transform of the entire polytope by traversing each of the elements in its boundary (j − 1)-simplex mesh exactly once BID41 ).The auxiliary node method performs Fourier transform over the the signed content bounded by an auxilliary node (a convenient choice being the origin of the Cartesian coordinate system) and each (j − 1)-simplex on the boundary mesh.

This forms an auxiliary j-simplex: DISPLAYFORM4 where N is the number of (j − 1)-simplices in the boundary mesh.

However due to the overlapping of these auxiliary j-simplices, we need a means of computing the sign of the transform for the overlapping regions to cancel out.

Equation (3) provides a general expression for computing the unsigned transform for a single j-simplex.

It is trivial to show that since the ordering of the nodes does not affect the determinant in Equation FORMULA6 , it gives the unsigned content value.

Therefore, to compute the Fourier transform of uniform signals in j-polytopes represented by its watertight (j − 1)-simplex mesh using the auxiliary node method, we modify Equation (3): s n γ j n is the signed content distortion factor for the n th auxiliary j-simplex where s n ∈ {−1, 1}. For practical purposes, assume that the auxiliary j-simplex is in R d where d = j. We can compute the signed content distortion factor using the determinant of the Jacobian matrix for parameterizing the auxiliary simplex to a unit orthogonal simplex: DISPLAYFORM5 DISPLAYFORM6 (a) Experiment setup 4 8 12 16 4 8 12 16 4 8 12 16 4 8 12 16 20 24 28 20 24 28 20 24 28Since this method requires the boundary simplices to be oriented, the right-hand rule can be used to infer the correct orientation of the boundary element.

For 2D polygons, it requires that the watertight boundary line mesh be oriented in a counter-clockwise fashion.

For 3D polytopes, it requires that the face normals of boundary triangles resulting from the right-hand rule be consistently outward-facing.

Algorithmic implementation: Several efficiencies can be exploited to achieve fast runtime and high robustness.

First, since the general expression Equation (4) involves division and is vulnerable to division-by-zero errors (that is not a singularity since it can be eliminated by taking the limit), add a minor random noise to vertex coordinates as well as to the k = 0 frequency mode for robustness.

Second, to avoid repeated computation, the value should be cached in memory and reused, but caching all σ and e −iσ values for all nodes and frequencies is infeasible for large mesh and/or high resolution output, thus the Breadth-First-Search (BFS) algorithm should be used to traverse the vertices for efficient memory management.

In this section, we will discuss the experiment setup, and we defer the details of our model architecture and training process to the supplementary material since it is not the focus of this paper.

We use the MNIST experiment as a first example to show that shape information in the input significantly affects the efficacy of the downstream learning process.

Since the scope of this research is on efficiently learning from nonuniform mesh-based representations, we compare our method with the state of the art in a slightly different scenario by treating MNIST characters as polygons.

We choose this experiment as a first toy example since it is easy to control the learning architecture and input resolution to highlight the effects of shape representation on deep learning performance.

We pre-process the original MNIST raw pixel images into polygons, which are represented by watertight line meshes on their boundaries.

The polygonized digits are converted into (n × n) binary pixel images and into distance functions by uniformly sampling the polygon at the (n × n) sample locations.

For NUFT, we first compute the lowest (n × n) Fourier modes for the polygonal shape function and then use an inverse Fourier transform to acquire the physical domain image.

We also compare the results with the raw pixel image downsampled to different resolutions, which serves as an oracle for information ceiling.

Then we perform the standard MNIST classification experiment on these representations with varying resolution n and with the same network architecture.

The experiment results are presented in FIG2 .

It is evident that binary pixel representation suffers the most information loss, especially at low resolutions, which leads to rapidly declining performance.

Using the distance function representation preserves more information, but underperforms our NUFT representation.

Due to its efficient information compression in the Spectral domain, NUFT even outperforms the downsampled raw pixel image at low resolutions.

Shape retrieval is a classic task in 3D shape learning.

SHREC17 BID24 which is based on the ShapeNet55 Core dataset serves as a compelling benchmark for 3D shape retrieval performance.

We compare the retrieval performance of our model utilizing the NUFT-surface (NUFT-S) and NUFT-volume (NUFT-V) at various resolutions against state-of-the-art shape retrieval algorithms to illustrate its potential in a range of 3D learning problems.

We performed the experiments on the normalized dataset in SHREC17.

Our model utilizes 3D DLA BID40 as a backbone architecture.

Results: Results from the experiment are tabulated in TAB1 .

For the shape retrieval task, most state-of-the-art methods are based on multi-view representation that utilize a 2D CNN pretrained on additional 2D datasets such as ImageNet.

We have achieved results on par with, though not better than, state-of-the-art pretrained 2D models.

We outperform other models in this benchmark that have not been pre-trained on additional data.

We also compared NUFT-volume and NUFT-surface representations in FIG3 .

Interestingly NUFT-volume and NUFT-surface representations lead to similar performances under the same resolution.

Table 3 : Quantitative comparison of surface reconstruction methods.

The metrics above are distances, hence lower value represents better performance.

Best result is highlighted in bold.

We achieve better results than the current stateof-the-art method by a sizable margin, and our results are robust to noise in the input.

We further illustrate the advantages of our representation with a unique yet important task in computational geometry that has been challenging to address with conventional deep learning techniques: surface reconstruction from point cloud.

The task is challenging for deep learning in two aspects: First, it requires input and output signals of different topologies and structures (i.e., input being a point cloud and output being a surface).

Second, it requires precise localization of the signal in 3D space.

Using our NUFT-point representation as input and NUFT-surface representation as output, we can frame the task as a 3D image-to-image translation problem, which is easy to address using analogous 2D techniques in 3D.

We use the U-Net BID22 architecture and train it with a single L 2 loss between output and ground truth NUFT-surface representation.

Experiment Setup: We train and test our model using shapes from three categories of ShapeNet55 Core dataset (car, sofa, bottle).

We trained our model individually for these three categories.

As a pre-processing step we removed faces not visible from the exterior and simplified the mesh for faster conversion to NUFT representation.

For the input point cloud, we performed uniform point sampling of 3000 points from each mesh and converted the points into the NUFT-point representation (128 3 ).

Then, we converted the triangular mesh into NUFT-surface representation (128 3 ).

At test time, we post-process the output NUFT-surface implicit function by using the marching cubes algorithm to extract 0.5 contours.

Since the extracted mesh has thickness, we further shrink the mesh by moving vertices to positions with higher density values while preserving the rigidity of each face.

Last but not least, we qualitatively and quantitatively compare the performance by showing the reconstructed mesh against results from the traditional Poisson Surface Reconstruction (PSR) method BID4 at various tree depths (5 and 8) and the Deep Marching Cubes (DMC) algorithm BID9 .

For quantitative comparison, we follow the literature BID25 and use Chamfer distance, Accuracy and Completeness as the metrics for comparison.

For comparison with BID9 , we also test the model with noisy inputs (Gaussian of sigma 0.15 voxel-length under 32 resolution), computed distance metrics after normalizing the models to the range of (0, 32).

Table 3 for quantitative comparisons with competing algorithms on the same task, and Figures 5 and 6 for visual comparisons.

GT stands for Ground Truth.

We achieve new state-of-the-art in the point to surface reconstruction task, due to the good localization properties of the NUFT representations and its flexibility across geometry topologies.

We present a general representation for multidimensional signals defined on simplicial complexes that is versatile across geometrical deep learning tasks and maximizes the preservation of shape information.

We develop a set of mathematical formulations and algorithmic tools to perform the transformations efficiently.

Last but not least, we illustrate the effectiveness of the NUFT representation with a well-controlled example (MNIST polygon), a classic 3D task (shape retrieval) and a difficult and mostly unexplored task by deep learning (point to surface reconstruction), achieving new state-of-the-art performance in the last task.

In conclusion, we offer an alternative representation for performing CNN based learning on geometrical signals that shows great potential in various 3D tasks, especially tasks involving mixed-topology signals.

Without loss of generality assume that the j-simplex is defined in R d space where d ≥ j, since it is not possible to define a j-simplex in a space with dimensions lower than j. For most cases below (except j = 0) we will parameterize the original simplex domain to a unit orthogonal simplex in R j (as shown in Figure 7 ).

Denote the original coordinate system in R d as x and the new coordinate system in the parametric R j space as p. Choose the following parameterization scheme: DISPLAYFORM0 By performing the Fourier transform integral in the parametric space and restoring the results by the content distortion factor γ j n we can get equivalent results as the Fourier transform on the original simplex domain.

Content is the generalization of volumes in arbitrary dimensions (i.e. unity for points, length for lines, area for triangles, volume for tetrahedron).

Content distortion factor γ j n is the ratio between the content of the original simplex and the content of the unit orthogonal simplex in parametric space.

The content is signed if switching any pair of nodes in the simplex changes the sign of the content, and it is unsigned otherwise.

See subsection 3.2 for means of computing the content and the content distortion factor.

Points have spatial position x but no size (length, area, volume), hence it can be mathematically modelled as a delta function.

The delta function (or Dirac delta function) has point mass as a function equal to zero everywhere except for zero and its integral over entire real space is one, i.e.: DISPLAYFORM0 For unit point mass at location x n : DISPLAYFORM1 Indeed, for 0-simplex, we have recovered the definition of the Discrete Fourier Transform (DFT).

For a line with vertices at location x 1 , x 2 ∈ R d , by parameterizing it onto a unit line, we get: DISPLAYFORM0 = iγ Figure 7 : Schematic of example for 2-simplex.

Original j-simplex is parameterized to a unit orthogonal j-simplex in R j space for performing the integration.

Parameterization incurs a content distortion factor γ j n which is the ratio between the original simplex and the unit-orthogonal simplex in parametric space.

DISPLAYFORM1

For a triangle with vertices x 1 , x 2 , x 3 ∈ R d , parameterization onto a unit orthogonal triangle gives: DISPLAYFORM0 = γ 2 n e −iω·x3 1 0 dp e −ipω·(x1−x3) DISPLAYFORM1 A.4 TETRAHEDRON: 3-SIMPLEX For a tetrahedron with vertices x 1 , x 2 , x 3 , x 4 ∈ R d , parameterization onto a unit orthogonal tetrahedron gives: DISPLAYFORM2

Besides evaluating and comparing the shape representation schemes in the context of machine learning problems, we evaluate the different representation schemes in terms of its geometrical shape information.

To evaluate geometrical shape information, we first convert the original polytopal shapes into the corresponding representations and then reconstruct the shape from these representations using interpolation based upsampling followed by contouring methods.

Finally, we use the mesh Intersection over Union (mesh-IoU) metric between the original and constructed mesh to quantify geometrical shape information.

Mesh boolean and volume computation for 2D polygons and 3D triangular mesh can be efficiently performed with standard computational geometry methods.

In three dimensions contouring and mesh extraction can be performed using marching cubes algorithm for mesh reconstruction.

For binarized representation, we perform bilinear upsampling (which does not affect the final result) followed by 0.5-contouring.

For our NUFT representation, we perform spectral domain upsampling (which corresponds to zero-padding of higher modes in spectral domain), followed by 0.

Training Details: We train the model with batch size of 64, learning rate of 0.01 and learning rate step size of 10.

We use the SGD optimizer with momentum of 0.9, weight decay of 1 × 10 −4 for 15 epochs.

Model Architecture: We use DLA34 architecture with all 2D convolutions modified to be 3D convolutions.

It consists of [1,1,1,2,2,1] levels, with [16, 32, 64, 128, 256, 512] filter numers.

Training Details We train the model with batch size of 64, learning rate of 1 × 10 −3 , learning rate step size of 30.

We use the Adam optimizer with momentum of 0.9, and weight decay of 1 × 10 −4 for 40 epochs.

Model Architecture We use a modified 3D version of the U-Net architecture consisting of 4 down convolutions and 4 up-convolutions with skip layers.

Number of filters for down convolutions are [32, 64, 128, 256] and double of that for up convolutions.

Training Details We train the model using Adam optimizer with learning rate 3 × 10 −4 for 200 epochs.

We use NUFT point representation as input and a single L 2 loss between output and ground truth (NUFT surface) to train the network.

We train and evaluate the model at 128 3 resolution.

@highlight

We use non-Euclidean Fourier Transformation of shapes defined by a simplicial complex for deep learning, achieving significantly better results than point-based sampling techiques used in current 3D learning literature.