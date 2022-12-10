Convolution is an efficient technique to obtain abstract feature representations using hierarchical layers in deep networks.

Although performing convolution in Euclidean geometries is fairly straightforward, its extension to other topological spaces---such as a sphere S^2 or a unit ball B^3---entails unique challenges.

In this work, we propose a novel `"volumetric convolution" operation that can effectively convolve arbitrary functions in B^3.

We develop a theoretical framework for "volumetric convolution" based on Zernike polynomials and efficiently implement it as a differentiable and an easily pluggable layer for deep networks.

Furthermore, our formulation leads to derivation of a  novel formula to measure the symmetry of a function in B^3 around an arbitrary axis, that is useful in 3D shape analysis tasks.

We demonstrate the efficacy of proposed volumetric convolution operation on a possible use-case i.e., 3D object recognition task.

Convolution-based deep neural networks have performed exceedingly well on 2D representation learning tasks BID11 BID7 .

The convolution layers perform parameter sharing to learn repetitive features across the spatial domain while having lower computational cost by using local neuron connectivity.

However, most state-of-the-art convolutional networks can only work on Euclidean geometries and their extension to other topological spaces e.g., spheres, is an open research problem.

Remarkably, the adaptation of convolutional networks to spherical domain can advance key application areas such as robotics, geoscience and medical imaging.

Some recent efforts have been reported in the literature that aim to extend convolutional networks to spherical signals.

Initial progress was made by BID1 , who performed conventional planar convolution with a careful padding on a spherical-polar representation and its cube-sphere transformation BID17 .

A recent pioneering contribution by used harmonic analysis to perform efficient convolution on the surface of the sphere (S 2 ) to achieve rotational equivariance.

These works, however, do not systematically consider radial information in a 3D shape and the feature representations are learned at specified radii.

Specifically, estimated similarity between spherical surface and convolutional filter in S 2 , where the kernel can be translated in S 2 .

Furthermore, BID23 recently solved the more general problem of SE(3) equivariance by modeling 3D data as dense vector fields in 3D Euclidean space.

In this work however, we focus on B 3 to achieve the equivariance to SO(3).In this paper, we propose a novel approach to perform volumetric convolutions inside unit ball (B 3 ) that explicitly learns representations across the radial axis.

Although we derive generic formulas to convolve functions in B 3 , we experiment on one possible use case in this work, i.e., 3D shape recognition.

In comparison to closely related spherical convolution approaches, modeling and convolving 3D shapes in B 3 entails two key advantages: 'volumetric convolution' can capture both 2D texture and 3D shape features and can handle non-polar 3D shapes.

We develop the theory of volumetric convolution using orthogonal Zernike polynomials BID3 , and use careful approximations to efficiently implement it using low computational-cost matrix multiplications.

Our experimental results demonstrate significant boost over spherical convolution and that confirm the high discriminative ability of features learned through volumetric convolution.

Furthermore, we derive an explicit formula based on Zernike Polynomials to measure the axial symmetry of a function in B 3 , around an arbitrary axis.

While this formula can be useful in many function analysis tasks, here we demonstrate one particular use-case with relevance to 3D shape recognition.

Specifically, we use the the derived formula to propose a hand-crafted descriptor that accurately encodes the axial symmetry of a 3D shape.

Moreover, we decompose the implementation of both volumetric convolution and axial symmetry measurement into differentiable steps, which enables them to be integrated to any end-to-end architecture.

Finally, we propose an experimental architecture to demonstrate the practical usefulness of proposed operations.

We use a capsule network after the convolution layer as it allows us to directly compare feature discriminability of spherical convolution and volumetric convolution without any bias.

In other words, the optimum deep architecture for spherical convolution may not be the same for volumetric convolution.

Capsules, however, do not deteriorate extracted features and the final accuracy only depends on the richness of input shape features.

Therefore, a fair comparison between spherical and volumetric convolutions can be done by simply replacing the convolution layer.

It is worth pointing out that the proposed experimental architecture is only a one possible example out of many possible architectures, and is primarily focused on three factors: 1) Capture useful features with a relatively shallow network compared to state-of-the-art.

2) Show richness of computed features through clear improvements over spherical convolution.

3) Demonstrate the usefulness of the volumetric convolution and axial symmetry feature layers as fully differentiable and easily pluggable layers, which can be used as building blocks for end-to-end deep architectures.

The main contributions of this work include:• Development of the theory for volumetric convolution that can efficiently model functions in B 3 .• Implementation of the proposed volumetric convolution as a fully differentiable module that can be plugged into any end-to-end deep learning framework.• The first approach to perform volumetric convolution on 3D objects that can simultaneously model 2D (appearance) and 3D (shape) features.• A novel formula to measure the axial symmetry of a function defined in B 3 , around an arbitrary axis using Zernike polynomials.• An experimental end-to-end trainable framework that combines hand-crafted feature representation with automatically learned representations to obtain rich 3D shape descriptors.

The rest of the paper is structured as follows.

In Sec. 2 we introduce the overall problem and our proposed solution.

Sec. 3 presents an overview of 3D Zernike polynomials.

Then, in Sec. 4 and Sec. 5 we derive the proposed volumetric convolution and axial symmetry measurement formula respectively.

Sec. 6.2 presents our experimental architecture, and in Sec. 7 we show the effectiveness of the derived operators through extensive experiments.

Finally, we conclude the paper in Sec. 8.

Convolution is an effective method to capture useful features from uniformly spaced grids in R n , within each dimension of n, such as gray scale images (R 2 ), RGB images (R 3 ), spatio-temporal data (R 3 ) and stacked planar feature maps (R n ).

In such cases, uniformity of the grid within each dimension ensures the translation equivariance of the convolution.

However, for topological spaces such as S 2 and B 3 , it is not possible to construct such a grid due to non-linearity.

A naive approach to perform convolution in B 3 would be to create a uniformly spaced three dimensional grid in (r, θ, φ) coordinates (with necessary padding) and perform 3D convolution.

However, the spaces between adjacent points in each axis are dependant on their absolute position and hence, modeling such a space as a uniformly spaced grid is not accurate.

To overcome these limitations, we propose a novel volumetric convolution operation which can effectively perform convolution on functions in B 3 .

It is important to note that ideally, the convolution in B 3 should be a signal on both 3D rotation group and 3D translation.

However, since Zernike polynomials do not have the necessary properties to automatically achieve translation equivariance, we stick to 3D rotation group in this work and refer to this operation as convolution from here onwards.

FIG0 shows the analogy between planar convolution and volumetric convolution.

In Sec. 3, we present an overview of 3D Zernike polynomials that will be later used in Sec. 4 to develop volumetric convolution operator.

3D Zernike polynomials are a complete and orthogonal set of basis functions in B 3 , that exhibits a 'form invariance' property under 3D rotation BID3 .

A (n, l, m) th order 3D Zernike basis function is defined as, DISPLAYFORM0 where R n,l is the Zernike radial polynomial (Appendix l] and n − l is even.

Since 3D Zernike polynomials are orthogonal and complete in B 3 , an arbitrary function f (r, θ, φ) in B 3 can be approximated using Zernike polynomials as follows.

DISPLAYFORM1 DISPLAYFORM2 where Ω n,l,m (f ) could be obtained using, DISPLAYFORM3 where † denotes the complex conjugate.

In Sec. 4, we will derive the proposed volumetric convolution.

When performing convolution in B 3 , a critical problem which arises is that several rotation operations exist for mapping a point p to a particular point p .

For example, using Euler angles, we can decompose a rotation into three rotation operations R(θ, φ) = R(θ) y R(φ) z R(θ) y , and the first rotation R(θ) y can differ while mapping p to p (if y is the north pole).

However, if we enforce the kernel function to be symmetric around y, the function of the kernel after rotation would only depend on p and p .

This observation is important for our next derivations because we can then uniquely define a 3D rotation on kernel in terms of azimuth and polar angles.

Let the kernel be symmetric around y and f (θ, φ, r), g(θ, φ, r) be the functions of object and kernel respectively.

Then we define volumetric convolution as, DISPLAYFORM0 where τ (α,β) is an arbitrary rotation, that aligns the north pole with the axis towards (α, β) direction (α and β are azimuth and polar angles respectively).

Eq. 4 is able to capture more complex patterns compared to spherical convolution due to two reasons: 1) the inner product integrates along the radius and 2) the projection onto spherical harmonics forces the function into a polar function, that can result in information loss.

In Sec. 4.1 we derive differentiable relations to compute 3D Zernike moments for functions in B 3 .

Instead of using Eq. 3, we derive an alternative method to obtain the set {Ω n,l,m }.

The motivations are two fold: 1) ease of computation and 2) the completeness property of 3D Zernike Polynomials ensures that lim n→∞ f − n l m Ω n,l,m Z n,l,m = 0 for any arbitrary function f .

However, since n should be finite in the implementation, aforementioned property may not hold, leading to increased distance between the Zernike representation and the original shape.

Therefore, minimizing the recon- DISPLAYFORM0 m Ω n,l,m Z n,l,m , pushes the set {Ω n,l,m } inside frequency space, where {Ω n,l,m } has a closer resemblance to the corresponding shape.

Following this conclusion, we derive the following method to obtain {Ω n,l,m }.

In planar convolution the kernel translates and inner product between the image and the kernel is computed in (x, y) plane.

In volumetric convolution a 3D rotation is applied to the kernel and the inner product is computed between 3D function and 3D kernel over B 3 .Since DISPLAYFORM1 and hence approximate Eq. 2 as, DISPLAYFORM2 where Re{Z n,l,m } and Img{Z n,l,m } are real and imaginary components of Z n,l,m respectively.

In matrix form, this can be rewritten as, DISPLAYFORM3 where c is the set of 3D Zernike moments Ω n,l,m .

Eq. 6 can be interpreted as an overdetermined linear system, with the set Ω n,l,m as the solution.

To find the least squared error solution to the Eq. 6 we use the pseudo inverse of X. Since this operation has to be differentiable to train the model end-to-end, a common approach like singular value decomposition cannot be used here.

Instead, we use an iterative method to calculate the pseudo inverse of a matrix BID12 .

It has been shown that V n converges to A + where A + is the Moore-Penrose pseudo inverse of A if, DISPLAYFORM4 for a suitable initial approximation V 0 .

They also showed that a suitable initial approximation would be V 0 = αA T with 0 < α < 2/ρ(AA T ), where ρ(·) denotes the spectral radius.

Empirically, we choose α = 0.001 in our experiments.

Next, we derive the theory of volumetric convolution within the unit ball.

3 USING 3D ZERNIKE POLYNOMIALSWe formally present our derivation of volumetric convolution using the following theorem.

A short version of the proof is then provided.

Please see Appendix A for the complete derivation.

Theorem 1: Suppose f, g : X −→ R 3 are square integrable complex functions defined in B 3 so that f, f < ∞ and g, g < ∞. Further, suppose g is symmetric around north pole and DISPLAYFORM0 th 3D Zernike moment of f , (n, l, 0) th 3D Zernike moment of g, and spherical harmonics function respectively.

Proof: Completeness property of 3D Zernike Polynomials ensures that it can approximate an arbitrary function in B 3 , as shown in Eq. 2.

Leveraging this property, Eq. 4 can be rewritten as, DISPLAYFORM1 However, since g(θ, φ, r) is symmetric around y, the rotation around y should not change the function.

This ensures, g(r, θ, φ) = g(r, θ − α, φ)and hence, DISPLAYFORM2 This is true, if and only if m = 0.

Therefore, a symmetric function around y, defined inside the unit sphere can be rewritten as, DISPLAYFORM3 which simplifies Eq. 9 to, DISPLAYFORM4 Using the properties of inner product, Eq. 13 can be rearranged as, DISPLAYFORM5 Using the rotational properties of Zernike polynomials, we obtain (see Appendix A for our full derivation), DISPLAYFORM6 Since we can calculate Ω n,l,m (f ) and Ω n,l,0 (g) easily using Eq. 6, f * g(θ, φ) can be found using a simple matrix multiplication.

It is interesting to note that, since the convolution kernel does not translate, the convolution produces a polar shape, which can be further convolved-if needed-using the DISPLAYFORM7 are the (l, m) th frequency components of f and g in spherical harmonics space.

Next, we present a theorem to show the equivariance of volumetric convolution with respect to 3D rotation group.

One key property of the proposed volumetric convolution is its equivariance to 3D rotation group.

To demonstrate this, we present the following theorem.

Theorem 1: Suppose f, g : X −→ R 3 are square integrable complex functions defined in B 3 so that f, f < ∞ and g, g < ∞. Also, let η α,β,γ be a 3D rotation operator that can be decomposed into three Eular rotations R y (α)R z (β)R y (γ) and τ α,β another rotation operator that can be decomposed into R y (α)R z (β).

Suppose η α,β,γ (g) = τ α,β (g).

Then, η (α,β,γ) (f ) * g(θ, φ) = τ (α,β) (f * g)(θ, φ), where * is the volumetric convolution operator.

The proof to our theorem can be found in Appendix B. The intuition behind the theorem is that if a 3D rotation is applied to a function defined in B 3 Hilbert space, the output feature map after volumetric convolution exhibits the same rotation.

The output feature map however, is symmetric around north pole, hence the rotation can be uniquely defined in terms of azimuth and polar angles.

In this section we present the following proposition to obtain the axial symmetry measure of a function in B 3 , around an arbitrary axis using 3D Zernike polynomials.

which allows encoding non-polar 3D shapes with texture.

In contrast, spherical convolution is performed in S 2 that can handle only polar 3D shapes with uniform texture.

Proposition: Suppose g : X −→ R 3 is a square integrable complex function defined in B 3 such that g, g < ∞. Then, the power of projection of g in to S = {Z i } where S is the set of Zernike basis functions that are symmetric around an axis towards (α, β) direction is given by, DISPLAYFORM0 where α and β are azimuth and polar angles respectively.

The proof to our proposition is given in Appendix C.6 A CASE STUDY: 3D OBJECT RECOGNITION

A 2D image is a function on Cartesian plane, where a unique value exists for any (x, y) coordinate.

Similarly, a polar 3D object can be expressed as a function on the surface of the sphere, where any direction vector (θ, φ) has a unique value.

To be precise, a 3D polar object has a boundary function in the form of f : DISPLAYFORM0 Translation of the convolution kernel on (x, y) plane in 2D case, extends to movements on the surface of the sphere in S 2 .

If both the object and the kernel have polar shapes, this task can be tackled by projecting both the kernel and the object onto spherical harmonic functions (Appendix E).

However, this technique suffers from two drawbacks.

1) Since spherical harmonics are defined on the surface of the unit sphere, projection of a 3D shape function into spherical harmonics approximates the object to a polar shape, which can cause critical loss of information for non-polar 3D shapes.

This is frequently the case in realistic scenarios.2) The integration happens over the surface of the sphere, which is unable to capture patterns across radius.

These limitations can be addressed by representing and convolving the shape function inside the unit ball (B 3 ).

Representing the object function inside B 3 allows the function to keep its complex shape information without any deterioration since each point is mapped to unique coordinates (r, θ, φ), where r is the radial distance, θ and φ are azimuth and polar angles respectively.

Additionally, it allows encoding of 2D texture information simultaneously.

FIG1 compares volumetric convolution and spherical convolution.

Since we conduct experiments only on 3D objects with uniform surface values, in this work we use the following transformation to apply a simple surface function f (θ, φ, r) to the 3D objects: DISPLAYFORM1

We implement an experimental architecture to demonstrate the usefulness of the proposed operations.

While these operations can be used as building-tools to construct any deep network, we focus on three key factors while developing the presented experimental architecture: 1) Shallowness: Volumetric convolution should be able to capture useful features compared to other methodologies with less number of layers.

2) Modularity: The architecture should have a modular nature so that a fair comparison can be made between volumetric and spherical convolution.

We use a capsule network after the convolution layer for this purpose.

Figure 3: Experimental architecture: An object is first mapped to three view angles.

For each angle, axial symmetry and volumetric convolution features are generated for P + and P − .

These two features are then separately combined using compact bilinear pooling.

Finally, the features are fed to two individual capsule networks, and the decisions are max-pooled.

of axial symmetry features as a hand-crafted and fully differentiable layer.

The motivation is to demonstrate one possible use case of axial symmetry measurements in 3D shape analysis.

The proposed architecture consists of four components.

First, we obtain three view angles, and later generate features for each view angle separately.

We optimize the view angles to capture complimentary shape details such that the total information content is maximized.

For each viewing angle 'k', we obtain two point sets P + k and P − k consisting of tuples denoted as: DISPLAYFORM0 such that y denotes the horizontal axis.

Second, the six point sets are volumetrically convolved with kernels to capture local patterns of the object.

The generated features for each point set are then combined using compact bilinear pooling.

Third, we use axial symmetry measurements to generate additional features.

The features that represent each point set are then combined using compact bilinear pooling.

Fourth, we feed features from second and third components of the overall architecture to two independent capsule networks and combine the outputs at decision level to obtain the final prediction.

The overall architecture of the proposed scheme is shown in FIG2

We use three view angles to generate features for better representation of the object.

First, we translate the center of mass of the set of (x, y, z) points to the origin.

The goal of this step is to achieve a general translational invariance, which allows us to free the convolution operation from the burden of detecting translated local patterns.

Subsequently, the point set is rearranged as an ordered set on x and z and a 1D convolution net is applied on y values of the points.

Here, the objective is to capture local variations of points along the y axis, since later we analyze point sets P + and P − independently.

The trained filters can be assumed to capture properties similar to ∂ n y/∂x n and ∂ n y/∂z n , where n is the order of derivative.

The output of the 1D convolution net is rotation parameters represented by a 1 × 9 vector r = {r 1 , r 2 , · · · , r 9 }.

Then, we compute R 1 = R x (r 1 )R y (r 2 )R z (r 3 ), R 2 = R x (r 4 )R y (r 5 )R z (r 6 ) and R 3 = R x (r 7 )R y (r 8 )R z (r 9 ) where R 1 , R 2 and R 3 are the rotations that map the points to three different view angles.

After mapping the original point set to three view angles, we extract the P + k and P − k point sets from each angle k that gives us six point sets.

These sets are then fed to the volumetric convolution layer to obtain feature maps for each point set.

We then measure the symmetry around four equi-angular axes using Eq. 16, and concatenate these measurement values to form a feature vector for the same point sets.

Compact bilinear pooling (CBP) provides a compact representation of the full bilinear representation, but has the same discriminative power.

The key advantage of compact bilinear pooling is the significantly reduced dimensionality of the pooled feature vector.

We first concatenate the obtained volumetric convolution features of the three angles, for P + and P − separately to establish two feature vectors.

These two features are then fused using compact bilinear pooling BID5 .

The same approach is used to combine the axial symmetry features.

These fused vectors are fed to two independent capsule nets.

Furthermore, we experiment with several other feature fusion techniques and present results in Sec. 7.2.

Capsule Network (CapsNet) BID18 brings a new paradigm to deep learning by modeling input domain variations through vector based representations.

CapsNets are inspired by so-called inverse graphics, i.e., the opposite operation of image rendering.

Given a feature representation, CapsNets attempt to generate the corresponding geometrical representation.

The motivation for using CapsNets in the network are twofold: 1) CapsNet promotes a dynamic 'routing-by-agreement' approach where only the features that are in agreement with high-level detectors are routed forward.

This property of CapsNets does not deteriorate extracted features and the final accuracy only depends on the richness of original shape features.

It allows us to directly compare feature discriminability of spherical and volumetric convolution without any bias.

For example, using multiple layers of volumetric or spherical convolution hampers a fair comparison since it can be argued that the optimum architecture may vary for two different operations.

2) CapsNet provides an ideal mechanism for disentangling 3D shape features through pose and view equivariance while maintaining an intrinsic co-ordinate frame where mutual relationships between object parts are preserved.

Inspired by these intuitions, we employ two independent CapsNets in our network for volumetric convolution features and axial symmetry features.

In this layer, we rearrange the input feature vectors as two sets of primary capsules-for each capsule net-and use the dynamic routing technique proposed by BID18 to predict the classification results.

The outputs are then combined using max-pooling, to obtain the final classification result.

For volumetric convolution features, our architecture uses 1000 primary capsules with 10 dimensions each.

For axial symmetry features, we use 2500 capsules, each with 10 dimensions.

In both networks, decision layer consist of 12 dimensional capsules.

We use n = 5 to implement Eq. 15 and three iterations to calculate the Moore-Penrose pseudo inverse using Eq. 7.

We use a decaying learning rate lr = 0.1 × 0.9 g step 3000 , where g step is incremented by one per each iteration.

For training, we use the Adam optimizer with β 1 = 0.9, β 2 = 0.999, = 1 × 10 DISPLAYFORM0 where parameters refer to the usual notation.

All these values are chosen empirically.

Since we have decomposed the theoretical derivations into sets of low-cost matrix multiplications, specifically aiming to reduce the computational complexity, the GPU implementation is highly efficient.

For example, the model takes less than 15 minutes for an epoch during the training phase for ModelNet10, with a batchsize 2, on a single GTX 1080Ti GPU.

In this section, we discuss and evaluate the performance of the proposed approach.

We first compare the accuracy of our model with relevant state-of-the-art work, and then present a thorough ablation study of our model, that highlights the importance of several architectural aspects.

We use ModelNet10 and ModelNet40 datasets in our experiments.

Next, we evaluate the robustness of our approach against loss of information and finally show that the proposed approach for computing 3D Zernike moments produce richer representations of 3D shapes, compared to the conventional approach.

TAB1 illustrates the performance comparison of our model with state-of-the-art.

The model attains an overall accuracy of 92.17% on ModelNet10 and 86.5% accuracy on ModelNet40, which is on par with state-of-the-art.

We do not compare with other recent work, such as BID9 BID2 45Conv 90M 93.11% 90.8% Pairwise BID8 23Conv 143M 92.8% 90.7% MVCNN BID22 60Conv + 36FC 200M -90.1% Ours 3Conv + 2Caps 4.4M 92.17% 86.5% PointNet BID16 2ST + 5Conv 80M -86.2% ECC BID21 4Conv + 1FC --83.2% DeepPano BID20 4Conv + 3FC -85.45% 77,63% 3DShapeNets BID25 4-3DConv + 2FC 38M 83.5% 77% PointNet BID6 2Conv + 2FC 80M 77.6% - ModelNet10 and ModelNet40.

These are not comparable with our proposed approach, as we propose a shallow, single model without any data augmentation, with a relatively low number of parameters.

Furthermore, our model reports these results by using only a single volumetric convolution layer for learning features.

FIG4 demonstrates effectiveness of our architecture by comparing accuracy against the number of trainable parameters in state-of-the-art models.

TAB3 depicts the performance comparison between several variants of our model.

To highlight the effectiveness of the learned optimum view points, we replace the optimum view point layer with three fixed orthogonal view points.

This modification causes an accuracy drop of 6.57%, emphasizing that the optimum view points indeed depends on the shape.

Another interestingperhaps the most important-aspect to study is the performance of the proposed volumetric convolution against spherical convolution.

To this end, we replace the volumetric convolution layer of our model with spherical convolution and compare the results.

It can be seen that our volumetric convolution scheme outperforms spherical convolution by a significant margin of 12.56%, indicating that volumetric convolution captures shape properties more effectively.

Furthermore, using mean-pooling instead of maxpooling, at the decision layer drops the accuracy to 87.27%.

We also evaluate performance of using a single capsule net.

In this scenario, we combine axial symmetry features with volumetric convolution features using compact bilinear pooling (CBP), and feed it a single capsule network.

This variant achieves an overall accuracy of 86.53%, is a 5.64% reduction in accuracy compared to the model with two capsule networks.

Moreover, we compare the performance of two feature categories-volumetric convolution features and axial symmetry features-individually.

Axial symmetry features alone are able to obtain an accuracy of 66.73%, while volumetric convolution features reach a significant 85.3% accuracy.

On the contrary, spherical convolution attains an accuracy of 71.6%, which again highlights the effectiveness of volumetric convolution.

Then we compare between different choices that can be applied to the experimental architecture.

We first replace the capsule network with a fully connected layer and achieve an accuracy of 87.3%.

This is perhaps because capsules are superior to a simple fully connected layer in modeling view- point invariant representations.

Then we try different substitutions for compact bilinear pooling and achieve 90.7%, 90.3% and 85.3% accuracies respectively for feature concatenation, max-pooling and average-pooling.

This justifies the choice of compact bilinear pooling as a feature fusion tool.

However, it should be noted that these choices may differ depending on the architecture.

One critical requirement of a 3D object classification task is to be robust against various information loss.

To demonstrate the effectiveness of our proposed features in this aspect, we randomly remove data points from the objects in validation set, and evaluate model performance.

The results are illustrated in Fig. 5 .

The model shows no performance loss until 20% of the data is lost, and only gradually drops to an accuracy level of 66.5 at a 50% data loss, which implies strong robustness against random information loss.

In Sec. 4.1, we proposed an alternative method to calculate 3D Zernike moments (Eq. 5, 6), instead of the conventional approach (Eq. 3).

We hypothesized that moments obtained using the former has a closer resemblance to the original shape, due to the impact of finite number of frequency terms.

In this section, we demonstrate the validity of our hypothesis through experiments.

To this end, we compute moments for the shapes in the validation set of ModelNet10 using both approaches, and compare the mean reconstruction error defined as: DISPLAYFORM0 where T is the total number of points and t ∈ S 3 .

Fig. 6 shows the results.

In both approaches, the mean reconstruction error decreases as n increases.

However, our approach shows a significantly low mean reconstruction error of 0.0467% at n = 5 compared to the conventional approach, which has a mean reconstruction error of 0.56% at same n.

This result also justifies the utility of Zernike moments for modeling complex 3D shapes.

In this work, we derive a novel 'volumetric convolution' using 3D Zernike polynomials, which can learn feature representations in B 3 .

We develop the underlying theoretical foundations for volumetric convolution and demonstrate how it can be efficiently computed and implemented using low-cost matrix multiplications.

Furthermore, we propose a novel, fully differentiable method to measure the axial symmetry of a function in B 3 around an arbitrary axis, using 3D Zernike polynomials.

Finally, using these operations as building tools, we propose an experimental architecture, that gives competitive results to state-of-the-art with a relatively shallow network, in 3D object recognition task.

An immediate extension to this work would be to explore weight sharing along the radius of the sphere.

Let f (θ, φ, r) and g(θ, φ, r) be the object function and kernel function (symmetric around north pole) respectively.

Then volumetric convolution is defined as, f * g(θ, φ) =< f, τ (θ,φ) g >Applying the rotation η (α,β,γ) to f , we get, η (α,β,γ) (f ) * g(θ, φ) =< η (α,β,γ) (f ), τ (θ,φ) g >By the result 33, we have, DISPLAYFORM0 However, since η α,β,γ (g) = τ α,β (g) we get, η (α,β,γ) (f ) * g(θ, φ) =< f, τ (θ−α,φ−β,) g >We know that, f * g(θ, φ) =< f, τ (θ,φ) g >= η (α,β,γ) (f ) * g(θ, φ) = (f * g)(θ − α, φ − β)η (α,β,γ) (f ) * g(θ, φ) = τ (α,β) (f * g) (Hence, we achieve equivariance over 3D rotations.

3 AROUND AN ARBITRARY AXIS.Proposition: Suppose g : X −→ R 3 is a square integrable complex function defined in B 3 such that g, g < ∞. Then, the power of projection of g in to S = {Z i } where S is the set of Zernike basis functions that are symmetric around an axis towards (α, β) direction is given by, where α and β are azimuth and polar angles respectively.

Proof: The subset of complex functions which are symmetric around north pole is S = {Z n,l,0 }.

Therefore, projection of a function into S gives, sym y (θ, φ) = n n l=0 f, Z n,l,0 z n,l,0 (θ, φ)To obtain the symmetry function around any axis which is defined by (α, β), we rotate the function by (−α, −β), project into S, and final compute the power of the projection.sym (α,β) (θ, φ) = n,l τ (−α,−β) (f ), Z n,l,0 z n,l,0 (θ, φ)For any rotation operator U , and for any two points defined on a complex Hilbert space, x and y, U (x), U (y) H = x, y HD FUNCTION DEFINITIONS

@highlight

A novel convolution operator for automatic representation learning inside unit ball

@highlight

This work is related to the recent spherical CNN and SE(n) equivariant network papers and extends previous ideas to volumetric data in the unit ball.

@highlight

Proposes using volumetric convolutions on convolutions networks in order to learn unit ball and discusses methodology and results of process.