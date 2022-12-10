Convolutional neural networks (CNNs) have achieved state of the art performance on recognizing and representing audio, images, videos and 3D volumes; that is, domains where the input can be characterized by a regular graph structure.

However, generalizing CNNs to irregular domains like 3D meshes is challenging.

Additionally, training data for 3D meshes is often limited.

In this work, we generalize convolutional autoencoders to mesh surfaces.

We perform spectral decomposition of meshes and apply convolutions directly in frequency space.

In addition, we use max pooling and introduce upsampling within the network to represent meshes in a low dimensional space.

We construct a complex dataset of 20,466 high resolution meshes with extreme facial expressions and encode it using our Convolutional Mesh Autoencoder.

Despite limited training data, our method outperforms state-of-the-art PCA models of faces with 50% lower error,  while using 75% fewer parameters.

Convolutional neural networks BID27 have achieved state of the art performance in a large number of problems in computer vision BID26 BID22 , natural language processing BID32 and speech processing BID20 .

In recent years, CNNs have also emerged as rich models for generating both images BID18 and audio .

These successes may be attributed to the multi-scale hierarchical structure of CNNs that allows them to learn translational-invariant localized features.

Since the learned filters are shared across the global domain, the number of filter parameters is independent of the domain size.

We refer the reader to BID19 for a comprehensive overview of deep learning methods and the recent developments in the field.

Despite the recent success, CNNs have mostly been successful in Euclidean domains with gridbased structured data.

In particular, most applications of CNNs deal with regular data structures such as images, videos, text and audio, while the generalization of CNNs to irregular structures like graphs and meshes is not trivial.

Extending CNNs to graph structures and meshes has only recently drawn significant attention BID8 BID14 .

Following the work of BID14 on generalizing the CNNs on graphs using fast Chebyshev filters, we introduce a convolutional mesh autoencoder architecture for realistically representing high-dimensional meshes of 3D human faces and heads.

The human face is highly variable in shape as it is affected by many factors such as age, gender, ethnicity etc.

The face also deforms significantly with expressions.

The existing state of the art 3D face representations mostly use linear transformations BID39 BID29 BID40 or higher-order tensor generalizations BID43 BID9 .

While these linear models achieve state of the art results in terms of realistic appearance and Euclidean reconstruction error, we show that CNNs can perform much better at capturing highly non-linear extreme facial expressions with many fewer model parameters.

One challenge of training CNNs on 3D facial data is the limited size of current datasets.

Here we demonstrate that, since these networks have fewer parameters than traditional linear models, they can be effectively learned with limited data.

This reduction in parameters is attributed to the locally invariant convolutional filters that can be shared on the surface of the mesh.

Recent work has exploited thousands of 3D scans and 4D scan sequences for learning detailed models of 3D faces BID13 BID46 BID37 BID11 .

The availability of this data enables us to a learn rich non-linear representation of 3D face meshes that can not be captured easily by existing linear models.

In summary, our work introduces a convolutional mesh autoencoder suitable for 3D mesh processing.

Our main contributions are:• We introduce a mesh convolutional autoencoder consisting of mesh downsampling and mesh upsampling layers with fast localized convolutional filters defined on the mesh surface.•

We use the mesh autoencoder to accurately represent 3D faces in a low-dimensional latent space performing 50% better than a PCA model that is used in state of the art methods BID39 for face representation.• Our autoencoder uses up to 75% fewer parameters than linear PCA models, while being more accurate on the reconstruction error.• We provide 20,466 frames of highly detailed and complex 3D meshes from 12 different subjects for a range of extreme facial expressions along with our code for research purposes.

Our data and code is located at http://withheld.for.review.This work takes a step towards the application of CNNs to problems in graphics involving 3D meshes.

Key aspects of such problems are the limited availability of training data and the need for realism.

Our work addresses these issues and provides a new tool for 3D mesh modeling.

Mesh Convolutional Networks.

give a comprehensive overview of generalizations of CNNs on non-Euclidean domains, including meshes and graphs.

defined the first mesh convolutions by locally parameterizing the surface around each point using geodesic polar coordinates, and defining convolutions on the resulting angular bins.

In a follow-up work, BID4 parametrized local intrinsic patches around each point using anisotropic heat kernels.

BID33 introduced d-dimensional pseudo-coordinates that defined a local system around each point with weight functions.

This method resembled the intrinsic mesh convolution of and BID4 for specific choices of the weight functions.

In contrast, Monti et al. used Gaussian kernels with trainable mean vector and covariance matrix as weight functions.

In other work, BID42 presented dynamic filtering on graphs where filter weights depend on the input data.

The work however did not focus on reducing the dimensionality of graphs or meshes.

BID45 also presented a spectral CNN for labeling nodes which did not involve any dimensionality reduction of the meshes.

BID38 and BID30 embedded mesh surfaces into planar images to apply conventional CNNs.

Sinha et al. used a robust spherical parametrization to project the surface onto an octahedron, which is then cut and unfolded to form a squared image.

BID30 introduced a conformal mapping from the mesh surface into a flat torus.

Although, the above methods presented generalizations of convolutions on meshes, they do not use a structure to reduce the meshes to a low dimensional space.

The proposed mesh autoencoder efficiently handles these problems by combining the mesh convolutions with efficient meshdownsampling and mesh-upsampling operators.

Graph Convolutional Networks.

BID8 proposed the first generalization of CNNs on graphs by exploiting the connection of the graph Laplacian and the Fourier basis (see Section 3 for more details).

This lead to spectral filters that generalize graph convolutions.

extended this using a windowed Fourier transform to localize in frequency space.

BID23 built upon the work of Bruna et al. by adding a procedure to estimate the structure of the graph.

To reduce the computational complexity of the spectral graph convolutions, BID14 approximated the spectral filters by truncated Chebyshev poynomials which avoids explicitly computing the Laplacian eigenvectors, and introduced an efficient pooling operator for graphs.

BID25 simplified this using only first-order Chebyshev polynomials.

However, these graph CNNs are not directly applied to 3D meshes.

Our mesh autoencoder is most similar to BID14 with truncated Chebyshev polynomials along with the efficient graph pooling.

In addition, we define mesh upsampling layer to obtain a complete mesh autoencoder structure and use our model for representation of highly complex 3D faces obtained state of the art results in realistic modeling of 3D faces.

Learning Face Representations.

BID2 introduced the first generic representation for 3D faces based on principal component analysis (PCA) to describe facial shape and texture variations.

We also refer the reader to BID10 for a comprehensive overview of 3D face representations.

Representing facial expressions with linear spaces has given state-of-the-art results till date.

The linear expression basis vectors are either computed using PCA (e.g. BID1 BID6 BID29 BID39 BID44 , or are manually defined using linear blendshapes (e.g. BID40 BID28 BID5 .

Multilinear models BID43 , i.e. higher-order generalizations of PCA are also used to model facial identity and expression variations.

In such methods, the model parameters globally influence the shape, i.e. each parameter affects all the vertices of the face mesh.

To capture localized facial details, BID34 and BID15 used sparse linear models.

BID9 used a hierarchical multiscale approach by computing localized multilinear models on wavelet coefficients.

BID9 also used a hierarchical multi-scale representation, but their method does not use shared parameters across the entire domain.

BID24 use a volumetric face representation in their CNN-based framework.

In contrast to existing face representation methods, our mesh autoencoder uses convolutional layers to represent faces with significantly fewer parameters.

Since, it is defined completely on the mesh space, we do not have memory constraints which effect volumetric convolutional methods for representing 3D models.

We define a face mesh as a set of vertices and edges F = (V, A), with |V| = n vertices that lie in 3D Euclidean space, V ∈ R n×3 .

The sparse adjacency matrix A ∈ {0, 1} n×n represents the edge connections, where A ij = 1 denotes an edge connecting vertices i and j, and A ij = 0 otherwise.

The non-normalized graph Laplacian is defined as L = D − A BID12 , with the diagonal matrix D that represents the degree of each vertex in V as D ii = j A ij .

n×n is a diagonal matrix with the associated real, nonnegative eigenvalues.

The graph Fourier transform BID12 of the mesh vertices x ∈ R n×3 is then defined as x ω = U T x, and the inverse Fourier transform as x = U x ω , respectively.

Fast spectral convolutions.

The convolution operator * can be defined in Fourier space as a Hadamard product, DISPLAYFORM0 .

This is computationally expensive with large number of vertices.

The problem is addressed by formulating mesh filtering with a kernel g θ using a recursive Chebyshev polynomial BID21 BID14 .

The filter g θ is parametrized as a Chebyshev polynomial of order K given by DISPLAYFORM1 whereL = 2L/λ max −

I n is the scaled Laplacian, the parameter θ ∈ R K is a vector of Chebyshev coefficients, and T k ∈ R n×n is the Chebyshev polynomial of order k that can be computed recursively as T k (x) = 2xT k−1 (x) − T k−2 (x) with T 0 = 1 and T 1 = x. The spectral convolution can then be defined as BID14 Table 2 : Decoder architecture that computes the j th feature of y ∈ R n×Fout .

The input x ∈ R n×Fin has F in features.

The input face mesh has F in = 3 features corresponding to its 3D vertex positions.

Each convolutional layer has F in × F out vectors of Chebyshev coefficients θ i,j ∈ R K as trainable parameters.

DISPLAYFORM2 Mesh Sampling The mesh sampling operators define the downscaling and upscaling of the mesh features in a neural net.

We perform the in-network downsampling of a mesh with m vertices using transform matrices Q d ∈ {0, 1} n×m , and upsampling using Q u ∈ R m×n where m > n.

The downsampling is obtained by contracting vertex pairs iteratively that maintains surface error approximations using quadric matrices BID16 .

The vertices after downsampling are a subset of the original mesh vertices DISPLAYFORM3 Since a loss-less downsampling and upsampling is not feasable for general surfaces, the upsampling matrix is built during downsampling.

Vertices kept during downsampling are kept during upsampling DISPLAYFORM4 Vertices v q ∈ V discarded during downsampling where Q d (p, q) = 0 ∀p, are mapped into the downsampled mesh surface.

This is done by projecting v q into the closest triangle (i, j, k) in the downsampled mesh, denoted by v p , and computing the Barycentric coordinates as DISPLAYFORM5 The weights are then updated in Q u as Q u (q, i) = w i , Q u (q, j) = w j , and Q u (q, k) = w k , and Q u (q, l) = 0 otherwise.

Figure 1: The effect of downsampling (red arrows) and upsampling (green arrows) on 3D face meshes.

The reconstructed face after upsampling maintains the overall structure but most of the finer details are lost.

Now that we have defined the basic operations needed for our neural network in Section 3, we can construct the architecture of the convolutional mesh autoencoder.

The structure of the encoder is shown in Table 1 .

The encoder consists of 4 Chebyshev convolutional filters with K = 6 Chebyshev polynomials.

Each of the convolutions is followed by a biased ReLU BID17 .

The downsampling layers are interleaved between convolution layers.

Each of the downsampling layers reduce the number of mesh vertices by 4 times.

The encoder transforms the face mesh from R n×3 to an 8 dimensional latent vector using a fully connected layer at the end.

The structure of the decoder is shown in Table 2 .

The decoder similarly consists of a fully connected layer that transforms the latent vector from R 8 to R 32×32 that can be further upsampled to reconstruct the mesh.

Following the decoder's fully connected layer, 4 convolutional layers with interleaved upsampling layers generated a 3D mesh in R 8192×3 .

Each of the convolutions is followed by a biased ReLU similar to the encoder network.

Each upsampling layer increases the numbers of vertices by 4x.

The FIG0 shows the complete structure of our mesh autoencoder.

Training.

We train our autoencoder for 300 epochs with a learning rate of 8e-3 with a learning rate decay of 0.99 every epoch.

We use stochastic gradient descent with a momentum of 0.9 to optimize the L1 loss between predicted mesh vertices and the ground truth samples.

We use a regularization on the weights of the network using weight decay of 5e-4.

The convolutions use Chebyshev filtering with K = 6.

Facial Expression Dataset.

Our dataset consists of 12 classes of extreme expressions from 12 different subjects.

These expressions are highly complex and uncorrelated with each other.

The expressions in our dataset are -bareteeth, cheeks in, eyebrow, high smile, lips back, lips up, mouth down, mouth extreme, mouth middle, mouth side and mouth up.

The number of frames of each sequence is shown in TAB2 .The data is captured at 60fps with a multi-camera active stereo system (3dMD LLC, Atlanta) with six stereo camera pairs, five speckle projectors, and six color cameras.

Our dataset contains 20,466 3D Meshes, each with about 120,000 vertices.

The data is pre-processed using a sequential mesh registration method BID29 to reduce the data dimensionality to 5023 vertices.

We preprocess the data by adding fake vertices to increase the number of vertices to 8192.

This enables pooling and upsampling of the mesh across the layers with a constant factor.

Implementation details We use Tensorflow BID0 for our network implementation.

We use Scikit-learn BID36 for computing PCA coefficients.

Training each network takes about 8 hours on a single Nvidia Tesla P100 GPU.

Each of the models is trained for 300 epochs with a batch size of 16.

We evaluate the performance of our model based on its ability to interpolate the training data and extrapolate outside its space.

We compare the performance of our model with a PCA model.

We consistently use an 8-dimensional latent space to encode the face mesh using both the PCA model and the Mesh Autoencoder.

Thus, the encoded latent vectors lie in R 8 .

Meanwhile, the number of parameters in our model is much smaller than PCA model (Table 4) .In order to evaluate the interpolation capability of the autoencoder, we split the dataset in training and test samples in the ratio of 1:9.

The test samples are obtained by picking consecutive frames of length 10 uniformly at random across the sequences.

We train our autoencoder for 300 epochs and evaluate it on the test set.

We use mean Euclidean distance for comparison with the PCA method.

The mean Euclidean distance of N test mesh samples with n vertices each is given by DISPLAYFORM0 where x ij ,x ij ∈ R 3 are j-th vertex predictions and ground truths respectively corresponding to i-th sample.

Table 4 shows the mean Euclidean distance along with standard deviation in the form [µ ± σ].

The median error is also shown in the table.

We show a performance improvement, as high as 50% over PCA models for capturing these highly non linear facial expressions.

At the same time, the number of parameters in the CNN is about 75% fewer than the PCA model as shown in Table 4 .

Visual inspection of our qualitative results in Figure 3 show that our reconstructions are more realistic and are effective in capturing extreme facial expressions.

We also show the histogram of cumulative errors in FIG1 .

We observe that Mesh Autoencoder has about 76.9% of the vertices within an Euclidean error of 2 mm, as compared to 51.7% for the PCA model.

To measure generalization of our model, we compare the performance of our model with a PCA model and FLAME BID29 .

For comparison, we train the expression and jaw model of FLAME with our dataset.

The FLAME reconstructions are obtained with with latent vector size of 16 with 8 components each for encoding identity and expression.

The latent vectors encoded using PCA model and Mesh autoencoder have a size of 8.We evaluate the generalization capability of the Mesh Autoencoder by attempting to reconstruct the expressions that are completely unseen by our model.

We split our dataset by completely excluding one expression set from all the subjects of the dataset.

We test our Mesh Autoencoder on the excluded expression as the test set.

We compare the performance of our model with PCA and FLAME using the same mean Euclidean distance.

We perform 12 cross validation experiments, one for each expression as shown in Table 5 .

For each experiment, we run our training procedure ten times initializing the weights at random.

We pick the best performing network for comparison.

We compare the results using mean Euclidean distance and median error metric in Table 5 .

Our method performs better than PCA model and FLAME BID29 on all expression sequences.

We show the qualitative results in FIG3 .

Our model performs much better on these extreme expressions.

We show the cumulative euclidean error histogram in FIG1 .

For a 2 mm accuracy, Mesh Autoencoder captures 84.9% of the vertices while the PCA model captures 73.6% of it.

The FLAME model BID29 uses several PCA-models to represent expression, jaw motion, face identity etc.

We evaluate the performance of mesh autoencoders by replacing the expression model of FLAME by our autoencoder.

We compare the reconstruction errors with the original FLAME model.

We run our experiment by varying the size of the latent vector for encoding.

We show the comparisons in Table 6 .

While our convolutional Mesh Autoencoder leads to a representation that generalizes better for unseen 3D faces than PCA with much fewer parameters, our model has several limitations.

Our network is restricted to learning face representation for a fixed topology, i.e., all our data samples needs to have the same adjacency matrix, A. The mesh sampling layers are also based on this fixed adjacency matrix A, which defines only the edge connections.

The adjacency matrix does not take in to account the vertex positions thus affecting the performance of our sampling operations.

In future, we would like to incorporate this information into our learning framework.

Mesh Autoencoder PCA FLAME BID29 Table 5 : Quantitative evaluation of Extrapolation experiment.

The training set consists of the rest of the expressions.

Mean error is of the form [µ ± σ] with mean Euclidean distance µ and standard deviation σ.

The median error and number of frames in each expression sequnece is also shown.

All errors are in millimeters (mm).The amount of data for high resolution faces is very limited.

We believe that generating more of such data with high variability between faces would improve the performance of Mesh Autoencoders for 3D face representations.

The data scarcity also limits our ability to learn models that can be trained for superior performance at higher dimensional latent space.

The data scarcity also produces noise in some reconstructions.

We have introduced a generalization of convolutional autoencoders to mesh surfaces with mesh downsampling and upsampling layers combined with fast localized convolutional filters in spectral space.

The locally invariant filters that are shared across the surface of the mesh significantly reduce the number of filter parameters in the network.

While the autoencoder is applicable to any class of mesh objects, we evaluated its quality on a dataset of realistic extreme facial expressions.

Table 6 : Comparison of FLAME and FLAME++.

FLAME++ is obtained by replacing expression model of FLAME with our mesh autoencoder.

All errors are in millimeters (mm).convolutional filters capture a lot of surface details that are generally missed in linear models like PCA while using 75% fewer parameters.

Our Mesh Autoencoder outperforms the linear PCA model by 50% on interpolation experiments and generalizes better on completely unseen facial expressions.

Face models are used in a large number of applications in computer animations, visual avatars and interactions.

In recent years, a lot of focus has been given to capturing highly detailed static and dynamic facial expressions.

This work introduces a direction in modeling these high dimensional face meshes that can be useful in a range of computer graphics applications.

<|TLDR|>

@highlight

Convolutional autoencoders generalized to mesh surfaces for encoding and reconstructing extreme 3D facial expressions.