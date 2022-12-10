This paper introduces the concept of continuous convolution to neural networks and deep learning applications in general.

Rather than directly using discretized information, input data is first projected into a high-dimensional Reproducing Kernel Hilbert Space (RKHS), where it can be modeled as a continuous function using a series of kernel bases.

We then proceed to derive a closed-form solution to the continuous convolution operation between two arbitrary functions operating in different RKHS.

Within this framework, convolutional filters also take the form of continuous functions, and the training procedure involves learning the RKHS to which each of these filters is projected, alongside their weight parameters.

This results in much more expressive filters, that do not require spatial discretization and benefit from properties such as adaptive support and non-stationarity.

Experiments on image classification are performed, using classical datasets, with results indicating that the proposed continuous convolutional neural network is able to achieve competitive accuracy rates with far fewer parameters and a faster convergence rate.

In recent years, convolutional neural networks (CNNs) have become widely popular as a deep learning tool for addressing various sorts of problems, most predominantly in computer vision, such as image classification BID22 BID17 , object detection BID30 and semantic segmentation BID8 .

The introduction of convolutional filters produces many desirable effects, including: translational invariance, since the same patterns are detected in the entire image; spatial connectivity, as neighboring information is taken into consideration during the convolution process; and shared weights, which results in significantly fewer training parameters and smaller memory footprint.

Even though the convolution operation is continuous in nature BID19 , a common assumption in most computational tasks is data discretization, since that is usually how information is obtained and stored: 2D images are divided into pixels, 3D point clouds are divided into voxels, and so forth.

Because of that, the exact convolution formulation is often substituted by a discrete approximation BID4 , calculated by sliding the filter over the input data and calculating the dot product of overlapping areas.

While much simpler to compute, it requires substantially more computational power, especially for larger datasets and filter sizes BID28 .

The fast Fourier transform has been shown to significantly increase performance in convolutional neural network calculations BID18 BID32 , however these improvements are mostly circumstantial, with the added cost of performing such transforms, and do not address memory requirements.

To the best of our knowledge, all versions of CNNs currently available in the literature use this discrete approximation to convolution, as a way to simplify calculations at the expense of a potentially more descriptive model.

In BID24 a sparse network was used to dramatically decrease computational times by exploiting redundancies, and in BID11 spatial sparseness was exploited to achieve state-of-the-art results in various image classification datasets.

Similarly, BID31 used octrees to efficiently partition the space during convolution, thus focusing memory allocation and computation to denser regions.

A quantized version was proposed in BID40 to improve performance on mobile devices, with simultaneous computational acceleration and model compression.

A lookup-based network is described in BID0 , that encodes convolution as a series of lookups to a dictionary that is trained to cover the observed weight space.

This paper takes a different approach and introduces the concept of continuous convolution to neural networks and deep learning applications in general.

This is achieved by projecting information into a Reproducing Kernel Hilbert Space (RKHS) BID33 , in which point evaluation takes the form of a continuous linear functional.

We employ the Hilbert Maps framework, initially described in BID29 , to reconstruct discrete input data as a continuous function, based on the methodology proposed in BID12 .

Within this framework, we derive a closed-form solution to the continuous convolution between two functions that takes place directly in this high-dimensional space, where arbitrarily complex patterns are represented using a series of simple kernels, that can be efficiently convolved to produce a third RKHS modeling activation values.

Optimizing this neural network involves learning not only weight parameters, but also the RKHS that defines each convolutional filter, which results is much more descriptive feature maps that can be used for both discriminative and generative tasks.

The use of high-dimensional projection, including infinite-layer neural networks BID15 ; BID9 , has been extensively studied in recent times, as a way to combine kernel-based learning with deep learning applications.

Note that, while works such as BID26 and BID25 take a similar approach of projecting input data into a RKHS, using the kernel trick, it still relies on discretized image patches, whereas ours operates solely on data already projected to these highdimensional spaces.

Also, in these works extra kernel parameters are predetermined and remain fixed during the training process, while ours jointly learns these parameters alongside traditional weight values, thus increasing the degrees of freedom in the resulting feature maps.

The proposed technique, entitled Continuous Convolutional Neural Networks (CCNNs), was evaluated in an image classification context, using standard computer vision benchmarks, and achieved competitive accuracy results with substantially smaller network sizes.

We also demonstrate its applicability to unsupervised learning, by describing a convolutional auto-encoder that is able to produce latent feature representations in the form of continuous functions, which are then used as initial filters for classification using labeled data.

The Hilbert Maps (HM) framework, initially proposed in BID29 , approximates realworld complexity by projecting discrete input data into a continuous Reproducing Kernel Hilbert Space (RKHS), where calculations take place.

Since its introduction, it has been primarily used as a classification tool for occupancy mapping BID12 BID5 BID34 and more recently terrain modeling BID13 .

This section provides an overview of its fundamental concepts, before moving on to a description of the feature vector used in this work, and finally we show how model weights and kernel parameters can be jointly learned to produce more flexible representations.

We start by assuming a dataset D = (x, y) N i=1 , in which x i ∈ R D are observed points and y i = {−1, +1} are their corresponding occupancy values (i.e. the probability of that particular point being occupied or not).

This dataset is used to learn a discriminative model p(y|x, w), parametrized by a weight vector w. Since calculations will be performed in a high-dimensional space, a simple linear classifier is almost always adequate to model even highly complex functions BID21 .

Here we use a Logistic Regression (LR) classifier BID6 , due to its computational speed and direct extension to online learning.

The probability of occupancy for a query point x * is given by: DISPLAYFORM0 where Φ(.) is the feature vector, that projects input data into the RKHS.

To optimize the weight parameters w based on the information contained in D, we minimize the following negative log-likelihood cost function: DISPLAYFORM1 with R(w) serving as a regularization term, used to avoid over-fitting.

Once training is complete, the resulting model can be used to query the occupancy state of any input point x * using Equation 1, at arbitrary resolutions and without the need of space discretization.

The choice of feature vector Φ(.) is very important, since it defines how input points will be represented when projected to the RKHS, and can be used to approximate popular kernels such that DISPLAYFORM0 In BID12 , the authors propose a feature vector that places a set M of inducing points throughout the input space, either as a grid-like structure or by clustering D. Inducing points are commonly used in machine learning tasks BID35 to project a large number of data points into a smaller subset, and here they serve to correlate input data based on a kernel function, here chosen to be a Gaussian distribution with automatic relevance determination: DISPLAYFORM1 where µ ∈ R D is a location in the input space and Σ is a symmetric positive-definite covariance matrix that models length-scale.

Each inducing point maintains its own mean and covariance parameters, so that M = {µ, Σ} M i=1 .

The resulting feature vector Φ(x, M) is given by the concatenation of all kernel values calculated for x in relation to M: DISPLAYFORM2 Note that Equation 4 is similar to the sparse random feature vector proposed in the original Hilbert Maps paper BID29 , but with different length-scale matrices for each inducing point.

This modification naturally embeds non-stationarity into the feature vector, since different areas of the input space are governed by their own subset of inducing points, with varying properties.

To increase efficiency, only a subset of nearest neighbors can be used for feature vector calculation, while all others are set to zero.

Indeed, this feature vector has been successfully applied to accurately reconstruct large-scale 3D datasets at a fraction of the computational cost required by other similar kernel-based techniques BID2 , which makes it attractive for big data processing.

In the original implementation, the parameters {µ, Σ} i of each kernel in M are fixed and calculated based on statistical information obtained directly from D. Only the classifier weights w are optimized during the training process, according to Equation 2.

However, this approach severely limits the descriptiveness of each kernel, especially if training data is not readily available for preprocessing.

Here we show how the HM training methodology can be reformulated to include the optimization of all its parameters P = {µ, Σ, w} DISPLAYFORM0 .

The key insight is realizing that the HM framework is analogous to a neural network layer BID14 , in which input data is described as a Gram Matrix BID16 ) in relation to the inducing set M, such that: DISPLAYFORM1 where 1998) can be used to jointly optimize kernel parameters, using the corresponding partial derivatives: DISPLAYFORM2 DISPLAYFORM3 which can be efficiently calculated during feature vector generation.

An example of this joint learning process can be found in FIG1 , for a simple 1D classification problem.

In the left column, the standard HM framework was used, with only weight learning, whereas in the right column kernel parameters were also learned, using the proposed HL framework (Hilbert Layer).

In the top row 20 inducing points were used, initially equally spaced, while in the bottom row only 6 inducing points were used.

Note that, for higher densities, HM converges to good results, however it fails to capture some occupancy behaviors in lower densities, due to its smaller descriptive power.

On the other hand, HL is able to achieve considerably better results in both cases, with reasonable convergence even in lower densities.

Particularly, in FIG1 we can see how inducing points migrate to discontinuous areas, thus ensuring sharper transitions, and in FIG1 one inducing points assumed a larger length-scale to reach a particular occupied area that was under-represented.

Lastly, note that, while the standard HM framework as described in Section 2.1 only addresses classification tasks, the proposed joint learning methodology can be easily modified to address general regression tasks, simply by removing the activation function σ and optimizing a different loss function (i.e. mean squared error instead of cross-entropy).

In this section we show how the Hilbert layer, as defined by Equation 5, can be extended to a convolutional scenario, in which two functions in different RKHS are convolved to produce a third RKHS defining functions that approximate activation values.

We start by formulating a closed-form solution to the continuous convolution between kernels that describe a feature vector for projection in the HL framework, move on to a formal definition of the Convolutional Hilbert Layer (CHL), and lastly describe how this novel framework can be used for image classification tasks.

Convolution is a mathematical operation that takes two functions f and g and produces a third function h = f * g, that can be viewed as the amount of overlapping between both functions as one is reversed and shifted over the other.

Formally, it is defined as: DISPLAYFORM0 Solving Equation FORMULA9 analytically can be a difficult (or even impossible) task for most functions that define real-world phenomena, due to their complexity.

However, as shown in Sec. 2.2, the Hilbert Maps framework is able to approximate arbitrarily complex functions using simple kernels, by projecting input data into a high-dimensional RKHS.

Although the proposed methodology can be applied to any two kernel functions with a closed-form convolution formula, for notation simplicity we assume, without loss of generality, that the kernels describing both functions are given by Equation 3.

This choice greatly simplifies the problem because the convolution of two Gaussian distributions is also a Gaussian distribution with automatic relevance determination BID1 : DISPLAYFORM1 where DISPLAYFORM2 Note that this new kernel does not model function states in the RKHS, but rather activation values, representing convolution results between M i and M j , and can be queried at arbitrary resolutions using the HL framework described in Section 2.3.

More importantly, it can be optimized using the same training methodology, to produce better generative or discriminative models.

Now that convolution between kernels that define functions in the RKHS has been established, here we show how convolution between two parameter sets P f and P g , each representing a different function in its respective RKHS, is performed.

Comparing to the Hilbert layer defined in Equation 5, the convolutional Hilbert Layer takes the form: DISPLAYFORM0 where DISPLAYFORM1 ).

Note that K f g is a block-matrix, so weight multiplications are performed independently for each entry before summation, which benefits the use of parallel computing for faster calculations.

The output h are the weights that approximate f * g in the RKHS defined by the cluster set M h , and together these compose the parameter set P h = {M h , h} = {µ h , Σ h , w h }.

Pseudo-code for this convolution process is given in Algorithm 1, where we can see how it operates: for each inducing point in P f and P g ,

Require: Input P f and filter Pg parameter sets, convolved M h cluster set Ensure: Convolved P h parameter set 1: P h ← M h % Convolved parameter set is initialized with cluster set values 2: w h ← 0 % Convolved parameter set weights are set to zero 3: for P i f ∈ P f do 4:for P j g ∈ Pg do 5:for DISPLAYFORM0 end for 8: end for 9: end for their kernel convolution is calculated (Equation 9) and used to evaluate all points in P h (Eq. 3), each one contributing to its corresponding weight parameter w k h .

Multiple P input channels and Q filters can be incorporated by concatenating the various K pq f g into a single block-matrix, while augmenting w f and w g accordingly, such that: DISPLAYFORM1 where K f g is a P × Q block-matrix with entries DISPLAYFORM2 , b h is now a Q × 1 bias vector and P h = {M h , H} = {µ h , Σ h , {w} Q q=1 } contains multiple weight channels defined in the same RKHS.

An example of the output produced by a convolutional Hilbert Layer, for a simple 1D classification problem, is depicted in FIG3 , alongside the corresponding discrete convolution results.

As expected, areas in the input that are similar to the filter have higher activation values, and these vary continuously throughout the input space, being able to capture small variations and partial matches to a higher detail than discrete convolution.

Lastly, note that pooling BID22 can be naturally incorporated into the convolutional Hilbert layer simply by decreasing the density of clusters in the output RKHS.

This process has two effects: 1) decrease computational cost, which allows for larger filter sizes and number of channels while combating over-fitting; and 2) decrease the resolution of the approximated continuous function, thus aggregating statistics of spatially close regions to capture patterns in a larger scale.

An un-pooling effect is similarly straightforward, generated by increasing the density of clusters in the output RKHS, thus increasing the resolution of the approximated continuous function at the expense of a larger number of parameters.

A diagram depicting how the proposed convolutional Hilbert layer can be applied to an image classification task, to create a Continuous Convolutional Neural Network (CCNN), is shown in FIG5 .

The original image, composed of discrete data D n = {x, y n }, with x ∈ R 2 being the pixel coordinates and y n = [0, 1] C0 their corresponding intensity values (C 0 is the number of input channels, i.e. 1 for grayscale and 3 for color images), is first modeled as a continuous function via projection to a RKHS.

Note that the same RKHS is used to model all input images, defined by the cluster set M f 0 , and within this projection each image is represented with a different set of model weights W n f 0 .

The resulting parameter set P n f is convolved with the filters contained in P g1 to produce the hidden feature maps P f 1 , that also share the same RKHS for all input images, defined by M f 1 , but with individual model weights W n f 1 .

This process is repeated for each convolutional Hilbert layer, with the final model weights being flattened to serve as input for a standard fully connected neural network, that performs the classification between different categories.

The convolutional trainable parameters to be optimized in this topology are the various cluster sets M f = {µ, Σ}, that define the RKHS for each feature map, and the various filter sets P g = {µ, Σ, W}, that perform the transformation between feature maps in each layer.

Note that each of these parameters represents a different property of the convolution process: µ defines location, Σ defines length-scale and W defines weight, and therefore should be treated accordingly during optimization.

To guarantee positive-definitiveness in variance values, we are in fact learning a lower triangular matrix V , such that Σ = V T V , which is assumed to be invertible (i.e. its determinant is not zero, or none of its main diagonal values are zero).

While this property cannot be strictly guaranteed during the optimization process, in practice the noisy nature of stochastic gradient descent naturally avoids exact values of zero for trainable parameters, and at no point during experiments this assumption was broken.

To improve parameter initialization, we employ a continuous fully-convolutional auto-encoder (CCAE), that first encodes data into a lower-dimensional latent feature vector representation and then decodes it back to produce a reconstruction of the original input.

Particularly, the encoding pipeline is composed by the convolutional Hilbert layers from the classification topology, and the decoding pipeline has these same layers in reverse order (without parameter sharing), as depicted in FIG6 .

A lower-dimensional representation is achieved by decreasing the number of clusters used for feature map projection in deeper layers, thus simulating a pooling effect.

Similarly, the number of clusters used for filter projection can also be modified, emulating different kernel sizes in standard discrete convolution, however since the location of these clusters is a trainable parameter their support is inherently adaptive, only changing in complexity as more clusters are added.

In all experiments, inducing points were initialized with mean values equally spaced in the 2D space and with the same variance value, so that the distance between mean values is equal to two standard deviations (weight values were initialized randomly, using a truncated Gaussian distribution with mean 0 and variance 0.1).

Here we present and discuss experimental results performed to validate the proposed convolutional Hilbert layer in an image classification scenario 1 .

Four different standard benchmarks were considered: the MNIST dataset, composed of 60000 training and 10000 test images with dimensionality 28 × 28 × 1; the CIFAR-10 dataset, composed of 50000 training and 10000 test images with dimensionality 32 × 32 × 3; the STL-10 dataset, composed of 5000 training and 8000 tests images with dimensionality 96 × 96 × 3 plus 100000 unlabeled images; and the SVHN dataset, composed of 604388 training and 26032 test images with dimensionality 32 × 32 × 3.

No preprocessing or data augmentation of any kind was performed in these datasets during training or test phases.

Examples of reconstruction results for the MNIST and CIFAR-10 datasets, using the proposed HL framework, are depicted in FIG7 .

These results were obtained by projecting all images from each dataset into the same input RKHS, defined by the cluster set M f 0 , and producing the weight parameters W n f 0 individual to each image (for RGB images, each channel was treated independently).

Both the cluster set parameters and individual model weights were optimized using the proposed joint learning methodology from Section 2.3, to minimize the squared reconstruction error.

Once training was complete, these parameters served as input for a continuous convolutional neural network (CCNN) for image classification, in which each projected image is mapped to its corresponding label via cross-entropy minimization and a softmax activation function in the output layer (see FIG5 .

To initialize the convolutional parameters of this network, a continuous convolutional auto-encoder (CCAE) was used, mirroring convolutional layers to produce a final reconstruction of the original input, via direct squared error minimization over the discretized output (see FIG6 .To test the expressiveness of the proposed continuous feature maps, we compared CCNN image classification results against the standard DCNN (Discrete Convolutional Neural Network) architecture, in the special case when very few filters are used (here, ranging from 1 to 20, with size 3 × 3 for discrete and 9 clusters for continuous).

A single convolutional layer was used, followed by a fully-connected layer with 200 nodes and the output layer (no dropout or regularization of any sort was used).

Classification results for the MNIST dataset are depicted in FIG8 , where we can see that a single continuous filter is able to achieve better overall results than all twenty discrete filters, both in training and test accuracy.

Interestingly, while a single discrete filter actually achieves worst loss function training values than a straightforward fully connected neural network (FCNN) without convolutional layers, a single continuous filter continues to improve the loss function over time, which was still decreasing after the alloted number of iterations.

We also noticed less over-fitting, as it can be shown by loss function values for testing data, that started to consistently increase for DCNN after a certain number of training iterations, while CCNN was able to maintain lower values throughout the entire training process.

Furthermore, we can see that CCNN produces much larger ranges of loss function values both for training and testing data, indicating that the choice of initial parameter values play a more significant role during the optimization process, especially when fewer filters are considered (which is to be expected, since they are able to capture a larger range of patterns to use during the convolution process).

Examples of convolutional filters obtained in these experiments are depicted in Figure 7 , where we can see their variability and ability to model different patterns that will be useful during the classification process.

Classification results, in terms of percentual test accuracy error, for the three datasets considered here are presented in TAB0 , in relation to other image classification techniques found in the literature.

The same CCNN architecture was used in all cases, composed of three convolutional layers with 20-40-60 filters of sizes 25-16-9 and pooling ratios of 2-3-4 in relation to input data dimen- Figure 7 : Examples of CCNN filters for the MNIST dataset.

Method Acc.

Frac.

Max-Pooling BID11 99.68 CCNN (CCAE init.)

99.63 Conv.

Kernel Net.

BID26 99.61 Maxout Net.

BID10 99.55 CCNN (random init.)

99.51 PCANet BID3 99.38 BID11 96.53 Maxout Net.

BID10 BID3 78.67 Table 2 : CIFAR-10 results.

Method Acc.

Multi-Task Bayes BID37 70.10 C-SVDDNet BID39 68.23 CCNN (CCAE init.)

63.81 Conv.

Kernel Net.

BID26 62.32 Disc.

Learning BID7 62.30 Pooling Invariant BID20 58.28 Table 3 : STL-10 results.

Method Acc.

ReNet BID38 97.62 Maxout Net.

BID10 97.53 Stoch.

Pooling BID41 97.02 CCNN (CCAE init.)

96.27 Shallow CNN BID27 96.02 CCNN (random init.)

93.48 Table 4 : SVHN results.sionality, followed by two fully-connected layers with 512-1024 nodes and 0.5 dropout BID36 .

Note that this architecture is much simpler than the ones presented by networks capable of achieving state-of-the-art classification results in these datasets, possessing a total of 144596 convolutional parameters, 588 + 261 + 147 = 996 of which define the intermediary RKHS for feature maps representation and the remaining 6000 + 51200 + 86400 = 143600 representing the filters within these RKHS.

Nevertheless, we can see that the proposed convolutional Hilbert layer is able to achieve competitive results in all three datasets, even with such shallow and narrow architecture, which further exemplifies the descriptive power of a continuous representation when applied in conjunction with the convolution operation.

Particularly, the introduction of unsupervised pretraining, using the proposed CCAE architecture to generate initial parameter estimates, significantly improves accuracy results.

This paper introduced a novel technique for data representation that takes place in a highdimensional Reproducing Kernel Hilbert Space (RKHS), where arbitrarily complex functions can be approximated in a continuous fashion using a series of simple kernels.

We show how these kernels can be efficiently convolved to produce approximations of convolution results between two functions in different RKHS, and how this can be applied in an image classification scenario, via the introduction of a novel deep learning architecture entitled Continuous Convolutional Neural Networks (CCNN).

Experimental tests using standard benchmark datasets show that this proposed architecture is able to achieve competitive results with much smaller network sizes, by focusing instead on more descriptive individual filters that are used to extract more complex patterns.

Although promising, there are still several potential improvements that are left for future work, such as: RKHS sparsification, so only a subset of clusters are used for feature vector calculation, which would greatly improve computational speed and memory requirements; different learning rates and optimization strategies for each class of parameter (cluster location, length-scale and weight), to improve convergence rates; and the use of different kernels for feature vector representation, as a way to encode different properties in the resulting feature maps.

<|TLDR|>

@highlight

This paper proposes a novel convolutional layer that operates in a continuous Reproducing Kernel Hilbert Space.

@highlight

Projecting examples into an RK Hilbert space and performing convolution and filtering into that space.

@highlight

This paper formulates a variant of convolutional neural networks which models both activations and filters as continuous functions composed from kernel bases