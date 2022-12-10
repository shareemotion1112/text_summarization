In recent years Convolutional Neural Networks (CNN) have been used extensively for Superresolution (SR).

In this paper, we use inverse problem and sparse representation solutions to form a mathematical basis for CNN operations.

We show how a single neuron is able to provide the optimum solution for inverse problem, given a low resolution image dictionary as an operator.

Introducing a new concept called Representation Dictionary Duality, we show that CNN elements (filters) are trained to be representation vectors and then, during reconstruction, used as dictionaries.

In the light of theoretical work, we propose a new algorithm which uses two networks with different structures that are separately trained with low and high coherency image patches and show that it performs faster compared to the state-of-the-art algorithms while not sacrificing from performance.

Recent years have witnessed an increased demand for superresolution (SR) algorithms.

Increased number of video devices boosted the need for displaying high quality videos online with lower bandwidth.

In addition, the social media required the storage of videos and images with lowest possible size for server optimization.

Other areas include 4K video displaying from Full HD broadcasts, increasing the output size for systems that have limited sized sensors, such as medical imaging, thermal cameras and surveillance systems.

SR algorithms aim to generate high-resolution (HR) image from single or ensemble of lowresolution (LR) images.

The observation model of a real imaging system relating a high resolution image to the low resolution observation frame can be given as DISPLAYFORM0 where H models the blurring effects, S models the downsampling operation, and n models the system noise.

The solution to this problem seeks a minimal energy of an energy functional comprised of the fidelity of the estimated imagef to the observational image f .State-of-the art algorithms that are addressing SR problem can be collected under Dictionary learning based methods (DLB) and Deep learning based methods (DLM) categories.

Although SR problem is an inverse problem by nature, performance of other methods such as Bayesian and Example based methods have been surpassed which is the reason why they are not included in this work.

Also the SR problem has never been directly dealt with inverse problem solutions as in BID3 BID4 DLB are generally solving optimization problems with sparsity constraints such as BID20 BID21 and L 2 norm regularization as in BID19 .

The main concern of DLB is creation of a compact dictionary for reconstruction of high resolution (HR) image.

Although useful, DLB methods become heavy and slow algorithms as reconstruction performance increases.

Recent advances on GPUs have fueled the usage of convolutional neural networks (CNNs) for SR problem.

CNN based algorithms such as BID5 and BID9 have used multi-layered networks which have successfully surpassed DLB methods in terms of run speed and performance.

State-of-the art algorithms also use Perceptual Loss (PL) to generate new textures from LR images BID11 .

By uniting PL and generative networks, photo realistic images can be generated BID10 .

PL minimization based algorithms are visually superior to MSE minimization based ones.

Stability of such algorithms have been improved since they have been first proposed BID7 .

Although, stability issue is not yet completely addressed for generative networksIn BID1 authors have described representation learning as a manifold learning for which a higher dimensional data is represented compactly in a lower dimensional manifold.

They have discussed that the variations in the input space is captured by the representations, for which we are explaining the mechanism at work.

Though CNNs are successful for SR problem experimentally, their mathematical validation is still lacking.

We summarized the contributions of this work.• We show that neurons solve an Iterative Shrinkage Thresholding (IST) equation during training for which the operator is dictionary matrix constructed from LR training data.

The solution yields a representation vector as the neuron filters.

Contrary to the discussion in literature for which an encoder-decoder structure is needed to obtain and use representations, we claim that the filters themselves become the representations.• We describe a new concept namely Representation Dictionary Duality (RDD) and show that neuron filters act as representation vectors during training phase.

Then in the testing phase, filters start acting as dictionaries upon which the HR reconstruction is made layer by layer.

This is a concept which helps us analyze CNNs with sparse representation and inverse problem mathematics.• After analyzing a neuron with inverse problem and DLB solutions and discussing how the entire network operates during training, we propose a new network structure which is able to recover certain details better, faster without sacrificing overall performance.

Rest of the paper organized as follows: in section 2 we refer to related literature for different areas of research.

Section 3 ties previous work into our analysis of CNNs.

In section 4 we propose a new network for SR problem.

In section 5 we give experimentation results.2 RELATED WORK 2.1 ANALYTIC APPROACHES Solution to eq. 1 is inherently ill-conditioned since a multiplicity of solutions exist for any given LR pixel.

Thus proper prior regularization for the high resolution image is crucial.

The regularization of the inversion is provided with a function, reg, which promotes the priori information from the desired output, reg takes different forms ranging from L 0 norm, Tikhonov regularization to orthogonal decomposition of the estimate.

Denoting the SH matrix in eq. 2 by K, the regularized solution is given byf DISPLAYFORM1 In BID4 DISPLAYFORM2 Where a class of proximity operators are defined, the special function for the case of L 1 regularization is soft thresholding function also known as shrinkage operator.

DISPLAYFORM3 Notice that K T (g − Kf n−1 ) is the negative gradient of data fidelity term in the original formulation.

Therefore the solution for the inverse problem using IST iterations is obtained in a gradient descent type method thresholded by Moreau proximity mapping which is also named as Proximal Landweber Iterations.

BID4 have proposed the usage of non-quadratic regularization constraints that promote sparsity by the help of an orthonormal (or overcomplete) basis ϕ l of a Hilbert space.

For the problem defined in eq. 2 it is proposed to use a functional φ b,p as DISPLAYFORM4 For the case when p = 1, a straightforward variational equation can be obtained in an iterative way.

DISPLAYFORM5 Iterations over the set of basis functions can be carried out in one formula DISPLAYFORM6 where DISPLAYFORM7 which can be seen as a method to file the elements of x in the direction of ϕ l .

Daubechies et.

al. have proven that the solution obtained by iterating f is the global minimum of the solution space.

The solution will reach to an optimum point if K is a bounded operator satisfying ||Kf || ≤ C||f || for any vector f and some constant C.We will use this result in proving that neurons in a CNN architecture are able to reach to the optimum solution for SR problem by solving for the exact same eq. 7.

A similar work is conducted by BID8 .

They have proposed a Learned IST algorithm which can be seen as a time unfolded recurrent neural network.

Later BID2 have discussed that LISTA and their own algorithms that extend LISTA are not mere approximations for an iterative algorithm but themselves are full featured sparse coders.

Our work diverges from theirs in showing how a convolutional neural network is able to learn image representation and reconstruction for SR problem inside network parameters.

We will unite inverse problem approaches, DLM and DLB methods in a representation-dictionary duality concept.

Instead of approaching the superresolution problem to directly invert an observation model, DLB learn mappings from LR to HR training images based on a dictionary.

The algorithms jointly solve for a compact dictionary and a representation vector.

Sparse representation has been applied to the dictionary learning based SR problem.

An LR image is sparsely represented by an LR dictionary.

The representation vector is either directly or by some changes applied to an HR library for reconstruction of HR image.

DLB algorithms both solve for creating dictionary and solve for a representation vector for any input.

The K-SVD algorithm BID0 is one of the keystones of dictionary learning for the purpose of sparse representation.

Aharon et.

al. have proposed the usage of a compact dictionary D, from which a set of atoms (columns or dictionary elements) are to be selected via a vector f and the combination of these atoms is constrained to be similar to a patch (or image) g via ||g − Df || p ≤ ε.

If the dimension of g is less than that of matrix D and if D is full-rank matrix then there are infinitely many solutions to the problem therefore a sparsity constraint is introduced.

DISPLAYFORM0 The L 0 norm gives the number of entries in f that are non-zero.

The usage of compact dictionaries for SR problem is introduced in BID20 .

The authors have used the approach of K-SVD.The optimization of L 0 norm regularized equation is hard and a closed form solution might not be available.

For the case when f is sufficiently sparse, eq. 1 can be approximated by L 1 norm.

The solution of such an equation can be obtained by Lagrange multipliers.

DISPLAYFORM1 During learning phase the library D is initialized by random gaussian noise and an iterative algorithm between a batch representation matrix Z and dictionary D refines the dictionary while maintaining sparsity for representation vectors of training set.

BID20 uses two dictionaries, one for LR representation, one for HR reconstruction as described in the beginning of this chapter.

BID19 have proposed the usage of L 2 norm instead of L 1 norm for even faster computations.

Although usage of L 2 norm eliminated the sparsity constraint from the equation it will play a role in understanding how CNNs work in later chapters.

The mapping between the high and low resoultion images can also be found by convolutional networks BID5 , BID9 ).The activation function plays an important role in neural network training.

In many state-of-the-art algorithms major functions such as tanh and softmax have been replaced by rectified linear units BID12 that are linear approximations of mathematically complex and computationally heavy functions.

BID6 has empirically shown that by using rectified activations the network can learn sparse representations easier.

For a given input, only a subset of hidden neurons are activated, leading to better gradient backpropagation for learning and better representations during forward pass.

Especially sparse representation has been shown BID6 to be useful.

Sparsity constraint provides information disentagling which allows the representation vectors to be robust against small changes in input data.

BID18 uses gradient information to separate image pixels during interpolation.

Separation is done according to three properties namely, strength, coherence and angle.

A low strength and coherence signifies as lack of content inside the patch.

A high strength but low coherence signifies corner or multi directional edge information.

High strength and high coherence signifies a strong edge.

Especially the coherence information will play an important role in section 3.

BID5 have provided the earliest relation of CNNs to Sparse Representation.

In their view outputs of the first layer constitute a representation vector for a patch around each pixel in LR image, second layer maps LR representations to HR representation vectors and the last layer reconstructs HR image using 5x5 sized filters (or atoms if we have used the jargon of sparse representations).

Although this idea qualitatively maps CNNs as a solution method for sparse representation problem, we will now show a more complete understanding with mathematical background.

Figure 5 in Appendix shows how SRCNN algorithm works.

Even though CNNs yield very good estimates of superesolved images, the connection between inversion of observation model and activation of neurons in CNNs is missing.

In this section, we will show the relation between the inverse problem solutions and sparse representation to CNNs.

FIG0 summarize how we are connecting all previous work to CNNs.

Considering single neuron we are going to generalize the solution.

For the training phase of CNNs, LR images are fed into the network for forward pass.

The resulting image from the network is compared against a ground truth HR image and the error is backpropagated.

Since the input image is convolved by the neuron filter, its size should be larger than the size of the output to prevent boundary conditions.

This will not be a problem in our case, as stated by BID9 , the results from a deep residual network are not spoiled even at the edges of the images.

The convolution operation can be carried out in an algebraic manner.

Let us assume that we are operating on a patch of LR image, that is named as superpatch.

The superpatch is divided into chunks, that are named as subpatches, which have the same support as the filter.

Filter and each During training CNN solves the mapping of LR input to HR image in training set.

The product of the network is going to be a mapping, f L , from LR superpatch, that is collected under D L , to HR patch, g L .

Therefore the vector f L will be the neuron filter, i.e. the only variable for the training phase, D L will be concatenated subpatch matrix, each subpatch in vectorized form will be named as ϕ l .

The vector g L will be a patch from HR image as in Figure 6 .

We will show that the CNN operations solve for the same equation as in eq. 7.

Since the D L matrix satisfies the boundedness constraint on the operator of the equation, the solution will be optimum.

We now modify gradient descent type learning process.

Convolution of subpatches with the filter can be algebraically written as D L f L .

Then, DISPLAYFORM0 Taking the gradient of MSE with respect to f L is tricky.

When an element of D L f L vector lies below the bias, the result will be zero causing the gradient to be zero.

We modify the equation by changing the bias vector to b to enable us to use MSE gradient formulation DISPLAYFORM1 .

This is a valid insertion since the addition of g L and multiplication of D T L are linear operators that can be used to scale elements of original bias vector b. Then, DISPLAYFORM2 Then the summation of thresholded decompositions is given as DISPLAYFORM3 ) which is the same as eq. 7 where Z b is defined in eq. 8.

Therefore gradient descent type learning of a single neuron is guarantied to reach an optimal solution using eq. 3.For the testing phase, a new representation -dictionary duality (RDD) concept is proposed.

RDD concept states that the representation vectors learned during the training phase can be used as atoms of a dictionary for the testing phase.

The cost function that is minimized by CNN training (learning) yields a representation vector as the neuron filter, for which the dictionary is matrix D L and the target is HR image patch.

During testing (scoring, reconstruction) phase, resulting representation vectors (filters) from a layer of neurons turn into a dictionary (later named as D R ) upon which the reconstruction of HR image is carried out.

A similar idea is proposed by BID15 and BID16 stating that each layer output which is a representation for inputs of previous layer, can also be seen as an input to be represented by the next layer.

The authors have argued that each layer output will contain a structure which can be represented by a convolutional sparse coding (CDC) layer.

A CDC layer is essentially a CNN layer.

The difference of our RDD is that we use the idea that dictionaries and representations swap roles during training and testing (forward pass).

Also during training, inputs to each layer is perceived as a dictionary for the next layer, contrary to previously proposed perception of BID15 .

Following the idea of RDD, the neuron filter, previously named as f L , can be viewed as an atom of a dictionary consisting of many other neuron filters.

During testing period, the filters are vectorized and concatenated to form the dictionary matrix D R , the vector g R will be the input image this time and the f R vector will be the neuron outputs, which will be the representation vector of input image in terms of the dictionary atoms, i.e. the neuron filters.

The mathematical insight for this is again given in eq. 3.

Considering the initial condition for the equation, during testing phase, f 0 R can be assumed as zero and the f 1 R is going to be the representation vector provided with DISPLAYFORM4 Again we reach to the conclusion that the ReLU operators provide the representation vector, f R , for the input image, g R , given the trained filter values collected under D R .

We have demonstrated this feature in experimentation chapter.

To provide a visually meaningful example we have used a training set that contain highly coherent edges with a narrow orientation range.

RDD is visually apparent only for the first layer and for training sets with similar information content.

Deeper layers feed from previous layer's outputs therefore it is hard to demonstrate for all layers.

Also while training with a more general training set, an observer will not see any patterns in learned filters.

Seeing apparent features would mean memorization which is a degrading property for a neural network.

To extend the understanding of single neuron to the entire network Theorem 1 will be used from BID15 .Theorem 1 Suppose g = y + n where n is noise whose the power of noise is bounded by ε 0 and y is a noiseless signal.

Considering a convolutional sparse coding structure where D l is the dictionary for l th layer DISPLAYFORM5 DISPLAYFORM6 where µ(D i ) is the mutual coherence of the dictionary then 1.

The support of the solutionf i is equal to the support of f i 2.

DISPLAYFORM7 The theorem shows that a network consisting of layered neurons could yield the same result as a layered sparse coding algorithm.

Therefore a network of neurons, whose optimality for inverse problem solutions has been proven individually, is now proven to reach to a solution for sparse coding.

Let us now recall the Landweber equation DISPLAYFORM8 to be able to use insights from this equation assume that all neurons in the network are activated for the inputs.

For that un-realistic case, the network filters can be convolved among themselves to produce an end point filter, f L .

This is feasible because when all neurons are activated, their linear unit outputs are going to be the convolution results minus a bias that can be added up at the end, simply enabling the convolution of all filters to be applied in a single instant.

A similar work is done by BID13 In other words, the result, f L , consists of scores which measure how similar g L vector is to each subpatch from the entire superpatch.

If the HR image has content that cannot be recovered by using certain region of LR image, the reconstructed image is going to be inferior.

This is due to the violation of overcompleteness assumption.

Selection of a larger area for the reconstruction of certain HR patches proves useful because of increased information included into the system that brings the subpatches, or bases, closer to being overcomplete.

This insight provides a method for determining how deep a network should be for certain features.

For example when the superpatch and corresponding HR region contains only texture, which can be modeled as gaussian noise, the D L matrix becomes linearly independent, meaning easily invertible.

Consequently when the training set consists solely of textured images, shallow networks should do as good as deep networks.

Then for the testing phase, same filters are used to construct the D L matrix and the result of the network is obtained by the same equation without normalization (without the inverse term) this time (since it is already normalized) as in eq. 11, i.e. projecting LR image onto filters' domain.

Notice that the error is generally not completely orthogonal to the LR images because of iterative nature of equations.

Therefore this is not going to be a meaningless operation.

The representations that are learned during training can only be called complete if the data can be completely recovered BID17 .

Since the method by which the network recovers HR details is through inner products, the assertion of RDD seems complete.

In general the training set contains various features with different variances.

Therefore the generalization of the new concepts that are introduced here are difficult.

Training with different structures enables the constant evolution of neuron filters during training.

However to have an activating branch for each feature either the network should have increased number of filters or the network will not converge which can be explained by the manifold hypothesis, as representations not covering the high dimensional input space BID1 .

This is the point where we tie theoretical work into a practical network.

The discussions from section 3 revealed that using a single training set for a single network is complicating the training process.

Because we expect the neuron filters to learn predominant patterns and information from the training set, training a single network either leads to a heavy network with lots of memory requirement or leads to insufficiently learned filters.

We are proposing usage of a double network SR (DNSR) for two different data.

The data separation is done according to gradient information, dividing data set into low and high coherence sets.

For low coherence data which is mainly texture, we have trained a shallower network as in FIG6 .

We have used network depth of 10 layers, as tests with shallower or deeper networks slightly turned out to be in favor of 10 layers.

High coherence data contains edge and corner information.

We trained a deeper network of 20 layers to reconstruct edge information.

This is, to the best of our knowledge, the first time proposition of separation of neural network for the purpose of recovering different contents for SR.In order to satisfy the assumption made in Theorem 1, which concerns the coherence of dictionary elements, we have used skip connections between layers to correlate the outcomes.

This is only done in low coherence network due to inherent lack of correlation of dictionary elements which are input LR images, as our RDD explains.

Since output of each layer of neurons is an input to next layer, acting as a new dictionary, skip layers qualitatively provide the required increase in coherence.

Usage of skip layers have also been proposed by BID14 .

The authors have used skip layers in a very deep network (30 layers) to prevent gradient vanishing problem and propagating information between two different structures (conv and de-conv layers).

In this work we are using skip layers in a network which is required to be as shallow as practically possible to increase coherence of layer outputs.

We also used cross entropy loss besides the MSE loss for low coherence network similar to GAN based algorithms BID7 .We have used bicubically upsampled inputs to the network which is the only pre-processing before neural networks.

The aggregation of two separate network are done in a post-processing block because the training operation uses separate validation data for error gradient calculations.

We have backprojected the results to upsampled input images and then simply added two outputs by giving more weight to high coherence network.

We have used an Intel i7-4770K CPU Nvidia GeForce GTX 760 GPU computer to run the training and testing operations.

Since we do not readily have a GPU implementation, we have given individual run times for each image while comparing speed.

Training is completed in 16 hours for high coherence network and 8 hours for low coherence network which is significantly less then the requirement of state of the art algorithms.

The run times are as fast as twice the speed of reference model BID9 as reported in Table 1 We have used the same 291 image training set from BID9 .

In similar fashion we have rotated and scaled the images to create an augmented set.

Then we have separated patches into two subsets according to their coherence values obtained from upsampled LR images.

Our tests are carried out in scaling factor of 3x.

We have conducted experiments to test out the RDD proposition which states that the learned filters for neurons resemble to the highlighted features from training data.

We have created two separate training set which contained high coherence data with edges of orientation 0-20 degrees and 40-60 degrees.

The results were showing that the learned filters for the first layer resemble the predominant features of the training set as in Figure 3 and Figure 4 5.3 SEPARATE NETWORKS The advantage of separate network training is to be able to recover details that otherwise might be dropped out during training due to stronger data.

Barbara image details can be clearly seen in Figure 7 in appendix, low coherence network output.

The PSNR and SSIM value of Barbara image shows the validity of this example.

Not only coherence but also strength and angle information can be used to divide the training data.

Initial experiments with increased number of parallel networks showed that networks with low strength data, which are almost flat patches, did not converge to a useful state after training.

Also aggregation of networks trained with different orientation edges became cumbersome and such a network was not feasible for a real time application.

We have decided on using two networks whose data are divided according to coherence which was the only option that we could theoretically support.

Numerical comparisons are done only with the referenced algorithm BID9 in Table 1 .

A comparison with previous DLB and DLM can be found in the reference paper BID9 .The proposed network is faster compared to BID9 due to lighter networks.

Since we have split the training set depending on its information content (i.e. textures and edges) both networks require less number of elements to represent the data which yields a faster algorithm.

We have proven that a neuron is able to solve an inverse problem optimally.

By introducing RDD we have shown that CNN layers act as sparse representation solvers.

We have proposed a method that addresses the texture recovery better.

Experiments have shown that RDD is valid and proposed network recovers some texture components better and faster than state of the art algorithms while not sacrificing performance and speed.

In the future we plan to investigate a content-aware aggregation method which might perform better than simple averaging.

We will investigate ways of jointly training or optimizing two networks and including aggregation step inside a unified network.

In parallel we are investigating a better network structure for texture recovery.

Also we are going to incorporate the initial upsampling step into the network by allowing the network to learn its own interpolation kernels.

A VISUAL RESULTS

@highlight

After proving that a neuron acts as an inverse problem solver for superresolution and a network of neurons is guarantied to provide a solution, we proposed a double network architecture that performs faster than state-of-the-art.

@highlight

Discusses using neural networks for super-resolution

@highlight

A new architecture for solving image super-resolution tasks, and an analysis aiming to establish a connection between CNNs for solving super resolution and solving sparse regularized inverse problems.