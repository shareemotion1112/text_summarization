Convolution neural networks typically consist of many convolutional layers followed by several fully-connected layers.

While convolutional layers map between high-order activation tensors, the fully-connected layers operate on flattened activation vectors.

Despite its success, this approach has notable drawbacks.

Flattening discards the multi-dimensional structure of the activations, and the fully-connected layers require a large number of parameters.

We present two new techniques to address these problems.

First, we introduce tensor contraction layers which can replace the ordinary fully-connected layers in a neural network.

Second, we introduce tensor regression layers, which express the output of a neural network as a low-rank multi-linear mapping from a high-order activation tensor to the softmax layer.

Both the contraction and regression weights are learned end-to-end by backpropagation.

By imposing low rank on both, we use significantly fewer parameters.

Experiments on the ImageNet dataset show that applied to the popular VGG and ResNet architectures, our methods significantly reduce the number of parameters in the fully connected layers (about 65% space savings) while negligibly impacting accuracy.

Many natural datasets exhibit pronounced multi-modal structure.

We represent audio spectrograms as 2nd-order tensors (matrices) with modes corresponding to frequency and time.

We represent images as third-order tensors with modes corresponding to width, height and the color channels.

Videos are expressed as 4th-order tensors, and the signal processed by an array of video sensors can be described as a 5th-order tensor.

A broad array of multi-modal data can be naturally encoded as tensors.

Tensor methods extend linear algebra to higher order tensors and are promising tools for manipulating and analyzing such data.

The mathematical properties of tensors have long been the subject of theoretical study.

Previously, in machine learning, data points were typically assumed to be vectors and datasets to be matrices.

Hence, spectral methods, such as matrix decompositions, have been popular in machine learning.

Recently, tensor methods, which generalize these techniques to higher-order tensors, have gained prominence.

One class of broadly useful techniques within tensor methods are tensor decompositions, which have been studied for learning latent variables BID0 .Deep Neural Networks (DNNs) frequently manipulate high-order tensors: in a standard deep convolutional Neural Network (CNN) for image recognition, the inputs and the activations of convolutional layers are 3 rd -order tensors.

And yet, to wit, most architectures output predictions by first flattening the activations tensors and then connecting to the output neurons via one or more fullyconnected layers.

This approach presents several issues: we lose multi-modal information during the flattening process and the fully-connected layers require a large number of parameters.

In this paper, we propose Tensor Contraction Layers (TCLs) and Tensor Regression Layers (TRLs) as end-to-end trainable components of neural networks.

In doing so, we exploit multilinear structure without giving up the power and flexibility offered by modern deep learning methods.

By replacing fully-connected layers with tensor contractions, we can aggregate long-range spatial information while preserving multi-modal structure.

Moreover, by enforcing low rank, we can significantly reduce the number of parameters needed with minimal impact on accuracy.

Our proposed TRL represent the regression weights through the factors of a low-rank tensor decomposition.

The TRL obviates the need for flattening when generating output.

By combining tensor regression with tensor contraction, we further increase efficiency.

Augmenting the VGG and ResNet architectures, we demonstrate improved performance on the ImageNet dataset despite significantly reducing the number of parameters (almost by 65%).

This is the first paper that presents an end-to-end trainable architecture that retains the multi-dimensional tensor structure throughout the network.

Related work: Several recent papers apply tensor decomposition to deep learning.

BID16 propose using CP decomposition to speed up convolutional layers.

BID13 take a pre-trained network and apply tensor (Tucker) decomposition on the convolutional kernel tensors and then fine-tune the resulting network.

BID23 propose weight sharing in multi-task learning and BID2 propose sharing residual units.

These contributions are orthogonal to ours and can be applied together.

BID17 use the Tensor-Train (TT) format to impose low-rank tensor structure on weights.

However, they still retain the fully-connected layers for the output, while we present an end-to-end tensorized network architecture.

Despite the success of DNNs, many open questions remain as to why they work so well and whether they really need so many parameters.

Tensor methods have emerged as promising tools of analysis to address these questions and to better understand the success of deep neural networks.

BID3 , for example, use tensor methods as tools of analysis to study the expressive power of CNNs.

BID5 derive sufficient conditions for global optimality and optimization of non-convex factorization problems, including tensor factorization and deep neural network training.

Other papers investigate tensor methods as tools for devising neural network learning algorithms with theoretical guarantees of convergence BID20 BID11 b) .

Several prior papers address the power of tensor regression to preserve natural multi-modal structure and learn compact predictive models BID4 BID18 BID25 BID24 .

However, these works typically rely on analytical solutions and require manipulating large tensors containing the data.

They are usually used for small dataset or require to downsampled datasets or extract compact features prior to fitting the model, and do not scale to large datasets such as ImageNet.

To our knowledge, no prior work combines tensor contraction or tensor regression with deep learning in an end-to-end trainable fashion.

Throughout the paper, we define tensors as multidimensional arrays, with indexing starting at 0.

First order tensors are vectors, denoted v. Second order tensors are matrices, denoted M and Id is the identity matrix.

We denoteX tensors of order 3 or greater.

For a third order tensor X , we denote its element (i, j, k) asX i1,i2,i3 .

A colon is used to denote all elements of a mode e.g. the mode-1 fibers ofX are denoted asX :,i2,i3 .

The transpose of M is denoted M and its pseudoinverse M † .

Finally, for any i, j ∈ N, [i . .

j] denotes the set of integers {i, i + 1, · · · , j − 1, j}.Tensor unfolding: Given a tensor,X ∈ R I0×I1×···×I N , its mode-n unfolding is a matrix DISPLAYFORM0 I k and is defined by the mapping from element DISPLAYFORM1 DISPLAYFORM2 Tensor vectorization: Given a tensor,X ∈ R I0×I1×···×I N , we can flatten it into a vector vec(X ) of size (I 0 × · · · × I N ) defined by the mapping from element DISPLAYFORM3 For a tensorX ∈ R I0×I1×···×I N and a matrix M ∈ R R×In , the n-mode product of a tensor is a tensor of size (I 0 × · · ·

× I n−1 × R ×

I n+1 ×

· × I N ) and can be expressed using unfolding ofX and the classical dot product as: DISPLAYFORM4 Generalized inner-product For two tensorsX ,Ỹ ∈ R I0×I1×···×I N of same size, their inner product is defined as X ,Ỹ = I0−1 i0=0 DISPLAYFORM5 N ×Dy sharing N modes of same size, we similarly defined the generalized inner product along the N last (respectively first) modes ofX (respectivelỹ DISPLAYFORM6 Tucker decomposition:

Given a tensorX ∈ R I0×I1×···×I N , we can decompose it into a low rank coreG ∈ R R0×R1×···×R N by projecting along each of its modes with projection factors DISPLAYFORM7 In other words, we can write: DISPLAYFORM8 Typically, the factors and core of the decomposition are obtained by solving a least squares problem.

In particular, closed form solutions can be obtained for the factor by considering the n−mode unfolding ofX that can be expressed as: DISPLAYFORM9 Similarly, we can optimize the core in a straightforward manner by isolating it using the equivalent rewriting of the above equality: DISPLAYFORM10 The interested reader is referred to the thorough review of the literature on tensor decompositions by BID14 .

In this section, we explain how to incorporate tensor contractions and tensor regressions into neural networks as differentiable layers.

One natural way to incorporate tensor operations into a neural network is to apply tensor contraction to an activation tensor in order to obtain a low-dimensional representation.

We call this technique the Tensor Contraction layer (TCL).

Compared to performing a similar rank reduction with a fullyconnected layer, TCLs require fewer parameters and less computation.

Tensor contraction layers Given an activation tensorX of size DISPLAYFORM0 , the TCL will produce a compact core tensorG of smaller size DISPLAYFORM1 Note that the projections start at the second mode because the first mode S 0 corresponds to the batch.

k∈ [1,···N ] are learned end-to-end with the rest of the network by gradient backpropagation.

In the rest of this paper, we denote size- In standard CNNs, the inputX is flattened and then passed to a fully-connected layer, where it is multiplied by a weight matrix W. DISPLAYFORM0 Gradient back-propagation In the case of the TCL, we simply need to take the gradients with respect to the factors V (k) for each k ∈ 0, · · · , N of the tensor contraction.

Specifically, we compute DISPLAYFORM1 By rewriting the previous equality in terms of unfolded tensors, we get an equivalent rewriting where we have isolated the considered factor: DISPLAYFORM2 Model complexity Considering an activation tensorX of size DISPLAYFORM3

In order to generate outputs, CNNs typically either flatten the activations or apply a spatial pooling operation.

In either case, the discard all multimodal structure, and subsequently apply a fullconnected output layer.

Instead, we propose leveraging the spatial structure in the activation tensor and formulate the output as lying in a low-rank subspace that jointly models the input and the output.

We do this by means of a low-rank tensor regression, where we enforce a low multilinear rank of the regression weight tensor.

Tensor regression as a layer Let us denote byX ∈ R S,I0×I1×···×I N the input activation tensor corresponding to S samples X 1 , · · · ,X S and Y ∈ R S,O the O corresponding labels for each sample.

We are interested in the problem of estimating the regression weight tensorW ∈ R I0×I1×···×I N ×O under some fixed low rank DISPLAYFORM0 DISPLAYFORM1 Previously, this problem has been studied as a standalone one where the input data is directly mapped to the output, and solved analytically.

However, this requires pre-processing the data to extract (hand-crafted) features to feed the model.

In addition, the analytical solution is prohibitive in terms of computation and memory usage for large datasets.

In this work, we incorporate tensor regressions as trainable layers in neural networks.

We do so by replacing the traditional flattening + fully-connected layers with a tensor regression applied directly to the high-order input and enforcing low rank constraints on the weights of the regression.

We call our layer the Tensor Regression Layer (TRL).

Intuitively, the advantage of the TRL comes from leveraging the multi-modal structure in the data and expressing the solution as lying on a low rank manifold encompassing both the data and the associated outputs.

We propose to first reduce the dimensionality of the activation tensor by applying tensor contraction before performing tensor regression.

We then replace flattening operators and fullyconnected layers by a TRL.

The output is a product between the activation tensor and a low-rank weight tensorW. For clarity, we illustrate the case of a binary classification, where y is a scalar.

For multi-class, y becomes a vector and the regression weights would become a 4 th order tensor.

The gradients of the regression weights and the core with respect to each factor can be obtained by writing: DISPLAYFORM0 Using the unfolded expression of the regression weights, we obtain the equivalent formulation: DISPLAYFORM1 Similarly, we can obtain the gradient with respect to the core by considering the vectorized expressions: DISPLAYFORM2 Model analysis We consider as input an activation tensorX ∈ R S,I0×I1×···×I N , and a rank-(R 0 , R 1 , · · · , R N , R N +1 ) tensor regression layer, where, typically, R k ≤ I k .

Let's assume the output is n-dimensional.

A fully-connected layer takingX as input will have n FC = n × N k=0 I k parameters.

By comparison, the TRL has a number of parameters n TRL , with: DISPLAYFORM3

We empirically demonstrate the effectiveness of preserving the tensor structure through tensor contraction and tensor regression by integrating it into state-of-the-art architectures and demonstrating similar performance on the popular ImageNet dataset.

In particular, we empirically verify the effectiveness of the TCL on VGG-19 BID22 and conduct thorough experiment with the tensor regression on ResNet-50 and ResNet-101 BID6 .

Synthetic data To illustrate the effectiveness of the low-rank tensor regression, we first apply it to synthetic data y = vec(X ) × W where each sampleX ∈ R (64) follows a Gaussian distribution Figure 4: Empirical comparison (4a) of the TRL against regression with a fully-connected layer.

We plot the weight matrix of both the TRL and a fully-connected layer.

Due to its low-rank weights, the TRL better captures the structure in the weights and is more robust to noise.

Evolution of the RMSE as a function of the training set size (4b) for both the TRL and fully-connected regression DISPLAYFORM0 W is a fixed matrix and the labels are generated as y = vec(X ) × W. We then train the data onX +Ẽ, whereẼ is added Gaussian noise sampled from N (0, 3).

We compare i) a TRL with squared loss and ii) a fully-connected layer with a squared loss.

In Figure 4a , we show the trained weight of both a linear regression based on a fully-connected layer and a TRL with various ranks, both obtained in the same setting.

As can be observed in Figure 5b , the TRL is easier to train on small datasets and less prone to over-fitting, due to the low rank structure of its regression weights, as opposed to typical Fully Connected based Linear Regression.

ImageNet Dataset We ran our experiments on the widely-used ImageNet-1K dataset, using several widely-popular network architectures.

The ILSVRC dataset (ImageNet) is composed of 1.2 million images for training and 50, 000 for validation, all labeled for 1,000 classes.

Following BID8 BID6 BID9 BID7 , we report results on the validation set in terms of Top-1 accuracy and Top-5 accuracy across all 1000 classes.

Specifically, we evaluate the classification error on single 224×224 single center crop from the raw input images.

Training the TCL + TRL When experimenting with the tensor regression layer, we did not retrain the whole network each time but started from a pre-trained ResNet.

We experimented with two settings: i) We replaced the last average pooling, flattening and fully-connected layer by either a TRL or a combination of TCL + TRL and trained these from scratch while keeping the rest of the network fixed.

ii) We investigate replacing the pooling and fully-connected layers with a TRL that jointly learns the spatial pooling as part of the tensor regression.

In that setting, we also explore initializing the TRL by performing a Tucker decomposition on the weights of the fully-connected layer.

We implemented all models using the MXNet library BID1 and ran all experiments training with data parallelism across multiple GPUs on Amazon Web Services, with 4 NVIDIA k80 GPUs.

For training, we adopt the same data augmentation procedure as in the original Residual Networks (ResNets) paper BID6 .When training the layers from scratch, we found it useful to add a batch normalization layer BID10 before and after the TCL/TRL to avoid vanishing or exploding gradients, and to make the layers more robust to changes in the initialization of the factors.

In addition we constrain the weights of the tensor regression by applying 2 normalization BID19 to the factors of the Tucker decomposition.

Impact of the tensor contraction layer We first investigate the effectiveness of the TCL using a VGG-19 network architecture BID22 .

This network is especially wellsuited for out methods because of its 138, 357, 544 parameters, 119, 545, 856 of which (more than 80% of the total number of parameters) are contained in the fully-connected layers.

By adding TCL to contract the activation tensor prior to the fully-connected layers we can achieve large space saving.

We can express the space saving of a model M with n M total parameters in its fully-connected layers with respect to a reference model R with n R total parameters in its fully-connected layers as 1 − n M nR (bias excluded).

TAB0 presents the accuracy obtained by the different combinations of TCL in terms of top-1 and top-5 accuracy as well as space saving.

By adding a TCL that preserves the size of its input we are able to obtain slightly higher performance with little impact on the space saving (0.21% of space loss) while by decreasing the size of the TCL we got more than 65% space saving with almost no performance deterioration.

Overcomplete TRL We first tested the TRL with a ResNet-50 and a ResNet-101 architectures on ImageNet, removing the average pooling layer to preserve the spatial information in the tensor.

The full activation tensor is directly passed on to a TRL which produces the outputs on which we apply softmax to get the final predictions.

This results in more parameters as the spatial dimensions are preserved.

To reduce the computational burden but preserve the multi-dimensional information, we alternatively insert a TCL before the TRL.

In TAB1 , we present results obtained in this setting on ImageNet for various configurations of the network architecture.

In each case, we report the size of the TCL (i.e. the dimension of the contracted tensor) and the rank of the TRL (i.e. the dimension of the core of the regression weights).Joint spatial pooling and low-rank regression Alternatively, we can learn the spatial pooling as part of the tensor regression.

In this case, we remove the average pooling layer and feed the tensor Figure 5 : 5a shows the Top-1 accuracy (in %) as we vary the size of the core along the number of outputs and number of channels (the TRL does spatial pooling along the spatial dimensions, i.e., the core has rank 1 along these dimensions).of size (batch size, number of channels, height, width) to the TRL, while imposing a rank of 1 on the spatial dimensions of the core tensor of the regression.

Effectively, this setting simultaneously learns weights for the multi-linear spatial pooling as well as the regression.

In practice, to initialize the weights of the TRL in this setting, we consider the weight of fullyconnected layer from a pre-trained model as a tensor of size (batch size, number of channels, 1, 1, number of classes) and apply a partial tucker decomposition to it by keeping the first dimension (batch-size) untouched.

The core and factors of the decomposition then give us the initialization of the TRL.

The projection vectors over the spatial dimension are then initialize to 1 height and 1 width , respectively.

The Tucker decomposition was performed using TensorLy BID15 .

In this setting, we show that we can drastically decrease the number of parameters with little impact on performance.

In Figure 5 , we show the change of the Top-1 and Top-5 accuracy as we decrease the size of the core tensor of the TRL and also the space savings.

Unlike fully-connected layers, TCLs and TRLs obviate the need to flatten input tensors.

Our experiments demonstrate that by imposing a low-rank constraint on the weights of the regression, we can learn a low-rank manifold on which both the data and the labels lie.

The result is a compact network, that achieves similar accuracies with many fewer parameters.

Going forward, we plan to apply the TCL and TRL to more network architectures.

We also plan to leverage recent work BID21 on extending BLAS primitives to avoid transpositions needed when computing tensor contractions.

<|TLDR|>

@highlight

We propose tensor contraction and low-rank tensor regression layers to preserve and leverage the multi-linear structure throughout the network, resulting in huge space savings with little to no impact on performance.

@highlight

This paper proposes new layer architectures of neural networks using a low-rank representation of tensors

@highlight

This paper incorporates tensor decomposition and tensor regression into CNN by using a new tensor regression layer.