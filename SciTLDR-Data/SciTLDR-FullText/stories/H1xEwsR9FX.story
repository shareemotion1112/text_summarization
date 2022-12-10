For the challenging semantic image segmentation task the best performing models have traditionally combined the structured modelling capabilities of Conditional Random Fields (CRFs) with the feature extraction power of CNNs.

In more recent works however, CRF post-processing has fallen out of favour.

We argue that this is mainly due to the slow training and inference speeds of CRFs, as well as the difficulty of learning the internal CRF parameters.

To overcome both issues we propose to add the assumption of conditional independence to the framework of fully-connected CRFs.

This allows us to reformulate the inference in terms of convolutions, which can be implemented highly efficiently on GPUs.

Doing so speeds up inference and training by two orders of magnitude.

All parameters of the convolutional CRFs can easily be optimized using backpropagation.

Towards the goal of facilitating further CRF research we have made our implementations publicly available.

Semantic image segmentation, which aims to produce a categorical label for each pixel in an image, is a very import task for visual perception.

Convolutional Neural Networks have been proven to be very strong in tackling semantic segmentation tasks BID23 BID5 BID40 .

While simple feed-forward CNNs are extremely powerful in extracting local features and performing good predictions utilizing a small field of view, they lack the capability to utilize context information and cannot model interactions between predictions directly.

Thus it has been suggested that such deep neural networks may not be the perfect model for structured predictions tasks such as semantic segmentation BID40 BID20 BID41 .

Several authors have successfully combined the effectiveness of CNNs to extract powerful features, with the modelling power of CRFs in order to address the discussed issues BID20 Chandra & Kokkinos, 2016; BID41 .

Despite their indisputable success, structured models have fallen out of favour in more recent approaches BID38 BID4 BID40 .We believe that the main reasons for this development are that CRFs are notoriously slow and hard to optimize.

Learning the features for the structured component of the CRF is an open research problem BID37 BID20 and many approaches rely on entirely hand-crafted Gaussian features BID41 BID31 BID5 .

In addition, CRF inference is typically two orders of magnitude slower than CNN inference.

This makes CRF based approaches too slow for many practical applications.

The long training times of the current generation of CRFs also make more in-depth research and experiments with such structured models impractical.

To solve both of these issues we propose to add the strong and valid assumption of conditional independence to the existing framework of fully-connected CRFs (FullCRFs) introduced by .

This allows us to reformulate a large proportion of the inference as convolutions, which can be implemented highly efficiently on GPUs.

We call our method convolutional CRFs (ConvCRFs).

Backpropagation BID30 can be used to train all parameters of the ConvCRF.

Inference in ConvCRFs can be performed in less then 10ms.

This is a speed increase of two-orders of magnitude compared to FullCRFs.

We believe that those fast train and inference speeds will greatly benefit future research and hope that our results help to revive CRFs as a popular method to solve structured tasks.

Recent advances in semantic segmentation are mainly driven by powerful deep neural network architectures BID18 BID32 BID12 BID38 .

Following the ideas introduced by BID23 , transposed convolution layers are applied at the end of the prediction pipeline to produce high-resolution output.

Atrous (dilated) convolutions BID3 BID39 are commonly applied to preserve spatial information in feature space.

Many architectures have been proposed BID25 BID29 BID1 BID27 BID34 BID38 , based on the ideas above.

All of those approaches have in common that they primarily rely on the powerful feature extraction provided by CNNs.

Predictions are pixel-wise and conditionally independent (given the common feature base of nearby pixels).

Structured knowledge and background context is ignored in these models.

One popular way to integrate structured predictions into CNN pipelines is to apply a fully-connected CRF (FullCRF) on top of the CNN prediction BID3 BID41 BID31 BID20 Chandra & Kokkinos, 2016) .

Utilizing the edge-awareness of CRFs, FullCRFs have been successfully utilized to solve weakly-and semisupervised segmentation tasks BID19 BID36 BID13 BID33 .

BID33 propose to use a CRF based loss function.

All of those approaches can benefit from our contributions.

Parameter Learning in CRFs FullCRFs rely on hand-crafted features for the pairwise (Gaussian) kernels.

In their first publication optimized the remaining parameters with a combination of expectation maximization and grid-search.

In a follow-up work BID17 proposed to use gradient descent.

The idea utilizes, that for the message passing the equation (k G * Q) = k G * Q holds.

This allows them to train all internal CRF parameters, using backpropagation, without being required to compute gradients with respect to the Gaussian kernel k G .

However the features of the Gaussian kernel cannot be learned with such an approach.

CRFasRNN BID41 uses the same ideas to implement joint CRF and CNN training.

Like Krähenbühl and Koltuns (2013) approach this requires hand-crafted pairwise (Gaussian) features.

Quadratic optimization (Chandra & Kokkinos, 2016; BID37 has been proposed to learn the Gaussian features of FullCRFs.

These approaches however do not fit well into many deep learning pipelines.

Another way of learning the pairwise features is piecewise training BID20 ).

An additional advantage of this method is that it avoids repeated CRF inference, speeding up the training considerably.

This approach is however of an approximate nature and inference speed is still very slow.

Inference speed of CRFs In order to circumvent the issue of very long training and inference times, some CRF based pipelines produce an output which is down-sampled by a factor of 8 × 8 (Chandra & Kokkinos, 2016; BID20 .

This speeds up the inference considerably.

However this harms their predictive capabilities.

Deep learning based semantic segmentation pipelines perform best when they are challenged to produce a full-resolution prediction BID23 BID39 BID4 .

To the best of our knowledge, no significant progress in inference speed has been made since the introduction of FullCRFs .

In the context of semantic segmentation most recent CRF based approaches are based on the Fully Connected CRF (FullCRF) model introduced by .

Consider an input image I consisting of n pixels and a segmentation task with k classes.

A segmentation of I is then modelled as a random field X = {X 1 , . . .

, X n }, where each random variable X i takes values in {1, . . .

, k}, i.e. the label of pixel i. Solving argmax X P (X|I) then leads to a segmentation X of the input image I. P (X|I) is modelled as a CRF over the Gibbs distribution: DISPLAYFORM0 where the energy function E(x|I) is given by DISPLAYFORM1 The function ψ u (x i |I) is called unary potential.

The unary itself can be considered a segmentation of the image and any segmentation pipeline can be used to predict the unary.

In practise most newer approaches BID5 BID31 BID41 utilize CNNs to compute the unary.

The function ψ p (x i , x j |I) is the pairwise potential.

It accounts for the joint distribution of pixels i, j.

It allows us to explicitly model interactions between pixels, such as pixels with similar colour are likely the same class.

In FullCRFs ψ p is defined as weighted sum of Gaussian kernels k DISPLAYFORM2 where w (m) are learnable parameters.

The feature vectors f I i can be chosen arbitrarily and may depend on the input Image I.

The function µ(x i , x j ) is the compatibility transformation, which only depends on the labels x i and x j , but not on the image I.A very widely used compatibility function BID5 BID41 ) is the Potts model µ(x i , x j ) = 1 [xi =xj] .

This model tries to assign pixels with similar features the same prediction.

BID41 propose to use 1 × 1 convolutions as compatibility transformation.

Such a function allows the model to learn more structured interactions between predictions.

FullCRFs utilize two Gaussian kernels with hand-crafted features.

The appearance kernel uses the raw colour values I j and I i as features.

The smoothness kernel is based on the spatial coordinates p i and p j .

The entire pairwise potential is then given as: DISPLAYFORM3 where w (1) , w (2) , as well as θ α , θ β and θ γ are the only learnable parameters of the model.

Most CRF based segmentation approaches BID5 BID41 BID31 utilize the very same handcrafted pairwise potentials proposed by .

CRFs are notoriously hard to optimize and utilizing hand-crafted features circumvents this problem.

Inference in FullCRFs is achieved using the mean field algorithm (see Algorithm 1).

All steps of algorithm 1, other then the message passing, are highly parallelized and can be implemented easily and efficiently on GPUs using standard deep learning libraries. (For details see BID41 ).The message passing however is the bottleneck of the CRF computation.

Exact computation is quadratic in the number of pixels and therefore infeasible.

instead proposed to utilize the permutohedral lattice BID0 approximation, a high-dimensional filtering algorithm.

The permutohedral lattice however is based on a complex data structure.

While there is a very sophisticated and fast CPU implementation, the permutohedral lattice does not follow the SIMD BID24 paradigm of efficient GPU computation.

In addition, efficient gradient computation of the permutohedral lattice approximation, is also a non-trivial problem.

This is the underlying reason why FullCRF based approaches use hand-crafted features.

DISPLAYFORM0 e.g. softmax 7: end while

The convolutional CRFs (ConvCRFs) supplement FullCRFs with a conditional independence assumption.

We assume that the label distribution of two pixels i, j are conditionally independent, if for the Manhattan distance d holds d(i, j) > k. We call the hyperparameter k filter-size.

This locality assumption is a very strong assumption.

It implies that the pairwise potential is zero, for all pixels whose distance exceed k. This reduces the complexity of the pairwise potential greatly.

The assumption can also be considered valid, given that CNNs are based on local feature processing and are highly successful.

This makes the theoretical foundation of ConvCRFs very promising, strong and valid assumptions are the powerhouse of machine learning modelling.

One of the key contribution of this paper is to show that exact message passing is efficient in ConvCRFs.

This eliminates the need to use the permutohedral lattice approximation, making highly efficient GPU computation and complete feature learning possible.

Towards this goal we reformulate the message passing step to be a convolution with truncated Gaussian kernel and observe that this can be implemented very similar to regular convolutions in CNNs.

Consider an input P with shape [bs, c, h, w] where bs, c, h, w denote batch size, number of classes, input height and width respectively.

For a Gaussian kernel g defined by feature vectors f 1 . . .

f d , each of shape [bs, h, w] we define its kernel matrix by DISPLAYFORM0 where θ i is a learnable parameter.

For a set of Gaussian kernels g 1 . . .

g s we define the merged kernel matrix DISPLAYFORM1 The result Q of the combined message passing of all s kernels is now given as: DISPLAYFORM2 This message passing operation is similar to standard 2d-convolutions of CNNs.

In our case however, the filter values depend on the spatial dimensions x and y. This is similar to locally connected layers BID6 .

Unlike locally connected layers (and unlike 2d-convolutions), our filters are however constant in the channel dimension c. One can view our operation as convolution over the dimension c 1 .It is possible to implement our convolution operation by using standard CNN operations only.

This however requires the data to be reorganized in GPU memory several times, which is a very slow process.

Profiling shows that 90 % of GPU time is spend for the reorganization of data.

We therefore opted to build a native low-level implementation, to gain an additional 10-fold speed up.

Efficient computation of our convolution can be implemented analogously to 2d-convolution (and locally connected layers).

The first step is to tile the input P in order to obtain data with shape [bs, c, k, k, h, w].

This process is usually referred to as im2col and the same as in 2d-convolutions BID7 .

2d-convolutions proceed by applying a batched matrix multiplication over the spatial dimension.

We replace this step with a batched dot-product over the channel dimension.

All other steps are the same.

For the sake of comparability we use the same design choices as FullCRFs in our baseline ConvCRF implementation.

In particular, we use softmax normalization, the Potts model as well as the same handcrafted gaussian features as proposed by .

Analogous to we also apply gaussian blur to the pairwise kernels.

This leads to an increase of the effective filter size by a factor of 4.In additional experiments we investigate the capability of our CRFs to learn Gaussian features.

Towards this goal we replace the input features p i of the smoothness kernel with learnable variables.

Those variables are initialized to the same values as the hand-crafted version, but are adjusted as part of the training process.

We also implement a learnable compatibility transformation using 1 × 1 convolution, following the ideas of BID41 .

Dataset: We evaluate our method on the challenging PASCAL VOC 2012 (Everingham et al.) image dataset.

Following the literature BID23 BID38 BID5 BID40 we use the additional annotation provided by (Hariharan et al., 2011) resulting in 10 582 labelled images for training.

Out of those images we hold back 200 images to fine-tune the internal CRF parameters and use the remaining 10 382 to train the unary CNN.

We report our results on the 1464 images of the official validation set.

We train a ResNet101 BID12 to compute the unary potentials.

We use the ResNet101 implementation provided by the PyTorch BID28 repository.

A simple FCN BID23 is added on top of the ResNet to decode the CNN features and obtain valid segmentation predictions.

The network is initialized using ImageNet Classification weights BID8 ) and then trained on Pascal VOC data directly.

Unlike many other projects, we do not train the network on large segmentation datasets such as MS COCO BID21 , but only use the images provided by the PASCAL VOC 2012 benchmark.

The CNN is trained for 200 epochs using a batch size of 16 and the adam optimizer BID14 .

The initial learning rate is set to 5 × 10 −5 and polynomially decreased BID22 BID5 by multiplying the initial learning rate with ((1 − step max_steps ) 0.9 ) 2 .

An L 2 weight decay with factor 5 × 10 −4 is applied to all kernel weights and 2d-Dropout BID35 with rate 0.5 is used on top of the final convolutional layer.

The same hyperparamters are also used for the end-to-end training.

The following data augmentation methods are applied: Random horizontal flip, random rotation (±10°) and random resize with a factor in (0.5, 2).

In addition the image colours are jittered using random brightness, random contrast, random saturation and random hue.

All random numbers are generated using a truncated normal distribution.

The trained model achieves validation mIoU of 71.23 % and a train mIoU of 91.84 %.CRF: Following the literature BID5 BID41 BID20 , the mean-field inference of the CRF is computed for 5 iterations in all experiments.

To show the capabilities of Convolutional CRFs we first evaluate their performance on a synthetic task.

We use the PASCAL VOC (Everingham et al.) dataset as a basis, but augment the ground-truth towards the goal to simulate prediction errors.

The noised labels are used as unary potentials for the DISPLAYFORM0 Figure 1: Visualization of the synthetic task.

Especially in the last example, the artefacts from the permutohedral lattice approximation can clearly be seen at object boundaries.

CRF, the CRF is then challenged to denoise the predictions.

The output of the CRF is then compared to the original label of the Pascal VOC dataset.

Towards the goal of creating a relevant task, the following augmentation procedure is used: First the ground-truth is down-sampled by a factor of 8.

Then, in low-resolution space random noise is added to the predictions and the result is up-sampled to the original resolution again.

This process simulates inaccuracies as a result of the low-resolution feature processing of CNNs as well as prediction errors similar to the checkerboard issue found in deconvolution based segmentation networks BID10 BID26 .

Some examples of the augmented ground-truth are shown in Figure 1 .In our first experiment we compare FullCRFs and ConvCRFs using the exact same parameters.

To do this we utilize the hand-crafted Gaussian features.

The remaining five parameters (namely w (1) , w (2) , as well as θ α , θ β and θ γ ) are initialized to the default values proposed by .

Note that this gives FullCRFs a natural advantage.

The performance of CRFs however is very robust with respect to these five parameters .The results of our first experiment are given in Table 2 : Performance comparison of CRFs on validation data using decoupled training.

+C uses convolutions as compatibility transformation and +T learns the Gaussian features.

The same unaries were used for all approaches, only the CRF code from DeepLab was utilized.performance of ConvCRFs with the same parameters can be explained by our exact message passing, which avoids the approximation errors compared of the permutohedral lattice approximation.

We provide a visual comparison in Figure 1 where ConvCRF clearly provide higher quality output.

The FullCRF output shows approximation artefacts at the boundary of objects.

In addition we note that ConvCRFs are faster by two orders of magnitude, making them favourable in almost every use case.

In this section we discuss our experiments on Pascal VOC data using a two stage training strategy.

First the unary CNN model is trained to perform semantic segmentation on the Pascal VOC data.

Those parameters are then fixed and in the second stage the internal CRF parameters are optimized with respect to the CNN predictions.

The same unary predictions are used across all experiments, to reduce variants between runs.

Decoupled training has various merits compared to an end-to-end pipeline.

Firstly it is very flexible.

A standalone CRF training can be applied on top of any segmentation approach.

The unary predictions are treated as a black-box input for the CRF training.

In practice this means that the two training stages do not need to interface at all, making fast prototyping very easy.

Additionally decoupled training keeps the system interpretable.

Lastly, piecewise training effectively tackles the vanishing gradient problem BID2 , which is still an issue in CNN based segmentation approaches BID38 .

This leads to overall faster, more robust and reliable training.

For our experiments we train the CRF models on the 200 held-out images from the training set and evaluate the CRF performance on the 1464 images of the official Pascal VOC dataset.

We compare the performance of the ConvCRF with filter size 11 to the unary baseline results as well as a FullCRF trained following the methodology of DeepLab BID5 .We report our results in Table 2 , the training curves are visualized in Figure 2 .

In all experiments, applying CRFs boost the performance considerably.

The experiments also confirm the observation of implementation utilizing a learnable compatibility transformation as well as learnable Gaussian features performs best.

Model output is visualized in FIG0 .

In this section we discuss our experiments using an end-to-end learning strategy for ConvCRFs.

In end-to-end training the gradients are propagated through the entire pipeline.

This allows the CNN and CRF model to co-adapt and therefore to produce the optimum output w.r.t the entire network.

The down-side of end-to-end training is that the gradients need to be propagated through five iterations of the mean-field inference, resulting in vanishing gradients BID41 .We train our network for 250 epochs using a training protocol similar to CRFasRNN BID41 .

Zheng et al. propose to first train the unary potential until convergence and then optimizing the CRF and CNN jointly.

Like Zheng et al. FORMULA0 we greatly reduce the learning rate to 10 −10 during the second training stage.

We use a batch size of 16 for the first and 8 for the second training stage.

In this regard we differ from BID41 , who proposes to reduce the batch size to 1 for the second training stage.

The entire training process takes about 30 hours using four 1080Ti GPUs in parallel.

We believe that the fast training and inference speeds will greatly benefit and ease future research using CRFs.

We compare our ConvCRF to the approach proposed in CRFasRNN BID41 and report the results in TAB3 .

Overall we see that ConvCRF slightly outperforms CRFasRNN at a much higher speed.

In this work we proposed Convolutional CRFs, a novel CRF design.

Adding the strong and valid assumption of conditional independence enables us to remove the permutohedral lattice approximation.

This allows us to implement the message passing highly efficiently on GPUs as convolution operations.

This increases training and inference speed by two orders of magnitude.

In addition we observe a modest accuracy improvement when computing the message passing exactly.

Our method also enables us to easily train the Gaussian features of the CRF using backpropagation.

In future work we will investigate the potential of learning Gaussian features further.

We are also going to examine more sophisticated CRF architectures, towards the goal of capturing context information even better.

Lastly we are particularly interested in exploring the potential of ConvCRFs in other structured applications such as instance segmentation, landmark recognition and weakly supervised learning.

@highlight

We propose Convolutional CRFs a fast, powerful and trainable alternative to Fully Connected CRFs.

@highlight

The authors replace the large filtering step in the permutohedral lattice with a spatially varying convolutional kernel and show that inference is more efficient and training is easier. 

@highlight

Proposes to perform message passing on a truncated Gaussian kernel CRF using a defined kernel and parallelized message passing on GPU.