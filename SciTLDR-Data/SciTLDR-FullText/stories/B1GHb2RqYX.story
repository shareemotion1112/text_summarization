In this work, we propose the polynomial convolutional neural network (PolyCNN), as a new design of a weight-learning efficient variant of the traditional CNN.

The biggest advantage of the PolyCNN is that at each convolutional layer, only one convolutional filter is needed for learning the weights, which we call the seed filter, and all the other convolutional filters are the polynomial transformations of the seed filter, which is termed as an early fan-out.

Alternatively, we can also perform late fan-out on the seed filter response to create the number of response maps needed to be input into the next layer.

Both early and late fan-out allow the PolyCNN to learn only one convolutional filter at each layer, which can dramatically reduce the model complexity by saving 10x to 50x parameters during learning.

While being efficient during both training and testing, the PolyCNN does not suffer performance due to the non-linear polynomial expansion which translates to richer representational power within the convolutional layers.

By allowing direct control over model complexity, PolyCNN provides a flexible trade-off between performance and efficiency.

We have verified the on-par performance between the proposed PolyCNN and the standard CNN on several visual datasets, such as MNIST, CIFAR-10, SVHN, and ImageNet.

Applications of deep convolutional neural networks (CNNs) have been overwhelmingly successful in all aspect of perception tasks, ranging from computer vision to speech recognition and understanding, from biomedical data analysis to quantum physics.

In the past couple of years, we have seen the evolution of many successful CNN architectures such as AlexNet BID13 , VGG BID25 , Inception , and ResNet BID8 a) .

However, training these networks end-to-end with fully learnable convolutional filters (as is standard practice) is still very computationally expensive and is prone to over-fitting due to the large number of parameters.

To alleviate this issue, we have come to think about this question: can we arrive at a more efficient CNN in terms of learnable parameters, without sacrificing the high CNN performance?In this paper, we present an alternative approach to reducing the computational complexity of CNNs while performing as well as standard CNNs.

We introduce the polynomial convolutional neural networks (PolyCNN).

The core idea behind the PolyCNN is that at each convolutional layer, only one convolutional filter is needed for learning the weights, which we call the seed filter, and all the other convolutional filters are the polynomial transformations of the seed filter, which is termed as an early fan-out.

Alternatively, we could also perform late fan-out on the seed filter response to create the number of response maps desired to be input into the next layer.

Both early and late fan-out allow the PolyCNN to learn only one convolutional filter at each layer, which can dramatically reduce the model complexity.

Parameter savings of at least 10??, 26??, 50??, etc. can be realized during the learning stage depending on the spatial dimensions of the convolutional filters (3 ?? 3, 5 ?? 5, 7 ?? 7 etc.

sized filters respectively).

While being efficient during both training and testing, the PolyCNN does not suffer performance due to the non-linear polynomial expansion which translates to richer representational power within the convolutional layers.

We have verified the on-par performance between the proposed PolyCNN and the standard CNN on several visual datasets, such as MNIST, CIFAR-10, SVHN, and ImageNet.

DISPLAYFORM0 PolyCNN Module (Late Fan-Out, Single-Seed) PolyCNN Module (Early Fan-Out, Single-Seed) x l x l+1PolyCNN Module (Early Fan-Out, Multi-Seed) x l x l+1PolyCNN Module (Late Fan-Out, Multi-Seed) (e) (

Response Map DISPLAYFORM0 The corresponding filter structure is shown in Figure 6 .10.

The objective is to find the filters h i (m, n) such that structure shown in Figure 6 .10 2 PROPOSED METHOD 2.1 POLYNOMIAL CONVOLUTIONAL NEURAL NETWORKS Two decades ago, BID17 generalized the traditional correlation filter and created the polynomial correlation filter (PCF), whose fundamental difference is that the correlation output from a PCF is a nonlinear function of the input.

As shown in Figure 2 (R), the input image x undergoes a set of point-wise nonlinear transformation (polynomial) for augmenting the input channels.

Based on some pre-defined objective function, usually in terms of simultaneously maximizing average correlation peak and minimizing some correlation filter performance criterion such as average similarity measure (ASM) BID18 , output noise variance (ONV) BID27 , the average correlation energy (ACE) BID27 , or any combination thereof, the filters h 1 , h 2 , . . .

, h N can be solved in closed-form BID17 BID27 BID0 .We draw inspiration from the design principles of the polynomial correlation filter and propose the polynomial convolutional neural network (PolyCNN) as a weight-learning efficient variant of the traditional convolutional neural networks.

The core idea of PolyCNN is that at each convolutional layer, only one convolutional filter (seed filter) needs to be learned, and we can augment other filters by taking point-wise polynomials of the seed filter.

The weights of these augmented filters need not to be updated during the network training.

When convolved with the input data, the learnable seed filter and k non-learnable augmented filters result in (k + 1) response maps.

We call this procedure: early fan-out.

Similarly, one can instead fan-out the response map from the seed filter to create (k + 1) response maps for the subsequent layers.

We call this procedure: late fan-out.

The details of both early and late fan-out are shown in the following sections.

The PolyCNN pipelines are depicted in FIG0 with distinctions between early and late fan-out, as well as single-seed vs. multi-seed cases.

Figure 2 (L) shows the polynomial expansion of seed filters as well as seed filter response maps.

At any given layer, given the seed weights w i for that layer, we generate many new filter weights.

The weights are generated via a non-linear transformation v = f (w i ) of the weights.

The convolutional outputs are computed as follows (1-D signals for simplicity): DISPLAYFORM0 where x j is the j th channel of the input image and w j i is the j th channel of the i th filter.

During the forward pass weights are generated from the seed convolutional kernel and are then convolved with the inputs i.e., DISPLAYFORM1 where we normalize the response maps to prevent the responses from vanishing or exploding and the normalized response map is now called v. f (??) is a non-linear function that operates on each element of w. Backpropagating through this filter transformation necessitates the computation of ???l ???w and ???l ???x .

DISPLAYFORM2 Plug in the normalized response map, we have: DISPLAYFORM3 Similarly, we can compute the gradient with respect to input x as follows: DISPLAYFORM4

At any given layer, we compute the new feature maps from the seed feature maps via non-linear transformations of the feature maps.

The forward pass for this layer involves the application of the following non-linear function DISPLAYFORM0 DISPLAYFORM1 where we normalize the response maps to prevent the responses from vanishing or exploding and the normalized response map is now called t. Backpropagating through such a transformation of the response maps requires the computation of DISPLAYFORM2

The core idea of the PolyCNN 1 is to restrict the network to learn only one (or a few) convolutional filter at each layer, and through polynomial transformations we can augment the convolutional filters, or the response maps.

The gist is that the augmented filters do not need to be updated or learned during the network back-propagation.

As shown in FIG0 , the basic module of PolyCNN (early fan-out, single-seed) starts with just one learnable convolutional filter W l , which we call the seed filter.

If we desire m filters in total for one layer, the remaining m ??? 1 filters are non-learnable and are the polynomial transformation of the seed filter W l .

The input image x l is filtered by these convolutional filters and becomes m response maps, which are then passed through a non-linear activation gate, such as ReLU, and become m feature maps.

Optionally, these m feature maps can be further lineally combined using m learnable weights, which is essentially another convolution operation with filters of size 1 ?? 1.Compared to the CNN module under the same structure (with 1 ?? 1 convolutions), the number of learnable parameters is significantly smaller in PolyCNN.

Let us assume that the number of input and output channels are p and q.

Therefore, the size of each 3D filter in both CNN and PolyCNN is p ?? h ?? w, where h and w are the spatial dimensions of the filter, and there are m such filters.

The 1 ?? 1 convolutions act on the m filters and create the q-channel output.

For standard CNN, the number of learnable weights is p ?? h ?? w ?? m + m ?? q. For PolyCNN, the number of learnable weights is p ?? h ?? w ?? 1 + m ?? q. For simplicity let us assume p = q, which is usually the case for multi-layer CNN architecture.

Then we have the parameter saving ratio: DISPLAYFORM0 and when the spatial filter size h = w = 3 and the number of convolutional filters desired for each layer m 3 2 , we have the parameter saving ratio ?? = 10m m+9 ??? 10.

Similarly for spatial filter size h = w = 5 and m 5 2 , the parameter saving ratio ?? = 26m m+25 ??? 26.

For spatial filter size h = w = 7 and m 7 2 , the parameter saving ratio ?? = 50m m+49 ??? 50.

If we do not include the 1 ?? 1 convolutions for both standard CNN and PolyCNN, and thus make m = q = p, readers can verify that the parameter saving ratio ?? becomes m. Numerically, PolyCNN saves around 10??, 26??, and 50?? parameters during learning for 3 ?? 3, 5 ?? 5, and 7 ?? 7 convolutional filters respectively.

The aforementioned calculation also applies to late fan-out of the PolyCNN.

The training of the PolyCNN is quite straightforward, where the back-propagation is the same for the learnable weights and the augmented weights that do not update.

Gradients get propagated through the polynomial augmented filters just like they would with learnable filters.

This is similar to propagating gradients through layers without learnable parameters e.g., ReLU, Max Pooling etc.).

However, we do not compute the gradient with respect to the fixed filters nor update them during the training process.

The non-learnable filter banks (tensor) of size p ?? h ?? w ?? (m ??? 1) (assuming a total of m filters in each layer) in the PolyCNN can be generated by taking polynomial transformations from the seed filter, by raising to some exponents, which can either be integer exponents, or fractional exponents that are randomly sampled from a distribution.

Strictly speaking, to qualify for polynomials, only non-negative integer powers are allowed.

In this work, without violating the forward and backward pass derivation, we allow the exponents to take negative numbers (relating to Laurent series), and even fractional numbers (relating to Puiseux series).

In this section, we will first analyze the PolyCNN layer and how the early fan-out and late fan-out can very well approximate the standard convolutional layer.

Then, we will extend the formulation to a generalized convolutional layer representation.

At layer l, let x ?? ??? R (p??h??w)??1 be a vectorized single patch from the p-channel input maps at location ??, where h and w are the spatial sizes of the convolutional filter.

Let w ??? R (p??h??w)??1 be a vectorized single convolution filter from the convolutional filter tensor W ??? R p??h??w??m which contains a total of m fan-out convolutional filters at layer l.

We drop the layer subscription l for brevity.

In a standard CNN, this patch x ?? is taken dot-product with (projected onto) the filter w, followed by the non-linear activation resulting in a single output feature value d ?? , at the corresponding location ?? on the feature map.

Similarly, each value of the output feature map is a direct result of convolving the entire input map x with a convolutional filter w.

This microscopic process can be expressed as: DISPLAYFORM0 Without loss of generality, we assume single-seed PolyCNN case for the following analysis.

For an early fan-out PolyCNN layer, a single seed filter w S is expanded into a set of m convolutional filters W ??? R m??p??h??w where w i = w?????i S , and the power terms ?? 1 , ?? 2 , . . .

, ?? m are pre-defined and are not updated during training.

The ??? is again the Hadamard power.

The corresponding output feature map value d early ?? for early fan-out PolyCNN layer is a linear combination of multiple elements from the intermediate response maps (implemented as 1 ?? 1 convolution).

Each slice of this response map is obtained by convolving the input map x with W , followed by a non-linear activation.

The corresponding output feature map value d early ?? is thus obtained by linearly combining the m response maps via 1 ?? 1 convolution with parameters ?? 1 , ?? 2 , . . .

, ?? m .

This entire process can be expressed as: DISPLAYFORM1 where W is now a 2D matrix of size m ?? (p ?? h ?? w) with m filters vec(w i ) stacked as rows, with a slight abuse of notation.

DISPLAYFORM2 DISPLAYFORM3 Comparing Equation 17 and 18, we consider the following two cases (it is obvious that the approximation does not hold when c relu = 0.

Therefore, under the mild assumption that c relu is not an all-zero vector, the approximation d DISPLAYFORM4 For the late fan-out PolyCNN layer, a single response map is a direct result of convolving the input map x with the seed convolutional filter w S .

Then, we obtain a set of m response maps by expanding the response map with Hadamard power coefficients ?? 1 , ?? 2 , . . .

, ?? m which are pre-defined and not updated during training, just like in the early fan-out case.

The corresponding output feature map value d late ?? is also a linear combination of the corresponding elements from the m response maps via 1 ?? 1 convolution with parameters ?? 1 , ?? 2 , . . .

, ?? m .

This process follows: DISPLAYFORM5 where ?? ??i (??) is the point-wise polynomial expansion with Hadamard power ?? i .

With similar reasoning and the mild assumption that c relu is not an all-zero vector, the approximation d late ????? d ?? will hold.

The formulation of PolyCNN discussed in the previous sections has facilitated the forming of a general convolution layer.

To simplify the notation, we use 1D convolution as an example.

The idea can be extended to 2D and higher dimensional convolution as well.

Here is a description of a general convolutional layer: DISPLAYFORM0 where ?? k (??) and ?? l (??) are kernel functions, ??(??) is a non-linearity and ?? kl are linear weights.

If both ?? k (??) and ?? l (??) are linear functions then the expression reduces to a module consisting of convolutional layer, non-linear activation and 1 ?? 1 convolutions.

Under this general setting, the learnable parameters are ?? and w. Setting the parameters w to fixed sparse binary values would allow us to arrive a general version of local binary CNN (Juefei-Xu et al., 2017).

Now we consider some special cases.

If ?? k (??) is a linear function, then the expression reduces to: DISPLAYFORM1 Similarly if ?? l (??) is a linear function, then the expression reduces to: DISPLAYFORM2 At location ?? of the input x, the convolutions can be reduced to: DISPLAYFORM3 where K kl is a base kernel function defined between w and image patch x ?? at the ?? location.

The base kernel can take many forms, including polynomials, random Fourier features, Gaussian radial basis functions, etc., that adhere to the Mercer's theorem BID24 .

The weights ?? kl can be learned via 1 ?? 1 convolutions.

In general ?? kl > 0 must hold for a valid overall kernel function, but perhaps this can be relaxed or imposed during the learning process.

We can also think of Equation FORMULA3 BID23 .

The MNIST dataset contains a training set of 60K and a testing set of 10K 32 ?? 32 gray-scale images showing hand-written digits from 0 to 9.

SVHN is also a widely used dataset for classifying digits, house number digits from street view images in this case.

It contains a training set of 604K and a testing set of 26K 32 ?? 32 color images showing house number digits.

CIFAR-10 is an image classification dataset containing a training set of 50K and a testing set of 10K 32 ?? 32 color images, which are across the following 10 classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The ImageNet ILSVRC-2012 classification dataset consists of 1000 classes, with 1.28 million images in the training set and 50K images in the validation set, where we use for testing as commonly practiced.

For faster roll-out, we first randomly select 100 classes with the largest number of images (1300 training images in each class, with a total of 130K training images and 5K testing images.), and report top-1 accuracy on this subset.

Full ImageNet experimental results are also reported in the subsequent section.

Conceptually PolyCNN can be easily implemented in any existing deep learning framework.

Since the convolutional weights are fixed, we do not have to compute the gradients nor update the weights.

This leads to savings both from a computational point of view and memory as well.

We have used a custom implementation of backpropagation through the PolyCNN layers that is 3x-5x more efficient than autograd-based back propagation in PyTorch.

We base the model architectures we evaluate in this paper on ResNet BID7 , with default 3 ?? 3 filter size.

Our basic module is the PolyCNN module shown in FIG0 along with an identity connection as in ResNet.

We experiment with different numbers of PolyCNN layers, 10, 20, 50, and 75, which is equivalent to 20, 40, 100, and 150 convolutional layers (1 ?? 1 convolution counted).For PolyCNN, the convolutional weights are generated following the procedure described in Section 2.5.

We use 511 randomly sampled fractional exponents for creating the polynomial filter weights (512 convolutional filters in total at each layer), for all of our MNIST, SVHN, and CIFAR-10 experiments.

Spatial average pooling is adopted after the convolution layers to reduce the spatial dimensions of the image to 6 ?? 6.

We use a learning rate of 1e-3 and following the learning rate decay schedule from BID7 .

We use ReLU nonlinear activation and batch normalization BID10 after PolyCNN convolutional module.

For our experiments with ImageNet-1k, we experiment with ad hoc CNN architectures such as the AlexNet and the native ResNet family, including .

Standard CNN layers are thus replaced with the proposed PolyCNN layers.

BID3 98.99 97.85 91.73 BNN BID2 98.60 97.49 89.85 ResNet BID8 / / 93.57 Maxout BID5 99.55 97.53 90.65 NIN BID16 99

For a fair comparison and to quantify the exact difference between our PolyCNN approach and traditional CNN, we compare ours against the exact corresponding network architecture with dense and learnable convolutional weights.

We also use the exact same data and hyper-parameters in terms of the number of convolutional weights, initial learning rate and the learning rate schedule.

In this sense, PolyCNN enjoys 10??, 26??, 50??, etc.

savings in the number of learnable parameters because the baseline CNNs also have the 1 ?? 1 convolutional layer.

The best performing single-seed PolyCNN models in terms of early fan-out are:??? For MNIST: 75 PolyCNN layers, m = 512, q = 256, 128 hidden units in the fc layer.??? For SVHN: 50 PolyCNN layers, m = 512, q = 256, 512 hidden units in the fc layer.??? For CIFAR-10: 50 PolyCNN layers, m = 512, q = 384, 512 hidden units in the fc layer.

TAB0 consolidates the images classification accuracies from our experiments.

The best performing PolyCNNs are compared to their particular baselines, as well as the state-of-the-art methods such as BinaryConnect BID3 , Binarized Neural Networks (BNN) BID2 , ResNet BID8 ), Maxout Network (Goodfellow et al., 2013 , Network in Network (NIN) BID16 .

The network structure for the late fan-out follows that of the early fan-out.

As can be seen, performance from late fan-out is slightly inferior, but early fan-out reaches on-par performance while enjoying huge parameter savings.

TAB1 compares the accuracy on CIFAR-10 achieved by various single-seed PolyCNN architectures (both early and late fan-out) as well as their standard CNN counterparts.

We can see that for a fixed number of convolution layers and filters, the more output channels q leads to higher performance.

Also, PolyCNN (early fan-out) is on par with the CNN counterpart, while saves 10?? parameters.

As can be seen from TAB1 and FIG3 (L), the early fan-out version of the PolyCNN is quite comparable to the standard CNN, and is better than its late fan-out counterpart.

Here we report CIFAR-10 accuracy by varying the number of seed filters in TAB2 .

The network has 20 PolyCNN layers, and the total number of filters per layer is set to 512.

We now vary the number of seed filters from 1 to 512, by a factor of 2.

So when the number of seed filters is approaching 512, PolyCNN reduces to standard CNN.

As can be seen, as we increase the number of seed filters, we are essentially increase the model complexity and the performance is rising monotonically.

This experiment will provide insight into trading-off between performance and model complexity.

We report the top-1 accuracy on 100-Class subset of ImageNet 2012 classification challenge dataset in TAB3 .

The input images of ImageNet is much larger than those of MNIST, SVHN, and CIFAR-10, which allows us to experiments with various convolutional filter sizes.

Both the PolyCNN and our baseline share the same architecture: 20 PolyCNN layers, 512 convolutional filters, 512 output channels, 4096 hidden units in the fully connected layer.

For this experiment, we omit the late fan-out and only use the better performing early fan-out version of the PolyCNN.

The first ad hoc network architecture we experiment with is the AlexNet BID13 .

We train a PolyCNN version of the AlexNet to take on the full ImageNet classification task.

The AlexNet architecture is comprised of five consecutive convolutional layers, and two fully connected layers, mapping from the image (224??224??3) to the 1000-dimension feature for the classification purposes in the forward pass.

The number of convolutional filters used and their spatial sizes are tabulated in TAB4 .

For this experiment, we create a single-seed PolyCNN (early fan-out) counterpart following the AlexNet architecture.

For each convolutional layer in AlexNet, we keep the same input and output channels.

Replacing the traditional convolution module with PolyCNN, we are allowed to specify another hyper-parameter, the fan-out channel m. TAB4 shows the comparison of the number of learnable parameters in convolutional layers in both AlexNet and its PolyCNN counterpart, by setting fan-out channel m = 256.

As can be seen, PolyCNN saves about 6.4873?? learnable parameters in the convolutional layers.

What's important is that, by doing so, PolyCNN does not suffer the performance as can be seen in FIG3 (R) and TAB5 .

We have plotted accuracy curves and loss curves after 55 epochs for both the AlexNet and its PolyCNN counterpart.

The second ad hoc network architecture we experiment with is the native ResNet family.

We create a single-seed PolyCNN (early fan-out) counterpart following the architectures, with the same number of input and output channels.

The number of convolutional filters in each layer is equivalent for both models.

The two baselines are the CNN ResNet implemented by ourselves and by Facebook BID4 .

TAB6 shows the top-1 accuracy on the two baselines as well as the PolyCNN.

Since ResNet is primarily composed of 3 ?? 3 convolutional filter, the PolyCNN enjoys around 10x parameters savings while achieving competitive performance.

We have shown the effectiveness of the proposed PolyCNN.

Not only can it achieve on-par performance with the state-of-the-art, but also enjoy a significant utility savings.

The PyTorch implementation of the PolyCNN will be made publicly available.

Given the proliferation and success of deep convolutional neural networks, there is growing interest in improving the efficiency of such models both in terms computational and memory requirements.

Multiple approaches have been proposed to compress existing models as well as to directly train efficient neural networks.

Approaches include pruning unnecessary weights in exiting models, sharing of parameters, binarization and more generally quantization of model parameters, transferring the knowledge of high-performance networks into a smaller more more compact network by learning a student network to mimic a teacher network.

The weights of existing networks can be pruned away using the magnitude of weights BID22 , or the Hessian of the loss function BID6 BID14 .

BID1 showed that it is possible to train a shallow but wider student network to mimic a teacher network, performing almost as well as the teacher.

Similarly BID9 proposed Knowledge Distillation to train a student network to mimic a teacher network.

Among recent approaches for training high-performance CNNs, PolyNet shares similar names to our proposed PolyCNN.

PolyNet considers higher-order compositions of learned residual functions while PolyCNN considers higher-order polynomials of the weights and response maps.

Inspired by the polynomial correlation filter, in this paper, we have proposed the PolyCNN as an alternative to the standard convolutional neural networks.

The PolyCNN module enjoys significant savings in the number of parameters to be learned at training, at least 10?? to 50??.

PolyCNN have much lower model complexity compared to traditional CNN with standard convolutional layers.

The proposed PolyCNN demonstrates performance on par with the state-of-the-art architectures on several image recognition datasets.

@highlight

PolyCNN only needs to learn one seed convolutional filter at each layer. This is an efficient variant of traditional CNN, with on-par performance.

@highlight

Attempts at reducing the number of CNN model parameters by using the polynomial transformation of filters to create blow-up the filter responses.

@highlight

The authors propose a weight sharing architecture for reducing the number of convolutional neural network parameters with seed filters