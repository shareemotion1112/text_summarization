To understand the inner work of deep neural networks and provide possible theoretical explanations, we study the deep representations through the untrained, random weight CNN-DCN architecture.

As a convolutional AutoEncoder, CNN indicates the portion of a convolutional neural network from the input to an intermediate convolutional layer, and DCN indicates the corresponding deconvolutional portion.

As compared with DCN training for pre-trained CNN, training the DCN for random-weight CNN converges more quickly and yields higher quality image reconstruction.

Then, what happens for the overall random CNN-DCN?

We gain intriguing results that the image can be reconstructed with good quality.

To gain more insight on the intermediate random representation, we investigate the impact of network width versus depth, number of random channels, and size of random kernels on the reconstruction quality, and provide theoretical justifications on empirical observations.

We further provide a fast style transfer application using the random weight CNN-DCN architecture to show the potential of our observation.

Deep neural networks have achieved impressive performance on various machine learning tasks.

However, our understanding of how these deep learning models operate remains limited.

Providing a theoretical explanation or empirical interpretation for their success is an important research area.

Existing works Arora et al. (2015; 2014) ; Paul & Venkatasubramanian (2014) propose mathematical models for learning architectures, however, the theoretical analysis of which fails to capture the state-of-the-art architectures.

Gilbert et al. (2017) ; Chang et al. (2018) leverage either compressive sensing or ordinary differential equations to facilitate the understanding of CNNs.

Ma et al. (2018) ; Hand & Voroninski (2017) deliver rigorous proofs about the invertibility of convolutional generative models.

Despite these promising progress, there is no solid theoretical foundation on why the overall random CNN-DCN architecture is capable for image reconstruction.

In this paper, we bridge the gap between the empirical observation and theoretical explanation of CNNs, especially the invertibility of the overall random CNN-DCN architecture.

To understand the deep representations of intermediate layers, a variety of visualization techniques have been developed in order to unveil the feature representation and hence the inner mechanism of convolutional neural networks (CNNs) Zeiler & Fergus (2014) ; Mahendran & Vedaldi (2015) ; Yosinski et al. (2015) ; Xu et al. (2015) .

In this work we propose applying randomization on deconvolutional networks (DCNs) for a systematic investigation of deep representations, and provide insights on the intrinsic properties of deep convolutional networks.

We first observe that training the DCN for reconstruction, the random CNN preserves richer information in the feature space.

The training on DCN converges faster for the random CNN contrasted to pre-trained CNN and yields higher quality image reconstruction.

It indicates there is rich information encoded in the random features; the pre-trained CNN discards some information irrelevant for classification and encodes relevant features in a way favorable for classification but harder for reconstruction.

This leads us to be curious about what happens if we feed the images to a CNN-DCN architecture where both the CNN and the DCN have random weights.

Our motivation for studying the overall random CNN-DCN architecture is threefold.

First, a series of works empirically showed that a certain feature learning architecture with random weights allowed satisfactory discriminative validity on object recognition tasks Jarrett et al. (2009) , and certain convolutional pooling architectures even with random weights can be inherently frequency selective and translation invariant, leading to the potential application of fast search of network architectures Saxe et al. (2011) .

Second, studying a complex system with random weights rather than learned determin-istic ones may lead to a better understanding of the system even in the learned case.

For example, in the field of compressed sensing, random sampling leads to breakthroughs in the understanding of the number of required measurements for a stable reconstruction of the signal Giryes et al. (2016) ; Gilbert et al. (2017) .

For highly complicated systems with nonlinear operations along the hidden layers, there are already some investigations on random deep neural networks Saxe et al. (2011); Arora et al. (2014) ; Ulyanov et al. (2017a) .

Third, as a reversible encoder-decoder architecture, deconvolution is a valuable visualization technique for studying the feature representation of deep convolutional nets.

To our knowledge there is no existing work on the random deconvolutional networks in the literature.

Our work on using deconvolution to study the random intermediate features of CNN provides new insights and inspires possible applications with untrained deep neural models.

Our main results and contributions are as follows.

We study the overall random CNN-DCN architecture to investigate the randomness in deconvolutional networks, i.e. there is no training at all for inverting the inputs that passes their information through a random weight convolutional network.

Surprisingly, the image is inverted with satisfactory quality.

The geometric and photometric features of the inputs are well preserved given a sufficient number of channels.

We provide empirical evidence as well as theoretical analysis on the reconstruction quality, and bound the error in terms of the number of random nonlinearities, the network architecture, the distribution of the random weights, and local similarity of the input which is high for natual images.

Extensive empirical study by varying the network width, depth, or kernel size has been performed to show the effectiveness on the inversion.

The CNN-DCN architecture with random weights can be very useful on texture synthesis, style transfer, image segmentation, image inpainting, etc.

As an example, we illustrate how fast style transfer can be applied using random weight CNN-DCN architecture.

Note that our approach can save a big amount of time and energy as we do not need to do the pre-training on deep models, and it is very flexible as we can easily try whatever nerual network architecture as we wish.

Two techniques are closely related to our work, deconvolution and randomization.

Deconvolution involves a CNN-DCN architecture, where CNN indicates the portion of a convolutional neural network from the input to an intermediate convolutional layer, and DCN indicates the corresponding deconvolutional network aiming to invert the intermediate features to the original images.

Randomization indicates the stochastic assignment of weights to the deep neural network.

As a generative model for encoder-decoder functions, deconvolutional networks (DCNs) are commonly used for deep feature visualization.

Zeiler et al. Zeiler & Fergus (2014) propose to use a multi-layered deconvolutional network Zeiler et al. (2011) to project the feature activations back to the input pixel space, and show that the features have many intuitively desirable properties such as compositionality, increasing invariance and class discrimination for deeper layers.

Dosovitskiy et al. Dosovitskiy & Brox (2016) design a deconvolution variant to invert image representations learned from a pre-trained CNN, and conclude that features in higher layers preserve colors and rough contours of the images and discard information irrelevant for the classification task that the convolutional model is trained on.

As there is no back propagation, their reconstruction is much quicker than the representation inverting method on gradient descent Mahendran & Vedaldi (2015) .

Randomization on neural networks can be tracked back to the 1960's where the bottom-most layer of shallow networks consisted of random binary connections Block (1962) .

In recent years, largely motivated by the fact that "randomization is computationally cheaper than optimization", randomization has been resurfacing repeatedly in the machine learning literature Scardapane & Wang (2017) .

For optimization problems such as regression or classification, this technique is used to stochastically assign a subset of weights in a feedforward network to derive a simpler optimization problem Igelnik & Pao (1995) ; Rahimi & Recht (2009) .

Specifically, they compute a weighted sum of the inputs after passing them through a bank of arbitrary randomized nonlinearities, such that the resulting optimization task is formulated as a linear least-squares problem.

Empirical comparisons as well as theoretical guarantees are provided for the approximation Rahimi & Recht (2008; 2009); Arora et al. (2014) .

Other related works include random kernel approximation Rahimi & Recht (2007) ; Sinha & Duchi (2016) and reservoir computing on random recurrent networks Lukosevicius & Jaeger (2009); Jaeger & Haas (2004) .

Specifically on convolutional neural networks (CNNs), there are a few works considering randomization.

Jarrett et al. Jarrett et al. (2009) observe that, on a one-layer convolutional pooling architecture, random weights perform only slightly worse than pre-trained weights.

Saxe et al. Saxe et al. (2011) prove that certain convolutional pooling architectures with random weights are inherently frequency selective and translation invariant, and argue that these properties underlie their performance.

He et al. He et al. (2016) accomplish three popular visualization tasks, image inversion, texture synthesize and style transfer, using random weight CNNs.

Daniely et al. Daniely et al. (2016) extend the scope from fully-connected and convolutional networks and prove that random networks induce representations which approximate the kernel space.

Gilbert et al. Gilbert et al. (2017) combine compressive sensing with random-weight CNNs to investigate the CNN architectures.

Dmitry et al. Ulyanov et al. (2017a) utilize randomly-initialized neural nets to finish denoising and inpainting tasks.

Motivated by the intuition that "random net is theoretically easier to comprehend than the complicated well-trained net", and that it may reveal the intrinsic property of the network architecture, we use randomization to explore the convolution followed by deconvolution architecture, provide theoretical analysis on empirical observations, and show its application potentials by a style transfer case study.

For the network architecture, we focus on VGG16 Simonyan & Zisserman (2015) for the deconvolution.

A convolutional layer is usually followed by a pooling layer, except for the last convolutional layer.

For consistency, we will explore the "feature representation" after the convolutional layer but before the pooling layer.

We build a CNN-DCN architecture on the layer of the feature representation to be studied.

The convolution operator of a deconvolutional layer in DCN is the same as the convolution operator in CNN, and an upsampling operator Dosovitskiy & Brox (2016) is applied in DCN to invert the corresponding pooling operator in CNN.

We will focus on the representations of the convolutional layers, and

We first explore the reconstruction ability of random CNNs.

We assign Gaussian random weights to the CNN part, and train the corresponding DCN to minimize the summation of the pixel-wise loss on the reconstructed images.

Training.

For each intermediate layer, using the feature vectors of all training images, we train the corresponding DCN such that the summation of L 2 -norm loss between the inputs and the outputs is minimized.

Let Φ(x i , w) represent the output image of the DCN, in which x i is the i th input image and w the weights of the DCN.

We train the DCN to get the desired weights w * that minimize the loss.

Then for a feature vector of a certain layer, the corresponding DCN can predict an estimation of the expected pre-image, the average of all natural images which would have produced the current feature vector.

training.

The weight decay is set to 0.0004 to avoid overfitting.

The maximum number of iterations is set at 200,000 empirically.

We also consider another network architecture, AlexNet Krizhevsky et al. (2012) .

For the random weights, we try several Gaussian distributions with zero mean and various variance.

We also try several other types of random distributions, Uniform, Logistic, Laplace, to have a sound exploration.

See more details and comparisons in Appendix 2.

In the following, we use CDk to represent a Conv[k]-DeConv[k] architecture.

Take the VGG CD2 for elaboration, the loss curves during the training process are shown in Figure 10 , which compares VGG and AlexNet on random as well as pre-trained weights.

Here Conv2_Pretrained or Conv2_Random indicates whether the CNN is pre-trained or with random weights.

We see that the training of DCN for reconstruction converges much quicker on random CNN and yields slightly lower loss.

It indicates that by pre-training for classification, CNN encodes relevant features of the input image in a way favorable for classification but harder for reconstruction.

And VGG has a much lower reconstruction loss than AlexNet.

Reconstruction.

We take 5,000 samples from the training set and validation set respectively from ImageNet, and compare their average reconstruction loss.

The statistics are shown in Figure 11 ("Pre-trained net" represents pre-trained CNN while "random net" represents random CNN when we train the corresponding DCN for reconstruction).

We see the pre-trained CNN and random CNN both have good generalization ability; a random VGG yields much less loss than the pre-trained VGG for the deconvolution reconstruction; for representations of deeper layers, the inverting loss increases significantly for pre-trained VGG but grows slowly for random VGG.

The results indicate that the random CNN encodes much richer information of the original images; the pre-trained CNN discards information not crucial for classification, especially on deeper layers, leading to a better classifier but a harder reconstruction task.

Figure 4 shows reconstructions of various layers of the random VGG on images outside the training set.

The reconstruction quality decays for intermediate representations of deeper layers.

The VGG structure with random weights yields accurate reconstruction, even on CD5, which involves 26 convolution layers and 4 pooling layers.

In Appendix 2, we also see that the reconstruction quality on VGG based deconvolution is better than that on AlexNet based deconvolution.

The above results inspire us to further explore what happens if both the CNN and DCN are of random weights.

In this section we consider the reconstructions on purely random VGG CNN-DCN architecture (denoted by rrVGG for brevity), and find that the images can still be reconstructed with satisfactory quality!

In other words, the CNN randomly extracts the image features and passes them to the DCN, then in an unsupervised manner the DCN reconstructs the input image by random feature extraction!

Such intriguing results show that the overall random CNN-DCN architecture substantially contributes to the geometric and photometric invariance for the image inversion.

In the following, we will systematically explore the reconstruction ability of the rrVGG architecture with ReLU nonlinearity.

We found that the network depth has a bigger impact than the network width, and the reconstruction quality decays with deeper layers; with plenty number of channels, an increasing number of random channels promotes the reconstruction quality; and the reconstruction quality decays with a larger kernel size.

For evaluation, we use the structural similarity (SSIM) index Wang et al. (2004) , which is accurate by considering the correlation and dependency of local spatially close pixels, and consistent to the perceptron of human eyes.

To remove the discrepancy on colors, we transform the inputs and outputs in grey-scale, and in case of negative SSIM value, we invert the luminosity of the grayscale image for calculation, so the value is in [0, 1].

A higher value indicates a higher similarity on the images.

We first explore the impact of network depth and network width for the random reconstruction, using a cat image outside the training data as an example.

The weights are random in N (0, 0.1) 1 .

We first study the reconstruction quality for different convolutional layers, as in Figure 5 .

Though there is no training at all, DCN can still perceive geometric positions and contours for CD1 to CD3.

The deeper the random representations are, the coarser the reconstructed image is.

We can still perceive a very rough contour for the random CD4 architecture, which is already 10 layers deep.

Our follow-up theoretical analysis will show that depth does affect the results, as it affects the size of receptive fields.

In Figure 6 , we build a Conv1-DeConv1 (CD1) architecture with different dimensions (width) using the actual width of VGG Conv1 to Conv5 for CD1 respectively.

We see that the smaller the dimension (width) is, the coarser the image is.

We investigate the reconstruction quality on the number of random channels using the rrVGG CD1 (Conv1-DeConv1) architecture.

For simplicity, for each network instance we use the same number of channels in all layers except the output layer.

We vary the number of random channels from 4, 8 up to 2048, and for each number of channels, we generate 30 rrVGG Conv1-DeConv1 networks and all random weights are in N (0, 0.1) distribution.

For input images we randomly pick 50 samples from the ImageNet validation set.

To reduce occasionality on the reconstruction, we transform the inputs and outputs in grey-scale and calculate the average SSIM value on each network, then we do statistics (mean and standard deviation) on the 30 average values.

Figure 7 shows the trends on SSIM when the number of channels increases, (a) is for the original rrVGG network and (b) is for a variant of rrVGG network.

The variant of rrVGG is almost the same as the original network except that the last convolutional layer is replaced by an average layer, which calculates the average over all the channels of the feature maps next to the last layer.

We see that the increasing number of random channels promotes the reconstruction quality.

Similar in spirit to the random forest method, different channels randomly and independently extract some feature from the previous layer.

With sufficient number of random channels we may encode and transform all information to the next layer.

In the next section, we will prove for the variant convolutional network, when the width of the random neural network goes to infinity, the output will converge to a fixed image close to the original image.

In Figure 8 , we pick some input images, and show the corresponding output images closest to the mean SSIM value for various number of channels.

We transform the randomly-generated colored image to grey-scale image, for the ease of comparing the structural similarity.

The SSIM value is on the top of each output image.

The increasing number of channels promotes the random reconstruction quality.

To show how the reconstruction quality decays with deeper convolutional layers, we also do experiments on the rrVGG CD2 architecture, and the quality decays by about a half as evaluated by SSIM.

We expect that the reconstruction quality decays with larger kernel size, as a large kernel size can not consider the local visual feature of the input.

In the extreme case when the kernel size equals the image dimension, the convolution operator actually combines all pixel values of the input to an output pixel using random weights.

We use the rrVGG Conv1_1 DeConv1_1 architecture, which simply contains two convolutional operators.

The random weights are in N (0, 0.1) distribution.

For each kernel size, we randomly generate 30 networks for the reconstruction on 50 sample images as selected above.

The results verifies our assumption, as in Figure 7 (c).

To show how our observation can be used for application, we provide a potential application using random CNN-DCN -Style Transfer with rrVGG.

By choosing the suitable number of filters, the rrVGG CD1 architecture can achieve high-quality reconstruction.

Besides, these reconstructions can also bring slight differences such as the background color and texture, which is suited to exploring more interesting style transfer results.

And multiple rrV GG models can be efficiently acquired without training.

Recent work Gatys et al.

In this section, we provide theoretical analysis to explain the empirical results.

We will show that a slight variant of the random CNN architecture has the ability to reconstruct the input image.

We also investigate how depth and width of the network will affect the reconstruction ability.

Intuitively, content style rrV GG 1 rrV GG 2 rrV GG 3

Figure 9: Style transfer from several rrVGG models.

Each model has the same architecture but different random weights.

as the depth of the network increases, the receptive field of each output image pixel becomes larger, which makes the reconstruction harder whereas the width of the network, or equivalently, the number of channels, gives more basis or ways to reconstruct the original input.

We theoretically show that the reconstruction ability of a random convolutional neural network rises when the number of channels in each layer increases and drops when the depth of the network increases.

Note that DCN is also a kind of CNN with up-sampling layers, so our result can be directly applied to the CNN-DCN architecture.

For the following part, we will first show the convergence of the output image when the width of the network goes to infinity.

Many researchers have worked on the infinite width fully connected networks.

For instance, Williams (1996); Lee et al. (2017) focus on its relationship with Gaussian Process.

They show the exact equivalence between infinitely wide deep networks and Gaussian Processes.

Our work focuses on the random convolutional neural network without any training, which is different from the above-mentioned works and is in accordance with our previous empirical analysis.

Then, we show the difference between the real output and the convergence value as a function of the width.

Finally, we give an upper bound on the angle between the input and the convergence value.

Thus, we can bound the reconstruction error.

Notations: We use A :,j to denote the j th column vector of matrix A and x to denote the l 2 -norm of vector x. Let L be the number of layers in the neural network and X (i)

∈ R Ni×di be the feature maps in the i th layer, where N i is the number of channels and d i is the dimension of a single channel feature map (i.e. the width of the map times its height).

is the output image.

w (i,j) , a row vector, is the j th convolutional filter of the i th layer if it is a convolutional layer.

We use ReLU (x) = max(x, 0) as the activation function in the following analysis.

Definition 1.

Random CNN architecture To make it possible for property proof, this structure is different from the classic CNN structure in the following three points: 1) Different filters in the same layer are i.i.d.

random vectors and filters in different layers are independent.

The probability density function of each filter is isotropic.

Let k

2) The last layer is the arithmetic mean of the channels of the previous layer, not the weighted combination.

3) Except for X (L−1) , each layer of convolutional feature maps are normalized by a factor of

, where N i is the number of channels of this layer.

From the previous experiments, we see that when the number of channels increases, the quality of the output image improves.

Here we prove that when the number of channels goes to infinity, the output will actually converge.

Each pixel in the final convergence value is a constant times the weighted norm of its receptive field.

Formally, we state our main results for the convergence value of the random CNN as follows.

Theorem 1. (Convergence Value) Suppose all the pooling layers use l 2 -norm pooling.

When the number of filters in each layer of a random CNN goes to infinity, the output f corresponding to a fixed input will converge to a fixed image f * with probability 1, where f * = kz * and k is a constant only related to the CNN architecture and the distribution of random filters and z * i = l∈Ri n (l,i) X :,l 2 , where R i is the index set of the receptive field of z * i and n (l,i) is the number of routes from the l th pixel of a single channel of the input image to the i th output pixel.

The proof of Theorem 1 is in Appendix 3, Theorem 4.

Here for the pooling layer, instead of average pooling, which calculates the arithmetic mean, we use l 2 -norm pooling which calculates the norm of the values in a patch.

Intuitively, if most pixels of the input image are similar to their adjacent pixels, the above two pooling methods should have similar outputs.

See details for average pooling in Appendix 3.

Now we consider the case of a finite number of channels in each layer.

We mainly focus on the difference between the real output and the convergence value as a function of the number of channels.

We prove that for our random CNN architecture, as the number of channels increases, with high probability, the angle between the real output and the convergence value becomes smaller, which is in accordance with the variant rrVGG experiment results shown in previous section.

Theorem 2. (Multilayer Variance) Suppose all the pooling layers use l 2 -norm pooling.

For a random CNN with L layers and N i filters in the i th layer, let Θ denote the angle between the output f and the convergence value f * , suppose that there is at most one route from an arbitrary input pixel to an arbitrary output pixel for simplicity, then with probability 1 − δ,

where

Here, (i) (x) actually measures the local similarity of x. The full definition of (i) (x) and the proof of this theorem is in Appendix 3.

Finally, we focus on how well our random CNN architecture can reconstruct the original input image.

From Theorem 2, we know that with high probability, the angle between the output of a random CNN with finite channels and the convergence value will be upper-bounded.

Therefore, to evaluate the performance of reconstruction, we focus on the difference between the convergence value and the input image.

We will show that if the input is an image whose pixels are similar to their adjacent pixels, then the angle between the input image X and the convergence value of the output image will be small.

To show the essence more clearly, we state our result for a two-layer random CNN and provide the multi-layer one in Appendix 3, which needs more complicated techniques but has the same insight as the two-layer one.

Theorem 3.

For a two-layer random CNN, suppose each layer has a zero-padding scheme to keep the output dimension equal to the dimension of the original input.

The kernel size is r and stride is 1.

The input image is X ∈ R d0 , which has only one channel, whose entries are all positive.

t = X t − X t means the difference between one pixel X t and the mean of the r-sized image patch whose center is X t .

Let Φ be the angle between the input image X and the convergence value of the output image, we have cos

The full proof of Theorem 3 is in Appendix 3.

Note that when the kernel size r increases, t will become larger as an image only has local similarity, so that the lower bound of the cosine value becomes worse, which explains the empirical results in previous section.

In this work, we introduce a novel investigation on deep random representations through the convolution-deconvolution architecture, which to our knowledge is the first study on the randomness of deconvolutional networks in the literature.

We extensively explore the potential of randomness for image reconstruction on deep neural networks, and found that images can be reconstructed with satisfactory quality when there are a sufficient number of channels.

Extensive investigations have been performed to show the effectiveness of the reconstruction.

We also provide theoretical analysis that a slight variant of the random CNN architecture has the ability to reconstruct the input image, and the output converges to the input image when the width of the network, i.e. number of channels, goes to infinity.

We also bound the reconstruction error between the input and the convergence value as a function of the network width and depth. (2015) and AlexNet Krizhevsky et al. (2012) .

A convolutional layer is usually followed by a pooling layer, except for the last convolutional layer, Conv5.

For consistency, we will explore the output after the convolutional layer but before the pooling layer.

In what follows, "feature representation" or "image representation" denotes the feature vectors after the linear convolutional operator and the nonlinear activation operator but before the pooling operator for dimension reduction.

We build a CNN-DCN architecture on the layer of feature representation to be studied.

The convolution operator of a deconvolutional layer in DCN is the same as the convolution operator in CNN, and an upsampling operator is applied in DCN to invert the corresponding pooling operator in CNN, as designed in Dosovitskiy & Brox (2016).

We will focus on the representations of the convolutional layers, since Dosovitskiy et al. Dosovitskiy & Brox (2016) build DCNs for each layer of the pre-trained AlexNet and find that the predicted image from the fully connected layers becomes very vague.

For the activation operator, we apply the leaky ReLU nonlinearity with slope 0.2, that is, r(x) = x if x ≥ 0 and otherwise r(x) = 0.2x.

At the end of the DCN, a final Crop layer is added to cut the output of DeConv1 to the same shape as the original images.

We build deconvolutional networks on both VGG16 and AlexNet, and most importantly, we focus on the random features of the CNN structure when training the corresponding DCN.

Then we do no training for deconvolution and explore the properties of the purely random CNN-DCN architecture on VGG16.

For the random weights assigned to CNN or DCN, we try several Gaussian distributions with zero mean and various variance to see if they have different impact on the DCN reconstruction.

Subsequent comparison shows that a small variance around 0.015 yields minimal inverting loss.

We also try several other types of random distributions, Uniform, Logistic, Laplace, to study their impact.

• The Uniform distribution is in [-0.04, 0.04), such that the interval equals [µ − 3δ, µ + 3δ] where µ = 0 and δ = 0.015 are parameters for Gaussian distribution.

• The Logistic distribution is 0-mean and 0.015-scale of decay.

It resembles the normal distribution in shape but has heavier tails.

• The Laplace distribution is with 0 mean and 2 * λ 2 variance (λ = 0.015), which puts more probability density at 0 to encourage sparsity.

For each intermediate layer, using the feature vectors of all training images, we train the corresponding DCN such that the summation of L 2 -norm loss between the inputs and the outputs is minimized.

Let Φ(x i , w) represent the output image of the DCN, in which x i is the input of the i th image and w is the weights of the DCN.

We train the DCN to get the desired weights w * that minimize the loss.

Then for a feature vector of a certain layer, the corresponding DCN can predict an estimation of the expected pre-image, the average of all natural images which would have produced the given feature vector.

training.

The weight decay is set to 0.0004 to avoid overfitting.

The maximum number of iterations is set at 200,000 empirically.

Training.

We observe similar results for the training loss in different layers.

Take the Conv2-DeConv2 architecture for elaboration, the loss curves during the training process are shown in Figure 10 .

Figure 10 (a) compares VGG and AlexNet on random as well as pre-trained weights.

The training for reconstruction converges much quicker on random CNN and yields slightly lower loss, and this trend is more apparent on VGG.

It indicates that by pre-training for classification, CNN encodes relevant features of the input image in a way favorable for classification but harder for reconstruction.

Also, VGG yields much lower inverting loss as compared with AlexNet.

Figure 10 (b) shows that random filters of different small-variance Gaussian distributions on CNN affect the initial training loss, but the loss eventually converges to the same magnitude.

(The loss curve of N (0, 1) is not included as the loss is much larger even after the converge.)

Figure 10 (c) shows that the four different random distributions with appropriate parameters acquire similar reconstruction loss.

Generalization.

We take 5000 samples from the training set and validation set respectively from ImageNet, and compare their average reconstruction loss.

The statistics is as shown in Figure  11 , where CDk represents a Conv[k]-DeConv[k] architecture.

Figure 11 (a) shows that the VGG architecture is good in generalization for the reconstruction, and random VGG yields much less loss than pre-trained VGG.

For representations of deeper layers, the inverting loss increases significantly for pre-trained VGG but grows slowly for random VGG.

This means that in deeper layers, the pre-trained VGG discards much more information that is not crucial for classification, leading to a better classifier but a harder reconstruction task.

Figure 11 (b) compares VGG and AlexNet on the CD3 architecture.

It shows that the reconstruction quality on random compares favourably against that on pre-trained in VGG.

Reconstruction.

Figure 12 shows reconstructions from various layers of random VGG and random AlexNet, denoted by rwVGG and rwAlexNet respectively.

2 On both rwVGG and rwAlexNet, the reconstruction quality decays for representations of deeper layers.

The rwVGG structure yields more accurate reconstruction, even on Conv5, which involves 26 convolution operations and 4 max pooling operations.

Figure 13 shows reconstructions from a cat example image for various distributions of rwVGG CD2.

Except for N (0, 1), the reconstruction quality is indistinguishable by naked eyes.

It shows that different random distributions work well when we set the random weights relatively sparse.

In a nutshell, it is interesting that random CNN can speed up the training process of the DCN on both VGG and AlexNet, obtain higher reconstruction quality and generalize well for other inputs.

Regarding weights in the convolutional part as a feature encoding of the original image, then the deconvolutional part can decode from the feature representations encoded by various methods.

The fact that the random encoding of CNN is easier to be decoded indicates that the training for classification moves the image features of different categories into different manifolds that are moving further apart.

Also, it may discard information irrelevant for the classification.

The pre-trained CNN benefits the classification but is adverse to the reconstruction.

For completeness, we repeat the notations and the definition of random CNN architecture.

Notations: We use A :,j to denote the j th column vector of matrix A and use A ij to denote its entry.

Let x i be the i th entry of vector x. Let L be the number of layers in the neural network and X (i)

∈ R Ni×di be the feature maps in the i th layer, where N i is the number of channels and d i is the dimension of a single channel feature map (i.e. the width of the map times its height).

X = X (0) is the input image and f = X (L) is the output image.

For convenience, we also define convolutional feature maps to be the feature maps after convolutional operation and define pooled feature maps and up-sampled feature maps in the same way.

In the i th layer, let r i be the fixed kernel size or the pool size (e.g. 3 × 3).

If X (i+1) is convolutional feature maps, let Y (i) ∈ R Niri×di be the patched feature for pooling and up-sampling layers.

For the j th pixel of the output image in the last layer, define its receptive filed on the input image in the first layer as X :,Rj = {X :,m | m ∈ R j }, where R j is a set of indexes.

The activation function ReLU (x) = max(x, 0) is the element-wise maximum operation between x and 0 and (·) m is the element-wise power operation.

Definition 2.

Random CNN architecture This structure is different from the classic CNN in the following three points:

• Different filters in the same layer are i.i.d.

random vectors and filters in different layers are independent.

The probability density function of each filter is isotropic.

Let k

4 all exist.

•

The last layer is the arithmetic mean of the channels of the previous layer, not the weighted combination.

• Except for X (L−1) , each layer of convolutional feature maps are normalized by a factor of

, where N i is the number of channels of this layer.

A.3.1 CONVERGENCE Theorem 4. (Convergence Value) Suppose all the pooling layers use l 2 -norm pooling.

When the number of filters in each layer of a random CNN goes to infinity, the output f corresponding to a fixed input will converge to a fixed image f * with probability 1, where f * = kz * and k is a constant only related to the CNN architecture and the distribution of random filters and z * i = l∈Ri n (l,i) X :,l 2 , where n (l,i) is the number of routes from the l th input pixel to the i th output pixel.

Here for the pooling layer, instead of average pooling, which calculates the arithmatic mean, we use l 2 -norm pooling which calculates the norm of the values in a patch.

We also show the result for average pooling in Theorem A.7.

To prove the theorem, we first prove the following lemma.

Lemma 5.

Suppose w ∈ R n , n ≥ 2 is a random row vector and its probability density function is isotropic.

Y ∈ R n×d is a constant matrix whose i th column vector is denoted by y i .

z ∈ R d is a row vector and

where θ ij is the angle between y i and y j .

Proof.

Note that max{·, ·} and (·) m are both element wise operations.

The i th element of Eg m is (Eg m ) i = Emax{wy i , 0} m .

Since the probability density function of w is isotropic, we can rotate y i to y i without affecting the value of Emax{wy i , 0} m .

Let

Where the third equality uses the fact that the marginal distribution of w 1 is also isotropic.

Similarly, we have:

Eg i g j = Emax{wy i , 0}max{wy j , 0}. We can also rotate y i and y j to y i and y j .

Let y i = ( y i , 0, 0, ..., 0)

T and y j = ( y j cos θ ij , y j sin θ ij , 0, ..., 0)

T and suppose the marginal probability density function of (w 1 , w 2 ) is p(ρ) which does not depend on φ since it is isotropic, where ρ = w 2 1 + w 2 2 is the radial coordinate and φ is the angular coordinate.

We have:

Note that:

We obtain the second part of this lemma.

Now, we come to proof of Theorem A.1.

Proof.

According to Lemma A.2, if X (i+1) is convolutional feature maps, we can directly obtain:

where we have fixed Y (i) and the expectation is taken over random filters in the i th layer only.

Since different channels in X (i+1) are i.i.d.

random variables, according to the strong law of large numbers, we have:

which implies that with probability 1,

Suppose that all N j for 1 ≤ j ≤ i have gone to infinity and z (i) has converged to z * (i) , the above expression is the recurrence relation between z * (i+1) and z * (i) in a convolutional layer:

If X (i+1) is up-sampled feature maps, a pixel X (i)

jp will be up-sampled to a r-sized block {X

jp and all the other elements are zeros.

Definẽ D

So far, we have obtained the recurrence relation in each layer.

In order to get z * (i+1) given z * (i) , we use the same sliding window scheme on z * (i) as that of the convolutional, pooling or upsampling operation on the feature maps.

The only difference is that in a convolutional layer, instead of calculating the inner product of a filter and the vector in a sliding window, we simply calculate the l 2 -norm of the vector in the sliding window and then multiply it by k (i) 2 .

Note that z * (0) can be directly obtained from the input image.

Repeat this process layer by layer and we can obtain z * (L−2) .

According to Lemma A.2, we have:

Suppose that z (L−2) has converged to z * (L−2) , and by Definition A.

, we have f * = kz * .

Note that z * is obtained through a multi-layer sliding window scheme similar to the CNN structure.

It only depends on the input image and the scheme.

It is easy to verify that z * i is the square root of the weighted sum of the square of input pixel values within the receptive field of the i th output pixel, where the weight of an input image pixel is the number of routes from it to the output pixel.

Theorem 6. (Variance) For a two-layer random CNN with N filters in the first convolutional layer, let Θ denote the angle between the output f and the convergence value f * , then with probability 1 − δ,

Proof.

According to Theorem A.1, we have

.

For a two-layer CNN, we can directly obtain:

Since different channels are i.i.d.

random variables, we have EX

(1)

According to Markov inequality, we have:

1 ) 2 ) z * 2 , then with probability 1 − δ:

To extend the above two-layer result to a multi-layer one, we first prove the following lemma.

Note that in this lemma, D (i) should be replaced byD (i) defined in the proof of Theorem A.1 if X (i) is up-sampled feature maps.

is a linear mapping.

For simplicity, suppose that for

2 for convolutional layers and k (i) = 1 for pooling and up-sampling layers and

Proof.

According to the definition of φ (i) (·) and Theorem A.1, we have:

It is easy to verify that for any c ∈ R and x, y ∈ R di+1 we have

x j , which is the average value of the m th patch.

.

We have:

m , which implies that

Theorem 8. (Multilayer Variance) Suppose all the pooling layers use l 2 -norm pooling.

For a random CNN with L layers and N i filters in the i th layer, let Θ denote the angle between the output f and the convergence value f * , suppose that there is at most one route from an arbitrary input pixel to an arbitrary output pixel for simplicity, then with probability 1 − δ,

where

Proof.

We will bound Θ recursively.

Suppose that the angle between (z (i) ) 2 and (z * (i) ) 2 is θ i .

We

tional feature maps, we have obtained in the proof of Theorem A.1 that

Using similar method to the proof of Theorem A.3 and let α i+1 denote the angle between g (i+1) and Eg (i+1) , we can derive that with probability 1 − δ i+1 ,

For a l 2 -norm pooling layer or an up-sampling layer, we have:

, we have

Let v denote the angle between z (L−2) and f , we have obtained its bound in Theorem A.3.

With

.

With all the bounds above, define N by

for simplicity, we can obtain the bound of Θ: with probability 1 − δ,

Here, R t is the index set of the receptive field of f t and n (α,t) is the number of routes from X α to f t .

Suppose that the receptive field of each f t has the same size and shape, X t is at a fixed relative position of the receptive field of f t and n (α,t) only depends on the relative position between X α and X t .

Let X t = α∈R t n (α,t) Xα α∈R t n (α,t) be the weighted average and t = X t − X t .

By using the same technique above, we can obtain that

Note that although the bound is the same as the two-layer convolutional neural network, as the receptive field is enlarged, t can be much larger, so that the above bound will be worse.

We also give the convergence value for average pooling in the next theorem.

Theorem 10. (Convergence Value, average pooling) Suppose all the pooling layers use average pooling.

When the number of filters in each layer of a random CNN goes to infinity, the output f corresponding to a fixed input will converge to a fixed image f * with probability 1.

is convolutional feature maps, according to Lemma A.2, we have:

where ϕ

Note that:

Suppose that all N j for 1 ≤ j ≤ i have gone to infinity and C (i) has converged to C * (i) , the above expressions are the recurrence relation between C * (i+1) and C

is average-pooled feature maps, we have:

We have:

which is the recurrence relation for an average pooling layer.

For an up-sampling layer, a pixel X (i) jk will be up-sampled to a block {X

jk and all the other elements are zeros.

We have:

otherwise.

Note that we can directly calculate C * (0) according to the input image.

So we can recursively obtain C * (L−2) and thus z * (L−2) .

According to Lemma A.2, we have:

Suppose that z (L−2) has converged to z * (L−2) , and by Definition A.

, we have: f a.s.

We can obtain the convergence value f * through the above process.

We observe that by choosing a suitable number of random filters, the rrVGG Conv1-DeConv1 architecture can achieve high-quality reconstruction.

The reconstructions also bring slight differences in the background color and texture, which is suited for exploring more interesting style transfer results.

Hence we utilize the framework, as shown in Fig (2016), we also adopted the linear combination of both content loss and style loss,

.

L c indicates the content loss which is the euclidean distance between the content vector V c and stylized feature vector V new , where A i is the activation vector in i-th layer of the FVT.

The style loss L s is obtained from the Gram matrix of the feature vectors.

In Eq. equation 4, G represents the gram matrix.

Inspired by Li et al. (2017 )Huang & Belongie (2017 , we also utilize the the mean value and standard deviation of feature vectors to calculate the style loss, the result of which is similar to the gram loss L s .

In Fig. 14, FVT (iterative optimization) only contains the convolutional layer from Conv2 to Conv5 and rrVGG contains Conv1 and DeConv1.

In addition, we can also utilize more layers on rrVGG and FVT will contain less layers on the optimization network correspondingly, which will further speed up the style transfer process.

In experiments, our framework is faster than the original optimization based approach and can transfer the arbitrary styles.

As for the stylization effectiveness, we compared our results with Gatys et al. Gatys et al. (2016) and Ulyanov et al. Ulyanov et al. (2017b) .

In Fig. 15 , rrV GG 1 and rrV GG 2 columns denote the stylization results acquired from our framework, applying two different rrVGG models.

As shown in Fig. 15 , our stylization results are competitive to other well-trained approaches.

Focused on rrV GG 1 column, our stylized result is inclined to extract more features from the style image and slightly weaken the representation of content image.

Since we utilize rrVGG CNN and DCN to complete the transformation between feature space and image space, some content information is possible to be lost during the reconstruction process.

Despite that, our approach is still capable of generating high quality stylized images.

In addition, we also investigate the stylized effectiveness when modifying the balance between style and content in FVT.

As shown in Figure 16 , the number below each column indicates the relative weightings between style and content reconstruction.

In our framework, the transition from content to style is smooth with increasing ratio.

As shown in Figure 17 , our stylized result is inclined to extract more features from the style image and slightly weaken the representation of the content image.

content style rrVGG 1 rrVGG 2 rrVGG 3

Figure 17: Style transfer from several rrVGG models.

Each model has the same architecture but different random weights.

As proposed in our paper, in terms of different distributions and number of filters, rrVGG can reconstruct images with diverse textures and background colors.

In Fig. 14, replacing CNN and DCN parts with different rrVGG models, our framework can generate abundant stylized images depending on a single style image.

Since rrVGG models are generated without training, it won't incur additional computational cost.

As shown in Fig. 17 , the rightmost three columns comprise the stylized images with different rrVGG model weights while the leftmost two columns represent input content and style images respectively.

For each row, given content and style images, we choose three stylized images generated by our framework using different rrVGG models.

For instance, in the 3-rd row of Fig. 17 , the parameters of chosen rrVGG models are as following: rrVGG 1 :(N (0, 0.01), filter size:3, filter num:128), rrVGG 2 :(N (0, 0.01), filter size:5, filter num:256) and rrVGG 3 :(N (0, 0.1), filter size:3, filter num:32).

As shown in Fig. 17 , those stylized images not only well preserve the style structure such as the shape of the curved lines, waves and abstract objects, but also exhibit novel combinations of the structure, shade and hue.

Coupled with various rrVGG models, the proposed style transfer framework is able to unleash the diversity and variation inside a single style image, which works well in practice.

Meanwhile, it's flexible as well as fast, since the FVT part can be implemented either by an optimization process or some feed-forward convolutional layers.

@highlight

We investigate the deep representation of untrained, random weight CNN-DCN architectures, and show their image reconstruction quality and possible applications.