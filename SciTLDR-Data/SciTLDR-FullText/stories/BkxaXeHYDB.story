A plethora of computer vision tasks, such as optical flow and image alignment, can be formulated as non-linear optimization problems.

Before the resurgence of deep learning, the dominant family for solving such optimization problems was numerical optimization, e.g, Gauss-Newton (GN).

More recently, several attempts were made to formulate learnable GN steps as cascade regression architectures.

In this paper, we investigate recent machine learning architectures, such as deep neural networks with residual connections, under the above perspective.

To this end, we first demonstrate how residual blocks (when considered as discretization of ODEs) can be viewed as GN steps.

Then, we go a step further and propose a new residual block, that is reminiscent of Newton's method in numerical optimization and exhibits faster convergence.

We thoroughly evaluate the proposed Newton-ResNet by conducting experiments on image and speech classification and image generation, using 4 datasets.

All the experiments demonstrate that Newton-ResNet requires less parameters to achieve the same performance with the original ResNet.

A wealth of computer vision problems (e.g., structure from motion (Buchanan & Fitzgibbon, 2005) , stereo (Lucas et al., 1981; Clark et al., 2018) , image alignment (Antonakos et al., 2015) , optical flow (Zikic et al., 2010; Baker & Matthews, 2004; Rosman et al., 2011) ) are posed as nonlinear optimization problems.

Before the resurgence of the machine learning era, the dominant family for solving such optimization problems 1 was numerical optimization, e.g, Gauss-Newton (GN).

Recently, it was proposed that the GN steps, called descent directions, can be learned and represented as a cascade regression to solve non-linear least square problems (Xiong & De la Torre, 2013) .

With the advent of deep learning, the aforementioned ideas were combined with learnable feature representations using deep convolutional neural networks for solving problems such as alignment and stereo (Trigeorgis et al., 2016; Clark et al., 2018) .

In this paper, we first try to draw similarities between learning descent directions and the structure of the popular residual networks.

Motivated by that, we further extend residual learning by adopting ideas from Newton's numerical optimization method, which exhibits faster convergence rate than Gauss-Newton based methods (both theoretically and empirically as we show in our experiments).

ResNet (He et al., 2016) is among the most popular architectures for approximating non-linear functions through learning.

The core component of ResNet is the residual block which can be seen as a linear difference equation.

That is, the t th residual block is expressed as x t`1 " x t`C x t for input x t .

By considering the residual block as a discretization of Euler ODEs (Haber et al., 2018; Chen et al., 2018) , each residual block expresses a learnable, first order descent direction.

We propose to accelerate the convergence (i.e., employ fewer residual blocks) in approximation of non-linear functions by introducing a novel residual block that exploits second-order information in analogy to Newton's method in non-linear optimization.

Since the (second order) derivative is not analytically accessible, we rely on the idea of Xiong & De la Torre (2014) to learn the descent directions by exploiting second order information of the input.

We build a deep model, called Newton-ResNet, that involves the proposed residual block.

Newton-ResNet requires less residual blocks to achieve the same accuracy compared to original ResNet.

This is depicted in Fig. 1 ; the contour 2 shows the loss landscape near the minimum of each method and indeed the proposed method requires fewer steps.

Our contributions are as follows:

• We first establish a conceptual link between residual blocks in deep networks and standard optimization techniques, such as Gauss-Newton.

This motivates us to design a novel residual block that learns the descent directions with second order information (akin to Newton steps in nonlinear optimization).

A deep network composed of the proposed residual blocks is coined as Newton-ResNet.

• We show that Newton-ResNet can effectively approximate non-linear functions, by demonstrating that it requires less residual blocks and hence significantly less parameters to achieve the performance of the original ResNet.

We experimentally verify our claim on four different datasets of images and speech in classification tasks.

Additionally, we conduct experiments on image generation with Newton-ResNet-based GAN (Goodfellow et al., 2014) .

• We empirically demonstrate that Newton-ResNet is a good function approximator even in the absence of activation functions, where the corresponding ResNet performs poorly.

The literature on resnets is vast; we focus below on the perspectives of a) theoretical understanding, b) alternative architectures and c) modifications of the transformation path.

A significant line of research is the theoretical understanding behind the performance of residual connections.

The work of Hardt & Ma (2017) proves that arbitrarily deep linear residual networks have no spurious local optima; all critical points correspond to a global minimum.

Zaeemzadeh et al. (2018) attribute the success of resnet to the norm presentation.

More recently, Shamir (2018) proves that a network with residual connections is provably better than the corresponding network without the residuals.

Balduzzi et al. (2017) focus on the gradients in residual connections; they study the correlations during initialization and introduce an appropriate initialization.

These works are orthogonal to ours; they methodologically study the theoretical properties of deep learning, while we focus on reducing the number of residual blocks required.

same topology with different feature fusion method (addition and concatenation respectively).

They propose a framework that generalizes both residual and dense connections.

The work that is most closely related to ours is that of Srivastava et al. (2015) ; they define a topology that includes residual connections and higher order correlations.

However, we offer a new perspective on the higher order correlations.

In addition, we experiment with a) large scale problems, b) with linear blocks that highway networks have not used.

A popular line of research modifies the transformation path of each residual block.

In ResNeXt (Xie et al., 2017) and Inception (Szegedy et al., 2017 ) the authors add group convolutions in the transformation path.

In Zhang et al. (2017) the transformation path is a (set of) residual blocks itself, i.e. they obfuscate one residual block inside another.

In wide residual networks (Zagoruyko & Komodakis, 2016) they advocate for increased width of each block.

All related works are complementary to ours, since we do not modify the transformation path modules.

The applications of residual networks are diverse and often with impressive results.

Such applications include object detection/recognition (Szegedy et al., 2017) , face recognition (Deng et al., 2019) , generative models (Miyato et al., 2018) .

However, these networks have tens or hundreds of residual blocks.

A line of research that reduces the number of parameters is that of pruning the network (Han et al., 2015; Chin et al., 2018) .

Han et al. (2015) propose to prune the weights with small magnitude, while Chin et al. (2018) propose a meta-learning technique to improve heuristic pruning techniques.

However, pruning does not reduce the training resources (it even increases the time because for a single model, we train the network at least twice), and it is largely based on hand-engineered heuristics.

That is, there is no solid understanding of the theoretical properties of pruning methods.

To develop our intuition, we explore the linear residual block in sec. 3.1, i.e. the residual block without any activation functions.

In sec. 3.2, we extend the proposed formulation in the presence of activation functions as typically used in ResNet.

Figure 2: Schematic of the (a) original, (b) our residual block for the t th layer.

The path that includes C is referred to as the transformation path; while the other (with the identity transformation) is referred to as the shortcut path.

The symbol C denotes the operations in the transformation path, e.g. convolutions in He et al. (2016) .

The symbols N 1 , N 2 are normalization layers, e.g. batch normalization or 1ˆ1 convolutions.

The symbol˚denotes an element-wise product.

Before the introduction of the residual block, all the neural networks were a composition of linear layers, e.g. fully-connected or convolutional layers, and activation functions.

The (input) representation was transformed in each layer through a linear operation if we ignore the activation functions.

The residual block of He et al. (2016) enables the input representation to pass through unchanged by introducing a two-pathway block consisting of a shortcut path and a transformation path.

The t th residual block (in a network) is x t`1 " x t`C x t for input x t 3 .

That is, the residual block expresses a linear difference equation.

We propose instead a new residual block that captures second order information.

The new residual block is:

for input x t with S the same dimensions as C and˚an element-wise product.

The scalar parameter α P R is learnable and plays the role of scaling the significance of the second order interactions.

To reduce the number of parameters, we can share the parameters of S and C; we introduce some normalization in the quadratic term.

The proposed residual block then is expressed as:

with N 1 , N 2 two normalization operators.

The proposed residual block is depicted in Fig. 2 .

Frequently activation functions are used in conjunction with the residual block.

We consider a residual block with activation functions and two convolutions in the transformation path.

If we define the function f t pxq " C 2 pφpC 1 xqq, the residual block is x t`1 " φpx t`ft px t qq with φ denoting an activation function, such as RELU.

To avoid cluttering the notation, batch normalization is ignored in the last equation.

The proposed residual block in the presence of activation functions becomes:

The proposed residual block can be used with different architectures, e.g. three convolutions or with group convolutions.

Implementation details: All the optimization-related hyper-parameters, e.g. the optimizer, the learning rate, the initializations, the number of epochs etc., remain the same as in the original ResNet.

Further improvement can be obtained by tuning those values for our residual block, but this is out of our scope.

Unless mentioned otherwise, each experiment is conducted 5 times and the average and the standard deviation are reported.

The following four datasets are used in this work:

1. CIFAR10 (Krizhevsky et al., 2014) : This is a widely used dataset that contains 60, 000 images of natural scenes.

Each image is of resolution 32ˆ32ˆ3 and is classified in one of the 10 classes.

2. CIFAR100 (Krizhevsky et al.) : This is an extension over CIFAR10; it includes the same amount of images but there are 100 classes.

3. ImageNet (Russakovsky et al., 2015) : The ImageNet 2012 dataset (Russakovsky et al., 2015) comprises 1.28 million training images and 50K validation images from 1000 different classes.

We train networks on the training set and report the top-1 and top-5 error on the validation set.

4.

Speech Commands (Warden, 2018) : This newly released dataset contains 60, 000 audio files; each audio contains a single word of a duration of one second.

There are 35 different words (classes) with each word having 1, 500´4, 100 recordings.

Every audio file is converted into a mel-spectrogram of resolution 32ˆ32.

Below, we conduct an experiment on image classification on CIFAR10 in sec. 4.1; we modify the experiment by removing the activation functions, i.e. have networks linear with respect to the weights, in sec. 4.2.

Sequentially, image classification experiments on CIFAR100 and ImageNet are conducted in sec. 4.3 and sec. 4.4 respectively.

In addition to the image classification experiments, we exhibit how the proposed residual block can be used on image generation in sec. 4.5.

Furthermore, an experiment on audio classification is conducted in sec. 4.6.

We utilize CIFAR10 as a popular dataset for classification.

We train each method for 120 epochs with batch size 128.

The SGD optimizer is used with initial learning rate of 0.1.

The learning rate is multiplied with a factor of 0.1 in epochs 40, 60, 80, 100.

We use two ResNet architectures, i.e. ResNet18 and ResNet34, as baselines.

Our model, called Newton-ResNet, is built with the proposed residual blocks; we add enough blocks to match the performance of the respective baseline.

In Table 1 the two different ResNet baselines are compared against Newton-ResNet; the respective Newton-ResNet models have the same accuracy.

However, each Newton-ResNet has " 40% less parameters than the respective baseline.

In addition, we visualize the test accuracy for ResNet18 and the respective Newton-ResNet in Fig. 3 .

The test error of the two models is similar throughout the training; a similar phenomenon is observed for ResNet34 in Fig. 6 .

We remove all the activation functions, both from the transformation path and the output activation functions.

The rest of the settings remain the same as in sec. 4.1.

As can be noticed in Table 2 , Newton-ResNet outperforms ResNet18 by a significant margin when removing the activation functions.

It is worth noting that the performance of Newton-ResNet with/without activation functions differs by 7%, i.e. decent performance can be obtained without any activation functions.

o Ð φ(BN(conv(o))); end 5:

for i=1:lin_proj do 8:

s Ð φ2(norm(conv1ˆ1(s))); end 9:

xt`1 Ð xt`o`s 10:

return xt`1 11: end function Table 3 : The differences of the proposed method with the original residual block are highlighted in blue.

The x_proj, lin_proj are 1ˆ1 convolutions added for normalization purposes in the proposed residual block.

We verify the aforementioned classification results on CIFAR100.

The settings remain the same as in sec. 4.1.

As can be noticed in Table 4 the test accuracy of ResNet34 and Newton-ResNet is similar, however Newton-ResNet has " 44% less parameters.

The experiment of sec. 4.2 with the linear blocks is repeated on CIFAR100.

That is, we remove all the activation functions and train the networks.

The accuracy of each method is reported on Table 5 .

The difference observed in sec. 4.1 becomes even more pronounced.

That is, ResNet performs poorly in this case and is substantially outperformed by Newton-ResNet.

We perform a large-scale classification experiment on ImageNet; due to the computational resources required, this experiment is conducted only once.

Following standard practices, we utilize the following data augmentation techniques: (1) normalization through mean RGB-channel subtraction, (2) random resized crop to 224ˆ224, (3) scale from 5% to 100%, (4) aspect ratio from (5) random horizontal flip.

During inference, we perform the following augmentation techniques: (1) normalization through mean RGB-channel subtraction, (2) scale to 256ˆ256, and (3) single center crop to 224ˆ224.

All models are trained on a DGX station with four Tesla V100 (32GB) GPUs.

We use Mxnet 4 and choose float16 instead of float32 to achieve 3.5ˆacceleration and half the GPU memory consumption.

In our preliminary experiments, we noticed that the second order might cause numeric overflow in float16; this was not observed in the rest of the experiments that use float32.

Hence, we use a tanh as a normalization for the second order term, i.e. the last term of (3).

Optimization is performed using SGD with momentum 0.9, weight decay 1e´4 and a mini-batch size of 1024.

The initial learning rate is set to 0.4 and decreased by a factor of 10 at 30, 60, and 80 epochs.

Models are trained for 90 epochs from scratch, using linear warm-up of the learning rate during first five epochs according to Goyal et al. (2017) .

For other batch sizes due to the limitation of GPU memory, we linearly scale the learning rate (e.g. learning rate 0.1 for batch size 256).

We report both the Top-1 and Top-5 single-crop validation error in Table 6 .

For a fair comparison, we report the results from our training in both the original ResNet and Newton-ResNet 5 .

Newton-ResNet consistently improves the performance with an extremely small increase in computational complexity and model size.

Remarkably, Newton-ResNet50 achieves a single-crop Top-5 validation error of 6.358%, exceeding ResNet50 (6.838%) by 0.48% and approaching the performance achieved by the much deeper ResNet101 network (6.068% Top-5 error).

The loss and Top-1 error throughout the training are visualized in Fig. 4 , which demonstrates that the proposed method performs favourably to the baseline ResNetwhen the same amount of residual blocks are used.

Table 7 , we remove all the "relu" activation functions for both baseline and the proposed method.

Without "relu", Newton-ResNet50 achieves a single-crop Top-5 validation error of 9.114%, significantly exceeding ResNet50 (71.562%) and approaching the performance achieved by the "relu" version (6.068% Top-5 error).

Deep discriminative networks include hundreds of layers, while their generative counterparts' depth is critical and restricted (mainly because they are hard to train and fit in existing hardware).

We explore whether we can reduce the number of residual blocks in generative models.

Generative Adversarial Networks (GAN) of Goodfellow et al. (2014) have dominated the related literature (Miyato et al., 2018) due to their impressive visual results.

GANs include two modules, a generator and a discriminator, which are both implemented with resnet-based neural networks.

The generator samples z from a prior distribution, e.g. uniform, and tries to model the target distribution; the discriminator tries to distinguish between the samples synthesized from the generator and the target distribution.

GAN is typically optimized with an alternating gradient descent method.

We select the architecture of Miyato et al. (2018) (SNGAN) as a strong baseline on CIFAR10.

The baseline includes 3 resnet blocks in the generator and 3 in the discriminator.

We replace the original residual blocks with the proposed residual blocks; two such blocks in each module suffice to achieve the same performance.

That is, we reduce by 1 the blocks in both the generator and the discriminator.

In Table 8 the experimental result is summarized.

Note that the experiment is conducted 10 times and the mean and variance are reported.

In Fig. 5 some random samples synthesized by the two methods are depicted; visually the generated samples are similar.

The quantitative results are added in Table 9 .

The two models share the same accuracy, however Newton-ResNet includes 38% less parameters.

This is consistent with the experiments on classical image datasets, i.e. sec. 4.1.

In this work, we establish a link between the residual blocks of ResNet architectures and learning decent directions in solving non-linear least squares (e.g., each block can be considered as a decent direction).

We exploit this link and we propose a novel residual block that uses second order interactions as reminiscent of Newton's numerical optimization method (i.e., learning Newton-like descent directions).

Newton-type methods are likely to converge faster than first order methods (e.g., Gauss-Newton).

We demonstrate that in the proposed architecture this translates to less residual blocks (i.e., less decent directions) in the network for achieving the same performance.

We conduct validation experiments on image and audio classification with residual networks and verify our intuition.

Furthermore, we illustrate that with our block it is possible to remove the non-linear activation functions and still achieve competitive performance.

@highlight

We demonstrate how residual blocks can be viewed as Gauss-Newton steps; we propose a new residual block that exploits second order information.