Successful training of convolutional neural networks is often associated with suffi- ciently deep architectures composed of high amounts of features.

These networks typically rely on a variety of regularization and pruning techniques to converge to less redundant states.

We introduce a novel bottom-up approach to expand representations in fixed-depth architectures.

These architectures start from just a single feature per layer and greedily increase width of individual layers to attain effective representational capacities needed for a specific task.

While network growth can rely on a family of metrics, we propose a computationally efficient version based on feature time evolution and demonstrate its potency in determin- ing feature importance and a networks’ effective capacity.

We demonstrate how automatically expanded architectures converge to similar topologies that benefit from lesser amount of parameters or improved accuracy and exhibit systematic correspondence in representational complexity with the specified task.

In contrast to conventional design patterns with a typical monotonic increase in the amount of features with increased depth, we observe that CNNs perform better when there is more learnable parameters in intermediate, with falloffs to earlier and later layers.

Estimating and consequently adequately setting representational capacity in deep neural networks for any given task has been a long standing challenge.

Fundamental understanding still seems to be insufficient to rapidly decide on suitable network sizes and architecture topologies.

While widely adopted convolutional neural networks (CNNs) such as proposed by BID17 ; BID27 ; BID12 ; Zagoruyko & Komodakis (2016) demonstrate high accuracies on a variety of problems, the memory footprint and computational complexity vary.

An increasing amount of recent work is already providing valuable insights and proposing new methodology to address these points.

For instance, the authors of BID2 propose a reinforcement learning based meta-learning approach to have an agent select potential CNN layers in a greedy, yet iterative fashion.

Other suggested architecture selection algorithms draw their inspiration from evolutionary synthesis concepts BID25 BID22 .

Although the former methods are capable of evolving architectures that rival those crafted by human design, it is currently only achievable at the cost of navigating large search spaces and hence excessive computation and time.

As a trade-off in present deep neural network design processes it thus seems plausible to consider layer types or depth of a network to be selected by an experienced engineer based on prior knowledge and former research.

A variety of techniques therefore focus on improving already well established architectures.

Procedures ranging from distillation of one network's knowledge into another , compressing and encoding learned representations BID8 , pruning alongside potential re-training of networks BID7 BID26 BID10 , small capacity increases on trained networks in transfer-learning scenarios (WardeFarley et al., 2014) and the employment of different regularization terms during training BID11 BID15 BID23 BID0 , are just a fraction of recent efforts in pursuit of reducing representational complexity while attempting to retain accuracy.

Underlying mechanisms rely on a multitude of criteria such as activation magnitudes BID26 and small weight values BID7 that are used as pruning metrics for either single neurons or complete feature maps, in addition to further combination with regularization and penalty terms.

Common to these approaches is the necessity of training networks with large parameter quantities for maximum representational capacity to full convergence and the lack of early identification of insufficient capacity.

In contrast, this work proposes a bottom-up approach with the following contributions:• We introduce a computationally efficient, intuitive metric to evaluate feature importance at any point of training a neural network.

The measure is based on feature time evolution, specifically the normalized cross-correlation of each feature with its initialization state.• We propose a bottom-up greedy algorithm to automatically expand fixed-depth networks that start with one feature per layer until adequate representational capacity is reached.

We base addition of features on our newly introduced metric due to its computationally efficient nature, while in principle a family of similarly constructed metrics is imaginable.• We revisit popular CNN architectures and compare them to automatically expanded networks.

We show how our architectures systematically scale in terms of complexity of different datasets and either maintain their reference accuracy at reduced amount of parameters or achieve better results through increased network capacity.•

We provide insights on how evolved network topologies differ from their reference counterparts where conventional design commonly increases the amount of features monotonically with increasing network depth.

We observe that expanded architectures exhibit increased feature counts at early to intermediate layers and then proceed to decrease in complexity.

While the choice and size of deep neural network model indicate the representational capacity and thus determine which functions can be learned to improve training accuracy, training of neural networks is further complicated by the complex interplay of choice of optimization algorithm and model regularization.

Together, these factors define define the effective capacity.

This makes training of deep neural networks particularly challenging.

One practical way of addressing this challenge is to boost model sizes at the cost of increased memory and computation times and then applying strong regularization to avoid over-fitting and minimize generalization error.

However, this approach seems unnecessarily cumbersome and relies on the assumption that optimization difficulties are not encountered.

We draw inspiration from this challenge and propose a bottom-up approach to increase capacity in neural networks along with a new metric to gauge the effective capacity in the training of (deep) neural networks with stochastic gradient descent (SGD) algorithms.

In SGD the objective function J (Θ) is commonly equipped with a penalty on the parameters R (Θ), yielding a regularized objective function: DISPLAYFORM0 Here, α weights the contribution of the penalty.

The regularization term R (Θ) is typically chosen as a L 2 -norm, coined weight-decay, to decrease model capacity or a L 1 -norm to enforce sparsity.

Methods like dropout BID29 and batch normalization BID14 are typically employed as further implicit regularizers.

In principle, our approach is inspired by earlier works of BID10 who measure a complete feature's importance by taking the L 1 -norm of the corresponding weight tensor instead of operating on individual weight values.

In the same spirit we assign a single importance value to each feature based on its values.

However we do not use the weight magnitude directly and instead base our metric on the following hypothesis: While a feature's absolute magnitude or relative change between two subsequent points in time might not be adequate measures for direct importance, the relative amount of change a feature experiences with respect to its original state provides an indicator for how many times and how much a feature is changed when presented with data.

Intuitively we suggest that features that experience high structural changes must play a more vital role than any feature that is initialized and does not deviate from its original states' structure.

There are two potential reasons why a feature that has randomly been initialized does not change in structure: The first being that its form is already initialized so well that it does not need to be altered and can serve either as is or after some scalar rescaling or shift in order to contribute.

The second possibility is that too high representational capacity, the nature of the cost function, too large regularization or the type of optimization algorithm prohibit the feature from being learned, ultimately rendering it obsolete.

As deep neural networks are commonly initialized from using a distribution over high-dimensional space the first possibility seems unlikely BID6 .

As one way of measuring the effective capacity at a given state of learning, we propose to monitor the time evolution of the normalized cross-correlation for all weights with respect to their state at initialization.

For a neural network composed of layers l = 1, 2, . . . , L − 1 and complementary weight-tensors W f l+1 ,t is the mean taken over input feature and potential spatial dimensions.• depicts the Hadamard product that we use in an extended fashion from matrices to tensors where each dimension is multiplied in an element-wise fashion analogously.

Similarly the terms in the denominator are defined as the L 2 -norm of the weight-tensor taken over said dimensions and thus resulting in a scalar value.

Above equation is applicable to multi-layer perceptrons as well as features with spatial dimensionality, where the sum over the input feature space F l then also includes a feature's spatial dimensions j l × k l .

The metric is easily interpretable as no structural changes of features lead to a value of zero and importance approaches unity the more a feature is deviating in structure.

The usage of normalized cross-correlation with the L 2 -norm in the denominator has the advantage of having an inherent invariance to effects such as translations or rescaling of weights stemming from various regularization contributions.

Therefore the contribution of the sum-term in equation 1 does not change the value of the metric if the gradient term vanishes.

This is in contrast to the measure proposed by BID10 , as absolute weight magnitudes are affected by rescaling and make it more difficult to interpret the metric in an absolute way and find corresponding thresholds.

Due to this normalized nature of our metric we are able to move away from a top-down pruning approach, as presented by BID10 , altogether and instead follow a bottom-up procedure where we incrementally add features, eradicating the need to train a large architecture in the first place.

We propose a new method to converge to architectures that encapsulate necessary task complexity without the necessity of training huge networks in the first place.

Starting with one feature in each layer, we expand our architecture as long as the effective capacity, as estimated through our metric, is not met and all features experience structural change.

In contrast to methods such as BID2 ; BID25 ; BID22 we do not consider flexible depth and treat the amount of layers in a network as a prior based on the belief of hierarchical composition of the underlying factors.

This fixed-depth prior is similar to the approach of BID21 , who modify fixed-depth fully-connected networks, albeit without explicitly introducing capacity as a term in the optimization loss, but as a modular auxiliary step on top of a conventional SGD algorithm.

Our method, shown in algorithm 1, can be summarized as follows:1.

For a given network arrangement in terms of function type, depth and a set of hyperparameters: initialize each layer with one feature and proceed with (mini-batch) SGD.

DISPLAYFORM0 for mini-batches in training set do 4: reset ← f alse 5: Compute gradient and perform update step 6: for l = 1 to L − 1 do 7: DISPLAYFORM1 Update c if max(c l t ) < 1 − then 11: DISPLAYFORM2 reset ← true if reset == true then

Re-initialize parameters Θ, t = 0, λ = λ 0 , . . .

end if

end for 19: DISPLAYFORM0 The constant is a numerical stability parameter that we set to a small value such as 10 −6 , but could in principle as well be used as a constraint.

We have decided to include the re-initialization in step 3 (lines 15 − 17) to avoid the pitfalls of falling into local minima (see appendix section A.5 for a brief discussion on the need of re-initialization).

Despite this sounding like a major detriment to our method, we show that networks nevertheless rapidly converge to a stable architectural solution that comes at less than perchance expected computational overhead and at the benefit of avoiding training of too large architectures.

Naturally at least one form of explicit or implicit regularization has to be present in the learning process in order to prevent infinite expansion of the architecture.

We would like to emphasize that we have chosen the metric defined in equation 2 as a basis for the decision of when to expand an architecture, but in principle a family of similarly constructed metrics is imaginable.

We have chosen this particular metric because it does not directly depend on gradient or higher-order term calculation and only requires multiplication of weights with themselves.

The algorithm furthermore doesn't intervene in the SGD optimization step as the formulation doesn't involve alteration of the cost function.

Thus, a major advantage is that computation of equation 2 can be executed modularly and independently on top of a conventional DNN optimization procedure, can be parallelized completely and therefore executed at less cost than a regular forward pass through the network.

We revisit some of the most established architectures "GFCNN" BID5 ) "VGG-A & E" BID27 and "Wide Residual Network: WRN" (Zagoruyko & Komodakis, 2016 ) (see appendix for architectural details) with batch normalization BID14 .

We compare the number of learnable parameters and achieved accuracies with those obtained through expanded architectures that started from a single feature in each layer.

For each architecture we include all-convolutional variants BID28 that are similar to WRNs (minus the skip-connections), where all pooling layers are replaced by convolutions with larger stride.

All fully-connected layers are furthermore replaced by a single convolution (affine, no activation function) that maps directly onto the space of classes.

Even though the value of more complex type of sub-sampling functions has already empirically been demonstrated BID19 , the amount of features of the replaced layers has been constrained to match in dimensionality with the preceding convolution layer.

We would thus like to further extend and analyze the role of layers involving sub-sampling by decoupling the dimensionality of these larger stride convolutional layers.

We consider these architectures as some of the best CNN architectures as each of them has been chosen and tuned carefully according to extensive amounts of hyper-parameter search.

As we would like to demonstrate how representational capacity in our automatically constructed networks scales with increasing task difficulty, we perform experiments on the MNIST (LeCun et al., 1998), CIFAR10 & 100 BID16 ) datasets that intuitively represent little to high classification challenge.

We also show some preliminary experiments on the ImageNet BID24 dataset with "Alexnet" BID17 to conceptually show that the algorithm is applicable to large scale challenges.

All training is closely inspired by the procedure specified in Zagoruyko & Komodakis (2016) with the main difference of avoiding heavy preprocessing.

We preprocess all data using only trainset mean and standard deviation (see appendix for exact training parameters).

Although we are in principle able to achieve higher results with different sets of hyper-parameters and preprocessing methods, we limit ourselves to this training methodology to provide a comprehensive comparison and avoid masking of our contribution.

We train all architectures five times on each dataset using a Intel i7-6800K CPU (data loading) and a single NVIDIA Titan-X GPU.

Code has been written in both Torch7 BID3 and PyTorch (http://pytorch.org/) and will be made publicly available.

We first provide a brief example for the use of equation 2 through the lens of pruning to demonstrate that our metric adequately measures feature importance.

We evaluate the contribution of the features by pruning the weight-tensor feature by feature in ascending order of feature importance values and re-evaluating the remaining architecture.

We compare our normalized cross-correlation metric 2 to the L 1 weight norm metric introduced by BID10 and ranked mean activations evaluated over an entire epoch.

In FIG2 we show the pruning of a trained GFCNN, expecting that such a network will be too large for the easier MNIST and too small for the difficult CIFAR100 task.

For all three metrics pruning any feature from the architecture trained on CIFAR100 immediately results in loss of accuracy, whereas the architecture trained on MNIST can be pruned to a smaller set of parameters by greedily dropping the next feature with the currently lowest feature importance value.

We notice how all three metrics perform comparably.

However, in contrast to the other two metrics, our normalized cross-correlation captures whether a feature is important on absolute scale.

For MNIST the curve is very close to zero, whereas the metric is close to unity for all CIFAR100 features.

Ultimately this is the reason our metric, in the way formulated in equation 2, is used for the algorithm presented in 1 as it doesn't require a difficult process to determine individual layer threshold values.

Nevertheless it is imaginable that similar metrics based on other tied quantities (gradients, activations) can be formulated in analogous fashion.

As our main contribution lies in the bottom-up widening of architectures we do not go into more detailed analysis and comparison of pruning strategies.

We also remark that in contrast to a bottomup approach to finding suitable architectures, pruning seems less desirable.

It requires convergent training of huge architectures with lots of regularization before complexity can be reduced, pruning is not capable of adding complexity if representational capacity is lacking, pruning percentages are difficult to interpret and compare (i.e. pruning percentage is 0 if the architecture is adequate), a majority of parameters are pruned only in the last "fully-connected" layers BID7 , and pruning strategies as suggested by BID7 ; BID26 ; BID10 tend to require many cross-validation with consecutive fine-tuning steps.

We thus continue with the bottom-up perspective of expanding architectures from low to high representational capacity.

We use the described training procedure in conjunction with algorithm 1 to expand representational complexity by adding features to architectures that started with just one feature per layer with the following additional settings:Architecture expansion settings and considerations: Our initial experiments added one feature at a time, but large speed-ups can be introduced by means of adding stacks of features.

Initially, we avoided suppression of late re-initialization to analyze the possibility that rarely encountered worst-case behavior of restarting on an almost completely trained architecture provides any benefit.

After some experimentation our final report used a stability parameter ending the network expansion if half of the training has been stable (no further change in architecture) and added F exp = 8 and F exp = 16 features per expansion step for MNIST and CIFAR10 & 100 experiments respectively.

We show an exemplary architecture expansion of the GFCNN architecture's layers for MNIST and CIFAR100 datasets in FIG3 and the evolution of the overall amount of parameters for five different experiments.

We observe that layers expand independently at different points in time and more features are allocated for CIFAR100 than for MNIST.

When comparing the five different runs we can identify that all architectures converge to a similar amount of network parameters, however at different points in time.

A good example to see this behavior is the solid (green) curve in the MNIST example, where the architecture at first seems to converge to a state with lower amount of parameters and after some epochs of stability starts to expand (and re-initialize) again until it ultimately converges similarly to the other experiments.

We continue to report results obtained for the different datasets and architectures in table 1.

The table illustrates the mean and standard deviation values for error, total amount of parameters and the mean overall time taken for five runs of algorithm 1 (tentative as heavily dependent on F exp .

Deviations can be fairly large due to the behavior observed in 2).

We make the following observations:• Without any prior on layer widths, expanding architectures converge to states with at least similar accuracies to the reference at reduced amount of parameters, or better accuracies by allocating more representational capacity.

• For each architecture type there is a clear trend in network capacity that is increasing with dataset complexity from MNIST to CIFAR10 to CIFAR100 1 .•

Even though we have introduced re-initialization of the architecture the time taken by algorithm 1 is much less than one would invest when doing a manual, grid-or random-search.1 For the WRN CIFAR100 architecture the * signifies hardware memory limitations due to the arrangement of architecture topology and thus expansion is limited.

This is because increased amount of early-layer features requires more memory in contrast to late layers, which is particularly intense for the coupled WRN architecture.• Shallow GFCNN architectures are able to gain accuracy by increasing layer width, although there seems to be a natural limit to what width alone can do, especially when heavy regularization is in place (we discuss such a non-trivial example with corresponding curves in appendix section A.4).

This is in agreement with observations pointed out in other works such as Ba & Caurana (2014); BID30 .•

The large reference VGG-E (lower accuracy than VGG-A on CIFAR) and WRN-28-10 (complete over-fit on MNIST) seem to run into optimization difficulties for these datasets.

However, expanded alternate architecture clearly perform significantly better.

In general we observe that these benefits are due to unconventional, yet always coinciding, network topology of our expanded architectures.

These topologies suggest that there is more to CNNs than simply following the rule of thumb of increasing the number of features with increasing architectural depth.

Before proceeding with more detail on these alternate architecture topologies, we want to again emphasize that we do not report experiments containing extended methodology such as excessive preprocessing, data augmentation, the oscillating learning rates proposed in BID20 or better sets of hyper-parameters for reasons of clarity, even though accuracies rivaling state-of-the-art performances can be achieved in this way.

Almost all popular convolutional neural network architectures follow a design pattern of monotonically increasing feature amount with increasing network depth BID18 BID5 BID27 BID28 BID12 Zagoruyko & Komodakis, 2016; BID20 BID30 VGG-E-all-conv Figure 3 : Mean and standard deviation of topologies as evolved from the expansion algorithm for a VGG-E and VGG-E all-convolutional architecture run five times on MNIST, CIFAR10 and CI-FAR100 datasets respectively.

Top panels show the reference architecture, whereas bottom shows automatically expanded architecture alternatives.

Expanded architectures vary in capacity with dataset complexity and topologically differ from their reference counterparts.as constructed by our expansion algorithm in five runs on the three datasets.

Apart from noticing the systematic variations in representational capacity with dataset difficulty, we furthermore find topological convergence with small deviations from one training to another.

We observe the highest feature dimensionality in early to intermediate layers with generally decreasing dimensionality towards the end of the network differing from conventional CNN design patterns.

Even if the expanded architectures sometimes do not deviate much from the reference parameter count, accuracy seems to be improved through this topological re-arrangement.

For architectures where pooling has been replaced with larger stride convolutions we also observe that dimensionality of layers with sub-sampling changes independently of the prior and following convolutional layers suggesting that highly-complex sub-sampling operations are learned.

This an extension to the proposed all-convolutional variant of BID28 , where introduced additional convolutional layers were constrained to match the dimensionality of the previously present pooling operations.

If we view the deep neural network as being able to represent any function that is limited rather by concepts of continuity and boundedness instead of a specific form of parameters, we can view the minimization of the cost function as learning a functional mapping instead of merely adopting a set of parameters BID6 .

We hypothesize that evolved network topologies containing higher feature amount in early to intermediate layers generally follow a process of first mapping into higher dimensional space to effectively separate the data into many clusters.

The network can then more readily aggregate specific sets of features to form clusters distinguishing the class subsets.

Empirically this behavior finds confirmation in all our evolved network topologies that are visualized in the appendix.

Similar formation of topologies, restricted by the dimensionality constraint of the identity mappings, can be found in the trained residual networks.

While BID11 has shown that deep VGG-like architectures do not perform well, an interesting question for future research could be whether plainly stacked architectures can perform similarly to residual networks if the arrangement of feature dimensionality is differing from the conventional design of monotonic increase with depth.

We show two first experiments on the ImageNet dataset using an all-convolutional Alexnet to show that our methodology can readily be applied to large scale.

The results for the two runs can be found in table 2 and corresponding expanded architectures are visualized in the appendix.

We observe that the experiments seem to follow the general pattern and again observe that topological rearrangement of the architecture yields substantial benefits.

In the future we would like to extend experimentation to more promising ImageNet architectures such as deep VGG and residual networks.

However, these architectures already require 4-8 GPUs and large amounts of time in their baseline evaluation, which is why we presently are not capable of evaluating these architectures and keep this section at a very brief proof of concept level.

In this work we have introduced a novel bottom-up algorithm to start neural network architectures with one feature per layer and widen them until a task depending suitable representational capacity is achieved.

For the use in this framework we have presented one potential computationally efficient and intuitive metric to gauge feature importance.

The proposed algorithm is capable of expanding architectures that provide either reduced amount of parameters or improved accuracies through higher amount of representations.

This advantage seems to be gained through alternative network topologies with respect to commonly applied designs in current literature.

Instead of increasing the amount of features monotonically with increasing depth of the network, we empirically observe that expanded neural network topologies have high amount of representations in early to intermediate layers.

Future work could include a re-evaluation of plainly stacked deep architectures with new insights on network topologies and extended evaluation on different domain data.

We have furthermore started to replace the currently present re-initialization step in the proposed expansion algorithm by keeping learned filters.

In principle this approach looks promising but does need further systematic analysis of new feature initialization with respect to the already learned feature subset and accompanied investigation of orthogonality to avoid falling into local minima.

A.1 DATASETS• MNIST BID18 : 50000 train images of hand-drawn digits of spatial size 28 × 28 belonging to one of 10 equally sampled classes.• CIFAR10 & 100 BID16 : 50000 natural train images of spatial size 32 × 32 each containing one object belonging to one of 10/100 equally sampled classes.• ImageNet BID24 : Approximately 1.2 million training images of objects belonging to one of 1000 classes.

Classes are not equally sampled with 732-1300 images per class.

Dataset contains 50 000 validation images, 50 per class.

Scale of objects and size of images varies.

All training is closely inspired by the procedure specified in Zagoruyko & Komodakis (2016) with the main difference of avoiding heavy preprocessing.

Independent of dataset, we preprocess all data using only trainset mean and standard deviation.

All training has been conducted using crossentropy as a loss function and weight initialization following the normal distribution as proposed by BID11 .

All architectures are trained with batch-normalization with a constant of 1 · 10 −3 , a batch-size of 128, a L 2 weight-decay of 5 · 10 −4 , a momentum of 0.9 and nesterov momentum.

We use initial learning rates of 0.1 and 0.005 for the CIFAR and MNIST datasets respectively.

We have rescaled MNIST images to 32 × 32 (CIFAR size) and repeat the image across color channels in order to use architectures without modifications.

CIFAR10 & 100 are trained for 200 epochs and the learning rate is scheduled to be reduced by a factor of 5 every multiple of 60 epochs.

MNIST is trained for 60 epochs and learning rate is reduced by factor of 5 once after 30 epochs.

We augment the CIFAR10 & 100 training by introducing horizontal flips and small translations of up to 4 pixels during training.

No data augmentation has been applied to the MNIST dataset.

We use the single-crop technique where we rescale the image such that the shorter side is equal to 224 and take a centered crop of spatial size 224 × 224.

In contrast to BID17 we limit preprocessing to subtraction and divison of trainset mean and standard deviation and do not include local response normalization layers.

We randomly augment training data with random horizontal flips.

We set an initial learning rate of 0.1 and follow the learning rate schedule proposed in BID17 that drops the learning rate by a factor of 0.1 every 30 epochs and train for a total of 74 epochs.

The amount of epochs for the expansion of architectures is larger due to the re-initialization.

For these architectures the mentioned amount of epochs corresponds to training during stable conditions, i.e. no further expansion.

The procedure is thus equivalent to training the converged architecture from scratch.

A.3 ARCHITECTURES GFCNN BID5 Three convolution layer network with larger filters (followed by two fully-connected layers, but without "maxout".

The exact sequence of operations is:VGG BID27 "VGG-A" (8 convolutions) and "VGG-E" (16 convolutions) networks.

Both architectures include three fully-connected layers.

We set the number of features in the MLP to 512 features per layer instead of 4096 because the last convolutional layer of these architecture already produces outputs of spatial size 1 × 1 (in contrast to 7 × 7 on ImageNet) on small datasets.

Batch normalization is used before the activation functions.

Examples of stacking convolutions that do not alter spatial dimensionality to create deeper architectures.

WRN (Zagoruyko & Komodakis, 2016) Wide Residual Network architecture: We use a depth of 28 convolutional layers (each block completely coupled, no bottlenecks) and a width-factor of 10 as reference.

When we expand these networks this implies an inherent coupling of layer blocks due to dimensional consistency constraints with outputs from identity mappings.

Alexnet BID17 We use the all convolutional variant where we replace the first fully-connected large 6 × 6 × 256 → 4096 layer with a convolution of corresponding spatial filter size and 256 filters and drop all further fully-connected layers.

The rationale behind this decision is that previous experiments, our own pruning experiments and those of BID10 BID7 , indicate that original fully-connected layers are largely obsolete.

As previously explained in the main-body in section 2 we generally differentiate between a deep neural network model's representational and effective capacity.

Whereas the former only includes the layer and feature choices, the latter reflects the actual capability of a neural network to fit the data when including the choice of optimization and corresponding regularization.

In practice this means that a network can manage to under-fit on the training data if those parameters aren't chosen appropriately, even if the network itself is comprised of an abundant amount of parameters.

Reference Expanded GFCNN-all-conv -CIFAR100 Figure 4 : Left to right: Loss, train and validation curves for a GFCNN-all-conv trained on the CIFAR100 dataset.

The dashed curve presents a reference architecture implementation whereas the solid lines correspond to the behavior of an expanded network (Note that we have omitted the nonstable part of the expansion and only show the 200 stable epochs).

It can be observed that loss and training accuracy improve by a large margin for the expanded architecture, whereas validation accuracy benefits only slightly.

In figure 4 we show an example of such behavior on a GFCNN-all-conv architecture trained on the CIFAR100 dataset and what the implications are with respect to algorithm 1 and the expansion framework introduced in this work.

The original architecture can be observed to severely under-fit the train-data which is largely due to the use of L 2 regularization, dropout and batch-normalization at the same time.

Without lifting these constraints we can observe that our expanded architecture bridges this gap in loss and train-accuracy completely by increasing the width of the layers and thus allocating a lot more parameters (see 1 where we observe a ≈ 5× increase in parameters).

We can further observe that the validation accuracy benefits only slightly from this increase (here ≈ 3%).

In other words, our expansion algorithm tries to counter the under-fitting due to the heavy regularization in order to fit the training set, which results in some over-fitting in return.

However we would like to make the following two remarks of why we believe this could be desirable: Initialization of new features during training: for a used initialization scheme, e.g. Xavier BID4 or Kaiming BID11 initialization, is such a scheme employed throughout the entire process of network expansion?

Adopting such practice would initialize newly added features differently from previously initialized features due to varying Fan-in and Fan-out dimensionality.

Depending on the precise nature of the initialization scheme each subsequent feature would be scaled to decreasing or increasing magnitude, which could lead to undesired behavior such as some features not being used or old features becoming obsolete.

One potential solution could be to make sure that newly added features are aligned in a sense that they follow the distribution of already learned features.

Explicitly, when drawing the first features from e.g. a normal distribution with mean 0 and some std, new features could stem from a normal distribution with mean and std of already learned features.

Whenever a new filter without re-initialization is introduced, although the currently learned knowledge base is not modified, a perturbation to the classifier is introduced.

Let us assume the extreme case where the initial amount of training epochs was 200 and we make an incremental addition of 1 or 2 features in epoch 199.

Although a classifier should be able to rapidly recover and get back to high accuracy by fine-tuning, this fine-tuning is also needed to make sure the training does not end in perturbed state.

In addition the magnitude of the perturbation depends on the size of the already existing feature base, leading to another important factor that should be regarded.

In addition to figure 3 we show mean evolved topologies including standard deviation for all architectures and datasets reported in table 1 and 2.

In figure 5 and 6 all shallow and VGG-A architectures and their respective all-convolutional variants are shown.

Figure 7 shows the constructed wide residual 28 layer network architectures where blocks of layers are coupled due to the identity mappings.

Figure 8 shows the two expanded Alexnet architectures as trained on ImageNet.

As explained in the main section we see that all evolved architectures feature topologies with large dimensionality in early to intermediate layers instead of in the highest layers of the architecture as usually present in conventional CNN design.

For architectures where pooling has been replaced with larger stride convolutions we also observe that dimensionality of layers with sub-sampling changes independently of the prior and following convolutional layers suggesting that highly-complex pooling operations are learned.

This an extension to the proposed all-convolutional variant of BID28 , where introduced additional convolutional layers were constrained to match the dimensionality of the previously present pooling operations.

<|TLDR|>

@highlight

A bottom-up algorithm that expands CNNs starting with one feature per layer to architectures with sufficient representational capacity.

@highlight

Proposes to dynamically adjust the feature map depth of a fully convolutional neural network, formulating a measure of self-resemblance and boosting performance.

@highlight

Introduces a simple correlation-based metric to measure whether filters in neural networks are being used effectively, as a proxy for effective capacity.

@highlight

Aims to address the deep learning architecture search problem via incremental addition and removal of channels in intermediate layers of the network.