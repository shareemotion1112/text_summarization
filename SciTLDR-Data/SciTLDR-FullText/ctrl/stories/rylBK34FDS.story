In seeking for sparse and efficient neural network models, many previous works investigated on enforcing L1 or L0 regularizers to encourage weight sparsity during training.

The L0 regularizer measures the parameter sparsity directly and is invariant to the scaling of parameter values.

But it cannot provide useful gradients and therefore requires complex optimization techniques.

The L1 regularizer is almost everywhere differentiable and can be easily optimized with gradient descent.

Yet it is not scale-invariant and causes the same shrinking rate to all parameters, which is inefficient in increasing sparsity.

Inspired by the Hoyer measure (the ratio between L1 and L2 norms) used in traditional compressed sensing problems, we present DeepHoyer, a set of sparsity-inducing regularizers that are both differentiable almost everywhere and scale-invariant.

Our experiments show that enforcing DeepHoyer regularizers can produce even sparser neural network models than previous works, under the same accuracy level.

We also show that DeepHoyer can be applied to both element-wise and structural pruning.

The use of deep neural network (DNN) models has been expanded from handwritten digit recognition (LeCun et al., 1998) to real-world applications, such as large-scale image classification (Simonyan & Zisserman, 2014) , self driving (Makantasis et al., 2015) and complex control problems (Mnih et al., 2013) .

However, a modern DNN model like AlexNet (Krizhevsky et al., 2012) or ResNet (He et al., 2016) often introduces a large number of parameters and computation load, which makes the deployment and real-time processing on embedded and edge devices extremely difficult (Han et al., 2015b; a; .

Thus, model compression techniques, especially pruning methods that increase the sparsity of weight matrices, have been extensively studied to reduce the memory consumption and computation cost of DNNs (Han et al., 2015b; a; Guo et al., 2016; Louizos et al., 2017b; Liu et al., 2015) .

Most of the previous works utilize some form of sparsity-inducing regularizer in searching for sparse neural networks.

The 1 regularizer, originally proposed by Tibshirani (1996) , can be easily optimized through gradient descent for its convex and almost everywhere differentiable property.

Therefore it is widely used in DNN pruning: Liu et al. (2015) directly apply 1 regularization to all the weights of a DNN to achieve element-wise sparsity; present structural sparsity via group lasso, which applies an 1 regularization over the 2 norms of different groups of parameters.

However, it has been noted that the value of the 1 regularizer is proportional to the scaling of parameters (i.e. ||??W || 1 = |??|??||W || 1 ), so it "scales down" all the elements in the weight matrices with the same speed.

This is not efficient in finding sparsity and may sacrifice the flexibility of the trained model.

On the other hand, the 0 regularizer directly reflects the real sparsity of weights and is scale invariant (i.e. ||??W || 0 = ||W || 0 , ????? = 0), yet the 0 norm cannot provide useful gradients.

Han et al. (2015b) enforce an element-wise 0 constraint by iterative pruning a fixed percentage of smallest weight elements, which is a heuristic method and therefore can hardly achieve optimal compression rate.

Some recent works mitigate the lack of gradient information by integrating 0 regularization with stochastic approximation (Louizos et al., 2017b) or more complex optimization methods (e.g. ADMM) .

These additional measures brought overheads to the optimization process, making the use of these methods on larger networks difficult.

To achieve even sparser neural networks, we argue to move beyond 0 and 1 regularizers and seek for a sparsity-inducing regularizer that is both almost everywhere differentiable (like 1 ) and scale-invariant (like 0 ).

Beyond the 1 regularizer, plenty of non-convex sparsity measurements have been used in the field of feature selection and compressed sensing (Hurley & Rickard, 2009; .

Some popular regularizers like SCAD (Fan & Li, 2001) , MDP (Zhang et al., 2010) and Trimmed 1 (Yun et al., 2019 ) use a piece-wise formulation to mitigate the proportional scaling problem of 1 .

The piece-wise formulation protects larger elements by having zero penalty to elements greater than a predefined threshold.

However, it is extremely costly to manually seek for the optimal trimming threshold, so it is hard to obtain optimal result in DNN pruning by using these regularizers.

The transformed 1 regularizer formulated as

(a+1)|wi| a+|wi| manages to smoothly interpolate between 1 and 0 by tuning the hyperparameter a (Ma et al., 2019) .

However, such an approximation is close to 0 only when a approaches infinity, so the practical formulation of the transformed 1 (i.e. a = 1) is still not scale-invariant.

Particularly, we are interested in the Hoyer regularizer (Hoyer, 2004) , which estimates the sparsity of a vector with the ratio between its 1 and 2 norms.

Comparing to other sparsity-inducing regularizers, Hoyer regularizer achieves superior performance in the fields of non-negative matrix factorization (Hoyer, 2004) , sparse reconstruction (Esser et al., 2013; Tran et al., 2018) and blend deconvolution (Krishnan et al., 2011; Repetti et al., 2015) .

We note that Hoyer regularizer is both almost everywhere differentiable and scale invariant, satisfying the desired property of a sparsityinducing regularizer.

We therefore propose DeepHoyer, which is the first Hoyer-inspired regularizers for DNN sparsification.

Specifically, the contributions of this work include:

??? Hoyer-Square (HS) regularizer for element-wise sparsity: We enhance the original Hoyer regularizer to the HS regularizer and achieve element-wise sparsity by applying it in the training of DNNs.

The HS regularizer is both almost everywhere differentiable and scale invariant.

It has the same range and minima structure as the 0 norm.

Thus, the HS regularizer presents the ability of turning small weights to zero while protecting and maintaining those weights that are larger than an induced, gradually adaptive threshold; ??? Group-HS regularizer for structural sparsity, which is extended from the HS regularizer;

??? Generating sparser DNN models: Our experiments show that the proposed regularizers beat state-of-the-arts in both element-wise and structural weight pruning of modern DNNs.

It is well known that high redundancy pervasively exists in DNNs.

Consequently, pruning methods have been extensively investigated to identify and remove unimportant weights.

Some heuristic pruning methods (Han et al., 2015b; Guo et al., 2016) simply remove weights in small values to generate sparse models.

These methods usually require long training time without ensuring the optimality, due to the lack of theoretical understanding and well-formulated optimization .

Some works formulate the problem as a sparsity-inducing optimization problem, such as 1 regularization (Liu et al., 2015; Park et al., 2016 ) that can be optimized using standard gradientbased algorithms, or 0 regularization (Louizos et al., 2017b; ) which requires stochastic approximation or special optimization techniques.

We propose DeepHoyer regularizers in this work, which belong to the line of sparsity-inducing optimization research.

More specific, the proposed Hoyer-Square regularizer for element-wise pruning is scale-invariant and can serve as an differentiable approximation to the 0 norm.

Furthermore, it can be optimized by gradient-based optimization methods in the same way as the 1 regularization.

With these properties, the HoyerSquare regularizer achieves a further 38% and 63% sparsity improvement on LeNet-300-100 model and LeNet-5 model respectively comparing to previous state-of-the-arts, and achieves the highest sparsity on AlexNet without accuracy loss.

Structurally sparse DNNs attempt to create regular sparse patterns that are friendly for hardware execution.

To achieve the goal, propose to remove filters with small norms; apply group Lasso regularization based methods to remove various structures (e.g., filters, channels, layers) in DNNs and the similar approaches are used to remove neurons (Alvarez & et al., 2017a; Neklyudov et al., 2017 ), yet these methods are not applicable in large-scale problems like ImageNet.

We further advance the DeepHoyer to learn structured sparsity (such as reducing filters and channels) with the newly proposed "Group-HS" regularization.

The Group-HS regularizer further improves the computation reduction of the LeNet-5 model by 8.8% from the 1 based method , and by 110.6% from the 0 based method (Louizos et al., 2017b) .

Experiments on ResNet models reveal that the accuracy-speedup tradeoff induced by Group-HS constantly stays above the Pareto frontier of previous methods.

More detailed results can be found in Section 5.

Sparsity measures provide tractable sparsity constraints for enforcement during problem solving and therefore have been extensively studied in the compressed sensing society.

In early non-negative matrix factorization (NMF) research, a consensus was that a sparsity measure should map a ndimensional vector X to a real number S ??? [0, 1], such that the possible sparsest vectors with only one nonzero element has S = 1, and a vector with all equal elements has S = 0 (Hoyer, 2004) .

Unders the assumption, the Hoyer measure was proposed as follows

It can be seen that

Thus, the normalization in Equation (1) fits the measure S(X) into the [0, 1] interval.

According to the survey by Hurley & Rickard (2009) , among the six desired heuristic criteria of sparsity measures, the Hoyer measure satisfies five, more than all other commonly applied sparsity measures.

Given its success as a sparsity measure in NMF, the Hoyer measure has been applied as a sparsity-inducing regularizer in optimization problems such as blind deconvolution (Repetti et al., 2015) and image deblurring (Krishnan et al., 2011) .

Without the range constraint, the Hoyer regularizer in these works adopts the form

directly, as the ratio of the 1 and 2 norms.

Figure 1 compares the Hoyer regularizer and the 1 regularizer.

Unlike the the 1 norm with a single minimum at the origin, the Hoyer regularizer has minima along axes, the structure of which is very similar to the 0 norm's.

Moreover, the Hoyer regularizer is scale-invariant, i.e. R(??X) = R(X), because both the 1 norm and the 2 norm are proportional to the scale of X. The gradients of the Hoyer regularizer are purely radial, leading to "rotations" towards the nearest axis.

These features make the Hoyer regularizer outperform the 1 regularizer on various tasks (Esser et al., 2013; Tran et al., 2018; Krishnan et al., 2011; Repetti et al., 2015) .

The theoretical analysis by Yin et al. (2014) also proves that the Hoyer regularizer has a better guarantee than the 1 norm on recovering sparse solutions from coherent and redundant representations.

Inspired by the Hoyer regularizer, we propose two types of DeepHoyer regularizers: the Hoyer-Square regularizer (HS) for element-wise pruning and the Group-HS regularizer for structural pruning.

Since the process of the element-wise pruning is equivalent to regularizing each layer's weight with the 0 norm, it is intuitive to configure the sparsity-inducing regularizer to have a similar behavior as the 0 norm.

As shown in Inequality (2), the value of the original Hoyer regularizer of a Ndimensional nonzero vector lies between 1 and ??? N , while its 0 norm is within the range of [1, N ].

Thus we propose to apply the square of Hoyer regularizer, namely Hoyer-Square (HS), to the weights W of a layer, like

The proposed HS regularizer behaves as a differentiable approximation to the 0 norm.

First, both regularizers now have the same range of

holds for ????? = 0, so as the 0 norm.

Moreover, as the squaring operator monotonously increases in the range of [1,

, the Hoyer-Square regularizer's minima remain along the axes as the Hoyer regularizer's do (see Figure 1) .

In other words, they have similar minima structure as the 0 norm.

At last, the Hoyer-Square regularizer is also almost everywhere differentiable and Equation (4) formulates the gradient of H S w.r.t.

an element w j in the weight matrix W :

Very importantly, this formulation induces a trimming effect: when H S (W ) is being minimized through gradient descent, w j moves towards 0 if |w j |<

i w 2 i i |wi| , otherwise moves away from 0.

In other words, unlike the 1 regularizer which tends to shrink all elements, our Hoyer-Square regularizer will turn weights in small value to zero meanwhile protecting large weights.

Traditional trimmed regularizers (Fan & Li, 2001; Zhang et al., 2010; Yun et al., 2019) usually define a trimming threshold as a fixed value or percentage.

Instead, the HS regularizer can gradually extend the scope of pruning as more weights coming close to zero.

This behavior can be observed in the gradient descent path shown in Figure 2 .

Beyond element-wise pruning, structural pruning is often more preferred because it can construct the sparsity in a structured way and therefore achieve higher computation speed-up on general computation platforms .

The structural pruning is previously empowered by the group lasso (Yuan & Lin, 2006; , which is the sum (i.e. 1 norm) of the 2 norms of all the groups within a weight matrix like

where ||W || 2 = i w 2 i represents the 2 norm, w (g) is a group of elements in the weight matrix W which consists of G such groups.

Following the same approach in Section 4.1, we use the Hoyer-Square regularizer to replace the 1 regularizer in the group lasso formulation and define the Group-HS (G H ) regularizer in Equation (6): Note that the second equality holds when and only when the groups cover all the elements of W without overlapping with each other.

Our experiments in this paper satisfy this requirement.

However, the Group-HS regularizer can always be used in the form of the first equality when overlapping exists across groups.

The gradient and the descent path of the Group-HS regularizer are very similar to those of the Hoyer-Square regularizer, and therefore we omit the detailed discussion here.

The derivation of the Group-HS regularizer's gradient shall be found in Appendix A.

The deployment of the DeepHoyer regularizers in DNN training follows the common layer-based regularization approach Liu et al., 2015) .

For element-wise pruning, we apply the Hoyer-Square regularizer to layer weight matrix W (l) for all L layers, and directly minimize it alongside the DNN's original training objective L(W (1:L) ).

The 2 regularizer can also be added to the objective if needed.

Equation (7) presents the training objective with H S defined in Equation (3).

Here, ?? and ?? are pre-selected weight decay parameters for the regularizers.

For structural pruning, we mainly focus on pruning the columns and rows of fully connected layers and the filters and channels of convolutional layers.

More specific, we group a layer in filter-wise and channel-wise fashion as proposed by and then apply the Group-HS regularizer to the layer.

The resulted optimization objective is formulated in Equation (8).

Here N l is the number of filters and C l is the number of channels in the l th layer if it is a convolutional layer.

If the l th layer is fully connected, then N l and C l is the number of rows and columns respectively.

?? n , ?? c and ?? are pre-selected weight decay parameters for the regularizers.

The recent advance in stochastic gradient descent (SGD) method provides satisfying results under large-scale non-convex settings (Sutskever et al., 2013; Kingma & Ba, 2014) , including DNNs with non-convex objectives (Auer et al., 1996) .

So we can directly optimize the DeepHoyer regularizers with the same SGD optimizer used for the original DNN training objective, despite their nonconvex formulations.

Our experiments show that the tiny-bit nonconvexity induced by DeepHoyer does not affect the performance of DNNs.

The pruning is conducted by following the common three-stage operations: (1) train the DNN with the DeepHoyer regularizer, (2) prune all the weight elements smaller than a predefined small threshold, and (3) finetune the model by fixing all the zero elements and removing the DeepHoyer regularizer.

The proposed DeepHoyer regularizers are first tested on the MNIST benchmark using the LeNet-300-100 fully connected model and the LeNet-5 CNN model (LeCun et al., 1998) .

We also conduct tests on the CIFAR-10 dataset (Krizhevsky & Hinton, 2009 ) with ResNet models (He et al., 2016) in various depths, and on ImageNet ILSVRC-2012 benchmark (Russakovsky et al., 2015) with the AlexNet model (Krizhevsky et al., 2012) and the ResNet-50 model (He et al., 2016) .

All the models are implemented and trained in the PyTorch deep learning framework (Paszke et al., 2017) , where we match the model structure and the benchmark performance with those of previous works for the fairness of comparison.

The experiment results presented in the rest of this section show that the proposed DeepHoyer regularizers consistently outperform previous works in both element-wise and structural pruning.

Detailed information on the experiment setups and the parameter choices of our reported results can be found in Appendix B. Table 1 and Table 2 summarize the performance of the proposed Hoyer-square regularizer on the MNIST benchmark, with comparisons against state of the art (SOTA) element-wise pruning methods.

Without losing the testing accuracy, training with the Hoyer-Square regularizer reduces the number of nonzero weights by 54.5?? on the LeNet-300-100 model and by 122?? on the LeNet-5 model.

Among all the methods, ours achieves the highest sparsity: it is a 38% improvement on the LeNet-300-100 model and a 63% improvement on the LeNet-5 model comparing to the best available methods.

Additional results in Appendix C.1 further illustrates the effect of the Hoyer-Square regularizer on each layer's weight distribution during the training process.

The element-wise pruning performance on the AlexNet model testing on the ImageNet benchmark is presented in Table 3 .

Without losing the testing accuracy, the Hoyer-Square regularizer improves the compression rate by 21.3??.

This result is the highest among all methods, even better than the ADMM method which requires two additional Lagrange multipliers and involves the optimization of two objectives.

Considering that the optimization of the Hoyer-Square regularizer can be directly realized on a single objective without additional variables, we conclude that the Hoyer-Square regularizer can achieve a sparse DNN model with a much lower cost.

A more detailed layer-by-layer sparsity comparison of the compressed model can be found in Appendix C.2.

We perform the ablation study for performance comparison between the Hoyer-Square regularizer and the original Hoyer regularizer.

The results in Tables 1, 2 and 3 all show that the Hoyer-Square regularizer always achieves a higher compression rate than the original Hoyer regularizer.

The layer-wise compression results show that the Hoyer-Square regularizer emphasizes more on the layers with more parameters (i.e. FC1 for the MNIST models).

This corresponds to the fact that the value of the Hoyer-Square regularizer is proportional to the number of non-zero elements in the weight.

These observations validate our choice to use the Hoyer-Square regularizer for DNN compression.

This section reports the effectiveness of the Group-HS regularizer in structural pruning tasks.

Here we mainly focus on the number of remaining neurons (output channels for convolution layers and rows for fully connected layers) after removing the all-zero channels or rows in the weight matrices.

The comparison is then made based on the required float-point operations (FLOPs) to inference with the remaining neurons, which indeed represents the potential inference speed of the pruned model.

As shown in Table 4 , training with the Group-HS regularizer can reduce the number of FLOPs by 16.2?? for the LeNet-300-100 model with a slight accuracy drop.

This is the highest speedup among all existing methods achieving the same testing accuracy.

Table 5 shows that the Group-HS regularizer can reduce the number of FLOPs of the LeNet-5 model by 12.4??, which outperforms most of the existing work-an 8.8% increase from the 1 based method ) and a 110.6% increase from the 0 based method (Louizos et al., 2017b) .

Only the Bayesian compression (BC) method with the group-horseshoe prior (BC-GHS) (Louizos et al., 2017a) achieves a slightly higher speedup on the LeNet-5 model.

However, the complexity of high dimensional Bayesian inference limits BC's capability.

It is difficult to apply BC to ImageNet-level problems and large DNN models like ResNet.

Sparse VD 99.0% 660.2k (28.79%) 14-19-242-131 GL 99.0% 201.8k (8.80%) 3-12-192-500 SBP (Neklyudov et al., 2017) 99.1% 212.8k (9.28%) 3-18-284-283 BC-GNJ (Louizos et al., 2017a) 99.0% 282.9k (12.34%) 8-13-88-13 BC-GHS (Louizos et al., 2017a) 99.0% 153.4k (6.69%) 5-10-76-16 0 hc (Louizos et al., 2017b) 99.0% 390.7k (17.04%) 9-18-26-25 Bayes 1trim (Yun et al., 2019) 99.0% 334.0k (14.57%) 8-17-53-19

Group-HS 99.0% 169.9k (7.41%) 5-12-139-13 In contrast, the effectiveness of the Group-HS regularizer can be easily extended to deeper models and larger datasets, which is demonstrated by our experiments.

We apply the Group-HS regularizer to ResNet models (He et al., 2016) on the CIFAR-10 and the ImageNet datasets.

Pruning ResNet has long been considered difficult due to the compact structure of the ResNet model.

Since previous works usually report the compression rate at different accuracy, we use the "accuracy-#FLOPs" plot to represent the tradeoff.

The tradeoff between the accuracy and the FLOPs are explored in this work by changing the strength of the Group-HS regularizer used in training.

Figure 3 shows the performance of DeepHoyer constantly stays above the Pareto frontier of previous methods.

In this work, we propose DeepHoyer, a set of sparsity-inducing regularizers that are both scaleinvariant and almost everywhere differentiable.

We show that the proposed regularizers have similar range and minima structure as the 0 norm, so it can effectively measure and regularize the sparsity of the weight matrices of DNN models.

Meanwhile, the differentiable property enables the proposed regularizers to be simply optimized with standard gradient-based methods, in the same way as the 1 regularizer is.

In the element-wise pruning experiment, the proposed Hoyer-Square regularizer achieves a 38% sparsity increase on the LeNet-300-100 model and a 63% sparsity increase on the LeNet-5 model without accuracy loss comparing to the state-of-the-art.

A 21.3?? model compression rate is achieved on AlexNet, which also surpass all previous methods.

In the structural pruning experiment, the proposed Group-HS regularizer further reduces the computation load by 24.4% from the state-of-the-art on LeNet-300-100 model.

It also achieves a 8.8% increase from the 1 based method and a 110.6% increase from the 0 based method of the computation reduction rate on the LeNet-5 model.

For CIFAR-10 and ImageNet dataset, the accuracy-FLOPs tradeoff achieved by training ResNet models with various strengths of the Group-HS regularizer constantly stays above the Pareto frontier of previous methods.

These results prove that the DeepHoyer regularizers are effective in achieving both element-wise and structural sparsity in deep neural networks, and can produce even sparser DNN models than previous works.

In this section we provide detailed derivation of the gradient of the Hoyer-Square regularizer and the Group-GS regularizer w.r.t.

an element w j in the weight matrix W .

The gradient of the Hoyer-Square regularizer is shown in Equation (9).

The formulation shown in Equation (4) is achieved at the end of the derivation.

The gradient of the Group-HS regularizer is shown in Equation (10).

For simplicity we use the form shown in the second equality of Equation (6), where there is no overlapping between the groups.

Here we assume that w j belongs to group w (??) .

B DETAILED EXPERIMENT SETUP

The MNIST dataset (LeCun et al., 1998 ) is a well known handwritten digit dataset consists of greyscale images with the size of 28 ?? 28 pixels.

We use the dataset API provided in the "torchvision" python package to access the dataset.

In our experiments we use the whole 60,000 training set images for the training and the whole 10,000 testing set images for the evaluation.

All the accuracy results reported in the paper are evaluated on the testing set.

Both the training set and the testing set are normalized to have zero mean and variance one.

Adam optimizer (Kingma & Ba, 2014) with learning rate 0.001 is used throughout the training process.

All the MNIST experiments are done with a single TITAN XP GPU.

Both the LeNet-300-100 model and the LeNet-5 model are firstly pretrained without the sparsityinducing regularizer, where they achieve the testing accuracy of 98.4% and 99.2% respectively.

Then the models are further trained for 250 epochs with the DeepHoyer regularizers applied in the objective.

The weight decay parameters (??s in Equation (7) and (8)) are picked by hand to reach the best result.

In the last step, we prune the weight of each layer with threshold proportional to the standard derivation of each layer's weight.

The threshold/std ratio is chosen to achieve the highest sparsity without accuracy loss.

All weight elements with a absolute value smaller than the threshold is set to zero and is fixed during the final finetuning.

The pruned model is finetuned for another 100 steps without DeepHoyer regularizers and the best testing accuracy achieved is reported.

Detailed parameter choices used in achieving the reported results are listed in Table 6 .

The ImageNet dataset is a large-scale color-image dataset containing 1.2 million images of 1000 categories (Russakovsky et al., 2015) , which has long been utilized as an important benchmark on image classification problems.

In this paper, we use the "ILSVRC2012" version of the dataset, which can be found at http://www.image-net.org/challenges/LSVRC/ 2012/nonpub-downloads.

We use all the data in the provided training set to train our model, and use the provided validation set to evaluate our model and report the testing accuracy.

We follow the data reading and preprocessing pipeline suggested by the official PyTorch ImageNet example (https://github.com/pytorch/examples/tree/master/imagenet).

For training images, we first randomly crop the training images to desired input size, then apply random horizontal flipping and finally normalize them before feeding them into the network.

Validation images are first resized to 256 ?? 256 pixels, then center cropped to desired input size and normalized in the end.

We use input size 227 ?? 227 pixels for experiments on the AlexNet, and input size 224 ?? 224 for experiments on the ResNet-50.

All the models are optimized with the SGD optimizer Sutskever et al. (2013) , and the batch size is chosen as 256 for all the experiments.

Two TITAN XP GPUs are used in parallel for the AlexNet training and four are used for the ResNet-50 training.

One thing worth noticing is that the AlexNet model provided in the "torchvision" package is not the ordinary version used in previous works Han et al. (2015b); .

Therefore we reimplement the AlexNet model in PyTorch for fair comparison.

We pretrain the implemented model for 90 epochs and achieve 19.8 % top-5 error, which is the same as reported in previous works.

In the AlexNet experiment, the reported result in Table 3 is achieved by applying the Hoyer-Square regularizer with decay parameter 1e-6.

Before the pruning, the model is firstly train from the pretrained model with the Hoyer-Square regularizer for 90 epochs, where an initial learning rate 0.001 is used.

An 2 regularization with 1e-4 decay is also applied.

We then prune the convolution layers with threshold 1e-4 and the FC layers with threshold equal to 0.4?? of their standard derivations.

The model is then finetuned until the best accuracy is reached.

The learning rate is decayed by 0.1 for every 30 epochs of training.

The training process with the Hoyer regularizer and the T 1 regularizer (Ma et al., 2019) is the same as the HS regularizer.

For the reported result, we use decay 1e-3 and FC threshold 0.8?? std for the Hoyer regularizer, and use decay 2e-5 and FC threshold 1.0?? std for the T 1 regularizer.

For the ResNet-50 experiments on ImageNet, the model architecture and pretrained model provided in the "torchvision" package is directly utilized, which achieves 23.85% top-1 error and 7.13% top-5 error.

All the reported results in Figure 3 and Table 8 are achieved with 90 epochs of training with the Group-HS regularizer from the pretrained model using initial learning rate 0.1.

All the models are pruned with 1e-4 as threshold and finetuned to the best accuracy.

We only tune the decay parameter of the Group-HS regularizer to explore the accuracy-FLOPs tradeoff.

The exact decay parameter used for each result is specified in Table 8 .

We also use the CIFAR-10 dataset (Krizhevsky & Hinton, 2009 ) to evaluate the structural pruning performance on ResNet-56 and ResNet-110 models.

The CIFAR-10 dataset can be directly accessed through the dataset API provided in the "torchvision" python package.

Standard preprocessing, including random crop, horizontal flip and normalization is used on the training set to train the model.

We implemented the ResNet models for CIFAR-10 following the description in (He et al., 2016) , and pretrain the models for 164 epochs.

Learning rate is set to 0.1 initially, and decayed by 0.1 at epoch 81 and epoch 122.

The pretrained ResNet-56 model reaches the testing accuracy of 93.14 %, while the ResNet-110 model reaches 93.62 %.

Similar to the ResNet-50 experiment, we start with the pretrained models and train with the Group-HS regularizer.

Same learning rate scheduling is used for both pretraining and training with Group-HS.

All the models are pruned with 1e-4 as threshold and finetuned to the best accuracy.

The decay parameters of the Group-HS regularizer used to get the result in Figure 3 is specified in Table 9 and Table 10 .

C ADDITIONAL EXPERIMENT RESULTS

Here we demonstrate how will the weight distribution change in each layer at different stages of our element-wise pruning process.

Since most of the weight elements will be zero in the end, we only plot the histogram of nonzero weight elements for better observation.

The histogram of each layer of the LeNet-300-100 model and the LeNet-5 model are visualized in Figure 4 and Figure 5 respectively.

It can be seen that majority of the weights will be concentrated near zero after applying the H S regularizer during training, while rest of the weight elements will spread out in a wide range.

The weights close to zero are then set to be exactly zero, and the model is finetuned with zero weights fixed.

The resulted histogram shows that most of the weights are pruned away, only a small amount of nonzero weights are remaining in the model.

Table 7 compares the element-wise pruning result of the Hoyer-Square regularizer on AlexNet with other methods in a layer-by-layer fashion.

It can be seen that the Hoyer-Square regularizer achieves high pruning rates on the largest layers (i.e. FC1-3).

This observation is consistent with the observation made on the element-wise pruning performance of models on the MNIST dataset.

In this section we list the data used to plot Figure 3 .

Table 8 shows the result of pruning ResNet-50 model on ImageNet, Table 9 shows the result of pruning ResNet-56 model on CIFAR-10 and Table 10 shows the result of pruning ResNet-110 model on CIFAR-10.

For all the tables, the results of previous works are listed on the top, and are ordered based on publication year.

Results achieved with the Group-HS regularizer are listed below, marked with the regularization strength used for the training.

<|TLDR|>

@highlight

We propose almost everywhere differentiable and scale invariant regularizers for DNN pruning, which can lead to supremum sparsity through standard SGD training.

@highlight

The paper proposes a scale-invariant regularizer (DeepHoyer) inspired by the Hoyer measure to enforce sparsity in neural networks. 