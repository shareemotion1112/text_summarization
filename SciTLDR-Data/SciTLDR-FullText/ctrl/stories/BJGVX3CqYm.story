Recent work in network quantization has substantially reduced the time and space complexity of neural network inference, enabling their deployment on embedded and mobile devices with limited computational and memory resources.

However, existing quantization methods often represent all weights and activations with the same precision (bit-width).

In this paper, we explore a new dimension of the design space: quantizing different layers with different bit-widths.

We formulate this problem as a neural architecture search problem and propose a novel differentiable neural architecture search (DNAS) framework to efficiently explore its exponential search space with gradient-based optimization.

Experiments show we surpass the state-of-the-art compression of ResNet on CIFAR-10 and ImageNet.

Our quantized models with 21.1x smaller model size or 103.9x lower computational cost can still outperform baseline quantized or even full precision models.

Recently, ConvNets have become the de-facto method in a wide range of computer vision tasks, achieving state-of-the-art performance.

However, due to high computation complexity, it is nontrivial to deploy ConvNets to embedded and mobile devices with limited computational and storage budgets.

In recent years, research efforts in both software and hardware have focused on lowprecision inference of ConvNets.

Most of the existing quantization methods use the same precision for all (or most of) the layers of a ConvNet.

However, such uniform bit-width assignment can be suboptimal since quantizing different layers can have different impact on the accuracy and efficiency of the overall network.

Although mixed precision computation is widely supported in a wide range of hardware platforms such as CPUs, FPGAs, and dedicated accelerators, prior efforts have not thoroughly explored the mixed precision quantization of ConvNets.

For a ConvNet with N layers and M candidate precisions in each layer, we want to find an optimal assignment of precisions to minimize the cost in terms of model size, memory footprint or computation, while keeping the accuracy.

An exhaustive combinatorial search has exponential time complexity (O(M N )).

Therefore, we need a more efficient approach to explore the design space.

In this work, we propose a novel, effective, and efficient differentiable neural architecture search (DNAS) framework to solve this problem.

The idea is illustrated in FIG0 .

The problem of neural architecture search (NAS) aims to find the optimal neural net architecture in a given search space.

In the DNAS framework, we represent the architecture search space with a stochastic super net where nodes represent intermediate data tensors of the super net (e.g., feature maps of a ConvNet) and edges represent operators (e.g., convolution layers in a ConvNet).

Any candidate architecture can be seen as a child network (sub-graph) of the super net.

When executing the super net, edges are executed stochastically and the probability of execution is parameterized by some architecture parameters θ.

Under this formulation, we can relax the NAS problem and focus on finding the optimal θ that gives the optimal expected performance of the stochastic super net.

The child network can then be sampled from the optimal architecture distribution.

We solve for the optimal architecture parameter θ by training the stochastic super net with SGD with respect to both the network's weights and the architecture parameter θ.

To compute the gradient of θ, we need to back propagate gradients through discrete random variables that control the stochastic edge execution.

To address this, we use the Gumbel SoftMax function BID9 ) to "soft-control" the edges.

This allows us to directly compute the gradient estimation of θ with a controllable trade-off between bias and variance.

Using this technique, the stochastic super net becomes fully differentiable and can be effectively and efficiently solved by SGD.

We apply the DNAS framework to solve the mixed precision quantization problem, by constructing a super net whose macro architecture (number of layers, filter size of each layer, etc.) is the same as the target network.

Each layer of the super net contains several parallel edges representing convolution operators with quantized weights and activations with different precisions.

We show that using DNAS to search for layer-wise precision assignments for ResNet models on CIFAR10 and ImageNet, we surpass the state-of-the-art compression.

Our quantized models with 21.1x smaller model size or 103.9x smaller computational cost can still outperform baseline quantized or even full precision models.

The DNAS pipeline is very fast, taking less than 5 hours on 8 V100 GPUs to complete a search on ResNet18 for ImageNet, while previous NAS algorithms (such as Zoph & Le (2016)) typically take a few hundred GPUs for several days.

Last, but not least, DNAS is a general architecture search framework that can be applied to other problems such as efficient ConvNet-structure discovery.

Due to the page limit, we will leave the discussion to future publications.

Network quantization received a lot of research attention in recent years.

Early works such as BID5 ; Zhu et al. (2016) ; BID13 mainly focus on quantizing neural network weights while still using 32-bit activations.

Quantizing weights can reduce the model size of the network and therefore reduce storage space and over-the-air communication cost.

More recent works such as BID17 Zhou et al. (2016) ; BID10 Zhuang et al. (2018) quantize both weights and activations to reduce the computational cost on CPUs and dedicated hardware accelerators.

Most of the works use the same precision for all or most of the layers of a network.

The problem of mixed precision quantization is rarely explored.

Neural Architecture Search becomes an active research area in recent two years.

Zoph & Le (2016) first propose to use reinforcement learning to generate neural network architectures with high accuracy and efficiency.

However, the proposed method requires huge amounts of computing resources.

BID16 propose an efficient neural architecture search (ENAS) framework that drastically reduces the computational cost.

ENAS constructs a super network whose weights are shared with its child networks.

They use reinforcement learning to train an RNN controller to sample better child networks from the super net.

More recently, propose DARTS, a differentiable architecture search framework.

DARTS also constructs a super net whose edges (candidate operators) are parameterized with coefficients computed by a SoftMax function.

The super net is trained and edges with the highest coefficients are kept to form the child network.

Our proposed DNAS framework is different from DARTS since we use a stochastic super net -in DARTS, the execution of edges are deterministic and the entire super net is trained together.

In DNAS, when training the super net, child networks are sampled, decoupled from the super net and trained independently.

The idea of super net and stochastic super net is also used in BID18 Veniat & Denoyer (2017) to explore macro architectures of neural nets.

Another related work is BID8 , which uses AutoML for model compression through network pruning.

To the best of our knowledge, we are the first to apply neural architecture search to model quantization.

Normally 32-bit (full-precision) floating point numbers are used to represent weights and activations of neural nets.

Quantization projects full-precision weights and activations to fixed-point numbers with lower bit-width, such as 8, 4, and 1 bit.

We follow DoReFa-Net (Zhou et al. (2016) ) to quantize weights and PACT ) to quantize activations.

See Appendix A for more details.

For mixed precision quantization, we assume that we have the flexibility to choose different precisions for different layers of a network.

Mixed precision computation is widely supported by hardware platforms such as CPUs, FPGAs, and dedicated accelerators.

Then the problem is how should we decide the precision for each layer such that we can maintain the accuracy of the network while minimizing the cost in terms of model size or computation.

Previous methods use the same precision for all or most of the layers.

We expand the design space by choosing different precision assignment from M candidate precisions at N different layers.

While exhaustive search yields O(M N ) time complexity, our automated approach is efficient in finding the optimal precision assignment.

4.1 NEURAL ARCHITECTURE SEARCH Formally, the neural architecture search (NAS) problem can be formulated as DISPLAYFORM0 Here, a denotes a neural architecture, A denotes the architecture space.

w a denotes the weights of architecture a. L(·, ·) represents the loss function on a target dataset given the architecture a and its parameter w a .

The loss function is differentiable with respect to w a , but not to a. As a consequence, the computational cost of solving the problem in FORMULA0 is very high.

To solve the inner optimization problem requires to train a neural network a to convergence, which can take days.

The outer problem has a discrete search space with exponential complexity.

To solve the problem efficiently, we need to avoid enumerating the search space and evaluating each candidate architecture one-by-one.

We discuss the idea of differentiable neural architecture search (DNAS).

The idea is illustrated in FIG0 .

We start by constructing a super net to represent the architecture space A. The super net is essentially a computational DAG (directed acyclic graph) that is denoted as G = (V, E).

Each node v i ∈ V of the super net represents a data tensor.

Between two nodes v i and v j , there can be K ij edges connecting them, indexed as e ij k .

Each edge represents an operator parameterized by its trainable weight w ij k .

The operator takes the data tensor at v i as its input and computes its output as e ij k (v i ; w ij k ).

To compute the data tensor at v j , we sum the output of all incoming edges as DISPLAYFORM0 With this representation, any neural net architecture a ∈ A can be represented by a subgraph DISPLAYFORM1 For simplicity, in a candidate architecture, we keep all the nodes of the graph, so V a = V .

And for a pair of nodes v i , v j that are connected by K ij candidate edges, we only select one edge.

Formally, in a candidate architecture a, we re-write equation FORMULA1 as DISPLAYFORM2 where m ij k ∈ {0, 1} is an "edge-mask" and k m ij k = 1.

Note that though the value of m ij k is discrete, we can still compute the gradient to m ij k .

Let m be a vector that consists of m ij k for all e ij k ∈ E. For any architecture a ∈ A, we can encode it using an "edge-mask" vector m a .

So we re-write the loss function in equation FORMULA0 to an equivalent form as L(m a , w a ).We next convert the super net to a stochastic super net whose edges are executed stochastically.

For each edge e ij k , we let m ij k ∈ {0, 1} be a random variable and we execute edge e ij k when m ij k is sampled to be 1.

We assign each edge a parameter θ ij k such that the probability of executing e ij k is DISPLAYFORM3 The stochastic super net is now parameterized by θ, a vector whose elements are θ ij k for all e ij k ∈ E. From the distribution P θ , we can sample a mask vector m a that corresponds to a candidate architecture a ∈ A. We can further compute the expected loss of the stochastic super net as DISPLAYFORM4 The expectation of the loss function is differentiable with respect to w a , but not directly to θ, since we cannot directly back-propagate the gradient to θ through the discrete random variable m a .

To estimate the gradient, we can use Straight-Through estimation BID1 ) or REINFORCE (Williams (1992)).

Our final choice is to use the Gumbel Softmax technique BID9 ), which will be explained in the next section.

Now that the expectation of the loss function becomes fully differentiable, we re-write the problem in equation FORMULA0 as DISPLAYFORM5 The combinatorial optimization problem of solving for the optimal architecture a ∈ A is relaxed to solving for the optimal architecture-distribution parameter θ that minimizes the expected loss.

Once we obtain the optimal θ, we acquire the optimal architecture by sampling from P θ .

We use stochastic gradient descent (SGD) to solve Equation (5).

The optimization process is also denoted as training the stochastic super net.

We compute the Monte Carlo estimation of the gradient DISPLAYFORM0 where a i is an architecture sampled from distribution P θ and B is the batch size.

Equation FORMULA7 provides an unbiased estimation of the gradient, but it has high variance, since the size of the architecture space is orders of magnitude larger than any feasible batch size B. Such high variance for gradient estimation makes it difficult for SGD to converge.

To address this issue, we use Gumbel Softmax proposed by BID9 ; BID15 to control the edge selection.

For a node pair (v i , v j ), instead of applying a "hard" sampling and execute only one edge, we use Gumbel Softmax to apply a "soft" sampling.

We compute m DISPLAYFORM1 g ij k is a random variable drawn from the Gumbel distribution.

Note that now m ij k becomes a continuous random variable.

It is directly differentiable with respect to θ ij k and we don't need to pass gradient through the random variable g ij k .

Therefore, the gradient of the loss function with respect to θ can be computed as DISPLAYFORM2 A temperature coefficient τ is used to control the behavior of the Gumbel Softmax.

As τ → ∞, m ij become continuous random variable following a uniform distribution.

Therefore, in equation FORMULA3 , all edges are executed and their outputs are averaged.

The gradient estimation in equation FORMULA7 are biased but the variance is low, which is favorable during the initial stage of the training.

As τ → 0, m ij gradually becomes a discrete random variable following the categorical distribution of P θ ij .

When computing equation (3), only one edge is sampled to be executed.

The gradient estimation then becomes unbiased but the variance is high.

This is favorable towards the end of the training.

In our experiment, we use an exponential decaying schedule to anneal the temperature as DISPLAYFORM3 where T 0 is the initial temperature when training begins.

We decay the temperature exponentially after every epoch.

Using the Gumbel Softmax trick effectively stabilizes the super net training.

In some sense, our work is in the middle ground of two previous works: ENAS by BID16 and DARTS by .

ENAS samples child networks from the super net to be trained independently while DARTS trains the entire super net together without decoupling child networks from the super net.

By using Gumbel Softmax with an annealing temperature, our DNAS pipeline behaves more like DARTS at the beginning of the search and behaves more like ENAS at the end.

Based on the analysis above, we propose a differentiable neural architecture search pipeline, summarized in Algorithm 1.

We first construct a stochastic super net G with architecture parameter θ and weight w. We train G with respect to w and θ separately and alternately.

Training the weight w optimizes all candidate edges (operators).

However, different edges can have different impact on the overall performance.

Therefore, we train the architecture parameter θ, to increase the probability to sample those edges with better performance, and to suppress those with worse performance.

To ensure generalization, we split the dataset for architecture search into X w , which is used specifically to train w, and X θ , which is used to train θ.

The idea is illustrated in FIG0 .In each epoch, we anneal the temperature τ for gumbel softmax with the schedule in equation FORMULA10 .

To ensure w is sufficiently trained before updating θ, we postpone the training of θ for N warmup epochs.

Through the training, we draw samples a ∼ P θ .

These sampled architectures are then trained on the training dataset X train and evaluated on the test set X test .Algorithm 1: The DNAS pipeline.

Input: Stochastic super net G = (V, E) with parameter θ and w, searching dataset X w and X θ , DISPLAYFORM0 Train G with respect to w for one epoch; Train a on X train to convergence;

Test a on X test ; 13 end Output: Trained architectures Q A .

We use the DNAS framework to solve the mixed precision quantization problem -deciding the optimal layer-wise precision assignment.

For a ConvNet, we first construct a super net that has the same "macro-structure" (number of layers, number of filters each layer, etc.) as the given network.

As shown in FIG3 .

Each node v i in the super net corresponds to the output tensor (feature map) of layer-i.

Each candidate edge e i,i+1 k represents a convolution operator whose weights or activation are quantized to a lower precision.

In order to encourage using lower-precision weights and activations, we define the loss function as L(a, w a ) = CrossEntropy(a) × C(Cost(a)).(10) Cost(a) denotes the cost of a candidate architecture and C(·) is a weighting function to balance the cross entropy term and the cost term.

To compress the model size, we define the cost as DISPLAYFORM0 where #PARAM(·) denotes the number of parameters of a convolution operator and weight-bit(·) denotes the bit-width of the weight.

m ij k is the edge selection mask described in equation (3).

Alternatively, to reduce the computational cost by jointly compressing both weights and activations, we use the cost function DISPLAYFORM1 where #FLOP(·) denotes the number of floating point operations of the convolution operator, weight-bit(·) denotes the bit-width of the weight and act-bit(·) denotes the bit-width of the activation.

Note that in a candidate architecture, m ij k have binary values {0, 1}. In the super net, we allow m ij k to be continuous so we can compute the expected cost of the super net..

To balance the cost term with the cross entropy term in equation FORMULA0 , we define C(Cost(a)) = β(log(Cost(a))) γ .(13) where β is a coefficient to adjust the initial value of C(Cost(a)) to be around 1.

γ is a coefficient to adjust the relative importance of the cost term vs. the cross-entropy term.

A larger γ leads to a stronger cost term in the loss function, which favors efficiency over accuracy.

In the first experiment, we focus on quantizing ResNet20, ResNet56, and ResNet110 BID6 ) on CIFAR10 BID12 ) dataset.

We start by focusing on reducing model size, since smaller models require less storage and communication cost, which is important for mobile and embedded devices.

We only perform quantization on weights and use full-precision activations.

We conduct mixed precision search at the block level -all layers in one block use the same precision.

Following the convention, we do not quantize the first and the last layer.

We construct a super net whose macro architecture is exactly the same as our target network.

For each block, we can choose a precision from {0, 1, 2, 3, 4, 8, 32}. If the precision is 0, we simply skip this block so the input and output are identical.

If the precision is 32, we use the full-precision floating point weights.

For all other precisions with k-bit, we quantize weights to k-bit fixed-point numbers.

See Appendix B for more experiment details.

Our experiment result is summarized in Table 1 .

For each quantized model, we report its accuracy and model size compression rate compared with 32-bit full precision models.

The model size is computed by equation FORMULA0 .

Among all the models we searched, we report the one with the highest test accuracy and the one with the highest compression rate.

We compare our method with Zhu et al. (2016) , where they use 2-bit (ternary) weights for all the layers of the network, except the first convolution and the last fully connect layer.

From the table, we have the following observations: 1) All of our most accurate models out-perform their full-precision counterparts by up to 0.37% while still achieves 11.6 -12.5X model size reduction.

2) Our most efficient models can achieve 16.6 -20.3X model size compression with accuracy drop less than 0.39%.

3) Compared with Zhu et al. (2016) , our model achieves up to 1.59% better accuracy.

This is partially due to our improved training recipe as our full-precision model's accuracy is also higher.

But it still demonstrates that our models with searched mixed precision assignment can very well preserve the accuracy.

Table 2 compares the precision assignment for the most accurate and the most efficient models for ResNet20.

Note that for the most efficient model, it directly skips the 3rd block in group-1.

In Fig. 3 , we plot the accuracy vs. compression rate of searched architectures of ResNet110.

We observe that models with random precision assignment (from epoch 0) have significantly worse compression while searched precision assignments generally have higher compression rate and accuracy.

TTQ (Zhu et al. FORMULA0 Table 2 : Layer-wise bit-widths for the most accurate vs. the most efficient architecture of ResNet20.g1b1 g1b2 g1b3 g2b1 g2b2 g2b3 g3b1 g3b2 g3b3 Most Accurate 4 4 3 3 3 4 4 3 1 Most Efficient 2 3 0 2 4 2 3 2 1

We quantize ResNet18 and ResNet34 on the ImageNet ILSVRC2012 (Deng et al. FORMULA1 dataset.

Different from the original ResNet BID6 ), we use the "ReLU-only preactivation" ResNet from BID7 .

Similar to the CIFAR10 experiments, we conduct mixed precision search at the block level.

We do not quanitze the first and the last layer.

See Appendix B for more details.

We conduct two sets of experiments.

In the first set, we aim at compressing the model size, so we only quantize weights and use the cost function from equation FORMULA0 .

Each block contains convolution operators with weights quantized to {1, 2, 4, 8, 32}-bit.

In the second set, we aim at compressing computational cost.

So we quantize both weights and activations and use the cost function from equation (12).

Each block in the super net contains convolution operators with weights and activations quantized to {(1, 4), (2, 4), (3, 3), (4, 4), (8, 8) , (32, 32)}-bit.

The first number in the tuple denotes the weight precision and the second denotes the activation precision.

The DNAS search is very efficient, taking less than 5 hours on 8 V100 GPUs to finish the search on ResNet18.Our model size compression experiment is reported in TAB2 .

We report two searched results for each model.

"MA" denotes the searched architecture with the highest accuracy, and "ME" denotes the most efficient.

We compare our results with TTQ (Zhu et al. FORMULA0 ) and ADMM BID13 ).

TTQ uses ternary weights (stored by 2 bits) to quantize a network.

For ADMM, we cite the result with {−4, 4} configuration where weights can have 7 values and are stored by 3 bits.

We report the accuracy and model size compression rate of each model.

From TAB2 , we have the following observations: 1) All of our most accurate models out-perform full-precision models by up to 0.5% while achieving 10.6-11.2X reduction of model size.

2) Our most efficient models can achieve 19.0 to 21.1X reduction of model size, still preserving competitive accuracy.

3) Compared with previous works, even our less accurate model has almost the same accuracy as the full-precision model with 21.1X smaller model size.

This is partially because we use label-refinery BID0 ) to effectively boost the accuracy of quantized models.

But it still demonstrate that our searched models can very well preserve the accuracy, despite its high compression rate.

Table 4 : Mixed Precision Quantization for ResNet on ImageNet for computational cost compression.

We abbreviate accuracy as "Acc" and compression rate as "Comp".

"arch-{1, 2, 3}" are three searched architectures ranked by accuracy.

Our experiment on computational cost compression is reported in Table 4 .

We report three searched architectures for each model.

We report the accuracy and the compression rate of the computational cost of each architecture.

We compute the computational cost of each model using equation FORMULA0 .

We compare our results with PACT ), DoReFA (Zhou et al. (2016) ), QIP BID10 ), and GroupNet (Zhuang et al. (2018) ).

The first three use 4-bit weights and activations.

We compute their compression rate as (32/4) × (32/4) = 64.

GroupNet uses binary weights and 2-bit activations, but its blocks contain 5 parallel branches.

We compute its compression rate as (32/1) × (32/2)/5 ≈ 102.4 The DoReFA result is cited from .

From table 4, we have the following observations: 1) Our most accurate architectures (arch-1) have almost the same accuracy (-0.02% or +0.09%) as the full-precision models with compression rates of 33.2x and 40.8X.

2) Comparing arch-2 with PACT, DoReFa, and QIP, we have a similar compression rate (62.9 vs 64), but the accuracy is 0.71-1.91% higher.

3) Comparing arch-3 with GroupNet, we have slightly higher compression rate (103.5 vs. 102.4), but 1.05% higher accuracy.

In this work we focus on the problem of mixed precision quantization of a ConvNet to determine its layer-wise bit-widths.

We formulate this problem as a neural architecture search (NAS) problem and propose a novel, efficient, and effective differentiable neural architecture search (DNAS) framework to solve it.

Under the DNAS framework, we efficiently explore the exponential search space of the NAS problem through gradient based optimization (SGD).

We use DNAS to search for layer-wise precision assignment for ResNet on CIFAR10 and ImageNet.

Our quantized models with 21.1x smaller model size or 103.9x smaller computational cost can still outperform baseline quantized or even full precision models.

DNAS is very efficient, taking less than 5 hours to finish a search on ResNet18 for ImageNet.

It is also a general architecture search framework that is not limited to the mixed precision quantization problem.

Its other applications will be discussed in future publications.

DISPLAYFORM0 w denotes the latent full-precision weight of a network.

Q k (·) denotes a k-bit quantization function that quantizes a continuous value w ∈ [0, 1] to its nearest neighbor in { DISPLAYFORM1 To quantize activations, we follow to use a bounded activation function followed by a quantization function as DISPLAYFORM2 Here, x is the full precision activation, y k is the quantized activation.

P ACT (·) is a function that bounds the output between [0, α].

α is a learnable upper bound of the activation function.

We discuss the experiment details for the CIFAR10 experiments.

CIFAR10 contains 50,000 training images and 10,000 testing images to be classified into 10 categories.

Image size is 32 × 32.

We report the accuracy on the test set.

To train the super net, we randomly split 80% of the CIFAR10 training set to train the weights w, and 20% to train the architecture parameter θ.

We train the super net for 90 epochs with a batch size of 512.

To train the model weights, we use SGD with momentum with an initial learning rate of 0.2, momentum of 0.9 and weight decay of 5 × 10 −4 .

We use the cosine decay schedule to reduce the learning rate.

For architecture parameters, we use Adam optimizer BID11 ) with a learning rate of 5 × 10 −3 and weight decay of 10 −3 .

We use the cost function from equation (11).

We set β from equation FORMULA0 to 0.1 and γ to 0.9.

To control Gumbel Softmax functions, we use an initial temperature of T 0 = 5.0, and we set the decaying factor η from equation (9) to be 0.025.

After every 10 epochs of training of super net, we sample 5 architectures from the distribution P θ .

We train each sampled architecture for 160 epochs and use cutout BID4 ) in data augmentation.

Other hyper parameters are the same as training the super net.

We next discuss the experiment details for ImageNet experiments.

ImageNet contains 1,000 classes, with roughly 1.3M training images and 50K validation images.

Images are scaled such that their shorter side is 256 pixels and are cropped to 224 × 224 before feeding into the network.

We report the accuracy on the validation set.

Training a super net on ImageNet can be very computationally expensive.

Instead, we randomly sample 40 categories from the ImageNet training set to train the super net.

We use SGD with momentum to train the super net weights for 60 epochs with a batch size of 256 for ResNet18 and 128 for ResNet34.

We set the initial learning rate to be 0.1 and reduce it with the cosine decay schedule.

We set the momentum to 0.9.

For architecture parameters, we use Adam optimizer with the a learning rate of 10 −3 and a weight decay of 5 × 10 −4 .

We set the cost coefficient β to 0.05, cost exponent γ to 1.2.

We set T 0 to be 5.0 and decay factor η to be 0.065.

We postpone the training of the architecture parameters by 10 epochs.

We sample 2 architectures from the architecture distribution P θ every 10 epochs.

The rest of the hyper parameters are the same as the CIFAR10 experiments.

We train sampled architectures for 120 epochs using SGD with an initial learning rate of 0.1 and cosine decay schedule.

We use label-refinery BID0 ) in training and we use the same data augmentation as this Pytorch example 1 .

<|TLDR|>

@highlight

A novel differentiable neural architecture search framework for mixed quantization of ConvNets.

@highlight

The authors introduce a new method for neural architecture search which selects the precision quantization of weights at each neural network layer, and use it in the context of network compression.

@highlight

The paper presents a new approach in network quantization by quantizing different layers with different bit-widths and introduces a new differentiable neural architecture search framework.