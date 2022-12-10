Neural network quantization has become an important research area due to its great impact on deployment of large models on resource constrained devices.

In order to train networks that can be effectively discretized without loss of performance, we introduce a differentiable quantization procedure.

Differentiability can be achieved by transforming continuous distributions over the weights and activations of the network to categorical distributions over the quantization grid.

These are subsequently relaxed to continuous surrogates that can allow for efficient gradient-based optimization.

We further show that stochastic rounding can be seen as a special case of the proposed approach and that under this formulation the quantization grid itself can also be optimized with gradient descent.

We experimentally validate the performance of our method on MNIST, CIFAR 10 and Imagenet classification.

Neural networks excel in a variety of large scale problems due to their highly flexible parametric nature.

However, deploying big models on resource constrained devices, such as mobile phones, drones or IoT devices is still challenging because they require a large amount of power, memory and computation.

Neural network compression is a means to tackle this issue and has therefore become an important research topic.

Neural network compression can be, roughly, divided into two not mutually exclusive categories: pruning and quantization.

While pruning BID18 BID10 aims to make the model "smaller" by altering the architecture, quantization aims to reduce the precision of the arithmetic operations in the network.

In this paper we focus on the latter.

Most network quantization methods either simulate or enforce discretization of the network during training, e.g. via rounding of the weights and activations.

Although seemingly straighforward, the discontinuity of the discretization makes the gradient-based optimization infeasible.

The reason is that there is no gradient of the loss with respect to the parameters.

A workaround to the discontinuity are the "pseudo-gradients" according to the straight-through estimator BID3 , which have been successfully used for training low-bit width architectures at e.g. BID13 ; Zhu et al. (2016) .The purpose of this work is to introduce a novel quantization procedure, Relaxed Quantization (RQ).

RQ can bypass the non-differentiability of the quantization operation during training by smoothing it appropriately.

The contributions of this paper are four-fold: First, we show how to make the set of quantization targets part of the training process such that we can optimize them with gradient descent.

Second, we introduce a way to discretize the network by converting distributions over the weights and activations to categorical distributions over the quantization grid.

Third, we show that we can obtain a "smooth" quantization procedure by replacing the categorical distributions with (a) (b)Figure 1: The proposed discretization process.

(a) Given a distribution p(x) over the real line we partition it into K intervals of width α where the center of each of the intervals is a grid point g i .

The shaded area corresponds to the probability ofx falling inside the interval containing that specific g i .(b) Categorical distribution over the grid obtained after discretization.

The probability of each of the grid points g i is equal to the probability ofx falling inside their respective intervals.concrete BID22 BID15 equivalents.

Finally we show that stochastic rounding BID8 , one of the most popular quantization techniques, can be seen as a special case of the proposed framework.

We present the details of our approach in Section 2, discuss related work in Section 3 and experimentally validate it in Section 4.

Finally we conclude and provide fruitful directions for future research in Section 5.

The central element for the discretization of weights and activations of a neural network is a quantizer q(·).

The quantizer receives a (usually) continous signal as input and discretizes it to a countable set of values.

This process is inherently lossy and non-invertible: given the output of the quantizer, it is impossible to determine the exact value of the input.

One of the simplest quantizers is the rounding function: DISPLAYFORM0 where α corresponds to the step size of the quantizer.

With α = 1, the quantizer rounds x to its nearest integer number.

Unfortunately, we cannot simply apply the rounding quantizer to discretize the weights and activations of a neural network.

Because of the quantizers' lossy and non-invertible nature, important information might be destroyed and lead to a decrease in accuracy.

To this end, it is preferable to train the neural network while simulating the effects of quantization during the training procedure.

This encourages the weights and activations to be robust to quantization and therefore decreases the performance gap between a full-precision neural network and its discretized version.

However, the aforementioned rounding process is non-differentiable.

As a result, we cannot directly optimize the discretized network with stochastic gradient descent, the workhorse of neural network optimization.

In this work, we posit a "smooth" quantizer as a possible way for enabling gradient based optimization.

The proposed quantizer comprises four elements: a vocabulary, its noise model and the resulting discretization procedure, as well as a final relaxation step to enable gradient based optimization.

The first element of the quantizer is the vocabulary: it is the set of (countable) output values that the quantizer can produce.

In our case, this vocabulary has an inherent structure, as it is a grid of ordered scalars.

For fixed point quantization the grid G is defined as DISPLAYFORM0 where b is the number of available bits that allow for K = 2 b possible integer values.

By construction this grid of values is agnostic to the input signal x and hence suboptimal; to allow for the grid to adapt to x we introduce two free parameters, a scale α and an offset β.

This leads to a learnable grid viaĜ = αG + β that can adapt to the range and location of the input signal.

The second element of the quantizer is the assumption about the input noise ; it determines how probable it is for a specific value of the input signal to move to each grid point.

Adding noise to x will result in a quantizer that is, on average, a smooth function of its input.

In essense, this is an application of variational optimization BID32 to the non-differentiable rounding function, which enables us to do gradient based optimization.

We model this form of noise as acting additively to the input signal x and being governed by a distribution p( ).

This process induces a distribution p(x) wherex = x + .

In the next step of the quantization procedure, we discretize p(x) according to the quantization gridĜ; this neccesitates the evaluation of the cumulative distribution function (CDF).

For this reason, we will assume that the noise is distributed according to a zero mean logistic distribution with a standard deviation σ, i.e. L(0, σ), hence leading to p(x) = L(x, σ).

The CDF of the logistic distribution is the sigmoid function which is easy to evaluate and backpropagate through.

Using Gaussian distributions proved to be less effective in preliminary experiments.

Other distributions are conceivable and we will briefly discuss the choice of a uniform distribution in Section 2.3.The third element is, given the aforementioned assumptions, how the quantizer determines an appropriate assignment for each realization of the input signal x. Due to the stochastic nature ofx, a deterministic round-to-nearest operation will result in a stochastic quantizer for x. Quantizing x in this manner corresponds to discretizing p(x) ontoĜ and then sampling grid points g i from it.

More specifically, we construct a categorical distribution over the grid by adopting intervals of width equal to α centered at each of the grid points.

The probability of selecting that particular grid point will now be equal to the probability ofx falling inside those intervals: DISPLAYFORM1 wherex corresponds to the quantized variable, P (·) corresponds to the CDF and the step from Equation 2 to Equation 3 is due to the logistic noise assumption.

A visualization of the aforementioned process can be seen in Figure 1 .

For the first and last grid point we will assume that they reside within DISPLAYFORM2 respectively.

Under this assumption we will have to truncate p(x) such that it only has support within (g 0 − α/2, g K + α/2].

Fortunately this is easy to do, as it corresponds to just a simple modification of the CDF: DISPLAYFORM3 Armed with this categorical distribution over the grid, the quantizer proceeds to assign a specific grid value tox by drawing a random sample.

This procedure emulates quantization noise, which prevents the model from fitting the data.

This noise can be reduced in two ways: by clustering the weights and activations around the points of the grid and by reducing the logistic noise σ.

As σ → 0, the CDF converges towards the step function, prohibiting gradient flow.

On the other hand, if is too high, the optimization procedure is very noisy, prohibiting convergence.

For this reason, during optimization we initialize σ in a sensible range, such that L(x, σ) covers a significant portion of the grid.

Please confer Appendix A for details.

We then let σ be freely optimized via gradient descent such that the loss is minimized.

Both effects reduce the gap between the function that the neural network computes during training time vs. test time.

We illustrate this in FIG0 .The fourth element of the procedure is the relaxation of the non-differentiable categorical distribution sampling.

While we can use an unbiased gradient estimator via REINFORCE (Williams, 1992), we opt for a continuous relaxation due to high variances with REINFORCE.

This is achieved by replacing the categorical distribution with a concrete distribution BID22 BID15 .

This relaxation procedure corresponds to adopting a "smooth" categorical distribution that Each color corresponds to an assignment to a particular grid point and the vertical dashed lines correspond to the grid points (β = 0).

We can clearly see that the real valued weights are naturally encouraged through training to cluster into multiple modes, one for each grid point.

It should also be mentioned, that for the right and leftmost grid points the probability of selecting them is maximized by moving the corresponding weight furthest right or left respectively.

Interestingly, we observe that the network converged to ternary weights for the input and (almost) binary weights for the output layer.can be seen as a "noisy" softmax.

Let π i be the categorical probability of sampling grid point i, i.e. DISPLAYFORM4 the "smoothed" quantized valuex can be obtained via: DISPLAYFORM5 where z i is the random sample from the concrete distribution and λ is a temperature parameter that controls the degree of approximation, since as λ → 0 the concrete distribution becomes a categorical.

We have thus defined a fully differentiable "soft" quantization procedure that allows for stochastic gradients for both the quantizer parameters α, β, σ as well as the input signal x (e.g. the weights or the activations of a neural network).

We refer to this algorithm as Relaxed Quantization (RQ).

We summarize its forward pass as performed during training in Algorithm 1.

It is also worthwhile to notice that if there were no noise at the input x then the categorical distribution would have non-zero mass only at a single value, thus prohibiting gradient based optimization for x and σ.

One drawback of this approach is that the smoothed quantized values defined in Equation 5 do not have to coincide with grid points, as z is not a one-hot vector.

Instead, these values can lie anywhere between the smallest and largest grid point, something which is impossible with e.g. stochastic rounding BID8 .

In order to make sure that only grid-points are sampled, we propose an alternative algorithm RQ ST in which we use the variant of the straight-through (ST) estimator proposed in BID15 .

Here we sample the actual categorical distribution during the forward pass but assume a sample from the concrete distribution for the backward pass.

While this gradient estimator is obviously biased, in practice it works as the "gradients" seem to point towards a valid direction.

This effect was also recently studied at BID38 .

We perform experiments with both variants.

After convergence, we can obtain a "hard" quantization procedure, i.e. select points from the grid, at test time by either reverting to a categorical distribution (instead of the continuous surrogate) or by rounding to the nearest grid point.

In this paper we chose the latter as it is more aligned with the low-resource environments in which quantized models will be deployed.

Furthermore, with this goal in mind, we employ two quantization grids with their own learnable scalar α, σ (and potentially β) parameters for each layer; one for the weights and one for the activations.

Samplingx based on drawing K random numbers for the concrete distribution as described in FORMULA6 can be very expensive for larger values of K. Firstly, drawing K random numbers Algorithm 1 Quantization during training.

Require: Input x, gridĜ, scale of the grid α, scale of noise σ, temperature λ, fuzz param.

DISPLAYFORM0 Algorithm 2 Quantization during testing.

Require: Input x, scale and offset of the grid α, β, minimum and maximum values g 0 , g DISPLAYFORM1 for every individual weight and activation in a neural network drastically increases the number of operations required in the forward pass.

Secondly, it also requires keeping many more numbers in memory for gradient computations during the backward pass.

Compared to a standard neural network or stochastic rounding approaches, the proposed procedure can thus be infeasible for larger models and datasets.

Fortunately, we can make samplingx independent of the grid size by assuming zero probability for grid-points that lie far away from the signal x. Specifically, by only considering grid points that are within δ standard deviations away from x, we truncate p(x) such that it lies within a "localized" grid around x. To simplify the computation required for determining the local grid elements, we choose the grid point closest to x, x , as the center of the local grid ( FIG1 ).

Since σ is shared between all elements of the weight matrix or activation, the local grid has the same width for every element.

The computation of the probabilities over the localized grid is similar to the truncation happening in Equation 4 and the smoothed quantized value is obtained via a manner similar to Equation 5: DISPLAYFORM2 2.3 RELATION TO STOCHASTIC ROUNDING One of the pioneering works in neural network quantization has been the work of BID8 ; it introduced stochastic rounding, a technique that is one of the most popular approaches for training neural networks with reduced numerical precision.

Instead of rounding to the nearest representable value, the stochastic rounding procedure selects one of the two closest grid points with probability depending on the distance of the high precision input from these grid points.

In fact, we can view stochastic rounding as a special case of RQ where DISPLAYFORM3 .

This uniform distribution centered at x of width equal to the grid width α generally has support only for the closest grid point.

Discretizing this distribution to a categorical over the quantization grid however assigns probabilities to the two closest grid points as in stochastic rounding, following Equation 2: DISPLAYFORM4 Stochastic rounding has proven to be a very powerful quantization scheme, even though it relies on biased gradient estimates for the rounding procedure.

On the one hand, RQ provides a way to circumvent this estimator at the cost of optimizing a surrogate objective.

On the other hand, RQ ST makes use of the unreasonably effective straight-through estimator as used in BID15 to avoid optimizing a surrogate objective, at the cost of biased gradients.

Compared to stochastic rounding, RQ ST further allows sampling of not only the two closest grid points, but also has support for more distant ones depending on the estimated input noise σ.

Intuitively, this allows for larger steps in the input space without first having to decrease variance at the traversion between grid sections.

In this work we focus on hardware oriented quantization approaches.

As opposed to methods that focus only on weight quantization and network compression for a reduced memory footprint, quantizing all operations within the network aims to additionally provide reduced execution times.

Within the body of work that considers quantizing weights and activations fall papers using stochastic rounding BID8 BID13 BID9 BID35 .

BID35 ) also consider quantized backpropagation, which is out-of-scope for this work.

Furthermore, another line of work considers binarizing BID6 Zhou et al., 2018) or ternarizing BID19 Zhou et al., 2018) weights and activations BID13 BID28 Zhou et al., 2016) via the straight-through gradient estimator BID3 ; these allow for fast implementations of convolutions using only bit-shift operations.

In a similar vein, the straight through estimator has also been used in Cai et al. FORMULA1 Another line of work quantizes networks through regularization.

BID20 ) formulate a variational approach that allows for heuristically determining the required bit-width precision for each weight of the model.

Improving upon this work, BID1 proposed a quantizing prior that encourages ternary weights during training.

Similarly to RQ, this method also allows for optimizing the scale of the ternary grid.

In contrast to RQ, this is only done implicitly via the regularization term.

One drawback of these approaches is that the strength of the regularization decays with the amount of training data, thus potentially reducing their effectiveness on large datasets.

Alternatively, one could directly regularize towards a set of specific values via the approach described at BID37 .Weights in a neural network are usually not distributed uniformly within a layer.

As a result, performing non-uniform quantization is usually more effective.

BID2 ) employ a stochastic quantizer by first uniformizing the weight or activation distribution through a non-linear transformation and then injecting uniform noise into this transformed space.

BID27 propose a version of their method in which the quantizer's code book is learned by gradient descent, resulting in a non-uniformly spaced grid.

Another line of works quantizes by clustering and therefore falls into this category; BID10 BID33 represent each of the weights by the centroid of its closest cluster.

While such non-uniform techniques can be indeed effective, they do not allow for efficient implementations on todays hardware.

Nevertheless, there is encouraging recent work on non-uniform grids that can be implemented with bit operations.

Within the liteterature on quantizing neural networks there are many approaches that are orthogonal to our work and could potentially be combined for additional improvements.

BID27 use knowledge distrillation techniques to good effect, whereas works such as modify the architecture to compensate for lower precision computations.

BID40 BID2 perform quantization in an step-by-step manner going from input layer to output, thus allowing the later layers to more easily adapt to the rounding errors

For the subsequent experiments RQ will correspond to the proposed procedure that has concrete sampling and RQ ST will correspond to the proposed procedure that uses the Gumbel-softmax straight-through estimator BID15 for the gradient.

We did not optimize an offset for the grids in order to be able to represent the number zero exactly, which allows for sparsity and is required for zero-padding.

Furthermore we assumed a grid that starts from zero when quantizing the outputs of ReLU.

We provide further details on the experimental settings at Appendix A. We will also provide results of our own implementation of stochastic rounding BID8 with the dynamic fixed point format BID9 ) (SR+DR).

Here we used the same hyperparameters as for RQ.

All experiments were implemented with TensorFlow BID0 , using the Keras library BID5 .

For the first task we considered the toy LeNet-5 network trained on MNIST with the 32C5 -MP2 -64C5 -MP2 -512FC -Softmax architecture and the VGG 2x(128C3) -MP2 -2x(256C3) -MP2 -2x(512C3) -MP2 -1024FC -Softmax architecture on the CIFAR 10 dataset.

Details about the hyperparameter settings can be found in Appendix A.By observing the results in TAB1 , we see that our method can achieve competitive results that improve upon several recent works on neural network quantization.

Considering that we achieve lower test error for 8 bit quantization than the high-precision models, we can see how RQ has a regularizing effect.

Generally speaking we found that the gradient variance for low bit-widths (i.e. 2-4 bits) in RQ needs to be kept in check through appropriate learning rates.

In order to demonstrate the effectiveness of our proposed approach on large scale tasks we considered the task of quantizing a Resnet-18 BID11 as well as a Mobilenet trained on the Imagenet (ILSVRC2012) dataset.

For the Resnet-18 experiment, we started from a pre-trained full precision model that was trained for 90 epochs.

We provide further details about the training procedure in Appendix C. The Mobilenet was initialized with the pretrained model available on the tensorflow github repository 1 .

We quantized the weights of all layers, post ReLU activations and average pooling layer for various bit-widths via fine-tuning for ten epochs.

Further details can be found in Appendix C.Some of the existing quantization works do not quantize the first (and sometimes) last layer.

Doing so simplifies the problem but it can, depending on the model and input dimensions, significantly increase the amount of computation required.

We therefore make use of the bit operations (BOPs) metric BID2 , which can be seen as a proxy for the execution speed on appropriate hardware.

In BOPs, the impact of not quantizing the first layer in, for example, the Resnet-18 model on Imagenet, becomes apparent: keeping the first layer in full precision requires roughly 1.3 times as many BOPs for one forward pass through the whole network compared to quantizing all weights and activations to 5 bits.

FIG3 compares a wide range of methods in terms of accuracy and BOPs.

We choose to compare only against methods that employ fixed-point quantization on Resnet-18 and Mobilenet, hence do not compare with non-uniform quantization techniques, such as the one described at BID2 .

In addition to our own implementation of BID8 with the dynamic fixed point format BID9 , we also report results of "rounding".

This corresponds to simply rounding the pre-trained high-precision model followed by re-estimation of the batchnorm statistics.

The grid BID8 BID9 4/4 0.66 -2/2 1.03 -Deep Comp.

BID10 (5-8)/32 0.74 -TWN BID19 2/32 0.65 a 7.44 BWN BID28 1/32 -9.88 XNOR-net BID28 1/1 -10.17 SWS BID33 3/32 0.97 -Bayesian Comp.

BID20 (7-18)/32 1.00 -VNQ BID1 2/32 0.73 -WAGE BID35 2/8 0.40 6.78LR Net BID29 in this case is defined as the initial grid used for fine-tuning with RQ.

For batchnorm re-estimation and grid initialization, please confer Appendix A.In FIG3 we observe that on ResNet-18 the RQ variants form the "Pareto frontier" in the trade-off between accuracy and efficiency, along with SYQ, Apprentice and Jacob et al. (2017) .

SYQ, however, employs "bucketing" and Apprentice uses distillation, both of which can be combined with RQ and improve performance.

does better than RQ with 8 bits, however RQ improved w.r.t.

to its pretrained model, whereas decreased slightly.

For experimental details with , please confer Appendix C.1.

SR+DR underperforms in this setting and is worse than simple rounding for 5 to 8 bits.

For Mobilenet, 4b shows that RQ is competitive to existing approaches.

Simple rounding resulted in almost random chance for all of the bit configurations.

SR+DR shows its strength for the 8 bit scenario, while in the lower bit regime, RQ outperforms competitive approaches.

We have introduced Relaxed Quantization (RQ), a powerful and versatile algorithm for learning low-bit neural networks using a uniform quantization scheme.

As such, the models trained by this method can be easily transferred and executed on low-bit fixed point chipsets.

We have extensively evaluated RQ on various image classification benchmarks and have shown that it allows for the better trade-offs between accuracy and bit operations per second.

Future hardware might enable us to cheaply do non-uniform quantization, for which this method can be easily extended.

BID17 BID25 for example, show the benefits of low-bit floating point weights that can be efficiently implemented in hardware.

The floating point quantization grid can be easily learned with RQ by redefiningĜ. General non-uniform quantization, as described TAB4 in the Appendix.

We compare against multiple works that employ fixed-point quantization: SR+DR BID8 BID9 , LR Net BID29 , , TWN BID19 , INQ BID40 , BWN BID28 , XNORnet BID28 , DoReFa (Zhou et al., 2016) , HWGQ BID4 , ELQ Zhou et al. (2018) , SYQ BID7 , Apprentice , QSM BID30 and rounding.for example in BID2 , is a natural extension to RQ, whose exploration we leave to future work.

For example, we could experiment with a base grid that is defined as in .

Currently, the bit-width of every quantizer is determined beforehand, but in future work we will explore learning the required bit precision within this framework.

In our experiments, batch normalization was implemented as a sequence of convolution, batch normalization and quantization.

On a low-precision chip, however, batch normalization would be "folded" into the kernel and bias of the convolution, the result of which is then rounded to low precision.

In order to accurately reflect this folding at test time, future work on the proposed algorithm will emulate folded batchnorm at training time and learn the corresponding quantization grid of the modified kernel and bias.

For fast model evaluation on low-precision hardware, quantization goes hand-in-hand with network pruning.

The proposed method is orthogonal to pruning methods such as, for example, L 0 regularization BID21 , which allows for group sparsity and pruning of hidden units.

The grid width α of each grid was initialized according to the bit-width b and the maximum and minimum values of the input x to the quantizer 2 .

Since the inputsx in both cases for our approach are stochastic it makes sense to assume a width for the grid that is slightly larger than the standard width t = (max(x) − min(x))/2 b ; for the activations, whenever b > 4, we initialize α = t + 3t/2 b , for 4 ≥ b > 2 we used α = t + 3t/2 b+1 and finally for b = 2 we used α = t. Since with ReLU activations the magnitude can become quite large (thus leading to increased quantization noise for smaller bit widths), this scheme keeps the noise injected to the network in check.

For the weights we always used an initial α = t + 3t/2 b .

The standard deviation of the logistic noise σ was initialized to be three times smaller than the width α, i.e. σ = α/3.

Under this specification, most of the probability mass of the logistic distribution is initially (roughly) in the bins containing the closest grid point and its' two neighbors.

The moving averages of layer statistics that are aggregated during the training phase for the batch normalization do not necessarily reflect the statistics of the quantized model accurately.

Even though RQ aims to minimize the gap between training and testing phase, we found that the aggregated statistics in combination with the learned scale and shift parameters of batch normalization lead to decreased test performance.

In order to avoid this drop in accuracy, we apply the insights from BID26 and recompute the statistics of the quantized model before reporting the final test error rate.

The final models were determined through early stopping using the validation loss computed with minibatch statistics, in case the model uses batch normalization.

For the MNIST experiment we rescaled the input to the [-1, 1] range, employed no regularization and the network was trained with Adam (Kingma & Ba, 2014 ) and a batch size of 128.

We used a local grid whenever the bit width was larger than 2 for both, weights and biases (shared grid parameters), as well as for the ouputs of the ReLU, with δ = 3.

For the 8 and 4 bit networks we used a temperature λ of 2 whereas for the 2 bit models we used a temperature of 1 for RQ.

We trained the 8 and 4 bit networks for 100 epochs using a learning rate of 1e-3 and the 2 bit networks for 200 epochs with a learning rate of 5e-4.

In all of the cases the learning rate was annealed to zero during the last 50 epochs.

For the CIFAR 10 experiment, the hyperparameters were chosen identically to the LeNet-5 experiments except a few differences.

We chose a learning rate ot 1e-4 instead of 1e-3 for 8 and 4 bit networks and trained for 300 epochs with a batch size of 100.

We also included a weight decay term of 1e-4 for the 8 bit networks.

For the 2 bit model we started with a learning rate of 1e-3.

The VGG model contains a batch normalization layer after every convolutional layer, but preceeded by max pooling, if present.

Training a neural network with RQ imposes an additional sampling burden for every weight and activation in the network.

Here, we investigate whether the extra "noise" that is introduced hampers the convergence speed of the network when we train from a random initialization.

We recorded the learning curves for a 2/2 bit RQ-VGG network on CIFAR 10 (as this quantization level exhibits the largest amount of noise) and compare it to the full precision baseline.

The results can be seen in FIG4 .

As we can observe, the 2/2 bit network has qualitatively similar trends to the full precision baseline.

Therefore we can conclude that the noise is not detrimental for the task at hand, at least for this particular model.

In terms of wall-clock time, training the RQ model with a full (4 elements) grid took approximately 15 times as long as the high-precision baseline with an implementation in Tensorflow v1.11.0 and running on a single Titan-X Nvidia GPU.

Each channel of the input images was preprocessed by subtracting the mean and dividing by the standard deviation of that channel across the training set.

We then resized the images such that the shorter side is set to 256 and then applied random 224x224 crops and random horizontal flips for data augmentation.

For evaluation we consider the center 224x224 crop of the images.

We trained the base Resnet-18 model with stochastic gradient descent, a batch size of 128, nesterov momentum of 0.9 and a learning rate of 0.1 which was multiplied by 0.1 at the 30th and 60th epoch.

We also applied weight decay with a strength of 1e-4.

For the quantized model fine-tuning phase, we used Adam with a learning rate of 5e −6 , a batch size of 24 and a momentum of 0.99.

We used a temperature of 2 for both RQ variants.

Following the strategy in , we did not quantize the biases.

TAB4 contains the error rates for Resnet-18 and Mobilenet on which Figure 1 is based on.

Algorithm and architecture specific changes are mentioned explicitly through footnotes.

We used the code provided at https://github.com/tensorflow/models/tree/ master/official/resnet and modified the construction of the training and evaluation graph by inserting quantization operations provided by the tensorflow.contrib.quantize package.

In a first step, the unmodified code was used to train a high-precision Resnet18 model using the hyper-parameter settings for the learning rate scheduling that are provided in the github repository.

More specifically, the model was trained for 90 epochs with a batch size of 128.

The learning rate scheduling involved a "warm up" period in which the learning rate was annealed from zero to 0.64 BID40 5/32 31.02 10.90 --BWN BID28 1/32 39.20 17.00 --XNOR-net BID28 1/1 48.80 26.80 --HWGQ BID4 over the first 50k steps, after which it was divided by 10 after epochs 30, 60 and 80 respectively.

Gradients were modified using a momentum of 0.9.

Final test performance under this procedure is 29.53% top-1 error and 10.44% top-5 error.

From the high-precision model checkpoint, the final quantized model was then fine-tuned for 10 epochs using a constant learning rate of 1e −4 and mo-

<|TLDR|>

@highlight

We introduce a technique that allows for gradient based training of quantized neural networks.

@highlight

Proposes a unified and general way of training neural networks with reduced precision quantized synaptic weights and activations.

@highlight

A new approach to quantizing activations which is state of the art or competitive on several real image problems.

@highlight

A method for learning neural networks with quantized weights and activations by stochastically quantizing values and replacing the resulting categotical distribution with a continuous relaxation