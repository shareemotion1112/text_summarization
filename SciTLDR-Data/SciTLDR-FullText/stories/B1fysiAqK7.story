Low bit-width weights and activations are an effective way of combating the increasing need for both memory and compute power of Deep Neural Networks.

In this work, we present a probabilistic training method for Neural Network with both binary weights and activations, called PBNet.

By embracing stochasticity during training, we circumvent the need to approximate the gradient of functions for which the derivative is zero almost always, such as $\textrm{sign}(\cdot)$, while still obtaining a fully Binary Neural Network at test time.

Moreover, it allows for anytime ensemble predictions for improved performance and uncertainty estimates by sampling from the weight distribution.

Since all operations in a layer of the PBNet operate on random variables, we introduce stochastic versions of Batch Normalization and max pooling, which transfer well to a deterministic network at test time.

We evaluate two related training methods for the PBNet: one in which activation distributions are propagated throughout the network, and one in which binary activations are sampled in each layer.

Our experiments indicate that sampling the binary activations is an important element for stochastic training of binary Neural Networks.

Deep Neural Networks are notorious for having vast memory and computation requirements, both during training and test/prediction time.

As such, Deep Neural Networks may be unfeasible in various environments such as battery powered devices, embedded devices (because of memory requirement), on body devices (due to heat dissipation), or environments in which constrains may be imposed by a limited economical budget.

Hence, there is a clear need for Neural Networks that can operate in these resource limited environments.

One method for reducing the memory and computational requirements for Neural Networks is to reduce the bit-width of the parameters and activations of the Neural Network.

This can be achieved either during training (e.g., BID15 ; BID0 ) or using post-training mechanisms (e.g., BID15 , BID5 ).

By taking the reduction of the bit-width for weights and activations to the extreme, i.e., a single bit, one obtains a Binary Neural Network.

Binary Neural Networks have several advantageous properties, i.e., a 32?? reduction in memory requirements and the forward pass can be implemented using XNOR operations and bit-counting, which results in a 58?? speedup on CPU BID20 .

Moreover, Binary Neural Networks are more robust to adversarial examples BID2 .

BID21 introduced a probabilistic training method for Neural Networks with binary weights, but allow for full precision activations.

In this paper, we propose a probabilistic training method for Neural Networks with both binary weights and binary activations, which are even more memory and computation efficient.

In short, obtain a closed form forward pass for probabilistic neural networks if we constrain the input and weights to binary (random) variables.

The output of the Multiply and Accumulate (MAC) operations, or pre-activation, is approximated using a factorized Normal distribution.

Subsequently, we introduce stochastic versions of Max-Pooling and Batch Normalization that allow us to propagate the pre-activatoins throughout a single layer.

By applying the sign(??) activation function to the random pre-activation, we not only obtain a distribution over binary activations, it also allows for backpropagation through the sign(??) operation.

This is especially convenient as this in a deterministic Neural Network all gradient information is zeroed out when using sign as activation.

We explore two different methods for training this probabilistic binary neural network: In the first method the activation distribution of layer l is propagated to layer (l + 1), which means the MAC operation is performed on two binary random variables.

In the second method the binary activation is sampled as the last operation in a layer using the concrete relaxation BID16 .

This can be thought of as a form of local reparametrization BID11 .

We call the networks obtained using these methods PBNet and PBNet-S, respectively.

At test time, we obtain a single deterministic Binary Neural Network, an ensemble of Binary Neural Networks by sampling from the parameter distribution, or a Ternary Neural Network based on the Binary weight distribution.

An advantage of our method is that we can take samples from the parameter distribution indefinitely-without retraining.

Hence, this method allows for anytime ensemble predictions and uncertainty estimates.

Note that while in this work we only consider the binary case, our method supports any discrete distribution over weights and activations.

Algorithm 1: Pseudo code for forward pass of single layer in PBNet(-S).

a l???1 denotes the activation of the previous layer, B the random binary weight matrix, ?? is the temperature used for the concrete distribution, f (??, ??) the linear transformation used in the layer, > 0 a small constant for numerical stability, D the dimensionality of the inner product in f , and ?? & ?? are the parameters for batch normalization.

DISPLAYFORM0 // Max pooling if max pooling required then n ??? N (0, I); s = ?? + ?? n; ?? = max-pooling-indices(s); ??, ?? 2 = select-at-indices(??, ?? 2 , ??); end // Binarization and sampling DISPLAYFORM1 In this section we introduce the probabilistic setting of the PBNet.

Moreover, the approximation of the distribution on the pre-activations is introduced.

For an explanation of the other operations in the PBNet, see Section 2.1 for the activation, Section 2.1.1 for the sampling of activations, and Section 2.2 for Pooling and Normalization.

We aim to train a probabilistic Binary Neural Network.

As such, we pose a binary distribution over the weights of the network and optimize the parameters of this distribution instead of the parameters directly.

This way, we obtain a distribution over parameters, but also deal with the inherent discreteness of a Binary Neural Network.

Given an objective function L(??), this approach can be thought of in terms of the variational optimization framework BID23 .

Specifically, by optimizing the parameters of the weight distributions, we optimize a bound on the actual loss: min DISPLAYFORM2 where B are the binary weights of the network and q ?? (B) is a distribution over the binary weights.

For q ?? (B) a slight reparametrization of the Bernoulli distribution is used, which we will refer to as the Binary distribution.

This distribution is parameterized by ?? ??? [???1, 1] and is defined by: DISPLAYFORM3 For the properties of this distribution, please refer to Appendix A.We will now consider using the Binary distribution for both the weights and the activations in a Neural Network.

Since the pre-activations in a Neural Network are computed using MAC operations, which are the same for each value in the pre-activation, we will only consider a single value in our discussion here.

Let w ??? Binary(??) and h ??? Binary(??) be the weight and input random variable for a given layer.

As such, the innerproduct between the weights and input is distributed according to a translated and scaled Poisson binomial distribution: DISPLAYFORM4 Where D is the dimensionality of h and w and denotes element-wise multiplication.

See the picket fence on the top in FIG0 for an illustration of the PMF of a Poisson binomial distribution.

Although the scaled and translated Poisson binomial distribution is the exact solution for the inner product between the weight and activation random variables, it is hard to work with in subsequent layers.

For this reason, and the fact that the Poisson binomial distribution is well approximated by a Normal distribution (Wang & Manning, 2013) , we use a Normal approximation to the Poisson binomial distribution, which allows for easier manipulations.

Using the properties of the Binary distribution and the Poisson binomial distribution, the approximation for the pre-activation a is given by: DISPLAYFORM5 Note that, this is exactly the approximation one would obtain by using the Lyapunov Central Limit Theorem (CLT), which was used by BID21 .

This allows us to obtain a close approximation to the pre-activation distribution, which we can propagate through the layer and/or network.

So far, only the MAC operation in a given layer is discussed, in Section 2.1 application of the binary activation is discussed and in Section 2.1.

The stochastic versions of Batch Normalization and Max Pooling are introduced in Section 2.2.

For specifics on sampling the binary activation, see Section 2.1.1.

The full forward pass for a single layer is given in detail in Algorithms 1.

Since the output of a linear operation using binary inputs is not restricted to be binary, it is required to apply a binarization operation to the pre-activation in order to obtain binary activations.

Various works -e.g., BID7 and BID20 -use either deterministic or stochastic binarization functions, i.e., DISPLAYFORM0 +1 with probability p = sigmoid(a) ???1 with probability 1 ??? p .In our case the pre-activations are random variables.

Hence, applying a deterministic binarization function to a random pre-activations results in a stochastic binary activation.

Specifically, let a i ??? N (?? i , ?? 2 i ) be a random pre-ctivation obtained using the normal approximation, as introduced in the previous section, then the activation (after binarization) is again given as a Binary random variable".

Interestingly, the Binary probability can be computed in closed form by evaluating the probability density that lies above the binarization threshold: DISPLAYFORM1 where ??(??|??, ?? 2 ) denotes the CDF of N (??, ?? 2 ).

Applying the binarization function to a random pre-activation has two advantages.

First, the derivatives ???q i /????? i and ???q i /????? i are not zero almost everywhere, in contrast to the derivatives of b det and b stoch when applied to a deterministic input.

Second, the distribution over h i reflects the true uncertainty about the sign of the activation, given the stochastic weights, whereas b stoch uses the magnitude of the pre-activation as a substitute.

For example, a pre-activation with a high positive magnitude and high variance will be deterministically mapped to 1 by b stoch .

In contrast, our method takes the variance into account and correctly assigns some probability mass to ???1.

See FIG0 for a graphical depiction of the stochastic binary activation.

So far, we have discussed propagating distributions throughout the network.

Alternatively, the binary activations can be sampled using the Concrete distribution BID16 during training.

specifically, we use the hard sample method as discussed by BID9 .

By sampling the activations, the input for subsequent layers will match the input that is observed at test time more closely.

As a consequence of sampling the activation, the input to a layer is no longer a distribution but a h ??? {???1, +1} D vector instead.

As such, the normal approximation to the pre-activation is computed slightly different.

From the Lyapunov CLT it follows that the approximation to the distribution of the pre-activation is given by: DISPLAYFORM0 where w ??? Binary(??) is a random weight.

Similarly, the pre-activation of the input layer is also computed using this approximation-given a real-valued input vector.

We will refer to a PBNet that uses activation sampling as PBNet-S.

Other than a linear operation and an (non-linear) activation function, Batch Normalization BID8 and pooling are two popular building blocks for Convolutional Neural Networks.

For Binary Neural Networks, applying Batch Normalization to a binarized activation will result in a non-binary result.

Moreover, the application of max pooling on a binary activation will result in a feature map containing mostly +1s.

Hence, both operations must be applied before binarization.

However, in the PBNet, the binarization operation is applied before sampling.

As a consequence, the Batch Normalization and pooling operations can only be applied on random pre-activations.

For this reason, we define these methods for random variables.

Although there are various ways to define these operation in a stochastic fashion, our guiding principle is to only leverage stochasticity during training, i.e., at test time, the stochastic operations are replaced by their conventional implementations and parameters learned in the stochastic setting must be transferred to their deterministic counterparts.

Batch Normalization (BN) BID8 -including an affine transformation -is defined as follows: DISPLAYFORM0 where a i denotes the pre-activation before BN,?? the pre-activation after BN, and m & v denote the sample mean and variance of DISPLAYFORM1 , for an M -dimensional pre-activation.

In essence, BN translates and scales the pre-activations such that they have approximately zero mean and unit variance, followed by an affine transformation.

Hence, in the stochastic case, our aim is that samples from the pre-activation distribution after BN also have approximately zero mean and unit variance-to ensure that the stochastic batch normalization can be transfered to a deterministic binary neural network.

This is achieved by subtracting the population mean from each pre-activation random variable and by dividing by the population variance.

However, since a i is a random variable in the PBNet, simply using the population mean and variance equations will result in non-standardized output.

Instead, to ensure a standardized distribution over activations, we compute the expected population mean and variance under the pre-activation distribution: DISPLAYFORM2 where M is the total number of activations and a i ??? N (?? i , ?? i ) are the random pre-activations.

By substituting m and v in Equation 8 by Equation 9 and 10, we obtain the following batch normalized Gaussian distributions for the pre-activations: DISPLAYFORM3 Note that this assumes a single channel, but is easily extended to 2d batch norm in a similar fashion as conventional Batch Normalization.

At test time, Batch Normalization in a Binary Neural Network can be reduced to an addition and sign flip of the activation, see Appendix B for more details.

In general, pooling applies an aggregation operation to a set of (spatially oriented) pre-activations.

Here we discuss max pooling for stochastic pre-activations, however, similar considerations apply for other types of aggregation functions.

In the case of max-pooling, given a spatial region containing stochastic pre-activations a 1 , . . .

, a K , we aim to stochastically select one of the a i .

Note that, although the distribution of max(a 1 , . . .

, a K ) is well-defined BID18 , its distribution is not Gaussian and thus does not match one of the input distributions.

Instead, we sample one of the input random variables in every spatial region according to the probability of that variable being greater than all other variables, i.e., ?? i = p(a i > z \i ), where z \i = max({a j } j =i ).

?? i could be obtained by evaluating the CDF of (z \i ??? a i ) at 0, but to our knowledge this has no analytical form.

Alternatively, we can use Monte-Carlo integration to obtain ??: DISPLAYFORM0 where one-hot(i) returns a K-dimensional one-hot vector with the ith elements set to one.

The pooling index ?? is then sampled from Cat(??).

However, more efficiently, we can sample s ??? p(a 1 , . . . , a K ) and select the index of the maximum in s, which is equivalent sampling from Cat(??).

Hence, for a given max pooling region, it is sufficient to obtain a single sample from each normal distribution associated with each pre-activation and keep the random variable for which this sample is maximum.

A graphical overview of this is given in Figure 2 .Other forms of stochastic or probabilistic max pooling were introduced by BID13 Zeiler & Fergus (2013) , however, in both cases a single activation is sampled based on the magnitude of the activations.

In contrast, in our procedure we stochastically propagate one of the input distributions over activations.

For the PBNet the parameters ?? for q ?? (B) are initialized from a uniform U (???1, 1) distribution.

Although the final parameter distribution more closely follows a Beta(??, ??) distribution, for ?? < 1, we did not observe any significant impact choosing another initialization method for the PBNet.

In the case of the PBNet-S, we observed a significant improvement in training speed and performance by initializing the parameters based on the parameters of a pre-trained full precission Neural Network.

This initializes the convolutional filters with more structure than a random initialization.

This is desirable as in order to flip the value of a weight, the parameter governing the weight has to pass through a high variance regime, which can slow down convergence considerably.

Select maximum per region 2 Sample from input distributions 1Keep maximum distribution for each region 3Figure 2: Max pooling for random variables is performed by taking a single sample from each of the input distributions.

The output random variable for each pooling region is the random variable that is associated with the maximum sample.

For the PBNet-S, We use the weight transfer method introduced by BID21 in which the parameters of the weight distribution for each layer are initialized such that the expected value of the random weights equals the full precision weight divided by the standard deviation of the weights in the given layer.

Since not all rescaled weights lay in the [???1, 1] range, all binary weight parameters are clipped between [???0.9, 0.9].

This transfer method transfers the structure present in the filters of the full precision network and ensures that a significant part of the parameter distributions is initialized with low variance.

In our training procedure, a stochastic neural network is trained.

However, at test time (or on hardware) we want to leverage all the advantages of a full binary Neural Network.

Therefore, we obtain a deterministic binary Neural Network from the parameter distribution q ?? (B) at test time.

We consider three approaches for obtaining a deterministic network: a deterministic network based on the mode of q ?? (B) called PBNET-MAP, an ensemble of binary Neural Networks sampled from q ?? (B) named PBNET-x, and a ternary Neural Network (PBNET-TERNARY), in which a single parameter W i may be set to zero based on q ?? , i.e.: DISPLAYFORM0 The ternary network can also be viewed as a sparse PBNet, however, sparse memory look-ups may slow down inference.

Note that, even when using multiple binary neural networks in an ensemble, the ensemble is still more efficient in terms of computation and memory when compared to a full precision alternative.

Moreover, it allows for anytime ensemble predictions for improved performance and uncertainty estimates by sampling from the weight distribution.

Since the trained weight distribution is not fully deterministic, the sampling of individual weight instantiations will result in a shift of the batch statistics.

As a consequence, the learned batch norm statistics no longer closely match the true statistics.

This is alleviated by re-estimating the batch norm statistics based on (a subset of) the training set after weight sampling using a moving mean and variance estimator.

We observed competitive results using as little as 20 batches from the training set.

Binary and low precision neural networks have received significant interest in recent years.

Most similar to our work, in terms of the final neural network, is the work on Binarized Neural Networks by BID7 .

in this work a real-valued shadow weight is used and binary weights are obtained by binarizing the shadow weights.

Similarly the pre-activations are binarized using the same binarization function.

In order to back-propagate through the binarization operation the straightthrough estimator BID6 ) is used.

Several extensions to Binarized Neural Networks have been proposed which -more or less -qualify as binary neural networks: XNOR-net BID20 in which the real-valued parameter tensor and activation tensor is approximated by a binary tensor and a scaling factor per channel.

ABC-nets take this approach one step further and approximate the weight tensor by a linear combination of binary tensors.

Both of these approaches perform the linear operations in the forward pass using binary weights and/or binary activations, followed by a scaling or linear combination of the pre-activations.

In McDonnell (2018), similar methods to BID7 are used to binarize a wide resnet (Zagoruyko & Komodakis, 2016 ) to obtain results on ImageNet very close to the full precision performance.

Another method for training binary neural networks is Expectation Backpropagation BID22 in which the central limit theorem and online expectation propagation is used to find an approximate posterior.

This method is similar in spirit to ours, but the training method is completely different.

Most related to our work is the work by BID21 which use the local reparametrization trick to train a Neural Network with binary weights and the work by BID1 which also discuss a binary Neural Network in which the activation distribution are propagated through the network.

Moreover, in (Wang & Manning, 2013 ) the CLT was used to approximate dropout noise during training in order to speed up training, however, there is no aim to learn binary (or discrete) weights or use binary activations in this work.

We evaluate the PBNet on the MNIST and CIFAR-10 benchmarks and compare the results to Binarized Neural Networks BID7 , since the architectures of the deterministic networks obtained by training the PBNet are equivalent.

The PBNets are trained using either a cross-entropy (CE) loss or a binary cross entropy for each class (BCE).

For the CE loss there is no binarization step in the final layer, instead the mean of the Gaussian approximation is used as the input to a softmax layer.

For BCE, there is a binarization step, and we treat the probability of the ith output being +1 as the probability of the input belonging to the ith class.

Specifically, for an output vector p ??? [0, 1] C for C classes and the true class y, the BCE loss for a single sample is defined as DISPLAYFORM0 The weights for the PBNet-S networks are initialized using the transfer method described in Section 2.3 and the PBNets are initialized using a uniform initialization scheme.

All models are optimized using Adam (Kingma & Ba, 2014) and a validation loss plateau learning rate decay scheme.

We keep the temperature for the binary concrete distribution static at 1.0 during training.

For all settings, we optimize model parameters until convergence, after which the best model is selected based on a validation set.

Our code is implemented using PyTorch BID19 .For Binarized Neural Networks we use the training procedure described by BID7 , i.e., a squared hinge loss and layer specific learning rates that are determined based on the Glorot initialization method BID4 .Experimental details specific to datasets are given in Appendix C and the results are presented in TAB0 .

We report both test set accuracy obtained after binarizing the network as well as the the test set accuracy obtained by the stochastic network during training (i.e., by propagating activation distributions).

As presented in TAB0 the accuracy improves when using an ensemble.

Moreover, the predictions of the ensemble members can be used to obtain an estimate of the certainty of the ensemble as a whole.

BID7 , PBNet, and a full precission network (FPNet).

PBNet-map refers to a deterministic PBNet using the map estimate, PBNet-Ternary is a ternary deterministic network obtained from q ?? , and PBNet-X refers to an ensemble of X networks, each sampled from the same weight distribution.

For the ensemble results both mean and standard deviation are presented.

The propagate column contains results obtained using the stochastic network whereas results in the binarized column are obtained using a deterministic binary Neural Network.

To evaluate this, we plot an error-coverage curve BID3 in FIG2 .

This curve is obtained by sorting the samples according to a statistic and computing the error percentage in the top x% of the samples -according to the statistic.

For the Binarized Neural Network and PBNet-MAP the highest softmax score is used, whereas for the ensembles the variance in the prediction of the top class is used.

The figure suggests that the ensemble variance is a better estimator of network certainty, and moreover, the estimation improves as the ensemble sizes increases.

As discussed in Section 2.4, after sampling the parameters of a deterministic network the batch statistics used by Batch Normalization must be re-estimated.

FIG2 shows the results obtained using a various number of batches from the training set to re-estimate the statistics.

This shows that even a small number of samples is sufficient to estimate the statistics.

We perform an ablation study on both the use of (stochastic) Batch Normalization and the use of weight transfer for the PBNet-S on CIFAR-10.

For Batch Normalization, we removed all batch normalization layers from the PBNet-S and retrained the model on CIFAR-10.

This resulted in a test set accuracy of 79.21%.

For the weight initialization experiment, the PBNet-S weights are initialized using a uniform initialization scheme and is trained on CIFAR-10, resulting in a test set accuracy of 83.61%.

Moreover, the accuracy on the validation set during training is presented in FIG2 .

Note that these numbers are obtained without sampling a binarized network from the weight distribution, i.e., local reparametrization and binary activation samples are used.

The PBNet-S that uses both weight transfer and stochastic Batch Normalization results in a significant performance improvement, indicating that both stochastic Batch Normalization and weight transfer are necessary components for the PBNet-S.

The results of our experiments show that, following our training procedure, sampling of the binary activations is a necessary component.

Although the stochastic PBNet generalizes well to unseen data, there is a significant drop in test accuracy when a binary Neural Network is obtained from the stochastic PBNet.

In contrast, this performance drop is not observed for PBNet-S. A potential explanation of this phenomenon is that by sampling the binary activation during training, the network is forced to become more robust to the inherent binarization noise that is present at test time of the binarized Neural Network.

If this is the case, then sampling the binary activation can be thought of as a regularization strategy that prepares the network for a more noisy binary setting.

However, other regularization strategies may also exist.

We have presented a stochastic method for training Binary Neural Networks.

The method is evaluated on multiple standardized benchmarks and reached competitive results.

The PBNet has various advantageous properties as a result of the training method.

The weight distribution allows one to generate ensembles online which results in improved accuracy and better uncertainty estimations.

Moreover, the Bayesian formulation of the PBNet allows for further pruning of the network, which we leave as future work.

A BINARY DISTRIBUTION For convenience, we have introduced the Binary distribution in this paper.

In this appendix we list some of the properties used in the paper, which all follow direcly from the properties of the Bernoulli distribution.

The Binary distribution is a reparametrization of the Bernoulli distribution such that: DISPLAYFORM0 This gives the following probability mass function: DISPLAYFORM1 where a ??? {???1, +1} and ?? ??? [???1, 1].

From this, the mean and variance are easily computed: DISPLAYFORM2 Finally, let b ??? Binary(??), then ab ??? Binary(????).

During training the PBNet is trained using stochastic Batch Normalization.

At test time, the parameters learned using stochastic Batch Normalization can be transferred to a conventional Batch Normalization implementation.

Alternatively, Batch Normalization can be reduced to an (integer) addition and multiplication by ??1 after applying the sign activation function.

Given a pre-activation a, the application of Batch Normalization followed by a sign binarization function can be rewritten as: DISPLAYFORM0 DISPLAYFORM1 when a ??? Z, which is the case for all but the first layer Note that we have used sign(0) = b det (0) = +1 here, as we have used everywhere in order to use sign as a binarization function.

The MNIST dataset consists of of 60K training and 10K test 28??28 grayscale handwritten digit images, divided over 10 classes.

The images are pre-processed by subtracting the global pixel mean and dividing by the global pixel standard deviation.

No other form of pre-processing or data augmentation is used.

For MNIST, we use the following architecture: DISPLAYFORM0 where XC3 denotes a binary convolutional layer using 3 ?? 3 filters and X output channels, Y FC denotes a fully connected layer with Y output neurons, SM10 denotes a softmax layer with 10 outputs, and MP2 denotes 2 ?? 2 (stochastic) max pooling with stride 2.

Note that if a convolutional layer is followed by a max pooling layer, the binarization is only performed after max pooling.

All layers are followed by (stochastic) batch normalization and binarization of the activations.

We use a batchsize of 128 and an initial learning rate of 10 ???2 Results are reported in TAB0 .

The CIFAR-10 ( BID12 ) dataset consists of 50K training and 10K test 32 ?? 32 RGB images divided over 10 classes.

The last 5,000 images from the training set are used as validation set.

Tthe images are only pre-processed by subtracting the channel-wise mean and dividing by the standard deviation.

We use the following architecture for our CIFAR-10 experiment (following BID21 ): DISPLAYFORM0 where we use the same notation as in the previous section.

The Binarized Neural Network baseline uses the same architecture, except for one extra 1024 neuron fully connected layer.

During training, the training set is augmented using random 0px to 4px translations and random horizontal fl Results are reported in TAB0 .

@highlight

We introduce a stochastic training method for training Binary Neural Network with both binary weights and activations.