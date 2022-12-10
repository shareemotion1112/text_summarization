We describe a simple and general neural network weight compression approach, in which the network parameters (weights and biases) are represented in a “latent” space, amounting to a reparameterization.

This space is equipped with a learned probability model, which is used to impose an entropy penalty on the parameter representation during training, and to compress the representation using a simple arithmetic coder after training.

Classification accuracy and model compressibility is maximized jointly, with the bitrate--accuracy trade-off specified by a hyperparameter.

We evaluate the method on the MNIST, CIFAR-10 and ImageNet classification benchmarks using six distinct model architectures.

Our results show that state-of-the-art model compression can be achieved in a scalable and general way without requiring complex procedures such as multi-stage training.

Artificial neural networks (ANNs) have proven to be highly successful on a variety of tasks, and as a result, there is an increasing interest in their practical deployment.

However, ANN parameters tend to require a large amount of space compared to manually designed algorithms.

This can be problematic, for instance, when deploying models onto devices over the air, where the bottleneck is often network speed, or onto devices holding many stored models, with only few used at a time.

To make these models more practical, several authors have proposed to compress model parameters (Han et al., 2015; Louizos, Ullrich, et al., 2017; Molchanov et al., 2017; Havasi et al., 2018) .

While other desiderata often exist, such as minimizing the number of layers or filters of the network, we focus here simply on model compression algorithms that 1. minimize compressed size while maintaining an acceptable classification accuracy, 2. are conceptually simple and easy to implement, and 3. can be scaled easily to large models.

Classic data compression in a Shannon sense (Shannon, 1948) requires discrete-valued data (i.e., the data can only take on a countable number of states) and a probability model on that data known to both sender and receiver.

Practical compression algorithms are often lossy, and consist of two steps.

First, the data is subjected to (re-)quantization.

Then, a Shannon-style entropy coding method such as arithmetic coding (Rissanen and Langdon, 1981 ) is applied to the discrete values, bringing them into a binary representation which can be easily stored or transmitted.

Shannon's source coding theorem establishes the entropy of the discrete representation as a lower bound on the average length of this binary sequence (the bit rate), and arithmetic coding achieves this bound asymptotically.

Thus, entropy is an excellent proxy for the expected model size.

The type of quantization scheme affects both the fidelity of the representation (in this case, the precision of the model parameters, which in turn affects the prediction accuracy) as well as the bit rate, since a reduced number of states coincides with reduced entropy.

ANN parameters are typically represented as floating point numbers.

While these technically have a finite (but large) number of states, the best results in terms of both accuracy and bit rate are typically achieved for a significantly reduced number of states.

Existing approaches to model compression often acknowledge this by quantizing each individual linear filter coefficient in an ANN to a small number of pre-determined values (Louizos, Reisser, et al., 2018; Baskin et al., 2018; F. Li et al., 2016) .

This is known as scalar quantization (SQ).

Other methods explore vector quantization (VQ), which is closely related to k-means clustering, in which each vector of filter coefficients is quantized jointly (Chen, J. Wilson, et al., 2015; Ullrich et al., 2017) .

This is equivalent to enumerating a finite set of representers Figure 1 : Visualization of representers in scalar quantization vs. reparameterized quantization.

The axes represent two different model parameters (e.g., linear filter coefficients).

Small black dots are samples of the model parameters, red and blue discs are the representers.

Left: in scalar quantization, the representers must be given by a Kronecker product of scalar representers along the cardinal axes, even though the distribution of samples may be skewed.

Right: in reparameterized scalar quantization, the representers are still given by a Kronecker product, but in a transformed (here, rotated) space.

This allows a better adaptation of the representers to the parameter distribution.

(representable vectors), while in SQ the set of representers is given by the Kronecker product of representable scalar elements.

VQ is much more general than SQ, in the sense that representers can be placed arbitrarily: if the set of useful filter vectors all live in a subset of the entire space, there is no benefit in having representers outside of that subset, which may be unavoidable with SQ (Figure 1 , left).

Thus, VQ has the potential to yield better results, but it also suffers from the "curse of dimensionality": the number of necessary states grows exponentially with the number of dimensions, making it computationally infeasible to perform VQ for much more than a handful of dimensions.

One of the key insights leading to this paper is that the strengths of SQ and VQ can be combined by representing the data in a "latent" space.

This space can be an arbitrary rescaling, rotation, or otherwise warping of the original data space.

SQ in this space, while making quantization computationally feasible, can provide substantially more flexibility in the choice of representers compared to the SQ in the data space (Figure 1, right) .

This is in analogy to recent image compression methods based on autoencoders (Ballé, Laparra, et al., 2016; Theis et al., 2017) .

The contribution of this paper is two-fold.

First, we propose a novel end-to-end trainable model compression method that uses scalar quantization and entropy penalization in a reparameterized space of model parameters.

The reparameterization allows us to use efficient SQ, while achieving flexibility in representing the model parameters.

Second, we provide state-of-the-art results on a variety of network architectures on several datasets.

This demonstrates that more complicated strategies involving pretraining, multi-stage training, sparsification, adaptive coding, etc., as employed by many previous methods, are not necessary to achieve good performance.

Our method scales to modern large image datasets and neural network architectures such as ResNet-50 on ImageNet.

We consider the classification setup, where we are given a dataset D = {(x 1 , y 1 ), ...(x N , y N )} consisting of pairs of examples x i and corresponding labels y i .

We wish to minimize the expected negative log-likelihood on D, or cross-entropy classification loss, over the set of model parameters Θ:

where p(y | x; Θ) is the likelihood our model assigns to a dataset sample (x, y).

The likelihood function is implemented using an ANN with parameters

where W k and b k denote the weight (including convolutional) and bias terms at layer k, respectively.

Compressing the model amounts to compressing each parameter in the set Θ. Instead of compressing each parameter directly, we compress reparameterized forms of them.

To be precise, we introduce the reparameterizations

Figure 2: Classifier architecture.

The Φ tensors (annotated with a tilde) are stored in their compressed form.

During inference, they are read from storage, uncompressed, and transformed via f into Θ, the usual parameters of a convolutional or dense layer (denoted without a tilde).

The internals of f conv and f dense in our experiments for layer k, annotated with the dimensionalities.

In f conv , H, W , I, O refer to the convolutional height, width, input channel, output channel, respectively.

For f dense , I and O refer to the number of input and output activations.

For f conv , we use an affine transform, while for f dense we use a scalar shift and scale, whose parameters are captured in Ψ. Note that in both cases, the number of parameters of f itself (labeled as ψ) is significantly smaller than the size of the model parameters it decodes.

f bias such that

We can think of each parameter decoder f as a mapping from reparameterization space to parameter space.

For ease of notation, we write F = {f conv , f dense , f bias } and Θ = F(Φ).

The parameter decoders themselves may have learnable parameters, which we denote Ψ. Our method is visually summarized in figures 2 and 3.

A central component of our approach is partitioning the set of model parameters into groups.

For the purpose of creating a model compression method, we interpret entire groups of model parameters as samples from the same learned distribution.

We define a fully factorized distribution q(Φ) = φ∈Φ q φ (φ), and introduce parameter sharing within the factors q φ of the distribution that correspond to the same group, as well as within the corresponding decoders.

These group assignments are fixed a priori.

For instance, in figure 2, W 1 and W 2 can be assumed to be samples of the same distribution, that is q W 1 (·) = q W 2 (·).

To be consistent, we also use the same parameter decoder f conv to decode them.

Further, each of the reparameterizations φ is defined as a rank-2 tensor (a matrix), where each row corresponds to a "sample" from the learned distribution.

The operations in f apply the same transformation to each row (figure 3).

As an example, in f conv , each spatial H × W matrix of filter coefficients is assumed to be a sample from the same distribution.

We describe how this aids in compression in the following section.

Our method can be applied analogously to various model partitionings.

In fact, in our experiments, we vary the size of the groups, i.e., the number of parameters assumed i.i.d., depending on the total number of parameters of the model (Θ).

The size of the groups parameterizes a trade-off between compressibility and overhead: if groups consisted of just one scalar parameter each, compressibility would be maximal, since q would degenerate (i.e., would capture the value of the parameter with certainty).

However, the overhead would be maximal, since F and q would have a large number of parameters that would need to be included in the model size (defeating the purpose of compression).

On the other hand, encoding all parameters of the model with one and the same decoder and scalar distribution would minimize overhead, but may be overly restrictive by failing to capture distributional differences amongst all the parameters, and hence lead to suboptimal compressibility.

We describe the group structure of each network that we use in more detail in the experiments section.

In order to apply a Shannon-style entropy coder efficiently to the reparameterizations Φ, we need a discrete alphabet of representers and associated probabilities for each representer.

Rather than handling an expressive set of representers, as in VQ, we choose to fix them to the integers, and achieve expressivity via the parameter decoders F instead.

Each φ ∈ Z d× is a matrix interpreted as consisting of d samples from a discrete probability distribution producing vectors of dimension .

We fit a factorized probability model

to each column i of φ, using different probability models q i for each corresponding parameter decoder (the form of q i is described in the next section).

Fitting of probability models is typically done by minimizing the negative log-likelihood.

Assuming φ follows the distribution q, Shannon's source coding theorem states that the minimal length of a bit sequence encoding φ is the selfinformation of φ under q:

which is identical to Shannon cross entropy up to an expectation operator, and identical to the negative log likelihood up to a constant factor.

By minimizing I over q and φ during training, we thus achieve two goals: 1) we fit q to the model parameters in a maximum likelihood sense, and 2) we directly optimize the parameters for compressibility.

After training, we design an arithmetic code for q, and use it to compress the model parameters.

This method incurs only a small overhead over the theoretical bound due to the finite length of the bit sequence (arithmetic coding is asymptotically optimal).

Practically, the overhead amounts to less than 1% of the size of the bit sequence; thus, self-information is an excellent proxy for model size.

Further overhead results from including a description of Ψ, the parameters of the parameter decoders, as well as of q itself (in the form of a table) in the model size.

However, these can be considered constant and small compared to the total model size, and thus do not need to be explicitly optimized for.

The overall loss function is simply the additive combination of the original cross-entropy classification loss under reparameterization with the self-information of all reparameterizations:

We refer to the second term (excluding the constant λ) as the rate loss.

By varying λ across different experiments, we can explore the Pareto frontier of compressed model size vs. model accuracy.

To compare our method to other work, we varied λ such that our method produced similar accuracy, and then compared the resulting model size.

Since Φ is discrete-valued, we need to make some further approximations in order to optimize L over it using stochastic gradient descent.

To get around this, we maintain continuous surrogatesΦ.

For optimizing the classification loss, we use the "straight-through" gradient estimator Bengio et al., 2013 , which provides a biased gradient estimate but has shown good results in practice.

This consists of rounding the continuous surrogate to the nearest integer during training, and ignoring the rounding for purposes of backpropagation.

After training, we only keep the discretized values.

In order to obtain good estimates for both the rate term and its gradient during training, we adopt a relaxation approach previously described in (Ballé, Minnen, et al., 2018) ; the code is provided as an open source library 1 .

In a nutshell, the method replaces the probability mass functions q i with a set of non-parametric continuous density functions, which are based on small ANNs.

These density models are fitted toφ j,i + n j,i , where n j,i ∼ U(− 1 2 , 1 2 ) is i.i.d.

uniformly distributed additive noise.

This turns out to work well in practice, because the negative log likelihood of these noise-affected variates under the continuous densities approximates the self-information I:

whereq i denote the density functions.

Once the density models are trained, the values of the probability mass functions modeling φ are derived from the substitutesq i and stored in a table, which is included in the model description.

The parameters ofq i are no longer needed after training.

For our MNIST and CIFAR-10 experiments, we evaluate our method by applying it to four distinct image classification networks: LeNet300-100 (Lecun et al., 1998) and LeNet-5-Caffe 2 on MNIST (LeCun and Cortes, 2010), and VGG-16 3 (Simonyan and Zisserman, 2015) and ResNet-20 (He et al., 2016b; Zagoruyko and Komodakis, 2016) with width multiplier 4 (ResNet-20-4) on CIFAR-10 (Zagoruyko and Komodakis, 2016).

For our ImageNet experiments, we evaluate our method on the ResNet-18 and ResNet-50 (He et al., 2016a) networks.

We train all our models from scratch and compare them with recent state-of-the-art methods by quoting performance from their respective papers.

Compared to many previous approaches, we do not initialize the network with pre-trained or pre-sparsified weights.

We found it useful to use two separate optimizers: one to optimize the variables of the probability model and one to optimize the variables of the network.

The optimizer for the probability model is always Adam (Kingma and Ba, 2014 ) with a learning rate of 0.0001.

We chose to always use Adam because the parameter updates used by Adam are independent of any scaling of the objective (when its hyper-parameter is sufficiently small).

In our method, the probability model variables only get gradients from the entropy loss which is scaled by the rate penalty λ.

Adam normalizes out this scale and makes the learning rate of the probability model independent of λ and of other hyperparameters such as the model partitioning.

We apply our method to two LeNet variants: LeNet300-100 and LeNet5-Caffe and report results in Table 1 .

We train the networks using Adam with a constant learning rate of 0.001 for 200,000 iterations.

To remedy some of the training noise from quantization, we maintain an exponential moving average (EMA) of the weights and evaluate using those.

Note that this does not affect the quantization, as quantization is performed after the EMA variables are restored.

LeNet300-100 consists of 3 fully connected layers.

We partitioned this network into three parameter groups: one for the first two fully connected layers, one for the classifier layer, and one for biases.

LeNet5-Caffe consists of two 5×5 convolutional layers followed by two fully connected layers, with max pooling following each convolutional layer.

We partitioned this network into four parameter groups: One for both of the convolutional layers, one for the penultimate fully connected layer, one for the final classifier layer, and one for the biases.

As evident from Table 1 , for the larger LeNet300-100 model, our method outperforms all the baselines while maintaining a comparable error rate.

For the smaller LeNet5-Caffe model, our method is second only to Minimal Random Code Learning (Havasi et al., 2018) .

Note that in both of the MNIST models, the number of probability distributions = 1 in every parameter group, including in the convolutional layers.

To be precise, the W k for the convolutional weights W k will be H · W ·

I · O × 1.

We found that this gives a better trade-off, since the model is small to begin with, and having = 5 · 5 = 25 scalar probability models for 5 × 5 convolutional layers would have too much overhead.

For both of the MNIST models, we found that letting each subcomponent of F be a simple dimension-wise scalar affine transform (similar to f dense in figure 3) , was sufficient.

Since each φ is quantized to integers, having a flexible scale and shift leads to flexible SQ, similar to in (Louizos, Reisser, et al., 2018) .

Due to the small size of the networks, more complex transformation functions lead to too much overhead.

We apply our method to VGG-16 (Simonyan and Zisserman, 2015) and ResNet-20-4 (He et al., 2016b; Zagoruyko and Komodakis, 2016) and report the results in Table 1 .

For both VGG-16 and ResNet-20-4, we use momentum of 0.9 with an initial learning rate of 0.1, and decay by 0.2 at iterations 256,000, 384,000, and 448,000 for a total of 512,000 iterations.

This learning rate schedule was fixed from the beginning and was not tuned in any way other than verifying that our models' training loss had converged.

VGG-16 consists of 13 convolutional layers of size 3 × 3 followed by 3 fully connected layers.

We split this network into four parameter groups: one for all convolutional layers and one each all fully connected layers.

We do not compress biases.

We found that our biases in float32 format add up to about 20 KB, and we add that to our reported numbers.

ResNet-20-4 consists of 3 ResNet groups with 3 residual blocks each.

There is also an initial convolution layer and a final fully connected classification layer.

We partition this network into two parameter groups: one for all convolutional layers and one for the final classification layer.

We also do not compress biases but include them in our results; they add up to about 11 KB.

For VGG-16 and ResNet-20-4 convolutions, = O × I = 9; f conv and f dense are exactly as pictured in figure 3.

To speed up training, we fixed ψ W .

We found that the inverse real-valued discrete Fourier transform (DFT) performs much better than SQ, or any random orthogonal matrix (Figure 4) .

From the error vs. rate plots, the benefit of reparameterization in the high compression regime is evident.

VGG-16 and ResNet-20-4 both contain batch normalization (Ioffe and Szegedy, 2015) layers that include a moving average for the mean and variance.

Following (Havasi et al., 2018) , we do not include the moving averages in our reported numbers.

We do, however, include the batch normalization bias term β and let it function as the bias for each layer (γ is set to a constant 1).

For the ImageNet dataset (Russakovsky et al., 2015) , we reproduce the training setup and hyperparameters from He et al. (2016a) .

All 3x3 convolutional layers belong to a single parameter group, similar to our CIFAR experiments, 1x1 convolutional layers to a single group (applicable to , and all the remaining layers in their own groups.

This gives a total of 4 parameter groups for ResNet-50 and 3 groups for ResNet-18.

Analogously to the CIFAR experiments, we compare SQ to using random orthogonal or DFT matrices for reparameterizing the convolution kernels (figure 4a).

Existing model compression methods are typically built on a combination of pruning, quantization, or coding.

Pruning involves sparsifying the network either by removing individual parameters or higher level structures such as convolutional filters, layers, activations, etc.

Various strategies for pruning weights include looking at the Hessian (Cun et al., 1990) or just their p norm (Han et al., 2015) .

Srinivas and Babu (2015) focus on pruning individual units, and H. Li et al. (2016) prunes convolutional filters.

Louizos, Ullrich, et al. (2017) and Molchanov et al. (2017) (Louizos, Ullrich, et al., 2017) 18.2 KB (58x) 1.8% Bayesian Compression (GHS) (Louizos, Ullrich, et al., 2017) 18.0 KB (59x) 2.0% Sparse Variational Dropout (Molchanov et al., 2017) 9.38 KB (113x) 1.8% Our Method (SQ) 8.56 KB (124x) 1.9%

LeNet5-Caffe (MNIST) Uncompressed 1.72 MB 0.7% Sparse Variational Dropout (Molchanov et al., 2017) 4.71 KB (365x) 1.0% Bayesian Compression (GHS) (Louizos, Ullrich, et al., 2017) 2.23 KB (771x) 1.0% Minimal Random Code Learning (Havasi et al., 2018) 1.52 KB (1110x) 1.0% Our Method (SQ) 2.84 KB (606x) 0.9%

Uncompressed 60 MB 6.6% Bayesian Compression (Louizos, Ullrich, et al., 2017) 525 KB (116x) 9.2% DeepCABAC (Wiedemann, Kirchhoffer, et al., 2019) 960 KB (62.5x) 9.0% Minimal Random Code Learning (Havasi et al., 2018) 417 KB (159x) 6.6% Minimal Random Code Learning (Havasi et al., 2018) 168 KB (452x) 10.0% Our Method (DFT) 101 KB (590x) 10.0%

ResNet-20-4 (CIFAR-10) (Dubey et al., 2018) 6.46 MB (16x) 26.0% DeepCABAC (Wiedemann, Kirchhoffer, et al., 2019) 6.06 MB (17x to in our compression experiments, also prune parts of the network.

Dubey et al. (2018) describes a dimensionality reduction technique specialized for CNN architectures.

Pruning is a simple approach to reduce memory requirements as well as computational complexity, but doesn't inherently tackle the problem of efficiently representing the parameters that are left.

Here, we primarily focus on the latter: given a model architecture and a task, we're interested in finding a set of parameters which can be described in a compact form and yield good prediction accuracy.

Our work is largely orthogonal to the pruning literature, and could be combined if reducing the number of units is desired.

Quantization involves restricting the parameters to a small set of unique values.

There is work in binarizing or ternarizing networks (Courbariaux et al., 2015; F. Li et al., 2016; Zhou et al., 2018) via either straight-through gradient approximation (Bengio et al., 2013) or stochastic rounding (Gupta et al., 2015) .

Recently, Louizos, Reisser, et al. (2018) introduced a new differentiable quantization procedure that relaxes quantization.

We use the straight-through heuristic, but could possibly use other stochastic approaches to improve our methods.

While most of these works focus on uniform quantization, Baskin et al. (2018) also extend to non-uniform quantization, which our generalized transformation function amounts to.

Han et al. (2015) and Ullrich et al. (2017) share weights and quantize by clustering, Chen, J. Wilson, et al. (2015) randomly enforce weight sharing, and thus effectively perform VQ with a pre-determined assignment of parameters to representers.

Other works also make the observation that representing weights in the frequency domain helps compression; Chen, J. T. Wilson, et al. (2016) randomly enforce weight sharing in the frequency domain and Wang et al. (2016) use K-means clustering in the frequency domain.

Coding (entropy coding, or Shannon-style compression) methods produce a bit sequence that can allow convenient storage or transmission of a trained model.

This generally involves quantization as a first step, followed by methods such as Huffman coding (Huffman, 1952) , arithmetic coding (Rissanen and Langdon, 1981), etc.

Entropy coding methods exploit a known probabilistic structure of the data to produce optimized binary sequences whose length ideally closely approximates the cross entropy of the data under the probability model.

In many cases, authors represent the quantized values directly as binary numbers with few digits (Courbariaux et al., 2015; F. Li et al., 2016; Louizos, Reisser, et al., 2018) , which effectively leaves the probability distribution over the values unexploited for minimizing model size; others do exploit it (Han et al., 2015) .

Wiedemann, Marban, et al. (2018) formulate model compression with an entropy constraint, but use (non-reparameterized) scalar quantization.

Their model significantly underperforms all the state-of-the-art models that we compare with (Table 1) .

Some recent work has claimed improved compression performance by skipping quantization altogether (Havasi et al., 2018) .

Our work focuses on coding with quantization.

Han et al. (2015) defined their method using a four-stage training process: 1. training the original network, 2.

pruning and re-training, 3. quantization and re-training, and 4.

entropy coding.

This approach has influenced many follow-up publications.

In the same vein, many current high-performing methods have significant complexity in implementation or require a multi-stage training process.

Havasi et al. (2018) requires several stages of training and retraining while keeping parts of the network fixed.

Wiedemann, Kirchhoffer, et al. (2019) require pre-sparsification of the network, which is computationally expensive, and use a more complex (context-adaptive) variant of arithmetic coding which may be affected by MPEG patents.

These complexities can prevent methods from scaling to larger architectures or decrease their practical usability.

In contrast, our method requires only a single training stage followed by a royalty-free version of arithmetic coding.

In addition, we commit to releasing the source code of our method for easy reproducibility (upon publication).

Our method has parallels to recent work in learned image compression (Ballé, Laparra, et al., 2016; Theis et al., 2017) that uses end-to-end trained deep models for significant performance improvements in lossy image compression.

These models operate in an autoencoder framework, where scalar quantization is applied in the latent space.

Our method can be viewed as having just a decoder that is used to transform the latent representation into the model parameters, but no encoder.

We describe a simple model compression method built on two ingredients: joint (i.e., end-to-end) optimization of compressibility and classification performance in only a single training stage, and reparameterization of model parameters, which increases the flexibility of the representation over scalar quantization, and is applicable to arbitrary network architectures.

We demonstrate that stateof-the-art model compression performance can be achieved with this simple framework, outperforming methods that rely on complex, multi-stage training procedures.

Due to its simplicity, the approach is particularly suitable for larger models, such as VGG and especially ResNets.

In future work, we may consider the potential benefits of even more flexible (deeper) parameter decoders.

@highlight

An end-to-end trainable model compression method optimizing accuracy jointly with the expected model size.