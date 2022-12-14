Flow-based generative models are powerful exact likelihood models with efficient sampling and inference.

Despite their computational efficiency, flow-based models generally have much worse density modeling performance compared to state-of-the-art autoregressive models.

In this paper, we investigate and improve upon three limiting design choices employed by flow-based models in prior work: the use of uniform noise for dequantization, the use of inexpressive affine flows, and the use of purely convolutional conditioning networks in coupling layers.

Based on our findings, we propose Flow++, a new flow-based model that is now the state-of-the-art non-autoregressive model for unconditional density estimation on standard image benchmarks.

Our work has begun to close the significant performance gap that has so far existed between autoregressive models and flow-based models.

Deep generative models -latent variable models in the form of variational autoencoders BID16 , implicit generative models in the form of GANs BID8 , and exact likelihood models like PixelRNN/CNN (van den c) , Image Transformer BID22 , PixelSNAIL , NICE, RealNVP, and Glow BID5 BID15 -have recently begun to successfully model high dimensional raw observations from complex real-world datasets, from natural images and videos, to audio signals and natural language BID14 BID34 .Autoregressive models, a certain subclass of exact likelihood models, achieve state-of-the-art density estimation performance on many challenging real-world datasets, but generally suffer from slow sampling time due to their autoregressive structure BID28 BID22 .

Inverse autoregressive models can sample quickly and potentially have strong modeling capacity, but they cannot be trained efficiently by maximum likelihood .

Non-autoregressive flow-based models (which we will refer to as "flow models"), such as NICE, RealNVP, and Glow, are efficient for sampling, but have so far lagged behind autoregressive models in density estimation benchmarks BID5 BID15 .In the hope of creating an ideal likelihood-based generative model that simultaneously has fast sampling, fast inference, and strong density estimation performance, we seek to close the density estimation performance gap between flow models and autoregressive models.

In subsequent sections, we present our new flow model, Flow++, which is powered by an improved training procedure for continuous likelihood models and a number of architectural extensions of the coupling layer defined by BID5 .

A flow model f is constructed as an invertible transformation that maps observed data x to a standard Gaussian latent variable z = f (x), as in nonlinear independent component analysis BID1 BID10 BID9 .

The key idea in the design of a flow model is to form f by stacking individual simple invertible transformations BID5 BID15 BID25 BID19 .

Explicitly, f is constructed by composing a series of invertible flows as DISPLAYFORM0 , with each f i having a tractable inverse and a tractable Jacobian determinant.

This way, sampling is efficient, as it can be performed by computing DISPLAYFORM1 1 (z) for z ??? N (0, I), and so is training by maximum likelihood, since the model density DISPLAYFORM2 is easy to compute and differentiate with respect to the parameters of the flows f i .

In this section, we describe three modeling inefficiencies in prior work on flow models: (1) uniform noise is a suboptimal dequantization choice that hurts both training loss and generalization; (2) commonly used affine coupling flows are not expressive enough; (3) convolutional layers in the conditioning networks of coupling layers are not powerful enough.

Our proposed model, Flow++, consists of a set of improved design choices: (1) variational flow-based dequantization instead of uniform dequantization; (2) logistic mixture CDF coupling flows; (3) self-attention in the conditioning networks of coupling layers.

Many real-world datasets, such as CIFAR10 and ImageNet, are recordings of continuous signals quantized into discrete representations.

Fitting a continuous density model to discrete data, however, will produce a degenerate solution that places all probability mass on discrete datapoints BID30 .

A common solution to this problem is to first convert the discrete data distribution into a continuous distribution via a process called "dequantization," and then model the resulting continuous distribution using the continuous density model BID30 BID6 BID28 .

Dequantization is usually performed in prior work by adding uniform noise to the discrete data over the width of each discrete bin: if each of the D components of the discrete data x takes on values in {0, 1, 2, . . .

, 255}, then the dequantized data is given by y = x + u, where u is drawn uniformly from [0, 1) D .

BID29 note that training a continuous density model p model on uniformly dequantized data y can be interpreted as maximizing a lower bound on the log-likelihood for a certain discrete model P model on the original discrete data x: DISPLAYFORM0 The argument of BID29 proceeds as follows.

Letting P data denote the original distribution of discrete data and p data denote the distribution of uniformly dequantized data, Jensen's inequality implies that DISPLAYFORM1 Consequently, maximizing the log-likelihood of the continuous model on uniformly dequantized data cannot lead to the continuous model degenerately collapsing onto the discrete data, because its objective is bounded above by the log-likelihood of a discrete model.

While uniform dequantization successfully prevents the continuous density model p model from collapsing to a degenerate mixture of point masses on discrete data, it asks p model to assign uniform density to unit hypercubes x + [0, 1) D around the data x.

It is difficult and unnatural for smooth function approximators, such as neural network density models, to excel at such a task.

To sidestep this issue, we now introduce a new dequantization technique based on variational inference.

Again, we are interested in modeling D-dimensional discrete data x ??? P data using a continuous density model p model , and we will do so by maximizing the log-likelihood of its associated discrete model P model (x) := [0,1) D p model (x + u) du.

Now, however, we introduce a dequantization noise distribution q(u|x), with support over u ??? [0, 1) D .

Treating q as an approximate posterior, we have the following variational lower bound, which holds for all q: DISPLAYFORM0 We will choose q itself to be a conditional flow-based generative model of the form u = q x ( ), where DISPLAYFORM1 x /???u , and thus we obtain the objective DISPLAYFORM2 which we maximize jointly over p model and q. When p model is also a flow model x = f ???1 (z) (as it is throughout this paper), it is straightforward to calculate a stochastic gradient of this objective using the pathwise derivative estimator, as f (x + q x ( )) is differentiable with respect to the parameters of f and q.

Notice that the lower bound for uniform dequantization -eqs.

(3) to (5) -is a special case of our variational lower bound -eqs.

(6) to (8), when the dequantization distribution q is a uniform distribution that ignores dependence on x. Because the gap between our objective (8) and the true expected log-likelihood DISPLAYFORM3 , using a uniform q forces p model to unnaturally place uniform density over each hypercube x + [0, 1) D to compensate for any potential looseness in the variational bound introduced by the inexpressive q. Using an expressive flow-based q, on the other hand, allows p model to place density in each hypercube x + [0, 1) D according to a much more flexible distribution q(u|x).

This is a more natural task for p model to perform, improving both training and generalization loss.

Recent progress in the design of flow models has involved carefully constructing flows to increase their expressiveness while preserving tractability of the inverse and Jacobian determinant computations.

One example is the invertible 1 ?? 1 convolution flow, whose inverse and Jacobian determinant can be calculated and differentiated with standard automatic differentiation libraries BID15 .

Another example, which we build upon in our work here, is the affine coupling layer BID6 .

It is a parameterized flow y = f ?? (x) that first splits the components of x into two parts x 1 , x 2 , and then computes y = (y 1 , y 2 ), given by DISPLAYFORM0 Here, a ?? and b ?? are outputs of a neural network that acts on x 1 in a complex, expressive manner, but the resulting behavior on x 2 always remains an elementwise affine transformation -effectively, a ?? and b ?? together form a data-parameterized family of invertible affine transformations.

This allows the affine coupling layer to express complex dependencies on the data while keeping inversion and log-likelihood computation tractable.

Using ?? and exp to respectively denote elementwise multiplication and exponentiation, DISPLAYFORM1 The splitting operation x ??? (x 1 , x 2 ) and merging operation (y 1 , y 2 ) ???

y are usually performed over channels or over space in a checkerboard-like pattern BID6 .

We found in our experiments that density modeling performance of these coupling layers could be improved by augmenting the data-parameterized elementwise affine transformations by more general nonlinear elementwise transformations.

For a given scalar component x of x 2 , we apply the cumulative distribution function (CDF) for a mixture of K logistics -parameterized by mixture probabilities, means, and log scales ??, ??, s -followed by an inverse sigmoid and an affine transformation parameterized by a and b: DISPLAYFORM0 where MixLogCDF(x; ??, ??, DISPLAYFORM1 The transformation parameters ??, ??, s, a, b for each component of x 2 are produced by a neural network acting on x 1 .

This neural network must produce these transformation parameters for each component of x 2 , hence it produces vectors a ?? (x 1 ) and b ?? (x 1 ) and tensors ?? ?? (x 1 ), ?? ?? (x 1 ), s ?? (x 1 ) (with last axis dimension K).

The coupling transformation is then given by: DISPLAYFORM2 where the formula for computing y 2 operates elementwise.

The inverse sigmoid ensures that the inverse of this coupling transformation always exists: the range of the logistic mixture CDF is (0, 1), so the domain of its inverse must stay within this interval.

The CDF itself can be inverted efficiently with bisection, because it is a monotonically increasing function.

Moreover, the Jacobian determinant of this transformation involves calculating the probability density function of the logistic mixtures, which poses no computational difficulty.

In addition to improving the expressiveness of the elementwise transformations on x 2 , we found it crucial to improve the expressiveness of the conditioning on x 1 -that is, the expressiveness of the neural network responsible for producing the elementwise transformation parameters ??, ??, s, a, b.

Our best results were obtained by stacking convolutions and multi-head self attention into a gated residual network BID20 , in a manner resembling the Transformer BID34 with pointwise feedforward layers replaced by 3??3 convolutional layers.

Our architecture is defined as a stack of blocks.

Each block consists of the following two layers connected in a residual fashion, with layer normalization BID0 after each residual connection: DISPLAYFORM0 where Gate refers to a 1 ?? 1 convolution that doubles the number of channels, followed by a gated linear unit BID3 .

The convolutional layer is identical to the one used by PixelCNN++ BID28 , and the multi-head self attention mechanism we use is identical to the one in the Transformer BID34 . (We always use 4 heads in our experiments, since we found it to be effective early on in our experimentation process.)With these blocks in hand, the network that outputs the elementwise transformation parameters is simply given by stacking blocks on top of each other, and finishing with a final convolution that increases the number of channels to the amount needed to specify the elementwise transformation parameters.

Here, we show that Flow++ achieves state-of-the-art density modeling performance among nonautoregressive models on CIFAR10 and 32x32 and 64x64 ImageNet.

We also present ablation experiments that quantify the improvements proposed in section 3, and we present example generative samples from Flow++ and compare them against samples from autoregressive models.

Our experiments employed weight normalization and data-dependent initialization .

We used the checkerboard-splitting, channel-splitting, and downsampling flows of BID6 ; we also used before every coupling flow an invertible 1x1 convolution flows of BID15 , as well as a variant of their "actnorm" flow that normalizes all activations independently (instead of normalizing per channel).

Our CIFAR10 model used 4 coupling layers with checkerboard splits at 32x32 resolution, 2 coupling layers with channel splits at 16x16 resolution, and 3 coupling layers with checkerboard splits at 16x16 resolution; each coupling layer used 10 convolution-attention blocks, all with 96 filters.

More details on architectures, as well as details for the other experiments, will be given in a source code release.

In table 1, we show that Flow++ achieves state-of-the-art density modeling results out of all nonautoregressive models, and it is competitive with autoregressive models: its performance is on par with the first generation of PixelCNN models , and it outperforms Multiscale PixelCNN BID24 .

As of submission, our models have not fully converged due to computational constraint and we expect further performance gain in future revision of this manuscript.

3.14 --PixelRNN (van den 3.00 3.86 3.63 Gated PixelCNN (van den BID33 3.03 3.83 3.57 PixelCNN++ BID28 2.92 --Image Transformer BID22 2.90 3.77 -PixelSNAIL 2.85 3.80 3.52

We ran the following ablations of our model on unconditional CIFAR10 density estimation: variational dequantization vs. uniform dequantization; logistic mixture coupling vs. affine coupling; and stacked self-attention vs. convolutions only.

As each ablation involves removing some component of the network, we increased the number of filters in all convolutional layers (and attention layers, if present) in order to match the total number of parameters with the full Flow++ model.

In FIG0 and table 2, we compare the performance of these ablations relative to Flow++ at 400 epochs of training, which was not enough for these models to converge, but far enough to see their relative performance differences.

Switching from our variational dequantization to the more standard uniform dequantization costs the most: approximately 0.127 bits/dim.

The remaining two ablations both cost approximately 0.03 bits/dim: switching from our logistic mixture coupling layers to affine coupling layers, and switching from our hybrid convolution-and-self-attention architecture to a pure convolutional residual architecture.

Note that these performance differences are present despite all networks having approximately the same number of parameters: the improved performance of Flow++ comes from improved inductive biases, not simply from increased parameter count.

The most interesting result is probably the effect of the dequantization scheme on training and generalization loss.

At 400 epochs of training, the full Flow++ model with variational dequantization has a train-test gap of approximately 0.02 bits/dim, but with uniform dequantization, the train-test gap is approximately 0.06 bits/dim.

This confirms our claim in Section 3.1.2 that training with variational dequantization is a more natural task for the model than training with uniform dequantization.

BID23 .

More samples are available in the appendix (section 7).

Likelihood-based models constitute a large family of deep generative models.

One subclass of such methods, based on variational inference, allows for efficient approximate inference and sampling, but does not admit exact log likelihood computation BID16 BID26 .

Another subclass, which we called exact likelihood models in this work, does admit exact log likelihood computation.

These exact likelihood models are typically specified as invertible transformations that are parameterized by neural networks BID4 BID18 BID30 BID5 BID7 BID28 .There is prior work that aims to improve the sampling speed of deep autoregressive models.

The Multiscale PixelCNN BID24 modifies the PixelCNN to be non-fully-expressive by introducing conditional independence assumptions among pixels in a way that permits sampling in a logarithmic number of steps, rather than linear.

Such a change in the autoregressive structure allows for faster sampling but also makes some statistical patterns impossible to capture, and hence reduces the capacity of the model for density estimation.

WaveRNN BID13 improves sampling speed for autoregressive models for audio via sparsity and other engineering considerations, some of which may apply to flow models as well.

There is also recent work that aims to improve the expressiveness of coupling layers in flow models.

BID15 demonstrate improved density estimation using an invertible 1x1 convolution flow, and demonstrate that very large flow models can be trained to produce photorealistic faces.

BID21 introduce piecewise polynomial couplings that are similar in spirit to our mixture of logistics couplings.

They found them to be more expressive than affine couplings, but reported little performance gains in density estimation.

We leave a detailed comparison between our coupling layer and the piecewise polynomial CDFs for future work.

We presented Flow++, a new flow-based generative model that begins to close the performance gap between flow models and autoregressive models.

Our work considers specific instantiations of design principles for flow models -dequantization, flow design, and conditioning architecture design -and we hope these principles will help guide future research in flow models and likelihoodbased models in general.7 APPENDIX A: SAMPLES

<|TLDR|>

@highlight

Improved training of current flow-based generative models (Glow and RealNVP) on density estimation benchmarks