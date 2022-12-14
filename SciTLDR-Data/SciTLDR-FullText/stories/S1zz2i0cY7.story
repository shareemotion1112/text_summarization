We consider the problem of using variational latent-variable models for data compression.

For such models to produce a compressed binary sequence, which is the universal data representation in a digital world, the latent representation needs to be subjected to entropy coding.

Range coding as an entropy coding technique is optimal, but it can fail catastrophically if the computation of the prior differs even slightly between the sending and the receiving side.

Unfortunately, this is a common scenario when floating point math is used and the sender and receiver operate on different hardware or software platforms, as numerical round-off is often platform dependent.

We propose using integer networks as a universal solution to this problem, and demonstrate that they enable reliable cross-platform encoding and decoding of images using variational models.

The task of information transmission in today's world is largely divided into two separate endeavors: source coding, or the representation of data (such as audio or images) as sequences of bits, and channel coding, representing sequences of bits as analog signals on imperfect, physical channels such as radio waves BID7 .

This decoupling has substantial benefits, as the binary representations of arbitrary data can be seamlessly transmitted over arbitrary physical channels by only changing the underlying channel code, rather than having to design a new code for every possible combination of data source and physical channel.

Hence, the universal representation of any compressed data today is the binary channel, a representation which consists of a variable number of binary symbols, each with probability 1 2 , and no noise (i.e. uncertainty).

As a latent representation, the binary channel unfortunately is a severe restriction compared to the richness of latent representations defined by many variational latent-variable models in the literature (e.g., BID13 BID22 BID18 , and in particular models targeted at data compression BID23 BID0 .

Variational latent-variable models such as VAEs BID13 consist of an encoder model distribution e(y | x) bringing the data x into a latent representation y, and a decoder model distribution d(x | y), which represents the data likelihood conditioned on the latents.

Given an encoder e, we observe the marginal distribution of latents m(y) = E x [e(y | x)], where the expectation runs over the (unknown) data distribution.

The prior p(y) is a variational estimate of the marginal BID1 .By choosing the parametric forms of these distributions and the training objective appropriately, many such models succeed in representing relevant information in the data they are trained for quite compactly (i.e., with a small expected Kullback-Leibler (KL) divergence between the encoder and the prior, E x D KL [e p]), and so may be called compressive in a sense.

However, not all of them can be directly used for practical data compression, as the representation needs to be further converted into binary (entropy encoded).

This conversion is typically performed by range coding, or arithmetic coding BID20 .

Range coding is asymptotically optimal: the length of the binary sequence quickly converges to the expected KL divergence in bits, for reasonably large sequences (such as, for one image).

For this to hold, the following requirements must be satisfied: Figure 1 : The same image, decoded with a model computing the prior using integer arithmetic (left), and the same model using floating point arithmetic (right).

The image was decoded correctly, beginning in the top-left corner, until floating point round-off error caused a small discrepancy between the sender's and the receiver's copy of the prior, at which point the error propagated catastrophically.??? The representation must be discrete-valued, i.e. have a finite number of states, and be noiseless -i.e.

the conditional entropy of the encoder must be zero: DISPLAYFORM0 ??? All scalar elements of the representation y must be brought into a total ordering, and the prior needs to be written using the chain rule of calculus (as a product of conditionals), as the algorithm can only encode or decode one scalar random variable at a time.??? Both sides of the binary channel (i.e. sender and receiver) must be able to evaluate the prior, and they must have identical instances of it.

The latter point is crucial, as range coding is extremely sensitive to differences in p between sender and receiver -so sensitive, in fact, that even small perturbations due to floating point round-off error can lead to catastrophic error propagation.

Unfortunately, numerical round-off is highly platform dependent, and in typical data compression applications, sender and receiver may well employ different hardware or software platforms.

Round-off error may even be non-deterministic on one and the same computer.

Figure 1 illustrates a decoding failure in a model which computes p using floating point math, caused by such computational non-determinism in sender vs. receiver.

Recently, latent-variable models have been explored that employ artificial neural networks (ANNs) to compute hierarchical or autoregressive priors BID22 BID18 , including some of the best-performing learned image compression models BID17 BID14 .

Because ANNs are typically based on floating point math, these methods are vulnerable to catastrophic failures when deployed on heterogeneous platforms.

To address this problem, and enable use of powerful learned variational models for real-world data compression, we propose to use integer arithmetic in these ANNs, as floating-point arithmetic cannot presently be made deterministic across arbitrary platforms.

We formulate a type of quantized neural network we call integer networks, which are specifically targeted at generative and compression models, and at preventing computational non-determinism in computation of the prior.

Because full determinism is a feature of many existing, widely used image and video compression methods, we also consider using integer networks end to end for computing the representation itself.

ANNs are typically composite functions that alternate between linear and elementwise nonlinear operations.

One linear operation followed by a nonlinearity is considered one layer of the network.

To ensure that such a network can be implemented deterministically on a wide variety of hardware platforms, we restrict all the data types to be integral, and all operations to be implemented either with basic arithmetic or lookup tables.

Because integer multiplications (including matrix multiplications or convolutions) increase the dynamic range of the output compared to their inputs, we introduce an additional step after each linear operator, where we divide each of its output by a learned parameter. .

This nonlinearity can be implemented deterministically either using a lookup table or simply using a clipping operation.

The corresponding scaled cumulative of a generalized Gaussian with ?? = 4 used for computing gradients is plotted in cyan, and other choices of ?? in gray.

Right: Example nonlinearity approximating hyperbolic tangent for 4-bit signed integer outputs, given by g Qtanh (v) = Q(7 tanh( v 15 )).

This nonlinearity can be implemented deterministically using a lookup table.

The corresponding scaled hyperbolic tangent used for computing gradients is plotted in cyan.

Concretely, we define the relationship between inputs u and outputs w of one layer as: DISPLAYFORM0 In order, the inputs u are subjected to a linear transform H (a matrix multiplication, or a convolution); a bias vector b is added; the result is divided elementwise by a vector c, yielding an intermediate result vector v; and finally, an elementwise nonlinearity g is applied to v.

The activations w and all intermediate results, as well as the parameters H, b, and c are all defined as integers.

However, they may use differing number formats.

For v to be integral, we define here to perform rounding division (equivalent to division followed by rounding to the nearest integer).

In programming languages such as C, this can be implemented with integer operands m, n as DISPLAYFORM1 where Q rounds to the nearest integer and / / is floor division; here, the addition can be folded into the bias b as an optimization.

We constrain the linear filter coefficients H and the bias vector b to generally use signed integers, and the scaling vector c to use unsigned integers.

We implement the accumulators of the linear transform with larger bit width than the activations and filter coefficients, in order to reflect the potentially increased dynamic range of multiplicative operations.

We assume here that the bias and scaling vectors, as well as the intermediate vector v, have the same bit width as the accumulators.

The elementwise nonlinearity g must be saturating on both ends of its domain, because integers can only represent finite number ranges.

In order to maximize utility of the dynamic range, we scale nonlinearities such that their range matches the bit width of w, while their domain can be scaled somewhat arbitrarily.

Depending on the range of the nonlinearity, the activations w may use a signed or unsigned number format.

For instance, a reasonable choice of number formats and nonlinearity would be:H : 8-bit signed b, v : 32-bit signed (same as accumulator)c : 32-bit unsigned w : 8-bit unsigned g QReLU (v) = max (min(v, 255), 0) In this example, the nonlinearity can be implemented with a simple clipping operation.

Refer to figure 2, left, for a visualization (for visualization purposes, the figure shows a smaller bit width).

H : 4-bit signed b, v : 16-bit signed (same as accumulator) c : 16-bit unsigned w : 4-bit signed DISPLAYFORM0 Here, the nonlinearity approximates the hyperbolic tangent, a widely used nonlinearity.

It may be best implemented using a lookup table (see figure 2, right, for a visualization).

We scale its range to fill the 4-bit signed integer number format of w by multiplying its output with 7.

The domain can be scaled somewhat arbitrarily, since v has a larger bit width than w. When it is chosen too small, w may not utilize all integer values, leading to a large quantization error.

When it is chosen too large, overflow may occur in v, or the size of the lookup table may grow too large for practical purposes.

Therefore, it is best to determine the input scaling based on the shape of the nonlinearity and the available dynamic range.

Here, we simply chose the value of 15 "by eye", so that the nonlinearity is reasonably well represented with the lookup table (i.e., we made sure that at least two or three input values are mapped to each output value, in order to preserve the approximate shape of the nonlinearity).

To effectively accumulate small gradient signals, we train the networks entirely using floating point computations, rounded to integers after every computational operation, while the backpropagation is done with full floating point precision.

More concretely, we define the integer parameters H, b, and c as functions of their floating point equivalents H , b , and c , respectively: DISPLAYFORM0 . . .

DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Here, we simply rescale each element of b using a constant K, which is the bit-width of the kernel H (e.g. 8-bits in the QReLu networks), and round it to the nearest integer.

The reparameterization mapping r is borrowed from : DISPLAYFORM4 When c is small, perturbations in c can lead to excessively large fluctuations of the quotient (i.e., the input to the nonlinearity).

This leads to instabilities in training.

r ensures that values of c are always positive, while gracefully scaling down gradient magnitudes on c near zero.

Effectively, the step size on c is multiplied with a factor that is approximately linear in c .Before rounding the linear filter coefficients in H = [h 1 , . . .

, h N ] , we apply a special rescaling function s to each of its filters h : DISPLAYFORM5 s rescales each filter such that at least one of its minimum and maximum coefficients hits one of the dynamic range bounds (???2 K???1 and 2 K???1 ???1), while keeping zero at zero.

This represents the finest possible quantization of the filter given its integer representation, and thus maximizes accuracy.

To prevent division by zero, we ensure the divisor is larger than or equal to a small constant (for example, = 10 ???20 ).In order to backpropagate gradient signals into the parameters, one cannot simply take gradients of the loss function with respect to H , b , or c , since the rounding function Q has zero gradients almost everywhere, except for the half-integer positions where the gradient is positive infinity.

A simple remedy is to replace the derivative of Q with the identity function, since this is the smoothed gradient across all rounded values.

Further, we treat the rescaling divisor s as if it were a constant.

That is, we compute the derivatives of the loss function with respect to H , b , and c as with the chain rule of calculus, but overriding: DISPLAYFORM6 where r is the replacement gradient function for r as proposed by .

After training is completed, we compute the integer parameters H, b and c one more time, and from then on use them for evaluation.

Note that further reparameterization of the kernels H , such as Sadam , or of the biases b or scaling parameters c , is possible by simply chaining reparameterizations.

In addition to rounding the parameters, it is necessary to round the activations.

To obtain gradients for the rounding division , we simply substitute the gradient of floating point division.

To estimate gradients for the rounded activation functions, we replace their gradient with the corresponding nonrounded activation function, plotted in cyan in figure 2.

In the case of QReLU, the gradient of the clipping operation is a box function, which can lead to training getting stuck, since if activations consistently hit one of the bounds, no gradients are propagated back (this is sometimes called the "dead unit" problem).

As a remedy, we replace the gradient instead with DISPLAYFORM7 where DISPLAYFORM8 ?? , and L is the bit width of w. This function corresponds to a scaled generalized Gaussian probability density with shape parameter ??.

In this context, we can think of ?? as a temperature parameter that makes the function converge to the gradient of the clipping operation as ?? goes to infinity.

Although this setting permits an annealing schedule, we simply chose ?? = 4 and obtained good results.

The integral of this function is plotted in figure 2 (left) in cyan, along with other choices of ?? in gray.

Suppose our prior on the latent representation is p(y | z), where z summarizes other latent variables of the representation (it may be empty).

To apply range coding, we need to impose a total ordering on the elements of y and write it as a chain of conditionals: DISPLAYFORM0 where y :i denotes the vector of all elements of y preceding the ith.

A common assumption is that p is a known distribution, with parameters ?? i computed by an ANN g: DISPLAYFORM1 We simply propose here to compute g deterministically using an integer network, discretizing the parameters ?? to a reasonable accuracy.

If p(y i | ?? i ) itself cannot be computed deterministically, we can precompute all possible values and express it as a lookup table over y i and ?? i .As an example, consider the prior used in the image compression model proposed by , which is a modified Gaussian with scale parameters conditioned on another latent variable: DISPLAYFORM2 We reformulate the scale parameters ?? as: DISPLAYFORM3 where ?? = g(z) is computed using an integer network.

The last activation function in g is chosen to have integer outputs of L levels in the range [0, L ??? 1].

Constants ?? min , ?? max , and L determine the discretized selection of scale parameters used in the model.

The discretization is chosen to be logarithmic, as this choice minimizes E x D KL [e p] for a given number of levels.

During training, we can simply backpropagate through this reformulation, and through g as described in the previous section.

After training, we precompute all possible values of p as a function of y i and ?? i and form a lookup table, while g is implemented with integer arithmetic.

For certain applications, it can be useful not only to be able to deploy a compression model across heterogenous platforms, but to go even further in also ensuring identical reconstructions of the data across platforms.

To this end, it can be attractive to make the entire model robust to non-determinism.

To use integer networks in the encoder or decoder, one can use the equivalent construction as in FORMULA0 Jang et al. FORMULA0 and ??g??stsson et al. (2017) are concerned with producing gradients for categorical distributions and vector quantization (VQ), respectively.

In both methods, the representation is found by evaluating an ANN followed by an arg max function, while useful gradients are obtained by substituting the arg max with a softmax function.

Since arg max can be evaluated deterministically in a platform-independent way, and evaluating a softmax function with rounded inputs is feasible, integer networks can be combined with these models without additional modifications.

Theis et al. FORMULA0 and differ mostly in the details of interaction between the encoder and the prior.

These two approaches are particularly interesting for image compression, as they scale well: Image compression models are often trained with a rate-distortion objective with a Lagrange parameter ??, equivalent to ?? in the ??-VAE objective BID10 BID1 .

Depending on the parameter, the latent representation carries vastly different amounts of information, and the optimal number of latent states in turn varies with that.

While the number of latent states is a hyperparameter that needs to be chosen ahead of time in the categorical/VQ case, the latter two approaches can extend it as needed during training, because the latent states are organized along the real line.

Further, for categorical distributions as well as VQ, the required dimensionality of the function computing the parameters grows linearly with the number of latent states due to their use of the arg max function.

In the latter two models, the number of states can grow arbitrarily without increasing the dimensionality of g.

Both BID23 and use deterministic encoder distributions (i.e. degenerating to delta distributions) during evaluation, but replace them with probabilistic versions for purposes of estimating E x D KL [e p] during training.

BID23 propose to use the following encoder distribution: DISPLAYFORM0 where U is the uniform distribution and g is an ANN.

They replace the gradient of the quantizer with the identity.

During evaluation, y = Q(g(x)) is used as the representation.

use the following distribution during training: DISPLAYFORM1 which makes y shift-invariant.

During evaluation, they determine the representation as y = Q(g(x) ??? o), where o is a sub-integer offset chosen such that the mode (or, if it cannot be estimated easily, the median) of the distribution is centered on one of the quantization bins.

If g is implemented with integer networks, the latter approach becomes equivalent to the former, because g then inherently computes integer outputs, and this is effectively equivalent to the quantization in (16).

However, we've found that training with this construction leads to instabilities, such that the prior distribution never converges to a stable set of parameters.

The reason may be that with quantization in e, the marginal m(y) = E x e(y | x) resembles a piecewise constant function, while the prior p must be forced to be smooth, or E x D KL [e p] would not yield any useful gradients.

Because the prior is a variational approximation of the marginal, this means that the prior must be regularized (which we did not attempt here -we used the nonparametric density model described in ).

On the other hand, when using (17) without quantization, the marginal is typically a smooth density, and the prior can approximate it closely without the need for regularization.

As a remedy for the instabilities, we propose the following trick: We simply use (17) during training, but define the last layer of g without a nonlinearity and with floating point division, such that the representation is compressed on CPU 1 CPU 1 CPU 1 CPU 1 GPU 1 GPU 1 GPU 1 GPU 1 decompressed on CPU 1 GPU 1 CPU 2 GPU 2 CPU 1 GPU 1 CPU 2 GPU 2 Tecnick dataset: 100 RGB images of 1200 ?? 1200 pixels 0% 71% 54% 66% 63% 41% 59% 34% ditto, integer prior 0% 0% 0% 0% 0% 0% 0% 0% CLIC dataset: 2021 RGB images of various pixel sizes 0% 78% 68% 78% 77% 52% 78% 54% ditto, integer prior 0% 0% 0% 0% 0% 0% 0% 0% DISPLAYFORM2 CPU 1: Intel Xeon E5-1650 GPU 1: NVIDIA Titan X (Pascal) CPU 2: Intel Xeon E5-2690 GPU 2: NVIDIA Titan X (Maxwell) Table 1 : Decompression failure rates due to floating point round-off error on Tecnick and CLIC image datasets.

When compressing and decompressing on the same CPU platform (first column), the model decompresses all images correctly.

However, when compressing on a GPU or decompressing on a different platform, a large percentage of the images fail to be decoded correctly.

Implementing the prior of the same model using integer networks ensures correct decompression across all tested platforms.during training, where u is the input to the last layer and / represents elementwise floating point division, and DISPLAYFORM3 during evaluation.

This can be rewritten strictly using integer arithmetic as: DISPLAYFORM4 where represents elementwise multiplication, and the rounded product can be folded into the bias b as an optimization.

This way, the representation is computed deterministically during evaluation, while during training, the marginal still resembles a smooth function, such that no regularization of the prior is necessary.

In order to assess the efficacy of integer networks to enable platform-independent compression and decompression, we re-implemented the image compression model described in , FORMULA0 model, evaluated on BID15 , corresponding to the rate point at approximately 0.7 bits per pixel in FIG2 , right panel.

Generally, training of integer models takes somewhat longer and is somewhat noisier than training of floating point models.

When matching floating point and integer networks for asymptotic performance (128 vs. 256 filters, respectively), integer networks take longer to converge (likely due to their larger number of filters).

When matching by number of filters FORMULA0 , it appears that the training time to convergence is about the same, but the performance ends up worse.which is defined with a hyperprior.

We compare the original model with a version in which the network h s computing the prior is replaced with an integer network.

We used the same network architectures in terms of number of layers, filters, etc., and the same training parameters as in the original paper.

The rate-distortion performance of the model was assessed on BID15 and is shown in FIG2 (left).

The modified model performs identically to the original model, as it maps out the same rate-distortion frontier.

However, it is much more robust to cross-platform compression and decompression (table 1) .

We tested compression and decompression on four different platforms (two CPU platforms and two GPU platforms) and two different datasets, Tecnick BID2 BID2 CLIC (2018) .

The original model fails to correctly decompress more than half of the images on average when compression and decompression occurs on different platforms.

The modified model brings the failure rate down to 0% in all cases.

It should be noted that the decreased accuracy of integer arithmetic generally leads to a lower approximation capacity than with floating point networks.

We found that when implementing the models described in Ball?? (2018) using integer networks throughout, the rate-distortion performance decreased (figure 3, right).

The loss in approximation capacity can be compensated for by increasing the number of filters per layer.

Note that this appears to increase the training time necessary for convergence (figure 4).

However, note that increasing the number of parameters may not necessarily increase the size of the model parameters or the runtime, as the storage requirements for integer parameters (kernels, biases, etc.) are lower than for floating point parameters, and integer arithmetic is computationally less complex than floating point arithmetic in general.

There is a large body of recent research considering quantization of ANNs mostly targeted at image recognition applications.

BID6 train classification networks on lower precision multiplication.

BID11 and BID19 perform quantization down to bilevel (i.e., 1-bit integers) at inference time to reduce computation in classification networks.

More recently, BID24 and others have used quantization during training as well as inference, to reduce computation on gradients as well as activations, and BID5 use non-uniform quantization to remove floating point computation, replacing it completely with integer offsets into an integer lookup table.

While the quantization of neural networks is not a new topic, the results from the above techniques focus almost exclusively on classification networks.

BID8 , BID9 , and others have demonstrated that these types of networks are particularly robust to capacity reduction.

Models used for image compression, like many generative models, are much more sensitive to capacity constraints since they tend to underfit.

As illustrated in and in figure 3 (right), this class of models is much more sensitive to reductions of capacity, both in terms of network size and the expressive power of the activation function.

This may explain why our experiments with post-hoc quantization of network activations have never yielded competitive results for this class of model (not shown).As illustrated in figure 1 and table 1, small floating point inconsistencies in variational latent-variable models can have disastrous effects when we use range coding to employ the models for data compression across different hardware or software platforms.

The reader may wonder whether there exists other entropy coding algorithms that can convert discrete latent-variable representations into a binary representation, and which do not suffer from a sensitivity to perturbations in the probability model.

Unfortunately, such an algorithm would always produce suboptimal results for the following reason.

The source coding theorem BID21 ) establishes a lower bound on the average length of the resulting bit sequences, which range coding achieves asymptotically (i.e. for long bit sequences).

The lower bound is given by the cross entropy between the marginal and the prior: DISPLAYFORM0 where |b(y)| is the length of the binary representation of y. If an entropy coding algorithm tolerates error in the values of p(y | ??), this means it must operate under the assumption of identical probability values for a range of values of ?? -in other words, discretize the probability values.

Since the cross entropy is minimal only for p(y | ??) = m(y) (for all y), this would impose a new lower bound on |b(y)| given by the cross entropy with the discretized probabilities, which is greater or equal to the cross entropy given above.

Thus, the more tolerant the entropy coding method is to errors in p, the further it deviates from optimal performance.

Moreover, it is hard to establish tolerance intervals for probability values computed with floating point arithmetic, in particular when ANNs are used, due to error propagation.

Hence, it is generally difficult to provide guarantees that a given tolerance will not be exceeded.

For similar reasons, current commercial compression methods model probabilities exclusively in the discrete domain (e.g., using lookup tables; BID16 .Our approach to neural network quantization is the first we are aware of which specifically addresses non-deterministic computation, as opposed to computational complexity.

It enables a variety of possible variational model architectures and distributions to be effectively used for platformindependent data compression.

While we aren't assessing its effects on computational complexity here, it is conceivable that complexity reductions can also be achieved with the same approach; this is a topic for future work.

@highlight

We train variational models with quantized networks for computational determinism. This enables using them for cross-platform data compression.