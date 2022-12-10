The increasing demand for neural networks (NNs) being employed on embedded devices has led to plenty of research investigating methods for training low precision NNs.

While most methods involve a quantization step, we propose a principled Bayesian approach where we first infer a distribution over a discrete weight space from which we subsequently derive hardware-friendly low precision NNs.

To this end, we introduce a probabilistic forward pass to approximate the intractable variational objective that allows us to optimize over discrete-valued weight distributions for NNs with sign activation functions.

In our experiments, we show that our model achieves state of the art performance on several real world data sets.

In addition, the resulting models exhibit a substantial amount of sparsity that can be utilized to further reduce the computational costs for inference.

With the advent of deep neural networks (NNs) impressive performances have been achieved in many applications such as computer vision BID13 , speech recognition , and machine translation , among others.

However, the performance improvements are largely attributed to increasing hardware capabilities that enabled the training of ever-increasing network architectures.

On the other side, there is also a growing interest in making NNs available for embedded devices with drastic memory and power limitations -a field with plenty of interesting applications that barely profit from the tendency towards larger and deeper network structures.

Thus, there is an emerging trend in developing NN architectures that allow fast and energy-efficient inference and require little storage for the parameters.

In this paper, we focus on reduced precision methods that restrict the number of bits per weight while keeping the network structures at a decent size.

While this reduces the memory footprint for the parameters accordingly, it can also result in drastic improvements in computation speed if appropriate representations for the weight values are used.

This direction of research has been pushed towards NNs that require in the extreme case only a single bit per weight.

In this case, assuming weights w ∈ {−1, 1} and binary inputs x ∈ {−1, 1}, costly floating point multiplications can be replaced by cheap and hardware-friendly logical XNOR operations.

However, training such NNs is inherently different as discrete valued NNs cannot be directly optimized using gradient based methods.

Furthermore, NNs with binary weights exhibit their full computational benefits only in case the sign activation function is used whose derivative is zero almost everywhere, and, therefore, is not suitable for backpropagation.

Most methods for training reduced precision NNs either quantize the weights of pre-trained full precision NNs BID3 or train reduced precision NNs by maintaining a set of full precision weights that are deterministically or stochastically quantized during forward or backward propagation.

Gradient updates computed with the quantized weights are then applied to the full precision weights BID4 BID19 BID8 .

This approach alone fails if the sign activation function is used.

A promising approach is based on the straight through gradient estimator (STE) BID1 which replaces the zero gradient of hard threshold functions by a non-zero surrogate derivative.

This allows information in computation graphs to flow backwards such that parameters can be updated using gradient based optimization methods.

Encouraging results are presented in BID8 where the STE is applied to the weight binarization and to the sign activation function.

These methods, although showing The aim is to obtain a single discrete-valued NN (top right) with a good performance.

We achieve this by training a distribution over discrete-valued NNs (bottom right) and subsequently deriving a single discrete-valued NN from that distribution.

(b) Probabilistic forward pass: The idea is to propagate distributions through the network by approximating a sum over random variables by a Gaussian and subsequently propagating that Gaussian through the sign activation function.convincing empirical performance, have in common that they appear rather heuristic and it is usually not clear whether they optimize any well defined objective.

Therefore, it is desired to develop principled methods that support discrete weights in NNs.

In this paper, we propose a Bayesian approach where we first infer a distribution q(W ) over a discrete weight space from which we subsequently derive discrete-valued NNs.

Thus, we can optimize over real-valued distribution parameters using gradient-based optimization instead of optimizing directly over the intractable combinatorial space of discrete weights.

The distribution q(W ) can be seen as an exponentially large ensemble of NNs where each NN is weighted by its probability q(W ).Rather than having a single value for each connection of the NN, we now maintain a whole distribution for each connection (see bottom right of FIG0 (a)).

To obtain q(W ), we employ variational inference where we approximate the true posterior p(W |D) by minimizing the variational objective KL(q(W )||p(W |D)).

Although the variational objective is intractable, this idea has recently received a lot of attention for real-valued NNs due to the reparameterization trick which expresses gradients of intractable expectations as expectations of tractable gradients BID20 BID12 BID2 .

This allows us to efficiently compute unbiased gradient samples of the intractable variational objective that can subsequently be used for stochastic optimization.

Unfortunately, the reparameterization trick is only suitable for real-valued distributions which renders it unusable for our case.

The recently proposed Gumbel softmax distribution BID10 BID16 overcomes this issue by relaxing one-hot encoded discrete distributions with probability vectors.

Subsequently, the reparameterization trick can again be applied.

However, for the sign activation function one still has to rely on the STE or similar heuristics.

The log-derivative trick offers an alternative for discrete distributions to express gradients of expectations with expectations of gradients BID18 .

However, the resulting gradient samples are known to suffer from high variance.

Therefore, the log-derivative trick is typically impractical unless suitable variance reduction techniques are used.

This lack of practical methods has led to a limited amount of literature investigating Bayesian NNs with discrete weights BID23 .In this work, we approximate the intractable variational objective with a probabilistic forward pass (PFP) BID26 BID23 BID6 BID21 .

The idea is to propagate probabilities through the network by successively approximating the distributions of activations with a Gaussian and propagating this Gaussian through the sign activation function FIG0 ).

This results in a well-defined objective whose gradient with respect to the variational parameters can be computed analytically.

This is true for discrete weight distributions as well as for the sign activation function with zero gradient almost everywhere.

The method is very flexible in the sense that different weight distributions can be used in different layers.

We utilize this flexibility to represent the weights in the first layer with 3 bits and we use ternary weights w ∈ {−1, 0, 1} in the remaining layers.

In our experiments, we evaluate the performance of our model by reporting the error of (i) the most probable model of the approximate posterior q(W ) and (ii) approximated expected predictions using the PFP.

We show that averaging over small ensembles of NNs sampled from W ∼ q(W ) can improve the performance while inference using the ensemble is still cheaper than inference using a single full precision NN.

Furthermore, our method exhibits a substantial amount of sparsity that further reduces the computational overhead.

Compared to BID8 , our method requires less precision for the first layer, and we do not introduce a computational overhead by using batch normalization which appears to be a crucial component of their method.

The paper is outlined as follows.

In Section 2, we introduce the notation and formally define the PFP.

Section 3 shows details of our model.

Section 4 shows experiments.

In Section 5 we discuss important issues concerning our model and Section 6 concludes the paper.

The structure of a feed-forward NN with L layers is determined by the number of neurons DISPLAYFORM0 Here, d 0 is the dimensionality of the input, d L is the dimensionality of the output, and d l for 0 < l < L is the number of hidden neurons in layer l. A NN defines a function y = x L = f (x 0 ) by iteratively applying a linear transformation a l = W l x l−1 to the inputs from the previous layer followed by a non-linear function DISPLAYFORM1 which is applied element-wise to its inputs.

For l = L, we use the softmax activation function smax i (a) = exp(a i )/ j exp(a j ).

Note that the expensive softmax function does not need to be computed at test time.

where D l is a finite set.1 .

For a Bayesian treatment of NNs, we assume a prior distribution p(W ) over the discrete weights and interpret the output of the NN after the softmax activation as likelihood p(D|W ) for the data set DISPLAYFORM0 is intractable for NNs of any decent size, we employ variational inference to approximate it by a simpler distribution q(W |ν) by minimizing KL(q(W |ν)||p(W |D)) with respect to the variational parameters ν.

We adopt the common mean-field assumption where the approximate posterior factorizes into a product of factors q(w|ν w ) for each weight w ∈ W .2 The variational objective is usually transformed as DISPLAYFORM1 Minimizing this expression with respect to ν does not involve the intractable posterior p(W |D) and the evidence log p(D) is constant with respect to the variational parameters ν.

The KL term can be seen as a regularizer that pulls the approximate posterior q(W ) towards the prior distribution p(W ) whereas the expected log-likelihood captures the data.

While the KL term is tractable if both the prior and the approximate posterior distribution assume independence of the weights, the expected log-likelihood is typically intractable due to a sum over exponentially many terms.

We propose a PFP as closed-form approximation to the expected log-likelihood.

The approximation of the expected log-likelihood resembles a PFP.

In particular, we have DISPLAYFORM0 where we defined DISPLAYFORM1 .

The overall aim is to successively get rid of the weights in each layer and consequently reduce the exponential number of terms to sum over.

In the first approximation in (3), we approximate the activation distribution with Gaussians using a central limit argument.

These Gaussian distributions are propagated through the sign activation function resulting in Bernoulli distributions in (4).

These two steps are iterated until a Gaussian approximation of the output activations in FORMULA5 is obtained.

This integral is approximated using a second-order Taylor expansion of the log-softmax around µ a L .

In the following subsections we provide more details of the individual approximations.

The activations of the neurons are computed as weighted sums over the outputs from the previous layers.

Since the inputs and the weights are random variables, the activations are random variables as well.

Given a sufficiently large number of input neurons, we can apply a central limit argument and approximate the activation distributions with Gaussians N (a DISPLAYFORM0 ).

For computational convenience, we further assume that the activations within a layer are independent.

Assuming that the inputs x l−1 j and the weights w l ij are independent, we have DISPLAYFORM1 where DISPLAYFORM2 In case of l = 1, we assume no variance at the inputs and thus the second term of σ a l i in (6) cancels.

In the next step, the Gaussian distributions N (a DISPLAYFORM3 ) over the activations are transformed by the sign activation function.

The expectation of the resulting Bernoulli distribution with values x ∈ {−1, 1} of the sign activation function is given by µ DISPLAYFORM4 2 ) where erf denotes the error function.

The raw second moment as needed for (6) is µ (x l i ) 2 = 1.

After iterating this approximation up to the last layer, it remains to calculate the expectation of the log-softmax with respect to the Gaussian approximation of the output activations in (5).

Since this integral does not allow for an analytic solution, we approximate the log-softmax by its second-order Taylor approximation around the mean µ a L with a diagonal covariance approximation, resulting in DISPLAYFORM0 The maximization of the first term in (7) enforces the softmax output of the true class to be maximized, whereas the second term becomes relevant if there is no output close to one.

For softmax outputs substantially different from zero or one, the product inside the sum is substantially larger than zero and the corresponding variance is penalized.

In short, the second term penalizes large output variances if their corresponding output activation means are large and close to each other.

In this section we provide details of the finite weight sets D l and their corresponding prior and approximate posterior distributions, respectively.

As reported in several papers BID8 BID0 , it appears to be crucial to represent the weights in the first layer using a higher precision.

Therefore, we use three bits for the weights in the first layer and ternary weights in the remaining layers.

We use D 1 = {−0.75, −0.5, . . .

, 0.75} for the first layer which can be represented as fixed point numbers using three bits.

Note that |D 1 | = 7 and we could actually represent one additional value with three bits.

However, for the sake of symmetry around zero, we refrain from doing so as we do not want to introduce a bias towards positive or negative values.

We empirically observed this range of values to perform well for our problems with inputs x ∈ [−1, 1].

The values in D 1 can be scaled with an arbitrary factor at training time without affecting the output of the NN since only the sign of the activations is relevant.

We investigated two different variational distributions q(W 1 ).(i) General distribution: We store for each weight the seven logits corresponding to the unnormalized log-probabilities for each of the seven values.

The normalized probabilities can be recovered using the softmax function.

This simple and most general distribution for finite discrete distributions has the advantage that the model can express a tendency towards individual discrete values.

Consequently, we expect the maximum of the distribution to be a reasonable value that the model explicitly favors.

This is fundamentally different from training full precision networks and quantizing the weights afterwards.

The disadvantage of this approach is that the number of variational parameters and the computation time for the means µ w and variances σ w scales with the size of D 1 .(ii) Discretized Gaussian: To get rid of the dependency of the number of parameters on |D 1 |, we also evaluated a discretized Gaussian.

The distribution is parameterized by a mean m w and a variance v w and the logits of the resulting discrete distribution are given by −(w − m w ) 2 /(2 v w ) for w ∈ D 1 FIG1 ).

We denote this distribution as N D1 (m w , v w ).3 This parameterization has the advantage that only two parameters are sufficient to represent the resulting discrete distribution for an arbitrary size of D 1 .

Furthermore, the resulting distribution is unimodal and neighboring values tend to have similar probabilities which appears natural.

Nevertheless, there is no closedform solution for the mean µ w and variance σ w of this distribution and we have to compute a weighted sum involving the |D 1 | normalized probabilities.

For the prior distribution p(W 1 ), we use for both aforementioned variational distributions the discretized Gaussian N D1 (0, γ) with γ being a tunable hyperparameter.

Computing the KL-divergence KL(q(W 1 )||p(W 1 )) also requires computing a weighted sum over |D 1 | values.

For the remaining layers we use ternary weights w ∈ D l = {−1, 0, 1}. We use a shifted binomial distribution, i.e. w ∼ Binomial(2, w p ) − 1.

This distribution requires only a single parameter w p per weight for the variational distribution.

The mean µ w is given by 2w p − 1 and the variance σ w is given by 2w p (1 − w p ).

This makes the Bernoulli distribution an efficient choice for computing the required moments.

It is convenient to select a binomial prior distribution p(w) = Binomial(2, 0.5) as it is centered at zero and we get KL(q(w)||p(w)) = |D l |(log(2w p )w p + log(2(1 − w p ))(1 − w p )).These favorable properties of the binomial distribution might raise the question why it is not used in the first layer, especially since the required expectations, variances and KL-divergences are available in closed-form independent of the size of D l .

We elaborate more on this in Section 5.

We normalize the activations of layer l by d l−1 .

4 This scales the activation means µ a towards zero and keeps the activation variances σ a independent of the number of incoming neurons.

Consequently, the expectation of the Bernoulli distribution after applying the sign activation function µ x = erf(µ a /(2σ a ) 1 2 ) is less prone to be in the saturated region of the error function and, thus, gradients can flow backwards in the computation graph.

Note that activation normalization influences only the PFP and does not change the classification result of individual NNs W ∼ q(W ) since only the sign of the activation is relevant.

The variational inference objective (1) does not allow to easily trade off between the importance of the expected log-likelihood E q(W ) [log p(D|W )] and the KL term KL(q(W )||p(W )).

This is problematic since there are usually many more NN weights than there are data samples and the 3 Note that the mean µw and the variance σw of N D 1 (mw, vw) are in general different from mw and vw.

4 The activation variance σa is normalized by d l−1 .

KL term tends to be orders of magnitudes larger than the expected log-likelihood.

As a result, the optimization procedure mainly focuses on keeping the approximate posterior q(W ) close to the prior p(W ) whereas the influence of the data is too small.

We propose to counteract this issue by trading off between the expected log-likelihood and the KL term using a convex combination, i.e., DISPLAYFORM0 Here, λ ∈ (0, 1) is a tunable hyperparameter that can be interpreted as creating λ/(1 − λ) copies of the data set D. A similar technique is used in BID17 to avoid getting stuck in poor local optima.

Another approach to counteract this issue is the KL-reweighting scheme proposed in BID2 .

Due to the exponential weight decay, only a few minibatches are influenced by the KL term whereas the vast majority is mostly influenced by the expected log-likelihood.

We evaluated the performance of our model (NN VI) on MNIST (LeCun et al., 1998), variants of the MNIST data set BID14 , and a TIMIT data set BID27 .

Details about the individual data sets can be found in the supplementary material.

We selected a three layer structure with d 1 = d 2 = 1200 hidden units for all experiments.

We evaluated both the general parameterization and the Gaussian parameterization as variational distribution q(W >1 ) for the first layer (Section 3.1), and a binomial variational distribution q(W >1 ) for the following layers (Section 3.2).

We optimized the variational distribution using ADAM BID11 without KL reweighting BID2 , and using rmsprop with KL reweighting, and report the experiment resulting in the better classification performance.

For both optimization algorithms, we employ an exponential decay of the learning rate η where we multiply η after every epoch by a factor α ≤ 1.

We use dropout BID24 with dropout rate p in for the input layer and a common dropout rate p hid for the remaining hidden layers.

We normalize the activation by d l−1 p where p is either p in or p hid to consider dropout in the activation normalization.

We tuned the hyperparameters λ DISPLAYFORM0 hid ∈ [0, 0.8] with 50 iterations of Bayesian optimization BID22 .

We report the results for the single most probable model from the approximate posterior W = arg max W q(W ).

This model is indeed a low precision network that can efficiently be implemented in hardware.

We also report results by computing predictions using the PFP which can be seen as approximating the expected prediction arg max t E q(W ) [p(t|x, W )] as it is desired in Bayesian inference.

We compare our model with real-valued NNs (NN real) trained with batch normalization BID9 , dropout, and ReLU activation function.

We also evaluated the model from BID8 (NN STE) which uses batch normalization, dropout, real-valued weights in the first layer and binary weights w ∈ {−1, 1} in the subsequent layers.

The binary weights and the sign activation function are handled using the STE.

For NN (real) and NN STE, we tuned the hyperparameters η ∈ [10 −4 , 10 DISPLAYFORM1 , and p hid ∈ [0, 0.8] on a separate held-out validation set using 50 iterations of Bayesian optimization.

The results are shown in TAB0 .

Our model (single) performs on par with NN (real) and it outperforms NN STE on the TIMIT data set and the more challenging variants of MNIST with different kinds of background artifacts.

Interestingly, our model outperforms the other models on MNIST Background and MNIST Background Random by quite a large margin which could be due to the Bayesian nature of our model.

The PFP outperforms the single most probable model.

This is no surprise since the PFP uses all the available information from the approximate posterior q(W ).

Overall, the performance of the general variational distribution seems to be slightly better than the discretized Gaussian at the cost of more computational overhead at training time.

On a Nvidia GTX 1080 graphics card, a training epoch on the MNIST data set took approximately 8.8 seconds for NN VI general and 7.5 seconds for NN VI Gauss compared to 1.3 seconds for NN (real).

The computational bottleneck of our method is the first layer since here the moments require computing weighted sums over all discrete values w ∈ D 1 .

Next, we approximate the expected predictions arg max t E q(W ) [p(t|x, W )] by sampling several models W ∼ q(W ).

We demonstrate this on the MNIST Rotated Background data set.

NNs with batch normalization, dropout, sign activation function, real-valued weights in the first layer and binary weights in the remaining layers.

NN VI (our method): 3 bits for the first layer and ternary weights for the remaining layers.

We evaluated the single most probable model, the probabilistic forward pass (pfp), the general 3 bit distribution for the first layer, and the discretized Gaussian for the first layer.

shows the classification error of Bayesian averaging over 1000 NNs sampled from the model with the best PFP performance using the discretized Gaussian.

We see that the performance approaches the PFP which indicates that the PFP is a good approximation to the true expected predictions.

However, the size of the ensemble needed to approach the PFP is quite large and the computation time of evaluating a large ensemble is much larger than a single PFP.

Therefore, we investigated a greedy forward selection strategy, where we sample 100 NNs out of which we include only the NN in the ensemble which leads to the lowest error.

This is shown in FIG1 (c).

Using this strategy results in a slightly better performance than Bayesian averaging.

Most importantly, averaging only a few models results in a decent performance increase while still allowing for faster inference than full precision NNs.

Our NNs obtained by taking the most probable model from q(W ) can be efficiently implemented in hardware.

They require only multiplications with 3 bit fixed point values as opposed to multiplications with floating point values in NN (real) and NN STE.

In the special case of image data, the inputs are also given as 8 bit fixed point numbers.

By scaling the inputs and the weights from the first layer appropriately, this results in an ordinary integer multiplication while leaving the output of the sign activation function unchanged.

In the following layers we only have to compute multiplications as logical XNOR operations and accumulate -1 and +1 values for the activations.

TAB2 shows the fraction of non-zero weights of the best performing single models from TAB0 .

Especially in the input layer where we have our most costly 3 bit weights, there are a lot of zero weights on most data sets.

This can be utilized to further reduce the computational costs.

For example, on the MNIST Background Random data set, evaluating a single NN requires only approximately 23000 integer multiplications and 1434000 XNOR operations instead of approximately 2393000 floating point multiplications.

The presented model has many tunable parameters, especially the type of variational distributions for the individual layers, that heavily influence the behavior in terms of convergence at training time and performance at test time.

The binomial distribution appears to be a natural choice for evenly spaced values with many desirable properties.

It is fully specified by only a single parameter, and its mean, variance, and KL divergence with another binomial has nice analytic expressions.

Furthermore, neighboring values have similar probabilities which rules out odd cases in which, for instance, there is a value with low probability in between of two values with high probability.

Unfortunately, the binomial distribution is not suited for the first layer as here it is crucial to be able to set weights with high confidence to zero.

However, when favoring zero weights by setting w p = 0.5, the variance of the binomial distribution takes on its largest possible value.

This might not be a problem in case predictions are computed as the true expectations with respect to q(W ) as in the PFP, but it results in bad classification errors when deriving a single model from q(W ).

We also observed that using the binomial distribution in deeper layers favor the weights −1 and 1 over 0 (cf.

TAB2 ).

This might indicate that binary weights w ∈ {−1, 1} using a Bernoulli distribution could be sufficient, but in our experiments we observed this to perform worse.

We believe this to stem partly from the importance of the zero weight and partly from the larger variance of 4w p (1 − w p ) of the Bernoulli distribution compared to the variance of 2w p (1 − w p ) of the binomial distribution.

Furthermore, there is a general issue with the sign activation functions if the activations are close to zero.

In this case, a small change to the inputs can cause the corresponding neuron to take on a completely different value which might have a large impact on the following layers of the NN.

We found dropout to be a very helpful tool to counteract this issue.

FIG1 shows histograms of the activations of the second hidden layer for both a model trained with dropout and the same model trained without dropout.

We can see that without dropout the activations are much closer to zero whereas dropout introduces a much larger spread of the activations and even causes the histogram to decrease slightly in the vicinity of zero.

Thus, the activations are much more often in regions that are stable with respect to changes of their inputs which makes them more robust.

We believe that such regularization techniques are crucial if the sign activation function is used.

We introduced a method to infer NNs with low precision weights.

As opposed to existing methods, our model neither quantizes the weights of existing full precision NNs nor does it rely on heuristics to compute "approximated" gradients of functions whose gradient is zero almost everywhere.

We perform variational inference to obtain a distribution over a discrete weight space from which we subsequently derive a single discrete-valued NN or a small ensemble of discrete-valued NNs.

Our method propagates probabilities through the network which results in a well defined function that allows us to optimize the discrete distribution even for the sign activation function.

The weights in the first layer are modeled using fixed point values with 3 bits precision and the weights in the remaining layers have values w ∈ {−1, 0, 1}.

This reduces costly floating point multiplications to cheaper multiplications with fixed point values of 3 bits precision in the first layer, and logical XNOR operations in the following layers.

In general, our approach allows flexible bit-widths for each individual layer.

We have shown that the performance of our model is on par with state of the art methods that use a higher precision for the weights.

Furthermore, our model exhibits a large amount of sparsity that can be utilized to further reduce the computational overhead.

A DATA SETS

The MNIST data set BID15 contains grayscale images of size 28 × 28 showing handwritten digits.

It is split into 50000 training samples, 10000 validation samples, and 10000 test samples.

The task is to classify the images to digits.

The pixel intensities are normalized to the range [−1, 1] by dividing through 128 and subtracting 1.

We use the MNIST data set in the permutationinvariant setting where the model is not allowed to use prior knowledge about the image structure, i.e., convolutional NNs are not allowed.

The variants of the MNIST data set BID14 contain images of size 28 × 28 showing images of the original MNIST data set that have been transformed by various operations in order to obtain more challenging data sets.

The variants of the MNIST data set are split into 10000 training samples, 2000 validation samples and 50000 test samples.

In particular, there are the following variants:• MNIST Basic: This data set has not been transformed.

The data set is merely split differently into training, validation, and test set, respectively.• MNIST Background: The background pixels of the images have been replaced by random image patches.• MNIST Background Random: The background pixels of the images have been set to a uniformly random pixel value.• MNIST Rotated: The images are randomly rotated.• MNIST Rotated Background: The transformations from MNIST Rotated and MNIST Background are combined Some samples of the individual data sets are shown in FIG3 .

We also normalized the pixel intensities of these data sets to lie in the range [−1, 1].

The TIMIT data set BID27 contains samples of 92 features representing a phonetic segment.

The task is to classify the phonetic segment to one of 39 phonemes.

The data is split into 140173 training samples, 50735 validation samples (test) and 7211 test samples (core test).

Details on data preprocessing can be found in BID5 .

We normalized the features to have zero mean and unit variance.

@highlight

Variational Inference for infering a discrete distribution from which a low-precision neural network is derived