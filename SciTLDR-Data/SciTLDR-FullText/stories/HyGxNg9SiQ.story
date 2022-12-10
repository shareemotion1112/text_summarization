To make deep neural networks feasible in resource-constrained environments (such as mobile devices), it is beneficial to quantize models by using low-precision weights.

One common technique for quantizing neural networks is the straight-through gradient method, which enables back-propagation through the quantization mapping.

Despite its empirical success, little is understood about why the straight-through gradient method works.

Building upon a novel observation that the straight-through gradient method is in fact identical to the well-known Nesterov’s dual-averaging algorithm on a quantization constrained optimization problem, we propose a more principled alternative approach, called ProxQuant , that formulates quantized network training as a regularized learning problem instead and optimizes it via the prox-gradient method.

ProxQuant does back-propagation on the underlying full-precision vector and applies an efficient prox-operator in between stochastic gradient steps to encourage quantizedness.

For quantizing ResNets and LSTMs, ProxQuant outperforms state-of-the-art results on binary quantization and is on par with state-of-the-art on multi-bit quantization.

For binary quantization, our analysis shows both theoretically and experimentally that ProxQuant is more stable than the straight-through gradient method (i.e. BinaryConnect), challenging the indispensability of the straight-through gradient method and providing a powerful alternative.

change metric.

We present the main ingredients of our contribution in this extended abstract.

See the Appendices B

for the prior work, C for the notation, A and D for the motivation and preliminary discussions about 50 the straight-through gradient method and prox operators.

2 Quantized net training via regularized learning 52 We propose the PROXQUANT algorithm, which adds a quantization-inducing regularizer onto the 53 loss and optimizes via the (non-lazy) prox-gradient method with a finite λ.

The prototypical version 54 of PROXQUANT is described in Algorithm 1.

Require: Regularizer R that induces desired quantizedness, initialization θ 0 , learning rates {η t } t≥0 , regularization strengths {λ t } t≥0 while not converged do Perform the prox-gradient step DISPLAYFORM0 = prox ηtλtR θ t − η t ∇L(θ t ) .The inner SGD step in eq. (2) can be replaced by any preferred stochastic optimization method such as Momentum SGD or Adam [Kingma and Ba, 2014] .

end whileCompared to usual full-precision training, PROXQUANT only adds a prox step after each stochastic Table 1 : Top-1 classification error of quantized ResNets on CIFAR-10.

Performance is reported in mean(std) over 4 runs, where for PQ-T we report in addition the best of 4 (Bo4).

We now show that BinaryConnect has a very stringent convergence condition.

Consider the Bina-to find a parameter θ ∈ {±1} d with low loss, the algorithm only has access to stochastic gradients at {±1} d .

As this is a discrete set, a priori, gradients in this set do not necessarily contain any

(a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " DISPLAYFORM0 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " DISPLAYFORM1 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " DISPLAYFORM2 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > rL(✓t) < l a t e x i t s h a 1 _ b a s e 6 4 = " k M y 6 L o 8 X P y Figure 1: (a) Comparison of the straight-through gradient method and our PROXQUANT method.

The straightthrough method computes the gradient at the quantized vector and performs the update at the original real vector; PROXQUANT performs a gradient update at the current real vector followed by a prox step which encourages quantizedness.

(b) A two-function toy failure case for BinaryConnect.

The two functions are f1(x) = |x + 0.5| − 0.5 (blue) and f−1(x) = |x − 0.5| − 0.5 (orange).

The derivatives of f1 and f−1 coincide at {−1, 1}, so any algorithm that only uses this information will have identical behaviors on these two functions.

However, the minimizers in {±1} are x 1 = −1 and x −1 = 1, so the algorithm must fail on one of them.

DISPLAYFORM3

sparsification, nearest-neighbor clustering, and Huffman coding.

This architecture is then made into 173 a specially designed hardware for efficient inference [Han et al., 2016] .

In a parallel line of work, parameter, but might be hard to tune due to the instability of the inner maximization problem.

While preparing this manuscript, we discovered the independent work of Carreira-Perpinán [2017] ,

Carreira-Perpinán and Idelbayev [2017] .

They formulate quantized network training as a constrained DISPLAYFORM0 This enables the real vector θ to move in the entire Euclidean space, and taking q(θ) at the end of training gives a valid quantized model.

Such a customized back-propagation rule yields good empiri-227 cal performance in training quantized nets and has thus become a standard practice [Courbariaux 228 et al., 2015 , Zhu et al., 2016 , Xu et al., 2018 .

However, as we have discussed, it is information 229 theoretically unclear how the straight-through method works, and it does fail on very simple convex

Lipschitz functions (Figure 1b) .

D.2 Straight-through gradient as lazy projection

Our first observation is that the straight-through gradient method is equivalent to Nesterov's dual-

averaging method, or a lazy projected SGD [Xiao, 2010] .

In the binary case, we wish to minimize 234 L(θ) over Q = {±1} d , and the lazy projected SGD proceeds as DISPLAYFORM0 Written compactly, this is θ t+1 = θ t −η t ∇L(θ)| θ=q(θt) , which is exactly the straight-through gradient 236 method: take the gradient at the quantized vector and perform the update on the original real vector.

We take a broader point of view that a projection is also a limiting proximal operator with a suitable 239 regularizer, to allow more generality and to motivate our proposed algorithm.

Given any set Q, one 240 could identify a regularizer R : R d → R ≥0 such that the following hold: DISPLAYFORM0 In the case Q = {±1} d for example, one could take DISPLAYFORM1 The proximal operator (or prox operator) [Parikh and Boyd, 2014] with respect to R and strength DISPLAYFORM2 In the limiting case λ = ∞, the argmin has to satisfy R(θ) = 0, i.e. θ ∈ Q, and the prox operator is 245 to minimize θ − θ 0 2 2 over θ ∈ Q, which is the Euclidean projection onto Q. Hence, projection is 246 also a prox operator with λ = ∞, and the straight-through gradient estimate is equivalent to a lazy 247 proximal gradient descent with and λ = ∞.

While the prox operator with λ = ∞ correponds to "hard" projection onto the discrete set Q, when 249 λ < ∞ it becomes a "soft" projection that moves towards Q. Compared with the hard projection, 250 a finite λ is less aggressive and has the potential advantage of avoiding overshoot early in training.

Further, as the prox operator does not strictly enforce quantizedness, it is in principle able to query 252 the gradients at every point in the space, and therefore has access to more information than the 253 straight-through gradient method.

E Details on the PROXQUANT algorithm 255 E.1 Regularization for model quantization

We define a flexible class of quantization-inducing regularizers through "distance to the quantized 257 set", derive efficient algorithms of their corresponding prox operator, and propose a homotopy method 258 for choosing the regularization strengths.

Our regularization perspective subsumes most existing 259 algorithms for model-quantization (e.g., [Courbariaux et al., 2015 , Han et al., 2015 , Xu et al., 2018 260 as limits of certain regularizers with strength λ → ∞. Our proposed method can be viewed as a 261 principled generalization of these methods to λ < ∞.

Let Q ⊂ R d be a set of quantized parameter vectors.

An ideal regularizer for quantization would be 263 to vanish on Q and reflect some type of distance to Q when θ / ∈ Q. To achieve this, we propose L 1

and L 2 regularizers of the form DISPLAYFORM0 This is a highly flexible framework for designing regularizers, as one could specify any Q and choose

between L 1 and L 2 .

Specifically, Q encodes certain desired quantization structure.

By appropriately 267 choosing Q, we can specify which part of the parameter vector to quantize 1 , the number of bits to 268 quantize to, whether we allow adaptively-chosen quantization levels and so on.

The choice of distance metrics will result in distinct properties in the regularized solutions.

For 270 example, choosing the L 1 version leads to non-smooth regularizers that induce exact quantizedness 271 in the same way that L 1 norm regularization induces sparsity [Tibshirani, 1996] , whereas choosing 272 the squared L 2 version leads to smooth regularizers that induce quantizedness "softly".

In the following, we present a few examples of regularizers under our framework eq. (7) which induce 274 binary weights, ternary weights and multi-bit quantization.

We will also derive efficient algorithms

(or approximation heuristics) for solving the prox operators corresponding to these regularizers,

which generalize the projection operators used in the straight-through gradient algorithms.

Binary neural nets In a binary neural net, the entries of θ are in {±1}. A natural choice would be DISPLAYFORM0 This is exactly the binary regularizer R bin that we discussed earlier in eq. (6).

FIG4 plots the 280 W-shaped one-dimensional component of R bin from which we see its effect for inducing {±1} 281 quantization in analog to L 1 regularization for inducing exact sparsity.

The prox operator with respect to R bin , despite being a non-convex 283 optimization problem, admits a simple analytical solution: DISPLAYFORM1 DISPLAYFORM2 1 Empirically, it is advantageous to keep the biases of each layers and the BatchNorm layers at full-precision, which is often a negligible fraction, say 1/ √ d of the total number of parameters alternating quantizer of [Xu et al., 2018] : Bα = q alt ( θ).

Together, the prox operator generalizes the alternating minimization procedure in [Xu et al., 2018] , as 300 λ governs a trade-off between quantization and closeness to θ.

To see that this is a strict generalization,

note that for any λ the solution of eq. (12) will be an interpolation between the input θ and its Euclidean 302 projection to Q. As λ → +∞, the prox operator collapses to the projection.

Ternary quantization Ternary quantization is a variant of 2-bit quantization, in which weights are 304 constrained to be in {−α, 0, β} for real values α, β > 0.

For ternary quantization, we use an approximate version of the alternating prox operator eq. FORMULA0 : DISPLAYFORM0 by initializing at θ = θ and repeating DISPLAYFORM1 where q is the ternary quantizer defined as DISPLAYFORM2 This is a straightforward extension of the TWN quantizer [Li and Liu, 2016] Recall that the larger λ t is, the more aggressive θ t+1 will move towards the quantized set.

An ideal 313 choice would be to (1) force the net to be exactly quantized upon convergence, and (2) not be too 314 aggressive such that the quantized net at convergence is sub-optimal.

We let λ t be a linearly increasing sequence, i.e. λ t := λ · t for some hyper-parameter λ > 0 which 316 we term as the regularization rate.

With this choice, the stochastic gradient steps will start off 317 close to full-precision training and gradually move towards exact quantizedness, hence the name 318 "homotopy method".

The parameter λ can be tuned by minimizing the validation loss, and controls 319 the aggressiveness of falling onto the quantization constraint.

There is nothing special about the 320 linear increasing scheme, but it is simple enough and works well as we shall see in the experiments.

Problem setup We perform language modeling with LSTMs Hochreiter and Schmidhuber [1997] 323 on the Penn Treebank (PTB) dataset [Marcus et al., 1993] , which contains 929K training tokens,

73K validation tokens, and 82K test tokens.

Our model is a standard one-hidden-layer LSTM with 325 embedding dimension 300 and hidden dimension 300.

We train quantized LSTMs with the encoder, 326 transition matrix, and the decoder quantized to k-bits for k ∈ {1, 2, 3}. The quantization is performed 327 in a row-wise fashion, so that each row of the matrix has its own codebook {α 1 , . . .

, α k }.

multi-bit quantization, we also report the results for binary LSTMs (weights in {±1}), comparing

BinaryConnect and our PROXQUANT-Binary.

Result We report the perplexity-per-word (PPW, lower is better) in TAB3 .

The performance 337 of PROXQUANT is comparable with the Straight-through gradient method.

On Binary LSTMs,

PROXQUANT-Binary beats BinaryConnect by a large margin.

These results demonstrate that PROX-

QUANT offers a powerful alternative for training recurrent networks.

We experimentally compare the training dynamics of PROXQUANT-Binary and BinaryConnect In R d , the space of all full-precision parameters, the sign change is a natural distance metric that 345 represents the closeness of the binarization of two parameters.

Recall in our CIFAR-10 experiments (Section 3.1), for both BinaryConnect and PROXQUANT, we 347 initialize at a good full-precision net θ 0 and stop at a converged binary network θ ∈ {±1} d .

We 348 are interested in SignChange(θ 0 , θ t ) along the training path, as well as SignChange(θ 0 , θ), i.e. the 349 distance of the final output model to the initialization.

As PROXQUANT converges to higher-performance solutions than BinaryConnect, we expect that if

we run both methods from a same warm start, the sign change of PROXQUANT should be higher than 352 that of BinaryConnect, as in general one needs to travel farther to find a better net.

However, we find that this is not the case: PROXQUANT produces binary nets with both lower sign BinaryConnect never stop changing until we manually freeze the signs at epoch 400.

G.1 Detailed sign change results on ResNet-20 362 2 We thank Xu et al. [2018] for sharing the implementation of this method through a personal communication.

There is a very clever trick not mentioned in their paper: after computing the alternating quantization q alt (θ), they multiply by a constant 0.3 before taking the gradient; in other words, their quantizer is a rescaled alternating quantizer: θ → 0.3q alt (θ).

This scaling step gives a significant gain in performance -without scaling the PPW is {116.7, 94.3, 87.3} for {1, 2, 3} bits.

In contrast, our PROXQUANT does not involve a scaling step and achieves better PPW than this unscaled ALT straight-through method.

BC 9.664, 9.430, 9.198, 9.663 0.386, 0.377, 0.390, 0.381 (8.06) PQ-B 9.

058, 8.901, 9.388, 9.237 0.288, 0.247, 0.284, 9.530, 9.623, 10.370 0.376, 0.379, 0.382, 0.386 (8.31 ) 9.474, 9.410, 9.370 0.291, 0.287, 0.289, 9.558, 9.538, 9.328 0.360, 0.357, 0.359, 0.360 (7.73) PQ-B 9.

284, 8.866, 9.301, 8.884 0.275, 0.276, 0.276, 0.275

@highlight

A principled framework for model quantization using the proximal gradient method.