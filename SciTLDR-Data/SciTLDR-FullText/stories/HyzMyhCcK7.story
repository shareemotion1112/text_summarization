To make deep neural networks feasible in resource-constrained environments (such as mobile devices), it is beneficial to quantize models by using low-precision weights.

One common technique for quantizing neural networks is the straight-through gradient method, which enables back-propagation through the quantization mapping.

Despite its empirical success, little is understood about why the straight-through gradient method works.

Building upon a novel observation that the straight-through gradient method is in fact identical to the well-known Nesterov’s dual-averaging algorithm on a quantization constrained optimization problem, we propose a more principled alternative approach, called ProxQuant , that formulates quantized network training as a regularized learning problem instead and optimizes it via the prox-gradient method.

ProxQuant does back-propagation on the underlying full-precision vector and applies an efficient prox-operator in between stochastic gradient steps to encourage quantizedness.

For quantizing ResNets and LSTMs, ProxQuant outperforms state-of-the-art results on binary quantization and is on par with state-of-the-art on multi-bit quantization.

We further perform theoretical analyses showing that ProxQuant converges to stationary points under mild smoothness assumptions, whereas variants such as lazy prox-gradient method can fail to converge in the same setting.

Deep neural networks (DNNs) have achieved impressive results in various machine learning tasks BID6 .

High-performance DNNs typically have over tens of layers and millions of parameters, resulting in a high memory usage and a high computational cost at inference time.

However, these networks are often desired in environments with limited memory and computational power (such as mobile devices), in which case we would like to compress the network into a smaller, faster network with comparable performance.

A popular way of achieving such compression is through quantization -training networks with lowprecision weights and/or activation functions.

In a quantized neural network, each weight and/or activation can be representable in k bits, with a possible codebook of negligible additional size compared to the network itself.

For example, in a binary neural network (k = 1), the weights are restricted to be in {±1}. Compared with a 32-bit single precision float, a quantized net reduces the memory usage to k/32 of a full-precision net with the same architecture BID7 BID4 BID19 BID13 BID26 BID27 .

In addition, the structuredness of the quantized weight matrix can often enable faster matrixvector product, thereby also accelerating inference BID13 .Typically, training a quantized network involves (1) the design of a quantizer q that maps a full-precision parameter to a k-bit quantized parameter, and (2) the straight-through gradient method BID4 that enables back-propagation from the quantized parameter back onto the original full-precision parameter, which is critical to the success of quantized network training.

With quantizer q, an iterate of the straight-through gradient method (see FIG7 proceeds Code available at https://github.com/allenbai01/ProxQuant.

as θ t+1 = θ t − η t ∇L(θ)| θ=q(θt) , and q( θ) (for the converged θ) is taken as the output model.

For training binary networks, choosing q(·) = sign(·) gives the BinaryConnect method BID4 .Though appealingly simple and empirically effective, it is information-theoretically rather mysterious why the straight-through gradient method works well, at least in the binary case: while the goal is to find a parameter θ ∈ {±1} d with low loss, the algorithm only has access to stochastic gradients at {±1} d .

As this is a discrete set, a priori, gradients in this set do not necessarily contain any information about the function values.

Indeed, a simple one-dimensional example (Figure 1b) shows that BinaryConnect fails to find the minimizer of fairly simple convex Lipschitz functions in {±1}, due to a lack of gradient information in between.rL(q(✓t)) DISPLAYFORM0 0 B N 6 N u 6 N R + P F e F 2 O 5 o z V z j H 6 A e P t E 7 v e l p c = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " k 8 o W Y E M 4 F 9 J U h g q 5 4 4 1 T N Q W 4 X m E = " > A A A B / n i c d V D L S g M x F M 3 U V 6 2 v U X H l J l g E V 2 W m t e 1 0 V 3 D j s o J 9 Q K e U T C Z t Q z M P k j t C G Q r + i h s X i r j 1 O 9 z 5 N 2 b a C i p 6 I O R w z r 3 k 5 H i x 4 A o s 6 8 P I r a 1 v b G 7 l t w s 7 u 3 v 7 DISPLAYFORM1 0 B N 6 N u 6 N R + P F e F 2 O 5 o z V z j H 6 A e P t E 7 v e l p c = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " k 8 o W Y E M 4 F 9 J U h g q 5 4 4 1 T N Q W 4 X m E = " > A A A B / n i c d V D L S g M x F M 3 U V 6 2 v U X H l J l g E V 2 W m t e 1 0 V 3 D j s o J 9 Q K e U T C Z t Q z M P k j t C G Q r + i h s X i r j 1 O 9 z 5 N 2 b a C i p 6 I O R w z r 3 k 5 H i x 4 A o s 6 8 P I r a 1 v b G 7 l t w s 7 u 3 v 7 DISPLAYFORM2 0 B N 6 N u 6 N R + P F e F 2 O 5 o z V z j H 6 A e P t E 7 v e l p c = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " k 8 o W Y E M 4 F 9 J U h g q 5 4 4 1 T N Q W 4 X m E = " > A A A B / n i c d V D L S g M x F M 3 U V 6 2 v U X H l J l g E V 2 W m t e 1 0 V 3 D j s o J 9 Q K e U T C Z t Q z M P k j t C G Q r + i h s X i r j 1 O 9 z 5 N 2 b a C i p 6 I O R w z r 3 k 5 H i x 4 A o s 6 8 P I r a 1 v b G 7 l t w s 7 u 3 v 7 DISPLAYFORM3 0 B N 6 N u 6 N R + P F e F 2 O 5 o z V z j H 6 A e P t E 7 v e l p c = < / l a t e x i t > q(✓t) DISPLAYFORM4 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " DISPLAYFORM5 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " DISPLAYFORM6 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " DISPLAYFORM7 a E H 9 I S e j X v j 0 X g x X h e l K 8 a y 5 w j 9 g P H 2 C a M i n A U = < / l a t e x i t > rL(✓t) < l a t e x i t s h a 1 _ b a s e 6 4 = " k M y 6 L o 8 X P y through method computes the gradient at the quantized vector and performs the update at the original real vector; PROXQUANT performs a gradient update at the current real vector followed by a prox step which encourages quantizedness.

(b) A two-function toy failure case for BinaryConnect.

The two functions are f1(x) = |x + 0.5| − 0.5 (blue) and f−1(x) = |x − 0.5| − 0.5 (orange).

The derivatives of f1 and f−1 coincide at {−1, 1}, so any algorithm that only uses this information will have identical behaviors on these two functions.

However, the minimizers in {±1} are x 1 = −1 and x −1 = 1, so the algorithm must fail on one of them.

DISPLAYFORM8 DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 DISPLAYFORM12 DISPLAYFORM13 In this paper, we formulate the problem of model quantization as a regularized learning problem and propose to solve it with a proximal gradient method.

Our contributions are summarized as follows.• We present a unified framework for defining regularization functionals that encourage binary, ternary, and multi-bit quantized parameters, through penalizing the distance to quantized sets (see Section 3.1).

For binary quantization, the resulting regularizer is a W -shaped non-smooth regularizer, which shrinks parameters towards either −1 or 1 in the same way that the L 1 norm regularization shrinks parameters towards 0.• We propose training quantized networks using PROXQUANT (Algorithm 1) -a stochastic proximal gradient method with a homotopy scheme.

Compared with the straightthrough gradient method, PROXQUANT has access to additional gradient information at non-quantized points, which avoids the problem in FIG7 and its homotopy scheme prevents potential overshoot early in the training (Section 3.2).• We demonstrate the effectiveness and flexibility of PROXQUANT through systematic experiments on (1) image classification with ResNets (Section 4.1); (2) language modeling with LSTMs (Section 4.2).

The PROXQUANT method outperforms the state-of-the-art results on binary quantization and is comparable with the state-of-the-art on ternary and multi-bit quantization.• We perform a systematic theoretical study of quantization algorithms, showing that our PROXQUANT (standard prox-gradient method) converges to stataionary points under mild smoothness assumptions (Section 5.1), where as lazy prox-gradient method such as BinaryRelax BID25 fails to converge in general (Section 5.2).

Further, we show that BinaryConnect has a very stringent condition to converge to any fixed point (Section 5.3), which we verify through a sign change experiment (Appendix C).

Methodologies BID7 propose Deep Compression, which compresses a DNN via sparsification, nearest-neighbor clustering, and Huffman coding.

This architecture is then made into a specially designed hardware for efficient inference .

In a parallel line of work, BID4 propose BinaryConnect that enables the training of binary neural networks, and BID15 ; BID27 extend this method into ternary quantization.

Training and inference on quantized nets can be made more efficient by also quantizing the activation BID13 BID19 BID26 , and such networks have achieved impressive performance on large-scale tasks such as ImageNet classification BID19 BID27 and object detection BID24 .

In the NLP land, quantized language models have been successfully trained using alternating multi-bit quantization BID23 .Theories BID16 prove the convergence rate of stochastic rounding and BinaryConnect on convex problems and demonstrate the advantage of BinaryConnect over stochastic rounding on non-convex problems.

BID0 demonstrate the effectiveness of binary networks through the observation that the angles between high-dimensional vectors are approximately preserved when binarized, and thus high-quality feature extraction with binary weights is possible.

BID5 show a universal approximation theorem for quantized ReLU networks.

Principled methods BID20 perform model quantization through a Wasserstein regularization term and minimize via the adversarial representation, similar as in Wasserstein GANs BID1 .

Their method has the potential of generalizing to other generic requirements on the parameter, but might be hard to tune due to the instability of the inner maximization problem.

Prior to our work, a couple of proximal or regularization based quantization algorithms were proposed as alternatives to the straight-through gradient method, which we now briefly review and compare with.

BID25 propose BinaryRelax, which corresponds to a lazy proximal gradient descent.

BID12 BID11 propose a proximal Newton method with a diagonal approximate Hessian.

Carreira-Perpinán (2017); Carreira-Perpinán & Idelbayev (2017) formulate quantized network training as a constrained optimization problem and propose to solve them via augmented Lagrangian methods.

Our algorithm is different with all the aformentioned work in using the non-lazy and "soft" proximal gradient descent with a choice of either 1 or 2 regularization, whose advantage over lazy prox-gradient methods is demonstrated both theoretically (Section 5) and experimentally (Section 4.1 and Appendix C).

The optimization difficulty of training quantized models is that they involve a discrete parameter space and hence efficient local-search methods are often prohibitive.

For example, the problem of training a binary neural network is to minimize L(θ) for θ ∈ {±1} d .

Projected SGD on this set will not move unless with an unreasonably large stepsize BID16 , whereas greedy nearestneighbor search requires d forward passes which is intractable for neural networks where d is on the order of millions.

Alternatively, quantized training can also be cast as minimizing L(q(θ)) for θ ∈ R d and an appropriate quantizer q that maps a real vector to a nearby quantized vector, but θ → q(θ) is often non-differentiable and piecewise constant (such as the binary case q(·) = sign(·)), and thus back-propagation through q does not work.

The pioneering work of BinaryConnect BID4 proposes to solve this problem via the straight-through gradient method, that is, propagate the gradient with respect to q(θ) unaltered to θ, i.e. to let ∂L ∂θ := ∂L ∂q(θ) .

One iterate of the straight-through gradient method (with the SGD optimizer) is DISPLAYFORM0 This enables the real vector θ to move in the entire Euclidean space, and taking q(θ) at the end of training gives a valid quantized model.

Such a customized back-propagation rule yields good empirical performance in training quantized nets and has thus become a standard practice BID4 BID27 BID23 .

However, as we have discussed, it is information theoretically unclear how the straight-through method works, and it does fail on very simple convex Lipschitz functions FIG7 ).

Our first observation is that the straight-through gradient method is equivalent to a dual-averaging method, or a lazy projected SGD BID22 .

In the binary case, we wish to minimize L(θ) over Q = {±1} d , and the lazy projected SGD proceeds as DISPLAYFORM0 Written compactly, this is θ t+1 = θ t − η t ∇L(θ)| θ=q(θt) , which is exactly the straight-through gradient method: take the gradient at the quantized vector and perform the update on the original real vector.

We take a broader point of view that a projection is also a limiting proximal operator with a suitable regularizer, to allow more generality and to motivate our proposed algorithm.

Given any set Q, one could identify a regularizer R : R d → R ≥0 such that the following hold: DISPLAYFORM0 In the case Q = {±1} d for example, one could take DISPLAYFORM1 The proximal operator (or prox operator) BID18 with respect to R and strength λ > 0 is prox λR (θ) := arg min DISPLAYFORM2 In the limiting case λ = ∞, the argmin has to satisfy R(θ) = 0, i.e. θ ∈ Q, and the prox operator is to minimize θ − θ 0 2 2 over θ ∈ Q, which is the Euclidean projection onto Q. Hence, projection is also a prox operator with λ = ∞, and the straight-through gradient estimate is equivalent to a lazy proximal gradient descent with and λ = ∞.While the prox operator with λ = ∞ correponds to "hard" projection onto the discrete set Q, when λ < ∞ it becomes a "soft" projection that moves towards Q. Compared with the hard projection, a finite λ is less aggressive and has the potential advantage of avoiding overshoot early in training.

Further, as the prox operator does not strictly enforce quantizedness, it is in principle able to query the gradients at every point in the space, and therefore has access to more information than the straight-through gradient method.

We propose the PROXQUANT algorithm, which adds a quantization-inducing regularizer onto the loss and optimizes via the (non-lazy) prox-gradient method with a finite λ.

The prototypical version of PROXQUANT is described in Algorithm 1.

Require: Regularizer R that induces desired quantizedness, initialization θ 0 , learning rates {η t } t≥0 , regularization strengths {λ t } t≥0 while not converged do Perform the prox-gradient step DISPLAYFORM0 The inner SGD step in eq. (4) can be replaced by any preferred stochastic optimization method such as Momentum SGD or Adam BID14 .

end whileCompared to usual full-precision training, PROXQUANT only adds a prox step after each stochastic gradient step, hence can be implemented straightforwardly upon existing full-precision training.

As the prox step does not need to know how the gradient step is performed, our method adapts to other stochastic optimizers as well such as Adam.

In the remainder of this section, we define a flexible class of quantization-inducing regularizers through "distance to the quantized set", derive efficient algorithms of their corresponding prox operator, and propose a homotopy method for choosing the regularization strengths.

Our regularization perspective subsumes most existing algorithms for model-quantization (e.g., BID4 BID7 BID23 ) as limits of certain regularizers with strength λ → ∞. Our proposed method can be viewed as a principled generalization of these methods to λ < ∞ with a non-lazy prox operator.

Let Q ⊂ R d be a set of quantized parameter vectors.

An ideal regularizer for quantization would be to vanish on Q and reflect some type of distance to Q when θ / ∈ Q. To achieve this, we propose L 1 and L 2 regularizers of the form DISPLAYFORM0 This is a highly flexible framework for designing regularizers, as one could specify any Q and choose between L 1 and L 2 .

Specifically, Q encodes certain desired quantization structure.

By appropriately choosing Q, we can specify which part of the parameter vector to quantize 1 , the number of bits to quantize to, whether we allow adaptively-chosen quantization levels and so on.

The choice between {L 1 , L 2 } will encourage {"hard","soft"} quantization respectively, similar as in standard regularized learning BID21 .In the following, we present a few examples of regularizers under our framework eq. (5) which induce binary weights, ternary weights and multi-bit quantization.

We will also derive efficient algorithms (or approximation heuristics) for solving the prox operators corresponding to these regularizers, which generalize the projection operators used in the straight-through gradient algorithms.

Binary neural nets In a binary neural net, the entries of θ are in {±1}. A natural choice would be taking DISPLAYFORM1 This is exactly the binary regularizer R bin that we discussed earlier in eq. (3).

Figure 2 plots the W-shaped one-dimensional component of R bin from which we see its effect for inducing {±1} quantization in analog to L 1 regularization for inducing exact sparsity.1 Empirically, it is advantageous to keep the biases of each layers and the BatchNorm layers at full-precision, which is often a negligible fraction, say 1/ √ d of the total number of parameters The prox operator with respect to R bin , despite being a non-convex optimization problem, admits a simple analytical solution: DISPLAYFORM2 We note that the choice of the L 1 version is not unique: the squared L 2 version works as well, whose prox operator is given by (θ + λ sign(θ))/(1 + λ).

See Appendix A.1 for the derivation of these prox operators and the definition of the soft thresholding operator.

Multi-bit quantization with adaptive levels.

Following BID23 , we consider k-bit quantized parameters with a structured adaptively-chosen set of quantization levels, which translates into DISPLAYFORM3 (8) The squared L 2 regularizer for this structure is DISPLAYFORM4 which is also the alternating minimization objective in BID23 .We now derive the prox operator for the regularizer eq. (9).

For any θ, we have DISPLAYFORM5 This is a joint minimization problem in ( θ, B, α), and we adopt an alternating minimization schedule to solve it:(1) Minimize over θ given (B, α), which has a closed-form solution θ = θ+2λBα 1+2λ .

(2) Minimize over (B, α) given θ, which does not depend on θ 0 , and can be done via calling the alternating quantizer of BID23 :

Bα = q alt ( θ).Together, the prox operator generalizes the alternating minimization procedure in BID23 , as λ governs a trade-off between quantization and closeness to θ.

To see that this is a strict generalization, note that for any λ the solution of eq. (10) will be an interpolation between the input θ and its Euclidean projection to Q. As λ → +∞, the prox operator collapses to the projection.

Ternary quantization Ternary quantization is a variant of 2-bit quantization, in which weights are constrained to be in {−α, 0, β} for real values α, β > 0.

We defer the derivation of the ternary prox operator into Appendix A.2.

Recall that the larger λ t is, the more aggressive θ t+1 will move towards the quantized set.

An ideal choice would be to (1) force the net to be exactly quantized upon convergence, and (2) not be too aggressive such that the quantized net at convergence is sub-optimal.

We let λ t be a linearly increasing sequence, i.e. λ t := λ · t for some hyper-parameter λ > 0 which we term as the regularization rate.

With this choice, the stochastic gradient steps will start off close to full-precision training and gradually move towards exact quantizedness, hence the name "homotopy method".

The parameter λ can be tuned by minimizing the validation loss, and controls the aggressiveness of falling onto the quantization constraint.

There is nothing special about the linear increasing scheme, but it is simple enough and works well as we shall see in the experiments.

We evaluate the performance of PROXQUANT on two tasks: image classification with ResNets, and language modeling with LSTMs.

On both tasks, we show that the default straight-through gradient method is not the only choice, and our PROXQUANT can achieve the same and often better results.

Problem setup We perform image classification on the CIFAR-10 dataset, which contains 50000 training images and 10000 test images of size 32x32.

We apply a commonly used data augmentation strategy (pad by 4 pixels on each side, randomly crop to 32x32, do a horizontal flip with probability 0.5, and normalize).

Our models are ResNets of depth 20, 32, 44, and 56 with weights quantized to binary or ternary.

Method We use PROXQUANT with regularizer eq. (3) in the binary case and eqs. FORMULA15 and FORMULA15 in the ternary case, which we respectively denote as PQ-B and PQ-T. We use the homotopy method λ t = λ · t with λ = 10 −4 as the regularization strength and Adam with constant learning rate 0.01 as the optimizer.

We compare with BinaryConnect (BC) for binary nets and Trained Ternary Quantization (TTQ) BID27 for ternary nets.

For BinaryConnect, we train with the recommended Adam optimizer with learning rate decay BID4 (initial learning rate 0.01, multiply by 0.1 at epoch 81 and 122), which we find leads to the best result for BinaryConnect.

For TTQ we compare with the reported results in BID27 .For binary quantization, both BC and our PROXQUANT are initialized at the same pre-trained fullprecision nets (warm-start) and trained for 300 epochs for fair comparison.

For both methods, we perform a hard quantization θ → q(θ) at epoch 200 and keeps training till the 300-th epoch to stabilize the BatchNorm layers.

We compare in addition the performance drop relative to full precision nets of BinaryConnect, BinaryRelax BID25 , and our PROXQUANT.Result The top-1 classification errors for binary quantization are reported in TAB0 .

Our PROX-QUANT consistently yields better results than BinaryConnect.

The performance drop of PROX-QUANT relative to full-precision nets is about 1%, better than BinaryConnect by 0.2% on average and significantly better than the reported result of BinaryRelax.

Results and additional details for ternary quantization are deferred to Appendix B.1.

Problem setup We perform language modeling with LSTMs BID10 on the Penn Treebank (PTB) dataset BID17 , which contains 929K training tokens, 73K validation tokens, and 82K test tokens.

Our model is a standard one-hidden-layer LSTM with embedding dimension 300 and hidden dimension 300.

We train quantized LSTMs with the encoder, transition matrix, and the decoder quantized to k-bits for k ∈ {1, 2, 3}. The quantization is performed in a row-wise fashion, so that each row of the matrix has its own codebook {α 1 , . . .

, α k }.Method We compare our multi-bit PROXQUANT (eq. (10)) to the state-of-the-art alternating minimization algorithm with straight-through gradients BID23 .

Training is initialized at a pre-trained full-precision LSTM.

We use the SGD optimizer with initial learning rate 20.0 and decay by a factor of 1.2 when the validation error does not improve over an epoch.

We train for 80 epochs with batch size 20, BPTT 30, dropout with probability 0.5, and clip the gradient norms to 0.25.

The regularization rate λ is tuned by finding the best performance on the validation set.

In addition to multi-bit quantization, we also report the results for binary LSTMs (weights in {±1}), comparing BinaryConnect and our PROXQUANT-Binary, where both learning rates are tuned on an exponential grid {2.5, 5, 10, 20, 40}.Result We report the perplexity-per-word (PPW, lower is better) in TAB1 .

The performance of PROXQUANT is comparable with the Straight-through gradient method.

On Binary LSTMs, PROXQUANT-Binary beats BinaryConnect by a large margin.

These results demonstrate that PROX-QUANT offers a powerful alternative for training recurrent networks.

In this section, we perform a theoretical study on the convergence of quantization algorithms.

We show in Section 5.1 that our PROXQUANT algorithm (i.e. non-lazy prox-gradient method) converges under mild smoothness assumptions on the problem.

In Section 5.2, we provide a simple example showing that the lazy prox-gradient method fails to converge under the same set of assumptions.

In Section 5.3, we show that BinaryConnect has a very stringent condition for converging to a fixed point.

Our theory demonstrates the superiority of our proposed PROXQUANT over lazy proxgradient type algorithms such as BinaryConnect and BinaryRelax BID25 .

All missing proofs are deferred to Appendix D.Prox-gradient algorithms (both lazy and non-lazy) with a fixed λ aim to solve the problem minimize DISPLAYFORM0 and BinaryConnect can be seen as the limiting case of the above with λ = ∞ (cf.

Section 2.2).

We consider PROXQUANT with batch gradient and constant regularization strength λ t ≡ λ: DISPLAYFORM0 Theorem 5.1 (Convergence of ProxQuant).

Assume that the loss L is β-smooth (i.e. has β-Lipschitz gradients) and the regularizer R is differentiable.

Let F λ (θ) = L(θ) + λR(θ) be the composite objective and assume that it is bounded below by F .

Running ProxQuant with batch gradient ∇L, constant stepsize η t ≡ η = 1 2β and λ t ≡ λ for T steps, we have the convergence guarantee DISPLAYFORM1 where C > 0 is a universal constant.2 We thank BID23 for sharing the implementation of this method through a personal communication.

There is a very clever trick not mentioned in their paper: after computing the alternating quantization q alt (θ), they multiply by a constant 0.3 before taking the gradient; in other words, their quantizer is a rescaled alternating quantizer: θ → 0.3q alt (θ).

This scaling step gives a significant gain in performance -without scaling the PPW is {116.7, 94.3, 87.3} for {1, 2, 3} bits.

In contrast, our PROXQUANT does not involve a scaling step and achieves better PPW than this unscaled ALT straight-through method.

Remark 5.1.

The convergence guarantee requires both the loss and the regularizer to be smooth.

Smoothness of the loss can be satisfied if we use a smooth activation function (such as tanh).

For the regularizer, the quantization-inducing regularizers defined in Section 3.1 (such as the W-shaped regularizer) are non-differentiable.

However, we can use a smoothed version of them that is differentiable and point-wise arbitrarily close to R, which will satisfy the assumptions of Theorem 5.1.

The proof of Theorem 5.1 is deferred to Appendix D.1.

The lazy prox-gradient algorithm (e.g. BinaryRelax BID25 ) for solving problem eq. FORMULA15 is a variant where the gradients are taken at proximal points but accumulated at the original sequence: DISPLAYFORM0 (13) Convergence of the lazy prox-gradient algorithm eq. FORMULA15 is only known to hold for convex problems BID22 ; on smooth non-convex problems it generally does not converge even in an ergodic sense.

We provide a concrete example that satisfies the assumptions in Theorem 5.1 (so that PROXQUANT converges ergodically) but lazy prox-gradient does not converge.

Theorem 5.2 (Non-convergence of lazy prox-gradient).

There exists L and R satisfying the assumptions of Theorem 5.1 such that for any constant stepsize η t ≡ η ≤ 1 2β , there exists some specific initialization θ 0 on which the lazy prox-gradient algorithm eq. (13) oscillates between two non-stataionry points and hence does not converge in the ergodic sense of eq. (12).

Remark 5.2.

Our construction is a fairly simple example in one-dimension and not very adversarial: L(θ) = 1 2 θ 2 and R is a smoothed W-shaped regularizer.

See Appendix D.2 for the details.

For BinaryConnect, the concept of stataionry points is no longer sensible (as the target points {±1} d are isolated and hence every point is stationary).

Here, we consider the alternative definition of convergence as converging to a fixed point and show that BinaryConnect has a very stringent convergence condition.

Consider the BinaryConnect method with batch gradients: DISPLAYFORM0 ) Definition 5.1 (Fixed point and convergence).

We say that s ∈ {±1}d is a fixed point of the BinaryConnect algorithm, if s 0 = s in eq. (14) implies that s t = s for all t = 1, 2, ....

We say that the BinaryConnect algorithm converges if there exists t < ∞ such that s t is a fixed point.

Theorem 5.3.

Assume that the learning rates satisfy DISPLAYFORM1 d is a fixed point for BinaryConnect eq. BID16 in the convex case, whose bound involves a an additive error O(∆) that does not vanish over iterations, where ∆ is the grid size for quantization.

Hence, their result is only useful when ∆ is small.

In contrast, we consider the original BinaryConnect with ∆ = 1, in which case the error makes BID16 's bound vacuous.

The proof of Theorem 5.3 is deferred to Appendix D.3.

We have already seen that such a fixed point s might not exist in the toy example in FIG7 .

In Appendix C, we perform a sign change experiment on CIFAR-10, showing that BinaryConnect indeed fails to converge to a fixed sign pattern, corroborating Theorem 5.3.

In this paper, we propose and experiment with the PROXQUANT method for training quantized networks.

Our results demonstrate that PROXQUANT offers a powerful alternative to the straightthrough gradient method and has theoretically better convergence properties.

For future work, it would be of interest to propose alternative regularizers for ternary and multi-bit PROXQUANT and experiment with our method on larger tasks.

This minimization problem is coordinate-wise separable.

For each θ j , the penalty term remains the same upon flipping the sign, but the quadratic term is smaller when sign( θ j ) = sign(θ j ).

Hence, the solution θ to the prox satisfies that sign(θ j ) = sign(θ j ), and the absolute value satisfies |θ j | = arg min Multiplying by sign(θ j ) = sign(θ j ), we have θ j = SoftThreshold(θ j , sign(θ j ), λ), which gives eq. (7).For the squared L 2 version, by a similar argument, the corresponding regularizer is DISPLAYFORM0 min (θ j − 1) 2 , (θ j + 1) 2 .For this regularizer we have prox λR bin (θ) = arg min DISPLAYFORM1 Using the same argument as in the L 1 case, the solution θ satisfies sign(θ j ) = sign(θ j ), and DISPLAYFORM2 Multiplying by sign(θ j ) = sign(θ j ) gives θ j = θ j + λ sign(θ j ) 1 + λ , or, in vector form, θ = (θ + λ sign(θ))/(1 + λ).

For ternary quantization, we use an approximate version of the alternating prox operator eq. (10): compute θ = prox λR (θ) by initializing at θ = θ and repeating θ = q( θ) and θ = θ + 2λ θ 1 + 2λ ,where q is the ternary quantizer defined as q(θ) = θ + 1{θ ≥ ∆} + θ − 1{θ ≤ −∆}, ∆ = 0.7 d θ 1 , θ + = θ| i:θi≥∆ , θ − = θ| i:θi≤−∆ .

FORMULA15 This is a straightforward extension of the TWN quantizer BID15 ) that allows different levels for positives and negatives.

We find that two rounds of alternating computation in eq. (15) achieves a good performance, which we use in our experiments.

Our models are ResNets of depth 20, 32, and 44.

Ternarized training is initialized at pre-trained full-precision nets.

We perform a hard quantization θ → q(θ) at epoch 400 and keeps training till the600-th epoch to stabilize the BatchNorm layers.

Result The top-1 classification errors for ternary quantization are reported in TAB2 .

Our results are comparable with the reported results of TTQ, 3 and the best performance of our method over 4 runs (from the same initialization) is slightly better than TTQ.

We experimentally compare the training dynamics of PROXQUANT-Binary and BinaryConnect through the sign change metric.

The sign change metric between any θ 1 and θ 2 is the proportion of their different signs, i.e. the (rescaled) Hamming distance: DISPLAYFORM0 In R d , the space of all full-precision parameters, the sign change is a natural distance metric that represents the closeness of the binarization of two parameters.

Recall in our CIFAR-10 experiments (Section 4.1), for both BinaryConnect and PROXQUANT, we initialize at a good full-precision net θ 0 and stop at a converged binary network θ ∈ {±1}d .

We are interested in SignChange(θ 0 , θ t ) along the training path, as well as SignChange(θ 0 , θ), i.e. the distance of the final output model to the initialization.

Our finding is that PROXQUANT produces binary nets with both lower sign changes and higher performances, compared with BinaryConnect.

Put differently, around the warm start, there is a good binary net nearby which can be found by PROXQUANT but not BinaryConnect, suggesting that BinaryConnect, and in general the straight-through gradient method, suffers from higher optimization instability than PROXQUANT.

This finding is consistent in all layers, across different warm starts, and across differnent runs from each same warm start (see FIG11 and TAB3 in Appendix C.1).

This result here is also consistent with Theorem 5.3: the signs in BinaryConnect never stop changing until we manually freeze the signs at epoch 400.

Recall that a function f : R d → R is said to be β-smooth if it is differentiable and ∇f is β-Lipschitz: for all x, y ∈ R d we have ∇f (x) − ∇f (y) 2 ≤ β x − y 2 .

For any β-smooth function, it satisfies the bound f (y) ≤ f (x) + ∇f (x), y − x + β 2 x − y

@highlight

A principled framework for model quantization using the proximal gradient method, with empirical evaluation and theoretical convergence analyses.

@highlight

Proposes ProxQuant method to train neural networks with quantized weights.

@highlight

Proposes solving binary nets and its variants using proximal gradient descent.