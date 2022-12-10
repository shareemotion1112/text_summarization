Deep neural networks are usually huge, which significantly limits the deployment on low-end devices.

In recent years, many weight-quantized models have  been proposed.

They have small storage and fast inference, but training can still be time-consuming.

This can be improved with distributed learning.

To reduce the high communication cost due to worker-server synchronization, recently gradient quantization has also been proposed to train deep networks with full-precision weights.

In this paper, we theoretically study how the combination of both weight and gradient quantization affects convergence.

We show  that (i) weight-quantized models converge to an error related to the weight quantization resolution and weight dimension; (ii) quantizing gradients slows convergence by a factor related to the gradient quantization resolution and dimension; and (iii) clipping the gradient before quantization renders this factor dimension-free, thus allowing the use of fewer bits for gradient quantization.

Empirical experiments confirm the theoretical convergence results, and demonstrate that quantized networks can speed up training and have comparable performance as full-precision networks.

Deep neural networks are usually huge.

The high demand in time and space can significantly limit deployment on low-end devices.

To alleviate this problem, many approaches have been recently proposed to compress deep networks.

One direction is network quantization, which represents each network weight with a small number of bits.

Besides significantly reducing the model size, it also accelerates network training and inference.

Many weight quantization methods aim at approximating the full-precision weights in each iteration BID3 BID20 BID23 BID15 BID19 BID8 .

Recently, loss-aware quantization minimizes the loss directly w.r.t.

the quantized weights BID11 BID10 BID14 , and often achieves better performance than approximation-based methods.

Distributed learning can further speed up training of weight-quantized networks BID6 .

A key challenge is on reducing the expensive communication cost incurred during synchronization of the gradients and model parameters BID17 .

Recently, algorithms that sparsify BID0 BID28 or quantize the gradients BID26 BID29 BID2 have been proposed.

In this paper, we consider quantization of both the weights and gradients in a distributed environment.

Quantizing both weights and gradients has been explored in the DoReFa-Net BID32 , QNN BID12 , WAGE BID30 and ZipML .

We differ from them in two aspects.

First, existing methods mainly consider learning on a single machine, and gradient quantization is used to reduce the computations in backpropagation.

On the other hand, we consider a distributed environment, and use gradient quantization to reduce communication cost and accelerate distributed learning of weight-quantized networks.

Second, while DoReFa-Net, QNN and WAGE show impressive empirical results on the quantized network, theoretical guarantees are not provided.

ZipML provides convergence analysis, but is limited to stochastic weight quantization, square loss with the linear model, and requires the stochastic gradients to be unbiased.

This can be restrictive as most state-of-the-art weight quantization methods BID23 BID20 BID15 BID8 BID11 BID10 ) are deterministic, and the resultant stochastic gradients are biased.

In this paper, we relax the restrictions on the loss function, and study in an online learning setting how the gradient precision affects convergence of weight-quantized networks in a distributed environment.

The main findings are:1.

With either full-precision or quantized gradients, the average regret of loss-aware weight quantization does not converge to zero, but to an error related to the weight quantization resolution ∆ w and dimension d. The smaller the ∆ w or d, the smaller is the error (Theorems 1 and 2).

2.

With either full-precision or quantized gradients, the average regret converges with a O(1/ √ T ) rate to the error, where T is the number of iterations.

However, gradient quantization slows convergence (relative to using full-precision gradients) by a factor related to gradient quantization resolution ∆ g and d. The larger the ∆ g or d, the slower is the convergence FIG0 ).

This can be problematic when (i) the weight quantized model has a large d (e.g., deep networks); and (ii) the communication cost is a bottleneck in the distributed setting, which favors a small number of bits for the gradients, and thus a large ∆ g .

3.

For gradients following the normal distribution, gradient clipping renders the speed degradation mentioned above dimension-free.

However, an additional error is incurred.

The convergence speedup and error are related to how aggressive clipping is performed.

More aggressive clipping results in faster convergence, but a larger error (Theorem 3).

4.

Empirical results show that quantizing gradients significantly reduce communication cost, and gradient clipping makes speed degradation caused by gradient quantization negligible.

With quantized clipped gradients, distributed training of weight-quantized networks is much faster, while comparable accuracy with the use of full-precision gradients is maintained (Section 4).Notations.

For a vector x, √ x is the element-wise square root, x 2 is the element-wise square, Diag(x) returns a diagonal matrix with x on the diagonal, and x y is the element-wise multiplication of vectors x and y. For a matrix Q, x 2 Q = x Qx.

For a matrix X, √ X is the element-wise square root, and diag(X) returns a vector extracted from the diagonal elements of X.

Online learning continually adapts the model with a sequence of observations.

It has been commonly used in the analysis of deep learning optimizers BID7 BID13 BID25 .

At time t, the algorithm picks a model with parameter w t ∈ S, where S is a convex compact set.

The algorithm then incurs a loss f t (w t ).

After T rounds, the performance is usually evaluated by the regret R(T ) = T t=1 f t (w t ) − f t (w * ) and average regret R(T )/T , where w * = arg min w∈S T t=1 f t (w) is the best model parameter in hindsight.

In BinaryConnect BID3 , each weight is binarized using the sign function either deterministically or stochastically.

In ternary-connect BID20 , each weight is stochastically quantized to {−1, 0, 1}. Stochastic weight quantization often suffers severe accuracy degradation, while deterministic weight quantization (as in the binary-weight-network (BWN) BID23 and ternary weight network (TWN) BID15 ) achieves much better performance.

In this paper, we will focus on loss-aware weight quantization, which further improves performance by considering the effect of weight quantization on the loss.

Examples include loss-aware binarization (LAB) BID11 and loss-aware quantization (LAQ) BID10 .

Let the full-precision weights from all L layers in the deep network be w. The corresponding quantized weight is denoted Q w (w) =ŵ, where Q w (·) is the weight quantization function.

At the (t + 1)th iteration, the second-order Taylor expansion of f t (ŵ), i.e., DISPLAYFORM0 DISPLAYFORM1 with g t the stochastic gradient, β 1, and is readily available in popular deep network optimizers such as RMSProp and Adam.

Diag( √v t ) is also an estimate of Diag( diag(H 2 t )) BID4 .

Computationally, the quantized weight is obtained by first performing a preconditioned gradient descent w t+1 = w t − η t Diag( DISPLAYFORM2 t , followed by quantization via solving the following problem: DISPLAYFORM3 For simplicity of notations, we assume that the same scaling parameter α is used for all layers.

Extension to layer-wise scaling is straightforward.

For binarization, S w = {−1, +1}, the weight quantization resolution is ∆ w = 1, and a simple closed-form solution is obtained in BID11 .

DISPLAYFORM4 An efficient approximate solution of (2) is obtained in BID10 .

In a distributed learning environment with data parallelism, the main bottleneck is often on the communication cost due to gradient synchronization.

By quantizing the gradients before synchronization BID26 BID29 , this cost can be significantly reduced.

For example, assuming that the full-precision gradient is 32-bit, the communication cost can be reduced 32/m times when gradients are quantized to m bits.

Most recent gradient quantization methods BID29 require the quantized gradient to be unbiased, and thus use stochastically quantized gradients.

On the other hand, deterministic gradient quantization makes the quantized gradient biased, and the resultant analysis more complex.

In this paper, we consider the more general m-bit stochastic linear quantization : DISPLAYFORM0 where DISPLAYFORM1 The gradient quantization resolution is defined as ∆ g = B r+1 − B r .

The ith element q t,i in q t is equal to B r+1 with probability (|g t,i |/s t − B r ) /(B r+1 − B r ), and B r otherwise.

Here, r is an index satisfying B r ≤ |g t,i |/s t < B r+1 .

Note that Q g (g t ) is an unbiased estimator of g t .

In this section, we consider quantization of both weights and gradients in a distributed environment with N workers using data parallelism.

For easy illustration, we use the parameter server model BID18 in Figure 1 , though it also holds for other configurations such as the AllReduce model BID22 .

At the tth iteration, worker n ∈ {1, 2, . . .

, N } computes the full-precision gradientĝ (n) t w.r.t.

the quantized weight and quantizesĝ DISPLAYFORM0 The quantized gradients are then synchronized and averaged at the parameter server as: DISPLAYFORM1 t .

The server updates the second momentṽ t based ong t , and also the full-precision weight as w t+1 = w t − η t Diag( DISPLAYFORM2 t .

The weight is quantized using loss-aware weight quantization to produceŵ t+1 = Q w (w t+1 ), which is then sent back to all the workers.

Analysis on quantized deep networks has only been performed on models with (i) full-precision gradients and weights quantized by stochastic weight quantization BID5 , or simple deterministic weight quantization using the sign ; (ii) full-precision weights and quantized gradients BID29 BID2 ; DISPLAYFORM0 Figure 1: Distributed weight and gradient quantization with data parallelism.(iii) quantized weights and quantized gradients , but limited to stochastic weight quantization, square loss on linear model (i.e., f t (w t ) = (x t w t −y t ) 2 ) in Section 2.1), and unbiased gradient.

In this paper, we study the more advanced loss-aware weight quantization, with both full-precision and quantized gradients.

As it is deterministic and has biased gradients, the above analysis do not apply here.

Moreover, we do not assume a linear model, and relax the assumptions on f t as:(A1) f t is convex; (A2) f t is twice differentiable with Lipschitz-continuous gradient; and (A3) f t has bounded gradient, i.e., ∇f t (w) ≤ G and ∇f t (w) ∞ ≤ G ∞ for all w ∈ S.These assumptions have been commonly used in convex online learning BID9 BID7 BID13 and quantized networks .

Obviously, the convexity assumption A1 does not hold for deep networks.

However, this facilitates analysis of deep learning models, and has been used in BID13 BID25 BID5 .

Moreover, as will be seen, it helps to explain the empirical behavior in Section 4.As in BID7 BID13 , we assume that w m − w n ≤ D and w m − w n ∞ ≤ D ∞ for all w m , w n ∈ S. Moreover, the learning rate η t decays as η/ √ t, where η is a constant BID9 BID7 BID13 .For simplicity of notations, we denote the full-precision gradient ∇f t (w t ) w.r.t.

the full-precision weight by g t , and the full-precision gradient ∇f t (Q w (w t )) w.r.t.

the quantized weight byĝ t .

As f t is twice differentiable (Assumption A2), using the mean value theorem, there exists p ∈ (0, 1) DISPLAYFORM1 Moreover, let α = max{α 1 , . . .

, α T }, where α t is the scaling parameter in (2) at the tth iteration.

When only weights are quantized, the update for loss-aware weight quantization is DISPLAYFORM0 t , wherev t is the moving average of the (squared) gradientsĝ 2 t in (1).

Theorem 1.

For loss-aware weight quantization with full-precision gradients and η t = η/ √ t, DISPLAYFORM1 DISPLAYFORM2 For standard online gradient descent with the same learning rate scheme, R(T )/T converges to zero at the rate of O(1/ √ T ) BID9 .

From Theorem 1, the average regret converges at the same rate, but only to a nonzero error LD D 2 + dα 2 ∆ 2 w 4 related to the weight quantization resolution ∆ w and dimension d.

When both weights and gradients are quantized, the update for loss-aware weight quantization is DISPLAYFORM0 whereg t is the stochastically quantized gradient Q g (∇f t (Q w (w t ))).

The second momentṽ t is the moving average of the (squared) quantized gradientsg 2 t .

The following Proposition shows that gradient quantization significantly blows up the norm of the quantized gradient relative to its fullprecision counterparts.

Moreover, the difference increases with the gradient quantization resolution ∆ g and dimension d. DISPLAYFORM1 Theorem 2.

For loss-aware weight quantization with quantized gradients and DISPLAYFORM2 DISPLAYFORM3 The regrets in FORMULA13 and FORMULA17 are of the same form and differ only in the gradient used.

Similarly, for the average regrets in FORMULA14 and FORMULA18 , quantizing gradients slows convergence by a factor of DISPLAYFORM4 which is a direct consequence of the blowup in Proposition 1.

These observations can be problematic as (i) deep networks typically have a large d; and (ii) distributed learning prefers using a small number of bits for the gradients, and thus a large ∆ g .

To reduce convergence speed degradation caused by gradient quantization, gradient clipping has been proposed as an empirical solution BID29 .

The gradientĝ t is clipped to Clip(ĝ t ), where DISPLAYFORM0 Here, c is a constant clipping factor, and σ is the standard deviation of elements inĝ t .

The update then becomes DISPLAYFORM1 ) is the quantized clipped gradient.

The second momentv t is computed using the (squared) quantized clipped gradientǧ 2 t .

As shown in FIG0 (a) of BID29 , the distribution of gradients before quantization is close to the normal distribution.

Recall from Section 3.3 that the difference between E( g t 2 ) of the quantized gradientg t and the full-precision gradient ĝ t 2 is related to the dimension d. The following Proposition shows that E( ǧ t 2 )/E( ĝ t 2 ) becomes independent of d ifĝ t follows the normal distribution and clipping is used.

DISPLAYFORM2 However, the quantized clipped gradient may now be biased (i.e., E(ǧ t ) = Clip(ĝ t ) =ĝ t ).

The following Proposition shows that the bias is related to the clipping factor c. A larger c (i.e., less severe gradient clipping) leads to smaller bias.

DISPLAYFORM3 )), and erf(z) = DISPLAYFORM4 is the error function.

Theorem 3.

Assume thatĝ t follows N (0, σ 2 I).

For loss-aware weight quantization with quantized clipped gradients and η t = η/ √ t, DISPLAYFORM5 DISPLAYFORM6 Note that terms involvingg t in Theorem 2 are replaced byǧ t .

Moreover, the regret has an additional term D T t=1 E( Clip(ĝ t ) −ĝ t 2 ) over that in Theorem 2.

Comparing the average regrets in Theorems 1 and 3, gradient clipping before quantization slows convergence by a factor of (2/π) 1 2 c∆ g + 1, as compared to using full-precision gradients.

This is independent of d as the increase in E( ǧ t 2 ) is independent of d (Proposition 2).

Hence, a ∆ g larger than the one in Theorem 2 can be used, and this reduces the communication cost in distributed learning.

F (c) in (9) is thus smaller, but convergence is also slower.

Hence, there is a trade-off between the two.

Remark 1.

There are two scaling schemes in distributed training with data parallelism: strong scaling and weak scaling BID27 .

In this work, we consider weak scaling, which is more popular in deep network training.

In weak scaling, the same data set size is used for each worker.

The gradients are averaged over the N workers as DISPLAYFORM7 t .

If the gradients before averaging are independent random variables with zero mean, and g DISPLAYFORM8 the convergence speed with one worker is determined by DISPLAYFORM9 with N workers by ( DISPLAYFORM10 Thus, with N workers, the number of iterations for convergence is subsequently reduced by a factor of 1/N as compared to using a single worker.

In this section, we first study the effect of dimension d on the convergence speed and final error of a simple linear model with square loss as in .

Each entry of the model parameter is generated by uniform sampling from [−0.5, 0.5].

Samples x i 's are generated such that each entry of x i is drawn uniformly from [−0.5, 0.5], and the corresponding output y i from N (x i w * , (0.2) 2 ).

At the tth iteration, a mini-batch of B = 64 samples are drawn to form X t = [x 1 , . . .

, x B ] and y t = [y 1 , . . . , y B ] .

The corresponding loss is f t (w t ) = X t w t − y t 2 /2B.

The weights are quantized to 1 bit using LAB.

The gradients are either full-precision (denoted FP) or stochastically quantized to 2 bits (denoted SQ2).

The optimizer is RMSProp, and the learning rate is η t = η/ √ t, where η = 0.03.

Training is terminated when the average training loss does not decrease for 5000 iterations.

Figure 3(a) shows 2 convergence of the average training loss T t=1 f t (w t )/T , which differs from the average regret only by only a constant.

As can be seen, for both full-precision and quantized gradients, a larger d leads to a larger loss upon convergence.

Moreover, convergence is slower for larger d, particularly when the gradients are quantized.

These agree with the results in Theorems 1 and 2.

In this experiment, we follow BID29 and use the same train/test split, data preprocessing, augmentation and distributed Tensorflow setup.

We first study the effect of d on deep networks.

Experiments are performed on two neural network models.

The first one is a multi-layer perceptron with one layer of d hidden units BID24 .

The weights are quantized to 3 bits using LAQ3.

The gradients are either full-precision (denoted FP) or stochastically quantized to 3 bits (denoted SQ3).

The optimizer is RMSProp, and the learning rate is η t = η/ √ t, where η = 0.1.

The second network is the Cifarnet BID29 .

We set d to be the number of filters in each convolutional layer.

The gradients are either full-precision or stochastically quantized to 2 bits (denoted SQ2).

Adam is used as the optimizer.

The learning rate is decayed from 0.0002 by a factor of 0.1 every 200 epochs as in BID29 .Figures 3(b) and 3(c) show convergence of the average training loss for both networks.

As can be seen, similar to that in Section 4.1, a larger d leads to larger convergence degradation of the quantized gradients as compared to using full-precision gradients.

However, unlike the linear model, a larger d does not necessarily lead to a larger loss upon convergence.

We use the same Cifarnet model as in BID29 , with d = 64.

Weights are quantized to 1 bit (LAB), 2 bits (LAQ2), or m bits (LAQm).

The gradients are full-precision (FP) or stochastically quantized to m = {2, 3, 4} bits (SQm) without gradient clipping.

Adam is used as the optimizer.

The learning rate is decayed from 0.0002 by a factor of 0.1 every 200 epochs.

Two workers are used in this experiment.

FIG4 shows convergence of the average training loss with different numbers of bits for the quantized weight.

With full-precision or quantized gradients, weight-quantized networks have larger training losses than full-precision networks upon convergence.

The more bits are used, the smaller is the final loss.

This agrees with the results in Theorems 1 and 2.

TAB1 shows the test set accuracies.

Weight-quantized networks are less accurate than their full-precision counterparts, but the degradation is small when 3 or 4 bits are used.

We use the same Cifarnet model as in BID29 .

Adam is used as the optimizer.

The learning rate is decayed from 0.0002 by a factor of 0.1 every 200 epochs.

FIG6 shows convergence of the average training loss with different numbers of bits for the quantized gradients, again without gradient clipping.

Using fewer bits yields a larger final error, and using 2-or 3-bit gradients yields larger training loss and worse accuracy than full-precision gradients ( FIG6 and TAB1 ).

The fewer bits for the gradients, the larger the gap.

The degradation is negligible when 4 bits are used.

Indeed, 4-bit gradient sometimes has even better accuracy than full-precision gradient, as its inherent randomness encourages escape from poor sharp minima BID29 .

Moreover, using a larger m results in faster convergence, which agrees with Theorem 2.

In this section, we perform experiments on gradient clipping, with clipping factor c in {1, 2, 3}, using the Cifarnet BID29 .

LAQ2 is used for weight quantization and SQ2 for gradient quantization.

Adam is used as the optimizer.

The learning rate is decayed from 0.0002 by a factor of 0.1 every 200 epochs.

FIG8 (a) shows histograms of the full-precision gradients before clipping.

As can be seen, the gradients at each layer before clipping roughly follow the normal distribution, which verifies the assumption in Section 3.4.

FIG8 (b) shows the average g t 2 / ĝ t 2 (for nonclipped gradients) and ǧ t 2 / ĝ t 2 (for clipped gradients) over all iterations.

The dimensionalities (d) of the various Cifarnet layers are "conv1": 1600, "conv2": 1600, "fc3": 884736, "fc4": 73728, (for non-clipped gradients) and ǧ t 2 / ĝ t 2 (for clipped gradients); and (c) Training curves."softmax": 1920.

Layers with large d have large g t 2 / ĝ t 2 values, which agrees with Proposition 1.

With clipped gradients, ǧ t 2 / ĝ t 2 is much smaller and does not depend on d, agreeing with Proposition 3.

FIG8 shows convergence of the average training loss.

Using a smaller c (more aggressive clipping) leads to faster training (at the early stage of training) but larger final training loss, agreeing with Theorem 3.

FIG10 shows convergence of the average training loss with different numbers of bits for the quantized clipped gradient, with c = 3.

By comparing 3 with FIG6 , gradient clipping achieves faster convergence, especially when the number of gradient bits is small.

For example, 2-bit clipped gradient has comparable speed ( FIG10 ) and accuracy TAB1 as full-precision gradient.

In Remark 1, we showed that using multiple workers can reduce the number of training iterations required.

In this section, we vary the number of workers in a distributed learning setting with weak scaling, using the Cifarnet BID29 .

We fix the mini-batch size for each worker to 64, and set a smaller number of iterations when more workers are used.

We use 3-bit quantized weight (LAQ3), and gradients are full-precision or stochastically quantized to m = {2, 3, 4} bits (SQm).

TAB2 shows the testing accuracies with varying number of workers N .

Observations are similar to those in Section 4.2.4.

2-bit quantized clipped gradient has comparable performance as full-precision gradient, while the non-clipped counterpart requires 3 to 4 bits for comparable performance.

In this section, we train the AlexNet on ImageNet.

We follow BID29 and use the same data preprocessing, augmentation, learning rate, and mini-batch size.

Quantization is not performed in the first and last layers, as is common in the literature BID32 BID33 BID21 BID29 .

We use Adam as the optimizer.

We experiment with 4-bit loss-aware weight quantization (LAQ4), and the gradients are either full-precision or quantized to 3 bits (SQ3).

TAB3 shows the accuracies with different numbers of workers.

Weight-quantized networks have slightly worse accuracies than full-precision networks.

Quantized clipped gradient outperforms the non-clipped counterpart, and achieves comparable accuracy as full-precision gradient.

FIG12 shows the speedup in distributed training of a weight-quantized network with quantized/full-precision gradient compared to training with one worker using full-precision gradient.

We use the performance model in BID29 , which combines lightweight profiling on a single node with analytical communication modeling.

We use the AllReduce communication model BID22 , in which each GPU communicates with its neighbor until all gradients are accumulated to a single GPU.

We do not include the server's computation effort on weight quantization and the worker's effort on gradient clipping, which are negligible compared to the forward and backward propagations in the worker.

As can be seen from the Figure, even though the number of bits used for gradients increases by one at every aggregation step in the AllReduce model, the proposed method still significantly reduces network communication and speeds up training.

When the bandwidth is small FIG12 ), communication is the bottleneck, and using quantizing gradients is significantly faster than the use of full-precision gradients.

With a larger bandwidth FIG12 ), the difference in speedups is smaller.

Moreover, note that on the 1Gbps Ethernet with quantized gradients, its speedup is similar to those on the 10Gbps Ethernet with full-precision gradients.

In this paper, we studied loss-aware weight-quantized networks with quantized gradient for efficient communication in a distributed environment.

Convergence analysis is provided for weight-quantized models with full-precision, quantized and quantized clipped gradients.

Empirical experiments confirm the theoretical results, and demonstrate that quantized networks can speed up training and have comparable performance as full-precision networks.

We thank NVIDIA for the gift of GPU card.

A.1 PROOF OF THEOREM 1First, we introduce the following two lemmas.

DISPLAYFORM0 where β ∈ [0, 1).

Assume that g j ∞ < G ∞ for j ∈ {1, 2 . . . , t}. Then, DISPLAYFORM1 DISPLAYFORM2 and DISPLAYFORM3 Lemma 2. [Lemma 10.3 in BID13 ] Let g 1:T,i = [g 1,i , g 2,i , . . . , g T,i ] be the vector containing the ith element of the gradients for all iterations up to T , and g t be bounded as in Assumption A3.

Then, DISPLAYFORM4 Proof. (of Theorem 1) When only weights are quantized, the update for loss-aware weight quantization is DISPLAYFORM5 The update for the ith entry of w t is w t+1,i = w t,i − η tĝ DISPLAYFORM6 .

This implies DISPLAYFORM7 After rearranging, DISPLAYFORM8 Since f t is convex, we have DISPLAYFORM9 As f t is convex (Assumption A1) and twice differentiable (Assumption A2), ∇ 2 f t LI, where I is the identity matrix.

Combining with the assumption in Section 3.1 that w m − w n ≤ D, we have DISPLAYFORM10 Let x, y = x y be the dot product between two vectors x and y. Combining FORMULA1 and FORMULA1 , sum over all the dimensions i ∈ {1, 2 . . .

, d} and over all iterations t ∈ {1, 2, . . . , T }, we have DISPLAYFORM11 The first inequality comes from (10) in Lemma 1.

In the second inequality, the first term comes from DISPLAYFORM12 2 and the domain bound assumption in Section 3.1 (i.e., (w DISPLAYFORM13 t (w t −ŵ t )), and Lemma 2.

The third inequality comes from Cauchy's inequality.

The fourth inequality comes from (16).

The last inequality comes from (12) in Lemma 1.For m-bit (m > 1) loss-aware weight quantization in (13), aŝ DISPLAYFORM14 where r is the index that satisfies α t M r,i ≤ w t,i ≤ α t M r+1,i .

Since α = max{α 1 , . . .

, α T }, we have DISPLAYFORM15 Otherwise (i.e., w t,i is exterior of the representable range), the optimalŵ t,i is just the nearest representable value of w t,i .

Thus, DISPLAYFORM16 From FORMULA1 and FORMULA1 , and sum over all the dimensions, we have DISPLAYFORM17 From FORMULA1 and FORMULA48 , DISPLAYFORM18 From FORMULA1 and Assumption A3, we have from (17) DISPLAYFORM19 Thus, the average regret is DISPLAYFORM20 A.2 PROOF OF PROPOSITION 1Lemma 3.

For stochastic gradient quantization in (3), E(g t ) =ĝ t , and E( g t −ĝ t 2 ) ≤ ∆ g ĝ t ∞ ĝ t 1 .Proof.

Denote the ith element of the quantized gradientg t byg t,i .

For two adjacent quantized values B r,i , B r+1,i with B r,i ≤ |g t,i |/s t < B r+1,i , E(g t,i ) = E(s t · sign(ĝ t,i ) · q t,i ) = s t · sign(ĝ t,i ) · E(q t,i ) = s t · sign(ĝ t,i ) · (pB r+1,i + (1 − p)B r,i ) = s t · sign(ĝ t,i ) · (p(B r+1,i − B r,i ) + B r,i ) = s t · sign(ĝ t,i ) · |ĝ t,i | s t =ĝ t,i .Thus, E(g t ) =ĝ t , and the variance of the quantized gradients satisfy Proof.

From Lemma 3, E( g t 2 ) = E( g t −ĝ t 2 ) + ĝ t 2 ≤ ∆ g ĝ t ∞ ĝ t 1 + ĝ t 2 .

DISPLAYFORM21 Denote {x 1 , x 2 , . . . , x d } as the absolute values of the elements inĝ t sorted in ascending order.

From Cauchy's inequality, we have DISPLAYFORM22 The equality holds iff x 1 = x 2 = · · · = A.3 PROOF OF THEOREM 2Proof.

When both weights and gradients are quantized, the update is DISPLAYFORM23 Similar to the proof of Theorem 1, and using that E(g t ) =ĝ t (Lemma 3), we have DISPLAYFORM24 (1 − β)g 2 DISPLAYFORM25 As (21) in the proof of Theorem 1 still holds, using Proposition 1 and Assumption A3, we have DISPLAYFORM26

@highlight

In this paper, we studied efficient training of loss-aware weight-quantized  networks with  quantized gradient  in a distributed environment, both theoretically and empirically.

@highlight

This paper studies convergence properties of loss-aware weight quantization with different gradient precisions in the distributed environment, and provides convergence analysis for weight quantization with full-precision, quantized and quantized clipped gradients.

@highlight

The authors proposes an analysis of the effect of simultaneously quantizing the weights and gradients in training a parametrized model in a fully-synchronized distributed environment.