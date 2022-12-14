In distributed training, the communication cost due to the transmission of gradients or the parameters of the deep model is a major bottleneck in scaling up the number of processing nodes.

To address this issue, we propose dithered quantization for the transmission of the stochastic gradients and show that training with Dithered Quantized Stochastic Gradients (DQSG) is similar to the training with unquantized SGs perturbed by an independent bounded uniform noise, in contrast to the other quantization methods where the perturbation depends on the gradients and hence, complicating the convergence analysis.

We study the convergence of training algorithms using DQSG and the trade off between the number of quantization levels and the training time.

Next, we observe that there is a correlation among the SGs computed by workers that can be utilized to further reduce the communication overhead without any performance loss.

Hence, we develop a simple yet effective quantization scheme, nested dithered quantized SG (NDQSG), that can reduce the communication significantly without requiring the workers communicating extra information to each other.

We prove that although NDQSG requires significantly less bits, it can achieve the same quantization variance bound as DQSG.

Our simulation results confirm the effectiveness of training using DQSG and NDQSG in reducing the communication bits or the convergence time compared to the existing methods without sacrificing the accuracy of the trained model.

In recent years, the size of deep learning problems has increased significantly both in terms of the number of available training samples as well as the complexity of the model.

Hence, training deep models on a single processing node is unappealing or nearly impossible.

As such, large-scale distributed machine learning in which the training samples are distributed among different repository or processing units (referred to as workers) has started to be a viable approach for tackling the memory, storage and computational constraints.

The requirement to exchange the gradients or the parameters of the model incurs significant communication overhead which is a major bottleneck in distributed training algorithms.

In recent years, there has been a great amount of effort on reducing the communication overhead.

The majority of existing methods can be categorized into two groups: The first group mitigates the communication bottleneck by reducing the overall transmission rate via sparsification, quantization and/or compression of the gradients.

For example, BID15 reduces the communication overhead significantly by one-bit quantization of the stochastic gradients (SG).

However, the reduced accuracy of gradient may impair the convergence rate.

Using different quantization levels or adaptive quantizers, one can alleviate such issues by decreasing the error in the quantized gradients in the expense of increased communication bits BID4 .

Moreover, applying entropy coding algorithms such as Huffman coding on the quantized values can further reduce the communication bit-rate BID13 ; BID17 .

BID0 introduced QSGD which uses probabilistic (stochastic) quantization of SGs instead of ordinary fixed (deterministic) quantization methods.

They investigated its convergence guarantee and the trade-off between the quantization precision and variance of QSG.

Terngrad BID18 probabilistically quantizes the gradients into {???1, 0, +1} and it is shown that the convergence rate can be improved by layer-wise quantization and gradient clipping.

The second group of works attempts to attenuate the communication bottleneck by relaxing the synchronization between workers.

Each worker may continue its own computations while some others are still communicating and exchanging parameters.

Carefully scheduling and managing the asynchronous parameter exchange can lead to a better utilization of both the communication bandwidth and the computational power of the distributed system.

Examples of such approaches include DownpourSGD BID3 , Hogwild!

Niu et al. (2011 ), Hogwild++ Zhang et al. (2016 and Stale Synchronous Parallel model of computation BID7 .Our Contributions.

Our work in this paper falls within the first line of research, i.e. reducing the communication overhead by quantizing and compressing the gradients.

We first introduce using dithered quantization in the distributed computations of the stochastic gradient and show that stochastic quantizer of BID0 and ternarization of BID18 can be considered as special cases of our proposed method, although the reconstruction algorithms are slightly different.

The convergence of dithered quantized stochastic gradient descent algorithm is analyzed and its convergence speed w.r.t.

the number of workers and quantization precision is investigated.

Next, we observe that in a typical distributed system, the stochastic gradients computed by the workers are correlated.

However, the existing communication methods ignore that correlation.

We tap into the question of how that correlation can be exploited to further reduce the communication without sacrificing the precision or convergence of the learning algorithm.

We model the correlation between the stochastic gradients computed by each worker and propose a nested quantization scheme to reduce the communication bits without increasing the variance of the quantization error or reducing the convergence speed of the distributed training algorithm.

Throughout the paper, bold lowercase letters represent vectors and the i-th element of the vector x is denoted as x i .

Matrices are denoted by bold capital letters such as X, with the (i, j)-th element represented by X i,j or [X] i,j .

Given a real number x ??? R, x is the nearest integer to x. For a random variable u, u ??? U [a, b] if its probability distribution is uniform over interval [a, b] and u ??? N (??, ?? 2 ) if it follows a Gaussian distribution with mean ?? and variance ?? 2 .

It is well-known that the error in ordinary quantization especially when the number of quantization levels is low, depends on the input signal and is not necessarily uniformly distributed.

In Dithered Quantization, a (pseudo-)random signal called dither is added to the input signal prior to quantization.

Adding this controlled perturbation can cause the statistical behavior of the quantization error to be more desirable BID14 ; BID6 ; BID5 .Let Q(??) be an M-level uniform quantizer with quantization step size of ???, i.e., Q(v) = ??? v/??? where ?? is the nearest integer to ??.

The dithered quantizer is defined as follows;1 Definition (Dithered Quantization).

For an input signal x, let u be a dither signal, independent of x. The dithered quantization of x is defined asx = Q(x + u) ??? u. Remark 1.

To transmit the dithered quantization of x, it is sufficient to send the index of the quantization bin that x + u resides in, i.e., (x + u)/??? .

The receiver reproduces the (pseudo-)random sequence u using the same random number generator algorithm and seed number as the sender.

It is then subtracted from Q(x + u) to form the dithered quantized value,x.

Theorem 1 BID14 ).

If 1) the quantizer does not overload, i.e., |x + u| ??? M ??? 2 for all input signals x and dither u, and 2) The characteristic function of the dither signal, defined as M u (j??) = E u e j??u , satisfies M u (j 2??l ??? ) = 0 for all l = 0, then the quantization error e = x ???x is uniform over [??????/2, ???/2] and it is independent of the signal x.

It is common to consider U[???/2, ???/2] as the distribution of the random dither signal.

It can be easily verified that this choice of the dither signal satisfies the conditions of Thm.

1, and it does not increase the bound of the quantization error, i.e, |x ??? x| ??? ???/2 which is the same as the traditional uniform quantization with the same step size.

In some cases, the receiver may not be able to reproduce the dither signal to subtract from Q(x + u).

Hence, quantization is simply defined as asx h = Q(x + u).

We refer to this approach as the half-dithered quantization as the dither signal is applied only to the quantizer, not the reconstruction of x. In this case, the quantization error is not necessarily independent of the signal, however by an appropriate choice of the dither signal, the moments of the quantization error will be independent BID6 .

For example, if the dither signal u is the sum of k independent random variables, each having uniform distribution U[??????/2, ???/2], then the k-th moment of the quantization error, = x ???x h , would be independent of the signal, given by DISPLAYFORM0 2.1.1 RELATIONSHIP WITH TERNARY AND STOCHASTIC QUANTIZATIONS Here, we examine the relation between the dithered quantization, Ternary quantization of BID18 and the stochastic quantization in BID0 .

Without loss of generality, assume that the vector x is normalized such that |x i | ??? 1.

Although the reconstruction of quantized values in our method is different from those in TernGrad and QSGD, we show that these quantizers can be considered as a special case of the half-dithered quantizer.

M -level Stochastic Quantization in BID0 is defined as DISPLAYFORM1 where DISPLAYFORM2 The ternary quantizer of BID18 can be considered as a special case of stochastic quantizer with M = 1.

Lemma 2.

Stochastic quantization is the same as (2M + 1)-level half-dithered quantizer with DISPLAYFORM3 In other words, stochastic quantizer adds a uniformly distributed dither to the input signal before quantization, but at the receiver, it does not subtract the dither from the quantized value.

Therefore, the quantization error is not independent of the signal BID6 .

It can be easily verified that although the quantization is unbiased, E x ??? Q (s) (x) = 0, its variance depends on the value of the input signal: DISPLAYFORM4 It can be easily verified that the variance of the quantization error varies in the interval [0, 1 4M 2 ] depending on the value of x. If x is uniformly distributed over [???1, 1], the average quantization variance would be 1 6M 2 , twice the variance of the dithered quantization.

Here, we briefly overview the definition and some properties of the nested quantization.

Especially we focus on the one dimensional case as our algorithm is based on scalar quantization.

Definition (Nested Quantizers).

The pair (Q 1 , Q 2 ) of two quantizers are nested if and only if ???x, Q 1 (Q 2 (x)) = Q 2 (x), but the opposite does not necessarily hold.

Q 1 (??) and Q 2 (??) are called the fine and coarse quantizers, respectively.

Figure 1: Nested one-dimensional quantizers, fine quantizer (blue) with ??? 1 = 1/3 and coarse quantizer (green) with ??? 2 = 1.As a result, the centers of the quantization bins in the coarse quantizer is a subset of those of the fine quantizer.

In the one dimensional case, if Q 1 and Q 2 have quantization step sizes equal to ??? 1 and ??? 2 , respectively, it can be easily verified that they are nested if and only if there exists a constant integer k > 1 such that ??? 2 = k??? 1 .

For the definition and properties of higher dimensional nested quantization using lattices please refer to BID20 ; BID19 and references therein.

Let W ??? R n be a known set of possible parameters w and L : W ??? R be a differentiable objective function to be minimized.

A stochastic gradient g of L(w) is an unbiased random estimator of the gradient, i.e., g is a random function such that DISPLAYFORM0 where X is the set of training data samples and f (x; w) is a smooth differentiable parametric function, then given a mini-batch {x 1 , . . .

, x L } of training samples, the stochastic gradient of L(w) can be computed as g = 1 L l ??? w f (x l ; w).

We consider the distributed training scenario shown in Fig. 2.

There are P separate workers (processing nodes) which have their own copy of the model to be trained.

At each iteration of the training, each worker computes a stochastic gradient of the parameters g k , or the update in the parameters ??W k , based on its own available data.

It is then transmitted to a server (in the centralized training) or communicated with other workers (in the decentralized topology) to compute the average.

The average of all gradients or the updates (??? or?? W ) is then broadcasted back to all workers.

In the following, we focus on the distributed training using stochastic gradients with a centralized aggregation node.

First, we consider the use of dithered quantization in training and analyze the convergence of the learning algorithm in both single worker and distributed (multiple workers) training scenarios.

Next, we observe that the stochastic gradients computed at the workers are correlated.

We define a correlation model to capture the dependency between SGs of the workers and show that how nested dithered quantization can help further reducing the communication bits at each iteration of training without sacrificing the accuracy or the number of iterations to converge.

We consider the dithered quantization of SG (DQSG) as follows: Let Q(??) be a uniform quantizer with quantization step size ???, and u ??? U[??????/2, ???/2] be the random dither signal.

The dithered quantized SG is given byg DISPLAYFORM0 where the scale factor ?? = g ??? = max i |g i | maps the gradient into the range [???1, 1].

By Thm.

1, the scaled quantization noise e = (g ??? g)/?? will be independent from g and uniformly distributed over [??????/2, ???/2].

Note that by setting ??? = 1/M , we will have a 2M + 1 level quantizer with quantization bins' indexes in {???M, . . .

, ???1, 0, 1, . . .

, M }.

Lemma 3.

Let g be a stochastic gradient of L(w).

Then, the DQSG,g, has the following properties: DISPLAYFORM1 Especially, if we assume that the difference between the stochastic gradients and the true ones behaves like a Gaussian noise, i.e DISPLAYFORM2 As a result of Lemma 3, we observe that the excess variance caused by quantization is proportional to ??? 2 .

Hence by adding 1 bit, i.e., doubling the number of quantization levels, it is reduced by a factor of 4.

Further, we notice that how partitioning the stochastic gradient into K sub-vectors can reduce the variance of DQSG at the expense of extra communication bits.

Letg K be the DQSG resulted from partitioning g into K sub-vectors and quantizing them separately.

For the simplicity of analysis assume that the partitions are of equal length, n/K. Simple calculations reveal that DISPLAYFORM3 2 Usually, the SG is computed as g = 1 L l ???wf (x l ; w) and for large enough L, due to the central limit DISPLAYFORM4 for an appropriate fixed covariance matrix ??.The first term decreases logarithmically w.r.t.

the number of partitions.

On the other hand, each partition requires transmitting an additional scale factor (?? in (2), see Alg.

1), incurring extra Kb bits in total, where b is the number of bits for each scale factor.

Hence, the excess communication bits due to partitioning increases linearly, while the first term in the excess variance decreases logarithmically.

Convergence Analysis.

We now analyze the convergence of the gradient descent algorithm with the dithered quantized stochastic gradients.

At the t-th iteration, the parameters are updated as DISPLAYFORM5 where ?? t is the learning rate and g t is the DQSG.Recall that g = g + g ??? , where ??? U[??????/2, ???/2] is the quantization noise, independent of g. Hence, training with dithered quantized SG is the same as training with non-quantized SG corrupted by an independent bounded uniform noise.

If the quantization step size and hence the noise is controlled appropriately, the quantization noise can improve the training of very deep models BID10 ; BID12 Moreover, analyzing the convergence of (DQSGD) is almost the same as the ordinary SGD.

For example, since E g DISPLAYFORM6 , under the same assumptions as of BID1 , the convergence of DQSGD can be proven, which is replicated here for the sake of completeness.

Theorem 4.

Assume that i) L(w) has a single minimum, w DISPLAYFORM7 t < +???, and iv) for some constants A and B, stochastic gradients satisfy E g(w) 2 2 ??? A + B w ??? w * 2 2 .

Then for any quantization step size ??? ??? 1, training with DQSGD converges to the solution almost surely.

Next, we investigate how the number of workers and quantization step size affects the training time in the proposed distributed training scheme.

Distributed Training with DQSGD.

Algorithm 1 summarizes the proposed distributed training with P workers using dithered quantization of SG (DQSG).

The p-th worker, first computes the stochastic gradient g p and then using the scale parameter ?? p = g p ??? , computes the quantization index q p (see Remark 1).

Hence, the DQSG is given by g p = ?? p (???.q p ??? u p ).

To be able to reproduce the (pseudo-)random sequences at the server, the same random number generator algorithm and seed number, s p , is used at both the worker and the server.

At each iteration of training, the seed numbers are updated according to a predetermined algorithm at all workers and the server, to prevent generating the same random sequences repeatedly.

Using the above distributed training algorithm, the following result on the convergence time of distributed (DQSGD) algorithm can be proved.

Theorem 5.

Let W ??? R n be a convex set and L(w) be a convex, Lipschitz-smooth function with constant 3 .

Further, assume that L achieves its minimum at w * and has bounded gradients almost everywhere, i.e., for a constant B > 0, ???L 2 ??? B.Let the initial point for the learning algorithm be w 0 and R = sup w???W w ??? w 0 2 .

Consider distributed training algorithm (Alg.

1) on P workers using (DQSGD) with quantization step size ???. Suppose that the workers can compute stochastic gradients with variance bound V , i.e,.

DISPLAYFORM8 /12.

Then for sufficiently small > 0, after T steps of training with constant step size ?? t , where DISPLAYFORM9 , and ?? t = /( + 1.1?? 2 /P ),we have DISPLAYFORM10 Let T c be the training time without any quantization in the above setup.

Then, it can be easily verified that the training time of the dithered quantization is increased by DISPLAYFORM11 Algorithm 1: Distributed Training Using Dithered Quantization of SG Initialization -Assign a random seed sp to the p-th worker and initialize the parameters with w0, p = 1, 2 . . .

, P .-Keep a copy of sp's at the server.-Set ???, the quantization step-size, and the associated uniform quantizer, Q(??).for each iteration of training do Workers p = 1, 2, . . .

, P : -Get a batch of training data and compute the stochastic gradients gp.-Generate a pseudo-random sequence up, uniformly distributed over [??????/2, ???/2] using seed sp.-Compute the quantization index: qp = t/??? where t = gp/??p + up and ??p = g ???.-Update the seed number sp.-Send ??p and qp (or the corresponding quantization bin).

Server :-Reproduce the pseudo-random sequence up using the seed number sp.-Reconstruct the gradient of the p-th worker asgp = ??p (???.qp ??? up).-Update the seed number sp.-Compute the average SG,?? g = 1 P pg p, and broadcast it to the workers.

Workers p = 1, 2, . . .

, P :-Receive average SG,?? g.-Update parameters according to the the preset training algorithm (SGD, ADAM, ...).

It is well-known that correlated signals can be communicated more efficiently via distributed compression than the traditional entropy based coding BID16 .

Nested Quantization has been proven to be a viable tool in communicating correlated data BID20 .

Here, we propose to use nested quantization in distributed learning.

Let (Q 1 , Q 2 ) be a pair of nested quantizers with quantization step sizes ??? 1 and ??? 2 , respectively and 0 < ?? ??? 1 be a shrinkage factor whose value to be determined later.

To quantize and transmit x, the worker first generates a random dither u ??? U[?????? 1 /2, ??? 1 /2] and computes t = ??x + u. Then it quantizes and encodes it as DISPLAYFORM0 i.e., it transmits the position of the fine quantization bin relative to the coarse one (shown by indexes ???1, 0, 1 in FIG2 .

At the receiver, by knowing s alone, x cannot be estimated reliably as multiple values can produce the same s. To resolve that ambiguity, it is required to know which coarse quantization bin x belongs to.

This is achieved by the help of the information provided by y, available at the receiver.

x is reconstructed from the received s and using y as follows: DISPLAYFORM1 Note that quantizing x does not require y, however estimating x at the server depends on the information provided by y. FIG2 shows an example of using nested quantization, where ??? 1 = 1 and ??? 2 = 3.

Let x = ???4.2 and u = 0.3 be the generated dither.

Assume ?? = 1, hence s = Q 1 (???3.9) ??? Q 2 (???3.9) = ???4 ??? (???3) = ???1 is the signal to be transmitted.

Note that multiple points can produce the same s with that dither signal, some are shown by in the figure, e.g., ???4.3., ???1.3, 2.7, . . .

all leads to the same s. However, having access to y = ???3.4 at the receiver can resolve the ambiguity.

The value which resides in the same coarse quantization bin as y is chosen, resulting inx = ???4.3.

Note that in this nested quantization scheme, the output of quantizer is in {???1, 0, +1}. If we wanted to achieve the same accuracy with a single quantizer, we had to transmit s = ???4 instead of s = ???1, increasing the number of bits depending on the range of x. For example, in FIG2 , nested quantization reduces the range of quantization indexes from {???4, ???3, . . .

, 4} to {???1, 0, 1}, reduction by a factor of 3.Algorithm 2: Distributed Training Using Nested Dithered Quantization of SG DISPLAYFORM2 Our proposed distributed training using nested dithered quantization is summarized in Alg.

2 for one iteration of training 4 .

The stochastic gradient, computed by the p-th worker in a distributed training system, can be considered as a noisy estimate of the true gradient, i.e., g p = ??? w L + ?? p where ?? p is a zero-mean noise.

However, as opposed to BID20 and other similar works, the exact gradient ??? w L is not available in distributed training.

To overcome this issue, we propose to divide the workers into two groups.

Set P 1 of workers use DQSG with quantization step size ??? 1 , to provide an initial estimate for the true gradient.

The parameter of the quantization and the number of workers in P 1 are chosen such that the variance of averaged DQSG (see Lemma 3) becomes in an acceptable range, determined by Thm.

6.

The workers in P 2 use nested quantizer with step-sizes (??? DISPLAYFORM3 2 ) and scale ?? p .

To decode the received Nested Dithered Quantized SG (NDQSG), the receiver uses the average of all SGs already received and decoded from other workers, denoted by g. We assume that the SG of the p-th worker can be modeled as g p = g + z p , where z p is an independent random noise.

Hence, the nested quantization uses g at the receiver as the side information to compute g p .

To find the quantization parameters, we can use the following result; Theorem 6.

If the SG at a worker is modeled by g =?? g + z, E z 2 i = ?? 2 z , and the worker uses nested quantizer with parameters ??? 1 , ??? 2 and ??, then with probability at least 1 ??? p, g i will be estimated correctly (i.e., g i and g i are in the same coarse quantization bin), where DISPLAYFORM4 DISPLAYFORM5 2?? , then p = 0.

In this case, DISPLAYFORM6 Note that setting ?? = 1 or ?? = 1 ??? ??? 2 1 /12?? 2 z results in the same quantization variance as dithered quantization with step-size ??? 1 .

However, nested quantization requires log 2 (??? p2 /??? p1 ) bits to transmit each value, i.e., less than the ordinary quantization methods which requires almost log 2 (2/??? p1 ) bits.

We examine the convergence and and the number of communication bits used by different learning algorithms based on DQSG and nested dithered quantized SG (NDQSG) for various number of workers, and compare them against the baseline (no quantization of gradients), one-bit quantization BID15 , TernGrad Wen et al. (2017) , and QSGD BID0 .

Although it is possible to evaluate the performance of the quantization and compression schemes in both synchronous and asynchronous settings, here we assume that the workers and server are synchronous.

The main reason for such a setting is to cancel-out the performance degradation (in terms of training accuracy or speed) that may be caused by the stale gradients in asynchronous updates, and to solely investigate the effect of the quantization/compression algorithms.

We have considered three different models, a fully connected neural network with two hidden layers of sizes 300 and 100 over MNIST dataset (herein, referred to as FC-300-100), a Lenet-5 like convolutional network BID9 First, we observe that using entropy coding algorithms such as Adaptive Arithmetic Coding (ACC) can further reduce the communication bits for all schemes close to the entropy limit (within 5% range).

Therefore, it suffices to report both the number of raw communication bits from quantization as well as the resulting entropy of the bit-stream for comparison.

TAB0 show the raw (uncompressed) communication bits and the entropy per worker at each iteration of training, respectively.

The communication bits of DQSGD and QSGD are close to each other.

Although One-bit quantization requires less raw bits to transmit, it is less compressible, e.g., using entropy coding for Lenet, DQSGD would use 6 times less number of bits per iteration compared to one-bit quantization.

Figure 4 shows the accuracy of the final trained model vs different number of workers for FC-300-100 and Lenet models.

TAB2 shows the results for CifarNet model after 50 epochs ot training.

From the simulations, it is seen that our proposed algorithm performs much better than the one-bit quantization method and is close to the baseline performance (non-quantized communication).Moreover, in Fig. 5 , we have compared the convergence rate of our dithered quantization scheme w.r.t.

baseline (no quantization), one-bit quantization BID15 and QSGD Alistarh et al. (2017) for 4 and 8 workers.

It is interesting to note that the dithered quantization improves the convergence of the training algorithm even when compared to the baseline (no quantization) in terms of number of training iterations.

Although we do not have any analytic proof that using dithered quantization would always improve the convergence speed w.r.t.

no quantization, because of the independency of the noise from the SGs in our proposed method, our method is likely to result in a better convergence property than the aforementioned techniques for complex training data BID10 ; BID12 .

On the other hand, as the number of workers increases, due to the averaging performed on the received quantized SGs, the noise would decrease proportionately and we expect the performance gap between different quantization methods eventually vanishes.

Next, we compare our nested dithered quantizer with the dithered quantization scheme.

To have fair comparison, we chose the same expected accuracy for both quantization schemes.

For DQSG, we chose M = 2, hence ??? = 0.5 and the output of quantizer would be in {???2, . . .

, ???2}. In NDQSG, for half of the workers, we divided the workers to two groups, half of the workers use DQSG with the same ??? and the other half, uses NDQSG with ??? 1 = 1/3 and ??? 2 = 1.

Hence, the output of NDQSG quantizer is in {???1, 0, 1}. In FIG5 we compared the accuracy of NDQSG with DQSG and baseline training during training.

As seen, the learning curve of NDQSG is almost the same as DQSG and the baseline.

However, the communication bits are much less.

For example, in training FC-300-100, with 2 level quantizers, QSG and DQSG requires 619.2 Kbits per worker to communicate, while NDQSG reduces that to 422.8 Kbits, more than 30% reduction in number of bits to communicate.

The Same is true for the other considered neural networks.

In this paper, first, we introduced DQSG, dithered quantized stochastic gradient, and showed that how it can reduce communication bits per training iteration both theoretically and via simulations, without affecting the accuracy of the trained model.

Next, we explored the correlation that exists among the SGs computed by workers in a distributed system and proposed NDQSG, a nested quantization method for the SGs.

Using theoretical analysis as well as simulations, we showed that NDSQG performs almost the same as DQSG in terms of accuracy and training speed, but with much fewer number of communication bits.

Finally, we would like to mention that although the simulations and analysis of the proposed distributed training method is done in synchronous training setup, it is applicable to the asynchronous training as well.

Further, our nested quantization scheme can be easily extended to hierarchical distributed structures.

A PROOF OF LEMMA 2Let Q(??) be a 2M + 1-level quantizer with step size ??? = 1/M .

Let u ??? U[??????/2, ???/2] be the dither signal.

Let 0 ??? x ??? 1 be an arbitrary number.

Assume that l/M ??? x < (l + 1)/M and define d = x ??? l/M .

Note that 0 ??? d < ??? and DISPLAYFORM0 Similarly, P (Q(x + u) = (l + 1)/M ) = M d. Comparing with stochastic quantizer, we see that they both assign the quantization points with the same probability.

The case x < 0 can be verified similarly.

B PROOF OF LEMMA 3To prove the unbiasedness, note that by Thm.

1, e = Q(g/?? + u) ??? (g/?? + u) is independent from g/?? and uniformly distributed over [??????/2, ???/2].

On the other hand,g = g + ??e.

Hence, DISPLAYFORM1 (a) DISPLAYFORM2 = ???L, where (a) is due to the fact that ?? = g ??? is independent of e and (b) because of unbiasedness of stochastic gradient and e having mean zero.

For the variance, DISPLAYFORM3 This is a direct result of (Bubeck, 2015, ??6) .

Note that E g ??? ??? w L 2 2 ??? V + n??? 2 12 E g 2 2 ??? V (1 + n ??? 2 /12) + nB ??? 2 /12 = ?? 2 and since there are P workers, the variance bound on g would be ?? 2 /P .

Then after T iterations of (DQSGD) with step size ?? t = 1/( + 1/??) for ?? = DISPLAYFORM4 For < 0.2?? 2 /P L, set T = 2.5 R 2 ?? 2 P 2 .

Then, it can be easily verified that for the given step-size, the results hold.

E PROOF OF THM.

6Let e = ??g + u ??? Q 1 (??g + u) and r = s ??? u ??? ???? g. Then, DISPLAYFORM5 Since?? g i = g i + z i , it can be shown that r i ??? Q 2 (r i ) = ??z i ??? e i ??? Q 2 (??z i ??? e i ).

i =?? g i + ??(??z i ??? e i ) ??? ??Q 2 (??z i ??? e i ).

The correct decoding occurs when Q 2 (??z i ??? e i ) = 0.

Hence, the probability of correct recovery would be 1 ??? p where DISPLAYFORM0 In that case,?? i = g i ??? (??e i + (1 ??? ?? 2 )z i ).Since e i ??? U[?????? 1 /2, ??? 1 /2] and z i are independent from each other and from g i , simple calculations show that DISPLAYFORM1

@highlight

The paper proposes and analyzes two quantization schemes for communicating Stochastic Gradients in distributed learning which would reduce communication costs compare to the state of the art while maintaining the same accuracy.  

@highlight

The authors propose applying dithered quantization to the stochastic gradients computed through the training process, which improves quantization error and achieves superior results compared to baselines, and propose a nested scheme to reduce communication cost.

@highlight

Authors establish a connection between communication reduction in distributed optimization and dithered quantization and develops two new distributed training algorithms where communication overhead is significantly reduced.