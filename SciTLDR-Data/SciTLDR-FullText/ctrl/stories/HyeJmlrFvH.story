As the size and complexity of models and datasets grow, so does the need for communication-efficient variants of stochastic gradient descent that can be deployed on clusters to perform model fitting in parallel.

Alistarh et al. (2017) describe two variants of data-parallel SGD that quantize and encode gradients to lessen communication costs.

For the first variant, QSGD, they provide strong theoretical guarantees.

For the second variant, which we call QSGDinf, they demonstrate impressive empirical gains for distributed training of large neural networks.

Building on their work, we propose an alternative scheme for quantizing gradients and show that it yields stronger theoretical guarantees than exist for QSGD while matching the empirical performance of QSGDinf.

Deep learning is booming thanks to enormous datasets and very large models, leading to the fact that the largest datasets and models can no longer be trained on a single machine.

One common solution to this problem is to use distributed systems for training.

The most common algorithms underlying deep learning are stochastic gradient descent (SGD) and its variants, which led to a significant amount of research on building and understanding distributed versions of SGD.

Implementations of SGD on distributed systems and data-parallel versions of SGD are scalable and take advantage of multi-GPU systems.

Data-parallel SGD, in particular, has received significant attention due to its excellent scalability properties (Zinkevich et al., 2010; Bekkerman et al., 2011; Recht et al., 2011; Dean et al., 2012; Coates et al., 2013; Chilimbi et al., 2014; Duchi et al., 2015; Xing et al., 2015; .

In data-parallel SGD, a large dataset is partitioned among K processors.

These processors work together to minimize an objective function.

Each processor has access to the current parameter vector of the model.

At each SGD iteration, each processor computes an updated stochastic gradient using its own local data.

It then shares the gradient update with its peers.

The processors collect and aggregate stochastic gradients to compute the updated parameter vector.

Increasing the number of processing machines reduces the computational costs significantly.

However, the communication costs to share and synchronize huge gradient vectors and parameters increases dramatically as the size of the distributed systems grows.

Communication costs may thwart the anticipated benefits of reducing computational costs.

Indeed, in practical scenarios, the communication time required to share stochastic gradients and parameters is the main performance bottleneck (Recht et al., 2011; Seide et al., 2014; Strom, 2015; .

Reducing communication costs in data-parallel SGD is an important problem.

One promising solution to the problem of reducing communication costs of data-parallel SGD is gradient compression, e.g., through gradient quantization (Dean et al., 2012; Seide et al., 2014; Sa et al., 2015; Gupta et al., 2015; Abadi et al., 2016; Zhou et al., 2016; Bernstein et al., 2018) .

(This should not be confused with weight quantization/sparsification, as studied by ; Hubara et al. (2016) ; Park et al. (2017) ; , which we do not discuss here.)

Unlike full-precision data-parallel SGD, where each processor is required to broadcast its local gradient in full-precision, i.e., transmit and receive huge full-precision vectors at each iteration, quantization requires each processor to transmit only a few communication bits per iteration for each component of the stochastic gradient.

One popular such proposal for communication-compression is quantized SGD (QSGD), due to .

In QSGD, stochastic gradient vectors are normalized to have unit L 2 norm, and then compressed by quantizing each element to a uniform grid of quantization levels using a randomized method.

While most lossy compression schemes do not provide convergence guarantees, QSGD's quantization scheme, is designed to be unbiased, which implies that the quantized stochastic gradient is itself a stochastic gradient, only with higher variance determined by the dimension and number of quantization levels.

As a result, are able to establish a number of theoretical guarantees for QSGD, including that it converges under standard assumptions.

By changing the number of quantization levels, QSGD allows the user to trade-off communication bandwidth and convergence time.

Despite their theoretical guarantees based on quantizing after L 2 normalization, Alistarh et al. opt to present empirical results using L ??? normalization.

We call this variation QSGDinf.

While the empirical performance of QSGDinf is strong, their theoretical guarantees on the number of bits transmitted no longer apply.

Indeed, in our own empirical evaluation of QSGD, we find the variance induced by quantization is substantial, and the performance is far from that of SGD and QSGDinf.

Given the popularity of this scheme, it is natural to ask one can obtain guarantees as strong as those of QSGD while matching the practical performance of the QSGDinf heuristic.

In this work, we answer this question in the affirmative by providing a new quantization scheme which fits into QSGD in a way that allows us to establish stronger theoretical guarantees on the variance, bandwidth, and cost to achieve a prescribed gap.

Instead of QSGD's uniform quantization scheme, we use an unbiased nonuniform logarithmic scheme, similar to those introduced in telephony systems for audio compression (Cattermole, 1969) .

We call the resulting algorithm nonuniformly quantized stochastic gradient descent (NUQSGD).

Like QSGD, NUQSGD is a quantized data-parallel SGD algorithm with strong theoretical guarantees that allows the user to trade off communication costs with convergence speed.

Unlike QSGD, NUQSGD has strong empirical performance on deep models and large datasets, matching that of QSGDinf.

In particular, we provide a new efficient implementation for these schemes using a modern computational framework (Pytorch), and benchmark it on classic large-scale image classification tasks.

The intuition behind the nonuniform quantization scheme underlying NUQSGD is that, after L 2 normalization, many elements of the normalized stochastic gradient will be near-zero.

By concentrating quantization levels near zero, we are able to establish stronger bounds on the excess variance.

In the overparametrized regime of interest, these bounds decrease rapidly as the number of quantization levels increases.

Combined with a bound on the expected code-length, we obtain a bound on the total communication costs of achieving an expected suboptimality gap.

The resulting bound is slightly stronger than the one provided by QSGD.

To study how quantization affects convergence on state-of-the-art deep models, we compare NUQSGD, QSGD, and QSGDinf, focusing on training loss, variance, and test accuracy on standard deep models and large datasets.

Using the same number of bits per iteration, experimental results show that NUQSGD has smaller variance than QSGD, as expected by our theoretical results.

This smaller variance also translates to improved optimization performance, in terms of both training loss and test accuracy.

We also observe that NUQSGD matches the performance of QSGDinf in terms of variance and loss/accuracy.

Further, our distributed implementation shows that the resulting algorithm considerably reduces communication cost of distributed training, without adversely impacting accuracy.

Our empirical results show that NUQSGD can provide faster end-to-end parallel training relative to data-parallel SGD, QSGD, and Error-Feedback SignSGD (Karimireddy et al., 2019) on the ImageNet dataset.

???

We establish stronger theoretical guarantees for the excess variance and communication costs of our gradient quantization method than those available for QSGD's uniform quantization method.

??? We then establish stronger convergence guarantees for the resulting algorithm, NUQSGD, under standard assumptions.

??? We demonstrate that NUQSGD has strong empirical performance on deep models and large datasets, both in terms of accuracy and scalability.

Thus, NUQSGD closes the gap between the theoretical guarantees of QSGD and the empirical performance of QSGDinf.

Seide et al. (2014) proposed signSGD, an efficient heuristic scheme to reduce communication costs drastically by quantizing each gradient component to two values.

Bernstein et al. (2018) later provided convergence guarantees for signSGD.

Note that the quantization employed by signSGD is not unbiased, and so a new analysis was required.

As the number of levels is fixed, SignSGD does not provide any trade-off between communication costs and convergence speed.

Sa et al. (2015) introduced Buckwild!, a lossy compressed SGD with convergence guarantees.

The authors provided bounds on the error probability of SGD, assuming convexity and gradient sparsity.

proposed TernGrad, a stochastic quantization scheme with three levels.

TernGrad also significantly reduces communication costs and obtains reasonable accuracy with a small degradation to performance compared to full-precision SGD.

Convergence guarantees for TernGrad rely on a nonstandard gradient norm assumption.

As discussed, proposed QSGD, a more general stochastic quantization scheme, for which they provide both theoretical guarantees and experimental validation (although for different variants of the same algorithm).

We note that their implementation was only provided in Microsoft CNTK; by contrast, here we provide a more generic implementation in Horovod (Sergeev and Del Balso, 2018) , a communication back-end which can support a range of modern frameworks such as Tensorflow, Keras, Pytorch, and MXNet.

NUQSGD uses a logarithmic quantization scheme.

Such schemes have long been used in telephony systems for audio compression (Cattermole, 1969) .

Logarithmic quantization schemes have appeared in other contexts recently: Hou and Kwok (2018) studied weight distributions of long short-term memory networks and proposed to use logarithm quantization for network compression.

Zhang et al. (2017) proposed a gradient compression scheme and introduced an optimal quantization scheme, but for the setting where the points to be quantized are known in advance.

As a result, their scheme is not applicable to the communication setting of quantized data-parallel SGD.

We consider a high-dimensional machine learning model, parametrized by a vector w ??? R d .

Let ??? ??? R d denote a closed and convex set.

Our objective is to minimize f : ??? ??? R, which is an unknown, differentiable, convex, and ?? -smooth function.

The following summary is based on .

where ?? denotes the Euclidean norm.

Let (S , ??, ??) be a probability space (and let E denote expectation).

Assume we have access to stochastic gradients of f , i.e., we have access to a function

In the rest of the paper, we let g(w) denote the stochastic gradient for notational simplicity.

The update rule for conventional full-precision projected SGD is w t+1 = P ??? (w t ??? ??g(w t )), where w t is the current parameter input, ?? is the learning rate, and P ??? is the Euclidean projection onto ???.

We say the stochastic gradient has a second-moment upper bound B when E[ g(w) 2 ] ??? B for all w ??? ???. Similarly, the stochastic gradient has a variance upper bound

Note that a second-moment upper bound implies a variance upper bound, because the stochastic gradient is unbiased.

We have classical convergence guarantees for conventional full-precision SGD given access to stochastic gradients at each iteration: Theorem 1 (Bubeck 2015, Theorem 6.3).

Let f : ??? ??? R denote a convex and ?? -smooth function and let R 2 sup w?????? w ??? w 0 2 .

Suppose that the projected SGD update is executed for T iterations with ?? = 1/(?? + 1/??) where ?? = r 2/T /?? .

Given repeated and independent access to stochastic gradients with a variance upper bound ?? 2 , projected SGD satisfies

Minibatched (with larger batch sizes) and data-parallel SGD are two common SGD variants used in practice to reduce variance and improve computational efficiency of conventional SGD.

At each iteration, each processor computes its own stochastic gradient based on its local data and then broadcasts it to all peers.

Each processor receives and aggregates the stochastic gradients from all peers to obtain the updated parameter vector.

In detail, the update rule for full-precision dataparallel SGD is

where g l (w t ) is the stochastic gradient computed and broadcasted by processor l. Provided that g l (w t ) is a stochastic gradient with a variance upper bound ?? 2 for all l, then Data-parallel SGD is described in Algorithm 1.

Full-precision data-parallel SGD is a special case of Algorithm 1 with identity encoding and decoding mappings.

Otherwise, the decoded stochastic gradient?? i (w t ) is likely to be different from the original local stochastic gradient g i (w t ).

By Theorem 1, we have the following convergence guarantees for full-precision data-parallel SGD: Corollary 1 (Alistarh et al. 2017, Corollary 2.2) .

Let f , R, and ?? be as defined in Theorem 1 and let ?? > 0.

Suppose that the projected SGD update is executed for T iterations with ?? = 1/(?? + ??? K/??) on K processors, each with access to independent stochastic gradients of f with a second-moment bound B. The smallest T for the full-precision data-parallel SGD that guarantees

Data-parallel SGD reduces computational costs significantly.

However, the communication costs of broadcasting stochastic gradients is the main performance bottleneck in large-scale distributed systems.

In order to reduce communication costs and accelerate training, introduced a compression scheme that produces a compressed and unbiased stochastic gradient, suitable for use in SGD.

At each iteration of QSGD, each processor broadcasts an encoding of its own compressed stochastic gradient, decodes the stochastic gradients received from other processors, and sums all the quantized vectors to produce a stochastic gradient.

In order to compress the gradients, every coordinate (with respect to the standard basis) of the stochastic gradient is normalized by the Euclidean norm of the gradient and then stochastically quantized to one of a small number quantization levels distributed uniformly in the unit interval.

The stochasticity of the quantization is necessary to not introduce bias.

give a simple argument that provides a lower bound on the number of coordinates that are quantized to zero in expectation.

Encoding these zeros efficiently provides communication savings at each iteration.

However, the cost of their scheme is greatly increased variance in the gradient, and thus slower overall convergence.

In order to optimize overall performance, we must balance communication savings with variance.

By simple counting arguments, the distribution of the (normalized) coordinates cannot be uniform.

Indeed, this is the basis of the lower bound on the number of zeros.

These arguments make no assumptions on the data distribution, and rely entirely on the fact that the quantities being quantized are the coordinates of a unit-norm vector.

Uniform quantization does not capture the properties of such vectors, leading to substantial gradient variance.

In this paper, we propose and study a new scheme to quantize normalized gradient vectors.

Instead of uniformly distributed quantization levels, as proposed by , we consider quantization levels that are nonuniformly distributed in the unit interval, as depicted in Figure 1 .

In order to obtain a quantized gradient that is suitable for SGD, we need the quantized gradient to remain unbiased. achieve this via a randomized quantization scheme, which can be easily generalized to the case of nonuniform quantization levels.

Using a carefully parametrized generalization of the unbiased quantization scheme introduced by Alistarh et al., we can control both the cost of communication and the variance of the gradient.

Compared to a uniform quantization scheme, our scheme reduces quantization error and variance by better matching the properties of normalized vectors.

In particular, by increasing the number of quantization levels near zero, we obtain a stronger variance bound.

Empirically, our scheme also better matches the distribution of normalized coordinates observed on real datasets and networks.

We now describe the nonuniform quantization scheme: Let s ??? {1, 2, ?? ?? ?? } be the number of internal quantization levels, and let L = (l 0 , l 1 , ?? ?? ?? , l s+1 ) denote the sequence of quantization levels, where

, lets(r) and p(r) satisfy ls (r) ??? r ??? ls (r)+1 and r = 1 ??? p(r) ls (r) + p(r)ls (r)+1 , respectively.

Define ??(r) = ls (r)+1 ??? ls (r) .

Note thats(r) ??? {0, 1, ?? ?? ?? , s}.

where, letting r i = |v i |/ v , the h i (v, s)'s are independent random variables such that h i (v, s) = ls (r i ) with probability 1 ??? p(r i ) and h i (v, s) = ls (r i )+1 otherwise.

We note that the distribution of h i (v, s) satisfies E[h i (v, s)] = r i and achieves the minimum variance over all distributions that satisfy E[h i (v, s)] = r i with support L .

In the following, we focus on a special case of nonuniform quantization withL = (0, 1/2 s , ?? ?? ?? , 2 s???1 /2 s , 1) as the quantization levels.

The intuition behind this quantization scheme is that it is very unlikely to observe large values of r i in the stochastic gradient vectors of machine learning models.

Stochastic gradients are observed to be dense vectors (Bernstein et al., 2018) .

Hence, it is natural to use fine intervals for small r i values to reduce quantization error and control the variance.

After quantizing the stochastic gradient with a small number of discrete levels, each processor must encode its local gradient into a binary string for broadcasting.

We describe this encoding in Appendix A.

In this section, we provide theoretical guarantees for NUQSGD, giving variance and code-length bounds, and using these in turn to compare NUQSGD and QSGD.

Please note that the proofs of Theorems 2, 3, 4, and 5 are provided in Appendices B, C, D, and E respectively.

where

The result in Theorem 2 implies that if g(w) is a stochastic gradient with a second-moment bound ??, then Q s (g(w)) is a stochastic gradient with a variance upper bound ?? Q ??.

In the range of interest where d is sufficiently large, i.e., s = o(log(d)), the variance upper bound decreases with the number of quantization levels.

To obtain this data-independent bound, we establish upper bounds on the number of coordinates of v falling into intervals defined byL .

where

Theorem 3 provides a bound on the expected number of communication bits to encode the quantized stochastic gradient.

Note that 2 2s + ??? d2 s ??? d/e is a mild assumption in practice.

As one would expect, the bound, (4), increases monotonically in d and s. In the sparse case, if we choose s = o(log d) levels, then the upper bound on the expected code-length is

Combining the upper bounds above on the variance and code-length, Corollary 1 implies the following guarantees for NUQSGD: Theorem 4 (NUQSGD for smooth convex optimization).

Let f and R be defined as in Theorem 1, let ?? Q be defined as in Theorem 2, let ?? > 0,B = (1 + ?? Q )B, and let ?? > 0 be given by ?? 2 = 2R 2 /(BT ).

With ENCODE and DECODE defined as in Appendix A, suppose that Algorithm 1 is executed for T iterations with a learning rate ?? = 1/(?? + ??? K/??) on K processors, each with access to independent stochastic gradients of f with a second-moment bound B.

Then

In addition, NUQSGD requires at most N Q communication bits per iteration in expectation.

On nonconvex problems, (weaker) convergence guarantees can be established along the lines of, e.g., (Ghadimi and Lan, 2013 , Theorem 2.1).

How do QSGD and NUQSGD compare in terms of bounds on the expected number of communication bits required to achieve a given suboptimality gap ???

The quantity that controls our guarantee on the convergence speed in both algorithms is the variance upper bound, which in turn is controlled by the quantization schemes.

Note that the number of quantization levels, s, is usually a small number in practice.

On the other hand, the dimension, d, can be very large, especially in overparameterized networks.

In Figure 2 , we show that the quantization scheme underlying NUQSGD results in substantially smaller variance upper bounds for plausible ranges of s and d. Note that these bounds do not make any assumptions on the dataset or the structure of the network.

For any (nonrandom) number of iterations T , an upper bound, N A , holding uniformly over iterations k ??? T on the expected number of bits used by an algorithm A to communicate the gradient on iteration k, yields an upper bound T N A , on the expected number of bits communicated over T iterations by algorithm A. Taking T = T A,?? to be the (minimum) number of iterations needed to guarantee an expected suboptimality gap of ?? based on the properties of A, we obtain an upper bound, ?? A,?? = T A,?? N A , on the expected number of bits of communicated on a run expected to achieve a suboptimality gap of at most ??.

Theorem 5 (Expected number of communication bits).

Provided that s = o(log(d)) and

Focusing on the dominant terms in the expressions of overall number of communication bits required to guarantee a suboptimality gap of ??, we observe that NUQSGD provides slightly stronger guarantees.

Note that our stronger guarantees come without any assumption about the data.

In this section, we examine the practical performance of NUQSGD in terms of both convergence (accuracy) and speedup.

The goal is to empirically show that NUQSGD can provide the same performance and accuracy compared to the QSGDInf heuristic, which has no theoretical compression guarantees.

For this, we implement and test these three methods (NUQSGD, QSGD, and QSGDInf), together with the distributed full-precision SGD baseline, which we call SuperSGD.

We split our study across two axes: first, we examine the convergence of the methods and their induced variance.

Second, we provide an efficient implementation of all four methods in Pytorch using the Horovod communication back-end (Sergeev and Del Balso, 2018) , adapted to efficiently support quantization, and examine speedup relative to the full-precision baseline.

We investigate the impact of quantization on training performance by measuring loss, variance, accuracy, and speedup for ResNet models (He et al., 2016) applied to ImageNet (Deng et al., 2009 ) and CIFAR10 (Krizhevsky) .

We evaluate these methods on two image classification datasets: ImageNet and CIFAR10.

We train ResNet110 on CIFAR10 and ResNet18 on ImageNet with mini-batch size 128 and base learning rate 0.1.

In all experiments, momentum and weight decay are set to 0.9 and 10 ???4 , respectively.

The bucket size and the number of quantization bits are set to 8192 and 4, respectively.

We observe similar results in experiments with various bucket sizes and number of bits.

We simulate a scenario with k GPUs for all three quantization methods by estimating the gradient from k independent mini-batches and aggregating them after quantization and dequantization.

In Figure 3 (left and middle), we show the training loss with 8 GPUs.

We observe that NUQSGD and QSGDinf improve training loss compared to QSGD on ImageNet.

We observe significant gap in training loss on CIFAR10 where the gap grows as training proceeds.

We also observe similar performance gaps in test accuracy (provided in Appendix F).

In particular, unlike NUQSGD, QSGD does not achieve test accuracy of full-precision SGD.

Figure 3 (right) shows the mean normalized variance of the gradient (defined in Appendix F) versus training iteration on the trajectory of single-GPU SGD on CIFAR10.

These observations validate our theoretical results that NUQSGD has smaller variance for large models with small number of quantization bits.

Efficient Implementation and Speedup.

To examine speedup behavior, we implemented all quantization methods in Horovod (Sergeev and Del Balso, 2018) , a communication back-end supporting Pytorch, Tensorflow and MXNet.

Doing so efficiently requires non-trivial refactoring of this framework, since it does not support communication compression-our framework will be open-sourced upon publication.

Our implementation diverges slightly from the theoretical analysis.

First, Horovod applies "tensor fusion" to multiple layers, by merging the resulting gradient tensors for more efficient transmission.

This causes the gradients for different layers to be quantized together, which can lead to loss of accuracy (due to e.g. different normalization factors across the layers).

We addressed this by tuning the way in which tensor fusion is applied to the layers such that it minimizes the accuracy loss.

Second, we noticed that quantizing the gradients corresponding to the biases has a significant adverse effect on accuracy; since the communication impact of biases is negligible, we transmit them at full precision.

We apply this for all methods considered.

Finally, for efficiency reasons, we directly pack the quantized values into 32-bit numbers, without additional encoding.

We implemented compression and de-compression via efficient CUDA kernels.

Our baselines are full-precision SGD (SuperSGD), Error-Feedback SignSGD (Karimireddy et al., 2019) , and the QSGDinf heuristic, which we compare against the 4-bit and 8-bit NUQSGD variants executing the same pattern.

The implementation of the QSGDinf heuristic provides almost identical convergence numbers, and is sometimes omitted for visibility.

(QSGD yields inferior convergence on this dataset and is therefore omitted.)

All variants are implemented using a standard all-to-all reduction pattern.

Figures 4 (left) , (middle) show the execution time per epoch for ResNet34 and ResNet50 models on ImageNet, on a cluster machine with 8 NVIDIA 2080 Ti GPUs, for the hyperparameter values quoted above.

The results confirm the efficiency and scalability of the compressed variant, mainly due to the reduced communication volume.

We note that the overhead of compression and decompression is less than 1% of the batch computation time for NUQSGD.

Figure 4 (right) presents end-to-end speedup numbers (time versus accuracy) for ResNet50/ImageNet, executed on 4 GPUs, under the same hyperparameter settings as the full-precision baseline, with bucket size 512.

First, notice that NUQSGD variants match the target accuracy of the 32-bit model, with non-trivial speedup over the standard data-parallel variant, directly proportional to the perepoch speedup.

The QSGDinf heuristic yields similar accuracy and performance, and is therefore omitted.

Second, we found that unfortunately EF-SignSGD does not converge under these standard hyperparameter settings.

To address this issue, we performed a non-trivial amount of hyperparameter tuning for this algorithm: in particular, we found that the scaling factors and the bucket size must be carefully adjusted for convergence on ImageNet.

We were able to recover full accuracy with EF-SignSGD on ResNet50, but that the cost of quantizing into buckets of size 64.

Unfortunately, in this setting the algorithm transmits a non-trivial amount of scaling data, and the GPU implementation becomes less efficient due to error computation and reduced parallelism.

The end-to-end speedup of this tuned variant is inferior to NUQSGD-4bit, and only slightly superior to that of NUQSGD-8bit.

Please see Figure 9 in the Appendix and the accompanying text for details.

We study data-parallel and communication-efficient version of stochastic gradient descent.

Building on QSGD , we study a nonuniform quantization scheme.

We establish upper bounds on the variance of nonuniform quantization and the expected code-length.

In the overparametrized regime of interest, the former decreases as the number of quantization levels increases, while the latter increases with the number of quantization levels.

Thus, this scheme provides a trade-off between the communication efficiency and the convergence speed.

We compare NUQSGD and QSGD in terms of their variance bounds and the expected number of communication bits required to meet a certain convergence error, and show that NUQSGD provides stronger guarantees.

Experimental results are consistent with our theoretical results and confirm that NUQSGD matches the performance of QSGDinf when applied to practical deep models and datasets including ImageNet.

Thus, NUQSGD closes the gap between the theoretical guarantees of QSGD and empirical performance of QSGDinf.

One limitation of our study which we aim to address in future work is that we focus on all-to-all reduction patterns, which interact easily with communication compression.

In particular, we aim to examine the interaction between more complex reduction patterns, such as ring-based reductions (Hannun et al., 2014) , which may yield superior performance in bandwidthbottlenecked settings, but which interact with communication-compression in non-trivial ways, since they may lead a gradient to be quantized at each reduction step.

Read that bit plus N following bits; The encoding, ENCODE(v), of a stochastic gradient is as follows: We first encode the norm v using b bits where, in practice, we use standard 32-bit floating point encoding.

We then proceed in rounds, r = 0, 1, ?? ?? ?? .

On round r, having transmitted all nonzero coordinates up to and including t r , we transmit ERC(i r ) where t r+1 = t r + i r is either (i) the index of the first nonzero coordinate of h after t r (with t 0 = 0) or (ii) the index of the last nonzero coordinate.

In the former case, we then transmit one bit encoding the sign ?? t r+1 , transmit ERC(log(2 s+1 h t r+1 )), and proceed to the next round.

In the latter case, the encoding is complete after transmitting ?? t r+1 and ERC(log(2 s+1 h t r+1 )).

The DECODE function (for Algorithm 1) simply reads b bits to reconstruct v .

Using ERC ???1 , it decodes the index of the first nonzero coordinate, reads the bit indicating the sign, and then uses ERC ???1 again to determines the quantization level of this first nonzero coordinate.

The process proceeds in rounds, mimicking the encoding process, finishing when all coordinates have been decoded.

Like , we use Elias recursive coding (Elias, 1975, ERC) to encode positive integers.

ERC is simple and has several desirable properties, including the property that the coding scheme assigns shorter codes to smaller values, which makes sense in our scheme as they are more likely to occur.

Elias coding is a universal lossless integer coding scheme with a recursive encoding and decoding structure.

The Elias recursive coding scheme is summarized in Algorithm 2.

For any positive integer N, the following results are known for ERC

We first find a simple expression of the variance of Q s (v) for every arbitrary quantization scheme in the following lemma:

), and fix s ??? 1.

The variance of Q s (v) for general sequence of quantization levels is given by

where r i = |v i |/ v and p(r),s(r), ??(r) are defined in Section 3.1.

Proof.

Noting the random quantization is i.i.d over elements of a stochastic gradient, we can decom-

where

In the following, we consider NUQSGD algorithm withL = (0, 1/2 s , ?? ?? ?? , 2 s???1 /2 s , 1) as the quantization levels.

Then, h i (v, s)'s are defined in two cases based on which quantization interval r i falls into:

where p 1 r, s = 2 s r.

where p 2 r, s = 2 s??? j r ??? 1.

Note that Q s (0) = 0.

Let S j denote the coordinates of vector v whose elements fall into the ( j + 1)-th bin, i.e., S 0 {i :

Applying the result of Lemma 1, we have

where ?? j l j+1 ??? l j for j ??? {0, ?? ?? ?? , s}.

Substituting ?? 0 = 2 ???s and ?? j = 2 j???1???s for j ??? {1, ?? ?? ?? , s} into (9), we have

We first note that ??? i???S 0 p 1 (r i , s) ??? d and ??? i???S j+1 p 2 (r i , s) ??? d for all j, i.e., an upper bound on the variance of

Substituting the upper bounds in (11) and (12) into (10), an upper bound on the variance of Q s (v) is given by

The upper bound in (13) cannot be used directly as it depends on {d 0 , ?? ?? ?? , d s }.

Note that d j 's depend on quantization intervals.

In the following, we obtain an upper bound on E[ Q s (v) ??? v 2 ], which depends only on d and s. To do so, we need to use this lemma inspired by (Alistarh et al., 2017, Lemma A.5 ): Let ?? 0 count the number of nonzero components.

Lemma 2.

Let v ??? R d .

The expected number of nonzeros in Q s (v) is bounded above by

For each i ??? S 0 , Q s (v i ) becomes zero with probability 1 ??? 2 s r i , which results in

Using a similar argument as in the proof of Lemma 2, we have

We defined

. . . . . .

Noting that the coefficients of the additive terms in the upper bound in (13) are monotonically increasing with j, we can find an upper bound on (13), which gives (3) and completes the proof.

Let | ?? | denote the length of a binary string.

In this section, we find an upper bound on E[|ENCODE(v)], i.e., the expected number of communication bits per iteration.

Recall from Appendix A that the quantized gradient Q s (v) is determined by the tuple ( v , ?? ?? ??, h).

Write i 1 < i 2 < ?? ?? ??

< i h 0 for the indices of the h 0 nonzero entries of h. Let i 0 = 0.

The encoding produced by ENCODE(v) can be partitioned into two parts, R and E, such that, for j = 1, . . .

, h 0 ,

??? R contains the codewords ERC(i j ??? i j???1 ) encoding the runs of zeros; and

??? E contains the sign bits and codewords ERC(log{2 s+1 h i j }) encoding the normalized quantized coordinates. (Alistarh et al., 2017, Lemma A.3) , the properties of Elias encoding imply that

We now turn to bounding |E|.

The following result in inspired by (Alistarh et al., 2017, Lemma A.3) .

Lemma 3.

Fix a vector q such that q p p ??? P, let i 1 < i 2 < . . .

i q 0 be the indices of its q 0 nonzero entries, and assume each nonzero entry is of form of 2 k , for some positive integer k.

Then

Proof.

Applying property (1) for ERC (end of Appendix A), we have

where the last bound is obtained by Jensen's inequality.

Taking q = 2 s+1 h, we note that q 2 = 2 2s+2 h 2 and

By Lemma 3 applied to q and the upper bound (20),

Combining (19) and (21), we obtain an upper bound on the expected code-length:

where

It is not difficult to show that, for all k > 0, g 1 (x) x log k x is concave.

Note that g 1 is an increasing function up to x = k/e.

Defining g 2 (x) x log log C x and taking the second derivative, we have

Hence g 2 is also concave on x < C. Furthermore, g 2 is increasing up to some C/5 < x * < C/4.

We note that E[ h 0 ] ??? 2 2s + ??? d2 s following Lemma 2.

By assumption 2 2s + ??? d2 s ??? d/e, and so, Jensen's inequality and (22) lead us to (4).

Let g(w) and??(w) denote the full-precision and decoded stochastic gradients, respectively.

Then

By Theorem 2,

The result follows by Corollary 1.

Notice that the variance for NUQSGD and QSGDinf is lower than SGD for almost all the training and it decreases after the learning rate drops.

All methods except SGD simulate training using 8 GPUs.

SuperSGD applies no quantization to the gradients and represents the lowest variance we could hope to achieve.

Ignoring all but terms depending on d and s, we have T ?? = O(B/?? 2 ).

Following Theorems 2 and 3 for NUQSGD, ?? NUQSGD,?? = O(N Q ?? Q B/?? 2 ).

For QSGD, following the results of

In overparameterized networks, where d ??? 2 2s+1 , we have Figure 6 : Accuracy on the hold-out set on CIFAR10 (left) and on ImageNet (right) for training ResNet models from random initialization until convergence.

For CIFAR10, the hold-out set is the test set and for ImageNet, the hold-out set is the validation set.

In this section, we present further experimental results in a similar setting to Section 5.

In Figure 6 , we show the test accuracy for training ResNet110 on CIFAR10 and validation accuracy for training ResNet34 on ImageNet from random initialization until convergence (discussed in Section 5).

Similar to the training loss performance, we observe that NUQSGD and QSGDinf outperform QSGD in terms of test accuracy in both experiments.

In both experiments, unlike NUQSGD, QSGD does not recover the test accuracy of SGD.

The gap between NUQSGD and QSGD on ImageNet is significant.

We argue that this is achieved because NUQSGD and QSGDinf have lower variance relative to QSGD.

It turns out both training loss and generalization error can benefit from the reduced variance.

For different methods, the variance is measured on their own trajectories.

Note that the normalized variance of NUQSGD and QSGDinf is lower than SGD for almost the entire training.

It decreases on CIFAR10 after the learning rate drops and does not grow as much as SGD on ImageNet.

Since the variance depends on the optimization trajectory, these curves are not directly comparable.

Rather the general trend should be studied.

We also measure the variance and normalized variance at fixed snapshots during training by evaluating multiple gradient estimates using each quantization method.

All methods are evaluated on the same trajectory traversed by the single-GPU SGD.

These plots answer this specific question: What would the variance of the first gradient estimate be if one were to train using SGD for any number of iterations then continue the optimization using another method?

The entire future trajectory may change by taking a single good or bad step.

We can study the variance along any trajectory.

However, the trajectory of SGD is particularly interesting because it covers a subset of points in the parameter space that is likely to be traversed by any first-order optimizer.

For multi-dimensional parameter space, we average the variance of each dimension.

Figure 5 (left), shows the variance of the gradient estimates on the trajectory of single-GPU SGD on CIFAR10.

We observe that QSGD has particularly high variance, while QSGDinf and NUQSGD have lower variance than single-GPU SGD.

We also propose another measure of stochasticity, normalized variance, that is the variance normalized by the norm of the gradient.

The mean normalized variance can be expressed as

where l(w; z) denotes the loss of the model parametrized by w on sample z and subscript A refers to randomness in the algorithm, e.g., randomness in sampling and quantization.

Normalized variance can be interpreted as the inverse of Signal to Noise Ratio (SNR) for each dimension.

We argue that the noise in optimization is more troubling when it is significantly larger than the gradient.

For sources of noise such as quantization that stay constant during training, their negative impact might only be observed when the norm of the gradient becomes small.

Figure 5 (right) shows the mean normalized variance of the gradient versus training iteration.

Observe that the normalized variance for QSGD stays relatively constant while the unnormalized variance of QSGD drops after the learning rate drops.

It shows that the quantization noise of QSGD can cause slower convergence at the end of the training than at the beginning.

In Figure 7 , we show the mean normalized variance of the gradient versus training iteration on CIFAR10 and ImageNet.

For different methods, the variance is measured on their own trajectories.

Since the variance depends on the optimization trajectory, these curves are not directly comparable.

Rather the general trend should be studied.

ResNet152 Weak Scaling.

In Figure 8 , we present the weak scaling results for ResNet152/ImageNet.

Each of the GPUs receives a batch of size 8, and we therefore scale up the global batch size by the number of nodes.

The results exhibit the same superior scaling behavior for NUQSGD relative to the uncompressed baseline.

EF-SignSGD Convergence.

In Figure 9 , we present a performance comparison for NUQSGD variants (bucket size 512) and a convergent variant of EF-SignSGD with significant levels of parameter tuning for convergence.

We believe this to be the first experiment to show convergence of the latter method at ImageNet scale, as the original paper only considers the CIFAR dataset.

For convergence, we have tuned the choice of scaling factor and the granularity at which quantization is applied (bucket size).

We have also considered learning rate tuning, but that did not appear to prevent divergence in the early stages of training for this model.

We did not attempt warm start, since that would significantly decrease the practicality of the algorithm.

We have found that bucket size 64 is the highest at which the algorithm will still converge on this model and dataset, and found 1-bit SGD scaling (Seide et al., 2014) , which consists of taking sums over positives and over negatives for each bucket, to yield good results.

The experiments are executed on a machine with 8 NVIDIA Titan X GPUs, and batch size 256, and can be found in Figure 9 .

Under these hyperparameter values the EF-SignSGD algorithm sends 128 bits per each bucket of 64 values (32 for each scaling factor, and 64 for the signs), doubling its baseline communication cost.

Moreover, the GPU implementation is not as efficient, as error feedback must be computed and updated at every step, and there is less parallelism to leverage inside each bucket.

This explains the fact that the end-to-end performance is in fact close to that of the 8-bit NUQSGD variant, and inferior to 4-bit NUQSGD.

In the following theorem, we show that for any given set of levels, there exists a distribution of points with dimension d such that the variance is in ???( ??? d), and so our bound is tight in d.

The variance optimization problem R 2 is an integer nonconvex problem.

We can obtain an upper bound on the optimal objective of problem R 2 by relaxing the integer constraint as follows.

The resulting QSQP is shown as follows: Note that problem Q 1 can be solved efficiently using standard standard interior point-based solvers, e.g., CVX (Boyd and Vandenberghe, 2004) .

In the following, we develop a coarser analysis that yields an upper bound expressed as the optimal value to an LP.

Theorem 8 (LP bound).

Let v ??? R d .

An upper bound on the nonuniform quantization of v is given by ?? LP v 2 where ?? LP is the optimal value of the following LP:

Corollary 2 (Optimal level).

For the special case with s = 1, the optimal level to minimize the worst-case bound obtained from problem P 1 is given by l * 1 = 1/2.

Proof.

For s = 1, problem P 1 is given by P 0 : max

Note that the objective of P 0 is monotonically increasing in (d 0 , d 1 ).

It is not difficult to verify that the optimal (d * 0 , d * 1 ) is a corner point on the boundary line of the feasibility region of P 0 .

Geometrical representation shows that that candidates for an optimal solution are (d ??? (1/l 1 ) 2 , (1/l 1 ) 2 ) and (d, 0).

Substituting into the objective of P 0 , the optimal value of P 0 is given by

Finally, note that ?? 0 = ?? 1 = 1/2 minimizes the optimal value of P 0 (38).

In this section, we focus on the special case of exponentially spaced collection of levels of the form L p = (0, p s , ?? ?? ?? , p 2 , p, 1) for p ??? [0, 1] and an integer number of levels, s. In this case, we have ?? 0 = p s and ?? j = (1 ??? p)p s??? j for j = 1, ?? ?? ?? , s.

For any given s and d, we can solve the corresponding quadratic and linear programs efficiently to find the worst-case variance bound.

As a bonus, we can find the optimal value of p that minimizes the worst-case variance bound.

In Figure 10 , we show the numerical results obtained by solving QCQP Q 1 with L p versus p using CVX (Boyd and Vandenberghe, 2004) .

In Figure 10 (left), we fix d and vary s, while in Figure 10 (right), we fix s and vary d. As expected, we note that the variance upper bound increases as d increases and the variance upper bound decreases as s increases.

We observe that our current scheme is nearly optimal (in the worst-case sense) in some cases.

Further, the optimal value of p shifts to the right as d increases and shifts to the left as s increases.

We can obtain convergence guarantees to various learning problems where we have convergence guarantees for SGD under standard assumptions.

On nonconvex problems, (weaker) convergence guarantees can be established along the lines of, e.g., (Ghadimi and Lan, 2013 , Theorem 2.1).

In particular, NUQSGD is guaranteed to converge to a local minima for smooth general loss functions.

Theorem 9 (NUQSGD for smooth nonconvex optimization).

Let f : ??? ??? R denote a possibly nonconvex and ?? -smooth function.

Let w 0 ??? ??? denote an initial point, ?? Q be defined as in Theorem 2, T ??? Z >0 , and f * = inf w?????? f (w).

Suppose that Algorithm 1 is executed for T iterations with a learning rate ?? = O(1/?? ) on K processors, each with access to independent stochastic gradients of f with a second-moment bound B. Then there exists a random stopping time R ??? {0, ?? ?? ?? , T } such that NUQSGD guarantees E[ ??? f (w R ) 2 ] ??? ?? where ?? = O ?? ( f (w 0 ) ??? f * )/T + (1 + ?? Q )B .

<|TLDR|>

@highlight

NUQSGD closes the gap between the theoretical guarantees of QSGD and the empirical performance of QSGDinf.