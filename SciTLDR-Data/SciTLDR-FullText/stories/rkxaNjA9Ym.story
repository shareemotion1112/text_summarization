The high computational and parameter complexity of neural networks makes their training very slow and difficult to deploy on energy and storage-constrained comput- ing systems.

Many network complexity reduction techniques have been proposed including fixed-point implementation.

However, a systematic approach for design- ing full fixed-point training and inference of deep neural networks remains elusive.

We describe a precision assignment methodology for neural network training in which all network parameters, i.e., activations and weights in the feedforward path, gradients and weight accumulators in the feedback path, are assigned close to minimal precision.

The precision assignment is derived analytically and enables tracking the convergence behavior of the full precision training, known to converge a priori.

Thus, our work leads to a systematic methodology of determining suit- able precision for fixed-point training.

The near optimality (minimality) of the resulting precision assignment is validated empirically for four networks on the CIFAR-10, CIFAR-100, and SVHN datasets.

The complexity reduction arising from our approach is compared with other fixed-point neural network designs.

Though deep neural networks (DNNs) have established themselves as powerful predictive models achieving human-level accuracy on many machine learning tasks BID12 , their excellent performance has been achieved at the expense of a very high computational and parameter complexity.

For instance, AlexNet BID17 requires over 800 × 10 6 multiply-accumulates (MACs) per image and has 60 million parameters, while Deepface (Taigman et al., 2014) requires over 500 × 10 6 MACs/image and involves more than 120 million parameters.

DNNs' enormous computational and parameter complexity leads to high energy consumption BID4 , makes their training via the stochastic gradient descent (SGD) algorithm very slow often requiring hours and days BID9 , and inhibits their deployment on energy and resource-constrained platforms such as mobile devices and autonomous agents.

A fundamental problem contributing to the high computational and parameter complexity of DNNs is their realization using 32-b floating-point (FL) arithmetic in GPUs and CPUs.

Reduced-precision representations such as quantized FL (QFL) and fixed-point (FX) have been employed in various combinations to both training and inference.

Many employ FX during inference but train in FL, e.g., fully binarized neural networks BID13 use 1-b FX in the forward inference path but the network is trained in 32-b FL.

Similarly, BID10 employs 16-b FX for all tensors except for the internal accumulators which use 32-b FL, and 3-level QFL gradients were employed (Wen et al., 2017; BID0 to accelerate training in a distributed setting.

Note that while QFL reduces storage and communication costs, it does not reduce the computational complexity as the arithmetic remains in 32-b FL.Thus, none of the previous works address the fundamental problem of realizing true fixed-point DNN training, i.e., an SGD algorithm in which all parameters/variables and all computations are implemented in FX with minimum precision required to guarantee the network's inference/prediction accuracy and training convergence.

The reasons for this gap are numerous including: 1) quantization Step 1: Forward PropagationStep 2: Back PropagationStep 3: Update errors propagate to the network output thereby directly affecting its accuracy (Lin et al., 2016) ; 2) precision requirements of different variables in a network are interdependent and involve hard-toquantify trade-offs (Sakr et al., 2017) ; 3) proper quantization requires the knowledge of the dynamic range which may not be available (Pascanu et al., 2013) ; and 4) quantization errors may accumulate during training and can lead to stability issues BID10 .Our work makes a major advance in closing this gap by proposing a systematic methodology to obtain close-to-minimum per-layer precision requirements of an FX network that guarantees statistical similarity with full precision training.

In particular, we jointly address the challenges of quantization noise, inter-layer and intra-layer precision trade-offs, dynamic range, and stability.

As in (Sakr et al., 2017) , we do assume that a fully-trained baseline FL network exists and one can observe its learning behavior.

While, in principle, such assumption requires extra FL computation prior to FX training, it is to be noted that much of training is done in FL anyway.

For instance, FL training is used in order to establish benchmarking baselines such as AlexNet BID17 , VGG-Net (Simonyan and Zisserman, 2014) , and ResNet BID12 , to name a few.

Even if that is not the case, in practice, this assumption can be accounted for via a warm-up FL training on a small held-out portion of the dataset BID6 .Applying our methodology to three benchmarks reveals several lessons.

First and foremost, our work shows that it is possible to FX quantize all variables including back-propagated gradients even though their dynamic range is unknown BID15 .

Second, we find that the per-layer weight precision requirements decrease from the input to the output while those of the activation gradients and weight accumulators increase.

Furthermore, the precision requirements for residual networks are found to be uniform across layers.

Finally, hyper-precision reduction techniques such as weight and activation binarization BID13 or gradient ternarization (Wen et al., 2017) are not as efficient as our methodology since these do not address the fundamental problem of realizing true fixed-point DNN training.

We demonstrate FX training on three deep learning benchmarks (CIFAR-10, CIFAR-100, SVHN) achieving high fidelity to our FL baseline in that we observe no loss of accuracy higher then 0.56% in all of our experiments.

Our precision assignment is further shown to be within 1-b per-tensor of the minimum.

We show that our precision assignment methodology reduces representational, computational, and communication costs of training by up to 6×, 8×, and 4×, respectively, compared to the FL baseline and related works.

We consider a L-layer DNN deployed on a M -class classification task using the setup in FIG1 .

We denote the precision configuration as the DISPLAYFORM0 whose l th row consists of the precision (in bits) of weight DISPLAYFORM1 ), and internal weight accumulator W DISPLAYFORM2 ) tensors at layer l. This DNN quantization setup is summarized in Appendix A.

We present definitions/constraints related to fixed-point arithmetic based on the design of fixed-point adaptive filters and signal processing systems (Parhi, 2007):• A signed fixed-point scalar a with precision B A and binary representation DISPLAYFORM0 2 −i a i , where r A is the predetermined dynamic range (PDR) of a. The PDR is constrained to be a constant power of 2 to minimize hardware overhead.• An unsigned fixed-point scalar a with precision B A and binary representation DISPLAYFORM1 • The precision B A is determined as: B A = log 2 r A ∆ A + 1, where ∆ A is the quantization step size which is the value of the least significant bit (LSB).• An additive model for quantization is assumed: a =ã + q a , where a is the fixed-point number obtained by quantizing the floating-point scalarã, q a is a random variable uniformly distributed on the interval − 12 .

The notion of quantization noise is most useful when there is limited knowledge of the distribution ofã.• The relative quantization bias η A is the offset: DISPLAYFORM2 , where the first unbiased quantization level µ A = E ã ã ∈ I 1 and DISPLAYFORM3 .

The notion of quantization bias is useful when there is some knowledge of the distribution ofã.• The reflected quantization noise variance from a tensor T to a scalar α = f (T ), for an arbitrary function f (), is : DISPLAYFORM4 , where ∆ T is the quantization step of T and E T →α is the quantization noise gain from T to α.• The clipping rate β T of a tensor T is the probability: β T = Pr ({|t| ≥ r T : t ∈ T }), where r T is the PDR of T .

We use a set of metrics inspired by those introduced by Sakr et al. (2017) which have also been used by Wu et al. (2018a) .

These metrics are algorithmic in nature which makes them easily reproducible.• Representational Cost for weights (C W ) and activations (C A ): DISPLAYFORM0 , which equals the total number of bits needed to represent the weights, weight gradients, and internal weight accumulators (C W ), and those for activations and activation gradients (C A ).

DISPLAYFORM1 , where D l is the dimensionality of the dot product needed to compute one output activation at layer l.

This cost is a measure of the number of 1-b full adders (FAs) utilized for all multiplications in one back-prop iteration.

DISPLAYFORM2 , which represents cost of communicating weight gradients in a distributed setting (Wen et al., 2017; BID0 .

We aim to obtain a minimal or close-to-minimal precision configuration C o of a FX network such that the mismatch probability p m = Pr{Ŷ f l =Ŷ f x } between its predicted label (Ŷ f x ) and that of an associated FL network (Ŷ f l ) is bounded, and the convergence behavior of the two networks is similar.

Hence, we require that: (1) all quantization noise sources in the forward path contribute identically to the mismatch budget p m (Sakr et al., 2017) , (2) the gradients be properly clipped in order to limit the dynamic range (Pascanu et al., 2013) , (3) the accumulation of quantization noise bias in the weight updates be limited BID10 , (4) the quantization noise in activation gradients be limited as these are back-propagated to calculate the weight gradients, and (5) the precision of weight accumulators should be set so as to avoid premature stoppage of convergence BID7 .

The above insights can be formally described via the following five quantization criteria.

Criterion 1.

Equalizing Feedforward Quantization Noise (EFQN) Criterion.

The reflected quantization noise variances onto the mismatch probability p m from all feedforward weights DISPLAYFORM0 Criterion 2.

Gradient Clipping (GC) Criterion.

The clipping rates of weight ({β DISPLAYFORM1 ) gradients should be less than a maximum value β 0 : DISPLAYFORM2 Criterion 3.

Relative Quantization Bias (RQB) Criterion.

The relative quantization bias of weight gradients ({η DISPLAYFORM3 DISPLAYFORM4 where Σ l is the total sum of element-wise variances of G (W ) l .

Criterion 5.

Accumulator Stopping (AS) Criterion.

The quantization noise of the internal accumulator should be zero, equivalently: DISPLAYFORM5 is the reflected quantization noise variance from W DISPLAYFORM6 , its total sum of element-wise variances.

Further explanations and motivations behind the above criteria are presented in Appendix B.

The following claim ensures the satisfiability of the above criteria.

This leads to closed form expressions for the precision requirements we are seeking and completes our methodology.

The validity of the claim is proved in Appendix C. Claim 1.

Satisfiability of Quantization Criteria.

The five quantization criteria (EFQN, GC, RQB, BQN, AS) are satisfied if:• The precisions B W l and B A l are set as follows: DISPLAYFORM7 for l = 1 . . .

L, where rnd() denotes the rounding operation, E W l →pm and E A l →pm are the weight and activation quantization noise gains at layer l, respectively, B (min) is a reference minimum precision, and DISPLAYFORM8 Published as a conference paper at ICLR 2019• The weight and activation gradients PDRs are lower bounded as follows: DISPLAYFORM9 where σ DISPLAYFORM10 are the largest recorded estimates of the weight and activation gradients DISPLAYFORM11 , respectively.• The weight and activation gradients quantization step sizes are upper bounded as follows: DISPLAYFORM12 where σ DISPLAYFORM13 is the largest singular value of the square-Jacobian (Jacobian matrix with squared entries) of G DISPLAYFORM14 l+1 .•

The accumulator PDR and step size satisfy: DISPLAYFORM15 where γ (min) is the smallest value of the learning rate used during training.

Practical considerations: Note that one of the 2L feedforward precisions will equal B (min) .

The formulas to compute the quantization noise gains are given in Appendix C and require only one forward-backward pass on an estimation set.

We would like the EFQN criterion to hold upon convergence; hence, FORMULA19 is computed using the converged model from the FL baseline.

For backward signals, setting the values of PDR and LSB is sufficient to determine the precision using the identity B A = log 2 r A ∆ A + 1, as explained in Section 2.1.

As per Claim 1, estimates of the second order statistics, e.g., DISPLAYFORM16 , of the gradient tensors, are required.

These are obtained via tensor spatial averaging, so that one estimate per tensor is required, and updated in a moving window fashion, as is done for normalization parameters in BatchNorm BID14 .

Furthermore, it might seem that computing the Jacobian in (3) is a difficult task; however, the values of its elements are already computed by the back-prop algorithm, requiring no additional computations (see Appendix C).

Thus, the Jacobians (at different layers) are also estimated during training.

Due to the typical very large size of modern neural networks, we average the Jacobians spatially, i.e., the activations are aggregated across channels and mini-batches while weights are aggregated across filters.

This is again inspired by the work on Batch Normalization BID14 and makes the probed Jacobians much smaller.

We conduct numerical simulations in order to illustrate the validity of the predicted precision configuration C o and investigate its minimality and benefits.

We employ three deep learning benchmarking datasets: CIFAR-10, CIFAR-100 BID16 SVHN (Netzer et al., 2011) .

All experiments were done using a Pascal P100 NVIDIA GPU.

We train the following networks:• CIFAR-10 ConvNet: a 9-layer convolutional neural network trained on the CIFAR-10 dataset described as 2 DISPLAYFORM0 where C3 denotes 3 × 3 convolutions, M P 2 denotes 2 × 2 max pooling operation, and F C denotes fully connected layers.• SVHN ConvNet: the same network as the CIFAR-10 ConvNet, but trained on the SVHN dataset.• CIFAR-10 ResNet: a wide deep residual network (Zagoruyko and Komodakis, 2016) with ResNet-20 architecture but having 8 times as many channels per layer compared to BID12 .• CIFAR-100 ResNet: same network as CIFAR-10 ResNet save for the last layer to match the number of classes (100) in CIFAR-100.A step by step description of the application of our method to the above four networks is provided in Appendix E. We hope the inclusion of these steps would: (1) clarify any ambiguity the reader may have from the previous section and (2) facilitate the reproduction of our results.

The precision configuration C o , with target p m ≤ 1%, β 0 ≤ 5%, and η 0 ≤ 1%, via our proposed method is depicted in FIG4 for each of the four networks considered.

We observe that C o is dependent on the network type.

Indeed, the precisions of the two ConvNets follow similar trends as do those the two ResNets.

Furthermore, the following observations are made for the ConvNets:• weight precision B W l decreases as depth increases.

This is consistent with the observation that weight perturbations in the earlier layers are the most destructive (Raghu et al., 2017) .• the precisions of activation gradients (B G (A) l ) and internal weight accumulators (B W (acc) l ) increases as depth increases which we interpret as follows: (1) the back-propagation of gradients is the dual of the forward-propagation of activations, and (2) accumulators store the most information as their precision is the highest.• the precisions of the weight gradients (B G (W ) l ) and activations (B A l ) are relatively constant across layers.

Interestingly, for ResNets, the precision is mostly uniform across the layers.

Furthermore, the gap between B W (acc) l and the other precisions is not as pronounced as in the case of ConvNets.

This suggests that information is spread equally among all signals which we speculate is due to the shortcut connections preventing the shattering of information BID2 .

FIG5 indicate that C o leads to convergence and consistently track FL curves with close fidelity.

This validates our analysis and justifies the choice of C o .

To determine that C o is a close-to-minimal precision assignment, we compare it with: (a) DISPLAYFORM0 is an L × 5 matrix with each entry equal to 1 3 , i.e., we perturb C o by 1-b in either direction.

FIG5 also contains the convergence curves for the two new configurations.

As shown, C −1 always results in a noticeable gap compared to C o for both the loss function (except for the CIFAR-10 ResNet) and the test error.

Furthermore, C +1 offers no observable improvements over C o (except for the test error of CIFAR-10 ConvNet).

These results support our contention that C o is close-to-minimal in that increasing the precision above C o leads to diminishing returns while reducing precision below C o leads to a noticeable degradation in accuracy.

Additional experimental results provided in Appendix D support our contention regarding the near minimality of C o .

Furthermore, by studying the impact of quantizing specific tensors we determine that that the accuracy is most sensitive to the precision assigned to weights and activation gradients.

We would like to quantify the reduction in training cost and expense in terms of accuracy resulting from our proposed method and compare them with those of other methods.

Importantly, for a fair comparison, the same network architecture and training procedure are used.

We report C W , C A , C M , C C , and test error, for each of the four networks considered for the following training methods:• baseline FL training and FX training using C o , • binarized network (BN) training, where feedforward weights and activations are binary (constrained to ±1) while gradients and accumulators are in floating-point and activation gradients are back- propagated via the straight through estimator BID3 as was done in BID13 , • fixed-point training with stochastic quantization (SQ).

As was done in BID10 , we quantize feedforward weights and activations as well as all gradients, but accumulators are kept in floating-point.

The precision configuration (excluding accumulators) is inherited from C o (hence we determine exactly how much stochastic quantization helps), • training with ternarized gradients (TG) as was done in TernGrad (Wen et al., 2017) .

All computations are done in floating-point but weight gradients are ternarized according to the instantaneous tensor spatial standard deviations {−2.5σ, 0, 2.5σ} as was suggested by Wen et al. (2017) .

To compute costs, we assume all weight gradients use two bits although they are not really fixed-point and do require computation of 32-b floating-point scalars for every tensor.

The comparison is presented in TAB3 .

The first observation is a massive complexity reduction compared to FL.

For instance, for the CIFAR-10 ConvNet, the complexity reduction is 2.6× (= 148/56.5), 5.5× (= 9.3/1.7), 7.9× (= 94.4/11.9), and 3.5× (= 49/14) for C W , C A , C M , and C C , respectively.

Similar trends are observed for the other four networks.

Such complexity reduction comes at the expense of no more than 0.56% increase in test error.

For the CIFAR-100 network, the accuracy when training in fixed-point is even better than that of the baseline.

The representational and communication costs of BN is significantly greater than that of FX because the gradients and accumulators are kept in full precision, which masks the benefits of binarizing feedforward tensors.

However, benefits are noticeable when considering the computational cost which is lowest as binarization eliminates multiplications.

Furthermore, binarization causes a severe accuracy drop for the ConvNets but surprisingly not for the ResNets.

We speculate that this is due to the high dimensional geometry of ResNets BID1 .As for SQ, since C o was inherited, all costs are identical to FX, save for C W which is larger due to full precision accumulators.

Furthermore, SQ has a positive effect only on the CIFAR-10 ConvNet where it clearly acted as a regularizer.

TG does not provide complexity reductions in terms of representational and computational costs which is expected as it only compresses weight gradients.

Additionally, the resulting accuracy is slightly worse than that of all other considered schemes, including FX.

Naturally, it has the lowest communication cost as weight gradients are quantized to just 2-b.

Many works have addressed the general problem of reduced precision/complexity deep learning.

Reducing the complexity of inference (forward path): several research efforts have addressed the problem of realizing a DNN's inference path in FX.

For instance, the works in (Lin et al., 2016; Sakr et al., 2017) address the problem of precision assignment.

While Lin et al. FORMULA19 proposed a non-uniform precision assignment using the signal-to-quantization-noise ratio (SQNR) metric, Sakr et al. (2017) analytically quantified the trade-off between activation and weight precisions while providing minimal precision requirements of the inference path computations that bounds the probability p m of a mismatch between predicted labels of the FX and its FL counterpart.

An orthogonal approach which can be applied on top of quantization is pruning BID11 .

While significant inference efficiency can be achieved, this approach incurs a substantial training overhead.

A subset of the FX training problem was addressed in binary weighted neural networks BID5 Rastegari et al., 2016) and fully binarized neural networks BID13 , where direct training of neural networks with pre-determined precisions in the inference path was explored with the feedback path computations being done in 32-b FL.Reducing the complexity of training (backward path): finite-precision training was explored in BID10 which employed stochastic quantization in order to counter quantization bias accumulation in the weight updates.

This was done by quantizing all tensors to 16-b FX, except for the internal accumulators which were stored in a 32-b floating-point format.

An important distinction our work makes is the circumvention of the overhead of implementing stochastic quantization BID13 .

Similarly, DoReFa-Net (Zhou et al., 2016) stores internal weight representations in 32-b FL, but quantizes the remaining tensors more aggressively.

Thus arises the need to re-scale and re-compute in floating-point format, which our work avoids.

Finally, BID15 suggests a new number format -Flexpoint -and were able to train neural networks using slightly 16-b per tensor element, with 5 shared exponent bits and a per-tensor dynamic range tracking algorithm.

Such tracking causes a hardware overhead bypassed by our work since the arithmetic is purely FX.

Augmenting Flexpoint with stochastic quantization effectively results in WAGE (Wu et al., 2018b) , and enables integer quantization of each tensor.

As seen above, none of the prior works address the problem of predicting precision requirements of all training signals.

Furthermore, the choice of precision is made in an ad-hoc manner.

In contrast, we propose a systematic methodology to determine close-to-minimal precision requirements for FX-only training of deep neural networks.

In this paper, we have presented a study of precision requirements in a typical back-propagation based training procedure of neural networks.

Using a set of quantization criteria, we have presented a precision assignment methodology for which FX training is made statistically similar to the FL baseline, known to converge a priori.

We realized FX training of four networks on the CIFAR-10, CIFAR-100, and SVHN datasets and quantified the associated complexity reduction gains in terms costs of training.

We also showed that our precision assignment is nearly minimal.

The presented work relies on the statistics of all tensors being quantized during training.

This necessitates an initial baseline run in floating-point which can be costly.

An open problem is to predict a suitable precision configuration by only observing the data statistics and the network architecture.

Future work can leverage the analysis presented in this paper to enhance the effectiveness of other network complexity reduction approaches.

For instance, weight pruning can be viewed as a coarse quantization process (quantize to zero) and thus can potentially be done in a targeted manner by leveraging the information provided by noise gains.

Furthermore, parameter sharing and clustering can be viewed as a form of vector quantization which presents yet another opportunity to leverage our method for complexity reduction.

The quantization setup depicted in FIG1 is summarized as follows:• Feedforward computation at layer l: DISPLAYFORM0 where f l () is the function implemented at layer l, A l (A l+1 ) is the activation tensor at layer l (l + 1) quantized to a normalized unsigned fixed-point format with precision B A l (B A l+1 ), and W l is the weight tensor at layer l quantized to a normalized signed fixed-point format with precision B W l .

We further assume the use of a ReLU-like activation function with a clipping level of 2 and a max-norm constraint on the weights which are clipped between [−1, 1] at every iteration.• Back-propagation of activation gradients at layer l: DISPLAYFORM1 where g l () (A) is the function that back-propagates the activation gradients at layer l, G DISPLAYFORM2 l+1 ) is the activation gradient tensor at layer l (l + 1) quantized to a signed fixed-point format with precision B G DISPLAYFORM3 ).•

Back-propagation of weight gradient tensor G (W ) l at layer l: DISPLAYFORM4 where g • Internal weight accumulator update at layer l: DISPLAYFORM5 where U () is the update function, γ is the learning rate, and W (acc) l is the internal weight accumulator tensor at layer l quantized to signed fixed-point with precision B W

Criterion 1 (EFQN) is used to ensure that all feedforward quantization noise sources contribute equally to the p m budget.

Indeed, if one of the 2L reflected quantization noise variances from the feedforward tensors onto p m , say V Wi→pm for i ∈ {1, . . .

, L}, largely dominates all others, it would imply that all tensors but W i are overly quantized.

It would therefore be necessary to either increase the precision of W i or decrease the precisions of all other tensors.

The application of Criterion 1 (EFQN) through the closed form expression (1) in Claim 1 solves this issue avoiding the need for a trial-and-error approach.

Because FX numbers require a constant PDR, clipping of gradients is needed since their dynamic range is arbitrary.

Ideally, a very small PDR would be preferred in order to obtain quantization steps of small magnitude, and hence less quantization noise.

We can draw parallels from signal processing theory, where it is known that for a given quantizer, the signal-to-quantization-noise ratio (SQNR) is equal to SQN R(dB) = 6B + 4.78 − P AR where P AR is the peak-to-average ratio, proportional to the PDR.

Thus, we would like to reduce the PDR as much as possible in order to increase the SQNR for a given precision.

However, this comes at the risk of overflows (due to clipping).

Criterion 2 (GC) addresses this trade-off between quantization noise and overflow errors.

Since the back-propagation training procedure is an iterative one, it is important to ensure that any form of bias does not corrupt the weight update accumulation in a positive feedback manner.

FX quantization, being a uniform one, is likely to induce such bias when quantized quantities, most notable gradients, are not uniformly distributed.

Criterion 3 (RQB) addresses this issue by using η as proxy to this bias accumulation a function of quantization step size and ensuring that its worst case value is small in magnitude.

Criterion 4 (BQN) is in fact an extension of Criterion 1 (EFQN), but for the back-propagation phase.

Indeed, once the precision (and hence quantization noise) of weight gradients is set as per Criterion 3 (RQB), it is needed to ensure that the quantization noise source at the activation gradients would not contribute more noise to the updates.

This criterion sets the quantization step of the activation gradients.

Criterion 5 (AS) ties together feedforward and gradient precisions through the weight accumulators.

It is required to increment/decrement the feedforward weights whenever the accumulated updates cross-over the weight quantization threshold.

This is used to set the PDR of the weight accumulators.

Furthermore, since the precision of weight gradients has already been designed to account for quantization noise (through Criteria 2-4), the criterion requires that the accumulators do not cause additional noise.

The validity of Claim 1 is derived from the following five lemmas.

Note that each lemma addresses the satisfiability of one of the five quantization criteria presented in the main text and corresponds to part of Claim 1.

Lemma 1.

The EFQN criterion holds if the precisions B W l and B A l are set as follows: DISPLAYFORM0 for l = 1 . . .

L, where rnd() denotes the rounding operation, B (min) is a reference minimum precision, and E (min) is given by: DISPLAYFORM1 Proof.

By definition of the reflected quantization noise variance, the EFQN, by definition, is satisfied if: DISPLAYFORM2 where the quantization noise gains are given by: DISPLAYFORM3 are the soft outputs and ZŶ f l is the soft output corresponding toŶ f l .

The expressions for these quantization gains are obtained by linearly expanding (across layers) those used in (Sakr et al., 2017) .

Note that a second order upper bound is used as a surrogate expression for p m .From the definition of quantization step size, the above is equivalent to: DISPLAYFORM4 Let E (min) be as defined in FORMULA38 : DISPLAYFORM5 .

We can divide each term by E (min) : DISPLAYFORM6 where each term is positive, so that we can take square roots and logarithms such that: DISPLAYFORM7 Thus we equate all of the above to a reference precision B (min) yielding: DISPLAYFORM8 for l = 1 . . .

L. Note that because E (min) is the least quantization noise gain, it is equal to one of the above quantization noise gains so that the corresponding precision actually equates B (min) .

As precisions must be integer valued, each of DISPLAYFORM9 have to be integers, and thus a rounding operation is to be applied on all logarithm terms.

Doing so results in (1) from Lemma 1 which completes this proof.

Lemma 2.

The GC criterion holds for β 0 = 5% provided the weight and activation gradients pre-defined dynamic ranges (PDRs) are lower bounded as follows: DISPLAYFORM10 are the largest ever recorded estimates of the weight and activation DISPLAYFORM11 , respectively.

Proof.

Let us consider the case of weight gradients.

The GC criterion, by definition requires: DISPLAYFORM12 Typically, weight gradients are obtained by computing the derivatives of a loss function with respect to a mini-batch.

By linearity of derivatives, weight gradients are themselves averages of instantaneous derivatives and are hence expected to follow a Gaussian distribution by application of the Central Limit Theorem.

Furthermore, the gradient mean was estimated during baseline training and was found to oscillate around zero.

DISPLAYFORM13 where we used the fact that a Gaussian distribution is symmetric and Q() is the elementary Q-function, which is a decreasing function.

Thus, in the worst case, we have: DISPLAYFORM14 Hence, for a PDR as suggested by the lower bound in (2): DISPLAYFORM15 in Lemma 2, we obtain the upper bound: DISPLAYFORM16 which means the GC criterion holds and completes the proof.

For activation gradients, the same reasoning applies, but the choice of a larger PDR in (2): DISPLAYFORM17 than for weight gradients is due to the fact that the true dynamic range of the activation gradients is larger than the value indicated by the second moment.

This stems from the use of activation functions such as ReLU which make the activation gradients sparse.

We also recommend increasing the PDR even more when using regularizers that sparsify gradients such as Dropout (Srivastava et al., 2014) or Maxout BID8 .Lemma 3.

The RQB criterion holds for η 0 = 1% provided the weight gradient quantization step size is upper bounded as follows: DISPLAYFORM18 is the smallest ever recorded estimate of σ G DISPLAYFORM19 We close this appendix by discussing the approximation made by invoking the Central Limit Theorem (CLT) in the proofs of Lemmas 2 & 3.

This approximation was made because, typically, a backpropagation iteration computes gradients of a loss function being averaged over a mini-batch of samples.

By linearity of derivatives, the gradients themselves are averages, which warrants the invocation of the CLT.

However, the CLT is an asymptotic result which might be imprecise for a finite number of samples.

In typical training of neural networks, the number of samples, or mini-batch size, is in the range of hundreds or thousands BID9 .

It is therefore important to quantify the preciseness, or lack thereof, of the CLT approximation.

On way to do so is via the Berry-Essen Theorem which considers the average of n independent, identically distributed random variables with finite absolute third moment ρ and standard deviation σ.

The worst case deviation of the cumulative distribution of the true average from the of the approximated Gaussian random variable (via the CLT), also known as the Kolmogorov-Smirnov distance, KS, is upper bounded as follows: KS < Cρ √ nσ 3 , where C < 0.4785 (Tyurin, 2010) .

Observe that the quantity ρ σ 3 is data dependent.

To estimate this quantity, we performed a forward-backward pass for all training samples at the start of each epoch for our four networks considered.

The statistics ρ and σ were estimated by spatial (over tensors) and sample (over training samples) averages.

The maximum value of the ratio ρ σ 3 for all gradient tensors was found to be 2.8097.

The mini-batch size we used in all our experiments was 256.

Hence, we claim that the CLT approximation in Lemmas 2 & 3 is valid in our context up to a worst case Kolmogorov-Smirnov distance of KS < The minimality experiments in the main paper only consider a full 1-b perturbation to the full precision configuration matrix.

We further investigate the minimality of C o and its sensitivity to precision perturbation per tenor type.

The results of this investigation are presented in FIG3 .

First, we consider random fractional precision perturbations, meaning perturbations to the precision configuration matrix where only a random fraction p of the 5L precision assignments is incremented or decremented.

A fractional precision perturbation of 1 (-1) corresponds to C +1 (C −1 ).

A fractional precision perturbation of 0.5 (-0.5) means that a randomly chosen half of the precision assignments is incremented (decremented).

FIG3 shows the relative test error deviation compared to the test error associated with C o for various fractional precision perturbations.

The error deviation is taken in a relative fashion to account for the variability of the different networks' accuracies.

For instance, an absolute 1% difference in accuracy on a network trained on SVHN is significantly more severe than one on a network trained on CIFAR-100.

It is observed that for negative precision perturbations the variation in test error is more important than for the case of positive perturbations.

This is further encouraging evidence that C o is nearly minimal, in that a negative perturbation causes significant accuracy degradation while a positive one offers diminishing returns.

It is also interesting to study which of the 5L tensor types is most sensitive to precision reduction.

To do so, we perform a similar experiment whereby we selectively decrement the precision of all tensors belonging to the same type (weights, activations, weight gradients, activation gradients, weight accumulators).

The results of this experiment are found in FIG3 .

It is found that the most sensitive tensor types are weights and activation gradients while the least sensitive ones are activations and weight gradients.

This is an interesting finding raising further evidence that there exists some form of duality between the forward propagation of activations and back propagation of derivatives as far as numerical precision is concerned.

We illustrate a step by step description of the application of our precision assignment methodology to the four networks we report results on.

Feedforward Precisions: The first step in our methodology consists of setting the feedforward precisions B W l and B A l .

As per Claim 1, this requires using (1).

To do so, it is first needed to compute the quantization noise gains using (6).

Using the converged weights from the baseline run we obtain: And therefore, E (min) = 94.7 and the feedforward precisions should be set according to (1) as follows: DISPLAYFORM0 The value of B (min) is swept and p m i evaluated on the validation set.

It is found that the smallest value of B (min) resulting in p m < 1% is equal to 4 bits.

Hence the feedforward precisions are set as follows and as illustrated in gradients.

As per Claim 1, an important statistic is the spatial variance of the gradient tensors.

We estimate these variances via moving window averages, where at each iteration, the running variance estimateσ 2 is updated using the instantaneous varianceσ 2 as follows: DISPLAYFORM1 where θ is the running average factor, chosen to be 0.1.

The running variance estimate of each gradient tensor is dumped every epoch.

Using the maximum recorded estimate and (2) we compute the PDRs of the gradient tensors (as a reminder, the PDR is forced to be a power of 2):Layer

(min)

(min)

(min)The value of B (min) is again swept, and it is found that the p m < 1% for B (min) = 3.

The feedforward precisions are therefore set as follows and as illustrated in FIG4 Layer Index l 1 2 3 4 5 6 7 8 9 B W l 9 8 9 9 10 9 7 5 5 B A l 8 4 5 4 6 6 7 4 3 Note that for weights, layer depths 21 and 22 correspond to the strided convolutions in the shortcut connections of residual blocks 4 and 7, respectively.

The value of B (min) is again swept, and it is found that the p m < 1% for B (min) = 3.

The feedforward precisions are therefore set as follows and as illustrated in FIG4

(min)The value of B (min) is again swept, and it is found that the p m < 1% for B (min) = 3.

The feedforward precisions are therefore set as follows and as illustrated in

@highlight

We analyze and determine the precision requirements for training neural networks when all tensors, including back-propagated signals and weight accumulators, are quantized to fixed-point format.