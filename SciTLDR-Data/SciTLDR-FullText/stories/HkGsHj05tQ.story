Deep Neural Networks (DNNs) thrive in recent years in which Batch Normalization (BN) plays an indispensable role.

However, it has been observed that BN is costly due to the reduction operations.

In this paper, we propose alleviating the BN’s cost by using only a small fraction of data for mean & variance estimation at each iteration.

The key challenge to reach this goal is how to achieve a satisfactory balance between normalization effectiveness and execution efficiency.

We identify that the effectiveness expects less data correlation while the efficiency expects regular execution pattern.

To this end, we propose two categories of approach: sampling or creating few uncorrelated data for statistics’ estimation with certain strategy constraints.

The former includes “Batch Sampling (BS)” that randomly selects few samples from each batch and “Feature Sampling (FS)” that randomly selects a small patch from each feature map of all samples, and the latter is “Virtual Dataset Normalization (VDN)” that generates few synthetic random samples.

Accordingly, multi-way strategies are designed to reduce the data correlation for accurate estimation and optimize the execution pattern for running acceleration in the meantime.

All the proposed methods are comprehensively evaluated on various DNN models, where an overall training speedup by up to 21.7% on modern GPUs can be practically achieved without the support of any specialized libraries, and the loss of model accuracy and convergence rate are negligible.

Furthermore, our methods demonstrate powerful performance when solving the well-known “micro-batch normalization” problem in the case of tiny batch size.

Recent years, Deep Neural Networks (DNNs) have achieved remarkable success in a wide spectrum of domains such as computer vision BID16 and language modeling BID4 .

The success of DNNs largely relies on the capability of presentation benefit from the deep structure BID5 .

However, training a deep network is so difficult to converge that batch normalization (BN) has been proposed to solve it BID14 .

BN leverages the statistics (mean & variance) of mini-batches to standardize the activations.

It allows the network to go deeper without significant gradient explosion or vanishing BID23 BID14 .

Moreover, previous work has demonstrated that BN enables the use of higher learning rate and less awareness on the initialization BID14 , as well as produces mutual information across samples BID21 or introduces estimation noises BID2 for better generalization.

Despite BN's effectiveness, it is observed that BN introduces considerable training overhead due to the costly reduction operations.

The use of BN can lower the overall training speed (mini second per image) by >45% , especially in deep models.

To alleviate this problem, several methods were reported.

Range Batch Normalization (RBN) BID1 accelerated the forward pass by estimating the variance according to the data range of activations within each batch.

A similar approach, L 1 -norm BN (L1BN) , simplified both the forward and backward passes by replacing the L 2 -norm variance with its L 1 -norm version and re-derived the gradients for backpropagation (BP) training.

Different from the above two methods, Self-normalization BID15 provided another solution which totally eliminates the need of BN operation with an elaborate activation function called "scaled exponential linear unit" (SELU).

SELU can automatically force the activation towards zero mean and unit variance for better convergence.

Nevertheless, all of these methods are not sufficiently effective.

The strengths of L1BN & RBN are very limited since GPU has sufficient resources to optimize the execution speed of complex arithmetic operations such as root for the vanilla calculation of L 2 -norm variance.

Since the derivation of SELU is based on the plain convolutional network, currently it cannot handle other modern structures with skip paths like ResNet and DenseNet.

In this paper, we propose mitigating BN's computational cost by just using few data to estimate the mean and variance at each iteration.

Whereas, the key challenge of this way lies at how to preserve the normalization effectiveness of the vanilla BN and improve the execution efficiency in the meantime, i.e. balance the effectiveness-efficiency trade-off.

We identify that the effectiveness preservation expects less data correlation and the efficiency improvement expects regular execution pattern.

This observation motivates us to propose two categories of approach to achieve the goal of effective and efficient BN: sampling or creating few uncorrelated data for statistics' estimation with certain strategy constraints.

Sampling data includes "Batch Sampling (BS)" that randomly selects few samples from each batch and "Feature Sampling (FS)" that randomly selects a small patch from each feature map (FM) of all samples; creating data means "Virtual Dataset Normalization (VDN)" that generates few synthetic random samples, inspired by BID22 .

Consequently, multi-way strategies including intra-layer regularity, inter-layer randomness, and static execution graph during each epoch, are designed to reduce the data correlation for accurate estimation and optimize the execution pattern for running acceleration in the meantime.

All the proposed approaches with single-use or joint-use are comprehensively evaluated on various DNN models, where the loss of model accuracy and convergence rate is negligible.

We practically achieve an overall training speedup by up to 21.7% on modern GPUs.

Note that any support of specialized libraries is not needed in our work, which is not like the network pruning BID32 or quantization BID12 requiring extra library for sparse or low-precision computation, respectively.

Most previous acceleration works targeted inference which remained the training inefficient BID26 BID20 BID19 BID31 BID9 , and the rest works for training acceleration were orthogonal to our approach BID7 BID29 .

Additionally, our methods further shows powerful performance when solving the well-known "micro-batch normalization" problem in the case of tiny batch sizes.

In summary, the major contributions of this work are summarized as follows.• We propose a new way to alleviate BN's computational cost by using few data to estimate the mean and variance, in which we identify that the key challenge is to balance the normalization effectiveness via less data correlation and execution efficiency via regular execution pattern.• We propose two categories of approach to achieve the above goal: sampling (BS/FS) or creating (VDN) few uncorrelated data for statistics' estimation, in which multi-way strategies are designed to reduce the data correlation for accurate estimation and optimize the execution pattern for running acceleration in the meantime.

The approaches can be used alone or jointly.• Various benchmarks are evaluated, on which up to 21.7% practical acceleration is achieved for overall training on modern GPUs with negligible accuracy loss and without specialized library support.• Our methods are also extended to the micro-BN problem and achieve advanced performance 1 .In order to make this paper easier for understanding, we present the organization of the whole paper in FIG0 The activations in one layer for normalization can be described by a d-dimensional activation feature DISPLAYFORM0 , where for each feature we have DISPLAYFORM1 Note that in convolutional (Conv) layer, d is the number of FMs and m equals to the number of points in each FM across all the samples in one batch; while in fully-connected (FC) layer, d and m are the neuron number and batch size, respectively.

BN uses the statistics (mean E[ DISPLAYFORM2 of the intra-batch data for each feature to normalize activation by DISPLAYFORM3 where DISPLAYFORM4 are trainable parameters introduced to recover the representation capability, is a small constant to avoid numerical error, and DISPLAYFORM5 The detailed operations of a BN layer in the backward pass can be found in Appendix C. DISPLAYFORM6 Iter.

per second.

TAB3 ; (b) usual optimization of the reduction operation using adder tree; (c) the computational graph of BN in the forward pass (upper) and backward pass (lower); (d) the computation graph of BN using few data for statistics' estimation in forward pass (upper) and backward pass (lower).

x is neuronal activations, µ and σ denote the mean and standard deviation of x within one batch, respectively, and is the summation operation.

From FIG0 , we can see that adding BN will significantly slow down the training speed (iterations per second) by 32%-43% on ImageNet.

The reason why BN is costly is that it contains several "reduction operations", i.e. m j=1 .

We offer more thorough data analysis in Appendix E. If the reduction operations are not optimized, it's computational complexity should be O(m).

With the optimized parallel algorithm proposed in BID3 , the reduction operation is transformed to cascaded adders of depth of log(m) as shown in FIG0 .

However, the computational cost is still high since we usually have m larger than one million.

As shown in FIG0 , the red " "s represent operations that contain summations, which cause the BN inefficiency.

Motivated by the above analysis, decreasing the effective value of m at each time for statistics estimation seems a promising way to reduce the BN cost for achieving acceleration.

To this end, we propose using few data to estimate the mean and variance at each iteration.

For example, if m changes to a much smaller value of s, equation FORMULA5 can be modified as DISPLAYFORM0 Under review as a conference paper at ICLR 2019 where x (k) s denotes the small fraction of data, s is the actual number of data points, and we usually have s m.

Here we denote s/m as Sampling Ratio (it includes both the cases of sampling and creating few data in Section 3.1 and 3.2, respectively).

Since the reduction operations in the backward pass can be parallelized whereas in the forward pass, the variance can not be calculated until mean is provided (which makes it nearly twice as slow as backward pass), we just use few data in the forward pass.

The computational graph of BN using few data is illustrated in FIG0 .

The key is how to estimate E[ DISPLAYFORM1 for each neuron or FM within one batch with much fewer data.

The influence on the backward pass is discussed in Appendix C.

Although using few data can reduce the BN's cost dramatically, it will meet an intractable challenge: how to simultaneously preserve normalization effectiveness and improve the execution efficiency.

On one side, using few data to estimate the statistics will increase the estimation error.

By regarding the layers with high estimation error as unnormalized ones, we did contrast test in FIG1 .

The mean & variance will be scaled up exponentially as network deepens, which causes the degradation of BN's effectiveness.

This degradation can be recovered from two aspects.• Intra-layer.

For the reason that the estimation error is not only determined by the amount of data but also the correlation between them, we can sample less correlated data within each layer to improve the estimation accuracy.• Inter-layer.

As depicted in by FIG1 , the intermittent BN configuration (i.e. discontinuously adding BN in different layers) can also prevent the statistics scaling up across layers.

This motivates us that as long as layers with high estimation error are discontinuous, the statistics shift can still be constrained to a smaller range.

Therefore, to reduce the correlation between estimation errors in different layers can also be beneficial to improve the accuracy of the entire model, which can be achieved by sampling less correlated data between layers.

On the other side, less data correlation indicates more randomness which usually causes irregular memory access irregular degrading the running efficiency.

In this paper, we recognize that the overhead of sampling can be well reduced by using regular and static execution patterns, which is demonstrated with ablation study at Fig In a nutshell, careful designs are needed to balance the normalization effectiveness via less data correlation and the execution efficiency via regular execution pattern.

Only in this way, it is possible to achieve practical acceleration with little accuracy loss, which is our major target in this work.

Based on the above analysis, we summarize the design considerations as follows.• Using few data for statistics' estimation can effectively reduce the computational cost of BN operations.

Whereas, the effectiveness-efficiency trade-off should be well balanced.• Less data correlation is promising to reduce the estimation error and then guarantees the normalization effectiveness.• More regular execution pattern is expected for efficient running on practical platforms.

To reach the aforementioned goal, we propose two categories of approach in this section: sampling or creating few uncorrelated data for statistics' estimation.

Furthermore, multi-way strategies to balance the data correlation and execution regularity.

Here "sampling" means to sample a small fraction of data from the activations at each layer for statistics' estimation.

As discussed in the previous section, mining uncorrelated data within and between layers is critical to the success of sampling-based BN.

However, the correlation property of activations in deep networks is complex and may vary across network structures and datasets.

Instead, we apply a hypothesis-testing approach in this work.

We first make two empirical assumptions:• Hypothesis 1.

Within each layer, data belonging to the different samples are more likely to be uncorrelated than those within the same sample.• Hypothesis 2.

Between layers, data belonging to different locations and different samples are less likely to be correlated.

Here "location" means coordinate within FMs.

These two assumptions are based on the basic nature of real-world data and the networks, thus they are likely to hold in most situations.

They are further evaluated through experiments in Section 4.1.Based on above hypotheses, we propose two uncorrelated-sampling strategies: Batch Sampling (BS) and Feature Sampling (FS).

The detailed Algorithms can be found in Alg.

1 and 2, respectively, Appendix A.• BS FIG2 ) randomly selects few samples from each batch for statistics' estimation.

To reduce the inter-layer data correlation, it selects different samples across layers following Hypothesis 2.• FS FIG2 ) randomly selects a small patch from each FM of all samples for statistics' estimation.

Since the sampled data come from all the samples thus it has lower correlation within each layer following Hypothesis 1.

Furthermore, it samples different patch locations across layers to reduce the inter-layer data correlation following Hypothesis 2.A Naive Sampling (NS) is additionally proposed as a comparison baseline, as shown in FIG2 .

NS is similar to BS while the sampling index is fixed across layers, i.e. consistently samples first few samples within each batch.

Regular and Static Sampling.

In order to achieve practical acceleration on GPU, we expect more regular sampling pattern and more static sampling index.

Therefore, to balance the estimation effectiveness and execution efficiency, we carefully design the following sampling rules: (1) In BS, the selected samples are continuous and the sample index for different channels are shared, while they are independent across layers; (2) In FS, the patch shape is rectangular and the patch location is shared by different channels and samples within each layer but variable as layer changes.

Furthermore, all the random indexes are updated only once for each epoch, which guarantees a static computational graph during the entire epoch.

Instead of sampling uncorrelated data, another plausible solution is to directly create uncorrelated data for statistics' estimation.

We propose Virtual Dataset Normalization (VDN) 2 to implement it, as illustrated in FIG2 .

VDN is realized with the following three steps: (1) calculating the statistics of the whole training dataset offline; (2) generating s virtual samples 3 at each iteration to concatenate with the original real inputs as the final network inputs.

(3) using data from only virtual samples at each layer for statistics' estimation.

Due to the independent property of the synthesized data, they are more uncorrelated than real samples thus VDN can produce much more accurate estimation.

The detailed implementation algorithm can be found in Alg.

3, Appendix A.

The sampling approach and the creating approach can be used either in a single way or in a joint way.

Besides the single use, a joint use can be described as DISPLAYFORM0 where x s denotes the sampled real data while x v represents the created virtual data.

β is a controlling variable, which indicates how large the sampled data occupy the whole data for statistics' estimation within each batch: (1) when β = 0 or 1, the statistics come from single approach (VDN or any sampling strategy); (2) when β ∈ (0, 1), the final statistics are a joint value as shown in equation 4.

A comparison between different using ways is presented in TAB2 , where "d.s." denotes "different samples"; "d.l." stands for "different locations", and "g.i." indicates "generated independent data".

Compared to NS, BS reduces the inter-layer correlation via selecting different samples across layers; FS reduces both the intra-layer and inter-layer correlation via using data from all samples within each layer and selecting different locations across layers, respectively.

Though VDN has a similar inter-layer correlation with NS, it slims the intra-layer correlation with strong data independence.

A combination of BS/FS and VDN can inherit the strength of different approaches, thus achieve much lower accuracy loss.

Experimental Setup.

All of our proposed approaches are validated on image classification task using CIFAR-10, CIFAR-100 and ImageNet datasets from two perspectives: (1) effectiveness evaluation and (2) efficiency execution.

To demonstrate the scalability and generality of our approaches on deep networks, we select ResNet-56 on CIFAR-10 & CIFAR-100 and select ResNet-18 and DenseNet-121 on ImageNet 4 .

The model configuration can be found in TAB3 .

The means and variances for BN are locally calculated in each GPU without inter-GPU synchronization as usual.

We denote our approaches as the format of "Approach-Sampled size/Original size-sampling ratio(%)".

For instance, if we assume batch size is 128, "BS-4/128-3.1%" denotes only 4 samples are sampled in BS and the sampling ratio equals to 4 128 = 3.1%.

Similarly, "FS-1/32-3.1%" implies a 1 32 = 3.1% patch is sampled from each FM, and "VDN-1/128-0.8%" indicates only one virtual sample is added.

The traditional BN is denoted as "BN-128/128-100.0%".

Other experimental configurations can be found in Appendix B.

Convergence Analysis.

FIG3 shows the top-1 validation accuracy and confidential interval of ResNet-56 on CIFAR-10 and CIFAR-100.

On one side, all of our approaches can well approximate the accuracy of normal BN when the sampling ratio is larger than 2%, which evidence their effectiveness.

On the other side, all the proposed approaches perform better than the NS baseline.

In particular, FS performs best, which is robust to the sampling ratio with negligible accuracy loss (e.g. at sampling ratio=1.6%, the accuracy degradation is -0.087% on CIFAR-10 and +0.396% on CIFAR-100).

VDN outperforms BS and NS with a large margin in extremely small sampling ratio (e.g. 0.8%), whereas the increase of virtual batch size leads to little improvement on accuracy.

BS is constantly better than NS.

Furthermore, an interesting observation is that the BN sampling could even achieve better accuracy sometimes, such as NS-8/128(72.6±1.5%), BS-8/128(72.3±1.5%), and FS-1/64(71.2±0.96%) against the baseline (70.8±1%) on CIFAR-100.

FIG4 further shows the training curves of ResNet-56 on CIFAR-10 under different approaches.

It reveals that FS and VDN would not harm the convergence rate, while BS and NS begin to degrade the convergence when the sampling ratio is smaller than 1.6% and 3.1%, respectively.

TAB4 shows the top-1 validation error on ImageNet under different approaches.

With the same sampling ratio, all the proposed approaches significantly outperform NS, and FS surpasses VDN and BS.

Under the extreme sampling ratio of 0.78%, NS and BS don't converge.

Due to the limitation of FM size, the smallest sampling ratio of FS is 1.6%, which has only -0.5% accuracy loss.

VDN can still achieve relatively low accuracy loss (1.4%) even if the sampling ratio decreases to 0.78%.

This implies that VDN is effective for normalization.

Moreover, by combining FS-1/64 and VDN-2/128, we get the lowest accuracy loss (-0.2%).

This further indicates that VDN can be combined with other sampling strategies to achieve better results.

Since training DenseNet-121 is time-consuming, we just report the results with FS/BS-VDN joint use.

Although DenseNet-121 is more challenging than ResNet-18 due to the much deeper structure, the "FS-1/64 + VDN-2/64" can still achieve very low accuracy loss (-0.6%).

"BS-1/64 + VDN-2/64" has a little higher accuracy loss, whereas it still achieves better result than NS.

In fact, we observed gradient explosion if we just use VDN on very deep network (i.e. DenseNet-121), which can be conquered through jointly applying VDN and other proposed sampling approach (e.g. FS+VDN).

FIG5 illustrates the training curves for better visualization of the convergence.

Except for the BS with extremely small sampling ratio (0.8%) and NS, other approaches and configurations can achieve satisfactory convergence.

Here, we further evaluate the fully random sampling (FRS) strategy, which samples completely random points in both the batch and FM dimensions.

We can see that FRS is less stable compared with our proposed approaches (except the NS baseline) and achieves much lower accuracy.

One possible reason is that under low sampling ratio, the sampled data may occasionally fall into the worse points, which lead to inaccurate estimation of the statistics.

Correlation Analysis.

In this section, we bring more empirical analysis to the data correlation that affects the error of statistical estimation.

Here we denote the estimation errors at l th layer as E DISPLAYFORM0 s are the estimated mean & variance from the sampled data (including the created data in VDN) while µ (l) & σ (l) are the ground truth from the vanilla BN for the whole batch.

The analysis is conducted on ResNet-56 over CIFAR-10.

The estimation errors of all layers are recorded throughout the first training epoch.

FIG6 and FIG7 present the distribution of estimation errors for all layers and the inter-layer correlation between estimation errors, respectively.

In FIG6 , FS demonstrates the least estimation error within each layer which implies its better convergence.

The estimation error of VDN seems similar to BS and NS here, but we should note that it uses much lower sampling ratio of 0.8% compared to others of 3.1%.

FIG7 , it can be seen that BS presents obviously less inter-layer correlation than NS, which is consistent with previous experimental results that BS can converge better than NS even though they have similar estimation error as shown in FIG6 .

For FS and VDN, although it looks like they present averagely higher correlations, there exist negative corrections which effectively improve the model accuracy.

Moreover, FS produces better accuracy than NS and BS since its selected data come from all the samples with less correlation.

After the normalization effectiveness evaluation, we will evaluate the execution efficiency which is the primary motivation.

FIG8 shows the BN speedup during training and overall training improvement.

In general, BS can gain higher acceleration ratio because it doesn't incur the fine-grained sampling within FMs like in FS and it doesn't require the additional calculation and concatenation of the virtual samples like in VDN.

As for FS, it fails to achieve speedup on CIFAR-10 due to the small image size that makes the reduction of operations unable to cover the sampling overhead.

The proposed approaches can obtain up to 2x BN acceleration and 21.8% overall training acceleration.

TAB5 gives more results on ResNet-18 using single approach and DenseNet-121 using joint approach.

On ResNet-18, we perform much faster training compared with two recent methods for BN simplification BID1 .

On ResNet-18 we can achieve up to 16.5% overall training speedup under BS; on very deep networks with more BN layers, such as DenseNet-121, the speedup is more significant that reaches 23.8% under "BS+VDN" joint approach.

"FS+VDN" is a little bit slower than "BS+VDN" since the latter one has a more regular execution pattern as aforementioned.

Nonetheless, on a very deep model, we still recommend the "FS+VDN" version because it can preserve the accuracy better.

The relationship be-tween sampling ratio and overall training speedup is represented in FIG0 which illustrates that 1) BS & FS can still achieve considerable speedup with a moderate sampling ratio; 2) BS can achieve more significant acceleration than FS, for its more regular execution pattern.

It's worth noting that, our training speedup is practically obtained on modern GPUs without the support of specialized library that makes it easy-to-use.

FIG8 (c) reveals that the regular execution pattern can significantly help us achieve practical speedup.

128/128 5.23 +0.38 RBN BID1 128

BN has been applied in most state-of-art DNN models BID8 BID24 since it was proposed.

As aforementioned, BN standardizes the activation distribution to reduce the internal covariate shift.

Models with BN have been demonstrated to converge faster and generalize better BID14 BID21 .

Recently, a model called Decorrelated Batch Normalization (DBN) was introduced which not only standardizes but also whitens the activations with ZCA whitening BID11 .

Although DBN further improves the normalization performance, it introduces significant extra computational cost.

Simplifying BN has been proposed to reduce BN's computational complexity.

For example, L1BN and RBN BID1 replace the original L 2 -norm variance with an L 1 -norm version and the range of activation values, respectively.

From another perspective, Selfnormalization uses the customized activation function (SELU) to automatically shift activation's distribution BID15 .

However, as mentioned in Introduction, all of these methods fail to obtain a satisfactory balance between the effective normalization and computational cost, especially on large-scale modern models and datasets.

Our work attempts to address this issue.

Motivated by the importance but high cost of BN layer, we propose using few data to estimate the mean and variance for training acceleration.

The key challenge towards this goal is how to balance the normalization effectiveness with much less data for statistics' estimation and the execution efficiency with irregular memory access.

To this end, we propose two categories of approach: sampling (BS/FS) or creating (VDN) few uncorrelated data, which can be used alone or jointly.

Specifically, BS randomly selects few samples from each batch, FS randomly selects a small patch from each FM of all samples, and VDN generates few synthetic random samples.

Then, multi-way strategies including intra-layer regularity, inter-layer randomness, and static execution graph are designed to reduce the data correlation and optimize the execution pattern in the meantime.

Comprehensive experiments evidence that the proposed approaches can achieve up to 21.7% overall training acceleration with negligible accuracy loss.

In addition, VDN can also be applied to the micro-BN scenario with advanced performance.

This paper preliminary proves the effectiveness and efficiency of BN using few data for statistics' estimation.

We emphasize that the training speedup is practically achieved on modern GPUs, and we do not need any support of specialized libraries making it easy-to-use.

Developing specialized kernel optimization deserves further investigation for more aggressive execution benefits.

Notations.

We use the Conv layer for illustration, which occupies the major part of most modern networks BID8 BID10 .

The batched features can be viewed as a 4D tensor.

We use "E 0,1,2 " and "V ar 0,1,2 " to represent the operations that calculate the means and variances, respectively, where "0, 1, 2" denotes the dimensions for reduction.

Data: input batch at layer l: DISPLAYFORM0 for ep ∈ all epochs do for l ∈ all layers do if BS: begin l = randint(0, N − n s ); else NS: DISPLAYFORM1 Algorithm 2: FS Algorithm Data: input batch at layer l: DISPLAYFORM2 for ep ∈ all epochs do for l ∈ all layers do begin DISPLAYFORM3

All the experiments on CIFAR-10 & CIFAR-100 are conducted on a single Nvidia Titan Xp GPU.

We use a weight decay of 0.0002 for all weight layers and all models are trained by 130 epochs.

The initial learning rate is set to 0.1 and it is decreased by 10x at 50, 80, 110 epoch.

During training, we adopt the "random flip left & right" and all the input images are randomly cropped to 32 × 32.

Each model is trained from scratch for 5 times in order to reduce random variation.

For ImageNet, We use 2 Nvidia Tesla V100 GPUs on DGX station for ResNet-18 and 3 for DenseNet-121.

We use a weight decay of 0.0001 for all weight layers and all models are trained by 100 epochs.

The initial learning rate is set to 0.1/256×"Gradient batch size" and we decrease the learning rate by 10x at 30, 60, 80, 90 epoch.

During training, all the input images are augmented by random flipping and cropped to 224 × 224.

We evaluate the top-1 validation error on the validation set using the centered crop of each image.

To reduce random variation, we use the average of last 5 epochs to represent its final error rate.

Besides, Winograd BID17 ) is applied in all models to speedup training.

Compute Cycle and Memory Access.

Our proposed approaches can effectively speedup forward pass.

Under the condition that we use s m data to estimation the statistics for each feature, the total accumulation operations are significantly reduced from m − 1 to s − 1.

If using the adder tree optimization illustrated in Section 2.1, the tree's depth can be reduced from log(m) to log(s).

Thus, the theoretical compute speedup for the forward pass can reach log s (m) times.

For instance, if the FM size is 56 × 56 with batch size of 128 (m = 56 × 56 × 128), under sampling ratio of 1/32, the compute speedup will be 36.7%.

The total memory access is reduced by m/s times.

For example, when the sampling ratio is 1/32, only 3.1% data need to be visited.

This also contributes a considerable part in the overall speedup.

Speedup in the Backward Pass.

The BN operations in the forward pass have been shown in equation FORMULA3 - FORMULA7 .

Based on the derivative chain rule, we can get the corresponding operations in the backward pass as follows DISPLAYFORM0 Figure 11: Influence of decay rate for moving average.

The above equation reveals that an appropriately smaller α might scale down the estimation error, thus produces better validation accuracy.

To verify this prediction, the experiments are conducted on ResNet-56 over CIFAR-10 and using BS-1/128(0.78%) sampling.

As shown in FIG0 , it's obvious that there exists a best decay rate setting (here is 0.7) whereas the popular decay rate is 0.9.

The performance also decays when decay rate is smaller than 0.7, which is because a too small α may lose the capability to record the moving estimation, thus degrade the validation accuracy.

This is interesting because the decay rate is usually ignored by researchers, but the default value is probably not the best setting in our context.

Figure 12: BN network for profiling.

To further demonstrate that reduction operations are the major bottleneck of BN, we build a pure BN network for profiling as shown in FIG0 .

The network's input is a variable with shape of [128, 112, 112, 64] which is initialized following a standard normal distribution.

The network is trained for 100 iterations and the training time is recorded.

We overwrite the original BN's codes and remove the reduction operations in both forward pass and backward pass for contrast test.

We use three different GPUs: K80, Titan Xp, and Tesla V100 to improve the reliability.

The results are shown in TAB7 .

We can see that on all the three GPUs, reduction operations take up to >60% of the entire operation time in BN.

As a result, it's solid to argue that the reduction operations are the bottleneck of BN.

Micro-BN aims to alleviate the diminishing of BN's effectiveness when the amount of data in each GPU node is too small to provide a reliable estimation of activation statistics.

Previous work can be classified to two categories: (1) Sync-BN BID30 and (2) Local-BN BID0 BID28 BID13 .

The former addresses this problem by synchronizing the estimations from different GPUs at each layer, which induces significant inter-GPU data dependency and slows down training process.

The latter solves this problem by either avoiding the use of batch dimension in the batched activation tensor for statistics or using additional information beyond current layer to calibrate the statistics.

In terms of BN's efficiency preservation, we are facing a similar problem with micro-BN, thus our framework can be further extended to the micro-BN scenario.

In Sync-BN: (1) With FS, each GPU node executes the same patch sampling as normal FS; (2) With BS, we can randomly select the statistics from a fraction of GPUs rather than all nodes; (3) With VDN, the virtual samples can be fed into a single or few GPUs.

The first one can just simplify the computational cost within each GPU, while the last two further optimize the inter-GPU data dependency.

In Local-BN, since the available data for each GPU is already tiny, the BN sampling strategy will be invalid.

Fortunately, the VDN can still be effective by feeding virtue samples into each node.

Experiments.

The normalization in Sync-BN is based on the statistics from multiple nodes through synchronization, which is equivalent to that in FIG5 with large batch size for each node.

Therefore, to avoid repetition, here we just show the results on Local-BN with VDN optimization.

We let the overall batch size of 256 breaks down to 64 workers (each one has only 4 samples for local normalization).

We use "(gradient batch size, statistics batch size)" of (256, 4) to denote the configuration .

A baseline of (256, 32) with BN and one previous work Group Normalization (GN) BID28 are used for comparison.

As shown in FIG0 , although the reduction of batch size will degrade the model accuracy, our VDN can achieve slightly better result (top-1 validation error rate: 30.88%) than GN (top-1 validation error rate: 30.96%), an advanced technique for this scenario with tiny batch size.

This promises the correct training of very large model so that each single GPU node can only accommodate several samples.

.

FIG0 : Illustration of the paper's organization.

The Green and Purple markers (round circle with a star in the center) represent whether the effectiveness is preserved by reducing inter-layer or intralayer correlation (Green: inter-layer; Purple: intra-layer).

Moreover, the consideration of regular & static execution pattern is applied to all approaches.

@highlight

We propose accelerating Batch Normalization (BN) through sampling less correlated data for reduction operations  with regular execution pattern, which achieves up to 2x and 20% speedup for BN itself and the overall training, respectively.