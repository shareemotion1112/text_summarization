Bottleneck structures with identity (e.g., residual) connection are now emerging popular paradigms for designing deep convolutional neural networks (CNN), for processing large-scale features efficiently.

In this paper, we focus on the information-preserving nature of identity connection and utilize this to enable a convolutional layer to have a new functionality of channel-selectivity, i.e., re-distributing its computations to important channels.

In particular, we propose Selective Convolutional Unit (SCU), a widely-applicable architectural unit that improves parameter efficiency of various modern CNNs with bottlenecks.

During training, SCU gradually learns the channel-selectivity on-the-fly via the alternative usage of (a) pruning unimportant channels, and (b) rewiring the pruned parameters to important channels.

The rewired parameters emphasize the target channel in a way that selectively enlarges the convolutional kernels corresponding to it.

Our experimental results demonstrate that the SCU-based models without any postprocessing generally achieve both model compression and accuracy improvement compared to the baselines, consistently for all tested architectures.

Nowadays, convolutional neural networks (CNNs) have become one of the most effective approaches in various fields of artificial intelligence.

With a growing interest of CNNs, there has been a lot of works on designing more advanced CNN architectures BID43 BID21 .

In particular, the simple idea of adding identity connection in ResNet BID11 has enabled breakthroughs in this direction, as it allows to train substantially deeper/wider networks than before by alleviating existed optimization difficulties in previous CNNs.

Recent CNNs can scale over a thousand of layers BID12 or channels BID18 without much overfitting, and most of these "giant" models consider identity connections in various ways BID49 BID18 .

However, as CNN models grow rapidly, deploying them in the real-world becomes increasingly difficult due to computing resource constraints.

This has motivated the recent literature such as network pruning BID9 BID28 BID35 , weight quantization BID36 BID3 , adaptive networks BID47 BID5 BID0 BID19 , and resource-efficient architectures BID17 BID40 BID32 .For designing a resource-efficient CNN architecture, it is important to process succinct representations of large-scale channels.

To this end, the identity connections are useful since they allow to reduce the representation dimension to a large extent while "preserving" information from the previous layer.

Such bottleneck architectures are now widely used in modern CNNs such as ResNet BID11 and DenseNet BID18 for parameter efficiency, and many state-of-the-art mobile-targeted architectures such as SqueezeNet BID20 , ShuffleNet BID53 BID32 , MoblileNet BID16 BID40 , and CondenseNet BID17 commonly address the importance of designing efficient bottlenecks.

Contribution.

In this paper, we propose Selective Convolutional Unit (SCU), a widely-applicable architectural unit for efficient utilization of parameters in particular as a bottleneck upon identity connection.

At a high-level, SCU performs a convolutional operation to transform a given input.

The main goal of SCU, however, is rather to re-distribute their computations only to selected channels (a) (b)Figure 1: (a) An illustration of channel de-allocation and re-allocation procedures.

The higher the saturation of the channel color, the higher the ECDS value.

(b) The overall structure of SCU.of importance, instead of processing the entire input naively.

To this end, SCU has two special operations: (a) de-allocate unnecessary input channels (dealloc), and (b) re-allocate the obstructed channels to other channels of importance (realloc) (see Figure 1a) .

They are performed without damaging the network output (i.e., function-preserving operations), and therefore one can call them safely at any time during training.

Consequently, training SCU is a process that increases the efficiency of CNN by iteratively pruning or rewiring its parameters on-the-fly along with learning them.

In some sense, it is similar to how hippocampus in human brain learn, where new neurons are generated daily, and rewired into the existing network while maintaining them via neuronal apoptosis or pruning BID38 BID49 .We combine several new ideas to tackle technical challenges for such on-demand, efficient trainable SCU.

First, we propose expected channel damage score (ECDS), a novel metric of channel importance that is used as the criterion to select channels for dealloc or realloc.

Compared to other popular magnitude-based metrics BID28 BID35 , ECDS allows capturing not only low-magnitude channels but also channels of low-contribution under the input distribution.

Second, we impose channel-wise spatial shifting bias when a channel is reallocated, providing much diversity in the input distribution.

It also has an effect of enlarging the convolutional kernel of SCU.

Finally, we place a channel-wise scaling layer inside SCU with sparsity-inducing regularization, which also promotes dealloc (and consequently realloc as well), without further overhead in inference and training.

We evaluate the effectiveness of SCU by applying it to several modern CNN models including ResNet BID11 , DenseNet BID18 , and ResNeXt BID49 , on various classification datasets.

Our experimental results consistently show that SCU improves the efficiency of bottlenecks both in model size and classification accuracy.

For example, SCU reduces the error rates of DenseNet-40 model (without any post-processing) by using even less parameters: 6.57% ??? 5.95% and 29.97% ??? 28.64% on CIFAR-10/100 datasets, respectively.

We also apply SCU to a mobile-targeted CondenseNet BID17 model, and further improve its efficiency: it even outperforms NASNet-C BID54 , an architecture searched with 500 GPUs for 4 days, while our model is constructed with minimal efforts automatically via SCU.There have been significant interests in the literature on discovering which parameters to be pruned during training of neural networks, e.g., see the literature of network sparsity learning BID48 BID25 BID41 BID35 BID30 BID4 .

On the other hand, the progress is, arguably, slower for how to rewire the pruned parameters of a given model to maximize its utility.

proposed Dense-Sparse-Dense (DSD), a multi-step training flow applicable for a wide range of DNNs showing that re-training with re-initializing the pruned parameters can improve the performance of the original network.

Dynamic network surgery BID7 , on the other hand, proposed a methodology of splicing the pruned connections so that mis-pruned ones can be recovered, yielding a better compression performance.

In this paper, we propose a new way of rewiring for parameter efficiency, i.e., rewiring for channel-selectivity, and a new architectural framework that enables both pruning and rewiring in a single pass of training without any postprocessing or re-training (as like human brain learning).

Under our framework, one can easily set a targeted trade-off between model compression and accuracy improvement depending on her purpose, simply by adjusting the calling policy of dealloc and realloc.

We believe that our work sheds a new direction on the important problem of training neural networks efficiently.

In this section, we describe Selective Convolutional Unit (SCU), a generic architectural unit for bottleneck CNN architectures.

The overall structure of SCU is described in Section 2.1 and 2.2.

In Section 2.3, we introduce a metric deciding channel-selectivity in SCU.

We present in Section 2.4 how to handle a network including SCUs in training and inference.

Bottleneck structures in modern CNNs.

We first consider a residual function defined in ResNet BID11 which has an identity mapping: for a given input random variable X ??? R

(H and W are the height and width of each channel, respectively, and I is the number of channels or feature maps) and a non-linear function F, the output of a residual function is written by Y = X + F(X).

This function has been commonly used as a building block for designing recent deep CNN models, in a form that F is modeled by a shallow CNN.

However, depending on how F is designed, computing F(X) can be expensive when I is large.

For tackling the issue, bottleneck structure is a prominent approach, that is, to model F by F ??? R by placing a bottleneck R that firstly maps X into a lower dimension of I < I features.

This approach, in essence, requires the identity connection, for avoiding information loss from X to Y. Namely, the identity connection enables a layer to save redundant computation (or parameters) for just "keeping" information from the input.

Bottleneck structures can be used other than ResNet as well, as long as the identity connection exists.

Recent architectures including DenseNet BID18 , PyramidNet BID8 and DPN develop this idea with using a different aggregation function W instead of addition in ResNet, e.g., W(X, X ) = [X, X ] (channel-wise concatenation) for DenseNet.

Designing R-F -W is now a common way of handling large features.

Channel-selectivity for efficient bottlenecks.

Although placing a bottleneck R reduces much computation of the main function F , we point out the majority of modern CNNs currently use inefficient design of R itself, so that even the computation of R often dominates the remaining.

In ResNet and DenseNet models, for example, bottlenecks are designed using a pointwise convolution with a batch normalization layer (BN) BID21 and ReLU BID34 : DISPLAYFORM0 where Conv

I???I denotes a pointwise convolution that maps I features into I features, i.e., its parameters can be represented by a I ?? I matrix.

This means that the parameters of R grows linearly on I, and it can be much larger than F if I I .

For example, in case of DenseNet-BC-190 BID18 , 70% of the total parameters are devoted for modeling R, which is inefficient as the expressivity of a pointwise convolution is somewhat limited.

In this paper, we attempt to improve the efficiency of R in two ways: (a) reducing the parameters in Conv

I???I by channel pruning, and (b) improving its expressivity by using the pruned parameters again.

This motivates our goal to learn both channel-selectivity and parameters jointly.

Overall architecture of SCU.

SCU is designed to learn the channel-selectivity via dynamic pruning and rewiring of channels during training.

In this paper, we focus on putting SCU as a bottleneck R, and show that the channel-selectivity of SCU improves its parameter efficiency.

Our intuition is that (a) the information-preserving nature of identity connection brings optimization benefits if neurons in its structure are dynamically pruned during training, and (b) such pruning can be particularly effective on bottlenecks as their outputs are in a much lower dimension compared to the input.

Nevertheless, we believe that our ideas on SCU are not limited to the bottleneck structures, as the concept of channel-selectivity can be generalized to other structures.

At a high level, SCU follows the bottleneck structure from (1), but for two additional layers: Channel Distributor (CD) and Noise Controller (NC) whose details are presented in Section 2.2.

We model a non-linear function SCU : R I??H??W ??? R I ??H??W as follows (see Figure 1b) : DISPLAYFORM0 SCU has two special operations which control its input channels to process: (a) channel deallocation (dealloc), which obstructs unnecessary channels from being used in future computations, and (b) channel re-allocation (realloc), which allocates more parameters to important, non-obstructed channels by copying them into the obstructed areas.

We design those operations to be function preserving, i.e. they do not change the original function of the unit, so that can be called at anytime during training without damage.

Repeating dealloc and realloc alternatively during training translates the original input to what has only a few important channels, potentially duplicated multiple times.

Namely, the parameters originally allocated to handle the entire input now operate on its important subset.

On the way of designing the operations of function preserving, we propose Expected Channel Damage Score (ECDS) that leads to an efficient, safe way to capture unimportant channels by measuring how much the output of SCU changes on average (w.r.t.

data distribution) after removing each channel.

The details of ECDS are in Section 2.3.

Channel Distributor (CD) is the principal mechanism of SCU and is placed at the beginning of the unit.

The role of CD is to "rebuild" the input, so that unnecessary channels can be discarded, and important channels are copied to be emphasized.

In essence, we implement this function by re-indexing and blocking the input channel-wise: CD(X) i := g i ?? X ??i with an index pointer ?? i ??? {1, 2, ?? ?? ?? , I}, a gate variable g i ??? {0, 1} for i = 1, 2, ?? ?? ?? , I. Here, we notice that CD(X) may contain a channel copied multiple times, i.e., multiple ?? i 's can have the same value.

Since SCU has different parameters for each channel, setting multiple ?? i 's has an effect of allocating more parameters to better process the channel pointed by ?? i .

We found that, however, it is hard to take advantage of the newly allocated parameters by simply copying a channel due to symmetry, i.e., the parameters for each channel usually degenerates.

Due to this, we consider spatial shifting biases DISPLAYFORM0 2 for each channel, as illustrated in FIG0 .

This trick can provide the copied channels much diversity in input distributions (and hence relaxing degeneracy), in a way that it is effective for the convolutional layer in SCU: it enlarges the convolutional kernel from 1 ?? 1 for the re-allocated channels only.

To summerize, CD(X) i takes (a) the channel which ?? i is pointing, in the spatially shifted form with bias b i , or (b) 0 if the gate g i is closed.

Formally, CD can be represented by DISPLAYFORM1 .

CD(X) has the same size to X, and defined as follows: DISPLAYFORM2 Here, shift(X, b h , b w ) denotes the "shifting" operation along spatial dimensions of X. For each pixel location (i, j) in X, we define shift(X, b h , b w ) i,j as: DISPLAYFORM3 using a bilinear interpolation kernel.

This formulation allows b h and b w to be continuous real values, thereby to be learned via gradient-based methods with other parameters jointly.

Noise Controller (NC) is a component for more effective training of SCU.

As SCU continuously performs channel pruning via dealloc during training, the efficiency of SCU depends on which regularization is used.

The key role of NC is to induce the training of SCU to get more channel-wise sparsity, so that more channels can be de-allocated safely.

Formally, NC is a channel-wise re-scaling layer: NC(X) := X ??, 1 where ?? = (?? i ) I i=1 are parameters to be learned.

For the channel-wise sparsity, we impose sparsity-inducing regularization specifically on ??.

Although any sparsity-inducing regularization can be used for ?? BID28 BID48 , in this paper we adopt the Bayesian pruning approach proposed by BID35 2 for two reasons: (a) it is easy to incorporate into training process, and (b) we found that noise incurred from Bayesian parameters helps to recover damage from channel pruning.

In general, a Bayesian scheme regards each parameter ?? as a random variable with prior p(??).

Updating the posterior p(??|D) from data D often leads the model to have much sparsity, if p(??) is set to induce sparsity, e.g., by 1 denotes the element-wise product.

2 For completeness, we present for the readers an overview of BID35 in Appendix B.log-uniform prior BID23 .

Meanwhile, p(??|D) is usually approximated with a simpler model q ?? (??), where ?? are parameters to be learned.

In case of NC, we regard each scaling parameter as a random variable, so that they become channel-wise multiplicative noises on input.

We follow BID35 for the design choices on q ?? (??) and p(??), by fully-factorized log-normal DISPLAYFORM4

Consider an input random variable X = (X i ??? R H??W ) DISPLAYFORM0 , and DISPLAYFORM1 I ??H??W , where S ???i denotes a SCU identical to S but g i = 0.

In other words, it is the expected amount of changes in outputs when S i is "damaged" or "pruned".

The primary goal of this criteria is to make dealloc to be function preserving.

We define ECDS(S) i by the (Euclidean) norm of the averaged values of E[SCU(X; S) ??? SCU(X; S ???i )] over the spatial dimensions: DISPLAYFORM2 Notice that the above definition requires a marginalization over random variable X. One can estimate it via Monte Carlo sampling using training data, but this is computationally too expensive compared to other popular magnitude-based metrics BID28 BID35 .

Instead, we utilize the BN layer inside SCU, to infer the current input distribution of each channel at any time of training.

This trick enables to approximate ECDS(S) i by a closed formula of S i , avoiding expensive computations of SCU(X; ??), as in what follows.

Consider a hidden neuron x following BN and ReLU, i.e., y = ReLU(BN(x)), and suppose one wants to estimate E[y] without sampling.

To this end, we exploit the fact that BN already "accumulates" its input statistics continuously during training.

Under assuming that BN(x) ??? N (??, ?? 2 ) where ?? and ?? are the scaling and shifting parameter in BN, respectively, it is elementary to check: DISPLAYFORM3 where ?? N and ?? N denote the p.d.f.

and the c.d.f. of the standard normal distribution, respectively.

The assumption is quite reasonable during training BN as each mini-batch is exactly normalized before applying the scaling and shifting inside BN.

The idea is directly extended to obtain a closed form formula of ECDS(S) i under some assumptions, as stated in the following proposition.

DISPLAYFORM4 , for all i.

The proof of the above proposition is given in Appendix D. In essence, there are three main terms in the formula: (a) a term that measures how much the input channel is active, (b) how much the NC amplifies the input, and (c) the total magnitude of weights in the convolutional layer.

Therefore, it allows a way to capture not only low-magnitude channels but also channels of low-contribution under the input distribution (see Section 3.2 for comparisons with other metrics).

Consider a CNN model p(Y|X, ??) employing SCU, where ?? denotes the collection of model parameters.

For easier explanation, we rewrite ?? by (V, W): V consists of (??, g) in CDs, and W is the remaining ones.

DISPLAYFORM0 , (V, W) is trained via alternating two phases: (a) training W via stochastic gradient descent (SGD), and (b) updating V via dealloc or realloc.

The overall training process is mainly driven by (a), and the usage of (b) is optional.

In (a), we use stochastic variational inference BID22 in order to incorporate stochasticity incurred from NC, so that SCU can learn its Bayesian parameters in NC jointly with the others via SGD.

On the other hand, in (b), dealloc and realloc are called on demand during training depending on the purpose.

For example, one may decide to call dealloc only throughout the training to obtain a highly compressed model, or one could use realloc as well to utilize more model parameters.

Once (b) is called, (a) is temporally paused and V are updated.

Training via stochastic variational inference.

We can safely ignore the effect of V during training of W, since they remain fixed.

Recall that, each noise ?? from a NC is assumed to follow q ?? (??) = LogN(??|??, ?? 2 ).

Then, ?? can be "re-parametrized" with a noise ?? from the standard normal distribution as follows: DISPLAYFORM1 2 ).

Stochastic variational inference BID22 ) allows a minibatch-based stochastic gradient method for ??, in such case that ?? can be re-parametrized with an non-parametric noise.

The final loss we minimize for a minibatch {( DISPLAYFORM2 becomes (see Appendix F for more details): DISPLAYFORM3 where DISPLAYFORM4 is a sampled vector from the fully-factorized standard normal distribution.

Channel de-allocation and re-allocation.

DISPLAYFORM5 The main role of dealloc and realloc is to update W CD in S that are not trained directly via SGD.

They are performed as follows: select slices to operate by thresholding ECDS(S), and update S from the selected channels.

More formally, when dealloc is called, S i 's where ECDS(S) i < T l for a fixed threshold T l are selected, and g i 's in W CD are set by 0.

If one chooses small T l , this operation does not hurt the original function.

On the other hand, realloc selects channels by collecting S i where ECDS(S) i > T h , for another threshold T h .

Each of the selected channels can be re-allocated only if there is a closed channel in S. If there does not exist a enough space, channels with higher ECDS have priority to be selected.

A single re-allocation of a channel S i to a closed channel S j consists of several steps: DISPLAYFORM6 ??? 0, (iv) re-initialize the shifting bias b j , and (v) set ?? j ??? ?? i .

This procedure is function-preserving, due to (iii).After training a SCU S, one can safely remove S i 's that are closed, to yield a compact unit.

Then, CDs are now operated by "selecting" channels rather than by obstructing, thereby the subsequent layers play with smaller dimensions.

Hence, at the end, SCU is trained to select only a subset of the input for performing the bottleneck operation.

For NC, on the other hand, one can still use it for inference, but efficient inference can be performed by replacing each noise ?? i by constant E[?? i ], following the well-known approximation used in many dropout-like techniques BID15 .

In our experiments, we apply SCU to several well-known CNN architectures that uses bottlenecks, and perform experiments on CIFAR-10/100 BID24 and ImageNet BID37 classification datasets.

The more details on our experimental setups, e.g., datasets, training details, and configurations of SCU, are given in Appendix G.

Improving existing CNNs with SCU.

We consider models using ResNet BID11 , DenseNet BID18 and ResNeXt BID49 architectures.

In general, every model we used in this paper forms a stack of multiple bottlenecks, where the definition of each bottleneck differs depending on its architecture except that it can be commonly expressed by R-F -W (the details are given in TAB6 in the appendix).

We compare the existing models with the corresponding new ones in which the bottlenecks are replaced by SCU.

For each SCU-based model, we consider three cases: (a) neither dealloc nor realloc is used during training, (b) only dealloc is used, and (c) both dealloc and realloc are used.

We measure the total number of parameters in bottlenecks, and error rates.

TAB1 compares the existing CNN models with the corresponding ones using SCU, on CIFAR-10/100.

The results consistently demonstrate that SCU improves the original models, showing their effectiveness in different ways.

When only dealloc is used, the model tends to be trained with minimizing their parameter to use.

Using realloc, SCU now can utilize the de-allocated parameters to improve their accuracy aggressively.

Note that SCU can improve the accuracy of the original model even neither dealloc nor realloc is used.

This gain is from the regularization effect of stochastic NC, acting a dropout-like layer.

We also emphasize that one can set a targeted trade-off between compression of SCU and accuracy improvement depending on her purpose, simply by adjusting the calling policy of dealloc and realloc.

For example, in case of DenseNet-100 model on CIFAR-10, one can easily trade-off between reductions in (compression, error) = (???51.4%, ???1.78%) and (???18.2%, ???8.24%).

In overall, SCU-based models achieve both model compression and accuracy improvement under all tested architectures.

TAB2 shows the results on ImageNet, which are consistent to those on CIFAR-10/100.

Notice that reducing parameters and error simultaneously is much more non-trivial in the case of ImageNet, e.g., reducing error 23.6% ??? 23.0% requires to add 51 more layers to ResNet-101 (i.e., ResNet-152), as reported in the official repository of ResNet BID13 .Designing efficient CNNs with SCU.

We also demonstrate that SCU can be used to design a totally new efficient architecture.

Recall that, in this paper, SCU focus on the bottlenecks inside the overall structure.

The other parts, F or W, are other orthogonal design choices.

To improve the efficiency of the parts, we adopt some components from CondenseNet BID17 , which is one of the state-of-the-art architectures in terms of computational efficiency, designed for mobile devices.

Although we do not adopt their main component, i.e., learned group convolution (LGC) as it also targets for the bottleneck as like SCU, we can still utilize other components of CondenseNet: increasing growth rate (IGR) (doubles the growth rate of DenseNet for every N blocks starting from 8) and the use of group convolution for F .

Namely, we construct a new model, coined CondenseNet-SCU by adopting IGR and GC upon a DenseNet-182 model with SCU.

We replace each 3??3 convolution for F by a group convolution of 4 groups.

We train this model using dealloc only to maximize the computational efficiency.

In TAB3 , we compare our model with state-of-the-art level CNNs, including ResNet-1001 BID12 , WRN-28-10 (Zagoruyko & Komodakis, 2016), NASNet-C BID54 , and the original CondenseNet-182.

As one can observe, our model shows better efficiency compared to the corresponding CondenseNet, suggesting the effectiveness of SCU overLGC.

Somewhat interestingly, ours even outperforms NASNet-C that is an architecture searched over thousands of candidates, in both model compression and accuracy improvement.

We finally remark that CondenseNet-SCU-182 model presented in TAB3 originally has 6.29M parameters in total before training, devoting 5.89M for bottlenecks, i.e., it is about 93.7% of the total number of parameters.

This is indeed an example in that reducing overhead from bottlenecks is important for better efficiency, which is addressed by SCU.

We also perform numerous ablation studies on the proposed SCU, investigating the effect of the key components: CD, NC, and ECDS.

For evaluation, we use the DenseNet-SCU-40 model (DenseNet-40 using SCU) trained for CIFAR-10.

We also follow the training details described in Appendix G.Spatial shifting and re-allocation.

We propose spatial shifting as a trick in realloc procedure to provide diversity in input distributions.

To evaluate its effect, we compare three DenseNet-SCU-40 models with different configurations of SCU: (D) only dealloc during training, (+R) realloc together but without spatial shifting, and (+R+S) further with the shifting.

FIG1 shows that +R does not improve the model performance much compared to D, despite +R+S outperforms both of them.

This suggests that copying a channel naively is not enough to fully utilize the rewired parameters, and spatial shifting is an effective way to overcome the issue.

DISPLAYFORM0 Sparsity-inducing effect of NC.

We place NC in SCU to encourage more sparse channels.

To verify such an effect, we consider DenseNet-SCU-40 model (say M1) and its variant removing NC from SCU (say M2).

We first train M1 and M2 calling neither dealloc nor realloc, and compare them how the ECDS of each channel is distributed.

FIG1 shows that M1 tends to have ECDS closer to zero, i.e., more channels will be de-allocated than M2.

Next, we train these models using dealloc, to confirm that NC indeed leads to more deallocation.

The left panel of FIG1 shows that the number of de-allocated channels of M1 is relatively larger than that of M2, which is the desired effect of NC.

Note that M1 also outperforms M2 on error rates, which is an additional advantage of NC from its stochastic regularization effect.

Nevertheless, remark that M2 in FIG1 already de-allocates many channels, which suggests that SBP (used in NC) is not crucial for efficient de-allocation.

Rather, the efficiency mainly comes from ECDS.

To prove this claim, we evaluate three variants of M1 which use different de-allocation policies than ECDS < T l : (a) SNR < 1 (thresholding the signal-to-noise ratio of NC in each channel by 1, proposed by the original SBP; M3), (b) SNR < 2.3 (M4) and (c) 2 < 0.25 (thresholding W Conv i 2 ; M5).

We train them using only dealloc, and compare the performances with the proposed model (M1).

The right panel of FIG1 shows the results of the three variants.

First, we found that the M3 could not de-allocate any channel in our setting (this is because we prune a network on-the-fly during training, while the original SBP only did it after training).

When we de-allocate competitive numbers of channels against M1 by tuning thresholds of others (M4 and M5), the error rates are much worse than that of M1.

These observations confirm that ECDS is a more effective de-allocation policy than other magnitude-based metrics.

We demonstrate that CNNs of large-scale features can be trained effectively via channel-selectivity, primarily focusing on bottleneck architectures.

The proposed ideas on channel-selectivity, however, would be applicable other than the bottlenecks, which we believe is an interesting future research direction.

We also expect that channel-selectivity has a potential to be used for other tasks as well, e.g., interpretability BID42 , robustness BID6 , and memorization BID51 .

Consider a probabilistic model p(Y|X, ??) between two random variables X and Y, and suppose one wants to infer ?? from a dataset D = {(x n , y n )} N n=1 consisting N i.i.d.

samples from the distribution of (X, Y).

In Bayesian inference, ?? is regarded as a random variable, under assuming some prior knowledge in terms of a prior distribution p(??).

The dataset D is then used to update the posterior belief on ??, namely p(??|D) = p(D|??)p(??)/p(D) from the Bayes rule.

In many cases, however, computing p(??|D) through Bayes rule is intractable since it requires to compute intractable integrals.

To address the issue, variational inference approximates p(??|D) by another parametric distribution q ?? (??), and tries to minimize the KL-divergence D KL (q ?? (??) p(??|D)) between q ?? (??) and p(??|D).

Instead of directly minimizing it, one typically maximizes the variational lower bound L(??), due to the following: DISPLAYFORM0 where DISPLAYFORM1 In case of complex models, however, expectations in (9) are still intractable.

BID22 proposed an unbiased minibatch-based Monte Carlo estimator for them, which can be used when q ?? (??) is representable by ?? = f (??, ??) with a non-parametric noise ?? ??? p(??).

For a minibatch DISPLAYFORM2 Now we can solve optimize L(??) by stochastic gradient ascent methods, if f is differentiable.

For a model having non-Bayesian parameters, say W, we can still apply the above approach by maximizing DISPLAYFORM3 where ?? and W can be jointly optimized under DISPLAYFORM4

Structured Bayesian pruning (SBP) BID35 ) is a good example to show how stochastic variational inference can be incorporated into deep neural networks.

The SBP framework assumes X to be an object of I features, that is, X = (X i ) DISPLAYFORM0 .

For example, X ??? R I??H??W can be a convolutional input consisting I channels, of the form X = ( DISPLAYFORM1 where W and H denote the width and the height of each channel, respectively.

It considers a dropout-like layer with a noise vector ?? = (?? i ) I i=1 ??? p noise (??), which outputs X ?? of the same size as X.4 Here, ?? is treated as a random vector, and the posterior p(??|D) is approximated by a fully-factorized truncated log-normal distribution q ?? (??): DISPLAYFORM2 DISPLAYFORM3 where 1 [a,b] denotes the indicator function for the inveral [a, b] .

Meanwhile, the prior p(??) is often chosen by a fully-factorized log-uniform distribution, e.g., Sparse Variational Dropout , and SBP use the truncated version: DISPLAYFORM4 The reason why they use truncations for q ?? (??) and p(??) is to prevent D KL (q ?? (??) p(??)) to be improper.

Previous works BID23 ignore this issue by implicitly regarding them as truncated distributions on a broad interval, but SBP treats this issue explicitly.

Note that, each ?? i ??? q ?? (?? i ) = LogN(?? i |?? i , ?? 2 i ) in the noise vector ?? can be re-parametrized with a non-parametric uniform noise ?? i ??? U(??|0, 1) by: DISPLAYFORM5 where DISPLAYFORM6 ??i , and ?? denotes the cumulative distribution function of the standard normal distribution.

Now one can optimize ?? = (??, ??) jointly with the weights W of a given neural network via stochastic variational inference described in Section A. Unlike , SBP regards W as a non-Bayesian parameter, and the final loss L SBP to optimize becomes DISPLAYFORM7 Here, the KL-divergence term is scaled by ?? to compensate the trade-off between sparsity and accuracy.

In practice, SBP starts from a pre-trained model, and re-trains it using the above loss.

Due to the sparsity-inducing behavior of log-uniform prior, ?? is forced to become more noisy troughout the re-training.

Neurons with ?? of signal-to-noise ratio (SNR) below 1 are selected, and removed after the re-training: DISPLAYFORM8 C BAYESIAN PRUNING AND IDENTITY CONNECTIONS SCU requires "training-time removal" of input channels for the channel de-allocation and reallocation to work.

But usually, this process should be done carefully since it can make the optimization much difficult and put the network into a bad local minima.

In particular, it occurs if we select channels to remove too aggressively.

It is known that this issue becomes more pronounced in Bayesian neural networks BID44 BID35 BID30 , such as SBP we use in this paper.

Recall the variational lower bound objective in (12), for Bayesian parameters ?? and non-Bayesian W. If the gradient of the first term DISPLAYFORM9 , that is, to follow the prior p(??).

Unfortunately, in practice, we usually observe this phenomena at the early stage of training, when W are randomly initialized.

In that case then, q ?? (??) will become p(??) too fast because of the "uncertain" W, thereby many channels will be pruned forever, in SBP for example.

This problem is usually dealt with in one of two ways: (a) using a pre-trained network as a starting point of W BID35 , and (b) a "warm-up" strategy, where the KL-divergence term is rescaled by ?? that increases linearly from 0 to 1 during training BID44 BID30 .

In this paper, however, neither methods are used, but instead we have found that the problem can be much eased with identity connections, as it can eliminate a possible cause of the optimization difficulty from removing channels: optimization difficulty from losing information as an input passes through a deep network.

The presence of identity connection implies that the information of an input will be fully preserved even in the case when all the parameters in a layer are pruned.

This may not be true in models without identity, for example, in VGGNet BID43 , one can see that the information of an input will be completely lost if any of the layers removes its entire channels.

This suggests us that identity connections can be advantageous not only for scaling up the network architectures, but also for reducing the size of them. ).

Then, we have: DISPLAYFORM10 Now, check that ECDS(S i ) becomes: DISPLAYFORM11 By the assumption that DISPLAYFORM12 ) for all h, w, we get: DISPLAYFORM13 DISPLAYFORM14 where ?? N and ?? N denote the probability distribution function and the cumulative distribution function of the standard normal distribution.

Therefore, the desired formula for ECDS(S i ) can be obtained by using the linearity of expectation: DISPLAYFORM15

To validate whether the assumption BN(CD(X; W CD ); W BN ) i,h,w ??? N (?? i , ?? 2 i ) holds in modern CNNs, we first observe that, once we ignore the effects from spatial shifting, 5 a necessary condition of the assumption is that (X i,:,: ) are identically distributed normal for a given channel X i .

This is because BN and CD do not change the "shape" of pixel-wise distributions of X i .

From this observation, we conduct a set of experiments focusing on a randomly chosen hidden layer in a DenseNet-40 model.

We analyze the empirical distribution of the hidden activation incoming to the layer calculated from CIFAR-10 test dataset.

Since the data consists of 10,000 samples, we get an hidden activation X test ??? R 10000??C??32??32 , 6 where C denotes the number of channels of the input.

These observations suggest us that the assumption of Proposition 1 can be reasonable except for the boundaries.

We also emphasize that these trends we found are also appeared even when the model is not trained at all FIG4 , that is, all the weights are randomly initialized, which implies that these properties are not "learned", but come from a structural property of CNN, e.g. equivariance on translation, or the central limit theorem.

This observation provides us another support why the ECDS formula stated in Proposition 1 is valid at any time during training.

From ?? = (V, W): V consists of (??, g) in CDs, we further rewrite W by (W NC , W C ): W NC the parameters in NCs, and W C is the remaining ones.

One can safely ignore the effect of V during training of (W NC , W C ), since they remain fixed.

Recall that each noise ?? from a NC is assumed to follow LogN(??|??, ?? 2 ).

They can be re-written with a noise ?? from the standard normal distribution, i.e., ?? = f ((??, ??), ??) = exp (?? + ?? ?? ??), where ?? ??? N (0, 1 2 ).

In such case that each noise ?? from NC can be "re-parametrized" with an non-parametric noise and the corresponding parameters ?? = (??, ??), we can then use stochastic variational inference BID22 for the optimization of (W NC , W C ) with a minibatch-based stochastic gradient method (see Appendix A for more details).

Then, the final loss we minimize for a minibatch {( DISPLAYFORM0 where DISPLAYFORM1 is a sampled vector from the fully-factorized standard normal distribution, and D KL (?? ??) denotes the KL-divergence.

Although not shown in (24), an extra regularization term R(W C ) can be added to the loss for the non-Bayesian parameters W C , e.g., weight decays.

In fact, in our case, i.e. q ?? (??) = LogN(??|??, ?? 2 ) and DISPLAYFORM2 As we explain in Appendix B, SBP bypasses this issue by using truncated distributions on a compact interval [a, b] for q ?? (??) and p(??).

We found that, however, this treatment also imposes extra computational overheads on several parts of training process, such as on sampling noises and computing D KL (q ?? (??) p(??)).

These overheads are non-negligible on large models like ResNet or DenseNet, which we are mainly focusing on.

Therefore, unlike SBP, here we do not take truncations on q ?? (??) and p(??) due to practical consideration, assuming an approximated form between the truncated distributions of q ?? (??) and p(??) on a large interval.

Then we can replace each D KL (q ?? (??) p(??)) in (24) by ??? log ?? for optimization.

In other words, each noise ?? in NC is regularized to a larger variance, i.e., the more "noisy".

We observed that this approximation does not harm much on the performance of SCU.

Nevertheless, one should be careful that q ?? (??) and p(??) should not be assumed as the un-truncated forms itself, but instead as approximated forms of truncated distributions on a large interval, not to make the problem ill-posed.

As used in SBP, if they are truncated, the KL-divergence becomes: DISPLAYFORM3

Datasets.

We perform our experiments extensively on CIFAR-10 and CIFAR-100 BID24 ) classification datasets.

CIFAR-10/100 contains 60,000 RGB images of size 32 ?? 32 pixels, 50,000 for training and 10,000 for test.

Each image in the two datasets is corresponded to one of 10 and 100 classes, respectively, and the number of data is set evenly for each class.

We use a common scheme for data-augmentation BID45 BID27 BID11 BID18 .

ImageNet classification dataset, on the other hand, consists of 1.2 million training images and 50,000 validation images, which are labeled with 1,000 classes.

We follow BID17 BID11 for preprocessing of data in training and inference time.

Training details.

All models in our experiments is trained by stochastic gradient descent (SGD) method, with Nesterov momentum of weight 0.9 without dampening.

We use a cosine shape learning rate schedule BID29 , i.e., decreasing the learning rate gradually from 0.1 to 0 throughout the training.

We set the weight decay 10 ???4 by for non-Bayesian parameters of each model.

We train each CIFAR model for 300 epochs with mini-batch size 64 following BID18 , except for the "DenseNet-BC-190+mixup" models as they are trained for 200 epochs following the original setting .

For ImageNet models, on the other hand, we train for 120 epochs with mini-batch size 256.

is employed in a model, we initialize W NC = (??, ??) by (0, e ???3 ), and DISPLAYFORM0 .

Initializations of W BN and W Conv may differ depending on models, and we follow the initialization scheme of the given model.

In our experiments, we follow a pre-defined calling policy when dealloc and realloc will be called throughout training.

If dealloc is used, it is called at the end of each epoch of training.

On the other hand, if realloc is used, it start to be called after 10% of the training is done, called for every 3 epochs, and stopped in 50% of training is done.

The thresholds for dealloc and realloc, i.e. T l and T h , is set by 0.0025 and 0.05, respectively, except for CondenseNet-SCU-182 TAB3 , in which T l is adjusted by 0.001 for an effective comparison with the baseline.

For all the CIFAR-10/100 models, we re-initialize b i by a random sample from [???1.5, 1.5] ?? [???1.5, 1.5] pixels uniformly whenever a channel slice S i is re-open via realloc process.

We set the weight decay on each b i to 10 ???5 separately from the other parameters.

For the ImageNet results TAB2 , however, we did not jointly train b for faster training.

Instead, each b i is set fixed unless it is re-initialized via realloc.

In this case, we sampled a point from [???2.5, 2.5] ?? [???2.5, 2.5] pixels uniformly for the re-initialization.

We found that this simple reallocation scheme can also improve the efficiency of SCU.

In general, every model we used here forms a stack of multiple bottlenecks, where the definition of each bottleneck differs depending on its architecture (see TAB6 ).

Each stack is separated into three (CIFAR-10/100) or four (ImageNet) stages by average pooling layers of kernel 2 ?? 2 to perform down-sampling.

Each of the stages consists N bottleneck blocks, and we report which N is used for all the tested models in TAB7 .

The whole stack of each model follows a global average pooling layer BID27 ) and a fully connected layer, and followed by single convolutional layer (See TAB8 ).

There exist some minor differences between the resulting models and the original papers BID11 BID18 BID49 .

In ResNet and ResNeXt models, we place an explicit 2 ?? 2 average pooling layer for down-sampling, instead of using convolutional layer of stride 2.

Also, we use a simple zero-padding scheme for doubling the number of channels between stages.

In case of DenseNet, on the other hand, our DenseNet models are different from DenseNet-BC proposed by BID18 , in a sense that we do not place a 1 ?? 1 convolutional layer between stages (which is referred as the "compression" layer in the original DenseNet).

Nevertheless, we observed that the models we used are trained as well as the originals.

<|TLDR|>

@highlight

We propose a new module that improves any ResNet-like architectures by enforcing "channel selective" behavior to convolutional layers