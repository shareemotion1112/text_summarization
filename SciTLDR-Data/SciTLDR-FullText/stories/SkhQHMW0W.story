Large-scale distributed training requires significant communication bandwidth for gradient exchange that limits the scalability of multi-node training, and requires expensive high-bandwidth network infrastructure.

The situation gets even worse with distributed training on mobile devices (federated learning), which suffers from higher latency, lower throughput, and intermittent poor connections.

In this paper, we find 99.9% of the gradient exchange in distributed SGD is redundant, and propose Deep Gradient Compression (DGC) to greatly reduce the communication bandwidth.

To preserve accuracy during compression, DGC employs four methods: momentum correction, local gradient clipping, momentum factor masking, and warm-up training.

We have applied Deep Gradient Compression to image classification, speech recognition, and language modeling with multiple datasets including Cifar10, ImageNet, Penn Treebank, and Librispeech Corpus.

On these scenarios, Deep Gradient Compression achieves a gradient compression ratio from 270x to 600x without losing accuracy, cutting the gradient size of ResNet-50 from 97MB to 0.35MB, and for DeepSpeech from 488MB to 0.74MB.

Deep gradient compression enables large-scale distributed training on inexpensive commodity 1Gbps Ethernet and facilitates distributed training on mobile.

Large-scale distributed training improves the productivity of training deeper and larger models BID7 BID35 BID24 BID37 .

Synchronous stochastic gradient descent (SGD) is widely used for distributed training.

By increasing the number of training nodes and taking advantage of data parallelism, the total computation time of the forward-backward passes on the same size training data can be dramatically reduced.

However, gradient exchange is costly and dwarfs the savings of computation time

Researchers have proposed many approaches to overcome the communication bottleneck in distributed training.

For instance, asynchronous SGD accelerates the training by removing gradient synchronization and updating parameters immediately once a node has completed back-propagation BID9 BID31 .

Gradient quantization and sparsification to reduce communication data size are also extensively studied.

Gradient Quantization Quantizing the gradients to low-precision values can reduce the communication bandwidth.

BID32 proposed 1-bit SGD to reduce gradients transfer data size and achieved 10?? speedup in traditional speech applications.

BID2 proposed another approach called QSGD which balance the trade-off between accuracy and gradient precision.

Similar to QSGD, BID34 developed TernGrad which uses 3-level gradients.

Both of these works demonstrate the convergence of quantized training, although TernGrad only examined CNNs and QSGD only examined the training loss of RNNs.

There are also attempts to quantize the entire model, including gradients.

DoReFa-Net BID36 uses 1-bit weights with 2-bit gradients.

BID33 proposed threshold quantization to only send gradients larger than a predefined constant threshold.

However, the threshold is hard to choose in practice.

Therefore, BID11 chose a fixed proportion of positive and negative gradient updates separately, and BID1 proposed Gradient Dropping to sparsify the gradients by a single threshold based on the absolute value.

To keep the convergence speed, Gradient Dropping

4: DISPLAYFORM0 Sample data x from ?? 6: DISPLAYFORM1 end for 8: DISPLAYFORM2 Select threshold: DISPLAYFORM3 end for 14:All-reduce DISPLAYFORM4 wt+1 ??? SGD (wt, Gt) 16: end for BID18 .

Gradient Dropping saves 99% of gradient exchange while incurring 0.3% loss of BLEU score on a machine translation task.

Concurrently, proposed to automatically tunes the compression rate depending on local gradient activity, and gained compression ratio around 200?? for fully-connected layers and 40?? for convolutional layers with negligible degradation of top-1 accuracy on ImageNet dataset.

Compared to the previous work, DGC pushes the gradient compression ratio to up to 600?? for the whole model (same compression ratio for all layers).

DGC does not require extra layer normalization, and thus does not need to change the model structure.

Most importantly, Deep Gradient Compression results in no loss of accuracy.

We reduce the communication bandwidth by sending only the important gradients (sparse update).

We use the gradient magnitude as a simple heuristics for importance: only gradients larger than a threshold are transmitted.

To avoid losing information, we accumulate the rest of the gradients locally.

Eventually, these gradients become large enough to be transmitted.

Thus, we send the large gradients immediately but eventually send all of the gradients over time, as shown in Algorithm 1.

The encode() function packs the 32-bit nonzero gradient values and 16-bit run lengths of zeros.

The insight is that the local gradient accumulation is equivalent to increasing the batch size over time.

Let F (w) be the loss function which we want to optimize.

Synchronous Distributed SGD performs the following update with N training nodes in total: DISPLAYFORM0 where ?? is the training dataset, w are the weights of a network, f (x, w) is the loss computed from samples x ??? ??, ?? is the learning rate, N is the number of training nodes, and B k,t for 1 ??? k < N is a sequence of N minibatches sampled from ?? at iteration t, each of size b.

Consider the weight value w (i) of i-th position in flattened weights w. After T iterations, we have DISPLAYFORM1 Equation FORMULA7 shows that local gradient accumulation can be considered as increasing the batch size from N b to N bT (the second summation over ?? ), where T is the length of the sparse update interval between two iterations at which the gradient of w (i) is sent.

Learning rate scaling BID12 ) is a commonly used technique to deal with large minibatch.

It is automatically satisfied in Equation 2 where the T in the learning rate ??T and batch size N bT are canceled out.

Without care, the sparse update will greatly harm convergence when sparsity is extremely high .

For example, Algorithm 1 incurred more than 1.0% loss of accuracy on the Cifar10 dataset, as shown in FIG3 (a).

We find momentum correction and local gradient clipping can mitigate this problem.

Momentum Correction Momentum SGD is widely used in place of vanilla SGD.

However, Algorithm 1 doesn't directly apply to SGD with the momentum term, since it ignores the discounting factor between the sparse update intervals.

Distributed training with vanilla momentum SGD on N training nodes follows BID29 , DISPLAYFORM0 where m is the momentum, N is the number of training nodes, and DISPLAYFORM1 Consider the weight value w (i) of i-th position in flattened weights w. After T iterations, the change in weight value w (i) shows as follows, DISPLAYFORM2 If SGD with the momentum is directly applied to the sparse gradient scenario (line 15 in Algorithm 1), the update rule is no longer equivalent to Equation 3, which becomes: DISPLAYFORM3 where the first term is the local gradient accumulation on the training node k. Once the accumulation result v k,t is larger than a threshold, it will pass hard thresholding in the sparse () function, and be encoded and get sent over the network in the second term.

Similarly to the line 12 in Algorithm 1, the accumulation result v k,t gets cleared by the mask in the sparse () function.

The change in weight value w (i) after the sparse update interval T becomes, DISPLAYFORM4 The disappearance of the accumulated discounting factor DISPLAYFORM5 compared to Equation 4 leads to the loss of convergence performance.

It is illustrated in FIG1 (a), where Equation 4 drives the optimization from point A to point B, but with local gradient accumulation, Equation 4 goes to point C. When the gradient sparsity is high, the update interval T dramatically increases, and thus the significant side effect will harm the model performance.

To avoid this error, we need momentum correction on top of Equation 5 to make sure the sparse update is equivalent to the dense update as in Equation FORMULA8 If we regard the velocity u t in Equation 3 as "gradient", the second term of Equation 3 can be considered as the vanilla SGD for the "gradient" u t .

The local gradient accumulation is proved to be effective for the vanilla SGD in Section 3.1.

Therefore, we can locally accumulate the velocity u t instead of the real gradient k,t to migrate Equation 5 to approach Equation 3: DISPLAYFORM6 where the first two terms are the corrected local gradient accumulation, and the accumulation result v k,t is used for the subsequent sparsification and communication.

By this simple change in the local accumulation, we can deduce the accumulated discounting factor We refer to this migration as the momentum correction.

It is a tweak to the update equation, it doesn't incur any hyper parameter.

Beyond the vanilla momentum SGD, we also look into Nesterov momentum SGD in Appendix B, which is similar to momentum SGD.

DISPLAYFORM7 Local Gradient Clipping Gradient clipping is widely adopted to avoid the exploding gradient problem BID4 .

The method proposed by BID27 rescales the gradients whenever the sum of their L2-norms exceeds a threshold.

This step is conventionally executed after gradient aggregation from all nodes.

Because we accumulate gradients over iterations on each node independently, we perform the gradient clipping locally before adding the current gradient G t to previous accumulation (G t???1 in Algorithm 1).

As explained in Appendix C, we scale the threshold by N ???1/2 , the current node's fraction of the global threshold if all N nodes had identical gradient distributions.

In practice, we find that the local gradient clipping behaves very similarly to the vanilla gradient clipping in training, which suggests that our assumption might be valid in real-world data.

As we will see in Section 4, momentum correction and local gradient clipping help improve the word error rate from 14.1% to 12.9% on the AN4 corpus, while training curves follow the momentum SGD more closely.

Because we delay the update of small gradients, when these updates do occur, they are outdated or stale.

In our experiments, most of the parameters are updated every 600 to 1000 iterations when gradient sparsity is 99.9%, which is quite long compared to the number of iterations per epoch.

Staleness can slow down convergence and degrade model performance.

We mitigate staleness with momentum factor masking and warm-up training.

BID23 discussed the staleness caused by asynchrony and attributed it to a term described as implicit momentum.

Inspired by their work, we introduce momentum factor masking, to alleviate staleness.

Instead of searching for a new momentum coefficient as suggested in BID23 , we simply apply the same mask to both the accumulated gradients v k,t and the momentum factor u k,t in Equation 7:

This mask stops the momentum for delayed gradients, preventing the stale momentum from carrying the weights in the wrong direction.

Warm-up Training In the early stages of training, the network is changing rapidly, and the gradients are more diverse and aggressive.

Sparsifying gradients limits the range of variation of the model, and thus prolongs the period when the network changes dramatically.

Meanwhile, the remaining aggressive gradients from the early stage are accumulated before being chosen for the next update, and therefore they may outweigh the latest gradients and misguide the optimization direction.

The warm-up training method introduced in large minibatch training BID12 is helpful.

During the warm-up period, we use a less aggressive learning rate to slow down the changing speed of the neural network at the start of training, and also less aggressive gradient sparsity, to reduce the number of extreme gradients being delayed.

Instead of linearly ramping up the learning rate during the first several epochs, we exponentially increase the gradient sparsity from a relatively small value to the final value, in order to help the training adapt to the gradients of larger sparsity.

As shown in TAB1 , momentum correction and local gradient clipping improve the local gradient accumulation, while the momentum factor masking and warm-up training alleviate the staleness effect.

On top of gradient sparsification and local gradient accumulation, these four techniques make up the Deep Gradient Compression (pseudo code in Appendix D), and help push the gradient compression ratio higher while maintaining the accuracy.

We validate our approach on three types of machine learning tasks: image classification on Cifar10 and ImageNet, language modeling on Penn Treebank dataset, and speech recognition on AN4 and Librispeech corpus.

The only hyper-parameter introduced by Deep Gradient Compression is the warm-up training strategy.

In all experiments related to DGC, we rise the sparsity in the warm-up period as follows: 75%, 93.75%, 98.4375%, 99.6%, 99.9% (exponentially increase till 99.9%).

We evaluate the reduction in the network bandwidth by the gradient compression ratio as follows, DISPLAYFORM0 where G k is the gradients computed on the training node k.

Image Classification We studied ResNet-110 on Cifar10, AlexNet and ResNet-50 on ImageNet.

Cifar10 consists of 50,000 training images and 10,000 validation images in 10 classes BID17 ), while ImageNet contains over 1 million training images and 50,000 validation images in 1000 classes BID10 ).

We train the models with momentum SGD following the training schedule in BID13 .

The warm-up period for DGC is 4 epochs out of164 epochs for Cifar10 and 4 epochs out of 90 epochs for ImageNet Dataset.

Language Modeling The Penn Treebank corpus (PTB) dataset consists of 923,000 training, 73,000 validation and 82,000 test words BID20 .

The vocabulary we select is the same as the one in BID22 .

We adopt the 2-layer LSTM language model architecture with 1500 hidden units per layer BID28 , tying the weights of encoder and decoder as suggested in BID15 and using vanilla SGD with gradient clipping, while learning rate decays when no improvement has been made in validation loss.

The warm-up period is 1 epoch out of 40 epochs.

BID0 while Librispeech corpus contains 960 hours of reading speech BID26 .

We use DeepSpeech architecture without n-gram language model, which is a multi-layer RNN following a stack of convolution layers BID14 .

We train a 5-layer LSTM of 800 hidden units per layer for AN4, and a 7-layer GRU of 1200 hidden units per layer for LibriSpeech, with Nesterov momentum SGD and gradient clipping, while learning rate anneals every epoch.

The warm-up period for DGC is 1 epoch out of 80 epochs.

We first examine Deep Gradient Compression on image classification task.

red) is worse than the baseline due to gradient staleness.

With momentum correction (yellow), the learning curve converges slightly faster, and the accuracy is much closer to the baseline.

With momentum factor masking and warm-up training techniques (blue), gradient staleness is eliminated, and the learning curve closely follows the baseline.

TAB2 shows the detailed accuracy.

The accuracy of ResNet-110 is fully maintained while using Deep Gradient Compression.

When scaling to the large-scale dataset, FIG3 (c) and 3(d) show the learning curve of ResNet-50 when the gradient sparsity is 99.9%.

The accuracy fully matches the baseline.

An interesting observation is that the top-1 error of training with sparse gradients decreases faster than the baseline with the same training loss.

TAB3 shows the results of AlexNet and ResNet-50 training on ImageNet with 4 nodes.

We compare the gradient compression ratio with Terngrad BID34 on AlexNet (ResNet is not studied in BID34 better compression than Terngrad with no loss of accuracy.

For ResNet-50, the compression ratio is slightly lower (277?? vs. 597??) with a slight increase in accuracy.

For language modeling, Figure 4 shows the perplexity and training loss of the language model trained with 4 nodes when the gradient sparsity is 99.9%.

The training loss with Deep Gradient Compression closely match the baseline, so does the validation perplexity.

From TAB5 , Deep Gradient Compression compresses the gradient by 462 ?? with a slight reduction in perplexity.

For speech recognition, Figure 5 shows the word error rate (WER) and training loss curve of 5-layer LSTM on AN4 Dataset with 4 nodes when the gradient sparsity is 99.9%.

The learning curves show the same improvement acquired from techniques in Deep Gradient Compression as for the image network.

TAB5 shows word error rate (WER) performance on LibriSpeech test dataset, where test-clean contains clean speech and test-other noisy speech.

The model trained with Deep Gradient Compression gains better recognition ability on both clean and noisy speech, even when gradients size is compressed by 608??.

Implementing DGC requires gradient top-k selection.

Given the target sparsity ratio of 99.9%, we need to pick the top 0.1% largest over millions of weights.

Its complexity is O(n), where n is the number of the gradient elements BID8 .

We propose to use sampling to reduce top-k selection time.

We sample only 0.1% to 1% of the gradients and perform top-k selection on the samples to estimate the threshold for the entire population.

If the number of gradients exceeding the threshold is far more than expected, a precise threshold is calculated from the already-selected gradients.

Hierarchically calculating the threshold significantly reduces top-k selection time.

In practice, total extra computation time is negligible compared to network communication time which is usually from hundreds of milliseconds to several seconds depending on the network bandwidth.

We use the performance model proposed in BID34 to perform the scalability analysis, combining the lightweight profiling on single training node with the analytical communication modeling.

With the all-reduce communication model BID30 BID5 , the density of sparse data doubles at every aggregation step in the worst case.

However, even considering this effect, Deep Gradient Compression still significantly reduces the network communication time, as implied in FIG6 .

In practice, each training node performs the forward-backward pass on different batches sampled from the training dataset with the same network model.

The gradients from all nodes are summed up to optimize their models.

By this synchronization step, models on different nodes are always the same during the training.

The aggregation step can be achieved in two ways.

One method is using the parameter servers as the intermediary which store the parameters among several servers BID9 .

The nodes push the gradients to the servers while the servers are waiting for the gradients from all nodes.

Once all gradients are sent, the servers update the parameters, and then all nodes pull the latest parameters from the servers.

The other method is to perform the All-reduce operation on the gradients among all nodes and to update the parameters on each node independently BID12 , as shown in Algorithm 2 and FIG7 .

In this paper, we adopt the latter approach by default.

Sample data x from ?? 5: DISPLAYFORM0 end for

All-reduce G w t+1 ??? SGD (w t , G t ) 9: end for

The conventional update rule for Nesterov momentum SGD BID25 follows, DISPLAYFORM0 where m is the momentum, N is the number of training nodes, and k,t = 1 N b x???B k,t f (x, w t ).Before momentum correction, the sparse update follows, DISPLAYFORM1 sparse (v k,t+1 ) , w t+1 = w t ??? ??u t+1After momentum correction sharing the same methodology with Equation 7, it becomes, u k,t+1 = mu k,t + k,t , v k,t+1 = v k,t +(m ??

u k,t+1 + k,t ) , w t+1 = w t ????? C LOCAL GRADIENT CLIPPING When training the recurrent neural network with gradient clipping, we perform the gradient clipping locally before adding the current gradient G k t to previous accumulation G k t???1 in Algorithm 1.

Denote the origin threshold for the gradients L2-norm ||G|| 2 as thr G , and the threshold for the local gradients L2-norm ||G k || 2 as as thr G k .

@highlight

we find 99.9% of the gradient exchange in distributed SGD is redundant; we reduce the communication bandwidth by two orders of magnitude without losing accuracy. 

@highlight

This paper proposes additional improvement over gradient dropping to improve communication efficiency