Data-parallel neural network training is network-intensive, so gradient dropping was designed to exchange only large gradients.

However, gradient dropping has been shown to slow convergence.

We propose to improve convergence by having each node combine its locally computed gradient with the sparse global gradient exchanged over the network.

We empirically confirm with machine translation tasks that gradient dropping with local gradients approaches convergence 48% faster than non-compressed multi-node training and 28% faster compared to vanilla gradient dropping.

We also show that gradient dropping with a local gradient update does not reduce the model's final quality.

Training a neural network can be slow, especially with a large model or dataset BID12 BID18 .

Distributed training is becoming essential to speed up the process.

In data-parallel training, multiple workers optimize the same parameters based on different parts of the training data then exchange parameters.

Data-parallel training is network intensive because workers send and fetch gradients that have the same size as the model.

Several techniques have been proposed to reduce the traffic in dataparallelism training by using quantization to compress the gradient sent BID13 BID1 or selecting sparse matrices BID17 BID5 BID0 BID10 .Gradient dropping, and its extension Deep Gradient Compression BID10 , is a recent approach that compresses the network by sending a small fraction (about 1%) of the largest gradients (by absolute value).

This technique is based on the observation that the gradient values are skewed, as most are close to zero.

An issue with gradient compression is that gradients are compressed so much that it slows the model's convergence rate and can reduce the model's final quality BID0 .In vanilla gradient dropping, all nodes update with the same sparse gradient exchanged over the network, while other parameters are unchanged.

However, each node has computed a local gradient on its own data.

Can we exploit this dense local gradient alongside the sparse global gradient to improve convergence?

We propose and evaluate three ways to combine them.

L n t is a local gradient computed by node n at step t.3: DISPLAYFORM0 ApplyOptimizer(G t ) 10: end procedure 2.1 GRADIENT DROPPING Gradient dropping compresses communication by selecting the top 1% of gradients from each node by absolute value.

An optimizer, such as SGD or Adam (Kingma & Ba, 2015) , uses the summed sparse gradients from all nodes.

An error feedback mechanism stores unsent gradients and adds them to the next gradient update before compression BID13 .

Without this mechanism, the model will not converge.

Formally, gradient dropping is outlined in Algorithm 1.

For each time step t, each node n produces a local gradient L n t .

First, we apply error feedback by adding the unsent gradients from past step E t−1 to the local gradient L n t .

The combined gradient is then broken into sparse gradient S n t and residual E t .

We combine sparse gradients from every node by using all-reduce and use it to update the parameter.

Although the gradient is sparse, we apply a parameter update on the entire parameter.

This is done to let the optimizer update its momentum and apply momentum movement to the entire parameter.

Gradient dropping significantly reduces the communication cost in data-parallel training, making each parameter update faster but degrading convergence per update.

When network bandwidth is the bottleneck, overall time to convergence is reduced.

Deep gradient compression (DGC) BID10 introduces four modifications that reduces the quality damage of gradient dropping.

For momentum-based optimizers like Adam BID8 , they apply a correction for missing gradients and masking in the error feedback mechanism.

Additionally, DGC applies gradient clipping locally before compression instead of globally before applying an optimizer.

DGC also warms up the compression rate, sending more in early training.

Our work differs in that we use the local gradients so every parameter has an update.

Masking therefore does not directly port to our work, while gradient clipping and compression warm-up are equally applicable.

The issue with gradient dropping is that through the use of lossy compression to compress the gradient, damage is caused in the gradient making the model harder to learn.

DGC fixes this issue to some extent, but the model still relies on a sparse gradient.

We may as well also use the dense locally computed gradient if we can find a way to combine the two.

Formally, we currently only use the compressed global gradient G t , as in Algorithm 1 line 9, to update the model.

Instead, we incorporate the local gradient context to gain a better approximation of the compressed gradient.

Let L n t be gradient computed locally at time t on node n. Our goal is to compute a combined gradient C n t incorporated with local gradient context from node n at time t. As described in Algorithm 2, we propose three formulas to obtain C n t that will be used to update the parameter.

L n t is a local gradient computed by node n at step t.3: DISPLAYFORM0 switch mode do 10:case SUM 11: DISPLAYFORM1 case PARTIAL 13: DISPLAYFORM2 case ERROR 15: DISPLAYFORM3 ApplyOptimizer(C n t ) 18: end procedure SUM We use the local gradient from each node to predict the general direction of the global gradient.

An arguably naïve way to incorporate the local gradient is to add it to the sparse global gradient by DISPLAYFORM4 where we divide the sum by 2 to avoid double counting the computed gradient.

We can ignore this if we apply a scale-invariant optimizer such as Adam.

PARTIAL Since some of the local gradients L n t make their way into the sparse global gradient G t , it seems unfair to count them twice in the SUM method.

We can correct for this by subtracting the locally-generated sparse gradients S n t .

DISPLAYFORM5 where the term G t − S n t is equal to the sum of the sparse gradients from every node except the local node.

Therefore, we only use sparse gradients from the other nodes while using a non-sparsified local gradient.

ERROR Finally, we attempt to incorporate the residual error stored in E n t by simply adding them to the sparse gradient.

However, to avoid using the same gradient over and over, we have to clear the residual.

Therefore, in this approach, instead of accumulating the unsent gradients, we just apply them as a local context instead.

We update the parameter with DISPLAYFORM6 Clearing the error at every step is equivalent to removing the error feedback mechanism.

As the error E n t is now only contains the current step's unsent gradient which is equal to L n t − S n t , ERROR is equivalent to PARTIAL without an error-feedback mechanism.

Parameters on each node will diverge because each local gradient is different.

To resolve this, we periodically average the parameters across nodes.

Since we must communicate all parameters, the synchronization should be infrequent enough that it does not significantly effect the overall training speed in terms of words per second.

In this research, we are also considering averaging the parameters across all nodes for every 500 steps.

Using Marian as the toolkit, we train on nodes with four P100s each.

Each node is connected with 40Gb Infiniband from Mellanox, and our multi-node experiments use four nodes.

We test our experiment on the following tasks.

Ro-En Machine Translation: We build a Romanian-to-English neural machine translation system using all the parallel corpora in the constrained WMT 2016 task BID3 .

The dataset consists of 2.6M pairs of sentences to which we apply byte-pair encoding BID14 .

Our model is based on the winning system by BID15 and is a single layer attentional encoder-decoder bidirectional LSTM consisting of 119M parameters.

We apply layer normalization BID9 and exponential smoothing to train the model for up to 14 epochs or until no improvement after five validations.

We optimize the model with the Adam optimizer.

Training the Ro-En NMT system with this model is fast, so we primarily use the dataset and model for our development experiments.

En-De Machine Translation: We train another machine translation system on English-to-German for our large and high-resource model.

The corpus consists of 19.1M pairs of sentences after backtranslation.

The model is based on the winning system by BID16 with eight layers of LSTM consisting of 225M parameters.

The configuration is similar to the previous system as we use layer normalization and exponential smoothing, train the model for up to eight epochs and optimize with Adam.

Concerned with time to convergence of a single model, we report single model scores rather than ensemble scores.

The baseline systems were trained on a single node BID16 but our experiments focus on multi-node settings.

Thus, we apply several adjustments to the hyperparameters to accommodate the larger effective batch size of synchronous stochastic gradient descent.

These hyperparameters are used in multi-node baselines and experimental conditions.

Synchronous: We follow BID10 in applying synchronous stochastic gradient descent, so that nodes can easily aggregate internally amongst their GPUs and exchange one copy of the gradients externally.

This differs from BID0 where experiments were asynchronous but confined within a node.

Batch size: Prior work BID0 BID10 on gradient dropping used relatively small batch sizes even though larger batches are generally faster due to using more of the GPU.

In all our experiments, we use a workspace of 10GB and dynamically fit as many sentences as possible for each batch, which provides an average of 450 and 250 sentences per batch per GPU for Ro-En and En-De, respectively.

Learning rate: The Adam optimizer is scale-invariant, so the parameter moves at the same magnitude with both single and multi-node settings despite having approximately 4x larger gradients in the multi-node.

Therefore, we linearly scale the learning rate by 4x in all experiments to resolve this, as suggested by BID6 .

We use a learning rate of 0.002 in the Ro-En multi-node and 0.0005 in the Ro-En single-node.

Similarly, we use a learning rate of 0.0012 in the En-De multi-node and 0.0003 in the En-De single-node.

Learning rate warm-up also helps in training with large mini-batch scenario to overcome model instability during the early stages of training BID6 .

So, we add a learning rate warm-up for all multi-node experiments by linearly increasing the rate until it reaches the desired amount after several steps.

We apply a warm-up for the first 2000 steps in the Ro-En and 4000 steps in the En-De experiments.

To provide a fair comparison, we also apply the warm-up for the same number of examples in the multi-node and single-node experiments.

The remaining hyperparameters are equivalent in both single-node and multi-node settings.

Gradient dropping increases the raw training speed from 73k words/second to 116k words/second in the multi-node Ro-En experiment.

However, gradient dropping also damages convergence in the sense that it takes more epochs to reach peak performance.

DGC is proposed to minimize the convergence damage caused by gradient dropping.

While DGC typically performs better than gradient dropping, we argue that most of the improvement is due to the compression ratio warm-up.

To confirm this, we ran an experiment on the multi-node Ro-En with a drop ratio warm-up.

At the tth step, we discard R t of the gradient, defined below, with a warm-up period T .

In this case we set T = 1000, which is equal to 3 epochs in Ro-En experiment.

DISPLAYFORM0 The result shown in FIG1 suggests that the compression ratio warm-up can improve the convergence with gradient dropping.

On the other hand, there is no contrast in terms of convergence by other methods proposed in DGC.

Based on this result, we choose to use compression ratio warm-up for the remainder of the experiments.

We test our proposed techniques to incorporate the local gradient while performing a sparse gradient update.

We base the experiment in a multi-node setting with gradient dropping configured with a dropping ratio of 99% and a dropping rate warm-up for the first three epochs on the Ro-En dataset.

We also apply each of our local gradient update techniques.

Figure 2 shows convergence after incorporating the local gradient.

Using the PARTIAL or SUM update techniques improves the convergence curve in the early stages of the training as the convergence curves are closer to the baseline.

However, the curves are becoming unstable after several epochs.

Finally, we can see that their final qualities are lower, which we attribute to divergence of the models.

However, it is interesting that the models are still capable of learning even with model inconsistencies between workers.

We apply periodic model synchronization by doing model averaging across all workers at every 500 steps.

As shown in Figure 3 , the model is capable of learning and maintaining the same final quality as the baseline.

To understand the performance better, we capture several details provided in TAB0 .

Without synchronization, the model suffers a reduced quality in the development and test BLEU scores.

Using the ERROR local gradient update technique does not seem to benefit the model.

On the other hand, using PARTIAL or SUM with periodic synchronization significantly improves the convergence curve of the gradient dropping technique, and using PARTIAL appears to provide a more stable result compared to SUM.

PARTIAL also helps the model obtain better cross-entropy and reach convergence faster, thereby reducing the training time.

We train a model on 4 nodes with 4 GPUs each (henceforth 4x4) with gradient dropping, drop ratio warm-up for 1000 steps, local gradient update using the PARTIAL strategy, and periodic synchronization.

The baselines are a single-node configuration with 4 GPUs (denoted 1x4) and a 4x4 multi-node configuration, both with ordinary synchronous SGD.

Additionally, we try a 4x4 configuration with gradient dropping with drop ratio warmup for 1000 steps.

TAB1 summarizes our end-to-end experimental results.

In the ideal case, using 4x more workers should provide a 4x speed improvement.

Compressing the gradient reduces the network cost and significantly improves the raw words/second speed by about 3x over a single-node experiment.

Using local gradient update slightly decreases the average speed as it requires extra communication cost for the periodic synchronization.

Although slightly slower, local gradient update significantly improves the convergence speed as shown in FIG3 .

In both cases, vanilla gradient dropping massively increases the raw speed, there is no clear improvement on overall convergence time compared to the uncompressed multinode training.

We significantly reduce training time and the time to reach a near-convergence BLEU by using a local gradient update.

It also shows that local gradient update reduces the quality damage caused by gradient dropping.

Note that the improvement in words/second is greater compared to the training time in the En-De experiment because the model spends additional time for data and I/O operations (e.g., model saving and loading or data shuffling and reading).

We significantly reduce convergence damage caused by compressing the gradient through gradient dropping in data-parallelism training.

We utilize a locally-computed gradient to predict and reconstruct the dense gradient.

Our experiments show that we can improve the training time up to 45% faster compared to a non-compressed multi-node system and 3x faster compared to a single-node system.

Local gradient update is also empirically shown to negate the quality loss caused by gradient dropping.

@highlight

We improve gradient dropping (a technique of only exchanging large gradients on distributed training) by incorporating local gradients while doing a parameter update to reduce quality loss and further improve the training time.

@highlight

This paper proposes a 3 modes for combining local and global gradients to better use more computing nodes

@highlight

Looks at the problem of reducing the communication requirement for implementing the distributed optimiztion techniques, particularly SGD