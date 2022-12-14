State-of-the-art results on neural machine translation often use attentional sequence-to-sequence models with some form of convolution or recursion.

Vaswani et.

al. (2017) propose a new architecture that avoids recurrence and convolution completely.

Instead, it uses only self-attention and feed-forward layers.

While the proposed architecture achieves state-of-the-art results on several machine translation tasks, it requires a large number of parameters and training iterations to converge.

We propose Weighted Transformer, a Transformer with modified attention layers, that not only outperforms the baseline network in BLEU score but also converges 15-40% faster.

Specifically, we replace the multi-head attention by multiple self-attention branches that the model learns to combine during the training process.

Our model improves the state-of-the-art performance by 0.5 BLEU points on the WMT 2014 English-to-German translation task and by 0.4 on the English-to-French translation task.

Recurrent neural networks (RNNs), such as long short-term memory networks (LSTMs) BID12 , form an important building block for many tasks that require modeling of sequential data.

RNNs have been successfully employed for several such tasks including language modeling BID23 BID24 BID25 , speech recognition BID9 BID19 , and machine translation BID34 .

RNNs make output predictions at each time step by computing a hidden state vector h t based on the current input token and the previous states.

This sequential computation underlies their ability to map arbitrary input-output sequence pairs.

However, because of their auto-regressive property of requiring previous hidden states to be computed before the current time step, they cannot benefit from parallelization.

Variants of recurrent networks that use strided convolutions eschew the traditional time-step based computation BID15 BID20 BID4 BID7 BID6 BID16 .

However, in these models, the operations needed to learn dependencies between distant positions can be difficult to learn BID13 BID11 .

Attention mechanisms, often used in conjunction with recurrent models, have become an integral part of complex sequential tasks because they facilitate learning of such dependencies BID22 BID26 BID27 BID17 .In BID33 , the authors introduce the Transformer network, a novel architecture that avoids the recurrence equation and maps the input sequences into hidden states solely using attention.

Specifically, the authors use positional encodings in conjunction with a multi-head attention mechanism.

This allows for increased parallel computation and reduces time to convergence.

The authors report results for neural machine translation that show the Transformer networks achieves state-of-the-art performance on the WMT 2014 English-to-German and English-to-French tasks while being orders-of-magnitude faster than prior approaches.

Transformer networks still require a large number of parameters to achieve state-of-the-art performance.

In the case of the newstest2013 English-to-German translation task, the base model required 65M parameters, and the large model required 213M parameters.

We propose a variant of the Transformer network which we call Weighted Transformer that uses self-attention branches in lieu of the multi-head attention.

The branches replace the multiple heads in the attention mechanism of the original Transformer network, and the model learns to combine these branches during training.

This branched architecture enables the network to achieve comparable performance at a significantly lower computational cost.

Indeed, through this modification, we improve the state-of-the-art performance by 0.5 and 0.4 BLEU scores on the WMT 2014 English-to-German and English-to-French tasks, respectively.

Finally, we present evidence that suggests a regularizing effect of the proposed architecture.

Most architectures for neural machine translation (NMT) use an encoder and a decoder that rely on deep recurrent neural networks like the LSTM BID22 BID34 BID3 .

Several architectures have been proposed to reduce the computational load associated with recurrence-based computation BID7 BID6 BID15 BID16 .

Self-attention, which relies on dot-products between elements of the input sequence to compute a weighted sum BID21 BID26 BID17 , has also been a critical ingredient in modern NMT architectures.

The Transformer network BID33 avoids the recurrence completely and uses only self-attention.

We propose a modified Transformer network wherein the multi-head attention layer is replaced by a branched self-attention layer.

The contributions of the various branches is learned as part of the training procedure.

The idea of multi-branch networks has been explored in several domains BID0 BID6 BID35 .

To the best of our knowledge, this is the first model using a branched structure in the Transformer network.

In , the authors use a large network, with billions of weights, in conjunction with a sparse expert model to achieve competitive performance.

BID0 analyze learned branching, through gates, in the context of computer vision while in BID6 , the author analyzes a two-branch model with randomly sampled weights in the context of image classification.

The original Transformer network uses an encoder-decoder architecture with each layer consisting of a novel attention mechanism, which the authors call multi-head attention, followed by a feedforward network.

We describe both these components below.

From the source tokens, learned embeddings of dimension d model are generated which are then modified by an additive positional encoding.

The positional encoding is necessary since the network does not otherwise possess any means of leveraging the order of the sequence since it contains no recurrence or convolution.

The authors use additive encoding which is defined as: DISPLAYFORM0 where pos is the position of a word in the sentence and i is the dimension of the vector.

The authors also experiment with learned embeddings BID7 BID6 but found no benefit in doing so.

The encoded word embeddings are then used as input to the encoder which consists of N layers each containing two sub-layers: (a) a multi-head attention mechanism, and (b) a feed-forward network.

A multi-head attention mechanism builds upon scaled dot-product attention, which operates on a query Q, key K and a value V : DISPLAYFORM1 where d k is the dimension of the key.

In the first layer, the inputs are concatenated such that each of (Q, K, V ) is equal to the word vector matrix.

This is identical to dot-product attention except for the scaling factor d k , which improves numerical stability.

Multi-head attention mechanisms obtain h different representations of (Q, K, V ), compute scaled dot-product attention for each representation, concatenate the results, and project the concatenation with a feed-forward layer.

This can be expressed in the same notation as Equation FORMULA1 : DISPLAYFORM2 where the W i and W O are parameter projection matrices that are learned.

Note that DISPLAYFORM3 where h denotes the number of heads in the multi-head attention.

BID33 proportionally reduce DISPLAYFORM4 that the computational load of the multi-head attention is the same as simple self-attention.

The second component of each layer of the Transformer network is a feed-forward network.

The authors propose using a two-layered network with a ReLU activation.

Given trainable weights W 1 , W 2 , b 1 , b 2 , the sub-layer is defined as: DISPLAYFORM5 The dimension of the inner layer is d f f which is set to 2048 in their experiments.

For the sake of brevity, we refer the reader to BID33 for additional details regarding the architecture.

For regularization and ease of training, the network uses layer normalization BID1 after each sub-layer and a residual connection around each full layer .

Analogously, each layer of the decoder contains the two sub-layers mentioned above as well as an additional multi-head attention sub-layer that receives as inputs (V, K) from the output of the corresponding encoding layer.

In the case of the decoder multi-head attention sub-layers, the scaled dot-product attention is masked to prevent future positions from being attended to, or in other words, to prevent illegal leftward-ward information flow.

One natural question regarding the Transformer network is why self-attention should be preferred to recurrent or convolutional models.

BID33 state three reasons for the preference: (a) computational complexity of each layer, (b) concurrency, and (c) path length between long-range dependencies.

Assuming a sequence length of n and vector dimension d, the complexity of each layer is O(n 2 d) for self-attention layers while it is O(nd 2 ) for recurrent layers.

Given that typically d > n, the complexity of self-attention layers is lower than that of recurrent layers.

Further, the number of sequential computations is O(1) for self-attention layers and O(n) for recurrent layers.

This helps improved utilization of parallel computing architectures.

Finally, the maximum path length between dependencies is O(1) for the self-attention layer while it is O(n) for the recurrent layer.

This difference is instrumental in impeding recurrent models' ability to learn long-range dependencies.

We now describe the proposed architecture, the Weighted Transformer, which is more efficient to train and makes better use of representational power.

In Equations FORMULA2 and (4) , we described the attention layer proposed in BID33 comprising the multi-head attention sub-layer and a FFN sub-layer.

For the Weighted Transformer, we propose a branched attention that modifies the entire attention layer in the Transformer network (including both the multi-head attention and the feed-forward network).The proposed attention layer can be mathematically described as: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 where M denotes the total number of branches, ?? i , ?? i ??? R + are learned parameters and W Oi ??? R dv??dmodel .

The FFN functions above are identical in form to Equation (4) them, they have commensurately reduced dimensionality to ensure that no additional parameters are added to the network.

Further, we require that ?? i = 1 and ?? i = 1 so that Equation FORMULA8 is a weighted sum of the individual branch attention values.

We now briefly contrast the modified architecture with the base Transformer model.

In the same notation as (5)-(7), the attention layer in the base model can be described as: DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 Instead of aggregating the contributions from the different heads through W O right away and using a feed-forward sub-layer, we retain head i for each of the M heads, learn to amplify or diminish their contribution, use a feed-forward sub-layer and then aggregate them, again in a learned fashion.

In the equations above, ?? can be interpreted as a learned concatenation weight and ?? as the learned addition weight.

Indeed, ?? scales the contribution of the various branches before ?? is used to sum them in a weighted fashion.

We ensure that the simplex constraint is respected during each training step by projection.

Finally, note that our modification does not add depth (i.e., through the FFN sublayers) to any of the attention head transformation since the feed-forward computation is merely split and not stacked.

One interpretation of our proposed architecture is that it replaces the multi-head attention by a multibranch attention.

Rather than concatenating the contributions of the different heads, they are instead treated as branches that a multi-branch network learns to combine.

While it is possible that ?? and ?? could be merged into one variable and trained, we found better training outcomes by separating them.

It also improves the interpretability of the models gives that (??, ??) can be thought of as probability masses on the various branches.

This mechanism adds O(M ) trainable weights.

This is an insignificant increase compared to the total number of weights.

Indeed, in our experiments, the proposed mechanism added 192 weights to a model containing 213M weights already.

Without these additional trainable weights, the proposed mechanism is identical to the multi-head attention mechanism in the Transformer.

The proposed attention mechanism is used in both the encoder and decoder layers and is masked in the decoder layers as in the Transformer network.

Similarly, the positional encoding, layer normalization, and residual connections in the encoder-decoder layers are retained.

We eliminate these details from FIG0 for clarity.

Instead of using (??, ??) learned weights, it is possible to also use a mixture-ofexperts normalization via a softmax layer .

However, we found this to perform worse than our proposal.

Unlike the Transformer, which weighs all heads equally, the proposed mechanism allows for ascribing importance to different heads.

This in turn prioritizes their gradients and eases the optimization process.

Further, as is known from multi-branch networks in computer vision BID6 , such mechanisms tend to cause the branches to learn decorrelated input-output mappings.

This reduces co-adaptation and improves generalization.

This observation also forms the basis for mixture-ofexperts models .

The weights ?? and ?? are initialized randomly, as with the rest of the Transformer weights.

In addition to the layer normalization and residual connections, we use label smoothing with ls = 0.1, attention dropout, and residual dropout with probability P drop = 0.1.

Attention dropout randomly drops out elements BID31 from the softmax in (1).As in BID33 , we used the Adam optimizer BID18 with (?? 1 , ?? 2 ) = (0.9, 0.98) and = 10 ???9 .

We also use the learning rate warm-up strategy for Adam wherein the learning rate lr takes on the form: DISPLAYFORM0 for the all parameters except (??, ??) and DISPLAYFORM1 for (??, ??).This corresponds to the warm-up strategy used for the original Transformer network except that we use a larger peak learning rate for (??, ??) to compensate for their bounds.

Further, we found that freezing the weights (??, ??) in the last 10K iterations aids convergence.

During this time, we continue training the rest of the network.

We hypothesize that this freezing process helps stabilize the rest of the network weights given the weighting scheme.

We note that the number of iterations required for convergence to the final score is substantially reduced for the Weighted Transformer.

We found that Weighted Transformer converges 15-40% faster as measured by the total number of iterations to achieve optimal performance.

We train the baseline model for 100K steps for the smaller variant and 300K for the larger.

We train the Weighted Transformer for the respective variants for 60K and 250K iterations.

We found that the objective did not significantly improve by running it for longer.

Further, we do not use any averaging strategies employed in BID33 and simply return the final model for testing purposes.

In order to reduce the computational load associated with padding, sentences were batched such that they were approximately of the same length.

All sentences were encoded using byte-pair encoding BID29 and shared a common vocabulary.

Weights for word embeddings were tied to corresponding entries in the final softmax layer BID14 BID28 .

We trained all our networks on NVIDIA K80 GPUs with a batch containing roughly 25,000 source and target tokens.

We benchmark our proposed architecture on the WMT 2014 English-to-German and English-toFrench tasks.

The WMT 2014 English-to-German data set contains 4.5M sentence pairs.

The English-to-French contains 36M sentence pairs.

Transformer (small) BID33 27.3 38.1 Weighted Transformer (small) 28.4 38.9Transformer (large) BID33 28.4 41.0 Weighted Transformer (large) 28.9 41.4ByteNet BID16 23.7 -Deep-Att+PosUnk BID37 -39.2 GNMT+RL BID34 24.6 39.9 ConvS2S BID8 25.2 40.5 MoE 26.0 40.6 Table 1 : Experimental results on the WMT 2014 English-to-German (EN-DE) and English-toFrench (EN-FR) translation tasks.

Our proposed model outperforms the state-of-the-art models including the Transformer BID33 .

The small model corresponds to configuration (A) in TAB1 while large corresponds to configuration (B).Results of our experiments are summarized in Table 1 .

The Weighted Transformer achieves a 1.1 BLEU score improvement over the state-of-the-art on the English-to-German task for the smaller network and 0.5 BLEU improvement for the larger network.

In the case of the larger English-toFrench task, we note a 0.8 BLEU improvement for the smaller model and a 0.4 improvement for the larger model.

Also, note that the performance of the smaller model for Weighted Transformer is close to that of the larger baseline model, especially for the English-to-German task.

This suggests that the Weighted Transformer better utilizes available model capacity since it needs only 30% of the parameters as the baseline transformer for matching its performance.

Our relative improvements do not hinge on using the BLEU scores for comparison; experiments with the GLEU score proposed in BID34 also yielded similar improvements.

Finally, we comment on the regularizing effect of the Weighted Transformer.

Given the improved results, a natural question is whether the results stem from improved regularization of the model.

To investigate this, we report the testing loss of the Weighted Transformer and the baseline Transformer against the training loss in FIG1 .

Models which have a regularizing effect tend to have lower testing losses for the same training loss.

We see this effect in our experiments suggesting that the proposed architecture may have better regularizing properties.

This is not unexpected given similar outcomes for other branching-based strategies such as Shake-Shake Gastaldi FORMULA1 BID33 architecture and our proposed Weighted Transformer.

Reported BLEU scores are evaluated on the English-to-German translation development set, newstest2013.

Weighted Transformer 24.8 Train ??, ?? fixed to 1 24.5 Train ??, ?? fixed to 1 23.9 ??, ?? both fixed to 1 23.6 Without the simplex constraints 24.5 Table 3 : Model ablations of Weighted Transformer on the newstest2013 English-to-German task for configuration (C).

This shows that the learning both (??, ??) and retaining the simplex constraints are critical for its performance.

In TAB1 , we report sensitivity results on the newstest2013 English-to-German task.

Specifically, we vary the number of layers in the encoder/decoder and compare the performance of the Weighted Transformer and the Transformer baseline.

Using the same notation as used in the original Transformer network, we label our configurations as (A), (B) and (C) with (C) being the smallest.

The results clearly demonstrate the benefit of the branched attention; for every experiment, the Weighted Transformer outperforms the baseline transformer, in some cases by up to 1.3 BLEU points.

As in the case of the baseline Transformer, increasing the number of layers does not necessarily improve performance; a modest improvement is seen when the number of layers N is increased from 2 to 4 and 4 to 6 but the performance degrades when N is increased to 8.

Increasing the number of heads from 8 to 16 in configuration (A) yielded an even better BLEU score.

However, preliminary experiments with h = 16 and h = 32, like in the case with N , degrade the performance of the model.

In Figure 3 , we present the behavior of the weights (??, ??) for the second encoder layer of the configuration (C) for the English-to-German newstest2013 task.

The figure shows that, in terms of relative weights, the network does prioritize some branches more than others; circumstantially by as much as 2??.

Further, the relative ordering of the branches changes over time suggesting that the network is not purely exploitative.

A purely exploitative network, which would learn to exploit a subset of the branches at the expense of the rest, would not be preferred since it would effectively reduce the number of available parameters and limit the representational power.

Similar results are seen for other layers, including the decoder layers; we omit them for brevity.

Finally, we present an ablation study to highlight the ingredients of our proposal that assisted the improved BLEU score in Table 3 .

The results show that having both ?? and ?? as learned, in conjunction with the simplex constraint, was necessary for improved performance.

Figure 3: Convergence of the (??, ??) weights for the second encoder layer of Configuration (C) for the English-to-German newstest2013 task.

We smoothen the curves using a mean filter.

This shows that the network does prioritize some branches more than others and that the architecture does not exploit a subset of the branches while ignoring others.

Weights (??, ??) BLEU Learned 24.8 Random 21.1 Uniform 23.4 Table 4 : Performance of the architecture with random and uniform normalization weights on the newstest2013 English-to-German task for configuration (C).

This shows that the learned (??, ??) weights of the Weighted Transformer are crucial to its performance.

The proposed modification can also be interpreted as a form of Shake-Shake regularization proposed in BID6 .

In this regularization strategy, random weights are sampled during forward and backward passes for weighing the various branches in a multi-branch network.

During test time, they are weighed equally.

In our strategy, the weights are learned instead of being sampled randomly.

Consequently, no changes to the model are required during test time.

In order to better understand whether the network benefits from the learned weights or if, at test time, random or uniform weights suffice, we propose the following experiment: the weights for the Weighted Transformer, including (??, ??) are trained as before, but, during test time, we replace them with (a) randomly sampled weights, and (b) 1/M where M is the number of incoming branches.

In Table 4 , we report experimental results on the configuration (C) of the Weighted Transformer on the English-to-German newstest2013 data set (see TAB1 for details regarding the configuration).

It is evident that random or uniform weights cannot replace the learned weights during test time.

Preliminary experiments suggest that a Shake-Shake-like strategy where the weights are sampled randomly during training also leads to inferior performance.

In order to analyze whether a hard (discrete) choice through gating will outperform our normalization strategy, we experimented with using gates instead of the proposed concatenation-addition strategy.

Specifically, we replaced the summation in Equation FORMULA8 by a gating structure that sums up the contributions of the top k branches with the highest probabilities.

This is similar to the sparselygated mixture of experts model in .

Despite significant hyper-parameter tuning of k and M , we found that this strategy performs worse than our proposed mechanism by a large margin.

We hypothesize that this is due to the fact that the number of branches is low, typically less than 16.

Hence, sparsely-gated models lose representational power due to reduced capacity in the model.

We plan to investigate the setup with a large number of branches and sparse gates in future work.

We present the Weighted Transformer that trains faster and achieves better performance than the original Transformer network.

The proposed architecture replaces the multi-head attention in the Transformer network by a multiple self-attention branches whose contributions are learned as a part of the training process.

We report numerical results on the WMT 2014 English-to-German and English-to-French tasks and show that the Weighted Transformer improves the state-of-the-art BLEU scores by 0.5 and 0.4 points respectively.

Further, our proposed architecture trains 15 ??? 40% faster than the baseline Transformer.

Finally, we present evidence suggesting the regularizing effect of the proposal and emphasize that the relative improvement in BLEU score is observed across various hyper-parameter settings for both small and large models.

<|TLDR|>

@highlight

Using branched attention with learned combination weights outperforms the baseline transformer for machine translation tasks.