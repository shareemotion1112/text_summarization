Recurrent neural networks (RNN), convolutional neural networks (CNN) and self-attention networks (SAN) are commonly used to produce context-aware representations.

RNN can capture long-range dependency but is hard to parallelize and not time-efficient.

CNN focuses on local dependency but does not perform well on some tasks.

SAN can model both such dependencies via highly parallelizable computation, but memory requirement grows rapidly in line with sequence length.

In this paper, we propose a model, called "bi-directional block self-attention network (Bi-BloSAN)", for RNN/CNN-free sequence encoding.

It requires as little memory as RNN but with all the merits of SAN.

Bi-BloSAN splits the entire sequence into blocks, and applies an intra-block SAN to each block for modeling local context, then applies an inter-block SAN to the outputs for all blocks to capture long-range dependency.

Thus, each SAN only needs to process a short sequence, and only a small amount of memory is required.

Additionally, we use feature-level attention to handle the variation of contexts around the same word, and use forward/backward masks to encode temporal order information.

On nine benchmark datasets for different NLP tasks, Bi-BloSAN achieves or improves upon state-of-the-art accuracy, and shows better efficiency-memory trade-off than existing RNN/CNN/SAN.

Context dependency provides critical information for most natural language processing (NLP) tasks.

In deep neural networks (DNN), context dependency is usually modeled by a context fusion module, whose goal is to learn a context-aware representation for each token from the input sequence.

Recurrent neural networks (RNN), convolutional neural networks (CNN) and self-attention networks (SAN) are commonly used as context fusion modules.

However, each has its own merits and defects, so which network to use is an open problem and mainly depends on the specific task.

RNN is broadly used given its capability in capturing long-range dependency through recurrent computation.

It has been applied to various NLP tasks, e.g., question answering BID51 , neural machine translation BID0 , sentiment analysis , natural language inference BID29 , etc.

However, training the basic RNN encounters the gradient dispersion problem, and is difficult to parallelize.

Long short-term memory (LSTM) BID14 effectively avoids the vanishing gradient.

Gated recurrent unit (GRU) BID5 and simple recurrent unit (SRU) BID24 improve the efficiency by reducing parameters and removing partial temporal-dependency, respectively.

However, they still suffer from expensive time cost, especially when applied to long sequences.

CNN becomes popular recently on some NLP tasks because of its the highly parallelizable convolution computation BID7 .

Unlike RNN, CNN can simultaneously apply convolutions defined by different kernels to multiple chunks of a sequence BID19 .

It is mainly used for sentence-encoding tasks BID25 BID17 .

Recently, hierarchical CNNs, e.g. ByteNet BID18 , and ConvS2S BID8 , are proposed to capture relatively long-range dependencies by using stacking CNNs to increase the number of input BID2 .

The details of all the models are provided in Section 4.We propose an attention mechanism, called "bidirectional block self-attention (Bi-BloSA)", for fast and memory-efficient context fusion.

The basic idea is to split a sequence into several length-equal blocks (with padding if necessary), and apply an intra-block SAN to each block independently.

The outputs for all the blocks are then processed by an inter-block SAN.

The intra-block SAN captures the local dependency within each block, while the inter-block SAN captures the long-range/global dependency.

Hence, every SAN only needs to process a short sequence.

Compared to a single SAN applied to the whole sequence, such two-layer stacked SAN saves a significant amount of memory.

A feature fusion gate combines the outputs of intra-block and inter-block SAN with the original input, to produce the final contextaware representations of all the tokens.

Similar to directional self-attention (DiSA) BID42 , BiBloSA uses forward/backward masks to encode the temporal order information, and feature-level attention to handle the variation of contexts around the same word.

Further, a RNN/CNN-free sequence encoding model we build based on Bi-BloSA, called "bi-directional block self-attention network (Bi-BloSAN)", uses an attention mechanism to compress the output of Bi-BloSA into a vector representation.

In experiments 1 , we implement Bi-BloSAN and popular sequence encoding models on several NLP tasks, e.g., language inference, sentiment analysis, semantic relatedness, reading comprehension, question-type classification, etc.

The baseline models include Bi-LSTM, Bi-GRU, Bi-SRU, CNNs, multi-head attention and DiSAN.

A thorough comparison on nine benchmark datasets demonstrates the advantages of Bi-BloSAN in terms of training speed, inference accuracy and memory consumption.

FIG0 shows that Bi-BloSAN obtains the best accuracy by costing similar training time to DiSAN, and as little memory as Bi-LSTM, Bi-GRU and multi-head attention.

This shows that Bi-BloSAN achieves a better efficiency-memory trade-off than existing RNN/CNN/SAN models.

Our notations follow these conventions: 1) lowercase denotes a vector; 2) bold lowercase denotes a sequence of vectors (stored as a matrix); and 3) uppercase denotes a matrix or a tensor.

Word embedding is the basic processing unit in most DNN for sequence modeling.

It transfers each discrete token into a representation vector of real values.

Given a sequence of tokens (e.g., words or characters) w = [w 1 , w 2 , . . .

, w n ] ∈ R N ×n , where w i is a one-hot vector, N is the vocabulary size and n is the sequence length.

A pre-trained token embedding (e.g. word2vec BID31 ) is applied to w, which outputs a sequence of low dimensional vectors DISPLAYFORM0 de×n .

This process can be formally written as x = W (e) w, where W (e) ∈ R de×N is the embedding weight matrix that can be fine-tuned during the training phase.

Vanilla Attention:

Given an input sequence x = [x 1 , x 2 , . . . , x n ] composed of token embeddings and a vector representation of a query q ∈ R dq , vanilla attention BID0 computes the alignment score between q and each token x i (reflecting the attention of q to x i ) using a compatibility function f (x i , q).

A softmax function then transforms the alignment scores a ∈ R n to a probability distribution p(z|x, q), where z is an indicator of which token is important to q. A large p(z = i|x, q) means that x i contributes important information to q. This process can be written as DISPLAYFORM0 The output s is the expectation of sampling a token according to its importance, i.e., DISPLAYFORM1 Multiplicative attention (or dot-product attention) BID49 BID46 BID40 and additive attention (or multi-layer perceptron attention) BID0 BID41 are two commonly used attention mechanisms.

They differ in the choice of compatibility function f (x i , q).

Multiplicative attention uses the cosine similarity for f (x i , q), i.e., DISPLAYFORM2 where DISPLAYFORM3 ∈ R d h ×dq are the learnable parameters.

Additive attention is defined as DISPLAYFORM4 where DISPLAYFORM5 (1) and b are the biases, and σ(·) is an activation function.

Additive attention usually achieves better empirical performance than multiplicative attention, but is expensive in time cost and memory consumption.

Multi-dimensional Attention: Unlike vanilla attention, in multi-dimensional (multi-dim) attention BID42 , the alignment score is computed for each feature, i.e., the score of a token pair is a vector rather than a scalar, so the score might be large for some features but small for others.

Therefore, it is more expressive than vanilla attention, especially for the words whose meaning varies in different contexts.

Multi-dim attention has d e indicators z 1 , . . .

, z de for d e features.

Each indicator has a probability distribution that is generated by applying softmax to the n alignment scores of the corresponding feature.

Hence, for each feature k in each token i, we have P ki p(z k = i|x, q) where P ∈ R de×n .A large P ki means that the feature k in token i is important to q. The output of multi-dim attention is written as DISPLAYFORM6 For simplicity, we ignore the subscript k where no confusion is caused.

Then, Eq.(6) can be rewritten as an element-wise product, i.e., s = n i=1 P ·i x i .

Here, P ·i is computed by the additive attention in Eq.(5) where w T is replaced with a weight matrix W ∈ R d h ×de , which leads to a score vector for each token pair.

token2token self-attention BID15 BID49 BID42 produces context-aware representations by exploring the dependency between two tokens x i and x j from the same sequence x. In particular, q in Eq. FORMULA5 is replaced with x j , i.e., DISPLAYFORM0 Similar to the P in multi-dim attention, each input token x j is associated with a probability matrix DISPLAYFORM1 source2token self-attention BID28 BID42 BID29 explores the importance of each token to the entire sentence given a specific task.

In particular, q is removed from Eq. FORMULA5 , and the following equation is used as the compatibility function.

DISPLAYFORM2 The probability matrix P is defined as P ki p(z k = i|x).

The final output of source2token selfattention has the same form as multi-dim attention, i.e., s = Temporal order information is difficult to encode in token2token self-attention introduced above because the alignment score between two tokens is symmetric.

Masked self-attention BID42 ) applies a mask M ∈ R n×n to the alignment score matrix (or tensor due to feature-level score) computed by Eq.(7), so it allows one-way attention from one token to another.

Specifically, the bias b in Eq. FORMULA8 is replaced with a constant vector M ij 1, where the 1 is an all-one vector.

In addition, W is fixed to a scalar c and tanh(·/c) is used as the activation function σ(·), i.e., DISPLAYFORM3 where DISPLAYFORM4 The procedures to calculate the attention output from f (x i , x j ) are identical to those in token2token self-attention.

We use s = g m (x, M ) to denote the complete process of masked self-attention with s = [s 1 , s 2 , . . .

, s n ] as the output sequence.

An illustration of masked self-attention is given in FIG1 .In order to model bi-directional order information, forward mask M f w and backward mask M bw are respectively substituted into Eq.(9), which results in forward and backward self-attentions.

These two masks are defined as DISPLAYFORM5 The outputs of forward and backward self-attentions are denoted by DISPLAYFORM6

In this section, we first introduce the "masked block self-attention (mBloSA)" (Section 3.1) as a fundamental self-attention module.

Then, we present the "bi-directional block self-attention network (Bi-BloSAN)" (Section 3.2) for sequence encoding, which uses the "bi-directional block selfattention (Bi-BloSA)" (mBloSA with forward and backward masks) as its context fusion module.

As shown in FIG2 , masked block self-attention (mBloSA) has three parts from its bottom to top, i.e., 1) intra-block self-attention, 2) inter-block self-attention, and 3) the context fusion.

Intra-block self-attention: We firstly split the input sequence of token/word embeddings into m blocks of equal length r, i.e., [ DISPLAYFORM0 Padding can be applied to the last block if necessary.

Intra-block self-attention applies the masked self-attentions g m (·, M ) with shared parameters to all the blocks , i.e., DISPLAYFORM1 Its goal is to capture the local context dependency inside each block.

Similar to x l , the output representations of the tokens in the l-th block are denoted by h l = [h r(l−1)+1 , h r(l−1)+2 , . . .

, h r×l ].

Note, the block length r is a hyper-parameter and m = n/r.

In Appendix A, we introduce an approach to selecting the optimal r, which results in the maximum memory utility rate in expectation.

Inter-block self-attention: To generate a vector representation v l of each block, a source2token self-attention g s2t (·) is applied to the output h l of the intra-block self-attention on each block, i.e., DISPLAYFORM2 Note we apply the parameter-shared g s2t (·) to h l for different blocks.

This provides us with a sequence v = [v 1 , v 2 , . . .

, v m ] of local-context representations at block level.

Inter-block self-attention then applies a masked self-attention to v in order to capture the long-range/global dependency among the blocks, i.e., DISPLAYFORM3 To combine the local and global context features at block level, a gate is used to merge the input and the output of the masked self-attention dynamically.

This is similar to the gates in LSTM.

The output sequence e = [e 1 , . . . , e m ] of the gate is computed by DISPLAYFORM4 DISPLAYFORM5 Context fusion: Given the long-range context representations e = [e 1 , . . .

, e m ] ∈ R de×m at block level, we duplicate e l for r times to get e l = [e l , e l , . . .

, e l ] (each token in block l has the global context feature representation e l ).

Let E [e l ] m l=1 ∈ R de×n .

Now, we have the input sequence x of word embeddings, the local context features h produced by intra-block self-attention, and the long-range/global context features E produced by inter-block self-attention.

A feature fusion gate BID11 ) is employed to combine them, and generates the final context-aware representations of all tokens, i.e., DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 where σ(·) is an activation function, and u = [u 1 , u 2 , . . . , u n ] ∈ R de×n is the mBloSA output, which consists of the context-aware representations of the n tokens.

We propose a sequence encoding model "Bi-directional block self-attention network (Bi-BloSAN)" with mBloSA as its major components.

Its architecture is shown in FIG3 .

In Bi-BloSAN, two fully connected layers (with untied parameters) are applied to the input sequence of token embeddings.

Their outputs are processed by two mBloSA modules respectively.

One uses the forward mask M f w and another uses the backward mask M bw .

Their outputs u f w and u bw are concatenated as

.

The idea of bi-directional attention follows the same spirit as Bi-LSTM and DiSAN.

It encodes temporal order information lacking in existing SAN models.

The context fusion module in Bi-BloSAN, with the input x and the output u bi , is called "Bi-BloSA".

In order to obtain a sequence encoding, a source2token self-attention transforms the sequence u bi of concatenated token representations into a vector representation s.

We conduct the experiments of Bi-BloSAN and several popular RNN/CNN/SAN-based sequence encoding models on nine benchmark datasets for multiple different NLP tasks.

Note that, in some baseline models, a source2token self-attention is on the top of the models to generate an encoding for the entire sequence.

All the models used for comparisons are listed as follows.• Bi-LSTM: 600D Bi-directional LSTM (300D forward LSTM + 300D backward LSTM) BID12 ).• Bi-GRU: 600D Bi-directional GRU BID5 .• Bi-SRU: 600D Bi-directional SRU BID24 ) (with sped-up recurrence but no CUDA level optimization for fair comparison).• Multi-CNN: 600D CNN sentence embedding model BID19 ) (200D for each of 3, 4, 5-gram).• Hrchy-CNN: 3-layer 300D CNN BID8 with kernel length 5, to which gated linear units BID6 and residual connection are applied.• Multi-head: 600D Multi-head attention BID49 ) (8 heads, each has 75 hidden units).

The positional encoding method used in BID49 is applied to the input sequence to encode temporal order information.• DiSAN: 600D Directional self-attention network BID42 (300D forward masked self-attention + 300D backward masked self-attention).All experimental codes are implemented in Python with Tensorflow and run on a single Nvidia GTX 1080Ti graphic card.

Both time cost and memory load data are collected under Tensorflow1.3 with CUDA8 and cuDNN6021.

In the rest of this section, we conduct the experiments on natural language inference in Section 4.1, reading comprehension in Section 4.2, semantic relatedness in Section 4.3 and sentence classifications in Section 4.4.

Finally, we analyze the time cost and memory load of the different models vs. the sequence length in Section 4.5.

Natural language inference (NLI) aims to reason the semantic relationship between a pair of sentences, i.e., a premise sentence and a hypothesis sentence.

This relationship could be entailment, neutral or contradiction.

In the experiment, we compare Bi-BloSAN to other baselines on the Stanford Natural Language Inference (Bowman et al., 2015) (SNLI) 2 dataset, which contains standard training/dev/test split of 549,367/9,842/9,824 samples.

BID3 3.0m 83.9 80.6 1024D GRU encoders BID50 15.0m 98.8 81.4 300D Tree-based CNN encoders BID32 3.5m 83.3 82.1 300D SPINN-PI encoders BID3 3.7m 89.2 83.2 600D Bi-LSTM encoders BID29 2.0m 86.4 83.3 300D NTI-SLSTM-LSTM encoders BID34 4.0m 82.5 83.4 600D Bi-LSTM encoders+intra-attention BID29 2.8m 84.5 84.2 300D NSE encoders BID33 3.0m 86.2 84.6 600D (300+300) Deep Gated Attn.

BID4 11.6m 90.5 85.5Bi-LSTM BID12 2.9m 90.4 85.0 Bi-GRU BID5 2.5m 91.9 84.9 Bi-SRU BID24 2.0m 88.4 84.8 Multi-CNN BID19 1.4m 89.3 83.2 Hrchy-CNN BID8 3.4m 91.3 83.9 Multi-head BID49 2.0m 89.6 84.2 DiSAN BID42 2 h is passed into a 300D fully connected layer, whose output is given to a 3-unit output layer with softmax to calculate the probability distribution over the three classes.

Training Setup: The optimization objective is the cross-entropy loss plus L2 regularization penalty.

We minimize the objective by Adadelta (Zeiler, 2012) optimizer which is empirically more stable than Adam BID21 on SNLI.

The batch size is set to 64 for all methods.

The training phase takes 50 epochs to converge.

All weight matrices are initialized by Glorot Initialization BID9 , and the biases are initialized with 0.

We use 300D GloVe 6B pre-trained vectors BID37 to initialize the word embeddings in x. The Out-of-Vocabulary words in the training set are randomly initialized by uniform distribution between (−0.05, 0.05).

The word embeddings are fine-tuned during the training.

The Dropout BID45 ) keep probability and the L2 regularization weight decay factor γ are set to 0.75 and 5×10−5 , respectively.

The number of hidden units is 300.

The unspecified activation functions in all models are set to Relu BID10 .In TAB0 , we report the number of parameters, and training/test accuracies of all baselines plus the methods from the official leaderboard.

For fair comparison, we use 480D Bi-BloSAN, which leads to the similar parameter number with that of baseline encoders.

Bi-BloSAN achieves the best test accuracy (similar to DiSAN) among all the sentence encoding models on SNLI.

In particular, compared to the RNN models, Bi-BloSAN outperforms Bi-LSTM encoder, Bi-LSTM with attention and deep gated attention by 2.4%, 1.5% and 0.2%, respectively.

Bi-BloSAN can even perform better than the semantic tree based models: SPINN-PI encoder (+2.5%)&Tree-based CNN encoder (+3.6%), and the memory network based model: NSE encoder (+1.1%).

Additionally, Bi-BloSAN achieves the best performance among the baselines which are based on RNN/CNN/SAN.

It outperforms Bi-LSTM (+0.7%), Bi-GRU (+0.8%), Bi-SRU (+0.9%), multi-CNN (+2.5%), Hrchy-CNN (+1.8%) and multi-head attention (+1.5%).

In addition, we compare time cost and memory consumption of all the baselines in TAB2 .

Compared to DiSAN with the same test accuracy, Bi-BloSAN is much faster and more memory efficient.

In terms of training and inference time, Bi-BloSAN is 3 ∼ 4× faster than the RNN models (Bi-LSTM, Bi-GRU, etc.) .

It is as fast as CNNs and multi-head attention but substantially outperforms them in test accuracy.

In terms of training memory, Bi-BloSAN requires similar GPU memory to the RNN-based models and multi-head attention, which is much less than that needed by DiSAN.Finally, we conduct an ablation study of Bi-BloSAN in TAB3 .

In particular, we evaluate the contribution of each part of Bi-BloSAN by the change of test accuracy after removing the part.

The removed part could be: 1) local context representations h, 2) global context representations E, 3) the context fusion module (mBloSA) or 4) all fundamental modules appeared in this paper.

The results show that both the local and global context representations play significant roles in BiBloSAN.

They make Bi-BloSAN surpass the state-of-the-art models.

Moreover, mBloSA improves the test accuracy from 83.1% to 85.7%.

Source2token self-attention performs much better than vanilla attention, and improves the test accuracy by 3.3%.

Given a passage and a corresponding question, the goal of reading comprehension is to find the correct answer from the passage for the question.

We use the Stanford Question Answering Dataset BID39 ) (SQuAD) 3 to evaluate all models.

SQuAD consists of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text, or a span, from the corresponding passage.

Since Bi-BloSAN and other baselines are designed for sequence encoding, such as sentence embedding, we change the task from predicting the answer span to locating the sentence containing the correct answer.

We build a network structure to test the power of sequence encoding in different models to find the correct answers.

The details are given in Appendix B.Training Setup: We use Adadelta optimizer to minimize the cross-entropy loss plus L2 regularization penalty, with batch size of 32.

The network parameters and word embeddings initialization methods are same as those for SNLI, except that both the word embedding dimension and the number of hidden units are set to 100.

We use 0.8 dropout keep probability and 10 −4 L2 regularization weight decay factor.

We evaluate the Bi-BloSAN and the baselines except DiSAN because the memory required by DiSAN largely exceeds the GPU memory of GTX 1080Ti (11GB).

The number of parameters, per epoch training time and the prediction accuracy on development set are given in TAB4 .

BID5 0.57m 782 67.98 Bi-SRU BID24 0.32m 737 67.32 Multi-CNN BID19 0.60m 114 63.58 Multi-head BID49 Compared to RNN/CNN models, Bi-BloSAN achieves state-of-the-art prediction accuracy in this modified task.

Bi-BloSAN shows its competitive context fusion and sequence encoding capability compared to Bi-LSTM, Bi-GRU, Bi-SRU but is much more time-efficient.

In addition, Bi-BloSAN significantly outperforms multi-CNN and multi-head attention.

The goal of semantic relatedness is to predict the similarity degree of a given pair of sentences.

Unlike the classification problems introduced above, predicting the semantic relatedness of sentences is a regression problem.

We use s 1 and s 2 to denote the encodings of the two sentences, and assume that the similarity degree is between [1, K] .

Following the method introduced by BID47 , the concatenation of s 1 s 2 and |s 1 −s 2 | is used as the representation of sentence related-ness.

This representation is fed into a 300D fully connected layer, followed by a K-unit output layer with softmax to calculate a probability distributionp.

The details of this regression problem can be found in Appendix C. We evaluate all models on Sentences Involving Compositional Knowledge (SICK) 4 dataset, where the similarity degree is denoted by a real number in the range of [1, 5] .

SICK comprises 9,927 sentence pairs with 4,500/500/4,927 instances for training/dev/test sets.

Training Setup: The optimization objective of this regression problem is the KL-divergence plus the L2 regularization penalty.

We minimize the objective using Adadelta with batch size of 64.

The network parameters and word embeddings are initialized as in SNLI experiment.

The keep probability of dropout is set to 0.7, and the L2 regularization weight decay factor is set to 10 −4 .

BID57 0.8414 / / DT-RNN 0.7923 (0.0070) 0.7319 (0.0071) 0.3822 (0.0137) SDT-RNN 0.7900 (0.0042) 0.7304 (0.0042) 0.3848 (0.0042) Constituency Tree-LSTM BID47 The performances of all models are shown in TAB6 , which shows that Bi-BloSAN achieves stateof-the-art prediction quality.

Although Dependency Tree-LSTM and DiSAN obtain the best performance, the Tree-LSTM needs external semantic parsing tree as the recursive input and expensive recursion computation, and DiSAN requires much larger memory for self-attention calculation.

By contrast, Bi-BloSAN, as a RNN/CNN-free model, shows appealing advantage in terms of memory and time efficiency.

Note that, performance of Bi-BloSAN is still better than some common models, including Bi-LSTM, CNNs and multi-head attention.

The goal of sentence classification is to correctly predict the class label of a given sentence in various scenarios.

We evaluate the models on six sentence classification benchmarks for various NLP tasks, such as sentiment analysis and question-type classification.

They are listed as follows.• CR 5 : Customer reviews BID16 of various products (cameras etc.).

This task is to predict whether the review is positive or negative.• MPQA 6 : Opinion polarity detection subtask of the MPQA dataset BID53 .•

SUBJ 7 : Subjectivity dataset BID35 , which includes a set of sentences.

The corresponding label indicates whether each sentence is subjective or objective.• TREC 8 : TREC question-type classification dataset BID26 which coarsely classifies the question sentences into six types.• SST-1 9 : Stanford Sentiment Treebank BID43 , which is a dataset consisting of movie reviews with five fine-grained sentiment labels, i.e., very positive, positive, neutral, negative and very negative.• SST-2: Stanford Sentiment Treebank BID43 with binary sentiment labels.

Compared to SST-1, SST-2 removes the neutral instances, and labels the rest with either negative or positive.

Note that only SST-1 and SST-2 have the standard training/dev/test split, and TREC has the training/dev split.

We implement 10-fold cross validation on SUBJ, CR and MPQA because the original datasets do not provide any split.

We do not use the Movie Reviews BID36 dataset because the SST-1/2 are extensions of it.

Training Setup: We use the cross-entropy loss plus L2 regularization penalty as the optimization objective.

We minimize it by Adam with training batch size of 32 (except DiSAN, which uses batch size of 16 due to the limit of GPU memory).

The network parameters and word embeddings are initialized as in SNLI experiment.

To avoid overfitting on small datasets, we decrease the dropout keep probability and the L2 regularization weight decay factor γ to 0.6 and 10 −4 , respectively.

BID56 83.6 (1.6) 90.4 (0.7) 92.2 (1.2) 91.1 (1.0) / / SRU BID24 84 .

The prediction accuracies of different models on the six benchmark datasets are given in TAB7 .

Bi-BloSAN achieves the best prediction accuracies on CR, MPQA and TREC, and state-of-theart performances on SUBJ, SST-1 and SST-2 datasets (slightly worse than the best performances).

Although Bi-BloSAN performs a little bit worse than the RNN models on SUBJ and SST-1, it is much more time-efficient than them.

Additionally, on the SST-2 dataset, Bi-BloSAN performs slightly worse than DiSAN in terms of prediction accuracy (−0.4%) but obtains a significantly higher memory utility rate.

We visualize the progress of training models on CR dataset in Figure 5 .

The convergence speed of Bi-BloSAN is∼ 6× and∼ 2× faster than Bi-LSTM and DiSAN respectively.

Although Bi-BloSAN is less time-efficient than CNN and multi-head attention, it has much better prediction quality.

To compare the efficiency-memory trade-off for each model on sequences of different lengths, we generate random tensor data, and feed them into the different sequence encoding models.

The models we evaluate include Bi-LSTM, Bi-GRU, Bi-SRU, CNN, multi-head attention, DiSAN and Bi-BloSAN.

The shape of the random data is [batch size, sequence length, features number].

We fix the batch size to 64 and the features number to 300, then change the sequence length from 16 to 384 with a step size 16.We first discuss the time cost vs. the sequence length.

As shown in Figure 6 (a), the inference time of Bi-BloSAN is similar to those of multi-head attention and multi-CNN, but Bi-BloSAN outperforms both by a large margin on prediction quality in previous experiments.

Moreover, Bi-BloSAN is much faster than the RNN models (Bi-LSTM, Bi-GRU, BI-SRU) .

In addition, although DiSAN requires less training time than the RNN models in the experiments above, it is much slower during the inference phase because the large memory allocation consumes a great amount of time.

By contrast, the block structure of Bi-BloSAN significantly reduces the inference time.

The GPU memory consumption vs. the sequence length for each model is visualized in Figure 6 (b).

DiSAN is not scalable because its memory grows explosively with the sequence length.

Bi-BloSAN is more memory-efficient and scalable than DiSAN as the growth of its memory is nearly linear.

Although Bi-BloSAN consumes more memory than the RNN models, it experimentally has better time efficiency and prediction quality.

Since multi-head attention uses multiplicative attention, it requires less memory than all additive attention based models, such as DiSAN and Bi-BloSAN, but multiplicative attention based models usually perform worse than additive attention based models.

This paper presents an attention network, called bi-directional block self-attention network (BiBloSAN) , for fast, memory-efficient and RNN/CNN-free sequence modeling.

To overcome large memory consumption of existing self-attention networks, Bi-BloSAN splits the sequence into several blocks and employs intra-block and inter-block self-attentions to capture both local and long-range context dependencies, respectively.

To encode temporal order information, Bi-BloSAN applies forward and backward masks to the alignment scores between tokens for asymmetric selfattentions.

Our experiments on nine benchmark datasets for various different NLP tasks show that Bi-BloSAN can achieve the best or state-of-the-art performance with better efficiency-memory trade-off than existing RNN/CNN/SAN models.

Bi-BloSAN is much more time-efficient than the RNN models (e.g., Bi-LSTM, Bi-GRU, etc.), requires much less memory than DiSAN, and significantly outperforms the CNN models and multi-head attention on prediction quality.

In mBloSA, the length r of each block is a hyper-parameter that determines memory consumption of mBloSA.

To minimize the memory consumption, we propose an approach that calculates the optimized block length r as follows.

We first introduce the method for determining r for a dataset that has fixed sentence length n. Given the sentence length n and the block number m = n/r, we have the following facts: 1) the major memory consumption in mBloSA is dominated by the masked self-attentions g m (·, M ); 2) the memory consumption of the masked self-attention is proportional to the square of the sentence length; and 3) mBloSA contains m masked self-attention with a sequence length of r, and 1 masked self-attention with a sequence length of m. Therefore, the memory ξ required by mBloSA can be calculated by DISPLAYFORM0 By setting the gradient of ξ w.r.t.

r to zero, we know that the memory consumption ξ is minimum when r = 3 √ 2n.

Second, we propose a method for selecting r given a dataset with the sentence lengths that follow a normal distribution N (µ, σ 2 ).

We consider the case where mini-batch SGD with a batch size of B or its variant is used for training.

We need to calculate the upper bound of the expectation of the maximal sentence length for each mini-batch.

Let us first consider B random variables [X 1 , X 2 , . . .

, X B ] in the distribution N (0, σ 2 ).

The goal is to find the upper bound of the expectation of random variable Z + µ, where Z is defined as Z = max DISPLAYFORM1 By Jensen's inequality, DISPLAYFORM2 Eq.(21) leads to DISPLAYFORM3 DISPLAYFORM4 and we obtain the following upper bound.

DISPLAYFORM5 Hence, the upper bound of the expectation of the maximal sentence length among all the B sentences in each mini-batch is σ √ 2 ln B + µ. Therefore, the block length r is computed by DISPLAYFORM6

Each sample in the Stanford Question Answering Dataset (SQuAD) BID39 ) is composed of three parts, i.e., a passage consisting of multiple sentences, a question sentence and a span in the passage indicating the position of the answer.

In order to evaluate the performance of sentence embedding models, we change the task from predicting the span of the answer to finding the sentence containing the correct answer.

Given a passage consisting of m sentences [s 1 , s 2 , . . . , s m ] where s k = [x k1 , x k2 , . . . , x kn ], and the embedded question token sequence q = [q 1 , q 2 , . . .

, q l ], the goal is to predict which sentence in the m sentences contains the correct answer to the question q.

The neural net we use to evaluate different sequence encoding models is given in FIG7 .

First, we process each sentence from the passage by a context fusion layer with shared parameters, followed by a source2token self-attention with shared parameters, which outputs a vector representation of the sentence.

Therefore, the m sentences are represented by m vectors [u 1 , u 2 , . . . , u m ].

The question sentence q is compressed into a vector representation q using source2token self-attention.

Second, we combine each sentence u k with q by concatenating u k , q, u k −q and u k q, i.e., c k = [u k ; q; u k −q; u k q], for k = 1, 2, . . . , m.(24) Then c = [c 1 , c 2 , . . .

, c m ] is fed into another context fusion layer that explores sentence-level dependencies.

Finally, the resultant output representation of each sentence is separately fed into a fully connected layer to compute a scalar score indicating the possibility of the sentence containing the answer.

A softmax function is applied to the scores of all m sentences, to generate a probability distributionp ∈ R m for cross-entropy loss function.

The sentence with the largest probability is predicted as the sentence containing the answer.

Following the setting introduced by BID47 and given a predicted probability distributionp as the output of a feedforward network, the regression model predicts the similarity degree aŝ DISPLAYFORM0 where β = [1, 2, . . .

, K].

The ground-truth similarity degree y should be mapped to a probability distribution p = [p i ]

We use KL-divergence between p andp as our loss function, i.e., DISPLAYFORM1 where the p (k) andp (k) represent the target and predicted probability distributions of the k-th sample, respectively.

Recently, several structured attention mechanisms BID20 BID23 are proposed for capturing structural information from input sequence(s).

When applied to selfattention, structured attentions share a similar idea to self-alignment attention BID15 and multi-head attention BID49 with one head, which aims to model the dependencies between the tokens.

Similar to the attention from multiple perspectives in multi-head attention, multi-perspective context matching explores the dependencies between passage and question from multiple perspectives for reading comprehension, while self-attentive structure BID28 embeds sentences from various perspectives to produce matrix representations of the sentences.

In recursive models, self-attention over children nodes BID48 can provide effective input for their parent node that has no standard input BID47 as long as it is a non-leaf node in a semantic constituency parsing tree.

BID27 applies the multi-hop attention mechanism to transfer learning for cross-domain sentiment analysis without any RNN/CNN structure.

<|TLDR|>

@highlight

A self-attention network for RNN/CNN-free sequence encoding with small memory consumption, highly parallelizable computation and state-of-the-art performance on several NLP tasks

@highlight

Proposes applyting self-attention at two levels to limit the memory requirement in attention-based models with a negligible impact on speed.

@highlight

This paper introduces bi-directional block self-attention model as a general-purpose encoder for various sequence modeling tasks in NLP