Deep learning has yielded state-of-the-art performance on many natural language processing tasks including named entity recognition (NER).

However, this typically requires large amounts of labeled data.

In this work, we demonstrate that the amount of labeled training data can be drastically reduced when deep learning is combined with active learning.

While active learning is sample-efficient, it can be computationally expensive since it requires iterative retraining.

To speed this up, we introduce a lightweight architecture for NER, viz., the CNN-CNN-LSTM model consisting of convolutional character and word encoders and a  long short term memory (LSTM) tag decoder.

The model achieves nearly state-of-the-art performance on standard datasets for the task while being computationally much more efficient than best performing models.

We carry out incremental active learning, during the training process, and are able to nearly match state-of-the-art performance with just 25\% of the original training data.

Over the past few years, papers applying deep neural networks (DNNs) to the task of named entity recognition (NER) have successively advanced the state-of-the-art BID7 BID17 BID24 BID6 BID48 .

However, under typical training procedures, the advantages of deep learning diminish when working with small datasets.

For instance, on the OntoNotes-5.0 English dataset, whose training set contains 1,088,503 words, a DNN model outperforms the best shallow model by 2.24% as measured by F1 score BID6 .

However, on the comparatively small CoNLL-2003 English dataset, whose training set contains 203,621 words, the best DNN model enjoys only a 0.4% advantage.

To make deep learning more broadly useful, it is crucial to reduce its training data requirements.

Generally, the annotation budget for labeling is far less than the total number of available (unlabeled) samples.

For NER, getting unlabeled data is practically free, owing to the large amount of content that can be efficiently scraped off the web.

On the other hand, it is especially expensive to obtain annotated data for NER since it requires multi-stage pipelines with sufficiently well-trained annotators BID19 BID5 .

In such cases, active learning offers a promising approach to efficiently select the set of samples for labeling.

Unlike the supervised learning setting, in which examples are drawn and labeled at random, in the active learning setting, the algorithm can choose which examples to label.

Active learning aims to select a more informative set of examples in contrast to supervised learning, which is trained on a set of randomly drawn examples.

A central challenge in active learning is to determine what constitutes more informative and how the active learner can recognize this based on what it already knows.

The most common approach is uncertainty sampling, in which the model preferentially selects examples for which it's current prediction is least confident.

Other approaches include representativeness-based sampling where the model selects a diverse set that represent the input space without adding too much redundancy.

In this work, we investigate practical active learning algorithms on lightweight deep neural network architectures for the NER task.

Training with active learning proceeds in multiple rounds.

Traditional active learning schemes are expensive for deep learning since they require complete retraining of the classifier with newly annotated samples after each round.

In our experiments, for example, the model must be retrained 54 times.

Because retraining from scratch is not practical, we instead carry out incremental training with each batch of new labels: we mix newly annotated samples with the older ones, and update our neural network weights for a small number of epochs, before querying for labels in a new round.

This modification drastically reduces the computational requirements of active learning methods and makes it practical to deploy them.

We further reduce the computational complexity by selecting a lightweight architecture for NER.

We propose a new CNN-CNN-LSTM architecture for NER consisting of a convolutional character-level encoder, convolutional word-level encoder, and long short term memory (LSTM) tag decoder.

This model handles out-of-vocabulary words gracefully and, owing to the greater reliance on convolutions (vs recurrent layers), trains much faster than other deep models while performing competitively.

We introduce a simple uncertainty-based heuristic for active learning with sequence tagging.

Our model selects those sentences for which the length-normalized log probability of the current prediction is the lowest.

Our experiments with the Onto-Notes 5.0 English and Chinese datasets demonstrate results comparable to the Bayesian active learning by disagreement method .

Moreover our heuristic is faster to compute since it does not require multiple forward passes.

On the OntoNotes-5.0 English dataset, our approach matches 99% of the F1 score achieved by the best deep models trained in a standard, supervised fashion despite using only a 24.9% of the data.

On the OntoNotes-5.0 Chinese dataset, we match 99% performance with only 30.1% of the data.

Thus, we are able to achieve state of art performance with drastically lower number of samples.

The use of DNNs for NER was pioneered by BID7 , who proposed an architecture based on temporal convolutional neural networks (CNNs) over the sequence of words.

Since then, many papers have proposed improvements to this architecture.

BID17 proposed to replace CNN encoder in BID7 with bidirectional LSTM encoder, while BID24 and BID6 introduced hierarchy in the architecture by replacing hand-engineered character-level features in prior works with additional bidirectional LSTM and CNN encoders respectively.

In other related work, BID30 and BID34 pioneered the use of recurrent neural networks (RNNs) for decoding tags.

However, most recent competitive approaches rely upon CRFs as decoder BID24 BID6 BID48 .

In this work, we demonstrate that LSTM decoders outperform CRF decoders and are faster to train when the number of entity types is large.

While learning-theoretic properties of active learning algorithms are wellstudied BID10 BID3 BID1 BID47 , classic algorithms and guarantees cannot be generalized to DNNs, which are currently are the state-of-the-art techniques for NER.

Owing to the limitations of current theoretical analysis, more practical active learning applications employ a range of heuristic procedures for selecting examples to query.

For example, BID44 suggests a margin-based selection criteria, while BID39 while BID40 combines multiple criteria for NLP tasks.

BID8 explores the application of least confidence criterion for linear CRF models on sequence prediction tasks.

For a more comprehensive review of the literature, we refer to BID38 and BID35 .

While DNNs have achieved impressive empirical results across diverse applications BID22 BID29 , active learning approaches for these models have yet to be well studied, and most current work addresses image classification.

BID45 claims to be the first to study active learning for image classification with CNNs and proposes methods based on uncertainty-based sampling, while and BID18 show that sampling based on a Bayesian uncertainty measure can be more advantageous.

In one related paper, CNNs.

However, to our knowledge, prior to this work, deep active learning for sequence tagging tasks, which often have structured output space and variable-length input, has not been studied.

Most active learning methods require frequent retraining of the model as new labeled examples are acquired.

Therefore, it is crucial that the model can be efficiently retrained.

On the other hand, we would still like to reach the level of performance rivaling state-of-the-art DNNs.

To accomplish this, we first identify that many DNN architectures for NER can be decomposed into three components: 1) the character-level encoder, which extracts features for each word from characters, 2) the word-level encoder which extracts features from the surrounding sequence of words, and 3) the tag decoder, which induces a probability distribution over any sequences of tags.

This conceptual framework allows us to view a variety of DNNs in a unified perspective; see Table 1 .Owing to the superior computational efficiency of CNNs over LSTMs, we propose a lightweight neural network architecture for NER, which we name CNN-CNN-LSTM and describe below.

We represent each input sentence as follows; First, special [BOS] and [EOS] tokens are added at the beginning and the end of the sentence, respectively.

In order to batch the computation of multiple sentences, sentences with similar length are grouped together into buckets, and [PAD] tokens are added at the end of sentences to make their lengths uniform inside of the bucket.

We follow an analogous procedure to represent the characters in each word.

For example, the sentence 'Kate lives on Mars' is formatted as shown in TAB2 .

The formatted sentence is denoted as {x ij }, where x ij is the one-hot encoding of the j-th character in the i-th word.

Character-Level Encoder For each word i, we use CNNs BID25 to extract characterlevel features w char i FIG0 ).

While LSTM recurrent neural network BID16 slightly outperforms CNN as a character-level encoder, the improvement is not statistically significant and the computational cost of LSTM encoders is much higher than CNNs (see Section 5, also BID37 for detailed analysis).We apply ReLU nonlinearities BID32 and dropout BID41 between CNN layers, and include a residual connection between input and output of each layer BID14 .

So that our representation of the word is of fixed length, we apply max-pooling on the outputs of the topmost layer of the character-level encoder BID20 .

DISPLAYFORM0 x22 x23 x24 x25 x26 x27 Word-Level Encoder To complete our representation of each word, we concatenate its characterlevel features with w emb i , a latent word embedding corresponding to that word: DISPLAYFORM1 DISPLAYFORM2 We initialize the latent word embeddings with with word2vec training BID31 and then update the embeddings over the course of training.

In order to generalize to words unseen in the training data, we replace each word with a special [UNK] (unknown) token with 50% probability during training, an approach that resembles the word-drop method due to BID24 .Given the sequence of word-level input features w DISPLAYFORM3 Enc n for each word position in the sentence using a CNN.

In FIG3 , we depict an instance of our architecture with two convolutional layers and kernels of width 3.

We concatenate the representation at the l-th convolutional layer h LSTM RNNs can also perform word-level encoding BID17 , and models with LSTM word-level encoding give a slight (but not significant) boost over CNN word-level encoders in terms of F1 score (see Section 5).

However, CNN word-level encoders are considerably faster BID42 , which is crucial for the iterative retraining in our active learning scheme.

DISPLAYFORM4

The tag decoder induces a probability distribution over sequences of tags, conditioned on the word-level encoder features: P y 2 , y 3 , . . .

, y n???1 | h Enc i 1 .

Chain CRF BID23 ) is a popular choice for tag decoder, adopted by most modern DNNs for NER: DISPLAYFORM0 where W , A, b are learnable parameters, and {??} ti refers to the t i -th coordinate of the vector.

To compute the partition function of (1), which is required for training, usually dynamic programming is employed, and its time complexity is O(nT 2 ) where T is the number of entity types BID7 .Alternatively, we use an LSTM RNN for the tag decoder, as depicted in FIG4 .

At the first time step, the [GO]-symbol is provided as y 1 to the decoder LSTM.

At each time step i, the LSTM decoder computes h Dec i+1 , the hidden state for decoding word i + 1, using the last tag y i , the current decoder hidden state h Dec i , and the learned representation of next word h Enc i+1 .

Using a softmax loss function, y i+1 is decoded; this is further fed as an input to the next time step.

DISPLAYFORM1 Kate lives on Mars Since this is a locally normalized model BID0 , it does not require the costly computation of partition function, and it allows us to significantly speed up training compared to using CRFs.

Also, we observed that while it is computationally intractable to find the best sequence of tags with an LSTM decoder, greedily decoding tags from left to right yields the performance comparable to chain CRF decoder (see Appendix A).

While the use of RNNs tag decoders has been explored BID30 BID34 BID49 , we demonstrate for the first time that models using RNNs instead of CRFs for tag decoder can achieve state-of-the-art performance.

See Section 5.

Labeling data for NER usually requires manual annotations by human experts, which are costly to acquire at scale.

Active learning seeks to ameliorate this problem by strategically choosing which examples to annotate, in the hope of getting greater performance with fewer annotations.

To this end, we consider the following setup for interactively acquiring annotations.

The learning process consists of multiple rounds: At the beginning of each round, the active learning algorithm chooses sentences to be annotated up to the predefined budget.

After receiving annotations, we update the model parameters by training on the augmented dataset, and proceeds to the next round.

We assume that the cost of annotating a sentence is proportional to the number of words in the sentence and that every word in the selected sentence must be annotated at once, i.e. we do not allow or account for partially annotated sentences.

While various existing active learning strategies suit this setup BID38 , we explore the uncertainty sampling strategy.

With the uncertainty-based sampling strategy (Lewis & Gale, 1994), we rank the unlabeled examples according to the current model's uncertainty in its prediction of the corresponding labels.

We consider three ranking methods, each of which can be easily implemented in the CNN-CNN-LSTM model or most other deep neural approaches to NER.Least Confidence (LC) BID8 proposed to sort examples in ascending order according to the probability assigned by the model to the most likely sequence of tags: DISPLAYFORM0 Exactly computing (2) requires identifying the most likely sequence of tags according to the LSTM decoder.

Because determining the most likely sequence is intractable, we approximate the score by using the probability assigned to the greedily decoded sequence.

Maximum Normalized Log-Probability (MNLP): Preliminary analysis revealed that the LC method disproportionately selects longer sentences.

Note that sorting unlabeled examples in descending order by (2) is equivalent to sorting in ascending order by the following scores: DISPLAYFORM1 Since (3) contains summation over words, LC method naturally favors longer sentences.

Because longer sentences requires more labor for annotation, we find this undesirable, and propose to normalize (3) as follows, which we call Maximum Normalized Log-Probability method: DISPLAYFORM2 Bayesian Active Learning by Disagreement (BALD): We also consider sampling according to the measure of uncertainty proposed by .

Observing a correspondence between dropout BID41 and deep Gaussian processes BID9 , they propose that the variability of the predictions over successive forward passes due to dropout can be interpreted as a measure of the model's uncertainty BID11 .

Denote P 1 , P 2 , . . .

P M as models resulting from applying M independently sampled dropout masks.

One measure of our uncertainty on the ith word is f i , the fraction of models which disagreed with the most popular choice: DISPLAYFORM3 where |??| denotes cardinality of a set.

We normalize this by the number of words as 1 n n j=1 f j , In this paper, we draw M = 100 independent dropout masks.

Other Sampling Strategies.

Consider that the confidence of the model can help to distinguish between hard and easy samples.

Thus, sampling examples where the model is uncertain might save us from sampling too heavily from regions where the model is already proficient.

But intuitively, when we query a batch of examples in each round, we might want to guard against querying examples that are too similar to each other, thus collecting redundant information.

We also might worry that a purely uncertainty-based approach would oversample outliers.

Thus we explore techniques to guard against these problems by selecting a set of samples that is representative of the dataset.

Following , we express the problem of maximizing representativeness of a labeled set as a submodular optimization problem, and provide an efficient streaming algorithm adapted to use a constraint suitable to the NER task.

Our approach to representativeness-based sampling proceeds as follows: Denote X as the set of all samples, and X L , X U representing the set of labeled and unlabeled samples respectively.

For an unlabeled set S ??? X U , the utility f w is defined as the summation of marginal utility gain over all unlabeled points, weighted by their uncertainty.

More formally, DISPLAYFORM4 where US(i) is the uncertainty score on example i.

In order to find a good set S with high f w value, we exploit the submodularity of the function, and use an online algorithm under knapsack constraint.

More details of this method can be found in the supplementary material (Appendix C).

In our experiments, this approach fails to match the uncertainty-based heuristics or to improve upon them when used in combination.

Nevertheless, we describe the algorithm and include the negative results for their scientific value.

FORMULA5 ).

Dropout probabilities are all set as 0.5.

We use structured skip-gram model BID28 trained on Gigawords-English corpus BID13 , which showed a good boost over vanilla skip-gram model BID31 we do not report here.

We use vanilla stochastic gradient descent, since it is commonly reported in the named entity recognition literature that this outperforms more sophisticated methods at convergence BID24 BID6 .

We uniformly set the step size as 0.001 and the batch size as 128.

When using LSTMs for the tag decoder, for inference, we only use greedy decoding; beam search gave very marginal improvement in our initial experiments.

We repeat each experiment four times, and report mean and standard deviation.

In terms of measuring the training speed of our models, we compute the time spent for one iteration of training on the dataset, with eight K80 GPUs in p2.8xlarge on Amazon Web Services 2 .

TAB5 show the comparison between our model and other best performing models.

LSTM tag decoder shows performance comparable to CRF tag decoder, and it works better than the CRF decoder when used with CNN encoder; compare CNN-CNN-LSTM vs. CNN-CNN-CRF on both tables.

On the CoNLL-2003 English dataset which has only four entity types, the training speed of CNN-CNN-LSTM and CNN-CNN-CRF are comparable.

However, on the OntoNotes 5.0 English dataset which has 18 entity types, the training speed of CNN-CNN-LSTM is twice faster than CNN-CNN-CRF because the time complexity of computing the partition function for CRF is quadratic to the number of entity types.

CNN-CNN-LSTM is also 44% faster than CNN-LSTM-LSTM on OntoNotes, showing the advantage of CNN over LSTM as word encoder; on CoNLL-2003, sentences tend to be shorter and this advantage was not clearly seen; its median number of words in sentences is 12 opposed 17 of OntoNotes.

Compared to the CNN-LSTM-CRF model, which is considered as a state-of-the-art model in terms of performance BID6 BID42 , CNN-CNN-LSTM provides four times speedup in terms of the training speed, and achieves comparatively high performance measured by F1 score.

We use OntoNotes-5.0 English and Chinese data BID36 for our experiments.

The training datasets contain 1,088,503 words and 756,063 words respectively.

State-of-the-art models trained on the full training sets achieve F1 scores of 86.

86 Strubell et al. (2017) and 75.63 (our CNN-CNN-LSTM) on the test sets.

We empirically compare selection algorithms proposed in Section 4, as well as uniformly random baseline (RAND).

All algorithms start with an identical 1% of original training data and a randomly initialized model.

In each round, every algorithm chooses sentences from the rest of the training data until 20,000 words have been selected, adding this data to Figure 5: Genre distribution of top 1,000 sentences chosen by an active learning algorithm FIG5 shows the results.

All active learning algorithms perform significantly better than the random baseline.

Among active learners, MNLP and BALD slightly outperformed traditional LC in early rounds.

Note that MNLP is computationally more efficient than BALD, since it only requires a single forward pass on the unlabeled dataset to compute uncertainty scores, whereas BALD requires multiple forward passes.

Impressively, active learning algorithms achieve 99% performance of the best deep model trained on full data using only 24.9% of the training data on the English dataset and 30.1% on Chinese.

Also, 12.0% and 16.9% of training data were enough for deep active learning algorithms to surpass the performance of the shallow models from BID36 trained on the full training data.

We repeated the experiment eight times and confirmed that the trend is replicated across multiple runs; see Appendix B for details.

Detection of under-explored genres To better understand how active learning algorithms choose informative examples, we designed the following experiment.

The OntoNotes datasets consist of six genres: broadcast conversation (bc), braodcast news (bn), magazine genre (mz), newswire (nw), telephone conversation (tc), weblogs (wb).

We created three training datasets: half-data, which contains random 50% of the original training data, nw-data, which contains sentences only from newswire (51.5% of words in the original data), and no-nw-data, which is the complement of nwdata.

Then, we trained CNN-CNN-LSTM model on each dataset.

The model trained on half-data achieved 85.10 F1, significantly outperforming others trained on biased datasets (no-nw-data: 81.49, nw-only-data: 82.08).

This showed the importance of good genre coverage in training data.

Then, we analyzed the genre distribution of 1,000 sentences MNLP chose for each model (see Figure 5 ).

For no-nw-data, the algorithm chose many more newswire (nw) sentences than it did for unbiased half-data (367 vs. 217).

On the other hand, it undersampled newswire sentences for nw-only-data and increased the proportion of broadcast news and telephone conversation, which are genres distant from newswire.

Impressively, although we did not provide the genre of sentences to the algorithm, it was able to automatically detect underexplored genres.

One potential concern when decoding with an LSTM decoder as compared to using a CRF decoder is that finding the best sequence of labels that maximizes the probability P t 2 , t 3 , . . .

, t n???1 | h Enc i is computationally intractable.

In practice, however, we find that simple greedy decoding, i.e., beam search with beam size 1, works surprisingly well.

TAB8 shows how changing the beam size of decoder affects the performance of the model.

It can be seen that the performance of the model changes very little with respect to the beam size.

Beam search with size 2 is marginally better than greedy decoding, and further increasing the beam size did not help at all.

Moreover, we note that while it may be computationally efficient to pick the most likely tag sequence given a CRF encoder, the LSTM decoder may give more accurate predictions, owing to it's greater representational power and ability to model long-range dependencies.

Thus even if we do not always choose the most probable tag sequence from the LSTM, we can still outperform the CRF (as our experiments demonstrate).

In order to understand the variability of learning curves in FIG5 across experiments, we repeated the active learning experiment on OntoNotes-5.0 English eight times, each of which started with different initial dataset chosen randomly.

FIG6 shows the result in first nine rounds of labeled data acquisition.

While MNLP, LC and BALD are all competitive against each other, there is a noticeable trend that MNLP and BALD outperforms LC in early rounds of data acquisition.

Consider that the confidence of the model can help to distinguish between hard and easy samples.

Thus, sampling examples where the model is uncertain might save us from sampling too heavily from regions where the model is already proficient.

But intuitively, when we query a batch of examples in each round, we might want to guard against querying examples that are too similar to each other, thus collecting redundant information.

We also might worry that a purely uncertainty-based approach would oversample outliers.

Thus we explore techniques to guard against these problems by selecting a set of samples that is representative of the dataset.

Following , we express the problem of maximizing representativeness of a labeled set as a submodular optimization problem, and provide an efficient streaming algorithm adapted to use a constraint suitable to the NER task.

We also provide some with theoretical guarantees.

Submodular utility function In order to reason about the similarity between samples, we first embed each sample i into a fixed-dimensional euclidean space as a vector x i .

We consider two embedding methods: 1) the average of pre-trained word embeddings, for p = 1, 2 which corresponds to closeness in L 1 and L 2 distance , and w(i, j) = 1 + xi??xj xi ?? xj , which corresponds to cosine similarity.

Now, we formally define the utility function for labeling new samples.

Denote X as the set of all samples which can be partitioned into two disjoint sets X L , X U representing labeled and unlabeled samples, respectively.

Let S ??? X U be a subset of unlabeled samples, then, the utility of labeling the set is defined as follows: DISPLAYFORM0 where the function measures incremental gain of similarity between the labeled set and the rest.

Given such utility function f (??), choosing a set S that maximizes the function within the budget can be seen as a monotone submodular maximization problem under a knapsack constraint BID21 : max DISPLAYFORM1 where k(S) is the budget for the sample set S, and K is the total budget within each round.

Note that we need to consider the knapsack constraint instead of the cardinality constraint used in the prior work , because the entire sentence needs to be labeled once selected and sequences of length confer different labeling costs.

Combination with uncertainty sampling Representation-based sampling can benefit from uncertainty-based sampling in the following two ways.

First, we can re-weight each sample in the utility function (5) to reflect current model's uncertainty on it: DISPLAYFORM2 Algorithm 1 Representativeness-based Sampling DISPLAYFORM3 while Test score of M less than th do 3:Rank X U according to Sec. 4, X U = top samples S within budget t ?? K.

Set f according to (5) or FORMULA14 5: DISPLAYFORM0 Train M with X L .

where US(i) is the uncertainty score on example i. Second, even with the state-of-the-art submodular optimization algorithms, the optimization problem (6) can be computationally intractable.

To improve the computational efficiency, we restrict the set of unlabeled examples to top samples from uncertainty sampling within budget t ?? K, where t is a multiplication factor we set as 4 in our experiments.

Streaming algorithm for sample selection Even with the reduction of candidates with uncertainty sampling, (6) is still a computationally challenging problem and requires careful design of optimization algorithms.

Suppose l is the number of samples we need to consider.

In the simplistic case in which all the samples have the same length and thus the knapsack constraint degenerates to the cardinality constraint, the greedy algorithm BID33 has an (1 ??? 1/e)-approximation guarantee.

However, it requires calculating the utility function O(l 2 n) times, where n is the number of unlabeled samples.

In practice, both l and n are large.

Alternatively, we can use lazy evaluation to decrease the computation complexity to O(ln) BID26 , but it requires an additional hyperparameter to be chosen in advance.

Instead of greedily selecting elements in an offline fashion, we adopt the two-pass streaming algorithm of BID2 , whose complexity is O(ln) 3 , and generalize it to the knapsack constraint (shown in Alg.

2).

In the first pass, we calculate the maximum function value of a single element normalized by its weight, which gives an estimate of the optimal value.

In the second pass, we create O( 1 log K) buckets and greedily update each of the bucket according to: DISPLAYFORM1 where each bucket has a different value v, and ??? g (e|S v ) := g({e} ??? S v ) ??? g(S v ) is the marginal improvement of submodular function g when adding element e to set S v .

The whole pipeline of the active learning algorithm is shown in Alg.

1.

The algorithm gives the following guarantee, which is proven in Appendix.

Theorem 1.

Alg.

2 gives a(1??? )(1?????) 2 -approximation guarantee for (6), where ?? = max e???S k({e})/K.Proof sketch: The criterion (8) we use guarantees that each update we make is reasonably good.

The set S v stops updating when either the current budget is almost K, or any sample in the stream after we reach S v does not provide enough marginal improvement.

While it is easy to give guarantees when the budget is exhausted, it is unlikely to happen; we use a difference expression between current set S v and the optimal set, and prove the gap between the two is under control.

In a practical label acquisition process, the budget we set for each round is usually much larger than the length of the longest sentence in the unlabeled set, making ?? negligible.

In our experiments, ?? was around 0.01.

<|TLDR|>

@highlight

We introduce a lightweight architecture for named entity recognition and carry out incremental active learning, which is able to match state-of-the-art performance with just 25% of the original training data.