In this paper, we design a generic framework for learning a robust text classification model that achieves accuracy comparable to standard full models under test-time budget constraints.

We take a different approach from existing methods and learn to dynamically delete a large fraction of unimportant words by a low-complexity selector such that the high-complexity classifier only needs to process a small fraction of important words.

In addition, we propose a new data aggregation method to train the classifier, allowing it to make accurate predictions even on fragmented sequence of words.

Our end-to-end method achieves state-of-the-art performance while its computational complexity scales linearly with the small fraction of important words in the whole corpus.

Besides, a single deep neural network classifier trained by our framework can be dynamically tuned to different budget levels at inference time.

Recent advances in deep neural networks (DNN) has improved the performance of natural language processing tasks such as document classification, question answering, and sentiment analysis BID29 BID20 BID22 BID31 .

These approaches process the entire text and construct representations of words and phrases in order to perform target tasks.

While these models do realize high accuracy, their computational-time scales linearly with the size of the documents, which can be slow for documents containing many sentences.

In this context, various approaches based on modifying the existing RNN or LSTM architecture have been proposed BID21 ; BID31 to speed-up processing.

However, processing is still fundamentally sequential, which in turn requires loading entire documents to process, limiting compute gains.

We propose a novel test-time prediction method for efficient text classification on long documents that mitigates sequential processing as seen in Fig. 1 .

Our method is a general framework consisting of a selector and a classifier.

The selector performs a coarse one-shot selection deleting unimportant words and choosing important words in the input document.

The collection of fragmented sentences is input into the classifier, which then performs the target task.

The problem is challenging due to competing goals and requires joint training of selector and classifier functions.

First, selector must have negligible overhead while being compatible with the terminal classification task, since uncontrolled word-deletions cannot be handled during classification.

We adopt an architecture that integrates dual embeddings, one based on word-embeddings and the other based on bag-of-words.

Second, the challenge encountered by the classifier is that its input is a sequence of fractured sentences that is incompatible with standard RNN/LSTM inputs and when used without modification leads to significant performance degradation.

One potential solution is to train classifiers with a diverse collection of sentence fragments but this is not meaningful since there are combinatorially many possibilities.

A different approach rooted in so-called "blanking-noise," that randomly blanks out text, leads to marginalized feature distortion BID15 but this also leads to poor accuracy.

This is because DNNs leverage word combinations and word sequences, which the marginalized distortion approach does not account for.

We propose a data aggregation framework (DAG) that augments the training corpus with outputs from selectors at different budget levels.

By training the classifier on the aggregated structured blank-out text, the classifier learns to fuse fragmented sentences into a feature representation that mirrors the representation obtained on full sentences and thus realizes high-accuracy.

We show the effectiveness of the proposed approach through comprehensive experiments on real-world datasets.

Figure 1 : An illustration of the proposed framework.

A selector is designed to select words that are relevant to the target task.

These words are input into the classifier for processing.

We aggregate output text from different selectors and train the classifier on the aggregated data.

Fast Reading Text: Recent works have proposed test-time speed-up for DNNs.

BID29 and BID6 propose CNN based approaches to speed up question answering.

Of particular relevance are LSTM-jump BID31 and skim-RNN BID21 , which are based on modifying existing RNN/LSTM architectures.

LSTM-jump learns to completely skip words deemed to be irrelevant and a variant, skim-RNN, uses a low-complexity LSTM to skim words rather than skipping.

In contrast, we adopt existing classifier architectures but modify the training method.

Interpretability of Neural Networks: Our framework resembles BID11 , who propose to find snippets of input-text to serve as justification (rationales) for text classification.

Their framework also consists of a selector in cascade with a classifier.

However, in their proposed embodiment, both of these modules have similar complexity and in turn, require similar processing times during runtime.

In contrast, our goal is speed-up and we show that a simple selector works as well as a complex one as long as the classifier can account for fragmented text.

Feature Selection in Text Classification:

While text preprocessing such as stop-word removal is conventional, they require pre-defined word lists and are not learned in conjunction with targeted tasks.

Various feature selection approaches BID3 have been discussed in the literature.

The most relevant to ours is to employ lasso BID25 or group lasso BID8 for learning sparse features.

Different from these approaches, we directly learn a selector along with the classifier.

Besides, our selector chooses salient words of an instance (long sentence).

These words serve as input to a classifier (e.g., LSTM).

This is very different from feature subspace selection methods, such as PCA or other dimensionality reduction methods, that map an instance into low dimension space as this representation is not aligned with required LSTM input.

Data Aggregation: Aggregating data or models to improve the performance of a classifier has been studied under various contexts.

Bagging BID2 has been proposed to aggregate models learned from different set of training samples.

Here, we aggregate the output from selector instead of models.

Similar to us the DAGGER algorithm BID19 has been proposed to account for distorted inputs in reinforcement learning and imitation learning.

DAGGER is iterative; at each iteration, it updates its policy by training a classifier in a different reinforcement learning context.

In contrast, our blank-out datasets originate from the given training data and we aggregate these datasets only once, as a means to obtain a rich collection of fragmented sentences.

Budgeted Learning:

The literature on budgeted learning is vast but much of it focuses on a different set of applications and problems than ours (see BID27 ; BID10 ; BID30 ; BID26 ; BID24 ; Weiss & Taskar (2013); BID9 .

Of relevance are methods for speed-up in DNN architectures BID0 ; BID12 ; BID13 ; BID1 .

Different from our method, those methods focus on gating different layers of an existing DNN towards conditional computation.

Generate a blank-out dataset I(X , S b )

Aggregate data: T ← T ∪ I(X , S b ) DISPLAYFORM0 Our goal is to build a robust classifier along with a suite of selectors to achieve good performance under test-time budgets.

Formally, a classifier C(x) takes a sequence of wordsx as input and predicts the corresponding output label y, and a selector S b (x) with test-time budget b takes an input word sequence x = {w 1 , w 2 , . . .

, w N } and generates a binary sequence S b (x) = {z w1 , z w2 , . . . , z w N } where z w k ∈ {0, 1} representing if the corresponding word w k is selected or not.

We denote the sub-sequence of words generated by the selector as I x, S b (x) = {w k : z w k = 1, ∀w k ∈ x}. Our framework aims to train a classifier C and selectors S b such that I x, S b (x) is sufficient to make accurate prediction on the output label (i.e., C I x, S b (x) ≈ C(x)).

Here, the test-time budget b can be viewed as a hyper-parameter of the selector to control the trade-off between test-time speed and accuracy.

Note that in contrast to some existing frameworks (e.g., Yu et al. FORMULA1 ), we build a single classifier for different budgets.

This design choice is due to a practical reason.

The learned parameters of a classifier is often much larger than of a selector (e.g., the number of parameters in one of the classifiers used in our experiment is more than 88 million, while the size of the selector is 300).

As a result, storing different classifiers for different budgets is impractical.

Our learning framework is designed to overcome two main challenges: 1) how to train a classifier C such that it can work with selectors S b with different budget levels and different architectures?

2) How to train a selector without explicit annotations about which words should be selected?

For the former, we propose a data aggregation framework (DAG) to augment blank-out outputs I(x, S b (x)) from different selectors and trained the classifier C on the aggregated data.

For the latter, we train the selectors by leveraging the feedback from task labels.

We discuss details below.

For the ease of discussion, given a set of training data X = {(x 1 , y 1 ), .., (x t , y t ), .., (x m , y m )}, we assume we have a set of selectors S = {S b } with different budget levels.

We will discuss how to obtain these selectors in Section 3.2.

To generate an aggregated corpus, we first apply each selector S b ∈ S on the training set, and generate corresponding blank-out corpus I(X , S b ) = I x t , S b (x t ) , ∀x t ∈ X .

Then, we create an new corpus by aggregating blank-out corpora with different budget level: T = S b ∈S I(X , S b ).

Finally, we train the classifier C T on the aggregated corpus T .

As C T is trained on documents with distortions, it learns to make predictions with different budget levels.

The data aggregation training framework is summarized in Algorithm 1.In the following, we discuss two extensions of the data aggregation framework.

First, the blank-out data can be generated from different classes of selectors with different features or architectures.

In practice, we observed that by aggregating selections from multiple selectors, the trained classifier C T is more robust, leading to higher accuracy.

Second, in the above discussion, we filter out unimportant words by selectors and aggregate the resulting corpora (we call it word-level aggregation (WAG)).

However, the blank-out and selection can be done in phrase or sentence level.

Specifically, if phrase boundaries are provided, we can leverage this information and design a phrase-level aggregation (PAG) to avoid a selector from breaking compound nouns or meaningful phrases (e.g., "New York", "not so bad").

Similarly, for documents consisting of many short sentences, we can enforce the selector to pick the whole sentence if any word in the sentence is selected.

In this way, we can design a sentence-level aggregation (SAG) to better capture long phrases.

A selector in our framework should satisfy the following criteria.

First, as our goal is to reduce overall test time, the selector has to be computationally efficient.

Second, the selected words have to be informative for the classifier to achieve similar performance using the selected words as the original input.

Several existing works (e.g., BID11 ) do not satisfy both conditions.

For example, BID11 proposed a framework to jointly learn a selector with a classifier, where they consider the selector has the same complexity as the classifier as both of components are implemented with RCNN architecture.

As a result, the time complexity of running a RCNN selector is as high as the classifier; therefore, it is not suitable to be used in our framework.

In the following, we consider two classes of selectors: 1) a selector with word embedding features trained jointly with the classifier by a doubly gradient descent method, and 2) a selector trained by a L1-regularized logistic regression with bag-of-words features.

Word Embedding (WE) selector.

To achieve overall speedup gains, we consider a parsimonious word-selector using word embeddings (e.g., GloVe BID18 ) as features to predict if a word should be passed to the classifier.

Intuitively, word embedding preserves the word semantics.

Therefore, for semantic-oriented tasks, word embedding is suitable to identify informative words for predicting target labels.

Formally, for each instance x = (w 1 , w 2 , . . . , w N ), the WE selector outputs a binary vector z, where z w k is associated with word w k .

Let w k ∈ R d be a word vector of word w k , where d is the dimension of the word embedding.

We assume the informative words can be identified independently by word embedding and consider modeling the probability that a word w k is selected by DISPLAYFORM0 where θ S ∈ R n is the model parameters of the selector S b .

Then, the selection of the entire document x = {w 1 , w 2 , . . .

, w N } is DISPLAYFORM1 Because we do not have explicit annotations about which words are important, directly optimizing S b is unfeasible.

Instead, we train the selector S b with a classifier C. We denote the model parameters of the classifier as θ C .

Given a training data (x t , y t ) ∈ X (X is the training set), the classifier C makes predictions based on a word sequences sampled from the selector (i.e., z t ∼ P S b (x t )|x t ).

For classification problems, we minimizing the negative log-likelihood (i.e., cross-entropy loss) l(C, y t , I(x t , z t )) = − log P C (y t ; I(x t , z t )), where P C is the probability distribution over candidate labels predicted by the classifier C. For regression problem, we minimizing the squared loss based on L2 distance: l(C, y t , I(x t , z t )) = y t −C(I(x t , z t )) 2 2 .

As in BID11 , we consider two 1 -regularizers to promote sparsity and continuity of selections, respectively, DISPLAYFORM2 where λ 1 and λ 2 are hyper-parameters (a.k.a.

budget level) and solve the overall objective DISPLAYFORM3 by doubly stochastic gradient descent.

Bag-of-Words selector.

We also consider a traditional approach to use an 1 -regularized linear model BID34 BID17 BID32 with bag-of-word features to identify important words necessary for a target task.

To build intuition, consider binary classification with output labels y ∈ {1, −1}. In the bag-of-word model, for each document x, we construct a feature vector x ∈ {0, 1} |V | , where |V | is the size of the vocabulary.

Each element of the feature vector x w represents if a specific word w appear in the document x. Given a training data set X , the 1 -regularized logistic regression model optimizes DISPLAYFORM4 where θ ∈ R |V | is a weight vector to be learned, θ w corresponds to word w ∈ V , and b is a hyper-parameter (i.e., selection budget).1 Based on the optimal solution θ * , we construct a selector that picks word w if the corresponding θ * w is non-zero.

That is the Bag-of-Words selector output, S b (x) = {δ(θ w = 0) : w ∈ x}, where δ is an indicator function.

In this section, we evaluate the proposed approach on five real-world text classification datasets.

We first compare the proposed approach with existing budget learning methods, then we conduct comprehensive analyses.

Experimental Setup We consider five datasets in the experiments.

The statistics of the datasets are summarized in Table 5 in the appendix.

Stanford Sentiment Treebank (SST-2) BID23 ) is a binary classification problem in sentiment analysis.

The dataset contains annotations of sentiment labels for entire sentences and phrases.

IMDB is described in BID14 .

Each instance in the dataset is a paragraph of a movie review.

Multi-Aspect is collected by BID11 .

For this dataset, we use word embeddings provided with the dataset and apply both RCNN and WE selector to aggregate the data.

To compare our model with other approaches, we follow BID11 to model this problem as a regression problem and use mean square error (MSE) as the evaluation metric.

AGNews:

We collect the dataset from a public repository 2 Zhang et al. (2015) .

Each instance consists of a title and a small paragraph.

Yelp is used in BID7 .

Each instance is a short paragraph of a restaurant review.

For all tasks, we apply the word-level aggregation (WAG).

For datasets consisting of documents with multiple sentences (YELP, IMDB) or have phrase boundary annotations, we also consider sentence-aggregation (SAG) and phrase-aggregation (PAG) schemes.

Our framework is generic and can leverage different types of classifiers.

Therefore, we evaluate our framework with the following two neural network architectures: Biattentive Classification Network (BCN): BCN is a generic text classification model BID16 .

It comprises Bi-LSTM, Bi-attention, and Maxout networks.

BCN provides a strong baseline on many datasets, including SQuAD, SST, IMDB, and several others.

We use the implementation in AllenNLP (https://allennlp.org/).

LSTM: LSTM model is widely used for text classification BID33 BID20 .

LSTM sequentially reads words in a passage and updates its hidden state to capture features from the text.

Both LSTM-jump and Skim-RNN are built upon LSTM.

Besides evaluating our framework with the BCN and the LSTM classifiers, we also analyze the data aggregation framework with Recurrent Convolution Neural Network (RCNN).

RCNN is a refined local n-gram convolutional neural network model.

The recurrent part learns the average features in a dynamic fashion and the convolution part learns the n-gram features that are not necessarily contiguous.

For selectors, we consider both selectors discussed in Sec. 3.2.

To demonstrate the speed and quality of the selectors, we compare them with an RCNN selector used in BID11 .

By default, we use WAG selection scheme and aggregate the data using different WE selectors with budgets (i.e., fraction of text to select) {0.5,0.6,..,1.0} (See Section 3.1, 3.2), Glove BID18 word embeddings) and evaluate in terms of accuracy or error (error = 1 -accuracy) unless stated otherwise.

results with four different parameter settings.

As we do not have access to the performance of their model on dev set, we cannot perform model selections.

Therefore, we report two of the best results shown in their paper.

4) Stop-Words: We filter out stop-words by the list of stop-words provided by NLTK (https://www.nltk.org/).

This approach is widely used as a prepossessing step and is viewed as a naive baseline.

5) Bag-of-Words: Filter words by the Bag-of-Word selector in Sec 3.2 and feed the fragments of sentences to the original classifier.

This approach has been considered in the context of linear models (e.g., BID5 ).We conduct experiments on all datasets except Multi-Aspect, as we do not have performances of LSTM-jump and Skim-RNN on it.

We will use Multi-Aspect to analyze the proposed approach and compare with BID11 .

We evaluate our framework with two widely used text classification models, LSTM, and BCN.

As both Skim-RNN and LSTM-jump are designed specifically for accelerating the LSTM model, we only compare them with our model with LSTM classifier.

Besides both of these models built upon LSTM with slighlty different baseline accuracy.

To make the comaprison fair, we also report the difference in accuracy with respect to each of their baseline model.

The accuracy and speed-up of all the methods are shown in TAB2 .

The results show that our framework achieves competitive performance to both LSTM-jump, and skim-RNN.

In particular, despite skim-RNN performs well in SST-2 and IMDB, it is unstable and is hard to control the tradeoff between performance and the test-time budget.

For example, Skim-RNN-2 is slower than the baseline method with significant accuracy drops.

In contrast, our model is more stable and achieves reasonable performance under different budgets (details will be demonstrated in Sec. 4.2).

Besides, our model allows to naturally incorporate fine-grained annotations in word and phrase levels.

For example, if we leverage the sentiment annotations for phrases in SST-2, our model achieves 86.4 with 1.3x speedup for LSTM and 86.7 with 1.7x speedup for BCN.

Although Stop-words achieves notable speedup, it sometimes comes with a significant performance drop.

This is due to the Stopwords used for filtering text are not learned with the class labels; therefore, some meaningful words (e.g., "but") are filtered out even if they play a very significant role in determining the polarity of the full sentence.

Besides, we are not able to control the budget in the Stop-words approach.

Compared to Bag-of-Words, our framework achieves better performance, highlighting the fact that the issue of classifier incompatibility is real.

By training classifier with the proposed aggregation framework, the model is robust to the distortions and achieves better performance.

Finally, we observed that the classifiers trained with data aggregation improves both the baselines with LSTM and BCN on fulltext.

By aggregating fragments picked by selectors, the model can put more emphasis on important words and be more robust to the noise in the input document.

The performance versus the fraction of selected words on Multi-Aspect and IMDB datasets.

We present results using RCNN and LSTM classifier and varying the sparsity, and coherent hyper-parameters (see TAB4 .2) of the corresponding selector.

(C), (S), and (A) denote classifier, selector and data aggregation scheme.

Results demonstrate that with the data aggregation framework (SAG/WAG), a simple WE selector is competitive with a complex RCNN selector.

In the rest of this section, we provide comprehensive analyses.

To compare with BID11 , we conduct experiments on Multi-Aspect and IMDB, but conclusions are similarly on other datasets.

Performance vs. Selected Words.

FIG2 demonstrates the trade-off between the performance and the fraction of words selected by each setting.

Overall, the error increases when the fraction of the text selected is lower.

On the Multi-Aspect dataset (see FIG2 (a)), the performance of the proposed WE selector is competitive with the complex RCNN selector.

With training the classifier with word-level data aggregation strategy, the model further improves and requires only 12% of selected text to achieves an error rate within 0.1% of full-text.

Similarly, the WE selector and its variant perform well when the classifier is an LSTM model (see FIG2 ).

The mean square error (MSE) of a standard LSTM classifier on Multi-Aspect dataset is 0.01250 and our framework outperforms it achieving MSE 0.01188 with only 28% of text.

On the IMDB data (see FIG2 (c)), WE selector has similar performance trade-off as the RCNN selector and further confirms that a simple selector is sufficient for identifying rationales.

With sentence-level data aggregation, the model performs the best and achieves lower error rate than the baseline RCNN model.

To achieve the same accuracy, our approach needs much smaller fraction of text.

Performance vs. Test Time.

Next, we report the performance versus test running time in FIG4 .

While RCNN selector performs well in identifying important words, its complexity is too high and the overall test-time is 2X higher in all cases (see FIG4 Robust Sentence Representation.

The DNN classifier can be viewed as a representation learner in cascade with a linear classifier (the last softmax layer).

Our data aggregation schema enables the representation learner to be robust to the distortions in the input sentences and effectively estimate the representation of a full sentence when only given its fragments.

To demonstrate this, we output the latent feature vectors produced by the representation learner and estimate the differences between the vectors when full documents and the fragmented documents are inputted.

Results show that, on the AGNews test corpus, the differences in average cosine distances are 0.81 and 0.56 when using the original classifier and the classifier trained with DAG, respectively.

This confirms the proposed approach has an effect of extrapolating to features obtained with full-text even when many words are deleted.

Qualitative Analysis.

One advantage of the proposed framework is that the output of the selector is interpretable.

In Table 2 , we present three examples from the AGNews dataset.

Results demonstrate that our framework correctly identifies words such as "Nokia", "nuclear", "plant", "Shane Warne", "software" and phrases such as "searched by police", "takes six but India established handy lead" as important to the document classification task.

It also learns to filter out words (e.g., "Aug.", "products", "users") that are less predictive to the classification labels.

Results demonstrate that RCNN selector is significantly slower than WE due to its high complexity.

The data aggregation framework (SAG/WAG) achieves better performance given the same test-time budget.

World News Japanese nuclear plant searched .

Kansai Electric Power #39;s nuclear power plant in Fukui, Japan, was searched by police Saturday during an investigation into an Aug. 9 mishap.

Warne takes six but India establish handy lead (Reuters) .

Reuters-World test wicket record holder Shane Warne grabbed six wickets as India established a handy 141-run first innings lead in the second test on Saturday.

Handset Makers Raising Virus Defenses (Reuters) .

Reuters -Software security companies and handset makers, including Finland's Nokia (NOK1V.HE), are gearing up to launch products intended to secure cell phones from variants of the Internet viruses that have become a scourge for personal computer users.

Table 2 : Examples of the WE selector output on AGNews.

Bold words are selected by the selector, while the remainder are filtered out.

Although words like "during an" seem unimportant, appearing in phrases like "bomb exploded during an Independence Day parade" (World-News) and "undefeated during an entire season" (Sports-News), provide a hint to understand the sentences.

Latency Analysis.

In contrast to skim-RNN and LSTM-Jump that sequentially visit the words in a passage.

Our model design allows the WE, and Bag-of-Words selectors to process words in a passage in parallel.

In practice, as the computation involved in our proposed selectors is simple, the running time of the selector can be negligible.

For example, the WE selector takes overall only 14s seconds to identify important words on the Yelp dataset, and the LSTM models take up to 316.5 seconds to process the selected words.

The benefit is more obvious when the text classification model is employed in a cloud computing setting.

The local devices (e.g., smart watches or mobile phones) do not have sufficient memory and computational power to execute a complex classifier.

Therefore, the test instance has to be sent to a cloud server and classified by the model on the cloud.

In this setting, our approach can employ the selector in the local device, and send only important words to the cloud server.

In contrast, skim-RNN and LSTM-jump, which process the text in a sequential nature must either send the entire text to the server or require multiple rounds of communication between the server and local devices.

In either case, the network latency and bandwidth may restrict the speed of the classification framework.

For WE, and Bag-of-Words selector, selection depends only on the embedding, and the unigram word itself respectively.

Instead, we can cache the predictions and store only a list of important words to save memory.

We proposed a budgeted learning framework for learning a robust classifier under test-time budget constraints.

We demonstrated that training classifiers with data aggregation work well with low-complexity selectors based on word-embedding or bag-of-word model and achieve good performance with fragmented input.

The future work includes applying the proposed framework to other text reading tasks and improving the data aggregation strategy by applying learning to search approaches BID4 .

We report the statistics of the datasets in TAB4 .

<|TLDR|>

@highlight

Modular framework for document classification and data aggregation technique for making the framework robust to various distortion, and noise and focus only on the important words. 

@highlight

The authors consider training a RNN-based text classification where there is a resource restriction on test-time prediction, and provide an approach using a masking mechanism to reduce words/phrases/sentences used in prediction followed by a classifier to handle those components.