In this paper, we propose a \textit{weak supervision} framework for neural ranking tasks based on the data programming paradigm \citep{Ratner2016}, which enables us to leverage multiple weak supervision signals from different sources.

Empirically, we consider two sources of weak supervision signals, unsupervised ranking functions and semantic feature similarities.

We train a BERT-based passage-ranking model (which achieves new state-of-the-art performances on two benchmark datasets with full supervision) in our weak supervision framework.

Without using ground-truth training labels, BERT-PR models outperform BM25 baseline by a large margin on all three datasets and even beat the previous state-of-the-art results with full supervision on two of datasets.

Recent advances in deep learning have allowed promising improvement in developing various stateof-the-art neural ranking models in the information retrieval (IR) community BID8 BID17 BID12 BID9 BID13 .

Similar achievement has been seen in the reading comprehension (RC) community using neural passage ranking (PR) models for answer selection tasks BID19 BID16 BID10 .

Most of these neural ranking models, however, require a large amount of training data.

As such, we have seen the progress of deep neural ranking models is coming along with the development of several large-scale datasets in both IR and RC communities, e.g. BID0 BID7 BID6 BID3 .

Admittedly, creating hand-labeled ranking datasets is very expensive in both human labor and time.

To overcome this issue, one strategy is to utilize weak supervision to replace human annotators.

Usually we can cheaply obtain large amount of low-quality labels from various sources, such as prior knowledge, domain expertise, human heuristics or even pretrained models.

The idea of weak supervision is to extract signals from the noisy labels to train our model.

BID4 first applied weak supervision technique to train deep neural ranking models.

They show that the neural ranking models trained on labels solely generated from BM25 scores can remarkably outperform the BM25 baseline in IR tasks.

BID11 further investigated this approach by using external news corpus for training.

In this work, we focus on the setting where queries and their associated candidate passages are given but no relevance judgment is available.

Instead of solely relying on the labels from single source (BM25 score), we propose to leverage the weak supervision signals from diverse sources.

BID14 proposed a general data programming framework to create data and train models in a weakly supervised manner.

To tailor to the ranking tasks, instead of generating a ranked list of passages for each query, we generate binary labels for each query-passage pair.

In our neural ranking models, we focus on BERT-based ranking model BID5 (architecture shown in FIG0 ), which achieves new state-of-the-art performance on two public benchmark datasets with full supervision.

The contributions of this work are in two fold: (a) we propose a simple data programming framework for ranking tasks; (b) we train a BERT ranking model using our framework, by considering two simple sources of weak supervision signals, unsupervised ranking methods (BM25 and TF-IDF scores) and unsupervised semantic feature representation, we show our model outperforms BM25 baseline by a large margin (around 20% relative improvement in top-1 accuracy on average) and the previous state-of-the-art performance (around 10% relative improvement in top-1 accuracy on average) on three datasets without using ground-truth training labels.

In this section, we will describe in detail how we train a neural ranking model using weak supervision.

We begin with introducing our BERT-PR model in Section 2.1.

Then in Section 2.2 we will describe the weakly supervised training pipeline.

The goal of a ranking model is to estimate the (relative) relevance of a set of passages {p i } to a given query q.

Here we apply BERT as our scoring model, which measures the relevance score of a candidate passage p i and the query q. Similar to the setup of sentence pair classification task in BID5 , we concatenate the query sentence and the candidate passage together as a single input of the BERT encoder.

We take the final hidden state for the first token ([CLS] word embedding) of the input, and feed it into a two-layer feedforward neural network (with hidden units 100,10 in each layer and ReLU activation).

The final output will be the relevance score between the input query and input passage.

In the supervised setting, we assume we have the ground truth relevance label of the candidate passages for each query in the training set.

To train our BERT ranking model, we use pairwise hinge loss.

Specifically, for each triplet {q, p i , p j }, where q is the query and p i , p j are two candidate passages, the train loss of this instance is DISPLAYFORM0 where Pos(p) indicates the ground truth ranking position of candidate passage p, S is our BERT scoring model as described and is the hyperparameter that determines the margin of hinge loss.

Note that our BERT-PR is different from BID13

Now we present the weak supervision training pipeline for PR tasks.

The main idea follows the paradigm in BID14 , which contains three major steps: (a) defining labeling functions that can generate noisy labels on the datasets (without true labels), (b) aggregating all the noisy labels to generate potentially more accurate labels as well as more coverage, (c) using the aggregated label to train a supervised model.

Ideally, we require a ranked list for each query in our training set for supervised training.

However, obtaining accurate ranking labels for all sets of documents is very difficult.

Instead, we reduce the task to a simpler problem, labeling whether the a candidate passage is strongly related to the query.

With the binary label on the question-passage pair, it is easy to generate triplet training instance by doing positive and negative sampling.

Formally, the labeling function is defined as λ : Q × P → {1, −1, 0}, i.e. for each query-passage pair, we would like to label it as positive, negative or neutral (undetermined).

We first define some score function to measure the similarity of query-passage pair.

Considering that the similarity scores across different queries may not be comparable, we categorize passages based on each individual query.

Specifically, for each query, we rank the candidate passages based on the similarity scores and we take the top-1 passages as positive ones, the bottom half as negative ones, and label the rest in this list as neutral.

With this schema, we obtain {(q i , p DISPLAYFORM0 j ) with y ij ∈ {1, −1, 0}. In this work, we apply 4 scoring functions: (1) BM25 score, (2) TF-IDF score, (3) cosine similarity of universal embedding representation BID1 and (4) cosine similarity of the last hidden layer activation of pretrained BERT model BID5 .Label Aggregation This step is to aggregate all the weak supervision signals from all the labeling functions.

Each label function may produce low quality labels.

The step can be considered as an ensemble step to improve the quality of labels.

We consider two simple strategies.

The first one is through majority voting, i.e., we assign the final label based on the majority agreement, with the majority fraction as the confidence score.

The second strategy is to learn a simple generative model based on the assumption that the labeling functions are conditionally independent given the true label.

We apply the same parameterization as proposed in BID14 ; see details in Appendix A. We predict the final label based on the learned simple generative model.

After label aggregation, we have a collection of query-passage pairs where each is associated with a binary label and confidence score, i.e. {q i , p (i) j , y ij , s ij }

where y ij ∈ {−1, 1} and s ij ∈ [0, 1].

In order to do supervised training, we can generate the triplet training instances by combining positive and negative pairs that share the same query through uniform sampling.

For confidence score of the triplet, we simply take the geometric mean of confidence scores of original two pairs.

Then we train our supervised model based on these labels.

We apply our approaches on three passage-ranking datasets, WikipassageQA BID3 , InsuranceQA v2 BID7 , and MS-MARCO BID0 .

In all these datasets, the groundtruth labels are binary, indicating whether the passage is relevant to the question.

TAB0 shows the basic statistics of these datasets.

In our weak supervision settings, we do not use any ground-truth labels or rank information of the datasets.

In all the experiments, we use pretrained BERT base model from BID5 .

For WikipassageQA dataset, we set the maximum sequence length to be 200 in BERT and batch size 64.

For InsuranceQA v2, we set the maximum sequence length to be 100 in BERT and batch size 128.

For MS-MARCO, we set the maximum sequence length to be 70 in BERT and batch size 256.

For all the training, we sweep over {1e−5, 2e−5, 3e−5} for learning rate and the maximum number of training steps is 10,000.

We use a learning rate warmup ratio of 0.1.

As we described in Section 2, we define four labeling functions.

We adopt the retrieval component in DrQA BID2 for the implementation of BM25 and TF-IDF scoring functions.

We calculate cosine-similarity of BERT features and universal sentence embedding.

To measure the quality of our labeling functions, we apply these labeling function on the training sets and compare our pseudo labels with the ground truth labels.

Note that in ranking datasets, positive and negative pairs are highly imbalanced.

So here we use precision and recall at 1 (P@1, R@1), and AUC to measure the quality of pseudo labels.

The results are shown in TAB1 .

We learn the simple generative model (GM) over labeling functions to estimate the true label.

Also we show the result of majority voting strategy.

The quality of aggregated labels is shown in the bottom rows of TAB1 3.2 PASSAGE RANKING PERFORMANCES After aggregating the results of labeling functions, we now train our BERT-PR model.

We compare the final performances of different models with different supervision signals along with the unsu-Published as a workshop paper at ICLR 2019 pervised BM25 baseline.

We use mean average precision (MAP), mean reciprocal rank (MRR), precision at 1 (P@1) and 5 (P@5) as our evaluation metrics.

The results are shown in TAB2 .Note that through weak supervision solely on BM25 scores, BERT-PR already outperforms the unsupervised BM25 baseline, which is consistent with the results from BID4 .

In our training pipeline, using the simple generative model over the 4 labeling functions, BERT-PR trained on GM labels outperforms BM25 baselines as well as BERT-PR trained solely on BM25 scores.

For example, in terms of P@1, BERT-PR trained on GM labels outperforms BERT-PR trained on BM25 by around 10% relatively on all three datasets.

In the case of WikipassageQA and InsuranceQA datasets, our weak supervision models even beat the previous SOTA performances in the fully supervised settings, exhibiting the great potential of our weak supervision models in real applications.

Also we report the results on supervised training on generated labels with confidence scores, as noise-aware training objective (See Eq. (4) in Appendix B), indicated by "noise" in the parenthesis.

In our experiment, noise-aware training does not improve the performances significantly, probably because using geometric mean of scores of the pairs as the confidence scores of the triplets is not very good approximation of actual probability of generated labels.

We leave this for future research.

In this work, we proposed a simple weak supervision pipeline for neural ranking models based on the data programming paradigm.

In particular, we also proposed a new PR model based on BERT, which achieves new SOTA results.

In our experiments on different datasets, our weakly supervised BERT-PR model outperforms the BM25 baseline by a large margin and remarkably, even beats the previous SOTA performances with full supervision on two datasets.

Further research can be done on how to better aggregate pseudo ranking labels.

In our pipeline we reduce the ranking labels into binary labels of relevance of query-passage pairs, which may result in loss of useful information.

It would be interesting to design generative models on the ranking labels directly.

In this section, we present the simple generate model on labeling functions and true labels as in BID14 .

For completeness, here we restate the formulation as in BID14 .

The basic model assumption is that given the true label, the labeling functions are conditionally independent.

Formally, suppose y ∈ {−1, 1} is the true label, λ 1 , · · · , λ k are the labels from k labeling functions.

The probabilistic graphic model is shown as in FIG2 .

Given the conditional independence assumption, we can parameterize the conditional distribution of the labeling function as follows: We also assume the prior of true label y that P r(y = 1) = γ (In our experiment,for WikipassageQA, we set γ = 0.01, for InsuranceQA v2, we set γ = 0.002, for MS-MARCO, we set γ = 0.001).

Then we can find the optimal parameter α i , β i as maximizing the marginal likelihood of (λ 1 , · · · , λ k ), i.e. where N is the total number of data points.

It is worth mentioning that given this formation, the model is not identifiable due to the symmetry of the model.

A simple solution to remedy this issue is assuming α i > 0.5, meaning the most of labeling functions are doing right.

With that, we can solve the problem Eq. (3) through projected gradient descent methods.

B NOISE-AWARE TRAINING OBJECTIVE BID14 introduces the noise-aware training objective to better incorporate the noise in the generated labels.

The exact objective is as follows: DISPLAYFORM0 where s i ∈ [0, 1] is the confidence score for data point z i being a correct training instance.

For example, in our case of PR, z i is a triplet (q, p + , p − ) and s i is the confidence score of p + being more relevant to q than p − .

@highlight

We propose a weak supervision training pipeline based on the data programming framework for ranking tasks, in which we train a BERT-base ranking model and establish the new SOTA.

@highlight

The authors propose a combination of BERT and the weak supervision framework to tackle the problem of passage ranking, obtaining results better than the fully supervised state-of-the-art.