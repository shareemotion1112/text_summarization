In learning to rank, one is interested in optimising the global ordering of a list of items according to their utility for users.

Popular approaches learn a scoring function that scores items individually (i.e. without the context of other items in the list) by optimising a pointwise, pairwise or listwise loss.

The list is then sorted in the descending order of the scores.

Possible interactions between items present in the same list are taken into account in the training phase at the loss level.

However, during inference, items are scored individually, and possible interactions between them are not considered.

In this paper, we propose a context-aware neural network model that learns item scores by applying a self-attention mechanism.

The relevance of a given item is thus determined in the context of all other items present in the list, both in training and in inference.

Finally, we empirically demonstrate significant performance gains of self-attention based neural architecture over Multi-Layer Perceptron baselines.

This effect is consistent across popular pointwise, pairwise and listwise losses on datasets with both implicit and explicit relevance feedback.

Learning to rank (LTR) is an important area of machine learning research, lying at the core of many information retrieval (IR) systems.

It arises in numerous industrial applications like search engines, recommender systems, question-answering systems, and others.

A typical machine learning solution to the LTR problem involves learning a scoring function, which assigns real-valued scores to each item of a given list, based on a dataset of item features and human-curated or implicit (e.g. clickthrough logs) relevance labels.

Items are then sorted in the descending order of scores [19] .

Performance of the trained scoring function is usually evaluated using an Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page.

Copyrights for third-party components of this work must be honored.

For all other uses, contact the owner/author(s).

The Web Conference, April, 2020, Taipei, Taiwan © 2019 Copyright held by the owner/author(s).

ACM ISBN 978-x-xxxx-xxxx-x/YY/MM.

https://doi.org/10.1145/nnnnnnn.nnnnnnn IR metric like Mean Reciprocal Rank (MRR) [29] , Normalised Discounted Cumulative Gain (NDCG) [16] or Mean Average Precision (MAP) [4] .

In contrast to other classic machine learning problems like classification or regression, the main goal of a ranking algorithm is to determine relative preference among a group of items.

Scoring items individually is a proxy of the actual learning to rank task.

Users' preference for a given item on a list depends on other items present in the same list: an otherwise preferable item might become less relevant in the presence of other, more relevant items.

Common learning to rank algorithms attempt to model such inter-item dependencies at the loss level.

That is, items in a list are still scored individually, but the effect of their interactions on evaluation metrics is accounted for in the loss function, which usually takes a form of a pairwise (RankNet [6] , LambdaLoss [30] ) or a listwise (ListNet [9] , ListMLE [31] ) objective.

For example, in LambdaMART [8] the gradient of the pairwise loss is rescaled by the change in NDCG of the list which would occur if a pair of items was swapped.

Pointwise objectives, on the other hand, do not take such dependencies into account.

In this work, we propose a learnable, context-aware, self-attention [27] based scoring function, which allows for modelling of interitem dependencies not only at the loss level but also in the computation of items' scores.

Self-attention is a mechanism first introduced in the context of natural language processing.

Unlike RNNs [14] , it does not process the input items sequentially but allows the model to attend to different parts of the input regardless of their distance from the currently processed item.

We adapt the Transformer [27] , a popular self-attention based neural machine translation architecture, to the ranking task.

We demonstrate that the obtained ranking model significantly improves performance over Multi-Layer Perceptron (MLP) baselines across a range of pointwise, pairwise and listwise ranking losses.

Evaluation is conducted on MSLR-WEB30K [24] , the benchmark LTR dataset with multi-level relevance judgements, as well as on clickthrough data coming from Allegro.pl, a large-scale e-commerce search engine.

We provide an open-source Pytorch [22] implementation of our self-attentive context-aware ranker available at url_removed.

The rest of the paper is organised as follows.

In Section 2 we review related work.

In Section 3 we formulate the problem solved in this work.

In Section 4 we describe our self-attentive ranking model.

Experimental results and their discussion are presented in Section 5.

In Section 6 we conduct an ablation study of various hyperparameters of our model.

Finally, a summary of our work is given in Section 7.

Learning to rank has been extensively studied and there is a plethora of resources available on classic pointwise, pairwise and listwise approaches.

We refer the reader to [19] for the overview of the most popular methods.

What the majority of LTR methods have in common is that their scoring functions score items individually.

Inter-item dependencies are (if at all) taken into account at the loss level only.

Previous attempts at modelling context of other items in a list in the scoring function include:

• a pairwise scoring function [12] and Groupwise Scoring Function (GSF) [2] , which incorporates the former work as its special case.

However, the proposed GSF method simply concatenates feature vectors of multiple items and passes them through an MLP.

To desensitize the model to the order of concatenated items, Monte-Carlo sampling is used, which yields an unscalable algorithm, • a seq2slate model [5] uses an RNN combined with a variant of Pointer Networks [28] in an encoder-decoder type architecture to both encode items in a context-aware fashion and then produce the optimal list by selecting items one-by-one.

Authors evaluate their approach only on clickthrough data (both real and simulated from WEB30K).

A similar, simpler approach known as Deep Listwise Context Model (DLCM) was proposed in [1] : an RNN is used to encode a set of items for re-ranking, followed by a single decoding step with attention, • in [15] , authors attempt to capture inter-item dependencies by adding so-called delta features which represent how different given item is from items surrounding it in the list.

It can be seen as a simplified version of a local self-attention mechanism.

Authors evaluate their approach on proprietary search logs only, • authors of [17] formulate the problem of re-ranking of a list of items as that of a whole-list generation.

They introduce ListCVAE, a variant of Conditional Variational Auto-Encoder [25] which learns the joint distribution of items in a list conditioned on users' relevance feedback and uses it to directly generate a ranked list of items.

Authors claim NDCG unfairly favours greedy ranking methods and thus do not use that metric in their evaluation, • similarly to our approach, Pei et al. [23] use the self-attention mechanism to model inter-item dependencies.

Their approach, however, was not evaluated on a standard WEB30K dataset and the only loss functions considered was ListNet.

Our proposed solution to the problem of context-aware ranking makes use of the self-attention mechanism.

It was first introduced as intra-attention in [11] and received more attention after the introduction of the Transformer architecture [27] .

Our model can be seen as a special case of the encoder part of the Transformer.

We compare the proposed approach with those of the aforementioned methods which provided an evaluation on WEB30K in terms of NDCG@5.

These include GSF of [2] and DLCM of [1] .

We outperform both competing methods.

In this section, we formulate problem at hand in learning to rank setting.

Let X be the training set.

It consists of pairs (x, y) of a list x of d f -dimensional real-valued vectors x i together with a list y of their relevance labels y i (multi-level or binary).

Note that lists x in the training set may be of varying length.

The goal is to find a scoring function f which maximises an IR metric of choice (e.g. NDCG) on the test set.

Since IR metrics are rank based (thus, nondifferentiable), the scoring function f is trained to minimise the average of a surrogate loss l over the training data.

while controlling for overfitting (e.g. by using dropout [26] in the neural network based scoring function f or adding L 1 or L 2 penalty term [20] to the loss function l ).

Thus, two crucial choices one needs to make when proposing a learning to rank algorithm are that of a scoring function f and loss function l. As discussed earlier, typically, f scores elements x i ∈ x individually to produce scores f (x i ), which are then input to loss function l together with ground truth labels y i .

In subsequent sections, we describe our construction of context-aware scoring function f which is able to model interactions between items x i in a list x. Our model is generic enough to be applicable with any of standard pointwise, pairwise or listwise loss.

We thus experiment with a variety of popular ranking losses l.

In this section, we describe the architecture of our self-attention based ranking model.

We modify the Transformer architecture to work in the ranking setting and obtain a scoring function which, when scoring a single item, takes into account all other items present in the same list.

The key component of our model is the self-attention mechanism introduced in [27] .

The attention mechanism can be described as taking the query vector and pairs of key and value vectors as input and producing a vector output.

The output of the attention mechanism for a given query is a weighted sum of the value vectors, where weights represent how relevant to the query is the key of the corresponding value vector.

Self-attention is a variant of attention in which query, key and value vectors are all the same -in our case, they are vector representations of items in the list.

The goal of the self-attention mechanism is to compute a new, higher-level representation for each item in a list, by taking a weighted sum over all items in a list according to weights representing the relevance of these items to the query item.

There are many ways in which one may compute the relevance of key vectors to query vectors.

We use the variant of selfattention known as Scaled Dot-Product Attention.

Suppose Q is a d model -dimensional matrix representing all items (queries) in the list.

Let K and V be the keys and values matrices, respectively.

Then

As described in [27] , it is beneficial to perform the self-attention operation multiple times and concatenate the outputs.

To avoid growing the size of the resulting output vector, matrices Q, K and V are first linearly projected H times to

Each of H computations of linear projection of Q, K, V , followed by self-attention mechanism is referred to as a single attention head.

Note that each head has its own learnable projection matrices.

The outputs of each head are concatenated and once again linearly projected, usually to the vector space of the same dimension as that of input matrix Q. Similarly to the Transformer, our model also uses multiple attention heads.

Transformer architecture was designed to solve a neural machine translation (NMT) task.

In NMT, the order of input tokens should be taken into account.

Unlike RNNs, self-attention based encoder has no way of discerning the order of input tokens.

Authors of the original Transformer paper proposed to solve the problem by using either fixed or learnable positional encodings.

The ranking problem can be viewed as either ordering a set of (unordered) items or as re-ranking, where the input list has already been sorted according to a weak ranking model.

In the former case, the use of positional encodings is not needed.

In the latter, they may boost model's performance.

We experiment with both ranking and re-ranking settings and when positional encodings are used, we test the fixed encodings variant 1 .

Details can be found in Section 5.

We adapt the Transformer model to the ranking setting as follows.

Items on a list are treated as tokens and item features as input token embeddings.

We denote the length of an input list as l and the number of features as d f .

Each item is first passed through a shared fully connected layer of size d f c .

Next, hidden representations are passed through an encoder part of Transformer architecture with N encoder blocks, H heads and hidden dimension d h .

Recall that an encoder block in the Transformer consists of a multi-head attention layer with a skip-connection [13] to the input, followed by layer normalisation [3] , time-distributed feed-forward layer and another skip connection followed by layer normalisation.

Dropout is applied before performing summation in residual blocks.

Finally, after N encoder blocks, a fully-connected layer shared across all items in the list is used to compute a score for each item.

The model can be seen as an encoder part of the Transformer with extra linear projection on the input.

By using self-attention in the encoder, we ensure that in the computation of a score of a given item, hidden representation of all other items were accounted for.

Obtained scores, together with ground truth labels, can provide input to any ranking loss of choice.

If the loss is a differentiable function of scores (and thus, of model's parameters), one can use SGD to optimise it.

We thus obtain a general, context-aware model for scoring items on a list that can readily be used with any differentiable ranking loss.

Learning to rank datasets come in two flavours: they can have either multi-level or binary relevance labels.

Usually, multi-level relevance 1 We found learnable positional encodings to yield similar results.

labels are human-curated, whereas binary labels are derived from clickthrough logs and are considered implicit feedback.

We evaluate our context-aware ranker on both types of data.

For the first type, we use the popular WEB30K dataset, which consists of more than 30,000 queries together with lists of associated search results.

Every search result is encoded as a 136-dimensional real-valued vector and has associated with it a relevance label on the scale from 0 (irrelevant) to 4 (most relevant).

We standardise the features before inputting them into a learning algorithm.

The dataset comes partitioned into five folds with roughly the same number of queries per fold.

We perform 5-fold cross-validation by training our models on three folds, validating on one and testing on the final fold.

All results reported are averages across five folds together with the standard deviation of results.

Since lists in the dataset are of unequal length, we pad or subsample to equal length for training, but use full length (i.e. pad to maximum length present in the dataset) for validation and testing.

For a dataset with binary labels, we use clickthrough logs of Allegro.pl, a large scale e-commerce search engine.

The search engine already has a ranking model deployed, which is trained using XGBoost [10] with rank:pairwise loss.

We thus treat learning on this dataset as a re-ranking problem and use fixed positional encodings in context-aware scoring functions.

This lets the models leverage items' positions returned by the base ranker.

The search logs consist of 1M lists, each of length at most 60.

Nearly all lists (95%) have only one relevant item with label 1; remaining items were not clicked and are deemed irrelevant (label 0).

Each item in a list is represented by a 45-dimensional, real-valued vector.

We do not perform cross-validation on this set, but we use the usual train, validation and test splits of the data.

To evaluate the performance of the proposed context-aware ranking model, we use several popular ranking losses.

Pointwise losses used are RMSE of predicted scores and ordinal loss [21] (with minor modification to make it suitable for ranking).

For pairwise losses, we use NDCGLoss 2++ (one of the losses of LambdaLoss framework) and its special cases, RankNet and LambdaRank [7] .

Listwise losses used consist of ListNet and ListMLE.

Below, we briefly describe all of the losses used.

For a more thorough treatment, please refer to the original papers.

Throughout, X denotes the training set, x denotes an input list of items, s = f (x) is a vector of scores obtained via the ranking function f and y is the vector of ground truth relevancy labels.

The simplest baseline is a pointwise loss, in which no interaction between items is taken into account.

We use RMSE loss:

In practice, we used sigmoid activation function on the outputs of the scoring function f and rescaled them by multiplying by maximum relevance value (e.g. 4 for WEB30K).

The self-attentive scoring function was modified to return four outputs and each output was passed through a sigmoid activation function.

Thus, each neuron of the output predicts a single relevancy level, but by the reformulation of ground truth, their relative order is maintained, i.e. if, say, label 2 is predicted, label 1 should be predicted as well (although it is not strictly enforced and model is allowed to predict label 2 without predicting label 1).

The final loss value is the mean of binary cross-entropy losses for each relevancy level.

During inference, the outputs of all output neurons are summed to produce the final score of an item.

LambdaLoss, RankNet and LambdaRank.

We used NDCGLoss2++ of [30] , formulated as follows:

where

and H (π |s) is a hard assignment distribution of permutations, i.e.

H (π |s) = 1 and H (π |s) = 0 for all π π whereπ is the permutation in which all items are sorted by decreasing scores s. Fixed parameter µ is set to 10.0.

By removing the exponent in l(s, y) formula we obtain the RankNet loss function, weighing each score pair identically.

Similarly, we may obtain differently weighted RankNet variants by changing the formula in the exponent.

To obtain a LambdaRank formula, replace the exponent with

ListNet loss [9] is given by the following formula:

In binary version, softmax of ground truth y is omitted for singleclick lists and replaced with normalisation by the number of clicks for multiple-click lists.

ListMLE [31] is given by:

where

and y(i) is the index of object which is ranked at position i.

We train both our context-aware ranking models and MLP models on both datasets, using all loss functions discussed in Section 5.2 2 .

We also train XGBoost models with rank:pairwise loss similar to the production model of the e-commerce search engine for both datasets.

Hyperparameters of all models (number of encoder blocks, number of attention heads, dropout, etc.) are tuned on the validation set of Fold 1 for each loss separately.

MLP models are constructed to have a similar number of parameters to context-aware ranking models.

For optimisation of neural network models, we use Adam optimiser [18] with the learning rate tuned separately for each model.

Details of hyperparameters used can be found in Appendix A. In Section 6 we provide an ablation study of the effect of various hyperparameters on the model's performance.

On WEB30K, models' performance is evaluated using NDCG@5 3 , which is the usual metric reported for this dataset.

Results are reported in Table 1 .

On e-commerce search logs, we report a relative percentage increase in NDCG@60 over production XGBoost model, presented in Table 2 .

We observe consistent and significant performance improvement of the proposed self-attention based model over MLP baseline across all types of loss functions considered.

In particular, for ListNet we observe a 7.9% performance improvement over MLP baseline on WEB30K.

Note also that the best performing MLP model is outperformed even by the worst-performing self-attention based model on both datasets.

We thus observe that incorporating context-awareness into the model architecture has a more pronounced effect on the performance of the model than varying the underlying loss function.

Surprisingly, ordinal loss outperforms more established and better-studied losses like ListNet, ListMLE or NDCGLoss 2++ on multi-level relevancy data.

Another surprising finding is a good performance of models trained with RMSE loss, especially as compared to models trained to optimise RankNet and ListMLE.

For comparison with the current state-ofthe-art, we provide results on WEB30K reported in other works in Table 3 .

For models with multiple variants, we cite the best result reported in the original work.

In all tables, boldface is the best value column-wise.

All experiments on WEB30K described above were conducted in the ranking setting -input lists of items were treated as unordered, thus positional encoding was not used.

To verify the effect of positional encoding on the model's performance, we conduct the following experiments on WEB30K.

To avoid information leak, training data 4 is divided into five folds and five XGBoost models are trained, each on four folds.

Each model predicts scores for the remaining fold, and the entire dataset is sorted according to these scores.

Finally, we train the same models 5 as earlier on the sorted dataset, but use fixed positional encoding.

Results are presented in Table 4 .

As expected, the models are able to learn positional information and demonstrates improved performance over the plain ranking setting.

To gauge the effect of various hyperparameters of self-attention based ranker on its performance, we performed the following ablation study.

We trained the context-aware ranker with the ordinal loss on Fold 1 of WEB30K dataset and experimented with a different number N of encoder blocks, H attention heads, length l of longest list used in training, dropout rate p dr op and size d h of hidden dimension.

Results are summarised in Table 5 .

Baseline model (i.e. the best performing context-aware ranker trained with ordinal loss) had the following values of hyperparameters: N = 4, H = 2, l = 240, p dr op = 0.4 and d h = 512.

We observe that a high value of dropout is essential to prevent overfitting.

Even though it is better to use multiple attention heads as opposed to a single attention head, using too many results in performance degradation.

Notice that increasing hidden dimension yields better performance than one reported in Table 1 , however, this comes at a price of a large increase in the number of parameters and thus longer training times.

Finally, stacking multiple encoder blocks increases performance.

However, we did not test the effect of stacking more than 4 encoder blocks due to GPU memory constraints.

In this work, we addressed the problem of constructing a contextaware scoring function for learning to rank.

We adapted the selfattention based Transformer architecture from the neural machine translation literature to propose a new type of scoring function for LTR.

We demonstrated considerable performance gains of proposed neural architecture over MLP baselines across different losses and types of data, both in ranking and re-ranking setting.

These experiments provide strong evidence that the gains are due to the ability of the model to score items simultaneously.

As a result of our empirical study, we observed the strong performance of models trained to optimise ordinal loss function.

Such models outperformed models trained with well-studied losses like LambdaLoss or LambdaMART, which were previously shown to provide tight bounds on IR metrics like NDCG.

On the other hand, we observed the surprisingly poor performance of models trained to optimise RankNet and ListMLE losses.

In future work, we plan to investigate the reasons for both good and poor performance of the aforementioned losses, in particular, the relation between ordinal loss and NDCG.

Above we provide hyperparameters used for all models reported in Table 1 .

Models trained on WEB30K were trained for 100 epochs with the learning rate decayed by 0.1 halfway through the training.

On e-commerce search logs, we trained the models for 10 epochs and decayed the learning rate by 0.1 after 5-th epoch.

The meaning of the columns in Table 6 is as follows: d f c is the dimension of the linear projection done on the input data before passing it to the context-aware ranker, N is the number of encoder blocks, H is the number of attention heads, d h is the hidden dimension used throughout computations in encoder blocks, lr is the learning rate, p drop is the dropout probability and l is the list length (lists of items were either padded or subsampled to that length).

The last column shows the number of learnable parameters of the model.

In Table 7 , Hidden dimensions column gives dimensions of subsequent layers of MLP models.

The remaining columns have the same meaning as in the previous table.

@highlight

Learning to rank using the Transformer architecture.