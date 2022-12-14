Ranking is a central task in machine learning and information retrieval.

In this task, it is especially important to present the user with a slate of items that is appealing as a whole.

This in turn requires taking into account interactions between items, since intuitively, placing an item on the slate affects the decision of which other items should be chosen alongside it.

In this work, we propose a sequence-to-sequence model for ranking called seq2slate.

At each step, the model predicts the next item to place on the slate given the items already chosen.

The recurrent nature of the model allows complex dependencies between items to be captured directly in a flexible and scalable way.

We show how to learn the model end-to-end from weak supervision in the form of easily obtained click-through data.

We further demonstrate the usefulness of our approach in experiments on standard ranking benchmarks as well as in a real-world recommendation system.

Ranking a set of candidate items is a central task in machine learning and information retrieval.

Many existing ranking systems are based on pointwise estimators, where the model assigns a score to each item in a candidate set and the resulting slate is obtained by sorting the list according to item scores ).

Such models are usually trained from click-through data to optimize an appropriate loss function BID17 .

This simple approach is computationally attractive as it only requires a sort operation over the candidate set at test (or serving) time, and can therefore scale to large problems.

On the other hand, in terms of modeling, pointwise rankers cannot easily express dependencies between ranked items.

In particular, the score of an item (e.g., its probability of being clicked) often depends on the other items in the slate and their joint placement.

Such interactions between items can be especially dominant in the common case where display area is limited or when strong position bias is present, so that only a few highly ranked items get the user's attention.

In this case it may be preferable, for example, to present a diverse set of items at the top positions of the slate in order to cover a wider range of user interests.

A significant amount of work on learning-to-rank does consider interactions between ranked items when training the model.

In pairwise approaches a classifier is trained to determine which item should be ranked first within a pair of items (e.g., BID13 BID17 BID6 .

Similarly, in listwise approaches the loss depends on the full permutation of items (e.g., BID7 BID47 .

Although these losses consider inter-item dependencies, the ranking function itself is pointwise, so at inference time the model still assigns a score to each item which does not depend on scores of other items.

There has been some work on trying to capture interactions between items in the ranking scores themselves (e.g., BID29 BID22 BID49 BID32 BID8 .

Such approaches can, for example, encourage a pair of items to appear next to (or far from) each other in the resulting ranking.

Approaches of this type often assume that the relationship between items takes a simple form (e.g., submodular) in order to obtain tractable inference and learning algorithms.

Unfortunately, this comes at the expense of the model's expressive power.

In this paper, we present a general, scalable approach to ranking, which naturally accounts for high-order interactions.

In particular, we apply a sequence-to-sequence (seq2seq) model BID35 to the ranking task, where the input is the list of candidate items and the output is the resulting ordering.

Since the output sequence corresponds to ranked items on the slate, we call this model sequence-to-slate (seq2slate).

The order in which the input is processed can significantly affect the performance of such models BID39 .

For this reason, we often assume the availability of a base (or "production") ranker with which the input sequence is ordered (e.g., a simple pointwise method that ignores the interactions we seek to model), and view the output of our model as a re-ranking of the items.

To address the seq2seq problem, we build on the recent success of recurrent neural networks (RNNs) in a wide range of applications (e.g., BID35 .

This allows us to use a deep model to capture rich dependencies between ranked items, while keeping the computational cost of inference manageable.

More specifically, we use pointer networks, which are seq2seq models with an attention mechanism for pointing at positions in the input BID38 .

We show how to train the network end-to-end to directly optimize several commonly used ranking measures.

To this end, we adapt RNN training to use weak supervision in the form of click-through data obtained from logs, instead of relying on ground-truth rankings, which are much more expensive to obtain.

Finally, we demonstrate the usefulness of the proposed approach in a number of learning-to-rank benchmarks and in a large-scale, real-world recommendeation system.

The ranking problem is that of computing a ranking of a set of items (or ordered list or slate) given some query or context.

We formalize the problem as follows.

Assume a set of n items, each represented by a feature vector x i ??? R m (which may depend on a query or context).

Let ?? ??? ?? denote a permutation of the items, where each ?? j ??? {1, . . .

, n} denotes the index of the item in position j.

Our goal is to predict the output ranking ?? given the input items x. For instance, given a specific user query, we might want to return an ordered set of music recommendations from a set of candidates that maximizes some measure of user engagement (e.g., number of tracks played).

In the seq2seq framework, the probability of an output permutation, or slate, given the inputs is expressed as a product of conditional probabilities according to the chain rule: DISPLAYFORM0 This expression is completely general and does not make any conditional independence assumptions.

In our case, the conditional p(?? j |?? <j , x) ??? ??? n (a point in the n-dimensional simplex) models the probability of any item being placed at the j'th position in the ranking given the items already placed at previous positions.

Therefore, this conditional exactly captures all high-order dependencies between items in the ranked list, including those due to diversity, similarity or other interactions.

Our setting is somewhat different than a standard seq2seq setting in that the output vocabulary is not fixed.

In particular, the same index (position) is populated by different items in different instances (queries).

Indeed, the vocabulary size n itself may vary per instance in the common case where the number of items to rank can change.

This is precisely the problem addressed by pointer networks, which we review next.

We employ the pointer-network architecture of BID38 to model the conditional p(?? j |?? <j , x).

A pointer network uses non-parametric softmax modules, akin to the attention mechanism of BID1 , and learns to point to items in its input sequence rather than predicting an index from a fixed-sized vocabulary.

Our seq2slate model, illustrated in FIG0 , consists of two recurrent neural networks (RNNs): an encoder and a decoder, both of which use Long Short-term Memory (LSTM) cells BID14 .

At each encoding step i ??? n, the encoder RNN reads the input vector x i and outputs a d-dimensional vector e i , thus transforming the input sequence {x i } n i=1 into a sequence of latent memory states {e i } n i=1 .

At each decoding step j, the decoder RNN outputs a d-dimensional vector d j which is used as a query in our attention function.

The attention function takes as input the query d j ??? R d and the set of latent memory states computed by the encoder {e i } n i=1 and produces a probability distribution over the next item to include in the output sequence as follows: DISPLAYFORM0 where W enc , W dec ??? R d??d and v ??? R d are learned parameters in our network, denoted collectively by parameter vector ??.

The probability p DISPLAYFORM1 , is obtained via a softmax over the remaining items and represents the degree to which the model points to input i at decoding step j. To output a permutation, the p j i are set to 0 for items i that already appear in the slate.

Once the next item ?? j is selected, typically greedily or by sampling (see below), its embedding x ??j is fed as input to the next decoder step.

The input of the first decoder step is a learned d-dimensional vector, denoted as go in FIG0 .

Importantly, p ?? (??|x) is differentiable for ant fixed permutation ?? which allows gradient-based learning (see Section 3).

We note the following.

(i) The model makes no explicit assumptions about the type of interactions between items.

If the learned conditional in Eq. (2) is close to the true conditional in Eq. (1), then the model can capture rich interactions-including diversity, similarity or others.

We demonstrate this flexibility in our experiments (Section 4). (ii) x can represent either raw inputs or embeddings thereof, which can be learned together with the sequence model. (iii) The computational cost of inference, dominated by the sequential decoding procedure, is O(n 2 ), which is standard in seq2seq models with attention.

We also consider a computationally cheaper single-step decoder with linear cost O(n), which outputs a single vector p 1 , from which we obtain ?? by sorting the values (similarly to pointwise ranking).

We now turn to the task of training the seq2slate model from data.

A typical approach to learning in ranking systems is to run an existing ranker "in the wild" and log click-through data, which are then used to train an improved ranking model.

This type of training data is relatively inexpensive to obtain, in contrast to human-curated labels such as relevance scores, ratings, or rankings BID17 .

Formally, each training example consists of a sequence of items {x 1 , . . .

, x n } and binary labels (y 1 , . . . , y n ), with y i ??? {0, 1}, representing user feedback (e.g., click/no-click).

Our approach easily extends to more informative feedback, such as the level of user engagement with the chosen item (e.g., time spent), but to simplify the presentation we focus on the binary case.

Our goal is to learn the parameters ?? of p ?? (?? j |?? <j , x) (Eq. (2)) such that permutations ?? corresponding to "good" rankings are assigned high probabilities.

Various performance measures R(??, y) can be used to evaluate the quality of a permutation ?? given the labels y, for example, mean average precision (MAP), precision at k, or normalized discounted cumulative gain at k (NDCG@k).

Generally speaking, permutations where the positive labels rank higher are considered better.

In the standard seq2seq setting, models are trained to maximize the likelihood of a target sequence of tokens given the input, which can be done by maximizing the likelihood of each target token given the previous target tokens using Eq. (1).

During training, the model is typically fed the ground-truth tokens as inputs to the next prediction step, an approach known as teacher forcing BID43 .

Unfortunately, this approach cannot be applied in our setting since we only have access to weak supervision in the form of labels y (e.g clicks), rather than ground-truth permutations.

Instead, we show how the seq2slate model can be trained directly from the labels y.

One potential approach, which has been applied successfully in related tasks BID3 BID48 , is to use reinforcement learning (RL) to directly optimize for the ranking measure R(??, y).

In this setup, the objective is to maximize the expected ranking metric obtained by sequences sampled from our model: E ?????p ?? (.|x) [R(??, y)].

One can use policy gradients and stochastic gradient ascent to optimize ??.

The gradient is formulated using the popular REINFORCE update BID42 and can be approximated via Monte-Carlo sampling as follows: DISPLAYFORM0 where k indexes ranking instances in a batch of size B, ?? k are permutations drawn from the model p ?? and b(x) denotes a baseline function that estimates the expected rewards to reduce the variance of the gradients.

RL, however, is known to be a challenging optimization problem and can suffer from sample inefficiency and difficult credit assignment.

As an alternative, we propose supervised learning using the labels y.

In particular, rather than waiting until the end of the output sequence (as in RL), we wish to give feedback to the model at each decoder step.

Consider the first step, and recall that the model assigns a score s i to each item in the input.

We define a per-step loss (s, y) which essentially acts as a multi-label classification loss with labels y as ground truth.

Two natural, simple choices for are cross-entropy loss and hinge loss: DISPLAYFORM0 hinge (s, y) = max{0, 1 ??? min i:yi=1 DISPLAYFORM1 where?? i = y i / j y j , and p i is a softmax of s, similar to Eq. (2).

Intuitively, with cross-entropy loss we try to assign high probabilities to positive labels (see also BID20 , while hinge loss is minimized when scores of items with positive labels are higher than scores of those with negative labels.

Notice that both losses are convex functions of the scores s. To improve convergence, we consider a smooth version of the hinge-loss where the maximum and minimum are replaced by their smooth counterparts: smooth-max(s; ??) = 1 ?? log i e ??si (and smooth minimum is defined similarly, using min i (s i ) = ??? max i (???s i )).If we simply apply a per-step loss from Eq. (4) to all steps of the output sequence while reusing the labels y at each step, then the loss is invariant to the actual output permutations (e.g., predicting a positive item at the beginning of the sequence has the same cost as predicting it at the end).

Instead, we let the loss at each decoding step j depend on the items already chosen, so no further loss is incurred after a label is predicted correctly.

In particular, for a fixed permutation ??, define the sequence loss: DISPLAYFORM2 where S = {s j } n j=1 , and ??<j (s j , y) depends only on the indices in s j and y which are not in the prefix permutation ?? <j = (?? 1 , . . .

, ?? j???1 ) (see Eq. FORMULA4 ).

Including a per-step weight w j can encourage better performance earlier in the sequence (e.g., w j = 1/ log(j + 1)).

Furthermore, if optimizing for a particular slate size k is desired, one can restrict this loss to just the first k output steps.

Since teacher-forcing is not an option, we resort to feeding the model its own previous predictions, as in ; BID31 .

In this case, the permutation ?? is not fixed, but rather depends on the scores S. Specifically, we consider two policies for producing a permutation during training, sampling and greedy decoding, and introduce their corresponding losses.

The greedy policy consists of selecting the item that maximizes p ?? (??|?? <j , x) at every time step j.

The resulting permutation ?? * then satisfies ?? * j = argmax i p ?? (?? j = i|?? * <j ) and our loss becomes L ?? * .

The greedy policy loss is not continuous everywhere since a small change in the scores s may result in a jump between permutations, and therefore L ?? .

Specifically, the loss is non-differentiable when any s j has multiple maximizing arguments.

Outside this measure-zero subspace, the loss is continuous (almost everywhere), and the gradient is well-defined.

Sampling policy The sampling policy consists of drawing each ?? j from p ?? (??|?? <j , x).

The corresponding loss E[L] = ?? p ?? (??)L ?? (??) is differentiable everywhere since both p ?? (??) and L ?? (??) are differentiable for any permutation ?? (See appendix for a direct derivation of E[L] as a function of S).

In this case, the gradient is formulated as: DISPLAYFORM0 which can be approximated by: DISPLAYFORM1 where b(x k ) is a baseline that approximates L ?? k (??).

Applying stochastic gradient descent intuitively decreases both the loss of any sample (right term) but also the probability of drawing samples with high losses (left term).

Notice that our gradient calculation differs from scheduled sampling which instead computes the loss of the sampled sequences (right term) but ignores the probability of sampling high loss sequences (left term).

We found it helpful to include both terms, which may apply more generally to training of sequence-to-sequence models BID11 .

For both training policies, we minimize the loss via stochastic gradient descent over mini-batches in an end-to-end fashion.

We evaluate the performance of our seq2slate model on a collection of ranking tasks.

In Section 4.1 we use learning-to-rank benchmark data to study the behavior of the model.

We then apply our approach to a large-scale commercial recommendation system and report the results in Section 4.2.

Implementation Details We set hyperparameters of our model to values inspired by the literature.

All experiments use mini-batches of 128 training examples and LSTM cells with 128 hidden units.

We train our models with the Adam optimizer BID19 and an initial learning rate of 0.0003 decayed every 1000 steps by a factor of 0.96.

Network parameters are initialized uniformly at random in [???0.1, 0.1].

To improve generalization, we regularize the model by using dropout with probability of dropping p dropout = 0.1 and L2 regularization with a penalty coefficient ?? = 0.0003.

Unless specified otherwise, all results use supervised training with cross-entropy loss xent and the sampling policy.

At inference time, we report metrics for the greedy policy.

We use an exponential moving average with a decay rate of 0.99 as the baseline b(x) in Eq. FORMULA3 and Eq. (6).

When training the seq2slate model with REINFORCE, we use R = NDGC@10 as the reward function and do not regularize the model.

We also considered a bidirectional encoder RNN BID34 but found that it did not lead to significant improvements in our experiments.

To understand the behavior of the proposed model, we conduct experiments using two learning-torank datasets.

We use two of the largest publicly available benchmarks: the Yahoo Learning to Rank Challenge data (set 1), 1 and the Web30k dataset.

Table 1 : Performance of seq2slate and other baselines on data generated with diverse-clicks.

We adapt the procedure proposed by BID18 to generate click data.

The original procedure is as follows: first, a base ranker is trained from the raw data.

We select this base ranker by training all models in the RankLib package, 3 and selecting the one with the best performance on each data set (MART for Yahoo and LambdaMART for Web30k).

We generate an item ranking using the base model, which is then used to generate training data by simulating a user "cascade" model: a user observes each item with decaying probability 1/i ?? , where i is the base rank of the item and ?? is a parameter of the generative model.

This simulates a noisy sequential scan.

An observed item is clicked if its ground-truth relevance score is above a threshold (relevant: {2, 3, 4}, irrelevant: {0, 1}), otherwise no click is generated.

To introduce high-order interactions, we augment the above procedure as follows, creating a generative process dubbed diverse-clicks.

When observing a relevant item, the user will only click if it is not too similar to previously clicked items (i.e, diverse enough), thus reducing the total number of clicks.

Similarity is defined as being in the smallest q percentile (i.e., q = 0.5 is the median) of Euclidean distances between pairs of feature vectors within the same ranking instance: d ij = x i ??? x j .

We use ?? = 0 (no decay, since clicks are sparse anyway due to the diversity term) and q = 0.5.

This modification to the generative model is essential for our purpose as the original data does not contain explicit inter-item dependencies.

We also discuss variations of this model below.

Using the generated training data, we train both our seq2slate model and baseline rankers from the RankLib package: AdaRank BID46 , Coordinate Ascent BID24 , LambdaMART BID45 , ListNet BID7 , MART BID10 , Random Forests BID5 , RankBoost BID9 , RankNet BID6 .

Some of these baselines use deep neural networks (e.g., RankNet, ListNet), so they are strong state-ofthe-art models with comparable complexity to seq2slate.

The results in Table 1 show that seq2slate significantly outperforms all the baselines, suggesting that it can better capture and exploit the dependencies between items in the data.

To better understand the behavior of the model, we visualize the probabilities of the attention from Eq. (2) for one of the test instances in Fig. 2 .

Interestingly, the model produces slates that are close to the input ranking, but with some items demoted to lower positions, presumably due to the interactions with previous items.

We next consider several variations of the generative model and of the seq2slate model itself.

Results are reported in TAB2 .

The rank-gain metric per example is computed by summing the positions change of all positive labels in the re-ranking, and this is averaged over all examples (queries).

TAB2 , we compare the different training variants outlined in Section 3, namely cross entropy with the greedy or sampling policy, a smooth hinge loss with ?? = 1.0, and REINFORCE.

We find that supervised learning with cross entropy generally performs best, with the smooth hinge loss doing slightly worse.

Our weakly supervised training methods have positive rank gain on all datasets, meaning they improve over the base ranker.

Results from TAB2 in the appendix) suggest that training with REINFORCE yields comparable results on Yahoo but significantly worse results on the more challenging Web30k dataset.

We find no significant difference in performance between relying on the greedy and sampling policies during training.

Table 3 : Performance compared to a competitive base production ranker on real data.

One-step decoding We compare seq2slate to the model which uses a single decoding step, referred to as one-step decoder (see Section 2).

In TAB2 we see that this model has comparable performance to the sequential decoder.

This suggests that when inference time is crucial, as in many real-world systems, one might prefer the faster single-shot option.

One possible explanation for the comparable performance of the one-step decoder is that the interactions in our generated data are rather simple and can be effectively learned by the encoder.

By contrast, in Section 4.2 we show that on more complex real-world data, sequential decoding can perform significantly better.

Sensitivity to input order Previous work suggests that the performance of seq2seq models are often sensitive to the order in which the input is processed BID39 BID26 .

To test this we consider the use of seq2slate without relying on the base ranker to order the input, but instead items are fed to the model in random order.

The results in TAB2 (see shuffled data) show that the performance is indeed significantly worse in this case, which is consistent with previous studies.

It suggests that reranking is an easier task than ranking from scratch.

Adaptivity to the type of interaction To demonstrate the flexibility of seq2slate, we generate data using a variant of the diverse-clicks model above.

In the similar-clicks model, the user also clicks on observed irrelevant items if they are similar to previously clicked items (increasing the number of total clicks).

As above, we use the pairwise distances in feature space d ij to determine similarity.

For this model we use q = 0.5, and ?? = 0.3 for Web30k, ?? = 0.1 for Yahoo, to keep the proportion of positive labels similar.

The results in the appendix (see TAB4 ) show that seq2slate has comparable performance to the baseline rankers, with slightly better performance on the harder Web30k data.

This demonstrates that our model can adapt to various types of interactions in the data.

We also apply seq2slate to a ranking problem from a large-scale commercial recommendation system.

We train the model using massive click-through logs (comprising roughly O(10 7 ) instances) with cross-entropy loss, the greedy policy, L2-regularization and dropout.

The data has item sets of varying size, with an average n of 10.24 items per example.

We learn embeddings of the raw inputs as part of training.

Table 3 shows the performance of seq2slate and the one-step decoder compared to the production base ranker on test data (of roughly the same size as the training data).

Significant gains are observed in all performance metrics, with sequential decoding outperforming the one-step decoder.

This suggests that sequential decoding may more faithfully capture complex dependencies between the items.

Finally, we let the learned seq2slate model run in a live experiment (A/B testing).

We compute the click-through rate (CTR) in each position (#clicks/#examples) for seq2slate.

The production base ranker serves traffic outside the experiment, and we compute CTR per position for this traffic as well.

Fig. 3 shows the difference in CTR per position, indicating that seq2slate has significantly higher CTR in the top positions.

This suggests that seq2slate indeed places items that are likely to be chosen higher in the ranking.

In this section we discuss additional related work.

Our work builds on the recent impressive success of seq2seq models in complex prediction tasks, including machine translation BID35 BID1 , parsing BID37 , combinatorial optimization BID38 BID3 , multi-label classification BID41 BID26 , and others.

Our work differs in that we explicitly target the ranking task, which requires a novel approach to training seq2seq models from weak feedback (click-through data).

Most of the work on ranking mentioned above uses shallow representations.

However, in recent years deep models have been used for information retrieval, focusing on embedding queries, documents and query-document pairs BID15 BID12 BID27 BID40 BID28 ) (see also recent survey by BID25 ).

Rather than embedding individual items, in seq2slate a representation of the entire slate of items is learned and encoded in the RNN state.

Moreover, learning the embeddings (x) can be easily incorporated into the training of the sequence model to optimize both simultaneously end-to-end.

Closest to ours is the recent work of BID0 , where an RNN is used to encode a set of items for re-ranking.

Their approach uses a single decoding step with attention, similar to our one-step decoder.

In contrast, we use sequential decoding, which we find crucial in certain applications (see Section 4.2).

Another important difference is that their training formulation assumes availability of full rankings or relevance scores, while we focus on learning from cheap click-through data.

Finally, Santa BID33 recently proposed an elegant framework for learning permutations based on the so called Sinkhorn operator.

Their approach uses a continuous relaxation of permutation matrices (i.e., the set of doubly-stochastic matrices).

Later, BID23 combined this with a Gumbel softmax distribution to enable efficient learning.

However, this approach is focused on reconstruction of scrambled objects, and it is not obvious how to extend it to our ranking setting, where no ground-truth permutation is available.

We presented a novel seq2slate approach to ranking sets of items.

We found the formalism of pointer-networks particularly suitable for this setting.

We addressed the challenge of training the model from weak user feedback to improve the ranking quality.

Our experiments show that the proposed approach is highly scalable and can deliver significant improvements in ranking results.

Our work can be extended in several directions.

In terms of architecture, we aim to explore the Transformer network BID36 in place of the RNN.

Several variants can potentially improve the performance of our model, including beam-search inference BID44 , and training with Actor-Critic BID2 or SeaRNN BID21 ) and it will be interesting to study their performance in the ranking setting.

Finally, an interesting future work direction will be to study off-policy correction BID16 Since the terms are continuous (and smooth) in S for all j and ?? <j , so is the entire function.

<|TLDR|>

@highlight

A pointer network architecture for re-ranking items, learned from click-through logs.