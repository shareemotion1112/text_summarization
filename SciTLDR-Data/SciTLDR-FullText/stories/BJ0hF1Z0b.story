We demonstrate that it is possible to train large recurrent language models with user-level differential privacy guarantees with only a negligible cost in predictive accuracy.

Our work builds on recent advances in the training of deep networks on user-partitioned data and privacy accounting for stochastic gradient descent.

In particular, we add user-level privacy protection to the federated averaging algorithm, which makes large step updates from user-level data.

Our work demonstrates that given a dataset with a sufficiently large number of users (a requirement easily met by even small internet-scale datasets), achieving differential privacy comes at the cost of increased computation, rather than in decreased utility as in most prior work.

We find that our private LSTM language models are quantitatively and qualitatively similar to un-noised models when trained on a large dataset.

Deep recurrent models like long short-term memory (LSTM) recurrent neural networks (RNNs) have become a standard building block in modern approaches to language modeling, with applications in speech recognition, input decoding for mobile keyboards, and language translation.

Because language usage varies widely by problem domain and dataset, training a language model on data from the right distribution is critical.

For example, a model to aid typing on a mobile keyboard is better served by training data typed in mobile apps rather than from scanned books or transcribed utterances.

However, language data can be uniquely privacy sensitive.

In the case of text typed on a mobile phone, this sensitive information might include passwords, text messages, and search queries.

In general, language data may identify a speaker-explicitly by name or implicitly, for example via a rare or unique phrase-and link that speaker to secret or sensitive information.

Ideally, a language model's parameters would encode patterns of language use common to many users without memorizing any individual user's unique input sequences.

However, we know convolutional NNs can memorize arbitrary labelings of the training data BID22 and recurrent language models are also capable of memorizing unique patterns in the training data BID5 .

Recent attacks on neural networks such as those of BID19 underscore the implicit risk.

The main goal of our work is to provide a strong guarantee that the trained model protects the privacy of individuals' data without undue sacrifice in model quality.

We are motivated by the problem of training models for next-word prediction in a mobile keyboard, and use this as a running example.

This problem is well suited to the techniques we introduce, as differential privacy may allow for training on data from the true distribution (actual mobile usage) rather than on proxy data from some other source that would produce inferior models.

However, to facilitate reproducibility and comparison to non-private models, our experiments are conducted on a public dataset as is standard in differential privacy research.

The remainder of this paper is structured around the following contributions:1.

We apply differential privacy to model training using the notion of user-adjacent datasets, leading to formal guarantees of user-level privacy, rather than privacy for single examples.4.

In extensive experiments in ??3, we offer guidelines for parameter tuning when training complex models with differential privacy guarantees.

We show that a small number of experiments can narrow the parameter space into a regime where we pay for privacy not in terms of a loss in utility but in terms of an increased computational cost.

We now introduce a few preliminaries.

Differential privacy (DP) BID10 BID8 BID9 ) provides a well-tested formalization for the release of information derived from private data.

Applied to machine learning, a differentially private training mechanism allows the public release of model parameters with a strong guarantee: adversaries are severely limited in what they can learn about the original training data based on analyzing the parameters, even when they have access to arbitrary side information.

Formally, it says: Definition 1.

Differential Privacy: A randomized mechanism M : D ??? R with a domain D (e.g., possible training datasets) and range R (e.g., all possible trained models) satisfies ( , ??)-differential privacy if for any two adjacent datasets d, d ??? D and for any subset of outputs S ??? R it holds that DISPLAYFORM0 The definition above leaves open the definition of adjacent datasets which will depend on the application.

Most prior work on differentially private machine learning (e.g. BID7 BID4 ; BID0 BID21 BID16 ) deals with example-level privacy: two datasets d and d are defined to be adjacent if d can be formed by adding or removing a single training example from d. We remark that while the recent PATE approach of BID16 can be adapted to give user-level privacy, it is not suited for a language model where the number of classes (possible output words) is large.

For problems like language modeling, protecting individual examples is insufficient-each typed word makes its own contribution to the RNN's training objective, so one user may contribute many thousands of examples to the training data.

A sensitive word or phrase may be typed several times by an individual user, but it should still be protected.2 In this work, we therefore apply the definition of differential privacy to protect whole user histories in the training set.

This user-level privacy is ensured by using an appropriate adjacency relation:Definition 2.

User-adjacent datasets: Let d and d be two datasets of training examples, where each example is associated with a user.

Then, d and d are adjacent if d can be formed by adding or removing all of the examples associated with a single user from d.

Model training that satisfies differential privacy with respect to datasets that are user-adjacent satisfies the intuitive notion of privacy we aim to protect for language modeling: the presence or absence of any specific user's data in the training set has an imperceptible impact on the (distribution over) the parameters of the learned model.

It follows that an adversary looking at the trained model cannot infer whether any specific user's data was used in the training, irrespective of what auxiliary information they may have.

In particular, differential privacy rules out memorization of sensitive information in a strong information theoretic sense.

Our private algorithm relies heavily on two prior works: the FederatedAveraging (or FedAvg) algorithm of BID14 , which trains deep networks on user-partitioned data, and the moments accountant of BID0 , which provides tight composition guarantees for the repeated application of the Gaussian mechanism combined with amplification-via-sampling.

While we have attempted to make the current work as self-contained as possible, the above references provide useful background.

FedAvg was introduced by BID14 for federated learning, where the goal is to train a shared model while leaving the training data on each user's mobile device.

Instead, devices download the current model and compute an update by performing local computation on their dataset.

It is worthwhile to perform extra computation on each user's data to minimize the number of communication rounds required to train a model, due to the significantly limited bandwidth when training data remains decentralized on mobile devices.

We observe, however, that FedAvg is of interest even in the datacenter when DP is applied: larger updates are more resistant to noise, and fewer rounds of training can imply less privacy cost.

Most importantly, the algorithm naturally forms peruser updates based on a single user's data, and these updates are then averaged to compute the final update applied to the shared model on each round.

As we will see, this structure makes it possible to extend the algorithm to provide a user-level differential privacy guarantee.

We also evaluate the FederatedSGD algorithm, essentially large-batch SGD where each minibatch is composed of "microbatches" that include data from a single distinct user.

In some datacenter applications FedSGD might be preferable to FedAvg, since fast networks make it more practical to run more iterations.

However, those additional iterations come at a privacy cost.

Further, the privacy benefits of federated learning are nicely complementary to those of differential privacy, and FedAvg can be applied in the datacenter as well, so we focus on this algorithm while showing that our results also extend to FedSGD.Both FedAvg and FedSGD are iterative procedures, and in both cases we make the following modifications to the non-private versions in order to achieve differential privacy: A) We use random-sized batches where we select users independently with probability q, rather than always selecting a fixed number of users.

B) We enforce clipping of per-user updates so the total update has bounded L 2 norm.

C) We use different estimators for the average update (introduced next).

D) We add Gaussian noise to the final average update.

The pseudocode for DP-FedAvg and DP-FedSGD is given as Algorithm 1.

In the remainder of this section, we introduce estimators for C) and then different clipping strategies for B).

Adding the sampling procedure from A) and noise added in D) allows us to apply the moments accountant to bound the total privacy loss of the algorithm, given in Theorem 1.

Finally, we consider the properties of the moments accountant that make training on large datasets particular attractive.

Bounded-sensitivity estimators for weighted average queries Randomly sampling users (or training examples) by selecting each independently with probability q is crucial for proving low privacy loss through the use of the moments accountant BID0 .

However, this procedure produces variable-sized samples C, and when the quantity to be estimated f (C) is an average rather than a sum (as in computing the weighted average update in FedAvg or the average loss on a minibatch in SGD with example-level DP), this has ramifications for the sensitivity of the query f .

Specifically, we consider weighted databases d where each row k ??? d is associated with a particular user, and has an associated weight w k ??? [0, 1].

This weight captures the desired influence of the , 1 for all users k W = k???d w k for each round t = 0, 1, 2, . . .

do C t ??? (sample users with probability q) DISPLAYFORM0 Algorithm 1: The main loop for DP-FedAvg and DP-FedSGD, the only difference being in the user update function (UserUpdateFedAvg or UserUpdateFedSGD).

The calls on the moments accountant M refer to the API of BID1 .

row on the final outcome.

For example, we might think of row k containing n k different training examples all generated by user k, with weight w k proportional to n k .

We are then interested in a bounded-sensitivity estimate of f (C) = k???C w k ??? k k???C w k for per-user vectors ??? k , for example to estimate the weighted-average user update in FedAvg.

Let W = k w k .

We consider two such estimators: DISPLAYFORM1 Notef f is an unbiased estimator, since E[ k???C w k ] = qW .

On the other hand,f c matches f exactly as long as we have sufficient weight in the sample.

For privacy protection, we need to control the sensitivity of our query functionf , defined as S(f ) = max C,k f (C ??? {k}) ???f (C) 2 , where the added user k can have arbitrary data.

The lower-bound qW min on the denominator off c is necessary to control sensitivity.

Assuming each w k ??? k has bounded norm, we have: Lemma 1.

If for all users k we have w k ??? k 2 ??? S, then the sensitivity of the two estimators is bounded as DISPLAYFORM2 A proof is given in Appendix ??A.Clipping strategies for multi-layer models Unfortunately, when the user vectors ??? k are gradients (or sums of gradients) from a neural network, we will generally have no a priori bound 3 S such that ??? k ??? S. Thus, we will need to "clip" our updates to enforce such a bound before applying f f orf c .

For a single vector ???, we can apply a simple L 2 projection when necessary: and report the value of for which ( , ??)-differential privacy holds after 1 to 10 6 rounds.

For large datasets, additional rounds of training incur only a minimal additional privacy loss.

However, for deep networks it is more natural to treat the parameters of each layer as a separate vector.

The updates to each of these layers could have vastly different L 2 norms, and so it can be preferable to clip each layer separately.

DISPLAYFORM3 Formally, suppose each update DISPLAYFORM4 We consider the following clipping strategies, both of which ensure the total update has norm at most S:1.

Flat clipping Given an overall clipping parameter S, we clip the concatenation of all the layers as ??? k = ??(??? k , S).

2.

Per-layer clipping Given a per-layer clipping parameter S j for each layer, we set DISPLAYFORM5 j .

The simplest model-independent choice is to take DISPLAYFORM6 for all j, which we use in experiments.

We remark here that clipping itself leads to additional bias, and ideally, we would choose the clipping parameter to be large enough that nearly all updates are smaller than the clip value.

On the other hand, a larger S will require more noise in order to achieve privacy, potentially slowing training.

We treat S as a hyper-parameter and tune it.

A privacy guarantee Once the sensitivity of the chosen estimator is bounded, we may add Gaussian noise scaled to this sensitivity to obtain a privacy guarantee.

A simple approach is to use an ( , ??)-DP bound for this Gaussian mechanism, and apply the privacy amplification lemma and the advanced composition theorem to get a bound on the total privacy cost.

We instead use the Moments Accountant of BID0 to achieve much tighter privacy bounds.

The moments accountant for the sampled Gaussian mechanism upper bounds the total privacy cost of T steps of the Gaussian mechanism with noise N (0, ?? 2 ) for ?? = z ?? S, where z is a parameter, S is the sensitivity of the query, and each row is selected with probability q. Given a ?? > 0, the accountant gives an for which this mechanism satisfies ( , ??)-DP.

The following theorem is a slight generalization of the results in BID0 ; see ??A for a proof sketch.

Theorem 1.

For the estimator (f f ,f c ), the moments accountant of the sampled Gaussian mechanism correctly computes the privacy loss with the noise scale of z = ??/S and steps T , where S = S/qW for (f f ) and 2S/qW min for (f c ).Differential privacy for large datasets We use the implementation of the moments accountant from BID1 .

The moments accountant makes strong use of amplification via sampling, which means increasing dataset size makes achieving high levels of privacy significantly easier.

Table 1 summarizes the privacy guarantees offered as we vary some of the key parameters.

The takeaway from this table is that as long as we can afford the cost in utility of adding noise proportional to z times the sensitivity of the updates, we can get reasonable privacy guarantees over a large range of parameters.

The size of the dataset has a modest impact on the privacy cost of a single query (1 round column), but a large effect on the number of queries that can be run without significantly increasing the privacy cost (compare the 10 6 round column).

For example, on a dataset with 10 users, the privacy upper bound is nearly constant between 1 and 10 6 calls to the mechanism (that is, rounds of the optimization algorithm).There is only a small cost in privacy for increasing the expected number of (equally weighted) users C = qW = qK selected on each round as long asC remains a small fraction of the size of the total dataset.

Since the sensitivity of an average query decreases like 1/C (and hence the amount of noise we need to add decreases proportionally), we can increaseC until we arrive at a noise level that does not adversely effect the optimization process.

We show empirically that such a level exists in the experiments.

In this section, we evaluate DP-FedAvg while training an LSTM RNN tuned for language modeling in a mobile keyboard.

We vary noise, clipping, and the number of users per round to develop an intuition of how privacy affects model quality in practice.

We defer our experimental results on FedSGD as well as on models with larger dictionaries to Appendix ??D. To summarize, they show that FedAvg gives better privacy-utility trade-offs than FedSGD, and that our empirical conclusions extend to larger dictionaries with relatively little need for additional parameter tuning despite the significantly larger models.

Some less important plots are deferred to ??C.Model structure The goal of a language model is to predict the next word in a sequence s t from the preceding words s 0 ...s t???1 .

The neural language model architecture used here is a variant of the LSTM recurrent neural network BID13 trained to predict the next word (from a fixed dictionary) given the current word and a state vector passed from the previous time step.

LSTM language models are competitive with traditional n-gram models BID20 and are a standard baseline for a variety of ever more advanced neural language model architectures BID12 BID15 BID11 .

Our model uses a few tricks to decrease the size for deployment on mobile devices (total size is 1.35M parameters), but is otherwise standard.

We evaluate using AccuracyTop1, the probability that the word to which the model assigns highest probability is correct .

Details on the model and evaluation metrics are given in ??B. All training began from a common random initialization, though for real-world applications pre-training on public data is likely preferable (see ??B for additional discussion).Dataset We use a large public dataset of Reddit posts, as described by BID2 .

Critically for our purposes, each post in the database is keyed by an author, so we can group the data by these keys in order to provide user-level privacy.

We preprocessed the dataset to K = 763, 430 users each with 1600 tokens.

Thus, we take w k = 1 for all users, so W = K. We writeC = qK = qW for the expected number of users sampled per round.

See ??B for details on the dataset and preprocessing.

To allow for frequent evaluation, we use a relatively small test set of 75122 tokens formed from random held-out posts.

We evaluate accuracy every 20 rounds and plot metrics smoothed over 5 evaluations (100 rounds).Building towards DP: sampling, estimators, clipping, and noise Recall achieving differential privacy for FedAvg required a number of changes ( ??2, items A-D).

In this section, we examine the impact of each of these changes, both to understand the immediate effects and to enable the selection of reasonable parameters for our final DP experiments.

This sequence of experiments also provides a general road-map for applying differentially private training to new models and datasets.

For these experiments, we use the FedAvg algorithm with a fixed learning rate of 6.0, which we verified was a reasonable choice in preliminary experiments.

4 In all FedAvg experiments, we used a local batch size of B = 8, an unroll size of 10 tokens, and made E = 1 passes over the local dataset; thus FedAvg processes 80 tokens per batch, processing a user's 1600 tokens in 20 batches per round.

First, we investigate the impact of changing the estimator used for the average per-round update, as well as replacing a fixed sample of C = 100 users per round to a variable-sized sample formed by selecting each user with probability q = 100/763430 for an expectation ofC = 100 users.

None of these changes significantly impacted the convergence rate of the algorithm (see Figure 5 in ??C).

In particular, the fixed denominator estimatorf f works just as well as the higher-sensitivity clipped-denominator estimatorf c .

Thus, in the remaining experiments we focus on estimatorf f .

Next, we investigate the impact of flat and per-layer clipping on the convergence rate of FedAvg.

The model has 11 parameter vectors, and for per-layer clipping we simply chose to distribute the clipping budget equally across layers with S j = S/ ??? 11.

Figure 2 shows that choosing S ??? [10, 20] has at most a small effect on convergence rate.

Finally, Figure 3 shows the impact of various levels of per-coordinate Gaussian noise N (0, ?? 2 ) added to the average update.

Early in training, we see almost no loss in convergence for a noise of ?? = 0.024; later in training noise has a larger effect, and we see a small decrease in convergence past ?? = 0.012.

These experiments, where we sample only an expected 100 users per round, are not sufficient to provide a meaningful privacy guarantee.

We have S = 20.0 andC = qW = 100, so the sensitivity of estimatorf f is 20/100.0 = 0.2.

Thus, to use the moments accountant with z = 1, we would need to add noise ?? = 0.2 (dashed red vertical line), which destroys accuracy.

Estimating the accuracy of private models for large datasets Continuing the above example, if instead we choose q soC = 1250, set the L 2 norm bound S = 15.0, then we have sensitivity Table 3 : Count histograms recording how many of a model's (row's) top 10 predictions are found in the n = 10, 50, or 100 most frequent words in the corpus.

Models that predict corpus top-n more frequently have more mass to the right.15/1250 = 0.012, and so we add noise ?? = 0.012 and can apply the moments account with noise scale z = 1.

The computation is now significantly more computationally expensive, but will give a guarantee of (1.97, 10 ???9 )-differential privacy after 3000 rounds of training.

Because running such experiments is so computationally expensive, for experimental purposes it is useful to ask: does using an expected 1250 users per round produce a model with different accuracy than a model trained with only 100 expected users per round?

If the answer is no, we can train a model with C = 100 and a particular noise level ??, and use that model to estimate the utility of a model trained with a much larger q (and hence a much better privacy guarantee).

We can then run the moments accountant (without actually training) to numerically upper bound the privacy loss.

To test this, we trained two models, both with S = 15 and ?? = 0.012, one withC = 100 and one withC = 1250; recall the first model achieves a vacuous privacy guarantee, while the second achieves (1.97, 10 ???9 )-differential privacy after 3000 rounds.

Figure 7 in ??C shows the two models produce almost identical accuracy curves during training.

Using this observation, we can use the accuracy of models trained withC = 100 to estimate the utility of private models trained with much largerC. See also Figure 6 in ??C, which also shows diminishing returns for larger C for the standard FedAvg algorithm.

FIG1 compares the true-average fixed-sample baseline model (see Figure 5 in ??C) with models that use varying levels of clipping S and noise ?? atC = 100.

Using the above approach, we can use these experiments to estimate the utility of LSTMs trained with differential privacy for different sized datasets and different values ofC. TAB2 shows representative values settingC so that z = 1.

For example, the model with ?? = 0.003 and S = 15 is only worse than the baseline by an additive ???0.13% in AccuracyTop1 and achieves (4.6, 10 ???9 )-differential privacy when trained with C = 5000 expected users per round.

As a point of comparison, we have observed that training on a different corpus can cost an additive ???2.50% in AccuracyTop1.

Adjusting noise and clipping as training progresses FIG1 shows that as training progresses, each level of noise eventually becomes detrimental (the line drops somewhat below the baseline).

This suggests using a smaller ?? and correspondingly smaller S (thus fixing z so the privacy cost of each round is unchanged) as training progresses.

FIG3 (and Figure 8 in ??C) shows this can be effective.

We indeed observe that early in training (red), S in the 10 -12.6 range works well (?? = 0.006 -0.0076).

However, if we adjust the clipping/noise tradeoff after 4885 rounds of training and continue for another 6000, switching to S = 7.9 and ?? = 0.0048 performs better.

Comparing DP and non-DP models While noised training with DP-FedAvg has only a small effect on predictive accuracy, it could still have a large qualitative effect on predictions.

We hy-pothesized that noising updates might bias the model away from rarer words (whose embeddings get less frequent actual updates and hence are potentially more influenced by noise) and toward the common "head" words.

To evaluate this hypothesis, we computed predictions on a sample of the test set using a variety of models.

At each s t we intersect the top 10 predictions with the most frequent 10, 50, 100 words in the dictionary.

So for example, an intersection of size two in the top 50 means two of the model's top 10 predictions are in the 50 most common words in the dictionary.

Table 3 gives histograms of these counts.

We find that better models (higher AccuracyTop1) tend to use fewer head words, but see little difference from changingC or the noise ?? (until, that is, enough noise has been added to compromise model quality, at which point the degraded model's bias toward the head matches models of similar quality with less noise).

In this work, we introduced an algorithm for user-level differentially private training of large neural networks, in particular a complex sequence model for next-word prediction.

We empirically evaluated the algorithm on a realistic dataset and demonstrated that such training is possible at a negligible loss in utility, instead paying a cost in additional computation.

Such private training, combined with federated learning (which leaves the sensitive training data on device rather than centralizing it), shows the possibility of training models with significant privacy guarantees for important real world applications.

Much future work remains, for example designing private algorithms that automate and make adaptive the tuning of the clipping/noise tradeoff, and the application to a wider range of model families and architectures, for example GRUs and character-level models.

Our work also highlights the open direction of reducing the computational overhead of differentially private training of non-convex models.

Proof of Lemma 1.

For the first bound, observe the numerator in the estimatorf f can change by at most S between neighboring databases, by assumption.

The denominator is a constant.

For the second bound, the estimatorf c can be thought of as the sum of the vectors w k ??? k divided by max(qW min , k???C ??? k ).

Writing Num(C) for the numerator k???C w k ??? k , and Den(C) for the denominator max(qW min , k???C w k ), the following are immediate for any C and C def = C ??? {k}: DISPLAYFORM0

Here in the last step, we used the fact that f c (C) ??? S. The claim follows.

Proof of Theorem 1.

It suffices to verify that 1.

the moments (of the privacy loss) at each step are correctly bounded; and, 2.

the composability holds when accumulating the moments of multiple steps.

At each step, users are selected randomly with probability q. If in addition the L 2 -norm of each user's update is upper-bounded by S, then the moments can be upper-bounded by that of the sampled Gaussian mechanism with sensitivity 1, noise scale ??/S, and sampling probability q.

Our algorithm, as described in FIG1 , uses a fixed noise variance and generates the i.i.d.

noise independent of the private data.

Hence we can apply the composability as in Theorem 2.1 in BID0 .We obtain the theorem by combining the above and the sensitivity boundsf f andf c .

Model The first step in training a word-level recurrent language model is selecting the vocabulary of words to model, with remaining words mapped to a special "UNK" (unknown) token.

Training a fully differentially private language model from scratch requires a private mechanism to discover which words are frequent across the corpus, for example using techniques like distributed heavyhitter estimation BID6 BID3 .

For this work, we simplified the problem by pre-selecting a dictionary of the most frequent 10,000 words (after normalization) in a large corpus of mixed material from the web and message boards (but not our training or test dataset).Our recurrent language model works as follows: word s t is mapped to an embedding vector e t ??? R

by looking up the word in the model's vocabulary.

The e t is composed with the state emitted by the model in the previous time step s t???1 ??? R 256 to emit a new state vector s t and an "output embedding" o t ??? R 96 .

The details of how the LSTM composes e t and s t???1 can be found in BID13 .

The output embedding is scored against the embedding of each item in the vocabulary via inner product, before being normalized via softmax to compute a probability distribution over the vocabulary.

Like other standard language modeling applications, we treat every input sequence as beginning with an implicit "BOS" (beginning of sequence) token and ending with an implicit "EOS" (end of sequence) token.

Unlike standard LSTM language models, our model uses the same learned embedding for the input tokens and for determining the predicted distribution on output tokens from the softmax.

6 This reduces the size of the model by about 40% for a small decrease in model quality, an advantageous tradeoff for mobile applications.

Another change from many standard LSTM RNN approaches is that we train these models to restrict the word embeddings to have a fixed L 2 norm of 1.0, a modification found in earlier experiments to improve convergence time.

In total the model has 1.35M trainable parameters.

Initialization and personalization For many applications public proxy data is available, e.g., for next-word prediction one could use public domain books, Wikipedia articles, or other web content.

In this case, an initial model trained with standard (non-private) algorithms on the public data (which is likely drawn from the wrong distribution) can then be further refined by continuing with differentially-private training on the private data for the precise problem at hand.

Such pre-training is likely the best approach for practical applications.

However, since training models purely on private data (starting from random initialization) is a strictly harder problem, we focus on this scenario for our experiments.

Our focus is also on training a single model which is shared by all users.

However, we note that our approach is fully compatible with further on-device personalization of these models to the particular data of each user.

It is also possible to give the central model some ability to personalize simply by providing information about the user as a feature vector along with the raw text input.

LSTMs are well-suited to incorporating such additional context.

We evaluate using AccuracyTop1, the probability that the word to which the model assigns highest probability is correct (after some minimal normalization).

We always count it as a mistake if the true next word is not in the dictionary, even if the model predicts UNK, in order to allow fair comparisons of models using different dictionaries.

In our experiments, we found that our model architecture is competitive on AccuracyTop1 and related metrics (Top3, Top5, and perplexity) across a variety of tasks and corpora.

Dataset The Reddit dataset can be accessed through Google BigQuery (Reddit Comments Dataset).

Since our goal is to limit the contribution of any one author to the final model, it is not necessary to include all the data from users with a large number of posts.

On the other hand, processing users with too little data slows experiments (due to constant per-user overhead).

Thus, we use a training set where we have removed all users with fewer than 1600 tokens (words), and truncated the remaining K = 763, 430 users to have exactly 1600 tokens.

We intentionally chose a public dataset for research purposes, but carefully chose one with a structure and contents similar to private datasets that arise in real-world language modeling task such as predicting the next-word in a mobile keyboard.

This allows for reproducibility, comparisons to nonprivate models, and inspection of the data to understand the impact of differential privacy beyond coarse aggregate statistics (as in Table 3 ).

Figure 5 : Comparison of sampling strategies and estimators.

Fixed sample is exactly C = 100 users per round, and variable sample selects uniformly with probability q forC = 100.

The true average corresponds to f , fixed denominator isf f , and clipped denominator isf c .

FIG5 , a smaller value would actually be better when doing private training).

FedSGD is more sensitive to noise than FedAvg, likely because the updates are smaller in magnitude.

Experiments with SGD We ran experiments using FedSGD taking B = 1600, that is, computing the gradient on each user's full local dataset.

To allow more iterations, we usedC = 50 rather than 100.

Examining Figures 9 and 10, we see S = 2 and ?? = 2 ?? 10 ???3 are reasonable values, which suggests for private training we would need in expectation qW = S/?? = 1500 users per round, whereas for FedAvg we might choose S = 15 and ?? = 10 ???2 forC = qW = 1000 users per round.

That is, the relative effect of the ratio of the clipping level to noise is similar between FedAvg and FedSGD.

However, FedSGD takes a significantly larger number of iterations to reach equivalent accuracy.

Fixing z = 1,C = 5000 (the value that produced the best accuracy for a private model in TAB2 ) and total of 763,430 users gives (3.81, 10 ???9 )-DP after 3000 rounds and (8.92, 10 ???9 )-DP after 20000 rounds, so there is indeed a significant cost in privacy to these additional iterations.

Models with larger dictionaries We repeated experiments on the impact of clipping and noise on models with 20000 and 30000 token dictionaries, again using FedAvg training with ?? = 6, equally weighted users with 1600 tokens, andC = 100 expected users per round.

The larger dictionaries give only a modest improvement in accuracy, and do not require changing the clipping and noise parameters despite having significantly more parameters.

Results are given in FIG1 .Other experiments We experimented with adding an explicit L 2 penalty on the model updates (not the full model) on each user, hoping this would decrease the need for clipping by preferring updates with a smaller L 2 norm.

However, we saw no positive effect from this.

@highlight

User-level differential privacy for recurrent neural network language models is possible with a sufficiently large dataset.