Modeling hypernymy, such as poodle is-a dog, is an important generalization aid to many NLP tasks, such as entailment, relation extraction, and question answering.

Supervised learning from labeled hypernym sources, such as WordNet, limit the coverage of these models, which can be addressed by learning hypernyms from unlabeled text.

Existing unsupervised methods either do not scale to large vocabularies or yield unacceptably poor accuracy.

This paper introduces {\it distributional inclusion vector embedding (DIVE)}, a simple-to-implement unsupervised method of hypernym discovery via per-word non-negative vector embeddings which preserve the inclusion property of word contexts.

In experimental evaluations more comprehensive than any previous literature of which we are aware---evaluating on 11 datasets using multiple existing as well as newly proposed scoring functions---we find that our method provides up to double the precision of previous unsupervised methods, and the highest average performance, using a much more compact word representation, and yielding many new state-of-the-art results.

In addition, the meaning of each dimension in DIVE is interpretable, which leads to a novel approach on word sense disambiguation as another promising application of DIVE.

Numerous applications benefit from compactly representing context distributions, which assign meaning to objects under the rubric of distributional semantics.

In natural language processing, distributional semantics has long been used to assign meanings to words (that is, to lexemes in the dictionary, not individual instances of word tokens).

The meaning of a word in the distributional sense is often taken to be the set of textual contexts (nearby tokens) in which that word appears, represented as a large sparse bag of words (SBOW).

Without any supervision, word2vec BID22 , among other approaches based on matrix factorization BID20 , successfully compress the SBOW into a much lower dimensional embedding space, increasing the scalability and applicability of the embeddings while preserving (or even improving) the correlation of geometric embedding similarities with human word similarity judgments.

While embedding models have achieved impressive results, context distributions capture more semantic features than just word similarity.

The distributional inclusion hypothesis (DIH) BID49 BID11 BID6 posits that the context set of a word tends to be a subset of the contexts of its hypernyms.

For a concrete example, most adjectives that can be applied to poodle can also be applied to dog, because dog is a hypernym of poodle.

For instance, both can be obedient.

However, the converse is not necessarily true -a dog can be straight-haired but a poodle cannot.

Therefore, dog tends to have a broader context set than poodle.

Many asymmetric scoring functions comparing SBOW based on DIH have been developed for automatic hypernymy detection BID49 BID11 BID38 .Hypernymy detection plays a key role in many challenging NLP tasks, such as textual entailment BID34 , coreference BID32 , relation extraction BID8 and question answering BID13 .

Leveraging the variety of contexts and inclusion properties in context distributions can greatly increase the ability to discover taxonomic structure among words BID38 .

The inability to preserve these features limits the semantic representation power and downstream applicability of some popular existing unsupervised learning approaches such as word2vec.

Several recently proposed methods aim to encode hypernym relations between words in dense embeddings, such as Gaussian embedding BID45 BID0 , order embedding BID44 , H-feature detector BID33 , HyperScore (Nguyen et al., 2017) , dual tensor BID12 , Poincar?? embedding BID28 , and LEAR BID46 .

However, the methods focus on supervised or semi-supervised setting BID44 BID33 BID27 BID12 BID46 , do not learn from raw text BID28 or lack comprehensive experiments on the hypernym detection task BID45 BID0 .Recent studies BID21 BID38 have underscored the difficulty of generalizing supervised hypernymy annotations to unseen pairs -classifiers often effectively memorize prototypical hypernyms ('general' words) and ignore relations between words.

These findings motivate us to develop more accurate and scalable unsupervised embeddings to detect hypernymy and propose several scoring functions to analyze the embeddings from different perspectives.

??? A novel unsupervised low-dimensional embedding method to model inclusion relations among word contexts via performing non-negative matrix factorization (NMF) on a weighted PMI matrix, which can be efficiently optimized using modified skip-grams.??? Several new asymmetric comparison functions to measure inclusion and generality properties and to evaluate different aspects of unsupervised embeddings.??? Extensive experiments on 11 datasets demonstrate the learned embeddings and comparison functions achieve state-of-the-art performances on unsupervised hypernym detection while requiring much less memory and compute than approaches based on the full SBOW.??? A qualitative experiment illustrates DIVE can be used to solve word sense disambiguation, especially when efficiently modeling word senses at multiple granularities is desirable.

The distributional inclusion hypothesis (DIH) suggests that the context set of a hypernym tends to contain the context set of its hyponyms.

That is, when representing a word as the counts of contextual co-occurrences, the count in every dimension of hypernym y tends to be larger than or equal to the corresponding count of its hyponym x: DISPLAYFORM0 where x y means y is a hypernym of x, V is the set of vocabulary, and #(x, c) indicates the number of times that word x and its context word c co-occur in a small window with size |W | in corpus D.Our goal is to produce lower-dimensional embeddings that preserve the inclusion property that the embedding of hypernym y is larger than or equal to the embedding of its hyponym x in every dimension.

Formally, the desirable property can be written as DISPLAYFORM1 where d 0 is number of dimensions in the embedding space.

We add additional non-negativity constraints, i.e. x[i] ??? 0, y[i] ??? 0, ???i, in order to increase the interpretability of the embeddings (the reason will be explained later in this section).

This is a challenging task.

In reality, there are a lot of noise and systematic biases which cause the violation of DIH in Equation (1) (i.e. #(x, c) > #(y, c) for some neighboring word c), but the general trend can be discovered by processing several thousands of neighboring words in SBOW together.

After the compression, the same trend has to be estimated in a much smaller embedding space which discards most of the information in SBOW, so it is not surprising to see most of the unsupervised hypernymy detection studies use SBOW BID38 and the existing unsupervised embeddings like Gaussian embedding have degraded accuracy BID47 .

Popular methods of unsupervised word embedding are usually based on matrix factorization BID20 .

The approaches first compute a co-occurrence statistic between the wth word and the cth context word as the (w, c)th element of the matrix M [w, c].

Next, the matrix M is factorized such that M [w, c] ??? w T c, where w is the low dimension embedding of wth word and c is the cth context embedding.

The statistic in M [w, c] is usually related to pointwise mutual information: P M I(w, c) = log( P (w,c) P (w)??P (c) ), where P (w, c) = #(w,c) |D| , |D| = w???V c???V #(w, c) is number of co-occurrence word pairs in the corpus, P (w) = #(w) |D| , #(w) = c???V #(w, c) is the frequency of the word w times the window size |W |, and similarly for P (c).

For example, M [w, c] could be set as positive PMI (PPMI), max(P M I(w, c), 0), or shifted PMI, P M I(w, c) ??? log(k), like skip-grams with negative sampling (SGNS) BID20 .

Intuitively, since M [w, c] ??? w T c, larger embedding values of w at every dimension seems to imply larger w T c, larger M [w, c], larger P M I(w, c), and thus larger co-occurrence count #(w, c).

However, the derivation has two flaws: (1) c could be negative and (2) lower #(w, c) could still lead to larger P M I(w, c) as long as the #(w) is small enough.

To preserve DIH, we propose a novel word embedding method, distributional inclusion vector embedding (DIVE), which fixes the two flaws by performing non-negative factorization (NMF) on the matrix M , where DISPLAYFORM0 where k is a constant which shifts PMI value like SGNS, Z = |D| |V | is the average word frequency, and |V | is the vocabulary size.

The design encourages the inclusion property in DIVE (i.e. Equation (2)) to be satisfied because the property implies that Equation (1) (DIH) holds if the matrix is reconstructed perfectly.

The derivation is simple: Since context vector c is non-negative, if the embedding of hypernym y is greater than or equal to the embedding of its hyponym x in every dimension, DISPLAYFORM1 and only #(w, c) change with w.

Due to its appealing scalability properties during training time BID20 , we optimize our embedding based on the skip-gram with negative sampling (SGNS) BID22 .

The objective function of SGNS is DISPLAYFORM0 where w ??? R, c ??? R, c N ??? R, k is a constant hyper-parameter indicating the ratio between positive and negative samples.

BID18 prove SGNS is equivalent to factorizing a shifted PMI matrix M , where (w) and applying non-negativity constraints to the embeddings, DIVE can be optimized using the similar objective function: DISPLAYFORM1 DISPLAYFORM2 where w ??? 0, c ??? 0, c N ??? 0, ?? is the logistic sigmoid function, and k is a constant hyper-parameter.

P D is the distribution of negative samples, which we set to be the corpus word frequency distribution in this paper.

Equation FORMULA6 is optimized by ADAM BID15 , a variant of stochastic gradient descent (SGD).

The non-negativity constraint is implemented by projection (i.e., clipping any embedding which crosses the zero boundary after an update).

The optimization process provides an alternative angle to explain how DIVE preserves DIH.

The gradients for the word embedding w is DISPLAYFORM3 Assume hyponym x and hypernym y satisfy DIH in Equation (1) and the embeddings x and y are the same at some point during the gradient ascent.

In the case, the gradients coming from negative sampling (the second term) decrease the same amount of embedding values for both x and y because k is a constant hyper-parameter.

However, the embedding of hypernym y would get higher or equal positive gradients from the first term than x in every dimension because #(x, c) ??? #(y, c).

This means Equation (1) tends to imply Equation (2).

Combining the analysis from the matrix factorization viewpoint, DIH in Equation FORMULA0 is approximately equivalent to the inclusion property in DIVE (i.e. Equation FORMULA1 ).

For a frequent target word, there must be many neighboring words that incidentally appear near the target word without being semantically meaningful, especially when a large context window size is used.

The unrelated context words cause noise in both the word vector and the context vector of DIVE.

We address this issue by filtering out context words c for each target word w when the PMI of the co-occurring words is too small (i.e., log( DISPLAYFORM0 .

That is, we set #(w, c) = 0 in the objective function.

This preprocessing step is similar with computing PPMI in SBOW BID5 , where low PMI co-occurrences are removed from the count-based representation.

After applying the non-negativity constraint, we observe that each dimension roughly corresponds to a topic, as previous findings suggest BID29 BID24 .

This gives rise to a natural and intuitive interpretation of our word embeddings: the word embeddings can be seen as unnormalized probability distributions over topics.

By removing the normalization of the target word frequency in the shifted PMI matrix, specific words have values in few dimensions (topics), while general words appear in more topics and correspondingly have high values in more dimensions, so the concreteness level of two words can be easily compared using the magnitude of their embeddings.

In other words, general words have more diverse context distributions, so we need more dimensions to store the information in order to compress SBOW well BID25 .In FIG0 , we present three mentions of the word core and its surrounding contexts.

These various context words increase the embedding values in different dimensions.

Each dimension of the learned embeddings roughly corresponds to a topic, and the more general or representative words for each topic tend to have the higher value in the corresponding dimension (e.g. words in the second column of the table).

The embedding is able to capture the common contexts where the word core appears.

For example, the context of the first mention is related to the atom topic (dimension id 1) and the electron topic (id 9), while the second and third mention occur in the computer architecture topic (id 2) and education topic (id 11), respectively.

We describe four experiments in Section 4-7.

The first 3 experiments compare DIVE with other unsupervised embeddings and SBOW using different hypernymy scoring functions.

In these experiments, unsupervised approaches refer to the methods that only train on plaintext corpus without using any hypernymy or lexicon annotation.

The last experiment presents qualitative results on word sense disambiguation.

The SBOW and embeddings are tested on 11 datasets.

The first 4 datasets come from the recent review of BID38 : BLESS BID1 , EVALution BID36 , Lenci/Benotto (Benotto, 2015) , and Weeds BID50 .

The next 4 datasets are downloaded from the code repository of the H-feature detector BID33 : Medical (i.e., Levy 2014) , LEDS (also referred to as ENTAILMENT or Baroni 2012) BID3 , TM14 (i.e., Turney 2014) BID43 , and Kotlerman 2010 BID16 .

In addition, the performance on the test set of HyperNet BID40 ) (using the random train/test split), the test set of WordNet BID44 , and all pairs in HyperLex BID47 are also evaluated.

The F1 and accuracy measurements are sometimes very similar even though the quality of prediction varies, so average precision AP@all is adopted as the main evaluation metric.

The HyperLex dataset has a continuous score on each candidate word pair, so we adopt Spearman rank coefficient ?? as suggested by the review study of BID47 .

Any OOV (out-of-vocabulary) word encountered in the testing data is pushed to the bottom of the prediction list (effectively assuming the word pair does not have a hypernym relation).

We use WaCkypedia corpus BID2 ), a 2009 Wikipedia dump, to compute SBOW and train the embedding.

For the datasets without Part of Speech (POS) information (i.e. Medical, LEDS, TM14, Kotlerman 2010, and HyperNet), the training data of SBOW and embeddings are raw text.

For other datasets, we concatenate each token with the Part of Speech (POS) of the token before training the models except the case when we need to match the training setup of another paper.

All words are lower cased.

Stop words and rare words (occurs less than 10 times) are removed during our preprocessing step.

The number of embedding dimensions in DIVE d 0 is set to be 100.

Other hyper-parameters used in the experiments are listed in the supplementary materials.

The hyper-parameters of DIVE were decided based on the performance of HyperNet training set.

To train embeddings more efficiently, we chunk the corpus into subsets/lines of 100 tokens instead of using sentence segmentation.

Preliminary experiments show that this implementation simplification does not hurt the performance.

In the following experiments, we train both SBOW and DIVE on only the first 512,000 lines (51.2 million tokens) because we find this way of training setting provides better performances (for both SBOW and DIVE) than training on the whole WaCkypedia or training on randomly sampled 512,000 lines.

We suspect this is due to the corpus being sorted by the Wikipedia page titles, which makes some categorical words such as animal and mammal occur 3-4 times more frequently in the first 51.2 million tokens than the rest.

The performances of training SBOW PPMI on the whole WaCkypedia is also provided for reference in TAB6 .

If a pair of words has the hypernym relation, the words tend to be similar and the hypernym should be more general than the hyponym.

As in HyperScore (Nguyen et al., 2017), we score the hypernym candidates by multiplying two factors corresponding to these properties.

The C?????S (i.e. the cosine similarity multiply the difference of summation) scoring function is defined as DISPLAYFORM0 where w p is the embedding of hypernym and w q is the embedding of hyponym.

As far as we know, Gaussian embedding (GE) is the only unsupervised embedding method which can capture the asymmetric relations between a hypernym and its hyponyms.

Using the same training and testing setup, we use the code implemented by BID0 1 to train Gaussian embedding on the first 51.2 million tokens and test the embeddings on 11 datasets.

Its hyper-parameters are determined using the same way as DIVE (i.e. maximizing the AP on HyperNet training set).

We compare DIVE with GE 2 in TAB1 , and the performances of random scores and only measuring word similarity using skip-grams are also presented for reference.

As we can see, DIVE is usually significantly better than other baselines.

In Experiment 1, we show that there exists a scoring function (C?????S) which detects hypernymy accurately using the embedding space of DIVE.

Nevertheless, different scoring functions measure different signals in SBOW or embeddings.

Since there are so many scoring functions and datasets available in the domain, we first introduce and test the performances of various scoring functions so as to select the representative ones for a more comprehensive evaluation of DIVE on the hypernymy detection tasks.

We denote the embedding/context vector of the hypernym candidate and the hyponym candidate as w p and w q , respectively.

The SBOW model which represents a word by the frequency of its neighboring words is denoted as SBOW Freq, while the SBOW which uses PPMI of its neighboring words as the features BID5 ) is denoted as SBOW PPMI.

A hypernym tends to be similar to its hyponym, so we measure the cosine similarity between word vectors of the SBOW features BID21 or DIVE.

We refer to the symmetric scoring function as Cosine or C for short in the following tables.

We also train the original skip-grams with 100 dimensions and measure the cosine similarity between the resulting word2vec embeddings.

This scoring function is referred to as Word2vec or W.

The distributional informativeness hypothesis BID35 observes that in many corpora, semantically 'general' words tend to appear more frequently and in more varied contexts.

Thus, BID35 advocate using entropy of context distributions to capture the diversity of context.

We adopt the two variations of the approach proposed by BID38 : SLQS Row and SLQS Sub functions.

We also refer to SLQS Row as ???E because it measures the entropy difference of context distributions.

For SLQS Sub, the number of top context words is fixed as 100.Although effective at measuring diversity, the entropy totally ignores the frequency signal from the corpus.

To leverage the information, we measure the generality of a word by its L1 norm (|w p | 1 ) and L2 norm (||w p || 2 ).

Recall that Equation (2) indicates that the embedding of the hypernym y should have a larger value at every dimension than the embedding of the hyponym x. When the inclusion property holds, |y| 1 = i y[i]

??? i x[i] = |x| 1 and similarly ||y|| 2 ??? ||x|| 2 .

Thus, we propose two scoring functions, difference of vector summation (|w p | 1 ??? |w q | 1 ) and the difference of vector 2-norm (||w p || 2 ??? ||w q || 2 ).

Notice that when applying the difference of vector summations (denoted as ???S) to SBOW Freq, it is equivalent to computing the word frequency difference between the hypernym candidate pair.

The combination of 2 similarity functions (Cosine and Word2vec) and the 3 generality functions (difference of entropy, summation, and 2-norm of vectors) leads to six different scoring functions as shown in TAB2 , and C?????S is the same scoring function we used in Experiment 1.

It should be noted that if we use skip-grams with negative sampling (word2vec) as the similarity measurement (i.e., W ?? ??? {E,S,Q}), the scores are determined by two embedding/feature spaces together (word2vec and DIVE/SBOW).

Several scoring functions are proposed to measure inclusion properties of SBOW based on DIH.

Weeds Precision BID49 and CDE BID7 ) both measure the magnitude of the intersection between feature vectors (|w p ??? w q |).

For example, w p ???

w q is defined by the elementwise minimum in CDE.

Then, both scoring functions divide the intersection by the magnitude of the potential hyponym vector (|w q |).

invCL BID17 ) (A variant of CDE) is also tested.

We choose these 3 functions because they have been shown to detect hypernymy well in a recent study BID38 .

However, it is hard to confirm that their good performances come from the inclusion property between context distributions -it is also possible that the context vectors of more general words have higher chance to overlap with all other words due to their high frequency.

For instance, considering a one dimension feature which stores only the frequency of words, the naive embedding could still have reasonable performance on the CDE function, but the embedding DISPLAYFORM0 where w 0 is a constant which emphasizes the inclusion penalty.

If w 0 = 1 and a = 1, AL 1 is equivalent to L1 distance.

The lower AL 1 distance implies a higher chance of observing the hypernym relation.

We tried w 0 = 5 and w 0 = 20.

w 0 = 20 produces a worse micro-average AP@all on SBOW Freq, SBOW PPMI and DIVE, so we fix w 0 to be 5 in all experiments.

An efficient way to solve the optimization in AL 1 is presented in the supplementary materials.

We show the micro average AP@all on 10 datasets using different hypernymy scoring functions in TAB2 .

We can see the combination functions such as C?????S and W?????S perform the best overall.

Among the unnormalized inclusion based scoring functions, CDE works the best.

AL 1 performs well compared with other functions which remove the frequency signal such as Word2vec, Cosine, and SLQS Row.

The summation is the most robust generality measurement.

In the In TAB4 , DIVE with two of the best scoring functions (C?????S and W?????S) is compared with the previous unsupervised state-of-the-art approaches based on SBOW on different datasets.

There are several reasons which might cause the large performance gaps in some datasets.

In addition to the effectiveness of DIVE, some improvements come from our proposed scoring functions.

The fact that every paper uses a different training corpus also affects the performances.

Furthermore, BID38 select the scoring functions and feature space for the first 4 datasets based on AP@100, which we believe is too sensitive to the hyper-parameter settings of different methods.

To isolate the impact of each factor, we perform a more comprehensive comparison next.

In this experiment, we examine whether DIVE successfully preserves the signals for hypernymy detection tasks, which are measured by the same scoring functions designed for SBOW.

Summation difference (???S) and CDE perform the best among generality and inclusion functions in TAB2 , BID17 , APSyn BID37 , and CDE BID7 ) are selected because they have the best AP@100 in the first 4 datasets BID38 .

Cosine similarity BID21 , balAPinc BID16 in 3 datasets BID43 , SLQS BID35 in HyperNet dataset BID40 , and Freq ratio (FR) BID47 respectively.

AL 1 could be used to examine the inclusion properties after removing the frequency signal.

Therefore, we will present the results using these 3 scoring functions, along with W?????S and C?????S.

In addition to classic representations such as SBOW Freq and SBOW PPMI, we compare distributional inclusion vector embedding (DIVE) with additional 4 baselines in TAB6 .???

SBOW PPMI with additional frequency weighting (PPMI w/ FW).

Specifically, w[c] = max(log( DISPLAYFORM0 ), 0).

This forms the matrix reconstructed by DIVE when k = 1.??? DIVE without the PMI filter (DIVE w/o PMI) ??? NMF on shifted PMI: Non-negative matrix factorization (NMF) on the shifted PMI without frequency weighting for DIVE (DIVE w/o FW).

This is the same as applying the nonnegative constraint on the skip-gram model.

??? K-means (Freq NMF): The method first uses Mini-batch k-means BID39 to cluster words in skip-gram embedding space into 100 topics, and hashes each frequency count in SBOW into the corresponding topic.

If running k-means on skip-grams is viewed as an approximation of clustering the SBOW context vectors, the method can be viewed as a kind of NMF BID9 .

Let the N ?? N context matrix be denoted as M c , where the (i, j)th element stores the count of word j appearing beside word i. K-means hashing creates a N ?? 100 matrix G with orthonormal rows (G T G = I), where the (i, k)th element is 0 if the word i does not belong to cluster k. The orthonormal G is also an approximated solution of a type of NMF (M c ??? F G T ) BID9 .

Hashing context vectors into topic vectors can be written DISPLAYFORM1 In the experiment, we also tried to apply a constant log(k) shifting to SBOW PPMI (i.e. max(P M I ??? log(k), 0)).

We found that the performance degrades as k increases.

Similarly, applying PMI filter to SBOW PPMI (set context feature to be 0 if the value is lower than log(k f )) usually makes the performances worse, especially when k f is large.

Applying PMI filter to SBOW Freq only makes its performances closer to (but still much worse than) SBOW PPMI, so we omit this baseline as well.

In TAB6 , we first confirm the finding of the previous review study of BID38 : there is no single hypernymy scoring function which always outperforms others.

One of the main reasons is that different datasets collect negative samples differently.

This is also why we evaluate our method on many datasets to make sure our conclusions hold in general.

For example, if negative samples come from random word pairs (e.g. WordNet dataset), a symmetric similarity measure is already a pretty good scoring function.

On the other hand, negative samples come from related or similar words in HyperNet, EVALution, Lenci/Benotto, and Weeds, so only computing generality difference leads to the best (or close to the best) performance.

The negative samples in many datasets are composed of both random samples and similar words (such as BLESS), so the combination of similarity and generality difference yields the most stable results.

DIVE performs similar or better on all the scoring functions compared with SBOW consistently across all datasets in TAB6 , while using many fewer dimensions (see TAB8 ).

Its results on combination scoring functions outperform SBOW Freq.

Meanwhile, its results on AL 1 outperform SBOW PPMI.

The fact that combination scoring functions (i.e., W?????S or C?????S) usually outperform generality functions suggests that only memorizing general words is not sufficient.

The best average performance on 4 and 10 datasets are both produced by W?????S on DIVE.SBOW PPMI improves the combination functions from SBOW Freq but sacrifices AP on the inclusion functions.

It generally hurts performance to change the frequency sampling of PPMI (PPMI w/ FW) or compute SBOW PPMI on the whole WaCkypedia (all wiki) instead of the first 51.2 million tokens.

The similar trend can also be seen in TAB7 .

Note that AL 1 completely fails in HyperLex dataset using SBOW PPMI, which suggests that PPMI might not necessarily preserve the distributional inclusion property, even though it can have good performance on combination functions.

Removing the PMI filter from DIVE slightly drops the overall precision while removing frequency weights on shifted PMI (w/o FW) leads to poor performances.

K-means (Freq NMF) produces similar AP compared with SBOW Freq, but has worse AL 1 scores.

Its best AP scores on different datasets are also significantly worse than the best AP of DIVE.

This means that only making word2vec (skip-grams with negative sampling) non-negative or naively accumulating topic distribution in contexts cannot lead to satisfactory embeddings.

In addition to hypernymy detection, BID0 show that the mixture of Gaussian distributions can also be used to discover multiple senses of each word.

In our qualitative experiment, we show that DIVE can achieve the similar goal without fixing the number of senses before training the embedding.

Recall that each dimension roughly corresponds to one topic.

Given a query word, the higher embedding value on a dimension implies higher likelihood to observe the word in the context of the topic.

The embedding of a polysemy would have high values on different groups of topics/dimensions.

This allows us to discover the senses by clustering the topics/dimensions of the polysemy.

We use the embedding values as the feature each dimension, compute the pairwise similarity between dimensions, and apply spectral clustering BID41 to group topics as shown in the TAB9 .

See more implementation details in the supplementary materials.

In the word sense disambiguation tasks, it is usually challenging to determine how many senses/clusters each word should have.

Many existing approaches fix the number of senses before training the embedding BID42 BID0 .

BID26 make the number of clusters approximately proportional to the diversity of the context, but the assumption does not always hold.

Furthermore, the training process cannot capture different granularity of senses.

For instance, race in the car context could share the same sense with the race in the game topic because they all mean contest, but the race in the car context actually refers to the specific contest of speed.

Therefore, they can also be viewed as separate senses (like the results in TAB9 ).

This means the correct number of clusters is not unique, and the methods, which fixes the cluster numbers, need to re-train the embedding many times to capture such granularity.

In our approach, clustering dimensions is done after the training process of DIVE is completed, so it is fairly efficient to change the cluster numbers and hierarchical clustering is also an option.

Similar to our method, BID31 also discover word senses by graph-based clustering.

The main difference is that they cluster the top n words which are most related to the query word instead of topics.

However, choosing the hyper-parameter n is difficult.

Large n would make graph clustering algorithm inefficient, while small n would make less frequent senses difficult to discover.

Most previous unsupervised approaches focus on designing better hypernymy scoring functions for sparse bag of word (SBOW) features.

They are well summarized in the recent study BID38 .

BID38 also evaluate the influence of different contexts, such as changing the window size of contexts or incorporating dependency parsing information, but neglect scalability issues inherent to SBOW methods.

A notable exception is the Gaussian embedding model BID45 .

The context distribution of each word is encoded as a multivariate Gaussian distribution, where the embeddings of hypernyms tend to have higher variance and overlap with the embedding of their hyponyms.

However, since a Gaussian distribution is normalized, it is difficult to retain frequency information during the embedding process, and experiments on HyperLex BID47 demonstrate that a simple baseline only relying on word frequency can achieve good results.

Follow-up work models contexts by a mixture of Gaussians BID0 relaxing the unimodality assumption but achieves little improvement on hypernym detection tasks.

BID14 show that images retrieved by a search engine can be a useful source of information to determine the generality of lexicons, but the resources might not be available for some corpora such as scientific literature.

Order embedding BID44 ) is a supervised approach to encode many annotated hypernym pairs (e.g. all of the whole WordNet BID23 ) into a compact embedding space, where the embedding of a hypernym should be smaller than the embedding of its hyponym in every dimension.

Our method learns embedding from raw text, where a hypernym embedding should be larger than the embedding of its hyponym in every dimension.

Thus, DIVE can be viewed as an unsupervised and reversed form of order embedding.

Other semi-supervised hypernym detection methods aim to generalize from sets of annotated word pairs using raw text corpora.

The goal of HyperScore BID27 is similar to our model: the embedding of a hypernym should be similar to its hyponym but with higher magnitude.

However, their training process relies heavily on annotated hypernym pairs, and the performance drops significantly when reducing the amount of supervision.

In addition to context distributions, previous work also leverages training data to discover useful text pattern indicating is-a relation BID40 BID33 , but it remains challenging to increase recall of hypernym detection because commonsense facts like cat is-a animal might not appear in the corpus.

Non-negative matrix factorization (NMF) has a long history in NLP, for example in the construction of topic models BID29 .

Non-negative sparse embedding (NNSE) BID24 and BID10 indicate that non-negativity can make embeddings more interpretable and improve word similarity evaluations.

The sparse NMF is also shown to be effective in cross-lingual lexical entailment tasks but does not necessarily improve monolingual hypernymy detection BID48 .

In our study, a new type of NMF is proposed, and the comprehensive experimental analysis demonstrates its state-of-the-art performances on unsupervised hypernymy detection.

Compressing unsupervised SBOW models into a compact representation is challenging while preserving the inclusion, generality, and similarity signals which are important for hypernym detection.

Our experiments suggest that simple baselines such as accumulating K-mean clusters and non-negative skip-grams do not lead to satisfactory performances in this task.

To achieve this goal, we proposed an interpretable and scalable embedding method called distributional inclusion vector embedding (DIVE) by performing non-negative matrix factorization (NMF) on a weighted PMI matrix.

We demonstrate that scoring functions which measure inclusion and generality properties in SBOW can also be applied to DIVE to detect hypernymy, and DIVE performs the best on average, slightly better than SBOW while using many fewer dimensions.

Our experiments also indicate that unsupervised scoring functions, which combine similarity and generality measurements, work the best in general, but no one scoring function dominates across all datasets.

A combination of unsupervised DIVE with the proposed scoring functions produces new state-of-the-art performances on many datasets under the unsupervised setup.

Finally, a qualitative experiment shows that clusters of the topics discovered by DIVE often correspond to the word senses, which allow us to do word sense disambiguation without the need to know the number of senses before training the embeddings.

In addition to the unsupervised approach, we also compare DIVE with semi-supervised approaches.

When there are sufficient training data, there is no doubt that the semi-supervised embedding approaches such as HyperNet BID40 , H-feature detector BID33 , and HyperScore (Nguyen et al., 2017) can achieve better performance than all unsupervised methods.

However, in many domains such as scientific literature, there are often not many annotated hypernymy pairs (e.g. Medical dataset ).Since we are comparing an unsupervised method with semi-supervised methods, it is hard to fairly control the experimental setups and tune the hyper-parameters.

In TAB10 , we only show several performances which are copied from the original paper when training data are limited 3 .

As we can see, the performance from DIVE is roughly comparable to the previous semi-supervised approaches trained on small amount of hypernym pairs.

This demonstrates the robustness of our approach and the difficulty of generalizing hypernymy annotations with semi-supervised approaches.

In TAB11 , we show the most general words in DIVE under different queries as constraints.

We also present the accuracy of judging which word is a hypernym (more general) given word pairs with hypernym relations in TAB1 .

The direction is classified correctly if the generality score is greater than 0 (hypernym is indeed predicted as the more general word).

For instance, summation difference (???S) classifies correctly if DISPLAYFORM0 From the table, we can see that the simple summation difference performs better than SQLS Sub, and DIVE predicts directionality as well as SBOW.

Notice that whenever we encounter OOV, the directionality is predicted randomly.

If OOV is excluded, the accuracy of predicting directionality using unsupervised methods can reach around 0.7-0.75.

In HyperNet and WordNet, some hypernym relations are determined between phrases instead of words.

Phrase embeddings are composed by averaging word embeddings or SBOW features.

For WordNet, we assume the Part of Speech (POS) tags of the words are the same as the phrase.

All part-of-speech (POS) tags in the experiments come from NLTK.The window size |W | of SBOW, DIVE, and GE are set as 20 (left 10 words and right 10 words).

For DIVE, the number of epochs is 15, the learning rate is 0.001, the batch size is 128, the threshold in PMI filter k f is set to be 30, and the ratio between negative and positive samples (k) is 1.5.

The hyper-parameters of DIVE were decided based on the performance of HyperNet training set.

The window size of skip-grams (word2vec) is 10.

The number of negative samples (k ) in skip-gram is set as 5.

When composing skip gram into phrase embedding, average embedding is used.

For Gaussian embedding (GE), the number of mixture is 1, the number of dimension is 100, the learning rate is 0.01, the lowest variance is 0.1, the highest variance is 100, the highest Gaussian mean is 10, and other hyper-parameters are the default value in https://github.com/ benathi/word2gm.

The hyper-parameters of GE were also decided based on the performance of HyperNet training set.

When determining the score between two phrases, we use the average score of every pair of tokens in two phrases.

The number of testing pairs N and the number of OOV word pairs is presented in TAB1 .

We use all the default hyper-parameters of the spectral clustering library in Scikit-learn 0.

However, clustering on the global features might group topics together based on the co-occurrence of words which are unrelated to the query words and we want to make the similarity dependent on the query word.

For example, a country topic should be clustered together with a city topic if the query word is place, but it makes more sense to group the country topic with the money topic together if the query word is bank like we did in the word sense disambiguation experiment TAB9 .

This means we want to focus on the geographical meaning of country when the query is related to geography, while focus on the economic meaning of country when the query is about economics.

To create query dependent similarity measurement, we only consider the embedding of words which are related to the query word when preparing the features of dimensions.

Specifically, given a query word q, the feature vector of the ith dimension f (c i , q) is defined as: DISPLAYFORM0 where w q [c j ] is the value of jth dimension of query word embedding, C j (n) is the set of embeddings of top n words in the jth dimension, and the operator ??? means concatenation.

This means instead of considering all the words in the vocabulary, we only take the top n words of every dimension j (n is fixed as 100 in the experiment), weight the feature based on how likely to observe query word in dimension j (w q [c j ]), and concatenate all features together.

That is, when measuring the similarity of dimensions, we only consider the aspects related to query word (e.g. mostly considering words related to facility and money when the query word is bank).After the features of all dimensions are collected, we normalize the feature of each dimension to have the norm 1, compute the pairwise similarity and run the spectral clustering to get the clustering results.

A.5 EFFICIENT WAY TO COMPUTE ASYMMETRIC L1 (AL 1 )Recall that Equation FORMULA10 FIG1 , an simple example is visualized to illustrate the intuition behind the distance function.

By adding slack variables ?? and ??, the problem could be converted into a linear programming problem: dq a*dq dp a*dq-dp dp-a*dq

@highlight

We propose a novel unsupervised word embedding which preserves the inclusion property in the context distribution and achieve state-of-the-art results on unsupervised hypernymy detection