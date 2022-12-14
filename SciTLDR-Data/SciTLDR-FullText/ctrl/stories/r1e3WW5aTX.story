Identifying the relations that connect words is an important step towards understanding human languages and is useful for various NLP tasks such as knowledge base completion and analogical reasoning.

Simple unsupervised operators such as vector offset between two-word embeddings have shown to recover some specific relationships between those words, if any.

Despite this, how to accurately learn generic relation representations from word representations remains unclear.

We model relation representation as a supervised learning problem and learn parametrised operators that map pre-trained word embeddings to relation representations.

We propose a method for learning relation representations using a feed-forward neural network that performs relation prediction.

Our evaluations on two benchmark datasets reveal that the penultimate layer of the trained neural network-based relational predictor acts as a good representation for the relations between words.

Different types of relations exist between words in a language such as Hypernym, Meronym, Synonym, etc.

Representing relations between words is important for various NLP tasks such as questions answering BID43 , knowledge base completion BID35 and relational information retrieval BID7 .Two main approaches have been proposed in the literature to represent relations between words.

In the first approach, a pair of words is represented by a vector derived from a statistical analysis of a text corpus BID39 .

In a text corpus, a relationship between two words X and Y can be expressed using lexical patterns containing X and Y as slot variables.

For example, "X is a Y" or "Y such as X" indicate that Y is a Hypernym of X BID34 .

The elements of the vector representing the relation between two words correspond to the number of times those two words co-occur with a particular pattern in a corpus.

Given such a relation representation, the relational similarity between the relations that exist between the two words in two word-pairs can be measured by the cosine of the angle between the corresponding vectors.

We call this the holistic approach because a pair of words is treated as a whole rather than the two constituent words separately when creating a relation representation .

Sparsity is a well-known problem for the holistic approach as two words have to co-occur enough in a corpus, or else no relation can be represented for rare or unseen word-pairs.

In contrast, the second approach for relation representation directly computes a relation representation from pre-trained word representations (i.e. word embeddings) using some relational operators.

Prediction-based word embedding learning methods BID27 BID24 represent the meaning of individual words by dense, low-dimensional real-valued vectors by optimising different language modelling objectives.

Although no explicit information is provided to the word embedding learning algorithms regarding the semantic relations that exist among words, prior work BID25 has shown that the learnt word embeddings encode remarkable structural properties pertaining to semantic relations.

They showed that the difference (vector offset) between two word vectors (here-onwards denoted by PairDiff) is an accurate method for solving analogical questions in the form "a is to b as c is to ?".

For example, king ??? man + woman results in a vector that is closest to the queen vector.

We call this approach compositional because the way in which the relation representation is composed by applying some linear algebraic relational operator on the the semantic representations of the the words that participate in a relation.

This interesting property of word embeddings sparked a renewed interest in methods that compose relation representations using word embeddings and besides PairDiff, several other unsupervised methods have been proposed such as 3CosAdd and 3CosMult BID19 .Despite the initial hype, recently, multiple independent works have raised concerns about of word embeddings capturing relational structural properties BID22 BID32 BID23 BID29 .

Although PairDiff performs well on the Google analogy dataset, its performance for other relation types has been poor BID3 BID41 BID18 .

BID41 tested for the generalisation ability of PairDiff using different relation types and found that semantic relations are captured less accurately compared to syntactic relations.

Likewise, BID18 showed that word embeddings are unable to detect paradigmatic relations such as Hypernym, Synonym and Antonyms.

Methods such as PairDiff are biased towards attributional similarities between individual words than relational similarities and fails in the presence of nearest neighbours.

We further discuss various limitations of the existing unsupervised relation composition methods in Section 2.2.Considering the above-mentioned limitations of the unsupervised relation composition methods, a natural question that arises is whether it is possible to learn supervised relation composition methods to overcome those limitations.

In this paper, we model relation representation as learning a parametrised operator f (a, b; ??) such that we can accurately represent the relation between two given words a and b from their word representations a and b, without modifying the input word embeddings.

For this purpose, we propose a Multi-class Neural Network Penultimate Layer (MnnPl), a simple and effective parametrised operator for computing relation representations from word representations.

Specifically, we train a nonlinear multilayer feed-forward neural network using a labelled dataset consisting of word-pairs for different relation types, where the task is to predict the relation between two input words represented by their pre-trained word embeddings.

We find that the penultimate layer of the trained neural network provides an accurate relation representation that generalises beyond the relations in the training dataset.

We emphasise that our focus here is not to classify a given pair to a relation in a pre-defined set (relation classification), but rather to obtain a good representation for the relation between the two words in the pair.

Our experimental results show that MnnPl significantly outperforms unsupervised relational operators including PairDiff in two standard benchmark datasets, and generalises well to unseen out-of-domain relations.

Relations between words can be classified into two types namely, contextual and lexical BID16 BID26 BID11 .

Contextual relations are relations that exist between two words in a given specific context such as a sentence.

For example, given the sentence "the machine makes a lot of noise", a Cause-Effect relation exists between the machine and noise in this particular sentence.

More examples of Contextual relations can be found in BID16 .

On the other hand, Lexical relations hold between two words independent of the contexts in which those two words occur.

For instance, the lexical relation capital-of exists between London and England.

WordNet, for example, organises words into various lexical relations such as is-a-synonym-of, is-a-hypernym-of, is-an-meronym-of, etc.

Our focus in this paper is on representing lexical relations.

Word embeddings learning methods map words to real-valued vectors that represent the meanings of those words.

Given the embeddings of two words, BID25 showed that relations that hold between those words can be represented by the vector-offset (difference) between the corresponding word embeddings.

This observation sparked a line of research on relational operators that can be used to discover relational information from word embeddings besides vector-offset.

Using pre-trained word embeddings to represent relations is attractive for computational reasons.

Unlike holistic approaches that represent the relation between two words by lexico-syntactic patterns extracted from the co-occurrence contexts of the two words, relational operators do not require any co-occurrence contexts.

This is particularly attractive from a computational point of view because the number of possible pairings of n words grows O(n 2 ), implying that we must retrieve co-occurrence contexts for all such pairings for extracting lexico-syntactic patterns for the purpose of representing the relations between words.

On the other hand, in the compositional approach, once we have pre-trained the word embeddings we can compute the relation representations for any two words without having to re-learn anything.

For example, in applications such as relational search BID8 , we must represent the relation between two words contained in a user query.

Because we cannot anticipate all user queries and cannot precompute relation representations user queries offline, relation compositional methods are attractive for relational search engines.

Compositional methods of relation representation differ from Knowledge Graph Embedding (KGE) methods such as TransE BID0 , DistMult BID42 , CompIE BID36 , etc.

in the sense that in KGE, given a knowledge graph of tuples (h, r, t) in which a relation r relates the (head) entity h to the (tail) entity t, we must jointly learn embeddings for the entities as well as for the relations such that some scoring function is optimised.

For example, TransE scores a tuple (h, t, r) by the 1 or 2 norm of the vector (h + r ??? t) (we use bold fonts to denote vectors throughout the paper).

On the other hand, the relation composition problem that we consider in this paper does not attempt to learn entity embeddings or relation embeddings from scratch but use pre-trained word/entity embeddings to compose relation representations.

Therefore, compositional methods of relation representation are attractive from a computational point of view because we no longer need to learn the word/entity embeddings and can focus only on the relation representation learning problem.

On the other hand, compositional methods for relation representation differ from those proposed for solving the analogy completion such as 3CosAdd BID25 , 3Cos-Mult BID19 , 3CosAvg and LRCos .

Analogy completion is the task of finding the missing word (d) in the two analogical word-pairs "a is to b as c is to d".

To solve analogy completion, one must first detect the relation in which the two words in the first pair (i.e. (a, b)) stand in, and then find the word d that is related in the same way to c. Methods that solve analogy questions typically consider the distances between the words of the two pairs in some common vector space.

For example, 3CosAdd computes the inner product between the vector (b ??? a + c) and the word embedding d for each word in the vocabulary.

If the vectors are l 2 normalised then inner-product is equivalent to cosine similarity, which can be seen as a calculation involving cosine similarity scores for three pairs of words (b, d), (a, d) and (c, d) explaining its name 3CosAdd.

3CosMult, on the other hand, considers the same three cosine similarity scores but in a multiplicative formula.

However, analogy completion methods such as 3CosAdd or 3CosMult cannot be considered as relation representation methods because they do not create a representation for the relation between a and b at any stage during the computation.

BID13 compared different unsupervised relational operators such as PairDiff, concatenation, vector addition and elementwise multiplication and reported that PairDiff to be the best operator for analogy completion whereas, elementwise multiplication was the best for link prediction in knowledge graphs.

A recent work BID15 has theoretically proven that PairDiff to be the best linear unsupervised operator for relation representation when the relational distance (similarity) between two word-pairs is measured in term of the squared Euclidean distance between the corresponding relation representation vectors.

Recently, several limitations have been reported of the existing unsupervised relational representation operators BID22 BID29 .

In particular, the distance between word embeddings in a semantic space significantly affects the performance of PairDiff in analogy completion.

Specifically, to measure the relational similarity between (a, b) and (c, d) pairs using PairDiff, prior work compute the inner-product between the normalised offset vectors: (a ??? b) (c ??? d).

This is problematic because the task of measuring relational similarity between the two word-pairs is simply decomposed into a task of measuring lexical similarities between individual words of the pairs.

Specifically, the above inner-product can be rewritten as a c ??? a d ??? b c + b d. This value can become large, for example when a is highly similar to c or b is highly similar to d, irrespective of the relationship between a and b, and c and d.

As a concrete example of this issue, consider measuring the relational similarity between (water, riverbed ) and each of the two word-pairs (traffic, street) and (water, drink )).

In this case, water flows-In riverbed is the implicit relation expressed by the two words in the stem word-pair (water, riverbed ).

Therefore, the candidate pair (traffic, street) is relationally more similar to the stem word-pair than (water, drink ) because flows-In also holds between traffic and street.

However, if we use pre-trained GloVe word embeddings BID27 with PairDiff as the relation representation, then (water, drink ) reports a higher relational similarity score (0.62) compared to that for (traffic, street) (0.42) because of the lexical similarities between the individual words.

PairDiff was originally evaluated by BID25 using semantic and syntactic relations in the Google dataset such as Capital-City, Male-Female, Currency, City-in-State, singular-plural, etc.

However, more recent works have shown that although PairDiff can accurately represent the relation types in the Google dataset, it fails on other types of relations BID18 BID10 .

For example, BID18 showed PairDiff cannot detect paradigmatic relations such as hypernymy, synonymy and antonymy, whereas BID10 reported that hypernym-hyponym relation is more complicated and a single offset vector cannot completely represent it.

The space of unsupervised operators proposed so far in the literature is limited in the sense that the operators pre-defined and fixed, and cannot be adjusted to capture the actual relations that exist between words.

It is unrealistic to assume that the same operator can represent all relation types from the word embeddings learnt from different word embedding learning algorithms.

On the other hand, there are many datasets such as SemEval 2012 Task2, Google, MSR, SAT verbal analogy questions etc., which already provide examples of the types of relations that actually exist between words.

Our proposed supervised relational composition method learns a parametrised operator implemented as a neural network, which can be trained to better represent relations between words.

Word embeddings have been used as features in prior work for learning lexical relations between words.

Given two words, BID41 first represent the relation between those words using PairDiff and then train a multi-class classifier for classifying different relation types.

Methods that focus on detecting a particular type of relation between two words such as hypernymy, have also used unsupervised relation composition operators such as PairDiff to create a feature vector for a word-pair BID2 BID21 BID30 BID10 .

BID2 and BID30 train respectively a logistic regression classifier and a linear support vector classifier using word-pairs represented by the PairDiff or concatenation of the corresponding pretrained word embeddings.

BID21 used PairDiff and vector concatenation as operators for representing the relation between two words and evaluated the representations in a lexical entailment task and a hypernym prediction task.

They found that the representations produced by these operators did not capture relational properties but simply retained the information in individual words, which was then used by the classifiers to make the predictions.

Similarly, BID10 observe that PairDiff is inadequate to induce the hypernym relation.

BID3 analysed PairDiff on a number of different relation types and found that its performance varies significantly across relations.

Our goal in this paper is to learn a parametrised two-argument function f (??, ??; ??) that can accurately represent the relation between two given words a and b using their pre-trained d-dimensional word embeddings a, b ??? R d .

Here, ?? denotes the set of parameters that governs the behaviour of f , which can be seen as a supervised operator that outputs a relation representation from two input word representations.

The output of f , for example, could be a vector that exists in the same or a different vector space as a and b, as given by (1).

DISPLAYFORM0 In general d = m and word and relation representations can have different dimensionalities and even when d = m they might be in different vector spaces.

We could extend this definition to include higher-order relation representations such as matrices or tensors but doing so would increase the computational overhead.

Therefore, we limit supervised relational operators such that they return vectors as given by FORMULA0 DISPLAYFORM1 , and for vector concatenation we have f (a, b; ??) = a ??? b (m = 2d), where ??? denotes concatenation of two vectors.

In unsupervised operators, ?? is a constant that does not influence the output relation embedding.

We implement the proposed supervised relation composition operator as a feed-forward neural network with one or more hidden layers followed by a softmax layer as shown in FIG0 .

Weight matrices for the hidden layers are W 1 and W 2 , whereas the biases are s 1 and s 2 .

g refers to the nonlinear activation for the hidden layers.

We experiment with different nonlinearities in the hidden layers.

Using a dataset DISPLAYFORM0 of word-pairs (a i , b i ) with relations r i , we train the neural network to predict r i given the concatenated pre-trained word embeddings a i ??? b i as the input.

We minimise the 2 regularised cross-entropy loss over the training instances.

After training the neural network, we use its penultimate layer (i.e. the output of the final hidden layer) as the relation representation for a word-pair.

We call this method Multi-class Neural Network Penultimate Layer (MnnPl).We emphasise that our goal is not to classify a given pair into a specific set of relations, but rather to find a representation of the relation between any pair of words.

Therefore, we test the learnt relation representation using relations that are not seen during training (i.e. out-of-domain examples) by holding out a subset of relations during training.

We evaluate the relation embeddings learnt by the proposed MnnPl on two standard tasks: out-of-domain relation prediction and measuring the degree of relational similarities between two word-pairs.

In Section 4.1, we first introduce the relational training datasets and the input word embedding models that we used to compose relation embeddings.

Next, in Section 4.2, we describe the experimental setup that we follow to train the proposed method.

We compare the performance of the MnnPl with various baseline methods as illustrated in Section 4.3.

In Section 4.4 and 4.5, we discuss the experiments conducted on the out-of-domain and in-domain relation prediction task, respectively.

The task of measuring the degree of relational similarities is presented in Section 4.6.

In short, each two word-pairs in the dataset for this task has a manually assigned relational similarity score, which we consider as the gold standard rating for relational similarity.

We used two previously proposed datasets for evaluating MnnPl: BATS 1 and DiffVec 2 BID41 .

BATS is a balanced dataset that contains 4 main relation types, two are semantic relations (Lexicographic and Encyclopaedic) and the other two are syntactic relations (Inflectional and Derivational).

Each main category has 10 different sub-relation types and 50 word-pairs are provided for each relation (2,000 unique word-pairs in total).

DiffVec covers 36 subcategories that are classified into 15 main relation types in total (31 semantic and 6 syntactic).

The dataset is unbalanced because a different number of word-pair examples assigned to each relation, in total it has 12,452 word-pairs.

We exclude relations that has less than 10 examples from experiments.

For word embeddings, we use CBOW, Skip-Gram (SG) BID24 and GloVe BID27 as the input to the proposed method.

For consistency of the comparison, we train all word embedding learning methods on the ukWaC corpus BID9 , a web-derived corpus of English consisting of ca.

2 billion words.

Words that appear less than 6 times in the entire corpus are truncated, resulting in a vocabulary of 1,371,950 unique words.

We use the publicly available implementations by the original authors for training the word embeddings using the recommended parameters settings.

Specifically, GloVe model was trained with window size 15, 50 iterations, weighting function parameters x max = 100, ?? = 0.75.

CBOW and SG embeddings were trained with window size 8, 25 negative samples, 15 iterations, sampling parameter equal to 10 ???4 .In addition to the prediction-based word embeddings created using CBOW and GloVe, we use Latent Semantic Analysis (LSA) to obtain counting-based word embeddings BID5 BID40 BID4 .

A co-occurrence matrix M ??? R n??n is first constructed considering the 50k most frequent words in the corpus to avoid data sparseness.

The raw counts are weighted following positive point-wise mutual information (PPMI) method.

Subsequently, singular value decomposition (SVD) is applied to reduce the dimensionality M to lower rank matrices U k S k V k , where S k is a diagonal matrix that has the largest k singular values of M as the diagonal elements.

U k and V k are orthogonal matrices of singular vectors of the corresponding k singular values.

Following BID20 , S k is ignored when representing the words (i.e. M = U k ).

We use the word embeddings trained on the ukWaC with 50 dimensions as the input to the neural network.

Overall, we found 2 normalisation of word embeddings to improve results.

We use Stochastic Gradient Descent (SGD) with Momentum BID28 with mini-batch size of 128 to minimise the 2 regularised cross-entropy error.

All parameters are initialised by uniformly sampling from [???1, +1] and the initial learning rate is set to 0.1.

Dropout regularisation is applied with a 0.25 rate.

Tensorflow is used to implement the model.

We train the models till the convergence on a validation split.

We used the Scholastic Aptitude Test (SAT) 374 multiple choice analogy questions dataset BID37 ] for validating the hyperparameter values.

Specifically, we selected the number of the hidden layers among {1, 2, 3} and the activation function g of the hidden layers among {tanh, relu, linear}. On the validation dataset, we found the optimal configuration was to set the number of hidden layers to two and the nonlinear activation to tanh.

The optimal 2 regularisation coefficient ?? was 0.001.

We train the models till the convergence on the validation dataset.

These settings performed consistently well in all our evaluations.

We compare the relation representations produced by MnnPl against several baselines as detailed next.

Note that the considered baselines produce relation representations for word-pairs.

Unsupervised Baselines: We implement the following unsupervised relational operators for creating relation representations using word embeddings BID14 : PairDiff, Concatenation (Concat), elementwise addition (Add) and elementwise multiplication (Mult).

These operators are unsupervised in the sense that there are no parameters in those operators that can be learnt from the training data.

Supervised Baselines: We design a supervised version of the Concat operator parametrised by a weight matrix W ??? R d??m and a bias vector s ??? R m to compute a relation representation r for two words a and b as given in (2).

DISPLAYFORM0 We call this baseline as the Supervised Concatenation (Super-Concat).

Likewise, we design a supervised version of PairDiff, which we name Super-Diff as follows: DISPLAYFORM1 In addition to the above supervised operators, we use the bilinear operator proposed by BID15 (given in (4)) as a supervised relation representation method.

DISPLAYFORM2 Here, A ??? R d??d??m is a 3-way tensor in which each slice is a d ?? d real matrix.

The first term in (4) corresponds to the pairwise interactions between a and b. P, Q ??? R d??d are the projection matrices involving first-order contributions respectively of a and b towards r. We refer to this operator as BiLin.

We train the above-mentioned three supervised relational operators using a marginbased rank loss objective.

Specifically, we minimise the distance between the relation representations of the analogous pairs (positive instances), while maximising the distance between the representations of non-analogous examples (negative instances) created via random perturbations.

Given a set of word pairs S r that are related by the same relation, we generate positive training instances ((a, b), (c, d)) by pairing word-pairs (a, b) ??? S r and (c, d) ??? S r .

Next, to generate negative training instances, we corrupt a positive instance by pairing (a, b) ??? S r with a word-pair (c , d ) ??? S r that belongs to a different relation r = r. One negative instance is generated for each analogous example in our experiments, resulting in a balanced binary labelled dataset.

The regularised training objective L(D; ??) is given by (5).

DISPLAYFORM3 Here, ?? is a margin hyperparameter set to 1 according to the best accuracy on the SAT validation dataset.

The regularisation coefficient ?? is set separately for each parameter in the different supervised relation composition operators using the SAT validation dataset.

For SuperConcat and Super-Diff , regularising W and s resulted in lowering the accuracy on SAT questions.

Therefore, no regularisation is applied for those two operators.

For the BLin operator, the best regularisation coefficient for the tensor A on the validation dataset was 0.1.

However, regularising P and Q decreased the performance on the validation set, and therefore were not regularised.

A critical evaluation criterion for a relation representation learning method is whether it can accurately represent not only the relations that exist in the training data that was used to learn the relation representation but can also generalise to unseen relations (out-of-domain).

Therefore, to evaluate the different relation representation methods, we employ them in an out-of-domain relation prediction task.

Specifically, we use different relations for testing than that used in training.

No training is required for unsupervised operators.

Next, we describe the evaluation protocol in detail.

Lets denote a set of relation types by R and a set of word-pairs covering the relations in R by D. First, we randomly sample five target relations from the dataset to construct a relation set R t for testing and the remainder represents a set of source relations R s that is used for training the supervised relational operators including the supervised baselines and the proposed MnnPl.

We use the set D s of word-pair instances covering R s to learn the supervised operators by predicting the relations in R s .

To evaluate the performance of such operators, we use the relational instances in the test split D t that cover the out-of-domain relations in R t .

We conduct 1-NN relation classification on D t dataset.

The task is to predict the relation that exists between two words a and b from the sampled relations in R t .

Specifically, we represent the relation between two words using each relational operator on the corresponding word embeddings.

Next, we measure the cosine similarity between representations for the stem pair and all the word-pairs in D t .

For each target word-pair, if the top-ranked word-pair has the same relation as the stem pair, then it is considered to be a correct match.

Note that we do not use D t for learning or updating the (supervised) relational operator but use it only for the 1-NN relation predictor.

We repeat this process ten times by selecting different R s and R t relation sets and use leave-one-out evaluation for the 1-NN as the evaluation criteria.

We compute the (micro-averaged) classification accuracy of the test sets as the evaluation measure.

Because each relation type in an out-of-domain relation set has multiple relational instances, a suitable relation representation method retrieves the related pairs for a target pair at the top of the ranked list.

For this purpose, we measure Mean Average Precision (MAP) for the relation representation methods.

To derive further insights into the relation representations learnt, following BID26 , we use the notion of "near" vs. "far" analogies considering the similarities between the corresponding words in the two related pairs.

For example, (tiger, feline), (cat, animal ) and (motorcycle, vehicle) are all instances of the is-a-hypernym-of relation.

One could see that (tiger, feline) is closer to (cat, animal ) than (motorcycle, vehicle).

Here, tiger and cat are similar because they are both animals; also feline and animal have shared attributes.

On the other hand, the corresponding words in the two pairs (tiger, feline) and (motorcycle, vehicle) have low attributional similarities between tiger and motorcycle or between feline and vehicle.

Detecting near analogies using word embeddings is easier compared to far analogies because attributional similarity can be measured accurately using word embeddings.

For this reason, we evaluate the accuracy of a relation representation method at different degrees of the analogy as follows.

Given two word-pairs, we compute the cross-pair attributional similarity using SimScore defined by (6).

DISPLAYFORM0 Here, sim(x, y) is the cosine similarity between x and y. Next, we sort the word-pairs in the descending order of their SimScores (i.e. from near to far analogies).

Examples of far and near analogies with SimScores for some selected word-pairs are presented in TAB2 .To alleviate the effect of attributional similarity between two word-pairs in our evaluation, we remove the 25% top-ranked (nearest) pairs for each stem pair.

Consequently, a relation representation method that relying only on attributional similarity is unlikely to accurately represent the relations between words.

The average accuracy (Acc) and the MAP of the relation representation operators for CBOW, SG, GloVe and LSA embeddings are presented in TAB3 .

As can be observed among the different embedding types, MnnPl consistently outperforms all other methods in both Acc and MAP score.

The differences between MnnPl and other methods for all rounds and target relations are statistically significant (p < 0.01) according to a paired t-tes.

CBOW embeddings report the best Acc and MAP scores for the two datasets in contrast to all other embedding models.

We also assess how good such relational operators are on the in-domain relation prediction task, wherein the task is to represent relational instances that belong to the relation set used on training the models.

We find that MnnPl can accurately represent relations in this in-domain setting as well (see Section 4.5).To further evaluate the accuracy of the different relational operators on different relation types, we break down the evaluation per major semantic relation type in the BATS dataset as shown in TAB4 .

We see that lexicographic relations are more difficult compared to encyclopaediac relations for all methods.

Overall, the proposed MnnPl consistently outperforms other methods for both types of semantic relations.

On the other hand, PairDiff performs significantly worse for lexicographic relations.

We believe that this result explains PairDiff's superior performance on the Google analogy dataset, which contains a large proportion of encyclopaediac relations such as capital-common-countries, capital-currency, city-in-state, and family.

ADD achieves the second best accuracy for Encyclopedic relations (where PairDiff is only slightly behind it), whereas Concat follows MnnPl in term of MAP scores.

For encyclopaediac relations, the head words can be grouped into a sub-space in the embedding space that is roughly aligned with the sub-space of the tail words BID23 BID1 .

For instance, in the country-capital relation the head words represent countries while the tail words represent cities.

On the other hand, lexicographic relation types do not have specific sub-spaces for the related head and tail words, which means that the offset vectors would not be sufficiently parallel for PairDiff to work well.

This is further evident from FIG2 where the average cosine similarity scores between the relation embeddings computed using PairDiff is significantly smaller for the lexicographic relations compared to that for the encyclopaediac relations on the BATS dataset.

Consequently, the performance of PairDiff on lexicographic relations is poor, whereas MnnPl reports the best results.

As mentioned in Section 2.2, PairDiff is biased towards the attributional similarity between words in two word-pairs compared.

To evaluate the effect of this, we group test cases in the DiffVec dataset into two categories: (a) lexical-overlap (i.e. there are test cases that have one word in common between two word-pairs) and (b) lexical-nonoverlap (i.e. no words are common between the two word-pairs in all the test cases).

In other words, given the test word-pair (a, b), then if there is a train word-pair (a, c), (b, c), (c, a) or (c, b) we consider this case in the lexical-overlap set.

For example, (animal, cat) and (animal, dog) has lexical-overlap because animal is a common word in the two pairs.

FIG3 shows the average 1-NN classification accuracy for the best unsupervised operator PairDiff and MnnPl.

We see that the performance drops significantly from lexical-overlap to lexicalnonoveralp by ca.

10% for PairDiff, whereas that drop is ca.

1.8% for MnnPl.

This result indicates that MnnPl is affected less by attributional similarity compared to PairDiff.

We evaluate the performance of the relation representation operators considering in-domain setting, wherein we test the performance on relational instances belong to relation types used in the training set.

Recall that R and D refer to the set of relations and the set of relational instances covering such relations, respectively.

In the in-domain setting, we do not need to split R to source and target relation sets.

Instead, we implement 5-stratified folds cross-validation considering the set of relational instances in the dataset D We use 1-NN and MAP as we did in the out-of-domain experiment.

So in-domain experiment setting is very similar to out-of-domain experiment expect in the latter we use R s = R t for the evaluation.

Detailed results for in-domain evaluation are presented in TAB6 .

The relational similarity is the correspondence between the relations of two word-pairs.

To measure a relational similarity score between two pairs of words, one must first identify the relation in each pair to perform such comparison.

Suitable relation embeddings should highly correlate with human judgments of relational similarity between word-pairs.

For this task, we use the dataset proposed by BID3 3 which is inspired by SemEval-2012 task 2 dataset BID17 .

In this dataset, humans are asked to score pairs of words directly focusing on a comparison between instances with similar relations.

For examples, in Location:Item relation, the pairs (cupboard, dishes) and (kitchen, food ) are assigned higher relational similarity score (6.18) than the pairs (cupboard, dishes) and (water, ocean) which is rated 3.8.

Instances of this relation (X, Y) can be expressed by multiple patterns such as "X holds Y" or "Y in the X", and one reason that the second example is assigned low score is that the words in the pair (water, ocean) are ordered reversely compared to other pairs.

BID3 dataset consist of 6,194 word-pairs across 20 semantic relation subtypes.

We calculated the relational similarity score of two pairs as the cosine similarity between the corresponding relation vectors generated by the considered operators.

Then, we measure the Pearson correlation coefficient between the average human relational similarity ratings and the predicted scores by the methods.

For this task, we choose to train the supervised methods on BATS as the overlap of the relation set between BATS and Chen datasets are small.

We exclude any word-pairs in Chen dataset that appears in the training data.

Table 5 shows Pearson correlations for all the four embedding models and the relational representation methods across all relations, where high values indicate a better agreement with the human notion of relational similarity.

As can be observed, the proposed MnnPl correlated better with human ratings than the supervised and unsupervised baselines.

According to the Fisher transformation test of statistical significant, the reported correlations of MnnPl is statistically significant at the 0.05 significant level.

Interestingly, the Concat baseline shows a stronger correlation coefficient than PairDiff.

Moreover, for SG and LSA embeddings, Add and Mult are considered stronger than PairDiff.

In consistent with out-of-domain relation prediction task, CBOW embedding perform better than other embeddings for measuring the degree of relational similarity.

Indeed, measuring the degree of relational similarity is a challenging task and required qualified fine-grained relation embeddings to obtain accurate scores of relational instances.

Table 5 : Results of measuring relational similarity scores (Pearson's correlations).

We considered the problem of learning relation embeddings from word embeddings using parametrised operators that can be learnt from relation-labelled word-pairs.

We experimentally showed that the penultimate layer of a feed-forward neural network trained for classifying relation types (MnnPl) can accurately represent relations between two given words.

In particular, some of the disfluencies of the popular PairDiff operator can be avoided by using MnnPl, which works consistently well for both lexicographic and encyclopaedic relations.

The relation representations learnt by MnnPl generalise well to previously unseen (out-of-domain) relations as well, even though the number of training instances is typically small for this purpose.

Our analysis highlighted some important limitations in the evaluation protocol used in prior work for relation composition operators.

Our work questions the belief that unsupervised operators such as vector offset can discover rich relational structures in the word embedding space.

More importantly we show that simple supervised relational composition operators can accurately recover the relational regularities hidden inside word embedding spaces.

We hope our work will inspire the NLP community to explore more sophisticated supervised operators to extract useful information from word embeddings in the future.

Recently, BID31 show that accessing lexical relations such as hypernym relying only on distributional word embeddings that are trained considering 2-ways cooccurrences between words is insufficient.

They illustrate the advantages of using the holistic (pattern-based) to detect such relations.

Indeed, it is expected that the holistic and the compositional approaches for representing relations have complementary properties since the holistic uses lexical contexts in which the two words of interest co-occur, while the compositional uses only their embeddings BID33 .

Interesting future work includes unifying the two approaches for relation representations.

<|TLDR|>

@highlight

Identifying the relations that connect words is important for various NLP tasks. We model relation representation as a supervised learning problem and learn parametrised operators that map pre-trained word embeddings to relation representations.

@highlight

This paper presents a novel method for representing lexical relations as vectors using just pre-trained word embeddings and a novel loss function operating over pairs of word pairs.

@highlight

A novel solution to the relation compositon problem when you already have pre trained word/entity embeddings and are interested only in learning to compose relation representations.