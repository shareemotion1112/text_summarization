Prepositions are among the most frequent words.

Good prepositional representation  is of great syntactic and semantic interest  in computational linguistics.

Existing methods on preposition representation either treat prepositions as content words (e.g., word2vec and GloVe) or depend heavily on external linguistic resources including syntactic parsing, training task and dataset-specific representations.

In this paper we use word-triple counts (one of the words is a preposition) to  capture the preposition's interaction with its head and children.

Prepositional  embeddings are derived via tensor decompositions on a large unlabeled corpus.

We reveal a new geometry involving Hadamard products and empirically demonstrate its utility in paraphrasing of phrasal verbs.

Furthermore, our prepositional  embeddings are used as simple features to two challenging downstream tasks: preposition selection and prepositional attachment disambiguation.

We achieve comparable to or better results than state of the art on  multiple standardized datasets.

Prepositions are a linguistically closed class comprising some of the most frequent words; they play an important role in the English language since they encode rich syntactic and semantic information.

Many preposition-related tasks still remain unsolved in computational linguistics because of their polysemous nature and flexible usage patterns.

An accurate understanding and representation of prepositions' linguistic role is key to several important NLP tasks such as grammatical error correction and prepositional phrase attachment.

A first-order approach is to represent prepositions as real-valued vectors via word embeddings such as word2vec BID21 and GloVe BID25 .Word embeddings have brought a renaissance in NLP research; they have been very successful in capturing word similarities as well as analogies (both syntactic and semantic) and are now mainstream in nearly all downstream NLP tasks (such as question-answering).

Despite this success, no specific properties of word embeddings of prepositions have been highlighted in the literature.

Indeed, many of the common prepositions have very similar vector representations as shown in TAB0 for preposition vectors trained using word2vec and GloVe.

While this suggests that using available representations for prepositions diminishes the distinguishing feature between prepositions, one could hypothesize that this is primarily because standard word embedding algorithms treat prepositions no different from other content words such as verbs and nouns, i.e., embeddings are created based on co-occurrences with other words.

However, prepositions are very frequent and co-occur with nearly all words, which means that their co-occurrence ought to be treated differently.

Modern descriptive linguistic theory proposes to understand a preposition via its interactions with both the head (attachment) and child (complement) BID12 ; BID8 .

This theory naturally suggests that one should count co-occurrences of a given preposition with pairs of neighboring words.

One way of achieving this would be by considering a tensor of triples (word 1 , word 2 , preposition), where we do not restrict word 1 and word 2 to be head-and child-words; instead we model a preposition's interaction with all pairs of neighboring words via a slice of a tensor X -the slice is populated by word co-occurrences restricted to a context window of the specific preposition.

Thus, the tensor dimension is V ?? V ?? K where V is the vocabulary and K is the number of prepositions; since K ??? 50, we note that V K.Using such a representation, we find that the resulting tensor is low rank and extract embeddings for both preposition and non-preposition words using a combination of standard ideas from word representations (such as weighted spectral decomposition as in GloVe BID25 ) and tensor decompositions (alternating least squares (ALS) methods BID29 ).

The preposition embeddings are discriminative, see preposition similarity of the tensor embedding in TAB0 .

We demonstrate that the resulting representation for prepositions captures the core linguistic property of prepositions.

We do this using both intrinsic evaluations and downstream tasks, where we provide new state-of-the-art results on well-known NLP tasks involving prepositions.

Intrinsic evaluations: We show that the Hadamard product of the embeddings of a verb and a preposition closely approximates the representation of this phrasal verb's paraphrase.

Example: v made v from ??? v produced where represents the Hadamard product (i.e., pointwise multiplication) of two vectors; this approximation does not hold for the standard word embeddings of prepositions (word2vec or GloVe).

We provide a mathematical interpretation for this new geometry as well as empirically demonstrate the generalization on a new data set of compositional phrasal verbs.

Extrinsic evaluations: Our preposition embeddings are used as features for a simple classifier in two well-known challenging downstream NLP classification tasks.

In both tasks, we perform comparable to or strictly better than the state-of-the-art on multiple standardized datasets.

Preposition selection: The choice of prepositions significantly influences (and is governed by) the semantics of the context they occur in.

Furthermore, the prepositional choice is usually very subtle (and consequently is one of the most frequent error types made by second language English speakers BID19 ).

This task tests the choice of a preposition in a large set of contexts (7, 000 instances of both CoNLL-2013 and SE datasets BID26 ).

Our approach achieves 6% and 2% absolute improvement over the previous state-of-the-art on the respective datasets.

Prepositional attachment disambiguation: Prepositional phrase attachment is a common cause of structural ambiguity in natural language.

In the sentence "Pierre Vinken joined the board as a voting member", the prepositional phrase "as a voting member" can attach to either "joined" (the VP) or "the board" (the NP); in this case the VP attachment is correct.

Despite extensive study over decades of research, prepositional attachment continues to be a major source of syntactic parsing errors BID4 ; Kummerfeld et al. (2012); BID7 .

We use our prepositional representations as simple features to a standard classifier on this task.

Our approach tested on a widely studied standard dataset BID2 achieves 89% accuracy, essentially the same performance as state-of-the-art (90% accuracy).

It is noteworthy that while the stateof-the-art results are obtained with significant linguistic resources, including syntactic parsers and WordNet, our approach does not rely on these resources to achieve a comparable performance.

We emphasize two aspects of our contributions:(1) It is folklore within the NLP community that representations via pairwise word counts capture much of the benefits of the unlabeled sentence-data; example: BID29 reports that their word representations via word-triple counts are better than others, but still significantly worse than regular word2vec representations.

One of our main observations is that considering word-triple counts makes most (linguistic) sense when one of the words is a preposition.

Furthermore, the sparsity of the corresponding tensor is no worse than the sparsity of the regular word co-occurrence matrix (since prepositions are so frequent and co-occur with essentially every word).

Taken together, these two points strongly suggest the benefits of tensor representations in the context of prepositions.(2) The word and preposition representations via tensor decomposition are simple features leading to a standard classifier.

In particular, we do not use syntactic parsing (which many prior methods have relied on) or handcrafted features BID26 or train task-specific representations on the annotated training dataset BID2 .

The simplicity combined with our strong empirical results (new state-of-the-art results on long standing datasets) lends credence to the strength of the prepositional representations found via tensor decompositions.

We begin with a description of how the tensor with triples (word,word,preposition) is formed and empirically show that its slices are low-rank.

Next, we derive low dimensional vector representations for words and prepositions via appropriate tensor decomposition methods.

Tensor creation: Suppose that K prepositions are in the preposition set P = {p 1 , . . .

, p K }; here K is 49 in our preposition selection task, and 76 in the attachment disambiguation task.

The vocabulary, the set of all words except prepositions, contains N words V = {w 1 , . . .

, w N }, and N ??? 1M .

We generate a third order tensor X N ??N ??(K+1) from WikiCorpus Al- BID0 in the following way.

We say two words co-occur if they appear within distance t of each other in a sentence.

For k ??? K, the entry X ijk is the number of occurrences where word w i co-occurs with preposition p k , and w j also co-occurs with preposition p k in the same sentence, and this is counted across all sentences in a large WikiCorpus.

Here we use a window of size t = 3.

There are also a number of words which do not occur in the context of any preposition.

To make full use of the data, we add an extra slice X[:, :, K + 1]: the entry X ij(K+1) is the number of occurrences where w i co-occurs with w j (within distance 2t = 6) but at least one of them is not within a distance of t of any preposition.

Note that the preposition window of 3 is smaller than the word window of 6, since it is known that the interaction between prepositions and neighboring words usually weakens more sharply with the distance as compared to content words BID11 .Empirical properties of X: We find that the tensor X is very sparse -only 1% of tensor elements are non-zeros.

Furthermore, every slice log(1 + X[:, :, k]) is low rank (here the logarithm is applied componentwise to every entry of the tensor slice).

We choose slices corresponding to prepositions "about", "before","for", "in" and "of", and plot their normalized singular values in FIG0 .

We see that the singular values decay dramatically, suggesting the low rank structure in each slice.

Tensor decomposition: We combine standard ideas from word embedding algorithms and tensor decomposition algorithms to arrive at the low rank approximation to the tensor log(1 + X).

In particular, we consider two separate methods:

A generic method to decompose the tensor into its modes is via the CANDECOMP/PARAFAC (CP) decomposition BID15 .

The tensor log(1 + X) is decomposed into three modes: U d??N , W d??N and Q d??(K+1) , based on the solutions to the optimization problem (1).

Here u i , w i and q i are the i-th column of U , W and Q, respectively.

DISPLAYFORM0 where a, b, c = 1 t (a b c) is the inner product of three vectors a, b and c. Here 1 is the column vector of all ones and refers to the Hadamard product.

We can interpret the columns of U as the word representations and the columns of Q as the preposition representations, each of dimension d (equal to 200 in this paper).

There are several algorithmic solutions to this optimization problem in the literature, most of which are based on alternating least squares methods BID15 BID5 BID1 and we employ a recent one named Orth-ALS BID29 in this paper.

Orth-ALS periodically orthogonalizes decomposed components while fixing two modes and updating the remaining one.

It is supported by theoretical guarantees and empirically outperforms standard ALS method in different applications.

DISPLAYFORM1 where b U i is the scalar bias for word i in matrix U .

Similarly, b W j is the bias for word j in matrix W , and b Qk is for preposition k in matrix Q. Bias terms are learned to minimize the loss function.

Here ?? ijk is the weight assigned to each tensor element X ijk , and we use the weighting proposed byGloVe: ?? ijk = min X ijk xmax ?? , 1 .

We set hyperparameters x max = 10, and ?? = 0.75 in this work.

We solve this optimization problem via standard gradient descent, arriving at word representations U and tensor representations Q.

Representation Interpretation Suppose that we have a phrase (h, p i , c) where h, p i and c are head word, preposition i(1 ??? i ??? K) and child respectively.

A phrase example is split off something.

The inner product of word vectors of h, p i and c reflects how frequently h and c cooccur in the context of p. It also reflects how cohesive the triple is.

Recall that there is an extra (K + 1)???th slice that describes the word co-occurrences outside the preposition window, which considers cases such as the phrasal verb (v, c) where v and c are the verb and the child.

The verb phrase divide something is equivalent to the phrase split off something.

For any word c that fits in this phrase semantically, we can expect that DISPLAYFORM0 In other words u h q i ??? u v q K+1 , where a b denotes the pointwise multiplication (Hadamard product) of vectors a and b. This suggests that we could paraphrase the verb phrase (h, p i ) by finding the verb v such that DISPLAYFORM1 Well-trained embeddings should be able to capture the relation between the prepositional phrases and their equivalent phrasal verbs.

In TAB1 , we list seven paraphrases of verb phrases, as generated from the weighted tensor decomposition.

A detailed list of paraphrases on a new dataset of compositional verb phrases is available in TAB0 in Appendix B, where we also compare paraphrasing results using regular word embeddings and via both addition and Hadamard product operations.

The combination of tensor representations and Hadamard product results in vastly superior paraphrasing.

In the next two sections, we evaluate tensor-based preposition embeddings in the context of two important NLP downstream tasks: preposition selection and preposition attachment disambiguation.

In this work, we use WikiCorpus as the training corpus for different sets of embeddings.

We train tensor embeddings with both Orth-ALS and weighted decomposition.

The implementation of Orth-ALS is built upon he SPLATT toolkit BID31 .

We perform orthogonalization in the first 5 iterations in Orth-ALS decomposition, and the training is completed when its performance stabilizes.

As for the weighted decomposition, we train for 20 iterations, and its hyperparameters are set as x max = 10, and ?? = 0.75.We also include two baselines, word2vec's CBOW model and GloVe, for comparison.

We set 20 training iterations to both models.

Hyperparameters in word2vec are set as: window size=6, negative sampling=25 and down sampling=1e-4.

Hyperparameters in GloVe are set as: window size=6, x max =10, ??=0.75 and minimum word count=5.

We note that all the representations in this studyword2vec, GloVe and our tensor embedding -are of dimension 200.

The detection and correction of grammatical errors is an important task in NLP.

Second language learners tend to make more mistakes and in particular, prepositional errors make up about 13% of all errors, ranking the second among most common error types BID19 .

This is due to the fact that prepositions are highly polysemous and have flexible usage.

Accurate preposition selection needs to well capture the interaction between preposition and its context.

This task is natural to evaluate how well the lexical interactions are captured by different methods.

Task.

Given a sentence in English with a preposition, we either replace the preposition (to the correct one) or retain it.

For example, "to" should be corrected as "of" in the sentence "It can save the effort to carrying a lot of cards".

Formally, there is a closed set of preposition candidates P = {p 1 , . . .

, p m }.

A preposition p is used in a sentence s consisting of words s = {. . .

, w ???2 , w ???1 , p, w 1 , w 2 , . . .}.

If used incorrectly, we need to replace p by another prepositionp ??? P based on the context.

TAB2 .

We focus on the most frequent 49 prepositions listed in Appendix A.Evaluation metric.

Three metrics, precision, recall and F1 score (harmonic mean of precision and recall) are used to evaluate preposition selection performance.

Our algorithm.

We first preprocess the dataset by removing articles, determiners and pronouns, and take a context window of 3.

We divide the task into two steps: error identification and error correction.

Firstly, we decide whether a preposition is used correctly in the context.

If not, we suggest another preposition as replacement in the second step.

Identification step uses only three features: cosine similarity between the current preposition embedding and the average context embedding, rank of the preposition in terms of cosine similarity, and probability that this preposition is not changed in training corpus.

We build a decision tree classifier with these three features and find that we can identify errors with 98% F1 score in the CoNLL dataset and 96% in the SE dataset.

When it comes to error correction, we only focus on identified errors in the first stage.

Suppose that the original preposition is q, and the candidate preposition is p. ; (4) Confusion probability: the probability that q is replaced by p in the training data.

A two-layer feedforward neural network (FNN) with hidden sizes of 500 and 10 is trained with these features to score prepositions in each sentence.

The one with the highest score is the suggested edit.

Baseline.

State-of-the-art on preposition selection uses n-gram statistics from a large corpus BID26 .

Features such as pointwise mutual information (PMI) and part-of-speech tags are fed into a supervised scoring system.

Prepositions with highest score are chosen as suggested ones.

The performance is affected by both the system architecture and features.

To evaluate the benefits brought by our tensor embedding-based features, we also consider other baselines which have the same two-step architecture whereas features are generated from word2vec and GloVe embeddings.

These baselines allow us to compare the representation power independent of the classifier.

Result.

We compare our methods against baselines mentioned above in TAB3 .

As is seen, tensor embeddings achieve the best performance among all approaches.

In particular, tensor with weighted decomposition has the highest F1 score on CoNLL dataset, 6% improvement over the state of the art.

The tensor with ALS decomposition performs the best on SE dataset, achieving 2% improvement.

We also note that with the same architecture, tensor embeddings perform much better than word2vec and GloVe embeddings on both datasets.

It validates the representation power of tensor embeddings.

To have a deeper insight into feature importance in the preposition selection task, we also perform an ablation analysis of the tensor method with weighted decomposition as shown in TAB4 .

We remove one feature each time, and report the performance achieved by remaining features.

It is found that left context is the most important feature in CoNLL dataset, whereas confusion score is the most important in SE dataset.

Pair similarity and triple similarity are less important compared with other features.

This is because the neural network could learn lexical similarity from embedding features, and diminishes the importance of similarity features.

Discussion.

Now we analyze the reasons why our approach selects wrong prepositions in some sentences.(1) Limited context window.

We focus on the local context within preposition's window.

In some cases, we find that head words might be out of the context window.

In the sentence "prevent more of this kind of tragedy to happening" where to should be corrected as from.

Given the context window of 3, we cannot get the lexical clues provided by prevent, which leads to the selection error in our approach.

(2) Preposition selection requires more context.

Even when the context window contains all words on which the preposition depends, it still may not be sufficient to select the right preposition.

For example, in the sentence "it is controlled by bad men in a not good purpose" where our approach replaces the preposition in with the preposition on given the high frequency of the phrase "on purpose".

The correct preposition should be for based on the whole sentence.

In this section, we discuss prepositional phrase (PP) attachment disambiguation, a well-studied, but still open, hard task in syntactic parsing.

A prepositional phrase usually consists of head words, a preposition and child words.

An example is "he saw an elephant with long tusks", where "with" is attached to the noun "elephant".

In another example "he saw an elephant with his telescope", "with" is attached to the verb "saw".

Head words can be different when only child word is changed.

PP attachment disambiguation inherently requires accurate description of interactions among head, preposition and child, which becomes an ideal task to evaluate our tensor-based embeddings.

Task.

The English dataset used in our work is collected from a linguistic treebank by BID2 .

TAB5 enumerates statistics associated with this dataset.

Each instance consists of several head candidates, a preposition and a child word.

We need to pick the head to which the preposition is attached.

In the examples above, words "saw" and "elephant" are head candidates.

Our algorithm.

Let v h , v p and v c be embeddings for the head candidate h, preposition p and child c respectively.

Features we use for the attachment disambiguation are: (1) embedding feature: candidate, preposition and child embedding; (2) triple similarity: triple sim(h, p, c) = DISPLAYFORM0 ; (5) part-of-speech (pos) tag of the candidate and its next word; (b) distance between h and p.

We use a basic neural network, a two-layer feedforward network (FNN) with hidden sizes of 1000 and 20 to take input features and predict the probability that a candidate is the head.

The candidate with the highest likelihood is chosen as the head.

Baseline.

We include following state-of-the-art approaches in preposition attachment disambiguation.

The linguistic resources they used to enrich features are listed in Table 7 .(1) Head-Prep-Child-Dist (HPCD) Model BID2 : this compositional neural network is used to train task-specific word representations.(2) Low-Rank Feature Representation (LRFR) BID33 : this method incorporates word parts, contexts and labels into a tensor, and uses decomposed vectors as features for disambiguation.

(3) Ontology LSTM (OntoLSTM) BID6 : Word vectors are initialized with GloVeextended from AutoExtend BID27 , and then trained via LSTMs for head selection.

Similar to the experiments in preposition selection, we also include baselines which have the same feedforward network architecture but generate features with vectors trained by word2vec and GloVe.

They are denoted as FNN with different initializations in Table 7 .

Since the attachment disambiguation is a selection task, accuracy is a natural evaluation metric.

Result.

We compare results and linguistic resources of different approaches in Table 7 , where we see that our simple classifier built on the tensor representations is within 1% of the state of the art; prior state of the art results have used significant linguistic resources enumerated in Table 7 .

With the same feedforward neural network as the classifier, our tensor-based approaches (both ALS and WD) achieve better performance than word2vec and GloVe.

Ablation analysis in TAB6 shows that head vector feature affects the performance most (indicating that heads interact more closely with prepositions), and POS tag comes second.

Similarity features appear less important since the classifier has access to lexical relatedness via the embedding features.

Distance feature is reported to be important in previous works since 81.7% sentences take the word closest to the preposition as their head.

In our experiments, distance becomes less important compared with embedding features.

Discussion.

We find that one source of attachment disambiguation error is the lack of broader context in our features.

Broader context is critical in examples such as "worked" and "system" which are head candidates of "for trades" in a sentence.

They are reasonable heads in expressions "worked for trades" and "system for trades".

It requires more context to decide that "system" rather than "worked" is the head in the given sentence.

We further explore the difference in identifying head verbs and head nouns.

We have found that tensor's geometry could aid in paraphrasing verb phrases, and thus it well captures the interaction between verbs and prepositions.

In this task, we want to see whether our approach could do better in identifying head verbs than head nouns.

There are 883 instances with head verbs on which we could achieve an accuracy of 0.897, and 1068 instances with head nouns where the accuracy is 0.887.

We do better in selecting head verbs, but performance does not differ too much across verbs and nouns.

Tensor Decomposition.

Tensors embed higher order interaction among different modes, and the tensor decomposition captures the relations via lower dimensional representations.

There are several decomposition methods such as Alternating Least Square (ALS) BID15 , Simultaneous Diagonalization (SD) BID16 and optimization-based methods BID20 BID23 .

Orthogonalized Alternating Least Square (Orth-ALS) adds the step of component orthogonalization to each update step in the ALS method BID29 .

Orth-ALS, supported by theoretical guarantees and, more relevantly, good empirical performance, is the algorithm of choice in this paper.

Preposition Selection.

Preposition selection, a major area of study in both syntactic and semantic computational linguistics, is also a very practical topic in the context of grammar correction and second language learning.

Prior works typically use hand-crafted heuristic rules in preposition correction BID32 ; lexical n-gram features are also known to be very useful BID26 ; BID28 .

Syntactic information such as POS tags and dependency parsing can further enrich features BID13 , and are standard in generic tasks involving prepositions.

Prepositional Attachment Disambiguation.

There is a storied literature on prepositional attachment disambiguation, long recognized as an important part of syntactic parsing BID14 .

Recent works, based on word embeddings have pushed the boundary of state of the art empirical results.

A seminal work in this direction is the Head-Prep-Child-Dist (HPCD) Model, which trained word embeddings in a compositional network designed to maximize the accuracy of head prediction BID2 .

A very recent work has proposed an initialization with semantics-enriched GloVe embeddings, and retrained representations with LSTM-RNNs BID6 .

Another recent work has used tensor decompositions to capture the relation between word representations and their labels BID33 .

Co-occurrence counts of word pairs in sentences and the resulting word vector representations (embeddings) have revolutionalized NLP research.

A natural generalization is to consider co-occurrence counts of word triples, resulting in a third order tensor.

Partly due to the size of the tensor (a vocabulary of 1M, leads to a tensor with 10 18 entries!) and partly due to the extreme dynamic range of entries (including sparsity), word vector representations via tensor decompositions have largely been inferior to their lower order cousins (i.e., regular word embeddings).In this work, we trek this well-trodden terrain but restricting word triples to the scenario when one of the words is a preposition.

This is linguistically justified, since prepositions are understood to model interactions between pairs of words.

Numerically, this is also very well justified since the sparsity and dynamic range of the resulting tensor is no worse than the original matrix of pairwise co-occurrence counts; this is because prepositions are very frequent and co-occur with essentially every word in the vocabulary.

Our intrinsic evaluations and new state of the art results in downstream evaluations lend strong credence to the tensor-based approach to prepositional representation.

We expect our vector representations of prepositions to be widely used in more complicated downstream NLP tasks where prepositional role is crucial, including "text to programs" BID10 .

The list of most frequent 49 Prepositions in the task of preposition selection is shown below: about, above, absent, across, after, against, along, alongside, amid, among, amongst, around, at, before, behind, below, beneath, beside, besides, between, beyond, but, by, despite, during, except, for, from, in, inside, into, of, off, on, onto, opposite, outside, over, since, than, through, to, toward, towards, under, underneath, until, upon, with.

B PARAPHRASING OF PHRASAL VERBS In Section 3 we have provided a simple linear algebraic method to generate paraphrases to compositional phrasal verbs.

We approximate the paraphrase representation u v via Eq. 3, and get a list of words which have similar representations as candidate paraphrases.

These candidates do not include words that are the same as component words in the phrase.

We also require that a reasonable paraphrase should be a verb.

Therefore we choose the verb which is most similar to u v among candidates.

We filter verbs with Python NLTK tools BID3 and Linguistics library of NodeBoxSmedt (2016) .Sample examples of the top paraphrases are provided in TAB1 .

Here we provide a detailed enumeration of the results of our linear algebraic method on a new dataset of 60 compositional phrases.

In the paraphrasing task, we consider three sets of embeddings, word2vec, GloVe and tensor embeddings from weighted decomposition.

We also have two composition methods: addition and Hadamard product to approximate the paraphrase representation from verb and preposition vectors.

Addition is included here because it has been widely used to approximate phrasal embedding in previous works BID22 BID9 .

We enumerate paraphrases generated by six combinations of embeddings and composition methods, validating the representation power of tensor embeddings and the multiplication (Hadamard product) composition method.

As we can see from TAB7 and 10, tensor embedding works better with multiplicative composition, whereas word2vec and GloVe work better with additive composition.

Overall, tensor embedding together with multiplication gives better paraphrases than other approaches.

<|TLDR|>

@highlight

This work is about tensor-based method for preposition representation training.