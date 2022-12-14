Entailment vectors are a principled way to encode in a vector what information is known and what is unknown.

They are designed to model relations where one vector should include all the information in another vector, called entailment.

This paper investigates the unsupervised learning of entailment vectors for the semantics of words.

Using simple entailment-based models of the semantics of words in text (distributional semantics), we induce entailment-vector word embeddings which outperform the best previous results for predicting entailment between words, in unsupervised and semi-supervised experiments on hyponymy.

Modelling entailment, is a fundamental issue in the semantics of natural language, and there has been a lot of interest in modelling entailment using vector-space representations.

But, until recently, unsupervised models such as word embeddings have performed surprisingly poorly at detecting entailment BID11 ; BID9 , not beating a frequency baseline BID11 .

Entailment is the relation of information inclusion, meaning that y entails x if and only if everything that is known given x is also known given y. As such, representations which support entailment need to encode not just what information is known, but also what information is unknown.

The results on lexical entailment seem to indicate that standard word embeddings, such as Word2Vec, do not reflect the relative abstractness of words, and in this sense do not reflect how much information is left unspecified by a word.

In contrast with the majority of the work in this area, which simply uses existing vector-space embeddings of words in their models of entailment, recent work has addressed this issue by proposing new vector-space models which are specifically designed to capture entailment.

In particular, BID10 use variances to represent the uncertainty in values in a continuous space, and BID4 use probabilities to represent uncertainty about a discrete space.

We will refer to the latter as the "entailment-vectors" framework.

In this work, we use this framework from BID4 to develop new entailment-based models for the unsupervised learning of word embeddings, and demonstrate that these embeddings achieve unprecedented results in predicting entailment between words.

Our unsupervised models use the distribution of words in a large text corpus to induce vector-space representations of the meaning of words.

This approach to word meaning is called distributional semantics.

The distributional semantic hypothesis BID3 says that the meaning of a word is reflected in the distribution of text contexts which it appears in.

Many methods (e.g. BID2 BID8 BID6 and this paper) have been proposed for inducing vector representations of the meaning of words (word embeddings) from the distribution of wordcontext pairs found in large corpora of text.

In the framework of BID4 , each dimension of the vector-space represents something that might be known, and continuous vectors represent probabilities of these features being known or unknown.

BID4 illustrate their framework by proposing a reinterpretation of existing Word2Vec BID6 word embeddings which maps them into entailment vectors, which in turn successfully predict entailment between words (hyponymy).

To motivate this reinterpretation of existing word embeddings, they propose a model of distributional semantics and argue that the Word2Vec training objective approximates the training objective of this distributional semantic model given the mapping.

In this paper, we implement this distributional semantic model and train new word embeddings using the exact objective.

Based on our analysis of this model, we propose that this implementation can be done in several ways, including the one which motivates BID4 's reinterpretation of Word2Vec embeddings.

In each case, training results in entailment vector embeddings, which directly encode what is known and unknown given a word, and thus do not require any reinterpretation to predict hyponymy.

To model the semantic relationship between a word and its context, the distributional semantic model postulates a latent pseudo-phrase vector for the unified semantics of the word and its neighbouring context word.

This latent vector must entail the features in both words' vectors and must be consistent with a prior over semantic vectors, thereby modelling the redundancy and consistency between the semantics of two neighbouring words.

Based on our analysis of this entailment-based distributional semantic model, we hypothesise that the word embeddings suggested by BID4 are in fact not the best way to extract information about the semantics of a word from this model.

They propose using a vector which represents the evidence about known features given the word (henceforth called the likelihood vectors).

We propose to instead use a vector which represents the posterior distribution of known features for a phrase containing only the word.

This posterior vector includes both the evidence from the word and its indirect consequences via the constraints imposed by the prior.

Our efficient implementation of this model allows us to test this hypothesis by outputting either the likelihood vectors or the posterior vectors as word embeddings.

To evaluate these word embeddings, we predict hyponymy between words, in both an unsupervised and semi-supervised setting.

Given the word embeddings for two words, we measure whether they are a hypernym-hyponym pair using an entailment operator from BID4 applied to the two embeddings.

We find that using the likelihood vectors performs as well as reinterpreting Word2Vec embeddings, confirming the claims of equivalence by BID4 .

But we also find that using the posterior vectors performs significantly better, confirming our hypothesis that posterior vectors are better, and achieving the best published results on this benchmark dataset.

In addition to these unsupervised experiments, we evaluate in a semi-supervised setting and find a similar pattern of results, again achieving state-of-the-art performance.

In the rest of this paper, section 2 presents the formal framework we use for modelling entailment in a vector space, the distributional semantic models, and how these are used to predict hyponymy.

Section 3 discusses additional related work, and then section 4 presents the empirical evaluation on hyponymy detection, in both unsupervised and semi-supervised experiments.

Some additional analysis of the induced vectors is presented in section 4.4.

Distributional semantics uses the distribution of contexts in which a word occurs to induce the semantics of the word BID3 BID2 BID8 .

The Word2Vec model BID6 ) introduced a set of refinements and computational optimisations of this idea which allowed the learning of vector-space embeddings for words from very large corpora with very good semantic generalisation.

BID4 motivate their reinterpretation the Word2Vec Skipgram BID6 distributional semantic model with an entailment-based model of the semantic relationship between a word and its context words.

We start by explaining our interpretation of the distributional semantic model proposed by BID4 , and then propose our alternative models.

BID4 postulate a latent vector y which is the consistent unification of the features of the middle word x e and the neighbouring context word x e , illustrated on the left in figure 1.

1 We can think of the latent vector y as representing the semantics of a pseudo-phrase consisting of the two words.

The unification requirement is defined as requiring that y entail both words, written y???x e and y???x e .

The consistency requirement is defined as y satisfying a prior ??(y), which embodies all the the constraints and correlations between features in the vector.

This approach models the relationship between the semantics of a word and its context as being redundant and consistent.

If x e and x e share features, then it will be easier for y to satisfy both y???x e and y???x e .

If the features of x e and x e are consistent, then it will be easier for y to satisfy the prior ??(y).

Henderson & Popa (2016) formalise the above model using their entailment-vectors framework.

This framework models distributions over discrete vectors where a 1 in position i means feature i is known and a 0 means it is unknown.

Entailment y???x requires that the 1s in x are a subset of the 1s in y, so 1???1, 0???0 and 1???0, but 0 / ???1.

Distributions over these discrete vectors are represented as continuous vectors of log-odds X, so P (x i =1) = ??(X i ), where ?? is the logistic sigmoid.

The probability of entailment y???x between two such "entailment vectors" Y, X can be measured using the operator > : DISPLAYFORM0 For each feature i in the vector, it calculates the expectation according to P (y i ) that, either y i =1 and thus the log-probability is zero, or y i =0 and thus the log-probability is log P ( DISPLAYFORM1 Henderson & Popa (2016) formalise the model on the left in FIG0 by first inferring the optimal latent vector distribution Y (equation FORMULA2 ), and then scoring how well the entailment and prior constraints have been satisfied (equation (2) ).

DISPLAYFORM2 where E Y,X e ,Xe is the expectation over the distribution defined by the log-odds vectors Y, X e , X e , and log and ?? are applied componentwise.

The term ??(Y ) is used to indicate the net effect of the prior on the vector Y .

Note that, in the formula (3) for inferring Y , the contribution ??? log ??(???X) of each word vector is also a component of the definition of Y > X from equation (1).

In this way, the score for measuring how well the entailment has been satisfied is using the same approximation as used in the inference to satisfy the entailment constraint.

This function ??? log ??(???X) is a nonnegative transform of X, as shown in figure 2.

Intuitively, for an entailed vector x, we only care about the probability that x i =1 (positive log-odds X i ), because that constrains the entailing vector y to have y i =1 (adding to the log-odds Y i ).The above model cannot be mapped directly to the Word2Vec model because Word2Vec has no way to model the prior ??(Y ).

On the other hand, the Word2Vec model postulates two vectors for every word, compared to one in the above model.

BID4 propose an approximation to the above model which incorporates the prior into one of the two vectors, resulting in each word having one vector X e as above plus another vector X p with the prior incorporated.

DISPLAYFORM3 Figure 2: The function ??? log ??(???X) used in inference and the > operator, versus X.Both vectors X e and X p are parameters of the model, which need to be learned.

Thus, there is no need to explicitly model the prior, thereby avoiding the need to choose a particular form for the prior ??, which in general may be very complex.

This gives us the following score for how well the constraints of this model can be satisfied.

DISPLAYFORM4 In BID4 , score (5) is only used to provide a reinterpretation of Word2Vec word embeddings.

They show that a transformation of the vectors output by Word2Vec ("W2V u.d.

> " below) can be seen as an approximation to the likelihood vector X e .

In Section 4, we empirically test this hypothesis by directly training X e ("W2H likelihood" below) and comparing the results to those with reinterpreted Word2Vec vectors.

In this paper, we implement distributional semantic models based on score (5) and use them to train new word embeddings.

We call these models the Word2Hyp models, because they are based on Word2Vec but are designed to predict hyponymy.

To motivate our models, we provide a better understanding of the model behind score (5).

In particular, we note that although we want X p to approximate the effects ??(Y ) of the prior as in equation 4, in fact X p is only dependent on one of the two words, and thus can only incorporate the portion of ??(Y ) which arises from that one word.

Thus, a better understanding of X p is provided by equation (7).

DISPLAYFORM0 In this framework, equation FORMULA5 is exactly the same formula as would be used to infer the vector for a single-word phrase (analogously to equation FORMULA2 ).This interpretation of the approximate model in equation 5 is given on the right side of figure 1.

As shown, X p is interpreted as the posterior vector for a single-word phrase, which incorporates the likelihood and the prior for that word.

In contrast, X e is just the likelihood, which provides the evidence about the features of Y from the other word, without including the indirect consequences of this information.

This model, as argued above, approximates the model on the left side in FIG0 .

But the grey part of the figure does not need to be explicitly modelled because X p is trained directly.

This interpretation suggests that the posterior vector X p should be a better reflection of the semantics of the word than the likelihood vector X e , since it includes both the direct evidence for some features and their indirect consequences for other features.

We test this hypothesis empirically in Section 4.To implement our distributional semantic models, we define new versions of the Word2Vec code BID6 b) .

The Word2Vec code trains two vectors for each word, where negative sampling is applied to one of these vectors, and the other is the output vector.

This applies to both the Skipgram and CBOW versions of training.

Both versions also use a dot product between vectors to try to predict whether the example is a positive or negative sample.

We simply replace this dot product with score (5) directly in the Word2Vec code, leaving the rest of the algorithm unchanged.

We make this change in one of two ways, one where the output vector corresponds to the likelihood vector X e , and one where the output vector corresponds to the posterior vector X p .

We will refer to the model where X p is output as the "posterior" model, and the model where X e is output as the "likelihood" model.

Both these methods can be applied to both the Skipgram and CBOW models, giving us four different models to evaluate.

The proposed distributional semantic models output a word embedding vector for every word in the vocabulary, which are directly interpretable as entailment vectors in the entailment-vectors framework.

Thus, to predict lexical entailment between two words, we can simply apply the > operator to their vectors, to get an approximation of the log-probability of entailment.

We evaluate these entailment predictions on hyponymy detection.

Hyponym-hypernym pairs should have associated embeddings Y, X which have a higher entailment scores Y > X than other pairs.

We rank the word pairs by the entailment scores for their embeddings, and evaluate this ranked list against the gold hyponymy annotations.

We evaluate on hyponymy detection because it reflects a direct form of lexical entailment; the semantic features of a hypernym (e.g. "animal") should be included in the semantic features of the hyponym (e.g. "cat").

Other forms of lexical entailment would benefit from some kind of reasoning or world knowledge, which we leave to future work on compositional models.

In this paper we propose a distributional semantic model which is based on entailment.

Most of the work on modelling entailment with vector space embeddings has simply used distributional semantic vectors within a model of entailment, and is therefore not directly relevant here.

See BID9 for a comprehensive review of such measures.

BID9 evaluate these measures as unsupervised models of hyponymy detection and run experiments on a number of hyponymy datasets.

We report their best comparable result in Table 1 .

BID10 propose an unsupervised model of entailment in a vector space, and evaluate it on hyponymy detection.

Instead of representing words as a point in a vector space, they represent words as a Gaussian distribution over points in a vector space.

The variance of this distribution in a given dimension indicates the extent to which the dimension's feature is unknown, so they use KL-divergence to detect hyponymy relations.

Although this model has a nice theoretical motivation, the word representations are more complex and training appears to be more computationally expensive than the method proposed here.

We empirically compare our models to their hyponymy detection accuracy and find equivalent results.

The semi-supervised model of BID5 learns a discrete Boolean vector space for predicting hyponymy.

But they do not propose any unsupervised method for learning these vectors.

BID11 report hyponymy detection results for a number of unsupervised and semisupervised models.

They propose a semi-supervised evaluation methodology where the words in the training and test sets are disjoint, so that the supervised component must learn about the unsupervised vector space and not about the individual words.

Following BID4 , we replicate their experimental setup in our evaluations, for both unsupervised and semi-supervised models, and compare to the best results among the models evaluated by BID11 , BID9 and BID4 .

We evaluate on hyponymy detection in both a fully unsupervised setup and a semi-supervised setup.

In the semi-supervised setup, we use labelled hyponymy data to train a linear mapping from the unsupervised vector space to a new vector space with the objective of correctly predicting hyponymy relations in the new vector space.

This prediction is done with the same (or equivalent) entailment operator as for the unsupervised experiments (called "map > " in Table 2 ).

Table 1 : Hyponymy detection accuracies (50% Acc) and average precision (Ave Prec), in the unsupervised experiments.

For the accuracies, * marks a significant improvement over the higher rows.

We replicate the experimental setup of BID11 , using their selection of hyponymhypernym pairs from the BLESS dataset BID0 , which consists of noun-noun pairs, including 50% positive hyponymy pairs plus 50% negative pairs consisting of some other hyponymy pairs reversed, some pairs in other semantic relations, and some random pairs.

As in BID11 , our semi-supervised experiments use ten-fold cross validation, where each fold has items removed from the training set if they contain a word that also occurs in the testing set.

The word embedding vectors which we train have 200 dimensions and were trained using our Word2Hyp modification of the Word2Vec code (with default settings), trained on a corpus of half a billion words of Wikipedia.

We also replicate the approach of BID4 by training Word2Vec embeddings on this data.

To quantify performance on hyponymy detection, for each model we rank the list of pairs according to the score given by the model, and report two measures of performance for this ranked lists.

The "50% Acc" measure treats the first half of the list as labelled positive and the second half as labelled negative.

This is motivated by the fact that we know a priori that the proportion of positive examples has been artificially set to (approximately) 50%.

Average precision is a measure of the accuracy for ranked lists, used in Information Retrieval and advocated as a measure of hyponymy detection by BID10 .

For each positive example, precision is measured at the threshold just below that example, and these precision scores are averaged over positive examples.

For cross validation, we average over the union of positive examples in all the test sets.

Both these measures are reported (when available) in Tables 1 and 2.

The first set of experiments evaluate the different embeddings in their unsupervised models of hyponymy detection.

Results are shown in Table 1 .

Our principal point of comparison is the best results from BID4 ) (called "W2V GoogleNews" in Table 1 ).

They use the preexisting publicly available GoogleNews word embeddings, which were trained with the Word2Vec software on 100 billion words of the GoogleNews dataset, and have 300 dimensions.

To provide a more direct comparison, we replicate the model of BID4 but using the same embedding training setup as for our Word2Hyp model ("W2V Skip").

Both cases use their proposed reinterpretation of these vectors for predicting entailment ("u.d.

> ").

We also report the best results from BID11 and the best comparable results from BID9 .

For our proposed Word2Hyp distributional semantic models ("W2H"), we report results for the four combinations of using the CBOW or Skipgram ("Skip") model to train the likelihood or posterior vectors.

The two Word2Hyp models with likelihood vectors perform slightly better than the best unsupervised model of BID11 , but similarly.

The reinterpretation of Word2Vec vectors ("W2V GoogleNews u.d.

> ") performs significantly better, but when the same method is applied to the smaller Wikipedia corpus ("W2V Skip u.d.

> "), this difference all but disappears.

This confirms the hypothesis of BID4 Table 2 : Hyponymy detection accuracies (50% Acc) and average precision (Ave Prec), in the semisupervised experiments.

However, even with this smaller corpus, using the proposed posterior vectors from the Word2Hyp model are significantly more accurate than the reinterpretation of Word2Vec vectors.

This confirms the hypothesis that the posterior vectors from the Word2Hyp model are a better model of the semantics of a word than the likelihood vectors suggested by BID4 .

Using the CBOW model or the Skipgram model makes only a small difference.

The average precision score shows the same pattern as the accuracy.

To allow a direct comparison to the model of BID10 , we also evaluated the unsupervised models on the hyponymy data from BID1 , which is not as carefully designed to evaluate hyponymy as the BID11 data.

Both the likelihood and posterior vectors of the Word2Hyp CBOW model achieved average precision (81%, 80%) which is not significantly different from the best model of BID10 (80%).

The semi-supervised experiments train a linear mapping from each unsupervised vector space to a new vector space, where the entailment operator > is used to predict hyponymy ("map > ").The semi-supervised results (shown in Table 2) 3 no longer show an advantage of GoogleNews vectors over Wikipedia vectors for the reinterpretation of Word2Vec vectors.

And the advantage of posterior vectors over the likelihood vectors is less pronounced.

However, the two posterior vectors still perform much better than all the previously proposed models, achieving 86% accuracy and nearly 93% average precision.

These semi-supervised results confirm the results from the unsupervised experiments, that Word2Vec embeddings and Word2Hyp likelihood embeddings perform similarly, but that using the posterior vectors of the Word2Hyp model perform better.

Because the similarity measure in equation 5 is more complex than a simple dot product, training a new distributional semantic model is slower than with the original Word2Vec code.

In our experiments, training took about 8 times longer for the CBOW model and about 15 times longer for the Skipgram model.

This meant that Word2Hyp CBOW trained about 8 times faster than Word2Hyp Skipgram.

As in the Word2Vec code, we used a quadrature approximation (i.e. a look-up table) to speed up the computation of the sigmoid function, and we added the same technique for computing the log-sigmoid function.

The relative success of our distributional semantic models at unsupervised hyponymy detection indicates that they are capturing some aspects of lexical entailment.

But the gap between the unsupervised and semi-supervised results indicates that other features are also being captured.

This is not surprising, since many other factors influence the co-occurrence statistics of words.

Table 3 : Ranking of the abstractness (0 > X) of frequent words from the hyponymy dataset, using Word2Hyp-Skipgram-posterior embeddings.

To get a better understanding of these word embeddings, we ranked them by degree of abstractness.

Table 3 shows the most abstract and least abstract frequent words that occur in the hyponymy data.

To measure abstractness, we used our best unsupervised embeddings and measured how well they are entailed by the zero log-odds vector, which represents a uniform half probability of knowing each feature.

For a vector to be entailed by the zero vector, it must be that its features are mostly probably unknown.

The less you know given a word, the more abstract it is.

An initial ranking found that six of the top ten abstract words had frequency less than 300 in the Wikipedia data, but none of the ten least abstract terms were infrequent.

This indicates a problem with the current method, since infrequent words are generally very specific (as was the case for these low-frequency words, submissiveness, implementer, overdraft, ruminant, warplane, and londoner).Although this is an interesting characteristic of the method, the terms themselves seem to be noise, so we rank only terms with frequency greater than 300.The most abstract terms in table 3 include some clearly semantically abstract terms, in particular something and anything are ranked highest.

Others may be affected by lexical ambiguity, since the model does not disambiguate words by part-of-speech (such as end, good, sense, back, and saw).

The least abstract terms are mostly very semantically specific, but it is indicative that this list includes primate, which is an abstract term in Zoology but presumably occurs in very specific contexts in Wikipedia.

In this paper, we propose unsupervised methods for efficiently training word embeddings which capture semantic entailment.

This work builds on the work of BID4 , who propose the entailment-vectors framework for modelling entailment in a vector-space, and a distributional semantic model for reinterpreting Word2Vec word embeddings.

Our contribution differs from theirs in that we provide a better understanding of their distributional semantic model, we choose different vectors in the model to use as word embeddings, and we train new word embeddings using our modification of the Word2Vec code.

Empirical results on unsupervised and semi-supervised hyponymy detection confirm that the model's likelihood vectors, which BID4 suggest to use, do indeed perform equivalently to their reinterpretation of Word2Vec vectors.

But these experiments also show that the model's posterior vectors, which we propose to use, perform significantly better, outperforming all previous results on this benchmark dataset.

The success of these unsupervised models demonstrates that the proposed distributional semantic models are effective at extracting information about lexical entailment from the redundancy and consistency of words with their contexts in large text corpora.

The use of the entailment-vectors framework to efficiently model entailment relations has been crucial to this success.

This result suggests future work using the entailment-vectors framework in unsupervised models that leverage other distributional evidence about semantics, particularly in models of compositional semantics.

The merger of word embeddings with compositional semantics to get representation learning for larger units of text is currently an important challenge in the semantics of natural language, and the work presented in this paper makes a significant contribution towards solving it.

<|TLDR|>

@highlight

We train word embeddings based on entailment instead of similarity, successfully predicting lexical entailment.

@highlight

The paper presents a word embedding algorithm for lexical entailment which follows the work of Henderson and Popa (ACL, 2016).