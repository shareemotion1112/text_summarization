Recent literature suggests that averaged word vectors followed by simple post-processing outperform many deep learning methods on semantic textual similarity tasks.

Furthermore, when averaged word vectors are trained supervised on large corpora of paraphrases, they achieve state-of-the-art results on standard STS benchmarks.

Inspired by these insights, we push the limits of word embeddings even further.

We propose a novel fuzzy bag-of-words (FBoW) representation for text that contains all the words in the vocabulary simultaneously but with different degrees of membership, which are derived from similarities between word vectors.

We show that max-pooled word vectors are only a special case of fuzzy BoW and should be compared via fuzzy Jaccard index rather than cosine similarity.

Finally, we propose DynaMax, a completely unsupervised and non-parametric similarity measure that dynamically extracts and max-pools good features depending on the sentence pair.

This method is both efficient and easy to implement, yet outperforms current baselines on STS tasks by a large margin and is even competitive with supervised word vectors trained to directly optimise cosine similarity.

Natural languages are able to encode sentences with similar meanings using very different vocabulary and grammatical constructs, which makes determining the semantic similarity between pieces of text a challenge.

It is common to cast semantic similarity between sentences as the proximity of their vector representations.

More than half a century since it was first proposed, the Bag-of-Words (BoW) representation (Harris, 1954; BID47 BID37 remains a popular baseline across machine learning (ML), natural language processing (NLP), and information retrieval (IR) communities.

In recent years, however, BoW was largely eclipsed by representations learned through neural networks, ranging from shallow BID36 BID21 to recurrent BID12 BID53 , recursive BID51 BID55 , convolutional BID30 BID32 , self-attentive BID57 BID9 and hybrid architectures BID19 BID56 BID66 .Interestingly, BID5 showed that averaged word vectors BID38 BID44 BID6 BID29 weighted with the Smooth Inverse Frequency (SIF) scheme and followed by a Principal Component Analysis (PCA) post-processing procedure were a formidable baseline for Semantic Textual Similarity (STS) tasks, outperforming deep representations.

Furthermore, BID59 and BID58 showed that averaged word vectors trained supervised on large corpora of paraphrases achieve state-of-the-art results, outperforming even the supervised systems trained directly on STS.Inspired by these insights, we push the boundaries of word vectors even further.

We propose a novel fuzzy bag-of-words (FBoW) representation for text.

Unlike classical BoW, fuzzy BoW contains all the words in the vocabulary simultaneously but with different degrees of membership, which are derived from similarities between word vectors.

Next, we show that max-pooled word vectors are a special case of fuzzy BoW. Max-pooling significantly outperforms averaging on standard benchmarks when word vectors are trained unsupervised.

Since max-pooled vectors are just a special case of fuzzy BoW, we show that the fuzzy Jaccard index is a more suitable alternative to cosine similarity for comparing these representations.

By contrast, the fuzzy Jaccard index completely fails for averaged word vectors as there is no connection between the two.

The max-pooling operation is commonplace throughout NLP and has been successfully used to extract features in supervised systems BID10 BID32 BID31 BID13 BID12 BID15 ; however, to the best of our knowledge, the present work is the first to study max-pooling of pre-trained word embeddings in isolation and to suggest theoretical underpinnings behind this operation.

Finally, we propose DynaMax, a completely unsupervised and non-parametric similarity measure that dynamically extracts and max-pools good features depending on the sentence pair.

DynaMax outperforms averaged word vector with cosine similarity on every benchmark STS task when word vectors are trained unsupervised.

It even performs comparably to BID58 's vectors under cosine similarity, which is a striking result as the latter are in fact trained supervised to directly optimise cosine similarity between paraphrases, while our approach is completely unrelated to that objective.

We believe this makes DynaMax a strong baseline that future algorithms should aim to beat in order to justify more complicated approaches to semantic similarity.

As an additional contribution, we conduct significance analysis of our results.

We found that recent literature on STS tends to apply unspecified or inappropriate parametric tests, or leave out significance analysis altogether in the majority of cases.

By contrast, we rely on nonparametric approaches with much milder assumptions on the test statistic; specifically, we construct bias-corrected and accelerated (BCa) bootstrap confidence intervals BID17 for the delta in performance between two systems.

We are not aware of any prior works that apply such methodology to STS benchmarks and hope the community finds our analysis to be a good starting point for conducting thorough significance testing on these types of experiments.

The bag-of-words (BoW) model of representing text remains a popular baseline across ML, NLP, and IR communities.

BoW, in fact, is an extension of a simpler set-of-words (SoW) model.

SoW treats sentences as sets, whereas BoW treats them as multisets (bags) and so additionally captures how many times a word occurs in a sentence.

Just like with any set, we can immediately compare SoW or BoW using set similarity measures (SSMs), such as DISPLAYFORM0 These coefficients usually follow the pattern #{shared elements} #{total elements} .

From this definition, it is clear that sets with no shared elements have a similarity of 0, which is undesirable in NLP as sentences with completely different words can still share the same meaning.

But can we do better?For concreteness, let's say we want to compare two sentences corresponding to the sets A = {'he', 'has', 'a', 'cat'} and B = {'she', 'had', 'one', 'dog'}. The situation here is that A ∩ B = ∅ and so their similarity according to any SSM is 0.

Yet, both A and B describe pet ownership and should be at least somewhat similar.

If a set contains the word 'cat', it should also contain a bit of 'pet', a bit of 'animal', also a little bit of 'tiger' but perhaps not too much of an 'airplane'.

If both A and B contained 'pet', 'animal', etc. to some degree, they would have a non-zero similarity.

This intuition is the main idea behind fuzzy sets: a fuzzy set includes all words in the vocabulary simultaneously, just with different degrees of membership.

This generalises classical sets where a word either belongs to a set or it doesn't.

We can easily convert a singleton set such as {'cat'} into a fuzzy set using a similarity function sim(w i , w j ) between words.

We simply compute the similarities between 'cat' and all the words w j in the vocabulary and treat those values as membership degrees.

As an example, the set {'cat'} really becomes {'cat' : 1, 'pet' : 0.9, 'animal' : 0.85, . . . , 'airplane' : 0.05, . . .}

Fuzzifying singleton sets is straightforward, but how do we go about fuzzifying the entire sentence {'he', 'has', 'a', 'cat'}?

Just as we use the classical union operation ∪ to build bigger sets from smaller ones, we use the fuzzy union to do the same but for fuzzy sets.

The membership degree of a word in the fuzzy union is determined as the maximum membership degree of that word among each of the fuzzy sets we want to unite.

This might sound somewhat arbitrary: after all, why max and not, say, sum or average?

We explain the rationale in Section 2.1; and in fact, we use the max for the classical union all the time without ever noticing it.

Indeed, {'cat'} ∪ {'cat'} = {'cat'} and not {'cat' : 2}. This is simply because we computed max(1, 1) = 1 and not sum(1, 1) = 2.

Similarly {'cat'} ∪ ∅ = {'cat'} since max(1, 0) = 1 and not avg(1, 0) = 1/2.The key insight here is the following.

An object that assigns the degrees of membership to words in a fuzzy set is called the membership function.

Each word defines a membership function, and even though 'cat' and 'dog' are different, they are semantically similar (in terms of cosine similarity between their word vectors, for example) and as such give rise to very similar membership functions.

This functional proximity will propagate into the SSMs, thus rendering them a much more realistic model for capturing semantic similarity between sentences.

To actually compute the fuzzy SSMs, we need just a few basic tools from fuzzy set theory, all of which we briefly cover in the next section.

Fuzzy set theory BID63 is a well-established formalism that extends classical set theory by incorporating the idea that elements can have degrees of membership in a set.

Constrained by space, we define the bare minimum needed to compute the fuzzy set similarity measures and refer the reader to BID34 for a much richer introduction.

Definition: A set of all possible terms V = {w 1 , w 2 , . . .

, w N } that occur in a certain domain is called a universe.

DISPLAYFORM0 Notice how the above definition covers all the set-like objects we discussed so far.

If L = {0, 1}, then A is simply a classical set and µ is its indicator (characteristic) function.

If L = N ≥0 (non-negative integers), then A is a multiset (a bag) and µ is called a count (multiplicity) function.

In literature, A is called a fuzzy set when L = [0, 1].

However, we make no restrictions on the range and call A a fuzzy set even when L = R, i.e. all real numbers.

Definition: Let A = (V, µ) and B = (V, ν) be two fuzzy sets.

The union of A and B is a fuzzy set A ∪ B = (V, max(µ, ν)).

The intersection of A and B is a fuzzy set A ∩ B = (V, min(µ, ν)).Interestingly, there are many other choices for the union and intersection operations in fuzzy set theory.

However, only the max-min pair makes these operations idempotent, i.e. such that A∪A = A and A ∩ A = A, just as in the classical set theory.

By contrast, it is not hard to verify that neither sum nor average satisfy the necessary axioms to qualify as a fuzzy union or intersection.

Definition: Let A = (V, µ) be a fuzzy set.

The number |A| = w∈V µ(w) is called the cardinality of a fuzzy set.

Fuzzy set theory provides a powerful framework for reasoning about sets with uncertainty, but the specification of membership functions depends heavily on the domain.

In practice these can be designed by experts or learned from data; below we describe a way of generating membership functions for text from word embeddings.

From the algorithmic point of view any bag-of-words is just a row vector.

The i-th term in the vocabulary has a corresponding N -dimensional one-hot encoding e (i) .

The vectors e (i) are orthonormal and in totality form the standard basis of R N .

The BoW vector for a sentence S is simply DISPLAYFORM0 , where c i is the count of the word w i in S.The first step in creating the fuzzy BoW representation is to convert every term vector e (i) into a membership vector µ (i) .

It really is the same as converting a singleton set {w i } into a fuzzy set.

We call this operation 'word fuzzification', and in the matrix form it is simply written as Algorithm 1 DynaMax-Jaccard Input: Word embeddings for the first sentence DISPLAYFORM1 Input: Word embeddings for the second sentence DISPLAYFORM2 Input: A vector with all zeros z ∈ R

Output: DISPLAYFORM0 Here W ∈ R N ×d is the word embedding matrix and U ∈ R K×d is the 'universe' matrix.

Let us dissect the above expression.

First, we convert a one-hot vector into a word embedding w (i) = e (i) W .

This is just an embedding lookup and is exactly the same as the embedding layer in neural networks.

Next, we compute a vector of similarities DISPLAYFORM1 and all the K vectors in the universe.

The most sensible choice for the universe matrix is the word embedding matrix itself, i.e. U = W .

In that case, the membership vector µ (i) has the same dimensionality as e (i) but contains similarities between the word w i and every word in the vocabulary (including itself).The second step is to combine all µ (i) back into a sentence membership vector µ s .

At this point, it's very tempting to just sum or average over all DISPLAYFORM2 .

But we remember: in fuzzy set theory the union of the membership vectors is realised by the element-wise max-pooling.

In other words, we don't take the average but max-pool instead: DISPLAYFORM3 Here the max returns a vector where each dimension contains the maximum value along that dimension across all N input vectors.

In NLP this is also known as max-over-time pooling BID10 .

Note that any given sentence S usually contains only a small portion of the total vocabulary and so most word counts c i will be 0.

If the count c i is 0, then we have no need for µ (i) and can avoid a lot of useless computations, though we must remember to include the zero vector in the max-pooling operation.

We call the sentence membership vector µ S the fuzzy bag-of-words (FBoW) and the procedure that converts classical BoW b S into fuzzy BoW µ S the 'sentence fuzzification'.

Suppose we have two fuzzy BoW µ A and µ B .

How can we compare them?

Since FBoW are just vectors, we can use the standard cosine similarity cos(µ A , µ B ).

On the other hand, FBoW are also fuzzy sets and as such can be compared via fuzzy SSMs.

We simply copy the definitions of fuzzy union, intersection and cardinality from Section 2.1 and write down the fuzzy Jaccard index: DISPLAYFORM0 .Exactly the same can be repeated for other SSMs.

In practice we found their performance to be almost equivalent but always better than standard cosine similarity (see Appendix B).

So far we considered the universe and the word embedding matrix to be the same, i.e. U = W .

This means any FBoW µ S contains similarities to all the words in the vocabulary and has exactly the same dimensionality as the original BoW b S .

Unlike BoW, however, FBoW is almost never sparse.

This motivates us to choose the matrix U with fewer rows that W .

For example, the top principal axes of W could work.

Alternatively, we could cluster W into k clusters and keep the centroids.

Of course, the rows of such U are no longer word vectors but instead some abstract entities.

A more radical but completely non-parametric solution is to choose U = I, where I ∈ R d×d is just the identity matrix.

Then the word fuzzifier reduces to a word embedding lookup: DISPLAYFORM0 The sentence fuzzifier then simply max-pools all the word embeddings found in the sentence: DISPLAYFORM1 From this we see that max-pooled word vectors are only a special case of fuzzy BoW. Remarkably, when word vectors are trained unsupervised, this simple representation combined with the fuzzy Jaccard index is already a stronger baseline for semantic textual similarity than the averaged word vector with cosine similarity, as we will see in Section 4.More importantly, the fuzzy Jaccard index works for max-pooled word vectors but completely fails for averaged word vectors.

This empirically validates the connection between fuzzy BoW representations and the max-pooling operation described above.

From the linear-algebraic point of view, fuzzy BoW is really the same as projecting word embeddings on a subspace of R d spanned by the rows of U , followed by max-pooling of the features extracted by this projection.

A fair question then is the following.

If we want to compare two sentences, what subspace should we project on?

It turns out that if we take word embeddings for the first sentence and the second sentence and stack them into matrix U , this seems to be a sufficient space to extract all the features needed for semantic similarity.

We noticed this empirically, and while some other choices of U do give better results, finding a principled way to construct them remains future work.

The matrix U is not static any more but instead changes dynamically depending on the sentence pair.

We call this approach Dynamic Max or DynaMax and provide pseudocode in Algorithm 1.

Just as SoW is a special case of BoW, we can build the fuzzy set-of-words (FSoW) where the word counts c i are binary.

The performance of FSoW and FBoW is comparable, with FBoW being marginally better.

For simplicity, we implement FSoW in Algorithm 1 and in all our experiments.

As evident from Equation FORMULA5 , we use dot product as opposed to (scaled or clipped) cosine similarity for the membership functions.

This is a reasonable choice as most unsupervised and some supervised word vectors maximise dot products in their objectives.

For further analysis, see Appendix A.

Any method that casts semantic similarity between sentences as the proximity of their vector representations is related to our work.

Among those, the ones that strengthen bag-of-words by incorporating the sense of similarity between individual words are the most relevant.

The standard Vector Space Model (VSM) basis e (i) is orthonormal and so the BoW model treats all words as equally different.

BID50 proposed the 'soft cosine measure' to alleviate this issue.

They build a non-orthogonal basis f (i) where cos(f (i) , f (j) ) = sim(w i , w j ), i.e. the cosine similarity between the basis vectors is given by similarity between words.

Next, they rewrite BoW in comparing different combinations of fuzzy BoW representation (either averaged or max-pooled, or the DynaMax approach) and similarity measure (either cosine or Jaccard).

The bolded methods are ones proposed in the present work.

Note that averaged vectors with Jaccard similarity are not included in these plots, as they consistently perform 20-50 points worse than other methods; this is predicted by our analysis as averaging is not an appropriate union operation in fuzzy set theory.

In virtually every case, max-pooled with cosine outperforms averaged with cosine, which is in turn outperformed by max-pooled and DynaMax with Jaccard.

An exception to the trend is STS13, for which the SMT subtask dataset is no longer publicly available; this may have impacted the performance when averaged over different types of subtasks.terms of f (i) and compute cosine similarity between transformed representations.

However, when cos( DISPLAYFORM0 , where w i , w j are word embeddings, their approach is equivalent to cosine similarity between averaged word embeddings, i.e. the standard baseline.

BID35 consider L1-normalised bags-of-words (nBoW) and view them as a probability distributions over words.

They propose the Word Mover's Distance (WMD) as a special case of the Earth Mover's Distance (EMD) between nBoW with the cost matrix given by pairwise Euclidean distances between word embeddings.

As such, WMD does not build any new representations but puts a lot of structure into the distance between BoW. BID65 proposed an alternative version of fuzzy BoW that is conceptually similar to ours but executed very differently.

They use clipped cosine similarity between word embeddings to compute the membership values in the word fuzzification step.

We use dot product not only because it is theoretically more general but also because dot product leads to significant improvements on the benchmarks.

More importantly, however, their sentence fuzzification step uses sum to aggregate word membership vectors into a sentence membership vector.

We argue that max-pooling is a better choice because it corresponds to the fuzzy union.

Had we used the sum, the representation would have really reduced to a (projected) summed word vector.

Lastly, they use FBoW as features for a supervised model but stop short of considering any fuzzy similarity measures, such as fuzzy Jaccard index.

BID24 BID32 BID0 proposed and developed soft cardinality as a generalisation to the classical set cardinality.

In their framework set membership is crisp, just as in classical set theory.

However, once the words are in a set, their contribution to the overall cardinality depends on how similar they are to each other.

The intuition is that the set A = {'lion', 'tiger', 'leopard'} should have cardinality much less than 3, because A contains very similar elements.

Likewise, the set B = {'lion', 'airplane', 'carrot'} deserves a cardinality closer to 3.

We see that the soft cardinality framework is very different from our approach, as it 'does not consider uncertainty in the membership of a particular element; only uncertainty as to the contribution of an element to the cardinality of the set' BID24 .

To evaluate the proposed similarity measures we set up a series of experiments on the established STS tasks, part of the SemEval shared task series 2012-2016 BID1 BID32 BID0 BID4 BID7 .

The idea behind the STS benchmarks is to measure comparing other BoW-based methods to ones using fuzzy Jaccard similarity.

The bolded methods are ones proposed in the present work.

We observe that even classical crisp Jaccard is a fairly reasonable baseline, but it is greatly improved by the fuzzy set treatment.

Both max-pooled word vectors with Jaccard and DynaMax outperform the other methods by a comfortable margin, and the max-pooled version in particular performs astonishingly well given its great simplicity.how well the semantic similarity scores computed by a system (algorithm) correlate with human judgements.

Each year's STS task itself consists of several subtasks.

By convention, we report the mean Pearson correlation between system and human scores, where the mean is taken across all the subtasks in a given year.

Our implementation wraps the SentEval toolkit BID11 and is available on GitHub 1 .

We also rely on the following publicly available word embeddings: GloVe BID44 trained on Common Crawl (840B tokens); fastText BID6 ) trained on Common Crawl (600B tokens); word2vec BID39 ;c) trained on Google News, CoNLL BID64 , and Book Corpus ; and several types of supervised paraphrastic vectors -PSL BID59 , PP-XXL BID60 , and PNMT BID58 .We estimated word frequencies on an English Wikipedia dump dated July 1 st 2017 and calculated word weights using the same approach and parameters as in BID5 .

Note that these weights can in fact be derived from word vectors and frequencies alone rather than being inferred from the validation set BID18 , making our techniques fully unsupervised.

Finally, as the STS'13 SMT dataset is no longer publicly available, the mean Pearson correlations reported in our experiments involving this task have been re-calculated accordingly.

We first ran a set of experiments validating the insights and derivations described in Section 2.

These results are presented in FIG0 .

The main takeaways are the following:• Max-pooled word vectors outperform averaged word vectors in most tasks.• Max-pooled vectors with cosine similarity perform worse than max-pooled vectors with fuzzy Jaccard similarity.

This supports our derivation of max-pooled vectors as a special case of fuzzy BoW, which thus should be compared via fuzzy set similarity measures and not cosine similarity (which would be an arbitrary choice).• Averaged vectors with fuzzy Jaccard similarity completely fail.

This is because fuzzy set theory tells us that the average is not a valid fuzzy union operation, so a fuzzy set similarity is not appropriate for this representation.• DynaMax shows the best performance across all tasks, possibly thanks to its superior ability to extract and max-pool good features from word vectors.

Next we ran experiments against some of the related methods described in Section 3, namely WMD BID35 and soft cardinality BID28 with clipped cosine similarity as an affinity function and the softness parameter p = 1.

From FIG1 , we see that even classical Jaccard index is a reasonable baseline, but fuzzy Jaccard especially in the DynaMax formulation handily outperforms comparable methods.

For context and completeness, we also compare against other popular sentence representations from the literature in TAB0 .

We include the following methods: BoW with ELMo embeddings BID54 .

Note that avg-cos refers to taking the average word vector and comparing by cosine similarity, and word2vec refers to the Google News version.

Clearly more sophisticated methods of computing sentence representations do not shine on the unsupervised STS tasks when compared to these simple BoW methods with high-quality word vectors and the appropriate similarity metric.

† indicates the only STS13 result (to our knowledge) that includes the SMT subtask.

BID46 , Skip-Thought , InferSent BID12 , Universal Sentence Encoder with DAN and Transformer BID8 , and STN multitask embeddings BID54 .

These experiments lead to an interesting observation:

• PNMT embeddings are the current state-of-the-art on STS tasks.

PP-XXL and PNMT were trained supervised to directly optimise cosine similarity between average word vectors on very large paraphrastic datasets.

By contrast, DynaMax is completely unrelated to the training objective of these vectors, yet has an equivalent performance.

Finally, another well-known and high-performing simple baseline was proposed by BID5 .

However, as also noted by BID41 , this method is still offline because it computes the sentence embeddings for the entire dataset, then performs PCA and removes the top principal component.

While their method makes more assumptions than ours, nonetheless we make a head-to-head comparison with them in TAB2 using the same word vectors as in BID5 , showing that DynaMax is still quite competitive.

To strengthen our empirical findings, we provide ablation studies for DynaMax in Appendix C, showing that the different components of the algorithm each contribute to its strong performance.

We also conduct significance testing in Appendix D by constructing bias-corrected and accelerated (BCa) bootstrap confidence intervals BID17 for the delta in performance between two algorithms.

This constitutes, to the best of our knowledge, the first attempt to study statistical significance on the STS benchmarks with this type of non-parametric analysis that respects the statistical peculiarities of these datasets.

In this work we combine word embeddings with classic BoW representations using fuzzy set theory.

We show that max-pooled word vectors are a special case of FBoW, which implies that they should be compared via the fuzzy Jaccard index rather than the more standard cosine similarity.

We also present a simple and novel algorithm, DynaMax, which corresponds to projecting word vectors onto a subspace dynamically generated by the given sentences before max-pooling over the features.

DynaMax outperforms averaged word vectors compared with cosine similarity on every benchmark STS task when word vectors are trained unsupervised.

It even performs comparably to supervised vectors that directly optimise cosine similarity between paraphrases, despite being completely unrelated to that objective.

Both max-pooled vectors and DynaMax constitute strong baselines for further studies in the area of sentence representations.

Yet, these methods are not limited to NLP and word embeddings, but can in fact be used in any setting where one needs to compute similarity between sets of elements that have rich vector representations.

We hope to have demonstrated the benefits of experimenting more with similarity metrics based on the building blocks of meaning such as words, rather than complex representations of the final objects such as sentences.

In the word fuzzification step the membership values for a word w are obtained through a similarity function sim (w, u (j) ) between the word embedding w and the rows of the universe matrix U , i.e. DISPLAYFORM0 In Section 2.2, sim(w, u (j) ) was the dot product w · u (j) and we could simply write µ = wU T .

There are several reasons why we chose a similarity function that takes values in R as opposed to DISPLAYFORM1 First, we can always map the membership values from R to (0, 1) and vice versa using, e.g. the logistic function σ(x) = 1 1+e −ax with an appropriate scaling factor a > 0.

Intuitively, large negative membership values would imply the element is really not in the set and large positive values mean it is really in the set.

Of course, here both 'large' and 'really' depend on the scaling factor a. In any case, we see that the choice of R vs. [0, 1] is not very important mathematically.

Interestingly, since we always max-pool with a zero vector, fuzzy BoW will not contain any negative membership values.

This was not our intention, just a by-product of the model.

For completeness, let us insist on the range [0, 1] and choose sim (w, u (j) ) to be the clipped cosine similarity max (0, cos(w, u (j) )).

This is in fact equivalent to simply normalising the word vectors.

Indeed, the dot product and cosine similarity become the same after normalisation, and max-pooling with the zero vector removes all the negative values, so the resulting representation is guaranteed to be a [0, 1]-fuzzy set.

Our results for normalised word vectors are presented in TAB3 .After comparing TAB0 we can draw two conclusions.

Namely, DynaMax still outperforms avg-cos by a large margin even when word vectors are normalised.

However, normalisation hurts both approaches and should generally be avoided.

This is not surprising since the length of word vectors is correlated with word importance, so normalisation essentially makes all words equally important BID48 .

In Section 2 we mentioned several set similarity measures such as Jaccard BID23 , OtsukaOchiai (Otsuka, 1936; BID42 and Sørensen-Dice (Dice, 1945; BID52 coefficients.

Here in TAB4 , we show that fuzzy versions of the above coefficients have almost identical performance, thus confirming that our results are in no way specific to the Jaccard index.

Table 5 : Mean Pearson correlation on STS tasks for the ablation studies.

As described in Appendix C, it is clear that the three components of the algorithm -the dynamic universe, the max-pooling operation, and the fuzzy Jaccard index -all contribute to the strong performance of DynaMax-Jaccard.

The DynaMax-Jaccard similarity (Algorithm 1) consists of three components: the dynamic universe, the max-pooling operation, and the fuzzy Jaccard index.

As with any algorithm, it is very important to track the sources of improvements.

Consequently, we perform a series of ablation studies in order to isolate the contribution of each component.

For brevity, we focus on fastText because it produced the strongest results for both the DynaMax and the baseline FIG0 ).The results of the ablation study are presented in Table 5 .

First, we show that the dynamic universe is superior to other sensible choices, such as the identity and random 300 × 300 projection with components drawn from N (0, 1).

Next, we show that the fuzzy Jaccard index beats the standard cosine similarity on 4 out 5 benchmarks.

Finally, we find that max considerably outperforms other pooling operations such as averaging, sum and min.

We conclude that all three components of DynaMax are very important.

It is clear that max-pooling is the top contributing factor, followed by the dynamic universe and the fuzzy Jaccard index, whose contributions are roughly equal.

As discussed in Section 4, the core idea behind the STS benchmarks is to measure how well the semantic similarity scores computed by a system (algorithm) correlate with human judgements.

In this section we provide detailed results and significance analysis for all 24 STS subtasks.

Our approach can be formally summarised as follows.

We assume that the human scores H, the system scores A and the baseline system scores B jointly come from some trivariate distribution P (H, A, B), which is specific to each subtask.

To compare the performance of two systems, we compute the sample Pearson correlation coefficients r AH and r BH .

Since these correlations share the variable H, they are themselves dependent.

There are several parametric tests for the difference between dependent correlations; however, their appropriateness beyond the assumptions of normality remains an active area of research BID22 BID62 BID61 .

The distributions of the human scores in the STS tasks are generally not normal; what's more, they vary greatly depending on the subtask (some are multimodal, others are skewed, etc.).Fortunately, nonparametric resampling-based approaches, such as bootstrap BID16 , present an attractive alternative to parametric tests when the distribution of the test statistic is unknown.

In our case, the statistic is simply the difference between two correlations∆ = r AH − r BH .

The main idea behind bootstrap is intuitive and elegant: just like a sample is drawn from the population, a large number of 'bootstrap' samples can be drawn from the actual sample.

In our case, the dataset consists of triplets DISPLAYFORM0 .

Each bootstrap sample is a result of drawing M data points from D with replacement.

Finally, we approximate the distribution of ∆ by evaluating it on a large number of bootstrap samples, in our case ten thousand.

We use this information to construct bias-corrected and accelerated (BCa) 95% confidence intervals for ∆. BCa BID17 ) is a fairly advanced second-order method that accounts for bias and skewness in the bootstrapped distributions, effects we did observe to a small degree in certain subtasks.

Once we have the confidence interval for ∆, the decision rule is then simple: if zero is inside the interval, then the difference between correlations is not significant.

Inversely, if zero is outside, we may conclude that the two approaches lead to statistically different results.

The location of the interval further tells us which one performs better.

The results are presented in TAB6 .

In summary, out of 72 experiments we significantly outperform the baseline in 56 (77.8%) and underperform in only one (1.39%), while in the remaining 15 (20.8%) the differences are nonsignificant.

We hope our analysis is useful to the community and will serve as a good starting point for conducting thorough significance testing on the current as well as future STS benchmarks.

@highlight

Max-pooled word vectors with fuzzy Jaccard set similarity are an extremely competitive baseline for semantic similarity; we propose a simple dynamic variant that performs even better.