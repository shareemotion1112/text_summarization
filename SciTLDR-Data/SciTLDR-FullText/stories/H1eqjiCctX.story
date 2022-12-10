Word embedding is a powerful tool in natural language processing.

In this paper we consider the problem of word embedding composition \--- given vector representations of two words, compute a vector for the entire phrase.

We give a generative model that can capture specific syntactic relations between words.

Under our model, we prove that the correlations between three words (measured by their PMI) form a tensor that has an approximate low rank Tucker decomposition.

The result of the Tucker decomposition gives the word embeddings as well as a core tensor, which can be used to produce better compositions of the word embeddings.

We also complement our theoretical results with experiments that verify our assumptions, and demonstrate the effectiveness of the new composition method.

Word embeddings have become one of the most popular techniques in natural language processing.

A word embedding maps each word in the vocabulary to a low dimensional vector.

Several algorithms (e.g., Mikolov et al. (2013) ; Pennington et al. (2014) ) can produce word embedding vectors whose distances or inner-products capture semantic relationships between words.

The vector representations are useful for solving many NLP tasks, such as analogy tasks (Mikolov et al., 2013) or serving as features for supervised learning problems (Maas et al., 2011)

.While word embeddings are good at capturing the semantic information of a single word, a key challenge is the problem of composition: how to combine the embeddings of two co-occurring, syntactically related words to an embedding of the entire phrase.

In practice composition is often done by simply adding the embeddings of the two words, but this may not be appropriate when the combined meaning of the two words differ significantly from the meaning of individual words (e.g., "complex number" should not just be "complex"+"number").In this paper, we try to learn a model for word embeddings that incorporates syntactic information and naturally leads to better compositions for syntactically related word pairs.

Our model is motivated by the principled approach for understanding word embeddings initiated by Arora et al. (2015) , and models for composition similar to Coecke et al. (2010) .

Arora et al. (2015) gave a generative model (RAND-WALK) for word embeddings, and showed several previous algorithms can be interpreted as finding the hidden parameters of this model.

However, the RAND-WALK model does not treat syntactically related word-pairs differently from other word pairs.

We give a generative model called syntactic RAND-WALK (see Section 3) that is capable of capturing specific syntactic relations (e.g., adjective-noun or verb-object pairs).

Taking adjective-noun pairs as an example, previous works (Socher et al., 2012; Baroni & Zamparelli, 2010; Maillard & Clark, 2015) have tried to model the adjective as a linear operator (a matrix) that can act on the embedding of the noun.

However, this would require learning a d × d matrix for each adjective while the normal embedding only has dimension d. In our model, we use a core tensor T ∈ R d×d×d to capture the relations between a pair of words and its context.

In particular, using the tensor T and the word embedding for the adjective, it is possible to define a matrix for the adjective that can be used as an operator on the embedding of the noun.

Therefore our model allows the same interpretations as many previous models while having much fewer parameters to train.

One salient feature of our model is that it makes good use of high order statistics.

Standard word embeddings are based on the observation that the semantic information of a word can be captured by words that appear close to it.

Hence most algorithms use pairwise co-occurrence between words to learn the embeddings.

However, for the composition problem, the phrase of interest already has two words, so it would be natural to consider co-occurrences between at least three words (the two words in the phrase and their neighbors).Based on the model, we can prove an elegant relationship between high order co-occurrences of words and the model parameters.

In particular, we show that if we measure the Pointwise Mutual Information (PMI) between three words, and form an n × n × n tensor that is indexed by three words a, b, w, then the tensor has a Tucker decomposition that exactly matches our core tensor T and the word embeddings (see Section 2, Theorem 1, and Corollary 1).

This suggests a natural way of learning our model using a tensor decomposition algorithm.

Our model also allows us to approach the composition problem with more theoretical insights.

Based on our model, if words a, b have the particular syntactic relationships we are modeling, their composition will be a vector v a + v b + T (v a , v b , ·).

Here v a , v b are the embeddings for word a and b, and the tensor gives an additional correction term.

By choosing different core tensors it is possible to recover many previous composition methods.

We discuss this further in Section 3.Finally, we train our new model on a large corpus and give experimental evaluations.

In the experiments, we show that the model learned satisfies the new assumptions that we need.

We also give both qualitative and quantitative results for the new embeddings.

Our embeddings and the novel composition method can capture the specific meaning of adjective-noun phrases in a way that is impossible by simply "adding" the meaning of the individual words.

Quantitative experiment also shows that our composition vector are better correlated with humans on a phrase similarity task.

Syntax and word embeddings Many well-known word embedding methods (e.g., Pennington et al. (2014) ; Mikolov et al. (2013) ) don't explicitly utilize or model syntactic structure within text.

Andreas & Klein (2014) find that such syntax-blind word embeddings fail to capture syntactic information above and beyond what a statistical parser can obtain, suggesting that more work is required to build syntax into word embeddings.

Several syntax-aware embedding algorithms have been proposed to address this.

Levy & Goldberg (2014a) propose a syntax-oriented variant of the well-known skip-gram algorithm of Mikolov et al. (2013) , using contexts generated from syntactic dependency-based contexts obtained with a parser.

Cheng & Kartsaklis (2015) build syntax-awareness into a neural network model for word embeddings by indroducing a negative set of samples in which the order of the context words is shuffled, in hopes that the syntactic elements which are sensitive to word order will be captured.

Word embedding composition Several works have addressed the problem of composition for word embeddings.

On the theoretical side, Gittens et al. (2017) give a theoretical justification for additive embedding composition in word models that satisfy certain assumptions, such as the skipgram model, but these assumptions don't address syntax explicitly.

Coecke et al. (2010) present a mathematical framework for reasoning about syntax-aware word embedding composition that motivated our syntactic RAND-WALK model.

Our new contribution is a concrete and practical learning algorithm with theoretical guarantees.

Mitchell & Lapata (2008; 2010) explore various composition methods that involve both additive and multiplicative interactions between the component embeddings, but some of these are limited by the need to learn additional parameters post-hoc in a supervised fashion.

Guevara (2010) get around this drawback by first training word embeddings for each word and also for tokenized adjective-noun pairs.

Then, the composition model is trained by using the constituent adjective and noun embeddings as input and the adjective-noun token embedding as the predictive target.

Maillard & Clark (2015) treat adjectives as matrices and nouns as vectors, so that the composition of an adjective and noun is just matrix-vector multiplication.

The matrices and vectors are learned through an extension of the skip-gram model with negative sampling.

In contrast to these approaches, our model gives rise to a syntax-aware composition function, which can be learned along with the word embeddings in an unsupervised fashion, and which generalizes many previous composition methods (see Section 3.3 for more discussion).Tensor factorization for word embeddings As Levy & Goldberg (2014b) and Li et al. (2015) point out, some popular word embedding methods are closely connected matrix factorization problems involving pointwise mutual information (PMI) and word-word co-occurrences.

It is natural to consider generalizing this basic approach to tensor decomposition.

Sharan & Valiant (2017) demonstrate this technique by performing a CP decomposition on triple word co-occurrence counts.

Bailey & Aeron (2017) explore this idea further by defining a third-order generalization of PMI, and then performing a symmetric CP decomposition on the resulting tensor.

In contrast to these recent works, our approach arives naturally at the more general Tucker decomposition due to the syntactic structure in our model.

Our model also suggests a different (yet still common) definition of third-order PMI.

Notation For a vector v, we use v to denote its Euclidean norm.

For vectors u, v we use u, v to denote their inner-product.

For a matrix M , we use M to denote its spectral norm, DISPLAYFORM0 to denote its Frobenius norm, and M i,: to denote it's i-th row.

In this paper, we will also often deal with 3rd order tensors, which are just three-way indexed arrays.

We use ⊗ to denote the tensor product: DISPLAYFORM1 Tensor basics Just as matrices are often viewed as bilinear functions, third order tensors can be interpreted as trilinear functions over three vectors.

Concretely, let T be a d × d × d tensor, and let x, y, z ∈ R d .

We define the scalar T (x, y, z) ∈ R as follows DISPLAYFORM2 This operation is linear in x, y and z. Analogous to applying a matrix M to a vector v (with the result vector M v), we can also apply a tensor T to one or two vectors, resulting in a matrix and a vector, respectively: DISPLAYFORM3 We will make use of the simple facts that z, T (x, y, ·) = T (x, y, z) and [T (x, ·, ·)]

y = T (x, y, ·).Tensor decompositions Unlike matrices, there are several different definitions for the rank of a tensor.

In this paper we mostly use the notion of Tucker rank Tucker (1966) .

A tensor T ∈ R n×n×n has Tucker rank d, if there exists a core tensor S ∈ R d×d×d and matrices A, B, C ∈ R n×d such that DISPLAYFORM4 The equation above is also called a Tucker decomposition of the tensor T .

The Tucker decomposition for a tensor can be computed efficiently.

When the core tensor S is restricted to a diagonal tensor (only nonzero at entries S i,i,i ), the decomposition is called a CP decomposition Carroll & Chang (1970); Harshman (1970) which can also be written as DISPLAYFORM5 .

In this case, the tensor T is the sum of d rank-1 tensors (A i,: ⊗ B i,: ⊗ C i,: ).

However, unlike matrix factorizations and the Tucker decomposition, the CP decomposition of a tensor is hard to compute in the general case (Håstad, 1990; Hillar & Lim, 2013) .

Later in Section 4 we will also see why our model for syntactic word embeddings naturally leads to a Tucker decomposition.

In this section, we introduce our syntactic RAND-WALK model and present formulas for inference in the model.

We also derive a novel composition technique that emerges from the model.

We first briefly review the RAND-WALK model (Arora et al., 2015) .

In this model, a corpus of text is considered as a sequence of random variables w 1 , w 2 , w 3 , . . ., where w t takes values in a vocabulary V of n words.

Each word w ∈ V has a word embedding v w ∈ R d .

The prior for the word embeddings is v w = s ·v, where s is a positive bounded scalar random variable with constant expectation τ and upper bound κ, andv ∼ N (0, I).The distribution of each w t is determined in part by a random walk {c t ∈ R d | t = 1, 2, 3 . . .}, where c t -called a discourse vector -represents the topic of the text at position t.

This random walk is slow-moving in the sense that c t+1 − c t is small, but mixes quickly to a stationary distribution that is uniform on the unit sphere, which we denote by C.Let C denote the sequence of discourse vectors, and let V denote the set of word embeddings.

Given these latent variables, the model specifies the following conditional probability distribution: DISPLAYFORM0 The graphical model depiction of RAND-WALK is shown in FIG0 .

One limitation of RAND-WALK is that it can't deal with syntactic relationships between words.

Observe that conditioned on c t and V , w t is independent of the other words in the text.

However, in natural language, words can exhibit more complex dependencies, e.g. adjective-noun pairs, subject-verb-object triples, and other syntactic or grammatical structures.

In our syntactic RAND-WALK model, we start to address this issue by introducing direct pairwise word dependencies in the model.

When there is a direct dependence between two words, we call the two words a syntactic word pair.

In RAND-WALK, the interaction between a word embedding v and a discourse vector c is mediated by their inner product v, c .

When modeling a syntactic word pair, we need to mediate the interaction between three quantities, namely a discourse vector c and the word embeddings v and v of the two relevant words.

A natural generalization is to use a trilinear form defined by a tensor T , i.e. DISPLAYFORM0 Here, T ∈ R d×d×d is also a latent random variable, which we call the composition tensor.

We model a syntactic word pair as a single semantic unit within the text (e.g. in the case of adjectivenoun phrases).

We realize this choice by allowing each discourse vector c t to generate a pair of words w t , w t with some small probability p syn .

To generate a syntactic word pair w t , w t , we first generate a root word w t conditioned on c t with probability proportional to exp( c t , w t ), and then we draw w t from a conditional distribution defined as follows: DISPLAYFORM1 (2) Here exp( c t , v b ) would be proportional to the probability of generating word b in the original RAND-WALK model, without considering the syntactic relationship.

The additional term T (v a , v b , c t ) can be viewed as an adjustment based on the syntactic relationship.

We call this extended model Syntactic RAND-WALK.

FIG0 gives the graphical model depiction for a syntactic word pair, and we summarize the model below.

Definition 1 (Syntactic RAND-WALK model).

The model consists of the following:1.

Each word w in vocabulary has a corresponding embedding v w ∼ s ·v w , where s ∈ R ≥0 is bounded by κ and DISPLAYFORM2 2.

The sequence of discourse vectors c 1 , ..., c t are generated by a random walk on the unit sphere, c t − c t+1 ≤ w / √ d and the stationary distribution is uniform.3.

For each c t , with probability 1−p syn , it generates one word w t with probability proportional to exp( c t , v wt ).4.

For each c t , with probability p syn , it generates a syntactic pair w t , w t with probability proportional to exp( c t , v wt ) and exp( c t , DISPLAYFORM3

We now calculate the marginal probabilities of observing pairs and triples of words under the syntactic RAND-WALK model.

We will show that these marginal probabilities are closely related to the model parameters (word embeddings and the composition tensor).

All proofs in this section are deferred to supplementary material.

Throughout this section, we consider two adjacent context vectors c t and c t+1 , and condition on the event that c t generated a single word and c t+1 generated a syntactic pair 1 .

The main bottleneck in computing the marginal probabilities is that the conditional probailities specified in equations FORMULA6 and (2) are not normalized.

Indeed, for these equations to be exact, we would need to divide by the appropriate partition functions, namely Z ct := w∈V exp( v w , c t ) for the former and Z ct,a := w∈V exp( c t , v w + T (v a , v w , c t )) for the latter.

Fortunately, we show that under mild assumptions these quantities are highly concentrated.

To do that we need to control the norm of the composition tensor.

Definition 2.

The composition tensor T is (K, )-bounded, if for any word embedding v a , v b , we have DISPLAYFORM0 To make sure exp( c t , v w +T (v a , v w , c t )) are within reasonable ranges, the value K in this definition should be interpreted as an absolute constant (like 5, similar to previous constants κ and τ ).

Intuitively these conditions make sure that the effect of the tensor cannot be too large, while still making sure the tensor component T (v a , v b , c) can be comparable (or even larger than) v b , c .

We have not tried to optimize the log factors in the constraint for T (v a , ·, ·) + I 2 .Note that if the tensor component T (v a , ·, ·) has constant singular values (hence comparable to I), we know these conditions will be satisfied with K = O(1) and = O( DISPLAYFORM1 ).

Later in Section 5 we verify that the tensors we learned indeed satisfy this condition.

Now we are ready to state the concentration of partition functions:Lemma 1 (Concentration of partition functions).

For the syntactic RAND-WALK model, there exists a constant Z such that Pr DISPLAYFORM2 Furthermore, if the tensor T is (K, )-bounded, then for any fixed word a ∈ V , there exists a constant Z a such that Pr DISPLAYFORM3 Using this lemma, we can obtain simple expressions for co-occurrence probabilities.

In particular, for any fixed w, a, b ∈ V , we adopt the following notation: DISPLAYFORM4 Here in particular we use [a, b] to highlight the fact that a and b form a syntactic pair.

Note p(w, a) is the same as the co-occurrence probability of words w and a if both of them are the only word generated by the discourse vector.

Later we will also use p(w, b) to denote Pr[ DISPLAYFORM5 We also require two additional properties of the word embeddings, namely that they are norm-bounded above by some constant times DISPLAYFORM6 and that all partition functions are bounded below by a positive constant.

Both of these properties hold with high probability over the word embeddings provided n d log d and d log n, as shown in the following lemma: Lemma 2.

Assume that the composition tensor T is (K, )-bounded, where K is a constant.

With probability at least 1 − δ 1 − δ 2 over the word vectors, where DISPLAYFORM7 , there exist positive absolute constants γ and β such that v i ≤ κγ for each i ∈ V and Z c ≥ β and Z c,a ≥ β for any unit vector c ∈ R d and any word a ∈ V .We can now state the main result.

Theorem 1.

Suppose that the events referred to in Lemma 1 hold.

Then DISPLAYFORM8 DISPLAYFORM9 where is from the (K, )-boundedness of T and w is from Definition 1.

Our model suggests that the latent discourse vectors contain the meaning of the text at each location.

It is therefore reasonable to view the discourse vector c corresponding to a syntactic word pair (a, b) as a suitable representation for the phrase as a whole.

The posterior distribution of c given (a, b) satisfies DISPLAYFORM0 Since Pr[c t = c] is constant, and since Z c and Z c,a concentrate on values that don't depend on c, the MAP estimate of c given [a, b], which we denote byĉ, satisfieŝ DISPLAYFORM1 Hence, we arrive at our basic tensor composition: for a syntactic word pair (a, b), the composite embedding for the phrase is DISPLAYFORM2 Note that our composition involves the traditional additive composition DISPLAYFORM3 e. the composition tensor allows us to compactly associate a matrix with each word in the same vein as Maillard & Clark (2015) .

Depending on the actual value of T , the term T (v a , v b , ·) can also recover any manner of linear or multiplicative interactions between v a and v b , such as those proposed in Mitchell & Lapata (2010) .

In this section we discuss how to learn the parameters of the syntactic RAND-WALK model.

Theorem 1 provides key insights into the learning problem, since it relates joint probabilities between words (which can be estimated via co-occurrence counts) to the word embeddings and composition tensor.

By examining these equations, we can derive a particularly simple formula that captures these relationships.

To state this equation, we define the PMI for 3 words as DISPLAYFORM0 We note that this is just one possible generalization of pointwise mutual information (PMI) to several random variables, but in the context of our model, it is a very natural definition as all the partition numbers will be canceled out.

Indeed, as an immediate corollary of Theorem 1, we have Corollary 1.

Suppose that the events referred to in Lemma 1 hold.

Then for p same as Theorem 1 DISPLAYFORM1 That is, if we consider P M I3(a, b, w) as a n × n × n tensor, Equation equation 8 is exactly a Tucker decomposition of this tensor of Tucker rank d. Therefore, all the parameters of the syntactic RAND-WALK model can be obtained by finding the Tucker decomposition of the PMI3 tensor.

This equation also provides a theoretical motivation for using third-order pointwise mutual information in learning word embeddings.

We now discuss concrete details about our implementation of the learning algorithm.

Corpus.

We train our model using a February 2018 dump of the English Wikipedia.

The text is pre-processed to remove non-textual elements, stopwords, and rare words (words that appear less than 1000 within the corpus), resulting in a vocabulary of size 68,279.

We generate a matrix of word-word co-occurrence counts using a window size of 5.

To generate the tensors of adjective-noun-word and verb-object-word co-occurrence counts, we first run the Stanford Dependency Parser (Chen & Manning, 2014) on the corpus in order to identify all adjective-noun and verb-object word pairs, and then use context windows that don't cross sentence boundaries to populate the triple co-occurrence counts.

Training.

We first train the word embeddings according to the RAND-WALK model, following Arora et al. (2015) .

Using the learned word embeddings, we next train the composition tensor T via the following optimization problem DISPLAYFORM0 where X (a,b),w denotes the number of co-occurrences of word w with the syntactic word pair (a, b) (a denotes the noun/object) and f (x) = min(x, 100).

This objective function isn't precisely targeting the Tucker decomposition of the PMI3 tensor, but it is analogous to the training criterion used in Arora et al. (2015) , and can be viewed as a negative log-likelihood for the model.

To reduce the number of parameters, we constrain T to have CP rank 1000.

We also trained the embeddings and tensor jointly, but found that this approach yields very similar results.

In all cases, we utilize the Tensorflow framework BID0 with the Adam optimizer (Kingma & Ba, 2014) (using default parameters), and train for 1-5 epochs.

In this section, we verify and evaluate our model empirically on select qualitative and quantitative tasks.

In all of our experiments, we focus solely on syntactic word pairs formed by adjective-noun phrases, where the noun is considered the root word.

Arora et al. (2015) empirically verify the model assumptions of RAND-WALK, and since we trained our embeddings in the same way, we don't repeat their verifications here.

Instead, we verify two key properties of syntactic RAND-WALK.

We check the assumptions that the tensor T is (K, )-bounded.

Ranging over all adjective-noun pairs in the corpus, we find that DISPLAYFORM0 2 has mean 0.052 and maximum 0.248, DISPLAYFORM1 F has mean 1.61 and maximum 3.23, and DISPLAYFORM2 has mean 0.016 and maximum 0.25.

Each of these three quantities has a well-bounded mean, but T (v a , ·, ·) + I 2 has some larger outliers.

If we ignore the log factors (which are likely due to artifacts in the proof) in Definition 2, the tensor is (K, ) bounded for K = 4 and = 0.25.

In addition to Definition 2, we also directly check its implications: our model predicts that the partition functions Z c,a concentrate around their means.

To check this, given a noun a, we draw 1000 random vectors c from the unit sphere, and plot the histogram of Z c,a .Results for a few randomly selected words a are given in FIG2 .

All partition functions that we inspected exhibited good concentration.

We test the performance of our new composition for adjective-noun and verb-object pairs by looking for the words with closest embedding to the composed vector.

For a phrase (a, b), we compute c = v a + v b + T (v a , v b , ·), and then retrieve the words w whose embeddings v w have the largest cosine similarity to c. We compare our results to the additive composition method.

TAB0 show results for three adjective-noun and verb-object phrases.

In each case, the tensor composition is able to retrieve some words that are more specifically related to the phrase.

However, the tensor composition also sometimes retrieves words that seem unrelated to either word in the phrase.

We conjecture that this might be due to the sparseness of co-occurrence of three words.

We also observed cases where the tensor composition method was about on par with or inferior to the additive composition method for retrieving relevant words, particularly in the case of low-frequency phrases.

More results can be found in supplementary material.

Published as a conference paper at ICLR 2019 We also test our tensor composition method on a adjective-noun phrase similarity task using the dataset introduced by Mitchell & Lapata (2010) .

The data consists of 108 pairs each of adjective-noun and verb-object phrases that have been given similarity ratings by a group of 54 humans.

The task is to use the word embeddings to produce similarity scores that correlate well with the human scores; we use both the Spearman rank correlation and the Pearson correlation as evaluation metrics for this task.

We note that the human similarity judgments are somewhat noisy; intersubject agreement for the task is 0.52 as reported in Mitchell & Lapata (2010) .Given a phrase (a, b) with embeddings v a , v b , respectively, we found that the tensor composition FORMULA6 , we split the data into a development set of 18 humans and a test set of the remaining 36 humans.

We use the development set to select the optimal scalar weight for the weighted tensor composition, and using this fixed parameter, we report the results using the test set.

We repeat this three times, rotating over folds of 18 subjects, and report the average results.

DISPLAYFORM0 DISPLAYFORM1 yields worse performance than the simple additive composition v a + v b .

For this reason, we consider a weighted tensor composition vAs a baseline, we also report the average results using just the additive composition, as well as a weighted additive composition βv a + v b , where β ≥ 0.

We select β using the development set ("weighted1") and the test set ("weighted2").

We allow weighted2 to cheat in this way because it provides an upper bound on the best possible weighted additive composition.

Additionally, we compare our method to the smoothed inverse frequency ("sif") weighting method that has been demonstrated to be near state-of-the-art for sentence embedding tasks (Arora et al., 2016) .

We also test embeddings of the form p + γω a ω b T (v a , v b , ·) ("sif+tensor"), where p is the sif embedding for (a, b), ω a and ω b are the smoothed inverse frequency weights used in the sif embeddings, and γ is a positive weight selected using the development set.

The motivation for this hybrid embedding is to evaluate the extent to which the sif embedding and tensor component can independently improve performance on this task.

We perform these same experiments using two other standard sets of pre-computed word embeddings, namely GloVe 3 and carefully optimized cbow vectors 4 (Mikolov et al., 2017) .

We re-trained the composition tensor using the same corpus and technique as before, but substituting these pre-computed embeddings in place of the RAND-WALK (rw) embeddings.

However, a bit of care must be taken here, since our syntactic RAND-WALK model constrains the norm of the word embeddings to be related to the frequency of the words, whereas this is not the case with the pre-computed embeddings.

To deal with this, we rescaled the pre-computed embeddings sets to have the same norms as their counterparts in the rw embeddings, and then trained the composition tensor using these rescaled embeddings.

At test time, we use the original embeddings to compute the additive components of our compositions, but use the rescaled versions when computing the tensor components.

The results for adjective-noun phrases are given in Tables 3.

We observe that the tensor composition outperforms the additive compositions on all embedding sets apart from the Spearman correlation on the cbow vectors, where the weighted additive 2 method has a slight edge.

The sif embeddings outperform the additive and tensor methods, but combining the sif embeddings and the tensor components yields the best performance across the board, suggesting that the composition tensor captures additional information beyond the individual word embeddings that is useful for this task.

There was high consistency across the folds for the optimal weight parameter α, with α = 0.4 for the rw embeddings, α = .2, .3 for the glove embeddings, and α = .3 for the cbow embeddings.

For the sif+tensor embeddings, γ was typically in the range [.1, .2].The results for verb-object phrases are given in Table 4 .

Predicting phrase similarity appears to be harder in this case.

Notably, the sif embeddings perform worse than unweighted vector addition.

As before, we can improve the sif embeddings by adding in the tensor component.

The tensor composition method achieves the best results for the glove and cbow vectors, but weighted addition works best for the randwalk vectors.

Overall, these results demonstrate that the composition tensor can improve the quality of the phrase embeddings in many cases, and the improvements are at least somewhat orthogonal to improvements

In this section we present additional qualitiative results demonstrating the use of the composition tensor for the retrieval of words related to adjective-noun and verb-object phrases.

In TAB3 , we show results for the phrases "giving birth", "solve problem", and "changing name".

These phrases are all among the top 500 most frequent verb-object phrases appearing in the training corpus.

In these examples, the tensor-based phrase embeddings retrieve words that are generally markedly more related to the phrase at hand, and there are no strange false positives.

These examples demonstrate how a verb-object phrase can encompass an action that isn't implied simply by the object or verb alone.

The additive composition doesn't capture this action as well as the tensor composition.

Moving on to adjective-noun phrases, in TAB4 , we show results for the phrases "United States", "Soviet Union", and "European Union".

These phrases, which all occur with comparatively high frequency in the corpus, were identified as adjective-noun phrases by the tagger, but they function more as compound proper nouns.

In each case, the additive composition retrieves reasonably relevant words, while the tensor composition is more of a mixed bag.

In the case of "European Union", the tensor composition does retrieve the highly relevant words eec (European Economic Community) and eea (European Economic Area), which the additive composition misses, but the tensor composition also produces several false positives.

It seems that for these types of phrases, the additive composition is sufficient to capture the meaning.

In TAB5 , we fix the noun "taste" and vary the modifying adjective to highlight different senses of the noun.

In the case of "expensive taste", both compositions retrieve words that seem to be either related to "expensive" or "taste", but there don't seem to be words that are intrinsically related to the phrase as a whole (with the exception, perhaps, of "luxurious", which the tensor composition retrieves).

In the case of "awful taste", both compositions retrieve fairly similar words, which mostly relate to the physical sense of taste (rather than the more abstract sense of the word).

For the phrase "refined taste", the additive composition fails to capture the sense of the phrase and retrieves many words related to food taste (which are irrelevant in this context), whereas the tensor composition retrieves more relevant words.

In TAB6 , we fix the noun "friend" and vary the modifying adjective, but in all three cases, the adjective-noun phrase has basically the same meaning.

In the case of "close friend" and "dear friend", both compositions retrieve fairly relevant and similar words.

In the case of "best friend", both compositions retrieve false positives: the additive composition seems to find words related to movie awards, while the tensor composition finds unintuitive false positives.

We note that in all three phrases, the tensor composition consistently retrieves the words "confidante", "confided" or "confides", "coworker", and "protoge", all of which are fairly relevant.

We test the effect of using the composition tensor for a sentiment analysis task.

We use the movie review dataset of Pang and Lee (Pang & Lee, 2004) as well as the Large Movie Review dataset (Maas et al., 2011) , which consist of 2,000 movie reviews and 50,000 movie reviews, respectively.

For a fixed review, we identify each adjective-noun pair (a, b) and compute T (v a , v b , ·).

We add these compositions together with the word embeddings for all of the words in the review, and then normalize the resulting sum.

This vector is used as the input to a regularized logistic regression classifier, which we train using scikit-learn (Pedregosa et al., 2011) with the default parameters.

We also consider a baseline method where we simply add together all of the word embeddings in the movie review, and then normalize the sum.

We evaluate the test accuracy of each method using TAB7 .

Although the tensor method seems to have a slight edge over the baseline, the differences are not significant.

In this section we will prove the main Theorem 1, which establishes the connection between the model parameters and the correlations of pairs/triples of words.

As we explained in Section 3, a crucial step is to analyze the partition function of the model and show that the partition functions are concentrated.

We will do that in Section B.1.

We then prove the main theorem in Section B.2.

More details and some technical lemmas are deferred to Section B.3

In this section we will prove concentrations of partition functions (Lemma 1).

Recall that we need the tensor to be K-bounded (where K is a constant) for this to work.

DISPLAYFORM0 Note that K here should be considered as an absolute constant (like 5, in fact in Section 5 we show K is less than 4).

We first restate Lemma 1 here: Lemma 3 (Lemma 1 restated).

For the syntactic RAND-WALK model, there exists a constant Z such that Pr DISPLAYFORM1 Furthermore, if the tensor T is (K, )-bounded, then for any fixed word a ∈ V , there exists a constant Z a such that Pr DISPLAYFORM2 In fact, the first part of this Lemma is exactly Lemma 2.1 in Arora et al. (2015) .

Therefore we will focus on the proof of the second part.

For the second part, we know the probability of choosing a word b is proportional to DISPLAYFORM3 If the probability of choosing word w is proportional to exp( r, v w ) for some vector r (think of r = T (v a , ·, c)+c), then in expectation the partition function should be equal to nE v∼D V [exp( r, v )] (here D V is the distribution of word embedding).

When the number of words is large enough, we hope that with high probability the partition function is close to its expectation.

Since the Gaussian distribution is spherical, we also know that the expected partition function nE v∼D V [exp( r, v )] should only depend on the norm of r. Therefore as long as we can prove the norm of r = T (v a , ·, c)+c remain similar for most c, we will be able to prove the desired result in the lemma.

We will first show the norm of r = T (v a , ·, c) + c is concentrated if the tensor T is (K, )-bounded.

Throughout all subsequent proofs, we assume that < 1 and d ≥ log 2 n/ 2 .Lemma 4.

Let v a be a fixed word vector, and let c be a random discourse vector.

If T is (K, )-bounded with d ≥ log 2 n/ 2 , we have DISPLAYFORM4 where 0 ≤ L ≤ K is a constant that depends on v a , and δ = exp(−Ω(log 2 n)).Proof.

Since c is a uniform random vector on the unit sphere, we can represent c as c = z/ z , where z ∼ N (0, I) is a standard spherical Gaussian vector.

For ease of notation, let M = T (v a , ·, ·)+I, and write the singular value decomposition of M as M = U ΣV T .

Note that Σ = diag(λ 1 , . . .

, λ d ) and U and V are orthogonal matrices, so that in particular, the random variable y = V T z has the same distribution as z, i.e. its entries are i.i.d.

standard normal random variables.

Further, U x 2 = x 2 for any vector x, since U is orthogonal.

Hence, we have DISPLAYFORM5 Since both the numerator and denominator of this quantity are generalized χ 2 random variables, we can apply Lemma 7 to get tail bounds on both.

Observe that by assumption, we have λ 2 i ≤ Kd 2 / log 2 n for all i, and DISPLAYFORM6 We will apply Lemma 7 to prove concentration bounds for A, in this case we have DISPLAYFORM7 Under our assumptions, we know λ 2 max ≤ Kd 2 / log 2 n and DISPLAYFORM8 Similarly, we can apply Lemma 7 to B (in fact we can apply simpler concentration bounds for standard χ 2 distribution), and we get DISPLAYFORM9 If we take x = 1 16 log 2 n, we know 2 DISPLAYFORM10 When both events happen we know considered as a constant) .

This finishes the proof.

DISPLAYFORM11 Using this lemma, we will show that the expected condition number nE v∼D V [exp( r, v )] (where r = T (v a , ·, c) + c) is concentrated Lemma 5.

Let v a be a fixed word vector, and let c be a random discourse vector.

If T is (K, )-bounded, there exists Z a such that we have DISPLAYFORM12 where Z a = Θ(n) depends on v a , and δ = exp(−Ω(log 2 n)).Proof.

We know v = s ·v wherev ∼ N (0, I) and s is a (random) scaling.

Let r = T (v a , ·, c) + c. Conditioned on s we know r, v is equivalent to a Gaussian random variable with standard deviation σ = r s. For this random variable we know DISPLAYFORM13 .

In particular, this implies g(x + γ) ≤ exp(κ 2 γ/2)g(x) (for small γ).By Lemma 4, we know with probability at least 1 − Ω(log 2 n), r 2 ∈ L ± O( ).

Therefore, when this holds, we have DISPLAYFORM14 The multiplicative factor on the RHS is bounded by 1 + O( ) when is small enough (and κ is a constant).

This finishes the proof.

Now we know the expected partition function is concentrated (for almost all discourse vectors c), it remains to show when we have finitely many words the partition function is concentrated around its expectation.

This was already proved in Arora et al. FORMULA6 , we use their lemma below: Lemma 6.

For any fixed vector r (whose norm is bounded by a constant), with probability at least 1 − exp(−Ω(log 2 n)) over the choices of the words, we have DISPLAYFORM15 DISPLAYFORM16 This is essentially Lemma 2.1 in Arora et al. (2015) (see Equation A.32).

The version we stated is a bit different because we allow r to have an arbitrary constant norm (while in their proof vector r is the discourse vector c and has norm 1).

This is a trivial corollary as we can move the norm of r into the distribution of the scaling factor s for the word embedding.

Finally we are ready to prove Lemma 1.Proof of Lemma 1.

The first part is exactly Lemma 2.1 in Arora et al. (2015) .For the second part, note that the partition function DISPLAYFORM17 We will use E[Z c,a ] to denote its expectation over the randomness of the word embedding {v i }.

By Lemma 5, we know for at least 1 − exp(−Ω(log 2 n)) fraction of discourse vectors c, the expected partition function is concentrated (E[Z c,a ] ∈ (1 ± O( ))Z a ).

Let S denote the set of c such that Lemma 5 holds.

Now by Lemma 6 we know for any x ∈ S, with probability at least 1 DISPLAYFORM18 Therefore we know if we consider both c and the embedding as random variables, Pr[Z c,a ∈ (1 ± O( + z ))Z a ] ≥ 1 − δ where δ = exp(−Ω(log 2 n)).

Let S be the set of word embedding such that there is at least DISPLAYFORM19 That is, with probability at least 1 − √ δ (over the word embeddings), there is at least 1 − √ δ fraction of c such that Z c,a ∈ (1 ± O( + z ))Z a .

In this section we prove Theorem 1 and Corollary 1.

The proof is very similar to the proof of Theorem 2.2 in Arora et al. (2015) .

We use several lemmas in that proof, and these lemmas are deferred to Section B.3.Proof of Theorem 1.

Throughout this proof we consider two adjacent discourse vectors c, c , where c generated a single word w and c generated a syntactic pair (a, b).The first two results in Theorem 1 are exactly the same as Theorem 2.2 in Arora et al. (2015) .

Therefore we only need to prove the result for p ([a, b] ) and p(w, [a, b] ).

Here the last step used Lemma 10.

Since both Z and Z a can be bounded by O(n), and DISPLAYFORM0 is bounded by (4κ + √ 2K) 2 , we know the first term is of order Ω(1/n 2 ), and the second term is negligible.

We end with the proof of Lemma 2.Proof of Lemma 2.

Just for this proof, we use the following notation.

Let I d×d be the d-dimensional identity matrix, and let x 1 , x 2 , . . . , x n be i.i.d.

draws from N (0, I d×d ).

Let y i = x i 2 , and note that y , v i , c) ).

We first cover the unit sphere by a finite number of metric balls of small radius.

Then we show that with high probability, the partition function at the center of these balls is indeed bounded below by a constant.

Finally, we show that the partition function evaluated at an arbitrary point on the unit sphere can't be too far from the partition function at one of the ball centers provided the norms of the v i are not too large.

We finish by appropriately controlling the norms of the v i .

≥ .

@highlight

We present a generative model for compositional word embeddings that captures syntactic relations, and provide empirical verification and evaluation.