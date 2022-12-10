We propose a unified framework for building unsupervised representations of individual objects or entities (and their compositions), by associating with each object both a distributional as well as a point estimate (vector embedding).

This is made possible by the use of optimal transport, which allows us to build these associated estimates while harnessing the underlying geometry of the ground space.

Our method gives a novel perspective for building rich and powerful feature representations that simultaneously capture uncertainty (via a distributional estimate) and interpretability (with the optimal transport map).

As a guiding example, we formulate unsupervised representations for text, in particular for sentence representation and entailment detection.

Empirical results show strong advantages gained through the proposed framework.

This approach can be used for any unsupervised or supervised problem (on text or other modalities) with a co-occurrence structure, such as any sequence data.

The key tools underlying the framework are Wasserstein distances and Wasserstein barycenters (and, hence the title!).

One of the main driving factors behind the recent surge of interest and successes in natural language processing and machine learning has been the development of better representation methods for data modalities.

Examples include continuous vector representations for language (Mikolov et al., 2013; Pennington et al., 2014) , convolutional neural network (CNN) based text representations (Kim, 2014; Kalchbrenner et al., 2014; Severyn and Moschitti, 2015; BID4 , or via other neural architectures such as RNNs, LSTMs BID14 Collobert and Weston, 1 And, hence the title! 2008), all sharing one core idea -to map input entities to dense vector embeddings lying in a lowdimensional latent space where the semantics of the inputs are preserved.

While existing methods represent each entity of interest (e.g., a word) as a single point in space (e.g., its embedding vector), we here propose a fundamentally different approach.

We represent each entity based on the histogram of contexts (cooccurring with it), with the contexts themselves being points in a suitable metric space.

This allows us to cast the distance between histograms associated with the entities as an instance of the optimal transport problem (Monge, 1781; Kantorovich, 1942; Villani, 2008) .

For example, in the case of words as entities, the resulting framework then intuitively seeks to minimize the cost of moving the set of contexts of a given word to the contexts of another.

Note that the contexts here can be words, phrases, sentences, or general entities cooccurring with our objects to be represented, and these objects further could be any type of events extracted from sequence data, including e.g., products such as movies or web-advertisements BID8 , nodes in a graph BID9 , or other entities (Wu et al., 2017) .

Any co-occurrence structure will allow the construction of the histogram information, which is the crucial building block for our approach.

A strong motivation for our proposed approach here comes from the domain of natural language, where the entities (words, phrases or sentences) generally have multiple semantics under which they are present.

Hence, it is important that we consider representations that are able to effectively capture such inherent uncertainty and polysemy, and we will argue that histograms (or probability distributions) over embeddings allows to capture more of this information compared to point-wise embeddings alone.

We will call the histogram as the distributional estimate of our object of interest, while we refer to the individual embeddings of single contexts as point estimates.

Next, for the sake of clarity, we discuss the framework in the concrete use-case of text representations, when the contexts are just words, by employing the well-known Positive Pointwise Mutual Information (PPMI) matrix to compute the histogram information for each word.

With the power of optimal transport, we show how this framework can be of significant use for a wide variety of important tasks in NLP, including word and sentence representations as well as hypernymy (entailment) detection, and can be readily employed on top of existing pre-trained embeddings for the contexts.

The connection to optimal transport at the level of words and contexts paves the way to make better use of its vast toolkit (like Wasserstein distances, barycenters, etc.) for applications in NLP, which in the past has primarily been restricted to document distances (Kusner et al., 2015; BID16 .We demonstrate that building the required histograms comes at almost no additional cost, as the co-occurrence counts are obtained in a single pass over the corpus.

Thanks to the entropic regularization introduced by Cuturi (2013), Optimal Transport distances can be computed efficiently in a parallel and batched manner on GPUs.

Lastly, the obtained transport map FIG0 ) also provides for interpretability of the suggested framework.

Most of the previous work in building representations for natural language has been focused towards vector space models, in particular, popularized through the groundbreaking work in Word2vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) .

The key idea in these models has been to map words which are similar in meaning to nearby points in a latent space.

Based on which, many works (Levy and Goldberg, 2014a; Melamud et al., 2015; Bojanowski et al., 2016) have suggested specializing the embeddings to capture some particular information required for the task at hand.

One of the problems that still persists is the inability to capture, within just a point embedding, the various semantics and uncertainties associated with the occurrence of a particular word BID15 .A recent line of work has proposed the view to represent words with Gaussian distributions or mixtures of Gaussian distributions (Vilnis and McCallum, 2014b; Athiwaratkun and Wilson, 2017) , or hyperbolic cones BID5 for this purpose.

Also, a concurrent work from Muzellec and Cuturi (2018) has suggested using elliptical distributions endowed with a Wasserstein metric.

While these already provide richer information than typical vector embeddings, their form restricts what could be gained by allowing for arbitrary distributions.

In addition, hyperbolic embeddings (Nickel and Kiela, 2017; BID5 are so far restricted to supervised tasks (and even elliptical embeddings (Muzellec and Cuturi, 2018) to a most extent), not allowing unsupervised representation learning as in the focus of the paper here.

To this end, we propose to associate with each word a distributional and a point estimate.

These two estimates together play an important role and enable us to make use of optimal transport.

Amongst the few explorations of optimal transport in NLP, i.e., document distances (Kusner et al., 2015; BID16 , document clustering (Ye et al., 2017) , bilingual lexicon induction (Zhang et al., 2017) , or learning an orthogonal Procrustes mapping in Wasserstein distance BID7 , the focus has been on transporting words directly.

For example, the Word Mover's Distance (Kusner et al., 2015) casts finding the distance between documents as an optimal transport problem between their bag of words representation.

Our approach is different as we consider the transport over contexts instead, and use it to propose a representation for words.

This enables us to establish any kind of distance (even asymmetric) between words by defining a suitable underlying cost on the movement of contexts, as we show for the case of entailment.

Another benefit of defining this transport over contexts is the added flexibility to extend the representation for sentences (or arbitrary length text) by utilizing the idea of Wasserstein barycenters, which to the best of our knowledge has never been considered in the past.

Lastly, the proposed framework is not specific to words or sentences but holds for building unsupervised representations for any entity and composition of entities, where a co-occurrence structure can be devised between entites and their contexts.

Optimal Transport (OT) provides a way to compare two probability distributions defined over a space G, given an underlying distance on this space (or more generally a cost of moving one point to another).

In other terms, it lifts distance between points to distance between distributions.

Below, we give a short yet formal background description on optimal transport for the discrete case.

Let's consider an empirical probability measure of the form µ = n i=1 a i δ(x i ) where X = (x 1 , . . . , x n ) ∈ G n , δ(x) denotes the Dirac (unit mass) distribution at point x ∈ G, and (a 1 , . . . , a n ) lives in the probability simplex DISPLAYFORM0 If the ground cost of moving from point x i to y j is denoted by M ij , then the Optimal Transport distance between µ and ν is the solution to the following linear program.

OT(µ, ν; M ) := min DISPLAYFORM1 Here, the optimal T ∈ R n×m is referred to as the transportation matrix: T ij denotes the optimal amount of mass to move from point x i to point y i .

Intuitively, OT is concerned with the problem of moving goods from factories to shops in such a way that all the demands are satisfied and the overall transportation cost is minimal.

When G = R d and the cost is defined with respect to a metric D G over G (i.e., M ij = D G (x i , y j ) p for any i, j), OT defines a distance between empirical probability distributions.

This is the p-Wasserstein distance, defined as DISPLAYFORM2 In most cases, we are only concerned with the case where p = 1 or 2.The cost of exactly solving OT problem scales at least in O(n 3 log(n)) (n being the cardinality of the support of the empirical measure) when using network simplex or interior point methods.

Following Cuturi (2013) we consider the entropy regularized Wasserstein distance, W λ p (µ, ν).

The above problem can then be solved efficiently using Sinkhorn iterations, albeit at the cost of some approximation error.

The regularization strength λ ≥ 0 controls the accuracy of approximation and recovers the true OT for λ = 0.

The cost of the Sinkhorn algorithm is only quadratic in n at each iteration.

Further on in our discussion, we will make use of the notion of averaging in the Wasserstein space.

More precisely the Wasserstein barycenter, introduced by Agueh and Carlier (2011) , is a probability measure that minimizes the sum of (p-th power) Wasserstein distances to the given measures.

Formally, given N measures {ν 1 , . . . , ν N } with corresponding weights η = {η 1 , . . .

, η N } ∈ Σ N , the Wasserstein barycenter can be written as follows: DISPLAYFORM3 (2) We similarly consider the regularized barycenter B λ p , using entropy regularized Wasserstein distances W λ p in the above minimization problem, following Cuturi and Doucet (2014) .

Employing the method of iterative Bregman projections (Benamou et al., 2015) , we obtain an approximation of the solution at a reasonable computational cost.

In this section, we elaborate on both the distributional and the point estimate that we attach to each word, as mentioned in the introduction.

A common method in NLP to empirically estimate the probability p(w|c) of occurrence of a word w in some context c, is to compute the number of times the word w co-occurs with context c relative to the total number of times context c appears in the corpus.

The context c could be a particular word, phrase, sentence or other definitions of co-occurrence of interest.

Distributional Estimate.

For a word w, its distributional estimate is built from a histogram over the set of contexts C, and an embedding of these contexts into a space G.A natural way to build this histogram is to maintain a co-occurrence matrix between words in our vocabulary and all possible contexts, such that its each entry indicates how often a word and context occur in an interval (or window) of a fixed size L. Then, the bin values ((H w ) c ) c∈C of the histogram (H w ) for a word w, can be viewed as the row corresponding to w in this co-occurrence matrix.

In Section 5, we discuss how to reduce the number of bins in the histogram, and possible modifications of the co-occurrence matrix to improve associations.

The simplest embedding of contexts is into the space of one-hot vectors of all the possible contexts.

However, this induces a lot of redundancy in the representation and the distance between contexts does not reflect their semantics.

A classical solution would be to instead find a dense lowdimensional embedding of contexts that captures the semantics, possibly using techniques such as SVD or deep neural networks.

We denote by V = (v c ) c∈C an embedding of the contexts into this low-dimensional space G ⊂ R d , which we refer to as the ground space. (We will consider prototypical cases of how this metric can be obtained in Sections 6 and 7.)Combining the histogram H w and the embedding V , we represent the word w by the following empirical distribution: DISPLAYFORM0 Recall that δ(v c ) denotes the Dirac measure at the position v c of the context c. We refer to this representation (Eq. (3)) as the distributional estimate of the word.

Together with its distributional estimate, the word w also has an associated point estimate v w when it occurs in the sense of a context, in the form of its position (or embedding) in the ground space.

This is what we mean by attaching the distributional and point estimate to each word.

Distance.

If we equip the ground space G with a meaningful metric D G , then we can subsequently define a distance between the representations of two words w i and w j , as the solution to the following optimal transport problem: DISPLAYFORM1 Intuitively, two words are similar in meaning if the contexts of one word can be easily or cheaply transported to the contexts of the other word, with this ease of transportation being measured by D G .

This idea still remains in line with the distributional hypothesis BID11 Rubenstein and Goodenough, 1965 ) that words in similar contexts have similar meanings, but provides a unique way to quantify it.

Interpretation.

In fact, both of these estimates are closely tied together and required to serve as an effective representation.

For instance, if we only have the distributional estimates, then we may have Here, we pick four contexts at random from a list of top 20 contexts (in terms of PPMI) for the two histograms.

Then using the regularized Wasserstein distance (as in Eq. (4) ), we plot the obtained transportation matrix (or commonly called transport map) T as above.

Note how 'ivory' adjusts its movement towards 'skin' (as in skin color) to allow 'poaching' to be easily moved to 'endangered' as going to other contexts of 'mammal' is costly for 'poaching', thus capturing a global perspective.two words such as 'tennis' and 'football' which occur in the contexts of {court, penalty, judge} and {stadium, foul, referee} respectively.

While these contexts are mutually disjoint, they are quite close in meaning.

Now there could be a third word such as 'law' which occurs in the exact same contexts as tennis.

So considering the distributional estimate alone, without making use of the point estimates of context, would lead us to have a smaller distance between tennis and law as compared to tennis and football.

Whereas, if we only considered the point estimates, then we would lose much of the uncertainty associated about the contexts in which they occur, except for maybe the restricted information of neighboring points in the ground space.

This is made clear in a related illustration shown in Figure 2.The family of problems where such a representation can be used is not restricted to entities pertaining to NLP: the framework can be similarly used in any domain where a co-occurrence structure exists between entities and their contexts.

For instance, in the case of movie recommendation where users correspond to the entities and movies to the contexts.

Lastly, this connection with optimal transport allows us to utilize its rich theoretical and algorithmic toolkit towards important problems in NLP.

In the next section, we discuss a concrete framework of how this can be applied and in Section 6 and 7, we detail how the tasks of sentence representation and hypernymy detection can be effectively carried out with this framework.

For the sake of brevity, we present the framework for the case where contexts consist of single words.

Making associations better.

Let's say that a word is considered to be a context word if it appears in a symmetric window of size L around the target word (the word whose distributional estimate we seek).

Now, the co-occurrence matrix is between the words of our vocabulary, with rows and columns indicating target words and context words respectively.

While each entry of this matrix reflects the co-occurrence count, it may not suggest a strong association between the target and its context.

For instance, in the sentence "She prefers her coffee to be brewed fast than being perfect", there is a stronger association between 'coffee' and 'brewed' rather than between 'coffee' and 'her', although the co-occurrence counts alone might imply the opposite.

Hence, to handle this we consider the well-known Positive Pointwise Mutual Information (PPMI) matrix (Church and Hanks, 1990; Levy et al., 2015) , whose entries are as follows: DISPLAYFORM0 The PPMI entries are non-zero when the joint probability of the target and context words cooccurring together is higher than the probability when they are independent.

Typically, these probabilities are estimated from the co-occurrence counts #(w, c) in the corpus and lead to DISPLAYFORM1 where, #(w) = c #(w, c), #(c) = w #(w, c) and |Z| = w c #(w, c).

Also, it is known that PPMI is biased towards infrequent words and assigns them a higher value.

A common solution is to smoothen 2 the context probabilities by raising them to an exponent of α lying between 0 and 1.

Levy and Goldberg (2014b) have also suggested the use of the shifted PPMI (SPPMI) matrix where the shift 3 by log(s) acts like a prior on the probability of co-occurrence of target and context pairs.

These variants of PPMI enable us to extract better semantic associations from the cooccurrence matrix.

Finally, we define DISPLAYFORM2 Hence, the bin values for our histogram in Eq. (3) are formed as: DISPLAYFORM3 Computational considerations.

The view of optimal transport between histograms of contexts introduced in Eq. (4) offers a pleasing interpretation (see FIG0 .

However, it might still be a computationally intractable in its current formulation.

Indeed the number of possible contexts can be as large as the size of vocabulary (if the contexts are just single words) or even exponential (if contexts are considered to be phrases, sentences and otherwise).

For instance, even with the use of SPPMI matrix, which also helps to sparsify the co-occurrences, the cardinality of the support the word histograms still varies from 10 3 to 5 × 10 4 context words, when considering a vocabulary of size around 2 × 10 5 .

This is a problem because the Sinkhorn algorithm for regularized optimal transport (Cuturi, 2013) (Section 3), scales roughly quadratically in the histogram size and the ground cost matrix can also become prohibitive to store in memory, for the range of histogram sizes mentioned.

One possible fix is to instead consider a few representative contexts in this ground space.

The hope is that with the dense low-dimensional embeddings and a meaningful metric between them, we may not require as many contexts as needed before.

For instance, this can be achieved by clustering the contexts with respect to metric D G .

Besides the computational gain, the clustering will lead us to consider this transport between more abstract contexts.

This will although come at the loss of some interpretability.

Another alternative for dealing with this computational issue could be to consider stochastic optimal transport techniques BID6 , where the intuition would be to randomly sample a subset of contexts while considering this transport.

But we leave that direction for a future work.

Now, consider that we have obtained K contexts, each representing some part C k of the set of contexts C. The histogram for word w with respect to these contexts can then be written as DISPLAYFORM4 Hereṽ k ∈Ṽ denotes the point estimate of the k th representative context, and (H w ) k are the new bin values for the histogram similar to that in Eq. (5), but with respect to these parts, DISPLAYFORM5 Furthermore, in certain cases 4 , it can be important to measure the relative portion of C k 's SPPMI (Eq. FORMULA12 ) that has been used towards a word w. Otherwise the process of making the histogram unit sum in Eq. (6) will misrepresent the actual underlying contribution (check Eq. (10) in Appendix A 4 when the SPPMI contributions towards the partitions (or clusters) have a large variance.

for more details): DISPLAYFORM6 Summary.

While we detailed the case of context as single words, this framework can be extended in a similar manner to take into account other contexts such as bi-grams, tri-grams, n-grams or other abstract semantic concepts.

Building this suggested representation comes at almost free cost during the typical learning of point-estimates for an NLP task, as the co-occurrence counts can simply be maintained while going through the corpus.

GloVe (Pennington et al., 2014) even constructs the co-occurrence matrix explicitly as a precursor to learning the point-estimates.

Traditionally, the goal of this task is to develop a representation for sentences, that captures the semantics conveyed by it.

Most unsupervised representations proposed in the past rely on the composition of vector embeddings for the words, through either additive, multiplicative, or other ways (Mitchell and Lapata, 2008; Arora et al., 2017; Pagliardini et al., 2017) .

We propose to represent sentences as probability distributions to better capture the inherent uncertainty and polysemy.

Our belief is that the meaning of a sentence can be understood as a concept that best explains the simultaneous occurrence of the words in it.

We hypothesize that a sentence, S = (w 1 , w 2 , . . .

, w N ), can be efficiently represented via the Wasserstein barycenter (see Eq. (2)) of distributional estimates of the words in the sentence, i.e., DISPLAYFORM0 which is itself again a distribution over G. Yet another interesting property is the nonassociativity 5 of the barycenter operation.

This can be utilized to take into account the order of the words in a sentence.

For now, we restrict our focus on exploring how well barycenters of words taken all at once can represent sentences and this direction is left for future work.

Interestingly, the classical weighted averaging of point-estimates (Arora et al., 2017) (Agirre et al., 2012 BID3 BID1 BID0 BID2 .

The objective here is to predict how similar or dissimilar are two sentences in their meanings.

Since with barycenter representation as in Eq. FORMULA13 , each sentence is also a histogram of contexts, we can again make use of optimal transport to define the distance between two sentences S 1 and S 2 , DISPLAYFORM1 As a ground metric, we consider the Euclidean distance between the point estimates of words.

This point estimate for a word is its embedding in the context space and can be obtained with the help of Word2vec (Mikolov et al., 2013) or GloVe (Pennington et al., 2014) .

For, this task we train the word embeddings on the Toronto Book Corpus (Kiros et al., 2015) via GloVe and in the process also gain the distributional estimates of words for free.

Since the word embeddings in these methods are constructed so that similar meaning words are close in cosine similarity, we find the representative points by performing K-means clustering with respect to this similarity.

We benchmark our performance against SIF (Smooth Inverse Frequency) method from Arora et al. FORMULA12 who regard it as a "simple but toughto-beat baseline", as well as against the common Bag of Words (BoW) averaging.

For this experiment, we use SIF's publicly available implementation 6 and perform the evaluation using SentEval 6 https://github.com/PrincetonML/SIF (Conneau and Kiela, 2018) .

Table 1 shows that we always beat BoW and SIF with weighted averaging on all tasks.

Further, we perform better than the best variant of SIF (which in addition removes the 1 st principal component) on 3 out of 5 tasks.

Also, on the other two tasks we still perform competitively and achieve an overall gain over their best variant with K = 300 clusters (refer to TAB5 in Appendix A for detailed results).

Note that, the hyperparameters for SIF are taken to be the best ones separately for each task.

Whereas we used the same set of hyperparameters for all the above tasks and the PPMI specific hyperparameters haven't been tuned much, but this should not give us an edge over them.

In our comparison, we do not include methods such as Sent2vec (Pagliardini et al., 2017) , as they are specifically trained to work well on the given task of sentence representation, and such an approach for training remains outside the scope of current work.

Our approach for representing barycenters does not require any additional training and is still able to match and outperform strong baselines for the task of semantic similarity.

This highlights the efficacy of proposed representation.

In linguistics, hypernymy is a relation between words (or sentences) where the semantics of one word (the hyponym) are contained within that of another word (the hypernym).

A simple form of this relation is the is-a relation, e.g., cat is an animal.

Hypernymy is a special case of the more general concept of lexical entailment which may be broadly defined as any semantic relations between two lexical items where the meaning of one is implied by the meaning of the other.

Detecting lexical entailment relations is relevant for numerous tasks in NLP.

Given a database of lexical entailment relations, e.g., containing Roger Federer is a tennis player might help a question answering system an- Table 2 : Comparison between entailment vectors and optimal transport / Wasserstein based entailment measure (WE).

The scores are AP@all (%).

The hyperparameter α refers to the smoothing exponent and s to the shift in the PPMI computation.

More datasets are presented in TAB4 Table 3 : Comparison between entailment vectors, optimal transport / Wasserstein based entailment measure (WE) and other state-of-the-art methods.

GE+C and GE+KL are Gaussian embeddings with cosine similarity and negative KL-divergence.

The scores for GE+C, GE+KL, and DIVE + C·∆S are taken from (Chang et al., 2017) as we use the same evaluation setup.

The scores are again AP@all (%).swering the question "

Who is Switzerland's most successful tennis player?".

First distributional approaches to detect hyponymy were unsupervised and exploited different linguistic properties of hypernymy (Weeds and Weir, 2003; Kotlerman et al., 2010; Santus et al., 2014; Rimell, 2014) .

While most of these methods are count-based, word embedding based methods (Chang et al., 2017; Nickel and Kiela, 2017; BID13 have become more popular in recent years.

Other approaches represent words by Gaussian distributions and use KL-divergence as a measure of entailment (Vilnis and McCallum, 2014a; Athiwaratkun and Wilson, 2017) .

Especially for tasks like hypernymy detection, these methods have proven to be powerful as they not only capture the semantics but also the uncertainty about various concepts in which the word appears.

Using the framework presented in Section 4, we define a measure of entailment as the optimal transport cost (see Eq. (4) ) between associated distributions under a suitable ground cost.

For this purpose, we rely on a model that was recently proposed by BID13 BID12 which explicitly models what information is known about a word by interpreting each entry of the embedding as the degree to which a certain feature is present.

Based on the logical definition of entailment they derive an approximate inference procedure and an operator measuring the degree of entailment between two so-called entailment vectors defined as follows: DISPLAYFORM0 , where the sigmoid function σ and log are applied component-wise on the embeddings v y , v x .

Thus, our choice for the ground cost D on the basis of this entailment operator is D DISPLAYFORM1 This asymmetric and not necessarily positive ground cost illustrates that our framework can be flexibly used with an arbitrary cost function defined on the ground space.

Evaluation.

In total, we evaluated our method on 9 standard datasets: BLESS (Baroni and Lenci, 2011), EVALution (Santus et al., 2015) , Lenci/Benotto (Benotto, 2015) , Weeds (Weeds et al., 2014 ), Henderson 7 (Henderson, 2017 , Baroni (Baroni et al., 2012) , Kotlerman (Kotlerman et al., 2010) , Levy (Levy et al., 2014) and Turney (Turney and Mohammad, 2015) .

As an evaluation metric, we use average precision AP@all (Zhu, 2004) .

For comparison we also report the performance of the entailment embeddings that were trained as described in BID12 8 .

Following (Chang et al., 2017) we pushed any OOV (out-ofvocabulary) words in the test data to the bottom of the list, effectively assuming that the word pairs do not have a hypernym relation.

Table 2 compares the performance 9 of entailment embeddings and the optimal transport measure based on the ground cost defined in Eq. (9).

Our method yields significant improvements over the entailment embeddings by BID12 on almost all of the datasets.

Only on the Baroni dataset, our method performs worse but nevertheless still achieves similar performance as other state-of-the-art methods.

It confirms the findings of (Shwartz et al., 2016) and (Chang et al., 2017) : there is no single hypernymy scoring function that performs best on all datasets.

Furthermore, on some datasets (EVALution, LenciBenotto, Weeds, Turney) we even outperform or match state-of-theart performance (cf .

Table 3) , by simply using our framework together with ground cost as defined in Eq. (9).Notably, our method is not specific to the entailment vectors by BID12 .

It can be used with any embedding vectors and ground cost measuring the degree of entailment, without requiring any additional training.

A more accurate ground cost or embedding vectors might even further improve the performance.

Furthermore, our training dataset (Wikipedia with 1.7B tokens) and our vocabulary with only 80'000 words are rather small compared to the datasets used, e.g., by (Vilnis and McCallum, 2014a) .

We expect to get even better results by using a larger vocabulary on a larger corpus.

To sum up, we advocate for associating both a distributional and point estimate as a representation for each entity.

We show how this allows us to use optimal transport over the set of contexts associated with these entities, in problems with a co-occurrence structure.

Further, the framework Aitor Gonzalez-Agirre.

2012.

Semeval-2012 In particular, when β = 1, we recover the equation for histograms as in Section 5, and β = 0 would imply normalization with respect to cluster sizes.

We make use of the Python Optimal Transport (POT) 12 for performing the computation of Wasserstein distances and barycenters on CPU.

For more efficient GPU implementation, we built custom implementation using PyTorch.

We also implement a batched version for barycenter computation, which to the best of our knowledge has not been done in the past.

The batched barycenter computation relies on a viewing computations in the form of blockdiagonal matrices.

As an example, this batched mode can compute around 200 barycenters in 0.09 seconds, where each barycenter is of 50 histograms (of size 100) and usually gives a speedup of about 10x.

For all our computations involving optimal transport, we typically use λ around 0.1 and make use of log or median normalization as common in POT to stabilize the Sinkhorn iterations.

Clustering: For clustering, we make use of kmcuda's 13 efficient implementation of K-Means algorithm on GPUs.

We plan to make all our code (for all these parts) and our pre-computed histograms (for the mentioned datasets) publicly available on GitHub soon.

Detailed results of the sentence representation and hypernymy detection experiments are listed on the following pages.

@highlight

Represent each entity based on its histogram of contexts and then Wasserstein is all you need!