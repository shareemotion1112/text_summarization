We present a framework for building unsupervised representations of entities and their compositions, where each entity is viewed as a probability distribution rather than a fixed length vector.

In particular, this distribution is supported over the contexts which co-occur with the entity and are embedded in a suitable low-dimensional space.

This enables us to consider the problem of representation learning with a perspective from Optimal Transport and take advantage of its numerous tools such as Wasserstein distance and Wasserstein barycenters.

We elaborate how the method can be applied for obtaining unsupervised representations of text and illustrate the performance quantitatively as well as qualitatively on tasks such as measuring sentence similarity and word entailment, where we empirically observe significant gains (e.g., 4.1% relative improvement over Sent2vec and GenSen).



The key benefits of the proposed approach include: (a) capturing uncertainty and polysemy via modeling the entities as distributions, (b) utilizing the underlying geometry of the particular task (with the ground cost), (c) simultaneously providing interpretability with the notion of optimal transport between contexts and (d) easy applicability on top of existing point embedding methods.

In essence, the framework can be useful for any unsupervised or supervised problem (on text or other modalities); and only requires a co-occurrence structure inherent to many problems.

The code, as well as pre-built histograms, are available under https://github.com/context-mover.

One of the driving factors behind recent successes in machine learning has been the development of better methods for data representation, thus forming the foundation around which rest of the model architecture gets built.

Examples include continuous vector representations for language (Mikolov et al., 2013; Pennington et al., 2014) , convolutional neural network based feature representations for images and text (LeCun et al., 1998; Collobert & Weston, 2008; Kalchbrenner et al., 2014) , or via the hidden state representations of LSTMs (Hochreiter & Schmidhuber, 1997; Sutskever et al., 2014) .

Pre-trained unsupervised representations in particular have been immensely useful as general purpose features for model initialization (Kim, 2014) , downstream tasks, (Severyn & Moschitti, 2015; Deriu et al., 2017) and in domains with limited supervised information (Qi et al., 2018) .The shared idea across these methods is to map input entities to dense vector embeddings lying in a low-dimensional latent space where the semantics of inputs are preserved.

Thus, each entity of interest (e.g., a word) is represented directly as a single point (i.e., its embedding vector) in space, which is typically Euclidean.

In contrast, we approach the problem of building unsupervised representations in a fundamentally different manner.

We focus on the co-occurrence information between the entities and their contexts, and represent each entity as a probability distribution (histogram) over its contexts.

Here the contexts themselves are embedded as points in a suitable low-dimensional space.

This allows us to cast finding distance between entities as an instance of the Optimal Transport problem (Monge, 1781; Kantorovich, 1942; Villani, 2008) .

So, our resulting framework intuitively compares the cost of moving the contexts of a given entity to the contexts of another, which motivates the naming Context Mover's Distance (CMD).

We will call this distribution over contexts embeddings the distributional estimate of our entity of interest (see FIG0 ), while we refer to the individual embeddings of contexts as point estimates.

More precisely, the contexts refer to any generic entities or objects (such as words, phrases, sentences, images, etc.) co-occurring with the entities to be represented.

The main motivation for our proposed approach originates from the domain of natural language, where the entities (words, phrases, or sentences) generally have different semantics depending on the context under which they are present.

Hence, it is important to consider representations that are able to effectively capture such inherent uncertainty and polysemy, and we will argue that distributional estimates capture more of this information compared to point-wise embedding vectors alone.

In particular, we will see that the co-occurrence information required to build the distributions is already obtained as the first step of point-wise embedding methods, like in GloVe (Pennington et al., 2014) , but has largely been ignored in the past.

Further, this co-occurrence information that is the crucial building block of our approach is inherent to a wide variety of problems, for instance, recommending products such as movies or web-advertisements (Grbovic et al., 2015) , nodes in a graph (Grover & Leskovec, 2016) , sequence data, or other entities (Wu et al., 2017) .

This means that, in principle, our framework can be employed to obtain a representation of various entities present across these problems.

Overall, we strongly advocate for representing entities with distributional estimates due to the above stated reasons.

But at the same time, our message isn't that point-wise embedding methods should cease to exist, rather that both kinds of methods should go hand in hand.

This will be reflected through building distributional estimates on the top of existing point embedding methods, as well as how we can combine them (cf.

Section 4) to get the best of these intrinsically different ideas.

Lastly, the connection to optimal transport at the level of entities and contexts paves the way to make better use of its vast toolkit (like Wasserstein distances, barycenters, barycentric coordinates, etc.) for applications in NLP, which in the past has primarily been restricted to document distances of original words (Kusner et al., 2015; Huang et al., 2016) , as opposed to contexts.

Thanks to the entropic regularization introduced by Cuturi (2013) , optimal transport computations can be carried out efficiently in a parallel and batched manner on GPUs.

Contributions: 1) Employing the notion of optimal transport of contexts as a distance measure, we illustrate how our framework can be of benefit for various important tasks, including word and sentence representations, sentence similarity, as well as hypernymy (entailment) detection.

The method is static and does not require any additional learning, and can be readily used on top of existing embedding methods.2) The resulting representations, as portrayed in FIG0 , 4, capture the various senses under which the entity occurs.

Next, the transport map obtained through CMD (see FIG1 gives a clear interpretation of the resulting distance obtained between two entities.3) Our Context Mover's Distance (CMD) can be used to measure any kind of distance (even asymmetric) between words, by defining a suitable underlying cost on the movement of contexts, which we show can lead to a state-of-the-art metric for word entailment.

4) Defining the transport over contexts has the additional benefit that the representations are compositional -they directly extend from entities to groups of entities (of any size), such as from word to sentence representations.

To this end, we utilize the notion of Wasserstein barycenters, which to the best of our knowledge has never been considered in the past.

This results in a significant performance boost on multiple datasets, and even outperforming supervised methods like InferSent (Conneau et al., 2017) and GenSen (Subramanian et al., 2018 ) by a decent margin.

Vector representations.

The idea of using vector space models for natural language dates back to Bengio et al. (2003) , but in particular has been popularized by the seminal works of Word2vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) .

Further, works such as (Levy & Goldberg, 2014a; Bojanowski et al., 2016) have suggested to enrich these embeddings to capture additional information.

One of the problems that still persists is the inability to capture, within just a point embedding, the various semantics and uncertainties associated with the occurrence of a particular word (Huang et al., 2012; Guo et al., 2014) .Representing with distributions.

This line of work is fairly recent, mainly originating from Vilnis & McCallum (2014) , who proposed to represent words with Gaussian distributions, and later extended to mixtures of Gaussians in (Athiwaratkun & Wilson, 2017) .

Concurrent to this work, Muzellec & Cuturi (2018) and Sun et al. (2018) have suggested using elliptical and Gaussian distributions endowed with a Wasserstein metric respectively.

While these methods already provide richer information than typical vector embeddings, their form restricts what could be gained by allowing for arbitrary distributions as possible here.

Our proposal of distributional estimate (i.e., distribution over context embeddings), inherently relies upon the empirically obtained co-occurrence information of a word and its contexts.

Hence, this naturally allows for the use of optimal transport (or Wasserstein metric) in the space containing the contexts, and leads to an interpretation ( FIG1 ) which is not available in the above approaches.

Another consequence is that the training procedure required in these methods is not necessary for our approach, as we can just utilize the existing point-embedding methods together with the co-occurrence information.

Apart from embedding entities in the Wasserstein space, other metric spaces like the Hyperbolic space, have recently gained attention for modelling hierarchical structures (Nickel & Kiela, 2017; Ganea et al., 2018) .

But, these are so far restricted to supervised tasks 1 , not allowing unsupervised representation learning, which is the focus here.

Optimal Transport in NLP.

The primary focus of the explorations of optimal transport in NLP has been on transporting words directly, and for downstream applications rather than representation learning in general.

These include document distances (Kusner et al., 2015; Huang et al., 2016) , topic modelling (Rolet et al., 2016; Xu et al., 2018) , document clustering (Ye et al., 2017) , and others (Zhang et al., 2017; Grave et al., 2018) .

For example, the Word Mover's Distance (WMD; Kusner et al., 2015) considers computing the distance between documents as an optimal transport between their bag-of-words, and in itself doesn't lead to a representation.

When the transport is defined at the level of words like in these approaches, it can not be used to represent words themselves.

In our approach, the transport is considered over contexts instead, which enables us to develop representations for words and also extend them to represent composition of words (i.e., sentences, documents) in a principled manner, as will be illustrated further through the examples of entailment detection and sentence representation respectively.

Optimal Transport (OT) provides a way to compare two probability distributions defined over a space G (commonly known as the ground space), given an underlying distance or more generally a cost of moving one point to another in the ground space.

In other terms, it lifts a distance between points to a distance between distributions.

Other divergences, such as Kullback-Leibler (KL), or f -divergence in general, only focus on the probability mass values, thus ignoring the geometry of the ground space: something which we utilize throughout this work via OT.

Also, KL(µ||ν) is defined only when the distribution µ is absolutely continuous with respect to ν.

Having motivated our choice, we give a short yet formal background on OT in the discrete case.

Linear Program Formulation.

Consider an empirical probability measure of the form µ = n i=1 a i δ(x (i) ) where X = (x (1) , . . .

, x (n) ) ∈ G n , δ(x) denotes the Dirac (unit mass) distribution at point x ∈ G, and (a 1 , . . .

, a n ) lives in the probability simplex DISPLAYFORM0 and (b 1 , . . .

, b m ) ∈ Σ m , and if the ground cost of moving from point x (i) to y (j) is denoted by M ij , then the Optimal Transport distance between µ and ν is the solution to the following linear program.

OT(µ, ν; M ) := min DISPLAYFORM1 Here, the optimal T ∈ R n×m + is referred to as the transportation matrix: T ij denotes the optimal amount of mass to move from point x (i) to point y (j) .

Intuitively, OT is concerned with the problem of moving a given supply of goods from certain factories to meet the demands at some shops, such that the overall transportation cost is minimal.

Distance.

When G = R d and the cost is defined with respect to a metric DISPLAYFORM2 p for any i, j), OT defines a distance between empirical probability distributions.

This is the p-Wasserstein distance, defined as DISPLAYFORM3 1/p .

In most cases, we are only concerned with the case where p = 1 or 2.Regularization and Sinkhorn iterations.

The cost of exactly solving OT scales at least in O(n 3 log(n)) (n being the cardinality of the support of the empirical measure) when using network simplex or interior point methods.

Following Cuturi (2013), we consider the entropy regularized Wasserstein distance, DISPLAYFORM4 1/p , where the search space for the optimal T is instead restricted to a smooth solution close to the extreme points of this linear program, as follows: DISPLAYFORM5 where H(T ) = − ij T ij log T ij .

The regularized problem can then be solved efficiently using Sinkhorn iterations (Sinkhorn, 1964) , albeit at the cost of some approximation error.

This can be controlled by the regularization strength λ ≥ 0, with the true OT recovered at λ = 0.

While the cost of each Sinkhorn iteration is quadratic in n, Altschuler et al. (2017) have shown that the convergence to an -accurate solution can be attained in a number of iterations that is independent of n, thus resulting in an overall complexity of O(n 2 / 3 ).Barycenters.

In Section 6, we will make use of the notion of averaging in the Wasserstein space.

More precisely, the Wasserstein barycenter, introduced by Agueh & Carlier (2011) , is a probability measure that minimizes the sum of (p-th power) Wasserstein distances to the given measures.

Formally, given N measures {ν 1 , . . . , ν N } with corresponding weights η = {η 1 , . . .

, η N } ∈ Σ N , the Wasserstein barycenter can be written as DISPLAYFORM6 For practical purposes, we consider the regularized barycenter B p,λ , using entropy regularized Wasserstein distances W p,λ in the above minimization problem, following Cuturi & Doucet (2014) .

Employing the iterative Bregman projections (Benamou et al., 2015) , we obtain an approximation of the solution at a reasonable computational cost.

In this section, we define the distributional estimate that we use to represent each entity.

Since we take the guiding example of building text representations, we consider each entity to be a word for simplicity.

Distributional Estimate (P w V ).

For a word w, its distributional estimate is built from a histogram H w over the set of contexts C, and an embedding of these contexts into a space G. The histogram essentially measures how likely it is for a word w to occur in a particular context c, i.e., probability p(w|c).

The exact formulation of this in closed form is generally intractable and hence it's common to empirically estimate this by the number of occurrences of the word w in context c, relative to the total frequency of context c in the corpus.

FORMULA8 ) between elephant & mammal (when represented with their distributional estimates and using entailment ground metric discussed in Section 7).

Here, we pick four contexts at random from their top 20 contexts in terms of PPMI.

The square cells above denote the entries of the transportation matrix (or transport map) T obtained in the process of computing CMD.

The darker a cell, the larger the amount of mass moved between the corresponding contexts.

Thus one natural way to build this histogram is to maintain a co-occurrence matrix between words in our vocabulary and all possible contexts, where each entry indicates how often a word and context occur in a window of fixed size L. Then, the bin values (H w ) c∈C of the histogram can be viewed as the row corresponding to w in this co-occurrence matrix.

Next, the simplest embedding of contexts is into the space of one-hot vectors of all the possible contexts.

However, this induces a lot of sparsity in the representation and the distance between such embeddings of contexts does not reflect their semantics.

A classical solution would be to instead find a dense low-dimensional embedding of contexts that captures the semantics, possibly using techniques such as SVD or deep neural networks.

We denote by V = (v c ) c∈C an embedding of the contexts into this low-dimensional space G ⊂ R d , which we refer to as the ground spaceCombining the histogram H w and the context embeddings V , we represent the word w by the following empirical distribution, referred to as the distributional estimate of the word: DISPLAYFORM0 Distance.

If we equip the ground space G with a meaningful metric D G and use distributional estimates (P w V ) to represent the words, then we can define a distance between two words w i and w j as the solution to the following optimal transport problem: DISPLAYFORM1 We call this distance in Eq.(5) as the Context Mover's Distance (CMD).Intuition.

Two words are similar in meaning if the contexts of one word can be easily transported to the contexts of the other, with this cost of transportation being measured by D G .

This idea still remains in line with the distributional hypothesis (Harris, 1954; Rubenstein & Goodenough, 1965) that words in similar contexts have similar meanings, but provides a precise way to quantify it.

Interpretation.

The particular definition of CMD in Eq.(5), lends a pleasing interpretation (see FIG1 in terms of the transportation map T .

This interpretation can be useful in understanding why and how are the two words being considered as similar in meaning, by looking at this movement of contexts.

Relating neural and count-based models.

While the histogram information is characteristic of count-based language models, the point estimates are reflective of embeddings in neural language models.

But in the distributional estimate which underlies CMD, both these elements are closely tied together.

Since CMD(w t , w 1...t−1 ) 2 can be interpreted as giving unnormalized negative log probabilities for a language model, we see how this combines neural and count-based models.

Mixed Distributional Estimate.

We also consider adding the information from point estimate into the distributional estimate to get best of both the worlds.

This is done by adding a point estimate (i.e., a Dirac at its location) as an additional context with a particular mixing weight, denoted as m. The other contexts in the distributional estimate are reweighted to sum to 1 − m.

Roadmap.

Next, we discuss concretely how this framework can be applied and for brevity we restrict to the particular case where contexts consist of single words.

Section 6 details how this framework can be extended to obtain a representation for the composition of entities via Wasserstein barycenter.

Lastly in section 7, we utilize the fact that the CMD in Eq.(5) is parameterized by ground cost, and show how this flexibility can be used to define an asymmetric cost measuring entailment.

Making associations better.

We consider that a word and a context word co-occur if the latter appears in a symmetric window of size L around the word whose distributional estimate we seek (i.e., the target word).

But, it is commonly understood that co-occurrence counts alone may not necessarily suggest a strong association between the two.

The well-known Positive Pointwise Mutual Information (PPMI) matrix (Church & Hanks, 1990; Levy et al., 2015) addresses this shortcoming, and is defined as follows: PPMI(w, c) := max(log( p(w,c) p(w)×p(c) ), 0).

This means that the PPMI entries are non-zero when the joint probability of target and context words co-occurring is higher than the probability when they are independent.

Typically, these probabilities are estimated from the co-occurrence counts in the corpus.

Further improvements to the PPMI matrix have been suggested, like in Levy & Goldberg (2014b) , and following them we make use of a shifted and smoothed PPMI matrix, denoted by SPPMI α,s where α and s denote the smoothing and k-shift parameters 3 .

Overall, these variants of PPMI enable us to extract better semantic associations from the co-occurrence matrix.

Hence, the bin values (at context c) for the histogram of word w in Eq. (4) can be formulated as: DISPLAYFORM0 c ∈C SPPMIα,s(w,c) .

Computational considerations.

A natural question could arise that CMD might be computationally intractable in its current formulation, as the possible number of contexts can be enormous.

Since the contexts are mapped to dense embeddings, it is possible to only consider K representative contexts (centroids of the clusters of contexts), each covering some part C k of the set of contexts C. The histogram for word w with respect to these contexts can then be written asP DISPLAYFORM1 whereṽ k ∈Ṽ is the point estimate of the k th representative context, and (H w ) k denotes the new histogram bin values (formed by combining the SPPMI contributions).

Precise definitions and a detailed discussion on the effect of the number of clusters are given in the supplementary Section S2.Overall efficiency.

With the above aspects in account and using batched implementations on (Nvidia TitanX) GPUs, it is possible to compute around 13,700 Wasserstein-distances/second (for histogram of size 100).

Same also holds for barycenters, where we can compute 4,600 barycenters/second for sentences of length 25 and histogram size of 100.

Building the histogram information comes almost for free during the typical learning of embeddings, as in GloVe (Pennington et al., 2014) .

The goal of this task is to develop a representation for sentences, that captures the semantics conveyed by it.

Most unsupervised representations proposed in the past rely on the composition of vector embeddings for the words, through either additive, multiplicative, or other ways (Mitchell & Lapata, 2008; Arora et al., 2017; Pagliardini et al., 2017) .

As before, our aim is to represent sentences as distributional estimates to better capture the inherent uncertainty and polysemy.

We hypothesize that a sentence, S = (w 1 , w 2 , . . .

, w N ), can be efficiently represented via the Wasserstein barycenter of the distributional estimates of its words, DISPLAYFORM0 The motivation is that since the barycenter minimizes the sum of optimal transports, cf.

Eq. (3), it should result in a representation which best captures the simultaneous occurrence of the words in a sentence.

For instance, consider two probability measures which are Diracs, δ(x) and δ(y), with equal weights and under Euclidean ground metric.

Then, the Wasserstein barycenter is δ( x+y 2 ) while simple averaging gives 1 2 (δ(x) + δ(y)).

In fact, Figure 3 says it all, and compares these two kinds of averaging based on the actual distributional estimates of the words.

Hence, illustrating Figure 3: Illustrates how Wasserstein barycenter takes into account the geometry of ground space, while the Euclidean averaging just focuses on the probability mass.5 that Wasserstein barycenter is better suited than naive averaging, for applications having an innate geometry.

We refer to this representation as the Context Mover's Barycenters (CoMB) henceforth.

Averaging of point-estimates, in many variants (Iyyer et al., 2015; Arora et al., 2017; Pagliardini et al., 2017) , has been shown to be surprisingly effective for multiple NLP tasks including sentence similarity.

Interestingly, this can be seen as a special case of CoMB, when the distribution associated to a word is just a Dirac at its point estimate.

It becomes apparent that having a rich distributional estimate for a word could turn out to be advantageous.

Since with CoMB (Eq. (6)), each sentence is also a distribution over the ground space G containing the contexts, we can utilize the Context Mover's Distance (CMD) defined in Eq. (5) to define the distance between two sentences S 1 and S 2 , under a given ground metric D G as follows, DISPLAYFORM1 6.1 EXPERIMENTAL SETUPTo evaluate CoMB as an effective sentence representation, we consider 24 datasets from SemEval semantic textual similarity (STS) tasks (Agirre et al., 2012; 2013; 2014; 2015; 2016) , containing sentences from domains such as news headlines, forums, Twitter, etc.

The objective here is to give a similarity score to each sentence pair and rank them, which is evaluated against the ground truth ranking via Pearson correlation.

As a ground metric (D G ), we consider Euclidean or angular distance between the point estimates (depending upon validation performance).

The point estimates are obtained by using GloVe (Pennington et al., 2014) on the Book Corpus (Zhu et al., 2015) , and via this we also get the histogram information needed for the distributional estimate.

The representative contexts are obtained by performing K-means clustering of the point estimates with respect to angular distance.

We benchmark 6 against a variety of unsupervised methods such as Neural Bag-of-Words (NBoW) averaging of point estimates, SIF from Arora et al. (2017) who regard it as a "simple but tough-to-beat baseline" and utilize weighted NBoW averaging with principal component removal, Sent2vec (Pagliardini et al., 2017) , Skip-thought (Kiros et al., 2015) , and Word Mover's Embedding (WME; Wu et al., 2018) which is a recent variant of WMD.

For comparison, we also show the performance of recent supervised methods such as InferSent (Conneau et al., 2017) and GenSen (Subramanian et al., 2018) , although these methods are clearly at an advantage due to training on labeled corpora.

Ground Metric: GloVe.

The best results overall are in bold while best results in a group are underlined.

the vanilla CoMB is better then SIF PC removed on average across the test set.

Next, using the mixed distributional estimate (CoMB Mix ) improves the average test performance by 10%, and interestingly this is for mixing weight 0.4 (towards the point estimate).

Further, when the PC removal is carried out for point estimates during mixing (i.e., CoMB Mix + PC removed ), the average performance goes to 57.7 and results in 18% relative improvement over SIF PC removed .Ground Metric: Sent2Vec.

In parts (b, c, d) of TAB0 , we see the effect of using an improved ground metric, by employing the word vectors obtained from Sent2vec 7 .

Here, we notice that CoMB Mix + PC removed results in a performance of 66.4 and thus a relative improvement of 4% over Sent2vec's score of 63.8.

This is a decent gain considering that for unstructured text corpora, Sent2vec is a state-of-the-art unsupervised method.

Next, WME performs much worse than CoMB, showing the benefit of defining transport over contexts than words.

Further, CoMB also outperforms popular supervised sentence embedding methods 8 such as GenSen and InferSent which utilize labeled corpora.

Ablation studies and Qualitative analysis.

We perform an extensive ablation and qualitative analysis for sentence similarity.

Due to space constraints, we enumerate the main observations here and details can be found in the supplementary section as follows.

(a) Section S4.3: CoMB and SIF appear complementary in the nature of errors they make.

CoMB outperforms when the difference in sentences stems from predicate while SIF is better when the distinguishing factor is the subject of the sentences.

This is likely the reason why mixed distributional estimate helps in practice.

(b) Section S2.3: we observe that by around K = 300 to 500, the performance gained by increasing the number of clusters starts to plateau, implying that it is sufficient to only consider the representative contexts.

(c) Section S4.4: CoMB generally fares better than SIF on datasets with longer sentences.

Summary and further prospects.

We observe that using CoMB along with either GloVe or Sent2vec leads to a substantial boost, even taking beyond the performance of popular supervised methods such as GenSen and InferSent.

Starting from the raw co-occurrence information, it takes less than 11 minutes to get all the STS results and see S1.4 for details.

A future avenue would be to utilize the important property of non-associativity for Wasserstein barycenters (i.e., DISPLAYFORM0 ).

This implies that we can take into account the word order with various aggregation strategies, like parse trees, to build the sentence representation by recursively computing barycenters phrase by phrase.

However, this remains beyond the scope of this paper.

Overall, this highlights the advantage of having distributional estimates for words, that can be extended to give a meaningful representation of sentences via CoMB in a principled manner.

In linguistics, hypernymy is a relation between words (or sentences) where the semantics of one word (the hyponym) are contained within that of another word (the hypernym).

A simple form of this relation is the is-A relation, e.g., cat is an animal.

Hypernymy is a special case of the more general concept of lexical entailment, the detection of which is relevant for tasks such as Question Answering (QA). .

These methods have proven to be powerful, as they not only capture the semantics but also the uncertainty about the contexts in which the word appears.

Therefore, hypernymy detection is a great testbed to verify the effectiveness of our approach to represent each entity by the distribution of its contexts.

The intuitive idea for the applicability of our method to this task originates from the Distributional Inclusion Hypothesis (Geffet & Dagan, 2005) , which states that a word v entails another word w if "the most characteristic contexts of v are expected to be included in all w's contexts (but not necessarily amongst the most characteristic ones for w)".

The inclusion of the contexts for the words rock and music is illustrated in FIG4 .

We see our method as a relaxation of this strict inclusion condition, as a suitable entailment based ground metric in combination with CMD can more flexibly model this condition.

Hence, it is natural to make use of the Context Mover's Distance (CMD), Eq. FORMULA8 , but with an appropriate ground cost that measures entailment relations well.

For this purpose, we utilize a recently proposed method by (Henderson & Popa, 2016; Henderson, 2017) which explicitly models what information is known about a word, by interpreting each entry of the embedding as the degree to which a certain feature is present.

Based on the logical definition of entailment they derive an operator measuring the entailment similarity between two so-called entailment vectors defined as follows: DISPLAYFORM0 , where the sigmoid σ and log are applied component-wise on the embeddings v i , v j .

Thus, we use as ground cost D DISPLAYFORM1 This asymmetric ground cost also shows that our framework can be flexibly used with an arbitrary cost function defined on the ground space.

Evaluation.

In total, we evaluate our method on 10 standard datasets: BLESS (Baroni & Lenci, 2011 ), EVALution (Santus et al., 2015 , (Benotto, 2015) , (Weeds et al., 2014) The foremost thing that we would like to check is the benefit of having a distributional estimate in comparison to just the point embeddings.

Here, we observe that employing CMD along with the entailment embeddings, leads to a significant boost on most of the datasets, except on Baroni and Turney, where the performance is still competitive with the other state of the art methods like Gaussian embeddings.

The more interesting observation is that on some datasets (EVALution, HypeNet, LenciBenotto) we even outperform or match state-of-the-art performance (cf.

TAB3 , by simply using CMD together with this ground cost D

based on the entailment embeddings.

Notably, this approach is not specific to the entailment vectors from (Henderson, 2017) and more accurate set of vectors might help additionally.

Alternatively, this also suggests that using CMD along with a method that produces embedding vectors (specialized for measuring the degree of entailment) can be a potential way to further improve the performance of that method.

Some qualitative results taken from the BIBLESS dataset are listed in TAB5

We advocate for representing entities by a distributional estimate on top of any given co-occurrence structure.

For each entity, we jointly consider the histogram information (with its contexts) as well as the point embeddings of the contexts.

We show how this enables the use of optimal transport over distributions of contexts.

Our framework results in an efficient, interpretable and compositional metric to represent and compare entities (e.g. words) and groups thereof (e.g. sentences), while leveraging existing point embeddings.

We demonstrate its performance on several NLP tasks such as sentence similarity and word entailment detection.

Thus, a practical take-home message is: do not throw away the co-occurrence information (e.g. when using GloVe), but instead pass it on to our method.

Motivated by the promising empirical results, applying the proposed framework on co-occurrence structures beyond NLP is an exciting direction.

In these appendices, we provide supplementary details on the experiments, mathematical framework, and detailed results in Section S1.

In Section S2 we discuss computational aspects and the importance of clustering the contexts.

Detailed results of the sentence representation and hypernymy detection experiments are listed on the following pages in Section S3 and S6 respectively.

Then we describe a qualitative analysis of sentence similarity in Section S4, and finally discuss a qualitative analysis of hypernymy detection in Section S7.

In this Section, we give further details on the experimental framework in Section S1.1, on the PPMI formulation (Section S1.2), and on Optimal Transport (Section S1.3).

In Section S1.5, we provide references for software release.

Sentence Representations.

While using the Toronto Book Corpus, we remove the errors caused by crawling and pre-process the corpus by filtering out sentences longer than 300 words, thereby removing a very small portion (500 sentences out of the 70 million sentences).

We utilize the code S1 from GloVe for building the vocabulary of size 205513 (obtained by setting min count=10) and the co-occurrence matrix (considering a symmetric window of size 10).

Note that as in GloVe, the contribution from a context word is inversely weighted by the distance to the target word, while computing the co-occurrence.

The vectors obtained via GloVe have 300 dimensions and were trained for 75 iterations at a learning rate of 0.005, other parameters being the default ones.

The performance of these vectors from GloVe was verified on standard word similarity tasks.

Hypernymy Detection.

The training of the entailment vector is performed on a Wikipedia dump from 2015 with 1.7B tokens that have been tokenized using the Stanford NLP library (Manning et al., 2014) .

In our experiments, we use a vocabulary with a size of 80'000 and word embeddings with 200 dimensions.

We followed the same training procedure as described in Henderson (2017) and were able to reproduce their scores on the hypernymy detection task.

For tuning the hyperparameters, we utilize the HypeNet training set of Shwartz et al. (2016) (from the random split), following the procedure indicated in Chang et al. FORMULA5 for tuning DIVE and Gaussian embeddings.

Formulation and Variants.

Typically, the probabilities used in PMIare estimated from the cooccurrence counts #(w, c) in the corpus and lead to DISPLAYFORM0 where, #(w) = c #(w, c), #(c) = w #(w, c) and |Z| = w c #(w, c).

Also, it is known that PPMI is biased towards infrequent words and assigns them a higher value.

A common solution is to smoothen S2 the context probabilities by raising them to an exponent of α lying between 0 and 1.

Levy & Goldberg (2014b) have also suggested the use of the shifted PPMI (SPPMI) matrix where the shift S3 by log(s) acts like a prior on the probability of co-occurrence of target and context pairs.

These variants of PPMI enable us to extract better semantic associations from the co-occurrence matrix.

Finally, we have DISPLAYFORM1 Computational aspect.

We utilize the sparse matrix support of Scipy S4 for efficiently carrying out all the PPMI computations.

PPMI Column Normalizations.

In certain cases, when the PPMI contributions towards the partitions (or clusters) have a large variance, it can be helpful to consider the fraction of C k 's SPPMI (Eq. (9), (10)) that has been used towards a word w, instead of aggregate values used in (13).

Otherwise the process of making the histogram unit sum might misrepresent the actual underlying contribution.

We call this PPMI column normalization (β).

In other words, the intuition is that the normalization will balance the effect of a possible non-uniform spread in total PPMI across the clusters.

We observe that setting β to 0.5 or 1 help in boosting performance on the STS tasks.

The basic form of column normalization is shown in (10).

DISPLAYFORM2 Another possibility while considering the normalization to have an associated parameter β that can interpolate between the above normalization and normalization with respect to cluster size.

DISPLAYFORM3 In particular, when β = 1, we recover the equation for histograms as in FORMULA18 , and β = 0 would imply normalization with respect to cluster sizes.

Implementation aspects.

We make use of the Python Optimal Transport (POT) S5 for performing the computation of Wasserstein distances and barycenters on CPU.

For more efficient GPU implementation, we built custom implementation using PyTorch.

We also implement a batched version for barycenter computation, which to the best of our knowledge has not been done in the past.

The batched barycenter computation relies on a viewing computations in the form of block-diagonal matrices.

As an example, this batched mode can compute around 200 barycenters in 0.09 seconds, where each barycenter is of 50 histograms (of size 100) and usually gives a speedup of about 10x.

Scalability.

For further scalability, an alternative is to consider stochastic optimal transport techniques (Genevay et al., 2016) .

Here, the idea would be to randomly sample a subset of contexts from the distributional estimate while considering this transport.

Stability of Sinkhorn Iterations.

For all our computations involving optimal transport, we typically use λ around 0.1 and make use of log or median normalization as common in POT to stabilize the Sinkhorn iterations.

Also, we observe that clipping the ground metric matrix (if it exceeds a particular large threshold) also sometimes results in performance gains.

S4 https://docs.scipy.org/doc/scipy/reference/sparse.html S5 http://pot.readthedocs.io/en/stable/ Value of p.

It has been shown in Agueh & Carlier (2011) that when the underlying space is Euclidean and p = 2, there exists a unique minimizer to the Wasserstein barycenter problem.

But, since we are anyways solving the regularized Wasserstein barycenter (Cuturi & Doucet, 2014) problem over here instead of the exact one, the particular value of p seems less of an issue.

Empirically in the sentence similarity experiments, we have observed p = 1 to perform better than p = 2 (by about 2-3 points).

Starting from scratch, it takes less than 11 minutes to get the results on all STS tasks which contains 25,000 sentences.

This includes about 3 minutes to cluster 200,000 words (1 GPU), 5 minutes to convert raw co-occurrences into histograms of size 300 (1 CPU core) and 3 minutes for STS (1 GPU).

Core code and histograms.

Our code to build the ppmi-matrix, clusters, histograms as well computing Wasserstein distances and barycenters is publicly available on Github under https: //github.com/context-mover.

Precomputed histograms, clusters and point embeddings used in our experiments can also be downloaded from https://drive.google.com/open?

id=13stRuUd--71hcOq92yWUF-0iY15DYKNf.

Standard evaluation suite for Hypernymy.

To ease the evaluation pipeline, we have collected the most common benchmark datasets and compiled the code for assessing a model's performance on hypernymy detection or directionality into a Python package, called HypEval, which is publicly available at https://github.com/context-mover/HypEval.

This also handles OOV (out-of-vocabulary) pairs in a standardized manner and allows for efficient, batched evaluation on GPU.

In this Section, we discuss computational aspects and how using clustering makes the problem scalable.

We give precise definition of the distributional estimate in Section S2.1, and show how the number of clusters affects the performance in Section S2.3.

The view of optimal transport between histograms of contexts introduced in Eq. (5) offers a pleasing interpretation (see FIG1 .

However, it might be computationally intractable in its current formulation, since the number of possible contexts can be as large as the size of vocabulary (if the contexts are just single words) or even exponential (if contexts are considered to be phrases, sentences and otherwise).

For instance, even with the use of SPPMI matrix, which also helps to sparsify the co-occurrences, the cardinality of the support of histograms still varies from 10 3 to 5 × 10 4 context words, when considering a vocabulary of size around 2 × 10 5 .

This is problematic because the Sinkhorn algorithm for regularized optimal transport (Cuturi, 2013, see Section 3) scales roughly quadratically in the histogram size, and the ground cost matrix can also become prohibitive to store in memory.

One possible fix is to instead consider a set of representative contexts in this ground space, for example via clustering.

We believe that with dense low-dimensional embeddings and a meaningful metric between them, we may not require as many contexts as needed before.

For instance, this can be achieved by clustering the contexts with respect to metric D G .

Apart from the computational gain, the clustering will lead to transport between more abstract contexts.

This will although come at the loss of some interpretability.

Now, consider that we have obtained K representative contexts, each covering some part C k of the set of contexts C. The histogram for word w with respect to these contexts can then be written as: DISPLAYFORM0 Hereṽ k ∈Ṽ is the point estimate of the k th representative context, and (H w ) k denote the new histogram bin values with respect to the part C k , DISPLAYFORM1 , with (13) DISPLAYFORM2 In the following subsection, we show the effect of the number of clusters on the performance.

For clustering, we make use of kmcuda's S6 efficient implementation of K-Means algorithm on GPUs.

Here, we analyze the impact of number of clusters on the performance of Context Mover's Barycenters (CoMB) for the sentence similarity experiments (cf.

Section 6).

In particular, we look at the three best performing variants (A, B, C) on the validation set (STS 16) as well as averaged across them.

The 'avg' plot shows the average trend across these three configurations.

We observe in FIG0 that on average the performance significantly improves when the number of clusters are increased until around K = 300, and beyond that mostly plateaus (± 0.5).

But, as can be seen for variants B and C the performance typically continues to rise until K = 500.

It seems that the amount of PPMI column normalization (β = 0.5 vs β = 1) might be at play here.

As going from K = 300 to K = 500 comes at the cost of increased computation time, and doesn't lead to a substantial gain in performance.

We use either K = 300 or 500 clusters, depending on validation results, for our results on sentence similarity tasks.

Such a trend seems to be in line with the ideal case where we wouldn't need to do any clustering and just take all possible contexts into account.

Thus, it suggests that better ways (other than clustering) to deal with this problem might further boost the performance.

We provide detailed results of the test set performance of Context Mover's Barycenters (CoMB) and related baselines on the STS-12, 13, 14 and STS-15 tasks in TAB0 and validation set performance in TAB5 .

TAB5 : Detailed validation set performance of Context Mover's Barycenters (CoMB) and related baselines on the STS16 using Toronto Book Corpus.

The numbers are average Pearson correlation x100 (with respect to groundtruth scores). '

Mix' denotes the mixed distributional estimate.

'PC removed' refers to removing contribution along the principal component of point estimates as done in SIF.

The part in brackets after CoMB refers to the underlying ground metric.

Note that, STS16 was used as the validation set to obtain the best hyperparameters for all the methods in these experiments.

As a result, high performance on STS16 may not be indicative of the overall performance.embeddings (second part of the Tables).

The numbers are average Pearson correlation (with respect to ground-truth scores).We observe empirically that the PPMI smoothing parameter α, which balances the bias of PPMI towards rare words, plays an important role.

While its ideal value would vary on each task, we found the settings mentioned in the Table S4 to work well uniformly across the above spectrum of tasks.

We also provide in Table S4 a comparison of the hyper-parameters used in each of the methods in TAB0 and S3 Table S4 : Detailed parameters for the methods presented in TAB0 and S3.

The parameters for CoMB α, β, s denote the PPMI smoothing, column normalization exponent (Eq. FORMULA19 ), and k-shift.

In this section, we aim to qualitatively analyse the particular examples where our method, Context Mover's Barycenters (CoMB), performs better or worse than the Smooth Inverse Frequency (SIF) approach from Arora et al. FORMULA5 .

Comparing by rank.

It doesn't make much sense to compare the raw distance values between two sentences as given by Context Mover's Distance (CMD) for CoMB and cosine distance for SIF.

This is because the spread of distance values across sentence pairs can be quite different.

Note that the quantitative evaluation of these tasks is also carried out by Pearson/Spearman rank correlation of the predicted distances/similarities with the ground-truth scores.

Thus, in accordance with this reasoning, we compare the similarity score of a sentence pair relative to its rank based on ground-truth score (amongst the sentence pairs for that dataset).

So, the better method should rank sentence pairs closer to the ranking obtained via ground-truth scores.

Ground-Truth Score Implied meaning

The two sentences are completely equivalent, as they mean the same thing.

4The two sentences are mostly equivalent, but some unimportant details differ.

3The two sentences are roughly equivalent, but some important information differs/missing.

2The two sentences are not equivalent, but share some details.

1The two sentences are not equivalent, but are on the same topic.

0The two sentences are completely dissimilar.

Table S5 : STS ground scores and their implied meanings, as taken from Agirre et al. FORMULA5 Ground-truth details.

The ground-truth scores (can be fractional) and range from 0 to 5, and the meaning implied by the integral score values can be seen in the Table S5.

In the case where different examples have the same ground-truth score, the ground-truth rank is then based on lexicographical ordering of sentences for our qualitative evaluation procedure. (This for instance means that sentence pairs ranging from 62 to 74 would correspond to the same ground-truth score of 4.6).

The ranking is done in the descending order of sentence similarity, i.e., most similar to least similar.

Example selection criteria.

For all the examples, we compare the best variants of CoMB and SIF on those datasets.

We particularly choose those examples where there is the maximum difference in ranks according to CoMB and SIF, as they would be more indicative of where a method succeeds or fails.

Nevertheless, such a qualitative evaluation is subjective and is meant to give a better understanding of things happening under the hood.

We look at examples from three datasets, namely: Images from STS15, News from STS14 and WordNet from STS14 to get a better idea of an overall behavior.

In terms of aggregate quantitative performance, on Images and News datasets, CoMB is better than SIF, while the opposite is true for WordNet.

These examples across the three datasets may not probably be exhaustive and are up to subjective interpretation, but hopefully will lend some indication as to where and why each method works.

We look in detail at the examples in News dataset from STS 2014 (Agirre et al., 2014) .

The results of qualitative analysis on Images and WordNet datasets can be found in Section S4.5.

For reference, CoMB results in a better performance overall with a Pearson correlation (x100) of 64.9 versus 43.0 for SIF, as presented in TAB0 .

The main observations are:Observation 1.

Examples 1, 2, 4, 5 are sentence pairs which are equivalent in meaning (cf .

Table S5 ), but typically have additional details in the predicates of the sentences.

Here, CoMB is better than SIF at ranking the pairs closer to the ground-truth ranking.

This probably suggests the averaging of word embeddings, which is the 1 st step in SIF, is not as resilient to the presence of such details than the Wasserstein barycenter of distributional estimates in CoMB.

We speculate that when having distributional estimates (where multiple senses or contexts are considered), adding details can help towards refining the particular meaning implied.

Let's consider the examples 3 and 6 where SIF is better than CoMB.

These are sentence pairs which are equivalent or roughly equivalent in meanings, but with a few words substituted (typically subjects) like "judicial order" instead of "court" in example 3.

Here it seems that the substitution is adverse for CoMB while considering varied senses through the distributional estimate, in comparison to looking at the "point" meaning given by SIF.Observation 3.

In 7, 8, and 10, each sentence pair is about a common topic, but the meaning of individual sentences is quite different.

For instance, example 8: "south korea launches new bullet train reaching 300 kph" & "south korea has had a bullet train system since the 1980s".

Or like in example 10: "china is north korea ' s closest ally" & "north korea is a reclusive state".

Note that typically in these examples, the subject is same in a sentence pair, and the difference is mainly in the predicate.

Here, CoMB identifies the difference and ranks them closer to the ground-truth.

Whereas, SIF fails to understand this and ranks them as more similar (and far away) than the ground-truth.

Observation 4.

The examples 9, 11, and 12 are related sentences and differ mainly in details such as the name of the country, person, department, i.e. proper nouns.

In particular, consider example 9: "south korea and israel oppose proliferation of weapons of mass destruction and an arms race" & "china will resolutely oppose the proliferation of mass destructive weapons".

The main difference in these examples stems from differences in the subject rather than the predicate.

CoMB considers these sentence pairs to be more similar than suggested by ground-truth.

Hence, in such scenarios where the subject (like the particular proper nouns) makes the most difference, SIF seems to be better.

Summarizing the observations from the above qualitative analysis on News dataset S7 , we conclude the following about the nature of success or failures of each method.• When the subject of the sentence is similar and main difference stems from the predicate, CoMB is the winner.

This can be seen for both the case when predicates are equivalent but described distinctly (observation 1) and when predicates are not equivalent (observation 3).• When the predicates are similar and the distinguishing factor is in the subject (or object), SIF takes the lead.

This seems to be true for both scenarios when the subject used increases or decreases the similarity as measured by CoMB, (observations 2 and 4).S7 Similar findings can also be seen for the two other datasets in Section S4.5.

Table S6 : Examples of some indicative sentence pairs, from News dataset in STS14, with ground-truth scores and ranking as obtained via (best variants of) CoMB and SIF.

The total number of sentences is 300 and the ranking is done in descending order of similarity.

The method which ranks an example closer to the ground-truth rank is better and is highlighted in blue.

CoMB ranking is the one produced when representing sentences via CoMB and then using CMD to compare them.

SIF ranking is when sentences are represented via SIF and then employing cosine similarity.•

The above two points in a way also signify where having distributional estimates can be better or worse than point estimates.• CoMB and SIF appear to be complementary in the kind of errors they make.

Hence, combining the two is an exciting future avenue.

Lastly, it also seems worthwhile to explore having different ground metrics for CoMB and CMD (which are currently shared).

The ground metric plays a crucial role in performance and the nature of these observations.

Employing a ground metric(s) that better handles the above subtleties would be a useful research direction.

In this section, we look at the length of sentences across all the datasets in each of the STS tasks.

Average sentence length is one measure of the complexity of a particular dataset.

But looking at just sentence lengths may not give a complete picture, especially for the textual similarity tasks where there can be many words common between the sentence pairs.

The Table S7 shows the various statistics of each dataset, with respect to the sentence lengths along with the better method on each of them (out of CoMB and SIF).

Table S7 : Analysis of sentence lengths in each of the datasets from STS12, STS13, STS14, and STS15.

Along with the average sentence lengths, we also measure average word overlap in the sentence pair and thus the average effective sentence length (i.e., after excluding the overlapping/common words in the sentence pair).

For reference, we also show which out of CoMB or SIF performs better.

On STS14-Twitter, the difference in performance isn't significant and we thus write 'equal' in the corresponding cell.

Observations.• We notice that on datasets with longer effective sentence lengths, CoMB performs better than SIF on average.

There might be other factors at play here, but if one had to pick on the axis of effective sentence length, CoMB leads over SIF S8 .S8 Effective sentence length averaged across datasets where CoMB is better is 7.48.

Contrast this to an average effective sentence length of 5.03 across datasets where SIF is better.• The above statement also aligns well with the observation 1 from the qualitative analysis (cf.

Section S4.2.1), that having more details can help in refining the particular meaning or sense implied by CoMB.

(Effective sentence length can serve as a good proxy for indicating the amount of details.)

• It also seems to explain why both methods don't perform well (see TAB0 ) on STS13-FNWN, which has on average the maximum effective sentence length (of 20.4).• To an extent, it also points towards the effect of corpora.

For instance, in a corpus such as WordNet, which has a low average sentence length and with examples typically concerned about word definitions (see Table S9 ), SIF seems to be better of the methods.

On the other hand, CoMB seems to be better for News (Table S6) , Image captions (Table S8) or Forum.

S4.5 ADDITIONAL QUALITATIVE ANALYSIS S4.5.1 TASK: STS15, DATASET: IMAGES We consider the sentence pairs from Images dataset in STS15 task (Agirre et al., 2015) , as presented in Table S8 .

As a reminder, CoMB outperforms SIF on this dataset with a Pearson correlation (x100) of 61.8 versus 51.7, as mentioned in TAB3 .

The main observations are: Table S8 : Examples of some indicative sentence pairs, from Images dataset in STS15, with groundtruth scores and ranking as obtained via (best variants of) CoMB and SIF.

The total number of sentences is 750 and the ranking is done in descending order of similarity.

The method which ranks an example closer to the ground-truth rank is better and is highlighted in blue.

CoMB ranking is the one produced when representing sentences via CoMB and then using CMD to compare them.

SIF ranking is when sentences are represented via SIF and then employing cosine similarity.

Observation A. Example 1 to 5 indicate pairs of sentences which are essentially equivalent in meaning, but with varying degrees of equivalence.

Here, we can see that CoMB with CMD is able to rank the similarity between these pairs quite well in comparison to SIF, even when their way of describing is different.

For instance, example 2 : "a boy waves around a sparkler" & "a young boy is twisting a sparkler around in the air".

This points towards the benefit of having multiple senses or contexts encoded through the distributional estimate in CoMB.Observation B. Next, in the examples 7 to 10, which consist of sentence pairs that are not equivalent but have commonalities (about the topic).

Here, SIF ranks the sentences closer to the ground-truth ranking while CoMB interprets these pairs as being more common in meaning than given by ground-truth.

This could be the consequence of comparing the various senses or contexts implied by the sentence pairs via CMD.

Take for instance, example 10, "three dogs running in the dirt" & "the yellow dog is running on the dirt road".

Since these sentences are about the similar topic (and the major difference is in their subject), this can result in CMD considering them more similar than cosine distance.

Observation C. For sentences which are completely dissimilar as per ground-truth, let's look at example 11 and 12.

Consider 11, which is "a little girl and a little boy hold hands on a shiny slide" & "a little girl in a paisley dress runs across a sandy playground", the sentences meaning totally different things and CoMB seems to be better at ranking than SIF.

But, consider example 12: "a little girl walks on a boardwalk with blue domes in the background" & "a man going over a jump on his bike with a river in the background".

One common theme S9 can be thought as "a person moving with something blue in the background", which can result in CoMB ranking the sentence as more similar.

SIF also ranks it higher (at 591) than ground-truth (696), but is more closer than CoMB which ranks it at 310.

Table S9 : Examples of some indicative sentence pairs, from WordNet dataset in STS14, with groundtruth scores and ranking as obtained via (best variants of) CoMB and SIF.

The total number of sentences is 750 and the ranking is done in descending order of similarity.

The method which ranks an example closer to the ground-truth rank is better and is highlighted in blue.

CoMB ranking is the one produced when representing sentences via CoMB and then using CMD to compare them.

SIF ranking is when sentences are represented via SIF and then employing cosine similarity.

Lastly, we discuss the examples and observations derived from the qualitative analysis on WordNet dataset from STS14 (Agirre et al., 2014) .

This dataset is comprised of sentences which are the definitions of words/phrases, and sentence length is typically smaller than the datasets discussed S9 Of course, this is upto subjective interpretation.

Published as a workshop paper at ICLR 2019 before.

For reference, SIF (76.8) does better than CoMB (64.4) in terms of average Pearson correlation (x100), as mentioned in TAB0 .Observation D. Consider examples 1 to 6 as shown in Table S9 , which fall in the category of equivalent sentences but in varying degrees.

The sentence pairs essentially indicate different ways of characterizing equivalent things.

Here, CoMB is able to rank the similarity between sentences in a better manner than SIF.

Specifically, see example 2: "( cause to ) sully the good name and reputation of" & "charge falsely or with malicious intent ; attack the good name and reputation of someone".

It seems that SIF is not able to properly handle the additional definition present in sentence 2 and ranks this pair much lower in similarity at 534 versus 235 for CoMB.

This is also in line with observation 1 about added details in the Section S4.2.1.Observation E. In the examples 7 to 9, where CoMB doesn't do well in comparison to SIF, mainly have a slight difference in the object of the sentence.

For instance, in example 9: "a person who is a member of the senate" & "a person who is a member of a partnership".

So based on the kind of substituted word, looking at its various contexts via the distributional estimate can make it more or less similar than desired.

In such cases, using the "point" meanings of the objects seems to fare better.

This also aligns with the observations 2 and 4 in the Section S4.2.1.

Here, we would like to qualitatively probe the kind of results obtained when computing Wasserstein barycenter of the distributional estimates, in particular, when using CoMB to represent sentences.

To this end, we consider a few simple sentences and find the closest word in the vocabulary for CoMB (with respect to CMD) and contrast it to SIF with cosine distance.

TAB0 : Top 10 closest neighbors for CoMB and SIF (no PC removed) found across the vocabulary, and sorted in ascending order of distance from the query sentence.

Words in italics are those which in our opinion would fit well when added to one of the places in the query sentence.

Note that, both CoMB (under current formulation) and SIF don't take the word order into account.

Observations.

We find that closest neighbors (see TAB0 ) for CoMB consist of relatively more diverse set of words which fit well in the context of given sentence.

For example, take the sentence "i love her", where CoMB captures a wide range of contexts, for example, "i actually love her", "i love her because", "i doubt her love" and more.

Also for an ambiguous sentence "he lives in europe for", the obtained closest neighbors for CoMB include: 'decades', 'masters', 'majority', 'commerce' , etc., while with SIF the closest neighbors are mostly words similar to one of the query words.

Further, if you look at the last three sentences in the Table S10, the first closest neighbor for CoMB even acts as a good next word for the given query.

This suggests that CoMB might perform well on the task of sentence completion, but this additional evaluation is beyond the scope of this paper.

In this Section, we provide detailed results for the hypernymy detection in Section S6.1 and mention the corresponding hyperparamters in Section S6. .

We also indicate the average gain in performance across these 10 datasets by using CMD along with the entailment vectors.

All scores are AP at all (%).

The above listed variants of CMD are the ones with best validation performance on HypeNet-Train (Shwartz et al., 2016) .

The other hyperparameters (common) for both of them are as follows:• PPMI smoothing, α = 0.5.• PPMI column normalization exponent, β=0.5.• PPMI k-shift, s=1.• Regularization constant for Wasserstein distance, λ=0.1• Number of Sinkhorn iterations = 500.• Log normalization of Ground Metric.

Out of Vocabulary Details.

The following shows the out of vocabulary information for entailment experiments.

This table was generated during an earlier version of the paper, when we were not considering the validation on HypeNet-Train.

Hence, the above table doesn't contain numbers on HypeNet-Test, but an indication of performance on it can be seen in Section S11.

In any case, this table suggests that our method works well for several PPMI hyper-parameter configurations.

Here, our objective is to qualitatively analyse the particular examples where our method of using Context Mover's Distance (CMD) along with embeddings from Henderson (2017) performs better or worse than just using these entailment embeddings alone.

Comparing by rank.

Again as in the qualitative analysis with sentence similarity, it doesn't make much sense to compare the raw distance/similarity values between two words as their spread across word pairs can be quite different.

We thus compare the ranks assigned to each word pair by both the methods.

Ground-truth details.

In contrast to graded ground-truth scores in the previous analysis, here we just have a binary ground truth: 'True' if the hyponym-hypernym relation exists and 'False' when it doesn't.

We consider the BIBLESS dataset (Kiela et al., 2015) for this analysis, which has a total of 1668 examples.

Out of these, 33 word pairs are not in the vocabulary (see TAB0 ), so we ignore them for this analysis.

Amongst the 1635 examples left, 814 are 'True' and 821 are 'False'.

A perfect method should rank the examples labeled as 'True' from 1 to 814 and the 'False' examples from 815 to 1635.

Of course, achieving this is quite hard, but the better of the methods should rank as many examples in the desired ranges.

Example selection criteria.

We look at the examples where the difference in ranks as per the two methods is the largest.

Also, for a few words, we also look at how each method ranks when present as a hypernym and a hyponym.

If the difference in ranks is defined as, CMD rank -Henderson Rank, we present the top pairs where this difference is most positive and most negative.

For reference on the BIBLESS dataset, CMD performs better than Henderson embeddings quantitatively (cf.

TAB3 ).

Let's take a look at some word pairs to get a better understanding.

These are essentially examples where CMD considers the entailment relation as 'False' while the Henderson embeddings predict it as 'True', and both are most certain about their decisions.

TAB0 shows these pairs, along with ranks assigned by the two methods and the ground-truth label for reference.

Some quick observations: many of the word pairs which the Henderson method gets wrong are co-hyponym pairs, such as: ('banjo', 'flute'), ('guitar', 'trumpet'), ('turnip, 'radish').

Additionally, ('bass', 'cello' ), ('creature', 'gorilla'), etc., are examples where the method has to assess not just if the relation exists, but also take into account the directionality between the pair, which the Henderson method seems unable to do.

Now the other way around, these are examples where CMD considers the entailment relation as 'True' while the Henderson embeddings predict it as 'False', and both are most certain about their decisions.

TAB0 shows these pairs.

The examples where CMD performs poorly like, ('box', 'mortality'), ('pistol', 'initiative') seem to be unrelated and we speculate that matching the various contexts or senses of the distributional estimate causes this behavior.

One possibility to deal with this can be to take into account the similarity between word pairs in the ground metric.

Overall, CMD does a good job at handling these pairs in comparison to the Henderson method.

<|TLDR|>

@highlight

Represent each entity as a probability distribution over contexts embedded in a ground space.

@highlight

Proposes to construct word embeddings from a histogram over context words, instead of as point vectors, which allows for measuring distances between two words in terms of optimal transport between the histograms through a method that augments representation of an entity from standard "point in a vector space" to a histogram with bins located at some points in that vector space. 