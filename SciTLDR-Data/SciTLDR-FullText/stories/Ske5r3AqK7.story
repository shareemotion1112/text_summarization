Words are not created equal.

In fact, they form an aristocratic graph with a latent hierarchical structure that the next generation of unsupervised learned word embeddings should reveal.

In this paper, justified by the notion of delta-hyperbolicity or tree-likeliness of a space, we propose to embed words in a Cartesian product of hyperbolic spaces which we theoretically connect to the Gaussian word embeddings and their Fisher geometry.

This connection allows us to introduce a novel principled hypernymy score for word embeddings.

Moreover, we adapt the well-known Glove algorithm to learn unsupervised word embeddings in this type of Riemannian manifolds.

We further explain how to solve the analogy task using the Riemannian parallel transport that generalizes vector arithmetics to this new type of geometry.

Empirically, based on extensive experiments, we prove that our embeddings, trained unsupervised, are the first to simultaneously outperform strong and popular baselines on the tasks of similarity, analogy and hypernymy detection.

In particular, for word hypernymy, we obtain new state-of-the-art on fully unsupervised WBLESS classification accuracy.

Word embeddings are ubiquitous nowadays as first layers in neural network and deep learning models for natural language processing.

They are essential in order to move from the discrete word space to the continuous space where differentiable loss functions can be optimized.

The popular models of Glove BID31 , Word2Vec BID25 or FastText BID6 , provide efficient ways to learn word vectors fully unsupervised from raw text corpora, solely based on word co-occurrence statistics.

These models are then successfully applied to word similarity and other downstream tasks and, surprisingly (or not BID1 ), exhibit a linear algebraic structure that is also useful to solve word analogy.

However, unsupervised word embeddings still largely suffer from revealing asymmetric word relations including the latent hierarchical structure of words.

This is currently one of the key limitations in automatic text understanding, e.g. for tasks such as textual entailment BID9 .

To address this issue, BID36 BID27 propose to move from point embeddings to probability density functions, the simplest being Gaussian or Elliptical distributions.

Their intuition is that the variance of such a distribution should encode the generality/specificity of the respective word.

However, this method results in losing the arithmetic properties of point embeddings (e.g. for analogy reasoning) and becomes unclear how to properly use them in downstream tasks.

To this end, we propose to take the best from both worlds: we embed words as points in a Cartesian product of hyperbolic spaces and, additionally, explain how they are bijectively mapped to Gaussian embeddings with diagonal covariance matrices, where the hyperbolic distance between two points becomes the Fisher distance between the corresponding probability distribution functions (PDFs).

This allows us to derive a novel principled is-a score on top of word embeddings that can be leveraged for hypernymy detection.

We learn these word embeddings unsupervised from raw text by generalizing the Glove method.

Moreover, the linear arithmetic property used for solving word analogy has a mathematical grounded correspondence in this new space based on the established notion of parallel transport in Riemannian manifolds.

In addition, these hyperbolic embeddings outperform Euclidean Glove on word similarity benchmarks.

We thus describe, to our knowledge, the first word embedding model that competitively addresses the above three tasks simultaneously.

Finally, these word vectors can also be used in downstream tasks as explained by BID17 .We provide additional reasons for choosing the hyperbolic geometry to embed words.

We explain the notion of average δ-hyperbolicity of a graph, a geometric quantity that measures its "democracy" BID8 .

A small hyperbolicity constant implies "aristocracy", namely the existence of a small set of nodes that "influence" most of the paths in the graph.

It is known that real-world graphs are mainly complex networks (e.g. scale-free exhibiting power-law node degree distributions) which in turn are better embedded in a tree-like space, i.e. hyperbolic BID20 .

Since, intuitively, words form an "aristocratic" community (few generic ones from different topics and many more specific ones) and since a significant subset of them exhibits a hierarchical structure (e.g. WordNet BID26 ), it is naturally to learn hyperbolic word embeddings.

Moreover, we empirically measure very low average δ-hyperbolicity constants of some variants of the word log-co-occurrence graph (used by the Glove method), providing additional quantitative reasons for why spaces of negative curvature (i.e. hyperbolic) are better suited for word representations.

Recent supervised methods can be applied to embed any tree or directed acyclic graph in a low dimensional space with the aim of improving link prediction either by imposing a partial order in the embedding space BID35 BID3 , by using hyperbolic geometry BID30 ), or both (Ganea et al., 2018a .To learn word embeddings that exhibit hypernymy or hierarchical information, supervised methods (Vulić & Mrkšić, 2018; BID28 leverage external information (e.g. WordNet) together with raw text corpora.

However, the same goal is also targeted by more ambitious fully unsupervised models which move away from the "point" assumption and learn various probability densities for each word BID36 BID27 BID2 BID32 .There have been two very recent attempts at learning unsupervised word embeddings in the hyperbolic space BID21 BID14 .

However, they suffer from either not being competitive on standard tasks in high dimensions, not showing the benefit of using hyperbolic spaces to model asymmetric relations, or not being trained on realistically large corpora.

We address these problems and, moreover, the connection with density based methods is made explicit and leveraged to improve hypernymy detection.

In order to work in the hyperbolic space, we have to choose one model, among the five isometric models that exist.

We choose to embed words in the Poincaré ball D n = {x ∈ R n | x 2 < 1}. This is illustrated in Figure 1a for n = 2, where dark lines represent geodesics.

The distance function in this space is given by DISPLAYFORM0 2 ) being the conformal factor.

We will also embed words in products of hyperbolic spaces, and explain why later on.

A product of p balls (D n ) p , with the induced product geometry, is known to have distance function DISPLAYFORM1 Finally, another model of interest for us is the Poincaré half-plane DISPLAYFORM2

Euclidean GLOVE.

The GLOVE BID31 algorithm is an unsupervised method for learning word representations in the Euclidean space from statistics of word co-occurrences in a text corpus, with the aim to geometrically capture the words' meaning and relations.

We use the notations: X ij is the number of times word j occurs in the same window context as word i; X i = k X ik ; P ij = X ij /X i is the probability that word j appears in the context of word i. An embedding of a (target) word i is written w i , while an embedding of a context word k is writtenw k .The initial formulation of the GLOVE model suggests to learn embeddings as to satisfy the equation w T iw k = log(P ik ) = log(X ik ) − log(X i ).

Since X ik is symmetric in (i, k) but P ik is not, BID31 propose to restore the symmetry by introducing biases for each word, absorbing log(X i ) into i's bias: DISPLAYFORM0 (1) Finally, the authors suggest to enforce this equality by optimizing a weighted least-square loss: DISPLAYFORM1 where V is the size of the vocabulary and f down-weights the signal coming from frequent words (it is typically chosen to be f (x) = min{1, (x/x m ) α }, with α = 3/4 and x m = 100).GLOVE in metric spaces.

Note that there is no clear correspondence of the Euclidean inner-product in a hyperbolic space.

However, we are provided with a distance function.

Further notice that one could rewrite Eq. (1) with the Euclidean distance as DISPLAYFORM2 , where we absorbed the squared norms of the embeddings into the biases.

We thus replace the GLOVE loss by: DISPLAYFORM3 where h is a function to be chosen as a hyperparameter of the model, and d can be any differentiable distance function.

Although the most direct correspondence with GLOVE would suggest h(x) = x 2 /2, we sometimes obtained better results with other functions, such as h = cosh 2 (see sections 8 & 9).

Note that BID13 also apply cosh to their distance matrix for hyperbolic MDS before applying PCA.

Understanding why h = cosh 2 is a good choice would be interesting future work.

In order to endow Euclidean word embeddings with richer information, BID36 proposed to represent words as Gaussians, i.e. with a mean vector and a covariance matrix 1 , expecting the variance parameters to capture how generic/specific a word is, and, hopefully, entailment relations.

On the other hand, BID30 proposed to embed words of the WordNet hierarchy BID26 in hyperbolic space, because this space is mathematically known to be better suited to embed tree-like graphs.

It is hence natural to wonder: is there a connection between the two?The Fisher geometry of Gaussians is hyperbolic.

It turns out that there exists a striking connection BID12 .

Note that a 1D Gaussian N (µ, σ 2 ) can be represented as a point (µ, σ) in R × R * + .

Then, the Fisher distance between two distributions relates to the hyperbolic distance in H 2 : DISPLAYFORM0 For n-dimensional Gaussians with diagonal covariance matrices written Σ = diag(σ) 2 , it becomes: DISPLAYFORM1 Hence there is a direct correspondence between diagonal Gaussians and the product space (H 2 ) n .Fisher distance, KL & Gaussian embeddings.

The above paragraph lets us relate the WORD2GAUSS algorithm BID36 to hyperbolic word embeddings.

Although one could object that WORD2GAUSS is trained using a KL divergence, while hyperbolic embeddings relate to Gaussian distributions via the Fisher distance d F , let us remind that KL and d F define the same local geometry.

Indeed, the KL is known to be related to d F , as its local approximation BID19 .

In short, if P (θ + dθ) and P (θ) denote two closeby probability distributions for a small dθ, then KL(P (θ + dθ)||P (θ)) = (1/2) ij g ij dθ i dθ j + O( dθ 3 ), where (g ij ) ij is the Fisher information metric, inducing d F .Riemannian optimization.

A benefit of representing words in (products of) hyperbolic spaces, as opposed to (diagonal) Gaussian distributions, is that one can use recent Riemannian adaptive optimization tools such as RADAGRAD BID5 .

Note that without this connection, it would be unclear how to define a variant of ADAGRAD BID15 intrinsic to the statistical manifold of Gaussians.

Empirically, we indeed noticed better results using RADAGRAD, rather than simply Riemannian SGD BID7 .

Similarly, note that GLOVE trains with ADAGRAD.

The connection exposed in section 5 allows us to provide mathematically grounded (i) analogy computations for Gaussian embeddings using hyperbolic geometry, and (ii) hypernymy detection for hyperbolic embeddings using Gaussian distributions.

A common task used to evaluate word embeddings, called analogy, consists in finding which word d is to the word c, what the word b is to the word a. For instance, queen is to woman what king is to man.

In the Euclidean embedding space, the solution to this problem is usually taken geometrically DISPLAYFORM0 How should one intrinsically define "analogy parallelograms" in a space of Gaussian distributions?

Note that BID36 do not evaluate their Gaussian embeddings on the analogy task, and that it would be unclear how to do so.

However, since we can go back and forth between (diagonal) Gaussians and (products of) hyperbolic spaces as explained in section 5, we can use the fact that parallelograms are naturally defined in the Poincaré ball, by the notion of gyro-translation (Ungar, 2012, section 4) .

In the Poincaré ball, the two solutions d 1 = c + (b − a) and d 2 = b + (c − a) are respectively generalized to DISPLAYFORM1 The formulas for these operations are described in closed-forms in appendix C, and are easy to implement.

The fact that d 1 and d 2 differ is due to the curvature of the space.

For evaluation, we chose a point m d2d1 , which is at equal hyperbolic distance from d 1 as from d 2 .

We explain in appendix A.2 how to select t, and that continuously deforming the Poincaré ball to the Euclidean space (by sending its radius to infinity) lets these analogy computations recover their Euclidean counterparts, which is a nice sanity check.

DISPLAYFORM2

We now use the connection explained in section 5 to introduce a novel principled score that can be applied on top of our unsupervised learned Poincaré Glove embeddings to address the task of hypernymy detection, i.e. to predict relations of type is-a(v,w) such as is-a(dog, animal).

For this purpose, we first explain how learned hyperbolic word embeddings are mapped to Gaussian embeddings, and subsequently we define our hypernymy score.

Invariance of distance-based embeddings to isometric transformations.

The method of BID30 uses a heuristic entailment score in order to predict whether u is-a v, defined for DISPLAYFORM0 , based on the intuition that the Euclidean norm should encode generality/specificity of a concept/word.

However, such a choice is not intrinsic to the hyperbolic space when the training loss involves only the distance function.

We say that training is intrinsic to D n , i.e. invariant to applying any isometric transformation ϕ : D n → D n to all word embeddings (such as hyperbolic translation).

But their "is-a" score is not intrinsic, i.e. depends on the parametrization.

For this reason, we argue that an isometry has to be found and fixed before using the trained word embeddings in any non-intrinsic manner, e.g. to define hypernymy scores.

To discover it, we leverage the connection between hyperbolic and Gaussian embeddings as follows.

Mapping hyperbolic embeddings to Gaussian embeddings via an isometry.

For a 1D Gaussian N (µ, σ 2 ) representing a concept, generality should be naturally encoded in the magnitude of σ.

As shown in section 5, the space of Gaussians endorsed with the Fisher distance is naturally mapped to the hyperbolic upper half-plane H 2 , where the variance σ corresponds to the (positive) second coordinate in H 2 = R × R * + .

Moreover, as shown in section 3, H 2 can be isometrically mapped to DISPLAYFORM1 , any (hyperbolic) translation or any rotation w.r.t.

the origin is an isometry 2 .

Hence, in order to map a word x ∈ D 2 to a Gaussian N (µ, σ 2 ) via H 2 , we first have to find the correct isometry.

This transformation should align {0} × (−1, 1) with whichever geodesic in D 2 encodes generality.

For simplicity, we assume it is composed of a centering and a rotation operations in D 2 .

Thus, we start by identifying two sets G and S of potentially generic and specific words, respectively.

For the re-centering, we then compute the means g and s of G and S respectively, and m := (s + g)/2, and Möbius translate all words by the global mean with the operation w → m ⊕ w.

For the rotation, we set u := ( m ⊕ g)/ m ⊕ g 2 , and rotate all words so that u is mapped to (0, 1).

FIG1 and Algorithm 1 illustrate these steps.

10 with our unsupervised hyperbolic GLOVE algorithm.

This illustrates the three steps of applying the isometry.

From left to right: the trained embeddings, raw; then after centering; then after rotation; finally after isometrically mapping them to H 2 as explained in section 3.

The isometry was obtained with the weakly-supervised method WordNet 400 + 400.

Legend: WordNet levels (root is 0).

Model: h = (·) 2 , full vocabulary of 190k words.

More of these plots for other D 2 spaces are shown in appendix A.3.In order to identify the two sets G and S, we propose the following two methods.• Unsupervised 5K+5K: a fully unsupervised method.

We first define a restricted vocabulary of the 50k most frequent words among the unrestricted one of 190k words, and rank them by frequency; we then define G as the 5k most frequent ones, and S as the 5k least frequent ones of the 50k vocabulary (to avoid extremely rare words which might have received less signal during training).• Weakly-supervised WN x+x: a weakly-supervised method that uses words from the WordNet hierarchy.

We define G as the top x words from the top 4 levels of the WordNet hierarchy, and S as x of the bottom words from the bottom 3 levels, randomly sampled in case of ties.

Gaussian embeddings.

BID36 propose using is-a(P, Q) := −KL(P ||Q) for distributions P, Q, the argument being that a low KL(P ||Q) indicates that we can encode Q easily as P , implying that Q entails P .

However, we would like to mitigate this statement.

Indeed, if P = N (µ, σ) and Q = N (µ, σ ) are two 1D Gaussian distributions with same mean, then KL(P ||Q) = z 2 − 1 − log(z) where z := σ/σ , which is not a monotonic function of z. This breaks the idea that the magnitude of the variance should encode the generality/specificity of the concept.

Another entailment score for Gaussian embeddings.

What would constitute a good number for the variance's magnitude of a n-dimensional Gaussian distribution N (µ, Σ)?

It is known that 95% of its mass is contained within a hyper-ellipsoid of volume V Σ = V n det(Σ), where V n denotes the volume of a ball of radius 1 in R n .

For simplicity, we propose dropping the dependence in µ and define a simple score is-a(Σ, DISPLAYFORM2 .

Note that using difference of logarithms has the benefit of removing the scaling constant V n , and makes the entailment score invariant to a rescaling of the covariance matrices: is-a(rΣ, rΣ ) = is-a(Σ, Σ ), ∀r > 0.To compute this is-a score between two hyperbolic word embeddings, we first map all word embeddings to Gaussians as explained above and, subsequently, apply the above proposed is-a score.

Algorithm 1 illustrates these steps.

Results are shown in section 9: Figure 4 and Tables 6, 7.Algorithm 1 is-a(v, w) hypernymy score using Poincaré embeddings DISPLAYFORM3 Output: is-a(v, w) lexical entailment score 4:G ← set of Poincaré embeddings of generic words

S ← set of Poincaré embeddings of specific words 6:for i from 1 to p do // Fixing the correct isometry.7: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 14: DISPLAYFORM3 // Convert half-plane coordinates to Gaussian parameters.17: DISPLAYFORM4 18: DISPLAYFORM5

Why would we embed words in a hyperbolic space?

Given some symbolic data, such as a vocabulary along with similarity measures between words − in our case, co-occurrence counts X ij − can we understand in a principled manner which geometry would represent it best?

Choosing the right metric space to embed words can be understood as selecting the right inductive bias − an essential step.

δ-hyperbolicity.

A particular quantity of interest describing qualitative aspects of metric spaces is the δ-hyperbolicity which we formally define in appendix B. This metric introduced by BID18 quantifies the tree-likeliness of a space.

However, for various reasons discussed in appendix B, we used the averaged δ-hyperbolicity, denoted δ avg , defined by BID0 .

Intuitively, a low δ avg of a finite metric space characterizes that this space has an underlying hyperbolic geometry, i.e. an approximate tree-like structure, and that the hyperbolic space would be well suited to isometrically embed it.

We also report the ratio 2 * δ avg /d avg (invariant to metric scaling), where d avg is the average distance in the finite space, as suggested by BID8 , whose low value also characterizes the "hyperbolicness" of the space.

Computing δ avg .

Since our methods are trained on a weighted graph of co-occurrences, it makes sense to look for the corresponding hyperbolicity δ avg of this symbolic data.

The lower this value, the more hyperbolic is the underlying nature of the graph, thus indicating that the hyperbolic space should be preferred over the Euclidean space for embedding words.

However, in order to do so, one needs to be provided with a distance d(i, j) for each pair of words (i, j), while our symbolic data is only made of similarity measures.

Note that one cannot simply associate the value − log(P ij ) to d(i, j), as this quantity is not symmetric.

Instead, inspired from Eq. (3), we associate to words i, j the distance 3 h(d(i, j)) := − log(X ij ) + b i + b j ≥ 0 with the choice b i := log(X i ), i.e. Table 1 shows values for different choices of h. The discrete metric spaces we obtained for our symbolic data of co-occurrences appear to have a very low hyperbolicity, i.e. to be very much "hyperbolic", which suggests to embed words in (products of) hyperbolic spaces.

We report in section 9 empirical results for h = (·) 2 and h = cosh 2 .

DISPLAYFORM0 DISPLAYFORM1

Experimental setup.

We trained all models on a corpus provided by BID22 ; BID23 used in other word embeddings related work.

Corpus preprocessing is explained in the above references.

The dataset has been obtained from an English Wikipedia dump and contains 1.4 billion tokens.

Words appearing less than one hundred times in the corpus have been discarded, leaving 189, 533 unique tokens.

The co-occurrence matrix contains approximately 700 millions non-zero entries, for a symmetric window size of 10.

All models were trained for 50 epochs, and unless stated otherwise, on the full corpus of 189,533 word types.

For certain experiments, we also trained the model on a restricted vocabulary of the 50, 000 most frequent words, which we specify by appending either "(190k)" or "(50k)" to the experiment's name in the table of results.

Poincaré models, Euclidean baselines.

We report results for both 100D embeddings trained in a 100D Poincaré ball, and for 50x2D embeddings, which were trained in the Cartesian product of 50 2D Poincaré balls.

Note that in the case of both models, one word will be represented by exactly 100 parameters.

For the Poincaré models we employ both h(x) = x 2 and h(x) = cosh 2 (x).

All hyperbolic models were optimized with RADAGRAD BID5 as explained in Sec. 5.

On the tasks of similarity and analogy, we compare against a 100D Euclidean GloVe model which was trained using the hyperparameters suggested in the original GloVe paper BID31 .

The vanilla GloVe model was optimized using ADAGRAD BID15 .

For the Euclidean baseline as well as for models with h(x) = x 2 we used a learning rate of 0.05.

For Poincaré models with h(x) = cosh 2 (x) we used a learning rate of 0.01.The initialization trick.

We obtained improvement in the majority of the metrics when initializing our Poincaré model with pretrained parameters.

These were obtained by first training the same model on the restricted (50k) vocabulary, and then using this model as an initialization for the full (190K) vocabulary.

This will be referred to as the "initialization trick".

For fairness, we also trained the vanilla (Euclidean) GloVe model in the same fashion.

Similarity.

Word similarity is assessed using a number of well established benchmarks shown in TAB2 .

We summarize here our main results, but more extensive experiments (including in lower dimensions) are in Appendix A.1.

We note that, with a single exception, our 100D and 50x2D models outperform the vanilla Glove baselines in all settings.

Analogy.

For word analogy, we evaluate on the Google benchmark BID24 ) and its two splits that contain semantic and syntactic analogy queries.

We also use a benchmark by MSR that is also commonly employed in other word embedding works.

For the Euclidean baselines we use 3COSADD BID23 .

For our models, the solution d to the problem "which d is to c, what b is to a" is selected as m t d1d2 , as described in section 6.

In order to select the best t without overfitting on the benchmark dataset, we used the same 2-fold cross-validation method used by BID23 , section 5.1) (see TAB15 ) − which resulted in selecting t = 0.3.

We report our main results in TAB4 , and more extensive experiments in various settings (including in lower dimensions) in appendix A.2.

We note that the vast majority of our models outperform the vanilla Glove baselines, with the 100D hyperbolic embeddings being the absolute best.

BID4 datasets.

We classify all the methods in three categories depending on the supervision used for word embedding learning and for the hypernymy score, respectively.

For Hyperlex we report results in Tab.

6 and use the baseline scores reported in BID30 Vulić et al., 2017) .

For WBLess we report results in Tab.

7 and use the baseline scores reported in (Nguyen et al., 2017).

Table 5 : Some words selected from the 100 nearest neighbors and ordered according to the hypernymy score function for a 50x2D hyperbolic embedding model using h(x) = x 2 .reptile amphibians, carnivore, crocodilian, fish-like, dinosaur, alligator, triceratops algebra mathematics, geometry, topology, relational, invertible, endomorphisms, quaternions music performance, composition, contemporary, rock, jazz, electroacoustic, trio feeling sense, perception, thoughts, impression, emotion, fear, shame, sorrow, joyHypernymy results discussion.

We first note that our fully unsupervised 50x2D, h(x) = x 2 model outperforms all its corresponding baselines setting a new state-of-the-art on unsupervised WBLESS accuracy and matching the previous state-of-the-art on unsupervised HyperLex Spearman correlation.

Second, once a small amount of weakly supervision is used for the hypernymy score, we obtain significant improvements as shown in the same tables and also in Fig. 4 .

We note that this weak supervision is only as a post-processing step (after word embeddings are trained) for identifying the best direction encoding the variance of the Gaussian distributions as described in Sec. 7.

Moreover, it does not consist of hypernymy pairs, but only of 400 or 800 generic and specific sets of words from WordNet.

Even so, our unsupervised learned embeddings are remarkably able to outperform all (except WN-Poincaré) supervised embedding learning baselines on HyperLex which have the great advantage of using the hypernymy pairs to train the word embeddings.

This plot describes how the Gaussian variances of our learned hyperbolic embeddings (trained unsupervised on co-occurrence statistics, isometry found with "Unsupervised 1k+1k") correlate with WordNet levels; (4b): This plot shows how the performance of our embeddings on hypernymy (HyperLex dataset) evolve when we increase the amount of supervision x used to find the correct isometry in the model WN x + x. As can be seen, a very small amount of supervision (e.g. 20 words from WordNet) can significantly boost performance compared to fully unsupervised methods.

Which model to choose?

While there is no single model that outperforms all the baselines on all presented tasks, one can remark that the model 50x2D, h(x) = x 2 , with the initialization trick obtains state-of-the-art results on hypernymy detection and is close to the best models for similarity and analogy (also Poincaré Glove models), but almost constantly outperforming the vanilla Glove baseline on these.

This is the first model that can achieve competitive results on all these three tasks, additionally offering interpretability via the connection to Gaussian word embeddings.

BID30 0.389 WN-Poincaré from BID30 0.512Unsupervised embedding learning & Weakly-supervised hypernymy score BID10 0.333 SBOW-PPMI-C∆S from BID10 0.345 BID30 0.86 BID28 0.87 DISPLAYFORM0 DISPLAYFORM1 Unsupervised embedding learning & Weakly-supervised hypernymy score DISPLAYFORM2 Unsupervised embedding learning & Unsupervised hypernymy score SGNS from BID28 ) 0.48 (Weeds et al., 2014 0.58 50x2D Poincaré GloVe, DISPLAYFORM3 0.652

We propose to adapt the GloVe algorithm to hyperbolic spaces and to leverage a connection between statistical manifolds of Gaussian distributions and hyperbolic geometry, in order to better interpret entailment relations between hyperbolic embeddings.

We justify the choice of products of hyperbolic spaces via this connection to Gaussian distributions and via computations of the hyperbolicity of the symbolic data upon which GloVe is based.

Empirically we present the first model that can simultaneously obtain state-of-the-art results or close on the three tasks of word similarity, analogy and hypernymy detection.

We show here extensive empirical results in various settings, including lower dimensions, different product structures, changing the vocabulary and using different h functions.

Experimental setup.

In the experiment's name, we first indicate which dimension was used: "nD" denotes D n while "p × kD" denotes (D k ) p .

"Vanilla" designates the baseline, i.e. the standard Euclidean GloVe from Eq. (1), while "Poincaré" designates our hyperbolic GloVe from Eq. (3).

For Poincaré models, we then append to the experiment's name which h function was applied to distances during training.

Every model was trained for 50 epochs.

Vanilla models were optimized with Adagrad BID15 while Poincaré models were optimized with RADAGRAD BID5 .

For each experiment we tried using learning rates in {0.01, 0.05}, and found that the best were 0.01 for h = cosh 2 and 0.05 for h = (·) 2 and for Vanilla models − accordingly, we only report the best results.

For similarity, we only considered the "target word vector" and always ignored the "context word vector".

We also tried using the Euclidean/Möbius average 5 of these, but obtained (almost) consistently worse results for all experiments (including baselines) and do not report them.

Reported scores are Spearman's correlations on the ranks for each benchmark dataset, as usual in the literature.

We used (minus) the Poincaré distance as a similarity measure to rank neighbors.

Table 8 : Unrestricted (190k) similarity results: models were trained and evaluated on the unrestricted (190k) vocabulary − "(init)" refers to the fact that the model was initialized with its counterpart (i.e. with same hyperparameters) on the restricted (50k) vocabulary, i.e. the initialization trick.

Remark.

Note that restricting the vocabulary incurs a loss of certain pairs of words from the benchmark similarity datasets, hence similarity results on the restricted (50k) vocabulary from TAB9 should be analyzed with caution, and in the light of TAB10 (especially for Rare Word).

A.2 ANALOGY Details and notations.

In the column "method", "3.c.a" denotes using 3COSADD to solve analogies, which was used for all Euclidean baselines; for Poincaré models, as explained in section 9, the solution to the analogy problem is computed as m t d1d2 with t = 0.3, and then the nearest neighbor in the vocabulary is selected either with the Poincaré distance on the corresponding space, which we denote as "d", or with cosine similarity on the full vector, which we denote as " cos".

Finally, note that each cell contains two numbers, designated by w and w +w respectively: w denotes ignoring the context vectors, while w +w denotes considering as our embeddings the Euclidean/Möbius average between the target vector w and the context vectorw.

In each dimension, we bold best results for w. About analogy computations.

Note that one can rewrite Eq. (6) with tools from differential geometry as DISPLAYFORM0 where P x→y = (λ x /λ y )gyr[y, x] denotes the parallel transport along the unique geodesic from x to y.

The exp and log maps of Riemannian geometry were related to the theory of gyrovector spaces BID34 by BID17 , who also mention that when continuously deforming the hyperbolic space D n into the Euclidean space R n , sending the curvature from −1 to 0 (i.e. the radius of D n from 1 to ∞), the Möbius operations ⊕ κ , κ , ⊗ κ , gyr κ recover their respective Euclidean counterparts +, −, ·, Id. Hence, the analogy solutions d 1 , d 2 , m t d1d2 of Eq. (6) would then all recover d = c + b − a, which seems a nice sanity check.

We show here more plots illustrating the method (described in section 7) that we use to map points from a (product of) Poincaré disk(s) to a (diagonal) Gaussian.

Colors indicate WordNet levels: low levels are closer to the root.

Figures 5, 6, 7, 8 show the three steps (centering, rotation, isometric mapping to half-plane) for 20D embeddings in (D 2 ) 10 , i.e. each of these steps in each of the 10 corresponding 2D spaces.

In these figures, centering and rotation were determined with our proposed semi-supervised method, i.e. selecting 400+400 top and bottom words from the WordNet hierarchy.

We show these plots for two models in (D 2 ) 10 : one trained with h = (·) 2 and one with h = cosh 2 .Remark.

It is easily noticeable that words trained with h = cosh 2 are embedded much closer to each other than those trained with h = (·)2 .

This is expected: h is applied to the distance function, and according to Eq. DISPLAYFORM0 Published as a conference paper at ICLR 2019

Let us start by defining the δ-hyperbolicity, introduced by BID18 .

The hyperbolicity δ(x, y, z, t) of a 4-tuple (x, y, z, t) is defined as half the difference between the biggest two of the following sums: d(x, y) + d(z, t), d(x, z) + d(y, t), d(x, t) + d(y, z).

The δ-hyperbolicity of a metric space is defined as the supremum of these numbers over all 4-tuples.

Following BID0 , we will denote this number by δ worst , and by δ avg the average of these over all 4-tuples, when the space is a finite set.

An equivalent and more intuitive definition holds for geodesic spaces, i.e. when we can define triangles: its δ-hyperbolicity (δ worst ) is the smallest δ > 0 such that for any triangle ∆xyz, there exists a point at distance at most δ from each side of the triangle.

BID11 and BID8 analyzed δ worst and δ avg for specific graphs, respectively.

A low hyperbolicity of a graph indicates that it has an underlying hyperbolic geometry, i.e. that it is approximately tree-like, or at least that there exists a taxonomy of nodes.

Conversely, a high hyperbolicity of a graph suggests that it possesses long cycles, or could not be embedded in a low dimensional hyperbolic space without distortion.

For instance, the Euclidean space R n is not δ-hyperbolic for any δ > 0, and is hence described as ∞-hyperbolic, while the Poincaré disk D 2 is known to have a δ-hyperbolicity of log(1 + √ 2) 0.88.

On the other-hand, a product D 2 × D 2 is ∞-hyperbolic, because a 2D plane R 2 could be isometrically embedded in it using for instance the first coordinates of each D 2 .

However, if D 2 would constitute a good choice to embed some given symbolic data, then most likely D 2 × D 2 would as well.

This stems from the fact that δ-hyperbolicity (δ worst ) is a worst case measure which does not reflect what one could call the "hyperbolic capacity" of the space.

Furthermore, note that computing δ worst requires O(n 4 ) for a graph of size n, while δ avg can be approximated via sampling.

Finally, δ avg is robust to adding/removing a node from the graph, while δ worst is not.

For all these reasons, we choose δ avg as a measure of hyperbolicity.

More experiments.

As explained in section 8, we computed hyperbolicities of the metric space induced by different h functions, on the matrix of co-occurrence counts, as reported in Table 1 .

We also conducted similarity experiments, reported in TAB7 .

Apart from WordSim, results improved for higher powers of cosh, corresponding to more hyperbolic spaces.

However, also note that higher powers will tend to result in words embedded much closer to each other, i.e. with smaller distances, as explained in appendix A.3.

In order to know whether this benefit comes from contracting distances or making the space more "hyperbolic", it would be interesting to learn (or cross-validate) the curvature c of the Poincaré ball (or equivalently, its radius) jointly with the h function.

Finally, it order to explain why WordSim behaved differently compared to other benchmarks, we investigated different properties of these, as reported in TAB5 .

The geometry of the words appearing in WordSim do not seem to have a different hyperbolicity compared to other benchmarks; however, WordSim seems to contain much more frequent words.

Since hyperbolicities are computed with the assumption that b i = log(X i ) (see Eq. FORMULA23 ), it would be interesting to explore whether learned biases indeed take these values.

We left this as future work.

@highlight

We embed words in the hyperbolic space and make the connection  with the Gaussian word embeddings.

@highlight

This paper adapts the Glove word embedding to a hyperbolic space given by the Poincare half-plane model

@highlight

This paper proposes an approach to implement a GLOVE-based hyperbolic word embedding model, which is optimized via the Riemannian Optimization methods.