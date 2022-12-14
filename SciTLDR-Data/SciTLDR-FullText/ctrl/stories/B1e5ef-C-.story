Low-dimensional vector embeddings, computed using LSTMs or simpler techniques, are a popular approach for capturing the “meaning” of text and a form of unsupervised learning useful for downstream tasks.

However, their power is not theoretically understood.

The current paper derives formal understanding by looking at the subcase of linear embedding schemes.

Using the theory of compressed sensing we show that representations combining the constituent word vectors are essentially information-preserving linear measurements of Bag-of-n-Grams (BonG) representations of text.

This leads to a new theoretical result about LSTMs: low-dimensional embeddings derived from a low-memory LSTM are provably at least as powerful on classification tasks, up to small error, as a linear classifier over BonG vectors, a result that extensive empirical work has thus far been unable to show.

Our experiments support these theoretical findings and establish strong, simple, and unsupervised baselines on standard benchmarks that in some cases are state of the art among word-level methods.

We also show a surprising new property of embeddings such as GloVe and word2vec: they form a good sensing matrix for text that is more efficient than random matrices, the standard sparse recovery tool, which may explain why they lead to better representations in practice.

Much attention has been paid to using LSTMs BID15 and similar models to compute text embeddings BID3 BID7 .

Once trained, the LSTM can sweep once or twice through a given piece of text, process it using only limited memory, and output a vector with moderate dimensionality (a few hundred to a few thousand), which can be used to measure text similarity via cosine similarity or as a featurization for downstream tasks.

The powers and limitations of this method have not been formally established.

For example, can such neural embeddings compete with and replace traditional linear classifiers trained on trivial Bag-of-n-Grams (BonG) representations?

Tweaked versions of BonG classifiers are known to be a surprisingly powerful baseline (Wang & Manning, 2012) and have fast implementations BID17 .

They continue to give better performance on many downstream supervised tasks such as IMDB sentiment classification BID21 than purely unsupervised LSTM representations BID19 BID13 BID25 .

Even a very successful character-level (and thus computation-intensive, taking a month of training) approach does not reach BonG performance on datasets larger than IMDB BID31 .

Meanwhile there is evidence suggesting that simpler linear schemes give compact representations that provide most of the benefits of word-level LSTM embeddings (Wieting et al., 2016; BID1 .

These linear schemes consist of simply adding up, with a few modifications, standard pretrained word embeddings such as GloVe or word2vec BID24 BID29 .The current paper ties these disparate threads together by giving an information-theoretic account of linear text embeddings.

We describe linear schemes that preserve n-gram information as lowdimensional embeddings with provable guarantees for any text classification task.

The previous linear schemes, which used unigram information, are subcases of our approach, but our best schemes can also capture n-gram information with low additional overhead.

Furthermore, we show that the original unigram information can be (approximately) extracted from the low-dimensional embedding using sparse recovery/compressed sensing BID6 .

Our approach also fits in the tradition of the older work on distributed representations of structured objects, especially the works of BID30 and BID18 .

The following are the main results achieved by this new world-view:1.

Using random vectors as word embeddings in our linear scheme (instead of pretrained vectors) already allows us to rigorously show that low-memory LSTMs are provably at least as good as every linear classifier operating on the full BonG vector.

This is a novel theoretical result in deep learning, obtained relatively easily.

By contrast, extensive empirical study of this issue has been inconclusive (apart from character-level models, and even then only on smaller datasets BID31 ).

Note also that empirical work by its nature can only establish performance on some available datasets, not on all possible classification tasks.

We prove this theorem in Section 4 by providing a nontrivial generalization of a result combining compressed sensing and learning BID5 ).

In fact, before our work we do not know of any provable quantification of the power of any text embedding.2.

We study theoretically and experimentally how our linear embedding scheme improves when it uses pretrained embeddings (GloVe etc.) instead of random vectors.

Empirically we find that this improves the ability to preserve Bag-of-Words (BoW) information, which has the following restatement in the language of sparse recovery: word embeddings are better than random matrices for "sensing" BoW signals (see Section 5).

We give some theoretical justification for this surprising finding using a new sparse recovery property characterizing when nonnegative signals can be reconstructed by 1 -minimization.3.

Section 6 provides empirical results supporting the above theoretical work, reporting accuracy of our linear schemes on multiple standard classification tasks.

Our embeddings are consistently competitive with recent results and perform much better than all previous linear methods.

Among unsupervised word-level representations they achieve state of the art performance on both the binary and fine-grained SST sentiment classification tasks BID33 .

Since our document representations are fast, compositional, and simple to implement given standard word embeddings, they provide strong baselines for future work.

Neural text embeddings are instances of distributed representations, long studied in connectionist approaches because they decay gracefully with noise and allow distributed processing.

BID14 provided an early problem formulation, and BID30 provided an elementary solution, the holographic distributed representation, which represents structured objects using circular vector convolution and has an easy and more compact implementation using the fast Fourier transform (FFT).

Plate suggested applying such ideas to text, where "structure" can be quantified using parse trees and other graph structures.

Our method is also closely related in form and composition to the sparse distributed memory system of BID18 .

In the unigram case our embedding reduces to the familiar sum of word embeddings, which is known to be surprisingly powerful (Wieting et al., 2016) , and with a few modifications even more so BID1 .Representations of BonG vectors have been studied through the lens of compression by BID28 , who computed representations based on classical lossless compression algorithms using a linear program (LP).

Their embeddings are still high-dimensional (d > 100K) and quite complicated to implement.

In contrast, linear projection schemes are simpler, more compact, and can leverage readily available word embeddings.

BID25 also used a linear scheme, representing documents as an average of learned word and bigram embeddings.

However, the motivation and benefits of encoding BonGs in low-dimensions are not made explicit.

The novelty in the current paper is the connection to compressed sensing, which is concerned with recovering high-dimensional sparse signals x ∈ R N from low-dimensional linear measurements Ax, specifically by studying conditions on matrix A ∈ R d×N when this is possible (see Appendix A for some background on compressed sensing and the previous work of BID5 that we build upon).

In this section we define the two types of representations that our analysis will relate:1.

high-dimensional BonG vectors counting the occurrences of each k-gram for k ≤ n 2.

low-dimensional embeddings, from simple vector sums to novel n-gram-based embeddings Although some of these representations have been previously studied and used, we define them so as to make clear their connection via compressed sensing, i.e. that representations of the second type are simply linear measurements of the first.

We now define some notation.

Let V be the number of words in the vocabulary and V n be the number of n-grams (independent of word order), so that V = V 1 .

Furthermore set V sum n = k≤n V k and V max n = max k≤n V k .

We will use words/n-grams and indices interchangeably, e.g. if (a, b) is the ith of V 2 bigrams then the one-hot vector e (a,b) will be 1 at index i.

Where necessary we will use {, } to denote a multi-set and (, ) to denote a tuple.

For any m vectors v i ∈ R d for i = 1, . . .

, m we define [v 1 , . . .

, v m ] to be their concatenation, which is thus an element of R md .

Finally, for any subset X ⊂ R N we denote by ∆X the set {x − x : x, x ∈ X }.

Assigning to each word a unique index i ∈ [V ] we define the Bag-of-Words (BoW) representation x BoW of a document to be the V -dimensional vector whose ith entry is the number of times word i occurs in the document.

The n-gram extension of BoW is the Bag-of-n-Grams (BonG) representation, which counts the number of times any k-gram for k ≤ n appears in a document.

Linear classification over such vectors has been found to be a strong baseline (Wang & Manning, 2012) .For ease of analysis we simplify the BonG approach by merging all n-grams in the vocabulary that contain the same words but in a different order.

We call these features n-cooccurrences and find that the modification does not affect performance significantly (see Table 3 in Appendix F.1).

Formally for a document w 1 , . . .

, w T we define the Bag-of-n-Cooccurrences (BonC) vector as the concatenation DISPLAYFORM0 e wt , . . .

, DISPLAYFORM1 which is thus a V sum n -dimensional vector.

Note that for unigrams this is equivalent to the BoW vector.

Now suppose each word w has a vector v w ∈ R d for some d V .

Then given a document w 1 , . . .

, w T we define its unigram embedding as z u = T t=1 v wt .

While this is a simple and widely used featurization, we focus on the following straightforward relation with BoW: if A ∈ R d×V is a matrix whose columns are word vectors v w then Ax BoW = T t=1 Ae wt = T t=1 v wt = z u .

Thus in terms of compressed sensing the unigram embedding of a document is a d-dimensional linear measurement of its Bag-of-Words vector.

We could extend this unigram embedding to n-grams by first defining a representation for each ngram as the tensor product of the vectors of its constituent words.

Thus for each bigram b = (w 1 , w 2 ) we would have v b = v w1 v T w2 and more generally v g = n t=1 v wt for each n-gram g = (w 1 , . . . , w n ).

The document embedding would then be the sum of the tensor representations of all n-grams.

The major drawback of this approach is of course the blowup in dimension, which in practice prevents its use beyond n = 2.

To combat this a low-dimensional sketch or projection of the tensor product can be used, such as the circular convolution operator of BID30 .

Since we are interested in representations that can also be constructed by an LSTM, we instead sketch this tensor product using the element-wise multiplication operation, which we find also usually works better than circular convolution in practice (see Table 4 in Appendix F.1).

Thus for the n-cooccurrence g = {w 1 , . . . , w n }, we define the distributed cooccurrence (DisC) embeddingṽ g = d n−1 2 n t=1 v wt .

The coefficient is required when the vectors v w are random and unit norm to ensure that the product also has close to unit norm (see Lemma B.1).

In addition to their convenient form, DisC embeddings have nice theoretical and practical properties: they preserve the original embedding dimension, they reduce to unigram (word) embeddings for n = 1, and under mild assumptions they satisfy useful compressed sensing properties with overwhelming probability (Lemma 4.1).We then define the DisC document embedding to be the nd-dimensional weighted concatenation, over k ≤ n, of the sum of the DisC vectors of all k-grams in a document: DISPLAYFORM0 Here scaling factors C k are set so that all spans of d coordinates have roughly equal norm (for random embeddings C k = 1; for word embeddings C k = 1/k works well).

Note that sinceṽ wt = v wt we have z (1) = z u in the unigram case.

Furthermore, as with unigram embeddings by comparing (1) and (2) one can easily construct a DISPLAYFORM1

As discussed previously, LSTMs have become a common way to apply the expressive power of RNNs, with success on a variety of classification, representation, and sequence-to-sequence tasks.

For document representation, starting with h 0 = 0 m an m-memory LSTM initialized with word vectors v w ∈ R d takes in words w 1 , . . .

, w T one-by-one and computes the document representation DISPLAYFORM0 where h t ∈ R m is the hidden representation at time t, the forget gate f , input gate i, and input function g are a.e.

differentiable nondecreasing elementwise "activation" functions R m → R m , and affine transformations T * (x, y) = W * x + U * y + b * have weight matrices W * ∈ R m×d , U * ∈ R m×m and bias vectors b * ∈ R m .

The LSTM representation of a document is then the state at the last time step, i.e. z LSTM = h T .

Note that we will follow the convention of using LSTM memory to refer to the dimensionality of the hidden states.

Since the LSTM is initialized with an embedding for each word it requires O(m 2 + md + V d) computer memory, but the last term is just a lookup table so the vocabulary size does not factor into iteration or representation complexity.

From our description of LSTMs it is intuitive to see that one can initialize the gates and input functions so as to construct the DisC embeddings defined in the previous section.

We state this formally and give the proof in the unigram case (the full proof appears in Appendix B.3): Proposition 3.1.

Given word vectors v w ∈ R d , one can initialize an O(nd)-memory LSTM (3) that takes in words w 1 , . . .

, w T (padded by an end-of-document token assigned vector 0 d ) and constructs the DisC embedding (2) (up to zero padding), i.e. such that for all documents z LSTM = z (n) .

DISPLAYFORM1 By Proposition 3.1 we can construct a fixed LSTM that can compute compressed BonC representations on the fly and be further trained by stochastic gradient descent using the same memory.

Our main contribution is to provide the first rigorous analysis of the performance of the text embeddings that we are aware of, showing that the embeddings of Section 3.2 can provide performance on downstream classification tasks at least as well any linear classifier over BonCs.

Before stating the theorem we make two mild simplifying assumptions on the BonC vectors:1.

The vectors are scaled by DISPLAYFORM0 , where T is the maximum document length.

This assumption is made without loss of generality.2.

No n-cooccurrence contains a word more than once.

While this is (infrequently) violated in practice, the problem can be circumvented by merging words as a preprocessing step.

DISPLAYFORM1 (1 − γ)(1 − 2δ) the classifierŵ minimizing the 2 -regularized logistic loss over its representations satisfies DISPLAYFORM2 The above theoretical bound shows that LSTMs match BonC performance as ε → 0, which can be realized by increasing the embedding dimension d (c.f.

FIG7 ).

Compressed sensing is concerned with recovering a high-dimensional k-sparse signal x ∈ R N from a few linear measurements; given a design matrix A ∈ R d×N this is formulated as DISPLAYFORM0 where z = Ax is the measurement vector.

As l 0 -minimization is NP-hard, research has focused on sufficient conditions for tractable recovery.

One such condition is the Restricted Isometry Property (RIP), for which BID6 proved that (5) can be solved by convex relaxation: DISPLAYFORM1 We will abuse notation and say (k, ε)-RIP when X is the set of k-sparse vectors.

This is the more common definition, but ours allows a more general Theorem 4.2 and a tighter bound in Theorem 4.1.

DISPLAYFORM2 If is a λ-Lipschitz convex loss function and w 0 ∈ R N is its minimizer over D then w.p.

1 − 2δ the linear classifierŵ A ∈ R d minimizing the 2 -regularized empirical loss function DISPLAYFORM3 for appropriate choice of C. Recall that ∆X = {x − x : x, x ∈ X } for any X ⊂ R N .While a detailed proof of this theorem is spelled out in Appendix C, the main idea is to compare the distributional loss incurred by a classifierŵ in the original space to the loss incurred by Aŵ in the compressed space.

We show that the minimizer of the regularized empirical loss in the original space (ŵ) is a bounded-coefficient linear combination of samples in S, so its loss depends only on inner products between points in X .

Thus using RIP and a generalization error result by BID34 we can bound the loss ofŵ A , the regularized classifier in the compressed domain.

Note that to get back from Theorem 4.2 the O( √ ε) bound for k-sparse inputs of BID5 we can set X to the be the set of k-sparse vectors and assume A is (2k, ε)-RIP.

To apply Theorem 4.2 we need the design matrix A (n) transforming BonCs into the DisC embeddings of Section 3.2 to satisfy the following RIP condition (Lemma 4.1), which we prove using a restricted isometry result for structured random sampling matrices in Appendix D: Lemma 4.1.

Assume the setting of Theorem 4.1 and let A (n) be the nd × V sum n matrix relating DisC and BonC representations of any document by DISPLAYFORM0 is the set of BonCs of documents of length at most T .Proof of Theorem 4.

DISPLAYFORM1 is the set of BonC vectors of documents of length at most T .

By BonC assumption (1) all BonCs lie within the unit ball, so we can apply Theorem 4.2 with the logistic loss, λ = 1, and R = 1 to get that a classifierŵ trained using 2 -regularized logistic loss overŜ will satisfy the required bound (4).

Since by Proposition 3.1 one can DISPLAYFORM2 T , this completes the proof.

Theorem 4.1 is proved using random vectors as the word embeddings in the scheme of Section 3.

However, in practice LSTMs are often initialized with standard word vectors such as GloVe.

Such embeddings cannot satisfy traditional compressed sensing properties such as RIP or incoherence.

This follows essentially from the definition: word embeddings seek to capture word similarity, so similar words (e.g. synonyms) have embeddings with high inner product, which violates both properties.

Thus the efficacy of real-life LSTMs must have some other explanation.

But in this section we present the surprising empirical finding that pretrained word embeddings are more efficient than random vectors at encoding and recovering BoW information via compressed sensing.

We further sketch a potential explanation for this result, though a rigorous explanation is left for subsequent work.

In recent years word embeddings have been discovered to have many remarkable properties, most famously the ability to solve analogies BID24 .

Our connection to compressed sensing indicates that they should have another: preservation of sparse signals as low-dimensional linear measurements.

To examine this we subsample documents from the SST BID33 and IMDB BID21 classification datasets, embed them as d-dimensional unigram embeddings z = Ax for d = 50, 100, 200, . . . , 1600 (where A ∈ R d×V is the matrix of word embeddings and x is a document's BoW vector), solve the following LP, known as Basis Pursuit (BP), which is the standard 1 -minimization problem for sparse recovery in the noiseless case (see Appendix A): DISPLAYFORM0 Success is measured as the F 1 score of retrieved words.

We use Squared Norm (SN) vectors BID0 trained on a corpus of Amazon reviews BID23 and normalized i.i.d.

Rademacher vectors as a baseline.

SN is used due to similarity to GloVe and its formulation via an easy-to-analyze generative model that may provide a framework to understand the results (see Appendix F.2), while the Amazon corpus is used for its semantic closeness to the sentiment datasets.

compared to dimension.

Pretrained word embeddings (SN trained on Amazon reviews) need half the dimensionality of normalized Rademacher vectors to achieve near-perfect recovery.

Note that IMDB documents are on average more than ten times longer than SST documents.

Figures 1 and 2 show that pretrained embeddings require a lower dimension d than random vectors to recover natural language BoW. This is surprising as the training objective goes against standard conditions such as approximate isometry and incoherence; indeed as shown in FIG3 recovery is poor for randomly generated word collections.

The latter outcome indicates that the fact that a document is a set of mutually meaningful words is important for sparse recovery using embeddings trained on co-occurrences.

We achieve similar results with other objectives (e.g. GloVe/word2vec) and other corpora (see Appendix F.1), although there is some sensitivity to the sparse recovery method, as other 1 -minimization methods work well but greedy methods, such as Orthogonal Matching Pursuit (OMP), work poorly, likely due to their dependence on incoherence BID36 .For the n-gram case (i.e. BonC recovery for n > 1), although we know by Lemma 4.1 that DisC embeddings composed from random vectors satisfy RIP, for pretrained vectors it is unclear how to reason about suitable n-gram embeddings without a rigorous understanding of the unigram case, and experiments do not show the same recovery benefits.

One could perhaps do well by training on cooccurrences of word tuples, but such embeddings could not be used by a word-level LSTM.

As shown in FIG3 , the success of pretrained embeddings for linear sensing is a local phenomenon; recovery is only efficient for naturally occurring collections of words.

However, applying statistical RIP/incoherence ideas BID2 to explain this is ruled out since they require collections to be incoherent with high probability, whereas word embeddings are trained to give high inner product to words appearing together.

Thus an explanation must come from some other, weaker condition.

The usual necessary and sufficient requirement for recovering all signals with support S ⊂ [N ] is the local nullspace property (NSP), which stipulates that vectors in the kernel of A not have too much mass on S (see Definition A.2).

While NSP and related properties such as restricted eigenvalue (see Definition A.3) are hard to check, we can impose some additional structure to formulate an intuitive, verifiable perfect recovery condition for our setting.

Specifically, since our signals (BoW vectors) are nonnegative, we can improve upon solving BP (8) by instead solving nonnegative BP (BP+): minimize w 1 subject to Aw = z, w ≥ 0 d (9) The following geometric result then characterizes when solutions of BP+ recover the correct signal: Theorem 5.1 BID9 .

Consider a matrix A ∈ R d×N and an index subset S ⊂ [N ] of size k. Then any nonnegative vector x ∈ R N + with support supp(x) = S is recovered from Ax by BP+ iff the set A S of columns of A indexed by S comprise the vertices of a k-dimensional face of the convex hull conv(A) of the columns of A together with the origin.

This theorem equates perfect recovery of a BoW vector via BP+ with the vectors of its words being the vertices of some face of the polytope conv(A).

The property holds for incoherent columns since the vectors are far enough that no one vector is inside the simplex formed by any k others.

On the other hand, pretrained embeddings satisfy it by having commonly co-occurring words close together and other words far away, making it easier to form a face from columns indexed by the support of a BoW. We formalize this intuition as the Supporting Hyperplane Property (SHP): SHP is a very weak property implied by NSP (Corollary E.1).

However, it can be checked by using convex optimization to see if the hyperplane exists (Appendix E.2).

Furthermore, we show (full proof in Appendix E.1) that this hyperplane is the supporting hyperplane of the face of conv(A) with vertices A S , from which it follows by Theorem 5.1 that SHP characterizes recovery using BP+: Corollary 5.1.

BP+ recovers any x ∈ R N + with supp(x) = S from Ax iff A satisfies S-SHP.Proof Sketch.

By Theorem 5.1 it suffices to show equivalence of S-SHP with the column set A S comprising the vertices of a k-dimensional face of conv(A).

A face F of polytope P is defined as its intersection with some hyperplane such that all points in P \F lie on one side of the hyperplane. ( =⇒ ) Let F be the face of conv(A) formed by the columns A S .

Then there must be a supporting hyperplane H containing F .

Since the columns of A are in general position, all columns A S = A\A S lie in conv(A)\F and hence must all be on one side of H, so H is the desired hyperplane. ( ⇐= ) Let H be the hyperplane supporting A S , with all other columns on one side of H. By convexity, H contains the simplex F of A S .

Any point in conv(A)\F can be written as a convex combination of points in F and columns A S , with a positive coefficient on at least one of the columns, and so must lie on the same side of H as A S .

Thus A S comprises the vertices of a face F .Thus perfect recovery of a BoW via BP+ is equivalent to the existence of a hyperplane separating embeddings of words in the document from those of the rest of the vocabulary.

Intuitively, words in the same document are trained to have similar embeddings and so will be easier to separate out, providing some justification for why pretrained vectors are better for sensing.

We verify that SHP is indeed more likely to be satisfied by such designs in FIG4 , which also serves as an empirical check of Corollary 5.1 since SHP satisfaction implies BP recovery as the latter can do no better than BP+.

We further compare to recovery using OMP/OMP+ (the latter removes negative values and recomputes the set of atoms at each iteration); interestingly, while OMP+ recovers the correct signal from SN almost as often as BP/BP+, it performs quite poorly for GloVe, indicating that these embeddings may have quite different sensing properties despite similar training objectives.

As similarity properties that may explain these results also relate to downstream task performance, we conjecture a relationship between embeddings, recovery, and classification that may be understood under a generative model (see Appendix F.2).

However, the Section 4 bounds depend on RIP, not recovery, so these experiments by themselves do not apply.

They do show that the compressed sensing framework remains relevant even in the case of non-random, pretrained word embeddings.

BID1 Reported performance of best hyperparameter using Amazon GloVe embeddings.

BID31 shown for comparison.

The top three results for each dataset are bolded, the best is italicized, and the best word-level performance is underlined.

Our theoretical results show that simple tensor product sketch-based n-gram embeddings can approach BonG performance and be computed by a low-memory LSTM.

In this section we compare these text representations and others on several standard tasks, verifying that DisC performance approaches that of BonCs as dimensionality increases and establishing several baselines for text classification.

Code to reproduce results is provided at https://github.com/NLPrinceton/text_embedding.

We test classification on MR movie reviews BID27 , CR customer reviews BID16 , SUBJ subjectivity dataset BID26 , MPQA opinion polarity subtask (Wiebe et al., 2005) , TREC question classification BID20 , SST sentiment classification (binary and fine-grained) BID33 , and IMDB movie reviews BID21 .

The first four are evaluated using 10-fold cross-validation, while the others have train-test splits.

In all cases we use logistic regression with 2 -regularization determined by cross-validation.

We further test DisC on the SICK relatedness and entailment tasks BID22 and the MRPC paraphrase detection task BID8 .

The inputs here are sentences pairs (a, b) and the standard featurization for document embeddings x a and x b of a and b is et al., 2015) .

We use logistic regression for SICK entailment and MRPC and use ridge regression to predict similarity scores for SICK relatedness, with 2 -regularization determined by cross-validation.

Since BonGs are not used for pairwise tasks our theory says nothing about performance here; we include these evaluations to show that our representations are also useful for other tasks.

DISPLAYFORM0 Embeddings: In the main evaluation TAB1 we use normalized 1600-dimensional GloVe embeddings BID29 ) trained on the Amazon Product Corpus BID23 , which are released at http://nlp.cs.princeton.edu/DisC. We also compare the SN vectors of Section 5 trained on the same corpus with random vectors when varying the dimension FIG7 ).

Results: We find that DisC representation performs consistently well relative to recent unsupervised methods; among word-level approaches it is the top performer on the SST tasks and competes on many others with skip-thoughts and CNN-LSTM, both concatenations of two LSTM representations.

While success may be explained by training on a large and in-domain corpus, being able to use so much text without extravagant computing resources is one of the advantages of a simple approach.

Overall our method is useful as a strong baseline, often beating BonCs and many more complicated approaches while taking much less time to represent and train on documents than neural representations FIG6 ).Finally, we analyze empirically how well our model approximates BonC performance.

As predicted by Theorem 4.1, the performance of random embeddings on IMDB approaches that of BonC as dimension increases and the isometry distortion ε decreases ( FIG7 ).

Using pretrained (SN) vectors, DisC embeddings approach BonC performance much earlier, surpassing it in the unigram case.

In this paper we explored the connection between compressed sensing, learning, and natural language representation.

We first related LSTM and BonG methods via word embeddings, coming up with simple new document embeddings based on tensor product sketches.

Then we studied their classification performance, proving a generalization of the compressed learning result of BID5 to convex Lipschitz losses and a bound on the loss of a low-dimensional LSTM classifier in terms of its (modified) BonG counterpart, an issue which neither experiments nor theory have been able to resolve.

Finally, we showed how pretrained embeddings fit into this sparse recovery framework, demonstrating and explaining their ability to efficiently preserve natural language information.

A COMPRESSED SENSING BACKGROUNDThe field of compressed sensing is concerned with recovering a high-dimensional k-sparse signal x ∈ R N from few linear measurements.

In the noiseless case this is formulated as minimize w 0 subject to Aw = zwhere A ∈ R d×N is the design matrix and z = Ax is the measurement vector.

Since 0 -minimization is NP-hard, a foundational approach is to use its convex surrogate, the 1 -norm, and characterize when the solution to (10) is equivalent to that of the following LP, known as basis pursuit (BP): DISPLAYFORM0 Related approaches such as Basis Pursuit Denoising (LASSO) and the Dantzig Selector generalize BP to handle signal or measurement noise BID11 ; however, the word embeddings case is noiseless so these methods reduce to BP.

Note that throughout Section 5 and the Appendix we say that an 1 -minimization method recovers x from Ax if its optimal solution is unique and equivalent to the optimal solution of (10).An alternative way to approximately solve FORMULA1 is to use a greedy algorithm such as matching pursuit (MP) or orthogonal matching pursuit (OMP), which pick basis vectors one at a time by multiplying the measurement vector by A T and choosing the column with the largest inner product BID36 .

One condition through which recovery can be guaranteed is the Restricted Isometry Property (RIP): DISPLAYFORM0 A line of work started by BID6 used the RIP property to characterize matrices A such that FORMULA1 and FORMULA1 have the same minimizer for any k-sparse signal x; this occurs with overwhelming probability when d = Ω k log N k and DISPLAYFORM1 Since the ability to recover a signal x from a representation Ax implies information preservation, a natural next step is to consider learning after compression.

BID5 DISPLAYFORM2 and a (2k, ε)-RIP matrix A, the hinge loss of a classifier trained on {( DISPLAYFORM3 is bounded by that of the best linear classifier over the original samples.

Theorem 4.2 provides a generalization of this result to any convex Lipschitz loss function.

RIP is a strong requirement, both because it is not necessary for perfect, stable recovery of k-sparse vectors usingÕ(k) measurements and because in certain settings we are interested in using the above ideas to recover specific signals -those statistically likely to occur-rather than all k-sparse signals.

The usual necessary and sufficient condition to recover any vector x ∈ R N with index support set S ⊂ [N ] is the local nullspace property (NSP), which is implied by RIP: Definition A.2 BID11 .

A matrix A ∈ R d×N satisfies NSP for a set S ⊂ [N ] if w S 1 < w S 1 for all nonzero w ∈ ker(A) = {v : Av = 0 d }.Theorem A.1 BID11 .

BP (11) recovers any x ∈ R N + with supp(x) = S from Ax iff A satisfies NSP for S.A related condition that implies NSP is the local restricted eigenvalue property (REP): Definition A.3 BID32 DISPLAYFORM4 Lastly, a simple condition that can sometimes provide recovery guarantees is mutual incoherence: Definition A.4.

A ∈ R d×N is µ-incoherent if max a,a |a T a | ≤ µ, where the maximum is taken over any two distinct columns a, a of A.While incoherence is easy to verify (unlike the previous recovery properties), word embeddings tend to have high coherence due to the training objective pushing together vectors of co-occurring words.

Apart from incoherence, the properties above are hard show empirically.

However, we are compressing BoW vectors, so our signals are nonnegative and we can impose an additional constraint on FORMULA1 The polytope condition is equivalent to nonnegative NSP (NSP+), a weaker form of NSP: Definition A.5 BID10 .

DISPLAYFORM0 w i > 0 for all nonzero w ∈ ker(A).Lemma A.1.

If A ∈ R d×N satisfies NSP for some S ⊂ [N ] then it also satisfies NSP+ for S.Proof (Adapted from BID10 ).

Since A satisfies NSP, we have w S 1 < w S 1 .

Then for a nonzero w ∈ ker(A) such that w S ≥ 0 we will have DISPLAYFORM1 Lemma A.2.

BP+ recovers any x ∈ R N + with supp(x) = S from Ax iff A satisfies NSP+ for S.

For any nonzero w ∈ ker(A) such that w S ≥ 0, ∃ λ > 0 such that x + λw ≥ 0 N and A(x + λw) = Ax.

Since BP+ uniquely recovers x, we have x + λw 1 > x 1 , so NSP+ follows from the following inequality and the fact that λ is positive: DISPLAYFORM0 For any x ≥ 0 such that Ax = Ax we have that w = x − x ∈ ker(A) and w S = x S ≥ 0 since the support of x is S. Thus by NSP+ we have that DISPLAYFORM1 w i > 0, which yields DISPLAYFORM2 Thus BP+ will recover x uniquely.

Lemma A.2 shows that NSP+ is equivalent to the polytope condition in Theorem A.2, as they are both necessary and sufficient conditions for BP+ recovery.

Table 3 : The performance of an l 2 -regularized logit classifier over Bag-of-n-Grams (BonG) vectors is generally similar to that of Bag-of-n-Cooccurrences (BonC) vectors for n = 2, 3 (largest differences bolded).

Evaluation settings are the same as in Section 6.

Note that for unigrams the two representations are equivalent.

Table 4 : Performance comparison of element-wise product (DisC) and circular convolution for encoding local cooccurrences (best result for each task is bolded).

Evaluation settings are the same as in Section 6.

Note that for unigrams the two representations are equivalent.

In this section we compare the performance of several alternative representations with the ones presented in the main evaluation TAB1 .

Table 3 provides a numerical justification for our use of unordered n-grams (cooccurrences) instead of n-grams, as the performance of the two featurizations are closely comparable.

In Table 4 we examine the use of circular convolution instead of elementwise multiplication as linear measurements of BonC vectors BID30 .

To construct the former from a document w 1 , . . . , w T we compute DISPLAYFORM0 where F is the discrete Fourier transform and F −1 its inverse.

Note that for n = 1 this is equivalent to the simple unigram embedding (and thus also to the DisC embedding in (2)).

d} then for any n-gram g = (w 1 , . . . , w n ) we have E ṽ g 2 2 = 1.

The same result holds true with the additional assumption that all words in g are distinct if the word vectors are i.i.d.

d-dimensional spherical Gaussians.

Proof.

DISPLAYFORM0 Substituting these parameters into the LSTM update (3) and using h 0 = 0 we have ∀ t > 0 that DISPLAYFORM1 . . .

DISPLAYFORM2 , by padding the end of the document with an end-of-document token whose word vector is 0 d the entries in those dimensions will be set to zero by the update at the last step.

Thus up to zero padding we will have z LSTM = h T =z (n) .C PROOF OF THEOREM 4.2Throughout this section we assume the setting described in Theorem 4.2.

Furthermore for some positive constant C define the 2 -regularization of the loss function as DISPLAYFORM3 , where (·, ·) is a convex λ-Lipschitz function in the first cordinate.

Then DISPLAYFORM4 where |α i | ≤ λC m ∀ i. This result holds in the compressed domain as well.

Proof.

If is an λ-Lipschitz function, its sub-gradient at every point is bounded by λ.

So by convexity, the unique optimizer is given by taking first-order conditions: DISPLAYFORM5 Since is Lipschitz, |∂ w T xi (w T x i , y i )| ≤ λ.

Therefore the first-order optimal solution (15) ofŵ can be expressed as FORMULA1 for some α 1 , . . .

, α m satisfying |α i | ≤ λC m ∀ i, which is the desired result.

DISPLAYFORM6 Also since 0 N ∈ X , A is also (X , ε)-RIP and the result then follows by the same argument as in (Calderbank et al., 2009, Lemma 4.2-3) .

DISPLAYFORM7 Proof.

The first bound follows by expanding ŵ 2 2 and using x 2 ≤ R; the second follows by expanding ŵ A 2 2 , applying Lemma C.2 to bound inner product distortion, and using x 2 ≤ R. Lemma C.3.

Letŵ be the linear classifier minimizing L S .

Then DISPLAYFORM8 Proof.

By Lemma C.1 we can re-expressŵ using Equation 14 and then apply the inequality from Lemma C.2 to get DISPLAYFORM9 for any x ∈ R N .

Since is λ-Lipschitz taking expectations over D implies DISPLAYFORM10 Substituting Equation 14 applying Lemma C.2 also yields DISPLAYFORM11 Together the inequalities bounding the loss term (16) and the regularization term FORMULA1 imply the result.

Lemma C.4.

Letŵ be the linear classifier minimizing L S and let w * be the linear classifier minimizing L D .

Then with probability 1 − γ DISPLAYFORM12 This result holds in the compressed domain as well.

Proof.

By Corollary C.1 we have thatŵ is contained in a closed convex subset independent of S. Therefore since is λ-Lipschitz, L is 1 C -strongly convex, and x 2 ≤ O(R), we have by BID34 , Theorem 1) that with probability 1 − γ DISPLAYFORM13 , which substituted into the previous equation completes the proof.

Proof of Theorem 4.2.

Applying Lemma C.4 in the compressed domain yields DISPLAYFORM14 , so together with Lemma C.3 and the previous inequality we have DISPLAYFORM15 We now apply Lemma C.4 in the sparse domain to get DISPLAYFORM16 2 , so by the previous inequality we have DISPLAYFORM17 Substituting the C that minimizes the r.h.s.

of this inequality completes the proof.

D PROOF OF LEMMA 4.1We assume the setting described in Lemma 4.1, where we are concerned with the RIP condition of the matrix A (n) when multiplying vectors x ∈ X (n)T , the set of BonC vectors for documents of length at most T .

This matrix can be written as DISPLAYFORM18 where A p is the d × V p matrix whose columns are the DisC embeddings of all p-grams in the vocabulary (and thus A (1) = A 1 = A, the matrix of the original word embeddings).

Note that from(1) any x ∈ X (n) T can be written as x = [x 1 , . . .

, x n ], where x p is a T -sparse vector whose entries correspond to p-grams.

Thus we also have DISPLAYFORM19 Proof.

By union bound we have that A p is (2k, ε)-RIP ∀ p ∈ [n] with probability at least 1 − nγ.

Thus by Definition 4.1 we have w. DISPLAYFORM20 Similarly, DISPLAYFORM21 .

From Definition 4.1, taking the square root of both sides of both inequalities completes the proof.

Definition D.1 BID11 .

Let D be a distribution over a subset S ⊂ R n .

Then the set Φ = {φ 1 , . . .

, φ N } of functions φ i : S → R is a bounded orthonormal system (BOS) with constant B if we have E D (φ i φ j ) = 1 i=j ∀ i, j and sup s∈S |φ i (s)| ≤ B ∀ i. Note that by definition B ≥ 1.

DISPLAYFORM22 Proof.

Note that by Theorem D.1 it suffices to show that √ dA p is a random sampling matrix associated with a BOS with constant B = 1.

Let D = U V {±1} be the uniform distribution over V i.i.d.

Rademacher random variables indexed by words in the vocabulary.

Then by definition the matrix A p ∈ R d×Vp can be constructed by drawing random variables DISPLAYFORM23 and assigning to the ijth entry of DISPLAYFORM24 gt , where each function φ j : {±1} V → R is uniquely associated to its p-gram.

It remains to be shown that this set of functions is a BOS with constant B = 1.For any two p-grams g, g and their functions φ i , φ j we have E D (φ i φ j ) = E x∼D p t=1 x gt x g t , which will be 1 iff each word in g ∪ g occurs an even number of times in the product and 0 otherwise.

Because all p-grams are uniquely defined under any permutation of its words (i.e. we are in fact using p-cooccurrences) and we have assumed that no p-gram contains a word more than once, each word occurs an even number of times in the product iff g = g ⇐⇒ i = j. Furthermore we have that DISPLAYFORM25 V ∀ i by construction.

Thus according to Definition D.1 the set of functions {φ 1 , . . .

, φ Vp } associated to the p-grams in the vocabulary is a BOS with constant B = 1.Proof of Lemma 4.1.

DISPLAYFORM26 .

Applying Lemma D.1 yields the result.

In Section 5.2, Definition 5.1 we introduced the Supporting Hyperplane Property (SHP), which by Corollary 5.1 characterizes when BP+ perfectly recovers a nonnegative signal.

Together with Lemmas A.1 and A.2 this fact also shows that SHP is a weaker condition than the well-known nullspace property (NSP): Corollary E.1.

If a matrix A ∈ R d×N with columns in general position satisfies NSP for some S ⊂ [N ] then it also satisfies S-SHP.In this section we give the entire proof of Corollary 5.1 and describe how to verify SHP given a design matrix and a set of support indices.

E.1 PROOF OF COROLLARY 5.1Recall that it suffices to show equivalence of A being S-SHP with the columns A S forming the vertices of a k-dimensional face of conv(A), where we can abuse notation to set A ∈ R d×(N +1) , with the extra column being the origin 0 d , so long as we constrain N + 1 ∈ S.( =⇒ ): The proof of the forward direction appeared in full in the proof sketch (see Section 5.2).

DISPLAYFORM0 We also know that F = { i∈S λ i A i : λ ∈ ∆ |S| } ⊆ H by convexity of H. Since any point y ∈ conv(A)\F can be written as y = DISPLAYFORM1 ∈ S such that λ j = 0, we have that DISPLAYFORM2 This implies that conv(A)\F ⊆ H − and F = conv(A) ∩ H, so since the columns of A are in general position F is a k-dimensional face of conv(A) whose vertices are the columns A S .

Recall that a matrix R d×N satisfies S-SHP for S ⊂ [N ] if there is a hyperplane containing the set of all columns of A indexed by S and the set of all other columns together with the origin are on one side of it.

Due to Corollary 5.1, checking S-SHP allows us to know whether all nonnegative signals with index support S will be recovered by BP+ without actually running the optimization on any one of them.

The property can be checked by solving a convex problem of the form The constraint enforces the property that the hyperplane contains all support embeddings, while the optimal objective value is zero iff SHP is satisfied (this follows from the fact that scaling h does not affect the constraint so if the minimal objective is zero for any single ε > 0 it is zero for all ε > 0).

The problem can be solved via using standard first or second-order equality-constrained convex optimization algorithms.

We set ε = 1 and p = 3 (to get a C 2 objective) and adapt the second-order method from Boyd & Vandenberghe (2004, Chapter 10) .

Our implementation can be found at https://github.com/NLPrinceton/sparse_recovery.

Figure 7: Efficiency of pretrained embeddings as sensing vectors at d = 300 dimensions, measured via the F 1 -score of the original BoW. 200 documents from each dataset were compressed and recovered in this experiment.

For fairness, the number of words V is the same for all embeddings so all documents are required to be subsets of the vocabulary of all corpora.

word2vec embeddings trained on Google News and GloVe vectors trained on Common Crawl were obtained from public repositories BID24 BID29 while Amazon and Wikipedia embeddings were trained for 100 iterations using a symmetric window of size 10, a min count of 100, for SN/GloVe a cooccurrence cutoff of 1000, and for word2vec a down-sampling frequency cutoff of 10 −5 and a negative example setting of 3.

300-dimensional normalized random vectors are used as a baseline.

We show in Figure 7 that the surprising effectiveness of word embeddings as linear measurement vectors for BoW signals holds for other embedding objectives and corpora as well.

Specifically, we see that widely used embeddings, when normalized, match the efficiency of random vectors for retrieving SST BoW and are more efficient when retrieving IMDB BoW. Interestingly, SN vectors are most efficient and are also the only embeddings for normalizing is not needed for good performance.

In Section 5.2 we gave some intuition for why pretrained word embeddings are efficient sensing vectors for natural language BoW by examining a geometric characterization of local equivalence due to BID9 in light of the usual similarity properties of word embeddings.

However, this analysis does not provide a rigorous theory for our empirical results.

In this section we briefly discuss a model-based justification that may lead to a stronger understanding.

We need a model relating BoW generation to the word embeddings trained over words co-occurring in the same BoW. As a starting point consider the model of BID0 , in which a corpus is generated by a random walk c t over the surface of a ball in R d ; at each t a word w is emitted w.p.

DISPLAYFORM0 Minimizing the SN objective approximately maximizes corpus likelihood under this model.

Thus in an approximate sense a document of length T is generated by setting a context vector c and emitting T words via (18) with c t = c. This model is a convenient one for analysis due its simplicity and invariance to word order as well as the fact that the approximate maximum likelihood document vector is the sum of the embeddings of words in the document.

Building upon the intuition established following Corollary 5.1 one can argue that, if we have the true latent SN vectors, then embeddings of words in the same document (i.e. emitted by the same context vector) will be close to each other and thus easy to separate from the embeddings of other words in the vocabulary.

However, we find empirically that not all of the T words closest to the sum of the word embeddings (i.e. the context vector) are the ones emitted; indeed individual word vectors in a document may have small, even negative inner product with the context vector and still be recovered via BP.

Thus any further theoretical argument must also be able to handle the recovery of lower probability words whose vectors are further away from the context vector than those of words that do not appear in the document.

We thus leave to future work the challenge of explaining why embeddings resulting from this (or another) model provide such efficient sensing matrices for natural language BoW.

<|TLDR|>

@highlight

We use the theory of compressed sensing to prove that LSTMs can do at least as well on linear text classification as Bag-of-n-Grams.