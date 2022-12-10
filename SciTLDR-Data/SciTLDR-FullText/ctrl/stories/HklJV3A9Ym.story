This paper extends the proof of density of neural networks in the space of continuous (or even measurable) functions on Euclidean spaces to functions on compact sets of probability measures.

By doing so the work parallels a more then a decade old results on mean-map embedding of probability measures in reproducing kernel Hilbert spaces.

The work has wide practical consequences for multi-instance learning, where it theoretically justifies some recently proposed constructions.

The result is then extended to Cartesian products, yielding universal approximation theorem for tree-structured domains, which naturally occur in data-exchange formats like JSON, XML, YAML, AVRO, and ProtoBuffer.

This has important practical implications, as it enables to automatically create an architecture of neural networks for processing structured data (AutoML paradigms), as demonstrated by an accompanied library for JSON format.

{"weekNumber":"39", "workouts":[ { "sport":"running", "distance":19738, "duration":1500, "calories":375, "avgPace":76, "speedData":{ "speed": [10, 9, 8] , "altitude": [100, 104, 103, 81] , "labels": ["0.0km","6.6km ","13.2km","19.7km "]}}, {"sport":"swimming", "distance":664, "duration":1800, "calories":250, "avgPace":2711}]} Prevalent machine learning methods assume their input to be a vector or a matrix of a fixed dimension, or a sequence, but many sources of data have the structure of a tree, imposed by data formats like JSON, XML, YAML, Avro, or ProtoBuffer (see FIG0 for an example).

While the obvious complication is that such a tree structure is more complicated than having a single variable, these formats also contain some "elementary" entries which are already difficult to handle in isolation.

Beside strings, for which a plethora conversions to real-valued vectors exists (one-hot encoding, histograms of n-gram models, word2vec BID15 , output of a recurrent network, etc.), the most problematic elements seem to be unordered lists (sets) of records (such as the "workouts" element and all of the subkeys of "speedData" in FIG0 ), whose length can differ from sample to sample and the classifier processing this input needs to be able to cope with this variability.

The variability exemplified above by "workouts" and "speedData" is the defining feature of Multiinstance learning (MIL) problems (also called Deep Sets in BID28 ), where it is intuitive to define a sample as a collection of feature vectors.

Although all vectors within the collection have the same dimension, their number can differ from sample to sample.

In MIL nomenclature, a sample is called a bag and an individual vector an instance.

The difference between sequences and bags is that the order of instances in the bag is not important and the output of the classifier should be the same for an arbitrary permutation of instances in the vector.

MIL was introduced in BID3 as a solution for a problem of learning a classifier on instances from labels available on the level of a whole bag.

To date, many approaches to solve the problem have been proposed, and the reader is referred to BID0 for an excellent review and taxonomy.

The setting has emerged from the assumption of a bag being considered positive if at least one instance was positive.

This assumption is nowadays used for problems with weakly-labeled data BID1 .

While many different definitions of the problem have been introduced (see BID6 for a review), this work adopts a general definition of BID16 , where each sample (bag) is viewed as a probability distribution observed through a set of realizations (instances) of a random variable with this distribution.

Rather than working with vectors, matrices or sequences, the classifier therefore classifies probability measures.

Independent works of BID28 ; BID5 and BID19 have proposed an adaptation of neural networks to MIL problems (hereinafter called MIL NN).

The adaptation uses two feed-forward neural networks, where the first network takes as an input individual instances, its output is an element-wise averaged, and the resulting vector describing the whole bag is sent to the second network.

This simple approach yields a very general, well performing and robust algorithm, which has been reported by all three works.

Since then, the MIL NN has been used in numerous applications, for example in causal reasoning BID20 , in computer vision to process point clouds BID25 BID27 , in medicine to predict prostate cancer BID11 , in training generative adversarial networks BID11 , or to process network traffic to detect infected computers BID18 .

The last work has demonstrated that the MIL NN construction can be nested (using sets of sets as an input), which allows the neural network to handle data with a hierarchical structure.

The wide-spread use of neural networks is theoretically justified by their universal approximation property -the fact that any continuous function on (a compact subset of) a Euclidean space to real numbers can be approximated by a neural network with arbitrary precision BID10 BID14 .

However, despite their good performance and increasing popularity, no general analogy of the universal approximation theorem has been proven for MIL NNs.

This would require showing that MIL NNs are dense in the space of continuous functions from the space of probability measures to real numbers and -to the best of our knowledge -the only result in this direction is restricted to input domains with finite cardinality BID28 .This work fills this gap by formally proving that MIL NNs with two non-linear layers, a linear output layer and mean aggregation after the first layer are dense in the space of continuous functions from the space of probability measures to real numbers (Theorem 2 and Corollary 3).

In Theorem 5, the proof is extended to data with an arbitrary tree-like schema (XML, JSON, ProtoBuffer).

The reasoning behind the proofs comes from kernel embedding of distributions (mean map) BID21 BID23 and related work on Maximum Mean Discrepancy BID7 .

This work can therefore be viewed as a formal adaptation of these tools to neural networks.

While these results are not surprising, the authors believe that as the number of applications of NNs to MIL and tree-structured data grows, it becomes important to have a formal proof of the soundness of this approach.

The paper only contains theoretical results -for experimental comparison to prior art, the reader is referred to BID28 BID18 .

However, the authors provide a proof of concept demonstration of processing JSON data at https: //codeocean.com/capsule/182df525-8417-441f-80ef-4d3c02fea970/?ID= f4d3be809b14466c87c45dfabbaccd32.

This section provides background for the proposed extensions of the universal approximation theorem BID10 BID14 .

For convenience, it also summarizes solutions to multiinstance learning problems proposed in BID19 ; BID5 .By C(K, R) we denote the space of continuous functions from K to R endowed with the topology of uniform convergence.

Recall that this topology is metrizable by the supremum metric ||f − g|| sup = sup x∈K |f (x) − g(x)|.Throughout the text, X will be an arbitrary metric space and P X will be some compact set of (Borel) probability measures on X .

Perhaps the most useful example of this setting is when X is a compact metric space and P X = P(X ) is the space of all Borel probability measures on X .

Endowing P X with the w topology turns it into a compact metric space (the metric being ρ * (p, q) = n 2 −n · | f n dp − f n dq| for some dense subset {f n | n ∈ N} ⊂ C(X , R) -see for example Proposition 62 from BID8 ).

Alternatively, one can define metric on P(X ) using for example integral probability metrics BID17 or total variation.

In this sense, the results presented below are general, as they are not tied to any particular topology.

The next definition introduces set of affine functions forming the base of linear and non-linear layers of neural networks.

DISPLAYFORM0 The main result of BID14 states that feed-forward neural networks with a single nonlinear hidden layer and linear output layer (hereinafter called Σ-networks) are dense in the space of continuous functions.

Lemma 1.1 then implies that the same holds for measurable functions.

Theorem 1 (Universal approximation theorem on R d ).

For any non-polynomial measurable function σ on R and every d ∈ N, the following family of functions is dense in C(R d , R): DISPLAYFORM1 The key insight of the theorem isn't that a single non-linear layer suffices, but the fact that any continuous function can be approximated by neural networks.

Recall that for K ⊂ R d compact, any f ∈ C(K, R) can be continuolusly extended to R d , and thus the same result holds for C(K, R).

Note that if σ was a polynomial of order k, Σ(σ, A d ) would only contain polynomials of order ≤ k.

The following metric corresponds to the notion of convergence in measure: Definition 2 (Def.

2.9 from BID10 ).

For a Borel probability measure µ on X , define a metric DISPLAYFORM2 on M (X , R), where M (X , R) denotes the collection of all (Borel) measurable functions.

Note that for finite µ, the uniform convergence implies convergence in ρ µ (Hornik, 1991, L. A.1) : DISPLAYFORM3

In Multi-instance learning it is assumed that a sample x consists of multiple vectors of a fixed dimension, i.e. x = {x 1 , . . .

, x l }, x i ∈ R d .

Furthermore, it is assumed that labels are provided on the level of samples x, rather than on the level of individual instances x i .To adapt feed-forward neural networks to MIL problems, the following construction has been proposed in BID19 ; BID5 .

Assuming mean aggregation function, the network consists of two feed-forward neural networks φ : DISPLAYFORM0 The output of function is calculated as follows: DISPLAYFORM1 where d, k, o is the dimension of the input, output of the first neural network, and the output.

This construction also allows the use of other aggregation functions such as maximum.

The general definition of a MIL problem BID16 ) adopted here views instances x i of a single sample x as realizations of a random variable with distribution p ∈ P X , where P X is a set of probability measures on X .

This means that the sample is not a single vector but a probability distribution observed through a finite number of realizations of the corresponding random variable.

The main result of Section 3 is that the set of neural networks with (i) φ being a single non-linear layer, (ii) ψ being one non-linear layer followed by a linear layer, and (iii) the aggregation function being mean as in Equation FORMULA5 is dense in the space C(P X , R) of continuous functions on any compact set of probability measures.

Lemma 1.1 extends the result to the space of measurable functions.

The theoretical analysis assumes functions f : P X → R of the form DISPLAYFORM2 whereas in practice p can only be observed through a finite set of observations x = {x i ∼ p|i ∈ {1, . . .

, l}}. This might seem as a discrepancy, but the sample x can be interpreted as a mixture of Dirac probability measures DISPLAYFORM3 from which it easy to recover Equation (4).

Since p x approaches p as l increases, f (x) can be seen as an estimate of f (p).

Indeed, if the non-linearities in neural networks implementing functions φ and ψ are continuous, the function f is bounded and from Hoeffding's inequality BID9 it follows that P (|f (p) − f (x)| ≥ t) ≤ 2 exp(−ct 2 l 2 ) for some constant c > 0.

To extend Theorem 1 to spaces of probability measures, the following definition introduces the set of functions which represent the layer that embedds probability measures into R. Definition 3.

For any X and set of functions F ⊂ {f : X → R}, we define A F as DISPLAYFORM0 A F can be viewed as an analogy of affine functions defined by Equation (1) in the context of probability measures P X on X .

Remark.

Let X ⊂ R d and suppose that F only contains the basic projections π i : x ∈ R d → x i ∈ R. If P X = {δ x |x ∈ X } is the set of Dirac measures, then A F coincides with A d .Using A F , the following definition extends the Σ-networks from Theorem 1 to probability spaces.

Definition 4 (Σ-networks).

For any X , set of functions F = {f : X → R}, and a measurable function σ : R → R, let Σ(σ, A F ) be class of functions f : DISPLAYFORM1 The main theorem of this work can now be presented.

As illustrated in a corollary below, when applied to F = Σ(σ, A d ) it states that three-layer neural networks, where first two layers are non-linear interposed with an integration (average) layer, allow arbitrarily precise approximations of continuous function on P X . (In other words this class of networks is dense in C(P X , R).)

Theorem 2.

Let P X be a compact set of Borel probability measures on a metric space X , F be a set of continuous functions dense in C(X , R), and finally σ : R → R be a measurable non-polynomial function.

Then the set of functions Σ(σ, A F ) is dense in C(P X , R).Using Lemma 1.1, an immediate corollary is that a similar result holds for measurable funcitons: Corollary 1 (Density of MIL NN in M (P X , R)).

Under the assumptions of Theorem 2, Σ(σ, A F ) is ρ µ -dense in M (P X , R) for any finite Borel measure µ on X .The proof of Theorem 2 is similar to the proof of Theorem 2.4 from BID10 .

One of the ingredients of the proof is the classical Stone-Weierstrass theorem BID24 .

Recall that a collection of functions is an algebra if it is closed under multiplication and linear combinations.

Stone-Weierstrass Theorem.

Let A ⊂ C(K, R) be an algebra of functions on a compact K. If (i) A separates points in K: (∀x, y ∈ K, x = y)(∃f ∈ A) : f (x) = f (y) and (ii) A vanishes at no point of K: (∀x ∈ K)(∃f ∈ A) : f (x) = 0, then the uniform closure of A is equal to C(K, R).Since Σ(σ, A F ) is not closed under multiplication, we cannot apply the SW theorem directly.

Instead, we firstly prove the density of the class of ΣΠ networks (Theorem 3) which does form an algebra, and then we extend the result to Σ-networks.

Theorem 3.

Let P X be a compact set of Borel probability measures on a metric space X , and F be a dense subset of C(X , R).

Then the following set of functions is dense in C(P X , R): DISPLAYFORM2 The proof shall use the following immediate corollary of Lemma 9.3.2 from BID4 .

Lemma 3.1 (Lemma 9.3.2 of Dudley FORMULA1 ).

Let (K, ρ) be a metric space and let p and q be two Borel probability measures on K. If p = q, then we have f dp = f dq for some f ∈ C(K, R).Proof of Theorem 3.

Since ΣΠ(F) is clearly an algebra of continuous functions on P X , it suffices to verify the assumptions of the SW theorem (separation and non-vanishing properties).(i) Separation: Let p 1 , p 2 ∈ P X be distinct.

By Lemma 3.1 there is some > 0 and f ∈ C(X , R) such that f dp 1 − f dp 2 = 3 .

Since F is dense in C(X , R), there exists g ∈ F such that max x∈X |f (x) − g(x)| < .

Using triangle inequality yields f dp 1 − f dp 2 = f (x) − g(x) + g(x)dp 1 (x) − f (x) − g(x) + g(x)dp 2 (x) ≤ f (x) − g(x)dp 1 (x) + f (x) − g(x)dp 2 (x) + g(x)dp 1 (x) − g(x)dp 2 (x) DISPLAYFORM3 Denoting f g (p) = gdp, it is trivial to see that f g ∈ ΣΠ(F).

It follows that ≤ |f g (p 1 ) − f g (p 2 )|, implying that ΣΠ(F) separates the points of X .(ii) Non-vanishing: DISPLAYFORM4 , we get 1 = f dp = (f − g + g) dp = (f (x) − g(x))dp(x) + g dp ≤ |f (x) − g(x)|dp(x) + g dp ≤ 1 2 + g dp.

Denote f g (q) = g dq, f g ∈ ΣΠ(F).

It follows that f g (p) ≥ 1 2 , and hence ΣΠ(F) vanishes at no point of P X .Since the assumptions of SW theorem are satisfied, ΣΠ(F) is dense in C(P X , R).The following simple lemma will be useful in proving Theorem 2.

Lemma 3.2.

If G is dense in C(Y, R), then for any h : X → Y, the collection of functions {g • h| g ∈ G} is dense in {φ • h| φ ∈ C(Y, R)}.Proof.

Let g ∈ C(Y, R) and g * ∈ G be such that max y∈Y |g(y) − g * (y)| ≤ .

Then we have DISPLAYFORM5 which proves the lemma.

Proof of Theorem 2.

Theorem 2 is a consequence of Theorem 3 and Σ-networks being dense in C(R k , R) for any k.

Let X , F, P X , and σ be as in the assumptions of the theorem.

Let f * ∈ C(P X , R) and fix > 0.

DISPLAYFORM6 This function is of the form DISPLAYFORM7 f ij dp for some α i ∈ R and f ij ∈ F. Moreover f can be written as a composition f = g • h, where h : p ∈ P X → f 11 dp, f 12 dp, . . . , f nln dp , (9) DISPLAYFORM8 Denoting s = n i=1 l i , we identify the range of h and the domain of g with R s .Since g is clearly continuous and DISPLAYFORM9 Sinceg ∈ Σ(σ, A s ), it is easy to see thatf belongs to Σ(σ, A F ), which concludes the proof.

The function h in the above construction (Equation (9) ) can be seen as a feature extraction layer embedding the space of probability measures into a Euclidean space.

It is similar to a meanmap BID21 BID23 ) -a well-established paradigm in kernel machines -in the sense that it characterizes a class of probability measures but, unlike mean-map, only in parts where positive and negative samples differ.

The next result is the extension of the universal approximation theorem to product spaces, which naturally occur in structured data.

The motivation here is for example if one sample consists of some real vector x, set of vectors DISPLAYFORM0 and another set of vectors DISPLAYFORM1 .Theorem 4.

Let X 1 × · · · × X l be a Cartesian product of metric compacts, F i , i = 1, . . .

, l be dense subsets of C(X i , R), and σ : R → R be a measurable function which is not an algebraic polynomial.

DISPLAYFORM2 The theorem is general in the sense that it covers cases where some X i are compact sets of probability measures as defined in Section 2, some are subsets of Euclidean spaces, and others can be general compact spaces for which the corresponding sets of continuous function are dense in C(X i , R).The theorem is a simple consequence of the following corollary of Stone-Weierstrass theorem.

Corollary 2.

For K 1 and K 2 compact, the following set of functions is dense in C(K 1 × K 2 , R) DISPLAYFORM3 Proof of Theorem 4.

The proof is technically similar to the proof of Theorem 2.

Specifically, let f be a continuous function on X 1 × · · · × X l and > 0.

By the aforementioned corollary of the SW theorem, there are some f ij ∈ F j , i = 1, . . .

, n, j = 1, . . .

, l such that DISPLAYFORM4 Again, the above function can be written as a composition of two functions DISPLAYFORM5 DISPLAYFORM6 Since g is continuous, Theorem 1 can be applied to obtain a functiong of the formg( DISPLAYFORM7 , for some α i ∈ R and a i ∈ A nl , which approximates g with error at most .

Applying Lemma 3.2 to g, h, andg concludes the proof.

The following corollary of Theorem 2 justifies the embedding paradigm of BID28 ; BID5 ; BID19 to MIL problems: Corollary 3 (Density of MIL NN in C(P X , R)).

Let X be a compact subset of R d and P X a compact set of probability measures on X .

Then any function f ∈ C(P X , R) can be arbitrarily closely approximated by a three-layer neural network composed of two non-linear layers with integral (mean) aggregation layer between them, and a linear output layer.

If F in Theorem 2 is set to all feed-forward networks with a single non-linear layer (that is, when F = Σ(σ, A d )) then the theorem says that for every f ∈ C(P X , R) and > 0, there is somẽ DISPLAYFORM0 Thisf can be written as DISPLAYFORM1 where for brevity the bias vectors are omitted, σ and are element-wise, and W (·) are matrices of appropriate sizes.

Since the integral in the middle is linear with respect to the matrix-vector multiplication, W 2 and W 3 can be replaced by a single matrix, which proves the corollary: DISPLAYFORM2 Since Theorem 2 does not have any special conditions on X except to be compact metric space and F to be continuous and uniformly dense in X , the theorem can be used as an induction step and the construction can be repeated.

For example, consider a compact set of probability measures P P X on a P X .

Then the space of neural networks with four layers is dense in C(P P X , R).

The network consists of three non-linear layers with integration (mean) layer between them, and the last layer which is linear.

The above induction is summarized in the following theorem.

Theorem 5.

Let S be the class of spaces which (i) contains all compact subsets of R d , d ∈ N, (ii) is closed under finite cartesian products, and (iii) for each X ∈ S we have P(X ) ∈ S.1 Then for each X ∈ S, every continuous function on X can be arbitrarilly well approximated by neural networks.

By Lemma 1.1, an analogous result holds for measurable functions.

Proof.

It suffices to show that S is contained in the class W of all compact metric spaces X for which functions realized by neural networks are dense in C(W, R).

By Theorem 1, W satisfies (i).

The properties (ii) and (iii) hold for W by Theorems 4 and 2.

It follows that W ⊃ S.

Works most similar to this one are on kernel mean embedding BID21 BID23 , showing that a probability measure can be uniquely embedded into high-dimensional space using characteristic kernel.

Kernel mean embedding is widely used in Maximum Mean Discrepancy BID7 and in Support Measure Machines BID16 BID2 , and is to our knowledge the only algorithm with proven approximation capabilities comparable to the present work.

Unfortunately its worst-case complexity of O(l 3 b 2 ), where l is the number of bags and b is the average size of a bag, prevents it from scaling to problems above thousands of bags.

The MIL problem has been studied in BID26 proposing to use a LSTM network augmented by memory.

The reduction from sets to vectors is indirect by computing a weighted average over elements in an associative memory.

Therefore the aggregation tackled here is an integral part of architecture.

The paper lacks any approximation guarantees.

Problems, where input data has a tree structure, naturally occur in language models, where they are typically solved by recurrent neural networks (Irsoy & Cardie, 2014; BID22 .

The difference between these models is that the tree is typically binary and all leaves are homogeneous in the sense that either each of them is a vector representation of a word or each of them is a vector representation of an internal node.

Contrary, here it is assumed that the tree can have an arbitrary number of heterogeneous leaves following a certain fixed scheme.

Due to lack of space, the authors cannot list all works on MIL.

The reader is instead invited to look at the excellent overview in BID0 and the works listed in the introductory part of this paper.

This work has been motivated by recently proposed solutions to multi-instance learning BID28 ; BID19 ; BID5 and by mean-map embedding of probability measures BID23 .

It generalizes the universal approximation theorem of neural networks to compact sets of probability measures over compact subsets of Euclidean spaces.

Therefore, it can be seen as an adaptation of the mean-map framework to the world of neural networks, which is important for comparing probability measures and for multi-instance learning, and it proves the soundness of the constructions of BID19 ; BID5 .The universal approximation theorem is extended to inputs with a tree schema (structure) which, being the basis of many data exchange formats like JSON, XML, ProtoBuffer, Avro, etc., are nowadays ubiquitous.

This theoretically justifies applications of (MIL) neural networks in this setting.

As the presented proof relies on the Stone-Weierstrass theorem, it restricts non-linear functions in neural networks to be continuous in all but the last non-linear layer.

Although this does not have an impact on practical applications (all commonly use nonlinear functions within neural networks are continuous) it would be interesting to generalize the result to non-continuous non-linearities, as has been done for feed-forward neural networks in BID14 .

<|TLDR|>

@highlight

This paper extends the proof of density of neural networks in the space of continuous (or even measurable) functions on Euclidean spaces to functions on compact sets of probability measures. 

@highlight

This paper investigates the approximation properties of a family of neural networks designed to address multi-instance learning problems, and shows that results for standard one layer architectures extend to these models.

@highlight

This paper generalizes the universal approximation theorem to real functions on the space of measures.