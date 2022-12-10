Stability is a key aspect of data analysis.

In many applications, the natural notion of stability is geometric, as illustrated for example in computer vision.

Scattering transforms construct deep convolutional representations which are certified stable to input deformations.

This stability to deformations can be interpreted as stability with respect to changes in the metric structure of the domain.



In this work, we show that scattering transforms can be generalized to non-Euclidean domains using diffusion wavelets, while preserving a notion of stability with respect to metric changes in the domain, measured with diffusion maps.

The resulting representation is stable to metric perturbations of the domain while being able to capture ''high-frequency'' information, akin to the Euclidean Scattering.

Convolutional Neural Networks (CNN) are layered information processing architectures.

Each of the layers in a CNN is itself the composition of a convolution operation with a pointwise nonlinearity where the filters used at different layers are the outcome of a data-driven optimization process BID22 .

Scattering transforms have an analogous layered architecture but differ from CNNs in that the convolutional filters used at different layers are not trained but selected from a multi-resolution filter bank BID25 BID3 .

The fact that they are not trained endows scattering transforms with intrinsic value in situations where training is impossible -and inherent limitations in the converse case.

That said, an equally important value of scattering transforms is that by isolating the convolutional layered architecture from training effects it permits analysis of the fundamental properties of CNN information processing architectures.

This analysis is undertaken in BID25 ; BID3 where the fundamental conclusion is about the stability of scattering transforms with respect to deformations in the underlying domain that are close to translations.

In this paper we consider graphs and signals supported on graphs such as brain connectivity networks and functional activity levels BID17 , social networks and opinions BID19 , or user similarity networks and ratings in recommendation systems BID18 .

Our specific goals are: (i) To define a family of graph-scattering transforms. (ii) To define a notion of deformation for graph signals. (iii) To study the stability of graph scattering transforms with respect to this notion of deformation.

To accomplish goal (i) we consider the family of graph diffusion wavelets which provide an appropriate construction of a multi-resolution filter bank BID8 .

Our diffusion scattering transforms are defined as the layered composition of diffusion wavelet filter banks and pointwise nonlinearities.

To accomplish goal (ii) we adopt the graph diffusion distance as a measure of deformation of the underlying domain BID27 .

Diffusion distances measure the similarity of two graphs through the time it takes for a signal to be diffused on the graph.

The major accomplishment of this paper is to show that the diffusion graph scattering transforms are stable with respect to deformations as measured with respect to diffusion distances.

Specifically, consider a signal x supported on graph G whose diffusion scattering transform is denoted by the operator Ψ G .

Consider now a deformation of the signal's domain so that the signal's support is now described by the graph G whose diffusion scattering operator is Ψ G .

We show that the operator norm distance Ψ G − Ψ G is bounded by a constant multiplied by the diffusion distance between the graphs G and G .

The constant in this bound depends on the spectral gap of G but, very importantly, does not depend on the number of nodes in the graph.

It is important to point out that finding stable representations is not difficult.

E.g., taking signal averages is a representation that is stable to domain deformations -indeed, invariant.

The challenge is finding a representation that is stable and rich in its description of the signal.

In our numerical analyses we show that linear filters can provide representations that are either stable or rich but that cannot be stable and rich at the same time.

The situation is analogous to (Euclidean) scattering transforms and is also associated with high frequency components.

We can obtain a stable representation by eliminating high frequency components but the representation loses important signal features.

Alternatively, we can retain high frequency components to have a rich representation but that representation is unstable to deformations.

Diffusion scattering transforms are observed to be not only stable -as predicted by our theoretical analysis -but also sufficiently rich to achieve good performance in graph signal classification examples.

Since graph and graph signals are of increasing interest but do not have the regular structure that would make use of CNNs appealing, it is pertinent to ask the question of what should be an appropriate generalization of CNNs to graphs and the graph signals whose topology they describe BID2 .

If one accepts the value of convolutions as prima facie, a natural solution is to replace convolutions with graph shift invariant filters which are known to be valid generalizations of (convolutional) time invariant filters BID4 .

This idea is not only natural but has been demonstrated to work well in practical implementations of Graph Neural Networks (GNNs) BID9 BID11 BID13 BID16 BID20 .

Same as Euclidean scattering transforms, our graph scattering transforms differ from GNNs in that they do not have to be trained.

The advantages and limitations of the absence of training notwithstanding, our work also sheds light on the question of why graph convolutions are appropriate generalizations of regular domain convolutions for signal classification problems.

Our work suggests that the value of GNNs stems from their stability relative to deformations of the underlying domain that are close to permutations -which is the property that a pair of graphs must satisfy to have small diffusion distance.

The stability results obtained in this paper build on the notion of scattering transforms.

These scattering representations were introduced by BID25 and further developed in BID3 with computer vision applications.

Since, these representations have been extended to handle transformations on more complex groups, such as roto-translations BID31 BID28 , and to domains such as audio processing BID0 and quantum chemistry BID10 .Similarly as in this work, extensions of scattering to general graphs have been considered in BID5 and BID34 .

BID5 focuses on Haar wavelets that hierarchically coarsen the graph, and relies on building multiresolution pairings.

The recent BID34 is closest to our work.

There, the authors define graph scattering using spectrally constructed wavelets from BID15 , and establish some properties of the resulting representation, such as energy conservation and stability to spectral perturbations.

In contrast, our stability results are established with respect to diffusion metric perturbations, which are generally weaker, in the sense that they define a weaker topology (see Section 3).

We use diffusion wavelets BID8 ) to obtain multi-resolution graph filter banks that are localized in frequency as well as in the graph domain, while spanning the whole spectrum.

Diffusion wavelets serve as the constructive basis for the obtained stability results.

Our work is also closely related to recent analysis of stability of Graph Neural Networks in the context of surface representations in BID21 .

In our work, however, we do not rely on extrinsic deformations and exploit the specific multiresolution structure of wavelets.

This section introduces our framework and states the desired stability properties of signal representations defined on general non-Euclidean domains.

Motivated by computer vision applications, our analysis starts with the notion of deformation stability.

If x(u) ∈ L 2 (Ω) is an image defined over an Euclidean domain Ω ⊂ R d , we are interested in signal representations Φ : L 2 (Ω) → R K that are stable with respect to small deformations.

If DISPLAYFORM0 ) denotes a change of variables with a differentiable field τ : Ω → Ω such that ∇τ < 1, then we ask DISPLAYFORM1 τ := ∇τ ∞ denoting a uniform bound on the operator norm of ∇τ .

In this setting, a notorious challenge to achieving (1) while keeping enough discriminative power in Φ(x) is to transform the high-frequency content of x in such a way that it becomes stable.

Scattering transforms BID25 BID3 provide such representations by cascading wavelet decompositions with pointwise modulus activation functions.

We briefly summarize here their basic definition.

Given a mother wavelet ψ ∈ L 1 (Ω) with at least a vanishing moment ψ(u)du = 0 and with good spatial localization, we consider rotated and dilated versions ψ j,c (u) = 2 −jd ψ(2 −j R c u) using scale parameter j and angle θ ∈ {2πc/C} c=0,...,C−1 .

A wavelet decomposition operator is defined as a filter bank spanning all scales up to a cutoff 2 J and all angles: DISPLAYFORM2 This filter bank is combined with a pointwise modulus activation function ρ(z) = |z|, as well as a low-pass average pooling operator U computing the average over the domain.

The resulting representation using m layers becomes DISPLAYFORM3 The resulting signal representation has the structure of a CNN, in which feature maps are not recombined with each other, and trainable filters are replaced by multiscale, oriented wavelets.

It is shown in BID25 that for appropriate signal classes and wavelet families, the resulting scattering transform satisfies a deformation stablity condition of the form (1), which has been subsequently generalised to broader multiresolution families BID33 .

In essence, the mechanism that provides stability is to capture high-frequency information with the appropriate spatio-temporal tradeoffs, using spatially localized wavelets.

Whereas deformations provide the natural framework to describe geometric stability in Euclidean domains, their generalization to non-Euclidean, non-smooth domains is not straightforward.

Let x ∈ L 2 (X ).

If X is embedded into a low-dimension Euclidean space Ω ⊂ R d , such as a 2-surface within a three-dimensional space, then one can still define meaningful deformations on X via extrinsic deformations of Ω BID21 .However, in this work we are interested in intrinsic notions of geometric stability, that do not necessarily rely on a pre-existent low-dimensional embedding of the domain.

The change of variables ϕ(u) = u − τ (u) defining the deformation can be seen as a perturbation of the Euclidean metric in DISPLAYFORM0 with dμ(u) = |I − ∇τ (u)|dµ(u), and |I − ∇τ (u)| ≈ 1 if ∇τ is small, where I is the identity.

Therefore, a possible way to extend the notion of deformation stability to general domains L 2 (X ) is to think of X as a metric space and reason in terms of stability of Φ : L 2 (X ) → R K to metric changes in X .

This requires a representation that can be defined on generic metric spaces, as well as a criteria to compare how close two metric spaces are.

Graphs are flexible data structures that enable general metric structures and modeling non-Euclidean domains.

The main ingredients of the scattering transform can be generalized using tools from computational harmonic analysis on graphs.

We note that, unlike the case of Euclidean domains, where deformations are equivalent whether they are analyzed from the function domain or its image, in the case of graphs, we focus on deformations on the underlying graph domain, while keeping the same function mapping (i.e. we model deformations as a change of the underlying graph support and analyze how this affects the interaction between the function mapping and the graph).In particular, diffusion wavelets BID8 provide a simple framework to define a multi-resolution analysis from powers of a diffusion operator defined on a graph.

A weighted, undirected graph G = (V, E, W ) with |V | = n nodes, edge set E and adjacency matrix W ∈ R n×n defines a diffusion process A in its nodes, given in its symmetric form by the normalized adjacency DISPLAYFORM0 where d i = (i,j)∈E W i,j denotes the degree of node i. Denote by d = W 1 the degree vector containing d i in the ith element.

By construction, A is well-localized in space (it is nonzero only where there is an edge connecting nodes), it is self-adjoint and satisfies A ≤ 1, where A is the operator norm.

Let λ 0 ≥ λ 1 ≥ . . .

λ n−1 denote its eigenvalues in decreasing order.

Defining DISPLAYFORM1 , one can easily verify that the normalized squared root degree vector DISPLAYFORM2 1 is the eigenvector with associated eigenvalue λ 0 = 1.

Also, note that λ n−1 = −1 if and only if G has a connected component that is non-trivial and bipartite BID6 .In the following, it will be convenient to assume that the spectrum of A (which is real and discrete since A is self-adjoint and in finite-dimensions) is non-negative.

Since we shall be taking powers of A, this will avoid folding negative eigenvalues into positive ones.

For that purpose, we adopt the so-called lazy diffusion, given by T = 1 2 (I + A).

In Section 4 we use this diffusion operator to define both a multiscale wavelet filter bank and a low-pass average pooling, leading to the diffusion scattering representation.

This diffusion operator can also be used to construct a metric on G. The so-called diffusion distances BID27 measure distances between two nodes x, x ∈ V in terms of their associated diffusion at time s: In this work, we build on this diffusion metric to define a distance between two graphs G, G .

Assuming first that G and G have the same size, the simplest formulation is to compare the diffusion metric generated by G and G up to a node permutation: DISPLAYFORM3 DISPLAYFORM4 where Π n is the space of n × n permutation matrices.

The diffusion distance is defined at a specific time s.

As s increases, this distance becomes weaker 1 , since it compares points at later stages of diffusion.

The role of time is thus to select the smoothness of the 'graph deformation', similarly as ∇τ measures the smoothness of the deformation in the Euclidean case.

For convenience, we denote d(G, G ) = d 1/2 (G, G ) and use the distance at s = 1/2 as our main deformation measure.

The quantity d defines a distance between graphs (seen as metric spaces) yielding a stronger topology than other alternatives such as the Gromov-Hausdorff distance, defined as DISPLAYFORM5 .

We choose d(G, G ) in this work for convenience and mathematical tractability, but leave for future work the study of stability relative to d s GH .

Finally, 1 In the sense that it defines a weaker topology, i.e., limm→∞ d DISPLAYFORM6 we consider for simplicity only the case where the sizes of G and G are equal, but definition (3.1) can be naturally extended to compare variable-sized graphs by replacing permutations by softcorrespondences (see BID1 .

Our goal is to build a stable and rich representation Φ G (x).

The stability property is stated in terms of the diffusion metric above: For a chosen diffusion time s, ∀ x ∈ R n , G = (V, E, W ), G = (V , E , W ) with |V | = |V | = n , we want DISPLAYFORM0 This representation can be used to model both signals and domains, or just domains G, by considering a prespecified x = f (G), such as the degree, or by marginalizing from an exchangeable distribution, DISPLAYFORM1 The motivation of FORMULA12 is two-fold: On the one hand, we are interested in applications where the signal of interest may be measured in dynamic environments that modify the domain, e.g. in measuring brain signals across different individuals.

On the other hand, in other applications, such as building generative models for graphs, we may be interested in representing the domain G itself.

A representation from the adjacency matrix of G needs to build invariance to node permutations, while capturing enough discriminative information to separate different graphs.

In particular, and similarly as with Gromov-Hausdorff distances, the definition of d(G, G ) involves a matching problem between two kernel matrices, which defines an NP-hard combinatorial problem.

This further motivates the need for efficient representations of graphs Φ G that can efficiently tell apart two graphs, and such that (θ) = Φ G − Φ G(θ) can be used as a differentiable loss for training generative models.

Let T be a lazy diffusion operator associated with a graph G of size n such as those described in Section 3.3.

Following BID8 , we construct a family of multiscale filters by exploiting the powers of the diffusion operator T 2 j .

We define DISPLAYFORM0 This corresponds to a graph wavelet filter bank with optimal spatial localization.

Graph diffusion wavelets are localized both in space and frequency, and favor a spatial localization, since they can be obtained with only two filter coefficients, namely h 0 = 1 for diffusion T 2 j−1 DISPLAYFORM1 The finest scale ψ 0 corresponds to one half of the normalized Laplacian operator DISPLAYFORM2 , here seen as a temporal difference in a diffusion process, seeing each diffusion step (each multiplication by ∆) as a time step.

The coarser scales ψ j capture temporal differences at increasingly spaced diffusion times.

For j = 0, . . .

, J n − 1, we consider the linear operator DISPLAYFORM3 which is the analog of the wavelet filter bank in the Euclidean domain.

Whereas several other options exist to define graph wavelet decompositions BID29 BID12 , and GNN designs that favor frequency localization, such as Cayley filters BID24 , we consider here wavelets that can be expressed with few diffusion terms, favoring spatial over frequential localization, for stability reasons that will become apparent next.

We choose dyadic scales for convenience, but the construction is analogous if one replaces scales 2 j by γ j for any γ > 1 in (6).If the graph G exhibits a spectral gap, i.e., β G = sup i=1,...

n−1 |λ i | < 1, the following proposition proves that the linear operator Ψ defines a stable frame.

Proposition 4.1.

For each n, let Ψ define the diffusion wavelet decomposition (7) and assume β G < 1.

Then there exists a constant 0 < C(β) depending only on β such that for any x ∈ R n satisfying x, v = 0, DISPLAYFORM4 This proposition thus provides the Littlewood-Paley bounds of Ψ, which control the ability of the filter bank to capture and amplify the signal x along each 'frequency' (i.e. the ability of the filter to increase or decrease the energy of the representation, relative to the energy of the x).

We note that diffusion wavelets are neither unitary nor analytic and therefore do not preserve energy.

However, the frame bounds in Proposition 4.1 provide lower bounds on the energy lost, such that the smaller 1 − β is, the less "unitary" our diffusion wavelets are.

It also informs us about how the spectral gap β determines the appropriate diffusion scale J: The maximum of p(u) = (u r − u 2r ) 2 is at u = 2 −1/r , thus the cutoff r * should align with β as r * = −1 log 2 β , since larger values of r capture energy in a spectral range where the graph has no information.

Therefore, the maximum scale can be adjusted as J = 1 + log 2 r * = 1 + log 2 −1 log 2 β .Recall that the Euclidean Scattering transform is constructed by cascading three building blocks: a wavelet decomposition operator, a pointwise modulus activation function, and an averaging operator.

Following the Euclidean scattering, given a graph G and x ∈ L 2 (G), we define an analogous Diffusion Scattering transform Φ G (x) by cascading three building blocks: the Wavelet decomposition operator Ψ, a pointwise activation function ρ, and an average operator U which extracts the average over the domain.

The average over a domain can be interpreted as the diffusion at infinite time, thus U x = lim t→∞ T t x = v T , x .

More specifically, we consider a first layer transformation given by DISPLAYFORM5 followed by second order coefficients DISPLAYFORM6 and so on.

The representation obtained from m layers of such transformation is thus DISPLAYFORM7 5 STABILITY OF GRAPH DIFFUSION SCATTERING

Given two graphs G, G of size n and a signal x ∈ R n , our objective is to bound DISPLAYFORM0 .

Let π * the permutation minimising the distortion between G and G in (4).

Since all operations in Φ are either equivariant or invariant with respect to permutations, we can assume w.l.o.g.

that π = 1, so that the diffusion distance can be directly computed by comparing nodes with the given order.

A key property of G that drives the stability of the diffusion scattering is given by its spectral gap 1 − β G = 1 − sup i=1,...

n−1 |λ i | ≥ 0.

In the following, we use 2 operator norms, unless stated otherwise.

Lemma 5.1.

DISPLAYFORM1 Remark: If diffusion distance is measured at time different from s = 1/2, the stability bound would be modified due to scales j such that 2 j < s.

The following lemma studies the stability of the low-pass operator U with respect to graph perturbations.

Lemma 5.2.

Let G, G be two graphs with same size, denote by v and v their respective squaredroot degree vectors, and by β, β their spectral gap.

Then DISPLAYFORM2 Spectral Gap asymptotic behavior Lemmas 5.1 and 5.2 leverage the spectral gap of the lazy diffusion operator associated with G. In some cases, such as regular graphs, the spectral gap vanishes asymptotically as n → ∞, thus degrading the upper bound asymptotically.

Improving the bound by leveraging other properties of the graph (such as regular degree distribution) is an important open task.

The scattering transform coefficients Φ G (x) obtained after m layers are given by equation 11, for low-pass operator U such that U x = v, x so that U = v T .From Lemma 5.1 we have that, DISPLAYFORM0 We also know, from Proposition 4.1 that Ψ conforms a frame, i.e. C(β) x 2 ≤ Ψx 2 ≤ x 2 for known constant C(β) given in Prop.

4.1.

Additionally, from Lemma 5.2 we get that DISPLAYFORM1 The objective now is to prove stability of the scattering coefficients Φ G (x), that is, to prove that DISPLAYFORM2 This is captured in the following Theorem: Theorem 5.3.

Let G, G be two graphs and let d(G, G ) be their distance measured as in equation 4.

Let T G and T G be the respective diffusion operators.

Denote by U G , ρ G and Ψ G and by U G , ρ G and Ψ G the low pass operator, pointwise nonlinearity and the wavelet filter bank used on the scattering transform defined on each graph, respectively, cf.

equation 11.

Assume ρ G = ρ G and that ρ G is non-expansive.

Let β − = min(β G , β G ), β + = max(β G , β G ) and assume β + < 1.

Then, we have that, for each k = 0, . . .

, m − 1, the following holds BID3 , it is straightforward to compute the stability bound on the scattering coefficients as follows.

Corollary 5.4.

In the context of Theorem 5.3, let x ∈ R n and let Φ G (x) be the scattering coefficients computed by means of equation 11 on graph G after m layers, and let Φ G (x) be the corresponding coefficients on graph G .

Then, DISPLAYFORM3 DISPLAYFORM4 Corollary 5.4 satisfies equation 5.

It also shows that the closer the graphs are in terms of the diffusion metric, the closer their scattering representations will be.

The constant is given by topological properties, the spectral gaps of G and G , as well as design parameters, the number of layers m. We observe that the stability bound grows the smaller the spectral gap is and also as more layers are considered.

The spectral gap is tightly linked with diffusion processes on graphs, and thus it does emerge from the choice of a diffusion metric.

Graphs with values of beta closer to 1, exhibit weaker diffusion paths, and thus a small perturbation on the edges of these graphs would lead to a larger diffusion distance.

The contrary holds as well.

In other words, the tolerance of the graph to edge perturbations (i.e., d(G, G) being small) depends on the spectral gap of the graph.

We also note that, as stated at the end of Section 5.1, the spectral gap appears in our upper bounds, but it is not necessarily sharp.

In particular, the spectral gap is a poor indication of stability in regular graphs, and we believe our bound can be improved by leveraging structural properties of regular domains.

Finally, we note that the size of the graphs impacts the stability result inasmuch as it impacts the distance measure d(G, G ).

This is expected, since graphs of different size can be compared, as mentioned in Section 3.3.

Different from BID34 , our focus is on obtaining graph wavelet banks that are localized in the graph domain to improve computational efficiency as discussed in BID9 .

We also notice that the scattering transform in BID34 is stable with respect to a graph measure that depends on the spectrum of the graph through both eigenvectors and eigenvalues.

More specifically, it is required that the spectrum gets concentrated as the graphs grow.

However, in general, it is not straightforward to relate the topological structure of the graph with its spectral properties.

As mentioned in Section 3.3, the stability is computed with a metric d(G, G ) which is stronger than what could be hoped for.

Our metric is permutation-invariant, in analogy with the rigid translation invariance in the Euclidean case, and stable to small perturbations around permutations.

The extension of (16) to weaker metrics, using e.g. multiscale deformations, is left for future work.

By denoting T j = T 2 j , observe that one can approximate the diffusion wavelets from (6) as a cascade of low-pass diffusions followed by a high-pass filter at resolution 2 j : DISPLAYFORM0 This pyramidal structure of multi-resolution analysis wavelets -in which each layer now corresponds to a different scale, shows that the diffusion scattering is a particular instance of GNNs where each layer j is generated by the pair of operators {I, T j−1 }.

If x (j) ∈ R n×dj denotes the feature representation at layer j using d j feature maps per node, the corresponding update is given by DISPLAYFORM1 where θ DISPLAYFORM2 In this case, a simple modification of the previous theorem shows that the resulting GNN representation DISPLAYFORM3 2 ) j≤J is also stable with respect to d(G, G ), albeit this time the constants are parameter-dependent:Corollary 5.5.

The J layer GNN with parameters Θ = (θ DISPLAYFORM4 This bound is thus learning-agnostic and is proved by elementary application of the diffusion distance definition.

An interesting question left for future work is to relate such stability to gradient descent optimization biases, similarly as in BID14 BID32 , which could provide stability certificates for learnt GNN representations.

In this section, we first show empirically the dependence of the stability result with respect to the spectral gap, and then we illustrate the discriminative power of the diffusion scattering transform in two different classification tasks; namely, the problems of authorship attribution and source localization.

Consider a small-world graph G with N = 200 nodes, edge probability p SW and rewiring probability q SW = 0.1.

Let x ∼ N (0, I) be a random graph signal defined on top of G and Φ G (x) the corresponding scattering transform.

Let G be another realization of the small-world graph, and let Φ G (x) be the scattering representation of the same graph signal x but on the different support G .

We can then proceed to compute Φ G (x) − Φ G (x) .

By changing the value of p SW we can change value of the spectral gap β and study the dependence of the difference in representations as a function of the spectral gap.

Results shown in FIG1 are obtained by varying p SW from 0.1 to 0.9.

For each value of p SW we generate one graph G and 50 different graphs G ; and for each graph G we compute Φ G (x) − Φ G (x) for 1, 000 different graph signal realizations x. The average across all signal realizations is considered the estimate of the representation difference, and then the mean as well as the variance across all graphs are computed and shown in the figure (error bars).

DISPLAYFORM0 as a function of the spectral gap (changing p SW from 0.1 to 0.9 led to values of spectral gap between 0.5 and close to 1).

First and foremost we observe that, indeed, as β reaches one, the stability result gets worse and the representation difference increases.

We also observe that, for deeper scattering representations, the difference also gets worse, although it is not a linear behaviour as predicted in equation 16, which suggest that the bound is not tight.

For classifying we train a SVM linear model fed by features obtained from different representations.

We thus compare with two non-trainable linear representations of the data: a data-based method (using the graph signals to feed the classifier) and a graph-based method (obtaining the GFT coefficients as features for the data).

Additionally, we consider the graph scattering with varying depth to analyze the richness of the representation.

Our aim is mainly to illustrate that the scattering representation is rich enough, relative to linear representations, and is stable to graph deformations.

First, we consider the problem of authorship attribution where the main task is to determine if a given text was written by a certain author.

We construct author profiles by means of word adjacency networks (WAN).

This WAN acts as the underlying graph support for the graph signal representing the word count (bag-of-words) of the target text of unknown authorship.

Intuitively, the choice of words of the target text should reflect the pairwise connections in the WAN, see BID30 for detailed construction of WANs.

In particular, we consider all works by Jane Austen.

To illustrate the stability result, we construct a WAN with 188 nodes (functional words) using a varying number of texts to form the training set, obtaining an array of graphs that are similar but not exactly the same.

For the test set, we include 154 excerpts by Jane Austen and 154 excerpts written by other contemporary authors, totaling 308 data points.

FIG1 shows classification error as a function of the number of training samples used.

We observe that graph scattering transforms monotonically improve while considering more training data, whereas other methods vary more erratically, showing their lack of stability (their representations vary more wildly when the underlying graph support changes).

This shows that scattering diffusion transforms strike a good balance between stability and discriminative power.

For the second task, let G be a 234-node graph modeling real-world Facebook interactions BID26 .

In the source localization problem, we observe a diffusion process after some unknown time t, that originated at some unknown node i, i.e. we observe x = W t δ i , where δ i is the signal with all zeros except a 1 on node i.

The objective is to determine which community the source node i belongs to.

These signals can be used to model rumors that percolate through the social network by interaction between users and the objective is to determine which user group generated said rumor (or initiated a discussion on some topic).

We generate a training sample of size 2, 000, for nodes i chosen at random and diffusion times t chosen as random as well.

The GFT is computed by projecting on the eigenbasis of the operator T .

We note that, to avoid numerical instabilities, the diffusion is carried out using the normalized operator (W/λ max (W )) and t ≤ t max = 20.

The representation coefficients (graph signals, GFT or scattering coefficients) obtained from this set are used to train different linear SVMs to perform classification.

For the test set, we draw 200 new signals.

We compute the classification errors on the test set as a measure of usefulness of the obtained representations.

Results are presented in FIG1 , where perturbations are illustrated by dropping edges with probability p (adding or removing friends in Facebook).

Again, it is observed that the scattering representation exhibits lower variations when the underlying graph changes, compared to the linear approaches.

Finally, to remark the discriminative power of the scattering representation, we observe that as the graph scattering grows deeper, the obtained features help in more accurate classification.

We remark that in regimes with sufficient labeled examples, trainable GNN architectures will generally outperform scattering-based representations.

In this work we addressed the problem of stability of graph representations.

We designed a scattering transform of graph signals using diffusion wavelets and we proved that this transform is stable under deformations of the underlying graph support.

More specifically, we showed that the scattering transform of a graph signal supported on two different graphs is proportional to the diffusion distance between those graphs.

As a byproduct of our analysis, we obtain stability bounds for Graph Neural Networks generated by diffusion operators.

Additionally, we showed that the resulting descriptions are also rich enough to be able to adequately classify plays by author in the context of authorship attribution, and identify the community origin of a signal in a source localization problem.

That said, there are a number of directions to build upon from these results.

First, our stability bounds depend on the spectral gap of the graph diffusion.

Although lazy diffusion prevents this spectral gap to vanish, as the size of the graph increases we generally do not have a tight bound, as illustrated by regular graphs.

An important direction of future research is thus to develop stability bounds which are robust to vanishing spectral gaps.

Next, and related to this first point, we are working on extending the analysis to broader families of wavelet decompositions on graphs and their corresponding graph neural network versions, including stability with respect to the GromovHausdorff metric, which can be achieved by using graph wavelet filter banks that achieve bounds analogous to those in Lemmas 5.1 and 5.2.A PROOF OF PROPOSITION 4.1Since all operators ψ j are polynomials of the diffusion T , they all diagonalise in the same basis.

Let T = V ΛV T , where V T V = I contains the eigenvectors of T and Λ = diag(λ 0 , . . .

, λ n−1 ) its eigenvalues.

The frame bounds C 1 , C 2 are obtained by evaluating Ψx 2 for x = v i , i = 1, . . .

, n− 1, since v 0 corresponds to the square-root degree vector and x is by assumption orthogonal to v 0 .We verify that the spectrum of ψ j is given by (p j (λ 0 ) , . . .

, p j (λ n−1 )), where DISPLAYFORM0 2 .

It follows from the definition that DISPLAYFORM1 . .

, n − 1 and therefore DISPLAYFORM2 We check that DISPLAYFORM3 2 .

One easily verifies that Q(x) is continuous in [0, 1) since it is bounded by a geometric series.

Also, observe that DISPLAYFORM4 since x ∈ [0, 1).

By continuity it thus follows that DISPLAYFORM5 which results in g (t) ≤ rβ r−1 B − A , proving (23).By plugging FORMULA5 into (22) we thus obtain DISPLAYFORM6 (1−β 2 ) 3 .

Finally, we observe that DISPLAYFORM7 Without loss of generality, assume that the node assignment that minimizes T G − ΠT G Π T is the identity.

We need to bound the leading eigenvectors of two symmetric matrices T G and T G with a spectral gap.

As before, let DISPLAYFORM8 Since we are free to swap the role of v and v , the result follows.

DISPLAYFORM9 First, note that ρ G = ρ G = ρ since it is a pointwise nonlinearity (an absolute value), and is independent of the graph topology.

Now, let's start with k = 0.

In this case, we get U G x − U G x which is immediately bounded by Lemma 5.2 satisfying equation 15.For k = 1 we have DISPLAYFORM10 where the triangular inequality of the norm was used, together with the fact that ρu − ρu ≤ ρ(u − u ) for any real vector u since ρ is the pointwise absolute value.

Using the submultiplicativity of the operator norm, we get DISPLAYFORM11 From Lemmas 5.1 and 5.2 we have that Ψ G − Ψ G ≤ ε Ψ and U G − U G ≤ ε U , and from Proposition 4.1 that Ψ G ≤ 1.

Note also that U G = U G = 1 and that ρ = 1.

This yields DISPLAYFORM12 satisfying equation 15 for k = 1.For k = 2, we observe that DISPLAYFORM13 The first term is bounded in a straightforward fashion by DISPLAYFORM14 analogy to the development for k = 1.

Since U G = 1, for the second term, we focus on DISPLAYFORM15 We note that, in the first term in equation 33, the first layer induces an error, but after that, the processing is through the same filter banks.

So we are basically interested in bounding the propagation of the error induced in the first layer.

Applying twice the fact that ρ(u) − ρ(u ) ≤ ρ(u − u ) we get DISPLAYFORM16 And following with submultiplicativity of the operator norm, DISPLAYFORM17 For the second term in equation 33, we see that the first layer applied is the same in both, namely ρΨ G so there is no error induced.

Therefore, we are interested in the error obtained after the first layer, which is precisely the same error obtained for k = 1.

Therefore, DISPLAYFORM18 Plugging equation 35 and equation 36 back in equation 31 we get DISPLAYFORM19 satisfying equation 15 for k = 2.For general k we see that we will have a first term that is the error induced by the mismatch on the low pass filter that amounts to ε U , a second term that accounts for the propagation through (k − 1) equal layers of an initial error, yielding ε Ψ , and a final third term that is the error induced by the previous layer, (k − 1)ε Ψ .

More formally, assume that equation 15 holds for k − 1, implying that DISPLAYFORM20 Then, for k, we can write DISPLAYFORM21 Again, the first term we bound it in a straightforward manner using submultiplicativity of the operator norm DISPLAYFORM22 For the second term, since U G = 1 we focus on DISPLAYFORM23 The first term in equation 42 computes the propagation in the initial error caused by the first layer.

Then, repeatedly applying ρ(u) − ρ(u ) ≤ ρ(u − u ) in analogy with k = 2 and using submultiplicativity, we get DISPLAYFORM24 The second term in equation 42 is the bounded by equation 38, since the first layer is exactly the same in this second term.

Then, combining equation 43 with equation 38, yields DISPLAYFORM25 Overall, we get DISPLAYFORM26 which satisfies equation 15 for k. Finally, since this holds for k = 2, the proof is completed by induction.

E PROOF OF COROLLARY 5.4From Theorem 5.3, we have DISPLAYFORM27 and, by definition (Bruna & Mallat, 2013, Sec. 3 .1), DISPLAYFORM28 so that DISPLAYFORM29 Then, applying the inequality of Theorem 5.3, we get DISPLAYFORM30 Now, considering each term, such that DISPLAYFORM31 + m−1 k=0 2 3/2 k β 2 + (1 + β 2 + ) (1 − β − )(1 − β 2 + ) 3 d

@highlight

Stability of scattering transform representations of graph data to deformations of the underlying graph support.