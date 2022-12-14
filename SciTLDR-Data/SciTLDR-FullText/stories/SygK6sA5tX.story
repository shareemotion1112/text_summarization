One of the most notable contributions of deep learning is the application of convolutional neural networks (ConvNets) to structured signal classification, and in particular image classification.

Beyond their impressive performances in supervised learning, the structure of such networks inspired the development of deep filter banks referred to as scattering transforms.

These transforms apply a cascade of wavelet transforms and complex modulus operators to extract features that are invariant to group operations and stable to deformations.

Furthermore, ConvNets inspired recent advances in geometric deep learning, which aim to generalize these networks to graph data by applying notions from graph signal processing to learn deep graph filter cascades.

We further advance these lines of research by proposing a geometric scattering transform using graph wavelets defined in terms of random walks on the graph.

We demonstrate the utility of features extracted with this designed deep filter bank in graph classification of biochemistry and social network data (incl.

state of the art results in the latter case), and in data exploration, where they enable inference of EC exchange preferences in enzyme evolution.

Over the past decade, numerous examples have established that deep neural networks (i.e., cascades of linear operations and simple nonlinearities) typically outperform traditional "shallow" models in various modern machine learning applications, especially given the increasing Big Data availability nowadays.

Perhaps the most well known example of the advantages of deep networks is in computer vision, where the utilization of 2D convolutions enable network designs that learn cascades of convolutional filters, which have several advantages over fully connected network architectures, both computationally and conceptually.

Indeed, in terms of supervised learning, convolutional neural networks (ConvNets) hold the current state of the art in image classification, and have become the standard machine learning approach towards processing big structured-signal data, including audio and video processing.

See, e.g., Goodfellow et al. (2016, Chapter 9) for a detailed discussion.

Beyond their performances when applied to specific tasks, pretrained ConvNet layers have been explored as image feature extractors by freezing the first few pretrained convolutional layers and then retraining only the last few layers for specific datasets or applications (e.g., BID47 BID33 .

Such transfer learning approaches provide evidence that suitably constructed deep filter banks should be able to extract task-agnostic semantic information from structured data, and in some sense mimic the operation of human visual and auditory cortices, thus supporting the neural terminology in deep learning.

An alternative approach towards such universal feature extraction was presented in BID28 , where a deep filter bank, known as the scattering transform, is designed, rather than trained, based on predetermined families of distruptive patterns that should be eliminated to extract informative representations.

The scattering transform is constructed as a cascade of linear wavelet transforms and nonlinear complex modulus operations that provides features with guaranteed invariance to a predetermined Lie group of operations such as rotations, translations, or scaling.

Further, it also provides Lipschitz stability to small diffeomorphisms of the inputted signal.

Scattering features have been shown to be effective in several audio (e.g., BID6 BID0 BID27 and image (e.g., BID7 BID40 BID34 processing applications, and their advantages over learned features are especially relevant in applications with relatively low data availability, such as quantum chemistry (e.g., BID15 BID35 .Following the recent interest in geometric deep learning approaches for processing graph-structured data (see, for example, BID4 and references therein), we present here a generalization of the scattering transform from Euclidean domains to graphs.

Similar to the Euclidean case, our construction is based on a cascade of bandpass filters, defined in this case using graph signal processing BID38 notions, and complex moduli, which in this case take the form of absolute values (see Sec. 3).

While several choices of filter banks could generally be used with the proposed cascade, we focus here on graph wavelet filters defined by lazy random walks (see Sec. 2).

These wavelet filters are also closely related to diffusion geometry and related notions of geometric harmonic analysis, e.g. the diffusion maps algorithm of BID10 and the associated diffusion wavelets of BID11 .

Therefore, we call the constructed cascade geometric scattering, which also follows the same terminology from geometric deep learning.

We note that similar attempts at generalizing the scattering transform to graphs have been presented in BID9 as well as BID49 and BID17 .

The latter two works are most closely related to the present paper.

In them, the authors focus on theoretical properties of the proposed graph scattering transforms, and show that such transforms are invariant to graph isomorphism.

The geometric scattering transform that we define here also possesses the same invariance property, and we expect similar stability properties to hold for the proposed construction as well.

However, in this paper we focus mainly on the practical applicability of geometric scattering transforms for graph-structured data analysis, with particular emphasis on the task of graph classification, which has received much attention recently in geometric deep learning (see Sec. 4) In supervised graph classification problems one is given a training database of graph/label pairs DISPLAYFORM0 ??? G ?? Y sampled from a set of potential graphs G and potential labels Y. The goal is to use the training data to learn a model f : G ??? Y that associates to any graph G ??? G a label y = f (G) ??? Y. These types of databases arise in biochemistry, in which the graphs may be molecules and the labels some property of the molecule (e.g., its toxicity), as well as in various types of social network databases.

Until recently, most approaches were kernel based methods, in which the model f was selected from the reproducing kernel Hilbert space generated by a kernel that measures the similarity between two graphs; one of the most successful examples of this approach is the Weisfeiler-Lehman graph kernel of BID37 .

Numerous feed forward deep learning algorithms, though, have appeared over the last few years.

In many of these algorithms, task based (i.e., dependent upon the labels Y) graph filters are learned from the training data as part of the larger network architecture.

These filters act on a characteristic signal x G that is defined on the vertices of any graph G, e.g., x G may be a vector of degrees of each vertex (we remark there are also edge based algorithms, such as BID20 and references within, but these have largely been developed for and tested on databases not considered in Sec. 4).

Here, we propose an alternative to these methods in the form of a geometric scattering classifier (GSC) that leverages graph-dependent (but not label dependent) scattering transforms to map each graph G to the scattering features extracted from x G .

Furthermore, inspired by transfer learning approaches such as BID33 , we consider treatment of our scattering cascade as frozen layers on x G , either followed by fully connected classification layers (see FIG2 ), or fed into other classifiers such as SVM or logistic regression.

We note that while the formulation in Sec. 3 is phrased for a single signal x G , it naturally extends to multiple signals by concatenating their scattering features.

In Sec. 4.1 we evaluate the quality of the scattering features and resulting classification by comparing it to numerous graph kernel and deep learning methods over 13 datasets (7 biochemistry ones and 6 social network ones) commonly studied in related literature.

In terms of classification accuracy on individual datasets, we show that the proposed approach obtains state of the art results on two datasets and performs competitively on the rest, despite only learning a classifier that come after the geometric scattering transform.

Furthermore, while other methods may excel on specific datasets, when considering average accuracy: within social network data, our proposed GSC outperforms all other methods; in biochemistry or over all datasets, it outperforms nearly all feed forward neural network approaches, and is competitive with state of the art results of graph kernels BID26 and graph recurrent neural networks BID41 .

We regard this result as crucial in establishing the universality of graph features extracted by geometric scattering, as they provide an effective task-independent representation of analyzed graphs.

Finally, to establish their unsupervised qualities, in Sec. 4.2 we use geometric scattering features extracted from enzyme data BID2 to infer emergent patterns of enzyme commission (EC) exchange preferences in enzyme evolution, validated with established knowledge from BID12 .

We define graph wavelets as the difference between lazy random walks that have propagated at different time scales, which mimics classical wavelet constructions found in BID29 as well as more recent constructions found in BID11 .

The underpinnings for this construction arise out of graph signal processing, and in particular the properties of the graph Laplacian.

Let G = (V, E, W ) be a weighted graph, consisting of n vertices V = {v 1 , . . .

, v n }, edges E ??? {(v , v m ) : 1 ??? , m ??? n}, and weights W = {w(v , v m ) > 0 : (v , v m ) ??? E}. Note that unweighted graphs are considered as a special case, by setting w (v , v m DISPLAYFORM0 and zero otherwise, where we use the notation A(v , v m ) to denote the ( , m) entry of the matrix A so as to emphasize the correspondence with the vertices in the graph and to reserve sub-indices for enumerating objects.

Define the (weighted) degree of vertex v as DISPLAYFORM1 The graph Laplacian is a symmetric, real valued positive semi-definite matrix, and thus has n nonnegative eigenvalues.

Furthermore, if we set 0 = (0, . . .

, 0)T to to be the n ?? 1 vector of all zeroes, and 1 = (1, . . . , 1)T to be the analogous vector of all ones, then it is easy to see that L1 = 0.

Therefore 0 is an eigenvalue of L and we write the n eigenvalues of L as 0 = ?? 0 ??? ?? 1 ??? ?? ?? ?? ??? ?? n???1 with corresponding n ?? 1 orthonormal eigenvectors 1/ ??? n = ?? 0 , ?? 1 , . . .

, ?? n???1 .

If the graph G is connected, then ?? 1 > 0.

In order to simplify the following discussion we assume that this is the case, although the discussion below can be amended to include disconnected graphs as well.

Since ?? 0 is constant and every other eigenvector is orthogonal to ?? 0 , it is natural to view the eigenvectors ?? k as the Fourier modes of the graph G, with a frequency magnitude ??? ?? k .

Let x : V ??? R be a signal defined on the vertices of the graph G, which we will consider as an n ?? 1 vector with entries x(v ).

It follows that the Fourier transform of x can be defined as x(k) = x ?? ?? k , where x ?? y is the standard dot product.

This analogy is one of the foundations of graph signal processing and indeed we could use this correspondence to define wavelet operators on the graph G, as in BID22 .

Rather than follow this path, though, we instead take a related path similar to BID11 ; BID17 by defining the graph wavelet operators in terms of random walks defined on G, which will avoid diagonalizing L and will allow us to control the "spatial" graph support of the filters directly.

Define the n ?? n transition matrix of a lazy random random walk as P = 1 2 D ???1 A + I .

Note that the row sums of P are all one and thus the entry P(v , v m ) corresponds to the transition probability of walking from vertex v to v m in one step.

Powers of P run the random walk forward, so that in particular P t (v , v m ) is the transition probability of walking from v to v m in exactly t steps.

We will use P as a left multiplier, in which case P acts a diffusion operator.

To understand this idea more precisely, first note that a simple calculation shows that P1 = 1 and furthermore if the graph G is connected, every other eigenvalue of P is contained in [0, 1).

Note in particular that L and P share the eigenvector 1.

It follows that P t x responds most significantly to the zero frequency x(0) of x while depressing the non-zero frequencies of x (where the frequency modes are defined in terms of the graph Laplacian L, as described above).

On the spatial side, the value P t x(v ) is the weighted average of x(v ) with all values x(v m ) such that v m is within t steps of v in the graph G.High frequency responses of x can be recovered in multiple different fashions, but we utilize multiscale wavelet transforms that group the non-zero frequencies of G into approximately dyadic bands.

As shown in Mallat (2012, Lemma 2.12), wavelet transforms are provably stable operators in the Euclidean domain, and the proof of Zou & Lerman (2018, Theorem 5.1) indicates that similar results on graphs may be possible.

Furthermore, the multiscale nature of wavelet transforms will allow the resulting geometric scattering transform (Sec. 3) to traverse the entire graph G in one layer, which is valuable for obtaining global descriptions of G. Following BID11 , define the n ?? n diffusion wavelet matrix at the scale 2 j as DISPLAYFORM2 Since P t 1 = 1 for every t, we see that ?? j 1 = 0 for each j ??? 1.

Thus each ?? j x partially recovers x(k) for k ??? 1.

The value ?? j x(v ) aggregates the signal information x(v m ) from the vertices v m .

Instead, it responds to sharp transitions or oscillations of the signal x within the neighborhood of v with radius 2 j (in terms of the graph path distance).

Generally, the smaller j the higher the frequencies ?? j x recovers in x. These high frequency wavelet coefficients up to the scale 2 J are denoted by: DISPLAYFORM3 Since 2 J controls the maximum scale of the wavelet, in the experiments of Sec. 4 we select J such that 2 J ??? diam(G).

FIG0 plots the diffusion wavelets at different scales on two different graphs.

A geometric wavelet scattering transform follows a similar construction as the (Euclidean) wavelet scattering transform of BID28 , but leverages a graph wavelet transform.

In this paper we utilize the wavelet transform defined in (2) of the previous section, but remark that in principle any graph wavelet transform could be used (see, e.g., BID49 .

In Sec. 3.1 we define the graph scattering transform, in Sec. 3.2 we discuss its relation to other recently proposed graph scattering constructions BID17 BID49 , and in Sec. 3.3 we describe several of its desirable properties as compared to other geometric deep learning algorithms on graphs.

Machine learning algorithms that compare and classify graphs must be invariant to graph isomorphism, i.e., re-indexations of the vertices and corresponding edges.

A common way to obtain invariant graph features is via summation operators, which act on a signal x = x G that can be defined on any graph G, e.g., DISPLAYFORM0 The geometric scattering transform, which is described in the remainder of this section, follows such an approach.

The simplest of such summation operators computes the sum of the responses of the signal x. As described in BID44 , this invariant can be complemented by higher order summary statistics of x, the collection of which form statistical moments, and which are also referred to as "capsules" in that work.

For example, the unnormalized q th moments of x yield the following "zero" order geometric scattering moments: DISPLAYFORM1 We can also replace (3) with normalized (i.e., standardized) moments of x, in which case we store its mean (q = 1), variance (q = 2), skew (q = 3), kurtosis (q = 4), and so on.

In the numerical experiments described in Sec. 4 we take Q = 2, 3, 4 depending upon the database, where Q is chosen via cross validation to optimize classification performance.

Higher order moments are not considered as they become increasingly unstable, and we report results for both normalized and unnormalized moments.

In what follows we discuss the unnormalized moments, since their presentation is simpler and we use them in conjunction with fully connected layers (FCL) for classification purposes, but the same principles also apply to normalized moments (e.g., used with SVM and logistic regression in our classification results).

The invariants Sx(q) do not capture the full variability of x and hence the graph G upon which the signal x is defined.

We thus complement these moments with summary statistics derived from the wavelet coefficients of x, which in turn will lead naturally to the graph ConvNet structure of the geometric scattering transform.

Observe, analogously to the Euclidean setting, that in computing Sx FORMULA3 , which is the summation of x(v ) over V , we have captured the zero frequency of x since DISPLAYFORM2 .

Higher order moments of x can incorporate the full range of frequencies in x, e.g. DISPLAYFORM3 2 , but they are mixed into one invariant coefficient.

We can separate and recapture the high frequencies of x by computing its wavelet coefficients ?? (J) x, which were defined in (2).

However, ?? (J) x is not invariant to permutations of the vertex indices; in fact, it is covariant (or equivariant).

Before summing the individual wavelet coefficient vectors ?? j x, though, we must first apply a pointwise nonlinearity.

Indeed, define the n ?? 1 vector d(v ) = deg(v ), and note that ?? j x ?? d = 0 since one can show that d is a left eigenvector of P with eigenvalue 1.

If G is a regular graph then d = c1 from which it follows that ?? j x ?? 1 = 0.

For more general graphs d(v ) ??? 0 for v ??? V , which implies that for many graphs 1 ?? d will be the dominating coefficient in an expansion of 1 in an orthogonal basis containing d; it follows that in these cases |?? j x ?? 1| 1.We thus apply the absolute value nonlinearity, to obtain nonlinear covariant coefficients |?? (J) x| = {|?? j x| : 1 ??? j ??? J}. We use absolute value because it is covariant to vertex permutations, nonexpansive, and when combined with traditional wavelet transforms on Euclidean domains, yields a provably stable scattering transform for q = 1.

Furthermore, initial theoretical results in BID49 ; BID17 indicate that similar graph based scattering transforms possess certain types of stability properties as well.

As in (3), we extract invariant coefficients from |?? j x| by computing its moments, which define the first order geometric scattering moments: DISPLAYFORM4 These first order scattering moments aggregate complimentary multiscale geometric descriptions of G into a collection of invariant multiscale statistics.

These invariants give a finer partition of the frequency responses of x. For example, whereas Sx(2) mixed all frequencies of x, we see that Sx(j, 2) only mixes the frequencies of x captured by the graph wavelet ?? j .First order geometric scattering moments can be augmented with second order geometric scattering moments by iterating the graph wavelet and absolute value transforms, which leads naturally to the structure of a graph ConvNet.

These moments are defined as: DISPLAYFORM5 which consists of reapplying the wavelet transform operator ?? (J) to each |?? j x| and computing the summary statistics of the magnitudes of the resulting coefficients.

The intermediate covariant coefficients |?? j |?? j x|| and resulting invariant statistics Sx(j, j , q) couple two scales 2 j and 2 j within the graph G, thus creating features that bind patterns of smaller subgraphs within G with patterns of larger subgraphs (e.g., circles of friends of individual people with larger community structures in social network graphs).

The transform can be iterated additional times, leading to third order features and beyond, and thus has the general structure of a graph ConvNet.

The collection of graph scattering moments Sx = {Sx(q), Sx(j, q), Sx(j, j , q)} (illustrated in FIG2 ) provides a rich set of multiscale invariants of the graph G. These can be used in supervised settings as input to graph classification or regression models, or in unsupervised settings to embed graphs into a Euclidean feature space for further exploration, as demonstrated in Sec. 4.

In order to assess the utility of scattering features for representing graphs, two properties have to be considered: stability and capacity.

First, the stability property aims to essentially provide an upper bound on distances between similar graphs that only differ by types of deformations that can be treated as noise.

This property has been the focus of both BID49 and BID17 , and in particular the latter shows that a diffusion scattering transform yields features that are stable to graph structure deformations whose size can be computed via the diffusion framework BID11 ) that forms the basis for their construction.

While there are some technical differences between the geometric scattering here and the diffusion scattering in BID17 , these constructions are sufficiently similar that we can expect both of them to have analogous stability properties.

Therefore, we mainly focus here on the complementary property of the scattering transform capacity to provide a rich feature space for representing graph data without eliminating informative variance in them.

We note that even in the classical Euclidean case, while the stability of scattering transforms to deformations can be established analytically BID28 , their capacity is typically examined by empirical evidence when applied to machine learning tasks (e.g., BID5 BID39 BID0 .

Similarly, in the graph processing settings, we examine the capacity of our proposed geometric scattering features via their discriminaive power in graph data analysis tasks.

In Sec. 4.1, we describe extensive numerical experiments for graph classification problems in which our scattering coefficients are utilized in conjunction with several classifiers, namely, fully connected layers (FCL, illustrated in FIG2 ), support vector machine (SVM), and logistic regression.

We note that SVM classification over scattering features leads to state of the art results on social network data, as well as outperforming all feed-forward neural network methods in general.

Furthermore, for biochemistry data (where graphs represent molecule structures), FCL classification over scattering features outperforms all other feed-forward neural networks, even though we only train the fully connected layers.

Finally, to assess the scattering feature space for data representation and exploration, in Sec. 4.2 we examine its qualities when analyzing biochemistry data, with emphasis on enzyme graphs.

We show that geometric scattering enables graph embedding in a relatively low dimensional Euclidean space, while preserving insightful properties in the data.

Beyond establishing the capacity of our specific construction, these results also indicate the viability of graph scattering transforms in general, as universal feature extractors on graph data, and complement the stability results established in BID49 and BID17 .

DISPLAYFORM0 (a) Representative zeroth-, first-, and second-order cascades of the geometric scattering transform for an input graph signal x.

The presented cascades, indexed by j, j , q, are collected together to form the set of scattering coefficients Sx defined in eqs. (3-5).

DISPLAYFORM1 A d ja c e n c y m a tr ix : DISPLAYFORM2 Si gn al ve ct or : DISPLAYFORM3 Diffusion wavelets: DISPLAYFORM4 Fully connected layers: DISPLAYFORM5

We give a brief comparison of geometric scattering with other graph ConvNets, with particular interest in isolating the key principles for building accurate graph ConvNet classifiers.

We begin by remarking that like several other successful graph neural networks, the graph scattering transform is covariant or equivariant to vertex permutations (i.e., commutes with them) until the final features are extracted.

This idea has been discussed in depth in various articles, including BID25 , so we limit the discussion to observing that the geometric scattering transform thus propagates nearly all of the information in x through the multiple wavelet and absolute value layers, since only the absolute value operation removes information on x. As in BID44 , we aggregate covariant responses via multiple summary statistics (i.e., moments), which are referred to there as a capsule.

In the scattering context, at least, this idea is in fact not new and has been previously used in the Euclidean setting for the regression of quantum mechanical energies in BID16 BID42 and texture synthesis in BID8 .

We also point out that, unlike many deep learning classifiers (graph included), a graph scattering transform extracts invariant statistics at each layer/order.

These intermediate layer statistics, while necessarily losing some information in x (and hence G), provide important coarse geometric invariants that eliminate needless complexity in subsequent classification or regression.

Furthermore, such layer by layer statistics have proven useful in characterizing signals of other types (e.g., texture synthesis in BID19 .A graph wavelet transform ?? (J) x decomposes the geometry of G through the lens of x, along different scales.

Graph ConvNet algorithms also obtain multiscale representations of G, but several works, including BID1 and , propagate information via a random walk.

While random walk operators like P t act at different scales on the graph G, per the analysis in Sec. 2 we see that P t for any t will be dominated by the low frequency responses of x. While subsequent nonlinearities may be able to recover this high frequency information, the resulting transform will most likely be unstable due to the suppression and then attempted recovery of the high frequency content of x. Alternatively, features derived from P t x may lose the high frequency responses of x, which are useful in distinguishing similar graphs.

The graph wavelet coefficients ?? (J) x, on the other hand, respond most strongly within bands of nearly non-overlapping frequencies, each with a center frequency k j that depends on ?? j .Finally, graph labels are often complex functions of both local and global subgraph structure within G. While graph ConvNets are adept at learning local structure within G, as detailed in BID44 they require many layers to obtain features that aggregate macroscopic patterns in the graph.

This is due in large part to the use of fixed size filters, which often only incorporate information from the neighbors of any individual vertex.

The training of such networks is difficult due to the limited size of many graph classification databases (see Table 4 in Appendix D).

Geometric scattering transforms have two advantages in this regard: (a) the wavelet filters are designed; and (b) they are multiscale, thus incorporating macroscopic graph patterns in every layer/order.

To evaluate the proposed geometric scattering features, we test their effectiveness for graph classification on thirteen datasets commonly used for this task.

Out of these, seven datasets contain biochemistry graphs that describe molecular structures of chemical compounds, as described in the following works that introduced them: NCI1 and NCI109, BID45 BID14 .

In these cases, each graph has several associated vertex features x that represent chemical properties of atoms in the molecule, and the classification is aimed to characterize compound properties (e.g., protein types).

The other six datasets, which are introduced in BID46 , contain social network data extracted from scientific collaborations (COLLAB), movie collaborations (IMDB-B & IMDB-M), and Reddit discussion threads (REDDIT-B, REDDIT-5K, REDDIT-12K).

In these cases there are no inherent graph signals in the data, and therefore we compute general node characteristics (e.g., degree, eccentricity, and clustering coefficient) over them, as is considered standard practice in relevant literature (see, for example, BID44 In all cases, we iterate over all graphs in the database and for each one we associate graph-wide features by (1) computing the scattering features of each of the available graph signals (provided or computed), and (2) concatenating the features of all such signals.

Then, the full scattering feature vectors of these graphs are passed to a classifier, which is trained from input labels, in order to infer the class for each graph.

We consider three classifiers here: neural network with two/three fully connected hidden layers (FCL), SVM with RBF kernel, or logistic regression.

We note that the scattering features (computed as described in Sec. 3) are based on either normalized or unnormalized moments over the entire graph.

Here we used unnormalized moments for FCL, and normalized ones for other classifiers, but the difference is subtle and similar results can be achieved for the other combinations.

Finally, we also note that all technical design choices for configuring our geometric scattering or the classifiers were done as part of the cross validation described in Appendix E.We evaluate the classification results of our three geometric scattering classification (GSC) settings using ten-fold cross validation (as explained in Appendix E) and compare them to 14 prominent methods for graph classification.

Out of these, six are graph kernel methods, namely: WeisfeilerLehman graph kernels (WL, BID37 , propagation kernel (PK, BID31 , Graphlet kernels BID36 , Random walks (RW, BID18 , deep graph kernels (DGK, BID46 , and Weisfeiler-Lehman optimal assignment kernels (WL-OA, BID26 .

Seven other methods are recent geometric feed forward deep learning algorithms, namely: deep graph convolutional neural network (DGCNN, Zhang et al., 2018) , Graph2vec BID30 , 2D convolutional neural networks (2DCNN, BID42 , covariant compositional networks (CCN, BID24 , Patchy-san (PSCN, BID32 , with k = 10), diffusion convolutional neural networks (DCNN, BID1 , and graph capsule convolutional neural networks (GCAPS-CNN, BID44 .

Finally, one method is the recently introduced recurrent neural network autoencoder for graphs (S2S-N2N-PP, BID41 .

Following the standard format of reported classification performances for these methods (per their respective references, see also Appendix A), our results are reported in the form of average accuracy ?? standard deviation (in percentages) over the ten crossvalidation folds.

We remark here that many of them are not reported for all datasets, and hence, we mark N/A when appropriate.

For brevity, the comparison is reported here in FIG4 in summarized form, as explained below, and in full in Appendix A.Since the scattering transform is independent of training labels, it provides universal graph features that might not be specifically optimal in each individual dataset, but overall provide stable classification results.

Further, careful examination of the results of previous methods (feed forward algorithms in particular) shows that while some may excel in specific cases, none of them achieves the best results in all reported datasets.

Therefore, to compare the overall classification quality of our GSC methods with related methods, we consider average accuracy aggregated over all datasets, and within each field (i.e., biochemistry and social networks) in the following way.

First, out of the thirteen datasets, classification results on four datasets (NCI109, ENZYMES, IMDB-M, REDDIT-12K) are reported significantly less frequently than the others, and therefore we discard them and use the remaining nine for the aggregation.

Next, to address reported values versus N/A ones, we set an inclusion criterion of 75% reported datasets for each method.

This translates into at most one N/A in each individual field, and at most two N/A overall.

For each method that qualifies for this inclusion criterion, we compute its average accuracy over reported values (ignoring N/A ones) within each field and over all datasets; this results in up to three reported values for each method.

The aggregated results of our GSC and 13 of the compared methods appears in FIG4 .

These results show that GSC (with SVM) outperforms all other methods on social network data, and in fact as shown Appendinx B, it achieves state of the art results on two datasets of this type.

Additionally, the aggregated results shows that our GSC approach (with FCL or SVM) outperforms all other feed forward methods both on biochemsitry data and overall in terms of universal average accuracy 2 .

The CCN method is omitted from these aggregated results, as its results in BID24 are only reported on four biochemistry datasets.

For completeness, detailed comparison of GSC with this method, which appears in FIG4 , shows that our method outperforms it on two datasets while CCN outperforms GSC on the other two.

Geometric scattering essentially provides a task independent representation of graphs in a Euclidean feature space.

Therefore, it is not limited to supervised learning applications, and can be also utilized for exploratory graph-data analysis, as we demonstrate in this section.

We focus our discussion on biochemistry data, and in particular on the ENZYMES dataset.

Here, geometric scattering features can be considered as providing "signature" vectors for individual enzymes, which can be used to explore interactions between the six top level enzyme classes, labelled by their Enzyme Commission (EC) numbers BID2 .

In order to emphasize the properties of scattering-based feature extraction, rather than downstream processing, we mostly limit our analysis of the scattering feature space to linear operations such as principal component analysis (PCA).We start by considering the viability of scattering-based embedding for dimensionality reduction of graph data.

To this end, we applied PCA to our scattering coefficients (computed from unnormalized moments), while choosing the number of principal components to capture 90% explained variance.

In the ENZYMES case, this yields a 16 dimensional subspace of the full scattering features space.

While the Euclidean notion of dimensionality is not naturally available in the original dataset, we note that graphs in it have, on average, 124.2 edges, 29.8 vertices, and 3 features per vertex, and therefore the effective embedding of the data into R 16 indeed provides a significant dimensionality reduction.

Next, to verify the resulting PCA subspace still captures sufficient discriminative information with respect to classes in the data, we compare SVM classification on the resulting low dimensional vectors to the the full feature space; indeed, projection on the PCA subspace results in only a small drop in accuracy from 56.85 ?? 4.97 (full) to 49.83 ?? 5.40 (PCA).

Finally, we also consider the dimensionality of each individual class (with PCA and > 90% exp.

variance) in the scattering feature space, as we expect scattering to reduce the variability in each class w.r.t.

the full feature space.

In the ENZYMES case, individual classes have PCA dimensionality ranging between 6 and 10, which is indeed significantly lower than the 16 dimensions of the entire PCA space.

Appendix C summarizes these findings, and repeats the described procedure for two addi- tional biochemistry datasets (from BID45 to verify that these are not unique to the specific ENZYMES dataset, but rather indicate a more general trend for geometric scattering feature spaces.

To further explore the scattering feature space, we now use it to infer relations between EC classes.

First, for each enzyme e, with scattering feature vector v e (i.e., with Sx for all vertex features x), we compute its distance from class EC-j, with PCA subspace C j , as the projection distance: dist(e, EC-j) = v e ??? proj Sj v e .

Then, for each enzyme class EC-i, we compute the mean distance of enzymes in it from the subspace of each EC-j class as D(i, j) = mean{dist(e, EC-j) : e ??? EC-i}. Appendix C summarizes these distances, as well as the proportion of points from each class that have their true EC as their nearest (or second nearest) subspace in the scattering feature space.

In general, 48% of enzymes select their true EC as the nearest subspace (with additional 19% as second nearest), but these proportions vary between individual EC classes.

Finally, we use these scatteringbased distances to infer EC exchange preferences during enzyme evolution, which are presented in FIG5 and validated with respect to established preferences observed and reported in BID12 .

We note that the result there is observed independently from the ENZYMES dataset.

In particular, the portion of enzymes considered from each EC is different between these data, since BID3 took special care to ensure each EC class in ENZYMES has exactly 100 enzymes in it.

However, we notice that in fact the portion of enzymes (in each EC) that choose the wrong EC as their nearest subspace, which can be considered as EC "incoherence" in the scattering feature space, correlates well with the proportion of evolutionary exchanges generally observed for each EC in BID12 , and therefore we use these as EC weights in FIG5 .

Our results in FIG5 demonstrate that scattering features are sufficiently rich to capture relations between enzyme classes, and indicate that geometric scattering has the capacity to uncover descriptive and exploratory insights in graph data analysis, beyond the supervised graph classification from Sec 4.1.

We presented the geometric scattering transform as a deep filter bank for feature extraction on graphs.

This transform generalizes the scattering transform, and augments the theoretical foundations of geometric deep learning.

Further, our evaluation results on graph classification and data exploration show the potential of the produced scattering features to serve as universal representations of graphs.

Indeed, classification with these features with relatively simple classifier models reaches high accuracy results on most commonly used graph classification datasets, and outperforms both traditional and recent deep learning feed forward methods in terms of average classification accuracy over multiple datasets.

We note that this might be partially due to the scarcity of labeled big data in this field, compared to more traditional ones (e.g., image or audio classification).

However, this trend also correlates with empirical results for the classic scattering transform, which excels in cases with low data availability.

Finally, the geometric scattering features provide a new way for computing and considering global graph representations, independent of specific learning tasks.

Therefore, they raise the possibility of embedding entire graphs in Euclidean space and computing meaningful distances between graphs with them, which can be used for both supervised and unsupervised learning, as well as exploratory analysis of graph-structured data.

APPENDIX A FULL COMPARISON TABLE DISPLAYFORM0 DISPLAYFORM1

The details of the datasets used in this work are as follows (see the main text in Sec. 3 for references):NCI1 contains 4,110 chemical compounds as graphs, with 37 node features.

Each compound is labeled according to is activity against non-small cell lung cancer and ovarian cancer cell lines, and these labels serve as classification goal on this data.

NCI109 is similar to NCI1, but with 4,127 chemical compounds and 38 node features.

MUTAG consists of 188 mutagenic aromatic and heteroaromatic nitro compounds (as graphs) with 7 node features.

The classification here is binary (i.e., two classes), based on whether or not a compound has a mutagenic effect on bacterium.

PTC is a dataset of 344 chemical compounds (as graphs) with nineteen node features that are divided into two classes depending on whether they are carcinogenic in rats.

PROTEINS dataset contains 1,113 proteins (as graphs) with three node features, where the goal of the classification is to predict whether the protein is enzyme or not.

D&D dataset contains 1,178 protein structures (as graphs) that, similar to the previous one, are classified as enzymes or non-enzymes.

ENZYMES is a dataset of 600 protein structures (as graphs) with three node features.

These proteins are divided into six classes of enzymes (labelled by enzyme commission numbers) for classification.

COLLAB is a scientific collaboration dataset contains 5K graphs.

The classification goal here is to predict whether the graph belongs to a subfield of Physics.

IMDB-B is a movie collaboration dataset with contains 1K graphs.

The graphs are generated on two genres: Action and Romance, the classification goal is to predict the correct genre for each graph.

IMDB-M is similar to IMDB-B, but with 1.5K graphs & 3 genres: Comedy, Romance, and Sci-Fi.

REDDIT-B is a dataset with 2K graphs, where each graph corresponds to an online discussion thread.

The classification goal is to predict whether the graph belongs to a Q&A-based community or discussion-based community.

REDDIT-5K consists of 5K threads (as graphs) from five different subreddits.

The classification goal is to predict the corresponding subreddit for each thread.

REDDIT-12K is similar to REDDIT-5k, but with 11,929 graphs from 12 different subreddits.

Table 4 summarizes the size of available graph data (i.e., number of graphs, and both max & mean number of vertices within graphs) in these datasets, as previously reported in the literature.

Graph signals for social network data: None of the social network datasets has ready-to-use node features.

Therefore, in the case of COLLAB, IMDB-B, and IMDB-M, we use the eccentricity, degree, and clustering coefficients for each vertex as characteristic graph signals.

In the case of REDDIT-B, REDDIT-5K and REDDIT-12K, on the other hand, we only use degree and clustering coefficient, due to presence of disconnected graphs in these datasets.

Software & hardware environment: Geometric scattering and related classification code were implemented in Python with TensorFlow.

All experiments were performed on HPC environment using an intel16-k80 cluster, with a job requesting one node with four processors and two Nvidia Tesla k80 GPUs.

@highlight

We present a new feed forward graph ConvNet based on generalizing the wavelet scattering transform of Mallat, and demonstrate its utility in graph classification and data exploration tasks.