Nodes residing in different parts of a graph can have similar structural roles within their local network topology.

The identification of such roles provides key insight into the organization of networks and can also be used to inform machine learning on graphs.

However, learning structural representations of nodes is a challenging unsupervised-learning task, which typically involves manually specifying and tailoring topological features for each node.

Here we develop GraphWave, a method that represents each node’s local network neighborhood via a low-dimensional embedding by leveraging spectral graph wavelet diffusion patterns.

We prove that nodes with similar local network neighborhoods will have similar GraphWave embeddings even though these nodes may reside in very different parts of the network.

Our method scales linearly with the number of edges and does not require any hand-tailoring of topological features.

We evaluate performance on both synthetic and real-world datasets, obtaining improvements of up to 71% over state-of-the-art baselines.

Structural role discovery in graphs focuses on identifying nodes which have topologically similar local neighborhoods (i.e., similar local structural roles) while residing in potentially distant areas of the network FIG0 ).

Such alternative definition of node similarity is very different than more traditional notions BID21 BID7 BID30 BID19 BID15 BID8 BID6 , which all assume some notion of "smoothness" over the graph and thus consider nodes residing in close network proximity to be similar.

Such structural role information about the nodes can be used for a variety of tasks, including as input to machine learning problems, or even to identify key nodes in a system (principal "influencers" in a social network, critical hubs in contagion graphs, etc.).When structural roles of nodes are defined over a discrete space, they correspond to different topologies of local network neighborhoods (e.g., edge of a chain, center of a star, a bridge between two clusters).

However, such discrete roles must be pre-defined, requiring domain expertise and manual inspection of the graph structure.

A more powerful and robust method for identifying structural similarity involves learning a continuous vector-valued structural signature χ a of each node a in an unsupervised way.

This motivates a natural definition of structural similarity in terms of closeness of topological signatures: For any > 0, nodes a and b are defined to be -structurally similar with respect to a given distance if: dist(χ a , χ b ) ≤ .

Thus, a robust structural similarity metric must introduce both an appropriate signature and an adequate distance metric.

While several methods have been proposed for structural role discovery in graphs, existing approaches are extremely sensitive to small perturbations in the topology and typically lack one or more desirable properties.

They often require manually hand-labeling topological features BID12 , rely on non-scalable heuristics BID22 , and/or return a single similarity score instead of a multidimensional structural signature BID13 .Here we address the problem of structure learning on graphs by developing GRAPHWAVE.

Building upon techniques from graph signal processing BID4 BID10 BID26 , our approach learns a structural embedding for each node based on the diffusion of a spectral graph wavelet centered at that node.

Intuitively, each node propagates a unit of energy over the graph and characterizes its neighboring topology based on the response of the network to While raw spectral graph wavelet signatures/coefficients Ψ of a and b might be very different, we treat them as probability distributions and show that the coefficient distributions are indeed similar.this probe.

In contrast to prior work that characterizes the wavelet diffusion as a function of the wavelet scaling parameter, we study how the wavelet diffuses through the network at a given scale as a function of the initial source node.

We prove that the coefficients of this wavelet directly relate to graph topological properties.

Hence, these coefficients contain all the necessary information to recover structurally similar nodes, without requiring the hand-labeling of features.

However, the wavelets are, by design, localized on the graph.

Therefore to compare structural signatures for nodes that are far away from each other, typical graph signal processing methods (using metrics like correlation between wavelets or 2 distance) cannot be used without specifying an exact one-to-one mapping between nodes for every pairwise comparison, a computationally intractable task.

To overcome this challenge, we propose a novel way of treating the wavelets as probability distributions over the graph.

This way the structural information is contained in how the diffusion spreads over the network rather than where it spreads.

In order to provide vector-valued signatures which can then be used as input to any machine learning algorithm, we embed these wavelet distributions using the empirical characteristic function BID18 .

The advantage of empirical characteristic functions is that they capture all the moments of a given distribution.

This allows GRAPHWAVE to be robust to small perturbations in the local edge structure, as we prove mathematically.

Computational complexity of GRAPHWAVE is linear in the number of edges, thus allowing it to scale to large (sparse) networks.

Finally, we compare GRAPHWAVE to several state-of-the-art baselines on both real and synthetic datasets, obtaining improvements of up to 71% and demonstrating how our approach is a useful tool for characterizing structural signatures in graphs.

Summary of contributions.

The main contributions of our paper are as follows:• We develop a novel use of spectral graph wavelets by treating them as probability distributions and characterizing the distributions using empirical characteristic functions.• We leverage these insights to develop a scalable method (GRAPHWAVE) for learning node embeddings based on structural similarity in graphs.• We prove that GRAPHWAVE accurately recovers structurally similar nodes.

Further related work.

Prior work on discovering nodes with similar structural roles has typically relied on explicit featurization of nodes.

These methods generate an exhaustive listing of each node's local topological properties (e.g., node degree, number of triangles it participates in, number of k-cliques, its PageRank score) before computing node similarities based on such heuristic representations.

A notable example of such approaches is RolX BID12 , which aims to recover a soft-clustering of nodes into a predetermined number of K distinct roles using recursive feature extraction BID11 .

Similarly, struc2vec BID22 uses a heuristic to construct a multilayered graph based on topological metrics and simulates random walks on the graph to capture structural information.

In contrast, our approach does not rely on heuristics (we mathematically prove its efficacy) and does not require explicit manual feature engineering or hand-tuning of parameters.

Another line of related work are graph diffusion kernels BID4 which have been utilized for various graph modeling purposes BID17 BID2 BID24 BID29 .

However, to the best of our knowledge, our paper is the first to apply graph diffusion kernels for determining structural roles in graphs.

Kernels have been shown to efficiently capture geometrical properties and have been successfully used for shape detection in the image processing community BID28 BID20 BID0 .

However, in contrast to shape-matching problems, GRAPHWAVE considers these kernels as probability distributions over real-world graphs.

This is because the graphs that we consider are highly irregular (as opposed to the Euclidean and manifold graphs).

Therefore, traditional wavelet methods, which typically analyze node diffusions across specific nodes that occur in regular and predictable patterns, do not apply.

Instead, by treating wavelets as distributions, GRAPHWAVE characterizes the shape of the diffusion, rather than the specific nodes where the diffusion occurs.

This key insight allows us to uncover structural signatures and to discover structurally similar nodes.

Given an undirected connected graph G = (V, E) with N nodes V = {a 1 , . . . , a N }, edges E, an adjacency matrix A (binary or weighted), and a degree matrix D ii = j A ij , we consider the problem of learning, for every node a i , a structural signature representing a i 's position in a continuous multidimensional space of structural roles.

We frame this as an unsupervised learning problem based on spectral graph wavelets BID10 and develop an approach called GRAPHWAVE that provides mathematical guarantees on the optimality of learned structural signatures.

In this section, we provide background on a spectral graph wavelet-based model BID10 BID26 ) that we will use in the rest of the paper.

Let U be the eigenvector decomposition of the unnormalized graph Laplacian L = D − A and let DISPLAYFORM0 denote the eigenvalues of L. Let g s be a filter kernel with scaling parameter s.

For simplicity, we use the heat kernel g s (λ) = e −λs throughout this paper, but our results apply to any low-pass filter kernel BID27 .

For now, we assume that s is given; we develop a method for selecting an appropriate value of s in Appendix C.Graph signal processing BID10 BID26 defines the spectral graph wavelet associated with g s as the signal resulting from the modulation in the spectral domain of a Dirac signal centered around node a. The spectral graph wavelet Ψ a is given by an N -dimensional vector: DISPLAYFORM1 where δ a = 1(a) is the one-hot vector for node a. For notational simplicity, we drop the explicit dependency of spectral graph wavelet Ψ a on s. The m-th wavelet coefficient of this column vector is thus given by Ψ ma = N l=1 g s (λ l )U ml U al .

In spectral graph wavelets, the kernel g s modulates the eigenspectrum such that the resulting signal is typically localized on the graph and in the spectral domain BID26 .

Spectral graph wavelets are based on an analogy between temporal frequencies of a signal and the Laplacian's eigenvalues.

Eigenvectors associated with smaller eigenvalues carry slow varying signal, encouraging nodes that are geographically close in the graph to share similar values.

In contrast, eigenvectors associated with larger eigenvalues carry faster-varying signal across edges.

The low-pass filter kernel g s can thus be seen as a modulation operator that discounts higher eigenvalues and enforces smoothness in the signal variation on the graph.

First we describe the GRAPHWAVE algorithm (Alg.

1) and then analyze it in the next section.

For every node a, GRAPHWAVE returns a 2d-dimensional vector χ a representing its structural signature, where nodes with structurally similar local network neighborhoods will have similar signatures.

We first apply spectral graph wavelets to obtain a diffusion pattern for every node (Line 3), which we gather in a matrix Ψ. Here, Ψ is a N × N matrix, where a-th column vector is the spectral graph wavelet for a heat kernel centered at node a. In contrast to prior work that studies wavelet coefficients as a function of the scaling parameter s, we study them as a function of the network (i.e., how the coefficients vary across the local network neighborhood around the node a).

In particular, coefficients in each wavelet are identified with the nodes and Ψ ma represents the amount of energy that node a Algorithm 1 GRAPHWAVE algorithm for learning structural signatures.

DISPLAYFORM0 for a ∈ V do 7:Append Re(φ a (t)) and Im(φ a (t)) to χ a has received from node m. As we will later show nodes a and b with similar network neighborhoods have similar spectral wavelet coefficients Ψ (assuming that we know how to solve the "isomorphism" problem and find the explicit one-to-one mapping of the nodes from a's neighborhood to the nodes of the b's neighborhood).

To resolve the node mapping problem GRAPHWAVE treats the wavelet coefficients as a probability distribution and characterizes the distribution via empirical characteristic functions.

This is the key insight that makes it possible for GRAPHWAVE to learn nodes' structural signatures via spectral graph wavelets.

More precisely, we embed spectral graph wavelet coefficient distributions into 2d-dimensional space (Line 4-7) by calculating the characteristic function for each node's coefficients Ψ a and sample it at d evenly spaced points.

The characteristic function of a probability distribution X is defined as: BID18 .

The function φ X (t) fully characterizes the distribution of X because it captures information about all the moments of probability distribution X BID18 .

For a given node a and scale s, the empirical characteristic function of Ψ a is defined as: DISPLAYFORM1 DISPLAYFORM2 Finally, structural signature χ a of node a is obtained by sampling the 2-dimensional parametric function (Eq. (2)) at d evenly spaced points t 1 , . . .

t d and concatenating the values: DISPLAYFORM3 Note that we sample/evaluate the empirical characteristic function φ a (t) at d points and this creates a structural signature of size 2d.

This means that the dimensionality of the structural signature is independent of the graph size.

Furthermore, nodes from different graphs can be embedded into the same space and their structural roles can be compared across different graphs.

Distance between structural signatures.

The final output of GRAPHWAVE is a structural signature χ a for each node a in the graph.

We can explore distances between the signatures through the use of the 2 distance on χ a .

The structural distance between nodes a and b is then defined as: DISPLAYFORM4 By definition of the characteristic function, this technique amounts to comparing moments of different orders defined on wavelet coefficient distributions.

Scaling parameter.

The scaling parameter s determines the radius of network neighborhood around each node a (Tremblay et al. FORMULA1 ; BID10 .

A small value of s determines node signatures based on similarity of nodes' immediate neighborhoods.

In contrast, a larger value of s allows the diffusion process to spread farther in the network, resulting in signatures based on neighborhoods with greater radii.

GRAPHWAVE can also integrate information across different radii of neighborhoods by jointly considering many different values of s. This is achieved by concatenating J representations χ (sj ) a , each associated with a scale s j , where s j ∈ [s min , s max ].

We provide a theoretically justified method for finding an appropriate range s min and s max in Appendix C. In this multiscale version of GRAPHWAVE, the final aggregated structural signature for node a is a vector χ a ∈ R 2dJ with the following form: DISPLAYFORM5 We use Chebyshev polynomials BID25 to compute Line 3 in Algorithm 1.

As in BID5 , each power of the Laplacian has a computational cost of O(|E|), yielding an overall complexity of O(K|E|), where K denotes the order Chebyshev polynomial approximation.

The overall complexity of GRAPHWAVE is linear in the number of edges, which allows GRAPHWAVE to scale to large sparse networks.

In this section, we provide theoretical motivation for our spectral graph wavelet-based model BID26 .

First we analytically show that spectral graph wavelet coefficients characterize the topological structure of local network neighborhoods (Section 3.1).

Then we show that structurally equivalent/similar nodes have near-identical/similar signatures (Sections 3.2 and 3.3), thereby providing a mathematical guarantee on the optimality of GRAPHWAVE.

We start by establishing the relationship between the spectral graph wavelet of a given node a and the topological properties of local network neighborhood centered at a. In particular, we prove that a wavelet coefficient Ψ ma provides a measure of network connectivity between nodes a and m.

We use the fact that the spectrum of the graph Laplacian is discrete and contained in the compact set [0, λ N ].

It then follows from the Stone-Weierstrass theorem that the restriction of kernel g s to the interval [0, λ N ] can be approximated by a polynomial.

This polynomial approximation, denoted as P , is tight and its error can be uniformly bounded.

Formally, this means: DISPLAYFORM0 where K is the order of polynomial approximation, α k are coefficients of the polynomial, and r(λ) = g s (λ) − P (λ) is the residual.

We can now express the spectral graph wavelet for node a in terms of the polynomial approximation as: DISPLAYFORM1 We note that Ψ a is a function of L k = (D − A) k and thus can be interpreted using graph theory.

In particular, it contains terms of the form D k (capturing the degree), A k (capturing the number of k-length paths that node a participates in), and terms containing both A and D, which denote paths of length up to k going from node a to every other node m.

Using the Cauchy-Schwartz's inequality and the facts that U is unitary and r(λ) is uniformly bounded (Eq. (4)), we can bound the second term on the right-hand side of Eq. (5) by: DISPLAYFORM2 As a consequence, each wavelet Ψ a can be approximated by a K-th order polynomial that captures information about the K-hop neighborhood of node a. The analysis of Eq. FORMULA9 , where we show that the second term is limited by , indicates that spectral graph wavelets are predominately governed by topological features (specifically, degrees, cycles and paths) according to the specified heat kernel.

The wavelets thus contain the information necessary to generate structural signatures of nodes.

Let us consider nodes a and b whose K-hop neighborhoods are identical (where K is an integer less than the diameter of the graph), meaning that nodes a and b are structurally equivalent.

We now show that a and b have -structurally similar signatures in GRAPHWAVE.First, we use the Taylor expansion to obtain an explicit K-th order polynomial approximation of g s as: DISPLAYFORM0 Then, for each eigenvalue λ, we use the Taylor-Lagrange equality to ensure the existence of c λ ∈ [0, s] such that: DISPLAYFORM1 If we take any s such that it satisfies: s ≤ ((K + 1)! )) 1/(K+1) /λ 2 , then the absolute residual |r(λ)| in Eq. (7) can be bounded by for each eigenvalue λ.

Here, is a parameter that we can specify depending on how close we want the signatures of structurally equivalent nodes to be (note that smaller values of the scale s lead to smaller values of and thus tighter bounds).Because a and b are structurally equivalent, there exists a one-to-one mapping π from the Khop neighborhood of a (i.e., N K (a)) to the K-hop neighborhood of b (i.e., N K (b)), such that: DISPLAYFORM2 .

We extend the mapping π to the whole graph G by randomly mapping the remaining nodes.

Following Eq. (5), we write the difference between each pair of mapped coefficients Ψ ma and Ψ π(m)b in terms of the K-th order approximation of the graph Laplacian: DISPLAYFORM3 Here, we analyze the first term on the second line in Eq. (8).

Since the K-hop neighborhoods around a and b are identical and by the localization properties of the k-th power of the Laplacian (k-length paths, Section 3.1), the following holds: DISPLAYFORM4 meaning that this term cancels out in Eq. (8) .

To analyze the second and third terms on the second line of Eq. (8) , we use bound for the residual term in the spectral graph wavelet (Eq. FORMULA10 ) to uniformly bound entries in matrix U r(Λ)U T by .

Therefore, each wavelet coefficient in Ψ a is within 2 of its corresponding wavelet coefficient in Ψ b , i.e., |Ψ ma − Ψ π(m)b | ≤ 2 .

As a result, because similarity in distributions translates to similarity in the resulting characteristic functions (Lévy's continuity theorem), then assuming the appropriate selection of scale, structurally equivalent nodes have -structurally similar signatures.

We now analyze structurally similar nodes, or nodes whose K-hop neighborhoods are identical up to a small perturbation of the edges.

We show that such nodes have similar GRAPHWAVE signatures.

Let N K (a) denote a perturbed K-hop neighborhood of node a obtained by rewiring edges in the original K-hop neighborhood N K (a).

We denote byL the graph Laplacian associated with that perturbation.

We next show that when perturbation of a node neighborhood is small, the changes in the wavelet coefficients for that node are small as well.

Formally, assuming a small perturbation of the graph structure (i.e., sup ||L k −L k || F ≤ , for all k ≤ K), we use K-th order Taylor expansion of kernel g s to express the wavelet coefficients in the perturbed graph as: DISPLAYFORM0 We then use the Weyl's theorem BID3 to relate perturbations in the graph structure to the change in the eigenvalues of the graph Laplacian.

In particular, a small perturbation of the graph yields small perturbations of the eigenvalues.

That is, for eachλ, r(λ) is close its original value r(λ): r(λ) = r(λ) + o( ) ≤ C , where C is a constant.

Taking everything together, we get: DISPLAYFORM1 indicating that structurally similar nodes have similar signatures in GRAPHWAVE.

Baselines.

We compare our GRAPHWAVE method against two state-of-the-art baselines, struc2vec BID22 and RolX BID12 .

We note that RolX requires the number of desired structural classes as input, whereas the two other methods learn embeddings that capture a continuous spectrum of roles rather than discrete classes.

We thus use RolX as an oracle estimator, providing it with the correct number of classes 1 .

We also note that homophily-based methods BID15 ; BID8 , etc.) are unable to recover structural similarities.

We consider a barbell graph consisting of two dense cliques connected by a long chain FIG1 .

We run GRAPHWAVE, RolX, and struc2vec and plot a 2D PCA representation of learned structural signatures in FIG1 -D.GRAPHWAVE correctly learns identical representations for structurally equivalent nodes, providing empirical evidence for our theoretical result in Section 3.2.

This can be seen by structurally equivalent nodes in FIG1 (nodes of the same color) having identical projections in the PCA plot ( FIG1 ).

In contrast, both RolX and struc2vec fail to recover the exact structural equivalences.

All three methods correctly group the clique nodes (purple) together.

However, only GRAPHWAVE correctly differentiates between nodes connecting the two dense cliques in the barbell graph, providing empirical evidence for our theoretical result in Section 3.3.

GRAPHWAVE represents those nodes in a gradient-like pattern that captures the spectrum of structural roles of those nodes ( FIG1 ).

Graphs.

We next consider four types of synthetic graphs where the structural role of each node is known and used as ground truth information to evaluate performance.

The graphs are given by basic shapes of one of different types ("house", "fan", "star") that are regularly placed along a cycle (Table 1 and FIG2 ).

In the "varied" setup, we mix the three basic shapes when placing them along a cycle, thus generating synthetic graphs with richer and more complex structural role patterns.

Additional graphs are generated by placing these shapes irregularly along the cycle followed by adding a number of edges uniformly at random.

In our experiments, we set this number to be around 5% of the edges in the original structure.

This setup is designed to assess the robustness of the methods to data perturbations ("house perturbed", "varied perturbed").Experimental setup.

For each graph, we run RolX, struc2vec, and GRAPHWAVE to learn the signatures.

We choose to use a multiscale version of GRAPHWAVE where the scale was set as explained in Appendix C. We then use k-means to cluster the learned signatures and use three standard metrics to evaluate the clustering quality.(1) Cluster homogeneity is the conditional entropy of the ground-truth structural roles given the proposed clustering BID23 .(2) Cluster completeness BID23 evaluates whether nodes with the same structural role are in the same cluster.

(3) Silhouette score compares the mean intra-cluster distance to the mean between-cluster distance, assessing the density of the recovered clusters.

This score takes a value in [-1,1] (higher is better).Results.

GRAPHWAVE consistently outperforms struc2vec, yielding improvements for the homogeneity of up to 50%, and completeness up to 69% in the "varied" setting (Table 1) .

Both GRAPHWAVE and RolX achieved perfect performance in the noise-free "house" setting, however, GRAPHWAVE outperformed RolX by up to 4% (completeness) in the more complex "varied" setting.

We evaluated methods on graphs in the presence of noise ("perturbed" in Table 1 ): GRAPHWAVE outperformed RolX and struc2vec by 4% and 50% (homogeneity), respectively, providing empirical evidence for our analytical result that GRAPHWAVE is robust to noise in the edge structure.

The silhouette scores also show that the clusters recovered by GRAPHWAVE are denser and better separated than for the other methods.

As an example, we show a cycle graph with attached "house" shapes ( FIG2 .

We plot 2D PCA projections of GRAPHWAVE's signatures in FIG2 , confirming that GRAPHWAVE accurately distinguishes between nodes with distinct structural roles.

We also visualize the resulting characteristic functions (Eq. (2)) in FIG2 .

In general, their interpretation is as follows (Appendix D):• Nodes located in the periphery of the graph struggle to diffuse the signal over the graph, and thus span wavelets that are characterized by a smaller number of non-zero coefficients.

Characteristic functions of such nodes thus span a small loop-like 2D curve.• Nodes located in the core (dense region) of the graph tend to diffuse the signal farther away and reach farther nodes for the same value of t. Characteristic functions of such nodes thus have a farther projection on the x and y axis.

In FIG2 , different shapes of the characteristic functions capture different structural roles.

We note the visual proximity between the roles of the blue, light green and red nodes that these curves carry, as well as their clear difference with the core dark green and purple nodes.

Data and setup.

Nodes represent Enron employees and edges correspond to email communication between the employees BID16 ).

An employee has one of seven functions in the company (e.g., CEO, president, manager).

These functions provide ground-truth information about roles of the corresponding nodes in the network.

We use GRAPHWAVE to learn a structural signature for every Enron employee.

We then use these signatures to compute the average 2 2 distance between every two categories of employees.

Results.

GRAPHWAVE captures intricate organizational structure of Enron (Figure 4 ).

For example, CEOs and presidents are structurally distant from all other job titles.

This indicates their unique position in the email exchange graph, which can be explained by their local graph connectivity patterns standing out from the others.

Traders, on the other hand, appear very far from presidents Figure 5 : PCA projection of the learned airport structural signatures.

A: Air France, a major airline, and B: Ryanair, a low-cost fare airline.

Selected nodes are labeled using three-letter airports codes.and are closer to directors.

In contrast, struc2vec is less successful at revealing intricate relationships between the job titles, yielding an almost uniform distribution of distances between every class.

We assess the separation between "top" job titles (CEO and President) and lower levels in the job title hierarchy.

GRAPHWAVE achieves 28% better homogeneity and 139% better completeness than RolX. We also note that the variability within each cluster of struc2vec is higher than the average distance between clusters (dark green colors on the diagonal, and lighter colors on the off-diagonal).

Data and setup.

The airline graphs are taken from the list of airlines operating flights between European airports BID1 .

Each airline is represented with a graph, where nodes represent airports and links stand for direct flights between two airports.

Given an airline graph, we use GRAPHWAVE to learn structural signature of every airport.

We then create a visualization using PCA on the learned signatures to layout the graph on a two-dimensional structural space.

Results.

Figure 5 shows graph visualizations of two very different airlines.

Air France is a national French airline, whose graph has the so-called hub and spoke structure, because the airline is designed to provide an almost complete coverage of the airports in France.

We note that the signature of CDG (Charles De Gaulle, which is Air France's central airport) clearly stands out in this 2D projection, indicating its unique role in the network.

In contrast to Air France, Ryanair is a low-cost airline whose graph avoids the overly centralized structure.

Ryanair's network has near-continuous spectrum of structural roles that range from regional French airports (Lille (LIL), Brest Bretagne (BES)) all the way to London Stansted (STN) and Dublin (DUB).

These airlines have thus developed according to different structural and commercial constraints, which is clearly reflected in their visualizations.

We have developed a new method for learning structural signatures in graphs.

Our approach, GRAPHWAVE, uses spectral graph wavelets to generate a structural embedding for each node, which we accomplish by treating the wavelets as a distributions and evaluating the resulting characteristic functions.

Considering the wavelets as distributions instead of vectors is a key insight needed to capture structural similarity in graphs.

Our method provides mathematical guarantees on the optimality of learned structural signatures.

Using spectral graph theory, we prove that structurally equivalent/similar nodes have nearidentical/similar structural signatures in GRAPHWAVE.

Experiments on real and synthetic networks provide empirical evidence for our analytical results and yield large gains in performance over stateof-the-art baselines.

For future work, these signatures could be used for transfer learning, leveraging data from a well-explored region of the graph to infer knowledge about less-explored regions.

In the appendix, we use the same notation as in the main paper.

However, in the main paper, we dropped the explicit dependency of the heat kernel wavelet Ψ on the scaling parameter s. We note that we explicitly keep this dependency throughout the appendix as we study the relationship between heat kernel wavelets and the scaling parameter.

We here list five known properties of heat kernel wavelets that we later use to derive a method for automatic selection of the scaling parameter in Appendix C.P1.

The heat kernel wavelet matrix Ψ (s) is symmetric: DISPLAYFORM0 P2.

First eigenvector: By definition of the graph Laplacian L = D − A, the vector 1 is an eigenvector of L, corresponding to the smallest eigenvalue λ 1 = 0.

This means that: DISPLAYFORM1 ) where the last equality follows from the fact that eigenvectors in U are orthogonal.

P3.

Scaling inequality: Using the Cauchy Schwartz inequality, we get: DISPLAYFORM2 It thus follows that: DISPLAYFORM3 DISPLAYFORM4 where: DISPLAYFORM5

We prove three propositions about distributions generated by heat kernel wavelets that will be used in Appendix C. a is thus independent of the value of scaling parameter s and node a.

Proof.

We expand the mean of heat kernel wavelet Ψ (s) a using spectral graph wavelet definition in Eq. (1) and Property 2 (P2) from Appendix A: DISPLAYFORM0 Proposition 2.

The heat kernel wavelet coefficient Ψaa at the initial source node a is a monotonically decreasing function of scaling parameter s. Its value is bounded by: DISPLAYFORM1 Proof.

This follows directly using definition Ψ DISPLAYFORM2 Additionally, for any m = a, the wavelet coefficient DISPLAYFORM3 ma is non-negative and bounded.

Specifically, the wavelet coefficient can be written as: DISPLAYFORM4 It can be bounded by: DISPLAYFORM5 mm ) (Property 4 (P4)).

Proof.

We use the definition of variance to get: DISPLAYFORM6 We rewrite the sum on the far right hand-side using the symmetry of wavelet matrix Ψ (Property 1): DISPLAYFORM7 a ] is decreasing, since it is sum of functions, all of which decrease as s gets larger.

We here develop a method that automatically finds an appropriate range of values for the scaling parameter s in heat kernel g s , which we use in the multiscale version of GRAPHWAVE (Section 2.2)We find the appropriate range of values for s by specifying an interval bounded by s min and s max through the analysis of variance in heat kernel wavelets.

Intuitively, whether or not a given value for s is appropriate for structural signature learning depends on the relationship between the scaling parameter and the temporal aspects of heat equation.

In particular, small values of s allow little time for the heat to propagate, yielding diffusion distributions (i.e., heat kernel wavelet distributions) that are trivial in the sense that only a few coefficients have non-zero values and are thus unfit for comparison.

For larger values of s the network converges to a state in which all nodes have an identical temperature equal to 1/N (Property 4 (P4)), meaning that diffusion distributions are data-independent, hence non-informative.

Next we prove Propositions 4 and 5 to provide new insights into the variance and convergence rate of heat kernel wavelets.

We then use these results to select s min and s max .Proposition 4.

Given the scaling parameter s, the variance of off-diagonal coefficients in heat kernel wavelet Ψ a is proportional to: DISPLAYFORM0 Proof.

Let us denote the mean of off-diagonal coefficients in wavelet Ψ a by:μ DISPLAYFORM1 ma /(N − 1).

We use the fact that m =a Ψ (s) DISPLAYFORM2 aa , along with the definition of the variance, to obtain: DISPLAYFORM3 2 ).

a is large enough that the diffusion has had time to spread, while remaining sufficiently small to ensure that the diffusion is far from its converged state.

Proposition 5.

The convergence of heat kernel wavelet coefficient Ψ (s) am is bounded by: DISPLAYFORM4 Proof.

We use Property 5 from Appendix A and induction over s to complete this proof.

For a given s ≥ 0 we analyze: DISPLAYFORM5 and conclude that: DISPLAYFORM6 Given any s ∈ N, we use the induction principle to get: a is smooth increasing function of s, we can take the floor/ceiling of any non-integer s ≥ 0 and this proposition must hold.

We select s max such that wavelet coefficients are localized in the network.

To do so, we use Proposition 5 and bound ∆ a is above a given threshold η < 1.

Indeed, this ensures that DISPLAYFORM0 a has shrunk to at most η * 100 % of its initial value at s = 0, and yields a bound of the form: ∆ (s) a /∆ (0) a ≥ η.

The bound implies that: e −λs ≥ η, or s ≤ − log(η)/λ.

To find a middle ground between the two convergence scenarios, we take λ to be the geometric mean of λ 2 and λ N .

Indeed, as opposed to the arithmetic mean, the geometric mean maintains an equal weighting across the range [λ 2 , λ N ], and a change of % in λ 2 has the same effect as a change in % of λ N .

We thus select s max as: s max = − log(η)/ √ λ 2 λ N .

We select s min to ensure the adequate diffusion resolution.

In particular, we select a minimum value s min such that each wavelet has sufficient time to spread.

That is, ∆a /∆ (0) a ≤ γ.

As in the case of s max above, we obtain a bound of s ≥ − log(γ)/λ.

Hence, we set s min to: s min = − log(γ)/ √ λ 2 λ N .To cover an appropriate range of scales, we suggest setting η = 0.90 and γ = 0.99.

Here, we study the properties of characteristic functions in GRAPHWAVE.

Our goal is to provide intuition to understand the behavior of these functions and how their resulting 2D parametric curves reflect nodes' local topology (see FIG2 ).We begin by reviewing the definition of the characteristic function.

Definition 1.

The empirical characteristic function BID18 of wavelet Ψ In the phase plot, a given value of t thus yields the following set of coordinates:φ .By varying t, we get a characteric curve in this 2D plane.

Here, we note several properties of this curve:1.

Value at t = 0: ∀s, φ .Since all the wavelet coefficients are non-negative, the curve is thus directed counterclockwise.

<|TLDR|>

@highlight

We develop a method for learning structural signatures in networks based on the diffusion of spectral graph wavelets.

@highlight

Using spectral graph wavelet diffusion patterns of a node's local meighbothood to embed the node in a low-dimensional space

@highlight

The paper derived a way to compare nodes in graph based on wavelet analysis of graph laplacian. 