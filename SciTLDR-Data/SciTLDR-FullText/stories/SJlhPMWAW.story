Deep learning on graphs has become a popular research topic with many applications.

However, past work has concentrated on learning graph embedding tasks only, which is in contrast with advances in generative models for images and text.

Is it possible to transfer this progress to the domain of graphs?

We propose to sidestep hurdles associated with linearization of such discrete structures by having a decoder output a probabilistic fully-connected graph of a predefined maximum size directly at once.

Our method is formulated as a variational autoencoder.

We evaluate on the challenging task of conditional molecule generation.

Deep learning on graphs has very recently become a popular research topic, with useful applications across fields such as chemistry BID5 , medicine (Ktena et al.) , or computer vision (Simonovsky & Komodakis, 2017) .

Past work has concentrated on learning graph embedding tasks so far, i.e. encoding an input graph into a vector representation.

This is in stark contrast with fastpaced advances in generative models for images and text, which have seen massive rise in quality of generated samples.

Hence, it is an intriguing question how one can transfer this progress to the domain of graphs, i.e. their decoding from a vector representation.

Moreover, the desire for such a method has been mentioned in the past by BID7 .However, learning to generate graphs is a difficult problem for methods based on gradient optimization, as graphs are discrete structures.

Incremental construction involves discrete decisions, which are not differentiable.

Unlike sequence (text) generation, graphs can have arbitrary connectivity and there is no clear best way how to linearize their construction in a sequence of steps.

In this work, we propose to sidestep these hurdles by having the decoder output a probabilistic fully-connected graph of a predefined maximum size directly at once.

In a probabilistic graph, the existence of nodes and edges, as well as their attributes, are modeled as independent random variables.

The method is formulated in the framework of variational autoencoders (VAE) by BID12 .We demonstrate our method, coined GraphVAE, in cheminformatics on the task of molecule generation.

Molecular datasets are a challenging but convenient testbed for our generative model, as they easily allow for both qualitative and quantitative tests of decoded samples.

While our method is applicable for generating smaller graphs only and its performance leaves space for improvement, we believe our work is an important initial step towards powerful and efficient graph decoders.

Graph Decoders.

Graph generation has been largely unexplored in deep learning.

The closest work to ours is by BID11 , who incrementally constructs a probabilistic (multi)graph as a world representation according to a sequence of input sentences to answer a query.

While our model also outputs a probabilistic graph, we do not assume having a prescribed order of construction transformations available and we formulate the learning problem as an autoencoder.

BID31 learns to produce a scene graph from an input image.

They construct a graph from a set of object proposals, provide initial embeddings to each node and edge, and use message passing Figure 1 : Illustration of the proposed variational graph autoencoder in its conditional form.

Starting from a discrete attributed graph G = (A, E, F ) on n nodes (e.g. a representation of propylene oxide), stochastic graph encoder q φ (z|G) embeds the graph into continuous representation z. Given a point in the latent space, our novel graph decoder p θ (G|z) outputs a probabilistic fully-connected graph G = ( A, E, F ) on predefined k ≥ n nodes, from which discrete samples may be drawn.

The process can be conditioned on label y for controlled sampling at test time.

Reconstruction ability of the autoencoder is facilitated by approximate graph matching for aligning G with G.to obtain a consistent prediction.

In contrast, our method is a generative model which produces a probabilistic graph from a single opaque vector, without specifying the number of nodes or the structure explicitly.

Related work pre-dating deep learning includes random graphs BID4 BID0 , stochastic blockmodels (Snijders & Nowicki, 1997), or state transition matrix learning BID8 .Discrete Data Decoders.

Text is the most common discrete representation.

Generative models there are usually trained by teacher forcing BID30 , which avoids the need to backpropagate through output discretization by feeding the ground truth instead of the past sample at each step.

Recently, efforts have been made to overcome this problem.

Notably, computing a differentiable approximation using Gumbel distribution BID14 or bypassing the problem by learning a stochastic policy in reinforcement learning BID32 .

Our work also circumvents the non-differentiability problem, namely by formulating the loss on a probabilistic graph.

Molecule Decoders.

Generative models may become promising for de novo design of molecules fulfilling certain criteria by being able to search for them over a continuous embedding space BID21 .

With that in mind, we propose a conditional version of our model.

While molecules have an intuitive representation as graphs, the field has had to resort to textual representations with fixed syntax, e.g. so-called SMILES strings, to exploit recent progress made in text generation with RNNs BID21 BID23 BID7 .

As their syntax is brittle, many invalid strings tend to be generated, which has been recently addressed by BID15 by incorporating grammar rules into decoding.

While encouraging, their approach does not guarantee semantic (chemical) validity, similarly as our method.

We approach the task of graph generation by devising a neural network able to translate vectors in a continuous code space to graphs.

Our main idea is to output a probabilistic fully-connected graph and use a standard graph matching algorithm to align it to the ground truth.

The proposed method is formulated in the framework of variational autoencoders (VAE) by BID12 , although other forms of regularized autoencoders would be equally suitable BID19 BID17 .

We briefly recapitulate VAE below and continue with introducing our novel graph decoder together with an appropriate loss function.

Let G = (A, E, F ) be a graph specified with its adjacency matrix A, edge attribute tensor E, and node attribute matrix F .

We wish to learn an encoder and a decoder to map between the space of graphs G and their continuous embedding z ∈ R c , see Figure 1 .

In the probabilistic setting of a VAE, the encoder is defined by a variational posterior q φ (z|G) and the decoder by a generative distribution p θ (G|z), where φ and θ are learned parameters.

Furthermore, there is a prior distribution p(z) imposed on the latent code representation as a regularization; we use a simplistic isotropic Gaussian prior p(z) = N (0, I).

The whole model is trained by minimizing the upper bound on negative log-likelihood − log p θ (G) BID12 : DISPLAYFORM0 The first term of L, the reconstruction loss, enforces high similarity of sampled generated graphs to the input graph G. The second term, KL-divergence, regularizes the code space to allow for sampling of z directly from p(z) instead from q φ (z|G) later.

The dimensionality of z is usually fairly small so that the autoencoder is encouraged to learn a high-level compression of the input instead of learning to simply copy any given input.

While the regularization is independent on the input space, the reconstruction loss must be specifically designed for each input modality.

In the following, we introduce our graph decoder together with an appropriate reconstruction loss.

Graphs are discrete objects, ultimately.

While this does not pose a challenge for encoding, demonstrated by the recent developments in graph convolution networks BID5 , graph generation has been an open problem so far.

In a related task of text sequence generation, the currently dominant approach is character-wise or word-wise prediction BID1 .

However, graphs can have arbitrary connectivity and there is no clear way how to linearize their construction in a sequence of steps 1 .

On the other hand, iterative construction of discrete structures during training without step-wise supervision involves discrete decisions, which are not differentiable and therefore problematic for back-propagation.

Fortunately, the task can become much simpler if we restrict the domain to the set of all graphs on maximum k nodes, where k is fairly small (in practice up to the order of tens).

Under this assumption, handling dense graph representations is still computationally tractable.

We propose to make the decoder output a probabilistic fully-connected graph G = ( A, E, F ) on k nodes at once.

This effectively sidesteps both problems mentioned above.

In probabilistic graphs, the existence of nodes and edges is modeled as Bernoulli variables, whereas node and edge attributes are multinomial variables.

While not discussed in this work, continuous attributes could be easily modeled as Gaussian variables represented by their mean and variance.

We assume all variables to be independent.

Each tensor of the representation of G has thus a probabilistic interpretation.

Specifically, the predicted adjacency matrix A ∈ [0, 1] k×k contains both node probabilities A a,a and edge probabilities A a,b for nodes a = b. The edge attribute tensor E ∈ R k×k×de indicates class probabilities for edges and, similarly, the node attribute matrix F ∈ R k×dn contains class probabilities for nodes.

The decoder itself is deterministic.

Its architecture is a simple multi-layer perceptron (MLP) with three outputs in its last layer.

Sigmoid activation function is used to compute A, whereas edge-and node-wise softmax is applied to obtain E and F , respectively.

At test time, we are often interested in a (discrete) point estimate of G, which can be obtained by taking edge-and node-wise argmax in A, E, and F .

Note that this can result in a discrete graph on less than k nodes.

Given a particular of a discrete input graph G on n ≤ k nodes and its probabilistic reconstruction G on k nodes, evaluation of Equation 1 requires computation of likelihood p θ (G|z) = P (G| G).Since no particular ordering of nodes is imposed in either G or G and matrix representation of graphs is not invariant to permutations of nodes, comparison of two graphs is hard.

However, approximate graph matching described further in Subsection 3.4 can obtain a binary assignment matrix X ∈ {0, 1} k×n , where X a,i = 1 only if node a ∈ G is assigned to i ∈ G and X a,i = 0 otherwise.

Knowledge of X allows to map information between both graphs.

Specifically, input adjacency matrix is mapped to the predicted graph as A = XAX T , whereas the predicted node attribute matrix and slices of edge attribute matrix are transferred to the input graph as F = X T F and DISPLAYFORM0 The maximum likelihood estimates, i.e. cross-entropy, of respective variables are as follows: DISPLAYFORM1 where we assumed that F and E are encoded in one-hot notation.

The formulation considers existence of both matched and unmatched nodes and edges but attributes of only the matched ones.

Furthermore, averaging over nodes and edges separately has shown beneficial in training as otherwise the edges dominate the likelihood.

The overall reconstruction loss is a weighed sum of the previous terms: DISPLAYFORM2

The goal of (second-order) graph matching is to find correspondences X ∈ {0, 1} k×n between nodes of graphs G and G based on the similarities of their node pairs S : (i, j) × (a, b) → R + for i, j ∈ G and a, b ∈ G. It can be expressed as integer quadratic programming problem of similarity maximization over X and is typically approximated by relaxation of X into continuous domain: BID2 .

For our use case, the similarity function is defined as follows: DISPLAYFORM0 DISPLAYFORM1 The first term evaluates similarity between edge pairs and the second term between node pairs, [·] being the Iverson bracket.

Note that the scores consider both feature compatibility ( F and E) and existential compatibility ( A), which has empirically led to more stable assignments during training.

To summarize the motivation behind both Equations 3 and 4, our method aims to find the best graph matching and then further improve on it by gradient descent on the loss.

Given the stochastic way of training deep network, we argue that solving the matching step only approximately is sufficient.

This is conceptually similar to the approach for learning to output unordered sets by BID29 , where the closest ordering of the training data is searched for.

In practice, we are looking for a graph matching algorithm robust to noisy correspondences which can be easily implemented on GPU in batch mode.

Max-pooling matching (MPM) by BID2 is a simple but effective algorithm following the iterative scheme of power methods, see Appendix A for details.

It can be used in batch mode if similarity tensors are zero-padded, i.e. S((i, j), (a, b)) = 0 for n < i, j ≤ k, and the amount of iterations is fixed.

Max-pooling matching outputs continuous assignment matrix X * .

Unfortunately, attempts to directly use X * instead of X in Equation 3 performed badly, as did experiments with direct maximization of X * or soft discretization with softmax or straight-through Gumbel softmax BID10 .

We therefore discretize X * to X using Hungarian algorithm to obtain a strict one-on-one mapping 2 .

While this operation is non-differentiable, gradient can still flow to the decoder directly through the loss function and training convergence proceeds without problems.

Note that this approach is often taken in works on object detection, e.g. BID27 , where a set of detections need to be matched to a set of ground truth bounding boxes and treated as fixed before computing a differentiable loss.

Encoder.

A feed forward network with edge-conditioned graph convolutions (ECC) (Simonovsky & Komodakis, 2017 ) is used as encoder, although any other graph embedding method is applicable.

As our edge attributes are categorical, a single linear layer for the filter generating network in ECC is sufficient.

Due to smaller graph sizes no pooling is used in encoder except for global pooling, for which we employ soft attention pooling of BID18 .

As usual in VAE, we formulate encoder as probabilistic and enforce Gaussian distribution of q φ (z|G) by having the last encoder layer outputs 2c features interpreted as mean and variance, allowing to sample z l ∼ N (µ l (G), σ l (G)) for l ∈ 1, .., c using the re-parameterization trick BID12 .Disentangled Embedding.

In practice, rather than random drawing of graphs, one often desires more control over the properties of generated graphs.

In such case, we follow BID26 and condition both encoder and decoder on label vector y associated with each input graph G. Decoder p θ (G|z, y) is fed a concatenation of z and y, while in encoder q φ (z|G, y), y is concatenated to every node's features just before the graph pooling layer.

If the size of latent space c is small, the decoder is encouraged to exploit information in the label.

Limitations.

The proposed model is expected to be useful only for generating small graphs.

This is due to growth of GPU memory requirements and number of parameters (O(k 2 )) as well matching complexity (O(k 4 )) with small decrease in quality for high values of k.

In Section 4 we demonstrate results for up to k = 38.

Nevertheless, for many applications even generation of small graphs is still very useful.

We demonstrate our method for the task of molecule generation by evaluating on two large public datasets of organic molecules, QM9 and ZINC.

Quantitative evaluation of generative models of images and texts has been troublesome BID28 , as it very difficult to measure realness of generated samples in an automated and objective way.

Thus, researchers frequently resort there to qualitative evaluation and embedding plots.

However, qualitative evaluation of graphs can be very unintuitive for humans to judge unless the graphs are planar and fairly simple.

Fortunately, we found graph representation of molecules, as undirected graphs with atoms as nodes and bonds as edges, to be a convenient testbed for generative models.

On one hand, generated graphs can be easily visualized in standardized structural diagrams.

On the other hand, chemical validity of graphs, as well as many further properties a molecule can fulfill, can be checked using software packages (SanitizeMol in RDKit) or simulations.

This makes both qualitative and quantitative tests possible.

Chemical constraints on compatible types of bonds and atom valences make the space of valid graphs complicated and molecule generation challenging.

In fact, a single addition or removal of edge or change in atom or bond type can make a molecule chemically invalid.

Comparably, flipping a single pixel in MNIST-like number generation problem is of no issue.

To help the network in this application, we introduce three remedies.

First, we make the decoder output symmetric A and E by predicting their (upper) triangular parts only, as undirected graphs are sufficient representation for molecules.

Second, we use prior knowledge that molecules are connected and, at test time only, construct maximum spanning tree on the set of probable nodes {a : A a,a ≥ 0.5} in order to include its edges (a, b) in the discrete pointwise estimate of the graph even if A a,b < 0.5 originally.

Third, we do not generate Hydrogen explicitly and let it be added as "padding" during chemical validity check.

QM9 dataset BID22 contains about 134k organic molecules of up to 9 heavy (non Hydrogen) atoms with 4 distinct atomic numbers and 4 bond types, we set k = 9, d e = 4 and d n = 4.

We set aside 10k samples for testing and 10k for validation (model selection).We compare our unconditional model to the character-based generator of BID7 (CVAE) and the grammar-based generator of BID15 (GVAE) .

We used the code and architecture in BID15 for both baselines, adapting the maximum input length to the smallest possible.

In addition, we demonstrate a conditional generative model for an artificial task of generating molecules given a histogram of heavy atoms as 4-dimensional label y, the success of which can be easily validated.

Setup.

The encoder has two graph convolutional layers (32 and 64 channels) with identity connection, batchnorm, and ReLU; followed by soft attention pooling BID18 with 128 channels and a fully-connected layer (FCL) to output (µ, σ).

The decoder has 3 FCLs (128, 256, and 512 channels) with batchnorm and ReLU; followed by parallel triplet of FCLs to output graph tensors.

We set c = 40, λ A = λ F = λ E = 1, batch size 32, 75 MPM iterations and train for 25 epochs with Adam with learning rate 1e-3 and β 1 =0.5.Embedding Visualization.

To visually judge the quality and smoothness of the learned embedding z of our model, we may traverse it in two ways: along a slice and along a line.

For the former, we randomly choose two c-dimensional orthonormal vectors and sample z in regular grid pattern over the induced 2D plane.

For the latter, we randomly choose two molecules G (1) , G (2) of the same label from test set and interpolate between their embeddings µ(G (1) ), µ(G (2) ).

This also evaluates the encoder, and therefore benefits from low reconstruction error.

We plot two planes in Figure 2 , for a frequent label (left) and a less frequent label in QM9 (right).

Both images show a varied and fairly smooth mix of molecules.

The left image has many valid samples broadly distributed across the plane, as presumably the autoencoder had to fit a large portion of database into this space.

The right exhibits stronger effect of regularization, as valid molecules tend to be only around center.

An example of several interpolations is shown in Figure 3 .

We can find both meaningful (1st, 2nd and 4th row) and less meaningful transitions, though many samples on the lines do not form chemically valid compounds.

Decoder Quality Metrics.

The quality of a conditional decoder can be evaluated by the validity and variety of generated graphs.

For a given label y (l) , we draw n s = 10 4 samples z (l,s) ∼ p(z) and compute the discrete point estimate of their decodingsĜ (l,s) = arg max p θ (G|z (l,s) , y (l) ).Let V (l) be the list of chemically valid molecules fromĜ (l,s) and C (l) be the list of chemically valid molecules with atom histograms equal to y (l) .

We are interested in ratios Valid (l) = |V (l) |/n s and correct graphs and Novel (l) = 1−|set(C (l) )∩QM9|/|set(C (l) )| the fraction of novel out-of-dataset graphs; we define Unique (l) = 0 and Novel (l) = 0 if |C (l) | = 0.

Finally, the introduced metrics are aggregated by frequencies of labels in QM9, e.g. Valid = l Valid (l) freq(y (l) ).

Unconditional decoders are evaluated by assuming there is just a single label, therefore Valid = Accurate.

DISPLAYFORM0 In Table 1 , we can see that on average 50% of generated molecules are chemically valid and, in the case of conditional models, about 40% have the correct label which the decoder was conditioned on.

Larger embedding sizes c are less regularized, demonstrated by a higher number of Unique samples and by lower accuracy of the conditional model, as the decoder is forced less to rely on actual labels.

The ratio of Valid samples shows less clear behavior, likely because the discrete performance is Table 1 : Performance on conditional and unconditional QM9 models evaluated by mean testtime reconstruction log-likelihood (log p θ (G|z)), mean test-time evidence lower bound (ELBO), and decoding quality metrics (Section 4.2).

Baselines CVAE BID7 and GVAE BID15 are listed only for the embedding size with the highest Valid.not directly optimized for.

For all models, it is remarkable that about 60% of generated molecules are out of the dataset, i.e. the network has never seen them during training.

In Appendix B we additionally trade uniqueness for validity.

Looking at the baselines, CVAE can output only very few valid samples as expected, while GVAE generates the highest number of valid samples (60%) but of very low variance (less than 10%).

Additionally, we investigate the importance of graph matching by using identity assignment X instead and thus learning to reproduce particular node permutations in the training set, which correspond to the canonical ordering of SMILES strings from rdkit.

This ablated model (denoted as NoGM in Table 1 ) produces many valid samples of lower variety and, surprisingly, outperforms GVAE in this regard.

In comparison, our model can achieve good performance in both metrics at the same time.

Likelihood.

Besides the application-specific metric introduced above, we also report evidence lower bound (ELBO) commonly used in VAE literature, which corresponds to −L(φ, θ; G) in our notation.

In Table 1 , we state mean bounds over train and test set, using a single z sample per graph.

We observe both reconstruction loss and KL-divergence decrease due to larger c providing more freedom.

However, there seems to be no strong correlation between ELBO and Valid, which makes model selection somewhat difficult.

ZINC dataset BID9 contains about 250k drug-like organic molecules of up to 38 heavy atoms with 9 distinct atomic numbers and 4 bond types, we set k = 38, d e = 4 and d n = 9 and use the same split strategy as with QM9.

We investigate the degree of scalability of an unconditional generative model.

Setup.

The setup is equivalent as for QM9 but with a wider encoder (64, 128, 256 channels).Decoder Quality Metrics.

Our best model with c = 40 has archived Valid = 0.135, which is clearly worse than for QM9.

For comparison, CVAE failed to generated any valid sample, while GVAE achieved Valid = 0.357 (models provided by BID15 , c = 56).We attribute such a low performance to a generally much higher chance of producing a chemicallyrelevant inconsistency (number of possible edges growing quadratically).

To confirm the relationship between performance and graph size k, we kept only graphs not larger than k = 20 nodes, corresponding to 21% of ZINC, and obtained Valid = 0.341 (and Valid = 0.185 for k = 30 nodes, 92% of ZINC).

To verify that the problem is likely not caused by our proposed graph matching loss, we synthetically evaluate it in the following.

Table 2 : Mean accuracy of matching ZINC graphs to their noisy counterparts in a synthetic benchmark as a function of maximum graph size k. DISPLAYFORM0 Matching Robustness.

Robust behavior of graph matching using our similarity function S is important for good performance of GraphVAE.

Here we study graph matching in isolation to investigate its scalability.

To that end, we add Gaussian noise N (0, A ), N (0, E ), N (0, F ) to each tensor of input graph G, truncating and renormalizing to keep their probabilistic interpretation, to create its noisy version G N .

We are interested in the quality of matching between self, P [G, G], using noisy assignment matrix X between G and G N .

The advantage to naive checking X for identity is the invariance to permutation of equivalent nodes.

In Table 2 we vary k and for each tensor separately and report mean accuracies (computed in the same fashion as losses in Equation 3) over 100 random samples from ZINC with size up to k nodes.

While we observe an expected fall of accuracy with stronger noise, the behavior is fairly robust with respect to increasing k at a fixed noise level, the most sensitive being the adjacency matrix.

Note that accuracies are not comparable across tables due to different dimensionalities of random variables.

We may conclude that the quality of the matching process is not a major hurdle to scalability.

In this work we addressed the problem of generating graphs from a continuous embedding in the context of variational autoencoders.

We evaluated our method on two molecular datasets of different maximum graph size.

While we achieved to learn embedding of reasonable quality on small molecules, our decoder had a hard time capturing complex chemical interactions for larger molecules.

Nevertheless, we believe our method is an important initial step towards more powerful decoders and will spark interesting in the community.

There are many avenues to follow for future work.

Besides the obvious desire to improve the current method (for example, by incorporating a more powerful prior distribution or adding a recurrent mechanism for correcting mistakes), we would like to extend it beyond a proof of concept by applying it to real problems in chemistry, such as optimization of certain properties or predicting chemical reactions.

An advantage of a graph-based decoder compared to SMILES-based decoder is the possibility to predict detailed attributes of atoms and bonds in addition to the base structure, which might be useful in these tasks.

Our autoencoder can also be used to pre-train graph encoders for fine-tuning on small datasets BID6 .

In this section we briefly review max-pooling matching algorithm of BID2 .

In its relaxed form, a continuous correspondence matrix X * ∈ [0, 1] k×n between nodes of graphs G and G is determined based on similarities of node pairs i, j ∈ G and a, b ∈ G represented as matrix elements S ia;jb ∈ R + .Let x * denote the column-wise replica of X * .

The relaxed graph matching problem is expressed as quadratic programming task x * = arg max x x T Sx such that DISPLAYFORM0 kn .

The optimization strategy of choice is derived to be equivalent to the power method with iterative update rule x (t+1) = Sx (t) /||Sx (t) || 2 .

The starting correspondences x (0) are initialized as uniform and the rule is iterated until convergence; in our use case we run for a fixed amount of iterations.

In the context of graph matching, the matrix-vector product Sx can be interpreted as sum-pooling over match candidates: x ia ← x ia S ia;ia + j∈Ni b∈Na x jb S ia;jb , where N i and N a denote the set of neighbors of node i and a. The authors argue that this formulation is strongly influenced by uninformative or irrelevant elements and propose a more robust max-pooling version, which considers only the best pairwise similarity from each neighbor: x ia ← x ia S ia;ia + j∈Ni max b∈Na x jb S ia;jb .

Table 3 : Performance on conditional and unconditional QM9 models with implicit node probabilities.

Improvement with respect to Table 1 is emphasized in italics.

Our decoder assumes independence of node and edge probabilities, which allows for isolated nodes or edges.

Making further use of the fact that molecules are connected graphs, we investigate the effect of making node probabilities a function of edge probabilities in this section.

Specifically, we define the probability for node a as that of its most probable edge: A a,a = max b A a,b .The evaluation on QM9 in Table 3 shows a clear improvement in Valid, Accurate, and Novel metrics in both the conditional and unconditional setting.

However, this is paid for by lower variability and higher reconstruction loss.

This indicates that while the new constraint is useful, the model cannot fully cope with it.

Moreover, we have seen no improvement on ZINC dataset.

The regularization in VAE works against achieving perfect reconstruction of training data, especially for small embedding sizes.

To understand the reconstruction ability of our architecture, we train it as unregularized in this section, i.e. with a deterministic encoder and without KL-divergence term in Equation 1.Unconditional models for QM9 achieve mean test log-likelihood log p θ (G|z) of roughly −0.37 (about −0.50 for the implicit model in Appendix B) for all c ∈ {20, 40, 60, 80}. While these loglikelihoods are significantly higher than in Tables 1 and 3 , our architecture can not achieve perfect reconstruction of inputs.

We were successful to increase training log-likelihood to zero only on fixed small training sets of hundreds of examples, where the network could overfit.

This indicates that the network has problems finding generally valid rules for assembly of output tensors.

@highlight

We demonstate an autoencoder for graphs.

@highlight

Learning to generate graphs using deep learning methods in "one shot", directly outputting node and edge existence probabilities, and node attribute vectors.

@highlight

A variational auto encoder to generate graphs