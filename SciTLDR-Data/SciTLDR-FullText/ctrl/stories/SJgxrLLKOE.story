Engineered proteins offer the potential to solve many problems in biomedicine, energy, and materials science, but creating designs that succeed is difficult in practice.

A significant aspect of this challenge is the complex coupling between protein sequence and 3D structure, and the task of finding a viable design is often referred to as the inverse protein folding problem.

We develop generative models for protein sequences conditioned on a graph-structured specification of the design target.

Our approach efficiently captures the complex dependencies in proteins by focusing on those that are long-range in sequence but local in 3D space.

Our framework significantly improves upon prior parametric models of protein sequences given structure, and takes a step toward rapid and targeted biomolecular design with the aid of deep generative models.

A central goal for computational protein design is to automate the invention of protein molecules with defined structural and functional properties.

This field has seen tremendous progess in the past two decades BID14 , including the design of novel 3D folds BID20 , enzymes BID30 , and complexes BID4 .

However, the current practice often requires multiple rounds of trial-and-error, with first designs frequently failing BID19 BID28 .

Several of the challenges stem from the bottom-up nature of contemporary approaches that rely on both the accuracy of energy functions to describe protein physics as well as on the efficiency of sampling algorithms to explore the protein sequence and structure space.

Here, we explore an alternative, top-down framework for protein design that directly learns a conditional generative model for protein sequences given a specification of the target structure, which is represented as a graph over the sequence elements.

Specifically, we augment the autoregressive self-attention of recent sequence models BID34 with graph-based descriptions of the 3D structure.

By composing multiple layers of structured self-attention, our model can effectively capture higher-order, interaction-based dependencies between sequence and structure, in contrast to previous parameteric approaches BID24 BID36 that are limited to only the first-order effects.

The graph-structured conditioning of a sequence model affords several benefits, including favorable computational efficiency, inductive bias, and representational flexibility.

We accomplish the first two by leveraging a well-evidenced finding in protein science, namely that long-range dependencies in sequence are generally short-range in 3D space BID23 BID3 .

By making the graph and self-attention similarly sparse and localized in 3D space, we achieve computational scaling that is linear in sequence length.

Additionally, graph structured inputs offer representational flexibility, as they accomodate both coarse, 'flexible backbone' (connectivity and topology) as well as fine-grained (precise atom locations) descriptions of structure.

We demonstrate the merits of our approach via a detailed empirical study.

Specifically, we evaluate our model at structural generalization to sequences of protein folds that were outside of the training set.

Our model achieves considerably improved generalization performance over the recent deep models of protein sequence given structure as well as structure-na??ve language models.

Generative models for proteins A number of works have explored the use of generative models for protein engineering and design .

Recently O'Connell et al. (2018) and BID36 proposed neural models for sequences given 3D structure, where the amino acids at different positions in the sequence are predicted independently of one another.

BID11 introduced a generative model for protein sequences conditioned on a 1D, context-free grammar based specification of the fold topology.

BID8 and BID37 used deep neural networks to model the conditional distribution of letters in a specific position given the structure and sequence of all surrounding residues.

In contrast to these works, our model captures the joint distribution of the full protein sequence while grounding these dependencies in terms of long-range interactions arising from the structure.

In parallel to the development of structure-based models, there has been considerable work on deep generative models for protein sequences in individual protein families with directed BID26 BID31 and undirected BID33 ) latent variable models.

These methods have proven useful for protein engineering, but presume the availability of a large number of sequences from a particular family.

More recently, several groups have obtained promising results using unconditional protein language models BID6 BID0 BID12 BID27 to learn protein sequence representations that can transfer well to supervised tasks.

While serving different purposes, we emphasize that one advantage of conditional generative modeling is to facilitate adaptation to specific (and potentially novel) parts of structure space.

Language models trained on hundreds of millions of evolutionary sequences are unfortunately still 'semantically' bottlenecked by the much smaller number of evolutionary 3D folds (perhaps thousands) that the sequences design.

We propose evaluating protein language models with structure-based splitting of sequence data (Section 3, albeit on much smaller sequence data), and begin to see how unconditional language models may struggle to assign high likelihoods to sequences from out-of-training folds.

In a complementary line of research, deep models of protein structure BID2 BID16 BID1 have been proposed recently that could be used to craft 3D structures for input to sequence design.

Protein design For classical approaches to computational protein design, which are based on joint modeling of structure and sequence, we refer the reader to a review of both methods and accomplishments in BID14 .

More recently, proposed a non-parametric approach to protein design in which a target design is decomposed into substructural motifs that are then queried against a protein database.

In this work we will focus on comparisons with direct parametric models of the sequence-structure relationship.

Self-Attention Our model extends the Transformer BID7 to additionally capture sparse, pairwise relational information between sequence elements.

The dense variation of this problem was explored in BID29 and .

As noted in those works, incorporating general pairwise information incurs O(N 2 ) memory (and computational) cost for sequences of length N , which can be highly limiting for training on GPUs.

We circumvent this cost by instead restricting the self-attention to the sparsity of the input graph.

Given this graph-structured self-attention, our model may also be reasonably cast in the framework of message-passing or graph neural networks BID10 BID5 .

Our approach is similar to Graph Attention Networks BID35 , but augmented with edge features and an autoregressive decoder.

We represent protein structure in terms of an attributed graph G = (V, E) with node features V = {v 1 , . . .

, v N } and edge features E = {e ij } i =j over the sequence residues (amino acids).

This formulation can accommodate different variations on the macromolecular design problem, including 3D considerations For a rigid-body design problem, the structure for conditioning is a fixed set of backbone coordinates X = {x i ??? R 3 : 1 ??? i ??? N }, where N is the number of positions 1 .

We desire a graph representation of the coordinates G(X ) that has two properties:??? Invariance.

The features are invariant to rotations and translations.??? Locally informative.

The edge features incident to v i due to its neighbors N(i),i.e.

{e ij } j???N(i) , contain sufficient information to reconstruct all adjacent coordinates {x j } j???N(i) up to rigid-body motion.

While invariance is motivated by standard symmetry considerations, the second property is motivated by limitations of current graph neural networks BID10 .

In these networks, updates to node features v i depend only on the edge and node features adjacent to v i .

However, typically, these features are insufficient to reconstruct the relative neighborhood positions {x j } j???N(i) , so individual updates cannot fully depend on the 'local environment'.

For example, pairwise distances D ij and D il are insufficient to determine if x j and x l are on the same or opposite sides of x i .

We develop invariant and locally informative features by first augmenting the points x i with 'orientations' O i that define a local coordinate system at each point.

We define these in terms of the backbone geometry as DISPLAYFORM0 where b i is the negative bisector of angle between the rays (x i???1 ??? x i ) and (x i+1 ??? x i ), and n i is a unit vector normal to that plane.

Formally, we have DISPLAYFORM1 Finally, we derive the spatial edge features e (s) ij from the rigid body transformation that relates reference frame (x i , O i ) to reference frame (x j , O j ).

While this transformation has 6 degrees of freedom, we decompose it into features for distance, direction, and orientation as DISPLAYFORM2 Here r(??) is a function that lifts the distances into a radial basis 2 , the term in the middle corresponds to the relative direction of x j in the reference frame of (x i , O i ), and q(??) converts the 3 ?? 3 relative rotation matrix to a quaternion representation.

Quaternions represent rotations as four-element vectors that can be efficiently and reasonably compared by inner products Huynh (2009).

Positional encodings Taking a cue from the original Transformer model, we obtain positional embeddings e (p) ij that encode the role of local structure around node i. Specifically, we need to model the positioning of each neighbor j relative to the node under consideration i.

Therefore, we obtain the position embedding as a sinusoidal function of the gap i ??? j. Note that this is in contrast to the absolute positional encodings of the original Transformer, and instead matches the relative encodings in BID29 .Node and edge features Finally, we obtain an aggregate edge encoding vector e ij by concatenating the structural encodings e (s) ij and the positional encodings e (p) ij and then linearly transforming them to have the same dimension as the model.

We only include edges in the k-nearest neighbors graph of X , with k = 30 for all experiments.

For node features, we compute the three dihedral angles of the protein backbone (?? i , ?? i , ?? i ) and embed these on the 3-torus as {sin, cos} ?? (?? i , ?? i , ?? i ).

We also consider 'flexible backbone' descriptions of 3D structure based solely on topological binary edge features.

We combine the relative positional encodings with two binary edge features: contacts that indicate when the distance between C ?? residues at i and j are less than 8 Angstroms and hydrogen bonds which are directed and defined by the electrostatic model of DSSP BID17 .

These features implicitly integrate over different 3D backbone configurations that are compatible with the specified topology.

In this work, we introduce a Structured Transformer model that draws inspiration from the selfattention based Transformer model BID34 and is augmented for scalable incorporation of relational information.

While general relational attention incurs quadratic memory and computation costs, we avert these by restricting the attention for each node i to the set N(i, k) of its k-nearest neighbors in 3D space.

Since our architecture is multilayered, iterated local attention can derive progressively more global estimates of context for each node i. Second, unlike the standard Transformer, we also include edge features to embed the spatial and positional dependencies in deriving the attention.

Thus, our model generalizes Transformer to spatially structured settings.

Autoregressive decomposition We decompose the joint distribution of the sequence given structure p(s|x) autoregressively as DISPLAYFORM0 where the conditional probability p(s i |x, s <i ) of amino acid s i at position i is conditioned on both the input structure x and the preceding amino acids s <i = {s 1 , . . .

s i???1 } 4 .

These conditionals are parameterized in terms of two sub-networks: an encoder that computes refined node embeddings from structure-based node features V(x) and edge features E(x) and a decoder that autoregressively predicts letter s i given the preceding sequence and structural embeddings from the encoder.

Encoder Our encoder module is designed as follows.

A transformation W h : DISPLAYFORM1 Each layer of the encoder implements a multi-head self-attention component, where head ??? [L] can attend to a separate subspace of the embeddings via learned query, key and value transformations BID34 .

The queries are derived from the current embedding at node i while the keys and values from the relational information r ij = (h j , e ij ) at adjacent nodes j ??? N (i, k).

Specifically, W DISPLAYFORM2 where m DISPLAYFORM3 The results of each attention head l are collected as the weighted sum h DISPLAYFORM4 ( ) ij and then concatenated and transformed to give the update DISPLAYFORM5 We update the embeddings with this residual and alternate between these self-attention layers and position-wise feedforward layers as in the original Transformer BID34 .

We stack multiple layers atop each other, and thereby obtain continually refined embeddings as we traverse the layers bottom up.

The encoder yields the embeddings produced by the topmost layer as its output.

Decoder Our decoder module has the same structure as the encoder but with augmented relational information r ij that allows access to the preceding sequence elements s <i in a causally consistent manner.

Whereas the keys and values of the encoder are based on the relational information r ij = (h j , e ij ), the decoder can additionally access sequence elements s j as DISPLAYFORM6 Here h DISPLAYFORM7 is the embedding of node j in the current layer of the decoder, h (enc) j is the embedding of node j in the final layer of the encoder, and g(s j ) is a sequence embedding of amino acid s j at node j. This concatenation and masking structure ensures that sequence information only flows to position i from positions j < i, but still allows position i to attend to subsequent structural information.

We stack three layers of self-attention and position-wise feedforward modules for the encoder and decoder with a hidden dimension of 128 throughout the experiments 5 .

Dataset To evaluate the ability of the models to generalize across different protein folds, we collected a dataset based on the CATH hierarchical classification of protein structure BID25 .

For all domains in the CATH 4.2 40% non-redundant set of proteins, we obtained full chains up to length 500 (which may contain more than one domain) and then cluster-split these at the CATH topology level (i.e. fold level) into training, validation, and test sets at an 80/10/10 split.

Chains containing multiple CATH tpologies were purged with precedence for test over validation over train.

Our splitting procedure ensured that no two domains from different sets would share the same topologies (folds).

The final splits contained 18025 chains in the training set, 1637 chains in the validation set, and 1911 chains in the test set.

Optimization We trained models using the learning rate schedule and initialization of BID34 , a dropout BID32 rate of 10%, and early stopping based on validation perplexity.5 except for the decoder-only language model experiment which used a hidden dimension of 256

Many protein sequences may reasonably design the same 3D structure BID21 , and so we focus on likelihood-based evaluations of model performance.

Specifically, we evaluate the perplexity per letter of test protein folds (topologies) that were held out from the training and validation sets.

Protein perplexities What kind of perplexities might be useful?

To provide context, we first present perplexities for some simple models of protein sequences in TAB0 .

The amino acid alphabet and its natural frequencies upper-bound perplexity at 20 and ???17.8, respectively.

Random protein sequences under these null models are unlikely to be functional without further selection BID18 .

First order profiles of protein sequences such as those from the Pfam database BID9 , however, are widely used for protein engineering.

We found the average perplexity per letter of profiles in Pfam 32 (ignoring alignment uncertainty) to be ???11.6.

This suggests that even models with high perplexities of this order have the potential to be useful models for the space of functional protein sequences.

The importance of structure We found that there was a significant gap between unconditional language models of protein sequences and models conditioned on structure.

Remarkably, for a range of structure-independent language models, the typical test perplexities turned out to be ???16-17 TAB1 , which were barely better than null letter frequencies TAB0 .

We emphasize that the RNNs were not broken and could still learn the training set in these capacity ranges.

It would seem that protein language models trained on one subset of 3D folds (in our cluster-splitting procedure) generalize poorly to predict the sequences of unseen folds, which is important to consider when training protein language models for protein engineering and design.

All structure-based models had (unsurprisingly) considerably lower perplexities.

In particular, our Structured Transformer model attained a perplexity of ???7 on the full test set.

When we compared different graph features of protein structure TAB2 , we indeed found that using local orientation information was important.

We also compared to a recent method SPIN2 that predicts, using deep neural networks, protein sequence profiles given protein structures BID24 .

Since SPIN2 is computationally intensive (minutes per protein for small proteins) and was trained on complete proteins rather than chains, we evaluated it on two subsets of the full test set: a 'Small' subset of the test set containing chains up to length 100 and a 'Single chain' subset containing only those models where the single chain accounted for the entire protein record in the Protein Data Bank.

Both subsets discarded any chains with structural gaps.

We found that our Structured Transformer model considerably improved upon the perplexities of SPIN2 TAB1 .

We presented a new deep generative model to 'design' protein sequences given a graph specification of their structure.

Our model augments the traditional sequence-level self-attention of Transformers BID34 with relational 3D structural encodings and is able to leverage the spatial locality of dependencies in molecular structures for efficient computation.

When evaluated on unseen folds, the model achieves significantly improved perplexities over the state-of-the-art parametric generative models.

Our framework suggests the possibility of being able to efficiently design and engineer protein sequences with structurally-guided deep generative models, and underscores the central role of modeling sparse long-range dependencies in biological sequences.

We thank members of the MIT MLPDS consortium for helpful feedback and discussions.

<|TLDR|>

@highlight

We learn to conditionally generate protein sequences given structures with a model that captures sparse, long-range dependencies.