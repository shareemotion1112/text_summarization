We view molecule optimization as a graph-to-graph translation problem.

The goal is to learn to map from one molecular graph to another with better properties based on an available corpus of paired molecules.

Since molecules can be optimized in different ways, there are multiple viable translations for each input graph.

A key challenge is therefore to model diverse translation outputs.

Our primary contributions include a junction tree encoder-decoder for learning diverse graph translations along with a novel adversarial training method for aligning distributions of molecules.

Diverse output distributions in our model are explicitly realized by low-dimensional latent vectors that modulate the translation process.

We evaluate our model on multiple molecule optimization tasks and show that our model outperforms previous state-of-the-art baselines by a significant margin.

The goal of drug discovery is to design molecules with desirable chemical properties.

The task is challenging since the chemical space is vast and often difficult to navigate.

One of the prevailing approaches, known as matched molecular pair analysis (MMPA) BID16 BID11 , learns rules for generating "molecular paraphrases" that are likely to improve target chemical properties.

The setup is analogous to machine translation: MMPA takes as input molecular pairs {(X, Y )}, where Y is a paraphrase of X with better chemical properties.

However, current MMPA methods distill the matched pairs into graph transformation rules rather than treating it as a general translation problem over graphs based on parallel data.

In this paper, we formulate molecular optimization as graph-to-graph translation.

Given a corpus of molecular pairs, our goal is to learn to translate input molecular graphs into better graphs.

The proposed translation task involves many challenges.

While several methods are available to encode graphs BID12 BID32 , generating graphs as output is more challenging without resorting to a domain-specific graph linearization.

In addition, the target molecular paraphrases are diverse since multiple strategies can be applied to improve a molecule.

Therefore, our goal is to learn multimodal output distributions over graphs.

To this end, we propose junction tree encoder-decoder, a refined graph-to-graph neural architecture that decodes molecular graphs with neural attention.

To capture diverse outputs, we introduce stochastic latent codes into the decoding process and guide these codes to capture meaningful molecular variations.

The basic learning problem can be cast as a variational autoencoder, where the posterior over the latent codes is inferred from input molecular pair (X, Y ).

Further, to avoid invalid translations, we propose a novel adversarial training method to align the distribution of graphs generated from the model using randomly selected latent codes with the observed distribution of valid targets.

Specifically, we perform adversarial regularization on the level of the hidden states created as part of the graph generation.

We evaluate our model on three molecular optimization tasks, with target properties ranging from drug likeness to biological activity.

1 As baselines, we utilize state-of-the-art graph generation methods BID23 BID47 and MMPA BID9 .

We demonstrate that our model excels in discovering molecules with desired properties, outperforming the baselines across 2 RELATED WORK Molecular Generation/Optimization Prior work on molecular optimization approached the graph translation task through generative modeling BID14 BID42 BID28 BID8 BID23 BID39 BID31 and reinforcement learning BID17 BID36 BID37 BID47 .

Earlier approaches represented molecules as SMILES strings BID46 , while more recent methods represented them as graphs.

Most of these methods coupled a molecule generator with a property predictor and solved the optimization problem through Bayesian optimization or reinforcement learning.

In contrast, our model is trained to translate a molecular graph into a better graph through supervised learning, which is more sample efficient.

Our approach is closely related to matched molecular pair analysis (MMPA) BID16 BID11 in drug de novo design, where the matched pairs are hard-coded into graph transformation rules.

MMPA's main drawback is that large numbers of rules have to be realized (e.g. millions) to cover all the complex transformation patterns.

In contrast, our approach uses neural networks to learn such transformations, which does not require the rules to be explicitly realized.

Graph Neural Networks Our work is related to graph encoders and decoders.

Previous work on graph encoders includes convolutional BID40 BID4 BID20 BID12 BID35 BID10 BID27 and recurrent architectures BID32 BID7 .

Graph encoders have been applied to social network analysis BID26 BID19 and chemistry BID24 BID13 BID41 .

Recently proposed graph decoders BID44 BID33 BID23 BID48 BID34 focus on learning generative models of graphs.

While our model builds on BID23 to generate graphs, we contribute new techniques to learn multimodal graph-to-graph mappings.

Image/Text Style Translation Our work is closely related to image-to-image translation BID21 , which was later extended by to learn multimodal mappings.

Our adversarial training technique is inspired by recent text style transfer methods BID43 BID49 ) that adversarially regularize the continuous representation of discrete structures to enable end-to-end training.

Our technical contribution is a novel adversarial regularization over graphs that constrains their scaffold structures in a continuous manner.

Our translation model extends the junction tree variational autoencoder BID23 to an encoder-decoder architecture for learning graph-to-graph mappings.

Following their work, we interpret each molecule as having been built from subgraphs (clusters of atoms) chosen from a vocabulary of valid chemical substructures.

The clusters form a junction tree representing the scaffold structure of molecules FIG0 ), which is an important factor in drug design.

Molecules are decoded hierarchically by first generating the junction trees and then combining the nodes of the tree into a molecule.

This coarse-to-fine approach allows us to easily enforce the chemical validity of generated graphs, and provides an enriched representation that encodes molecules at different scales.

In terms of model architecture, the encoder is a graph message passing network that embeds both nodes in the tree and graph into continuous vectors.

The decoder consists of a tree-structured decoder for predicting junction trees, and a graph decoder that learns to combine clusters in the predicted junction tree into a molecule.

Our key departures from BID23 include a unified encoder architecture for trees and graphs, along with an attention mechanism in the tree decoding process.

Viewing trees as graphs, we encode both junction trees and graphs using graph message passing networks.

Specifically, a graph is defined as G = (V, E) where V is the vertex set and E the edge set.

Each node v has a feature vector f v .

For atoms, it includes the atom type, valence, and other atomic properties.

For clusters in the junction tree, f v is a one-hot vector indicating its cluster label.

Similarly, each edge (u, v) ??? E has a feature vector f uv .

Let N (v) be the set of neighbor nodes of v. There are two hidden vectors ?? uv and ?? vu for each edge (u, v) representing the message from u to v and vice versa.

These messages are updated iteratively via neural network g 1 (??): DISPLAYFORM0 uv is the message computed in the t-th iteration, initialized with ?? (0) uv = 0.

In each iteration, all messages are updated asynchronously, as there is no natural order among the nodes.

This is different from the tree encoding algorithm in BID23 , where a root node was specified and an artificial order was imposed on the message updates.

Removing this artifact is necessary as the learned embeddings will be biased by the artificial order.

After T steps of iteration, we aggregate messages via another neural network g 2 (??) to derive the latent vector of each vertex, which captures its local graph (or tree) structure: DISPLAYFORM1 Applying the above message passing network to junction tree T and graph G yields two sets of vectors {x DISPLAYFORM2 is the embedding of tree node i, and the graph vector x G j is the embedding of graph node j.

We generate a junction tree T = (V, E) with a tree recurrent neural network with an attention mechanism.

The tree is constructed in a top-down fashion by expanding the tree one node at a time.

Formally, let??? = {(i 1 , j 1 ), ?? ?? ?? , (i m , j m )} be the edges traversed in a depth first traversal over tree T , where m = 2|E| as each edge is traversed in both directions.

Let??? t be the first t edges in???. At the t-th decoding step, the model visits node i t and receives message vectors h ij from its neighbors.

The message h it,jt is updated through a tree Gated Recurrent Unit BID23 : DISPLAYFORM0 Topological Prediction When the model visits node i t , it first computes a predictive hidden state h t by combining node features f it and inward messages {h k,it } via a one hidden layer network.

The model then makes a binary prediction on whether to expand a new node or backtrack to the parent of i t .

This probability is computed by aggregating the source encodings {x T * } and {x G * } through an attention layer, followed by a feed-forward network (?? (??) stands for ReLU and ??(??) for sigmoid): DISPLAYFORM1 DISPLAYFORM2 Here we use attention(??; U DISPLAYFORM3 Label Prediction If node j t is a new child to be generated from parent i t , we predict its label by DISPLAYFORM4 where q t is a distribution over the label vocabulary and U l att is another set of attention parameters.

The second step in the decoding process is to construct a molecular graph G from a predicted junction tree T .

This step is not deterministic since multiple molecules could correspond to the same junction tree.

For instance, the junction tree in FIG2 can be assembled into three different molecules.

The underlying degree of freedom pertains to how neighboring clusters are attached to each other.

Let G i be the set of possible candidate attachments at tree node i.

Each graph G i ??? G i is a particular realization of how cluster C i is attached to its neighboring clusters {C j , j ??? N T (i)}.The goal of the graph decoder is to predict the correct attachment between the clusters.

To this end, we design the following scoring function f (??) for ranking candidate attachments within the set G i .

We first apply a graph message passing network over graph G i to compute atom representations {?? Gi v }.

Then we derive a vector representation of G i through sum-pooling: DISPLAYFORM0 Finally, we score candidate G i by computing dot products between m Gi and the encoded source graph vectors: DISPLAYFORM1 The graph decoder is trained to maximize the log-likelihood of ground truth subgraphs at all tree nodes (Eq. (10)).

During training, we apply teacher forcing by feeding the graph decoder with ground truth junction tree as input.

During testing, we assemble the graph one neighborhood at a time, following the order in which the junction tree was decoded.

DISPLAYFORM2

Our goal is to learn a multimodal mapping between two molecule domains, such as molecules with low and high solubility, or molecules that are potent and impotent.

During training, we are given a dataset of paired molecules DISPLAYFORM0 where X , Y are the source and target domains.

It is important to note that this joint distribution is a many-to-many mapping.

For instance, there exist many ways to modify molecule X to increase its solubility.

Given a new molecule X, the model should be able to generate a diverse set of outputs.

To this end, we propose to augment the basic encoder-decoder model with low-dimensional latent vectors z to explicitly encode the multimodal aspect of the output distribution.

The mapping to be learned now becomes F : (X, z)

??? Y , with latent code z drawn from a prior distribution P (z), which is a standard Gaussian N (0, I).

There are two challenges in learning this mapping.

First, as shown in the image domain , the latent codes are often ignored by the model unless we explicitly enforce the latent codes to encode meaningful variations.

Second, the model should be properly regularized so that it does not produce invalid translations.

That is, the translated molecule F(X, z) should always belong to the target domain Y given latent code z ??? N (0, I).

In this section, we propose two techniques to address these issues.

First, to encode meaningful variations, we derive latent code z from the embedding of ground truth molecule Y .

The decoder is trained to reconstruct Y when taking as input both its vector encoding z Y and source molecule X. For efficient sampling, the latent code distribution is regularized to be close to the prior distribution, similar to a variational autoencoder.

We also restrict z Y to be a low dimensional vector to prevent the model from ignoring input X and degenerating to an autoencoder.

Specifically, we first embed molecules X and Y into their tree and graph vectors {x T * }, {x G * }; {y T * }, {y G * }, using the same encoder with shared parameters (Sec 3.1).

Then we compute the difference vector ?? X,Y between molecules X and Y as in Eq.(11).

Since each tree and graph vector y i represents local substructure in the junction tree and molecular graph, the difference vector encodes the structural changes occurred from molecule X to Y : DISPLAYFORM0 Following BID25 , the approximate posterior Q(??|X, Y ) is modeled as a normal distribution, allowing us to sample latent codes z T and z G via reparameterization trick.

The mean and log variance of Q(??|X, Y ) is computed from ?? X,Y with two separate affine layers ??(??) and ??(??): DISPLAYFORM1 Finally, we combine the latent code z T and z G with source tree and graph vectors: DISPLAYFORM2 wherex T * andx G * are "perturbed" tree and graph vectors of molecule X. The perturbed inputs are then fed into the decoder to synthesize the target molecule Y .

The training objective follows a conditional variational autoencoder, including a reconstruction loss and a KL regularization term: DISPLAYFORM3 4.2 ADVERSARIAL SCAFFOLD REGULARIZATION Second, to avoid invalid translations, we force molecules decoded from latent codes z ??? N (0, I) to follow the distribution of the target domain through adversarial training BID15 .

The adversarial game involves two components.

The discriminator tries to distinguish real molecules in the target domain from fake molecules generated by the model.

The generator (i.e. our encoderdecoder) tries to generate molecules indistinguishable from the molecules in the target domain.

The main challenge is how to integrate adversarial training into our decoder, as the discrete decisions in tree and graph decoding hinder gradient propagation.

To this end, we apply adversarial regularization over continuous representations of decoded molecular structures, derived from the hidden states in the decoder BID43 BID49 .

That is, we replace the input of the discriminator with continuous embeddings of discrete outputs.

For efficiency reasons, we only enforce the adversarial regularization in the tree decoding step.

As a result, the adversary only matches the scaffold structure between translated molecules and true samples.

The continuous representation is computed as follows.

The decoder first predicts the label distribution q root of the root of tree T .

Starting from the root, we incrementally expand the tree, guided by topological predictions, and compute the hidden messages {h it,jt } between nodes in the partial tree.

At timestep t, the model decides to either expand a new node j t or backtrack to the parent of Sample batch DISPLAYFORM4 Let T (i) be the junction tree of molecule Y (i) .

For each T (i) , compute its continuous representation h (i) by unrolling the decoder with teacher forcing.

Encode each molecule X (i) with latent codes z (i) ??? N (0, I).

For each i, unroll the decoder by feeding the predicted labels and tree topologies to construct the translated junction tree T (i) , and compute its continuous representation h (i) .

Update D(??) by minimizing DISPLAYFORM0 ) along with gradient penalty.

7: end for 8: Sample batch DISPLAYFORM1 Generator training 9: Repeat lines 3-5. 10: Update encoder/decoder by minimizing DISPLAYFORM2 node i t .

We denote this binary decision as d(i t , j t ) = 1 pt>0.5 , which is determined by the topological score p t in Eq.(6).

For the true samples Y , the hidden messages are computed by Eq. FORMULA3 with teacher-forcing, namely replacing the label and topological predictions with their ground truth values.

For the translated samples Y from source molecules X, we replace the one-hot encoding f it with its softmax distribution q it over cluster labels in Eq. FORMULA3 and FORMULA4 .

Moreover, we multiply message h it,jt with the binary gate d(i t , j t ), to account for the fact that the messages should depend on the topological layout of the tree: DISPLAYFORM3 As d(i t , j t ) is computed by a non-differentiable threshold function, we approximate its gradient with a straight-through estimator BID2 BID6 .

Specifically, we replace the threshold function with a differentiable hard sigmoid function during back-propagation, while using the threshold function in the forward pass.

This technique has been successfully applied to training neural networks with dynamic computational graphs BID5 .Finally, after the tree T is completely decoded, we derive its continuous representation h T by concatenating the root label distribution q root and the sum of its inward messages: DISPLAYFORM4 We implement the discriminator D(??) as a multi-layer feedforward network, and train the adversary using Wasserstein GAN with gradient penalty BID18 .

The whole algorithm is described in Algorithm 1.

Data Our graph-to-graph translation models are evaluated on three molecular optimization tasks.

Following standard practice in MMPA, we construct training sets by sampling molecular pairs (X, Y ) with significant property improvement and molecular similarity sim(X, Y ) ??? ??.

The similarity constraint is also enforced at evaluation time to exclude arbitrary mappings that completely ignore the input X. We measure the molecular similarity by computing Tanimoto similarity over Morgan fingerprints BID38 ).

Next we describe how these tasks are constructed.??? Penalized logP We first evaluate our methods on the constrained optimization task proposed by BID23 .

The goal is to improve the penalized logP score of molecules under the similarity constraint.

Following their setup, we experiment with two similarity constraints (?? = 0.4 and 0.6), and we extracted 99K and 79K translation pairs respectively from the ZINC dataset BID45 BID23 for training.

We use their validation and test sets for evaluation.??? Drug likeness (QED) Our second task is to improve drug likeness of compounds.

Specifically, the model needs to translate molecules with QED scores BID3 within the range [0.7, 0.8] into the higher range [0.9, 1.0].

This task is challenging as the target range contains only the top 6.6% of molecules in the ZINC dataset.

We extracted a training set of 88K molecule pairs with similarity constraint ?? = 0.4.

The test set contains 800 molecules.??? Dopamine Receptor (DRD2) The third task is to improve a molecule's biological activity against a biological target named the dopamine type 2 receptor (DRD2).

We use a trained model from BID36 to assess the probability that a compound is active.

We ask the model to translate molecules with predicted probability p < 0.05 into active compounds with p > 0.5.

The active compounds represent only 1.9% of the dataset.

With similarity constraint ?? = 0.4, we derived a training set of 34K molecular pairs from ZINC and the dataset collected by BID36 .

The test set contains 1000 molecules.

Baselines We compare our approaches (VJTNN and VJTNN+GAN) with the following baselines:??? MMPA: We utilized BID9 's implementation to perform MMPA.

Molecular transformation rules are extracted from the ZINC and Olivecrona et al. (2017) 's dataset for corresponding tasks.

During testing, we translate a molecule multiple times using different matching transformation rules that have the highest average property improvements in the database (Appendix B).??? Junction Tree VAE: Jin et al. (2018) is a state-of-the-art generative model over molecules that applies gradient ascent over the learned latent space to generate molecules with improved properties.

Our encoder-decoder architecture is closely related to their autoencoder model.

??? VSeq2Seq: Our second baseline is a variational sequence-to-sequence translation model that uses SMILES strings to represent molecules and has been successfully applied to other molecule generation tasks BID14 .

Specifically, we augment the architecture of BID1 with stochastic latent codes learned in the same way as our VJTNN model.

??? GCPN: GCPN BID47 ) is a reinforcement learning based model that modifies a molecule by iteratively adding or deleting atoms and bonds.

They also adopt adversarial training to enforce naturalness of the generated molecules.

Model Configuration Both VSeq2Seq and our models use latent codes of dimension |z| = 8, and we set the KL regularization weight ?? KL = 1/|z|.

For the VSeq2Seq model, the encoder is a one-layer bidirectional LSTM and the decoder is a one-layer LSTM with hidden state dimension 600.

For fair comparison, we control the size of both VSeq2Seq and our models to be around 4M parameters.

Due to limited space, we defer other hyper-parameter settings to the appendix.

We quantitatively analyze the translation accuracy, diversity, and novelty of different methods.

Translation Accuracy We measure the translation accuracy as follows.

On the penalized logP task, we follow the same evaluation protocol as JT-VAE.

That is, for each source molecule, we decode K times with different latent codes z ??? N (0, I), and report the molecule having the highest property improvement under the similarity constraint.

We set K = 20 so that it is comparable with the baselines.

On the QED and DRD2 datasets, we report the success rate of the learned translations.

We define a translation as successful if one of the K translation candidates satisfies the similarity constraint and its property score falls in the target range (QED ??? [0.9, 1.0] and DRD2 > 0.5).

TAB0 give the performance of all models across the three datasets.

Our models outperform the MMPA baseline with a large margin across all the tasks, clearly showing the advantage of molecular translation approach over rule based methods.

Compared to JT-VAE and GCPN baselines, our models perform significantly better because they are trained on parallel data that provides direct supervision, and therefore more sample efficient.

Overall, our graph-to-graph approach performs better than the VSeq2Seq baseline, indicating the benefit of graph based representation.

The proposed adversarial training method also provides slight improvement over VJTNN model.

The VJTNN+GAN is only evaluated on the QED and DRD2 tasks with well-defined target domains that are explicitly constrained by property ranges.

Diversity We define the diversity of a set of molecules as the average pairwise Tanimoto distance between them, where Tanimoto distance dist(X, Y ) = 1 ??? sim(X, Y ).

For each source molecule, we translate it K times (each with different latent codes), and compute the diversity over the set of validly translated molecules.

2 As we require valid translated molecules to be similar to a given compound, the diversity score is upper-bounded by the maximum allowed distance (e.g. the maximum diversity score is around 0.6 on the QED and DRD2 tasks).

As shown in TAB0 , our methods achieve higher diversity score than MMPA and VSeq2Seq on two and three tasks respectively.

FIG6 shows some examples of diverse translation over the QED and DRD2 tasks.

Novelty Lastly, we report how often our model discovers new molecules in the target domain that are unseen during training.

This is an important metric as the ultimate goal of drug discovery is to design new molecules.

Let M be the set of molecules generated by the model and S be the molecules given during training.

We define novelty as 1 ??? |M ??? S|/|S|.

On the QED and DRD2 datasets, our models discover new compounds most of the time, but less frequently than MMPA and GCPN.

Nonetheless, these methods have much lower translation success rate.

In conclusion, we have evaluated various graph-to-graph translation models for molecular optimization.

By combining the variational junction tree encoder-decoder with adversarial training, we can generate better and more diverse molecules than the baselines.

Tree and Graph Encoder For the graph encoder, functions g 1 (??) and g 2 (??) are parameterized as a one-layer neural network (?? (??) represents the ReLU function): DISPLAYFORM0 For the tree encoder, since it updates the messages with more iterations, we parameterize function g 1 (??) as a tree GRU function for learning stability (edge features f uv are omitted because they are always zero).

We keep the same parameterization for g 2 (??), with a different set of parameters.

Tree Gated Recurrent Unit The tree GRU function GRU(??) for computing message h ij in Eq. FORMULA3 is defined as follows BID23 : DISPLAYFORM1 h ij = tanh DISPLAYFORM2 Tree Decoder Attention The attention mechanism is implemented as a bilinear function between decoder state h t and source tree and graph vectors normalized by the softmax function: DISPLAYFORM3 Graph Decoder We use the same graph neural architecture BID23 for scoring candidate attachments.

Let G i be the graph resulting from a particular merging of cluster C i in the tree with its neighbors C j , j ??? N T (i), and let u, v denote atoms in the graph G i .

The main challenge of attachment scoring is local isomorphism: Suppose there are two neighbors C j and C k with the same cluster labels.

Since they share the same cluster label, exchanging the position of C j and C k will lead to isomorphic graphs.

However, these two cliques are actually not exchangeable if the subtree under j and k are different (Illustrations can be found in BID23 ).

Therefore, we need to incorporate information about those subtrees when scoring the attachments.

To this end, we define index ?? v = i if v ??? C i and ?? v = j if v ??? C j \ C i .

The index ?? v is used to mark the position of the atoms in the junction tree, and to retrieve messages h i,j summarizing the subtree under i along the edge (i, j) obtained by running the tree encoding algorithm.

The tree messages are augmented into the graph message passing network to avoid local isomorphism: Adversarial Scaffold Regularization Algorithm 2 describes the tree decoding algorithm for adversarial training.

It replaces the ground truth input f * with predicted label distributions q * , enabling gradient propagation from the discriminator.

@highlight

We introduce a graph-to-graph encoder-decoder framework for learning diverse graph translations.

@highlight

Proposes a graph-to-graph translation model for molecule optimization inspired by matched molecular pair analysis.

@highlight

Extension of JT-VAE into the graph to graph translation scenario by adding the latent variable to capture multi-modality and an adversarial regularization in the latent space

@highlight

Proposes a quite complex system, involving many different choices and components, for obtaining chemical compouds with improved properties starting from a given corpora.