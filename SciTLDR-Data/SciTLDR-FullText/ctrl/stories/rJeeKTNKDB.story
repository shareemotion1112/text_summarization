The problem of accelerating drug discovery relies heavily on automatic tools to optimize precursor molecules to afford them with better biochemical properties.

Our work in this paper substantially extends prior state-of-the-art on graph-to-graph translation methods for molecular optimization.

In particular, we realize coherent multi-resolution representations by interweaving the encoding of substructure components with the atom-level encoding of the original molecular graph.

Moreover, our graph decoder is fully autoregressive, and interleaves each step of adding a new substructure with the process of resolving its attachment to the emerging molecule.

We evaluate our model on multiple molecular optimization tasks and show that our model significantly outperforms previous state-of-the-art baselines.

Molecular optimization seeks to modify compounds in order to improve their biochemical properties.

This task can be formulated as a graph-to-graph translation problem analogous to machine translation.

Given a corpus of molecular pairs {(X, Y )}, where Y is a paraphrase of X with better chemical properties, the model is trained to translate an input molecular graph into its better form.

The task is difficult since the space of potential candidates is vast, and molecular properties can be complex functions of structural features.

Moreover, graph generation is computationally challenging due to complex dependencies involved in the joint distribution over nodes and edges.

Similar to machine translation, success in this task is predicated on the inductive biases built into the encoder-decoder architecture, in particular the process of generating molecular graphs.

Prior work (Jin et al., 2019) proposed a junction tree encoder-decoder that utilized valid chemical substructures (e.g., aromatic rings) as building blocks to generate graphs.

Each molecule was represented as a junction tree over chemical substructures in addition to the original atom-level graph.

While successful, the approach remains limited in several ways.

The tree and graph encoding were carried out separately, and decoding proceeded in strictly successive steps: first generating the junction tree for the new molecule, and then attaching its substructures together.

This means the predicted attachments do not impact the subsequent substructure choices (see Figure 1a) .

Moreover, the attachment prediction process is non-autoregressive, thus it can predict inconsistent substructure attachments across different nodes in the junction tree (see Figure 1b ).

We propose a multi-resolution, hierarchically coupled encoder-decoder for graph generation.

Our auto-regressive decoder interleaves the prediction of substructure components with their attachments to the molecule being generated.

In particular, a target graph is unraveled as a sequence of triplet predictions (where to expand the graph, new substructure type, its attachment).

This enables us to model strong dependencies between successive attachments and substructure choices.

The encoder is designed to represent molecules at different resolutions in order to match the proposed decoding process.

Specifically, the encoding of each molecule proceeds across three levels, with each layer capturing essential information for its corresponding decoding step.

The graph convolution of atoms at the lowest level supports the prediction of attachments and the convolution over substructures at the highest level supports the prediction of successive substructures.

Compared to prior work, our decoding process is much more efficient because it decomposes each generation step into a hierarchy of smaller steps in order to avoid combinatorial explosion.

We also extend the method to handle conditional translation where desired criteria are fed as input to the translation process.

This enables our method to handle different combinations of criteria at test time.

Since their tree and graph decoders are isolated, the model can generate invalid junction trees which cannot be assembled into any molecule.

This problem can be solved when we interleave the tree and graph decoding steps, allowing the predicted attachments to guide the substructure prediction; b) Their non-autoregressive graph decoder often predicts inconsistent local substructure attachments during training.

To this end, we propose an autoregressive decoder that interleaves the prediction of substructures with their attachments.

We evaluate our new model on multiple molecular optimization tasks.

Our baselines include previous state-of-the-art graph generation methods (You et al., 2018a; Liu et al., 2018; Jin et al., 2019) and an atom-based translation model we implemented for a more comprehensive comparison.

Our model significantly outperforms these methods in discovering molecules with desired properties, yielding 3.3% and 8.1% improvement on QED and DRD2 optimization tasks.

During decoding, our model runs 6.3 times faster than previous substructure-based generation methods.

We further conduct ablation studies to validate the advantage of our hierarchical decoding and multi-resolution encoding.

Finally, we show that conditional translation can succeed (generalize) even when trained on molecular pairs with only 1.6% of them having desired target property combination.

Molecular Graph Generation Previous work have adopted various approaches for generating molecular graphs.

Methods (Gómez-Bombarelli et al., 2018; Segler et al., 2017; Kusner et al., 2017; Dai et al., 2018; Guimaraes et al., 2017; Olivecrona et al., 2017; Popova et al., 2018; Kang & Cho, 2018) generate molecules based on their SMILES strings (Weininger, 1988) .

Simonovsky & Komodakis (2018) ; De Cao & Kipf (2018) ; Ma et al. (2018) developed generative models which output the adjacency matrices and node labels of the graphs at once.

You et al. (2018b) ; ; Samanta et al. (2018); Liu et al. (2018) proposed generative models decoding molecules sequentially node by node.

You et al. (2018a); Zhou et al. (2018) adopted similar node-by-node approaches in the context of reinforcement learning.

Kajino (2018) developed a hypergraph grammar based method for molecule generation.

Our work is most closely related to Jin et al. (2018; that generate molecules based on substructures.

They adopted a two-stage procedure for realizing graphs.

The first step generates a junction tree with substructures as nodes, capturing their coarse relative arrangements.

The second step resolves the full graph by specifying how the substructures should be attached to each other.

Their major drawbacks are 1) The second step introduced local independence assumptions and therefore the decoder is not autoregressive.

2) These two steps are applied stage-wise during decoding -first realizing the junction tree and then reconciling attachments without feedback.

In contrast, our method jointly predicts the substructures and their attachments with an autoregressive decoder.

Graph Encoders Graph neural networks have been extensively studied for graph encoding (Scarselli et al., 2009; Bruna et al., 2013; Li et al., 2015; Niepert et al., 2016; Kipf & Welling, 2017; Hamilton et al., 2017; Lei et al., 2017; Velickovic et al., 2017; Xu et al., 2018) .

Our method is related to graph encoders for molecules (Duvenaud et al., 2015; Kearnes et al., 2016; Dai et al., 2016; Gilmer et al., 2017; Schütt et al., 2017) .

Different to these approaches, our method represents molecules as hierarchical graphs spanning from atom-level graphs to substructure-level trees.

Our work is most closely related to (Defferrard et al., 2016; Ying et al., 2018; Gao & Ji, 2019 ) that learn to represent graphs in a hierarchical manner.

In particular, Defferrard et al. (2016) utilized graph coarsening algorithms to construct multiple layers of graph hierarchy and Ying et al. (2018) ; Figure 2 : Overview of our approach.

Each substructure S i is a subgraph of a molecule (e.g., rings).

In each step, our decoder adds a new substructure and predicts its attachment to current graph.

Our encoder represents each molecule across three levels (atom layer, attachment layer and substructure layer), with each layer capturing relevant information for the corresponding decoding step.

Gao & Ji (2019) proposed to learn the graph hierarchy jointly with the encoding process.

Despite some differences, all of these methods seek to represent graphs as a single vector for regression or classification tasks.

In contrast, our focus is graph generation and a molecule is encoded into multiple sets of vectors, each representing the input at different resolutions.

Those vectors are dynamically aggregated by decoder attention modules in each graph generation step.

The graph translation task seeks to learn a function F that maps a molecule X into another molecule G with better chemical properties.

F is parameterized as an encoder-decoder with neural attention.

Both our encoder and decoder are illustrated in Figure 2 .

In each generation step, our decoder adds a new substructure (substructure prediction) and decides how it should be attached to the current graph.

The attachment prediction proceeds in two steps: predicting attaching points in the new substructure and their corresponding attaching points in the current graph (attachment prediction 1-2).

To support the above hierarchical generation, we need to design a matching encoder representing molecules at multiple resolutions in order to provide necessary information for each decoding step.

Therefore, we propose to represent a molecule X by a hierarchical graph H X with three components: 1) substructure layer representing how substructures are coarsely connected; 2) attachment layer showing the attachment configuration of each substructure; 3) atom layer showing how atoms are connected in the graph.

Our model encodes nodes in H X into substructure vectors c S X , attachment vectors c A X and atom vectors c G X , which are fed to the decoder for corresponding prediction steps.

As our encoder is tailored for the decoder, we first describe our decoder to clarify relevant concepts.

Notations We denote the sigmoid function as σ(·).

MLP(a, b) represents a multi-layer neural network whose input is the concatenation of a and b. attention θ (h * , c X ) stands for a bilinear attention over vectors c X with query vector h * .

Substructures We define a substructure S i = (V i , E i ) as subgraph of molecule G induced by atoms in V i and bonds in E i .

Given a molecule, we extract its substructures S 1 , · · · , S n such that their union covers the entire molecular graph: V = i V i and E = i E i .

In this paper, we consider two types of substructures: rings and bonds.

We denote the vocabulary of substructures as S, which is constructed from the training set.

In our experiments, |S| < 500 and it has over 99.5% coverage on test sets.

Substructure Tree To characterize how substructures are connected in the molecule G, we construct its corresponding substructure tree T , whose nodes are substructures S 1 , · · · , S n .

Specifically, we construct the tree by first drawing edges between S i and S j if they share common atoms, and then applying tree decomposition over T to ensure it is tree-structured.

Generation Our graph decoder generates a molecule G by incrementally expanding its substructure tree in its depth-first order.

Suppose the model is currently visiting substructure node S k .

It makes the following predictions conditioned on encoding of input X (see 2) It predicts that new substructure S t should be a ring (substructure prediction) 3) It predicts how this new ring should be attached to the graph (attachment prediction).

Finally, the decoder moves to S t and repeats the process.

1.

Topological Prediction:

It first predicts whether there will be a new substructure attached to S k .

If not, the model backtracks to its parent node S d k in the tree.

Let h S k be the hidden representation of S k learned by decoder (which will be elaborated in §3.2).

This probability is predicted by a MLP with attention over substructure vectors c S X of X:

2.

Substructure Prediction: If p k > 0.5, the model decides to create a new substructure S t from S k and sets its parent d t = k. It then predicts the substructure type of S t using another MLP that outputs a distribution over the vocabulary S:

3.

Attachment Prediction: Now the model needs to decide how S t should be attached to S k .

The attachment between S t and S k is defined as atom pairs M t = {(u j , v j )|u j ∈ S t , v j ∈ S k } where atom u j and v j are attached together.

We predict those atom pairs in two steps: 1) We first predict the atoms {v j } ⊂ S t that will be attached to S k .

Since the graph S t is always fixed and the number of attaching atoms between two substructures is usually small, we can enumerate all possible configurations {v j } to form a vocabulary A(S t ) for each substructure S t .

This allows us to formulate the prediction of {v j } as a classification task -predicting the correct configuration A t = (S t , {v j }) from the vocabulary A(S t ):

2) Given the predicted attaching points {v j }, we need to find the corresponding atoms {u j } in the substructure S k .

As the attaching points are always consecutive, there exist at most

The probability of a candidate attachment M is computed based on the atom representations h uj and h vj learned by the decoder:

The above three predictions together give an autoregressive factorization of the distribution over the next substructure and its attachment.

Each of the three decoding steps depends on the outcome of previous step, and predicted attachments will in turn affect the prediction of subsequent substructures.

During training, we apply teacher forcing to the above generation process, where the generation order is determined by a depth-first traversal over the ground truth substructure tree.

The attachment enumeration is tractable because most of the substructures are small.

In our experiments, the average size of attachment vocabulary |A(S t )| < 5 and the number of candidate attachments is less than 20.

Our encoder represents a molecule X by a hierarchical graph H X in order to support the above decoding process.

The hierarchical graph has three components (see Figure 4 ):

1.

Atom layer: The atom layer is the molecular graph of X representing how its atoms are connected.

Each atom node v is associated with a label a v indicating its atom type and charge.

Each edge (u, v) in the atom layer is labeled with b uv indicating its bond type.

2.

Attachment layer: This layer is derived from the substructure tree of molecule X. Each node A i in this layer represents a particular attachment configuration of substructure S i in the vocabulary A(S i ).

Specifically, A i = (S i , {v j }) where {v j } are the attaching atoms between S i and its parent S di in the tree.

This layer provides necessary information for the attachment prediction (step 1).

Figure 4 illustrates how A i and the vocabulary A(S i ) look like.

Dashed arrows connect each atom to the substructures it belongs.

In the attachment layer, each node A i is a particular attachment configuration of substructure S i .

Right: Attachment vocabulary for a ring.

The attaching points in each configuration (highlighted in red) must be consecutive.

3.

Substructure layer: This layer is the same as the substructure tree.

This layer provides essential information for the substructure prediction in the decoding process.

We further introduce edges that connect the atoms and substructures between different layers in order to propagate information in between.

In particular, we draw a directed edge from atom v in the atom layer to node A i in the attachment layer if v ∈ S i .

We also draw edges from node A i to node S i in the substructure layer.

This gives us the hierarchical graph H X for molecule X, which will be encoded by a hierarchical message passing network (MPN) (see Figure 4) .

The encoder contains three MPNs that encode each of the three layer.

We use the MPN architecture from Jin et al. (2019) .

For simplicity, we denote the MPN encoding process as MPN ψ (·) with parameter ψ.

Atom Layer MPN We first encode the atom layer of H X (denoted as H g X ).

The inputs to this MPN are the embedding vectors {e(a u )}, {e(b uv )} of all the atoms and bonds in X. During encoding, the network propagates the message vectors between different atoms for T iterations and then outputs the atom representation h v for each atom v:

Attachment Layer MPN The input feature of each node A i in the attachment layer H a X is an concatenation of the embedding e(A i ) and the sum of its atom vectors {h v | v ∈ S i }:

The input feature for each edge (A i , A j ) in this layer is an embedding vector e(d ij ), where d ij describes the relative ordering between node A i and A j during decoding.

Specifically, we set d ij = k if node A i is the k-th child of node A j and d ij = 0 if A i is the parent.

We then run T iterations of message passing over H a X to compute the substructure representations: c

Substructure Layer MPN Similarly, the input feature of node S i in this layer is computed as the concatenation of embedding e(S i ) and the node vector h Ai from the previous layer.

Finally, we run message passing over the substructure layer H s X to obtain the substructure representations:

In summary, the output of our hierarchical encoder is a set of vectors c X = c

X that represent a molecule X at multiple resolutions.

These vectors are input to the decoder attention.

Decoder MPN During decoding, we use the same hierarchical MPN architecture to encode the hierarchical graph H G at each step t.

This gives us the substructure vectors h S k and atom vectors h vj in §3.1.

All future nodes and edges are masked to ensure the prediction of current substructure and attachment only depends on previously generated outputs.

Our training set contains molecular pairs (X, Y ) where each compound X can be associated with multiple outputs Y since there are many ways to modify X to improve its properties.

In order to Generate molecule Y given c X and z 6: end for Figure 5 : Conditional translation.

generate diverse outputs, we follow Jin et al. (2019) and extend our method to a variational translation model F : (X, z) →

Y with an additional input z. The latent vector z indicates the intended mode of translation which is sampled from a Gaussian prior P (z) during testing.

We train our model using variational inference (Kingma & Welling, 2013) .

Given a training example (X, Y ), we sample z from the posterior Q(z|X, Y ) = N (µ X,Y , σ X,Y ).

To compute Q(z|X, Y ), we first encode X and Y into their representations c X and c Y and then compute vector δ X,Y that summarizes the structural changes from molecule X to Y at both atom and substructure level:

Finally, we compute

and sample z using reparameterization trick.

The latent code z is passed to the decoder along with the input representation c X to reconstruct output Y .

The overall training objective follows a standard conditional VAE:

Conditional Translation In the above formulation, the model does not know what properties are being optimized during translation.

During testing, users cannot change the behavior of a trained model (i.e., what properties should be changed).

This may become a limitation of our method in a multi-property optimization setting.

Therefore, we extend our method to handle conditional translation where the desired criteria are also fed as input to the translation process.

In particular, let g X,Y be a translation criteria indicating what properties should be changed.

During variational inference, we compute µ X,Y and σ X,Y with an additional input g X,Y :

We then augment the latent code as [z, g X,Y ] and pass it to the decoder.

During testing, the user can specify their criteria in g X,Y to control the outcome (e.g., Y should be drug-like and bioactive).

We follow the experimental design by Jin et al. (2019) and evaluate our translation model on their single-property optimization tasks.

As molecular optimization in the real-world often involves different property criteria, we further construct a novel conditional optimization task where the desired criteria is fed as input to the translation process.

To prevent the model from ignoring input X and translating it into arbitrary compound, we require the molecular similarity between X and output Y to be above certain threshold sim(X, Y )

≥ δ at test time.

The molecular similarity is defined as the Tanimoto similarity over Morgan fingerprints (Rogers & Hahn, 2010) of two molecules.

Single-property Optimization This dataset consists of four different tasks.

For each task, we train and evaluate our model on their provided training and test sets.

For these tasks, our model is trained under an unconditional setting (without g X,Y as input).

• LogP Optimization: The penalized logP score (Kusner et al., 2017) measures the solubility and synthetic accessibility of a compound.

In this task, the model needs to translate input X into output Y such that logP(Y ) > logP(X).

We experiment with two similarity thresholds δ = {0.4, 0.6}. Olivecrona et al. (2017) .

The similarity constraint is sim(X, Y ) ≥ 0.4.

1.

Y is both drug-like and DRD2-active.

Here both properties need to be improved after translation.

2.

Y is drug-like but DRD2-inactive.

In this case, DRD2 is an off-target that may cause side effects.

Therefore only the drug-likeness should be improved after translation.

As different users may be interested in different settings, we encode the desired criteria as vector g and train our model under the conditional translation setup in §3.3.

Like single-property tasks, we impose a similarity constraint sim(X, Y ) ≥ 0.4 for both settings.

Evaluation Metrics Our evaluation metrics include translation accuracy and diversity.

Each test molecule X i is translated K = 20 times with different latent codes sampled from the prior distribution.

On the logP optimization, we select compound Y i as the final translation of X i that gives the highest property improvement and satisfies sim(X i , Y i ) ≥ δ.

We then report the average property improvement

For other tasks, we report the translation success rate.

A compound is successfully translated if one of its K translation candidates satisfies all the similarity and property constraints of the task.

To measure the diversity, for each molecule we compute the average pairwise Tanimoto distance between all its successfully translated compounds.

Here the Tanimoto distance is defined as dist(X, Y ) = 1 − sim(X, Y ).

Baselines We compare our method (HierG2G) against the baselines including GCPN (You et al., 2018a) , MMPA (Dalke et al., 2018) and translation based methods Seq2Seq and JTNN (Jin et al., 2019) .

Seq2Seq is a sequence-to-sequence model that generates molecules by their SMILES strings.

JTNN is a graph-to-graph architecture that generates molecules structure by structure, but its decoder is not fully autoregressive.

We also compare with CG-VAE (Liu et al., 2018 ), a generative model that decodes molecules atom by atom and optimizes properties in the latent space using gradient ascent.

To make a direct comparison possible between our method and atom-based generation, we further developed an atom-based translation model (AtomG2G) as baseline.

It makes three predictions in each generation step.

First, it predicts whether the decoding process has completed (no more new atoms).

If not, it creates a new atom a t and predicts its atom type.

Lastly, it predicts the bond type between a t and other atoms autoregressively to fully capture edge dependencies (You et al., 2018b) .

The encoder of AtomG2G encodes only the atom-layer graph and the decoder attention only sees the atom vectors c G X .

All translation models are trained under the same variational objective ( §3.3).

Single-property Optimization Results As shown in Table 1 , our model achieves the new state-ofthe-art on the four translation tasks.

In particular, our model significantly outperforms JTNN in both translation accuracy (e.g., 76.9% versus 59.9% on the QED task) and output diversity (e.g., 0.564 versus 0.480 on the logP task).

While both methods generate molecules by structures, our decoder is autoregressive which can learn more expressive mappings.

More importantly, our model runs 6.3 times faster than JTNN during decoding.

Our model also outperforms AtomG2G on three datasets, with over 10% improvement on the DRD2 task.

This shows the advantage of our hierarchical model.

For this task, we compare our method with other translation methods: Seq2Seq, JTNN and AtomG2G.

All these models are trained under the conditional translation setup where we feed the desired criteria g X,Y as input.

As shown in Table 2a , our model outperforms other models in both translation accuracy and output diversity.

Notably, all models achieved very low success rate on c = [1, 1] because it has the strongest constraints and only 1.6K of the training pairs satisfy this criteria.

In fact, training our model on the 1.6K examples only gives 4.2% success rate as compared to 13.0% when trained with other pairs.

This shows our conditional translation setup can transfer the knowledge from other pairs with g X,Y = [1, 0], [0, 1].

Ablation Study To understand the importance of different architecture choices, we report ablation studies over the QED and DRD2 tasks in Table 2b .

We first replace our hierarchical decoder with atom-based decoder of AtomG2G to see how much the structure-based decoding benefits us.

We keep the same hierarchical encoder but modified the input of the decoder attention to include both atom and substructure vectors.

Using this setup, the model performance decreases by 0.8% and 10.9% on the two tasks.

We suspect the DRD2 task benefits more from structure-based decoding because biological target binding often depends on the presence of specific functional groups.

Our second experiment reduces the number of hierarchies in our encoder and decoder MPN, while keeping the same hierarchical decoding process.

When the top substructure layer is removed, the translation accuracy drops slightly by 0.8% and 2.4%.

When we further remove the attachment layer, the performance degrades significantly on both datasets.

This is because all the substructure information is lost and the model needs to infer what substructures are and how substructure layers are constructed for each molecule.

Implementation details of those ablations are shown in the appendix.

Lastly, we replaced our LSTM MPN with the original GRU MPN used in JTNN.

While the translation performance decreased by 4% and 2.2%, our method still outperforms JTNN by a wide margin.

Therefore we use the LSTM MPN architecture for both HierG2G and AtomG2G baseline.

In this paper, we developed a hierarchical graph-to-graph translation model that generates molecular graphs using chemical substructures as building blocks.

In contrast to previous work, our model is fully autoregressive and learns coherent multi-resolution representations.

The experimental results show that our method outperforms previous models under various settings.

A ADDITIONAL FIGURES

The message passing network MPN ψ (H, {x u }, {x uv }) over graph H is defined as:

Algorithm 3 LSTM MPN with T message passing iterations

wu } w∈N (u)\v for all edges (u, v) ∈ H simultaneously.

end for Return node representations

end function Attention Layer Our attention layer is a bilinear attention function with parameter θ = {A θ }:

Figure 7: Illustration of AtomG2G decoding process.

Atoms marked with red circles are frontier nodes in the queue Q. In each step, the model picks the first node v t from Q and predict whether there will be new atoms attached to v t .

If so, it predicts the atom type of new node u t (atom prediction).

Then the model predicts the bond type between u t and other nodes in Q sequentially for |Q| steps (bond prediction, |Q| = 2).

Finally, it adds the new atom to the queue Q.

AtomG2G Architecture AtomG2G is an atom-based translation method that is directly comparable to HierG2G.

Here molecules are represented solely as molecular graphs rather than a hierarchical graph with substructures.

Table 3 : Training set size and substructure vocabulary size for each dataset.

Data The single-property optimization datasets are directly downloaded from the link provided in Jin et al. (2019) .

The training set and substructure vocabulary size for each dataset is listed in Table 3 .

We constructed the multi-property optimization by combining the training set of QED and DRD2 optimization task.

The test set contains 780 compounds that are not drug-like and DRD2-inactive.

The training and test set is attached as part of the supplementary material.

Hyperparameters For HierG2G, we set the hidden layer dimension to be 270 and the embedding layer dimension 200.

We set the latent code dimension |z| = 8 and KL regularization weight λ KL = 0.3.

We run T = 20 iterations of message passing in each layer of the encoder.

For AtomG2G, we set the hidden layer and embedding layer dimension to be 400 so that both models have roughly the same number of parameters.

We also set λ KL = 0.3 and number of message passing iterations to be T = 20.

We train both models with Adam optimizer with default parameters.

For CG-VAE (Liu et al., 2018) , we used their official implementation for our experiments.

Specifically, for each dataset, we trained a CG-VAE to generate molecules and predict property from the latent space.

This gives us three CG-VAE models for logP, QED and DRD2 optimization tasks, respectively.

At test time, each compound X is translated following the same procedure as in Jin et al. (2018) .

First, we embed X into its latent representation z and perform gradient ascent over z to maximize the predicted property score.

This gives us z 1 , · · · , z K vectors for K gradient steps.

Then we decode K molecules from z 1 , · · · , z K and select the one with the best property improvement within similarity constraint.

We found that it is necessary to keep the KL regularization weight low (λ KL = 0.005) to achieve meaningful results.

When λ KL = 1.0, the above gradient ascent procedure always generate molecules very dissimilar to the input X.

Ablation Study Our ablation studies are illustrated in Figure 8 .

In our first experiment, we changed our decoder to the atom-based decoder of AtomG2G.

As the encoder is still hierarchical, we modified the input of the decoder attention to include both atom and substructure vectors.

We set the hidden layer and embedding layer dimension to be 300 to match the original model size.

Our next two experiments reduces the number of hierarchies in both our encoder and decoder MPN.

In the two-layer model, molecules are represented by c X = c G X ∪ c A X .

We make topological and substructure predictions based on hidden vector h A k instead of h S k because the substructure layer is removed.

In the one-layer model, molecules are represented by c X = c G X and we make topological and substructure predictions based on atom vectors v∈S k h v .

The hidden layer dimension is adjusted accordingly to match the original model size.

<|TLDR|>

@highlight

We propose a multi-resolution, hierarchically coupled encoder-decoder for graph-to-graph translation.

@highlight

A hierarchical graph-to-graph translation model to generate molecular graphs using chemical substructures as building blocks that is fully autoregressive and learns coherent multi-resolution representations, outperforming previous models.

@highlight

The authors present a hierarchical graph-to-graph translation method for generating novel organic molecules.