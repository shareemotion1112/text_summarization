Estimating the importance of each atom in a molecule is one of the most appealing and challenging problems in chemistry, physics, and material engineering.

The most common way to estimate the atomic importance is to compute the electronic structure using density-functional theory (DFT), and then to interpret it using domain knowledge of human experts.

However, this conventional approach is impractical to the large molecular database because DFT calculation requires huge computation, specifically, O(n^4) time complexity w.r.t.

the number of electrons in a molecule.

Furthermore, the calculation results should be interpreted by the human experts to estimate the atomic importance in terms of the target molecular property.

To tackle this problem, we first exploit machine learning-based approach for the atomic importance estimation.

To this end, we propose reverse self-attention on graph neural networks and integrate it with graph-based molecular description.

Our method provides an efficiently-automated and target-directed way to estimate the atomic importance without any domain knowledge on chemistry and physics.

In molecules, each atom has the importance in manifesting the entire molecular properties, and estimating such atomic importance plays a key role in interpreting molecular systems.

For these reasons, the atomic importance estimation has been consistently studied in the scientific communities (Yen & Winefordner, 1976; Tang et al., 2016; Pan et al., 2018) .

However, estimating the atomic importance is one of the most challenging tasks in chemistry and quantum mechanics because the importance of each atom is comprehensively determined based on atomic properties, neighbor atoms, bonding types, and target molecular property.

The most common approach for estimating the atomic importance is to interpret the electronic structure using density-function theory (DFT) (Sholl & Steckel, 2009) .

In this approach, the atomic importance is estimated through three steps: 1) A human expert selects appropriate functional and basis sets for a given molecule to apply DFT; 2) The electronic structure of the molecule is calculated based on DFT calculation; 3) The human expert estimates the atomic importance by interpreting the calculated electronic structure in terms of target molecular property.

Although some methods are developed to estimate relative contributions of atoms in molecules, their generality is typically limited to the molecular properties (Marenich et al., 2012; Glendening et al., 2019) .

For this reason, DFT that can generate a general description of the molecule has been most widely used to interpret the molecular systems and to reveal important atoms for target molecular property (Crimme et al., 2010; Lee et al., 2018b; Chibani et al., 2018) .

However, the conventional approach based on DFT has three fundamental limitations in efficiency, automation, and generality.

• Efficiency: As an example of the electronic structure computations, DFT calculation requires O(n 4 ) time complexity to compute the electronic structure, where n is the number of basis functions that describe electrons in the molecule.

Generally, molecules have more electrons than atoms.

• Automation: DFT cannot automatically generate all target-specified physical properties in principle, so human expert should manually select additional calculation method to com-pute target molecular property from the electronic distributions.

That is, domain knowledge of the human experts is necessarily required to estimate the atomic importance in terms of the target molecular property.

• Generality: For some molecular properties, the relationship between them and the electronic distributions is not clear.

Moreover, sometimes the estimation is impossible because the relationships between molecular property and molecular structure are not interpretable.

For these limitations, estimating the atomic importance is remaining as one of the most challenging problems in both science and engineering such as physics, chemistry, pharmacy, and material engineering.

To overcome the limitations of the conventional approach in estimating the atomic importance, we first exploit machine learning-based approach.

To this end, we propose a new concept of reverse self-attention and integrate it with the graph neural networks.

The self-attention mechanism was originally designed to determine important elements within the input data to accurately predict its corresponding target or label in natural language processing (Vaswani et al., 2017) .

Similarly, in graph neural networks, self-attention is used to determine important neighbor nodes within the input graph to generate more accurate node or graph embeddings (Velickovic et al., 2018) .

Our reverse self-attention is defined as the inverse of the self-attention to calculate how important a selected node is considered in the graph.

For a given molecule and target property, the proposed estimation method selects the atom that has the largest reverse self-attention score as the most important atom.

The proposed method estimates the target-directed atomic importance through two steps: 1) For the given molecular graphs and their corresponding target properties, self-attention-based graph neural network is trained; 2) After the training, the reverse self-attention scores are calculated, and then the atomic importance is estimated according to the reverse self-attention scores.

As shown in this estimation process, neither huge computation nor human experts in chemistry and physics is required.

Thus, the proposed method provides an efficient and fully-automated way to estimate the atomic importance in terms of the target molecular property via target-aware training of the graph self-attention.

To validate the effectiveness of the proposed method, we conducted comprehensive experiments and evaluated the estimation performance using both quantitative and qualitative analyses.

The contributions of this paper are summarized as:

• This paper first proposes a machine learning-based approach to estimate the atomic importance in the molecule.

• The proposed method drastically reduced the computational cost for the atomic importance estimation from O(n 4 ) time complexity to the practical time complexity of the graph-based deep learning.

• The proposed method provides a fully-automated and target-directed way to estimate the atomic importance.

• We comprehensively validated the effectiveness of the proposed method using both quantitative and qualitative evaluations.

However, since none of a labeled dataset for the atomic importance estimation and a systematic way to quantitatively evaluate the estimation accuracy, we devised a systematic quantitative evaluation method and validated the effectiveness of the proposed method using it.

Before describing the atomic importance estimation based on the reverse self-attention, we briefly two essential concepts for understanding the proposed method in this section: 1) graph-based molecular analysis; 2) graph self-attention and graph attention network.

Graph-based molecular machine learning has attracted significant attention from both scientific and machine learning communities.

It has shown state-of-the-art performance in various scientific applications such as molecular or crystal property prediction (Wu et al., 2018; Xie & Grossman, 2018; Lu Figure 1: Overall process of the graph-based molecular machine learning.

In the molecular graph, three and two-dimensional vectors mean atom-features and bond-features, respectively.

et al., 2019), molecular generation Gao & Ji, 2019) , and atomic reaction analysis (Coley et al., 2019) .

In the graph-based molecular machine learning, a molecule is represented as an undirected feature graph G = (V, A, X, U ), where V is a set of atoms (nodes); A ∈ {0, 1} |V|×|V| is an adjacency matrix indicating the existence of the atomic bonds (edges); X ∈ R |V|×d is a ddimensional atom-feature matrix; U ∈ R |B|×m is a m-dimensional bond-feature matrix; and B is a set of atomic bonds.

For a given labeled molecular dataset D = {(G 1 , y 1 ), ..., (G |D| , y |D| )}, the goal of the graph-based molecular machine learning is to build a model f : G → y, where y can be target value, class label, or another molecular graph.

In this paper, we will refer atom and bond as node and edge, respectively.

Fig. 1 shows the overall process of the graph-based molecular machine learning.

First, molecular or crystal structures formatted by .xyz or .cif are converted into the molecular graph with the adjacency matrix A, node-feature matrix X, and edge-feature matrix U .

Then, graph neural network generates the graph-level embedding of the input molecular graph through the aggregation layers and predict the corresponding target y.

In the aggregation layer of the graph neural network, each node in the graph is converted into the node-embedding vector, and it is stacked to the node-embedding matrix in column-wise.

The output node-embedding matrix of the k th aggregation layer, H (k) , is calculated by:

where ψ is an aggregation function, and W (k) is a trainable weight matrix of the k th aggregation layer.

Note that H (0) is the input node-feature matrix X. For example, in graph convolutional network (GCN) (Kipf & Welling, 2017) , H (k) , is calculated by:

whereÃ is an adjacency matrix containing self-loop and can be calculated byÃ = A + I. After the node-embedding, the node-embedding matrix of the last aggregation layer is converted into the graph-level embedding vector by readout function (Zhou et al., 2018) .

For the implementation of the readout function, mean or max-based operations are commonly used (Lee et al., 2018a; .

Finally, the target corresponding the input graph is predicted by interpreting the output graph-level embedding vector through the fully-connected layers.

The attention and the self-attention mechanisms are originally introduced in natural language processing to refer other data and elements to improve prediction and classification accuracies, respectively (Luong et al., 2017; Vaswani et al., 2017) .

In graph neural networks based on neighbor node aggregation approach such as GCN, the self-attention mechanism is used to calculate the importance of neighbor nodes in generating the node-embedding vectors.

Graph attention network (GAT) (Velickovic et al., 2018 ) is the first neural network to apply the self-attention mechanism in graph-based deep learning.

By exploiting the self-attention mechanism, GAT showed highly improved prediction and classification accuracies (Velickovic et al., 2018; .

In the k th aggregation layer of GAT, the attention coefficient between node i and its neighbor node j is defined by:

where f is a feedforward neural network; V is a trainable weight matrix; and ⊕ is a vector concatenation.

Based on the attention coefficient, the attention score, α (k) ij , is calculated by:

where N(i) is a set of neighbor nodes of node i in the graph.

Finally, in the k th aggregation layer of GAT, the node-embedding vector of node i is calculated by:

In GAT, the aggregation layer that generates the node-embeddings based on the graph self-attention is called graph attention layer.

In this section, we explain our machine learning-based approach for estimating the atomic importance.

To devise the machine learning-based importance estimator, we define a new concept of reverse self-attention and integrate it with GAT.

The self-attention mechanism in the molecular graph provides numerical importance of each neighbor atom in terms of target molecular property.

To exploit the concept of the self-attention for estimating the atomic importance, reverse self-attention in the k th graph attention layer is defined as the inverse of the self-attention: ρ

That is, the reverse self-attention of node i means how much attention node i receives from its neighbor nodes.

Fig. 2 shows an example of the reverse self-attention in the graph with five nodes.

In this example, the reverse self-attention of n 1 is calculated as the sum of α 21 + α 31 + α 41 .

As shown in the definition of the reverse self-attention, the additional time and space complexities of the reverse self-attention are negligible because calculating the reverse self-attention is just the sum of pre-computed self-attention scores in prediction or classification time.

In GAT, the self-attention scores of the k th graph attention layer are calculated based the nodeembeddings of the (k − 1) th graph attention layer.

That is, the self-attention scores of the first layer are calculated by considering 1-hop neighbor nodes, and the second layer's self-attention scores are determined by considering 2-hop neighbor nodes.

Thus, the reverse self-attention in the first Figure 3 : Overall process of building machine learning-based atomic importance estimator with the reverse self-attention mechanism and selecting important atom or group of atoms using the estimator.

graph attention layer indicates the importance of each atom itself, and the second layer's reverse self-attention means the importance of each atom group consists of 1-hop neighbor atoms, not an atom itself.

These interpretations can be extended to the atomic importance for the group of atoms that consists of k-hop neighbor atoms by the reverse self-attention in the (k + 1) th graph attention layer.

This section explains the way to build a fully-automated and target-directed atomic importance estimator based on the reverse self-attention and GAT.

We call this estimator Machine Intelligencebased Atomic Importance Estimator (MIAIE).

Fig. 3 shows overall process of building MIAIE and estimating the target-directed atomic importance via MIAIE.

The atomic importance estimation using MIAIE consists of four steps:

• Step 1: Train GAT for regression or classification on the molecular dataset with the target molecular property.

In this training process, GAT automatically learns the self-attentions scores in terms of predicting the target molecular property.

• Step 2:

Predict the target property of the interesting molecule and extract self-attention scores for each graph attention layer.

• Step 3: Calculate the reverse self-attention scores using Eq. (6) and sort them for each graph attention layer.

• Step 4: Select atom or group of atoms that have the largest reverse self-attention score as the most important element in the molecule in terms of the target property.

As shown in the overall process of estimating the atomic importance using MIAIE, the estimation process is fully-automated and does not require any domain knowledge of the human experts in chemistry and physics.

MIAIE is also incomparably efficient than the conventional approach using DFT calculations with O(n 4 ) time complexity because it uses only the graph neural networks to describe the molecules, and the time complexity of the estimation process (step 3 and 4) is negligible.

To accurately validate the effectiveness of MIAIE, we conducted both quantitative and qualitative evaluations on two well-known molecular datasets.

However, to the best of our knowledge, neither a labeled dataset for the atomic importance estimation nor a systematic way to evaluate the performance of the atomic importance estimator exists.

For this reason, we devised a validation method to quantitatively evaluate the performance of the atomic importance estimators.

We will explain this validation method in Section 4.3.

We used MolView 1 to visualize the estimation results of MIAIE.

For the experiments, we used two well-known molecular datasets: Quantum Mechanics9 (QM9) (Ramakrishnan et al., 2014; Ruddigkeit et al., 2012) and Estimated SOLubility (ESOL) (Delaney, 2004) .

These datasets were randomly split into 90% train and 10% test subsets because the test dataset for them are not provided.

In the experiments, we validated the effectiveness of MIAIE by splitting train and test datasets to evaluate the generalization capability as well as the atomic importance estimation accuracy.

However, we can use MIAIE by fitting it to a given dataset only without considering the generalization capability-for example, training without L 2 -regularization (Krogh & Hertz, 1992) , dropout (Srivastava et al., 2014) , or batch normalization (Ioffe & Szegedy, 2015) if our analysis is only focused on specific molecular datasets containing the interesting molecules.

To quantitatively evaluate the effectiveness of MIAIE, QM9 dataset is used.

It is a subset of GDB-17 database (Ramakrishnan et al., 2014; Ruddigkeit et al., 2012) , in which the structural information of the molecules is given in Cartesian coordinates.

QM9 dataset contains 133,886 organic molecules and several target molecular properties such as highest occupied molecular orbital (HOMO), lowest unoccupied molecular orbital (LUMO), and their gap (HOMO-LUMO gap).

In the experiments, we used HOMO-LUMO gap as the target molecular property because it is an essential property describing the molecular systems and one of the difficult properties to be interpreted from the molecular structure.

Molecules in QM9 dataset have 0.0245∼0.6221 HOMO-LUMO gaps.

For the qualitative analysis, we used ESOL dataset that contains aqueous solubility as a target property because the important atoms in terms of aqueous solubility can be easily determined by a human expert.

ESOL dataset was originally published with the aqueous solubility of 2,874 compounds but a smaller subset of 1,128 compounds is recently used in chemistry (Wu et al., 2018) , and we also used the subset of ESOL for the experiment.

Unlike QM9 dataset, the structural information of the molecules in ESOL is provided by SMILES (Weininger, 1988) representation that does not present hydrogen.

For this reason, hydrogen in the molecule is ignored in estimating the atomic importance.

We implemented MIAIE using PyTorch 2 and PyTorch Deep Graph Library (DGL) 3 .

GAT in MI-AIE is also implemented based on the neural network modules of DGL.

In the experiments, we used GAT with two graph attention layers and two fully-connected layers.

To generate graphlevel embedding, mean-based readout function is applied.

Mean squared error (MAE) was used as an objective function to train GAT, and L 2 -regularization with 0.001 regularization coefficient was applied to improve the generalization capability of the model.

As a training algorithm, Adam SGD (Kingma & Ba, 2015) was used with an initial learning rate of 0.001 to fit model parameters of GAT.

To accelerate the training and improve the prediction performance of GAT, we concatenated additional molecular features about molecular weight and the number of rings to the graphlevel embedding vector.

The source code of MIAIE and the experiment scripts are available at http://----------------------(open after the review).

To quantitatively evaluate the effectiveness of MIAIE, we assume that if the selected atom or group of atoms are truly important in terms of the target property, then the gap between the target properties of the original molecule and its selected sub-molecule (atom or group of atoms) will be small.

Based : An original molecule made of the carbon ring and its selected sub-molecule that consists of 1-hop neighbor atoms.

Note that hydrogens are automatically attached to make a valid molecular structure.

The group of atoms that is marked by a red mask means the most important atomic group estimated by MIAIE.

White node: hydrogen (H); Gray node: carbon (C).

on this assumption, we define atomic importance estimation error (AIEE) as:

where G i is a sub-molecule of G i selected by an atomic importance estimator; DF T means the value of the target property of G i computed by DFT calculations; and size(G i ) is the number of atoms in G i .

Since DFT can accurately calculate the value of the molecular properties in most cases even though it requires huge computation, we used DFT to calculate the value of the target molecular property of unknown G i .

In AIEE, the error of each molecule is divided by the number of atoms in the molecule.

It is an essential part of AIEE to accurately measure the target property gaps because the larger the size of the original molecule, the greater the probability of increasing the gap between the properties of the original molecule and its smaller sub-molecule with a fixed size.

In this experiment, we quantitatively evaluated the effectiveness of MIAIE using AIEE in Eq. (7) on QM9 dataset.

We estimated the atomic importance using MIAIE with the reverse self-attention of the second graph attention layer for 13,3889 test molecules in terms of HOMO-LUMO gap and observed 0.00544 on AIEE.

This error is relatively small when considering the fact that the molecular properties of the original molecule cannot be preserved completely in sub-molecules.

However, we observed a relatively large error of 0.01007 compared to the mean of error (0.00544) on the molecule made of the carbon ring as shown in Fig. 4 .

This large error is caused due to the chemical characteristics of HOMO-LUMO gap that it is determined by the overall electronic distributions in ring-shaped molecules with the same atoms such as Fig. 4-(a) .

Thus, this large error in HOMO-LUMO gap is inevitable in splitting a ring-shaped molecule with the same atoms into the sub-molecules because each atom has similar importance.

It is a limitation of our validation method for the quantitative evaluation.

One possible future work is to modify the estimation step of MIAIE to answer "each atom has similar importance" instead of selecting important atom for the molecules like Fig. 4 -(a).

For ESOL dataset, we conducted a qualitative evaluation because the important atom or group of atoms in terms of aqueous solubility can be easily determined by human experts.

Furthermore, we denoted the normalized atomic importance of the most important group of atoms, which is selected by MIAIE.

Due to the limitation of the space, we present only three natural and two interesting estimation results in this paper.

can be easily justified: 1) Since nitrogen and oxygen exist only in the selected sub-molecule, the estimation result of 2-Nitropropane is natural; 2) Similar to the result of 2-Nitropropane, MIAIE accurately selected a group of atoms that contains the largest number of nitrogen and oxygen in Metronidazole; 3) In Indoline, only three sub-molecules that consists of 1-hop neighbor atoms can contain the nitrogen.

However, among these sub-molecules, the selected group of atoms has the largest electronic charge that improve the reactivity to water, so the estimation result of MIAIE is also chemically resonable.

Some interesting results are and (e).

Benfluralin in Fig. 5 -(d) has a low aqueous solubility of -5.53 in log-scale because the C-CF 3 group is attached (Purser et al., 2008) .

Interestingly, although we did not tell this chemical observation about the relationship between the C-CF 3 group and aqueous solubility to MIAIE, it exactly selected the C-CF 3 group as the most important group of atoms in terms of aqueous solubility.

Another interesting result is Coumaphos in Fig. 5 -(e).

In this molecule, the C-CO 2 group was selected as the most important group of atoms rather than PO 3 S group that contains the largest number of oxygen.

However, this estimation result is chemically reasonable because the carbons surrounding the PO 3 S group prevent the reaction between the PO 3 S group and water.

In these estimation results, we observed that MIAIE can estimate the atomic importance by considering complex chemical rules related to the functional properties of the atomic groups and the overall environment of the molecules, even though it is trained without any domain knowledge of chemistry and quantum mechanics.

To check the correctness of the estimation result of MIAIE, we also measured the atomic importance on the molecules that have the symmetric structure because the structurally-equivalent groups of atoms must have the same atomic importance.

Fig. 6 shows the estimation results of MIAIE for the molecules containing the symmetric structure.

Benfluralin has a locally-symmetric structure in two C-NO 2 groups, and MIAIE correctly estimated the atomic importance of two C-NO 2 groups as the same value.

Similarly, the C 24 O 4 molecule has globally-symmetric structure throughout the overall molecular structure.

For the C 24 O 4 molecule, MIAIE correctly estimated two C-CO 2 groups as the most important group of atoms with the same score.

Figure 6: Two original molecules with symmetric structure and their selected sub-molecules in terms of aqueous solubility.

The group of atoms that is marked by a blue mask means the second most important atomic group estimated by MIAIE.

This paper first exploited machine learning approach to estimate the atomic importance in molecules.

To this end, the reverse self-attention was proposed and integrated with graph attention network.

The proposed method is efficient and fully-automated.

Furthermore, it can estimate the atomic importance in terms of the given target molecular property without human experts.

However, the proposed method can estimate the importance of the group of atoms that consists of k-hop neighbor atoms only, even though some important group of atoms may have an arbitrary shape such as ring and bar.

As the future work, it is necessary to modify the proposed method to estimate the importance of the group of atoms with arbitrary shape.

Fig. 7 shows the original molecules and their selected sub-molecules with the extremely small error.

As shown in the results, even though the molecules have carbon rings, the important group of atoms was correctly captured because the molecules are characterized by nitrogen or oxygen.

On the other hand, Fig. 8 shows the original molecules and their sub-molecules with the extremely large error.

Chemically, double bond plays an important role in determining HOMO-LUMO gap.

However, as shown in the results of Fig. 8 , sub-structures that do not contain the double bonds are selected as an important group of atoms.

Thus, we need to develop a descriptor for the molecules that can emphasis the bond-features more strongly.

<|TLDR|>

@highlight

We first propose a fully-automated and target-directed atomic importance estimator based on the graph neural networks and a new concept of reverse self-attention.