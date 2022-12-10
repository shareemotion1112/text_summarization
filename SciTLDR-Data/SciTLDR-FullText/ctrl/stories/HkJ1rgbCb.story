Deep learning algorithms are increasingly used in modeling chemical processes.

However, black box predictions without rationales have limited used in practical applications, such as drug design.

To this end, we learn to identify molecular substructures -- rationales -- that are associated with the target chemical property (e.g., toxicity).

The rationales are learned in an unsupervised fashion, requiring no additional information beyond the end-to-end task.

We formulate this problem as a reinforcement learning problem over the molecular graph, parametrized by two convolution networks corresponding to the rationale selection and prediction based on it, where the latter induces the reward function.

We evaluate the approach on two benchmark toxicity datasets.

We demonstrate that our model sustains high performance under the additional constraint that predictions strictly follow the rationales.

Additionally, we validate the extracted rationales through comparison against those described in chemical literature and through synthetic experiments.

Recently, deep learning has been successfully applied to the development of predictive models relating chemical structures to physical or biological properties, outperforming existing methods BID8 BID14 .

However, these gains in accuracy have come at the cost of interpretability.

Often, complex neural models operate as black boxes, offering little transparency concerning their inner workings.

Interpretability plays a critical role in many areas including cheminformatics.

Consider, for example, the problem of toxicity prediction.

Over 90% of small molecule drug candidates entering Phase I trials fail due to lack of efficacy or due to adverse side effects.

In order to propose a modified compound with improved properties, medicinal chemists must know which regions of the molecule are responsible for toxicity, not only the overall level of toxicity BID1 .

We call the key molecular substructures relating to the outcome rationales.

In traditional cheminformatics approaches such as pharmacophore mapping, obtaining such a rationale behind the prediction is an intrinsic part of the model BID24 BID7 BID12 In this paper, we propose a novel approach to incorporate rationale identification as an integral part of the overall property prediction problem.

We assume access to the same training data as in the original prediction task, without requiring annotated rationales.

At the first glance, the problem seems solvable using existing tools.

For instance, attention-based models offer the means to highlight the importance of individual atoms for the target prediction.

However, it is challenging to control how soft selections are exploited by later processing steps towards the prediction.

In this sense, the soft weighting can be misleading.

In contrast, hard selection confers the guarantee that the excluded atoms are not relied upon for prediction.

The hard selection of substructures in a molecule is, however, a hard combinatorial problem.

Prior approaches circumvent this challenge by considering a limited set of predefined substructures (typically of 1-6 atoms), like the ones encoded in some molecular fingerprints BID7 .

Ideally, we would like the model to derive these structures adaptively based on their utility for the target prediction task.

We formulate the problem of selecting important regions of the molecule as a reinforcement learning problem.

The model is parametrized by a convolutional network over a molecular graph in which the atoms and bonds are the nodes and edges of the graph, respectively.

Different from traditional reinforcement learning methods that have a reward function provided by the environment, our model seeks to learn such a reward function alongside the reinforcement learning algorithm.

More generally, our model works as a search mechanism for combinatorial sets, which readily expands to applications beyond chemistry or graphs.

Our iterative construction of rationales provides several advantages over standard architectures.

First, sequential selection enables us to incorporate contextual features associated with past selections, as well as global properties of the whole molecule.

Second, we can explicitly enforce desirable rationale properties (e.g., number of substructures) by including appropriate regularization terms in the reward function.

We test our model on two toxicity datasets: the Tox21 challenge dataset, which is a series of 12 toxicity tests, and the human ether-a-go-go-related gene (hERG) channel blocking.

The reinforcement learning model identifies the structural components of the molecule that are relevant to these toxicity prediction tasks while simultaneously highlighting opportunities for molecular modification at these sites.

We show that by only selecting about 40-50% of the atoms in the molecules, we can create models that nearly match the performance of models that use the entire molecule.

By comparing selected regions with rationales described in chemical literature, we further validate the rationales extracted by the model.

Deep Learning for Computing Chemical Properties One of the major shifts in chemical property prediction is towards the use of deep learning.

The existing models fall into one of two classes.

The first class of models is based on an expert-constructed molecular representation such as fingerprints that encapsulate substructures thought to be important and a range of molecular properties BID33 BID26 .

These models are not well suited for extracting rationales because desired structures may not be part of the fingerprint.

Moreover, it may be challenging to attribute properties recorded in fingerprints to specific substructures in the molecule.

One would have to restrict the feature space of the fingerprint, which can harm the performance of the model.

The second class of models move beyond traditional molecular fingerprints, instead learning tasktailored representations.

Specifically, they employ convolutional networks to learn a continuous representation of the molecule BID8 BID14 .

BID10 's work takes this a step further, and uses the Weisfeiler-Lehman kernel inspired neural model as a way to generate better local representations.

Following this direction, our work is also based on learned molecular representation.

Our focus, however, is on augmenting these models with rationales.

As articulated in the introduction, the task is challenging due to the number of candidate substructures and the need to attribute properties aggregated in convolutions to individual atoms.

Reinforcement Learning on Graphs Our work utilizes a similar framework as the reinforcement learning model over graphs described by BID5 .

However, their work focuses on solving computational problems where the reward is provided by the environment (i.e., a deterministically computable property of the graph, such as max-cut or the traveling salesman problem).

In contrast, in our formulation, the rewards are also learned by the system.

Another related work in this area BID18 ) utilizes a policy gradient method to search for substructures starting from random points in the graph.

Since their model is designed for large graphs that do not fit into memory, it does not consider the global context of the molecule as a whole.

This design is limiting in the context of molecular prediction, where such information is valuable for prediction, and where it is also feasible to compute since the graphs are small.

Moving away from the convolutional network approach, their model imposes an artificial ordering of the nodes through a sequence network as the reinforcement learning agent traverses the graph.

Both of the above models focus on the prediction task, whereas the emphasis of our work is on interpretability.

Learning Rationales The topic of interpretability has recently gained significant attention BID20 BID15 .

The proposed approaches cover a wide spectrum in terms of the desired rationales and the underlying methods.

Work in this area include visualization of activations in the network BID11 BID9 , and examination of the most influential data examples to provide interpretability BID16 .

Attention-based models have also been widely used to extract interpretability (Ba et al., Figure 1: Our model makes sequential selections of atoms (light blue) in the molecule and is specified by two networks, the Q-Network and the P-Network.

The former constitutes the reinforcement learning agent that assigns a Q-value to each atom, and the latter takes the atom selections of the Q-Network and trains a classifier to predict based solely on those atoms.

This prediction is used as a reward that is fed back to the Q-Network.2014; BID4 BID25 BID31 BID32 .

Our work is mostly closely related to approaches focused on extractive rationales BID19 BID27 .

BID19 present a model to extract parts of text as rationale, but their model does not readily generalize to graphs, and the sequential nature of our model can place a meaningful ordinal ranking over the atom selections.

Our approach uses reinforcement learning as a method to iteratively select important atoms from the graph.

We use a Q-learning approach similar to that of BID5 .

The state of the system at time t corresponds to the atoms selected thus far.

The agent takes an action a t at each time step t, where the action is the selection of an atom that has not already been selected.

After the agent takes an action, the state s t is updated to include the newly selected atom.

Unlike traditional reinforcement learning algorithms in which the agent receives a reward from the environment, we use a separate model (P-Network) to generate the reward r t to the agent.

The P-network learns to predict molecular properties such as toxicity based on the selected atoms, and rewards the agent according to the accuracy of its predictions.

The agent itself learns a Q-Network to represent the action-value function Q(s, a) needed to maximize the reward throughout the process.

In this case, maximizing the reward is equivalent to selecting atoms that help P-Network make accurate predictions based on the partial selections of atoms.

The overall approach is illustrated in Figure 1 .In the following sections, we will describe the two networks, the Q-Network that guides the agent's actions, and the P-Network that maps agent's selections to predicted molecular properties.

The two networks are trained iteratively, in a feedback loop, so that the Q-network can provide good selections of atoms that, when fed to the P-Network, result in good overall predictions of molecular properties.

Both networks are parametrized by the graph convolutional network that we describe separately below.

In a convolutional network, the model updates the feature representation of individual atoms iteratively based on local contexts.

The nature of the operations is parameterized and can be learned to support the end-to-end task.

We prefer convolutional networks due to their expressiveness and adaptability as compared to traditional molecular fingerprint methods.

Define a molecular graph as G = (A, B), in which A is the set of nodes denoting the atoms, and B is the set of edges denoting the bonds between atom pairs.

In each successive layer of the network, an atom's representation is updated to incorporate the currently available information from its neighbors, while the bond features remain unchanged.

These updates propagate information across the molecule, and allow the network to generalize beyond standard local substructures.

Let h l i be a vector-valued atom representation for atom i at layer l, and let A i be the input feature vector for atom i, and B i,j be the input feature vector for the bond between atoms i and j. In this notation, we use h 0 i = A i . (The input atom features include: one-hot encoding of the atomic number, degree, valence, hybridization, aromaticity, and whether or not the atom is in a ring.

Bond features include: one-hot encoding of the bond order, aromaticity, and whether or not the bond is in a ring).

This initialization differs slightly for the Q-Network so as to incorporate the current state of selections into the convolutional architecture.

The update step for atom feature vectors involves a gated unit that receives separate contributions from the atom itself and its neighbors.

Specifically, DISPLAYFORM0 where N (i) is the set of neighbors of atom i, W l 's are the specific weight matrices that vary across layers (bias weights omitted for brevity), and σ is the sigmoid activation function, applied coordinate-wise.

After N iterations, we arrive at final atom representations h N t .

The Q-Network is parametrized by the convolutional network described in section 3.1, in which the size of the atom representation at the final layer is set to 1 so that h N i is scalar and interpreted as the Q-value for selecting atom i next.

The initial atom features in the Q-Network include the binary indicator of whether the atom has already been selected.

In other words, if we define the state s as a binary vector encoding the atoms that have already been selected, then we use an augmented h DISPLAYFORM0 The convolutional network is rerun with these initializations, using the same parameters, after each selection.

Thus, despite the fact that the selections are greedy with respect to the Q-values, the model will choose atoms in a manner that is aware of the global context of the molecule as well as state s representing already selected atoms.

The P-Network is a prediction network that takes the partial selection of atoms from the Q-Network, and makes a prediction about the label of the molecule using only those atoms selected by the Q-Network.

Like the Q-Network, the P-Network is separately parametrized by the convolutional network.

To incorporate the selected atoms, we zero out all the initial atom features that are not in the current selection before running the convolutional model.

It is important to note that we do not zero out the updated hidden states of these atoms.

This allows interaction between disjoint selections on the molecular graph and preserves information related to graph distances.

The reasoning behind this is that there are often several substructures of interest on the molecule and their interactions might prove important.

We want to allow the network to learn these interactions by facilitating the propagation of information throughout the whole molecule.

The P-Network is geared towards predicting molecular properties rather than atomic properties and therefore requires an additional aggregation of the atom vectors h N i .

In our model, these atom vectors are first renormalized via an attention mechanism: DISPLAYFORM0 and turned into a neural (adaptable) fingerprint by summing the resulting renormalized feature vectors f = iĥ N i .

This fingerprint is then passed through a sigmoid function to generate the class prediction,ŷ = σ(W f * f ).

The prediction loss is measured through standard cross-entropy loss L P = −y log(ŷ) − (1 − y) log(1 −ŷ) where y is the actual label andŷ is the predicted probability of y = 1.

The reward for the Q-network is induced by the P-network and defined as: DISPLAYFORM0 where θ refers to the parameters of the P-network and s t,· is the binary state vector updated from s t−1,· in response to action a t .

Because we are interested in selecting the important substructures of a molecule, which consist of at least several atoms, we found it useful to train the Q-network after n selections rather than a single selection.

The Q-Network is therefore trained using an n-step Q-learning algorithm, in which the network receives a signal for each sequence of n actions that it makes.

The loss function for the Q-network is then: DISPLAYFORM1 Where γ is a decay constant, Q t specifies the Q-value of the current state and action, and Q target t+n is the max Q-value of the state n steps from t induced by a separate but jointly trained target QNetwork.

During the training process, we use an -greedy search policy, where the agent will choose a random atom with probability instead of the one with the highest Q-value.

Keeping the idea of molecular substructures in mind, we find that it is helpful to search for a random neighbor of a selected atom, than a completely random atom.

We also employ action-replay using a target Q network that utilizes soft parameter updates in order to increase training stability BID23 .In our model, we place limits on the numbers of atoms to be selected.

This number is proportional to the number of atoms in the molecule up to a certain ceiling.

We note that fixing the number of atoms chosen by the agent is a limitation of our model, but seems to work well in practice.

Specifically, we find that taking 40-50% of the atoms up to a limit of 12-15 atoms for larger molecules works well, although this number varies with the problem.

We also impose a lower limit of 5 atoms for smaller molecules, because it becomes impossible to distinguish distinct molecules when too few atoms are chosen.

Additionally, we impose regularization constraints on the model, in order to enforce certain properties on the selections.

Since we are interested in the model selecting specific substructures from the molecule, we impose a penalty to the model for selecting too many disjoint groups.

That is, we define a variable C g t = # {disjoint groups of atoms at step t}. We then modify the reward equation 3 as follows: DISPLAYFORM2

Datasets We evaluate our model on two toxicity datasets.

The first dataset that we explore is the Tox21 challenge dataset which contains a series of 12 toxicity tests categorized by nuclear response signals and stress response pathways 1 .

We parse the data using RDKit BID17 , removing duplicate entries and cleaning the molecules by removing salts.

Because this dataset has data coming from multiple sources, there are conflicting labels for some molecules, which we remove from the dataset entirely.

TAB0 contains the information about each of the 12 datasets, highlighting the small number of positive examples in many of the datasets.

The second toxicity dataset that we evaluate our model on is the inhibition of the hERG channel 2 .

Because this protein channel is well-studied, we explore this dataset to see if we can create a predictive model that can generate rationales that match the information in chemical literature.

This dataset, taken from BID22 , consists of a training set with 3792 molecules, and a test set with 1090 molecules, with 25% positive labels.

Since the data was already cleaned, we do no further preprocessing of this dataset.

Evaluation Measures: Predictions For each dataset, we compare our model against the top reported systems BID26 ; BID22 .

These approaches utilize extensive feature engineering through molecular fingerprints and other computed molecular descriptors.

In addition, they use additional training sources for data augmentation.

Specifically, BID26 utilize a data augmentation method called kernel-based structural and pharmacological analoging (KSPA), which uses public databases containing similar toxicity tests.

We measure the predictive performance of the convolutional model (Mol-CNN, which utilizes the full molecule) to demonstrate that is comparable to the state-of-the-art results.

Next, we evaluate the performance of our reinforcement learning method (RL-CNN) that makes predictions on a fraction of atoms in the molecule.

We compare these different models using the AUC metric, since the datasets contain an unbalanced set of labels.

Evaluation Measures: Rationales Because the RL-CNN model makes predictions using only atoms selected as rationales, its quantitative performance indirectly measures the quality of these rationales.

However, we are also interested in directly evaluating their quality relative to rationales described in the chemical literature by domain experts.

In the ideal case, we would identify rationales that are characteristic to a single class of examples -either positive or negative.

Unfortunately, many known toxic substructures are prevalent in both positively and negatively labeled compounds.

In fact, BID26 show that adding features representing structural similarity to 2,500 common toxicophores (toxic substructures) to their model does not improve performance on the Tox21 challenge dataset.

This shows that the expert-derived "important" regions are not actually sufficient nor necessary for the prediction task.

Rationales extracted for the hERG dataset are directly compared with rationales described in the literature.

Multiple studies have shown that the presence of a protonable nitrogen atom and 3 hydrophobic cores has a high affinity for binding to this particular protein BID2 BID30 .

Usually, this nitrogen is secondary or tertiary so that it is more basic.

When protonated, the positive charge exhibits cation-pi interactions with certain residues of the hERG channel, which is the crux of the binding mechanism.

We show that our model can identify these basic nitrogen atoms within the dataset.

For a baseline comparison, we also evaluate rationales obtained by selecting atoms with the strongest influence on the logistic regression model prediction using Morgan fingerprints of radius 3 and length 2048.

Morgan fingerprints are boolean vectors constructed by enumerating atom-centered neighborhoods up to a certain radius, assigning integer values to each neighborhood by hashing a categorical feature vector of that neighborhood, and reducing those integers to vector indeces BID29 .

The importance of an atom can be approximated by the absolute difference between a prediction made with the full molecular fingerprint and a prediction made when substructures containing that atom are excluded from the fingerprint BID28 .

We restrict the baseline rationales to select the same number of atoms as in the RL-CNN model.

Evaluation Measures: Rationales (Synthetic Experiment) Since we do not find well-defined substructures in literature for the tests offered in the Tox21 dataset, we also construct a synthetic experiment.

For this experiment, we select specific substructures and set the labels of all molecules containing those substructures to be positive; all other molecules' labels are left unchanged.

We specifically focus on 3 toxic substructures: the aromatic diazo group, polyhalogenation, and the aromatic nitro group, two of which are from BID13 's work on toxicophores common to multiple toxicity assays.

We demonstrate that our model can capture important substructures if the data provides a clear enough signal.

Quantitative Evaluation of Convolutional Representation We first demonstrate that our convolutional model performs competitively compared to neural models using molecular fingerprints.

Columns four and five in TAB0 compare our results for Mol-CNN with the highest performing model DeepTox BID26 , both run in the multitask setting.

The DeepTox model performs better than our convolutional model by 2.3% on average across the twelve tasks.

This result is not surprising, as their method uses substantial amounts of additional training data which is not available to our model.

The results on the hERG dataset are summarized in Table 2 .

Our model outperforms the top performing model BID22 , which uses molecular fingerprints, on the external test set.

Quantitative Evaluation of Reinforcement Learning Algorithm For the Tox21 dataset, we use the multi-task instance to compare the results of our base convolutional model (Mol-CNN).

How- Table 2 : Results of different models on the hERG dataset using AUC as the metric.

The first 4 models are baselines from BID22 and use molecular fingerprints as input to random forest (RF), support vector machine (SVM), k nearest neighbors (KNN) and associative neural networks (ASNN).

For each of their models, we take the average performance for the same model run with different input features.

ever, we turn to the single-task instance to evaluate the performance of our rationale model.

This is due to the fact that different toxicity tests warrant different important substructures.

Therefore, we run individual models for each of the toxicity tests using the base convolution model as well as the reinforcement learning model.

We observe a small decrease in performance, resulting in a 0.7% decrease in AUC on average, using around 50% of the atoms in the dataset as seen in TAB0 .

On the hERG dataset, we selected 45% of the atoms, and also observe that the reinforcement learning algorithm performs similarly to the convolution network as seen in Table 2 , with a 3.4% decrease in AUC.

We see a smaller decrease in performance for the Tox21 datasets on average, likely because many of the datasets have comparatively few number of positive examples, so predicting on fewer atoms allows the model to generalize better.

Evaluation of Rationales using Human Rationales In the absence of ground truth rationales, we turn to a specific structural motif-a tertiary nitrogen atom-that is known to exhibit cation-pi interactions with residues of the hERG protein when copresent with certain hydrophobic side chains in the correct 3-D conformation BID6 BID2 .

In the dataset we used, these tertiary nitrogen substructure occurs more often in positive examples compared to negative examples (78.4 % vs 44.9 %).

This suggests that while this substructure is important in positive examples, it is not sufficient to indicate that a molecule is positive.

We observe that our model captures this important substructure frequently, and more often in positive examples than negative Figure 2 : Two examples of rationales selected by the reinforcement learning model.

The selected atoms are highlighted in large light blue circles.

In both cases, we see that the model selects the tertiary nitrogen motif, highlighted in small green circles, which is implicated in many inhibitors of the hERG channel.

examples (63.6 % vs 46.1 %).

Here, we require the model to have selected the nitrogen and at least two of its three carbon neighbors.

Figure 2 shows two example selections made by the model.

The similar statistical bias in this prediction demonstrates that the model can provide insights at least consistent with prior domain expertise.

In contrast, when the fingerprint baseline approach is used, the baseline model matches this substructure less frequently and with no discriminating power between positive and negative examples (19.1 % vs 23.9 %).Evaluation of Rationales using Synthetic Experiments Here, we evaluate how often our model would capture target substructures if those substructures were a sufficient indicator of our target task, toxicity.

TAB2 summarizes the results, and shows that our model can reliably identify them.

Examples of the generated rationales can be seen in Figure 3 .The baseline approach matches fewer instances of the aromatic diazo and polyhalogenation motifs, but does identify more of the aromic nitro groups.

Substructures symmetrically centered around a single atom -as in the nitro group -directly correspond to a fingerprint index and are well-described by the baseline model.

Correlation between adjacent atoms can cause the RL-CNN model to make an incomplete selection; that is, some of the initial atom features implicitly contain information about its neighbors, which leads to these neighbors appearing less important as rationales.

Simplifying the initial atom featurization to decrease this correlation causes the model to successfully select the aromatic nitro group in 17/17 cases.

Figure 3 : From left to right, example rationales generated for the dataset altered based on the presence of aromatic diazo group, polyhalogenation, and aromatic nitro group.

The selected atoms are highlighted in large light blue circles; the predefined toxicophores are highlighted in small green circles.

This confirms that while fingerprint models can do well when the relevant features happen to coincide with a fingerprint index, our rationale model is superior when the relevant features are less well-captured by the exact features of the fingerprint.

We present a model that treats the problem of selecting rationales from molecules as a reinforcement learning problem.

By creating an auxiliary prediction network, we use a learned reward structure to facilitate the selection of atoms in the molecule that are relevant to the prediction task, without significant loss in predictive performance.

In this work, we explore the applicability of rationales in the chemistry domain.

Through various experiments on the Tox21 and hERG datasets, we demonstrate that our model successfully learns to select important substructures in an unsupervised manner, requiring the same data as an end-to-end prediction task, which is relevant to many applications including drug design and discovery.

Molecules are far more complicated to reason about as compared to images or text due to complex chemical theories and a lack of definitive ground truth rationale labels.

As deep learning algorithms continue to permeate the chemistry domain, it will be ever more important to consider the interpretability of such models.

<|TLDR|>

@highlight

We use a reinforcement learning over molecular graphs to generate rationales for interpretable molecular property prediction.