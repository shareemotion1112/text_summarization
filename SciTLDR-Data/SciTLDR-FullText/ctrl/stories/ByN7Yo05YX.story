Deep neural networks and decision trees operate on largely separate paradigms; typically, the former performs representation learning with pre-specified architectures, while the latter is characterised by learning hierarchies over pre-specified features with data-driven architectures.

We unite the two via adaptive neural trees (ANTs), a model that incorporates representation learning into edges, routing functions and leaf nodes of a decision tree, along with a backpropagation-based training algorithm that adaptively grows the architecture from primitive modules (e.g., convolutional layers).

ANTs allow increased interpretability via hierarchical clustering, e.g., learning meaningful class associations, such as separating natural vs. man-made objects.

We demonstrate this on classification and regression tasks, achieving over 99% and 90% accuracy on the MNIST and CIFAR-10 datasets, and outperforming standard neural networks, random forests and gradient boosted trees on the SARCOS dataset.

Furthermore, ANT optimisation naturally adapts the architecture to the size and complexity of the training data.

Neural networks (NNs) and decision trees (DTs) are both powerful classes of machine learning models with proven successes in academic and commercial applications.

The two approaches, however, typically come with mutually exclusive benefits and limitations.

NNs are characterised by learning hierarchical representations of data through the composition of nonlinear transformations BID59 BID2 , which has alleviated the need for feature engineering, in contrast with many other machine learning models.

In addition, NNs are trained with stochastic optimisers, such as stochastic gradient descent (SGD), allowing training to scale to large datasets.

Consequently, with modern hardware, we can train NNs of many layers on large datasets, solving numerous problems ranging from object detection to speech recognition with unprecedented accuracy BID35 .

However, their architectures typically need to be designed by hand and fixed per task or dataset, requiring domain expertise BID62 .

Inference can also be heavy-weight for large models, as each sample engages every part of the network, i.e., increasing capacity causes a proportional increase in computation .Alternatively, DTs are characterised by learning hierarchical clusters of data .

A DT learns how to split the input space, so that in each subset, linear models suffice to explain the data.

In contrast to standard NNs, the architectures of DTs are optimised based on training data, and are particularly advantageous in data-scarce scenarios.

DTs also enjoy lightweight inference as only a single root-to-leaf path on the tree is used for each input sample.

However, successful applications of DTs often require hand-engineered features of data.

We can ascribe the limited expressivity of single DTs to the common use of simplistic routing functions, such as splitting on axis-aligned features.

The loss function for optimising hard partitioning is non-differentiable, which hinders the use of gradient descent-based optimization and thus complex splitting functions.

Current techniques for increasing capacity include ensemble methods such as random forests (RFs) BID4 ) and gradient-boosted trees (GBTs) BID14 , which are known to achieve state-of-the-art performance in various tasks, including medical imaging and financial forecasting BID49 BID27 BID33 BID56 .The goal of this work is to combine NNs and DTs to gain the complementary benefits of both approaches.

To this end, we propose adaptive neural trees (ANTs), which generalise previous work that attempted the same unification (Su??rez & Lutsko, 1999; ??rsoy et al., 2012; BID32 BID46 BID30 BID57 and address their limitations (see Tab.

1).

ANTs represent routing decisions and root-to-leaf computational paths within the tree structures as NNs, which lets them benefit from hierarchical representation learning, rather than being restricted to partitioning the raw data space.

In addition, we propose a backpropagation-based training algorithm to grow ANTs based on a series of decisions between making the ANT deeper-the central NN paradigm-or partitioning the data-the central DT paradigm (see FIG0 ).

This allows the architectures of ANTs to adapt to the data available.

By our design, ANTs inherit the following desirable properties from both DTs and NNs:??? Representation learning: as each root-to-leaf path in an ANT is a NN, features can be learnt end-to-end with gradient-based optimisation.

This, in turn, allows for learning complex data partitioning.

The training algorithm is also amenable to SGD.

??? Architecture learning: by progressively growing ANTs, the architecture adapts to the availability and complexity of data, embodying Occams razor.

The growth procedure can be viewed as architecture search with a hard constraint over the model class.??? Lightweight inference: at inference time, ANTs perform conditional computation, selecting a single root-to-leaf path on the tree on a per-sample basis, activating only a subset of the parameters of the model.

We empirically validate these benefits for classification and regression through experiments on the MNIST BID34 , CIFAR-10 (Krizhevsky & Hinton, 2009 ) and SARCOS BID55 datasets.

Along with other forms of neural networks, ANTs far outperform state-of-the-art random forest (RF) BID61 and gradient boosted tree (GBT) BID43 ) methods on the image-based classification datasets, with architectures achieving over 99% accuracy on MNIST and over 90% accuracy on CIFAR-10.

On the other hand, the best performing methods on the SARCOS multivariate regression dataset are all tree-based, with soft decision trees (SDTs) (Su??rez & Lutsko, 1999; BID26 , GBTs (Friedman, 2001) and ANTs achieving the lowest mean squared error.

At the same time, ANTs can learn meaningful hierarchical partitionings of data, e.g., grouping man-made and natural objects (see FIG2 ).

ANTs also have reduced time and memory requirements during inference, conferred by conditional computation.

In one case, we discover an architecture that achieves over 98% accuracy on MNIST using approximately the same number of parameters as a linear classifier on raw image pixels, showing the benefits of modelling a hierarchical structure that reflects the underlying data structure in enhancing both computational and predictive performance.

Finally, we demonstrate the benefits of architecture learning by training ANTs on subsets of CIFAR-10 of varying sizes.

The method can construct architectures of adequate size, leading to better generalisation, particularly on small datasets.

Our work is primarily related to research into combining DTs and NNs to benefit from the power of representation learning.

Here we explain how ANTs subsumes a large body of such prior work as specific cases and address their limitations.

We include additional reviews of work in conditional computation and neural architecture search in Sec. B in the supplementary material.

The very first SDT introduced in (Su??rez & Lutsko, 1999 ) is a specific case where in our terminology the routers are axis-aligned features, the transformers are identity functions, and the routers are static distributions over classes or linear functions.

The hierarchical mixture of experts (HMEs) proposed by BID26 ) is a variant of SDTs whose routers are linear classifiers and the tree structure is fixed.

More modern SDTs in BID46 BID32 used multilayer perceptrons (MLPs) or convolutional layers in the routers to learn more complex partitionings of the input space.

However, the simplicity of identity transformers used in these methods means that input data is never transformed and thus each path on the tree does not perform representation learning, limiting their performance.

More recent work suggested that integrating non-linear transformations of data into DTs would enhance model performance.

The neural decision forest (NDF) BID30 , which held cutting-edge performance on ImageNet BID11 (Deng et al., ) in 2015 , is an ensemble of DTs, each of which is also an instance of ANTs where the whole GoogLeNet architecture (Szegedy et al., 2015) (except for the last linear layer) is used as the root transformer, prior to learning tree-structured BT (??rsoy et al., 2014) Conv DT BID32 NDT BID46 NDT 2 BID57 NDF BID30 CNet BID22 ANT (ours) classifiers with linear routers.

BID57 employed a similar approach with a MLP at the root transformer, and is optimised to minimise a differentiable information gain loss.

The conditional network proposed in BID22 sparsified CNN architectures by distributing computations on hierarchical structures based on directed acyclic graphs with MLP-based routers, and designed models with the same accuracy with reduced compute cost and number of parameters.

However, in all cases, the model architectures are pre-specified and fixed.

In contrast, ANTs satisfy all criteria in Tab.

1; they provide a general framework for learning treestructured models with the capacity of representation learning along each path and within routing functions, and a mechanism for learning its architecture.

Architecture growth is a key facet of DTs , and typically performed in a greedy fashion with a termination criteria based on validation set error (Su??rez & Lutsko, 1999; Irsoy et al., 2012) .

Here we review previous attempts to improve upon this greedy growth strategy in the DT literature.

Decision jungles ) employ a training mechanism to merge partitioned input spaces between different sub-trees, and thus to rectify suboptimal "splits" made due to the locality of optimisation.

??rsoy et al. (2014) proposes budding trees, which are grown and pruned incrementally based on global optimisation of all existing nodes.

While our proposed training algorithm, for simplicity, grows the architecture by greedily choosing the best option between going deeper and splitting the input space (see FIG0 ), it is certainly amenable to the above advances.

Another related strand of work for feature learning is cascaded forests-stacks of RFs where the outputs of intermediate models are fed into the subsequent ones BID39 BID29 BID61 .

It has been shown how a cascade of DTs can be mapped to NNs with sparse connections BID51 , and more recently BID45 extended this argument to RFs.

However, the features obtained in this approach are the intermediate outputs of respective component models, which are not optimised for the target task, and cannot be learned end-to-end, thus limiting its representational quality.

We now formalise the definition of Adaptive Neural Trees (ANTs), which are a form of DTs enhanced with deep, learned representations.

We focus on supervised learning, where the aim is to learn the conditional distribution p(y|x) from a set of N labelled samples (x (1) , y (1) ), ..., (x (N ) , y (N ) ) ??? X ?? Y as training data.

In short, an ANT is a tree-structured model, characterized by a set of hierarchical partitions of the input space X , a series of nonlinear transformations, and separate predictive models in the respective component regions.

More formally, we define an ANT as a pair (T, O) where T defines the model topology, and O denotes the set of operations on it.

We restrict the model topology T to be instances of binary trees, defined as a set of finite graphs where every node is either an internal node or a leaf, and is the child of exactly one parent node (apart from the parent-less root node).

We define the topology of a tree as T := {N , E} where N is the set of all nodes, and E is the set of edges between them.

Nodes with no children are leaf nodes, N leaf , and all others are internal nodes, N int .

Every internal node j ??? N int has exactly two children nodes, represented by left(j) and right(j).

Unlike standard trees, E contains an edge which connects input data x with the root node, as shown in FIG0

Every node and edge is assigned with operations which acts on the allocated samples of data FIG0 .

Starting at the root, each sample gets transformed and traverses the tree according to the set of operations O. An ANT is constructed based on three primitive modules of differentiable operations:1.

Routers, R: each internal node j ??? N int holds a router module, r ?? j : X j ??? [0, 1] ??? R, parametrised by ??, which sends samples from the incoming edge to either the left or right child.

Here X j denotes the representation at node j.

We use stochastic routing, where the binary decision (1 for the left and 0 for the right branch) is sampled from Bernoulli distribution with mean r ?? j (x j ) for input x j ??? X j .

As an example, r ?? j can be defined as a small convolutional neural network (CNN).

2. Transformers, T : every edge e ??? E of the tree has one or a composition of multiple transformer module(s).

Each transformer t ?? e ??? T is a nonlinear function, parametrised by ??, that transforms samples from the previous module and passes them to the next one.

For example, t ?? e can be a single convolutional layer followed by ReLU BID40 .

Unlike in standard DTs, edges transform data and are allowed to "grow" by adding more operations (Sec. 4), learning "deeper" representations as needed.3.

Solvers, S: each leaf node l ??? N leaf is assigned to a solver module, s ?? l : X l ??? Y ??? S, parametrised by ??, which operates on the transformed input data and outputs an estimate for the conditional distribution p(y|x).

For classification tasks, we can define, for example, s ?? as a linear classifier on the feature space X l , which outputs a distribution over classes.

Defining operations on the graph T amounts to a specification of the triplet O = (R, T , S).

For example, given image inputs, we would choose the operations of each module to be from the set of operations commonly used in CNNs (examples are given in Tab.

2).

In this case, every computational path on the resultant ANT, as well as the set of routers that guide inputs to one of these paths, are given by CNNs.

In Sec. 4, we discuss methods for constructing such tree-shaped NNs end-toend from simple building blocks.

Lastly, many existing tree-structured models (Su??rez & Lutsko, 1999; ??rsoy et al., 2012; BID32 BID46 BID30 BID57 are instantiations of ANTs with limitations which we will address with our model (see Sec. 2 for a more detailed discussion).

An ANT (T, O) models the conditional distribution p(y|x) as a HME BID26 , each of which is defined as a NN and corresponds to a particular root-to-leaf path in the tree.

The key Table 2 : Primitive module specification for ANTs.

The 1 st & 2 nd rows describe modules for MNIST and CIFAR-10.

"conv5-40" denotes a 2D convolution with 40 kernels of spatial size 5 ?? 5.

"GAP", "FC" and "LC" stand for global-average-pooling, fully connected layer and linear classifier, respectively.

"

Downsample Freq" denotes the frequency at which 2 ?? 2 max-pooling is applied.

DISPLAYFORM0 difference with traditional HMEs is that the input is not only routed but also transformed within the tree hierarchy.

Each input x stochastically traverses the tree based on decisions of routers and undergoes a sequence of selected transformations until it reaches a leaf node where the corresponding solver module predicts the label y. Supposing we have L leaf nodes, the full predictive distribution is given by DISPLAYFORM1 , which describes the choice of leaf node (e.g. z l = 1 means that leaf l is used).

Here ??, ??, ?? summarise the parameters of router, transformer and solver modules in the tree.

The mixing coefficient ?? ??,?? l (x) := p(z l = 1|x, ??, ??) quantifies the probability that x is assigned to leaf l and is given by a product of decision probabilities over all router modules on the unique path P l from the root to leaf node l: ?? DISPLAYFORM2 where l j is a binary relation and is only true if leaf l is in the left subtree of internal node j, and x ?? j is the feature representation of x at node j. Let T j = {t ?? e1 , ..., t ?? en } denote the ordered set of the n transformer modules on the path from the root to node j, then the feature vector x ?? j is given by DISPLAYFORM3 On the other hand, the leaf-specific conditional distribution p ??,?? l (y) := p(y|x, z l = 1, ??, ??) in eq. equation 1 yields an estimate for the distribution over target y for leaf node l and is given by its solver's output s DISPLAYFORM4 We consider two schemes of inference, based on a trade-off between accuracy and computation.

Firstly, the full predictive distribution given in eq. equation 1 is used as the estimate for the target conditional distribution p(y|x).

However, averaging the distributions over all the leaves, weighted by their respective path probabilities, involves computing all operations at all nodes and edges of the tree, which makes inference expensive for a large ANT.

We therefore consider a second scheme which uses the predictive distribution at the leaf node chosen by greedily traversing the tree in the directions of highest confidence of the routers.

This approximation constrains computations to a single path, allowing for more memory-and time-efficient inference.

Training of an ANT proceeds in two stages: 1) growth phase during which the model architecture is learned based on local optimisation, and 2) refinement phase which further tunes the parameters of the model discovered in the first phase based on global optimisation.

We include pseudocode for the joint training algorithm in Sec. A in the supplementary material.

For both phases, we use the negative log-likelihood (NLL) as the common objective function to minimise, which is given by ???log p (Y|X, ??, ??, ??) DISPLAYFORM0 .., y (N ) } denote the training inputs and targets.

As all component modules (routers, transformers and solvers) are differentiable with respect to their parameters ?? = (??, ??, ??), we can use gradient-based optimisation.

Given an ANT with fixed topology T, we use backpropagation BID47 for gradient computation and use gradient descent to minimise the NLL for learning the parameters.

We next describe our proposed method for growing the tree T to an architecture of adequate complexity for the availability of training data.

Starting from the root, we choose one of the leaf nodes in breadth-first order and incrementally modify the architecture by adding extra computational modules to it.

In particular, we evaluate 3 choices ( FIG0 ) at each leaf node; (1)."split data" extends the current model by splitting the node with an addition of a new router; (2) "deepen transform" increases the depth of the incoming edge by adding a new transformer; (3) "keep" retains the current model.

We then locally optimise the parameters of the newly added modules in the architectures of FORMULA2 and (2) by minimising NLL via gradient descent, while fixing the parameters of the previous part of the computational graph.

Lastly, we select the model with the lowest validation NLL if it improves on the previously observed lowest NLL, otherwise we execute (3) and keep the original model.

This process is repeated to all new nodes level-by-level until no more "split data" or "deepen transform" operations pass the validation test.

The rationale for evaluating the two choices is to the give the model a freedom to choose the most effective option between "going deeper" or splitting the data space.

Splitting a node is equivalent to a soft partitioning of the feature space of incoming data, and gives birth to two new leaf nodes (left and right children solvers).

In this case, the added transformer modules on the two branches are identity functions.

Deepening an edge on the other hand does not change the number of leaf nodes, but instead seeks to learn richer representation via an extra nonlinear transformation, and replaces the old solver with a new one.

Local optimisation saves time, memory and compute.

Gradients only need to be computed for the parameters of the new peripheral parts of the architecture, reducing the amount of time and computation needed.

Forward activations prior to the new parts do not need to be stored in memory, saving space.

Once the model topology is determined in the growth phase, we finish by performing global optimisation to refine the parameters of the model, now with a fixed architecture.

This time, we perform gradient descent on the NLL with respect to the parameters of all modules in the graph, jointly optimising the hierarchical grouping of data to paths on the tree and the associated expert NNs.

The refinement phase can correct suboptimal decisions made during the local optimisation of the growth phase, and empirically improves the generalisation error (see Sec. 5.3).

We evaluate ANTs using the MNIST BID34 and CIFAR-10 (Krizhevsky & Hinton, 2009 ) object classification datasets, and the SARCOS multivariate regression dataset BID55 (see Supp.

Sec. H for regression and Supp.

Sec. I for ensembling details).

Here, we first show that ANTs learn hierarchical structures in the data, while still achieving favourable classification accuracies against relevant DT and NN models.

Next, we examine the effects of refinement phase on ANTs, and show that it can automatically prune the tree.

Finally, we demonstrate that our proposed training procedure adapts the model size appropriately under varying amounts of labelled data.

All of our models are constructed using the PyTorch framework BID41 .

Params.

BID19 6.43 N/A 1.7M N/A 1 DenseNet-BC (k=40) 3.46 N/A 25.6M N/A 1

We train ANTs with a range of primitive modules (Tab.

2) and compare against relevant DT and NN models (Tab.

3).

In general, DT methods without feature learning, such as RFs BID4 BID61 and GBTs BID43 , perform poorly on complex image data BID31 ).

In comparison with CNNs without shortcut connections BID34 BID16 BID37 Springenberg et al., 2015) , different ANTs balance between strong performance with comparable numbers of trainable parameters, and reasonable performance with a relatively small amount of parameters.

At the other end of the spectrum, state-of-the-art NNs BID48 contain significantly more parameters.

For simplicity, we define primitive modules based on three types of NN layers: convolutional, global-average-pooling (GAP) and fully-connected (FC).

Solver modules are fixed as linear classifiers (LC) with a softmax output.

Router modules are binary classifiers with a sigmoid output.

All convolutional and FC layer are followed by ReLUs, except in the last layers of solvers and routers.

We also apply 2 ?? 2 max-pooling to feature maps after every d transformer modules where d is the downsample frequency.

We balance the number of parameters in the router and transformer modules to be of the same order of magnitude to avoid favouring either partitioning the data or learning more expressive features.

We hold out 10% of training images as a validation set, on which the best performing model is selected.

Full training details, including training times, are provided in the supplementary material.

Two inference schemes: for each ANT, classification is performed in two ways: multi-path inference with the full predictive distribution (eq. equation 1), and single-path inference based on the greedily-selected leaf node (Sec. 3.2).

We observed that with our training scheme the splitting probabilities in the routers tend to be very confident, being close to 0 or 1 (see histograms in blue in FIG2 ).

This means that single path inference gives a good approximation of the multi-path inference but is more efficient to compute.

We show this holds empirically in Tab.

3, where the largest difference between Error (Full) and Error (Path) is 0.06% while number of parameters is reduced from Params (Full) to Params (Path) across all ANT models.

Patience-based local optimisation: in the growth phase the parameters for the new modules are trained until convergence, as determined by patience-based early stopping on the validation set.

We observe that very low or high patience levels result in new modules underfitting or overfitting locally, respectively, thus preventing meaningful further growth.

We tuned this hyperparameter using the validation sets, and set the patience level to 5, which produced consistently good performance on both MNIST and CIFAR-10 datasets across different specifications of primitive modules.

A quantitative evaluation is given in the supplementary (Sec. E).MNIST digit classification: we observe that ANT-MNIST-A outperforms state-of-the-art GBT BID43 and RF BID61 ) methods in accuracy.

This performance is attained despite the use of a single tree, while RF methods operate with ensembles of classifiers (the size shown in Tab.

2).

In particular, the NDF BID30 ) has a pre-specified architecture where LeNet-5 (LeCun et al., 1998 ) is used as the root transformer module, and 10 trees of fixed depth 5 are constructed from this base feature extractor.

On the other hand, ANT-MNIST-A is constructed in a data-driven manner from primitive modules, and displays an improvement over the NDF both in terms of accuracy and number of parameters.

In addition, reducing the size of convolution kernels (ANT-MNIST-B) reduces the total number of parameters by 25% and the path-wise average by almost 40% while only increasing absolute error by < 0.1%.We also compare against the LeNet-5 CNN (LeCun et al., 1998), comprised of the same types of operations used in our primitive modules (i.e. convolutional, max-pooling and FC layers).

For a fair comparison, the network is trained with the same protocol as that of the ANT refinement phase, achieving an error rate of 0.82% (lower than the reported value of 0.87%) on the test set.

Both ANT-MNIST-A and ANT-MNIST-B attain better accuracy with a smaller number of parameters than LeNet-5.

The current state-of-the-art, capsule networks (CapsNets) BID48 , have more parameters than ANT-MNIST-A by almost two orders of magnitude.

1 By ensembling ANTs we can reach similar performance (0.29% versus 0.25%; see Tab.

9) with an order of magnitude less parameters (see Tab.

10).Lastly, we highlight the observation that ANT-MNIST-C, with the simplest primitive modules, achieves an error rate of 1.68% with single-path inference, which is significantly better than that of the linear classifier (7.91%), while engaging almost the same number of parameters (7, 956 vs. 7, 840) on average.

To isolate the benefit of convolutions, we took one of the root-to-path CNNs on ANT-MNIST-C and increased the number of kernels to adjust the number of parameters to the same value.

We observe a higher error rate of 3.55%, which indicates that while convolutions are beneficial, data partitioning has additional benefits in improving accuracy.

This result demonstrates the potential of ANT growth protocol for constructing performant models with lightweight inference.

See Sec. G in the supplementary materials for the architecture of ANT-MNIST-C.CIFAR-10 object recognition: we see that variants of ANTs outperform the state-of-the-art DT method, gcForest BID61 by a large margin, achieving over 90% accuracy, demonstrating the benefit of representation learning in tree-structured models.

Secondly, with fewer number of parameters in single-path inference, ANT-CIFAR-A achieves higher accuracy than CNN models without shortcut connections BID16 BID37 Springenberg et al., 2015) that held the state-of-the-art performance at the time of publication.

With simpler primitive modules we learn more compact models (ANT-MNIST-B and -C) with a marginal compromise in accuracy.

In addition, initialising the parameters of transformers and routers from a pre-trained single-path CNN further reduced the error rate of ANT-MNIST-A by 20% (see ANT-MNIST-A* in Tab.

3), which indicates room for improvement in our proposed optimisation method.

Shortcut connections BID12 have recently lead to leaps in performance in deep CNNs BID19 .

We observe that our best network, ANT-MNIST-A*, has a comparable error rate and half the parameter count (with single-path inference) to the bestperforming residual network, ResNet-110 BID19 .

Densely connected networks leads to substantially better accuracy, but with an order of magnitude more parameters .

We expect that shortcut connections could also improve ANT performance, and leave integrating them to future work. shows that the refinement phase polarises path probabilities, pruning a branch.

Ablation study: we lastly compare the classification errors of different variants of ANTs in cases where the options for adding transformer or router modules are disabled (see Tab.

4).

In this experiment, patience levels are tuned separately for respective models.

In the first case, the resulting models are equivalent to SDTs (Su??rez & Lutsko, 1999) or HMEs (Jordan & Jacobs, 1994) with locally grown architectures, while the second case is equivalent to standard CNNs, grown adaptively layer by layer.

We observe that either ablation consistently leads to higher classification errors across different module configurations.

The growth procedure of ANTs is capable of discovering hierarchical structures in the data that are useful to the end task.

Learned hierarchies often display strong specialisation of paths to certain classes or categories of data on both the MNIST and CIFAR-10 datasets.

FIG2 (a) displays an example with particularly "human-interpretable" partitions e.g. man-made versus natural objects, and road vehicles versus other types of vehicles.

It should, however, be noted that human intuitions on relevant hierarchical structures do not necessarily equate to optimal representations, particularly as datasets may not necessarily have an underlying hierarchical structure, e.g., MNIST.

Rather, what needs to be highlighted is the ability of ANTs to learn when to share or separate the representation of data to optimise end-task performance, which gives rise to automatically discovering such hierarchies.

To further attest that the model learns a meaningful routing strategy, we also present the test accuracy of the predictions from the leaf node with the smallest reaching probability in Supp.

Sec. F. We observe that using the least likely "expert" leads to a substantial drop in classification accuracy.

In addition, we observe that most learned trees are unbalanced (see Supp.

Sec. G for more examples).

This property of adaptive computation is plausible since certain types of images may be easier to classify than others, as seen in prior work BID13 .

We observe that global refinement phase improves the generalisation error.

FIG1 shows the generalisation error of various ANT models on CIFAR-10, with vertical dotted lines indicating the epoch when the models enter the refinement phase.

As we switch from optimising parts of the ANT in isolation to optimising all parameters, we shift the optimisation landscape, resulting in an initial drop in performance.

However, they all consistently converge to higher test accuracy than the best value attained during the growth phase.

This provides evidence that refinement phase remedies suboptimal decisions made during the locally-optimised growth phase.

In many cases, we observed that global optimisation polarises the decision probability of routers, which occasionally leads to the effective "pruning" of some branches.

For example, in the case of the tree shown in FIG2 , we observe that the decision probability of routers are more concentrated near 0 or 1 after global refinement, and as a result, the empirical probability of visiting one of the leaf nodes, calculated over the validation set, reduces to 0.09%-meaning that the corresponding branch could be pruned without a negligible change in the network's accuracy.

The resultant model attains lower generalisation error, showing that the pruning has resolved a suboptimal partioning of data.

We emphasise that this is a consequence of global fine-tuning, and does not involve additional algorithms that would be used to prune or compress standard NNs.

Overparametrised models, trained without regularization, are vulnerable to overfitting on small datasets.

Here we assess the ability of our proposed ANT training method to adapt the model complexity to varying amounts of labelled data.

We run classfication experiments on CIFAR-10 and train three variants of ANTs, the baseline All-CNN (Springenberg et al., 2015) and linear classifier on subsets of the dataset of sizes 50, 250, 500, 2.5k, 5k, 25k and 45k (the full training set).

We choose All-CNN as the baseline as it reports the lowest error among the comparison targets and is the closest in terms of constituent operations (convolutional, GAP and FC layers).

FIG1 shows the corresponding test performances.

The best model is picked based on the performance on the same validation set of 5k examples as before.

As the dataset gets smaller, the margin between the test accuracy of the ANT models and All-CNN/linear classifier increases (up to 13%).

FIG1 shows the model size of discovered ANTs as the dataset size varies.

It can be observed that for different settings of primitive modules, the number of parameters generally increases as a function of the dataset size.

All-CNN has a fixed number of parameters, consistently larger than the discovered ANTs, and suffers from overfitting, particularly on small datasets.

The linear classifier, on the other hand, underfits to the data.

Our method constructs models of adequate complexity, leading to better generalisation.

This shows the added value of our tree-building algorithm over using models of fixed-size structures.

We introduced Adaptive Neural Trees (ANTs), a holistic way to marry the architecture learning, conditional computation and hierarchical clustering of decision trees (DTs) with the hierarchical representation learning and gradient descent optimization of deep neural networks (DNNs).

Our proposed training algorithm optimises both the parameters and architectures of ANTs through progressive growth, tuning them to the size and complexity of the training dataset.

Together, these properties make ANTs a generalisation of previous work attempting to unite NNs and DTs.

Finally, we validated the claimed benefits of ANTs on standard regression and object classification datasets, whilst still achieving high performance.

A TRAINING ALGORITHM

Initialise topology T and parameters O T is set to a root node with one solver and one transformer Optimise parameters in O via gradient descent on NLL Learning root classifier Set the root node "suboptimal" while true do Growth of T begins Freeze all parameters O Pick next "suboptimal" leaf node l ??? N leaf in the breadth-first order Add (1) router to l and train new parameters Split data Add (2) transformer to the incoming edge of l and train new parameters Deepen transform Add (1) or (2) permanently to T if validation error decreases, otherwise leaf is set to "optimal" Add any new modules to O if no "suboptimal" leaves remain then Break Unfreeze and train all parameters in O Global refinement with fixed T

The tree-structure of ANTs naturally performs conditional computation.

We can also view the proposed tree-building algorithm as a form of neural architecture search.

Here we provide surveys of these areas and their relations to ANTs.

In NNs, computation of each sample engages every parameter of the model.

In contrast, DTs route each sample to a single path, only activating a small fraction of the model.

Bengio BID2 advocated for this notion of conditional computation to be integrated into NNs, and this has become a topic of growing interest.

Rationales for using conditional computation ranges from attaining better capacity-to-computation ratio BID10 BID52 to adapting the required computation to the difficulty of the input and task BID0 Teerapittayanon et al., 2016; BID17 BID13 Veit & Belongie, 2017) .

We view the growth procedure of ANTs as having a similar motivation with the latter-processing raw pixels is suboptimal for computer vision tasks, but we have no reason to believe that the hundreds of convolutional layers in current state-of-the-art architectures BID19 are necessary either.

Growing ANTs adapts the architecture complexity to the dataset as a whole, with routers determining the computation needed on a per-sample basis.

Neural Architecture Search: The ANT growing procedure is related to the progressive growing of NNs BID12 BID20 BID58 BID7 Srivastava et al., 2015; BID36 BID6 ??rsoy & Alpayd??n, 2018) , or more broadly, the field of neural architecture search BID62 BID5 BID8 .

This approach, mainly via greedy layerwise training, has historically been one solution to optimising NNs BID12 BID20 .

However, nowadays it is possible to train NNs in an end-toend fashion.

One area which still uses progressive growing is lifelong learning, in which a model needs to adapt to new tasks while retaining performance on previous ones BID58 BID36 .

In particular, BID58 introduced a method that grows a tree-shaped network to accommodate new classes.

However, their method never transforms the data before passing it to the children classifiers, and hence never benefit from the parent's representations.

Whilst we learn the architecture of an ANT in a greedy, layerwise fashion, several other methods search globally.

Based on a variety of techniques, including evolutionary algorithms (Stanley & Miikkulainen, 2002; BID44 , reinforcement learning BID62 , sequential optimisation and boosting BID8 , these methods find extremely high-performance yet complex architectures.

In our case, we constrain the search space to simple tree-structured NNs, retaining desirable properties of DTs such as data-dependent computation and interpretable structures, while keeping the space and time requirement of architecture search tractable thanks to the locality of our growth procedure.

We perform our experiments on the MNIST digit classification task BID34 and CIFAR-10 object recognition task BID31 .

The MNIST dataset consists of 60, 000 training and 10, 000 testing examples, all of which are 28 ?? 28 grayscale images of digits from 0 to 9 (10 classes).

The dataset is preprocessed by subtracting the mean, but no data augmentation is used.

The CIFAR-10 dataset consists of 50, 000 training and 10, 000 testing examples, all of which are 32 ?? 32 coloured natural images drawn from 10 classes.

We adopt an augmentation scheme widely used in the literature BID16 BID37 Springenberg et al., 2015; BID19 where images are zero-padded with 4 pixels on each side, randomly cropped and horizontally mirrored.

For both datasets, we hold out 10% of training images as a validation set.

The best model is selected based on the validation accuracy over the course of ANT training, spanning both the growth phase and the refinement phase, and its accuracy on the testing set is reported.

The hyperparameters are also selected based on the validation performance alone.

Both the growth and refinement phase of ANTs takes up to 2 hours on a single Titan X GPU on both datasets.

For all the experiments in this paper, we employ the following training protocol:(1) optimize parameters using Adam BID28 with initial learning rate of 10 ???3 and ?? = [0.9, 0.999], with minibatches of size 512; (2) during the growth phase, employ early stopping with a patience of 5, that is, training is stopped after 5 epochs of no progress on the validation set; (3) during the refinement phase, train for 100 epochs for MNIST and 200 epochs for CIFAR-10, decreasing the learning rate by a factor of 10 at every multiple of 50.

Tab.

5 summarises the time taken on a single Titan X GPU for the growth phase and refinement phase of various ANTs, and compares against the training time of All-CNN (Springenberg et al., 2015) .

Local optimisation during the growth phase means that the gradient computation is constrained to the newly added component of the graph, allowing us to grow a good candidate model under 2 hours on a single GPU.

When the patience level is 1, the architecture growth terminates prematurely and plateaus at low accuracy at 80%.

On the other hand, a patience level of 15 causes the model to overfit locally with 87%.

In between these, the patience level of 5 gives the best results with 91% validation accuracy.

We investigate if the learned routing strategy is meaningful by comparing the classification accuracy of our default path-wise inference against that of the predictions from the leaf node with the smallest reaching probability.

Tab.

6 shows that using the least likely "expert" leads to a substantial drop in classification accuracy, down to close to that of random guess or even worse for large trees (ANT-MNIST-C and ANT-CIFAR10-C).

This demonstrates that ANTs have the capability to split the input space in a meaningful way.

FIG4 shows ANT architectures discovered on the MNIST (i-iii) and CIFAR-10 (iv-vi) datasets.

We observe two notable trends.

Firstly, most architectures learn a few levels of features before resorting to primarily splits.

However, over half of the architectures (ii-v) still learn further representations beyond the first split.

Secondly, all architectures are unbalanced.

This reflects the fact that some groups of samples may be easier to classify than others.

This property is reflected by traditional DT algorithms, but not "neural" tree-structured models that stick to pre-specified architectures BID32 BID30 BID22 .

DISPLAYFORM0

The ANT algorithm is general purpose, and can be applied to problems other than classification on image data.

To demonstrate this, we also grow ANTs to perform (multivariate) regression on the SARCOS robot inverse dynamics dataset 2 , which consists of 44,484 training and 4,449 testing examples, where the goal is to map from the 21-dimensional input space (7 joint positions, 7 joint velocities and 7 joint accelerations) to the corresponding 7 joint torques BID55 .

No dataset preprocessing or augmentation is used.

We hold out 10% of the training examples as a validation set.

Baseline MLPs, routers and transformers are composed of single fully connected layers with 256 units with tanh nonlinearities, and the solver is a linear regressor.

Other training details are the same as for classification (see Supp.

Sec. C).

All non-NN-based methods were trained using scikit-learn BID42 ; only single-output GBT models were available so 7 separate GBTs were trained.

The results are shown in Tab.

7. ANT-SARCOS outperforms all other methods in mean squared error with the full set of parameters, with GBTs performing slightly better using single-path inference.

In comparison with results on MNIST and CIFAR-10, we note that the top 3 performing methods are all tree-based, with the third best method being an SDT (with MLP routers).

This highlights the power of splitting the input space and conditional computation, both of which standard NNs are not capable of.

Meanwhile, we still reap the benefits of representation learning, as shown by both ANT-SARCOS and the SDT (which is a specific form of ANT) requiring fewer parameters than the best-performing GBT configuration.

Finally, we note that deeper NNs (5 vs. 3 hidden layers) can overfit on this small dataset, which makes the adaptive growth procedure of tree-based methods ideal for finding a model that exhibits good generalisation.

BID60 5 Ablation study: we compare the regression error of our ANT in cases where the options for adding transformer or router modules are disabled (see Tab.

8).

In this experiment, patience levels are tuned separately for respective models.

In the first case, the resulting models are equivalent to SDTs (Su??rez & Lutsko, 1999) or HMEs BID26 with locally grown architectures, while the second case is equivalent to standard NNs, grown adaptively layer by layer.

We observe that either ablation consistently leads to higher regression errors across different module configurations.

Table 8 : Ablation study to compare the effects of different components of ANTs on regression performance.

"NN" refers to the case where the ANT is grown without routers while "SDT/HME" refers to the case where transformer modules on the edges are disabled.

BID4 ) and NNs BID18 , ANTs can be ensembled to gain improved performance.

In Tab.

9 we show the results of ensembling 8 ANTs (using the "-A" configurations for classification), each of which is trained with a randomly chosen split between training and validation sets.

We compare against the single tree models, trained with the default split as used in the training of models reported in Tab.

3 and Tab.

7.

In all cases both the full and single-path inference performance is noticeably improved, and in MNIST we reach close to state-of-the-art performance (0.29% versus 0.25% BID48 ) with significantly fewer parameters (851k versus 8.2M; see Tab.

10 for ensemble and Tab.

3 for baseline parameter counts).

<|TLDR|>

@highlight

We propose a framework to combine decision trees and neural networks, and show on image classification tasks that it enjoys the complementary benefits of the two approaches, while addressing the limitations of prior work.

@highlight

The authors proposed a new model, Adaptive Neural Trees, by combining the representation learning and gradient optimization of neural networks with architecture learning of decision trees

@highlight

This paper proposes the Adaptive Neural Trees approach to combine the two learning paradigms of deep neural nets and decision trees