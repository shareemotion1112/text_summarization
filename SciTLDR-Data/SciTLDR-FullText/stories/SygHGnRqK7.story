In federated learning problems, data is scattered across different servers and exchanging or pooling it is often impractical or prohibited.

We develop a Bayesian nonparametric framework for federated learning with neural networks.

Each data server is assumed to train local neural network weights, which are modeled through our framework.

We then develop an inference approach that allows us to synthesize a more expressive global network without additional supervision or data pooling.

We then demonstrate the efficacy of our approach on federated learning problems simulated from two popular image classification datasets.

The standard machine learning paradigm involves algorithms that learn from centralized data, possibly pooled together from multiple data sources.

The computations involved may be done on a single machine or farmed out to a cluster of machines.

However, in the real world, data often lives in silos and amalgamating them may be rendered prohibitively expensive by communication costs, time sensitivity, or privacy concerns.

Consider, for instance, data recorded from sensors embedded in wearable devices.

Such data is inherently private, can be voluminous depending on the sampling rate of the sensing modality, and may be time sensitive depending on the analysis of interest.

Pooling data from many users is technically challenging owing to the severe computational burden of moving large amounts of data, and fraught with privacy concerns stemming from potential data breaches that may expose the user's protected health information (PHI).Federated learning avoids these pitfalls by obviating the need for centralized data and instead designs algorithms that learn from sequestered data sources with different data distributions.

To be effective, such algorithms must be able to extract and distill important statistical patterns from various independent local learners coherently into an effective global model without centralizing data.

This will allow us to avoid the prohibitively expensive cost of data communication.

To achieve this, we develop and investigate a probabilistic federated learning framework with a particular emphasis on training and aggregating neural network models on siloed data.

We proceed by training local models for each data source, in parallel.

We then match the estimated local model parameters (groups of weight vectors in the case of neural networks) across data sources to construct a global network.

The matching, to be formally defined later, is governed by the posterior of a Beta-Bernoulli process (BBP) (Thibaux & Jordan, 2007; Yurochkin et al., 2018) , a Bayesian nonparametric model that allows the local parameters to either match existing global ones or create a new global parameter if existing ones are poor matches.

Our construction allows the size of the global network to flexibly grow or shrink as needed to best explain the observed data.

Crucially, we make no assumptions about how the data is distributed between the different sources or even about the local learning algorithms.

These may be adapted as necessary, for instance to account for non-identically distributed data.

Further, we only require communication after the local algorithms have converged.

This is in contrast with popular distributed training algorithms that rely on frequent communication between the local machines.

Our construction also leads to compressed global models with fewer parameters than the set of all local parameters.

Unlike naive ensembles of local models, this allows us to store fewer parameters and leads to more efficient inference at test time, requiring only a single forward pass through the compressed model as opposed to J forward passes, once for each local model.

While techniques such as distillation allow for the cost of multiple forward passes to be amortized, training the distilled model itself requires access to data pooled across all sources, a luxury unavailable in our federated learning scenario.

In summary, the key question we seek to answer in this paper is the following: given pre-trained neural networks trained locally on non-centralized data, can we learn a compressed federated model without accessing the original data, while improving on the performance of the local networks?The remainder of the paper is organized as follows.

We briefly introduce the Beta-Bernoulli process in Section 2 before describing our model for federated learning in Section 3.

We thoroughly vet the proposed models and demonstrate the utility of the proposed approach in Section 4.

Finally, Section 5 discusses limitations and open questions.

Our approach builds on tools from Bayesian nonparametrics, in particular the Beta-Bernoulli Process (BBP) (Thibaux & Jordan, 2007) and the closely related Indian Buffet Process (IBP) BID5 .

We briefly review these ideas before describing our approach.

Consider a random measure Q drawn from a Beta Process with mass parameter 0 and base measure H, Q| 0 , H ⇠ BP(1, 0 H).

It follows that Q is a discrete (not probability) measure Q = P i q i ✓i formed by pairs (q i , ✓ i ) 2 [0, 1] ⇥ ⌦ of weights and atoms.

The weights {q i } 1 i=1 follow a stickbreaking construction (Teh et al., 2007) : c i ⇠ Beta( 0 , 1), q i = Q i j=1 c j and the atoms are drawn i.i.d from the (scaled) base measure ✓ i ⇠ H/H(⌦) with domain ⌦.

In this paper, ⌦ is simply R D for some D. Subsets of atoms in the random measure Q are then selected using a Bernoulli process with a base measure Q, T j |Q ⇠ BeP(Q) for j = 1, . . .

, J. Each T j is also a discrete measure formed by DISPLAYFORM0 Together, this hierarchical construction describes the Beta-Bernoulli process.

Marginalizing Q induces dependencies among T j , i.e. DISPLAYFORM1 j=1 b ji (dependency on J is suppressed in the notation for simplicity) and is sometimes called the Indian Buffet Process.

The IBP can be equivalently described by the following culinary metaphor.

J customers arrive sequentially at a buffet and choose dishes to sample as follows, the first customer tries Poisson( 0 ) dishes.

Every subsequent j-th customer tries each of the previously selected dishes according to their popularity, i.e. dish i with probability m i /j, and then tries Poisson( 0 /j) new dishes.

The IBP, which specifies a distribution over sparse binary matrices with infinitely many columns, was originally demonstrated for latent factor analysis BID4 .

Several extensions to the IBP (and the equivalent BBP) have been developed, see BID5 for a review.

Our work is related to a recent application of these ideas to distributed topic modeling (Yurochkin et al., 2018) , where the authors use the BBP for modeling topics learned from multiple collections of document, and provide an inference scheme based on the Hungarian algorithm BID8 .

Extending these ideas to federated learning of neural networks requires significant innovations and is the primary focus of our paper.

Federated learning has recently garnered attention from the machine learning community.

Smith et al. (2017) pose federated learning as a multi-task learning problem, which exploits the convexity and decomposability of the cost function of the underlying support vector machine (SVM) model for distributed learning.

This approach however does not extend to the neural network structure considered in our work.

Others BID15 use strategies based on simple averaging of the local learner weights to learn the federated model.

However, as pointed out by the authors, such naive averaging of model parameters can be disastrous for non-convex cost functions.

To cope, they have to use a heuristic scheme where the local learners are forced to share the same random initialization.

In contrast, our proposed framework is naturally immune to such issues since its development assumes nothing specific about how the local models were trained.

Moreover, unlike the previous work of BID15 , our framework is non-parametric in nature and it therefore allows the federated model to flexibly grow or shrink its complexity (i.e., its sizes) to account for the varying data complexity.

There is also significant work on distributed deep learning BID12 ; BID16 ; BID11 ; .

However, the emphasis of these works is on scalable training from large data and they typically require frequent communication between the distributed nodes to be effective.

Yet others explore distributed optimization with a specific emphasis on communication efficiency (Zhang et al., 2013; Shamir et al., 2014; Yang, 2013; BID14 Zhang & Lin, 2015) .

However, as pointed out by BID15 , these works primarily focus on settings with convex cost functions and often assume that each distributed data source contains an equal number of data instances.

These assumptions, in general, do not hold in our scenario.

We now apply this Bayesian nonparametric machinery to the problem of federated learning with neural networks.

Our goal will be to identify subsets of neurons in each of the J local models that match to neurons in other local models, and then use these to form an aggregate model where the matched parts of each of the local models are fused together.

Our approach to federated learning builds upon the following basic problem.

Suppose we have trained J Multilayer Perceptrons (MLPs) with one hidden layer each.

For the jth MLP j = 1, . . .

, J, let DISPLAYFORM0 2 R Lj be weights and biases of the hidden layer; V(1) j 2 R Lj ⇥K and v (1) j 2 R K be weights and biases of the softmax layer; D be the data dimension, L j the number of neurons on the hidden layer; and K the number of classes.

We consider a simple architecture: DISPLAYFORM1 j ) where (·) is some nonlinearity (sigmoid, ReLU, etc.) .

Given the collection of weights and biases {V DISPLAYFORM2 we want to learn a global neural network with weights and biases DISPLAYFORM3 L j is an unknown number of hidden units of the global network to be inferred.

Our first observation is that ordering of neurons of the hidden layer of an MLP is permutation invariant.

Consider any permutation ⌧ (1, . . .

, L j ) of the j-th MLP -reordering columns of V according to ⌧ (1, . . .

, L j ) will not affect the outputs f j (x) for any value of x. Therefore, instead of treating weights as matrices and biases as vectors we view them as unordered collections of vectors DISPLAYFORM4 Hidden layers in neural networks are commonly viewed as feature extractors.

This perspective can be justified by the fact that last layer of a neural networks is simply a softmax regression.

Since neural networks greatly outperform basic softmax regression in a majority of applications, neural networks must be supplying high quality features constructed from the input features.

Mathematically, in our problem setup, every hidden neuron of j-th MLP represents a new featurẽ DISPLAYFORM5 jl ) acts as a parameterization of the corresponding neuron's feature extractor.

Since each of the given MLPs was trained on the same general type of data (not necessarily homogeneous), we assume that they should share at least some feature extractors that serve the same purpose.

However, due to the permutation invariance described previously, a feature extractor indexed by l from the j-th MLP is unlikely to correspond to a feature extractor with the same index from a different MLP.

In order to construct a set of global feature extractors (neurons) {✓ DISPLAYFORM6 we must model the process of grouping and combining feature extractors of collection of MLPs.

We now present the key building block of our modeling framework, our Hierarchical BBP (Thibaux & Jordan, 2007) based model of the neurons and weights of multiple MLPs.

Our generative model is as follows.

First, draw a collection of global atoms (hidden layer neurons) from a Beta process prior with a base measure H and mass parameter 0 , Q = P i q i ✓i .

In our experiments we choose H = N (µ 0 , ⌃ 0 ) as the base measure with µ 0 2 R D+1+K and diagonal DISPLAYFORM0 formed from the feature extractor weight-bias pairs with the corresponding weights of the softmax regression.

Next, for each batch (server) j = 1, . . .

, J, generate a batch specific distribution over global atoms (neurons): DISPLAYFORM1 where the p ji s vary around corresponding q i .

The distributional properties of p ji are described in Thibaux & Jordan (2007) .

Now, for each j = 1, . . .

, J select a subset of the global atoms for batch j via the Bernoulli process: DISPLAYFORM2 T j is supported by atoms {✓ i : b ji = 1, i = 1, 2, . . .}, which represent the identities of the atoms (neurons) used by batch (server) j. Finally, assume that observed local atoms are noisy measurements of the corresponding global atoms: DISPLAYFORM3 where DISPLAYFORM4 jl ] are the weights, biases, and softmax regression weights corresponding to the l-th neuron of the j-th MLP trained with L j neurons on the data of batch j.

Under this model, the key quantity to be inferred is the collection of random variables that match observed atoms (neurons) at any batch to the global atoms.

We denote the collection of these random variables as DISPLAYFORM5 , where DISPLAYFORM6 Maximum a posteriori estimation.

We now derive an algorithm for MAP estimation of global atoms for the model presented above.

The objective function to be maximized is the posterior of DISPLAYFORM7 arg max DISPLAYFORM8 Note that the next proposition easily follows from Gaussian-Gaussian conjugacy (Supplement 1): Proposition 1.

Given {B j }, the MAP estimate of {✓ i } is given bŷ DISPLAYFORM9 where for simplicity we assume ⌃ 0 = I 2 0 and ⌃ j = I 2 j .Using this fact we can cast optimization corresponding to (4) with respect to only DISPLAYFORM10 .

Taking natural logarithm we obtain: arg max DISPLAYFORM11 Detailed derivation of this and subsequent results are given in Supplement 1.

We consider an iterative optimization approach: fixing all but one B j we find corresponding optimal assignment, then pick a new j at random and proceed until convergence.

In the following we will use notation j to say "all but j".

Let L j = max{i : B j i,l = 1} denote number of active global weights outside of group j.

We now rearrange the first term of (6) by partitioning it into i = 1, . . .

, L j and DISPLAYFORM12 We are interested in solving for B j , hence we can modify objective function by subtracting terms independent of B j and noting that P l B j i,l 2 {0, 1}, i.e. it is 1 if some neuron from batch j is matched to global neuron i and 0 otherwise: DISPLAYFORM13 (7) Now we consider the second term of (6): DISPLAYFORM14 First, because we are optimizing for B j , we can ignore log P (B j ).

Second, due to exchangeability of batches (i.e. customers of the IBP), we can always consider B j to be the last batch (i.e. lastUnder review as a conference paper at ICLR 2019 customer of the IBP).

Let m DISPLAYFORM15 i,l denote number of times batch weights were assigned to global weight i outside of group j. We now obtain the following: DISPLAYFORM16 Combining FORMULA42 and FORMULA25 we obtain the assignment cost objective, which we solve with the Hungarian algorithm.

Proposition 2.

The assignment cost specification for finding B j is: DISPLAYFORM17 (9) We then apply the Hungarian algorithm described in Supplement 1 to find the minimizer of DISPLAYFORM18 and obtain the neuron matching assignments.

We summarize the overall single layer inference procedure in FIG3 below.

Nodes in the graphs indicate neurons, neurons of the same color have been matched.

Our approach consists of using the corresponding neurons in the output layer to convert the neurons in each of the J servers to weight vectors referencing the output layer.

These weight vectors are then used to form a cost matrix, which the Hungarian algorithm then uses to do the matching.

Finally, the matched neurons are then aggregated and averaged to form the new layer of the global model.

The model we have presented thus far can handle any arbitrary width single layer neural network, which is known to be theoretically sufficient for approximating any function of interest BID7 .

However, deep neural networks with moderate layer widths are known to be beneficial both practically BID10 and theoretically BID17 .

We extend our neural matching approach to these deep architectures by defining a generative model of deep neural network weights from outputs back to inputs (top-down).

Let C denote the number of hidden layers and L c the number of neurons on the cth layer.

Then L C+1 = K is the number of labels and L 0 = D is the input dimension.

In the top down approach, we consider the global atoms to be vectors of outgoing weights from a neuron instead of weights forming a neuron as it was in the single hidden layer model.

This change is needed to avoid base measures with unbounded dimensions.

Starting with the top hidden layer c = C, we generate each layer following a model similar to that used in the single layer case.

For each layer we generate a collection of global atoms and select a subset of them for each batch using Hierarchical Beta-Bernoulli process construction.

L c+1 is the number of neurons on the layer c + 1, which controls the dimension of the atoms in layer c.

Definition 1 (Multilayer generative process).

Starting with layer c = C, generate (as in the single layer process) DISPLAYFORM0 This T c j is the set of global atoms (neurons) used by batch j in layer c, it is contains atoms {✓ for c = 1.

We also note that the bias term can be added to the model, we omitted it to simplify notation.

Inference Following the top-down generative model, we adopt a greedy inference procedure that first infers the matching of the top layer and then proceeds down the layers of the network.

This is possible because the generative process for each layer depends only on the identity and number of the global neurons in the layer above it, hence once we infer the c + 1th layer of the global model we can apply the single layer inference algorithm (Algorithm 1) to the cth layer.

This greedy setup is illustrated in FIG3 in Supplement 2.The per-layer inference derivation is a straightforward copy of the single layer case, yielding the following propositions.

Proposition 3.

The assignment cost specification for finding B j,c is: DISPLAYFORM1 where for simplicity we assume ⌃ DISPLAYFORM2 We combine these propositions and summarize the overall multilayer inference procedure in Algorithm 1 in Supplement 2.

In this section we propose an extension of our modeling framework to handle streaming data.

Such data naturally arises in many federated learning settings.

Consider, again the example of wearable devices.

Data recorded by sensors on these devices is naturally temporal and memory constraints typically require streaming processing of the data.

Bayesian paradigm naturally fits into the streaming scenario -posterior of step s becomes prior for step s + 1.

We generalize our single hidden layer model to streaming setting (our approach naturally extends to multilayer scenario).The differences in the generative model effect (2) and (3), which become: DISPLAYFORM0 We derive cost expression for the streaming extension in the Supplementary.

To verify our methodology we simulate federated learning scenarios using two standard datasets: MNIST and CIFAR-10.

We randomly partition each of these datasets into J batches.

Two partition strategies are of interest: (a) homogeneous partition when each batch has approximately equal proportion of each of the K classes; and (b) heterogeneous when batch sizes and class proportions are unbalanced.

We achieve the latter by simulating p k ⇠ Dir J (0.2) and allocating p k,j proportion of instances of class k to batch j. Note that due to the small concentration parameter (0.2) of the Dirichlet distribution, some sampled batches may not have any examples of certain classes of data.

For each pair of partition strategy and dataset we run 10 trials to obtain mean accuracies and standard deviations.

In our empirical studies below, we will show that our framework can aggregate multiple local neural networks (NNs) trained independently on different batches of data into an efficient, modest-size global neural network that performs competitively against ensemble methods and outperforms distributed optimization.

Baselines satisfying the constraints.

First, we will conduct experiments to demonstrate that PFNM is the best performing approach among methods restricted to single communication, compressed global model and no access to data after training of the local models.

Studying such constraints is not only important in the context of federated learning, but also to understand model averaging of neural networks in the parameter space.

A good neural network averaging approach may serve as initialization for Knowledge Distillation when additional data is available or for distributed optimization when it is possible to perform additional communication rounds.

BID15 FIG3 in their paper) showed that naive averaging of weights of two independently trained neural networks does not perform well, unless these weights were trained with same initial values.

We will show experimentally that even with shared initialization, Federated Averaging BID15 with single post-training communication quickly degrades for more than 2 networks and/or when trained on datasets with different class distributions.

On the contrary, PFNM does not require shared initialization and can produce meaningful average of many neural networks in the parameter space.

We also compare to nonparametric clustering of weight vectors based on DP-means BID9 .

This method is inspired by the Dirichlet Process mixtures BID3 BID0 , and may serve as an alternative approach to our Beta Process based construction.

We note that, to the best of our knowledge, DP-means has not been considered in such context previously.

Additionally, the average test set performance of the local models serves as a basic baseline.

Test set performance for varying number of batches and homogeneous and heterogeneous partitionings of MNIST are summarized in FIG6 .

PFNM with 0 = 1 (hyperparameters 2 0 = 10 and 2 = 1 are fixed across experiments) consistently outperforms all baselines.

We also consider a degenerate case of PFNM with 0 = 10 5 .

It can be seen from Proposition 2 that 0 /J controls the size of the global model and when set to very small value will result in the global model of the size of the local model, however potentially at a cost of performance quality.

In this experiment each of the J local neural networks is trained for 10 epochs and has 100 neurons (for Federated averaging we consider 300 neurons per local network to increase global model capacity, since this approach constraints the global model size to be equal to local model) and maximum number of neurons for PFNM and DP-means is truncated at 700.

FIG6 and 2d summarize global model sizes, which show significant compression over maximum possible 100J.

Our experiments demonstrate that it is possible to efficiently perform model averaging of neural networks in the parameter space by accounting for permutation invariance of the hidden neurons.

We reiterate that the global model learned by PFNM may either be used as final solution for a federated learning problem or serve as an initialization to obtain better performance with distributed optimization, Federated Averaging or Knowledge Distillation when some of our problem constraints are relaxed.

Baselines with extra resources.

We next consider four additional baselines, however each of them violates at least one of the three constraints of our federated learning problem, i.e. no data pooling, infrequent communication, and a modest-size global model.

Our goal iss to demonstrate that PFNM is competitive even when put at a disadvantage.

Uniform ensemble (U-Ens) (Dietterich, 2000) is a classic technique for aggregating multiple learners.

For a given test case, each batch neural network outputs class probabilities which are averaged across batches to produce the prediction of class probabilities.

The disadvantage of this approach is high computational cost at testing time since it essentially stacks all batch neural networks into a master classifier with P j,c L c j hidden units.

Weighted ensemble (W-Ens) is a heuristic extension for heterogeneous partitioning -where class k probability of batch j is weighted by the proportion of instances of class k on batch j when taking the average across batch network outputs.

Knowledge distillation (KD) is an extension of ensemble, where a new, modest size neural network is trained to mimic the behavior of an ensemble.

This, however, requires pooling training examples on the master node.

Our final baseline is the distributed optimization approach downpour SGD (D-SGD) of .

The limitation of this method is that it requires frequent communication between batch servers and the master node in order to exchange gradient information and update local copies of weights.

In our experiments downpour SGD was allowed to communicate once every training epoch (total of 10 rounds of communications), while our method and other baselines only communicated once, i.e. after the batch neural networks have been trained.

We compare Probabilistic Federated Neural Matching (PFNM) against the above four extra-resource baselines for varying number of batches J FIG7 .

When number of batches grows, average size of a single batch decreases and corresponding neural networks do not converge to a good solution (or result in a bad gradient after an epoch).

This significantly degrades performance of the downpour SGD and also affects PFNM in the case of heterogeneous CIFAR-10 FIG7 .

We observe that D-SGD at first improves with increasing number of batches and then drops down in performance abruptly -at first increasing number of batches essentially increases number of communications, since each batch sends gradients to the server, without hurting the quality of gradients, however when size of batches decreases gradients become worse and D-SGD behaves poorly.

On the other hand, ensemble approaches only require a collection of weak classifiers to perform well, hence their performance does not noticeably degrade as the quality of batch neural networks deteriorates.

This advantage comes at a price of high computational burden when making a prediction, since we need to do a forward pass for an input observation through each of the batch networks.

Interestingly, weighted ensemble performs worse than uniform ensemble on heterogeneous CIFAR-10 case -this again could be due to the low quality of batch networks which hurts our method and makes uniform ensemble more robust than weighted.

In the second experiment we fix J = 10 and consider multilayer batch neural networks with number of layers C from 1 to 6.

We see FIG8 that our multilayer PFNM can handle deep networks as it continues to be comparable to ensemble techniques and outperform D-SGD.

In the Supplementary we analyze sizes of the master neural network learned by PFNM, parameter sensitivity, streaming extension and explore performance of downpour SGD with more frequent communications.

We conclude that for federated learning applications when prediction time is limited (hence ensemble approaches are not suitable) and communication is expensive, PFNM is a strong solution candidate.

In this work we have developed models for matching fully connected networks, and experimentally demonstrated the capabilities of our methodology, particularly when prediction time is limited and communication is expensive.

We also observed the importance of convergent local neural networks that serve as inputs to our matching algorithms.

Poor quality local neural network weights will affect the quality of the master network.

In future work we plan to explore more sophisticated ways to account for uncertainty in the weights of small batches.

Additionally, our matching approach is completely unsupervised -incorporating some form of supervised signal may help to improve the performance of the global network when local networks are low quality.

Finally, it is of interest to extend our modeling framework to other architectures such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

The permutation invariance necessitating matching inference arises in CNNs too -any permutation of the filters results in same output, however additional bookkeeping is needed due to pooling operations.

Ohad Shamir, Nati Srebro, and Tong Zhang.

Communication-efficient distributed optimization using an approximate newton-type method.

In International conference on machine learning, pp.

The goal of maximum a posteriori (MAP) estimation is to maximize posterior probability of the latent variables: global atoms DISPLAYFORM0 and assignments of observed neural network weight estimates to global atoms {B j } J j=1 , given estimates of the batch weights DISPLAYFORM1 arg max DISPLAYFORM2 MAP estimates given matching (Proposition 1 in the main text) First we note that given {B j } it is straightforward to find MAP estimates of {✓ i } based on Gaussian-Gaussian conjugacy: DISPLAYFORM3 where L = max{i : DISPLAYFORM4 . .

, J} is the number of active global atoms, which is an (unknown) latent random variable identified by {B j }.

For simplicity we assume ⌃ 0 = I 2 0 , ⌃ j = I 2 j and µ 0 = 0.Inference of atom assignment.

We can now cast optimization corresponding to (1) with respect to only {B j } J j=1 .

Taking natural logarithm we obtain: DISPLAYFORM5 Let us first simplify the first term of (3):1 2 DISPLAYFORM6 We consider an iterative optimization approach: fixing all but one B j we find corresponding optimal assignment, then pick a new j at random and proceed until convergence.

In the following we will use notation j to say "all but j".

Let L j = max{i : B j i,l = 1} denote number of active global weights outside of group j.

We now rearrange (4) by partitioning it into i = 1, . . .

, L j and i = L j + 1, . . .

, L j + L j .

We are interested in solving for B j , hence we can modify objective function by subtracting terms independent of B j : DISPLAYFORM7 Now observe that P l B j i,l 2 {0, 1}, i.e. it is 1 if some neuron from batch j is matched to global neuron i and 0 otherwise.

Due to this we can rewrite (5) as a linear sum assignment problem: DISPLAYFORM8 Now we consider second term of FORMULA12 : DISPLAYFORM9 First, because we are optimizing for B j , we can ignore log P (B j ).

Second, due to exchangeability of batches (i.e. customers of the IBP), we can always consider B j to be the last batch (i.e. last customer of the IBP).

Let m j i = P j,l B j i,l denote number of times batch weights were assigned to global atom i outside of group j. We now obtain the following: DISPLAYFORM10 We now rearrange (7) as linear sum assignment problem: DISPLAYFORM11 Combining FORMULA20 and FORMULA25 we arrive at the cost specification for finding B j as minimizer of DISPLAYFORM12 , where: DISPLAYFORM13 This completes the proof of Proposition 2 in the main text.

FIG3 illustrates the overall multilayer inference procedure visually, and Algorithm 1 provides the details.

Nodes in the graphs indicate neurons, neurons of the same color have been matched.

On the left, the individual layer matching approach is shown, consisting of using the matching assignments of the next highest layer to convert the neurons in each of the J servers to weight vectors referencing the global previous layer.

These weight vectors are then used to form a cost matrix, which the Hungarian algorithm then uses to do the matching.

Finally, the matched neurons are then aggregated and averaged to form the new layer of the global model.

As shown on the right, in the multilayer setting the resulting global layer is then used to match the next lower layer, etc.

until the bottom hidden layer is reached FIG3 ,... in order).

In this section we present inference for the streaming extension of our model described in Section 3.3 of the main text.

Bayesian paradigm naturally fits into the streaming scenario -posterior of step s becomes prior for step s + 1: DISPLAYFORM0 The cost for finding B j,s becomes: DISPLAYFORM1 where first case is for i  L s j and m DISPLAYFORM2 i,l is the popularity of global atom i in group j up to step s.

We note that log P (B j,s |B j,s , DISPLAYFORM3 ) is not available in closed form in the Bayesian nonparametric literature, to the best of our knowledge, and we replaced corresponding terms in the cost with a heuristic.

Code to reproduce our results will be released after the review period.

Below are the details of the experiments.

Data partitioning.

In the federated learning setup, we analyze data from multiple sources, which we call batches.

Data on the batches in general does not overlap and may have different distributions.

To simulate federated learning scenario we consider two partition strategies of MNIST and CIFAR-10.

For each pair of partition strategy and dataset we run 10 trials to obtain mean accuracies and standard deviations.

The easier case is homogeneous partitioning, i.e. when class distributions on batches are approximately equal as well as batch sizes.

To generate homogeneous partitioning with J batches we split examples for each of the classes into J approximately equal parts to form J batches.

In the heterogeneous case batches are allowed to have highly imbalanced class distributions as well as highly variable sizes.

To simulate heterogeneous partition, for each class k, we sample p k ⇠ Dir J (0.2) and allocate p k,j proportion of instances of class k of the complete dataset to batch j. Note that due to small concentration parameter, 0.2, of the Dirichlet distribution, some batches may entirely miss examples of a subset of classes.

Batch networks training.

Our modeling framework and ensemble related methods operate on collection of weights of neural networks from all batches.

Any optimization procedure and software can be used locally on batches for training neural networks.

We used PyTorch BID21 as software framework and Adam optimizer BID20 with default parameters unless otherwise specified.

For reproducibility we summarize all parameter settings in TAB1 .

We first formally define the ensemble procedure.

Letŷ j 2 K 1 denote probability distribution over K classes output by neural network trained on data from batch j for some test input x. Then uniform ensemble prediction is arg max DISPLAYFORM0 To define weighted ensemble, let n j,k denote number of examples of class k on batch j and n k = P J j=1 n j,k denote total number of examples of class k across all batches.

Prediction of the weighted ensemble is as follows arg max k 1 n k P J j=1 n j,kŷj,k .

This is a heuristic approach we defined to potentially better handle heterogeneous partitioning with ensemble.

Knowledge distillation approach trains a new master neural network to minimize cross entropy between output of the master neural network and outputs of the batch neural networks.

The architecture of the master neural network has to be set manually -we use 500 neurons per layer.

Note that PFNM infers the number of neurons per layer of the master network from the batch weights.

For the knowledge distillation approach it is required to pool input data from all of the batches to the master server.

For training master neural network we used PyTorch, Adam optimizer and parameter settings as in TAB1 For the downpour SGD we used PyTorch, Adam optimizer and parameter settings as in TAB1 for the local learners.

Master neural network was also optimized with Adam and same learning rate as in the TAB1 .

Weights of the master neural network were updated in the end of every epoch (total of 10 rounds of communication) and then sent to each of the local learners to proceed with the next epoch.

Note that with this approach global network and networks for each of the batches are bounded to have identical number of neurons per layer, which is 50 in our experiments.

We tried increasing number of neurons per layer, however did not observe any performance improvements.

Master network size of PFNM.

Our model for matching neural networks is nonparametric and hence can infer appropriate size of the master network from the batch weights.

The "discovery" of new neurons is controlled by the second case of our cost term expression in (9), i.e. when L j < i  L j + L j .

In practice however we want to avoid impractically large master networks, hence we truncate the largest possible value of i in the cost computation to min(L j +L j , max(L j , 700)+1).

This means that when global network has 700 or more neurons, we only allow for it to grow by 1 in a single multibatch Hungarian algorithm iteration.

In FIG6 we summarize network sizes learned by PFNM in experiments corresponding to increasing number of batches ( FIG6 of the main text) and increasing number of hidden layers ( FIG7 of the main text).

The maximum possible size is 50JC (because of 50 neurons per batch per layer), which is practically the size of the master model of the ensemble approaches.

We see that size of the master network of PFNM is noticeably more compact than simply stacking batch neural networks.

The saturation around 700 neurons in FIG6 is due to the truncation procedure described previously.

for batches to send their gradients to the master server and 10J for the master server to send copy of the global neural network to each of the batch neural networks.

In our federated learning problem setup frequent communication is discouraged, however it is interesting to study the minimum number of communications needed for D-SGD to produce competitive result.

To empirically quantify this we show test accuracy of D-SGD with increasing number of communication rounds on MNIST with heterogeneous partitioning and J = 25 FIG7 .

PFNM and ensemble based methods are shown for comparison -they communicate only once (post batch networks training) in all of our experiments.

We see that in this case D-SGD requires more than 4000 communication rounds (8000J communications) to produce good result.

Such large amount of communication is impossible in practice for the majority of federate learning scenarios.

Streaming federated learning experiment To simulate streaming federated learning setup we partition data into J groups using both homogeneous and heterogeneous strategies as before.

Then each group is randomly split into S parts.

At step s = 1, . . .

, S the part of data indexed by s from each of the groups is revealed and used for updating the models.

For this experiment we consider heterogeneous partitioning of MNIST with J = 3 batches and S = 15 steps.

On every step we evaluate accuracy on the test dataset and summarize performance of all methods in FIG7 .

For our method, PFNM-Streaming, we perform matching based on the cost computations from Section 3 to update the global neural network.

We initialize weights of the j-th batch neural network for the next step s + 1 according to the model posterior after s steps (which is the prior for step s + 1): subsample L s+1 j 0 neurons from the global network according to popularity counts {m s j,i } i and concatenate with 0 neurons initialized with µ 0 0 (prior mean before any data is observed, which is set to 0), then add small amount of Gaussian noise.

For D-SGD we update global neural network weights after each step and then use these weights as initialization for batch neural networks on the next step.

To extend ensemble based methods to streaming setting we simply update local neural networks sequentially as the new data becomes available and use them to evaluate ensemble performance at each step.

Our models presented in Section 3 of the main text have three parameters , is the prior variance of weights of the global neural network.

Second parameter, 0 , controls discovery of new neurons and correspondingly increasing 0 increases the size of the learned master network.

The third parameter, 2 , is the variance of the local neural network weights around corresponding master network weights.

We analyze empirically effect of these parameters on the accuracy for single hidden layer model with J = 25 batches in FIG8 .

The heatmap indicates the accuracy on the training data -we see that for all parameter values considered performance doesn't not fluctuate significantly.

PFNM appears to be robust to choices of 2 0 and 0 , which we set to 10 and 1 respectively in all of our experiments.

Parameter 2 has slightly higher impact on the performance and we set it using training data during experiments.

To quantify importance of 2 for fixed 2 0 = 10 and 0 = 1 we plot average train data accuracies for varying 2 in Figure 5 .

We see that for homogeneous partitioning and one hidden layer 2 has almost no effect on the performance (Fig. 5a and Fig. 5c ).

In the case of heterogeneous partitioning (Fig. 5b and Fig. 5d ), effect of 2 is more noticeable, however all considered values result in competitive performance.

@highlight

We propose a Bayesian nonparametric model for federated learning with neural networks.

@highlight

Uses beta process to do federated neural matching.

@highlight

The paper considers federate learning of neural networks, where data is distributed on multiple machines and the allocation of data points is potentially inhomogenous and unbalanced.