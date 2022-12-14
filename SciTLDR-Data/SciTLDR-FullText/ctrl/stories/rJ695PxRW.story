The assumption that data samples are independently identically distributed is the backbone of many learning algorithms.

Nevertheless, datasets often exhibit rich structures in practice, and we argue that there exist some unknown orders within the data instances.

Aiming to find such orders, we introduce a novel Generative Markov Network (GMN) which we use to extract the order of data instances automatically.

Specifically, we assume that the instances are sampled from a Markov chain.

Our goal is to learn the transitional operator of the chain as well as the generation order by maximizing the generation probability under all possible data permutations.

One of our key ideas is to use neural networks as a soft lookup table for approximating the possibly huge, but discrete transition matrix.

This strategy allows us to amortize the space complexity with a single model and make the transitional operator generalizable to unseen instances.

To ensure the learned Markov chain is ergodic, we propose a greedy batch-wise permutation scheme that allows fast training.

Empirically, we evaluate the learned Markov chain by showing that GMNs are able to discover orders among data instances and also perform comparably well to state-of-the-art methods on the one-shot recognition benchmark task.

Recent advances in deep neural networks offer great potentials for machines to learn automatically without humans interventions.

For instance, Convolutional Neural Networks (CNNs) BID16 provided an automated way for learning image feature representations.

Compared to hand-crafted ones such as SIFT and SURF, these hierarchical deep features demonstrate superior performance in recognition BID35 and transfer learning BID7 problems.

Another example would be learning to learn for automatic parameter estimation.

BID0 proposed to update model parameters without any pre-defined update rule such as stochastic gradient descent (SGD) or ADAM .

Surprisingly, this update-rule-free framework showed better performance and faster convergence on both object recognition and image style transformation tasks.

In our paper, we investigate the following novel question: given an unordered dataset where instances may be exhibiting some implicit order, can we order a dataset automatically according to this order?We argue that such order often exists even when we are dealing with the data that are naturally thought of as being i.i.d.

sampled from a common though complex distribution.

For example, let's consider a dataset consisting of the joint locations on the body of the same person taken on different days.

The data i.i.d.

assumption is justified since postures of a person took on different days are likely unrelated.

However, we can arrange the data instances such that the joints follow an articulated motion or a set of motions in a way that makes each pose highly predictable given the previous ones.

Although this arrangement depends on the person as ballerinas' poses might obey different dynamics than the poses of tennis players, the simultaneous inference on the pose dynamics can lead to a robust model that explains the correlations among joints.

To put it differently, if we reshuffle the frames of a video clip, the data can now be modeled by an i.i.d.

model.

Nevertheless, reconstructing the order leads to an alternative model where transitions between the frames are easier to fit the links between the latent structures and observations.

The ballerina's dancing, if sampled very sparsely, can be thought of as a reshuffled video sequence that needs to be reordered such that a temporal model can generate it.

One naive and obvious way to find the order in a dataset is to perform sorting based on a predefined distance metric; e.g., the Euclidean distance between image pixel values.

However, the distance metrics have to be predefined differently and empirically according to distinct types/characteristics of the datasets at hand.

A proper distance metric for one domain may not be a good one for other domains.

For instance, p distance is a good measure for DNA/RNA sequences BID23 while it does not characterize the semantic distances between images.

We argue that the key component of the ordering problem lies in the discovery of proper distance metric in an automatic and adaptive way.

To approach this problem, we propose to learn a distance-metric-free model to discover the ordering in the dataset.

More specifically, we model the data by treating them as if they were generated from a Markov chain.

We propose to simultaneously train the transitional operator and find the best order by a joint optimization over the parameter space as well as all possible permutations.

We term our model Generative Markov Networks (GMNs).

One of the key ideas in the design of GMNs is to use neural networks as a soft lookup table to approximate the possibly huge but discrete transition matrix.

This strategy allows GMNs to amortize the space complexity using a unified model.

Furthermore, due to the differentiable property of neural networks, the transitional operator of GMNs can also generalize on unseen but similar data instances.

As an additional contribution, to ensure the Markov chain learned by GMNs is ergodic, we propose a greedy batch-wise permutation scheme that allows fast training.

One related task is one-shot recognition which has only one labeled data per category in the target domain.

Most of the work in this area considered learning a specific distance metric BID15 BID33 BID28 or category-separation metric BID24 for the data.

During the inference phase, they computed either the smallest distance or highest class prediction score between the support and query instances.

Alternatively, from a generative modeling perspective, we can first generate the Markov chain for the support instances, then we fit the query instances into the Markov chain and decide the labels with the highest log-likelihood.

Empirically, we evaluate the learned Markov chain by showing that GMNs are able to discover implicit orders among data instances and also perform comparably well to state-of-the-art methods on the benchmark one-shot recognition task.

The literature on deep generative models and stochastic sampling is abundant.

Due to the space limit, we discuss the ones that are most relevant to our work.

We consider two classes of deep generative models based on ancestral sampling and iterative sampling, respectively.

Variational Autoencoders (VAEs) BID14 and Generative Adversarial Networks (GANs) BID8 can be cast as ancestral sampling-based methods.

In the inference phase, these approaches generated one sample from the model by performing a single inference pass from the underlying graphical models.

As a comparison, methods based on iterative sampling performed multiple and iterative passes through all the variables in the corresponding graphical models.

Usually, these methods involved simulating a Markov chain in the entire state space, and they aimed at improving quality of generated samples by mixing the underlying chain.

Recent works on this line of research included BID1 BID29 BID3 BID30 .Our approach can be categorized as an iterative sampling-based model.

However, it has three significant differences comparing to previous works.

First, all the existing works assumed that training instances are i.i.d.

sampled from the stationary distribution of a Markov chain.

This assumption is risky since, often the case, it is hard to measure whether a Markov chain has mixed or not.

On the contrary, we only assume that data instances are sampled from the chain, without expecting the chain has mixed.

As we will see later, the stationarity assumption in previous works often prevents them from observing the implicit data relationships.

Second, prior approaches were proposed based on the notion of denoising models.

In other words, their goal was generating high-quality images; on the other hand, we aim at discovering orders in datasets.

Third, to the best of our knowledge, all the existing works were implicit models in the sense that they only admitted efficient sampling schemes.

In contrast, the proposed GMN is an explicit model where besides an efficient sampling procedure, the model maintains a tractable likelihood function that can be computed efficiently.

One-Shot Learning: Deep one-shot learning approaches could be divided into two categories:distance-metric-learning and categories-separation-metric-learning approaches.

The former aimed at either learning a similarity measurement between instance pairs BID15 or applying specific metric loss based on cosine distance BID33 )/ Euclidean distance BID28 .

These methods referred to nonparametric classifiers and relied heavily upon human design.

As a comparison, methods in the second category offered more generalities.

Typically, this type of methods tackled the problem using a meta-learning framework to train parametric classifiers.

Precisely, they considered two levels of learning: the first stage is to update base learners' parameters and the second stage is to update parameters for the meta learner.

Recent works BID24 BID12 belonged to this category.

The methods mentioned above viewed one-shot recognition as a discriminative task; on the contrary, we hold a generative perspective.

Since we consider a Markov chain data generation assumption, we can directly decide the labels for query instances by fitting them into the Markov chain (or the orders we observe) generated from support instances.

This generative nature significantly decreases the difficulty of training as we no longer rely on any designed metric.

More details will be covered in Sec. 4.

Let {s i } n i=1 denote our training data which are assumed being generated from an unknown Markov chain.

Our goal is to jointly recover the unknown Markov chain as well as the order of generation process.

Note that since the generation order is unknown, even if the true Markov chain was given, it would still be computationally intractable to find the optimal order that best fits our data.

To get around of this intrinsic difficulty, as we will see in Sec. 3.2 , we propose a greedy algorithm to find an order given the current estimation of the transitional operator.

We denote the underlying data order to be a permutation over [n]: ??? = {???(t)} n t=1 , where ???(t) represents the index of the instance that is generated at the t-th step of the Markov chain.

In other words, a Markov chain is formed as follows: DISPLAYFORM0 We consider all the possible permutations ??? and arbitrary distribution over these permutations, which leads to a joint log-likelihood estimation problem: DISPLAYFORM1 where P(1) (??) is the initial distribution of the Markov chain and T (s 0 |s; ???) is the transitional operator parametrized by model parameters ???.

Note that the effect of the initial distribution P(1) (??) diminishes with the increase of the data size n. Hence, without loss of generality, we assume P DISPLAYFORM2 is uniform over all possible states, leading to the following optimization problem: DISPLAYFORM3 where ???(n) is the set of all possible permutations over [n].

Unfortunately, direct optimization of FORMULA3 is computationally intractable.

For each fixed ??? and P, the number of all possible permutations (i.e., |???(n)|) is n!. To approximate this expensive function, we present an efficient greedy algorithm in Sec. 3.3.

In practice, when the state space is huge, often we cannot afford to maintain the tabular transition matrix directly, which takes up to O(d 2 ) space, where d is the number of states in the chain.

For example, if the state refers to a binary image I 2 {0, 1} p , the size of the state space is d = 2 p which is nearly infeasible to compute.

Hence, before optimizing (1), we should first find a family of functions to parametrize the transitional operator T (??|??).Being universal function approximators BID9 , neural networks could be used to approximate the discrete structures which led to the recent success of deep reinforcement learning BID21 .

In our case, we utilize neural networks to approximate the discrete tabular transition matrix.

The advantages are two-fold: first, it significantly reduces the space complexity by amortizing the space required by each separate state into a unified model.

Since all the states share the same model as the transitional operator, there is no need to store the transition vector for each separate state explicitly.

Second, neural networks allow better generalization for the transition probabilities across states.

The reason is that, in most real-world applications, states, represented as feature vectors, are not independent from each other.

As a result, the differentiable approximation to a discrete structure has the additional smoothness properties, which allows the transitional operator to have a good estimate even for the unseen states.

Let ??? be the parameters of the neural networks and we can define DISPLAYFORM0 to be the transition function that takes two states s and s 0 as inputs and returns the corresponding transition probability.

Note that one can consider each discrete transitional operator as a lookup table; for example, we use s and s 0 to locate the corresponding row and column of the table and read out its probability.

From this perspective, the neural network works as a soft lookup table that outputs the transition probability given two states (features).

As mentioned above, the direct evaluation of eq. FORMULA3 is computationally intractable given P and ???.

Here, we develop a coordinate ascent style training algorithm to optimize eq. (1) efficiently.

The key insight comes from the following observation: for each fixed ???, there exists a point mass distribution over ???(n) that achieves the maximum value for eq. (1).

More precisely, DISPLAYFORM0 .

We leave the proof in Supplementary.

In other words, given each ???, the optimization problem over ??? now reduces to finding the optimal permutation ??? ??? that gives the maximum likelihood on generating the data.

However, without further assumption on the structure of the transitional operator, this is still a hard problem which takes time O(n!).

Instead, we propose a greedy algorithm to approximate the optimal order, which takes time O(n 2 log n).

We list the pseudocode in Alg.

1.

At first, Alg.

1 enumerates all the possible states appearing in the first time step.

For each of the following steps, it finds the next state by maximizing the transition probability at the current step, i.e., a local search to find the next state.

The final approximate order is then defined to be the maximum of all these n orders.

A naive implementation of this algorithm has time complexity O(n 3 ).

However, we can reduce it to O(n 2 log n) by pre-computing T (s i |s j ; ???), 8i, j 2 [n] and sorting them so that the maximum finding operation in line 5 can be done in constant time.

Given the approximate order???, we then proceed to optimize the model parameter ??? by gradient based optimization.

By now it should be clear that the whole algorithm is an instance of the famous coordinate ascent algorithm, where we alternatively optimize over the order ??? and the model parameters ???.

Since both optimizations over ??? and ??? will not decrease the objective function, the algorithm is guaranteed to converge.

The O(n 2 log n) computation to find the approximate order in Alg.

1 can be expensive when the size of the data is large.

In this section we provide batch-wise permutation training to avoid this

Input: Input data {si} n i=1 and transitional operator T (si|sj; ???) DISPLAYFORM0 for j = 2 to n do 5:???i(j) max DISPLAYFORM1 end for 7:vi DISPLAYFORM2 end if 12: end for 13: return??? Algorithm 2 Optimization with Batch-Wise Permutation Training DISPLAYFORM3 end if 7:Compute??? (k) using the Greedy Approximate Order (Alg.

1) DISPLAYFORM4 ) 10: end for issue.

The idea is to partition the original training set into batches with size b and perform greedy approximate order on each batch.

Assuming b ??? n is a constant, the effective time complexity becomes: O(b 2 log b) ?? n/b = O(nb log b), which is linear in n. However, since training data are partitioned into chunks, the learned transitional operator is not guaranteed to have nonzero transition probabilities between different chunks of data.

In other words, the learned transitional operator does not necessarily induce an ergodic Markov chain due to the isolated states.

To avoid this problem, we propose a simple strategy to enforce some samples are overlapping between the consecutive batches.

We show the pseudocode in Alg.

2.

In Alg.

2, b means the batch size, is the learning rate and b 0 < b is the number of overlap states between consecutive batches.

In this section we give a detailed description on how to implement the transitional operator where the state can be both discrete or continuous.

At the first step, to prevent our GMNs from simply memorizing all the training data and their transitions, we introduce stochastic latent variables z 2 R z via Variational Bayes Inference BID34 .

The evidence lower bound (ELBO) of the log likelihood for the transitional operator (i.e., log T (s 0 |s; ???)) becomes: DISPLAYFORM0 where T (s 0 |s; ???) has been replaced by a distribution P(s 0 |s, z; ) parametrized by , which allows us to make the dependence of s on z. Moreover, KL is the KL-divergence, Q(z|s; ) is an encoder function parametrized by that encodes latent code z given current state s, and P(z) is a fixed prior which we take its form as Gaussian distribution N (0, I).

We use reparametrized trick to draw Q(z|s; ) from Gaussian N ?? Q, (s), 2 Q, (s)I where ?? Q, (s) and Q, (s) are learnable functions.

Next, we consider two types of distribution family for P(s 0 |s, z; ???): Bernoulli and Gaussian.

If s 2 {0, 1} p (i.e., a binary image), we define log P(s 0 |s, z; ) as: DISPLAYFORM1 where is element-wise multiplication and g (s, z) : DISPLAYFORM2 If s 2 R p (i.e., a real-valued feature vector), we choose P(s 0 |s, z; ) to be fixed variance factored DISPLAYFORM3 ??? , where ?? P, (s, z) : R p+z !

R p and P is a fixed variance.

We simply choose P in all the experiments.

log P(s 0 |s, z; ???) can thus be defined as DISPLAYFORM4 where const. is not related to the optimization of .For simplicity, we specify ??? = { [ }.

Therefore, the model parameters update for ??? in (2) refers to the updates for and .

We perform experiments on ordering data in three datasets: MNIST (LeCun et al., 1990), Horse , and MSR SenseCam BID11 .

We also provide another experiment on Moving MNIST in Supplementary.

Among these datasets, MNIST, Horse, and MSR SenseCam do not have explicit orders.

On the other hand, Moving MNIST can be seen as a collection of short video clips, and thus each sequence of frames has an explicit order.

Due to the space limit, we only show partial ordering results.

Please see Supplementary for the full version.<MNIST> MNIST BID17 ) is a well-studied dataset that contains 60,000 training examples.

Each example is a digit image with size 28x28.

We rescale the pixel values to [0, 1].

Note that since MNIST contains a large number of instances, we perform the ordering in a randomly sampled batch to demonstrate our results.<Horse> Horse dataset consists of 328 horse images collected from the Internet.

Each horse is centered in a 30x40 image.

For the preprocessing, the object-background segmentation is applied, and the binary pixel value is set to 1 and 0 for object and background, respectively.

Examples are show in Supplementary.<MSR SenseCam> MSR SenseCam BID11 ) is a dataset consisting of images taken by SenseCam wearable camera.

It contains 45 classes with approximately 150 images per class.

Each image has size 480x640.

We resize each image into 224x224 and extract the feature from VGG-19 network BID27 .

In this dataset, we consider only office category which has 362 images.

We apply Alg.

2 to train our Generative Markov Networks.

When the training converges, we plot the images following permutation??? in Alg.

1. Note that??? can be seen as the implicit order suggested by GMNs.

For comparison, we also plot the images following nearest neighbor sorting using Euclidean distances.

The parameters {b overlap , b, t} in Alg.

2 are {50, 500, 600}, {328, 328, 1}, and {362, 362, 1} for MNIST, Horse, and MSR SenseCam, respectively.

Network architectures for parameterizing T (??|??; ???) are specified in Supplementary.

The results are shown in FIG0 .

We first observe that data following the order suggested by our proposed GMN have visually high autocorrelation.

This result implies that our proposed GMN can discover nice implicit orders for the dataset.

Comparing to the strong ordering baseline Nearest Neighbor sorting, one could hardly tell which one is better.

Nevertheless, GMN is a distance-metricfree model which requires no predefined distance metric.

Moreover, the implicit order suggested by GMN considers a generative modeling viewpoint: the order is the optimal permutation under the Markov chain data generation assumption (see Sec. 3.2).

Next, we examine the data generation using the learned transitional operator.

Conditioned on a given sample s, instead of sampling s 0 ??? T (s 0 |s; ???) directly, we sample s DISPLAYFORM0 .

We make this modification based on the reason that our model aims at discovering datasets' orders, while other iterative sampling models BID1 BID29 BID3 BID30 intended to denoise generated samples.

Similar to Sec. 4.1.1, we exploit nearest neighbor search using Euclidean distance for comparison.

More precisely, s 0 NN = arg max FIG1 illustrates the sampling of GMN and Nearest Neighbor search.

We can see that Nearest Neighbor search is not able to perform efficient sampling since it would stick between two similar images.

On the other hand, our proposed GMN can perform consecutive sampling.

This tremendous difference implies the distinction between the discriminative (sampling by a fixed distance metric) and the generative (sampling through the transitional operator in a Markov chain) model.

DISPLAYFORM1

Now, we perform one-shot recognition task on the miniImageNet BID33 BID24 , which is a benchmark dataset designed for the evaluation of few-shot learning BID33 BID24 .

Being a subset of ImageNet BID25 , it contains 100 classes and each class has 600 images.

Each image is downsampled to size 84x84.

As suggested in BID24 , the dataset is divided into three parts: 64 classes for training, 16 classes for validation, and 20 classes for testing.

Identical to BID24 , we consider the 5 way 1 shot problem.

That is, from testing classes, we sample 5 classes with each class containing 1 labeled example.

The labeled examples refer to support instances.

Then, we randomly sample 500 unlabeled query examples in these 5 classes for evaluation.

We repeat this procedure for 10, 000 times and report the average with 95% confidence intervals in Tbl.

1.

Instead of viewing one-shot recognition as a discriminative task, we hold it as a generative one.

To achieve this goal, we train our Generative Markov Networks on training classes and then apply it to testing classes.

More precisely, for each training episode, we sample 1 class from the training classes and let {s i } n i=1 be all the data from this class.

Then, we apply Alg.

2 with {b overlap , b, t} = {20, 100, 10}. We consider 3, 000 training episodes.

On the other hand, for each testing episode, we apply GMNs to generate a chain from each support instance:s DISPLAYFORM0 , where s c 0 is the support instance belonging to class c ands c is the generated samples from the Markov chain.

Next, we fit each query example into each chain by computing the average approximating loglikelihood.

Namely, the probability for generating the query sample s q in the chain of class c is DISPLAYFORM1 In a generative viewpoint, the predicted class?? for s q is determined b?? c = arg max c P(s q |c).For fair comparisons, we use the same architecture specified in BID24 to extract 1600-dimensional features.

We pretrain the architecture using standard softmax regression on image-label pairs in training and validation classes.

The architecture consists of 4 blocks.

Each block comprises a CNN layer with 64 3x3 convolutional filters, Batch Normalization BID10 layer, ReLU activation, and 2x2 Max-Pooling layer.

Then, we train our Generative Markov Networks based on these 1, 600 dimensional features.

Network architecture for parameterizing T (??|??; ???) is specified in Supplementary.

For a comprehensive analysis, we also provide the variant of our GMN with fine-tuning.

In other words, we fine-tune GMN by applying Alg.

2 with {b overlap , b, t} = {20, 100, 10} on support and query instances.

Note that in eq. (3), k is chosen to be 1 and 5 for the non-fine-tuned and fine-tuned version, respectively.

We compare our proposed method and the related approaches in Tbl.

1, in which Basic model refers the architecture in BID24 and Advanced models refer to more complicated designs.

Generally, it is not fair to compare the methods using different models; therefore, we only discuss the methods using Basic model in the following.

First, we observe that the performance of GMN is comparable to other works.

For example, the best result of all methods is reported by Meta-SGD BID22 with 50.47 ?? 1.87.

Although GMN suffers from slight performance drop, it requires a much less computational budget.

The reason is that the meta-learning (parametric) approaches BID24 BID22 BID18 BID20 rely on huge networks to manage complicated intersections between meta and base learners, while parameters for GMN exist only in ??? which is a relatively tiny network.

On the other hand, the best performance reported in the distance-metric learning (nonparametric) approaches is Prototypical Networks BID28 with 49.42 ?? 0.78.

Sacrificing from little performance deterioration, our proposed GMN enjoys more flexibility without the need of defining any distance metric as in BID33 BID15 BID32 BID28 BID26 BID19 .

More importantly, except for our proposed GMN, all the works belong to discriminative models, which means they are optimized based on carefully chosen objectives for one-shot learning purpose.

Next, our proposed GMN enjoys a significant improvement (45.36 ?? 0.94 !

48.87 ?? 1.10) from fine-tuning over support and query instances.

This result verifies that GMN is able to simulate the Markov chain data generation process since the query instances can be better fitted in the chains generated from the support instances.

In this paper, we argue that data i.i.d.

assumption is not always the case in most of the datasets.

Often, data instances are exhibiting some implicit orders which may benefit our understanding and analysis of the dataset.

To observe the implicit orders, we propose a novel Generative Markov Network which considers a Markov chain data generation scheme.

Specifically, we simultaneously learn the transitional operator as a generative model in the Markov chain as well as find the optimal orders of the data under all possible permutations.

In lots of experiments, we show that our model is able to observe implicit orders from unordered datasets and also perform well on the one-shot recognition task.

Paper under double-blind review 1 PROOF FOR SEC.

3.2Here, we prove that DISPLAYFORM0

For any ??? over ???(n), we have: DISPLAYFORM0 is the indicator function that takes value 1 iff ??? = ??? ??? otherwise 0.

Realize that I(??? = ??? ??? ) also defines a valid distribution over ???(n), which proves our claim.

<Moving MNIST> Moving MNIST contains 10, 000 sequences each of length 20 showing 2 digits moving in a 64x64 frame.

We rescale the pixel values to [0, 1] .

For each training episode, we apply Alg.

2 to train GMN on one randomly chosen sequence with parameters {b overlap , b, t} set as {0, 20, 10}. We consider 6, 000 training episodes.

For evaluation, we randomly sample a disjoint sequence from training sequences and observe the optimal permutation (implicit order) from Alg.

1.

Fig. 9 illustrates the results for the implicit order observed from Generative Markov Networks, the order inferred from Nearest Neighbor sorting using Euclidean distance, and the suggested explicit order.

We find that both the orders observed from GMN and NN sorting manifest smooth motions for two digits in the frame.

It is worth noting that our proposed GMN enjoys the freedom of not defining any distance metric.

The sampling results for Moving MNIST dataset are shown in FIG0 .

We consider two approaches: the proposed Generative Markov Networks and Nearest Neighbor search.

We find that, by learning the transition operator in a Markov chain as a generative model, GMN performs much better sampling results than Nearest Neighbor search which is a discriminative model.

We elaborate the design of the transition operator in FIG0 .

In our design, U can be seen as a gating mechanism between input X t and the learned updateX. More precisely, the output can be written as DISPLAYFORM0 where denotes element-wise product.

We specify each function f in Tbl.

1, 2, 3, 4, and 5. Note that we omit the bias term for simplicity.

We use ADAM with learning rate 0.001 and 0.2 dropout rate to train our T (??|??; ???).

<|TLDR|>

@highlight

Propose to observe implicit orders in datasets in a generative model viewpoint.

@highlight

The authors deal with the problem of implicit ordering in a dataset and the challenge of recovering it and propose to learn a distance-metric-free model that assumes a Markov chain as the generative mechanism of the data 

@highlight

The paper proposes ???Generative Markov Networks??? - a deep-learning-based approach to modeling sequences and discovering order in datasets.

@highlight

Proposes learning the order of an unordered data sample by learning a Markov chain.