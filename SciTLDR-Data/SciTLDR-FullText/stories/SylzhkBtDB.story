We investigate multi-task learning approaches which use a shared feature representation for all tasks.

To better understand the transfer of task information, we study an architecture with a shared module for all tasks and a separate output module for each task.

We study the theory of this setting on linear and ReLU-activated models.

Our key observation is that whether or not tasks' data are well-aligned can significantly affect the performance of multi-task learning.

We show that misalignment between task data can cause negative transfer (or hurt performance) and provide sufficient conditions for positive transfer.

Inspired by the theoretical insights, we show that aligning tasks' embedding layers leads to performance gains for multi-task training and transfer learning on the GLUE benchmark and sentiment analysis tasks; for example, we obtained a 2.35% GLUE score average improvement on 5 GLUE tasks over BERT LARGE using our alignment method.

We also design an SVD-based task re-weighting scheme and show that it improves the robustness of multi-task training on a multi-label image dataset.

Multi-task learning has recently emerged as a powerful paradigm in deep learning to obtain language (Devlin et al. (2018) ; Liu et al. (2019a; b) ) and visual representations (Kokkinos (2017) ) from large-scale data.

By leveraging supervised data from related tasks, multi-task learning approaches reduce the expensive cost of curating the massive per-task training data sets needed by deep learning methods and provide a shared representation which is also more efficient for learning over multiple tasks.

While in some cases, great improvements have been reported compared to single-task learning (McCann et al. (2018) ), practitioners have also observed problematic outcomes, where the performances of certain tasks have decreased due to task interference (Alonso and Plank (2016) ; Bingel and Søgaard (2017) ).

Predicting when and for which tasks this occurs is a challenge exacerbated by the lack of analytic tools.

In this work, we investigate key components to determine whether tasks interfere constructively or destructively from theoretical and empirical perspectives.

Based on these insights, we develop methods to improve the effectiveness and robustness of multi-task training.

There has been a large body of algorithmic and theoretical studies for kernel-based multi-task learning, but less is known for neural networks.

The conceptual message from the earlier work (Baxter (2000) ; Evgeniou and Pontil (2004) ; Micchelli and Pontil (2005) ; Xue et al. (2007) ) show that multi-task learning is effective over "similar" tasks, where the notion of similarity is based on the single-task models (e.g. decision boundaries are close).

The work on structural correspondence learning (Ando and Zhang (2005) ; Blitzer et al. (2006) ) uses alternating minimization to learn a shared parameter and separate task parameters.

Zhang and Yeung (2014) use a parameter vector for each task and learn task relationships via l 2 regularization, which implicitly controls the capacity of the model.

These results are difficult to apply to neural networks: it is unclear how to reason about neural networks whose feature space is given by layer-wise embeddings.

To determine whether two tasks interfere constructively or destructively, we investigate an architecture with a shared module for all tasks and a separate output module for each task (Ruder (2017) ).

See Figure 1 for an illustration.

Our motivating observation is that in addition to model similarity which affects the type of interference, task data similarity plays a second-order effect after controlling model similarity.

To illustrate the idea, we consider three tasks with the same number of data samples where task 2 and 3 have the same decision boundary but different data distributions (see Figure 2 for an illustration).

We observe that training task 1 with task 2 or task 3 can either improve or hurt task 1's performance, depending on the amount of contributing data along the decision boundary!

This observation shows that by measuring the similarities of the task data and the models separately, we can analyze the interference of tasks and attribute the cause more precisely.

Motivated by the above observation, we study the theory of multi-task learning through the shared module in linear and ReLU-activated settings.

Our theoretical contribution involves three components: the capacity of the shared module, task covariance, and the per-task weight of the training procedure.

The capacity plays a fundamental role because, if the shared module's capacity is too large, there is no interference between tasks; if it is too small, there can be destructive interference.

Then, we show how to determine interference by proposing a more fine-grained notion called task covariance which can be used to measure the alignment of task data.

By varying task covariances, we observe both positive and negative transfers from one task to another!

We then provide sufficient conditions which guarantee that one task can transfer positively to another task, provided with sufficiently many data points from the contributor task.

Finally, we study how to assign per-task weights for settings where different tasks share the same data but have different labels.

Our theory leads to the design of two algorithms with practical interest.

First, we propose to align the covariances of the task embedding layers and present empirical evaluations on well-known benchmarks and tasks.

On 5 tasks from the General Language Understanding Evaluation (GLUE) benchmark (Wang et al. (2018b) ) trained with the BERT LARGE model by Devlin et al. (2018) , our method improves the result of BERT LARGE by a 2.35% average GLUE score, which is the standard metric for the benchmark.

Further, we show that our method is applicable to transfer learning settings; we observe up to 2.5% higher accuracy by transferring between six sentiment analysis tasks using the LSTM model of Lei et al. (2018) .

Second, we propose an SVD-based task reweighting scheme to improve multi-task training for settings where different tasks have the same data but different labels.

On the ChestX-ray14 image classification dataset, we compare our method to the unweighted scheme and observe an improvement of 5.6 AUC score for all tasks.

In conclusion, these evaluations confirm that our theoretical insights are applicable to a broad range of settings and applications.

We study multi-task learning (MTL) models with a shared module for all tasks and a separate output module for each task.

We ask: What are the key components to determine whether or not MTL is better than single-task learning (STL)?

In response, we identify three components: model capacity, task covariance, and optimization scheme.

After setting up the model, we briefly describe the role of model capacity.

We then introduce the notion of task covariance, which comprises the bulk of the section.

We finish by showing the implications of our results for choosing optimization schemes.

We are given k tasks.

Let m i denote the number of data samples of task i. For task i, let X i ∈ R mi×d denote its covariates and let y i ∈ R mi denote its labels, where d is the dimension of the data.

We have assumed that all the tasks have the same input dimension d. This is not a restrictive assumption and is typically satisfied, e.g. for word embeddings on BERT, or by padding zeros to the input otherwise.

Our model assumes the output label is 1-dimensional.

We can also model a multi-label problem with k types of labels by having k tasks with the same covariates but different labels.

We consider an MTL model with a shared module B ∈ R d×r and a separate output module A i ∈ R r for task i, where r denotes the output dimension of B. See Figure 1 for the illustration.

We define the objective of finding an MTL model as minimizing the following equation over B and the A i 's:

where L is a loss function such as the squared loss.

The activation function g : R → R is applied on every entry of X i B. In equation 1, all data samples contribute equally.

Because of the differences between tasks such as data size, it is natural to re-weight tasks during training:

This setup is an abstraction of the hard parameter sharing architecture (Ruder (2017) ).

The shared module B provides a universal representation (e.g., an LSTM for encoding sentences) for all tasks.

Each task-specific module A i is optimized for its output.

We focus on two models as follows.

The single-task linear model.

The labels y of each task follow a linear model with parameter θ ∈ R d : y = Xθ + ε.

Every entry of ε follows the normal distribution N (0, σ 2 ) with variance σ 2 .

The function g(XB) = XB.

This is a well-studied setting for linear regression (Hastie et al. (2005) ).

The single-task ReLU model.

Denote by ReLU(x) = max(x, 0) for any x ∈ R. We will also consider a non-linear model where Xθ goes through the ReLU activation function with a ∈ R and θ ∈ R d : y = a · ReLU(Xθ) + ε, which applies the ReLU activation on Xθ entrywise.

The encoding function g(XB) then maps to ReLU(XB).

Positive vs. negative transfer.

For a source task and a target task, we say the source task transfers positively to the target task, if training both through equation 1 improves over just training the target task (measured on its validation set).

Negative transfer is the converse of positive transfer.

Our goal is to analyze the three components to determine positive vs. negative transfer between tasks: model capacity (r), task covariances (

)

and the per-task weights

).

We focus on regression tasks under the squared loss but we also provide synthetic experiments on classification tasks to validate our theory.

Notations.

For a matrix X, its column span is the set of all linear combinations of the column vectors of X. Let X † denote its pseudoinverse.

We begin by revisiting the role of model capacity, i.e. the output dimension of B (denoted by r).

We show that as a rule of thumb, r should be smaller than the sum of capacities of the STL modules.

Example.

Suppose we have k linear regression tasks using the squared loss, equation 1 becomes:

The optimal solution of equation 3 for task i is

.

Hence a capacity of 1 suffices for each task.

We show that if r ≥ k, then there is no transfer between any two tasks.

Proposition 1.

Let r ≥ k. There exists an optimum B and {A i } k i=1 of equation 3 where B A i = θ i , for all i = 1, 2, . . .

, k.

To illustrate the idea, as long as B contains {θ i } k i=1 in its column span, there exists A i such that B A i = θ i , which is optimal for equation 3 with minimum error.

But this means no transfer among any two tasks.

This can hurt generalization if a task has limited data, in which case its STL solution overfits training data, whereas the MTL solution can leverage other tasks' data to improve generalization.

The proof of Proposition 1 and its extension to ReLU settings are in Appendix B.1.

Figure 3: Performance improvement of a target task (Task 1) by MTL with a source task vs. STL.

Red: positive transfer when the source is Task 2, which has the same covariance matrix with target.

Green: negative (to positive) transfer when the source is Task 3, which has a different covariance from the target, as its # of samples increases.

See the example below for the definition of each task.

To show how to quantify task data similarity, we illustrate with two regression tasks under the linear model without noise: y 1 = X 1 θ 1 and y 2 = X 2 θ 2 .

By Section 2.2, it is necessary to limit the capacity of the shared module to enforce information transfer.

Therefore, we consider the case of r = 1.

Hence, the shared module B is now a d-dimensional vector, and A 1 , A 2 are both scalars.

A natural requirement of task similarity is for the STL models to be similar, i.e. |cos(θ 1 , θ 2 )| to be large.

To see this, the optimal STL model for task 1 is (X 1 X 1 ) −1 X 1 y 1 = θ 1 .

Hence if |cos(θ 1 , θ 2 )| is 1, then tasks 1 and 2 can share a model B ∈ R d which is either θ 1 or −θ 1 .

The scalar A 1 and A 2 can then transform B to be equal to θ 1 and θ 2 .

Is this requirement sufficient?

Recall that in equation 3, the task data X 1 and X 2 are both multiplied by B. If they are poorly "aligned" geometrically, the performance could suffer.

How do we formalize the geometry between task alignment?

In the following, we show that the covariance matrices of X 1 and X 2 , which we define to be X 1 X 1 and X 2 X 2 , captures the geometry.

We fix |cos(θ 1 , θ 2 )| to be close to 1 to examine the effects of task covariances.

In Appendix B.2.1 we fix task covariances to examine the effects of model cosine similarity.

Concretely, equation 3 reduces to:

where we apply the first-order optimality condition on A 1 and A 2 and simplify the equation.

Specifically, we focus on a scenario where task 1 is the source and task 2 is the target.

Our goal is to determine when the source transfers to the target positively or negatively in MTL.

Determining the type of transfer from task 2 to task 1 can be done similarly.

Answering the question boils down to studying the angle or cosine similarity between the optimum of equation 4 and θ 2 .

Example.

In Figure 3 , we show that by varying task covariances and the number of samples, we can observe both positive and negative transfers.

The conceptual message is the same as Figure 2 ; we describe the data generation process in more detail.

We use 3 tasks and measure the type of transfer from the source to the target.

The x-axis is the number of data samples from the source.

The y-axis is the target's performance improvement measured on its validation set between MTL minus STL.

Data generation.

We have |cos(θ 1 , θ 2 )| ≈ 1 (say 0.96).

For i ∈ {1, 2, 3}, let R i ⊆

R mi×d denote a random Gaussian matrix drawn from N (0, 1).

Let S 1 , S 2 ⊆ {1, 2, . . .

, d} be two disjoint sets of size d/10.

For i = 1, 2, let D i be a diagonal matrix whose entries are equal to a large value κ (e.g. κ = 100) for coordinates in S i and 1 otherwise.

Let Q i ⊆

R d×d denote an orthonormal matrix, i.e. Q i Q i is equal to the identity matrix, orthogonalized from a random Gaussian matrix.

Then, we define the 3 tasks as follows.

(i) Task 1 (target): X 1 = R 1 Q 1 D 1 and y 1 = X 1 θ 1 . (ii) Task 2 (source task for red line): X 2 = R 2 Q 1 D 1 and y 2 = X 2 θ 2 . (iii) Task 3 (source task for green line): X 3 = R 3 Q 2 D 2 and y 3 = X 3 θ 2 .

Task 1 and 2 have the same covariance matrices but task 1 and 3 have different covariance matrices.

Intuitively, the signals of task 1 and 3 lie in different subspaces, which arise from the difference in the diagonals of D i and the orthonormal matrices.

Analysis.

Unless the source task has lots of samples to estimate θ 2 , which is much more than the samples needed to estimate only the coordinates of S 1 , the effect of transferring to the target is small.

We observe similar results for logistic regression tasks and for ReLU-activated regression tasks.

Require: Task embedding layers X1 ∈ R m 1 ×d , X2 ∈ R m 2 ×d , . . .

, X k ∈ R m k ×d , shared module B Parameter: Alignment matrices R1, R2, . . .

, R k ∈ R d×d and output modules A1, A2 . . .

, A k ∈ R r 1: Let Zi = XiRi, for 1 ≤ i ≤ k. Consider the following modified loss (with B being fixed):

Minimizef by alternatively applying a gradient descent update on Ai and Ri, given a sampled data batch from task i. Other implementation details are described in Appendix C.3.

Theory.

We rigorously quantify how many data points is needed to guarantee positive transfer.

The folklore in MTL is that when a source task has a lot of data but the related target task has limited data, then the source can often transfer positively to the target task.

Our previous example shows that by varying the source's number of samples and its covariance, we can observe both types of transfer.

How much data do we need from the source to guarantee a positive transfer to the target?

We show that this depends on the condition numbers of both tasks' covariances.

Theorem 2 (informal).

For i = 1, 2, let y i = X i θ i + ε i denote two linear regression tasks with parameters θ i ∈ R d and m i number of samples.

Suppose that each row of the source task X 1 is drawn independently from a distribution with covariance Σ 1 ⊆

R d×d and bounded l 2 -norm.

Let c = κ(X 2 )sin(θ 1 , θ 2 ) and assume that c ≤ 1/3.

Denote by (B , A 1 , A 2 ) the optimal MTL solution.

With high probability, when m 1 is at least on the order of (κ

Recall that for a matrix X, κ(X) denotes its condition number.

Theorem 2 quantifies the trend in Figure 3 , where the improvements for task 2 reaches the plateau when m 1 becomes large enough.

The formal statement, its proof and discussions on the assumptions are deferred to Appendix B.2.2.

The ReLU model.

We show a similar result for the ReLU model, which requires resolving the challenge of analyzing the ReLU function.

We use a geometric characterization for the ReLU function under distributional input assumptions by Du et al. (2017) .

The result is deferred to Appendix B.2.3.

Algorithmic consequence.

An implication of our theory is a covariance alignment method to improve multi-task training.

For the i-th task, we add an alignment matrix R i before its input X i passes through the shared module B. Algorithm 1 shows the procedure.

We also propose a metric called covariance similarity score to measure the similarity between two tasks.

Given X 1 ∈ R m1×d and X 2 ∈ R m2×d , we measure their similarity in three steps: (a) The covariance matrix is X 1 X 1 .

(b) Find the best rank-r 1 approximation to be U 1,r1 D 1,r1 U 1,r1 , where r 1 is chosen to contain 99% of the singular values.

(c) Apply step (a),(b) to X 2 , compute the score:

The nice property of the score is that it is invariant to rotations of the columns of X 1 and X 2 .

2.4 OPTIMIZATION SCHEME Lastly, we consider the effect of re-weighting the tasks (or their losses in equation 2).

When does reweighting the tasks help?

In this part, we show a use case for improving the robustness of multi-task training in the presence of label noise.

The settings involving label noise can arise when some tasks only have weakly-supervised labels, which have been studied before in the literature (e.g. Mintz et al. (2009); Pentina and Lampert (2017) ).

We start by describing a motivating example.

Consider two tasks where task 1 is y 1 = Xθ and task 2 is y 2 = Xθ + ε 2 .

If we train the two tasks together, the error ε 2 will add noise to the trained model.

However, by up weighting task 1, we reduce the noise from task 2 and get better performance.

To rigorously study the effect of task weights, we consider a setting where all the tasks have the same data but different labels.

This setting arises for example in multi-label image tasks.

We derive the optimial solution in the linear model.

Proposition 3.

Let the shared module have capacity r ≤ k. Given k tasks with the same covariates

.

Let X be full rank and U DV be its SVD.

Let Q r Q r be the best rank-r approximation to

d×r be an optimal solution for the re-weighted loss.

Then the column span of B is equal to the column span of (X X) −1 V DQ r .

We can also extend Proposition 3 to show that all local minima of equation 3 are global minima in the linear setting.

We leave the proof to Appendix B.3.

We remark that this result does not extend to the non-linear ReLU setting and leave this for future work.

Based on Proposition 3, we provide a rigorous proof of the previous example.

Suppose that X is full rank, (X X)

Hence, when we increase α 1 , cos(B , θ) increases closer to 1.

Algorithm 2 An SVD-based task reweighting scheme Input: k tasks: (X, yi) ∈ (R m×d , R m ); a rank parameter r ∈ {1, 2, . . .

, k} Output: A weight vector: {α1, α2, . . .

, α k } 1: Let θi = X yi.

2: Ur, Dr, Vr = SVDr(θ1, θ2, . . .

, θ k ), i.e. the best rank-r approximation to the θi's.

Algorithmic consequence.

Inspired by our theory, we describe a re-weighting scheme in the presence of label noise.

We compute the per-task weights by computing the SVD over X y i , for 1 ≤ i ≤ k.

The intuition is that if the label vector of a task y i is noisy, then the entropy of y i is small.

Therefore, we would like to design a procedure that removes the noise.

The SVD procedure does this, where the weight of a task is calculated by its projection into the principal r directions.

See Algorithm 2 for the description.

We describe connections between our theoretical results and practical problems of interest.

We show three claims on real world datasets.

(i) The shared MTL module is best performing when its capacity is smaller than the total capacities of the single-task models. (ii) Our proposed covariance alignment method improves multi-task training on a variety of settings including the GLUE benchmarks and six sentiment analysis tasks.

Our method can be naturally extended to transfer learning settings and we validate this as well. (iii) Our SVD-based reweighed scheme is more robust than the standard unweighted scheme on multi-label image classification tasks in the presence of label noise.

Datasets and models.

We describe the datasets and models we use in the experiments.

GLUE: GLUE is a natural language understanding dataset including question answering, sentiment analysis, text similarity and textual entailment problems.

We choose BERT LARGE as our model, which is a 24 layer transformer network from Devlin et al. (2018) .

Sentiment Analysis: This dataset includes six tasks: movie review sentiment (MR), sentence subjectivity (SUBJ), customer reviews polarity (CR), question type (TREC), opinion polarity (MPQA), and the Stanford sentiment treebank (SST) tasks.

For each task, the goal is to categorize sentiment opinions expressed in the text.

We use an embedding layer followed by an LSTM layer proposed by Lei et al. (2018) .

We use the GloVe embeddings (http://nlp.stanford.edu/data/wordvecs/glove.6B.zip).

ChestX-ray14: This dataset contains 112,120 frontal-view X-ray images and each image has up to 14 diseases.

This is a 14-task multi-label image classification problem.

We use the CheXNet model from Rajpurkar et al. (2017) , which is a 121-layer convolutional neural network on all tasks.

For all models, we share the main module across all tasks (BERT LARGE for GLUE, LSTM for sentiment analysis, CheXNet for ChestX-ray14) and assign a separate regression or classification layer on top of the shared module for each tasks.

Comparison methods.

For the experiment on multi-task training, we compare Algorithm 1 by training with our method and training without it.

Specifically, we apply the alignment procedure on the task embedding layers.

See Figure 4 for an illustration, where E i denotes the embedding of task i, R i denotes its alignment module and Z i = E i R i is the rotated embedding.

For transfer learning, we first train an STL model on the source task by tuning its model capacity (e.g. the output dimension of the LSTM layer).

Then, we fine-tune the STL model on the target task for 5-10 epochs.

To apply Algorithm 1, we add an alignment module for the target task during fine-tuning.

Figure 4: Illustration of the covariance alignment module on task embeddings.

For the experiment on reweighted schemes, we first compute the per-task weights as described in Algorithm 2.

Then, we reweight the loss function as in equation 2.

We compare with the reweighting techniques of Kendall et al. (2018) .

Informally, the latter uses the Gaussian likelihood to model classification outputs.

The weights, defined as inversely proportional to the variances of the Gaussian, are optimized during training.

We also compare with the unweighted loss (cf.

equation 1) as a baseline.

Metric.

We measure performance on the GLUE benchmark using a standard metric called the GLUE score, which contains accuracy and correlation scores for each task.

For the sentiment analysis tasks, we measure the accuracy of predicting the sentiment opinion.

For the image classification task, we measure the area under the curve (AUC) score.

We run five different random seeds to report the average results.

The result of an MTL experiment is averaged over the results of all the tasks, unless specified otherwise.

For the training procedures and other details on the setup, we refer the reader to Appendix C.

We present use cases of our methods on open-source datasets.

We expected to see improvements via our methods in multi-task and other settings, and indeed we saw such gains across a variety of tasks.

Improving multi-task training.

We apply Algorithm 1 on five tasks (CoLA, MRPC, QNLI, RTE, SST-2) from the GLUE benchmark using a state-of-the-art language model BERT LARGE .

We compare the average performance over all five tasks and find that our method outperforms BERT LARGE by 2.35% average GLUE score for the five tasks.

For the particular setting of training two tasks, our method outperforms BERT LARGE on 7 of the 10 task pairs.

See Figure 5a for the results.

Improving transfer learning.

While our study has focused on multi-task learning, transfer learning is a naturally related goal -and we find that our method is also useful in this case.

We validate this by training an LSTM on sentiment analysis.

Figure 5b shows the result with SST being the source task and the rest being the target task.

Algorithm 1 improves accuracy on four tasks by up to 2.5%.

Reweighting training for the same task covariates.

We evaluate Algorithm 2 on the ChestX-ray14 dataset.

This setting satisfies the assumption of Algorithm 2, which requires different tasks to have the same input data.

Across all 14 tasks, we find that our reweighting method improves the technique of Kendall et al. (2018) by 1.3% AUC score.

Compared to training with the unweighted loss, our method improves performance by 5.6% AUC score over all tasks.

Model capacity.

We verify our hypothesis that the capacity of the MTL model should not exceed the total capacities of the STL model.

We show this on an LSTM model with the sentiment analysis tasks.

Recall that the capacity of an LSTM model is its output dimension (before the last classification layer).

First, we train an MTL model with all tasks and vary the shared module's capacity to find the optimal setting (from 5 to 500).

Then, we train an STL model for each task and find the optimal setting similarly.

In Figure 6 , we find that the performance of MTL peaks when the shared module has capacity 100.

This is much smaller than the total capacities of all the STL models.

The result confirms that constraining the shared module's capacity is crucial to achieve the ideal performance.

Extended results on CNN/MLP to support our hypothesis are shown in Appendix C.5.

Task covariance.

We apply our metric of task covariance similarity score from Section 2.3 to provide an in-depth study of the covariance alignment method.

The hypothesis is that: (a) aligning the covariances helps, which we have shown in Figure 5a ; (b) the similarity score between two tasks increases after applying the alignment.

We verify the hypothesis on the sentiment analysis tasks.

We use the single-task model's embedding before the LSTM layer to compute the covariance.

First, we measure the similarity score using equation 6 between all six single-task models.

Then, for each task pair, we train an MTL model using Algorithm 1.

We measure the similarity score on the trained MTL model.

Our results confirm the hypothesis ( Figure 7 ): (a) we observe increased accuracy on 13 of 15 task pairs by up to 4.1%; (b) the similarity score increases for all 15 task pairs.

Optimization scheme.

We verify the robustness of Algorithm 2.

After selecting two tasks from the ChestX-ray14 dataset, we test our method by assigning random labels to 20% of the data on one task.

On 20 randomly selected pairs, our method improves over the unweighted scheme by an average 2.4% AUC score and the techniques of Kendall et al. (2018) by an average 0.5% AUC score.

There has been a large body of recent work on using the multi-task learning approach to train deep neural networks.

Of particular relevance to this work are those that study the theory of multi-task learning.

The earlier works of Baxter (2000); Ben-David and Schuller (2003) are among the first to formally study the importance of task relatedness for learning multiple tasks.

See also the follow-up work of Maurer (2006)

In this work, we studied the theory of multi-task learning in linear and ReLU-activated settings.

We verified our theory and its practical implications through extensive synthetic and real world experiments.

Our work opens up many interesting future questions.

First, could we extend the guarantees for choosing optimization schemes to non-linear settings?

Second, a limitation of our SVD-based optimization scheduler is that it only applies to settings with the same data.

Could we extend the method for heterogeneous task data?

More broadly, we hope our work inspires further studies to better understand multi-task learning in neural networks and to guide its practice.

Hard parameter sharing vs soft parameter sharing.

The architecture that we study in this work is also known as the hard parameter sharing architecture.

There is another kind of architecture called soft parameter sharing.

The idea is that each task has its own parameters and modules.

The relationships between these parameters are regularized in order to encourage the parameters to be similar.

Other architectures that have been studied before include the work of Misra et al. (2016), where the authors explore trainable architectures for convolutional neural networks.

Domain adaptation.

Another closely related line of work is on domain adaptation.

The acute reader may notice the similarity between our study in Section 2.3 and domain adaptation.

The crucial difference here is that we are minimizing the multi-task learning objective, whereas in domain adaptation the objective is typically to minimize the objective on the target task.

See Ben

We fill in the missing details left from Section 2.

In Section B.1, we provide rigorous arguments regarding the capacity of the shared module.

In Section B.2, we fill in the details left from Section 2.3, including the proof of Theorem 2 and its extension to the ReLU model.

In Section B.3, we provide the proof of Proposition 3 on the task reweighting schemes.

We first describe the notations.

Notations.

We define the notations to be used later on.

We denote f (x) g(x) if there exists an absolute constant

Suppose A ∈ R m×n , then λ max (A) denotes its largest singular value and λ min (A) denotes its min{m, n}-th largest singular value.

Alternatively, we have λ min (A) = min x: x =1 Ax .

Let κ(A) = λ max (A)/λ min (A) denote the condition number of A. Let Id denotes the identity matrix.

Let U † denote the Moore-Penrose pseudo-inverse of the matrix U .

Let · denote the Euclidean norm for vectors and spectral norm for matrices.

Let · F denote the Frobenius norm of a matrix.

Let A, B, = Tr(A B) denote the inner product of two matrices.

The sine function is define as sin(u, v) = 1 − cos(u, v) 2 , where we assume that sin(u, v) ≥ 0 which is without loss of generality for our study.

We describe the full detail to show that our model setup captures the phenomenon that the shared module should be smaller than the sum of capacities of the single-task models.

We state the following proposition which shows that the quality of the subspace B in equation 1 determines the performance of multi-task learning.

This supplements the result of Proposition 1.

Proposition 4.

In the optimum of f (·) (equation 1), each A i selects the vector v within the column span of g B (X i ) to minimize L(v, y i ).

As a corollary, in the linear setting, the optimal B can be achieved at a rotation matrix B ⊆ R d×r by maximizing

Furthermore, any B which contains {θ i } k i=1 in its column subspace is optimal.

In particular, for such a B , there exists {A i } so that B A i = θ i for all 1 ≤ i ≤ k.

Proof.

Recall the MTL objective in the linear setting from equation 3 as follows:

Note that the linear layer A i can pick any combination within the subspace of B. Therefore, we could assume without loss of generality that B is a rotation matrix.

i.e. B B = Id. After fixing B, since objective f (·) is linear in A i for all i, by the local optimality condition, we obtain that

Replacing the solution of A i to f (·), we obtain an objective over B.

Next, note that

where we used the fact that The above result on linear regression suggests the intuition that optimizing an MTL model reduces to optimizing over the span of B. The intuition can be easily extended to linear classification tasks as well as mixtures of regression and classification tasks.

Extension to the ReLU setting.

If the shared module's capacity is larger than the total capacities of the STL models, then we can put all the STL model parameters into the shared module.

As in the linear setting, the final output layer A i can pick out the optimal parameter for the i-th task.

This remains an optimal solution to the MTL problem in the ReLU setting.

Furthermore, there is no transfer between any two tasks through the shared module.

We consider the effect of varying the cosine similarity between single task models in multi-task learning.

We first describe the following proposition to solve the multi-task learning objective when the covariances of the task data are the same.

The idea is similar to the work of Ando and Zhang (2005) and we adapt it here for our study.

where C C is the best rank-r approximation subspace of

As a corollary, denote by λ 1 , λ 2 , . . .

, λ k as the singular values of Proof.

Note that B is obtained by maximizing

Clearly, there is a one to one mapping between B and C. And we have B = V D −1 C. Hence the above is equivalent to maximizing over C ⊆ R d×r with

Note that C(C C) −1 C is a projection matrix onto a subspace of dimension r. Hence the maximum (denote by C ) is attained at the best rank-r approximation subspace of

To illustrate the above proposition, consider a simple setting where X i is identity for every 1 ≤ i ≤ k, and y i = e i , i.e. the i-th basis vector.

Note that the optimal solution for the i-th task is (X i X i )

−1 X i y i = y i .

Hence the optimal solutions are orthogonal to each other for all the tasks, with λ i = 1 for all 1 ≤ i ≤ k. And the minimum STL error is zero for all tasks.

Consider the MTL model with hidden dimension r. By Proposition 5, the minimum MTL error is achieved by the best rank-r approximation subspace to

Denote the optimum as B r .

The MTL error is:

Different data covariance.

We provide upper bounds on the quality of MTL solutions for different data covariance, which depend on the relatedness of all the tasks.

The following procedure gives the precise statement.

Consider k regression tasks with data {(

† X i y i denote the optimal solution of each regression task.

Let W ⊆ R d×k denote the matrix where the i-th column is equal to θ i .

Consider the following procedure for orthogonalizing W for 1 ≤ i ≤ k.

Step a).

Proposition 6.

Suppose that r ≤ d. Let B denote the optimal MTL solution of capacity r in the shared module.

Denote by

Proof.

It suffices to show that OP T is equal to k i=1 λ i .

The result then follows since h(B ) is less than the error given by W 1 , . . .

, W k , which is equal to OP T − d i=r+1 λ i .

We fill in the proof of Theorem 2.

First, we restate the result rigorously as follows.

Theorem 2.

For i = 1, 2, let (X i , y i ) ∈ (R mi×d , R mi ) denote two linear regression tasks with parameters θ i ∈ R d .

Suppose that each row of X 1 is drawn independently from a distribution with covariance Σ 1 ⊆

R d×d and bounded l 2 -norm √ L. Assume that θ 1 Σ 1 θ 1 = 1 w.l.o.g.

Let c ∈ [κ(X 2 ) sin(θ 1 , θ 2 ), 1/3] denote the desired error margin.

Denote by (B , A 1 , A 2 ) the optimal MTL solution.

With probability 1 − δ over the randomness of (X 1 , y 1 ), when

we have that B A 2 − θ 2 / θ 2 ≤ 6c + 1 1−3c ε 2 / X 2 θ 2 .

We make several remarks to provide more insight on Theorem 2.

• Theorem 2 guarantees positive transfers in MTL, when the source and target models are close and the number of source samples is large.

While the intuition is folklore in MTL, we provide a formal justification in the linear and ReLU models to quantify the phenomenon.

• The error bound decreases with c, hence the smaller c is the better.

On the other hand, the required number of data points m 1 increases.

Hence there is a trade-off between accuracy and the amount of data.

• c is assumed to be at most 1/3.

This assumption arises when we deal with the label noise of task 2.

If there is no noise for task 2, then this assumption is not needed.

If there is noise for task 2, this assumption is satisfied when sin(θ 1 , θ 2 ) is less than 1/(3κ(X 2 )).

In synthetic experiments, we observe that the dependence on κ(X 2 ) and sin(θ 1 , θ 2 ) both arise in the performance of task 2, cf.

Figure 3 and Figure 8 , respectively.

The proof of Theorem 2 consists of two steps.

a) We show that the angle between B and θ 1 will be small.

Once this is established, we get a bound on the angle between B and θ 2 via the triangle inequality.

b) We bound the distance between B A 2 and θ 2 .

The distance consists of two parts.

One part comes from B , i.e. the angle between B and θ 2 .

The second part comes from A 2 , i.e. the estimation error of the norm of θ 2 , which involves the signal to noise ratio of task two.

We first show the following geometric fact, which will be used later in the proof.

Fact 7.

Let a, b ∈ R d denote two unit vectors.

Suppose that X ∈ R m×d has full column rank with condition number denoted by κ = κ(X).

Then we have

Proof.

Let X = U DV be the SVD of X. Since X has full column rank by assumption, we have X X = XX = Id. Clearly, we have sin(Xa, Xb) = sin(DV a, DV b).

Denote by a = V a and b = V b.

We also have that a and b are both unit vectors, and sin(a , b ) = sin(a, b).

Let λ 1 , . . .

, λ d denote the singular values of X. Then,

This concludes the proof.

We first show the following Lemma, which bounds the angle between B and θ 2 .

Lemma 8.

In the setting of Theorem 2, with probability 1 − δ over the randomness of task one, we have that |sin(B , θ 2 )| ≤ sin(θ 1 , θ 2 ) + c/κ(X 2 ).

Proof.

We note that h(B ) ≥ y 1 2 by the optimality of B .

Furthermore, X2B X2B , y 2 ≤ y 2 2 .

Hence we obtain that

For the left hand side,

Note that the second term is a chi-squared random variable with expectation σ 2 1 .

Hence it is bounded by σ 2 1 log 1 δ with probability at least 1 − δ.

Similarly, the third term is bounded by 2 X 1 θ 1 σ 1 log 1 δ with probability 1 − δ.

Therefore, we obtain the following:

Therefore,

By matrix Bernstein inequality (see e.g. Tropp et al. (2015)), when m 1 ≥ 10 Σ 1 log d δ /λ 2 min (Σ 1 ), we have that:

Hence we obtain that κ 2 (X 1 ) ≤ 3κ(Σ 1 ) and X 1 θ 1 2 ≥ m 1 · θ 1 Σ 1 θ 1 /2 ≥ m 1 /2 (where we assumed that θ 1 Σ 1 θ 1 = 1).

Therefore,

which is at most c 2 /κ 2 (X 2 ) by our setting of m 1 .

Therefore, the conclusion follows by triangle inequality (noting that both c and sin(θ 1 , θ 2 ) are less than 1/2).

Based on the above Lemma, we are now to ready to prove Theorem 2.

Proof of Theorem 2.

Note that in the MTL model, after obtaining B , we then solve the linear layer for each task.

For task 2, this gives weight value A 2 := X 2θ , y 2 / X 2θ 2 .

Thus the regression coefficients for task 2 is B A 2 .

For the rest of the proof, we focus on bounding the distance between B A 2 and θ 2 .

By triangle inequality,

Note that the second term of equation 8 is equal to

The first term of equation 8 is bounded by

.

Lastly, we have that

By Lemma 8, we have

Therefore, we conclude that equation 9 is at most

Thus equation 8 is at most the following.

Hence we obtain the desired estimation error of BA 2 .

In this part, we extend Theorem 2 to the ReLU model.

Note that the problem is reduced to the following objective.

We make a crucial assumption that task 1's input X 1 follows the Gaussian distribution.

Note that making distributional assumptions is necessary because for worst-case inputs, even optimizing a single ReLU function under the squared loss is NP-hard (Manurangsi and Reichman (2018)).

We state our result formally as follows.

Theorem 9.

Let (X 1 , y 1 ) ∈ (R m1×d , R m1 ) and (X 2 , y 2 ) ∈ (R m2×d , R m2 ) denote two tasks.

Suppose that each row of X 1 is drawn from the standard Gaussian distribution.

And y i = a i · ReLU(X i θ i ) + ε i are generated via the ReLU model with

2 j = 1 for every 1 ≤ j ≤ m 1 without loss of generality, and let σ 2 1 denote the variance of every entry of ε 1 .

Suppose that c ≥ sin(θ 1 , θ 2 )/κ(X 2 ).

Denote by (B , A 1 , A 2 ) the optimal MTL solution of equation 10.

With probability 1 − δ over the randomness of (X 1 , y 1 ), when

we have that the estimation error is at most:

Proof.

The proof follows a similar structure to that of Theorem 2.

Without loss of generality, we can assume that θ 1 , θ 2 are both unit vectors.

We first bound the angle between B and θ 1 .

By the optimality of B , we have that:

From this we obtain:

Note that each entry of ReLU(X 1 θ 1 ) is a truncated Gaussian random variable.

By the Hoeffding bound, with probability 1 − δ we have

As for ReLU(X 1 B ), ReLU(X 1 θ 1 ) , we will use an epsilon-net argument over B to show the concentration.

For a fixed B , we note that this is a sum of independent random variables that are all bounded within O(log m1 δ ) with probability 1 − δ.

Denote by φ the angle between B and θ 1 , a standard geometric fact states that (see e.g. Lemma 1 of Du et al. (2017) ) for a random Gaussian vector

Therefore, by applying Bernstein's inequality and union bound, with probability 1 − η we have:

By standard arguments, there exists a set of d O(d) unit vectors S such that for any other unit vector

and take union bound over all unit vectors in S, we have that there existsû ∈ S satisfying B −û ≤ min(1/d 3 , c 2 /κ 2 (X 2 )) and the following:

where φ is the angle betweenû and θ 1 .

Note that

Together we have shown that

Combined with equation 11, by our setting of m 1 , it is not hard to show that

Overall, we conclude that

For the estimation of a 2 , we have

Similarly, we can show that the second part is at most O(c).

Therefore, the proof is complete.

In this part, we present the proof of Proposition 3.

In fact, we present a more refined result, by showing that all local minima are global minima for the reweighted loss in the linear case.

The key is to reduce the MTL objective f (·) to low rank matrix approximation, and apply recent results by Balcan et al. (2018) which show that there is no spurious local minima for the latter problem .

Lemma 10.

Assume that X i X i = α i Σ with α i > 0 for all 1 ≤ i ≤ k.

Then all the local minima of f (A 1 , . . .

, A k ; B) are global minima of equation 3.

Proof.

We first transform the problem from the space of B to the space of C. Note that this is without loss of generality, since there is a one to one mapping between B and C with C = DV B.

In this case, the corresponding objective becomes the following.

The latter expression is a constant.

Hence it does not affect the optimization solution.

For the former, denote by A ∈ R r×k as stacking the √ α i A i 's together column-wise.

Similarly, denote by Z ∈ R d×k as stacking √ α i U i y i together column-wise.

Then minimizing g(·) reduces solving low rank matrix approximation: CA − Z times the best rank-r approximation to α i U y i y i U , where we denote the SVD of X as U DV .

Denote by Q r Q r as the best rank-r approximation to U ZZ U , where we denote by Z = [ √ α 1 y 1 , √ α 2 y 2 , . . . , √ α k y k ] as stacking the k vectors to a d by k matrix.

Hence the result of Proposition 5 shows that the optimal solution B is V D −1 Q r , which is equal to (X X) −1 XQ r .

By Proposition 4, the optimality of B is the same up to transformations on the column space.

Hence the proof is complete.

To show that all local minima are also equal to (X X) −1 XQ r , we can simply apply Lemma 10 and Proposition 3.

Remark.

This result only applies to the linear model and does not work on ReLU models.

The question of characterizing the optimization landscape in non-linear ReLU models is not well-understood based on the current theoretical understanding of neural networks.

We leave this for future work.

We fill in the details left from our experimental section.

In Appendix C.1, we review the datasets used in our experiments.

In Appendix C.2, we describe the models we use on each dataset.

In Appendix C.3, we describe the training procedures for all experiments.

In Appendix C.4 and Appendix C.5, we show extended synthetic and real world experiments to support our claims.

We describe the synthetic settings and the datasets Sentiment Analysis, General Language Understanding Evaluation (GLUE) benchmark, and ChestX-ray14 used in the experiments.

Synthetic settings.

For the synthetic experiments, we draw 10,000 random data samples with dimension d = 100 from the standard Gaussian N (0, 1) and calculate the corresponding labels based on the model described in experiment.

We split the data samples into training and validation sets with 9,000 and 1,000 samples in each.

For classification tasks, we generate the labels by applying a sigmoid function and then thresholding the value to binary labels at 0.5.

For ReLU regression tasks, we apply the ReLU activation function on the real-valued labels.

The number of data samples used in the experiments varies depending on the specification.

Specifically, for the task covariance experiment of Figure 3 , we fix task 1's data with m 1 = 9, 000 training data and vary task 2's data under three settings: (i) same rotation

Sentiment analysis.

For the sentiment analysis task, the goal is to understand the sentiment opinions expressed in the text based on the context provided.

This is a popular text classification task which is usually formulated as a multi-label classification task over different ratings such as positive (+1), negative (-1), or neutral (0).

We use six sentiment analysis benchmarks in our experiments:

• Movie review sentiment (MR): In the MR dataset (Pang and Lee (2005)), each movie review consists of a single sentence.

The goal is to detect positive vs. negative reviews.

• Sentence subjectivity (SUBJ): The SUBJ dataset is proposed in Pang and Lee (2004) and the goal is to classify whether a given sentence is subjective or objective.

• Customer reviews polarity (CR): The CR dataset (Hu and Liu (2004)) provides customer reviews of various products.

The goal is to categorize positive and negative reviews.

• Question type (TREC): The TREC dataset is collected by Li and Roth (2002) .

The aim is to classify a question into 6 question types.

• Opinion polarity (MPQA):

The MPQA dataset detects whether an opinion is polarized or not (Wiebe et al. (2005) ).

• Stanford sentiment treebank (SST): The SST dataset, created by Socher et al. (2013) , is an extension of the MR dataset.

The General Language Understanding Evaluation (GLUE) benchmark.

GLUE is a collection of NLP tasks including question answering, sentiment analysis, text similarity and textual entailment problems.

The GLUE benchmark is a state-of-the-art MTL benchmark for both academia and industry.

We select five representative tasks including CoLA, MRPC, QNLI, RTE, and SST-2 to validate our proposed method.

We emphasize that the goal of this work is not to come up with a state-of-the-art result but rather to provide insights into the working of multi-task learning.

It is conceivable that our results can be extended to the entire dataset as well.

This is left for future work.

More details about the GLUE benchmark can be found in the original paper (Wang et al. (2018a) ).

ChestX-ray14.

The ChestX-ray14 dataset (Wang et al. (2017) ) is the largest publicly available chest X-ray dataset.

It contains 112,120 frontal-view X-ray images of 30,805 unique patients.

Each image contains up to 14 different thoracic pathology labels using automatic extraction methods on radiology reports.

This can be formulated as a 14-task multi-label image classification problem.

The ChestX-ray14 dataset is a representative dataset in the medical imaging domain as well as in computer vision.

We use this dataset to examine our proposed task reweighting scheme since it satisfies the assumption that all tasks have the same input data but different labels.

Synthetic settings.

For the synthetic experiments, we use the linear regression model, the logistic regression model and a one-layer neural network with the ReLU activation function.

Sentiment analysis.

For the sentiment analysis experiments, we consider three different models including multi-layer perceptron (MLP), LSTM, CNN:

• For the MLP model, we average the word embeddings of a sentence and feed the result into a two layer perceptron, followed by a classification layer.

• For the LSTM model, we use the standard one-layer single direction LSTM as proposed by Lei et al. (2018) , followed by a classification layer.

• For the CNN model, we use the model proposed by Kim (2014) which uses one convolutional layer with multiple filters, followed by a ReLU layer, max-pooling layer, and classification layer.

We follow the protocol of Kim (2014) and set the filter size as {3, 4, 5}. We use the pre-trained GLoVe embeddings trained on Wikipedia 2014 and Gigaword 5 corpora 2 .

We fine-tune the entire model in our experiments.

In the multi-task learning setting, the shared modules include the embedding layer and the feature extraction layer (i.e. the MLP, LSTM, or CNN model).

Each task has its separate output module.

For the experiments on the GLUE benchmark, we use a state-of-the-art language model called BERT (Devlin et al. (2018) ).

For each task, we add a classification/regression layer on top it as our model.

For all the experiments, we use the BERT LARGE uncased model, which is a 24 layer network as described in Devlin et al. (2018) .

For the multi-task learning setting, we follow the work of Liu et al. (2019a) and use BERT LARGE as the shared module.

ChestX-ray14.

For the experiments on the ChestX-ray14 dataset, we use the DenseNet model proposed by Rajpurkar et al. (2017) as the shared module, which is a 121 layer network.

For each task, we use a separate classification output layer.

We use the pre-trained model 3 in our experiments.

In this subsection, we describe the training procedures for our experiments.

Mini-batch SGD.

We describe the details of task data sampling in our SGD implementation.

• For tasks with different features such as GLUE, we first divide each task data into small batches.

Then, we mix all the batches from all tasks and shuffle randomly.

During every epoch, a SGD step is applied on every batch over the corresponding task.

If the current batch is for task i, then the SGD is applied on A i , and possibly R i or B depending on the setup.

The other parameters for other tasks are fixed.

• For tasks with the same features such as ChestX-ray14, the SGD is applied on all the tasks jointly to update all the A i 's and B together.

For classification tasks, we use accuracy as the metric.

We report the average model performance over two tasks.

The x-axis denotes the cosine distance, i.e. 1 − cos(θ 1 , θ 2 ).

Synthetic settings.

For the synthetic experiments, we do a grid search over the learning rate from {1e − 4, 1e − 3, 1e − 2, 1e − 1} and the number of epochs from {10, 20, 30, 40, 50}. We pick the best results for all the experiments.

We choose the learning rate to be 1e − 3, the number of epochs to be 30, and the batch size to be 50.

For regression task, we report the Spearman's correlation score For classification task, we report the classification accuracy.

Sentiment analysis.

For the sentiment analysis experiments, we randomly split the data into training, dev and test sets with percentages 80%, 10%, and 10% respectively.

We follow the protocol of Lei et al. (2018) to set up our model for the sentiment analysis experiments.

The default hidden dimension of the model (e.g. LSTM) is set to be 200, but we vary this parameter for the model capacity experiments.

We report the accuracy score on the test set as the performance metric.

GLUE.

For the GLUE experiments, the training procedure is used on the alignment modules and the output modules.

Due to the complexity of the BERT LARGE module, which involves 24 layers of non-linear transformations, we fix the BERT LARGE module during the training process to examine the effect of adding the alignment modules to the training process.

In general, even after fine-tuning the BERT LARGE module on a set of tasks, it is always possible to add our alignment modules and apply Algorithm 1.

For the training parameters, we apply grid search to tune the learning rate from {2e−5, 3e−5, 1e−5} and the number of epochs from {2, 3, 5, 10}. We choose the learning rate to be 2e−5, the number of epochs to be 5, and with batch size 16 for all the experiments.

We use the GLUE evaluation metric (cf.

Wang et al. (2018b) ) and report the scores on the development set as the performance metric.

ChestX-ray14.

For the ChestX-ray14 experiments, we use the configuration suggested by Rajpurkar et al. (2017) and report the AUC score on the test set after fine-tuning the model for 20 epochs.

Varying cosine similarity on linear and ReLU models.

We demonstrate the effect of cosine similarity in synthetic settings for both regression and classification tasks.

Synthetic tasks.

We start with linear settings.

We generate 20 synthetic task datasets (either for regression tasks, or classification tasks) based on data generation procedure and vary the task similarity between task 1 and task i.

We run the experiment with a different dataset pairs (dataset 1 and dataset i).

We compare the performance gap between MTL and STL model.

Figure 8a and Figure 8a , we find that for both regression and classification settings, with the larger task similarity the MTL outperforms more than STL model and the negative transfer could occur if the task similarity is too small.

(b) Classification tasks with non-linearity Figure 9 : The performance improvement on the target task (MTL minus STL) by varying the cosine similarity of the two tasks' STL models.

We observe that higher similarity between the STL models leads to better improvement on the target task.

ReLU settings.

We also consider a ReLU-activated model.

We use the same setup as the linear setting, but apply a ReLU activation to generate the data.

Similar results are shown in Figure 8c , 8d.

Higher rank regimes for ReLU settings.

We provide further validation of our results on ReLUactivated models.

Synthetic tasks.

In this synthetic experiment, there are two sets of model parameters Θ 1 ⊆ R d×r and Θ 2 ⊆

R d×r (d = 100 and r = 10).

Θ 1 is a fixed random rotation matrix and there are m 1 = 100 data points for task 1.

Task 2's model parameter is Θ 2 = αΘ 1 + (1 − α)Θ , where Θ is also a fixed rotation matrix that is orthogonal to Θ 1 .

Note that α is the cosine value/similarity of the principal angle between Θ 1 and Θ 2 .

We then generate X 1 ⊆ R m1×d and X 2 ⊆ R m2×d from Gaussian.

For each task, the labels are y i = ReLU(X i Θ i )e + ε i , where e ∈ R r is the all ones vector and ε i is a random Gaussian noise.

Given the two tasks, we use MTL with ReLU activations and capacity H = 10 to co-train the two tasks.

The goal is to see how different levels of α or similarity affects the transfer from task two to task one.

Note that this setting parallels the ReLU setting of Theorem 9 but applies to rank r = 5.

Results.

In Figure 9 we show that the data size, the cosine similarity between the STL solutions and the alignment of covariances continue to affect the rate of transfer in the new settings.

The study shows that our conceptual results are applicable to a wide range of settings.

Evaluating Algorithm 1 on linear and ReLU-activated models.

We consider the synthetic example in Section 2.3 to compare Algorithm 1 and the baseline MTL training.

Recall that in the example, when the source and target tasks have different covariance matrices, MTL causes negative transfer on the target task.

Our hypothesis in this experiment is to show that Algorithm 1 can correct the misalignment and the negative transfer.

Synthetic tasks.

We evaluate on both linear and ReLU regression tasks.

The linear case follows the example in Section 2.3.

For the ReLU case, the data is generated according to the previous example.

Results.

Figure 10 confirms the hypothesis.

We observe that Algorithm 1 corrects the negative transfer in the regime where the source task only has limited amount of data.

Furthermore, Algorithm 1 matches the baseline MTL training when the source task has sufficiently many data points.

Cross validation for choosing model capacities.

We provide an cross validation experiment to indicate how we choose the best performing model capacities in Figure 6 .

This is done on the six sentiment analysis tasks trained with an LSTM layer.

In Figure 11 , we vary the model capacities to plot the validation accuracies of the MTL model trained with all six tasks and the STL model for each task.

The result complements Figure 6 in Section 3.3.

Choosing model capacities for CNN and MLP.

Next we verify our result on model capacities for CNN and MLP models.

We select the SST and MR datasets from the sentiment analysis tasks for this experiment.

We train all three models CNN, MLP and LSTM by varying the capacities.

Results.

From Figure 12 we observe that the best performing MTL model capacity is less than total best performing model capacities of STL model on all models.

The effect of label noise on Algorithm 2.

To evaluate the robustness of Algorithm 2 in the presence of label noise, we conduct the following experiment.

First, we select two tasks from the ChestXray14 dataset.

Then, we randomly pick one task to add 20% of noise to its labels by randomly flipping them with probability 0.2.

We compare the performance of training both tasks using our reweighting scheme (Algorithm 2) vs. the reweighting techniques of Kendall et al. (2018) and the unweighted loss scheme.

Results.

On 20 randomly chosen task pairs, our method improves over the unweighted training scheme by 2.4% AUC score and 0.5% AUC score over Kendall et al. (2018) averaged over the 20 task pairs.

Figure 13 shows 5 example task pairs from our evaluation.

@highlight

A Theoretical Study of Multi-Task Learning with Practical Implications for Improving Multi-Task Training and Transfer Learning