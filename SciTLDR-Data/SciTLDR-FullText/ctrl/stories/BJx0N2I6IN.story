In this paper, we present a reproduction of the paper of Bertinetto et al. [2019] "Meta-learning with differentiable closed-form solvers" as part of the ICLR 2019 Reproducibility Challenge.

In successfully reproducing the most crucial part of the paper, we reach a performance that is comparable with or superior to the original paper on two benchmarks for several settings.

We evaluate new baseline results, using a new dataset presented in the paper.

Yet, we also provide multiple remarks and recommendations about reproducibility and comparability.

After we brought our reproducibility work to the authors’ attention, they have updated the original paper on which this work is based and released code as well.

Our contributions mainly consist in reproducing the most important results of their original paper, in giving insight in the reproducibility and in providing a first open-source implementation.

The ability to adapt to new situations and learn quickly is a cornerstone of human intelligence.

When given a previously unseen task, humans can use their previous experience and learning abilities to perform well on this new task in a matter of seconds and with a relatively small amount of new data.

Artificial learning methods have been shown to be very effective for specific tasks, often times surpassing human performance BID11 , BID2 ).

However, by relying on standard supervised-learning or reinforcement learning training paradigms, these artificial methods still require much training data and training time to adapt to a new task.

An area of machine learning that learns and adapts from a small amount of data is called few-shot learning.

A shot corresponds to a single example, e.g. an image and its label.

In few-shot learning the learning scope is expanded to a variety of tasks with a few shots each, compared to the classic setting of a single task with many shots.

A promising approach for few-shot learning is the field of meta-learning.

Meta-learning, also known as learning-to-learn, is a paradigm that exploits cross-task information and training experience to perform well on a new unseen task.

In this work we reproduce the paper of BID1 (referenced as "their paper"); it falls into the class of gradient-based meta-learning algorithms that learn a model parameter intialization for rapid fine-tuning with a few shots BID4 , BID9 ).

The authors present a new meta-learning method that combines a deep neural network feature extractor with differentiable learning algorithms that have closed-form solutions.

This reduces the overall complexity of the gradient based meta-learning process, while advancing the state-of-the-art in terms of accuracy across multiple few-shot benchmarks.

We interacted with the authors through OpenReview 1 , bringing our reproducibility work and TensorFlow code 2,3 to their attention.

Because of this, they have recently updated their original paper with more details to facilitate reproduction and they have released an official PyTorch implementation 4 .

The objective of few-shot meta-learning is to train a model that can quickly adapt to a new task by using only a few datapoints and training iterations.

In our work we will consider only classification tasks, but it should be noted that meta-learning is also generally applicable to regression or reinforcement learning tasks BID4 ).In order to provide a solid definition of meta-learning, we need to define its different components.

We denote the set of tasks by T. A task T i ∈ T corresponds to a classification problem, with a probability distribution of example inputs x and (class) labels y, (x, y) ∼ T i .

For each task, we are given training samples Z T = {(x i , y i )} ∼ T with K shots per class and evaluation samples Z T = {(x i , y i )} ∼ T with Q shots (queries) per class, all sampled independently from the same distribution T .

In meta-learning, we reuse the learning experience used for tasks T i , i ∈ [0, L] to learn a new task T j , where j > L, from only K examples, for every single one of the N classes in the task.

Commonly, this is denoted as an N -way K-shot problem.

To this end, in meta-learning two different kinds of learners can be at play: (1) a base-learner that works at the task level and learns a single task (e.g. classifier with N classes) and (2) a meta-learner that produces those model parameters that enable the fastest average fine-tuning (using the base-learner) on unseen tasks.

The authors put a specific view of meta-learning forward.

Their meta-learning system consists of a generic feature extractor Φ(x) that is parametrized by ω, and a task-specific predictor f T (X) that is parametrized by w T and adapts separately to every task T ∈ T based on the few shots available.

In the case of a deep neural network architecture, this task-specific predictor f T can be seen as the last layer(s) of the network and is specific to a task T .

The preceding layers Φ can be trained across tasks to provide the best feature extraction on which the task-specific predictor can finetune with maximum performance.

The base-learning phase in their paper assumes that the parameters ω of the feature extractor Φ are fixed and computes the parameters w T of f T through closed-form learning process Λ. Λ, on its own, is parametrized by ρ.

The meta-learning phase in the paper learns a parametrization of Φ and Λ (respectively ω and ρ).

In order to learn those meta-parameters, the algorithm minimizes the expected loss on test sets from unseen tasks in T with gradient descent.

The base-learning and meta-learning phases are shown in FIG0 , respectively.

Most of the recent meta-learning works are tested against image datasets and their feature extractor consists of a convolutional neural network (CNN).

The variability between works resides mainly in the base learner f T and its parameter obtaining training procedure Λ. Examples are an (unparametrized) k-nearest-neighbour algorithm BID13 ), a CNN with SGD BID8 , and a nested SGD BID4 ).

Systems in BID13 and BID12 are based on comparing new examples in a learned metric space and rely on matching.

In particular, MATCHINGNET from BID13 uses neural networks augmented with memory and recurrence with attention in a few-shot image recognition context.

BID8 build on this attention technique by adding temporal convolutions to reuse information from past tasks.

Another example of a matching-based method is introduced in BID5 , where a graph neural network learns the correspondence between the training and testing sets.

A different approach is to consider the SGD update as a learnable function for meta-learning.

In particular, sequential learning algorithms, such as recurrent neural networks and LSTM-based methods, enable the use of long-term dependencies between the data and gradient updates as pointed out by BID10 .

Finally, BID4 introduce a technique called model-agnostic meta-learning (MAML).

In MAML, meta-learning is done by backpropagating through the fine-tuning stochastic gradient descent update of the model parameters. : Meta-learning of the meta-parameters ω and ρ over the evaluation sets of each task Z Ti using the previously learned w Ti following steps 7 to 9 of Algorithm 1.

In their paper, BID1 present a new approach that relies on using fast and simple base learners such as ridge regression differentiable discriminator (R2D2) or (regularized) logistic regression differentiable discriminator (LRD2).

In our reproducibility work we will focus on the R2D2 algorithm, because it is the only proposed algorithm with a truly closed-form solver for the base-learner.

For reproducibility purposes, we transformed the original textual description of R2D2 in their paper into an algorithmic description in Algorithm 1, elaborated upon in the following.

Algorithm 1 Ridge Regression Differentiable Discriminator (R2D2) Require: Distribution of tasks T. Require: Feature extractor Φ parameterized by ω.

Require: Finetuning predictor f T with base-learning algorithm Λ and task-specific parameters w T , and meta-parameters ρ = (α, β, λ) 1: Initialize Φ, Λ, and f T with pre-trained or random parameters ω 0 and ρ 0 2: while not done do Sample K datapoints for every class from T i and put in them in the training set Z Ti

Base-learn f Ti using Λ: DISPLAYFORM0 and Y i the one-hot labels from Z Ti .

Sample datapoints for every class from T i and put in them in the evaluation set Z Ti Update meta-parameters θ = (ω, ρ) through gradient descent : DISPLAYFORM0 with ε the learning rate, L the cross-entropy loss, and f Ti (X i ) = αX i W i + β.

10: end while In R2D2, during base-learning with Z T , the linear predictor f T is adapted for each training task T , by using the learning algorithm Λ; and the meta-parameters ω (of Φ) and ρ (of Λ) remain fixed.

It is only in the meta-training phase that meta-parameters ω and ρ are updated, by using Z T .

The linear predictor is seen as f T (x) = xW with W a matrix of task-specific weights w T , and x the feature extracted version of x, x = Φ(x).

This approach leads to a ridge regression evaluation such that it learns the task weights w T : DISPLAYFORM1 where X contains all N K feature extracted inputs from the training set of the considered task.

A key insight in their paper is that the closed-form solution of Equation 2 can be simplified using the Woodbury matrix identity yielding W = Λ(X, Y ) = X T (XX T + λI) −1 Y .

This considerably reduces the complexity of the matrix calculations in the special case of few-shot learning.

Specifically, XX T is of size N K × N K, in the case of an N -way K-shot task; this matrix will, together with the regularization, be relatively easily inverted.

Normally, regression is not adequate for classification, but the authors noticed that it still has considerable performance.

Therefore, in order to transform the regression outputs (which are only effectively calculated when updating the meta-parameters using Z T ) to work with the cross-entropy loss function, the meta-parameters (α, β) ∈ R 2 serve as a scale and bias, respectively: DISPLAYFORM2

Figure 3: Overall architecture of the R2D2 system considering [96, 192, 384, 512] filters in the feature extractor with 4 convolutional blocks for the CIFAR-FS dataset.

As a first step in the reproducibility, we reproduce the results of a baseline algorithm on different datasets used in their paper.

In this perspective, we first consider the MAML algorithm from Finn et al. [2017] .

We use the official TensorFlow implementation of MAML BID3 ) to reproduce the baseline's results.

Then, we amend this MAML implementation to reproduce the results on the new CIFAR-FS dataset proposed by their paper BID1 ).When reproducing the R2D2 algorithm, our first consideration is that the feature extractors in MAML and R2D2 are very different.

MAML uses four convolutional blocks with an organization of [32, 32, 32, 32] filters.

Whereas, R2D2's four blocks employ a [96, 192, 384, 512] scheme, as shown in Figure 3 .

In other words, the feature extractor in R2D2 is more complex hence is expected to yield better results BID7 ).

In order to provide a meaningful comparison, we implement and evaluate both the simple and more complex feature extractors for the R2D2 algorithm, denoted by R2D2* and R2D2 respectively.

In order to make a working reproduction of their paper we had to make the following assumptions.

We first considered the aforementioned complex architecture and feature extractor.

In particular, for the feature extractor, we made assumptions on the convolutional block options.

We considered a 3x3 convolution block with a 'same' padding and a stride of 1.

For the 2x2 maximum pooling, we use a stride of 2 and no padding.

Second, concerning the ridge regression base-learner, we opted for a multinomial regression that returns the class with the maximum value through one-hot encoding.

Following the guidelines for the feature extractor presented in Section 4.2 of their paper, we were not successful in reproducing the exact number of features at the output of the feature extractor.

In their paper, the overall numbers of features at the output of the extractor are 3584, 72576 and 8064 for Omniglot, miniImageNet and CIFAR-FS, respectively.

However, by implementing the feature extractor described in their paper, we obtain 3988, 51200 and 8192 respectively.

For comparison purposes, we use the same number of classes (e.g. 5) and shots during (e.g. 1) training and testing, despite their paper using a higher number of classes during training (16 for miniImageNet, 20 for CIFAR-FS) than during testing (5 for miniImageNet and CIFAR-FS).

Regarding the amount of shots, their paper uses a random number of shots during training.

This is different from the way most baselines are trained using the same number of shots per class during training and testing BID3 , BID9 , BID13 ).

For comparability, it is paramount to keep the training and testing procedures similar, if not the same.

In particular, as in their paper the 5-way results are exactly the same as those reported in MAML BID4 ), using the same number of classes and shots during training and testing allows for a justifiable comparison.

Finally, a last assumption is made on the algorithm's stopping criterion.

In their paper, the stopping criterion is vaguely defined as "the error on the meta-validation set does not decrease meaningfully for 20,000 episodes".

Therefore, in line with the MAML training procedure, we meta-train using 60,000 iterations.

To update the meta-parameters, in line with their paper, we use the Adam optimizer BID6 ) with an initial learning rate of 0.005, dampened by 0.5 every 2,000 episodes.

We use 15 examples per class for evaluating the post-update meta-gradient.

We use a meta batch-size of 4 and 2 tasks for 1-shot and 5-shot training respectively.

For MAML we use a task-level learning rate of 0.01, with 5 steps during training and 10 steps during testing.

The results of the different implemented architectures and algorithms for several datasets are shown in Figures 4 and 5.

More detailed results with 95% confidence intervals are shown in Tables 1 and 2 .

The first and last column correspond to the baselines in original papers.

Our implementations were made in Python 3.6.2 and TensorFlow 1.8.0 BID0 ).

The source code of all implementations is available 5 online 6 .

The simulations were run on a machine with 24 Xeon e5 2680s at 2.5 GHz, 252GB RAM and a Titan X GPU with 12 GB RAM.Although our results differ slightly from the original paper of BID1 , R2D2 (with its more complex network architecture) performs better than the MAML method for most simulations.

It is not a surprise that, in most of the cases, with a more complex feature extractor better results are obtained for the same algorithm (R2D2 vs R2D2*).

Overall, our study confirms that the R2D2 meta-learning method, with its corresponding complex architecture, yields better performance than basic MAML (with its simpler architecture).

The differences between reproduced results and reported values might be due to our assumptions or the stopping criterion in the training.

Also, as expected, the complexity (N-ways) and the amount of data (K-shots) play a major role in the classification accuracy.

The accuracy drops when the number of ways increases and number of shots decreases.

An outlier worth mentioning is our MAML simulation on miniImageNet: the 2-way 1-shot classification accuracy of 78.8 ± 2.8% is much better than the 74.9 ± 3.0% reported in BID4 .In summary, we successfully reproduced the most important results presented in BID1 .

Although our reproduced results and their paper results differ slightly, the general observations of the authors remain valid.

Their meta-learning with differentiable closed-form solvers yields stateof-the-art results and improves over another state-of-the-art method.

The assumptions made, however, could have been clarified in their original paper.

Indeed, these assumptions could be the source of the discrepancy in the reproduction results.

In this reproducibility work we did not focus on the logistic regression based algorithm (LRD2) from their paper because the logistic regression solver does not have a closed-form solution.

Overall, with this reproducibility project we make the following contributions:• Algorithmic description of the R2D2 version of meta-learning with differentiable closedform solvers (Algorithm 1).• Evaluation of the MAML pipeline from BID3 on two datasets: the existing miniImageNet and new CIFAR-FS for different few-shot multi-class settings.• Implementation of R2D2* in TensorFlow on the pipeline following Algorithm 1 with the original MAML feature extractor.• Implementation of R2D2 in TensorFlow on the pipeline following Algorithm 1 with the Figure 3 architecture as mimicked from in the original paper BID1 ).•

Evaluation and insights in the reproducibility of BID1 .

In this work we have presented a reproducibility analysis of the ICLR 2019 paper "Meta-learning with differentiable closed-form solvers" by BID1 .

Some parameters and training methodologies, which would be required for full reproducibility, such as stride and padding of the convolutional filters, and a clear stopping criterion, are not mentioned in the original paper or in its appendix BID1 ).

However, by making reasonable assumptions, we have been able to reproduce the most important parts of the paper and to achieve similar results.

Most importantly we have succeeded in reproducing the increase in performance of the proposed method over some reproduced baseline results, which supports the conclusions of the original paper.

However, the different neural network architectures should be taken into consideration when comparing results.

Table 1 .

Table 1 : N -way K-shot classification accuracies on CIFAR-FS with 95% confidence intervals.

MAML paper BID1 MAML ours R2D2* ours R2D2 ours R2D2 paper BID1 5-way, 1-shot 58.9 ± 1.9% 56.8 ± 1.9% 54.3 ± 1.8% 60.2 ± 1.8% 65.3 ± 0.2% 5-way, 5-shot 71.5 ± 1.0% 70.8 ± 0.9% 69.7 ± 0.9% 70.9 ± 0.9% 79.4 ± 0.1% 2-way, 1-shot 82.8 ± 2.7% 83.1 ± 2.6% 78.3 ± 2.8% 83.6 ± 2.6% 83.4 ± 0.3% 2-way, 5-shot 88.3 ± 1.1% 88.5 ± 1.1% 87.7 ± 1.1% 89.0 ± 1.0% 91.1 ± 0.2% Table 2 .

Table 2 : N -way K-shot classification accuracies on miniImageNet with 95% confidence intervals Method MAML paper BID4 MAML code BID3 R2D2* ours R2D2 ours R2D2 paper BID1 5-way, 1-shot 48.7 ± 1.8% 47.6 ± 1.9% 45.7 ± 1.8% 51.7 ± 1.8% 51.5 ± 0.2% 5-way, 5-shot 63.1 ± 0.9% 62.3 ± 0.9% 63.7 ± 1.3% 63.3 ± 0.9% 68.8 ± 0.2% 2-way, 1-shot 74.9 ± 3.0% 78.8 ± 2.8% 74.7 ± 2.9% 74.6 ± 2.9% 76.7 ± 0.3% 2-way, 5-shot 84.4 ± 1.2% 82.6 ± 1.2% 83.0 ± 1.2% 84.6 ± 1.2% 86.8 ± 0.2%

<|TLDR|>

@highlight

We successfully reproduce and give remarks on the comparison with baselines of a meta-learning approach for few-shot classification that works by backpropagating through the solution of a closed-form solver.