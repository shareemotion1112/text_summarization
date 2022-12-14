While tasks could come with varying the number of instances and classes in realistic settings, the existing meta-learning approaches for few-shot classification assume that number of instances per task and class is fixed.

Due to such restriction, they learn to equally utilize the meta-knowledge across all the tasks, even when the number of instances per task and class largely varies.

Moreover, they do not consider distributional difference in unseen tasks, on which the meta-knowledge may have less usefulness depending on the task relatedness.

To overcome these limitations, we propose a novel meta-learning model that adaptively balances the effect of the meta-learning and task-specific learning within each task.

Through the learning of the balancing variables, we can decide whether to obtain a solution by relying on the meta-knowledge or task-specific learning.

We formulate this objective into a Bayesian inference framework and tackle it using variational inference.

We validate our Bayesian Task-Adaptive Meta-Learning (Bayesian TAML) on two realistic task- and class-imbalanced datasets, on which it significantly outperforms existing meta-learning approaches.

Further ablation study confirms the effectiveness of each balancing component and the Bayesian learning framework.

Despite the success of deep learning in many real-world tasks such as visual recognition and machine translation, such good performances are achievable at the availability of large training data, and many fail to generalize well in small data regimes.

To overcome this limitation of conventional deep learning, recently, researchers have explored meta-learning (Schmidhuber, 1987; Thrun & Pratt, 1998) approaches, whose goal is to learn a model that generalizes well over distribution of tasks, rather than instances from a single task, in order to utilize the obtained meta-knowledge across tasks to compensate for the lack of training data for each task.

However, so far, most existing meta-learning approaches (Santoro et al., 2016; Vinyals et al., 2016; Snell et al., 2017; Ravi & Larochelle, 2017; Finn et al., 2017; have only targeted an artificial scenario where all tasks participating in the multi-class classification problem have equal number of training instances per class.

Yet, this is a highly restrictive setting, as in real-world scenarios, tasks that arrive at the model may have different training instances (task imbalance), and within each task, the number of training instances per class may largely vary (class imbalance).

Moreover, the new task may come from a distribution that is different from the task distribution the model has been trained on (out-of-distribution task) (See (a) of Figure 1 ).

Under such a realistic setting, the meta-knowledge may have a varying degree of utility to each task.

Tasks with small number of training data, or close to the tasks trained in meta-training step may want to rely mostly on meta-knowledge obtained over other tasks, whereas tasks that are out-of-distribution or come with more number of training data may obtain better solutions when trained in a task-specific manner.

Furthermore, for multi-class classification, we may want to treat the learning for each class differently to handle class imbalance.

Thus, to optimally leverage meta-learning under various imbalances, it would be beneficial for the model to task-and class-adaptively decide how much to use from the meta-learner, and how much to learn specifically for each task and class.

Head class

Figure 1: Concept.

(a) To handle task imbalance (Task Imbal.), class imbalance (Class Imbal.) and outof-distribution tasks (OOD) for each task ?? , we introduce task-specific balancing variables ?? ?? , ?? ?? and z ?? , respectively.

(b) With those variables, we learn to balance between the meta-knowledge ?? and task-specific update to handle imbalances and distributional discrepancies.

To this end, we propose a novel Bayesian meta-learning framework, which we refer to as Bayesian Task-Adaptive Meta-Learning (Bayesian TAML), that learns variables to adaptively balance the effect of meta-and task-specific learning.

Specifically, we first obtain set-representations for each task, which are learned to convey useful statistics about the task or class distribution, such as mean, variance, tailedness (kurtosis), and skewness, and then learn the distribution of three balancing variables as the function of the set: 1) task-dependent learning rate decay, which decides how far away to deviate from the meta-knowledge, when performing task-specific learning.

Tasks with higher shots could benefit from taking gradient steps afar, while tasks with few shots may need to stay close to the initial parameter.

2) class-dependent learning rate, which decides how much information to use from each class, to automatically handle class imbalance where the number of instances per class can largely vary.

3) task-dependent attention mask, which modifies the shared parameter for each task by learning a set-dependent attention mask to it, such that the task can decide how much and what to use from the initial shared parameter and what to ignore based on its set representation.

This is especially useful when handling out-of-distribution task, which may need to ignore some of the meta-knowledge.

We validate our model on Omniglot and mini-ImageNet dataset, as well as a new dataset that consists of heterogeneous datasets, under a scenario where every class in each episode can have any number of shots, that leads to task and class imbalance, and where the dataset at meta-test time is different from that of meta-training time.

The experimental results show that our Bayesian TAML significantly improves the performance over the existing approaches under these realistic scenarios.

Further analysis of each component reveals that the improvement is due to the effectiveness of the balancing terms for handling task and class imbalance, and out-of-distribution tasks.

To summarize, our contribution in this work is threefold:

??? We consider a novel problem of meta-learning under a realistic task distribution, where the number of instances across classes and tasks could largely vary, or the unseen task at the meta-test time is largely different from the seen tasks.

??? For effective meta-learning with such imbalances, we propose a Bayesian task-adaptive meta-learning (Bayesian TAML) framework that can adaptively adjust the effect of the meta-learner and the task-specific learner, differently for each task and class.

??? We validate our model on realistic imbalanced few-shot classification tasks with a varying number of shots per task and class and show that it significantly outperforms existing meta-learning models.

Meta-learning Meta-learning (Schmidhuber, 1987; Thrun & Pratt, 1998) is an approach to learn a model to generalize over a distribution of task.

The approaches in general can be categorized into either memory-based, metric-based, and optimization-based methods.

A memory-based approach (Santoro et al., 2016) learns to store correct instance and label into the same memory slot and retrieve it later, in a task-generic manner.

Metric-based approaches learn a shared metric space (Vinyals et al., 2016; Snell et al., 2017) .

Snell et al. (2017) defines the distance between the instance and the class prototype, such that the instances are closer to their correct prototypes than to others.

As for optimization-based meta-learning, MAML Finn et al. (2017) learns a shared initialization parameter that is optimal for any tasks within few gradient steps from the initial parameter.

Meta-SGD improves upon MAML by learning the learning rate differently for each parameter.

For effective learning of a meta-learner, meta-learning approaches adopt the episodic training strategy (Vinyals et al., 2016) which trains and evaluates a model over a large number of tasks, which are called meta-training and meta-test phase, respectively.

However, existing approaches only consider an artificial scenario which samples the classification of classes with exactly the same number of training instances, both within each episode and across episodes.

On the other hand, we consider a more challenging scenario where the number of shots per class and task could vary at each episode, and that the task given at meta-test time could be an out-of-distribution task.

Task-adaptive meta-learning The goal of learning a single meta-learner that works well for all tasks may be overly ambitious and leads to suboptimal performances for each task.

Thus recent approaches adopt task-adaptively modified meta-learning models.

Oreshkin et al. (2018) proposed to learn the temperature scaling parameter to work with the optimal similarity metric.

Qiao et al. (2018) also suggested a model that generates task-specific parameters for the network layers, but it only trains with many-shot classes, and implicitly expects generalization to few-shot cases.

Rusu et al.

(2018) proposed a network type task-specific parameter producer, and Lee & Choi (2018) proposed to differentiate the network weights into task-shared and task-specific weights.

Our model also aims to obtain task-specific parameter for each task, but is rather focused on learning how to balance between the meta-learning and task-/class-specific learning.

To our knowledge, none of the existing approaches explicitly tackle this balancing problem since they only consider few-shot learning with the fixed number of instances for each class and task.

Probabilistic meta-learning Recently, a probabilistic version of MAML has been proposed (Finn et al., 2018) , where they interpret a task-specific gradient update as a posterior inference process under variational inference framework.

Kim et al. (2018) proposed Bayesian MAML with a similar motivation but with a stein variational inference framework and chaser loss.

Gordon et al. (2018) proposed a probabilistic meta-learning framework where the paramter for a novel task is rapidly estimated under decision theoretic framework, given a set representation of a task.

The motivation behind these works is to represent the inherent uncertainty in few-shot classification tasks.

Our model also uses Bayesian modeling, but it focuses on leveraging the uncertainties of the meta-learner and the gradient-direction in order to balance between meta-and task-or class-specific learning.

We first introduce notations and briefly recap the model-agnostic meta-learning (MAML) by Finn et al. (2017) .

Suppose a task distribution p(?? ) that randomly generates task ?? consisting of a training set

Then, the goal of MAML is to meta-learn the initial model parameter ?? as a meta-knowledge to generalize over the task distribution p(?? ), such that we can easily obtain the task-specific predictor ?? ?? in a single (or a few) gradient step from the initial ??.

Toward this goal, MAML optimizes the following gradient-based meta-learning objective:

where ?? denotes stepsize and L denotes empirical loss such as negative log-likelihood of observations.

Note that by meta-learning the initial point ??, the task-specific predictor ??

even with D ?? which only contains few samples.

We can easily extend the Eq. (1), such that we obtain ?? ?? with more than one inner-gradient steps from the initial ??.

However, the existing MAML framework has the following limitations that prevent the model from efficiently solving real-world problems involving task/class imbalance and out-of-distribution tasks.

1.

Task imbalance.

MAML has a fixed number of inner-gradient steps and stepsize ?? across all tasks, which prevents the model from adaptively deciding how much to use from the meta-knowledge depending on the number of the training examples per task.

2.

Class imbalance.

The model does not provide any framework to handle class imbalance within each task.

Therefore, classes with large number of training instances (head classes) may dominate the task-specific learning during the inner-gradient steps, yielding low performance on classes with fewer shots (tail classes).

3.

Out-of-distribution tasks.

The model assumes that the meta-knowledge will be equally useful for the unseen tasks, but for unseen tasks that are out-of-distribution, the metaknowledge may be less useful.

As shown in Figure 1 for the concepts, we introduce three balancing variables ?? ?? , ?? ?? , z ?? to tackle each problem mentioned above.

How to compute these variables will be described in Section 4.

In order to learn with realistic scenarios, we assume that the task distribution p(?? ) samples some fixed number of C classes ("way"), and then sample uniform-random number of instances for each class ("shots"), thereby simulating both task and class imbalance at the same time.

Tackling task imbalance.

To control whether to stay close to the initial parameter or deviate far from it, we introduce a clipping function f (??) = max(0, min(??, 1)) and a task-dependent learning-rate decaying factor f (?? ?? ), such that the learning rate exponentially decays as

to be large for large tasks, such that they rely more on task-specific updates, while small tasks use small f (?? ?? ) to benefit from the meta-knowledge.

Tackling class imbalance.

To handle class imbalance, we vary the learning rate of classspecific gradient update for each task-specific gradient update step.

Specifically, for class c = 1, . . .

, C, we introduce a non-negative activation function g(??) = SoftPlus(??) and a set of classspecific non-negative scalars g(??

is the set of instances and labels for class c.

We expect g(?? ?? c ) to be large for tail-classes to consider them more in task-specific gradient updates.

Tackling out-of-distribution tasks.

Lastly, we introduce an additional task-dependent variable z ?? with the non-negative activation function g(??) which weights the initial parameter ?? according to the usefulness for each task.

We expect the variable g(z ?? ) to heavily emphasize the meta-knowledge ?? when D ?? is similar to the trained dataset, and use less of it when D ?? is unfamilar.

This behavior can be implemented with Bayesian modeling on the latent z ?? , which we introduce in the next subsection.

A unified framework.

Finally, we assemble all these components together into a single unified framework.

The update rule for the task-specific ?? ?? is recursively defined as follows:

where the last step ?? K corresponds to the task-specific predictor ?? ?? and ?? is a multi-dimensional global learning rate vector that is learned such as .

As previously mentioned, we need a Bayesian framework for modeling z ?? , since it needs a prior in order to prevent the posterior of z ?? from overly utilizing the meta-knowledge ?? when the task is out-ofdistribution.

Moreover, for the learning of balancing variables ?? ?? and ?? ?? , Bayesian modeling improve the quality of the inference on them, which we empirically verified through extensive experiments.

We allow the three variables to share the same inference network pipeline to minimize the computational cost, and thereby effectively amortize the inference rule across variables as well.

for training, and

for test.

Let ?? ?? denote the collection of three latent variables, ?? ?? , ?? ?? and z ?? for uncluttered notation.

Then, the generative process is as follows for each task ?? (See Figure 2) :

for the complete data likelihood.

Note that the deterministic ?? is shared across all the tasks.

The goal of learning for each task ?? is to maximize the log-likelihood of the joint datasetD ?? and

.

However, solving it involves the true posterior p(?? ?? |D ?? ,D ?? ), which is intractable.

Thus, we resort to amortized variational inference with a tractable form of approximate posterior q(?? ?? |D ?? ,D ?? ; ??) parameterized by ??.

Further, similarly to Ravi & Beatson (2018) , we drop the dependency on the test datasetD ?? for the approximate posterior, in order to make the two different pipelines consistent; one for meta-training where we observe the whole test dataset, and the other for meta-testing where the test labels are unknown.

The form of our approximate posterior is now q(?? ?? |D ?? ; ??).

It greatly simplifies the inference framework, while ensuring that the following objective is still a valid lower bound of the log evidence.

Also, considering that performing the inner-gradient steps with the training dataset D ?? automatically maximizes the training log-likelihood in MAML framework, we slightly modify the objective so that the expected loss term only involves the test examples.

The resultant form of the lower bound that suits for our meta-learning purpose is as follows:

We assume q(?? ?? |D ?? ; ??) fully factorizes for each variable and also for each dimension as well:

where we assume that each single dimension of q(?? ?? |D ?? ; ??) follows univariate gaussian having trainable mean and variance.

We also let each dimension of prior p(?? ?? ) factorize into N (0, 1).

The KL-divergence between two univariate gaussians has a simple closed form (Kingma & Welling, 2014) , thereby we obtain the low-variance estimator for the lower bound L ?? ??,?? .

The final form of the meta-training minimization objective with Monte-Carlo approximation for the expection in (5) is as follows:

, and T is the number of tasks.

We implicitly assume the reparameterization trick for ?? ?? to obtain stable and unbiased gradient estimate w.r.t.

?? (Kingma & Welling, 2014).

We set the number of MC samples to S = 1 for meta-training for computational efficiency.

When meta-testing, we can set S = 10 or naively approximate the expectation by taking the expectation inside:

, which works well in practice.

The main challenge in modeling our variational distribution q(?? ?? |D ?? ; ??) is how to refine the training dataset D ?? into informative representation capturing the dataset as a distribution, which is not trivial.

This inference network should capture all the necessary statistical information in the dataset D ?? to solve both imblanace and out-of-distribution problems.

DeepSets (Zaheer et al., 2017 ) is frequently used as a practical set-encoder, where each instance in the set is transformed by the shared nonlinearity, and then summed together to generate a single vector summarizing the set.

However, for the classification dataset D ?? which is the set of (class) sets, we cannot use DeepSets directly as it will completely ignore the label information.

Therefore, we need to stack the structure of DeepSets twice according to the hierarchical set of sets structure of classification dataset.

However, there exists additional limitation of DeepSets with sum-pooling when describing the distribution.

Suppose that we have a set containing a replication of single instance.

Then, its representation will change based on the number of replications, although distribution-wise all sets should be the same.

Mean-pooling may alleviate the problem; however, it does not recognize the number of elements in the set, which is a critical limitation in encoding imbalance.

To overcome the limitations of the two pooling methods, we propose to use higher-order statistics in addition to the sample mean, namely element-wise sample variance, skewness and kurtosis.

For instance, the sample variance could capture task imbalance and skewness will capture class imbalance (imbalance in the number of instances per class).

Based on this intuition, we propose the following encoder network StatisticsPooling(??) that generates the concatenation of those statistics (See Figure 3) :

for classes c = 1, . . . , C, and X ?? c is the collection of class c examples in task t. NN 1 and NN 2 are some appropriate neural networks parameterized by ??.

The vector v ?? finally summarizes the whole classification dataset D ?? and our balancing variables ?? ?? , ?? ?? and z ?? are generated from it with an additional affine transformation.

See Appendix B for the justification.

We validate our method in imbalanced scenarios, where each task, or every class within a task can have different shots, and the tasks at the evaluation time could come from a different task distribution from the seen task distribution.

Following Finn et al. (2017) , we use 4-block convolutional neural networks with 64 channels for each layer for Omniglot and MNIST.

We reduce the number of channels into 32 for other datasets.

Imbalanced Omniglot.

This dataset (Lake et al., 2015) consists of 1623 hand-written character classes, with 20 training instances per class.

We consider 10-way classification problems, where we have 5 queries per each class.

To generate imbalanced tasks, we randomly set the number of training instances to be sampled within the range of 1 to 15.

We set 5 inner-gradient steps for all gradient-based models and train all models on Omniglot, and evaluate on both Omniglot and MNIST, where the latter is used to evaluate the performance on out-of-distribution task.

Imbalanced tiered-ImageNet.

This is sub-sampled ImageNet dataset including 608 classes (Ren et al., 2018) .

As similarly with the Imbalanced Omniglot dataset, we consider 5-way classification problems, while randomly setting the number of training instances per class within the range of 1 to 50, and use 15 instances per class for test (queries).

We set 5 inner-gradient steps for all gradient-based models and train all models on the tiered-ImageNet dataset, and evaluate the model on the test split of tiered-ImageNet and mini-ImageNet, where the latter is used to evaluate on out-of-distribution task.

See the Appendix A for more details of the experimental setup.

Analysis.

Multi-Dataset OVD.

We further test our model under a more challenging setting where tasks could come from a highly heterogeneous dataset.

To this end, we combine Omniglot, VGG flower (Nilsback & Zisserman, 2008) , DTD (Cimpoi et al., 2014 ) into a single dataset OVD, and randomly sample each class from the combined dataset for every task.

We train all models with 10-way any-shot tasks with 3 inner gradient steps and test on OVD and FasionMNIST (Xiao et al., 2017) , where the latter is used to generate out-of-distribution tasks at evaluation time.

The results in Table 2 shows that under this challenging multi-dataset setting, our Bayesian TAML outperforms all baselines, especially with larger gains on the out-of-distribution tasks (Fashion MNIST) consistent with the result of Table 1.

We now validate the effectiveness of each balancing variable.

For all of the ablations, we set the meta-training condition to Omniglot 5-way any-shot (i.e. 1-to 15-shot) classification with 5 innergradient steps.

For the meta-testing condition for each ablation, see each of the tables below.

Further, in order to correctly evaluate each variable, we add in each individual component to Meta-SGD, one at a time, dropping all other balancing variables, for both meta-training and meta-testing.

We report mean accuracies over 1000 random episodes with 95% confidence intervals.

g(z ?? ) for handling distributional discrepancy.

g(z ?? ) modulates the initial model parameter ??, deciding what and how much to use from the meta-knowledge ?? based on the relatedness between ?? and the task at hand.

Table 3 shows that by adding in Bayesian g(z ?? ) component to Meta-SGD, we can effectively handle out-of-distribution tasks (MNIST) by significant margin.

The histogram in Figure 4 shows that the actual distribution of mask g(z ?? ) of OOD tasks is more skewed toward zero than the distribution of ID tasks, which agrees with our expectation.

f (?? ?? ) for handling task imbalance.

f (?? ?? ), which is a decaying factor for inner gradient steps, handles inter-task imbalance where each task has different number of examples.

Figure 5 shows f (?? ?? ) values w.r.t.

the task size, where it increases monotonically with the number of instances allowing the model to stay close to the initial parameter for few-shot cases and deviate far from it for many-shot cases.

Table 4 shows that the larger gains of our model for 1-shot than 5 or 15 shots support that relying on meta-knowledge is useful for improving the performance on smaller tasks.

g(?? ?? ) for handling class imbalance.

g(?? ?? ) rescales the class-specific gradients to handle class imbalance where the number of instances per class largely varies.

Table 5 shows the results under the varying degree of class imbalance across the task distribution.

The degree of class imbalance (??N ) means that the maximum number of shots is N times larger than the minimum number of shots within the given task.

We observe that our model significantly outperforms baselines especially under the high degree of class imbalance (??5 and ??15).

Figure 6 shows that g(?? ?? ) actually increases the gradient scale of tail-classes (classes with fewer instances), so that we obtain the big improvements on those tail classes.

Effectiveness of Bayesian modeling We further demonstrate the effectiveness of Bayesian modeling by comparing it with the deterministic version of our model (Deterministic TAML), where three balancing variables are no longer stochastic and we apply 2 regularization of 10 ???3 on them instead of KL-divergence in Eq. 5.

Table 6 shows the results under the same setting of Table 3 .

The results clearly show that the Bayesian modeling greatly contribute to addressing the imbalance problem, especially for the OOD tasks (MNIST).

Figure 7 further shows that our balancing variables g(?? ?? ) and f (?? ?? ), that are responsible for handling class/task imbalance, more sensitively react to the actual imbalance conditions with the Bayesian modeling (Bayesian TAML) than without Bayesian (Deterministic TAML).

Dataset encoding Lastly, we perform an ablation study to validate the effectiveness of the proposed dataset encoding scheme, Set of Sets, for generating the balancing variables.

Table 7 shows the performance of various encoding schemes on the imbalanced tiered-ImageNet 5-way classification, the same setting as in Table 1 .

We see that Set of Sets, equipped with higher-order statistics and hierarchical set encoding, is far more effective than simple mean-pooling method (Zaheer et al., 2017; Edwards & Storkey, 2016; Garnelo et al., 2018) .

We propose Bayesian TAML that learns to balance the effect of meta-learning and task-adaptive learning, to consider meta-learning under a more realistic task distribution where each task and class can have varying number of instances.

Specifically, we encode the dataset for each task into hierarchical set-of-sets representations, and use it to generate attention mask for the original parameter, learning rate decay, and the class-specific learning rate.

We use a Bayesian framework to infer the posterior of these balancing variables, and propose an effective variational inference framework to solve for them.

Our model outperforms existing meta-learning methods when validated on imbalanced few-shot classification tasks.

Further analysis of each balancing variable shows that each variable effectively handles task imbalance, class imbalance, and out-of-distribution tasks respectively.

We believe that our work makes a meaningful step toward application of meta-learning to real-world problems.

A EXPERIMENTAL SETUP A.1 BASELINES AND NETWORK ARCHITECTURE.

We describe baseline models and our task-adaptive learning to balance model.

Note that all gradientbased models can be extended to take K inner-gradient steps for both meta-training and meta-testing.

1) Meta-Learner LSTM.

A meta-learner that learns optimization algorithm with LSTM (Ravi & Larochelle, 2017) .

The model performs few-shot classification using cosine similarities between the embeddings generated from a shared convolutional network.

2) Prototypical Networks.

A metric-based few-shot classification model proposed by (Snell et al., 2017) .

The model learns the metric space based on Euclidean distance between class prototypes and query embeddings.

3) MAML.

The Model-Agnostic Meta-Learning (MAML) model by (Finn et al., 2017) , which aims to learn the global initial model parameter, from which we can take a few gradient steps to get task-specific predictors.

A base MAML with the learnable learning-rate vector (without any restriction on sign) element-wisely multiplied to each step inner-gradient .

A gradient-based meta-learning model proposed by Lee & Choi (2018) .

The model obtains a task-specific parameter only w.r.t.

a subset of the whole dimension (M-Net), followed by a linear transformation to learn a metric space (T-Net).

6) Probabilistic MAML.

A probabilistic version of MAML by Finn et al. (2018) , where they model task-adaptive inner-gradient steps as a posterior inference process under hierarchical Bayesian framework.

This model also interprets MAML under hierarchical Bayesian framework, but they propose to share and amortize the inference rules across both global initial parameters as well as the task-specific parameters.

8) Bayesian TAML.

Our learning to balance model that can adaptively balance between meta-and task-specific learners for each task and class.

A.2 REALISTIC ANY-SHOT CLASSIFICATION.

We describe more detailed settings for realistic any-shot classification.

Imbalanced Omniglot.

We modified the episode generating strategy of C-way classification, which selects the number of shots randomly between 1 to 15 for each of the classes.

The metalearning rate ?? and the total number of iterations are set to 1e-3 and 60000, respectively for all models, and the inner gradient step size ?? is set to 0.05 for MAML, MT-NET, and ABML and is set to learnable parameters for Meta-SGD and our model.

The number of inner-gradient steps is 5 for all gradient-based models and all other components are the same as reported for fixed-way and fixed-shot few-shot classification in the paper for each model.

We keep the meta-batch as 1 for all experiments to clearly see the effect of imbalance scenario.

We trained models in 10-way 5 inner-gradient steps on the Omniglot, and evaluated with the test split of Omniglot and MNIST.

Imbalanced tiered-ImageNet.

We modified the episode generating strategy of C-way classification, which selects the number of shots randomly between 1 to 50 for each of the classes.

We set the number of query points as 15 and the meta learning rate ?? is set to 1e-4.

Other components are set to the same as referred in Imbalanced Omniglot.

We trained models on the tiered-ImageNet, and evaluated with the test split of tiered-ImageNet and mini-ImageNet.

We describe the network architecture of the inference network that takes a classification dataset as an input and generates three balancing variables as output.

We additionally used two average pooling with 2 ?? 2 strides before the shared encoder NN 1 with large inputs such as tiered-ImageNet and mini-ImageNet.

We empirically found that attaching average pooling reduces computation cost while improving performance.

Based on the previous justification of DeepSets (Zaheer et al., 2017) , we can easily justify the Set-ofSets structure proposed in the main paper as well, in terms of the two-level permutation invariance properties required for any classification dataset.

The main theorem of DeepSets is: Theorem 1.

A function f operating on a set X ??? X is a valid set function (i.e. permutation invariant), iff it can be decomposed as f (X) = ?? 2 ( x???X ?? 1 (x)), where ?? 1 and ?? 2 are appropriate nonlinearities.

See (Zaheer et al., 2017) for the proof.

Here we apply the same argument twice as follows.

1.

A function f operating on a set of representations {s 1 , . . . , s C } (we assume each s c is an output from a shared function g) is a valid set function (i.e. permutation invariant w.r.t.

the order of {s 1 , . . .

, s C }), iff it can be decomposed as f ({s 1 , . . .

, s C }) = ?? 2 ( C c=1 ?? 1 (s c )) with appropriate nonlinearities ?? 1 and ?? 2 .

2.

A function g operating on a set of examples {x c,1 , . . .

, x c,N } is a valid set function (i.e. permutation invariant w.r.t.

the order of {x c,1 , . . .

, x c,N }) iff it can be decomposed as

) with appropriate nonlinearities ?? 3 and ?? 4 .

Inserting s c = g({x c,1 , . . .

, x c,N }) into the expression of f , we arrive at the following valid composite function operating on a set of sets:

Let F denote the composite of f and (multiple) g and let NN 2 denote the composite of ?? 1 and ?? 4 .

Further define NN 1 := ?? 3 and NN 3 := ?? 2 .

Then, we have

where C is the number of classes and N is the number of examples per class.

See Section A.3 for the correspondence between Eq. (9) and the actual encoder structure.

We provide the comparison between the two approximation schemes for evaluating the expectation of the test example predictions at meta-testing time.

Naive approximation means that we take the expectation inside (i.e. we do not sample) and MC approximation means that we perform MonteCarlo integration with sample size S = 10 1 .

We see from the Table 8 that MC ingetration performs better than the naive approximation, especially with OOD tasks (e.g. MNIST, mini-ImageNet).

This is because the predictive distributions involve higher uncertainty for OOD tasks, hence there exists more benefit from considering the large variance than simply ignoring it.

and tiered-ImageNet(right) with imbalanced setting.

All reported results are average performances over 1000 for Omniglot and MNIST and 600 for tiered-ImageNet and mini-ImageNet randomly selected episodes with standard errors for 95% confidence interval over tasks.

We further compare our model with the existing meta-learning approaches on conventional few-shot classification with fixed-way and fixed-shot.

Omniglot.

We report the 20-way classification performance for this dataset.

Following Finn et al. (2017) , we use 4-block convolutional neural network architecture with 64 channels for each layer.

We set the number of inner-gradient steps K to 5 for both meta-training and meta-testing.

Mini-ImageNet.

We report the 5-way classification performance with the meta-batch 4 and 2 for 1-and 5-shot, respectively.

We reduce the convolution filter size of the 4-block CNN network into 32 to prevent overfitting.

We set K = 5 for multi-step models for both meta-training and meta-testing. (Finn et al., 2017) 95.80 ?? 0.30 98.90 ?? 0.20 48.70 ?? 1.84 63.11 ?? 0.92 Meta-SGD 95. (Yang et al., 2017) 97.60 ?? 0.20 99.10 ?? 0.10 50.44 ?? 0.82 65.32 ?? 0.70 MT-NET (Lee & Choi, 2018) 96.20 ?? 0.40 -51.70 ?? 1.84 -Probabilistic MAML (Finn et al., 2018) --50.13 ?? 1.86 -Reptile (Nichol et al., 2018) 89.43 ?? 0.14 97.12 ?? 0.32 49.97 ?? 0.32 65.99 ?? 0.58 BMAML (Kim et al., 2018) --53.80 ?? 1.46 -VERSA (Gordon et al., 2018) 97.

We first compare our method on conventional fixed-way fixed-shot classification task against existing meta-learning methods.

Though the classification with uniformly distributed instances is not the task we aim to tackle, we find that Bayesian TAML outperforms most baseline models, except for the mini-imagenet 5-way 1-shot experiment where BMAML works the best.

Table 9 : Classification results to ??0 inititalization with g(z).

?? 0 inititalization with g(z).

Applying g(z) to the shared initial parameter ?? may set some of the initial parameters to smaller values.

Thus, we further compare our model with applying the linear interpolation between ?? and a random initialization vector ?? random with coefficient g(z) as ?? 0 = g(z)

??? ?? + (1 ??? g(z)

??? ?? random ) to Bayesian TAML (Bayesian g(z) TAML + Interpolation).

As shown in the Table 9 , those models has no significant difference in the performance.

1 At meta-training time, we perform MC approximation with a single sample for computational efficiency 2 We adopt the accuracies of the Prototypical Network in the setting which the number of shot and way are the same for training and testing phase for the consistency with other methods.

In the "higher way" setting where 20-way is used during training for 5-way testing, the reported performance of the model is 68.20 ?? 0.66%.

<|TLDR|>

@highlight

A novel meta-learning model that adaptively balances the effect of the meta-learning and task-specific learning, and also class-specific learning within each task.