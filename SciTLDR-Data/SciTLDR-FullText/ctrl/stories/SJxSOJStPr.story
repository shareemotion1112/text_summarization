Despite the growing interest in continual learning, most of its contemporary works have been studied in a rather restricted setting where tasks are clearly distinguishable, and task boundaries are known during training.

However, if our goal is to develop an algorithm that learns as humans do, this setting is far from realistic, and it is essential to develop a methodology that works in a task-free manner.

Meanwhile, among several branches of continual learning, expansion-based methods have the advantage of eliminating catastrophic forgetting by allocating new resources to learn new data.

In this work, we propose an expansion-based approach for task-free continual learning.

Our model, named Continual Neural Dirichlet Process Mixture (CN-DPM), consists of a set of neural network experts that are in charge of a subset of the data.

CN-DPM expands the number of experts in a principled way under the Bayesian nonparametric framework.

With extensive experiments, we show that our model successfully performs task-free continual learning for both discriminative and generative tasks such as image classification and image generation.

Humans consistently encounter new information throughout their lifetime.

The way the information is provided, however, is vastly different from that of conventional deep learning where each minibatch is iid-sampled from the whole dataset.

Data points adjacent in time can be highly correlated, and the overall distribution of the data can shift drastically as the training progresses.

Continual learning (CL) aims at imitating incredible human's ability to learn from a non-iid stream of data without catastrophically forgetting the previously learned knowledge.

Most CL approaches (Aljundi et al., 2018; 2017; Lopez-Paz & Ranzato, 2017; Kirkpatrick et al., 2017; Rusu et al., 2016; Shin et al., 2017; Yoon et al., 2018) assume that the data stream is explicitly divided into a sequence of tasks that are known at training time.

Since this assumption is far from realistic, task-free CL is more practical and demanding but has been largely understudied with only a few exceptions of (Aljundi et al., 2019a; b) .

In this general CL, not only is explicit task definition unavailable but also the data distribution gradually shifts without a clear task boundary.

Meanwhile, existing CL methods can be classified into three different categories (Parisi et al., 2019) : regularization, replay, and expansion methods.

Regularization and replay approaches address the catastrophic forgetting by regularizing the update of a specific set of weights or replaying the previously seen data, respectively.

On the other hand, the expansion methods are different from the two approaches in that it can expand the model architecture to accommodate new data instead of fixing it beforehand.

Therefore, the expansion methods can bypass catastrophic forgetting by preventing pre-existing components from being overwritten by the new information.

The critical limitation of prior expansion methods, however, is that the decisions of when to expand and which resource to use heavily rely on explicitly given task definition and heuristics.

In this work, our goal is to propose a novel expansion-based approach for task-free CL.

Inspired by the Mixture of Experts (MoE) (Jacobs et al., 1991) , our model consists of a set of experts, each of which is in charge of a subset of the data in a stream.

The model expansion (i.e., adding more experts) is governed by the Bayesian nonparametric framework, which determines the model complexity by the data, as opposed to the parametric methods that fix the model complexity before training.

We formulate the task-free CL as an online variational inference of Dirichlet process mixture models consisting of a set of neural experts; thus, we name our approach as the Continual Neural Dirichlet Process Mixture (CN-DPM) model.

We highlight the key contributions of this work as follows.

•

We are one of the first to propose an expansion-based approach for task-free CL.

Hence, our model not only prevents catastrophic forgetting but also applies to the setting where no task definition and boundaries are given at both training and test time.

Our model named CN-DPM consists of a set of neural network experts, which are expanded in a principled way built upon the Bayesian nonparametrics that have not been adopted in general CL research.

• Our model can deal with both generative and discriminative tasks of CL.

With several benchmark experiments of CL literature on MNIST, SVHN, and CIFAR 10/100, we show that our model successfully performs multiple types of CL tasks, including image classification and generation.

2 BACKGROUND AND RELATED WORK 2.1 CONTINUAL LEARNING Parisi et al. (2019) classify CL approaches into three branches: regularization (Kirkpatrick et al., 2017; Aljundi et al., 2018) , replay (Shin et al., 2017) and expansion (Aljundi et al., 2017; Rusu et al., 2016; Yoon et al., 2018) methods.

Regularization and replay approaches fix the model architecture before training and prevent catastrophic forgetting by regularizing the change of a specific set of weights or replaying previously learned data.

Hybrids of replay and regularization also exist, such as Gradient Episodic Memory (GEM) (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2019a) .

On the other hand, methods based on expansion add new network components to learn new data.

Conceptually, such direction has the following advantages compared to the first two: (i) catastrophic forgetting can be eliminated since new information is not overwritten on pre-existing components and (ii) the model capacity is determined adaptively depending on the data.

Task-Free Continual Learning.

All the works mentioned above heavily rely on explicit task definition.

However, in real-world scenarios, task definition is rarely given at training time.

Moreover, the data domain may gradually shift without any clear task boundary.

Despite its importance, taskfree CL has been largely understudied; to the best of our knowledge, there are only a few works (Aljundi et al., 2019a; b; Rao et al., 2019) , each of which is respectively based on regularization, replay, and a hybrid of replay and expansion.

Specifically, Aljundi et al. (2019a) extend MAS (Aljundi et al., 2018) by adding heuristics to determine when to update the importance weights with no task definition.

In their following work (Aljundi et al., 2019b) , they improve the memory management algorithm of GEM (Lopez-Paz & Ranzato, 2017) such that the memory elements are carefully selected to minimize catastrophic forgetting.

While focused on unsupervised learning, Rao et al. (2019) is a parallel work that shares several similarities with our method, e.g., model expansion and short-term memory.

However, due to their model architecture, expansion is not enough to stop catastrophic forgetting; consequently, generative replay plays a crucial role in Rao et al. (2019) .

As such, it can be categorized as a hybrid of replay and expansion.

More detailed comparison between our method and Rao et al. (2019) is deferred to Appendix M.

We briefly review the Dirichlet process mixture (DPM) model (Antoniak, 1974; Ferguson, 1983) , and a variational method to approximate the posterior of DPM models in an online setting: Sequential Variational Approximation (SVA) (Lin, 2013) .

For a more detailed review, refer to Appendix A.

The DPM model is often applied to clustering problems where the number of clusters is not known in advance.

The generative process of a DPM model is

where x n is the n-th data, and θ n is the n-th latent variable sampled from G, which itself is a distribution sampled from a Dirichlet process (DP).

The DP is parameterized by a concentration parameter α and a base distribution G 0 .

The expected number of clusters is proportional to α, and G 0 is the marginal distribution of θ when G is marginalized out.

Since G is discrete with probability 1 (Teh, 2010) , same values can be sampled multiple times for θ.

If θ n = θ m , the two data points x n and x m belong to the same cluster.

An alternative formulation uses the variable z n that indicates to which cluster the n-th data belongs such that θ n = φ zn where φ k is the parameter of the k-th cluster.

In the context of this paper, φ k refers to the parameters of the k-th expert.

Approximation of the Posterior of DPM Models.

Since the exact inference of the posterior of DPM models is infeasible, approximate inference methods are applied.

Among many approximation methods, we adopt the Sequential Variational Approximation (SVA) (Lin, 2013) .

While the data is given one by one, SVA sequentially determines ρ n and ν k , which are the variational approximation for the distribution of z n and φ k respectively.

Since ρ n satisfies k ρ n,k = 1 and ρ n,k >= 0, ρ n,k can be interpreted as the probability of n-th data belonging to k-th cluster and is often called responsibility.

ρ n+1 and ν (n+1) at step n + 1 are computed as:

In practice, SVA adds a new component only when ρ K+1 is greater than a certain threshold .

If G 0 and p(x i |φ) are not a conjugate pair, stochastic gradient descent (SGD) is used to find the MAP estimationφ with a learning rate of λ instead of calculating the whole distribution ν k :

DPM for Discriminative Tasks.

DPM can be extended to discriminative tasks where each data point is an input-output pair (x, y), and the goal is to learn the conditional distribution p(y|x).

To use DPM, which is a generative model, for discriminative tasks, we first learn the joint distribution p(x, y) and induce the conditional distribution from it: p(y|x) = p(x, y)/ y p(x, y).

The joint distribution modeled by each component can be decomposed as p(x, y|z) = p(y|x, z)p(x|z) (Rasmussen & Ghahramani, 2002; Shahbaba & Neal, 2009 Finn et al., 2017) , they assume that similar tasks can be grouped into a super-task in which the parameter initialization is shared among tasks.

DPM is exploited to find the super-tasks and the parameter initialization for each super-task.

Therefore, it can be regarded as a meta-level CL method.

These works, however, lack generative components, which are often essential to infer the responsible component at test time, as will be described in the next section.

As a consequence, it is not straightforward to extend their algorithms to other CL settings beyond modelbased RL or meta-learning.

In contrast, our method implements a DPM model that is applicable to general task-free CL.

We aim at general task-free CL, where the number of tasks and task descriptions are not available at both training and test time.

We even consider the case where the data stream cannot be split into separate tasks in Appendix F. All of the existing expansion methods are not task-free since they require task definition at training (Aljundi et al., 2017) or even at test time (Rusu et al., 2016; Xu & Zhu, 2018; Li et al., 2019) .

We propose a novel expansion method that automatically determines when to expand and which component to use.

We first deal with generative tasks and generalize them into discriminative ones.

We can formulate a CL scenario as a stream of data involving different tasks D 1 , D 2 , ... where each task D k is a set of data sampled from a (possibly) distinct distribution p(x|z = k).

If K tasks are given so far, the overall distribution is expressed as the mixture distribution:

where

The goal of CL is to learn the mixture distribution in an online manner.

Regularization and replay methods directly model the approximate distribution p(x; φ) parameterized by a single component φ and update it to fit the overall distribution p(x).

When updating φ, however, they do not have full access to all the previous data, and thus the information of previous tasks is at risk of being lost as more tasks are learned.

Another way to solve CL is to use a mixture model: approximating each p(x|z = k) with p(x; φ k ).

If we learn a new task distribution p(x|z = K + 1) with new parameter φ K+1 and leave the existing parameters intact, we can preserve the knowledge of the previous tasks.

The expansion-based CL methods follow this idea.

Similarly, in the discriminative task, the goal of CL is to model the overall conditional distribution, which is a mixture of task-wise conditional distribution p(y|x, z = k):

Prior expansion methods use expert networks each of which models a task-wise conditional distribution p(y|x; φ k ) 1 .

However, a new problem arises in expansion methods: choosing the right expert given x, i.e., p(z|x) in Eq.(6).

Existing methods assume that explicit task descriptor z is given, which is generally not true in human-like learning scenarios.

That is, we need a gating mechanism that can infer p(z|x) only from x (i.e., which expert should process x).

With the gating, the model prediction naturally reduces to the sum of expert outputs weighted by the gate values, which is the mixture of experts (MoE) (Jacobs et al., 1991)

However, it is not possible to use a single gate network as in Shazeer et al. (2017) to model p(z|x) in CL; since the gate network is a classifier that finds the correct expert for a given data, training it in an online setting causes catastrophic forgetting.

Thus, one possible solution to replace a gating network is to couple each expert k with a generative model that represents p(x|z = k) as in Rasmussen & Ghahramani (2002) and Shahbaba & Neal (2009) .

As a result, we can build a gating mechanism without catastrophic forgetting as

where p(z = k) ≈ N k /N .

We also differentiate the notation for the parameters of discriminative models for classification and generative models for gating by the superscript D and G.

If we know the true assignment of z, which is the case of task-based CL, we can independently train a discriminative model (i.e., p(y|x; φ D k )) and a generative model (i.e., p(x; φ G k )) for each task k. In task-free CL, however, z is unknown, so the model needs to infer the posterior p(z|x, y).

Even worse, the total number of experts is unknown beforehand.

Therefore, we propose to employ a Bayesian nonparametric framework, specifically the Dirichlet process mixture (DPM) model, which can fit a mixture distribution with no prefixed number of components.

We use SVA described in , jointly representing p(x, y; φ k ).

We also keep the assigned data count N k per expert.

(a) During training, each sample (x, y) coming in a sequence is evaluated by every expert to calculate the responsibility ρ k of each expert.

If ρ K+1 is high enough, i.e., none of the existing experts is responsible, the data is stored into short-term memory (STM).

Otherwise, it is learned by the corresponding expert.

When STM is full, a new expert is created from the data in STM.

(b) Since CN-DPM is a generative model, we first compute the joint distribution p(x, y) for a given x, from which it is trivial to infer p(y|x).

section 2.2 to approximate the posterior in an online setting.

Although SVA is originally designed for the generative tasks, it is easily applicable to discriminative tasks by making each component k to model p(x, y|z) = p(y|x, z)p(x|z).

The proposed approach for task-free CL, named Continual Neural Dirichlet Process Mixture (CN-DPM) model, consists of a set of experts, each of which is associated with a discriminative model (classifier) and a generative model (density estimator).

More specifically, the classifier models p(y|x, z = k), for which we can adopt any classifier or regressor using deep neural networks, while the density estimator describes the marginal likelihood p(x|z = k), for which we can use any explicit density model such as VAEs (Kingma & Welling, 2014) and PixelRNN (Oord et al., 2016) .

We respectively denote the classifier and the density estimator of expert k as p(y|x; φ (7) by plugging in the output of the classifier and the density estimator.

Note that the number of experts is not prefixed but expanded via the DPM framework.

Figure 1 illustrates the overall training and inference process of our model.

Training.

We assume that samples sequentially arrive one at a time during training.

For a new sample, we first decide whether the sample should be assigned to an existing expert or a new expert should be created for it.

Suppose that samples up to (x n , y n ) are sequentially processed and K experts are already created when a new sample (x n+1 , y n+1 ) arrives.

We compute the responsibility ρ n+1,k as follows:

where G 0 is a distribution corresponding to the weight initialization.

If arg max k ρ n+1,k = K + 1, the sample is assigned to the existing experts proportional to ρ n+1,k , and the parameters of the experts are updated with the new sample by Eq.(4) such thatφ k is the MAP approximation given the data assigned up to the current time step.

Otherwise, we create a new expert.

Short-Term Memory.

However, it is not a good idea to create a new expert immediately and initialize it to be the MAP estimation given x n+1 .

Since both the classifier and density estimator of an expert are neural networks, training the new expert with only a single example leads to severe overfitting.

To mitigate this issue, we employ short-term memory (STM) to collect sufficient data before creating a new expert.

When a data point is classified as new, we store it to the STM.

Once the STM reaches its maximum capacity M , we stop the data inflow for a while and train a new expert with the data in the STM for multiple epochs until convergence.

We call this procedure sleep phase.

After sleep, the STM is emptied, and the newly trained expert is added to the expert pool.

During the subsequent wake phase, the expert is learned from the data assigned to it.

This STM trick assumes that the data in the STM belong to the same expert.

We empirically find that this assumption is acceptable in many CL settings where adjacent data are highly correlated.

The overall training procedure is described in Algorithm 1.

Note that we use ρ n,0 instead of ρ n,K+1 in the algorithm for brevity.

Inference.

At test time, we infer p(y|x) from the collaboration of the learned experts as in Eq.(7).

Techniques for Practicality.

Naively adding a new expert has two major problems: (i) the number of parameters grows unnecessarily large as the experts redundantly learn common features and (ii) there is no positive transfer of knowledge between experts.

Therefore, we propose a simple method to share parameters between experts.

When creating a new expert, we add lateral connections to the features of the previous experts similar to Rusu et al. (2016) .

To prevent catastrophic forgetting in the existing experts, we block the gradient from the new expert.

In this way, we can greatly reduce the number of parameters while allowing positive knowledge transfer.

More techniques such as sparse regularization in Yoon et al. (2018) can be employed to reduce redundant parameters further.

As they are orthogonal to our approach, we do not use such techniques in our experiments.

Another effective technique that we use in the classification experiments is adding a temperature parameter to the classifier.

Since the range of log p(x|z) is far broader than log p(y|x, z), the classifier has almost no effect without proper scaling.

Thus, we can increase overall accuracy by adjusting the relative importance of images and labels.

We also introduce an algorithm to prune redundant experts in Appendix D, and discuss further practical issues of CN-DPM in Appendix B.

Require:

if arg max k ρ n,k = 0 then

We evaluate the proposed CN-DPM model in task-free CL with four benchmark datasets.

Appendices include more detailed model architecture, additional experiments, and analyses.

A CL scenario defines a sequence of tasks where the data distribution for each task is assumed to be different from others.

Below we describe the task-free CL scenarios used in the experiments.

At both train and test time, the model cannot access the task information.

Unless stated otherwise, each task is presented for a single epoch (i.e., a completely online setting) with a batch size of 10. (Zenke et al., 2017) .

The MNIST dataset (LeCun et al., 1998 ) is split into five tasks, each containing approximately 12K images of two classes, namely (0/1, 2/3, 4/5, 6/7, 8/9).

We conduct both classification and generation experiments in this scenario. (Shin et al., 2017) .

It is a two-stage scenario where the first consists of MNIST, and the second contains SVHN (Netzer et al., 2011) .

This scenario is different from Split-MNIST; in Split-MNIST, new classes are introduced when transitioning into a new task, whereas the two stages in MNIST-SVHN share the same set of class labels and have different input domains.

Split-CIFAR10 and Split-CIFAR100.

In Split-CIFAR10, we split CIFAR10 (Krizhevsky & Hinton, 2009 ) into five tasks in the same manner as Split-MNIST.

For Split-CIFAR100, we build 20 tasks, each containing five classes according to the pre-defined superclasses in CIFAR100.

The training sets of CIFAR10 and CIFAR100 consist of 50K examples each.

Note that most of the previous works (Rebuffi et al., 2017; Zenke et al., 2017; Lopez-Paz & Ranzato, 2017; Aljundi et al., 2019c; Chaudhry et al., 2019a) , except Maltoni & Lomonaco (2019) , use task information at test time in Split-CIFAR100 experiments.

They assign distinct output heads for each task and utilize the task identity to choose the responsible output head at both training and test time.

Knowing the right output head, however, the task reduces to 5-way classification.

Therefore, our setting is far more difficult than the prior works since the model has to perform 100-way classification only from the given input.

All the following baselines use the same base network that will be discussed in section 4.3.

iid-offline and iid-online.

iid-offline shows the maximum performance achieved by combining standard training techniques such as data augmentation, learning rate decay, multiple iterations (up to 100 epochs), and larger batch size.

iid-online is the model trained with the same number of epoch and batch size with other CL baselines.

Fine-tune.

As a popular baseline in the previous works, the base model is naively trained as data enters.

Reservoir.

As Chaudhry et al. (2019b) show that simple experience replay (ER) can outperform most CL methods, we test the ER with reservoir sampling as a strong baseline.

Reservoir sampling randomly chooses a fixed number of samples with a uniform probability from an indefinitely long stream of data, and thus, it is suitable for managing the replay memory in task-free CL.

At each training step, the model is trained using a mini-batch from the data stream and another one of the same sizes from the memory.

Gradient-Based Sample Selection (GSS).

Aljundi et al. (2019b) propose a sampling method called GSS that diversifies the gradients of the samples in the replay memory.

Since it is designed to work in task-free settings, we report the scores in their paper for comparison.

Split-MNIST.

Following Hsu et al. (2018), we use a simple two-hidden-layer MLP classifier with ReLU activation as the base model for classification.

The dimension of each layer is 400.

For generation experiments, we use VAE, whose encoder and decoder have the same hidden layer configuration with the classifier.

Each expert in CN-DPM has a similar classifier and VAE with smaller hidden dimensions.

The first expert starts with 64 hidden units per layer and adds 16 units when a new expert is added.

For classification, we adjust hyperparameter α such that five experts are created.

For generation, we set α to produce 12 experts since more experts produce a better score.

We set the memory size in both Reservoir and CN-DPM to 500 for classification and 1000 for generation.

MNIST-SVHN and Split-CIFAR10/100.

We use ResNet-18 (He et al., 2016) as the base model.

In CN-DPM, we use a 10-layer ResNet for the classifier and a CNN-based VAE.

The encoder and the decoder of VAE have two CONV layers and two FC layers.

We set α such that 2, 5, and 20 experts are created for each scenario.

The memory sizes in Reservoir, GSS, and CN-DPM are set to 500 for MNIST-SVHN and 1000 for Split-CIFAR10/100.

More details can be found in Appendix C.

All reported numbers in our experiments are the average of 10 runs.

Table 1 and 2 show our main experimental results.

In every setting, CN-DPM outperforms the baselines by significant margins with reasonable parameter usage.

Table 2 and Figure 2 shows the results of Split-CIFAR10 experiments.

Since Aljundi et al. (2019b) test GSS using only 10K examples of CIFAR10, which is 1/5 of the whole train set, we follow their setting (denoted by 0.2 Epoch) for a fair comparison.

We also test a Split-CIFAR10 variant where each task is presented for 10 epochs.

The accuracy and the training graph of GSS are excerpted from the original paper, where the accuracy is the average of three runs, and the graph is from one of the runs.

In Figure 2 , the bold line represents the average of 10 runs (except GSS, which is a single run), and the faint lines are the individual runs.

Surprisingly, Reservoir even surpasses the accuracy of GSS and proves to be a simple but powerful CL method.

Table 2 is that the performance of Reservoir degrades as each task is extended up to 10 epochs.

This is due to the nature of replay methods; since the same samples are replayed repeatedly as representatives of the previous tasks, the model tends to be overfitted to the replay memory as training continues.

This degradation is more severe when the memory size is small, as presented in Appendix I. Our CN-DPM, on the other hand, uses the memory to buffer recent examples temporarily, so there is no such overfitting problem.

This is also confirmed by the CN-DPM's accuracy consistently increasing as learning progresses.

In addition, CN-DPM is particularly strong compared to other baselines when the number of tasks increases.

For example, Reservoir, which performs reasonably well in other tasks, scores poorly in Split-CIFAR100, which involves 20 tasks and 100 classes.

Even with the large replay memory of size 1000, the Reservoir suffers from the shortage of memory (e.g., only 50 slots per task).

In contrast, CN-DPM's accuracy is more than double of Reservoir and comparable to that of iid-online.

Table 3 analyzes the accuracy of CN-DPM in Split-CIFAR10/100.

We assess the performance and forgetting of individual components.

At the end of each task, we measure the test accuracy of the responsible classifier and report the average of such task-wise classifier accuracies as Classifier (init).

We report the average of the task-wise accuracies after learning all tasks as Classifier (final).

With little difference between the two scores, we confirm that forgetting barely occurs in the classifiers.

In addition, we report the gating accuracy measured after training as Gating (VAEs), which is the accuracy of the task identification performed jointly by the VAEs.

The relatively low gating accuracy suggests that CN-DPM has much room for improvement through better density estimates.

Overall, CN-DPM does not suffer from catastrophic forgetting, which is a major problem in regularization and replay methods.

As a trade-off, however, choosing the right expert arises as another problem in CN-DPM.

Nonetheless, the results show that this new direction is especially promising when the number of tasks is very large.

In this work, we formulated expansion-based task-free CL as learning of a Dirichlet process mixture model with neural experts.

We demonstrated that the proposed CN-DPM model achieves great performance in multiple task-free settings, better than the existing methods.

We believe there are several interesting research directions beyond this work: (i) improving the accuracy of expert selection, which is the main bottleneck of our method, and (ii) applying our method to different domains such as natural language processing and reinforcement learning.

We review the Dirichlet process mixture (DPM) model and a variational method to approximate the posterior of DPM models in an online setting: Sequential Variational Approximation (SVA) (Lin, 2013) .

Dirichlet Process.

Dirichlet process (DP) is a distribution over distributions that are defined over infinitely many dimensions.

DP is parameterized by a concentration parameter α ∈ R + and a base distribution G 0 .

For a distribution G sampled from DP(α, G 0 ), the following holds for any finite measurable partition {A 1 , A 2 , ..., A K } of probability space Θ (Teh, 2010):

The stick-breaking process is often used as a more intuitive construction of DP:

Initially, we start with a stick of length one, which represents the total probability.

At each step k, we cut a proportion v k off from the remaining stick (probability) and assign it to the atom φ k sampled from the base distribution G 0 .

This formulation shows DP is discrete with probability 1 (Teh, 2010).

In our problem setting, G is a distribution over expert's parameter space and has positive probability only at the countably many φ k , which are independently sampled from the base distribution.

Dirichlet Process Mixture (DPM) Model.

The DPM model is often applied to clustering problems where the number of clusters is not known in advance.

The generative process of DPM model is

where x n is the n-th data, and θ n is the n-th latent variable sampled from G, which itself is a distribution sampled from a Dirichlet process (DP).

Since G is discrete with probability 1, the same values can be sampled multiple times for θ.

If θ n = θ m , the two data points x n and x m belong to the same cluster.

An alternative formulation uses the indicator variable z n that indicates to which cluster the n-th data belongs such that θ n = φ zn where φ k is the parameter of k-th cluster.

The data x n is sampled from a distribution parameterized by θ n .

For a DP Gaussian mixture model as an example, each θ = {µ, σ 2 } parameterizes a Gaussian distribution.

The Posterior of DPM Models.

The posterior of a DPM model for given θ 1 , ..., θ n is also a DP (Teh, 2010):

The base distribution of the posterior, which is a weighted average of G 0 and the empirical distribution

, is in fact the predictive distribution of θ n+1 given θ 1:n (Teh, 2010):

If we additionally condition x n and reflect the likelihood, we obtain (Neal, 2000):

where Z is the normalizing constant.

Note that θ n+1 is independent from x 1:n given θ 1:n .

Approximation of the Posterior of DPM Models.

Since the exact inference of the posterior of DPM models is infeasible, approximate inference methods are adopted such as Markov chain Monte Carlo (MCMC) (Maceachern, 1994; Escobar & West, 1995; Neal, 2000) or variational inference (Blei & Jordan, 2006; Wang & Dunson, 2011; Lin, 2013) .

Among many variational methods, the Sequential Variational Approximation (SVA) (Lin, 2013) approximates the posterior as

where p(z 1:n |x 1:n ) is represented by the product of individual variational probabilities ρ zi for z i , which greatly simplifies the distribution.

Moreover, p(G|x 1:n , z 1:n ) is approximated by a stochastic process q (z) ν (G|z 1:n ).

Sampling from q (z) ν (G|z 1:n ) is equivalent to constructing a distribution as

K } is the partition of x 1:n characterized by z. The approximation yields the following tractable predictive distribution:

SVA uses this predictive distribution for sequential approximation of the posterior of z and φ.

While the data is given one by one, SVA sequentially updates the variational parameters; the following ρ n+1 and ν (n+1) at step n + 1 minimizes the KL divergence between q(z n+1 , φ (n+1) |ρ 1:n+1 , ν (n+1) ) and the posterior:

In practice, SVA adds a new component only when ρ n+1,K+1 is greater than a threshold .

It uses stochastic gradient descent to find and maintain the MAP estimation of parameters instead of calculating the whole distribution ν k :

where

k is a learning rate of component k at step n, which decreases as in the Robbins-Monro algorithm.

CN-DPN is designed based on strong theoretical foundations, including the nonparametric Bayesian framework.

In this section, we further discuss some practical issues of CN-DPM with intuitive explanations.

Bounded expansion of CN-DPM.

The number of components in the DPM model is determined by the data distribution and the concentration parameter.

If the true distribution consists of K clusters, the number of effective components converges to K under an appropriate concentration parameter α.

Typically, the number of components is bounded by O(α log N ) (Teh, 2010) .

Experiments in Appendix H empirically show that CN-DPM does not blindly increase the number of experts.

The continued increase in model capacity.

Our model capacity keeps increasing as it learns new tasks.

However, we believe this is one of the strengths of our method, since it may not make sense to use a fixed-capacity neural network to learn an indefinitely long sequence of tasks.

The underlying assumption of using a fixed-capacity model is that the pre-set model capacity is adequate (at least not insufficient) to learn the incoming tasks.

On the other hand, CN-DPM approaches the problem in a different direction: start small and add more as needed.

This property is essential in task-free settings where the total number of tasks is not known.

If there are too many tasks than expected, a fixed-capacity model would not be able to learn them successfully.

Conversely, if there are fewer tasks than expected, resources would be wasted.

We argue that expansion is a promising direction since it does not need to fix the model capacity beforehand.

Moreover, we also introduce an algorithm to prune redundant experts in Appendix D, Generality of the concentration parameter.

The concentration parameter controls how sensitive the model is to new data.

In other words, it determines the level of discrepancy between tasks, that makes the tasks modeled by distinct components.

As an example, suppose that we are designing a hand-written alphabet classifier that continually learns in the real world.

In the development, we only have the character images for half of the alphabets, i.e., from 'a' to 'm'.

If we can find a good concentration parameter α for the data from 'a' to 'm', the same α can work well with novel alphabets (i.e., from 'n' to 'z') because the alphabets would have a similar level of discrepancies between tasks.

Therefore, we do not need to access the whole data to determine α if the discrepancy between tasks is steady.

We use ResNet-18 (He et al., 2016) .

The input images are transformed to 32×32 RGB images.

For the classifiers in experts, we use a smaller version of the base MLP classifier.

In the first expert, we set the number of hidden units per layer to 64.

In the second or later experts, we introduce 16 new units per layer which are connected to the lower layers of the existing experts.

For the encoder and decoder of VAEs, we use a two-layer MLP.

The encoder is expanded in the same manner as the classifier.

However, we do not share the parameters beyond the encoders; with a latent code of dimension 16, we use the two-hidden-layer MLP decoder as done in the classifier.

For generation tasks, we double the size; for example, we set the size of initial and additional hidden units to 128 and 32, respectively.

The ResNet-18 base network has eight residual blocks.

After passing through 2 residual blocks, the width and height of the feature are halved, and the number of channels is doubled.

The initial number of channels is set to 64.

For the classifiers in CN-DPM, we use a smaller version of ResNet that has only four residual blocks and resizes the feature every block.

The initial number of channels is set to 20 in the first expert, and four initial channels are added with a new expert.

Thus, 4, 8, 16 , and 32 channels are added for the four blocks.

The first layer of each block is connected to the last layer of the previous block of prior experts.

For the VAEs, we use a simple CNN-based VAEs.

The encoder has two 3×3 convolutions followed by two fully connected layers.

Each convolution is followed by 2×2 max-pool and ReLU activation.

The numbers of channels and hidden units are doubled after each layer.

In the first expert, the first convolution outputs 32 channels, while four new channels are added with each new expert.

As done for the VAE in Split-MNIST, each expert's VAE has an unshared decoder with a 64-dimensional latent code.

The decoder is the mirrored encoder where 3×3 convolution is replaced by 4×4 transposed convolution with a stride of 2.

For the classifier, we use ResNet-18 with 32 channels for the first expert and additional 32 channels for each new expert.

We use the same VAE as in Split-CIFAR10.

We use the classifier temperature parameter of 0.01 for Split-MNIST, Split-CIFAR10/100, and no temperature parameter on MNIST-SVHN.

Weight decay 0.00001 has been used for every model in the paper.

Gradients are clipped by value with a threshold of 0.5.

All the CN-DPM models are trained by Adam optimizer.

During the sleep phase, we train the new expert for multiple epochs with a batch size of 50.

In classification tasks, we improve the density estimation of VAEs by sampling 16 latent codes and averaging the ELBOs, following Burda et al. (2015) .

The learning rate of 0.0001 and 0.0004 has been used for the classifier and VAE of each expert in the classification task.

We use learning rate 0.003 for the VAE of each expert in generation task.

In the generation task, we decay the learning rate of the expert by 0.003 before it enters the wake phase.

Following the existing works in VAE literature, we use binarized MNIST for the generation experiments.

VAEs are trained to maximize Bernoulli log-likelihood in the generation task, while Gaussian log-likelihood is used for the classification task.

The learning rate of 0.005 and 0.0002 has been used for the classifier and VAE of each expert in CIFAR10.

We decay the learning rate of the expert by 0.1 before it enters the wake phase.

VAEs are trained to maximize Gaussian log-likelihood.

The learning rate of 0.0002 and 0.0001 has been used for the classifier and VAE of each expert in CIFAR10.

We decay the learning rate of the expert by 0.2 before it enters the wake phase.

VAEs are trained to maximize Gaussian log-likelihood.

The learning rate of 0.0001 and 0.0003 has been used for the classifier and VAE of each expert in CIFAR10.

We decay the learning rates of classifier and VAE of each expert by 0.5 and 0.1 before it enters the wake phase.

VAEs are trained to maximize Gaussian log-likelihood.

Lin (2013) propose a simple algorithm to prune and merge redundant components in DPM models.

Following the basic principle of the algorithm, we provide a pruning algorithm for CN-DPM.

First, we need to measure the similarities between experts to choose which expert to prune.

We compute the log-likelihood l nk = p(x n , y n |φ k ) of each expert k for data (x 1:N , y 1:N ).

As a result, we can obtain K vectors with N dimensions.

We define the similarity s(k, k ) between two experts k and k as the cosine similarity between the two corresponding vectors l ·k and l ·k , i.e., s(k, k ) = l ·k ·l ·k |l ·k ||l ·k | .

If the similarity is greater than a certain threshold , we remove one of the experts with smaller N k = n ρ n,k .

The N k data of the removed expert are added to the remaining experts.

Figure 4 shows an example of an expert pruning.

We test CN-DPM on Split-MNIST with an α higher than the optimal value such that more than five experts are created.

In this case, seven experts are created.

If we build a similarity matrix as shown in Figure 4b , we can see which pair of experts are similar.

We then threshold the matrix at 0.9 in Figure 4c and choose expert pairs (2/3) and (5/6) for pruning.

Comparing N k within each pair, we can finally choose to prune expert 3 and 6.

After pruning, the test accuracy marginally drops from 87.07% to 86.01%.

Table 4 compares our method with task-based methods for Split-MNIST classification.

All the numbers except for our CN-DPM are excerpted from Hsu et al. (2018) , in which all methods are trained for four epochs per task with a batch size of 128.

Our method is trained for four epochs per task with a batch size of 10.

The model architecture used in compared methods is the same as our baselines: a two-hidden-layer MLP with 400 hidden units per layer.

All compared methods use a single output head, and the task information is given at training time but not at test time.

For CN-DPM, we test two training settings where the first one uses task information to select experts, while the second one infers the responsible expert by the DPM principle.

Task information is not given at test time in both cases.

Notice that regularization methods often suffer from catastrophic forgetting while replay methods yield decent accuracies.

Even though the task-free condition is a far more difficult setting, the performance of our method is significantly better than regularization and replay methods that exploit the task description.

If task information is available at train time, we can utilize it to improve the performance even more. (Schwarz et al., 2018) 19.77 ± 0.04 SI (Zenke et al., 2017) 19.67 ± 0.09 MAS (Aljundi et al., 2018) 19.52 ± 0.04 LwF (Li & Hoiem, 2017) 24.17 ± 0.33

Replay GEM (Lopez-Paz & Ranzato, 2017) 92.20 ± 0.12 DGR (Shin et al., 2017) 91.24 ± 0.33 RtF (van de Ven & Tolias, 2018) 92.

In addition, we experiment with the case where the task boundaries are not clearly defined, which we call Fuzzy-Split-MNIST.

Instead of discrete task boundaries, we have transition stages between tasks where the data of existing and new tasks are mixed, but the proportion of the new task linearly increases.

This condition adds another level of difficulty since it makes the methods unable to rely on clear task boundaries.

The scenario is visualized in Figure 5 .

As shown in Table 5 , CN-DPM can perform continual learning without task boundaries.

Even in discriminative tasks where the goal is to model p(y|x), CN-DPM learns the joint distribution p(x, y).

Since CN-DPM is a complete generative model, it can generate (x, y) pairs.

To generate a sample, we first sample z from p(z) which is modeled by the categorical distribution Cat(

, choose an expert.

Given z = k, we first sample x from the generator p(x; φ G k ), and then sample y from the discriminator p(y|x; φ D k ).

Figure 6 presents 50 sample examples generated from a CN-DPM trained on Split-MNIST for a single epoch.

We observe that CN-DPM successfully generates examples of all tasks with no catastrophic forgetting.

We present experiments with much longer continual learning scenarios on Split-MNIST, Split-CIFAR10 and Split-CIFAR100 in Table 6 , 7 and 8, respectively.

We report the average of 10 runs with ± standard error of the mean.

To compare with the default 1-epoch scenario, we carry out experiments that repeat each task 10 times, which are denoted 10 Epochs.

In addition, we also present the results of repeating the whole scenario 10 times, which are denoted 1 Epoch ×10.

For example, in Split-MNIST, the 10 Epochs scenario consists of 10-epoch 0/1, 10-epoch 2/3, ..., 10-epoch 8/9 tasks.

On the other hand, the 1 Epoch ×10 scenario revisits each task multiple times, i.e., 1-epoch 0/1, 1-epoch 2/3, ..., 1-epoch 8/9, 1-epoch 0/1, ..., 1-epoch 8/9.

We use the same hyperparameters tuned for the 1-epoch scenario.

We find that the accuracy of Reservoir drops as the length of each task increases.

As mentioned in the main text, this phenomenon seems to be caused by overfitting on the samples in the replay memory.

Since only a small number of examples in the memory represent each task, replaying them for a long period degrades the performance.

On the other hand, the performance of our CN-DPM improves as the learning process is extended.

In the 1 Epoch ×10 setting, CN-DPM shows similar performance with 10 Epoch since the model sees each data point 10 times in both scenarios.

On the other hand, Reservoir's scores in the 1 Epoch ×10 largely increase compared to both 1 Epoch and 10 Epoch This difference can be explained by how the replay memory changes while training progresses.

In the 10 Epoch setting, if a task is finished, it is not visited again.

Therefore, the examples of the task in the replay memory monotonically decreases, and the remaining examples are replayed repeatedly.

As the training progresses, the model is overfitted to the old examples in the memory and fails to generalize in the old tasks.

In contrast, in 1 Epoch ×10 setting, each task is revisited multiple times, and each time a task is revisited, the replay memory is also updated with the new examples of the task.

Therefore, the overfitting problem in the old tasks is greatly relieved.

Another important remark is that CN-DPM does not blindly increase the number of experts.

If we add a new expert at every constant steps, we would have 10 times more experts in the longer scenarios.

However, this is not the case.

CN-DPM determines whether it needs a new expert on a data-by-data basis such that the number of experts is determined by the task distribution, not by the length of training.

Aljundi et al. (2019b) .

Table 9 compares the experimental results with different memory sizes of 500 and 1000 on Split-CIFAR10/100.

Compared to Reservoir, whose performance drops significantly with smaller memory, CN-DPM's accuracy drop is relatively marginal.

Table 10 shows the results of CN-DPM on Split-MNIST classification according to the concentration parameter α, which defines the prior of how sensitive CN-DPM is to new data.

With a higher α, an expert tends to be created more easily.

In the experiment reported in the prior sections, we set log α = −400.

At log α = −600, too few experts are created, and the accuracy is rather low.

As α increases, the number of experts grows along with the accuracy.

Although the CN-DPM model is task-free and automatically decides the task assignments to experts, we still need to tune the concentration parameter to find the best balance point between performance and model capacity, as all Bayesian nonparametric models require.

K THE EFFECT OF PARAMETER SHARING Table 11 compares when the parameters are shared between experts and when they are not shared.

By sharing the parameters, we could reduce the number of parameters by approximately 38% without sacrificing accuracy.

L TRAINING GRAPHS Figure 8 shows the training graphs of our experiments.

In addition to the performance metrics, we present the number of experts in CN-DPM and compare the total number of parameters with the baselines.

The bold lines represent the average of the 10 runs while the faint lines represent individual runs.

Figure 9 and Figure 10 show how the accuracy of each task changes during training.

We also present the average accuracy of learned tasks at the bottom right.

Continual Unsupervised Representation Learning (CURL) (Rao et al., 2019 ) is a parallel work that shares some characteristics with our CN-DPM in terms of model expansion and short-term memory.

However, there are several key differences that distinguish our method from CURL, which will be elaborated in this section.

Following the notations of Rao et al. (2019), here y denotes the cluster assignment, and z denotes the latent variable.

1.

The Generative Process.

The primary goal of CURL is to continually learn a unified latent representation z, which is shared across all tasks.

Therefore, the generative model of CURL explicitly consists of the latent variable z as summarized as follows:

p(x, y, z) = p(y)p(z|y)p(x|z) where y ∼ Cat(π), z ∼ N (µ z (y), σ 2 z (y)), x ∼ Bernoulli(µ x (z)).

The overall distribution of z is the mixture of Gaussians, and z includes the information of y such that x and y are conditionally independent given z. Then, z is fed into a single decoder network µ x to generate the mean of x, which is modeled by a Bernoulli distribution.

On the other hand, the generative version of CN-DPM, which does not include classifiers, has a simpler generative process:

p(x, y) = p(y)p(x|y) where y ∼ Cat(π), x ∼ p(x|y).

The choice of p(x|y) here is not necessarily restricted to VAEs (Kingma & Welling, 2014) ; one may use other kinds of explicit density models such as PixelRNN (Oord et al., 2016) .

Even if we use VAEs to model p(x|y), the generative process is different from CURL:

p(x, y, z) = p(y)p(z)p(x|y, z) where y ∼ Cat(π), z ∼ N (0, I), x ∼ Bernoulli(µ y x (z)).

Unlike CURL, CN-DPM generates y and z independently and maintains a separate decoder µ y x for each cluster y.

2.

The Necessity for Generative Replay in CURL.

CURL periodically saves a copy of its parameters and use it to generate samples of learned distribution.

The generated samples are played together with new data such that the main model does not forget previously learned knowledge.

This process is called generative replay.

The generative replay is an essential element in CURL, unlike our CN-DPM.

CURL assumes a factorized variational posterior q(y, z|x) = q(y|x)q(z|x, y) where q(y|x) and q(z|x, y) are modeled by separate output heads of the encoder neural network.

However, the output head for q(y|x) is basically a gating network that could be vulnerable to catastrophic forgetting, as mentioned in Section 3.1.

Moreover, CURL shares a single decoder µ x across all tasks.

As a consequence, expansion alone is not enough to stop catastrophic forgetting, and CURL needs another CL method to prevent catastrophic forgetting in the shared components.

This is the main reason why the generative replay is crucial in CURL.

As shown in the ablation test of Rao et al. (2019) , the performance of CURL drops without the generative replay.

In contrast, the components of CN-DPM are separated for each task (although they may share low-level representations) such that no additional treatment is needed.

T1  T2  T3  T4  T5  0  20  40  60  80  100   Task 1   T1  T2  T3  T4  T5  0  20  40  60  80  100   Task 2   T1  T2  T3  T4  T5  0  20  40  60  80  100   Task 3   T1  T2  T3  T4  T5  0  20  40  60  80  100   Task 4   T1  T2  T3  T4  T5  0  20

40  60  80  100   Task 5   T1  T2  T3  T4  T5  0 T5  T10  T15  T20  0  20  40  60   Task 1   T5  T10  T15  T20  0  20  40  60   Task 2   T5  T10  T15  T20  0  20  40  60   Task 3   T5  T10  T15  T20  0  20  40  60   Task 4   T5  T10  T15  T20  0  20  40  60   Task 5   T5  T10  T15  T20  0  20  40  60   Task 6   T5  T10  T15  T20  0  20  40  60   Task 7   T5  T10  T15  T20  0  20  40  60   Task 8   T5  T10  T15  T20  0  20  40  60   Task 9   T5  T10  T15  T20  0  20  40  60   Task 10   T5  T10  T15  T20  0  20  40  60   Task 11   T5  T10  T15  T20  0  20  40  60   Task 12   T5  T10  T15  T20  0  20  40  60   Task 13   T5  T10  T15  T20  0  20  40  60   Task 14   T5  T10  T15  T20  0  20  40  60   Task 15   T5  T10  T15  T20  0  20  40  60   Task 16   T5  T10  T15  T20  0  20  40  60   Task 17   T5  T10  T15  T20  0  20  40  60   Task 18   T5  T10  T15  T20  0  20  40  60   Task 19   T5  T10  T15  T20  0  20  40  60   Task 20   T5  T10  T15  T20  0  20

40  60  Average  Fine-

<|TLDR|>

@highlight

We propose an expansion-based approach for task-free continual learning for the first time. Our model consists of a set of neural network experts and expands the number of experts under the Bayesian nonparametric principle.