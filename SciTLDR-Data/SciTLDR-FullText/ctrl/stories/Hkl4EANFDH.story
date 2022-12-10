Regularization-based continual learning approaches generally prevent catastrophic forgetting by augmenting the training loss with an auxiliary objective.

However in most practical optimization scenarios with noisy data and/or gradients, it is possible that stochastic gradient descent can inadvertently change critical parameters.

In this paper, we argue for the importance of regularizing optimization trajectories directly.

We derive a new co-natural gradient update rule for continual learning whereby the new task gradients are preconditioned with the empirical Fisher information of previously learnt tasks.

We show that using the co-natural gradient systematically reduces forgetting in continual learning.

Moreover, it helps combat overfitting when learning a new task in a low resource scenario.

It is good to have an end to journey toward; but it is the journey that matters, in the end.

Endowing machine learning models with the capability to learn a variety of tasks in a sequential manner is critical to obtain agents that are both versatile and persistent.

However, continual learning of multiple tasks is hampered by catastrophic forgetting (McCloskey & Cohen, 1989; Ratcliff, 1990) , the tendency of previously acquired knowledge to be overwritten when learning a new task.

Techniques to mitigate catastrophic forgetting can be roughly categorized into 3 lines of work (see Parisi et al. (2019) for a comprehensive overview): 1.

regularization-based approaches, where forgetting is mitigated by the addition of a penalty term in the learning objective (Kirkpatrick et al. (2017) ; Chaudhry et al. (2018a) , inter alia), 2.

dynamic architectures approaches, which incrementally increase the model's capacity to accomodate the new tasks (Rusu et al., 2016) , and 3.

memorybased approaches, which retain data from learned tasks for later reuse (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018b; .

Among these, regularization-based approaches are particularly appealing because they do not increase the model size and do not require access to past data.

This is particularly relevant to real-world scenarios where keeping data from previous training tasks may be impractical because of infrastructural or privacy-related reasons.

Moreover, they are of independent intellectual interest because of their biological inspiration rooted in the idea of synaptic consolidation (Kirkpatrick et al., 2017) .

A good regularizer ensures that, when learning a new task, gradient descent will ultimately converge to parameters that yield good results on the new task while preserving performance on previously learned tasks.

Critically, this is predicated upon successful optimization of the regularized objective, a fact that has been largely taken for granted in previous work.

Non-convexity of the loss function, along with noise in the data (due to small or biased datasets) or in the gradients (due to stochastic gradient descent), can yield optimization trajectories -and ultimately convergence points -that are highly non-deterministic, even for the same starting parameters.

As we demonstrate in this paper, this can cause unintended catastrophic forgetting along the optimization path.

This is illustrated in a toy setting in Figure 1 : a two parameter model is trained to perform task T 2 (an arbitrary bi-modal loss function) after having learned task T 1 (a logistic regression task).

Standard finetuning, even in

T 2 Finetuning EWC Co-natural finetuning Figure 1 : On the importance of trajectories: an example with 2-dimensional logistic regression.

Having learned task T 1 , the model is trained on T 2 with two different objectives: minimizing the loss on T 2 (Finetuning) and a regularized objective (EWC; Kirkpatrick et al. (2017) ).

We add a small amount of Gaussian noise to gradients in order to simulate the stochasticity of the trajectory.

Plain finetuning and EWC often converge to a solution with high loss for T 1 , but the co-natural optimization trajectory consistently converges towards the optimum with lowest loss for T 1 .

the presence of a regularized objective (EWC; Kirkpatrick et al. (2017) ), quickly changes the loss of T 1 and tends converge to a solution with high T 1 loss.

We propose to remedy this issue by regularizing the optimization trajectory itself, specifically by preconditioning gradient descent with the empirical Fisher information of previously learned tasks ( §3).

This yields what we refer to as a co-natural gradient, an update rule inspired by the natural gradient (Amari, 1997), but taking the Fisher information of previous tasks as a natural Riemannian metric 2 of the parameter space, instead of the Fisher information of the task being optimized for.

When we introduce our proposed co-natural gradient for the toy example of Figure 1 , the learning trajectory follows a path that changes the loss on T 1 much more slowly, and tends to converges to the optimum that incurs the lowest performance degradation on T 1 .

We test the validity of our approach in a continual learning scenario ( §4).

We show that the co-natural gradient consistently reduces forgetting in a variety of existing continual learning approaches by a factor of ≈ 1.5 to 9, and greatly improves performance over simple finetuning, without modification to the training objective.

We further investigate the special case of transfer learning in a two-task, low-resource scenario.

In this specific case, control over the optimization trajectory is particularly useful because the optimizer has to rely on early stopping to prevent overfitting to the meager amount of training data in the target task.

We show that the co-natural gradient yields the best trade-offs between source and target domain performance over a variety of hyper-parameters ( §5).

We first give a brief overview of the continual learning paradigm and existing approaches for overcoming catastrophic forgetting.

Let us define a task as a triplet containing an input space X and an output space Y, both measurable spaces, as well as a distribution D over X × Y. In general, learning a task will consist of training a model to approximate the conditional distribution p(y | x) induced by D.

Consider a probabilistic model p θ parametrized by θ ∈ R d where d is the size of the model, trained to perform a source task S = X S , Y S , D S to some level of performance, yielding parameters θ S .

In the most simple instance of continual learning, we are tasked with learning a second target task T = X T , Y T , D T .

In general in a multitask setting, it is not the case that the input or output spaces are the same.

The discrepancy between input/output space can be addressed in various ways, e.g. by adding a minimal number of task-specific parameters (for example, different softmax layers for different label sets).

To simplify exposition, we set these more specific considerations aside for now, and assume that X S = X T and Y S = Y T .

At any given point during training for task T , our objective will be to minimize the loss function L T (θ) -generally the expected log-likelihood E x,y∼D T [− log p θ (y | x)].

Typically, this will be performed by iteratively adding incremental update vectors δ ∈ R d to the parameters θ ←− θ + δ.

In this paper, we focus on those models that have a fixed architecture over the course of continual learning.

The study of continual learning for models of fixed capacity can be split into two distinct (but often overlapping) streams of work:

Regularization-based approaches introduce a penalty in the loss function L T , typically quadratic, pushing the weights θ back towards θ S :

where Ω S is a matrix, typically diagonal, that encodes the respective importance of each parameter with respect to task S, and λ is a regularization strength hyper-parameter.

Various choices have been proposed for Ω S ; the diagonal empirical Fisher information matrix (Kirkpatrick et al., 2017) , or pathintegral based importance measures (Zenke et al., 2017; Chaudhry et al., 2018a) .

More elaborate regularizers have been proposed based on e.g. a Bayesian formulation of continual learning (Nguyen et al., 2017; Ahn et al., 2019) or a distillation term (Li & Hoiem, 2016; Dhar et al., 2019) .

The main advantage of these approaches is that they do not rely on having access to training data of previous tasks.

Memory-based approaches store data from previously seen tasks for re-use in continued learning, either as a form of constraint, by e.g. ensuring that training on the new task doesn't increase the loss on previous tasks (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018b) , or for replay i.e. by retraining on instances from previous tasks (Rebuffi et al., 2017; Chaudhry et al., 2019; Aljundi et al., 2019b; a) .

Various techniques have been proposed for the selection of samples to store in the memory (Chaudhry et al., 2019; Aljundi et al., 2019b) or for retrieval of the samples to be used for replay Aljundi et al. (2019a) .

All of these methods rely on stochastic gradient descent to optimize their regularized objective or to perform experience replay, with the notable exception of GEM (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018b) , where the gradients are projected onto the orthogonal complement of previous task's gradients.

However, this method has been shown to perform poorly in comparison with simple replay (Chaudhry et al., 2019) , and it still necessitates access to data from previous tasks.

After briefly recalling how the usual update is obtained in gradient descent, we derive a new, conatural update designed to better preserve the distribution induced by the model over previous tasks.

At point θ in the parameter space, gradient descent finds the optimal update δ that is (1) small and (2) locally minimizes the decrease in loss

Traditionally this can be formulated as minimizing the Lagrangian:

with Lagrangian multiplier µ > 0.

Minimizing L for δ yields the well-known optimal update δ * :

1 2µ corresponds to the learning rate (see Appendix A.1 for the full derivation).

The δ 2 term in L implicitly expresses the underlying assumption that the best measure of distance between parameters θ and θ + δ is the Euclidean distance.

In a continual learning setting however, the quantity we are most interested in preserving is the probability distribution that θ models on the source task S:

Therefore, a more natural distance between θ and θ + δ is the Kullback-Leibler divergence KL(p S θ p S θ+δ ) (Kullback & Leibler, 1951) .

For preventing catastrophic forgetting along the optimization path, we incorporate incorporate this KL term into the Lagrangian L itself:

Doing so means that the optimization trajectory will tend to follow the direction that changes the distribution of the model the least.

Notably, this is not a function of the previous objective L S , so knowledge of the original training objective is not necessary during continual learning (which is typically the case in path-integral based regularization methods (Zenke et al., 2017) or experience replay (Chaudhry et al., 2019) ).

Presuming that δ is small, we can perform a second order Taylor approximation of the function δ → KL(p

where F S θ is the Hessian of the KL divergence around θ.

A crucial, well-known property of this matrix is that it coincides with the Fisher information matrix

(the expectation being taken over the model's distribution p θ ; see Appendix A.1 for details).

This is appealing from a computational perspective because the Fisher can be computed by means of first order derivatives only.

Minimizing for δ yields the following optimal update:

where coefficients µ and ν are folded into two hyper-parameters: the learning rate λ and a damping coefficient α (the step-by-step derivation can be found in Appendix A.1).

In practice, especially with low damping coefficients, it is common to obtain updates that are too large (typically when some parameters have no effect on the KL divergence).

To address this, we re-normalize δ * to have the same norm as the original gradient, ∇L T .

For computational reasons, we will make 3 key practical approximations to the Fisher:

: we maintain the Fisher computed at θ S , instead of recomputing F S at every step of training.

This relieves us of the computational burden of updating the Fisher for every new value of θ.

This approximation (shared by previous work, e.g. Kirkpatrick et al. (2017) ; Chaudhry et al. (2018a) ) is only valid insofar as θ S and θ are close.

Empirically we observe that this still leads to good results.

S is diagonal: this is a common approximation in practice with two appealing properties.

First, this makes it possible to store the d diagonal Fisher coefficients in memory.

Second, this trivializes the inverse operation (simply invert the diagonal elements).

this common approximation replaces the expectation under the model's distribution by the expected log-likelihood of the true distribution:

T ] (mind the subscript).

This is particularly useful in tasks with a large or unbounded number of classes (e.g. structured prediction), where summing over all possible outputs is intractable.

We can then compute the diagonal of the empirical Fisher using Monte Carlo sampling:

2 with (x i , y i ) sampled from D S (we use N = 1000 for all experiments).

This formulation bears many similarities with the natural gradient from Amari (1997), which also uses the KL divergence as a metric for choosing the optimal update δ * .

There is a however a crucial difference, both in execution and purpose: where the natural gradient uses knowledge of the curvature of the KL divergence of D T to speed-up convergence, our proposed method leverages the curvature of the KL divergence on D S to slow-down divergence from p S θ S .

To highlight the resemblance and complementarity between these two concepts, we refer to the new update as the co-natural gradient.

In a continual learning scenario, we are confronted with a large number of tasks T 1 . . .

T n presented in sequential order.

When learning T n , we can change the Lagrangian L from 5 to incorporate the constraints for all previous tasks T 1 . . .

T n−1 :

This in turn changes the Fisher in Eq. 8 toF n−1 :=

The choice of the coefficients ν i is crucial.

Setting all ν i to the same value, i.e. assigning the same importance to all tasks is suboptimal for a few reasons.

First and foremost, it is unreasonable to expect of a model with finite capacity to remember an unbounded number of tasks (as tasks "fill-up" the model capacity, F n−1 is likely to become more "homogeneous").

Second, as training progresses and θ changes, our approximation that

is less and less likely to hold.

We address this issue in the same fashion as Schwarz et al. (2018) , by keeping a rolling exponential average of the Fisher matrices:

In this case, previous tasks are gracefully forgotten at an exponential rate controlled by γ.

We account for the damping α term in Eq. 7 by settingF 0 := α γ I. In preliminary experiments, we have found γ = 0.9 to yield consistently good results, and use this value in all presented experiments.

To corroborate our hypothesis that controlling the optimization trajectory with the co-natural gradient reduces catastrophic forgetting, we perform experiments on two continual learning testbeds:

• Split CIFAR: The CIFAR100 dataset, split into 20 independent 5-way classification tasks.

Similarly to Chaudhry et al. (2018b) , we use a smaller version of the ResNet architecture (He et al., 2016) .

• Omniglot: the Omniglot dataset (Lake et al., 2015) consists of 50 independent character recognition datasets on different alphabet.

We adopt the setting of Schwarz et al. (2018) and consider each alphabet as a separate task.

4 On this dataset we use the same small CNN architecture as Schwarz et al. (2018) .

• Split MiniImageNet: The MiniImageNet dataset (a subset of the popular ImageNet (Deng et al., 2009 ) dataset 5 ; Vinyals et al. (2016) ).

Split the dataset into 20 independent 5-way classification tasks, similarly to Split CIFAR, and use the same smaller ResNet.

We adopt the experimental setup from Chaudhry et al. (2019) : in each dataset we create a "validation set" of 3 tasks, used to select the best hyper-parameters, and keep the remaining tasks for evaluation.

This split is chosen at random and kept the same across all experiments.

In these datasets, the nature and possibly the number of classes for each task changes.

We account for this by training a separate softmax layer for each task, and apply continual learning only to the remaining, "feature-extraction" part of the model.

We report results along two common metrics for continual learning: average accuracy, the accuracy at the end of training averaged over all tasks, and forgetting.

Forgetting is defined in Chaudhry et al. (2018a) as the difference in performance on a task between the current model and the best performing model on this task.

Formally if A T t represents the accuracy on task T at step t of training, the forgetting F T t at step t is defined as F We implement the co-natural update rule on top of 3 baselines:

• Finetuning: Simply train the model on the task at hand, without any form of regularization.

• EWC: Proposed by Kirkpatrick et al. (2017) , it is a simple but effective quadratic regularization approach.

While neither the most recent nor sophisticate regularization technique, it is a natural baseline for us to compare to in that it also consists in a Fisher-based penalty -although in the For visibility we only show accuracies for every fifth task.

The rectangular shaded regions delineate the period during which each task is being trained upon; with the exception of ER, this is the only period the model has access to the data for this task.

loss function instead of the optimization dynamics.

We also use the rolling Fisher described in Section 3.4, making our EWC baseline equivalent to the superior online EWC introduced by Schwarz et al. (2018) .

• ER: Experience replay with a fixed sized episodic memory proposed by Chaudhry et al. (2019) .

While not directly comparable to EWC in that it presupposes access to data from previous tasks, ER is a simple approach that boasts the best performances on a variety of benchmarks (Chaudhry et al., 2019) .

In all experiments, we use memory size 1,000 with reservoir sampling.

Training proceeds as follows: we perform exhaustive search on all the hyper-parameter combinations using the validation tasks.

Every combination is reran 3 times (the order of tasks, model initialization and order of training examples changes with each restart), and rated by accuracy averaged over tasks and restarts.

We then evaluate the best hyper-parameters by continual training on the evaluation tasks.

Results are reported over 5 random restarts (3 for MiniImageNet), and we control for statistical significance using a paired t-test (we pair together runs with the same task ordering).

We refer to Appendix A.2 for more details regarding fine-grained design choices.

The upper half of Table 1 reports the average accuracy of all the tasks at the end of training (higher is better).

We observe that the co-natural gradient always improves greatly over simple finetuning, and occasionally over EWC and ER.

We note that on both datasets, bare-bone co-natural finetuning matches or exceeds the performance of EWC and ER even though it requires strictly fewer resources (no need to store the previous parameters as in EWC, or data in ER).

Even more appreciable is the effect of the co-natural trajectories on forgetting, as shown in the lower half of Table 1 .

As evidenced by the results in the lowest row, using the co-natural gradient systematically results in large drops in forgetting across all approaches and both datasets, even when the average accuracy is not increased.

To get a qualitative assessment of the learning trajectories that yield such results, we visualize the accuracy curves of 10 out of the 47 evaluation tasks of Omniglot in Figure 2 .

We observe that previous approaches do poorly at keeping stable levels of performance over a long period of time (especially for tasks learned early in training), a problem that is largely resolved by the co-natural preconditioning.

This seems to come at the cost of more intransigence (Chaudhry et al., 2018a) , i.e. some of the later tasks are not being learnt properly.

In models of fixed capacity, there is a natural trade-off between intransigence and forgetting (see also the "stability-plasticity" dilemma in neuroscience Grossberg (1982) ).

Our results position the co-natural gradient as a strong lowforgetting/moderate intransigence basis for future work.

In this section we take a closer look at the specific case of adapting a model from a single task to another, when we only have access to a minimal amount of data in the target task.

In this case, controlling the learning trajectory is particularly important because the model is being trained on an unreliable sample of the true distribution of the target task, and we have to rely on early-stopping to prevent overfitting.

We show that using the co-natural gradient during adaptation helps both at preserving source task performance and reach higher overall target task performance.

We perform experiments on two different scenarios:

Image classification We take MiniImagenet as a source task and CUB (a 200-way birds species classification dataset; Welinder et al. (2010) ) as a target task.

To guarantee a strong base model despite the small size of MiniImageNet, we start off from a ResNet18 model (He et al., 2016) pretrained on the full ImageNet, which we retrofit to MiniImageNet by replacing the last fully connected layer with a separate linear layer regressed over the MiniImageNet training data.

To simulate a low-resource setting, we sub-sample the CUB training set to 200 images (≈ 1 per class).

Scores for these tasks are reported in terms of accuracy.

Machine translation We consider adaptation of an English to French model trained on WMT15 (a dataset of parallel sentences crawled from parliamentary proceedings, news commentary and web page crawls; Bojar et al. (2015) ) to MTNT (a dataset of Reddit comments; Michel & Neubig (2018) ).

Our model is a Transformer (Vaswani et al., 2017 ) pretrained on WMT15.

Similarly to CUB, we simulate a low-resource setting by taking a sub-sample of 1000 sentence pairs as a training set.

Scores for these two datasets are reported in terms of BLEU score.

6 (Papineni et al., 2002) Here we do not allow any access to data in the source task when training on the target task.

We compare four methods Finetuning (our baseline), Co-natural finetuning, EWC (which has been proven effective for domain adaptation, see Thompson et al. (2019) ) and Co-natural EWC.

Given that different methods might lead to different trade-offs between source and target task performance, with some variation depending on the hyper-parameters (e.g. learning rate, regularization strength. . . ), we take inspiration from Thompson et al. (2019) and graphically report results for all hyper-parameter configuration of each method on the 2 dimensional space defined by the score on source and target tasks 7 .

Additionally, we highlight the Pareto frontier of each method i.e. the set of configurations that are not strictly worse than any other configuration for the same model.

The adaptation results for both scenarios are reported in Figure 3 .

We find that in both cases, the co-natural gradient not only helps preserving the source task performance, but to some extent it also allows the model to reach better performance on the target task as well.

We take this to corroborate our starting hypothesis: while introducing a regularizer does help, controlling the optimization dynamics actively helps counteract overfitting to the very small amount of training data, because the co-natural pre-conditioning makes it harder for stochastic gradient descent to push the model towards directions that would also hurt the source task.

We have presented the co-natural gradient, a technique that regularizes the optimization trajectory of models trained in a continual setting.

We have shown that the co-natural gradient stands on its own as an efficient approach for overcoming catastrophic forgetting, and that it effectively complements and stabilizes other existing techniques at a minimal cost.

We believe that the co-natural gradientand more generally, trajectory regularization -can serve as a solid bedrock for building agents that learn without forgetting.

We solve the Lagrangian from Eq. 6 in a similar fashion as in A.1.1.

First we compute its gradient and Hessian: ∇L = ∇L T + 2µδ + 2νF

This section is intended to facilitate the reproduction of our results.

The full details can be found with our code at anonymized url.

We split the dataset into 20 disjoint sub-tasks with each 5 classes, 2500 training examples and 500 test examples.

This split, performed at random, is kept the same across all experiments, only the order of these tasks is changed.

During continual training, we train the model for one epoch on each task with batch size 10, following the setup in Chaudhry et al. (2018b) .

We consider each alphabet as a separate task, and split each task such that every character is present 12, 4 and 4 times in the training, validation and test set respectively (out of the 20 images for each character).

During continual training, we train for 2500 steps with batch size 32 (in keeping with Schwarz et al. (2018) ).

We ignore the validation data and simply evaluate on the test set at the end of training.

For each method, we perform grid-search over the following parameter values:

• Learning rate (all methods): 0.1, 0.03, 0.01

• Regularization strength (EWC, Co-natural EWC): 0.5, 1, 5

• Fisher damping coefficient (Co-natural finetuning, Co-natural EWC): 0,1.0,0.1 for Split CIFAR and 0,0.1,0.01 for Omniglot

For ER, we simply set the batch size to the same value as standard training (10 and 32 for Split CIFAR and Omniglot respectively).

Note that whenever applicable, we re-normalize the diagonal Fisher so that the sum of its weights is equal to the number of parameters in the model.

This is so that the hyper-parameter choice is less dependent on the size of the model.

In particular this means that the magnitude of each diagonal element is much bigger, which is why we do grid-search over smaller regularization parameters for EWC.

<|TLDR|>

@highlight

Regularizing the optimization trajectory with the Fisher information of old tasks reduces catastrophic forgetting greatly