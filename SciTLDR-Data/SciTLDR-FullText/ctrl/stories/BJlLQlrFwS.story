Continual learning is a longstanding goal of artificial intelligence, but is often counfounded by catastrophic forgetting that prevents neural networks from learning tasks sequentially.

Previous methods in continual learning have demonstrated how to mitigate catastrophic forgetting, and learn new tasks while retaining performance on the previous tasks.

We analyze catastrophic forgetting from the perspective of change in classifier likelihood and propose a simple L1 minimization criterion which can be adapted to different use cases.

We further investigate two ways to minimize forgetting as quantified by this criterion and propose strategies to achieve finer control over forgetting.

Finally, we evaluate our strategies on 3 datasets of varying difficulty and demonstrate improvements over previously known L2 strategies for mitigating catastrophic forgetting.

Machine learning has achieved successes in many applications, including image recognition, gameplaying, content recommendation and health-care (LeCun et al., 2015) .

Most of these systems require large amounts of training data and careful selection of architecture and parameters.

Moreover, such systems often have to adapt to changing real-world requirements, and therefore changes in the data.

Under these circumstances it is usually desired to retain performance on previous data while learning to perform well on training data with a different distribution.

This is what constitutes continual learning (McCloskey, 1989) .

A well known problem in the context of continual learning is "catastrophic forgetting" (Goodfellow et al., 2013) , which occurs when the training process ends up modifying weights crucial to the performance on the previous data.

There has been a lot of work in trying to overcome catastrophic forgetting.

Broadly, the approaches in the literature try to mitigate forgetting in three ways: (a) architectural approaches (Yoon et al., 2018; Li et al., 2019) try to incrementally grow the network to learn the new task through added capacity, (b) regularization approaches (Kirkpatrick et al., 2016; Zenke et al., 2017; Wiewel & Yang, 2019) regularize changes to crucial weights, so that the network can learn to perform well on the new task while preserving the performance on the previous tasks (assuming the network has enough capacity for all tasks), and (c) memory approaches (Lopez-Paz, 2017; Nguyen et al., 2018 ) store examples from each task being learned and then learn a new task while simultaneously maximizing performance on each of the stored memories.

Performance in these works is often judged with respect to overall accuracy.

In the present work, we specifically consider exactly what has been forgotten and what has been learned.

Such considerations may be important in safety-critical systems or in systems that have been calibrated.

For example, in safety-critical systems, it may not be acceptable to maintain overall performance by trading validated decisions for correct decisions that have not been validated.

Likewise, the calibration of a system may require that all decisions, good and bad, remain the same.

For the purposes of this paper, we focus on regularization strategies.

Regularization strategies typically formulate continual learning in two ways: (a) from a Bayesian perspective (Kirkpatrick et al., 2016; Lee et al., 2017; Liu et al., 2018; Chaudhry et al., 2018) where the goal is to learn the newest task while simultaneously minimizing the KL-divergence between the posterior log likelihood distribution and the prior (see Section 2), or (b) by trying to minimize large changes to influential weights for previous tasks (Zenke et al., 2017; Wiewel & Yang, 2019) .

Both these formulations produce an L 2 regularization objective and mitigate forgetting by penalizing changes to weights important to task performances.

However, their exact effect on change in classifier likelihood is not known.

In this paper, we attempt to quantify this change in classifier likelihood more directly and then use it to provide a generic criterion that can be adapted to different use cases of likelihood preservation.

Our contributions are as follows: we propose a more general framework to mitigate catastrophic forgetting, which involves directly penalizing the change in the classifier likelihood functions.

Specifically: (a) we analyze catastrophic forgetting and provide a generic L 1 minimization criterion to mitigate it, (b) we propose two strategies to utilize this criteria and discuss how the cross-entropy loss can be reformulated to achieve finer control over forgetting, and (c) we evaluate these strategies on three datasets and demonstrate improvements over traditional L 2 regularization strategies like elastic weight consolidation (EWC) (Kirkpatrick et al., 2016) and synaptic intelligence (SI) (Zenke et al., 2017) .

Formally, let the tasks correspond to datasets D 1 , D 2 , ?? ?? ?? , D n such that the goal in task i is to achieve the maximum performance on a dataset

), which has K i examples.

Let the likelihood function be approximated by a ReLU feedforward neural network (final layer is followed by softmax) with weights ??, that is, given an example x, the network produces a set of probabilities

, where M is the number of classes.

For notational simplicity, we denote P ?? (y = j|x) as P j ?? (??|x).

If the ground truth for x is g, where (1 ??? g ??? M ), then we use the shorthand for the predicted likelihood of g as P ?? (??|x) := P g ?? (??|x).

For any task i, minimizing its specific cross entropy loss L i (??) achieves the best performance for task i, which can be written as:

For any task i, the ideal weights achieved at the end of task i should also retain performances on tasks 1, 2, ?? ?? ?? , i ??? 1.

Therefore, ideally, the overall joint cross entropy loss over datasets D 1 , D 2 , . . .

, D i should be minimized:

Joint training quickly becomes expensive as the number of tasks grow, but has the best performance across all tasks that were trained on (Li & Hoiem, 2017) .

Bayesian continual learning formulates learning a new task as trying to maximize a posterior p(??|D 1:i ) given a prior p(??|D 1:i???1 ).

So, if the weights ?? at the end of task i ??? 1 and i are denoted by ?? * 1:i???1 and ?? * 1:i respectively, then the prior and the posterior can be thought of as the predicted likelihood distributions represented by the neural network at ?? * 1:i???1 and ?? * 1:i .

For every task i, the Bayesian formulation tries to minimize L i (??) as well as the dissimilarity between the prior and the posterior.

EWC uses the KL-divergence of the two predicted likelihood distributions as the dissimilarity metric.

Assuming the difference between ?? * 1:i???1 and ?? * 1:i is small, the second order Taylor approximation of the KL-divergence produces (with a further diagonal approximation of the Fisher matrix):

For any task i ??? 2, EWC minimizes the sum of L i (??) and this approximation (multiplied by a ?? ??? 0).

?? acts as a hyperparameter that controls the weight of the penalty for a specific learned task.

It is typically kept the same for all learned tasks.

After learning a task i, the weights of the network are ?? * 1:i .

For simplicity, let ?? * 1:i ??? ?? * and that afterwards, at any point in the sequential training process, the weights are at ?? * + ?????.

Assuming ????? is small, we can apply a first order Taylor approximation of the individual predicted likelihood P j ?? * +????? in the neighbourhood of ?? = ?? * :

The individual predicted likelihood P j ?? on an example x ??? D i changes by the magnitude |P j ?? * +????? (??|x) ??? P j ?? * (??|x)|.

The average magnitude change in P j ?? over the dataset D i is given by the expectation:

At every task i, we can minimize directly the average change in predicted likelihood for the previous datasets, and this minimization should mitigate catastrophic forgetting.

This constitutes our minimization criterion.

Depending on the requirement, the minimization criterion can be interpreted to provide a regularization objective.

In this paper, we identify four broad use cases of this criterion:

Case I. We can preserve the entire set of predicted likelihoods from ?? * to ?? * + ?????, which would penalize changes to any individual predicted likelihood.

This is the most restrictive version of the criterion and can be achieved by regularizing a sum over j = 1 to M of the individual changes.

Case II.

We can preserve the change in predicted likelihood for the predicted label at ?? * , which corresponds to the highest individual probability in {P

.

This may be desired in tasks related to safety-critical systems (e.g., autonomous driving), where a network has been safety-calibrated at deployment and now needs to add some more knowledge without violating previously satisfied safety constraints.

To achieve this, we can use the expectation over (P j ?? * +????? (??|x)???P j ?? * (??|x))??P j ?? * (??|x) rather than the original formulation in (1) and then regularize a sum over j = 1 to M like in Case I. In most cases, this term would evaluate to the difference in the individual predicted likelihoods for the predicted label at ?? * (since the probabilities are output by a softmax layer).

Case III.

We can preserve the change in predicted likelihood for the ground truth by computing the expectation for P ?? * +????? (??|x) ??? P ?? * (??|x) .

Case IV.

We can partially preserve the change in predicted likelihood for the ground truth, that is, penalize the change P ?? * (??|x) = 1 ??? P ?? * +????? (??|x) = 0 but allow the change P ?? * (??|x) = 0 ??? P ?? * +????? (??|x) = 1 for the ground truth predicted likelihood.

This applies the penalty only when a correctly classified x at ?? * becomes incorrectly classified at ?? * + ?????.

The expectation is then computed over (P ?? * +????? (??|x) ??? P ?? * (??|x)) ?? P ?? * (??|x) , similar to Case II.

In all of these cases, we end up with a direct minimization criterion that can be minimized just like in EWC.

In fact, the quadratic loss penalty proposed in Kirkpatrick et al. (2016) , which was later corrected in Husz??r (2018) to more appropriately represent Bayesian updating, can be interpreted as the upper-bound of the squared L 2 version of the change in predicted likelihood as described in Case III.

Intuitively, therefore, the quadratic loss penalty works even when not computed as specified in Husz??r (2018) because it penalizes the upper bound of the squared L 2 change in likelihood for task i.

Given (1), we propose two strategies to minimize the expected change in predicted likelihood.

Method 1 (Soft Regularization).

The upper bound in (1) can be directly regularized per task.

With the L 1 change, the loss per task i becomes:

L 1 regularization is known to produce sparser solutions than L 2 regularization above a critical ??, that is, requiring fewer non zero weights, in the context of plain weight regularization (Moore & DeNero, 2011) .

This is because the L 1 objective penalizes change in weights more strongly than in L 2 and forces the weights to stay closer to 0.

In the context of predicted likelihood preservation, similarly it should be expected that the L 1 penalty penalizes change in predicted likelihoods more strongly than L 2 and forces the change in likelihoods to stay closer to 0.

With the 4 cases described in Section 3, Method 1 can be used in 4 ways.

We denote these 4 methods as DM-I, DM-II, DM-III and DM-IV respectively, where DM stands for "Direct Minimization".

Constrained learning.

Better preservation of predicted likelihood also has a downside, that is, if the previous behaviour is preserved too much, the network is unable to learn the new task well.

To counteract this, we introduce two parameters -c 1 and c 2 , to constrain the learning.

For notational simplicity, let us denote the expectation in (2) as G(?? * 1:k , D k ).

After task i has been learned, the absolute change in predicted likelihood (for the use case) is upper bounded by |?? ??? ?? * 1:i |??G(?? * 1:i , D i ).

We can turn off the training on the cross entropy loss after the upper bound on absolute change in likelihood is ??? c.

This can be achieved by modifying the MLE loss to be:

In fact, it is more advantageous to maintain a moving c i for every task i which is initialized with c i ??? c 1 , and then increased to c i ??? c i + c 2 after every new task (c 1 , c 2 ??? 0).

This kind of thresholding provides a direct way to bound the amount of forgetting, compared to the unconstrained learning in EWC or SI.

The advantage of this kind of thresholding is evident from our experiments (Table 3) .

With any soft regularization strategy, all the weights are always updated, even if the changes to some weights are very small.

This might perturb sensitive weights, even if by a small amount.

These small perturbations can add up over multiple tasks and eventually end up affecting the classifier likelihood irreversibly.

The upper bound of change in classifier likelihood for a dataset D i is dependent on two terms (see (1)), |?????| and the expectation of the absolute gradients.

To minimize the change in classifier likelihood, we may opt to minimize |?????| more conventionally, by freezing the most important weights.

This reduces the magnitude of ????? and therefore results in a lesser change in likelihood.

Other strategies in the literature have tried similar approaches (for eg.

Serra et al. (2018) ).

To assess the effects of this kind of freezing separately from L 1 criterion, we freeze weights on EWC.

We denote this method as DM-p.

Specifically, the Fisher information matrix already contains information about which parameters are important, and should not be disturbed.

We impair the gradient update by setting the gradients of top p% important parameters to 0 for each task i ??? 2.

Table 2 : Mean (std) of the final average accuracy (%) with the best hyperparameters, 5 seeds.

Only the best result from DM-I, II, III, IV (constrained as well as unconstrained) is shown.

The comparison among the constrained and unconstrained variants of DM are given in Table 3 .

In this section we describe the methodology and results of our experiments (Tables 1-4) .

Evaluated Methods.

To assess the performance of our strategies, we evaluate our proposed methods and compare its performance with other L 2 variants in continual learning literature.

Following are the methods we evaluate:

??? Baseline: Trained with just the likelihood loss (no regularization).

??? EWC: Accumulated Fisher information matrices and combined quadratic losses, as described in Kirkpatrick et al. (2016); Husz??r (2018) ; Kirkpatrick et al. (2018) .

This was implemented from scratch.

??? SI: Synaptic Intelligence strategy as described in Zenke et al. (2017) .

We use the code provided by the original authors.

??? DM-I, II, III, IV: Proposed in Section 4, soft regularization strategy (Method 1); 4 variants described in Section 3.

For each variant, we conduct experiments with both constrained and unconstrained learning of L 1 criterion.

??? DM-p: Freezing strategy (Method 2) described in Section 4; implemented on EWC.

Training methodology.

Training is done on feedforward ReLU networks for each strategy with 2 hidden layers (h = 128, ?? = 0.0001) for 20 epochs.

For hyperparameter search, we evaluate all methods on a single seed, that is, choose a configuration which has the highest final average validation accuracy.

Then the final results are reported across 5 seeds with the best parameter configuration (mean and standard deviation are shown).

All hyperparameters used are reported in Table 1 .

Table  2 , 3 show the performance (accuracy) of the proposed methods.

Additionally, we also assess the retention of predicted likelihood.

To calculate the likelihood retention, we first compute the predictions per task after the task has been fully trained, then calculate how many of these predictions have changed at the end of future tasks.

If the retained predictions for task i at the end of task j (j > i) is R i,j , then we define likelihood retention after n tasks as (reported in Table 4 ):

Datasets.

We evaluate on the following datasets:

??? Permuted MNIST: 5 task version; every task is 10-class classification on the MNIST dataset with permuted pixels; used in Kirkpatrick et al. (2016)

In this section we give further insights about our results.

Hyperparameter choice.

As can be seen in Table 1 , EWC often requires a high ?? to remember previous tasks better.

In contrast, the L 1 methods perform well even with a small ?? .

This can be explained by the fact that minimization with an L 2 method contains a (|?????|) 2 term instead of (|?????|).

This means that the weights (which are typically quite small) are squared in the L 2 methods, which then requires a stronger ?? to compensate for the squaring.

So, L 1 methods require a hyperparameter search over a smaller range of values.

Degree of preservation.

A higher p in DM-p has the same effect as as a low c 1 , c 2 in constrained DM-I, II, III, IV.

If c 1 , c 2 are too low, then the training switches off very early, and likewise, if p is too high, the requisite weights never change enough to adapt to the newest task.

For the datasets considered, we find that fixing 20 ??? 40% of the weights typically works the best in DM-p.

Improvements over L 2 methods.

??? P-MNIST and Sim-EMNIST: EWC and SI are already known to perform well on P-MNIST.

In our experiments with the 5-task variant of P-MNIST, they reach an average final accuracy of ??? 94%.

All of DM-I, DM-II, DM-III, DM-IV and DM-p outperform EWC and SI on the 5 task P-MNIST for the same number of epochs, as evidenced by Table 2 .

A large improvement was not expected, since EWC already performs well on these datasets.

??? S-MNIST: S-MNIST is a difficult dataset because it only involves 2-class classification for each task, which means that the decision boundary found by the network at each task is very susceptible to change in the decision boundary at the next task.

This is why EWC is unable to reach beyond ??? 69% on S-MNIST.

DM-p improves on this by a few points, but DM-I, II, III, IV all improve on EWC by ??? 7 ??? 10%.

Effect of constrained learning.

As can be seen in Table 3 , tuned constrained DM-I, II, III, IV all perform better or similar than the tuned unconstrained counterparts.

Effect of different types of preservation on performance.

While DM-I, II might be suited to specific applications, DM-III, IV typically perform the best in terms of accuracy improvement.

This is expected, since DM-III, IV directly regularize change in predicted likelihood for the ground truth.

Effect of different types of preservation on retention.

We observe mixed results with respect to retention.

While it is expected that a higher retention should correspond to a lower amount of forgetting, Table 4 does not show that the L 1 criterion universally has the best retention across the tested datasets.

Specifically, the retention advantage of the L 1 criterion is clear for P-MNIST, but it is not as clear for S-MNIST or Sim-EMNIST.

We speculate that this is because of the ?? chosen for S-MNIST and Sim-EMNIST during the hyperparameter search.

During the search, ?? is optimized for the best accuracy.

In order for EWC to have the best accuracy for these datasets (S-MNIST, Sim-EMNIST), the required hyperparameter ?? is huge (10 4 ), which leads to an over-preservation of past classifier likelihood at the expense of the learning the likelihood for the newest task, while the proposed DM strategies use a normal ?? for their corresponding best performance.

In fact, the huge ?? leads to sub-optimal performance for the newest task in EWC, but maximizes the average final accuracy.

The retention metric does not capture this sub-optimal performance.

Out of DM-I, II, III, IV, the method DM-III retains the most amount of predictions, empirically.

For DM-p, the retention advantage is clearly better than plain EWC for P-MNIST and S-MNIST, and close to plain EWC for Sim-EMNIST.

Most real-world classification systems rely on connectionist networks, which are known to suffer from catastrophic forgetting when subjected to sequential learning tasks.

Existing (regularization) strategies to mitigate catastrophic forgetting typically minimize an L 2 criterion, which can produce non-sparse solutions and require a costly hyperparameter search for the appropriate penalty weight.

In this paper, we proposed a more general criterion that involves direct minimization of the change in classifier likelihood and explained how to adapt the criterion to four broad use cases.

Using this criterion, we identified two ways to improve the classifier performance: (a) by directly softregularizing the change in classifier likelihood and (b) by freezing influential weights.

Both of these perform better than, or at least similar to, existing L 2 strategies.

We further discussed the effect of various proposed classifier likelihood preservation methods and showed that preserving the classifier likelihood with respect to the ground truth is a good strategy to preserve classifier performance.

Future Work.

Having compared our method to existing L 2 strategies, it would be interesting to compare and contrast the benefits and problems of the proposed L 1 strategies with other non-L 2 strategies for continual learning, e.g., IMM (Lee et al., 2017) and VCL (Nguyen et al., 2018) .

It would be also be interesting to see the effect of direct minimization strategies for more complicated and realistic image classification datasets, like CIFAR100 (Krizhevsky et al., 2009) and ImageNet (Deng et al., 2009 ).

<|TLDR|>

@highlight

Another perspective on catastrophic forgetting

@highlight

This paper introduces a framework for combatting catastrophic forgetting based upon changing the loss term to minimise changes in classifier likelihood, obtained via a Taylor series approximation.

@highlight

This paper tries to solve the continual learning prolem by focusing on regularization approaches, and it proposes a L_1 strategy to mitigate the problem.