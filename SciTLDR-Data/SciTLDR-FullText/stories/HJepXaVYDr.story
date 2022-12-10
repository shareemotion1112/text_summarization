Stochastic AUC maximization has garnered an increasing interest due to better fit to imbalanced data classification.

However, existing works are limited to stochastic AUC maximization with a linear predictive model, which restricts its predictive power when dealing with extremely complex data.

In this paper, we consider stochastic AUC maximization problem with a deep neural network as the predictive model.

Building on the saddle point reformulation of a surrogated loss of AUC, the problem can be cast into a {\it non-convex concave} min-max problem.

The main contribution made in this paper is to make stochastic AUC maximization more practical for deep neural networks and big data with theoretical insights as well.

In particular, we propose to explore Polyak-\L{}ojasiewicz (PL) condition that has been proved and observed in deep learning, which enables us to develop new stochastic algorithms with even faster convergence rate and more practical step size scheme.

An AdaGrad-style algorithm is also analyzed under the PL condition with adaptive convergence rate.

Our experimental results demonstrate the effectiveness of the proposed algorithms.

Deep learning has been witnessed with tremendous success for various tasks, including computer vision (Krizhevsky et al., 2012; Simonyan & Zisserman, 2014; He et al., 2016; Ren et al., 2015) , speech recognition (Hinton et al., 2012; Mohamed et al., 2012; Graves, 2013) , natural language processing (Bahdanau et al., 2014; Sutskever et al., 2014; Devlin et al., 2018) , etc.

From an optimization perspective, all of them are solving an empirical risk minimization problem in which the objective function is a surrogate loss of the prediction error made by a deep neural network in comparison with the ground-truth label.

For example, for image classification task, the objective function is often chosen as the cross entropy between the probability distribution calculated by forward propagation of a convolutional neural network and the vector encoding true label information (Krizhevsky et al., 2012; Simonyan & Zisserman, 2014; He et al., 2016) , where the cross entropy is a surrogate loss of the misclassification rate.

However, when the data is imbalanced, this formulation is not reasonable since the data coming from minor class have little effect in this case and the model is almost determined by the data from the majority class.

To address this issue, AUC maximization has been proposed as a new learning paradigm (Zhao et al., 2011) .

Statistically, AUC (short for Area Under the ROC curve) is defined as the probability that the prediction score of a positive example is higher than that of a negative example (Hanley & McNeil, 1982; 1983) .

Compared with misclassification rate and its corresponding surrogate loss, AUC is more suitable for imbalanced data setting (Elkan, 2001) .

Several online or stochastic algorithms for time based on a new sampled/received training data.

Instead of storing all examples in the memory, Zhao et al. (2011) employ reservoir sampling technique to maintain representative samples in a buffer, based on which their algorithms update the model.

To get optimal regret bound, their buffer size needs to be O( √ n), where n is the number of received training examples.

Gao et al. (2013) design a new algorithm which is not buffer-based.

Instead, their algorithm needs to maintain the first-order and second-order statistics of the received data to compute the stochastic gradient, which is prohibitive for high dimensional data.

Based on a novel saddle-point reformulation of a surrogate loss of AUC proposed by (Ying et al., 2016) , there are several studies (Ying et al., 2016; Liu et al., 2018; Natole et al., 2018) trying to design stochastic primal-dual algorithms.

Ying et al. (2016) employ the classical primal-dual stochastic gradient (Nemirovski et al., 2009 ) and obtain O(1/ √ t) convergence rate.

Natole et al. (2018) add a strongly convex regularizer, invoke composite mirror descent (Duchi et al., 2010 ) and achieve O(1/t) convergence rate.

Liu et al. (2018) leverage the structure of the formulation, design a multi-stage algorithm and achieve O(1/t) convergence rate without strong convexity assumptions.

However, all of them only consider learning a linear model, which results in a convex objective function.

Non-Convex Min-max Optimization.

Stochastic optimization of non-convex min-max problems have received increasing interests recently (Rafique et al., 2018; Lin et al., 2018; Sanjabi et al., 2018; Lu et al., 2019; Jin et al., 2019) .

When the objective function is weakly convex in the primal variable and is concave in the dual variable, Rafique et al. (2018) design a proximal guided algorithm in spirit of the inexact proximal point method (Rockafellar, 1976) , which solves a sequence of convexconcave subproblems constructed by adding a quadratic proximal term in the primal variable with a periodically updated reference point.

Due to the potential non-smoothness of objective function, they show the convergence to a nearly-stationary point for the equivalent minimization problem.

In the same vein as (Rafique et al., 2018) , Lu et al. (2019) design an algorithm by adopting the block alternating minimization/maximization strategy and show the convergence in terms of the proximal gradient.

When the objective is weakly convex and weakly concave, Lin et al. (2018) propose a proximal algorithm which solves a strongly monotone variational inequality in each epoch and establish its convergence to stationary point.

Sanjabi et al. (2018) consider non-convex non-concave min-max games where the inner maximization problem satisfies a PL condition, based on which they design a multi-step deterministic gradient descent ascent with convergence to a stationary point.

It is notable that our work is different in that (i) we explore the PL condition for the outer minimization problem instead of the inner maximization problem; (ii) we focus on designing stochastic algorithms instead of deterministic algorithms.

Leveraging PL Condition for Minimization.

PL condition is first introduced by Polyak (Polyak, 1963) , which shows that gradient descent is able to enjoy linear convergence to a global minimum under this condition.

Karimi et al. (2016) show that stochastic gradient descent, randomized coordinate descent, greedy coordinate descent are able to converge to a global minimum with faster rates under the PL condition.

If the objective function has a finite-sum structure and satisfies PL condition, there are several non-convex SVRG-style algorithms (Reddi et al., 2016; Lei et al., 2017; Nguyen et al., 2017; Zhou et al., 2018; Li & Li, 2018; Wang et al., 2018) , which are guaranteed to converge to a global minimum with a linear convergence rate.

However, the stochastic algorithms in these works are developed for a minimization problem, and hence is not applicable to the min-max formulation for stochastic AUC maximization.

To the best of our knowledge, Liu et al. (2018) is the only work that leverages an equivalent condition to the PL condition (namely quadratic growth condition) to develop a stochastic primal-dual algorithm for AUC maximization with a fast rate.

However, as mentioned before their algorithm and analysis rely on the convexity of the objective function, which does not hold for AUC maximization with a deep neural network.

Finally, we notice that PL condition is the key to many recent works in deep learning for showing there is no spurious local minima or for showing global convergence of gradient descent and stochastic gradient descent methods (Hardt & Ma, 2016; Li & Yuan, 2017; Arora et al., 2018; Allen-Zhu et al., 2018; Du et al., 2018b; a; Li & Liang, 2018; Allen-Zhu et al., 2018; Zou et al., 2018; Zou & Gu, 2019) .

Using the square loss, it has also been proved that the PL condition holds globally or locally for deep linear residual network (Hardt & Ma, 2016) , deep linear network, one hidden layer neural network with Leaky ReLU activation (Charles & Papailiopoulos, 2017; Zhou & Liang, 2017) .

Several studies (Li & Yuan, 2017; Arora et al., 2018; Allen-Zhu et al., 2018; Du et al., 2018b; Li & Liang, 2018) consider the trajectory of (stochastic) gradient descent on learning neural networks, and their analysis imply the PL condition in a certain form.

For example, Du et al. (2018b) show that when the width of a two layer neural network is sufficiently large, a global optimum would lie in the ball centered at the initial solution, in which PL condition holds.

Allen-Zhu et al. (2018) extends this insight further to overparameterized deep neural networks with ReLU activation, and show that the PL condition holds for a global minimum around a random initial solution.

Let · denote the Euclidean norm.

A function f (x) is ρ-weakly convex if f (x) + ρ 2 x 2 is convex, where ρ is the so-called weak-convexity parameter.

A function f (x) satisfies PL condition with

2 , where x * stands for the optimal solution of f .

Let z = (x, y) ∼ P denote a random data following an unknown distribution P, where x ∈ X represents the feature vector and y ∈ Y = {−1, +1} represents the label.

Denote by Z = X × Y and by p = Pr(y = 1) = E y I [y=1] , where I(·) is the indicator function.

The area under the curve (AUC) on a population level for a scoring function h : X → R is defined as AUC(h) = Pr (h(x) ≥ h(x )|y = 1, y = −1) , where z = (x, y) and z = (x , y ) are drawn independently from P. By employing the squared loss as the surrogate for the indicator function that is a common choice used by previous studies (Ying et al., 2016; Gao et al., 2013) , the AUC maximization problem can be formulated as min

where H denotes a hypothesis class.

All previous works of AUC maximization assume h(x) = w x for simplicity.

Instead, we consider learning a general nonlinear model parameterized by w, i.e. h(w; x), which is not necessarily linear or convex in terms of w (e.g., h(w; x) can be a score function defined by a neural network with weights denoted by w).

Hence, the corresponding optimization problem becomes min

The following proposition converts the original optimization problem (1) into a saddle-point problem, which is similar to Theorem 1 in (Ying et al., 2016) .

For completeness, the proof is included in the supplement.

Proposition 1.

The optimization problem (1) is equivalent to min

where z = (x, y) ∼ P, and

Remark:

It is notable that the min-max formulation (2) is more favorable than the original formulation (1) for developing a stochastic algorithm that updates the model parameters based on one example or a mini-batch of samples.

For stochastic optimization of (1), one has to carefully sample both positive and negative examples, which is not allowed in an online setting.

It is notable that in the classical batch-learning setting, p becomes the ratio of positive training examples and the expectation in (2) becomes average over n individual functions.

However, our algorithms are applicable to both batch-learning setting and online learning setting.

It is clear that min w P (w) = min v φ(v) and P (w) ≤ φ(v) for any v = (w , a, b) .

The following assumption is made throughout the paper.

, where µ > 0 and v * is the optimal solution

where v * is the global minimum of φ.

The first condition is inspired by a PL condition on the objective function P (w) for learning a deep neural network.

The following lemma establishes the connection.

Algorithm 1 Proximally Guided Algorithm (PGA) (Rafique et al., 2018) 1: Initializev 0 = 0 ∈ R d+2 ,ᾱ 0 = 0, the global index j = 0 2: for k = 1, . . .

, K do 3:

10: end for 11: Sample τ uniformly randomly from {1, . . .

, K} 12: returnv τ ,ᾱ τ Lemma 1.

Suppose ∇ w h(w; x) ≤L for all w and x. If P (w) satisfies PL condition, i.e. there exists

Remark: The PL condition of P (w) could be proved for learning a neural network similar to existing studies, which is not the main focus of this paper.

Nevertheless, In Appendix A.7, we provide an example for AUC maximization with one-hidden layer neural network.

Warmup.

We first discuss the algorithms and their convergence results of (Rafique et al., 2018) applied to the considered min-max problem.

They have algorithms for problems in batch-learning setting and online learning setting.

Since the algorithms for the batch-learning setting have complexities scaling with n, we will concentrate on the algorithm for the online learning setting.

The algorithm is presented in Algorithm 1, which is a direct application of Algorithm 2 of (Rafique et al., 2018) to an online setting.

Since their analysis requires the domain of the primal and the dual variable to be bounded, hence we add a ball constraint on the primal variable and the dual variable as well.

As long as R 1 and R 2 is sufficiently large, they should not affect the solution.

The convergence result of Algorithm 1 is stated below.

Remark: Under the condition φ(v) is smooth and the returned solution is within the added bounded ball constraint, the above result implies

We can see that this complexity result under the PL condition of φ(v) is worse than the typical complexity result of stochastic gradient descent method under the PL condition (i.e., O(1/ )) (Karimi et al., 2016) .

It remains an open problem how to design a stochastic primal-dual algorithm for solving min v max α F (v, α) in order to achieve a complexity of O(1/ ) in terms of minimizing φ(v).

A naive idea is to solve the inner maximization problem of α first and the use SGD on the primal variable v. However, this is not viable since exact maximization over α is a non-trivial task.

In this section, we present two primal-dual algorithms for solving the min-max optimization problem (2) with corresponding theoretical convergence results.

For simplicity, we first assume the positive ratio p is known in advance, which is true in the batch-learning setting.

Handling the unknown p in an online learning setting is a simple extension, which will be discussed in Section 4.3.

The proposed algorithms follow the same proximal point framework proposed in (Rafique et al., 2018) , i.e., we

Draw a minibatch {z j , . . .

,

solve the following convex-concave problems approximately and iteratively:

where γ < 1/L to ensure that the new objective function becomes convex and concave, and v 0 is periodically updated.

Algorithm 2.

Similar to Algorithm 1, it has a nested loop, where the inner loop is to approximately solve a regularized min-max optimization problem (3) using stochastic primal-dual gradient method, and the outer loop updates the reference point and learning rate.

One key difference is that PPD-SG uses a geometrically decaying step size scheme, while Algorithm 1 uses a polynomially decaying step size scheme.

Another key difference is that at the end of k-th outer loop, we update the dual variableᾱ k in Step 12, which is motivated by its closed-form solution givenv k .

In particular, the givenv k , the dual solution that optimizes the inner maximization problem is given by:

In the algorithm, we only use a small number of samples in Step 11 to compute an estimation of the optimal α givenv k .

These differences are important for us to achieve lower iteration complexity of PPD-SG.

Next, we present our convergence results of PPD-SG.

where E k−1 stands for the conditional expectation conditioning on all the stochastic events until v k−1 is generated.

Theorem 2.

Suppose the same conditions in Lemma 2 hold.

Set

2 ) Lη0

in Algorithm 2, where O(·) hides logarithmic factor of L, µ, , G, σ.

Remark: The above complexity result is similar to that of (Karimi et al., 2016) for solving nonconvex minimization problem under the PL condition up to a logarithmic factor.

Compared with the complexity result of Algorithm 1 discussed earlier, i.e., O(1/(µ 3 3 )), the above complexity in the

) is much better -it not only improves the dependence on but also improves the dependence on µ.

Our second algorithm named Proximal Primal-Dual Adagrad (PPD-Adagrad) is a AdaGrad-style algorithm.

Since it only differs from PPD-SG in the updates of the inner loop, we only present the inner loop in Algorithm 3.

The updates in the inner loop are similar to the adaptive updates of traditional AdaGrad (Duchi et al., 2011) .

We aim to achieve an adaptive convergence by using PPD-AdaGrad.

The analysis of PPD-AdaGrad is inspired by the analysis of AdaGrad for non-convex minimization problems (Chen et al., 2019) .

The key difference is that we have to carefully deal with the primal-dual updates for the non-convex min-max problem.

We summarize the convergence results of PPD-AdaGrad below.

where E k−1 stands for the conditional expectation conditioning on all the stochastic events until v k−1 is generated.

Theorem 3.

Suppose the same conditions as in Lemma 3 hold.

Set

The number of iterations is at most O

, and the required number of samples is at

, where O(·) hides logarithmic factors of L, µ, , δ.

Remark: When the cumulative growth of stochastic gradient is slow, i.e., α < 1/2, the number of iterations is less than that in Theorem 2, which exhibits adaptive iteration complexity.

It is notable that the setting of η k , T k , m k depends on unknown parameters µ, L, etc., which are typically unknown.

One heuristic to address this issue is that we can decrease η k by a constant factor larger than 1 (e.g., 2 or 5 or 10), and similarly increase T k and m k by a constant factor.

Another heuristic is to decrease the step size by a constant factor when the performance on a validation data saturates (Krizhevsky et al., 2012) .

Variants when p is unknown.

In the online learning setting when p is unknown, the stochastic gradients of f in both v and α are not directly available.

To address this issue, we can keep unbiased estimators for both p and p(1 − p) which are independent of the new arrived data, and update these estimators during the optimization procedure.

All values depending on p and p(1 − p) (i.e., F, g v , g α ) are estimated by substituting p and p(1 − p) by p and p(1 − p) (i.e.,F ,ĝ v ,ĝ α ) respectively.

The approach for keeping unbiased estimatorp and p(1 − p) during the optimization is described in Algorithm 4, where j is the global index, and m is the number of examples received.

Extensions to multi-class problems.

In the previous analysis, we only consider the binary classification problem.

We can extend it to the multi-class setting.

To this end, we first introduce the definition of AUC in this setting according to (Hand & Till, 2001) .

Suppose there are c classes, we have c scoring functions for each class, namely h(w 1 ; x), . . .

, h(w c ; x).

We assume that these scores are normalized such that c k=1 h(w c ; x) = 1.

Note that if these functions are implemented by a deep neural network, they can share the lower layers and have individual last layer of connections.

The AUC is defined as

Similar to Proposition 1, we can cast the problem into

where

ij .

Then we can modify our algorithms to accommodate the multiple class pairs.

We can also add another level of sampling of class pairs into computing the stochastic gradients.

In this section, we present some empirical results to verify the effectiveness of the proposed algorithms.

We compare our algorithms (PPD-SG and PPD-AdaGrad) with three baseline methods including PGA (Algorithm 1), Online AUC method (Ying et al., 2016) (OAUC) that directly employs the standard primal-dual stochastic gradient method with a decreasing step size for solving the min-max formulation, and the standard stochastic gradient descent (SGD) for minimizing cross-entropy loss.

Comparing with PGA and OAUC allows us to verify the effectiveness of the proposed algorithms for solving the same formulation, and comparing with SGD allows us to verify the effectiveness of maximizing AUC for imbalanced data.

We use a residual network with 20 layers to implement the deep neural network for all algorithms.

We use the stagewise step size strategy as in (He et al., 2016) for SGD, i.e. the step size is decreased by 10 times at 40K, 60K.

For PPD-SG and PPD-AdaGrad, we set We conduct the comparisons on four benchmark datasets, i.e., Cat&Dog (C2), CIFAR10 (C10), CIFAR100 (C100), STL10.

STL10 is an extension of CIFAR10 and the images are acquired from ImageNet.

Cat&Dog is from Kaggle containing 25,000 images of dogs and cats and we choose an 80:20 split to construct training and testing set.

We use 19k/1k, 45k/5k, 45k/5k, 4k/1k training/validation split on C2, C10, C100, and STL10 respectively.

For each dataset, we construct multiple binary classification tasks with varying imbalanced ratio of number negative examples to number of positive examples.

For details of construction of binary classification tasks, please refer to the Appendix A.8.

We report the convergence of AUC on testing data in Figure 1 , where the title shows the ratio of the majority class to the minority class.

The results about the convergence of AUC versus the time in seconds are also presented in Figure 3 .

From the results we can see that for the balanced settings with ratio equal to 50%, SGD performs consistently better than other methods on C2 and CIFAR10 data.

However, it is worse than AUC optimization based methods on CIFAR100 and STL10.

For imbalanced settings, AUC maximization based methods are more advantageous than SGD in most cases.

In addition, PPD-SG and PPD-AdaGrad are mostly better than other baseline algorithms.

In certain cases, PPD-AdaGrad can be faster than PPD-SG.

Finally, we observe even better performance (in Appendix) by a mixed strategy that pre-trains the model with SGD and then switchs to PPD-SG.

In this paper, we consider stochastic AUC maximization problem when the predictive model is a deep neural network.

By abuilding on the saddle point reformulation and exploring Polyak-Łojasiewicz condition in deep learning, we have proposed two algorithms with state-of-the-art complexities for stochastic AUC maximization problem.

We have also demonstrated the efficiency of our proposed algorithms on several benchmark datasets, and the experimental results indicate that our algorithms converge faster than other baselines.

One may consider to extend the analysis techniques to other problems with the min-max formulation.

Proof.

It suffices to prove that

Note that the optimal values of a, b, α are chosen as a *

2 , (c) comes from the standard analysis of primal-dual stochastic gradient method.

Denote E k−1 by taking the conditional expectation conditioning on all the stochastic events until v k−1 is generated.

Taking E k−1 on both sides and noting thatĝ k t is an unbiased estimator of g k t for ∀t, k, we have

By the update ofᾱ k−1 , 2L-Lipschitz continuity of E [h(w; x)|y = −1] − E [h(w; x)|y = 1], and noting that α

, then we have

We can see that φ k (v) is convex and smooth function since γ ≤ 1/L. The smoothness parameter of φ k isL = L+γ −1 .

Define s k = arg min v∈R d+2 φ k (v).

According to Theorem 2.1.5 of (Nesterov, 2013), we have

Combining (8) with Lemma 2 yields

Note that φ k (v) is (γ −1 − L)-strongly convex, and γ = 1 2L , we have

Plugging in s k into Lemma 2 and combining (10) yield

2 ), rearranging the terms, and noting that

Combining (11) and (9) yields

(12) Taking expectation on both sides over all randomness untilv k−1 is generated and by the tower property, we have

is L-smooth and hence is L-weakly convex, so we have

where (a) and (b) hold by the definition of φ k .

Rearranging the terms in (14) yields

where (a) holds by using a, b ≤ 1 2 ( a 2 + b 2 ), and (b) holds by the PL property of φ.

Combining (13) and (15), we can see that

As a result, we have

Published as a conference paper at ICLR 2020

2 ), by the setting of η k , we set

The required number of samples is

A.4 PROOF OF LEMMA 3

2 , (c) holds by Jensen's inequality.

Now we bound I and II separately.

Define

Combining (17) and (20), we have

By Lemma 4 of (Duchi et al., 2011) and setting δ ≥ max t ĝ k t ∞ , we know that

T k 2 , and hence

Denote E k−1 by taking the conditional expectation conditioning on filtration F k−1 , where F k−1 is the σ-algebra generated by all random variables untilv k−1 is generated.

Taking E k−1 on both sides of (16), and employing (22) yields

where the equality holds sincev k−1 − s k is measurable with respect to F k−1 .

Note that

where (

By setting

, then T k is a stopping time which is bounded almost surely.

By stopping time argument, we have E k−1 (II) = 0, and hence

A.5 PROOF OF THEOREM 3

We can see that φ k (v) is convex and smooth function since γ ≤ 1/L. The smoothness parameter of φ k isL = L+γ −1 .

Define s k = arg min v∈R d+2 φ k (v).

According to Theorem 2.1.5 of (Nesterov, 2013), we have

Combining (24) with Lemma 3 yields

Note that

(26) Plugging in s k into Lemma 3 and combining (26) yield

, rearranging the terms, and noting that

Combining (27) and (25) yields

Taking expectation on both sides over all randomness untilv k−1 is generated and by the tower property, we have

Note that φ(v) is L-smooth and hence is L-weakly convex, so we have

where (a) and (b) hold by the definition of φ k .

Rearranging the terms in (30) yields

where (a) holds by using a, b ≤ 1 2 ( a 2 + b 2 ), and (b) holds by the PL property of φ.

Combining (29) and (31), we can see that

which implies that

As a result, we have

, and note that when τ ≥ 1,

, and hence

, we can see that the total iteration complexity is

.

The required number of samples is

A.6 PROOF OF LEMMA 1

Proof.

For any fixed w, define (a * w , b * w ) = arg min a,b φ(w, a, b) (φ(w, a, b) is strongly convex in terms of (a, b), so the argmin is well-defined and unique).

Note that

we can write σ(w x) and σ(w x ) as aw x and bw x respectively, and it is obvious that a 2 ≥ min(c

We construct the datasets in the following ways: For CIFAR10/STL10, we label the first 5 classes as negative ("-") class and the last 5 classes as positive ("+") class, which leads to a 50/50 class ratio.

For CIFAR100, we label the first 50 classes as negative ("-") class and the last 50 classes as positve ("+") class.

For the imbalanced cases, we randomly remove 90%, 80%, 60% data from negative samples on all training data, which lead to 91/9, 83/17, 71/29 ratio respectively.

For testing data, we keep them unchanged.

Model pretraining is effective in many deep learning tasks, and thus we further evaluate the performance of the proposed methods on pretrained models.

We first train the model using SGD up to 2000 iterations with an initial step size of 0.1, and then continue training using PPD-SG.

We denote this method as PPD-SG+pretrain and the results are shown in Figure 2 .

The parameters are tuned in the same range as in Section 5.

It is observed that pretraining model helps the convergence of model and it can achieve the better performance in terms of AUC in most cases.

To investigate the effects of labeling order, we also attempt to randomly partition the classes as positive or negative equally.

For CIFAR10 and STL10 dataset, we randomly partition the 10 classes into two labels (i.e., randomly select 5 classes as positive label and other 5 classes as negative label).

For CIFAR100 dataset, we randomly partition the 100 classes into two labels (i.e., randomly select 50 classes as positive label and other 50 classes as negative label).

After that we randomly remove 95%, 90%, from negative samples on all training data, which lead to 20:1, 10:1 ratios respectively.

For testing data, we keep them unchanged.

We also add AdaGrad for minimizing cross-entropy loss as a new baseline.

The corresponding experimental results are included in Figure 3 .

We can see that PPD-Adagrad and PPD-SG converge faster than other baselines.

and STL10 dataset, we randomly partition the 10 classes into two labels (i.e., randomly select 5 classes as positive label and other 5 classes as negative label).

For CIFAR100 dataset, we randomly partition the 100 classes into two labels (i.e., randomly select 50 classes as positive label and other 50 classes as negative label).

@highlight

The paper designs two algorithms for the stochastic AUC maximization problem with state-of-the-art complexities when using deep neural network as predictive model, which are also verified by empirical studies.