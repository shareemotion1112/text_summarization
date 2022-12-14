Adaptive optimization methods such as AdaGrad, RMSprop and Adam have been proposed to achieve a rapid training process with an element-wise scaling term on learning rates.

Though prevailing, they are observed to generalize poorly compared with SGD or even fail to converge due to unstable and extreme learning rates.

Recent work has put forward some algorithms such as AMSGrad to tackle this issue but they failed to achieve considerable improvement over existing methods.

In our paper, we demonstrate that extreme learning rates can lead to poor performance.

We provide new variants of Adam and AMSGrad, called AdaBound and AMSBound respectively, which employ dynamic bounds on learning rates to achieve a gradual and smooth transition from adaptive methods to SGD and give a theoretical proof of convergence.

We further conduct experiments on various popular tasks and models, which is often insufficient in previous work.

Experimental results show that new variants can eliminate the generalization gap between adaptive methods and SGD and maintain higher learning speed early in training at the same time.

Moreover, they can bring significant improvement over their prototypes, especially on complex deep networks.

The implementation of the algorithm can be found at https://github.com/Luolc/AdaBound .

There has been tremendous progress in first-order optimization algorithms for training deep neural networks.

One of the most dominant algorithms is stochastic gradient descent (SGD) BID15 , which performs well across many applications in spite of its simplicity.

However, there is a disadvantage of SGD that it scales the gradient uniformly in all directions.

This may lead to poor performance as well as limited training speed when the training data are sparse.

To address this problem, recent work has proposed a variety of adaptive methods that scale the gradient by square roots of some form of the average of the squared values of past gradients.

Examples of such methods include ADAM BID7 , ADAGRAD BID2 and RMSPROP BID16 .

ADAM in particular has become the default algorithm leveraged across many deep learning frameworks due to its rapid training speed BID17 .Despite their popularity, the generalization ability and out-of-sample behavior of these adaptive methods are likely worse than their non-adaptive counterparts.

Adaptive methods often display faster progress in the initial portion of the training, but their performance quickly plateaus on the unseen data (development/test set) BID17 .

Indeed, the optimizer is chosen as SGD (or with momentum) in several recent state-of-the-art works in natural language processing and computer vision BID11 BID18 , wherein these instances SGD does perform better than adaptive methods.

BID14 have recently proposed a variant of ADAM called AMSGRAD, hoping to solve this problem.

The authors provide a theoretical guarantee of convergence but only illustrate its better performance on training data.

However, the generalization ability of AMSGRAD on unseen data is found to be similar to that of ADAM while a considerable performance gap still exists between AMSGRAD and SGD BID6 BID1 .In this paper, we first conduct an empirical study on ADAM and illustrate that both extremely large and small learning rates exist by the end of training.

The results correspond with the perspective pointed out by BID17 that the lack of generalization performance of adaptive methods may stem from unstable and extreme learning rates.

In fact, introducing non-increasing learning rates, the key point in AMSGRAD, may help abate the impact of huge learning rates, while it neglects possible effects of small ones.

We further provide an example of a simple convex optimization problem to elucidate how tiny learning rates of adaptive methods can lead to undesirable non-convergence.

In such settings, RMSPROP and ADAM provably do not converge to an optimal solution, and furthermore, however large the initial step size ?? is, it is impossible for ADAM to fight against the scale-down term.

Based on the above analysis, we propose new variants of ADAM and AMSGRAD, named AD-ABOUND and AMSBOUND, which do not suffer from the negative impact of extreme learning rates.

We employ dynamic bounds on learning rates in these adaptive methods, where the lower and upper bound are initialized as zero and infinity respectively, and they both smoothly converge to a constant final step size.

The new variants can be regarded as adaptive methods at the beginning of training, and they gradually and smoothly transform to SGD (or with momentum) as time step increases.

In this framework, we can enjoy a rapid initial training process as well as good final generalization ability.

We provide a convergence analysis for the new variants in the convex setting.

We finally turn to an empirical study of the proposed methods on various popular tasks and models in computer vision and natural language processing.

Experimental results demonstrate that our methods have higher learning speed early in training and in the meantime guarantee strong generalization performance compared to several adaptive and non-adaptive methods.

Moreover, they can bring considerable improvement over their prototypes especially on complex deep networks.

Notations Given a vector ?? ??? R d we denote its i-th coordinate by ?? i ; we use ?? k to denote elementwise power of k and ?? to denote its 2 -norm; for a vector ?? t in the t-th iteration, the i-th coordinate of ?? t is denoted as ?? t,i by adding a subscript i.

Given two vectors v, w ??? R d , we use v, w to denote their inner product, v w to denote element-wise product, v/w to denote element-wise division, max(v, w) to denote element-wise maximum and min(v, w) to denote element-wise minimum.

We use S d + to denote the set of all positive definite d ?? d matrices.

For a vector a ??? R d and a positive definite matrix M ??? R d??d , we use a/M to denote M ???1 a and DISPLAYFORM0 Online convex programming A flexible framework to analyze iterative optimization methods is the online optimization problem.

It can be formulated as a repeated game between a player (the algorithm) and an adversary.

At step t, the algorithm chooses an decision x t ??? F, where F ??? R d is a convex feasible set.

Then the adversary chooses a convex loss function f t and the algorithm incurs loss f t (x t ).

The difference between the total loss T t=1 f t (x t ) and its minimum value for a fixed decision is known as the regret, which is represented by DISPLAYFORM1 Throughout this paper, we assume that the feasible set F has bounded diameter and ???f t (x) ??? is bounded for all t ??? [T ] and x ??? F. We are interested in algorithms with little regret.

Formally speaking, our aim is to devise an algorithm that ensures R T = o(T ), which implies that on average, the model's performance converges to the optimal one.

It has been pointed out that an online optimization algorithm with vanishing average regret yields a corresponding stochastic optimization algorithm BID0 .

Thus, following BID14 , we use online gradient descent and stochastic gradient descent synonymously.

A generic overview of optimization methods We follow BID14 to provide a generic framework of optimization methods in Algorithm 1 that encapsulates many popular adaptive and non-adaptive methods.

This is useful for understanding the properties of different optimization methods.

Note that the algorithm is still abstract since the functions ?? t : F t ??? R d and ?? t : DISPLAYFORM2 + have not been specified.

In this paper, we refer to ?? as initial step size and ?? t / ??? V t as Algorithm 1 Generic framework of optimization methods Input: x 1 ??? F, initial step size ??, sequence of functions {?? t , ?? t }

1: for t = 1 to T do 2: DISPLAYFORM0 Vt (x t+1 ) 7: end for learning rate of the algorithm.

Note that we employ a design of decreasing step size by ?? t = ??/ ??? t for it is required for theoretical proof of convergence.

However such an aggressive decay of step size typically translates into poor empirical performance, while a simple constant step size ?? t = ?? usually works well in practice.

For the sake of clarity, we will use the decreasing scheme for theoretical analysis and the constant schemem for empirical study in the rest of the paper.

Under such a framework, we can summarize the popular optimization methods in Table 1.

1 A few remarks are in order.

We can see the scaling term ?? t is I in SGD(M), while adaptive methods introduce different kinds of averaging of the squared values of past gradients.

ADAM and RMSPROP can be seen as variants of ADAGRAD, where the former ones use an exponential moving average as function ?? t instead of the simple average used in ADAGRAD.

In particular, RMSPROP is essentially a special case of ADAM with ?? 1 = 0.

AMSGRAD is not listed in the table as it does not has a simple expression of ?? t .

It can be defined as ?? t = diag(v t ) wherev t is obtained by the following recursion: DISPLAYFORM1 The definition of ?? t is same with that of ADAM.

In the rest of the paper we will mainly focus on ADAM due to its generality but our arguments also apply to other similar adaptive methods such as RMSPROP and AMSGRAD.

Table 1 : An overview of popular optimization methods using the generic framework.

DISPLAYFORM2 3 THE NON-CONVERGENCE CAUSED BY EXTREME LEARNING RATEIn this section, we elaborate the primary defect in current adaptive methods with a preliminary experiment and a rigorous proof.

As mentioned above, adaptive methods like ADAM are observed to perform worse than SGD.

BID14 proposed AMSGRAD to solve this problem but recent work has pointed out AMSGRAD does not show evident improvement over ADAM BID6 BID1 .

Since AMSGRAD is claimed to have a smaller learning rate compared with ADAM, the authors only consider large learning rates as the cause for bad performance of ADAM.

However, small ones might be a pitfall as well.

Thus, we speculate both extremely large and small learning rates of ADAM are likely to account for its ordinary generalization ability.

For corroborating our speculation, we sample learning rates of several weights and biases of ResNet-34 on CIFAR-10 using ADAM.

Specifically, we randomly select nine 3 ?? 3 convolutional kernels from different layers and the biases in the last linear layer.

As parameters of the same layer usually have similar properties, here we only demonstrate learning rates of nine weights sampled from nine kernels respectively and one bias from the last layer by the end of training, and employ a heatmap to visualize them.

As shown in Figure 1 , we can find that when the model is close to convergence, learning rates are composed of tiny ones less than 0.01 as well as huge ones greater than 1000.w1 w2 w3 w4 w5 w6 w7 w8 w9 b -5.8 -3.7 -3.4 -3.7 4.5 -3 8.6 2 -1.6 -4Figure 1: Learning rates of sampled parameters.

Each cell contains a value obtained by conducting a logarithmic operation on the learning rate.

The lighter cell stands for the smaller learning rate.

The above analysis and observation show that there are indeed learning rates which are too large or too small in the final stage of the training process.

AMSGRAD may help abate the impact of huge learning rates, but it neglects the other side of the coin.

Insofar, we still have the following two doubts.

First, does the tiny learning rate really do harm to the convergence of ADAM?

Second, as the learning rate highly depends on the initial step size, can we use a relatively larger initial step size ?? to get rid of too small learning rates?To answer these questions, we show that undesirable convergence behavior for ADAM and RM-SPROP can be caused by extremely small learning rates, and furthermore, in some cases no matter how large the initial step size ?? is, ADAM will still fail to find the right path and converge to some highly suboptimal points.

Consider the following sequence of linear functions for F = [???1, 1]: DISPLAYFORM3 where C ??? N satisfies: 5?? DISPLAYFORM4 For this function sequence, it is easy to see that the point x = ???1 provides the minimum regret.

Supposing ?? 1 = 0, we show that ADAM converges to a highly suboptimal solution of x ??? 0 for this setting.

Intuitively, the reasoning is as follows.

The algorithm obtains a gradient ???1 once every C steps, which moves the algorithm in the wrong direction.

Then, at the next step it observes a gradient 2.

But the larger gradient 2 is unable to counteract the effect to wrong direction since the learning rate at this step is scaled down to a value much less than the previous one, and hence x becomes larger and larger as the time step increases.

We formalize this intuition in the result below.

Theorem 1.

There is an online convex optimization problem where for any initial step size ??, ADAM has non-zero average regret i.e., R T /T 0 as T ??? ???.We relegate all proofs to the appendix.

Note that the above example also holds for constant step size ?? t = ??.

Also note that vanilla SGD does not suffer from this problem.

There is a wide range of valid choices of initial step size ?? where the average regret of SGD asymptotically goes to 0, in other words, converges to the optimal solution.

This problem can be more obvious in the later stage of a training process in practice when the algorithm gets stuck in some suboptimal points.

In such cases, gradients at most steps are close to 0 and the average of the second order momentum may be highly various due to the property of exponential moving average.

Therefore, "correct" signals which appear with a relatively low frequency (i.e. gradient 2 every C steps in the above example) may not be able to lead the algorithm to a right path, if they come after some "wrong" signals (i.e. gradient 1 in the example), even though the correct ones have larger absolute value of gradients.

One may wonder if using large ?? 1 helps as we usually use ?? 1 close to 1 in practice.

However, the following result shows that for any constant ?? 1 and ?? 2 with ?? 1 < ??? ?? 2 , there exists an example where ADAM has non-zero average regret asymptotically regardless of the initial step size ??.

Theorem 2.

For any constant ?? 1 , ?? 2 ??? [0, 1) such that ?? 1 < ??? ?? 2 , there is an online convex optimization problem where for any initial step size ??, ADAM has non-zero average regret i.e., DISPLAYFORM5 Furthermore, a stronger result stands in the easier stochastic optimization setting.

Theorem 3.

For any constant ?? 1 , ?? 2 ??? [0, 1) such that ?? 1 < ??? ?? 2 , there is a stochastic convex optimization problem where for any initial step size ??, ADAM does not converge to the optimal solution.

Remark.

The analysis of ADAM in BID7 relies on decreasing ?? 1 over time, while here we use constant ?? 1 .

Indeed, since the critical parameter is ?? 2 rather than ?? 1 in our analysis, it is quite easy to extend our examples to the case using decreasing scheme of ?? 1 .As mentioned by BID14 , the condition ?? 1 < ??? ?? 2 is benign and is typically satisfied in the parameter settings used in practice.

Such condition is also assumed in convergence proof of BID7 .

The above results illustrate the potential bad impact of extreme learning rates and algorithms are unlikely to achieve good generalization ability without solving this problem.

In this section we develop new variants of optimization methods and provide their convergence analysis.

Our aim is to devise a strategy that combines the benefits of adaptive methods, viz.

fast initial progress, and the good final generalization properties of SGD.

Intuitively, we would like to construct an algorithm that behaves like adaptive methods early in training and like SGD at the end.

DISPLAYFORM0 Inspired by gradient clipping, a popular technique used in practice that clips the gradients larger than a threshold to avoid gradient explosion, we employ clipping on learning rates in ADAM to propose ADABOUND in Algorithm 2.

Consider applying the following operation in ADAM DISPLAYFORM1 which clips the learning rate element-wisely such that the output is constrained to be in [?? l , ?? u ].

It follows that SGD(M) with ?? = ?? * can be considered as the case where ?? l = ?? u = ?? * .

As for ADAM, ?? l = 0 and ?? u = ???. Now we can provide the new strategy with the following steps.

We employ ?? l and ?? u as functions of t instead of constant lower and upper bound, where ?? l (t) is a non-decreasing function that starts from 0 as t = 0 and converges to ?? * asymptotically; and ?? u (t) is a non-increasing function that starts from ??? as t = 0 and also converges to ?? * asymptotically.

In this setting, ADABOUND behaves just like ADAM at the beginning as the bounds have very little impact on learning rates, and it gradually transforms to SGD(M) as the bounds become more and more restricted.

We prove the following key result for ADABOUND.

Theorem 4.

Let {x t } and {v t } be the sequences obtained from Algorithm 2, DISPLAYFORM0 and x ??? F. For x t generated using the ADABOUND algorithm, we have the following bound on the regret DISPLAYFORM1 The following result falls as an immediate corollary of the above result.

Corollary 4.1.

Suppose ?? 1t = ?? 1 ?? t???1 in Theorem 4, we have DISPLAYFORM2 It is easy to see that the regret of ADABOUND is upper bounded by O( et al. (2018) , one can use a much more modest momentum decay of ?? 1t = ?? 1 /t and still ensure a regret of O( ??? T ).

It should be mentioned that one can also incorporate the dynamic bound in AMSGRAD.

The resulting algorithm, namely AMSBOUND, also holds a regret of O( ??? T ) and the proof of convergence is almost same to Theorem 4 (see Appendix F for details).

In next section we will see that AMSBOUND has similar performance to ADABOUND in several well-known tasks.

DISPLAYFORM3 We end this section with a comparison to the previous work.

For the idea of transforming ADAM to SGD, there is a similar work by BID6 .

The authors propose a measure that uses ADAM at first and switches the algorithm to SGD at some specific step.

Compared with their approach, our methods have two advantages.

First, whether there exists a fixed turning point to distinguish ADAM and SGD is uncertain.

So we address this problem with a continuous transforming procedure rather than a "hard" switch.

Second, they introduce an extra hyperparameter to decide the switching time, which is not very easy to fine-tune.

As for our methods, the flexible parts introduced are two bound functions.

We conduct an empirical study of the impact of different kinds of bound functions.

The results are placed in Appendix G for we find that the convergence target ?? * and convergence speed are not very important to the final results.

For the sake of clarity, we will use ?? l (t) = 0.1??? 0.1 (1?????2)t+1 and ?? u (t) = 0.1+ 0.1(1?????2)t in the rest of the paper unless otherwise specified.

In this section, we turn to an empirical study of different models to compare new variants with popular optimization methods including SGD(M), ADAGRAD, ADAM, and AMSGRAD.

We focus on three tasks: the MNIST image classification task BID9 , the CIFAR-10 image classification task BID8 , and the language modeling task on Penn Treebank BID12 .

We choose them due to their broad importance and availability of their architectures for reproducibility.

The setup for each task is detailed in TAB0 .

We run each experiment three times with the specified initialization method from random starting points.

A fixed budget on the number of epochs is assigned for training and the decay strategy is introduced in following parts.

We choose the settings that achieve the lowest training loss at the end.

Optimization hyperparameters can exert great impact on ultimate solutions found by optimization algorithms so here we describe how we tune them.

To tune the step size, we follow the method in BID17 .

We implement a logarithmically-spaced grid of five step sizes.

If the best performing parameter is at one of the extremes of the grid, we will try new grid points so that the best performing parameters are at one of the middle points in the grid.

Specifically, we tune over hyperparameters in the following way.

For tuning the step size of SGD(M), we first coarsely tune the step size on a logarithmic scale from {100, 10, 1, 0.1, 0.01} and then fine-tune it.

Whether the momentum is used depends on the specific model but we set the momentum parameter to default value 0.9 for all our experiments.

We find this strategy effective given the vastly different scales of learning rates needed for different modalities.

For instance, SGD with ?? = 10 performs best for language modeling on PTB but for the ResNet-34 architecture on CIFAR-10, a learning rate of 0.1 for SGD is necessary.

ADAGRAD The initial set of step sizes used for ADAGRAD are: {5e-2, 1e-2, 5e-3, 1e-3, 5e-4}. For the initial accumulator value, we choose the recommended value as 0.ADAM & AMSGRAD We employ the same hyperparameters for these two methods.

The initial step sizes are chosen from: {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}. We turn over ?? 1 values of {0.9, 0.99} and ?? 2 values of {0.99, 0.999}. We use for the perturbation value = 1e-8.ADABOUND & AMSBOUND We directly apply the default hyperparameters for ADAM (a learning rate of 0.001, ?? 1 = 0.9 and ?? 2 = 0.999) in our proposed methods.

Note that for other hyperparameters such as batch size, dropout probability, weight decay and so on, we choose them to match the recommendations of the respective base architectures.

We train a simple fully connected neural network with one hidden layer for the multiclass classification problem on MNIST dataset.

We run 100 epochs and omit the decay scheme for this experiment.

FIG0 shows the learning curve for each optimization method on both the training and test set.

We find that for training, all algorithms can achieve the accuracy approaching 100%.

For the test part, SGD performs slightly better than adaptive methods ADAM and AMSGRAD.

Our two proposed methods, ADABOUND and AMSBOUND, display slight improvement, but compared with their prototypes there are still visible increases in test accuracy.

Using DenseNet-121 BID5 and ResNet-34 BID3 , we then consider the task of image classification on the standard CIFAR-10 dataset.

In this experiment, we employ the fixed budget of 200 epochs and reduce the learning rates by 10 after 150 epochs.

DenseNet We first run a DenseNet-121 model on CIFAR-10 and our results are shown in FIG1 .

We can see that adaptive methods such as ADAGRAD, ADAM and AMSGRAD appear to perform better than the non-adaptive ones early in training.

But by epoch 150 when the learning rates are decayed, SGDM begins to outperform those adaptive methods.

As for our methods, ADABOUND and AMSBOUND, they converge as fast as adaptive ones and achieve a bit higher accuracy than SGDM on the test set at the end of training.

In addition, compared with their prototypes, their performances are enhanced evidently with approximately 2% improvement in the test accuracy.

ResNet Results for this experiment are reported in FIG1 .

As is expected, the overall performance of each algorithm on ResNet-34 is similar to that on DenseNet-121.

ADABOUND and AMSBOUND even surpass SGDM by 1%.

Despite the relative bad generalization ability of adaptive methods, our proposed methods overcome this drawback by allocating bounds for their learning rates and obtain almost the best accuracy on the test set for both DenseNet and ResNet on CIFAR-10.

Finally, we conduct an experiment on the language modeling task with Long Short-Term Memory (LSTM) network BID4 .

From two experiments above, we observe that our methods show much more improvement in deep convolutional neural networks than in perceptrons.

Therefore, we suppose that the enhancement is related to the complexity of the architecture and run three models with (L1) 1-layer, (L2) 2-layer and (L3) 3-layer LSTM respectively.

We train them on Penn Treebank, running for a fixed budget of 200 epochs.

We use perplexity as the metric to evaluate the performance and report results in We find that in all models, ADAM has the fastest initial progress but stagnates in worse performance than SGD and our methods.

Different from phenomena in previous experiments on the image classification tasks, ADABOUND and AMSBOUND does not display rapid speed at the early training stage but the curves are smoother than that of SGD.Comparing L1, L2 and L3, we can easily notice a distinct difference of the improvement degree.

In L1, the simplest model, our methods perform slightly 1.1% better than ADAM while in L3, the most complex model, they show evident improvement over 2.8% in terms of perplexity.

It serves as evidence for the relationship between the model's complexity and the improvement degree.

To investigate the efficacy of our proposed algorithms, we select popular tasks from computer vision and natural language processing.

Based on results shown above, it is easy to find that ADAM and AMSGRAD usually perform similarly and the latter does not show much improvement for most cases.

Their variants, ADABOUND and AMSBOUND, on the other hand, demonstrate a fast speed of convergence compared with SGD while they also exceed two original methods greatly with respect to test accuracy at the end of training.

This phenomenon exactly confirms our view mentioned in Section 3 that both large and small learning rates can influence the convergence.

Besides, we implement our experiments on models with different complexities, consisting of a perceptron, two deep convolutional neural networks and a recurrent neural network.

The perceptron used on the MNIST is the simplest and our methods perform slightly better than others.

As for DenseNet and ResNet, obvious increases in test accuracy can be observed.

We attribute this difference to the complexity of the model.

Specifically, for deep CNN models, convolutional and fully connected layers play different parts in the task.

Also, different convolutional layers are likely to be responsible for different roles BID10 , which may lead to a distinct variation of gradients of parameters.

In other words, extreme learning rates (huge or tiny) may appear more frequently in complex models such as ResNet.

As our algorithms are proposed to avoid them, the greater enhancement of performance in complex architectures can be explained intuitively.

The higher improvement degree on LSTM with more layers on language modeling task also consists with the above analysis.

Despite superior results of our methods, there still remain several problems to explore.

For example, the improvement on simple models are not very inspiring, we can investigate how to achieve higher improvement on such models.

Besides, we only discuss reasons for the weak generalization ability of adaptive methods, however, why SGD usually performs well across diverse applications of machine learning still remains uncertain.

Last but not least, applying dynamic bounds on learning rates is only one particular way to conduct gradual transformation from adaptive methods to SGD.

There might be other ways such as well-designed decay that can also work, which remains to explore.

We investigate existing adaptive algorithms and find that extremely large or small learning rates can result in the poor convergence behavior.

A rigorous proof of non-convergence for ADAM is provided to demonstrate the above problem.

Motivated by the strong generalization ability of SGD, we design a strategy to constrain the learning rates of ADAM and AMSGRAD to avoid a violent oscillation.

Our proposed algorithms, AD-ABOUND and AMSBOUND, which employ dynamic bounds on their learning rates, achieve a smooth transition to SGD.

They show the great efficacy on several standard benchmarks while maintaining advantageous properties of adaptive methods such as rapid initial progress and hyperparameter insensitivity.

We thank all reviewers for providing the constructive suggestions.

We also thank Junyang Lin and Ruixuan Luo for proofreading and doing auxiliary experiments.

Xu Sun is the corresponding author of this paper.

Lemma 1 (Mcmahan & Streeter FORMULA17 ).

For any Q ??? S d + and convex feasible set F ??? R d , suppose DISPLAYFORM0 Proof.

We provide the proof here for completeness.

Since u 1 = min x???F Q 1/2 (x ??? z 1 ) and u 2 = min x???F Q 1/2 (x ??? z 2 ) and from the property of projection operator we have the following: DISPLAYFORM1 Combining the above inequalities, we have DISPLAYFORM2 Also, observe the following: DISPLAYFORM3 The above inequality can be obtained from the fact that DISPLAYFORM4 and rearranging the terms.

Combining the above inequality with Equation FORMULA17 , we have the required the result.

Lemma 2.

Suppose m t = ?? 1 m t???1 + (1 ??? ?? 1 )g t with m 0 = 0 and 0 ??? ?? 1 < 1.

We have DISPLAYFORM5 Proof.

If ?? 1 = 0, the equality directly holds due to m t = g t .

Otherwise, 0 < ?? 1 < 1.

For any ?? > 0 we have DISPLAYFORM6 The inequality follows from Cauchy-Schwarz and Young's inequality.

In particular, let ?? = 1/?? 1 ??? 1.

Then we have DISPLAYFORM7 Dividing both sides by ?? t 1 , we get DISPLAYFORM8 Then multiplying both sides by ?? t 1 we obtain DISPLAYFORM9 Published as a conference paper at ICLR 2019Take the summation of above inequality over t = 1, 2, ?? ?? ?? , T , we have DISPLAYFORM10 The second inequality is due to the following fact of geometric series DISPLAYFORM11 We complete the proof.

Proof.

First, we rewrite the update of ADAM in Algorithm 1 in the following recursion form: DISPLAYFORM0 where m 0,i = 0 and v 0,i = 0 for all i ??? [d] and ?? t = diag(v t ).

We consider the setting where f t are linear functions and F = [???1, 1].

In particular, we define the following function sequence: DISPLAYFORM1 for t mod C = 1; 2x, for t mod C = 2; 0, otherwise where C ??? N satisfies the following: DISPLAYFORM2 It is not hard to see that the condition hold for large constant C that depends on ?? 2 .Since the problem is one-dimensional, we drop indices representing coordinates from all quantities in Algorithm 1.

For this function sequence, it is easy to see that the point x = ???1 provides the minimum regret.

Consider the execution of ADAM algorithm for this sequence of functions with ?? 1 = 0.

Note that since gradients of these functions are bounded, F has bounded D ??? diameter and ?? 2 1 /?? 2 < 1 as ?? 1 = 0, the conditions on the parameters required for ADAM are satisfied BID7 .

The gradients have the following form: DISPLAYFORM3 for i mod C = 2; 0, otherwise.

Let ?? ??? N, ?? > 1 be such that DISPLAYFORM4 DISPLAYFORM5 for all t ??? ?? .

We start with the following preliminary result.

Lemma 3.

For the parameter settings and conditions assumed in Theorem 1, there is a t ??? ?? such that x Ct +1 ??? 0.Proof by contradiction.

Assume that x Ct+1 < 0 for all t ??? ?? .

Firstly, for t ??? ?? , we observe the following inequalities: DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 From the (C?? + 1)-th update of ADAM in Equation (2), we obtain: DISPLAYFORM9 The first inequality follows from x Ct+1 < 0 and Equation (6).

The last inequality follows from Equation (4).

Therefore, we have ???1 ??? x C?? +1 <x C?? +2 < 1 and hence x C?? +2 =x C?? +2 .

Then after the (C?? + 2)-th update, we have: DISPLAYFORM10 where ?? = ???? 2 (1 ??? ?? 2 )/162 ??? C is a constant that depends on ??, ?? 2 and C. The first inequality follows from Equation (2).

The second inequality follows from Equations FORMULA34 and (8).

The last inequality is due to the following lower bound: DISPLAYFORM11 where the last inequality follows from Equation (3).

Therefore, we have ???1 ??? x C?? +1 <x C?? +3 < x C?? +2 < 1.

Furthermore, since gradients ???f i (x) = 0 when i mod C = 1 or 2, we have DISPLAYFORM12 Then, following Equation FORMULA37 we have DISPLAYFORM13 Similarly, we can subsequently obtain DISPLAYFORM14 and generally DISPLAYFORM15 for t ??? ?? .

Let t be such that 2??( DISPLAYFORM16 This contradicts the assumption that x Ct+1 < 0 for all t ??? ?? .

We complete the proof of this lemma.

We now return to the proof of Theorem 1.

The following analysis focuses on iterations after Ct + 1 such that x Ct +1 ??? 0.

Note that any regret before Ct + 1 is just a constant since t is independent of T and thus, the average regret is negligible as T ??? ???.Our claim is that, x k ??? 0 for all k ??? N, k ??? Ct + 1.

To prove this, we resort to the principle of mathematical induction.

Suppose for some t ??? N, t ??? t , we have x Ct+1 ??? 0.

Our aim is to prove that x i ??? 0 for all i ??? N ??? [Ct + 2, C(t + 1) + 1].From the (Ct + 1)-th update of ADAM in Equation FORMULA27 , we obtain: DISPLAYFORM17 We consider the following two cases:1.

Supposex Ct+2 > 1, then x Ct+2 = ?? F (x Ct+2 ) = min{x Ct+2 , 1} = 1 (note that in one-dimension, ?? F , ??? Vt = ?? F is the simple Euclidean projection).

After the (Ct + 2)-th update, we have: DISPLAYFORM18 The last inequality follows from Equation (5).

The first inequality follows from DISPLAYFORM19 2.

Supposex Ct+2 ??? 1, then after the (Ct + 2)-th update, similar to Equation FORMULA37 , we have: DISPLAYFORM20 In both cases,x Ct+3 ??? 0, which translates to x Ct+3 =x Ct+3 ??? 0.

Furthermore, since gradients ???f i (x) = 0 when i mod C = 1 or 2, we have DISPLAYFORM21 Therefore, given x Ct +1 = 0, it holds for all k ??? N, k ??? Ct + 1 by the principle of mathematical induction.

Thus, we have DISPLAYFORM22 where k ??? N, k ??? t .

Therefore, when t ??? t , for every C steps, ADAM suffers a regret of at least 1.

More specifically, R T ??? (T ???t )/C.

Thus, R T /T 0 as T ??? ???, which completes the proof.

Theorem 2 generalizes the optimization setting used in Theorem 1.

We notice that the example proposed by BID14 in their Appendix B already satisfies the constraints listed in Theorem 2.Here we provide the setting of the example for completeness.

Proof.

Consider the setting where f t are linear functions and F = [???1, 1].

In particular, we define the following function sequence: DISPLAYFORM0 where C ??? N, C mod 2 = 0 satisfies the following: DISPLAYFORM1 where ?? = ?? 1 / ??? ?? 2 < 1.

It is not hard to see that these conditions hold for large constant C that depends on ?? 1 and ?? 2 .

According to the proof given by BID14 in their Appendix B, in such a setting R T /T 0 as T ??? ???, which completes the proof.

The example proposed by BID14 in their Appendix C already satisfies the constraints listed in Theorem 3.

Here we provide the setting of the example for completeness.

Proof.

Let ?? be an arbitrary small positive constant.

Consider the following one dimensional stochastic optimization setting over the domain [???1, 1].

At each time step t, the function f t (x) is chosen as follows: DISPLAYFORM0 where C is a large constant that depends on ?? 1 , ?? 2 and ??.

The expected function is F (x) = ??x.

Thus the optimal point over [???1, 1] is x * = ???1.

The step taken by ADAM is DISPLAYFORM1 According to the proof given by BID14 in their Appendix C, there exists a large enough C such that E[??? t ] ??? 0, which then implies that the ADAM's step keep drifting away from the optimal solution x * = ???1.

Note that there is no limitation of the initial step size ?? by now.

Therefore, we complete the proof.

Proof.

Let x * = arg min x???F T t=1 f t (x), which exists since F is closed and convex.

We begin with the following observation: DISPLAYFORM0 Using Lemma 1 with u 1 = x t+1 and u 2 = x * , we have the following: DISPLAYFORM1 Rearranging the above inequality, we have DISPLAYFORM2 (10) DISPLAYFORM3 The second inequality use the fact that ?? 1t ??? ?? 1 < 1.

In order to further simplify the bound in Equation FORMULA17 , we need to use telescopic sum.

We observe that, by definition of ?? t , we have DISPLAYFORM4 t???1,i .

Using the D ??? bound on the feasible region and making use of the above property in Equation (12), we have DISPLAYFORM5 The equality follows from simple telescopic sum, which yields the desired result.

It is easy to see that the regret of ADABOUND is upper bounded by O( ??? T ).

Theorem 5.

Let {x t } and {v t } be the sequences obtained from Algorithm 3, ?? 1 = ?? 11 , ?? 1t ??? ?? 1 for all t ??? [T ] and ?? 1 / ??? ?? 2 < 1.

Suppose ?? l (t + 1) ??? ?? l (t) > 0, ?? u (t + 1) ??? ?? u (t), ?? l (t) ??? ?? * as t ??? ???, ?? u (t) ??? ?? * as t ??? ???, L ??? = ?? l (1) and R ??? = ?? u (1).

Assume that x ??? y ??? ??? D ??? for all x, y ??? F and ???f t (x) ??? G 2 for all t ??? [T ] and x ??? F. For x t generated using the ADABOUND algorithm, we have the following bound on the regret We further directly compare the performance between SGDM and ADABOUND with each ?? (or ?? * ).

The results are shown in Figure 7 .

We can see that ADABOUND outperforms SGDM for all the step sizes.

Since the form of bound functions has minor impact on the performance of ADABOUND, it is likely to beat SGDM even without carefully tuning the hyperparameters.

DISPLAYFORM0 To summarize, the form of bound functions does not much influence the final performance of the methods.

In other words, ADABOUND is not sensitive to its hyperparameters.

Moreover, it can achieve a higher or similar performance to SGDM even if it is not carefully fine-tuned.

Therefore, we can expect a better performance by using ADABOUND regardless of the choice of bound functions.

Here we provide an empirical study on the evolution of learning rates of ADABOUND over time.

We conduct an experiment using ResNet-34 model on CIFAR-10 dataset with the same settings in Section 5.

We randomly choose two layers in the network.

For each layer, the learning rates of its parameters are recorded at each time step.

We pick the min/median/max values of the learning rates in each layer and plot them against epochs in FIG4 .We can see that the learning rates increase rapidly in the early stage of training, then after a few epochs its max/median values gradually decrease over time, and finally converge to the final step size.

The increasing at the beginning is due to the property of the exponential moving average of ?? t of ADAM, while the gradually decreasing indicates the transition from ADAM to SGD.

<|TLDR|>

@highlight

Novel variants of optimization methods that combine the benefits of both adaptive and non-adaptive methods.