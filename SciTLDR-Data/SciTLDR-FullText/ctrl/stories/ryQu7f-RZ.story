Several recently proposed stochastic optimization methods that have been successfully used in training deep networks such as RMSProp, Adam, Adadelta, Nadam are based on using gradient updates scaled by square roots of exponential moving averages of squared past gradients.

In many applications, e.g. learning with large output spaces, it has been empirically observed that these algorithms fail to converge to an optimal solution (or a critical point in nonconvex settings).

We show that one cause for such failures is the exponential moving average used in the algorithms.

We provide an explicit example of a simple convex optimization setting where Adam does not converge to the optimal solution, and describe the precise problems with the previous analysis of Adam algorithm.

Our analysis suggests that the convergence issues can be fixed by endowing such algorithms with ``long-term memory'' of past gradients, and propose new variants of the Adam algorithm which not only fix the convergence issues but often also lead to improved empirical performance.

Stochastic gradient descent (SGD) is the dominant method to train deep networks today.

This method iteratively updates the parameters of a model by moving them in the direction of the negative gradient of the loss evaluated on a minibatch.

In particular, variants of SGD that scale coordinates of the gradient by square roots of some form of averaging of the squared coordinates in the past gradients have been particularly successful, because they automatically adjust the learning rate on a per-feature basis.

The first popular algorithm in this line of research is ADAGRAD BID2 BID5 , which can achieve significantly better performance compared to vanilla SGD when the gradients are sparse, or in general small.

Although ADAGRAD works well for sparse settings, its performance has been observed to deteriorate in settings where the loss functions are nonconvex and gradients are dense due to rapid decay of the learning rate in these settings since it uses all the past gradients in the update.

This problem is especially exacerbated in high dimensional problems arising in deep learning.

To tackle this issue, several variants of ADAGRAD, such as RMSPROP BID7 , ADAM BID3 , ADADELTA (Zeiler, 2012) , NADAM BID1 , etc, have been proposed which mitigate the rapid decay of the learning rate using the exponential moving averages of squared past gradients, essentially limiting the reliance of the update to only the past few gradients.

While these algorithms have been successfully employed in several practical applications, they have also been observed to not converge in some other settings.

It has been typically observed that in these settings some minibatches provide large gradients but only quite rarely, and while these large gradients are quite informative, their influence dies out rather quickly due to the exponential averaging, thus leading to poor convergence.

In this paper, we analyze this situation in detail.

We rigorously prove that the intuition conveyed in the above paragraph is indeed correct; that limiting the reliance of the update on essentially only the past few gradients can indeed cause significant convergence issues.

In particular, we make the following key contributions:??? We elucidate how the exponential moving average in the RMSPROP and ADAM algorithms can cause non-convergence by providing an example of simple convex optimization prob-lem where RMSPROP and ADAM provably do not converge to an optimal solution.

Our analysis easily extends to other algorithms using exponential moving averages such as ADADELTA and NADAM as well, but we omit this for the sake of clarity.

In fact, the analysis is flexible enough to extend to other algorithms that employ averaging squared gradients over essentially a fixed size window (for exponential moving averages, the influences of gradients beyond a fixed window size becomes negligibly small) in the immediate past.

We omit the general analysis in this paper for the sake of clarity.??? The above result indicates that in order to have guaranteed convergence the optimization algorithm must have "long-term memory" of past gradients.

Specifically, we point out a problem with the proof of convergence of the ADAM algorithm given by BID3 .

To resolve this issue, we propose new variants of ADAM which rely on long-term memory of past gradients, but can be implemented in the same time and space requirements as the original ADAM algorithm.

We provide a convergence analysis for the new variants in the convex setting, based on the analysis of BID3 , and show a datadependent regret bound similar to the one in ADAGRAD.??? We provide a preliminary empirical study of one of the variants we proposed and show that it either performs similarly, or better, on some commonly used problems in machine learning.

Notation.

We use S is defined as arg min x???F A 1/2 (x ??? y) for y ??? R d .

Finally, we say F has bounded diameter D ??? if x ??? y ??? ??? D ??? for all x, y ??? F.

A flexible framework to analyze iterative optimization methods is the online optimization problem in the full information feedback setting.

In this online setup, at each time step t, the optimization algorithm picks a point (i.e. the parameters of the model to be learned) x t ??? F, where F ??? R d is the feasible set of points.

A loss function f t (to be interpreted as the loss of the model with the chosen parameters in the next minibatch) is then revealed, and the algorithm incurs loss f t (x t ).

The algorithm's regret at the end of T rounds of this process is given by DISPLAYFORM0 .

Throughout this paper, we assume that the feasible set F has bounded diameter and ???f t (x) ??? is bounded for all t ??? [T ] and x ??? F.Our aim to is to devise an algorithm that ensures R T = o(T ), which implies that on average, the model's performance converges to the optimal one.

The simplest algorithm for this setting is the standard online gradient descent algorithm (Zinkevich, 2003) , which moves the point x t in the opposite direction of the gradient g t = ???f t (x t ) while maintaining the feasibility by projecting onto the set F via the update rule x t+1 = ?? F (x t ??? ?? t g t ), where ?? F (y) denotes the projection of y ??? R d onto the set F i.e., ?? F (y) = min x???F x ??? y , and ?? t is typically set to ??/ ??? t for some constant ??.

The aforementioned online learning problem is closely related to the stochastic optimization problem: min x???F E z [f (x, z)], popularly referred to as empirical risk minimization (ERM), where z is a training example drawn training sample over which a model with parameters x is to be learned, and f (x, z) is the loss of the model with parameters x on the sample z. In particular, an online optimization algorithm with vanishing average regret yields a stochastic optimization algorithm for the ERM problem (Cesa-Bianchi et al., 2004) .

Thus, we use online gradient descent and stochastic gradient descent (SGD) synonymously.

Generic adaptive methods setup.

We now provide a framework of adaptive methods that gives us insights into the differences between different adaptive methods and is useful for understanding the flaws in a few popular adaptive methods.

Algorithm 1 provides a generic adaptive framework that encapsulates many popular adaptive methods.

Note the algorithm is still abstract because the

Input: x1 ??? F, step size {??t > 0} T t=1 , sequence of functions {??t, ??t} T t=1for t = 1 to T do gt = ???ft(xt) mt = ??t(g1, . . .

, gt) and Vt = ??t(g1, . . .

, gt) DISPLAYFORM0 "averaging" functions ?? t and ?? t have not been specified.

Here ?? t : DISPLAYFORM1 For ease of exposition, we refer to ?? t as step size and ?? t V ???1/2 t as learning rate of the algorithm and furthermore, restrict ourselves to diagonal variants of adaptive methods encapsulated by Algorithm 1 where V t = diag(v t ) .

We first observe that standard stochastic gradient algorithm falls in this framework by using: DISPLAYFORM2 and DISPLAYFORM3 .

While the decreasing step size is required for convergence, such an aggressive decay of learning rate typically translates into poor empirical performance.

The key idea of adaptive methods is to choose averaging functions appropriately so as to entail good convergence.

For instance, the first adaptive method ADAGRAD BID2 , which propelled the research on adaptive methods, uses the following averaging functions: DISPLAYFORM4 and step size ?? t = ??/ ??? t for all t ??? [T ].

In contrast to a learning rate of ??/ ??? t in SGD, such a setting effectively implies a modest learning rate decay of DISPLAYFORM5 When the gradients are sparse, this can potentially lead to huge gains in terms of convergence (see BID2 ).

These gains have also been observed in practice for even few non-sparse settings.

Adaptive methods based on Exponential Moving Averages.

Exponential moving average variants of ADAGRAD are popular in the deep learning community.

RMSPROP, ADAM, NADAM, and ADADELTA are some prominent algorithms that fall in this category.

The key difference is to use an exponential moving average as function ?? t instead of the simple average function used in ADAGRAD.

ADAM 1 , a particularly popular variant, uses the following averaging functions: DISPLAYFORM6 for some ?? 1 , ?? 2 ??? [0, 1).

This update can alternatively be stated by the following simple recursion: DISPLAYFORM7 and m 0,i = 0 and v 0,i = 0 for all i ??? [d].

and t ??? [T ].

A value of ?? 1 = 0.9 and ?? 2 = 0.999 is typically recommended in practice.

We note the additional projection operation in Algorithm 1 in comparison to ADAM.

When F = R d , the projection operation is an identity operation and this corresponds to the algorithm in BID3 .

For theoretical analysis, one requires ?? t = 1/ ??? t for t ??? [T ], although, a more aggressive choice of constant step size seems to work well in practice.

RMSPROP, which appeared in an earlier unpublished work BID7 is essentially a variant of ADAM with ?? 1 = 0.

In practice, especially in deep learning applications, the momentum term arising due to non-zero ?? 1 appears to significantly boost the performance.

We will mainly focus on ADAM algorithm due to this generality but our arguments also apply to RMSPROP and other algorithms such as ADADELTA, NADAM.

With the problem setup in the previous section, we discuss fundamental flaw in the current exponential moving average methods like ADAM.

We show that ADAM can fail to converge to an optimal solution even in simple one-dimensional convex settings.

These examples of non-convergence contradict the claim of convergence in BID3 , and the main issue lies in the following quantity of interest: DISPLAYFORM0 This quantity essentially measures the change in the inverse of learning rate of the adaptive method with respect to time.

One key observation is that for SGD and ADAGRAD, ?? t 0 for all t ??? [T ].

This simply follows from update rules of SGD and ADAGRAD in the previous section.

In particular, update rules for these algorithms lead to "non-increasing" learning rates.

However, this is not necessarily the case for exponential moving average variants like ADAM and RMSPROP i.e., ?? t can potentially be indefinite for t ??? [T ] .

We show that this violation of positive definiteness can lead to undesirable convergence behavior for ADAM and RMSPROP.

Consider the following simple sequence of linear functions for F = [???1, 1]: DISPLAYFORM1 where C > 2.

For this function sequence, it is easy to see that the point x = ???1 provides the minimum regret.

Suppose ?? 1 = 0 and ?? 2 = 1/(1 + C 2 ).

We show that ADAM converges to a highly suboptimal solution of x = +1 for this setting.

Intuitively, the reasoning is as follows.

The algorithm obtains the large gradient C once every 3 steps, and while the other 2 steps it observes the gradient ???1, which moves the algorithm in the wrong direction.

The large gradient C is unable to counteract this effect since it is scaled down by a factor of almost C for the given value of ?? 2 , and hence the algorithm converges to 1 rather than ???1.

We formalize this intuition in the result below.

Theorem 1.

There is an online convex optimization problem where ADAM has non-zero average regret i.e., R T /T 0 as T ??? ???.We relegate all proofs to the appendix.

A few remarks are in order.

One might wonder if adding a small constant in the denominator of the update helps in circumventing this problem i.e., the update for ADAM in Algorithm 1 ofx t+1 is modified as follows: DISPLAYFORM2 The algorithm in BID3 uses such an update in practice, although their analysis does not.

In practice, selection of the parameter appears to be critical for the performance of the algorithm.

However, we show that for any constant > 0, there exists an online optimization setting where, again, ADAM has non-zero average regret asymptotically (see Theorem 6 in Section F of the appendix).The above examples of non-convergence are catastrophic insofar that ADAM and RMSPROP converge to a point that is worst amongst all points in the set [???1, 1].

Note that above example also holds for constant step size ?? t = ??.

Also note that classic SGD and ADAGRAD do not suffer from this problem and for these algorithms, average regret asymptotically goes to 0.

This problem is especially aggravated in high dimensional settings and when the variance of the gradients with respect to time is large.

This example also provides intuition for why large ?? 2 is advisable while using ADAM algorithm, and indeed in practice using large ?? 2 helps.

However the following result shows that for any constant ?? 1 and ?? 2 with ?? 1 < ??? ?? 2 , we can design an example where ADAM has non-zero average rate asymptotically.

Theorem 2.

For any constant ?? 1 , ?? 2 ??? [0, 1) such that ?? 1 < ??? ?? 2 , there is an online convex optimization problem where ADAM has non-zero average regret i.e., R T /T 0 as T ??? ???.The above results show that with constant ?? 1 and ?? 2 , momentum or regularization via will not help in convergence of the algorithm to the optimal solution.

Note that the condition ?? 1 < ??? ?? 2 is benign and is typically satisfied in the parameter settings used in practice.

Furthermore, such condition is assumed in convergence proof of BID3 .

We can strengthen this result by providing a similar example of non-convergence even in the easier stochastic optimization setting: DISPLAYFORM3 end for Theorem 3.

For any constant ?? 1 , ?? 2 ??? [0, 1) such that ?? 1 < ??? ?? 2 , there is a stochastic convex optimization problem for which ADAM does not converge to the optimal solution.

These results have important consequences insofar that one has to use "problem-dependent" , ?? 1 and ?? 2 in order to avoid bad convergence behavior.

In high-dimensional problems, this typically amounts to using, unlike the update in Equation (3), a different , ?? 1 and ?? 2 for each dimension.

However, this defeats the purpose of adaptive methods since it requires tuning a large set of parameters.

We would also like to emphasize that while the example of non-convergence is carefully constructed to demonstrate the problems in ADAM, it is not unrealistic to imagine scenarios where such an issue can at the very least slow down convergence.

We end this section with the following important remark.

While the results stated above use constant ?? 1 and ?? 2 , the analysis of ADAM in BID3 actually relies on decreasing ?? 1 over time.

It is quite easy to extend our examples to the case where ?? 1 is decreased over time, since the critical parameter is ?? 2 rather than ?? 1 , and as long as ?? 2 is bounded away from 1, our analysis goes through.

Thus for the sake of clarity, in this paper we only prove non-convergence of ADAM in the setting where ?? 1 is held constant.

In this section, we develop a new principled exponential moving average variant and provide its convergence analysis.

Our aim is to devise a new strategy with guaranteed convergence while preserving the practical benefits of ADAM and RMSPROP.

To understand the design of our algorithms, let us revisit the quantity ?? t in (2).

For ADAM and RMSPROP, this quantity can potentially be negative.

The proof in the original paper of ADAM erroneously assumes that ?? t is positive semi-definite and is hence, incorrect (refer to Appendix D for more details).

For the first part, we modify these algorithms to satisfy this additional constraint.

Later on, we also explore an alternative approach where ?? t can be made positive semi-definite by using values of ?? 1 and ?? 2 that change with t.

AMSGRAD uses a smaller learning rate in comparison to ADAM and yet incorporates the intuition of slowly decaying the effect of past gradients on the learning rate as long as ?? t is positive semidefinite.

Algorithm 2 presents the pseudocode for the algorithm.

The key difference of AMSGRAD with ADAM is that it maintains the maximum of all v t until the present time step and uses this maximum value for normalizing the running average of the gradient instead of v t in ADAM.

By doing this, AMSGRAD results in a non-increasing step size and avoids the pitfalls of ADAM and RMSPROP i.e., ?? t 0 for all t ??? [T ] even with constant ?? 2 .

Also, in Algorithm 2, one typically uses a constant ?? 1t in practice (although, the proof requires a decreasing schedule for proving convergence of the algorithm).To gain more intuition for the updates of AMSGRAD, it is instructive to compare its update with ADAM and ADAGRAD.

Suppose at particular time step t and coordinate i ??? [d], we have v t???1,i > g 2 t,i > 0, then ADAM aggressively increases the learning rate, however, as we have seen in the previous section, this can be detrimental to the overall performance of the algorithm.

On the other hand, ADAGRAD slightly decreases the learning rate, which often leads to poor performance in practice since such an accumulation of gradients over a large time period can significantly decrease the learning rate.

In contrast, AMSGRAD neither increases nor decreases the learning rate and furthermore, decreases v t which can potentially lead to non-decreasing learning rate even if gradient is large in the future iterations.

For rest of the paper, we use g 1:t = [g 1 . . .

g t ] to denote the matrix obtained by concatenating the gradient sequence.

We prove the following key result for AMSGRAD.Theorem 4.

Let {x t } and {v t } be the sequences obtained from Algorithm 2, DISPLAYFORM0 and x ??? F. For x t generated using the AMSGRAD (Algorithm 2), we have the following bound on the regret DISPLAYFORM1 The following result falls as an immediate corollary of the above result.

Corollary 1.

Suppose ?? 1t = ?? 1 ?? t???1 in Theorem 4, then we have DISPLAYFORM2 The above bound can be considerably better than O( BID2 .

Furthermore, in Theorem 4, one can use a much more modest momentum decay of ?? 1t = ?? 1 /t and still ensure a regret of O( ??? T ).

We would also like to point out that one could consider taking a simple average of all the previous values of v t instead of their maximum.

The resulting algorithm is very similar to ADAGRAD except for normalization with smoothed gradients rather than actual gradients and can be shown to have similar convergence as ADAGRAD.

DISPLAYFORM3

In this section, we present empirical results on both synthetic and real-world datasets.

For our experiments, we study the problem of multiclass classification using logistic regression and neural networks, representing convex and nonconvex settings, respectively.

Synthetic Experiments: To demonstrate the convergence issue of ADAM, we first consider the following simple convex setting inspired from our examples of non-convergence: DISPLAYFORM0 with the constraint set F = [???1, 1].

We first observe that, similar to the examples of nonconvergence we have considered, the optimal solution is x = ???1; thus, for convergence, we expect the algorithms to converge to x = ???1.

For this sequence of functions, we investigate the regret and the value of the iterate x t for ADAM and AMSGRAD.

To enable fair comparison, we set ?? 1 = 0.9 and ?? 2 = 0.99 for ADAM and AMSGRAD algorithm, which are typically the parameters settings used for ADAM in practice.

FIG1 shows the average regret (R t /t) and value of the iterate (x t ) for this problem.

We first note that the average regret of ADAM does not converge to 0 with increasing t. Furthermore, its iterates x t converge to x = 1, which unfortunately has the largest regret amongst all points in the domain.

On the other hand, the average regret of AMSGRAD converges to 0 and its iterate converges to the optimal solution.

FIG1 also shows the stochastic optimization setting: DISPLAYFORM1 , with probability 0.01 ???10x, otherwise.

Similar to the aforementioned online setting, the optimal solution for this problem is x = ???1.

Again, we see that the iterate x t of ADAM converges to the highly suboptimal solution x = 1.Logistic Regression: To investigate the performance of the algorithm on convex problems, we compare AMSGRAD with ADAM on logistic regression problem.

We use MNIST dataset for this experiment, the classification is based on 784 dimensional image vector to one of the 10 class labels.

The step size parameter ?? t is set to ??/ ??? t for both ADAM and AMSGRAD in for our experiments, consistent with the theory.

We use a minibatch version of these algorithms with minibatch size set to 128.

We set ?? 1 = 0.9 and ?? 2 is chosen from the set {0.99, 0.999}, but they are fixed throughout the experiment.

The parameters ?? and ?? 2 are chosen by grid search.

We report the train and test loss with respect to iterations in FIG2 .

We can see that AMSGRAD performs better than ADAM with respect to both train and test loss.

We also observed that AMSGRAD is relatively more robust to parameter changes in comparison to ADAM.Neural Networks:

For our first experiment, we trained a simple 1-hidden fully connected layer neural network for the multiclass classification problem on MNIST.

Similar to the previous experiment, we use ?? 1 = 0.9 and ?? 2 is chosen from {0.99, 0.999}. We use a fully connected 100 rectified linear units (ReLU) as the hidden layer for this experiment.

Furthermore, we use constant ?? t = ?? throughout all our experiments on neural networks.

Such a parameter setting choice of ADAM is consistent with the ones typically used in the deep learning community for training neural networks.

A grid search is used to determine parameters that provides the best performance for the algorithm.

Finally, we consider the multiclass classification problem on the standard CIFAR-10 dataset, which consists of 60,000 labeled examples of 32 ?? 32 images.

We use CIFARNET, a convolutional neural network (CNN) with several layers of convolution, pooling and non-linear units, for training a multiclass classifer for this problem.

In particular, this architecture has 2 convolutional layers with 64 channels and kernel size of 6 ?? 6 followed by 2 fully connected layers of size 384 and 192.

The network uses 2 ?? 2 max pooling and layer response normalization between the convolutional layers BID4 .

A dropout layer with keep probability of 0.5 is applied in between the fully connected layers BID6 .

The minibatch size is also set to 128 similar to previous experiments.

The results for this problem are reported in FIG2 .

The parameters for ADAM and AMSGRAD are selected in a way similar to the previous experiments.

We can see that AMSGRAD performs considerably better than ADAM on train loss and accuracy.

Furthermore, this performance gain also translates into good performance on test loss.

An alternative approach is to use an increasing schedule of ?? 2 in ADAM.

This approach, unlike Algorithm 2 does not require changing the structure of ADAM but rather uses a non-constant ?? 1 and ?? 2 .

The pseudocode for the algorithm, ADAMNC, is provided in the appendix (Algorithm 3).

We show that by appropriate selection of ?? 1t and ?? 2t , we can achieve good convergence rates.

Theorem 5.

Let {x t } and {v t } be the sequences obtained from Algorithm 3, ?? t = ??/ ??? t, ?? 1 = ?? 11 and ?? 1t ??? ?? 1 for all t ??? [T ].

Assume that F has bounded diameter D ??? and ???f t (x) ??? ??? G ??? for all t ??? [T ] and x ??? F. Furthermore, let {?? 2t } be such that the following conditions are satisfied:

DISPLAYFORM0 Then for x t generated using the ADAMNC (Algorithm 3), we have the following bound on the regret DISPLAYFORM1 The above result assumes selection of {(?? t , ?? 2t )} such that ?? t 0 for all t ??? {2, ?? ?? ?? , T }.

However, one can generalize the result to deal with the case where this constraint is violated as long as the violation is not too large or frequent.

Following is an immediate consequence of the above result.

Corollary 2.

Suppose ?? 1t = ?? 1 ?? t???1 and ?? 2t = 1 ??? 1/t in Theorem 5, then we have DISPLAYFORM2 The above corollary follows from a trivial fact that v t,i = t j=1 g 2 j,i /t for all i ??? [d] when ?? 2t = 1 ??? 1/t.

This corollary is interesting insofar that such a parameter setting effectively yields a momentum based variant of ADAGRAD.

Similar to ADAGRAD, the regret is data-dependent and can be considerably better than O( BID2 .

It is easy to generalize this result for setting similar settings of ?? 2t .

Similar to Corollary 1, one can use a more modest decay of ?? 1t = ?? 1 /t and still ensure a data-dependent regret of O( ??? T ).

DISPLAYFORM3

In this paper, we study exponential moving variants of ADAGRAD and identify an important flaw in these algorithms which can lead to undesirable convergence behavior.

We demonstrate these problems through carefully constructed examples where RMSPROP and ADAM converge to highly suboptimal solutions.

In general, any algorithm that relies on an essentially fixed sized window of past gradients to scale the gradient updates will suffer from this problem.

We proposed fixes to this problem by slightly modifying the algorithms, essentially endowing the algorithms with a long-term memory of past gradients.

These fixes retain the good practical performance of the original algorithms, and in some cases actually show improvements.

The primary goal of this paper is to highlight the problems with popular exponential moving average variants of ADAGRAD from a theoretical perspective.

RMSPROP and ADAM have been immensely successful in development of several state-of-the-art solutions for a wide range of problems.

Thus, it is important to understand their behavior in a rigorous manner and be aware of potential pitfalls while using them in practice.

We believe this paper is a first step in this direction and suggests good design principles for faster and better stochastic optimization.

A PROOF OF THEOREM 1Proof.

We consider the setting where f t are linear functions and F = [???1, 1].

In particular, we define the following function sequence: DISPLAYFORM0 where C ??? 2.

For this function sequence, it is easy to see that the point x = ???1 provides the minimum regret.

Without loss of generality, assume that the initial point is x 1 = 1.

This can be assumed without any loss of generality because for any choice of initial point, we can always translate the coordinate system such that the initial point is x 1 = 1 in the new coordinate system and then choose the sequence of functions as above in the new coordinate system.

Also, since the problem is one-dimensional, we drop indices representing coordinates from all quantities in Algorithm 1.

Consider the execution of ADAM algorithm for this sequence of functions with DISPLAYFORM1 Note that since gradients of these functions are bounded, F has bounded L ??? diameter and ?? 2 1 / ??? ?? 2 < 1.

Hence, the conditions on the parameters required for ADAM are satisfied (refer to BID3 for more details).Our main claim is that for iterates {x t } ??? t=1 arising from the updates of ADAM, we have x t > 0 for all t ??? N and furthermore, x 3t+1 = 1 for all t ??? N ??? {0}. For proving this, we resort to the principle of mathematical induction.

Since x 1 = 1, both the aforementioned conditions hold for the base case.

Suppose for some t ??? N ??? {0}, we have x i > 0 for all i ??? [3t + 1] and x 3t+1 = 1.

Our aim is to prove that x 3t+2 and x 3t+3 are positive and x 3t+4 = 1.

We first observe that the gradients have the following form: DISPLAYFORM2 th update of ADAM in Equation (1), we obtain DISPLAYFORM3 The equality follows from the induction hypothesis.

We observe the following: ??C DISPLAYFORM4 The second inequality follows from the step size choice that ?? < ??? 1 ??? ?? 2 .

Therefore, we have 0 <x 3t+2 < 1 and hence x 3t+2 =x 3t+2 > 0.

Furthermore, after the (3t + 2) th and (3t + 3) th updates of ADAM in Equation FORMULA8 , we have the following: DISPLAYFORM5 Since x 3t+2 > 0, it is easy to see that x 3t+3 > 0.

To complete the proof, we need to show that x 3t+4 = 1.

In order to prove this claim, we show thatx 3t+4 ??? 1, which readily translates to x 3t+4 = 1 because x 3t+4 = ?? F (x 3t+4 ) and F = [???1, 1] here ?? F is the simple Euclidean projection (note that in one-dimension, ?? F , ??? Vt = ?? F ).

We observe the following: DISPLAYFORM6 The above equality is due to the fact thatx 3t+3 > 0 and property of projection operation onto the set F = [???1, 1].

We consider the following two cases:1.

Supposex 3t+3 ??? 1, then it is easy to see from the above equality thatx 3t+4 > 1.2.

Supposex 3t+3 < 1, then we have the following: DISPLAYFORM7 The third equality is due to the fact that x 3t+2 =x 3t+2 .

Thus, to provex 3t+4 > 1, it is enough to the prove: DISPLAYFORM8 We have the following bound on term T 1 from Equation FORMULA27 : DISPLAYFORM9 Furthermore, we lower bound T 2 in the following manner: DISPLAYFORM10 The first inequality is due to the fact that v t ??? C 2 for all t ??? N. The last inequality follows from inequality in Equation (5).

The last equality is due to following fact: DISPLAYFORM11 for the choice of ?? 2 = 1/(1 + C 2 ).

Therefore, we have T 2 ??? T 1 and hence,x 3t+4 ??? 1.Therefore, from both the cases, we see that x 3t+4 = 1.

Therefore, by the principle of mathematical induction it holds for all t ??? N ??? {0}. Thus, we have DISPLAYFORM12 Therefore, for every 3 steps, ADAM suffers a regret of at least 2C ??? 4.

More specifically, R T ??? (2C ??? 4)T /3.

Since C ??? 2, this regret can be very large and furthermore, R T /T 0 as T ??? ???, which completes the proof.

Proof.

The proof generalizes the optimization setting used in Theorem 1.

Throughout the proof, we assume ?? 1 < ??? ?? 2 , which is also a condition (Kingma & Ba, 2015) assume in their paper.

In this proof, we consider the setting where f t are linear functions and F = [???1, 1].

In particular, we define the following function sequence: DISPLAYFORM0 where C ??? N, C mod 2 = 0 satisfies the following: DISPLAYFORM1 where ?? = ?? 1 / ??? ?? 2 < 1.

It is not hard to see that these conditions hold for large constant C that depends on ?? 1 and ?? 2 .

Since the problem is one-dimensional, we drop indices representing coordinates from all quantities in Algorithm 1.

For this function sequence, it is easy to see that the point x = ???1 provides the minimum regret since C ??? 2.

Furthermore, the gradients have the following form: DISPLAYFORM2 for t mod C = 1 ???1, otherwise Our first observation is that m kC ??? 0 for all k ??? N ??? {0}. For k = 0, this holds trivially due to our initialization.

For the general case, observe the following: DISPLAYFORM3 If m kC ??? 0, it can be easily shown that m kC+C ??? 0 for our selection of C in Equation (7) by using the principle of mathematical induction.

With this observation we continue to the main part of the proof.

Let T be such that t + C ??? ?? 2 t for all t ??? T where ?? ??? 3/2.

All our analysis focuses on iterations t ??? T .

Note that any regret before T is just a constant because T is independent of T and thus, the average regret is negligible as T ??? ???. Consider an iterate at time step t of the form kC after T .

Our claim is that DISPLAYFORM4 for some c t > 0.

To see this, consider the updates of ADAM for the particular sequence of functions we considered are: DISPLAYFORM5 For i ??? {2, ?? ?? ?? , C}, we use the following notation: DISPLAYFORM6 Note that if ?? t+j ??? 0 for some j ??? {1, ?? ?? ?? , C ??? 1} then ?? t+l ??? 0 for all l ??? {j, ?? ?? ?? , C ??? 1}. This follows from the fact that the gradient is negative for all time steps i ??? {2, ?? ?? ?? , C}. Using Lemma 6 for {x t+1 , ?? ?? ?? , x t+C } and {?? t , ?? ?? ?? , ?? t+C???1 }, we have the following: DISPLAYFORM7 Let i = C/2.

In order to prove our claim in Equation FORMULA8 , we need to prove the following: DISPLAYFORM8 To this end, we observe the following: DISPLAYFORM9 The first equality follows from the definition of m t+i+1 .

The first inequality follows from the fact that m t ??? 0 when t mod C = 0 (see Equation FORMULA39 and arguments based on it).

The second inequality follows from the definition of ?? that t + C ??? ?? 2 t for all t ??? T .

The third inequality is due to the fact that v t+i???1 ??? (1 ??? ?? 2 )?? i???2 2 C 2 .

The last inequality follows from our choice of C. The fourth inequality is due to the following upper bound that applies for all i ??? i ??? C: DISPLAYFORM10 The first inequality follows from online problem setting for the counter-example i.e., gradient is C once every C iterations and ???1 for the rest.

The last inequality follows from the fact that ?? i ???1 2 C 2 ??? 1 and ?? C 2 ??? ?? 2 .

Furthermore, from the above inequality, we have DISPLAYFORM11 Note that from our choice of C, it is easy to see that ?? ??? 0.

Also, observe that ?? is independent of t. Thus, x t+C ??? min{1, x t + ??/ ??? t}. From this fact, we also see the following:1.

If x t = 1, then x t+C = 1 for all t ??? T such that t mod C = 0.2.

There exists constant T 1 ??? T such that x T 1 = 1 where T 1 mod C = 0.The first point simply follows from the relation x t+C ??? min{1, x t + ??/ ??? t}. The second point is due to divergent nature of the sum DISPLAYFORM12 where kC ??? T 1 .

Thus, when t ??? T 1 , for every C steps, ADAM suffers a regret of at least 2.

More specifically, R T ??? 2(T ??? T 1 )/C.

Thus, R T /T 0 as T ??? ???, which completes the proof.

Proof.

Let ?? be an arbitrary small positive constant, and C be a large enough constant chosen as a function of ?? 1 , ?? 2 , ?? that will be determined in the proof.

Consider the following one dimensional stochastic optimization setting over the domain [???1, 1].

At each time step t, the function f t (x) is chosen i.i.d.

as follows: DISPLAYFORM0 Cx with probability p := 1+?? C+1 ???x with probability 1 ??? pThe expected function is F (x) = ??x; thus the optimum point over [???1, 1] is x = ???1.

At each time step t the gradient g t equals C with probability p and ???1 with probability 1 ??? p. Thus, the step taken by ADAM is DISPLAYFORM1 We now show that for a large enough constant C, E[??? t ] ??? 0, which implies that the ADAM's steps keep drifting away from the optimal solution x = ???1.Lemma 1.

For a large enough constant C (as a function of ?? 1 , ?? 2 , ??), DISPLAYFORM2 denote expectation conditioned on all randomness up to and including time t ??? 1.

Taking conditional expectation of the step, we have DISPLAYFORM3 We will bound the expectation of the terms T 1 , T 2 and T 3 above separately.

First, for T 1 , we have DISPLAYFORM4 Next, we bound DISPLAYFORM5 log(1/??1) .

This choice of k ensures that ?? DISPLAYFORM6 Let E denote the event that for every DISPLAYFORM7 Assuming E happens, we can bound m t???1 as follows: DISPLAYFORM8 and so T 2 ??? 0.With probability at most kp, the event E doesn't happen.

In this case, we bound T 2 as follows.

We first bound m t???1 in terms of v t???1 using the Cauchy-Schwarz inequality as follows: DISPLAYFORM9 Thus, v t???1 ??? m 2 t???1 /A 2 .

Thus, we have DISPLAYFORM10 .Hence, we have DISPLAYFORM11 Finally, we lower bound E[T 3 ] using Jensen's inequality applied to the convex function DISPLAYFORM12 The last inequality follows by using the facts DISPLAYFORM13 , and the random variables g DISPLAYFORM14 Combining the bounds in (12), (13), and (14) in the expression for ADAM's step, (11), and plugging in the values of the parameters k and p we get the following lower bound on E[??? t ]: DISPLAYFORM15 It is evident that for C large enough (as a function of ??, ?? 1 , ?? 2 ), the above expression can be made non-negative.

For the sake of simplicity, let us assume, as is routinely done in practice, that we are using a version of ADAM that doesn't perform any projection steps 2 .

Then the lemma implies that DISPLAYFORM16 .

Via a simple induction, we conclude that E[x t ] ??? x 1 for all t. Thus, if we assume that the starting point x 1 ??? 0, then E[x t ] ??? 0.

Since F is a monotonically increasing function, we have E[F (x t )] ??? F (0) = 0, whereas F (???1) = ?????.

Thus the expected suboptimality gap is always ?? > 0, which implies that ADAM doesn't converge to the optimal solution.

The proof of Theorem 4 presented below is along the lines of the Theorem 4.1 in BID3 which provides a claim of convergence for ADAM.

As our examples showing nonconvergence of ADAM indicate, the proof in BID3 has problems.

The main issue in their proof is the incorrect assumption that ?? t defined in their equation FORMULA11 is positive semidefinite, and we also identified problems in lemmas 10.3 and 10.4 in their paper.

The following proof fixes these issues and provides a proof of convergence for AMSGRAD.Proof.

We begin with the following observation: DISPLAYFORM0 In this proof, we will use x * i to denote the i th coordinate of x * .

Using Lemma 4 with u 1 = x t+1 and u 2 = x * , we have the following: DISPLAYFORM1 Rearranging the above inequality, we have DISPLAYFORM2 The second inequality follows from simple application of Cauchy-Schwarz and Young's inequality.

We now use the standard approach of bounding the regret at each step using convexity of the function f t in the following manner: DISPLAYFORM3 The first inequality is due to convexity of function f t .

The second inequality follows from the bound in Equation FORMULA8 .

For further bounding this inequality, we need the following intermediate result.

Lemma 2.

For the parameter settings and conditions assumed in Theorem 4, we have DISPLAYFORM4 Proof.

We start with the following: DISPLAYFORM5 The first inequality follows from the definition ofv T,i , which is maximum of all v T,i until the current time step.

The second inequality follows from the update rule of Algorithm 2.

We further bound the above inequality in the following manner: DISPLAYFORM6 The first inequality follows from Cauchy-Schwarz inequality.

The second inequality is due to the fact that ?? 1k ??? ?? 1 for all k ??? [T ].

The third inequality follows from the inequality DISPLAYFORM7 .

By using similar upper bounds for all time steps, the quantity in Equation FORMULA8 can further be bounded as follows: DISPLAYFORM8 The third inequality follows from the fact that DISPLAYFORM9 The fourth inequality is due to simple application of Cauchy-Schwarz inequality.

The final inequality is due to the following bound on harmonic sum: T t=1 1/t ??? (1 + log T ).

This completes the proof of the lemma.

We now return to the proof of Theorem 4.

Using the above lemma in Equation FORMULA8 , we have: DISPLAYFORM10 The first inequality and second inequality use the fact that ?? 1t ??? ?? 1 .

In order to further simplify the bound in Equation FORMULA8 , we need to use telescopic sum.

We observe that, by definition ofv t,i , we havev DISPLAYFORM11 Using the L ??? bound on the feasible region and making use of the above property in Equation (18), we have: DISPLAYFORM12 Set m0 = 0 and v0 = 0 DISPLAYFORM13 The equality follows from simple telescopic sum, which yields the desired result.

One important point to note here is that the regret of AMSGRAD can be bounded by O(G ??? ??? T ).

This can be easily seen from the proof of the aforementioned lemma where in the analysis the term DISPLAYFORM14 Thus, the regret of AMSGRAD is upper bounded by minimum of O(G ??? ??? T ) and the bound in the Theorem 4 and therefore, the worst case dependence of regret on T in our case is O( ??? T ).

Proof.

Using similar argument to proof of Theorem 4 until Equation FORMULA8 , we have the following DISPLAYFORM0 The second inequality follows from simple application of Cauchy-Schwarz and Young's inequality.

We now use the standard approach of bounding the regret at each step using convexity of the function f t in the following manner: DISPLAYFORM1 The inequalities follow due to convexity of function f t and Equation (19).

For further bounding this inequality, we need the following intermediate result.

Lemma 3.

For the parameter settings and conditions assumed in Theorem 5, we have DISPLAYFORM2 Proof.

We start with the following: DISPLAYFORM3 The first inequality follows from the update rule of Algorithm 2.

We further bound the above inequality in the following manner: DISPLAYFORM4 The first inequality and second inequality use the fact that ?? 1t ??? ?? 1 .

Furthermore, from the theorem statement, we know that that {(?? t .?? 2t )} are selected such that the following holds: DISPLAYFORM5 Using the L ??? bound on the feasible region and making use of the above property in Equation FORMULA9 , we have: DISPLAYFORM6 The equality follows from simple telescopic sum, which yields the desired result.

Theorem 6.

For any > 0, ADAM with the modified update in Equation (3) and with parameter setting such that all the conditions in BID3 are satisfied can have non-zero average regret i.e., R T /T 0 as T ??? ??? for convex DISPLAYFORM0 with bounded gradients on a feasible set F having bounded D ??? diameter.

Proof.

Let us first consider the case where = 1 (in fact, the same setting works for any ??? 1).

The general case can be proved by simply rescaling the sequence of functions by a factor of ??? .

We show that the same optimization setting in Theorem 1 where f t are linear functions and F = [???1, 1], hence, we only discuss the details that differ from the proof of Theorem 1.

In particular, we define the following function sequence: DISPLAYFORM1 Cx, for t mod 3 = 1 ???x, otherwise, where C ??? 2.

Similar to the proof of Theorem 1, we assume that the initial point is x 1 = 1 and the parameters are:?? 1 = 0, ?? 2 = 2 (1 + C 2 )C 2 and ?? t = ?? ??? t where ?? < ??? 1 ??? ?? 2 .

The proof essentially follows along the lines of that of Theorem 1 and is through principle of mathematical induction.

Our aim is to prove that x 3t+2 and x 3t+3 are positive and x 3t+4 = 1.

The base case holds trivially.

Suppose for some t ??? N ??? {0}, we have x i > 0 for all i ??? [3t + 1] and x 3t+1 = 1.

For (3t + 1) th update, the only change from the update of in Equation FORMULA8 is the additional in the denominator i.e., we hav?? x 3t+2 = x 3t+1 ??? ??C (3t + 1)(?? 2 v 3t + (1 ??? ?? 2 )C 2 + ) DISPLAYFORM2 The last inequality follows by simply dropping v 3t term and using the relation that ?? < ??? 1 ??? ?? 2 .

Therefore, we have 0 <x 3t+2 < 1 and hence x 3t+2 =x 3t+2 > 0.

Furthermore, after the (3t + 2) th and (3t + 3) th updates of ADAM in Equation FORMULA8 , we have the following:x 3t+3 = x 3t+2 + ?? (3t + 2)(?? 2 v 3t+1 + (1 ??? ?? 2 ) + ) , x 3t+4 = x 3t+3 + ?? (3t + 3)(?? 2 v 3t+2 + (1 ??? ?? 2 ) + ) .Since x 3t+2 > 0, it is easy to see that x 3t+3 > 0.

To complete the proof, we need to show that x 3t+4 = 1.

The only change here from the proof of Theorem 1 is that we need to show the The first inequality is due to the fact that v t ??? C 2 for all t ??? N. The last equality is due to following fact: DISPLAYFORM3 for the choice of ?? 2 = 2/[(1 + C 2 )C 2 ] and = 1.

Therefore, we see that x 3t+4 = 1.

Therefore, by the principle of mathematical induction it holds for all t ??? N ??? {0}. Thus, we have f 3t+1 (x 3t+1 ) + f 3t+2 (x 3t+2 ) + f 3t+2 (x 3t+2 ) ??? f 3t+1 (???1) ??? f 3t+2 (???1) ??? f 3t+3 (???1) ??? 2C ??? 4.Therefore, for every 3 steps, ADAM suffers a regret of at least 2C ??? 4.

More specifically, R T ??? (2C ??? 4)T /3.

Since C ??? 2, this regret can be very large and furthermore, R T /T 0 as T ??? ???, which completes the proof of the case where = 1.

For the general case, we consider the following sequence of functions: DISPLAYFORM4 The functions are essentially rescaled in a manner so that the resultant updates of ADAM correspond to the one in the optimization setting described above.

Using essentially the same argument as above, it is easy to show that the regret R T ??? (2C ??? 4) ??? T /3 and thus, the average regret is non-zero asymptotically, which completes the proof.

G AUXILIARY LEMMA Lemma 4 ((McMahan & Streeter, 2010) ).

For any Q ??? S d + and convex feasible set F ??? R d , suppose u 1 = min x???F Q 1/2 (x???z 1 ) and u 2 = min x???F Q 1/2 (x???z 2 ) then we have Q 1/2 (u 1 ??? u 2 ) ??? Q 1/2 (z 1 ??? z 2 ) .Proof.

We provide the proof here for completeness.

Since u 1 = min x???F Q 1/2 (x ??? z 1 ) and u 2 = min x???F Q 1/2 (x ??? z 2 ) and from the property of projection operator we have the following: z 1 ??? u 1 , Q(z 2 ??? z 1 ) ??? 0 and z 2 ??? u 2 , Q(z 1 ??? z 2 ) ??? 0.Combining the above inequalities, we have u 2 ??? u 1 , Q(z 2 ??? z 1 ) ??? z 2 ??? z 1 , Q(z 2 ??? z 1 ) .Also, observe the following: DISPLAYFORM5 The above inequality can be obtained from the fact that (u 2 ??? u 1 ) ??? (z 2 ??? z 1 ), Q((u 2 ??? u 1 ) ??? (z 2 ??? z 1 )) ??? 0 as Q ??? S d + and rearranging the terms.

Combining the above inequality with Equation FORMULA9 , we have the required result.

Lemma 5 ( BID0 ).

For any non-negative real numbers y 1 , ?? ?? ?? , y t , the following holds: for all the t ??? [T ], y 1 ??? F and furthermore, there exists i ??? [T ] such that ?? j ??? 0 for all j ??? i and ?? j > 0 for all j >

i.

Then we have, DISPLAYFORM6 Proof.

It is first easy to see that y i+1 ???

y 1 + i j=1 ?? j since ?? j ??? 0 for all j ??? i. Furthermore, also observe that y T +1 ??? min{b, y i+1 + T j=i+1 ?? j } since ?? j ??? 0 for all j > i. Combining the above two inequalities gives us the desired result.

<|TLDR|>

@highlight

We investigate the convergence of popular optimization algorithms like Adam , RMSProp and propose new variants of these methods which provably converge to optimal solution in convex  settings. 