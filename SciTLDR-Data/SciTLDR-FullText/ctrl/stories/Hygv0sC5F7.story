We study the implicit bias of gradient descent methods in solving a binary classification problem over a linearly separable dataset.

The classifier is described by a nonlinear ReLU model and the objective function adopts the exponential loss function.

We first characterize the landscape of the loss function and show that there can exist spurious asymptotic local minima besides asymptotic global minima.

We then show that gradient descent (GD) can converge to either a global or a local max-margin direction, or may diverge from the desired max-margin direction in a general context.

For stochastic gradient descent (SGD), we show that it converges in expectation to either the global or the local max-margin direction if SGD converges.

We further explore the implicit bias of these algorithms in learning a multi-neuron network under certain stationary conditions, and show that the learned classifier maximizes the margins of each sample pattern partition under the ReLU activation.

It has been observed in various machine learning problems recently that the gradient descent (GD) algorithm and the stochastic gradient descent (SGD) algorithm converge to solutions with certain properties even without explicit regularization in the objective function.

Correspondingly, theoretical analysis has been developed to explain such implicit regularization property.

For example, it has been shown in Gunasekar et al. (2018; 2017) that GD converges to the solution with the minimum norm under certain initialization for regression problems, even without an explicit norm constraint.

Another type of implicit regularization, where GD converges to the max-margin classifier, has been recently studied in Gunasekar et al. (2018) ; Ji & Telgarsky (2018) ; Nacson et al. (2018a) ; Soudry et al. (2017; 2018) for classification problems as we describe below.

Given a set of training samples z i = (x i , y i ) for i = 1, . . .

, n, where x i denotes a feature vector and y i ∈ {−1, +1} denotes the corresponding label, the goal is to find a desirable linear model (i.e., a classifier) by solving the following empirical risk minimization problem It has been shown in Nacson et al. (2018a) ; Soudry et al. (2017; 2018) that if the loss function (·) is monotonically strictly decreasing and satisfies proper tail conditions (e.g., the exponential loss), and the data are linearly separable, then GD converges to the solution w with infinite norm and the maximum margin direction of the data, although there is no explicit regularization towards the maxmargin direction in the objective function.

Such a phenomenon is referred to as the implicit bias of GD, and can help to explain some experimental results.

For example, even when the training error achieves zero (i.e., the resulting model enters into the linearly separable region that correctly classifies the data), the testing error continues to decrease, because the direction of the model parameter continues to have an improved margin.

Such a study has been further generalized to hold for various other types of gradient-based algorithms Gunasekar et al. (2018) .

Moreover, Ji & Telgarsky (2018) analyzed the convergence of GD with no assumption on the data separability, and characterized the implicit regularization to be in a subspace-based form.

The focus of this paper is on the following two fundamental issues, which have not been well addressed by existing studies.• Existing studies so far focused only on the linear classifier model.

An important question one naturally asks is what happens for the more general nonlinear leaky ReLU and ReLU models.

Will GD still converge, and if so will it converge to the max-margin direction?

Our study here provides new insights for the ReLU model that have not been observed for the linear model in the previous studies.• Existing studies mainly analyzed the convergence of GD with the only exceptions Ji & Telgarsky (2018) ; Nacson et al. (2018b) on SGD.

However, Ji & Telgarsky (2018) did not establish the convergence to the max-margin direction for SGD, and Nacson et al. (2018b) established the convergence to the max-margin solution only epochwisely for cyclic SGD (not iterationwise for SGD under random sampling with replacement).

Moreover, both studies considered only the linear model.

Here, our interest is to explore the iterationwise convergence of SGD under random sampling with replacement to the max-margin direction, and our result can shed insights for online SGD.

Furthermore, our study provides new understanding for the nonlinear ReLU and leaky ReLU models.

We summarize our main contributions, where our focus is on the exponential loss function under ReLU model.

We first characterize the landscape of the empirical risk function under the ReLU model, which is nonconvex and nonsmooth.

We show that such a risk function has asymptotic global minima and asymptotic spurious local minima.

Such a landscape is in sharp contrast to that under the linear model previously studied in Soudry et al. (2017) , where there exist only equivalent global minima.

Based on the landscape property, we show that the implicit bias property in the course of the convergence of GD can fall into four cases: converges to the asymptotic global minimum along the max-margin direction, converges to an asymptotic local minimum along a local max-margin direction, stops at a finite spurious local minimum, or oscillates between the linearly separable and misclassified regions without convergence.

Such a diverse behavior is also in sharp difference from that under the linear model Soudry et al. (2017) , where GD always converges to the max-margin direction.

We then take a further step to study the implicit bias of SGD.

We show that the expected averaged weight vector normalized by its expected l 2 norm converges to the global max-margin direction or local max-margin direction, as long as SGD stays either in the linearly separable region or in a region of the local minima defined by a subset of data samples with positive label.

The proof here requires considerable new technical developments, which are very different from the traditional analysis of SGD, e.g., Bottou et al. (2016); Duchi & Singer (2009); Nemirovskii et al. (1983); Shalev-Shwartz et al. (2009); Xiao (2010) ; BID0 BID1 .

This is because our focus here is on the exponential loss function without attainable global/local minima, whereas traditional analysis typically assumed that the minimum of the loss function is attainable.

Furthermore, our goal is to analyze the implicit bias property of SGD, which is also beyond traditional analysis of SGD.We further extend our analysis to the leaky ReLU model and multi-neuron networks.

Implicit bias of gradient descent: Gunasekar et al. (2018) studied the implicit bias of GD and SGD for minimizing the squared loss function under bounded global minimum, and showed that some of these algorithms converge to a global minimum that is closest to the initial point.

Another collection of papers Gunasekar et al. Telgarsky (2013) showed that AdaBoost converges to an approximate max-margin classifier.

Soudry et al. (2017; 2018) studied the convergence of GD in logistic regression with linearly separable data and showed that GD converges in direction to the solution of support vector machine at a rate of 1/ ln(t).

Nacson et al. (2018a) improved this rate to ln(t)/ √ t under the exponential loss via normalized gradient descent.

Gunasekar et al. (2018) further showed that steepest descent can lead to margin maximization under generic norms.

Ji & Telgarsky (2018) analyzed the convergence of GD on an arbitrary dataset, and provided the convergence rates along the strongly convex subspace and the separable subspace.

Our work studies the convergence of GD and SGD under the nonlinear ReLU model with the exponential loss, as opposed to the linear model studied by all the above previous work on the same type of loss functions.

Ji & Telgarsky (2018) analyzed the average SGD (under random sampling) with fixed learning rate and proved the convergence of the population risk, but did not establish the parameter convergence of SGD in the max-margin direction.

Nacson et al. (2018b) established the convergence of cyclic SGD epochwisely in direction to the max-margin classifier at a rate O(1/ ln t).

Our work differs from these two studies first in that we study the ReLU model, whereas both of these studies analyzed the linear model.

Furthermore, we showed that under SGD with random sampling, the expectation of the averaged weight vector converges in direction to the max-margin classifier at a rate O(1/ √ ln t).

There have been extensive studies of the convergence and generalization performance of SGD under various models, of which we cannot provide a comprehensive list due to the space limitations.

In general, these type of studies either characterize the convergence rate of SGD or provide the generalization error bounds at the convergence of SGD, e.g., Brutzkus et al. (2017); Wang et al. (2018); Li & Liang (2018) , but did not characterize the implicit regularization property of SGD, such as the convergence to the max-margin direction as provided in our paper.

We consider the binary classification problem, in which we are given a set of training samples {z 1 , . . .

, z n }.

Each training sample z i = (x i , y i ) contains an input data x i and a corresponding binary label y i ∈ {−1, +1}.

We denote I + := {i : y i = +1} as the set of indices of samples with label +1 and denote I − := {i : y i = −1} in a similar way.

Their cardinalities are denoted as n + and n − , respectively, and are assumed to be non-zero.

We consider all datasets that are linearly separable, i.e., there exists a linear classifier w such that y i w x i > 0 for all i = 1, . . .

, n.

We are interested in training a ReLU model for the classification task.

In specific, for a given input data x, the model outputs σ(w x i ), where σ(v) = max{0, v} is the ReLU activation function and w denotes the weight parameters.

The predicted label is set to be sgn(w x).

Our goal is to learn a classifier by solving the following empirical risk minimization problem, where we adopt the exponential loss.

DISPLAYFORM0 The ReLU activation causes the loss function in problem (P) to be nonconvex and nonsmooth.

Therefore, it is important to first understand the landscape property of the loss function, which is critical for characterizing the implicit bias property of the GD and SGD algorithms.

In order to understand the convergence of GD under the ReLU model, we first study the landscape of the loss function in problem (P), which turns out to be very different from that under the linear activation model.

As been shown in Soudry et al. (2017); Ji & Telgarsky (2018) , the loss function in problem (P) under linear activation is convex, and achieves asymptotic global minimum, i.e., ∇L(αw * ) α → 0 and L(αw * ) α → 0 as the scaling constant α → +∞, only if w * is in the linearly separable region.

In contrast, under the ReLU model, the asymptotic critical points can be either global minimum or (spurious) local minimum depending on the training datasets, and hence the convergence property of GD can be very different in nature from that under the linear model.

The following theorem characterizes the landscape properties of problem (P).

Throughout, we denote the infimum of the objective function in problem (P) as L * = n − n .

Furthermore, we call a direction w * asymptotically critical if it satisfies ∇L(αw * )→0 as α → +∞.Theorem 3.1 (Asymptotic landscape property).

For problem (P) under the ReLU model, any corresponding asymptotic critical direction w * fall into one of the following cases: DISPLAYFORM0 To further elaborate Theorem 3.1, if w * classifies all data correctly (i.e., item 1), then the objective function possibly achieves global minimum L * along this direction.

On the other hand, if w * classifies some data with label +1 as −1 (item 2), then the objective function achieves a sub-optimal value along this direction.

In the worst case where all data samples are classified as −1 (item 3), the ReLU unit is never activated and hence the corresponding objective function has constant value 1.

We note that the cases in items 2 and 3 may or may not take place depending on specific datasets, but if they do occur, the corresponding w * are spurious (asymptotic) local minima.

In summary, the landscape under the ReLU model can be partitioned into different regions, where gradient descent algorithms can have different implicit bias as we show next.

In this subsection, we analyze the convergence of GD in learning the ReLU model.

At each iteration t, GD performs the update DISPLAYFORM0 where η denotes the stepsize.

For the linear model whose loss function has infinitely many asymptotic global minima, it has been shown in Soudry et al. (2017) that GD always converges to the max-margin direction.

Such a phenomenon is regarded as the implicit bias property of GD.

Here, for the ReLU model, we are also interested in analyzing whether such an implicit-bias property still holds.

Furthermore, since the loss function under the ReLU model possibly contains spurious asymptotic local minima, the convergence of GD under the ReLU model should be very different from that under the linear model.

Next, we introduce various notions of margin in order to characterize the implicit bias under the ReLU model.

The global max-margin direction of samples in I + is defined as DISPLAYFORM1 Such a notion of max-margin is natural because the ReLU activation function can suppress negative inputs.

We note that here w + may not locate in the linearly separable region, and hence it may not be parallel to any (asymptotic) global minimum.

As we show next, only when w + is in the linearly separable region, GD may converge in direction to such a max-margin direction under the ReLU model.

Furthermore, for each given subset J + ⊆ I + , we define the associated local max-margin direction w DISPLAYFORM2 We further denote the set of asymptotic local minima with respect to J + ⊆ I + (see Theorem 3.1 item 2) as DISPLAYFORM3 Theorem 3.2.

Apply GD to solve problem (P) with arbitrary initialization and a small enough constant stepsize.

Then, the sequence {w t } t generated by GD falls into one of the following cases.

DISPLAYFORM4 DISPLAYFORM5 n , and w t =ŵ + J , where J + = ∅, i.e., GD terminates within finite steps.

Theorem 3.2 characterizes various instances of implicit bias of GD in learning the ReLU model, which the nature of the convergence is different from that in learning the linear model.

In specific, GD can either converge in direction to the global max-margin direction w + that leads to the global minimum, or converge to the local max-margin direction w + J that leads to a spurious local minimum.

Furthermore, it may occur that GD oscillates between the linearly separable region and the misclassified region due to the suppression effect of ReLU function.

In this case, GD does not have an implicit bias property and convergence guarantee.

We provide two simple examples in the supplementary material to further elaborate these cases.

In this subsection, we analyze the convergence property and the implicit bias of SGD for solving problem (P).

At each iteration t, SGD samples an index ξ t ∈ {1, . . .

, n} uniformly at random with replacement, and performs the update DISPLAYFORM0 Similarly to the convergence of GD characterized in Theorem 3.2, SGD may oscillate between the linearly separable and misclassified regions.

Therefore, our major interest here is the implicit bias of SGD when it does converge either to the asymptotic global minimum or local minimum.

Thus, without loss of generality, we implicitly assume that w + is in the linearly separable region, and the relevant w + J ∈ W + J .

Otherwise, SGD does not even converge.

The implicit bias of SGD with replacement sampling has not been studied in the existing literature, and the proof of the convergence and the characterization of the implicit bias requires substantial new technical developments.

In particular, traditional analysis of SGD under convex functions requires the assumption that the variance of the gradient is bounded Bottou et al. (2016); BID1 ; BID0 .

Instead of making such an assumption, we next prove that SGD enjoys a nearlyconstant bound on the variance up to a logarithmic factor of t in learning the ReLU model.

Proposition 1 (Variance bound).

Apply SGD to solve problem (P) with any initialization.

If there exists T such that for all t > T , w t either stays in the linearly separable region, or in W + J , then with stepsize η k = (k + 1) −α where 0.5 < α < 1, the variances of the stochastic gradients sampled by SGD along the iteration path satisfy that for all t, DISPLAYFORM1 Proposition 1 shows that the summation of the norms of the stochastic gradients grows logarithmically fast.

This implies that the variance of the stochastic gradients is well-controlled.

In particular, if we choose η k = (k+1) −1/2 , then the bound in Proposition 1 implies that the term E ∇ (w k , z ξ k ) 2 stays at a constant level.

Based on the variance bound in Proposition 1, we next establish the convergence rate of SGD for learning the ReLU model.

Throughout, we denote w t := then with the stepsize η k = (k + 1) −α , where 0.5 < α < 1, the averaged iterates generated by SGD satisfies DISPLAYFORM2 If there exist T such that for all t > T , w t stays in W + J , then with the same stepsize DISPLAYFORM3 Theorem 3.3 establishes the convergence rate of the expected risk of the averaged iterates generated by SGD.

It can be seen that the convergence of SGD achieves different loss values corresponding to global and local minimum in different regions.

The stepsize is set to be diminishing to compensate the variance introduced by SGD.

In particular, if α is chosen to be sufficiently close to 0.5, then the convergence rate is nearly of the order O(ln 2 t/ √ t), which matches the standard result of SGD in convex optimization up to an logarithmic order.

Theorem 3.3 also implies that the convergence of SGD is attained as Ew t → +∞ at a rate of O(ln t).

We note that the analysis of Theorem 3.3 is different from that of SGD in traditional convex optimization, which requires the global minimum to be achieved at a bounded point and assumes the variance of the stochastic gradients is bounded by a constant Shalev-Shwartz et al. FORMULA19 −α where 0.5 < α < 1, the sequence of the averaged iterate {w t } t generated by SGD satisfies DISPLAYFORM4 If there exist T such that for all t > T , w t stays in W + J , then with the same stepsize DISPLAYFORM5 Theorem 3.4 shows that the direction of the expected averaged iterate E[w t ] generated by SGD converges to the max-margin direction w + , without any explicit regularizer in the objective function.

The proof of Theorem 3.4 requires a detailed analysis of the SGD update under the ReLU model and is substantially different from that under the linear model Soudry et al. (2018); Ji & Telgarsky (2018); Nacson et al. (2018a; b) .

In particular, we need to handle the variance of the stochastic gradients introduced by SGD and exploit its classification properties under the ReLU model.

We next provide an example class of datasets (which has been studied in Combes et al. (2018) ), for which we show that SGD stays stably in the linearly separable region.

Proposition 2.

If the linear separable samples {z 1 , . . .

, z n } satisfy the following conditions given in Combes et al. FORMULA0 : DISPLAYFORM6 then there exists at ∈ N such that for all t ≥t the sequence generated by SGD stays in the linearly separable region, as long as SGD is not initialized at the local minima described in item 3 of Theorem 3.1.

The leaky ReLU activation takes the form σ(v) = max (αv, v) , where the parameter (0 ≤ α ≤ 1).

Clearly, leaky ReLU takes the linear and ReLU models as two special cases, respectively corresponding to α = 0 and α = 1.

Since the convergence of GD/SGD of the ReLU model is very different from that of the linear model, a natural question to ask is whether leaky ReLU with intermediate parameters 0 < α < 1 takes the same behavior as the linear or ReLU model.

It can be shown that the loss function in problem (P) under the leaky ReLU model has only asymptotic global minima achieved by w * in the separable region with infinite norm (there does not exist asymptotic local minima).

Hence, the convergence of GD is similar to that under the linear model, where the only difference is that the max-margin classifier needs to be defined based on leaky ReLU as follows.

For the given set of linearly separable data samples, we construct a new set of data z * DISPLAYFORM0 + , x * i = αx i , ∀i ∈ I − , and y * i = y i , ∀i ∈ I + ∪ I − .

Essentially, the data samples with label −1 are scaled by the parameter α of leaky ReLU.

Without loss of generality, we assume that the max-margin classifier for data {x * i } passes through the origin after a proper translation.

Then, we define the max-margin direction of data X * as DISPLAYFORM1 Then, following the result under the linear model in Soudry et al. (2017) , it can be shown that GD with arbitrary initialization and small constant stepsize for solving problem (P) under the leaky ReLU model satisfies that L(w) converges to zero, and w converges to the max-margin direction, i.e., lim t→∞ wt wt = w * , with its norm going to infinity.

Furthermore, following our result of Theorem 3.4, it can be shown that for SGD applied to solve problem (P) with any initialization, if there exists T such that for all t > T w t stays in the linearly separable region, then with the stepsize η k = (k + 1) −α , 0.5 < α < 1, the sequence of the averaged iterate {w t } t generated by SGD satisfies DISPLAYFORM2 Thus, for SGD under the leaky ReLU model, the normalized average of the parameter vector converges in direction to the max-margin classifier.

In this subsection, we extend our study of the ReLU model to the problem of training a one-hiddenlayer ReLU neural network with K hidden neurons for binary classification.

Here, we do not assume linear separability of the dataset.

The output of the network is given by DISPLAYFORM0 where W = [ w 1 , w 2 , · · · , w K ] with each column w k representing the weights of the kth neuron in the hidden layer, DISPLAYFORM1 denotes the weights of the output neuron, and σ(·) represents the entry-wise ReLU activation function.

We assume that v is a fixed vector whose entries are nonzero and have both positive and negative values.

Such an assumption is natural as it allows the model to have enough capacity to achieve zero loss.

The predicted label is set to be the sign of f (x), and the objective function under the exponential loss is given by DISPLAYFORM2 Our goal is to characterize the implicit bias of GD and SGD for learning the weight parameters W of the multi-neuron model.

In general, such a problem is challenging, as we have shown that GD may not converge to a desirable classifier even under the single-neuron ReLU model.

For this reason, we adopt the same setting as that in (Soudry et al., 2017, Corollary 8) , which assumes that the activated neurons do not change their activation status and the training error converges to zero after a sufficient number of iterations, but our result presented below characterizes the implicit bias of GD and SGD in the original feature space, which is different from that in (Soudry et al., 2017, Corollary 8) .

We define a set of vectors DISPLAYFORM3 , where A j i = 1 if the sample x i is activated on the jth neuron, i.e., w j x i > 0, and set A j i = 0 otherwise.

Such an A i vector is referred to as the activation pattern of x i .

We then partition the set of all training samples into m subsets B 1 , B 2 , · · · , B m , so that the samples in the same subset have the same ReLU activation pattern, and the samples in different subsets have different ReLU activation patterns.

We call B h , h ∈ [m] as the h-th pattern partition.

Let w h = k∈{j:A j h =1} v k w k .

Then, for any sample x ∈ B h , the output of the network is given by DISPLAYFORM4 We next present our characterization of the implicit bias property of GD and SGD under the above ReLU network model.

We define the corresponding max-margin direction of the samples in B h as DISPLAYFORM5 Then the following theorem characterizes the implicit bias of GD under the multi-neuron network.

Theorem 4.1.

Suppose that GD optimizes the loss L(W) in eq. (3) to zero and there exists T such that for all t > T , the neurons in the hidden layer do not change their activation status.

If A h1 ∧ A h2 = 0 (where "∧" denotes the entry-wise logic operator "AND" between digits zero or one) for any h 1 = h 2 , then the samples in the same pattern partition of the ReLU activation have the same label, and DISPLAYFORM6 Differently from (Soudry et al., 2017, Corollary 8) which studies the convergence of the vectorized weight matrix so that the implicit bias of GD is with respect to features being lifted to an extended dimensional space, Theorem 4.1 characterizes the convergence of the weight parameters and the implicit bias in the original feature space.

In particular, Theorem 4.1 implies that although the ReLU neural network is a nonlinear classifier, f (x) is equivalent to a ReLU classifier for the samples in the same pattern partition (that are from the same class), which converges in direction to the maxmargin classifier w h of those data samples.

We next letw k=0 w h (t).

Then the following theorem establishes the implicit bias of SGD.

Theorem 4.2.

Suppose that SGD optimizes the loss L(W) in eq. (3) so that there exists T such that for any t > T , L(W) < 1/n, the neurons in the hidden layer do not change their activation status, and for any h 1 = h 2 , A h1 ∧ A h2 = 0.

Then, for the stepsize η k = (k + 1) −α , 0.5 < α < 1, the samples in the same pattern partition of the ReLU activation have the same label, and DISPLAYFORM7 Similarly to GD, the averaged SGD in expectation maximizes the margin for every sample partition.

At the high level, Theorem 4.1 and Theorem 4.2 imply the following generalization performance of the ReLU network under study.

After a sufficiently large number of iterations, the neural network partitions the data samples into different subsets, and for each subset, the distance from the samples to the decision boundary is maximized by GD and SGD.

Thus, the learned classifier is robust to small perturbations of the data, resulting in good generalization performance.

In this paper, we study the problem of learning a ReLU neural network via gradient descent methods, and establish the corresponding risk and parameter convergence under the exponential loss function.

In particular, we show that due to the possible existence of spurious asymptotic local minima, GD and SGD can converge either to the global or local max-margin direction, which in the nature of convergence is very different from that under the linear model in the previous studies.

We also discuss the extensions of our analysis to the more general leaky ReLU model and multi-neuron networks.

In the future, it is worthy to explore the implicit bias of GD and SGD in learning multilayer neural network models and under more general (not necessarily linearly separable) datasets.

A PROOF OF THEOREM 3.1The gradient ∇L(w) is given by DISPLAYFORM0 If y i w * x i ≥ 0 for all x i ∈ I + ∪ I − , then as α → +∞, we have, , DISPLAYFORM1 and ∇L(αw DISPLAYFORM2 and ∇L(αw DISPLAYFORM3 The proof is now complete.

First consider the case when w + is in linearly separable region and the local minimum does not exist along the updating path.

We call the region where all vectors w ∈ R d satisfy w x i < 0 for all i ∈ I − as negative correctly classified region.

As shown in Soudry et al. (2017) , L(w) is non-negative and L-smooth, which implies that DISPLAYFORM0 Based on the above inequality, we have DISPLAYFORM1 which, in conjunction with 0 < η < 2/L, implies that DISPLAYFORM2 Thus, we have ∇L(w k ) 2 → 0 as k → +∞. By Theorem 3.1, ∇L(w k ) vanishes only when all samples with label −1 are correctly classified, and thus GD enters into the negative correctly classified region eventually and diverges to infinity.

Soudry et al. (2017) Theorem 3 shows that when GD diverges to infinity, it simultaneously converges in the direction of the max-margin classifier of all samples satisfying w t x i > 0.

Thus, under our setting, GD either converges in the direction of the global max-margin classifer w + : DISPLAYFORM3 or the local max-margin classifier w DISPLAYFORM4 Next, consider the case when w + is not in linearly separable region, and the local minimum does not exist along the updating path.

In such a case, we conclude that GD cannot stay in the linearly separable region.

Otherwise, it converges in the direction of w + that is not in linearly separable region, which leads to a contradiction.

If the asymptotic local minimum w + J exists, then GD may converge in its direction.

If w + J does not exist, GD cannot stay in both the misclassified region and linearly separable region, and thus oscillates between these two regions.

In the case when GD reaches a local minimum, by Theorem 3.1, we have ∇L(w * ) = 0, and thus GD stops immediately and does not diverges to infinity.

Example 1 FIG4 .

The dataset consists of two samples with label +1 and one sample with label −1.

These samples satisfy x 1 x 3 < 0 and x 1 x 2 < 0.For this example, if we initialize GD at the green classifier, then GD converges to the max-margin direction of the sample (x 1 , +1).

Clearly, such a classifier misclassifies the data sample (x 2 , +1).

Example 2 FIG4 .

The dataset consists of one sample with label +1 and one sample with label −1.

These two samples satisfy 0 < x 1 x 2 ≤ 0.5 x 2 2 .For this example, if we initialize at the green classifier, then GD oscillates around the direction x 2 / x 2 and does not converge.

Consider the first iteration.

Note that the sample z 3 has label −1, and from the illustration of FIG4 (left) we have w 0 x 3 < 0, w 0 x 2 < 0 and w 0 x 1 > 0.

Therefore, only the sample z 1 contributes to the gradient, which is given by DISPLAYFORM0 By the update rule of GD, we obtain that for all t DISPLAYFORM1 By telescoping eq. FORMULA37 , it is clear that any w t x 2 < 0 for all t since x 1 x 2 < 0.

This implies that the sample z 2 is always misclassified.

Since we initialize GD at w 0 such that w 0 x 1 > 0 and w 0 x 2 < 0, the sample z 2 does not contribute to the GD update due to the ReLU activation.

Next, we argue that there must exists a t such that w t x 2 > 0.

Suppose such t does not exist, we always have w t x 1 = (w 0 + t−1 k=0 exp(−w k x 1 )x 1 ) x 1 > 0.

Then, the linear classifier w t generated by GD stays between x 1 and x 2 , and the corresponding objective function reduces to a linear model that depends on the sample z 1 (Note that z 2 contributes a constant due to ReLU activation).

Following from the results in Ji & Telgarsky (2018); Soudry et al. FORMULA0 for linear model, we conclude that w t converges to the max-margin direction x1 x1 as t → +∞. Since x 1 x 2 > 0, this implies that w t x 2 > 0 as t → +∞, contradicting with the assumption.

Next, we consider the t such that w t x 1 > 0 and w t x 2 > 0, the objective function is given by L(w t ) = exp(−w t x 1 ) + exp(w t x 2 ), and the corresponding gradient is given by DISPLAYFORM0 Next, we consider the case that w t x 1 > 0 for all t. Otherwise, both of x 1 and x 2 are on the negative side of the classifier and GD cannot make any progress as the corresponding gradient is zero.

In the case that w t x 1 > 0 for all t, by the update rule of GD, we obtain that DISPLAYFORM1 Clearly, the sequence {w t x 2 } t is strictly decreasing with a constant gap, and hence within finite steps we must have w t x 2 ≤ 0.

Since SGD stays in the linearly separable region eventually, and hence only the data samples in I + contribute to the gradient update due to the ReLU activation function.

For this reason, we reduce the original minimization problem (P) to the following optimization DISPLAYFORM0 which corresponds to a linear model with samples in I + .

Similarly, if SGD stays in W + J , only the data samples in J + contribute the the gradient update, the original minimization problem (P) is reduced to DISPLAYFORM1 The proof contains three main steps.

Step 1: For any u, bounding the term E w t − u 2 : By the update rule of SGD, we have DISPLAYFORM2 where M t = 2η t−1 ∇L(w t−1 ) − ∇ (w t−1 , z ξt ), w t−1 − u .

By convexity we obtain that ∇L(w t−1 ), w t−1 − u ≥ L(w t−1 ) − L(u).

Then, eq. (9) further becomes DISPLAYFORM3 Telescoping the above inequality yields that DISPLAYFORM4 (11) Taking expectation on both sides of the above inequality and note that EM t = 0 for all t, we further obtain that DISPLAYFORM5 Note that ≤ 1 whenever the data samples are correctly classified and for all i ∈ I + , x i ≤ B, and without loss of generality, we can assume B < √ 2.

Hence, the term E ∇ (w k , z ξ k ) 2 can be upper bounded by DISPLAYFORM6 Then, noting that η k ≤ 1, eq. (12) can be upper bounded by DISPLAYFORM7 Next, set u = (ln(t)/γ)ŵ + and note thatŵ + x i ≥ γ for all i ∈ I + , we conclude that DISPLAYFORM8 Substituting this into the above inequality and noting that η k = (k + 1) −α and 0.5 < α < 1, we further obtain that DISPLAYFORM9 Step 2: lower bounding E w t − u 2 : Note that only the samples in I + contribute to the update rule.

By the update rule of SGD, we obtain that DISPLAYFORM10 which further implies that DISPLAYFORM11 Then, we can lower bound w t − u as DISPLAYFORM12 Taking the expectation of w t − u 2 : DISPLAYFORM13 where (i) follows from Jensen's inequality.

Step 3: Upper bounding DISPLAYFORM14 Combining the upper bound obtained in step 1 and the lower bound obtained in step 2 yields that DISPLAYFORM15 Solving the above quadratic inequality yields that DISPLAYFORM16 E PROOF OF THEOREM 3.3The proof exploits the iteration properties of SGD and the bound on the variance of SGD established in Proposition 1.We start the proof from eq. FORMULA0 , following which we obtain DISPLAYFORM17 (16) Taking the expectation on both sides of the above inequality yields that DISPLAYFORM18 which, after telescoping, further yields that DISPLAYFORM19 (17) DISPLAYFORM20 in conjunction with eq. (17), yields that DISPLAYFORM21 where (i) follows from the fact that DISPLAYFORM22 Thus, we can see that L(w t ) decreases to 0 at a rate of O(ln 2 (t)/t 1−α ).

If we choose α to be close to 0.5, the best convergence rate that can be achieved is O(ln 2 (t)/ √ t).F PROOF OF THEOREM 3.4 DISPLAYFORM23 We first present four technical lemmas that are useful for the proof of the main theorem.

Lemma F.1.

Given the stepsize η k+1 = 1/(k + 1) −α and the initialization w 0s , then for t ≥ 1, we have DISPLAYFORM24 Lemma F.2.

Let X + represent the data matrix of all samples with the label +1, with each row representing one sample.

Then we have: DISPLAYFORM25 ∆ n−1 is the simplex in R n .

If the equality holds (i.e., the strong duality holds) at q andŵ + , then they satisfyŵ DISPLAYFORM26 andŵ + is the max-margin classifier of samples with the label +1.

DISPLAYFORM27 We next apply the lemmas to prove the main theorem.

Taking the expectation of the SGD update rule yields that DISPLAYFORM28 ).

Applying the above equation recursively, we further obtain that DISPLAYFORM29 which further leads to DISPLAYFORM30 Next, we prove the convergence of the direction of E[w t ] to the max-margin direction as follows.

1 2 DISPLAYFORM31 where (i) follows from Lemma F.4, (ii) follows from Lemma F.3 and (iii) is due to eq. (19).

Since following from Lemma F.1 we have that Ew t = O(ln(t)), the above inequality then implies that DISPLAYFORM32

Proof of Lemma F.1.

Since x i ≤ B for all i, we obtain that DISPLAYFORM0 By convexity, we have that EL(w t ) ≥ L(Ew t ), combining which with the above bounds further yields DISPLAYFORM1 That is, the increasing rate of E w t is at least O(ln(t)).Proof of Lemma F.2.

Following from the definition of the max-margin, we have DISPLAYFORM2 where f * (a) = max and g(e) = e , respectively, where a, b, c, d are generic vectors.

We also denote ∂f (c) and ∂g(e) the subgradient set of f and g at e and c respectively.

By the Fenchel-Rockafellar duality Borwein & Lewis (2010), we obtain that DISPLAYFORM3 In particular, the strong duality holds at q andŵ + if and only if −X + w ∈ ∂f (q) andŵ + ∈ ∂g(X + q).

Thus, we conclude thatŵ DISPLAYFORM4 Proof of Lemma F.3.

By Taylor's expansion and the update of SGD, we obtain that DISPLAYFORM5 where w = θw k−1 + (1 − θ)w k for certain 0 ≤ θ ≤ 1, and is in the linear separable region.

Note that for any v, DISPLAYFORM6 where S is the maximum of L(w) in the linearly separable region.

We note that S < +∞ because w → ∞ in the linearly separable region and hence L(w) → 0.

Taking the expectation on both sides of eq. (20) and recalling that DISPLAYFORM7 we obtain that DISPLAYFORM8 Denote X ∈ R n×d as the data matrix with each row corresponding to one data sample.

The derivative of the empirical risk can be written as ∇L(w) = X T l(w)/n, where l(w) = [ (w, z 1 ), (w, z 2 ), . . .

, (w, z n )] .

Then, we obtain that EL(w p ) = 1 n + i∈I + E exp(−w p x i ) = 1 n + E(l(w p )) 1 and E∇L(w p ) = 1 n + i∈I + E exp(−w p x i )x i = 1 n + X + E(l(w p )).Based on the above relationships and Lemma F.2, we obtain that Proof of Lemma F.4.

Define h(y) = ln 1 n + i∈I + exp(y i ) , and then its dual function h * (q) = ln n + + q i ln(q i ) ≤ ln n + .

Following from Lemma F.2,ŵ + = 1 γ X +T q. Then, by the FenchelYoung inequality, we obtain that DISPLAYFORM9 (ln E(L(w k )) + ln n + ).

Under our ReLU model, in the linearly separable region, the gradient ∇L(w) is given by ∇L(w) = − 1 n n i=1 y i 1 {w xi>0} exp(−y i w x i )

x i = − 1 n i∈I + exp(−w x i )x i .Thus, only samples with positive classification output, i.e. σ(w t x ξt ) > 0, contribute to the SGD updates.

We first prove w t < +∞ when there exist misclassified samples.

Suppose, toward contradiction, that w t = +∞ as t → +∞ when misclassified samples exist.

Note that DISPLAYFORM0 Since w t is infinite, at least one of the coefficients α i , i = 1, · · · , n is infinite.

No loss of generality, we assume α p = +∞. Then, the inner product DISPLAYFORM1 Based on the data selected in Proposition 2, we obtain for ∀i ∈

I − ∪ I + ∀j ∈

I + , y i x i x j > 0 ∀j ∈ I − , y i x i x j < 0, which, in conjunction with eq. (24), implies that, if there exist j ∈ I + , then the first term in the right side of eq. FORMULA19 is finite, the second term is positive, and the third term is positive and infinite.

As a result, we conclude that for ∀j ∈ I + , w t x j > 0 as t → +∞. Similarly, we can prove that for ∀j ∈ I − , w t x j ≤ 0 as t → +∞, which contracts that w t x j > 0 .

Thus, if there exist misclassified samples, then we have w t < +∞.Based on the update rule of SGD, we have, for any j w t+1 x j − w t x j = η exp(−y ξt w t x ξt )

y ξt x ξt x j = ξt,j .It can be shown that ∀j ∈ I − ∪ I + , y ξt ξt,j > 0, which, combined with eq. (25), implies that, if one sample is correctly classified at iteration t, it remains to be correctly classified in the following iterations.

Next, we prove that when w t < +∞, all samples are correctly classified within finite steps.

Define DISPLAYFORM2 Since w t < ∞, there exists a constant C such that w t < C for all t. Let D = max i∈I + x i .

Then, we obtain, for any j ∈ I + and ξ t ∈ I + , ξt,j = η exp(−w t x ξt )x ξt x j ≥ η exp(−CD) ++ , and for any j ∈ I + and ξ t ∈ I − , ξt,j = −η exp(w t x ξt )x ξt x j ≥ η +− .Combining the above two inequalities ∀j ∈ I + yields ξt,j ≥ η min {exp(−CD) ++ , η +− }.

<|TLDR|>

@highlight

We study the implicit bias of gradient methods in solving a binary classification problem with nonlinear ReLU models.