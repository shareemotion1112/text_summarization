The field of deep learning has been craving for an optimization method that shows outstanding property for both optimization and generalization.

We propose a method for mathematical optimization based on flows along geodesics, that is, the shortest paths between two points, with respect to the Riemannian metric induced by a non-linear function.

In our method, the flows refer to Exponentially Decaying Flows (EDF), as they can be designed to converge on the local solutions exponentially.

In this paper, we conduct experiments to show its high performance on optimization benchmarks (i.e., convergence properties), as well as its potential for producing good machine learning benchmarks (i.e., generalization properties).

Due to recent progress in the field of machine learning, it becomes more and more important to develop and sophisticate methods of solving hard optimization problems.

At the same time, in this field, such methods are additionally required to elicit decent generalization performance from statistical models.

An efficient method of mathematical optimization, however, does not always produce sufficient generalization properties, since these are involved with two distinct mathematical problems; The former is to find one of the solutions which minimize a given (possibly non-convex) objective function, and the latter is to adjust parameters so that a statistical estimator achieves its best result.

To address such a hard issue, we introduce a new mathematical perspective on optimization, and develop a method for machine learning based on this perspective.

We then empirically show its rapid convergence rate and high compatibility with deep learning techniques, as well as good statistical properties.

In this field, many optimization methods have been proposed and modified so that they fit specific problems or models.

One of the current standard methods is the gradient descent method.

The method tends to converge slowly in general optimization problems.

However, with various specific techniques, such as mini-batch training and batch normalization BID9 ), it has been found to be efficient for state-of-the-art purposes in the field of deep learning.

Another class of methods that are now becoming popular and standard is adaptive methods, such as AdaGrad BID6 ) and Adam (Kingma & Ba (2015) ).

Compared to the gradient descent method, these methods have been shown to improve convergence rates with almost the same computational cost as the gradient descent method, but are reported to result in poor statistical outcomes in some cases of machine learning BID16 ).Other class of methods that have been thoroughly studied in the theory of mathematical optimization is second-order methods, such as the Newton method and the Gauss-Newton method.

These methods possess great convergence properties, and in particular, have a potential to overcome plateau's problems BID5 ).

Furthermore, when it comes to applications in stochastic settings, the method based on the Gauss-Newton Matrix (or Fisher information Matrix) is shown to asymptotically attain the best statistical result, which is called Fisher efficiency (see BID0 ).

Despite these attractive characteristics, the methods have not yet been spotlighted in the field of machine learning due to several severe drawbacks; They suffer from high computational cost in general and their useful properties are no longer guaranteed in practical settings (see Section 12 in BID12 ).

One of the continuously developing second-order methods in this field, K-FAC BID1 , BID7 ), successfully produced high convergence rate empirically with relatively low computational cost.

However, it still requires much effort to become compatible with some deep learning techniques.

In addition, it is unclear whether the method has advantages in generalization performance.

In our approach, by introducing a Riemannian metric induced by non-linear functions, we constitute dynamical systems which describe motions along the shortest route from arbitrary initial points to the zeros of non-linear functions on the corresponding Riemannian manifold, that is, geodesic with respect to the Riemannian metric.

One of the remarkable characteristics of our approach is that it enables us to flexibly design flows of such dynamical systems to control convergence rates.

The results for the flows are then applicable to mathematical optimization problems, in particular, with deep neural network (DNN) models.

In this paper, after providing mathematical ground of our methods, we experimentally demonstrate their performance in various aspects, from convergence rates to statistical properties.

We start by establishing some essential properties of dynamics which are effective for the analysis of non-linear equations.

Let F : R N ??? R N be a smooth function and J be the Jacobian with variable w ??? R N , that is, J = ???F/???w.

In this section, we deal with the well-posed case that there exists a connected closed subset ??? ??? R N , where J is regular and the equation has a unique solution ??.

Therefore, the positive matrix G = J T J induces a Riemannian metric g on ??? and (???, g) becomes a Riemannian manifold under some appropriate conditions.

Let us then consider time evolution of variable w on this manifold.

Our main purpose in this section is to study the characteristics of dynamical systems in which w(t) moves on geodesics between any point in ??? and ?? with respect to the metric g.

Let L be a Lagrangian given by DISPLAYFORM0 with v = dw/dt (also written as???).

The Euler-Lagrange equation for L is then expressed as dp dt DISPLAYFORM1 with momentum vector p = Gv.

If the boundary condition at two points in ???, w(t 0 ) = w 0 , w(t 1 ) = w 1 , is imposed on (2), a geodesic between w 0 and w 1 is obtained as the solution.

In contrast, if we give an appropriate initial condition, w describes the motion along the geodesic from w 0 to w 1 .

In fact, the following statement holds; Theorem 2.1.

Let w 0 be an arbitrary point in ??? and (w(t), p(t)) be a solution of equation FORMULA1 with the following initial condition; DISPLAYFORM2 Then w satisfies F w(t) = (1 ??? t)F w 0 , (4) for t ??? [0, 1].

In particular, w(t) passes through the point which is a solution of non-linear equation DISPLAYFORM3 We briefly describe the outline of the proof for the statement above.

Throughout this paper, we regard functions of w as those of t in the trivial way; for instance, F (t) = F (w(t)).

Note that p can be expressed as p = J T dF/dt.

Then using the Beltrami identity for (2) with the initial condition above leads to the equation DISPLAYFORM4 where F 0 = F (0).

Thus, a closed form expression is obtained as DISPLAYFORM5 which gives F (1) = 0 as asserted in Theorem 2.1.

Now we take a different expression that the coefficient (1 ??? t) in (6) is replaced by a different monotonically decreasing function, that is, DISPLAYFORM6 where ?? denotes a monotonically decreasing smooth function from (t 0 , t 1 ) onto (0, 1).

Then, we give the following differential equation whose solution is of the closed form (7); DISPLAYFORM7 A motion described by this differential equation differs from the one that is described by (2), but these two motions are along the same geodesic.

Theorem 2.2.

Let w 0 ??? ??? be an arbitrary point.

The differential equation DISPLAYFORM8 with an initial condition w 0 = w(t 0 ) has a unique solution that satisfies F (w(t 1 )) = 0.

Furthermore, the orbit under flow f defined by f (w 0 , t) = w(t) coincides with that of the geodesic equation (2).Note that since equation FORMULA8 is equivalent to equation FORMULA7 , the orbit is invariant under coordinate transformations.

With respect to the choice of ??, the end point t 1 can be set as ??? under some appropriate conditions and Theorem 2.2 still holds in the sense that DISPLAYFORM9 In particular, if we set ??(t) = e ???t , then ??(t) = ???1 and F can be represented as F (t) = e ???t F 0 , so that the convergence rate of F is exponential.

Definition 2.3.

Let w be a solution of the differential equation DISPLAYFORM10 with an initial condition w 0 = w(t 0 ).

The flow f (w 0 , t) = w(t) is called an exponentially decaying flow (EDF) of non-linear equation F (w) = 0.For the end of this section, we present a connection between EDF and the classical Newton method.

If we apply the Euler method to the differential equation (9) with step size t, the corresponding iteration step can be written as DISPLAYFORM11 which recovers the Newton method with step size ?? i = t ?? ??(?? i ).

Consider smooth functions ?? : DISPLAYFORM0 In this section, we develop a method based on EDF for a mathematical optimization problem of the form DISPLAYFORM1 In this section, the area ??? ??? R N is supposed to be a compact subset where there exists no stationary point except for a minima.

DISPLAYFORM2 An example of such problems is least squares one in which L is given by DISPLAYFORM3 In this case, F = ??. In particular, if M = N and a minimal value of loss function L = L ??? ?? is zero, the optimization problem is equivalent to solving non-linear equation F (w) = 0.For optimization problems, we set up a standard equation that the gradient of loss function is zero, that is, ??? L(w) = 0.

Note that the gradient can be expressed as ??? L(w) = J T ?? F (w) with Jacobian J ?? = ?????/???w.

Applying Theorem 2.2, we obtain the differential equation DISPLAYFORM4 where H is the Hessian of loss function L with respect to w. Since second order differentiation is typically computationally heavy, we seek other equations which possess almost the same property as (15) especially in terms of asymptotic behavior.

Let us decompose momentum H??? as DISPLAYFORM5 where J F denotes the Jacobian of F , and G is a symmetric matrix defined by DISPLAYFORM6 with Hessian matrix H L of L with respect to ??. We then consider the following equation instead of FORMULA0 ; DISPLAYFORM7 This equation no longer describes the motion along the geodesic related to (15) in general.

However, if M = N and J ?? is invertible, then w moves on another geodesic with respect to a different metric DISPLAYFORM8 In addition, if ??(t) = e ???t , F converges exponentially, which implies that ??? L = J ?? F also converges exponentially in (18) as well as in FORMULA0 .

In general cases, if a condition that DISPLAYFORM9 is satisfied, then F converges to 0.

This shows that in the neighborhood of solution ?? of equation ??? L = 0, the momentum G??? sufficiently approximates H??? by (16).

Definition 3.1.

The flow given by DISPLAYFORM10 is referred to EDF of type H and written as EDF-H. Similarly, the flow given by DISPLAYFORM11 is referred to EDF of type G and written as EDF-G.

Like second order methods, in EDF-based methods, matrix inversion has to be carried out, which requires expensive computational cost, particularly in large systems.

Moreover, we often encounter rank deficient matrices in optimization problems.

To deal with rank deficiency, in general, we need pseudo-inverse matrices which are more computationally heavy.

Therefore, instead of the inverse of matrix A = G, H, for fixed v ??? R M , we consider a projection which maps r = arg min DISPLAYFORM0 One of the basic methods to construct such a projection for indefinite symmetric systems is the minimum residual method BID13 ), which requires only the matrix multiplication.

Therefore, for numerical computations, we use the following differential equation approximated by the k-th order Krylov subspace; DISPLAYFORM1 k is a hyperparameter that interpolates between the gradient method and the method based on (15) or (18).

In fact, in the case that k = 1, equation FORMULA1 has the form DISPLAYFORM2 which reduces to a kind of gradient methods, but the coefficient c conveys information about G or H unlike the standard gradient methods.

Next, similar to the Levenberg-Marquardt algorithm, we modify equation FORMULA0 by adding a damping factor to G in order that the method becomes more stable.

So, we set DISPLAYFORM3 where ?? is a continuous positive function of t. Then, we take the same approximation as FORMULA1 with A = G+??I. The damping factor ?? plays several important roles in solving practical problems.

First, it has solutions well-behaved near singularities.

Even in the case that G is not invertible, equation FORMULA1 can be defined and solved unlike (18).

If we choose ?? such that it approaches to 0 as rapidly as the gradient in (18), the asymptotic behavior of FORMULA1 is almost the same as that of FORMULA0 .

Particularly, in the case that ?? = ???1, we set ?? = a J T ?? F b with a, b > 0, so that the convergence rate of (24) stays exponential.

Second, the damping factor makes the method compatible with stochastic approaches such as mini-batch training, in deep learning models.

Since the orbit of FORMULA1 is affected by the gradient due to the damping factor ??, the method based on this equation could take advantage of stochastic approaches as other gradient methods do. (For implementation of the algorithm, see Appendix A.) Finally, to accelerate the EDF-based methods, it is sometimes effective to change equation FORMULA0 into a second-order differential equation, particularly in the case in which the approximation method with k is applied.

Specifically, we take the equation DISPLAYFORM4 where ?? is a real-valued function of t. The convergence properties of the methods are controlled by the following differential equation; DISPLAYFORM5 There are two ways of determining ??.

The first one is to set ?? = ?? with constant ?? (W1), which leads to a similar scheme to the momentum gradient decent method.

The other one is to set ??(t) = ??t ???1 (W2).

In this setting, the equation can be discretized in a similar way as described in BID15 , which is analogous to the Nesterov's acceleration scheme.

In this section, we present how optimization problems are set up in the field of deep learning.

Let x = {x j } n???1 j=0 and y = {y j } n???1 j=0 denote training datasets of input and output, respectively, where n is the size of dataset.

Let d x and d y be dimensions of input data and output data, respectively.

We write ?? nn for neural networks with parameters w ??? R N , and define ?? by the direct sum of vectors {?? j } given by ?? j (w) = ?? nn (x j , w), that is, ?? = ????? j .

Note that M = n ?? d y in this case.

Then finding a minima of a given loss function is proposed as a standard optimization problem to train networks.

For the image classification tasks, there are two typical cases of setting loss functions.

In the first case, the loss is set as (14).

As already mentioned, in this case, F = ?? and H L = I. In the second case, the loss is given by cross entropy with softmax function, that is, DISPLAYFORM0 where ?? denotes the softmax function and ?? j = ??(?? j ).

In this case, F is expressed by the direct sum such that F = ???F j with DISPLAYFORM1 where s j denotes a sum of all elements in vector y j for each j. Note that if each y j is given as a probability vector, then s j = 1.

Moreover, H L is expressed as H L = ???H j with DISPLAYFORM2 where ??? denotes the outer product.

In both cases, the loss functions take the minimum value 0 if and only if F = 0.

In this study, we conducted following three groups of experiments; First, we examined optimization performance of the EDF-based methods on a data-fitting problem with a simple convolutional neural network model.

Second, we tested both optimization and generalization performance in standard settings of classification tasks in which we employed residual network models (Resnet He et al. (2016) ) and CIFAR-10/100 datasets.

Third, we incorporated some techniques of data augmentation and regularization in the training into our experiment as we would like to measure effectiveness of our methods in more practical settings.

The primary purpose of the experiments in this paper is not to pursue the state-of-the-art performance, but to explore how the statistical results pertain to those of optimization when the loss functions are minimized.

Therefore, we tuned hyperparameters of each method in accordance with the optimization benchmark.

It should be noted that the conjecture concerning non-convex optimization problems in the field of deep learning BID2 , BID4 ) is still an open problem (studied for linear cases in BID3 , BID10 ).

Hence, for experiments in this paper, we do not discuss whether each optimization method actually reaches to a global solution.

We evaluated convergence performance of EDF-based methods (type G and type H) on a data-fitting problem of CIFAR-10, that is, full-batch training in the context of deep learning.

The model we employed in these experiments was the convolutional neural network that consisted of two convolution filters with rectified linear units and max pooling layers, followed by two fully connected layers.

For EDF-based methods, the step size was fixed to 1.0, and no damping factors were used.

In addition, the acceleration technique derived from W2 of (25) was adapted, since W2 achieved better performance than W1.

A similar experiment with a different type of second-order methods was conducted in BID14 .First, we examined change in convergence performance of EDF-G depending on hyperparameter k, the order of Krylov subspace.

The results are illustrated in the left-hand side of FIG0 .

The presented trend is consistent with the theoretical fact that k interpolates between the gradient method (for small k) and the method based on dynamics (18) (for large k).

In other words, when k is small, the method is similar to a gradient method, which converges slow, but as k becomes larger, the method leads to better approximation of the inverse matrix, which gives a rapidly decaying flow.

Next, we compared EDF-G (k = 1, 30) with EDF-H and other standard optimizers in deep learning: gradient descent methods with Polyaks momentum scheme (Momentum) and Nesterov's acceleration scheme (NAG), and Adam.

The step sizes for Momentum, NAG, and Adam were fixed to 0.01, 0.001, and 0.001, respectively.

The right-hand side of FIG0 shows that both EDF-based methods made the loss decrease more rapidly than other standard optimizers.

Even when EDF-based methods were reduced to the gradient methods at k = 1, EDF-G outperformed standard optimizers.

The figure also presents the difference between EDF-G and EDF-H. While their convergence rates around extremes were almost the same, the overall convergence rates of EDF-G were better than that of EDF-H.As has been found, second-order-methods on full-batch training converge to the solution within a few iterations.

However, with such a training, the quality of generalization becomes worse compared to the time when stochastic approach with a mini-batch of small size is taken.

Like other secondorder-methods, EDF also suffers from a negative impact of full-batch training on generalization performance, though setting the hyperparameter k small makes EDF be compatible with stochastic approaches and take their advantages (see Appendix B).

In the following experiments, we compared both optimization and generalization performance of EDF-G with other methods, momentum stochastic gradient decent method (MSGD) and Adam on classification tasks for CIFAR-10/100.

The experiments were conducted, employing residual network models with batch normalization (Resnet-56 for CIFAR-10 and Resnet-110 for CIFAR-100), and working with a mini-batch of size 250.For CIFAR-100, during pre-investigation of the dataset, we found that 9 pairs of data, each of which has the same image but different labels, contaminated the neural network and might have had a non-negligible influence on both optimization and generalization performance (See Appendix C).

Therefore, in our experiments on CIFAR-100, we excluded them from the training dataset and used the rest of the data (size = 49982).At the end of each epoch, we computed training loss in the following manner.

First, the loss was defined as FORMULA1 where n is the total number of data.

Second, to calculate the statistics for batch normalization in the loss, as described in BID9 , we adopted so-called "inference mode" in which moving averages of mean and variance over mini-batches are used for batch-normalization.

For EDF, we tested the case k = 1 and k = 2, with the damping factor set as a = b = 1.

The momentum coefficient was fixed to ?? = 0.9 for the W1 acceleration (see 25).For each of the tasks, we ran tests on 10 different learning rates; for EDF between 0.1 and 2.0, for Momentum SGD between 0.1 and 10.0, for Adam between 0.0001 and 0.1.

We then chose the one with the best convergence performance for each optimizer.

For EDF with W2, to obtain stability in convergence, the learning rate was set to 0.2 times that of the initial values at the end of the 20-th epoch.

Such a change in learning rate in the middle of optimization did not bring advantages to either optimization or generalization performances for other methods including EDF with W1 (see Appendix D).The results are presented in Figures 2 and 3 .

As shown in the figures, with respect to the optimization performance, EDF reached an optimal solution with smaller error at higher convergence rate than Adam and Momentum SGD, even when k = 1, 2.

Moreover, EDF overall attained better generalization performance than other methods.

For classification tasks, generalization performance often improves by adopting several techniques, such as data augmentation and regularization.

In this group of experiments, employing the data augmentation based on random horizontal flip and shift and the L 2 regularization, we conducted comparisons between EDF and other methods that are similar to those in the previous sections.

When adopting these techniques, rate decay scheme has as large impact on optimization performance as learning rate itself.

Because effectiveness of each rate decay scheme much depends on optimization methods, it is almost impossible to find the best scheme that is applicable to all the methods.

For this reason, for MSGD and Adam, as initial learning rates, we chose one of the standard rates for MSGD and Adam, 0.1 and 0.001, respectively, and at the ends of the 100-th, 140-th, 180-th epochs, reset the rates to 0.1 times the rates at the moment.

For EDF, we ran tests on two different rates 0.5 and 0.75, the optimal rates found in the experiments of Section 6.2, and reset to 0.2 times those of the initial values, only at the end of the 100-th epoch, in order to demonstrate performance of the EDF more clearly.

Among the results obtained with EDF, we chose the ones with the best optimization performance.

The result of the comparison is presented in Figures 4 and 5.

As can be seen, we found a condition in which EDF achieved better performance than other methods on optimization while achieving sufficient levels of generalization.

Obtaining good statistical results from limited available data is a critical goal in machine learning.

To reach this goal, while developing an effective model is an essential approach, eliciting the best performance from the fixed model through optimization is important as well.

In our study, to examine the performance of our optimization methods, Exponentially Decaying Flows (EDF) based methods, we explored their generalization properties pertaining to results of optimization.

Our experiments showed that EDF-based methods are more likely to achieve optimal solutions which generalize the test data well than other standard optimizers are.

Therefore, EDF-based methods are considered to be optimization methods that have a high potential in their application to various tasks, and thus, are worthwhile to be sophisticated through future studies.

In terms of computation of the EDF-based methods with GPU, the Jacobian-vector-product can be carried out at almost the same cost as the gradient of loss function.

In fact, multiplying a vector by Jacobian and its transposed matrix (written as R-op and L-op, respectively) are implemented in combination with gradients of scholar functions.

For the psuedo-code for update scheme of the EDF-G with L/R-op, refer to Algorithm 1, and Algorithm 2 in particular case that k = 1.Algorithm 1: Update scheme for EDF-G with non-preconditioned MINRES Input : FIG3 shows the results of experiments using EDF for simple examples that compare a full-batch training with stochastic trainings.

In this example, the convolutional network similar to that used in Section 6.1 was employed on MNIST.

The curve labeled as "EDF F" depicts the result of Full-batch training per step, and those labeled as "EDF S" illustrate the results of stochastic trainings per epoch with a mini-batch of size 500.

DISPLAYFORM0

Let us divide the dataset {x, y} = {x j , y j } n???1 j=0 into distinct subsets of size p such that DISPLAYFORM0 Then, gradients and Jacobian-vector-multiplications are calculated as DISPLAYFORM1 where ?? (i) and F (i) are the subcomponents of ?? and F , respectively, corresponding to the decomposition above.

Thus, for a fixed k, if we set p as the same size as a mini-batch, then the computational cost per step of a full-batch training becomes almost the same as that per epoch of a mini-batch training.

The dataset of CIFAR-100 includes 9 pairs of irregular data, each of which has the same image but different labels.

These data are enumerated in Figure 7 .

For instance, the 8393-rd and the 36874-th images are the same, but they have different labels, "girl (class 35)" and "baby (class 2)."

Such pairs contaminate the training process.

In fact, in our experiment, when the network model was optimized with the full dataset, one of the images in each 9 pairs above could not be classified correctly, which resulted in stagnated training accuracy at 99.982 %.

Moreover, generalization also deteriorated when irregular data were contained.

For the details of these results, see Figure 8 .

D PERFORMANCES WITH THE RATE DECAY SCHEME Figure 9 and FIG0 present the optimization and generalization performance of each optimizer when the rate decay scheme was adopted to the experiment with the same setting as in Figure 2 and Figure 3 .

The rate decay scheme was that the learning rate was set to 0.2 times that of the initial values at the end of the 20-th epoch.

<|TLDR|>

@highlight

Introduction of a new optimization method and its application to deep learning.