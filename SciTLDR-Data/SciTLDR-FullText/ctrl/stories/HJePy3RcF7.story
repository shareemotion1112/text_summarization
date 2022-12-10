There is a stark disparity between the learning rate schedules used in the practice of large scale machine learning and what are considered admissible learning rate schedules prescribed in the theory of stochastic approximation.

Recent results, such as in the 'super-convergence' methods which use oscillating learning rates, serve to emphasize this point even more.

One plausible explanation is that non-convex neural network training procedures are better suited to the use of fundamentally different learning rate  schedules, such as the ``cut the learning rate every constant number of epochs'' method (which more closely resembles an exponentially decaying learning rate schedule); note that this widely used schedule is in stark contrast to the polynomial decay schemes prescribed in the stochastic approximation literature, which are indeed shown to be (worst case) optimal for classes of convex optimization problems.



The main contribution of this work shows that the picture is far more nuanced, where we do not even need to move to non-convex optimization to show other learning rate schemes can be far more effective.

In fact, even for the simple case of stochastic linear regression with a fixed time horizon, the rate achieved by any polynomial decay scheme is sub-optimal compared to the statistical minimax rate (by a factor of condition number); in contrast the ```''cut the learning rate every constant number of epochs'' provides an exponential improvement (depending only logarithmically on the condition number) compared to any polynomial decay scheme.

Finally, it is important to ask if our theoretical insights are somehow fundamentally tied to quadratic loss minimization (where we have circumvented minimax lower bounds for more general convex optimization problems)?

Here, we conjecture that recent results which make the gradient norm small at a near optimal rate, for both convex and non-convex optimization, may also provide more insights into learning rate schedules used in practice.

The recent advances in machine learning and deep learning rely almost exclusively on stochastic optimization methods, primarily SGD and its variants.

Here, these large scale stochastic optimization methods are manually (and often painstakingly) tuned to the problem at hand (often with parallelized hyper-parameter searches), where there is, as of yet, no class of "universal methods" which uniformly work well on a wide range of problems with little to no hyper-parameter tuning.

This is in stark contrast to non-stochastic numerical optimization methods, where it is not an overstatement to argue that the l-BFGS and non-linear conjugate gradient methods (with no hyper-parameter tuning whatsoever) have provided nearly unbeatable procedures (for a number of decades) on nearly every unconstrained convex and non-convex problem.

In the land of stochastic optimization, there are two dominant (and somewhat compatible approaches): those methods which often manually tune learning rate schedules to achieve the best performance BID13 Sutskever et al., 2013; BID11 BID10 and those methods which rely on various forms of approximate preconditioning BID6 Tieleman & Hinton, 2012; BID11 .

This works examines the former class of methods, where we seek a more refined understanding of the issues of learning rate scheduling, through both theoretical analysis and empirical studies.

Learning rate schedules for SGD is a rather enigmatic topic since there is a stark disparity between what is considered admissible in theory and what is employed in practice to achieve the best re-sults.

Let us elaborate on this distinction more clearly.

In theory, a vast majority of works starting with Robbins & Monro (1951) ; Polyak & Juditsky (1992) consider learning rates that have the form of η t = a b+t α for some a, b ≥ 0 and 1/2 < α ≤ 1 -we call these polynomial decay schemes.

The key property enjoyed by these polynomial decay schemes is that they are not summable but are square summable.

A number of works obtain bounds on the asymptotic convergence rates of such schemes.

Note that the focus of these works is to design learning rate schemes that work well for all large values of t. In contrast, practitioners are interested in achieving the best performance given a computational budget or equivalently a fixed time horizon T e.g., 100 passes on training dataset with a batch size of 128.The corresponding practically best performing learning rate scheme is often one where the step size is cut by a constant factor once every few epochs, or, equivalently, when no progress is made on a validation set BID13 BID8 ) (often called a dev set based decay scheme).

Such schemes are widely popular to the extent that they are available as schemes in deep learning libraries such as PyTorch 1 and several such useful tools of the trade are taught on popular deep learning courses 2 .

Furthermore, what is (often) puzzling (from a theory perspective) is the emphasis that is laid on "babysitting" the learning rates 3 to achieve the best performance.

Why do practitioners use constant and cut learning rate schemes while most of the theory work routinely works with polynomial decaying schemes?

Of course, implicit to this question is the view that both of these schemes are not equivalent.

Indeed if both of these were equivalent, one could parameterize the learning rate as a b+t α and do hyperparameter search over a, b and α.

In practice, this simply does not give results comparable to the constant and cut schemes.

4 One potential explanation for this could be that, in the context of neural network training, local minima found by constant and cut schemes are of much better quality than those found by polynomial decay schemes, while for convex problems, polynomial decay schemes are indeed optimal.

The primary contribution of this work is to show that this is simply not the case.

We concretely show how minimax optimal theoretical learning rates (i.e. polynomial decay schemes for wide classes of convex optimization problems) may be misleading (and sub-optimal for locally quadratic problems), and the story in practice is more nuanced.

There important issues at play with regards to this suboptimality.

First, even for the simple case of stochastic linear regression, with a fixed time horizon, the rate achieved by any polynomial decay scheme (i.e., any choice of a, b and α) is suboptimal compared to the statistical minimax rate (i.e., information theoretically best possible rate achievable by any algorithm) by a factor of condition number κ (see Section 3 for definitions), while there exist constant and cut schemes that are suboptimal only by a factor of log κ.

Second, this work shows that a factor of κ suboptimality is unavoidable if we wish to bound the error of each iterate of SGD.

In other words, we show that the convergence rate of lim sup of the error, as t → ∞, has to be necessarily suboptimal by a factor ofΩ(κ) compared to the statistical minimax rate, for any learning rate sequence (polynomial or not).

In fact, at leastΩ1/κ fraction of the iterates have this suboptimality.

With this result, things become quite clear -all the works in stochastic approximation try to bound the error of each iterate of SGD asymptotically (or lim sup of the error in other words).

Since this necessarily has to be suboptimal by a factor ofΩ(κ) compared to the statistical minimax rates, the suboptimality of polynomial decay rates is not an issue.

However, with a fixed time horizon, there exist learning rate schemes with much better convergence rates, while polynomial decay schemes fail to get better rates in this simpler setting (of known time horizon).Thirdly, the work shows that, for stochastic linear regression, if we consider lim inf (rather than lim sup) of the error, it is possible to design schemes that are suboptimal by only a factor of log κ compared to the minimax rates.

Variants of the constant and cut schemes achieve this guarantee.

In summary, the contributions of this paper are showing how widely used pratical learning rate schedules are, in fact, highly effective even in the convex case.

In particular, our theory and empirical results demonstrate this showing that:• For a fixed time horizon, constant and cut schemes are provably, significantly better than polynomial decay schemes.• There is a fundamental difference between fixed time horizon and infinite time horizon.• The above difference can be mitigated by considering lim inf of error instead of lim sup.• In addition to our theoretical contributions, we empirically verify the above claims for neural network training on cifar-10.Extending results on the performance of constant and cut schemes to more general convex optimization problems, beyond stochastic linear regression, is an important future direction.

However, the fact that the suboptimality of polynomial decay schemes even for the simple case of stochastic linear regression, has not been realized after decades of research on stochastic approximation is striking.

In summary, the results of this paper show that, even for stochastic linear regression, the popular in practice, constant and cut learning rate schedules are provably better than polynomial decay schemes popular in theory and that there is a need to rethink learning rate schemes and convergence guarantees for stochastic approximation.

Our results also suggest that current approaches to hyperparameter tuning of learning rate schedules might not be right headed and further suggest potential ways of improving them.

Paper organization: The paper is organized as follows.

We review related work in Section 2.

Section 3 describes the notation and problem setup.

Section 4 presents our results on the suboptimality of both polynomial decay schemes and constant and cut schemes.

Section 5 presents results on infinite horizon setting.

Section 6 presents experimental results and Section 7 concludes the paper.

We will split related work into two parts, one based on theory and the other based on practice.

Related efforts in theory: SGD and the problem of stochastic approximation was introduced in the seminal work of Robbins & Monro (1951) ; this work also elaborates on stepsize schemes that are satisfied by asymptotically convergent stochastic gradient methods: we refer to these schemes as "convergent" stepsize sequences.

The (asymptotic) statistical optimality of iterate averaged SGD with larger stepsize schemes of O(1/n α ) with α ∈ (0.5, 1) was proven in the seminal works of Ruppert (1988); Polyak & Juditsky (1992) .

The notions of convergent learning rate schemes in stochastic approximation literature has been studied in great detail BID18 BID14 BID4 BID16 .

Nearly all of the aforementioned works rely on function value sub-optimality to measure convergence and rely on the notion of asymptotic convergence (i.e. in the limit of the number of updates of SGD tending to infinity) to derive related "convergent stepsize schedules".

Along this line of thought, there are several efforts that prove (minimax) optimality of the aforementioned rates (in a worst case sense and not per problem sense) e.g., BID20 ; Raginsky & Rakhlin (2011); BID0 .An alternative viewpoint is to consider gradient norm as a means to measure the progress of an algorithm.

Along this line of thought are several works including the stochastic process viewpoint considered by Polyak & Juditsky (1992) and more recently, the work of Nesterov (2012) (working with deterministic (exact) gradients).

The work of Allen-Zhu (2018) considers questions relating to making the gradient norm small when working with stochastic gradients, and provides an improved rate.

We return to this criterion in Section 7.In terms of oracle models, note that both this paper, as well as other results BID15 Rakhlin et al., 2012; BID5 , work in an oracle model that assumes bounded variance of stochastic gradients or similar assumptions.

There is an alternative oracle model for analyzing SGD as followed in papers includingBach & Moulines (2013); Bach (2014); BID9 which is arguably more reflective of SGD's behavior in practice.

For more details, refer to BID9 .

It is an important direction to prove the results of this paper working in the alternative practically more applicable oracle model.

As highlighted in the introduction, practical efforts in stochastic optimization have diverged from the classical theory of stochastic approximation, with several deep learning libraries like pytorch 5 providing unconventional alternatives such as cosine/sawtooth/dev set decay schemes, or even exponentially decaying learning rate schemes.

In fact, a natural scheme used in training convolutional neural networks for vision is where the learning rate is cut by a constant factor after a certain number of epochs.

Such schemes are essentially discretized variants of exponentially decaying learning rate schedules.

We note that there are other learning rate schedules that have been recently proposed such as sgd with warm restarts BID19 , oscillating learning rates (Smith & Topin, 2017) etc., that are unconventional and have attracted a fair bit of attention.

Furthermore, exponential learning rates appear to be considered in more recent NLP papers (see for e.g., BID12 6 .

Notation: We represent scalars with normal font a, b, L etc., vectors with boldface lowercase characters a, b etc.

and matrices with boldface uppercase characters A, B etc.

We represent positive semidefinite (PSD) ordering between two matrices using .

The symbol represents that the direction of inequality holds for some universal constant.

Our theoretical results focus on the following additive noise stochastic linear regression problem.

We present the setup and associated notation in this section.

We wish to solve: min DISPLAYFORM0 for some positive definite matrix H and vector b. 7 We denote the smallest and largest eigenvalues DISPLAYFORM1 µ denotes the condition number of H. We have access to a stochastic gradient oracle which gives us ∇f (w) = ∇f (w) + e, where e is a random vector satisfying 8 E [e] = 0 and E ee = σ 2 H.Given an initial point w 0 and step size sequence η t , the SGD algorithm proceeds with the update DISPLAYFORM2 , where e t are independent for various t and satisfy the above mean and variance conditions.

Let w * def = arg min w∈R d f (w).

The suboptimality of a point w is given by f (w) − f (w * ).

It is well known that given t accesses to the stochastic gradient oracle above, any algorithm that uses these stochastic gradients and outputs w t has suboptimality that is lower bounded by DISPLAYFORM3 t .

More concretely (Van der Vaart, 2000), we have that DISPLAYFORM4 Moreover there exist schemes that achieve this rate of (1 + o(1)) DISPLAYFORM5 t e.g., constant step size SGD with averaging (Polyak & Juditsky, 1992) .

This rate of σ 2 d/t is called the statistical minimax rate.

In this section, we will show that polynomial decay schemes are suboptimal compared to the statistical minimax rate by at least a factor of κ while constant and cut schemes are suboptimal by at most a factor of log κ.

8 While this might seem very special, this is indeed a fairly natural scenario.

For instance, in stochastic linear regression with independent additive noise, i.e., yt = x t w * + t where t is a random variable independent of xt and E [ t] = 0 and E 2 t = σ 2 , the noise in the gradient has this property.

On the other hand, the results in this paper can also be generalized to the setting where E ee = V for some arbitrary matrix V. However, error covariance of σ 2 H significantly simplifies exposition.

Our first result shows that there exist problem instances where all polynomial decay schemes i.e., those of the form a b+t α , for any choice of a, b and α are suboptimal by at least a factor of Ω(κ) compared to the statistical minimax rate.

Theorem 1.

There exists a problem instance such that the initial function value f (w 0 ) ≤ σ 2 d, and for any fixed time T satisfying T ≥ κ 2 , for all a, b ≥ 0 and 0.5 ≤ α ≤ 1, and for the learning rate DISPLAYFORM0

Our next result shows that there exist constant and cut schemes that achieve statistical minimax rate upto a multiplicative factor of only log κ log 2 T .Theorem 2.

For any problem and fixed time horizon T > κ log(κ), there exists a constant and cut learning rate scheme that achieves DISPLAYFORM0 T .

We will now consider an exponential decay scheme (in contrast to polynomial ones from Section 4.1) which is a smoother version of constant and cut scheme.

We show that the same result above for constant and cut scheme can also be extended to the exponential decay scheme.

Theorem 3.

For any problem and fixed horizon T , there exist constants a and b such that learning rate scheme of DISPLAYFORM1 The above results show that constant and cut as well as exponential decay schemes, that depend on the time horizon, are much better than polynomial decay schemes.

Between these, exponential decay schemes are smoother versions of constant and cut schemes, and so one would hope that they might have better performance than constant and cut schemes -we do see a log T difference in our bounds.

One unsatisfying aspect of the above results is that the rate behaves as log T T , which is asymptotically worse than the statistical rate of 1 T .

It turns out that it is indeed possible to improve the rate to 1 T using a more sophisticated scheme.

The main idea is to use constant and polynomial schemes in the beginning and then switch to constant and cut (or exponential decay) scheme later.

To the best of our knowledge, these kind of schemes have never been considered in the stochastic optimization literature before.

Using this learning rate sequence successively for increasing time horizons would lead to oscillating learning rates.

We leave a complete analysis of oscillating learning rates (for moving time horizon) to future work.

Theorem 4.

Fix κ ≥ 2.

For any problem and fixed time horizon T / log T > 5κ, there exists a learning rate scheme that achieves DISPLAYFORM2 T .

In this section we show a fundamental limitation of the SGD algorithm.

First we will prove that the SGD algorithm, for any learning rate sequence, needs to query a point with suboptimality more than Ω(κ/ log κ) · σ 2 d/T for infinitely many time steps T .Theorem 5.

There exists a universal constant C > 0 such that for any SGD algorithm with η t ≤ 1/2κ for all t 9 , we have lim sup T →∞ DISPLAYFORM0 Next we will show that in some sense the "fraction" of query points that has value more than τ σ 2 /T is at least Ω(1/τ ) when τ is smaller than the threshold in Theorem 5.

Theorem 6.

There exists universal constants C 1 , C 2 > 0 such that for any τ ≤ κ CC1 log(κ+1) where C is the constant in Theorem 5, for any SGD algorithm and any number of iteration T > 0 there DISPLAYFORM1 Finally, we now show that there are constant and cut or exponentially decaying schemes that achieve the statistical minimax rate up to a factor of log κ log 2 T in the lim inf sense.9 Learning rate more than 2/κ will make the algorithm diverge.

Theorem 7.

There exists an absolute constant C and a constant and cut learning rate scheme that DISPLAYFORM2 Similar results can be obtained for the exponential decay scheme of Theorems 3 and 4 with moving time horizon.

However the resultant learning rates might have oscillatory behavior.

This might partly explain the benefits of oscillating learning rates observed in practice (Smith & Topin, 2017) .

We present experimental validation of our claims through controlled synthetic experiments on a twodimensional quadratic objective and on a real world non-convex optimization problem of training a residual network on the cifar-10 dataset, to illustrate the shortcomings of the traditional stochastic approximation perspective (and the advantages of non-convergent exponentially decaying and oscillating learning rate schemes) for a realistic problem encountered in practice.

Complete details of experimental setup are given in Appendix D.

We consider the problem of optimizing a two-dimensional quadratic objective, similar in spirit as what is considered in the theoretical results of this paper.

In particular, for a two-dimensional quadratic, we have two eigenvalues, one of magnitude κ and the other being 1.

We vary our condition number κ ∈ {50, 100, 200} and use a total of 200κ iterations for optimization.

The results expressed in this section are obtained by averaging over two random seeds.

The learning rate schemes we search over are: DISPLAYFORM0 For the schemes detailed above, there are two parameters that need to be searched over: (i) the starting learning rate η 0 and, (ii) the decay factor b. We perform a grid search over both these parameters and choose ones that yield the best possible final error at a given end time (i.e. 200κ).

We also make sure to extend the grid should a best performing grid search parameter fall at the edge of the grid so that all presented results lie in the interior of our final grid searched parameters.

We will present results for the following experiments: (i) behavior of the error of the final iterate of the SGD method with the three learning rate schemes (1),(2), and (3) as we vary the condition number, and (ii) how the exponentially decaying learning rate scheme (3) optimized for a shorter time horizon behaves for a longer horizon.

For the variation of the final iterate's excess risk when considered with respect to the condition number FIG1 , we note that polynomially decaying schemes have excess risk that scales linearly with condition number, corroborating Theorem 1.

In contrast, exponentially decaying learning rate scheme admits excess risk that nearly appears to be a constant and corroborates Theorem 3.

Finally, we note that the learning rate schedule that offers the best possible error in 50κ or 100κ steps does not offer the best error at 200κ steps TAB0 ).

We consider here the task of training a 44−layer deep residual network BID8 ) with pre-activation blocks BID7 ) (dubbed preresnet-44) for classifying images in the cifar-10 dataset.

The code for implementing the network employed in this paper can be found here 10 .

For all the experiments, we use the Nesterov's Accelerated gradient method (Nesterov, 1983) implemented in pytorch 11 with a momentum set to 0.9 and batchsize set to 128, total number of training epochs set to 100, 2 regularization set to 0.0005.Our experiments are based on grid searching for the best learning rate decay scheme on four parametric family of learning rate schemes described above 1,2,3; all gridsearches are performed on a separate validation set (obtained by setting aside one-tenth of the training dataset = 5000 images) and with models trained on the remaining 45000 images.

For presenting the final numbers in the plots/tables, we employ the best hyperparameters from the validation stage and train it on the entire 50, 000 images and average results run with 10 different random seeds.

The parameters for gridsearches and related details are presented in Appendix D. Furthermore, just as with the synthetic experiments, we always extend the grid so that the best performing grid search parameter lies in the interior of our grid search.

Comparison between different schemes: FIG2 and TAB2 present a comparison of the performance of the three schemes (1)-(3).

They clearly demonstrate that the best exponential scheme outperforms the best polynomial schemes.

Hyperparameter selection using truncated runs: FIG3 and Tables 3 and 4 present a comparison of the performance of three exponential decay schemes each of which has the best performance at 33, 66 and 100 epochs respectively.

The key point to note is that best performing hyperparameters at 33 and 66 epochs are not the best performing at 100 epochs (which is made stark from the perspective of the validation error).

This demonstrates that selecting hyper parameters using truncated runs, which has been proposed in some recent efforts such as hyperband BID17 14.42 ± 1.47% 9.8 ± 0.66% 7.58 ± 0.21% Table 4 : Comparing Test 0/1 error of various learning rate decay schemes for the classification task on cifar-10 using a 44−layer residual net with pre-activations.

The main contribution of this work shows that the picture of learning rate scheduling is far more nuanced than suggested by prior theoretical results, where we do not even need to move to nonconvex optimization to show other learning rate schemes can be far more effective than the standard polynomially decaying rates considered in theory.

Is quadratic loss minimization special?

One may ask if there is something particularly special about why the minimax rates are different for quadratic loss minimization as opposed to more general convex (and non-convex) optimization problems?

Ideally, we would hope that our theoretical insights (and improvements) can be formally established in more general cases.

Here, an alternative viewpoint is to consider gradient norm as a means to measure the progress of an algorithm.

The recent work of Allen-Zhu (2018) shows marked improvements for making the gradient norm small (when working with stochastic gradients) for both convex and non-convex, in comparison to prior results.

In particular, for the strongly convex case, Allen-Zhu (2018) provides results which have only a logarithmic dependency on κ, an exponential improvement over what is implied by standard analyses for the gradient norm BID15 Rakhlin et al., 2012; BID5 ; Allen-Zhu (2018) also provides improvements for the smooth and non-convex cases.

Thus, for the case of making the gradient norm small, there does not appear to be a notable discrepancy between the minimax rate of quadratic loss minimization in comparison to more general strongly convex (or smooth) convex optimization problems.

Interestingly, the algorithm of Allen-Zhu (2018) provides a recursive regularization procedure that obtains an SGD procedure, where the doubling regularization can be viewed as being analogous to an exponentially decaying learning rate schedule.

Further work in this direction may be promising in providing improved algorithms.

DISPLAYFORM0 the variance in the i th direction at time step t. Let the initialization be such that v DISPLAYFORM1 and v DISPLAYFORM2 .

This means that the variances for all directions with eigenvalue κ remain equal as t progresses and similarly for all directions with eigenvalue 1.

We have DISPLAYFORM3 We consider a recursion for v DISPLAYFORM4 t with eigenvalue λ i (1 or κ).

By the design of the algorithm, we know v DISPLAYFORM5 1−(1−ηλ) 2 be the solution to the stationary point equation DISPLAYFORM6 Intuitively if we keep using the same learning rate η, then v DISPLAYFORM7 t is going to converge to s(η, λ i ).

Also note that s(η, λ) ≈ σ 2 η/2 when ηλ 1.We first prove the following claim showing that eventually the variance in direction i is going to be at least s(η T , λ i ).

DISPLAYFORM8 Proof.

We can rewrite the recursion as DISPLAYFORM9 In this form, it is easy to see that the iteration is a contraction towards s(η t , λ i ).

Further, v DISPLAYFORM10 t − s(η t , λ i ) have the same sign.

In particular, let t 0 be the first time such that DISPLAYFORM11 0 (note that η t is monotone and so is s(η t , λ i )), it is easy to see that v DISPLAYFORM12 The claim then follows from a simple induction.

DISPLAYFORM13 Therefore we must have s(η T , κ) ≤ v(1) 0 = σ 2 /κ, and by Claim 1 we know v DISPLAYFORM14 we must have η T ≤ 1 8T .

Next we will show that when this happens, v DISPLAYFORM15 T must be large so the function value is still large.

We will consider two cases, in the first case, b ≥ T α .

Since DISPLAYFORM16 T , and we are done.

In the second case, b < T α .

Since DISPLAYFORM17 The sum of learning rates satisfy DISPLAYFORM18 Here the second inequality uses the fact that T α−1 i −α ≤ i −1 when i ≤ T .

Similarly, we also know DISPLAYFORM19 32T .

This concludes the second case and proves the theorem.

Proof of Theorem 2.

The learning rate scheme is as follows.

Divide the total time horizon into log(κ) equal sized phases.

In the th phase, the learning rate to be used is DISPLAYFORM0 .

Note that the learning rate in the first phase depends on strong convexity and that in the last phase depends on smoothness (since the last phase has = log κ).

Recall the variance in the k th coordinate can be upper bounded by DISPLAYFORM1 We will show that for every k, we have DISPLAYFORM2 which directly implies the theorem.

Now choose any k. Let * denote the number satisfying 2 * ·µ ≤ DISPLAYFORM3 Note that * depends on k but we suppressed the dependence for notational simplicity.

DISPLAYFORM4 This finishes the proof.

Proof of Theorem 3.

The learning rate scheme we consider is γ t = γ 0 · c t−1 with γ 0 = log T /(µT e ), T e = T / log κ and c = (1 − 1/T e ).

Further, just as in previous lemmas, we consider a specific eigen direction λ (k) and write out the progress made along this direction by iteration say,T ≤ T : DISPLAYFORM5 Substituting necessary values of the quantities, we have α = 2 log T · λ (k) /µ ≥ 2.

Now, the second term is upper bounded using the corresponding integral, which is the following: DISPLAYFORM6 Substituting this in the previous bound, we have: DISPLAYFORM7 Now, settingT = T e log(Cλ (k) /µ), and using 1 − a ≤ exp (−a), with C > 1 being some (large) universal constant, we have: DISPLAYFORM8 Now, in order to argue the progress of the algorithm fromT + 1 to T , we can literally view the algorithm as starting with the iterate obtained from running for the firstT steps (thus satisfying the excess risk guarantee in equation 4) and then adding in variance by running for the duration of time betweenT + 1 to T .

For this part, we basically upper bound this behavior by first assuming that there is no contraction of the bias and then consider the variance introduced by running the algorithm fromT + 1 to T .

This can be written as: DISPLAYFORM9 Now, to consider bounding the variance of the process with decreasing sequence of learning rates, we will instead work with a constant learning rate and understand its variance: DISPLAYFORM10 What this implies in particular is that the variance is a monotonic function of the learning rate and thus the overall variance can be bounded using the variance of the process run with a learning rate of γT .

DISPLAYFORM11 ≤ σ 2 log T log κ T Plugging this into equation 4 and summing over all directions, we have the desired result.

Proof of Theorem 4.

The learning rate scheme is as follows.

We first break T into three equal sized parts.

Let A = T /3 and B = 2T /3.

In the first T /3 steps, we use a constant learning rate of 1/L. In the second T /3 steps, we use a polynomial decay learning rate η A+t = 1 µ(κ+t/2) .

In the third T /3 steps, we break the steps into log 2 (κ) equal sized phases.

In the th phase, the learning rate to be used is 5 log 2 κ 2 ·µ·T. Note that the learning rate in the first phase depends on strong convexity and that in the last phase depends on smoothness (since the last phase has = log κ).Recall the variance in the k th coordinate can be upper bounded by DISPLAYFORM12 We will show that for every k, we have DISPLAYFORM13 which directly implies the theorem.

We will consider the first T /3 steps.

The guarantee that we will prove for these iterations is: for any DISPLAYFORM14 L .

This can be proved easily by induction.

Clearly this is true when t = 0.

Suppose it is true for t − 1, let's consider step t. By recursion of v DISPLAYFORM15 Here the second step uses induction hypothesis and the third step uses the fact that DISPLAYFORM16 we know at the end of the first phase, v DISPLAYFORM17 eigendirection (with eigenvalue 1).

As a useful tool, we will decompose the variance in the two directions corresponding to κ eigenvalue and 1 eigenvalue respectively as follows: DISPLAYFORM18 Proof of Theorem 5.

Fix τ = κ/C log(κ + 1) where C is a universal constant that we choose later.

We need to exhibit that the lim sup is larger than τ .

For simplicity we will also round κ up to the nearest integer.

Let T be a given number.

Our goal is to exhibit aT > T such that DISPLAYFORM19 Given the step size sequence η t , consider the sequence of numbers DISPLAYFORM20 Note that such a number always exists because all the step sizes are at most 2/κ.

We will also let ∆ i be T i − T i−1 .

Firstly, from (6) and FORMULA52 , we see that t η t = ∞. Otherwise, the bias will never decay to zero.

If DISPLAYFORM21 Ti−1+∆i for some i = 1, · · · , κ, we are done.

If not, we obtain the following relations: DISPLAYFORM22 Here the second inequality is based on (6).

We will use C 1 to denote exp(3).

Similarly, we have DISPLAYFORM23 Repeating this argument, we can show that DISPLAYFORM24 We will use i = 1 in particular, which specializes to DISPLAYFORM25 Using the above inequality, we can lower bound the sum of ∆ j as DISPLAYFORM26 Here the first inequality just uses the second term in Equation (6) or (7), the second inequality is because DISPLAYFORM27 j=T +1 η j ≤ 3/λ i and the last inequality is just based on the value of S 2 .In this case as we can see as long as C 2 is large enough, T + ∆ is also a point with E [f (w T +∆ )]

≥ DISPLAYFORM28 ≥ C 1 τ σ 2 /(T + ∆), so we can repeat the argument there.

Eventually we either stop because we hit case 1: S 2 ≤ C 2 τ /λ 2 i T or the case 2 S 2 > C 2 τ /λ 2 i T happened more than T /C 2 τ times.

In either case we know for anyT ∈ [T, (1 + 1/C 2 )T ] E [f (wT )]

≥ τ σ 2 /T ≥ τ σ 2 /T as the lemma claimed.

Theorem 6 is an immediate corollary of Theorem 5 and Lemma 8.Proof of Theorem 7.

This result follows by running the constant and cut scheme for a fixed time horizon T and then increasing the time horizon to κ · T .

The learning rate of the initial phase for the new T = κ · T is 1/µT = 1/µ · κT = 1/LT which is the final learning rate for time horizon T .

Theorem 2 will then directly imply the current theorem.

As mentioned in the main paper, we consider three condition numbers namely κ ∈ {50, 100, 200}. We run all experiments for a total of 200κ iterations.

The two eigenvalues of the Hessian are κ and 1 respectively, and noise level is σ 2 = 1 and we average our results with two random seeds.

All our grid search results are conducted on a 10 × 10 grid of learning rates × decay factor and whenever a best run lands at the edge of the grid, the grid is extended so that we have the best run in the interior of the gridsearch.

For the O(1/t) learning rate, we search for decay parameter over 10−points logarithmically spaced between {1/(100κ), 3000/κ}. The starting learning rate is searched over 10 points logarithmically spaced between {1/(20κ), 1000/κ}.For the O(1/ (t)) learning rate, the decay parameter is searched over 10 logarithmically spaced points between {100/κ, 200000.0/κ}. The starting learning rate is searched between {0.01, 2}.For the exponential learning rate schemes, the decay parameter is searched between {exp (−2/N ), exp (−10 6 /N )}.

The learning rate is searched between {1/5000, 1/10}.

As mentioned in the main paper, for all the experiments, we use the Nesterov's Accelerated gradient method (Nesterov, 1983) implemented in pytorch 12 with a momentum set to 0.9 and batchsize set to 128, total number of training epochs set to 100, 2 regularization set to 0.0005.With regards to learning rates, we consider 10−values geometrically spaced as {1, 0.6, · · · , 0.01}. To set the decay factor for any of the schemes such as 1,2, and 3, we use the following rule.

Suppose we have a desired learning rate that we wish to use towards the end of the optimization (say, something that is 100 times lower than the starting learning rate, which is a reasonable estimate of what is typically employed in practice), this can be used to obtain a decay factor for the corresponding decay scheme.

In our case, we found it advantageous to use an additively spaced grid for the learning rate γ t , i.e., one which is searched over a range {0.0001, 0.0002, · · · , 0.0009, 0.001, · · · , 0.009} at the 80 th epoch, and cap off the minimum possible learning rate to be used to be 0.0001 to ensure that there is progress made by the optimization routine.

For any of the experiments that yield the best performing gridsearch parameter that falls at the edge of the grid, we extend the grid to ensure that the finally chosen hyperparameter lies in the interior of the grid.

All our gridsearches are run such that we separate a tenth of the training dataset as a validation set and train on the remaining 9/10 th dataset.

Once the best grid search parameter is chosen, we train on the entire training dataset and 12 https://github.com/pytorch

<|TLDR|>

@highlight

This paper presents a rigorous study of why practically used learning rate schedules (for a given computational budget) offer significant advantages even though these schemes are not advocated by the classical theory of Stochastic Approximation.

@highlight

This paper presents a theoretical study of different learning rate schedules that resulted in statistical minimax lower bounds for both polynomial and constant-and-cut schemes.

@highlight

The paper studies the effect of learning-rate choices for stochastic optimization, focusing on least-mean-squares with decaying stepsizes