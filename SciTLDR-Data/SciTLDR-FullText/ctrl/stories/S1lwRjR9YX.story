While momentum-based methods, in conjunction with the stochastic gradient descent, are widely used when training machine learning models, there is little theoretical understanding on the generalization error of such methods.

In practice, the momentum parameter is often chosen in a heuristic fashion with little theoretical guidance.

In this work, we use the framework of algorithmic stability to provide an upper-bound on the generalization error for the class of strongly convex loss functions, under mild technical assumptions.

Our bound decays to zero inversely with the size of the training set, and increases as the momentum parameter is increased.

We also develop an upper-bound on the expected true risk,  in terms of the number of training steps, the size of the training set, and the momentum parameter.

A fundamental issue for any machine learning algorithm is its ability to generalize from the training dataset to the test data.

A classical framework used to study the generalization error in machine learning is PAC learning BID0 BID1 .

However, the associated bounds using this approach can be conservative.

Recently, the notion of uniform stability, introduced in the seminal work of Bousquet and Elisseeff BID2 , is leveraged to analyze the generalization error of the stochastic gradient method (SGM) BID3 .

The result in BID3 ) is a substantial step forward, since SGM is widely used in many practical systems.

This method is scalable, robust, and widely adopted in a broad range of problems.

To accelerate the convergence of SGM, a momentum term is often added in the iterative update of the stochastic gradient BID4 .

This approach has a long history, with proven benefits in various settings.

The heavy-ball momentum method was first introduced by Polyak BID5 , where a weighted version of the previous update is added to the current gradient update.

Polyak motivated his method by its resemblance to a heavy ball moving in a potential well defined by the objective function.

Momentum methods have been used to accelerate the backpropagation algorithm when training neural networks BID6 .

Intuitively, adding momentum accelerates convergence by circumventing sharp curvatures and long ravines of the sublevel sets of the objective function BID7 .

For example, Ochs et al. has presented an illustrative example to show that the momentum can potentially avoid local minima BID8 .

Nesterov has proposed an accelerated gradient method, which converges as O(1/k 2 ) where k is the number of iterations (Nesterov, 1983) .

However, the Netstrov momentum does not seem to improve the rate of convergence for stochastic gradient (Goodfellow et al., 2016, Section 8.3.3) .

In this work, we focus on the heavy-ball momentum.

Although momentum methods are well known to improve the convergence in SGM, their effect on the generalization error is not well understood.

In this work, we first build upon the framework in BID3 to obtain a bound on the generalization error of SGM with momentum (SGMM) for the case of strongly convex loss functions.

Our bound is independent of the number of training iterations and decreases inversely with the size of the training set.

Secondly, we develop an upper-bound on the optimization error, which quantifies the gap between the empirical risk of SGMM and the global optimum.

Our bound can be made arbitrarily small by choosing sufficiently many iterations and a sufficiently small learning rate.

Finally, we establish an upper-bound on the expected true risk of SGMM as a function of various problem parameters.

We note that the class of strongly convex loss functions appears in several important machine learning problems, including linear and logistic regression with a weight decay regularization term.

Other related works: convergence analysis of first order methods with momentum is studied in (Nesterov, 1983; BID11 BID12 BID13 BID14 BID15 BID16 BID17 .

Most of these works consider the deterministic setting for gradient update.

Only a few works have analyzed the stochastic setting BID15 BID16 BID17 .

Our convergence analysis results are not directly comparable with these works due to their different assumptions regarding the properties of loss functions.

In particular, we analyze the convergence of SGMM for a smooth and strongly convex loss function as in BID3 , which is new.

First-order methods with noisy gradient are studied in BID18 and references therein.

In BID18 , the authors show that there exists linear regression problems for which SGM outperforms SGMM in terms of convergence.

Our main focus in this work is on the generalization, and hence true risk, of SGMM.

We are aware of only one similar work in this regard, which provides stability bounds for quadratic loss functions BID19 .

In this paper, we obtain stability bounds for the general case of strongly convex loss functions.

In addition, unlike BID19 , our results show that machine learning models can be trained for multiple epochs of SGMM with bounded generalization errors.

We use E[??] to denote the expectation and ?? to represent the Euclidean norm of a vector.

We use lower-case bold font to denote vectors.

We use sans-serif font to denote random quantities.

Sets and scalars are represented by calligraphic and standard fonts, respectively.

We consider a general supervised learning problem, where S = {z 1 , ?? ?? ?? , z n } denotes the set of samples of size n drawn i.i.d.

from some space Z with an unknown distribution D. We assume a learning model described by parameter vector w. Let f (w; z) denote the loss of the model described by parameter w on example z ??? Z. Our ultimate goal is to minimize the true or population risk: DISPLAYFORM0 Since the distribution D is unknown, we replace the objective by the empirical risk, i.e., DISPLAYFORM1 We assume w = A(S) for a potentially randomized algorithm A(??).

In order to find an upper-bound on the true risk, we consider the generalization error, which is the expected difference of empirical and true risk: DISPLAYFORM2 Finally, to upper bound g , we consider uniform stability:Definition 1 Let S and S denote two data sets from space Z n such that S and S differ in at most one example.

Algorithm A is s -uniformly stable if for all data sets S, S , we have DISPLAYFORM3 It is shown in BID3 ) that uniform stability implies generalization in expectation:Theorem 1 BID3 If A is an s -uniformly stable algorithm, then the generalization error of A is upper-bounded by s .Theorem 1 shows that it is enough to control the uniform stability of an algorithm to upper bound the generalization error.

In our analysis, we will assume that the loss function satisfies the following properties.

DISPLAYFORM0 We assume that the parameter space ??? is a convex set.

Furthermore, for the loss function to be L-Lipschitz and and strongly convex, we further assume that ??? is compact.

Since ??? is compact, the SGMM update requires projection.

The update rule for projected SGMM is given by: DISPLAYFORM0 where P denotes the Euclidean projection onto ???, ?? > 0 is the learning rate 1 , ?? > 0 is the momentum parameter, i t is a randomly selected index, and f (w t ; z it ) is the loss evaluated on sample z it .

In SGMM, we run the update (5) iteratively for T steps and let w T denote the final output.

Note that there are two typical approaches to select i t .

The first approach is to select i t ??? {1, ?? ?? ?? , n} uniformly at random at each iteration.

The second approach is to permutate {1, ?? ?? ?? , n} randomly once and then select the examples repeatedly in a cyclic manner.

Our results are valid for both approaches.

The key quantity of interest in this paper is the generalization error for SGMM given by: DISPLAYFORM1 since the randomness in A arises from the choice of i 0 , ?? ?? ?? , i T ???1 .

In the following, we assume that the loss function f (??; z) is ??-smooth, L-Lipschitz, and ??-strongly convex for all z.

Theorem 2 (Stability bound) Suppose that the SGMM update (5) is executed for T steps with constant learning rate ?? and momentum ??. Provided that DISPLAYFORM0 The result in Theorem 2 implies that the stability bound decreases inversely with the size of the training set.

It increases as the momentum parameter ?? increases.

These properties are also verified in our experimental evaluation.

Theorem 3 (Convergence bound) Suppose that the SGMM update (5) is executed for T steps with constant learning rate ?? and momentum ??.

Then we have DISPLAYFORM1 where?? T denotes the average of T steps of the algorithm, i.e.,?? T = DISPLAYFORM2 Theorem 3 bounds the optimization error, i.e., the expected difference between the empirical risk achieved by SGMM and the global minimum.

Upon setting ?? = 0 and ?? = 0 in FORMULA8 , we can recover the classical bound on optimization error for SGM BID20 , (Hardt et al., 2016, Theorem 5.2) .

The first two terms in (7) vanish as T increases.

The terms with negative sign improve the convergence due to the strongly convexity.

The last term depends on the learning rate, ??, the momentum parameter ??, and the Lipschitz constant L. This term can be controlled by selecting ?? sufficiently small.

Proposition 1 (Upper-bound on true risk) Suppose that the SGMM update (5) is executed for T steps with constant learning rate ?? and momentum ??, satisfying the conditions in Theorem 2 and DISPLAYFORM3 T , we have: DISPLAYFORM4 where DISPLAYFORM5 and?? T as well as the constants W 0 , ?? ?? ?? , W 3 are defined in Theorem 3.Proposition 1 provides a bound on the expected true risk of SGMM in terms of the global minimum of the empirical risk.

The bound in FORMULA11 is obtained by combining Theorem 2 and Theorem 3 and minimizing the expression over ??.

The choice of ?? simplifies considerably when ?? is sufficiently small, as stated in Proposition 1.

Due to the page constraint, the proof of this result is provided in the supplementary material.

Note that the first two terms in (8) vanish as T increases.

The last term in (8) vanishes as the number of samples n increases.

Following BID3 , we track the divergence of two different iterative sequences of update rules with the same starting point.

However, our analysis is more involved as the presence of momentum term requires a more careful bound on the iterative expressions.

To keep the notation uncluttered, we first consider SGMM without projection and defer the discussion of projection to the end of this proof.

Let S = {z 1 , ?? ?? ?? , z n } and S = {z 1 , ?? ?? ?? , z n } be two samples of size n that differ in at most one example.

Let w T and w T denote the outputs of SGMM on S and S , respectively.

We consider the updates w t+1 = G t (w t ) + ??(w t ??? w t???1 ) and w t+1 = G t (w t ) + ??(w t ??? w t???1 ) with G t (w t ) = w t ??? ????? w f (w t ; z it ) and G t (w t ) = w t ??? ????? w f (w t ; z it ), respectively, for t = 1, ?? ?? ?? , T .

We denote ?? t ??? = w t ??? w t .

Suppose w 0 = w 0 , i.e., ?? 0 = 0.

We first establish an upper-bound on E A [?? t+1 ] in terms of E A [?? t ] and E A [?? t???1 ] in the following lemma, whose proof is provided in the supplementary document.

DISPLAYFORM0 Using the result of Lemma 1, in the following, we develop an upper bound on E A [?? T ].

Let us consider the recursion DISPLAYFORM1 with?? 0 = ?? 0 = 0.

Upon inspecting (10) it is clear that DISPLAYFORM2 as we simply drop the remainder of positive terms.

Substituting (11) into (10), we have DISPLAYFORM3 where the second inequality holds due to ?? ??? ?????? ??+?? ??? 1 2 .Noting that DISPLAYFORM4 where the second expression holds since 0 ??? ?? < ?????? 3(??+??) is assumed.

Applying the L-Lipschitz property on f (??, z), it follows that DISPLAYFORM5 Since this bound holds for all S, S and z, we obtain an upper-bound on the uniform stability and the proof is complete.

Our stability bound in Theorem 2 holds for the projected SGMM update (5) because Euclidean projection does not increase the distance between projected points (the argument is essentially analogous to BID3 , Lemma 4.6)).

In particular, note that Lemma 1 holds for the projected SGMM.

Again, we first consider SGMM without projection and discuss the extension to projection at the end of this proof.

Our proof is inspired by the convergence analysis in BID15 BID13 for a convex loss function with bounded variance and time-decaying learning rate.

Different from these works, we analyze the convergence of SGMM for a smooth and strongly convex loss function with constant learning rate.

To facilitate the convergence analysis, we define: DISPLAYFORM0 with p 0 = 0.

Substituting into the SGMM update, the parameter recursion is given by DISPLAYFORM1 It follows that DISPLAYFORM2 Substituting p t (15) into (17), the recursion (16) can be written as DISPLAYFORM3 (18) Upon taking the expectation with respect to i t in FORMULA0 we have DISPLAYFORM4 where we use the fact that ??? w f (w t ; z it ) ??? L, due to L-Lipschitz, and that E it [??? w f (w t ; z it )] = ??? w R S (w t ).

Furthermore, since R S (??) is a ??-strongly convex function, for all w t and w t???1 , we have DISPLAYFORM5 Substituting FORMULA1 in FORMULA0 , we have DISPLAYFORM6 Taking expectation over i 0 , ?? ?? ?? , i t for a given S, summing (21) for t = 0, ?? ?? ?? , T , and rearranging terms, we have DISPLAYFORM7 Since ?? is a convex function, for all w T and w, we have DISPLAYFORM8 Furthermore, due to convexity of R S (??), we have DISPLAYFORM9 Taking expectation over S, applying inequalities (23) and FORMULA1 into FORMULA1 , and substituting w = w * S , we obtain (7) and the proof is complete.

Our convergence bound in Theorem 3 can be extended to projected SGMM (5).

Let use denote y t+1 ??? = w t + ??(w t ??? w t???1 ) ??? ????? w f (w t ; z it ).

Then, for any feasible w ??? ???, (17) holds for y t+1 , i.e., DISPLAYFORM10 Note that the LHS of (25) can be written as DISPLAYFORM11 We note that ??w t + (1 ??? ??)w ??? ??? for any w ??? ??? and w t ??? ??? since ??? is convex.

Now in projected SGMM, we have DISPLAYFORM12 since projection a point onto ??? moves it closer to any point in ???. This shows inequality (19) holds, and the convergence results do not change.

In this section, we validate the insights obtained in our theoretical results in experimental evaluation.

Our main goal is to study how adding momentum affects the convergence and generalization of SGM.

We study the performance of SGMM when applied to the notMINIST dataset.

Please note that similar results are provided for the MNIST dataset in the supplementary document.

We train a logistic regression model with the weight decay regularization using SGMM for binary classification on the two-class notMNIST dataset that contains the images from letter classes "C" and "J", which leads to a smooth and strongly convex loss function.

We set the learning rate ?? = 0.01.

The weight decay coefficient and the minibatch size are set to 0.001 and 10, respectively.

We use 100 SGMM realizations to evaluate the average performance.

We compare the training and generalization performance of SGM without momentum with that of SGMM under ?? = 0.5 and ?? = 0.9, which are common momentum values used in practice (Goodfellow et al., 2016, Section 8.3 .2).The generalization error (with respect to cross entropy) and training error versus the number of training samples, n, under SGMM with fixed T = 1000 iterations are shown in FIG0 , respectively, for ?? = 0, 0.5, 0.9.

In FIG1 , we plot the generalization error (with respect to classification accuracy) and the training accuracy as a function of the number of training samples for the same dataset.

First, we observe that the generalization error (with respect to both cross entropy and classification accuracy) decreases as n increases for all values of ??, which is suggested by our stability upper-bound in Theorem 2.

In addition, for sufficiently large n, we observe that the generalization error increases with ??, consistent with Theorem 2.

On the other hand, the training error increases as n increases, which is expected.

We can observe that adding momentum reduces training error as it improves the convergence rate.

The training accuracy also improves by adding momentum as illustrated in FIG1 .

In order to study the optimization error of SGMM, we show the training error and test error versus the number of epochs, under SGMM trained with n = 500 samples in Figures 3a and 3b , respectively.

We plot the classification accuracy for training and test datasets in Figures 4a and 4b , respectively.

We observe that the training error decreases as the number of epochs increases for all values of ??, which is consistent with the convergence analysis in Theorem 3.

Furthermore, as expected, we see that adding momentum improves the training error and accuracy.

However, as the number of epochs increases, we note that the benefit of momentum on the test error and accuracy becomes negligible.

This happens because adding momentum also results in a higher generalization error thus penalizing the gain in training error.

We study the generalization error and convergence of SGMM for the class of strongly convex loss functions, under mild technical conditions.

We establish an upper-bound on the generalization error, which decreases with the size of the training set, and increases as the momentum parameter is increased.

Secondly, we analyze the convergence of SGMM during training, by establishing an upper-bound on the gap between the empirical risk of SGMM and the global minimum.

Our proposed bound reduces to a classical bound on the optimization error of SGM BID20 for convex functions, when the momentum parameter is set to zero.

Finally, we establish an upper-bound on the expected difference between the true risk of SGMM and the global minimum of the empirical risk, and illustrate how it scales with the number of training steps and the size of the training set.

Although our results are established for the case when the learning rate is constant, they can be easily extended to the case when the learning rate decreases with the number of iterations.

We also present experimental evaluations on the notMNIST dataset and show that the numerical plots are consistent with our theoretical bounds on the generalization error and the convergence gap.

<|TLDR|>

@highlight

Stochastic gradient method with momentum generalizes.