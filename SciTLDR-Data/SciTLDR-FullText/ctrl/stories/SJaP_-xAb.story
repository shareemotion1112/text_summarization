We propose a new output layer for deep neural networks that permits the use of logged contextual bandit feedback for training.

Such contextual bandit feedback can be available in huge quantities (e.g., logs of search engines, recommender systems) at little cost, opening up a path for training deep networks on orders of magnitude more data.

To this effect, we propose a Counterfactual Risk Minimization (CRM) approach for training deep networks using an equivariant empirical risk estimator with variance regularization, BanditNet, and show how the resulting objective can be decomposed in a way that allows Stochastic Gradient Descent (SGD) training.

We empirically demonstrate the effectiveness of the method by showing how deep networks -- ResNets in particular -- can be trained for object recognition without conventionally labeled images.

Log data can be recorded from online systems such as search engines, recommender systems, or online stores at little cost and in huge quantities.

For concreteness, consider the interaction logs of an ad-placement system for banner ads.

Such logs typically contain a record of the input to the system (e.g., features describing the user, banner ad, and page), the action that was taken by the system (e.g., a specific banner ad that was placed) and the feedback furnished by the user (e.g., clicks on the ad, or monetary payoff).

This feedback, however, provides only partial information -"contextual-bandit feedback" -limited to the actions taken by the system.

We do not get to see how the user would have responded, if the system had chosen a different action (e.g., other ads or banner types).

Thus, the feedback for all other actions the system could have taken is typically not known.

This makes learning from log data fundamentally different from traditional supervised learning, where "correct" predictions and a loss function provide feedback for all actions.

In this paper, we propose a new output layer for deep neural networks that allows training on logged contextual bandit feedback.

By circumventing the need for full-information feedback, our approach opens a new and intriguing pathway for acquiring knowledge at unprecedented scale, giving deep neural networks access to this abundant and ubiquitous type of data.

Similarly, it enables the application of deep learning even in domains where manually labeling full-information feedback is not viable.

In contrast to online learning with contextual bandit feedback (e.g., BID11 BID0 ), we perform batch learning from bandit feedback (BLBF) BID1 BID5 and the algorithm does not require the ability to make interactive interventions.

At the core of the new output layer for BLBF training of deep neural networks lies a counterfactual training objective that replaces the conventional cross-entropy objective.

Our approach -called BanditNet -follows the view of a deep neural network as a stochastic policy.

We propose a counterfactual risk minimization (CRM) objective that is based on an equivariant estimator of the true error that only requires propensity-logged contextual bandit feedback.

This makes our training objective fundamentally different from the conventional cross-entropy objective for supervised classification, which requires full-information feedback.

Equivariance in our context means that the learning result is invariant to additive translations of the loss, and it is more formally defined in Section 3.2.

To enable large-scale training, we show how this training objective can be decomposed to allow stochastic gradient descent (SGD) optimization.

In addition to the theoretical derivation of BanditNet, we present an empirical evaluation that verifies the applicability of the theoretical argument.

It demonstrates how a deep neural network architec-ture can be trained in the BLBF setting.

In particular, we derive a BanditNet version of ResNet (He et al., 2016) for visual object classification.

Despite using potentially much cheaper data, we find that Bandit-ResNet can achieve the same classification performance given sufficient amounts of contextual bandit feedback as ResNet trained with cross-entropy on conventionally (full-information) annotated images.

To easily enable experimentation on other applications, we share an implementation of BanditNet.

1 2 RELATED WORK Several recent works have studied weak supervision approaches for deep learning.

Weak supervision has been used to pre-train good image features (Joulin et al., 2016) and for information retrieval BID3 .

Closely related works have studied label corruption on CIFAR-10 recently BID12 .

However, all these approaches use weak supervision/corruption to construct noisy proxies for labels, and proceed with traditional supervised training (using crossentropy or mean-squared-error loss) with these proxies.

In contrast, we work in the BLBF setting, which is an orthogonal data-source, and modify the loss functions optimized by deep nets to directly implement risk minimization.

Virtually all previous methods that can learn from logged bandit feedback employ some form of risk minimization principle BID9 over a model class.

Most of the methods BID1 BID2 BID5 employ an inverse propensity scoring (IPS) estimator (Rosenbaum & Rubin, 1983) as empirical risk and use stochastic gradient descent (SGD) to optimize the estimate over large datasets.

Recently, the self-normalized estimator BID8 ) has been shown to be a more suitable estimator for BLBF BID7 .

The self-normalized estimator, however, is not amenable to stochastic optimization and scales poorly with dataset size.

In our work, we demonstrate how we can efficiently optimize a reformulation of the self-normalized estimator using SGD.Previous BLBF methods focus on simple model classes: log-linear and exponential models (Swaminathan & Joachims, 2015a) or tree-based reductions BID1 ).

In contrast, we demonstrate how current deep learning models can be trained effectively via batch learning from bandit feedback (BLBF), and compare these with existing approaches on a benchmark dataset (Krizhevsky & Hinton, 2009 ).Our work, together with independent concurrent work BID4 , demonstrates success with off-policy variants of the REINFORCE BID11 algorithm.

In particular, our algorithm employs a Lagrangian reformulation of the self-normalized estimator, and the objective and gradients of this reformulation are similar in spirit to the updates of the REINFORCE algorithm.

This connection sheds new light on the role of the baseline hyper-parameters in REINFORCE: rather than simply reduce the variance of policy gradients, our work proposes a constructive algorithm for selecting the baseline in the off-policy setting and it suggests that the baseline is instrumental in creating an equivariant counterfactual learning objective.

To formalize the problem of batch learning from bandit feedback for deep neural networks, consider the contextual bandit setting where a policy π takes as input x ∈ X and outputs an action y ∈ Y. In response, we observe the loss (or payoff) δ(x, y) of the selected action y, where δ(x, y) is an arbitrary (unknown) function that maps actions and contexts to a bounded real number.

For example, in display advertising, the context x could be a representation of the user and page, y denotes the displayed ad, and δ(x, y) could be the monetary payoff from placing the ad (zero if no click, or dollar amount if clicked).

The contexts are drawn i.i.d.

from a fixed but unknown distribution Pr(X).In this paper, a (deep) neural network is viewed as implementing a stochastic policy π.

We can think of such a network policy as a conditional distribution π w (Y | x) over actions y ∈ Y , where w are the parameters of the network.

The network makes a prediction by sampling an action y ∼ π w (Y | x), where deterministic π w (Y | x) are a special case.

As we will show as part of the empirical evaluation, many existing network architectures are compatible with this stochastic-policy view.

For example, any network f w (x, y) with a softmax output layer DISPLAYFORM0 can be re-purposed as a conditional distribution from which one can sample actions, instead of interpreting it as a conditional likelihood like in full-information supervised learning.

The goal of learning is to find a policy π w that minimizes the risk (analogously: maximizes the payoff) defined as DISPLAYFORM1 Any data collected from an interactive system depends on the policy π 0 that was running on the system at the time, determining which actions y and losses δ(x, y) are observed.

We call π 0 the logging policy, and for simplicity assume that it is stationary.

The logged data D are n tuples of observed context x i ∼ Pr(X), action y i ∼ π 0 (Y | x i ) taken by the logging policy, the probability of this action p i ≡ π 0 (y i | x i ), which we call the propensity, and the received loss δ i ≡ δ(x i , y i ): DISPLAYFORM2 We will now discuss how we can use this logged contextual bandit feedback to train a neural network policy π w (Y | x) that has low risk R(π w ).

While conditional maximum likelihood is a standard approach for training deep neural networks, it requires that the loss δ(x i , y) is known for all y ∈ Y. However, we only know δ(x i , y i ) for the particular y i chosen by the logging policy π 0 .

We therefore take a different approach following (Langford et al., 2008; BID6 ), where we directly minimize an empirical risk that can be estimated from the logged bandit data D. This approach is called counterfactual risk minimization (CRM) BID6 , since for any policy π w it addresses the counterfactual question of how well that policy would have performed, if it had been used instead of π 0 .While minimizing an empirical risk as an estimate of the true risk R(π w ) is a common principle in machine learning BID9 , getting a reliable estimate based on the training data D produced by π 0 is not straightforward.

The logged bandit data D is not only incomplete (i.e., we lack knowledge of δ(x i , y) for many y ∈ Y that π w would have chosen differently from π 0 ), but it is also biased (i.e., the actions preferred by π 0 are over-represented).

This is why existing work on training deep neural networks either requires full knowledge of the loss function, or requires the ability to interactively draw new samples y i ∼ π w (Y | x i ) for any new policy π w .

In our setting we can do neither -we have a fixed dataset D that is limited to samples from π 0 .To nevertheless get a useful estimate of the empirical risk, we explicitly address both the bias and the variance of the risk estimate.

To correct for sampling bias and handle missing data, we approach the risk estimation problem using importance sampling and thus remove the distribution mismatch between π 0 and π w (Langford et al., 2008; Owen, 2013; BID6 : DISPLAYFORM0 The latter expectation can be estimated on a sample D of n bandit-feedback examples using the following IPS estimator (Langford et al., 2008; Owen, 2013; BID6 : DISPLAYFORM1 This IPS estimator is unbiased and has bounded variance, if the logging policy has full support in the sense that ∀x, y : π 0 (y | x) ≥ > 0.

While at first glance it may seem natural to directly train the parameters w of a network to optimize this IPS estimate as an empirical risk, there are at least three obstacles to overcome.

First, we will argue in the following section that the naive IPS estimator's lack of equivariance makes it sub-optimal for use as an empirical risk for high-capacity models.

Second, we have to find an efficient algorithm for minimizing the empirical risk, especially making it accessible to stochastic gradient descent (SGD) optimization.

And, finally, we are faced with an unusual type of bias-variance trade-off since "distance" from the exploration policy impacts the variance of the empirical risk estimate for different w.

While Eq. FORMULA4 provides an unbiased empirical risk estimate, it exhibits the -possibly severe -problem of "propensity overfitting" when directly optimized within a learning algorithm BID7 .

It is a problem of overfitting to the choices y i of the logging policy, and it occurs on top of the normal overfitting to the δ i .

Propensity overfitting is linked to the lack of equivariance of the IPS estimator: while the minimizer of true risk R(π w ) does not change when translating the loss by a constant (i.e., ∀x, y : δ(x, y) + c) by linearity of expectation, DISPLAYFORM0 the minimizer of the IPS-estimated empirical riskR IPS (π w ) can change dramatically for finite training samples, and c + min DISPLAYFORM1 Intuitively, when c shifts losses to be positive numbers, policies π w that put as little probability mass as possible on the observed actions have low risk estimates.

If c shifts the losses to the negative range, the exact opposite is the case.

For either choice of c, the choice of the policy eventually selected by the learning algorithm can be dominated by where π 0 happens to sample data, not by which actions have low loss.

The following self-normalized IPS estimator (SNIPS) addresses the propensity overfitting problem BID7 and is equivariant: DISPLAYFORM2 In addition to being equivariant, this estimate can also have substantially lower variance than Eq. FORMULA4 , since it exploits the knowledge that the denominator DISPLAYFORM3 always has expectation 1: DISPLAYFORM4 The SNIPS estimator uses this knowledge as a multiplicative control variate BID7 .

While the SNIPS estimator has some bias, this bias asymptotically vanishes at a rate of O( 1 n ) (Hesterberg, 1995) .

Using the SNIPS estimator as our empirical risk implies that we need to solve the following optimization problem for training: DISPLAYFORM5 Thus, we now turn to designing efficient optimization methods for this training objective.

Unfortunately, the training objective in Eq. FORMULA0 does not permit stochastic gradient descent (SGD) optimization in the given form (see Appendix C), which presents an obstacle to efficient and effective training of the network.

To remedy this problem, we will now develop a reformulation that retains both the desirable properties of the SNIPS estimator, as well as the ability to reuse established SGD training algorithms.

Instead of optimizing a ratio as in Eq. (11), we will reformulate the problem into a series of constrained optimization problems.

Letŵ be a solution of Eq. (11), and at that solution let S * be the value of the control variate for πŵ as defined in Eq. (9).

For simplicity, assume that the minimizerŵ is unique.

If we knew S * , we could equivalently solve the following constrained optimization problem: DISPLAYFORM0 Of course, we do not actually know S * .

However, we can do a grid search in {S 1 , . . .

, S k } for S * and solve the above optimization problem for each value, giving us a set of solutions {ŵ 1 , . . .

,ŵ k }.

Note that S is just a one-dimensional quantity, and that the sensible range we need to search for S * concentrates around 1 as n increases (see Appendix B).

To find the overall (approximate)ŵ that optimizes the SNIPS estimate, we then simply take the minimum: DISPLAYFORM1 This still leaves the question of how to solve each equality constrained risk minimization problem using SGD.

Fortunately, we can perform an equivalent search for S * without constrained optimization.

To this effect, consider the Lagrangian of the constrained optimization problem in Eq. FORMULA0 with S j in the constraint instead of S * : DISPLAYFORM2 The variable λ is an unconstrained Lagrange multiplier.

To find the minimum of Eq. (12) for a particular S j , we need to minimize L(w, λ) w.r.t.

w and maximize w.r.t.

λ.

DISPLAYFORM3 However, we are not actually interested in the constrained solution of Eq. (12) for any specific S j .

We are merely interested in exploring a certain range S ∈ [S 1 , S k ] in our search for S * .

So, we can reverse the roles of λ and S, where we keep λ fixed and determine the corresponding S in hindsight.

In particular, for each {λ 1 , . . .

, λ k } we solvê DISPLAYFORM4 Note that the solutionŵ j does not depend on S j , so we can compute S j after we have found the minimumŵ j .

In particular, we can determine the S j that corresponds to the given λ j using the necessary optimality conditions, DISPLAYFORM5 by solving the second equality of Eq. (16).

In this way, the sequence of λ j produces solutionsŵ j corresponding to a sequence of {S 1 , . . .

, S k }.To identify the sensible range of S to explore, we can make use of the fact that Eq. (9) concentrates around its expectation of 1 for each π w as n increases.

Theorem 2 in Appendix B provides a characterization of how large the range needs to be.

Furthermore, we can steer the exploration of S via λ, since the resulting S changes monotonically with λ: (λ a < λ b ) and (ŵ a =ŵ b are not equivalent optima in Eq. (15)) ⇒ (S a < S b ).(17) A more formal statement and proof are given as Theorem 1 in Appendix A. In the simplest form one could therefore perform a grid search on λ, but more sophisticated search methods are possible too.

After this reformulation, the key computational problem is finding the solution of Eq. (15) for each λ j .

Note that in this unconstrained optimization problem, the Lagrange multiplier effectively translates the loss values in the conventional IPS estimate: We denote this λ-translated IPS estimate withR λ IPS (π w ).

Note that each such optimization problem is now in the form required for SGD, where we merely weight the derivative of the stochastic policy network π w (y | x) by a factor (δ i − λ j )/π 0 (y i | x i ).

This opens the door for re-purposing existing fast methods for training deep neural networks, and we demonstrate experimentally that SGD with momentum is able to optimize our objective scalably.

DISPLAYFORM6 Similar loss translations have previously been used in on-policy reinforcement learning BID11 , where they are motivated as minimizing the variance of the gradient estimate BID10 Greensmith et al., 2004) .

However, the situation is different in the off-policy setting we consider.

First, we cannot sample new roll-outs from the current policy under consideration, which means we cannot use the standard variance-optimal estimator used in REINFORCE.

Second, we tried using the (estimated) expected loss of the learned policy as the baseline as is commonly done in REINFORCE, but will see in the experiment section that this value for λ is far from optimal.

Finally, it is unclear whether gradient variance, as opposed to variance of the ERM objective, is really the key issue in batch learning from bandit feedback.

In this sense, our approach provides a rigorous justification and a constructive way of picking the value of λ in the off-policy settingnamely the value for which the corresponding S j minimizes Eq. (13).

In addition, one can further add variance regularization BID6 to improve the robustness of the risk estimate in Eq. (18) (see Appendix D for details).

The empirical evaluation is designed to address three key questions.

First, it verifies that deep models can indeed be trained effectively using our approach.

Second, we will compare how the same deep neural network architecture performs under different types of data and training objectives -in particular, conventional cross-entropy training using full-information data.

In order to be able to do this comparison, we focus on synthetic contextual bandit feedback data for training BanditNet that is sampled from the full-information labels.

Third, we explore the effectiveness and fidelity of the approximate SNIPS objective.

For the following BanditNet experiments, we adapted the ResNet20 architecture (He et al., 2016) by replacing the conventional cross-entropy objective with our counterfactual risk minimization objective.

We evaluate the performance of this Bandit-ResNet on the CIFAR-10 (Krizhevsky & Hinton, 2009) dataset, where we can compare training on full-information data with training on bandit feedback, and where there is a full-information test set for estimating prediction error.

To simulate logged bandit feedback, we perform the standard supervised to bandit conversion BID1 ).

We use a hand-coded logging policy that achieves about 49% error rate on the training data, which is substantially worse than what we hope to achieve after learning.

This emulates a real world scenario where one would bootstrap an operational system with a mediocre policy (e.g., derived from a small hand-labeled dataset) and then deploys it to log bandit feedback.

This logged bandit feedback data is then used to train the Bandit-ResNet.

We evaluate the trained model using error rate on the held out (full-information) test set.

We compare this model against the skyline of training a conventional ResNet using the full-information feedback from the 50,000 training examples.

Both the conventional full-information ResNet as well as the Bandit-ResNet use the same network architecture, the same hyperparameters, the same data augmentation scheme, and the same optimization method that were set in the CNTK implementation of ResNet20.

Since CIFAR10 does not come with a validation set for tuning the variance-regularization constant γ, we do not use variance regularization for Bandit-ResNet.

The Lagrange multiplier λ ∈ {0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05} is selected on the training set via Eq. (13).

The only parameter we adjusted for Bandit-ResNet is lowering the learning rate to 0.1 and slowing down the learning rate schedule.

The latter was done to avoid confounding the Bandit-ResNet results with potential effects from early stopping, and we report test performance after 1000 training epochs, which is well beyond the point of convergence in all runs.

Learning curve.

FIG0 shows the prediction error of the Bandit-ResNet as more and more bandit feedback is provided for training.

First, even though the logging policy that generated the bandit feedback has an error rate of 49%, the prediction error of the policy learned by the Bandit-ResNet is substantially better.

It is between 13% and 8.2%, depending on the amount of training data.

Second, the horizontal line is the performance of a conventional ResNet trained on the full-information training set.

It serves as a skyline of how good Bandit-ResNet could possibly get given that it is sampling bandit feedback from the same full-information training set.

The learning curve in FIG0 shows that Bandit-ResNet converges to the skyline performance given enough bandit feedback training data, providing strong evidence that our training objective and method can effectively extract the available information provided in the bandit feedback.

Effect of the choice of Lagrange multiplier.

The left-hand plot in FIG1 shows the test error of solutionsŵ j depending on the value of the Lagrange multiplier λ j used during training.

It shows that λ in the range 0.8 to 1.0 results in good prediction performance, but that performance degrades outside this area.

The SNIPS estimates in the right-hand plot of FIG1 roughly reflects this optimal range, given empirical support for both the SNIPS estimator and the use of Eq. (13).We also explored two other methods for selecting λ.

First, we used the straightforward IPS estimator as the objective (i.e., λ = 0), which leads to prediction performance worse than that of the logging policy (not shown).

Second, we tried using the (estimated) expected loss of the learned policy as the baseline as is commonly done in REINFORCE.

As FIG0 shows, it is between 0.130 and 0.083 for the best policies we found.

FIG1 (left) shows that these baseline values are well outside of the optimum range.

Also shown in the right-hand plot of FIG1 is the value of the control variate in the denominator of the SNIPS estimate.

As expected, it increases from below 1 to above 1 as λ is increased.

Note that large deviations of the control variate from 1 are a sign of propensity overfitting BID7 .

In particular, for all solutionsŵ j the estimated standard error of the control variate S j was less than 0.013, meaning that the normal 95% confidence interval for each S j is contained in [0.974, 1.026].

If we see aŵ j with control variate S j outside this range, we should be suspicious of propensity overfitting to the choices of the logging policy and discard this solution.

We proposed a new output layer for deep neural networks that enables the use of logged contextual bandit feedback for training.

This type of feedback is abundant and ubiquitous in the form of interaction logs from autonomous systems, opening up the possibility of training deep neural networks on unprecedented amounts of data.

In principle, this new output layer can replace the conventional cross-entropy layer for any network architecture.

We provide a rigorous derivation of the training objective, linking it to an equivariant counterfactual risk estimator that enables counterfactual risk minimization.

Most importantly, we show how the resulting training objective can be decomposed and reformulated to make it feasible for SGD training.

We find that the BanditNet approach applied to the ResNet architecture achieves predictive accuracy comparable to conventional full-information training for visual object recognition.

The paper opens up several directions for future work.

First, it enables many new applications where contextual bandit feedback is readily available.

Second, in settings where it is infeasible to log propensity-scored data, it would be interesting to combine BanditNet with propensity estimation techniques.

Third, there may be improvements to BanditNet, like smarter search techniques for S, more efficient counterfactual estimators beyond SNIPS, and the ability to handle continuous outputs.

DISPLAYFORM0 If the optimaŵ a andŵ b are not equivalent in the sense thatR DISPLAYFORM1 where g(w) corresponds to the value of the control variate S. Sinceŵ a andŵ b are not equivalent optima, we know that DISPLAYFORM2 Adding the two inequalities and solving implies that DISPLAYFORM3 B APPENDIX: CHARACTERIZING THE RANGE OF S TO EXPLORE.Theorem 2.

Let p ≤ π 0 (y | x) be a lower bound on the propensity for the logging policy, then constraining the solution of Eq. (11) to the w with control variate S ∈ [1 − , 1 + ] for a training set of size n will not exclude the minimizer of the true risk w * = arg min w∈W R(π w ) in the policy space W with probability at least DISPLAYFORM4 Proof.

For the optimal w * , let DISPLAYFORM5 be the control variate in the denominator of the SNIPS estimator.

S is a random variable that is a sum of bounded random variables between 0 and DISPLAYFORM6 We can bound the probability that the control variate S of the optimum w * lies outside of [1− , 1+ ] via Hoeffding's inequality: DISPLAYFORM7 The same argument applies to any individual policy π w , not just w * .

Note, however, that it can still be highly likely that at least one policy π w with w ∈ W shows a large deviation in the control variate for high-capacity W , which can lead to propensity overfitting when using the naive IPS estimator.

Suppose we have a dataset of n BLBF samples D = {(x 1 , y 1 , δ 1 , p 1 ) . . . (x n , y n , δ n , p n )} where each instance is an i.i.d.

sample from the data generating distribution.

In the sequel we will be considering two datasets of n + 1 samples, D = D ∪ {(x , y , δ , p )} and D = D ∪ {(x , y , δ , p )} where (x , y , δ , p ) = (x , y , δ , p ) and (x , y , δ , p ), (x , y , δ , p ) / ∈ D.For notational convenience, let DISPLAYFORM8 π0(yi|xi) , andġ i := ∇ w g i .

First consider the vanilla IPS risk estimate of Eq. (5).

DISPLAYFORM9 To maximize this estimate using stochastic optimization, we must construct an unbiased gradient estimate.

That is, we randomly select one sample from D and compute a gradient α((x i , y i , δ i , p i )) and we require that DISPLAYFORM10 Here the expectation is over our random choice of 1 out of n samples.

Observe that α((x i , y i , δ i , p i )) =ḟ i suffices (and indeed, this corresponds to vanilla SGD): DISPLAYFORM11 Other choices of α(·) can also produce unbiased gradient estimates, and this leads to the study of stochastic variance-reduced gradient optimization.

Now let us attempt to construct an unbiased gradient estimate for Eq. (8): DISPLAYFORM12 Suppose such a gradient estimate exists, β((x i , y i , δ i , p i )).

Then, DISPLAYFORM13 This identity is true for any sample of BLBF instances -in particular, for D and D : DISPLAYFORM14 1 n + 1 β((x i , y i , δ i , p i )) + β((x , y , δ , p )) n + 1 , DISPLAYFORM15 1 n + 1 β((x i , y i , δ i , p i )) + β((x , y , δ , p )) n + 1 .Subtracting these two equations, DISPLAYFORM16 = β((x , y , δ , p )) − β((x , y , δ , p )) n + 1 .The LHS clearly depends on {(x i , y i , δ i , p i )} n i=1 in general, while the RHS does not!

This contradiction indicates that no construction of β that only looks at a sub-sample of the data can yield an unbiased gradient estimate ofR SNIPS (π w ).

Unlike in conventional supervised learning, a counterfactual empirical risk estimator likeR IPS (π w ) can have vastly different variances Var(R IPS (π w )) for different π w in the hypothesis space (and R SNIPS (π w ) as well) BID6 .

Intuitively, the "closer" the particular π w is to the exploration policy π 0 , the larger the effective sample size (Owen, 2013) will be and the smaller the variance of the empirical risk estimate.

For the optimization problems we solve in Eq. FORMULA0 , this means that we should trust the λ-translated risk estimateR λj IPS (π w ) more for some w than for others, as we useR λj IPS (π w ) only as a proxy for finding the policy that minimizes its expected value (i.e., the true loss).

To this effect, generalization error bounds that account for this variance difference BID6 ) motivate a new type of overfitting control.

This leads to the following training objective BID6 , which can be thought of as a more reliable version of Eq. FORMULA0 : DISPLAYFORM0 Here, Var(R λj IPS (π w )) is the estimated variance ofR λj IPS (π w ) on the training data, and γ is a regularization constant to be selected via cross validation.

The intuition behind this objective is that we optimize the upper confidence interval, which depends on the variance of the risk estimate for each π w .

While this objective again does not permit SGD optimization in its given form, it has been shown that a Taylor-majorization can be used to successively upper bound the objective in Eq. FORMULA2 , and that typically a small number of iterations suffices to converge to a local optimum BID6 .

Each such Taylor-majorization is again of a form DISPLAYFORM1 for easily computable constants A and B BID6 , which allows for SGD optimization.

<|TLDR|>

@highlight

The paper proposes a new output layer for deep networks that permits the use of logged contextual bandit feedback for training. 