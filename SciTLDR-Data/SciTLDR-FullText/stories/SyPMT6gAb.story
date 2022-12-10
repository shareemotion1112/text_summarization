Off-policy learning, the task of evaluating and improving policies using historic data collected from a logging policy, is important because on-policy evaluation is usually expensive and has adverse impacts.

One of the major challenge of off-policy learning is to derive counterfactual estimators that also has low variance and thus low generalization error.

In this work, inspired by learning bounds for importance sampling problems, we present a new counterfactual learning principle for off-policy learning with bandit feedbacks.

Our method regularizes the generalization error by minimizing the distribution divergence between the logging policy and the new policy, and removes the need for iterating through all training samples to compute sample variance regularization in prior work.

With neural network policies, our end-to-end training algorithms using variational divergence minimization showed significant improvement over conventional baseline algorithms and is also consistent with our theoretical results.

Off-policy learning refers to evaluating and improving a deterministic policy using historic data collected from a stationary policy, which is important because in real-world scenarios on-policy evaluation is oftentimes expensive and has adverse impacts.

For instance, evaluating a new treatment option, a clinical policy, by administering it to patients requires rigorous human clinical trials, in which patients are exposed to risks of serious side effects.

As another example, an online advertising A/B testing can incur high cost for advertisers and bring them few gains.

Therefore, we need to utilize historic data to perform off-policy evaluation and learning that can enable safe exploration of the hypothesis space of policies before deploying them.

There has been extensive studies on off-policy learning in the context of reinforcement learning and contextual bandits, including various methods such as Q learning BID33 ), doubly robust estimator BID8 ), self-normalized (Swaminathan & Joachims (2015b) ), etc.

A recently emerging direction of off-policy learning involves the use of logged interaction data with bandit feedback.

However, in this setting, we can only observe limited feedback, often in the form of a scalar reward or loss, for every action; a larger amount of information about other possibilities is never revealed, such as what reward we could have obtained had we taken another action, the best action we should have take, and the relationship between the change in policy and the change in reward.

For example, after an item is suggested to a user by an online recommendation system, although we can observe the user's subsequent interactions with this particular item, we cannot anticipate the user's reaction to other items that could have been the better options.

Using historic data to perform off-policy learning in bandit feedback case faces a common challenge in counterfactual inference:

How do we handle the distribution mismatch between the logging policy and a new policy and the induced generalization error?

To answer this question, BID34 derived the new counterfactual risk minimization framework, that added the sample variance as a regularization term into conventional empirical risk minimization objective.

However, the parametrization of policies in their work as linear stochastic models has limited representation power, and the computation of sample variance regularization requires iterating through all training samples.

Although a first-order approximation technique was proposed in the paper, deriving accurate and efficient end-to-end training algorithms under this framework still remains a challenging task.

Our contribution in this paper is three-fold:1.

By drawing a connection to the generalization error bound of importance sampling BID6 ), we propose a new learning principle for off-policy learning with bandit feedback.

We explicitly regularize the generalization error of the new policy by minimizing the distribution divergence between it and the logging policy.

The proposed learning objective automatically trade off between emipircal risk and sample variance.

2.

To enable end-to-end training, we propose to parametrize the policy as a neural network, and solves the divergence minimization problem using recent work on variational divergence minimization BID26 ) and Gumbel soft-max BID18 ) sampling.

3.

Our experiment evaluation on benchmark datasets shows significant improvement in performance over conventional baselines, and case studies also corroborates the soundness of our theoretical proofs.

We first review the framework of off-policy learning with logged bandit feedback introduced in BID34 .

A policy maps an input x ∈ X to a structured (discrete) output y ∈ Y. For example, the input x can be profiles of users, and we recommend movies of relevance to the users as the output y; or in the reinforcement learning setting, the input is the trajectory of the agent, and the output is the action the agent should take in the next time point.

We use a family of stochastic policies, where each policy defines a posterior distribution over the output space given the input x, parametrized by some θ, i.e., h θ (Y|x).

Note that here a distribution which has all its probability density mass on one action corresponds to a deterministic policy.

With the distribution h(Y|x), we take actions by sampling from it, and each action y has a probability of h(y|x) being selected.

In the discussion later, we will use h and h(y|x) interchangeably when there will not create any confusion.

In online systems, we observe feedbacks δ(x, y; y * ) for the action y sampled from h(Y|x) by comparing it to some underlying 'best'

y * that was not revealed to the system.

For example, in recommendation system, we can use a scalar loss function δ(x, y; y DISPLAYFORM0 with smaller values indicating higher satisfaction with recommended items.

The expected risk of a policy h(Y|x) is defined as DISPLAYFORM1 , and the goal of off-policy learning is to find a policy with minimum expected risk on test data.

In the off-line logged learning setting, we only have data collected from a logging policy h 0 (Y|x), and we aim to find an improved policy h(Y|x) that has lower expected risks R(h) < R(h 0 ).

Specifically, the data we will use will be DISPLAYFORM2 , where δ i and p i are the observed loss feedback and the logging probability (also called propensity score), and N is the number of training samples.

Two main challenges are associated with this task: 1) If the distribution of a logging policy is skewed towards a specific region of the whole space, and doesn't have support everywhere, feedbacks of certain actions cannot be obtained and improvement for these actions is not possible as a result.

2) since we cannot compute the expectation exactly, we need to resort to empirical estimation using finite samples, which creates generalization error and needs additional regularization.

A vanilla approach to solve the problem is propensity scoring approach using importance sampling BID28 ), by accounting for the distribution mismatch between h and h 0 .

Specifically, we can rewrite the expected risk w.r.t h as the risk w.r.t h 0 using an importance reweighting: DISPLAYFORM3 With the collected historic dataset D, we can estimate the empirical riskR D (h), short asR(h) DISPLAYFORM4 2.2 COUNTERFACTUAL RISK MINIMIZATION BID34 pointed out several flaws with the vanilla approach, namely, not being invariant to loss scaling, large and potentially unbounded variance.

To regularize the variance, the authors proposed a regularization term for sample variance derived from empirical Bernstein bounds.

The modified objective function to minimize is now: DISPLAYFORM5 , where DISPLAYFORM6 u i is the average of {u i } obtained from training data, and V ar(ū) is the sample variance of {u i }.As the variance term is dependent on the whole dataset, stochastic training is difficult, the authors approximated the regularization term via first-order Taylor expansion and obtained a stochastic optimization algorithm.

Despite its simplicity, such first-order approximation neglects the non-linear terms from second-order and above, and introduces approximation errors while trying to reduce the sample variance.

Instead of estimating variance empirically from the samples, which prohibits direct stochastic training, the fact that we have a parametrized version of the policy h(Y|x) motivates us to think: can we derive a variance bound directly from the parametrized distribution?We first note that the empirical risk termR(h) is the average loss reweigthed by importance sampling function DISPLAYFORM0 h0(y|x) , and a general learning bound exist for importance sampling weights.

Let z be a random variable and the importance sampling weight w(z) = p(z) p0(z) , where p and p 0 are two probability density functions, the following identity holds Lemma 1.

BID6 ) For a random variable z, let p(z) and p 0 (z) be two distribuion density function defined for z, and l(z) be a loss function of z bounded in [0, 1] .

Let w = w(z) = p(z)/p 0 (z) be the importance sampling weight, the following identity holds: DISPLAYFORM1 , where D 2 is the Rényi divergence D α BID27 ) with α = 2, i.e. squared Chi-2 divergence.

Based on this lemma, we can derive an upper bound for the second moment of the weighted loss Theorem 1.

Let X be a random variable distributed according to distribution P with density p(x), Y be a random variable, and δ(x, y) be a loss function over DISPLAYFORM2 For two sampling distributions of y, h(y|x) and h 0 (y|x), define their conditional divergence as d 2 (h(y|x)||h 0 (y|x); P(x)), we have DISPLAYFORM3 The bound is similar to Eq. (4) with the difference that we are now working with a joint distribution over x, y. Detailed proofs can be found in Appendix 1.From the above theorem, we are able to derive a generalization bound between the expected risk R(h) and empirical riskR(h) using the distribution divergence function as Theorem 2.

Let R h be the expected risk of the new policy on loss function δ, andR h be the emprical risk.

We additionally assume the divergence is bounded by DISPLAYFORM4 Then with probability at least 1 − η, DISPLAYFORM5 The proof of this theorem is an application of Bernstein inequality and the second moment bound, and detailed proof is in Appendix 7.

This result highlights the bias-variance trade-offs as seen in empirical risk minimization (ERM) problems, whereR h approximates the emipircal risk/ bias, and the third term characterize the variance of the solution with distribution divergence (Recall V ar(w) = d 2 (h||h 0 ) − 1).

It thus motivates us that in bandit learning setting, instead of directly optimizing the reweighed loss and suffer huge variance in test setting, we can try to minimize the variance regularized objectives as DISPLAYFORM6 λ = 2L 2 log 1/η is a model hyper-parameter controlling the trade-off between empirical risk and model variance, but we are still faced with the challenge of setting λ empirically and the difficuty in optimizing the objective (See Appendix for a comparison).

Thus, in light of the recent success of distributionally robust learning, we explore an alternative formulation of the above regularized ERM in the next subsection.

Instead of solving a 'loss + regularizer' objective function, we here study a closely related constrained optimizationf formulation, whose intuition comes from the method of Langaragian mutliplier for constrained optimization.

The new formulation is: DISPLAYFORM0 , where ρ is a pre-determined constant as the regularization hyper-parameter.

By applying Theorem , for a policy h, we have DISPLAYFORM1 This inequality shows that the robust objectiveR d(h||h0≤ρ) (h) is also a good surrogate of the true risk R(h), with their difference bouned by the regularization hyper-parameter ρ and approaches 0 when N → ∞.At first glance, the new objective function removes the needs to compute the sample variance in existing bounds (3), but when we have a parametrized distribution of h(y|x), and finite samples DISPLAYFORM2 , estimating the divergence function is not an easy task.

In the next subsection, we will present how recent f-gan networks for variational divergence minimization BID26 ) and Gumbel soft-max sampling BID18 ) can help solve the task.

Discussion: Possibility of Counterfactual Learning: One interesting aspect of our bounds also stresses the need for the stochasticity of the logging policy BID21 ).

For a deterministic logging policy, if the corresponding probability distribution can only have some peaked masses, and zeros elsewhere in its domain, our intution suggests that learning will be difficult, as those regions are never explored.

Our theory well reflects this intuition in the calculation of the divergence term, the integral of form y h 2 (y|x)/h 0 (y|x)dy.

A deterministic policy has a non-zero measure region of h 0 (Y|x) with probability density of h 0 (y|x) = 0, while the corresponding h(y|x) can have finite values in the region.

The resulting integral results is thus unbounded, and in turn induces an unbounded generalization bound, making counterfactual learning in this case not possible.

The derived variance regularized objective (6) requires us to minimize the square root of the condi- DISPLAYFORM0 dy.

For simplicity, we can examine the term inside the expectation operation first.

With simple calculation, we have DISPLAYFORM1 , where f (t) = t 2 − 1 is a convex function in the domain {t : t ≥ 0} with f (1) = 0.

Combining with the expectation operator gives a minimization objective of D f (h||h 0 ; P(X)) (+1 omitted as constant).The above calculation draws connection between our divergence and the f-divergence measure BID25 ).

Follow the f-GAN for variational divergence minimization method proposed in BID26 , we can reach a lower bound of the above objective as DISPLAYFORM2 For the second equality, as f is a convex function and applying Fenchel convex duality (f * = sup u {u v − f (u)}) gives the dual formulation.

Because the expectation is taken w.r.t to x while the supreme is taken w.r.t.

all functions T , we can safely swap the two operators.

We note that the bound is tight when T 0 (x) = f (h/h 0 ), where f is the first order derivative of f as f (t) = 2t BID25 ).The third inequality follows because we restrict T to a family of functions instead of all functions.

Luckily, the universal approximation theorem of neural networks BID15 states that neural networks with arbitrary number of hidden units can approximate continous functions on a compact set with any desired precision.

Thus, by choosing the family of T to be the family of neural networks, the equality condition of the second equality can be satisfied theoretically.

The final objective (10) is a saddle point of a function T (x, y) : X × Y → R that maps input pairs to a scalar value, and the policy we want to learn h(Y|x) acts as a sampling distribution.

Although being a lower bound with achievable equality conditions, theoretically, this saddle point trained with mini-batch estimation is a consistent estimator of the true divergence.

We use D f = sup T T dhdx − f * (T )dh 0 dx to denote the true divergence, and DISPLAYFORM3 dx the empirical estimator we use, whereĥ and h 0 are the emipircal distribution obtained by sampling from the two distribution respectively.

DISPLAYFORM4 Proof.

Let's start by decomposing the estimation error.

DISPLAYFORM5 , where the first term of error comes from restricting the parametric family of T to a family of neural networks, and the second term of error involves the approximation error of an emipirical mean estimation to the true distribution.

By the universal approximation theorem, we have e 0 = 0, and that ∃T ∈ T , such that T = T 0 .For the second term e 1 , we plug in T 0 and have it rewritten as DISPLAYFORM6 For the first term, DISPLAYFORM7 we can see that this is the diffrence between an empirical distribution and the underlying population distribution.

We can verify that the strong law of large numbers (SLLN) applies.

By optimality condition, T 0 = h(y|x) h0(y|x) , where both h and h 0 are probability density functions.

By the bounded loss assumption, the ratio is integrable.

Similarly, f * (T 0 ) = 2T 0 − 1 is also integrable.

Thus, we can apply SLLN and conclude the term → 0.

For the second term, we can apply Theroem 5 from [] and also obtain it → a.s.0.Again, a generative-adversarial approach BID12 ) can be applied.

Toward this end, we represent the T function as a discriminator network parametrized as T w (x, y).

We then parametrize the distribution of our policy h(y|x) as another generator neural network h θ (y|x) mapping x to the probability of sampling y. For structured output problems with discrete values of y, to allow the gradients of samples obtained from sampling backpropagated to all other parameters, we use the Gumbel soft-max sampling BID18 ) methods for differential sampling from the distribution h(y|x).

We list the complete training procedure Alg.

1 for completeness.

DISPLAYFORM8 sampled from logging policy h 0 ; a predefined threshold D 0 ; an initial generator distribution h θ 0 (y|x); an initial discriminator function T w 0 (x, y) ; max iteration I Result: An optimized generator h θ * (y|x) distribution that has minimum divergence to h 0 initialization; whileD f (h||h 0 ; P(X)) > D 0 or iter < I do Sample a mini-batch 'real' samples (x i , y i ) from D ; Sample a mini-batch x from D, and construct 'fake' samples (x i ,ŷ i ) by samplingŷ from h θ t (y|x) with Gubmel soft-max ; Update DISPLAYFORM9 For our purpose of minimizing the variance regularization term, we can similarly derive a training algorithm, as the gradient of t → √ t + 1 can also be backpropagated.

With the above two components, we are now ready to present the full treatment of our end-to-end learning for counterfactual risk minimization from logged data.

The following algorithm solve the robust regularized formulation and for completeness, training for the original ERM formulation in Sec. 3.1 (referred to co-training version in the later experiment sections) is included in Appendix 7.

DISPLAYFORM0 sampled from h 0 ; regularization hyper-parameter ρ, and maximum iteration of divergence minimization steps I, and max epochs for the whole algorithm M AX Result: An optimized generator h * θ (y|x) that is an approximate minimizer of R(w) initialization; while epoch < M AX do / * Update θ to minimize the reweighted loss * / Sample a mini-batch of m samples from D ; Update θ t+1 = θ t − η θ g 1 ; / * Update discriminator and generator for divergence minimization * / Call Algorithm 1 to minimize the divergence D 2 (h||h 0 ; P(X)) with threshold = ρ, and max iter set to I ; end Algorithm 2: Minimizing Variance Regularized Risk -Separate TrainingThe algorithm works in two seperate training steps: 1) update the parameters of the policy h to minimize the reweighed loss 2) update the parameters of the policy/ generator and the discriminator to regularize the variance thus to improve the generalization performance of the new policy.

Exploiting historic data is an important problem in multi-armed bandit and its variants such as contextual bandit and has wide applications BID31 ; BID30 ; BID4 ).

Approaches such as doubly robust estimators BID8 ) have been proposed, and recent theoretical study explored the finite-time minimax risk lower bound of the problem ), and an adaptive learning algorithm (Wang et al. (2017) ) using the theoretical analysis.

Bandits problems can be interpreted as a single-state reinforcement learning (RL) problems, and techniques including doubly robust estimators BID19 ; Thomas & Brunskill (2016) ; BID24 ) have also been extended to RL domains.

Conventional techniques such as Q function learning, and temporal difference learning BID33 ) are alternatives for off-policy learning in RL by accounting for the Markov property of the decision process.

Recent works in deep RL studies have also addressed off-policy updates by methods such as multi-step bootstrapping (Mahmood et al. FORMULA3 ), off-policy training of Q functions BID14 ).Learning from logs traces backs to BID16 and BID28 , where propensity scores are applied to evaluate candidate policies.

In statistics, the problem is also described as treatment effect estimation (Imbens FORMULA4 ), where the focus is to estimate the effect of an intervention from observational studies that are collected by a different intervention.

BID5 derived unbiased counterfactual estimators to study an example of computational advertising; another set of techniques reduce the bandit learning to a weighted supervised learning problems (Zadrozny et al. FORMULA4 ), but is shown to have poor generalization performance BID4 ).Although our variance regularization aims at off-policy learning with bandit feedback, part of the proof comes from the study of generalization bounds in importance sampling problems BID6 ), where the original purpose was to account for the distribution mismatch between training data and testing distribution, also called covariate shift, in supervised learning.

BID7 also discussed variance regularized empirical risk minimization for supervised learning with a convex objective function, which has connections to distributionally robust optimization problem BID2 ).

It will be of further interest to study how our divergence minimization technique can be applied to supervised learning and domain adaptation BID32 ; BID13 ) problems as an alternative to address the distribution match issue.

Regularization for our objective function has close connection to the distributionally robust optimization techniques BID3 ), where instead of minizing the emiprical risk to learn a classifier, we minimize the supreme the emipirical risk over an ellipsoid uncertainty set.

Wasserstein distance between emipircal distribution and test distribution is one of the most well studied contraint and is proven to achieve robust generalization performance (Esfahani & Kuhn (

For empirical evaluation of our proposed algorithms, we follow the conversion from supervised learning to bandit feedback method BID0 ).

For a given supervised dataset DISPLAYFORM0 , we first construct a logging policy h 0 (Y|x), and then for each sample x i , we sample a prediction y i ∼ h 0 (y|x i ), and collect the feedback as δ(y * i , y i ).

For the purpose of benchmarks, we also use the conditional random field (CRF) policy trained on 5% of D * as the logging policy h 0 , and use hamming loss, the number of incorrectly misclassified labels between y i and y * i , as the loss function δ BID34 ).

To create bandit feedback datasets D = {x i , y i , δ i , p i }, each of the samples x i were passed four times to the logging policy h 0 and sampled actions y i were recorded along with the loss value δ i and the propensity score p i = h 0 (y i |x i ).In evaluation, we use two type of evaluation metrics for the probabilistic policy h(Y|x).

The first is the expected loss (referred to as 'EXP' later) R(h) = 1 Ntest i E y∼h(y|xi) δ(y * i , y), a direct measure of the generalization performance of the learned policy.

The second is the average hamming loss of maximum a posteriori probability (MAP) prediction y MAP = arg max h(y|x) derived from the learned policy, as MAP is a faster way to generate predictions without the need for sampling in practice.

However, since MAP predictions only depend on the regions with highest probability, and doesn't take into account the diverse of predictions, two policies with same MAP performance could have very different generalization performance.

Thus, a model with high MAP performance but low EXP performance might be over-fitting, as it may be centering most of its probability masses in the regions where h 0 policy obtained good performance.

Baselines Vanilla importance sampling algorithms using inverse propensity score (IPS), and the counterfactual risk minimization algorithm from Swaminathan & Joachims (2015a) (POEM) are compared, with both L-BFGS optimization and stochastic optimization solvers.

The hyperparameters are selected by performance on validation set and more details of their methods can be found in the original paper BID34 ).Neural network policies without divergence regularization (short as "NN-NoReg" in later discussions) is also compared as baselines, to verify the effectiveness of variance regularization.

Dataset We use four multi-label classification dataset collected in the UCI machine learning repo BID1 ), and perform the supervised to bandit conversion.

We report the statistics in TAB2 in the Appendix.

For these datasets, we choose a three-layer feed-forward neural network for our policy distribution, and a two or three layer feed-forward neural network as the discriminator for divergence minimization.

Detailed configurations can be found in the Appendix 7.For benchmark comparison, we use the separate training version 2 as it has faster convergence and better performance (See Sec. 6.5 for an empirical comparison).

The networks are trained with Adam BID20 ) of learning rate 0.001 and 0.01 respectively for the reweighted loss and the divergence minimization part.

We used PyTorch to implement the pipelines and trained networks with Nvidia K80 GPU cards.

Codes for reproducing the results as well as preprocessed data can be downloaded with the link 1Results by an average of 10 experiment runs are obtained and we report the two evaluation metrics in TAB0 .

We report the regularized neural network policies with two Gumbel-softmax sampling schemes, soft Gumbel soft-max (NN-Soft), and straight-through Gumbel soft-max (NN-Hard).As we can see from the result, by introducing a neural network parametrization of the polices, we are able to improve the test performance by a large margin compared to the baseline CRF policies, as the representation power of networks are often reported to be stronger than other models.

The introduction of additional variance regularization term (comparing NN-Hard/Soft to NN-NoReg), we can observe an additional improvement in both testing loss and MAP prediction loss.

We observe no significant difference between the two Gumbel soft-max sampling schemes.

To study the effectiveness of variance regularization quantitatively, we vary the maximum number of iterations (I in Alg.

2) we take in each divergence minimization sub loop.

For example, 'NNHard-10' indicates that we use ST Gubmel soft-max and set the maximum number of iterations to 10.

Here we set the thresholds for divergence slightly larger so maximum iterations are executed so that results are more comparable.

We plot the expected loss in test sets against the epochs average over 10 runs with error bars using the dataset yeast.

As we can see from the figure, models with no regularization (gray lines in the figure) have higher loss, and slower convergence rate.

As the number of maximum iterations for divergence minimization increases, the test loss decreased faster and the final test loss is also lower.

This behavior suggests that by adding the regularization term, our learned policies are able to generalize better to test sets, and the stronger the regularization we impose by taking more divergence minimization steps, the better the test performance is.

The regularization also helps the training algorithm to converge faster, as shown by the trend.

Our theoretical bounds implies that the generalization performance of our algorithm improves as the number of training samples increases.

We vary the number of passes of training data x was passed to the logging policy to sample an action y, and vary it in the range 2 [1,2,...,8] with log scales.

When the number of training samples in the bandit dataset increases, both models with and without regularization have an increasing test performance in the expected loss and reaches a relatively stable level in the end.

Moreover, regularized policies have a better generalization performance compared to the model without regularization constantly.

This matches our theoretical intuitions that explicitly regularizing the variance can help improve the generalization ability, and that stronger regularization induces better generalization performance.

But as indicated by the MAP performance, after the replay of training samples are more than 2 4 , MAP prediction performance starts to decrease, which suggests the models may be starting over-fitting already.

In this section, we use some experiments to present the difference in two training schemes: cotraining in Alg.

3 and the easier version Alg.

2.

For the second algorithm, we also compare the two Gumbel-softmax sampling schemes in addition, denoted as Gumbel-softmax, and Straight-Through (ST) Gumbel-softmax respectively.

The figures suggest that blending the weighted loss and distribution divergence performs slightly better than the model without regularization, however, the training is much more difficult compared to the separate training scheme, as it's hard to balance the gradient of the two parts of the objective function.

We also observe no significant performance difference between the two sampling schemes of the Gumbel-softmax.

In this section, we discuss how the effect of logging policies, in terms of stochasticity and quality, will affect the learning performance and additional visualizations of other metrics can be found in the Appendix 7.As discussed before, the ability of our algorithm to learn an improved policy relies on the stochasticity of the logging policy.

To test how this stochasticity affects our learning, we modify the parameter of h 0 by introducing a temperature multiplier α.

For CRF logging policies, the prediction is made by normalizing values of w T φ(x, y), where w is the model parameter and can be modified by α with w → αw.

As α becomes higher, h 0 will have a more peaked distribution, and ultimately become a deterministic policy with α → ∞.We varied α in the range of 2 [−1,1,...,8] , and report the average ratio of expected test loss to the logging policy loss of our algorithms (Y-axis in Fig 4a, where smaller values indicate a larger improvement).

We can see that NN polices are performing better than logging policy when the stochasticity of h 0 is sufficient, while after the temperature parameter increases greater than 2 3 , it's much harder and even impossible (ratio ¿ 1) to learn improved NN policies.

We also note here that the stochasticity doesn't affect the expected loss values themselves, and the drop in the ratios mainly resulted from the decreased loss of the logging policy h 0 .

In addition, comparing within NN policies, policies with stronger regularization have slight better performance against models with weaker ones, which in some extent shows the robustness of our learning principle.

The decreasing stochasticity of h 0 makes it harder to obtain an improved NN policy, and our regularization can help the model be more robust and achieve better generalization performance.

b) As h 0 improves, the models constantly outperform the baselines, however, the difficulty is increasing with the quality of h 0 .

Note: more visualizations of other metrics can be found in the appendix 7.Finally, we discusses the impact of logging policies to the our learned improved policies.

Intuitively, a better policy that has lower hamming loss can produce bandit datasets with more correct predictions, however, it's also possible that the sampling biases introduced by the logging policy is larger, and such that some predictions might not be available for feedbacks.

To study the trade-off between better policy accuracy and the sampling biases, we vary the proportion of training data points used to train the logging policy from 0.05 to 1, and compare the performance of our improved policies obtained by in Fig. 4b .

We can see that as the logging policy improves gradually, both NN and NN-Reg policies are outperforming the logging policy, indicating that they are able to address the sampling biases.

The increasing ratios of test expected loss to h 0 performance, as a proxy for relative policy improvement, also matches our intuition that h 0 with better quality is harder to beat.

In this paper, we started from an intuition that explicitly regularizing variance can help improve the generalization performance of off-policy learning for logged bandit datasets, and proposed a new training principle inspired by learning bounds for importance sampling problems.

The theoretical discussion guided us to a training objective as the combination of importance reweighted loss and a regularization term of distribution divergence measuring the distribution match between the logging policy and the policy we are learning.

By applying variational divergence minimization and Gumbel soft-max sampling techniques, we are able to train neural network policies end-to-end to minimize the variance regularized objective.

Evaluations on benchmark datasets proved the effectiveness of our learning principle and training algorithm, and further case studies also verified our theoretical discussion.

Limitations of the work mainly lies in the need for the propensity scores (the probability an action is taken by the logging policy), which may not always be available.

Learning to estimate propensity scores and plug the estimation into our training framework will increase the applicability of our algorithms.

For example, as suggested by BID6 , directly learning importance weights (the ratio between new policy probability to the logging policy probability) has comparable theoretical guarantees, which might be a good extension for the proposed algorithm.

Although the work focuses on off-policy from logged data, the techniques and theorems may be extended to general supervised learning and reinforcement learning.

It will be interesting to study how A. PROOFS DISPLAYFORM0 We apply Lemma 1 to z, importance sampling weight function w(z) = p(z)/p 0 (z) = h(y|x)/h 0 (y|x), and loss l(z)/L, we have DISPLAYFORM1 Thus, we have DISPLAYFORM2 Proof.

For a single hypothesis denoted as δ with values DISPLAYFORM3 By Lemma 1, the variance can be bounded using Reni divergence as DISPLAYFORM4 Applying Bernstein's concentration bounds we have DISPLAYFORM5 σ 2 (Z)+ LM/3 ), we can obtain that with probability at least 1 − η, the following bounds for importance sampling of bandit learning holds DISPLAYFORM6 , where the second inequality comes from the fact that DISPLAYFORM7 sampled from logging policy h 0 ; regularization hyper-parameter λ Result: An optimized generator h * θ (y|x) that is an approximate minimizer of R(w) initialization; while Not Converged do / * Update discriminator * / Sample a mini-batch of 'fake' samples (x i ,ŷ i ) with x i from D andŷ i ∼ h θ t (y|x i ); Sample a mini-batch of 'real' samples (x i , y i ) from D ; Update w t+1 = w t + η w ∂F (T w , h θ )(10) ; / * Update generator * / Sample a mini-batch of m samples from D ; Sample a mini-batch of m 1 'fake' samples ; Estimate the generator gradient as g 2 = F (T w , h θ )(10) ; Update θ t+1 = θ t − η θ (g 1 + λg 2 ) ; end Algorithm 3: Minimizing Variance Regularized Risk -Co-Training Version

We report the statistics of the datasets as in the following table.

For the latter two datasets TMC, (c) The effect of stochasticity of h0 vs ratio of test loss with MAP Figure 5 : As the logging policy becomes more deterministic, NN policies are still able to find improvement over h 0 in a) expected loss and b) loss with MAP predictions.

c) We cannot observe a clear trend in terms of the performance of MAP predictions.

We hypothesize it results from that h 0 policy already has good MAP prediction performance by centering some of the masses.

While NN policies can easily pick up the patterns, it will be difficult to beat the baselines.

We believe this phenomenon worth further investigation.

(c) The quality of h0 vs ratio of expected test loss with MAP Figure 6 : a) As the quality of the logging policy increases, NN policies are still able to find improvement over h 0 in expected loss.

and b) c) For MAP predictions, however, it will be really difficult for NN policies to beat if the logging policy was already exposed to full training data and trained in a supervised fashion.

@highlight

For off-policy learning with bandit feedbacks, we propose a new variance regularized counterfactual learning algorithm, which has both theoretical foundations and superior empirical performance.