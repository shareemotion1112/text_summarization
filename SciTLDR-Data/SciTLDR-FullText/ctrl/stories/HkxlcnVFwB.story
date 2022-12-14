An important problem that arises in reinforcement learning and Monte Carlo methods is estimating quantities defined by the stationary distribution of a Markov chain.

In many real-world applications, access to the underlying transition operator is limited to a fixed set of data that has already been collected, without additional interaction with the environment being available.

We show that consistent estimation remains possible in this scenario, and that effective estimation can still be achieved in important applications.

Our approach is based on estimating a ratio that corrects for the discrepancy between the stationary and empirical distributions, derived from fundamental properties of the stationary distribution, and exploiting constraint reformulations based on variational divergence minimization.

The resulting algorithm, GenDICE, is straightforward and effective.

We prove the consistency of the method under general conditions, provide a detailed error analysis, and demonstrate strong empirical performance on benchmark tasks, including off-line PageRank and off-policy policy evaluation.

Estimation of quantities defined by the stationary distribution of a Markov chain lies at the heart of many scientific and engineering problems.

Famously, the steady-state distribution of a random walk on the World Wide Web provides the foundation of the PageRank algorithm (Langville & Meyer, 2004) .

In many areas of machine learning, Markov chain Monte Carlo (MCMC) methods are used to conduct approximate Bayesian inference by considering Markov chains whose equilibrium distribution is a desired posterior (Andrieu et al., 2002 ).

An example from engineering is queueing theory, where the queue lengths and waiting time under the limiting distribution have been extensively studied (Gross et al., 2018) .

As we will also see below, stationary distribution quantities are of fundamental importance in reinforcement learning (RL) (e.g., Tsitsiklis & Van Roy, 1997) .

Classical algorithms for estimating stationary distribution quantities rely on the ability to sample next states from the current state by directly interacting with the environment (as in on-line RL or MCMC), or even require the transition probability distribution to be given explicitly (as in PageRank).

Unfortunately, these classical approaches are inapplicable when direct access to the environment is not available, which is often the case in practice.

There are many practical scenarios where a collection of sampled trajectories is available, having been collected off-line by an external mechanism that chose states and recorded the subsequent next states.

Given such data, we still wish to estimate a stationary quantity.

One important example is off-policy policy evaluation in RL, where we wish to estimate the value of a policy different from that used to collect experience.

Another example is off-line PageRank (OPR), where we seek to estimate the relative importance of webpages given a sample of the web graph.

Motivated by the importance of these off-line scenarios, and by the inapplicability of classical methods, we study the problem of off-line estimation of stationary values via a stationary distribution corrector.

Instead of having access to the transition probabilities or a next-state sampler, we assume only access to a fixed sample of state transitions, where states have been sampled from an unknown distribution and next-states are sampled according to the Markov chain's transition operator.

This off-line setting is distinct from that considered by most MCMC or on-line RL methods, where it is assumed that new observations can be continually sampled by demand from the environment.

The off-line setting is indeed more challenging than its more traditional on-line counterpart, given that one must infer an asymptotic quantity from finite data.

Nevertheless, we develop techniques that still allow consistent estimation under general conditions, and provide effective estimates in practice.

The main contributions of this work are:

??? We formalize the problem of off-line estimation of stationary quantities, which captures a wide range of practical applications.

??? We propose a novel stationary distribution estimator, GenDICE, for this task.

The resulting algorithm is based on a new dual embedding formulation for divergence minimization, with a carefully designed mechanism that explicitly eliminates degenerate solutions.

??? We theoretically establish consistency and other statistical properties of GenDICE, and empirically demonstrate that it achieves significant improvements on several behavior-agnostic offpolicy evaluation benchmarks and an off-line version of PageRank.

The methods we develop in this paper fundamentally extend recent work in off-policy policy evaluation (Liu et al., 2018; Nachum et al., 2019) by introducing a new formulation that leads to a more general, and as we will show, more effective estimation method.

We first introduce off-line PageRank (OPR) and off-policy policy evaluation (OPE) as two motivating domains, where the goal is to estimate stationary quantities given only off-line access to a set of sampled transitions from an environment.

The celebrated PageRank algorithm (Page et al., 1999) defines the ranking of a web page in terms of its asymptotic visitation probability under a random walk on the (augmented) directed graph specified by the hyperlinks.

If we denote the World Wide Web by a directed graph G = (V, E) with vertices (web pages) v ??? V and edges (hyperlinks) (v, u) ??? E, PageRank considers the random walk defined by the Markov transition operator v ??? u: P (u|v) =

(1?????)

where |v| denotes the out-degree of vertex v and ?? ??? [0, 1) is a probability of "teleporting" to any page uniformly.

Define d t (v) := P (s t = v|s 0 ??? ?? 0 , ???i < t, s i+1 ??? P(??|s i )), where ?? 0 is the initial distribution over vertices, then the original PageRank algorithm explicitly iterates for the limit

The classical version of this problem is solved by tabular methods that simulate Equation 1.

However, we are interested in a more scalable off-line version of the problem where the transition model is not explicitly given.

Instead, consider estimating the rank of a particular web page v from a large web graph, given only a sample D = { (v, u)

from a random walk on G as specified above.

We would still like to estimate d(v ) based on this data.

First, note that if one knew the distribution p by which any vertex v appeared in D, the target quantity could be re-expressed by a simple importance ratio d Policy Evaluation (Preliminaries) An important generalization of this stationary value estimation problem arises in RL in the form of policy evaluation.

Consider a Markov Decision Process (MDP) M = S, A, P, R, ??, ?? 0 (Puterman, 2014), where S is a state space, A is an action space, P (s |s, a) denotes the transition dynamics, R is a reward function, ?? ??? (0, 1] is a discounted factor, and ?? 0 is the initial state distribution.

Given a policy, which chooses actions in any state s according to the probability distribution ??(??|s), a trajectory ?? = (s 0 , a 0 , r 0 , s 1 , a 1 , r 1 , . . .) is generated by first sampling the initial state s 0 ??? ?? 0 , and then for t ??? 0, a t ??? ??(??|s t ), r t ??? R(s t , a t ), and s t+1 ??? P(??|s t , a t ).

The value of a policy ?? is the expected per-step reward defined as:

Here the expectation is taken with respect to the randomness in the state-action pair P (s |s, a) ?? (a |s ) and the reward R (s t , a t ).

Without loss of generality, we assume the limit exists for the average case, and hence R(??) is finite.

Behavior-agnostic Off-Policy Evaluation (OPE) A natural version of policy evaluation that often arises in practice is to estimate

, where p (s, a) is an unknown distribution induced by multiple unknown behavior policies.

This problem is different from the classical form of OPE, where it is assumed that a known behavior policy ?? b is used to collect transitions; in the behavior-agnostic scenario we are considering here, standard importance sampling (IS) estimators (Precup et al., 2000 ) cannot be applied.

Let d ?? t (s, a) = P (s t = s, a t = a|s 0 ??? ?? 0 , ???i < t, a i ??? ?? (??|s i ) , s i+1 ??? P(??|s i , a i )).

The stationary distribution can then be defined as

From this definition note that R(??) and R ?? (??) can be equivalently re-expressed as

Here we see once again that if we had the correction ratio function ?? (s, a) =

is an empirical estimate of p (s, a).

In this way, the behavior-agnostic OPE problem can be reduced to estimating the correction ratio function ?? , as above.

We note that Liu et al. (2018) and Nachum et al. (2019) also exploit Equation 5 to reduce OPE to stationary distribution correction, but these prior works are distinct from the current proposal in different ways.

First, the inverse propensity score (IPS) method of Liu et al. (2018) assumes the transitions are sampled from a single behavior policy, which must be known beforehand; hence that approach is not applicable in behavior-agnostic OPE setting.

Second, the recent DualDICE algorithm (Nachum et al., 2019 ) is also a behavior-agnostic OPE estimator, but its derivation relies on a change-of-variable trick that is only valid for ?? < 1.

This previous formulation becomes unstable when ?? ??? 1, as shown in Section 6 and Appendix E. The behavior-agnostic OPE estimator we derive below in Section 3 is applicable both when ?? = 1 and ?? ??? (0, 1).

This connection is why we name the new estimator GenDICE, for GENeralized stationary DIstribution Correction Estimation.

As noted, there are important estimation problems in the Markov chain and MDP settings that can be recast as estimating a stationary distribution correction ratio.

We first outline the conditions that characterize the correction ratio function ?? , upon which we construct the objective for the GenDICE estimator, and design efficient algorithm for optimization.

We will develop our approach for the more general MDP setting, with the understanding that all the methods and results can be easily specialized to the Markov chain setting.

The stationary distribution ?? ?? ?? defined in Equation 4 can also be characterized via

At first glance, this equation shares a superficial similarity to the Bellman equation, but there is a fundamental difference.

The Bellman operator recursively integrates out future (s , a ) pairs to characterize a current pair (s, a) value, whereas the distribution operator T defined in Equation 6 operates in the reverse temporal direction.

When ?? ??? 1, Equation 6 always has a fixed-point solution.

For ?? = 1, in the discrete case, the fixedpoint exists as long as T is ergodic; in the continuous case, the conditions for fixed-point existence become more complicated (Meyn & Tweedie, 2012) and beyond the scope of this paper.

The development below is based on a divergence D and the following default assumption.

Assumption 1 (Markov chain regularity) For the given target policy ??, the resulting state-action transition operator T has a unique stationary distribution ?? that satisfies D(T ??? ?? ??) = 0.

In the behavior-agnostic setting we consider, one does not have direct access to P for element-wise evaluation or sampling, but instead is given a fixed set of samples from P (s |s, a) p (s, a) with respect to some distribution p (s, a) over S ?? A. Define T p ??,??0 to be a mixture of ?? 0 ?? and T p ; i.e., let

Obviously, conditioning on (s, a, s ) one could easily sample a ??? ?? (a |s ) to form (s, a, s , a ) ??? T p ((s , a ) , (s, a)); similarly, a sample (s , a ) ??? ?? 0 (s ) ?? (a |s ) could be formed from s .

Mixing such samples with probability ?? and 1 ??? ?? respectively yields a sample (s, a, s , a ) ??? T p ??,??0 ((s , a ) , (s, a)).

Based on these observations, the stationary condition for the ratio from Equation 6 can be re-expressed in terms of T p ??,??0 as

where

is the correction ratio function we seek to estimate.

One natural approach to estimating ?? * is to match the LHS and RHS of Equation 8 with respect to some divergence D (?? ??) over the empirical samples.

That is, we consider estimating ?? * by solving the optimization problem min

Although this forms the basis of our approach, there are two severe issues with this naive formulation that first need to be rectified:

??? ?? , which in general involves an intractable integral.

Thus, evaluation of the exact objective is intractable, and neglects the assumption that we only have access to samples from T p ??,??0 and are not able to evaluate it at arbitrary points.

We address each of these two issues in a principled manner.

To avoid degenerate solutions when ?? = 1, we ensure that the solution is a proper density ratio; that is, the property ?? ??? ?? := {?? (??) ??? 0, E p [?? ] = 1} must be true of any ?? that is a ratio of some density to p.

This provides an additional constraint that we add to the optimization formulation min

With this additional constraint, it is obvious that the trivial solution ?? (s, a) = 0 is eliminated as an infeasible point of Eqn (10), along with other degenerate solutions ?? (s, a) = c?? * (s, a) with c = 1.

Unfortunately, exactly solving an optimization with expectation constraints is very complicated in general (Lan & Zhou, 2016) , particularly given a nonlinear parameterization for ?? .

The penalty method (Luenberger & Ye, 2015) provides a much simpler alternative, where a sequence of regularized problems are solved min

with ?? increasing.

The drawback of the penalty method is that it generally requires ?? ??? ??? to ensure the strict feasibility, which is still impractical, especially in stochastic gradient descent.

The infinite ?? may induce unbounded variance in the gradient estimator, and thus, divergence in optimization.

However, by exploiting the special structure of the solution sets to Equation 11, we can show that, remarkably, it is unnecessary to increase ??.

Theorem 1 For ?? ??? (0, 1] and any ?? > 0, the solution to Equation 11 is given by ?? * (s, a) = u(s,a) p(s,a) .

The detailed proof for Theorem 1 is given in Appendix A.1.

By Theorem 1, we can estimate the desired correction ratio function ?? * by solving only one optimization with an arbitrary ?? > 0.

The optimization in Equation 11 involves the integrals T p ??,??0 ??? ?? and E p [?? ] inside nonlinear loss functions, hence appears difficult to solve.

Moreover, obtaining unbiased gradients with a naive approach requires double sampling (Baird, 1995) .

Instead, we bypass both difficulties by applying a dual embedding technique (Dai et al., 2016; .

In particular, we assume the divergence D is in the form of an f -divergence (Nowozin et al., 2016)

ds da where ?? (??) : R + ??? R is a convex, lower-semicontinuous function with ?? (1) = 0.

Plugging this into J (?? ) in Equation 11 we can easily check the convexity of the objective Theorem 2 For an f -divergence with valid ?? defining D ?? , the objective J (?? ) is convex w.r.t.

?? .

The detailed proof is provided in Appendix A.2.

Recall that a suitable convex function can be represented as ?? (x) = max f x??f ????? * (f ), where ?? * is the Fenchel conjugate of ?? (??).

In particular, we have the representation 1 2 x 2 = max u ux ??? 1 2 u 2 , which allows us to re-express the objective as

Applying the interchangeability principle (Shapiro et al., 2014; Dai et al., 2016) , one can replace the inner max in the first term over scalar f to maximize over a function f (??, ??) :

This yields the main optimization formulation, which avoids the aforementioned difficulties and is well-suited for practical optimization as discussed in Section 3.4.

In addition to f -divergence, the proposed estimator Equation 11 is compatible with other divergences, such as the integral probability metrics (IPM) (M??ller, 1997; Sriperumbudur et al., 2009) , while retaining consistency.

Based on the definition of the IPM, these divergences directly lead to min-max optimizations similar to Equation 13 with the identity function as ?? * (??) and different feasible sets for the dual functions.

Specifically, maximum mean discrepancy (MMD) (Smola et al., 2006) requires f H k ??? 1 where H k denotes the RKHS with kernel k; the Dudley metric (Dudley, 2002) requires f BL ??? 1 where f BL := f ??? + ???f 2 ; and Wasserstein distance requires ???f 2 ??? 1.

These additional requirements on the dual function might incur some extra difficulty in practice.

For example, with Wasserstein distance and the Dudley metric, we might need to include an extra gradient penalty (Gulrajani et al., 2017) , which requires additional computation to take the gradient through a gradient.

Meanwhile, the consistency of the surrogate loss under regularization is not clear.

For MMD, we can obtain a closed-form solution for the dual function, which saves the cost of the inner optimization (Gretton et al., 2012) , but with the tradeoff of requiring two independent samples in each outer optimization update.

Moreover, MMD relies on the condition that the dual function lies in some RKHS, which introduces additional kernel parameters to be tuned and in practice may not be sufficiently flexible compared to neural networks.

We have derived a consistent stationary distribution correction estimator in the form of a min-max saddle point optimization Equation 13.

Here, we present a practical instantiation of GenDICE with a concrete objective and parametrization.

We choose the ?? 2 -divergence, which is an f -divergence with ?? (x) = (x ??? 1) 2 and ??

There two major reasons for adopting ?? 2 -divergence:

i) In the behavior-agnostic OPE problem, we mainly use the ratio correction function for estimating

, which is an expectation.

Recall that the error between the estimate and ground-truth can then be bounded by total variation, which is a lower bound of ?? 2 -divergence.

ii) For the alternative divergences, the conjugate of the KL-divergence involves exp (??), which may lead to instability in optimization; while the IPM variants introduce extra constraints on dual function, which may be difficult to be optimized.

The conjugate function of ?? 2 -divergence en-joys suitable numerical properties and provides squared regularization.

We have provided an additional empirical ablation study that investigates the alternative divergences in Appendix E.3.

To parameterize the correction ratio ?? and dual function f we use neural networks, ?? (s, a) = nn w?? (s, a) and f (s, a) = nn w f (s, a), where w ?? and w f denotes the parameters of ?? and f respectively.

Since the optimization requires ?? to be non-negative, we add an extra positive neuron, such as exp (??), log (1 + exp (??)) or (??) 2 at the final layer of nn w?? (s, a).

We empirically compare the different positive neurons in Section 6.3.

For these representations, and unbiased gradient estimator ??? (w?? ,u,w f ) J (??, u, f ) can be obtained straightforwardly, as shown in Appendix B. This allows us to apply stochastic gradient descent to solve the saddle-point problem Equation 14 in a scalable manner, as illustrated in Algorithm 1.

We provide a theoretical analysis for the proposed GenDICE algorithm, following a similar learning setting and assumptions to (Nachum et al., 2019) .

The main result is summarized in the following theorem.

A formal statement, together with the proof, is given in Appendix C. Theorem 3 (Informal) Under mild conditions, with learnable F and H, the error in the objective between the GenDICE estimate,?? , to the solution ??

where E [??] is w.r.t.

the randomness in D and in the optimization algorithms, opt is the optimization error, and approx (F, H) is the approximation induced by (F, H) for parametrization of (??, f ).

The theorem shows that the suboptimality of GenDICE's solution, measured in terms of the objective function value, can be decomposed into three terms: (1) the approximation error approx , which is controlled by the representation flexibility of function classes; (2) the estimation error due to sample randomness, which decays at the order of 1/ ??? N ; and (3) the optimization error, which arises from the suboptimality of the solution found by the optimization algorithm.

As discussed in Appendix C, in special cases, this suboptimality can be bounded below by a divergence between?? and ?? * , and therefore directly bounds the error in the estimated policy value.

There is also a tradeoff between these three error terms.

With more flexible function classes (e.g., neural networks) for F and H, the approximation error approx becomes smaller.

However, it may increase the estimation error (through the constant in front of 1/ ??? N ) and the optimization error (by solving a harder optimization problem).

On the other hand, if F and H are linearly parameterized, estimation and optimization errors tend to be smaller and can often be upper-bounded explicitly in Appendix C.3.

However, the corresponding approximation error will be larger.

Off-policy evaluation with importance sampling (IS) has has been explored in the contextual bandits (Strehl et al., 2010; Dud??k et al., 2011; Wang et al., 2017) , and episodic RL settings (Murphy et al., 2001; Precup et al., 2001) , achieving many empirical successes (e.g., Strehl et al., 2010; Dud??k et al., 2011; Bottou et al., 2013) .

Unfortunately, IS-based methods suffer from exponential variance in long-horizon problems, known as the "curse of horizon" (Liu et al., 2018) .

A few variancereduction techniques have been introduced, but still cannot eliminate this fundamental issue (Jiang & Li, 2015; Thomas & Brunskill, 2016; Guo et al., 2017) .

By rewriting the accumulated reward as an expectation w.r.t.

a stationary distribution, Liu et al. (2018); Gelada & Bellemare (2019) recast OPE as estimating a correction ratio function, which significantly alleviates variance.

However, these methods still require the off-policy data to be collected by a single and known behavior policy, which restricts their practical applicability.

The only published algorithm in the literature, to the best of our knowledge, that solves agnostic-behavior off-policy evaluation is DualDICE (Nachum et al., 2019) .

However, DualDICE was developed for discounted problems and its results become unstable when the discount factor approaches 1 (see below).

By contrast, GenDICE can cope with the more challenging problem of undiscounted reward estimation in the general behavior-agnostic setting.

Note that standard model-based methods (Sutton & Barto, 1998) , which estimate the transition and reward models directly then calculate the expected reward based on the learned model, are also applicable to the behavior-agnostic setting considered here.

Unfortunately, model-based methods typically rely heavily on modeling assumptions about rewards and transition dynamics.

In practice, these assumptions do not always hold, and the evaluation results can become unreliable.

For more related work on MCMC, density ratio estimation and PageRank, please refer to Appendix F.

In this section, we evaluate GenDICE on OPE and OPR problems.

For OPE, we use one or multiple behavior policies to collect a fixed number of trajectories at some fixed trajectory length.

This data is used to recover a correction ratio function for a target policy ?? that is then used to estimate the average reward in two different settings: i) average reward; and ii) discounted reward.

In both settings, we compare with a model-based approach and step-wise weighted IS (Precup et al., 2000) .

We also compare to Liu et al. (2018) (referred to as "IPS" here) in the Taxi domain with a learned behavior policy 1 .

We specifically compare to DualDICE (Nachum et al., 2019) in the discounted reward setting, which is a direct and current state-of-the-art baseline.

For OPR, the main comparison is with the model-based method, where the transition operator is empirically estimated and stationary distribution recovered via an exact solver.

We validate GenDICE in both tabular and continuous cases, and perform an ablation study to further demonstrate its effectiveness.

All results are based on 20 random seeds, with mean and standard deviation plotted.

Our code will be publicly available for reproduction.

Offline PageRank on Graphs One direct application of GenDICE is off-line PageRank (OPR).

We test GenDICE on a Barabasi-Albert (BA) graph (synthetic), and two realworld graphs, Cora and Citeseer.

Details of the graphs are given in Appendix D.

We use the log KL-divergence between estimated stationary distribution and the ground truth as the evaluation metric, with the ground truth computed by an exact solver based on the exact transition operator of the graphs.

We compared GenDICE with modelbased methods in terms of the sample efficiency.

From the results in Figure 1 , GenDICE outperforms the modelbased method when limited data is given.

Even with 20k samples for a BA graph with 100 nodes, where a transition matrix has 10k entries, GenDICE still shows better performance in the offline setting.

This is reasonable since Gen-DICE directly estimates the stationary distribution vector or ratio, while the model-based method needs to learn an entire transition matrix that has many more parameters.

We use a similar taxi domain as in Liu et al. (2018) , where a grid size of 5 ?? 5 yields 2000 states in total (25 ?? 16 ?? 5, corresponding to 25 taxi locations, 16 passenger appearance status and 5 taxi status).

We set the target policy to a final policy ?? after running tabular Q-learning for 1000 iterations, and set another policy ?? + after 950 iterations as the base policy.

The behavior policy is a mixture controlled by ?? as ?? b = (1 ??? ??)?? + ???? + .

For the model-based method, we use a tabular representation for the reward and transition functions, whose entries are estimated from behavior data.

For IS and IPS, we fit a policy via behavior cloning to estimate the policy ratio.

In this specific setting, our methods achieve better results compared to IS, IPS and the model-based method.

Interestingly, with longer horizons, IS cannot improve as much as other methods even with more data, while GenDICE consistently improve and achieves much better results than the baselines.

DualDICE only works with ?? < 1.

GenDICE is more stable than DualDICE when ?? becomes larger (close to 1), while still showing competitive performance for smaller discount factors ??.

We further test our method for OPE on three control tasks: a discrete-control task Cartpole and two continuous-control tasks Reacher and HalfCheetah.

In these tasks, observations (or states) are continuous, thus we use neural network function approximators and stochastic optimization.

Since DualDICE (Nachum et al., 2019) has shown the state-of-the-art performance on discounted OPE, we mainly compare with it in the discounted reward case.

We also compare to IS with a learned policy via behavior cloning and a neural model-based method, similar to the tabular case, but with neural network as the function approximator.

All neural networks are feed-forward with two hidden layers of dimension 64 and tanh activations.

More details can be found in Appendix D.

Due to limited space, we put the discrete control results in Appendix E and focus on the more challenging continuous control tasks.

Here, the good performance of IS and model-based methods in Section 6.1 quickly deteriorates as the environment becomes complex, i.e., with a continuous action space.

Note that GenDICE is able to maintain good performance in this scenario, even when using function approximation and stochastic optimization.

This is reasonable because of the difficulty of fitting to the coupled policy-environment dynamics with a continuous action space.

Here we also empirically validate GenDICE with off-policy data collected by multiple policies.

As illustrated in Figure 3 , all methods perform better with longer trajectory length or more trajectories.

When ?? becomes larger, i.e., the behavior policies are closer to the target policy, all methods performs better, as expected.

Here, GenDICE demonstrates good performance both on averagereward and discounted reward cases in different settings.

The right two figures in each row show the log MSE curve versus optimization steps, where GenDICE achieves the smallest loss.

In the discounted reward case, GenDICE shows significantly better and more stable performance than the strong baseline, DualDICE.

Figure 4 also shows better performance of GenDICE than all baselines in the more challenging HalfCheetah domain.

Each plot in the second row shows the average reward case.

Finally, we conduct an ablation study on GenDICE to study its robustness and implementation sensitivities.

We investigate the effects of learning rate, activation function, discount factor, and the specifically designed ratio constraint.

We further demonstrate the effect of the choice of divergences and the penalty weight in Appendix E.3.

Model-Based Importance Sampling DualDICE GenDICE (ours) Figure 4 : Results on HalfCheetah.

Plots from left to the right show the log MSE of estimated average per-step reward over different truncated lengths, numbers of trajectories, and behavior policies in discounted and average reward cases.

Effects of the Learning Rate Since we are using neural network as the function approximator, and stochastic optimization, it is necessary to show sensitivity to the learning rate with {0.0001, 0.0003, 0.001, 0.003}, with results in Figure 5 .

When ?? = 0.33, i.e., the OPE tasks are relatively easier and GenDICE obtains better results at all learning rate settings.

However, when ?? = 0.0, i.e., the estimation becomes more difficult and only GenDICE only obtains reasonable results with the larger learning rate.

Generally, this ablation study shows that the proposed method is not sensitive to the learning rate, and is easy to train.

We further investigate the effects of the activation function on the last layer, which ensure the non-negative outputs required for the ratio.

To better understand which activation function will lead to stable trainig for the neural correction estimator, we empirically compare using i) (??) 2 ; ii) log(1 + exp(??)); and iii) exp(??).

In practice, we use the (??) 2 since it achieves low variance and better performance in most cases, as shown in Figure 5 .

We vary ?? ??? {0.95, 0.99, 0.995, 0.999, 1.0} to probe the sensitivity of GenDICE.

Specifically, we compare to DualDICE, and find that GenDICE is stable, while DualDICE becomes unstable when the ?? becomes large, as shown in Figure 6 .

GenDICE is also more general than DualDICE, as it can be applied to both the average and discounted reward cases.

In Section 3, we highlighted the importance of the ratio constraint.

Here we investigate the trivial solution issue without the constraint.

The results in Figure 6 demonstrate the necessity of adding the constraint penalty, since a trivial solution prevents an accurate corrector from being recovered (green line in left two figures).

In this paper, we proposed a novel algorithm GenDICE for general stationary distribution correction estimation, which can handle both the discounted and average stationary distribution given multiple behavior-agnostic samples.

Empirical results on off-policy evaluation and offline PageRank show the superiority of proposed method over the existing state-of-the-art methods.

the existence of the stationary distribution.

Our discussion is all based on this assumption.

Assumption 1 Under the target policy, the resulted state-action transition operator T has a unique stationary distribution in terms of the divergence D (??||??).

If the total variation divergence is selected, the Assumption 1 requires the transition operator should be ergodic, as discussed in Meyn & Tweedie (2012).

Theorem 1 For arbitrary ?? > 0, the solution to the optimization Eqn (11) is

Proof For ?? ??? (0, 1), there is not degenerate solutions to D T p ??,??0 ??? ?? ||p ?? ?? .

The optimal solution is a density ratio.

Therefore, the extra penalty E p(x) [?? (x)] ??? 1 2 does not affect the optimality for ????? > 0.

negative, and the the density ratio

p(x) leads to zero for both terms.

Then, the density ratio is a solution to J (?? ).

For any other non-negative function ?? (x) ??? 0, if it is the optimal solution to J (?? ), then, we have

We denote ?? (x) = p (x) ?? (x), which is clearly a density function.

Then, the optimal conditions in Equation 15 imply

or equivalently, ?? is the stationary distribution of T .

We have thus shown the optimal ?? (x) =

is the target density ratio.

Proof Since the ?? is convex, we consider the Fenchel dual representation of the f -divergence

2 is also convex, which concludes the proof.

Inputs: Convex function ?? and its Fenchel conjugate ??

, initial state s 0 ??? ?? 0 , target policy ??, distribution corrector nn w?? (??, ??), nn w f (??, ??), constraint scalar u, learning rates ?? ?? , ?? f , ?? u , number of iterations K, batch size B.

Sample actions a

end for Return nn w?? .

We provide the unbiased gradient estimator for ??? w?? ,u,w f J (??, u, f ) in Eqn (14) below:

Then, we have the psuedo code which applies SGD for solving Eqn (14).

For convenience, we repeat here the notation defined in the main text.

The saddle-point reformulation of the objective function of GenDICE is:

To avoid the numerical infinity in D ?? (??||??), we induced the bounded version as

is still a valid divergence, and therefore the optimal solution ?? * is still the stationary density ratio

p(x) .

We denote the J (??, ??, f ) as the empirical surrogate of

with optimal (f * , u * ), and

We apply some optimization algorithm for J (??, u, f ) over space (H, F, R), leading to the output ?? , u, f .

Under Assumption 2, we need only consider ?? ??? ??? C, then, the corresponding dual u = E p (?? ) ??? 1 ??? u ??? U := {|u| ??? (C + 1)}. We choose the ?? * (??) is a ??-Lipschitz continuous, then, the We consider the error between?? and ?? * using standard arguments (Shalev-Shwartz & Ben-David, 2014; Bach, 2014)

Remark: In some special cases, the suboptimality also implies the distance between?? and ?? * .

Specifically,for ?? = 1, if the transition operator P ?? can be represented as P ?? = Q??Q ???1 where Q denotes the (countable) eigenfunctions and ?? denotes the diagonal matrix with eigenvalues, the largest of which is 1.

We consider ?? (??) as identity and f ??? F := span (Q) , f p,2 ??? 1 , then the d (??, ?? * ) will bounded from below by a metric between ?? and ?? * .

Particularly, we have

Rewrite ?? = ???? * + ??, where ?? ??? span Q \?? * , then

Recall the optimality of ??

We start with the following error decomposition:

??? For 1 , we have

.

We consider the terms one-by-one.

By definition, we have

which is induced by introducing F for dual approximation.

For the third term

where we define est :

Therefore, we can now bound 1 as

We consider the terms from right to left.

For the term J (?? * H ) ??? J (?? * ), we have

, which is induced by restricting the function space to H. The second term is nonpositive, due to the optimality of (u * , f * ).

The final inequality comes from the fact that

where the second term is nonpositive, thanks to the optimality of?? * H .

Finally, for the term J (?? * H ) ??? J (?? * H ), using the same argument in Equation 21, we have

Therefore, we can bound 2 by 2 ??? C ??,C,?? approx (H) + C P ?? ,??,?? (F) + 2 est .

In sum, we have d (?? , ?? * ) ??? 4 est +?? opt + 2C P ?? ,??,?? approx (F) + C ??,C,?? approx (H) .

In the following sections, we will bound the est and?? opt .

In this section, we analyze the statistical error

We mainly focus on the batch RL setting with

, which has been studied by previous authors (e.g., Sutton et al., 2012; Nachum et al., 2019) .

However, as discussed in the literature (Antos et al., 2008; Lazaric et al., 2012; Dai et al., 2018; Nachum et al., 2019) , using the blocking technique of Yu (1994) , the statistical error provided here can be generalized to ??-mixing samples in a single sample path.

We omit this generalization for the sake of expositional simplicity.

To bound the est , we follow similar arguments by Dai et al. (2018) ; Nachum et al. (2019) via the covering number.

For completeness, the definition is given below.

The Pollard's tail inequality bounds the maximum deviation via the covering number of a function class:

are i.i.d.

samples from some distribution.

Then, for any given > 0,

The covering number can then be bounded in terms of the function class's pseudo-dimension:

Lemma 5 (Haussler (1995), Corollary 3) For any set X , any points x 1:N ??? X N , any class F of functions on X taking values in [0, M ] with pseudo-dimension D F < ???, and any > 0,

The statistical error est can be bounded using these lemmas.

Lemma 6 (Stochastic error) Under the Assumption 2, if ?? * is ??-Lipschitz continuous and the psuedo-dimension of H and F are finite, with probability at least 1 ??? ??, we have

Proof The proof works by verifying the conditions in Lemma 4 and computing the covering number.

2 u 2 , we will apply Lemma 4 with Z = ??? ?? ???, Z i = (x i , x i ), and G = h H??F ??U .

We check the boundedness of h ??,u,f (x, x ).

Based on Assumption 2, we only consider the ?? ??? H and u ??? U bounded by C and C + 1.

We also rectify the f ??? ??? C. Then, we can bound the h ??? :

where C ?? = max t???[???C,C] ????? * (t).

Thus, by Lemma 4, we have

Next, we check the covering number of G. Firstly, we bound the distance in G,

Denote the pseudo-dimension of H and F as D H and D F , respectively, we have

, we obtain the bound for the statistical error:

In this section, we investigate the optimization error

.

Notice our estimator min ?? ???H max f ???F ,u???U J (??, u, f ) is compatible with different parametrizations for (H, F) and different optimization algorithms, the optimization error will be different.

For the general neural network for (??, f ), although there are several progress recently (Lin et al., 2018; Jin et al., 2019; Lin et al., 2019) about the convergence to a stationary point or local minimum, it remains a largely open problem to quantify the optimization error, which is out of the scope of this paper.

Here, we mainly discuss the convergence rate with tabular, linear and kernel parametrization for (??, f ).

Particularly, we consider the linear parametrization particularly, i.e., ?? (x) = ?? w ?? ?? (x) , f (x) = w f ?? (x), and ?? (??) : R ??? R + is convex.

There are many choices of the ?? (??), e.g., exp (??), log (1 + exp (??)) and (??) 2 .

Obviously, even with such nonlinear mapping, the J (??, u, f ) is still convex-concave w.r.t (w ?? , w f , u) by the convex composition rule.

We can bound the?? opt by the primal-dual gap gap :

With vanilla SGD, we have

, where T is the optimization steps (Nemirovski et al., 2009) .

, where the E [??] is taken w.r.t.

randomness in SGD.

We are now ready to state the main theorm in a precise way:

Under Assumptions 2 and 1 , the stationary distribution ?? exists, i.e.,

* , and the psuedo-dimension of H and F are finite, the error between the GenDICE estimate to ??

where E [??] is w.r.t.

the randomness in sample D and in the optimization algorithms.

opt is the optimization error, and approx (F, H) is the approximation induced by (F, H) for parametrization of (??, f ).

Proof We have the total error as

where approx := 2C T ,??,?? approx (F) + C ??,C,?? approx (H).

For opt , we can apply the results for SGD in Appendix C.3.

We can bound the E [ est ] by Lemma 6.

Specifically, we have

Plug all these bounds into Equation 25, we achieve the conclusion.

For the Taxi domain, we follow the same protocol as used in Liu et al. (2018) .

The behavior and target policies are also taken from Liu et al. (2018) (referred in their work as the behavior policy for ?? = 0).

We use a similar taxi domain, where a grid size of 5??5 yields 2000 states in total (25??16??5, cor-responding to 25 taxi locations, 16 passenger appearance status and 5 taxi status).

We set our target policy as the final policy ?? * after running Q-learning (Sutton & Barto, 1998) for 1000 iterations, and set another policy ?? + after 950 iterations as our base policy.

The behavior policy is a mixture policy controlled by ?? as ?? = (1 ??? ??)?? * + ???? + , i.e., the larger ?? is, the behavior policy is more close to the target policy.

In this setting, we solve for the optimal stationary ratio ?? exactly using matrix operations.

Since Liu et al. (2018) perform a similar exact solve for |S| variables ??(s), for better comparison we also perform our exact solve with respect to |S| variables ?? (s).

Specifically, the final objective of importance sampling will require knowledge of the importance weights ??(a|s)/p(a|s).

For offline PageRank, the graph statistics are illustrated in Table 1 , and the degree statistics and graph visualization are shown in Figure 7 .

For the BarabasiAlbert (BA) Graph, it begins with an initial connected network of m 0 nodes in the network.

Each new node is connected to m ??? m 0 existing nodes with a probability that is proportional to the number of links that the existing nodes already have.

Intuitively, heavily linked nodes ('hubs') tend to quickly accumulate even more links, while nodes with only a few links are unlikely to be chosen as the destination for a new link.

The new nodes have a 'preference' to attach themselves to the already heavily linked nodes.

For two real-world graphs, it is built upon the real-world citation networks.

In our experiments, the weights of the BA graph is randomly drawn from a standard Gaussian distribution with normalization to ensure the property of the transition matrix.

The offline data is collected by a random walker on the graph, which consists the initial state and next state in a single trajectory.

In experiments, we vary the number of off-policy samples to validate the effectiveness of GenDICE with limited offline samples provided.

We use the Cartpole, Reacher and HalfCheetah tasks as given by OpenAI Gym.

In importance sampling, we learn a neural network policy via behavior cloning, and use its probabilities for computing importance weights ?? * (a|s)/??(a|s).

All neural networks are feed-forward with two hidden layers of dimension 64 and tanh activations.

Discrete Control Tasks We modify the Cartpole task to be infinite horizon: We use the same dynamics as in the original task but change the reward to be ???1 if the original task returns a termination (when the pole falls below some threshold) and 1 otherwise.

We train a policy on this task with standard Deep Q-Learning (Mnih et al., 2013) until convergence.

We then define the target policy ?? * as a weighted combination of this pre-trained policy (weight 0.7) and a uniformly random policy (weight 0.3).

The behavior policy ?? for a specific 0 ??? ?? ??? 1 is taken to be a weighted combination of the pre-trained policy (weight 0.55 + 0.15??) and a uniformly random policy (weight 0.45 ??? 0.15??).

We train each stationary distribution correction estimation method using the Adam optimizer with batches of size 2048 and learning rates chosen using a hyperparameter search from {0.0001, 0.0003, 0.001, 0.003} and choose the best one as 0.0003.

For the Reacher task, we train a deterministic policy until convergence via DDPG (Lillicrap et al., 2015) .

We define the target policy ?? as a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.1.

The behavior policy ?? b for a specific 0 ??? ?? ??? 1 is taken to be a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.4 ??? 0.3??.

We train each stationary distribution correction estimation method using the Adam optimizer with batches of size 2048 and learning rates chosen using a hyperparameter search from {0.0001, 0.0003, 0.001, 0.003} and the optimal learning rate found was 0.003).

For the HalfCheetah task, we also train a deterministic policy until convergence via DDPG (Lillicrap et al., 2015) .

We define the target policy ?? as a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.1.

The behavior policy ?? b for a specific 0 ??? ?? ??? 1 is taken to be a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.2 ??? 0.1??.

We train each stationary distribution correction estimation method using the Adam optimizer with batches of size 2048 and learning rates chosen using a hyperparameter search from {0.0001, 0.0003, 0.001, 0.003} and the optimal learning rate found was 0.003.

E.1 OPE FOR DISCRETE CONTROL On the discrete control task, we modify the Cartpole task to be infinite horizon: the original dynamics is used but with a modified reward function: the agent will receive ???1 if the environment returns a termination (i.e., the pole falls below some threshold) and 1 otherwise.

As shown in Figure  3 , our method shows competitive results with IS and Model-Based in average reward case, but our proposed method finally outperforms these two methods in terms of log MSE loss.

Specifically, it is relatively difficult to fit a policy with data collected by multiple policies, which renders the poor performance of IS.

In this section, we show more results on the continuous control tasks, i.e., HalfCheetah and Reacher.

Figure 9 shows the log MSE towards training steps, and GenDICE outperforms other baselines with different behavior policies.

Figure 10 better illustrates how our method beat other baselines, and can accurately estimate the reward of the target policy.

Besides, Figure 11 shows GenDICE gives better reward estimation of the target policy.

In these figures, the left three figures show the performance with off-policy dataset collected by single behavior policy from more difficult to easier tasks.

The right two figures show the results, where off-policy dataset collected by multiple behavior policies.

Figure 12 shows the ablation study results in terms of estimated rewards.

The left two figures shows the effects of different learning rate.

When ?? = 0.33, i.e., the OPE tasks are relatively easier, GenDICE gets relatively good results in all learning rate settings.

However, when ?? = 0.0, i.e., the estimation becomes more difficult, only GenDICE in larger learning rate gets reasonable estimation.

Interestingly, we can see with larger learning rates, the performance becomes better, and when learning rate is 0.001 with ?? = 0.0, the variance is very high, showing some cases the estimation becomes more accurate.

The right three figures show different activation functions with different behavior policy.

The square and softplus function works well; while the exponential function shows poor performance under some settings.

In practice, we use the square function since its low variance and better performance in most cases.

Model-Based Importance Sampling DualDICE GenDICE (ours) Oracle Figure 11 : Results on HalfCheetah.

Each plot in the first row shows the estimated average step reward over training and different behavior policies (higher ?? corresponds to a behavior policy closer to the target policy.

Although any valid divergence between p ?? ?? and T p ??,??0 ??? ?? in our estimator is consistent, which will always lead to the stationary distribution correction ratio asymptotically, and any ?? > 0 will guarantee the normalization constraint, i.e., E p [?? ] = 1, as we discussed in main text, different Figure 12 : Results of ablation study with different learning rates and activation functions.

The plots show the estimated average step reward over training and different behavior policies .

choices of the divergences and ?? may incur difficulty in the numerical optimization procedure.

In this section, we investigate the empirical effects of the choice of f -divergence and IPM, and the weight of constrant regularization ??.

To avoid the effects of other factors in the estimator, e.g., function parametrization, we focus on the offline PageRank task on BA graph with 100 nodes and 10k offline samples.

All the performances are evaluated with 20 random trials.

We test the GenDICE with several other alternative divergences, e.g., Wasserstein-1 distance, Jensen-Shannon divergence, KL-divergence, Hellinger divergence, and MMD.

To ensure the dual function to be 1-Lipchitz, we add the gradient penalty.

We use a learned Gaussian kernel in MMD, similar to Li et al. (2017) .

As we can see in Figure 13 (a), with these different divergences, the proposed GenDICE estimator can always achieve significantly better performance compared with the model-based estimator, showing that the GenDICE estimator is compatible with many different divergences.

Most of the divergences, with appropriate extra techniques to handle the difficulties in optimization and carefully tuning for extra parameters, can achieve similar performances, consistent with phenomena in the variants of GANs (Lucic et al., 2018) .

However, KL-divergence is an outlier, performing noticeably worse, which might be caused by the ill-behaved exp (??) in its conjugate function.

The ?? 2 -divergence and JS-divergence are better, which achieve good performances with fewer parameters to be tuned.

The effect of the penalty weight ?? is illustrated the in Figure 13(b) .

We vary the ?? ??? [0.1, 5] with ?? 2 -divergence in the GenDICE estimator.

Within a large range of ??, the performances of the proposed GenDICE are quite consistent, which justifies Theorem 1.

The penalty multiplies with ??.

Therefore, with ?? increases, the variance of the stochastic gradient estimator also increases, which explains the variance increasing in large ?? in Figure 13(b) .

In practice, ?? = 1 is a reasonable choice for general cases.

Markov Chain Monte Carlo Classical MCMC (Brooks et al., 2011; Gelman et al., 2013) aims at sampling from ?? ?? by iteratively simulting from the transition operator.

It requires continuous interaction with the transition operator and heavy computational cost to update many particles.

Amor-tized SVGD (Wang & Liu, 2016) and Adversarial MCMC (Song et al., 2017; Li et al., 2019) alleviate this issue via combining with neural network, but they still interact with the transition operator directly, i.e., in an on-policy setting.

The major difference of our GenDICE is the learning setting: we only access the off-policy dataset, and cannot sample from the transition operator.

The proposed GenDICE leverages stationary density ratio estimation for approximating the stationary quantities, which distinct it from classical methods.

Density ratio estimation is a fundamental tool in machine learning and much related work exists.

Classical density ratio estimation includes moment matching (Gretton et al.) , probabilistic classification (Bickel et al., 2007) , and ratio matching (Nguyen et al., 2008; Sugiyama et al., 2008; Kanamori et al., 2009) .

These classical methods focus on estimating the ratio between two distributions with samples from both of them, while GenDICE estimates the density ratio to a stationary distribution of a transition operator, from which even one sample is difficult to obtain.

PageRank Yao & Schuurmans (2013) developed a reverse-time RL framework for PageRank via solving a reverse Bellman equation, which is less sensitive to graph topology and shows faster adaptation with graph change.

However, Yao & Schuurmans (2013) still considers the online manner, which is different with our OPR setting.

<|TLDR|>

@highlight

In this paper, we proposed a novel algorithm, GenDICE, for general stationary distribution correction estimation, which can handle both discounted and average off-policy evaluation on multiple behavior-agnostic samples.