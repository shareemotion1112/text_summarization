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

• We formalize the problem of off-line estimation of stationary quantities, which captures a wide range of practical applications.

• We propose a novel stationary distribution estimator, GenDICE, for this task.

The resulting algorithm is based on a new dual embedding formulation for divergence minimization, with a carefully designed mechanism that explicitly eliminates degenerate solutions.

• We theoretically establish consistency and other statistical properties of GenDICE, and empirically demonstrate that it achieves significant improvements on several behavior-agnostic offpolicy evaluation benchmarks and an off-line version of PageRank.

The methods we develop in this paper fundamentally extend recent work in off-policy policy evaluation (Liu et al., 2018; Nachum et al., 2019) by introducing a new formulation that leads to a more general, and as we will show, more effective estimation method.

We first introduce off-line PageRank (OPR) and off-policy policy evaluation (OPE) as two motivating domains, where the goal is to estimate stationary quantities given only off-line access to a set of sampled transitions from an environment.

The celebrated PageRank algorithm (Page et al., 1999) defines the ranking of a web page in terms of its asymptotic visitation probability under a random walk on the (augmented) directed graph specified by the hyperlinks.

If we denote the World Wide Web by a directed graph G = (V, E) with vertices (web pages) v ∈ V and edges (hyperlinks) (v, u) ∈ E, PageRank considers the random walk defined by the Markov transition operator v → u: P (u|v) =

(1−η)

where |v| denotes the out-degree of vertex v and η ∈ [0, 1) is a probability of "teleporting" to any page uniformly.

Define d t (v) := P (s t = v|s 0 ∼ µ 0 , ∀i < t, s i+1 ∼ P(·|s i )), where µ 0 is the initial distribution over vertices, then the original PageRank algorithm explicitly iterates for the limit

The classical version of this problem is solved by tabular methods that simulate Equation 1.

However, we are interested in a more scalable off-line version of the problem where the transition model is not explicitly given.

Instead, consider estimating the rank of a particular web page v from a large web graph, given only a sample D = { (v, u)

from a random walk on G as specified above.

We would still like to estimate d(v ) based on this data.

First, note that if one knew the distribution p by which any vertex v appeared in D, the target quantity could be re-expressed by a simple importance ratio d Policy Evaluation (Preliminaries) An important generalization of this stationary value estimation problem arises in RL in the form of policy evaluation.

Consider a Markov Decision Process (MDP) M = S, A, P, R, γ, µ 0 (Puterman, 2014), where S is a state space, A is an action space, P (s |s, a) denotes the transition dynamics, R is a reward function, γ ∈ (0, 1] is a discounted factor, and µ 0 is the initial state distribution.

Given a policy, which chooses actions in any state s according to the probability distribution π(·|s), a trajectory β = (s 0 , a 0 , r 0 , s 1 , a 1 , r 1 , . . .) is generated by first sampling the initial state s 0 ∼ µ 0 , and then for t ≥ 0, a t ∼ π(·|s t ), r t ∼ R(s t , a t ), and s t+1 ∼ P(·|s t , a t ).

The value of a policy π is the expected per-step reward defined as:

Here the expectation is taken with respect to the randomness in the state-action pair P (s |s, a) π (a |s ) and the reward R (s t , a t ).

Without loss of generality, we assume the limit exists for the average case, and hence R(π) is finite.

Behavior-agnostic Off-Policy Evaluation (OPE) A natural version of policy evaluation that often arises in practice is to estimate

, where p (s, a) is an unknown distribution induced by multiple unknown behavior policies.

This problem is different from the classical form of OPE, where it is assumed that a known behavior policy π b is used to collect transitions; in the behavior-agnostic scenario we are considering here, standard importance sampling (IS) estimators (Precup et al., 2000 ) cannot be applied.

Let d π t (s, a) = P (s t = s, a t = a|s 0 ∼ µ 0 , ∀i < t, a i ∼ π (·|s i ) , s i+1 ∼ P(·|s i , a i )).

The stationary distribution can then be defined as

From this definition note that R(π) and R γ (π) can be equivalently re-expressed as

Here we see once again that if we had the correction ratio function τ (s, a) =

is an empirical estimate of p (s, a).

In this way, the behavior-agnostic OPE problem can be reduced to estimating the correction ratio function τ , as above.

We note that Liu et al. (2018) and Nachum et al. (2019) also exploit Equation 5 to reduce OPE to stationary distribution correction, but these prior works are distinct from the current proposal in different ways.

First, the inverse propensity score (IPS) method of Liu et al. (2018) assumes the transitions are sampled from a single behavior policy, which must be known beforehand; hence that approach is not applicable in behavior-agnostic OPE setting.

Second, the recent DualDICE algorithm (Nachum et al., 2019 ) is also a behavior-agnostic OPE estimator, but its derivation relies on a change-of-variable trick that is only valid for γ < 1.

This previous formulation becomes unstable when γ → 1, as shown in Section 6 and Appendix E. The behavior-agnostic OPE estimator we derive below in Section 3 is applicable both when γ = 1 and γ ∈ (0, 1).

This connection is why we name the new estimator GenDICE, for GENeralized stationary DIstribution Correction Estimation.

As noted, there are important estimation problems in the Markov chain and MDP settings that can be recast as estimating a stationary distribution correction ratio.

We first outline the conditions that characterize the correction ratio function τ , upon which we construct the objective for the GenDICE estimator, and design efficient algorithm for optimization.

We will develop our approach for the more general MDP setting, with the understanding that all the methods and results can be easily specialized to the Markov chain setting.

The stationary distribution µ π γ defined in Equation 4 can also be characterized via

At first glance, this equation shares a superficial similarity to the Bellman equation, but there is a fundamental difference.

The Bellman operator recursively integrates out future (s , a ) pairs to characterize a current pair (s, a) value, whereas the distribution operator T defined in Equation 6 operates in the reverse temporal direction.

When γ ≤ 1, Equation 6 always has a fixed-point solution.

For γ = 1, in the discrete case, the fixedpoint exists as long as T is ergodic; in the continuous case, the conditions for fixed-point existence become more complicated (Meyn & Tweedie, 2012) and beyond the scope of this paper.

The development below is based on a divergence D and the following default assumption.

Assumption 1 (Markov chain regularity) For the given target policy π, the resulting state-action transition operator T has a unique stationary distribution µ that satisfies D(T • µ µ) = 0.

In the behavior-agnostic setting we consider, one does not have direct access to P for element-wise evaluation or sampling, but instead is given a fixed set of samples from P (s |s, a) p (s, a) with respect to some distribution p (s, a) over S × A. Define T p γ,µ0 to be a mixture of µ 0 π and T p ; i.e., let

Obviously, conditioning on (s, a, s ) one could easily sample a ∼ π (a |s ) to form (s, a, s , a ) ∼ T p ((s , a ) , (s, a)); similarly, a sample (s , a ) ∼ µ 0 (s ) π (a |s ) could be formed from s .

Mixing such samples with probability γ and 1 − γ respectively yields a sample (s, a, s , a ) ∼ T p γ,µ0 ((s , a ) , (s, a)).

Based on these observations, the stationary condition for the ratio from Equation 6 can be re-expressed in terms of T p γ,µ0 as

where

is the correction ratio function we seek to estimate.

One natural approach to estimating τ * is to match the LHS and RHS of Equation 8 with respect to some divergence D (· ·) over the empirical samples.

That is, we consider estimating τ * by solving the optimization problem min

Although this forms the basis of our approach, there are two severe issues with this naive formulation that first need to be rectified:

• τ , which in general involves an intractable integral.

Thus, evaluation of the exact objective is intractable, and neglects the assumption that we only have access to samples from T p γ,µ0 and are not able to evaluate it at arbitrary points.

We address each of these two issues in a principled manner.

To avoid degenerate solutions when γ = 1, we ensure that the solution is a proper density ratio; that is, the property τ ∈ Ξ := {τ (·) ≥ 0, E p [τ ] = 1} must be true of any τ that is a ratio of some density to p.

This provides an additional constraint that we add to the optimization formulation min

With this additional constraint, it is obvious that the trivial solution τ (s, a) = 0 is eliminated as an infeasible point of Eqn (10), along with other degenerate solutions τ (s, a) = cτ * (s, a) with c = 1.

Unfortunately, exactly solving an optimization with expectation constraints is very complicated in general (Lan & Zhou, 2016) , particularly given a nonlinear parameterization for τ .

The penalty method (Luenberger & Ye, 2015) provides a much simpler alternative, where a sequence of regularized problems are solved min

with λ increasing.

The drawback of the penalty method is that it generally requires λ → ∞ to ensure the strict feasibility, which is still impractical, especially in stochastic gradient descent.

The infinite λ may induce unbounded variance in the gradient estimator, and thus, divergence in optimization.

However, by exploiting the special structure of the solution sets to Equation 11, we can show that, remarkably, it is unnecessary to increase λ.

Theorem 1 For γ ∈ (0, 1] and any λ > 0, the solution to Equation 11 is given by τ * (s, a) = u(s,a) p(s,a) .

The detailed proof for Theorem 1 is given in Appendix A.1.

By Theorem 1, we can estimate the desired correction ratio function τ * by solving only one optimization with an arbitrary λ > 0.

The optimization in Equation 11 involves the integrals T p γ,µ0 • τ and E p [τ ] inside nonlinear loss functions, hence appears difficult to solve.

Moreover, obtaining unbiased gradients with a naive approach requires double sampling (Baird, 1995) .

Instead, we bypass both difficulties by applying a dual embedding technique (Dai et al., 2016; .

In particular, we assume the divergence D is in the form of an f -divergence (Nowozin et al., 2016)

ds da where φ (·) : R + → R is a convex, lower-semicontinuous function with φ (1) = 0.

Plugging this into J (τ ) in Equation 11 we can easily check the convexity of the objective Theorem 2 For an f -divergence with valid φ defining D φ , the objective J (τ ) is convex w.r.t.

τ .

The detailed proof is provided in Appendix A.2.

Recall that a suitable convex function can be represented as φ (x) = max f x·f −φ * (f ), where φ * is the Fenchel conjugate of φ (·).

In particular, we have the representation 1 2 x 2 = max u ux − 1 2 u 2 , which allows us to re-express the objective as

Applying the interchangeability principle (Shapiro et al., 2014; Dai et al., 2016) , one can replace the inner max in the first term over scalar f to maximize over a function f (·, ·) :

This yields the main optimization formulation, which avoids the aforementioned difficulties and is well-suited for practical optimization as discussed in Section 3.4.

In addition to f -divergence, the proposed estimator Equation 11 is compatible with other divergences, such as the integral probability metrics (IPM) (Müller, 1997; Sriperumbudur et al., 2009) , while retaining consistency.

Based on the definition of the IPM, these divergences directly lead to min-max optimizations similar to Equation 13 with the identity function as φ * (·) and different feasible sets for the dual functions.

Specifically, maximum mean discrepancy (MMD) (Smola et al., 2006) requires f H k ≤ 1 where H k denotes the RKHS with kernel k; the Dudley metric (Dudley, 2002) requires f BL ≤ 1 where f BL := f ∞ + ∇f 2 ; and Wasserstein distance requires ∇f 2 ≤ 1.

These additional requirements on the dual function might incur some extra difficulty in practice.

For example, with Wasserstein distance and the Dudley metric, we might need to include an extra gradient penalty (Gulrajani et al., 2017) , which requires additional computation to take the gradient through a gradient.

Meanwhile, the consistency of the surrogate loss under regularization is not clear.

For MMD, we can obtain a closed-form solution for the dual function, which saves the cost of the inner optimization (Gretton et al., 2012) , but with the tradeoff of requiring two independent samples in each outer optimization update.

Moreover, MMD relies on the condition that the dual function lies in some RKHS, which introduces additional kernel parameters to be tuned and in practice may not be sufficiently flexible compared to neural networks.

We have derived a consistent stationary distribution correction estimator in the form of a min-max saddle point optimization Equation 13.

Here, we present a practical instantiation of GenDICE with a concrete objective and parametrization.

We choose the χ 2 -divergence, which is an f -divergence with φ (x) = (x − 1) 2 and φ

There two major reasons for adopting χ 2 -divergence:

i) In the behavior-agnostic OPE problem, we mainly use the ratio correction function for estimating

, which is an expectation.

Recall that the error between the estimate and ground-truth can then be bounded by total variation, which is a lower bound of χ 2 -divergence.

ii) For the alternative divergences, the conjugate of the KL-divergence involves exp (·), which may lead to instability in optimization; while the IPM variants introduce extra constraints on dual function, which may be difficult to be optimized.

The conjugate function of χ 2 -divergence en-joys suitable numerical properties and provides squared regularization.

We have provided an additional empirical ablation study that investigates the alternative divergences in Appendix E.3.

To parameterize the correction ratio τ and dual function f we use neural networks, τ (s, a) = nn wτ (s, a) and f (s, a) = nn w f (s, a), where w τ and w f denotes the parameters of τ and f respectively.

Since the optimization requires τ to be non-negative, we add an extra positive neuron, such as exp (·), log (1 + exp (·)) or (·) 2 at the final layer of nn wτ (s, a).

We empirically compare the different positive neurons in Section 6.3.

For these representations, and unbiased gradient estimator ∇ (wτ ,u,w f ) J (τ, u, f ) can be obtained straightforwardly, as shown in Appendix B. This allows us to apply stochastic gradient descent to solve the saddle-point problem Equation 14 in a scalable manner, as illustrated in Algorithm 1.

We provide a theoretical analysis for the proposed GenDICE algorithm, following a similar learning setting and assumptions to (Nachum et al., 2019) .

The main result is summarized in the following theorem.

A formal statement, together with the proof, is given in Appendix C. Theorem 3 (Informal) Under mild conditions, with learnable F and H, the error in the objective between the GenDICE estimate,τ , to the solution τ

where E [·] is w.r.t.

the randomness in D and in the optimization algorithms, opt is the optimization error, and approx (F, H) is the approximation induced by (F, H) for parametrization of (τ, f ).

The theorem shows that the suboptimality of GenDICE's solution, measured in terms of the objective function value, can be decomposed into three terms: (1) the approximation error approx , which is controlled by the representation flexibility of function classes; (2) the estimation error due to sample randomness, which decays at the order of 1/ √ N ; and (3) the optimization error, which arises from the suboptimality of the solution found by the optimization algorithm.

As discussed in Appendix C, in special cases, this suboptimality can be bounded below by a divergence betweenτ and τ * , and therefore directly bounds the error in the estimated policy value.

There is also a tradeoff between these three error terms.

With more flexible function classes (e.g., neural networks) for F and H, the approximation error approx becomes smaller.

However, it may increase the estimation error (through the constant in front of 1/ √ N ) and the optimization error (by solving a harder optimization problem).

On the other hand, if F and H are linearly parameterized, estimation and optimization errors tend to be smaller and can often be upper-bounded explicitly in Appendix C.3.

However, the corresponding approximation error will be larger.

Off-policy evaluation with importance sampling (IS) has has been explored in the contextual bandits (Strehl et al., 2010; Dudík et al., 2011; Wang et al., 2017) , and episodic RL settings (Murphy et al., 2001; Precup et al., 2001) , achieving many empirical successes (e.g., Strehl et al., 2010; Dudík et al., 2011; Bottou et al., 2013) .

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

This data is used to recover a correction ratio function for a target policy π that is then used to estimate the average reward in two different settings: i) average reward; and ii) discounted reward.

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

We use a similar taxi domain as in Liu et al. (2018) , where a grid size of 5 × 5 yields 2000 states in total (25 × 16 × 5, corresponding to 25 taxi locations, 16 passenger appearance status and 5 taxi status).

We set the target policy to a final policy π after running tabular Q-learning for 1000 iterations, and set another policy π + after 950 iterations as the base policy.

The behavior policy is a mixture controlled by α as π b = (1 − α)π + απ + .

For the model-based method, we use a tabular representation for the reward and transition functions, whose entries are estimated from behavior data.

For IS and IPS, we fit a policy via behavior cloning to estimate the policy ratio.

In this specific setting, our methods achieve better results compared to IS, IPS and the model-based method.

Interestingly, with longer horizons, IS cannot improve as much as other methods even with more data, while GenDICE consistently improve and achieves much better results than the baselines.

DualDICE only works with γ < 1.

GenDICE is more stable than DualDICE when γ becomes larger (close to 1), while still showing competitive performance for smaller discount factors γ.

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

When α becomes larger, i.e., the behavior policies are closer to the target policy, all methods performs better, as expected.

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

When α = 0.33, i.e., the OPE tasks are relatively easier and GenDICE obtains better results at all learning rate settings.

However, when α = 0.0, i.e., the estimation becomes more difficult and only GenDICE only obtains reasonable results with the larger learning rate.

Generally, this ablation study shows that the proposed method is not sensitive to the learning rate, and is easy to train.

We further investigate the effects of the activation function on the last layer, which ensure the non-negative outputs required for the ratio.

To better understand which activation function will lead to stable trainig for the neural correction estimator, we empirically compare using i) (·) 2 ; ii) log(1 + exp(·)); and iii) exp(·).

In practice, we use the (·) 2 since it achieves low variance and better performance in most cases, as shown in Figure 5 .

We vary γ ∈ {0.95, 0.99, 0.995, 0.999, 1.0} to probe the sensitivity of GenDICE.

Specifically, we compare to DualDICE, and find that GenDICE is stable, while DualDICE becomes unstable when the γ becomes large, as shown in Figure 6 .

GenDICE is also more general than DualDICE, as it can be applied to both the average and discounted reward cases.

In Section 3, we highlighted the importance of the ratio constraint.

Here we investigate the trivial solution issue without the constraint.

The results in Figure 6 demonstrate the necessity of adding the constraint penalty, since a trivial solution prevents an accurate corrector from being recovered (green line in left two figures).

In this paper, we proposed a novel algorithm GenDICE for general stationary distribution correction estimation, which can handle both the discounted and average stationary distribution given multiple behavior-agnostic samples.

Empirical results on off-policy evaluation and offline PageRank show the superiority of proposed method over the existing state-of-the-art methods.

the existence of the stationary distribution.

Our discussion is all based on this assumption.

Assumption 1 Under the target policy, the resulted state-action transition operator T has a unique stationary distribution in terms of the divergence D (·||·).

If the total variation divergence is selected, the Assumption 1 requires the transition operator should be ergodic, as discussed in Meyn & Tweedie (2012).

Theorem 1 For arbitrary λ > 0, the solution to the optimization Eqn (11) is

Proof For γ ∈ (0, 1), there is not degenerate solutions to D T p γ,µ0 • τ ||p · τ .

The optimal solution is a density ratio.

Therefore, the extra penalty E p(x) [τ (x)] − 1 2 does not affect the optimality for ∀λ > 0.

negative, and the the density ratio

p(x) leads to zero for both terms.

Then, the density ratio is a solution to J (τ ).

For any other non-negative function τ (x) ≥ 0, if it is the optimal solution to J (τ ), then, we have

We denote µ (x) = p (x) τ (x), which is clearly a density function.

Then, the optimal conditions in Equation 15 imply

or equivalently, µ is the stationary distribution of T .

We have thus shown the optimal τ (x) =

is the target density ratio.

Proof Since the φ is convex, we consider the Fenchel dual representation of the f -divergence

2 is also convex, which concludes the proof.

Inputs: Convex function φ and its Fenchel conjugate φ

, initial state s 0 ∼ µ 0 , target policy π, distribution corrector nn wτ (·, ·), nn w f (·, ·), constraint scalar u, learning rates η τ , η f , η u , number of iterations K, batch size B.

Sample actions a

end for Return nn wτ .

We provide the unbiased gradient estimator for ∇ wτ ,u,w f J (τ, u, f ) in Eqn (14) below:

Then, we have the psuedo code which applies SGD for solving Eqn (14).

For convenience, we repeat here the notation defined in the main text.

The saddle-point reformulation of the objective function of GenDICE is:

To avoid the numerical infinity in D φ (·||·), we induced the bounded version as

is still a valid divergence, and therefore the optimal solution τ * is still the stationary density ratio

p(x) .

We denote the J (τ, µ, f ) as the empirical surrogate of

with optimal (f * , u * ), and

We apply some optimization algorithm for J (τ, u, f ) over space (H, F, R), leading to the output τ , u, f .

Under Assumption 2, we need only consider τ ∞ ≤ C, then, the corresponding dual u = E p (τ ) − 1 ⇒ u ∈ U := {|u| ≤ (C + 1)}. We choose the φ * (·) is a κ-Lipschitz continuous, then, the We consider the error betweenτ and τ * using standard arguments (Shalev-Shwartz & Ben-David, 2014; Bach, 2014)

Remark: In some special cases, the suboptimality also implies the distance betweenτ and τ * .

Specifically,for γ = 1, if the transition operator P π can be represented as P π = QΛQ −1 where Q denotes the (countable) eigenfunctions and Λ denotes the diagonal matrix with eigenvalues, the largest of which is 1.

We consider φ (·) as identity and f ∈ F := span (Q) , f p,2 ≤ 1 , then the d (τ, τ * ) will bounded from below by a metric between τ and τ * .

Particularly, we have

Rewrite τ = ατ * + ζ, where ζ ∈ span Q \τ * , then

Recall the optimality of τ

We start with the following error decomposition:

• For 1 , we have

.

We consider the terms one-by-one.

By definition, we have

which is induced by introducing F for dual approximation.

For the third term

where we define est :

Therefore, we can now bound 1 as

We consider the terms from right to left.

For the term J (τ * H ) − J (τ * ), we have

, which is induced by restricting the function space to H. The second term is nonpositive, due to the optimality of (u * , f * ).

The final inequality comes from the fact that

where the second term is nonpositive, thanks to the optimality ofτ * H .

Finally, for the term J (τ * H ) − J (τ * H ), using the same argument in Equation 21, we have

Therefore, we can bound 2 by 2 ≤ C φ,C,λ approx (H) + C P π ,κ,λ (F) + 2 est .

In sum, we have d (τ , τ * ) ≤ 4 est +ˆ opt + 2C P π ,κ,λ approx (F) + C φ,C,λ approx (H) .

In the following sections, we will bound the est andˆ opt .

In this section, we analyze the statistical error

We mainly focus on the batch RL setting with

, which has been studied by previous authors (e.g., Sutton et al., 2012; Nachum et al., 2019) .

However, as discussed in the literature (Antos et al., 2008; Lazaric et al., 2012; Dai et al., 2018; Nachum et al., 2019) , using the blocking technique of Yu (1994) , the statistical error provided here can be generalized to β-mixing samples in a single sample path.

We omit this generalization for the sake of expositional simplicity.

To bound the est , we follow similar arguments by Dai et al. (2018) ; Nachum et al. (2019) via the covering number.

For completeness, the definition is given below.

The Pollard's tail inequality bounds the maximum deviation via the covering number of a function class:

are i.i.d.

samples from some distribution.

Then, for any given > 0,

The covering number can then be bounded in terms of the function class's pseudo-dimension:

Lemma 5 (Haussler (1995), Corollary 3) For any set X , any points x 1:N ∈ X N , any class F of functions on X taking values in [0, M ] with pseudo-dimension D F < ∞, and any > 0,

The statistical error est can be bounded using these lemmas.

Lemma 6 (Stochastic error) Under the Assumption 2, if φ * is κ-Lipschitz continuous and the psuedo-dimension of H and F are finite, with probability at least 1 − δ, we have

Proof The proof works by verifying the conditions in Lemma 4 and computing the covering number.

2 u 2 , we will apply Lemma 4 with Z = Ω × Ω, Z i = (x i , x i ), and G = h H×F ×U .

We check the boundedness of h ζ,u,f (x, x ).

Based on Assumption 2, we only consider the τ ∈ H and u ∈ U bounded by C and C + 1.

We also rectify the f ∞ ≤ C. Then, we can bound the h ∞ :

where C φ = max t∈[−C,C] −φ * (t).

Thus, by Lemma 4, we have

Next, we check the covering number of G. Firstly, we bound the distance in G,

Denote the pseudo-dimension of H and F as D H and D F , respectively, we have

, we obtain the bound for the statistical error:

In this section, we investigate the optimization error

.

Notice our estimator min τ ∈H max f ∈F ,u∈U J (τ, u, f ) is compatible with different parametrizations for (H, F) and different optimization algorithms, the optimization error will be different.

For the general neural network for (τ, f ), although there are several progress recently (Lin et al., 2018; Jin et al., 2019; Lin et al., 2019) about the convergence to a stationary point or local minimum, it remains a largely open problem to quantify the optimization error, which is out of the scope of this paper.

Here, we mainly discuss the convergence rate with tabular, linear and kernel parametrization for (τ, f ).

Particularly, we consider the linear parametrization particularly, i.e., τ (x) = σ w τ ψ (x) , f (x) = w f ψ (x), and σ (·) : R → R + is convex.

There are many choices of the σ (·), e.g., exp (·), log (1 + exp (·)) and (·) 2 .

Obviously, even with such nonlinear mapping, the J (τ, u, f ) is still convex-concave w.r.t (w τ , w f , u) by the convex composition rule.

We can bound theˆ opt by the primal-dual gap gap :

With vanilla SGD, we have

, where T is the optimization steps (Nemirovski et al., 2009) .

, where the E [·] is taken w.r.t.

randomness in SGD.

We are now ready to state the main theorm in a precise way:

Under Assumptions 2 and 1 , the stationary distribution µ exists, i.e.,

* , and the psuedo-dimension of H and F are finite, the error between the GenDICE estimate to τ

where E [·] is w.r.t.

the randomness in sample D and in the optimization algorithms.

opt is the optimization error, and approx (F, H) is the approximation induced by (F, H) for parametrization of (τ, f ).

Proof We have the total error as

where approx := 2C T ,κ,λ approx (F) + C φ,C,λ approx (H).

For opt , we can apply the results for SGD in Appendix C.3.

We can bound the E [ est ] by Lemma 6.

Specifically, we have

Plug all these bounds into Equation 25, we achieve the conclusion.

For the Taxi domain, we follow the same protocol as used in Liu et al. (2018) .

The behavior and target policies are also taken from Liu et al. (2018) (referred in their work as the behavior policy for α = 0).

We use a similar taxi domain, where a grid size of 5×5 yields 2000 states in total (25×16×5, cor-responding to 25 taxi locations, 16 passenger appearance status and 5 taxi status).

We set our target policy as the final policy π * after running Q-learning (Sutton & Barto, 1998) for 1000 iterations, and set another policy π + after 950 iterations as our base policy.

The behavior policy is a mixture policy controlled by α as π = (1 − α)π * + απ + , i.e., the larger α is, the behavior policy is more close to the target policy.

In this setting, we solve for the optimal stationary ratio τ exactly using matrix operations.

Since Liu et al. (2018) perform a similar exact solve for |S| variables µ(s), for better comparison we also perform our exact solve with respect to |S| variables τ (s).

Specifically, the final objective of importance sampling will require knowledge of the importance weights µ(a|s)/p(a|s).

For offline PageRank, the graph statistics are illustrated in Table 1 , and the degree statistics and graph visualization are shown in Figure 7 .

For the BarabasiAlbert (BA) Graph, it begins with an initial connected network of m 0 nodes in the network.

Each new node is connected to m ≤ m 0 existing nodes with a probability that is proportional to the number of links that the existing nodes already have.

Intuitively, heavily linked nodes ('hubs') tend to quickly accumulate even more links, while nodes with only a few links are unlikely to be chosen as the destination for a new link.

The new nodes have a 'preference' to attach themselves to the already heavily linked nodes.

For two real-world graphs, it is built upon the real-world citation networks.

In our experiments, the weights of the BA graph is randomly drawn from a standard Gaussian distribution with normalization to ensure the property of the transition matrix.

The offline data is collected by a random walker on the graph, which consists the initial state and next state in a single trajectory.

In experiments, we vary the number of off-policy samples to validate the effectiveness of GenDICE with limited offline samples provided.

We use the Cartpole, Reacher and HalfCheetah tasks as given by OpenAI Gym.

In importance sampling, we learn a neural network policy via behavior cloning, and use its probabilities for computing importance weights π * (a|s)/π(a|s).

All neural networks are feed-forward with two hidden layers of dimension 64 and tanh activations.

Discrete Control Tasks We modify the Cartpole task to be infinite horizon: We use the same dynamics as in the original task but change the reward to be −1 if the original task returns a termination (when the pole falls below some threshold) and 1 otherwise.

We train a policy on this task with standard Deep Q-Learning (Mnih et al., 2013) until convergence.

We then define the target policy π * as a weighted combination of this pre-trained policy (weight 0.7) and a uniformly random policy (weight 0.3).

The behavior policy π for a specific 0 ≤ α ≤ 1 is taken to be a weighted combination of the pre-trained policy (weight 0.55 + 0.15α) and a uniformly random policy (weight 0.45 − 0.15α).

We train each stationary distribution correction estimation method using the Adam optimizer with batches of size 2048 and learning rates chosen using a hyperparameter search from {0.0001, 0.0003, 0.001, 0.003} and choose the best one as 0.0003.

For the Reacher task, we train a deterministic policy until convergence via DDPG (Lillicrap et al., 2015) .

We define the target policy π as a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.1.

The behavior policy π b for a specific 0 ≤ α ≤ 1 is taken to be a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.4 − 0.3α.

We train each stationary distribution correction estimation method using the Adam optimizer with batches of size 2048 and learning rates chosen using a hyperparameter search from {0.0001, 0.0003, 0.001, 0.003} and the optimal learning rate found was 0.003).

For the HalfCheetah task, we also train a deterministic policy until convergence via DDPG (Lillicrap et al., 2015) .

We define the target policy π as a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.1.

The behavior policy π b for a specific 0 ≤ α ≤ 1 is taken to be a Gaussian with mean given by the pre-trained policy and standard deviation given by 0.2 − 0.1α.

We train each stationary distribution correction estimation method using the Adam optimizer with batches of size 2048 and learning rates chosen using a hyperparameter search from {0.0001, 0.0003, 0.001, 0.003} and the optimal learning rate found was 0.003.

E.1 OPE FOR DISCRETE CONTROL On the discrete control task, we modify the Cartpole task to be infinite horizon: the original dynamics is used but with a modified reward function: the agent will receive −1 if the environment returns a termination (i.e., the pole falls below some threshold) and 1 otherwise.

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

When α = 0.33, i.e., the OPE tasks are relatively easier, GenDICE gets relatively good results in all learning rate settings.

However, when α = 0.0, i.e., the estimation becomes more difficult, only GenDICE in larger learning rate gets reasonable estimation.

Interestingly, we can see with larger learning rates, the performance becomes better, and when learning rate is 0.001 with α = 0.0, the variance is very high, showing some cases the estimation becomes more accurate.

The right three figures show different activation functions with different behavior policy.

The square and softplus function works well; while the exponential function shows poor performance under some settings.

In practice, we use the square function since its low variance and better performance in most cases.

Model-Based Importance Sampling DualDICE GenDICE (ours) Oracle Figure 11 : Results on HalfCheetah.

Each plot in the first row shows the estimated average step reward over training and different behavior policies (higher α corresponds to a behavior policy closer to the target policy.

Although any valid divergence between p · τ and T p γ,µ0 • τ in our estimator is consistent, which will always lead to the stationary distribution correction ratio asymptotically, and any λ > 0 will guarantee the normalization constraint, i.e., E p [τ ] = 1, as we discussed in main text, different Figure 12 : Results of ablation study with different learning rates and activation functions.

The plots show the estimated average step reward over training and different behavior policies .

choices of the divergences and λ may incur difficulty in the numerical optimization procedure.

In this section, we investigate the empirical effects of the choice of f -divergence and IPM, and the weight of constrant regularization λ.

To avoid the effects of other factors in the estimator, e.g., function parametrization, we focus on the offline PageRank task on BA graph with 100 nodes and 10k offline samples.

All the performances are evaluated with 20 random trials.

We test the GenDICE with several other alternative divergences, e.g., Wasserstein-1 distance, Jensen-Shannon divergence, KL-divergence, Hellinger divergence, and MMD.

To ensure the dual function to be 1-Lipchitz, we add the gradient penalty.

We use a learned Gaussian kernel in MMD, similar to Li et al. (2017) .

As we can see in Figure 13 (a), with these different divergences, the proposed GenDICE estimator can always achieve significantly better performance compared with the model-based estimator, showing that the GenDICE estimator is compatible with many different divergences.

Most of the divergences, with appropriate extra techniques to handle the difficulties in optimization and carefully tuning for extra parameters, can achieve similar performances, consistent with phenomena in the variants of GANs (Lucic et al., 2018) .

However, KL-divergence is an outlier, performing noticeably worse, which might be caused by the ill-behaved exp (·) in its conjugate function.

The χ 2 -divergence and JS-divergence are better, which achieve good performances with fewer parameters to be tuned.

The effect of the penalty weight λ is illustrated the in Figure 13(b) .

We vary the λ ∈ [0.1, 5] with χ 2 -divergence in the GenDICE estimator.

Within a large range of λ, the performances of the proposed GenDICE are quite consistent, which justifies Theorem 1.

The penalty multiplies with λ.

Therefore, with λ increases, the variance of the stochastic gradient estimator also increases, which explains the variance increasing in large λ in Figure 13(b) .

In practice, λ = 1 is a reasonable choice for general cases.

Markov Chain Monte Carlo Classical MCMC (Brooks et al., 2011; Gelman et al., 2013) aims at sampling from µ π by iteratively simulting from the transition operator.

It requires continuous interaction with the transition operator and heavy computational cost to update many particles.

Amor-tized SVGD (Wang & Liu, 2016) and Adversarial MCMC (Song et al., 2017; Li et al., 2019) alleviate this issue via combining with neural network, but they still interact with the transition operator directly, i.e., in an on-policy setting.

The major difference of our GenDICE is the learning setting: we only access the off-policy dataset, and cannot sample from the transition operator.

The proposed GenDICE leverages stationary density ratio estimation for approximating the stationary quantities, which distinct it from classical methods.

Density ratio estimation is a fundamental tool in machine learning and much related work exists.

Classical density ratio estimation includes moment matching (Gretton et al.) , probabilistic classification (Bickel et al., 2007) , and ratio matching (Nguyen et al., 2008; Sugiyama et al., 2008; Kanamori et al., 2009) .

These classical methods focus on estimating the ratio between two distributions with samples from both of them, while GenDICE estimates the density ratio to a stationary distribution of a transition operator, from which even one sample is difficult to obtain.

PageRank Yao & Schuurmans (2013) developed a reverse-time RL framework for PageRank via solving a reverse Bellman equation, which is less sensitive to graph topology and shows faster adaptation with graph change.

However, Yao & Schuurmans (2013) still considers the online manner, which is different with our OPR setting.

@highlight

In this paper, we proposed a novel algorithm, GenDICE, for general stationary distribution correction estimation, which can handle both discounted and average off-policy evaluation on multiple behavior-agnostic samples.