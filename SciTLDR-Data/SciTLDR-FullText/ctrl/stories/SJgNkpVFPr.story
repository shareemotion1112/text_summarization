The goal of imitation learning (IL) is to learn a good policy from high-quality demonstrations.

However, the quality of demonstrations in reality can be diverse, since it is easier and cheaper to collect demonstrations from a mix of experts and amateurs.

IL in such situations can be challenging, especially when the level of demonstrators' expertise is unknown.

We propose a new IL paradigm called Variational Imitation Learning with Diverse-quality demonstrations (VILD), where we explicitly model the level of demonstrators' expertise with a probabilistic graphical model and estimate it along with a reward function.

We show that a naive estimation approach is not suitable to large state and action spaces, and fix this issue by using a variational approach that can be easily implemented using existing reinforcement learning methods.

Experiments on continuous-control benchmarks demonstrate that VILD outperforms state-of-the-art methods.

Our work enables scalable and data-efficient IL under more realistic settings than before.

The goal of sequential decision making is to learn a policy that makes good decisions (Puterman, 1994) .

As an important branch of sequential decision making, imitation learning (IL) (Schaal, 1999) aims to learn such a policy from demonstrations (i.e., sequences of decisions) collected from experts.

However, high-quality demonstrations can be difficult to obtain in reality, since such experts may not always be available and sometimes are too costly (Osa et al., 2018) .

This is especially true when the quality of decisions depends on specific domain-knowledge not typically available to amateurs; e.g., in applications such as robot control (Osa et al., 2018) and autonomous driving (Silver et al., 2012) .

In practice, demonstrations are often diverse in quality, since it is cheaper to collect demonstrations from mixed demonstrators, containing both experts and amateurs (Audiffren et al., 2015) .

Unfortunately, IL in such settings tends to perform poorly, since low-quality demonstrations often negatively affect the performance of IL methods (Shiarlis et al., 2016) .

For example, amateurs' demonstrations for robotics can be cheaply collected via a robot simulation (Mandlekar et al., 2018 ), but such demonstrations may cause damages to the robot which is catastrophic in the real-world (Shiarlis et al., 2016) .

Similarly, demonstrations for autonomous driving can be collected from drivers in public roads (Fridman et al., 2017) , which may contain traffic-accident demonstrations.

Learning a self-driving car from these low-quality demonstrations may cause traffic accidents.

When the level of demonstrators' expertise is known, multi-modal IL (MM-IL) can be used to learn a good policy with diverse-quality demonstrations Hausman et al., 2017; Wang et al., 2017) .

Specifically, MM-IL aims to learn a multi-modal policy, where each mode of the policy represents the decision making of each demonstrator.

When knowing the level of demonstrators' expertise, good policies can be obtained by selecting modes that correspond to the decision making of high-expertise demonstrators.

However, in practice, it is difficult to truly determine the level of demonstrators' expertise beforehand.

Without knowing the level of expertise, it is difficult to distinguish the decision making of experts and amateurs, and learning a good policy is challenging.

To overcome the issue of MM-IL, pioneer works have proposed to estimate the quality of each demonstration using auxiliary information from experts (Audiffren et al., 2015; Wu et al., 2019; Brown et al., 2019) .

Specifically, Audiffren et al. (2015) inferred the demonstration quality using similarities between diverse-quality demonstrations and high-quality demonstrations, where the latter are collected in a small number from experts.

In contrast, Wu et al. (2019) proposed to estimate the demonstration quality using a small number of demonstrations with confidence scores.

Namely, the score value given by an expert is proportion to the demonstration quality.

Similarly, the demonstration quality can be estimated by ranked demonstrations, where ranking from an expert is evaluated due to the relative quality (Brown et al., 2019) .

To sum up, these methods rely on auxiliary information from experts, namely high-quality demonstrations, confidence scores, and ranking.

In practice, these pieces of information can be scarce or noisy, which leads to a poor performance of these methods.

In this paper, we consider a novel but realistic setting of IL where only diverse-quality demonstrations are available.

Meanwhile, the level of demonstrators' expertise and auxiliary information from experts are fully absent.

To tackle this challenging setting, we propose a new learning paradigm called variational imitation learning with diverse-quality demonstrations (VILD).

The central idea of VILD is to model the level of demonstrators' expertise via a probabilistic graphical model, and learn it along with a reward function that represents an intention of expert's decision making.

To scale up our model for large state and action spaces, we leverage the variational approach (Jordan et al., 1999) , which can be implemented using reinforcement learning (RL) for flexibility (Sutton & Barto, 1998) .

To further improve data-efficiency of VILD when learning the reward function, we utilize importance sampling (IS) to re-weight a sampling distribution according to the estimated level of demonstrators' expertise.

Experiments on continuous-control benchmarks and real-world crowdsourced demonstrations (Mandlekar et al., 2018) denote that: 1) VILD is robust against diverse-quality demonstrations and outperforms existing methods significantly.

2) VILD with IS is data-efficient, since it learns the policy using a less number of transition samples.

Before delving into our main contribution, we first give the minimum background about RL and IL.

Then, we formulate a new setting in IL called diverse-quality demonstrations, discuss its challenge, and reveal the deficiency of existing methods.

Reinforcement learning.

Reinforcement learning (RL) (Sutton & Barto, 1998) aims to learn an optimal policy of a sequential decision making problem, which is often mathematically formulated as a Markov decision process (MDP) (Puterman, 1994) .

We consider a finite-horizon MDP with continuous state and action spaces defined by a tuple M " pS, A, pps 1 |s, aq, p 1 ps 1 q, rps, aqq with a state s t P S Ď R ds , an action a t P A Ď R da , an initial state density p 1 ps 1 q, a transition probability density pps t`1 |s t , a t q, and a reward function r : SˆA Þ Ñ R, where the subscript t P t1, . . .

, T u denotes the time step.

A sequence of states and actions, ps 1:T , a 1:T q, is called a trajectory.

A decision making of an agent is determined by a policy πpa t |s t q, which is a conditional probability density of action given state.

RL seeks for an optimal policy π ‹ pa t |s t q which maximizes the expected cumulative reward: E pπps 1:T ,a 1:T q rΣ T t"1 rps t , a t qs, where p π ps 1:T , a 1:T q " p 1 ps 1 qΠ T t"1 pps t`1 |s t , a t qπpa t |s t q is a trajectory probability density induced by π.

RL has shown great successes recently, especially when combined with deep neural networks (Silver et al., 2017 ).

However, a major limitation of RL is that it relies on the reward function which may be unavailable in practice (Schaal, 1999) .

Imitation learning.

To address the above limitation of RL, imitation learning (IL) was proposed (Schaal, 1999) .

Without using the reward function, IL aims to learn the optimal policy from demonstrations that encode information about the optimal policy.

A common assumption in most IL methods is that, demonstrations are collected by K ě 1 demonstrators who execute actions a t drawn from π ‹ pa t |s t q for every states s t .

A graphical model describing this data collection process is depicted in Figure 1 (a), where a random variable k P t1, . . .

, Ku denotes each demonstrator's identification number and ppkq denotes the probability of collecting a demonstration from the k-th demonstrator.

Under this assumption, demonstrations tps 1:T , a 1:T , kq n u N n"1 (i.e., observed random variables in Figure 1 (a)) are called expert demonstrations and are regarded to be drawn independently from a probability density p ‹ ps 1:T , a 1:T qppkq " ppkqp 1 ps 1 qΠ T t"1 pps t`1 |s t , a t qπ ‹ pa t |s t q. We note that k does not affect the trajectory density p ‹ ps 1:T , a 1:T q and can be omitted.

We assume a common assumption that p 1 ps 1 q and pps t`1 |s t , a t q are unknown but we can sample states from them.

IL has shown great successes in benchmark settings (Ho & Ermon, 2016; Fu et al., 2018; Peng et al., 2019) .

However, practical applications of IL in the real-world is relatively few (Schroecker et al., 2019) .

One of the main reasons is that most IL methods aim to learn with expert demonstrations.

In practice, such demonstrations are often too costly to obtain due to a limited number of experts, and

Figure 1: Graphical models describe expert demonstrations and diverse-quality demonstrations.

Shaded and unshaded nodes indicate observed and unobserved random variables, respectively.

Plate notations indicate that the sampling process is repeated for N times.

s t P S is a state with transition densities pps t`1 |s t , a t q, a t P A is an action with density π ‹ pa t |s t q, u t P A is a noisy action with density ppu t |s t , a t , kq, and k P t1, . . .

, Ku is an identification number with distribution ppkq.

even when we obtain them, the number of demonstrations is often too few to accurately learn the optimal policy (Audiffren et al., 2015; Wu et al., 2019; Brown et al., 2019) .

New setting in IL: Diverse-quality demonstrations.

To improve practicality of IL, we consider a new learning paradigm called IL with diverse-quality demonstrations, where demonstrations are collected from demonstrators with different level of expertise.

Compared to expert demonstrations, diverse-quality demonstrations can be collected more cheaply, e.g., via crowdsourcing (Mandlekar et al., 2018 ).

The graphical model in Figure 1 (b) depicts the process of collecting such demonstrations from K ą 1 demonstrators.

Formally, we select the k-th demonstrator according to a distribution ppkq.

After selecting k, for each time step t, the k-th demonstrator observes state s t and samples action a t using π ‹ pa t |s t q. However, the demonstrator may not execute a t in the MDP if this demonstrator is not expertised.

Instead, he/she may sample an action u t P A with another probability density ppu t |s t , a t , kq and execute it.

Then, the next state s t`1 is observed with a probability density pps t`1 |s t , u t q, and the demonstrator continues making decision until time step T .

We repeat this process for N times to collect diverse-quality demonstrations D d " tps 1:T , u 1:T , kq n u N n"1 .

These demonstrations are regarded to be drawn independently from a probability density

We refer to ppu t |s t , a t , kq as a noisy policy of the k-th demonstrator, since it is used to execute a noisy action u t .

Our goal is to learn the optimal policy π ‹ using diverse-quality demonstrations D d .

Note that Eq. (1) can be described equivalently by using a marginal density πpu t |s t , kq " ş A π ‹ pa t |s t qppu t |s t , a t , kqda t and removing a t from the graphical model.

However, we explicitly write a t as above to emphasize the dependency between π ‹ pa t |s t q and ppu t |s t , a t , kq.

This emphasis will be made more clear in Section 3.1 when we describe our choice of model.

The deficiency of existing methods.

We conjecture that existing IL methods are not suitable to learn with diverse-quality demonstrations according to p d .

Specifically, these methods always treat observed demonstrations as if they were drawn from p ‹ .

By comparing p ‹ and p d , we can see that existing methods would learn πpu t |s t q such that πpu t |s t q « Σ

‹ pa t |s t qppu t |s t , a t , kqda t .

In other words, they learn a policy that averages over decisions of all demonstrators.

This would be problematic when amateurs are present, as averaged decisions of all demonstrators would be highly different from those of all experts.

Worse yet, state distributions of amateurs and experts tend to be highly different, which often leads to the unstable learning: The learned policy oscillated between well-performed policy and poorly-performed policy.

For these reasons, we believe that existing methods tend to learn a policy that achieves average performances, and are not suitable for handling the setting of diverse-quality demonstrations.

This section presents VILD, namely a robust method for tackling the challenge from diverse-quality demonstrations.

Specifically, we build a probabilistic model that explicitly describes the level of demonstrators' expertise and a reward function (Section 3.1), and estimate its parameters by a variational approach (Section 3.2), which can be implemented easily by RL (Section 3.3).

We also improve data-efficiency by using importance sampling (Section 3.4).

Mathematical derivations are provided in Appendix A.

This section describes a model which enables estimating the level of demonstrators' expertise.

We first describe a naive model, whose parameters can be estimated trivially via supervised learning, but suffers from the issue of compounding error.

Then, we describe our proposed model, which avoids the issue of the naive model by learning a reward function.

Naive model.

Based on p d , one of the simplest models to handle diverse-quality demonstrations is p θ,ω ps 1:T , u 1:T , kq " ppkqpps 1 qΠ T t"1 pps t`1 |s t , u t q ş A π θ pa t |s t qp ω pu t |s t , a t , kqda t , where π θ and p ω are learned to estimate the optimal policy and the noisy policy, respectively.

The parameters θ and ω can be learned by minimizing the Kullback-Leibler (KL) divergence from the data distribution to the model.

This naive model can be regarded as an extension of a model proposed by Raykar et al. (2010) for handling diverse-quality data in supervised learning.

The main advantage of this naive model is that its parameters can be estimated trivially via supervised learning.

However, this native model suffers from the issue of compounding error (Ross & Bagnell, 2010) and tends to perform poorly.

Specifically, supervised-learning methods assume that data distributions during training and testing are identical.

However, data distributions during training and testing are different in IL, since data distributions depend on policies (Puterman, 1994) .

A discrepancy of data distributions causes compounding errors during testing, where prediction errors increase further in future predictions.

Due to this issue, supervised-learning methods often perform poorly in IL (Ross & Bagnell, 2010) .

The issue becomes even worse with diverse-quality demonstrations, since data distributions of different demonstrators tend to be highly different.

For these reasons, this naive model is not suitable for our setting.

Proposed model.

To avoid the issue of compounding error, our method utilizes the inverse RL (IRL) approach (Ng & Russell, 2000) , where we aim to learn a reward function from diverse-quality demonstrations 1 .

IL problems can be solved by a combination of IRL and RL, where we learn a reward function by IRL and then learn a policy from the reward function by RL.

This combination avoids the issue of compounding error, since the policy is learned by RL which generalizes to states not presented in demonstrations (Ho & Ermon, 2016) .

Specifically, our proposed model is based on a model of maximum entropy IRL (MaxEnt-IRL) (Ziebart et al., 2010) .

Briefly speaking, MaxEnt-IRL learns a reward function from expert demonstrations by using a model p φ ps 1:T , a 1:T q 9 pps 1 qΠ T t"1 p 1 ps t`1 |s t , a t q exppr φ ps t , a t qq.

Based on this model, we propose to learn the reward function and the level of expertise by a model p φ,ω ps 1:T , u 1:T , kq 9 ppkqp 1 ps 1 q

where φ and ω are parameters.

We denote a normalization term of this model by Z φ,ω .

By comparing the proposed model p φ,ω to the data distribution p d , the reward parameter φ should be learned so that the cumulative reward is proportion to a joint probability density of actions given by the optimal policy, i.e., exppΣ T t"1 r φ ps t , a t qq 9 Π T t"1 π ‹ pa t |s t q. In other words, the cumulative reward is large for trajectories induced by the optimal policy.

Therefore, the optimal policy can be learned by maximizing the cumulative reward.

Meanwhile, the density p ω pu t |s t , a t , kq is learned to estimate the noisy policy ppu t |s t , a t , kq.

In the remainder, we refer to ω as an expertise parameter.

To learn parameters of this model, we propose to minimize the KL divergence from the data distribution to the model: min φ,ω KLpp d ps 1:T , u 1:T |kqppkq||p φ,ω ps 1:T , u 1:T , kqq.

By rearranging terms and ignoring constant terms, minimizing this KL divergence is equivalent to solving an optimization problem max φ,ω f pφ, ωq´gpφ, ωq, where f pφ, ωq " E p d ps 1:T ,u 1:T |kqppkq rΣ T t"1 logp ş A exppr φ ps t , a t qqp ω pu t |s t , a t , kqda t qs and gpφ, ωq " log Z φ,ω .

To solve this optimization, we need to compute the integrals over both state space S and action space A. Computing these integrals is feasible for small state and action spaces, but is infeasible for large state and action spaces.

To scale up our model to MDPs with large state and action spaces, we leverage a variational approach in the followings.

The central idea of the variational approach is to lower-bound an integral by the Jensen inequality and a variational distribution (Jordan et al., 1999) .

The main benefit of the variational approach is that the integral can be indirectly computed via the lower-bound, given an optimal variational distribution.

However, finding the optimal distribution often requires solving a sub-optimization problem.

Before we proceed, notice that f pφ, ωq´gpφ, ωq is not a joint concave function of the integrals, and this prohibits using the Jensen inequality.

However, we can separately lower-bound f and g by the Jensen inequality, since they are concave functions of their corresponding integrals.

Specifically, let l φ,ω ps t , a t , u t , kq " r φ ps t , a t q`log p ω pu t |s t , a t , kq.

By using a variational distribution q ψ pa t |s t , u t , kq with parameter ψ, we obtain an inequality f pφ, ωq ě Fpφ, ω, ψq, where

and H t pq ψ q "´E q ψ pat|st,ut,kq rlog q ψ pa t |s t , u t , kqs.

It is trivial to verify that the equality f pφ, ωq " max ψ Fpφ, ω, ψq holds (Jordan et al., 1999) , where the maximizer ψ ‹ of the lowerbound yields q ψ ‹ pa t |s t , u t , kq 9 exppl φ,ω ps t , a t , u t , kqq.

Therefore, the function f pφ, ωq can be substituted by max ψ Fpφ, ω, ψq.

Meanwhile, by using a variational distribution q θ pa t , u t |s t , kq with parameter θ, we obtain an inequality gpφ, ωq ě Gpφ, ω, θq, where

and r q θ ps 1:T , u 1:T , a 1:T , kq " ppkqp 1 ps 1 qΠ T t"1 pps t`1 |s t , u t qq θ pa t , u t |s t , kq.

The lower-bound G resembles an objective function of maximum entropy RL (MaxEnt-RL) (Ziebart et al., 2010) .

By using the optimality results of MaxEnt-RL (Haarnoja et al., 2018) , we have an equality gpφ, ωq " max θ Gpφ, ω, θq.

Therefore, the function gpφ, ωq can be substituted by max θ Gpφ, ω, θq.

By using these lower-bounds, we have that max φ,ω f pφ, ωq´gpφ, ωq " max φ,ω,ψ Fpφ, ω, ψqḿ ax θ Gpφ, ω, θq " max φ,ω,ψ min θ Fpφ, ω, ψq´Gpφ, ω, θq.

Solving the max-min problem is often feasible even for large state and action spaces, since Fpφ, ω, ψq and Gpφ, ω, θq are defined as an expectation and can be optimized straightforwardly.

Nevertheless, in practice, we represent the variational distributions by parameterized functions, and iteratively solve the sub-optimization (w.r.t.

ψ and θ) by stochastic optimization methods.

However, in this scenario, the equalities f pφ, ωq " max ψ Fpφ, ω, ψq and gpφ, ωq " max θ Gpφ, ω, θq may not hold for two reasons.

First, the optimal variational distributions may not be in the space of our parameterized functions.

Second, stochastic optimization methods may yield local solutions.

Nonetheless, when the variational distributions are represented by deep neural networks, the obtained variational distributions are often reasonably accurate and the equalities approximately hold (Ranganath et al., 2014) .

In practice, we are required to specify models for q θ and p ω .

We propose to use q θ pa t , u t |s t , kq " q θ pa t |s t qN pu t |a t , Σq and p ω pu t |s t , a t , kq " N pu t |a t , C ω pkqq.

As shown below, the choice for q θ pa t , u t |s t , kq enables us to solve the sub-optimization w.r.t.

θ by using RL with reward function r φ .

Meanwhile, the choice for p ω pu t |s t , a t , kq incorporates our prior assumption that the noisy policy tends to Gaussian, which is a reasonable assumption for actual human motor behavior (van Beers et al., 2004) .

Under these model specifications, solving max φ,ω,ψ min θ Fpφ, ω, ψq´Gpφ, ω, θq is equivalent to solving max φ,ω,ψ min θ Hpφ, ω, ψ, θq, where

Here, r q θ ps 1:T , a 1:T q " p 1 ps 1 qΠ T t"1 ş A pps t`1 |s t , u t qN pu t |a t , Σqdu t q θ pa t |s t q is a noisy trajectory density induced by a policy q θ pa t |s t q, where N pu t |a t , Σq can be regarded as an approximation of the noisy policy in Figure 1(b) .

Minimizing H w.r.t.

θ resembles solving a MaxEnt-RL problem with a reward function r φ ps t , a t q, except that trajectories are collected according to the noisy trajectory density.

In other words, this minimization problem can be solved using RL, and q θ pa t |s t q can be regarded as an approximation of the optimal policy.

The hyper-parameter Σ determines the quality of this approximation: smaller value of Σ gives a better approximation.

Therefore, by choosing a reasonably small value of Σ, solving the max-min problem in Eq. (5) yields a reward function r φ ps t , a t q and a policy q θ pa t |s t q. This policy imitates the optimal policy, which is the goal of IL.

The model specification for p ω incorporates our prior assumption about the noisy policy.

Namely, p ω pu t |s t , a t , kq " N pu t |a t , C ω pkqq assumes that the noisy policy tends to Gaussian, where C ω pkq gives an estimated expertise of the k-th demonstrator: High-expertise demonstrators have small C ω pkq and vice-versa for low-expertise demonstrators.

Note that VILD is not restricted to this choice.

Different choices of p ω incorporate different prior assumptions.

For example, a Laplace distribution incorporates a prior assumption about demonstrators who tend to execute outlier actions (Murphy, 2013) .

In such a case, the squared error in H is replaced by the absolute error (see Appendix A.3).

It should be mentioned that q ψ pa t |s t , u t , kq maximizes the immediate reward and minimizes the weighted squared error between u t and a t .

The trade-off between the reward and squared-error is determined by C ω pkq.

Specifically, for demonstrators with a small C ω pkq (i.e., high-expertise demonstrators), the squared error has a large magnitude and q ψ tends to minimize the squared error.

Meanwhile, for demonstrators with a large value of C ω pkq (i.e., low-expertise demonstrators), the squared error has a small magnitude and q ψ tends to maximize the immediate reward.

We implement VILD with deep neural networks where we iteratively update φ, ω, and ψ by stochastic gradient methods, and update θ by policy gradient methods.

A pseudo-code of VILD and implementation details are given in Appendix B. In our implementation, we include a regularization term Lpωq " T E ppkq rlog |C´1 ω pkq|s{2, to penalize large value of C ω pkq.

Without this regularization, C ω pkq can be overly large which makes learning degenerate.

We note that H already includes such a penalty via the trace term: E ppkq rTrpC´1 ω pkqΣqs.

However, the strength of this penalty tends to be too small, since we choose Σ to be small.

VILD requires variable k to be given along with demonstrations.

However, There is no need for this variable to be provided by experts.

When k is not given, a simple strategy is to set k " n and K " N .

In other words, this strategy assumes that there is a one-to-one mapping between demonstration and demonstrator.

We apply this strategy in our experiments with real-world demonstrations.

To improve the convergence rate of VILD when updating φ, we use importance sampling (IS).

Specifically, by analyzing the gradient ∇ φ H " ∇ φ tE p d ps 1:T ,u 1:T |kqppkq rΣ T t"1 E q ψ pat|st,ut,kq rr φ ps t , a t qssÉ r q θ ps 1:T ,a 1:T q rΣ T t"1 r φ ps t , a t qsu, we can see that the reward function is updated to maximize the expected cumulative reward obtained by demonstrators and q ψ , while minimizing the expected cumulative reward obtained by q θ .

However, low-quality demonstrations often yield low reward values.

For this reason, stochastic gradients estimated by these demonstrations tend to be uninformative, which leads to slow convergence and poor data-efficiency.

To avoid estimating such uninformative gradients, we use IS to estimate gradients using high-quality demonstrations which are sampled with high probability.

Briefly, IS is a technique for estimating an expectation over a distribution by using samples from a different distribution (Robert & Casella, 2005) .

For VILD, we propose to sample k from a distributionppkq 9 }vecpC´1 ω pkqq} 1 .

This distribution assigns high probabilities to demonstrators with high estimated level of expertise (i.e., demonstrators with a small C ω pkq).

With this distribution, the estimated gradients tend to be more informative which leads to a faster convergence.

To reduce a sampling bias, we use a truncated importance weight: wpkq " minpppkq{ppkq, 1q (Ionides, 2008) , which leads to an IS gradient: ∇ φ H IS " ∇ φ tE p d ps 1:T ,u 1:T |kqppkq rwpkqΣ T t"1 E q ψ pat|st,ut,kq rr φ ps t , a t qssÉ r q θ ps 1:T ,a 1:T q rΣ T t"1 r φ ps t , a t qsu.

Computing wpkq requires ppkq, which can be estimated accurately since k is a discrete random variable.

For simplicity, we assume that ppkq is a uniform distribution.

In this section, we will discuss a related area of supervised learning with diverse-quality data.

Besides, we will discuss existing IL methods that use the variational approach.

Supervised learning with diverse-quality data.

In supervised learning, diverse-quality data has been extensively studied, e.g. learning with noisy labels (Angluin & Laird, 1988) .

This task assumes that human labelers may assign incorrect labels to training inputs.

With such labelers, the obtained dataset consists of high-quality data with correct labels and low-quality data with incorrect labels.

To handle this setting, many methods were proposed (Natarajan et al., 2013; Han et al., 2018) .

The most related methods are probabilistic models, which aim to infer correct labels and the level of labelers' expertise (Raykar et al., 2010; Khetan et al., 2018) .

Specifically, Raykar et al. (2010) proposed a method based on a two-coin model which enables estimating the correct labels and level of expertise.

Recently, Khetan et al. (2018) proposed a method based on weighted loss functions, where the weight is determined by the estimated labels and level of expertise.

Methods for supervised learning with diverse-quality data can be leveraged to learn a policy in our setting.

However, they tend to perform poorly due to the issue of compounding error, as discussed previously in Section 3.1.

Variational approach in IL.

The variational approach has been previously utilized in IL to perform MM-IL and reduce over-fitting.

Specifically, MM-IL aims to learn a multi-modal policy from diverse demonstrations collected by many experts , where each mode of the policy represents decision making of each expert 2 .

A multi-modal policy is commonly represented by a contextdependent policy, where each context represents each mode of the policy.

The variational approach has been used to learn such contexts, i.e., by learning a variational auto-encoder (Wang et al., 2017) and maximizing a variational lower-bound of mutual information Hausman et al., 2017) .

Meanwhile, variational information bottleneck (VIB) (Alemi et al., 2017) has been used to reduce over-fitting in IL (Peng et al., 2019) .

Specifically, VIB aims to compress information flow by minimizing a variational bound of mutual information.

This compression filters irrelevant signals, which leads to less over-fitting.

Unlike these existing works, we utilize the variational approach to aid computing integrals in large state-action spaces, but not for learning a variational auto-encoder or optimizing a variational bound of mutual information.

In this section, we experimentally evaluate the performance of VILD (with and without IS) in continuous-control benchmarks and real-world crowdsourced demonstrations.

For benchmarks, we use four continuous-control tasks from OpenAI gym (Brockman et al., 2016) with demonstrations from a pre-trained RL agent.

For real-world demonstrations, we use a robosuite reaching task (Fan et al., 2018) with demonstrations from real-world crowdsourcing platform (Mandlekar et al., 2018) .

Performance is evaluated using a cumulative ground-truth reward along trajectories (i.e., higher is better) (Ho & Ermon, 2016) , and this cumulative reward is computed using test trajectories generated by learned policies (i.e., q θ pa t |s t q).

We use 10 test trajectories for the benchmark tasks, and use 100 test trajectories for the robosuite reaching task.

Note that we use a larger number of test trajectories due to high variability of initial states in the robosuite reaching task.

We repeat experiments for 5 trials with different random seeds and report the mean and standard error.

Baseline.

We compare VILD against GAIL (Ho & Ermon, 2016) , AIRL (Fu et al., 2018) , VAIL (Peng et al., 2019) , MaxEnt-IRL (Ziebart et al., 2010) , and InfoGAIL .

These are online IL methods which collect transition samples to learn policies.

We use trust region policy optimization (TRPO) (Schulman et al., 2015) to update policies, except for the Humanoid task where we use soft actor-critic (SAC) (Haarnoja et al., 2018) .

For InfoGAIL, we report the performance averaged over uniformly sampled contexts, as well as the performance with the best context chosen during testing.

Data generation.

To generate demonstrations from π ‹ (pre-trained by TRPO) according to Figure 1(b) , we use two types of noisy policy ppu t |a t , s t , kq: Gaussian noisy policy: N pu t |a t , σ 2 k Iq and time-signal-dependent (TSD) noisy policy: N pu t |a t , diagpb k ptqˆ}a t } 1 {d a qq, where b k ptq is sampled from a noise process.

We use K " 10 demonstrators with different σ k and noise processes for b k ptq.

Each demonstrator generates trajectories with approximately T " 1000 time steps.

The number of state-action pairs in each dataset is approximately 10000.

Notice that for TSD, the noise variance depends on time and magnitude of actions.

This characteristic of TSD has been observed in human motor control (van Beers et al., 2004) .

More details of data generation are given in Appendix C.

Results against online IL methods.

Figure 2 shows learning curves of VILD and existing methods against the number of transition samples in HalfCheetah and Ant 3 , whereas Table 1 reports the performance achieved in the last 100 iterations.

Clearly, VILD with IS overall outperforms existing methods in terms of both data-efficiency and final performance, i.e., VILD with IS learns better policies using less numbers of transition samples.

VILD without IS tends to outperform existing methods in terms of the final performance.

However, it is less data-efficient when compared to VILD with IS, except on Humanoid with the Gaussian noisy policy, where VILD without IS tends to perform better than VILD with IS.

We conjecture that this is because IS slightly biases gradient estimation, which may have a negative effect on the performance.

Nonetheless, the overall good performance of VILD with IS suggests that it is an effective method to handle diverse-quality demonstrations.

On the contrary, existing methods perform poorly as expected, except on the Humanoid task.

For the Humanoid task, VILD tends to perform the best in terms of the mean performance.

Nonetheless, all methods except GAIL achieve statistically comparable performance according to t-test.

This is perhaps because amateurs in this task perform relatively well compared to amateurs in other tasks, as seen from demonstrators' performance given in Table 2 and 3 (Appendix C).

Since amateurs perform relatively well, demonstrations from these amateurs do not severely affect the performance of IL methods in this task when compared to the other tasks.

We found that InfoGAIL, which learns a context-dependent policy, may achieve good performance when the policy is conditioned on specific contexts.

For instance, InfoGAIL (best context) performs quite well in the Walker2d task with the TSD noisy policy (the learning curves are provided in Figure 7 (b)).

However, as shown in Figure 10 , its performance varies across contexts and is quite poor on average when using contexts from a uniform distribution.

These results support our conjecture that MM-IL methods are not suitable for our setting where the level of demonstrators' expertise is absent.

It can be seen that VILD without IS performs better for the Gaussian noisy policy when compared to the TSD noisy policy.

This is because the model of VILD is correctly specified for the Gaussian noisy policy, but the model is incorrectly specified for the TSD noisy policy; misspecified model indeed leads to the reduction in performance.

Nonetheless, VILD with IS still performs well for both types of noisy policy.

This is perhaps because negative effects of a misspecified model are not too severe for learning expertise parameters, which are required to compute r ppkq.

We also conduct the following evaluations.

Due to space limitation, figures are given in Appendix D.

Results against offline IL methods.

We compare VILD against offline IL methods based on supervised learning, namely behavior cloning (BC) (Pomerleau, 1988) , Co-Teaching which is based on a method for learning with noisy labels (Han et al., 2018) , and BC from diverse-quality demonstrations (BC-D) which optimizes the naive model described in Section 3.1.

Results in Figure 8 show that these methods perform worse than VILD overall; BC performs the worst since it severely suffers from both the compounding error and low-quality demonstrations.

Compared to BC, BC-D and Co-teaching are quite robust against low-quality demonstrations, but they still perform worse than VILD with IS.

Accuracy of estimated expertise parameter.

To evaluate accuracy of estimated expertise parameter, we compare the ground-truth value of σ k under the Gaussian noisy policy against the learned covariance C ω pkq.

Figure 9 shows that VILD learns an accurate ranking of demonstrators' expertise.

The values of these parameters are also quite accurate compared to the ground-truth, except for demonstrators with low-level of expertise.

A reason for this phenomena is that low-quality demonstrations are highly dissimilar, which makes learning the expertise more challenging.

In this experiment, we evaluate the robustness of VILD against real-world demonstrations.

Specifically, we conduct an experiment using real-world demonstrations collected by a robotic crowdsourcing platform (Mandlekar et al., 2018) .

The public datasets were collected in the robosuite environment for object-manipulation tasks such as assembly tasks (Fan et al., 2018) .

In our experiment, we consider a reaching task, where demonstrations come from clipped assembly tasks when the robot's end-effector contacts the target object.

We uses N " 10 demonstrations whose length are approximately T " 500 and set K " 10.

The number of state-action pairs in a demonstration dataset is approximately 5000.

For VILD, we apply the log-sigmoid function to the reward function, which improves the performance in this task.

More details of the experimental setting are provided in Appendix C.2.

Figure 3 shows the performance of all methods, except VILD without IS and VAIL.

We do not evaluate VILD without IS and VAIL since IS improves the performance and VAIL is comparable to GAIL.

It can be seen that VILD with IS performs better than GAIL, AIRL, and MaxEnt-IRL.

VILD also performs better than InfoGAIL in terms of the final performance; InfoGAIL learns faster in the early stage of learning, but its performance saturates and VILD eventually outperforms InfoGAIL.

These experimental results show that VILD is more robust against real-world demonstrations with diversequality when compared to existing state-of-the-art methods.

An example of trajectory generated by VILD's policy is shown in Figure 5 .

Figure 4 shows the performance of InfoGAIL with different context variables z .

We can see that InfoGAIL performs well when the policy is conditioned on specific contexts, e.g., z " 7.

Indeed, the best context during testing can improve the performance of InfoGAIL.

The effectiveness of such an approach is demonstrated in Figure 3 , where InfoGAIL (best context) performs very well.

However, InfoGAIL (best context) is less practical than VILD, since choosing the best context requires an expert to evaluate the performance of all contexts.

In contrast, the performance of VILD does not depend on contexts, since VILD does not learn a context-dependent policy.

Moreover, the performance of InfoGAIL (best context) is quite unstable, and it is still outperformed by VILD in terms of the final performance.

In this paper, we explored a practical setting in IL where demonstrations have diverse-quality.

We showed the deficiency of existing methods, and proposed a robust method called VILD, which learns both the reward function and the level of demonstrators' expertise by using the variational approach.

Empirical results demonstrated that our work enables scalable and data-efficient IL under this practical setting.

In future, we will explore other approaches to efficiently estimate parameters of the proposed model except the variational approach.

We will also explore approaches to handle model misspecification, i.e., scenarios where the noisy policy differs from the model p ω .

Specifically, we will explore more flexible models of p ω such as neural networks, as well as using the tempered posterior approach (Grünwald & van Ommen, 2017) to improve robustness of our model.

This section derives the lower-bounds of f pφ, ωq and gpφ, ωq presented in the paper.

We also derive the objective function Hpφ, ω, ψ, θq of VILD.

Let l φ,ω pst, at, ut, kq " r φ pst, atq`log pωput|st, at, kq, we have that f pφ, ωq "

, where ftpφ, ωq " log ş A exp pl φ,ω pst, at, ut, kqq dat.

By using a variational distribution q ψ pat|st, ut, kq with parameter ψ, we can bound ftpφ, ωq from below by using the Jensen inequality as follows:

at " E q ψ pa t |s t ,u t ,kq rl φ,ω pst, at, ut, kq´log q ψ pat|st, ut, kqs

Then, by using the linearity of expectation, we obtain the lower-bound of f pφ, ωq as follows:

To verify that f pφ, ωq " max ψ Fpφ, ω, ψq, we maximize Ftpφ, ω, ψq w.r.t.

q ψ under the constraint that q ψ is a valid probability density, i.e., q ψ pat|st, ut, kq ą 0 and ş A q ψ pat|st, ut, kqdat " 1.

By setting the derivative of Ftpφ, ω, ψq w.r.t.

q ψ to zero, we obtain q ψ pat|st, ut, kq " exp pl φ,ω pst, at, ut, kq´1q " exp pl φ,ω pst, at, ut, kqq ş A exp pl φ,ω pst, at, ut, kqq dat , where the last line follows from the constraint ş A q ψ pat|st, ut, kqdat " 1.

To show that this is indeed the maximizer, we substitute q ψ ‹ pat|st, ut, kq " expplps t ,a t ,u t ,kqq ş A expplps t ,a t ,u t ,kqqda t into Ftpφ, ω, ψq:

This equality verifies that ftpφ, ωq " max ψ Ftpφ, ω, ψq.

Finally, by using the linearity of expectation, we have that f pφ, ωq " max ψ Fpφ, ω, ψq.

Next, we derive the lower-bound of gpφ, ωq presented in the paper.

We first derive a trivial lower-bound using a general variational distribution over trajectories and reveal its issues.

Then, we derive a lower-bound presented in the paper by using a structured variational distribution.

Recall that the function gpφ, ωq " log Z φ,ω is gpφ, ωq " log¨K

ppst`1|st, utq exp plpst, at, ut, kqq ds1:T du1:T da1:T‹ ‚.

Lower-bound via a variational distribution A lower-bound of g can be obtained by using a variational distribution s q β ps1:T , u1:T , a1:T , kq with parameter β.

We note that this variational distribution allows any dependency between the random variables s1:T , u1:T , a1:T , and k. By using this distribution, we have a lower-bound gpφ, ωq " log˜K

ppst`1|st, utq exp pl φ,ω pst, at, ut, kqq s q β ps1:T , u1:T , a1:T , kq s q β ps1:T , u1:T , a1:T , kq ds1:T du1:T da1:Tȩ E s q β ps 1:T ,u 1:T ,a 1:T ,kq « log ppkqp1ps1q`T ÿ t"1 tlog ppst`1|st, utq`l φ,ω pst, at, ut, kqú log s q β ps1:T , u1:T , a1:T , kq

The main issue of using this lower-bound is that, s Gpφ, ω, βq can be computed or approximated only when we have an access to the transition probability ppst`1|st, utq.

In many practical tasks, the transition probability is unknown and needs to be estimated.

However, estimating the transition probability for large state and action spaces is known to be highly challenging (Sutton & Barto, 1998) .

For these reasons, this lower-bound is not suitable for our method.

Lower-bound via a structured variational distribution To avoid the above issue, we use the structure variational approach (Hoffman & Blei, 2015) , where the key idea is to pre-define conditional dependency to ease computation.

Specifically, we use a variational distribution q θ pat, ut|st, kq with parameter θ and define dependencies between states according to the transition probability of an MDP.

With this variational distribution, we lower-bound g as follows:

ppst`1|st, utq exp pl φ,ω pst, at, ut, kqq q θ pat, ut|st, kq q θ pat, ut|st, kq ds1:T du1:T da1:Tȩ E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

where r q θ ps1:T , u1:T , a1:T , kq " ppkqp1ps1qΠ T t"1 ppst`1|st, utqq θ pat, ut|st, kq.

The optimal variational distribution q θ ‹ pat, ut|st, kq can be founded by maximizing Gpφ, ω, θq w.r.t.

q θ .

Solving this maximization problem is identical to solving a maximum entropy RL (MaxEnt-RL) problem (Ziebart et al., 2010) for an MDP defined by a tuple M " pSˆN`, AˆA, pps 1 , |s, uqI k"k 1 , p1ps1qppk1q, l φ,ω ps, a, u, kqq.

Specifically, this MDP is defined with a state variable pst, ktq P SˆN, an action variable pat, utq P AˆA, a transition probability density ppst`1, |st, utqI k t "k t`1 , an initial state density p1ps1qppk1q, and a reward function l φ,ω pst, at, ut, kq.

Here, I a"b is the indicator function which equals to 1 if a " b and 0 otherwise.

By adopting the optimality results of MaxEnt-RL (Ziebart et al., 2010; Haarnoja et al., 2018) , we have gpφ, ωq " max θ Gpφ, ω, θq, where the optimal variational distribution is

The functions Q and V are soft-value functions defined as Qpst, k, at, utq " l φ,ω pst, at, ut, kq`E pps t`1 |s t ,u t q rV pst`1, kqs ,

V pst, kq " log

This section derives the objective function Hpφ, ω, ψ, θq from Fpφ, ω, ψq´Gpφ, ω, θq.

Specifically, we substitute the models pωput|st, at, kq " N put|at, Cωpkqq and q θ pat, ut|st, kq " q θ pat|stqN put|at, Σq.

We also give an example when using a Laplace distribution for pωput|st, at, kq instead of the Gaussian distribution.

First, we substitute q θ pat, ut|st, kq " q θ pat|stqN put|at, Σq into G:

Gpφ, ω, θq " E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

where c1 is a constant corresponding to the log-normalization term of the Gaussian distribution.

Next, by using the re-parameterization trick, we rewrite r q θ ps1:T , u1:T , a1:T , kq as r q θ ps1:T , u1:T , a1:T , kq " ppkqp1ps1q

where we use ut " at`Σ 1{2 t with t " N p t|0, Iq.

With this, the expectation of Σ T t"1 }ut´at} 2 Σ´1 over r q θ ps1:T , u1:T , a1:T , kq can be written as E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

ff " E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

ff " E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

which is a constant.

Then, the quantity G can be expressed as Gpφ, ω, θq " E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

By ignoring the constant, the optimization problem max φ,ω,ψ min θ Fpφ, ω, ψq´Gpφ, ω, θq is equivalent to

E q ψ pa t |s t ,u t ,kq rl φ,ω pst, at, ut, kq´log q ψ pat|st, ut, kqs f E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

Our next step is to substitute pωput|st, at, kq by our choice of model.

First, let us consider a Gaussian distribution pωput|st, at, kq " N put|at, Cωpst, kqq, where the covariance depends on state.

With this model, the second term in Eq. (13) is given by E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

where c2 "´d a 2 log 2π is a constant.

By using the reparameterization trick, we write the expectation of Σ ff .

Using this equality, the second term in Eq. (13) is given by E r q θ ps 1:T ,u 1:T ,a 1:T ,kq

Maximizing this quantity w.r.t.

θ has an implication as follows: q θ pat|stq maximizes the expected cumulative reward while avoiding states that are difficult for demonstrators.

Specifically, a large value of E ppkq rlog |Cωpst, kq|s indicates that demonstrators have a low level of expertise for state st on average, given by our estimated covariance.

In other words, this state is difficult to accurately execute optimal actions for all demonstrators on averages.

Since the policy q θ pat|stq should minimize E ppkq rlog |Cωpst, kq|s, the policy should avoid states that are difficult for demonstrators.

We expect that this property may improve exploration-exploitation trade-off in IL.

Nonetheless, we leave an investigation of this property for future work, since this is not in the scope of the paper.

In this paper, we specify that the covariance does not depend on state:

Cωpst, kq " Cωpkq.

This model specification enables us to simplify Eq. (14) as follows:

where r q θ ps1:T , a1:T q " p1ps1q

ppst`1|st, utqN put|at, Σqdutq θ pat|stq.

The last line follows from the quadratic form identity:

Next, we substitute pωput|st, at, kq " N put|at, Cωpkqq into the first term of Eq. (13).

Lastly, by ignoring constants, Eq. (13) is equivalent to max φ,ω,ψ min θ Hpφ, ω, ψ, θq, where

This concludes the derivation of VILD.

As mentioned, other distributions beside the Gaussian distribution can be used for pω.

For instance, let us consider a multivariate-independent Laplace distribution: pωput|st, at, kq " Π

where a division of vector by vector denotes element-wise division.

The Laplace distribution has heavier tails when compared to the Gaussian distribution, which makes the Laplace distribution more suitable for modeling demonstrators who tend to execute outlier actions.

By using the Laplace distribution for pωput|st, at, kq, we obtain an objective

We can see that differences between HLap and H are the absolute error and scaling of the trace term.

We implement VILD using the PyTorch deep learning framework.

For all function approximators, we use neural networks with 2 hidden-layers of 100 tanh units, except for the Humanoid task and the robosuite reaching task Update q ψ by an estimate of ∇ ψ Hpφ, ω, ψ, θq.

Update p ω by an estimate of ∇ ω Hpφ, ω, ψ, θq`∇ ω Lpωq.

Update r φ by an estimate of ∇ φ H IS pφ, ω, ψ, θq.

Update q θ by an RL method (e.g., TRPO or SAC) with reward function r φ .

where we use neural networks with 2 hidden-layers of 100 relu units.

We optimize parameters φ, ω, and ψ by Adam with step-size 3ˆ10´4, β1 " 0.9, β2 " 0.999 and mini-batch size 256.

To optimize the policy parameter θ, we use trust region policy optimization (TRPO) (Schulman et al., 2015) with batch size 1000, except on the Humanoid task where we use soft actor-critic (SAC) (Haarnoja et al., 2018) with mini-batch size 256.

Note that TRPO is an on-policy RL method that uses only trajectories collected by the current policy, while SAC is an off-policy RL method that use trajectories collected by previous policies.

On-policy methods are generally more stable than off-policy methods, while off-policy methods are generally more data-efficient (Gu et al., 2017) .

We use SAC for Humanoid mainly due to its high data-efficiency.

When SAC is used, we also use trajectories collected by previous policies to approximate the expectation over the trajectory densityq θ ps1:T , a1:T q.

For the distribution pωput|st, at, kq " N put|at, Cωpkqq, we use diagonal covariances Cωpkq " diagpc k q, where ω " tc k u K k"1 and c k P R dà are parameter vectors to be learned.

For the distribution q ψ pat|st, ut, kq, we use a Gaussian distribution with diagonal covariance, where the mean and logarithm of the standard deviation are the outputs of neural networks.

Since k is a discrete variable, we represent q ψ pat|st, ut, kq by neural networks that have K output heads and take input vectors pst, utq; The k-th output head corresponds to (the mean and log-standard-deviation of) q ψ pat|st, ut, kq.

We also pre-train the mean function of q ψ pat|st, ut, kq, by performing least-squares regression for 1000 gradient steps with target value ut.

This pre-training is done to obtain reasonable initial predictions.

For the policy q θ pat|stq, we use a Gaussian policy with diagonal covariance, where the mean and logarithm of the standard deviation are outputs of neural networks.

We use Σ " 10´8I in experiments.

To control exploration-exploitation trade-off, we use an entropy coefficient α " 0.0001 in TRPO.

In SAC, the value of α is optimized so that the policy has a certain value of entropy, as described by Haarnoja et al. (2018) .

Note that including α in VILD is equivalent to rescaling quantities in the model by α, i.e., exppr φ pst, atq{αq and ppωput|st, at, kqq 1 α .

A discount factor 0 ă γ ă 1 may be included similarly, and we use γ " 0.99 in experiments.

For all methods, we regularize the reward/discriminator function by the gradient penalty (Gulrajani et al., 2017) with coefficient 10, since it was previously shown to improve performance of generative adversarial learning methods.

For methods that learn a reward function, namely VILD, AIRL, and MaxEnt-IRL, we apply a sigmoid function to the output of a reward network to bound reward values.

We found that without the bounds, reward values of the agent can be highly negative in the early stage of learning, which makes RL methods prematurely converge to poor policies.

An explanation of this phenomenon is that, in MDPs with large state and action spaces, distribution of demonstrations and distribution of agent's trajectories are not overlapped in the early stage of learning.

In such a scenario, it is trivial to learn a reward function which tends to positive-infinity values for demonstrations and negative-infinity values for agent's trajectories.

While the gradient penalty regularizer slightly remedies this issue, we found that the regularizer alone is insufficient to prevent this scenario.

Moreover, for VILD, it is beneficial to bound the reward function to control a trade-off between the immediate reward and the squared error when optimizing ψ.

A pseudo-code of VILD with IS is given in Algorithm 1, where the reward parameter is updated by IS gradient in line 8.

For VILD without IS, the reward parameter is instead updated by an estimate of ∇ φ Hpφ, ω, ψ, θq.

The regularizer Lpωq " T E ppkq rlog |C´1 ω pkq|s{2 penalizes large value of Cωpkq.

A source-code of our implementation will be publicly available.

In this section, we describe experimental settings and data generation.

We also give brief reviews of methods compared against VILD in the experiments.

For the benchmark experiment in Section 5.1, we evaluate VILD on four continuous-control benchmark tasks from OpenAI gym platform (Brockman et al., 2016) with the Mujoco physics simulator: HalfCheetah, Ant, Walker2d, and Humanoid.

To obtain the optimal policy for generating demonstrations, we use the ground-truth reward function of each task to pre-train π ‹ with TRPO.

We generate diverse-quality demonstrations by using K " 10 demonstrators according to the graphical model in Figure 1(b) .

We consider two types of the noisy policy pput|st, at, kq: a Gaussian noisy policy and a time-signal-dependent (TSD) noisy policy.

Gaussian noisy policy.

We use a Gaussian noisy policy N put|at, σ 2 k Iq with a constant covariance.

The value of σ k for each of the 10 demonstrators is 0.01, 0.05, 0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9 and 1.0, respectively.

Note that our model assumption on pω corresponds to this Gaussian noisy policy.

Table 2 shows the performance of demonstrators (in terms of cumulative ground-truth rewards) with this Gaussian noisy policy.

A random policy π0 is an initial policy neural network for learning; The network weights are initialized such that the magnitude of actions is small.

Note that this initialization scheme is a common practice in deep RL (Gu et al., 2017) .

TSD noisy policy.

To make learning more challenging, we generate demonstrations according to a noise characteristic of human motor control, where a magnitude of noises is proportion to a magnitude of actions and increases with execution time (van Beers et al., 2004) .

Specifically, we generate demonstrations using a Gaussian distribution N put|at, diagpb k ptqˆ}at}1{daqq, where the covariance is proportion to the magnitude of action and depends on time steps andd enotes an element-wise product.

We call this policy time-signaldependent (TSD) noisy policy.

Here, b k ptq is a sample of a noise process whose noise variance increases over time, as shown in Figure 6 .

We obtain this noise process for the k-th demonstrator by reversing Ornstein-Uhlenbeck (OU) processes with parameters θ " 0.15 and σ " σ k (Uhlenbeck & Ornstein, 1930) 4 .

The value of σ k for each demonstrator is 0.01, 0.05, 0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9, and 1.0, respectively.

Table 3 shows the performance of demonstrators with this TSD noisy policy.

Learning from demonstrations generated by TSD is challenging; The Gaussian model of pω cannot perfectly model the TSD noisy policy, since the ground-truth variance is a function of actions and time steps.

For the real-world data experiment in Section 5.2, we use a robot control task from the robosuite environment Fan et al. (2018) and a crowdsourced demonstration dataset from Mandlekar et al. (2018) 5 .

These demonstrations are collected for object-manipulation tasks such as assembly tasks.

These object-manipulation tasks require the agent to perform three subtasks: reaching, picking, and placing.

In our preliminary experiments, none of IL methods successfully learns object-manipulation policies, since the agent often fails at picking the object.

We expect that a hierarchical policy is necessary to perform these manipulation tasks, due to the hierarchical structure (i.e., subtasks) of these tasks.

Since hierarchical IL is not in the scope of this paper, we consider the subtask of reaching where non-hierarchical policies suffice.

We leave an extension of VILD to hierarchical policy for future work.

In this experiment, we consider the subtask of reaching, which is still challenging for IL due to diverse quality of crowdsourced demonstrations.

To obtain reaching demonstrations from the original object-manipulation demonstrations (we use the SawyerNutAssemblyRound dataset), we terminate demonstrations after the robot's end-effector contacts the target object.

After applying such a termination procedure, the dataset used in this experiment consists of 10 randomly chosen demonstrations (N " 10) whose length T is approximately 500 time steps.

The number of state-action pairs in this demonstration dataset is approximately 5000.

Since we do not know the actual number of demonstrators that collected these N " 10 demonstrations, we use the strategy described in Section 3.3; we set K " N and k " n.

We use true states of the robot and do not use visual observations.

Since the reaching task does not require picking the object, we disable the gripper control command of the robot.

The state space of this task is S Ď R 44 , and the action space of this task is A Ď R 7 .

Figure 11 shows three examples of demonstrations used in this experiment.

We can notice the differences in qualities of demonstrations, e.g., demonstration 3 is better than demonstration 2 since the robot reaches the object faster.

The performance of learned policies are evaluated using a reward function whose values are inverse proportion to the distance between the object and the end-effector (i.e., small distance yields high reward).

We repeat the experiment for 5 trials using the same dataset and report the average performance (undiscounted cumulative rewards).

For each trial, we generate 100 test trajectories for evaluating the performance.

Note that the number of test trajectories in this experiment is larger than that in the benchmark experiments.

This is because the initial states of this reaching task is much more varied than those in benchmark tasks.

We do not evaluate VILD without IS and VAIL, since in benchmarks VILD with IS performs better than VILD without IS and VAIL is comparable to GAIL.

For all methods, we use neural networks with 2 hidden-layers of 100 relu units.

We update policy parameters by TRPO with the same hyper-parameters as the benchmark experiments.

We pre-train the mean of Gaussian policies for all methods by behavior cloning (i.e., we apply 1000 gradient descent steps of least-squares regression).

To pre-train InfoGAIL which learns a context-dependent policy, we use the variable k as context for pre-training.

For VILD, we apply the log-sigmoid function to the reward function.

Specifically, we parameterize the reward function as r φ ps, aq " log D φ ps, aq where D φ ps, aq " exppd φ ps,aqq exppd φ ps,aqq`1 and d φ : SˆA Ñ R. We also apply a substitution´log D φ ps, aq Ñ logp1´D φ ps, aqq, which is a common practice in GAN literature (Fedus et al., 2018) .

By doing so, we obtain an objective of VILD that closely resembles the objective of GAIL:

We use this variant of VILD in this experiment since it performs better than VILD with the standard reward function.

Although we omit the IS distribution in this equation for clarity, we use IS in this experiment.

Here, we briefly review methods compared against VILD in our experiments.

We firstly review online IL methods, which learn a policy by RL and require additional transition samples from MDPs.

MaxEnt-IRL.

Maximum (causal) entropy IRL (MaxEnt-IRL) (Ziebart et al., 2010 ) is a well-known IRL method.

The original derivation of the method is based on the maximum entropy principle (Jaynes, 1957) but for causal interactions, and uses a linear-in-parameter reward function: r φ pst, atq " φ J bpst, atq with a basis function b. Here, we consider an alternative derivation which is applicable to nonlinear reward function (Finn et al., 2016) .

Briefly speaking, MaxEnt-IRL learns a reward parameter by minimizing a KL divergence from a data distribution p ‹ ps1:T , a1:T q to a model p φ ps1:T , a1:T q "

T t"1 ppst`1|st, atq exppr φ pst, atq{αq, where Z φ is the normalization term.

Minimizing this KL divergence is equivalent to solving max φ E p ‹ ps 1:T ,a 1:T q " Σ T t"1 r φ pst, atq ‰´l og Z φ .

To compute log Z φ , we can use the importance sampling approach (Finn et al., 2016) or the variational approache as done in VILD.

The latter leads to a max-min problem

where q θ ps1:T , a1:T q " p1ps1qΠ T t"1 ppst`1|st, atqq θ pat|stq.

The policy q θ pat|stq maximizes the learned reward function and is the solution of IL.

As we mentioned, the proposed model in VILD is based on the model of MaxEnt-IRL.

By comparing the max-min problem of MaxEnt-IRL and the max-min problem of VILD, we can see that the main difference are the variational distribution q ψ and the noisy policy model pω.

If we assume that q ψ and pω are Dirac delta functions: q ψ pat|st, ut, kq " δa t "u t and pωput|at, st, kq " δu t "a t , then the max-min problem of VILD reduces to the max-min problem of MaxEnt-IRL.

In other words, if we assume that all demonstrators execute the optimal policy and have an equal level of expertise, then VILD reduces to MaxEnt-IRL.

GAIL.

Generative adversarial IL (GAIL) (Ho & Ermon, 2016) performs occupancy measure matching via generative adversarial networks (GAN) to learn the optimal policy from expert demonstrations.

Specifically, GAIL finds a parameterized policy π θ such that the occupancy measure ρπ θ ps, aq of π θ is similar to the occupancy measure ρ π ‹ ps, aq of π ‹ .

Here, ρπps, aq " E pπ ps 1:T ,a 1:T q rΣ T t"0 δpst´s, at´aqs is the state-action occupancy measure of π and satisfies the equality E pπ ps 1:T ,a 1:T q rΣ T t"1 rpst, atqs " ť SˆA ρπps, aqrps, aqdsda " Eπ rrps, aqs.

To measure the similarity, GAIL uses the Jensen-Shannon divergence, which is estimated and minimized by the following generative-adversarial training objective:

where D φ ps, aq " exppd φ ps,aqq exppd φ ps,aqq`1 is called a discriminator.

The minimization problem w.r.t.

θ is achieved using RL with a reward function´logp1´D φ ps, aqq.

AIRL.

Adversarial IRL (AIRL) (Fu et al., 2018) was proposed to overcome a limitation of GAIL regarding reward function: GAIL does not learn the expert reward function, since GAIL has D φ ps, aq " 0.5 at the saddle point for every states and actions.

To overcome this limitation while taking advantage of generative-adversarial training, AIRL learns a reward function by solving

where D φ ps, aq " exppr φ ps,aqq exppr φ ps,aqq`q θ pa|sq .

The policy q θ pat|stq is learned by RL with a reward function r φ pst, atq.

Fu et al. (2018) showed that the gradient of this objective w.r.t.

φ is equivalent to the gradient of MaxEnt-IRL w.r.t.

φ.

The authors also proposed an approach to disentangle reward function, which leads to a better performance in transfer learning settings.

Nonetheless, this disentangle approach is general and can be applied to other IRL methods, including MaxEnt-IRL and VILD.

We do not evaluate AIRL with disentangle reward function.

We note that, based on the relation between MaxEnt-IRL and VILD, we can extend VILD to use a training procedure of AIRL.

Specifically, by applying the same derivation from MaxEnt-IRL to AIRL by Fu et al. (2018) , we can derive a variant of VILD which learns a reward parameter by solving max φ E p d ps 1:T ,u 1:T |kqppkq rΣ T t"1 E q ψ pa t |s t ,u t ,kq rlog D φ ps, aqss`E r q θ ps 1:T ,a 1:T q rΣ T t"1 logp1´D φ ps, aqqs.

We do not evaluate this variant of VILD in our experiment. (Peng et al., 2019) improves upon GAIL by using variational information bottleneck (VIB) (Alemi et al., 2017) .

VIB aims to compress information flow by minimizing a variational bound of mutual information.

This compression filters irrelevant signals, which leads to less over-fitting.

To achieve this in GAIL, VAIL learns the discriminator D φ by an optimization problem

where z is an encode vector, Epz|s, aq is an encoder, ppzq is a prior distribution of z, Ic is the target value of mutual information, and β ą 0 is a Lagrange multiplier.

With this discriminator, the policy π θ pat|stq is learned by RL with a reward function´logp1´D φ pE Epz|s,aq rzsqq.

It might be expected that the compression may make VAIL robust against diverse-quality demonstrations, since irrelevant signals in low-quality demonstrations are filtered out via the encoder.

However, we find that this is not the case, and VAIL does not improve much upon GAIL in our experiments.

This is perhaps because VAIL compress information from both demonstrators and agent's trajectories.

Meanwhile in our setting, irrelevant signals are generated only by demonstrators.

Therefore, the information bottleneck may also filter out relevant signals in agent's trajectories, which lead to poor performances.

InfoGAIL.

Information maximizing GAIL (InfoGAIL) is an extension of GAIL for learning a multi-modal policy in MM-IL.

The key idea of InfoGAIL is to introduce a context variable z to the GAIL formulation and learn a context-dependent policy π θ pa|s, zq, where each context represents each mode of the multi-modal policy.

To ensure that the context is not ignored during learning, InfoGAIL regularizes GAIL's objective so that a mutual information between contexts and state-action variables is maximized.

This mutual information is indirectly maximized via maximizing a variational lower-bound of mutual information.

By doing so, InfoGAIL solves a min-max problem min θ,Q max φ Eρ π ‹ rlog D φ ps, aqs`Eρ π θ rlogp1´D φ ps, aqq`α log π θ pa|s, zqs`λLpπ θ , Qq, where Lpπ θ , Qq " E ppzqπ θ pa|s,zq rlog Qpz|s, aq´log ppzqs is a lower-bound of mutual information, Qpz|s, aq is an encoder neural network, and ppzq is a prior distribution of contexts.

In our experiment, the number of context z is set to be the number of demonstrators K. As discussed in Section 1, when knowing the level of demonstrators' expertise, we may choose contexts that correspond to high-expertise demonstrator.

In other words, we may hand-craft the prior distribution ppzq so that a probability of contexts is proportion to the level of demonstrators' expertise.

Nonetheless, for fair comparison, we do not use the oracle knowledge about the level of demonstrators' expertise, and set ppzq to be a uniform distribution.

For the Humanoid task in our experiment, we use the Wasserstein-distance variant of InfoGAIL , since the Jensen-Shannon-divergence variant does not perform well in this task.

Next, we review offline IL methods.

These methods learn a policy based on supervised learning and do not require additional transition samples from MDPs.

BC.

Behavior cloning (BC) (Pomerleau, 1988) is perhaps the simplest IL method.

BC treats an IL problem as a standard supervised learning problem and ignores dependency between states distributions and policy.

For continuous action space, BC solves a least-square regression problem to learn a parameter θ of a deterministic policy π θ pstq:

BC-D. BC with Diverse-quality demonstrations (BC-D) is a simple extension of BC for handling diversequality demonstrations.

This method is based on the naive model in Section 3.1, and we consider it mainly for evaluation purpose.

BC-D uses supervised learning to learn a policy parameter θ and expertise parameter ω of a model p θ,ω ps1:T , u1:T , kq " ppkqpps1qΣ

To learn the parameters, we minimize the KL divergence from data distribution to the model.

By using the variational approach to handle integration over the action space, BC-D solves an optimization problem max θ,ω,ν

" log π θ pa t |s t qpω pu t |s t ,a t ,kq qν pa t |s t ,u t ,kq ıı , where qν pat|st, ut, kq is a variational distribution with parameters ν.

We note that the model p θ,ω ps1:T , u1:T , kq of BC-D can be regarded as a regression-extension of the two-coin model proposed by Raykar et al. (2010) for classification with noisy labels.

Co-teaching.

Co-teaching (Han et al., 2018) is the state-of-the-art method to perform classification with noisy labels.

This method trains two neural networks such that mini-batch samples are exchanged under a small loss criteria.

We extend this method to learn a policy by least-square regression.

Specifically, let π θ 1 pstq and π θ 2 pstq be two neural networks representing policies, and ∇ θ Lpθ, Bq " ∇ θ Σ ps,aqPB }a´π θ psq} 2 2 be gradients of a least-square loss estimated by using a mini-batch B. The parameters θ1 and θ2 are updated by iterates:

The mini-batch B θ 2 for updating θ1 is obtained such that B θ 2 incurs small loss when using prediction from π θ 2 , i.e., B θ 2 " argmin B 1 Lpθ2, B 1 q. Similarly, the mini-batch B θ 1 for updating θ2 is obtained such that B θ 1 incurs small loss when using prediction from π θ 1 .

For evaluating the performance, we use the policy network π θ 1 .

Results against online IL methods.

Figure 7 shows the learning curves of VILD and existing online IL methods against the number of transition samples.

It can be seen that for both types of noisy policy, VILD with and without IS outperform existing methods overall, except on the Humanoid tasks where most methods achieve comparable performance.

Results against offline IL methods.

Figure 8 shows learning curves of offline IL methods, namely BC, BC-D, and Co-teaching.

For comparison, the figure also shows the final performance of VILD with and without IS, according to Table 1 .

We can see that these offline methods do not perform well, especially on the highdimensional Humanoid task.

The poor performance of these methods is due to the issues of compounding error and low-quality demonstrations.

Specifically, BC performs the worst, since it suffers from both issues.

Still, BC may learn well in the early stage of learning, but its performance sharply degrades, as seen in Ant and Walker2d.

This phenomena can be explained as an empirical effect of memorization in deep neural networks (Arpit et al., 2017).

Namely, deep neural networks learn to remember samples with simple patterns first (i.e., high-quality demonstrations from experts), but as learning progresses the networks overfit to samples with difficult patterns (i.e., low-quality demonstrations from amateurs).

Co-teaching is the-state-of-the-art method to avoid this effect, and we can see that it performs significantly better than BC.

Meanwhile, BC-D, which learns the policy and level of demonstrators' expertise, also performs better than BC and is comparable to Co-teaching.

Nonetheless, the performance of Co-teaching and BC-D is still much worse than VILD with IS.

Accuracy of estimated expertise parameter.

Figure 9 shows the estimated parameters ω " tc k u K k"1 of N put|at, diagpc k qq and the ground-truth variance tσ 2 k u K k"1 of the Gaussian noisy policy N put|at, σ 2 k Iq.

The results show that VILD learns an accurate ranking of the variance compared to the ground-truth.

The values of these parameters are also quite accurate compared to the ground truth, except for demonstrators with low-levels of expertise.

A possible reason for this phenomena is that low-quality demonstrations are highly dissimilar, which makes learning the expertise more challenging.

We can also see that the difference between expertise parameters of VILD with IS and VILD without IS is small and negligible.

InfoGAIL with different values of context.

Figure 10 shows the learning curves of InfoGAIL across different values of context z. We can see that the performance of InfoGAIL depends on the context, i.e., there is a discrepancy between the best and worst performances of InfoGAIL.

The discrepancy is clearer in the Walker2d task with the TSD noisy policy and in the robosuite reaching task (Figure 4) .

Table 4 reports the performance in the last iterations in the robosuite reaching task experiments.

It can be observed that VILD with IS outperforms comparison methods in terms of the mean performance.

(a) Demonstration number 1 (k " 1).

(b) Demonstration number 2 (k " 2).

(c) Demonstration number 3 (k " 3).

<|TLDR|>

@highlight

We propose an imitation learning method to learn from diverse-quality demonstrations collected by demonstrators with different level of expertise.