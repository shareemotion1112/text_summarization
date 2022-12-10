In many settings, it is desirable to learn decision-making and control policies through learning or from expert demonstrations.

The most common approaches under this framework are Behaviour Cloning (BC), and Inverse Reinforcement Learning (IRL).

Recent methods for IRL have demonstrated the capacity to learn effective policies with access to a very limited set of demonstrations, a scenario in which BC methods often fail.

Unfortunately, directly comparing the algorithms for these methods does not provide adequate intuition for understanding this difference in performance.

This is the motivating factor for our work.

We begin by presenting $f$-MAX, a generalization of AIRL (Fu et al., 2018), a state-of-the-art IRL method.

$f$-MAX provides grounds for more directly comparing the objectives for LfD. We demonstrate that $f$-MAX, and by inheritance AIRL, is a subset of the cost-regularized IRL framework laid out by Ho & Ermon (2016).

We conclude by empirically evaluating the factors of difference between various LfD objectives in the continuous control domain.

Modern advances in reinforcement learning aim to alleviate the need for hand-engineered decisionmaking and control algorithms by designing general purpose methods that learn to optimize provided reward functions.

In many cases however, it is either too challenging to optimize a given reward (e.g. due to sparsity of signal), or it is simply impossible to design a reward function that captures the intricate details of desired outcomes.

One approach to overcoming such hurdles is Learning from Demonstrations (LfD), where algorithms are provided with expert demonstrations of how to accomplish desired tasks.

The most common approaches in the LfD framework are Behaviour Cloning (BC) and Inverse Reinforcement Learning (IRL) BID22 BID15 .

In standard BC, learning from demonstrations is treated as a supervised learning problem and policies are trained to regress expert actions from a dataset of expert demonstrations.

Other forms of Behaviour Cloning, such as DAgger BID21 , consider how to make use of an expert in a more optimal fashion.

On the other hand, in IRL the aim is to infer the reward function of the expert, and subsequently train a policy to optimize this reward.

The motivation for IRL stems from the intuition that the reward function is the most concise and portable representation of a task BID15 BID0 .Unfortunately, the standard IRL formulation BID15 faces degeneracy issues 1 .

A successful framework for overcoming such challenges is the Maximum-Entropy (Max-Ent) IRL method BID28 BID27 .

A line of research stemming from the Max-Ent IRL framework has lead to recent "adversarial" methods BID12 BID4 BID7 1 for example, any policy is optimal for the constant reward function r(s, a) = 0 2 BACKGROUND

Consider a Markov Decision Process (MDP) represented as a tuple (S, A, P, r, ρ 0 , γ) with statespace S, action-space A, dynamics P : S × A × S → [0, 1], reward function r(s, a), initial state distribution ρ 0 , and discount factor γ ∈ (0, 1).

In Maximum Entropy (Max-Ent) reinforcement learning BID23 BID25 BID20 BID6 BID10 , the goal is to find a policy π such that trajectories sampled using this policy follow the distribution DISPLAYFORM0 where τ = (s 0 , a 0 , s 1 , a 1 , ...) denotes a trajectory, and R(τ ) = t r(s t , a t ) and Z is the partition function.

Hence, trajectories that accumulate more reward are exponentially more likely to be sampled.

Converse to the standard RL setting, in Max-Ent IRL BID28 BID27 we are instead presented with an optimal policy π exp , or more realistically, sample trajectories from such a policy, and we seek to find a reward function r that maximizes the likelihood of the trajectories sampled from π exp .

Formally, our objective is: DISPLAYFORM1 Being an energy-based modelling objective, the difficulty in performing this optimization arises from estimating the partition function Z. Initial methods addressed this problem using dynamic programming BID28 BID27 , and recent approaches present methods aimed at intractable domains with unknown dynamics BID5 BID12 BID4 BID7 BID13 .Instead of recovering the expert's reward function and policy, recent successful methods in Max-Ent IRL aim to directly recover the policy that would result from the full process.

Since such methods only recover the policy, it would be more accurate to refer to them as Imitation Learning algorithms.

However, to avoid confusion with Behaviour Cloning methods, in this work we will refer to them as direct methods for Max-Ent IRL.GAIL: Generative Adversarial Imitation Learning Before describing the work of BID12 , we establish the definition of causal entropy BID2 .

Intuitively, causal entropy can be thought of as the "amount of options" the policy has in each state, in expectation.

DISPLAYFORM2 Let C denote a class of cost functions (negative reward functions).Furthermore, let ρ exp (s, a), ρ π (s, a) denote the state-action marginal distributions of the expert and student policy respectively.

BID12 begin with a regularized Max-Ent IRL objective, DISPLAYFORM3 where ψ : C → R is a convex regularization function on the space of cost functions, and IRL ψ (π exp ) returns the optimal cost function given the expert and choice of regularization.

Also, while not immediately clear, note that DISPLAYFORM4 , be a function that returns the optimal Max-Ent policy given cost c(s, a).

BID12 show that DISPLAYFORM5 where ψ * denotes the convex conjugate of ψ.

This tells us that if we were to find the cost function c(s, a) using the regularized Max-Ent IRL objective 3, and subsequently find the optimal Max-Ent policy for this cost, we would arrive at the same policy had we directly optimized objective 4 by searching for the policy.

Directly optimizing 4 is challenging for many choices of ψ.

Interestingly however, BID12 show that for any symmetric f -divergences BID14 , there exists a choice of ψ such that equation 4 is equivalent to RL ) ).

In such settings, due to a close connection between binary classifiers and symmetric f -divergences BID16 , efficient algorithms can be formed.

DISPLAYFORM6 The special case for Jensen-Shannon divergence leads to the successful method dubbed Generative Adversarial Imitation Learning (GAIL).

As before, let ρ exp (s, a), ρ π (s, a) denote the state-action marginal distributions of the expert and student policy respectively.

Let D(s, a) : S × A → [0, 1] be a binary classifier -often referred to as the discriminator -for identifying positive samples (sampled from ρ exp (s, a)) from negative samples (sampled from ρ π (s, a)).

Using RL, the student policy is trained to maximize E τ ∼π [ t log D(s t , a t )] − λH causal (π), where λ is a hyperparameter.

The training procedure alternates between optimizing the discriminator and updating the policy.

As noted, it is shown that this training procedure minimizes the Jensen-Shannon divergence between ρ exp (s, a) and ρ π (s, a) BID12 .AIRL: Adversarial Inverse Reinforcement Learning Subsequent to the advent of GAIL BID12 , BID4 present a theoretical discussion relating Generative Adversarial Networks (GANs) BID8 , IRL, and energy-based models.

They demonstrate how an adversarial training approach could recover the Max-Ent reward function and simultaneously train the Max-Ent policy corresponding to that reward.

Building on this discussion, BID7 present a practical implementation of this method, named Adversarial Inverse Reinforcement Learning (AIRL).As before, let ρ exp (s, a), ρ π (s, a) denote the state-action marginal distributions of the expert and student policy respectively and let D(s, a) : S × A → [0, 1] be the discriminator.

In AIRL, the discriminator is parameterized as, DISPLAYFORM7 where f (s, a) : S × A → R, and π(a|s) denotes the likelihood of the action under the policy.

AIRL defines the reward function, r(s, a) := log D(s, a) − log (1 − D(s, a)), and sets the objective for the student policy to be the RL objective, max π E τ ∼π [ t r(s t , a t )].

As in GAIL, this leads to an iterative optimization process alternating between optimizing the discriminator and the policy.

At convergence, the advantage function of the expert is recovered.

Given this observation, important considerations are made regarding how to extract the true reward function from f (s, a).

When the objective is only to perform Imitation Learning, and we do not care to recover the reward function, the discriminator does not use the special parameterization discussed above and is instead direclty represented as a function D(s, a) : S × A → [0, 1], as done in GAIL BID12 .Performance With Respect to BC Methods such as GAIL and AIRL have demonstrated significant performance gains compared to Behaviour Cloning.

In particular, in standard Mujoco benchmarks BID24 BID3 , adversarial methods for Max-Ent IRL achieve strong performance using a very limited amount of demonstrations from an expert policy, an important failure scenario for standard Behaviour Cloning.

Ho & Ermon FORMULA0 demonstrate that Max-Ent IRL is the dual problem of matching ρ π (s, a) to ρ exp (s, a); indeed as noted above, GAIL BID12 optimizes the Jensen-Shannon divergence between the two distributions.

In section 3 we present f -MAX, a method for matching ρ π (s, a) to ρ exp (s, a) using arbitrary f -divergences BID14 .

Hence, in this section we recall this class of statistical divergences as well as methods for using them for training generative models.

Let P, Q be two distributions with density functions p, q. For any convex, lower-semicontinuous function f : R + → R a statistical divergence can be defined as: DISPLAYFORM0 q(x) .

Divergences derived in this manner are called f-divergences and amongst many interesting divergences, include the forward and reverse KL.

BID17 present a variational estimation method for f -divergences between arbitrary distributions P, Q. Using the notation of BID18 we can write, DISPLAYFORM1 where T is an arbitrary class of functions T ω : X → R, and f * is the convex conjugate of f .

Under mild conditions BID17 equality holds between the two sides.

Motivated by this variational approximation as well as Generative Adversarial Networks (GANs) BID8 , BID18 present an iterative optimization scheme for matching an implicit distribution 2 Q to a fixed distribution P using any f -divergence.

For a given f -divergence, the corresponding minimax optimization is, DISPLAYFORM2 Nowozin et al. FORMULA0 discuss practical parameterizations of T ω , but to avoid notational clutter we will use the form above.

DISPLAYFORM3 We begin by presenting f -MAX, a generalization of AIRL BID7 which provides a more intuitive interpretation of what similar algorithms accomplish.

Imagine, for some f , we aim to train a policy by optimizing the f -divergence DISPLAYFORM4 To do so, we propose the following iterative optimization procedure, DISPLAYFORM5 DISPLAYFORM6 where f * and T ω are as defined in section 2.2.

Equation 8 is the same as the inner maximization of the f -GAN objective in equation 7; this objective optimizes T ω so that equation 8 best approximates DISPLAYFORM7 On the other hand, for the policy objective, using the identities in appendix A we have, DISPLAYFORM8 which implies that the policy objective is equivalent to minimizing equation 8 with respect to π.

With an identical proof as in Goodfellow et al. (2014, Proposition 2) , if in each iteration the optimal T ω is found, the described optimization procedure converges to the global optimum where the policy's state-action marginal distribution matches that of the expert's.

This is equivalent to iteratively computing D f (ρ exp (s, a)||ρ π (s, a)) and optimiizing the policy to minimize it.

Choosing DISPLAYFORM0 ).

This divergence is commonly referred to as the "reverse" KL divergence.

In this setting we have, BID18 .

Hence, given T π ω , the policy objective in equation 9 takes the form, DISPLAYFORM1 DISPLAYFORM2 On the other hand, plugging the optimal discriminator BID8 into the AIRL BID7 policy objective, we get, DISPLAYFORM3 DISPLAYFORM4 As can be seen, the right hand side of equation 12 matches that of equation 11 up to a constant 3 , meaning that AIRL is solving the Max-Ent IRL problem by minimizing the reverse KL divergence, DISPLAYFORM5 As discussed above, Ho & Ermon (2016) present a class of methods for Max-Ent IRL that directly retrieve the expert policy without explicitly finding the reward function of the expert (sec. 2.1).Using an interesting connection between surrogate cost functions for binary classification and fdivergences BID16 ), BID12 derive a special case of their method for minimizing any symmetric 4 f -divergence between ρ exp (s, a) and ρ π (s, a).

Choosing the symmetric f -divergence to be the Jensen-Shannon divergence leads to the successful special case, GAIL (sec 2.1).Surprisingly, we now show that f -MAX is a subset of the cost-regularized Max-Ent IRL framework laid out in BID12 !

Recall the following equations from this framework, DISPLAYFORM6 DISPLAYFORM7 Method Optimized Objective (Minimization) DISPLAYFORM8 is the expert, and ρ DISPLAYFORM9 where ψ(c) : C → R was a closed, proper, and convex regularization function on the space of cost function, and ψ * its convex conjugate.

For our proof we will operate in the finite state-action space, as in the original work BID12 .

In this setting, cost functions can be represented as vectors in R S×A , and joint state-action distributions can be represented as vectors in [0, 1] S×A .

Let f be the function defining some fdivergence.

Given the expert for the task, we can define the following cost function regularizer, DISPLAYFORM10 where f * is the convex conjugate of f .

Given this choice, with simple algebraic manipulation done in appendix B we have, DISPLAYFORM11 DISPLAYFORM12 Typically, the causal entropy term is considered a policy regularizer, and is weighted by 0 ≤ λ ≤ 1.

Therefore, modulo the term H causal (π), our derivations show that f -MAX, and by inheritance AIRL BID7 , all fall under the cost-regularized Max-Ent IRL framework of BID12

Given results derived in the prior section, we can now begin to populate table 1, writing various Imitation Learning algorithms in a common form, as the minimization of some statistical divergence between ρ exp (s, a) and ρ π (s, a).

In Behaviour Cloning we minimize DISPLAYFORM0 On the other hand, the corollary in section 3.1 demonstrates that AIRL BID7 minimizes KL (ρ π (s, a)||ρ exp (s, a)), while GAIL BID12 DISPLAYFORM1 Hence, there are two ways in which the direct IRL methods differ from BC.

First, in standard BC the policy is optimized to match the conditional distribution ρ exp (a|s), whereas in the other two the policy is explicitly encouraged to match the marginal state distributions as well.

Second, in BC we make use of the forward KL divergence, whereas AIRL and GAIL use divergences that exhibit more mode-seeking behaviour.

These observations allow us to generate the following two hypotheses about why direct IRL methods outperform BC, particularly in the low-data regime, Hypothesis 1 In common MDPs of interest, the reward function depends more on the state than the action.

Hence it is plausible that matching state marginals is more useful than matching action conditional marginals.

Hypothesis 2 It is known that optimization using the forward KL divergence results in distributions with a mode-covering behaviour, whereas using the reverse KL results in modeseeking behaviour BID1 .

Therefore, since in Reinforcement Learning we care about the "quality of trajectories", being mode-seeking is more beneficial than mode-covering, particularly in the low-data regime.

In what follows, we seek to experimentally evaluate our hypotheses.

To tease apart the differences between the direct Max-Ent IRL methods and BC, we present an algorithm that optimizes KL (ρ exp (s, a)||ρ π (s, a)).

We then compare its performance to Behaviour Cloning and the standard AIRL algorithm using varying amounts of expert demonstrations.

While f -MAX is a general algorithm, useful for most choices of f , it unfortunately cannot be used for the special case of forward KL, i.e. KL (ρ exp (s, a)||ρ π (s, a)).

In the following sections we identify the problem and present a separate direct Max-Ent IRL method that optimizes this divergence.

Let T π ω denote the maximizer of equation 8 for a given policy π.

For the case of forward KL, drawing upon equations from BID18 we have, DISPLAYFORM0 Given this, the objective for the policy (equation 9) under the optimal T π ω becomes, DISPLAYFORM1 Hence, there is no signal to train the policy!

6

In this section we derive an algorithm for optimizing KL (ρ exp (s, a)||ρ π (s, a)).

Similar to AIRL BID7 , let us have a discriminator, D(s, a) whose objective is to discriminate between expert and policy state-action pairs, DISPLAYFORM0 Figure 1: r(s, a) as the function of the logits of the optimal discriminator, π (s, a) = log DISPLAYFORM1 We now define the objective for the policy to be, DISPLAYFORM2 In appendix C we show, DISPLAYFORM3 This is a refreshing result since it demonstrates that we can convert the AIRL algorithm BID7 into its forward KL counterpart by simply modifying the reward function used; in AIRL (reverse KL) the reward is defined as r(s, a) := log D(s, a) − log (1 − D(s, a)), whereas for forward KL it is defined as r(s, a) := s,a) .

We refer to this forward KL version of AIRL as FAIRL.

DISPLAYFORM4 If we parameterize the discriminator as D(s, a) := σ ( (s, a) ), where σ represents the sigmoid activation function, the logit of the discriminator, (s, a), is equal to log D(s, a) − log (1 − D(s, a) ).

Hence, for an optimal discriminator, D π , we have π (s, a) = log s,a) .

It is instructive to plot the reward functions under the two different settings as a function of π (s, a); figure 1 presents these plots.

As can be seen, in the forward KL version of AIRL, if for a state-action pair the expert puts more probability mass than the policy, the policy is severely punished.

However, if for some stateaction pairs the policy places a lot more mass than the expert, it almost does not matter.

As a result, the policy spreads its mass.

On the other hand, in the original AIRL formulation (reverse KL), the policy is always encouraged to put less mass than the expert.

These observations are in line with standard intuitions about the mode-covering/mode-seeking behaviours of the two KL divergences BID1 .

DISPLAYFORM5

In this section we provide empirical comparisons between AIRL, FAIRL, and standard BC in the Ant and Halfcheetah environments found in Open-AI Gym BID3 .

Figure 2: Average return on 50 evaluation trajectories as a function of number of expert demonstrations (higher is better).

Models evaluated deterministically.

As we ran two seeds per experiment, we do not present standard deviations.

While FAIRL performs comparably to AIRL, Behaviour cloning lags behind quite significantly.

Considering the form of their objectives (table 1), this demonstrates that the advantage of direct Max-Ent IRL methods over BC is a result of the additional aspect of their objectives explicitly matching marginal state distributions.

Expert Policy To simulate access to expert demonstrations we train an expert policy using SoftActor-Critic (SAC) BID11 , a state-of-the-art reinforcement learning algorithm for continuous control.

The expert policy consists of a 2-layer MLP with 256-dim layers, ReLU activations, and two output streams for the mean and the diagonal covariance of a Tanh(Normal(µ, σ)) distribution 7 .

We use the default hyperparameter settings for training the expert.

Evaluation Setup Using a trained expert policy, we generated 4 sets of expert demonstrations of that contain {4, 8, 16, 32} trajectories.

Starting from a random offset, each trajectory is subsampled by a factor of 20.

This is standard protocol employed in prior direct methods for Max-Ent IRL BID12 BID7 .

Also note that when generating demonstrations we sample from the expert's action distribution rather than taking the mode.

This way, since the expert was trained using Soft-Actor-Critic, the expert should correspond to the Max-Ent optimal policy for the reward function 1 τ r g (s, a), where τ is the SAC temperature used and r g (s, a) is the groundtruth reward function.

To compare the various learning-from-demonstration algorithms we train each method at each amount of expert demonstrations using 2 random seeds.

For each seed, we checkpoint the model at its best validation loss 8 throughout training.

At the end of training, the resulting checkpoints are evaluated on 50 test episodes.

Details for AIRL & FAIRL For AIRL and FAIRL, the student policy has an identical architecture to that of the expert, and the discriminator is a 2-layer MLP with 256-dim layers and Tanh activations.

We normalize the observations from the environment by computing the mean and standard deviations of the expert demonstrations.

The RL algorithm used for the student policies is SAC BID11 , and the temperature parameter is tuned separately for AIRL & FAIRL.Details for BC For BC, we use an identical architecture as the expert.

The model was fit using Maximum Likelihood Estimation 9 .

As before, the observations from the environment are normalized using the mean and standard deviation of the expert demonstrations.

To match state-action marginals, the optimal student policy must sample actions from the stateconditional distribution, π(a|s).

On the other hand, when we deploy a trained policy it is reasonable to instead choose the mode of this distribution, which we call the deterministic setting.

Here, we present evaluation results under the former setting, and defer the results for the deterministic setting to the appendix.

DISPLAYFORM0 Figure 3: Validation curves throughout training using stochastic evaluation (refer to appendix D for deterministic evaluation results).

Top row Ant, Bottom row Halfcheetah.

n represents the number of expert demonstrations provided.

Due to its mode-covering behaviour, FAIRL does not perform as well as AIRL when evaluated stochastically.

However, with determinisitc evaluation FAIRL outperforms AIRL in the Ant environment.

Figure 4 demonstrates that both AIRL and FAIRL outperform BC by a large margin, especially in the low data regime.

Specifically, the fact that FAIRL outperforms BC supports hypothesis 1 that the performance gain of Max-Ent IRL is not necessarily due to the direction of KL divergence used, but is the result of explicitly encouraging the policy to match the marginal state distribution of the expert in addition to the matching of conditional action distribution.

To compare AIRL and FAIRL, in figure 3 we plot the validation curves throughout training using stochastic evaluation.

Across the two tasks and various number of expert demonstrations, AIRL consistently outperforms FAIRL.

When using deterministic evaluation (figure 5), FAIRL achieves a significant performance gain to the point that it outperforms AIRL on the Ant environment across all demonstrations set sizes.

Such observations provide initial positive support for hypothesis 2; as more expert demonstrations are provided, the policy trained with FAIRL broadens its distribution to cover the data-distribution, resulting in trajectories accumulating less reward in expectation.

We note however that more detailed experiments are necessary for adequately comparing the two methods.

The motivation for this work stemmed from the superior performance of recent direct Max-Ent IRL methods BID12 BID7 compared to BC in the low-data regime, and the desire to understand the relation between various approaches for Learning from Demonstrations.

We first presented f -MAX, a generalization of AIRL BID7 , which allowed us to interpret AIRL as optimizing for KL (ρ π (s, a)||ρ exp (s, a)).

We demonstrated that f -MAX, and by inhertance AIRL, is a subset of the cost-regularized IRL framework laid out by BID12 .

Comparing to the standard BC objective, E ρ exp (s) [KL (ρ exp (a|s)||ρ π (a|s))], we hypothesized two reasons for the superior performance of AIRL: 1) the additional terms in the objective encouraging the matching of marginal state distributions, and 2) the direction of the KL divergence being optimized.

Setting out to empirically evaluate these claims we presented FAIRL, a one-line modification of the AIRL algorithm that optimizes KL (ρ exp (s, a)||ρ π (s, a)).

FAIRL outperformed BC in a similar fashion to AIRL, which allowed us to conclude the key factor being the matching of state marginals.

Additional comparisons between FAIRL and AIRL provided initial understanding about the role of the direction of the KL being optimized.

In future work we aim to produce results on a more diverse set of more challenging environments.

Additionally, evaluating other choices of f -divergence beyond forward and reverse KL may present interesting avenues for improvement BID26 .

Lastly, but importantly, we would like to understand whether the mode-covering behaviour of FAIRL could result in more robust policies BID19 .A SOME USEFUL IDENTITIES Let h : S × A → R be an arbitrary function.

If all episodes have the same length T , we have, DISPLAYFORM0 DISPLAYFORM1 In a somewhat similar fashion, in the infinite horizon case with fixed probability γ ∈ (0, 1) of transitioning to a terminal state, for the discounted sum below we have, DISPLAYFORM2 DISPLAYFORM3 where Γ := 1 1−γ is the normalizer of the sum t γ t .

Since the integral of an infinite series is not always equal to the infinite series of integrals, some analytic considerations must be made to go from equation 34 to 35.

But, one simple case in which it holds is when the ranges of h and all ρ π (s t , a t ) are bounded.

and assuming the discriminator is optimal 10 , we have, Figure 4 : Average return on 50 evaluation trajectories as a function of number of expert demonstrations (higher is better).

Models evaluated stochastically.

As we ran two seeds per experiment, we do not present standard deviations.

While FAIRL performs comparably to AIRL, Behaviour cloning lags behind quite significantly.

Considering the form of their objectives (table 1), this demonstrates that the advantage of direct Max-Ent IRL methods over BC is a result of the additional aspect of their objectives explicitly matching marginal state distributions.10 As a reminder, the optimal discriminator has the form, D(s, a) = ρ exp (s,a) ρ exp (s,a)+ρ π (s,a).

A simple proof of which can be found in BID8 .

@highlight

Distribution matching through divergence minimization provides a common ground for comparing adversarial Maximum-Entropy Inverse Reinforcement Learning methods to Behaviour Cloning.