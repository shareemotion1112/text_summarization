It has been established that diverse behaviors spanning the controllable subspace of a Markov decision process can be trained by rewarding a policy for being distinguishable from other policies.

However, one limitation of this formulation is the difficulty to generalize beyond the finite set of behaviors being explicitly learned, as may be needed in subsequent tasks.

Successor features provide an appealing solution to this generalization problem, but require defining the reward function as linear in some grounded feature space.

In this paper, we show that these two techniques can be combined, and that each method solves the other's primary limitation.

To do so we introduce Variational Intrinsic Successor FeatuRes (VISR), a novel algorithm which learns controllable features that can be leveraged to provide enhanced generalization and fast task inference through the successor features framework.

We empirically validate VISR on the full Atari suite, in a novel setup wherein the rewards are only exposed briefly after a long unsupervised phase.

Achieving human-level performance on 12 games and beating all baselines, we believe VISR represents a step towards agents that rapidly learn from limited feedback.

Unsupervised learning has played a major role in the recent progress of deep learning.

Some of the earliest work of the present deep learning era posited unsupervised pre-training as a method for overcoming optimization difficulties inherent in contemporary supervised deep neural networks (Hinton et al., 2006; Bengio et al., 2007) .

Since then, modern deep neural networks have enabled a renaissance in generative models, with neural decoders allowing for the training of large scale, highly expressive families of directed models (Goodfellow et al., 2014; Van den Oord et al., 2016) as well as enabling powerful amortized variational inference over latent variables (Kingma and Welling, 2013) .

We have repeatedly seen how representations from unsupervised learning can be leveraged to dramatically improve sample efficiency in a variety of supervised learning domains (Rasmus et al., 2015; Salimans et al., 2016) .

In the reinforcement learning (RL) setting, the coupling between behavior, state visitation, and the algorithmic processes that give rise to behavior complicate the development of "unsupervised" methods.

The generation of behaviors by means other than seeking to maximize an extrinsic reward has long been studied under the psychological auspice of intrinsic motivation (Barto et al., 2004; Barto, 2013; Mohamed and Rezende, 2015) , often with the goal of improved exploration (??im??ek and Barto, 2006; Oudeyer and Kaplan, 2009; Bellemare et al., 2016) .

However, while exploration is classically concerned with the discovery of rewarding states, the acquisition of useful state representations and behavioral skills can also be cast as an unsupervised (i.e. extrinsically unrewarded) learning problem for agents interacting with an environment.

In the traditional supervised learning setting, popular classification benchmarks have been employed (with labels removed) as unsupervised representation learning benchmarks, wherein the acquired representations are evaluated based on their usefulness for some downstream task (most commonly the original classification task with only a fraction of the labels reinstated).

Analogously, we propose removing the rewards from an RL benchmark environment for unsupervised pre-training of an agent, with their subsequent reinstatement testing for dataefficient adaptation.

This setup emulates scenarios where unstructured interaction with the environment, or a closely related environment, is relatively inexpensive to acquire and the agent is expected to perform one or more tasks defined in this environment in the form of rewards.

The current state-of-the-art for RL with unsupervised pre-training comes from a class of algorithms which, independent of reward, maximize the mutual information between latent variable policies and their behavior in terms of state visitation, an objective which we refer to as behavioral mutual information (Mohamed and Rezende, 2015; Gregor et al., 2016; Eysenbach et al., 2018; Warde-Farley et al., 2018) .

These objectives yield policies which exhibit a great deal of diversity in behavior, with variational intrinsic control (Gregor et al., 2016, VIC) and diversity is all you need (Eysenbach et al., 2018, DIAYN) even providing a natural formalism for adapting to the downstream RL problem.

However, both methods suffer from poor generalization and a slow inference process when the reward signal is introduced.

The fundamental problem faced by these methods is the requirement to effectively interpolate between points in the latent behavior space, as the most task-appropriate latent skill likely lies "between" those learnt during the unsupervised period.

The construction of conditional policies which efficiently and effectively generalize to latent codes not encountered during training is an open problem for such methods.

Our main contribution is to address this generalization and slow inference problem by making use of another recent advance in RL, successor features (Barreto et al., 2017) .

Successor features (SF) enable fast transfer learning between tasks that differ only in their reward function, which is assumed to be linear in some features.

Prior to this work, the automatic construction of these reward function features was an open research problem .

We show that, despite being previously cast as learning a policy space, behavioral mutual information (BMI) maximization provides a compelling solution to this feature learning problem.

Specifically, we show that the BMI objective can be adapted to learn precisely the features required by SF.

Together, these methods give rise to an algorithm, Variational Intrinsic Successor FeatuRes (VISR), which significantly improves performance in the RL with unsupervised pre-training scenario.

In order to illustrate the efficacy of the proposed method, we augment the popular 57-game Atari suite with such an unsupervised phase.

The use of this well-understood collection of tasks allows us to position our contribution more clearly against the current literature.

VISR achieves human-level performance on 12 games and outperforms all baselines, which includes algorithms that operate in three regimes: strictly unsupervised, supervised with limited data, and both.

As usual, we assume that the interaction between agent and environment can be modeled as a Markov decision process (MDP, Puterman, 1994 ).

An MDP is defined as a tuple M ??? (S, A, p, r, ??) where S and A are the state and action spaces, p(??|s, a) gives the nextstate distribution upon taking action a in state s, and ?? ??? [0, 1) is a discount factor that gives smaller weights to future rewards.

The function r : S ?? A ?? S ??? R specifies the reward received at transition s a ??? ??? s ; more generally, we call any signal defined as c : S ?? A ?? S ??? R a cumulant (Sutton and Barto, 2018) .

As previously noted, we consider the scenario where the interaction of the agent with the environment can be split into two stages: an initial unsupervised phase in which the agent does not observe any rewards, and the usual reinforcement learning phase in which rewards are observable.

During the reinforcement learning phase the goal of the agent is to find a policy ?? : S ??? A that maximizes the expected return G t = ??? i=0 ?? i R t+i , where R t = r(S t , A t , S t+1 ).

A principled way to address this problem is to use methods derived from dynamic programming, which heavily rely on the concept of a value function (Puterman, 1994) .

The action-value function of a policy ?? is defined as

expected value when following policy ??.

Based on Q ?? we can compute a greedy policy

?? is guaranteed to do at least as well as ??, that is:

and ?? are called policy evaluation and policy improvement, respectively; under certain conditions their successive application leads to the optimal value function Q * , from which one can derive an optimal policy using (1).

The alternation between policy evaluation and policy improvement is at the core of many RL algorithms, which usually carry out these steps only approximately (Sutton and Barto, 2018) .

Clearly, if we replace the reward r(s, a, s ) with an arbitrary cumulant c(s, a, s ) all the above still holds.

In this case we will use Q ?? c to refer to the value of ?? under cumulant c and the associated optimal policies will be referred to as ?? c , where ?? c (s) is the greedy policy (1) on Q * c (s, a).

Usually it is assumed, either explicitly or implicitly, that during learning there is a cost associated with each transition in the environment, and therefore the agent must learn a policy as quickly as possible.

Here we consider that such a cost is only significant in the reinforcement learning phase, and therefore during the unsupervised phase the agent is essentially free to interact with the environment as much as desired.

The goal in this stage is to collect information about the environment to speed up the reinforcement learning phase as much as possible.

In what follows we will make this definition more precise.

Following Barreto et al. (2017; , we assume that there exist features ??(s, a, s ) ??? R d such that the reward function which specifies a task of interest can be written as

where w ??? R d are weights that specify how desirable each feature component is, or a 'task vector' for short.

Note that, unless we constrain ?? somehow, (2) is not restrictive in any way: for example, by making ?? i (s, a, s ) = r(s, a, s ) for some i we can clearly recover the rewards exactly.

Barreto et al. (2017) note that (2) allows one to decompose the value of a policy ?? as

where ?? t = ??(S t , A t , S t+1 ) and ?? ?? (s, a) are the successor features (SFs) of ??.

SFs can be seen as multidimensional value functions in which ??(s, a, s ) play the role of rewards, and as such they can be computed using standard RL algorithms (Szepesv??ri, 2010) .

One of the benefits provided by SFs is the possibility of quickly evaluating a policy ??.

Suppose that during the unsupervised learning phase we have computed ?? ?? ; then, during the supervised phase, we can find a w ??? R d by solving a regression problem based on (2) and then compute Q ?? through (3).

Once we have Q ?? , we can apply (1) to derive a policy ?? that will likely outperform ??.

Since ?? was computed without access to the reward, its is not deliberately trying to maximize it.

Thus, the solution ?? relies on a single step of policy improvement (1) over a policy that is agnostic to the rewards.

It turns out that we can do better than that by extending the strategy above to multiple policies.

Let e : (S ??? A) ??? R k be a policy-encoding mapping, that is, a function that turns policies ?? into vectors in R k .

Borsa et al.'s (2019) universal successor feature (USFs) are defined as ??(s, a, e(??)) ??? ?? ?? (s, a).

Note that, using USFs, we can evaluate any policy ?? by simply computing

Now that we can compute Q ?? for any ??, we should be able to leverage this information to improve our previous solution based on a single policy.

This is possible through generalized policy improvement (Barreto et al., 2017, GPI) .

Let ?? be USFs, let ?? 1 , ?? 2 , ..., ?? n be arbitrary policies, and let

It can be shown that (5) is a strict generalization of (1), in the sense that Q ?? (s, a) ??? Q ??i (s, a) for all ?? i , s, and a. This result can be extended to the case in which (2) holds only approximately and ?? is replaced by a universal successor feature approximator (USFA) ?? ?? ??? ??(s, a) (Barreto et al., 2017; Borsa et al., 2019) .

The above suggests an approach to leveraging unsupervised pre-training for more dataefficient reinforcement learning.

First, during the unsupervised phase, the agent learns a USFA ?? ?? .

Then, the rewards observed at the early stages of the RL phase are used to find an approximate solution w for (2).

Finally, n policies ?? i are generated and a policy ?? is derived through (5).

If the approximations used in this process are reasonably accurate, ?? will be an improvement over ?? 1 , ?? 2 , .., ?? n .

However, in order to actually implement the strategy above we have to answer two fundamental questions: (i) Where do the features ?? in (2) come from? (ii) How do we define the policies ?? i used in (5)?

It turns out that these questions allow for complementary answers, as we discuss next.

Features ?? should be defined in such a way that the down-stream task reward is likely to be a simple function of them (see (2)).

Since in the RL with unsupervised pre-training regime the task reward is not available during the long unsupervised phase, this amounts to utilizing a strong inductive bias that is likely to yield features relevant to the rewards of any 'reasonable' task.

One such bias is to only represent the subset of observation space that the agent can control (Gregor et al., 2016) .

This can be accomplished by maximizing the mutual information between a policy conditioning variable and the agent's behavior.

There exist many algorithms that maximize this quantity through various means and for various definitions of 'behavior' (Eysenbach et al., 2018; Warde-Farley et al., 2018) .

The objective F(??) is to find policy parameters ?? that maximize the mutual information (I) between some policy-conditioning variable, z, and some function f of the trajectory ?? induced by the conditioned policy, where H is the entropy of some variable:

While in general z will be a function of the state (Gregor et al., 2016) , it is common to assume that z is drawn from a fixed (or at least state-independent) distribution for the purposes of stability (Eysenbach et al., 2018) .

This simplifies the objective to minimizing the conditional entropy of the conditioning variable given the trajectory.

When the trajectory is sufficiently long, this corresponds to sampling from the steady state distribution induced by the policy.

Commonly f is assumed to return the final state, but for simplicity we will consider that f samples a single state s uniformly over ?? ?? ?? .

This intractable conditional distribution can be lower-bounded by a variational approximation (q) which produces the loss function used in practice (see Section 8.1 for a derivation based on Agakov (2004))

The variational parameters can be optimized by minimizing the negative log likelihood of samples from the true conditional distribution, i.e., q is a discriminator trying to predict the correct z from behavior.

However, it is not obvious how to optimize the policy parameters ??, as they only affect the loss through the non-differentiable environment.

The appropriate intrinsic reward function can be derived (see Section 8.2 for details) through application of the REINFORCE trick, which results in log q(z|s) serving this role.

Figure 1: VISR model diagram.

In practice w t is also fed into ?? as an input, which also allows for GPI to be used (see Algorithm 1 in Appendix).

For the random feature baseline, the discriminator q is frozen after initialization, but the same objective is used to train ??.

Traditionally, the desired product of this optimization was the conditional policy (??).

While the discriminator q could be used for imitating demonstrated behaviors (i.e. by inferring the most likely z for a given ?? ), for down-stream RL it was typically discarded in favor of explicit search over all possible z (Eysenbach et al., 2018) .

In the next section we discuss an alternative approach to leverage the behaviors learned during the unsupervised phase.

The primary motivation behind our proposed approach is to combine the rapid task inference mechanism provided by SFs with the ability of BMI methods to learn many diverse behaviors in an unsupervised way.

We begin by observing that both approaches use vectors to parameterize tasks.

In the SF formulation tasks correspond to linear weightings w of features ??(s).

The reward for a task given by w is r SF (s; w) = ??(s) T w. BMI objectives, on the other hand, define tasks using conditioning vectors z, with the reward for task z given by r BM I (s; z) = log q(z|s).

We propose restricting conditioning vectors z to correspond to task-vectors w of the SFs formulation.

The restriction that z ??? w, in turn, requires that r SF (s; w) = r BM I (s; w), which implies that the BMI discriminator q must have the form log q(w|s) = ??(s) T w.

One way to satisfy this requirement is by restricting the task vectors w and features ??(s) to be unit length and paremeterizing the discriminator q as the Von Mises-Fisher distribution with a scale parameter of 1.

Note that this form of discriminator differs from the standard choice of parameterizing q as a multivariate Gaussian, which does not satisfy equation 10.

With this variational family for the discriminator, all that is left to complete the base algorithm is to factorize the conditional policy into the policy-conditional successor features (??) and the task vector (w).

This is straightforward as any conditional policy can be represented by a UVFA (Schaul et al., 2015) , and any UVFA can be represented by a USFA given an appropriate feature basis, such as the one we have just derived.

Figure 1 shows the resulting model.

Training proceeds as in other algorithms maximizing BMI: by randomly sampling a task vector w and then trying to infer it from the state produced by the conditioned policy (in our case w is sampled from a uniform distribution over the unit circle).

The key difference is that in VISR the structure of the conditional policy (equation 5) enforces the task/dynamics factorization as in SF (equations 2 and 4), which in turn reduces task inference to a regression problem derived from equation 2.

Now that SFs have been given a feature-learning mechanism, we can return to the second question raised at the end of Section 3: how can we obtain a diverse set of policies over which to apply GPI?

Recall that we are training a USFA ??(s, a, e(??)) whose encoding function is e(??) = w (that is, ?? is the policy that tries to maximize the reward in (10) for a particular value of w).

So, the question of which policies to use with GPI comes down to the selection of a set of vectors w.

One natural w candidate is the solution for a regression problem derived from (2).

Let us call this solution w base , that is, ??(s, a, s ) w base ??? r(s, a, s ).

But what should the other task vectors w's be?

Given that task vectors are sampled from a uniform distribution over the unit circle during training, there is no single subset that has any privileged status.

So, following , we sample additional w's on the basis of similarity to w base .

Since the discriminator q enforces similarity on the basis of probability under a Von Mises-Fisher distribution, these additional w's are sampled from such a distribution centered on w base , with the concentration parameter ?? acting as a hyper-parameter specifying how diverse the additional w's should be.

Calculating the improved policy is thus done as follows:

Our experiments are divided in four groups corresponding to Sections 6.1 to 6.4.

First, we assess how well VISR does in the RL setup with an unsupervised pre-training phase described in Section 2.

Since this setup is unique in the literature on the Atari Suite, for the full two-phase process we only compare to ablations on the full VISR model and a variant of DIAYN adapted for these tasks (Table 1 , bottom section).

In order to frame performance relative to prior work, in Section 6.2 we also compare to results for algorithms that operate in a purely unsupervised manner (Table 1, top section).

Next, in Section 6.3, we contrast VISR's performance to that of standard RL algorithms in a low data regime (Table 1, middle  section) .

Finally, we assess how well the proposed approach of inferring the task through the solution of a regression derived from (2) does as compared to random search.

To evaluate VISR, we impose a two-phase setup on the full suite of 57 Atari games (Bellemare et al., 2013) .

Agents are allowed a long unsupervised training phase (250M steps) without access to rewards, followed by a short test phase with rewards (100k steps).

The full VISR algorithm includes features learned through the BMI objective and GPI to improve the execution of policies during both the training and test phases (see Algorithm 1 in the Appendix).

The main baseline model, RF VISR, removes the BMI objective, instead learning SFs over features given by a random convolutional network (the same architecture as the ?? network in the full model).

The remaining ablations remove GPI from each of these models.

The ablation results shown in Table 1 (bottom) confirm that these components of VISR play complementary roles in the overall functioning of our model (also see Figure 2a ).

In addition, DIAYN has been adapted for the Atari domain, using the same training and testing conditions, base RL algorithm, and network architecture as VISR (Eysenbach et al., 2018) .

With the standard 50-dimensional categorical z, performance was worse than random.

While decreasing the dimensionality to 5 (matching that of VISR) improved this, it was still significantly weaker than even the ablated versions of VISR.

Table 1 : Atari Suite comparisons.

@N represents the amount of RL interaction utilized.

M dn is median, M is mean, > 0 is the number of games with better than random performance, and > H is the number of games with human-level performance as defined in .

Top: unsupervised learning only (Sec. 6.2).

Mid: data-limited RL (Sec. 6.3).

Bottom: RL with unsupervised pre-training (Sec. 6.1).

Standard deviations given in Table 2 (Appendix).

Comparing against fully unsupervised approaches, our main external baseline is the Intrinsic Curiosity Module (Pathak et al., 2017) .

This uses forward model prediction error in some feature-space to produce an intrinsic reward signal.

Two variants have been evaluated on a 47 game subset of the Atari suite (Burda et al., 2018) .

One uses random features as the basis of their forward model (RF Curiosity), and the other uses features learned via an inverse-dynamics model (IDF Curiosity).

It is important to note that, in addition to the extrinsic rewards, these methods did not use the terminal signals provided by the environment, whereas all other methods reported here do use them.

The reason for not using the terminal signal was to avoid the possibility of the intrinsic reward reducing to a simple "do not die" signal.

To rule this out, an explicit "do not die" baseline was run (Pos Reward NSQ), wherein the terminal signal remains and a small constant reward is given at every time-step.

Finally, the full VISR model was run purely unsupervised.

In practice this means not performing the fast-adaptation step (i.e. reward regression), instead switching between random w vectors every 40 time-steps (as is done during the training phase).

Results shown in Table 1 (top and bottom) make it clear that while VISR is not a particularly outstanding in the unsupervised regime, when allowed 100k steps of RL it can vastly outperform these existing unsupervised methods on all criteria.

Comparisons to reinforcement learning algorithms in the low-data regime are largely based on similar analysis by Kaiser et al. (2019) on the 26 easiest games in the Atari suite (as judged by above random performance for their algorithm).

In that work the authors introduce a model-based agent (SimPLe) and show that it compares favorably to standard RL algorithms when data is limited.

Three canonical RL algorithms are compared against: proximal policy optimization (PPO) (Schulman et al., 2017) , Rainbow (Hessel et al., 2017) , and DQN .

For each, the results from the lowest data regime reported in the literature are used.

In addition, we also compare to a version of N-step Q-learning (NSQ) that uses the same codebase and base network architecture as VISR.

Results shown in Table 1 (middle) indicate that VISR is highly competitive with the other RL methods.

Note that, while these methods are actually solving the full RL problem, VISR's performance is based exclusively on the solution of a linear regression problem (equation 2).

Obviously, this solution can be used to "warm start" an agent which can then refine its policy using any RL algorithm.

We expect this version of VISR to have even better performance.

In the previous results, it was assumed that solving the linear reward-regression problem is the best way to infer the appropriate task vector.

However, Eysenbach et al. (2018) suggest a simpler approach: exhaustive search.

As there are no guarantees that extrinsic rewards will be linear in the learned features (??), it is not obvious which approach is best in practice.

We hypothesize that exploiting the reward-regression task inference mechanism provided by VISR should yield more efficient inference than random search.

To show this, 50 episodes (or 100k steps, whichever comes first) are rolled out using a trained VISR, each conditioned on a task vector chosen uniformly on a 5-dimensional sphere.

From these initial episodes, one can either pick the task vector corresponding to the trajectory with the highest return (random search), or combine the data across all episodes and solve the linear regression problem.

In each condition the VISR policy given by the inferred task vector is executed for 30 episodes and the average returns compared.

As shown in Figure 2b , linear regression substantially improves performance despite using data generated specifically to aid in random search.

The mean performance across all 57 games was 109.16 for reward-regression, compared to random search at 63.57.

Even more dramatically, the median score for reward-regression was 8.99 compared to random search at 3.45.

Overall, VISR outperformed the random search alternative on 41 of the 57 games, with one tie, using the exact same data for task inference.

This corroborates the main hypothesis of this paper, namely, that endowing features derived from BMI with the fast task-inference provided by SFs gives rise to a powerful method able to quickly learn competent policies when exposed to a reward signal.

Our results suggest that VISR is the first algorithm to achieve notable performance on the full Atari task suite in a setting of few-step RL with unsupervised pre-training, outperforming all baselines and buying performance equivalent to hundreds of millions of interaction steps compared to DQN on some games ( Figure 2c ).

As a suggestion for future investigations, the somewhat underwhelming results for the fully unsupervised version of VISR suggest that there is much room for improvement.

While curiosity-based methods are transient (i.e., asymptotically their intrinsic reward vanishes) and lack a fast adaptation mechanism, they do seem to encourage exploratory behavior slightly more than VISR.

A possible direction for future work would be to use a curiosity-based intrinsic reward inside of VISR, to encourage it to better explore the space of controllable policies.

Another interesting avenue for future investigation would be to combine the approach recently proposed by Ozair et al. (2019) to enforce the policies computed by VISR to be not only distinguishable but also far apart in a given metric space.

By using SFs on features that maximize BMI, we proposed an approach, VISR, that solves two open questions in the literature: how to compute features for the former and how to infer tasks in the latter.

Beyond the concrete method proposed here, we believe bridging the gap between BMI and SFs is an insightful contribution that may inspire other useful methods.

For convenience, we can refer to maximizing F(??) as minimizing the loss function for parameters ?? = (?? ?? , ?? q ),

where ?? ?? and ?? q refer to the parameters of the policy ?? and variational approximation q, respectively.

We can minimize L ?? with respect to ?? q , the parameters of q, using back-propagation.

However, properly adjusting the parameters of ??, ?? ?? , is more difficult, as we lack a differentiable model of the environment.

We now show that we can still derive an appropriate score function estimator using the log-likelihood (Glynn, 1987) or REINFORCE trick (Williams, 1992) .

Since in this section we will be talking about ?? ?? only (that is, we will not discuss ?? q ), we will drop the subscript and refer to the parameters of ?? as simply ??.

Let ?? be a length T trajectory sampled under policy ??, and let p ?? be the probability of the trajectory ?? under the combination of the policy and environment transition probabilities.

We can compute the gradient of p ?? with respect to ?? as:

This means that we can adjust p ?? to make ?? more likely under it.

If we interpret p ?? as the distribution induced by the policy ??, then minimizing (13) corresponds to maximizing the following value function:

We can then use the policy gradient theorem to calculate the gradient of our loss function with respect to the parameters of the policy, ??, for trajectories ?? beginning in state s,

Since standard policy gradient (with rewards r t = log q(z|s t )) can be expressed as:

Figure 3: VISR features ?? learned by a variational distribution q(w|s) in a 10-by-10 gridworld.

we can conclude that log q(z | s) serves the role of the reward function and treat it as such for arbitrary reinforcement learning algorithms (n-step Q-learning is used throughout this paper).

The complexity and black-box nature of the Atari task suite make any significant analisis of the representations learned by VISR difficult (apart from their indirect effect on fastinference).

Thus, in order to analyze the representation learned by VISR we have conducted a much smaller-scale experiment on a standard 10-by-10 grid-world.

Here VISR still uses the full 5-sphere for its space of tasks, but it is trained with a much smaller network architecture for both the successor features ?? and variational approximation ?? (both consist of 2 fullyconnected layers of 100 units with ReLU non-linearities, the latter L2-normalized so as to make mean predictions on the 5-sphere).

We train this model for longer than necessary (960,000 trajectories of length 40 for 38,400,000 total steps) so as to best capture what representations might look like at convergence.

Figure 3 shows each of the 5 dimension of ?? across all states of the grid-world.

It should be noted that, since these states were observed as one-hot vectors, all of the structure present is the result of the mutual information training objective rather than any correlations in the input space.

Figure 4 shows 49 randomly sampled reward functions, generated by sampling a w vector uniformly on the 5-sphere and taking the inner product with ??.

This demonstrates that the space of ?? contains many different partitionings of the state-space, which lends credence to our claim that externally defined reward functions are likely to be not far outside of this space, and thus fast-inference can yield substantial benefits.

Figure 5 shows the 49 value functions corresponding to the reward function sampled in Figure  4 .

These value functions were computed via generalized policy improvement over the policies from 10 uniformly sampled w's.

The clear correspondance between these value functions and their respective reward functions demonstrate that even though VISR is tasked with learning an infinite space of value functions, it does not significantly suffer from underfitting.

These value functions can be thought of as the desired cumulative state-occupancies, and appear to represent distinct regions of the state space.

A distributed reinforcement learning setup was utilized to accelerate experimentation as per Espeholt et al. (2018) .

This involved having 100 separate actors, each running on its own instance of the environment.

After every roll-out of 40 steps, the experiences are added to a queue.

This queue is used by the centralized learner to calculate all of the losses and change the weights of the network, which are then passed back to the actors.

The roll-out length implicitly determines other hyper-parameters out of convenience, namely the amount of backpropagation through time is done before truncation (Werbos et al., 1990) , as the sequential structure of the data is lost outside of the roll-out window.

The task vector W is also resampled every 40 steps for similar reasons.

Under review as a conference paper at ICLR 2020 The approximations to the optimal value functions for the reward functions in Figure 4 , computed by VISR through GPI on 10 randomly sampled policies.

In all results (modulo some reported from other papers) are the average of 3 random seeds per game per condition.

Due to the high computational cost of the controlled fast-inference experiments, for the experiments comparing the effect of training steps on fast-inference performance (e.g. Figure 6 ), an online evaluation scheme was utilized.

Rather than actually performing no-reward reinforcement learning as 2 distinct phases, reward information 2 was exposed to 5 of the 100 actors which used the task vector resulting from solving the reward regression via OLS.

This regression was continuously solved using the most recent 100, 000 experiences from these actors.

@highlight

We introduce Variational Intrinsic Successor FeatuRes (VISR), a novel algorithm which learns controllable features that can be leveraged to provide fast task inference through the successor features framework.