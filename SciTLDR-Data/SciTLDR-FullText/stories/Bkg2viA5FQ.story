A reinforcement learning agent that needs to pursue different goals across episodes requires a goal-conditional policy.

In addition to their potential to generalize desirable behavior to unseen goals, such policies may also enable higher-level planning based on subgoals.

In sparse-reward environments, the capacity to exploit information about the degree to which an arbitrary goal has been achieved while another goal was intended appears crucial to enable sample efficient learning.

However, reinforcement learning agents have only recently been endowed with such capacity for hindsight.

In this paper, we demonstrate how hindsight can be introduced to policy gradient methods, generalizing this idea to a broad class of successful algorithms.

Our experiments on a diverse selection of sparse-reward environments show that hindsight leads to a remarkable increase in sample efficiency.

In a traditional reinforcement learning setting, an agent interacts with an environment in a sequence of episodes, observing states and acting according to a policy that ideally maximizes expected cumulative reward.

If an agent is required to pursue different goals across episodes, its goal-conditional policy may be represented by a probability distribution over actions for every combination of state and goal.

This distinction between states and goals is particularly useful when the probability of a state transition given an action is independent of the goal pursued by the agent.

Learning such goal-conditional behavior has received significant attention in machine learning and robotics, especially because a goal-conditional policy may generalize desirable behavior to goals that were never encountered by the agent BID17 BID3 Kupcsik et al., 2013; Deisenroth et al., 2014; BID16 BID29 Kober et al., 2012; Ghosh et al., 2018; Mankowitz et al., 2018; BID11 .

Consequently, developing goal-based curricula to facilitate learning has also attracted considerable interest (Fabisch & Metzen, 2014; Florensa et al., 2017; BID20 BID19 .

In hierarchical reinforcement learning, goal-conditional policies may enable agents to plan using subgoals, which abstracts the details involved in lower-level decisions BID10 BID26 Kulkarni et al., 2016; Levy et al., 2017) .In a typical sparse-reward environment, an agent receives a non-zero reward only upon reaching a goal state.

Besides being natural, this task formulation avoids the potentially difficult problem of reward shaping, which often biases the learning process towards suboptimal behavior BID9 .

Unfortunately, sparse-reward environments remain particularly challenging for traditional reinforcement learning algorithms BID0 Florensa et al., 2017) .

For example, consider an agent tasked with traveling between cities.

In a sparse-reward formulation, if reaching a desired destination by chance is unlikely, a learning agent will rarely obtain reward signals.

At the same time, it seems natural to expect that an agent will learn how to reach the cities it visited regardless of its desired destinations.

In this context, the capacity to exploit information about the degree to which an arbitrary goal has been achieved while another goal was intended is called hindsight.

This capacity was recently introduced by BID0 to off-policy reinforcement learning algorithms that rely on experience replay (Lin, 1992) .

In earlier work, Karkus et al. (2016) introduced hindsight to policy search based on Bayesian optimization BID5 .In this paper, we demonstrate how hindsight can be introduced to policy gradient methods BID27 BID28 BID22 , generalizing this idea to a successful class of reinforcement learning algorithms BID13 Duan et al., 2016) .In contrast to previous work on hindsight, our approach relies on importance sampling BID2 .

In reinforcement learning, importance sampling has been traditionally employed in order to efficiently reuse information obtained by earlier policies during learning BID15 BID12 Jie & Abbeel, 2010; BID7 .

In comparison, our approach attempts to efficiently learn about different goals using information obtained by the current policy for a specific goal.

This approach leads to multiple formulations of a hindsight policy gradient that relate to well-known policy gradient results.

In comparison to conventional (goal-conditional) policy gradient estimators, our proposed estimators lead to remarkable sample efficiency on a diverse selection of sparse-reward environments.

We denote random variables by upper case letters and assignments to these variables by corresponding lower case letters.

We let Val(X) denote the set of valid assignments to a random variable X. We also omit the subscript that typically relates a probability function to random variables when there is no risk of ambiguity.

For instance, we may use p(x) to denote p X (x) and p(y) to denote p Y (y).Consider an agent that interacts with its environment in a sequence of episodes, each of which lasts for exactly T time steps.

The agent receives a goal g ∈ Val(G) at the beginning of each episode.

At every time step t, the agent observes a state s t ∈ Val(S t ), receives a reward r(s t , g) ∈ R, and chooses an action a t ∈ Val(A t ).

For simplicity of notation, suppose that Val(G), Val(S t ), and Val(A t ) are finite for every t.

In our setting, a goal-conditional policy defines a probability distribution over actions for every combination of state and goal.

The same policy is used to make decisions at every time step.

Let τ = s 1 , a 1 , s 2 , a 2 , . . . , s T −1 , a T −1 , s T denote a trajectory.

We assume that the probability p(τ | g, θ) of trajectory τ given goal g and a policy parameterized by θ ∈ Val(Θ) is given by p(τ | g, θ) = p(s 1 ) T −1 t=1 p(a t | s t , g, θ)p(s t+1 | s t , a t ).(In contrast to a Markov decision process, this formulation allows the probability of a state transition given an action to change across time steps within an episode.

More importantly, it implicitly states that the probability of a state transition given an action is independent of the goal pursued by the agent, which we denote by S t+1 ⊥ ⊥ G | S t , A t .

For every τ , g, and θ, we also assume that p(τ | g, θ) is non-zero and differentiable with respect to θ.

Assuming that G ⊥ ⊥ Θ, the expected return η(θ) of a policy parameterized by θ is given by DISPLAYFORM0 T t=1 r(s t , g).The action-value function is given by Q

This section presents results for goal-conditional policies that are analogous to well-known results for conventional policies BID13 .

They establish the foundation for the results presented in the next section.

The corresponding proofs are included in Appendix A for completeness.

The objective of policy gradient methods is finding policy parameters that achieve maximum expected return.

When combined with Monte Carlo techniques BID2 , the following result allows pursuing this objective using gradient-based optimization.

Theorem 3.1 (Goal-conditional policy gradient).

The gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM0 The following result allows employing a baseline to reduce the variance of the gradient estimator.

Theorem 3.2 (Goal-conditional policy gradient, baseline formulation).

For every t, θ, and associated real-valued (baseline) function b θ t , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM1 Appendix A.7 presents the constant baselines that minimize the (elementwise) variance of the corresponding estimator.

However, such baselines are usually impractical to compute (or estimate), and the variance of the estimator may be reduced further by a baseline function that depends on state and goal.

Although generally suboptimal, it is typical to let the baseline function b θ t approximate the value function V θ t (Greensmith et al., 2004) .

Lastly, actor-critic methods may rely on the following result for goal-conditional policies.

Theorem 3.3 (Goal-conditional policy gradient, advantage formulation).

The gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM2

This section presents the novel ideas that introduce hindsight to policy gradient methods.

The corresponding proofs can be found in Appendix B.Suppose that the reward r(s, g) is known for every combination of state s and goal g, as in previous work on hindsight BID0 Karkus et al., 2016) .

In that case, it is possible to evaluate a trajectory obtained while trying to achieve an original goal g for an alternative goal g. Using importance sampling, this information can be exploited using the following central result.

Theorem 4.1 (Every-decision hindsight policy gradient).

For an arbitrary (original) goal g , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM0 In the formulation presented above, every reward is multiplied by the ratio between the likelihood of the corresponding trajectory under an alternative goal and the likelihood under the original goal (see Eq. 1).

Intuitively, every reward should instead be multiplied by a likelihood ratio that only considers the corresponding trajectory up to the previous action.

This intuition underlies the following important result, named after an analogous result for action-value functions by BID15 .Theorem 4.2 (Per-decision hindsight policy gradient).

For an arbitrary (original) goal g , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM1

This section details gradient estimation based on the results presented in the previous section.

The corresponding proofs can be found in Appendix C. DISPLAYFORM0 where each trajectory τ (i) is obtained using a policy parameterized by θ in an attempt to achieve a goal g (i) chosen by the environment.

The following result points to a straightforward estimator based on Theorem 4.2.

Theorem 5.1.

The per-decision hindsight policy gradient estimator, given by DISPLAYFORM1 is a consistent and unbiased estimator of the gradient ∇η(θ) of the expected return.

In preliminary experiments, we found that this estimator leads to unstable learning progress, which is probably due to its potential high variance.

The following result, inspired by weighted importance sampling BID2 , represents our attempt to trade variance for bias.

Theorem 5.2.

The weighted per-decision hindsight policy gradient estimator, given by DISPLAYFORM2 is a consistent estimator of the gradient ∇η(θ) of the expected return.

In simple terms, the likelihood ratio for every combination of trajectory, (alternative) goal, and time step is normalized across trajectories by this estimator.

In Appendix C.3, we present a result that enables the corresponding consistency-preserving weighted baseline.

t , g) = 0} composed of so-called active goals during the i-th episode.

The feasibility of the proposed estimators relies on the fact that only active goals correspond to non-zero terms inside the expectation over goals in Expressions 11 and 12.

In many natural sparse-reward environments, active goals will correspond directly to states visited during episodes (for instance, the cities visited while trying to reach other cities), which enables computing said expectation exactly when the goal distribution is known.

The proposed estimators have remarkable properties that differentiate them from previous (weighted) importance sampling estimators for off-policy learning.

For instance, although a trajectory is often more likely under the original goal than under an alternative goal, in policies with strong optimal substructure, a high probability of a trajectory between the state a and the goal (state) c that goes through the state b may naturally allow for a high probability of the corresponding (sub)trajectory between the state a and the goal (state) b. In other cases, the (unnormalized) likelihood ratios may become very small for some (alternative) goals after a few time steps across all trajectories.

After normalization, in the worst case, this may even lead to equivalent ratios for such goals for a given time step across all trajectories.

In any case, it is important to note that only likelihood ratios associated to active goals for a given episode will affect the gradient estimate.

Additionally, an original goal will always have (unnormalized) likelihood ratios equal to one for the corresponding episode.

Under mild additional assumptions, the proposed estimators also allow using a dataset containing goals chosen arbitrarily (instead of goals drawn from the goal distribution).

Although this feature is not required by our experiments, we believe that it may be useful to circumvent catastrophic forgetting during curriculum learning BID4 Kirkpatrick et al., 2017) .

This section reports results of an empirical comparison between goal-conditional policy gradient estimators and hindsight policy gradient estimators.1 Because there are no well-established sparsereward environments intended to test agents under multiple goals, this comparison focuses on our own selection of environments.

These environments are diverse in terms of stochasticity, state space dimensionality and size, relationship between goals and states, and number of actions.

In every one of these environments, the agent receives the remaining number of time steps plus one as a reward for reaching the goal state, which also ends the episode.

In every other situation, the agent receives no reward.

Importantly, the weighted per-decision hindsight policy gradient estimator used in our experiments (HPG) does not precisely correspond to Expression 12.

Firstly, the original estimator requires a constant number of time steps T , which would often require the agent to act after the end of an episode in the environments that we consider.

Secondly, although it is feasible to compute Expression 12 exactly when the goal distribution is known (as explained in Sec. 5), we sometimes subsample the sets of active goals per episode.

Furthermore, when including a baseline that approximates the value function, we again consider only active goals, which by itself generally results in an inconsistent estimator (HPG+B).

As will become evident in the following sections, these compromised estimators still lead to remarkable sample efficiency.

We assess sample efficiency through learning curves and average performance scores, which are obtained as follows.

After collecting a number of batches (composed of trajectories and goals), each of which enables one step of gradient ascent, an agent undergoes evaluation.

During evaluation, the agent interacts with the environment for a number of episodes, selecting actions with maximum probability according to its policy.

A learning curve shows the average return obtained during each evaluation step, averaged across multiple runs (independent learning procedures).

The curves presented in this text also include a 95% bootstrapped confidence interval.

The average performance is given by the average return across evaluation steps, averaged across runs.

During both training and evaluation, goals are drawn uniformly at random.

Note that there is no held-out set of goals for evaluation, since we are interested in evaluating sample efficiency instead of generalization.

For every combination of environment and batch size, grid search is used to select hyperparameters for each estimator according to average performance scores (after the corresponding standard deviation across runs is subtracted, as suggested by Duan et al. (2016) ).

Definitive results are obtained by using the best hyperparameters found for each estimator in additional runs.

In this section, we discuss definitive results for small (2) and medium (16) batch sizes.

More details about our experiments can be found in Appendices E.1 and E.2.

Appendix E.3 contains unabridged results, a supplementary empirical study of likelihood ratios (Appendix E.3.6), and an empirical comparison with hindsight experience replay (Appendix E.3.7).

In a bit flipping environment, the agent starts every episode in the same state (0, represented by k bits), and its goal is to reach a randomly chosen state.

The actions allow the agent to toggle (flip) each bit individually.

The maximum number of time steps is k + 1.

Despite its apparent simplicity, this environment is an ideal testbed for reinforcement learning algorithms intended to deal with sparse rewards, since obtaining a reward by chance is unlikely even for a relatively small k. BID0 employed a similar environment to evaluate their hindsight approach.

Figure 4 presents the learning curves for k = 8.

Goal-conditional policy gradient estimators with and without an approximate value function baseline (GCPG+B and GCPG, respectively) obtain excellent policies and lead to comparable sample efficiency.

HPG+B obtains excellent policies more than 400 batches earlier than these estimators, but its policies degrade upon additional training.

Additional experiments strongly suggest that the main cause of this issue is the fact that the value function baseline is still very poorly fit by the time that the policy exhibits desirable behavior.

In comparison, HPG obtains excellent policies as early as HPG+B, but its policies remain remarkably stable upon additional training.

The learning curves for k = 16 are presented in Figure 5 .

Clearly, both GCPG and GCPG+B are unable to obtain policies that perform better than chance, which is explained by the fact that they rarely incorporate reward signals during training.

Confirming the importance of hindsight, HPG leads to stable and sample efficient learning.

Although HPG+B also obtains excellent policies, they deteriorate upon additional training.

Similar results can be observed for a small batch size (see App.

E.3.3).

The average performance results documented in Appendix E.3.5 confirm that HPG leads to remarkable sample efficiency.

Importantly, Appendices E.3.1 and E.3.2 present hyperparameter sensitivity graphs suggesting that HPG is less sensitive to hyperparameter settings than the other estimators.

The same two appendices also document an ablation study where the likelihood ratios are removed from HPG, which notably promotes increased hyperparameter sensitivity.

This study confirms the usefulness of the correction prescribed by importance sampling.

In the grid world environments that we consider, the agent starts every episode in a (possibly random) position on an 11 × 11 grid, and its goal is to reach a randomly chosen (non-initial) position.

Some of the positions on the grid may contain impassable obstacles (walls).

The actions allow the agent to move in the four cardinal directions.

Moving towards walls causes the agent to remain in its current position.

A state or goal is represented by a pair of integers between 0 and 10.

The maximum number of time steps is 32.

In the empty room environment, the agent starts every episode in the upper left corner of the grid, and there are no walls.

In the four rooms environment BID23 , the agent starts every episode in one of the four corners of the grid (see FIG0 ).

There are walls that partition the grid into four rooms, such that each room provides access to two other rooms through single openings (doors).

With probability 0.2, the action chosen by the agent is ignored and replaced by a random action.

Figure 6 shows the learning curves for the empty room environment.

Clearly, every estimator obtains excellent policies, although HPG and HPG+B improve sample efficiency by at least 200 batches.

The learning curves for the four rooms environment are presented in Figure 7 .

In this surprisingly challenging environment, every estimator obtains unsatisfactory policies.

However, it is still clear that HPG and HPG+B improve sample efficiency.

In contrast to the experiments presented in the previous section, HPG+B does not give rise to instability, which we attribute to easier value function estimation.

Similar results can be observed for a small batch size (see App.

E.3.3).

HPG achieves the best average performance in every grid world experiment except for a single case, where the best average performance is achieved by HPG+B (see App.

E.3.5).

The hyperparameter sensitivity graphs presented in Appendices E.3.1 and E.3.2 once again suggest that HPG is less sensitive to hyperparameter choices, and that ignoring likelihood ratios promotes increased sensitivity (at least in the four rooms environment).

The Ms. Pac-man environment is a variant of the homonymous game for ATARI 2600 (see FIG1 ).

The agent starts every episode close to the center of the map, and its goal is to reach a randomly chosen (non-initial) position on a 14 × 19 grid defined on the game screen.

The actions allow the agent to move in the four cardinal directions for 13 game ticks.

A state is represented by the result of preprocessing a sequence of game screens (images) as described in Appendix E.1.

A goal is represented by a pair of integers.

The maximum number of time steps is 28, although an episode will also end if the agent is captured by an enemy.

In comparison to the grid world environments considered in the previous section, this environment is additionally challenging due to its high-dimensional states and the presence of enemies.

Figure 8 presents the learning curves for a medium batch size.

Approximate value function baselines are excluded from this experiment due to the significant cost of systematic hyperparameter search.

Although HPG obtains better policies during early training, GCPG obtains better final policies.

However, for such a medium batch size, only 3 active goals per episode (out of potentially 28) are subsampled for HPG.

Although this harsh subsampling brings computational efficiency, it also appears to handicap the estimator.

This hypothesis is supported by the fact that HPG outperforms GCPG for a small batch size, when all active goals are used (see Apps.

E.3.3 and E.3.5).

Policies obtained using each estimator are illustrated by videos included on the project website.

The FetchPush environment is a variant of the environment recently proposed by BID14 to assess goal-conditional policy learning algorithms in a challenging task of practical interest (see FIG2 ).

In a simulation, a robotic arm with seven degrees of freedom is required to push a randomly placed object (block) towards a randomly chosen position.

The arm starts every episode in the same configuration.

In contrast to the original environment, the actions in our variant allow increasing the desired velocity of the gripper along each of two orthogonal directions by ±0.1 or ±1, leading to a total of eight actions.

A state is represented by a 28-dimensional real vector that contains the following information: positions of the gripper and block; rotational and positional velocities of the gripper and block; relative position of the block with respect to the gripper; state of the gripper; and current desired velocity of the gripper along each direction.

A goal is represented by three coordinates.

The maximum number of time steps is 50.

Figure 9 presents the learning curves for a medium batch size.

HPG obtains good policies after a reasonable number of batches, in sharp contrast to GCPG.

For such a medium batch size, only 3 active goals per episode (out of potentially 50) are subsampled for HPG, showing that subsampling is a viable alternative to reduce the computational cost of hindsight.

Similar results are observed for a small batch size, when all active goals are used (see Apps.

E.3.3 and E.3.5).

Policies obtained using each estimator are illustrated by videos included on the project website.

We introduced techniques that enable learning goal-conditional policies using hindsight.

In this context, hindsight refers to the capacity to exploit information about the degree to which an arbitrary goal has been achieved while another goal was intended.

Prior to our work, hindsight has been limited to off-policy reinforcement learning algorithms that rely on experience replay BID0 and policy search based on Bayesian optimization (Karkus et al., 2016) .In addition to the fundamental hindsight policy gradient, our technical results include its baseline and advantage formulations.

These results are based on a self-contained goal-conditional policy framework that is also introduced in this text.

Besides the straightforward estimator built upon the per-decision hindsight policy gradient, we also presented a consistent estimator inspired by weighted importance sampling, together with the corresponding baseline formulation.

A variant of this estimator leads to remarkable comparative sample efficiency on a diverse selection of sparsereward environments, especially in cases where direct reward signals are extremely difficult to obtain.

This crucial feature allows natural task formulations that require just trivial reward shaping.

The main drawback of hindsight policy gradient estimators appears to be their computational cost, which is directly related to the number of active goals in a batch.

This issue may be mitigated by subsampling active goals, which generally leads to inconsistent estimators.

Fortunately, our experiments suggest that this is a viable alternative.

Note that the success of hindsight experience replay also depends on an active goal subsampling heuristic (Andrychowicz et al., 2017, Sec. 4.5) .The inconsistent hindsight policy gradient estimator with a value function baseline employed in our experiments sometimes leads to unstable learning, which is likely related to the difficulty of fitting such a value function without hindsight.

This hypothesis is consistent with the fact that such instability is observed only in the most extreme examples of sparse-reward environments.

Although our preliminary experiments in using hindsight to fit a value function baseline have been successful, this may be accomplished in several ways, and requires a careful study of its own.

Further experiments are also required to evaluate hindsight on dense-reward environments.

There are many possibilities for future work besides integrating hindsight policy gradients into systems that rely on goal-conditional policies: deriving additional estimators; implementing and evaluating hindsight (advantage) actor-critic methods; assessing whether hindsight policy gradients can successfully circumvent catastrophic forgetting during curriculum learning of goal-conditional policies; approximating the reward function to reduce required supervision; analysing the variance of the proposed estimators; studying the impact of active goal subsampling; and evaluating every technique on continuous action spaces.

Theorem A.1.

The gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM0 Proof.

The partial derivative ∂η(θ)/∂θ j of the expected return η(θ) with respect to θ j is given by DISPLAYFORM1 The likelihood-ratio trick allows rewriting the previous equation as DISPLAYFORM2 Note that DISPLAYFORM3 Therefore, DISPLAYFORM4 A.2 THEOREM 3.1Theorem 3.1 (Goal-conditional policy gradient).

The gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM5 Proof.

Starting from Eq. 17, the partial derivative ∂η(θ)/∂θ j of η(θ) with respect to θ j is given by DISPLAYFORM6 The previous equation can be rewritten as DISPLAYFORM7 Let c denote an expectation inside Eq. 19 for t ≥ t. In that case, A t ⊥ ⊥ S t | S t , G, Θ, and so DISPLAYFORM8 Reversing the likelihood-ratio trick, DISPLAYFORM9 Therefore, the terms where t ≥ t can be dismissed from Eq. 19, leading to DISPLAYFORM10 The previous equation can be conveniently rewritten as DISPLAYFORM11 A.3 LEMMA A.1Lemma A.1.

For every j, t, θ, and associated real-valued (baseline) function b DISPLAYFORM12 Proof.

Letting c denote an expectation inside Eq. 24, DISPLAYFORM13 Reversing the likelihood-ratio trick, DISPLAYFORM14 A.4 THEOREM 3.2 Theorem 3.2 (Goal-conditional policy gradient, baseline formulation).

For every t, θ, and associated real-valued (baseline) function b θ t , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM15 Proof.

The result is obtained by subtracting Eq. 24 from Eq. 23.

Importantly, for every combination of θ and t, it would also be possible to have a distinct baseline function for each parameter in θ.

A.5 LEMMA A.2 Lemma A.2.

The gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM16 Proof.

Starting from Eq. 23 and rearranging terms, DISPLAYFORM17 By the definition of action-value function, DISPLAYFORM18 A.6 THEOREM 3.3Theorem 3.3 (Goal-conditional policy gradient, advantage formulation).

The gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM19 Proof.

The result is obtained by choosing b θ t = V θ t and subtracting Eq. 24 from Eq. 29.A.7 THEOREM A.2For arbitrary j and θ, consider the following definitions of f and h. DISPLAYFORM20 DISPLAYFORM21 For every b j ∈ R, using Theorem 3.1 and the fact that DISPLAYFORM22 Proof.

The result is an application of Lemma D.4.

The following theorem relies on importance sampling, a traditional technique used to obtain estimates related to a random variable X ∼ p using samples from an arbitrary positive distribution q. This technique relies on the following equalities:

Theorem 4.1 (Every-decision hindsight policy gradient).

For an arbitrary (original) goal g , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM0 Proof.

Starting from Theorem 3.1, importance sampling allows rewriting the partial derivative ∂η(θ)/∂θ j as DISPLAYFORM1 B.2 THEOREM 4.2 Theorem 4.2 (Per-decision hindsight policy gradient).

For an arbitrary (original) goal g , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM2 Proof.

Starting from Eq. 36, the partial derivative ∂η(θ)/∂θ j can be rewritten as DISPLAYFORM3 If we split every trajectory into states and actions before and after t , then ∂η(θ)/∂θ j is given by g p(g) DISPLAYFORM4 where z is defined by DISPLAYFORM5

Using Lemma D.2 and canceling terms, DISPLAYFORM0 Using Lemma D.2 once again, DISPLAYFORM1 Using the fact that DISPLAYFORM2 Substituting z into Expression 38 and returning to an expectation over trajectories, DISPLAYFORM3 B.3 LEMMA 4.1Lemma 4.1.

For every g , t, θ, and associated real-valued (baseline) function b DISPLAYFORM4 Proof.

Let c denote the j-th element of the vector in the left-hand side of Eq. 8, such that DISPLAYFORM5 Using Lemma D.1 and writing the expectations explicitly, DISPLAYFORM6 Canceling terms, using Lemma D.1 once again, and reversing the likelihood-ratio trick, DISPLAYFORM7 Pushing constants outside the summation over actions at time step t, DISPLAYFORM8 Theorem B.1 (Hindsight policy gradient, baseline formulation).

For every g , t, θ, and associated real-valued (baseline) function b θ t , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM9 where DISPLAYFORM10 Proof.

The result is obtained by subtracting Eq. 8 from Eq. 7.

Importantly, for every combination of θ and t, it would also be possible to have a distinct baseline function for each parameter in θ.

Lemma B.1 (Hindsight policy gradient, action-value formulation).

For an arbitrary goal g , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM0 Proof.

Starting from Eq. 29, the partial derivative ∂η(θ)/∂θ j can be written as DISPLAYFORM1 Using importance sampling, for an arbitrary goal g , DISPLAYFORM2 Using Lemma D.1 and rewriting the previous equation using expectations, DISPLAYFORM3 B.6 THEOREM 4.3 Theorem 4.3 (Hindsight policy gradient, advantage formulation).

For an arbitrary (original) goal g , the gradient ∇η(θ) of the expected return with respect to θ is given by DISPLAYFORM4 Proof.

The result is obtained by choosing b θ t = V θ t and subtracting Eq. 44 from Eq. 53.

For arbitrary g , j, and θ, consider the following definitions of f and h. DISPLAYFORM0 DISPLAYFORM1 For every b j ∈ R, using Theorem 4.2 and the fact that E [h(T ) | g , θ] = 0 by Lemma 4.1, DISPLAYFORM2 is given by DISPLAYFORM3 .Proof.

The result is an application of Lemma D.4.

This appendix contains proofs related to the estimators presented in Section 5: Theorem 5.1 (App.

C.1) and Theorem 5.2 (App.

C.2).

Appendix C.3 presents a result that enables a consistency-preserving weighted baseline.

In this appendix, we will consider a dataset DISPLAYFORM0 where each trajectory τ (i) is obtained using a policy parameterized by θ in an attempt to achieve a goal g (i) chosen by the environment.

Because D is an iid dataset given Θ, DISPLAYFORM1 C.1 THEOREM 5.1Theorem 5.1.

The per-decision hindsight policy gradient estimator, given by DISPLAYFORM2 is a consistent and unbiased estimator of the gradient ∇η(θ) of the expected return.

Proof.

Let I (N ) j denote the j-th element of the estimator, which can be written as DISPLAYFORM3 where DISPLAYFORM4 Using Theorem 4.2, the expected value E I (N ) j | θ is given by DISPLAYFORM5 Therefore, I(N ) j is an unbiased estimator of ∂η(θ)/∂θ j .Conditionally on Θ, the random variable DISPLAYFORM6 is an average of iid random variables with expected value ∂η(θ)/∂θ j (see Eq. 61).

By the strong law of large numbers BID18 , Theorem 2.3.13), DISPLAYFORM7 Therefore, I(N ) j is a consistent estimator of ∂η(θ)/∂θ j .

Theorem 5.2.

The weighted per-decision hindsight policy gradient estimator, given by DISPLAYFORM0 , FORMULA2 is a consistent estimator of the gradient ∇η(θ) of the expected return.

Proof.

Let W (N ) j denote the j-th element of the estimator, which can be written as DISPLAYFORM1 where X(g, t, t ) DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 Consider the expected value DISPLAYFORM6

Using the fact that t > t, Lemma D.1, and canceling terms, E Xi can be written as DISPLAYFORM0 Because S t ⊥ ⊥ G | S 1:t −1 , A 1:t −1 , Θ, DISPLAYFORM1 Conditionally on Θ, the variable X(g, t, t ) DISPLAYFORM2 is an average of iid random variables with expected value E Xi .

By the strong law of large numbers BID18 , Theorem 2.3.13), X(g, t, t ) DISPLAYFORM3 Conditionally on Θ, the variable Y (g, t, t ) DISPLAYFORM4 is an average of iid random variables with expected value 1.

By the strong law of large numbers, Y (g, t, t ) (N ) j a.s.− − → 1.

and Y (g, t, t ) (N ) j converge almost surely to real numbers (Thomas, 2015, Ch.

3 , Property 2), DISPLAYFORM0 By Theorem 3.1 and the fact that W (N ) j is a linear combination of terms X(g, t, t ) DISPLAYFORM1 C.3 THEOREM C.1Theorem C.1.

The weighted baseline estimator, given by DISPLAYFORM2 , (74) converges almost surely to zero.

Proof.

Let B (N ) j denote the j-th element of the estimator, which can be written as DISPLAYFORM3 where DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 Using Eqs. 44 and 47, the expected value DISPLAYFORM8 Conditionally on Θ, the variable X(g, t) DISPLAYFORM9 is an average of iid random variables with expected value zero.

By the strong law of large numbers BID18 , Theorem 2.3.13), X(g, t) DISPLAYFORM10 The fact that Y (g, t)(N ) j a.s.− − → 1 is already established in the proof of Theorem 5.2.

Because both DISPLAYFORM11 and Y (g, t)(N ) j converge almost surely to real numbers (Thomas, 2015, Ch.

3 , Property 2), DISPLAYFORM12 Because B(N ) j is a linear combination of terms X(g, t) DISPLAYFORM13 is a consistent estimator of a some quantity given θ, then so is E (N ) − B DISPLAYFORM14 Proof.

In order to employ backward induction, consider the case t = T − 1.

By marginalization, DISPLAYFORM15 which completes the proof of the base case.

Assuming the inductive hypothesis is true for a given 2 ≤ t ≤ T − 1 and considering the case t − 1, DISPLAYFORM16 Lemma D.2.

For every τ , g, θ, and 1 ≤ t ≤ T , DISPLAYFORM17 Proof.

The case t = 1 can be inspected easily.

Consider 2 ≤ t ≤ T .

By definition, DISPLAYFORM18 Using Lemma D.1, DISPLAYFORM19 DISPLAYFORM20 DISPLAYFORM21 Proof.

From the definition of action-value function and using the fact that DISPLAYFORM22 Let z denote the second term in the right-hand side of the previous equation, which can also be written as DISPLAYFORM23 Consider the following three independence properties: DISPLAYFORM24 DISPLAYFORM25 DISPLAYFORM26 Together, these properties can be used to demonstrate that DISPLAYFORM27 From the definition of value function, DISPLAYFORM28 Theorem 4.4.

For every t and θ, the advantage function DISPLAYFORM29 Proof.

The result follows from the definition of advantage function and Lemma D.3.

DISPLAYFORM30 Consider a discrete random variable X and real-valued functions f and h. Suppose also that DISPLAYFORM31 Proof.

Let v = Var [f (X) − bh(X)].

Using our assumptions and the definition of variance, DISPLAYFORM32 The first and second derivatives of v with respect to b are given by dv DISPLAYFORM33 Our assumptions guarantee that E h(X) 2 > 0.

Therefore, by Fermat's theorem, if b is a local minimum, then dv/db = 0, leading to the desired equality.

By the second derivative test, b must be a local minimum.

This appendix contains additional information about the experiments introduced in Section 6.

Appendix E.1 details policy and baseline representations.

Appendix E.2 documents experimental settings.

Appendix E.3 presents unabridged results.

In every experiment, a policy is represented by a feedforward neural network with a softmax output layer.

The input to such a policy is a pair composed of state and goal.

A baseline function is represented by a feedforward neural network with a single (linear) output neuron.

The input to such a baseline function is a triple composed of state, goal, and time step.

The baseline function is trained to approximate the value function using the mean squared (one-step) temporal difference error BID21 .

Parameters are updated using Adam (Kingma & Ba, 2014) .

The networks are given by the following.

Bit flipping environments and grid world environments.

Both policy and baseline networks have two hidden layers, each with 256 hyperbolic tangent units.

Every weight is initially drawn from a Gaussian distribution with mean 0 and standard deviation 0.01 (and redrawn if far from the mean by two standard deviations), and every bias is initially zero.

Ms.

Pac-man environment.

The policy network is represented by a convolutional neural network.

The network architecture is given by a convolutional layer with 32 filters (8×8, stride 4); convolutional layer with 64 filters (4 × 4, stride 2); convolutional layer with 64 filters (3 × 3, stride 1); and three fully-connected layers, each with 256 units.

Every unit uses a hyperbolic tangent activation function.

Every weight is initially set using variance scaling (Glorot & Bengio, 2010) , and every bias is initially zero.

These design decisions are similar to the ones made by BID6 .A sequence of images obtained from the Arcade Learning Environment BID1 ) is preprocessed as follows.

Individually for each color channel, an elementwise maximum operation is employed between two consecutive images to reduce rendering artifacts.

Such 210 × 160 × 3 preprocessed image is converted to grayscale, cropped, and rescaled into an 84 × 84 image x t .

A sequence of images x t−12 , x t−8 , x t−4 , x t obtained in this way is stacked into an 84 × 84 × 4 image, which is an input to the policy network (recall that each action is repeated for 13 game ticks).

The goal information is concatenated with the flattened output of the last convolutional layer.

FetchPush environment.

The policy network has three hidden layers, each with 256 hyperbolic tangent units.

Every weight is initially set using variance scaling (Glorot & Bengio, 2010) , and every bias is initially zero.

Tables 1 and 2 document the experimental settings.

The number of runs, training batches, and batches between evaluations are reported separately for hyperparameter search and definitive runs.

The number of training batches is adapted according to how soon each estimator leads to apparent convergence.

Note that it is very difficult to establish this setting before hyperparameter search.

The number of batches between evaluations is adapted so that there are 100 evaluation steps in total.

Other settings include the sets of policy and baseline learning rates under consideration for hyperparameter search, and the number of active goals subsampled per episode.

In Tables 1 and 2 , R 1 = {α×10 −k | α ∈ {1, 5} and k ∈ {2, 3, 4, 5}} and R 2 = {β ×10 −5 | β ∈ {1, 2.5, 5, 7.5, 10}}.As already mentioned in Section 6, the definitive runs use the best combination of hyperparameters (learning rates) found for each estimator.

Every setting was carefully chosen during preliminary experiments to ensure that the best result for each estimator is representative.

In particular, the best performing learning rates rarely lie on the extrema of the corresponding search range.

In the single case where the best performing learning rate found by hyperparameter search for a goal-conditional policy gradient estimator was such an extreme value (FetchPush, for a small batch size), evaluating one additional learning rate lead to decreased average performance.

This appendix contains unabridged experimental results.

Appendices E.3.1 and E.3.2 present hyperparameter sensitivity plots for every combination of environment and batch size.

A hyperparameter sensitivity plot displays the average performance achieved by each hyperparameter setting (sorted from best to worst along the horizontal axis).

Appendices E.3.3 and E.3.4 present learning curves for every combination of environment and batch size.

Appendix E.3.5 presents average performance results.

Appendix E.3.6 presents an empirical study of likelihood ratios.

Appendix E.3.7 presents an empirical comparison with hindsight experience replay BID0 ).

hyperparameter setting (best to worst) This study is conveyed through plots that encode the distribution of active likelihood ratios computed during training, individually for each time step within an episode.

Each plot corresponds to an agent that employs HPG and obtains the highest definitive average performance for a given environment FIG2 .

Note that the length of the largest bar for a given time step is fixed to aid visualization.

The most important insight provided by these plots is that likelihood ratios behave very differently across environments, even for equivalent time steps (for instance, compare bit flipping environments to grid world environments).

In contrast, after the first time step, the behavior of likelihood ratios changes slowly across time steps within the same environment.

In any case, alternative goals have a significant effect on gradient estimates, which agrees with the results presented in Section 6.

This appendix documents an empirical comparison between goal-conditional policy gradients (GCPG), hindsight policy gradients (HPG), deep Q-networks (Mnih et al., 2015, DQN) , and a combination of DQN and hindsight experience replay (Andrychowicz et al., 2017, DQN+HER) .Experience replay.

Our implementations of both DQN and DQN+HER are based on OpenAI Baselines (Dhariwal et al., 2017) , and use mostly the same hyperparameters that BID0 used in their experiments on environments with discrete action spaces, all of which resemble our bit flipping environments.

The only notable differences in our implementations are the lack of both Polyak-averaging and temporal difference target clipping.

Concretely, a cycle begins when an agent collects a number of episodes FORMULA6 by following an -greedy policy derived from its deep Q-network ( = 0.2).

The corresponding transitions are included in a replay buffer, which contains at most 10 6 transitions.

In the case of DQN+HER, hindsight transitions derived from a final strategy are also included in this replay buffer.

Completing the cycle, for a total of 40 different batches, a batch composed of 128 transitions chosen at random from the replay buffer is used to define a loss function and allow one step of gradient-based minimization.

The targets required to define these loss functions are computed using a copy of the deep Q-network from the start of the corresponding cycle.

Parameters are updated using Adam (Kingma & Ba, 2014) .

A discount factor of γ = 0.98 is used, and seems necessary to improve the stability of both DQN and DQN+HER.Network architectures.

In every experiment, the deep Q-network is implemented by a feedforward neural network with a linear output neuron corresponding to each action.

The input to such a network is a triple composed of state, goal, and time step.

The network architectures are the same as those described in Appendix E.1, except that every weight is initially set using variance scaling (Glorot & Bengio, 2010) , and all hidden layers use rectified linear units BID8 .

For the Ms. Pac-man environment, the time step information is concatenated with the flattened output of the last convolutional layer (together with the goal information).

In comparison to the architecture employed by BID0 for environments with discrete action spaces, our architectures have one or two additional hidden layers (besides the convolutional architecture used for Ms. Pac-man).Experimental protocol.

The experimental protocol employed in our comparison is very similar to the one described in Section 6.

Each agent is evaluated periodically, after a number of cycles that depends on the environment.

During this evaluation, the agent collects a number of episodes by following a greedy policy derived from its deep Q-network.

For each environment, grid search is used to select the learning rates for both DQN and DQN+HER according to average performance scores (after the corresponding standard deviation across runs is subtracted, as in Section 6).

The candidate sets of learning rates are the following.

Bit flipping and grid world environments: {α × 10 −k | α ∈ {1, 5} and k ∈ {2, 3, 4, 5}}, FetchPush: {10 −2 , 5 × 10 −3 , 10 −3 , 5 × 10 −4 , 10 −4 }, Ms. Pac-man: {10 −3 , 5 × 10 −4 , 10 −4 , 5 × 10 −5 , 10 −5 }.

These sets were carefully chosen such that the best performing learning rates do not lie on their extrema.

Definitive results for a given environment are obtained by using the best hyperparameters found for each method in additional runs.

These definitive results are directly comparable to our previous results for GCPG and HPG (batch size 16), since every method will have interacted with the environment for the same number of episodes before each evaluation step.

For each environment, the number of runs, the number of training batches (cycles), the number of batches (cycles) between evaluations, and the number of episodes per evaluation step are the same as those listed in Tables 1 and 2 .Results.

The definitive results for the different environments are represented by learning curves Pg.

38) .

In the bit flipping environment for k = 8 (Figure 40 ), HPG and DQN+HER lead to equivalent sample efficiency, while GCPG lags far behind and DQN is completely unable to learn.

In the bit flipping environment for k = 16 FIG0 ), HPG surpasses DQN+HER in sample efficiency by a small margin, while both GCPG and DQN are completely unable to learn.

In the empty room environment FIG1 ), HPG is arguably the most sample efficient method, although DQN+HER is more stable upon obtaining a good policy.

GCPG eventually obtains a good policy, whereas DQN exhibits instability.

In the four rooms environment FIG2 ), DQN+HER outperforms all other methods by a large margin.

Although DQN takes much longer to obtain good policies, it would likely surpass both HPG and GCPG given additional training cycles.

In the Ms. Pac-man environment (Figure 44 ), DQN+HER once again outperforms all other methods, which achieve equivalent sample efficiency (although DQN appears unstable by the end of training).

In the FetchPush environment FIG3 , HPG dramatically outperforms all other methods.

Both DQN+HER and DQN are completely unable to learn, while GCPG appears to start learning by the end of the training process.

Note that active goals are harshly subsampled to increase the computational efficiency of HPG for both Ms. Pac-man and FetchPush (see Sec. 6.3 and Sec. 6.4).Discussion.

Our results suggest that the decision between applying HPG or DQN+HER in a particular sparse-reward environment requires experimentation.

In contrast, the decision to apply hindsight was always successful.

Note that we have not employed heuristics that are known to sometimes increase the performance of policy gradient methods (such as entropy bonuses, reward scaling, learning rate annealing, and simple statistical baselines) to avoid introducing confounding factors.

We believe that such heuristics would allow both GCPG and HPG to achieve good results in both the four rooms environment and Ms. Pac-man.

Furthermore, whereas hindsight experience replay is directly applicable to state-of-the-art techniques, our work can probably benefit from being extended to state-of-the-art policy gradient approaches, which we intend to explore in future work.

Similarly, we believe that additional heuristics and careful hyperparameter settings would allow DQN+HER to achieve good results in the FetchPush environment.

This is evidenced by the fact that BID0 achieve good results using the deep deterministic policy gradient (Lillicrap et al., 2016, DDPG) in a similar environment (with a continuous action space and a different reward function).

The empirical comparisons between either GCPG and HPG or DQN and DQN+HER are comparatively more conclusive, since the similarities between the methods minimize confounding factors.

Regardless of these empirical results, policy gradient approaches constitute one of the most important classes of model-free reinforcement learning methods, which by itself warrants studying how they can benefit from hindsight.

Our approach is also complementary to previous work, since it is entirely possible to combine a critic trained by hindsight experience replay with an actor that employs hindsight policy gradients.

Although hindsight experience replay does not require a correction analogous to importance sampling, indiscriminately adding hindsight transitions to the replay buffer is problematic, which has mostly been tackled by heuristics (Andrychowicz et al., 2017, Sec. 4.5) .

In contrast, our approach seems to benefit from incorporating all available information about goals at every update, which also avoids the need for a replay buffer.

@highlight

We introduce the capacity to exploit information about the degree to which an arbitrary goal has been achieved while another goal was intended to policy gradient methods.