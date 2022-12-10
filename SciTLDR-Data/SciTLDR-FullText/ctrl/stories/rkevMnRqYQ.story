Reinforcement learning (RL) agents optimize only the features specified in a reward function and are indifferent to anything left out inadvertently.

This means that we must not only specify what to do, but also the much larger space of what not to do.

It is easy to forget these preferences, since these preferences are already satisfied in our environment.

This motivates our key insight: when a robot is deployed in an environment that humans act in, the state of the environment is already optimized for what humans want.

We can therefore use this implicit preference information from the state to fill in the blanks.

We develop an algorithm based on Maximum Causal Entropy IRL and use it to evaluate the idea in a suite of proof-of-concept environments designed to show its properties.

We find that information from the initial state can be used to infer both side effects that should be avoided as well as preferences for how the environment should be organized.

Our code can be found at https://github.com/HumanCompatibleAI/rlsp.

Deep reinforcement learning (deep RL) has been shown to succeed at a wide variety of complex tasks given a correctly specified reward function.

Unfortunately, for many real-world tasks it can be challenging to specify a reward function that captures human preferences, particularly the preference for avoiding unnecessary side effects while still accomplishing the goal BID2 .

As a result, there has been much recent work BID8 BID13 Sadigh et al., 2017) that aims to learn specifications for tasks a robot should perform.

Typically when learning about what people want and don't want, we look to human action as evidence: what reward they specify BID14 , how they perform a task (Ziebart et al., 2010; BID13 , what choices they make BID8 Sadigh et al., 2017) , or how they rate certain options BID9 .

Here, we argue that there is an additional source of information that is potentially rather helpful, but that we have been ignoring thus far:The key insight of this paper is that when a robot is deployed in an environment that humans have been acting in, the state of the environment is already optimized for what humans want.

For example, consider an environment in which a household robot must navigate to a goal location without breaking any vases in its path, illustrated in FIG5 .

The human operator, Alice, asks the robot to go to the purple door, forgetting to specify that it should also avoid breaking vases along the way.

However, since the robot has been deployed in a state that only contains unbroken vases, it can infer that while acting in the environment (prior to robot's deployment), Alice was using one of the relatively few policies that do not break vases, and so must have cared about keeping vases intact.

Figure 1: An illustration of learning preferences from an initial state.

Alice attempts to accomplish a goal in an environment with an easily breakable vase in the center.

The robot observes the state of the environment, s 0 , after Alice has acted for some time from an even earlier state s −T .

It considers multiple possible human reward functions, and infers that states where vases are intact usually occur when Alice's reward penalizes breaking vases.

In contrast, it doesn't matter much what the reward function says about carpets, as we would observe the same final state either way.

Note that while we consider a specific s −T for clarity here, the robot could also reason using a distribution over s −T .The initial state s 0 can contain information about arbitrary preferences, including tasks that the robot should actively perform.

For example, if the robot observes a basket full of apples near an apple tree, it can reasonably infer that Alice wants to harvest apples.

However, s 0 is particularly useful for inferring which side effects humans care about.

Recent approaches avoid unnecessary side effects by penalizing changes from an inaction baseline (Krakovna et al., 2018; Turner, 2018) .

However, this penalizes all side effects.

The inaction baseline is appealing precisely because the initial state has already been optimized for human preferences, and action is more likely to ruin s 0 than inaction.

If our robot infers preferences from s 0 , it can avoid negative side effects while allowing positive ones.

This work is about highlighting the potential of this observation, and as such makes unrealistic assumptions, such as known dynamics and hand-coded features.

Given just s 0 , these assumptions are necessary: without dynamics, it is hard to tell whether some feature of s 0 was created by humans or not.

Nonetheless, we are optimistic that these assumptions can be relaxed, so that this insight can be used to improve deep RL systems.

We suggest some approaches in our discussion.

Our contributions are threefold.

First, we identify the state of the world at initialization as a source of information about human preferences.

Second, we leverage this insight to derive an algorithm, Reward Learning by Simulating the Past (RLSP), which infers reward from initial state based on a Maximum Causal Entropy (Ziebart et al., 2010) model of human behavior.

Third, we demonstrate the properties and limitations of RLSP on a suite of proof-of-concept environments: we use it to avoid side effects, as well as to learn implicit preferences that require active action.

In FIG5 the robot moves to the purple door without breaking the vase, despite the lack of a penalty for breaking vases.

Preference learning.

Much recent work has learned preferences from different sources of data, such as demonstrations (Ziebart et al., 2010; Ramachandran and Amir, 2007; BID15 BID13 BID12 ), comparisons (Christiano et al., 2017 Sadigh et al., 2017; Wirth et al., 2017) , ratings BID9 , human reinforcement signals (Knox and Stone, 2009; Warnell et al., 2017; MacGlashan et al., 2017) , proxy rewards BID14 , etc.

We suggest preference learning with a new source of data: the state of the environment when the robot is first deployed.

It can also be seen as a variant of Maximum Causal Entropy Inverse Reinforcement Learning (Ziebart et al., 2010) : while inverse reinforcement learning (IRL) requires demonstrations, or at least state sequences without actions Yu et al., 2018) , we learn a reward function from a single state, albeit with the simplifying assumption of known dynamics.

This can also be seen as an instance of IRL from summary data BID17 .Frame properties.

The frame problem in AI (McCarthy and Hayes, 1981) refers to the issue that we must specify what stays the same in addition to what changes.

In formal verification, this manifests as a requirement to explicitly specify the many quantities that the program does not change BID3 .

Analogously, rewards are likely to specify what to do (the task), but may forget to say what not to do (the frame properties).

One of our goals is to infer frame properties automatically.

Side effects.

An impact penalty can mitigate reward specification problems, since it penalizes unnecessary "large" changes BID5 .

We could penalize a reduction in the number of reachable states (Krakovna et al., 2018) or attainable utility (Turner, 2018) .

However, such approaches will penalize all irreversible effects, including ones that humans want.

In contrast, by taking a preference inference approach, we can infer which effects humans care about.

Goal states as specifications.

Desired behavior in RL can be specified with an explicitly chosen goal state BID16 Schaul et al., 2015; Nair et al., 2018; BID6 BID4 .

In our setting, the robot observes the initial state s 0 where it starts acting, which is not explicitly chosen by the designer, but nonetheless contains preference information.

A finite-horizon Markov decision process (MDP) is a tuple M = S, A, T , r, T , where S is the set of states, A is the set of actions, T : S × A × S → [0, 1] is the transition probability function, r : S → R is the reward function, and T ∈ Z + is the finite planning horizon.

We consider MDPs where the reward is linear in features, and does not depend on action: r(s; θ) = θ T f (s), where θ are the parameters defining the reward function and f computes features of a given state.

Inverse Reinforcement Learning (IRL).

In IRL, the aim is to infer the reward function r given an MDP without reward M\r and expert demonstrations D = {τ 1 , ..., τ n }, where each τ i = (s 0 , a 0 , ..., s T , a T ) is a trajectory sampled from the expert policy acting in the MDP.

It is assumed that each τ i is feasible, so that T (s j+1 | s j , a j ) > 0 for every j.

Maximum Causal Entropy IRL (MCEIRL).

As human demonstrations are rarely optimal, Ziebart et al. (2010) models the expert as a Boltzmann-rational agent that maximizes total reward and causal entropy of the policy.

This leads to the policy π t (a | s, θ) = exp(Q t (s, a; θ) − V t (s; θ)), where V t (s; θ) = ln a exp(Q t (s, a; θ)) plays the role of a normalizing constant.

Intuitively, the expert is assumed to act close to randomly when the difference in expected total reward across the actions is small, but nearly always chooses the best action when it leads to a substantially higher expected return.

The soft Bellman backup for the state-action value function Q is the same as usual, and is given by DISPLAYFORM0 The likelihood of a trajectory τ given the reward parameters θ is: DISPLAYFORM1 MCEIRL finds the reward parameters θ * that maximize the log-likelihood of the demonstrations: DISPLAYFORM2 DISPLAYFORM3 where p(τ 0 | θ) is given in Equation 1.

We could invert this and sample from p(θ | s 0 ); the resulting algorithm is presented in Appendix C, but is relatively noisy and slow.

We instead find the MLE: DISPLAYFORM4 Solution.

Similarly to MCEIRL, we use a gradient ascent algorithm to solve the IRL from one state problem.

We explain the key steps here and give the full derivation in Appendix B. First, we express the gradient in terms of the gradients of trajectories: DISPLAYFORM5 This has a nice interpretation -compute the Maximum Causal Entropy gradients for each trajectory, and then take their weighted sum, where each weight is the probability of the trajectory given the evidence s 0 and current reward θ.

We derive the exact gradient for a trajectory instead of the approximate one in Ziebart et al. FORMULA1 in Appendix A and substitute it in to get: DISPLAYFORM6 where we have suppressed the dependence on θ for readability.

F t (s t ) denotes the expected features when starting at s t at time t and acting until time 0 under the policy implied by θ.

Since we combine gradients from simulated past trajectories, we name our algorithm Reward Learning by Simulating the Past (RLSP).

The algorithm computes the gradient using dynamic programming, detailed in Appendix B.

We can easily incorporate a prior on θ by adding the gradient of the log prior to the gradient in Equation 5.

Evaluation of RLSP is non-trivial.

The inferred reward is very likely to assign state s 0 maximal reward, since it was inferred under the assumption that when Alice optimized the reward she ended up at s 0 .

If the robot then starts in state s 0 , if a no-op action is available (as it often is), the RLSP reward is likely to incentivize no-ops, which is not very interesting.

Ultimately, we hope to use RLSP to correct badly specified instructions or reward functions.

So, we created a suite of environments with a true reward R true , a specified reward R spec , Alice's first state s −T , and the robot's initial state s 0 , where R spec ignores some aspect(s) of R true .

RLSP is used to infer a reward θ Alice from s 0 , which is then combined with the specified reward to get a final reward θ final = θ Alice + λθ spec .

(We considered another method for combining rewards; see Appendix D for details.)

We inspect the inferred reward qualitatively and measure the expected amount of true reward obtained when planning with θ final , as a fraction of the expected true reward from the optimal policy.

We tune the hyperparameter λ controlling the tradeoff between R spec and the human reward for all algorithms, including baselines.

We use a Gaussian prior over the reward parameters.

Specified reward policy π spec .

We act as if the true reward is exactly the specified reward.

Policy that penalizes deviations π deviation .

This baseline minimizes change by penalizing deviations from the observed features DISPLAYFORM0 Relative reachability policy π reachability .

Relative reachability (Krakovna et al., 2018) considers a change to be negative when it decreases coverage, relative to what would have happened had the agent done nothing.

Here, coverage is a measure of how easily states can be reached from the current state.

We compare against the variant of relative reachability that uses undiscounted coverage and a baseline policy where the agent takes no-op actions, as in the original paper.

Relative reachability requires known dynamics but not a handcoded featurization.

A version of relative reachability that operates in feature space instead of state space would behave similarly.

We compare RLSP to our baselines with the assumption of known s −T , because it makes it easier to analyze RLSP's properties.

We consider the case of unknown s −T in Section 5.3.

We summarize the results in TAB1 , and show the environments and trajectories in Figure 2 .

DISPLAYFORM0 Figure 2: Evaluation of RLSP on our environments.

Silhouettes indicate the initial position of an object or agent, while filled in version indicate their positions after an agent has acted.

The first row depicts the information given to RLSP.

The second row shows the trajectory taken by the robot when following the policy π spec that is optimal for θ spec .

The third row shows the trajectory taken when following the policy π RLSP that is optimal for θ final = θ Alice + λθ spec .

(a) Side effects: Room with vase (b) Distinguishing environment effects: Toy train (c) Implicit reward: Apple collection (d) Desirable side effect: Batteries (e) "Unseen" side effect: Room with far away vase.

Side effects: Room with vase ( Figure 2a ).

The room tests whether the robot can avoid breaking a vase as a side effect of going to the purple door.

There are features for the number of broken vases, standing on a carpet, and each door location.

Since Alice didn't walk over the vase, RLSP infers a negative reward on broken vases, and a small positive reward on carpets (since paths to the top door usually involve carpets).

So, π RLSP successfully avoids breaking the vase.

The penalties also achieve the desired behavior: π deviation avoids breaking the vase since it would change the "number of broken vases" feature, while relative reachability avoids breaking the vase since doing so would result in all states with intact vases becoming unreachable.

Distinguishing environment effects: Toy train (Figure 2b ).

To test whether algorithms can distinguish between effects caused by the agent and effects caused by the environment, as suggested in Krakovna et al. FORMULA1 , we add a toy train that moves along a predefined track.

The train breaks if the agent steps on it.

We add a new feature indicating whether the train is broken and new features for each possible train location.

As before, the specified reward only has a positive weight on the purple door, while the true reward also penalizes broken trains and vases.

RLSP infers a negative reward on broken vases and broken trains, for the same reason as before.

It also infers not to put any weight on any particular train location, even though it changes frequently, because it doesn't help explain s 0 .

As a result, π RLSP walks over a carpet, but not a vase or a train.

π deviation immediately breaks the train to keep the train location the same.

π reachability deduces that breaking the train is irreversible, and so follows the same trajectory as π RLSP .Implicit reward: Apple collection (Figure 2d ).

This environment tests whether the algorithms can learn tasks implicit in s 0 .

There are three trees that grow apples, as well as a basket for collecting apples, and the goal is for the robot to harvest apples.

However, the specified reward is zero: the robot must infer the task from the observed state.

We have features for the number of apples in baskets, the number of apples on trees, whether the robot is carrying an apple, and each location that the agent could be in.

s 0 has two apples in the basket, while s −T has none.π spec is arbitrary since every policy is optimal for the zero reward.

π deviation does nothing, achieving zero reward, since its reward can never be positive.

π reachability also does not harvest apples.

RLSP infers a positive reward on apples in baskets, a negative reward for apples on trees, and a small positive reward for carrying apples.

Despite the spurious weights, π RLSP harvests apples as desired.

Desirable side effect: Batteries (Figure 2c ).

This environment tests whether the algorithms can tell when a side effect is allowed.

We take the toy train environment, remove vases and carpets, and add batteries.

The robot can pick up batteries and put them into the (now unbreakable) toy train, but the batteries are never replenished.

If the train runs for 10 timesteps without a new battery, it stops operating.

There are features for the number of batteries, whether the train is operational, each train location, and each door location.

There are two batteries at s −T but only one at s 0 .

The true reward incentivizes an operational train and being at the purple door.

We consider two variants for the task reward -an "easy" case, where the task reward equals the true reward, and a "hard" case, where the task reward only rewards being at the purple door.

Unsurprisingly, π spec succeeds at the easy case, and fails on the hard case by allowing the train to run out of power.

Both π deviation and π reachability see the action of putting a battery in the train as a side effect to be penalized, and so neither can solve the hard case.

They penalize picking up the batteries, and so only solve the easy case if the penalty weight is small.

RLSP sees that one battery is gone and that the train is operational, and infers that Alice wants the train to be operational and doesn't want batteries (since a preference against batteries and a preference for an operational train are nearly indistinguishable).

So, it solves both the easy and the hard case, with π RLSP picking up the battery, then staying at the purple door except to deliver the battery to the train."Unseen" side effect: Room with far away vase (Figure 2e ).

This environment demonstrates a limitation of our algorithm: it cannot identify side effects that Alice would never have triggered.

In this room, the vase is nowhere close to the shortest path from the Alice's original position to her goal, but is on the path to the robot's goal.

Since our baselines don't care about the trajectory the human takes, they all perform as before: π spec walks over the vase, while π deviation and π reachability both avoid it.

Our method infers a near zero weight on the broken vase feature, since it is not present on any reasonable trajectory to the goal, and so breaks it when moving to the goal.

Note that this only applies when Alice is known to be at the bottom left corner at s −T : if we have a uniform prior over s −T (considered in Section 5.3) then we do consider trajectories where vases are broken.

Side effects: Room with vase ( Figure 2a ) and toy train (Figure 2b ).

In both room with vase and toy train, RLSP learns a smaller negative reward on broken vases when using a uniform prior.

This is because RLSP considers many more feasible trajectories when using a uniform prior, many of which do not give Alice a chance to break the vase, as in Room with far away vase in Section 5.2.

In room with vase, the small positive reward on carpets changes to a near-zero negative reward on carpets.

With known s −T , RLSP overfits to the few consistent trajectories, which usually go over carpets, whereas with a uniform prior it considers many more trajectories that often don't go over carpets, and so it correctly infers a near-zero weight.

In toy train, the negative reward on broken trains becomes slightly more negative, while other features remain approximately the same.

This may be because when Alice starts out closer to the toy train, she has more of an opportunity to break it, compared to the known s −T case.

Implicit preference: Apple collection (Figure 2d ).

Here, a uniform prior leads to a smaller positive weight on the number of apples in baskets compared to the case with known s −T .

Intuitively, this is because RLSP is considering cases where s −T already has one or two apples in the basket, which implies that Alice has collected fewer apples and so must have been less interested in them.

States where the basket starts with three or more apples are inconsistent with the observed s 0 and so are not considered.

Following the inferred reward still leads to good apple harvesting behavior.

Desirable side effects: Batteries (Figure 2c ).

With the uniform prior, we see the same behavior as in Apple collection, where RLSP with a uniform prior learns a slightly smaller negative reward on the batteries, since it considers states s −T where the battery was already gone.

In addition, due to the particular setup the battery must have been given to the train two timesteps prior, which means that in any state where the train started with very little charge, it was allowed to die even though a battery could have been provided before, leading to a near-zero positive weight on the train losing charge.

Despite this, RLSP successfully delivers the battery to the train in both easy and hard cases."Unseen" side effect: Room with far away vase (Figure 2e ).

With a uniform prior, we "see" the side effect: if Alice started at the purple door, then the shortest trajectory to the black door would break a vase.

As a result, π RLSP successfully avoids the vase (whereas it previously did not).

Here, uncertainty over the initial state s −T can counterintuitively improve the results, because it increases the diversity of trajectories considered, which prevents RLSP from "overfitting" to the few trajectories consistent with a known s −T and s 0 .Overall, RLSP is quite robust to the use of a uniform prior over s −T , suggesting that we do not need to be particularly careful in the design of that prior.

We investigate how RLSP performs when assuming the wrong value of Alice's planning horizon T .

We vary the value of T assumed by RLSP, and report the true return achieved by π RLSP obtained using the inferred reward and a fixed horizon for the robot to act.

For this experiment, we used a uniform prior over s −T , since with known s −T , RLSP often detects that the given s −T and s 0 are incompatible (when T is misspecified).

The results are presented in FIG1 .

The performance worsens when RLSP assumes that Alice had a smaller planning horizon than she actually had.

Intuitively, if we assume that Alice has only taken one or two actions ever, then even if we knew the actions they could have been in service of many goals, and so we end up quite uncertain about Alice's reward.

When the assumed T is larger than the true horizon, RLSP correctly infers things the robot should not do.

Knowing that the vase was not broken for longer than T timesteps is more evidence to suspect that Alice cared about not breaking the vase.

However, overestimated T leads to worse performance at inferring implicit preferences, as in the Apples environment.

If we assume Alice has only collected two apples in 100 timesteps, she must not have cared about them much, since she could have collected many more.

The batteries environment is unusual -assuming that Alice has been acting for 100 timesteps, the only explanation for the observed s 0 is that Alice waited until the 98th timestep to put the battery into the train.

This is not particularly consistent with any reward function, and performance degrades.

Overall, T is an important parameter and needs to be set appropriately.

However, even when T is misspecified, performance degrades gracefully to what would have happened if we optimized θ spec by itself, so RLSP does not hurt.

In addition, if T is larger than it should be, then RLSP still tends to accurately infer parts of the reward that specify what not to do.

Summary.

Our key insight is that when a robot is deployed, the state that it observes has already been optimized to satisfy human preferences.

This explains our preference for a policy that generally avoids side effects.

We formalized this by assuming that Alice has been acting in the environment prior to the robot's deployment.

We developed an algorithm, RLSP, that computes a MAP estimate of Alice's reward function.

The robot then acts according to a tradeoff between Alice's reward function and the specified reward function.

Our evaluation showed that information from the initial state can be used to successfully infer side effects to avoid as well as tasks to complete, though there are cases in which we cannot infer the relevant preferences.

While we believe this is an important step forward, there is still much work to be done to make this accurate and practical.

Realistic environments.

The primary avenue for future work is to scale to realistic environments, where we cannot enumerate states, we don't know dynamics, and the reward function may be nonlinear.

This could be done by adapting existing IRL algorithms BID13 BID15 BID12 .

Unknown dynamics is particularly challenging, since we cannot learn dynamics from a single state observation.

While acting in the environment, we would have to learn a dynamics model or an inverse dynamics model that can be used to simulate the past, and update the learned preferences as our model improves over time.

Alternatively, if we use unsupervised skill learning BID0 BID11 Nair et al., 2018) or exploration BID7 , or learn a goal-conditioned policy (Schaul et al., 2015; BID4 , we could compare the explored states with the observed s 0 .Hyperparameter choice.

While our evaluation showed that RLSP is reasonably robust to the choice of planning horizon T and prior over s −T , this may be specific to our gridworlds.

In the real world, we often make long term hierarchical plans, and if we don't observe the entire plan (corresponding to a choice of T that is too small) it seems possible that we infer bad rewards, especially if we have an uninformative prior over s −T .

We do not know whether this will be a problem, and if so how bad it will be, and hope to investigate it in future work with more realistic environments.

Conflicts between θ spec and θ Alice .

RLSP allows us to infer θ Alice from s 0 , which we must somehow combine with θ spec to produce a reward θ final for the robot to optimize.

θ Alice will usually prefer the status quo of keeping the state similar to s 0 , while θ spec will probably incentivize some change to the state, leading to conflict.

We traded off between the two by optimizing their sum, but future work could improve upon this.

For example, θ Alice could be decomposed into θ Alice,task , which says which task Alice is performing ("go to the black door"), and θ frame , which consists of the frame conditions ("don't break vases").

The robot then optimizes θ frame + λθ spec .

This requires some way of performing the decomposition.

We could model the human as pursuing multiple different subgoals, or the environment as being created by multiple humans with different goals.

θ frame would be shared, while θ task would vary, allowing us to distinguish between them.

However, combination may not be the answer -instead, perhaps the robot ought to use the inferred reward to inform Alice of any conflicts and actively query her for more information, along the lines of BID1 .Learning tasks to perform.

The apples and batteries environments demonstrate that RLSP can learn preferences that require the robot to actively perform a task.

It is not clear that this is desirable, since the robot may perform an inferred task instead of the task Alice explicitly sets for it.

Preferences that are not a result of human optimization.

While the initial state is optimized for human preferences, this may not be a result of human optimization, as assumed in this paper.

For example, we prefer that the atmosphere contain oxygen for us to breathe.

The atmosphere meets this preference in spite of human action, and so RLSP would not infer this preference.

While this is of limited relevance for household robots, it may become important for more capable AI systems.

et al. (2010) , as the existing approximation is insufficient for our purposes.

Given a trajectory τ T = s 0 a 0 . . .

s T a T , we seek the gradient ∇ θ ln p(τ T ).

We assume that the expert has been acting according to the maximum causal entropy IRL model given in Section 3 (where we have dropped θ from the notation for clarity): DISPLAYFORM0 In the following, unless otherwise specified, all expectations over states and actions use the probability distribution over trajectories from the above model, starting from the state and action just prior.

For example, DISPLAYFORM1 In addition, for all probability distributions over states and actions, we drop the dependence on θ for readability, so the probability of reaching state s T is written as p(s T ) instead of p(s T | θ).First, we compute the gradient of V t (s).

We have ∇ θ V T +1 (s) = 0, and for 0 ≤ t ≤ T : DISPLAYFORM2 Unrolling the recursion, we get that the gradient is the expected feature counts under the policy implied by θ from s t onwards, which we could prove using induction.

Define: DISPLAYFORM3 f (s t ) .Then we have: DISPLAYFORM4 We can now calculate the gradient we actually care about: DISPLAYFORM5 The last term of the summation is f ( DISPLAYFORM6 , so we can drop it.

Thus, our gradient is: DISPLAYFORM7 This is the gradient we will use in Appendix B, but a little more manipulation allows us to compare with the gradient in Ziebart et al. (2010) .

We reintroduce the terms that we cancelled above: DISPLAYFORM8 Ziebart et al. (2010) states that the gradient is given by the expert policy feature expectations minus the learned policy feature expectations, and in practice uses the feature expectations from demonstrations to approximate the expert policy feature expectations.

Assuming we have N trajectories {τ i }, the gradient would be DISPLAYFORM9 The first term matches our first term exactly.

Our second term matches the second term in the limit of sufficiently many trajectories, so that the starting states s 0 follow the distribution p(s 0 ).

Our third term converges to zero with sufficiently many trajectories, since any s t , a t pair in a demonstration will be present sufficiently often that the empirical counts of s t+1 will match the expected proportions prescribed by T (· | s t , a t ).In a deterministic environment, we have T (s t+1 | s t , a t ) = 1[s t+1 = s t+1 ] since only one transition is possible.

Thus, the third term is zero and even for one trajectory the gradient reduces to In a stochastic environment, the third term need not be zero, and corrects for the "bias" in the observed states s t+1 .

Intuitively, when the expert chose action a t , she did not know which next state s t+1 would arise, but the first term of our gradient upweights the particular next state s t+1 that we observed.

The third term downweights the future value of the observed state and upweights the future value of all other states, all in proportion to their prior probability T (s t+1 | s t , a t ).

This section provides a derivation of the gradient ∇ θ ln p(s 0 ), which is needed to solve argmax θ ln p(s 0 ) with gradient ascent.

We provide the results first as a quick reference: DISPLAYFORM0 Base cases: first, p(s −T ) is given, second, G −T (s −T ) = 0, and third, DISPLAYFORM1 For the derivation, we start by expressing the gradient in terms of gradients of trajectories, so that we can use the result from Appendix A. Note that, by inspecting the final form of the gradient in Appendix A, we can see that ∇ θ p(τ −T :0 ) is independent of a 0 .

Then, we have: DISPLAYFORM2 This has a nice interpretation -compute the gradient for each trajectory and take the weighted sum, where each weight is the probability of the trajectory given the evidence s 0 and current reward θ.

We can rewrite the gradient in Equation 6 as ∇ θ ln p(τ T ) =T −1 t=0 g(s t , a t ), where DISPLAYFORM3 We can now substitute this to get: Note that we can compute p(s t ) since we are given the distribution p(s −T ) and we can use the recursive rule p(s t+1 ) = st,at p(s t )π t (a t | s t )T (s t+1 | s t , a t ).

DISPLAYFORM4 In order to compute g(s t , a t ) we need to compute F t (s t ), which has base case F 0 (s 0 ) = f (s 0 ) and recursive rule: Comparing this to the recursive rule, for the base case we can set G −T (s −T ) = 0.

DISPLAYFORM5

Instead of estimating the MLE (or MAP if we have a prior) using RLSP, we could approximate the entire posterior distribution.

One standard way to address the computational challenges involved with the continuous and high-dimensional nature of θ is to use MCMC sampling to sample from p(θ | s 0 ) ∝ p(s 0 | θ)p(θ).

The resulting algorithm resembles Bayesian IRL (Ramachandran and Amir, 2007) and is presented in Algorithm 1.While this algorithm is less efficient and noisier than RLSP, it gives us an estimate of the full posterior distribution.

In our experiments, we collapsed the full distribution into a point estimate by taking the mean.

Initial experiments showed that the algorithm was slower and noisier than the gradientbased RLSP, so we did not test it further.

However, in future work we could better leverage the full distribution, for example to create risk-averse policies, to identify features that are uncertain, or to identify features that are certain but conflict with the specified reward, after which we could actively query Alice for more information.

append θ to the list of samples 12: until have generated the desired number of samples

<|TLDR|>

@highlight

When a robot is deployed in an environment that humans have been acting in, the state of the environment is already optimized for what humans want, and we can use this to infer human preferences.

@highlight

The authors propose to augment the explicitly stated reward function of an RL agent with auxiliary rewards/costs inferred from the initial state and a model of the state dynamics

@highlight

This work proposes a way to infer the implicit information in the initial state using IRL and combine the inferred reward with a specified reward.