We study the problem of representation learning in goal-conditioned hierarchical reinforcement learning.

In such hierarchical structures, a higher-level controller solves tasks by iteratively communicating goals which a lower-level policy is trained to reach.

Accordingly, the choice of representation -- the mapping of observation space to goal space -- is crucial.

To study this problem, we develop a notion of sub-optimality of a representation, defined in terms of expected reward of the optimal hierarchical policy using this representation.

We derive expressions which bound the sub-optimality and show how these expressions can be translated to representation learning objectives which may be optimized in practice.

Results on a number of difficult continuous-control tasks show that our approach to representation learning yields qualitatively better representations as well as quantitatively better hierarchical policies, compared to existing methods.

Hierarchical reinforcement learning has long held the promise of extending the successes of existing reinforcement learning (RL) methods BID9 BID22 BID16 to more complex, difficult, and temporally extended tasks BID18 BID25 BID3 .

Recently, goal-conditioned hierarchical designs, in which higher-level policies communicate goals to lower-levels and lower-level policies are rewarded for reaching states (i.e. observations) which are close to these desired goals, have emerged as an effective paradigm for hierarchical RL BID17 ; BID14 ; Vezhnevets et al. (2017) , inspired by earlier work BID6 ; BID21 ).

In this hierarchical design, representation learning is crucial; that is, a representation function must be chosen mapping state observations to an abstract space.

Goals (desired states) are then specified by the choice of a point in this abstract space.

Previous works have largely studied two ways to choose the representation: learning the representation end-to-end together with the higher-and lower-level policies (Vezhnevets et al., 2017) , or using the state space as-is for the goal space (i.e., the goal space is a subspace of the state space) BID17 BID14 .

The former approach is appealing, but in practice often produces poor results (see BID17 and our own experiments), since the resulting representation is under-defined; i.e., not all possible sub-tasks are expressible as goals in the space.

On the other hand, fixing the representation to be the full state means that no information is lost, but this choice is difficult to scale to higher dimensions.

For example, if the state observations are entire images, the higher-level must output target images for the lower-level, which can be very difficult.

We instead study how unsupervised objectives can be used to train a representation that is more concise than the full state, but also not as under-determined as in the end-to-end approach.

In order to do so in a principled manner, we propose a measure of sub-optimality of a given representation.

This measure aims to answer the question: How much does using the learned representation in place of the full representation cause us to lose, in terms of expected reward, against the optimal policy?

This question is important, because a useful representation will compress the state, hopefully making the learning problem easier.

At the same time, the compression might cause the representation to lose information, making the optimal policy impossible to express.

It is therefore critical to understand how lossy a learned representation is, not in terms of reconstruction, but in terms of the ability to represent near-optimal policies on top of this representation.

Our main theoretical result shows that, for a particular choice of representation learning objective, we can learn representations for which the return of the hierarchical policy approaches the return of the optimal policy within a bounded error.

This suggests that, if the representation is learned with a principled objective, the 'lossy-ness' in the resulting representation should not cause a decrease in overall task performance.

We then formulate a representation learning approach that optimizes this bound.

We further extend our result to the case of temporal abstraction, where the higher-level controller only chooses new goals at fixed time intervals.

To our knowledge, this is the first result showing that hierarchical goal-setting policies with learned representations and temporal abstraction can achieve bounded sub-optimality against the optimal policy.

We further observe that the representation learning objective suggested by our theoretical result closely resembles several other recently proposed objectives based on mutual information (van den Oord et al., 2018; BID12 , suggesting an intriguing connection between mutual information and goal representations for hierarchical RL.

Results on a number of difficult continuous-control navigation tasks show that our principled representation learning objective yields good qualitative and quantitative performance compared to existing methods.

Following previous work BID17 , we consider a two-level hierarchical policy on an MDP M = (S, A, R, T ), in which the higher-level policy modulates the behavior of a lowerlevel policy by choosing a desired goal state and rewarding the lower-level policy for reaching this state.

While prior work has used a sub-space of the state space as goals BID17 , in more general settings, some type of state representation is necessary.

That is, consider a state representation function f : S ??? R d .

A two-level hierarchical policy on M is composed of a higher-level policy ?? hi (g|s), where g ??? G = R d is the goal space, that samples a high-level action (or goal) g t ??? ?? hi (g|s t ) every c steps, for fixed c. A non-stationary, goal-conditioned, lower-level policy ?? lo (a|s t , g t , s t+k , k) then translates these high-level actions into low-level actions a t+k ??? A for k ??? [0, c ??? 1].

The process is then repeated, beginning with the higher-level policy selecting another goal according to s t+c .

The policy ?? lo is trained using a goal-conditioned reward; e.g. the reward of a transition g, s, s is ???D(f (s ), g), where D is a distance function.

In this work we adopt a slightly different interpretation of the lower-level policy and its relation to ?? hi .

Every c steps, the higher-level policy chooses a goal g t based on a state s t .

We interpret this state-goal pair as being mapped to a nonstationary policy ??(a|s t+k , k), ?? ??? ??, where ?? denotes the set of all possible c-step policies acting on M. We use ?? to denote this mapping from S ?? G to ??. In other words, on every c th step, we encounter some state s t ??? S.

We use the higher-level policy to sample a goal g t ??? ?? hi (g|s t ) and translate this to a policy ?? t = ??(s t , g t ).

We then use ?? t to sample actions a t+k ??? ?? t (a|s t+k , k) for k ??? [0, c ??? 1].

The process is then repeated from s t+c .Although the difference in this interpretation is subtle, the introduction of ?? is crucial for our subsequent analysis.

The communication of g t is no longer as a goal which ?? hi desires to reach, but rather more precisely, as an identifier to a low-level behavior which ?? hi desires to induce or activate.

The mapping ?? is usually expressed as the result of an RL optimization over ??; e.g., DISPLAYFORM0 where we use P ?? (s t+k |s t ) to denote the probability of being in state s t+k after following ?? for k steps starting from s t .

We will consider variations on this low-level objective in later sections.

From Equation 1 it is clear how the choice of representation f affects ?? (albeit indirectly).We will restrict the environment reward function R to be defined only on states.

We use R max to denote the maximal absolute reward: R max = sup S |R(s)|.

In the previous section, we introduced two-level policies where a higher-level policy ?? hi chooses goals g, which are translated to lower-level behaviors via ??. The introduction of this hierarchy leads to a natural question: How much do we lose by learning ?? hi which is only able to act on M via ???

The choice of ?? restricts the type and number of lower-level behaviors that the higher-level policy can induce.

Thus, the optimal policy on M is potentially not expressible by ?? hi .

Despite the potential lossy-ness of ??, can one still learn a hierarchical policy which is near-optimal?To approach this question, we introduce a notion of sub-optimality with respect to the form of ??: Let ?? * hi (g|s, ??) be the optimal higher-level policy acting on G and using ?? as the mapping from G to low-level behaviors.

Let ?? * hier be the corresponding full hierarchical policy on M. We will compare ?? * hier to an optimal hierarchical policy ?? * agnostic to ??. To define ?? * we begin by introducing an optimal higher-level policy ?? * * hi (??|s) agnostic to ??; i.e. every c steps, ?? * * hi samples a low-level behavior ?? ??? ?? which is applied to M for the following c steps.

In this way, ?? * * hi may express all possible low-level behaviors.

We then denote ?? * as the full hierarchical policy resulting from ?? * * hi .

We would like to compare ?? * hier to ?? * .

A natural and common way to do so is in terms of state values.

Let V ?? (s) be the future value achieved by a policy ?? starting at state s. We define the sub-optimality of ?? as DISPLAYFORM0 The state values V ?? * hier (s) are determined by the form of ??, which is in turn determined by the choice of representation f .

However, none of these relationships are direct.

It is unclear how a change in f will result in a change to the sub-optimality.

In the following section, we derive a series of bounds which establish a more direct relationship between SubOpt(??) and f .

Our main result will show that if one defines ?? as a slight modification of the traditional objective given in Equation 1, then one may translate sub-optimality of ?? to a practical representation learning objective for f .

In this section, we provide proxy expressions that bound the sub-optimality induced by a specific choice of ??. Our main result is Claim 4, which connects the sub-optimality of ?? to both goalconditioned policy objectives (i.e., the objective in 1) and representation learning (i.e., an objective for the function f ).

For ease of presentation, we begin by presenting our results in the restricted case of c = 1 and deterministic lower-level policies.

In this setting, the class of low-level policies ?? may be taken to be simply A, where a ??? ?? corresponds to a policy which always chooses action a. There is no temporal abstraction: The higher-level policy chooses a high-level action g ??? G at every step, which is translated via ?? to a low-level action a ??? A. Our claims are based on quantifying how many of the possible low-level behaviors (i.e., all possible state to state transitions) can be produced by ?? for different choices of g. To quantify this, we make use of an auxiliary inverse goal model ??(s, a), which aims to predict which goal g will cause ?? to yield an action?? = ??(s, g) that induces a next state distribution P (s |s,??) similar to P (s |s, a).3 We have the following theorem, which bounds the sub-optimality in terms of total variation divergences between P (s |s, a) and P (s |s,??): DISPLAYFORM0 DISPLAYFORM1 Proof.

See Appendices A and B for all proofs.

Theorem 1 allows us to bound the sub-optimality of ?? in terms of how recoverable the effect of any action in A is, in terms of transition to the next state.

One way to ensure that effects of actions in A are recoverable is to have an invertible ??. That is, if there exists ?? : S ?? A ??? G such that ??(s, ??(s, a)) = a for all s, a, then the sub-optimality of ?? is 0.However, in many cases it may not be desirable or feasible to have an invertible ??. Looking back at Theorem 1, we emphasize that its statement requires only the effect of any action to be recoverable.

That is, for any s, ??? S, a ??? A, we require only that there exist some g ??? G (given by ??(s, a)) which yields a similar next-state distribution.

To this end, we have the following claim, which connects the sub-optimality of ?? to both representation learning and the form of the low-level objective.

Claim 2.

Let ??(s) be a prior and f, ?? be so that, for DISPLAYFORM2 If the low-level objective is defined as DISPLAYFORM3 then the sub-optimality of ?? is bounded by C .We provide an intuitive explanation of the statement of Claim 2.

First, consider that the distribution K(s |s, a) appearing in Equation 4 may be interpreted as a dynamics model determined by f and ??. By bounding the difference between the true dynamics P (s |s, a) and the dynamics K(s |s, a) implied by f and ??, Equation FORMULA4 states that the representation f should be chosen in such a way that dynamics in representation space are roughly given by ??(s, a).

This is essentially a representation learning objective for choosing f , and in Section 5 we describe how to optimize it in practice.

Moving on to Equation 5, we note that the form of ?? here is only slightly different than the onestep form of the standard goal-conditioned objective in Equation FORMULA0 .

5 Therefore, all together Claim 2 establishes a deep connection between representation learning (Equation 4), goal-conditioned policy learning (Equation 5), and sub-optimality.

We now move on to presenting the same results in the fully general, temporally abstracted setting, in which the higher-level policy chooses a high-level action g ??? G every c steps, which is transformed via ?? to a c-step lower-level behavior policy ?? ??? ??. In this setting, the auxiliary inverse goal model ??(s, ??) is a mapping from S ?? ?? to G and aims to predict which goal g will cause ?? to yield a policy?? = ??(s, g) that induces future state distributions P??(s t+k |s t ) similar to P ?? (s t+k |s t ), for k ??? [1, c].

We weight the divergences between the distributions by weights w k = 1 for k < c and DISPLAYFORM0 The analogue to Theorem 1 is as follows: DISPLAYFORM1 then SubOpt(??) ??? C , where C = 2?? 1????? c R max w.

For the analogue to Claim 2, we simply replace the single-step KL divergences and low-level rewards with a discounted weighted sum thereof:Claim 4.

Let ??(s) be a prior over S. Let f, ?? be such that, DISPLAYFORM2 DISPLAYFORM3 If the low-level objective is defined as DISPLAYFORM4 then the sub-optimality of ?? is bounded by C .Claim 4 is the main theoretical contribution of our work.

As in the previous claim, we have a strong statement, saying that if the low-level objective is defined as in Equation 9, then minimizing the sub-optimality may be done by optimizing a representation learning objective based on Equation 8.

We emphasize that Claim 4 applies to any class of low-level policies ??, including either closed-loop or open-loop policies.

We now have the mathematical foundations necessary to learn representations that are provably good for use in hierarchical RL.

We begin by elaborating on how we translate Equation 8 into a practical training objective for f and auxiliary ?? (as well as a practical parameterization of policies ?? as input to ??).

We then continue to describe how one may train a lower-level policy to match the objective presented in Equation FORMULA10 .

In this way, we may learn f and lower-level policy to directly optimize a bound on the sub-optimality of ??. A pseudocode of the full algorithm is presented in the Appendix (see Algorithm 1).

Consider a representation function f ?? : S ??? R d and an auxiliary function ?? ?? : S ?? ?? ??? R d , parameterized by vector ??.

In practice, these are separate neural networks: DISPLAYFORM0 While the form of Equation 8 suggests to optimize a supremum over all s t and ??, in practice we only have access to a replay buffer which stores experience s 0 , a 0 , s 1 , a 1 , . . .

sampled from our hierarchical behavior policy.

Therefore, we propose to choose s t sampled uniformly from the replay buffer and use the subsequent c actions a t:t+c???1 as a representation of the policy ??, where we use a t:t+c???1 to denote the sequence a t , . . .

, a t+c???1 .

Note that this is equivalent to setting the set of candidate policies ?? to A c (i.e., ?? is the set of c-step, deterministic, open-loop policies).

This choice additionally simplifies the possible structure of the function approximator used for ?? ?? (a standard neural net which takes in s t and a t:t+c???1 ).

Our proposed representation learning objective is thus, DISPLAYFORM1 where J(??, s t , a t:t+c???1 ) will correspond to the inner part of the supremum in Equation 8.We now define the inner objective J(??, s t , a t:t+c???1 ).

To simplify notation, we use DISPLAYFORM2 suggests the following learning objective on each s t , ?? ??? a t:t+c???1 : DISPLAYFORM3 DISPLAYFORM4 where B is a constant.

The gradient with respect to ?? is then, DISPLAYFORM5 Published as a conference paper at ICLR 2019The first term of Equation 14 is straightforward to estimate using experienced s t+1:t+k .

We set ?? to be the replay buffer distribution, so that the numerator of the second term is also straightforward.

We approximate the denominator of the second term using a mini-batch S of states independently sampled from the replay buffer: DISPLAYFORM6 This completes the description of our representation learning algorithm.

Connection to Mutual Information Estimators.

The form of the objective we optimize (i.e. Equation 13) is very similar to mutual information estimators, mostly CPC (van den Oord et al., 2018) .

Indeed, one may interpret our objective as maximizing a mutual information M I(s t+k ; s t , ??) via an energy function given by E ?? (s t+k , s t , ??).

The main differences between our approach and these previous proposals are as follows: (1) Previous approaches maximize a mutual information M I(s t+k ; s t ) agnostic to actions or policy.

(2) Previous approaches suggest to define the energy function as exp(f (s t+k ) T M k f (s t )) for some matrix M k , whereas our energy function is based on the distance D used for low-level reward.

(3) Our approach is provably good for use in hierarchical RL, and hence our theoretical results may justify some of the good performance observed by others using mutual information estimators for representation learning.

Different approaches to translating our theoretical findings to practical implementations may yield objectives more or less similar to CPC, some of which perform better than others (see Appendix D).

Equation 9 suggests to optimize a policy ?? st,g (a|s t+k , k) for every s t , g. This is equivalent to the parameterization ?? lo (a|s t , g, s t+k , k), which is standard in goal-conditioned hierarchical designs.

Standard RL algorithms may be employed to maximize the low-level reward implied by Equation 9: DISPLAYFORM0 weighted by w k and where ?? corresponds to ?? lo when the state s t and goal g are fixed.

While the first term of Equation 16 is straightforward to compute, the log probabilities log ??(s t+k ), log P ?? (s t+k |s t ) are in general unknown.

To approach this issue, we take advantage of the representation learning objective for f, ??. When f, ?? are optimized as dictated by Equation 8, we have DISPLAYFORM1 We may therefore approximate the low-level reward as DISPLAYFORM2 As in Section 5.1, we use the sampled actions a t:t+c???1 to represent ?? as input to ??. We approximate the third term of Equation 18 analogously to Equation 15.

Note that this is a slight difference from standard low-level rewards, which use only the first term of Equation 18 and are unweighted.

Representation learning for RL has a rich and diverse existing literature, often interpreted as an abstraction of the original MDP.

Previous works have interpreted the hierarchy introduced in hierarchical RL as an MDP abstraction of state, action, and temporal spaces BID25 BID8 BID26 BID2 .

In goal-conditioned hierarchical designs, although the representation is learned on states, it is in fact a form of action abstraction (since goals g are high-level actions).

While previous successful applications of goal-conditioned hierarchical designs have either learned representations naively end-to-end (Vezhnevets et al., 2017), or not learned them at all BID14 BID17 , we take a principled approach to representation learning in hierarchical RL, translating a bound on sub-optimality to a practical learning objective.

Bounding sub-optimality in abstracted MDPs has a long history, from early work in theoretical analysis on approximations to dynamic programming models (Whitt, 1978; BID4 .

Extensive theoretical work on state abstraction, also known as state aggregation or model minimization, has been done in both operational research BID20 Van Roy, 2006) and RL BID7 BID19 BID0 .

Notably, BID15 introduce a formalism for categorizing classic work on state abstractions such as bisimulation BID7 and homomorphism BID19 based on what information is preserved, which is similar in spirit to our approach.

Exact state abstractions BID15 incur no performance loss BID7 BID19 , while their approximate variants generally have bounded sub-optimality BID4 BID7 BID23 BID0 .

While some of the prior work also focuses on learning state abstractions BID15 BID23 BID0 , they often exclusively apply to simple MDP domains as they rely on techniques such as state partitioning or Q-value based aggregation, which are difficult to scale to our experimented domains.

Thus, the key differentiation of our work from these prior works is that we derive bounds which may be translated to practical representation learning objectives.

Our impressive results on difficult continuous-control, high-dimensional domains is a testament to the potential impact of our theoretical findings.

Lastly, we note the similarity of our representation learning algorithm to recently introduced scalable mutual information maximization objectives such as CPC (van den Oord et al., 2018) and MINE BID12 .

This is not a surprise, since maximizing mutual information relates closely with maximum likelihood learning of energy-based models, and our bounds effectively correspond to bounds based on model-based predictive errors, a basic family of bounds in representation learning in MDPs BID23 BID5 BID0 .

Although similar information theoretic measures have been used previously for exploration in RL BID24 , to our knowledge, no prior work has connected these mutual information estimators to representation learning in hierarchical RL, and ours is the first to formulate theoretical guarantees on sub-optimality of the resulting representations in such a framework.

We evaluate our proposed representation learning objective compared to a number of baselines:??? XY: The oracle baseline which uses the x, y position of the agent as the representation.??? VAE: A variational autoencoder (Kingma & Welling, 2013) on raw observations.??? E2C: Embed to control (Watter et al., 2015) .

A method which uses variational objectives to train a representation of states and actions which have locally linear dynamics.??? E2E: End-to-end learning of the representation.

The representation is fed as input to the higher-level policy and learned using gradients from the RL objective.??? Whole obs: The raw observation is used as the representation.

No representation learning.

This is distinct from BID17 , in which a subset of the observation space was pre-determined for use as the goal space.

Figure 2: Learned representations (2D embeddings) of our method and a number of variants on a MuJoCo Ant Maze environment, with color gradient based on episode time-step (black for beginning of episode, yellow for end).

The ant travels from beginning to end of a ???-shaped corridor along an x, y trajectory shown under XY.

Without any supervision, our method is able to deduce this nearideal representation, even when the raw observation is given as a top-down image.

Other approaches are unable to properly recover a good representation.

Figure 3: Results of our method and a number of variants on a suite of tasks in 10M steps of training, plotted according to median over 10 trials with 30 th and 70 th percentiles.

We find that outside of simple point environments, our method is the only one which can approach the performance of oracle x, y representations.

These results show that our method can be successful, even when the representation is learned online concurrently while learning a hierarchical policy.

We evaluate on the following continuous-control MuJoCo (Todorov et al., 2012) tasks (see Appendix C for details):??? Ant (or Point) Maze: An ant (or point mass) must navigate a ???-shaped corridor.??? Ant Push: An ant must push a large block to the side to reach a point behind it.??? Ant Fall: An ant must push a large block into a chasm so that it may walk over it to the other side without falling.??? Ant Block: An ant must push a small block to various locations in a square room.??? Ant Block Maze: An ant must push a small block through a ???-shaped corridor.

In these tasks, the raw observation is the agent's x, y coordinates and orientation as well as local coordinates and orientations of its limbs.

In the Ant Block and Ant Block Maze environments we also include the x, y coordinates and orientation of the block.

We also experiment with more difficult raw representations by replacing the x, y coordinates of the agent with a low-resolution 5 ?? 5 ?? 3 top-down image of the agent and its surroundings.

These experiments are labeled 'Images'.For the baseline representation learning methods which are agnostic to the RL training (VAE and E2C), we provide comparative qualitative results in Figure 2 .

These representations are the result Ant and block Ant pushing small block through corridor Representations Figure 4 : We investigate importance of various observation coordinates in learned representations on a difficult block-moving task.

In this task, a simulated robotic ant must move a small red block from beginning to end of a ???-shaped corridor.

Observations include both ant and block x, y coordinates.

We show the trajectory of the learned representations on the right (cyan).

At four time steps, we also plot the resulting representations after perturbing the observation's ant coordinates (green) or the observation's block coordinates (magenta).

The learned representations put a greater emphasis (i.e., higher sensitivity) on the block coordinates, which makes sense for this task as the external reward is primarily determined by the position of the block.of taking a trained policy, fixing it, and using its sampled experience to learn 2D representations of the raw observations.

We find that our method can successfully deduce the underlying near-optimal x, y representation, even when the raw observation is given as an image.

We provide quantitative results in Figure 3 .

In these experiments, the representation is learned concurrently while learning a full hierarchical policy (according to the procedure in BID17 ).

Therefore, this setting is especially difficult since the representation learning must learn good representations even when the behavior policy is very far from optimal.

Accordingly, we find that most baseline methods completely fail to make any progress.

Only our proposed method is able to approach the performance of the XY oracle.

For the 'Block' environments, we were curious what our representation learning objective would learn, since the x, y coordinate of the agent is not the only near-optimal representation.

For example, another suitable representation is the x, y coordinates of the small block.

To investigate this, we plotted ( Figure 4 ) the trajectory of the learned representations of a successful policy (cyan), along with the representations of the same observations with agent x, y perturbed (green) or with block x, y perturbed (magenta).

We find that the learned representations greatly emphasize the block x, y coordinates over the agent x, y coordinates, although in the beginning of the episode, there is a healthy mix of the two.

We have presented a principled approach to representation learning in hierarchical RL.

Our approach is motivated by the desire to achieve maximum possible return, hence our notion of sub-optimality is in terms of optimal state values.

Although this notion of sub-optimality is intractable to optimize directly, we are able to derive a mathematical relationship between it and a specific form of representation learning.

Our resulting representation learning objective is practical and achieves impressive results on a suite of high-dimensional, continuous-control tasks.

We thank Bo Dai, Luke Metz, and others on the Google Brain team for insightful comments and discussions.

A PROOF OF THEOREM 3 (GENERALIZATION OF THEOREM 1)Consider the sub-optimality with respect to a specific state s 0 , V DISPLAYFORM0 Recall that ?? * is the hierarchical result of a policy ?? *

* hi : S ??? ???(??), and note that ?? * * hi may be assumed to be deterministic due to the Markovian nature of M. We may use the mapping ?? to transform ?? * * hi to a high-level policy ?? hi on G and using the mapping ??: DISPLAYFORM1 Let ?? hier be the corresponding hierarchical policy.

We will bound the quantity V DISPLAYFORM2 hier (s 0 ).

We follow logic similar to BID1 and begin by bounding the total variation divergence between the ??-discounted state visitation frequencies of the two policies.

Denote the k-step state transition distributions using either ?? * or ?? hier as, DISPLAYFORM3 DISPLAYFORM4 for k ??? [1, c].

Considering P * , P hier as linear operators, we may express the state visitation frequencies d * , d hier of ?? * , ?? hier , respectively, as DISPLAYFORM5 DISPLAYFORM6 where ?? is a Dirac ?? distribution centered at s 0 and DISPLAYFORM7 DISPLAYFORM8 We will use d c * , dc hier to denote the every-c-steps ??-discounted state frequencies of ?? * , ?? hier ; i.e., DISPLAYFORM9 DISPLAYFORM10 By the triangle inequality, we have the following bound on the total variation divergence |d hier ???d * |: DISPLAYFORM11 We begin by attacking the first term of Equation 28.

We note that DISPLAYFORM12 Thus the first term of Equation 28 is bounded by DISPLAYFORM13 ???1 as a geometric series and employing the triangle inequality, we have DISPLAYFORM14 , and we thus bound the whole quantity (30) by DISPLAYFORM15 We now move to attack the second term of Equation 28.

We may express this term as DISPLAYFORM16 Furthermore, by the triangle inequality we have DISPLAYFORM17 Therefore, recalling w k = 1 for k < c and w k = (1 ??? ??) ???1 for k = c, we may bound the total variation of the state visitation frequencies as DISPLAYFORM18 By condition 7 of Theorem 3 we have, DISPLAYFORM19 We now move to considering the difference in values.

We have DISPLAYFORM20 DISPLAYFORM21 Therefore, we have DISPLAYFORM22 as desired.

Consider a specific s t , ??.

Let K(s |s t , ??) ??? ??(s ) exp(???D(f (s ), ??(s t , ??))).

Note that the definition of ??(s t , ??(s t , ??)) may be expressed in terms of a KL: DISPLAYFORM0 Therefore we have, DISPLAYFORM1 By condition 8 we have, DISPLAYFORM2 Jensen's inequality on the sqrt function then implies DISPLAYFORM3 Pinsker's inequality now yields, DISPLAYFORM4 Similarly Jensen's and Pinsker's inequality on the LHS of Equation 43 yields DISPLAYFORM5 The triangle inequality and Equations 46 and 47 then give us, DISPLAYFORM6 as desired.

C EXPERIMENTAL DETAILS

The environments for Ant Maze, Ant Push, and Ant Fall are as described in BID17 .

During training, target (x, y) locations are selected randomly from all possible points in the environment (in Ant Fall, the target includes a z coordinate as well).

Final results are evaluated on a single difficult target point, equal to that used in BID17 .The Point Maze is equivalent to the Ant Maze, with size scaled down by a factor of 2 and the agent replaced with a point mass, which is controlled by actions of dimension two -one action determines a rotation on the pivot of the point mass and the other action determines a push or pull on the point mass in the direction of the pivot.

For the 'Images' versions of these environments, we zero-out the x, y coordinates in the observation and append a low-resolution 5 ?? 5 ?? 3 top-down view of the environment.

The view is centered on the agent and each pixel covers the size of a large block (size equal to width of the corridor in Ant Maze).

The 3 channels correspond to (1) immovable blocks (walls, gray in the videos), (2) movable blocks (shown in red in videos), and (3) chasms where the agent may fall.

Figure 5 : The tasks we consider in this paper.

Each task is a form of navigation.

The agent must navigate itself (or a small red block in 'Block' tasks) to the target location (green arrow).

We also show an example top-down view image (from an episode on the Ant Maze task).

The image is centered on the agent and shows walls and blocks (at times split across multiple pixels).The Ant Block environment puts the ant in a 16 ?? 16 square room next to a 0.8 ?? 0.8 ?? 0.4 small movable block.

The agent is rewarded based on negative L2 distance of the block to a desired target location.

During training, these target locations are sampled randomly from all possible locations.

Evaluation is on a target location diagonally opposite the ant.

The Ant Block Maze environment consists of the same ant and small movable block in a ???-shaped corridor.

During training, these target locations are sampled randomly from all possible locations.

Evaluation is on a target location at the end of the corridor.

We follow the basic training details used in BID17 .

Some differences are listed below:??? We input the whole observation to the lower-level policy BID17 zero-out the x, y coordinates for the lower-level policy).??? We use a Huber function for D, the distance function used to compute the low-level reward.???

We use a goal dimension of size 2.

We train the higher-level policy to output actions in [???10, 10] 2 .

These actions correspond to desired deltas in state representation.??? We use a Gaussian with standard deviation 5 for high-level exploration.??? Additional differences in low-level training (e.g. reward weights and discounting) are implemented according to Section 5.We parameterize f ?? with a feed-forward neural network with two hidden layers of dimension 100 using relu activations.

The network structure for ?? ?? is identical, except using hidden layer dimensions 400 and 300.

We also parameterize ??(s, ??) := f ?? (s) + ?? ?? (s, ??).

These networks are trained with the Adam optimizer using learning rate 0.0001.

FORMULA0 ).

We compare to variants of our method that are implemented more in the style of CPC.

Although we find that using a dot product rather than distance function D is detrimental, a number of distance-based variants of our approach may perform similarly.

Figure 7: We provide additional results comparing to variants of ??-VAE BID10 .

We find that even with this additional hyperparameter, the VAE approach to representation learning does not perform well outside of the simple point mass environment.

The drawback of the VAE is that it is encouraged to reconstruct the entire observation, despite the fact that much of it is unimportant and possibly exhibits high variance (e.g. ant joint velocities).

This means that outside of environments with high-information state observation features, a VAE approach to representation learning will suffer.

Figure 8: We evaluate the ability of our learned representations to transfer from one task to another.

For these experiments, we took a representation function f learned on Ant Maze, fixed it, and then used it to learn a hierarchical policy on a completely different task.

We evaluated the ability of the representation to transfer to "Reflected Ant Maze" (same as Ant Maze but the maze shape is changed from '???' to '???') and "Ant Push".

We find that the representations are robust these changes to the environment and can generalize successfully.

We are able to learn well-performing policies in these distinct environments even though the representation used was learned with respect to a different task.

Ant Maze Env XY Ours Ours (Images) Figure 9 : We replicate the results of Figure 2 but with representations learned according to data collected by a random higher-level policy.

In this setting, when there is even less of a connection between the representation learning objective and the task objective, our method is able to recover near-ideal representations.

BID17 .

The representation used in the original formulation of HIRO is a type of oracle -sub-goals are defined as only the position-based (i.e., not velocity-based) components of the agent observation.

In our own experiments, we found this method to perform similarly to the XY oracle in non-image tasks.

However, when the state observation is more complex (images) performance is much worse.

@highlight

We translate a bound on sub-optimality of representations to a practical training objective in the context of hierarchical reinforcement learning.

@highlight

The authors proposes a novel approach in learning a representation for HRL and state an intriguing connection between representation learning and bounding the sub-optimality which results in a gradient based algorithm

@highlight

This paper proposes a way to handle sub-optimality in the context of learning representations which refer to the sub-optimality of hierarchical polity with respect to the task reward.