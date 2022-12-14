In the pursuit of increasingly intelligent learning systems, abstraction plays a vital role in enabling sophisticated decisions to be made in complex environments.

The options framework provides formalism for such abstraction over sequences of decisions.

However most models require that options be given a priori, presumably specified by hand, which is neither efficient, nor scalable.

Indeed, it is preferable to learn options directly from interaction with the environment.

Despite several efforts, this remains a difficult problem: many approaches require access to a model of the environmental dynamics, and inferred options are often not interpretable, which limits our ability to explain the system behavior for verification or debugging purposes.

In this work we develop a novel policy gradient method for the automatic learning of policies with options.

This algorithm uses inference methods to simultaneously improve all of the options available to an agent, and thus can be employed in an off-policy manner, without observing option labels.

Experimental results show that the options learned can be interpreted.

Further, we find that the method presented here is more sample efficient than existing methods, leading to faster and more stable learning of policies with options.

Recent developments in reinforcement learning (RL) methods have enabled agents to solve problems in increasingly complicated domains BID14 .

However, in order for agents to solve more difficult and realistic environments-potentially involving long sequences of decisions-more sample efficient techniques are needed.

One way to improve on existing agents is to leverage abstraction.

By reasoning at various levels of abstraction, it is possible to infer, learn and plan much more efficiently.

Recent developments have lead to breakthroughs in terms of learning rich representations for perceptual information BID2 .

In RL domains, however, while efficient methods exist to plan and learn when abstraction over sequences of actions is provided a priori, it has proven more difficult to learn this type of temporal abstraction from interaction data.

Many frameworks for formalizing temporal abstraction have been proposed; most recent developments build on the Options framework BID21 BID16 , which offers a flexible parameterization, potentially amenable to learning.

The majority of prior work on learning options has centered around the idea of discovering subgoals in state space, and constructing a set of options such that each option represents a policy leading to that subgoal BID11 BID12 BID20 ).

These methods can lead to useful abstraction, however, they often require access to a model of the environment dynamics, which is not always available, and can be infeasible to learn.

Our contributions instead build on the work of BID1 , and exploits a careful parameterization of the policy of the agent, in order to simultaneously learn a set of options, while directly optimizing returns.

We relax a few key assumptions of this previous work, including the expectation that only options that were actually executed during training can be learned, and the focus on executing options in an on-policy manner, with option labels available.

By relaxing these, we can improve sample efficiency and practical applicability, including the possibility to seed control policies from expert demonstrations.

We present an algorithm that solves the problem of learning control abstractions by viewing the set of options as latent variables that concisely represent the agent's behaviour.

More precisely, we do not only improve those options that were actually executed in a trajectory.

Instead, we allow intra-option learning by simultaneously improving all individual options that could have been executed, and the policy over options, in an end-to-end manner.

We evaluate this algorithm on continuous MDP benchmark domains and compare it to earlier reinforcement learning methods that use flat and hierarchical policies.

Recent attention in the field of option discovery generally falls into one of two categories.

One branch of work focuses on learning options that are able to reach specific subgoals within the environment.

Much work in this category has focused on problems with discrete state and action spaces, indentifying salient or bottleneck states as subgoals BID11 BID12 BID20 BID19 .

Recent work has focused on finding subgoal states in continuous state spaces using clustering BID15 or spectral methods BID9 .

BID6 describes an approach where subgoals of new policies are defined by the initiation conditions of existing options.

Specifying options using subgoals generally requires a given or a-priori learned system model, or specific assumptions about the environment.

Furthermore, the policies to reach each subgoal have to be trained independently, which can be expensive in terms of data and training time BID1 .A second body of work has learned options by directly optimizing over the parameters of function approximations that are structured in a way to yield hierarchical policies.

One possibility is to augment states or trajectories with the indexes of the chosen options.

Option termination, selection, and inter-option behavior than all depend on both the regular system state and the current option index.

This approach was suggested by BID7 for learning the parameters of a hierarchical model consisting of pre-structured policies.

In the option-critic architecture BID1 , a similar model is employed, with option-specific value functions to learn more efficiently.

Furthermore, neural networks are used instead of a task-specific given structure.

BID10 use an explicit partitioning of the state space to ensure policy specialization.

An alternative to state augmentation was proposed by BID10 .

In that paper, options were considered latent variables rather than observable variables.

That paper employed a policy structure that allowed maximizing the objective in the presence of these latent variables using an expectation-maximization approach.

However, the optimization of this structure requires option policies to be linear in state features, which imposes the need to specify good state features a priori.

Further, this approach necessitates the use of information from the entire trajectory before policy improvement can be done, eliminating the possibility of an on-line approach.

BID5 uses a similar approach in the imitation learning setting with neural network policies instead of a task specific structure.

There are several other related works in hierarchical reinforcement learning outside of the options framework.

One possibility is to have a higher-level policy learn to set goals for a learning lowerlevel policy BID26 , or to set a sequence of lower-level actions to be followed BID25 .

Another possibility is to have a higher-level policy specify a prior over lower-level policies for different tasks, such that the system can acquire useful learning biases for new tasks BID29 .

We consider an agent interacting with its environment, at several discrete time steps.

Generally, the state of the environment at step t, is provided in the form of a vector, s t , with s 0 determined by an initial state distribution.

At every step, the agent observes s t , and selects a vector-valued action a t , according to a stochastic policy ??(a t |s t ), which gives the probability that an agent executes a particular action from a particular state.

The agent then receives a reward r t and the next state s t+1 from the environment.

We consider episodic setups where, eventually, the agent reaches a terminal state, s T upon which the environment is reset, to a state drawn from an initial state distribution.

A sequence of states, actions and rewards generated in this manner is referred to as a trajectory ?? .We define the discounted return from step t within a trajectory to be R DISPLAYFORM0 The objective of the learning agent is to maximize the expected per-trajectory return, given by ?? = DISPLAYFORM1

While several methods exist for learning a policy from interaction with the environment, here, we focus on policy gradient methods, which have benefited from a recent resurgence in popularity.

Policy gradient methods directly optimize ?? by performing stochastic gradient ascent on the parameters ?? of a family of policies ?? ?? .

Policy gradients can be estimated from sample trajectories, or in an online manner.

The full return likelihood ratio gradient estimator (Williams, 1992) takes the form: DISPLAYFORM0 where b is a baseline, used to reduce variance.

This is one of the simplest, most general policy gradient estimators, and can be importance sampled if observed trajectories are not generated from the agent's policy.

The policy gradient theorem BID22 expands on this result in the on-policy case, giving a gradient estimate of the form: DISPLAYFORM1 which can be shown to yield lower variance gradient estimates.

The options framework provides the necessary formalism for abstraction over sequences of decisions in RL BID21 BID16 .

The agent is given access to a set of options, indexed by ??.

Each option has its own policy: ?? ?? (a t |s t ), an initiation set, representing the states in which the option is available, and a termination function ?? ?? (s t ), which represents the state-dependent probability of terminating the option.

Additionally, the policy over options, ?? ??? (?? t |s t ) is employed to select from available options once termination of the previous option occurs.

During execution, option are used as follows: in the initial state, an option is sampled from the policy over options.

An action is then taken according to the policy belonging to the currently active option.

After selecting this action and observing the next state, the policy then terminates, or does not, according to the termination function.

If the option does not terminate, the current option remains active.

Otherwise the policy over options can be sampled in the new state in order to determine the next active option.

The policy over options can be combined with the termination function in order to yield the optionto-option policy function: DISPLAYFORM0 where ?? is the Kronecker delta.

To learn options using a policy gradient method we parametrize all aspects of the policy: ?? ???,?? denotes the policy over options, parametrized by ??.

?? ??,?? then denotes the intra-option policy of option ??, parametrized by ??. Finally ?? ??,?? is the termination function for ??, parametrized by ??.

We aim to optimize the performance of the agent with respect to a set of policy parameters.

The loss function is identical to that employed by traditional policy gradient methods: we optimize the expected return of trajectories in the MDP sampled using the current policy, DISPLAYFORM0 where E ?? denotes expectation over sampled trajectories.

The expected performance can be maximized by increased the probability of visiting highly rewarded state-action pairs.

To increase this probability, it does not matter which option originally generated that state-action pair, rather, we will derive an algorithm that updates all options that could have generated that state-action pair.

Determining these options is done in a differentiable inference step.

As a result the policy can be optimized end-to-end, yielding our Inferred Option Policy Gradient algorithm.

In order to compute the gradient of the loss objective, we decompose P (?? ) into the relevant conditional probabilities, and employ the "likelihood ratio" method, so that it is possible to estimate the gradient from samples: DISPLAYFORM1 Note that this is similar to the REINFORCE policy gradient, though here actions are not independent, even when conditioned on states, since information can still pass through the unobserved options.

In order to compute the inner gradient, we marginalize over the hidden options at each time step, leading to: ) term can be expressed in a recursive form, simply as an application of the forward algorithm: DISPLAYFORM2 DISPLAYFORM3 where c i is a normalization factor, given by: DISPLAYFORM4 and our initial value is P (?? 0 |s 0 ) = ?? ???,?? (?? 0 |s 0 ).

If our policies are differentiable, then this recursive term is differentiable as well, allowing us to perform gradient descent to maximize our objective, using the sampled data to compute the full return Monte Carlo gradient estimate: DISPLAYFORM5 where ?? = (s 0 , a 0 , . . . , a T ???1 , s T ) is a trajectory sampled from the system using the current policy ?? ?? .

The variance of this estimator can be reduced through inclusion of a constant baseline, through an argument identical to that used for REINFORCE BID27 .Here, we notice that actions at any given time step are conditionally independent of rewards received in the past, given the trajectory up that action.

As in other policy gradient methods, we can reduce variance further by removing these terms from our gradient estimator.

This is formally expressed as: DISPLAYFORM6 With this realization, we can simplify our estimator to: DISPLAYFORM7 where b(s j ) is a state-dependent baseline.

Note that the estimate is unbiased regardless of the baseline, although good baselines can reduce the variance.

In this work, we use a learned parametric approximation of the value function V ?? as baseline.

The value function is learned using gradient descent on the mean squared prediction error of Monte-Carlo returns: DISPLAYFORM8 Estimating the value function can also be done using other standard methods such as LSTD or TD(??).Below, we describe the algorithm for learning options to optimize returns through a series of interactions with the environment.

While Algorithm 1 can only be applied in the episodic RL setup, it is also possible to employ the technical insight shown here in an online manner.

One potential method for doing so is described in Appendix A.

Initialize parameters randomly foreach episode do ?? 0 ??? ?? ??? (??|s 0 ) // sample an option from the policy over options at the initial state for t ??? 0, . . .

, T do a t ??? ?? ??t (s t ) // sample an action according to the current intra-option policy Get next state s t+1 and reward r t from the system ?? t+1 ????? ??? (?? t+1 |?? t , s t+1 ) // sample the next option according to the policy over option end Update ?? according to (4), using sampled episode ??, ??, and ?? according to (3), using sampled episode end

In order to evaluate the effectiveness of our algorithm, as well as the qualitative attributes of the options learned, we examine its performance across several standardized continuous control environments as implemented in the OpenAI Gym BID3 in the MuJoCo physics simulator BID24 .

In particular, we examine the Hopper-v1 (observation dimension: 11, action dimension: 3), Walker2d-v1 (observation dimension: 17, action dimension: 6), HalfCheetah-v1 (observation dimension: 17, action dimension: 6), and Swimmer-v1 (observation dimension: 8, action dimension: 2) environments.

Generally, they all require the agent to learn to operate joint motors in order to move the agent in a particular direction, with penalties for unnecessary actions.

Together, they are considered to be reasonable benchmarks for state-of-the art continuous RL algorithms.

We compared the performance of our algorithm (IOPG) with results from option-critic (OC) and asynchronous actor-critic (A3C) methods, as described in .In order to ensure an appropriate comparison, IOPG and OC were also implemented using multiple agents operating in parallel, as is done in A3C.

The option-critic algorithm as described in Bacon To ensure a fair comparison, we employed the same parametrized actor as the inter-option policy in our option-critic baseline as was used in IOPG.

Since option-critic already learns option-value functions, no SMDP-level value function approximation is needed.

Our model architecture for all three algorithms closely follows that of BID18 .

The policies and value functions were represented using separate feed-forward neural networks, with no parameters shared.

For each agent, both the value function and the policies used two hidden layers of 64 units with tanh activation functions.

The IOPG and OC methods shared these parameters across all policy and termination networks.

The option sub-policies and A3C policies were implemented as linear layers on top of this, representing the mean of a Gaussian distribution.

The variance of the policy was parametrized by a linear softplus layer.

Option termination was given by a linear sigmoid layer for each option.

The policy over options, for OC and IOPG methods, was represented using a final linear softmax layer, of size equal to the number of options available.

The value function for IOPG and AC methods was represented using a final linear layer of size 1, and for OC, size |???|.

All weight matrices were initialized to have normalized rows.

RMSProp BID23 ) was used to optimize parameters for all agents.

We employ a single shared set of RMSProp parameters across all asynchronous threads.

Additionally, entropy regularization was used during optimization for the AC policies, the option policies and the policies over options.

This was done in order to encourage exploration, and to prevent the policies from converging to single repeated actions, as policy gradient methods parametrized by neural networks often suffer from this problem.

The results of these experiments are shown in FIG2 .

We see that IOPG, despite having significantly more parameters to optimize, and recovering additional structure, is able to learn as quickly as A3C across all of the domains, and learns significantly faster in the Walker2d environment.

This is likely enabled by the fact that all of the options in IOPG can make use of all of the data gathered.

OC, on the other hand seems to suffer a reduction in learning speed due to the fact that options are not all learned simultaneously, preventing experience from being shared between them.

In order to further understand the nature of the options learned, we performed a visualization of them over a random subsample of states in the last 8000 frames.

We perform T-SNE BID8 on these states in order to represent the high-dimensional state space in two dimensions, while preserving some structure.

FIG3 shows the results of this procedure.

We can see that different options are active in different regions of state space.

This indicates that the options learned can be interpreted as having some local structure.

Options appear to be spatially coherent, as well as having structure in the policy.

The relation between state and action abstraction has been observed previously in the RL literature BID0 BID17 .

It is also likely that options employed are temporally coherent, since in smooth, continuous domains, it is likely the case that spatially close states are also close in time, matching the intuitive notion that options represent abstract behaviours, which can extend over several actions.

FIG4 displays additional analyses of the options learned in the Walker2d environment.

We found that in this particular environment, agents with either four or eight options available perform roughly equally, while having only two options led to sub-optimal performance FIG4 .

This effect can be explained by the fact that three options seem to be sufficient, and if more options are given only three of them tend to get frequently selected FIG4 .

This finding suggests that only three of the options that IOPG learns are useful here, perhaps due to the relative simplicity of the environment.

In FIG4 , we observe further evidence that the options learned by IOPG are temporally extended.

A moving average of the continuation probability (1 ??? ?? ?? (s t )) during training indicates that early on, when the options are not well optimized, termination occurs quite frequently.

As the options improve, termination decreases, until the policy over options is only queried approximately every ten steps on average.

In this paper, we have introduced a new algorithm for learning hierarchical policies within the options framework, called inferred option policy gradients.

This algorithm treats options as latent variables.

Gradients are propagated through a differentiable inference step that allows end-to-end learning of option policies, as well as option selection and termination probabilities.

In our algorithms policies take responsibility for state-actions pairs they could have generated.

In contrast, in learning algorithms for hierarchical policies that use an augmented state space, option policies are updated using only those state-action pairs the actually generated.

As a result, in our algorithm options do not tend to become 'responsible' for unlikely states or actions they generated.

Thus, options are stimulated more strongly to specialize in a part of the state space.

We conjecture that this specialization caused the discussed increase in the interpretability of options.(a) Performance in the Walker2d environment as a function of available options.

We see that in this environment, having several options available to the agent leads to an improved policy.(b) In the Walker2d environment, initially option selection is uniform.

After training only 3 options tend to be selected, even when more are available.

Selection frequencies are averaged over 100 sampled states.(c) As the options improve during training, the probability of remaining in the active option increases, plateauing at around 0.85.

This suggests that the options learned here exhibit temporal extension.

Furthermore, in our experiments learning with inferred options was significantly faster than learning with an option-augmented state space.

In fact, learning with inferred options proved equally fast, or sometimes even faster, than using a comparable non-hierarchical policy gradient method despite IOPG having many more parameters.

We conjecture that option inference encourages intra-option learning, thus allowing multiple options to improve as the result of a single learning experience, causing this speed-up.

In future work, we want to quantify the suitability of the learned options for transfer between tasks.

Our experiments so far were in the episodic setting.

We want to investigate an on-line, actor-critic version of learning with inferred options to learn continuously in infinite-horizon problems.

@highlight

We develop a novel policy gradient method for the automatic learning of policies with options using a differentiable inference step.

@highlight

The paper presents a new policy gradient technique for learning options, where a single sample can be used to update all options.

@highlight

Proposes an off-policy method for learning options in complex continuous problems.