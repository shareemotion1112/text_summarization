An obstacle that prevents the wide adoption of (deep) reinforcement learning (RL) in control systems is its need for a large number of interactions with the environment in order to master a skill.

The learned skill usually generalizes poorly across domains and re-training is often necessary when presented with a new task.

We present a framework that combines techniques in \textit{formal methods} with \textit{hierarchical reinforcement learning} (HRL).

The set of techniques we provide allows for the convenient specification of tasks with logical expressions, learns hierarchical policies (meta-controller and low-level controllers) with well-defined intrinsic rewards using any RL methods and is able to construct new skills from existing ones without additional learning.

We evaluate the proposed methods in a simple grid world simulation as well as simulation on a Baxter robot.

Reinforcement learning has received much attention in the recent years because of its achievements in games BID17 , , robotics manipulation Jang et al., , BID5 and autonomous driving BID8 , BID16 .

However, training a policy that sufficiently masters a skill requires an enormous amount of interactions with the environment and acquiring such experience can be difficult on physical systems.

Moreover, most learned policies are tailored to mastering one skill (by maximizing the reward) and are hardly reusable on a new skill.

Skill composition is the idea of constructing new skills out of existing skills (and hence their policies) with little to no additional learning.

In stochastic optimal control, this idea has been adopted by authors of BID25 and BID4 to construct provably optimal control laws based on linearly solvable Markov decision processes.

Authors of BID6 , Tang & Haarnoja have showed in simulated manipulation tasks that approximately optimal policies can result from adding the Q-functions of the existing policies.

Hierarchical reinforcement learning is an effective means of achieving transfer among tasks.

The goal is to obtain task-invariant low-level policies, and by re-training the meta-policy that schedules over the low-level policies, different skills can be obtain with less samples than training from scratch.

Authors of BID7 have adopted this idea in learning locomotor controllers and have shown successful transfer among simulated locomotion tasks.

Authors of BID18 have utilized a deep hierarchical architecture for multi-task learning using natural language instructions.

Temporal logic is a formal language commonly used in software and digital circuit verification BID1 as well as formal synthesis BID2 .

It allows for convenient expression of complex behaviors and causal relationships.

TL has been used by BID19 , BID13 to synthesize provably correct control policies.

Authors of BID0 have also combined TL with Q-learning to learn satisfiable policies in discrete state and action spaces.

In this work, we focus on hierarchical skill acquisition and zero-shot skill composition.

Once a set of skills is acquired, we provide a technique that can synthesize new skills without the need to further interact with the environment (given the state and action spaces as well as the transition remain the same).

We adopt temporal logic as the task specification language.

Compared to most heuristic reward structures used in the RL literature to specify tasks, formal specification language excels at its semantic rigor and interpretability of specified behaviors.

Our main contributions are:• We take advantage of the transformation between TL formula and finite state automata (FSA) to construct deterministic meta-controllers directly from the task specification without the necessity for additional learning.

We show that by adding one discrete dimension to the original state space, structurally simple parameterized policies such as feed-forward neural networks can be used to learn tasks that require complex temporal reasoning.• Intrinsic motivation has been shown to help RL agents learn complicated behaviors with less interactions with the environment BID22 , BID11 BID9 .

However, designing a well-behaved intrinsic reward that aligns with the extrinsic reward takes effort and experience.

In our work, we construct intrinsic rewards directly from the input alphabets of the FSA (a component of the automaton), which guarantees that maximizing each intrinsic reward makes positive progress towards satisfying the entire task specification.

From a user's perspective, the intrinsic rewards are constructed automatically from the TL formula.•

In our framework, each FSA represents a hierarchical policy with low-level controllers that can be re-modulated to achieve different tasks.

Skill composition is achieved by manipulating the FSA that results from their TL specifications in a deterministic fashion.

Instead of interpolating/extrapolating among existing skills, we present a simple policy switching scheme based on graph manipulation of the FSA.

Therefore, the compositional outcome is much more transparent.

We introduce a method that allows learning of such hierarchical policies with any non-hierarchical RL algorithm.

Compared with previous work on skill composition, we impose no constraints on the policy representation or the problem class.

In this section, we briefly introduce the options framework BID23 , especially the terminologies that we will inherit in later sections.

We start with the definition of a Markov Decision Process.

Definition 1.

An MDP is defined as a tuple M = S, A, p(·|·, ·), R(·, ·, ·) , where S ⊆ IR n is the state space ; A ⊆ IR m is the action space (S and A can also be discrete sets); p : S ×A×S → [0, 1] is the transition function with p(s |s, a) being the conditional probability density of taking action a ∈ A at state s ∈ S and ending up in state s ∈ S; R : S ×A×S → IR is the reward function.

let T be the length of a fixed time horizon.

The goal is to find a policy π : S → A (or π : S ×A → [0, 1] for stochastic policies) that maximizes the expected return, i.e. DISPLAYFORM0 where τ T = (s 0 , a 0 , ..., s T , ) denotes the state-action trajectory from time 0 to T .The options framework exploits temporal abstractions over the action space.

An option is defined as a tuple o = I, π o , β where I is the set of states that option o can be initiated (here we let I = S for all options), π o : S → A is an options policy and β : S → [0, 1] is the termination probability for the option at state s.

In addition, there is a policy over options π h : S → O (where O is a set of available options) that schedules among options.

At a given time step t, an option o is chosen according to π h (s t ) and the options policy π o is followed until the termination probability β(s) > threshold at time t + k, and the next option is chosen by π h (s t+k ).

We consider tasks specified with Truncated Linear Temporal Logic (TLTL).

We restrict the set of allowed operators to be DISPLAYFORM0 where f (s) < c is a predicate, ¬ (negation/not), ∧ (conjunction/and), and ∨ (disjunction/or) are Boolean connectives, and ♦ (eventually), U (until), T (then), (next), are temporal operators.

Implication is denoted by ⇒ (implication).

Essentially we excluded the Always operator (2) with reasons similar to BID12 .

We refer to this restricted TLTL as syntactically co-safe TLTL (scTLTL) BID27 used similar idea for LTL).

There exists a real-value function ρ(s 0:T , φ) called robustness degree that measures the level of satisfaction of trajectory s 0:T with respective to φ.

ρ(s 0:T , φ) > 0 indicates that s 0:T satisfies φ and vice versa.

Definitions for the boolean semantics and robustness degree are provided in Appendix E.Any scTLTL formula can be translated into a finite state automata (FSA) with the following definition: Definition 2.

An FSA is defined as a tuple DISPLAYFORM1 where Q φ is a set of automaton states; Ψ φ is an input alphabet, we denote ψ qi,qj ∈ Ψ φ the predicate guarding the transition from q i to q j (as illustrated in FIG0 ); q 0 ∈ Q φ is the initial state; DISPLAYFORM2 In addition, given an MDP state s, we can calculate the transition in automata states at s by DISPLAYFORM3 We abuse the notation p φ to represent both kinds of transitions when the context is clear.

F φ is a set of final automaton states.

The translation from TLTL formula to FSA to can be done automatically with available packages like Lomap BID26 .

Example 1.

FIG0 (left) illustrates the FSA resulting from formula φ = ¬b U a. In English, φ entails during a run, b cannot be true until a is true and a needs to be true at least once.

The FSA has three automaton states Q φ = {q 0 , q f , trap} with q 0 being the input(initial) state (here q i serves to track the progress in satisfying φ).

The input alphabet is defined as the Ψ φ = {¬a ∧ ¬b, ¬a ∧ b, a∧¬b, a∧b}. Shorthands are used in the figure, for example a = (a∧b)∨(a∧¬b).

Ψ φ represents the power set of {a, b}, i.e. Ψ φ = 2 {a,b} .

During execution, the FSA always starts from state q 0 and transitions according to Equation (3) or (4).

The specification is satisfied when q f is reached and violated when trap is reached.

In this example, q f is reached only when a becomes true before b becomes true.

We start with the following problem definition:

Problem 1.

Given an MDP in Definition 1 with unknown transition dynamics p(s |s, a) and a scTLTL specification φ over state predicates (along with its FSA A φ ) as in Definition 2.

Find a policy π φ such that DISPLAYFORM0 where 1(ρ(s 0:T , φ) > 0) is an indicator function with value 1 if ρ(s 0:T , φ) > 0 and 0 otherwise.

Problem 1 defines a policy search problem where the trajectories resulting from following the optimal policy should satisfy the given scTLTL formula in expectation.

Problem 2.

Given two scTLTL formula φ 1 and φ 2 along with policy π φ1 that satisfies φ 1 and π φ2 that satisfies φ 2 .

Obtain a policy π φ that satisfies φ = φ 1 ∧ φ 2 .Problem 2 defines the problem of task composition.

Given two policies each satisfying a scTLTL specification, construct the policy that satisfies the conjunction of the given specifications.

Solving this problem is useful when we want to break a complex task into simple and manageable components, learn a policy that satisfies each component and "stitch" all the components together so that the original task is satisfied.

It can also be the case that as the scope of the task grows with time, the original task specification is amended with new items.

Instead of having to re-learn the task from scratch, we can only learn a policy that satisfies the new items and combine them with the old policy.

We propose to solve Problem 1 by constructing a product MDP from the given MDP and FSA that can be solved using any state-of-the-art RL algorithm.

The idea of using product automaton for control synthesis has been adopted in various literature BID13 , BID3 .

However, the methods proposed in these works are restricted to discrete state and actions spaces.

We extend this idea to continuous state-action spaces and show its applicability on robotics systems.

For Problem 2, we propose a policy switching scheme that satisfies the compositional task specification.

The switching policy takes advantage of the characteristics of FSA and uses robustness comparison at each step for decision making.

Problem 1 can be solved with any episode-based RL algorithm.

However, doing so the agent suffers from sparse feedback because a reward signal can only be obtained at the end of each episode.

To address this problem as well as setting up ground for automata guided HRL, we introduce the FSA augmented MDP Definition 3.

An FSA augmented MDP corresponding to scTLTL formula φ is defined as M φ = S , A,p(·|·, ·),R(·, ·) whereS ⊆ S × Q φ , A is the same as the original MDP.p(s |s, a) is the probability of transitioning tos givens and a, in particular DISPLAYFORM0 Here p φ is defined in Equation (4).R :S ×S → IR is the FSA augmented reward function, defined byR DISPLAYFORM1 where Ω q is the set of automata states that are connected with q through outgoing edges.

D q φ = q ∈Ωq ψ q,q represents the disjunction of all predicates guarding the transitions that originate from q. The goal is to find the optimal policy that maximizes the expected sum of discounted return, i.e. DISPLAYFORM2 where γ < 1 is the discount factor, T is the time horizon.

As a quick example to the notation D q φ , consider the state q 0 in the FSA in FIG0 , DISPLAYFORM3 The goal is then to find a policy π :S → A that maximizes the expected sum ofR over the horizon T .

The FSA augmented MDP can be constructed with any standard MDP and a scTLTL formula.

And it can be solved with any off-the-shelf RL algorithm.

By directly learning the flat policy π we bypass the need to learn multiple options policies separately.

After obtaining the optimal policy π , the optimal options policy for any option o q can be extracted by executing π (a|s, q) without transitioning the automata state, i.e. keeping q i fixed (denoted π q ).

And π q satisfies π qi = arg max DISPLAYFORM4 In other words, the purpose of π qi is to activate one of the outgoing edges of q i as soon as possible and by doing so repeatedly eventually reach q f .The reward function in Equation (7) encourages the system to exit the current automata state and move on to the next, and by doing so eventually reach the final state q f .

However, this reward does not distinguish between the trap state and other states and therefore will also promote entering of the trap state.

One way to address this issue is to impose a terminal reward on both q f and trap.

Because the reward is an indicator function with maximum value of 1, we assign terminal rewards R q f = 2 and R trap = −2.Appendix D describes the typical learning routine using FSA augmented MDP.

The algorithm utilizes memory replay which is popular among off-policy RL methods (DQN, A3C, etc) but this is not a requirement for learning withM φ .

On-policy methods can also be used.

In section, we provide a solution for Problem 2 by constructing the FSA of φ from that of φ 1 and φ 2 and using φ to synthesize the policy for the combined skill.

We start with the following definition.

DISPLAYFORM0 is the set of product automaton, states, q 0 = (q 0 1 , q 0 2 ) is the product initial state, F ⊆ F φ1 ∩ F φ2 is the final accepting states.

Following Definition 2, for states q = (q 1 , q 2 ) ∈ Q φ and q = (q 1 , q 2 ) ∈ Q φ , the transition probability p φ is defined as DISPLAYFORM1 Example 2.

FIG0 (right) illustrates the FSA of A φ1 and A φ2 and their product automaton A φ .Here φ 1 = ♦a ∧ ♦b which entails that both a and b needs to be true at least once (order does not matter), and φ 2 = ¬b U a which is the same as Example 1.

The resultant product corresponds to the formula φ = (♦a ∧ ♦b) ∧ (¬b U a) which dictates that a and b need to be true at least once, and a needs to be true before b becomes true (an ordered visit).

We can see that the trap state occurs in A φ2 and A φ , this is because if b is ever true before a is true, the specification is violated and q f can never be reached.

In the product automaton, we aggregate all state pairs with a trap state component into one trap state.

For q = (q 1 , q 2 ) ∈ Q φ , let Ψ q , Ψ q1 and Ψ q2 denote the set of predicates guarding the outgoing edges of q, q 1 and q 2 respectively.

Equation (10) entails that a transition at q in the product automaton A φ exists only if corresponding transitions at q 1 , q 2 exist in A φ1 and A φ2 respectively.

Therefore, ψ q,q = ψ q1,q 1 ∧ ψ q2,q 2 , for ψ q,q ∈ Ψ q , ψ q1,q 1 ∈ Ψ q1 , ψ q2,q 2 ∈ Ψ q2 (here q i is a state such that p φi (q i |q i ) = 1).

Following Equation FORMULA10 , DISPLAYFORM2 where DISPLAYFORM3 Repeatedly applying the distributive law DISPLAYFORM4 to the logic formula D q φ transforms the formula to DISPLAYFORM5 Therefore, DISPLAYFORM6 The second step in Equation FORMULA0 follows the robustness definition.

Recall that the optimal options policies for q 1 and q 2 satisfy π qi = arg max DISPLAYFORM7 Equation FORMULA0 provides a relationship among π q , π q1 and π q2 .

Given this relationship, We propose a simple switching policy based on stepwise robustness comparison that satisfies φ = φ 1 ∧ φ 2 as follows DISPLAYFORM8 We show empirically the use of this switching policy for skill composition and discuss its limitations in the following sections.

In this section, we provide a simple grid world navigation example to illustrate the techniques presented in Sections 4 and 5.

Here we have a robot navigating in a discrete 1 dimensional space.

Its MDP state space S = {s|s ∈ [−5, 5), s is discrete}, its action space A = {lef t, stay, right}. The robot navigates in the commanded direction with probability 0.8, and with probability 0.2 it randomly chooses to go in the opposite direction or stay in the same place.

The robot stays in the same place if the action leads it to go out of bounds.

We define two regions a : −3 < s < −1 and b : 2 < s < 4.

For the first task, the scTLTL specification φ 1 = ♦ a ∧ ♦b needs to be satisfied.

In English, φ 1 entails that the robot needs to visit regions a and b at least once.

To learn a deterministic optimal policy π φ1 : S × Q → A, we use standard Q-Learning BID28 on the FSA augmented MDP for this problem.

We used a learning rate of 0.1, a discount factor of 0.99, epsilon-greedy exploration strategy with decaying linearly from 0.0 to 0.01 in 1500 steps.

The episode horizon is T = 50 and trained for 500 iterations.

All Q-values are initialized to zero.

The resultant optimal policy is illustrated in FIG1 .We can observe from the figure above that the policy on each automaton state q serves a specific purpose.

π q0 tries to reach region a or b depending on which is closer.

π q1 always proceeds to region a. π q2 always proceeds to region b. This agrees with the definition in Equation 9.

The robot can start anywhere on the s axis but must always start at automata state q 0 .

Following π φ1 , the robot will first reach region a or b (whichever is nearer), and then aim for the other region which in turn satisfies φ.

The states that have stay as their action are either goal regions (states (−2, q 0 ), (3, q 1 ), etc) where a transition on q happens or states that are never reached (states (−3, q 1 ), (−4, q 2 ), etc) because a transition on q occurs before they can be reached.

To illustrate automata guided task composition described in Example 2, instead of learning the task described by φ from scratch, we can simply learn policy π φ2 for the added requirement φ 2 = ¬b U a. We use the same learning setup and the resultant optimal policy is depicted in FIG4 .

It can be observed that π φ2 tries to reach a while avoiding b. This behavior agrees with the specification φ 2 and its FSA provided in FIG1 .

The action at s = 4 is stay because in order for the robot to reach a it has to pass through b, therefore it prefers to obtain a low reward over violating the task.

Having learned policies π φ1 and π φ2 , we can now use Equation 15 to construct policy π φ1∧φ2 .

The resulting policy for π φ1∧φ2 is illustrated in FIG1 (upper right).

This policy guides the robot to first reach a (except for state s = 4) and then go to b which agrees with the specification.

Looking at FIG0 , the FSA of φ = φ 1 ∧φ 2 have two options policies π φ (·, q 0 ) and π φ (·, q 1 ) 2 (trap state and q f are terminal states which don't have options).

State q 1 has only one outgoing edge with the guarding predicate ψ q1,q f : b, which means π φ (·, q 1 ) = π φ1 (·, q 2 )(they have the same guarding predicate).

Policy π φ (·, q 0 ) is a switching policy between π φ1 (·, q 0 ) and π φ2 (·, q 0 ).

FORMULA0 .

We can see that the robustness of both policies are the same from s = −5 to s = 0.

And their policies agree in this range FIG3 .

As s becomes larger, disagreement emerge because π φ1 (·, q 0 ) wants to stay closer to b but π φ2 (·, q 0 ) wants otherwise.

To maximize the robustness of their conjunction, the decisions of π φ2 (·, q 0 ) are chosen for states s > 0.

In this section, we construct a set of more complicated tasks that require temporal reasoning and evaluate the proposed techniques on a simulated Baxter robot.

The environment is shown in FIG3 (left).

In front of the robot are three square regions and two circular regions.

An object with planar coordinates p = (x, y) can use predicates S red (p), S blue (p), S black (p), C red (p), C blue (p) to evaluate whether or not it is within the each region.

The predicates are defined by S : (x min < x < x max ) ∧ (y min < y < y max ) and C : dist((x, y), (x, y) center ) < r. (x min , y min ) and (x max , y max ) are the boundary coordinates of the square region, (x, y) center and r are the center and radius of the circular region.

There are also two boxes which planar positions are denoted as p redbox = (x, y) redbox and p bluebox = (x, y) bluebox .

And lastly there is an interactive ball that a user can move in space which 2D coordinate is denoted as p sphere = (x, y) sphere (all objects move in the table plane).We design seven tasks each specified by a scTLTL formula.

The task specifications and their English translations are provided in Appendix A. Throughout the experiments in this section, we use proximal policy search BID20 as the policy optimization method.

The hyperparameters are kept fixed across the experiments and are listed in Appendix B.

The policy is a Gaussian distribution parameterized by a feed-forward neural network with 2 hidden layers, each layer has 64 relu units.

The state and action spaces vary across tasks and comparison cases, and are described in Appendix C. We use the first task φ 1 to evaluate the learning outcome using the FSA augmented MDP.

As comparisons, we design two other rewards structures.

The first is to use the robustness ρ(s 0:T , φ) as the terminal reward for each episode and zero everywhere else, the second is a heuristic reward that aims to align with φ 1 .

The heuristic reward consists of a state that keeps track of whether the sphere is in a region and a set of quadratic distance functions.

For φ 1 , the heuristic reward is DISPLAYFORM0 Heuristic rewards for other tasks are defined in a similar manner and are not presented explicitly.

The results are illustrated in FIG3 (right).

The upper right plot shows the average robustness over training iterations.

Robustness is chosen as the comparison metric for its semantic rigor (robustness greater than zero satisfies the task specification).

The reported values are averaged over 60 episodes and the plot shows the mean and 2 standard deviations over 5 random seeds.

From the plot we can observe that the FSA augmented MDP and the terminal robustness reward performed comparatively in terms of convergence rate, whereas the heuristic reward fails to learn the task.

The FSA augmented MDP also learns a policy with lower variance in final performance.

We deploy the learned policy on the robot in simulation and record the task success rate.

For each of the three cases, we deploy the 5 policies learned from 5 random seeds on the robot and perform 10 sets of tests with randomly initialized states resulting in 50 test trials for each case.

The average success rate is presented in FIG3 (lower right).

From the results we can see that the FSA augmented MDP is able to achieve the highest rate of success and this advantage over the robustness reward is due to the low variance of its final policy.

To evaluate the policy switching technique for skill composition, we first learn four relatively simple policies π φ2 , π φ3 , π φ4 , π φ5 using the FSA augmented MDP.

Then we construct π φ6 = π φ2∧φ3 and π φ7 = π φ2∧φ3∧φ4∧φ4 using Equation (15) (It is worth mentioning that the policies learned by the robustness and heuristic rewards do not have an automaton state in them, therefore the skill composition technique does not apply).

We deploy π φ6 and π φ7 on tasks 6 and 7 for 10 trials and record the average robustness of the resulting trajectories.

As comparisons, we also learn tasks 6 and 7 from scratch using terminal robustness rewards and heuristic rewards, the results are presented in FIG4 .

We can observe from the plots that as the complexity of the tasks increase, using the robustness and heuristic rewards fail to learn a policy that satisfies the specifications while the constructed policy can reliably achieve a robustness of greater than zero.

We perform the same deployment test as previously described and looking at FIG4 (right) we can see that for both tasks 6 and 7, only the policies constructed by skill composition are able to consistently complete the tasks.

In this paper, we proposed the FSA augmented MDP, a product MDP that enables effective learning of hierarchical policies using any RL algorithm for tasks specified by scTLTL.

We also introduced automata guided skill composition, a technique that combines existing skills to create new skills without additional learning.

We show in robotic simulations that using the proposed methods we enable simple policies to perform logically complex tasks.

Limitations of the current framework include discontinuity at the point of switching (for Equation (15)), which makes this method suitable for high level decision tasks but not for low level control tasks.

The technique only compares robustness at the current step and chooses to follow a sub-policy for one time-step, making the switching policy short-sighted and may miss long term opportunities.

One way to address this is to impose a termination condition for following each sub-policy and terminate only when the condition is triggered (as in the original options framework).

This termination condition can be hand designed or learned

For experiments with the simulated Baxter robot, we delegate low level control to motion planning packages and only learn high level decisions.

Depending on the task, the states are the planar positions of objects (red box, blue box, ball) and the automata state.

The actions are the target positions of the objects.

We assume that the low level controller can take objects to the desired target position with minor uncertainty that will be dealt with by the learning agent.

The For tasks φ 6 and φ 7 , the action space is three dimensional, the first two dimension p = (x, y) is a target position, the third dimension d controls which object should be placed at p. for t =0 to T do a t = π(s t ) 7:s t+1 = GetNextState(s t , a t )

<|TLDR|>

@highlight

Combine temporal logic with hierarchical reinforcement learning for skill composition

@highlight

The paper offers a strategy for constructing a product MDP out of an original MDP and the automaton associated with an LTL formula.

@highlight

Proposes to join temporal logic with hierarchical reinforcement learning to simplify skill composition.