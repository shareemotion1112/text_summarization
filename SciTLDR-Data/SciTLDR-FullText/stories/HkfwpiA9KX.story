Skills learned through (deep) reinforcement learning often generalizes poorly across tasks and re-training is necessary when presented with a new task.

We present a framework that combines techniques in formal methods with reinforcement learning (RL) that allows for the convenient specification of complex temporal dependent tasks with logical expressions and construction of new skills from existing ones with no additional exploration.

We provide theoretical results for our composition technique and evaluate on a simple grid world simulation as well as a robotic manipulation task.

Policies learned using reinforcement learning aim to maximize the given reward function and are often difficult to transfer to other problem domains.

Skill composition is the process of constructing new skills out of existing ones (policies) with little to no additional learning.

In stochastic optimal control, this idea has been adopted by BID20 and BID9 to construct provably optimal control laws based on linearly solvable Markov decision processes.

Temporal logic (TL) is a formal language commonly used in software and digital circuit verification BID7 as well as formal synthesis BID8 .

It allows for convenient expression of complex behaviors and causal relationships.

TL has been used by BID19 , BID11 , BID10 to synthesize provably correct control policies.

BID6 have also combined TL with Q-learning to learn satisfiable policies in discrete state and action spaces.

We make the distinction between skill composition and multi-task learning/meta-learning where the latter often requires a predefined set of tasks/task distributions to learn and generalize from, whereas the focus of the former is to construct new policies from a library of already learned policies that achieve new tasks (often some combination of the constituent tasks) with little to no additional constraints on task distribution at learning time.

In this work, we focus on skill composition with policies learned using automata guided reinforcement learning BID15 .

We adopt the syntactically co-safe truncated linear temporal logic (scTLTL) as the task specification language.

Compared to most heuristic reward structures used in the RL literature, formal specification language has the advantage of semantic rigor and interpretability.

In our framework, skill composition is accomplished by taking the product of finite state automata (FSA).

Instead of interpolating/extrapolating among learned skills/latent features, our method is based on graph manipulation of the FSA.

Therefore, the outcome is much more transparent.

Compared with previous work on skill composition, we impose no constraints on the policy representation or the problem class.

We validate our framework in simulation (discrete state and action spaces) and experimentally on a Baxter robot (continuous state and action spaces).

Recent efforts in skill composition have mainly adopted the approach of combining value functions learned using different rewards.

BID16 constructs a composite policy by combining the value functions of individual policies using the Boltzmann distribution.

With a similar goal, BID24 achieves task space transfer using deep successor representations BID14 .

However, it is required that the reward function be represented as a linear combination of state-action features.

have showed that when using energy-based models BID12 , an approximately optimal composite policy can result from taking the average of the Q-functions of existing policies.

The resulting composite policy achieves the −AN D− task composition i.e. the composite policy maximizes the average reward of individual tasks.van BID21 have taken this idea a step further and showed that by combining individual Q-functions using the log-sum-exponential function, the −OR− task composition (the composite policy maximizes the (soft) maximum of the reward of constituent tasks) can be achieved optimally.

We build on the results of BID21 and show that incorporating temporal logic allows us to compose tasks of greater logical complexity with higher interpretability.

Our composite policy is optimal in both −AN D− and −OR− task compositions.

We start with the definition of a Markov Decision Process.

Definition 1.

An MDP is defined as a tuple M = S, A, p(·|·, ·), r(·, ·, ·) , where S ⊆ IR n is the state space ; A ⊆ IR m is the action space (S and A can also be discrete sets); p : S ×A×S → [0, 1] is the transition function with p(s |s, a) being the conditional probability density of taking action a ∈ A at state s ∈ S and ending up in state s ∈ S; r : S × A × S → IR is the reward function with r(s, a, s ) being the reward obtained by executing action a at state s and transitioning to s .In entropy-regularized reinforcement learning BID18 , the goal is to maximize the following objective DISPLAYFORM0 where π : S × A → [0, 1] is a stochastic policy.

E π is the expectation following π.

H(π(·|s t )) is the entropy of π.

α is the temperature parameter.

In the limit α → 0, Equation (1) becomes the standard RL objective.

The soft Q-learning algorithm introduced by BID12 optimizes the above objective and finds a policy represented by an energy-based model DISPLAYFORM1 where E(s t , a t ) is an energy function that can be represented by a function approximator.

Let r ent t = r t + αH(π(·|s t )), the state-action value function (Q-function) following π is defined as DISPLAYFORM2 Suppose we have a set of n tasks indexed by i, i ∈ {0, ..., n}, each task is defined by an MDP M i that differs only in their reward function r i .

Let Q π i α be the optimal entropy-regularized Q-function.

Authors of BID21 provide the following results DISPLAYFORM3 .

Given a set of nonnegative weights w with ||w|| = 1, the optimal Q-function for a new task defined by r = α log(|| exp(r/α)|| w ) is given by DISPLAYFORM4 where || · || w is the weighted 1-norm.

The authors proceed to provide the following corollary DISPLAYFORM5 is the optimal Q-function for the objective DISPLAYFORM6 Corollary 1 states that in the low temperature limit, the maximum of the optimal entropy-regularized Q-functions approaches the standard optimal Q-function 3.2 SCTLTL AND FINITE STATE AUTOMATAWe consider tasks specified with syntactically co-safe Truncated Linear Temporal Logic (scTLTL) which is derived from truncated linear temporal logic(TLTL) BID15 .

The syntax of scTLTL is defined as DISPLAYFORM7 where is the True Boolean constant.

s ∈ S is a MDP state in Definition 1.

f (s) < c is a predicate over the MDP states where c ∈ IR.

¬ (negation/not), ∧ (conjunction/and) are Boolean connectives.

♦ (eventually), U (until), T (then), (next), are temporal operators.⇒ (implication) and and ∨ (disjunction/or) can be derived from the above operators.

We denote s t ∈ S to be the MDP state at time t, and s t:t+k to be a sequence of states (state trajectory) from time t to t + k, i.e., s t:t+k = s t s t+1 ...s t+k .

The Boolean semantics of scTLTL is defined as: DISPLAYFORM8 DISPLAYFORM9 A trajectory s 0:T is said to satisfy formula φ if s 0:T |= φ.

The quantitative semantics (also referred to as robustness) is defined recursively as DISPLAYFORM10 where ρ max represents the maximum robustness value.

A robustness of greater than zero implies that s t:t+k satisfies φ and vice versa (ρ(s t:t+k , φ) > 0 ⇒ s t:t+k |= φ and ρ(s t:t+k , φ) < 0 ⇒ s t:t+k |= φ).

The robustness is used as a measure of the level of satisfaction of a trajectory s 0:T with respect to a scTLTL formula φ.

An FSA corresponding to a scTLTL formula φ. is defined as a tuple DISPLAYFORM0 where Q φ is a set of automaton states; Ψ φ is the input alphabet (a set of first order logic formula); q φ,0 ∈ Q φ is the initial state; DISPLAYFORM1 F φ is a set of final automaton states.

Here q φ,i is the i th automaton state of A φ .

ψ q φ,i ,q φ,j ∈ Ψ φ is the predicate guarding the transition from q φ,i to q φ,j .

Because ψ q φ,i ,q φ,j is a predicate without temporal operators, the robustness ρ(s t:t+k , ψ q φ,i ,q φ,j ) is only evaluated at s t .

Therefore, we use the shorthand ρ(s t , ψ q φ,i ,q φ,j ) = ρ(s t:t+k , ψ q φ,i ,q φ,j ).

The translation from a TLTL formula to a FSA can be done automatically with available packages like Lomap BID22 .

The FSA Augmented MDP is defined as follows DISPLAYFORM0 is the probability of transitioning tos givens and a, DISPLAYFORM1 p φ is defined in Equation (5).r :S ×S → IR is the FSA augmented reward function, defined bỹ DISPLAYFORM2 where D q φ φ = q φ ∈Ωq φ ψ q φ ,q φ represents the disjunction of all predicates guarding the transitions that originate from q φ (Ω q φ is the set of automata states that are connected with q through outgoing edges).A policy π φ is said to satisfy φ if DISPLAYFORM3 where 1(ρ(s 0:T , φ) > 0) is an indicator function with value 1 if ρ(s 0:T , φ) > 0 and 0 otherwise.

As is mentioned in the original paper, there can be multiple policies that meet the requirement of Equation FORMULA16 , therefore, a discount factor is used to find a maximally satisfying policy -one that leads to satisfaction in the least amount of time.

The FSA augmented MDP M φ establishes a connection between the TL specification and the standard reinforcement learning problem.

A policy learned using M φ has implicit knowledge of the FSA through the automaton state q φ ∈ Q φ .

We will take advantage of this characteristic in our skill composition framework.

Problem 1.

Given two scTLTL formula φ 1 and φ 2 and their optimal Q-functions Q φ1 and Q φ2 , obtain the optimal policy π φ∧ that satisfies φ ∧ = φ 1 ∧ φ 2 and π φ∨ that satisfies φ ∨ = φ 1 ∨ φ 2 .Here Q φ1 and Q φ2 can be the optimal Q-functions for the entropy-regularized MDP or the standard MDP.

Problem 1 defines the problem of skill composition: given two policies each satisfying a scTLTL specification, construct the policy that satisfies the conjunction (−AN D−)/disjunction (−OR−) of the given specifications.

Solving this problem is useful when we want to break a complex task into simple and manageable components, learn a policy that satisfies each component and "stitch" all the components together so that the original task is satisfied.

It can also be the case that as the scope of the task grows with time, the original task specification is amended with new items.

Instead of having to re-learn the task from scratch, we can learn only policies that satisfies the new items and combine them with the old policy.

In this section, we provide a solution for Problem 1 by constructing the FSA of φ ∧ from that of φ 1 and φ 2 and using φ ∧ to synthesize the policy for the combined skill.

We start with the following definition.

Definition 4.

Given A φ1 = Q φ1 , Ψ φ1 , q φ1,0 , p φ1 , F φ1 and A φ2 = Q φ2 , Ψ φ2 , q φ2,0 , p φ2 , F φ2 corresponding to formulas φ 1 and φ 2 , the FSA of φ ∧ = φ 1 ∧φ 2 is the product automaton of A φ1 and DISPLAYFORM0 is the set of product automaton states, q φ∧,0 = (q φ1,0 , q φ2,0 ) is the product initial state, F φ∧ ⊆ F φ1 ∩ F φ2 are the final accepting states.

Following Definition 2, for states q φ∧ = (q φ1 , q φ2 ) ∈ Q φ∧ and q φ∧ = (q φ1 , q φ2 ) ∈ Q φ∧ , the transition probability p φ∧ is defined as DISPLAYFORM1 Example 1.

FIG1 illustrates the FSA of A φ1 and A φ2 and their product automaton A φ∧ .

Here φ 1 = ♦r ∧ ♦g which entails that both r and g needs to be true at least once (order does not matter), and φ 2 = ♦b. The resultant product corresponds to the formula φ = ♦r ∧ ♦g ∧ ♦b.

We provide the following theorem on automata guided skill composition DISPLAYFORM2 Proof.

For q φ∧ = (q φ1 , q φ2 ) ∈ Q φ∧ , let Ψ q φ ∧ , Ψ q φ 1 and Ψ q φ 2 denote the set of predicates guarding the edges originating from q φ∧ , q φ1 and q φ2 respectively.

Equation (9) entails that a transition at q φ∧ in the product automaton A φ∧ exists only if corresponding transitions at q φ1 , q φ2 exist in A φ1 and A φ2 respectively.

Therefore, DISPLAYFORM3 ∈ Ψ q φ 2 (here q φi is a state such that p φi (q φi |q φi ) = 1).

Therefore, we have DISPLAYFORM4 where q φ1 , q φ2 don't equal to q φ1 , q φ2 at the same time (to avoid self looping edges).

Using the fact that ψ q φ i ,q φ i = ¬ q φ i =q φ i ψ q φ i ,q φ i and repeatedly applying the distributive laws DISPLAYFORM5 Letr φ∧ ,r φ1 ,r φ2 ands φ∧ ,s φ1 ,s φ2 be the reward functions and states for FSA augmented MDP M φ∧ , M φ1 , M φ2 respectively.

s φ∧ , s φ1 , s φ2 are the states for the corresponding MDPs.

Plugging Equation (11) into Equation (7) and using the robustness definition for disjunction results iñ DISPLAYFORM6 Looking at Theorem 1, the log-sum-exp of the composite reward r = α log(|| exp(r/α)|| w ) is in fact an approximation of the maximum function.

In the low temperature limit we have r → max(r) as α → 0.

Applying Corollary 1 results in Theorem 2.Having obtained the optimal Q-function, a policy can be constructed by taking the greedy step with respective to the Q-function in the discrete action case.

For the case of continuous action space where the policy is represented by a function approximator, the policy update procedure in actorcritic methods can be used to extract a policy from the Q-function.

In our framework, −AN D− and −OR− task compositions follow the same procedure (Theorem 2).

The only difference is the termination condition.

For −AN D− task, the final state F φ∧ = F φi in Definition 4 needs to be reached (i.e. all the constituent FSAs are required to reach terminal state, as in state q φ∧,f in FIG1 .

Whereas for the −OR− task, only F φ∨ = F φi needs to be reached (one of states q φ∧,2 , q φ∧,4 , q φ∧,5 , q φ∧,6 , q φ∧,f in FIG1 .

A summary of the composition procedure is provided in Algorithm 1.In Algorithm 1, steps 3 and 4 seeks to obtain the optimal policy and Q-function using any off-policy actor critic RL algorithm.

B φ1,2 are the replay buffers collected while training for each skill.

Step φ1 , B φ1 ← ActorCritic(M φ1 ) learns the optimal policy and Q-function DISPLAYFORM7 ) construct the optimal composed Q-function using Theorem 2 6: B φ∧ ← ConstructP roductBuf f er(B φ1 , B φ2 ) DISPLAYFORM8 6 constructs the product replay buffer for policy extraction.

This step is necessary because each B φi contains state of form (s, q i ), i ∈ {1, 2} whereas the composed policy takes state (s, q 1 , q 2 ) as input (as in Definition 4).

Therefore, we transform each experience ((s, q i ), a, (s , q i ), r) to ((s, q i , q j =i ), a, (s , q i , q j =i ), r) where q j =i is chosen at random from the automaton states of A φj and q j =i is calculated from Equation BID3 .

The reward r will not be used in policy extraction as the Q-function will not be updated.

Step 7 extracts the optimal composed policy from the optimal composed Q-function (this corresponds to running only the policy update step in the actor critic algorithm).

We evaluate the our composition method in two environments.

The first is a simple 2D grid world environment that is used as for proof of concept and policy visualization.

The second is a robot manipulation environment.

Consider an agent that navigates in a 8 × 10 grid world.

Its MDP state space is S : X × Y where x, y ∈ X, Y are its integer coordinates on the grid.

The action space is A : [up, down, left, right, stay] .

The transition is such that for each action command, the agent follows that command with probability 0.8 or chooses a random action with probability 0.2.

We train the agent on two tasks, φ 1 = ♦r ∧ ♦g and φ 2 = ♦b (same as in Example 1).

The regions are defined by the predicates r = (1 < x < 3) ∧ (1 < y < 3) and g = (4 < x < 6) ∧ (4 < y < 6).

Because the coordinates are integers, a and b define a point goal rather than regions.

φ 2 expresses a similar task for b = (1 < x < 3) ∧ (6 < y < 8).

FIG1 shows the FSA for each task.

We apply standard tabular Q-learning BID23 on the FSA augmented MDP of this environment.

For all experiments, we use a discount factor of 0.95, learning rate of 0.1, episode horizon of 200 steps, a random exploration policy and a total number of 2000 update steps which is enough to reach convergence (learning curve is not presented here as it is not the focus of this paper).

FIG3 show the learned optimal policies extracted by π φi (x, y, q φi ) = arg max a Q φi (x, y, q φi , a).

We plot π φi (x, y, q φi ) for each q φi and observe that each represents a sub-policy whose goal is given by Equation 7.Figure 2 (c) shows the composed policy of φ ∧ = φ 1 ∧ φ 2 using Theorem 2.

It can be observed that the composed policy is able to act optimally in terms maximizing the expected sum of discounted rewards given by Equation (12).

Following the composed policy and transitioning the FSA in FIG1 will in fact satisfy φ ∧ (−AN D−).

As discussed in the previous section, if the −OR− task is desired, following the same composed policy and terminate at any of the states q φ∧,2 , q φ∧,4 , q φ∧,5 , q φ∧,6 , q φ∧,f will satisfy φ ∨ = φ 1 ∨ φ 2 .

In this sub-section, we test our method on a more complex manipulation task.

FIG5 (a) presents our experiment setup.

Our policy controls the 7 degree-of-freedom joint velocities of the right arm of a Baxter robot.

In front of the robot are three circular regions (red, green, blue plates) and it has to learn to traverse in user specified ways.

The positions of the plates are tracked by motion capture systems and thus fully observable.

In addition, we also track the position of one of the user's hands (by wearing a glove with trackers attached).

Our MDP state space is 22 dimensional that includes 7 joint angles, xyz positions of the three regions (denoted by p r , p g , p b ), the user's hand (p h ) and the robot's end-effector (p ee ).

State and action spaces are continuous in this case.

We define the following predicates DISPLAYFORM0 where is a threshold which we set to be 5 centimeters.

ψ i constrains the relative distance between the robot's end-effector and the selected object.

DISPLAYFORM1 This predicate evaluates to true if the user's hand appears in the cubic region defined by [x min , x max , y min , y max , z min , z max ].

In this experiment, we take this region to be 40 centimeters above the table (length and width the same as the table).We test our method on the following composition task episode, the joint angles, the FSA state, the position of the plates as well as the position of the hand (represented by the yellow sphere in FIG5 (a)) are randomly reset (within certain boundaries) to ensure generalization across different task configurations.

The robot is controlled at 20 Hz.

Each episode is 100 time-steps (about 5 seconds).

The episode restarts if the final automaton state q f is reached.

During training, we perform 100 policy and Q-function updates every 5 episodes of exploration.

All of our training is performed in simulation and for this set of tasks, the policy is able to transfer to the real robot without further fine-tuning.

In FIG6 (left), we report the discounted return as a function of policy update steps for task φ ∧ .

5 evaluation episodes are collected after each set policy updates to calculate the performance statistics.

As comparison, we learn the same task using SQL with FSA augmented MDP.

We can see that our composition method takes less update steps to reach a policy that achieves higher returns with lower variance than the policy obtained from learning.

FIG6 (right) shows the episode length as a function of policy update (upper bound clipped at 100 steps).

As mentioned in the previous section, a shorter episode length indicates faster accomplishment of the task.

It can be observed that both the composition and learning method result in high variances likely due to the randomized task configuration (some plate/joint/hand configurations make the task easier to accomplish than others).

However, the policy obtained from composition achieves a noticeable decrease in the average episode length.

It is important to note that the wall time for learning a policy is significantly longer than that from composition.

For robotic tasks with relatively simple policy representations (feed-forward neural networks), learning time is dominated by the time used to collect experiences and the average episode length (recall that we update the policy 100 times with each 5 episodes of exploration).

Since skill composition uses already collected experience, obtaining a policy can be much faster.

TAB1 shows the mean training time and standard deviation (over 5 random seeds) for each task (tasks φ traverse , φ interrupt and φ ∧ (learned) are trained for 80K policy updates.

φ ∧ (composed) is trained for 40K policy updates).

In general, training time is shorter for tasks with higher episodic success rate and shorter episode length.

We also show the task success rate evaluated on the real robot over 20 evaluation trials.

Task success is evaluated by calculating the robustness of the trajectories resulting from executing each policy.

A robustness of greater than 0 evaluates to success and vice versa.

π φ∧ (learned) fails to complete the task even though a convergence is reached during training.

This is likely due to the large FSA of φ ∧ with complex per-step reward (D q φ in Equation FORMULA15 ) which makes learning difficult.

FIG5 shows an evaluation run of the composed policy for task φ ∧ .

We provide a technique that takes advantage of the product of finite state automata to perform deterministic skill composition.

Our method is able to synthesize optimal composite policies for −AN D− and −OR− tasks.

We provide theoretical results on our method and show its effectiveness on a grid world simulation and a real world robotic task.

For future work, we will adapt our method to the more general case of task-space transfer -given a library of optimal policies (Qfunctions) that each satisfies its own specification, construct a policy that satisfies a specification that's an arbitrary (temporal) logical combination of the constituent specifications.

@highlight

A formal method's approach to skill composition in reinforcement learning tasks

@highlight

The paper combines RL and constraints expressed by logical formulas by setting up an automation from scTLTL formulas.

@highlight

Proposes a method that helps to construct policy from learned subtasks on the topic of combining RL tasks with linear temporal logic formulas.