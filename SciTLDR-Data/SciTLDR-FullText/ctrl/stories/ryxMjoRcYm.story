This paper proposes a method for efficient training of Q-function for continuous-state Markov Decision Processes (MDP), such that the traces of the resulting policies satisfy a Linear Temporal Logic (LTL) property.

LTL, a modal logic, can express a wide range of time-dependent logical properties including safety and liveness.

We convert the LTL property into a limit deterministic Buchi automaton with which a synchronized product MDP is constructed.

The control policy is then synthesised by a reinforcement learning algorithm assuming that no prior knowledge is available from the MDP.

The proposed method is evaluated in a numerical study to test the quality of the generated control policy and is compared against conventional methods for policy synthesis such as MDP abstraction (Voronoi quantizer) and approximate dynamic programming (fitted value iteration).

Markov Decision Processes (MDPs) are extensively used as a family of stochastic processes in automatic control, computer science, economics, etc.

to model sequential decision-making problems.

Reinforcement Learning (RL) is a machine learning algorithm that is widely used to train an agent to interact with an MDP when the stochastic behaviour of the MDP is initially unknown.

However, conventional RL is mostly focused on problems in which MDP states and actions are finite.

Nonetheless, many interesting real-world tasks, require actions to be taken in response to high-dimensional or real-valued sensory inputs BID5 .

For example, consider the problem of drone control in which the drone state is represented as its Euclidean position (x, y, z) ∈ R 3 .Apart from state space discretisation and then running vanilla RL on the abstracted MDP, an alternative solution is to use an approximation function which is achieved via regression over the set of samples.

At a given state, this function is able to estimate the value of the expected reward.

Therefore, in continuous-state RL, this approximation replaces conventional RL state-action-reward look-up table which is used in finite-state MDPs.

A number of methods are available to approximate the expected reward, e.g. CMACs BID34 , kernel-based modelling BID22 , tree-based regression BID7 , basis functions BID3 , etc.

Among these methods, neural networks offer great promise in reward modelling due to their ability to approximate any non-linear function BID13 .

There exist numerous successful applications of neural networks in RL for infinite or large-state space MDPs, e.g. Deep Q-networks BID19 , TD-Gammon BID36 , Asynchronous Deep RL BID20 , Neural Fitted Q-iteration BID26 , CACLA BID39 .In this paper, we propose to employ feedforward networks (multi-layer perceptrons) to synthesise a control policy for infinite-state MDPs such that the generated traces satisfy a Linear Temporal Logic (LTL) property.

LTL allows to specify complex mission tasks in a rich time-dependent formal language.

By employing LTL we are able to express complex high-level control objectives that are hard to express and achieve for other methods from vanilla RL BID35 BID31 to more recent developments such as Policy Sketching BID1 .

Examples include liveness and cyclic properties, where the agent is required to make progress while concurrently executing components, to take turns in critical sections or to execute a sequence of tasks periodically.

The purpose of this work is to show that the proposed architecture efficiently performs and is compatible with RL algorithms that are core of recent developments in the community.

Unfortunately, in the domain of continuous-state MDPs, to the best of our knowledge, no research has been done to enable RL to generate policies according to full LTL properties.

On the other hand, the problem of control synthesis in finite-state MDPs for temporal logic has been considered in a number of works.

In BID41 , the property of interest is an LTL property, which is converted to a Deterministic Rabin Automaton (DRA).

A modified Dynamic Programming (DP) algorithm is then proposed to maximise the worst-case probability of satisfying the specification over all transition probabilities.

Notice that in this work the MDP must be known a priori.

BID8 and BID2 assume that the given MDP has unknown transition probabilities and build a Probably Approximately Correct MDP (PAC MDP), which is producted with the logical property after conversion to DRA.

The goal is to calculate the value function for each state such that the value is within an error bound of the actual state value where the value is the probability of satisfying the given LTL property.

The PAC MDP is generated via an RL-like algorithm and standard value iteration is applied to calculate the values of states.

Moving away from full LTL logic, scLTL is proposed for mission specification, with which a linear programming solver is used to find optimal policies.

The concept of shielding is employed in BID0 to synthesise a reactive system that ensures that the agent stays safe during and after learning.

However, unlike our focus on full LTL expressivity, BID0 adopted the safety fragment of LTL as the specification language.

This approach is closely related to teacher-guided RL BID37 , since a shield can be considered as a teacher, which provides safe actions only if absolutely necessary.

The generated policy always needs the shield to be online, as the shield maps every unsafe action to a safe action.

Almost all other approaches in safe RL either rely on ergodicity of the underlying MDP, e.g. (Moldovan & Abbeel, 2012) , which guarantees that any state is reachable from any other state, or they rely on initial or partial knowledge about the MDP, e.g. BID32 and BID17 ).

Definition 2.1 (Continuous-state Space MDP) The tuple M = (S, A, s 0 , P, AP, L) is an MDP over a set of states S = R n , where A is a finite set of actions, s 0 is the initial state and P : DISPLAYFORM0 ] is a Borel-measurable transition kernel which assigns to any state and any action a probability measure on the Borel space (R n , B(R n )) BID6 .

AP is a finite set of atomic propositions and a labelling function L : S → 2 AP assigns to each state s ∈ S a set of atomic propositions L(s) ⊆ 2 AP BID6 .A finite-state MDP is a special case of continuous-state space MDP in which |S| < ∞ and P : DISPLAYFORM1 is the transition probability function.

The transition function P induces a matrix which is usually known as transition probability matrix in the literature.

Theorem 2.1 In any MDP M with bounded reward function and finite action space, if there exists an optimal policy, then that policy is stationary and deterministic.

BID24 BID4 .An MDP M is said to be solved if the agent discovers an optimal policy Pol * : S → A to maximize the expected reward.

From Definitions A.3 and A.4 in Appendix, it means that the agent has to take actions that return the highest expected reward.

Note that the reward function for us as the designer is known in the sense that we know over which state (or under what circumstances) the agent will receive a given reward.

The reward function specifies what the agent needs to achieve but not how to achieve it.

Thus, the objective is that the agent itself comes up with an optimal policy.

In the supplementary materials in Section A.2, we present fundamentals of approaches introduced in this paper for solving infinite-state MDPs.

In order to specify a set of desirable constraints (i.e. properties) over the agent policy we employ Linear Temporal Logic (LTL) BID23 ).

An LTL formula can express a wide range of properties, such as safety and persistence.

LTL formulas over a given set of atomic propositions AP are syntactically defined as DISPLAYFORM0 (1) We define the semantics of LTL formula next, as interpreted over MDPs.

Given a path ρ, the i-th state of ρ is denoted by DISPLAYFORM1 Definition 3.1 (LTL Semantics) For an LTL formula ϕ and for a path ρ, the satisfaction relation ρ |= ϕ is defined as DISPLAYFORM2 .]

|= ϕ 1 Using the until operator we are able to define two temporal modalities: (1) eventually, ♦ϕ = true∪ϕ; and (2) always, ϕ = ¬♦¬ϕ. LTL extends propositional logic with the temporal modalities until ∪, eventually ♦, and always .

For example, in a robot control problem, statements such as "eventually get to this point" or "always stay safe" are expressible by these modalities and can be combined via logical connectives and nesting to provide general and complex task specifications.

Any LTL task specification ϕ over AP expresses the following set of words: DISPLAYFORM3 Definition 3.2 (Policy Satisfaction) We say that a stationary deterministic policy Pol satisfies an LTL formula ϕ if: DISPLAYFORM4 For an LTL formula ϕ, an alternative method to express the set of associated words, i.e., W ords(ϕ), is to employ an automaton.

Limit Deterministic Büchi Automatons (LDBA) are one of the most succinct and simplest automatons for that purpose .

We need to first define a Generalized Büchi Automaton (GBA) and then we formally introduce an LDBA.

Definition 3.3 (Generalized Büchi Automaton) A GBA N = (Q, q 0 , Σ, F, ∆) is a structure where Q is a finite set of states, q 0 ⊆ Q is the set of initial states, Σ = 2 AP is a finite alphabet, F = {F 1 , ..., F f } is the set of accepting conditions where F j ⊂ Q, 1 ≤ j ≤ f , and ∆ : Q × Σ → 2 Q is a transition relation.

Let Σ ω be the set of all infinite words over Σ. An infinite word w ∈ Σ ω is accepted by a GBA N if there exists an infinite run θ ∈ Q ω starting from q 0 where DISPLAYFORM5 where inf (θ) is the set of states that are visited infinitely often in the sequence θ.

): DISPLAYFORM6 • ∆(q, α) ⊆ Q D and |∆(q, α)| = 1 for every state q ∈ Q D and for every corresponding α ∈ Σ, DISPLAYFORM7 An LDBA is a GBA that has two partitions: initial (Q N ) and accepting (Q D ).

The accepting part includes all the accepting states and has deterministic transitions.

In this section, we propose an algorithm based on Neural Fitted Q-iteration (NFQ) that is able to synthesize a policy that satisfies a temporal logic property.

We call this algorithm LogicallyConstrained NFQ (LCNFQ).

We relate the notion of MDP and automaton by synchronizing them to create a new structure that is first of all compatible with RL and second that embraces the logical property.

DISPLAYFORM0 ⊗ is the set of accepting states such that for each s ⊗ = (s, q) ∈ F ⊗ , q ∈ F. The intuition behind the transition kernel P ⊗ is that given the current state (s i , q i ) and action a the new state is (s j , q j ) where s j ∼ P (·|s i , a) and q j ∈ ∆(q i , L(s j )).By constructing the product MDP we add an extra dimension to the state space of the original MDP.

The role of the added dimension is to track the automaton state and, hence, to synchronize the current state of the MDP with the state of the automaton and thus to evaluate the satisfaction of the associated LTL property.

Definition 4.2 (Absorbing Set) We define the set A ∈ B(S ⊗ ) to be an absorbing set if P ⊗ (A|s ⊗ , a) = 1 for all s ⊗ ∈ A and for all a ∈ A. An absorbing set is called accepting if it includes F ⊗ .

We denote the set of all accepting absorbing sets by A.Note that the defined notion of absorbing set in continuous-state MDPs is equivalent to the notion of maximum end components in finite-state MDPs.

In another word, once a trace ends up in an absorbing set (or a maximum end component) it can never escape from it BID38 .The product MDP encompasses transition relations of the original MDP and the structure of the Büchi automaton, thus it inherits characteristics of both.

Therefore, a proper reward function can lead the RL agent to find a policy that is optimal and that respects both the original MDP and the LTL property ϕ. In this paper, we propose an on-the-fly random variable reward function that observes the current state s ⊗ , the action a and observes the subsequent state s ⊗ and gives the agent a scalar value according to the following rule: DISPLAYFORM1 where DISPLAYFORM2 is a positive reward and r n = y × m × rand (s ⊗ ) is a neutral reward where y ∈ {0, 1} is a constant, 0 < m M , and rand : S ⊗ → (0, 1) is a function that generates a random number in (0, 1) for each state s ⊗ each time R is being evaluated.

The role of function rand is to break the symmetry in LCNFQ neural nets.

Note that parameter y essentially acts as a switch to bypass the effect of the rand function on R. As we will see later, this switch is only active for LCNFQ.In LCNFQ, the temporal logic property is initially specified as a high-level LTL formula ϕ.

The LTL formula is then converted to an LDBA N to form a product MDP M N (see Definition 4.1).

In order to use the experience replay technique we let the agent explore the MDP and reinitialize it when a positive reward is received or when no positive reward is received after th iterations.

The parameter th is set manually according to the MDP such that allows the agent to explore the MDP and also to prevent the sample set to explode in size.

All episode traces, i.e. experiences, are stored in the form of (s DISPLAYFORM3 is the current state in the product MDP, a is the chosen action, s ⊗ = (s , q ) is the resulting state, and R(s ⊗ , a) is the reward.

The set of past experiences is called the sample set E.Once the exploration phase is finished and the sample set is created, we move forward to the learning phase.

In the learning phase, we employ n separate multi-layer perceptrons with just one hidden layer where n = |Q| and Q is the finite cardinality of the automaton N. Each neural net is associated with a state in the LDBA and together the neural nets approximate the Q-function in the product MDP.

For each automaton state q i ∈ Q the associated neural net is called B qi : S ⊗ × A → R. Once the agent is at state s ⊗ = (s, q i ) the neural net B qi is used for the local Q-function approximation.

The set of neural nets acts as a global hybrid Q-function approximator Q : S ⊗ × A → R. Note that the neural nets are not fully decoupled.

For example, assume that by taking action a in state s ⊗ = (s, q i ) the agent is moved to state s ⊗ = (s , q j ) where q i = q j .

According to (13) the weights of B qi are updated such that B qi (s ⊗ , a) has minimum possible error to R(s DISPLAYFORM4 Therefore, the value of DISPLAYFORM5 Let q i ∈ Q be a state in the LDBA.

Then define E qi := {(·, ·, ·, ·, x) ∈ E|x = q i } as the set of experiences within E that is associated with state q i , i.e., E qi is the projection of E onto q i .

Once the Algorithm 1: LCNFQ input :MDP M, a set of transition samples E output :Approximated Q-function 1 initialize all neural nets Bq i with (s0, qi, a) as the input and rn as the output where a ∈ A is a random action 2 repeat 3 for qi = |Q| to 1 do DISPLAYFORM6 end 10 until end of trial experience set E is gathered, each neural net B qi is trained by its associated experience set E qi .

At each iteration a pattern set P qi is generated based on E qi : DISPLAYFORM7 The pattern set is used to train the neural net B qi .

We use Rprop BID27 to update the weights in each neural net, as it is known to be a fast and efficient method for batch learning BID26 .

In each cycle of LCNFQ (Algorithm 1), the training schedule starts from networks that are associated with accepting states of the automaton and goes backward until it reaches the networks that are associated to the initial states.

In this way we allow the Q-value to back-propagate through the networks.

LCNFQ stops when the generated policy satisfies the LTL property and stops improving for long enough.

Remark 4.1 We tried different embeddings such as one hot encoding BID11 and integer encoding in order to approximate the global Q-function with a single feedforward net.

However, we observed poor performance since these encoding allows the network to assume an ordinal relationship between automaton states.

Therefore, we turned to the final solution of employing n separate neural nets that work together in a hybrid manner to approximate the global Q-function.

Recall that the reward function (3) only returns a positive value when the agent has a transition to an accepting state in the product MDP.

Therefore, if accepting states are reachable, by following this reward function the agent is able to come up with a policy Pol ⊗ * that leads to the accepting states.

This means that the trace of read labels over S (see Definition 4.1) results in an automaton state to be accepting.

Therefore, the trace over the original MDP is a trace that satisfies the given logical property.

Recall that the optimal policy has the highest expected reward comparing to other policies.

Consequently, the optimal policy has the highest expected probability of reaching to the accepting set, i.e. satisfying the LTL property.

The next section studies state space discretization as the most popular alternative approach to solving infinite-state MDPs.

Inspired by BID15 , we propose a version of Voronoi quantizer that is able to discretize the state space of the product MDP S ⊗ .

In the beginning, C is initialized to consist of just one c 1 , which corresponds to the initial state.

This means that the agent views the entire state space as a homogeneous region when no apriori knowledge is available.

Subsequently, when the agent explores, the Euclidean distance between each newly visited state and its nearest neighbor is calculated.

If this distance is greater than a threshold value ∆ called "minimum resolution", or if the new state s ⊗ has a never-visited automaton state then the newly visited state is appended to C. Therefore, as the agent continues to explore, the size of C would increase until the relevant parts of the state space are DISPLAYFORM0 || 2 } is defined by the nearest neighbor rule for any i = i. The VQ algorithm is presented in Algorithm 2.

The proposed algorithm consist of several resets at which the agent is forced to re-localize to its initial state s 0 .

Each reset is called an episode, as such in the rest of the paper we call this algorithm episodic VQ.

In this section we propose a modified version of FVI that can handle the product MDP.

The global value function v : S ⊗ → R, or more specifically v : S × Q → R, consists of n number of sub-value functions where n = |Q|.

For each q j ∈ Q, the sub-value function v qj : S → R returns the value the states of the form (s, q j ).

As we will see shortly, in a same manner as LCNFQ, the sub-value functions are not decoupled.

Let P ⊗ (dy|s ⊗ , a) be the distribution over S ⊗ for the successive state given that the current state is s ⊗ and the current action is a. For each state (s, q j ), the Bellman update over each sub-value function v qj is defined as: DISPLAYFORM0 where T is the Bellman operator BID12 .

The update in (4) is a special case of general Bellman update as it does not have a running reward and the (terminal) reward is embedded via value function initialization.

The value function is initialized according to the following rule: DISPLAYFORM1 Algorithm 3: FVI input :MDP M, a set of samples {s DISPLAYFORM2 for each qj ∈ Q, Monte Carlo sampling number Z, smoothing parameter h output :approximated value function Lv 1 initialize Lv 2 sample Y Z a (si, qj), ∀qj ∈ Q, ∀i = 1, ..., k , ∀a ∈ A 3 repeat 4 for j = |Q| to 1 do 5 ∀qj ∈ Q, ∀i = 1, ..., k , ∀a ∈ A calculate Ia((si, qj)) = 1/Z y∈Y Z a (s i ,q j ) Lv(y) using FORMULA23 6 for each state (si, qj), update v q j (si) = sup a∈A {Ia((si, qj))} in (6) 7 end 8 until end of trial where r p and r n are defined in (3).

The main hurdle in executing the Bellman operator in continuous state MDPs, as in FORMULA19 , is that no analytical representation of the value function v and also sub-value functions v qj , q j ∈ Q is available.

Therefore, we employ an approximation method by introducing the operator L. The operator L constructs an approximation of the value function denoted by Lv and of each sub-value function v qj which we denote by Lv qj .

For each q j ∈ Q the approximation is based on a set of points {( DISPLAYFORM3 ⊗ which are called centers.

For each q j , the centers i = 1, ..., k are distributed uniformly over S such that they uniformly cover S.We employ a kernel-based approximator for our FVI algorithm.

Kernel-based approximators have attracted a lot of attention mostly because they perform very well in high-dimensional state spaces BID33 .

One of these methods is the kernel averager in which for any state (s, q j ) the approximate value function is represented by DISPLAYFORM4 where the kernel K : S → R is a radial basis function, such as e −|s−si|/h , and h is smoothing parameter.

Each kernel has a center s i and the value of it decays to zero as s diverges from s i .

This means that for each q j ∈ Q the approximation operator L is a convex combination of the values of the centers {s i } k i=1 with larger weight given to those values v qj (s i ) for which s i is close to s. Note that the smoothing parameter h controls the weight assigned to more distant values (see Section A.3).In order to approximate the integral in Bellman update (4) we use a Monte Carlo sampling technique BID28 ).

For each center (s i , q j ) and for each action a, we sample the next state y z a (s i , q j ) for z = 1, ..., Z times and append it to set of Z subsequent states Y Z a (s i , q j ).

We then replace the integral with DISPLAYFORM5 The approximate value function Lv is initialized according to (5).

In each cycle of FVI, the approximate Bellman update is first performed over the sub-value functions that are associated with accepting states of the automaton, i.e. those that have initial value of r p , and then goes backward until it reaches the sub-value functions that are associated to the initial states.

In this manner, we allow the state values to back-propagate through the transitions that connects the sub-value function via (7).

Once we have the approximated value function, we can generate the optimal policy by following the maximum value (Algorithm 3).

We describe a mission planning architecture for an autonomous Mars-rover that uses LCNFQ to follow a mission on Mars.

The scenario of interest is that we start with an image from the surface of Mars and then we add the desired labels from 2 AP , e.g. safe or unsafe, to the image.

We assume that we know the highest possible disturbance caused by different factors (such as sand storms) on the rover motion.

This assumption can be set to be very conservative given the fact that there might be some unforeseen factors that we did not take into account.

The next step is to express the desired mission in LTL format and run LCNFQ on the labeled image before sending the rover to Mars.

We would like the rover to satisfy the given LTL property with the highest probability possible starting from any random initial state (as we can not predict the landing location exactly).

Once LCNFQ is trained we use the network to guide the rover on the Mars surface.

We compare LCNFQ with Voronoi quantizer and FVI and we show that LCNFQ outperforms these methods.

In this numerical experiment the area of interest on Mars is Coprates quadrangle, which is named after the Coprates River in ancient Persia (see Section A.4).

There exist a significant number of signs of water, with ancient river valleys and networks of stream channels showing up as sinuous and meandering ridges and lakes.

We consider two parts of Valles Marineris, a canyon system in Coprates quadrangle FIG2 .

The blue dots, provided by NASA, indicate locations of recurring slope lineae (RSL) in the canyon network.

RSL are seasonal dark streaks regarded as the strongest evidence for the possibility of liquid water on the surface of Mars.

RSL extend downslope during a warm season and then disappear in the colder part of the Martian year BID18 .

The two areas mapped in FIG2 , Melas Chasma and Coprates Chasma, have the highest density of known RSL.For each case, let the entire area be our MDP state space S, where the rover location is a single state s ∈ S. At each state s ∈ S, the rover has a set of actions A = {left, right, up, down, stay} by which it is able to move to other states: at each state s ∈ S, when the rover takes an action a ∈ {left, right, up, down} it is moved to another state (e.g., s ) towards the direction of the action with a range of movement that is randomly drawn from (0, D] unless the rover hits the boundary of the area which forces the rover to remain on the boundary.

In the case when the rover chooses action a = stay it is again moved to a random place within a circle centered at its current state and with radius d D. Again, d captures disturbances on the surface of Mars and can be tuned accordingly.

With S and A defined we are only left with the labelling function L : S → 2 AP which assigns to each state s ∈ S a set of atomic propositions L(s) ⊆ 2AP .

With the labelling function, we are able to divide the area into different regions and define a logical property over the traces that the agent generates.

In this particular experiment, we divide areas into three main regions: neutral, unsafe and target.

The target label goes on RSL (blue dots), the unsafe label lays on the parts with very high elevation (red coloured) and the rest is neutral.

In this example we assume that the labels do not overlap each other.

Note that when the rover is deployed to its real mission, the precise landing location is not known.

Therefore, we should take into account the randomness of the initial state s 0 .

The dimensions of the area of interest in FIG2 .a are 456.98 × 322.58 km and in FIG2 DISPLAYFORM0

The first control objective in this numerical example is expressed by the following LTL formula over Melas Chasma FIG2 : DISPLAYFORM0 where n stands for "neutral", t 1 stands for "target 1", t 2 stands for "target 2" and u stands for "unsafe".

Target 1 are the RSL (blue bots) on the right with a lower risk of the rover going to unsafe region and the target 2 label goes on the left RSL that are a bit riskier to explore.

Conforming to (8) the rover has to visit the target 1 (any of the right dots) at least once and then proceed to the target 2 (left dots) while avoiding unsafe areas.

Note that according to (u → u) in (8) the agent is able to go to unsafe area u (by climbing up the slope) but it is not able to come back due to the risk of falling.

With FORMULA26 we can build the associated Büchi automaton as in FIG3 .a.

The second formula focuses more on safety and we are going to employ it in exploring Coprates Chasma ( FIG2 where a critical unsafe slope exists in the middle of this region.

DISPLAYFORM1 In (9), t refers to "target", i.e. RSL in the map, and u stands for "unsafe".

According to this LTL formula, the agent has to eventually reach the target (♦t) and stays there ( (t → t)).

However, if the agent hits the unsafe area it can never comes back and remains there forever ( (u → u)).

With (9) we can build the associated Büchi automaton as in FIG3 .b.

Having the Büchi automaton for each formula, we are able to use Definition 4.1 to build product MDPs and run LCNFQ on both.

This section presents the simulation results.

All simulations are carried on a machine with a 3.2GHz Core i5 processor and 8GB of RAM, running Windows 7.

LCNFQ has four feedforward neural networks for (8) and three feedforward neural networks for (9), each associated with an automaton state in FIG3 .a and FIG3 .b.

We assume that the rover lands on a random safe place and has to find its way to satisfy the given property in the face of uncertainty.

The learning discount factor γ is also set to be equal to 0.9.

Fig. 4 in Section A.5 gives the results of learning for LTL formulas FORMULA26 and FORMULA27 .

At each state s ⊗ , the robot picks an action that yields highest Q(s ⊗ , ·) and by doing so the robot is able to generate a control policy Pol ⊗ * over the state space S ⊗ .

The control policy Pol ⊗ * induces a policy Pol * over the state space S and its performance is shown in Fig. 4 .Next, we investigate the episodic VQ algorithm as an alternative solution to LCNFQ.

Three different resolutions (∆ = 0.4, 1.2, 2 km) are used to see the effect of the resolution on the quality of the generated policy.

The results are presented in TAB1 , where VQ with ∆ = 2 km fails to find a satisfying policy in both regions, due to the coarseness of the resulted discretisation.

A coarse partitioning result in the RL not to be able to efficiently back-propagate the reward or the agent to be stuck in some random-action loop as sometimes the agent's current cell is large enough that all actions have the same value.

In TAB1 , training time is the empirical time that is taken to train the algorithm and travel distance is the distance that agent traverses from initial state to final state.

We show the generated policy for ∆ = 1.2 km in Fig. 5 in Section A.5.

Additionally, Fig. 7 in Section A.6 depicts the resulted Voronoi discretisation after implementing the VQ algorithm.

Note that with VQ only those parts of the state space that are relevant to satisfying the property are accurately partitioned.

Finally, we present the results of FVI method in Fig 6 in Section A.5 for the LTL formulas (8) and (9).

The FVI smoothing parameter is h = 0.18 and the sampling time is Z = 25 for both regions where both are empirically adjusted to have the minimum possible value for FVI to generate satisfying policies.

The number of basis points also is set to be 100, so the sample complexity of FVI is 100 × Z × |A| × (|Q| − 1).

We do not sample the states in the product automaton that are associated to the accepting state of the automaton since when we reach the accepting state the property is satisfied and there is no need for further exploration.

Hence, the last term is (|Q| − 1).

However, if the property of interest produces an automaton that has multiple accepting states, then we need to sample those states as well.

Note that in TAB1 , in terms of timing, FVI outperforms the other methods.

However, we have to remember that FVI is an approximate DP algorithm, which inherently needs an approximation of the transition probabilities.

Therefore, as we have seen in Section 6 in (7), for the set of basis points we need to sample the subsequent states.

This reduces FVI applicability as it might not be possible in practice.

Additionally, both FVI and episodic VQ need careful hyper-parameter tuning to generate a satisfying policy, i.e., h and Z for FVI and ∆ for VQ.

The big merit of LCNFQ is that it does not need any external intervention.

Further, as in TAB1 , LCNFQ succeeds to efficiently generate a better policy compared to FVI and VQ.

LCNFQ has less sample complexity while at the same time produces policies that are more reliable and also has better expected reward, i.e. higher probability of satisfying the given property.

This paper proposes LCNFQ, a method to train Q-function in a continuous-state MDP such that the resulting traces satisfy a logical property.

The proposed algorithm uses hybrid modes to automatically switch between neural nets when it is necessary.

LCNFQ is successfully tested in a numerical example to verify its performance.

e. s i+1 belongs to the smallest Borel set B such that P (B|s i , a i ) = 1 (or in a discrete MDP, P (s i+1 |s i , a i ) > 0).

We might also denote ρ as s 0 .. to emphasize that ρ starts from s 0 .Definition A.2 (Stationary Policy) A stationary (randomized) policy Pol : S × A → [0, 1] is a mapping from each state s ∈ S, and action a ∈ A to the probability of taking action a in state s. A deterministic policy is a degenerate case of a randomized policy which outputs a single action at a given state, that is ∀s ∈ S, ∃a ∈ A, Pol (s, a) = 1.In an MDP M, we define a function R : S × A → R + 0 that denotes the immediate scalar bounded reward received by the agent from the environment after performing action a ∈ A in state s ∈ S.Definition A.3 (Expected (Infinite-Horizon) Discounted Reward) For a policy Pol on an MDP M, the expected discounted reward is defined as BID35 : DISPLAYFORM0 where E Pol [·] denotes the expected value given that the agent follows policy Pol , γ ∈ [0, 1) is a discount factor and s 0 , ..., s n is the sequence of states generated by policy Pol up to time step n. Definition A.4 (Optimal Policy) Optimal policy Pol * is defined as follows: DISPLAYFORM1 where D is the set of all stationary deterministic policies over the state space S.

The simplest way to solve an infinite-state MDP with RL is to discretise the state space and then to use the conventional methods in RL to find the optimal policy BID33 .

Although this method can work well for many problems, the resulting discrete MDP is often inaccurate and may not capture the full dynamics of the original MDP.

One might argue that by increasing the number of discrete states the latter problem can be resolved.

However, the more states we have the more expensive and time-consuming our computations will be.

Thus, MDP discretisation has to always deal with the trade off between accuracy and the curse of dimensionality.

Let the MDP M be a finite-state MDP.

Q-learning (QL), a sub-class of RL algorithms, is extensively used to find the optimal policy for a given finite-state MDP BID35 .

For each state s ∈ S and for any available action a ∈ A, QL assigns a quantitative value Q : S × A → R, which is initialized with an arbitrary and finite value for all state-action pairs.

As the agent starts learning and receiving rewards, the Q-function is updated by the following rule when the agent takes action a at state s: DISPLAYFORM0 where Q(s, a) is the Q-value corresponding to state-action (s, a), 0 < µ ≤ 1 is called learning rate or step size, R(s, a) is the reward obtained for performing action a in state s, γ is the discount factor, and s is the state obtained after performing action a. Q-function for the rest of the state-action pairs remains unchanged.

Under mild assumptions, for finite-state and finite-action spaces QL converges to a unique limit, as long as every state action pair is visited infinitely often BID40 .

Once QL converges, the optimal policy Pol * : S → A can be generated by selecting the action that yields the highest Q, i.e., Pol DISPLAYFORM1 where Pol * is the same optimal policy that can be generated via DP with Bellman operation.

This means that when QL converges, we have DISPLAYFORM2 where s ∈ B is the agent new state after choosing action a at s such that P (B|s, a) = 1.

Recall the QL update rule (11), in which the agent stores the Q-values for all possible state-action pairs.

In the case when the MDP has a continuous state space it is not possible to directly use standard QL since it is practically infeasible to store Q(s, a) for every s ∈ S and a ∈ A. Thus, we have to turn to function approximators in order to approximate the Q-values of different state-action pairs of the Q-function.

Neural Fitted Q-iteration (NFQ) BID26 is an algorithm that employs neural networks BID14 to approximate the Q-function, due to the ability of neural networks to generalize and exploit the set of samples.

NFQ, is the core behind Google famous algorithm Deep Reinforcement Learning BID19 .The update rule in (11) can be directly implemented in NFQ.

In order to do so, a loss function has to be introduced that measures the error between the current Q-value and the new value that has to be assigned to the current Q-value, namely DISPLAYFORM0 Over this error, common gradient descent techniques can be applied to adjust the weights of the neural network, so that the error is minimized.

In classical QL, the Q-function is updated whenever a state-action pair is visited.

In the continuous state-space case, we may update the approximation in the same way, i.e., update the neural net weights once a new state-action pair is visited.

However, in practice, a large number of trainings might need to be carried out until an optimal or near optimal policy is found.

This is due to the uncontrollable changes occurring in the Q-function approximation caused by unpredictable changes in the network weights when the weights are adjusted for one certain state-action pair BID25 .

More specifically, if at each iteration we only introduce a single sample point the training algorithm tries to adjust the weights of the neural network such that the loss function becomes minimum for that specific sample point.

This might result in some changes in the network weights such that the error between the network output and the previous output of sample points becomes large and failure to approximate the Q-function correctly.

Therefore, we have to make sure that when we update the weights of the neural network, we explicitly introduce previous samples as well: this technique is called "experience replay" (Lin, 1992) and detailed later.

The core idea underlying NFQ is to store all previous experiences and then reuse this data every time the neural Q-function is updated.

NFQ can be seen as a batch learning method in which there exists a training set that is repeatedly used to train the agent.

In this sense NFQ is an offline algorithm as experience gathering and learning happens separately.

We would like to emphasize that neural-net-based algorithms exploit the positive effects of generalization in approximation while at the same time avoid the negative effects of disturbing previously learned experiences when the network properly learns BID26 .

The positive effect of generalization is that the learning algorithm requires less experience and the learning process is highly data efficient.

As stated earlier, many existing RL algorithms, e.g. QL, assume a finite state space, which means that they are not directly applicable to continuous state-space MDPs.

Therefore, if classical RL is employed to solve an infinite-state MDP, the state space has to be discretized first and then the new discrete version of the problem has to be tackled.

The discretization can be done manually over the state space.

However, one of the most appealing features of RL is its autonomy.

In other words, RL is able to achieve its goal, defined by the reward function, with minimum supervision from a human.

Therefore, the state space discretization should be performed as part of the learning task, instead of being fixed at the start of the learning process.

Nearest neighbor vector quantization is a method for discretizing the state space into a set of disjoint regions BID10 .

The Voronoi Quantizer (VQ) BID15 , a nearest neighbor quantizer, maps the state space S onto a finite set of disjoint regions called Voronoi cells.

The set of centroids of these cells is denoted by C = {c i } m i=1 , c i ∈ S, where m is the number of the cells.

Therefore, designing a nearest neighbor vector quantizer boils down to coming up with the set C. With C, we are able to use QL and find an approximation of the optimal policy for a continuous-state space MDP.

The details of how the set of centroids C is generated as part of the learning task in discussed in the body of the paper.

Finally, this section introduces Fitted Value Iteration (FVI) for continuous-state numerical dynamic programming using a function approximator BID9 .

In standard value iteration the goal is to find a mapping (called value function) from the state space to R such that it can lead the agent to find the optimal policy.

The value function in our setup is (10) when Pol is the optimal policy, i.e. U Pol * .

In continuous state spaces, no analytical representation of the value function is in general available.

Thus, an approximation can be obtained numerically through approximate value iteration, which involves approximately iterating the Bellman operator T on some initial value function BID33 .

FVI is explored more in the paper.

It has been proven that FVI is stable and converging when the approximation operator is non-expansive BID9 .

The operator L is said to be non-expansive if:

<|TLDR|>

@highlight

As safety is becoming a critical notion in machine learning we believe that this work can act as a foundation for a number of research directions such as safety-aware learning algorithms.