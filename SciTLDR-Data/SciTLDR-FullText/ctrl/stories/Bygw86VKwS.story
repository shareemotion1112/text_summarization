Recently, there has been a surge in interest in safe and robust techniques within reinforcement learning (RL).

Current notions of risk in RL fail to capture the potential for systemic failures such as abrupt stoppages from system failures or surpassing of safety thresholds and the appropriate responsive controls in such instances.

We propose a novel approach to fault-tolerance within RL in which the controller learns a policy can cope with adversarial attacks and random stoppages that lead to failures of the system subcomponents.

The results of the paper also cover fault-tolerant (FT) control so that the controller learns to avoid states that carry risk of system failures.

By demonstrating that the class of problems is represented by a variant of SGs, we prove the existence of a solution which is a unique fixed point equilibrium of the game and characterise the optimal controller behaviour.

We then introduce a value function approximation algorithm that converges to the solution through simulation in unknown environments.

Reinforcement learning (RL) provides the promise of adaptive agents being able to discover solutions merely through repeated interaction with their environment.

RL has been deployed in a number of real-world settings in which, using RL, an adaptive agent learns to perform complex tasks, often in environments shared by human beings.

Large scale factory industrial applications, traffic light control (Arel et al., 2010) , robotics (Deisenroth et al., 2013) and autonomous vehicles (Shalev-Shwartz et al., 2016) are notable examples of settings to which RL methods have been applied.

Numerous automated systems are however, susceptible to failures and unanticipated outcomes.

Moreover, many real-world systems amenable to RL suffer the potential for random stoppages and abrupt failures; actuator faults, failing mechanical system components, sensor failures are few such examples.

In these settings, executing preprogrammed behaviours or policies that have been trained in idealised simulated environments can prove vastly inadequate for the task of ensuring the safe execution of tasks.

Consequently, in the presence of such occurrences, the deployment of RL agents introduces a risk of catastrophic outcomes whenever the agent is required to act so as to avoid adverse outcomes in unseen conditions.

The important question of how to control the system in a way that is both robust against systemic faults and, minimises the risk of faults or damage therefore arises.

In response to the need to produce RL algorithms that execute tasks with safety guarantees, a significant amount of focus has recently been placed on safe execution, robust control and riskminimisation (Garcıa and Fernández, 2015) .

Examples include H ∞ control (Morimoto and Doya, 2001) , coherent risk, conditional value at risk (Tamar et al., 2015) .

In general, these methods introduce an objective 1 defined with an expectation measure that either penalises actions that lead to greater uncertainty or embeds a more pessimistic view of the world (for example, by biasing the transition predictions towards less desirable states).

In both cases, the resulting policies act more cautiously over the horizon of the problem as compared to policies trained with a standard objective function.

Despite the recent focus on safe methods within RL, the question of how to train an RL agent that can cope with random failures remains unaddressed.

In particular, at present the question of how to produce an RL policy that can cope with an abrupt failure of some system subcomponent has received no systematic treatment.

Similarly, the task of addressing how to produce RL policies that account for the risk of states in which such failures occur has not been addressed.

In this paper, we for the first time produce a method that learns optimal policies in response to random and adversarial systems attacks that lead to stoppages of system (sub)components that may produce adverse events.

Our method works by introducing an adversary that seeks to determine a stopping criterion to stop the system at states that lead to the worst possible (overall) outcomes for the controller.

Using a game-theoretic construction, we then show how a policy that is robust against adversarial attacks that lead to abrupt failure can be learned by an adaptive agent using an RL updating method.

In particular, the introduction of an adversary that performs attacks at states that lead to worst outcomes generates experiences for the adaptive RL agent to learn a best-response policy against such scenarios.

To tackle this problem, we construct a novel two-player stochastic game (SG) in which one of the players, the controller, is delegated the task of learning to modify the system dynamics through its actions that maximise its payoff and an adversary or 'stopper' that enacts a strategy that stops the system in such a way that maximises the controller's costs.

This produces a framework that finds optimal policies that are robust against stoppages at times that pose the greatest risk of catastrophe.

The main contribution of the paper is to perform the first systematic treatment of the problem of robust control under worst-case failures.

In particular, we perform a formal analysis of the game between the controller and the stopper.

Our main results are centered around a minimax proof that establishes the existence of a value of the game.

This is necessary for simulating the stopping action to induce fault-tolerance.

Although minimax proofs are well-known in game theory (Shapley, 1953; Maitra and Parthasarathy, 1970; Filar et al., 1991) , replacing a player's action set with stopping rules necessitates a minimax proof (which now relies on a construction of open sets) which markedly differs to the standard methods within game theory.

Additionally, crucial to our analysis is the characterisation of the adversary optimal stopping rule (Theorem 3).

Our results tackle optimal stopping problems (OSPs) under worst-case transitions.

OSPs are a subclass of optimal stochastic control (OSC) problems in which the goal is to determine a criterion for stopping at a time that maximises some state-dependent payoff (Peskir and Shiryaev, 2006) .

The framework is developed through a series of theoretical results: first, we establish the existence of a value of the game which characterises the payoff for the saddle point equilibrium (SPE).

Second, we prove a contraction mapping property of a Bellman operator of the game and that the value is a unique fixed point of the operator.

Third, we prove the existence and characterise the optimal stopping time.

We then prove an equivalence between the game of control and stopping and worst-case OSPs and show that the fixed point solution of the game solves the OSP.

Finally, using an approximate dynamic programming method, we develop a simulation-based iterative scheme that computes the optimal controls.

The method applies in settings in which neither the system dynamics nor the reward function are known.

Hence, the agent need only observe its realised rewards by interacting with the environment.

At present, the coverage of FT within RL is limited.

In (Zhang and Gao, 2018) RL is applied to tackle systems in which faults might occur and subsequently incur a large cost.

Similarly, RL is applied to a problem in (Yasuda et al., 2006) in which an RL method for Bayesian discrimination which is used to segment the state and action spaces.

Unlike these methods in which infrequent faults from the environment generate negative feedback, our method introduces an adversary that performs the task of simulating high-cost stoppages (hence, modelling faults) that induce an FT trained policy.

A relevant framework is a two-player optimal stopping game (Dynkin game) in which each player chooses one of two actions; to stop the game or continue (Dynkin, 1967) .

Dynkin games have generated a vast literature since the setting requires a markedly different analysis from standard SG theory.

In the case with one stopper and one controller such as we are concerned with, the minimax proof requires a novel construction using open sets to cope with the stopping problem for the minimax result.

Presently, the study of optimal control that combines control and stopping is limited to a few studies e.g. (Chancelier et al., 2002) .

Similarly, games of control and stopping have been analysed in continuous-time (Bayraktar et al., 2011; Baghery et al., 2013; Mguni, 2018) .

In these analyses, all aspects of the environment are known and in general, solving these problems requires computing analytic solutions to non-linear partial differential equations which are often analytically insoluble and whose solutions can only be approximated numerically at very low dimensions.

Current iterative methods in OSPs (and approximated dynamic programming methods e.g. (Bertsekas, 2008) ) in unknown environments are restricted to risk-neutral settings (Tsitsiklis and Van Roy, 1999) -introducing a notion of risk (generated adversarially) adds considerable difficulty as it requires generalisation to an SG involving a controller and stopper which alters the proofs throughout.

In particular, the solution concept is now an SG SPE, the existence of which must be established.

As we show, our framework provides an iterative method of solving OSPs with worst-case transitions in unknown environments and hence, generalises existing OSP analyses to incorporate a notion of risk.

The paper is organised as follows: we firstly give a formal description of the FT RL problem we tackle and the OSP with worst-case transitions and give a concrete example to illustrate an application of the problem.

In Sec. 2, we introduce the underlying SG framework which we use within the main theoretical analysis which we perform in Sec. 3.

Lastly, in Sec. 4, we develop an approximate dynamic programming approach that enables the optimal controls to be computed through simulation, followed by some concluding remarks.

We now describe the main problem with which we are concerned that is, FT RL.

We later prove an equivalence between the OSPs under worst-case transitions and the FT RL problem and characterise the solution of each problem.

We concern ourselves with finding a control policy that copes with abrupt system stoppages and failures at the worst possible states.

Unlike standard methods in RL and game theory that have fixed time horizons (or purely random exit times) in the following, the process is stopped by a fictitious adversary that uses a stopping strategy or rule to decide when to stop given its state observations.

In order to generate an FT control, we simulate the adversary's action whilst the controller determines its optimal policy.

This as we show, induces a form of control that is an FT best-response control.

A formal description is as follows: an agent exercises actions that influence the sequence of states visited by the system.

At each state, the agent receives a reward which is dependent on the state and the chosen action.

The agent's actions are selected by a policy π : S× A → [0, 1] -a map from the set of states S and the set of actions A to a probability.

We assume that the action set is a discrete compact set and that the agent's policy π is drawn from a compact policy set Π. The horizon of the problem is T ∈ N × {∞}. However, at any given point τ S ≤ T the system may stop (randomly) and the problem terminates where τ S ∼ f ({0, . . .

, T }) is a measurable, random exit time and f is some distribution on {0, . . .

, T }.

If after k ≤ T time steps the system stops, the agent incurs a cost of G(S k ) and the process terminates.

For any s ∈ S and for any π ∈ Π, the agent's performance function is given by:

where a ∧ b := min{a, b}, E is taken w.r.t.

the transition function P .

The performance function (1) consists of a reward function R : S× A → R which quantifies the agent's immediate reward when the system transitions from one state to the next, a bequest function G : S → R which quantifies the penalty incurred by the agent when the system is stopped and γ ∈ [0, 1[, a discount factor.

We assume R and G are bounded and measurable.

The FT control problem which we tackle is one in which the controller acts both with concern for abrupt system failures and stoppages.

In particular, the analysis is performed in sympathy with addressing the problem of how the controller should act in two scenarios -the first involves acting in environments that are susceptible to adversarial attacks or random stoppages in high costs states.

Such situations are often produced in various real-world scenarios such as engine failures in autonomous vehicles, network power failures and digital (communication) networks attacks.

The second scenario involves a controller that seeks to avoid system states that yield a high likelihood of systemic (subcomponent) failure.

Examples of this case include an agent that seeks to avoid performing tasks that increase the risk of some system failure, for example increasing stress that results in component failure or breakages within robotics.

To produce a control that is robust in these scenarios, it is firstly necessary to determine a stopping rule that stops the system at states that incur the highest overall costs.

Applying this stopping rule to the system subsequently induces a response by the controller that is robust against systemic faults at states in which stopping inflicts the greatest overall costs.

This necessitates a formalism that combines an OSP to determine an optimal (adversarial) stopping rule and secondly, a RL problem.

Hence, problem we consider is the following:

where the minimisation is taken pointwise and V is a set of stochastic processes of the form v : Ω → T where T ⊆ {0, 1, 2 . . .} is a set of stopping times.

Hereon, we employ the following shorthand R(s, a) ≡ R a s for any s ∈ S, a ∈ A. The dual objective (2) consists of finding both a stopping rule that minimises J and an optimal policy that maximises J. By considering the tasks as being delegated to two individual players, the problem becomes an SG between a controller that seeks to maximise J by manipulating state visitations through its actions and an adversarial stopper that chooses a stopping rule to stop the process in order to minimise J. We later consider a setting in which neither player has up-front knowledge of the transition model or objective function but each only observes their realised rewards.

The results of this paper also tackle OSPs under a worst-case transitions -problems in which the goal is to find a stopping ruleτ under the adverse non-linear expectation E P := min π∈Π E P,π s.th.

Here, the agent seeks to find an optimal stopping time in a problem in which the system transitions according to an adversarial (worst-case) probability measure.

To elucidate the ideas, we now provide a concrete practical example namely that of actuator failure within RL applications.

Consider an adaptive learner, for example a robot that uses a set of actuators to perform actions.

Given full operability of its set of actuators, the agent's actions are determined by a policy π : S ×A → [0, 1] which maps from the state space S and the set of actions A to a probability.

In many systems, there exists some risk of actuator failure at which point the agent thereafter can affect the state transitions by operating only a subset of its actuators.

In this instance, the agent's can only execute actions drawn from a subset of its action spaceÂ ⊂ A and hence, the agent is now restricted to policies of the form π partial : S ×Â → [0, 1] -thereafter its expected return is given by the value function V π partial (this plays the role of the bequest function G in (1)).

In order to perform robustly against actuator failure, it is therefore necessary to consider a set of stopping times T ⊆ {0, 1, 2, . . .} and a stopping criterionτ : Ω → T which determines the worst states for the agent's functionality to be impaired so that it can only use some subset of its set of actuators.

The problem involves finding a pair (τ ,π) ∈ V × Π -a stopping time and control policy s.th.

where s := s 0 , a t ∼ π and

Hence the role of the adversary is to determine and execute the stopping actionτ that leads to the greatest reduction in the controller's overall payoff.

The controller in turn learns to execute the policyπ which involves playing a policyπ partial ∈ arg max V π partial after the adversary has executed its stopping action.

The resulting policyπ is hence robust against actuator failure at the worst possible states.

Embedded within problem (4) is an interdependence between the actions of the players -that is, the solution to the problem is jointly determined by the actions of both players and their responses to each other.

The appropriate framework to tackle this problem is therefore an SG (Shapley, 1953).

In this setting, the state of the system is determined by a stochastic process {s t |t = 0, 1, 2, . . .} whose values are drawn from a state space S ⊆ R p for some p ∈ N. The state space is defined on a probability space (Ω, B, P ), where Ω is the sample space, B is the set of events and P is a map from events to probabilities.

We denote by F = (F n ) n≥0 the filtration over (Ω, B, P ) which is an increasing family of σ−algebras generated by the random variables s 1 , s 2 , . .

..

We operate in a Hilbert space V of real-valued functions on L 2 , i.e. a complete 2 vector space which we equip with a norm · :

is a probability measure.

The problem occurs over a time interval {0, . . .

K} where K ∈ N × {∞} is the time horizon.

A stopping time is defined as a random variable τ : Ω → {0, . . .

, K} for which {ω ∈ Ω|τ (ω) ≤ t} ∈ F t for any t ∈ {0, . . . , K} -this says that given the information generated by the state process, we can determine if the stopping criterion has occurred.

An SG is an augmented Markov decision process which proceeds by two players tacking actions that jointly manipulate the transitions of a system over K rounds which may be infinite.

At each round, the players receive some immediate reward or cost which is a function of the players' joint actions.

The framework is zero-sum so that a reward for player I simultaneously represents a cost for player II.

Formally, a two-player zero-sum SG is a 6−tuple S, A i∈{1,2} , P, R, γ where S = {s 1 , s 2 , . . . , s n } is a set of n ∈ N states, A i is an action set for each player i ∈ {1, 2}. The map P : S×A 1 ×A 2 ×S → [0, 1] is a Markov transition probability matrix i.e. P (s ; s, a 1 , a 2 ) is the probability of the state s being the next state given the system is in state s and actions a 1 ∈ A 1 and a 2 ∈ A 2 are applied by player I and player II (resp.).

The function R : S× A 1 × A 2 is the one-step reward for player I and represents one-step cost for player II when player I takes action a 1 ∈ A 1 and player II takes action a 2 ∈ A 2 and γ ∈ [0, 1[ is a discount factor.

The goal of each player is to maximise its expected cumulative return -since the game is antagonistic, the total expected reward received by player I which we denote by J, represents a total expected cost for player II.

Denote by Π i , the space of strategies for each player i ∈ {1, 2} .

For SGs with Markovian transition dynamics, we can safely dispense with path dependencies in the space of strategies.

3 Consequently, w.log.

we restrict ourselves to the class of behavioural strategies that depend only on the current state and round, namely Markov strategies, hence for each player i, the strategy space Π i consists of strategies of the form

It is well-known that for SGs, an equilibrium exists in Markov strategies even when the opponent can draw from non-Markovian strategies (Hill, 1979) .

In SGs, it is usual to consider the case A 1 = A 2 so that the players' actions are drawn from the same set.

We depart from this model and consider a game in which player II can choose a strategy which determines a time to stop the process contained within the set T ⊆ {0, 1, 2, . . .} which consists of F− measurable stopping times.

In this setting, player I can manipulate the system dynamics by taking actions drawn from A 1 (we hereon use A) and at each point, player II can decide to intervene to stop the game.

The value of the game exists if we can commute the max and min operators:

We denote the value by J := val

and denote by (k,π) ∈ V × Π the pair that satisfies Jk ,π ≡ J .

The value, should it exist, is the minimum payoff each player can guarantee itself under the equilibrium strategy.

In general, the functions val + [J] and val − [J] may not coincide.

Should J exist, it constitutes an SPE of the game in which neither player can improve their payoff by playing some other control -an analogous concept to a Nash equilibrium for the case of two-player zero-sum games.

Thus the central task to establish an equilibrium involves unambiguously assigning a value to the game, that is proving the existence of J .

In this section, we present the key results and perform the main analysis of the paper.

Our first task is to prove the existence of a value of the game.

This establishes a fixed or stable point which describes the equilibrium policies enacted by each player.

Crucially, the equilibrium describes the maximum payoff that the controller can expect in an environment that is subject to adversarial attacks that stop the system or some subcomponent.

Unlike standard SGs with two controllers, introducing a stopping criterion requires an alternative analysis in which i) an equilibrium with Markov strategies in which one of the players uses a stopping criterion is determined and ii) the stopping criterion is characterised.

It is well-known that introducing a stopping action to one of the players alters the analysis of SGs the standard methods of which cannot be directly applied (c.f.

Dynkin games (Dynkin, 1967)).

Our second task is to perform an analysis that enables us to construct an approximate dynamic programming method.

This enables the value function to be computed through simulation.

This, as we show in Sec. 4, underpins a simulation-based scheme that is suitable for settings in which the transition model and reward function is a priori unknown.

Lastly, we construct an equivalence between robust OSPs and games of control and stopping.

We defer some of the proofs to the appendix.

Our results develop the theory of risk within RL to cover instances in which the agent has concern the process at a catastrophic system state.

Consequently, we develop the theory of SGs to cover games of control and stopping when neither player has up-front environment knowledge.

We prove an equivalence between robust OSPs and games of control and stopping and demonstrate how each problem can be solved in unknown environments.

A central task is to prove that the Bellman operator for the game is a contraction mapping.

Thereafter, we prove convergence to the unique value.

Consider a Borel measurable function which is absolutely integrable w.r.t.

the transition kernel

ss , where P a ss ≡ P (s ; s, a) is the probability of the state s being the next state given the action a ∈ A and the current state is s .

In this paper, we denote by (P J)(s) := S J[s ]P a sds .

We now introduce the operator of the game which is of central importance:

The operator T enables the game to be broken down into a sequence of sub minimax problems.

It will later play a crucial role in establishing a value iterative method for computing the value of the game.

We now briefly discuss strategies.

A player strategy is a map from the opponent's policy set to the player's own policy set.

In general, in two player games the player who performs an action first employs the use of a strategy.

Typically, this allows the player to increase its rewards since their action is now a function of the other player's later decisions.

Markov controls use only information about the current state and duration of the game rather than using information about the opponent's decisions or the game history.

Seemingly, limiting the analysis to Markov controls in the current game may restrict the abilities of the players to perform optimally.

Our first result however proves the existence of the value in Markov controls:

Theorem 1 establishes the existence of the game which permits commuting the max and min operators of the objective (2).

Crucially, the theorem secures the existence of an equilibrium pair (τ ,π) ∈ V × Π, whereπ ∈ Π is the controller's optimal Markov policy when it faces adversarial attacks that stop the system.

Additionally, Theorem 1 establishes the existence of a given by J , the computation of which, is the subject of the next section.

We can now establish the optimal strategies for each player.

To this end, we now define best-response strategies which shall be useful for further characterising the equilibrium: Definition 1.

The set of best-response (BR) strategies for player I against the stopping time τ ∈ V (BR strategies for player II against the control policy π ∈ Π) is defined by:

The question of computing the value of the game remains.

To this end, we now prove that repeatedly applying T produces a sequence that converges to the value.

In particular, the game has a fixed point property which is stated in the following:

There exists a unique function J ∈ L 2 s.th.

Theorem 2 establishes the existence of a fixed point of T and that the fixed point coincides with the value of the game.

Crucially, it suggests that J can be computed by an iterative application of the Bellman operator which underpins a value iterative method.

We study this aspect in Sec. 4 where we develop an iterative scheme for computing J .

Definition 2.

The pair (τ ,π) ∈ V × Π is an SPE iff:

An SPE therefore defines a strategic configuration in which both players play their BR strategies.

With reference to the FT RL problem, an SPE describes a scenario in which the controller optimally responds against stoppages at the set of states that inflict the greatest costs to the controller.

In particular, we will demonstrate thatπ ∈ Π is a BR to a system that undergoes adversarial attacks.

Proposition 1.

The pair (τ ,π) ∈ V × Π consists of BR strategies and constitutes an SPE.

By Prop.

1, when the pair (τ ,π) is played, each player executes its BR strategy.

The strategic response then induces FT behaviour by the controller.

We now turn to the existence and characterising the optimal stopping time for player II.

The following result establishes its existence.

Theorem 3.

There exists an F-measurable stopping time:

The theorem characterises and establishes the existence of the player II optimal stopping time which, when executed by the adversary, induces an FT control by the controller.

Having shown the existence of the optimal stopping time τ , by Theorem 3 and Theorem 1, we find: Theorem 4.

Letτ be the player II optimal stopping time defined in (3) and let τ be the optimal stopping time for the robust OSP (c.f.

(3)) then τ =τ .

Theorem 4 establishes an equivalence between the robust OSP and the SG of control and stopping hence, any method that computesτ for the SG yields a solution to the robust OSP.

We now develop a simulation-based value-iterative scheme.

We show that the method produces an iterative sequence that converges to the value of the game from which the optimal controls can be extracted.

The method is suitable for environments in which the transition model and reward functions are not known to either player.

The fixed point property of the game established in Theorem 2 immediately suggests a solution method for finding the value.

In particular, we may seek to solve the fixed point equation (FPE) J = T J .

Direct approaches at solving the FPE are not generally fruitful as closed solutions are typically unavailable.

To compute the value function, we develop an iterative method that tunes weights of a set of basis functions {φ k : R p → R|k ∈ 1, 2, . . .

D} to approximate J through simulated system trajectories and associated costs.

Algorithms of this type were first introduced by Watkins (Watkins and Dayan, 1992) as an approximate dynamic programming method and have since been augmented to cover various settings.

Therefore the following can be considered as a generalised Q-learning algorithm for zero-sum controller stopper games.

Let us denote by Φr := D j=1 r(j)φ j an operator representation of the basis expansion.

The algorithm is initialised with weight vector r 0 = (r 0 (1), . . .

, r 0 (P )) ∈ R d .

Then as the trajectory {s t |t = 0, 1, 2, . . .} is simulated, the algorithm produces an updated series of vectors {r t |t = 0, 1, 2, . . .} by the update:

Theorem 5 demonstrates that the method converges to an approximation of J .

We provide a bound for the approximation error in terms of the basis choice.

We define the function Q which the algorithm approximates by:

We later show that Q serves to approximate the value J .

In particular, we show that the algorithm generates a sequence of weights r n that converge to a vector r and that Φr , in turn approximates Q .

To complete the connection, we provide a bound between the outcome of the game when the players use controls generated by the algorithm.

We introduce our player II stopping criterion which now takes the form:

(11) Let us define a orthogonal projection Π and the function F by the following:

ΠQ := arg min

We now state the main results of the section: Theorem 5.

r n converges to r where r is the unique solution: ΠF (Φr ) = Φr .

The following results provide approximation bounds when employing the projection Π:

, then the following hold:

Hence the error bound in approximation of J is determined by the goodness of the projection.

Theorem 5 and Theorem 6 thus enable the FT RL problem to be solved by way of simulating the behaviour of the environment and using the update rule (10) to approximate the value function.

Applying the stopping rule in (11), by Theorem 6 and Theorem 2, means the pair (τ ,π) is generated where the policyπ approximates the policyπ which is FT against adversarial stoppages and faults.

In this paper, we tackled the problem of fault-tolerance within RL in which the controller seeks to obtain a control that is robust against catastrophic failures.

To formally characterise the optimal behaviour, we constructed a new discrete-time SG of control and stopping.

We established the existence of an equilibrium value then, using a contraction mapping argument, showed that the game can be solved by iterative application of a Bellman operator and constructed an approximate dynamic programming algorithm so that the game can be solved by simulation.

Assumption A.2.

Ergodicity: i) Any invariant random variable of the state process is P −almost surely (P −a.s.) a constant.

Assumption A.3.

Markovian transition dynamics: the transition probability function P satisfies the following equality:

Assumption A.4.

The constituent functions {R, G} in J are square integrable: that is, R, G ∈ L 2 (µ).

We begin the analysis with some preliminary lemmata and definitions which are useful for proving the main results.

Definition A.1.

An operator T : V → V is said to be a contraction w.r.t a norm · if there exists a constant c ∈ [0, 1[ s.th for any V 1 , V 2 ∈ V we have that:

Definition A.2.

An operator T : V → V is non-expansive if ∀V 1 , V 2 ∈ V we have:

Definition A.3.

The residual of a vector V ∈ V w.r.t the operator T : V → V is:

Lemma A.1.

Define val + [f ] := min b∈B max a∈A f (a, b) and define val − [f ] := max a∈A min b∈B f (a, b), then for any b ∈ B we have that for any f, g ∈ L and for any c ∈ R >0 :

Lemma A.2.

For any f, g, h ∈ L and for any c ∈ R >0 we have that:

The following lemma, whose proof is deferred is a required result for proving the contraction mapping property of the operator T .

Lemma A.4.

The probability transition kernel P is non-expansive, that is:

The following estimates provide bounds on the value J which we use later in the development of the iterative algorithm.

We defer the proof of the results to the appendix.

Proposition A.1.

The operator T in (5) is a contraction.

Lemma A.5.

Let T : V → V be a contraction mapping in · and let J be a fixed point so that T J = J then there exists a constant c ∈ [0, 1[ s.th:

Lemma A.6.

Let T 1 : V → V, T 2 : V → V be contraction mappings and suppose there exists vectors J 1 , J 2 s.th T 1 J 1 = J 1 and T 2 J 2 = J 2 (i.e. J 1 , J 2 are fixed points w.r.t T 1 and T 2 respectively) then ∃c 1 , c 2 ∈ [0, 1[ s.th:

Lemma A.7.

The operator T satisfies the following:

2. (Constant shift) Let I(s) ≡ 1 be the unit function, then for any J ∈ L 2 and for any scalar α ∈ R, T satisfies T (J + αI)(s) = T J(s) + αI(s).

Proof of Lemma A.1.

We begin by noting the following inequality for any f :

have that for all b ∈ V:

From (21) we can straightforwardly derive the fact that for any b ∈ V:

(this can be seen by negating each of the functions in (21) and using the properties of the max operator).

Assume that for any b ∈ V the following inequality holds:

Since (22) holds for any b ∈ V and, by (21), we have in particular that

whenever (23) holds which gives the required result.

Lemma A.2 and Lemma A.3 are given without proof but can be straightforwardly checked.

Proof of Lemma A.4.

The proof is standard, we give the details for the sake of completion.

Indeed, using the Tonelli-Fubini theorem and the iterated law of expectations, we have that:

where we have used Jensen's inequality to generate the inequality.

This completes the proof.

Proof of Proposition A.1.

We wish to prove that:

Firstly, we observe that: 1[) and (30) .

The result follows after applying Lemma A.2 and Lemma A.3.

Proof of Lemma A.5.

The proof follows almost immediately from the triangle inequality, indeed for any J ∈ L 2 :

where we have added and subtracted T J to produce the inequality.

The result then follows after inserting the definition of T (J).

Proof of Lemma A.6.

The proof follows directly from Lemma A.5.

Indeed, we observe that for any J ∈ L 2 we have

where we have added and subtracted J to produce the inequality.

The result then follows from Lemma A.5.

Proof of Lemma A.7.

Part 2 immediately follows from the properties of the max and min operators.

It remains only to prove part 1.

We seek to prove that for any s ∈ S, if J ≤J then

We begin by firstly making the following observations: 1.

For any x, y, h ∈ V x ≤ y =⇒ min{x, h} ≤ min{y, h}.

2.

For any

Assume that J ≤J, then we observe that:

≤ γ max

where we have used (30) in the penultimate line.

The result immediately follows after applying (29).

Proof of Theorem 1.

We begin by noting the following inequality holds:

The inequality follows by noticing J k,π ≤ max We now observe that:

where we have used the stationarity property and, in the limit m → ∞ and, in the last line we used the Fatou lemma.

The constant c is given by c :

Hence, we now find that

Now since (33) holds ∀π ∈ Π we find that:

Lastly, applying min operator we observe that:

It now remains to show the reverse inequality holds:

Indeed, we observe that

= min We now apply the min operator to both sides of (46) which gives:

After taking expectations, we find that:

Now by Jensen's inequality and, using the stationarity of the state process (recall the expectation is taken under π) we have that:

By standard arguments of dynamic programming, the value of the game with horizon n can be obtained from n iterations of the dynamic recursion; in particular, we have that:

Inserting (49) and (50) into (48) gives:

where c(m) :=

Hence, we find that:

we deduce the result after noting that G(

The proofs of the results in Sec. 4 are constructed in a similar fashion that in (Bertsekas, 2008) (approximate dynamic programming).

However, the analysis incorporates some important departures due to the need to accommodate the actions of two players that operate antagonistically.

We now prove the first of the two results of Sec. 4.

Proof of Theorem 5.

We firstly notice the construction ofτ given bŷ

is sensible since we observe that min{t|G(s t ) ≤ J } = min{t|G(s t ) ≤ min{G(s t ), Q (s t )} = min{t|G(s t ) ≤ Q }.

Step 1 Our first step is to prove the following bound:

Proof.

which is the required result.

Step 2 Our next task is to prove that the quantity Q is a fixed point of F and hence we can apply the operator F to achieve the approximation of the value.

Proof.

Using the definition of T (c.f. (13) Step 3 We now prove that the operator ΠF is a contraction on Q, that is the following inequality holds:

Proof.

The proof follows straightforwardly by the properties of a projection mapping:

Step 4

The result is proven using the orthogonality of the (orthogonal) projection and by the Pythagorean theorem.

Indeed, we have that:

Proof.

Φr − Q 2 = Φr − ΠQ 2 + ΠQ − Q Hence, we find that

which is the required result.

Result 2

Proof.

The proof by Jensen's inequality, stationarity and the non-expansive property of P .

In particular, we have

Inserting the definitions of Q andQ into (57) then gives:

It remains therefore to place a bound on the term Q −Q. We observe that by the triangle inequality and the fixed point properties of F on Q andF onQ we have

≤ γ Q − Φr + Q − Φr (60)

So that

The result then follows after substituting the result of step 4 (55).

Let us now define the following quantity:

Step 5

Proof.

We now observe that s k can be described in terms of an inner product.

Indeed, using the iterated law of expectations we have that

Step 5 enables us to use classic arguments for approximate dynamic programming.

In particular, following step 5, Theorem 6 follows directly from Theorem 2 in (Tsitsiklis & Van Roy, 1999) with only a minor adjustment in substituting the max operator with min.

<|TLDR|>

@highlight

The paper tackles fault-tolerance under random and adversarial stoppages.