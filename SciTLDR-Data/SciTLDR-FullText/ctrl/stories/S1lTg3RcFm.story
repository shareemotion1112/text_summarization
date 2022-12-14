Partially observable Markov decision processes (POMDPs) are a widely-used framework to model decision-making with uncertainty about the environment and under stochastic outcome.

In conventional POMDP models, the observations that the agent receives originate from fixed known distribution.

However, in a variety of real-world scenarios the agent has an active role in its perception by selecting which observations to receive.

Due to combinatorial nature of such selection process, it is computationally intractable to integrate the perception decision with the planning decision.

To prevent such expansion of the action space, we propose a greedy strategy for observation selection that aims to minimize the uncertainty in state.

We develop a novel point-based value iteration algorithm that incorporates the greedy strategy to achieve near-optimal uncertainty reduction for sampled belief points.

This in turn enables the solver to efficiently approximate the reachable subspace of belief simplex by essentially separating computations related to perception from planning.

Lastly, we implement the proposed solver and demonstrate its performance and computational advantage in a range of robotic scenarios where the robot simultaneously performs active perception and planning.

In the era of information explosion it is crucial to develop decision-making platforms that are able to judiciously extract useful information to accomplish a defined task.

The importance of mining useful data appears in many applications including artificial intelligence, robotics, networked systems and Internet of things.

Generally in these applications, a decision-maker, called an agent, must exploit the available information to compute an optimal strategy toward a given objective.

Partially observable Markov decision processes (POMDPs) provide a framework to model sequential decision-making with partial perception of the environment and under stochastic outcomes.

The flexibility of POMDPs and their variants in modeling real-world problems has led to extensive research on developing efficient algorithms for finding near-optimal policies.

Nevertheless, the majority of previous work on POMDPs either deal with sole perception or sole planning.

While independent treatment of perception and planning deteriorates performance, an integrated approach usually becomes computationally intractable.

Thereupon, one must establish a trade-off between optimality and tractability when determining how much perception and planning rely on each other.

We show that by restricting the perception to the class of subset selection problems and exploiting submodular optimization techniques, it is possible to partially decouple computing perception and planning policies while considering their mutual effect on the overall policy value.

In this work, we consider joint perception and planning in POMDPs.

More specifically, we consider an agent that decides about two sets of actions; perception actions and planning actions.

The perception actions, such as employing a sensor, only affect the belief of the agent regarding the state of the environment.

The planning actions, such as choosing navigation direction, are the ones that affect the transition of the environment from one state to another.

In subset selection problems, at each time step, due to power, processing capability, and cost constraints, the agent can pick a subset of available information sources along a planning action.

The subset selection problem arise in various applications in control systems and signal processing, in wireless sensor networks, as well as machine learning BID7 and have been widely-studied (Shamaiah et al., 2010; BID11 .

However, the previous work on sensor selection problems assume that the planning strategy is known, while in this work, we simultaneously learn a selection strategy and a planning strategy.

Exact POMDP solvers optimize the value function over all reachable belief points.

However, finding exact solution to POMDPs is PSPACE-complete BID18 which deems solving even small POMDPs computationally intractable.

This has led to extensive search for nearoptimal algorithms.

A common technique is to sample a finite set of belief points that approximate the reachable subspace of belief and apply value iteration over this set, e.g., (Sondik, 1978; BID3 BID14 Zhang & Zhang, 2001; Spaan & Vlassis, 2005; BID19 .

BID19 proved that the errors due to belief sampling is bounded where the bound depends on the density of the belief set.

A well-established offline POMDP solver is SARSOP BID13 .

SARSOP, similar to HSVI (Smith & Simmons, 2012) , aims to minimize the gap between the lower and upper bounds on the value function by guiding the sampling toward the belief points that are reachable under union of optimal policies.

In this paper, we show that the proposed greedy observation selection scheme leads to belief points that are on expectation close to the ones from the optimal (with respect to uncertainty reduction) set of observations, and hence value loss is small.

An instance of active perception is dynamic sensor selection.

BID12 proposes a reinforcement learning approach that uses R??nyi divergence to compute utility of sensing actions.

BID8 formulated a single step sensor selection problem as semi-definite programming, however, it lacks theoretical guarantee.

In Kalman filtering setting, Shamaiah et al. (2010) developed a greedy selection scheme with near-optimal guarantee to minimize log-determinant of the error covariance matrix of estimated state.

Some prior work such as (Spaan, 2008; Spaan & Lima, 2009; BID16 model active perception as a POMDP.

However, the most relevant work to ours are that of BID0 Spaan et al., 2015; Satsangi et al., 2018) .

BID0 proposed ??POMDP framework where the reward depends on entropy of the belief.

Spaan et al. (2015) introduced POMDP-IR where the reward depends on accurate prediction about the state.

Satsangi et al. (2018) established an equivalence property between ??POMDP and POMDP-IR.

Furthermore, they employed the submodularity of value function, under some conditions, to use greedy scheme for sensor selection.

The main difference of our work is that we consider active perception as a means to accomplishing the original task while in these work, the active perception is the task itself and hence the POMDP rewards are metrics to capture perception quality.

The problem of selecting an optimal set of sensors from a ground set under cardinality constraint is NP-hard (Williams & Young, 2007) .

This hardness result has motivated design of greedy algorithms since they make polynomial oracle calls to the objective function.

Additionally, if the objective function is monotone non-decreasing and submodular, BID17 showed that a greedy selection achieves (1 ??? 1/e) approximation factor.

BID15 and BID7 developed randomized greedy schemes that accelerate the selection process for monotone submodular and weak-submodular objective functions, respectively.

BID11 ; BID10 have introduced different submodular information-theoretic objectives for greedy selection and have studied the theoretical guarantees of their maximization under different constraints.

Here, we use the entropy of belief to capture the level of uncertainty in state and aim to select a subset of sensors that leads to maximum expected reduction in entropy.

We employ the monotonicity and submodularity of the proposed objective to establish near-optimal approximation factor for entropy minimization.

A summary of our contributions are as follows:??? Formulating the active perception problem for POMDPs: We introduce a new mathematical definition of POMDPs, called AP 2 -POMDP, that captures active perception as well as planning.

The objective is to find deterministic belief-based policies for perception and planning such that the expected discounted cumulative reward is maximized.??? Developing a perception-aware point-based value iteration algorithm: To solve AP 2 -POMDP, we develop a novel point-based method that approximates the value function using a finite set of belief points.

Each belief point is associated with a perception action and a planning action.

We use the near-optimal guarantees for greedy maximization of monotone submodular functions to compute the perception action while the planning action is the result of Bellman optimality equation.

We further prove that greedy perception action leads to an expected reward that is close to that of optimal perception action.

This section starts by giving an overview of the related concepts and then stating the problem.

The standard POMDP definition models does not capture the actions related to perception.

We present a different definition which we call AP 2 -POMDP as it models active perception actions as well as original planning actions.

The active perception actions determine which subset of sensors (observations) the agent should receive.

We restrict the set of states, actions, and observations to be discrete and finite.

We formally define an AP 2 -POMDP below.

DISPLAYFORM0 ??? S is the finite set of states.??? A = A pl ?? A pr is the finite set of actions with A pl being the set of planning actions and A pr being the set of perception actions.

A pr = {?? ??? {0, 1} n | |??| 0 ??? k} constructs an n-dimensional lattice.

Each component of an action ?? ??? A pr determines whether the corresponding sensor is selected.??? k is the maximum number of sensor to be selected.

DISPLAYFORM1 is the probabilistic transition function.??? ??? = ??? 1 ?? ??? 2 ?? . . .

?? ??? n is the partitioned set of observations, where each ??? i corresponds to the set of measurements observable by sensor i.??? O : S ?? A ?? ??? ??? [0, 1] is the probabilistic observation function.??? R : S ?? A pl ??? R is the reward function, and DISPLAYFORM2 At each time step, the environment is in some state s ??? S. The agent takes an action ?? ??? A pl that causes a transition to a state s ??? S with probability P r(s |s, ??) = T (s, ??, s ).

At the same time step, the agent also picks k sensors by ?? ??? A pr .

Then it receives an observation ?? ??? ??? with probability P r(??|s , ??, ??) = O(s , ??, ??, ??), and a scalar reward R(s, ??).

Assumption 1.

We assume that the observations from sensors are mutually independent given the current state and the previous action, i.e., ???I 1 , I 2 ??? {1, 2, . . .

, n}, I 1 ??? I 2 = ??? : DISPLAYFORM3 Let ??(??) = {i|??(i) = 1} to denote the subset of sensors that are selected by ??.

If Assumption 1 holds, then: DISPLAYFORM4 DISPLAYFORM5 The belief of the agent at each time step, denoted by b t is the posterior probability distribution of states given the history of previous actions and observations, i.e., h t = (a 0 , ?? 1 , a 1 , . . . , a t???1 , ?? t ).A well-known fact is that due to Markovian property, a sufficient statistics to represent history of actions and observations is belief (??str??m, 1965; Smallwood & Sondik, 1973) .

Given the initial belief b 0 , the following update equation holds between previous belief b and the belief b a,?? b after taking action a = (??, ??) and receiving observation ??: DISPLAYFORM6 The goal is to learn a deterministic policy to maximize E[ DISPLAYFORM7 .

A deterministic policy is a mapping from belief to actions ?? : B ??? A, where B is the set of belief states.

Note that B constructs a (|S| ??? 1)-dimensional probability simplex.

The POMDP solvers apply value iteration (Sondik, 1978) , a dynamic programming technique, to find the optimal policy.

Let V be a value function that maps beliefs to values in R. The following recursive expression holds for V : DISPLAYFORM8 The value iteration converges to the optimal value function V * which satisfies the Bellman's optimality equation BID2 .

Once the optimal value function is learned, an optimal policy can be derived.

An important outcome of FORMULA8 is that at any horizon, the value function is piecewise-linear and convex (PWLC) (Smallwood & Sondik, 1973) and hence, can be represented by a finite set of hyperplanes.

Each hyperplane is associated with an action.

Let ??'s to denote the corresponding vectors of the hyperplanes and let ?? t to be the set of ?? vectors at horizon t. Then, DISPLAYFORM9 This fact has motivated approximate point based solvers that try to approximate the value function by updating the hyperplanes over a finite set of belief points.

Since the proposed algorithm is founded upon the theoretical results from the field of submodular optimization, here, we overview the necessary definitions.

Let X to denote a ground set and f a set function that maps an input set to a real number.

DISPLAYFORM0 is the marginal value of adding element i to set T 1 .Monotonicity states that adding elements to a set increases the function value while submodularity refers to diminishing returns property.

Having stated the required background, next, we state the problem.

Problem 1.

Consider a AP 2 -POMDP P = (S, A, k, T, ???, O, R, ??) and an initial belief b 0 .

We aim to learn a policy ??(b) = (??, ??) such that the expected discounted cumulative reward is maximized, i.e, It is worth noting that the perception actions affect the belief and subsequently the received reward in the objective function.

DISPLAYFORM0 3 ACTIVE PERCEPTION WITH GREEDY SCHEME For variety of performance metrics, finding an optimal subset of sensors poses a computationally challenging combinatorial optimization problem that is NP-hard.

Augmenting POMDP planning actions with n k active perception actions results in a combinatorial expansion of the action space.

Thereupon, it is infeasible to directly apply existing POMDP solvers to Problem 1.

Instead of concatenating both sets of actions and treating them similarly, we propose a greedy strategy for selecting perception actions that aims to pick the sensors that result in minimal uncertainty about the state.

The key enabling factor is that the perception actions does not affect the transition, consequently, we can decompose the single-step belief update in (2) into two steps: DISPLAYFORM1 This in turn implies that after a transition is made, the agent should pick a subset of observations that lead to minimal uncertainty in b DISPLAYFORM2 To quantify uncertainty in state, we use Shannon entropy of the belief.

For a discrete random variable x, the entropy is defined as H(x) = ??? i p(x i ) log p(x i ).

An important property of entropy is its strict concavity on the simplex of belief points, denoted by ??? B (Cover & BID4 .

Further, the entropy is zero at the vertices of ??? B and achieves its maximum, log |S|, at the center of ??? B that corresponds to uniform distribution, i.e., when the uncertainty about the state is the highest.

FIG0 demonstrates the entropy and its level sets for |S| = 3.

Since the observation values are unknown before selecting the sensors, we optimize conditional entropy that yields the expected value of entropy over all possible observations.

For discrete random variables x and y, conditional entropy is defined as DISPLAYFORM3 .

Subsequently, with some algebraic manipulation, it can be shown that the conditional entropy of state given current belief with respect to ?? is: DISPLAYFORM4 where ??(??) = {i 1 , i 2 , . . .

, i k }.

It is worth mentioning that b is the current distribution of s and is explicitly written only for the purpose of better clarity, otherwise, H(s|b, ??) = H(s|??).To minimize entropy, we define the objective function as the following set function: DISPLAYFORM5 and the optimization problem as: DISPLAYFORM6 We propose a greedy algorithm, outlined in Algorithm 1 to find a near-optimal, yet efficient solution to (9).

The algorithm takes as input the agent's belief and planning action.

Then it iteratively adds elements from the ground set (set of all sensors) whose marginal gain with respect to f is maximal and terminates when k observations are selected.

Algorithm 1 Greedy policy for perception action DISPLAYFORM7 ?? ??? ?? ??? {j * } 7: end for 8: return ?? corresponding to ??.

Next, we derive a theoretical guarantee for the performance of the proposed greedy algorithm.

The following lemma states the required properties to prove the theorem.

The proof of the lemma follows from monotonicity and submodularity of conditional entropy BID9 .

See the appendix for the complete proof.

Lemma 1.

Let ??? = {?? 1 , ?? 2 , . . .

, ?? n } to represent a set of observations of the state s that conditioned on the state, are mutually independent (Assumption 1 holds).

Then, f (??), defined in (8), realizes the following properties: DISPLAYFORM8 2.

f is monotone nondecreasing, and 3.

f is submodular.

The above lemma enables us to establish the approximation factor using the classical analysis in BID17 .

Theorem 1.

Let ?? * to denote the optimal subset of observations with regard to objective function f (??), and ?? g to denote the output of the greedy algorithm in Algorithm 1.

Then, the following performance guarantee holds: DISPLAYFORM9 Remark 1.

Intuitively, one can interpret the minimization of conditional entropy as pushing the agent's belief toward the boundary of the probability simplex ??? B .

Due to convexity of POMDP value function on ??? B (Sondik, 1978) , this in turn implies that the agent is moving toward regions of belief space that have higher value.

Although Theorem 1 proves that the entropy of the belief point achieved by the greedy algorithm is close to the entropy of the belief point from the optimal solution, the key question is whether the value of these points are close.

We assess this question in the following and show that at each time step, on expectation, the value from greedy scheme is close to the value from optimal observation selection with regard to (9).

To that end, we first show that the distance between the two belief points is upper-bounded.

Thereafter, using a similar analysis as that of BID19 , we conclude that the difference between value function at these two points is upper-bounded.

Theorem 2.

Let the agent's current belief to be b and its planning action to be ??.

Consider the optimization problem in (9), and let ?? * and ?? g to denote the optimal perception action and the perception action obtained by the greedy algorithm, respectively.

It holds that: Proof.

We outline the sketch of the proof and bring the complete proof in the appendix.

First, we show that minimizing conditional entropy of posterior belief is equivalent to maximizing KullbackLeibler (KL-) divergence between current belief and the posterior belief, i.e., D KL (b ??,?? b b).

Next, we exploit Pythagorean theorem for KL-divergence alongside its convexity to find a relation between DISPLAYFORM10 DISPLAYFORM11 ) .

Afterwards, using Pinkster's inequality, we prove that the total variation distance between b DISPLAYFORM12 Proof.

The proof is omitted for brevity.

See the appendix for the proof.

In this section, we propose a novel point-based value iteration algorithm to approximate the value function for AP 2 -POMDPs.

The algorithm relies on the performance guarantee of the proposed greedy observation selection in previous section.

Before describing the new point-based solver, we first overview how point-based solvers operate.

Algorithm 2 outlines the general procedure for a point-based solver.

It starts with an initial set of belief points B 0 and their corresponding ?? vectors.

Then it performs a Bellman backup for each point to update ?? vectors.

Next, it prunes ?? vectors to remove dominated ones.

Afterwards, it samples a new set of belief points and repeats these steps until convergence or other termination criteria is met.

The difference between solvers is in how they apply sampling and pruning.

The sampling step usually depends on the reachability tree of belief space, see FIG2 .

The state-of-the-art point-based methods do not traverse the whole reachability tree, but they try to have enough sample points to provide a good coverage of the reachable space.

Note that the combinatorial number of actions due to observation selection highly expand the size of the reachability tree.

To avoid dealing with perception actions in the reachability tree, we apply the greedy scheme to make the choice of ?? deterministically dependent on ?? and previous belief.

To that end, we modify the BackUp step of point-based value iteration.

The proposed BackUp step can be combined with any sampling and pruning method in other solvers, such as the ones developed by Spaan & Vlassis FORMULA6

In point-based solver each witness belief point is associated with an ?? vector and an action.

Nevertheless, for AP 2 -POMDPs, each witness point is associated with two actions, ?? and ??.

We compute ?? based on greedy maximization of (9) so that given b and ??, ?? is uniquely determined.

Henceforth, we can rewrite (3) using (4) to obtain: DISPLAYFORM0 where?? = argmax ?????A pr f (??(??)) and f is computed atb ?? b .

This way, we can partially decouple the computation of perception action from the computation necessary for learning the planning policy.

Inspired by the results in the previous section, we propose the BackUp step detailed in Algorithm 3 to compute the new set of ?? vectors from the previous ones using Bellman backup operation.

What distinguishes this algorithm from conventional Bellman backup step is the inclusion of perception actions.

Basically, we need to compute the greedy perception action for each belief point and each action (Line 7).

This in turn affects computation of ?? b,??,?? t as it represents a different set for each belief point (Lines 9-13).

However, notice that this added complexity is significantly lower than concatenating the combinatorial perception actions with the planning actions and using conventional point-based solvers.

See the appendix for detailed complexity analysis.

To evaluate the proposed algorithm for active perception and planning, we developed a point-based value iteration solver for AP 2 -POMDPs.

We initialized the belief set by uniform sampling from ??? B BID6 .

To focus on the effect of perception, we did not apply a sampling step, i.e, the belief set is fixed throughout the iterations.

However, one can integrate any sampling method such as the ones proposed by Smith & Simmons (2012); BID13 .

The ?? vectors are initialized by 1 1????? min s,a R(s, a).Ones(|S|) (Shani et al., 2013) .

Furthermore, to speedup the solver, one can employ a randomized backup step, as suggested by Spaan & Vlassis (2005) .

The solver terminates once the difference between value functions in two consecutive iterations falls below a predefined threshold.

We also implemented a random perception policy that selects a subset of information sources, uniformly at random, at each backup step.

We implemented the solver in Python 2.7 and ran the simulations on a laptop with 2.0 GHz Intel Core i7-4510U CPU and with 8.00 GB RAM.

The first scenario models a robot that is moving in a 1-D discrete environment.

The robot can only move to adjacent cells and its navigation actions are A pl = {lef t, right, stop}. The robot's transitions are probabilistic due to possible actuation errors.

The robot does not have any sensor and it relies on a set of cameras for localization.

There is one camera at each cell that outputs a probability for b ??? B t do 7:?? = Greedy argmax ?????A pr f (??(??)) 8: DISPLAYFORM0 for ?? ??? ?? t???1 do 11: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Figure 3: The robot moves in a grid while communicating with the cameras to localize itself.

There is a camera at each state on the perimeter.

The accuracy of measurements made by each camera depends on the distance of the camera from that state.

The robot's objective is to reach the goal state, labeled by star, while avoiding the obstacles.distribution over the position of the robot.

The camera's certainty is higher when the robot's position is close to it.

To model the effect of robot's position on the accuracy of measurements, we use a binomial distribution with its mean at the cell that camera is on.

The binomial distribution represents the state-dependent accuracy.

The robot's objective is to reach an specific cell in the map.

For that purpose, at each time step, the robot picks a navigation action and selects k camera from the set of n cameras.

After the solver terminates, we evaluate the computed policy.

To that end, we run 1000 iterations of Monte Carlo simulations.

The initial state of the robot is the origin of the map and its initial belief is uniform over the map.

Figure 4 -(a) demonstrates the discounted cumulative reward, averaged over 1000 Monte Carlo runs, for random selection of 1 and 2 information sources, and greedy selection of 1 and 2 information sources.

It can be seen that the greedy perception policy significantly outperforms the random perception.

entropy of greedy perception, compared to random perception, shows less uncertainty of the robot when taking planning actions.

See the appendix for further results.

The second setting is a variant of first scenario where the map is 2-D. Therefore the navigation actions of robot are A pl = {up, right, down, lef t, stop}. The rest of the setting is similar to 1-D case, except the cameras' positions, as they are now placed around the perimeter of the map.

Additionally, the robot has to now avoid the obstacles in the map.

The reward is 10 at the goal state.

-4 at the obstacles, and -1 in other states.

We applied the proposed point-based solver with both random perception and greedy perception on the 2-D example.

Next, we let the robot to run for a horizon of 25 steps and we terminated the simulations once the robot reached the goal.

Figure 5 illustrates the normalized frequency of visiting each state for each perception algorithm.

It can be seen that the policy learned by greedy active perception leads to better obstacle avoidance.

See the appendix for further results.

In this paper, we studied joint active perception and planning in POMDP models.

To capture the structure of the problem, we introduced AP 2 -POMDPs that have to pick a cardinality-constrained subset of observations, in addition to original planning action.

To tackle the computational challenge of adding combinatorial actions, we proposed a greedy scheme for observation selection.

The greedy scheme aims to minimize the conditional entropy of belief which is a metric of uncertainty about the state.

We provided a theoretical analysis for the greedy algorithm that led to boundedness of value function difference between optimal entropy reduction and its greedy counterpart.

Furthermore, founded upon the theoretical guarantee of greedy active perception, we developed a point-based value iteration solver for AP 2 -POMDPs.

The idea introduced in the solver to address active perception is general and can be applied on state-of-the-art point-based solvers.

Lastly, we implemented and evaluated the proposed solver on a variety of robotic navigation scenarios.

In this section, we provide the proofs to the lemmas and theorems stated in the paper.

First, in the next lemma, we show that the objective function defined for uncertainty reduction has the required properties for the analysis by BID17 , namely being normalized, monotone, and submodular.

Lemma 1.

Let ??? = {?? 1 , ?? 2 , . . .

, ?? n } to represent a set of observations of the state s that conditioned on the state, are mutually independent (Assumption 1 holds).

Then, f (??), defined in (8), realizes the following properties: DISPLAYFORM0 2.

f is monotone nondecreasing, and 3.

f is submodular.

Proof.

Notice thatb ?? b is explicitly present to determine the current distribution of s and it is not a random variable.

Therefore, for simplicity, we omit that in the following proof.

It is clear that DISPLAYFORM1 To prove the monotonicity, consider ?? 1 ??? [n] and j ??? [n]\?? 1 .

Then, DISPLAYFORM2 where (a) and (c) are due to Bayes' rule for entropy, (b) follows from the conditional independence assumption and joint entropy definition, (d) is due to the conditional independence assumption, and (e) stems from the fact that conditioning does not increase entropy.

The monotonicity of the objective function means that if the number of obtained observations are higher, the conditional entropy will be lower, and hence, on expectation, the uncertainty in the state will be lower.

Furthermore, from the third line of above proof, we can derive the marginal gain, i.e., the value of adding one sensor, as: DISPLAYFORM3 To prove submodularity, let ?? 1 ??? ?? 2 ??? [n] and j ??? [n]\?? 2 .

Then, DISPLAYFORM4 where (a) is based on the fact that conditioning does not increase entropy, and (b) results from ?? 1 ??? ?? 2 .

The submodularity (diminishing returns property) of objective function indicates that as the number of obtained observations increases, the value of adding a new observation will decrease.

In the next theorem, we exploit the properties of the proposed objective function to analyze the performance of the greedy scheme.

Theorem 1.

Let ?? * to denote the optimal subset of observations with regard to objective function f (??), and ?? g to denote the output of the greedy algorithm in Algorithm 1.

Then, the following performance guarantee holds: DISPLAYFORM5 Proof.

The properties of f stated in Lemma 1 along the theoretical analysis of greedy algorithm by BID17 yields DISPLAYFORM6 Using the definition of f (??) and rearranging the terms, we obtain the desired result.

Before stating the proof to Theorem 2, that bounds the distance of belief points from the greedy and optimal entropy minimization algorithms, we need to present a series of propositions and lemmas.

Mutual information between two random variables is a positive and symmetric measure of their dependence and is defined as: DISPLAYFORM0 Mutual information, due to its monotonicity and submodularity, has inspired many subset selection algorithms BID10 .

In the following proposition, we express the relation between conditional entropy and mutual information.

Proposition 1.

Minimizing conditional entropy of the state with respect to a set of observations is equivalent to maximizing the mutual information of state and the set of observations.

This equivalency is due to the definition of mutual information, i.e., I(s; DISPLAYFORM1 and the fact that H(s) is computed atb ?? b which amounts to a constant value that does not affect selection procedure.

Additionally, notice that (12) is the same as the definition of normalized objective function of greedy algorithm in (8).Another closely-related information-theoretic concept is Kullback-Leibler (KL-) divergence.

The KL-divergence, also known as relative entropy, is a non-negative and non-symmetric measure of difference between two distributions.

The KL-divergence from q(x) to p(x) is: DISPLAYFORM2 The following relation between mutual information and KL-divergence exists: DISPLAYFORM3 which allows us to state the next proposition.

Proposition 2.

The mutual information of state and a set of observations is the expected value of the KL-divergence from prior belief to posterior belief over all realizations of observations, i.e., I(s; , ?? ??? i????? * ??? i to denote prior belief (after taking planning action), posterior belief after greedy perception action, and posterior belief after optimal perception action, respectively.

So far, we have established a relation between minimizing the conditional entropy of posterior belief and maximizing the expected KL-divergence from prior belief to posterior belief, i.e., D KL (p g p 0 ) (See Proposition 1 and Proposition 2).

To relate DISPLAYFORM4 DISPLAYFORM5 , we state the next lemma.

But first, we bring information-geometric definitions necessary for proving the lemma.

Definition 4.

Let p to be a probability distribution over a finite alphabet.

An I-sphere with center p and radius ?? is defined as: DISPLAYFORM6 is called the I-projection of p on ??. Lemma 2.

Instate the definition of p 0 , p g , and p * .

The following inequality holds on expectation: DISPLAYFORM7 Proof.

Consider the set ?? g = {p ??? ??? B |H(p) ??? H(p g ) that contain probability distributions whose entropy is lower-bounded by entropy of p g .

Since entropy is concave over ??? B , its hypographs are convex.

Consequently ?? g , the projection of a hypograph onto ??? B , is a convex set.

Furthermore, due to monotonicity of conditional entropy, i.e., expected value of entropy over observations, we know that p 0 ??? ?? g .

Besides, Due to optimality of ?? * , it holds that DISPLAYFORM8 which in turn yields p * ??? ??? B \?? g .

FIG10 demonstrates these facts for an alphabet of size 3.

p g is the I-projection of p * on ?? g .

Therefore, by exploiting the analogue of Pythagoras' theorem for KL-divergence (Csisz??r, 1975), we conclude: DISPLAYFORM9 A direct result of the above lemma, after taking the expectation over i???[n] ?? i , is: DISPLAYFORM10 In the following theorem, we use the stated lemma to bound the expected KL-divergence distance between greedy and optimal selection strategies.

Theorem 4.

The KL-divergence between p g and p * is upper-bounded, i.e., DISPLAYFORM11 where C 3 is a constant.

Proof.

Notice that while KL-divergence is not symmetric, the following fact still holds: where (a) follows from the fact that ?? * is the gradient of the optimal value function, (b) is due toH??lder's inequality, and (c) is the result of Theorem 2.

Taking C 2 = C 1 max{|Rmax|,|Rmin|} 1????? yields the desired result.

In this section, we compare the computational complexity of a point-based value iteration method that works with the concatenated action space, with the computational complexity of the proposed point-based method that picks the perception actions based on a greedy approach.

First, we compute the computations required for a single backup step in the point-based method with concatenated action space.

To that end, consider a fixed set of sampled belief points B. Let ?? to denote the current set of ?? vectors.

Further, for the simplicity of analysis, assume that the number of possible observations from each information source is |??? i | =??, ???i ??? [n].

The cardinality of a concatenated action space is |A| = |A pr ||A pl | = n k |A pl |.

Therefore, the complexity of a single backup step would be O( On the other hand, applying greedy algorithm to pick a perception action requires O(n ?? k) calls to an oracle that computes the objective function (or equivalently, the marginal gain).

Here the objective function is the conditional entropy whose complexity with a naive approach in the k th iteration is O(?? k ?? |S| 2 ).

Therefore, applying Algorithm 3 as the backup step leads to O(|A pl | ?? |B| ?? n ?? k ???? k ?? |S| 2 + |A pl | ?? |B| ???? k ?? |??| ?? |S| 2 + |B| ?? |A pl | ?? |S| ???? k ) operations.

Hence, the proposed approach, as a result of exploiting the structure of action space, would lead to significant computational gain, especially for large n. Figure 7 depicts the history of the belief entropy for the 2-D navigation when applying the proposed point-based method with random selection step and the proposed greedy selection step.

As expected, the greedy selection leads to smaller entropy and hence, less uncertainty about the state.

The corresponding average discounted cumulative reward after running 1000 Monte Carlo simulations is -18.8 for point-based value iteration with random selection step and -14.5 for point-based value iteration with greedy selection step, which demonstrates the superiority of the proposed method.

We further analyzed the effect of number of selected cameras on the agent's performance in the 1-D navigation scenario.

Figure 8 illustrates the value function for a subset of sampled belief points after the algorithm has been terminated.

It can be seen that the diminishing returns property of entropy with respect to number of selected observations is propagated through the value function as well.

<|TLDR|>

@highlight

We develop a point-based value iteration solver for POMDPs with active perception and planning tasks.