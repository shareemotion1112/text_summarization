We consider the problem of representing collective behavior of large populations and predicting the evolution of a population distribution over a discrete state space.

A discrete time mean field game (MFG) is motivated as an interpretable model founded on game theory for understanding the aggregate effect of individual actions and predicting the temporal evolution of population distributions.

We achieve a synthesis of MFG and Markov decision processes (MDP) by showing that a special MFG is reducible to an MDP.

This enables us to broaden the scope of mean field game theory and infer MFG models of large real-world systems via deep inverse reinforcement learning.

Our method learns both the reward function and forward dynamics of an MFG from real data, and we report the first empirical test of a mean field game model of a real-world social media population.

Nothing takes place in the world whose meaning is not that of some maximum or minimum.(Leonhard Euler)Major global events shaped by large populations in social media, such as the Arab Spring, the Black Lives Matter movement, and the fake news controversy during the 2016 U.S. presidential election, provide significant impetus for devising new models that account for macroscopic population behavior resulting from the aggregate decisions and actions taken by all individuals BID14 BID1 BID30 .

Just as physical systems behave according to the principle of least action, to which Euler's statement alludes, population behavior consists of individual actions that may be optimal with respect to some objective.

The increasing usage of social media in modern societies lends plausibility to this hypothesis BID27 , since the availability of information enables individuals to plan and act based on their observations of the global population state.

For example, a population's behavior directly affects the ranking of a set of trending topics on social media, represented by the global population distribution over topics, while each user's observation of this global state influences their choice of the next topic in which to participate, thereby contributing to future population behavior (Twitter, 2017) .

In general, this feedback may be present in any system where the distribution of a large population over a state space is observable (or partially observable) by each individual, whose behavior policy generates actions given such observations.

This motivates multiple criteria for a model of population behavior that is learnable from real data:can be specialized to many settings: optimal production rate of exhaustible resources such as oil among many producers BID13 ; optimizing between conformity to popular opinion and consistency with one's initial position in opinion networks BID2 ; and the transition between competing technologies with economy of scale BID19 .

Representing agents as a distribution means that MFG is scalable to arbitrary population sizes, enabling it to simulate real-world phenomenon such as the Mexican wave in stadiums BID13 .As the model detailed in Section 3 will show, MFG naturally addresses the modeling criteria in our problem context while overcoming limitations of alternative predictive methods.

For example, time series analysis builds predictive models from data, but these models are incapable of representing any motivation (i.e. reward) that may produce a population's behavior policy.

Alternatively, methods that employ the underlying population network structure have assumed that nodes are only influenced by a local neighborhood, do not account for a global state, and may face difficulty in explaining events as the result of any implicit optimization.

BID8 BID7 .

MFG is unique as a descriptive model whose solution tells us how a system naturally behaves according to its underlying optimal control policy.

This observation enables us to draw a connection with the framework of Markov decision processes (MDP) and reinforcement learning (RL) BID31 .

The crucial difference from a traditional MDP viewpoint is that we frame the problem as MFG model inference via MDP policy optimization: we use the MFG model to describe natural system behavior by solving an associated MDP, without imposing any control on the system.

MFG offers a computationally tractable framework for adapting inverse reinforcement learning (IRL) methods BID25 BID35 BID9 , with flexible neural networks as function approximators, to learn complex reward functions that may explain behavior of arbitrarily large populations.

In the other direction, RL enables us to devise a data-driven method for solving an MFG model of a real-world system for temporal prediction.

While research on the theory of MFG has progressed rapidly in recent years, with some examples of numerical simulation of synthetic toy problems, there is a conspicuous absence of scalable methods for empirical validation BID19 BID0 BID2 .

Therefore, while we show how MFG is well-suited for the specific problem of modeling population behavior, we also demonstrate a general data-driven approach to MFG inference via a synthesis of MFG and MDP.Our main contributions are the following.

We propose a data-driven approach to learn an MFG model along with its reward function, showing that research in MFG need not be confined to toy problems with artificial reward functions.

Specifically, we derive a discrete time graph-state MFG from general MFG and provide detailed interpretation in a real-world setting (Section 3).

Then we prove that a special case can be reduced to an MDP and show that finding an optimal policy and reward function in the MDP is equivalent to inference of the MFG model (Section 4).

Using our approach, we empirically validate an MFG model of a population's activity distribution on social media, achieving significantly better predictive performance compared to baselines (Section 5).

Our synthesis of MFG with MDP has potential to open new research directions for both fields.

Mean field games originated in the work of BID20 , and independently as stochastic dynamic games in BID16 , both of which proposed mean field problems in the form of differential equations for modeling problems in economics and analyzed the existence and uniqueness of solutions.

BID13 provided a survey of MFG models and discussed various applications in continuous time and space, such as a model of population distribution that informed the choice of application in our work.

Even though the MFG framework is agnostic towards the choice of cost function (i.e. negative reward), prior work make strong assumptions on the cost in order to attain analytic solutions.

We take a view that the dynamics of any game is heavily impacted by the reward function, and hence we propose methods to learn the MFG reward function from data.

Discretization of MFGs in time and space have been proposed BID10 BID0 BID12 , serving as the starting point for our model of population distribution over discrete topics; while these early work analyze solution properties and lack empirical verification, we focus on algorithms for attaining solutions in real-world settings.

Related to our application case, prior work by BID2 analyzed the evolution of opinion dynamics in multi-population environments, but they imposed a Gaussian density assumption on the initial population distribution and restrictions on agent actions, both of which limit the generality of the model and are not assumed in our work.

There is a collection of work on numerical finite-difference methods for solving continuous mean field games BID0 BID19 BID6 .

These methods involve forward-backward or Newton iterations that are sensitive to initialization and have inherent computational challenges for large real-valued state and action spaces, which limit these methods to toy problems and cannot be scaled to real-world problems.

We overcome these limitations by showing how the MFG framework enables adaptation of RL algorithms that have been successful for problems involving unknown reward functions in large real-world domains.

In reinforcement learning, there are numerous value-and policy-based algorithms employing deep neural networks as function approximators for solving MDPs with large state and action spaces BID24 BID29 BID21 .

Even though there are generalizations to multi-agent settings BID15 BID22 BID23 , the MDP and Markov game frameworks do not easily suggest how to represent systems involving thousands of interacting agents whose actions induce an optimal trajectory through time.

In our work, mean field game theory is the key to framing the modeling problem such that RL can be applied.

Methods in unknown MDP estimation and inverse reinforcement learning aim to learn an optimal policy while estimating an unknown quantity of the MDP, such as the transition law (Burnetas & Katehakis, 1997), secondary parameters BID4 , and the reward function BID25 .

The maximum entropy IRL framework has proved successful at learning reward functions from expert demonstrations BID35 BID3 BID18 .

This probabilistic framework can be augmented with deep neural networks for learning complex reward functions from demonstration samples BID34 BID9 .

Our MFG model enables us to extend the sample-based IRL algorithm in BID9 to the problem of learning a reward function under which a large population's behavior is optimal, and we employ a neural network to process MFG states and actions efficiently.

We begin with an overview of a continuous-time mean field games over graphs, and derive a general discrete-time graph-state MFG BID12 .

Then we give a detailed presentation of a discretetime MFG over a complete graph, which will be the focus for the rest of this paper.

Let G = (V, E) be a directed graph, where the vertex set V = {1, . . .

, d} represents d possible states of each agent, and E ⊆ V ×V is the edge set consisting of all possible direct transition between states (i.e., a agent can hop from i to j only if (i, j) ∈ E).

For each node i ∈ V, define V DISPLAYFORM0 Let π i (t) be the density (proportion) of agent population in state i at time t, and π(t) := (π 1 (t), . . .

, π d (t)).

Population dynamics are generated by right stochastic matrices P (t) ∈ S(G), where DISPLAYFORM1 Moreover, we have a value function V i (t) of state i at time t, and a reward function r i (π(t), P i (t)) 1 , quantifying the instantaneous reward for agents in state i taking transitions with probability P i (t) when the current distribution is π(t).

We are mainly interested in a discrete time graph state MFG, which is derived from a continuous time MFG by the following proposition.

Appendix A provides a derivation from the continuous time MFG.

Proposition 1.

Under a semi-implicit discretization scheme with unit time step labeled by n, the backward Hamilton-Jacobi-Bellman (HJB) equation and the forward Fokker-Planck equation for each i ∈ {1, . . .

, d} and n = 0, . . . , N − 1 in a discrete time graph state MFG are given by: DISPLAYFORM2 1 We here consider a rather special formulation where the reward function ri only depends on the overall population distribution π(t) and the choice Pi the players in state i made.3.2 DISCRETE TIME MFG OVER COMPLETE GRAPH Proposition 1 shows that a discrete time MFG given in BID10 can be seen as a special case of a discrete time graph state MFG with a complete graph (such that DISPLAYFORM3 We focus on the complete graph in this paper, as the methodology can be readily applied to general directed graphs.

While Section 4 will show a connection between MFG and MDP, we note here that a "state" in the MFG sense is a node in V and not an MDP state.

2 We now interpret the model using the example of evolution of user activity distribution over topics on social media, to provide intuition and set the context for our real-world experiments in Section 5.

Independent of any particular interpretation, the MFG approach is generally applicable to any problem where population size vastly outnumbers a set of discrete states.• Population distribution π n ∈ ∆ d−1 for n = 0, . . .

, N .

Each π n is a discrete probability distribution over d topics, where π n i is the fraction of people who posted on topic i at time n. Although a person may participate in more than one topic within a time interval, normalization can be enforced by a small time discretization or by using a notion of "effective population size", defined as population size multiplied by the max participation count of any person during any time interval.

π 0 is a given initial distribution.• Transition matrix P n ∈ S(G).

P n ij is the probability of people in topic i switching to topic j at time n, so we refer to P n i as the action of people in topic i. P n generates the forward equation DISPLAYFORM4 .

This is the reward received by people in topic i who choose action P n i at time n, when the distribution is π n .

In contrast to previous work, we learn the reward function from data (Section 4.1).

We make a locality assumption: reward for i depends only on P n i , not on the entire P n , which means that actions by people in j = i have no instantaneous effect on the reward for people in topic i. DISPLAYFORM5 is the expected maximum total reward of being in topic i at time n. A terminal value V N is given, which we set to zero to avoid making any assumption on the problem structure beyond what is contained in the learned reward function.• Average reward e i (π, P, V ), for i ∈ {1, . . .

, d} and V ∈ R d and P ∈ S(G).

This is the average reward received by agents at topic i when the current distribution is π, action P is chosen, and the subsequent expected maximum total reward is V .

For a general r ij (π, P ), it is defined as: DISPLAYFORM6 Intuitively, agents want to act optimally in order to maximize their expected total average reward.

For P ∈ S(G) and a vector q ∈ S i (G), define P(P, i, q) to be the matrix equal to P , except with the i-th row replaced by q.

Then a Nash maximizer is defined as follows: Definition 1.

A right stochastic matrix P ∈ S(G) is a Nash maximizer of e(π, P, V ) if, given a fixed DISPLAYFORM7 for any i ∈ {1, . . .

, d} and any q ∈ S i (G).The rows of P form a Nash equilibrium set of actions, since for any topic i, the people in topic i cannot increase their reward by unilaterally switching their action from P i to any q. Under Definition 1, the value function of each topic i at each time n satisfies the optimality criteria: DISPLAYFORM8 A solution of the MFG is a sequence of pairs {(π n , V n )} n=0,...,N satisfying optimality criteria (6) and forward equation (3).

A Markov decision process is a well-known framework for optimization problems.

We focus on the discrete time MFG in Section 3.2 and prove a reduction to a finite-horizon deterministic MDP, whose state trajectory under an optimal policy coincides with the forward evolution of the MFG.

This leads to the essential insight that solving the optimization problem of an MDP is equivalent to solving an MFG that describes population behavior.

This connection will enable us to apply efficient inverse RL methods, using measured population trajectories, to learn an MFG model along with its reward function in Section 4.1.

The MDP is constructed as follows: Definition 2.

A finite-horizon deterministic MDP for a discrete time MFG over a complete graph is defined as:• States: π n ∈ ∆ d−1 , the population distribution at time n.• Actions: P n ∈ S(G), the transition probability matrix at time n.• Reward: DISPLAYFORM0 • Finite-horizon state transition, given by Eq (3): ∀n ∈ {0, . . .

, N − 1} : DISPLAYFORM1 The value function of a solution to the discrete time MFG over a complete graph defined by optimality criteria (6) and forward equation (3) is a solution to the Bellman optimality equation of the MDP in Definition 2.Proof.

Since r ij depends on P n only through row P n i , optimality criteria 6 can be written as DISPLAYFORM2 We now define V * (π n ) as follows and show that it is the value function of the constructed MDP in Definition 2 by verifying that it satisfies the Bellman optimality equation: DISPLAYFORM3 which is the Bellman optimality equation for the MDP in Definition 2.Corollary 1.

Given a start state π 0 , the state trajectory under the optimal policy of the MDP in Definition 2 is equivalent to the forward evolution part of the solution to the MFG.Proof.

Under the optimal policy, equations 11 and 8 are satisfied, which means the matrix P generated by the optimal policy at any state π n is the Nash maximizer matrix.

Therefore, the state trajectory {π n } n=0,...,N is the forward part of the MFG solution.

MFG provides a general framework for addressing the problem of modeling population dynamics, while the new connection between MFG and MDP enables us to apply inverse RL algorithms to solve the MDP in Definition 2 with unknown reward.

In contrast to previous MFG research, most of which impose reward functions that are quadratic in actions and logarithmic in the state distribution BID11 BID19 BID2 , we learn a reward function using demonstration trajectories measured from actual population behavior, to ground the MFG representation of population dynamics on real data.

We leverage the MFG forward dynamics (Eq 3) in a sample-based IRL method based on the maximum entropy IRL framework BID35 .

From this probabilistic viewpoint, we minimize the relative entropy between a probability distribution p(τ ) over a space of trajectories T := {τ i } i and a distribution q(τ ) from which demonstrated expert trajectories are generated BID3 .

This is related to a path integral IRL formulation, where the likelihood of measured optimal trajectories is evaluated only using trajectories generated from their local neighborhood, rather than uniformly over the whole trajectory space BID18 .

Specifically, making no assumption on the true distribution of optimal demonstration other than matching of reward expectation, we posit that demonstration trajectories τ i = (π 0 , P 1 , . . .

, π N −1 , P N −1 ) i are sampled from the maximum entropy distribution (Jaynes, 1957): DISPLAYFORM0 where DISPLAYFORM1 is the sum of reward of single state-action pairs over a trajectory τ , and W are the parameters of the reward function approximator (derivation in Appendix E).

Intuitively, this means that trajectories with higher reward are exponentially more likely to be sampled.

Given M sample trajectories τ j ∈ D samp from k distributions F 1 (τ ), . . .

, F k (τ ), an unbiased estimator of the partition function Z = exp(R W (τ ))dτ using multiple importance sam- BID26 , where importance weights are DISPLAYFORM2 DISPLAYFORM3 −1 (derivation in Appendix F).

Each action matrix P is sampled from a stochastic policy F k (P ; π, θ) (overloading notation with F (τ )), where π is the current state and θ the policy parameter.

The negative log likelihood of L demonstration trajectories τ i ∈ D demo is: DISPLAYFORM4 We build on Guided Cost Learning (GCL) in BID9 (Alg 1) to learn a deep neural network approximation of R W (π, P ) via stochastic gradient descent on L(W ), and learn a policy F (P ; π, θ) using a simple actor-critic algorithm BID31 .

In contrast to GCL, we employ a combination of convolutional neural nets and fully-connected layers to process both the action matrix P and state vector π efficiently in a single architecture (Appendix C), analogous to how BID21 handle image states in Atari games.

Due to our choice of policy parameterization (described below), we also set importance weights to unity for numerical stability.

These implementation choices result in successful learning of a reward representation FIG0 .Our forward MDP solver (Alg 2) performs gradient ascent on the policy's expected start value E[v(π 0 )|F (P ; π, θ)] w.r.t.

θ, to find successively better policies F k (P ; π, θ).

We construct the joint distribution F (P ; π, θ) informed by domain knowledge about human population behavior on social media, but this does not reduce the generality of the MFG framework since it is straightforward to employ flexible policy and value networks in a DDPG algorithm when intuition is not available BID29 BID21 .

Our joint distribution is d instances of a d-dimensional Dirichlet distribution, each parameterized by an DISPLAYFORM5 where B(·) is the Beta function and α i j is defined using the softplus function α i j (π, θ) := ln(1 + exp{θ(π j − π i )}), which is a monotonically increasing function of the population density difference π j − π i .

In practice, a constant scaling factor c ∈ R can be applied to α for variance reduction.

Finally, we let θ) ) denote the parameterized policy, from which P n is sampled based on π n , and whose logarithmic gradient ∇ θ ln(F ) can be used in a policy gradient algorithm.

We learned an approximate value functionV (π; w) as a baseline for variance reduction, approximated as a linear combination of all polynomial features of π up to second order, with parameter w BID32 .

DISPLAYFORM6

We demonstrate the effectiveness of our method with two sets of experiments: (i) inference of an interpretable reward function and (ii) prediction of population trajectory over time.

Our experiment matches the discrete time mean field game given in Section 3.2: we use data representing the activity of a Twitter population consisting of 406 users.

We model the evolution of the population distribution over d = 15 topics and N = 16 time steps (9am to midnight) each day for 27 days.

The sequence of state-action pairs {(π n , P n )} n=0,...,N −1 measured on each day shall be called a demonstration trajectory.

Although the set of topics differ semantically each day, indexing topics in order of decreasing initial popularity suffices for identifying the topic sets across all days.

As explained earlier, the MFG framework can model populations of arbitrarily large size, and we find that our chosen size is sufficient for extracting an informative reward and policy from the data.

For evaluating performance on trajectory prediction, we compare MFG with two baselines: VAR.

Vector autoregression of order 18 trained on 21 demonstration trajectories.

RNN.

Recurrent neural network with a single fully-connected layer and rectifier nonlinearity.

We use Jenson-Shanon Divergence (JSD) as metric to report all our results.

Appendix D provides comprehensive implementation details.

We evaluated the reward using four sets of state-action pairs acquired from: 1.

all train demo trajectories; 2. trajectories generated by the learned policy given initial states π 0 of train trajectories; 3.

all test demo trajectories; 4. trajectories generated by the learned policy given initial states π 0 of test trajectories.

We find three distinct modes in the density of reward values for both the train group of sets 1 and 2 FIG0 and the test group of sets 3 and 4 FIG0 .

Although we do not have access to a ground truth reward function, the low JSD values of 0.13 and 0.017 between reward distributions for demo and generated state-action pairs show generalizability of the learned reward function.

We further investigated the reward landscape with nine state-action pairs FIG0 , and find that the mode with highest rewards is attained by pairing states that have large mass in topics having high initial popularity (S0) with action matrices that favor transition to topics with higher density (A0).

Uniformly distributed state vectors (S2) attain the lowest rewards, and states with a small negative mass gradient from topic 1 to topic d (S1) attain medium rewards.

Simply put, MFG agents who optimize for this reward are more likely to move towards more popular topics.

While this numerical exploration of the reward reveals interpretable patterns, the connection between such rewards learned via our method and any optimization process in the population requires more empirical study.

To test the usefulness of the reward and MFG model for prediction, the learned policy was used with the forward equation to generate complete trajectories, given initial distributions.

FIG1 shows that MFG has 58% smaller error than VAR when evaluated on the JSD between generated averaged element-wise over demo train set, and absolute difference between average demo action matrix and average matrix generated from learned policy. .

Both measures were averaged over M = 6 held-out test trajectories.

It is worth emphasizing that learning the MFG model required only the initial population distribution of each day in the training set (line 4 in Alg 2), while VAR and RNN used the distributions over all hours of each day.

MFG achieves better prediction performance even with fewer training samples, possibly because it is a more structured approximation of the true mechanism underlying population dynamics, in contrast to VAR and RNN that rely on regression.

As shown by sample trajectories for topic 0 and 2 in Figures 3, and the average transition matrices in FIG1 , MFG correctly represents the fact that the real population tends to congregate to topics with higher initial popularity (lower topic indices), and that the popularity of topic 0 becomes more dominant across time in each day.

The small real-world dataset size, and the fact that RNN mainly learns state transitions without accounting for actions, could be contributing factors to the lower performance of RNN.

We acknowledge that our design of policy parameterization, although informed by domain knowledge, introduced bias and resulted in noticeable differences between demonstration and generated transition matrices.

This can be addressed using deep policy and value networks, since the MFG framework is agnostic towards choice of policy representation.

We have motivated and demonstrated a data-driven method to solve a mean field game model of population evolution, by proving a connection to Markov decision processes and building on methods in reinforcement learning.

Our method is scalable to arbitrarily large populations, because the MFG framework represents population density rather than individual agents, while the representations are linear in the number of MFG states and quadratic in the transition matrix.

Our experiments on real data show that MFG is a powerful framework for learning a reward and policy that can predict trajectories of a real world population more accurately than alternatives.

Even with a simple policy parameterization designed via some domain knowledge, our method attained superior performance on test data.

It motivates exploration of flexible neural networks for more complex applications.

An interesting extension is to develop an efficient method for solving the discrete time MFG in a more general setting, where the reward at each state i is coupled to the full population transition matrix.

Our work also opens the path to a variety of real-world applications, such as a synthesis of MFG with models of social networks at the level of individual connections to construct a more complete model of social dynamics, and mean field models of interdependent systems that may display complex interactions via coupling through global states and reward functions.

Given the definitions in Section 3.1, a mean field game is defined by a Hamilton-Jacobi-Bellman (HJB) equation evolving backwards in time and a Fokker-Planck equation evolving forward in time.

The continuous-time Hamilton-Jacobi-Bellman (HJB) equation on G is DISPLAYFORM0 where r i (π, P i ) is the reward function, and V i (t) is the value function of state i at time t. Note that the reward function r i (π(t), P i (t)) is often presented as −c i (π(t), P i (t)) for some cost function c i (π(t), P i (t)) in the MFG context, and similarly for V i (t).

In addition, we set r i (π(t), DISPLAYFORM1 e. P (t) must be a valid transition matrix).

For any fixed π(t), let H i (π(t), ·) be the Legendre transform of c i (π(t), ·) defined by DISPLAYFORM2 Then the HJB equation FORMULA7 is an analogue to the backward equation in mean field games DISPLAYFORM3 where DISPLAYFORM4 | is the dual variable of P i .

We can discretize (15) using a semi-implicit scheme with unit time step labeled by n to obtain DISPLAYFORM5 Rearranging FORMULA25 yields the discrete time HJB equation over a graph (19) DISPLAYFORM6 The forward evolving Fokker-Planck equation for the continuous-time graph-state MFG is given by DISPLAYFORM7 where DISPLAYFORM8 where ∂ ui H j (π, u) is the partial derivative w.r.t.

the coordinate corresponding to the i-th index of the argument u ∈ R |V − j | .

We can set Q ji (t) = 0 for all (j, i) / ∈ E, so that Q(t) := [Q ji (t)] can be regarded as the d-by-d infinitesimal generator matrix of states π(t), and hence (20) can be written as π (t) = π(t)Q(t), where π(t) ∈ R d is a row vector.

Then an Euler discretization of FORMULA2 with unit time step reduces to π n+1 − π n = π n Q n , which can be written as DISPLAYFORM9 where P n ij := Q n ij + δ ij .

If the graph G is complete, meaning E = {(i, j) : 1 ≤ i, j ≤ d}, then the summation is taken over j = 1, . . . , d. For ease of presentation, we only consider the complete graph in this paper, as all derivations can be carried out similarly for general directed graphs.

A solution of a mean field game defined by (19) and FORMULA2

We learn a reward function and policy using an adaptation of GCL BID9 in Alg 1 and a simple actor-critic Alg 2 BID31 Generate sample trajectories D traj from F k (P ; π, θ) 5: DISPLAYFORM0 Sample demonstrationD demo ⊂ D demo from expert demonstration return Final reward function R W (π, P ) and policy F (P ; π, θ) 14: end procedure Sample action P n ∼ F (P ; π n , θ) DISPLAYFORM1 Generate π n+1 using Eq 3 8:Receive reward R W (π n , P n )

δ ← R +V (π n+1 ; w) −V (π n ; w)

w ← w + ξδ∇ wV (π n ; w)

θ ← θ + βδ∇ θ log(F (P ; π n , θ))

end for

end for 14: end procedure C REWARD NETWORK Our reward network uses two convolutional layers to process the 15 × 15 action matrix P , which is then flattened and concatenated with the state vector π and processed by two fully-connected layers regularized with L1 and L2 penalties and dropout (probability 0.6).

The first convolutional layer zero-pads the input into a 19 × 19 matrix and convolves one filter of kernel size 5 × 5 with stride 1 and applies a rectifier nonlinearity.

The second convolutional layer zero-pads its input into a 17 × 17 matrix and convolves 2 filters of kernel size 3 × 3 with stride 1 and applies a rectifier nonlinearity.

The fully connected layers have 8 and 4 hidden rectifier units respectively, and the output is a single fully connected tanh unit.

All layers were initialized using the Xavier normal initializer in Tensorflow.

By default, Twitter users in a certain geographical region primarily see the trending topics specific to that region (Twitter, 2017) .

This experiment focused on the population and trending topics in the city of Atlanta in the U.S. state of Georgia.

First, a set of 406 active users were collected to form the fixed population.

This was done by collecting a set of high-visibility accounts in Atlanta (e.g. the Atlanta Falcons team), gathering all Twitter users who follow these accounts, filtering for those whose location was set to Atlanta, and filtering for those who responded to least two trending topics within four days.

Data collection proceeded as follows for 27 days: at 9am of each day, a list of the top 14 trending topics on Twitter in Atlanta was recorded; for each hour until midnight, for each topic, the number of users who responded to the topic and the transition counts among topics within the past hour was recorded.

Whether or not a user responded to a topic was determined by checking for posts by the user containing unique words for that topic; the "hashtag" convention of trending topics on Twitter reduces the likelihood of false positives.

The hourly count of people who did not respond to any topic was recorded as the count for a "null topic".

Although some users may respond to more than one topic within each hour, the data shows that this is negligible, and a shorter time interval can be used to reduce this effect.

The result of data collection is a set of trajectories, one trajectory per day, where each trajectory consists of hourly measurements of the population distribution over d = 15 topics and their transition matrix over N = 16 hours.

The training set consists of trajectories {π 0,m , P 0,m , . . .

, P N −2,m , π N −1,m } m=1,...,M over the first M = 21 days.

MFG uses the initial distribution π 0 of each day, along with the transition equation of the constructed MDP and the policy F (P ; π n , θ), to produce complete trajectories for training (Alg 2 lines 4,6,7).

In contrast, VAR and RNN are supervised learning methods and they use all measured distributions.

RNN employs a simple recurrent unit with ReLU as nonlinear activation and weight matrix of dimension d × d. VAR was implemented using the Statsmodels module in Python, with order 18 selected via random sub-sampling validation with validation set size 5 BID28 .

For prediction accuracy, all three methods were evaluated against data from 6 held-out test days.

TAB1 shows parameters of Alg 2 and 1.

FIG0 , . . . , ).

Suppose each trajectory τ i has an unknown probability p i .

The entropy of the probability distribution is H = − i p i ln(p i ).

In the continuous case, we write the differential entropy: DISPLAYFORM0 where p(·) is the probability density we want to derive.

The constraints are: DISPLAYFORM1 The first constraint says: the expected reward over all trajectories is equal to an empirical measurement µ r .

We write the Lagrangian L: DISPLAYFORM2 For L to be stationary, the Euler-Lagrange equation with integrand denoted by L says ∂L ∂p = 0 since L does not depend on dp dτ .

Hence DISPLAYFORM3 where Z := e −λ2r(τ ) dτ .

Then the constant λ 2 is determined by: DISPLAYFORM4 We show how multiple importance sampling BID26 can be used to estimate the partition function in the maximum entropy IRL framework.

The problem is to estimate Z := f (x)dx.

Let p 1 , . . . , p m be m proposal distributions, with n j samples from the j-th proposal distribution, so that samples can be denoted X ij for i = 1, . . .

, n j and j = 1, . . .

, m. Let w j (x) for j = 1, . . .

, m satisfy DISPLAYFORM5 Then define the estimatorẐ DISPLAYFORM6 Let S(p j ) = {x | p j (x) > 0} be the support of p j and S(w j ) = {x | w j (x) > 0} be the support of w j , and let them satisfy S(w j ) ⊂ S(p j ).

Under these assumptions: DISPLAYFORM7 In particular, choose DISPLAYFORM8 where n = m j=1 n j is the total count of samples.

Further assuming that samples are drawn uniformly from all proposal distributions, so that n j = n k = n/m for all j, k ∈ {1, . . .

, m}, the expression forẐ reduces to the form used in Eq 13: DISPLAYFORM9

In this section, we discuss the reason that the general MFG, whose reward function r ij (π n , P n ) depends on the full Nash maximizer matrix P n , is neither reducible to a collection of distinct singleagent MDPs nor equivalent to a multi-agent MDP.

Let a state in the discete space MFG be called a "topic", to avoid confounding with an MDP state.

Consider each topic i as a separate entity associated with a value, rather than subsuming it into an average (as is the case in Section 4).

In order to assign a value to each topic, each tuple (i, π n ) must be defined as a state, which leads to the problem: since a state requires specification of π n , and state transitions depend on the actions for all other topics, the action at each topic is not sufficient for fully specifying the next state.

More formally, consider a value function on the state: DISPLAYFORM0 Superficially, this resembles the Bellman optimality equation for the value function in a single-agent stochastic MDP, where s is a state, a is an action, R is an immediate reward, and P (s |s, a) is the probability of transition to state s from state s, given action a: DISPLAYFORM1 In equation 23, q j can be interpreted as a transition probability, conditioned on the fact that the current topic is i. The action q selected in the state (i, π n ) induces a stochastic transition to a next topic j, but the next distribution π n+1 is given by the deterministic forward equation π n+1 = (P n ) T π n , where P n is the true Nash maximizer matrix.

This means that q j does not completely specify the next state (j, π n+1 ), and there is a formal difference between P (s |s, a)V * (s ) and q j V (j, (P n ) T π n ).

Also notice that the Bellman equation sums over all possible next states s , but equation 23 only sums over topics j rather than full states (j, π).

Short of modeling every single agent in the MFG, an exact reduction from the MFG to a multi-agent MDP (i.e. Markov game) is not possible.

A discrete state space discrete action space multi-agent MDP is defined by d agents moving within a set S of environment states; a collection {A 1 , . . .

, A d } of action spaces; a transition function P (s |s, a 1 , . . .

, a d ) giving the probability of the environment transitioning from current state s to next state s , given that agents choose actionsā := (a 1 , . . .

, a d ); a collection of reward functions {R i (s, a 1 , . . .

, a d )} i ; and a discount factor γ.

Let the set of π n (with appropriate discretization) be the state space and limit the set of actions to some discretization of the simplex.

The alternative to modeling individual MFG agents is to consider each topic as a single "agent".

Now, the agent representing topic i is no longer identified with the set of people who selected topic i: topics have fixed labels for all time, so an agent can only accumulate reward for a single topic, whereas people in the MFG can move among topics.

Therefore, the value function for agent i in a Markov game is defined only in terms of itself, never depending on the value function of agents j = i: where µ := (µ 1 , . . .

, µ d ) is a set of stationary policies of all agents.

However, recall that the MFG equation for V n i explicitly depends on V n+1 j of all topics j, which would require a different form such as the following: DISPLAYFORM0 where the last terms sums over value functions V µ k for all topics k. This mixing between value functions prevents a reduction from the MFG to a standard Markov game.

@highlight

Inference of a mean field game (MFG) model of large population behavior via a synthesis of MFG and Markov decision processes.

@highlight

The authors deal with inference in models of collective behavior by using inverse reinforcement learning to learn the reward functions of agents in the model.