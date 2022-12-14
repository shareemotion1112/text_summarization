The aim of this study is to introduce a formal framework for analysis and synthesis of driver assistance systems.

It applies formal methods to the verification of a stochastic human driver model built using the cognitive architecture ACT-R, and then bootstraps safety in semi-autonomous vehicles through the design of provably correct Advanced Driver Assistance Systems.

The main contributions include the integration of probabilistic ACT-R models in the formal analysis of semi-autonomous systems and an abstraction technique that enables a finite representation of a large dimensional, continuous system in the form of a Markov model.

The effectiveness of the method is illustrated in several case studies under various conditions.

When it comes to driving, the numbers do not lie; more than 90% of road accidents in the US are caused by human error BID18 .

In an effort to increase driver safety, some car manufacturers have introduced semi-autonomous features in the form of Advanced Driver Assistance Systems (ADAS).

Despite this, guaranteeing safety in semi-autonomous vehicles remains a challenge, with most of the existing methods being based on testing and simulation BID4 BID9 BID19 BID21 , which do not provide the guarantees required for a safety critical system BID10 .

Some recent works use formal verification to obtain strong guarantees about the ADAS BID5 BID13 BID15 ], yet they present engineering approaches to the problem which ignore the cognitive process of the human driver, leading to solutions that might perform poorly in corner cases.

This study focuses on designing an ADAS that takes into account a stochastic model of the driver cognitive process.

It employs the cognitive architecture known as Adaptive Control of Thought-Rational (ACT-R), a framework for specifying computational behavioral models of human cognitive performance which embodies both the abilities (e.g. memory storage or perception) and constraints (e.g. limited motor performance) of humans BID0 BID1 BID3 BID16 BID17 BID20 .

The work builds on the human driver model in a multi-lane highway driving scenario presented in BID17 .

It also expands upon BID5 BID12 by applying verification techniques to an efficient abstraction of the model and extends it to allow the intervention of a provably correct (up to the level of representation of the model) ADAS based on specifications given as temporal logic statements.

The problem is defined as follows.

Given the vehicle model from BID14 , a human driver model represented by ACT-R BID17 a set of initial conditions S, and a temporal logic formula ?? [3], we are interested in (1) verification: computing the probability that the Human-Vehicle model satisfies ?? in S, i.e, P S (??); and (2) synthesis: designing an ADAS that optimizes the probability of satisfying ?? by the Human-Vehicle-ADAS system in S, i.e., P S (??) with ??? {max, min}.

To verify the human driver model under ??, we first abstract it to a Markov Chain M h = (S, P, s 0 , AP, L), where S is a finite set of states, P : S ?? S ??? [0, 1] is a transition probability function, s 0 ??? S is the initial state, AP is a set of atomic propositions, and L : S ??? 2 AP is a labeling function.

We achieve this by discretizing the integrated human driver ACT-R model in BID17 through the use of a vehicle model BID14 .

We can then verify it using off-the-shelf tools, e.g. PRISM BID11 .

We assume a two vehicle scenario, where the ego-vehicle interacts with a lead vehicle whose motion is predictable BID8 .

The driver model presented in BID17 is divided into three sequential modules: (i) control, which manages low level perception cues and the manipulation of the vehicle, (ii) monitoring, which maintains awareness of the position of other vehicles around the ego-vehicle; and (iii) decision making, which determines the tactical decision to be taken.

Our abstraction combines decision making and monitoring into one module for the sake of efficiency.

We define M h which unifies both modules through the use of ?? ??? {1, 2}, where ?? = 1 corresponds to the control step and ?? = 2 to the decision making and monitoring stage.

A state s ??? S is a tuple s = (??, x, ??, a, v, t), where x is bounded to a finite length of the road according to the situation, v is the speed of the vehicle, a is the acceleration and ?? represents the index of the lane of the ego, abstracting away the y variable which reduces the size of the model.

A time discretization is induced for all the continuous variables.

For a given set of initial conditions S, the state space S is automatically generated.

The transition probabilities for all s, s ??? S are given by: DISPLAYFORM0 where CONTROL is a deterministic transition table resulting from the simulation of the control laws from BID17 and DMM is a table of transition probabilities based on the introduction of (1) Gaussian noise to the decision making and monitoring processes presented in BID17 as a way to model the uncertainty of perception; and (2) stochastic uncertainty in terms of the lane changing decision based on driver variability.

To obtain the Human-Vehicle-ADAS system, we augment M h with possible realistic interventions by the ADAS, as presented in FIG0 .

These interventions can be of two types: passive suggestions (PS) and active control (AC).

In passive suggestions, we assume that the assistance system cannot change the decision making directly, as it is a human cognitive process, but it can influence it to a certain degree through suggestions BID7 , i.e. an action at this level, a P S i , induces the probability distribution DMM i (s, s ), which is biased towards the desired outcome.

In active control, the actions available to the ADAS, a AC i , can have corrective control-based interventions at the level of acceleration and steering (with ADAS variables constrained to ensure incremental interventions), deterministically leading to different states according to CONTROL i (s).

The optimal ADAS design is reduced to finding an optimal policy over M ADAS for a certain specification given as a temporal logic formula ?? defined over AP .

Off-the-shelf tools, such as PRISM BID11 , can be employed for this computation.

III.

EXPERIMENTAL RESULTS The framework was implemented in Python using PRISM and the code is available on Github 1 .

To study its applications, we considered a simplified two lane scenario of length x max where the lead vehicle is assumed to be moving at a constant speed.

We also considered two interesting properties: DISPLAYFORM1 and DISPLAYFORM2 which we want to minimize and maximize, respectively.

Intuitively, ?? 1 refers to how unsafe the system is, while ?? 2 corresponds to the time efficiency of it.

FIG1 shows an example of a trajectory under ?? 1 for a given highly unsafe initial situation S, in which the ADAS effectively leads the system to a safer situation.

Fig. 3 showcases the difference in probabilities of satisfying (a) ?? 1 and (b) ?? 2 , assuming each specification to be optimized individually.

In both cases, all randomly generated scenarios tested lead to a decrease in the case of ?? 1 and an increase in the case of ?? 2 , i.e. improved satisfaction of the desired properties.

These results refer to optimizing each of the properties individually and do not offer any insight into how optimizing one influences the satisfaction of the other.

Through our framework, we are also able to study the relationships between properties using multi-objective optimization techniques.

FIG2 presents the Pareto frontier of optimizing ?? 1 and ?? 2 for a given S in a multi-objective setting, showing that, as expected, there is a trade-off between the two properties.

A more in-depth analysis can be found in BID6 , including situations with more vehicles and complex specifications.

The approach proposed in this paper enables the study of safety of semi-autonomous vehicles in various conditions and the design of ADAS that are robust with formal guarantees.

In the future, the specifications passed to the ADAS could be learnt so as to match the behavior of expert drivers.

<|TLDR|>

@highlight

Verification of a human driver model based on a cognitive architecture and synthesis of a correct-by-construction ADAS from it.