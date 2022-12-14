Goal recognition is the problem of inferring the correct goal towards which an agent executes a plan, given a set of goal hypotheses, a domain model, and a (possibly noisy) sample of the plan being executed.

This is a key problem in both cooperative and competitive agent interactions and recent approaches have produced fast and accurate goal recognition algorithms.

In this paper, we leverage advances in operator-counting heuristics computed using linear programs over constraints derived from classical planning problems to solve goal recognition problems.

Our approach uses additional operator-counting constraints derived from the observations to efficiently infer the correct goal, and serves as basis for a number of further methods with additional constraints.

Agents that act autonomously on behalf of a human user must choose goals independently of user input and generate plans to achieve such goals ).

When such agents have complex sets goals and require interaction with multiple agents that are not under the user's control, the resulting plans are likely to be equally complex and non-obvious for human users to interpret BID0 .

In such environments, the ability to accurately and quickly identify the goals and plans of all involved agents is key to provide meaningful explanation for the observed behavior.

Goal recognition is the problem of inferring one or more goals from a set of hypotheses that best account for a sequence of observations, given a fixed initial state, a goal state, and a behavior model of the agent under observation.

Recent approaches to goal recognition based on classical planning domains have leveraged data-structures and heuristic information used to improve planner efficiency to develop increasingly accurate and faster goal recognition algorithms BID1 BID2 .

Specifically, BID2 use heuristics based on planning landmarks BID1 ) to accurately and efficiently recognize goals in a wide range of domains with various degrees of observability and noise.

This approach, however, does not deal with noise explicitly, relying on the implicit necessity of landmarks in valid plans for goal hypotheses to achieve com- petitive accuracy with other methods BID3 BID3 , while increasing the number of recognized goals (spread).Thus, goal recognition under partial observability (i.e., missing observations) in the presence of noisy observation is a difficult problem to address while achieving both reasonable recognition time (i.e., a few seconds), high accuracy and low spread.

In this paper, we address these limitations by leveraging recent advances on operator-counting heuristics (Pommerening et al. 2014; BID4 ).

Operator-counting heuristics provide a unifying framework for a variety of sources of information from planning heuristics BID1 ) that provide both an estimate of the total cost of a goal from any given state and and indication of the actual operators likely to be in such plans.

This information proves to be effective at differentiating between goal hypotheses in goal recognition.

Our contributions are threefold.

First, we develop three, increasingly more accurate goal recognition approaches using operator-counting heuristics.

Second, we empirically show that these heuristics are very effective at goal recognition, overcoming existing approaches in almost all domains in terms of accuracy while diminishing the spread of recognized goals.

Such approaches are substantially more effective for noisy settings.

Third, we discuss a broad class of operator-counting heuristics for goal recognition that can use additional constraints to provide even finer handling of noise and missing observations.

We review the key background for the approaches we develop in this paper.

First, the recognition settings we assume for our approach follows the standard formalization of goal recognition as planningSecond, while there is substantial literature on linear programming heuristic unified on the operator-counting framework, we focus on the specific types of operator-counting constraints we actually use in our experimentation.

Definition 1 (Predicates and State).

A predicate is denoted by an n-ary predicate symbol p applied to a sequence of zero or more terms (?? 1 , ?? 2 , ..., ?? n ) -terms are either constants or variables.

We refer to grounded predicates that represent logical values according to some interpretation as facts, which are divided into two types: positive and negated facts, as well as constants for truth ( ) and falsehood (???).

A state S is a finite set of positive facts f that follows the closed world assumption so that if f ??? S, then f is true in S. We assume a simple inference relation |= such that S |= f iff f ??? S, S |= f iff f ??? S, and S |= f 1 ??? ...

??? f n iff {f 1 , ..., f n } ??? S.Definition 2 (Operator and Action).

An operator a is represented by a triple name(a), pre(a), eff(a) : name(a) represents the description or signature of a; pre(a) describes the preconditions of a, a set of predicates that must exist in the current state for a to be executed; eff(a) represents the effects of a. These effects are divided into eff(a) + (i.e., an add-list of positive predicates) and eff(a) ??? (i.e., a delete-list of negated predicates).

An action is a ground operator instantiated over its free variables.

Definition 3 (Planning Domain).

A planning domain definition ?? is represented by a pair ??, A , which specifies the knowledge of the domain, and consists of a finite set of facts ?? (e.g., environment properties) and a finite set of actions A.Definition 4 (Planning Instance).

A planning instance ?? is represented by a triple ??, I, G , where ?? = ??, A is the domain definition; I ??? ?? is the initial state specification, which is defined by specifying values for all facts in the initial state; and G ??? ?? is the goal state specification, which represents a desired state to be achieved.

Definition 5 (Plan).

An s-plan ?? for a planning instance ?? = ??, I, G is a sequence of actions a 1 , a 2 , ..., a n that modifies a state s into a state S |= G in which the goal state G holds by the successive execution of actions in ?? starting from s. An I-plan is just called a plan.

A plan ?? * with length |?? * | is optimal if there exists no other plan ?? for ?? such that |?? | < |?? * |.A goal recognition problem aims to select the correct goal of an agent among a set of possible goals using as evidence a sequence of observations.

These observations might be actions executed by the agent or noise observation which are part of a valid plan but are not executed by the agent.

Definition 6 (Observation Sequence).

An observation sequence O = o 1 , o 2 , ..., o n is said to be satisfied by a plan ?? = a 1 , a 2 , ..., a m , if there is a monotonic function f that maps the observation indices j = 1, ..., n into action indices i = 1, ..., n, such that a f (j) = o j .Definition 7 (Goal Recognition Problem).

A goal recognition problem is a tuple T GR = ??, I, G, O , where: ?? = ??, A is a planning domain definition; I is the initial state; G is the set of possible goals, which include a correct hidden goal G * (i.e., G * ??? G); and O = o 1 , o 2 , ..., o n is an observation sequence of executed actions, with each observation o i ??? A, and the corresponding action being part of a valid plan ?? (from Definition 5) that transitions I into G * through the sequential execution of actions in ??.

Definition 8 (Solution to a Goal Recognition Problem).

A solution to a goal recognition problem T GR = ??, I, G, O is a nonempty subset of the set of possible goals G ??? G such that ???G ??? G there exists a plan ?? G generated from a planning instance ??, I, G and ?? G is consistent with O.

Recent work on linear programming (LP) based heuristics has generated a number of informative and efficient heuristics for optimal-cost planning BID4 Pommerening et al. 2014; BID0 .

These heuristics rely on constraints from different sources of information that every plan ?? (Definition 5) must satisfy.

All operator-counting constraints contain variables of the form Y a for each operator a such that setting Y a to the number of occurrences of a in ?? satisfies the constraints.

In this paper we adopt the formalism and definitions of Pommerening et al. for LP-based heuristics 1 .

for all a ??? A. A constraint set for s is a set of operatorcounting constraints for s where the only common variables between constraints are the operator-counting constraints.

While the framework from Pommerening et al. 2013 unifies many types of constraints for operator-counting heuristics, we rely on three types of constraints for our goal recognition approaches: landmarks, state-equations, and post-hoc optimization.

Planning landmarks consist of actions (alternatively state-formulas) that must be executed (alternatively made true) in any valid plan for a particular goal BID1 .

Thus, landmarks are necessary conditions for all valid plans towards a goal, and, as such, provide the basis for a number of admissible heuristics (Karpas and Domshlak 2009) and as conditions to strengthen existing heuristics (Bonet 2013).

Importantly, planning landmarks form the basis for the current state-of-the-art goal recognition algorithms BID2 BID2 .

Disjunctive action landmarks BID5 ) for a state s are sets of actions such that at least one action in the set must be true for any s-plan, and make for a natural operator-counting constraint.

DISPLAYFORM0 Net change constraints generalize Bonet's (2013) state equation heuristic, which itself relate the planning instance in question with Petri nets that represent the transitions of state variables induced by the actions, and such that tokens in this task represent net changes to the states of the problem.

Finally, Post-hoc optimization constraints (Pommerening et al. 2013) use the fact that certain heuristics can rule out the necessity of certain operators from plans (and thus from the heuristic estimate).

For example, Pattern Database (PDBs) heuristics (Culberson and Schaeffer 1998) create projections of the planning task into a subset of state variables (with this subset being the pattern), such that the heuristic can partition operators into two sets of each pattern, one that changes variables in the pattern (i.e., contributes towards transitions) and the other than does not (i.e., is non-contributing).

Definition 13 (Post-hoc Optimization Constraint).

Let ?? be a planning task with operator set A, let h be an admissible heuristic for ??, and let N ??? A be a set of operators that are noncontributing in that h is still admissible in a modified planning task where the cost of all operators in N is set to 0.Then the post-hoc optimization constraint c P H s,h,N for h, N , and state s of ?? consists of the inequality.

DISPLAYFORM1

We now bring together the operator-counting constraints into three operator-counting heuristics suitable for goal recognition, ranging from the simplest way to employ operator counts to compute the overlap between counts and observed actions, to modifying the constraints used by the operator counts to enforce solutions that agree with such observations,and finally accounting for possible noise by comparing heuristic values.

We start with a basic operator-counting heuristic h(s), which we define to be the LP-heuristic of Def.

11 where C comAlgorithm 1 Goal Recognition using the Operator Counts.

Input: ?? planning domain definition, I initial state, G set of candidate goals, and O observations.

Output: Recognized goal(s).

DISPLAYFORM0 Hits ??? Initialize empty dictionary 3:for all G ??? G do Compute overlap for G 4: DISPLAYFORM1 for all o ??? O do 8:if Yo > 0 then 9:HitsG ??? HitsG + 1 10: DISPLAYFORM2 prises the constraints generated by Landmarks (Def.

12), post-hoc optimization (Def.

13), and net change constraints as described by Pommerening et al. (2014) .

This heuristic, computed following Def.

11, yields two important bits of information for our first technique, first, it generates the actual operator counts Y a for all a ??? A from Def.

10, whose minimization comprises the objective function h(s).The heuristic values h(s) of each goal candidate G ??? G tells us about the optimal distance between the initial state I and G, while the operator counts indicate possible operators in a valid plan from I to G. We can use these counts to account for the observations O by computing the overlap between operators with counts greater than one and operators observed for recognition.

Algorithm 1 shows how we can use the operator counts directly in a goal recognition technique.

In order to rank the goal hypotheses we keep a dictionary of Hits (Line 2) to store the overlap, or count the times operators counts hit observed actions.

The algorithm then iterates over all goal hypotheses (Lines 3-10) computing the operator counts for each hypothesis G and comparing these counts with the actual observations (Lines 7-10).

We recognize goals by choosing the hypotheses whose operator counts hit the most observations (Line 11).

The technique of Algorithm 1 is conceptually similar to the Goal Completion heuristic of BID2 in that it tries to compare heuristically computed information with the observations.

However, this initial approach has a number of shortcomings in relation to their technique.

First, while the landmarks themselves are enforced by the LP used to compute the operator counts (and thus observations that correspond to landmarks count as hits), the overlap computation loses the ordering of the landmarks that the Goal Completion heuristic uses to account for missing observations.

Second, a solution to a set of operator-constraints, i.e., a set of operators with non-negative counts may not be a feasible plan for a planning instance.

Thus, these counts may not correspond to the plan that generated the observations.

While operator-counting heuristics on their own are fast and informative enough to help guide search when dealing with millions of nodes, goal recognition problems often reAlgorithm 2 Goal Recognition using ObservationConstrained Operator Counts.

Input: ?? planning domain definition, I initial state, G set of candidate goals, and O observations.

Output: Recognized goal(s).

DISPLAYFORM0 quire the disambiguation of a dozen or less goal hypotheses.

Such goal hypotheses are often very similar so that the operator-counting heuristic value (i.e., the objective function over the operator counts) for each goal hypothesis is very similar, especially if the goals are more or less equidistant from the initial state.

Thus, we refine the technique of Observation Overlap by introducing additional constraints into the LP used to compute operator counts.

Specifically, we force the operator counting heuristic to only consider operator counts that include every single observation o ??? O. The resulting LP heuristic (which we call h C ) then minimizes the cost of the operator counts for plans that necessarily agree with all observations.

We summarize this Observation Constraint Enforcement approach in Algorithm 2.

This technique is similar to that of Algorithm 1 in that it iterates over all goals computing a heuristic value.

However, instead of computing observation hits by looking at individual counts, it generates the constraints for the operator-counting heuristic (Line 3) and adds constraints to ensure that the count of the operators corresponding to each observation is greater than one (Lines 4-5).

Finally, we choose the goal hypotheses that minimize the operator count heuristic distance from the initial state (Line 8).

Although enforcing constraints to ensure that the LP heuristic computes only plans that do contain all observations helps us overcome the limitations of computing the overlap of the operator counts, this approach has a major shortcoming: it considers all observations as valid operators generated by the observed agent.

Therefore, the heuristic resulting from the minimization of the LP might overestimate the actual length of the plan for the goal hypothesis due to noise.

This may happen for one of two reasons: either the noise is simply a sub-optimal operator in a valid plan, or it is an operator that is completely unrelated to the plan that generated the observations.

In both cases, the resulting heuristic value may prevent the algorithm from selecting the actual goal from among the goal hypotheses.

This overestimation, however, has an important property in relation to the basic operator counting heuristic, which is that h C always dominates the operator counting heuristic h, in Proposition 1.Algorithm 3 Goal Recognition using Heuristic Difference of Operator Counts.

Input: ?? planning domain definition, I initial state, G set of candidate goals, and O observations.

Output: Recognized goal(s).

DISPLAYFORM0 HG ??? a???A Ya 6: DISPLAYFORM1 HC,G ??? a???A Ya 10: DISPLAYFORM2 Proposition 1 (h C dominates h).

Let h be the operatorcounting heuristic from Defs.

10-11, h C be the overconstrained heuristic that accounts for all observations o ??? O, and s a state of ??. Then h C (s) ??? h(s).Proof.

Let C h be set of constraints used in h(s), and C h C be set of constraint used to compute h C (s).

Every feasible solution to C h C is a solution to C h .

This is because to generate C h C we only add constraints to C h .

Thus, a solution to C h C has to satisfy all constraints in C h .

Therefore, since we are solving a minimization problem the value of the solution for C h cannot be larger than the solution to C h C .The intuition here is that the operator-counting heuristic h estimates the total cost of any optimal plan, regardless of the observations, while h C estimates a plan following all observations, including noise, if any.

If there is no noise, the sum of the counts must agree (even if the counts are different), whereas if there is noise and assuming the noise is evenly distributed, there will be differences in all counts.

Thus, our last approach consists of computing the difference between h C and h, and infer that the goal hypothesis for which these values are closer must be the correct goal.

We call the resulting heuristic h ?? and formalize this approach in Algorithm 3.

Here we compute the LP twice, once with only the basic operator-counting constraints (Line 4), and once with the constraints enforcing the observations in the operator counts (Line 8), using these two values to compute h ?? (Line 10).

The algorithm then returns goal hypotheses that minimize h ?? (Line 11).

To evaluate the effectiveness of our approaches, we implemented each of the algorithms described earlier and performed the goal recognition process over the large dataset introduced by BID2 .

This dataset contains thousands of problems for goal and plan recognition under varying levels of observability for a number of traditional IPC domains (Vallati et al. 2018 2 .

We implemented our approaches using PYTHON 2.7 for the main recognition algorithms with external calls to a customized version of the FAST-DOWNWARD BID1 ) planning system to compute the operator counts.

Our customized planner returns not only the operator counts and can also introduce additional constraints before running the CPLEX 128 optimization system.

We ran experiments in a single core of a 24 core Intel R Xeon R CPU E5-2620 @2.00Ghz with 48GB of RAM, with a 2-minute time limit and a 2GB memory limit.

Table 1 shows the results for the partially observable, nonnoisy fragment of the dataset, whereas TAB3 shows the noisy fragment of the dataset 3 .

For the noisy experiments, each set of observations contained at least two spurious actions, which, while valid for the plan, were not actually executed by the agent being observed.

These results show that, while not nearly as fast as the h uniq approach from Pereira et al. with a ?? = 0 recognition threshold, the accuracy (Acc %) of our h ?? approach is either competitive or superior in virtually all domains (except for some levels of observability in IPC-GRID, DWR and KITCHEN), and, even for the domains where the accuracy is similar, or lower, the spread (SinG) of the resulting goals is consistently lower, i.e., the returned goals are unique for most problems.

The accuracy of our approach, thus, consistently matches or surpasses that of R&G 2009, with a computational cost that is also often smaller than FG 2015.

Importantly, the cost of all of our approaches is basically the same within each domain, regardless of the level of observability and noise, since our technique relies on a single call to a planner that computes the operator counts for a single state and then stops the planner.

We argue that this is attributable to our inefficient implementation rather than the technique, for the h ?? approach, the overhead of the FAST-DOWNWARD pre-processing step is paid multiple times.

Unlike R&G 2009, that uses a modified planning heuristic, and FG 2015, that builds a data structure and explores it at very high computational cost.

We note that the results for noisy observations show the greatest impact of h ?? with an overall higher accuracy and lower spread across all domains but KITCHEN.Finally, results for the KITCHEN domain stand out in our experiments in that our some of our approaches consistently show underwhelming performance both in noisy and nonnoisy domains.

Counter-intuitively, for this particular do-main, the more observations we have available, the worse the performance.

This seems to be a problem for all other approaches under noisy conditions, though not under incomplete observations.

Moreover, since the loss of accuracy with fuller observability also occurs for the non-noisy setting, we surmise this to stem from the domain itself, rather than the algorithm's ability to handle noise, and defer investigation of this issue to future work.

Our work follows the traditional of goal and plan recognition as planning algorithms as defined by BID2 BID3 .

The former work yields higher recognition accuracy in our settings (and hence we chose it as a baseline), whereas the latter models goal recognition as a problem of estimating the probability of a goal given the observations.

Such work uses a Bayesian framework to compute the probability of goals given observations by computing the probability of generating a plan given a goal, which they accomplish by running a planner multiple times to estimate the probability of the plans that either comply or not with the observations.

Recent research on goal recognition has yielded a number of approaches to deal with partial observability and noisy observations, of which we single out three key contributions.

First, BID1 developed a goal recognition approach based on constructing a planning graph and propagating operator costs and the interaction among operators to provide an estimate of the probabilities of each goal hypothesis.

While their approach provides probabilistic estimates for each goal, its precision in inferring the topmost goals is consistently lower than ours, often ranking multiple goals with equal probabilities (i.e., having a large spread).

Second, BID3 developed an approach that also provides a probabilistic interpretation and explicitly deals with noisy observations.

Their approach works through a compilation of the recognition problem into a planning problem that is processed by a planner that computes a number of approximately optimal plans to compute goal probabilities under R&G's Bayesian framework.

Finally, BID2 develop heuristic goal recognition approaches using landmark information.

This approach is conceptually closer to ours in that we also compute heuristics, but we aim to overcome the potential sparsity of landmarks in each domain by using operator-count information, as well as explicitly handle noise by introducing additional constraints in heuristic h C and comparing the distance to the unconstrained h heuristic.

We developed a novel class goal recognition technique based on operator-counting heuristics from classical planning (Pommerening et al. 2014) which, themselves rely on ILP constraints to estimate which operators occur in valid optimal plans towards a goal.

The resulting approaches are competitive with the state of the art in terms of high accuracy and low false positive rate (i.e., the spread of returned goals), at a moderate computational cost.

We show empirically that the overall accuracy of our best approach is sub- stantially superior to the state-of-the-art over a large dataset.

Importantly, the values of the operator-counting constraints we compute for each of the heuristics can be used as explanations for recognized goals.

The techniques described in this paper use a set of simple additional constraints in the ILP formulation to achieve substantial performance, so we expect substantial future work towards further goal recognition approaches and heuristics that explore more refined constraints to improve accuracy and reduce spread, as well as deriving a probabilistic approach using operator-counting information.

Examples of such work include using the constraints to force the LP to generate the counterfactual operator-counts (i.e., non-compliant with the observations) used by the R&G approach, or, given an estimate of the noise, relax the observation constraints to allow a number of observations to not be included in the resulting operator-counts.

DISPLAYFORM0

@highlight

A goal recognition approach based on operator counting heuristics used to account for noise in the dataset.