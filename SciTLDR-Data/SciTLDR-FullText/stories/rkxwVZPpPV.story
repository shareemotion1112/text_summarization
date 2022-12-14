Heuristic search research often deals with finding algorithms for offline planning which aim to minimize the number of expanded nodes or planning time.

In online planning, algorithms for real-time search or deadline-aware search have been considered before.

However, in this paper, we are interested in the problem of {\em situated temporal planning} in which an agent's plan can depend on exogenous events in the external world, and thus it becomes important to take the passage of time into account during the planning process.

Previous work on situated temporal planning has proposed simple pruning strategies, as well as complex schemes for a simplified version of the associated metareasoning problem.

In this paper, we propose a simple metareasoning technique,  called the crude greedy scheme, which can be applied in a situated temporal planner.

Our empirical evaluation shows that the crude greedy scheme outperforms standard heuristic search based on cost-to-go estimates.

For many years, research in heuristic search has focused on the objective of minimizing the number of nodes expanded during search (e.g BID7 ).

While this is the right objective under various scenarios, there are various scenarios where it is not.

For example, if we still want an optimal plan but want to minimize search time, selective max BID10 or Rational Lazy A˚ ) can be used.

Other work has dealt with finding a boundedly suboptimal plan as quickly as possible BID20 , or with finding any solution as quickly as possible BID21 .

Departing from this paradigm even more, in motion planning the setting is that edge-cost evaluations are the most expensive operation, requiring different search algorithms BID14 BID11 .While the settings and objectives mentioned above are quite different from each other, they are all forms of offline planning.

Addressing online planning raises a new set of objectives and scenarios.

For example, in real-time search, an agent must interleave planning and execution, requiring still different search algorithms BID13 BID18 BID6 BID5 .

Deadline-aware search BID9 must find a plan within some deadline.

The BUGSY planner BID1 attempts to optimize the utility of a plan, which depends on both plan quality and search time.

In this paper we are concerned with a recent setting, called situated temporal planning BID2 .

Situated temporal planning addresses a problem where planning happens online, in the presence of external temporal constraints such as deadlines.

In situated temporal planning, a plan must be found quickly enough that it is possible to execute that plan after planning completes.

Situated temporal planning is inspired by the planning problem a robot faces when it has to replan BID3 , but the problem statement is independent of this motivation.

The first planner to address situated temporal planning BID2 ) used temporal reasoning BID8 prune search nodes for which it is provably too late to start execution.

It also used estimates of remaining search time BID9 together with information from the temporal relaxed planning graph BID4 ) to estimate whether a given search node is likely to be timely, meaning that it is likely to lead to a solution which will be executable when planning finishes.

It also used dual open lists: one only for timely nodes, and another one for all nodes (including nodes for which it is likely too late to start execution).

However, the planner still used standard heuristic search algorithms (GBFS or Weighted A˚) with these open lists, while noting that this is the wrong thing to do, and leaving for future work finding the right search control rules.

Inspired by this problem, a recent paper BID19 proposed a rational metareasoning BID17 approach for a simplified version of the problem faced by the situated planner.

The problem was simplified in several ways: first, the paper addressed a one-shot version of the metareasoning problem, and second, the paper assumed distributions on the remaining search time and on the deadline for each node are known.

The paper then formulated the metareasoning problem as an MDP, with the objective of maximizing the probability of finding a timely plan, and showed that it is intractable.

It also gave a greedy decision rule, which worked well in an empirical evaluation with various types of distributions.

In this paper, we explore using such a metareasoning approach as an integrated part of a situated temporal planner.

This involves addressing the two simplifications described above.

The naive way of addressing the first simplification -the one-shot nature of the greedy rule -is to apply it at every expansion decision the underlying heuristic search algorithm makes, in order to choose which node from the open list to expand.

The problem with this approach is that the number of nodes on the open list grows very quickly (typically exponentially), and so even a linear time metareasoning algorithm would incur too much overhead.

Thus, we introduce an even simpler decision rule, which we call the crude greedy scheme, which does not require access to the distributions, but only to their estimated means.

Additionally, the crude greedy scheme allows us to compute one number for each node,Q, and expand nodes with a highQvalue first.

This allows us to use a regular open list, although one that is not sorted according to cost-to-go estimates, as in standard heuristic search.

In fact, as we will see, cost-to-go estimates play no role in the ordering criterion at all.

An empirical evaluation on a set of problems from the Robocup Logistics League (RCLL) domain BID16 BID15 shows that using the crude greedy scheme in the situated temporal planner BID2 leads to a timely solution of significantly more problems than using standard heuristic search, even with pruning late nodes and dual open lists.

Next, we briefly survey the main results of the metareasoning paper BID19 , and then describe how we derive the crude greedy decision rule, and conclude with an empirical evaluation that demonstrates its efficacy.

A model called AE2 ('allocating effort when actions expire') that assigns processing time under the simplifying assumption of n independent processes was proposed by BID19 .

In order to make this paper self-contained, we re-state the model and its properties below.

The AE2 model abstracts away from the search, and assumes n processes (e.g., each process can be thought of as a node on the open list) that each attempts to solve the same problem under time constraints.

(For example, these may represent promising partial plans for a certain goal, implemented as nodes on the frontier of a search tree, but as discussed below the problems may be completely unrelated to planning.)

There is a single computing thread or processor to run all the processes, so it must be shared.

When process i terminates, it will, with probability P i , deliver a solution or, otherwise, indicate its failure to find one.

For each process, there is a deadline, defined in absolute wall clock time, by which the computation must be completed in order for any solution it finds to be valid, although that deadline may only be known to us with uncertainty.

For process i, let D i ptq be the CDF over wall clock times of the random variable denoting the deadline.

Note that the actual deadline for a process is only discovered with certainty when its computation is complete.

This models the fact that, in planning, a dependence on an external timed event might not become clear until the final action in the plan is added.

If a process terminates with a solution before its deadline, we say that it is timely.

The processes have performance profiles described by CDFs M i ptq giving the probability that process i will terminate given an accumulated computation time on that process of t or less.

Although some of the algorithms we present may work with dependent random variables, we assume in our analysis that all the variables are independent.

Given the D i ptq, M i ptq, and P i , the objective of AE2 is to schedule processing time over the n processes such that the probability that at least one process finds a solution before its deadline is maximized.

This is the essential metareasoning problem in planning when actions expire.

We now represent the AE2 problem of deliberation scheduling with uncertain deadlines as a Markov decision process.

For simplicity, we initially assume that time is discrete and the smallest unit of time is 1.

Allowing continuous time is more complex because one needs to define what is done if some time-slice is allocated to a process i, and that process terminates before the end of the time-slice.

Discretization avoids this complication.

We can now define our deliberation scheduling problem as an the following MDP, with distributions represented by their discrete probability function (pmf).

Denote m i ptq " M i ptq´M i pt´1q, the probability that process i completes after exactly t time units of computation time, and d i ptq " D i ptq´D i pt´1q, the probability that the deadline for process i is exactly at time t. Without loss of generality, we can assume that P i " 1: otherwise modify the deadline distribution for process i to have d i p´1q " 1´P i , simulating failure of the process to find a solution at all with probability 1´P i , and multiply all other d i ptq by P i .

This simplified problem we call SEA2.

We formalize the SEA2 MDP as an indefinite duration MDP with terminal states, where we keep track of time as part of the state. (An alternate definition would be as a finite-horizon MDP, given a finite value d for the last possible deadline.)The actions in the MDP are: assign the next time unit to process i, denoted by a i with i P r1, ns.

We allow action a i only if process i has not already failed.

The state variables are the wall clock time T and one state variable T i for each process, with domain N Y tF u. T i denotes the cumulative time assigned to each process i until the current state, or that the process has completed computation and resulted in failure to find a solution within the deadline.

We also have special terminal states SUCCESS and FAIL.

Thus the state space is: DISPLAYFORM0 The initial state is T " 0 and T i " 0 for all 1 ď i ď n.

The transition distribution is determined by which process i has last been scheduled (the action a i ), and the M i and D i distributions.

If all processes fail, transition into FAIL with probability 1.

If some process is successful, transition into SUCCESS with probability 1.

More precisely:• The current time T is always incremented by 1.• Accumulated computation time is preserved, i.e. for action a i , T j pt`1q " T j ptq for all processes j ‰ i.• T i ptq " F always leads to T i pt`1q " F .•

For action a i (assign time to process i), the probability that process i's computation is complete given that it has not previously completed is P pC i q " DISPLAYFORM1 1´MipTiq .

If completion occurs, the respective deadline will be met with probability 1´D i pT i q. Therefore, transition probabilities are: with probability 1´P pC i q set T i pt`1q " T i ptq`1, with probability P pC i qD i pT i q set T i pt`1q " F (process i failed to meet its deadline), and otherwise (probability P pC i qp1´D i pT i q) transition into SUCCESS (the value of T i in this case is 'don't care').• If T i pt`1q " F for all i, transition into FAIL.

The reward function is 0 for all states, except SUCCESS, which has a reward of 1.

It was shown in BID19 that solving the AE2 MDP is NP-hard, and it was conjectured to be even harder (possibly even PSPACE-complete, like similar MDPs).

On the other hand, under the restriction of known deadlines and a special condition of diminishing returns (non-decreasing logarithm of probability of failure) that an optimal schedule can be found in polynomial time.

However, neither known deadlines nor diminishing returns strictly hold in practice in planning processes.

Still, the algorithm for diminishing returns provided insights that were used to create an appropriate greedy scheme.

The greedy scheme, briefly repeated below, is relatively easy to compute and achieved good results empirically.

Define m i ptq " M i ptq´M i pt´1q, the probability that process i completes after exactly t time units of computation time.

Under an allocation A i " p0, tq in which all processing time starting from time 0 until time t is allocated to process i, the success distribution for process i is: DISPLAYFORM0 Define the most effective computation time for process i under this assumption to be: DISPLAYFORM1 The latter is justified by observing that the term´logp1f i ptqq behaves like utility, as it is monotonically increasing with the probability of finding a timely plan in process i; and on the other hand it behaves additively with the terms for other processes.

That is, if we could start all processes at time 0 and run them for time t, and if all the random variables were jointly independent, then indeed maximizing the sum of the´logp1´f i ptqq terms results in maximum probability of a timely plan.

However, since not all processes can start at time 0, the intuition from the diminishing returns optimization is thus to prefer the process i that has the best utility per time unit.

i.e. such that´logp1´f i ptqq{pe i q is greatest.

Still, allocating time now to process i delays other processes, so it is also important to allocate the time now to a process that has a deadline as early as possible, as this is most critical.

BID19 therefore suggested the following greedy algorithm: Whenever assigning computation time, allocate t d units of computation time to process i that maximizes: DISPLAYFORM2 where α and t d are positive empirically determined parameters, and ErD i s is the expectation of the random variable that has the CDF D i , which we use as a proxy for 'deadline of process i'.

(This is a slight abuse of notation in the interest of conciseness, as ErD i s could be taken to mean the expectation of the CDF, which is not what we want here.)

The α parameter trades off between preferring earlier expected deadlines (large α) and better performance slopes (small α).

Using the proxy ErD i s in the value Qpiq is reasonable, but somewhat ad-hoc.

It also encounters problems if ErD i s is zero or even near-zero.

A more disciplined scheme can indeed use the utility per time unit as in Qpiq, but the first term should be better justified theoretically.

The reason for including the deadline in Qpiq is in order to give preference to processes with an early deadline, because deferring their processing may cause them to be unable to complete before their deadline (even if they would have been timely had they been scheduled for processing immediately).

Therefore, instead of the first term it makes sense to provide a measure of the "utility damage" to a process i due to delaying its processing start time from time 0 to time t d .

This can be computed exactly, as follows.

Define a 'generalized' f 1 i , the probability of process i finding a timely plan given a contiguous computation time t starting at time t d , as follows: DISPLAYFORM0 (4) Note that this is the same as f , except that processing starts at time t d , which is the same as saying that the deadline distribution is advanced by t d (and indeed, f i ptq " f 1 i pt, 0q).

Assuming that the time we wish to assign to process i is e i , before the delay we can achieve a utility of: logp1´f i pe i qq, and after delay of t d can achieve´logp1f DISPLAYFORM1 The difference between the former and the latter values is the 'damage' caused by the delay.

Thus, our improved greedy scheme is to assign t d time units to the process that maximizes: DISPLAYFORM2 Observe that the first term is proportional to the logarithm of: 1´fIntegrating the greedy scheme into a plannerIn order to actually use the greedy scheme in a planner, several issues must be handled.

Foremost is the issue of obtaining the distributions, which is non-trivial.

Second, although the greedy scheme is quite efficient, it is not quite efficient enough for making decisions about node expansions, which must be done in essentially negligible time.

Hence, we consider a crude version of the greedy scheme below.

Crude version of the greedy scheme Consider Equation 3 defining Qpiq.

The estimate for ErD i s in the first term can use any current estimate of the deadline time.

For the second term in Qpiq, we can approximate e i by the expected time to return a solution.

We use estimate both of these quantities as described by BID2 .

We now briefly review these estimates, but refer the interested reader to the original paper.

To estimate the current deadline ErD i s, we use the temporal relaxed planning graph (TRPG) BID4 .

Specifically, we compute the slack of the chosen relaxed plan, that is, how much we can delay execution of the entire plan (the actions leading to the current node together with the actions in the relaxed plan).

Note that, because the relaxed plan is not guaranteed to be optimal, this is not necessarily an admissible estimate.

To estimate the remaining search time e i , we use an idea from Deadline Aware Search BID9 .

We estimate the 'distance from solution' (i.e. estimation of number of expansions from the current node, also based on the relaxed), and divide it by the 'progress rate' (i.e. the reciprocal of the time difference between the time a node is expanded and the time its parent was expanded, averaged over multiple nodes).The numerator logp1´f i pe i qq is more problematic, as it requires f i , which uses the complete distribution.

Note that this term is negative, and we want it to be as large as possible in absolute value.

The simplest crude approximation is a constant logp1´f i pe i qq, but that is an oversimplification.

Note that if the most effective computation time e i is greater than ErD i s then in fact we are not likely to find a solution in time in process i.

For simplicity, we thus use´logp1´f i pe i qq « maxp0, βpErD i s´e i qq for some parameter β as a first approximation.

The idea here is that ErD i s´e i is an estimate of the slack (spare time) we have in completing the computation before the deadline.

We are then assuming that the negative logarithm of the probability of not completing in time is approximately proportional to the slack.

This slack is also already estimated by the situated temporal planner, based on the partial plan to the current node and the temporal relaxed planning graph from it BID2 .Note that once we plug this into Equation 3, the β can be absorbed into the α parameter.

An additional issue is that in a planner, since ErD i s is relative to the time now, this value keeps decreasing and may approach 0.

This may cause the α ErDis term to grow without bound.

To fix this, we bound the denominator away from 0 to the time t 10 , the time required for 10 node expansions.

In summary, for our crude greedy approach, we expand next the node with the highest value of Qpiq " maxp0, ErD i s´e i q e i`α maxpErD i s, t 10 qThis crude version of the greedy scheme has two advantages: it does not require the complete distributions D i and M i , and is more computationally efficient as it does not have to compute the summation in the equation for f i .

This comes at the cost of a potential oversimplification that may cause schedule quality to excessively degrade.

An additional problem is that the original greedy scheme itself using the ErD i s was only a first-order approximation, and in fact distributions can be devised where it fails badly.

To evaluate the crude greedy scheme, we implemented it on top of the situated temporal planner of BID2 , which itself is implemented on top of OPTIC BID0 .

We ran the planner using the crude greedy scheme, with different values of α, and compared it to the original situated temporal planner, which sorts its open lists based on cost-to-go estimates (denoted h below).

Both planners used exactly the same pruning method for nodes which are guaranteed to be too late, and the same dual open list mechanism for preferring nodes which are likely to be timely.

We compared the results of the different planners on instances of the Robocup Logistic League Challenge BID16 BID15 ), a domain that involves robots moving workpieces between different workstations.

The goal is to manufacture and deliver an order within some time window, and thus situated temporal planning is very natural here.

TAB1 shows the planning time for the baseline planner (h) and the planner using the crude greedy scheme with different values of α.

In the table, 'x' means 'failed to find a plan in time to satisfy the deadline(s)'.

As these results show, the crude greedy scheme solves significantly more problems than the baseline for any value of α.

This provides support for a metareasoning approach to allocating search effort in situated planning.

It also suggests that, for situated temporal planning, costto-go estimates are not the right primary source of heuristic guidance.

In this paper, we have provided the first practical metareasoning approach for situated temporal planning.

We showed empirically that this approach outperforms standard heuristic search based on cost-to-go estimates.

Nevertheless, the temporal relaxed planning graph BID4 ) serves an important purpose here, allowing us to estimate both remaining planning time and the deadline for a node.

Thus, we believe our results suggest that cost-to-go estimates are not as important for situated temporal planning as they are for minimizing the number of expanded nodes or planning time as in classical heuristic search.

The metareasoning scheme we provided is a crude version of the greedy scheme of BID19 .

We introduced approximations in order to make the metareasoning sufficiently fast and in order to utilize only readily available information generated during the search.

We also proposed a more refined and better theoretically justified version of the algorithm ('improved greedy'), but making the improved version applicable in the planner is a non-trivial challenge that forms part of our future research.

Ongoing Work: Crude version of the improved greedy schemeThe improved greedy scheme is better justified, but has an additional term where we need the complete distribution (f 1 pt, t d q is needed, rather than just the expectation ErD i s).We would like to replace this distribution with a small number of parameters than can be easier to obtain.

Basically the same considerations apply here as well, except that the the term involving f 1 i requires access to the full distributions m i , D i .

Given specific distribution types, it may be possible to compute this term as a function of ErD i s and e i .

However, this part of the work is still in progress and at present we are not sure what parameters we can obtain during the search that would support the improved scheme.

@highlight

Metareasoning in a Situated Temporal Planner

@highlight

This paper addresses the problem of situated temporal planning, proposing a further simplification on  greedy strategies previously proposed by Shperberg.