Oversubscription planning (OSP) is the problem of finding plans that maximize the utility value of their end state while staying within a specified cost bound.

Recently, it has been shown that OSP problems can be reformulated as classical planning problems with multiple cost functions but no utilities.

Here we take advantage of this reformulation to show that OSP problems can be solved optimally using the A* search algorithm, in contrast to previous approaches that have used variations on branch-and-bound search.

This allows many powerful techniques developed for classical planning to be applied to OSP problems.

We also introduce novel bound-sensitive heuristics, which are able to reason about the primary cost of a solution while taking into account secondary cost functions and bounds, to provide superior guidance compared to heuristics that do not take these bounds into account.

We implement two such bound-sensitive variants of existing classical planning heuristics, and show experimentally that the resulting search is significantly more informed than comparable heuristics that do not consider bounds.

Oversubscription planning (OSP) problems are a family of deterministic planning problems.

In contrast to classical planning, where a set of hard goals is specified and the planner searches for a minimal (or low) cost plan that reaches a state in which all of the goals are made true, oversubscription planning specifies a utility function that describes the benefit associated with achieving different possible states, and asks for a plan whose cost does not exceed a set bound and achieves as high a utility as possible BID5 .While domain-independent classical planning approaches have increasingly standardized around variations on A * search and heuristics that are automatically extracted from the problem description BID0 BID3 BID0 Edelkamp, 2001; BID2 Helmert and Domshlak, 2009] , OSP has generally been solved with branch-and-bound algorithms and heuristics that compute an admissible (in this context nonunder) estimate of the utility achievable from a state.

In order to obtain these estimates, recent approaches often adapt classical planning techniques such as landmarks BID4 BID5 or abstractions BID4 , and enhance them with reasoning that is specific to the context of OSP, such as the knowledge that there always exists an optimal plan that ends with a utility-increasing action, or that the cost bound for the problem can be reduced under specific conditions to aid the search algorithm in detecting that improving over the currently achieved utility is impossible.

In contrast to these approaches, our aim here is to show that general methods from classical planning, including A * search, can be used in the OSP setting nearly as is.

This previously turned out to be the case for the related net-benefit planning problem, where classical planners solving a compilation were shown to outperform planners designed specifically for that task BID3 .

Here, we use a similar, recently proposed compilation that converts OSP problems into classical planning problems with multiple cost functions but no utilities BID3 .

In addition, we demonstrate that existing classical planning heuristics can be used to guide the search for optimal plans.

While these heuristics are typically uninformative out-of-the-box, they require only minor modifications (and no specific reasoning about utilities) to render them sensitive to the secondary cost functions and bounds that are introduced by the compilation.

Our experiments with A * and the newly introduced estimators that we refer to as bound-sensitive heuristics show that they lead to informed searches that are competitive with, and in some cases outperform, the state of the art for optimal OSP.One related area of research in the classical setting is that of bounded-cost planning, where the planner looks for any plan with (primary) cost below a given bound, similar to the treatment of the secondary cost in the OSP setting.

Approaches proposed for this setting include dedicated search algorithms BID6 and heuristics that take into account accumulated cost and plan length at the current search node BID7 BID0 .

These approaches work by preferentially expanding nodes in areas of the search space that are likely to have a solution under the cost bound.

Optimal OSP, however, requires expanding all nodes that potentially lie on a path to state with maximal utility.

Furthermore, it cannot be assumed that solutions necessarily achieve all soft goals.

Heuristics that are able to take into account bounds on secondary cost functions have also been investigated in the stochastic shortest path setting, where they were used as additional constraints in an LP-based heuristic to consider limitations on fuel or time resources BID7 .We now briefly review the various flavors of planning that we consider in this work, and introduce the formalisms by which we describe them.

We describe planning problems in terms of extensions to the SAS + formalism [B??ckstr??m and Nebel, 1995] .

A classical planning task ?? = V , O; s I , G, C is given by a set of variables V , with each variable v ??? V having a finite domain dom(v), a set of actions O, with each action o ??? O described by a pair pre(o), eff(o) of partial assignments to V , called the precondition and effect of o, respectively, initial state s I and goal condition G, which are full and partial assignments to V , respectively, and the cost function C : O ??? R 0+ .

A state s is given by a full assignment to V .

An action is said to be applicable in a state s if pre(o) ??? s, and s o denotes the result of applying o in s, where the value of each v ??? V is given by eff(o) [v] if defined and s [v] otherwise.

An operator sequence ?? = o 1 , . . . , o k is applicable in s if there exist states s 0 , ?? ?? ?? , s k such that (i) s 0 = s, and (ii) for each 1 ??? i ??? k, o i is applicable in s i???1 and s i = s i???1 o i .

We refer to the state s k by s ?? and call it the end state of ??.

An operator sequence ?? is a plan for a classical planning problem if it is applicable in s I and G ??? s I ?? .

The cost of a plan ?? is given by C(??) = o????? C(o); the goal of optimal classical planning is to find a plan with minimal cost.

We refer to a pair of variable v and its value ?? ??? dom(v) as a fact and denote it by v, ?? .

We sometimes abuse notation and treat partial assignments as sets of facts.

An oversubscription planning (OSP) problem is given by ?? OSP = V , O, s I , C , u, B , where V , O, s I , and C are as in classical planning, u : ( v, ?? ) ??? R 0+ is a non-negative valued utility function over variable assignments (facts), and B is a cost bound for the plan, imposing the additional requirement that only plans ?? such that C(??) ??? B are valid.

The utility of a plan ?? is given by v,?? ???s I ?? u( v, ?? ); the objective of OSP problems is to find valid plans with maximal utility.

A multiple cost function (MCF) problem is given by ?? MCF = V, O, s I , G, C 0 , C , where V , O, s I , and C 0 are as in classical planning, C 0 is the primary cost function, and C = { C i , B i | 1 ??? i ??? n} is a set of secondary cost functions C i : O ??? R 0+ , and bounds, both non-negative.

Valid plans for MCF planning problems fulfill the condition C i (??) ??? B i for all secondary cost functions, and optimal plans for MCF planning have minimal primary cost C 0 (??).

In this paper we only consider MCF problems with a single secondary cost function, i.e. n = 1.

It has recently been shown that an OSP problem can be compiled into an MCF planning problem with a single secondary cost function that corresponds to the cost function C of the original problem, and is constrained to not exceed the specified bound B BID3 .

The primary cost function for the problem, or the cost function to be optimized, results from compiling the utilities from the original problem into costs.

Two different compilations have been proposed for this task: (i) the soft goals compilation, which adds for each variable v that has some value ?? ??? dom(v) for which a utility is specified, a hard goal, along with actions that are able to achieve this hard goal at different costs, and (ii) the state delta compilation which encodes in the cost of each action the change in state utility that results from applying it.

Here we consider only (i), as (ii) introduces negative action costs that A * and existing classical planning heuristics are not designed to handle.

Note, however, that our methods do not depend on the specific choice of compilation, as long as they remove utilities from the problem and do not introduce negative action costs.

The soft goals compilation was originally introduced in the context of net-benefit planning, which is similar to oversubscription planning but does not specify a bound on plan cost, having instead as an objective the minimization of the difference between the achieved utility and the cost of the plan BID3 .

It can be applied in the OSP setting to result in an MCF planning problem as follows: DISPLAYFORM0 denoting the maximum utility over the values of the variable v.

In the reformulated problem, only the o v,?? actions for which ?? is not the maximum utility value of v have positive primary costs.

These actions make explicit that a particular utility will not be achieved, and that the plan has instead chosen to achieve the associated g v by accepting the associated cost penalty.

The primary cost of a plan ?? for the reformulated problem is then given by v???V u max (v) ??? f ???s ?? u(f ).Note that this compilation assumes that utilities are defined for single facts.

The more general case, in which utilities are instead defined for logical formulae ??, can be handled as in the soft goals compilation by introducing a new variable v ?? , and two actions that achieve its goal value with cost 0 and precondition ??, and cost u(??) and precondition ???, respectively BID3 ].

Since we consider only single fact utilities here, we do not discuss this case in detail.

While this compilation is sound as stated, two further optimizations can be made to reduce the state space of the resulting compiled problem.

First, an arbitrary ordering can be introduced over V to ensure that the g v values are achieved in a fixed sequence, to avoid searching over different orderings.

Second, a new precondition fact that is deleted by the o v,?? actions can be added to the original domain actions to ensure that o v,?? actions happen only at the end of the plan and are not interleaved with the original domain actions.

We make use of both of these optimizations here.

The A * algorithm extends blind search techniques such as Dijkstra's algorithm by allowing the incorporation of admissible (non-overestimating) heuristics [Hart et al., 1968] .

In each iteration of its main loop, A * picks a node n to expand with minimal f (n) = g(n) + h(n) value, where g(n) is the cost of the path to n, and h(n)

is an admissible estimate of the remaining cost to the goal.

An optimal solution to the problem is found when a node n with minimal f (n) value is a goal node.

To adapt A * to the MCF planning setting, we store at each node n a set of accumulated path costs g i (n) resulting from each of the secondary cost functions C 1 , . . .

, C n , in addition to the accumulated primary cost g 0 (n).

When a node is taken from the priority queue and expanded, generated successor nodes for which any g i (n) > B i can be immediately pruned, as all C i are assumed to be non-negative, and they cannot constitute valid prefixes for solution paths.

One key optimization used in modern A * implementations in the classical setting is duplicate detection, which allows states that are rediscovered during search to be discarded, if the new g value exceeds the cost of the path to the state that was previously found, or to be updated with a new parent, if the cost of the new path is less.

In the MCF setting, care must be taken to ensure that newly discovered nodes are discarded (or replace existing nodes), only when they are dominated by (or dominate), the existing node in all cost dimensions.

While the only necessary property of the open list from a correctness perspective is that it order nodes by increasing primary f (n) value, the choice of a secondary ordering heuristic plays a role here: an ordering that causes a dominating node to be generated first and enables subsequently generated nodes to be immediately discarded as dominated results in superior performance.

In our implementation of the algorithm, we therefore use an open list that orders nodes by increasing g i (n) value when their primary f (n) values are the same.

While any admissible heuristic can be used to guide search in MCF planning, classical planning heuristics that ignore bounds entirely are typically extremely uninformative.

Consider the problem shown in FIG0 : the agent is initially at l 0 , and can obtain a utility of 10 by visiting each of the locations l 1 and l 2 .

The costs of the actions move(l 0 , l 1 ) and move(l 1 , l 2 ) are both 1.

In the compiled MCF version of this problem, an optimal but naive heuristic that ignores the bound will give an estimate for the primary cost of 0, as both visited(l 1 ) and visited(l 2 ) can be made true, and the associated 0-primary cost o visited(l * ) actions applied to reach the newly introduced hard goals corresponding to each utility.

If, however, B = 1, the optimal C 0 cost at l 0 is 10, since l 2 cannot be reached at cost ??? B and the agent must use the o not-visited(l2) action to achieve the associated hard goal with a cost of 10.

Similarly, if B = 0, the C 0 cost of the optimal plan is 20, since the value of C 1 for all available actions exceeds the bound B. In practice, it turns out that the OSP versions of many classical planning problems have similar behavior: their state spaces are strongly connected, so any variable assignment can be achieved from any state, and classical planning heuristics that ignore bounds are no more informed than blind search.

In order to obtain estimates that take secondary cost bounds into account and can guide heuristic search towards feasible solutions, we therefore introduce bound-sensitive heuristics.

In the following, we use b to denote a budget vector of nonnegative reals that indicate the unused component of each of the secondary cost bounds B i at a given search node.

Definition 2 (Optimal bound-sensitive heuristic) Given an MCF planning problem ?? MCF = V, O, s I , G, C 0 , C , the optimal bound-sensitive heuristic h * (s, b) for a state s and budget vector b is given by the minimal primary cost C 0 (??) of a plan ?? for s such that C i (??) ??? b i for i = 1, . . .

, n.

By analogy with standard admissible heuristics, an admissible bound-sensitive heuristic is a non-overestimating bound-sensitive heuristic:Definition 3 (Admissible bound-sensitive heuristic) Given an MCF planning problem ?? MCF = V, O, s I , G, C 0 , C , an admissible bound-sensitive heuristic h(s, b) for a state s and budget vector b is a heuristic h such that h(s, b) ??? h * (s, b) for all s, b.

Any classical planning heuristic that completely ignores C i and B i can be thought of as an admissible bound-sensitive heuristic that assumes b = ???. As the value of b decreases, the value of h * (s, b) can only increase.

In general, it is useful to keep in mind the following property: DISPLAYFORM0 DISPLAYFORM1 Given a state s and budget vectors b, b' such that bProof sketch: This follows from the fact that any plan ?? for s such that C i (??) ??? b i also has the property that C i (??) ??? b' i for i = 1, . . . , n since b ??? b', yet the opposite is not the case.

Theorem 1 applied to MCF planning problems obtained as the soft goals compilations of OSP problems states that for any s, decreasing b increases h * (s, b), and decreases the achievable utility, since the primary cost here indicates the utility that the plan must declare unachievable through o DISPLAYFORM2 Bound-Sensitive h maxThe admissible classical heuristic h max estimates the cost of a set of facts F as the cost of the most expensive fact f ??? F , and applies this approximation recursively to action preconditions in order to obtain the cost of the goal BID0 : DISPLAYFORM3 where h max C denotes the value of h max computed with a cost function C, and achievers(f, s) denotes the set of actions o for which f ??? eff(o).

Note that the h max cost of a fact f that is not present in s is computed by choosing an action o from this set that achieves it with minimum possible cost.

Given a set of secondary cost functions and bounds C = { C 1 , B 1 , . . .

, C n , B n }, a bound-sensitive version of h max can easily be obtained by replacing the set of achievers used to compute h max C0 with achievers(f, DISPLAYFORM4 where actions o for which any estimate h max Ci (o, s) exceeds B i are not considered.

Note that due to the admissibility of h max , this restriction of the set of achievers is sound but not complete: it is guaranteed that any action removed from the set of achievers cannot be used in a valid plan, but there may be additional actions that cannot be achievers but are not pruned by the heuristic.

In general, any admissible estimate h max Ci (o, s) could be used to compute achievers(f, s) C0 , but we have chosen h max here for simplicity.

Theorem 2 Bound-sensitive h max C0 is an admissible boundsensitive heuristic.

Proof sketch: This follows from the admissibility of the heuristic used to compute achievers(f, s) C0 .

Merge-and-shrink heuristics are a family of abstraction heuristics that incrementally build a representation of the full state space of a problem BID2 .

The construction process begins with the set of transition systems induced over each state variable; at each step, two transition systems are selected to be merged and replaced with their synchronized product.

Since the transition systems need to be represented explicitly in memory, before the merge a shrinking step is perfomed on the two selected transition systems to enforce a user-specified threshold on the size of the synchronized product.

This step is performed by abstracting multiple states in the current representation into a single state (and thereby losing optimality).

The final output of the algorithm consists of a single abstract transition system in which multiple states and actions from the original task are mapped to a single state or transition, respectively.

h MS (s) is then given by the cost of a shortest path from the abstract state representing s to the closest abstract goal state in the final transition system.

This estimate is admissible by definition.

To adapt merge-and-shrink to the MCF setting, we maintain for each transition in the abstract state space the minimum C i cost for i = 1, . . .

, n among all of the transitions from the original task represented by that transition.

The distance C i between any two abstract states s, s then represents a non-overestimate of the secondary cost of reaching s from s. A bound-sensitive heuristic value for a state s can be computed as the minimum C 0 cost of a path ?? from s to an abstract goal state s g whose C i cost in the abstract state space does not exceed B i , for any i. The C 0 cost of such such a path can be computed with a modified version of Dijkstra's algorithm that stores secondary cost information for each node and discards nodes for which C i > B i for any i.

MS is an admissible boundsensitive heuristic.

Proof sketch: This follows from the fact that the secondary costs used in the abstract state space are the minimums of the secondary costs C i of the represented transitions in the original problem, and the proof of admissibility of standard h MS .While the ms b heuristic can be implemented by running Dijkstra's algorithm in the abstract state space for each heuristic computation, an important optimization when a single secondary cost function is present (which is the case in the compiled OSP problems that we consider) is to run Dijkstra only once during preprocessing, and compute the primary cost in the presence of different bounds on the secondary cost.

This information can then be stored as a sequence of pairs b 0 , c 0 , . . .

, b n , c n , where b 0 , . . . , b n is strictly increasing and c 0 , . . .

, c n is strictly decreasing (recall Theorem 1).

h MS (s, b) is then given by the first c i such that b i ??? b.

We implemented our approach in the Fast Downward planner BID2 , and evaluated it on a set of publically DISPLAYFORM0 77 ??0 -14 -33 -12 -6 30 ??0 -2 -13 -2 -1 21 ??0 -6 -6 -1 -1 20 ??0 -6 -6 -2 -4 ged14 20 ??0 ??0 ??0 ??0 ??0 20 ??0 ??0 ??0 ??0 ??0 20 ??0 ??0 ??0 -1 ??0 20 ??0 ??0 ??0 ??0 ??0 DISPLAYFORM1 20 ??0 ??0 ??0 ??0 ??0 20 ??0 ??0 ??0 ??0 ??0 19 -2 ??0 -2 ??0 ??0 17 ??0 ??0 ??0 ??0 ??0 pipes-notank 45 ??0 ??0 -2 -30 -27 30 ??0 -1 -5 -14 -12 22 ??0 -2 -6 -5 -5 15 ??0 -1 -2 ??0 -1 pipes-tank 35 -2 -6 -11 -9 -9 20 ??0 -3 -5 -3 -3 16 -1 -4 -5 -1 -1 11 ??0 -1 -3 ??0 ??0 psr-small 50 ??0 ??0 ??0 ??0 ??0 50 ??0 ??0 ??0 ??0 ??0 49 ??0 ??0 ??0 +1 ??0 49 ??0 ??0 ??0 ??0 ??0 rovers 15 ??0 +1 -2 -1 ??0 8 ??0 +1 ??0 +1 ??0 6 ??0 ??0 ??0 +1 ??0 5 +1 +1 +1 +1 +1 satellite 9 ??0 +2 ??0 +2 ??0 7 ??0 ??0 ??0 +1 ??0 6 ??0 ??0 ??0 +1 ??0 5 ??0 ??0 -1 +1 ??0 scanalyzer08 13 +1 ??0 ??0 -1 -1 12 ??0 ??0 -3 ??0 ??0 12 ??0 -3 -3 ??0 ??0 12 ??0 -3 -3 ??0 ??0 scanalyzer11 10 ??0 ??0 ??0 -1 -1 9 ??0 ??0 -3 ??0 ??0 9 ??0 -3 -3 ??0 ??0 9 ??0 -4 -3 ??0 ??0 sokoban08 30 ??0 ??0 ??0 ??0 ??0 29 ??0 ??0 -1 ??0 ??0 24 ??0 +3 ??0 ??0 ??0 22 ??0 +3 +1 ??0 ??0 sokoban11 20 ??0 ??0 ??0 ??0 ??0 20 ??0 ??0 ??0 ??0 ??0 20 ??0 ??0 ??0 ??0 ??0 19 ??0 +1 -1 ??0 ??0 storage 20 ??0 ??0 -1 -1 ??0 17 ??0 ??0 -1 -1 ??0 15 ??0 ??0 ??0 ??0 ??0 14 ??0 ??0 ??0 ??0 ??0 tetris14 17 ??0 ??0 ??0 -15 -15 14 ??0 -3 -4 -13 -12 11 -1 -3 -3 -11 -9 9 ??0 -4 -4 -8 -7 tidybot11 20 ??0 ??0 ??0 -19 -19 20 ??0 -1 -3 -19 -19 18 -1 -4 -6 -17 -17 13 ??0 -6 -8 -13 -12 tidybot14 20 ??0 ??0 ??0 -20 -20 18 ??0 -2 -5 -18 -18 14 -1 -6 -10 -14 -14 6 ??0 -6 -6 -6 -6 tpp 9 ??0 ??0 ??0 -1 ??0 DISPLAYFORM2 10 ??0 ??0 -1 -1 ??0 6 ??0 ??0 ??0 ??0 ??0 4 ??0 ??0 ??0 ??0 ??0 3 ??0 ??0 ??0 +1 ??0 woodwork08 25 ??0 ??0 -3 -6 -11 15 ??0 -1 -3 -7 -4 10 ??0 +1 -1 ??0 -1 7 ??0 +2 +2 +2 ??0 woodwork11 18 ??0 -1 -2 -3 -5 10 ??0 -1 -3 -4 -4 5 ??0 +1 -1 ??0 -2 2 ??0 +2 +2 +3 -1 zenotravel 13 ??0 ??0 ??0 ??0 ??0 10 ??0 ??0 ??0 +2 ??0 8 ??0 +1 ??0 +2 ??0 8 ??0 ??0 ??0 ??0 ??0 Sum all1190 -8 -20 -92 -139 -143 897 -5 -16 -85 -82 -84 748 -5 -22 -66 -22 -53 651 +7 -20 -39 +13 -26 available OSP benchmarks BID3 .

The set of benchmarks is taken from the International Planning Competitions of recent years, in which goal facts are replaced with utilities, and the bound set at 25%, 50%, 75%, or 100% of the cost of the optimal or best known solution to each problem.

The baseline for our comparison is a blind branch-and-bound search, currently the best available configuration for oversubscription planning that we know of BID3 .

We compare this baseline to our proposed approach of A * search on the MCF compilation of the OSP task.

Since the compilation introduces intermediate states at which some but not all of the o v,?? have been applied, we use a further optimization that avoids generating these nodes and applies all of the o v,?? actions in a single step, reducing the state space to that of the original OSP task.

We experiment with blind A * search, and A * using classical h max and h MS , as well as the two heuristics' bound-sensitive variants introduced here.

For h MS , we used exact bisimulation with an abstract state space threshold of 50000 states and exact generalized label reduction BID5 .

The experiments were performed on Intel(R) Xeon(R) CPU E7-8837 @2.67GHz machines, with time and memory limits of 30min and 3.5GB, respectively.

Per-domain and overall coverage, as well as per-task node expansions for the various configurations and problem suites are shown in TAB1 and FIG1 , respectively.

We now report some observations from our results.??? Blind branch-and-bound search usually slightly outperforms blind A * in terms of coverage, except for the 100% suite.

The difference between the two may come down to the fact that A * must do extra work in ordering the priority queue, while the variant of branch and bound search that we consider uses no ordering heuristic and can use a simple stack as its search queue.

Alternately it may be due to small differences in implementation.??? Bound-sensitive heuristics are much more informative than their classical variants on OSP problems, sometimes decreasing expansions by orders of magnitude.

Compared to non-bound-sensitive heuristics, they also almost always result in better coverage.??? Blind search dominates informed search in terms of coverage when bounds are low, but the effect diminishes as the bound increases and it becomes intractable to explore the full state space under the bound.

For the 25% suite of problems, heuristic configurations solve an average of approximately 100 instances fewer than the baseline, compared to approximately 15 instances fewer on the 100% suite.

Notably, bound-sensitive h MS has the best coverage in the 100% suite, solving 13 problems more than the baseline, and 6 more than blind A * .???

Coverage on several domains benefits from more informed search schemes.

On BLOCKSWORLD, DRIVER-LOG, and MICONIC, bound-sensitive h MS solves the largest number of problems, and this is also the case for bound-sensitive h max on FLOORTILE, PARC-PRINTER, and SOKOBAN.??? h MS often times out in the construction phase and before search has begun.

This occurs on average in approximately 300 problems per suite, or 1200 problems total.

This is especially pronounced in the TIDYBOT, TETRIS, and PIPESWORLD-NOTANKAGE domains.

This suggests a hybrid approach that combines the strengths of blind search and h MS : setting an upper bound on the time allotted to heuristic construction, and running blind search instead if construction does not terminate within this bound.

Using this configuration with a value of 10 minutes for the upper bound results in a planner that outperforms blind A * by +11, +16, +37, and +38 instances for the 25%, 50%, 75%, and 100% suites, respectively.

This makes h MS schemes that are less expensive to construct but maintain informativeness in this setting an appealing future subject of research.

We have shown that a previously introduced compilation to multiple cost function classical planning allows the A * algorithm to be used to solve oversubscription planning problems, and introduced a family of bound-sensitive heuristics that are much more informed than their classical counterparts in this setting.

Our experiments show that this approach results in a state-of-the-art method for some bound settings and domains.

One future research direction we would like to explore that builds on the methods introduced here is the use of non-admissible heuristics for satisficing OSP.

The method by which bound-sensitive h max is obtained is fairly general and should be equally applicable for h add or general relaxed plan heuristics BID3 .

A second direction is the use of these heuristics in other planning settings in which tradeoffs must be made between different cost functions, e.g. minimizing fuel use in the presence of bounds on time or vice versa in logistics problems.

Finally, our methods may be applicable to numeric planning problems in which the variables describe resources that are strictly decreasing and can be expressed in terms of secondary cost functions and associated bounds.

Boundsensitive heuristics could provide a principled way of reasoning about numeric variables in this context.

@highlight

We show that oversubscription planning tasks can be solved using A* and introduce novel bound-sensitive heuristics for oversubscription planning tasks.

@highlight

Presents an approach to solve oversubscription planning (OSP) tasks optimally by using a translation to classical planning with multiple cost functions.

@highlight

The paper proposes modifications to admissible heuristics to make them better informed in a multi-criteria setting where.