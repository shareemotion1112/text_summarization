Pattern databases are the foundation of some of the strongest admissible heuristics for optimal classical planning.

Experiments showed that the most informative way of combining information from multiple pattern databases is to use saturated cost partitioning.

Previous work selected patterns and computed saturated cost partitionings over the resulting pattern database heuristics in two separate steps.

We introduce a new method that uses saturated cost partitioning to select patterns and show that it outperforms all existing pattern selection algorithms.

A * search BID10 with an admissible heuristic BID23 ) is one of the most successful methods for solving classical planning tasks optimally.

An important building block of some of the strongest admissible heuristics are pattern database (PDB) heuristics.

A PDB heuristic precomputes all goal distances in a simplified state space obtained by projecting the task to a subset of state variables, the pattern, and uses these distances as lower bounds on the true goal distances.

PDB heuristics were originally introduced for solving the 15-puzzle BID2 and have later been generalized to many other combinatorial search tasks (e.g., BID21 BID7 and to the setting of domainindependent planning BID3 .Using a single PDB heuristic of reasonable size is usually not enough to cover sufficiently many aspects of challenging planning tasks.

It is therefore often beneficial to compute multiple PDB heuristics and to combine their estimates admissibly BID15 .

The simplest approach for this is to choose the PDB with the highest estimate in each state.

Instead of this maximization scheme, we would like to sum estimates, but this renders the resulting heuristic inadmissible in general.

However, if two PDBs are affected by disjoint sets of operators, they are independent and we can admissibly add their estimates BID19 BID7 .

BID11 later generalized this idea by introducing the canonical heuristic for PDBs, which computes all maximal subsets of pairwise independent PDBs and then uses the maximum over the sums of independent PDBs as the heuristic value.

Cost partitioning BID17 BID40 ) is a generalization of the independence-based methods above.

It makes the sum of heuristic estimates admissible by distributing the costs of each operator among the heuristics.

The literature contains many different cost partitioning algorithms such as zero-one cost partitioning BID4 BID11 ), uniform cost partitioning BID17 , optimal cost partitioning BID17 BID16 BID18 BID25 , posthoc optimization BID26 and delta cost partitioning BID6 .In previous work BID34 , we showed experimentally for the benchmark tasks from previous International Planning Competitions (IPC) that saturated cost partitioning (SCP) BID30 BID37 is the cost partitioning algorithm of choice for PDB heuristics.

Saturated cost partitioning considers an ordered sequence of heuristics.

Iteratively, it gives each heuristic the minimum amount of costs that the heuristic needs to justify all its estimates and then uses the remaining costs for subsequent heuristics until all heuristics have been served this way.

Before we can compute a saturated cost partitioning over pattern database heuristics, we need to select a collection of patterns.

The first domain-independent automated pattern selection algorithm is due to BID3 .

It partitions the state variables into patterns via best-fit bin packing.

BID5 later used a genetic algorithm to search for a pattern collection that maximizes the average heuristic value of a zero-one cost partitioning over the PDB heuristics.

BID11 proposed an algorithm that performs a hill-climbing search in the space of pattern collections (HC).

HC evaluates a collection C by estimating the search effort of the canonical heuristic over C based on a model of IDA * runtime BID20 .

BID8 presented the Complementary PDBs Creation (CPC) method, that combines bin packing and genetic algorithms to create a pattern collection minimizing the estimated search effort of an A * search BID22 .

BID28 repeatedly compute patterns using counterexample-guided abstraction refinement (CEGAR): starting from a random goal variable, their CEGAR algorithm iteratively finds solutions in the corresponding projection and executes them in the original state space.

Whenever a solution cannot be executed due to a violated precondition, it adds the missing precondition variable to the pattern.

Finally, BID26 systematically generate all interesting patterns up to a given size X (SYS-X).

Experiments showed that cost-partitioned heuristics over SYS-2 and SYS-3 yield accurate estimates BID26 BID34 , but using all interesting patterns of larger sizes is usually infeasible.

We introduce SYS-SCP, a new pattern selection algorithm based on saturated cost partitioning that potentially considers all interesting patterns, but only selects useful ones.

SYS-SCP builds multiple pattern sequences that together form the resulting pattern collection.

For each sequence σ, it considers the interesting patterns in increasing order by size and adds a pattern P to σ if P is not part of an earlier sequence and the saturated cost partitioning heuristic over σ plus P is more informative than the one over σ alone.

We consider optimal classical planning tasks in a SAS + -like notation BID1 and represent a planning task Π as a tuple V, O, s 0 , s .

Each variable v in the finite set of variables V has a finite domain dom(v).

A partial state s is defined over a subset of variables vars(s) ⊆ V and maps each v ∈ vars(s) to a value in dom(v), written as s [v] .

We call the pair v, s [v] an atom and interchangeably treat partial states as mappings from variables to values or as sets of atoms.

If vars(s) = V, we call s a state.

We write S(Π) for the set of all states in Π.Each operator o in the finite set of operators O has a precondition pre(o) and an effect eff(o), both of which are partial states, and a cost cost(o) ∈ R Transition systems assign semantics to planning tasks.

Definition 1 (Transition Systems).

A transition system T is a labeled digraph defined by a finite set of states S(T ), a finite set of labels L(T ), a set T (T ) of labeled transitions s − → s with s, s ∈ S(T ) and ∈ L(T ), an initial state s 0 (T ), and a set S (T ) of goal states.

A planning task Π = V, O, s 0 , s induces a transition system T with states S(Π), labels O, transitions {s DISPLAYFORM0 Separating transition systems from cost functions allows us to evaluate the same transition system under different cost functions, which is important for cost partitioning.

A cost function for transition system T is a function cost : L(T ) → R ∪ {−∞, ∞}. It is finite if −∞ < cost( ) < ∞ for all labels .

It is nonnegative if cost( ) ≥ 0 for all labels .

We write C(T ) for the set of all cost functions for T .Note that we assume that the cost function of the planning task is non-negative and finite, but as in previous work we allow negative BID25 and infinite costs BID32 in cost partitionings.

The generalization to infinite costs is necessary to cleanly state some of our definitions.

Definition 3 (Weighted Transition Systems).

A weighted transition system is a pair T , cost where T is a transition system and cost ∈ C(T ) is a cost function for T .The cost of a path π = s DISPLAYFORM0 It is ∞ if the sum contains both +∞ and −∞. If s n is a goal state, π is called a goal path for s 0 .Definition 4 (Goal Distances and Optimal Paths).

The goal distance of a state s ∈ S(T ) in a weighted transition system T , cost is defined as inf π∈Π (T ,s) cost(π), where Π (T , s) is the set of goal paths from s in T . (The infimum of the empty set is ∞.) We write h * DISPLAYFORM1 Optimal classical planning is the problem of finding an optimal goal path from s 0 or showing that s 0 is unsolvable.

We use heuristics to estimate goal distances BID23 .

DISPLAYFORM2 Cost partitioning makes adding heuristics admissible by distributing the costs of each operator among the heuristics.

Definition 6 (Cost Partitioning).

Let T be a transition system.

A cost partitioning for a cost function cost ∈ C(T )

is a tuple cost 1 , . . . , cost n ∈ C(T ) n whose sum is bounded by cost: DISPLAYFORM3 n over the heuristics h 1 , . . .

, h n for T induces the cost-partitioned heuristic DISPLAYFORM4 .

If the sum contains +∞ and −∞, it evaluates to the leftmost infinite value.

One of the cost partitioning algorithms from the literature is saturated cost partitioning BID31 .

It is based on the insight that we can often reduce the amount of costs given to a heuristic without changing any heuristic estimates.

Saturated cost functions formalize this idea.

Definition 7 (Saturated Cost Function).

Consider a transition system T , a heuristic h for T and a cost function cost ∈ C(T ).

A cost function scf ∈ C(T ) is saturated for h and cost if 1. scf( ) ≤ cost( ) for all labels ∈ L(T ) and 2.

h(scf, s) = h(cost, s) for all states s ∈ S(T ).A saturated cost function scf is minimal if there is no other saturated cost function scf for h and cost with scf( ) ≤ scf ( ) for all labels ∈ L(T ).Whether we can efficiently compute a minimal saturated cost function depends on the type of heuristic.

In earlier work BID31 , we showed that this is possible for explicitly-represented abstraction heuristics BID12 , which include PDB heuristics.

Definition 8 (Minimum Saturated Cost Function for Abstraction Heuristics).

Let T , cost be a weighted transition system and h an abstraction heuristic for T with abstract transition system T .

The minimum saturated cost function mscf for h and cost is DISPLAYFORM5 Given a sequence of abstraction heuristics, the saturated cost partitioning algorithm iteratively assigns to each heuristic only the costs that the heuristic needs to preserve its estimates and uses the remaining costs for subsequent heuristics.

Definition 9 (Saturated Cost Partitioning).

Consider a transition system T and a sequence of abstraction heuristics DISPLAYFORM6 receives a cost function rem and returns the minimum saturated cost function for h i and rem.

The saturated cost partitioning cost 1 , . . . , cost n of a function cost ∈ C(T ) over H is defined as: DISPLAYFORM7 where the auxiliary cost functions rem i represent the remaining costs after processing the first i heuristics in H. We write h SCP H for the saturated cost partitioning heuristic over the sequence of heuristics H. In this work, we compute saturated cost partitionings over pattern database heuristics.

A pattern for task Π with variables V is a subset P ⊆ V. By syntactically removing all variables from Π that are not in P , we obtain the projected task Π| P inducing the abstract transition system T P .

The PDB heuristic h P for a pattern P is defined as h P (cost, s) = h * T P (cost, s| P ), where s| P is the abstract state that s is projected to in Π| P .

For the pattern sequence P 1 , . . .

, P n we define h DISPLAYFORM8 One of the simplest pattern selection algorithms is to generate all patterns up to a given size X (Felner, Korf, and Hanan 2004) and we call this approach SYS-NAIVE-X. It is easy to see that for tasks with n variables, SYS-NAIVE-X generates X i=1 n i patterns.

Usually, many of these patterns do not add much information to a cost-partitioned heuristic over the patterns.

Unfortunately, there is no efficiently computable test that allows us to discard such uninformative patterns.

Even patterns without any goal variables can increase heuristic estimates in a cost partitioning BID27 .However, in the setting where only non-negative cost functions are allowed in cost partitionings, there are efficiently computable criteria for deciding whether a pattern Algorithm 1 SYS-SCP: Given a planning task with states S(T ), cost function cost and interesting patterns SYS, select a subset C ⊆ SYS.1: function SYS-SCP(Π) 2: DISPLAYFORM9 repeat for at most T x seconds 4: DISPLAYFORM10 for P ∈ ORDER(SYS) and at most T y seconds do 6:if P / ∈ C and PATTERNUSEFUL(σ, P ) then 7: DISPLAYFORM11 until σ = 10: DISPLAYFORM12 is interesting, i.e., whether it cannot be replaced by a set of smaller patterns that together yield the same heuristic estimates BID26 .

The criteria are based on the causal graph CG(Π) of a task Π BID13 .

CG(Π) is a directed graph with a node for each variable in Π. If there is an operator with a precondition on u and an effect on v = u, CG(Π) contains a precondition arc from u to v. If an operator affects both u and v, CG(Π) contains co-effect arcs from u to v and from v to u. Definition 10 (Interesting Patterns).

A pattern P is interesting if 1.

CG(Π| P ) is weakly connected, and 2.

CG(Π| P ) contains a directed path via precondition arcs from each node to some goal variable node.

The systematic pattern generation method SYS-X generates all interesting patterns up to size X. We let SYS denote the set of all interesting patterns for a given task.

On IPC benchmark tasks, SYS-X often generates much fewer patterns than SYS-NAIVE-X for the same size limit X. Still, it is usually infeasible to compute all SYS-X patterns and the corresponding projections for X > 3 within reasonable amounts of time and memory.

Also, we hypothesize that even when considering only interesting patterns, usually only a small percentage of the systematic patterns up to size 3 contribute much information to the resulting heuristic.

For these two reasons we propose a new pattern selection algorithm that potentially considers all interesting patterns, but only selects the ones that it deems useful.

Our new pattern selection algorithm repeatedly creates a new empty pattern sequence σ and only appends those interesting patterns to σ that increase any finite heuristic values of a saturated cost partitioning heuristic computed over σ.

Algorithm 1 shows pseudo-code for the procedure, which we call SYS-SCP.

It starts with an empty pattern collection C. In each iteration of the outer loop, SYS-SCP creates a new empty pattern sequence σ and then loops over the interesting patterns P ∈ SYS in the order chosen by ORDER (see Section 3.2) for at most T y seconds.

SYS-SCP appends a pattern P to σ and includes it in C if there is a state s for which the saturated cost partitioning over σ extended by P has a higher finite heuristic value than the one over σ alone.

Once an iteration selects no new patterns or SYS-SCP hits the time limit T x , the algorithm stops and returns C.We impose a time limit T x on the outer loop of the algorithm since the number of interesting patterns is exponential in the number of variables and therefore SYS-SCP usually cannot evaluate them all in a reasonable amount of time.

By imposing a time limit T y on the inner loop, we allow SYS-SCP to periodically start over with a new empty pattern sequence.

The most important component of the SYS-SCP algorithm is the PATTERNUSEFUL function that decides whether to select a pattern P .

The function enumerates all states s ∈ S(Π), which is obviously infeasible for all but the smallest tasks Π. Fortunately, we can efficiently compute an equivalent test in the projection to P .

Lemma 1.

Consider a planning task Π with non-negative cost function cost and induced transition system T .

Let s ∈ S(T ) be a state, P be a pattern for Π and σ be a (possibly empty) sequence of patterns P 1 , . . .

, P n for Π. Finally, let rem be the remaining cost function after computing h DISPLAYFORM0 ⇔ 0 < h * T P (rem, s| P ) < ∞ Step 1 substitutes P 1 , . . .

, P n for σ and Step 2 uses the definition of saturated cost partitioning heuristics.

For Step 3 we need to show that x = n i=1 h Pi (cost i , s) is finite.

The inequality states x < ∞. We now show x ≥ 0, which implies x > −∞. Using requirement 1 for saturated cost functions from Definition 7 and the fact that rem 0 = cost is non-negative, it is easy to see that all remaining cost functions are non-negative.

Consequently, h Pi (cost i , s) = h Pi (rem i−1 , s) ≥ 0 for all s ∈ S(T ), which uses requirement 2 from Definition 7 and the fact that goal distances are non-negative in transition systems with non-negative weights.

Step 4 uses the definition of PDB heuristics.

Consider a planning task Π with non-negative cost function cost and induced transition system T .

Let P be a single pattern and σ be a (possibly empty) sequence of patterns.

Finally, let rem be the remaining cost function after computing DISPLAYFORM0 Follows directly from Lemma 1 and the fact that projections are induced abstractions: for each abstract state s in an induced abstraction there is at least one concrete state s which is projected to s .We use Theorem 1 in our SYS-SCP implementation by keeping track of the cost function rem, i.e., the costs that remain after computing h SCP σ .

We select a pattern P if there are any goal distances d with 0 < d < ∞ in T P under rem.

Theorem 1 also removes the need to compute h SCP σ⊕P from scratch for every pattern P .

This is important since we want to decide whether or not to add P quickly and this operation should not become slower when σ contains more patterns.

To obtain high finite heuristic values for solvable states it is important to choose good cost partitionings.

In contrast, cost functions are irrelevant for detecting unsolvable states.

This is the underlying reason why Lemma 1 only holds for finite values and therefore why SYS-SCP ignores unsolvable states.

However, we can still use the information about unsolvable states contained in projections.

It is easy to see that each abstract state in a projection corresponds to a partial state in the original task.

If an abstract state is unsolvable in a projection, we call the corresponding partial state a dead end.

Since projections preserve all paths, any state in the original task subsuming a dead end is unsolvable.

We can extract all dead ends from the projections that SYS-SCP evaluates and use this information to prune unsolvable states during the A * search BID24 .

We showed in earlier work that the order in which saturated cost partitioning considers the component heuristics has a strong influence on the quality of the resulting heuristic BID35 .

Choosing a good order is even more important for SYS-SCP, since it usually only sees a subset of interesting patterns within the allotted time.

To ensure that this subset of interesting patterns covers different aspects of the planning task, we let the ORDER function generate the interesting patterns in increasing order by size.

This leaves the question how to sort patterns of the same size.

We propose four methods for making this decision.

The first one (random) simply orders patterns of the same size randomly.

The remaining three assign a key to each pattern, allowing us to sort by key in increasing or decreasing order.

Causal Graph.

The first ordering method is based on the insight that it is often more important to have accurate heuristic estimates near the goal states rather than elsewhere in the state space (e.g., BID15 BID39 .

We therefore want to focus on patterns containing goal variables or variables that are closely connected to goal variables.

To quantify "goalconnectedness" we use an approximate topological ordering ≺ of the causal graph CG(Π).

We let the function cg : V → N + 0 assign each variable v ∈ V to its index in ≺. For a given pattern P , the cg ordering method returns the key cg(v 1 ), . . .

, cg(v n ) , where v i ∈ P and cg(v i ) < cg(v j ) for all 1 ≤ i < j ≤ n. Since the keys are unique, they define a total order.

Sorting the patterns by cg in decreasing order (cg-down), yields the desired order which starts with "goal-connected" patterns.

States in Projection.

Given a pattern P , the ordering method states returns the key |S(Π| P )|, i.e., the number of states in the projection to P .

We use cg-down to break ties.

Active Operators.

Given a pattern P , the ops ordering method returns the number of operators that affect a variable in P .

We break ties with cg-down.

We implemented the SYS-SCP pattern selection algorithm in the Fast Downward planning system BID14 and conducted experiments with the Downward Lab toolkit on Intel Xeon Silver 4114 processors.

Our benchmark set consists of all 1827 tasks without conditional effects from the optimization tracks of the 1998-2018 IPCs.

The tasks belong to 48 different domains.

We limit time by 30 minutes and memory by 3.5 GiB. All benchmarks 1 , code 2 and experimental data 3 have been published online.

To fairly compare the quality of different pattern collections, we use the same cost partitioning algorithm for all collections.

Saturated cost partitioning is the obvious choice for the evaluation since experiments showed that it is preferable to all other cost partitioning algorithms for HC, SYS-2 and CPC patterns in almost all evaluated benchmark domains BID34 BID28 .Diverse Saturated Cost Partitioning Heuristics.

For a given pattern collection C, we compute diverse saturated cost partitioning heuristics using the diversification procedure by BID35 : we start with an empty family of saturated cost partitioning heuristics F and a setŜ of 1000 sample states obtained with random walks ).

Then we iteratively sample a new state s and compute a greedy order ω of C that works well for s BID36 .

If h SCP ω has a higher heuristic estimate for any state s ∈Ŝ than all heuristics in F, we add h SCP ω to F. We stop this diversification procedure after 200 seconds and then perform an A * search using the maximum over the heuristics in F. TAB1 : Number of tasks solved by SYS-SCP using different time limits T x and T y for the outer loop (x axis) and inner loop (y axis).cg-up states-up random ops-down states-down ops-up cg-down Coverage cg-up -5 6 5 4 3 3 1140.0 states-up 6 -6 8 5 2 2 1153.0 random 10 10 -8 7 6 3 1148.2 ops-down 7 8 9 -4 7 3 1141.0 states-down 9 8 9 7 -4 2 1152.0 ops-up 11 12 12 11 11 -6 1166.0 cg-down 12 10 12 10 9 6 -1168.0 TAB2 : Per-domain coverage comparison of different orders for patterns of the same size.

The entry in row r and column c shows the number of domains in which order r solves more tasks than order c. For each order pair we highlight the maximum of the entries (r, c) and (c, r) in bold.

Right: Total number of solved tasks.

The results for random are averaged over 10 runs (standard deviation: 3.36).Before we compare SYS-SCP to other pattern selection algorithms, we evaluate the effects of changing its parameters in four ablation studies.

We use at most 2M states per PDB and 20M states in the PDB collection for all SYS-SCP runs.

TAB1 shows that a time limit for the outer loop is more important than one for the inner loop, but for maximum coverage we need both limits.

The combination that solves the highest number of tasks is 10s for the inner and 100s for the outer loop.

We use these values in all other experiments.

All configurations from

Instead of discarding the computed pattern sequences when SYS-SCP finishes, we can turn each pattern sequence σ into a full pattern order by randomly appending all SYS-SCP patterns missing from σ to σ and pass the resulting order to the diversification procedure.

Feeding the diversification exclusively with such orders leads to solving 1130 tasks, while using only greedy orders for sample states BID36 ) solves 1156 tasks.

We obtain the best results by diversifying both types of orders, solving 1168 tasks, and we use this variant in all other experiments.

In the next experiment, we evaluate the obvious baseline for SYS-SCP: selecting all (interesting) patterns up to a fixed size.

TAB3 holds coverage results of SYS-NAIVE-X and SYS-X for 1 ≤ X ≤ 5.

We also include variants (*-LIM) that use at most 100 seconds, no more than 2M states in each projection and at most 20M states per collection.

For the *-LIM variants, we sort the patterns in the cg-down order.

The results show that interesting patterns are always preferable to naive patterns, both with and without limits, which is why we only consider interesting patterns in SYS-SCP.

Imposing limits is not important for SYS-1 and SYS-2, but leads to solving many more tasks for X ≥ 3.

Overall, SYS-3-LIM has the highest total coverage (1088 tasks).

In Table 4 we compare SYS-SCP to the strongest pattern selection algorithms from the literature: HC, SYS-3-LIM, CPC and CEGAR.

(See Table 6 for per-domain coverage results.)

We run each algorithm with its preferred parameter values, which implies using at most 900s for HC and CPC and 100s for the other algorithms.

HC is outperformed by all other algorithms.

Interestingly, already the simple SYS-3-LIM approach is competitive with Table 4 : Per-domain coverage comparison of pattern selection algorithms.

For an explanation of the data see the caption of TAB2 .CPC and CEGAR.

However, we obtain the best results with SYS-SCP.

It is preferable to all other pattern selection algorithms in per-domain comparisons: no algorithm has higher coverage than SYS-SCP in more than three domains, while SYS-SCP solves more tasks than each of the other algorithms in at least 21 domains.

SYS-SCP also has the highest total coverage of 1168 tasks, solving 70 more tasks than the strongest contender.

This is a considerable improvement in the setting of optimal classical planning, where task difficulty tends to scale exponentially.

In our final experiment, we evaluate whether Scorpion BID37 , one of the strongest optimal planners in IPC 2018, benefits from using SYS-SCP patterns.

Scorpion computes diverse saturated cost partitioning heuristics over HC and SYS-2 PDB heuristics and Cartesian abstraction heuristics (CART) BID31 .

We abbreviate this combination with COMB=HC+SYS-2+CART.

In TAB6 we compare the original Scorpion planner, three Scorpion variants that use different sets of heuristics and the top three optimal planners from IPC 2018, Delfi 1 (Sievers et al. 2019), Complementary 1 BID9 and Complementary 2 BID8 ).

(Table 6 holds perdomain coverage results.)

In contrast to the configurations we evaluated above, all planners in TAB6 prune irrelevant operators in a preprocessing step BID0 .The results show that all Scorpion variants outperform the top three IPC 2018 planners in per-domain comparisons.

We also see that Scorpion benefits from using SYS-SCP PDBs instead of the COMB heuristics in many domains.

Using the union of both sets is clearly preferable to using either COMB or SYS-SCP alone, since it raises the total coverage to 1261 by 56 and 44 tasks, respectively.

For maximum coverage (1265 tasks), Scorpion only needs SYS-SCP PDBs and Cartesian abstraction heuristics.

We introduced a new pattern selection algorithm based on saturated cost partitioning and showed that it outperforms Table 6 : Number of tasks solved by different planners.

@highlight

Using saturated cost partitioning to select patterns is preferable to all existing pattern selection algorithms.