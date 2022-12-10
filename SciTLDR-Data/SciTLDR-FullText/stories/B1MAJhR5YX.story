One form of characterizing the expressiveness of a piecewise linear neural network is by the number of linear regions, or pieces, of the function modeled.

We have observed substantial progress in this topic through lower and upper bounds on the maximum number of linear regions and a counting procedure.

However, these bounds only account for the dimensions of the network and the exact counting may take a prohibitive amount of time, therefore making it infeasible to benchmark the expressiveness of networks.

In this work, we approximate the number of linear regions of specific rectifier networks with an algorithm for probabilistic lower bounds of mixed-integer linear sets.

In addition, we present a tighter upper bound that leverages network coefficients.

We test both on trained networks.

The algorithm for probabilistic lower bounds is several orders of magnitude faster than exact counting and the values reach similar orders of magnitude, hence making our approach a viable method to compare the expressiveness of such networks.

The refined upper bound is particularly stronger on networks with narrow layers.

Neural networks with piecewise linear activations have become increasingly more common along the past decade, in particular since BID40 and BID25 .

The simplest and most commonly used among such forms of activation is the Rectifier Linear Unit (ReLU), which outputs the maximum between 0 and its input argument BID30 BID35 .

In the functions modeled by these networks, we can associate each part of the domain in which the network corresponds to an affine function with a particular set of units having positive outputs.

We say that those are the active units for that part of the domain.

Counting these "pieces" into which the domain is split, which are often denoted as linear regions or decision regions, is one way to compare the expressiveness of models defined by networks with different configurations or coefficients.

The theoretical analysis of the number of input regions in deep learning dates back to at least BID8 , and more recently BID45 have shown empirical evidence that the accuracy of similar rectifier networks can be associated with the number of such regions.

From the study of how many linear regions can be defined on such a rectifier network with n ReLUs, we already know that not all configurations -and in some cases none -can reach the ceiling of 2 n regions.

We have learned that the number of regions may depend on the dimension of the input as well as on the number of layers and how the units are distributed among these layers.

On the one hand, it is possible to obtain neural networks where the number of regions is exponential on network depth .

On the other hand, there is a bottleneck effect by which the width of each layer affects how the regions are partitioned by subsequent layers due to the dimension of the space containing the image of the function, up to the point that shallow networks define the largest number of linear regions if the input dimension exceeds n BID45 .The literature on this topic has mainly focused on bounding the maximum number of linear regions.

Lower bounds are obtained by constructing networks defining increasingly larger number of linear regions BID2 BID45 .

Upper bounds are proven using the theory of hyperplane arrangements by BID54 along with other analytical insights BID44 BID38 BID45 .

These bounds are only identical -and thus tight -in the case of one-dimensional inputs BID45 .

Both of these lines have explored deepening connections with polyhedral theory, but some of these results have also been recently revisited using tropical algebra BID55 BID13 .

In addition, BID45 have shown that the linear regions of a trained network correspond to a set of projected solutions of a Mixed-Integer Linear Program (MILP).Other methods to study neural network expressiveness include universal approximation theory BID16 , VC dimension BID5 , and trajectory length BID44 .

Different networks can be compared by transforming one network to another with different number of layers or activation functions.

For example, it has been shown that any continuous function can be modeled using a single hidden layer of sigmoid activation functions BID16 .

In the context of ReLUs, BID36 have shown that the popular ResNet architecture BID31 with a single ReLU neuron in every hidden layer can be a universal approximator.

Furthermore, BID2 have shown that a network with single hidden layer of ReLUs can be trained for global optimality with a runtime polynomial in the data size, but exponential in the input dimension.

The use of trajectory length for expressiveness is related to linear regions, i.e., by changing the input along a one dimensional path we study the transition in the linear regions.

Certain critical network architectures using leaky ReLUs (f (x) = max(x, αx), α ∈ (0, 1)) are identified to produce connected decision regions BID42 .

In order to avoid such degenerate cases, we need to use sufficiently wide hidden layers.

However, this result is mainly applicable for leaky ReLUs and not for the standard ReLUs BID7 .Although the number of linear regions has been long conjectured and recently shown to work for comparing similar networks, this metric would only be used in practice if we come up with faster methods to count or reasonably approximate such number.

Our approach in this paper consists of introducing empirical upper and lower bounds, both of which based on the weight and bias coefficients of the networks, and thus able to compare networks having the same configuration of layers.

In particular, we reframe the problem of determining the potential number of linear regions N of an architecture with that of estimating the representation efficiency η = log 2 N of a network, which can be interpreted as the minimum number of units to define as many linear regions, thereby providing a more practical and interpretable metric for expressiveness.

We present the following contributions:(i) We adapt approximate model counting methods for propositional satisfiability (SAT) to obtain probabilistic bounds on the number of solutions of MILP formulations, which we use to count regions.

Interestingly, these methods are particularly simpler and faster when restricted to lower bounds on the order of magnitude.

See results in FIG2 and algorithm in Section 5. (ii) We refine the best known upper bound by considering the coefficients of the trained network.

With such information, we identify that unit activity further contributes to the bottleneck effect caused by narrow layers BID45 .

Furthermore, we are able to compare networks with the same configuration of layers.

See results in Table 1 and theory in Section 4. (iii) We also survey and contribute to the literature on MILP formulations of rectifier networks due to the impact of the formulation on obtaining better empirical bounds.

See Section 3.

In this paper, we consider feedforward Deep Neural Networks (DNNs) with Rectifier Linear Unit (ReLU) activations.

Each network has n 0 input variables given by DISPLAYFORM0 T with a bounded domain X and m output variables given by y = [y 1 y 2 . . .

y m ] T .

Each hidden layer l = {1, 2, . . .

, L} has n l hidden neurons with outputs given by DISPLAYFORM1 For notation simplicity, we may use h 0 for x and h L+1 for y. Let W l be the n l × n l−1 matrix where each row corresponds to the weights of a neuron of layer l. Let b l be the bias vector used to obtain the activation functions of neurons in layer l. The output of unit i in layer l consists of an affine transformation g DISPLAYFORM2 to which we apply the ReLU activation h l i = max{0, g l i }.

We may regard the DNN as a piecewise linear function F : R n0 → R m that maps the input x ∈ X ⊂ R n0 to y ∈ R m .

Hence, the domain is partitioned into regions within which F corresponds to an affine function, which we denote as linear regions.

Following the same convention as BID44 ; BID38 ; BID45 , we characterize each linear region by the set of units that are active in that domain.

For each layer l, let S l ⊆ {1, . . .

, n l } be the activation set in which i ∈ S l if and only if h l i > 0.

Let S = (S 1 , . . .

, S l ) be the activation pattern aggregating those activation sets.

Consequently, the number of linear regions defined by the DNN corresponds to the number of nonempty sets in x defined by all possible activation patterns.

DISPLAYFORM3 The solutions of this projection can be enumerated using the one-tree algorithm BID17 , in which the branch-and-bound tree used to obtain the optimal solution is further expanded to collect near-optimal solutions up to a given limit.

In general, finding a feasible solution to a MILP is NPcomplete (Cook, 1971 ) and thus optimization is NP-hard.

However, BID24 note that a feasible solution can always be obtained by evaluating any valid input.

While that does not directly imply that optimization problems on DNNs are easy, it hints at the possibility of good properties.

Several MILP formulations with an equivalent feasible set have been used in the context of network verification to determine the image of the function modeled BID37 BID18 and evaluate adversarial perturbations in the domain X BID14 BID24 BID48 BID52 .

There are also similar applications relaxing the binary variables as continuous variables in the domain [0, 1] or using the linear formulation of a particular linear region BID6 BID19 BID51 , which can be simply defined using W l i h l−1 + b l i ≥ 0 for active units and the complement for inactive units.

Although equivalent, some authors have explored how these formulations may differ in strength BID24 BID48 BID32 .

When the binary variables are relaxed as continuous variables in the domain [0, 1], we obtain a linear relaxation that may differ across formulations.

We say that an MILP formulation A is stronger than another formulation B if, when projected on a common set of variables, the linear relaxation of A is a proper subset of the linear relaxation of B. Formulation strength is commonly regarded as a proxy for MILP solver performance.

Differences in strength may be due to changes in constants such as H l i andH l i , use of additional valid inequalities that remove fractional solutions, or even additional variables defining an extended formulation.

For mapping DNNs, we can discuss strength in at least three levels of scope.

(2) - (7) defines the convex outer approximation on (g DISPLAYFORM4 Lemma 1 evidences that, for formulations like the one above, constants for both the maximum and the minimum values of h l i are necessary to obtain a strong formulation.

The proof can be found in Appendix A. We note that a similar claim without proof is made by BID32 .

DISPLAYFORM5 and (c) If P maps a vertex x P of the input, by convexifying the layer the rest of PQ is infeasible for x P .When the domain X of the network is defined by a box, in which case the domain of each input variable x i is an independent continuous interval, then the smallest possible values for DISPLAYFORM6 can be computed with interval arithmetic by taking element-wise maxima BID14 BID45 .

When extended to subsequent layers, however, this approach is prone to overestimate the values for H l i andH l i because the output of subsequent layers is not necessarily a box in which the maximum value of each input are independent from each other.

More generally, if X is polyhedral, BID24 and BID48 show that we can obtain the smallest values for these constants by solving a sequence of MILPs on layers l = 1, . . .

, L of the form DISPLAYFORM7 and replacing (equation 12) with max −g l i to computeH l i .

In large rectifier networks, BID48 found that many units are always active becauseH DISPLAYFORM8 At the very least, we can use 0 on either case.

In the former case, which they denote as stably active, we can simply replace constraints (1)- (7) with h l i = g l i .

In the latter case, which they denote as stably inactive, we note that the unit can be removed from the formulation without any loss.

They denote as unstable the remaining units, which can be active or not depending on their inputs.

Second, we can consider the strength of the formulation to represent the mapping of h l−1 to h l on each layer.

BID32 argues that this additional strengthening may remove certain combinations of h l−1 and h l i that can never occur, and has shown that this can be done using an extended formulation following BID3 .

Figure 1 (c) describes one such example.

However, we observed a slower performance to count linear regions due to the larger number of variables.

BID32 has also shown that these variables can be projected out, with the resulting formulation having an exponential number of constraints on n l .

In the context of finding a single optimal solution, usually not requiring all of them, these constraints can be efficiently generated as needed.

Third, we can consider constraints strengthening the formulation across different layers.

For example, BID32 presents such a family of valid inequalities that resemble those obtained by projecting out the extra variables after convexifying each layer as described above.

We propose some valid inequalities involving consecutive layers of the network.

The first is inspired by how constants H l i andH l i can be bounded using interval arithmetic.

Depending on which units are active in the previous layer, the output of a given unit may be further restricted as follows: DISPLAYFORM0 The max term is necessary in case b l i is negative, since none of the units on the summation term being negative merely implies that the unit itself is inactive instead of rendering the system infeasible.

Following the same logic, we may actually define inequalities on the binary variables alone, which may be preferable since large constants create numerical difficulties and deteriorate solver performance.

For the unit to be active when b l i ≤ 0, there must be a positive contribution from the previous layer, and thus some unit j in layer l − 1 such that W l ij > 0 should be also active: DISPLAYFORM1 Similarly, unit i is only inactive when b DISPLAYFORM2 Let us denote unstable units in which b l i ≤ 0, and thus (16) applies, as inactive leaning; and those in which b l i > 0, and thus (17) applies, as active leaning.

Within linear regions where none among the units of the previous layer in the corresponding inequalities is active, these units can be regarded as stably inactive and stably active, respectively.

We will use that to obtain better bounds in Section 4.

We prove a tighter -and empirical -upper bound in this section.

This bound is obtained by taking into accounnt which units are stably active and stably inactive on the input domain X and also how many among the unstable units are locally stable in some of the linear regions.

Prior to discussing this bound in Section 4.2 and how to compute its parameters in Section 4.3, we discuss in Section 4.1 other factors that have been found to affect such bounds in prior work on this topic.

The two main building blocks to bound the number of linear regions are activation hyperplanes and the theory of hyperplane arrangements.

We explain their use in prior work in this Section.

For each unit i in layer l, the activation hyperplane W into the regions where the unit is active (W DISPLAYFORM0 .

In order to bound the number of regions defined by multiple hyperplanes on the same space, we use a result from BID54 that n l hyperplanes in an n l−1 -dimensional space define at most n l−1 j=0 n l j regions.

However, if the normal vectors of these hyperplanes span a smaller space, then the same number of regions can be defined in less dimensions.

In particular, BID45 shows that we can actually assume a maximum of DISPLAYFORM1 We can obtain a bound for deep networks by recursively combining the bounds obtained on each layer.

By assuming that every linear region defined by the first l − 1 layers is then subdivided into the maximum possible number of linear regions defined by the activation hyperplanes of layer l, we obtain the implicit bound of L l=1 n l−1 j=0 n l j from BID44 .

By observing that the dimension of the input of layer l on each linear region is also constrained by the smallest input dimension among layers 1 to l − 1, we can obtain the bound in BID38 DISPLAYFORM2 n l j , where d l = min{n 0 , n 1 , . . .

, n l }.

If we refine the effect on the input dimension by also considering that the number of units that are active on each layer varies across the linear regions, we can obtain the tighter bound in BID45 DISPLAYFORM3

Now we show that we can further improve on the sequence of bounds previously found in the literature by leveraging the local and global stability of units of a trained network, which can be particularly useful to compare networks having the same configuration of layers.

First, note that only units that can be active in a given linear region produced by layers 1 to l − 1 affect the dimension of the space in which the linear region can be further partitioned by layers l to L. Second, only the subset of these units that can also be inactive within that region, i.e., the unstable ones, counts toward the number of hyperplanes partitioning the linear region at layer l. Hence, let A l (k) be the maximum number of units that can be active in layer l if k units are active in layer l − 1; and I l (k) be the corresponding maximum number of units that are unstable, hence potentially defining hyperplanes that intersect the interior of the linear region.

Note that every linear region is contained in one side of the hyperplane defined by each stable unit.

We state our main result below and discuss how to compute A l (k) and I l (k) using W l and b l next.

Theorem 2 improves the result by BID45 when not all hyperplanes partition every linear region from previous layers (I l (k l−1 ) < n l ) or not all units can be active (smaller intervals for j l ): Theorem 2.

Consider a deep rectifier network with L layers with input dimension n 0 and at most A l (k) active units and I l (k) unstable units in layer l for every linear region defined by layers 1 to l − 1 when k units are active in layer l − 1.

Then the maximum number of linear regions is at most DISPLAYFORM0 Proof.

In resemblance to BID45 , we define a recurrence to recursively bound the number of subregions within a region.

Let R(l, k, d) be an upper bound to the maximal number of regions attainable from partitioning a region with dimension at most d among those defined by layers 1 to l − 1 in which at most k units are active in layer l − 1 by using the remaining layers l to L. For the base case DISPLAYFORM1 The recurrence groups regions with same number of active units in layer l as R(l, k, d) = DISPLAYFORM2 ,j represents the maximum number of regions with j active units in layer l from partitioning a space of dimension d using p hyperplanes.

We also use the observation in BID45 that there are at most DISPLAYFORM3 regions defined by layer l when j unstable units are active and there are k active units in layer l − 1, which can be regarded as the subsets of I l (k) units of size j. Since layer l defines at most DISPLAYFORM4 regions with an input dimension d and k active units above, by allowing the largest number of active hyperplanes among the unstable units and also using DISPLAYFORM5 Without loss of generality, we assume that the input is generated by n 0 active units feeding the network, hence implying that the bound can be evaluated as R(1, n 0 , n 0 ): DISPLAYFORM6 We obtain the final expression by nesting the values of j 1 , . . .

, j L .

Finally, we discuss how the parameters introduced with the empirical bound in Section 4.2 can be computed exactly, or else approximated.

We first bound the value of I l (k).

Let U − l and U + l denote the sets of inactive leaning and active leaning units in layer l, and DISPLAYFORM0 , we can define a set J − (l, i) of units from layer l − 1 that, if active, can potentially make i active.

In fact, we can define the set in the summation of inequality FORMULA10 , and therefore let J − (l, i) := {j : 1 ≤ j ≤ n l−1 , W l ij > 0}. For a given unit i ∈ U + l , we can similarly use the set in inequality FORMULA11 , and let DISPLAYFORM1 , j ∈ J − (l + 1, i)} be the set of units in layer l + 1 that may be locally unstable if unit j in layer l is active.

DISPLAYFORM2 In other words, we look for the subsets of at most k units in layer l − 1 that together may affect the stability of the largest number of units in layer l. Nonetheless, we may only need to inspect a small number of such subsets in practice.

Assuming that each row of W l and vector b l have about the same number of positive and negative elements, then we can expect that each set I(l − 1, j) contains half of the units in U l .

If these positive and negative elements are distributed randomly for each unit, then a logarithmic number of the units in layer l − 1 being active may suffice to entirely cover U l .

Hence, we can reasonably expect to evaluate a linear number of subsets of n l−1 on average.

In cases where this assumption does not hold, we discuss later how to approximate I l (k).Next we bound the value of A l (k).

In this case, we consider a larger subset of the units in l that only excludes locally inactive units.

Let n + l denote the number of stably active units in layer l, which is such that n + l ≤ n l − |U l |, and let I − (l, j) := {i : i ∈ U − l+1 , j ∈ J − (l + 1, i)} be the set of inactive leaning units in layer l + 1 that may become active when unit j in layer l is active.

DISPLAYFORM3 We can approximate I l (k) and A l (k) with strong optimality guarantees (1 − 1 e ) using simple greedy algorithms for submodular function maximization BID41 .

See Appendix E.

We can think of SAT as a particular form of encoding solutions on a set V of Boolean variables, where the solutions have to satisfy a set of predicates, and which can therefore represent solutions on binary variables of an MILP.

BID49 has shown that counting solutions of SAT formulas is #P-complete.

However, thanks to the improving performance of SAT solvers, many practical approaches to approximate the number of solutions have been proposed since BID26 , all of which making a relatively small number of solver calls to solve restricted formulas.

The idea in this line of work is to use hash functions with good statistical properties to partition the set of solutions S into subsets having approximately half of the solutions each.

After restricting a given formula to one of such subsets r times, we may intuitively assume that, with some probability, |S| ≥ 2 r if the resulting subset is more often feasible or else |S| < 2 r .

Most of the literature has restricted SAT formulas with predicates that encode XOR constraints, which can be interpreted in terms of 0-1 variables as restricting the sum of a subset U of the variables to be even or odd.

Probabilistic lower bounds can be obtained using XOR constraints on fixed or variable sizes of subset k = |U |.

Although they get better as k increases, even small values of k yield good approximations in practice BID29 ).

Since we are mainly interested in the order of magnitude, we focus on extending the classic MBound algorithm BID26 .

We opt for a fixed -and also small -size k to avoid scalability issues as the number of ReLUs increase.

We refer the reader to Appendix C for a survey on XOR constraints and approximate model counting.

The key difference when devising an algorithm for MILP is that these solvers are not used in the same way as SAT solvers.

The assumption in SAT-based approaches is that each restricted formula entails a new call to the solver.

BID12 improves to a logarithmic number of calls by orderly applying the same sequence constraints up to each value of r, and then applying binary search to find the smallest r that makes the formula unsatisfiable.

In MILP solvers, we can test for all values of r with a single call to the solver by generating parity constraints as lazy cuts, which can be implemented through callbacks.

When a new solution is found, a callback is invoked to generate parity constraints.

Each constraint may or may not remove the solution just found, since we preserve the independence between the solutions found and the constraints generated, and thus we may need to generate multiple parity constraints before yielding the control back to the solver.

Algorithm 1, which we denote MIPBound, illustrates the idea.

We refer the reader to Appendix C for details on how to translate parity constraints to MILP and Appendix D for how the probabilities are derived.

We test on the instances used in BID45 to benchmark against exact counting.

The results are reported in FIG2 and Table 1 .

We adapt Algorithm 1 to count linear regions by ignoring solutions with value 0.

For each size of parity constraints k, which we denote as XOR-k, we measure the time to find the smallest coefficients H l i andH l i for each unit along with the subsequent time of Algorithm 1.

We let Algorithm 1 run for enough steps to obtain a probability of 99.5% in case all tested restrictions of a given size preserve the formulation feasible, and we report the largest lower bound with probability at least 95%.

We define a DNN with η < 12 as small and large otherwise to illustrate how these points are distributed with respect to the identity line, since counting is faster than sampling for smaller sets.

The upper bound from Theorem 2, which we denote as Empirical Upper Bound (Empirical UB), is computed at a fraction of the time to obtain the constants.

We use Configuration Upper Bound (Configuration UB) for the bound in BID45 .

The code is written in C++ (gcc 4.8.4) using CPLEX Studio 12.8 as a solver and ran in Ubuntu 14.04.4 on a machine with 40 Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz processors and 132 GB of RAM.

Table 1 : Gap between configuration UB and actual values that is closed (%) by empirical UB.Following up on the discussion from Section 4.3 about computing the values of A l (k) and I l (k), we report in Table 2 of Appendix F the minimum value of k to find the maximum value of both expressions for layers 2 and 3 of the trained networks.

We observe in practice that such values of k remain small and so does the number of subsets of units from layer l − 1 that we need to inspect.

This paper introduced methods to obtain upper and lower bounds on a rectifier network.

The upper bound refines the best known result for the network configuration by taking into account the coefficients of the network.

By analyzing how the network coefficients affect when each unit can be active, we break the commonly used theoretical assumption that the activation hyperplane of each unit intersects every linear region defined by the previous layers.

The resulting bound is particularly stronger when the network has a narrow layer, hence evidencing that the bottleneck effected identified by BID45 can be even stronger in those cases.

The lower bound is based on extending an approximate model counting algorithm of SAT formulas to MILP formulations, which can then be used on MILP formulations of rectifier networks.

The resulting algorithm is orders of magnitude faster than exact counting on networks with a large number of linear regions.

The probabilistic bounds obtained can be parameterized for a balance between precision and speed, but it is interesting to observe that the the bounds obtained for different networks preserve a certain ordering in their sizes as we make the estimate more precise.

Hence, we have some indication that faster approximations could suffice if we just want to compare networks for their relative expressiveness.

Algorithm 1 Computes probabilistic lower bounds on the number of distinct solutions on n binary variables of a formulation F using parity constraints of size k DISPLAYFORM0 end for 6:while Termination criterion not satisfied do 7:F ← F Start over with F as formulation F 8: DISPLAYFORM1 Number of times that we have made F infeasible 9:r ← 0 Number of parity constraints added this time 10:while F has some solution s do 11:repeat 12:Generate parity constraint C of size k among n variables 13: DISPLAYFORM2 r ← r + 1 15: until C removes s This loop is implemented as a lazy cut callback 16: end while 17: DISPLAYFORM3 Number of times that F is feasible after adding j constraints 19:end for 20:end while 21:for j ← 0 → n − 1 do Computes probabilities after last call to the solver 22: BID46 and BID47 used these functions to show that approximate counting can be done in polynomial time with an NP-oracle, whereas BID50 have shown that SAT formulas with unique solution are as hard as those with multiple solutions.

Hence, from a theoretical standpoint, such approximations are not much harder than solving for a single solution.

DISPLAYFORM4 The seminal work by BID26 introduced the MBound algorithm, where XOR constraints on sets of variables with a fixed size k are used to compute the probability that 2 r is either a lower or an upper bound.

These probabilistic lower bounds are always valid but get better as k increases, whereas the probabilistic upper bound is only valid if k = |V |/2.

However, BID29 have shown that these lower bounds can be very good in practice for small values of k. The same principles have also been applied to constraint satisfaction problems BID28 .With time, this topic has gradually shifted to more precise estimates and to reducing the value of k needed to obtain valid upper bounds.

Some of the subsequent work has been influenced by uniform sampling results from BID27 , where the fixed size k is replaced with an independent probability p of including each variable in each XOR constraint.

That work includes the ApproxMC and the WISH algorithms BID10 BID21 , which rely on finding more solutions of the restricted formulas but generate (σ, ) certificates by which, with probability 1 − σ, the result is within (1 ± )|S|.

The following work by BID22 and BID56 aimed at providing upper bound guarantees when p < 1/2, showing that the size of those sets can be Θ log(|V |) .

Other groups tackled this issue differently.

BID11 and BID33 have limited the counting to any set of variables I for which any assignment leads to at most one solution in V , denoting those as minimal independent supports.

BID0 and BID1 have broken with the independent probability p by using each variable the same number of times across the r XOR constraints.

Similarly to the case of SAT formulas, we need to find a suitable way of translating a XOR constraint to a MILP formulation.

Let w be the set of binary variables and U ⊆ V the set of indices of w variables of a XOR constraint.

To remove all assignments to that subset of variables with an even sum, we can use a family of canonical cuts on the unit hypercube BID4 : DISPLAYFORM0 which is effectively separating each such assignment with one constraint.

Although exponential in k, BID34 has shown that each of those constraints -and only those -are necessary to define a convex hull of the feasible assignments in the absence of other constraints.

However, we note that we can do better when k = 2 by using DISPLAYFORM1 Due to the multiple XOR constraints used and the small k, we avoid moving away from the original space of variables.

Alternatively, BID53 provides an extended formulation requiring a polynomial number of constraints.

We note that these two possibilities have also been discussed by BID20 for a related application of probabilistic inference.

The probabilities given to the lower bounds by Algorithm 1 are due to the main result in BID26 , which is based on the following parameters: XOR size k; number of restrictions r; loop repetitions i; number of repetitions that remain feasible after j restrictions f [j]; deviation δ ∈ (0, 1/2]; and precision slack α ≥ 1.

We choose the values for the latter two.

A strict lower bound of 2 r−α can be defined if DISPLAYFORM0

@highlight

We provide improved upper bounds for the number of linear regions used in network expressivity, and an highly efficient algorithm (w.r.t. exact counting) to obtain probabilistic lower bounds on the actual number of linear regions.

@highlight

Contributes to the study of the number of linear regions in RELU neural networks by using an approximate probabilistic counting algorithm and analysis

@highlight

Builds off previous work studying the counting of linear regions in deep neural networks, and improves the upper bound previously proposed by changing the dimensionality constraint

@highlight

The paper deals with expressiveness of a piecewise linear neural network, characterized by the number of linear regions of the function modeled, and leverages probabilistic algorithms to compute the bounds faster, and proves tighter bounds.