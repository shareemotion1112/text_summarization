Combinatorial optimization is a common theme in computer science.

While in general such problems are NP-Hard, from a practical point of view, locally optimal solutions can be useful.

In some combinatorial problems however, it can be hard to define meaningful solution neighborhoods that connect large portions of the search space, thus hindering methods that search this space directly.

We suggest to circumvent such cases by utilizing a policy gradient algorithm that transforms the problem to the continuous domain, and to optimize a new surrogate objective that renders the former as generic stochastic optimizer.

This is achieved by producing a surrogate objective whose distribution is fixed and predetermined, thus removing the need to fine-tune various hyper-parameters in a case by case manner.

Since we are interested in methods which can successfully recover locally optimal solutions, we use the problem of finding locally maximal cliques as a challenging experimental benchmark, and we report results on a large dataset of graphs that is designed to test clique finding algorithms.

Notably, we show in this benchmark that fixing the distribution of the surrogate is key to consistently recovering locally optimal solutions, and that our surrogate objective leads to an algorithm that outperforms other methods we have tested in a number of measures.

Combinatorial optimization is one of the foundational problems of computer science.

Though in general such problems are NP-hard BID20 , it is often the case that locally optimal solutions can be useful in practice.

In clustering for example, a common objective is to divide a given set of examples into a fixed number of groups in a manner that would minimize the distances between group members.

As enumerating all the possible groupings is usually intractable, local search methods such as k-means BID16 are frequently used to approach such problems.

We find the persistent use of k-means in a wide variety of applications as convincing evidence that from a practical perspective, locally optimal solutions can be useful.

In the combinatorial setting however, solution neighborhoods are not always available, and even when they are, in many interesting cases they only connect small parts of the search space.

For example, when the search space involves computer programs, it is not clear how replacing one operation with another (for example, an if clause with an addition operation) impacts the program behavior even if the program validity is preserved.

Though one can define a limited but sensible set of neighboring solutions (e.g., replace an addition with a multiplication), neighborhoods that build on those usually connect only a tiny fraction of the search space.

Another interesting case involves natural language sentences where replacing one word with another (say, 'very' to 'not'), or changing clauses order can completely change the meaning of a sentence.

A third popular scenario involves sequential decision making as is the case in reinforcement learning problems with discrete action spaces, where it is not always clear that two action sequences can be related if the initial actions are different.

In such combinatorial problems, methods that transforms one solution to another (either directly or through smoothing) might be confined to a small sub-space, and therefore in such problems, searching the solution space directly is unfavorable.

One type of algorithms which are suitable to such combinatorial problems, and have drawn considerable interest in the last few years are policy gradient methods BID25 .

The general strategy these methods adopt is to construct a parametric sampling distribution over the search space, and to optimize the expected value of some given objective function by applying gradient updates in the parameters' space.

In spite of their apparent generality, these gradient updates require special attention.

In particular, the sampled objective values affect both the sign and the magnitude of the gradient step size.

On the one hand, such dependence on the objective values is what allows these algorithms to give higher likelihood to examples which achieve better objective values.

On the other hand, such direct dependence makes it hard to tune the step sizes by means of predetermined hyper-parameters.

As our goal is to extend such constructions to any objective in a generic fashion, we seek to transform the construction so that it will only be sensitive to the order relation the objective induces.

In this construction however, the objective is essentially a random variable whose distribution changes from one problem to another, and not only that, it keeps on changing throughout the optimization.

As a result, it seems that finding a generic rule for tuning various hyper-parameters in a manner that fits all scenarios seems impractical.

Following this understanding, we purpose to utilize a generic surrogate objective function that has the following two properties.

First, the surrogate should preserve the set of locally optimal solutions if solution neighborhoods can be defined.

Second, the surrogate should have a fixed and predetermined distribution for every possible objective, and this distribution should remain fixed throughout the optimization.

Once in this form, generic rules for setting various hyper-parameters can be found, and that can provide us with a generic stochastic optimizer.

Though it might seem that such general purpose surrogate objectives could be hard to find, we show that by utilizing the empirical cumulative distribution function (CDF henceforth) of the original objective, these can be easily constructed.

We discuss few possible surrogate objectives, and purpose one such version which makes the basis our method.

Since the crux of our method is based on capitalizing on the CDF of the original objective, we refer to our method as CAkEWaLK which stands for CumulAtivEly Weighted LiKelihood.

We start by considering policy gradient methods as stochastic optimization algorithms for combinatorial problems in section 2, and proceed to present the Cakewalk method in section 3.

In section 4 we discuss how Cakewalk is related to the cross-entropy (CE henceforth) method, to policy-gradient methods in reinforcement learning, and to multi-arm bandit algorithms.

Since we are interested in methods that can recover locally optimal solutions when these can be defined, we use the problem of finding inclusion maximal cliques in undirected graphs as a controlled experiment for testing this property in a non-trivial setting.

For that matter, in section 6 we investigate how to apply such methods to the clique problem, and in section 7 we report experimental results on a dataset of graphs on which results are regularly published.

Lastly, as an additional experimental task, we show in appendix section B how Cakewalk can be used to produce an algorithm that outperforms the most commonly used algorithms for k-medoids clustering, the combinatorial counterpart of k-means.

Notably, we use this task to demonstrate that Cakewalk can also be used to optimize the starting point of greedy algorithms that search the input space directly, thus providing empirical evidence that supports Cakewalk's effectivity in a greater variety of combinatorial problems.

We set out on constructing a stochastic optimization algorithm for combinatorial optimization problems, and start by stating the problem.

Let f be an objective function which we need to maximize, and let DISPLAYFORM0 N be a string that describes N items such that each x j is one of a discrete set of M items.

In this text we denote discrete sets {1, . . .

, K} using [K] .

Our goal is to search the space X = [M ] N for some x that achieves an optimal f (x ) = y .

Since X is discrete, in general this problem is NP-Hard (maximum clique can be reduced to this description), hence we focus only on finding locally optimal solutions rather than the global optimum x .

For the purpose of defining locally optimal solutions, we'll rely on a neighborhood function N that maps each x to its neighboring set, though the methods we consider treat f as a black-box, and don't require such N for their operation.

Our goal is to find some locally optimal solution x * ??? X * f where the set of locally optimal solutions is defined as X * f = {x ??? X |???x ??? N (x) .f (x) ??? f (x )}.

Preferably, we would like to find some x * whose objective value y * = f (x * ) is as large as possible, though in general, this cannot be guaranteed.

We describe a stochastic optimization algorithm for problems of this form.

Let X be a random variable that is defined over X , and which is distributed according to a parametric distribution P ?? that the algorithm maintains.

In addition, let Y be a random variable that is defined over the values of the objective function f , i.e. Y = f (X).

We emphasize that in this text we refer to random variables using capital English letters in bold such as X or Y , and we use x and y to refer to elements in their appropriate sample spaces (deterministic quantities).

During the optimization, the algorithm iteratively samples solutions x according to P ?? , and updates the parameters ?? ??? R d which govern P ?? in a manner that reflects how good is the objective value y = f (x) with which x is associated.

Initially P ?? (X = x) is multi-variate uniform (fully specified in section 5), but as the optimization continues, the algorithm decreases the entropy in the distribution until eventually only few solutions become likely, and sampling some x from P ?? , with high probability, returns some locally optimal solution.

Since we discuss an iterative algorithm that at each iteration t updates the parameters ?? t , we refer to the random variables that are associated with P ??t by X t and Y t .

Lastly, as a short hand notation, we refer to P ?? (X = x) simply by P ?? (x).Since we learn a distribution function, we say that our learning objective J(??) is to maximize the expectation over x ??? P ?? of the original objective which we denote as E ?? [Y ] .

To find the parameters ?? which maximize E ?? [Y ], we derive a gradient ascent algorithm which relies on estimates of DISPLAYFORM1 To calculate the gradient, we use the log-derivative trick, DISPLAYFORM2 .

Next, we can use Monte Carlo estimation BID27 DISPLAYFORM3 of some fixed size K is sampled using P ??t .

Denoting this estimate by ??? t , then the update at iteration t takes the following form, DISPLAYFORM4 (1) DISPLAYFORM5 where ?? t is a learning rate parameter that is predetermined.

We describe the update step using a vanilla gradient update mostly for illustratory purposes, though in practice any gradient based update such as AdaGrad BID10 ) or Adam Kingma & Ba (2014 can be used instead.

Turns out that for positive learning rates this stochastic optimization scheme converges to a local maximum of J, and when using the optimal parameters ?? * , sampling from P ?? * returns locally optimal solutions x * ??? X * f with high probability BID30 .

Nonetheless, such gradient estimates are known to be highly variable BID19 , which requires drawing large samples at each iteration which is costly.

Though there are techniques for reducing the variance of such estimates BID22 , these are mostly useful when tied to the specifics of a given objective.

We approach this problem differently, and consider instead how can we adapt the optimization objective in a manner that allows us to rely on noisy gradient estimates that only involve a single example (i.e., setting K = 1), while ensuring we converge to a distribution that still allows us to sample some x * ??? X * f .

Since we focus on online updates, for the reminder of the text we drop the superscript k when referring to x k t and y k t .

At this point, we've set the stage for discussing how can we transform the previous construction into a generic stochastic optimizer.

We start by examining equations 1 and 2 and observing that if we update ?? t in the direction of ??? ?? log P ?? (x t ), ?? t y t can be considered as the step we're taking in that direction.

Thus, the sign and magnitude of ?? t y t essentially determine whether we increase or decrease the likelihood of x t , and to what extent we do so.

The implication this has over the optimization is that the distributions of {?? t Y t } T t=1 determine the course of the optimization.

If for example |?? t Y t | is unbounded from above, we might take steps that are too large, which might cause us to diverge.

Steps that are too small are unfavorable as well, as these will keep the sampling distribution too close to uniform, and due to the combinatorial nature of X , finding good xs can take exponentially many examples.

This extends to scenarios that involve functions that have a different scale.

For example, suppose that we have two functions such that f 2 (x) = cf 1 (x) for every x, with c being some fixed positive constant.

Clearly, X * f1 = X * f2 , nonetheless, sampling and updating the parameters using equations 1 and 2 would change the speed of the optimization by a factor c. Though one can adjust the learning rates to the particularities of some given objective, such an approach would require that we tune the optimization on a case by case basis.

Lastly, since in general we don't know ahead of time the distribution of each Y t , it seems that if we follow the construction presented in section 2, we won't be able to determine the series {?? t } T t=1 in a manner that would fit all scenarios.

This reasoning leads us to conclude that if we wish to obtain generic updates, we must come up with some fixed surrogate objective function which preserves X * f , and for which we can determine the distributions of Preserving the original set of optimal solution is the easy part, as all we need to do is to require that w will be monotonic increasing, and that would imply that X * f ???

X * w???f (and strict monotonicity would ensure that X * f = X * w though we don't insist on that).

The harder part is to construct w in a manner that would fix the distribution of w(Y t ) for all t. Nonetheless, basic probability tells us that if F t is the CDF of Y t , then F t (Y t ) is uniformly distributed on [0, 1] Wasserman (2013).

Since every CDF is monotonic increasing, if we construct w using F t , we can preserve the original set of optimal solutions.

More importantly, if we can estimate F t , we could use it to produce our surrogate objective as it would fix the surrogate's distribution once and for all, thus making significant progress towards our goal.

Next, since insisting that w (Y t ) ??? U (0, 1) might not be ideal, we take this idea one step further, and utilize another monotonic increasing function g for which g (F t (Y t )) can be distributed differently.

For purposes that we specify next, we also require that g will be bounded.

Since we don't have access to F t in general, as was the case with the gradient, we need to estimate it from data.

Fortunately enough, since the image of f is one dimensional (an optimization objective), order statistics can supply us with highly reliable non-parametric estimates for each F t .

The only question that comes up is how can we perform the aforementioned estimation without drawing a large sample at each iteration.

Due to equation 2, if we use a sampling distribution for which ??? ?? log P ?? (x t ) is bounded, then since w (y t ) is bounded as well, ??? t will be bounded for every x t and y t .

The main implication of this property is that we can control how different the parameters will be between any two iterations, i.e., that for any two iterations t and t ??? k where k ??? [t ??? 1] we can make ?? t ??? ?? t???k as small as we want simply by changing ?? t .

Thus, instead of drawing a large sample in each iteration, we can say the last objective values y t???1 , . . .

, y t???k are approximately i.i.d from P ??t???1 .

Therefore, if we use small enough learning rates, we can us?? DISPLAYFORM0 is the indicator function.

In our experiments, using some fixed learning rate ?? ??? (0, 1) along with k = 1 ?? seem to work quite well.

Overall, the parameters' updates we suggest have the following form, DISPLAYFORM1

In this section, we focus on a single iteration t, and thus, drop the subscript t in all cases.

The purpose of the first option we present is to illustrate the connection between our algorithm, and the CE method, and for that reason we denote this weight function by?? CE (y) = g CE F (y) , and its associated transformation by g CE .

Given some small ?? ??? [0, 1] which is decided by the user a-priori (typically, 0.1 or 0.01), g CE is a thresholding function g CE (z) = I [z ??? 1 ??? ??].

Clearly, for any fixed ??,?? CE is monotonic increasing and bounded, and?? CE (Y ) is a Bernoulli random variable with probability ??.

Notably, using g CE in equation 3 leads to an update which can be considered as an online version of the CE method.

There are two main disadvantages to?? CE .

First of all, it relies on another parameter ?? that requires manual tuning.

More importantly,?? CE uses only the highest ?? percentile of the examples to update P ?? while in fact the worst xs supply valuable information -they have low objective values, and thus, their likelihood should be decreased rather than simply ignored.

Thus, we suggest two weight functions which fix these issues.

Probably the simplest option is to use the empirical CDFF directly, which would makeF (Y ) uniform discrete on [0, 1].

While this surrogate doesn't involve any extra parameters, nor does it ignore the information supplied by every x, it still has one major drawback, it leads to an increase in the likelihood of every example it sees.

This create a bias towards xs that have already been sampled, compared with xs that weren't, even though their associated objective value might be better.

Since X grows exponentially fast with N , as N grows, examples that are drawn early in the process can influence the course of the optimization dramatically.

Following this reasoning, we adjustF so that it would only increase the likelihood of only half of the examples, and decrease the likelihood of the other half.

To do so, we make?? (y) = 2F (y) ??? 1.

By construction, it follows that?? (Y ) is uniform discrete on [???1, 1].

In this fashion, when applied with some fixed learning rate,?? determines whether the likelihood of DISPLAYFORM0 * which had the highest y * some example will be increased or decreased, and to what extent.

Notably, this is achieved along with full specification of the distribution of?? (Y ).

This is a major advantage compared with, for example, transforming Y with its estimated z-score, as in this case we can't determine how w (Y ) is distributed, nor can we guarantee that |w (Y )| is bounded (leading to a risk of divergence, and disrupting of the online estimation of F ).

We summarize Cakewalk with??, and any gradient addition rule Add (this includes hyper-parameters such as learning rate) in algorithm 1.

Our method is closely related to the CE method.

CE was introduced by Rubinstein initially for estimating low probability events BID23 , and later adapted to combinatorial optimization problems BID24 .

Turns out that when CE is applied with discrete sampling distributions the likelihood-ratio term cancels out, and the construction is equivalent to maximizing the likelihood of the examples whose objective values belong to the highest ?? percentile BID9 .

Thus, in this case CE's update step is equivalent to maximizing the surrogate objective?? CE described in section 3.1.

As discussed in section 3.1,?? CE has two major shortcomings, and these lead us to suggest a different surrogate objective which makes the basis for Cakewalk.

In addition to these differences, Cakewalk is an online algorithm whereas CE requires drawing a large sample in each iteration so as to estimate the CDF.

Our construction enables us to rely on bounded gradient updates that facilitate online estimation of the CDF, and therefore Cakewalk's iterations are considerably less computationally expensive than those of CE.

The next family of algorithms to which our method is related to are policy gradient methods.

The research on these was initiated by Williams with REINFORCE Williams (1988) , an algorithm which we consider as the prototype to Cakewalk, and which provides Cakewalk with convergence guarantees.

Most of the work on policy gradient methods derives from REINFORCE, essentially discussing how to rescale the objective in various scenarios.

For example, actor critic methods Sutton & Barto (2017) use estimates?? of E (Y ) that are produced with some model of the objective, and can be used to make y ????? zero mean.

As these methods rely on a particular model of the objective, they are inherently problem specific.

Of these methods, probably the natural actor-critic algorithm BID21 better fits Cakewalk's general purpose nature.

This algorithm rescales the estimated gradient by multiplying it by the inverse of the Fisher information matrix.

As this requires large sample to accurately estimate both the gradient, and of the Fisher information matrix, the natural actor-critic is considerably more computationally expensive than online algorithms such as Cakewalk or REINFORCE.

The third family of algorithms to which our method is related are multi-arm bandit algorithms.

In the bandit setting, a learner is faced with a sequential decision problem, where in each round an arm is chosen, and each arm is associated with some non-deterministic loss.

Initially suggested by Thompson BID26 , this setting has been explored extensively with the notable successes of the UCB algorithm Auer (2002) ; BID2 for cases where the losses are stochastic, and the Exp3 BID1 BID3 for when they can even be determined by an adversary.

Over the years these have become a basis for a wide variety of algorithms BID7 for various settings which that even extend to cases that involve high dimensional structured arms BID4 ; McMahan & Blum BID8 .

The key difference between the bandit and the optimization setting is that the losses associated with each of the arms are non-deterministic, and thus in the bandit setting the main challenge is to balance estimating the statistics associated with each of the arms, with exploiting the information gathered thus far.

In the optimization setting however, the goal is simply to find the best deterministic solution using the least number of steps.

Thus, in spite of the apparent similarity, it is this fundamental difference that separates the optimization from bandit settings.

Before we specify a particular distribution, we wish to emphasize that the Cakewalk update rule isn't tied to any particular sampling distribution.

The distribution we specify next is only used as an example, and as a basis for the experiments we report later.

Following Rubinstein's construction, we use a simple distribution that factorizes into a sequence of independent distributions, each defined over a different dimension.

In this manner, the number of parameters required to represent P ?? (x) grows only linearly with M N , instead of the exponential number of parameters that is required to represent the full joint distribution.

Formally, each x j is drawn independently according to a softmax distribution P ?? (x j = i) = Next, we describe ??? ?? log P ?? (x) in terms of partial derivatives, DISPLAYFORM0 Note that since ??? ?? log P ?? (x) is bounded, we can estimateF in an online manner.

In this section, we set the grounds for investigating whether algorithms that only rely on function evaluations can recover locally optimal solutions.

We emphasize that our goal is to investigate this question, and not compete with algorithms that rely on some neighborhood function for their operation, and which search the input space directly.

We study this question on a NP-hard problem instead of problem for which we can find the global optimum in polynomial time, as it important to verify that such methods can recover non-trivial optima in challenging scenarios.

We focus on the problem of finding locally maximal cliques, as the notion of inclusion maximal cliques naturally entails what neighborhood function should be used to judge this property.

Formally, a graph G is a pair (V, E) where V = [N ] is a set of vertices, and E ??? V ?? V is a set of edges.

G is undirected if for every (i, j) ??? E it follows that (j, i) ??? E. A clique in an undirected graph is a subset of vertices U ??? V such that each pair of which is connected by an edge.

An inclusion maximal clique U is such that there is no other v ??? V \ U for which U ??? {v} is also a clique.

Next, we design an objective that could inform algorithms that only rely on function evaluations how densely connected is some subgraph, and which favors larger subgraphs.

We refer to this function as the soft-clique-size function, and denote it by f SCS .

For our purposes, we say the space X = {0, 1} N correspond to strings which determine membership in some subgraph U .

Let x ??? X , then for each vertex j ??? V , we say that j ??? U if and only if x j = 1, and accordingly we denote such subgraphs by U x .

If some U x is a clique, for every i, j ??? U x , i = j it follows that (i, j) ??? E, and therefore DISPLAYFORM0 .

As a consequence, dividing by the RHS produces a subgraph density term.

Next, we add a parameter ?? ??? [0, 1] that rewards larger subgraphs, and which could indicate to an algorithm it should prefer larger subgraph over smaller ones.

To achieve this, we change aforementioned denominator to |U x | (|U x | ??? 1 + ??).

Lastly, to avoid division by zero for cases |U x | < 2, we can wrap the denominator with max (??, 1).

Altogether, DISPLAYFORM1 max (|U x | (|U x | ??? 1 + ??) , 1) To see why higher ?? can reward larger cliques we focus on the case that |U x | ??? 2, and observe that for U x which is clique, when ?? = 0, f SCS (x, G, 0) = 1.

However, when ?? = 1, f SCS (x, G, 1) = |Ux|???1 |Ux| , and thus, the larger U x is, the closer this ratio is to 1.

In this manner, increasing ?? gives larger subgraphs a 'boost' compared with smaller one, though it could be that some subgraph which isn't a clique will receive a higher value than some smaller subgraph which is a clique (only for ?? = 0 we get that f SCS (x, G, 0) = 1 necessarily means that U x is clique).

Empirically, we see that the algorithms we've tested aren't very sensitive to the value of ??.

As a benchmark for clique finding, we used 80 undirected graphs that were published as part of the second DIMACS challenge BID13 .

Each graph was generated by a random generator that specializes in a particular graph type that conceals cliques in a different manner.

The graphs contain up to 4000 nodes, and are varied both in their number of nodes and in their edge density.

We tested each method on all 80 graphs, letting it maximize the soft-clique-size function using various values of ??.

To determine if a clique is inclusion maximal, since a-priori we don't know which ?? will lead to such clique, we've executed each method using each of the values 0.0, 0.1, . . .

, 1.0 as ??.

In each execution, we've executed a method for 100 |V | samples (hence runtime is fixed per graph), and at the execution's end, we recorded both the best solution along with its objective value, as well as the sample number in which that solution was found.

In terms of the methods tested, following the discussion on related work, we experimented with the CE method, three versions of REINFORCE, and of the bandit algorithms we've used Exp3.

As we focus on online algorithms, for CE, we used the online version that we derived in this work, using two threshold values suggested by Rubinstein, ?? = 0.1 and ?? = 0.01, and refer to these as OCE 0.1 and OCE 0.01 , with O standing for online.

Next, we've experimented with three versions of REINFORCE.

First is the vanilla version, second is a version where the mean?? is subtracted from y as a baseline, and a third uses the objective's estimated z-score y????? ?? .

We refer to these by REINF, REINF B , and REINF Z .

For Cakewalk, we used both the unscaled empirical CDFF , and its scaled counterpart w, denoting these as CWF and CW??.

Note however that the former is only used for comparisons, and that we identify Cakewalk with the latter.

For estimating??,?? andF , we've used the last 100 objective values, and thus, both REINF B , and REINF Z make for important comparison as these only transform the objective values, but do not fix its distribution as CE and Cakewalk do.

For gradient update methods, we've used vanilla stochastic gradient ascent (SGA henceforth), AdaGrad, and the Adam updates.

The latter two methods are considered scale invariant, and thus could help Exp3, REINF, and REINF B handle changes in the objective's scale.

Altogether, we've tested 8 optimization methods, 3 update steps, on 80 graphs, and 11 values of ??, leading to a total of 21120 separate executions.

We specify the complete experimental details in the appendix section A.We analyzed 4 performance measures for each of the 8 optimizers, and the 3 gradient update types, and accordingly report results in four 3 ?? 8 tables.

In the following, we refer to each combination of an optimizer and gradient update as a method.

First, we examined whether a locally optimal solution was found.

To test for local optimality for the soft-clique-size, given a result x in some graph, we compared it to every other x for that graph whose Hamming distance from x is 1, and checked that no x s achieved higher soft-clique-size.

We report average local optimality in such Hamming neighborhoods in table 1.

Then, to test inclusion maximality of the returned solutions, since the soft-clique-size doesn't guarantee convergence to cliques, for every graph, we tested whether a method returned at least one inclusion maximal clique when applied with some ??.

We report average inclusion maximality in table 2.

Next, since some methods find their best solution earlier than others, to analyze the sampling efficiency of each method, we calculated the ratio of the best sample number and the total number of samples used in that execution.

Since this comparison only makes sense when controlling for the quality of the solution, we excluded REINF and Exp3 from it as they didn't return locally optimal solutions.

We report average best-sample to total-samples ratio in table 3.

To ensure returned solutions aren't trivial (say cliques of size 2), for each graph, we compared the largest inclusion maximal clique found by that method, and compared it to the best known solution for that graph, using results from Nguyen (2017).

We report average largest-found-clique to largest-known-clique ratios in table 4.

Lastly, we performed multiple hypothesis tests to compare every optimizer to CW?? in all the experimental conditions using one sided sign test BID11 , and to control the false discovery rate Wasserman (2013), we determined the significance threshold at a level of 10 ???2 using the Benjamini-Hochberg method BID27 .

The best optimizer in each table is marked using bold fonts.

The results in tables 1 and 2 clearly support our main proposition that in the considered setting, a surrogate objective whose distribution is fixed and predetermined significantly improves the rate in which locally optimal solutions are recovered.

Both CW?? and OCE 0.1 rely on such surrogates, and both outperform Exp3 and all versions of REINFORCE which do not employ such surrogates.

Interestingly, it appears that having a surrogate whose distribution is fixed is more effective than to normalize the objective values as the previous comparison also includes REINF Z .

Nonetheless, not all distributions are as effective (OCE 0.01 and CWF didn't perform as well), and of the ones that we have tested, uniform on [???1, 1] seems to be favorable.

CW?? clearly outperforms OCE 0.1 in table 1, and the latter only comes close in the more permissive comparison which selects the best result out of 11 different executions (different values of ??) as reported in table 2.

In terms of sample efficiency, the results in table 3 show that even though OCE 0.1 can recover locally optimal solutions, it is not as efficient as CW?? which finds the best solution considerably faster.

When considering the various gradient updates, it appears that CW?? with AdaGrad produces the best combination as it outperforms all others methods in almost all measures (CW?? with Adam converges slightly faster, though at the cost of worse optimality rates).

Lastly, the comparisons to the best known results in table 4 show that the recovered solutions are far from trivial, and that Cakewalk might even approach the performance of problem specific algorithms which have access to a complete specification of the problem.

Overall, we find these results are a strong indication that Cakewalk is a highly effective optimization method, and we believe that future research will prove its effectiveness in other domains such as continuous non-convex optimization, and in reinforcement learning problems.

As a benchmark, we used 80 undirected graphs that were published as part of the second DIMACS challenge BID13 which specifically focused on combinatorial optimization, and included instances of the clique problem.

Over the years, this dataset has become a standard benchmark for clique finding algorithms, and results on it are regularly published.

In terms of the methods we use for comparison, of the bandits family of algorithms, we considered Exp3 as more suitable than UCB due to the multi-dimensionality of the problem.

For example, adding an isolated vertex v to a set of vertices U who is a clique will damage the objective.

Due to such cases, we used Exp3 instead of UCB.

We applied Exp3 to each of the N elements independently.

Note that the assumption of bounded losses/gains that the Exp3 algorithm is dependent upon is met by the soft-clique-size function.

For the gradient updates, we used SGA, AdaGrad, and Adam.

We note that AdaGrad is particularly suited to our setting as applying it on indicator data is one of its classical use cases (I [x j = i] can be considered as our data).

Adam on the other hand has proven as effective for training neural networks in a wide variety of problems, and nowadays is probably the mostly commonly used gradient update.

We decided to experiment with Exp3 in conjunction with AdaGrad and Adam even though this revokes the theoretical guarantees of Exp3 for completeness purposes.

We applied AdaGrad with ?? = 10 ???6 , and Adam with ?? 1 = 0.9, ?? 2 = 0.999, = 10 ???6 .

We used a fixed learning rate of 0.01 in all the executions.

All the algorithms were implemented in Julia BID5 by the authors.

As mentioned in the introduction, clustering is a classical problem in which practitioners regularly rely on optimization methods that return locally optimal solutions.

For that matter, in this section we study how to apply Cakewalk to the k-medoids BID12 problem, the combinatorial counterpart of k-means.

As in the k-means, we're given a set of m data points from some input space, and our goal is to divide these into k clusters in a manner that would minimize their distances to one of k representatives.

In k-means, each representative can be any point in the input space, and in k-medoids, the representatives are a subset of original points that we're given.

Since in k-medoids the representatives are known in advance, it is enough to consider as input a distance matrix D ??? R m??m + where D i,j is the distance between point i and j, and R + is the set of non-negative reals.

Thus, one can think of the problem as selecting k representatives from the m data points, and in the general case where we allow points to represent more than one cluster, the solution space becomes [m] k .

Given a set of representatives x ??? [m] k , each point i is assigned to the representative x j which minimizes the distance D i,xj to it.

In this formulation, the k-medoids optimization problem can be stated as follows, DISPLAYFORM0 Since the problem is combinatorial, going over all the possible solutions quickly becomes intractable, and greedy algorithms are usually used to approach the problem.

Of these, probably the two most commonly used algorithms are the Voronoi iteration BID12 , and the more computationally expensive, Partitioning Around Medoids (PAM henceforth) BID14 .

In both methods, first some initial set of representatives is determined, and the appropriate cluster assignments are determined.

In the former method, in each iteration, we seek to replace each representative with some other cluster member, and in the latter we seek to replace each representative with any non-representative point.

After the new representatives are determined, cluster assignments are determined, and the process is then repeated as long as the objective is improved.

In spite of the obvious computational benefits of the Voronoi iteration, PAM is probably more commonly used as it is known to achieve lower objective values.

Since both methods are greedy, the objective to which they converge is determined by how they are initialized (the algorithms are deterministic).

Thus, we can try to find the a good initialization for such greedy algorithms with some optimization algorithm.

Since Cakewalk only relies on function evaluations, it doesn't matter if we let it optimize some function g : X ??? R, or a composition g ??? f where f is some deterministic transformation X ??? X of inputs.

As long as some input x is associated with some fixed objective value y = g (f (x)), any of the methods discussed earlier will be able to optimize such an objective.

The only detail that requires attention is that now instead of returning the best x * associated with the optimal y * = g (f (x * )), we'll need to return f (x * ).

In terms of implementation, we can do this by either keeping f (x * ) instead of x * , or by applying f to the x * which is returned by Cakewalk.

In this manner, optimization algorithms that only rely on function evaluations, and greedy algorithms can come together to produce powerful algorithms that outperform the components that make them up.

To test the effectivity of each of the aforementioned optimizers on the k-medoids problem we've setup the following experiment.

Using datasets that are publicly available on White (2017), we collected 38 datasets that had between 500 and 1000 data points, and which had numerical attributes.

In order to transform these to a valid input for a k-medoid algorithm, for each dataset, we extracted all the numerical attributes, and used them as a numerical vector that represents some data point.

Then, we calculated Mahalanobis distance BID6 between each pair of points, which resulted in a distance matrix for that dataset.

For the Mahalanobis distances, we used diagonal covariance matrices.

As this point, we were able to run each of the aforementioned algorithms on these datasets.

Specifically, we used both the Voronoi iteration and PAM algorithms, as well as vanilla Cakewalk.

In order to see if we can combine Cakewalk with a greedy method to produce a combined algorithm that is more powerful, we also used Cakewalk with the Voronoi iteration using the setup mentioned earlier.

We decided to use the Voronoi iteration instead of PAM, as in our experiments the former was considerably more efficient.

We used Cakewalk with AdaGrad BID10 using the same hyper-parameters as specified in section A, except for the learning rate ??, which was set to 0.02 instead of 0.01 as we've seen that it led to faster convergence than the latter.

We used these hyper-parameters both when applying Cakewalk alone, and when applying the Cakewalk-Voronoi combination.

As a convergence criterion for Cakewalk, we use two exponentially running averages of the objective values, and determined convergence has occurred when their absolute difference ratio was smaller than 0.01.

Each running average was produced using a time constant that was calculated using the following formula 1 ??? exp DISPLAYFORM0 with a always being 0.01, and b being a parameter that is adjusted to the size of the problem.

Thus, for each dataset with m data points and k clusters, we've set b = max (mk, 1000) for the short running average, and 2b for the long running average.

We used the same converge criterion for Cakewalk+Voronoi.

Altogether, this provided us with 4 clustering algorithms.

All methods we're implemented in Julia by the authors.

As a benchmark, we executed each algorithm on all datasets with k = 10, and recorded the smallest objective value that was returned, as well as the number of objective function evaluations that were performed.

Since the Voronoi iteration does not fully reevaluate the objective completely after every step (only within each cluster), we didn't record the latter measurement for it.

Nonetheless, Cakewalk, PAM, and Cakewalk+Voronoi fully evaluate the objective in each step, and therefore can be compared in terms of their total number of function evaluations.

In the analysis our goal was to produce a ranking of the tested algorithms in terms of best objective values that were found.

Since it could be some method achieved a better objective by performing more function evaluations, we also ranked the different algorithm in terms of their total number of function evaluations.

Thus, in the following, we refer either to the best objective value, or to the total number of function evaluations as a measurement.

To determine the best to worst order of each of the 4 algorithms, we first calculated the ratio between the measurement achieved on some dataset, and the minimal value achieved by any of the algorithms on this dataset.

This is important so as to make the ranking invariant to the specifics of each dataset by fixing their scale.

Then, we calculated the median of the scaled measurements for each of the four algorithms.

This produced 4 values for the objective values, and 3 for the function evaluations.

Then, we sorted these to determine the best to worst order.

Next, to see if the differences between any two algorithms in some measurement were statistically significant, we validated their order using a one sided sign test BID11 , applying it directly to the original measurements (unscaled).

This procedure produced 3 p-values for the objective values, and two for the function evaluations.

Next, to control the false discovery rate BID27 , we determined the significance threshold at a level of 10 ???2 using the Benjamini-Hochberg method BID27 .

Following this analysis, the best to worst algorithm in terms of objective value (smallest to largest) was as follows, Cakewalk+Voronoi * < PAM * < Cakewalk * < Voronoi where A < B means that A achieved smaller value than B, and * means that the difference between the two is statistically significant.

Next, the best to worst algorithms algorithm in terms of the number of objective function evaluation (smallest to largest, excluding the Voronoi iteration) is as follows, Cakewalk+Voronoi * < PAM < Cakewalk

Following the analysis presented in section B.1, we conclude that combining Cakewalk with a greedy algorithm produces a clustering method that outperforms the two most commonly used algorithms for the k-medoids problem.

Notably, here we combined Cakewalk with the Voronoi iteration, the weaker of the two in terms of performance, and that already produced a method that outperforms PAM.

This suggests that probably combining Cakewalk with PAM can produce an even better clustering method, though we leave this to future research.

Furthermore, it seems that applying Cakewalk without any greedy method already outperforms the Voronoi iteration, showing that vanilla Cakewalk can outperform some greedy algorithms as these might be limited by the neighborhood function they rely on, a limitation that doesn't apply to a sampling algorithm such as Cakewalk.

In terms of function evaluations, it appears that PAM and Cakewalk perform about the same number of function evaluations (the difference is not statistically significant), and both perform more evaluations than the combination of Cakewalk+Voronoi.

Taken together, these results not only show that combining Cakewalk with a greedy method can produce an optimizer that outperforms the components that make it up, it also leads to a combined algorithm that converges faster.

<|TLDR|>

@highlight

A new policy gradient algorithm designed to approach black-box combinatorial optimization problems. The algorithm relies only on function evaluations, and returns locally optimal solutions with high probability.

@highlight

The paper proposes an approach to construct surrogate objectives for the application of policy gradient methods to combinatorial optimization with the goal of reducing the need of hyper-parameter tuning.

@highlight

The paper propose to replace the reward term in the policy gradient algorithm with its centered empirical cumulative distribution. 