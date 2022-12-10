We propose a novel score-based approach to learning a directed acyclic graph (DAG) from observational data.

We adapt a recently proposed continuous constrained optimization formulation to allow for nonlinear relationships between variables using neural networks.

This extension allows to model complex interactions while being more global in its search compared to other greedy approaches.

In addition to comparing our method to existing continuous optimization methods, we provide missing empirical comparisons to nonlinear greedy search methods.

On both synthetic and real-world data sets, this new method outperforms current continuous methods on most tasks while being competitive with existing greedy search methods on important metrics for causal inference.

Structure learning and causal inference have many important applications in different areas of science such as genetics [5, 12] , biology [13] and economics [7] .

Bayesian networks (BN), which encode conditional independencies using directed acyclic graphs (DAG), are powerful models which are both interpretable and computationally tractable.

Causal graphical models (CGM) [12] are BNs which support interventional queries like: What will happen if someone external to the system intervene on variable X?

Recent work suggests that causality could partially solve challenges faced by current machine learning systems such as robustness to out-of-distribution samples, adaptability and explainability [8, 6] .

However, structure and causal learning are daunting tasks due to both the combinatorial nature of the space of structures and the question of structure identifiability [12] .

Nevertheless, these graphical models known qualities and promises of improvement for machine intelligence renders the quest for structure/causal learning appealing.

The problem of structure learning can be seen as an inverse problem in which the learner tries to infer the causal structure which has generated the observation.

In this work, we propose a novel score-based method [5, 12] for structure learning named GraN-DAG which makes use of a recent reformulation of the original combinatorial problem of finding an optimal DAG into a continuous constrained optimization problem.

In the original method named NOTEARS [18] , the directed graph is encoded as a weighted adjacency matrix W which represents coefficients in a linear structural equation model (SEM) [7] .

To enforce acyclicity, the authors propose a constraint which is both efficiently computable and easily differentiable.

Most popular score-based methods for DAG learning usually tackle the combinatorial nature of the problem via greedy search procedures relying on multiple heuristics [3, 2, 11] .

Moving toward the continuous paradigm allows one to use gradient-based optimization algorithms instead of handdesigned greedy search algorithms.

Our first contribution is to extend the work of [18] to deal with nonlinear relationships between variables using neural networks (NN) [4] .

GraN-DAG is general enough to deal with a large variety of parametric families of conditional probability distributions.

To adapt the acyclicity constraint to our nonlinear model, we use an argument similar to what is used in [18] and apply it first at the level of neural network paths and then at the level of graph paths.

Our adapted constraint allows us to exploit the full flexibility of NNs.

On both synthetic and real-world tasks, we show GraN-DAG outperforms other approaches which leverage the continuous paradigm, including DAG-GNN [16] , a recent nonlinear extension of [18] independently developed which uses an evidence lower bound as score.

Our second contribution is to provide a missing empirical comparison to existing methods that support nonlinear relationships but tackle the optimization problem in its discrete form using greedy search procedures such as CAM [2] .

We show that GraN-DAG is competitive on the wide range of tasks we considered.

We suppose the natural phenomenon of interest can be described by a random vector X ∈ R d entailed by an underlying CGM (P X , G) where P X is a probability distribution over X and G = (V, E) is a DAG [12] .

Each node i ∈ V corresponds to exactly one variable in the system.

Let π G i denote the set of parents of node i in G and let X π G i denote the random vector containing the variables corresponding to the parents of i in G.

We assume there are no hidden variables.

In a CGM, the distribution P X is said to be Markov to G which means we can write the probability density function (pdf) as p(

.

A CGM can be thought of as a BN in which directed edges are given a causal meaning, allowing it to answer queries regarding interventional distributions [5] .

In general, it is impossible to recover G only given samples from P X .

It is, however, customary to rely on a set of assumptions to render the structure fully or partially identifiable.

Definition 1.

Given a set of assumptions A on a CGM M = (P X , G), its graph G is said to be identifiable from P X if there exists no other CGMM = (P X ,G) satisfying all assumptions in A such thatG = G andP X = P X .

There are many examples of graph identifiability results for continuous variables [11, 9, 14, 17] as well as for discrete variables [10] .

Those results are obtained by assuming that the conditional pdf p i ∀i belongs to a specific parametric family P. For example, if one assumes that

where f i is a nonlinear function satisfying some mild regularity conditions, then G is identifiable from P X (see [11] for the complete theorem and its proof).

We will make use of this results in Section 4.

Structure learning is the problem of learning G using a data set of n samples {x (1) , ..., x (n) } from P X .

Score-based approaches [12] cast this estimation problem as an optimization problem over the space of DAGs, i.e.

Ĝ = arg max G∈DAG Score(G).

The score is usually the maximum likelihood of your data given a certain model.

Most score-based methods embrace the combinatorial nature of the problem via greedy search procedures [3, 2] .

We now present the work of [18] which approaches the problem from a continuous optimization perspective.

To cast the combinatorial optimization problem into a continuous constrained one, [18] proposes to encode the graph G on d nodes as a weighted adjacency matrix

d×d which represents (possibly negative) coefficients in a linear structural equation model (SEM) [7] of the form X i := u i X + N i ∀i where N i is a noise variable.

Let G U be the directed graph associated with the SEM and let A U be the (binary) adjacency matrix associated with G U .

One can see that the following equivalence holds:

To make sure G U is acyclic, the authors propose the following constraint on U :

where e M ∞ k=0 M k k! is the matrix exponential and is the Hadamard product.

It can be shown that G U is acyclic iff the constraint is satisfied (see [18] for a proof).

The authors propose to use a regularized negative least square score (maximum likelihood for a Gaussian noise model).

The resulting continuous constrained problem is

where X ∈ R n×d is the design matrix containing all n samples.

The nature of the problem has been drastically changed: we went from a combinatorial to a continuous problem.

The difficulties of combinatorial optimization have been replaced by those of non-convex optimization, since the feasible set is non-convex.

Nevertheless, a standard numerical solver for constrained optimization such has an augmented Lagrangian method (AL) [1] can be applied to get an approximate solution.

3 GraN-DAG: Gradient-based neural DAG learning

We propose a new nonlinear extension to the framework presented in Section 2.3.

For each variable X i , we learn a fully connected neural network with L hidden layers parametrized by

} where W ( ) (i) is the th weight matrix of the ith NN.

Each NN takes as input X −i ∈ R d , i.e. the vector X with the ith component masked to zero, and outputs θ (i) ∈ R m , the m-dimensional parameter vector of the desired distribution family for variable X i .

The fully connected NNs have the following form

where g is a nonlinearity applied element-wise.

Let φ {φ (1) , . . .

, φ (d) } represents all parameters of all d NNs.

Without any constraint on its parameter φ (i) , neural network i models the conditional pdf p i (x i |x −i ; φ (i) ).

Note that the product

is not a valid joint pdf since it does not decompose according to a DAG.

We now show how one can constrain φ to make sure the product of all conditionals outputted by the NNs is a valid joint pdf.

The idea is to define a new weighted adjacency matrix A φ similar to the matrix U encountered in Section 2.3, which can be directly used inside the constraint of Equation 3 to enforce acyclicity.

Before defining the weighted adjacency matrix A φ , we need to focus on how one can make some NN outputs unaffected by some inputs.

Since we will discuss properties of a single NN, we drop the NN subscript (i) to improve readability.

We will use the term neural network path to refer to a computation path in a NN.

For example, in a NN with two hidden layers, the sequence of weights (W

kh2 ) is a NN path from input j to output k. We say that a NN path is inactive if at least one weight along the path is zero.

We can loosely interpret the path product |W (1) h1j ||W (2) h2h1 ||W (3) kh2 | ≥ 0 as the strength of the NN path, where a path product equal to zero if and only if the path is inactive.

Note that if all NN paths from input j to output k are inactive (i.e. the sum of their path products is zero), then output k does not depend on input j anymore since the information in input j will never reach output k. The sum of all path products from every input j to every output k can be easily computed by taking the product of all the weight matrices in absolute value.

where |W | is the element-wise absolute value of W .

It can be verified that C kj is the sum of all NN path products from input j to output k. To have θ independent of variable X j , it is sufficient to have m k=1 C kj = 0.

This is useful since, to render our architecture acyclic, we need to force some neural network inputs to be inactive (this corresponds to removing edges in our graph).

We now define a weighted adjacency matrix A φ that can be used in constraint of Equation 3 .

where C (i) denotes the connectivity matrix of the NN associated with variable X i .

As the notation suggests, A φ ∈ R d×d ≥0 depends on all weights of all NNs.

Moreover, it can effectively be interpreted as a weighted adjacency matrix similarly to what we presented in Section 2.3, since we have that

We note G φ to be the directed graph entailed by parameter φ.

We can now write our adapted acyclicity constraint:

This guarantees acyclicity.

The argument is identical to the linear case, except that now we rely on implication (8) instead of (2).

We propose solving the maximum likelihood optimization problem

where π φ i denotes the set of parents of variable i in graph G φ .

Note that

; θ (i) ) is a valid log-likelihood function when constraint (9) is satisfied.

As suggested in [18] , we apply an augmented Lagrangian approach to get an approximate solution to program (10) .

Augmented Lagrangian methods consist of optimizing a sequence of subproblems for which the exact solutions are known to converge to a stationary point of the constrained problem under some regularity conditions [1] .

We approximately solve each subproblem using RMSprop [15] , a stochastic gradient descent variant popular for NN.

We empirically compare GraN-DAG to various baselines (both in the continuous and combinatorial paradigm), namely DAG-GNN [16] , NOTEARS [18] , RESIT We first present a comparison on synthetic data sets.

We sampled 10 graphs (e.g. with 50 nodes and an average of 200 edges) and data distributions of the form

) with f i ∼ GP and evaluated different methods using SHD and SID (we report the average and the standard deviation over those data sets).

Note that we are in the identifiable case presented in Section 2.2.

GraN-DAG, NOTEARS and CAM all make the correct gaussian assumption in their respective models.

In Table 1 we report a subset of our results.

GraN-DAG outperforms other continuous approaches while being competitive with the best performing discrete approach we considered.

In addition, we considered a well known real world data set which measures the expression level of different proteins and phospholipids in human cells [13] (the ground truth graph has 11 nodes and 17 edges).

We found GraN-DAG to be competitive with other approaches.

Our implementation of GraN-DAG can be found here.

@highlight

We are proposing a new score-based approach to structure/causal learning leveraging neural networks and a recent continuous constrained formulation to this problem