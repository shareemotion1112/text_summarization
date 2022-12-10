Deep Learning has received significant attention due to its impressive performance in many state-of-the-art learning tasks.

Unfortunately, while very powerful, Deep Learning is not well understood theoretically and in particular only recently results for the complexity of training deep neural networks have been obtained.

In this work we show that large classes of deep neural networks with various architectures (e.g., DNNs, CNNs, Binary Neural Networks, and ResNets), activation functions (e.g., ReLUs and leaky ReLUs), and loss functions (e.g., Hinge loss, Euclidean loss, etc) can be trained to near optimality with desired target accuracy using linear programming in time that is exponential in the input data and parameter space dimension and polynomial in the size of the data set; improvements of the dependence in the input dimension are known to be unlikely assuming $P\neq NP$, and improving the dependence on the parameter space dimension remains open.

In particular, we obtain polynomial time algorithms for training for a given fixed network architecture.

Our work applies more broadly to empirical risk minimization problems which allows us to generalize various previous results and obtain new complexity results for previously unstudied architectures in the proper learning setting.

Deep Learning has received significant attention due to its impressive performance in many state-of-the-art learning tasks.

Unfortunately, while very powerful, Deep Learning is not well understood theoretically and in particular only recently results for the complexity of training deep neural networks have been obtained.

In this work we show that large classes of deep neural networks with various architectures (e.g., DNNs, CNNs, Binary Neural Networks, and ResNets), activation functions (e.g., ReLUs and leaky ReLUs), and loss functions (e.g., Hinge loss, Euclidean loss, etc) can be trained to near optimality with desired target accuracy using linear programming in time that is exponential in the input data and parameter space dimension and polynomial in the size of the data set; improvements of the dependence in the input dimension are known to be unlikely assuming P N P, and improving the dependence on the parameter space dimension remains open.

In particular, we obtain polynomial time algorithms for training for a given fixed network architecture.

Our work applies more broadly to empirical risk minimization problems which allows us to generalize various previous results and obtain new complexity results for previously unstudied architectures in the proper learning setting.

Deep Learning is a powerful tool for modeling complex learning tasks.

Its versatility allows for nuanced architectures that capture various setups of interest and has demonstrated a nearly unrivaled performance on state-of-the-art learning tasks across many domains.

At the same time, the fundamental behavior of Deep Learning methods is not well understood.

One particular aspect that recently gained significant interest is the computational complexity of training such networks.

The basic training problem is usually formulated as an empirical risk minimization problem (ERM) that can be phrased as DISPLAYFORM0 where is some loss function, DISPLAYFORM1 is an i.i.d.

sample from some data distribution D, and f is a neural network architecture parameterized by φ ∈ Φ with Φ being the parameter space of the considered architecture (e.g., network weights).

The empirical risk minimization problem is solved in lieu of the general risk minimization problem (GRM) min φ ∈Φ E (x,y)∈D [ ( f (x, φ) , y)] which is usually impossible to solve due to the inaccessibility of D. Several works have studied the training problem for specific architectures, both in the proper and improper learning setup.

In the former, the resulting "predictor" obtained from (1) is always of the form f (·,φ) for someφ ∈ Φ, whereas in the latter, the predictor is allowed to be outside the class of functions { f (·, φ) : φ ∈ Φ} as long as it satisfies certain approximation guarantee to the solution of (1)1.

In both cases, all results basically establish trainability in time that is exponential in the network parameters but polynomial in the amount of data.

In this work we complement and significantly extend previous work by providing a principled method to convert the empirical risk minimization problem in (1) associated with the learning problem for various architectures into a linear programming problem (LP) in the proper learning setting.

The obtained linear programming formulations are of size roughly exponential in the input dimension and in the parameter space dimension and linear in the size of the data-set.

This result provides new bounds on the computational complexity of the training problem.

For an overview on Complexity Theory we refer the reader to BID6 .

Our work is most closely related to BID18 , BID27 , and BID5 .

In BID27 the authors show that 1 -regularized networks can be learned improperly in polynomial time in the size of the data (with a possibly exponential architecture dependent constant) for networks with ReLU-like activations (but not actual ReLUs) and an arbitrary number of layers k. These results were then generalized in BID18 to actual ReLU activations.

In both cases the improper learning setup is considered, i.e., the learned predictor is not a neural network itself and the learning problem is solved approximately for a given target accuracy.

In contrast to these works, BID5 considered proper and exact learning however only for k = 2 (i.e., one hidden layer).In relation to these works, we consider the proper learning setup for an arbitrary number of layers k and a wide range of activations, loss functions, and architectures.

As previous works, except BID5 , we consider the approximate learning setup as we are solving the empirical risk minimization problem and we also establish generalization of our so-trained models.

Our approach makes use of BID9 that allows for reformulating non-convex optimization problems with small treewidth and discrete as well as continuous decision variables as an approximate linear programming formulations.

To the best of our knowledge, there are no previous papers that propose LP-based approaches for training neural networks.

There are, however, proposed uses of Mixed-Integer and Linear Programming technology in other aspects of Deep Learning.

Some examples of this include feature visualization BID17 , generating adversarial examples BID15 BID3 BID17 , counting linear regions of a Deep Neural Network BID25 , performing inference BID1 and providing strong convex relaxations for trained neural networks BID2 .

We first establish a general framework that allows us to reformulate (regularized) ERM problems arising in Deep Learning (among others!) into approximate linear programs with explicit bounds on their complexity.

The resulting methodology allows for providing complexity upper bounds for specific setups simply by plugging-in complexity measures for the constituting elements such as layer architecture, activation functions, and loss functions.

In particular our approach overcomes limitations of previous approaches in terms of handling the accuracy of approximations of non-linearities used in the approximation functions to achieve the overall target accuracy.

Principled Training through LPs.

If > 0 is arbitrary, then for any sample size D there exists a dataindependent linear program, i.e., the LP can be written down before seeing the data, with the following properties:1In the case of Neural Networks, for example, improper learning could lead to a predictor that does not correspond to a Neural Network, but that might behave closely to one.

Solving the ERM problem to -optimality.

The linear program describes a polyhedron P such that for every realized data set (X,Ŷ ) DISPLAYFORM0 there is a face FX ,Ŷ ⊆ P such that optimizing certain linear function over FX ,Ŷ solves (1) to -optimality returning a feasible parametrizationφ ∈ Φ which is part of our hypothesis class (i.e., we consider proper learning).

The face FX ,Ŷ ⊆ P is simply obtained from P by fixing certain variables of the linear program using the values of the actual sample; equivalently, by Farkas' lemma, this can be achieved by modifying the objective function to ensure optimization over the face belonging to the data.

As such, the linear program has a build-once-solve-many feature.

We will also show that a possible data-dependent LP formulation is meaningless (see Appendix B).Size of the linear program.

The size, measured as bit complexity, of the linear program is roughly DISPLAYFORM1 where L is a constant depending on , f , and Φ that we will introduce later, n, m are the dimensions of the data points, i.e.,x i ∈ R n andŷ i ∈ R m for all i ∈ [D] , and N is the dimension of the parameter space Φ. The overall learning algorithm is obtained then by formulating and solving the linear program, e.g., with the ellipsoid method whose running time is polynomial in the size of the input BID20 .

Even sharper size bounds can be obtained for specific architectures assuming network structure (see Appendix F) and our approach immediately extends to regularized ERMs (see Appendix E).

It is important to mention that the constant L measures a certain Lipschitzness of the ERM training problem.

While not exactly requiring Lipschitz continuity in the same way, Lipschitz constants have been used before for measuring complexity in the improper learning framework (see BID18 ) and more recently have been shown to be linked to generalization in BID19 .Generalization.

Additionally, we establish that the solutions obtained for the ERM problem via our linear programming approach generalize, utilizing techniques from stochastic optimization.

We also show that using our approach one can obtain a significant improvement on the results of BID18 when approximating the general risk minimization problem.

Due to space limitations, however, we relegate this discussion to Appendix I.Throughout this work we assume both data and parameters to be well-scaled, which is a common assumption and mainly serves to simplify the representation of our results; the main assumption is the reasonable boundedness, which can be assumed without significant loss of generality as actual computations assume boundedness in any case (see also BID22 for arguments advocating the use of normalized coefficients in neural networks).

More specifically, we assume DISPLAYFORM2 We point out three important features of our results.

First, we provide a solution method that has provable optimality guarantees for the ERM problem, ensures generalization, and linear dependency on the data (in terms of the complexity of the LP) without assuming convexity of the optimization problem.

To the best of our knowledge, the only result presenting optimality guarantees in a proper learning, non-convex setting is that of BID5 .

Second, the linear program that we construct for a given sample size D is data-independent in the sense that it can be written down before seeing the actual data realization and as such it encodes reasonable approximations of all possible data sets that can be given as an input to the ERM problem.

This in particular shows that our linear programs are not simply discretizing space: if one considers a discretization of data contained in [−1, 1] n × [−1, 1] m , the total number of possible data sets of size D is exponential in D, which makes the linear dependence on D of the size of our LPs a remarkable feature.

Finally, our approach can be directly extended to handle commonly used regularizers as we show in Appendix E; for ease of presentation though we omit regularizers throughout our main discussions.

Complexity results for various network architectures.

We apply our methodology to various well-known neural network architectures and either generalize previous results or provide completely new results.

We provide an overview of our results in TAB0 , where k is the number of layers, w is width of the network, n/m are the input/output dimensions and N is the total number of parameters.

We use G to denote the directed graph defining the neural network and ∆ the maximum vertex in-degree in G. In all results the node DISPLAYFORM3 computations are linear with bias term and normalized coefficients, and activation functions with Lipschitz constant at most 1 and with 0 as a fixed point; these include ReLU, Leaky ReLU, eLU, Tanh, among others.

We would like to point out that certain improvements in the results in TAB0 can be obtained by further specifying if the ERM problem corresponds to regression or classification.

For example, the choice of loss functions and the nature of the output data y (discrete or continuous) typically rely on this.

We can exploit such features in the construction of the LPs (see the proof of Theorem 3.1) and provide a sharper bound on the LP size.

Nonetheless, these improvements are not especially significant and in the interest of clarity and brevity we prefer to provide a unified discussion on ERM.

Missing proofs have been relegated to the appendix due to space limitations.

The complexity of the training problem for the Fully Connected DNN case is arguably the most studied and, to the best of our knowledge, all training algorithms with approximation or optimality guarantees have a polynomial dependency on D only after fixing the architecture (depth, width, input dimension, etc.).

In our setting, once the architecture is fixed, we obtain a polynomial dependence in both D and 1/ .

Moreover, our results show that in the bounded case one can obtain a training algorithm with polynomial dependence on D across all architectures, assuming very little on the specific details of the network (loss, activation, etc) .

This answers an open question left by BID5 regarding the possibility of a training algorithm with polynomial dependence on D. In addition, we show that a uniform LP can be obtained without compromising that dependence on D.The reader might wonder if the exponential dependence on the other parameters of our LP sizes can be improved, namely the input dimension n + m and the parameter space dimension N (we are ignoring for the moment the exponent involving the depth k, as it will be typically dominated by N).

The dependence on the input dimension is unlikely to be improved due to the NP-hardness result in BID10 , whereas obtaining a polynomial dependence on the parameter space dimension remains open (see BID5 ).

A recent paper BID4 provides an NP-hard DNN training problem that becomes polynomially solvable when the input dimension is fixed.

However, this result considers a fixed architecture, thus the parameter space dimension is a constant and the running time is measured with respect to D.

In the following let [n] {1, . . .

, n} and [n] 0 {0, . . .

, n}. Given a graph H, we will use V(H) and E(H) to denote the vertex-set and edge-set of H, respectively, and δ H (u) will be the set of edges incident to vertex u. We will need:Definition 2.1.

For a function g : K ⊆ R n → R, we denote its Lipschitz constant with respect to the p-norm DISPLAYFORM0 Moreover, in the following let E ω ∈Ω [·] and V ω ∈Ω [·] denote the expectation and variance with respect to the random variable ω ∈ Ω, respectively.

The basic ERM problem is typically of the form (1), where is some loss function, DISPLAYFORM0 is an i.i.d.

sample from some data distribution D that we have reasonable sampling access to, and f is a model that is parametrized by φ ∈ Φ. We consider the proper learning setting here, where the computed solution to the ERM problem has to belong to the hypothesis class induced by Φ; for a detailed discussion see Appendix A.2.

We next define the Lipschitz constant of an ERM problem with respect to the infinity norm.

DISPLAYFORM1 over the domain DISPLAYFORM2 We emphasize that in FORMULA9 we are considering the data-dependent entries as variables as well, and not only the parameters Φ as it is usually done in the literature.

This is because we will construct data-independent LPs, a subtlety that will become clear later.

A neural network can be understood as a function f defined over a directed graph that maps inputs x ∈ R n to f (x) ∈ R m .

The directed graph G = (V, E), which represents the network architecture, often naturally decomposes into layers DISPLAYFORM0 where V 0 is referred to as the input layer and V k as the output layer.

To all other layers we refer to as hidden layers.

These graphs do neither have to be acyclic (as in the case of recurrent neural networks) nor does the layer decomposition imply that arcs are only allowed between adjacent layers (as in the case of ResNets).

In feed-forward networks, however, the graph is assumed to be acyclic.

For the unfamiliar reader we provide a more formal definition in Appendix A.3.

We will introduce the key concepts that we need to formulate and solve binary optimization problems with small treewidth, which will be the main workhorse behind our results.

The treewidth of a graph is a parameter used to measure how tree-like a given graph is.

Among all its equivalent definitions, the one we will use in this work is the following:Definition 2.3.

Let G be an undirected graph.

A tree-decomposition BID24 ) of G is a pair (T, Q) where T is a tree and Q = {Q t : t ∈ V(T)} is a family of subsets of V(G) such that DISPLAYFORM0 The width of the decomposition is defined as max {|Q t | : t ∈ V(T)} − 1.

The treewidth of G is the minimum width over all tree-decompositions of G.We refer to the Q t as bags as customary.

An example of a tree-decomposition is given in FIG0 in Appendix A.1.

In addition to width, another important feature of a tree-decomposition (T, Q) we use is the size of the tree-decomposition given by |V(T)|.Consider a problem of the form DISPLAYFORM1 where the f i and g j are arbitrary functions that we access via a function value oracle, i.e., an oracle that returns the function upon presentation with an input.

We will further use the concept of intersection graph.

Definition 2.4.

The intersection graph Γ[I] for an instance I of BO is the undirected graph which has a vertex for each x variable and an edge for each pair of x variables that appear in any common constraint.

Note that in the above definition we have ignored the y variables which will be of great importance later.

The sparsity of a problem is now given by the treewidth of its intersection graph and we obtain: Theorem 2.5 is an immediate generalization of a theorem in BID9 distinguishing the variables y, which do not need to be binary in nature, but are fully determined by the binary variables x. The proof is omitted as it is almost identical to the proof in BID9 .

For the sake of completeness, we include a proof sketch in Appendix C.1.

We will now show how we can obtain an approximate LP formulation for the ERM problem.

A notable feature is that our LP formulation is data-independent in the sense that we can write down the LP, for a given sample size D, before having seen the actual data; the LP is later specialized to a given data set by fixing some of its variables.

This subtlety is extremely important as it prevents trivial solutions, where some non-deterministic guess provides a solution to the ERM problem for a given data set and then simply writes down a small LP that outputs the network configuration; such an LP would be of small size (the typical notion of complexity used for LPs) however not efficiently computable.

By making the construction independent of the data we circumvent this issue; we provide a short discussion in Appendix B and refer the interested reader to BID13 BID12 ; BID11 for an in-depth discussion of this subtlety.

Slightly simplifying, we might say for now that in the same way we do not want algorithms to be designed for a fixed data set, we do not want to construct LPs for a specific data set but for a wide range of data sets.

As mentioned before, we assume DISPLAYFORM0 , 1] m as normalization to simplify the exposition.

Since the BO problem only considers linear objective functions, we begin by reformulating the ERM problem (1) in the following equivalent form: DISPLAYFORM1

Motivated by this reformulation, we study an approximation to the following set: DISPLAYFORM0 The variables DISPLAYFORM1 denote the data variables, that will be assigned values upon a specification of a data set of sample size D.Let r ∈ R with −1 ≤ r ≤ 1.

Given γ ∈ (0, 1) we can approximate r as a sum of inverse powers of 2, within additive error proportional to γ.

For N γ log 2 γ −1 there exist values z h ∈ {0, 1} with h ∈ [N γ ], so that DISPLAYFORM2 Our strategy is now to approximately represent the x, y, φ variables via these binary approximations, i.e., as FORMULA9 , and consider the following approximation of S(D, Φ, , f ): DISPLAYFORM3 DISPLAYFORM4 We can readily describe the error of the approximation of S(D, Φ, , f ) by S (D, Φ, , f ) in the ERM problem (1) induced by the discretization: DISPLAYFORM5 By substituting out the x, y, φ by means of the equations of S (D, Φ, , f ), we obtain a feasible region as BO.

So far, we have phrased the ERM problem (1) in terms of a binary optimization problem using a discretization of the continuous variables.

This in and of itself is neither insightful nor useful.

In this section we will perform the key step, reformulating the convex hull of S (D, Φ, , f ) as a moderate-sized linear program by means of Theorem 2.5 exploiting small treewidth of the ERM problem.

After replacing the (x, y, φ) variables in S (D, Φ, , f ) using the z variables, we can see that the intersection graph of S (D, Φ, , f ) is given by Figure 1a , where we use (x, y, φ) as stand-ins for corresponding the binary variables z x , z y , z φ .

Recall that the intersection graph does not include the L variables.

It is not hard to see that a valid tree-decomposition for this graph is given by Figure 1b .

This tree-decomposition has size D and width N γ (n + m + N) − 1 (much less than the N γ (N + Dn + Dm) binary variables).

This yields our main theorem: Proof.

The proof of part (a) follows directly from Theorem 2.5 using N γ = log(2L/ ) along with the tree-decomposition of Figure 1b , which implies |V(T )| + p = 2D in this case.

Parts (b), (c) and (d) rely on the explicit construction for Theorem 2.5 and they are given in Appendix D. DISPLAYFORM0 DISPLAYFORM1 Observe that equivalently, by Farkas' lemma, optimizing over the face can be also achieved by modifying the objective function in a straightforward way.

Also note that the number of evaluations of and f is independent of D. We would like to further point out that we can provide an interesting refinement of the theorem from above: if Φ has an inherent network structure (as in the case of Neural Networks) one can exploit treewidth-based sparsity of the network itself in order to obtain a smaller linear program with the same approximation guarantees as before.

This allows us to reduce the exponent in the exponential term of the LP size to an expression that depends on the sparsity of the network, instead of its size.

For brevity of exposition, we relegate this discussion to Appendix F. Another improvement can be obtained by using more information about the input data.

Assuming extra structure on the input, one could potentially improve the n + m exponent on the LP size.

We relegate the discussion of this feature to Remark D.1 in the Appendix, as it requires the explicit construction of the LP, which we also provide in the Appendix.

DISPLAYFORM2 where σ is the ReLU activation function σ(x) max{0, x} applied component-wise and each T i : DISPLAYFORM3 is an affine linear function.

Here w 0 = n is the dimension of the input data and w k = m is the dimension of the output of the network.

We write DISPLAYFORM4 normalization.

Thus, if v is a node in layer i, the node computation performed in v is of the formâ T z +b, whereâ is a row of A i andb is a component of b i .

Note that in this case the dimension of the parameter space Φ is exactly the number of edges of the network.

Hence, we use N to represent the number of edges as well.

We begin with a short technical Lemma, with which we can immediately establish the following corollary.

DISPLAYFORM5 DISPLAYFORM6 Proof.

Proving that the architecture Lipschitz constant is L ∞ ( )w O(k 2 ) suffices.

Note that all node computations take the form h(z, a, b) = z T a + b for a ∈ [−1, 1] w and b ∈ [−1, 1].

The only difference is made in the domain of z, which varies from layer to layer.

The 1-norm of the gradient of h is at most z 1 + a 1 + 1 ≤ z 1 + w + 1 which, in virtue of Lemma 4.1, implies that a node computation on layer i (where the weights are considered variables as well) has Lipschitz constant at most DISPLAYFORM7 , which shows that the Lipschitz constants can be multiplied layer-by-layer to obtain the overall architecture Lipschitz constant.

Since ReLUs have Lipschitz constant equal to 1, and DISPLAYFORM8 The reader might have noticed that a sharper bound for the Lipschitz constant above could have been used, however we chose simpler bounds for the sake of presentation.

It is worthwhile to compare the previous lemma to the following closely related result.

Remark 4.4.

We point out a few key differences of this result with the algorithm we can obtain from solving the LP in Corollary 4.2: (a) One advantage of our result is the benign dependency on D. An algorithm that solves the training problem using our proposed LP has polynomial dependency on the data-size regardless of the architecture.

(b) As we have mentioned before, our approach is able to construct an LP before seeing the data.

(c) The dependency on w of our algorithm is also polynomial.

To be fair, we are including an extra parameter N-the number of edges of the Neural Network-on which the size of our LP depends exponentially.

(d) We are able to handle any output dimension m and any number of layers k. (e) We do not assume convexity of the loss function , which causes the resulting LP size to depend on how well behaved is in terms of its Lipschitzness.

(f) The result of BID5 has two advantages over our result: there is no boundedness assumption on the coefficients, and they are able to provide a globally optimal solution instead of an -approximation.

Another interesting point can be made with respect to Convolutional Neural Networks (CNN).

In these, convolutional layers are included which help to significantly reduce the number of parameters involved in the neural network.

From a theoretical perspective, a CNN can be obtained by simply enforcing certain parameters of a fully-connected DNN to be equal.

This implies that Lemma 4.5 can also be applied to CNNs, with the key difference residing in parameter N, which is the dimension of the parameter space and does not correspond to the number of edges in a CNN.

In TAB0 we provide explicit LP sizes for common architectures.

These results can be directly obtained from Lemma 4.5, using the specific Lipschitz constants of the loss functions.

We provide explicit computations in Appendix G.

We have presented a novel framework which shows that training a wide variety of neural networks can be done in time which depends polynomially on the data set size, while satisfying a predetermined arbitrary optimality tolerance.

Our approach is realized by approaching training through the lens of linear programming.

Moreover, we show that training using a particular data set is closely related to the face structure of a data-independent polytope.

Our contributions not only improve the best known algorithmic results for neural network training with optimality/approximation guarantees, but also shed new light on (theoretical) neural network training by bringing together concepts of graph theory, polyhedral geometry, and non-convex optimization as a tool for Deep Learning.

While the LPs we are constructing are large, and likely to be difficult to solve by straightforward use of our formulation, we strongly believe the theoretical foundations we lay here can also have practical implications in the Machine Learning community.

First of all, we emphasize that all our architecture dependent terms are worst-case bounds, which can be improved by assuming more structure in the corresponding problems.

Additionally, the history of Linear Programming has provided many interesting cases of extremely large LPs that can be solved to near-optimality without necessarily generating the complete LP description.

In these cases the theoretical understanding of the LP structure is crucial to drive the development of incremental solution strategies.

Finally, we would like to point out an interesting connection between the way our approach works and the current most practically effective training algorithm: stochastic gradient descent.

Our LP approach implicitly "decomposes" the problem for each data point, and the LP merges them back together without losing any information nor optimality guarantee, even in the non-convex setting.

This is the core reason why our LP has a linear dependence on D, and bears close resemblance to SGD where single data points (or batches of those) are used in a given step.

As such, our results might provide a new perspective, through low treewidth, on why the current practical algorithms work so well, and perhaps hints at a synergy between the two approaches.

We believe this can be an interesting path to bring our ideas to practice.

structure.

An alternative definition to Definition 2.3 of treewidth that the reader might find useful is the following; recall that a chordal graph is a graph where every induced cycle has length exactly 3.

Definition A.1.

An undirected graph G = (V, E) has treewidth ≤ ω if there exists a chordal graph H = (V, E ) with E ⊆ E and clique number ≤ ω + 1.H in the definition above is sometimes referred to as a chordal completion of G. In FIG0 we present an example of a graph and a valid tree-decomposition.

The reader can easily verify that the conditions of Definition 2.3 are met in this example.

Moreover, using Definition A.1 one can verify that the treewidth of the graph in FIG0 is exactly 2.Two important folklore results we use in Section C.1 and Section F are the following.

Then there exists t ∈ T such that K ⊆ Q t .

An important distinction is the type of solution to the ERM that we allow.

In proper learning we require the solution to satisfy φ ∈ Φ, i.e., the model has to be from the considered model class induced by Φ and takes the form f (·, φ * ) for some φ * ∈ Ω, with DISPLAYFORM0 and this can be relaxed to -approximate (proper) learning by allowing for an additive error > 0 in the above.

In contrast, in improper learning we allow for a model g(·), that cannot be obtained as f (·, φ) with φ ∈ Φ, DISPLAYFORM1 with a similar approximate version.

As we mentioned in the main body, this article considers the proper learning setup.

In a Neural Network, the graph G defining the network can be partitioned in layers.

This means that V(G) = k i=0 V i for some sets V i -the layers of the network.

Each vertex v ∈ V i with i ∈ [k] 0 has an associated set of in-nodes denoted by δ + (v) ⊆ V, so that (w, v) ∈ E for all w ∈ δ + (v) and an associated set of out-nodes δ − (v) ⊆ V defined analogously.

If i = 0, then δ + (v) are the inputs (from data) and if i = k, then δ − (v) are the outputs of the network.

Moreover, each node v ∈ V performs a node computation g i (δ + (v)), where g i : R |δ + (v) | → R with i ∈ [k] is typically a smooth function (often these are linear or affine linear functions) and then the node activation is computed as a i (g i (δ + (v))), where a i : R → R with i ∈ [k] is a (not necessarily smooth) function (e.g., ReLU activations of the form a i (x) = max{0, x}) and the value on all out-nodes w ∈ δ − (v) is set to a i (g i (δ + (v))) for nodes in layer i ∈ [k].

In feed-forward networks, we can further assume that if v ∈ V i , then δ + (v) ⊆ ∪ i−1 j=0 V j , i.e., all arcs move forward in the layers.

As mentioned before, the assumption that the construction of the LP is independent of the specific data is important and reasonable as it basically prevents us from constructing an LP for a specific data set, which would be akin to designing an algorithm for a specific data set in ERM problem (1).

To further illustrate the point, suppose we would do the latter, then a correct algorithm would be a simple print statement of the optimal configurationφ.

Clearly this is nonsensical and we want the algorithm to work for all types of data sets as inputs.

We have a similar requirement for the construction of the LP, with the only difference that number of data points D has to be known at time of construction.

As such LPs more closely resemble a circuit model of computation (similar to the complexity class P/poly); see BID13 BID12 ; Braun and Pokutta (2018+) for details.

The curious reader might still wonder how our main result changes if we allow the LPs in Theorem 3.1 to be data-dependent, i.e., if we construct a specific linear program after we have seen the data set: Remark B.1.

To obtain a data-dependent linear program we can follow the same approach as in Section 3 and certainly produce an LP that will provide the same approximation guarantees for a fixed data set.

This result is not particularly insightful, as it is based on a straight-forward enumeration which takes a significant amount of time, considering that it only serves one data set.

On the other hand, our result shows that by including the input data as a variable, we do not induce an exponential term in the size of the data set D and we can keep the number function evaluations to be roughly the same.

Our approach shares some similarities with stochastic gradient descent (SGD) based training: data points are considered separately (or in small batches) and the method (in case of SGD) or the LP (in our case) ensure that the information gained from a single data point is integrated into the overall ERM solution.

In the case of SGD this happens through sequential updates of the form x t+1 ← x t − η∇ f i (x t ), where i is a random function corresponding to a training data point (X i ,Ŷ i ) from the ERM problem.

In our case, it is the LP that 'glues together' solutions obtained from single training data points by means of leveraging the low treewidth.

This is reflected in the linear dependence in D in the problem formulation size.

Proof of Lemma 3.1.

Choose binary valuesz so as to attain the approximation for variables x, y, φ as in FORMULA18 and definex,ŷ,φ,L fromz according to the definition of S (D, Φ, , f ).

Since DISPLAYFORM0 by Lipschitzness we obtain |L d −L d | ≤ .

The result then follows.

Proof of Lemma 4.1.

The result can be verified directly, since for a ∈ [−1, 1] w and b ∈ [−1, 1] it holds |z T a + b| ≤ w z ∞ + 1.Proof of Lemma 4.5.

The proof follows almost directly from the proof of Corollary 4.2.

The two main differences are (1) the input dimension of a node computation, which can be at most ∆ instead of w and FORMULA9 the fact that an activation function a with Lipchitz constant 1 and that a(0) = 0 satisfies |a(z)| ≤ |z|, thus the domain of each node computation computed in Lemma 4.1 applies.

The layer-by-layer argument can be applied as the network is feed-forward.

DISPLAYFORM1 Let us recall the definition of BO: DISPLAYFORM2 We sketch the proof of Proof.

Since the support of each f i induces a clique in the intersection graph, there must exist a bag Q such that supp( f i ) ⊆ Q (Lemma A.3).

The same holds for each g j .

We modify the tree-decomposition to include the y j variables the following way:• For each j ∈ [p], choose a bag Q containing supp(g j ) and add a new bag Q ( j) consisting of Q ∪ {y j } and connected to Q.• We do this for every j ∈ [p], with a different Q ( j) for each different j.

This creates a new tree-decomposition (T , Q ) of width at most ω + 1, which has each variable y j contained in a single bag Q ( j) which is a leaf.• The size of the tree-decomposition is |T | = |T | + p.

From here, we proceed as follows:• For each t ∈ T , if Q t y j for some j ∈ [p], then we construct DISPLAYFORM3 otherwise we simply construct DISPLAYFORM4 Note that these sets have size at most 2 |Q t | .•

We define variables X[Y, N] where Y, N form a partition of Q t 1 ∩ Q t 2 .

These are at most 2 ω |V(T )|.• For each t ∈ T and v ∈ F t , we create a variable λ v .

These are at most 2 ω |V(T )|.We formulate the following linear optimization problem DISPLAYFORM5 Note that the notation in the last constraint is justified since by construction supp(g j ) ⊆ Q ( j).

The proof of the fact that LBO is equivalent to BO follows from the arguments in BID9 .

The key difference justifying the addition of the y variables relies in the fact that they only appear in leaves of the tree decomposition (T , Q ), and thus in no intersection of two bags.

The gluing argument using variables X[Y, N] then follows directly, as it is then only needed for the x variables to be binary.

We can substitute out the x and y variables and obtain an LP whose variables are only λ v and X [Y, N] .

This produces an LP with at most 2 · 2 ω |V(T )| variables and (2 · 2 ω + 1)|V(T )| constraints.

This proves the size of the LP is O(2 ω (|V(T)| + p)) as required.

DISPLAYFORM6 In this Section we show how to construct the polytope in Theorem 3.1.

We first recall the following definition: DISPLAYFORM7 and recall that S (D, Φ, , f ) is a discretized version of the set mentioned above.

From the tree-decomposition detailed in Section 3.2, we see that data-dependent variables x, y, L are partitioned in different bags for each DISPLAYFORM8 .

Let us index the bags using d. Since all data variables have the same domain, the sets F d we construct in the proof of Theorem 2.5 will be the same for all d ∈ [D] .

Using this observation, we can construct the LP as follows:1.

Fix, say, d = 1 and enumerate all binary vectors corresponding to the discretization of The only evaluations of and f are performed in the construction of F 1 .

As for the additional computations, the bottleneck lies in creating all λ variables, which takes time O((2L/ ) n+m+N D).

Remark D.1.

Note that in step 1 of the LP construction we are enumerating all possible discretized values of x 1 , y 1 , i.e., we are implicitly assuming all points in [−1, 1] n+m are possible inputs.

This is reflected in the (2L/ ) n+m term in the LP size estimation.

If one were to use another discretization method (or a different "point generation" technique) using more information about the input data, this term could be improved and the explicit exponential dependency on the input dimension of the LP size could be alleviated significantly.

However, note that in a fully-connected neural network we have N ≥ n + m and thus an implicit exponential dependency on the input dimension could remain unless more structure is assumed.

This is in line with the NP-hardness results.

We leave the full development of this potential improvement for future work.

DISPLAYFORM9 DISPLAYFORM10 and let φ * be an optimal solution to the ERM problem with input data (X,Ŷ ).

Consider now binary variables zx, zŷ to attain the approximation (6) and definex,ỹ from zx, zŷ, i.e., DISPLAYFORM11 and similarly forỹ.

Define the set DISPLAYFORM12 and similarly as before define S (X,Ỹ, Φ, , f ) to be the discretized version (on variables φ).

The following Lemma shows the quality of approximation to the ERM problem obtained using S(X,Ỹ, Φ, , f ) and subsequently S (X,Ỹ, Φ, , f ).

DISPLAYFORM13 Proof.

The first inequality follows from the same proof as in Lemma 3.1.

For the second inequality, let φ be the binary approximation to φ, and L defined by DISPLAYFORM14 Sincex,ỹ, φ are approximations to x,ŷ, φ, by Lipschitzness we know that DISPLAYFORM15 Proof.

Sinceφ ∈ Φ, and φ * is a "true" optimal solution to the ERM problem, we immediately have DISPLAYFORM16 On the other hand, by the previous Lemma we know there exists DISPLAYFORM17 Note that since the objective is linear, the optimization problem in the previous Corollary is equivalent if we replace S (X,Ỹ, Φ, , f ) by its convex hull.

Therefore the only missing link to the face property of the data-independent polytope is the following: DISPLAYFORM18 Proof.

The proof follows from simply fixing variables in the corresponding LBO that describes DISPLAYFORM19 and v ∈ F d , we simply need to make λ v = 0 whenever the (x, y) components of v do not correspond toX,Ỹ .

We know this is well defined, sinceX,Ỹ are already discretized, thus there must be some v ∈ F d corresponding to them.

The structure of the resulting LP is the same as LBO, so the fact that it is exactly conv(S (X,Ỹ, Φ, , f )) follows.

The fact that it is a face of conv(S (D, Φ, , f )) follows from the fact that the procedure simply fixed some inequalities to be tight.

A common practice to avoid over-fitting is the inclusion of regularizer terms in (1).

This leads to problems of the form DISPLAYFORM0 where R(·) is a function, typically a norm, and λ > 0 is a parameter to control the strength of the regularization.

Regularization is generally used to promote generalization and discourage over-fitting of the obtained ERM solution.

The reader might notice that our arguments in Section 3 regarding the epigraph reformulation of the ERM problem and the tree-decomposition of its intersection graph can be applied as well, since the regularizer term does not add any extra interaction between the data-dependent variables.

The previous analysis extends immediately to the case with regularizers after appropriate modification of the architecture Lipschitz constant L to include R(·).Definition E.1.

Consider a regularized ERM problem (9) with parameters D, Φ, , f , R, and λ.

We define its DISPLAYFORM1 over the domain DISPLAYFORM2

So far we have considered general ERM problems exploiting only the structure of the ERM induced by the finite sum formulations.

We will now study ERM under Network Structure, i.e., specifically ERM problems as they arise in the context of Neural Network training.

We will see that in the case of Neural Networks, we can exploit the sparsity of the network itself to obtain better LP formulations of conv(S (D, Φ, , f )).Suppose the network is defined by a graph G, and recall that in this case, Φ ⊆ [−1, 1] E(G) .

By using additional auxiliary variables s representing the node computations and activations, we can describe S(D, Φ, , f ) in the following way: DISPLAYFORM0 The only difference with our original description of S(D, Φ, , f ) in (5) is that we explicitly "store" node computations in variables s. These new variables will allow us to better use the structure of G. Assumption F.1.

To apply our approach in this context we need to further assume Φ to be the class of Neural Networks with normalized coefficients and bounded node computations.

This means that we restrict to the case when s ∈ [−1, 1] |V (G) |D .Under Assumption F.1 we can easily derive an analog description of S (D, Φ, , f ) using this node-based representation of S (D, Φ, , f ).

In such description we also include a binary representation of the auxiliary variables s. Let Γ be the intersection graph of such a formulation of S (D, Φ, , f ) and Γ φ be the sub-graph of Γ induced by variables φ.

Using a tree-decomposition (T, Q) of Γ φ we can construct a tree-decomposition of Γ the following way: DISPLAYFORM1 is a copy of (T, Q).2.

We connect the trees T i in a way that the resulting graph is a tree (e.g., they can be simply concatenated one after the other).

It is not hard to see that this is a valid tree-decomposition of Γ, of size |T | · D -since the bags were duplicated D times-and width N γ (tw(Γ φ ) + |V(G)| + n + m).We now turn to providing a bound to tw(Γ φ ).

To this end we observe the following:1.

The architecture variables φ are associated to edges of G. Moreover, two variables φ e , φ f , with e, f ∈ E appear in a common constraint if and only if there is a vertex v such that e, f ∈ δ + (v).

2.

This implies that Γ φ is a sub-graph of the line graph of G. Recall that the line graph of a graph G is obtained by creating a node for each edge of G and connecting two nodes whenever the respective edges share a common endpoint.

The treewidth of a line graph is related to the treewidth of the base graph (see BID8 BID14 BID7 ; BID21 DISPLAYFORM2

In Section 4 we specified our results -the size of the data-independent LPs-for feed-forward networks with 1-Lipschitz activation functions.

However, we kept as a parameter L ∞ ( ); the Lipschitz constant of (·, ·) over DISPLAYFORM0 w j a valid bound on the output of the node computations, as proved in Lemma 4.1.

Note that U k ≤ w k+1 .In this Section we compute this Lipschitz constant for various common loss functions.

It is important to mention that we are interested in the Lipschitznes of with respect to both the output layer and the data-dependent variables as well -not a usual consideration in the literature.

Note that a bound on the Lipschitz constant L ∞ ( ) is given by sup z,y ∇ (z, y) 1 .• Quadratic Loss (z, y) = z − y 2 2 .

In this case it is easy to see that DISPLAYFORM1 • Absolute Loss (z, y) = z − y 1 .

In this case we can directly verify that the Lipschitz constant with respect to the infinity norm is at most 2m.• Cross Entropy Loss with Soft-max Layer.

In this case we include the Soft-max computation in the definition of , therefore DISPLAYFORM2 where S(z) is the Soft-max function defined as DISPLAYFORM3 which in principle cannot be bounded.

Nonetheless, since we are interested in the domain DISPLAYFORM4 • Hinge Loss (z, y) = max{1 − z T x, 0}. Using a similar argument as for the Quadratic Loss, one can easily see that the Lipschitz constant with respect to the infinity norm is at most m(U k + 1) ≤ m(w k+1 + 1).

A Binarized activation unit (BiU) is parametrized by p + 1 values b, a 1 , . . .

, a p .

Upon a binary input vector z 1 , z 2 , . . .

, z p the output is binary value y defined by: DISPLAYFORM0 , and y = 0 otherwise.

Now suppose we form a network using BiUs, possibly using different values for the parameter p.

In terms of the training problem we have a family of (binary) vectors x 1 , . . .

, x D in R n and binary labels and corresponding binary label vectors y 1 , . . . , y D in R m , and as before we want to solve the ERM problem (1).

Here, the parametrization φ refers to a choice for the pair (a, b) at each unit.

In the specific case of a network with 2 nodes in the first layer and 1 node in the second layer, and m = 1, BID10 showed that it is NP-hard to train the network so as to obtain zero loss, when n = D. Moreover, the authors argued that even if the parameters (a, b) are restricted to be in {−1, 1}, the problem remains NP-Hard.

See BID16 for an empirically efficient training algorithm for BiUs.

In this section we apply our techniques to the ERM problem (1) to obtain an exact polynomial-size dataindependent formulation for each fixed network (but arbitrary D) when the parameters (a, b) are restricted to be in {−1, 1}.We begin by noticing that we can reformulate (1) using an epigraph formulation as in (4).

Moreover, since the data points in a BiU are binary, if we keep the data points as variables, the resulting linear-objective optimization problem is a binary optimization problem as BO.

This allows us to claim the following: Proof.

The result follows from applying Theorem 2.5 directly to the epigraph formulation of BiU keeping x and y as variables.

In this case an approximation is not necessary.

The construction time and the data-independence follow along the same arguments used in the approximate setting before.

DISPLAYFORM1 , where is some loss function, f is a neural network architecture with parameter space Φ, and (x, y) ∈ R n+m drawn from the distribution D. We solve the finite sum problem, i.e., the empirical risk minimization problem DISPLAYFORM2 We will show in this section, for any 1 > α > 0, > 0, we can choose a (reasonably small!) sample size D, so that with probability 1 − α it holds: DISPLAYFORM3 As the size of the linear program that we use for training only linearly depends on the number of data points, this also implies that we will have a linear program of reasonable size as a function of α and .The following proposition summaries the generalization argument used in stochastic programming as presented in Ahmed (2017) (see also BID26 F(x, γ i )

.Ifx ∈ X is an -approximate solution to (11), i.e., , with L and σ 2 as above, then with probability 1 − α it holds GRM(φ) ≤ min φ ∈Φ GRM(φ) + 6 , i.e.,φ is a 6 -approximate solution to min φ ∈Φ GRM(φ).

A closely related result regarding an approximation to the GRM problem for neural networks is provided by BID18 in the improper learning setting.

The following corollary to (Goel et al., 2017, Corollary 4.5) can be directly obtained, rephrased to match our notation:Theorem I.4 BID18 ).

There exists an algorithm that outputsφ such that with probability 1 − α, DISPLAYFORM4 Remark I.5.

In contrast to the above result of BID18 , note that in our paper we consider the proper learning setting, where we actually obtain a neural network.

In addition we point out several key differences between Theorem I.4 and the algorithmic version of our result when solving the LP in Corollary I.3 of size as FORMULA1 : (a) In FORMULA1 , the dependency on the input dimension is better than in (12).

(b) The dependency on the Lipschitz constant is significantly better in (12), although we have to point out that we are relying on the Lipschitz constant with respect to all inputs of the loss function and in a potentially larger domain.

(c) The dependency on is also better in (12).

(d) We are not assuming convexity of and we consider general m.(e) The dependency on k in FORMULA1 is much more benign than the one in (13), which is doubly exponential.

Remark I.6.

Since the first submission of this article, a manuscript by BID23 was published which extended the results by BID18 .

This work provides an algorithm with similar characteristics to the one by BID18 but in the proper learning setting, for depth-2 ReLU networks with convex loss functions.

The running time of the algorithm (rephrased to match our notation) is (n/α) O(1) 2 (w/ ) O(1) .

Analogous to the comparison in Remark I.5, we obtain a much better dependence with respect to and we do not rely on convexity of the loss function or on constant depth of the neural network.

<|TLDR|>

@highlight

Using linear programming we show that the computational complexity of approximate Deep Neural Network training depends polynomially on the data size for several architectures