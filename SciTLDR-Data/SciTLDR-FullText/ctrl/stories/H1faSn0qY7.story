We present DL2, a system for training and querying neural networks with logical constraints.

The key idea is to translate these constraints into a differentiable loss with desirable mathematical properties and to then either train with this loss in an iterative manner or to use the loss for querying the network for inputs subject to the constraints.

We empirically demonstrate that DL2 is effective in both training and querying scenarios, across a range of constraints and data sets.

With the success of neural networks across a wide range of important application domains, a key challenge that has emerged is that of making neural networks more reliable.

Promising directions to address this challenge are incorporating constraints during training BID14 BID16 and inspecting already trained networks by posing specific queries BID7 BID21 BID27 ).

While useful, these approaches are described and hardcoded to particular kinds of constraints, making their application to other settings difficult.

Inspired by prior work (e.g., BID3 ; BID5 ; BID10 ; BID0 ), we introduce a new method and system, called DL2 (acronym for Deep Learning with Differentiable Logic), which can be used to: (i) query networks for inputs meeting constraints, and (ii) train networks to meet logical specifications, all in a declarative fashion.

Our constraint language can express rich combinations of arithmetic comparisons over inputs, neurons and outputs of neural networks using negations, conjunctions, and disjunctions.

Thanks to its expressiveness, DL2 enables users to enforce domain knowledge during training or interact with the network in order to learn about its behavior via querying.

DL2 works by translating logical constraints into non-negative loss functions with two key properties: (P1) a value where the loss is zero is guaranteed to satisfy the constraints, and (P2) the resulting loss is differentiable almost everywhere.

Combined, these properties enable us to solve the problem of querying or training with constraints by minimizing a loss with off-the-shelf optimizers.

Training with DL2 To make optimization tractable, we exclude constraints on inputs that capture convex sets and include them as constraints to the optimization goal.

We then optimize with projected gradient descent (PGD), shown successful for training with robustness constraints BID14 .

The expressiveness of DL2 along with tractable optimization through PGD enables us to train with new, interesting constraints.

For example, we can express constraints over probabilities which are not explicitly computed by the network.

Consider the following:∀x.

p θ people (x) < ∨ p θ people (x) > 1 − This constraint, in the context of CIFAR-100, says that for any network input x (network is parameterized by θ), the probability of people (p people ) is either very small or very large.

However, CIFAR-100 does not have the class people, and thus we define it as a function of other probabilities, in particular: p people = p baby + p boy + p girl + p man + p woman .

We show that with a similar constraint (but with 20 classes), DL2 increases the prediction accuracy of CIFAR-100 networks in the semi-supervised setting, outperforming prior work whose expressiveness is more restricted.

DL2 can capture constraints arising in both, classification and regression tasks.

For example, GalaxyGAN BID22 , a generator of galaxy images, requires the network to respect constraints imposed by the underlying physical systems, e.g., flux: the sum of input pixels should equal the sum of output pixels.

Instead of hardcoding such a constraint into the network in an ad hoc way, with DL2, this can now be expressed declaratively: sum(x) = sum(GalaxyGAN(x)).Global training A prominent feature of DL2 is its ability to train with constraints that place restrictions on inputs outside the training set.

Prior work on training with constraints (e.g., BID27 ) focus on the given training set to locally train the network to meet the constraints.

With DL2, we can, for the first time, query for inputs which are outside the training set, and use them to globally train the network.

Previous methods that trained on examples outside the training set were either tailored to a specific task BID14 or types of networks BID16 .

Our approach splits the task of global training between: (i) the optimizer, which trains the network to meet the constraints for the given inputs, and (ii) the oracle, which provides the optimizer with new inputs that aim to violate the constraints.

To illustrate, consider the following Lipshcitz condition: DISPLAYFORM0 Here, for two inputs from the training set (x 1 , x 2 ), any point in their -neighborhood (z 1 , z 2 ) must satisfy the condition.

This constraint is inspired by recent works (e.g., BID8 ; BID1 ) which showed that neural networks are more stable if satisfying the Lipschitz condition.

Querying with DL2 We also designed an SQL-like language which enables users to interact with the model by posing declarative queries.

For example, consider the scenarios studied by a recent work BID23 where authors show how to generate adversarial examples with ACGANs BID17 .

The generator is used to create images from a certain class (e.g., 1) which fools a classifier (to classify as, e.g., 7).

With DL2, this can be phrased as: DISPLAYFORM1 where n i n [-1, 1], c l a s s (M_NN1(M_ACGAN_G(n, 1))) = 7 r e t u r n M_ACGAN_G (n, 1) This query aims to find an input n ∈ R 100 to the generator satisfying two constraints: its entries are between −1 and 1 (enforcing a domain constraint) and it results in the generator producing an image, which it believes to be classified as 1 (enforced by M_ACGAN_G(n, 1)) but is classified by the network (M_NN1) as 7.

DL2 automatically translates this query to a DL2 loss and optimizes it with an off-the-shelf optimizer (L-BFGS-B) to find solutions, in this case, the image to the right.

Our language can naturally capture many prior works at the declarative level, including finding neurons responsible for a given prediction BID18 , inputs that differentiate two networks BID21 , and adversarial example generation (e.g., BID24 ).

The DL2 system is based on the following contributions:• An approach for training and querying neural networks with logical constraints based on translating these into a differentiable loss with desirable properties ((P1) and (P2)).• A training procedure which extracts constraints on inputs that capture convex sets and includes them as PGD constraints, making optimization tractable.• A declarative language for posing queries over neural network's inputs, outputs, and internal neurons.

Queries are compiled into a differentiable loss and optimized with L-BFGS-B.• An extensive evaluation demonstrating the effectiveness of DL2 in querying and training neural networks.

Among other experimental results, we show for the first time, the ability to successfully train networks with constraints on inputs not in the training set.

Adversarial example generation BID21 BID7 can be seen as a fixed query to the network, while adversarial training BID14 aims to enforce a specific constraint.

Most works aiming to train networks with logic impose soft constraints, often through an additional loss BID20 BID27 ; BID15 shows that hard constraints have no empirical advantage over soft constraints.

Probabilistic Soft Logic (PSL) BID11 translates logic into continuous functions over [0, 1] .

As we show, PSL is not amenable to gradient-based optimization as gradients may easily become zero.

BID10 builds on PSL and presents a teacher-student framework which distills rules into the training phase.

The idea is to formulate rule satisfaction as a convex problem with a closed-form solution.

However, this formulation is restricted to rules over random variables and cannot express rules over probability distributions.

In contrast, DL2 can express such constraints, e.g., p 1 > p 2 , which requires the network probability for class 1 is greater than for 2.

Also, the convexity and the closed-form solution stem from the linearity of the rules in the network's output, meaning that non-linear constraints (e.g., Lipschitz condition, expressible with DL2) are fundamentally beyond the reach of this method.

The work of BID27 is also restricted to constraints over random variables and is intractable for complicated constraints.

BID5 reduces the satisfiability of floating-point formulas into numerical optimization, however, their loss is not differentiable and they do not support constraints on distributions.

Finally, unlike DL2, no prior work supports constraints for regression tasks.

We now present our constraint language and show how to translate constraints into a differentiable loss.

To simplify presentation, we treat all tensors as vectors with matching dimensions.

Logical Language Our language consists of quantifier-free constraints which can be formed with conjunction (∧), disjunction (∨) and negation (¬).

Atomic constraints (literals) are comparisons of terms (here ∈ {=, =, ≤, <, ≥, >}).

Comparisons are defined for scalars and applied elementwise on vectors.

A term t is: (i) A variable z or a constant c , representing real-valued vectors; constants can be samples from the dataset.(ii) An expression over terms, including arithmetic expressions or function applications f : R m → R n , for m, n ∈ Z + .

Functions can be defined overvariables, constants, and network parameters θ 1 , . . .

, θ l .

Functions can be the application of a network with parameters θ, the application of a specific neuron, or a computation over multiple networks.

The only assumption on functions is that they are differentiable (almost everywhere) in the variables and network parameters.

We write DISPLAYFORM0 to emphasize the variables, constants, and network parameters that t can be defined over (that is, t may refer to only a subset of these symbols).

We sometimes omit the constants and network parameters (which are also constant) and abbreviate variables byz, i.e., we write t(z).

Similarly, we write ϕ(z) to denote a constraint defined over variablesz.

When variables are not important, we write ϕ.Translation into loss Given a formula ϕ, we define the corresponding loss L(ϕ) recursively on the structure of ϕ. The obtained loss is non-negative: for any assignmentx to the variablesz, we have L(ϕ)(x) ∈ R ≥0 .

Further, the translation has two properties: (P1) anyx for which the loss is zero (L(ϕ)(x) = 0) is a satisfying assignment to ϕ (denoted byx |= ϕ) and (P2) the loss is differentiable almost everywhere.

This construction avoids pitfalls of other approaches (see Appendix B).

We next formally define the translation rules.

Formula ϕ is parametrized by ξ > 0 which denotes tolerance for strict inequality constraints.

Since comparisons are applied element-wise (i.e., on scalars), atomic constraints are transformed into a conjunction of scalar comparisons: DISPLAYFORM1 The comparisons = and ≤ are translated based on a function d : R × R → R which is a continuous, differentiable almost everywhere, distance function with DISPLAYFORM2 Here, 1 t 1 >t 2 is an indicator function: it is 1 if t 1 > t 2 , and 0 otherwise.

The function d is a parameter of the translation and in our implementation we use the absolute distance |t 1 − t 2 |.

For the other comparisons, we define the loss as follows: DISPLAYFORM3 , and L(t 1 > t 2 ) and L(t 1 ≥ t 2 ) are defined analogously.

Conjunctions and disjunctions of formulas ϕ and ψ are translated into loss as follows: Translating negations Negations are handled by first eliminating them from the constraint through rewrite rules, and then computing the loss of their equivalent, negation-free constraint.

Negations of atomic constraints are rewritten to an equivalent atomic constraint that has no negation (note that = is not a negation).

For example, the constraint ¬(t 1 ≤ t 2 ) is rewritten to t 2 < t 1 , while negations of conjunctions and disjunctions are rewritten by repeatedly applying De Morgan's laws: ¬(ϕ ∧ ψ) is equivalent to ¬ϕ ∨ ¬ψ and ¬(ϕ ∨ ψ) is equivalent to ¬ϕ ∧ ¬ψ.

DISPLAYFORM4 With our construction, we get the following theorem: DISPLAYFORM5 Essentially, as we make ξ smaller (and δ approaches 0), we get closer to an if and only if theorem: ifx makes the loss 0, then we have a satisfying assignment; otherwise, ifx makes the loss > δ, then x is not a satisfying assignment.

We provide the proof of the theorem in Appendix A.

In this section, we present our method for training neural networks with constraints.

We first define the problem, then provide our min-max formulation, and finally, discuss how we solve the problem.

We write [ϕ] to denote the indicator function that is 1 if the predicate holds and 0 otherwise.

Training with constraints To train with a single constraint, we consider the following maximization problem over neural network weights θ:arg max DISPLAYFORM0 Here, S 1 , . . .

, S m (abbreviated byS) are independently drawn from an underlying distribution D and ϕ is a constraint over variablesz, constantsS andc, and network weights θ.

This objective is bounded between 0 and 1, and attains 1 if and only if the probability the network satisfies the constraint ϕ is 1.

We extend this definition to multiple constraints, by forming a convex combination of their respective objectives: for w with t i=1 w i = 1 and w i > 0 for all i, we consider DISPLAYFORM1 As standard, we train with the empirical objective where instead of the (unknown) distribution D, we use the training set T to draw samples.

To use our system, the user specifies the constraints ϕ 1 , . . .

, ϕ t along with their weights w 1 , . . .

, w t .

In the following, to simplify the presentation, we assume that there is only one constraint.

Formulation as min-max optimization We can rephrase training networks with constraints as minimizing the expectation of the maximal violation.

The maximal violation is an assignment to the variablesz which violates the constraint (if it exists).

That is, it suffices to solve the problem arg min θ E S 1 ,...

S m ∼T max z 1 ,...,z k ¬ϕ(z,S,c, θ) .

Assume that one can compute, for a givenS and θ, an optimal solutionx * S,θ for the inner maximization problem: DISPLAYFORM2 Then, we can rephrase the optimization problem in terms ofx * S,θ : arg min DISPLAYFORM3 The advantage of this formulation is that it splits the problem into two sub-problems and the overall optimization can be seen as a game between an oracle (solving FORMULA10 ) and an optimizer (solving FORMULA11 ).Solving the optimization problems We solve FORMULA10 and FORMULA11 by translating the logical constraints into differentiable loss (as shown in Sec. 3).

Inspired by Theorem 1, for the oracle (Eq. FORMULA10 ), we approximate the inner maximization by a minimization of the translated loss L(¬ϕ): DISPLAYFORM4 GivenxS ,θ from the oracle, we optimize the following loss using Adam BID12 : DISPLAYFORM5 Constrained optimization In general, the loss in (4) can sometimes be difficult to optimize.

To illustrate, assume that the random samples are input-label pairs (x, y) and consider the constraint: DISPLAYFORM6 Our translation of this constraint to a differentiable loss produces DISPLAYFORM7 This function is difficult to minimize because the magnitude of the two terms is different.

This causes first-order methods to optimize only a single term in an overly greedy manner, as reported by BID2 .

However, some constraints have a closed-form analytical solution, e.g., the minimization of max(0, ||x − z|| ∞ − ) can be solved by projecting into the L ∞ ball.

To leverage this, we identify logical constraints which restrict the variables z to convex sets that have an efficient algorithm for projection, e.g., line segments, L 2 , L ∞ or L 1 balls BID4 .

Note that in general, projection to a convex set is a difficult problem.

We exclude such constraints from ϕ and add them as constraints of the optimization.

We thus rewrite (4) as: DISPLAYFORM8 where the D i denote functions which map random samples to a convex set.

To solve (6), we employ Projected Gradient Descent (PGD) which was shown to have strong performance in the case of adversarial training with L ∞ balls BID14 .Algorithm 1: Training with constraints.input : Training set T , network parameters θ, and a constraint ϕ(z,S,c, θ) for epoch = 1 to n epochs do Sample mini-batch of m-tuples DISPLAYFORM9 Training procedure Algorithm 1 shows our training procedure.

We first form a mini-batch of random samples from the training set T .

Then, the oracle finds a solution for (4) using the formulation in (6).

This solution is given to the optimizer, which solves (5).

Note that if ϕ has no variables (k = 0), the oracle becomes trivial and the loss is computed directly.

We build on DL2 and design a declarative language for querying networks.

Interestingly, the hardcoded questions investigated by prior work can now be phrased as DL2 queries: neurons responsible for a prediction BID18 , inputs that differentiate networks BID21 , and adversarial examples (e.g., BID24 ).

We support the following class of queries: DISPLAYFORM0 Here, find defines the variables and their shape in parentheses, where defines the constraint (over the fragment described in Sec. 3), init defines initial values for (part or all of) the variables, and return defines a target term to compute at the end of search; if missing, z 1 , . . .

, z k are returned.

Networks (loaded so to be used in the queries) and constants are defined outside the queries.

We note that the user can specify tensors in our language (we do not assume these are simplified to vectors).

In queries, we write comma (,) for conjunction (∧); in for box-constraints and class for constraining the target label, which is interpreted asconstraints over the labels' probabilities.

Examples Fig. 1 shows few interesting queries.

The first two are defined over networks trained for CIFAR-10, while the last is for MNIST.

The goal of the first query is to find an adversarial example i of shape (32, 32, 3), classified as a truck (class 9) where the distance of i to a given deer image (deer) is between 6 and 24, with respect to the infinity norm.

Fig. 1b is similar, but the goal is to find i classified as a deer where a specific neuron is deactivated.

The last query's goal is to find i classified differently by two networks where part of i is fixed to pixels of the image nine.

Figure 1: DL2 queries enable to declaratively search for inputs satisfying constraints over networks.

Solving queries As with training, we compile the constraints to a loss, but unlike training, we optimize with L-BFGS-B. While training requires batches of inputs in PGD optimization, querying looks for one assignment, and thus there is more time to employ the more sophisticated, but slower, L-BFGS-B. We discuss further optimizations in Appendix C.

We now present a thorough experimental evaluation on the effectiveness of DL2 for querying and training neural networks with logical constraints.

Our system is implemented in PyTorch BID19 and evaluated on an Nvidia GTX 1080 Ti and Intel Core i7-7700K with 4.20 GHz.

We evaluated DL2 on various tasks (supervised, semi-supervised and unsupervised learning) across four datasets: MNIST, FASHION BID26 , CIFAR-10, and CIFAR-100 BID13 Figure 2 : Supervised learning, P/C is prediction/constraint accuracy.

Supervised learning We consider two types of constraints for supervised learning: global constraints, which have z-s, and training set constraints, where the only variables are from the training set (no z-s).

Note that none of prior work applies to global constraints in general.

Furthermore, because of limitations of their encoding explained in Sec. 2, they are not able to handle complex training set constraints considered in our experiments (e.g., constraints between probability distributions).

To ease notation, we write random samples (the S-s) as x i and y i for inputs from the training set (x i ) and their corresponding label (y i ).For local robustness BID24 , the training set constraint says that if two inputs from the dataset are close (their distance is less than a given 1 , with respect to L 2 norm), then the KL divergence of their output probabilities is smaller than 2 : DISPLAYFORM0 Second, the global constraint requires that for any input x, whose classification is y, inputs in its neighborhood which are valid images (pixels are between 0 and 1), have a high probability for y. For numerical stability, instead of the probability we check that the corresponding logit is larger than a given threshold δ: DISPLAYFORM1 Similarly, we have two definitions for the Lipschitz condition.

The training set constraint requires that for every two inputs from the training set, the distance between their output probabilities is less than the Lipschitz constant (L) times the distance between the inputs: DISPLAYFORM2 The global constraint poses the same constraint for valid images in the neighborhood x 1 and x 2 : DISPLAYFORM3 We also consider a training set constraint called C-similarity, which imposes domain knowledge constraints for CIFAR-10 networks.

The constraint requires that inputs classified as a car have a higher probability for the label truck than the probability for dog: DISPLAYFORM4 The global constraint is similar but applied for valid images in the -neighborhood of x: DISPLAYFORM5 Finally, we consider a Segment constraint which requires that if an input z is on the line between two inputs x 1 and x 2 in position λ, then its output probabilities are on position λ on the line between the output probabilities: DISPLAYFORM6 Fig .

2 shows the prediction accuracy (P) and the constraint accuracy (C) when training with (i) crossed-entropy only (CE) and (ii) CE and the constraint.

Results indicate that DL2 can significantly improve constraint accuracy (0% to 99% for Lipschitz G ), while prediction accuracy slightly decreases.

The decrease is expected in light of a recent work BID25 ), which shows that adversarial robustness comes with decrease of prediction accuracy.

Since adversarial robustness is a type of DL2 constraint, we suspect that we observe a similar phenomenon here.

Accuracy ( Semi-supervised learning For semi-supervised learning, we focus on the CIFAR-100 dataset, and split the training set into labeled, unlabeled and validation set in ratio of 20/60/20.

In the spirit of the experiments of BID27 , we consider the constraint which requires that the probabilities of groups of classes have either very high probability or very low probability.

A group consists of classes of a similar type (e.g., the classes baby, boy, girl, man, and woman are part of the people group), and the group's probability is the sum of its classes' probabilities.

Formally, our constraint consists of 20 groups and its structure is: DISPLAYFORM0 for a small .

We use this constraint to compare the performance of several approaches.

For all approaches, we use the Wide Residual Network BID28 ) as the network architecture.

As a baseline, we train in a purely-supervised fashion, without using the unlabeled data.

We also compare to semantic loss BID27 and rule distillation BID10 .

Note that this constraint is restricting the probability distribution and not samples drawn from it which makes other methods inapplicable (as shown in Sec. 2).

As these methods cannot encode our constraint, we replace them with a closest approximation (e.g., the exactly-one constraint from BID27 for semantic loss).

Details are shown in Appendix D. FIG2 shows the prediction accuracy on the test set for all approaches.

Results indicate that our approach outperforms all existing works.

Unsupervised learning We also consider a regression task in an unsupervised setting, namely training MLP (Multilayer perceptron) to predict the minimum distance from a source to every node in an unweighted graph, G = (V, E).

One can notice that minimum distance is a function with certain properties (e.g., triangle inequality) which form a logical constraint listed below.

Source is denoted as 0.

Table 1 : Results for queries: (#) number of completed instances (out of 10), © is the average running time in seconds, and ©the average running time of successful runs (in seconds).

DISPLAYFORM1 Additionally, we constrain d(0) = 0.

Next, we train the model in an unsupervised fashion with the DL2 loss.

In each experiment, we generate random graphs with 15 vertices and split the graphs into training (300), validation (150) and test set (150).

As an unsupervised baseline, we consider a model which always predicts d(v) = 1.

We also train a supervised model with the mean squared error (MSE) loss.

Remarkably, our approach was able to obtain an error very close to supervised model, without using any labels at all.

This confirms that loss generated by DL2 can be used to guide the network to satisfy even very complex constraints with many nested conjunctions and disjunctions.

We evaluated DL2 on the task of querying with constraints, implemented in TensorFlow.

We considered five image datasets, and for each, we considered at least two classifiers; for some we also considered a generator and a discriminator (trained using GAN BID6 ).

Table 3 (Appendix E) provides statistics on the networks.

Our benchmark consists of 18 template queries (Appendix E), which are instantiated with the different networks, classes, and images.

Table 1 shows the results (-denotes an inapplicable query).

Queries ran with a timeout of 2 minutes.

Results indicate that our system often finds solutions.

It is unknown whether queries for which it did not find a solution even have a solution.

We observe that the success of a query depends on the dataset.

For example, queries 9-11 are successful for all datasets but GTSBR.

This may be attributed to the robustness of GTSBR networks against the adversarial examples that these queries aim to find.

Query 14, which leverages a discriminator to find adversarial examples, is only successful for the CIFAR dataset.

A possible explanation can be that discriminators were trained against real images or images created by a generator, and thus the discriminator performs poorly in classifying arbitrary images.

Query 15, which leverages the generators, succeeds in all tested datasets, but has only few successes in each.

As for overall solving time, our results indicate that, successful executions terminate relatively quickly and that our system scales well to large networks (e.g., for ImageNet).

We presented DL2, a system for training and querying neural networks.

DL2 supports an expressive logical fragment and provides translation rules into a differentiable (almost everywhere) loss, which is zero only for inputs satisfying the constraints.

To make training tractable, we handle input constraints which capture convex sets through PGD.

We also introduce a declarative language for querying networks which uses the logic and the translated loss.

Experimental results indicate that DL2 is effective in both, training and querying neural networks.

DISPLAYFORM0 We start by giving a proof for the if direction of the Theorem 1, i.e. if L(ϕ)(x) = 0, thenx satisfies ϕ. The proof is by induction on the formula structure (we assume ϕ is negation-free as negations can be eliminated as described in the text).As a base case, we consider formulas consisting of a single atomic constraint.• DISPLAYFORM1 , and ϕ is satisfied.• DISPLAYFORM2 and ϕ is satisfied.• DISPLAYFORM3 , and since ξ > 0, we get t 1 (x) < t 2 (x).

Thus, ϕ is satisfied.

As an induction step, we consider combination of formulas using single logical and or logical or operation.• DISPLAYFORM4 By the induction hypothesis, either ϕ is satisfied or ψ is satisfied, implying that ϕ ∨ ψ is satisfied.

DISPLAYFORM5 By the induction hypothesis, ϕ and ψ are satisfied, implying that ϕ ∧ ψ is satisfied.

As all variables come from a bounded set, it is easy to see that for every formula ϕ there exists bound T (ϕ) such that L(ϕ)(x) ≤ T (ϕ).

In other words, loss can not be arbitrarily large for a fixed formula ϕ. Given formula ϕ, we will define N (ϕ) such that the following statement holds: DISPLAYFORM0 We prove Lemma 1 by induction on the formula structure.

Base case of the induction is logical formula consisting of one atomic expression.

In this case it is easy to see that if L(ϕ)(x) > ξ then x does not satisfy the formula.

This means we can set N (ϕ) = 1 for such formulas and statement of the theorem holds.

We distinguish between two cases: DISPLAYFORM1 In this case we define: DISPLAYFORM2 Letx be an assignment which satisfies the formula ϕ.

This implies thatx satisfies both ϕ 1 and ϕ 2 .

From the assumption of induction we know that L(ϕ 1 )(x) < N (ϕ 1 )ξ and L(ϕ 2 )(x) < N (ϕ 2 )ξ.

Adding these inequalities (and using definitions of L and N ) we get: DISPLAYFORM3 In this case we define: DISPLAYFORM4 Letx be an assignment which satisfies the formula ϕ. This implies thatx satisfies one of ϕ 1 and ϕ 2 .

We can assume (without loss of generality) thatx satisfies ϕ 1 .

From the assumption of induction we know that L(ϕ 1 )(x) < N (ϕ 1 )ξ and also L(ϕ 2 )(x) < T (ϕ 2 ).

Multiplying these inequalities (and using definitions of L and N ) we get: DISPLAYFORM5 Thus, one can choose δ(ξ) = N (ϕ)ξ.

Then, lim ξ→0 δ(ξ) = 0 and for every assignmentx, L(ϕ)(x) > δ(ξ) implies thatx does not satisfy ϕ (x |= ϕ), thus proving the Theorem 1.To illustrate this construction we provide an example formula ϕ = x 1 < 1 ∧ x 2 < 2.

The loss encoding for this formula is L(ϕ) = max{x 1 + ξ − 1, 0} + max{x 2 + ξ − 2, 0}, where ξ is the precision used for strong inequalities.

For the given example our inductive proof gives δ(ξ) = 2ξ.

It is not difficult to show that assignments with loss greater than this value do not satisfy the formula.

For example, consider x 1 = 1 + ξ and x 2 = 2 + 3ξ.

In this case L(ϕ)(x) = 6ξ > δ(ξ) = 2ξ and the assignment obviously does not satisfy ϕ. But also consider the assignment x 1 = 1 − 0.5ξ and x 2 = 2 − 0.5ξ.

In this case L(ϕ)(x) > 0 and L(ϕ)(x) = ξ < δ(ξ) = 2ξ and the assignment is indeed satisfying.

XSAT BID5 ) also translates logical constraints into numerical loss, but its atomic constraints are translated into non-differentiable loss, making the whole loss non-differentiable.

Probabilistic soft logic (e.g., BID3 ; BID10 ) translates logical constraints into differentiable loss, which ranges between [0, 1].

However, using their loss to find satisfying assignments with gradient methods can be futile, as the gradient may be zero.

To illustrate, consider the toy example of ϕ(z) := (z = ( 1 1 )).

PSL translates this formula into the loss L PSL (ϕ) = max{z 0 + z 1 − 1, 0} (it assumes z 0 , z 1 ∈ [0, 1]).

Assuming optimization starts from x = ( 0.2 0.2 ) (or any pair of numbers such that z 0 + z 1 − 1 ≤ 0), the gradient is ∇ z L PSL (ϕ)(x) = ( 0 0 ), which means that the optimization cannot continue from this point, even though x is not a satisfying assignment to ϕ. In contrast, with our translation, we obtain L(ϕ)(z) = |z 0 − 1| + |z 1 − 1|, for which the gradient for the same x is ∇ z L(ϕ)(x) = −1 −1 .

Here we discuss how the loss compilation can be optimized for L-BFGS-B. While our translation is defined for arbitrary large constraints, in general, it is hard to optimize for a loss with many terms.

Thus, we mitigate the size of the loss by extracting box constraints out of the expression.

The loss is then compiled from remaining constraints.

Extracted box constraints are passed to the L-BFGS-B solver which is then used to find the minimum of the loss.

This "shifting" enables us to exclude a dominant part of ϕ from the loss, thereby making our loss amenable to optimization.

To illustrate the benefit, consider the query in Fig. 1a .

Its box constraint, i in [0, 255] , is a syntactic sugar to a conjunction with 2 · 32 · 32 · 3 = 6, 144 atomic constraints (two for each variables, i.e., for every index j, we have i j ≥ 0 and i j ≤ 255).

In contrast, the second constraint consists of 9 atomic constraints (one for each possible class different from 9, as we shortly explain), and the third and fourth constraints are already atomic.

If we consider 6, 155 atomic constraints in the loss, finding a solution (with gradient descent) would be slow.

For larger inputs (e.g., inputs for ImageNet, whose size is 224 · 224 · 3 > 150, 000), it may not terminate in a reasonable time.

By excluding the box constraints from the loss, the obtained loss consists of only 11 terms, making it amenable for gradient optimization.

We note that while a solution is not found (and given enough timeout), we restart L-BFGS-B and initialize the variables using MCMC sampling.

Table 2 : Hyperparameters used for supervised learning experiment

Here we describe implementation details (including hyperaparameters) used during our experiments.

Supervised learning For our experiments with supervised learning we used batch size 128, Adam optimizer with learning rate 0.0001.

All other parameters are listed in 2.

Additionally, for CIFAR-10 experiments we use data augmentation with random cropping and random horizontal flipping.

Experiments with Segment constraints are done by first embedding images in 40-dimensional space using PCA.

In lower dimensional space it is sensible to consider linear interpolation between images which is not the case otherwise.

Note that this experiment is not performed for CIFAR-10 because we do not observe good prediction accuracy with baseline model using lower dimensional embeddings.

This is likely because dimensionality of CIFAR-10 images is much higher than MNIST or FASHION.We used ResNet-18 BID9 for experiments on CIFAR-10 and convolutional neural network (CNN) with 6 convolutional and 2 linear layers for MNIST and FASHION (trained with batchnorm after each convolutional layer).

The layer dimensions of CNN are (1, 32, 5x5) -(32, 32, 5x5) -(32, 64, 3x3) -(64, 64, 3x3) -(64, 128, 3x3) -(128, 128, 1x1) -100 -10 where (in, out, kernel-size) denotes a convolutional layer and a number denotes a linear layer with corresponding number of neurons.

Semi-supervised learning All methods use the same Wide Residual Network model.

We use depth 28 and widening factor 10.

Neural network is optimized using Adam with learning rate 0.001.

We use λ = 0.6 as weighting factor for DL2 loss.

For semantic loss experiment we follow the encoding from BID27 .

Please consult the original work to see how exactly-one constraint is encoded into semantic loss.

Since rule distillation does not support our constraint, we use the following approximation (following notation from BID10 ): DISPLAYFORM0 We denote G(Y ) as set of labels sharing the same group as Y .

Note that rule is meant to encourage putting more probability mass into the groups which already have high probability mass.

This should result in the entire probability mass collapsed in one group in the end, as we want.

We use π t = max(0, 1.0 − 0.97 t ) as mixing factor.

Other constants used are C = 1 and λ = 1.In this experiment, we used Wide Residual Networks BID28 ) with n=28 and k=10 (i.e. 28 layers).

Our model is the multilayer perceptron with N *N input neurons, three hidden layers with 1000 neurons each and an output layer of N neurons.

N is the number of vertices in the graph, in our case 15.

The input takes all vertices in the graph and the output is the distance for each node.

The network uses ReLU activations and dropout of 0.3 after at each hidden layer.

Network is optimized using Adam with learning rate 0.0001.

ResNet-50 from Keras 0.759 * Table 3 : The datasets and networks used to evaluate DL2.

The reported accuracy is top-1 accuracy and it was either computed by the authors ( * ), users that implemented the work ( # ), or by us ( † ).

Note that for GTSRB the images have dimensions 32×32×3, but the Cs take inputs of 32×32(×1), which are pre-processed grayscale versions.

DISPLAYFORM0

Here we provide further experiments to investigate scalability and run-time behavior of DL.

For all experiments we use the same hyperparameters as in Section 6.2, but ran experiments I and II on a laptop CPU and experiment III on the same GPU setup as in Section 6.2 and increased the timeout to 300 s. (a) Run-time for experiment 1.

Runs up to 2 13 variables are between 0.1 − 0.2 s and don't show increase with number of variables.

Afterwards growth is linear.

Experiment I: Number of variables To study the run-time behavior in the number of variables we consider a simple toy query DISPLAYFORM0 w h e r e 1000 < sum(i), sum(i) < 1001 r e t u r n i for different integers c.

We execute this query for a wide range of c values, 10 times each and report the average run-time in FIG10 .

All runs succeeded and found a correct solution.

We observe constant run-time behavior for up to 2 13 variables and linear run-time in the number of variables afterwards.

Experiment II: Opposing constrains To study the impact of (almost) opposing constraints we again consider a simple toy query for an integer c. This query requires optimizing two opposing terms until one of them is fulfilled.

The larger c the more opposed the two objectives are and indeed for c → ∞ we would obtain an unsatisfiable objective.

Again all runs succeeded and found a correct solution.

In FIG10 we present the average run-time over 10 runs for different c. Up to c = 5000 the run-time is constant with roughly 0.15 s.

Experiment III: Scaling in the number of constraints To study the scaling of DL2 in the number of constraints consider the following query for an adversarial example:For this experiment we consider the query: The query looks for an adversarial perturbation p to a given image of a nine (M_nine) such that the resulting image gets classifies as class c. The query returns the found perturbation and the resulting image.

The clamp(I, a, b) operation takes an input I and cuts off all it's values such that they are between a and b.

Additionally we impose constraints the rows and columns of the image.

For a row i we want to enforce that the values of the perturbation vector are increasing from left to right:

<|TLDR|>

@highlight

A differentiable loss for logic constraints for training and querying neural networks.

@highlight

A framework for turning queries over parameters and input, ouput pairs to neural networks into differentiable loss functions and an associated declarative language for specifying these queries

@highlight

This paper tackles the problem of combining logical approaches with neural networks by translating a logical formula into a non-negative loss function for a neural network.