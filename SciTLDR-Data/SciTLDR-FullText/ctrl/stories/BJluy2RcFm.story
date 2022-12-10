We consider a simple and overarching representation for permutation-invariant functions of sequences (or set functions).

Our approach, which we call Janossy pooling, expresses a permutation-invariant function as the average of a permutation-sensitive function applied to all reorderings of the input sequence.

This allows us to leverage the rich and mature literature on permutation-sensitive functions to construct novel and flexible permutation-invariant functions.

If carried out naively, Janossy pooling can be computationally prohibitive.

To allow computational tractability, we consider three kinds of approximations: canonical orderings of sequences, functions with k-order interactions, and stochastic optimization algorithms with random permutations.

Our framework unifies a variety of existing work in the literature, and suggests possible modeling and algorithmic extensions.

We explore a few in our experiments, which demonstrate improved performance over current state-of-the-art methods.

Pooling is a fundamental operation in deep learning architectures BID23 .

The role of pooling is to merge a collection of related features into a single, possibly vector-valued, summary feature.

A prototypical example is in convolutional neural networks (CNNs) BID22 , where linear activations of features in neighborhoods of image locations are pooled together to construct more abstract features.

A more modern example is in neural networks for graphs, where each layer pools together embeddings of neighbors of a vertex to form a new embedding for that vertex, see for instance, BID20 BID0 BID15 Velickovic et al., 2017; BID28 Xu et al., 2018; BID26 BID25 van den Berg et al., 2017; BID12 BID13 Ying et al., 2018; Xu et al., 2019)

.A common requirement of a pooling operator is invariance to the ordering of the input features.

In CNNs for images, pooling allows invariance to translations and rotations, while for graphs, it allows invariance to graph isomorphisms.

Existing pooling operators are mostly limited to predefined heuristics such as max-pool, min-pool, sum, or average.

Another desirable characteristic of pooling layers is the ability to take variable-size inputs.

This is less important in images, where neighborhoods are usually fixed a priori.

However in applications involving graphs, the number of neighbors of different vertices can vary widely.

Our goal is to design flexible and learnable pooling operators satisfying these two desiderata.

Abstractly, we will view pooling as a permutation-invariant (or symmetric) function acting on finite but arbitrary length sequences h. All elements h i of the sequences are features lying in some space H (which itself could be a high-dimensional Euclidean space R d or some subset thereof).

The sequences h are themselves elements of the union of products of the H-space: h ∈ ∞ j=0 H j ≡ H ∪ .

Throughout the paper, we will use Π n to represent the set of all permutations of the integers 1 to n, where n will often be clear from the context.

In addition, h π , π ∈ Π |h| , will represent a reordering of the elements of a sequence h according to π, where |h| is the length of the sequence h. We will use the double bar superscript f to indicate that a function is permutation-invariant, returning the same value no matter the order of its arguments: f (h) = f (h π ), ∀π ∈ Π |h| .

We will use the arrow superscript f to indicate general functions on sequences h which may or may not be permutationinvariant 1 .

Functions f without any markers are 'simple' functions, acting on elements in H, scalars or any other argument that is not a sequence of elements in H.Our goal in this paper is to model and learn permutation-sensitive functions f that can be used to construct flexible and learnable permutation-invariant neural networks.

A recent step in this direction is work on DeepSets by Zaheer et al. (2017) , who argued for learning permutation-invariant functions through the following composition: DISPLAYFORM0 f (|h|, h; θ (f ) ) = |h| j=1 f (h j ; θ (f ) ) and h ≡ h(x; θ (h) ).Here, (a) x ∈ X is one observation in the training data (X itself may contain variable-length sequences), h ∈ H is the embedding (output) of the data given by the lower layers h : X × R a → H ∪ , a > 0 with parameters θ (h) ∈ R a ; (b) f : H × R b → F is a middle-layer embedding function with parameters θ (f ) ∈ R b , b > 0, and F is the embedding space of f ; and (c) ρ : F × R c → Y is a neural network with parameters θ (ρ) ∈ R c , c > 0, that maps to the final output space Y. Typically H and F are high-dimensional real-valued spaces; Y is often R d in d-dimensional regression problems or the simplex in classification problems.

Effectively, the neural network f learns an embedding for each element in H, and given a sequence h, its component embeddings are added together before a second neural network transformation ρ is applied.

Note that the function h may be the identity mapping h(x; ·) = x that makes f act directly on the input data.

Zaheer et al. (2017) argue that if ρ is a universal function approximator, the above architecture is capable of approximating any symmetric function on h-sequences, which justifies the widespread use of average (sum) pooling to make neural networks permutation-invariant in BID12 , BID15 , BID20 , BID0 , among other works.

We note that Zaheer et al. (2017) focus on functions of sets but the work was extended to functions of multisets by Xu et al. (2019) and that Janossy pooling can be used to represent multiset functions.

The embedding h is permuted in all |h|!

possible ways, and for each permutation h π , f (|h|, h π ; θ (f ) ) is computed.

These are summed and passed to a second function ρ(·; θ (ρ) ) which gives the final permutation-invariant output y(x; θ (ρ) , θ (f ) , θ (h) ); the gray rectangle represents Janossy pooling.

We discuss how this can be made computationally tractable.

In practice, there is a gap between flexibility and learnability.

While the architecture of equations 1 and 2 is a universal approximator to permutationinvariant functions, it does not easily encode structural knowledge about y.

Consider trying to learn the permutation-invariant function y(x) = max i,j≤|x| |x i − x j |.

With higherorder interactions between the elements of h, the functions f of equation 2 cannot capture any useful intermediate representations towards the final output, with the burden shifted entirely to the function ρ.

Learning ρ means learning to undo mixing performed by the summation layer f (|h|, h; θ (f ) ) = |h| j=1 f (h j ; θ (f ) ).

As we show in our experiments, in many applications this is too much to ask of ρ.

Contributions.

We investigate a learnable permutation-invariant pooling layer for variable-size inputs inspired by the Janossy density framework, widely used in the theory of point processes (Daley & Vere-Jones, 2003, Chapter 7) .

This approach, which we call Janossy pooling, directly allows the user to model what higher-order dependencies in h are relevant in the pooling.

FIG0 summarizes a neural network with a single Janossy pooling layer f (detailed in Definition 2.1 below): given an input embedding h, we apply a learnable (permutation-sensitive) function f to every permutation h π of the input sequence h. These outputs are added together, and fed to the second function ρ.

Examples of function f include feedforward and recurrent neural networks (RNNs).

We call the operation used to construct f from f the Janossy pooling.

Definition 2.1 gives a more detailed description.

We will detail three broad strategies for making this computation tractable and discuss how existing methods can be seen as tractability strategies under the Janossy pooling framework.

Thus, we propose a framework and tractability strategies that unify and extend existing methods in the literature.

We contribute the following analysis: (a) We show DeepSets (Zaheer et al., 2017) is a special case of Janossy pooling where the function f depends only on the first element of the sequence h π .

In the most general form of Janossy pooling (as described above), f depends on its entire input sequence h π .

This naturally raises the possibility of intermediate choices of f that allow practitioners to trade between flexibility and tractability.

We will show that functions f that depend on their first k arguments of h π allow the Janossy pooling layer to capture up to k-ary dependencies in h. (b) We show Janossy pooling can be used to learn permutation-invariant neural networks y(x) by sampling a random permutation of h during training, and then modeling this permuted sequence using a sequence model such as a recurrent neural network (LSTMs BID17 , GRUs BID6 ) or a vector model such as a feedforward network.

We call this permutation-sampling learning algorithm π-SGD (π-Stochastic Gradient Descent).

Our analysis explains why this seemingly unsound procedure is theoretically justified, which sheds light on the recent puzzling success of permutation sampling and LSTMs in relational models BID29 BID15 .

We show that this property relates to randomized model ensemble techniques.

(c) In Zaheer et al. (2017) , the authors describe a connection between DeepSets and infinite de Finetti exchangeabilty.

We provide a probabilistic connection between Janossy pooling and finite de Finetti exchangeabilty BID11 .

We first formalize the Janossy pooling function f .

Start with a function f , parameterized by θ (f ) , which can take any variable-size sequence as input: a sequence of matrices (such as images), a sequence of vectors (such as a sequence of vector embeddings), or a variable-size sequence of features or embeddings representing the neighbors of a node in an attributed graph.

In practice, we implement f with a neural network.

Formalizing FIG0 from Section 1, we use f to define f : DISPLAYFORM0 where Π |h| is the set of all permutations of the integers 1 to |h|, and h π represents a particular reordering of the elements of sequence h according to π ∈ Π |h| .

We refer the operation used to construct f from f as Janossy pooling.

♦ Definition 2.1 provides a conceptually simple approach for constructing permutation-invariant functions from arbitrary and powerful permutation-sensitive functions such as feedforward networks, recurrent neural networks, or convolutional neural networks.

If f is a vector-valued function, then so is f , and in practice, one might pass this vector output of f through a second function ρ (e.g. a neural network parameterized by θ (ρ) ): DISPLAYFORM1 Equation FORMULA2 can capture any permutation-invariant function g for a flexible enough family of permutation-sensitive functions f (for instance, one could always set f = g).

Thus, at least theoretically, ρ in equation 4 provides no additional representational power.

In practice, however, ρ can improve learnability by capturing common aspects across all terms in the summation.

Furthermore, when we look at approximations to equation 3 or restrictions of f to more tractable families, adding ρ can help recover some of the lost model capacity.

Overall then, equation 4 represents one layer of Janossy pooling, forming a constituent part of a bigger neural network.

FIG0 summarizes this.

Janossy pooling, as defined in equation 3 and 4 is intractable; the computational cost of summing over all permutations (for prediction), and backpropagating gradients (for learning) is likely prohibitive for most problems of interest.

Nevertheless, it provides an overarching framework to unify existing methods, and to extend them.

In what follows we present strategies for mitigating this, allowing novel and effective trade-offs between learnability and computational cost.

A simple way to achieve permutation-invariance without the summation in equation 3 is to order the elements of h according to some canonical ordering based on its values, and then feed the reordered sequence to f .

More precisely, one defines a function CANONICAL : H ∪ → H ∪ such that CANONICAL(h) = CANONICAL(h π )∀π ∈ Π |h| and only considers functions f based on the composition f = CANONICAL• f .

Note that specifying a permutation-invariant CANONICAL is not equivalent to the original problem since one may define a function of only the data and not of learnable parameters (e.g. sort).

This input constraint then allows the use of complex f models, such as RNNs, that can capture arbitrary relationships in the canonical ordering of h without the need to sum over all permutations of the input.

Examples of the canonical ordering approach already exist in the literature, for example, BID30 order nodes in a graph according to a user-specified ranking such as betweenness centrality (say from high to low).

This approach is useful only if the canonical ordering is relevant to the task at hand.

BID30 acknowledges this shortcoming and BID29 demonstrates that an ordering by Personalized PageRank BID32 BID18 achieves a lower classification accuracy than a random ordering.

As an idealized example, consider DISPLAYFORM0 , with (h i,1 , h i,2 ) ∈ H = R 2 , and components h i,1 and h i,2 sampled independently of each other.

Choosing to sort h according to h ·,1 when the task at hand depends on sorting according to h ·,2 can lead to poor prediction accuracy.

Rather than pre-defining a good canonical order, one can try to learn it from the data.

This requires searching over the discrete space of all |h|!

permutations of the input vector h. In practice, this discrete optimization relies on heuristics (Vinyals et al., 2016; BID36 .

Alternatively, instead of choosing a single canonical ordering, one can choose multiple orderings, resulting in ensemble methods that average across multiple permutations.

These can be viewed as more refined (possibly data-driven) approximations to equation 3.

Here, we provide a different spectrum of options to trade-off flexibility, complexity, and generalizability in Janossy pooling.

Now, to simplify the sum over permutations in equation 3, we impose structural constraints where f (h) depends only on the first k elements of its input sequence.

This amounts to the assumption that only k-ary dependencies in h are relevant to the task at hand.

Definition 2.2: [k-ary Janossy pooling]

Fix k ∈ N. For any sequence h, define ↓ k (h) as its projection to a length k sequence; in particular, if |h| ≥ k, we keep the first k elements.

Then, a k-ary permutation-invariant Janossy function f is given by DISPLAYFORM0 Note that if some of the embeddings have length |h| < k, then we can zero pad to form the length-k sequence (↓ k (h π ), 0, . . . , 0).

Proposition 2.1 shows that if |h| > k, equation 5 only needs to sum over |h|!/(|h| − k)!

terms, which can be tractable for small k. Proposition 2.1.

The Janossy pooling in equation 5 requires summing over only |h|! (|h|−k)!

terms, thus saving computation when k < |h|.

In particular, equation 5 can be written as DISPLAYFORM1 , where I |h| is the set of all permutations of {1, 2, . . .

, |h|} taken k at a time, and h j is the j-th element of h.

Note that the value of k balances computational savings and the capacity to model higher-order interactions; it can be selected as a hyperparameter based on a-priori beliefs or through typical hyperparameter tuning strategies.

Remark 2.1 (DeepSets (Zaheer et al., 2017 ) is a 1-ary (unary) Janossy pooling).

Equation 5 represented with k = 1 and composing with ρ as in equation 4 yields the model ρ DISPLAYFORM2 and thus equations 1 and 2 for an appropriate choice of f .Not surprisingly, the computational savings obtained from k-ary Janossy pooling come at the cost of reduced model flexibility.

The next result formalizes this.

Theorem 2.1.

For any k ∈ N, define F k as the set of all permutation-invariant functions that can be represented by Janossy pooling with k-ary dependencies.

Then, F k−1 is a proper subset of F k if the space H is not trivial (i.e. if the cardinality of H is greater than 1).

Thus, Janossy pooling with k-ary dependencies can express any Janossy pooling function with (k − 1)-ary dependencies, but the converse does not hold.

The proof is given in the Supplementary Material.

Theorem 2.1 has the following implication: Corollary 2.1.

For k > 1, the DeepSets function in equation 1 (Zaheer et al., 2017) pushes the modeling of k-ary relationships to ρ.

Proof.

DeepSets functions can be expressed via Janossy pooling with k = 1.

Thus, by Theorem 2.1, f in equation 2 cannot express all functions that can be expressed by higher-order (i.e. k > 1) Janossy pooling operations.

Consequently, if the DeepSets function can express any permutationinvariant function, the expressive power must have been pushed to ρ.

Another approach to tractable Janossy pooling samples random permutations of the input h during training.

Like the canonical ordering approach of Section 2.1, this offers significant computational savings, allowing more complex models for f such as LSTMs and GRUs.

However, in contrast with that approach, this is considerably more flexible, avoiding the need to learn a canonical ordering or to make assumptions about the dependencies between the elements of h and the objective function.

Rather, it can be viewed as implicitly assuming simpler structure in these functions.

The approach of sampling random permutations has been previously used in relational learning tasks BID29 BID15 as a heuristic with an LSTM as f .

Both these papers report that permutation sampling outperforms or closely matches other tested neural network models they tried.

Therefore, this section not only proposes a tractable approximation for equation 3 but also provides a theoretical framework to understand and extend such approaches.

For the sake of simplicity, we analyze the optimization with a single sampled permutation.

However, note that increasing the number of sampled permutations in the estimate of f decreases variance, and we recover the exact algorithm when all |h|! permutations are sampled.

We assume a supervised learning setting, though our analysis easily extends to unsupervised learning.

We are given training data D ≡ {(x(1), y(1)), . . . , (x(N ), y(N ))}, where y(i) ∈ Y is the target output and x(i) its corresponding input.

Our original goal was to minimize the empirical loss DISPLAYFORM0 , where DISPLAYFORM1 and DISPLAYFORM2 Computing the gradient of equation 6 is intractable for large inputs h (i) , as the backpropagation computation graph branches out for every permutation in the sum.

To address this computational challenge, we will turn our attention to stochastic optimization.

Permutation sampling.

Consider replacing the Janossy sum in equation 7 with the estimatê DISPLAYFORM3 where s is a random permutation sampled uniformly, s ∼ Unif(Π |h| ).

The estimator in equation 8 is unbiased: DISPLAYFORM4 .

Note however that when f is chained with another nonlinear function ρ and/or nonlinear loss L, the composition is no longer unbiased: DISPLAYFORM5 .

Nevertheless, we use this estimate to propose the following stochastic approximation algorithm for gradient descent: DISPLAYFORM6 uniformly from the training data D. At step t, consider the stochastic gradient descent update DISPLAYFORM7 where DISPLAYFORM8 is the random gradient, where h DISPLAYFORM9 Effectively, this is a Robbins-Monro stochastic approximation algorithm of gradient descent BID37 BID3 and optimizes the following modified objective: DISPLAYFORM10 Observe that the expectation over permutations is now outside the L and ρ functions.

Like equation 6, the loss in equation 10 is also permutation-invariant, though we note that π-SGD, after a finite number of iterations, returns a ρ( f (· · · , h (i) , · · · )) sensitive to the random input permutations of h (i) presented to the algorithm.

Further, unless the function f itself is permutation-invariant (f = f ), the optima of J are different from those of the original objective function L. Instead, J is an upper bound to L via Jensen's inequality if L is convex and ρ is the identity function (equation 3); minimizing this upper bound forms a tractable surrogate to the original Janossy objective.

If the function class used to model f is rich enough to include permutation-invariant functions, then the global minima of J will include those of L. In general, minimizing the upper bound implicitly regularizes f to return functions that are insensitive to permutations of the training data.

While a general ρ no longer upper bounds the original objective, the implicit regularization of permutationsensitive functions still applies to the composition f ≡ ρ • f and we show competitive results.

It is important to observe that the function ρ plays a very different role in our π-SGD formulation compared to k-ary Janossy pooling.

Previously ρ was composed with an average over f to model dependencies not captured in the average-and was in some sense separate from f -whereas here it becomes absorbed directly into f = ρ • f .The next result, which we state and prove more formally in the Supplementary Material, provides some insight into the convergence properties of our algorithm.

Although the conditions are difficult to check, they are similar to those used to demonstrate the convergence of SGD, which has been empirically demonstrated to yield strong performance in practice.

Proposition 2.2. [π-SGD Convergence]

The optimization of π-SGD enjoys properties of almost sure convergence to the optimal θ under similar conditions as SGD.Variance reduction.

Variance reduction of the output of a sampled permutation f (|h|, DISPLAYFORM11 , inducing a nearequivalence between optimizing equation 6 and equation 10.

Possible approaches include importance sampling (used by BID5 for 1-ary Janossy), control variates (also used by BID4 also used for 1-ary Janossy), Rao-Blackwellization (Lehmann & Casella, 2006, Section 1.7), and an output regularization, which includes a penalty for two distinct sampled permutations s and s , f (|h|, h s ; DISPLAYFORM12 , so as to reduce the variance of the sampled Janossy pooling output (used before to improve Dropout masks by Zolna et al. FORMULA0 ).Inference.

The use of π-SGD to optimize the Janossy pooling layer optimizes the objective J, and thus has the following implication on how outputs should be calculated at inference time: Remark 2.2 (Inference).

Assume L(y,ŷ) is convex as a function ofŷ (e.g., L is the L 2 norm, crossentropy, or negative log-likelihood losses).

At test time we estimate the output y(i) of input x(i) by computing (or estimating) DISPLAYFORM13 Combining π-SGD and Janossy with k-ary Dependencies.

In some cases one may consider k-ary Janossy pooling with a moderately large value of k in which case even the summation over |h|! (|h|−k)!

terms (see proposition 2.1) becomes expensive.

In these cases, one may sample s ∼ Unif Π |h| and computef k = f (|h|, ↓ k (h s ); θ (f ) ) in lieu of the sum in equation 5.

Note that equation 5 defining k-ary Janossy pooling constitutes exact inference of a simplified model whereas π-SGD with k-ary dependencies constitutes approximate inference.

We will return to this idea in our results section where we note that the GraphSAGE model of BID15 can be cast as a π-SGD approximation of k-ary Janossy pooling.

In what follows we empirically evaluate two tractable Janossy pooling approaches, k-ary dependencies (section 2.2) and sampling permutations for stochastic optimization (section 2.3), to learn permutation-invariant functions for tasks of different complexities.

One baseline we compare against is DeepSets (Zaheer et al., 2017) ; recall that this corresponds to unary (k = 1) Janossy pooling (Remark 2.1).

Corollary 2.1 shows that explicitly modeling higher-order dependencies during pooling simplifies the task of the upper layers (ρ) of the neural network, and we evaluate this experimentally by letting k = 1, 2, 3, |h| over different arithmetic tasks.

We also evaluate Janossy pooling in graph tasks, where it can be used as a permutation-invariant function to aggregate the features and embeddings of the neighbors of a vertex in the graph.

Note that in graph tasks, permutation-invariance is required to ensure that the neural network is invariant to permutations in the adjacency matrix (graph isomorphism).

The code used to generate the results in this section are available on GitHub 2 .

We first consider the task of predicting the sum of a sequence of integers (Zaheer et al., 2017) and extend it to predicting other permutation-invariant functions: range, unique sum, unique count, and variance.

In the sum task we predict the sum of a sequence of 5 integers drawn uniformly at random with replacement from {0, 1, . . .

, 99}; the range task also receives a sequence 5 integers distributed the same way and tries to predict the range (the difference between the maximum and minimum values); the unique sum task receives a sequence of 10 integers, sampled uniformly with replacement from {0, 1, . . .

, 9}, and predicts the sum of all unique elements; the unique count task also receives a sequence of repeating elements from {0, 1, . . .

, 9}, distributed in the same was as with the unique sum task, and predicts the number of unique elements; the variance task receives a sequence of 10 integers drawn uniformly with replacement from {0, 1, . . .

, 99} and tries to predict the variance DISPLAYFORM0 2 , wherex denotes the mean of x. Unlike Zaheer et al. FORMULA0 , we choose to work with the digits themselves, to allow a more direct assessment of the different Janossy pooling approximations.

Note that the summation task of Zaheer et al. FORMULA0 is naturally a unary task that lends itself to the approach of embedding individual digits then adding them together while the other tasks require exploiting high-order relationships within the sequence.

Following Zaheer et al. FORMULA0 , we report accuracy (0-1 loss) for all tasks with an integer target; we report root mean squared error (RMSE) for the variance task.

Here we explore two Janossy pooling tractable approximations: (a) (k-ary dependencies) Janossy (k = 1) (DeepSets), and Janossy k = 2, 3 where f is a feedforward network with a single hidden layer comprised of 30 neurons.

As detailed in the Supplementary Material, the models are constructed to have the same number of parameters regardless of k by modifying the embedding (output) dimension of h. In the Supplementary Material, we also show results for experiments that relax this constraint.

(b) (π-SGD) Full k = |h| Janossy pooling where f is an LSTM or a GRU that returns the shortterm hidden state of the last temporal unit (the h t of Cho et al. with t = |h|).

The LSTM has 50 hidden units and the GRU 80, trained with the π-SGD stochastic optimization.

The number of hidden units was chosen to be consistent with Zaheer et al. (2017) .

At test time, we experiment with approximating (estimating) equation 11 using 1 and 20 sampled permutations.

FORMULA0 ] a feedforward network with one hidden layer using tanh activations and 100 units.

Choosing a simple and complex form for ρ allows insight into the extent to which ρ supplements the capacity of the model by capturing relationships not exploited during pooling, and serves as an evaluation of the strategy of optimizing J as a tractable approximation of L.Much of our implementation, architectural, and experimental design are based on the DeepSets code 3 of Zaheer et al. (2017) , see the Supplementary Material for details.

We tuned the Adam learning rate for each model and report the results using the rate yielding top performance on the validation set.

TAB0 shows the accuracy (average 0-1 loss) of all tasks except variance, for which we report RMSE in the last column.

Performance was similar between the LSTM and GRU models, with the GRU performing slightly better, thus we moved the LSTM results to Table 3 in the Supplementary Material for the sake of clarity.

We trained each model with 15 random initializations of the weights to quantify variability.

TAB2 in the Supplementary Material shows the same results measured by mean absolute error.

The data consists of 100,000 training examples and 10,000 test examples.

The results in TAB0 show that: (1) models trained with π-SGD using LSTMs and GRUs as f typically achieve top performance or are comparable to the top performer (within confidence intervals) on all tasks, for any choice of ρ.

We also observe for LSTMs and GRUs that adding complexity to ρ can yield small but meaningful performance gains or maintain similar performance, lending credence to the approach of optimizing J as a tractable approximation to L. (2) Specifically, in the variance task, GRUs and LSTMs with π-SGD provide significant accuracy gains over k ∈ {1, 2, 3}, showing that modeling full-dependencies can be advantageous even if model training with π-SGD is approximate.

(3) For a more complex ρ (MLP as opposed to Linear), lowercomplexity Janossy pooling achieves consistently better results: k ∈ {2, 3} gives good results when ρ is linear but poorer results when ρ is an MLP (

as these models are more expressive, the only feasible explanation is an optimization issue since we also observed poorer performance on the training data).

We also note that when ρ is an MLP, it takes significantly more epochs for k ∈ {2, 3} to find the best model (2000 epochs) while k = 1 finds good models much quicker (1000 epochs).

The results we report come from training with 1000 epochs on all models with a linear ρ and 2000 epochs for all models where ρ is an MLP.

(4) We observe that for k = 1 (DeepSets), a more complex ρ (MLP) is required as the pooling pushes the complexity of modeling high-order interactions over the input to ρ.

The converse is also true, if ρ is simple (Linear) then a Janossy pooling that models high-order interactions k ∈ {2, 3, |h|} gives higher accuracy, as shown in the range, unique sum, unique count, and variance tasks.

Here we consider Janossy pooling in the context of graph neural networks to learn vertex representations enabling vertex classification.

The GraphSAGE algorithm BID15 consists of sampling vertex attributes from the neighbor multiset of each vertex v before performing an aggregation operation which generates an embedding of v; the authors consider permutation-invariant operations such as mean and max as well as the permutation-sensitive operation of feeding a randomly permuted neighborhood sequence to an LSTM.

The sample and aggregate procedure is repeated twice to generate an embedding.

Each step can be considered as Janossy pooling with π-SGD and k-ary subsequences, where k l , l ∈ {1, 2} is the number of vertices sampled from each neighborhood and f is for instance a mean, max, or LSTM.

However, at test time, GraphSAGE only samples one permutation s of each neighborhood to estimate equation 11.In our experiments, we also consider computing the mean of the entire neighborhood.

Here we say k = 1 to reinforce the connection to unary Janossy pooling whereas with the LSTM model, k refers to the number of samples of the neighborhood.

In this section we investigate two conditions: (a) the impact of increasing k in the k-ary dependencies; and (b) the benefits of increasing the number of sampled permutations at inference time.

To implement the model and design our experiments, we modified the reference PyTorch code provided by the authors 4 .

We consider the three graph datasets considered in BID15 : Cora and Pubmed (Sen et al., 2008 ) and the larger Protein-Protein Interaction (PPI) (Zitnik & Leskovec, 2017) .

The first two are citation networks where vertices represent papers, edges represent citations, and vertex features are bag-of-words representations of the document text.

The task is to classify the paper topic.

The PPI dataset is a collection of several graphs each representing human tissue; vertices represent proteins, edges represent protein interaction, features include genetic and immunological features, and we try to classify protein roles (there are 121 targets).

More details of these experiments are shown in Table 9 in the Supplementary Material.(a) TAB1 shows the impact (on accuracy) of increasing the number of k-ary dependencies.

We use k 1 , k 2 ∈ {3, 5, 10, 25} for the two pooling layers of our graph neural network (GNN).

The function f is an LSTM (except for when we try mean-pooling).

Note that for the LSTM, the number of parameters of the model is independent of k. At inference time, we sample 20 random permutations of each sequence and average the predicted probabilities before making a final prediction of the class label.

The results in TAB1 show that the choice of k 1 , k 2 ∈ {3, 5, 10, 25} makes little difference on Cora and Pubmed due to the small neighborhood sizes: k 1 , k 2 ≥ 5 often amounts to a Entries denoted by -all differ by less than 0.01.

Typical neighborhoods in Cora and Pubmed are small, so that sampling ≥ 5 neighbors is often equivalent to using the entire neighborhood.

b Some neighbor sequences in PPI are prohibitively large, so we take k1 = k2 = 100.sampling the entire neighborhood.

In PPI, whose average degree is 28.8, increasing k yields consistent improvement.

The strong performance of mean-pooling points to both a relatively easy task 5 and the benefits of utilizing the entire neighborhood of each vertex.

(b) We now investigate whether increasing the number of sampled permutations used to estimate equation 11 at test (inference) time impacts accuracy.

FIG8 in the Supplementary Material shows that increasing the number of sampled permutations from one to three leads to an increase in accuracy in the PPI task (Cora and Pubmed degrees are too small for this test) but diminishing returns set in by the seventh sample.

Using paired tests -t and Wilcoxon signed rank -we see that test inference with seven sampled permutations versus one permutation is significant with p < 10 −3 over 12 replicates.

Sampling permutations at inference time is thus a cheap method for achieving modest but potentially important gains at inference time.

Under the Janossy pooling framework presented in this work, existing literature falls under one of three approaches to approximating to the intractable Janossy-pooling layer: Canonical orderings, k-ary dependencies, and permutation sampling.

We also discuss the broader context of invariant models and probabilistic interpretations.

Canonical Ordering Approaches.

In section 2.1, we saw how permutation invariance can be achieved by mapping permutations to a canonical ordering.

Rather than trying to define a good canonical ordering, one can try to learn it from the data, however searching among all |h|!

permutations for one that correlates with the task of interest is a difficult discrete optimization problem.

Recently, BID36 proposed a method that computes the posterior distribution of all permutations, conditioned on the model and the data.

This posterior-sampling approach is intractable for large inputs, unfortunately.

We note in passing that BID36 is interested in permutation-invariant outputs, and that Janossy pooling is also trivially applicable to these tasks.

Vinyals et al. (2016) proposes a heuristic using ancestral sampling while learning the model.

k-ary Janossy Pooling Approaches.

In section 2.2 we described k-ary Janossy pooling, which considers k-order relationships in the input vector h to simplify optimization.

DeepSets (Zaheer et al., 2017) can be characterized as unary Janossy pooling (i.e., k-ary for k = 1). .

Qi et al. FORMULA0 and BID34 propose similar unary Janossy pooling models.

BID8 proposes to add inductive biases to the DeepSets model in the form of monotonicity constraints with respect to the vector valued elements of the input sequence by modeling f and ρ with Deep Lattice Networks (You et al., 2017) ; one can extend BID8 by using higher-order (k > 1) pooling.

Exploiting dependencies within a sequence to learn a permutation-invariant function has been discussed elsewhere.

For instance BID38 exploits pairwise relationships to perform relational reasoning about pairs of objects in an image and Battaglia et al. FORMULA0 contemplates modeling the center of mass of a solar system by including the pairwise interactions among planets.

However, Janossy pooling provides a general framework for capturing dependencies within a permutation-invariant pooling layer.

Permutation Sampling Approaches.

In section 2.3 we have seen a that permutation sampling can be used as a stochastic gradient procedure (π-SGD) to learn a model with a Janossy pooling layer.

The learned model provides only an approximate solution to original permutation-invariant function.

Permutation sampling has been used as a heuristic (without a theoretical justification) in both BID29 and BID15 , which found that randomly permuting sequences and feeding them forward to an LSTM is effective in relational learning tasks that require permutation-invariant pooling layers.

Probabilistic Interpretation and Other Invariances Our work has a strong connection with finite exchangeability.

Some researchers may be more familiar with the concept of infinite exchangeability through de Finetti's theorem BID10 BID11 , which imposes strong structural requirements: the probability of any subsequence must equal the marginalized probability of the original sequence (projectivity).

BID21 noted the importance of this property for generative models and propose a model that learns a distribution without variational approximations.

Finite exchangeability drops this projectivity requirement BID11 , which in general, cannot be simplified beyond first sampling the number of observations m, and then sampling their locations from some exchangeable but non- BID9 .

Equivalently, de Finetti's theorem for infinitely exchangeable sequences implies that the joint distribution can represented as a mixture distribution over conditionally independent random variables (given θ) BID10 BID31 whereas the probability distribution of a finitely exchangeability sequence is a mixture over dependent random variables as shown by BID11 .

DISPLAYFORM0 In comparison, the restrictive assumption of letting k = 1 in k-ary Janossy Pooling yields the form of a log-likelihood of conditionally iid random variables (consider f a log pdf), the strong requirement of de Finetti's theorem for infinitely exchangeable sequences.

Conversely, higher-order Janossy pooling was designed to exploit dependencies among the random variables such as those that arise under finitely exchangeable distributions.

Indeed, finite exchangeability also arises from the theory of spatial point processes; our framework of Janossy pooling is inspired by Janossy densities BID9 , which model the finite exchangeable distributions as mixtures of non-exchangeable distributions applied to permutations.

This literature also studies simplified exchangeable point processes such as finite Gibbs models (Vo et al., 2018; BID27 that restrict the structure of p exch to fixed-order dependencies, and are related to k-ary Janossy.

More broadly, there are other connections between permutation-invariant deterministic functions and exchangeability in probability distributions, as recently discussed by BID2 .

There, the authors also contemplate more general invariances through the language of group actions.

An example is permutation equivariance: one form of permutation equivariance asserts that f (X π ) = f (X) π ∀π ∈ Π |X| where f (X) is a sequence of length greater than 1.

BID35 provides a weight-sharing scheme for maintaining general neural network equivariances characterized as automorphisms of a colored multi-edged bipartite graph.

BID16 proposes a matrix completion model invariant to (possibly separate) permutations of the rows or columns.

Other invariances are studied through a probabilistic perspective in BID31 .

Our approach of permutation-invariance through Janossy pooling unifies a number of existing approaches, and opens up avenues to develop both new methodological extensions, as well as better theory.

Our paper focused on two main approaches: k-ary interactions and random permutations.

The former involves exact Janossy pooling for a restricted class of functions f .

Adding an additional neural network ρ can recover lost model capacity and capture additional higher-order interactions, but hurts tractability and identifiability.

Placing restrictions on ρ (convexity, Lipschitz continuity etc.) can allow a more refined control of this trade-off, allowing theoretical and empirical work to shed light on the compromises involved.

The second was a random permutation approach which conversely involves no clear trade-offs between model capacity and computation when ρ is made more complex, instead it modifies the relationship between the tractable approximate loss J and the original Janossy loss L.

While there is a difference between J and L, we saw the strongest empirical performance coming from this approach in our experiments (shown in the last row of TAB0 ; future work is required to identify which problems π-SGD is best suited for and when its conver-gence criteria are satisfied.

Further, a better understanding how the loss-functions L and J relate to each other can shed light on the slightly black-box nature of this procedure.

It is also important to understand the relationship between the random permutation optimization to canonical ordering and how one might be used to improve the other.

Finally, it is important to apply our methodology to a wider range of applications.

Two immediate domains are more challenging tasks involving graphs and tasks involving non-Poisson point processes.

is now a summation over only |h|!/(|h| − k)!

terms.

We can conclude that

Next, we restate and prove the remaining portion of Theorem 2.1.

Theorem 2.1.

For any k ∈ N, define F k as the set of all permutation-invariant functions that can be represented by Janossy pooling with k-ary dependencies.

Then, F k−1 is a proper subset of F k if the space H is not trivial (i.e. if the cardinality of H is greater than 1).

Thus, Janossy pooling with k-ary dependencies can express any Janossy pooling function with (k − 1)-ary dependencies, but the converse does not hold.

Proof.

DISPLAYFORM0 Consider any element f k−1 ∈ F k−1 , and write f (|h|, · ; θ (f ) ) for its associated Janossy function.

For any sequence h, f (|h|, DISPLAYFORM1 , where the function f + looks at its first k elements.

Thus, DISPLAYFORM2 where f k is the Janossy function associated with f + and thus belongs to F k .(F k ⊂ F k−1 ): the case where k = 1 is trivial, so assume k > 1.

We will demonstrate the existence of DISPLAYFORM3 Let f k and f k−1 be associated with f k and f k−1 , respectively.

Thus, for any f k−1 and any θ DISPLAYFORM0 where Π {1,...,|h|}\j denotes the set of permutation functions defined on {1, 2, . . . , j − 1, j + 1, . . .

, |h|} and (h −j )π is a permutation of the sequence h 1 , . . .

, h j−1 , h j+1 , . . .

, h |h| .

This can be written as DISPLAYFORM1 Now, f k−1 = f k if and only if their quotient in equation 13 is unity for all h. But this is clearly not possible in general unless H is a singleton, which we have precluded in our assumptions.

Proposition 2.2 is repeated below and is followed by a more rigorous restatement.

Proposition 2.2. [π-SGD Convergence]

The optimization of π-SGD enjoys properties of almost sure convergence to the optimal θ under similar conditions as SGD.The following statement is similar to that in Yuille (2004) , which also provides intuition behind the theoretical assumptions, which are indeed quite general.

See also (Younes, 1999) .

This is a familiar application of stochastic approximation algorithms already used in training neural networks.

Proposition A.1 (π-SGD Convergence).

Consider the π-SGD algorithm in Definition 2.3.

If (a) there exists a constant M > 0 such that for all θ, −G DISPLAYFORM2 , where G t is the true gradient for the full batch over all permutations, DISPLAYFORM3 , and θ is the optimum.(b) there exists a constant δ > 0 such that for all θ, DISPLAYFORM4 2 ), where the expectation is taken with respect to all the data prior to step t.

Then, the algorithm in equation 9 converges to θ with probability one.

Proof.

First, we can show that E t [Z t ] = G t by equation 10, the linearity of the derivative operator, and the fact that the permutations are independently sampled for each training example in the minibatch and are assumed independent of θ.

That equation 9 converges to θ is a consequence of our conditions and the supermartingale convergence theorem (Grimmett & Stirzaker, 2001, pp.

481) .

The following argument follows Yuille (2004) .

DISPLAYFORM5 t , and DISPLAYFORM6 .

Note that C t is positive for a sufficiently large t, and ∞ t=1 B t ≤ ∞ by our definition of η t (Definition 2.3).

We will demonstrate that E t [A t ] ≤ A t−1 +B t−1 −C t−1 , for all t, in the Supplementary Material from which it follows that A t converges to zero with probability one and The accuracy scores for all models (including the LSTM) on the sequence arithmetic tasks are shown in TAB0 , except here we show additional rows representing models that use LSTM as f .

We chose accuracy (0-1 loss) to be consistent with Zaheer et al. (2017) ; here we report mean absolute error to evaluate the differences it makes on our results.

These can be found in TAB2 .

The message is similar to the one told by accuracy scores; there is a drop in the mean absolute error as the value of k increases and when using more sampled permutations at test-time (e.g., Janossy-20inf-LSTM versus Janossy-1inf-LSTM).

Again, the power of using an RNN for f and training with π-SGD is salient on the variance task where it is important to exploit dependencies in the sequence.

Beyond the performance gains, we also observe a drop in variance when sampling more permutations at test time.

Furthermore, as discussed in the implementation section, we constructed k-ary models to have the same number of parameters regardless of k for the results reported in the main body.

We show results where this constraint is relaxed in TAB4 .

Here we see a modest improvement of k-ary models which stands to reason considering the embedding dimension fed to the Janossy pooling layer was reduced from 100 with k = 1 to 33 with k = 3 (please see the implementation section for details).

DISPLAYFORM7 For the graph tasks, the plot of performance as a function of number of inference-time permutations is shown in FIG8 .

Table 3: Full table showing the Accuracy (and RMSE for (2017) , which was written in Keras BID7 , and subsequently ported to PyTorch.

For k-ary models with k ∈ {2, 3}, we always sort the sequence x beforehand to reduce the number of combinations we need to sum over.

In the notation of FIG0 , h is an Embedding with dimension of floor( 100 k ) (to keep the total number of parameters consistent for each k as discussed below), f is either an MLP with a single hidden layer or an RNN depending on the model (k-ary Janossy or full-Janossy, respectively), and ρ is either a linear dense layer or one hidden layer followed by a linear dense layer.

The MLPs in f have 30 neurons whereas the MLPs in ρ have 100 neurons, the LSTMs have 50 neurons, and the GRUs have 80 hidden neurons.

All activations are tanh except for the output layer which is linear.

We chose 100 for the embedding dimension to be consistent with Zaheer et al. (2017) .For the k-ary results shown in the body, we made sure the number of parameters was consistent for k ∈ {1, 2, 3} (see Table 8 ).

We unify the number of parameters by adjusting the output dimension of the embedding.

We also experimented with relaxing the restriction that k-ary models have the same numbers of parameters TAB4 , and the numbers of parameters in these models is also shown in Table 8 .

For the LSTM than GRU models, we follow the choice of Zaheer et al. (2017) which also reports that the choices were made to keep the numbers of parameters consistent.

Optimization is done with Adam (Kingma & Ba, 2015) with a tuned the learning rate, searching over {0.01, 0.001, 0.0001, 0.00001}. Training was performed on GeForce GTX 1080 Ti GPUs.

Graph-based tasks The datasets used for this task are summarized in Table 9 .

Our implementation is in PyTorch using Python 2.7, following the PyTorch code associated with BID15 .

That repo did not include an LSTM aggregator, so we implemented our own following the TensorFlow implementation of GraphSAGE, and describe it here.

At the beginning of every forward pass, each vertex v is associated with a p-dimensional vertex attribute h (see Table9) .

For every vertex in a batch, k 1 neighbors of v are sampled, their order is shuffled, and their features are fed through an LSTM.

From the LSTM, we take the short-term hidden state associated with the last element in the input sequence (denoted h (T ) in the LSTM literature, but this h is not to be confused with a vertex attribute).

This short-term hidden state is passed through a fully connected layer to yield a vector of dimension q 2 , where q is a user-specified positive even integer referred to as the embedding dimension.

The vertex's own attribute h is also fed forward through a fully connected layer with q 2 output neurons.

At this point, for each vertex, we have two representation vectors of size q 2 representing the vertex v and its neighbor multiset, which we concatenate to form an embedding of size q. This describes one convolution layer, and it is repeated a second time with a distinct set of learnable weights for the fully connected and LSTM layers, sampling k 2 vertices from each neighborhood and using the embeddings of the first layer as features.

After each convolution, we may optionally apply a ReLU activation and/or embedding normalization, and we follow the decisions shown in the GraphSAGE code BID15 .

After both convolution operations, we apply a final fully connected layer to obtain the score, followed by a softmax (Cora, Pubmed) or sigmoid (PPI).

The loss function is cross entropy for Cora and Pubmed, and binary cross entropy for PPI.

Last, the definition above for f caused difficulties in environments such as figure, so we defined and occasionally used in L A T E X \newcommand{\harrowStable}[1]{\overset{\rightharpoonup}{#1}}.

<|TLDR|>

@highlight

We propose Janossy pooling, a method for learning deep permutation invariant functions designed to exploit relationships within the input sequence and tractable inference strategies such as a stochastic optimization procedure we call piSGD