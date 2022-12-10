The capability of making interpretable and self-explanatory decisions is essential for developing responsible machine learning systems.

In this work, we study the learning to explain the problem in the scope of inductive logic programming (ILP).

We propose Neural Logic Inductive Learning (NLIL), an efficient differentiable ILP framework that learns first-order logic rules that can explain the patterns in the data.

In experiments, compared with the state-of-the-art models, we find NLIL is able to search for rules that are x10 times longer while remaining x3 times faster.

We also show that NLIL can scale to large image datasets, i.e. Visual Genome, with 1M entities.

: A scene-graph can describe the relations of objects in an image.

The NLIL can utilize this graph and explain the presence of objects Car and Person by learning the first-order logic rules that characterize the common sub-patterns in the graph.

The explanation is globally consistent and can be interpreted as commonsense knowledge.

The recent years have witnessed the growing success of deep learning models in a wide range of applications.

However, these models are also criticized for the lack of interpretability in its behavior and decision making process (Lipton, 2016; Mittelstadt et al., 2019) , and for being data-hungry.

The ability to explain its decision is essential for developing a responsible and robust decision system (Guidotti et al., 2019) .

On the other hand, logic programming methods, in the form of first-order logic (FOL), are capable of discovering and representing knowledge in explicit symbolic structure that can be understood and examined by human (Evans & Grefenstette, 2018) .

In this paper, we investigate the learning to explain problem in the scope of inductive logic programming (ILP) which seeks to learn first-order logic rules that explain the data.

Traditional ILP methods (Galárraga et al., 2015) rely on hard matching and discrete logic for rule search which is not tolerant for ambiguous and noisy data (Evans & Grefenstette, 2018) .

A number of works are proposed for developing differentiable ILP models that combine the strength of neural and logicbased computation (Evans & Grefenstette, 2018; Campero et al., 2018; Rocktäschel & Riedel, 2017; Payani & Fekri, 2019; Dong et al., 2019) .

Methods such as ∂ILP (Evans & Grefenstette, 2018) are referred to as forward-chaining methods.

It constructs rules using a set of pre-defined templates and evaluates them by applying the rule on background data multiple times to deduce new facts that lie in the held-out set (related works available at Appendix A).

However, general ILP problem involves several steps that are NP-hard: (i) the rule search space grows exponentially in the length of the rule; (ii) assigning the logic variables to be shared by predicates grows exponentially in the number of arguments, which we refer as variable binding problem; (iii) the number of rule instantiations needed for formula evaluation grows exponentially in the size of data.

To alleviate these complexities, most works have limited the search length to within 3 and resort to template-based variable assignments, limiting the expressiveness of the learned rules (detailed discussion available at Appendix B).

Still, most of the works are limited in small scale problems with less than 10 relations and 1K entities.

On the other hand, multi-hop reasoning methods (Guu et al., 2015; Lao & Cohen, 2010; Lin et al., 2015; Gardner & Mitchell, 2015; Das et al., 2016) are proposed for the knowledge base (KB) completion task.

Methods such as NeuralLP can answer the KB queries by searching for a relational path that leads from the subject to the object.

These methods can be interpreted in the ILP domain where the learned relational path is equivalent to a chain-like first-order rule.

Compared to the template-based counterparts, methods such as NeuralLP is highly efficient in variable binding and rule evaluation.

However, they are limited in two aspects: (i) the chain-like rules represent a subset of the Horn clauses, and are limited in expressing complex rules such as those shown in Figure 1;  (ii) the relational path is generated while conditioning on the specific query, meaning that the learned rule is only valid for the current query.

This makes it difficult to learn rules that are globally consistent in the KB, which is an important aspect of a good explanation.

In this work, we propose Neural Logic Inductive Learning (NLIL), a differentiable ILP method that extends the multi-hop reasoning framework for general ILP problem.

NLIL is highly efficient and expressive.

We propose a divide-and-conquer strategy and decompose the search space into 3 subspaces in a hierarchy, where each of them can be searched efficiently using attentions.

This enables us to search for x10 times longer rules while remaining x3 times faster than the state-of-theart methods.

We maintain the global consistency of rules by splitting the training into rule generation and rule evaluation phase, where the former is only conditioned on the predicate type that is shared globally.

And more importantly, we show that a scalable ILP method is widely applicable for model explanations in supervised learning scenario.

We apply NLIL on Visual Genome (Krishna et al., 2016) dataset for learning explanations for 150 object classes over 1M entities.

We demonstrate that the learned rules, while maintaining the interpretability, have comparable predictive power as densely supervised models, and generalize well with less than 1% of the data.

Supervised learning typically involves learning classifiers that map an object from its input space to a score between 0 and 1.

How can one explain the outcome of a classifier?

Recent works on interpretability focus on generating heatmaps or attention that self-explains a classifier (Ribeiro et al., 2016; Chen et al., 2018; Olah et al., 2018) .

We argue that a more effective and humanintelligent explanation is through the description of the connection with other classifiers.

For example, consider an object detector with classifiers Person(X), Car(X), Clothing(X) and Inside(X, X ) that detects if certain region contains a person, a car, a clothing or is inside another region, respectively.

To explain why a person is present, one can leverage its connection with other attributes, such as "X is a person if it's inside a car and wearing clothing", as shown in Figure 1 .

This intuition draws a close connection to a longstanding problem of first-order logic literature, i.e. Inductive Logic Programming (ILP).

A typical first-order logic system consists of 3 components: entity, predicate and formula.

Entities are objects x ∈ X .

For example, for a given image, a certain region is an entity x, and the set of all possible regions is X .

Predicates are functions that map entities to 0 or 1, for example Person : x → {0, 1}, x ∈ X .

Classifiers can be seen as soft predicates.

Predicates can take multiple arguments, e.g. Inside is a predicate with 2 inputs.

The number of arguments is referred to as the arity.

Atom is a predicate symbol applied to a logic variable, e.g. Person(X) and Inside(X, X ).

A logic variable such as X can be instantiated into any object in X .

A first-order logic (FOL) formula is a combination of atoms using logical operations {∧, ∨, ¬} which correspond to logic and, or and not respectively.

given a set of predicates P = {P 1 ...P K }, we define the explanation of a predicate P k as a first-order logic entailment

where P k (X, X ) is the head of the entailment, and it will become P k (X) if it is a unary predicate.

A is defined as the rule body and is a general formula, e.g. conjunction normal form (CNF), that is made of atoms with predicate symbols from P and logic variables that are either head variables X, X or one of the body variables Y = {Y 1 , Y 2 , ...}.

By using the logic variables, the explanation becomes transferrable as it represents the "lifted" knowledge that does not depend on the specific data.

It can be easily interpreted.

For example,

represents the knowledge that "if an object is inside the car with clothing on it, then it's a person".

To evaluate a formula on the actual data, one grounds the formula by instantiating all the variables into objects.

For example, in Figure 1 , Eq. (2) is applied to the specific regions of an image.

Given a relational knowledge base (KB) that consists of a set of facts

where P i ∈ P and x i , x i ∈ X .

The task of learning FOL rules in the form of Eq.(1) that entail target predicate P * ∈ P is called inductive logic programming.

For simplicity, we consider unary and binary predicates for the following contents, but this definition can be extended to predicates with higher arity as well.

The ILP problem is closely related to the multi-hop reasoning task on the knowledge graph (Guu et al., 2015; Lao & Cohen, 2010; Lin et al., 2015; Gardner & Mitchell, 2015; Das et al., 2016) .

Similar to ILP, the task operates on a KB that consists of a set of predicates P. Here the facts are stored with respect to the predicate P k which is represented as a binary matrix M k in {0, 1}

|X |×|X | .

This is an adjacency matrix, meaning that x i , P k , x j is in the KB if and only if the (i, j) entry of M k is 1.

Given a query q = x, P * , x .

The task is to find a relational path x

− −− → x , such that the two query entities are connected.

Formally, let v x be the one-hot encoding of object x with dimension of |X |.

Then, the (t)th hop of the reasoning along the path is represented as

where M (t) is the adjacency matrix of the predicate used in (t)th hop.

The v (t) is the path features vector, where the jth element v (t) j counts the number of unique paths from x to x j (Guu et al., 2015) .

After T steps of reasoning, the score of the query is computed as

For each q, the goal is to (i) find an appropriate T and (ii) for each t ∈ [1, 2, ..., T ], find the appropriate M (t) to multiply, such that Eq.(3) is maximized.

These two discrete picks can be relaxed as learning the weighted sum of scores from all possible paths, and weighted sum of matrices at each step.

Let

be the soft path selection function parameterized by (i) the path attention vector s ψ = [s

ψ ] T that softly picks the best path with length between 1 to T that answers the query, and (ii) the operator attention vectors S ϕ = [s

ϕ softly picks the M (t) at (t)th step.

Here we omit the dependence on M k for notation clarity.

These two attentions are generated with a model

with learnable parameters w. For methods such as (Guu et al., 2015; Lao & Cohen, 2010) , T(x; w) is a random walk sampler which generates one-hot vectors that simulate the random walk on the graph starting from x. And in NeuralLP , T(x; w) is an RNN controller that generates a sequence of normalized attention vectors with v x as the initial input.

Therefore, the objective is defined as arg max

Learning the relational path in the multi-hop reasoning can be interpreted as solving an ILP problem with chain-like FOL rules )

Compared to the template-based ILP methods such as ∂ILP, this class of methods is efficient in rule exploration and evaluation.

However, (P1) generating explanations for supervised models puts a high demand on the rule expressiveness.

The chain-like rule space is limited in its expressive power because it represents a constrained subspace of the Horn clauses rule space.

For example, Eq. (2) is a Horn clause and is not chain-like.

And the ability to efficiently search beyond the chain-like rule space is still lacking in these methods.

On the other hand, (P2) the attention generator T(x; w) is dependent on x, the subject of a specific query q, meaning that the explanation generated for target P * can vary from query to query.

This makes it difficult to learn FOL rules that are globally consistent in the KB.

In this section, we show the connection between the multi-hop reasoning methods with the general logic entailment defined in Eq.(1).

Then we propose a hierarchical rule space to solve (P1), i.e. we extend the chain-like space for efficient learning of more expressive rules.

In Eq.(1), variables that only appear in the body are under existential quantifier.

We can turn Eq.(1) into Skolem normal form by replacing all variables under existential quantifier with functions with respect to X and X ,

If the functions are known, Eq.(7) will be much easier to evaluate than Eq.(1).

Because grounding this formula only requires to instantiate the head variables, and the rest of the body variables are then determined by the deterministic functions.

Functions in Eq. (7) can be arbitrary.

But what are the functions that one can utilize?

We propose to adopt the notions in section 2.2 and treat each predicate as an operator, such that we have a subspace of the functions Φ = {ϕ 1 , ..., ϕ K }, where

where U and B are the sets of unary and binary predicates respectively.

The operator of the unary predicate takes no input and is parameterized with a diagonal matrix.

Intuitively, given a subject entity x, ϕ k returns the set embedding (Guu et al., 2015) that represents the object entities that, together with the subject, satisfy the predicate P k .

For example, let v x be the one-hot encoding of an object in the image, then ϕ Inside (v x ) returns the objects that spatially contain the input box.

For unary predicate such as Car(X), its operator ϕ Car () = M car 1 takes no input and returns the set of all objects labelled as car.

Since we only use Φ, a subspace of the functions, the existential variables that can be represented by the operator calls, denoted asŶ, also form the subsetŶ ⊆ Y.

This is slightly constrained from Eq.(1).

For example, in Person(X) ← Car(Y ), Y can not be interpreted as the operator call from X. However, we argue that such rules are generally trivial.

For example, it's not likely to infer "an image contains a person" by simply checking if "there is any car in the image".

Therefore, any FOL formula that complies with Eq. (7) can now be converted into the operator form and vice versa.

For example, Eq.(2) can be written as

where the variable Y 1 and Y 2 are eliminated.

Note that this conversion is not unique.

For example, Car(ϕ Inside (X)) can be also written as Inside(X, ϕ Car ()).

The variable binding problem now becomes equivalent to the path-finding problem in section 2.2, where one searches for the appropriate chain of operator calls that can represent the variable inŶ.

Tree-like

Figure 2: Factor graphs of example chainlike, tree-like and conjunctions of rules.

Each rule type is the subset of the latter.

Succ stands for successor.

As discussed above, the Eq.(3) is equivalent to a chain-like rule.

We want to extend this notion and be able to represent more expressive rules.

To do this, we introduce the notion of primitive statement ψ.

Note that an atom is defined as a predicate symbol applied to specific logic variables.

Similarly, we define a predicate symbol applied to the head variables or those inŶ as a primitive statement.

For example, in Eq. (8), ψ 1 = Car(ϕ Inside (X)) and ψ 2 = On(ϕ Clothing (), X) are two primitive statements.

Similar to an atom, each primitive statement is a mapping from the input space to a scalar confidence score, i.e. ψ :

, their mappings are defined as

where σ(·) is the sigmoid function.

Note that we give unary ψ a dummy input x for notation convenience.

For example, in

Its value is computed as

Compared to Eq.(3), Eq.(9) replaces the target v x into another relational path.

This makes it possible to represent "correlations" between two variables, and the path that starts from the unary operator, e.g. ϕ Eye ().

To see this, one can view a FOL rule as a factor graph with logic variables as the nodes and predicates as the potentials .

And running the operator call is essentially conducting the belief propagation over the graph in a fixed direction.

As shown in Figure 2 , primitive statement is capable of representing the tree-like factor graphs, which significantly improves the expressive power of the learned rules.

Similarly, Eq.(9) can be relaxed into weighted sums.

In Eq.(6), all relational paths are summed with a single path attention vector s ψ .

We extend this notion by assigning separate vectors for each argument of the statement ψ.

Let S ψ , S ψ ∈ R K×T be the path attention matrices for the first and second argument of all statements in Ψ, i.e. s ψ,k and s ψ,k are the path attention vectors of the first and second argument of the kth statement.

Then we have we want to further extend the rule search space by exploring the logic combinations of primitive statements, via {∧, ∨, ¬}, as shown in Figure 2 .

To do this, we utilize the soft logic not and soft logic and operations

where p, q ∈ [0, 1].

Here we do not include the logic ∨ operation because it can be implicitly represented as

be the set of primitive statements with all possible predicate symbols.

We define the formula set at lth level as

, where each element in the formula set {f : f ∈ F l } is called a formula such that f :

Intuitively, we define the logic combination space in a similar way as that in pathfinding: the initial formula set contains only primitive statements Ψ, because they are formulas by themselves.

For the l − 1th formula set F l−1 , we concatenate it with its logic negation, which yieldŝ F l−1 .

Then each formula in the next level is the logic and of two formulas fromF l−1 .

Enumerating all possible combinations at each level is expensive, so we set up a memory limitation C to indicate the maximum number of combinations each level can keep track of 1 .

In other words, each level F l is to search for C logic and combinations on formulas from the previous levelF l−1 , such that the cth formula at the lth level f lc is

As an example, for Ψ = {ψ 1 , ψ 2 } and C = 2, one possible level sequence is F 0 = {ψ 1 , ψ 2 },

..} and etc.

To collect the rules from all levels, the final level L is the union of previous sets, i.e.

Note that Eq.(11) does not explicitly forbid trivial rules such as ψ 1 * (1 − ψ 1 ) that is always true regardless of the input.

This is alleviated by introducing nonexistent queries during the training (detailed discussion at section 5).

Again, the rule selection can be parameterized into the weighted-sum form with respect to the attentions.

We define the formula attention tensors as S f , S f ∈ R L−1×C×2C , such that f lc is the product of two summations over the previous outputs weighted by attention vectors s f,lc and s f,lc respectively 2 .

Formally, we have

where f l−1 (x, x ) ∈ R 2C is the stacked outputs of all formulas f ∈F l−1 with arguments x, x .

Finally, we want to select the best explanation and compute the score for each query.

Let s o be the

Iterate Concat attention vector over F L , so the output score is defined as

An overview of the relaxed hierarchical rule space is illustrated in Figure 3 .

We have defined a hierarchical rule space as shown in Figure 3 , where the discrete selections on the operators, statements and logic combinations are all relaxed into the weight sums with respect to a series of attention parameters S ϕ , S ψ , S ψ , S f , S f and s o .

In this section, we solve (P2), i.e. we propose a differentiable model that generates these attentions without conditioning on the specific query.

The goal of NLIL is to generate data-independent FOL rules.

In other words, for each target predicate P * , its rule set F L and the final output rule should remain unchanged for all the queries q = x, P * , x (which is different from that in Eq.(5)).

To do this, we define the learnable embeddings of all predicates as H = [h 1 , .., h K ] T ∈ R K×d , and the embeddings for the "dummy" arguments X and X as e X , e X ∈ R d .

We define the attention generation model as

where h * is the embedding of P * , such that attentions only vary with respect to P * .

As shown in Figure 4 , we propose a stack of three Transformer (Vaswani et al., 2017 ) networks for attention generator T. Each module is designed to mimic the actual evaluation that could happen during the operator call, primitive statement evaluation and formula computation respectively with neural networks and "dummy" embeddings.

And the attention matrices generated during this simulated evaluation process are kept for evaluating Eq.(14).

A MultiHeadAttn is a standard Transformer module such that MultiHeadAttn :

where d is the latent dimension and q, v are the query and value dimensions respectively.

It takes the query Q and input value V (which will be internally transformed into keys and values), and returns the output value O and attention matrix S. Intuitively, S encodes the "compatibility" between query and the value, and O represents the "outcome" of a query given its compatibility with the input.

Operator search: For target predicate P * , we alter the embedding matrix H witĥ

such that the rule generation is predicate-specific.

Let q (t) ϕ be the learnable tth step operator query embedding.

The operator transformer module is parameterized aŝ

Here,V

ϕ is the dummy input embedding representing the starting points of the paths.

e ϕ is a learnable operator encoding such thatQ ϕ represents the embeddings of all operators Φ. Therefore, we consider thatV

ϕ encodes the outputs of the operator calls of K predicates.

And we aggregate the outputs with another MultiHeadAttn with respect to a single query q Primitive statement search:

T be the output embedding of T paths.

The path attention is generated as

Here, e ψ and e ψ are the first and second argument encodings, such that Q ψ and Q ψ encode the arguments of each statement in Ψ. The compatibility between paths and the arguments are computed with two MultiHeadAttns.

Finally, a FeedForward is used to aggregate the selections.

Its output V ψ ∈ R K×d represents the results of all statement evaluations in Ψ.

Formula search: Let Q f,l , Q f,l ∈ R C×d be the learnable queries of the first and second argument of formulas at lth level, and let V f,0 = V ψ .

The formula attention is generated aŝ

Here, e + , e ¬ are the learnable embeddings, such thatV f,l−1 represents the positive and negative states of the formulas at l − 1th level.

Similar to the statement search, the compatibility between the logic and arguments and the previous formulas are computed with two MultiHeadAttns.

And the embeddings of formulas at lth level V f,l are aggregated by a FeedForward.

Finally, let q o be the learnable final output query and let

The training of NLIL consists of two phases: rule generation and rule evaluation.

During generation, we run Eq.(15) to obtain the attentions S ϕ , S ψ , S ψ , S f , S f and s o for all P * s. For the evaluation phase, we sample a mini-batch of queries { x,

, and evaluate the formulas using Eq.(14).

Here, y is the query label indicating if the triplet exists in the KB or not.

We sample nonexistent queries to prevent the model from learning trivial rules that always output 1.

In the experiments, these negative queries are sampled uniformly from the target query matrix M * where the entry is 0.

Then the objective becomes arg min

Since the attentions are generated from Eq.(15) differentiably, the loss is back-propagated through the attentions into the Transformer networks for end-to-end training.

During training, the results from operator calls and logic combinations are averaged via attentions.

For validation and testing, we evaluate the model with the explicit FOL rules extracted from the ϕ is such a distribution over random variables k ∈ [1, K].

And the weighted sum is the expectation over M k .

Therefore, one can extract the explicit rules by sampling from the distributions (Kool et al., 2018; .

However, since we are interested in the best rules and the attentions usually become highly concentrated on one entity after convergence.

We replace the sampling with the arg max, where we get the one-hot encoding of the entity with the largest probability mass.

We first evaluate NLIL on classical ILP benchmarks and compare it with 4 state-of-the-art KB completion methods in terms of their accuracy and efficiency.

Then we show NLIL is capable of learning FOL explanations for object classifiers on a large image dataset when scene-graphs are present.

Though each scene-graph corresponds to a small KB, the total amount of the graphs makes it infeasible for all classical ILP methods.

We show that NLIL can overcome it via efficient stochastic training.

We evaluate NLIL together with two state-of-the-art differentiable ILP methods, i.e. NeuralLP and ∂ILP (Evans & Grefenstette, 2018) , and two structure embedding methods, TransE (Bordes et al., 2013) and RotatE (Sun et al., 2019) .

Detailed experiments setup is available at Appendix C.

Benchmark datasets: (i) Even-and-Successor (ES) benchmark is introduced in (Evans & Grefenstette, 2018) , which involves two unary predicates Even(X), Zero(X) and one binary predicate Succ(X, Y ).

The goal is to learn FOL rules over a set of integers.

The benchmark is evaluated with 10, 50 and 1K consecutive integers starting at 0; (ii) FB15K-237 is a subset of the Freebase knowledge base (Toutanova & Chen, 2015) containing general knowledge facts; (iii) WN18 (Bordes et al., 2013 ) is the subset of WordNet containing relations between words.

Statistics of datasets are provided in Table 2 .

Knowledge base completion: All models are evaluated on the KB completion task.

The benchmark datasets are split into train/valid/test sets.

The model is tasked to predict the probability of a fact triplet (query) being present in the KB.

We use Mean Reciprocal Ranks (MRR) and Hits@10 for evaluation metrics (see Appendix C for details).

Results on Even-and-Successor benchmark are shown in Table 5a .

Since the benchmark is noisefree, we only show the wall clock time for completely solving the task.

As we have previously mentioned, the forward-chaining method, i.e. ∂ILP scales exponentially in the number of facts and quickly becomes infeasible for 1K entities.

Thus, we skip its evaluation for other benchmarks.

Results on FB15K-237 and WN18 are shown in Table.

1.

Compared to NeuralLP, NLIL yields slightly higher scores.

This is due to the benchmarks favor symmetric/asymmetric relations or compositions of a few relations (Sun et al., 2019) , such that most valuable rules will already lie within the chain-like search space of NeuralLP.

Thus the improvements gained from a larger search space with NLIL are limited.

On the other hand, with the Transformer block and smaller model created for each target predicate, NLIL can achieve a similar score at least 3 times faster.

Compared to the structure embedding methods, NLIL is significantly outperformed by the current state-of-the-art, i.e. RotatE, on FB15K.

This is expected because NLIL searches over the symbolic space that is highly constrained.

However, the learned rules are still reasonably predictive, as its performance is comparable to that of TransE.

Scalability for long rules: we demonstrate that NLIL can explore longer rules efficiently.

We compare the wall clock time of NeuralLP and NLIL for performing one epoch of training against different maximum rule lengths.

As shown in Figure 5b , NeuralLP searches over a chain-like rule space thus scales linearly with the length, while NLIL searches over a hierarchical space thus grows in log scale.

The search time for length 32 in NLIL is similar to that for length 3 in NerualLP.

The ability to perform ILP efficiently extends the applications of NLIL to beyond canonical KB completion.

For example in visual object detection and relation learning, supervised models can learn to generate a scene-graph (As shown in Figure 1 ) for each image.

It consists of nodes each labeled as an object class.

And each pair of objects are connected with one type of relation.

The scene-graph can then be represented as a relational KB where one can perform ILP.

Learning the FOL rules on such an output of a supervised model is beneficial.

As it provides an alternative way of interpreting model behaviors in terms of its relations with other classifiers that are consistent across the dataset.

To show this, we conduct experiments on Visual Genome dataset (Krishna et al., 2016) .

The original dataset is highly noisy (Zellers et al., 2018) , so we use a pre-processed version available as the GQA dataset (Hudson & Manning, 2019) .

The scene-graphs are converted to a collection KBs, and its statistics are shown in Table 2 .

We filter out the predicates with less than 1500 occurrences.

The processed KBs contain 213 predicates.

Then we perform ILP on learning the explanations for the top 150 objects in the dataset.

Quantitatively, we evaluate the learned rules on predicting the object class labels on a held-out set in terms of their R@1 and R@5.

As none of the ILP works scale to this benchmark, we compare NLIL with two supervised baselines: (i) MLP-RCNN: a MLP classifier with RCNN features of the object (available in GQA dataset) as input; and (ii) Freq: a frequency-based baseline that predicts object label by looking at the mostly occurred object class in the relation that contains the target.

This method is nontrivial.

As noted in (Zellers et al., 2018) , a large number of triples in Visual Genome are highly predictive by knowing only the relation type and either one of the objects or subjects.

Explaining objects with rules: Results are shown in Table 3 .

We see that the supervised method achieves the best scores, as it relies on highly informative visual features.

On the other hand, NLIL achieves a comparable score on R@1 solely relying on KBs with sparse binary labels.

We note that NLIL outperforms Freq significantly.

This means the FOL rules learned by NLIL are beyond the superficial correlations exhibited by the dataset.

We verify this finding by showing the rules for top objects in Table 4 .

Induction for few-shot learning: Logic inductive learning is data-efficient and the learned rules are highly transferrable.

To see this, we vary the size of the training set and compare the R@1 scores for 3 methods.

As shown in Figure 5c , the NLIL maintains a similar R@1 score with less than 1% of the training set.

In this work, we propose Neural Logic Inductive Learning, a differentiable ILP framework that learns explanatory rules from data.

We demonstrate that NLIL can scale to very large datasets while being able to search over complex and expressive rules.

More importantly, we show that a scalable ILP method is effective in explaining decisions of supervised models, which provides an alternative perspective for inspecting the decision process of machine learning systems.

Inductive Logic Programming (ILP) is the task that seeks to summarize the underlying patterns shared in the data and express it as a set of logic programs (or rule/formulae) (Lavrac & Dzeroski, 1994) .

Traditional ILP methods such as AMIE+ (Galárraga et al., 2015) and RLvLR (Omran et al., 2018) relies on explicit search-based method for rule mining with various pruning techniques.

These works can scale up to very large knowledge bases.

However, the algorithm complexity grows exponentially in the size of the variables and predicates involved.

The acquired rules are often restricted to Horn clauses with a maximum length of less than 3, limiting the expressiveness of the rules.

On the other hand, compared to the differentiable approach, traditional methods make use of hard matching and discrete logic for rule search, which lacks the tolerance for ambiguous and noisy data.

The state-of-the-art differentiable forward-chaining methods focus on rule learning on predefined templates (Evans & Grefenstette, 2018; Campero et al., 2018; Ho et al., 2018) , typically in the form of a Horn clause with one head predicate and two body predicates with chain-like variables, i.e.

To evaluate the rules, one starts with a background set of facts and repeatedly apply rules for every possible triple until no new facts can be deduced.

Then the deduced facts are compared with a heldout ground-truth set.

Rules that are learned in this approach are in first-order, i.e. data-independent and can be readily interpreted.

However, the deducing phase can quickly become infeasible with a larger background set.

Although ∂ILP (Evans & Grefenstette, 2018 ) has proposed to alleviate by performing only a fixed number of steps, works of this type could generally scale to KBs with less than 1K facts and 100 entities.

On the other hand, differentiable backward-chaining methods such as NTP (Rocktäschel & Riedel, 2017) are more efficient in rule evaluation.

In (Minervini et al., 2018) , NTP 2.0 can scale to larges KBs such as WordNet.

However, FOL rules are searched with templates, so the expressiveness is still limited.

Another differentiable ILP method, i.e. Neural Logic Machine (NLM), is proposed in (Dong et al., 2019) , which learns to represent logic predicates with tensorized operations.

NLM is capable of both deductive and inductive learning on predicates with unknown arity.

However, as a forward-chaining method, it also suffers from the scalability issue as ∂ILP.

It involves a permutation operation over the tensors when performing logic deductions, making it difficult to scale to real-world KBs.

On the other hand, the inductive rules learned by NLM are encoded by the network parameters implicitly, so it does not support representing the rules with explicit predicate and logic variable symbols.

Multi-hop reasoning: Multi-hop reasoning methods (Guu et al., 2015; Lao & Cohen, 2010; Lin et al., 2015; Gardner & Mitchell, 2015; Das et al., 2016; such as NeuralLP construct rule on-the-fly when given a specific query.

It adopts a flexible ILP setting: instead of pre-defining templates, it assumes a chain-like Horn clause can be constructed to answer the query

And each step of the reasoning in the chain can be efficiently represented by matrix multiplication.

The resulting algorithm is highly scalable compared to the forward-chaining counter-parts and can learn rules on large datasets such as FreeBase.

However, this approach reasons over a single chainlike path, and the path is sampled by performing random walks that are independent on the task context (Das et al., 2017) , limiting the rule expressiveness.

On the other hand, the FOL rule is generated while conditioning on the specific query, making it difficult to extract rules that are globally consistent.

Link prediction with relational embeddings: Besides multi-hop reasoning methods, a number of works are proposed for KB completion using learnable embeddings for KB relations.

For example, In (Bordes et al., 2013; Sun et al., 2019; Balažević et al., 2019) it learns to map KB relations into vector space and predict links with scoring functions.

NTN (Socher et al., 2013) , on the other hand, parameterizes each relation into a neural network.

In this approach, embeddings are used for predicting links directly, thus its prediction cannot be interpreted as explicit FOL rules.

This is different from that in NLIL, where predicate embeddings are used for generating data-independent rules.

Standard ILP approaches are difficult and involve several procedures that have been proved to be NP-hard.

The complexity comes from 3 levels: first, the search space for a formula is vast.

The body of the entailment can be arbitrarily long and the same predicate can appear multiple times with different variables, for example, the Inside predicate in Eq.

(2) appears twice.

Most ILP works constrain the logic entailment to be Horn clause, i.e. the body of the entailment is a flat conjunction over literals, and the length limited within 3 for large datasets.

Second, constructing formulas also involves assigning logic variables that are shared across different predicates, which we refer to as variable binding.

For example, in Eq.(2), to express that a person is inside the car, we use X and Y to represent the region of a person and that of a car, and the same two variables are used in Inside to express their relations.

Different bindings lead to different meanings.

For a formula with n arguments (Eq.(2) has 7), there are O(n n ) possible assignments.

Existing ILP works either resort to constructing formula from pre-defined templates (Evans & Grefenstette, 2018; Campero et al., 2018) or from chain-like variable reference , limiting the expressiveness of the learned rules.

Finally, evaluating a formula candidate is expensive.

A FOL rule is data-independent.

To evaluate it, one needs to replace the variables with actual entities and compute its value.

This is referred to as grounding or instantiation.

Each variable used in a formula can be grounded independently, meaning a formula with n variables can be instantiated into O(C n ) grounded formulas, where C is the number of total entities.

For example, Eq.(2) contains 3 logic variables: X, Y and Z. To evaluate this formula, one needs to instantiate these variables into C 3 possible combinations, and check if the rule holds or not in each case.

However in many domains, such as object detection, such grounding space is vast (e.g. all possible bounding boxes of an image) making the full evaluation infeasible.

Many forward-chaining methods such as ∂ILP (Evans & Grefenstette, 2018) scales exponentially in the size of the grounding space, thus are limited to small scale datasets with less than 10 predicates and 1K entities.

Baselines: For NeuralLP, we use the official implementation at here.

For ∂ILP, we use the thirdparty implementation at here.

For TransE, we use the implementation at here.

For RotatE, we use the official implementation at here.

Model setting:

For NLIL, we create separate Transformer blocks for each target predicate.

All experiments are conducted on a machine with i7-8700K, 32G RAM and one GTX1080ti.

We use the embedding size d = 32.

We use 3 layers of multi-head attentions for each Transformer network.

The number of attention heads are set to number of heads = 4 for encoder, and the first two layers of the decoder.

The last layer of the decoder has one attention head to produce the final attention required for rule evaluation.

For KB completion task, we set the number of operator calls T = 2 and formula combinations L = 0, as most of the relations in those benchmarks can be recovered by symmetric/asymmetric relations or compositions of a few relations (Sun et al., 2019) .

Thus complex formulas are not preferred.

For FB15K-237, binary predicates are grouped hierarchically into domains.

To avoid unnecessary search overhead, we use the most frequent 20 predicates that share the same root domain (e.g. "award", "location") with the head predicate for rule body construction, which is a similar treatment as in .

For VG dataset, we set T = 3, L = 2 and C = 4.

Evaluation metrics:

Following the conventions in Bordes et al., 2013) we use Mean Reciprocal Ranks (MRR) and Hits@10 for evaluation metrics.

For each query x, P k , x , the model generates a ranking list over all possible groundings of predicate P k , with other groundtruth triplets filtered out.

Then MRR is the average of the reciprocal rank of the queries in their corresponding lists, and Hits@10 is the percentage of queries that are ranked within the top 10 in the list.

@highlight

An efficient differentiable ILP model that learns first-order logic rules that can explain the data.