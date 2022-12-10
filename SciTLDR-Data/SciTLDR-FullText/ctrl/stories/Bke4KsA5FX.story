Generative models forsource code are an interesting structured prediction problem, requiring to reason about both hard syntactic and semantic constraints as well as about natural, likely programs.

We present a novel model for this problem that uses a graph to represent the intermediate state of the generated output.

Our model generates code by interleaving grammar-driven expansion steps with graph augmentation and neural message passing steps.

An experimental evaluation shows that our new model can generate semantically meaningful expressions, outperforming a range of strong baselines.

Learning to understand and generate programs is an important building block for procedural artificial intelligence and more intelligent software engineering tools.

It is also an interesting task in the research of structured prediction methods: while imbued with formal semantics and strict syntactic rules, natural source code carries aspects of natural languages, since it acts as a means of communicating intent among developers.

Early works in the area have shown that approaches from natural language processing can be applied successfully to source code BID11 , whereas the programming languages community has had successes in focusing exclusively on formal semantics.

More recently, methods handling both modalities (i.e., the formal and natural language aspects) have shown successes on important software engineering tasks BID22 BID4 BID1 and semantic parsing (Yin & Neubig, 2017; BID20 ).However, current generative models of source code mostly focus on only one of these modalities at a time.

For example, program synthesis tools based on enumeration and deduction BID24 BID19 BID8 BID7 are successful at generating programs that satisfy some (usually incomplete) formal specification but are often obviously wrong on manual inspection, as they cannot distinguish unlikely from likely, "natural" programs.

On the other hand, learned code models have succeeded in generating realistic-looking programs BID17 BID5 BID18 BID20 Yin & Neubig, 2017) .

However, these programs often fail to be semantically relevant, for example because variables are not used consistently.

In this work, we try to overcome these challenges for generative code models and present a general method for generative models that can incorporate structured information that is deterministically available at generation time.

We focus our attention on generating source code and follow the ideas of program graphs BID1 ) that have been shown to learn semantically meaningful representations of (pre-existing) programs.

To achieve this, we lift grammar-based tree decoder models into the graph setting, where the diverse relationships between various elements of the generated code can be modeled.

For this, the syntax tree under generation is augmented with additional edges denoting known relationships (e.g., last use of variables).

We then interleave the steps of the generative procedure with neural message passing BID9 to compute more precise representations of the intermediate states of the program generation.

This is fundamentally different from sequential generative models of graphs BID14 BID23 , which aim to generate all edges and nodes, whereas our graphs are deterministic augmentations of generated trees.

To summarize, we present a) a general graph-based generative procedure for highly structured objects, incorporating rich structural information; b) ExprGen, a new code generation task focused on (a, u) ← insertChild(a, )

if is nonterminal type then 6:a ← Expand(c, a, u) 7: return a int ilOffsetIdx = Array.

IndexOf(sortedILOffsets, map.

ILOffset); int nextILOffsetIdx = ilOffsetIdx + 1; int nextMapILOffset = nextILOffsetIdx < sortedILOffsets.

Length ? sortedILOffsets [nextILOffsetIdx]

: int.

MaxValue; Figure 1 : Example for ExprGen, target expression to be generated is marked .

Taken from BenchmarkDotNet, lightly edited for formatting.generating small, but semantically complex expressions conditioned on source code context; and c) a comprehensive experimental evaluation of our generative procedure and a range of baseline methods from the literature.

The most general form of the code generation task is to produce a (partial) program in a programming language given some context information c. This context information can be natural language (as in, e.g., semantic parsing), input-output examples (e.g., inductive program synthesis), partial program sketches, etc.

Early methods generate source code as a sequence of tokens BID11 BID10 and sometimes fail to produce syntactically correct code.

More recent models are sidestepping this issue by using the target language's grammar to generate abstract syntax trees (ASTs) BID17 BID5 BID18 Yin & Neubig, 2017; BID20 , which are syntactically correct by construction.

In this work, we follow the AST generation approach.

The key idea is to construct the AST a sequentially, by expanding one node at a time using production rules from the underlying programming language grammar.

This simplifies the code generation task to a sequence of classification problems, in which an appropriate production rule has to be chosen based on the context information and the partial AST generated so far.

In this work, we simplify the problem further -similar to BID17 ; BID5 -by fixing the order of the sequence to always expand the left-most, bottom-most nonterminal node.

Alg.

1 illustrates the common structure of AST-generating models.

Then, the probability of generating a given AST a given some context c is DISPLAYFORM0 where a t is the production choice at step t and a <t the partial syntax tree generated before step t.

Code Generation as Hole Completion We introduce the ExprGen task of filling in code within a hole of an otherwise existing program.

This is similar, but not identical to the auto-completion function in a code editor, as we assume information about the following code as well and aim to generate whole expressions rather than single tokens.

The ExprGen task also resembles program sketching (Solar-Lezama, 2008) but we give no other (formal) specification other than the surrounding code.

Concretely, we restrict ourselves to expressions that have Boolean, arithmetic or string type, or arrays of such types, excluding expressions of other types or expressions that use project-specific APIs.

An example is shown in Fig. 1 .

We picked this subset because it already has rich semantics that can require reasoning about the interplay of different variables, while it still only relies on few operators and does not require to solve the problem of open vocabularies of full programs, where an unbounded number of methods would need to be considered.

In our setting, the context c is the pre-existing code around a hole for which we want to generate an expression.

This also includes the set of variables v 1 , . . .

, v that are in scope at this point, which can be used to guide the decoding procedure BID17 .

Note, however, that our method is not restricted to code generation and can be easily extended to all other tasks and domains that can be captured by variations of Alg.

1 (e.g. in NLP).

To tackle the code generation task presented in the previous section, we have to make two design choices: (a) we need to find a way to encode the code context c, v 1 , . . .

, v and (b) we need to construct a model that can learn p(a t | c, a <t ) well.

We do not investigate the question of encoding the context in this paper, and use two existing methods in our experiments in Sect.

5.

Both these encoders yield a distributed vector representation for the overall context, representations h t1 , . . .

, h t T for all tokens in the context, and separate representations for each of the in-scope variables v 1 , . . .

, v , summarizing how each variable is used in the context.

This information can then be used in the generation process, which is the main contribution of our work and is described in this section.

Overview Our decoder model follows the grammar-driven AST generation strategy of prior work as shown in Alg.

1.

The core difference is in how we compute the representation of the node to expand.

BID17 construct it entirely from the representation of its parent in the AST using a log-bilinear model.

BID20 construct the representation of a node using the parents of the AST node but also found it helpful to take the relationship to the parent node (e.g. "condition of a while") into account.

Yin & Neubig (2017) on the other hand propose to take the last expansion step into account, which may have finished a subtree "to the left".

In practice, these additional relationships are usually encoded by using gated recurrent units with varying input sizes.

We propose to generalize and unify these ideas using a graph to structure the flow of information in the model.

Concretely, we use a variation of attribute grammars BID12 from compiler theory to derive the structure of this graph.

We associate each node in the AST with two fresh nodes representing inherited resp.

synthesized information (or attributes).

Inherited information is derived from the context and parts of the AST that are already generated, whereas synthesized information can be viewed as a "summary" of a subtree.

In classical compiler theory, inherited attributes usually contain information such as declared variables and their types (to allow the compiler to check that only declared variables are used), whereas synthesized attributes carry information about a subtree "to the right" (e.g., which variables have been declared).

Traditionally, to implement this, the language grammar has to be extended with explicit rules for deriving and synthesizing attributes.

To transfer this idea to the deep learning domain, we represent attributes by distributed vector representations and train neural networks to learn how to compute attributes.

Our method for getRepresentation from Alg.

1 thus factors into two parts: a deterministic procedure that turns a partial AST a <t into a graph by adding additional edges that encode attribute relationships, and a graph neural network that learns from this graph.

Notation Formally, we represent programs as graphs where nodes u, v, . . .

are either the AST nodes or their associated attribute nodes, and typed directed edges u, τ, v ∈ E connect the nodes according to the flow of information in the model.

The edge types τ represent different syntactic or semantic relations in the information flow, discussed in detail below.

We write E v for the set of incoming edges into v. We also use functions like parent(a, v) and lastSibling(a, v) that look up and return nodes from the AST a (e.g. resp.

the parent node of v or the preceding AST sibling of v).

if v is variable then 7: DISPLAYFORM0 Example Consider the AST of the expression i -j shown in FIG0 (annotated with attribute relationships) constructed step by step by our model.

The AST derivation using the programming language grammar is indicated by shaded backgrounds, nonterminal nodes are shown as rounded rectangles, and terminal nodes are shown as rectangles.

We additionally show the variables given within the context as dashed rectangles at the bottom.

First, the root node, Expr, was expanded using the production rule (1) : Expr =⇒ Expr -Expr.

Then, its two nonterminal children were in turn expanded to the set of known variables using the produc-Published as a conference paper at ICLR 2019 Attribute nodes are shown overlaying their corresponding AST nodes.

For example, the root node is associated with its inherited attributes node 0 and with node 10 for its synthesized attributes.

For simplicity, we use the same representation for inherited and synthesized attributes of terminal nodes.

Edges in a <t We discuss the edges used in our neural attribute grammars (NAG) on our example below, and show them in FIG0 using different edge drawing styles for different edge types.

Once a node is generated, the edges connecting this node can be deterministically added to a <t (precisely defined in Alg.

2).

The list of different edge types used in our model is as follows:• Child (red) edges connect an inherited attribute node to the inherited attributes nodes of its children, as seen in the edges from node 0.

These are the connections in standard syntaxdriven decoders BID17 BID18 Yin & Neubig, 2017; BID20 ).•

Parent (green) edges connect a synthesized attribute node to the synthesized attribute node of its AST parent, as seen in the edges leading to node 10.

These are the additional connections used by the R3NN decoder introduced by BID18 .•

NextSib (black) edges connect the synthesized attribute node to the inherited attribute node of its next sibling (e.g. from node 5 to node 6).

These allow information about the synthesized attribute nodes from a fully generated subtree to flow to the next subtree.• NextUse (orange) edges connect the attribute nodes of a variable (since variables are always terminal nodes, we do not distinguish inherited from synthesized attributes) to their next use.

Unlike BID1 , we do not perform a dataflow analysis, but instead just follow the lexical order.

This can create edges from nodes of variables in the context c (for example, from node 1 to 4 in FIG0 ), or can connect AST leaf nodes that represent multiple uses of the same variable within the generated expressions.• NextToken (blue) edges connect a terminal node (a token) to the next token in the program text, for example between nodes 4 and 6.

• InhToSyn edges (not shown in FIG0 ) connect the inherited attributes nodes to its synthesized attribute nodes.

This is not strictly adding any information, but we found it to help with training.

The panels of FIG0 show the timesteps at which the representations of particular attribute nodes are computed and added to the graph.

For example, in the second step, the attributes for the terminal token i (node 4) in FIG0 are computed from the inherited attributes of its AST parent Expr (node 3), the attributes of the last use of the variable i (node 1), and the node label i. In the third step, this computed attribute is used to compute the synthesized attributes of its AST parent Expr (node 5).Attribute Node Representations To compute the neural attribute representation h v of an attribute node v whose corresponding AST node is labeled with v , we first obtain its incoming edges using Alg.

2 and then use the state update function from Gated Graph Neural Networks (GGNN) BID13 .

Thus, we take the attribute representations h ui at edge sources u i , transform them according to the corresponding edge type t i using a learned function f ti , aggregate them (by elementwise summation) and combine them with the learned embedding emb( v ) of the node label v using a function g: DISPLAYFORM1 In practice, we use a single linear layer for f ti and implement g as a gated recurrent unit .

We compute node representations in such an order that all h ui appearing on the right of (2) are already computed.

This is possible as the graphs obtained by repeated application of Alg.

2 are directed acyclic graphs rooted in the inherited attribute node of the root node of the AST.

We initialize the representation of the root inherited attribute to the representation returned by the encoder for the context information.

Choosing Productions, Variables & Literals We can treat picking production rules as a simple classification problem over all valid production rules, masking out those choices that do not correspond to the currently considered nonterminal.

For a nonterminal node v with label v and inherited attributes h v , we thus define DISPLAYFORM2 Here, m v is a mask vector whose value is 0 for valid productions v ⇒ . . .

and −∞ for all other productions.

In practice, we implement e using a linear layer.

Similarly, we pick variables from the set of variables V in scope using their representations h vvar (initially the representation obtained from the context, and later the attribute representation of the last node in the graph in which they have been used) by using a pointer network BID25 .

Concretely, to pick a variable at node v, we use learnable linear function k and define DISPLAYFORM3 Note that since the model always picks a variable from the set of in-scope variables V, this generation model can never predict an unknown or out-of-scope variable.

Finally, to generate literals, we combine a small vocabulary L of common literals observed in the training data and special UNK tokens for each type of literal with another pointer network that can copy one of the tokens t 1 . . .

t T from the context.

Thus, to pick a literal at node v, we define DISPLAYFORM4 Note that this is the only operation that may produce an unknown token (i.e. an UNK literal).

In practice, we implement this by learning two functions s L and s c , such that s L (h v ) produces a score for each token from the vocabulary and s c (h v , h ti ) computes a score for copying token t i from the context.

By computing a softmax over all resulting values and normalizing it by summing up entries corresponding to the same constant, we can learn to approximate the desired P (lit | h v ).

The different shapes and sizes of generated expressions complicate an efficient training regime.

However, note that given a ground truth target tree, we can easily augment it with all additional edges according to Alg.

2.

Given that full graph, we can compute a propagation schedule (intuitively, a topological ordering of the nodes in the graph, starting in the root node) that allows to repeatedly apply (2) to obtain representations for all nodes in the graph.

By representing a batch of graphs as one large (sparse) graph with many disconnected components, similar to BID1 , we can train our graph neural network efficiently.

We have released the code for this on https://github.com/Microsoft/graph-based-code-modelling.Our training procedure thus combines an encoder (cf.

Sect.

5), whose output is used to initialize the representation of the root and context variable nodes in our augmented syntax graph, the sequential graph propagation procedure described above, and the decoder choice functions (3) and (4).

We train the system end-to-end using a maximum likelihood objective without pre-trained components.

Additional Improvements We extend (3) with an attention mechanism BID16 that uses the state h v of the currently expanded node v as a key and the context token representations h t1 , . . .

, h t T as memories.

Experimentally, we found that extending Eqs. 4, 5 similarly did not improve results, probably due to the fact that they already are highly dependent on the context information.

Following BID20 , we provide additional information for Child edges.

To allow this, we change our setup so that some edge types also require an additional label, which is used when computing the messages sent between different nodes in the graph.

Concretely, we extend (2) by considering sets of unlabeled edges E v and labeled edges E v : DISPLAYFORM0 Thus for labeled edge types, f ti takes two inputs and we additionally introduce a learnable embedding for the edge labels.

In our experiments, we found it useful to label Child with tuples consisting of the chosen production and the index of the child, i.e., in FIG0 , we would label the edge from 0 to 3 with (2, 0), the edge from 0 to 6 with (2, 1), etc.

Furthermore, we have extended pickProduction to also take the information about available variables into account.

Intuitively, this is useful in cases of productions such as Expr =⇒ Expr.

Length, which can only be used in a well-typed derivation if an array-typed variable is available.

Thus, we extend e(h v ) from (3) to additionally take the representation of all variables in scope into account, i.e., e(h v , r({h vvar | var ∈ V})), where we have implemented r as a max pooling operation.

Source code generation has been studied in a wide range of different settings BID0 .

We focus on the most closely related works in language modeling here.

Early works approach the task by generating code as sequences of tokens BID11 BID10 , whereas newer methods have focused on leveraging the known target grammar and generate code as trees BID17 BID5 BID18 Yin & Neubig, 2017; BID20 ) (cf.

Sect.

2 for an overview).

While modern models succeed at generating "natural-looking" programs, they often fail to respect simple semantic rules.

For example, variables are often used without initialization or written several times without being read inbetween.

Existing tree-based generative models primarily differ in what information they use to decide which expansion rule to use next.

BID17 consider the representation of the immediate parent node, and suggest to consider more information (e.g., nearby tokens).

BID18 compute a fresh representation of the partial tree at each expansion step using R3NNs (which intuitively perform a leaf-to-root traversal followed by root-to-leaf traversal of the AST).

The PHOG model BID5 conditions generation steps on the result of learned (decision tree-style) programs, which can do bounded AST traversals to consider nearby tokens and non-terminal nodes.

The language also supports a jump to the last node with the same identifier, which can serve as syntactic approximation of data-flow analysis.

BID20 only use information about the parent node, but use neural networks specialized to different non-terminals to gain more fine-grained control about the flow of information to different successor nodes.

Finally, BID2 and Yin & Neubig (2017) follow a left-to-right, depth-first expansion strategy, but thread updates to single state (via a gated recurrent unit) through the overall generation procedure, thus giving the pickProduction procedure access to the full generation history as well as the representation of the parent node.

BID2 also suggest the use of attribute grammars, but use them to define a deterministic procedure that collects information throughout the generation process, which is provided as additional feature.

As far as we are aware, previous work has not considered a task in which a generative model fills a hole in a program with an expression.

Lanuage model-like methods take into account only the lexicographically previous context of code.

The task of BID21 is near to our ExprGen, but instead focuses on filling holes in sequences of API calls.

There, the core problem is identifying the correct function to call from a potentially large set of functions, given a sequence context.

In contrast, ExprGen requires to handle arbitrary code in the context, and then to build possibly complex expressions from a small set of operators.

BID1 consider similar context, but are only picking a single variable from a set of candidates, and thus require no generative modeling.

Dataset We have collected a dataset for our ExprGen task from 593 highly-starred open-source C # projects on GitHub, removing any near-duplicate files, following the work of BID15 .

We parsed all C # files and identified all expressions of the fragment that we are considering (i.e., restricted to numeric, Boolean and string types, or arrays of such values; and not using any user-defined functions).

We then remove the expression, perform a static analysis to determine the necessary context information and extract a sample.

For each sample, we create an abstract syntax tree by coarsening the syntax tree generated by the C # compiler Roslyn.

This resulted in 343 974 samples overall with 4.3 (±3.8) tokens per expression to generate, or alternatively 3.7 (± 3.1) production steps.

We split the data into four separate sets.

A "test-only" dataset is made up from ∼100k samples generated from 114 projects.

The remaining data we split into training-validation-test sets (3 : 1 : 1), keeping all expressions collected from a single source file within a single fold.

Samples from our dataset can be found in the supplementary material.

Our decoder uses the grammar made up by 222 production rules observed in the ASTs of the training set, which includes rules such as Expr =⇒ Expr + Expr for binary operations, Expr =⇒ Expr.

Equals(Expr) for built-in methods, etc.

Encoders We consider two models to encode context information.

Seq is a two-layer bi-directional recurrent neural network (using a GRU ) to encode the tokens before and after the "hole" in which we want to generate an expression.

Additionally, it computes a representation for each variable var in scope in the context in a similar manner: For each variable var it identifies usages before/after the hole and encodes each of them independently using a second bi-directional two-layer GRU, which processes a window of tokens around each variable usage.

It then computes a representation for var by average pooling of the final states of these GRU runs.

The second encoder G is an implementation of the program graph approach introduced by BID1 .

We follow the transformation used for the Varmisuse task presented in that paper, i.e., the program is transformed into a graph, and the target expression is replaced by a fresh dummy node.

We then run a graph neural network for 8 steps to obtain representations for all nodes in the graph, allowing us to read out a representation for the "hole" (from the introduced dummy node) and for all variables in context.

The used context information captured by the GNN is a superset of what existing methods (e.g. language models) consider.

Baseline Decoders We compare our model to re-implementations of baselines from the literature.

As our ExprGen task is new, re-using existing implementations is hard and problematic in comparison.

Most recent baseline methods can be approximated by ablations of our model.

We experimented with a simple sequence decoder with attention and copying over the input, but found it to be substantially weaker than other models in all regards.

Next, we consider T ree, our model restricted to using only Child edges without edge labels.

This can be viewed as an evolution of BID17 , with the difference that instead of a log-bilinear network that does not maintain state during the generation, we use a GRU.

ASN is similar to abstract syntax networks BID20 and arises as an extension of the T ree model by adding edge labels on Child that encode the chosen production and the index of the child (corresponding to the "field name" Rabinovich et al. FORMULA0 ).

Finally, Syn follows the work of Yin & Neubig (2017) , but uses a GRU instead of an LSTM.

For this, we extend T ree by a new NextExp edge that connects nodes to each other in the expansion sequence of the tree, thus corresponding to the action flow (Yin & Neubig, 2017) .In all cases, our re-implementations improve on prior work in our variable selection mechanism, which ensures that generated programs only use variables that are defined and in scope.

Both Rabinovich et al. FORMULA0 and Yin & Neubig (2017) instead use a copying mechanism from the context.

On the other hand, they use RNN modules to generate function names and choose arguments from the context (Yin & Neubig, 2017) and to generate string literals BID20 .

Our ExprGen task limits the set of allowed functions and string literals substantially and thus no RNN decoder generating such things is required in our experiments.

The authors of the PHOG BID5 language model kindly ran experiments on our data for the ExprGen task, to provide baseline results of a non-neural language model.

Note, however, that PHOG does not consider the code context to the right of the expression to generate, and does no additional analyses to determine which variable choices are valid.

Extending the model to take more context into account and do some analyses to restrict choices would certainly improve its results.

Metrics We are interested in the ability of a model to generate valid expressions based on the current code context.

To evaluate this, we consider four metrics.

As our ExprGen task requires a conditional language model of code, we first consider the per-token perplexity of the model; the lower the perplexity, the better the model fits the real data distribution.

We then evaluate how often the generated expression is well-typed (i.e., can be typed in the original code context).

We report these metrics for the most likely expression returned by beam search decoding with beam width 5.

Finally, we compute how often the ground truth expression was generated (reported for the most likely expression, as well as for the top five expressions).

This measure is stricter than semantic equivalence, as an expression j > i will not match the equivalent i < j.

Results We show the results of our evaluation in Tab.

1.

Overall, the graph encoder architecture seems to be best-suited for this task.

All models learn to generate syntactically valid code (which is relatively simple in our domain).

However, the different encoder models perform very differently on semantic measures such as well-typedness and the retrieval of the ground truth expression.

Most of the type errors are due to usage of an "UNK" literal (for example, the G → NAG model only has 4% type error when filtering out such unknown literals).

The results show a clear trend that correlates better semantic results with the amount of information about the partially generated programs employed by the generative models.

Transferring a trained model to unseen projects with a new project-specific vocabulary substantially worsens results, as expected.

Overall, our NAG model, combining and adding additional signal sources, seems to perform best on most measures, and seems to be leastimpacted by the transfer.

As the results in the previous section suggest, the proposed ExprGen task is hard even for the strongest models we evaluated, achieving no more than 50% accuracy on the top prediction.

It is also unsolvable for classical logico-deductive program synthesis systems, as the provided code context does not form a precise specification.

However, we do know that most instances of the task are (easily) solvable for professional software developers, and thus believe that machine learning systems can have considerable success on the task.

FIG1 shows two (abbreviated) samples from our test set, together with the predictions made by the two strongest models we evaluated.

In the first example, we can see that the G → NAG model correctly identifies that the relationship between paramCount and methParamCount is important (as they appear together in the blocked guarded by the expression to generate), and thus generates comparison expressions between the two variables.

The G → ASN model lacks the ability to recognize that paramCount (or any variable) was already used and thus fails to insert both relevant variables.

We found this to be a common failure, often leading to suggestions using only one variable (possibly repeatedly).

In the second example, both G → NAG and G → Syn have learned the common if (var.

StartsWith(...)) { ... var.

Substring(num) ... } pattern, but of course fail to produce the correct string literal in the condition.

We show results for all of our models for these examples, as well as for as additional examples, in the supplementary material B.

We presented a generative code model that leverages known semantics of partially generated programs to direct the generative procedure.

The key idea is to augment partial programs to obtain a graph, and then use graph neural networks to compute a precise representation for the partial program.

This representation then helps to better guide the remainder of the generative procedure.

We have shown that this approach can be used to generate small but semantically interesting expressions from very imprecise context information.

The presented model could be useful in program repair scenarios (where repair proposals need to be scored, based on their context) or in the code review setting (where it could highlight very unlikely expressions).

We also believe that similar models could have applications in related domains, such as semantic parsing, neural program synthesis and text generation.

Below we list some sample snippets from the training set for our ExprGen task.

The highlighted expressions are to be generated.

On the following pages, we list some sample snippets from the test set for our ExprGen task, together with suggestions produced by different models.

The highlighted expressions are the ground truth expression that should be generated.

<|TLDR|>

@highlight

Representing programs as graphs including semantics helps when generating programs