Learning tasks on source code (i.e., formal languages) have been considered recently, but most work has tried to transfer natural language methods and does not capitalize on the unique opportunities offered by code's known syntax.

For example, long-range dependencies induced by using the same variable or function in distant locations are often not considered.

We propose to use graphs to represent both the syntactic and semantic structure of code and use graph-based deep learning methods to learn to reason over program structures.



In this work, we present how to construct graphs from source code and how to scale Gated Graph Neural Networks training to such large graphs.

We evaluate our method on two tasks: VarNaming, in which a network attempts to predict the name of a variable given its usage, and VarMisuse, in which the network learns to reason about selecting the correct variable that should be used at a given program location.

Our comparison to methods that use less structured program representations shows the advantages of modeling known structure, and suggests that our models learn to infer meaningful names and to solve the VarMisuse task in many cases.

Additionally, our testing showed that VarMisuse identifies a number of bugs in mature open-source projects.

The advent of large repositories of source code as well as scalable machine learning methods naturally leads to the idea of "big code", i.e., largely unsupervised methods that support software engineers by generalizing from existing source code BID4 .

Currently, existing deep learning models of source code capture its shallow, textual structure, e.g. as a sequence of tokens BID15 BID22 BID3 , as parse trees BID18 , or as a flat dependency networks of variables BID23 .

Such models miss out on the opportunity to capitalize on the rich and well-defined semantics of source code.

In this work, we take a step to alleviate this by including two additional signal sources in source code: data flow and type hierarchies.

We do this by encoding programs as graphs, in which edges represent syntactic relationships (e.g. "token before/after") as well as semantic relationships ("variable last used/written here", "formal parameter for argument is called stream", etc.).

Our key insight is that exposing these semantics explicitly as structured input to a machine learning model lessens the requirements on amounts of training data, model capacity and training regime and allows us to solve tasks that are beyond the current state of the art.

We explore two tasks to illustrate the advantages of exposing more semantic structure of programs.

First, we consider the VARNAMING task BID1 BID23 , in which given some source code, the "correct" variable name is inferred as a sequence of subtokens.

This requires some understanding of how a variable is used, i.e., requires reasoning about lines of code far var clazz=classTypes ["Root"] .Single() as JsonCodeGenerator.

ClassType; Assert.

NotNull(clazz); var first=classTypes ["RecClass"] .Single() as JsonCodeGenerator.

ClassType; Assert.

NotNull( clazz ); Assert.

Equal("string", first.

Properties ["Name"] .Name); Assert.

False(clazz.

Properties ["Name"] .IsArray); Figure 1 : A snippet of a detected bug in RavenDB an open-source C# project.

The code has been slightly simplified.

Our model detects correctly that the variable used in the highlighted (yellow) slot is incorrect.

Instead, first should have been placed at the slot.

We reported this problem which was fixed in PR 4138.

apart in the source file.

Secondly, we introduce the variable misuse prediction task (VARMISUSE), in which the network aims to infer which variable should be used in a program location.

To illustrate the task, Figure 1 shows a slightly simplified snippet of a bug our model detected in a popular open-source project.

Specifically, instead of the variable clazz, variable first should have been used in the yellow highlighted slot.

Existing static analysis methods cannot detect such issues, even though a software engineer would easily identify this as an error from experience.

To achieve high accuracy on these tasks, we need to learn representations of program semantics.

For both tasks, we need to learn the semantic role of a variable (e.g., "is it a counter?

", "is it a filename?

").

Additionally, for VARMISUSE, learning variable usage semantics (e.g., "a filename is needed here") is required.

This "fill the blank element" task is related to methods for learning distributed representations of natural language words, such as Word2Vec BID20 and GLoVe BID21 .

However, we can learn from a much richer structure such as data flow information.

This work is a step towards learning program representations, and we expect them to be valuable in a wide range of other tasks, such as code completion ("this is the variable you are looking for") and more advanced bug finding ("you should lock before using this object").To summarize, our contributions are: (i) We define the VARMISUSE task as a challenge for machine learning modeling of source code, that requires to learn (some) semantics of programs (cf.

section 3).(ii) We present deep learning models for solving the VARNAMING and VARMISUSE tasks by modeling the code's graph structure and learning program representations over those graphs (cf. section 4). (iii) We evaluate our models on a large dataset of 2.9 million lines of real-world source code, showing that our best model achieves 32.9% accuracy on the VARNAMING task and 85.5% accuracy on the VARMISUSE task, beating simpler baselines (cf.

section 5). (iv) We document practical relevance of VARMISUSE by summarizing some bugs that we found in mature open-source software projects (cf. subsection 5.3).

Our implementation of graph neural networks (on a simpler task) can be found at https://github.com/Microsoft/gated-graph-neural-network-samples and the dataset can be found at https://aka.ms/iclr18-prog-graphs-dataset.

Our work builds upon the recent field of using machine learning for source code artifacts BID4 .

For example, BID15 BID7 model the code as a sequence of tokens, while BID18 ; model the syntax tree structure of code.

All works on language models of code find that predicting variable and method identifiers is one of biggest challenges in the task.

Closest to our work is the work of BID2 who learn distributed representations of variables using all their usages to predict their names.

However, they do not use data flow information and we are not aware of any model that does so.

BID23 and BID8 use conditional random fields to model a variety of relationships between variables, AST elements and types to predict variable names and types (resp.

to deobfuscate Android apps), but without considering the flow of data explicitly.

In these works, all variable usages are deterministically known beforehand (as the code is complete and remains unmodified), as in BID1 .Our work is remotely related to work on program synthesis using sketches BID27 and automated code transplantation .

However, these approaches require a set of specifications (e.g. input-output examples, test suites) to complete the gaps, rather than statistics learned from big code.

These approaches can be thought as complementary to ours, since we learn to statistically complete the gaps without any need for specifications, by learning common variable usage patterns from code.

Neural networks on graphs BID13 BID17 BID11 BID16 BID12 ) adapt a variety of deep learning methods to graph-structured input.

They have been used in a series of applications, such as link prediction and classification BID14 and semantic role labeling in NLP BID19 .

Somewhat related to source code is the work of BID28 who learn graph-based representations of mathematical formulas for premise selection in theorem proving.

Detecting variable misuses in code is a task that requires understanding and reasoning about program semantics.

To successfully tackle the task one needs to infer the role and function of the program elements and understand how they relate.

For example, given a program such as Fig. 1 , the task is to automatically detect that the marked use of clazz is a mistake and that first should be used instead.

While this task resembles standard code completion, it differs significantly in its scope and purpose, by considering only variable identifiers and a mostly complete program.

Task Description We view a source code file as a sequence of tokens t 0 . . .

t N = T , in which some tokens t λ0 , t λ1 . . .

are variables.

Furthermore, let V t ⊂ V refer to the set of all type-correct variables in scope at the location of t, i.e., those variables that can be used at t without raising a compiler error.

We call a token tok λ where we want to predict the correct variable usage a slot.

We define a separate task for each slot t λ : Given t 0 . . .

t λ−1 and t λ+1 , . . .

, t N , correctly select t λ from V t λ .

For training and evaluation purposes, a correct solution is one that simply matches the ground truth, but note that in practice, several possible assignments could be considered correct (i.e., when several variables refer to the same value in memory).

In this section, we discuss how to transform program source code into program graphs and learn representations over them.

These program graphs not only encode the program text but also the semantic information that can be obtained using standard compiler tools.

Gated Graph Neural Networks Our work builds on Gated Graph Neural Networks BID17 (GGNN) and we summarize them here.

A graph G = (V, E, X) is composed of a set of nodes V, node features X, and a list of directed edge sets E = (E 1 , . . .

, E K ) where K is the number of edge types.

We annotate each v ∈ V with a real-valued vector x (v) ∈ R D representing the features of the node (e.g., the embedding of a string label of that node).We associate every node v with a state vector h (v) , initialized from the node label x (v) .

The sizes of the state vector and feature vector are typically the same, but we can use larger state vectors through padding of node features.

To propagate information throughout the graph, "messages" of type k are sent from each v to its neighbors, where each message is computed from its current state vector as m DISPLAYFORM0 Here, f k can be an arbitrary function; we choose a linear layer in our case.

By computing messages for all graph edges at the same time, all states can be updated at the same time.

In particular, a new state for a node v is computed by aggregating all incoming messages as DISPLAYFORM1 k | there is an edge of type k from u to v}).

g is an aggregation function, which we implement as elementwise summation.

Given the aggregated messagem (v) and the current state vector h (v) of node v, the state of the next time step DISPLAYFORM2 , where GRU is the recurrent cell function of gated recurrent unit (GRU) BID10 (a) Simplified syntax graph for line 2 of Fig. 1 , where blue rounded boxes are syntax nodes, black rectangular boxes syntax tokens, blue edges Child edges and double black edges NextToken edges.

Program Graphs We represent program source code as graphs and use different edge types to model syntactic and semantic relationships between different tokens.

The backbone of a program graph is the program's abstract syntax tree (AST), consisting of syntax nodes (corresponding to nonterminals in the programming language's grammar) and syntax tokens (corresponding to terminals).

We label syntax nodes with the name of the nonterminal from the program's grammar, whereas syntax tokens are labeled with the string that they represent.

We use Child edges to connect nodes according to the AST.

As this does not induce an order on children of a syntax node, we additionally add NextToken edges connecting each syntax token to its successor.

An example of this is shown in FIG1 .To capture the flow of control and data through a program, we add additional edges connecting different uses and updates of syntax tokens corresponding to variables.

For such a token v, let D R (v) be the set of syntax tokens at which the variable could have been used last.

This set may contain several nodes (for example, when using a variable after a conditional in which it was used in both branches), and even syntax tokens that follow in the program code (in the case of loops).

Similarly, let D W (v) be the set of syntax tokens at which the variable was last written to.

Using these, we add LastRead (resp.

LastWrite) edges connecting v to all elements of D R (v) (resp.

D W (v)).

Additionally, whenever we observe an assignment v = expr , we connect v to all variable tokens occurring in expr using ComputedFrom edges.

An example of such semantic edges is shown in FIG1 .We extend the graph to chain all uses of the same variable using LastLexicalUse edges (independent of data flow, i.e., in if (...) { ... v ...} else { ... v ...}, we link the two occurrences of v).

We also connect return tokens to the method declaration using ReturnsTo edges (this creates a "shortcut" to its name and type).

Inspired by BID25 , we connect arguments in method calls to the formal parameters that they are matched to with FormalArgName edges, i.e., if we observe a call Foo(bar) and a method declaration Foo(InputStream stream), we connect the bar token to the stream token.

Finally, we connect every token corresponding to a variable to enclosing guard expressions that use the variable with GuardedBy and GuardedByNegation edges.

For example, in if (x > y) { ... x ...} else {

... y ...}, we add a GuardedBy edge from x (resp.

a GuardedByNegation edge from y) to the AST node corresponding to x > y.

Finally, for all types of edges we introduce their respective backwards edges (transposing the adjacency matrix), doubling the number of edges and edge types.

Backwards edges help with propagating information faster across the GGNN and make the model more expressive.

We assume a statically typed language and that the source code can be compiled, and thus each variable has a (known) type τ (v).

To use it, we define a learnable embedding function r(τ ) for known types and additionally define an "UNKTYPE" for all unknown/unrepresented types.

We also leverage the rich type hierarchy that is available in many object-oriented languages.

For this, we map a variable's type τ (v) to the set of its supertypes, i.e. τ * (v) = {τ : τ (v) implements type τ } ∪ {τ (v)}. We then compute the type representation r * (v) of a variable v as the element-wise maximum of {r(τ ) : τ ∈ τ * (v)}. We chose the maximum here, as it is a natural pooling operation for representing partial ordering relations (such as type lattices).

Using all types in τ * (v) allows us to generalize to unseen types that implement common supertypes or interfaces.

For example, List<K> has multiple concrete types (e.g. List<int>, List<string>).

Nevertheless, these types implement a common interface (IList) and share common characteristics.

During training, we randomly select a non-empty subset of τ * (v) which ensures training of all known types in the lattice.

This acts both like a dropout mechanism and allows us to learn a good representation for all types in the type lattice.

Initial Node Representation To compute the initial node state, we combine information from the textual representation of the token and its type.

Concretely, we split the name of a node representing a token into subtokens (e.g. classTypes will be split into two subtokens class and types) on camelCase and pascal_case.

We then average the embeddings of all subtokens to retrieve an embedding for the node name.

Finally, we concatenate the learned type representation r * (v), computed as discussed earlier, with the node name representation, and pass it through a linear layer to obtain the initial representations for each node in the graph.

Programs Graphs for VARNAMING Given a program and an existing variable v, we build a program graph as discussed above and then replace the variable name in all corresponding variable tokens by a special <SLOT> token.

To predict a name, we use the initial node labels computed as the concatenation of learnable token embeddings and type embeddings as discussed above, run GGNN propagation for 8 time steps 2 and then compute a variable usage representation by averaging the representations for all <SLOT> tokens.

This representation is then used as the initial state of a one-layer GRU, which predicts the target name as a sequence of subtokens (e.g., the name inputStreamBuffer is treated as the sequence [input, stream, buffer]).

We train this graph2seq architecture using a maximum likelihood objective.

In section 5, we report the accuracy for predicting the exact name and the F1 score for predicting its subtokens.

Program Graphs for VARMISUSE To model VARMISUSE with program graphs we need to modify the graph.

First, to compute a context representation c(t) for a slot t where we want to predict the used variable, we insert a new node v <SLOT> at the position of t, corresponding to a "hole" at this point, and connect it to the remaining graph using all applicable edges that do not depend on the chosen variable at the slot (i.e., everything but LastUse, LastWrite, LastLexicalUse, and GuardedBy edges).

Then, to compute the usage representation u(t, v) of each candidate variable v at the target slot, we insert a "candidate" node v t,v for all v in V t , and connect it to the graph by inserting the LastUse, LastWrite and LastLexicalUse edges that would be used if the variable were to be used at this slot.

Each of these candidate nodes represents the speculative placement of the variable within the scope.

Using the initial node representations, concatenated with an extra bit that is set to one for the candidate nodes v t,v , we run GGNN propagation for 8 time steps.2 The context and usage representation are then the final node states of the nodes, i.e., c(t) = h (v<SLOT>) and u(t, v) = h (vt,v) .

Finally, the correct variable usage at the location is computed as arg max v W [c(t), u(t, v)] where W is a linear layer that uses the concatenation of c(t) and u(t, v).

We train using a max-margin objective.

Using GGNNs for sets of large, diverse graphs requires some engineering effort, as efficient batching is hard in the presence of diverse shapes.

An important observation is that large graphs are normally very sparse, and thus a representation of edges as an adjacency list would usually be advantageous to reduce memory consumption.

In our case, this can be easily implemented using a sparse tensor representation, allowing large batch sizes that exploit the parallelism of modern GPUs efficiently.

A second key insight is to represent a batch of graphs as one large graph with many disconnected components.

This just requires appropriate pre-processing to make node identities unique.

As this makes batch construction somewhat CPU-intensive, we found it useful to prepare minibatches on a separate thread.

Our TensorFlow BID0 implementation scales to 55 graphs per second during training and 219 graphs per second during test-time using a single NVidia GeForce GTX Titan X with graphs having on average 2,228 (median 936) nodes and 8,350 (median 3,274) edges and 8 GGNN unrolling iterations, all 20 edge types (forward and backward edges for 10 original edge types) and the size of the hidden layer set to 64.

The number of types of edges in the GGNN contributes proportionally to the running time.

For example, a GGNN run for our ablation study using only the two most common edge types (NextToken, Child) achieves 105 graphs/second during training and 419 graphs/second at test time with the same hyperparameters.

Our (generic) implementation of GGNNs is available at https://github.com/Microsoft/ gated-graph-neural-network-samples, using a simpler demonstration task.

Dataset We collected a dataset for the VARMISUSE task from open source C # projects on GitHub.

To select projects, we picked the top-starred (non-fork) projects in GitHub.

We then filtered out projects that we could not (easily) compile in full using Roslyn 3 , as we require a compilation to extract precise type information for the code (including those types present in external libraries).

Our final dataset contains 29 projects from a diverse set of domains (compilers, databases, . . . ) with about 2.9 million non-empty lines of code.

A full table is shown in Appendix D.For the task of detecting variable misuses, we collect data from all projects by selecting all variable usage locations, filtering out variable declarations, where at least one other type-compatible replacement variable is in scope.

The task is then to infer the correct variable that originally existed in that location.

Thus, by construction there is at least one type-correct replacement variable, i.e. picking it would not raise an error during type checking.

In our test datasets, at each slot there are on average 3.8 type-correct alternative variables (median 3, σ = 2.6).From our dataset, we selected two projects as our development set.

From the rest of the projects, we selected three projects for UNSEENPROJTEST to allow testing on projects with completely unknown structure and types.

We split the remaining 23 projects into train/validation/test sets in the proportion 60-10-30, splitting along files (i.e., all examples from one source file are in the same set).

We call the test set obtained like this SEENPROJTEST.Baselines For VARMISUSE, we consider two bidirectional RNN-based baselines.

The local model (LOC) is a simple two-layer bidirectional GRU run over the tokens before and after the target location.

For this baseline, c(t) is set to the slot representation computed by the RNN, and the usage context of each variable u(t, v) is the embedding of the name and type of the variable, computed in the same way as the initial node labels in the GGNN.

This baseline allows us to evaluate how important the usage context information is for this task.

The flat dataflow model (AVGBIRNN) is an extension to LOC, where the usage representation u(t, v) is computed using another two-layer bidirectional RNN run over the tokens before/after each usage, and then averaging over the computed representations at the variable token v. The local context, c(t), is identical to LOC.

AVGBIRNN is a significantly stronger baseline that already takes some structural information into account, as the averaging over all variables usages helps with long-range dependencies.

Both models pick the variable that maximizes c(t)T u(t, v).For VARNAMING, we replace LOC by AVGLBL, which uses a log-bilinear model for 4 left and 4 right context tokens of each variable usage, and then averages over these context representations (this corresponds to the model in BID2 ).

We also test AVGBIRNN on VARNAMING, which essentially replaces the log-bilinear context model by a bidirectional RNN.

TAB1 shows the evaluation results of the models for both tasks.

4 As LOC captures very little information, it performs relatively badly.

AVGLBL and AVGBIRNN, which capture information from many variable usage sites, but do not explicitly encode the rich structure of the problem, still lag behind the GGNN by a wide margin.

The performance difference is larger for VARMISUSE, since the structure and the semantics of code are far more important within this setting.

Generalization to new projects Generalizing across a diverse set of source code projects with different domains is an important challenge in machine learning.

We repeat the evaluation using the UNSEENPROJTEST set stemming from projects that have no files in the training set.

The right side of TAB1 shows that our models still achieve good performance, although it is slightly lower compared to SEENPROJTEST.

This is expected since the type lattice is mostly unknown in UNSEENPROJTEST.We believe that the dominant problem in applying a trained model to an unknown project (i.e., domain) is the fact that its type hierarchy is unknown and the used vocabulary (e.g. in variables, method and class names, etc.) can differ substantially.

Ablation Study To study the effect of some of the design choices for our models, we have run some additional experiments and show their results in TAB2 .

First, we varied the edges used in the program graph.

We find that restricting the model to syntactic information has a large impact on performance on both tasks, whereas restricting it to semantic edges seems to mostly impact performance on VARMISUSE.

Similarly, the ComputedFrom, FormalArgName and ReturnsTo edges give a small boost on VARMISUSE, but greatly improve performance on VARNAMING.

As evidenced by the experiments with the node label representation, syntax node and token names seem to matter little for VARMISUSE, but naturally have a great impact on VARNAMING.

Figure 3 illustrates the predictions that GGNN makes on a sample test snippet.

The snippet recursively searches for the global directives file by gradually descending into the root folder.

Reasoning about the correct variable usages is hard, even for humans, but the GGNN correctly predicts the variable 3 http://roslyn.io 4 Sect.

A additionally shows ROC and precision-recall curves for the GGNN model on the VARMISUSE task.

.TrimEnd(Path.

DirectorySeparatorChar); } path 13 = null; return false; } 1: path:59%, baseDirectory:35%, fullPath:6%, GlobalDirectivesFileName:1% 2: baseDirectory:92%, fullPath:5%, GlobalDirectivesFileName:2%, path:0.4% 3: fullPath:88%, baseDirectory:9%, GlobalDirectivesFileName:2%, path:1% 4: directivesDirectory:86%, path:8%, baseDirectory:2%, GlobalDirectivesFileName:1%, fullPath:0.1% 5: directivesDirectory:46%, path:24%, baseDirectory:16%, GlobalDirectivesFileName:10%, fullPath:3% 6: baseDirectory:64%, path:26%, directivesDirectory:5%, fullPath:2%, GlobalDirectivesFileName:2% 7: path:99%, directivesDirectory:1%, GlobalDirectivesFileName:0.5%, baseDirectory:7e-5, fullPath:4e-7 8: fullPath:60%, directivesDirectory:21%, baseDirectory:18%, path:1%, GlobalDirectivesFileName:4e-4 9: GlobalDirectivesFileName:61%, baseDirectory:26%, fullPath:8%, path:4%, directivesDirectory:0.5% 10: path:70%, directivesDirectory:17%, baseDirectory:10%, GlobalDirectivesFileName:1%, fullPath:0.6% 11: directivesDirectory:93%, path:5%, GlobalDirectivesFileName:1%, baseDirectory:0.1%, fullPath:4e-5% 12: directivesDirectory:65%, path:16%, baseDirectory:12%, fullPath:5%, GlobalDirectivesFileName:3% 13: path:97%, baseDirectory:2%, directivesDirectory:0.4%, fullPath:0.3%, GlobalDirectivesFileName:4e-4

Figure 3: VARMISUSE predictions on slots within a snippet of the SEENPROJTEST set for the ServiceStack project.

Additional visualizations are available in Appendix B.

The underlined tokens are the correct tokens.

The model has to select among a number of string variables at each slot, where all of them represent some kind of path.

The GGNN accurately predicts the correct variable usage in 11 out of the 13 slots reasoning about the complex ways the variables interact among them.

usages at all locations except two (slot 1 and 8).

As a software engineer is writing the code, it is imaginable that she may make a mistake misusing one variable in the place of another.

Since all variables are string variables, no type errors will be raised.

As the probabilities in Fig. 3 suggest most potential variable misuses can be flagged by the model yielding valuable warnings to software engineers.

Additional samples with comments can be found in Appendix B.Furthermore, Appendix C shows samples of pairs of code snippets that share similar representations as computed by the cosine similarity of the usage representation u(t, v) of GGNN.

The reader can notice that the network learns to group variable usages that share semantic similarities together.

For example, checking for null before the use of a variable yields similar distributed representations across code segments (Sample 1 in Appendix C).

We have used our VARMISUSE model to identify likely locations of bugs in RavenDB (a document database) and Roslyn (Microsoft's C # compiler framework).

For this, we manually reviewed a sample of the top 500 locations in both projects where our model was most confident about a choosing a variable differing from the ground truth, and found three bugs in each of the projects.

Figs.

1,4,5 show the issues discovered in RavenDB.

The bug in Fig. 1 was possibly caused by copy-pasting, and cannot be easily caught by traditional methods.

A compiler will not warn about if (IsValidBackup(backupFilename) == false) { output("Error:"+ backupLocation +" doesn't look like a valid backup"); throw new InvalidOperationException( backupLocation + " doesn't look like a valid backup");Figure 5: A bug found (yellow) in the RavenDB open-source project.

Although backupFilename is found to be invalid by IsValidBackup, the user is notified that backupLocation is invalid instead.unused variables (since first is used) and virtually nobody would write a test testing another test.

FIG3 shows an issue that, although not critical, can lead to increased memory consumption.

Fig. 5 shows another issue arising from a non-informative error message.

We privately reported three additional bugs to the Roslyn developers, who have fixed the issues in the meantime (cf.

https://github.com/dotnet/roslyn/pull/23437).

One of the reported bugs could cause a crash in Visual Studio when using certain Roslyn features.

Finding these issues in widely released and tested code suggests that our model can be useful during the software development process, complementing classic program analysis tools.

For example, one usage scenario would be to guide the code reviewing process to locations a VARMISUSE model has identified as unusual, or use it as a prior to focus testing or expensive code analysis efforts.

Although source code is well understood and studied within other disciplines such as programming language research, it is a relatively new domain for deep learning.

It presents novel opportunities compared to textual or perceptual data, as its (local) semantics are well-defined and rich additional information can be extracted using well-known, efficient program analyses.

On the other hand, integrating this wealth of structured information poses an interesting challenge.

Our VARMISUSE task exposes these opportunities, going beyond simpler tasks such as code completion.

We consider it as a first proxy for the core challenge of learning the meaning of source code, as it requires to probabilistically refine standard information included in type systems.

A PERFORMANCE CURVES FIG4 shows the ROC and precision-recall curves for the GGNN model.

As the reader may observe, setting a false positive rate to 10% we get a true positive rate 5 of 73% for the SEENPROJTEST and 69% for the unseen test.

This suggests that this model can be practically used at a high precision setting with acceptable performance.

Below we list a set of samples from our SEENPROJTEST projects with comments about the model performance.

Code comments and formatting may have been altered for typesetting reasons.

The ground truth choice is underlined.

The model predicts correctly all usages except from the one in slot #3.

Reasoning about this snippet requires additional semantic information about the intent of the code.

var response = ResultsFilter(typeof(TResponse), #1 , #2 , request);#1 httpMethod: 99%, absoluteUrl: 1%, UserName: 0%, UserAgent: 0% #2 absoluteUrl: 99%, httpMethod: 1%, UserName: 0%, UserAgent: 0%The model knows about selecting the correct string parameters because it matches them to the formal parameter names.

#1 n: 100%, MAXERROR: 0%, SYNC_MAXRETRIES: 0% #2 MAXERROR: 62%, SYNC_MAXRETRIES: 22%, n: 16%

It is hard for the model to reason about conditionals, especially with rare constants as in slot #2.

Here we show pairs of nearest neighbors based on the cosine similarity of the learned representations u(t, v).

Each slot t is marked in dark blue and all usages of v are marked in yellow (i.e. variableName ).

This is a set of hand-picked examples showing good and bad examples.

A brief description follows after each pair.

HasAddress is a local function, seen only in the testset.

For this work, we released a large portion of the data, with the exception of projects with a GPL license.

The data can be found at https://aka.ms/iclr18-prog-graphs-dataset.

Since we are excluding some projects from the data, below we report the results, averaged over three runs, on the published dataset: Accuracy (%) PR AUC SEENPROJTEST 84.0 0.976 UNSEENPROJTEST 74.1 0.934

@highlight

Programs have structure that can be represented as graphs, and graph neural networks can learn to find bugs on such graphs