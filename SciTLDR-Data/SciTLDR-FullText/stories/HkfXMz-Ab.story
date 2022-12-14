We study the problem of generating source code in a strongly typed, Java-like programming language, given a label (for example a set of API calls or types) carrying a small amount of information about the code that is desired.

The generated programs are expected to respect a `"realistic" relationship between programs and labels, as exemplified by a corpus of labeled programs available during training.



Two challenges in such *conditional program generation* are that the generated programs must satisfy a rich set of syntactic and semantic constraints, and that source code contains many low-level features that impede learning.

We address these problems by training a neural generator not on code but on *program sketches*, or models of program syntax that abstract out names and operations that do not generalize across programs.

During generation, we infer a posterior distribution over sketches, then concretize samples from this distribution into type-safe programs using combinatorial techniques.

We implement our ideas in a system for generating API-heavy Java code, and show that it can often predict the entire body of a method given just a few API calls or data types that appear in the method.

Neural networks have been successfully applied to many generative modeling tasks in the recent past BID22 BID11 BID33 .

However, the use of these models in generating highly structured text remains relatively understudied.

In this paper, we present a method, combining neural and combinatorial techniques, for the condition generation of an important category of such text: the source code of programs in Java-like programming languages.

The specific problem we consider is one of supervised learning.

During training, we are given a set of programs, each program annotated with a label, which may contain information such as the set of API calls or the types used in the code.

Our goal is to learn a function g such that for a test case of the form (X, Prog) (where Prog is a program and X is a label), g(X) is a compilable, type-safe program that is equivalent to Prog.

This problem has immediate applications in helping humans solve programming tasks BID12 BID26 .

In the usage scenario that we envision, a human programmer uses a label to specify a small amount of information about a program that they have in mind.

Based on this information, our generator seeks to produce a program equivalent to the "target" program, thus performing a particularly powerful form of code completion.

Conditional program generation is a special case of program synthesis BID19 BID32 , the classic problem of generating a program given a constraint on its behavior.

This problem has received significant interest in recent years BID2 BID10 .

In particular, several neural approaches to program synthesis driven by input-output examples have emerged BID3 BID23 BID5 .

Fundamentally, these approaches are tasked with associating a program's syntax with its semantics.

As doing so in general is extremely hard, these methods choose to only generate programs in highly controlled domainspecific languages.

For example, BID3 consider a functional language in which the only data types permitted are integers and integer arrays, control flow is linear, and there is a sum total of 15 library functions.

Given a set of input-output examples, their method predicts a vector of binary attributes indicating the presence or absence of various tokens (library functions) in the target program, and uses this prediction to guide a combinatorial search for programs.

In contrast, in conditional program generation, we are already given a set of tokens (for example library functions or types) that appear in a program or its metadata.

Thus, we sidestep the problem of learning the semantics of the programming language from data.

We ask: does this simpler setting permit the generation of programs from a much richer, Java-like language, with one has thousands of data types and API methods, rich control flow and exception handling, and a strong type system?

While simpler than general program synthesis, this problem is still highly nontrivial.

Perhaps the central issue is that to be acceptable to a compiler, a generated program must satisfy a rich set of structural and semantic constraints such as "do not use undeclared variables as arguments to a procedure call" or "only use API calls and variables in a type-safe way".

Learning such constraints automatically from data is hard.

Moreover, as this is also a supervised learning problem, the generated programs also have to follow the patterns in the data while satisfying these constraints.

We approach this problem with a combination of neural learning and type-guided combinatorial search BID6 .

Our central idea is to learn not over source code, but over tree-structured syntactic models, or sketches, of programs.

A sketch abstracts out low-level names and operations from a program, but retains information about the program's control structure, the orders in which it invokes API methods, and the types of arguments and return values of these methods.

We propose a particular kind of probabilistic encoder-decoder, called a Gaussian Encoder-Decoder or GED, to learn a distribution over sketches conditioned on labels.

During synthesis, we sample sketches from this distribution, then flesh out these samples into type-safe programs using a combinatorial method for program synthesis.

Doing so effectively is possible because our sketches are designed to contain rich information about control flow and types.

We have implemented our approach in a system called BAYOU.

1 We evaluate BAYOU in the generation of API-manipulating Android methods, using a corpus of about 150,000 methods drawn from an online repository.

Our experiments show that BAYOU can often generate complex method bodies, including methods implementing tasks not encountered during training, given a few tokens as input.

Now we define conditional program generation.

Assume a universe P of programs and a universe X of labels.

Also assume a set of training examples of the form {(X 1 , Prog 1 ), (X 2 , Prog 2 ), ...}, where each X i is a label and each Prog i is a program.

These examples are sampled from an unknown distribution Q(X, Prog), where X and Prog range over labels and programs, respectively.

2 We assume an equivalence relation Eqv ???

P ?? P over programs.

If (Prog 1 , Prog 2 ) ??? Eqv , then Prog 1 and Prog 2 are functionally equivalent.

The definition of functional equivalence differs across applications, but in general it asserts that two programs are "just as good as" one another.

The goal of conditional program generation is to use the training set to learn a function g : X ??? P such that the expected value E[I((g(X), Prog) ??? Eqv )] is maximized.

Here, I is the indicator function, returning 1 if its boolean argument is true, and 0 otherwise.

Informally, we are attempting to learn a function g such that if we sample (X, Prog) ??? Q(X, P rog), g should be able to reconstitute a program that is functionally equivalent to Prog, using only the label X.

In this paper, we consider a particular form of conditional program generation.

We take the domain P to be the set of possible programs in a programming language called AML that captures the essence of API-heavy Java programs (see Appendix A for more details).

AML includes complex control flow such as loops, if-then statements, and exceptions; access to Java API data types; and calls to Java API methods.

AML is a strongly typed language, and by definition, P only includes programs DISPLAYFORM0 Figure 1: Programs generated by BAYOU with the API method name readLine as a label.

Names of variables of type T whose values are obtained from the environment are of the form $T.that are type-safe.

3 To define labels, we assume three finite sets: a set Calls of possible API calls in AML, a set Types of possible object types, and a set Keys of keywords, defined as words, such as "read" and "file", that often appear in textual descriptions of what programs do.

The space of possible labels is X = 2 Calls ?? 2 T ypes ?? 2 Keys (here 2 S is the power set of S).Defining Eqv in practice is tricky.

For example, a reasonable definition of Eqv is that (Prog 1 , Prog 2 ) ??? Eqv iff Prog 1 and Prog 2 produce the same outputs on all inputs.

But given the richness of AML, the problem of determining whether two AML programs always produce the same output is undecidable.

As such, in practice we can only measure success indirectly, by checking whether the programs use the same control structures, and whether they can produce the same API call sequences.

We will discuss this issue more in Section 6.

Consider the label X = (X Calls , X Types , X Keys ) where X Calls = {readLine} and X Types and X Keys are empty.

Figure 1 (a) shows a program that our best learner stochastically returns given this input.

As we see, this program indeed reads lines from a file, whose name is given by a special variable $String that the code takes as input.

It also handles exceptions and closes the reader, even though these actions were not directly specified.

Although the program in Figure 1 -(a) matches the label well, failures do occur.

Sometimes, the system generates a program as in Figure 1 -(b), which uses an InputStreamReader rather than a FileReader.

It is possible to rule out this program by adding to the label.

Suppose we amend X T ypes so that X T ypes = {FileReader}. BAYOU now tends to only generate programs that use FileReader.

The variations then arise from different ways of handling exceptions and constructing FileReader objects (some programs use a String argument, while others use a File object).

Our approach is to learn g via maximum conditional likelihood estimation (CLE).

That is, given a distribution family P (P rog|X, ??) for a parameter set ??, we choose ?? * = arg max ?? i log P (Prog i | X i , ??).

Then, g(X) = arg max Prog P (Prog|X, ?? * ).The key innovation of our approach is that here, learning happens at a higher level of abstraction than (X i , Prog i ) pairs.

In practice, Java-like programs contain many low-level details (for example, variable names and intermediate results) that can obscure patterns in code.

Further, they contain complicated semantic rules (for example, for type safety) that are difficult to learn from data.

In contrast, these are relatively easy for a combinatorial, syntax-guided program synthesizer BID2 to deal with.

However, synthesizers have a notoriously difficult time figuring out the correct "shape" of a program (such as the placement of loops and conditionals), which we hypothesize should be relatively easy for a statistical learner.

Specifically, our approach learns over sketches: tree-structured data that capture key facets of program syntax.

A sketch Y does not contain low-level variable names and operations, but carries information about broadly shared facets of programs such as the types and API calls.

During generation, a program synthesizer is used to generate programs from sketches produced by the learner.

Let the universe of all sketches be denoted by Y. The sketch for a given program is computed by applying an abstraction function ?? : P ??? Y. We call a sketch Y satisfiable, and write sat(Y), if ?? ???1 (Y) = ???. The process of generating (type-safe) programs given a satisfiable sketch Y is probabilistic, and captured by a concretization distribution P (Prog | Y, sat(Y)).

We require that for all programs Prog and sketches Y such that sat(Y), we have DISPLAYFORM0 Importantly, the concretization distribution is fixed and chosen heuristically.

The alternative of learning this distribution from source code poses difficulties: a single sketch can correspond to many programs that only differ in superficial details, and deciding which differences between programs are superficial and which are not requires knowledge about program semantics.

In contrast, our heuristic approach utilizes known semantic properties of programming languages like oursfor example, that local variable names do not matter, and that some algebraic expressions are semantically equivalent.

This knowledge allows us to limit the set of programs that we generate.

Let us define a random variable Y = ??(Prog).

We assume that the variables X, Y and Prog are related as in the Bayes net in Figure 2 .

Specifically, given Y , Prog is conditionally independent of X. Further, let us assume a distribution family P (Y |X, ??) parameterized on ??.

DISPLAYFORM1 DISPLAYFORM2 Our problem now simplifies to learning over sketches, i.e., finding FIG1 shows the full grammar for sketches in our implementation.

Here, ?? 0 , ?? 1 , . . . range over a finite set of API data types that AML programs can use.

A data type, akin to a Java class, is identified with a finite set of API method names (including constructors), and a ranges over these names.

Note that sketches do not contain constants or variable names.

DISPLAYFORM3

A full definition of the abstraction function for AML appears in Appendix B. As an example, API calls in AML have the syntax "call e.a(e 1 , . . .

, e k )", where a is an API method, the expression e evaluates to the object on which the method is called, and the expressions e 1 , . . .

, e k evaluate to the arguments of the method call.

We abstract this call into an abstract method call "call ??.a(?? 1 , . . .

, ?? k )", where ?? is the type of e and ?? i is the type of e i .

The keywords skip, while, if-then-else, and trycatch preserve information about control flow and exception handling.

Boolean conditions Cseq are replaced by abstract expressions: lists whose elements abstract the API calls in Cseq.

Now we describe our learning approach.

Equation 1 leaves us with the problem of computing arg max ?? i log P (Y i |X i , ??), when each X i is a label and Y i is a sketch.

Our answer is to utilize an encoder-decoder and introduce a real vector-valued latent variable Z to stochastically link labels and sketches: DISPLAYFORM0 is realized as a probabilistic decoder mapping a vector-valued variable to a distribution over trees.

We describe this decoder in Appendix C. As for P (Z|X, ??), this distribution can, in principle, be picked in any way we like.

In practice, because both P (Y |Z, ??) and P (Z|X, ??) have neural components with numerous parameters, we wish this distribution to regularize the learner.

To provide this regularization, we assume a Normal ( 0, I) prior on Z.Recall that our labels are of the form X = (X Calls , X T ypes , X Keys ), where X Calls , X Types , and X Keys are sets.

Assuming that the j-th elements X Calls,j , X Types,j , and X Keys,j of these sets are generated independently, and assuming a function f for encoding these elements, let: DISPLAYFORM1 That is, the encoded value of each X Types,j , X Calls,j or X Keys,j is sampled from a high-dimensional Normal distribution centered at Z. If f is 1-1 and onto with the set R m then from Normal-Normal conjugacy, we have: DISPLAYFORM2 1+n I , where DISPLAYFORM3 Keys .

Here, n Types is the number of types supplied, and n Calls and n Keys are defined similarly.

Note that this particular P (Z|X, ??) only follows directly from the Normal ( 0, I) prior on Z and Normal likelihood P (X|Z, ??) if the encoding function f is 1-1 and onto.

However, even if f is not 1-1 and onto (as will be the case if f is implemented with a standard feed-forward neural network) we can still use this probabilistic encoder, and in practice we still tend to see the benefits of the regularizing prior on Z, with P (Z) distributed approximately according to a unit Normal.

We call this type of encoder-decoder, with a single, Normally-distributed latent variable Z linking the input and output, a Gaussian encoder-decoder, or GED for short.

Now that we have chosen P (X|Z, ??) and P (Y |Z, ??), we must choose ?? to perform CLE.

Note that: DISPLAYFORM4 where the ??? holds due to Jensen's inequality.

Hence, L(??) serves as a lower bound on the loglikelihood, and so we can compute ?? * = arg max ?? L(??) as a proxy for the CLE.

We maximize this lower bound using stochastic gradient ascent; as P (Z|X i , ??) is Normal, we can use the reparameterization trick common in variational auto-encoders BID14 while doing so.

The parameter set ?? contains all of the parameters of the encoding function f as well as ?? Types , ?? Calls , and ?? Keys , and the parameters used in the decoding distribution funciton P (Y |Z, ??).

The final step in our algorithm is to "concretize" sketches into programs, following the distribution P (Prog|Y).

Our method of doing so is a type-directed, stochastic search procedure that builds on combinatorial methods for program synthesis BID28 BID6 .Given a sketch Y, our procedure performs a random walk in a space of partially concretized sketches (PCSs).

A PCS is a term obtained by replacing some of the abstract method calls and expressions in a sketch by AML method calls and AML expressions.

For example, the term "x 1 .a(x 2 ); ?? 1 .b(?? 2 )", which sequential composes an abstract method call to b and a "concrete" method call to a, is a PCS.

The state of the procedure at the i-th point of the walk is a PCS H i .

The initial state is Y.Each state H has a set of neighbors Next(H).

This set consists of all PCS-s H that are obtained by concretizing a single abstract method call or expression in H, using variable names in a way that is consistent with the types of all API methods and declared variables in H.The (i + 1)-th state in a walk is a sample from a predefined, heuristically chosen distribution P (H i+1 | H i , ).

The only requirement on this distribution is that it assigns nonzero probability to a state iff it belongs to Next(H i ).

In practice, our implementation of this distribution prioritizes programs that are simpler.

The random walk ends when it reaches a state H * that has no neighbors.

If H * is fully concrete (that is, an AML program), then the walk is successful and H * is returned as a sample.

If not, the current walk is rejected, and a fresh walk is started from the initial state.

Recall that the concretization distribution P (Prog|Y) is only defined for sketches Y that are satisfiable.

Our concretization procedure does not assume that its input Y is satisfiable.

However, if Y is not satisfiable, all random walks that it performs end with rejection, causing it to never terminate.

While the worst-case complexity of this procedure is exponential in the generated programs, it performs well in practice because of our chosen language of sketches.

For instance, our search does not need to discover the high-level structure of programs.

Also, sketches specify the types of method arguments and return values, and this significantly limits the search space.

Now we present an empirical evaluation of the effectiveness of our method.

The experiments we describe utilize data from an online repository of about 1500 Android apps (and, 2017).

We decompiled the APKs using JADX BID29 to generate their source code.

Analyzing about 100 million lines of code that were generated, we extracted 150,000 methods that used Android APIs or the Java library.

We then pre-processed all method bodies to translate the code from Java to AML, preserving names of relevant API calls and data types as well as the high-level control flow.

Hereafter, when we say "program" we refer to an AML program.

Figure 4: Statistics on labelsFrom each program, we extracted the sets X Calls , X Types , and X Keys as well as a sketch Y. Lacking separate natural language dscriptions for programs, we defined keywords to be words obtained by splitting the names of the API types and calls that the program uses, based on camel case.

For instance, the keywords obtained from the API call readLine are "read" and "line".

As API method and types in Java tend to be carefully named, these words often contain rich information about what programs do.

Figure 4 gives some statistics on the sizes of the labels in the data.

From the extracted data, we randomly selected 10,000 programs to be in the testing and validation data each.

We implemented our approach in our tool called BAYOU, using TensorFlow BID1 to implement the GED neural model, and the Eclipse IDE for the abstraction from Java to the language of sketches and the combinatorial concretization.

In all our experiments we performed cross-validation through grid search and picked the best performing model.

Our hyper-parameters for training the model are as follows.

We used 64, 32 and 64 units in the encoder for API calls, types and keywords, respectively, and 128 units in the decoder.

The latent space was 32-dimensional.

We used a mini-batch size of 50, a learning rate of 0.0006 for the Adam gradient-descent optimizer BID13 , and ran the training for 50 epochs.

The training was performed on an AWS "p2.xlarge" machine with an NVIDIA K80 GPU with 12GB GPU memory.

As each sketch was broken down into a set of production paths, the total number of data points fed to the model was around 700,000 per epoch.

Training took 10 hours to complete.

To visualize clustering in the 32-dimensional latent space, we provided labels X from the testing data and sampled Z from P (Z|X), and then used it to sample a sketch from P (Y |Z).

We then used t-SNE BID17 to reduce the dimensionality of Z to 2-dimensions, and labeled each point with the API used in the sketch Y. Figure 5 shows this 2-dimensional space, where each label has been coded with a different color.

It is immediately apparent from the plot that the model has learned to cluster the latent space neatly according to different APIs.

Some APIs such as java.io have several modes, and we noticed separately that each mode corresponds to different usage scenarios of the API, such as reading versus writing in this case.

To evaluate prediction accuracy, we provided labels from the testing data to our model, sampled sketches from the distribution P (Y |X) and concretized each sketch into an AML program using our combinatorial search.

We then measured the number of test programs for which a program that is equivalent to the expected one appeared in the top-10 results from the model.

As there is no universal metric to measure program equivalence (in fact, it is an undecidable problem in general), we used several metrics to approximate the notion of equivalence.

We defined the following metrics on the top-10 programs predicted by the model:M1.

This binary metric measures whether the expected program appeared in a syntactically equivalent form in the results.

Of course, an impediment to measuring this is that the names of variables used in the expected and predicted programs may not match.

It is neither reasonable nor useful for any model of code to learn the exact variable names in the training data.

Therefore, in performing this equivalence check, we abstract away the variable names and compare the rest of the program's Abstract Syntax Tree (AST) instead.

M2.

This metric measures the minimum Jaccard distance between the sets of sequences of API calls made by the expected and predicted programs.

It is a measure of how close to the original program were we able to get in terms of sequences of API calls.

M3.

Similar to metric M2, this metric measures the minimum Jaccard distance between the sets of API calls in the expected and predicted programs.

M4.

This metric computes the minimum absolute difference between the number of statements in the expected and sampled programs, as a ratio of that in the former.

M5.

Similar to metric M4, this metric computes the minumum absolute difference between the number of control structures in the expected and sampled programs, as a ratio of that in the former.

Examples of control structures are branches, loops, and try-catch statements.

To evaluate our model's ability to predict programs given a small amount of information about its code, we varied the fraction of the set of API calls, types, and keywords provided as input from the testing data.

We experimented with 75%, 50% and 25% observability in the testing data; the median number of items in a label in these cases were 9, 6, and 2, respectively.

Figure 6 : Accuracy of different models on testing data.

GED-AML and GSNN-AML are baseline models trained over AML ASTs, GED-Sk and GSNN-Sk are models trained over sketches.

In order to compare our model with state-of-the-art conditional generative models, we implemented the Gaussian Stochastic Neural Network (GSNN) presented by BID30 , using the same tree-structured decoder as the GED.

There are two main differences: (i) the GSNN's decoder is also conditioned directly on the input label X in addition to Z, which we accomplish by concatenating its initial state with the encoding of X, (ii) the GSNN loss function has an additional KL-divergence term weighted by a hyper-parameter ??.

We subjected the GSNN to the same training and crossvalidation process as our model.

In the end, we selected a model that happened to have very similar hyper-parameters as ours, with ?? = 0.001.

In order to evaluate the effect of sketch learning for program generation, we implemented and compared with a model that learns directly over programs.

Specifically, the neural network structure is exactly the same as ours, except that instead of being trained on production paths in the sketches, the model is trained on production paths in the ASTs of the AML programs.

We selected a model that had more units in the decoder (256) compared to our model (128), as the AML grammar is more complex than the grammar of sketches.

We also implemented a similar GSNN model to train over AML ASTs directly.

Figure 6 shows the collated results of this evaluation, where each entry computes the average of the corresponding metric over the 10000 test programs.

It takes our model about 8 seconds, on average, to generate and rank 10 programs.

When testing models that were trained on AML ASTs, namely the GED-AML and GSNN-AML models, we observed that out of a total of 87,486 AML ASTs sampled from the two models, 2525 (or 3%) ASTs were not even well-formed, i.e., they would not pass a parser, and hence had to be discarded from the metrics.

This number is 0 for the GED-Sk and GSNN-Sk models, meaning that all AML ASTs that were obtained by concretizing sketches were well-formed.

In general, one can observe that the GED-Sk model performs best overall, with GSNN-Sk a reasonable alternative.

We hypothesize that the reason GED-Sk performs slightly better is the regularizing prior on Z; since the GSNN has a direct link from X to Y , it can choose to ignore this regularization.

We would classify both these models as suitable for conditional program generation.

However, the other two models GED-AML and GSNN-AML perform quite worse, showing that sketch learning is key in addressing the problem of conditional program generation.

To evaluate how well our model generalizes to unseen data, we gather a subset of the testing data whose data points, consisting of label-sketch pairs (X, Y), never occurred in the training data.

We then evaluate the same metrics in Figure 6 (a)-(e), but due to space reasons we focus on the 50% observability column.

Figure 6 (f) shows the results of this evaluation on the subset of 5126 (out of 10000) unseen test data points.

The metrics exhibit a similar trend, showing that the models based on sketch learning are able to generalize much better than the baseline models, and that the GED-Sk model performs the best.

Unconditional, corpus-driven generation of programs has been studied before BID18 Allamanis & Sutton, 2014; BID4 , as has the generation of code snippets conditioned on a context into which the snippet is merged BID21 BID26 BID20 .

These prior efforts often use models like n-grams BID21 and recurrent neural networks BID26 that are primarily suited to the generation of straight-line programs; almost universally, they cannot guarantee semantic properties of generated programs.

Among prominent exceptions, BID18 use log-bilinear tree-traversal models, a class of probabilistic pushdown automata, for program generation.

BID4 study a generalization of probabilistic grammars known as probabilistic higher-order grammars.

Like our work, these papers address the generation of programs that satisfy rich constraints such as the type-safe use of names.

In principle, one could replace our decoder and the combinatorial concretizer, which together form an unconditional program generator, with one of these models.

However, given our experiments, doing so is unlikely to lead to good performance in the end-to-end problem of conditional program generation.

There is a line of existing work considering the generation of programs from text BID34 BID16 BID25 .

These papers use decoders similar to the one used in BAYOU, and since they are solving the text-to-code problem, they utilize attention mechanisms not found in BAYOU.

Those attention mechanisms could be particularly useful were BAYOU extended to handle natural language evidence.

The fundamental difference between these works and BAYOU, however, is the level of abstraction at which learning takes place.

These papers attempt to translate text directly into code, whereas BAYOU uses neural methods to produce higher-level sketches that are translated into program code using symbolic methods.

This two-step code generation process is central to BAYOU.

It ensures key semantic properties of the generated code (such as type safety) and by abstracting away from the learner many lower-level details, it may make learning easier.

We have given experimental evidence that this approach can give better results than translating directly into code.

BID15 propose a variational autoencoder for context-free grammars.

As an autoencoder, this model is generative, but it is not a conditional model such as ours.

In their application of synthesizing molecular structures, given a particular molecular structure, their model can be used to search the latent space for similar valid structures.

In our setting, however, we are not given a sketch but only a label for the sketch, and our task is learn a conditional model that can predict a whole sketch given a label.

Conditional program generation is closely related to program synthesis BID10 , the problem of producing programs that satisfy a given semantic specification.

The programming language community has studied this problem thoroughly using the tools of combinatorial search and symbolic reasoning BID2 BID31 BID8 BID6 .

A common tactic in this literature is to put syntactic limitations on the space of feasible programs BID2 .

This is done either by adding a human-provided sketch to a problem instance BID31 , or by restricting synthesis to a narrow DSL BID8 BID24 .A recent body of work has developed neural approaches to program synthesis.

Terpret (Gaunt et al., 2016) and Neural Forth BID27 use neural learning over a set of user-provided examples to complete a user-provided sketch.

In neuro-symbolic synthesis BID23 and RobustFill BID5 , a neural architecture is used to encode a set of input-output examples and decode the resulting representation into a Flashfill program.

DeepCoder BID3 uses neural techniques to speed up the synthesis of Flashfill programs.

These efforts differ from ours in goals as well as methods.

Our problem is simpler, as it is conditioned on syntactic, rather than semantic, facets of programs.

This allows us to generate programs in a complex programming language over a large number of data types and API methods, without needing a human-provided sketch.

The key methodological difference between our work and symbolic program synthesis lies in our use of data, which allows us to generalize from a very small amount of specification.

Unlike our approach, most neural approaches to program synthesis do not combine learning and combinatorial techniques.

The prominent exception is Deepcoder BID3 , whose relationship with our work was discussed in Section 1.

We have given a method for generating type-safe programs in a Java-like language, given a label containing a small amount of information about a program's code or metadata.

Our main idea is to learn a model that can predict sketches of programs relevant to a label.

The predicted sketches are concretized into code using combinatorial techniques.

We have implemented our ideas in BAYOU, a system for the generation of API-heavy code.

Our experiments indicate that the system can often generate complex method bodies from just a few tokens, and that learning at the level of sketches is key to performing such generation effectively.

An important distinction between our work and classical program synthesis is that our generator is conditioned on uncertain, syntactic information about the target program, as opposed to hard constraints on the program's semantics.

Of course, the programs that we generate are type-safe, and therefore guaranteed to satisfy certain semantic constraints.

However, these constraints are invariant across generation tasks; in contrast, traditional program synthesis permits instance-specific semantic constraints.

Future work will seek to condition program generation on syntactic labels as well as semantic constraints.

As mentioned earlier, learning correlations between the syntax and semantics of programs written in complex languages is difficult.

However, the approach of first generating and then concretizing a sketch could reduce this difficulty: sketches could be generated using a limited amount of semantic information, and the concretizer could use logic-based techniques BID2 BID10 to ensure that the programs synthesized from these sketches match the semantic constraints exactly.

A key challenge here would be to calibrate the amount of semantic information on which sketch generation is conditioned.

A THE AML LANGUAGE AML is a core language that is designed to capture the essence of API usage in Java-like languages.

Now we present this language.

DISPLAYFORM0 AML uses a finite set of API data types.

A type is identified with a finite set of API method names (including constructors); the type for which this set is empty is said to be void.

Each method name a is associated with a type signature (?? 1 , . . .

, ?? k ) ??? ?? 0 , where ?? 1 , . . . , ?? k are the method's input types and ?? 0 is its return type.

A method for which ?? 0 is void is interpreted to not return a value.

Finally, we assume predefined universes of constants and variable names.

The grammar for AML is as in FIG4 .

Here, x, x 1 , . . .

are variable names, c is a constant, and a is a method name.

The syntax for programs Prog includes method calls, loops, branches, statement sequencing, and exception handling.

We use variables to feed the output of one method into another, and the keyword let to store the return value of a call in a fresh variable.

Exp stands for (objectvalued) expressions, which include constants, variables, method calls, and let-expressions such as "let x = Call : Exp", which stores the return value of a call in a fresh variable x, then uses this binding to evaluate the expression Exp. (Arithmetic and relational operators are assumed to be encompassed by API methods.)The operational semantics and type system for AML are standard, and consequently, we do not describe these in detail.

We define the abstraction function ?? for the AML language in Figure 9 .

In this section we present the details of the neural networks used by BAYOU.

The task of the neural encoder is to implement the encoding function f for labels, which accepts an element from a label, say X Calls,i as input and maps it into a vector in d-dimensional space, where d is the dimensionality of the latent space of Z. To achieve this, we first convert each element X Calls,i into its one-hot vector representation, denoted X Calls,i .

Then, let h be the number of neural hidden DISPLAYFORM0 DISPLAYFORM1 be real-valued weight and bias matrices of the neural network.

The encoding function f (X Calls,i ) can be defined as follows: DISPLAYFORM2 where tanh is a non-linearity defined as tanh(x) = 1???e ???2x 1+e ???2x .

This would map any given API call into a d-dimensional real-valued vector.

The values of entries in the matrices W h , b h , W d and b d will be learned during training.

The encoder for types can be defined analogously, with its own set of matrices and hidden state.

The task of the neural decoder is to implement the sampler for Y ??? P (Y |Z).

This is implemented recursively via repeated samples of production rules Y i in the grammar of sketches, drawn as DISPLAYFORM0 The generation of each Y i requires the generation of a new "path" from a series of previous "paths", where each path corresponds to a series of production rules fired in the grammar.

As a sketch is tree-structured, we use a top-down tree-structured recurrent neural network similar to BID35 , which we elaborate in this section.

First, similar to the notion of a "dependency path" in BID35 , we define a production path as a sequence of pairs (v 1 , e 1 ), (v 2 , e 2 ), . . .

, (v k , e k ) where v i is a node in the sketch (i.e., a term in the grammar) and e i is the type of edge that connects v i with v i+1 .

Our representation has two types of edges: sibling and child.

A sibling edge connects two nodes at the same level of the tree and under the same parent node (i.e., two terms in the RHS of the same rule).

A child edge connects a node with another that is DISPLAYFORM1 1 + e ???2x and DISPLAYFORM2 for j ??? 1 . . .

K Figure 11 : Computing the hidden state and output of the decoder one level deeper in the tree (i.e., the LHS with a term in the RHS of a rule).

We consider a sequence of API calls connected by sequential composition as siblings.

The root of the entire tree is a special node named root, and so the first pair in all production paths is (root, child).

The last edge in a production path is irrelevant (??) as it does not connect the node to any subsequent nodes.

As an example, consider the sketch in FIG3 (a), whose representation as a tree for the decoder is shown in Figure 10 .

For brevity, we use s and c for sibling and child edges respectively, abbreviate some classnames with uppercase letters in their name, and omit the first pair (root, c) that occurs in all paths.

There are four production paths in the tree of this sketch:Let h be the number of hidden units in the decoder, and |G| be the size of the decoder's output vocabulary, i.e., the total number of terminals and non-terminals in the grammar of sketches.

and b e v ??? R h be the input weight and bias matrices, and W e y ??? R h??|G| and b e y ??? R |G| be the output weight and bias matrices, where e is the type of edge: either c (child) or s (sibling).

We also use "lifting" matrices W l ??? R d??h and b l ??? R h , to lift the d-dimensional vector Z onto the (typically) higher-dimensional hidden state space h of the decoder.

Let h i and y i be the hidden state and output of the network at time point i.

We compute these quantities as given in Figure 11 , where tanh is a non-linear activation function that converts any given value to a value between -1 and 1, and softmax converts a given K-sized vector of arbitrary values to another K-sized vector of values in the range [0, 1] that sum to 1-essentially a probability distribution.

The type of edge at time i decides which RNN to choose to update the (shared) hidden state h i and the output y i .

Training consists of learning values for the entries in all the W and b matrices.

During training, v i , e i and the target output are known from the data point, and so we optimize a standard cross-entropy loss function (over all i) between the output y i and the target output.

During inference, P (v i+1 |Y i , Z) is simply the probability distribution y i , the result of the softmax.

A sketch is obtained by starting with the root node pair (v 1 , e 1 ) = (root, child), recursively applying Equation 2 to get the output distribution y i , sampling a value for v i+1 from y i , and growing the tree by adding the sampled node to it.

The edge e i+1 is provided as c or s depending on the v i+1 that was Figure 12 : 2-dimensional projection of latent space of the GSNN-Sk model sampled.

If only one type of edge is feasible (for instance, if the node is a terminal in the grammar, only a sibling edge is possible with the next node), then only that edge is provided.

If both edges are feasible, then both possibilities are recursively explored, growing the tree in both directions.

Remarks.

In our implementation, we generate trees in a depth-first fashion, by exploring a child edge before a sibling edge if both are possible.

If a node has two children, a neural encoding of the nodes that were generated on the left is carried onto the right sub-tree so that the generation of this tree can leverage additional information about its previously generated sibling.

We refer the reader to Section 2.4 of BID35 for more details.

In this section, we provide results of additional experimental evaluation.

Similar to the visualization of the 2-dimensional latent space in Figure 5 , we also plotted the latent space of the GSNN-Sk model trained on sketches.

Figure 12 shows this plot.

We observed that the latent space is clustered, relatively, more densely than that of our model (keep in mind that the plot colors are different when comparing them).

To give a sense of the quality of the end-to-end generation, we present and discuss a few usage scenarios for our system, BAYOU.

In each scenario, we started with a set of API calls, types or keywords as labels that indicate what we (as the user) would like the generated code to perform.

We then pick a single program in the top-5 results returned by BAYOU and discuss it.

FIG1 shows three such example usage scenarios.

In the first scenario, we would like the system to generate a program to write something to a file by calling write using the type FileWriter.

With this label, we invoked BAYOU and it returned with a program that actually accomplishes the task.

Note that even though we only specified FileWriter, the program uses it to feed a BufferedWriter to write to a file.

This is an interesting pattern learned from data, that file reads and writes in Java often take place in a buffered manner.

Also note that the program correctly flushes the buffer before closing it, even though none of this was explicitly specified in the input.

In the second scenario, we would like the generated program to set the title and message of an Android dialog.

This time we provide no API calls or types but only keywords.

With this, BAYOU generated a program that first builds an Android dialog box using the helper class AlertDialog.

Builder, and does set its title and message.

In addition, the program also adds a

@highlight

We give a method for generating type-safe programs in a Java-like language, given a small amount of syntactic information about the desired code.