Neural program embeddings have shown much promise recently for a variety of program analysis tasks, including program synthesis, program repair, code completion, and fault localization.

However, most existing program embeddings are based on syntactic features of programs, such as token sequences or abstract syntax trees.

Unlike images and text, a program has well-deﬁned semantics that can be difﬁcult to capture by only considering its syntax (i.e. syntactically similar programs can exhibit vastly different run-time behavior), which makes syntax-based program embeddings fundamentally limited.

We propose a novel semantic program embedding that is learned from program execution traces.

Our key insight is that program states expressed as sequential tuples of live variable values not only capture program semantics more precisely, but also offer a more natural ﬁt for Recurrent Neural Networks to model.

We evaluate different syntactic and semantic program embeddings on the task of classifying the types of errors that students make in their submissions to an introductory programming class and on the CodeHunt education platform.

Our evaluation results show that the semantic program embeddings signiﬁcantly outperform the syntactic program embeddings based on token sequences and abstract syntax trees.

In addition, we augment a search-based program repair system with predictions made from our semantic embedding and demonstrate signiﬁcantly improved search efﬁciency.

Recent breakthroughs in deep learning techniques for computer vision and natural language processing have led to a growing interest in their applications in programming languages and software engineering.

Several well-explored areas include program classification, similarity detection, program repair, and program synthesis.

One of the key steps in using neural networks for such tasks is to design suitable program representations for the networks to exploit.

Most existing approaches in the neural program analysis literature have used syntax-based program representations.

BID6 proposed a convolutional neural network over abstract syntax trees (ASTs) as the program representation to classify programs based on their functionalities and detecting different sorting routines.

DeepFix BID4 , SynFix BID1 , and sk p BID9 are recent neural program repair techniques for correcting errors in student programs for MOOC assignments, and they all represent programs as sequences of tokens.

Even program synthesis techniques that generate programs as output, such as RobustFill BID3 , also adopt a token-based program representation for the output decoder.

The only exception is BID8 , which introduces a novel perspective of representing programs using input-output pairs.

However, such representations are too coarse-grained to accurately capture program properties -programs with the same input-output behavior may have very different syntactic characteristics.

Consequently, the embeddings learned from input-output pairs are not precise enough for many program analysis tasks.

Although these pioneering efforts have made significant contributions to bridge the gap between deep learning techniques and program analysis tasks, syntax-based program representations are fundamentally limited due to the enormous gap between program syntax (i.e. static expression) and Bubble Insertion [5,5,1,4,3] [5,5,1,4,3] [5,8,1,4,3] [5,8,1,4,3] [5, 1,1,4,3] [5,1,1,4,3] [5, 1,8,4,3] [5,1,8,4,3] [1, 1,8,4,3] [5,1,4,4,3] [ 1,5,8,4,3] [5,1,4,8,3] [1, 5,4,4,3] [5,1,4,3,3] [1, 5,4,8,3] [5,1,4,3,8] [1, 4, 4, 8, 3] [1, 1, 4, 3, 8] [1, 4, 5, 8, 3] [1, 5, 4, 3, 8] [ 1, 4, 5, 3, 3] [1, 4, 4, 3, 8] [ 1, 4, 5, 3, 8] [1, 4, 5, 3, 8] [ 1, 4, 3, 3, 8] [1, 4, 3, 3, 8] [ 1, 4, 3, 5, 8] [1,4,3,5,8] [1,3,3,5,8] [1,3,3,5,8] [1,3,4,5,8] [1, 3, 4, 5, 8] Figure 1: Bubble sort and insertion sort (code highlighted in shadow box are the only syntactic differences between the two algorithms).

Their execution traces for the input vector A = [8, 5, 1, 4, 3] are displayed on the right, where, for brevity, only values for variable A are shown.

semantics (i.e. dynamic execution).

This gap can be illustrated as follows.

First, when a program is executed at runtime, its statements are almost never interpreted in the order in which the corresponding token sequence is presented to the deep learning models (the only exception being straightline programs, i.e., ones without any control-flow statements).

For example, a conditional statement only executes one branch each time, but its token sequence is expressed sequentially as multiple branches.

Similarly, when iterating over a looping structure at runtime, it is unclear in which order any two tokens are executed when considering different loop iterations.

Second, program dependency (i.e. data and control) is not exploited in token sequences and ASTs despite its essential role in defining program semantics.

FIG0 shows an example using a simple max function.

On line 8, the assignment statement means variable max val is data-dependent on item.

In addition, the execution of this statement depends on the evaluation of the if condition on line 7, i.e., max val is also control-dependent on item as well as itself.

Third, from a pure program analysis standpoint, the gap between program syntax and semantics is manifested in that similar program syntax may lead to vastly different program semantics.

For example, consider the two sorting functions shown in Figure 1 .

Both functions sort the array via two nested loops, compare the current element to its successor, and swap them if the order is incorrect.

However, the two functions implement different algorithms, namely Bubble Sort and Insertion Sort.

Therefore minor syntactic discrepancies can lead to significant semantic differences.

This intrinsic weakness will be inherited by any deep learning technique that adopts a syntax-based program representation.

We have evaluated our dynamic program embeddings in the context of automated program repair.

In particular, we use the program embeddings to classify the type of mistakes students made to their programming assignments based on a set of common error patterns (described in the appendix).

The dataset for the experiments consists of the programming submissions made to Module 2 assignment in Microsoft-DEV204.1X and two additional problems from the Microsoft CodeHunt platform.

The results show that our dynamic embeddings significantly outperform syntax-based program embeddings, including those trained on token sequences and abstract syntax trees.

In addition, we show that our dynamic embeddings can be leveraged to significantly improve the efficiency of a searchbased program corrector SARFGEN 1 BID13 ) (the algorithm is presented in the appendix).

More importantly, we believe that our dynamic program embeddings can be useful for many other program analysis tasks, such as program synthesis, fault localization, and similarity detection.

To summarize, the main contributions of this paper are: (1) we show the fundamental limitation of representing programs using syntax-level features; (2) we propose dynamic program embeddings learned from runtime execution traces to overcome key issues with syntactic program representations; (3) we evaluate our dynamic program embeddings for predicting common mistake patterns students make in program assignments, and results show that the dynamic program embeddings outperform state-of-the-art syntactic program embeddings; and (4) we show how the dynamic program embeddings can be utilized to improve an existing production program repair system.

This section briefly reviews dynamic program analysis BID0 , an influential program analysis technique that lays the foundation for constructing our new program embeddings.

Unlike static analysis BID7 , i.e., the analysis of program source code, dynamic analysis focuses on program executions.

An execution is modeled by a set of atomic actions, or events, organized as a trace (or event history).

For simplicity, this paper considers sequential executions only (as opposed to parallel executions) which lead to a single sequence of events, specifically, the executions of statements in the program.

Detailed information about executions is often not readily available, and separate mechanisms are needed to capture the tracing information.

An often adopted approach is to instrument a program's source code (i.e., by adding additional monitoring code) to record the execution of statements of interest.

In particular, those inserted instrumentation statements act as a monitoring window through which the values of variables are inspected.

This instrumentation process can occur in a fully automated manner, e.g., a common approach is to traverse a program's abstract syntax tree and insert "write" statements right after each program statement that causes a side-effect (i.e., changing the values of some variables).Consider the two sorting algorithms depicted in Figure 1 .

If we assume A to be the only variable of interest and subject to monitoring, we can instrument the two algorithms with Console.

WriteLine(A) after each program location in the code whenever A is modified 2 (i.e. the lines marked by comments).

Given the input vector A = [8, 5, 1, 4, 3] , the execution traces of the two sorting routines are shown on the right in Figure 1 .One of the key benefits of dynamic analysis is its ability to easily and precisely identify relevant parts of the program that affect execution behavior.

As shown in the example above, despite the very similar program syntax of bubble sort and insertion sort, dynamic analysis is able to discover their distinct program semantics by exposing their execution traces.

Since understanding program semantics is a central issue in program analysis, dynamic analysis has seen remarkable success over the past several decades and has resulted in many successful program analysis tools such as debuggers, profilers, monitors, or explanation generators.

We now present an overview of our approach.

Given a program and the execution traces extracted for all its variables, we introduce three neural network models to learn dynamic program embeddings.

To demonstrate the utility of these embeddings, we apply them to predict common error patterns (detailed in Section 5) that students make in their submissions to an online introductory programming course.

Variable Trace Embedding As shown in TAB1 , each row denotes a new program point where a variable gets updated.

3 The entire variable trace consists of those variable values at all program points.

As a subsequent step, we split the complete trace into a list of sub-traces (one for each variable).

We use one single RNN to encode each sub-trace independently and then perform max pooling on the final states of the same RNN to obtain the program embedding.

Finally, we add a one layer softmax regression to make the predictions.

The entire workflow is show in FIG1 .State Trace Embedding Because each variable trace is handled individually in the previous approach, variable dependencies/interactions are not precisely captured.

To address this issue, we propose the state trace embedding.

As depicted in TAB1 , each program point l introduces a new program state expressed by the latest variable valuations at l. The entire state trace is a sequence of program states.

To learn the state trace embedding, we first use one RNN to encode each program state (i.e., a tuple of values) and feed the resulting RNN states as a sequence to another RNN.

Note that we do not assume that the order in which variables values are encoded by the RNN for each program state but rather maintain a consistent order throughout all program states for a given trace.

Finally, we feed a softmax regression layer with the final state of the second RNN (shown in FIG2 .

The benefit of state trace embedding is its ability to capture dependencies among variables in each program state as well as the relationship among program states.

Dependency Enforcement for Variable Trace Embedding Although state trace embedding can better capture program dependencies, it also comes with some challenges, the most significant of which is redundancy.

Consider a looping structure in a program.

During an iteration, whenever one variable gets modified, a new program state will be created containing the values of all variables, even of those unmodified by the loop.

This issue becomes more severe for loops with larger numbers of iterations.

To tackle this challenge, we propose the third and final approach, dependency enforcement for variable trace embedding (hereinafter referred as dependency enforcement embedding), that combines the advantages of variable trace embedding (i.e., compact representation of execution traces) and state trace embedding (i.e., precise capturing of program dependencies).

In dependency enforcement embedding, a program is represented by separate variable traces, with each variable being handled by a different RNN.

In order to enforce program dependencies, the hidden states from different RNNs will be interleaved in a way that simulates the needed data and control dependencies.

Unlike variable trace embedding, we perform an average pooling on the final states of all RNNs to obtain the program embedding on which we build the final layer of softmax regression.

FIG3 describes the workflow.

We now formally define the three program embedding models.

Given a program P , and its variable set V (v 0 , v 1 ,.

.., v n ∈ V ), a variable trace is a sequence of values a variable has been assigned during the execution of P .4 Let x t vn denote the value from the variable trace of v n that is fed to the RNN encoder (Gated Recurrent Unit) at time t as the input, and h t vn as the resulting RNN's hidden state.

We compute the variable trace embedding for P in Equation (3) as follows (h T vn denotes the last hidden state of the encoder): DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 We compute the representation of the program trace by performing max pooling over the last hidden state representation of each variable trace embedding.

The hidden states h t v1 , . . .

, h t vn , h P ∈ R k where k denotes the size of hidden layers of the RNN encoder.

Evidence denotes the output of a linear model through the program embedding vector h P , and we obtain the predicted error pattern class Y by using a softmax operation.

The key idea in state trace model is to embed each program state as a numerical vector first and then feed all program state embeddings as a sequence to another RNN encoder to obtain the program embedding.

Suppose x t vn is the value of variable v n at t-th program state, and h t vn is the resulting hidden state of the program state encoder.

Equation (8) computes the t-th program state embedding.

Equations (9-11) encode the sequence of all program state embeddings (i.e., h t vn , h t+1 vn , . . .

, h t+m vn ) with another RNN to compute the program embedding.

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 h t+1 vn = GRU(h t vn , h t+1 vn ) (10) ...

h P = GRU(h t+m−1 vn , x t+m vn ) (11) h t v1 , . . .

, h t vn ∈ R k1 ; h t vn , . . .

, h P ∈ R k2 where k 1 and k 2 denote, respectively, the sizes of hidden layers of the first and second RNN encoders.

The motivation behind this model is to combine the advantages of the previous two approaches, i.e. representing the execution trace compactly while enforcing the dependency relationship among variables as much as possible.

In this model, each variable trace is handled with a different RNN.

A potential issue to be addressed is variable matching/renaming (i.e., α-renaming).

In other words same variables may be named differently in different programs.

Processing each variable id with a single RNN among all programs in the dataset will not only cause memory issues, but more importantly the loss of precision.

Our solution is to (1) execute all programs to collect traces for all variables, (2) perform dynamic time wrapping (Vintsyuk, 1968) on the variable traces across all programs to find the top-n most used variables that account for the vast majority of variable usage, and (3) rename the top-n most used variables consistently across all programs, and rename all other variables to a same special variable.

Given the same set of variables among all programs, the mechanism of dependency enforcement on the top ones is to fuse the hidden states of multiple RNNs based on how a new value of a variable is produced.

For example, in FIG0 at line 8, the new value of max val is data-dependent on item, and control-dependent on both item and itself.

So at the time step when the new value of max val is produced, the latest hidden states of the RNNs encode variable item as well as itself; they together determine the previous state of the RNN upon which the new value of max val is produced.

If a value is produced without any dependencies, this mechanism will not take effect.

In other words, the RNN will act normally to handle data sequences on its own.

In this work we enforce the data-dependency in assignment statement, declaration statement and method calls; and control-dependency in control statements such as if , f or and while statements.

Equations (11 and 12) expose the inner workflow.

h LT vm denotes the latest hidden state of the RNN encoding variable trace of v m up to the point of time t when x t vn is the input of the RNN encoding variable trace of v n .

denotes element-wise matrix product.

DISPLAYFORM0 Given v n depends on v 1 and v m (11) We train our dynamic program embeddings on the programming submissions obtained from Assignment 2 from Microsoft-DEV204.1X: "Introduction to C#" offered on edx and two other problems on Microsoft CodeHunt platform.

DISPLAYFORM1 • Print Chessboard: Print the chessboard pattern using "X" and "O" to represent the squares as shown in FIG4 .• Count Parentheses: Count the depth of nesting parentheses in a given string.• Generate Binary Digits: Generate the string of binary digits for a given integer.

Regarding the three programming problems, the errors students made in their submissions can be roughly classified into low-level technical issues (e.g., list indexing, branching conditions or looping bounds) and high-level conceptual issues (e.g., mishandling corner case, misunderstanding problem requirement or misconceptions on the underlying data structure of test inputs).

In order to have sufficient data for training our models to predict the error patterns, we (1) convert each incorrect program into multiple programs such that each new program will have only one error, and (2) mutate all the correct programs to generate synthetic incorrect programs such that they exhibit similar errors that students made in real program submissions.

These two steps allow us to set up a dataset depicted in TAB4 .

Based on the same set of training data, we evaluate the dynamic embeddings trained with the three network models and compare them with the syntax-based program embeddings (on the same error prediction task) on the same testing data.

The syntax-based models include (1) one trained with a RNN that encodes the run-time syntactic traces of programs BID10 ; (2) another trained with a RNN that encodes token sequences of programs; and (3) the third trained with a RNN on abstract syntax trees of programs BID11 5 Please refer to the Appendix for a detailed summary of the error patterns for each problem.

All models are implemented in TensorFlow.

All encoders in each of the trace model have two stacked GRU layers with 200 hidden units in each layer except that the state encoder in the state trace model has one single layer of 100 hidden units.

We adopt random initialization for weight initialization.

Our vocabulary has 5,568 unique tokens (i.e., the values of all variables at each time step), each of which is embedded into a 100-dimensional vector.

All networks are trained using the Adam optimizer BID5 with the learning and the decay rates set to their default values (learning rate = 0.0001, beta1 = 0.9, beta2 = 0.999) and a mini-batch size of 500.

For the variable trace and dependency enforcement models, each trace is padded to have the same length across each batch; for the state trace model, both the number of variables in each program state as well as the length of the entire state trace are padded.

During the training of the dependency enforcement model, we have observed that when dependencies become complex, the network suffers from optimization issues, such as diminishing and exploding gradients.

This is likely due to the complex nature of fusing hidden states among RNNs, echoing the errors back and forth through the network.

We resolve this issue by truncating each trace into multiple sub-sequences and only back-propagate on the last sub-sequence while only feedforwarding on the rest.

Regarding the baseline network trained on syntactic traces/token sequences, we use the same encoder architecture (i.e., two layer GRU of 200 hidden units) processing the same 100-dimension embedding vector for each statement/token.

As for the AST model, we learn an embedding (100-dimension) for each type of the syntax node by propagating the leaf (a simple look up) to the root through the learned production rules.

Finally, we use the root embeddings to represent programs.

Table 3 : Comparing dynamic program embeddings with syntax-based program embedding in predicting common error patterns made by students.

As shown in Table 3 , our embeddings trained on execution traces significantly outperform those trained on program syntax (greater than 92% accuracy compared to less than 27% for syntax-based embeddings).

We conjecture this is because of the fact that minor syntactic discrepancies can lead to major semantic differences as shown in Figure 1 .

In our dataset, there are a large number of programs with distinct labels that differ by only a few number of tokens or AST nodes, which causes difficulty for the syntax models to generalize.

Even for the simpler syntax-level errors, they are buried in large number of other syntactic variations and the size of the training dataset is relatively small for the syntax-based models to learn precise patterns.

In contrast, dynamic embeddings are able to canonicalize the syntactical variations and pinpoint the underlying semantic differences, which results in the trace-based models learning the correct error patterns more effectively even with relatively smaller size of the training data.

In addition, we incorporated our dynamic program embeddings into SARFGEN BID13 -a program repair system -to demonstrate their benefit in producing fixes to correct students errors in programming assignments.

Given a set of potential repair candidates, SARFGEN uses an enumerative search-based technique to find minimal changes to an incorrect program.

We use the dynamic embeddings to learn a distribution over the corrections to prioritize the search for the repair algorithm.

6 To establish the baseline, we obtain the set of all corrections from SARFGEN for each of the real incorrect program to all three problems and enumerate each subset until we find the minimum fixes.

On the contrary, we also run another experiment where we prioritize each correction according to the prediction of errors with the dynamic embeddings.

It is worth mentioning that one incorrect program may be caused by multiple errors.

Therefore, we only predict the top-1 error each time and repair the program with the corresponding corrections.

If the program is still incorrect, we repeat this procedure till the program is fixed.

The comparison between the two approaches is based on how long it takes them to repair the programs.

Enumerative Search Table 4 : Comparing the enumerative search with those guided by dynamic program embeddings in finding the minimum fixes.

Time is measured in seconds.

As shown in Table 4 , the more fixes required, the more speedups dynamic program embeddings yield -more than an order of magnitude speedups when the number of fixes is four or greater.

When the number of fixes is greater than seven, the performance gain drops significantly due to poor prediction accuracy for programs with too many errors.

In other words, our dynamic embeddings are not viewed by the network as capturing incorrect execution traces, but rather new execution traces.

Therefore, the predictions become unreliable.

Note that we ignored incorrect programs having greater than 10 errors when most experiments run out of memory for the baseline approach.

There has been significant recent interest in learning neural program representations for various applications, such as program induction and synthesis, program repair, and program completion.

Specifically for neural program repair techniques, none of the existing techniques, such as DeepFix BID4 , SynFix BID1 and sk p BID9 , have considered dynamic embeddings proposed in this paper.

In fact, dynamic embeddings can be naturally extended to be a new feature dimension for these existing neural program repair techniques.

BID8 is a notable recent effort targeting program representation.

Piech et al. explore the possibility of using input-output pairs to represent a program.

Despite their new perspective, the direct mapping between input and output of programs usually are not precise enough, i.e., the same input-output pair may correspond to two completely different programs, such as the two sorting algorithms in Figure 1 .

As we often observe in our own dataset, programs with the same error patterns can also result in different input-output pairs.

Their approach is clearly ineffective for these scenarios.

BID10 introduced the novel approach of using execution traces to induce and execute algorithms, such as addition and sorting, from very few examples.

The differences from our work are (1) they use a sequence of instructions to represent dynamic execution trace as opposed to using dynamic program states; (2) their goal is to synthesize a neural controller to execute a program as a sequence of actions rather than learning a semantic program representation; and (3) they deal with programs in a language with low-level primitives such as function stack push/pop actions rather than a high-level programming language.

As for learning representations, there are several related efforts in modeling semantics in sentence or symbolic expressions BID11 BID14 BID2 .

These approaches are similar to our work in spirit, but target different domains than programs.

We have presented a new program embedding that learns program representations from runtime execution traces.

We have used the new embeddings to predict error patterns that students make in their online programming submissions.

Our evaluation shows that the dynamic program embeddings significantly outperform those learned via program syntax.

We also demonstrate, via an additional application, that our dynamic program embeddings yield more than 10x speedups compared to an enumerative baseline for search-based program repair.

Beyond neural program repair, we believe that our dynamic program embeddings can be fruitfully utilized for many other neural program analysis tasks such as program induction and synthesis.

for Pc ∈ Pcs do // Generates the syntactic discrepencies w.r.t.

each Pc 7 C(P , Pc) ← DiscrepenciesGeneration(P , Ps) // Executing P to extract the dynamic execution trace 8 T (P ) ← DynamicTraceExtraction(P ) // Prioritizing subsets of C(P , Pc) through pre-trained model 9 C subs (P , Pc) ← Prioritization(C(P , Pc), T (P ), M) 10 for C sub (P , Pc) ∈ C subs (P , Pc) do

<|TLDR|>

@highlight

A new way of learning semantic program embedding