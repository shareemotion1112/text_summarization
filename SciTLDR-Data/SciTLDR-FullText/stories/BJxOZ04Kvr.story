Implementing correct method invocation is an important task for software developers.

However, this is challenging work, since the structure of method invocation can be complicated.

In this paper, we propose InvocMap, a code completion tool allows developers to obtain an implementation of multiple method invocations from a list of method names inside code context.

InvocMap is able to predict the nested method invocations which their names didn’t appear in the list of input method names given by developers.

To achieve this, we analyze the Method Invocations by four levels of abstraction.

We build a Machine Translation engine to learn the mapping from the first level to the third level of abstraction of multiple method invocations, which only requires developers to manually add local variables from generated expression to get the final code.

We evaluate our proposed approach on six popular libraries: JDK, Android, GWT, Joda-Time, Hibernate, and Xstream.

With the training corpus of 2.86 million method invocations extracted from 1000 Java Github projects and the testing corpus extracted from 120 online forums code snippets, InvocMap achieves the accuracy rate up to 84 in F1- score depending on how much information of context provided along with method names, that shows its potential for auto code completion.

Writing code is a challenge for non-experienced software developers.

To write the code that implements a specific task in a programming language, developers need to remember the syntax of that language and be familiar with how to implement method invocations.

While the syntax of the language is easier to learn since it contains a permanent set of words in the vocabulary, implementing Method Invocations (MI)s is more challenging due to the following reasons.

First of all, developers need to remember the structure and the combination of invocations depending on their purpose.

Secondly, the implementation of method invocation is also depending on the surrounding context of the code.

Thus, the code developed by non-experience developers may be in the risks of being semantic error.

To help developers with interacting and analyzing by a given Java source code snippet, Java Development Tool (JDT) library defines a list of Abstract Syntax Tree (AST) Node types (Eclipse, 2019) .

With the list of these AST Node types, JDT is able to interact with the structure of each elements inside the source code.

MI, which is defined as sub-type of Expression, is one of the fundamental AST Nodes that developers need to implement.

MI has been used to make Application Programming Interface (API) calls from other libraries or from other methods inside a Java project.

The structure of a syntactically correct MI contains method name, receiver and the list of arguments which could be empty.

Since receiver and arguments are types of expression (Eclipse, 2019) , the structure of an MI could be complicated as a deep AST tree.

The reason for this issue is that expression can be composed by different types of AST Node including MI.

An example of a complicated MI is shown in Listing 1.

Within this Listing, the outside MI contains four nested MI in its implementation.

Additionally, there are five positions that requires local variables inside the expression.

Type casting to integer is embedded to this MI to provide a semantically correct MI.

This MI is used along with other calculated MIs inside the body of method, providing the a specific surrounding context for this MI.

Without doubt, the outer method name set is just one word while the respected MI is a deep AST tree.

The representation of MI also relies on code context.

Consider examples 2A and 2B on Listing 2 and Listing 3.

These Listings show the implementation of API android.content.Intent.getBooleanExtra().

Although 2 MIs share the same information about context of using the same local variable Intent and the false boolean literal, they are differ in the structure of AST.

Since the MI in Listing 2 associates with the action of add or remove an application package from an android device, the MI on Listing 3 associates with actions of network status checking.

The difference in contexts brings 2 MIs, which represents in 2 static Field Accesses Intent.

EXTRA REPLACING and ConnectivityManager.

EXTRA NO CONNECTIVITY.

Listing 1: Example in Android (2019a) 1 p u b l i c v o i d s e t O f f s e t s ( i n t n e w H o r i z o n t a l O f f s e t , i n t n e w V e r t i c a l O f f s e t ) { 2 . . .

. . . 5 i n v a l i d a t e R e c t f .

o f f s e t (− x o f f s e t , −y o f f s e t ) ; 6 i n v a l i d a t e R e c t .

s e t ( ( i n t ) Math .

f l o o r ( i n v a l i d a t e R e c t f .

l e f t ) , ( i n t ) Math .

f l o o r ( i n v a l i d a t e R e c t f .

t o p ) , ( i n t ) Math .

c e i l ( i n v a l i d a t e R e c t f .

r i g h t ) , ( i n t ) Math .

c e i l ( i n v a l i d a t e R e c t f .

b o t t o m ) ) ; 7 . . .

Listing 2: Example 2A in Android (2019b) 1 p u b l i c v o i d o n R e c e i v e ( C o n t e x t c o n t e x t , I n t e n t i n t e n t ) { 2 . . .

3 i f ( ( I n t e n t .

ACTION PACKAGE REMOVED .

e q u a l s ( a c t i o n ) | | 4 I n t e n t .

ACTION PACKAGE 5

ADDED .

e q u a l s ( a c t i o n ) ) 6 && !

i n t e n t .

g e t B o o l e a n E x t r a ( I n t e n t .

EXTRA REPLACING , f a l s e ) ) { 7 . . .

Listing 3: Example 2B in Android (2019c) 1 p u b l i c v o i d o n R e c e i v e ( C o n t e x t c o n t e x t , I n t e n t i n t e n t ) { 2 . . .

3 i f ( a c t i v e N e t w o r k == n u l l ) { 4 . . .

5 } e l s e i f ( a c t i v e N e t w o r k .

g e t T y p e ( ) == n e t w o r k T y p e ) { 6 mNetworkUnmetered = f a l s e ; 7 mNetworkConnected = !

i n t e n t .

g e t B o o l e a n E x t r a ( C o n n e c t i v i t y M a n a g e r .

EXTRA NO CONNECTIVITY , f a l s e ) ; 8 . . .

From the examples above, we recognize that implementing an effective method invocation requires strong background and experiences of developers.

Even two MIs that belong to the same API and share the same context of local variables and literal still have ambiguous in the way of implementation like Listing 2 and Listing 3.

These challenges hinders the ability of writing a appropriate MI and as well as developers need to spend time to remember or identify the correct structure of AST in MI for software development.

With this work, we want to tackle this problem by providing InvocMap, a code completion tool for helping developers to achieve the implementation of method invocation efficiently.

InvocMap accepts input as a sequence of method names inside the code environment of a method declaration, then produce the output as the list of ASTs as translation results for each input method names.

The generated ASTs will only require developers to input information about local variables and literals in order to obtain the complete code.

For instance, in Listing 2, developer can write the list of method names including the name getBooleanExtra.

The output for the suggestion will be #.getBooleanExtra( Intent.

EXTRA REPLACING,#), which can be completed manually by a variable of type android.content.Intent in the first "#" and a boolean literal in the second "#".

Statistical Machine Translation (SMT) is a well-known approach in Natural Language Processing (NLP) for translating between languages (Green et al., 2014) .

For taking advantage from SMT, we propose a direction of code completion for Method Invocation by a Statistical approach, which learn the translation from the abstract information of MIs to the their detail information, which are represented by AST with complicate structure.

First and foremost, we analyze the information inside a typical MI.

We divide the MI by four levels of abstraction.

We also define information of context for each MI which can help to predict the AST structure.

Next, we build an SMT engine specified for our work to infer from the very abstract layer of MI, means Method Name, to the third level of MI, which is an AST tree that requires to be fulfill by local variables and literals.

In order to evaluate our approach, we do experiments to check the accuracy of our code completion technique in two data sets collected from Github and from online forums.

Resources of this paper can be found in (InvocMap, 2019) .

This research has following contributions: 2.

Designing rules for extracting code tokens for representing abstract level and details level for various types of AST nodes.

3.

Proposing an algorithm for visiting a method invocation inside the code environment to abstract and encode their structure in AST as an object for statistical learning.

4.

Building a SMT system for learning from the context of code environment, including MIs from large scale Github high quality projects.

This SMT system is able to predict the sequences of AST structure given sequences of method name and context.

We summarize the engines inside InvocMap on Figure 1 .

From the perspective of developers, InvocMap provides a plugin inside with Java code editor to allow them to write a single or multiple method names inside the code environment.

Starting with this input, InvocMap translates each method names to respective ASTs.

These ASTs reflect the complex structure of method invocations which might be inconvenient for developers to remember.

They are abstracted at level 3 in our definition.

That means they only require developers to add local variables, local methods or literals to obtain the final code.

We will discuss about MI at level 3 of abstraction in the next section.

The ability of inferring ASTs for code completion relies on the Statistical Translation module.

The training process is done by the Statistical Learning module.

This module learns information from the data extracted from large scale Github code corpus (Github, 2019) .

In general, our statistical approach takes advantages of the knowledge of implementing MIs from experienced developers, representing it by a machine learning model to help non-experienced developers in retrieving effective implementation of MIs.

Both the source code at developers side and code corpus are analyzed to extract sequences of tokens by the Train AST Visitor and Test AST Visitor modules we developed.

Inside these visitors, we handle each AST Node types by functions of module Context Extractor and MI Abstractor, which we discuss in next sections.

Definition 1 Level 1 of abstraction of a method invocation is the information about method name of that method invocation.

Definition 2 Level 2 of abstraction of a method invocation is the information about type (or signature) of that method invocation.

Definition 3 Level 3 of abstraction of a method invocation is the Abstract Syntax Tree of that method invocation with abstracted place holder for local variables, local methods and literal.

Definition 4 Level 4 of abstraction of a method invocation is the complete Abstract Syntax Tree of that method invocation.

Along with 4 levels of abstraction in MI, we have the definition of local context provided for each MI.

An example of 4 levels is shown in Figure 2 (a).

In this code snippet, we have level 1 as method name println.

The level 2 of abstraction brings us information about type, which is java.io.PrintStream.println.The level 4 is the final source code which is compile-able.

The level 3 is the AST that is having places which are local entities are abstracted by their type information.

In the implementation, we represent this AST in level 3 by 4 fields: the code with abstracted places for local entities, the list of types of required arguments to add to get level 4, the list of imported APIs and the type of MI.

These 4 fields will make an unique identification for the expression, which will serve as a representative token for the AST.

Therefore, developers could know which types of local variables to obtain the final code along with the set of imported APIs when they receive an AST at level 3 of abstraction.

In our work, we focus on the inference from level 1 to level 3 by translation.

We will use information of local context to help developers who already remember what variables should run inside the MI and some words inside the MI to better retrieve the AST of implementation.

In Figure 2 (a), we see 2 local entities, including the string literal "index" and the integer variable i.

The suggested terms can be "System" and "+" sign.

Definition 6 Level 1 of abstraction of other AST Nodes is the information about the Partial Qualified Name (PQN) of type of those nodes.

Definition 7 Level 2 of abstraction of other AST Nodes is the information about Fully Qualified Name (FQN) of type of those nodes.

In the context of this work, we call other AST Nodes as all kinds of AST except the MI that are defined in Eclipse (2019) .

According to definitions of Phan et al. (2018) , an example is the API java.io.File.

In this API, we have File as PQN while we have java.io.File as FQN.

Other AST Nodes tokens.

We extract information about other AST Nodes to provide useful context for MIs prediction.

In the source language, we extract all tokens of level 1 of abstraction for each AST Node, and extract all tokens in level 2 of that AST Node to put into target language.

The implementation of the extraction is the Context Extractor module, which is called inside Train AST Visitor and Test AST Visitor.

MI tokens.

There are two types of information we want to embed for MI: the mapping between method name and the AST along with the information relate to local context.

For the first type of information, the source language will store information about token as level 1 of abstraction of MI, while the target language stores information about level 3 of abstraction of MI.

Besides, information about local context will be stored by level 1 of abstraction in the source and level 2 of abstraction in the target language.

A sequence of tokens for MI in Figure 2 (a) is shown in Figure 2

} 21 e l s e { 22

r e s u l t .

s e t I m p o r t e d A P I s .

add ( g e t T y p e ( node ) ) ; 27 } 28 }

We get information about level 3 of abstraction in MI by proposing an algorithm in Listings 4 and 5.

The abstractMethodInvocation() function is invoked when the Train AST Visitor or Test AST Visitor visit a MI and return the abstraction in level 3 by an instance of AST Level3 class.

This function will use the child class of ASTVisitor called InvocAbstractVisitor defined in Listing 5 (line #12).

This visitor will visit each element inside the MI, check and abstract if the element is a local entity.

This visitor also stores other information about the code of AST, the list of required types for each local entities and the set of imported APIs.

The handling strategy for each types of AST Node inside the MI is implemented in the visitStructure() function in Listing 5(#23).

After visiting and abstracting of MI to an AST Level3, this object is checked by the first four fields defined in Listing 5(#1-#10) to see if its exist in the dictionary or not.

If yes, it will have the id of the existing object in the dictionary.

Otherwise, it will generate a new unique id and will be added to the dictionary.

The dictionary stores information about abstraction at layer 3 of MIs in the training step.

An example of AST Level3 object is shown in Figure 2 (a).

To learn the mapping between source and target language, we apply the SMT (Green et al. (2014) ).

SMT was built from two models: the language model and the translation model.

.

LM is used to predict the next token given a sequence of previous tokens (Koehn et al., 2003) .

The more comprehensive corpus of target language we have, the higher quality of prediction the LM achieves.

LM had been used widely in Software Engineering (SE) researches (Hindle et al., 2012; Hellendoorn et al., 2015; Liu, 2016) with potential results.

The most basic LM in NLP is uni-gram LM, which calculates the probability of each word based on the number of the appearance of that word in the corpus.

This LM provides drawbacks that it doesn't take into account the history of how a word was used in a sequence from training data.

Here we use the n-gram language model, which proposed by Jurafsky & Martin (2009) .

Assume that we have m tokens in the target language AST 1 , ..., AST m , the probability provided by LM is shown in the above equation of Equations 1.

Translation Model.

This model calculates the probability of a phrase from source language that can be translated to a phrase in a target language.

If we have a sentence D as the translated result of sentence S as tokens in the source language, the selection of D as the best candidate is calculated by the below equation in Equations 1.

Since we infer from method names to MIs which are consistent in order, we don't apply the reordering probability in the translation model.

Data Preparation.

To do the evaluation, we select corpus on six well-known libraries.

They are Java Development Kit (JDK), Android, GWT, Joda-Time, Hibernate, and XStream.

These libraries were selected to generate the corpus for other research works (Phan et al. (2018) ; Subramanian et al. (2014) ).

To generate corpus, we select 1000 highest stars Java projects from Github (Github (2019) ), which have most files used APIs from libraries in Table 1a .

For each Java project, InvocMap parses each Java source files by an the Train ASTVisitor module on Figure 1 .

The number of pairs respected to each method body we collect is shown in Table 1a .

Training and Testing Configuration.

To train the SMT model, we use a high-end computer with core-i7 Intel processor and use 32 GB of memory.

We apply our solution using Phrasal Green et al. (2014) .

We allocate Phrasal with phrase length equals to 7.

The total training time requires about 6 hours.

For testing, we evaluate the ability of translation from a sequence of method names to ASTs in level 3 of abstraction.

We simulate 3 configurations sequences of method names regarding to its local context defined in Table 1b .

We can see the local context provided for method names is increasing from configurations at level 1 to level 3.

At level 1, the input for translation contains only method names with the code context in the source language for translation.

It simulates the case that developers write a list of method names inside the code environment.

At level 2, information about partial class name of types of local entities is attached along with each method names.

This case simulates the case developers remember and write method name and local variables they remember as part of the MI, but they don't remember the structure of AST.

At level 3, each method names in the source language will be attached the information about local entities and half of words appeared inside the MI.

This case simulates the case that developers remember some words inside the MI along with local entities.

Metrics.

Information about tokens of method name and MI can be recognized by the annotation #identifier in the source, and the expected results can be recognized by prefix "E-Total" of tokens in the target.

We use Precision and Recall as 2 metrics for the evaluation.

Out of Vocabulary (OOV) result is the case that the method name token does not in the corpus (Out of Source -OOS) or the expected AST in level 3 does not appear in the target corpus (Out of Target -OOT).

We split the pairs of our parallel corpus for training and testing.

We get 10% of the data for testing and the other with training and do ten-fold cross-validation to test the ability of prediction on our full data set.

In total, there are 2.86 Million of MIs collected from 1000 projects from Github Github (2019) .

The evaluation result for intrinsic data is shown in Table 2 .

We show that from configuration 1 to configuration 3, the F1 score increases from 73.06% to 84.4%.

This seems to be feasible, since the fact that if we provide more local context information along with method names, the ability to predict correctly AST in level 3 for the translation model is better.

We see one observation is that the number of Out of Vocabulary expressions are higher in percentage, cause decreasing in recall compare to the research work that applied Machine Translation for inferring Fully Qualified Name from incomplete code (Phan et al. (2018) ).

This is reasonable, since our work requires to infer the MI in level 3 of abstraction, which contains detail structure compared to output of Phan et al. (2018) , which only infers the type information of MI.

We study an example in the Intrinsic Evaluation in Figure 2 (b) .

This example is a function collected from Nexus (2019) from our corpus.

The testing for intrinsic evaluation simulates the case developers input only println inside the code environment, the output of this case will be the implementation of java.io.PrintWriter.println() function.

We can see that the surrounding code is useful to infer the correct expression.

If we do not have the context information, which means developer input println in an empty method, the translated result will return the most popular MI, System.out.println().

To do this experiment, we collect the data as code snippets from Online Forums (StackOverflow, 2019; ProgramCreek, 2019; GeeksForGeeks, 2019) .

A Software Engineer who has 5 years of experience in Java programming was hired to collect code snippets from 120 posts in Online Forums, with 20 posts for each library in Table 1a .

The result for extrinsic evaluation is shown in Table 2 .

We see that with level 1, since the case that only method names are provided in the source language, our approach stills predict correctly 68.5% in F1-score.

With the configuration levels that developers add more information, the F1-score increases to 84%.

For each library, we achieved the highest accuracy on GWT and lowest on Hibernate with input as detail information like configuration 3.

This result seems reasonable, since Hibernate is a bigger library compared to GWT but it is not as popular as JDK, causes the variety of ASTs for APIs in this library.

In this evaluation, we analyze the relation of the expression prediction result relates to the number of mapping of each method name from the parallel corpus.

We use data collected for the Intrinsic Evaluation with configuration 3.

The result, which is shown in Table 1c , reveals that from the number of method name that has more than 100 mappings in the parallel corpus are about 72% of the total data.

It proves the complexity of kinds of implementation for each method names.

The total precision tends to decrease from 96.47 % to 87.68% from low to high number of mappings, means that the prediction is still acceptable although the method names are too ambiguous.

Machine Learning has been applied widely in Software Engineering applications Allamanis et al. (2018) .

Generating code by machine learning is an interesting but also confront challenges.

There is a research by Barone & Sennrich (2017) shows that the inference of code from documentation by machine translation achieved very low accuracy results on both SMT and Neural Machine Translation (NMT) models learned from practical large scale code corpus.

There are two reasons cause to this challenge.

First, large scale code corpus contains noise data Pascarella & Bacchelli (2017) .

Second, the structure of AST Node is complicate for a machine translation system to learn about the syntactically correct of generated code as shown in Barone & Sennrich (2017) .

Gu et al. (2016) propose an approach to achieve the implementation from in natural language description.

However, the output of their tool consists only sequence of APIs which is in level 2 of our abstraction for MIs.

In our work, we target the inference of MI in level 3 with the ability of complex AST structure of MIs.

There are several other inputs to get the complete code in other researches.

Nguyen et al. (2015) derive the code in C# language from code in Java language by machine translation. (Gvero & Kuncak, 2015; Gu et al., 2016) generate the code from natural language descriptions.

In these works, they consider the textual description as the full information for the inference.

We consider our code generation problem in a different angle, which we take advantage of the surrounding context along with the textual description of method name in our work.

Nguyen et al. (2012) propose a graph based code completion tool that suggest the full code snippet when developers are writing an incomplete code.

This work focuses on completing the code from a part of the code.

We propose an inference from the skeleton of method invocations, which is in form of sequence of method names, to the implementation of method invocations.

In this work, we proposed InvocMap, a SMT engine for inferring the ASTs of method invocations from a list of method names and code context.

By the evaluation on corpus collected from Github projects and online forums, we demonstrated the potential of our approach for auto code completion.

A major advantage of InvocMap is that it is built on the idea of abstracting method invocations by four different levels.

We provided an algorithm to achieve AST of method invocations for the method invocations inference.

As future works, we will work on extending the SMT model to support inputs from multiple natural language descriptions of multiple method invocations, along with investigation of machine learning techniques for improving the accuracy.

@highlight

This paper proposes a theory of classifying Method Invocations by different abstraction levels and conducting a statistical approach for code completion from method name to method invocation.