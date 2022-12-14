We investigate the internal representations that a recurrent neural network (RNN) uses while learning to recognize a regular formal language.

Specifically, we train a RNN on positive and negative examples from a regular language, and ask if there is a simple decoding function that maps states of this RNN to states of the minimal deterministic finite automaton (MDFA) for the language.

Our experiments show that such a decoding function indeed exists, and that it maps states of the RNN not to MDFA states, but to states of an {\em abstraction} obtained by clustering small sets of MDFA states into ``''superstates''.

A qualitative analysis reveals that the abstraction often has a simple interpretation.

Overall, the results suggest a strong structural relationship between internal representations used by RNNs and finite automata, and explain the well-known ability of RNNs to recognize formal grammatical structure.

Recurrent neural networks (RNNs) seem "unreasonably" effective at modeling patterns in noisy realworld sequences.

In particular, they seem effective at recognizing grammatical structure in sequences, as evidenced by their ability to generate structured data, such as source code (C++, LaTeX, etc.) , with few syntactic grammatical errors BID9 .

The ability of RNNs to recognize formal languages -sets of strings that possess rigorously defined grammatical structure -is less well-studied.

Furthermore, there remains little systematic understanding of how RNNs recognize rigorous structure.

We aim to explain this internal algorithm of RNNs through comparison to fundamental concepts in formal languages, namely, finite automata and regular languages.

In this paper, we propose a new way of understanding how trained RNNs represent grammatical structure, by comparing them to finite automata that solve the same language recognition task.

We ask: Can the internal knowledge representations of RNNs trained to recognize formal languages be easily mapped to the states of automata-theoretic models that are traditionally used to define these same formal languages?

Specifically, we investigate this question for the class of regular languages, or formal languages accepted by finite automata (FA).In our experiments, RNNs are trained on a dataset of positive and negative examples of strings randomly generated from a given formal language.

Next, we ask if there exists a decoding function: an isomorphism that maps the hidden states of the trained RNN to the states of a canonical FA.

Since there exist infinitely many FA that accept the same language, we focus on the minimal deterministic finite automaton (MDFA) -the deterministic finite automaton (DFA) with the smallest possible number of states -that perfectly recognizes the language.

Our experiments, spanning 500 regular languages, suggest that such a decoding function exists and can be understood in terms of a notion of abstraction that is fundamental in classical system theory.

An abstraction A of a machine M (either finite-state, like an FA, or infinite-state, like a RNN) is a machine obtained by clustering some of the states of M into "superstates".

Intuitively, an abstraction Figure 1: t-SNE plot (Left) of the hidden states of a RNN trained to recognize a regular language specified by a 6-state DFA (Right).

Color denotes DFA state.

The trained RNN has abstracted DFA states 1(green) and 2(blue) (each independently model the pattern [4-6] * ) into a single state.

A loses some of the discerning power of the original machine M, and as such recognizes a superset of the language that M recognizes.

We observe that the states of a RNN R, trained to recognize a regular language L, commonly exibit this abstraction behavior in practice.

These states can be decoded into states of an abstraction A of the MDFA for the language, such that with high probability, A accepts any input string that is accepted by R. Figure 1 shows a t-SNE embedding BID13 of RNN states trained to perform language recognition on strings from the regex [(([4-6] {2}[4-6]+)?)3[4-6]+].

Although the MDFA has 6 states, we observe the RNN abstracting two states into one.

Remarkably, a linear decoding function suffices to achieve maximal decoding accuracy: allowing nonlinearity in the decoder does not lead to significant gain.

Also, we find the abstraction has low "coarseness", in the sense that only a few of the MDFA states need be clustered, and a qualitative analysis reveals that the abstractions often have simple interpretations.

RNNs have long been known to be excellent at recognizing patterns in text BID10 BID9 .

Extensive work has been done on exploring the expressive power of RNNs.

For example, finite RNNs have been shown to be capable of simulating a universal Turing machine BID16 .

BID4 showed that the hidden state of a RNN can approximately represent dynamical systems of the same or less dimensional complexity.

In particularly similar work, BID19 showed that second order RNNs with linear activation functions are expressively equivalent to weighted finite automata.

Recent work has also explored the relationship between RNN internals and DFAs through a variety of methods.

Although there have been multiple attempts at having RNNs learn a DFA structure based on input languages generated from DFAs and push down automata BID3 BID5 BID7 BID14 BID17 , most work has focused on extracting a DFA from the hidden states of a learned RNN.

Early work in this field BID6 demonstrated that grammar rules for regular grammars could indeed be extracted from a learned RNN.

Other studies BID18 tried to directly extract a DFA structure from the internal space of the RNN, often by clustering the hidden state activations from input stimuli, noting the transitions from one state to another given a particular new input stimuli.

Clustering was done by a series of methods, such as K-Nearest Neighbor BID2 , K-means BID11 , and Density Based Spatial Clustering of Applications with Noise (DBSCAN) BID2 BID12 .

Another extraction effort BID1 uses spectral algorithm techniques to extract weighted automata from RNNs.

Most recently, BID21 have achieved state-of-the-art accuracy in DFA extraction by utilizing the L* query learning algorithm.

Our work is different from these efforts in that we directly relate the RNN to a ground-truth minimal DFA, rather than extracting a machine from the RNN's state space.

The closest piece of related work is by BID20 .

Like our work, this seeks to relate a RNN state with the state of a DFA.

However, the RNN in BID20 exactly mimics the DFA; also, the study is carried out in the context of a few specific regular languages that are recognized by automata with 2-3 states.

In contrast, our work does not require exact behavioral correspondence between RNNs and DFAs: DFA states are allowed to be abstracted, leading to loss of information.

Also, in our approach the mapping from RNN states to FA states can be approximate, and the accuracy of the mapping is evaluated quantitatively.

We show that this allows us to establish connections between RNNs and DFAs in the setting of a broad class of regular languages that often demand significantly larger automata (with up to 14 states) than those studied by BID20 .

We start by introducing some definitions and notation.

A formal language is a set of strings over a finite alphabet ?? of input symbols.

A Deterministic Finite Automaton (DFA) is a tuple A = (Q, ??, ??, q 0 , F ) where Q is a finite set of states, ?? is a finite alphabet, ?? : Q ?? ?? ??? Q is a deterministic transition function, q 0 ??? Q is a starting state and F ??? Q is a set of accepting states.

A reads strings over ?? symbol by symbol, starting from the state q 0 and making state transitions, defined by ??, at each step.

It accepts the string if it reaches a final accepting state in F after reading it to the end.

The set of strings accepted by a DFA is a special kind of formal language, known as a regular language.

A regular language L can be accepted by multiple DFAs; such a DFA A L is minimal if there exists no other DFA A = A L such that A exactly recognizes L and has fewer states than A L .

It can be shown that this minimal DFA (MDFA), which we denote by A 0 L , is unique BID8 .Abstractions.

A Nondeterministic Finite Automaton (NFA) is similar to a DFA, except that the deterministic transition function ?? is now a non-deterministic transition relation ?? N F A .

This means that for a state q in the NFA and a ??? ??, we have that ?? N F A (q, a) is now a subset of NFA states.

For a given regular language L we denote by A n L a Nondeterministic Finite Automaton (NFA) with n states that recognizes a superset of the language L. An abstraction map is a map ?? : DISPLAYFORM0 DISPLAYFORM1 We define the coarseness of an abstraction A n L , as the number of applications of ?? on the MDFA required to arrive at A n L .

Intuitively, repeated applications of ?? create NFAs that accept supersets of the language L recognized by the MDFA, and can hence be seen as coarse-grained versions of the original MDFA.

The coarsest NFA, given by A DISPLAYFORM2 , is a NFA with only one accepting node and it accepts all strings on the alphabet ??.Given a regular language L, we define R L to be a RNN that is trained to recognize the language L, with a certain threshold accuracy.

Each RNN R L will have a corresponding set of hidden states denoted by H. More details about the RNN are provided in ??4.1.

Note that a RNN can also be viewed as a transition system with 5-tuple R = (H, ??, ?? R , h 0 , F R ), where H is a set of possible 'hidden' states (typically H ??? R K ), ?? R is the transition function of the trained RNN, h 0 is the initial state of the RNN, and F R is the set of accepting RNN states.

The key distinction between a DFA and a RNN is that the latter has a continuous state space, endowed with a topology and a metric.

Decoding DFA States from RNNs.

Inspired by methods in computational neuroscience BID0 , we can define a decoding function or decoder f : H ??? Q 0 as a function from the hidden states of a RNN R L to the states of the corresponding DISPLAYFORM3 .

We are interested in finding decoding functions that provide an insight into the internal knowledge representation of the RNN, which we quantify via the decoding and transitional accuracy metrics defined below.

DISPLAYFORM4 repeatedly n times, and let Q n be the set of states of A n L .

We can define an abstraction decoding functionf : DISPLAYFORM5 , that is the composition of f with ?? |n| .

The function ?? |n| is the function obtained by taking n compositions of ?? with itself.

Given a dataset of input strings D ??? ?? * , we can define the decoding accuracy of a mapf for an abstraction DISPLAYFORM6 where 1(C) is the boolean indicator function that evaluates to 1 if condition C is true and to 0 otherwise, h t+1 = ?? R (h t , a t ) and q t+1 = ?? 0 (q t , a t ).

Note in particular, that for decoding abstraction states the condition is only checking if the (t + 1) RNN state is mapped to the (t + 1) NFA state byf , which may be true even if the (t + 1) RNN state is not mapped to the (t + 1) MDFA state by the decoding function f .

Therefore a functionf can have a high decoding accuracy even if the underlying f does not preserve transitions.

Decoding Abstract State Transitions.

We now define an accuracy measure that takes into account how well transitions are preserved by the underlying function f .Intuitively, for a given decoding functionf and NFA A n L , we want to check whether the RNN transition on a is mapped to the abstraction of the MDFA transition on a. We note that in the definition of the decoding function, we take into account only the states at (t + 1) and not the underlying transitions in the original MDFA A 0 L , unlike we do here.

More precisely, the transitional accuracy of a mapf for a given RNN and abstraction, with respect to a data-set D, is defined as: DISPLAYFORM7 Our experiments in the next section demonstrate that decoding functions with high decoding and transitional accuracies exist for abstractions with relatively low coarseness.

Our goal is to experimentally test the hypothesis that a high accuracy, low coarseness decoder exists from R L to A ??? L) example strings (see Appendix for details).

We then train R L with D on the language recognition task: given an input string x ??? ?? * , is x ??? L?

Thus, we have two language recognition models corresponding to state transition systems from which state sequences are extracted.

Given a length T input string x = (x 1 , x 2 , x t , ..., x T ) ??? D, let the categorical states generated by A 0 L be denoted by q = (q 0 , q 1 , q t , ..., q T ) and continuous states generated by the R L be h = (h 0 , h 1 , h t , ..., h T ).

The recorded state trajectories (q and h) for all input strings x ??? D are used as inputs into our analysis..

For our experiments, we sample a total of ??? 500 unique L, and thus perform an analysis of ??? 500 recognizer MDFAs and ??? 500 trained recognizer RNNs.

As mentioned in the beginning of ??4, we must first determine what is a reasonable form for the decoders f andf to ensure high accuracy on the decoding task.

FIG4 shows decoding accuracy DISPLAYFORM0 for several different decoding functions f .

We test two linear classifiers (Multinomial Logistic Regression and Linear Support Vector Machines (SVM)) and two non-linear classifiers (SVM with a RBF kernel, Multi-Layer Perceptrons with varying layers and hidden unit sizes).

In order to evaluate whether accuracy varies significantly amongst all decoders, we use a statistically appropriate F-test.

Surprisingly, we find there to be no statistical difference umong our sampled languages: the nonlinear decoders achieve no greater accuracy than the simpler linear decoders.

We also observe in our experiments that as the size of the MDFA M increases, the decoding accuracy decreases for all decoders in a similar manner.

FIG4 shows this relationship for the multinomial logistic regression classifier.

Taken together, these results have several implications.

First, we find that a highly expressive nonlinear decoder does not yield any increase in decoding accuracy, even as we scale up in MDFA complexity.

We can conclude from this finding and our extensive hyperparameter search for each decoder model that the decoder models we chose are expressive enough for the decoding task.

Second, we find that decoding accuracy for MDFA states is in general not very high.

These two observations suggest linear decoders are sufficient for the decoding task, but also suggests the need for a different interpretation of the internal representation of the trained RNN.

Given the information above, how is the hidden state space of the R L organized?

One hypothesis that is consistent with the observations above is that the trained RNN reflects a coarse-grained abstraction of the state space Q 0 (Figure 1) , rather than the MDFA states themselves.

To test this hypothesis, we propose a simple greedy algorithm to find an abstraction mapping ??: (a) given an NFA A n L with n unique states in Q n , consider all (n ??? 1)-partitions of Q n???1 (i.e. two NFA states s, s have merged into a single superstate {s, s }); (b) select the partition with the highest decoding accuracy; (c) Repeat this iterative merging process until only a 2-partition remains.

We note that this algorithm does not explicitly take into consideration the transitions between states which are essential to evaluating ??f (R L , A n L ).

Instead, the transitions are taken into account implicitly while learning the decoder f at each iteration of the abstraction algorithm.

Decreasing the number of states in a classification trivially increases DISPLAYFORM0 .

We compare to a baseline where the states abstracted are random to validate our method.

We compute the normalized Area Under the Curve (AUC) of a decoder accuracy vs coarseness plot.

Higher normalized AUC indicates a more accurate abstraction process.

We argue through FIG5 that our method gives a non-trivial increase over the abstraction performance of a random baseline.

The abstraction algorithm is greedy in the sense that we may not find the globally optimal partition (i.e. with the highest decoding accuracy and lowest coarseness), but an exhaustive search over all partitions is computationally intractable.

The greedy method we have proposed has O(M 2 ) complexity instead, and in practice gives satisfactory results.

Despite it being greedy, we note that the resulting sequence of clusterings are stable with respect to randomly chosen initial conditions and model parameters.

Recognizer RNNs with a different number of hidden units result in clustering sequences that are consistent with each other in the critical first few abstractions.

Once an abstraction ?? has been found, we can evaluate whether the learned abstraction decoderf is of high accuracy, and whether the ?? found is of low coarseness.

Results showing the relationship between high decoding accuracy ??f (R L , A n L ) as a function of coarseness is presented in FIG6 conditioned on the number of nodes in the original MDFA.

As stated in ??4.2, as M increases, ??f (R L , A n L ) decreases on the MDFA (i.e. n = 0).

We attribute this to two factors, (1) as M increases, the decoding problem naturally increases in difficulty, and (2) R L abstracts multiple states of A L into a single state in H as can be seen empirically from Figure 1 .

We validate the second factor by training a overparameterized non-linear decoder on the decoding task and find no instances where the decoder obtains 0% training error.

Alongside the decoding accuracy, we also present transitional accuracy ??f (R L , A n L ) as a function of coarseness FIG6 .

Both of these figures showcase that for a given DFA, in general we can find a low coarseness NFA that the hidden state space of R L can be decoded to with high accuracy.

FIG5 shows the average ratio of abstractions relative to M needed to decode to 90% accuracy, indicating low coarseness relative to a random baseline.

For completeness, we also present decoder and transition accuracy for a nonlinear decoder in FIG7 showing similar results as the linear decoder.

Our fundamental work shows a large scale analysis of how RNNs R L relate to abstracted NFAs A n L for hundreds of minimal DFAs, most of which are much larger and more complex than DFAs typically used in the literature.

By evaluating the transition accuracy between R and A n L we empirically validate our claim.

We show that there does exist high accuracy decoders from R to an abstracted NFA A n L .

4.5 INTERPRETING THE RNN HIDDEN STATE SPACE WITH RESPECT TO THE MINIMAL DFA With an established high accuracyf with low coarseness ?? reveals a unique interpretation of H with respect to A 0 L .

Using ?? and f to relate the two, we uncover an interpretation of how R organizes H with respect to A n L ??? n ??? [M ].

We can then determine the appropriate level of abstraction the network uses to accomplish the logical language recognition task in relation to the underlying MDFA.

We provide two example 'real-world' DFAs to illustrate this interpretation and show several interesting patterns.

We present in FIG9 the clustering sequences of two regular expressions that have real-world interpretations, namely the SIMPLE EMAILS and DATES languages that recognize simple emails and simple dates respectively.

To explain, FIG9 shows the DATES language with its clustering sequence superimposed on the MDFA in the form of a dendrogram.

The dendrogram can be read in a top-down fashion, which displays the membership of the MDFA states and the sequence of abstractions up to n = M ??? 1.

A question then arises: How should one pick a correct level of abstraction n?.

The answer can be seen in the corresponding accuracies ??f (R L , A n L ) in FIG10 .

As n increases and the number of total NFA states decreases, the linear decoding (LDC) prediction task obviously gets easier (100% accuracy when the number of NFA states Q |Q|???1 is 1), and hence it is important to consider how to choose the number of abstractions in the final partition.

We typically set a threshold for ??f (R L , A n L ) and select the minimum n required to achieve the threshold accuracy.

Consider the first two abstractions of the SIMPLE EMAILS DFA.

We notice that both states 2 and 5 represent the pattern matching task [a-d] * , because they are agglomerated by the algorithm.

Once two abstractions have been made, the decoder accuracy is at a sufficient point, as seen in FIG10 .

This suggests that the collection of hidden states for the two states are not linearly separable.

One possible and very likely reason for this is the network has learned an abstraction of the pattern [a-d] * and uses the same hidden state space regardless of location in string to recognize this pattern, which has been indicated in past work BID9 .

This intuitive example demonstrates the RNN's capability to learn and abstract patterns from the DFA.

This makes intuitive sense because R L does not have any direct access to A 0 L , only to samples generated from A 0 L .

The flexibility of RNNs allows such abstractions to be created easily.

The second major pattern that arises can be seen in the dendrogram in the bottom row of FIG9 .

We notice that, generally, multiple states that represent the same location in the input string get merged (1 and 4, 3 and 6, 0 and 5).

The SIMPLE EMAILS dendrogram shows patterns that are location-independent, while the fixed length pattern in the DATES regex shows location-dependent patterns.

We also notice that the algorithm tends to agglomerate states that are within close sequential proximity to each other in the DFA, again indicating location-dependent hierarchical priors.

Overall, our new interpretation of H reveals some new intuitions, empirically backed by our decoding and transitional accuracy scores, regarding how the RNN R L structures the hidden state space H in the task of language recognition.

We find patterns such as these in almost all of the DFA's tested.

We provide five additional random DFA's in the appendix FIG4 ) to show the wide variability of the regular expressions we generate/evaluate on.

We have studied how RNNs trained to recognize regular formal languages represent knowledge in their hidden state.

Specifically, we have asked if this internal representation can be decoded into canonical, minimal DFA that exactly recognizes the language, and can therefore be seen to be the "ground truth".

We have shown that a linear function does a remarkably good job at performing such a decoding.

Critically, however, this decoder maps states of the RNN not to MDFA states, but to states of an abstraction obtained by clustering small sets of MDFA states into "abstractions".

Overall, the results suggest a strong structural relationship between internal representations used by RNNs and finite automata, and explain the well-known ability of RNNs to recognize formal grammatical structure.

We see our work as a fundamental step in the larger effort to study how neural networks learn formal logical concepts.

We intend to explore more complex and richer classes of formal languages, such as context-free languages and recursively enumerable languages, and their neural analogs.

In order to generate a wide variety of strings that are both accepted and rejected by the DFA corresponding to a given regex R, we use the Xeger Java library, built atop the dk.brics.automaton library BID15 .

The Xeger library, given a regular expression, generates strings that are accepted by the regular expression's corresponding DFA.

However, there is no standard method to generate examples that would be rejected by the DFA.

These rejected examples need to be diverse to properly train an acceptor/rejector model: if the rejected examples are completely different from the accepted examples, the model will not be able to discern between similar input strings, even if one is an accepted string and the other is a rejected string.

However, if the rejected examples were too similar to the accepted examples, the model would not be able to make a judgment on a completely new string that does not resemble any input string seen during training.

In other words we want the rejected strings to be drawn from two distinct distributions, one similar and one independent compared to the distribution of the accepted strings.

In order to achieve this, we generate negative examples in two ways: First, we randomly swap two characters in an accepted example enough times until we no longer have an accepted string.

And secondly, we take an accepted string and randomly shuffle the characters, adding it to our dataset if the resulting string is indeed rejected.

In our experiments we generate 1000 training examples with a 50:50 accept/reject ratio.

When applicable we generate strings of varying length capped at some constant, for example with the SIMPLE EMAILS language we generate strings of at most 20 characters.

@highlight

Finite Automata Can be Linearly decoded from Language-Recognizing RNNs using low coarseness abstraction functions and high accuracy decoders. 