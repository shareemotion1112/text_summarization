The key attribute that drives the unprecedented success of modern Recurrent Neural Networks (RNNs) on learning tasks which involve sequential data, is their ever-improving ability to model intricate long-term temporal dependencies.

However, a well established measure of RNNs' long-term memory capacity is lacking, and thus formal understanding of their ability to correlate data throughout time is limited.

Though depth efficiency in convolutional networks is well established by now, it does not suffice in order to account for the success of deep RNNs on inputs of varying lengths, and the need to address their 'time-series expressive power' arises.

In this paper, we analyze the effect of depth on the ability of recurrent networks to express correlations ranging over long time-scales.

To meet the above need, we introduce a measure of the information flow across time that can be supported by the network, referred to as the Start-End separation rank.

Essentially, this measure reflects the distance of the function realized by the recurrent network from a function that models no interaction whatsoever between the beginning and end of the input sequence.

We prove that deep recurrent networks support Start-End separation ranks which are exponentially higher than those supported by their shallow counterparts.

Moreover, we show that the ability of deep recurrent networks to correlate different parts of the input sequence increases exponentially as the input sequence extends, while that of vanilla shallow recurrent networks does not adapt to the sequence length at all.

Thus, we establish that depth brings forth an overwhelming advantage in the ability of recurrent networks to model long-term dependencies, and provide an exemplar of quantifying this key attribute which may be readily extended to other RNN architectures of interest, e.g. variants of LSTM networks.

We obtain our results by considering a class of recurrent networks referred to as Recurrent Arithmetic Circuits (RACs), which merge the hidden state with the input via the Multiplicative Integration operation.

Over the past few years, Recurrent Neural Networks (RNNs) have become the prominent machine learning architecture for modeling sequential data, having been successfully employed for language modeling (Sutskever et al., 2011; Graves, 2013) , neural machine translation (Bahdanau et al., 2014) , speech recognition (Graves et al., 2013; BID1 , and more.

The success of recurrent networks in learning complex functional dependencies for sequences of varying lengths, readily implies that long-term and elaborate correlations in the given inputs are somehow supported by these networks.

However, formal understanding of the influence of a recurrent network's structure on its expressiveness, and specifically on its ever-improving ability to integrate data throughout time (e.g. translating long sentences, answering elaborate questions), is lacking.

An ongoing empirical effort to successfully apply recurrent networks to tasks of increasing complexity and temporal extent, includes augmentations of the recurrent unit such as Long Short Term Memory (LSTM) networks (Hochreiter and Schmidhuber, 1997) and their variants (e.g. Cho et al. (2014) ).

A parallel avenue, which we focus on in this paper, includes the stacking of layers to form deep recurrent networks (Schmidhuber, 1992) .

Deep recurrent networks, which exhibit empirical superiority over shallow ones (see e.g. Graves et al. (2013) ), implement hierarchical processing of information at every time-step that accompanies their inherent time-advancing computation.

Evidence for a time-scale related effect arises from experiments (Hermans and Schrauwen, 2013) -deep recurrent networks appear to model correlations which correspond to longer time-scales than shallow ones.

These findings, which imply that depth brings forth a considerable advantage in complexity and in temporal capacity of recurrent networks, have no adequate theoretical explanation.

In this paper, we address the above presented issues.

Based on the relative maturity of depth efficiency results in neural networks, namely results that show that deep networks efficiently express functions that would require shallow ones to have a super-polynomial size (e.g. Cohen et al. (2016) ; Eldan and Shamir (2016) ), it is natural to assume that depth has a similar effect on the expressiveness of recurrent networks.

Indeed, we show that depth efficiency holds for recurrent networks.

However, the distinguishing attribute of recurrent networks, is their inherent ability to cope with varying input sequence length.

Thus, once establishing the above depth efficiency in recurrent networks, a basic question arises, which relates to the apparent depth enhanced long-term memory in recurrent networks: Do the functions which are efficiently expressed by deep recurrent networks correspond to dependencies over longer time-scales?

We answer this question, by showing that depth provides an exponential boost to the ability of recurrent networks to model long-term dependencies.

In order to take-on the above question, we introduce in section 2 a recurrent network referred to as a recurrent arithmetic circuit (RAC) that shares the architectural features of RNNs, and differs from them in the type of non-linearity used in the calculation.

This type of connection between state-of-the-art machine learning algorithms and arithmetic circuits (also known as Sum-Product Networks (Poon and Domingos, 2011)) has well-established precedence in the context of neural networks.

Delalleau and Bengio (2011) prove a depth efficiency result on such networks, and Cohen et al. (2016) theoretically analyze the class of Convolutional Arithmetic Circuits which differ from common ConvNets in the exact same fashion in which RACs differ from more standard RNNs.

Conclusions drawn from such analyses were empirically shown to extend to common ConvNets (e.g. Sharir and Shashua (2017) ; Levine et al. (2017) ).

Beyond their connection to theoretical models, the modification which defines RACs resembles that of Multiplicative RNNs (Sutskever et al., 2011) and of Multiplicative Integration networks (Wu et al., 2016) , which provide a substantial performance boost over many of the existing RNN models.

In order to obtain our results, we make a connection between RACs and the Tensor Train (TT) decomposition (Oseledets, 2011) , which suggests that Multiplicative RNNs may be related to a generalized TT-decomposition, similar to the way Cohen and Shashua (2016) connected ReLU ConvNets to generalized tensor decompositions.

We move on to introduce in section 3 the notion of Start-End separation rank as a measure of the recurrent network's ability to model elaborate long-term dependencies.

In order to analyze the longterm correlations of a function over a sequential input which extends T time-steps, we partition the inputs to those which arrive at the first T /2 time-steps ("Start") and the last T /2 time-steps ("End"), and ask how far the function realized by the recurrent network is from being separable w.r.t.

this partition.

Distance from separability is measured through the notion of separation rank (Beylkin and Mohlenkamp, 2002) , which can be viewed as a surrogate of the L 2 distance from the closest separable function.

For a given function, high Start-End separation rank implies that the function induces strong correlation between the beginning and end of the input sequence, and vice versa.

In section 4 we directly address the depth enhanced long-term memory question above, by examining depth L = 2 RACs and proving that functions realized by these deep networks enjoy Start-End separation ranks that are exponentially higher than those of shallow networks, implying that indeed these functions can model more elaborate input dependencies over longer periods of time.

An additional reinforcing result is that the Start-End separation rank of the deep recurrent network grows exponentially with the sequence length, while that of the shallow recurrent network is independent of the sequence length.

Informally, this implies that vanilla shallow recurrent networks are inadequate in modeling correlations of long input sequences, since in contrast to the case of deep recurrent networks, the modeled dependencies achievable by shallow ones do not adapt to the actual length of the input.

Finally, we present and motivate a quantitative conjecture by which the Start-End separation rank of recurrent networks grows exponentially with the network depth.

A proof of this conjecture, which will provide an even deeper insight regarding the advantages of depth in recurrent networks, is left as an open problem.

In this section, we introduce a class of recurrent networks referred to as Recurrent Arithmetic Circuits (RACs), which shares the architectural features of standard RNNs.

As demonstrated below, Figure 1 : Shallow and deep recurrent networks, as described by eqs. 1 and 4, respectively.

the operation of RACs on sequential data is identical to the operation of RNNs, where a hidden state mixes information from previous time-steps with new incoming data (see fig. 1 ).

The two classes differ only in the type of non-linearity used in the calculation, as described by eqs. 1-3.

In the following sections, we utilize the algebraic properties of RACs for proving results regarding their ability to model long-term dependencies of their inputs.

We present below the basic framework of shallow recurrent networks ( fig. 1(a) ), which describes both the common RNNs and the newly introduced RACs.

A recurrent network is a network that models a discrete-time dynamical system; we focus on an example of a sequence to sequence classification task into one of the categories {1, ..., C} ??? [C].

Denoting the temporal dependence by t, the sequential input to the network is {x t ??? X } T t=1 , and the output is a sequence of class scores vectors DISPLAYFORM0 , where L is the network depth, ?? denotes the parameters of the recurrent network, and T represents the extent of the sequence in time-steps.

We assume the input lies in some input space X that may be discrete (e.g. text data) or continuous (e.g. audio data), and that some initial mapping f : X ??? R M is preformed on the input, so that all input types are mapped to vectors f (x t ) ??? R M .

The function f (??) may be viewed as an encoding, e.g. words to vectors or images to a final dense layer via some trained ConvNet.

The output at time t ??? [T ] of the shallow (depth L = 1) recurrent network with R hidden channels, depicted in fig. 1(a) , is given by: DISPLAYFORM1 DISPLAYFORM2 where h t ??? R R is the hidden state of the network at time t (h 0 is some initial hidden state), ?? denotes the learned parameters DISPLAYFORM3 , which are the input, hidden and output weights matrices respectively, and g is some non-linear operation.

A bias term is usually added to eq. 1, however, because it bears no effect on our analysis, we omit it for simplicity.

For common RNNs, the non-linearity is given by: DISPLAYFORM4 where ??(??) is typically some point-wise non-linearity such as sigmoid, tanh etc.

For the newly introduced class of RACs, g is given by: DISPLAYFORM5 where the operation stands for element-wise multiplication between vectors, for which the resultant vector upholds (a b) i = a i ?? b i .

This form of merging the input and the hidden state by multiplication rather than addition is referred to as Multiplicative Integration (Wu et al., 2016) .The extension to deep recurrent networks is natural, and we follow the common approach (see e.g. Hermans and Schrauwen (2013) ) where each layer acts as a recurrent network which receives the hidden state of the previous layer as its input.

The output at time t of the depth L recurrent network with R hidden channels in each layer, 1 depicted in fig. 1(b) , is constructed by the following: DISPLAYFORM6 where h t,l ??? R R is the state of the depth l hidden unit at time t (h 0,l is some initial hidden state per layer), and ?? denotes the learned parameters.

Specifically, DISPLAYFORM7 are the input and hidden weights matrices at depth l, respectively.

For l = 1, the weights matrix which multiplies the inputs vector has the appropriate dimensions: W I,1 ??? R R??M .

The output weights matrix is W O ??? R C??R as in the shallow case, representing a final calculation of the scores for all classes 1 through C at every time-step.

The non-linear operation g determines the type of the deep recurrent network, where a common deep RNN is obtained by choosing g = g RNN (eq. 2), and a deep RAC is obtained for g = g RAC (eq. 3).We consider the newly presented class of RACs to be a good surrogate of common RNNs.

Firstly, there is an obvious structural resemblance between the two classes, as the recurrent aspect of the calculation has the exact same form in both networks ( fig. 1 ).

In fact, recurrent networks that include Multiplicative Integration similarly to RACs, have been shown to outperform many of the existing RNN models (Sutskever et al., 2011; Wu et al., 2016) .

Secondly, as mentioned above, arithmetic circuits have been successfully used as surrogates of convolutional networks.

The fact that Cohen and Shashua (2016) laid the foundation for extending the proof methodologies of convolutional arithmetic circuits to common ConvNets with ReLU activations, suggests that such adaptations may be made in the recurrent network analog, rendering the newly proposed class of recurrent networks all the more interesting.

In the following sections, we make use of the algebraic properties of RACs in order to obtain clear-cut observations regarding the benefits of depth in recurrent networks.

In this section, we establish means for quantifying the ability of recurrent networks to model longterm temporal dependencies in the sequential input data.

We begin by introducing the Start-End separation-rank of the function realized by a recurrent network as a measure of the amount of information flow across time that can be supported by the network.

We then tie the Start-End separation rank to the algebraic concept of grid tensors (Hackbusch, 2012), which will allow us to employ tools and results from tensorial analysis in order to show that depth provides an exponential boost to the ability of recurrent networks to model elaborate long-term temporal dependencies.

We define below the concept of the Start-End separation rank for functions realized by recurrent networks after T time-steps, i.e. real functions that take as input X = (x 1 , . . .

, x T ) ??? X T .

The separation rank quantifies a function's distance from separability with respect to two disjoint subsets of its inputs.

Specifically, let (S, E) be a partition of input indices, such that S = {1, . . .

, T /2} and E = { T /2 + 1, . . .

, T } (we consider even values of T throughout the paper for convenience of presentation).

This implies that {x s } s???S are the first T /2 ("Start") inputs to the network, and {x e } e???E are the last T /2 ("End") inputs to the network.

For a function y : X T ??? R, the Start-End separation rank is defined as follows: DISPLAYFORM0 DISPLAYFORM1 In words, it is the minimal number of summands that together give y, where each summand is separable w.r.t. (S, E), i.e. is equal to a product of two functions -one that intakes only inputs from the first T /2 time-steps, and another that intakes only inputs from the last T /2 time-steps.

The separation rank w.r.t.

a general partition of the inputs was introduced in Beylkin and Mohlenkamp (2002) for high-dimensional numerical analysis, and was employed for various applications, e.g. chemistry (Harrison et al., 2003) , particle engineering (Hackbusch, 2006) , and machine learning (Beylkin et al., 2009) .

Cohen and Shashua (2017) connect the separation rank to the L 2 distance of the function from the set of separable functions, and use it to measure correlations modeled by deep convolutional networks.

Levine et al. (2017) tie the separation rank to the family of quantum entanglement measures, which quantify correlations in many-body quantum systems.

In our context, if the Start-End separation rank of a function realized by a recurrent network is equal to 1, then the function is separable, meaning it cannot model any interaction between the inputs which arrive at the beginning of the sequence and the inputs that follow later, towards the end of the sequence.

Specifically, if sep (S,E) (y) = 1 then there exist g s : X T /2 ??? R and DISPLAYFORM2 , and the function y cannot take into account consistency between the values of {x 1 , . . .

, x T /2 } and those of {x T /2+1 , . . .

, x T }.

In a statistical setting, if y were a probability density function, this would imply that {x 1 , . . .

, x T /2 } and {x T /2+1 , . . .

, x T } are statistically independent.

The higher sep (S,E) (y) is, the farther y is from this situation, i.e. the more it models dependency between the beginning and the end of the inputs sequence.

Stated differently, if the recurrent network's architecture restricts the hypothesis space to functions with low Start-End separation ranks, a more elaborate long-term temporal dependence, which corresponds to a function with a higher Start-End separation rank, cannot be learned.

In section 4 we show that deep RACs support Start-End separations ranks which are exponentially larger than those supported by shallow RACs, and are therefore much better fit to model long-term temporal dependencies.

To this end, we employ in the following sub-section the algebraic tool of grid tensors that will allow us to evaluate the Start-End separation ranks of deep and shallow RACs.

We begin by laying out basic concepts in tensor theory required for the upcoming analysis.

The core concept of a tensor may be thought of as a multi-dimensional array.

The order of a tensor is defined to be the number of indexing entries in the array, referred to as modes.

The dimension of a tensor in a particular mode is defined as the number of values taken by the index in that mode.

If A is a tensor of order T and dimension M i in each mode i ??? [T ], its entries are denoted A d1...d T , where the index in each mode takes values d i ??? [M i ].

A fundamental operator in tensor analysis is the tensor product, which we denote by ???. It is an operator that intakes two tensors A ??? R M1??????????M P and B ??? R M P +1 ??????????M P +Q , and returns a tensor A ??? B ??? R M1??????????M P +Q defined by: DISPLAYFORM0 An additional concept we will make use of is the matricization of A w.r.t.

the partition (S, E), denoted A S,E , which is essentially the arrangement of the tensor elements as a matrix whose rows correspond to S and columns to E (formally presented in appendix C).We consider the function realized by a shallow RAC with R hidden channels, which computes the score of class c ??? [C] at time T .

This function, which is given by a recursive definition in eqs. 1 and 3, can be alternatively written in the following closed form: DISPLAYFORM1 where the order T tensor A T,1,?? c, which lies at the heart of the above expression, is referred to as the shallow RAC weights tensor, since its entries are polynomials in the network weights ??. Specifically, denoting the rows of the input weights matrix, W I , by a I,?? ??? R M (or element-wise: a DISPLAYFORM2 DISPLAYFORM3 , the shallow RAC weights tensor can be gradually constructed in the following fashion: DISPLAYFORM4 having set h 0 = W H ??? 1, where ??? is the pseudoinverse operation.

In the above equation, the tensor products, which appear inside the sums, are directly related to the Multiplicative Integration property of RACs (eq. 3).

The sums originate in the multiplication of the hidden states vector by the hidden weights matrix at every time-step (eq. 1).

The construction of the shallow RAC weights tensor, presented in eq. 7, is referred to as a Tensor Train (TT) decomposition of TT-rank R in the tensor analysis community and is analogously described by a Matrix Product State (MPS) Tensor Network (see Or??s (2014) ) in the quantum physics community.

See appendix A for the Tensor Networks construction of deep and shallow RACs, which provides graphical insight regarding the exponential complexity brought forth by depth in recurrent networks.

We now present the concept of grid tensors, which are a form of function discretization.

Essentially, the function is evaluated for a set of points on an exponentially large grid in the input space and the outcomes are stored in a tensor.

Formally, fixing a set of template vectors x(1) , . . .

, x (M ) ??? X , the points on the grid are the set {( DISPLAYFORM5 , the set of its values on the grid arranged in the form of a tensor are called the grid tensor induced DISPLAYFORM6 The grid tensors of functions realized by recurrent networks, will allow us to calculate their separations ranks and establish definitive conclusions regarding the benefits of depth these networks.

Having presented the tensorial structure of the function realized by a shallow RAC, as given by eqs. 6 and 7 above, we are now in a position to tie its Start-End separation rank to its grid tensor, as formulated in the following claim: be its shallow RAC weights tensor, constructed according to eq. 7.

Assume that the network's initial mapping functions DISPLAYFORM7 DISPLAYFORM8 are linearly independent, and that they, as well as the functions g ?? , g ?? in the definition of Start-End separation rank (eq. 5), are measurable and squareintegrable.2 Then, there exist template vectors x (1) , . . . , x (M ) ??? X such that the following holds: DISPLAYFORM9 where A(y DISPLAYFORM10 ) is the grid tensor of y DISPLAYFORM11 with respect to the above template vectors.

Proof.

See appendix B.1.The above claim establishes an equality between the Start-End separation rank and the rank of the matrix obtained by the corresponding grid tensor matricization, denoted A(y T,1,?? c ) S,E , with respect to a specific set of template vectors.

Note that the limitation to specific template vectors does not restrict our results, as grid tensors are merely a tool used to bound the separation rank.

The additional equality to the rank of the matrix obtained by matricizing the shallow RAC weights tensor, will be of use to us when proving our main results below (theorem 1).Due to the inherent use of data duplication in the computation preformed by a deep RAC (see appendix A.3 for further details), it cannot be written in a closed tensorial form similar to that of eq. 6.

This in turn implies that the equality shown in claim 1 does not hold for functions realized by deep RACs.

The following claim introduces a fundamental relation between a function's StartEnd separation rank and the rank of the matrix obtained by the corresponding matricization.

This relation, which holds for all functions, is formulated below for functions realized by deep RACs: (1) , . . .

, x (M ) ??? X it holds that: DISPLAYFORM12 DISPLAYFORM13 where A(y DISPLAYFORM14 ) is the grid tensor of y T,L,?? c with respect to the above template vectors.

Proof.

See appendix B.2.Claim 2 will allow us to provide a lower bound on the Start-End separation rank of functions realized by deep RACs, which we show to be exponentially higher than the Start-End separation rank of functions realized by shallow RACs (to be obtained via claim 1).

Thus, in the next section, we employ the above presented tools to show that an exponential enhancement of the Start-End separation rank is brought forth by depth in recurrent networks.

In this section, we present the main theoretical contributions of this paper.

In section 4.1, we formally present a result which exponentially separates between the memory capacity of a deep (L = 2) recurrent network and a shallow (L = 1) one.

Following the formal presentation of results in theorem 1, we discuss some of their implications and then conclude by sketching a proof outline for the theorem (full proof is relegated to appendix B.3).

In section 4.2, we present a quantitative conjecture regarding the enhanced memory capacity of deep recurrent networks of general depth L, which relies on the inherent combinatorial properties of the recurrent network's computation.

We leave the formal proof of this conjecture for future work.4.1 SEPARATING BETWEEN SHALLOW AND DEEP RECURRENT NETWORKS Theorem 1 states, that the correlations modeled between the beginning and end of the input sequence to a recurrent network, as measured by the Start-End separation rank (see section 3.1), can be exponentially more complex for deep networks than for shallow ones: DISPLAYFORM0 be the function computing the output after T time-steps of an RAC with L layers, R hidden channels per layer, weights denoted by ??, and initial hidden states DISPLAYFORM1 Assume that the network's initial mapping functions DISPLAYFORM2 be the Start-End separation rank of y T,L,?? c (eq. 5).

Then, the following holds almost everywhere, i.e. for all values of ????h 0,l but a set of Lebesgue measure zero: DISPLAYFORM3 is the multiset coefficient, given in the binomial form by DISPLAYFORM4 The above theorem readily implies that depth entails an enhanced ability of recurrent networks to model long-term temporal dependencies in the sequential input.

Specifically, theorem 1 indicates depth efficiency -it ensures us that upon randomizing the weights of a deep RAC with R hidden channels per layer, with probability 1 the function realized by it after T time-steps may only be realized by a shallow RAC with a number of hidden channels that is exponentially large.

3 Stated alternatively, this means that almost all functional dependencies which lie in the hypothesis space of deep RACs with R hidden channels per layer, calculated after T time-steps, are inaccessible to shallow RACs with less than an exponential number of hidden channels.

Thus, a shallow recurrent network would require exponentially more parameters than a deep recurrent network, if it is to implement the same function.

The established role of the Start-End separation rank as a correlation measure between the beginning and the end of the sequence (see section 3.1), implies that these functions, which are realized by almost any deep network and can never be realized by a shallow network of a reasonable size, represent more elaborate correlations over longer periods of time.

The above notion is strengthened by the fact that the Start-End separation rank of deep RACs increases with the sequence length T , while the Start-End separation rank of shallow RACs is independent of it.

This indicates that shallow recurrent networks are much more restricted in modeling long-term correlations than the deep ones, which enjoy an exponentially increasing Start-End separation rank as time progresses.

Below, we present an outline of the proof for theorem 1 (see appendix B.3 for the full version):Proof sketch of theorem 1.1.

For a shallow network, claim 1 establishes that the Start-End separation rank of the function realized by a shallow (L = 1) RAC is equal to the rank of the matrix obtained by matricizing the corresponding shallow RAC weights tensor (eq. 6) according to the StartEnd partition: DISPLAYFORM5 ) S,E .

Thus, it suffices to prove that rank A T,1,?? c ) S,E = R in order to satisfy bullet (1) of the theorem, as the rank is trivially upper-bounded by the dimension of the matrix, M T /2 .

To this end, we call upon the TT-decomposition of A T,1,?? c , given by eq. 7, which corresponds to the MPS Tensor Network presented in appendix A. We rely on a recent result by Levine et al. (2017) , who 3 The combinatorial coefficient DISPLAYFORM6 is exponentially dependent onR ??? min{M, R}: for T > 2 * (R ??? 1) this value is larger than state that the rank of the matrix obtained by matricizing any tensor according to a partition (S, E), is equal to a min-cut separating S from E in the Tensor Network graph representing this tensor.

The required equality follows from the fact that the TT-decomposition in eq. 7 is of TT-rank R, which in turn implies that the min-cut in the appropriate Tensor Network graph is equal to R.2.

For a deep network, claim 2 assures us that the Start-End separation rank of the function realized by a depth L = 2 RAC is lower bounded by the rank of the matrix obtained by the corresponding grid tensor matricization: DISPLAYFORM7 for all of the values of parameters ?? ?? h 0,l but a set of Lebesgue measure zero, would satisfy the theorem, and again, the rank is trivially upper-bounded by the dimension of the matrix, M T /2 .

We use a lemma proved in Sharir et al. (2016) , which states that since the entries of A(y T,L,?? c ) are polynomials in the deep recurrent network's weights, it suffices to find a single example for which the rank of the matricized grid tensor is greater than the desired lower bound.

Finding such an example would indeed imply that for almost all of the values of the network parameters, the desired inequality holds.

We choose a weight assignment such that the resulting matricized grid tensor resembles a matrix obtained by raising a rank-R ??? min{M, R} matrix to the Hadamard power of degree T /2.

This operation, which raises each element of the original rank-R matrix to the power of T /2, was shown to yield a matrix with a rank upper-bounded by the multiset coefficient DISPLAYFORM8 (see e.g. BID0 ).

We show that our assignment results in a matricized grid tensor with a rank which is not only upper-bounded by this value, but actually achieves it.

Theorem 1 provides a lower bound of DISPLAYFORM0 on the Start-End separation rank of depth L = 2 recurrent networks, exponentially separating deep recurrent networks from shallow ones.

By a trivial assignment of weights in higher layers, the Start-End separation rank of even deeper recurrent networks (L > 2) is also lower-bounded by this expression, which does not depend on L. In the following, we conjecture that a tighter lower bound holds for networks of depth L > 2, the form of which implies that the memory capacity of deep recurrent networks grows exponentially with the network depth: Conjecture 1.

Under the same conditions as in theorem 1, for all values of ?? ?? h 0,l but a set of Lebesgue measure zero, it holds for any L that: DISPLAYFORM1 We motivate conjecture 1 by investigating the combinatorial nature of the computation performed by a deep RAC.

By constructing Tensor Networks which correspond to deep RACs, we attain an informative visualization of this combinatorial perspective.

In appendix A, we provide full details of this construction and present the formal motivation for the conjecture.

Below, we qualitatively outline this combinatorial approach.

A Tensor Network is essentially a graphical tool for representing algebraic operations which resemble multiplications of vectors and matrices, between higher order tensors.

FIG3 shows an example of the Tensor Network representing the computation of a depth L = 3 RAC after T = 6 time-steps.

This well-defined computation graph hosts the values of the weight matrices at its nodes.

The inputs {x 1 , . . . , x T } are marked by their corresponding time-step {1, . . .

, T }, and are integrated in a depth dependent and time-advancing manner (see further discussion regarding this form in appendix A.3), as portrayed in the example of FIG3 .

We highlight in red the basic unit in the Tensor Network which connects "Start" inputs {1, . . .

, T /2} and "End" inputs { T /2+1, . . .

, T }.

In order to estimate a lower bound on the Start-End separation rank of a depth L > 2 recurrent network, we employ a similar strategy to that presented in the proof sketch of the L = 2 case (see section 4.1).

Specifically, we rely on the fact that it is sufficient to find a specific instance of the network parameters ?? ?? h 0,l for which A(y T,L,?? c ) S,E achieves a certain rank, in order for this rank to bound the Start-End separation rank of the network from below.

Indeed, we find a specific assignment of the network weights, presented in appendix A.4, for which the Tensor Network effectively takes the form of the basic unit connecting "Start" and "End", raised to the power of the number of its repetitions in the graph (bottom of FIG3 ).

This basic unit corresponds to a simple computation represented by a grid tensor with Start-End matricization of rank R. Raising such a matrix to the Hadamard power of any p ??? Z, results in a matrix with a rank upper bounded by DISPLAYFORM2 , and the challenge of proving the conjecture amounts to proving that the upper bound is tight in this case.

In appendix A.4, we prove that the number of repetitions of the basic unit connecting "Start" and "End" in the deep RAC Tensor Network graph, is exactly equal to DISPLAYFORM3 for any depth L. For example, in the T = 6, L = 3 network illustrated in FIG3 , the number of repetitions indeed corresponds to p = 3 2 = 6.

It is noteworthy that for L = 1, 2 the bound in conjecture 1 coincides with the bounds that were proved for these depths in theorem 1.Conjecture 1 indicates that beyond the proved exponential advantage in memory capacity of deep networks over shallow ones, a further exponential separation may be shown between recurrent networks of different depths.

We leave the proof of this result, which can reinforce and refine the understanding of advantages brought forth by depth in recurrent networks, as an open problem.

The notion of depth efficiency, by which deep networks efficiently express functions that would require shallow networks to have a super-polynomial size, is well established in the context of convolutional networks.

However, recurrent networks differ from convolutional networks, as they are suited by design to tackle inputs of varying lengths.

Accordingly, depth efficiency alone does not account for the remarkable performance of recurrent networks on long input sequences.

In this paper, we identified a fundamental need for a quantifier of 'time-series expressivity', quantifying the memory capacity of recurrent networks.

In order to meet this need, we proposed a measure of the ability of recurrent networks to model long-term temporal dependencies, in the form of the Start-End separation rank.

The separation rank was used to quantify correlations in convolutional networks, and has roots in the field of quantum physics.

The proposed measure adjusts itself to the temporal extent of the input series, and quantifies the ability of the recurrent network to correlate the incoming sequential data as time progresses.

We analyzed the class of Recurrent Arithmetic Circuits, which are closely related to successful RNN architectures, and proved that the Start-End separation rank of deep RACs increases exponentially as the input sequence extends, while that of shallow RACs is independent of the input length.

These results, which demonstrate that depth brings forth an overwhelming advantage in the ability of recurrent networks to model long-term dependencies, were achieved by combining tools from the fields of measure theory, tensorial analysis, combinatorics, graph theory and quantum physics.

Such analyses may be readily extended to other architectural features employed in modern recurrent networks.

Indeed, the same time-series expressivity question may now be applied to the different variants of LSTM networks, and the proposed notion of Start-End separation rank may be employed for quantifying their memory capacity.

We have demonstrated that such a treatment can go beyond unveiling the origins of the success of a certain architectural choice, and leads to new insights.

The above established observation that correlations achievable by vanilla shallow recurrent network do not adapt at all to the sequence length, is an exemplar of this potential.

Moreover, practical recipes may emerge by such theoretical analyses.

The experiments preformed in Hermans and Schrauwen (2013) , suggest that shallow layers of recurrent networks are related to short time-scales, e.g. in speech: phonemes, syllables, words, while deeper layers appear to support correlations of longer time-scales, e.g. full sentences, elaborate questions.

These findings open the door to further depth related investigations in recurrent networks, and specifically the role of each layer in modeling temporal correlations may be better understood.

Levine et al. (2017) establish theoretical observations which translate into practical conclusions regarding the number of hidden channels to be chosen for each layer in a deep convolutional network.

The conjecture presented in this paper, by which the Start-End separation rank of recurrent networks grows exponentially with depth, can similarly entail practical recipes for enhancing their memory capacity.

Such analyses can be reinforced by experiments, and lead to a profound understanding of the contribution of deep layers to the recurrent network's memory.

Indeed, we view this work as an important step towards novel methods of matching the recurrent network architecture to the temporal correlations in a given sequential data set.

We begin in section A.1 by providing a brief introduction to TNs.

Next, we present in section A.2 the TN which corresponds to the calculation of a shallow RAC, and tie it to a common TN architecture referred to as a Matrix Product State (MPS) (see overview in e.g. Or??s (2014)), and equivalently to the tensor train (TT) decomposition (Oseledets, 2011) .

Subsequently, we present in section A.3 a TN construction of a deep RAC, and emphasize the characteristics of this construction that are the origin of the enhanced ability of deep RACs to model elaborate temporal dependencies.

Finally, in section A.4, we make use of the above TNs construction in order to formally motivate conjecture 1, according to which the Start-End separation rank of RACs grows exponentially with depth.

A TN is a weighted graph, where each node corresponds to a tensor whose order is equal to the degree of the node in the graph.

Accordingly, the edges emanating out of a node, also referred to as its legs, represent the different modes of the corresponding tensor.

The weight of each edge in the graph, also referred to as its bond dimension, is equal to the dimension of the appropriate tensor mode.

In accordance with the relation between mode, dimension and index of a tensor presented in section 3.2, each edge in a TN is represented by an index that runs between 1 and its bond dimension.

FIG4 shows three examples: (1) A vector, which is a tensor of order 1, is represented by a node with one leg.

(2) A matrix, which is a tensor of order 2, is represented by a node with two legs.

(3) Accordingly, a tensor of order N is represented in the TN as a node with N legs.

We move on to present the connectivity properties of a TN.

Edges which connect two nodes in the TN represent an operation between the two corresponding tensors.

A index which represents such an edge is called a contracted index, and the operation of contracting that index is in fact a summation over all of the values it can take.

An index representing an edge with one loose end is called an open index.

The tensor represented by the entire TN, whose order is equal to the number of open indices, can be calculated by summing over all of the contracted indices in the network.

An example for a contraction of a simple TN is depicted in FIG4 .

There, a TN corresponding to the operation of multiplying a vector v ??? R r 1 by a matrix M ??? R r 2 ??r 1 is performed by summing over the only contracted index, k. As there is only one open index, d, the result of contracting the network is an order 1 tensor (a vector): u ??? R r 2 which upholds u = M v. Though we use below the contraction of indices in more elaborate TNs, this operation can be essentially viewed as a generalization of matrix multiplication.

The computation of the output at time T that is preformed by the shallow recurrent network given by eqs. 1 and 3, or alternatively by eqs. 6 and 7, can be written in terms of a TN.

FIG5 shows this TN, which given some initial hidden state h0, is essentially a temporal concatenation of a unit cell that preforms a similar computation at every time-step, as depicted in FIG5 .

For any time t < T , this unit cell is composed of the input weights matrix, W I , contracted with the inputs vector, f (x t ), and the hidden weights matrix, W H , contracted with the hidden state vector of the previous time-step, h t???1 .

The final component in each unit cell is the 3 legged triangle representing the order 3 tensor ?? ??? R R??R??R , referred to as the ?? tensor, defined by: DISPLAYFORM0 with ij ??? [R] ???j ??? [3], i.e. its entries are equal to 1 only on the super-diagonal and are zero otherwise.

The use of a triangular node in the TN is intended to remind the reader of the restriction given in eq. 10.

The recursive relation that is defined by the unit cell, is given by the TN in FIG5 (b): DISPLAYFORM1 where kt ??? [R].

In the first equality, we simply follow the TN prescription and write a summation over all of the contracted indices in the left hand side of FIG5 , in the second equality we use the definition of matrix multiplication, and in the last equality we use the definition of the ?? tensor.

The component-wise equality of eq. 11 readily implies h t = (W H h t???1 ) (W I f (x t )), reproducing the recursive relation in eqs. 1 and 3, which defines the operation of the shallow RAC.

From the above treatment, it is evident that the restricted ?? tensor is in fact the component in the TN that yields the element-wise multiplication property.

After T repetitions of the unit cell calculation with the sequential input {x t } T t=1 , a final multiplication of the hidden state vector h T by the output weights matrix W O yields the output vector y T,1,?? .The tensor network which represents the order T shallow RAC weights tensor A

, which appears in eqs. 6 and 7, is given by the TN in the upper part of FIG5 .

In FIG5 , we show that by a simple contraction of indices, the TN representing the shallow RAC weights tensor A T,1,?? c can be drawn in the form of a standard MPS TN.

This TN allows the representation of an order T tensor with a linear (in T ) amount of parameters, rather than the regular exponential amount (A has M T entries).

The decomposition which corresponds to this TN is known as the Tensor Train (TT) decomposition of rank R in the tensor analysis community, its explicit form given in eq. 7.The presentation of the shallow recurrent network in terms of a TN allows the employment of the min-cut analysis, which was introduced by Levine et al. (2017) in the context of convolutional networks, for quantification of the information flow across time modeled by the shallow recurrent network.

This was indeed preformed in our proof of the shallow case of theorem 1.

We now move on to present the computation preformed by a deep recurrent network in the language of TNs.

The construction of a TN which matches the calculation of a deep recurrent network is far less trivial than that of the shallow case, due to the seemingly innocent property of reusing information which lies at the heart of the calculation of deep recurrent networks.

Specifically, all of the hidden states of the network are reused, since the state of each layer at every time-step is duplicated and sent as an input to the calculation of the same layer in the next time-step, and also as an input to the next layer up in the same time-step (see fig. 1(b) ).

The required operation of duplicating a vector and sending it to be part of two different calculations, which is simply achieved in any practical setting, is actually impossible to represent in the framework of TNs.

We formulate this notion in the following claim:Claim 3.

Let v ??? R P , P ??? N be a vector.

v is represented by a node with one leg in the TN notation.

The operation of duplicating this node, i.e. forming two separate nodes of degree 1, each equal to v, cannot be achieved by any TN.Proof.

We assume by contradiction that there exists a Tensor Network ?? which operates on any vector v ??? R P and clones it to two separate nodes of degree 1, each equal to v, to form an overall TN representing v ??? v. Component wise, this implies that ?? upholds ???v ??? R P : DISPLAYFORM0 , meaning that ????? ??? [P ]: DISPLAYFORM1 By definition of the standard basis elements, the left hand side of eq. 12 takes the form ?? ??jk while the right hand side equals 1 only if j = k = ??, and otherwise 0.

Utilizing the ??-tensor notation presented in eq. 10, in order to successfully clone the standard basis elements, eq. 12 implies that ?? must uphold ?? ??jk = ?? ??jk .

However, for v = 1, i.e. ???j ??? [P ] : vj = 1, a cloning operation does not take place when using this value of ??, since DISPLAYFORM2 Claim 3 seems to pose a hurdle in our pursuit of a TN representing a deep recurrent network.

Nonetheless, a form of such a TN may be attained by a simple 'trick' -in order to model the duplication that is inherently present in the deep recurrent network computation, we resort to duplicating the input data itself.

By this technique, for every duplication that takes place along the calculation, the input is inserted into the TN multiple times, once for each sequence that leads to the duplication point.

This principle, which allows us to circumvent the restriction imposed by claim 3, yields the elaborate TN construction of deep RACs depicted in FIG6 .It is important to note that these TNs, which grow exponentially in size as the depth L of the recurrent network represented by them increases, are merely a theoretical tool for analysis and not a suggested implementation scheme for deep recurrent networks.

The actual deep recurrent network is constructed according to the simple scheme given in fig. 1(b) , which grows only linearly in size as the depth L increases, despite the corresponding TN growing exponentially.

In fact, this exponential 'blow-up' in the size of the TNs representing the deep recurrent networks is closely related to their ability to model more intricate correlations over longer periods of time in comparison with their shallower counterparts, which was established in section 4.

FIG6 , a depth L = 2 recurrent network is presented, spread out in time onto T = 4 time-steps.

To understand the logic underlying the input duplication process, which in turn entails duplication of entire segments of the TN, we focus on the calculation of the hidden state vector h 2,2 that is presented in FIG6 .

When the first inputs vector, f (x 1 ), is inserted into the network, it is multiplied by W I,1 and the outcome is equal to h 1,1 .

4 Next, h 1,1 is used in two different places, as an inputs vector to layer L = 2 at time t = 1, and as a hidden state vector in layer L = 1 for time t = 2 calculation.

Our input duplication technique inserts f (x 1 ) into the network twice, so that the same exact h 1,1 is achieved twice in the TN, as marked by the red dotted line in FIG6 .

This way, every copy of h 1,1 4 In this figure, the initial condition for each layer l ??? L, h l,0 , is chosen such that a vector of ones will be present in the initial element-wise multiplication: goes to the appropriate segment of the calculation, and indeed the TN in FIG6 holds the correct value of h 2,2 : DISPLAYFORM3 DISPLAYFORM4 The extension to deeper layers leads us to a fractal structure of the TNs, involving many self similarities, as in the L = 3 example given in FIG6 .

The duplication of intermediate hidden states, marked in red and blue in this example, is the source of the apparent complexity of this L = 3 RAC TN.

Generalizing the above L = 1, 2, 3 examples, a TN representing an RAC of general depth L and of T time-steps, would involve in its structure T duplications of TNs representing RACs of depth L ??? 1, each of which has a distinct length in time-steps i, where i ??? [T ].

This fractal structure leads to an increasing with depth complexity of the TN representing the depth L RAC computation, which we show in the next subsection to motivate the combinatorial lower bound on the Start-End separation rank of deep RACs, given in conjecture 1.

The above presented construction of TNs which correspond to deep RACs, allows us to further investigate the effect of network depth on its ability to model long-term temporal dependencies.

We present below a formal motivation for the lower bound on the Start-End separation rank of deep recurrent networks, given in conjecture 1.

Though our analysis employs TNs visualizations, it is formal nonetheless -these graphs represent the computation in a well-defined manner (see sections A.1-A.3).Our conjecture relies on the fact that it is sufficient to find a specific instance of the network parameters ????h DISPLAYFORM0 ) S,E achieves a certain rank, in order for this rank to be a lower bound on the StartEnd separation rank of the network.

This follows from combining claim 2 and lemma 1.

Claim 2 assures us that the Start-End separation rank of the function realized by an RAC of any depth L, is lower bounded by the rank of the matrix obtained by the corresponding grid tensor matricization: DISPLAYFORM1 ??? for all of the values of parameters ?? ?? h 0,l but a set of Lebesgue measure zero, in order to establish the lower bound in conjecture 1.

Next, we rely on lemma 1, which states that since the entries of A(y T,L,?? c ) are polynomials in the deep recurrent network's weights, it suffices to find a single example for which the rank of the matricized grid tensor is greater than the desired lower bound.

Finding such an example would indeed imply that for almost all of the values of the network parameters, the desired inequality holds.

In the following, we choose a weight assignment that effectively 'separates' between the first layer and higher layers, in the sense that W I,2 is of rank-1.

This is done in similar spirit to the assignment used in the proof of theorem 1, in which W I,2 ij ??? ??i1 (see section B.3).

Under this simplifying assignment, which suffices for our purposes according to the above discussion, the entire computation performed in deeper layers contributes only a constant factor to the matricized grid tensor.

In this case, the example of the TN corresponding to an RAC of depth L = 3 after T = 6 time-steps, which is shown in full in FIG3 , takes the form shown in the upper half of FIG9 .

Next, in order to evaluate rank A(y T,L,?? c ) S,E , we note that graph segments which involve only indices from the "Start" set, will not affect the rank of the matrix under mild conditions on W I,1 , W H,1 .

Specifically, under the Start-End matricization these segments will amount to a different constant multiplying each row of the matrix.

For the example of the RAC of depth L = 3 after T = 6 time-steps, this amounts to the effective TN given in the bottom left side of FIG9 .

Finally, the dependence of this TN on the indices of timesteps { T /2 + 2, . . . , T }, namely those outside of the basic unit involving indices of time-steps {1, . . .

, T /2 + 1}, may only increase the resulting Start-End matricization rank.

6 Thus, we are left with an effective TN resembling the one shown in section 4.2, where the basic unit separating "Start" and "End" indices is raised to the power of the number of its repetitions in the graph.

In the following, we prove a claim according to which the number of repetitions of this basic unit in the TN graph increases exponentially with the depth of the RAC:Claim 4.

Let ??(T, L, R) be the TN representing the computation performed after T time-steps by an RAC with L layers and R hidden channels per layer.

Then, the number of occurrences in layer L = 1 of the basic unit connecting "Start" and "End" indices (bottom right in FIG9 ), is exactly DISPLAYFORM0 be the function computing the output after T time-steps of an RAC with L layers, R hidden channels per layer and weights denoted by ??. In order to focus on repetitions in layer L = 1, we assign W I,2 ij ??? ??i1 for which the following upholds 7 :A(y DISPLAYFORM1 , where the constant term in the first line is the contribution of the deeper layers under this assignment, and the tensor V d 1 ...d T/2 , which becomes a vector under the Start-End matricization, reflects the contribution of the "Start" set indices.

Observing the argument of the chain of products in the above expression, DISPLAYFORM2 r j r j+1 , it is an order t2 tensor, exactly given by the TN representing the computation of a depth L = 1 RAC after t2 time-steps.

Specifically, for t2 = T /2 + 1, it is exactly equal to the basic TN unit connecting "Start" and "End" indices, and for T /2 + 1 < t2 ??? T it contains this basic unit.

This means that in order to obtain the number of repetition of this basic unit in ??, we must count the number of 5 For example, this holds if W I,1 is fully ranked and does not have vanishing elements, and W H,1 = I. 6 This is not true for any TN of this shape but holds due to the temporal invariance of the recurrent network's weights.7 See a similar and more detailed derivation in section B.3.multiplications implemented by the chain of products in the above expression.

Indeed this number is equal to: DISPLAYFORM3 Finally, the form of the lower bound presented in conjecture 1 is obtained by considering a rank R matrix, such as the one obtained by the Start-End matricization of the TN basic unit discussed above, raised to the Hadamard (1) , . . .

, x (M ) ??? X , we define the DISPLAYFORM4 , for which it holds that: Otherwise, assume that DISPLAYFORM5 DISPLAYFORM6 be the functions of the respective decomposition to a sum of separable functions, i.e. that the following holds: DISPLAYFORM7 Then, by definition of the grid tensor, for any template vectors x (1) , . . .

, x (M ) ??? X the following equality holds: are column and row vectors, respectively, which we denote by v?? and u T ?? .

It follows that the matricization of the grid tensor is given by: DISPLAYFORM8 DISPLAYFORM9 In this sub-section, we follow the proof strategy that is outlined in section 4, and prove theorem 1, which shows an exponential advantage of deep recurrent networks over shallow ones in the ability to model long-term dependencies, as measured by the Start-End separation rank (see section 3.1).

In sections B.3.1 and B.3.2, we prove the bounds on the Start-End separation rank of the shallow and deep RACs, respectively, while more technical lemmas which are employed during the proof are relegated to section B.3.3.

We consider the Tensor Network construction of the calculation carried out by a shallow RAC, given in FIG5 .

According to the presented construction, the shallow RAC weights tensor (eqs. 6 and 7) is represented by a Matrix Product State (MPS) Tensor Network (Or??s, 2014) , with the following order-3 tensor building block: DISPLAYFORM0 , where dt ??? [M ] is the input index and kt???1, kt ??? [R] are the internal indices (see FIG5 ).

In TN terms, this means that the bond dimension of this MPS is equal to R. We apply the result of Levine et al. (2017) , who state that the rank of the matrix obtained by matricizing any tensor according to a partition (S, E) is equal to a min-cut separating S from E in the Tensor Network graph representing this tensor, for all of the values of the TN parameters but a set of Lebesgue measure zero.

In this MPS Tensor Network, the minimal cut w.r.t.

the partition (S, E) is equal to the bond dimension R, unless R > M T/2 , in which case the minimal cut contains the external legs instead.

Thus, in the TN representing A

For a deep network, claim 2 assures us that the Start-End separation rank of the function realized by a depth L = 2 RAC is lower bounded by the rank of the matrix obtained by the corresponding grid tensor matricization, for any choice of template vectors.

Specifically: DISPLAYFORM0 Thus, since it trivially holds that rank A(y DISPLAYFORM1 T/2 (the rank is smaller than the dimension of the matrix), proving that rank A(y DISPLAYFORM2 for all of the values of parameters ????h 0,l but a set of Lebesgue measure zero, would satisfy the theorem.

In the following, we provide an assignment of weight matrices and initial hidden states for which rank A(y DISPLAYFORM3 .

In accordance with claim 5, this will suffice as such an assignment implies this rank is achieved for all configurations of the recurrent network weights but a set of Lebesgue measure zero.

We begin by choosing a specific set of template vectors x (1) , . . .

, x (M ) ??? X .

Let F ??? R M ??M be a matrix with entries defined by Fij ??? fj(x (i) ).

According to Cohen and Shashua (2016) , since {f d } M d=1 are linearly independent, then there is a choice of template vectors for which F is non-singular.

Next, we describe our assignment.

In the expressions below we use the notation ??ij = 1 i = j 0 i = j .

Let z ??? R \ {0} be an arbitrary non-zero real number, let ??? ??? R+ be an arbitrary positive real number, and let Z ??? R R??M be a matrix with entries Zij ??? z DISPLAYFORM4 We set W I,1 ??? Z ?? (F T ) ???1 and set W I,2 such that its entries are W I,2 DISPLAYFORM5 to the identity matrix, and additionally we set the entries of DISPLAYFORM6 Finally, we choose the initial hidden state values so they bear no effect on the calculation, namely h 0,l = W H,l ???1 1 = 1 for l = 1, 2.Under the above assignment, the output for the corresponding class c after T time-steps is equal to: DISPLAYFORM7 When evaluating the grid tensor for our chosen set of template vectors, i.e. A(y DISPLAYFORM8 , we can substitute fj(x (i) ) ??? Fij, and thus DISPLAYFORM9 Since we defined Z such that for r ??? min{R, M } Z rd = 0, and denotingR ??? min{R, M } for brevity of notation, the grid tensor takes the following form: DISPLAYFORM10 where we split the product into two expressions, the left part that contains only the indices in the start set S, i.e. d1, . . .

, dT /2 , and the right part which contains all external indices (in the start set S and the end set E).

Thus, under matricization w.r.t.

the Start-End partition, the left part is mapped to a vector a ??? T/2 t=1 R r=1 t j=1 Z rd j S,E containing only non-zero entries per the definition of Z, and the right part is mapped to a matrix B ??? DISPLAYFORM11 , where each entry of u multiplies the corresponding row of B. This results in: DISPLAYFORM12 Since a contains only non-zero entries, diag(a) is of full rank, and so rank A(y DISPLAYFORM13 .

For brevity of notation, we define N ??? DISPLAYFORM14 To prove the above, it is sufficient to show that B can be written as a sum of N rank-1 matrices, i.e. B = DISPLAYFORM15 are two sets of linearly independent vectors.

Indeed, applying claim 6 on the entries of B, specified w.r.t.

the row (d1, . . .

, dT /2 ) and column (dT /2+1 , . . . , dT ), yields the following form: DISPLAYFORM16 where for all k, p (k) isR-dimensional vector of non-negative integer numbers which sum to k, and we explicitly define states R , T /2 and trajectory p (T/2) in claim 6, providing a softer more intuitive definition ) is the accumulated reward of the optimal strategy of emptying the bucket.

In lemma 3 we prove that there exists a value of ??? such that for every sequence of colors d, i.e. a row ofV , the maximal reward over all possible initial states is solely attained at the state q for all values of z but a finite set, we know there exists a value of z for which rank (B) = N , and the theorem follows.

In this section we prove a series of useful technical lemmas, that we have employed in our proof for the case of deep RACs, as described in section B.3.2.

We begin by quoting a claim regarding the prevalence of the maximal matrix rank for matrices whose entries are polynomial functions: Claim 5.

Let M, N, K ??? N, 1 ??? r ??? min{M, N } and a polynomial mapping A : R K ??? R M ??N , i.e. for every i ??? [M ] and j ??? [N ] it holds that Aij : R K ??? R is a polynomial function.

If there exists a point x ??? R K s.t.

rank(A(x))

??? r, then the set {x ??? R K : rank(A(x)) < r} has zero measure (w.r.t.

the Lebesgue measure over R K ).

Claim 5 implies that it suffices to show a specific assignment of the recurrent network weights for which the corresponding grid tensor matricization achieves a certain rank, in order to show this is a lower bound on its rank for all configurations of the network weights but a set of Lebesgue measure zero.

Essentially, this means that it is enough to provide a specific assignment that achieves the required bound in theorem 1 in order to prove the theorem.

Next, we show that for a matrix with entries that are polynomials in x, if a single contributor to the determinant has the highest degree of x, then the matrix is fully ranked for all values of x but a finite set: Lemma 1.

Let A ??? R N ??N be a matrix whose entries are polynomials in x ??? R. In this case, its determinant may be written as det(A) = ?????S N sgn(??)p??(x), where SN is the symmetric group on N elements and p??(x) are polynomials defined by p??(x) ??? N i=1 A i??(i) (x), ????? ??? Sn.

Additionally, let there exist?? such that deg(p??(x)) > deg(p??(x)) ????? =??.

Then, for all values of x but a finite set, A is fully ranked.

Proof.

We show that in this case det(A), which is a polynomial in x by its definition, is not the zero polynomial.

Accordingly, det(A) = 0 for all values of x but a finite set.

Denoting t ??? deg(p??(x)), since t > deg(p??(x)) ????? =??, a monomial of the form c ?? x t , c ??? R \ {0} exists in p??(x) and doesn't exist in any p??(x), ?? =??.

This implies that det(A) is not the zero polynomial, since its leading term has a non-vanishing coefficient sgn(??) ?? c = 0, and the lemma follows from the basic identity: det(A) = 0 ?????? A is fully ranked.

The above lemma assisted us in confirming that the assignment provided for the recurrent network weights indeed achieves the required grid tensor matricization rank of R T /2 .

The following lemma, establishes a useful relation we refer to as the vector rearrangement inequality: DISPLAYFORM0 Proof.

We rely on theorem 368 in Hardy et al. (1952) , which implies that for a set of non-negative numbers {a (1) , . . .

, a (N ) } the following holds for all ?? ??? SN : DISPLAYFORM1 with equality obtained only for ?? which upholds ??(i) = j ?????? a (i) = a (j) .

The above relation, referred to as the rearrangement inequality, holds separately for each component j ??? [R] of the given vectors: DISPLAYFORM2 We now prove that for all ?? ??? SN such that ?? = IN , ????? ??? [R] for which the above inequality is hard, i.e.: DISPLAYFORM3 By contradiction, assume that ????? = IN for which ???j ??? [R]: DISPLAYFORM4 From the conditions of achieving equality in the rearrangement inequality defined in eq. 14, it holds that ???j ??? being a set of N different vectors in RR.

Finally, the hard inequality of the lemma for ?? = IN is implied from eq. 15: DISPLAYFORM5 The vector rearrangement inequality in lemma 2, helped us ensure that our matrix of interest denoted?? upholds the conditions of lemma 1 and is thus fully ranked.

Below, we show an identity that allowed us to make combinatoric sense of a convoluted expression:which is strictly less than the respective contribution in ?? * .

@highlight

We propose a measure of long-term memory and prove that deep recurrent networks are much better fit to model long-term temporal dependencies than shallow ones.