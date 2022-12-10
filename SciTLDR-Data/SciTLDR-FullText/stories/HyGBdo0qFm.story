Alternatives to recurrent neural networks, in particular, architectures based on attention or convolutions, have been gaining momentum for processing input sequences.

In spite of their relevance, the computational properties of these alternatives have not yet been fully explored.

We study the computational power of two of the most paradigmatic architectures exemplifying these mechanisms: the Transformer (Vaswani et al., 2017) and the Neural GPU (Kaiser & Sutskever, 2016).

We show both models to be Turing complete exclusively based on their capacity to compute and access internal dense representations of the data.

In particular, neither the Transformer nor the Neural GPU requires access to an external memory to become Turing complete.

Our study also reveals some minimal sets of elements needed to obtain these completeness results.

There is an increasing interest in designing neural network architectures capable of learning algorithms from examples BID6 BID7 BID10 BID11 BID13 BID4 .

A key requirement for any such an architecture is thus to have the capacity of implementing arbitrary algorithms, that is, to be Turing complete.

Turing completeness often follows for these networks as they can be seen as a control unit with access to an unbounded memory; as such, they are capable of simulating any Turing machine.

On the other hand, the work by Siegelmann & Sontag (1995) has established a different way of looking at the Turing completeness of neural networks.

In particular, their work establishes that recurrent neural networks (RNNs) are Turing complete even if only a bounded number of resources (i.e., neurons and weights) is allowed.

This is based on two conditions: (1) the ability of RNNs to compute internal dense representations of the data, and (2) the mechanisms they use for accessing such representations.

Hence, the view proposed by Siegelmann & Sontag shows that it is possible to release the full computational power of RNNs without arbitrarily increasing its model complexity.

Most of the early neural architectures proposed for learning algorithms correspond to extensions of RNNs -e.g., Neural Turing Machines BID6 ) -, and hence they are Turing complete in the sense of Siegelmann & Sontag.

However, a recent trend has shown the benefits of designing networks that manipulate sequences but do not directly apply a recurrence to sequentially process their input symbols.

Architectures based on attention or convolutions are two prominent examples of this approach.

In this work we look at the problem of Turing completenessà la Siegelmann & Sontag for two of the most paradigmatic models exemplifying these features: the Transformer (Vaswani et al., 2017) and the Neural GPU BID11 .The main contribution of our paper is to show that the Transformer and the Neural GPU are Turing complete based on their capacity to compute and access internal dense representations of the data.

In particular, neither the Transformer nor the Neural GPU requires access to an external additional memory to become Turing complete.

Thus the completeness holds for bounded architectures (bounded number of neurons and parameters).

To prove this we assume that internal activations are represented as rational numbers with arbitrary precision.

For the case of the Transformer we provide a direct simulation of a Turing machine, while for the case of the Neural GPU our result follows by simulating standard sequence-to-sequence RNNs.

Our study also reveals some minimal sets of elements needed to obtain these completeness results.

The computational power of Transformers and of Neural GPUs has been compared in the current literature BID4 , but both are only informally used.

Our paper provides a formal way of approaching this comparison.

For the sake of space, we only include sketch of some proofs in the body of the paper.

The details for every proof can be found in the appendix.

Background work The study of the computational power of neural networks can be traced back to BID14 which established an analogy between neurons with hard-threshold activations and first order logic sentences, and BID12 that draw a connection between neural networks and finite automata.

As mentioned earlier, the first work showing the Turing completeness of finite neural networks with linear connections was carried out by BID18 1995) .

Since being Turing complete does not ensure the ability to actually learn algorithms in practice, there has been an increasing interest in enhancing RNNs with mechanisms for supporting this task.

One strategy has been the addition of inductive biases in the form of external memory, being the Neural Turing Machine (NTM) BID6 ) a paradigmatic example.

To ensure that NTMs are differentiable, their memory is accessed via a soft attention mechanism .

Other examples of architectures that extend RNNs with memory are the Stack-RNN BID10 , and the (De)Queue-RNNs BID7 .

By Siegelmann & Sontag's results, all these architectures are Turing complete.

The Transformer architecture (Vaswani et al., 2017) is almost exclusively based on the attention mechanism, and it has achieved state of the art results on many language-processing tasks.

While not initially designed to learn general algorithms, BID4 have advocated the need for enriching its architecture with several new features as a way to learn general procedures in practice.

This enrichment is motivated by the empirical observation that the original Transformer architecture struggles to generalize to input of lengths not seen during training.

We, in contrast, show that the original Transformer architecture is Turing complete, based on different considerations.

These results do not contradict each other, but show the differences that may arise between theory and practice.

For instance, BID4 assume fixed precision, while we allow arbitrary internal precision during computation.

We think that both approaches can be complementary as our theoretical results can shed light on what are the intricacies of the original architecture, which aspects of it are candidates for change or improvement, and which others are strictly needed.

For instance, our proof uses hard attention while the Transformer is often trained with soft attention (Vaswani et al., 2017) .

See Section 3.3 for a discussion on these differences.

The Neural GPU is an architecture that mixes convolutions and gated recurrences over tridimensional tensors.

It has been shown that NeuralGPUs are powerful enough to learn decimal multiplication from examples BID5 , being the first neural architecture capable of solving this problem end-to-end.

The similarity of Neural GPUs and cellular automata has been used as an argument to state the Turing completeness of the architecture BID11 BID16 .

Cellular automata are Turing complete (Smith III, 1971; BID15 and their completeness is established assuming an unbounded number of cells.

In the Neural GPU architecture, in contrast, the number of cells that can be used during a computation is proportional to the size of the input sequence BID11 .

One can cope with the need for more cells by padding the Neural GPU input with additional (dummy) symbols, as much as needed for a particular computation.

Nevertheless, this is only a partial solution, as for a Turing-complete model of computation, one cannot decide a priori how much memory is needed to solve a particular problem.

Our results in this paper are somehow orthogonal to the previous argument; we show that one can leverage the dense representations of the Neural GPU cells to obtain Turing completeness without requiring to add cells beyond the ones used to store the input.

We assume all weights and activations to be rational numbers of arbitrary precision.

Moreover, we only allow the use of rational functions with rational coefficients.

Most of our positive results make use of the piecewise-linear sigmoidal activation function σ : Q → Q, which is defined as σ(x) = 0 x < 0, x 0 ≤ x ≤ 1, 1 x > 1.(1)We are mostly interested in sequence-to-sequence (seq-to-seq) neural network architectures that we next formalize.

A seq-to-seq network N receives as input a sequence X = (x 1 , . . .

, x n ) of vectors x i ∈ Q d , for some d > 0, and produces as output a sequence Y = (y 1 , . . .

, y m ) of vectors DISPLAYFORM0 Most of these types of architectures require a seed vector s and some stopping criterion for determining the length of the output.

The latter is usually based on the generation of a particular output vector called an end of sequence mark.

In our formalization instead, we allow a network to produce a fixed number r ≥ 0 of output vectors.

Thus, for convenience we see a general seq-toseq network as a function N such that the value N (X, s, r) corresponds to an output sequence of the form Y = (y 1 , y 2 , . . .

, y r ).

With this definition, we can view every seq-to-seq network as a language recognizer of strings as follows.

Definition 2.1.

A seq-to-seq language recognizer is a tuple A = (Σ, f, N, s, F), where Σ is a finite alphabet, f : Σ → Q d is an embedding function, N is a seq-to-seq network, s ∈ Q d is a seed vector, and F ⊆ Q d is a set of final vectors.

We say that A accepts the string w ∈ Σ * , if there exists an integer r ∈ N such that N (f (w), s, r) = (y 1 , . . .

, y r ) and y r ∈ F. The language accepted by A, denoted by L(A), is the set of all strings accepted by A.We impose two additional restrictions over recognizers.

The embedding function f : Σ → Q d should be computed by a Turing machine in time linear w.r.t.

the size of Σ. This covers the two most typical ways of computing input embeddings from symbols: the one-hot encoding, and embeddings computed by fixed feed-forward networks.

Moreover, the set F should also be recognizable in lineartime; given a vector f , the membership f ∈ F should be decided by a Turing machine working in linear time with respect to the size (in bits) of f .

This covers the usual way of checking equality with a fixed end-of-sequence vector.

We impose these restrictions to disallow the possibility of cheating by encoding arbitrary computations in the input embedding or the stop condition, while being permissive enough to construct meaningful embeddings and stoping criterions.

Finally, a class N of seq-to-seq neural network architectures defines the class L N composed of all the languages accepted by language recognizers that use networks in N .

From these notions, the formalization of Turing completeness of a class N naturally follows.

Given an input sequence X = (x 1 , . . .

, x n ), a seed vector y 0 , and r ∈ N, an encoder-decoder RNN is given by the following two recursions DISPLAYFORM1 where V , W , U , R are matrices, b 1 and b 2 are vectors, O(·) is an output function, and f 1 and f 2 are activations functions.

Equation (2) is called the RNN-encoder and (3) the RNN-decoder.

The next Theorem follows by inspection of the proof by BID18 1995) after adapting it to our formalization of encoder-decoder RNNs.

BID18 1995) ).

The class of encoder-decoder RNNs is Turing complete.

Turing completeness holds even if we restrict to the class in which R is the zero matrix, b 1 and b 2 are the zero vector, O(·) is the identity function, and f 1 and f 2 are the piecewise-linear sigmoidal activation σ.

In this section we present a formalization of the Transformer architecture (Vaswani et al., 2017) , abstracting away from specific choices of functions and parameters.

Our formalization is not meant to produce an efficient implementation of the Transformer, but to provide a simple setting over which its mathematical properties can be established in a formal way.

The Transformer is heavily based on the attention mechanism introduced next.

Consider a scoring function score : DISPLAYFORM0 (5) Usually, q is called the query, K the keys, and V the values.

We do not pose any restriction on the scoring and normalization functions, as some of our results hold in general.

We only require the normalization function to satisfy that there is a function f ρ from Q to Q + such that for each x = (x 1 , . . .

, x n ) ∈ Q n it is the case that the i-th DISPLAYFORM1 .

Thus, a in Equation (5) is a convex combination of the vectors in V .

When proving possibility results, we will need to pick specific scoring and normalization functions.

A usual choice for the scoring function is a feed forward network with input (q, k i ) sometimes called additive attention .

Another possibility is to use the dot product q, k i called multiplicative attention (Vaswani et al., 2017) .

We use a combination of both: multiplicative attention plus a non linear function.

For the normalization function, softmax is a standard choice.

Nevertheless, in our proofs we use the hardmax function, which is obtained by setting f hardmax (x i ) = 1 if x i is the maximum value, and f hardmax (x i ) = 0 otherwise.

Thus, for a vector x in which the maximum value occurs r times, we have that hardmax i (x) = 1 r if x i is the maximum value of x, and hardmax i (x) = 0 otherwise.

We call it hard attention whenever hardmax is used as normalization function.

As customary, for a function F : DISPLAYFORM2 Transformer Encoder and Decoder A single-layer encoder of the Transformer is a parametric function Enc(X; θ) receiving a sequence X = (x 1 , . . . , x n ) of vectors in Q d and returning a sequence Z = (z 1 , . . .

, z n ) of the same length of vectors in Q d .

In general, we consider the parameters in Enc(X; θ) as functions Q(·), K(·), V (·), and O(·), all of them from Q d to Q d .

The single-layer encoder is then defined as follows DISPLAYFORM3 In practice Q(·), K(·), V (·) are typically matrix multiplications, and O(·) a feed-forward network.

The + x i and + a i summands are usually called residual connections BID8 b) .

When the particular functions used as parameters are not important, we simply write Z = Enc(X).The Transformer encoder is defined simply as the repeated application of single-layer encoders (with independent parameters), plus two final transformation functions K(·) and V (·) applied to every vector in the output sequence of the final layer.

Thus the L-layer Transformer encoder is defined by the following recursion (with 1 ≤ ≤ L − 1 and X 1 = X).

DISPLAYFORM4 to denote an L-layer Transformer encoder over the sequence X.A single-layer decoder is similar to a single-layer encoder but with additional attention to an external pair of key-value vectors (K e , V e ).

The input for the single-layer decoder is a sequence Y = (y 1 , . . .

, y k ) plus the external pair (K e , V e ), and the output is a sequence Z = (z 1 , . . . , z k ).

When defining a decoder layer we denote by Y j the sequence (y 1 , . . .

, y j ), for 1 ≤ j ≤ k. The layer is also parameterized by four functions Q(·), K(·), V (·) and O(·) and is defined as follows.

DISPLAYFORM5 considers the subsequence of Y only until index i and is used to generate a query p i to attend the external pair (K e , V e ).

We denote the single-decoder layer by Dec((K e , V e ), Y ; θ).The Transformer decoder is a repeated application of single-layer decoders, plus a transformation function F : Q d → Q d applied to the final vector of the decoded sequence.

Thus, the output of the decoder is a single vector z ∈ Q d .

Formally, the L-layer Transformer decoder is defined as DISPLAYFORM6 We use z = TDec L ((K e , V e ), Y ) to denote an L-layer Transformer decoder.

The complete Tansformer A Transformer network receives an input sequence X, a seed vector y 0 , and a value r ∈ N. Its output is a sequence Y = (y 1 , . . .

, y r ) defined as y t+1 = TDec(TEnc(X), (y 0 , y 1 , . . .

, y t )), for 0 ≤ t ≤ r − 1.(13) We denote the output sequence of the transformer as Y = (y 1 , y 2 , . . .

, y r ) = Trans(X, y 0 , r).

The Transformer, as defined above, is order-invariant: two input sequences that are permutations of each other produce exactly the same output.

This is a consequence of the following property of the attention function: if K = (k 1 , . . .

, k n ), V = (v 1 , . . . , v n ), and π : {1, . . . , n} → {1, . . . , n} is a permutation, then Att(q, K, V ) = Att(q, π(K), π(V )) for every query q. This weakness has motivated the need for including information about the order of the input sequence by other means; in particular, this is often achieved by using the so-called positional encodings (Vaswani et al., 2017; BID17 , which we study below.

But before going into positional encodings, a natural question is what languages the Transformer can recognize without them.

As a standard yardstick we use the well-studied class of regular languages, i.e., languages recognized by finite automata.

Order-invariance implies that not every regular language can be recognized by a Transformer network.

As an example, there is no Transformer network that can recognize the regular language (ab) * , as the latter is not order-invariant.

A reasonable question then is whether the Transformer can express all regular languages which are order-invariant.

It is possible to show that this is not the case by proving that the Transformer actually satisfies a stronger invariance property, which we call proportion invariance.

For a string w ∈ Σ * and a symbol a ∈ Σ, we use prop(a, w) to denote the ratio between the number of times that a appears in w and the length of w. Consider now the set PropInv(w) = {u ∈ Σ * | prop(a, w) = prop(a, u) for every a ∈ Σ}. Proposition 3.1.

Let Trans be a Transformer, s a seed, r ∈ N, and f : Σ → Q d an embedding function.

Then Trans(f (w), s, r) = Trans(f (u), s, r), for each u, w ∈ Σ * with u ∈ PropInv(w).As an immediate corollary we obtain the following.

Corollary 3.2.

Consider the order-invariant regular language L = {w ∈ {a, b} * | w has an even number of a symbols}. Then L cannot be recognized by a Transformer network.

On the other hand, languages recognized by Transformer networks are not necessarily regular.

Proposition 3.3.

There is a Transformer network that recognizes the non-regular language S = {w ∈ {a, b} * | w has strictly more symbols a than symbols b}.That is, the computational power of Transformer networks without positional encoding is both rather weak (they do not even contain order-invariant regular languages) and not so easy to capture (as they can express counting properties that go beyond regularity).

As we show in the next section, the inclusion of positional encodings radically changes the picture.

Thus, given an input string w = a 1 a 2 · · · a n ∈ Σ, the result of the embedding function f pos (w) provides a "new" input f pos (a 1 , 1), f pos (a 2 , 2), . . .

, f pos (a n , n) to the Transformer encoder.

Similarly, the Transformer decoder instead of receiving the sequence Y = (y 0 , y 1 , . . .

, y t ) as input, it receives now the sequence Y = y 0 + pos(1), y 1 + pos(2), . . .

, y t + pos(t + 1)As for the case of the embedding functions, we require the positional encoding pos(i) to be computable by a Turing machine working in linear time w.r.t.

the size (in bits) of i.

The main result of this section is the completeness of Transformers with positional encodings.

DISPLAYFORM0 .

DISPLAYFORM1 . . .

DISPLAYFORM2 attends to the encoder and copies the corresponding symbol uses self attention to compute next state and the next symbol under M 's head Proof Sketch.

We show that for every Turing machine M = (Q, Σ, δ, q init , F ) there exists a transformer that simulates the complete execution of M .

We represent a string w = s 1 s 2 · · · s n ∈ Σ * as a sequence X of one-hot vectors with their corresponding positional encodings.

Denote by q (t) ∈ Q the state of M at time t when processing w, and s (t) ∈ Σ the symbol under M 's head at time t. Similarly, v (t) ∈ Σ is the symbol written by M and m (t) ∈ {←, →} the head direction.

We next describe how to construct a transformer Trans M that with input X produces a sequence y 0 , y 1 , y 2 , . . .

such that y i contains information about q (i) and s (i) (encoded as one-hot vectors).

DISPLAYFORM3 The construction and proof goes by induction.

Assume the decoder receives y 0 , . . .

, y t such that y i contains q (i) and s (i) .

To construct y t+1 , in the first layer we just implement M 's transition function δ; note that δ( (t+1) which is the index to which M is going to be pointing to in the next time step.

By using the residual connections we also store q (i+1) and v (i) in z 2 i .

The final piece of our construction is to compute the symbol that the tape holds at index c (t+1) , that is, the symbol under M 's head at time t + 1.

For this we use the following observation: the symbol at index c (t+1) in time t + 1 coincides with the last symbol written by M at index c (t+1) .

Thus, we need to find the maximum value i ≤ t such that c (i ) = c (t+1) and then copy v (i ) which is the symbol that was written by M at time step i .

This last computation can also be done with a self-attention layer.

Thus, we attend directly to position i (hard attention plus positional encodings) and copy v (i ) which is exactly s (t+1) .

We finally copy q (t+1) and s (t+1) into the output to construct y t+1 .

FIG2 shows a high-level diagram of the decoder computation.

DISPLAYFORM4 There are several other details in the construction, in particular, at the beginning of the computation (first n steps), the decoder needs to attend to the encoder and copy the input symbols so they can later be processed as described above.

Another detail is when M reaches a cell that has not been visited before, then the symbol under the head has to be set as # (the blank symbol).

We show that all these decisions can be implemented with feed-forward networks plus attention.

The complete construction uses one encoder layer, three decoder layers and vectors of dimension d = 2|Q| + 4|Σ| + 11 to store one-hot representations of states, symbols and some additional working space.

All details can be found in the appendix.

Although the general architecture that we presented closely follows that of Vaswani et al. (2017) , some choices for functions and parameters in our positive results are different to the usual choices in practice.

For instance, we use hard attention which allow us to attend directly to specific positions.

In contrast, Vaswani et al. (2017) use softmax to attend, plus sin-cos functions as positional encodings.

The softmax, sin and cos are not rational functions, and thus, are forbidden in our formalization.

An interesting line for future work is to consider arbitrary functions but with additional restrictions, such as finite precision as done by Weiss et al. (2018) .

Another difference is that for the function O(·) in Equation (11) The need of arbitrary precision Our Turing-complete proof relies on having arbitrary precision for internal representations, in particular, for storing and manipulating positional encodings.

Although having arbitrary precision is a standard assumption when studying the expressive power of neural networks BID3 Siegelmann & Sontag (1995) ) practical implementations rely on fixed precision hardware.

If fixed precision is used, then positional encodings can be seen as functions of the form pos : N → A where A is a finite subset of Q d .

Thus, the embedding function f pos can be seen as a regular embedding function f : Σ → Q d where Σ = Σ × A. Thus, whenever fixed precision is used, the net effect of having positional encodings is to just increase the size of the input alphabet.

Then from Proposition 3.1 we obtain that the Transformer with positional encodings and fixed precision is not Turing complete.

Although no longer Turing complete, one can still study the computational power of fixed-precision Transformers.

We left this as future work.

The Neural GPU BID11 is an architecture that mixes convolutions and gated recurrences over tridimensional tensors.

It is parameterized by three functions U (·) (update function), R(·) (reset function), and F (·).

Given a tensor S ∈ Q h×w×d and a value r ∈ N, it produces a sequence S 1 , S 2 , . . .

, S r given by the following recursive definition (with S 0 = S).

DISPLAYFORM0 where denotes the element-wise product, and 1 is a tensor with only 1's.

Neural GPUs force functions U (·) and R(·) to produce a tensor of the same shape as its input with all values in [0, 1].

Thus, a Neural GPU resembles a gated recurrent unit , with U working as the update gate and R as the reset gate.

Functions U (·), R(·), and F (·) are defined as a convolution of its input with a 4-dimensional kernel bank with shape (k H , k W , d, d) plus a bias tensor, followed by a point-wise transformation DISPLAYFORM1 with different kernels and biases for U (·), R(·), and F (·).To have an intuition on how the convolution K * S works, it is illustrative to think of S as an (h × w)-grid of (row) vectors and DISPLAYFORM2 , and K ij = K i,j,:,: , then K * S is a regular two-dimensional convolution in which scalar multiplication has been replaced by vector-matrix multiplication as in the following expression DISPLAYFORM3 where DISPLAYFORM4 This intuition makes evident the similarity between Neural GPUs and cellular automata: S is a grid of cells, and in every iteration each cell is updated considering the values of its neighbors according to a fixed rule given by K BID11 .

As customary, we assume zero-padding when convolving outside S.

To study the computational power of Neural GPUs, we cast them as a standard seq-to-seq architecture.

Given an input sequence, we put every vector in the first column of the tensor S. We also need to pick a special cell of S as the output cell from which we read the output vector in every iteration.

We pick the last cell of the first column of S. Formally, given a sequence X = (x 1 , . . .

, x n ) with x i ∈ Q d , and a fixed value w ∈ N, we construct the tensor S ∈ Q n×w×d by leting S i,1,: = x i and S i,j,: = 0 for j > 1.

The output of the Neural GPU, denoted by NGPU(X, r), is the sequence of vectors Y = (y 1 , y 2 , . . .

, y r ) such that y t = S t n,1,: .

Given this definition, we can naturally view the Neural GPUs as language recognizers (as formalized in Section 2).Since the bias tensor B in Equation FORMULA15 is of the same size than S, the number of parameters in a Neural GPU grows with the size of the input.

Thus, a Neural GPU cannot be considered as a fixed architecture.

To tackle this issue we introduce the notion of uniform Neural GPU, as one in which for every bias B there exists a matrix B ∈ Q w×d such that B i,:,: = B for each i. Thus, uniform Neural GPUs can be finitely specified (as they have a constant number of parameters, not depending on the length of the input).

We now establish the Turing completeness of this model.

Theorem 4.1.

The class of uniform Neural GPUs is Turing complete.

Proof sketch.

The proof is based on simulating a seq-to-seq RNN; thus, completeness follows from Theorem 2.3.

Consider an RNN encoder-decoder language recognizer, such that N is of dimension d and its encoder and decoder are defined by the equations h i = σ(x i W + h i−1 V ) and g t = σ(g t−1 U ), respectively, where g 0 = h n and n is the length of the input.

We use a Neural GPU with input tensor S ∈ Q n×1×3d+3 .

Let E i = S i,1,1:d and D i = S i,1,d+1:2d .

The idea is to use E for the encoder and D for the decoder.

We use kernel banks of shape (2, 1, 3d + 3, 3d + 3) with uniform bias tensors to simulate the following computation.

In every step t, we first compute the value of σ(E t W + E t−1 V ) and store it in E t , and then reset E t−1 to zero.

Similarly, in step t we update the vector in position D t−1 storing in it the value σ(D t−1 U + E t−1 U ) (for the value of E t−1 before the reset).

We use the gating mechanism to ensure a sequential update of the cells such that at time t we update only positions E i and D j for i ≤ t and j ≤ t − 1.

Thus the updates on the D are always one iteration behind the update of E. Since the vectors in D are never reset to zero, they keep being updated which allows us to simulate an arbitrary long computation.

In particular we prove that at iteration t it holds that E t = h t , and at iteration n + t it holds that D n = g t .

We require 3d + 3 components, as we need to implement several gadgets for properly using the update and reset gates.

In particular, we need to store the value of E t−1 before we reset it.

The detailed construction and the correctness proof can be found in the appendix.

The proof above makes use of kernels of shape (2, 1, d, d) to obtain Turing completeness.

This is, in a sense, optimal, as one can easily prove that Neural GPUs with kernels of shape (1, 1, d, d) are not Turing complete, regardless of the size of d. In fact, for kernels of this shape the value of a cell of S at time t depends only on the value of the same cell in time t − 1.Zero padding vs circular convolution The proof of Theorem 4.1 requires the application of zero padding in convolution.

This allows us to clearly differentiate internal cells from cells corresponding to the endpoints of the input sequence.

Interestingly, Turing-completeness is lost if we replace zero padding with circular convolution.

Formally, given S ∈ Q h×w×d , a circular convolution is obtained by defining S h+n,:,: = S n,:,: for n ∈ Z. One can prove that uniform Neural GPUs with circular convolutions cannot differentiate among periodic sequences of different length; in particular, they cannot check if a periodic input sequence is of even or odd length.

This yields the following: Proposition 4.2.

Uniform Neural GPUs with circular convolutions are not Turing complete.

Related to this last result is the empirical observation by BID16 that Neural GPUs that learn to solve hard problems, e.g., binary multiplication, and which generalize to most of the inputs, struggle with highly symmetric (and nearly periodic) inputs.

Actually, BID16 exhibit examples of the form 11111111 × 11111111 failing for all inputs with eight or more 1s.

We leave as future work to explore the implications of our theoretical results on this practical observation.

BID5 simplified Neural GPUs and proved that, by considering piecewise linear activations and bidimensional input tensors instead of the original smooth activations and tridimensional tensors used by BID11 , it is possible to achieve substantially better results in terms of training time and generalization.

Our Turing completeness proof also relies on a bidimensional tensor and uses piecewise linear activations, thus providing theoretical evidence that these simplifications actually retain the full expressiveness of Neural GPUs while simplifying its practical applicability.

We have presented an analysis of the Turing completeness of two popular neural architectures for sequence-processing tasks; namely, the Transformer, based on attention, and the Neural GPU, based on recurrent convolutions.

We plan to further refine this analysis in the future.

For example, our proof of Turing completeness for the Transformer requires the presence of residual connections, i.e., the +x i , +a i , +y i , and +p i summands in Equations (6-11), while our proof for Neural GPUs heavily relies on the gating mechanism.

We will study whether these features are actually essential to obtain completeness.

We presented general abstract versions of both architectures in order to prove our theoretical results.

Although we closely follow their original definitions, some choices for functions and parameters in our positive results are different to the usual choices in practice, most notably, the use of hard attention for the case of the Transformer, and the piecewise linear activation functions for both architectures.

As we have mentioned, BID5 showed that for Neural GPUs piecewise linear activations actually help in practice, but for the case of the Transformer architecture more experimentation is needed to have a conclusive response.

This is part of our future work.

Although our results are mostly of theoretical interest, they might lead to observations of practical interest.

For example, BID1 have established the undecidability of several practical problems related to probabilistic language modeling with RNNs.

This means that such problems can only be approached in practice via heuristics solutions.

Many of the results in BID1 are, in fact, a consequence of the Turing completeness of RNNs as established by Siegelmann & Sontag (1995) .

We plan to study to what extent our analogous undecidability results for Transformers and Neural GPUs imply undecidability for language modeling problems based on these architectures.

Finally, our results rely on being able to compute internal representations of arbitrary precision.

It would be interesting to perform a theoretical study of the main properties of both architectures in a setting in which only finite precision is allowed, as have been recently carried out for RNNs (Weiss et al., 2018) .

We also plan to tackle this problem in our future work.

We first sketch the main idea of Siegelmann & Sontag's proof.

We refer the reader to the original paper for details.

Siegelmann & Sontag show how to simulate a two-stack machine M (and subsequently, a Turing machine) with a single RNN N with σ as activation.

They first construct a network N 1 that, with 0 as initial state (h N1 0 = 0) and with a binary string w ∈ {0, 1} * as input sequence, produces a representation of w as a rational number and stores it as one of its internal values.

Their internal representation of strings encodes every w as a rational number between 0 and 1.

In particular, they use base 4 such that, for example, a string w = 100110 is encoded as (0.311331) 4 that is, its encoding is DISPLAYFORM0 This representation allows one to easily simulate stack operations as affine transformations plus σ activations.

For instance, if x w is the value representing string w = b 1 b 2 · · · b n seen as a stack, then the top(w) operation can be defined as simply y = σ(4x w − 2), since y = 1 if and only if b 1 = 1, and y = 0 if and only if b 1 = 0.

Other stack operations can de similarly simulated.

Using this representation, they construct a second network N 2 that simulates the two-stacks machine by using one neuron value to simulate each stack.

The input w for the simulated machine M is assumed to be at an internal value given to N 2 as an initial state (h N2 0 ).

Thus, N 2 expects only zeros as input.

Actually, to make N 2 work for r steps, an input of the form 0 r should be provided.

Finally, they combine N 1 and N 2 to construct a network N which expects an input of the following form: It is clear that Siegelmann & Sontag's proof resembles a modern encoder-decoder RNN architecture, where N 1 is the encoder and N 2 is the decoder, thus it is straightforward to use the same construction to provide an RNN encoder-decoder N and a language recognizer A that uses N and simulates the two-stacks machine M .

There are some details that is important to notice.

Assume that N is given by the formulas in Equations (2) and (3).

First, since N 2 in the above construction expects no input, we can safely assume that R in Equation FORMULA1 is the null matrix.

Moreover, since A defines its own embedding function, we can ensure that every vector that we provide for the encoder part of N has a 1 in a fixed component, and thus we do not need the bias b 1 in Equation (2) since it can be simulated with one row of matrix V .

We can do a similar construction for the bias b 2 (Equation FORMULA1 ).

Finally, Siegelmann & Sontag show that its construction can be modified such that a particular neuron of N 2 , say n , is always 0 except for the first time an accepting state of M is reached, in which case n = 1.

Thus, one can consider O(·) (Equation FORMULA1 ) as the identity function and add to A the stopping criterion that just checks if n is 1.

We extend the definition of the function PropInv to sequences of vectors.

Given a sequence X = (x 1 , . . . , x n ) we use vals(X) to denote the set of all vectors occurring in X. Similarly as for strings, we use prop(v, X) as the number of times that v occurs in X divided by the length of X. Now we are ready to extend PropInv with the following definition: DISPLAYFORM1 PropInv(X) = {X | vals(X ) = vals(X) and prop(v, X) = prop(v, X ) for all v ∈ vals(X)} Notice that for every embedding function f : Σ → Q d and string w ∈ Σ * , we have that if u ∈ PropInv(w) then f (u) ∈ PropInv(f (w)).

Thus in order to prove that Trans(f (w), s, r) = Trans(f (u), s, r) for every u ∈ PropInv(w), it is enough to prove that Trans(X, s, r) = Trans(X , s, r) for every X ∈ PropInv(X)To further simplify the exposition of the proof we introduce another notation.

We denote by p X v as the number of times that vector v occurs in X. Thus we have that X ∈ PropInv(X) if and only if, there exists a value γ ∈ Q + such that for every v ∈ vals(X) it holds that p DISPLAYFORM2 We now have all the necessary to proceed with the proof of Proposition 3.1.

We will prove it by proving the property in (16).

Let X = (x 1 , . . .

, x n ) be an arbitrary sequence of vectors, and let X = (x 1 , . . .

, x m ) ∈ PropInv(X).

Moreover, let Z = (z 1 , . . .

, z n ) = Enc(X; θ) and Z = (z 1 , . . .

, z m ) = Enc(X ; θ).

We first prove the following property:For every pair of indices (i, j) ∈ {1, . . .

, n} × {1, . . .

, m}, if DISPLAYFORM3 Lets (i, j) be a pair of indices such that x i = x j .

From Equations (6-7) we have that DISPLAYFORM4 .

By equations (4-5) and the restriction over the form of normalization functions we have that DISPLAYFORM5 where α = n =1 f ρ (score(Q(x ), K(x ))).

The above equation can be rewritten as DISPLAYFORM6 .

By a similar reasoning we can write DISPLAYFORM7 ).

Now, since X ∈ PropInv(X) we know that vals(X) = vals(X ) and there exists a γ ∈ Q + such that p X v = γp X v for every v ∈ vals(X).

Finally, from this last property, plus the fact that x i = x j we have DISPLAYFORM8 Which completes the proof of Property FORMULA23 above.

Consider now the complete encoder TEnc.

Let (K, V ) = TEnc(X) and (K , V ) = TEnc(X ), and let q be an arbitrary vector.

We will prove now that Att(q, K, V ) = Att(q, K , V ).

By following a similar reasoning as for proving Property (17) (plus induction on the layers of TEnc) we obtain that if x i = x j then k i = k j and v i = v j , for every i ∈ {1, . . .

, n} and j ∈ {1, . . .

, m}. Thus, there exists a mapping DISPLAYFORM9 for every i ∈ {1, . . .

, n} and j ∈ {1, . . .

, m}. Lets focus now on Att(q, K, V ).

We have: DISPLAYFORM10 .

Similarly as before, we can rewrite this as DISPLAYFORM11 And finally using that X ∈ PropInv(X) we obtain DISPLAYFORM12 which is what we wanted.

To complete the rest proof, consider Trans(X, y 0 , r) which is defined by the recursion y k+1 = TDec(TEnc(X), (y 0 , y 1 , . . .

, y k ))To prove that Trans(X, y 0 , r) = Trans(X , y 0 , r) we use an inductive argument.

We know that y 1 = TDec(TEnc(X), (y 0 )) = TDec((K, V ), (y 0 )).

Now TDec only access (K, V ) via attentions of the form Att(q, K, V ) and for the case of y 1 the vector q can only depend on y 0 , thus, from Att(q, K, V ) = Att(q, K , V ) we have that DISPLAYFORM13 ).

The rest of the steps follow by a simple induction on k.

To obtain a contradiction, assume that there is a language recognizer A that uses a Transformer network and such that L = L(A).

Now consider the strings w 1 = aabb and w 2 = aaabbb.

Since w 1 ∈ PropInv(w 2 ) by Proposition 3.1 we have that w 1 ∈ L(A) if and only if w 2 ∈ L(A) which is a contradiction since w 1 ∈ L but w 2 / ∈ L.

This completes the proof of the corollary.

We construct a language recognizer A = (Σ, f, Trans, s, F) with Trans a very simple Transformer network with dimension d = 2 and using just one layer of encoder and one layer of decoder, such that L(A) = {w ∈ {a, b} * | w has strictly more symbols a than symbols b}. As embedding function, we use f (a) = [0, 1] and f (b) = [0, −1].Assume that the output for the encoder part of the transformer is X = (x 1 , . . .

, x n ).

First we use an encoder layer that implements the identity function.

This can be trivially done using null functions for the self attention and through the residual connections this encoder layer shall preserve the original x i values.

For the final V (·) and K(·) functions of the Transformer encoder (Equation (8) For the decoder we use a similar approach.

We consider the identity in the self attention plus the residual (which can be done by just using the null functions for the self attention).

Considering the external attention, that is the attention over (K e , V e ), we let score and ρ be arbitrary scoring and normalization functions.

And finally for the function O(·) (Equation (11) In order to complete the proof we introduce some notation.

Lets denote by # a (w) as the number of a's in w, and similarly # b (w) for the number of b's in w. Lets call c w as the value #a(w)−# b (w) n .

We now prove that, for any string w ∈ {a, b} * if we consider f (w) = X = (x 1 , . . .

, x n ) as the input sequence for Trans and we use initial value s = [0, 0] for the decoder, the complete network shall compute a sequence y 1 , y 2 , . . .

, y r such that: DISPLAYFORM0 We proceed by induction.

The base case trivially holds since y 0 = s = [0, 0].

Assume now that we are at step r and the input for the decoder is (y 0 , y 1 , . . .

, y r ).

We will show that y r+1 = [c w , 0].

Since we consider the identity in the self attention (Equation (9)), we have that p i = y i for every i in {0, . . .

, i}. Now considering the external attention, that is the attention over (K e , V e ), Since all key vectors in K e are [0, 0], the external attention will produce the same score value for all positions.

That is, score(p i , k j1 ) = score(p i , k j2 ) for every j 1 , j 2 .

Lets call this value s .

Thus we have that DISPLAYFORM1 Then, since V e = X we have that DISPLAYFORM2 for every i ∈ {0, . . .

, r}. The last equality holds since our embedding are f (a) = [0, 1] and f (b) = [0, −1], and so every a in w sums one and every b subtracts one.

Thus, we have that DISPLAYFORM3 for every i ∈ {0, . . .

, r}. In the next step, after the external attention plus the residual connection (Equation (10)) we have DISPLAYFORM4 Finally, y r+1 = F (z r ) = z r = [c w , 0] which is exactly what we wanted to prove.

To complete the proof, notice that # a (w) > # b (w) if and only if c w > 0.

If we define F as Q + × Q, the recognizer A = (Σ, f, Trans, s, F) will accept the string w exactly when c w > 0, that is, w ∈ L(A) if and only if # a (w) > # b (w).

That is exactly the language S, and so the proof is complete.

Let M = (Q, Σ, δ, q init , F ) be a Turing machine with a infinite tape and assume that the special symbol # ∈ Σ is used to mark blank positions in the tape.

We make the following assumptions about how M works when processing an input string:• M always moves its head either to the left or to the right (it never stays at the same cell).• M begins at state q init pointing to the cell immediately to the left of the input string.• M never makes a transition to the left of the initial position.• Q has a special state q read used to read the complete input.• Initially (time 0), M makes a transition to state q read and move its head to the right.• While in state q read it moves to the right until symbol # is read.• There are no transitions going out from accepting states (states in F ).It is easy to prove that every general Turing machine is equivalent to one that satisfies the above assumptions.

We prove that one can construct a transformer network Trans M that is able to simulate M on every possible input string.

The construction is somehow involved and uses several helping values, sequences and intermediate results.

To make the reading more easy we divide the construction and proof in three parts.

We first give a high-level view of the strategy we use.

Then we give some details on the architecture of the encoder and decoder needed to implement our strategy, and finally we formally prove that every part of our architecture can be actually implemented.

In the encoder part of Trans M we receive as input the string w = s 1 s 2 . . .

s n .

We first use an embedding function to represent every s i as a one-hot vector and add a positional encoding for every index.

The encoder produces output (K e , V e ) where K e = (k In the decoder part of Trans M we simulate a complete execution of M over w = s 1 s 2 · · · s n .

For this we define the following sequences (for i ≥ 0): DISPLAYFORM0 : state of M at time i : head direction in the transition of M at time iFor the case of m (i) we assume that −1 represents a movement to the left and 1 represents a movement to the right.

In our construction we show how to build a decoder that computes all the above values for every time step i using self attention plus attention over the encoder part.

Since the above values contain all the needed information to reconstruct the complete history of the computation, we can effectively simulate M .In particular our construction produces the sequence of output vectors y 1 , y 2 , . . .

such that, for every i, the vector y i contains information about q (i) and s (i) encoded as one-hot vectors.

The construction and proof goes by induction.

We begin with an initial vector y 0 that represents the state of the computation before it has started, that is q (0) = q init and s (0) = #.

For the induction step we assume that we have already computed y 1 , . . .

, y r such that y i contains information about q (i) and s (i) , and we show how with input (y 0 , y 1 , . . .

, y r ) the decoder produces the next vector y r+1 containing q (r+1) and s (r+1) .The overview of the construction is as follows.

First notice that the transition function δ relates the above values with the following equation: DISPLAYFORM1 We prove that we can use a two-layer feed-forward network to mimic the transition function δ (Lemma B.2).

Thus, given that the input vector y i contains q (i) and s (i) , we can produce the values q (i+1) , v (i) and m (i) (and store them as values in the decoder).

In particular, since y r is in the input, we can produce q (r+1) which is part of what we need for y r+1 .

In order to complete the construction we also need to compute the value s (r+1) , that is, we need to compute the symbol under the head of machine M at the next time step (time r + 1).

We next describe at a high level, how this symbol can be computed with two additional decoder layers.

We first make some observations about s (i) that are fundamental in our computation.

Assume that at time i the head of M is pointing to the cell at index k.

Then we have three possibilities:1.

If i ≤ n, then s (i) = s i since M is still reading its input string.2.

If i > n and M has never written at index k, then s (i) = #, the blank symbol.3.

In other case, that is, if i > n and time i is not the first time that M is pointing to index k, then s (i) is the last symbol written by M at index k.

For the case (1) we can produce s (i) by simply attending to position i in the encoder part.

Thus, if r + 1 ≤ n to produce s (r+1) we can just attend to index r + 1 in the encoder and copy this value to y r+1 .

For cases (2) and FORMULA1 the solution is a bit more complicated, but almost all the important work is to compute what is the index that M is going to be pointing to in time r + 1.To formalize this computation, lets denote by c (i) ∈ Z the following value: DISPLAYFORM2 : the index of the cell to which the head of M is pointing to at time i DISPLAYFORM3 .

If we unroll this equation and assuming that c (0) = 0 we obtain that DISPLAYFORM4 Then, at the step i in the decoder we have all the necessary to compute value c (i) but also the necessary to compute c (i+1) .

We actually show that the computation (of a representation) of c (i) and c (i+1) can be done by using one layer of self attention (Lemma B.3).We still need to define a final notion.

With c (i) one can define the helping value (i) as follows: DISPLAYFORM5 Thus, (i) is a value such that c ( (i)) = c (i) , which means that at time i and at time (i) the head of M was pointing to the same cell.

Moreover, (i) is the maximum value less than i that satisfies such condition.

That is (i) is the last time (previous to i) in which M was pointing to position c (i) .

First notice that in every step, M moves its head either to the right or to the left (it never stays in the same cell).

This implies that for every i it holds that c (i) = c (i−1) , from which we obtain that (i) < i − 1.

Moreover, in the case that c (i) is visited for the first time at time step i, the value (i) is ill-defined.

In such a case we let (i) = i − 1.

This makes (i) ≤ i − 1 for all i, and allows us to check that c (i) is visited for the first time at time step i by just checking that (i) = i − 1.We now have all the necessary to explain how we compute our desired s (r+1) value.

Assume that r + 1 > n (the case r + 1 ≤ n was already covered before).

We first note that if (r + 1) = r then s (r+1) = # since this is the first time that cell c (r+1) is visited.

On the other hand, if (r + 1) < r then s (r+1) is the value written by M at time (r + 1) which is exactly v ( (r+1)) .

Thus, in this case we only need to attend to position (r + 1) and copy the value v ( (r+1)) to produce s (r+1) .

We show that all this can be done with an additional self-attention decoder layer (Lemma B.4).We have described at a high-level a decoder that, with input (y 0 , y 1 , . . .

, y r ), computes the values q (r+1) and s (r+1) which is what we need to produce y r+1 .

We next show all the details of this construction.

In this section we give more details on the architecture of the encoder and decoder needed to implement our strategy.

We let several intermediate claims as lemmas that we formally prove in Section B.4.3.

For our attention mechanism we use the following non-linear function: DISPLAYFORM0 We note that ϕ(x) = −|x| and it can be implemented as ϕ(x) = − relu(x) − relu(−x).

We use ϕ(·) to define a scoring function score ϕ : DISPLAYFORM1 Now, let q ∈ Q d , and K = (k 1 , . . .

, k n ) and V = (v 1 , . . . , v n ) be tuples of elements in Q d .

We now describe how Att(q, K, V ) is generally computed when hard attention is considered.

Assume first that there exists a single j ∈ {1, . . .

, n} that maximizes score ϕ (q, k j ).

In that case we have that Att(q, K, V ) = v j with DISPLAYFORM2 Thus, when computing hard attention with the function score ϕ (·) we essentially select the vector v j such that the dot product q, k j is as close to 0 as possible.

If there is more than one index, say indexes j 1 , j 2 , . . .

, j r , that minimizes the dot product q, k j then we have that DISPLAYFORM3 Thus, in the extreme case in which all dot products are equal q, k j for every index j, attention behaves just as an average of all value vectors, that is Att(q, K, V ) = 1 n n j=1 v j .

We use all these properties of the hard attention in our proof.

We now describe the vectors that we use in the encoder and decoder parts of Trans M .

The vectors that we use in the Trans M layers are of dimension d = 2|Q|+4|Σ|+11.

To simplify the exposition, whenever we use a vector v ∈ Q d , we write it arranged in four groups of values as follows DISPLAYFORM0 where q i ∈ Q |Q| , s i ∈ Q |Σ| , and x i ∈ Q. Whenever in a vector of the above form any of the four groups of values is composed only of 0's, we just write '0, . . .

, 0' where the length of this sequence is implicit in the length of the corresponding group.

Finally, we denote by 0 q the vector in Q |Q| that has only 0's, and similarly 0 s the vector in Q |Σ| that has only 0's.

For a symbol s ∈ Σ, we use s to denote a one-hot vector in Q |Σ| that represents s. That is, given an enumeration π : Σ → {1, . . .

, |Σ|}, the vector s has a 1 in position π(s) and a 0 in all other positions.

Similarly, for q ∈ Q, we use q to denote a one-hot vector in Q |Q| that represents q.

We have the necessary to introduce the embedding and positional encoding used in our construction.

We use an embedding function f : Σ → Q d defined as DISPLAYFORM0 Our construction uses the positional encoding pos : N → Q d such that DISPLAYFORM1 Thus, given an input sequence s 1 s 2 · · · s n ∈ Σ * , we have that DISPLAYFORM2 We denote this last vector by x i .

That is, if M receives the input string w = s 1 s 2 · · · s n , then the input for Trans M is the sequence (x 1 , x 2 , . . .

, x n ).

The need for using a positional encoding having values 1/i and 1/i 2 will be clear when we formally prove the correctness of our construction.

We need a final preliminary notion.

In the formal construction of Trans M we also use the following helping sequences: DISPLAYFORM3 These are used to identify when M is still reading the input string.

The encoder part of Trans M is very simple.

For TEnc M we use a single-layer encoder, such that DISPLAYFORM0 It is straightforward to see that these vectors can be produced with a single encoder layer by using a trivial self attention, taking advantage of the residual connections in Equations (6) and (7) , and then using linear transformations for V (·) and K(·) in Equation (8) .When constructing the decoder we use the following property.

Lemma B.1.

Let q ∈ Q d be a vector such that q = [ , . . .

, , 1, j, , ] where j ∈ N and ' ' denotes an arbitrary value.

Then we have that DISPLAYFORM1 We next show how to construct the decoder part of Trans M to produce the sequence of outputs y 1 , y 2 , . . .

, where y i is given by: DISPLAYFORM2 That is, y i contains information about the state of M at time i, the symbol under the head of M at time i, and the last direction followed by M (the direction of the head movement at time i − 1).

The need to include m (i−1) will be clear in the construction.

We consider as the starting vector for the decoder the vector DISPLAYFORM3 We are assuming that m (−1) = 0 to represent that previous to time 0 there was no head movement.

Our construction resembles a proof by induction; we describe the architecture piece by piece and at the same time we show how for every r ≥ 0 our architecture constructs y r+1 from the previous vectors (y 0 , . . .

, y r ).Thus, assume that y 0 , . . .

, y r satisfy the properties stated above.

Since we are using positional encodings, the actual input for the first layer of the decoder is the sequence y 0 + pos(1), y 1 + pos (2) , . . .

, y r + pos(r + 1).We denote by y i the vector y i plus its positional encoding.

Thus we have that DISPLAYFORM4 For the first self attention in Equation (9) we just produce the identity which can be easily implemented with a trivial attention plus the residual connection.

Thus, we produce the sequence of vectors (p DISPLAYFORM5 by Lemma B.1 we know that if we use p 1 i to attend over the encoder we obtain Att(p DISPLAYFORM6 Thus in Equation (10) we finally produce the vector a DISPLAYFORM7 As the final piece of the first decoder layer we use a function O 1 (·) (Equation (11) ) that satisfies the following lemma.

Lemma B.2.

There exists a two-layer feed-forward network DISPLAYFORM8 besides some other linear transformations.

We finally produce as the output of the first decoder layer, the sequence (z DISPLAYFORM9 Notice that z 1 r already holds info about q (r+1) and m (r) which we need for constructing vector y r+1 .

The single piece of information that we still need to construct is s (r+1) , that is, the symbol under the head of machine M at the next time step (time r + 1).

We next describe how this symbol can be computed with two additional decoder layers.

Recall that c (i) is the cell to which M is pointing to at time i, and that it satisfies that c DISPLAYFORM10 .

We can take advantage of this property to prove the following lemma.

DISPLAYFORM11 , and V 2 (·) defined by feed-forward networks such that DISPLAYFORM12 Lemma B.3 essentially shows that one can construct a representation for values c (i) and c (i+1) for every possible index i.

In particular we will know the value c (r+1) that represents the cell to which the machine is pointing to in the next time step.

Continuing with the decoder layer, when using the self attention above and after adding the residual in Equation (9) we obtain the sequence of vectors (p DISPLAYFORM13 DISPLAYFORM14 We now describe how can we use a third and final decoder layer to produce our desired s (r+1) value (the symbol under the head of M in the next time step).

Recall that (i) is the last time (previous to i) in which M was pointing to position c (i) , or it is i − 1 if this is the first time that M is pointing to c (i) .

We can prove the following lemma.

Lemma B.4.

There exists functions Q 3 (·), K 3 (·), and V 3 (·) defined by feed-forward networks such that DISPLAYFORM15 We prove Lemma B.4 by just showing that, for every i one can attend exactly to position (i+1) and then just copy both values.

We do this by taking advantage of the values c (i) and c (i+1) previously computed for every index i.

Then we have that p

Proof of Lemma B.1.

Let q ∈ Q d be a vector such that q = [ , . . .

, , 1, j, , ] where j ∈ N and ' ' is an arbitrary value.

We next prove that DISPLAYFORM0 where α (j) and β (j) are defined as DISPLAYFORM1 Then we have that DISPLAYFORM2 Notice that, if j ≤ n, then the above expression is maximized when i = j. Otherwise, if j > n then the expression is maximized when i = n. Then Att(q, K e , V e ) = v i where i = j if j ≤ n and i = n if j > n. We note that i as just defined is exactly β (j) .

Thus, given that v i is defined as DISPLAYFORM3 we obtain that DISPLAYFORM4 which is what we wanted to prove.

Proof of Lemma B.2.

In order to prove the lemma we need some intermediate notions and properties.

Assume that the enumeration π 1 : Σ → {1, . . .

, |Σ|} is the one used to construct the one-hot vectors s for s ∈ Σ, and that π 2 : Q → {1, . . .

, |Q|} is the one used to construct q with q ∈ Q. Using π 1 and π 2 one can construct an enumeration for the pairs in Q × Σ and then construct one-hot vectors for pairs in this set.

Formally, given (q, s) ∈ Q × Σ we denote by (q, s) a one-hot vector with a 1 in position (π 1 (s) − 1)|Q| + π 2 (q) and a 0 in every other position.

To simplify the notation we use π(q, s) to denote (π 1 (s) − 1)|Q| + π 2 (q).

One can similarly construct an enumeration π for Q × Σ × {−1, 1} such that π (q, s, m) = π(q, s) if m = −1 and π (q, s, m) = |Q||Σ| + π(q, s) if m = 1.

We denote by (q, s, m) the corresponding one-hot vector for every (q, s, m) ∈ Q × Σ × {−1, 1}. We next prove three helping properties.

In every case q ∈ Q, s ∈ Σ, m ∈ {−1, 1}, and δ(·, ·) is the transition function of machine M .1.

There exists f 1 : DISPLAYFORM5 2.

There exists f δ : DISPLAYFORM6 3.

There exists f 2 : DISPLAYFORM7 Published as a conference paper at ICLR 2019To show (1), lets denote by S i , with i ∈ {1, . . .

, |Σ|}, a matrix of dimensions |Σ| × |Q| such that S i has its i-th row with 1's and it is 0 everywhere else.

We note that for every s ∈ Σ it holds that s S i = 1 if and only if i = π 1 (s) and it is 0 otherwise.

Now, consider the vector v (q,s) DISPLAYFORM8 We first note that for every i ∈ {1, . . .

, |Σ|}, if i = π 1 (s) then q + s S i = q + 0 = q .

Moreover q + s S π1(s) = q + 1 is a vector that has a 2 exactly at index π 2 (q), and it is 1 in all other positions.

Thus, the vector v (q,s) has a 2 exactly at position (π 1 (s) − 1)|Q| + π 2 (q) and it is either 0 or 1 in every other position.

Now, lets denote by o a vector in Q |Q||Σ| that has a 1 in every position and consider the following affine transformation DISPLAYFORM9 Vector g 1 ([ q , s ]) has a 1 only at position (π 1 (s) − 1)|Q| + π 2 (q) = π(q, s) and it is less than or equal to 0 in every other position.

Thus, to construct f 1 (·) we apply the piecewise-linear sigmoidal activation σ(·) (see Equation (1) ) to obtain DISPLAYFORM10 which is what we wanted.

Now, to show (2), lets denote by M δ a matrix of dimensions (|Q||Σ|) × (2|Q||Σ|) constructed as follows.

For (q, s) ∈ Q × Σ, if δ(q, s) = (p, r, m) then M δ has a 1 at position (π(q, s), π (p, r, m)) and it has a 0 in every other position, that is DISPLAYFORM11 It is straightforward to see that (q, s) M δ = δ(q, s) , and thus we can define f 2 (·) as DISPLAYFORM12 To show (3), consider the matrix A of dimensions (2|Q||Σ|) × (|Q| + |Σ| + 1) such that DISPLAYFORM13 Then we define f 3 (·) as DISPLAYFORM14 We are now ready to begin with the proof of the lemma.

Recall that a 1 i is given by a DISPLAYFORM15 We need to construct a function O 1 : DISPLAYFORM16 We first use function h 1 (·) that works as follows.

Lets denote bym (i−1) the value DISPLAYFORM17 where g 1 (·) is the function defined above in Equation FORMULA79 .

It is clear that h 1 (·) is an affine transformation.

Moreover, we note that except for g 1 ([ DISPLAYFORM18 ) are between 0 and 1.

Thus if we apply function σ(·) to h 1 (a DISPLAYFORM19 is the vector with only zeros, then score ϕ (Q 2 (z 1 i ), K 2 (z 1 j )) = 0 for every j ∈ {0, . . .

, i}. Thus, we have that the attention Att(Q 2 (z DISPLAYFORM20 ) that we need to compute is just the average of all the vectors in V 2 (Z DISPLAYFORM21 which is exactly what we wanted to show.

Proof of Lemma B.4.

Recall that z 2 i is the following vector z DISPLAYFORM22 We need to construct functions Q 3 (·), K 3 (·), and V 3 (·) such that DISPLAYFORM23 We first define the query function Q 3 : DISPLAYFORM24 Now, for every j ∈ {0, 1, . . .

, i} we define DISPLAYFORM25 ]

It is clear that the three functions are linear transformations and thus they can be defined by feedforward networks.

Consider now the attention Att(Q 3 (z DISPLAYFORM26 .

In order to compute this value, and since we are considering hard attention, we need to find the value j ∈ {0, 1, . . .

, i} that maximizes DISPLAYFORM27 ).

Actually, assumming that such value is unique, lets say j , then we have that DISPLAYFORM28 We next show that given our definitions above, it always holds that j = (i + 1) and then V 3 (z 2 j ) is exactly the vector that we wanted to obtain.

To simplify the notation, we denote by χ Now, by our definition of Q 3 (·) and K 3 (·) we have that DISPLAYFORM29 where ε k = 1 (k+1) .

We next prove the following auxiliary property.

If j 1 is such that c (j1) = c (i+1) and j 2 is such that DISPLAYFORM30 In order to prove (26), assume first that j 1 ∈ {0, . . .

, i} is such that c (j1) = c (i+1) .

Then we have that |c (i+1) − c (j1) | ≥ 1 since c (i+1) and c (j1) are integer values.

From this we have two possibilities for χ i j1 : DISPLAYFORM31 Notice that 1 ≥

ε j1 ≥ ε i > 0.

Then we have that ε i ε j1 ≥ (ε i ε j1 ) 2 > 1 3 (ε i ε j1 ) 2 , and thus DISPLAYFORM32 Finally, and using again that 1 ≥ ε j1 ≥ ε i > 0, from the above equation we obtain that DISPLAYFORM33 Thus, we have that if DISPLAYFORM34 Now assume j 2 ∈ {0, . . .

, i} is such that c (j2) = c (i+1) .

In this case we have that DISPLAYFORM35 We showed that if c DISPLAYFORM36 This completes the proof of the property in (26).

We have now all the necessary to prove that arg min j |χ i j | = (i + 1).

Recall first that (i + 1) is defined as DISPLAYFORM37 in other case.

Assume first that there exists j ≤ i such that c (j) = c (i+1) .

By (26) we know that arg min j∈{0,...,i} DISPLAYFORM38 On the contrary, assume that for every j ≤ i it holds that c (j) = c (i+1) .

We will prove that in this case |χ for every j ≤ i, then c (i+1) is a cell that has never been visited before by M .

Given that M never makes a transition to the left of its initial cell, then cell c (i+1) is a cell to the right of every other previously visited cell.

This implies that c (i+1) > c (j) for every j ≤ i. Thus, for every j ≤ i we have c DISPLAYFORM39 Moreover, notice that if j < i then ε j >

ε i and thus, if j < i we have that DISPLAYFORM40 The formulas of the Neural GPU in detail are as follows (with S 0 the initial input tensor): DISPLAYFORM41 With U (·), R(·), and F (·) defined as DISPLAYFORM42 Consider now an RNN encoder-decoder N of dimension d and composed of the equations DISPLAYFORM43 with h 0 = 0 and g 0 = h n where n is the length of the input.

We construct a Neural GPU network NGPU that simulates N as follows.

Assume that the input of N is X = (x 1 , . . .

, x n ).

Then we first construct the sequence X = (x 1 , . . .

, x n ) such that x i = [x i , 0, 0, 1, 1, 0] with 0 ∈ Q d the vector with all values as 0.

Notice that x i ∈ Q 3d+3 , moreover it is straightforward that if x i was constructed from an embedding function f : Σ → Q d applied to a symbol a ∈ Σ, then x i can also be constructed with an embedding function f : DISPLAYFORM0 We consider an input tensor S ∈ Q n×1×3d+3 such that for every i ∈ {1, . . .

, n} it holds that S i,1,: = x i = [x i , 0, 0, 1, 1, 0].

Notice that since we picked w = 1, our tensor S is actually a 2D grid.

Our proof shows that a bi-dimensional tensor is enough for simulating an RNN.We now describe how to construct the kernel banks K U , K R and K F of shape (2, 1, 3d + 3, 3d + 3).Notice that for each kernel K X we essentially have to define two matrices K X 1,1,:,: and K X 2,1,:,: each one of dimension (3d + 3) × (3d + 3).

We begin by defining every matrix in K F as block matrices.

When defining the matrices, all blank spaces are considered to be 0.

DISPLAYFORM1 where F 1 and F 2 are 3 × 3 matrices defined by Before continuing with the proof we note that for every kernel K X and tensor S we have that

We now prove that the following properties hold for every t ≥ 0: DISPLAYFORM0 where α k j is given by the recurrence α DISPLAYFORM1 That is, we are going to prove that our construction actually simulates N .

By (27) one can see that the intuition in our construction is to use the first d components to simulate the encoder part, the next d components to communicate data between the encoder and decoder simulation, and the next d components to simulate the decoder part.

The last three components are needed as gadgets for the gates to actually simulate a sequencial read of the input, and to ensure that the hidden state of the encoder and decoder are updated properly.

We prove the above statement by induction in t. First notice that the property trivially holds for S 0 .Now assume that this holds for t − 1 and lets prove it for t. We know that U t is computed as DISPLAYFORM2 Thus we have that: DISPLAYFORM3 By the induction hypothesis we have DISPLAYFORM4 Now, notice that K for i = t − 1 σ FIG2 for i = t σ FIG2 for i > t We are almost done with the inductive step, we only need to compute σ((K F * (R t S t−1 ))

i,1,:).Given what we have for R t and S t−1 we have that R and then, T i,:,: = T i+p,:,: .Consider now an arbitrary uniform Neural GPU that processes tensor S above, and assume that S 1 , S 2 , . . .

, S r is the sequence produced by it.

Next we prove that for every t and for every i it holds that S t i,:,: = S t i+p,:,: .

We prove it by induction in t. For the case S 0 it holds by definition.

Thus assume that S t−1 satisfies the property.

Let DISPLAYFORM5 Since we are considering uniform Neural GPUs, we know that there exist three matrices B U , B This completes the first part of the proof.

We have shown that if the input of a uniform neural GPU is periodic, then the output is also periodic.

We make a final observation.

Let N be a uniform Neural GPU, and S ∈ Q kp×w×d be a tensor such that S i,:,: = S i+p,:,: for every i. Moreover, let T ∈ Q k p×w×d be a tensor such that T i,:,: = T i+p,:,:for every i, and assume that S 1:p,:,: = T 1:p,:,: .

Lets S 1 , S 2 , . . .

and T 1 , T 2 , . . .

be the sequences produced by N .

Then with a similar argument as above it is easy to prove that for every t it holds that S t 1:p,:,: = T t 1:p,:,: .

From this it is easy to prove that uniform Neural GPUs will no be able to recognize the length of periodic inputs.

Thus assume that there is a language recognizer A defined by of a uniform neural GPU N such that L(A) contains all strings of even length.

Assume that u is an arbitrary string in Σ such that |u| = p with p an odd number, and let w = uu and w = uuu.

Notice that |w| = 2p and thus w ∈ L(A), but |w | = 3p and thus w ∈ L(A).Let f : Σ → Q d and let X = f (w) = (x 1 , x 2 , . . . , x 2p ) and X = f (w ) = (x 1 , x 2 , . . . , x 3p ).Consider now the tensor S ∈ Q 2p×w×d such that S i,1,: = x i for i ∈ {1, . . .

, 2p}, thus S i,:,: = S i+p,:,: .

Similarly, consider T ∈ Q 3p×w×d such that such that T i,1,: = x i for i ∈ {1, . . .

, 3p}, and thus T i,:,: = T i+p,:,: .

Notice that S 1:p,:,: = T 1:p,:,: then by the property above we have that for every t it holds that S .

From this we conclude that the outputs of N for both inputs X and X are the same, and thus if A accepts w then A accepts w which is a contradiction.

@highlight

We show that the Transformer architecture and the Neural GPU are Turing complete.