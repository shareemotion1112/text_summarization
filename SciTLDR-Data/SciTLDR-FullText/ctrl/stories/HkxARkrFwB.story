Deep learning natural language processing models often use vector word embeddings, such as word2vec or GloVe, to represent words.

A discrete sequence of words can be much more easily integrated with downstream neural layers if it is represented as a  sequence of continuous vectors.

Also, semantic relationships between words, learned from a text corpus, can be encoded in the relative configurations of the embedding vectors.

However, storing and accessing embedding vectors for all words in a dictionary requires large amount of space, and may stain systems with limited GPU memory.

Here, we used approaches inspired by quantum computing to propose two related methods, word2ket and word2ketXS, for storing word embedding matrix during training and inference in a highly efficient way.

Our approach achieves a hundred-fold or more reduction in the space required to store the embeddings with almost no relative drop in accuracy in practical natural language processing tasks.

Modern deep learning approaches for natural language processing (NLP) often rely on vector representation of words to convert discrete space of human language into continuous space best suited for further processing through a neural network.

For a language with vocabulary of size d, a simple way to achieve this mapping is to use one-hot representation -each word is mapped to its own row of a d × d identity matrix.

There is no need to actually store the identity matrix in memory, it is trivial to reconstruct the row from the word identifier.

Word embedding approaches such as word2vec (Mikolov et al., 2013) or GloVe (Pennington et al., 2014) use instead vectors of dimensionality p much smaller than d to represent words, but the vectors are not necessarily extremely sparse nor mutually orthogonal.

This has two benefits: the embeddings can be trained on large text corpora to capture the semantic relationship between words, and the downstream neural network layers only need to be of width proportional to p, not d, to accept a word or a sentence.

We do, however, need to explicitly store the d × p embedding matrix in GPU memory for efficient access during training and inference.

Vocabulary sizes can reach d = 10 5 or 10 6 (Pennington et al., 2014) , and dimensionality of the embeddings used in current systems ranges from p = 300 (Mikolov et al., 2013; Pennington et al., 2014) to p = 1024 (Devlin et al., 2018) .

The d × p embedding matrix thus becomes a substantial, often dominating, part of the parameter space of a learning model.

In classical computing, information is stored in bits -a single bit represents an element from the set B = {0, 1}, it can be in one of two possible states.

A quantum equivalent of a bit, a qubit, is fully described by a single two-dimensional complex unit-norm vector, that is, an element from the set C 2 .

A state of an n-qubit quantum register corresponds to a vector in C 2 n .

To have exponential dimensionality of the state space, though, the qubits in the register have to be interconnected so that their states can become entangled; a set of all possible states of n completely separated, independent qubits can be fully represented by C 2n instead of C 2 n .

Entanglement is a purely quantum phenomenon -we can make quantum bits interconnected, so that a state of a two-qubit system cannot be decomposed into states of individual qubits.

We do not see entanglement in classical bits, which are always independent -we can describe a byte by separately listing the state of each of the eight bits.

We can, however, approximate quantum register classically -store vectors of size m using O (log m) space, at the cost of losing the ability to express all possible m-dimensional vectors that an actual O (log m)-qubit quantum register would be able to represent.

As we show in this paper, the loss of representation power does not have a significant impact on NLP machine learning algorithms that use the approximation approaches to store and manipulate the high-dimensional word embedding matrix.

Here, we used approaches inspired by quantum computing to propose two related methods, word2ket and word2ketXS, for storing word embedding matrix during training and inference in a highly efficient way 1 .

The first method operates independently on the embedding of each word, allowing for more efficient processing, while the second method operates jointly on all word embeddings, offering even higher efficiency in storing the embedding matrix, at the cost of more complex processing.

Empirical evidence from three NLP tasks shows that the new word2ket embeddings offer high space saving rate at little cost in terms of accuracy of the downstream NLP model.

Consider two separable 2 Hilbert spaces V and W. A tensor product space of V and W, denoted as V ⊗ W, is a separable Hilbert space H constructed using ordered pairs v ⊗ w, where v ∈ V and w ∈ W. In the tensor product space, the addition and multiplication in H have the following properties

The inner product between v ⊗ w and v ⊗ w is defined as a product of individual inner products v ⊗ w, v ⊗ w = v, v w, w .

(2) It immediately follows that ||v ⊗ w|| = ||v|| ||w||; in particular, a tensor product of two unit-norm vectors, from V and W, respectively, is a unit norm vector in V ⊗ W.

The Hilbert space V ⊗ W is a space of equivalence classes of pairs v ⊗ w; for example {cv} ⊗ w and v ⊗ {cw} are equivalent ways to write the same vector.

A vector in a tensor product space is often simply called a tensor.

Let {ψ j } and {φ k } be orthonormal basis sets in V and W, respectively.

From eq. 1 and 2 we can see that

where δ z is the Kronecker delta, equal to one at z = 0 and to null elsewhere.

That is, the set {ψ j ⊗ φ k } jk forms an orthonormal basis in V ⊗ W, with coefficients indexed by pairs jk and numerically equal to the products of the corresponding coefficients in V and W. We can add any pairs of vectors in the new spaces by adding the coefficients.

The dimensionality of V ⊗ W is the product of dimensionalities of V and W.

We can create tensor product spaces by more than one application of tensor product, H = U ⊗V ⊗W, with arbitrary bracketing, since tensor product is associative.

Tensor product space of the form

is said to have tensor order 3 of n.

1 In Dirac notation popular in quantum mechanics and quantum computing, a vector u ∈ C 2 n is written as |u , and called a ket.

2 That is, with countable orthonormal basis.

3 Note that some sources alternatively call n a degree or a rank of a tensor.

Here, we use tensor rank to refer to a property similar to matrix rank, see below.

Consider H = V ⊗ W. We have seen the addition property v ⊗ w + v ⊗ w = {v + v } ⊗ w and similar property with linearity in the first argument -tensor product is bilinear.

We have not, however, seen how to express v ⊗ w + v ⊗ w as φ ⊗ ψ for some φ ∈ V, ψ ∈ W. In many cases, while the left side is a proper vector from the tensor product space, it is not possible to find such φ and ψ.

The tensor product space contains not only vectors of the form v ⊗ w, but also their linear combinations, some of which cannot be expressed as φ ⊗ ψ.

For example,

can be decomposed as

cannot; no matter what we choose as coefficients a, b, c, d, we have

since we require ac = 1/ √ 2, that is, a = 0, c = 0, and similarly bd = 1/ √ 2, that is, b = 0, c = 0, yet we also require bd = ad = 0, which is incompatible with a, b, c, d = 0.

For tensor product spaces of order n, that is, is a tensor of rank 2.

Tensors with rank greater than one are called entangled.

Maximum rank of a tensor in a tensor product space of order higher than two is not known in general (Buczyński & Landsberg, 2013) .

A p-dimensional word embedding model involving a d-token vocabulary is 5 a mapping f :

p , that is, it maps word identifiers into a p-dimensional real Hilbert space, an inner product space with the standard inner product ·, · leading to the L 2 norm.

Function f is trained to capture semantic information from the language corpus it is trained on, for example, two words i, j with f (i), f (j) ∼ 0 are expected to be semantically unrelated.

In practical implementations, we represent f as a collection of vectors f i ∈ R p indexed by i, typically in the form of d × p matrix M , with embeddings of individual words as rows.

We propose to represent an embedding v ∈ R p of each a single word as an entangled tensor.

Specifically, in word2ket, we use tensor of rank r and order n of the form

where v jk ∈ R q .

The resulting vector v has dimension p = q n , but takes rnq = O (rq log q log p) space.

We use q ≥ 4; it does not make sense to reduce it to q = 2 since a tensor product of two vectors in R 2 takes the same space as a vector in R 4 , but not every vector in R 4 can be expressed as a rank-one tensor in R 2 ⊗ R 2 .

If the downstream computation involving the word embedding vectors is limited to inner products of embedding vectors, there is no need to explicitly calculate the q n -dimensional vectors.

Indeed, we have (see eq. 2)

Thus, the calculation of inner product between two p-dimensional word embeddings, v and w, represented via word2ket takes O (rq log q log p) time and O (1) additional space.

In most applications, a small number of embedding vectors do need to be made available for processing through subsequent neural network layers -for example, embeddings of all words in all sentences in a batch.

For a batch consisting of b words, the total space requirement is O (bp + rq log q log p), instead of O (rp) in traditional word embeddings.

Reconstructing a single p-dimensional word embedding vector from a tensor of rank r and order n takes O (rn log 2 p) arithmetic operations.

To facilitate parallel processing, we arrange the order-n tensor product space into a balanced tensor product tree (see Figure 1) , with the underlying vectors v jk as leaves, and v as root.

For example, for n = 4,

Instead of performing n multiplications sequentially, we can perform them in parallel along branches of the tree, reducing the length of the sequential processing to O (log 2 n).

Typically, word embeddings are trained using gradient descent.

The proposed embedding representation involves only differentiable arithmetic operations, so gradients with respect to individual elements of vectors v jk can always be defined.

With the balanced tree structure, word2ket representation can be seen as a sequence of O (log 2 n) linear layers with linear activation functions, where n is already small.

Still, the gradient of the embedding vector v with respect to an underlying tunable parameters v lk involves products ∂ k n j=1 v jk /∂v lk = j =l v jk , leading to potentially high Lipschitz constant of the gradient, which may harm training.

To alleviate this problem, at each node in the balanced tensor product tree we use LayerNorm (Ba et al., 2016) .

Let A : V → U be a linear operator that maps vectors from Hilbert space V into vector in Hilbert space U; that is, for v, v , ∈ V, α, β ∈ R, the vector A(αv + βv ) = αAv + βAv is a member of U. Let us also define a linear operator B : W → Y. A mapping A ⊗ B is a linear operator that maps vectors from V ⊗ W into vectors in U ⊗ Y.

We define A ⊗ B : V ⊗ W → U ⊗ Y through its action on simple vectors and through linearity

for ψ j ∈ V and φ k ∈ U. Same as for vectors, tensor product of linear operators is bilinear

In finite-dimensional case, for n × n matrix representation of linear operator A and m × m matrix representing B, we can represent A ⊗ B as an mn × m n matrix composed of blocks a jk B.

We can see a p-dimensional word embedding model involving a d-token vocabulary as a linear operator F : R d → R p that maps the one-hot vector corresponding to a word into the corresponding word embedding vector.

Specifically, if e i is the i-th basis vector in R d representing i-th word in the vocabulary, and v i is the embedding vector for that word in R p , then the word embedding linear operator is

T .

If we store the word embeddings a d × p matrix M , we can then interpret that matrix's transpose, M T , as the matrix representation of the linear operator F .

Consider q and t such that q n = p and t n = d, and a series of n linear operators

In word2ketXS, we represent the d × p word embedding matrix as

where F jk can be represented by a q × t matrix.

The resulting matrix F has dimension p × d, but takes rnqt = O (rq log qt log t log p log d) space.

Intuitively, the additional space efficiency comes from applying tensor product-based exponential compression not only horizontally, individually to each row, but horizontally and vertically at the same time, to the whole embedding matrix.

We use the same balanced binary tree structure as in word2ket.

To avoid reconstructing the full embedding matrix each time a small number of rows is needed for a multiplication by a weight matrix in the downstream layer of the neural NLP model, which would eliminate any space saving, we use lazy tensors (Gardner et al., 2018; Charlier et al., 2018) .

If A is an m × n matrix and matrix B is p × q, then ij th entry of A ⊗ B is equal to

As we can see, reconstructing a row of the full embedding matrix involves only single rows of the underlying matrices, and can be done efficiently using lazy tensors.

In order to evaluate the ability of the proposed space-efficient word embeddings in capturing semantic information about words, we used them in three different downstream NLP tasks: text summarization, language translation, and question answering.

In all three cases, we compared the accuracy in the downstream task for the proposed space-efficient embeddings with the accuracy achieved by regular embeddings, that is, embeddings that store p-dimensional vectors for d-word vocabulary using a single d × p matrix.

In text summarization experiments, we used the GIGAWORD text summarization dataset (Graff et al., 2003) using the same preprocessing as , that is, using 200K examples in training.

We used an encoder-decoder sequence-to-sequence architecture with bidirectional forwardbackward RNN encoder and an attention-based RNN decoder (Luong et al., 2015) , as implemented in PyTorch-Texar Hu et al. (2018) .

In both the encoder and the decoder we used internal layers with dimensionality of 256 and dropout rate of 0.2, and trained the models, starting from random weights and embeddings, for 20 epochs.

We used the validation set to select the best model epoch, and reported results on a separate test set.

We used Rouge 1, 2, and L scores (Lin, 2004) .

In addition to testing the regular dimensionality of 256, we also explored 400, and 8000, but kept the dimensionality of other layers constant.

The results in Table 1 show that word2ket can achieve 16-fold reduction in trainable parameters at the cost of a drop of Rouge scores by about 2 points.

As expected, word2ketXS is much more spaceefficient, matching the scores of word2ket while allowing for 34,000 fold reduction in trainable parameters.

More importantly, it offers over 100-fold space reduction while reducing the Rouge scores by only about 0.5.

Thus, in the evaluation on the remaining two NLP tasks we focused on word2ketXS.

The second task we explored is German-English machine translation, using the IWSLT2014 (DE-EN) dataset of TED and TEDx talks as preprocessed in (Ranzato et al., 2016) .

We used the same sequence-to-sequence model as in GIGAWORD summarization task above.

We used BLEU score to measure test set performance.

We explored embedding dimensions of 100, 256, 400, 1000, and 8000 by using different values for the tensor order and the dimensions of the underlying matrices F jk .

The results in Table 2 show a drop of about 1 point on the BLEU scale for 100-fold reduction in the parameter space, with drops of 0.5 and 1.5 for lower and higher space saving rates, respectively.

The third task we used involves the Stanford Question Answering Dataset (SQuAD) dataset.

We used the DrQA's model (Chen et al., 2017) , a 3-layer bidirectional LSTMs with 128 hidden units for both paragraph and question encoding.

We trained the model for 40 epochs, starting from random weights and embeddings, and reported the test set F1 score.

DrQA uses an embedding with vocabulary size of 118,655 and embedding dimensionality of 300.

As the embedding matrix is larger, we can increase the tensor order in word2ketXS to four, which allows for much higher space savings.

Results in Table 3 show a 0.5 point drop in F1 score with 1000-fold saving of the parameter space required to store the embeddings.

For order-4 tensor word2ketXS, we see almost 10 5 -fold space saving rate, at the cost of a drop of F1 by less than two points, that is, by a relative drop of less than 3%.

We also investigated the computational overhead introduced by the word2ketXS embeddings.

For tensors order 2, the training time for 40 epochs increased from 5.8 for the model using regular embedding to 7.4 hours for the word2ketXS-based model.

Using tensors of order 4, to gain additional space savings, increased the time to 9 hours.

Each run was executed on a single NVIDIA Tesla V100 GPU card, on a machine with 2 Intel Xeon Gold 6146 CPUs and 384 GB RAM.

While the training time increased, as shown in Fig. 3 , the dynamics of model training remains largely unchanged.

The results of the experiments show substantial decreases in the memory footprint of the word embedding part of the model, used in the input layers of the encoder and decoder of sequence-tosequence models.

These also have other parameters, including weight matrices in the intermediate layers, as well as the matrix of word probabilities prior to the last, softmax activation, that are not compressed by our method.

During inference, embedding and other layers dominate the memory footprint of the model.

Recent successful transformer models like BERT by (Devlin et al., 2018) , GPT-2 by , RoBERTa by (Liu et al., 2019) and Sparse Transformers by require hundreds of millions of parameters to work.

In RoBERTa BASE , 30% of the parameters belong to the word embeddings.

During training, there is an additional memory need to store activations in the forward phase in all layers, to make them available for calculating the gradients in the backwards phase.

These often dominate the memory footprint during training, but one can decrease the memory required for Figure 2 : Dynamics of the test-set F1 score on SQuAD dataset using DrQA model with different embeddings, for rank-2 order-2 word2ketXS, for rank-1 order-4 word2ketXS, for for the regular embedding.

Given the current hardware limitation for training and inference, it is crucial to be able to decrease the amount of memory these networks requires to work.

A number of approaches have been used in lowering the space requirements for word embeddings.

Dictionary learning (Shu & Nakayama, 2018) and word embedding clustering (Andrews, 2016) approaches have been proposed.

Bit encoding has been also proposed Gupta et al. (2015) .

An optimized method for uniform quantization of floating point numbers in the embedding matrix has been proposed recently (May et al., 2019) .

To compress a model for low-memory inference, (Han et al., 2015) used pruning and quantization for lowering the number of parameters.

For low-memory training sparsity (Mostafa & Wang, 2019) (Gale et al., 2019) (Sohoni et al., 2019) and low numerical precision (De Sa et al., 2018) (Micikevicius et al., 2017) approaches were proposed.

In approximating matrices in general, Fourier-based approximation methods have also been used Avron et al., 2017) .

None of these approaches can mach space saving rates achieved by word2ketXS.

The methods based on bit encoding, such as Andrews (2016); Gupta et al. (2015) ; May et al. (2019) are limited to space saving rate of at most 32 for 32-bit architectures.

Other methods, for example based on parameter sharing Suzuki & Nagata (2016) or on PCA, can offer higher saving rates, but their storage requirement is limited by d + p, the vocabulary size and embedding dimensionality.

In more distantly related work, tensor product spaces have been used in studying document embeddings, by using sketching of a tensor representing n-grams in the document Arora et al. (2018) .

<|TLDR|>

@highlight

We use ideas from quantum computing to proposed word embeddings that utilize much fewer trainable parameters.