Generating formal-language represented by relational tuples, such as Lisp programs or mathematical expressions, from a natural-language input is an extremely challenging task because it requires to explicitly capture discrete symbolic structural information from the input to generate the output.

Most state-of-the-art neural sequence models do not explicitly capture such structure information, and thus do not perform well on these tasks.

In this paper, we propose a new encoder-decoder model based on Tensor Product Representations (TPRs) for Natural- to Formal-language generation, called TP-N2F.

The encoder of TP-N2F employs TPR 'binding' to encode natural-language symbolic structure in vector space and the decoder uses TPR 'unbinding' to generate a sequence of relational tuples, each consisting of a relation (or operation) and a number of arguments, in symbolic space.

TP-N2F considerably outperforms LSTM-based Seq2Seq models, creating a new state of the art results on two benchmarks: the MathQA dataset for math problem solving, and the AlgoList dataset for program synthesis.

Ablation studies show that improvements are mainly attributed to the use of TPRs in both the encoder and decoder to explicitly capture relational structure information for symbolic reasoning.

When people perform explicit reasoning, they can typically describe the way to the conclusion step by step via relational descriptions.

There is ample evidence that relational representations are important for human cognition (e.g., (Goldin-Meadow & Gentner, 2003; Forbus et al., 2017; Crouse et al., 2018; Chen & Forbus, 2018; Chen et al., 2019) ).

Although a rapidly growing number of researchers use deep learning to solve complex symbolic reasoning and language tasks (a recent review is (Gao et al., 2019) ), most existing deep learning models, including sequence models such as LSTMs, do not explicitly capture human-like relational structure information.

In this paper we propose a novel neural architecture, TP-N2F, to solve natural-to formal-language generation tasks (N2F).

In the tasks we study, math or programming problems are stated in naturallanguage, and answers are given as programs, sequences of relational representations, to solve the problem.

TP-N2F encodes the natural-language symbolic structure of the problem in an input vector space, maps this to a vector in an intermediate space, and uses that vector to produce a sequence of output vectors that are decoded as relational structures.

Both input and output structures are modelled as Tensor Product Representations (TPRs) (Smolensky, 1990) .

During encoding, NL-input symbolic structures are encoded as vector space embeddings using TPR 'binding' (following Palangi et al. (2018) ); during decoding, symbolic constituents are extracted from structure-embedding output vectors using TPR 'unbinding' (following Huang et al. (2018; ).

Our contributions in this work are as follows.

(i) We propose a role-level analysis of N2F tasks. (ii) We present a new TP-N2F model which gives a neural-network-level implementation of a model solving the N2F task under the role-level description proposed in (i).

To our knowledge, this is the first model to be proposed which combines both the binding and unbinding operations of TPRs to achieve generation tasks through deep learning. (iii) State-of-the-art performance on two recently developed N2F tasks shows that the TP-N2F model has significant structure learning ability on tasks requiring symbolic reasoning through program synthesis.

The TPR mechanism is a method to create a vector space embedding of complex symbolic structures.

The type of a symbol structure is defined by a set of structural positions or roles, such as the leftchild-of-root position in a tree, or the second-argument-of-R position of a given relation R. In a particular instance of a structural type, each of these roles may be occupied by a particular filler, which can be an atomic symbol or a substructure (e.g., the entire left sub-tree of a binary tree can serve as the filler of the role left-child-of-root).

For now, we assume the fillers to be atomic symbols.

1 The TPR embedding of a symbol structure is the sum of the embeddings of all its constituents, each constituent comprising a role together with its filler.

The embedding of a constituent is constructed from the embedding of a role and the embedding of the filler of that role: these are joined together by the TPR 'binding' operation, the tensor (or generalized outer) product ???.

Formally, suppose a symbolic type is defined by the roles {r i }, and suppose that in a particular instance of that type, S, role r i is bound by filler f i .

The TPR embedding of S is the order-2 tensor

where {f i } are vector embeddings of the fillers and {r i } are vector embeddings of the roles.

In Eq. 1, and below, for notational simplicity we conflate order-2 tensors and matrices.

As a simple example, consider the symbolic type string, and choose roles to be r 1 = first element, r 2 = second element, etc.

Then in the specific string S = cba, the first role r 1 is filled by c, and r 2 and r 3 by b and a, respectively.

The TPR for S is c ??? r 1 + b ??? r 2 + a ??? r 3 , where a, b, c are the vector embeddings of the symbols a, b, c, and r i is the vector embedding of role r i .

A TPR scheme for embedding a set of symbol structures is defined by a decomposition of those structures into roles bound to fillers, an embedding of each role as a role vector, and an embedding of each filler as a filler vector.

Let the total number of roles and fillers available be n R , n F , respectively.

Define the matrix of all possible role vectors to be R ??? R dR??nR , with column i, [R] :i = r i ??? R dR , comprising the embedding of r i .

Similarly let F ??? R dF??nF be the matrix of all possible filler vectors.

The TPR T ??? R dF??dR .

Below, d R , n R , d F , n F will be hyper-parameters, while R, F will be learned parameter matrices.

Using summation in Eq.1 to combine the vectors embedding the constituents of a structure risks non-recoverability of those constituents given the embedding T of the the structure as a whole.

The tensor product is chosen as the binding operation in order to enable recovery of the filler of any role in a structure S given its TPR T.

This can be done with perfect precision if the embeddings of the roles are linearly independent.

In that case the role matrix R has a left inverse U : U R = I. Now define the unbinding (or dual) vector for role r j , u j , to be the j th column of U : U :j .

Then,

we have r i u j = ?? ji .

This means that, to recover the filler of r j in the structure with TPR T, we can take its tensor inner product (or matrix-vector product) with u j :

In the architecture proposed here, we will make use of both TPR binding using the tensor product with role vectors r i and TPR unbinding using the tensor inner product with unbinding vectors u j .

Binding will be used to produce the order-2 tensor T S embedding of the NL problem statement.

Unbinding will be used to generate output relational tuples from an order-3 tensor H. Because they pertain to different representations (of different orders in fact), the binding and unbinding vectors we will use are not related to one another.

We propose a general TP-N2F neural network architecture operating over TPRs to solve N2F tasks under a proposed role-level description of those tasks.

In this description, natural-language input is represented as a straightforward order-2 role structure, and formal-language relational representations of outputs are represented with a new order-3 recursive role structure proposed here.

Figure 1 shows an overview diagram of the TP-N2F model.

It depicts the following high-level description.

As shown in Figure 1 , while the natural-language input is a sequence of words, the output is a sequence of multi-argument relational tuples such as (R A 1 A 2 ), a 3-tuple consisting of a binary relation (or operation) R with its two arguments.

The "TP-N2F encoder" uses two LSTMs to produce a pair consisting of a filler vector and a role vector, which are bound together with the tensor product.

These tensor products, concatenated, comprise the "context" over which attention will operate in the decoder.

The sum of the word-level TPRs, flattened to a vector, is treated as a representation of the entire problem statement; it is fed to the "Reasoning MLP", which transforms this encoding of the problem into a vector encoding the solution.

This is the initial state of the "TP-N2F decoder" attentional LSTM, which outputs at each time step an order-3 tensor representing a relational tuple.

To generate a correct tuple from decoder operations, the model must learn to give the order-3 tensor the form of a TPR for a (R A 1 A 2 ) tuple (detailed explanation in Sec. 3.1.2).

In the following sections, we first introduce the details of our proposed role-level description for N2F tasks, and then present how our proposed TP-N2F model uses TPR binding and unbinding operations to create a neural network implementation of this description of N2F tasks.

In this section, we propose a role-level description of N2F tasks, which specifies the filler/role structures of the input natural-language symbolic expressions and the output relational representations.

Instead of encoding each token of a sentence with a non-compositional embedding vector looked up in a learned dictionary, we use a learned role-filler decomposition to compose a tensor representation for each token.

Given a sentence S with n word tokens {w 0 , w 1 , ..., w n???1 }, each word token w t is assigned a learned role vector r t , soft-selected from the learned dictionary R, and a learned filler vector f t , soft-selected from the learned dictionary F (Sec. 2).

The mechanism closely follows that of Palangi et al. (2018) , and we hypothesize similar results: the role and filler approximately encode the grammatical role of the token and its lexical semantics, respectively.

3 Then each word token w to the set of all its token embeddings {T 0 , . . .

, T n???1 }, the sentence S as a whole is assigned a TPR equal to the sum of the TPR embeddings of all its word tokens:

Using TPRs to encode natural language has several advantages.

First, natural language TPRs can be interpreted by exploring the distribution of tokens grouped by the role and filler vectors they are assigned by a trained model (as in Palangi et al. (2018) ).

Second, TPRs avoid the Bag of Word (BoW) confusion (Huang et al., 2018) : the BoW encoding of Jay saw Kay is the same as the BoW encoding of Kay saw Jay but the encodings are different with TPR embedding, because the role filled by a symbol changes with its context.

In this section, we propose a novel recursive role-level description for representing symbolic relational tuples.

Each relational tuple contains a relation token and multiple argument tokens.

Given a binary relation R, a relational tuple can be written as (rel arg 1 arg 2 ) where arg 1 , arg 2 indicate two arguments of relation rel.

Let us adopt the two positional roles, p

is arg i .

Now let us use role decomposition recursively, noting that the role p rel i can itself be decomposed into a sub-role p i = arg i -of-which has a sub-filler rel.

Suppose that arg i , rel, p i are embedded as vectors a i , r, p i .

Then the TPR encoding of p rel i is r ??? p i , so the TPR encoding of filler arg i bound to role p rel i is a i ??? (r ??? p i ).

The tensor product is associative, so we can omit parentheses and write the TPR for the formal-language expression, the relational tuple (rel arg 1 arg 2 ), as:

Given the unbinding vectors p i for positional role vectors p i and the unbinding vector r for the vector r that embeds relation rel, each argument can be unbound in two steps as shown in Eqs. 4-5.

Here ?? denotes the tensor inner product, which for the order-3 H and order-1 p i in Eq. 4 can be defined as

in Eq. 5, ?? is equivalent to the matrix-vector product.

Our proposed scheme can be contrasted with the TPR scheme in which (rel arg 1 arg 2 ) is embedded as r ??? a 1 ??? a 2 (e.g., Smolensky et al. (2016) ; Schlag & Schmidhuber (2018) ).

In that scheme, an n-ary-relation tuple is embedded as an order-(n + 1) tensor, and unbinding an argument requires knowing all the other arguments (to use their unbinding vectors).

In the scheme proposed here, an n-ary-relation tuple is still embedded as an order-3 tensor: there are just n terms in the sum in Eq. 3, using n position vectors p 1 , . . .

, p n ; unbinding simply requires knowing the unbinding vectors for these fixed position vectors.

In the model, the order-3 tensor H of Eq. 3 has a different status than the order-2 tensor T S of Sec. 3.1.1.

T S is a TPR by construction, whereas H is a TPR as a result of successful learning.

To generate the output relational tuples, the decoder assumes each tuple has the form of Eq. 3, and performs the unbinding operations which that structure calls for.

In Appendix Sec. A.3, it is shown that, if unbinding each of a set of roles from some unknown tensor T gives a target set of fillers, then T must equal the TPR generated by those role/filler pairs, plus some tensor that is irrelevant because unbinding from it produces the zero vector.

In other words, if the decoder succeeds in producing filler vectors that correspond to output relational tuples that match the target, then, as far as what the decoder can see, the tensor that it operates on is the TPR of Eq. 3.

To generate formal relational tuples from natural-language descriptions, a learning strategy for the mapping between the two structures is particularly important.

As shown in (6), we formalize the learning scheme as learning a mapping function f mapping (??), which, given a structural representation of the natural-language input, T S , outputs a tensor T F from which the structural representation of the output can be generated.

At the role level of description, there's nothing more to be said about this mapping; how it is modeled at the neural network level is discussed in Sec. 3.2.1.

As shown in Figure 1 , the TP-N2F model is implemented with three steps: encoding, mapping, and decoding.

The encoding step is implemented by the TP-N2F natural-language encoder (TP-N2F Encoder), which takes the sequence of word tokens as inputs, and encodes them via TPR binding according to the TP-N2F role scheme for natural-language input given in Sec. 3.1.1.

The mapping step is implemented by an MLP called the Reasoning Module, which takes the encoding produced by the TP-N2F Encoder as input.

It learns to map the natural-language-structure encoding of the input to a representation that will be processed under the assumption that it follows the role scheme for output relational-tuples specified in Sec. 3.1.2: the model needs to learn to produce TPRs such that this processing generates correct output programs.

The decoding step is implemented by the TP-N2F relational tuples decoder (TP-N2F Decoder), which takes the output from the Reasoning Module (Sec. 3.1.3) and decodes the target sequence of relational tuples via TPR unbinding.

The TP-N2F Decoder utilizes an attention mechanism over the individual-word TPRs T t produced by the TP-N2F Encoder.

The detailed implementations are introduced below.

The TP-N2F encoder follows the role scheme in Sec. 3.1.1 to encode each word token w t by softselecting one of n F fillers and one of n R roles.

The fillers and roles are embedded as vectors.

These embedding vectors, and the functions for selecting fillers and roles, are learned by two LSTMs, the Filler-LSTM and the Role-LSTM.

(See Figure 2. )

At each time-step t, the Filler-LSTM and the Role-LSTM take a learned word-token embedding w t as input.

The hidden state of the Filler-LSTM, h t F , is used to compute softmax scores u F k over n F filler slots, and a filler vector f t = F u F is computed from the softmax scores (recall from Sec. 2 that F is the learned matrix of filler vectors).

Similarly, a role vector is computed from the hidden state of the Role-LSTM, h t R .

f F and f R denote the functions that generate f t and r t from the hidden states of the two LSTMs.

The token w t is encoded as T t , the tensor product of f t and r t .

T t replaces the hidden vector in each LSTM and is passed to the next time step, together with the LSTM cell-state vector c t : see (7)-(8).

After encoding the whole sequence, the TP-N2F encoder outputs the sum of all tensor products t T t to the next module.

We use an MLP, called the Reasoning MLP, for TPR mapping; it takes an order-2 TPR from the encoder and maps it to the initial state of the decoder.

Detailed equations and implementation are provided in Sec. A.2.1 of the Appendix.

Figure 2: Implementation of the TP-N2F encoder.

The TP-N2F Decoder is an RNN that takes the output from the reasoning MLP as its initial hidden state for generating a sequence of relational tuples (Figure 3 ).

This decoder contains an attentional LSTM called the Tuple-LSTM which feeds an unbinding module: attention operates on the context vector of the encoder, consisting of all individual encoder outputs {T t }.

The hidden-state H of the Tuple-LSTM is treated as a TPR of a relational tuple and is unbound to a relation and arguments.

During training, the Tuple-LSTM needs to learn a way to make H suitably approximate a TPR.

At each time step t, the hidden state H t of the Tuple-LSTM with attention (The version in Luong et al. (2015)) (9) is fed as input to the unbinding module, which regards H t as if it were the TPR of a relational tuple with m arguments possessing the role structure described in Sec. 3.1.2: Figure 3 , the assumed hypothetical form of H t , as well as that of B t i below, is shown in a bubble with dashed border.)

To decode a binary relational tuple, the unbinding module decodes it from H t using the two steps of TPR unbinding given in (4)-(5).

The positional unbinding vectors p i are learned during training and shared across all time steps.

After the first unbinding step (4), i.e., the inner product of H t with p i , we get tensors B t i (10).

These are treated as the TPRs of two arguments a t i bound to a relation r t .

A relational unbinding vector r t is computed by a linear function from the sum of the B t i and used to compute the inner product with each B t i to yield a t i , which are treated as the embedding of argument vectors (11).

Based on the TPR theory, r t is passed to a linear function to get r t as the embedding of a relation vector.

Finally, the softmax probability distribution over symbolic outputs is computed for relations and arguments separately.

In generation, the most probable symbol is selected. (More detailed equations are in Appendix Sec. A.2.3)

Figure 3: Implementation of the TP-N2F decoder.

During inference time, natural language questions are encoded via the encoder and the Reasoning MLP maps the output of the encoder to the input of the decoder.

We use greedy decoding (selecting the most likely class) to decode one relation and its arguments.

The relation and argument vectors are concatenated to construct a new vector as the input for the Tuple-LSTM in the next step.

TP-N2F is trained using back-propagation (Rumelhart et al., 1986) with the Adam optimizer (Kingma & Ba, 2017) and teacher-forcing.

At each time step, the ground-truth relational tuple is provided as the input for the next time step.

As the TP-N2F decoder decodes a relational tuple at each time step, the relation token is selected only from the relation vocabulary and the argument tokens from the argument vocabulary.

For an input I that generates N output relational tuples, the loss is the sum of the cross entropy loss L between the true labels L and predicted tokens for relations and arguments as shown in (12).

The proposed TP-N2F model is evaluated on two N2F tasks, generating operation sequences to solve math problems and generating Lisp programs.

In both tasks, TP-N2F achieves state-of-the-art performance.

We further analyze the behavior of the unbinding relation vectors in the proposed model.

Results of each task and the analysis of the unbinding relation vectors are introduced in turn.

Details of experiments and datasets are described in Sec. A.1 in the Appendix.

Given a natural-language math problem, we need to generate a sequence of operations (operators and corresponding arguments) from a set of operators and arguments to solve the given problem.

Each operation is regarded as a relational tuple by viewing the operator as relation, e.g., (add, n1, n2).

We test TP-N2F for this task on the MathQA dataset (Amini et al., 2019) .

The MathQA dataset consists of about 37k math word problems, each with a corresponding list of multi-choice options and the corresponding operation sequence.

In this task, TP-N2F is deployed to generate the operation sequence given the question.

The generated operations are executed with the execution script from Amini et al. (2019) to select a multi-choice answer.

As there are about 30% noisy data (where the execution script returns the wrong answer when given the ground-truth program; see Sec. A.1 of the Appendix), we report both execution accuracy (of the final multi-choice answer after running the execution engine) and operation sequence accuracy (where the generated operation sequence must match the ground truth sequence exactly).

TP-N2F is compared to a baseline provided by the seq2prog model in Amini et al. (2019) , an LSTM-based seq2seq model with attention.

Our model outperforms both the original seq2prog, designated SEQ2PROG-orig, and the best reimplemented seq2prog after an extensive hyperparameter search, designated SEQ2PROG-best.

Table 1 presents the results.

To verify the importance of the TP-N2F encoder and decoder, we conducted experiments to replace either the encoder with a standard LSTM (denoted LSTM2TP) or the decoder with a standard attentional LSTM (denoted TP2LSTM).

We observe that both the TPR components of TP-N2F are important for achieving the observed performance gain relative to the baseline.

Generating Lisp programs requires sensitivity to structural information because Lisp code can be regarded as tree-structured.

Given a natural-language query, we need to generate code containing function calls with parameters.

Each function call is a relational tuple, which has a function as the relation and parameters as arguments.

We evaluate our model on the AlgoLisp dataset for this task and achieve state-of-the-art performance.

The AlgoLisp dataset (Polosukhin & Skidanov, 2018 ) is a program synthesis dataset.

Each sample contains a problem description, a corresponding Lisp program tree, and 10 input-output testing pairs.

We parse the program tree into a straight-line sequence of tuples (same style as in MathQA).

AlgoLisp provides an execution script to run the generated program and has three evaluation metrics: the accuracy of passing all test cases (Acc), the accuracy of passing 50% of test cases (50p-Acc), and the accuracy of generating an exactly matching program (M-Acc).

AlgoLisp has about 10% noisy data (details in the Appendix), so we report results both on the full test set and the cleaned test set (in which all noisy testing samples are removed).

TP-N2F is compared with an LSTM seq2seq with attention model, the Seq2Tree model in Polosukhin & Skidanov (2018) , and a seq2seq model with a pre-trained tree decoder from the Tree2Tree autoencoder (SAPS) reported in Bednarek et al. (2019) .

As shown in Table 2 , TP-N2F outperforms all existing models on both the full test set and the cleaned test set.

Ablation experiments with TP2LSTM and LSTM2TP show that, for this task, the TP-N2F Decoder is more helpful than TP-N2F Encoder.

This may be because lisp codes rely more heavily on structure representations.

To interpret the structure learned by the model, we extract the trained unbinding relation vectors from the TP-N2F Decoder and reduce the dimension of vectors via Principal Component Analysis.

Kmeans clustering results on the average vectors are presented in Figure 4 and Figure 5 (in Appendix A.6).

Results show that unbinding vectors for operators or functions with similar semantics tend to be close to each other.

For example, with 5 clusters in the MathQA dataset, arithmetic operators such as add, subtract, multiply, divide are clustered together, and operators related to square or volume of geometry are clustered together.

With 4 clusters in the AlgoLisp dataset, partial/lambda functions and sort functions are in one cluster, and string processing functions are clustered together.

Note that there is no direct supervision to inform the model about the nature of the operations, and the TP-N2F decoder has induced this role structure using weak supervision signals from question/operationsequence-answer pairs.

More clustering results are presented in the Appendix A.6.

N2F tasks include many different subtasks such as symbolic reasoning or semantic parsing (Kamath & Das, 2019; Cai & Lam, 2019; Liao et al., 2018; Amini et al., 2019; Polosukhin & Skidanov, 2018; Bednarek et al., 2019) .

These tasks require models with strong structure-learning ability.

TPR is a promising technique for encoding symbolic structural information and modeling symbolic reasoning in vector space.

TPR binding has been used for encoding and exploring grammatical structural information of natural language (Palangi et al., 2018; Huang et al., 2019) .

TPR unbinding has also been used to generate natural language captions from images (Huang et al., 2018) .

Some researchers use TPRs for modeling deductive reasoning processes both on a rule-based model and deep learning models in vector space (Lee et al., 2016; Smolensky et al., 2016; Schlag & Schmidhuber, 2018) .

However, none of these previous models takes advantage of combining TPR binding and TPR unbinding to learn structure representation mappings explicitly, as done in our model.

Although researchers are paying increasing attention to N2F tasks, most of the proposed models either do not encode structural information explicitly or are specialized to particular tasks.

Our proposed TP-N2F neural model can be applied to many tasks.

In this paper we propose a new scheme for neural-symbolic relational representations and a new architecture, TP-N2F, for formal-language generation from natural-language descriptions.

To our knowledge, TP-N2F is the first model that combines TPR binding and TPR unbinding in the encoderdecoder fashion.

TP-N2F achieves the state-of-the-art on two instances of N2F tasks, showing significant structure learning ability.

The results show that both the TP-N2F encoder and the TP-N2F decoder are important for improving natural-to formal-language generation.

We believe that the interpretation and symbolic structure encoding of TPRs are a promising direction for future work.

We also plan to combine large-scale deep learning models such as BERT with TP-N2F to take advantage of structure learning for other generation tasks.

In this section, we present details of the experiments of TP-N2F on the two datasets.

We present the implementation of TP-N2F on each dataset.

The MathQA dataset consists of about 37k math word problems ((80/12/8)% training/dev/testing problems), each with a corresponding list of multi-choice options and an straight-line operation sequence program to solve the problem.

An example from the dataset is presented in the Appendix A.4.

In this task, TP-N2F is deployed to generate the operation sequence given the question.

The generated operations are executed to generate the solution for the given math problem.

We use the execution script from Amini et al. (2019) to execute the generated operation sequence and compute the multi-choice accuracy for each problem.

During our experiments we observed that there are about 30% noisy examples (on which the execution script fails to get the correct answer on the ground truth program).

Therefore, we report both execution accuracy (the final multi-choice answer after running the execution engine) and operation sequence accuracy (where the generated operation sequence must match the ground truth sequence exactly).

The AlgoLisp dataset (Polosukhin & Skidanov, 2018 ) is a program synthesis dataset, which has 79k/9k/10k training/dev/testing samples.

Each sample contains a problem description, a corresponding Lisp program tree, and 10 input-output testing pairs.

We parse the program tree into a straight-line sequence of commands from leaves to root and (as in MathQA) use the symbol # i to indicate the result of the i th command (generated previously by the model).

A dataset sample with our parsed command sequence is presented in the Appendix A.4.

AlgoLisp provides an execution script to run the generated program and has three evaluation metrics: accuracy of passing all test cases (Acc), accuracy of passing 50% of test cases (50p-Acc), and accuracy of generating an exactly matched program (M-Acc).

AlgoLisp has about 10% noise data (where the execution script fails to pass all test cases on the ground truth program), so we report results both on the full test set and the cleaned test set (in which all noisy testing samples are removed).

We use d R , n R , d F , n F to indicate the TP-N2F encoder hyperparameters, the dimension of role vectors, the number of roles, the dimension of filler vectors and the number of fillers.

d Rel , d Arg , d P os indicate the TP-N2F decoder hyper-parameters, the dimension of relation vectors, the dimension of argument vectors, and the dimension of position vectors.

In the experiment on the MathQA dataset, we use n F = 150, n R = 50, d F = 30, d R = 20, d Rel = 20, d Arg = 10, d P os = 5 and we train the model for 60 epochs with learning rate 0.00115.

The reasoning module only contains one layer.

As most of the math operators in this dataset are binary, we replace all operators taking three arguments with a set of binary operators based on hand-encoded rules, and for all operators taking one argument, a padding symbol is appended.

For the baseline SEQ2PROG-orig, TP2LSTM and LSTM2TP, we use hidden size 100, single-direction, one-layer LSTM.

For the SEQ2PROG-best, we performed a hyperparameter search on the hidden size for both encoder and decoder; the best score is reported.

In the experiment on the AlgoLisp dataset, we use n F = 150, n R = 50, d F = 30, d R = 30, d Rel = 30, d Arg = 20, d P os = 5 and we train the model for 50 epochs with learning rate 0.00115.

We also use one-layer in the reasoning module like in MathQA.

For this dataset, most function calls take three arguments so we simply add padding symbols for those functions with fewer than three arguments.

A.2.1 TP-N2F ENCODER

Atten is the attention mechanism used in Luong et al. (2015) , which computes the dot product between h t input and each T t .

Then a linear function is used on the concatenation of h t input and the softmax scores on all dot products to generate H t .

The following equations show the attention mechanism:

score is the score function of the attention.

In this paper, the score function is dot product.

At each timestep t, the 2-step unbinding process described in Sec. 3.1.2 operates first on an encoding of the triple as a whole, H, using two unbinding vectors p i that are learned but fixed for all tuples.

This first unbinding gives an encoding of the two operator-argument bindings, B i .

The second unbinding operates on the B i , using a generated unbinding vector for the operator, r , giving encodings of the arguments, a i .

The generated unbinding vector for the operator, r , and the generated encodings of the arguments, a i , each produce a probability distribution over symbolic operator outputs Rel and symbolic argument outputs Arg i ; these probabilities are used in the cross-entropy loss function.

For generating a single symbolic output, the most-probable symbols are selected.

The dimensions are:

Question:

Consider a number a, compute factorial of a TP-N2F(correct): ( ??=,arg1,1 ) ( -,arg1,1 ) ( self,#1 ) ( *,#2,arg1 ) ( if,#0,1,#3 ) ( lambda1,#4 ) ( invoke1,#5,a ) LSTM(wrong): ( ??=,arg1,1 ) ( -,arg1,1 ) ( self,#1 ) ( *,#2,arg1 ) ( if,#0,1,#3 ) ( lambda1,#4 ) ( len,a ) ( invoke1,#5,#6 )

Question:

Given an array of numbers and numbers b and c, add c to elements of the product of elements of the given array and b, what is the product of elements of the given array and b?

TP-N2F(correct): ( partial, b,* ) ( partial1,c,+ ) ( map,a,#0 ) ( map,#2,#1 ) LSTM(wrong): ( partial1,b,+ ) ( partial1,c,+ ) ( map,a,#0 ) ( map,#2,#1 )

Question: You are given an array of numbers a and numbers b , c and d , let how many times you can replace the median in a with sum of its digits before it becomes a single digit number and b be the coordinates of one end and c and d be the coordinates of another end of segment e , your task is to find the length of segment e rounded down TP-N2F(correct): ( digits arg1 ) ( len #0 ) ( == #1 1 ) ( digits arg1 ) ( reduce #3 0 + ) ( self #4 ) ( + 1 #5 ) ( if #2 0 #6 ) ( lambda1 #7 ) ( sort a ) ( len a ) ( / #10 2 ) ( deref #9 #11 ) ( invoke1 #8 #12 ) ( -#13 c ) ( digits arg1 ) ( len #15 ) ( == #16 1 ) ( digits arg1 ) ( reduce #18 0 + ) ( self #19 ) ( + 1 #20 ) ( if #17 0 #21 ) ( lambda1 #22 ) ( sort a ) ( len a ) ( / #25 2 ) ( deref #24 #26 ) ( invoke1 #23 #27 ) ( -#28 c ) ( * #14 #29 ) ( -b d ) ( -b d ) ( * #31 #32 ) ( + #30 #33 ) ( sqrt #34 ) ( floor #35 ) LSTM(wrong): ( digits arg1 ) ( len #0 ) ( == #1 1 ) ( digits arg1 ) ( reduce #3 0 + ) ( self #4 ) ( + 1 #5 ) ( if #2 0 #6 ) ( lambda1 #7 ) ( sort a ) ( len a ) ( / #10 2 ) ( deref #9 #11 ) ( invoke1 #8 #12 c ) ( -#13 ) ( -b d ) ( -b d ) ( * #15 #16 ) ( * #14 #17 ) ( + #18 ) ( sqrt #19 ) ( floor #20 )

Question:

Given numbers a , b , c and e , let d be c , reverse digits in d , let a and the number in the range from 1 to b inclusive that has the maximum value when its digits are reversed be the coordinates of one end and d and e be the coordinates of another end of segment f , find the length of segment f squared TP-N2F(correct): ( digits c ) ( reverse #0 ) ( * arg1 10 ) ( + #2 arg2 ) ( lambda2 #3 ) ( reduce #1 0 #4 ) ( -a #5 ) ( digits c ) ( reverse #7 ) ( * arg1 10 ) ( + #9 arg2 ) ( lambda2 #10 ) ( reduce #8 0 #11 ) ( -a #12 ) ( * #6 #13 ) ( + b 1 ) ( range 0 #15 ) ( digits arg1 ) ( reverse #17 ) ( * arg1 10 ) ( + #19 arg2 ) ( lambda2 #20 ) ( reduce #18 0 #21 ) ( digits arg2 ) ( reverse #23 ) ( * arg1 10 ) ( + #25 arg2 ) ( lambda2 #26 ) ( reduce #24 0 #27 ) ( ?? #22 #28 ) ( if #29 arg1 arg2 ) ( lambda2 #30 ) ( reduce #16 0 #31 ) ( -#32 e ) ( + b 1 ) ( range 0 #34 ) ( digits arg1 ) ( reverse #36 ) ( * arg1 10 ) ( + #38 arg2 ) ( lambda2 #39 ) ( reduce #37 0 #40 ) ( digits arg2 ) ( reverse #42 ) ( * arg1 10 ) ( + #44 arg2 ) ( lambda2 #45 ) ( reduce #43 0 #46 ) ( ?? #41 #47 ) ( if #48 arg1 arg2 ) ( lambda2 #49 ) ( reduce #35 0 #50 ) ( -#51 e ) ( * #33 #52 ) ( + #14 #53 ) LSTM(wrong): ( -a d ) ( -a d ) ( * #0 #1 ) ( digits c ) ( reverse #3 ) ( * arg1 10 ) ( + #5 arg2 ) ( lambda2 #6 ) ( reduce #4 0 #7 ) ( -#8 e ) ( + b 1 ) ( range 0 #10 ) ( digits arg1 ) ( reverse #12 ) ( * arg1 10 ) ( + #14 arg2 ) ( lambda2 #15 ) ( reduce #13 0 #16 ) ( digits arg2 ) ( reverse #18 ) ( * arg1 10 ) ( + #20 arg2 ) ( lambda2 #21 ) ( reduce #19 0 #22 ) ( ?? #17 #23 ) ( if #24 arg1 arg2 ) ( lambda2 #25 ) ( reduce #11 0 #26 ) ( -#27 e ) ( * #9 #28 ) ( + #2 #29 )

A.6 UNBINDING RELATION VECTOR CLUSTERING

We run K-means clustering on both datasets with k = 3, 4, 5, 6 clusters and the results are displayed in Figure 4 and Figure 5 .

As described before, unbinding-vectors for operators or functions with similar semantics tend to be closer to each other.

For example, in the MathQA dataset, arithmetic operators such as add, subtract, multiply, divide are clustered together at middle, and operators related to geometry such as square or volume are clustered together at bottom left.

In AlgoLisp dataset, basic arithmetic functions are clustered at middle, and string processing functions are clustered at right.

<|TLDR|>

@highlight

In this paper, we propose a new encoder-decoder model based on Tensor Product Representations for Natural- to Formal-language generation, called TP-N2F.