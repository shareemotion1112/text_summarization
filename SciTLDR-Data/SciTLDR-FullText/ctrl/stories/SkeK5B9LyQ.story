Deep learning (DL) has in recent years been widely used in natural language processing (NLP) applications due to its superior performance.

However, while natural languages are rich in grammatical structure, DL has not been able to explicitly represent and enforce such structures.

This paper proposes a new architecture to bridge this gap by exploiting tensor product representations (TPR), a structured neural-symbolic framework developed in cognitive science over the past 20 years, with the aim of integrating DL with explicit language structures and rules.

We call it the Tensor Product Generation Network (TPGN), and apply it to image captioning.

The key ideas of TPGN are: 1) unsupervised learning of role-unbinding vectors of words via a TPR-based deep neural network, and 2) integration of TPR with typical DL architectures including Long Short-Term Memory (LSTM) models.

The novelty of our approach lies in its ability to generate a sentence and extract partial grammatical structure of the sentence by using role-unbinding vectors, which are obtained in an unsupervised manner.

Experimental results demonstrate the effectiveness of the proposed approach.

Deep learning is an important tool in many current natural language processing (NLP) applications.

However, language rules or structures cannot be explicitly represented in deep learning architectures.

The tensor product representation developed in BID22 ; BID24 has the potential of integrating deep learning with explicit rules (such as logical rules, grammar rules, or rules that summarize real-world knowledge).

This paper develops a TPR approach for deep-learning-based NLP applications, introducing the Tensor Product Generation Network (TPGN) architecture.

To demonstrate the effectiveness of the proposed architecture, we apply it to a important NLP application: image captioning.

A TPGN model generates natural language descriptions via learned representations.

The representations learned in the TPGN can be interpreted as encoding grammatical roles for the words being generated.

This layer corresponds to the role-encoding component of a general, independentlydeveloped architecture for neural computation of symbolic functions, including the generation of linguistic structures.

The key to this architecture is the notion of Tensor Product Representation (TPR), in which vectors embedding symbols (e.g., lives, frodo) are bound to vectors embedding structural roles (e.g., verb, subject) and combined to generate vectors embedding symbol structures ([frodo lives]).

TPRs provide the representational foundations for a general computational architecture called Gradient Symbolic Computation (GSC), and applying GSC to the task of natural language generation yields the specialized architecture defining the model presented here.

The generality of GSC means that the results reported here have implications well beyond the particular tasks we address here.

The paper is organized as follows.

Section 2 discusses related work.

In Section 3, we review the basics of tensor product representation.

Section 4 presents the rationale for our proposed architecture.

Section 5 describes our proposed model in detail.

In Section 6, we present our experimental results.

Finally, Section 7 concludes the paper.

Deep learning plays a dominant role in many NLP applications due to its exceptional performance.

Hence, we focus on recent deep-learning-based literature for an important NLP application, i.e., image captioning.

This work follows a great deal of recent caption-generation literature in exploiting end-to-end deep learning with a CNN image-analysis front end producing a distributed representation that is then used to drive a natural-language generation process, typically using RNNs BID17 ; BID28 ; BID5 ; BID3 ; BID6 ; BID11 ; BID12 BID21 .

Our grammatical interpretation of the structural roles of words in sentences makes contact with other work that incorporates deep learning into grammatically-structured networks BID26 ; BID15 ; BID14 ; BID1 ; BID29 ; BID16 ; BID25 BID21 .

Here, the network is not itself structured to match the grammatical structure of sentences being processed; the structure is fixed, but is designed to support the learning of distributed representations that incorporate structure internal to the representations themselves -filler/role structure.

TPRs are also used in NLP in BID18 but there the representation of each individual input word is constrained to be a literal TPR filler/role binding.

(The idea of using the outer product to construct internal representations was also explored in BID8 .)

Here, by contrast, the learned representations are not themselves constrained, but the global structure of the network is designed to display the somewhat abstract property of being TPR-capable: the architecture uses the TPR unbinding operation of the matrix-vector product to extract individual words for sequential output.

Tensor product representation (TPR) is a general framework for embedding a space of symbol structures S into a vector space.

This embedding enables neural network operations to perform symbolic computation, including computations that provide considerable power to symbolic NLP systems BID24 BID23 ).

Motivated by these successful examples, we are inspired to extend the TPR to the challenging task of learning image captioning.

And as a by-product, the symbolic character of TPRs makes them amenable to conceptual interpretation in a way that standard learned neural network representations are not.

A particular TPR embedding is based in a filler/role decomposition of S .

A relevant example is when S is the set of strings over an alphabet {a, b, . . .}.

One filler/role decomposition deploys the positional roles {r k }, k ∈ N, where the filler/role binding a/r k assigns the 'filler' (symbol) a to the k th position in the string.

A string such as abc is uniquely determined by its filler/role bindings, which comprise the (unordered) set B(abc) = {b/r 2 , a/r 1 , c/r 3 }.

Reifying the notion role in this way is key to TPR's ability to encode complex symbol structures.

Given a selected filler/role decomposition of the symbol space, a particular TPR is determined by an embedding that assigns to each filler a vector in a vector space V F ∼ = R d F , and a second embedding that assigns to each role a vector in a space DISPLAYFORM0 The vector embedding a symbol a is denoted by f a and is called a filler vector; the vector embedding a role r k is r k and called a role vector.

The TPR for abc is then the following 2-index tensor in DISPLAYFORM1 DISPLAYFORM2 where ⊗ denotes the tensor product.

The tensor product is a generalization of the vector outer product that is recursive; recursion is exploited in TPRs for, e.g., the distributed representation of trees, the neural encoding of formal grammars in connection weights, and the theory of neural computation of recursive symbolic functions.

Here, however, it suffices to use the outer product; using matrix notation we can write (1) as: DISPLAYFORM3 Generally, the embedding of any symbol structure S ∈ S is {f i ⊗ r i | f i /r i ∈ B(S)}; here: BID22 ; BID24 ).

DISPLAYFORM4 A key operation on TPRs, central to the work presented here, is unbinding, which undoes binding.

Given the TPR in (2), for example, we can unbind r 2 to get f b ; this is achieved simply by f b = S abc u 2 .

Here u 2 is the unbinding vector dual to the binding vector r 2 .

To make such exact unbinding possible, the role vectors should be chosen to be linearly independent. (In that case the unbinding vectors are the rows of the inverse of the matrix containing the binding vectors as columns, so that r 2 · u 2 = 1 while r k · u 2 = 0 for all other role vectors r k = r 2 ; this entails that S abc u 2 = b, the filler vector bound to r 2 .

Replacing the matrix inverse with the pseudo-inverse allows approximate unbinding when the role vectors are not linearly independent).Figure 1: Architecture of TPGN, a TPR-capable generation network.

" ×" denotes the matrix-vector product.

In this work we propose an approach to network architecture design we call the TPR-capable method.

The architecture we use (see Fig. 1 ) is designed so that TPRs could, in theory, be used within the architecture to perform the target task -here, generating a caption one word at a time.

Unlike previous work where TPRs are hand-crafted, in our work, end-to-end deep learning will induce representations which the architecture can use to generate captions effectively.

In this section, we consider the problem of image captioning.

As shown in Fig. 1 , our proposed system is denoted by N , which is from "N" in "TPGN".

The input of N is an image feature vector v and the output of N is a caption.

The image feature vector v is extracted from a given image by a pre-trained CNN.

The first part of our system N is a sentence-encoding subnetwork S which maps v to a representation S which will drive the entire caption-generation process; S contains all the image-specific information for producing the caption. (We will call a caption a "sentence" even though it may in fact be just a noun phrase.)If S were a TPR of the caption itself, it would be a matrix (or 2-index tensor) S which is a sum of matrices, each of which encodes the binding of one word to its role in the sentence constituting the caption.

To serially read out the words encoded in S, in iteration 1 we would unbind the first word from S, then in iteration 2 the second, and so on.

As each word is generated, S could update itself, for example, by subtracting out the contribution made to it by the word just generated; S t denotes the value of S when word w t is generated.

At time step t we would unbind the role r t occupied by word w t of the caption.

So the second part of our system N -the unbinding subnetwork U -would generate, at iteration t, the unbinding vector u t .

Once U produces the unbinding vector u t , this vector would then be applied to S to extract the symbol f t that occupies word t's role; the symbol represented by f t would then be decoded into word w t by the third part of N , i.e., the lexical decoding subnetwork L, which outputs x t , the 1-hot-vector encoding of w t .Recalling that unbinding in TPR is achieved by the matrix-vector product, the key operation in generating w t is thus the unbinding of r t within S, which amounts to simply: DISPLAYFORM0 This matrix-vector product is denoted " ×" in Fig. 1 .Thus the system N of Fig. 1is TPR-capable.

This is what we propose as the Tensor-Product Generation Network (TPGN) architecture.

The learned representation S will not be proven to literally be a TPR, but by analyzing the unbinding vectors u t the network learns, we will gain insight into the process by which the learned matrix S gives rise to the generated caption.

What type of roles might the unbinding vectors be unbinding?

A TPR for a caption could in principle be built upon positional roles, syntactic/semantic roles, or some combination of the two.

In the caption a man standing in a room with a suitcase, the initial a and man might respectively occupy the positional roles of POS(ITION) 1 and POS 2 ; standing might occupy the syntactic role of VERB; in the role of SPATIAL-P(REPOSITION); while a room with a suitcase might fill a 5-role schema DET(ERMINER) 1 N(OUN) 1 P DET 2 N 2 .

In fact we will see evidence below that our network learns just this kind of hybrid role decomposition.

What form of information does the sentence-encoding subnetwork S need to encode in S?

Continuing with the example of the previous paragraph, S needs to be some approximation to the TPR summing several filler/role binding matrices.

In one of these bindings, a filler vector f a -which the lexical subnetwork L will map to the article a -is bound (via the outer product) to a role vector r POS1 which is the dual of the first unbinding vector produced by the unbinding subnetwork U: u POS1 .

In the first iteration of generation the model computes S 1 u POS1 = f a , which L then maps to a. Analogously, another binding approximately contained in S 2 is f man r POS2 .

There are corresponding bindings for the remaining words of the caption; these employ syntactic/semantic roles.

One example is f standing r V .

At iteration 3, U decides the next word should be a verb, so it generates the unbinding vector u V which when multiplied by the current output of S, the matrix S 3 , yields a filler vector f standing which L maps to the output standing.

S decided the caption should deploy standing as a verb and included in S the binding f standing r V .

It similarly decided the caption should deploy in as a spatial preposition, including in S the binding f in r SPATIAL-P ; and so on for the other words in their respective roles in the caption.

The unbinding subnetwork U and the sentence-encoding network S of Fig. 1 are each implemented as (1-layer, 1-directional) LSTMs (see Fig. 2) ; the lexical subnetwork L is implemented as a linear transformation followed by a softmax operation.

In the equations below, the LSTM variables internal to the S subnet are indexed by 1 (e.g., the forget-, input-, and output-gates are respectivelyf 1 ,î 1 ,ô 1 ) while those of the unbinding subnet U are indexed by 2.Thus the state updating equations for S are, for t = 1, · · · , T = caption length: DISPLAYFORM0 c1,t =f1,t c1,t−1 +î1,t g1,tSt =ô1,t σ h (c1,t)wheref 1,t ,î 1,t ,ô 1,t , g 1,t , c 1, DISPLAYFORM1 is the (element-wise) logistic sigmoid function; σ h (·) is the hyperbolic tangent function; the operator denotes the Hadamard (elementwise) product; DISPLAYFORM2 d×d×d×d .

For clarity, biases -included throughout the model -are omitted from all equations in this paper.

The initial stateŜ 0 is initialized by: DISPLAYFORM3 Figure 2: The sentence-encoding subnet S and the unbinding subnet U are inter-connected LSTMs; v encodes the visual input while the x t encode the words of the output caption.where v ∈ R 2048 is the vector of visual features extracted from the current image by ResNet BID9 ) andv is the mean of all such vectors; C s ∈ R d×d×2048 .

On the output side, x t ∈ R V is a 1-hot vector with dimension equal to the size of the caption vocabulary, V , and W e ∈ R d×V is a word embedding matrix, the i-th column of which is the embedding vector of the i-th word in the vocabulary; it is obtained by the Stanford GLoVe algorithm with zero mean BID20 ).

x 0 is initialized as the one-hot vector corresponding to a "start-of-sentence" symbol.

For U in Fig. 1 , the state updating equations are: DISPLAYFORM4 c2,t =f2,t c2,t−1 +î2,t g2,t (15) pt =ô2,t σ h (c2,t)where DISPLAYFORM5 The initial state p 0 is the zero vector.

The dimensionality of the crucial vectors shown in Fig. 1 , u t and f t , is increased from DISPLAYFORM6 DISPLAYFORM7 Here DISPLAYFORM8 , the output of the unbinding subnetwork U, is computed as in Eq. FORMULA2 , where DISPLAYFORM9 2 ×d is U's output weight matrix.

DISPLAYFORM10 Finally, the lexical subnetwork L produces a decoded word x t ∈ R V by DISPLAYFORM11 where σ s (·) is the softmax function and W x ∈ R V ×d 2 is the overall output weight matrix.

Since W x plays the role of a word de-embedding matrix, we can set DISPLAYFORM12 where W e is the word-embedding matrix.

Since W e is pre-defined, we directly set W x by Eq. FORMULA3 without training L through Eq. (19).

Note that S and U are learned jointly through the end-to-end training.6 EXPERIMENTAL RESULTS

To evaluate the performance of our proposed architecture, we use the COCO dataset (COCO FORMULA2 ).

The COCO dataset contains 123,287 images, each of which is annotated with at least 5 captions.

We use the same pre-defined splits as BID11 ; BID9 : 113,287 images for training, 5,000 images for validation, and 5,000 images for testing.

We use the same vocabulary as that employed in BID9 , which consists of 8,791 words.

For the CNN of Fig. 1 , we used ResNet-152 BID10 ), pretrained on the ImageNet dataset.

The feature vector v has 2048 dimensions.

Word embedding vectors in W e are downloaded from the web BID20 ).

The model is implemented in TensorFlow BID0 ) with the default settings for random initialization and optimization by backpropagation.

In our experiments, we choose d = 25 (where d is the dimension of vector p t ).

The dimension of S t is 625 × 625 (whileŜ t is 25 × 25); the vocabulary size V = 8, 791; the dimension of u t and f t is d 2 = 625.

The main evaluation results on the MS COCO dataset are reported in TAB0 .

The widely-used BLEU BID19 ), METEOR BID2 ), and CIDEr BID27 ) metrics are reported in our quantitative evaluation of the performance of the proposed schemes.

In evaluation, our baseline is the widely used CNN-LSTM captioning method originally proposed in BID28 .

For comparison, we include results in that paper in the first line of TAB0 .

We also re-implemented the model using the latest ResNet feature and report the results in the second line of TAB0 .

Our re-implementation of the CNN-LSTM method matches the performance reported in BID9 , showing that the baseline is a state-of-the-art implementation.

As shown in TAB0 , compared to the CNN-LSTM baseline, the proposed TPGN significantly outperforms the benchmark schemes in all metrics across the board.

The improvement in BLEU-n is greater for greater n; TPGN particularly improves generation of longer subsequences.

The results clearly attest to the effectiveness of the TPGN architecture.

In this paper, we proposed a new Tensor Product Generation Network (TPGN) for natural language generation and related tasks.

The model has a novel architecture based on a rationale derived from the use of Tensor Product Representations for encoding and processing symbolic structure through neural network computation.

In evaluation, we tested the proposed model on captioning with the MS COCO dataset, a large-scale image captioning benchmark.

Compared to widely adopted LSTM-based models, the proposed TPGN gives significant improvements on all major metrics including METEOR, BLEU, and CIDEr.

Moreover, we observe that the unbinding vectors contain important grammatical information.

Our findings in this paper show great promise of TPRs.

In the future, we will explore extending TPR to a variety of other NLP tasks.

<|TLDR|>

@highlight

This paper is intended to develop a tensor product representation approach for deep-learning-based natural language processinig applications.