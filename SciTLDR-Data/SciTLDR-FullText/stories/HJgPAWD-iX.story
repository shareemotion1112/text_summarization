Deep neural networks have shown incredible performance for inference tasks in a variety of domains.

Unfortunately, most current deep networks are enormous cloud-based structures that require significant storage space, which limits scaling of deep learning as a service (DLaaS) and use for on-device augmented intelligence.

This paper finds algorithms that directly use lossless compressed representations of deep feedforward networks (with synaptic weights drawn from discrete sets), to perform inference without full decompression.

The basic insight that allows less rate than naive approaches is the recognition that the bipartite graph layers of feedforward networks have a kind of permutation invariance to the labeling of nodes, in terms of inferential operation and that the inference operation depends locally on the edges directly connected to it.

We also provide experimental results of our approach on the MNIST dataset.

Deep learning has achieved incredible performance for inference tasks such as speech recognition, image recognition, and natural language processing.

Most current deep neural networks, however, are enormous cloud-based structures that are too large and too complex to perform fast, energyefficient inference on device or for scaling deep learning as a service (DLaaS).

Compression, with the capability of providing inference without full decompression, is important.

Universal source coding for feedforward deep networks having synaptic weights drawn from finite sets that essentially achieve the entropy lower bound were introduced in BID0 .

Here, we provide-for the first time-an algorithm that directly uses these compressed representations for inference tasks without complete decompression.

Structures that can represent information near the entropy bound while also allowing efficient operations on them are called succinct structures (2; 3; 4).

Thus, we provide a succinct structure for feedforward neural networks, which may fit on-device and enable scaling of DLaaS.Related Work: There has been recent interest in compact representations of neural networks (5; 6; 7; 8; 9; 10; 11; 12; 13; 14) .

While most of these algorithms are lossy, we provide an efficient lossless algorithm, which can be used on top of any lossy algorithm that quantizes or prunes network weights; prior work on lossless compression of neural networks either used Huffman coding in a way that did not exploit invariances or was not succinct and required full decompression for inference.

The proposed algorithm builds on the sublinear entropy-achieving representation in (1) but is the first time succinctness-the further ability to perform inference with negligible space needed for partial decompression-has been attempted or achieved.

Our inference algorithm is similar to arithmetic decoding and so computational performance is also governed by efficient implementations of arithmetic coding.

Efficient high-throughput implementations of arithmetic coding/decoding have been developed for video, e.g. as part of the H.264/AVC and HEVC standards (15; 16).

Let us describe the neural network model considered in (1) which will be used here to develop succinct structures of deep neural networks.

In a feedforward neural network, each node j com- putes an activation function g(·) applied to the weighted sum of its inputs, which we can note is a permutation-invariant function: DISPLAYFORM0 , for any permutation π.

Consider a feedforward neural network with K − 1 hidden layers where each node contains N nodes (for notational convenience) such that the nodes in all the K − 1 hidden layers are indistinguishable from each other (when edges are ignored) but the nodes in the input and output layers are labeled and can be distinguished.

There is an edge of color i, i = 0, . . .

, m, between any two nodes from two different layers independently with probability p i , where p 0 is the probability of no edge.

Consider a substructure: partially-labeled bipartite graphs, see FIG1 , which consists of two sets of vertices containing N vertices each with one of the sets containing labeled vertices and the other set containing unlabeled vertices.

An edge of color i exists between any two nodes taken one from each set with probability p i , i = 0, . . .

, m where p 0 is the probability of no edge.

Refer to (1) for detailed discussion on the structure.

To construct the K-layer neural network, think of it as made of a partially-labeled bipartite graph for the first two layers but then each time the nodes of an unlabeled layer are connected, we treat it as a labeled layer, based on its connection to the previous labeled layer (i.e. we can label the unlabeled nodes based on the nodes of the previous layer it is connected to), and iteratively complete the K-layer neural network.

First, we consider the succinct representation of a partially labeled bipartite graph, followed by that of a K-layered neural network.

Alg.

1 is an inference algorithm for a partially-labeled bipartite graph with input to the graph X and output Y .

Later we use this algorithm to make inferences in a K-layered neural network where outputs of unlabeled layers correspond to outputs of a hidden layer.

The optimally compressed representation of a partially-labeled bipartite graph produced by (1, Alg.1) is taken as an input by Alg.

1, in addition to the input X to the graph, and the output Y of the graph is given out.

If the graph has N nodes in each layer, then only an additional O(N ) bits of dynamic space is required by Alg.

1 for the inference task while it takes O(N 2 ) bits to store the representation and hence the structure in succinct as discussed below.

Lemma 1.

Output Y obtained from Alg.

1 is a permutation ofỸ , the output from the uncompressed neural network representation.

Proof.

Say, we have an m × 1 vector X to be multiplied with an m × n weight matrix W , to get the outputỸ , an n × 1 vector.

Then,Ỹ = W T X, and so the jth element ofỸ , DISPLAYFORM0 In Alg.

1, while traversing a particular depth i, we multiply all Y j s with X i W i,j and hence when we reach depth N , we get the Y vector as required.

The change in permutation ofỸ with respect to Y is because while compressing W , we do not encode the permutation of the columns, retaining the row permutation.

Proof.

The major dynamic space requirement is for decoding of individual nodes, and the queue, Q. Clearly, the space required for Q, is much more than the space required for decoding a single node.

We show the expected space complexity corresponding to Q is less than or equal to 2(m + 1)N (1 + 2 log 2 ( m+2 m+1 )) using Elias-Gamma integer codes for each entry in Q. Note that Q has nodes from at most two consecutive depths, and since only the child nodes of non-zero nodes are encoded, and the number of non-zero nodes at any depth is less than N , we can have a maximum of 2(m + 1)N nodes encoded in Q. Let α 0 , ..., α k be the non-zero tree nodes at some depth d of the tree, where k = (m+1)N .

Let S be the total space required to store Q. Using integer codes, we can encode any positive number x in 2 log 2 (x) + 1 bits, and to allow 0, we need 2 log 2 (x + 1) + 1 bits.

Thus, the Set i = 0.

Set f = the first element obtained after dequeuing Q.

while i ≤ m and f > 0 do 7: decode the child node of f corresponding to color i and store it as c.

Encode c back in L 1 .

Enqueue c in Q.

Add x l × w i to each of y j to y (j+c) .

Add c to j.

if j = 1, at least one non-zero node has been processed at the current depth then 11: DISPLAYFORM0 end if 13: end while 14: end while 15: Update the Y vector using the required activation function. .

Thus, the structure is succinct.

Now consider the structure of the K-layered neural network as in Sec. 2 and provide its succinct representation.

The extra dynamic space for K-layers remains the same as for 2-layers as described in Alg.

1 as inference is done one layer at a time.

Theorem 3.

The compressed structure obtained by the iterative use of (1, Alg.

1) is succinct.

We trained a feedforward neural network of dimension 784 × 50 × 50 × 50 × 50 × 10 on the MNIST dataset using gradient descent algorithm to get 98.4% accuracy on the test data.

Network weights were quantized using a uniform quantizer into 33 steps to get a network with an accuracy of 97.5% on the training data and an accuracy of 93.48% on the test data.

The weight matrices from the second to the last layer were rearranged based on the weight matrices corresponding to the previous layers as needed for Alg.

1 to work.

These matrices, except the last matrix connected to the output, were compressed using (1, Alg. 1) to get the compressed network, and arithmetic coding was implemented by modification of an existing implementation.

The compressed network performs exactly as the quantized network as it should, since we compress losslessly.

We observe that the extra memory required for inference is negligible compared to the size of the compressed network.

Detailed results from the experiment and dynamic space requirements are described in TAB0 , where H(p) is the empirical entropy calculated from the weight matrices.

@highlight

This paper finds algorithms that directly use lossless compressed representations of deep feedforward networks, to perform inference without full decompression.