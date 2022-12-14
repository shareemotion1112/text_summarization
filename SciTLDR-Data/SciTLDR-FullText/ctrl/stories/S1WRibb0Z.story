Deep neural networks are surprisingly efficient at solving practical tasks, but the theory behind this phenomenon is only starting to catch up with the practice.

Numerous works show that depth is the key to this efficiency.

A certain class of deep convolutional networks – namely those that correspond to the Hierarchical Tucker (HT) tensor decomposition – has been proven to have exponentially higher expressive power than shallow networks.

I.e. a shallow network of exponential width is required to realize the same score function as computed by the deep architecture.

In this paper, we prove the expressive power theorem (an exponential lower bound on the width of the equivalent shallow network) for a class of recurrent neural networks – ones that correspond to the Tensor Train (TT) decomposition.

This means that even processing an image patch by patch with an RNN can be exponentially more efficient than a (shallow) convolutional network with one hidden layer.

Using theoretical results on the relation between the tensor decompositions we compare expressive powers of the HT- and TT-Networks.

We also implement the recurrent TT-Networks and provide numerical evidence of their expressivity.

Deep neural networks solve many practical problems both in computer vision via Convolutional Neural Networks (CNNs) BID19 ; BID31 ; BID14 ) and in audio and text processing via Recurrent Neural Networks (RNNs) BID11 ; BID21 BID8 ).

However, although many works focus on expanding the theoretical explanation of neural networks success BID20 ; BID6 ; ), the full theory is yet to be developed.

One line of work focuses on expressive power, i.e. proving that some architectures are more expressive than others. showed the connection between Hierarchical Tucker (HT) tensor decomposition and CNNs, and used this connection to prove that deep CNNs are exponentially more expressive than their shallow counterparts.

However, no such result exists for Recurrent Neural Networks.

The contributions of this paper are three-fold.1.

We show the connection between recurrent neural networks and Tensor Train decomposition (see Sec. 4); 2.

We formulate and prove the expressive power theorem for the Tensor Train decomposition (see Sec. 5), which -on the language of RNNs -can be interpreted as follows: to (exactly) emulate a recurrent neural network, a shallow (non-recurrent) architecture of exponentially larger width is required; DISPLAYFORM0 Figure 1: Recurrent-type neural architecture that corresponds to the Tensor Train decomposition.

Gray circles are bilinear maps (for details see Section 4).

In this section, we review the known connections between tensor decompositions and deep learning and then show the new connection between Tensor Train decomposition and recurrent neural networks.

Suppose that we have a classification problem and a dataset of pairs {(X (b) , y (b) )} N b=1 .

Let us assume that each object X (b) is represented as a sequence of vectors DISPLAYFORM0 which is often the case.

To find this kind of representation for images, several approaches are possible.

The approach that we follow is to split an image into patches of small size, possibly overlapping, and arrange the vectorized patches in a certain order.

An example of this procedure is 4 1 3 2 5 … 6 We use lower-dimensional representations of {x k } d k=1 .

For this we introduce a collection of parameter dependent feature maps {f θ : R n → R} m =1 , which are organized into a representation map DISPLAYFORM1 A typical choice for such a map is DISPLAYFORM2 that is an affine map followed by some nonlinear activation σ.

In the image case if X was constructed using the procedure described above, the map f θ resembles the traditional convolutional maps -each image patch is projected by an affine map with parameters shared across all the patches, which is followed by a pointwise activation function.

Score functions considered in can be written in the form DISPLAYFORM3 where Φ(X) is a feature tensor, defined as DISPLAYFORM4 and W y ∈ R m×m×...m is a trainable weight tensor.

Inner product in Eq. (2) is just a total sum of the entry-wise product of Φ(X) and W y .

It is also shown that the hypothesis space of the form Eq. (2) has the universal representation property for m → ∞. Similar score functions were considered in BID23 ; BID29 .Storing the full tensor W y requires an exponential amount of memory, and to reduce the number of degrees of freedom one can use a tensor decompositions.

Various decompositions lead to specific network architectures and in this context, expressive power of such a network is effectively measured by ranks of the decomposition, which determine the complexity and a total number of degrees of freedom.

For the Hierarchical Tucker (HT) decomposition, proved the expressive power property, i.e. that for almost any tensor W y its HT-rank is exponentially smaller than its CPrank.

We analyze Tensor Train-Networks (TT-Networks), which correspond to a recurrent-type architecture.

We prove that these networks also have exponentially larger representation power than shallow networks (which correspond to the CP-decomposition).

In this section we briefly review all the necessary definitions.

As a d-dimensional tensor X we simply understand a multidimensional array: DISPLAYFORM0 To work with tensors it is convenient to use their matricizations, which are defined as follows.

Let us choose some subset of axes s = {i 1 , i 2 . . .

i ms } of X , and denote its compliment by t = {j 1 , j 2 . . .

j d−ms }, e.g. for a 4 dimensional tensor s could be {1, 3} and t is {2, 4}. Then matricization of X specified by (s, t) is a matrix DISPLAYFORM1 obtained simply by transposing and reshaping the tensor X into matrix, which in practice e.g. in Python, is performed using numpy.reshape function.

Let us now introduce tensor decompositions we will use later.

Canonical decomposition, also known as CANDECOMP/PARAFAC or CP-decomposition for short BID13 BID2 ), is defined as follows DISPLAYFORM0 The minimal r such that this decomposition exists is called the canonical or CP-rank of X .

We will use the following notation rank CP X = r. When rank CP X = 1 it can be written simply as DISPLAYFORM1 , which means that modes of X are perfectly separated from each other.

Note that storing all entries of a tensor X requires O(n d ) memory, while its canonical decomposition takes only O(dnr).

However, the problems of determining the exact CP-rank of a tensor and finding its canonical decomposition are NP-hard, and the problem of approximating a tensor by a tensor of lower CP-rank is ill-posed.

A tensor X is said to be represented in the Tensor Train (TT) format (Oseledets (2011)) if each element of X can be computed as follows DISPLAYFORM0 . . .

DISPLAYFORM1 where the tensors G k ∈ R r k−1 ×n k ×r k (r 0 = r d = 1 by definition) are the so-called TT-cores.

The element-wise minimal ranks r = (r 1 , . . .

r d−1 ) such that decomposition (5) exists are called TT-ranks rank T T X = r.

Note that for fixed values of i 1 , i 2 . . .

, i d , the right-hand side of Eq. FORMULA11 is just a product of matrices DISPLAYFORM2 Storing X in the TT-format requires O(dnr 2 ) memory and thus also achieves significant compression of the data.

Given some tensor X , the algorithm for finding its TT-decomposition is constructive and is based on a sequence of Singular Value Decompositions (SVDs), which makes it more numerically stable than CP-format.

We also note that when all the TT-ranks equal to each other rank T T X = (r, r, . . .

, r), we will sometimes write for simplicity rank T T X = r.

A further generalization of the TT-format leads to the so-called Hierarchical Tucker (HT) format.

The definition of the HT-format is a bit technical and requires introducing the dimension tree (Grasedyck, 2010, Definition 3.1) .

In the next section we will provide an informal introduction into the HT-format, and for more details, we refer the reader to BID9 ; BID10 BID12 .

To construct the tensorial networks we introduce bilinear and multilinear units, which perform a bilinear (multilinear) map of their inputs (see FIG1 for an illustration).

Suppose that x ∈ R n , y ∈ R m and G ∈ R n×m×k .

Then a bilinear unit G performs a bilinear map G :

Similarly, for DISPLAYFORM0 In the rest of this section, we describe how to compute the score functions l y (X) (see Eq.(1)) for each class label y, which then could be fed into the loss function (such as cross-entropy).

The architecture we propose to implement the score functions is illustrated on Fig. 1 .

For a vector r = (r 1 , r 2 , . . .

r d−1 ) of positive integers (rank hyperparameter) we define bilinear units DISPLAYFORM1 with r 0 = r d = 1.

Note that because r 0 = 1, the first unit G 1 is in fact just a linear map, and because r d = 1 the output of the network is just a number.

On a step k ≥ 2 the representation f θ (x k ) and output of the unit G k−1 of size r k are fed into the unit G k .

Thus we obtain a recurrent-type neural network with multiplicative connections and without non-linearities.

To draw a connection with the Tensor Train decomposition we make the following observation.

For each of the class labels y let us construct the tensor W y using the definition of TT-decomposition (Eq. FORMULA11 ) and taking {G k } d k=1 used for constructing l y (X) as its TT-cores.

Using the definition of the Eq. (3) we find that the score functions computed by the network from Fig. 1 are given by the formula DISPLAYFORM2 which is verified using Eq. FORMULA11 and Eq. (3).

Thus, we can conclude that the network presented on Fig. 1 realizes the TT-decomposition of the weight tensor.

We also note that the size of the output of the bilinear unit G k in the TT-Network is equal to r k , which means that the TT-ranks correspond to the width of the network.

Let us now consider other tensor decompositions of the weight tensors W y , construct corresponding network architectures, and compare their properties with the original TT-Network.

DISPLAYFORM3 Figure 4: Examples of networks corresponding to various tensor decompositions.

A network corresponding to the CP-decomposition is visualized on Fig. 4a .

Each multilinear unit G α is given by a summand in the formula Eq. (4), namely DISPLAYFORM4 Note that the output of each G α in this case is just a number, and in total there are rank CP W y multilinear units.

Their outputs are then summed up by the Σ node.

As before rank of the decomposition corresponds to the width of the network.

However, in this case the network is shallow, meaning that there is only one hidden layer.

On the Fig. 4b a network of other kind is presented.

Tensor decomposition which underlies it is the Hierarchical Tucker decomposition, and hence we call it the HT-Network.

It is constructed using a binary tree, where each node other than leaf corresponds to a bilinear unit, and leaves correspond to linear units.

Inputs are fed into leaves, and this data is passed along the tree to the root, which outputs a number.

Ranks, in this case, are just the sizes of the outputs of the intermediate units.

We will denote them by rank HT X .

These are networks considered in , where the expressive power of such networks was analyzed and was argued that they resemble traditional CNNs.

In general Hierarchical Tucker decomposition may be constructed using an arbitrary tree, but not much theory is known in general case.

Our main theoretical results are related to a comparison of the expressive power of these kinds of networks.

Namely, the question that we ask is as follows.

Suppose that we are given a TT-Network.

How complex would be a CP-or HT-Network realizing the same score function?

A natural measure of complexity, in this case, would be the rank of the corresponding tensor decomposition.

To make transitioning between tensor decompositions and deep learning vocabulary easier, we introduce the following table.

In this section we prove the expressive power theorem for the Tensor Train decomposition, that is we prove that given a random d-dimensional tensor in the TT format with ranks r and modes n, with probability 1 this tensor will have exponentially large CP-rank.

Note that the reverse result can not hold true since TT-ranks can not be larger than CP-ranks: rank T T X ≤ rank CP X .

To bound CP-rank of a tensor the following lemma is useful.

Lemma 1.

Let X i1i2...i d and rank CP X = r. Then for any matricization X (s,t) we have rank X (s,t) ≤ r, where the ordinary matrix rank is assumed.

Proof.

Proof is based on the following observation.

Let DISPLAYFORM0 d , be a CP-rank 1 tensor.

Note for any s, t rank A (s,t) = 1, because A (s,t) can be written as uw T for some u and w.

Then the statement of the lemma follows from the facts that matricization is a linear operation, and that for matrices rank(A + B) ≤ rank A + rank B.We use this lemma to provide a lower bound on the CP-rank in the theorem formulated below.

For example, suppose that we found some matricization of a tensor X which has matrix rank r. Then, by using the lemma we can estimate that rank CP X ≥ r.

Let us denote n = (n 1 , n 2 . . .

n d ).

Set of all tensors X with mode sizes n representable in TT-format with rank T T X ≤ r, for some vector of positive integers r (inequality is understood entry-wise) forms an irreducible algebraic variety BID27 ), which we denote by M r .

This means that M r is defined by a set of polynomial equations in R n1×n2...

n d , and that it can not be written as a union (not necessarily disjoint) of two proper non-empty algebraic subsets.

An example where the latter property does not hold would be the union of axes x = 0 and y = 0 in R 2 , which is an algebraic set defined by the equation xy = 0.

The main fact that we use about irreducible algebraic varieties is that any proper algebraic subset of them necessarily has measure 0 (Ilyashenko & Yakovenko FORMULA4 ).For simplicity let us assume that number of modes d is even, that all mode sizes are equal to n, and we consider M r with r = (r, r . . .

r), so for any X ∈ M r we have rank T T X ≤ (r, r, . . .

, r), entry-wise.

As the main result we prove the following theorem Theorem 1.

Suppose that d = 2k is even.

Define the following set DISPLAYFORM1 where q = min{n, r}. DISPLAYFORM2 where µ is the standard Lebesgue measure on M r .Proof.

Our proof is based on applying Lemma 1 to a particular matricization of X .

Namely, we would like to show that for s = {1, 3, . . .

d − 1}, t = {2, 4, . . .

d} the following set show that µ(B (s,t) ) = 0 we need to find at least one X such that rank X (s,t) ≥ q d 2 .

This follows from the fact that because B (s,t) is an algebraic subset of the irreducible algebraic variety M r , it is either equal to M r or has measure 0, as was explained before.

One way to construct such tensor is as follows.

Let us define the following tensors: DISPLAYFORM3 DISPLAYFORM4 where δ iα is the Kronecker delta symbol: DISPLAYFORM5 The TT-ranks of the tensor X defined by the TT-cores (9) are equal to rank T T X = (r, 1, r, . . .

, r, 1, r).

The following identity holds true for any values of indices such that i k = 1, . . .

, q, k = 1, . . . , d. DISPLAYFORM0 The last equality holds because DISPLAYFORM1 where I is the identity matrix of size q d/2 × q d/2 where q = min{n, r}.To summarize, we found an example of a tensor X such that rank T T X ≤ r and the matricization X (i1,i3,...,i d−1 ),(i2,i4,...,i d ) has a submatrix being equal to the identity matrix of size q d/2 × q d/2 , and hence rank X (i1,i3,... DISPLAYFORM2 This means that the canonical rank CP X ≥ q d/2 which concludes the proof.

In other words, we have proved that for all TT-Networks besides negligible set, the equivalent CPNetwork will have exponentially large width.

To compare the expressive powers of the HT-and TT-Networks we use the following theorem (Grasedyck, 2010, Section 5.3.2) .

Theorem 2.

For any tensor X the following estimates hold.• If rank T T X ≤ r, then rank HT X ≤ r 2 .•

If rank HT X ≤ r, then rank T T X ≤ r log 2 (d) /2 .It is also known that this bounds are sharp (see BID1 ).

Thus, we can summarize all the results in the following Table 2 .

Table 2 : Comparison of the expressive power of various networks.

Given a network of width r, specified in a column, rows correspond to the upper bound on the width of the equivalent network of other type (we assume that the number of feature maps m is greater than the width of the network r).TT-Network HT-Network CP-Network TT-Network r r DISPLAYFORM3 Example that requires exponential width in a shallow network A particular example used to prove Theorem 1 is not important per se since the Theorem states that TT is exponentially more expressive than CP for almost any tensor (for a set of tensors of measure one).

However, to illustrate how the Theorem translates into neural networks consider the following example.

Consider the task of getting d input vectors with n elements each and aiming to compute the following measure of similarity between x 1 , . . .

, x d/2 and x d/2+1 , . . .

, x d : DISPLAYFORM4 We argue that it can be done with a TT-Network of width n by using the TT-tensor X defined in the proof of Theorem 1 and feeding the input vectors in the following order: DISPLAYFORM5 The CP-network representing the same function will have n d/2 terms (and hence n d/2 width) and will correspond to expanding brackets in the expression (12).The case of equal TT-cores In analogy to the traditional RNNs we can consider a special class of Tensor Trains with the property that all the intermediate TT-cores are equal to each other: G 2 = G 3 = · · · = G d−1 , which allows for processing sequences of varied length.

We hypothesize that for this class exactly the same result as in Theorem 1 holds i.e. if we denote the variety of Tensor Trains with equal TT-cores by M eq r , we believe that the following hypothesis holds true: Hypothesis 1.

Theorem 1 is also valid if M r is replaced by M eq r .To prove it we can follow the same route as in the proof of Theorem 1.

While we leave finding an analytical example of a tensor with the desired property of rank maximality to a future work, we have verified numerically that randomly generated tensors X from M eq r with d = 6, n ranging from 2 to 10 and r ranging from 2 to 20 (we have checked 1000 examples for each possible combination) indeed satisfy rank CP X ≥ q

In this section, we experimentally check if indeed -as suggested by Theorem 1 -the CP-Networks require exponentially larger width compared to the TT-Networks to fit a dataset to the same level of accuracy.

This is not clear from the theorem since for natural data, functions that fit this data may lay in the neglectable set where the ranks of the TT-and CP-networks are related via a polynomial function (in contrast to the exponential relationship for all function outside the neglectable set).

Other possible reasons why the theory may be disconnected with practice are optimization issues (although a certain low-rank tensor exists, we may fail to find it with SGD) and the existence of the feature maps, which were not taken into account in the theory.

To train the TT-and CP-Networks, we implemented them in TensorFlow BID0 ) and used Adam optimizer with batch size 32 and learning rate sweeping across {4e-3, 2e-3, 1e-3, 5e-4} values.

Since we are focused on assessing the expressivity of the format (in contrast to its sensitivity to hyperparameters), we always choose the best performing run according to the training loss.

For the first experiment, we generate two-dimensional datasets with Scikit-learn tools 'moons' and 'circles' BID25 ) and for each training example feed the two features as two patches into the TT-Network (see FIG4 ).

This example shows that the TT-Networks can implement nontrivial decision boundaries.

For the next experiments, we use computer vision datasets MNIST BID18 ) and CIFAR-10 ( BID17 ).

MNIST is a collection of 70000 handwritten digits, CIFAR-10 is a dataset of 60000 natural images which are to be classified into 10 classes such as bird or cat.

We feed raw pixel data into the TT-and CP-Networks (which extract patches and apply a trainable feature map to them, see Section 2).

In our experiments we choose patch size to be 8 × 8, feature maps to be affine maps followed by the ReLU activation and we set number of such feature maps to 4.

For MNIST, both TT-and CP-Networks show reasonable performance (1.0 train accuracy, 0.95 test accuracy without regularizers, and 0.98 test accuracy with dropout 0.8 applied to each patch) even with ranks less than 5, which may indicate that the dataset is too simple to draw any conclusion, but serves as a sanity check.

We report the training accuracy for CIFAR-10 on Fig. 6 .

Note that we did not use regularizers of any sort for this experiment since we wanted to compare expressive power of networks (the best test accuracy we achieved this way on CIFAR-10 is 0.45 for the TT-Network and 0.2 for the CPNetwork).

On practice, the expressive power of the TT-Network is only polynomially better than that of the CP-network (Fig. 6) , probably because of the reasons discussed above.

A large body of work is devoted to analyzing the theoretical properties of neural networks (Cybenko (1989); BID15 ; BID28 ).

Recent studies focus on depth efficiency BID26 ; BID22 ; BID7 ; BID30 ), in most cases providing worst-case guaranties such as bounds between deep and shallow networks width.

Two works are especially relevant since they analyze depth efficiency from the viewpoint of tensor decompositions: expressive power of the Hierarchical Tucker decomposition ) and its generalization to handle activation functions such as ReLU (Cohen Figure 6 : Train accuracy on CIFAR-10 for the TT-and CP-Networks wrt rank of the decomposition and total number of parameters (feature size 4 was used).

Note that with rank increase the CPNetworks sometimes perform worse due to optimization issues.

& Shashua FORMULA1 ).

However, all of the works above focus on feedforward networks, while we tackle recurrent architectures.

The only other work that tackles expressivity of RNNs is the concurrent work that applies the TT-decomposition to explicitly modeling high-order interactions of the previous hidden states and analyses the expressive power of the resulting architecture BID33 .

This work, although very related to ours, analyses a different class of recurrent models.

Models similar to the TT-Network were proposed in the literature but were considered from the practical point of view in contrast to the theoretical analyses provided in this paper.

BID23 ; BID29 proposed a model that implements Eq. (2), but with a predefined (not learnable) feature map Φ. BID32 explored recurrent neural networks with multiplicative connections, which can be interpreted as the TT-Networks with bilinear maps that are shared G k = G and have low-rank structure imposed on them.

In this paper, we explored the connection between recurrent neural networks and Tensor Train decomposition and used it to prove the expressive power theorem, which states that a shallow network of exponentially large width is required to mimic a recurrent neural network.

The downsides of this approach is that it provides worst-case analysis and do not take optimization issues into account.

In the future work, we would like to address the optimization issues by exploiting the Riemannian geometry properties of the set of TT-tensors of fixed rank and extend the analysis to networks with non-linearity functions inside the recurrent connections (as was done for CNNs in ).

<|TLDR|>

@highlight

We prove the exponential efficiency of recurrent-type neural networks over shallow networks.

@highlight

The authors compare the complexity of tensor train networks with networks structured by CP decomposition