Several state of the art convolutional networks rely on inter-connecting different layers to ease the flow of information and gradient between their input and output layers.

These techniques have enabled practitioners to successfully train deep convolutional networks with hundreds of layers.

Particularly, a novel way of interconnecting layers was introduced as the Dense Convolutional Network (DenseNet) and has achieved state of the art performance on relevant image recognition tasks.

Despite their notable empirical success, their theoretical understanding is still limited.

In this work, we address this problem by analyzing the effect of layer interconnection on the overall expressive power of a convolutional network.

In particular, the  connections used in DenseNet are compared with other types of inter-layer connectivity.

We carry out a tensor analysis on the expressive power inter-connections on convolutional arithmetic circuits (ConvACs) and relate our results to standard convolutional networks.

The analysis leads to performance bounds and practical guidelines for design of ConvACs.

The generalization of these results are discussed for other kinds of convolutional networks via generalized tensor decompositions.

Recently, densely connected networks such as FractalNet BID8 , ResNet BID6 , and DenseNet BID7 , have obtained state of the art performance on large problems where highly deep network configurations are used.

Adding dense connections between different layers of a network virtually shortens its depth, thus allowing a better flow of information and gradient through the network.

This makes possible the training of highly deep models.

Models with these types of connections have been successfully trained with hundreds of layers.

More specifically, DenseNets have achieved state of the art performance on the CIFAR-10, CIFAR-100, SVHN, and ImageNet datasets, using models of up to 1 thousand layers in depth.

Nevertheless, whether these connections provide a fundamental enhancement on the expressive power of a network, or just improve the training of the model, is still an open question.

In BID7 , DenseNet models with 3 times less parameters than its counterpart (ResNets) were able to achieve the same performance on the ImageNet challenge.

Moreover, a theoretical understanding of why the connections used by DenseNets lead to better performance compared with FractalNets or ResNets is still pending.

Despite the popularity of these models, there are few theoretical frameworks explaining the power of these models and providing insights to their performance.

In , the authors considered convolutional networks with linear activations and product pooling layers, called convolutional arithmetic circuits (ConvACs), and argued for the expressiveness of deep networks using a tensor based analysis.

This analysis has been extended to rectifier based convolutional networks via generalization of the tensor product .

In , it was shown that ConvACs enjoy a greater expressive power than rectifier based models despite the popularity of rectifier based networks in practice.

Indeed the empirical relevance of ConvAC was demonstrated through an architecture called SimNets .

In addition, the generative ConvAC of BID11 achieved state of the art performance in classification of images with missing pixels.

These results served as motivation for the works of ; ; BID9 ; BID10 , where different aspects of ConvACs were studied from a theoretical perspective.

In the inductive bias introduced by pooling geometries was studied.

Later, BID9 makes use of the quantum entanglement measure to analyze the inductive bias introduced by the correlations among the channels of ConvACs.

Moreover, BID10 generalizes the convolutional layer of ConvACs by allowing overlapping receptive fields, in other words permitting stride values lower than the convolution patch size.

These locally overlapping connections led to an enhancement on the expressive capacity of ConvACs.

The notion of inter-layer connectivity for ConvACs was addressed by in the context of sequential data processing, such as audio and text related tasks.

In that work, the expressive capabilities of interconnecting processing blocks from a sequence was studied.

Nevertheless, these types of interconnections are related to the sequential nature of the problem and different from the ones used in ResNet, FractalNet and DenseNet.

In this work, we extend the tensor analysis framework of to obtain insightful knowledge about the effect of dense connections, from the kind used in DenseNets, FractalNet and ResNet, on the expressiveness of deep ConvACs.

We study the expressive capabilities provided by different types of dense connections.

Moreover, from these results we derive performance bounds and practical guidelines for selection of the hyperparameters of a deep ConvAC, such as layer widths and the topology of dense connections.

These results serve as the first step into understanding dense connectivity in rectifier networks as well, since they can be further extended to include rectifier linear units, in the same spirit as the generalization of the tensor products done by .The remainder of this paper is organized as follows.

In Section 2, we introduce the notation and basic concepts from tensor algebra.

In Section 3, we present the tensor representation of ConvACs as introduced by , and later in Section 4, we obtain tensor representations for densely connected ConvACs.

In Section 5, performance bounds and design guidelines are derived for densely connected ConvACs.

The term tensor refers to a multi-dimensional array, where the order of the tensor corresponds to the number of indexes required to access one of its entries.

For instance, a vector is a tensor of order 1 while a matrix is a tensor of order 2.

In general a tensor A of order N requires N indexes (d 1 , . . .

, d N ) to access one of its elements.

For the sake of notation, given I ??? N, we use the expression [I] to denote the set {1, 2, . . .

, I}. In addition, the (d 1 , . . . , d N )-th entry of a given tensor of order N and size DISPLAYFORM0 .

Moreover, for the particular case of tensors of order N with symmetric sizes DISPLAYFORM1 ???N as shorthand notation for R M ??????????M .

A crucial operator in tensor analysis is the tensor product ???, since it is necessary for defining the rank of a tensor.

For two tensors B ??? R M1??????????Mp and C ??? R Mp+1??????????Mp+q , the tensor product is defined such that B ??? C ??? R M1??????????Mp+q and (B ??? C) d1,...,dp+q = B d1,...,dp C dp+1,...,dp+q for all (d 1 , . . . , d p+q ).

In tensor algebra, a tensor A ??? R M1??M2??????????M N is said to have rank 1 if it can be expressed as DISPLAYFORM2 Moreover, any tensor A ??? R M1??M2??????????M N can be expressed as a sum of rank-1 tensors, that is DISPLAYFORM3 where Z ??? N is sufficiently large and v DISPLAYFORM4 .

Note that this statement is DISPLAYFORM5 On the other hand, when Z is the minimum number such that (1) is satisfied, the rank of the tensor is defined to be rank(A) = Z and (1) becomes equivalent to the well known CANDECOMP/PARAFAC (CP) decomposition of A. Another operator, that is on the core of the former works of ; ; BID9 , is the matricization operator.

The operator [A] denotes the matricization of a tensor A ??? R M1??????????M N of order N .

This matricization of the tensor A re-orders its elements into a matrix A ConvAC is a convolutional neural network that utilizes linear activation functions with product pooling, unlike most popular convolutional networks which make use of rectifier activations with max or average pooling.

Moreover, the input of the network is modeled by X = (x 1 , . . .

, x N ) ??? (R s ) N , where x i ??? R s denotes the vectorization of the i-th patch of the input image.

For this analysis, it is assumed that a set of M features is obtained from every patch, that is DISPLAYFORM6

.

These features are selected from a given parametric family F = {f ?? : R s ??? R : ?? ??? ??}, such as Gaussian kernels, wavelet functions, or learned features.

Then, to determine whether an input X belongs to a class belonging to the set Y, the network evaluates the some score functions h y (X) ??? R and decides for the class y ??? Y such that DISPLAYFORM0 Using this formulation, in FIG0 (a) we observe an example of a single hidden layer ConvAC, while in FIG0 we observe the general case of a deep arithmetic circuit of L layers.

As shown by , any score function of a ConvAC can be expressed as an homogeneous polynomial with degree N on the input features of the form DISPLAYFORM1 where A y d1,...,d N ??? R are the polynomial coefficients stored in the grid-tensor DISPLAYFORM2 , degree N , and M N polynomial coefficients stored in the grid-tensor A y .For the special case of a shallow ConvAC with 1 ?? 1 convolutions and Z hidden units 1 , shown in FIG0 (a), the score functions are computed from the weight vectors a DISPLAYFORM3 .

This leads to the score function DISPLAYFORM4 The first step of the tensor analysis framework is to obtain an expression (in terms of the network parameters a y z and a z,i d ) of the grid-tensor A y that represents this concrete network architecture.

In other words, obtaining the expression for A y that transforms (2) into (3).

This expression was already obtained in as DISPLAYFORM5 where ??? denotes the tensor product.

Note that FORMULA14 is in the form of a standard CP decomposition of the grid tensor A y .

This implies that the rank of A y is bounded by rank(A y ) ??? Z. Moreover, the obtained results where generalized in for the case of a deep ConvAC with size-2 pooling windows 2 , thus L = log 2 N hidden layers as shown in FIG0 (b), leading to a grid-tensor given by the hierarchical tensor decomposition DISPLAYFORM6 where r 0 , . . .

, r L???1 ??? N are the number of channels in the hidden layers, {a ?????[r0] are the weights in the first hidden convolutions, DISPLAYFORM7 DISPLAYFORM8 are the weights of the hidden layers, and a L,1,y ??? R r L???1 stores the weights corresponding to the output y in the output layer.

The recent empirical success of densely connected networks (DenseNets), presented by BID7 , has served as motivation for our theoretical analysis on dense connectivity.

Dense connectivity in a convolutional neural network refers to the case when a number k ??? N (known as growth rate) of previous layers serve as input of the forthcoming layer.

More precisely, in BID7 , a DenseNet performs this via concatenation along the feature dimension of the current layer inputs with the preceding layer features.

Note that these feature must have compatible sizes along the spatial dimension for the concatenation to be possible.

To address this issue, BID7 proposed to group blocks of the same spatial dimensions into a dense block, as shown in FIG1 .

These dense blocks do not contain operations such as pooling, that alter the spatial dimensions of the input features.

Moreover, in the DenseNet architecture the layers that perform the pooling operation are called transition layers, since they serve as transition between dense blocks.

For example, in FIG1 we depict a dense block of 4 layers with growth rate k = 2, followed by a transition layer.

1 We must mention that the generalization to w ?? w convolutions is straightforward and was already covered by .2 Note that the generalization to different pooling sizes is straight forward and was done by .

In the original DenseNet these transition layers included one convolution layer before the pooling operation.

Nevertheless, for this work we consider transition layers composed of only pooling operations.

Note that this does not affect the generality of the model, since avoiding dense connections on the convolutional layer preceding the transition layer is equivalent to including a convolution in that transition layer 3 .In the case of ConvACs, any dense block of size greater than 1 can be represented as a dense block of size 1, since the activation function is the linear function (the non-linearity comes from the product pooling operator in the transition layer).

Therefore, for ConvACs, it is only reasonable to analyze dense blocks of size 1.

Note that, if we only allow dense connections between hidden layers within a dense block, a ConvAC is limited to a maximum growth rate of k = 1.

In order to analyze the effect of broader connectivity we extend the concept of growth rate by allowing dense connections between dense blocks.

With proper pooling, outputs of hidden layers belonging to different dense blocks can also be concatenated along the feature dimension.

In the reminder of this paper we refer to the dense connections between hidden layers of the same block as intra-block connections, while the connections between hidden layers of different blocks as inter-block connections.

In this section we analyze the effect of intra-block connections.

We first start by constructing a densely connected version of a single hidden layer ConvAC.

The resulting network with growth rate k = 1 is shown in FIG2 (a).

In the same manner as in (3), this architecture leads to the score function DISPLAYFORM0 Then, we present the following proposition regarding shallow ConvACs with dense connections of growth rate k = 1.

The network's function of a densely connected shallow ConvAC shown in (6) corresponds to the grid tensor DISPLAYFORM0 where DISPLAYFORM1 Proof See appendix B.1.Note that the rank of this tensor is now bounded by rank(A y ) ??? Z + M instead of Z, but adding these dense connections increases the number of parameters of the network from M N Z + Z to We now generalize the obtained results for the case of a L-layered dense arithmetic circuit, with growth rare k = 1, as the one in FIG2 (b).

Similarly to (5), the obtained grid tensor has the hierarchical decomposition given by DISPLAYFORM2 DISPLAYFORM3 . . .

DISPLAYFORM4 From this result we observe that inter block connections account for virtually increasing the width of the network's hidden layers from r l tor l r l + r l???1 for all l = 0, 1, . . .

, L ??? 1, where r ???1 M .

Note that this increased width comes at the expense of increasing the network's parameters.

Moreover, in Section 5 we discuss whether increasing the network's width via intra block dense connections leads to an enhancement in its overall expressive power.

In this section we study broader connectivity via dense inter-block connections.

As discussed in Section 4, proper pooling of the preceding features must take place before the concatenating them into the current layer.

Since this type of connections have not been considered in the former DenseNets, we propose 3 possible ways of realizing such connections (via product, average, or max pooling).

For a ConvAC with pooling window size w pool , an inter block connection that connects block DISPLAYFORM0 An example of an inter block connection of jump length L jump = 1 can be seen in FIG2 (c).

To perform this inter block connections, the sizes along the spatial dimensions of preceding features must be reduced by L jump w pool , before concatenating them along the feature dimension of layer l.

This spatial size reduction may be realized via pooling of the preceding features with window size L jump w pool .

When using a pooling layer the size along the feature dimension remains unchanged.

Moreover, the type of pooling employed (product, average, or maximum) affects the expressive potential of the resulting ConvAC.

Furthermore, the following proposition addresses the effect that adding dense inter block connections, via average pooling, has on the network function of a ConvAC.Proposition 2 Adding inter block connections via average pooling of jump length L jump ??? 1 to a standard ConvAC with grid-tensor A y ??? (R M ) ???N leads to a network function of the form DISPLAYFORM1 where g(X) contains polynomial terms on DISPLAYFORM2 Remark 1 This result is also valid when the connections are done by addition instead of concatenation, as it is done in ResNet and FractalNet.

Proof See appendix B.2.From this proposition we conclude that adding inter block connections average pooling does not alter the grid tensor A y , instead these connections account for extra polynomial terms of degree strictly less than N .

Note that, for the special case where the input features belong to an exponential kernel family, such as F = {f ?? (x) = e ?? T x : R s ??? R : ?? ??? ??} or F = {f ?? (x) = e ?????x p : R s ??? R : ?? ??? ??} where ?? p denotes the p norm with p ??? N, the number of polynomial terms is equivalent to the number of exponential basis that the network function can realize.

Therefore, the another valid measure of expressiveness is the number of polynomial terms a ConvAC is able to realize.

Given a certain ConvAC topology, the number of polynomial terms can be computed inductively by expanding the polynomial products of every layer via generalized binomial expansions.

Such an analysis is left for future contributions.

Moreover, if we perform this connections via product poling, the features to be concatenated correspond to polynomial terms of the same order.

This leads to a generalization of the intra-block connections from 4.1, leading to virtually increased widths r l r l + Ljump q=1 r l???1???q .

Finally, we leave the analysis of inter-block connections via maximum pooling for future work and consider only product pooling inter-block connections in the remainder of this paper.

For the sake of comparison, let us assume networks with hidden layer widths r l decaying (or increasing) at an exponential rate of ?? ??? R. Formally, this is r l = ??r l???1 ??? N, thus r l = (??) l r for all l = 0, 1, . . .

, L ??? 1, where r r 0 .

To shorten the notation, we denote as (L, r, ??, k) to a ConvAC with of exponential width decay ?? ??? R, length L ??? N, initial with r ??? N and growth-rate k ??? N. A growth-rate of k = 0 refers to a standard ConvAC with no dense connections.

Definition 1 Suppose that the weights of a (L, r, ??, k) ConvAC, with L, k ??? N and r, ?? ??? R, are randomly drawn according to some continuous non-vanishing distribution.

Then, this (L, r, ??, k) ConvAC is said to have weak dense gain G w ??? R if, with probability p > 0, we obtain score functions that cannot be realized by a (L, r , ??, 0) ConvAC with r < G w r. When p = 1, this (L, r, ??, k) ConvAC is said to have a strong dense gain G s = G w ??? R.Using this definition we present a bound for the weak dense gain G w in the following theorem.

DISPLAYFORM0 Proof See appendix B.3.This general bound may serve as guideline for tayloring M and the widths r 0 , . . .

, r L???1 such that we exploit the expressiveness added by dense connections.

Proof See appendix B.3.Using this result, we able able to quantify the expressive gain provided by dense inter block connections.

If a ConvAC has a dense gain G w = (1 + 1 ?? ) that is already close to the general bound from Theorem 5.1 it is less encouraging to include broader dense connections, since it would increase the number of parameters of the model while there is no room for a significant expressive gain increase.

In this scenario, connections as the ones in ResNet and FractalNet may result more beneficial since they do not increase the size of the model, while at the same time enhancing its trainability.

This last theorem shows that there exist a regime where this bounds can be achieved with strong dense gain.

Whether this is true outside this regime is still an open question, since further knowledge about the rank of random tensors is limited.

Moreover, these theorems does not consider the additional amount of parameters added by dense connections.

We complete our analysis by addressing this issue in the following proposition.

Proposition 3 Let ???P dense ??? N be the additional number of parameters that are added to a (L, r, ??, 0) ConvAC when we introduce dense connections of growth-rate k > 0.

In the same manner, let ???P stand ??? N be the number of parameters that are added to a (L, r, ??, 0) ConvAC when we increase its initial width r by a factor G ??? R. Then the ratio between ???P dense and ???P stand is greater than DISPLAYFORM1 Proof See appendix B.4.The factor G from this proposition directly relates to the dense gain of a ConvAC, thus this ratio may be used to decide whether is interesting to add dense connections to a model (we want this ratio to be as large as possible).

Finally Theorems 5.1 and 5.2 directly bound this ratio, which give the practitioner a guideline to decide which connections (if any) should be added to a given model.

Lemma 1 Given Z ??? N, let A ??? (R M ) ??? P be a random tensor of even order P ??? 2 such that DISPLAYFORM0 where a (k) z ??? R M are randomly drawn from a non-vanishing continuous distribution for all k ??? [P ] and z ??? [Z].

Then, if Z ??? M P/2 we have that rank(A) = rank([A]) = Z with probability 1.

This lemma also holds when for a subset Z ???

[Z] we have that a (k) z = a z e z ??? R M for all z ??? Z, where a z ??? R are randomly drawn from a non-vanishing continuous distribution.

Proof Using the definition of the matricization operator, we get that the matricization A is DISPLAYFORM1 .

FORMULA3 Note that, from this expression, it is straight forward to see that the rank of [A] is always less or equal than Z. DISPLAYFORM2 be a permuted version of [A] such that the first Z rows of U correspond to the rowsz ???Z of [A] , and the first Z columns of U correspond to the columnsz ???Z of [A] .

Since permuting the rows and the columns of a matrix does not alter its rank, we have that U has the same rank as [A] .

Now, let us partition U into blocks as DISPLAYFORM3 where P is of size Z-by-Z, and Q, W, Z have matching dimensions.

Note that, if rank(P) = Z then rank(U) ??? Z, which leads to DISPLAYFORM4 Therefore, it is sufficient to show that rank(P) = Z with probability 1 to conclude this proof.

To that end, let us define the mapping from x ??? R M P Z to P = P(x) as DISPLAYFORM5 Note that this definition of x implies that a DISPLAYFORM6 is computed as in FORMULA3 , we have that [A] = [A](x), thus Q = Q(x) and P = P(x).

Now, det P(x) is a polynomial on x, then it either vanishes in a set of measure zero or its the zero-polynomial (see BID0 ).If we set x to be equal to some x 0 ??? R M P Z such that a DISPLAYFORM7 is now a matrix with 1 on the entry (z,z) and zero elsewhere, we have that [A](x 0 ) is a diagonal matrix with ones on the diagonal elementsz ???Z and zero elsewhere.

This leads to P(x 0 ) = I Z which has a determinant det P(x 0 ) = 0.

Finally, since there exist x 0 such that the polynomial det P(x 0 ) is not zero, we conclude that det P(x) is not the zero-polynomial, which means that det P(x) = 0 with probability 1, thus proving this lemma.

DISPLAYFORM8 ??? P be random tensors of even order P ??? 2 and Z ??? N be tensors such that DISPLAYFORM9 z ??? R M are randomly drawn from a non-vanishing continuous distribution.

Then, if Z 1 ??? M P/2 and Z 2 ??? M P/2 , we have that rank (A ??? B) = Z 1 Z 2 with probability 1.Proof Let C ??? (R M ) ???2P be a random tensor defined as C = A ??? B. Therefore, we may express C as DISPLAYFORM10 Then, we define rank-1 tensors C (q,z) to be C (q,z) = a(1) DISPLAYFORM11 Since C is now expressed as a sum of Z 1 Z 2 rank-1 tensors, we have that rank(C) ??? Z 1 Z 2 .Since Z 1 ??? M P/2 and Z 2 ??? M P/2 we may use Lemma 1, this leads to rank([A]) = Z 1 and rank([B]) = Z 2 with probability 1.

Finally we, use the properties of the Kronecker product to obtain the rank of the matricization C as rank( DISPLAYFORM12 with probability 1, thus proving the Lemma.

DISPLAYFORM13 ??? P be tensors of order P > 2 and Z ??? N be tensors such that

Proof We reformulate this (6) to have the same form as (3).

To that end we define a DISPLAYFORM0 which has the same form as (3).

Therefore, as done in FORMULA14 , we obtain the grid tensor for this architecture as DISPLAYFORM1 , thus proving this proposition.

Proof DISPLAYFORM0 T ??? R M N , the output of the l-th layer of a (L, r, ??, 0) ConvAC can be stored into the vectors of mappings DISPLAYFORM1 .

Moreover, since the entries of these vectors are the result of l ??? 1 convolution-pooling layers with product pooling of window size 2, all the mappings ?? l,j 1 (x) can be expressed as a sum of polynomial terms on x of degree 2 l .

Now, let the coefficient vectors a DISPLAYFORM2 ], be the weight vectors for the convolution of the l-th layer.

To shorten the notation we use a l,j,?? , DISPLAYFORM3 as shorthand for the convolution between these vectors.

Then, the outputs the the layer l of this ConvAC are given by ?? l+1,j ??? R r l+1 with ?? DISPLAYFORM4 If we recursively calculate these out vectors up to the L-th layer we obtain the score functions h DISPLAYFORM5 1 (x) ??? R. We now consider the effecct of adding dense connections via average pooling from some k ??? N preceding layers l ??? 1, . . .

, l ??? k. To this end, letr l = k q=1 r l???q be the total size along the feature dimesnion of the vectors to be concatenated.

In addition, let DISPLAYFORM6 Rr l be the vectors of mappings of the corresponding preceeding features at the layer l for j ??? [N/r l ].

In order to compute the convolutions of this layer, an additional vector of coefficients is required as b DISPLAYFORM7 Then, the outputs of the l-th layer of this (L, r, ??, k) ConvAC are the denoted as the vectors?? DISPLAYFORM8 where DISPLAYFORM9 Note that the entries of ?? l,j (x) are assumed to come from preceding layers with an appropiate average pooling.

Since performing avergae pooling does not increase the degree of the polynomial terms involved (only product pooling does) and the jump length Ljump is at least 1, the entries of ?? l,j (x) have at most polynomial degree 2 l???1 , which is strictly less than the degree of the entries of ?? l,j (x) (i.e., 2 l ).

Therefore, from the obtained expression of ?? l+1,j we observe that it has polynomials withb degree no greater than 2 l + 2 l???1 , while the entries of ?? l+1,j have a strictly higher degree of DISPLAYFORM10 Moreover, since a l,j,?? , ?? l,j + ?? l,j can be expressed as DISPLAYFORM11 we can make use of the obtained results in an unductive manner up to the L-th layer, thus leading to DISPLAYFORM12 , where g(x) contains polynomial terms of x of order strictly less than N , thus proving this theorem.

Note that this result also applies to additive and resudial connections, as de ones used in ResNet and FractalNet, since they can be expressed as in (11).B.3 PROOF OF THEOREMS 5.1 TO 5.3 DISPLAYFORM13 For the forthcoming analysis let us assume r 0 ??? M .

This assumption is done, so that we can write min{r 0 , M } = r 0 , merely for notation purposes since we show that this does not affect the generality of the results.

Using this assumption, we upper bound the rank of the grid tensor as DISPLAYFORM14 It was shown in that, when the weights are independently generated from some continuous distribution, we have that rank ?? 1,j,?? = min{r 0 , M } with probability 1.

Note that, the bounds obtained for r 0 values greater than M is the same as for r 0 = M , thus implying that the assumption of r 0 ??? M does not affect the generality of the results.

Finally, by induction up to the L-th layer, we obtain a bound for the grid tensor rank as DISPLAYFORM15 Since we assumed networks with hidden layer widths r l decaying (or increasing) at an exponential rate of ?? ??? R. Formally, this is r l = ??r l???1 ??? N, thus r l = (??) l r for all l = 0, 1, . . .

, L ??? 1, where r r 0 .

Therefore, we may simplify the obtained bound to DISPLAYFORM16 We this analysis by proving Theorem 5.1.

To that end let A y dense be the grid tensor of a dense (L, r, ??, k) ConvAC with k > 0, while A y stand is the grid tensor of a (L, r , ??, 0) ConvAC with r ??? R. As discussed in Section 4, this dense version of the former (L, r, ??, 0) ConvAC is equivalent to virtually increasing the widths of the ConvAC, which translates extra additive terms in the expressions from 12.

Moreover, using corollary 1 we observe that, if the ranks of the tensors ?? l,j,?? are additive and multiplicative up to rank(A y dense ) > rank(A y stand ), so they are up to rank(A y stand ).

A weak dense gain value G w ??? R is achieved when there is a set of functions realized by the (L, r, ??, k) ConvAC that cannot be realized by (L, r , ??, 0) ConvAC unless r = G w r. To bound this gain, let us assume the best case scenario where rank(A DISPLAYFORM17 which proves Theorem 5.1.For Theorem 5.2 we use consider the particular case of k = 1, which yields a core tensor given by the hierarchical tensor decomposition from (9).

We use the same assumption of r 0 ??? M and define the virtually increased widthsr l r l + r l???1 ??? N for l = 1, . . .

, L ??? 1 andr 0 M .

This leads to rank ?? 1,j,?? = rank Note that for r l = ??r l???1 ??? N (?? ??? R), we get virtually increased widthsr l = (1 + ??) l r = ?? 1 + As in for the proof of Theorem 5.1, the maximum dense gain G w is obtained when rank(A y dense ) reaches the maximum possible rank.

In this particular case, this corresponds to rank(A By definition, we have that ???P std = P (L, Gr, ??, 0) ??? P (L, Gr, ??, 0) and ???P dense = P (L, Gr, ??, k) ??? P (L, r, ??, 0), thus yielding Finnaly, we use these expressions to compute the ratio of interest as DISPLAYFORM18 2 ) l + (G 2 ??? 1) k q=1 ?? ???q which proves this proposition.

@highlight

We analyze the expressive power of the connections used in DenseNets via tensor decompositions.