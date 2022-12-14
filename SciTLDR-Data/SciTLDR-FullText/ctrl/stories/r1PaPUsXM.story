This paper gives a rigorous analysis of trained Generalized Hamming Networks (GHN) proposed by Fan (2017) and discloses an interesting finding about GHNs, i.e. stacked convolution layers in a GHN is equivalent to a single yet wide convolution layer.

The revealed equivalence, on the theoretical side, can be regarded as a constructive manifestation of the universal approximation theorem Cybenko (1989); Hornik (1991).

In practice, it has profound and multi-fold implications.

For network visualization, the constructed deep epitomes at each layer provide a visualization of network internal representation that does not rely on the input data.

Moreover, deep epitomes allows the direct extraction of features in just one step, without resorting to regularized optimizations used in existing visualization tools.

Despite the great success in recent years, neural networks have long been criticized for their blackbox natures and the lack of comprehensive understanding of underlying mechanisms e.g. in BID3 ; BID12 ; BID30 ; BID29 .

The earliest effort to interpret neural computing in terms of logic inferencing indeed dated back to the seminal paper of BID24 , followed by recent attempts to provide explanations from a multitude of perspectives (reviewed in Section 2).As an alternative approach to deciphering the mysterious neural networks, various network visualization techniques have been actively developed in recent years (e.g. BID11 ; BID28 and references therein).

Such visualizations not only provide general understanding about the learning process of networks, but also disclose operational instructions on how to adjust network architecture for performance improvements.

Majority of visualization approaches probe the relations between input data and neuron activations, by showing either how neurons react to some sample inputs or, reversely, how desired activations are attained or maximized with regularized reconstruction of inputs BID7 ; BID20 ; BID36 ; BID31 ; BID23 ; BID33 ; BID0 .

Input data are invariably used in visualization to probe how the information flow is transformed through the different layers of neural networks.

Although insightful, visualization approaches as such have to face a critical open question: to what extend the conclusions drawn from the analysis of sample inputs can be safely applied to new data?In order to furnish confirmatory answer to the above-mentioned question, ideally, one would have to employ a visualization tool that is independent of input data.

This ambitious mission appears impossible at a first glance -the final neuron outputs cannot be readily decomposed as the product of inputs and neuron weights because the thresholding in ReLU activations is input data dependent.

By following the principle of fuzzy logic, BID8 recently demonstrated that ReLUs are not essential and can be removed from the so called generalized hamming network (GHN) .

This simplified network architecture, as reviewed in section 3, facilitates the analysis of neuron interplay based on connection weights only.

Consequently, stacked convolution layers can be merged into a single hidden layer without taking into account of inputs from previous layers.

Equivalent weights of the merged GHN, which is called deep epitome, are computed analytically without resorting to any learning or optimization processes.

Moreover, deep epitomes constructed at different layers can be readily applied to new data to extract hierarchical features in just one step (section 4).

Despite the great success in recent years, neural networks have long been criticized for their blackbox natures e.g. in BID3 : "they capture hidden relations between inputs and outputs with a highly accurate approximation, but no definitive answer is offered for the question of how they work".

The spearhead BID24 attempted to interpret neural computing in terms of logic inferencing, followed by more "recent" interpretations e.g. in terms of the universal approximation framework BID6 ; BID15 , restricted Boltzmann machine BID14 , information bottleneck theory BID30 , Nevertheless the mission is far from complete and the training of neural networks (especially deep ones) is still a trail-and-error based practice.

The early 1990s witnessed the birth of fuzzy neural networks (FNN) BID21 ; BID13 which attempted to furnish neural networks with the interpretability of fuzzy logic BID34 BID37 ; BID2 .

On the other hand, neural networks have been used as a computational tool to come up with both membership functions and fuzzy inference rules BID9 BID32 .

This joint force endeavour remains active in the new millennium e.g. BID27 ; BID22 ; Nauck & N??rnberger (2013); BID19 ; BID16 .

Nevertheless, FNNs have been largely overlooked nowadays by scholars and engineers in machine learning (ML) community, partially due to the lack of convincing demonstrations on ML problems with large datasets.

The exception case is the recent BID8 , which re-interpreted celebrated ReLU and batch normalization with a novel Generalized Hamming Network (GHN) and demonstrated the state-of-the-art performances on a variety of machine learning tasks.

While GHNs adopted deep networks with multiple convolution layers, in this paper, we will show how to merge multiple stacked convolution layers into a single yet wide convolution layer.

There are abundant empirical evidences backing the belief that deep network structures is preferred to shallow ones BID10 , on the other hand, it was theoretically proved by the universal approximation theorem that, a single hidden layer network with non-linear activation can well approximate any arbitrary decision functions BID6 BID15 .

Also, empirically, it was shown that one may reduce depth and increase width of network architecture while still attaining or outperforming the accuracies of deep CNN Ba & Caurana (2013) and residual network BID35 .

Nevertheless, it was unclear how to convert a trained deep network into a shallow equivalent network.

To this end, the equivalence revealed in Section 3 can be treated as a constructive manifestation of the universal approximation theorem.

Various network visualization techniques have been actively developed in recent years, with BID7 interpreting high level features via maximizing activation and sampling; BID20 BID36 learning hierarchical convolutional features via energy or cost minimization; BID31 computing class saliency maps for given images; BID23 reconstructing images from CNN features with an natural image prior applied; BID33 visualizing live activations as well as deep features via regularized optimization; BID0 monitoring prediction errors of individual linear classifiers at multiple iterations.

Since all these visualization methods are based on the analysis of examples, the applicability of visualization methods to new data is questionable and no confirmatory answers are provided in a principled manner.

The name "deep epitome" is reminiscent of BID17 ; BID4 ; BID18 ; BID5 , in which miniature, condensed "epitomes" consisting of the most essential elements were extracted to model and reconstruct a set of given images.

During the learning process, the self-similarity of image(s), either in terms of pixel-to-pixel comparison or spatial configuration, was exploited and a "smooth" mapping between epitome and input image pixels was estimated.

We briefly review generalized hamming networks (GHN) introduced in BID8 and present in great detail a method to derive the deep epitome of a trained GHN.

Note that we follow notations in BID8 with minor modifications for the sake of clarity and brevity.

According to BID8 , the cornerstone notion of generalized hamming distance (GHD) is defined as g(a, b) := a ??? b = a + b ??? 2 ?? a ?? b for any a, b ??? R. Then the negative GHD is used to quantify the similarity between neuron inputs x and weights w: DISPLAYFORM0 in which L denotes the length of neuron weights e.g. in convolution kernels, and g(w, x) is the arithmetic mean of generalized hamming distance between elements of w and x. By dividing the constant 2 L , (1) becomes the common representation of neuron computing (w ?? x + b) provided that: DISPLAYFORM1 It was proposed by BID8 that neuron bias terms should follow the condition (2) analytically without resorting to an optimization approach.

Any networks that fulfil this requirement are thus called generalized hamming networks (GHN).

In the light of fuzzy logic, the negative of GHD quantifies the degree of equivalence between inputs x and weights w, i.e. the fuzzy truth value of the statement x ??? w where ??? denotes a fuzzy equivalence relation.

Moreover, g(x, x) leads to a measurement of fuzziness in x, which reaches the maximal fuzziness when x = 0.5 and monotonically decreases when x deviates from 0.5.

Also it can be shown that GHD followed by a non-linear activation induces a fuzzy XOR connective BID8 .When viewed in this GHN framework, the ReLU activation function max(0, 0.5???g(x, w)) actually sets a minimal hamming distance threshold of 0.5 on neuron outputs.

BID8 then argued that the use of ReLU activation is not essential because bias terms are analytically set in GHNs.

BID8 reported only negligible influences when ReLU was completely skipped for the easy MNIST classification problem.

For more challenging CIFAR10/100 classifications, removing ReLUs merely prolonged the learning process but the final classification accuracies remained almost the same.

To this end, we restrict our investigation in this paper to those GHNs which have no ReLUs.

As illustrated below, this simplification allows for strict derivation of deep epitome from individual convolution layers in GHNs.3.2 GENERALIZED HAMMING DISTANCE AND EPITOME BID8 postulated that one may analyse the entire GHN in terms of fuzzy logic inference rules, yet no elaboration on the analysis was given.

Inspired by the universal approximation framework, we show below how to unravel a deep GHN by merging multiple convolution layers into a single hidden layer.

We first reformulate the convolution operation in terms of generalized hamming distance (GHD) for each layer, then illustrate how to combine multiple convolution operations across different layers.

As said, this combination is only made possible with GHNs in which bias terms strictly follow condition (2).

Without loss of generality, we illustrate derivations and proofs for 1D neuron inputs and weights (with complete proofs elaborated in appendix A).

Nevertheless, it is straightforward to extend the derivation to 2D or high dimensions.

And appendices B to D illustrate deep epitomes of GHNs trained for 2D MNIST and CIFAR10/100 image classifications.

Definition 1.

For two given tuples DISPLAYFORM2 1 . . .

L , where ??? denotes the generalized hamming distance operator.

Then the product has following properties, DISPLAYFORM3 K but they are permutation equivalent, in the sense that there exist permutation matrices P and Q such that x DISPLAYFORM4 2.

non-linear: in contrast to the standard outer product which is bilinear in each of its entry, the hamming outer product is non-linear since in general x DISPLAYFORM5 where ?? ??? R is a scalar.

Therefore, the hamming outer product defined as such is a pseudo outer product.

DISPLAYFORM6 M because of the associativity of GHD.

This property holds for arbitrary number of tuples.

iterated operation: the definition can be trivially extended to multiple tuples DISPLAYFORM0 Definition 2.

The convolution of hamming outer product or hamming convolution, denoted * , of two tuples is a binary operation that sums up corresponding hamming outer product entries: DISPLAYFORM1 where the subsets DISPLAYFORM2 DISPLAYFORM3 The hamming convolution has following properties, DISPLAYFORM4 K since the partition subsets S(n) remains the same.2.

non-linear: this property is inherited from the non-linearity of the hamming outer product.

DISPLAYFORM5 M since the summation of GHDs is non-associative.

Note this is in contrast to the associativity of the hamming outer product.

iterated operation: likewise, the definition can be extended to multiple tuples x DISPLAYFORM0 Figure 1 illustrates an example in which GHDs are accumulated through two consecutive convolutions.

Note that the conversion from the hamming outer products to its convolution is non-invertible, Figure 2 : The hamming convolution of two banks of epitomes.

Remarks: a) for the inputs A, B the number of epitomes M a must be the same as the number of channels C b ; and for the output bank DISPLAYFORM1 .

b) the notation * refers to the hamming convolution between two banks of epitomes (see Definition 5 for details).

The convolution of two single-layered epitomes is treated as a special case with all M a , C a , M b , C b = 1.

c) the notation refers to the summation of multiple epitomes of the same length, which is defined in Definition 7.

d) multiple (coloured) epitomes in D correspond to different (coloured) epitomes in B; and different (shaded) channels in D correspond to different (shaded) channels of inputs in A.in the sense that, it is impossible to recover individual summands x k ??? y l from the summation DISPLAYFORM2 As proved in proposition 4, it is possible to compute the convolution of tuples in two (or more) stacked layers without explicitly recovering individual outer product entries of each layer.

Due to the non-linearity of the hamming convolutions, computing the composite of two hamming convolutions is non-trivial as elaborated in Section 3.3.

In order to illustrate how to carry out this operation, let us first introduce the epitome of a hamming convolution as follows.

Definition 3.

An epitome consists of a set of N pairs E = (g n , s n ), n = 1, . . .

, N where g n denotes the summation of GHD entries from some hamming convolutions, s n the number of summands or the cardinality of the subset S(n) defined above, and N is called the length of the epitome.

A normalized epitome is an epitome with s n = 1 for all n = 1, . . .

N .

Any epitome can then be normalized by setting (g n /s n , 1) for all elements.

A normalized epitome may also refer to input data x or neuron weights w that are not yet involved in any convolution operations.

In the latter case, g n is simply the input data x or neuron weights w.

Remark: the summation of GHD entries g n is defined abstractly, and depending on different scenarios, the underlying outer product may operate on arbitrary number of tuples DISPLAYFORM3 Fuzzy logic interpretation: in contrast to the traditional signal processing point of view, in which neuron weights w are treated as parameters of linear transformation and bias terms b are appropriate thresholds for non-linear activations, the generalized hamming distance approach treats w as fuzzy templates and sets bias terms analytically according to (2).

In this view, the normalization g n /s n is nothing but the mean GHD of entries in the subset S(n), which indicates a grade of fitness (or a fuzzy set) between templates w and inputs x at location n. This kind of arithmetic mean operator has been used for aggregating evidences in fuzzy sets and empirically performed quite well in decision making environments (e.g. see BID37 ).Still in the light of signal processing, the generalized hamming distance naturally induces an information enhancement and suppression mechanism.

Since the gradient of g(x, w) with respect to x is 1 ??? 2w, the information in x is then either enhanced or suppressed according to w : a) the output g(x, w) is always x for w = 0 (conversely 1 ??? x for w = 1) with no information loss in x; b) for w = 0.5, the output g(x, w) is always 0.5 regardless of x, thus input information in x is completely suppressed; c) for w < 0.0 or w > 1.0 information in x is proportionally enhanced.

It was indeed observed, during the learning process in our experiments, a small faction of prominent feature pixels in weights w gradually attain large positive or negative values, so that corresponding input pixels play decisive roles in classification.

On the other hand, large majority of obscure pixels remain in the fuzzy regime near 0.5, and correspondingly, input pixels have virtually no influence on the final decision (see experimental results in Section 4).

This observation is also in accordance with the information compression interpretation advocated by BID30 , and the connection indicates an interesting research direction for future work.

This subsection only illustrates main results concerning how to merge multiple hamming convolution operations in stacked layers into a single-layer of epitomes i.e. deep epitome.

Detailed proofs are given in appendix A. Theorem 10.

A generalized hamming network consisting of multiple convolution layers, is equivalent to a bank of epitome, called deep epitome [ D ] , which can be computed by iteratively applying the composite hamming convolution in equation (8) to individual layer of epitomes: DISPLAYFORM0 in which = C a is the number of channels in the first bank A, = M z is the number of epitomes in the last bank Z, and DISPLAYFORM1 is the length of composite deep epitome.

Note that for the hamming convolution to be a valid operation, the number of epitomes in the previous layer and the number channels in the current layer must be the same e.g. DISPLAYFORM2 Proof.

For given inputs represented as a bank of normalized epitomes DISPLAYFORM3 Ly Cx ] is obtained by recursively applying equation (8) to outputs from the previous layers, and factoring out the input due to the associativity proved in proposition 9: DISPLAYFORM4 Remark: due to the non-linearity of underlying hamming outer products, to prove the associativity of the convolution of epitomes is by no means trivial (see proposition 9).

In essence, we have to use proposition 4 to compute the convolution of two epitomes even though individual entries of the underlying hamming outer product are not directly accessible.

Consequently, the updating rule outlined in equations (4) and (5) play the crucial role in setting due bias terms analytically for generalized hamming networks (GHN), as opposed to the optimization approach often adopted by many non-GHN deep convolution networks.

Fuzzy logic inferencing with deep epitomes: Eq. (11) can be treated as a fuzzy logic inferencing rule, with which elements of input x are compared with respect to corresponding elements of deep epitomes d. More specifically, the negative of GHD quantifies the degree of equivalence between inputs x and epitome weights d, i.e. the fuzzy truth value of the assertion x ??? d where ??? denotes a fuzzy logical biconditional.

Therefore, output scores in y indicate the grade of fuzzy equivalences truth values between x and the shifted d at different spatial locations.

This inferencing rule, in the same vein of BID8 , is applicable to either a single layer neuron weights or the composite deep epitomes as proved by (11).

Constructive manifestation of the universal approximation theorem: it was proved that a single hidden layer network with non-linear activation can well approximate any arbitrary decision functions BID6 ; BID15 , yet it was also argued by BID10 that such a single layer may be infeasibly large and may fail to learn and generalize correctly.

Theorem 10 proves that such a simplified single hidden layer network can actually be constructed from a trained GNH.

In this sense Theorem 10 illustrates a concrete solution which materializes the universal approximation theorem.

We illustrate below deep epitomes extracted from three generalized hamming networks trained with MNIST, CIFAR10/100 classification respectively.

Detailed descriptions about the network architectures (number of layers, channels etc.) are included in the appendix.

Deep epitomes derived in the previous section allows one to build up and visualize hierarchical features in an on-line manner during the learning process.

This approach is in contrast to many existing approaches, which often apply additional optimization or learning processes with various type of regularizations e.g. in BID7 BID33 .

Figures 5, 8 and 11, 12 in appendices illustrate deep epitomes learnt by three generalized hamming networks for the MNIST and CIFAR10/100 image classification tasks.

It was observed that geometrical structures of hierarchical features were formed at different layers, rather early during the learning process (e.g. 1000 out of 10000 iterations).

Substantial follow up efforts were invested on refining features for improved details.

The scrutinization of normalized epitome histograms in FIG2 showed that a majority of pixel values remain relatively small during the learning process, while a small fraction of epitome weights gradually accumulate large values over thousands of iterations to form prominent features.

The observation of sparse features has been reported and interpreted in terms of sparse coding e.g. BID26 or the information compression mechanism as advocated by BID30 .

Following Fan (2017) we adopt the notion of fuzziness (also reviewed in Section 3.1) to provide a fuzzy logic interpretation: prominent features correspond to neuron weights with low fuzziness.

It was indeed observed in FIG4 that fuzziness of deep epitomes in general decrease during the learning process despite of fluctuations at some layers.

The inclination towards reduced fuzziness seems in accord with the minimization of classification errors, although the fuzziness is not explicitly minimized.

Finally we re-iterate that the internal representation of deep epitomes is input data independent.

For instance in MNIST handwritten images, it is certain constellations of strokes instead of digits that are learnt at layer 3 (see Figure 5 ).

The matching of arbitrary input data with such "fuzzy templates" is then quantified by the generalized hamming distance, and can be treated as generic fuzzy logic It must be noted that the extraction of these hierarchical salient features is not entirely new and has been reported e.g. in BID7 BID20 .

Nevertheless, the equivalence of deep epitomes disclosed in Theorem 10 leads to an unique characteristic of GHNs -deep layer features do not necessarily rely on features extracted from previous layers, instead, they can be extracted in one step using deep epitomes at desired layers.

For extremely deep convolution networks e.g. those with over 100 layers, this simplification may bring about substantial reduction of computational and algorithmic complexities.

This potential advantage is worth follow up exploration in future research.

We have proposed in this paper a novel network representation, called deep epitome, which is proved to be equivalent to stacked convolution layers in generalized hamming networks (GHN).

Theoretically this representation provides a constructive manifestation for the universal approximation theorem BID6 BID15 , which states that a single layered network, in principle, is able to approximate any arbitrary decision functions up to any desired accuracy.

On the other hand, it is a dominant belief BID10 , which is supported by abundant empirical evidences, that deep structures play an indispensable role in decomposing the combinatorial optimization problem into layer-wise manageable sub-problems.

We concur with the view and supplement with our demonstration that, a trained deep GHN can be converted into a simplified networks for the sake of high interpretability, reduced algorithmic and computational complexities.

The success of our endeavours lies in the rigorous derivation of convolving epitomes across different layers in eq. (4) and (5), which set due bias terms analytically without resorting to optimizationbased approaches.

Consequently, deep epitomes at all convolution layers can be computed without using any input data.

Moreover, deep epitomes can be used to extract hierarchical features in just one step at any desired layers.

In the light of fuzzy logic, the normalized epitome (definition 3) encodes a grade of fitness between the learnt templates and given inputs at certain spatial locations.

This fuzzy logic interpretation furnishes a refreshing perspective that, in our view, will open the black box of deep learning eventually.

APPENDIX A Definition 1.

For two given tuples DISPLAYFORM0 . .

, y L }, the hamming outer product, denoted , is a set of corresponding elements x DISPLAYFORM1 . .

L , where ??? denotes the generalized hamming distance operator.

Then the product has following properties, DISPLAYFORM2 K but they are permutation equivalent, in the sense that there exist permutation matrices P and Q such that x DISPLAYFORM3 2.

non-linear: in contrast to the standard outer product which is bilinear in each of its entry, the hamming outer product is non-linear since in general x DISPLAYFORM4 where ?? ??? R is a scalar.

Therefore, the hamming outer product defined as such is a pseudo outer product.

DISPLAYFORM5 M because of the associativity of GHD.

This property holds for arbitrary number of tuples.

iterated operation: the definition can be trivially extended to multiple tuples DISPLAYFORM0 Proof.

associativity: by definition it suffices to prove element-wise (x k ???y l )???z m = x k ???(y l ???z m ) because of the associativity of the generalized hamming distance.

DISPLAYFORM1 , then it suffices to prove non-linearity for each element i.e. DISPLAYFORM2 Definition 2.

The convolution of hamming outer product or hamming convolution, denoted * , of two tuples is a binary operation that sums up corresponding hamming outer product entries: DISPLAYFORM3 where the subsets S(n) := {(k, l) k + (L ??? l) = n} for n = 1, . . .

, K + L ??? 1, and the union of all subsets constitute a partition of all indices n=1,...

,K+L???1 DISPLAYFORM4 The hamming convolution has following properties, DISPLAYFORM5 K since the partition subsets S(n) remains the same.2.

non-linear: this property is inherited from the non-linearity of the hamming outer product.3.

non-associative: DISPLAYFORM6 M since the summation of GHDs is non-associative.

Note this is in contrast to the associativity of the hamming outer product.

iterated operation: likewise, the definition can be extended to multiple tuples x DISPLAYFORM0 Proof.

non-associativity: by definition it suffices to prove element-wise in general DISPLAYFORM1 Definition 3.

An epitome consists of a set of N pairs E = (g n , s n ), n = 1, . . .

, N where g n denotes the summation of GHD entries from some hamming convolutions, s n the number of summands or the cardinality of the subset S(n) defined above, and N is called the length of the epitome.

A normalized epitome is an epitome with s n = 1 for all n = 1, . . .

N .

Any epitome can then be normalized by setting (g n /s n , 1) for all elements.

A normalized epitome may also refer to input data x or neuron weights w that are not yet involved in any convolution operations.

In the latter case, g n is simply the input data x or neuron weights w.

Proposition 4.

Given two tuples x = {x k |k = 1 . . .

K} and y = {y l |l = 1 . . .

L}, then DISPLAYFORM2 Proof.

DISPLAYFORM3 Remark: eq. (4) allows one to compute summation of all hamming outer product elements on the right hand side, even though individual elements x k and y l are unable to recover from the given summands k x k and l y l .

The definition below immediately follows and illustrates how to merge elements of two epitomes.

DISPLAYFORM4 the convolution of two epitomes E c = E a * E b is given by: DISPLAYFORM5 where Remark: this operation is applicable to the case when two epitomes are merged via spatial convolution (see Figure 2 for an example).

Note that this merging operation is associative due to the following theorem.

Theorem 6.

The convolution of multiple epitomes, as defined in 5, is associative: DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 Proof.

By definition 5, elements of E a * Remark: this associative property is of paramount importance for the derivation of deep epitomes, which factor out the inputs x from subsequent convolutions with neuron weights w.

Definition 7.

Given two epitomes of the same size E a = {(g n , s n )|n = 1, . . .

N }, E b = {(g n , s n )|n = 1, . . .

N }, the summation of two epitomes E c = E a E b is trivially defined by element-wise summation:E c = {(g n , s n )|n = 1, . . .

, N }; where g n = g n + g n , s n = s n + s n .

Remark: the summation operation is applicable to the case when epitomes are (iteratively) merged cross different channels (see Figure 2 for an example).

Note that the size of two input epitomes must be the same, and the size of output epitome remain unchanged.

Moreover, the operation is trivially extended to multiple epitomes DISPLAYFORM0 The output of this operation, in turn, is a bank with Figure 2 for an example.

DISPLAYFORM1 Proposition 9.

The composite convolutions of multiple epitome banks, as given in definition 8, is associative: DISPLAYFORM2 Proof.

The associativity immediately follows the associativity of Theorem 6 and definition 7.Remark: this associative property, which is inherited from theorem 6, can be trivially extended to multiple banks and lead to the main theorem of the paper as follows.

Theorem 10.

A generalized hamming network consisting of multiple convolution layers, is equivalent to a bank of epitome, called deep epitome [ D ] , which can be computed by iteratively applying the composite hamming convolution in equation (8) to individual layer of epitomes: DISPLAYFORM3 in which = C a is the number of channels in the first bank A, = M z is the number of epitomes in the last bank Z, and = L a +(L b ???1)+. .

.+(L z ???1) is the length of composite deep epitome.

Note that for the hamming convolution to be a valid operation, the number of epitomes in the previous layer and the number channels in the current layer must be the same e.g. DISPLAYFORM4 Proof.

.APPENDIX B: DEEP EPITOMES WITH MNIST HANDWRITTEN RECOGNITION Figure 5 : Deep epitomes at layers 1,2 and 3 for a GHN trained with MNIST classification at iterations 100 and 10000 respectively. .

Pseudo colour images correspond to three channels of features outputs for input RGB colour channels.

<|TLDR|>

@highlight

bridge the gap in soft computing