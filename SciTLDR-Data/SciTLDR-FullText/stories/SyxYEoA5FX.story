We flip the usual approach to study invariance and robustness of neural networks by considering the non-uniqueness and instability of the inverse mapping.

We provide theoretical and numerical results on the inverse of ReLU-layers.

First, we derive a necessary and sufficient condition on the existence of invariance that provides a geometric interpretation.

Next, we move to robustness via analyzing local effects on the inverse.

To conclude, we show how this reverse point of view not only provides insights into key effects, but also enables to view adversarial examples from different perspectives.

Invariance and stability/robustness are two of the most important properties characterizing the behavior of a neural network.

Due to growing requirements like robustness to adversarial examples BID16 and the increasing use of deep learning in safety-critical applications, there has been a surge in interest in these properties.

Invariance and stability are considered to be the key mechanisms in dealing with uninformative properties of the input BID0 BID6 and are studied from the information theoretical perspective in form of the loss of information about the input BID17 BID11 .Invariance and stability are also tightly linked to robustness against adversarial attacks (Cisse et al., 2017; BID18 BID14 , generalization BID15 Gouk et al., 2018) and even the training of Generative Adversarial Networks BID7 .

In general, stability is studied via two basic properties: 1) locally via a norm of the Jacobian BID15 BID14 , 2) globally via the Lipschitz constant (Cisse et al., 2017; BID7 BID18 ).

From a high-level perspective, both of these approaches study an upper bound on stability as the Lipschitz constant and a Jacobian norm quantifies the highest possible change under a perturbation with a given magnitude.

We, unlike the approaches above, aim to broaden our understanding by analyzing the lowest possible change under a perturbation.

More formally, we study which perturbations ∆x do not (or only little) affect the outcome of a network F .

Our analysis considers a given input data point x and investigates the ∆x's, such that DISPLAYFORM0 where a small ε > 0 is given.

While these properties can be crucial for many discriminative tasks BID6 , the model could be flawed if perturbations that alter the semantics have only a minor impact on the features.

This is a reverse perspective on adversarial examples BID16 , which commonly considers small input perturbations that lead to large changes and thus to arbitrary decisions of the network.

This flipped view and the study of smallest changes calls for a different approach: we study the instabilities of the inverse instead of the stabilities of the forward mapping.

In particular, if F is invariant to perturbations ∆x, then x and x + ∆x lie in the preimage of the output z = F (x), i.e. F is not uniquely invertible.

Robustness towards large perturbations induces an instable inverse mapping as small changes in the output can be due to large changes in the input.

Based on the piecewise linear nature of ReLU networks BID8 , we characterize the preimage of ReLU-activations as a single point, finite (bounded) or infinite (unbounded) .

Further, we study the stability of the linearization of rectifier networks via its singular values.

To illustrate these locally changing properties and to demonstrate their tight connection, we visualize the behavior on a synthetic problem in FIG0 .

As ReLU-layers are piecewise linear, the local behavior is constant on polytopes.

Further, the regions with infinite/finite preimages correspond to regions with condition number of one or zero, while singleton preimages link to condition numbers larger than one.

Thus, both properties are tightly connected and investigating one property alone yields an incomplete picture.

Our contributions are as follows:• We derive conditions when the preimage of an output of a ReLU-layer has finite or infinite volume or is a single point.

Based on these conditions, we derive an algorithm to check these conditions and exemplify its usability by applying it to investigate the preimages of a trained network.

(See Section 2.)

• We study the stability of the inverse via analyzing the linearization at a point in input space, which is accurate within a polytope.

We provide upper bounds on the smallest singular value of a linearization and prove how the removal of uncorrelated features could effect the stability of the inverse mapping.

Based on these ideas, we experimentally demonstrate how singular values evolve over the different layers in rectifier networks.

(See Section 3.)

• We introduce a reverse view on adversarial examples and connect it to invariance and robustness by leveraging our analysis of preimages. (see Section 5)

While analyzing invariance and robustness properties is a major topic in theoretical treatments of deep networks BID6 , studying it via the inverse is less common.

Several works like Mahendran & Vedaldi (2015) , Mahendran & Vedaldi (2016) or Dosovitskiy & Brox (2016) focus on reconstructing inputs from features of convolutional neural networks (CNNs) to visualize the information content of features.

Instead, we investigate potential mechanisms affecting the invertibility.

BID4 gives a first geometrical view on the shape of preimages of outputs from ReLU layers, which is directly related to the question of injectivity of the mapping under ReLU.

BID13 analyzes the reconstruction property of cReLU (concatenated ReLU); however, the more general situation of using the standard rectifier is not studied.

A notable other line of work assumes random weights in order to derive guarantees for invertibility, see Gilbert et al. (2017) or BID1 , whereas we focus on the preimage of ReLU-activations without assumptions on the weights.

Moreover, several reversible network structures were recently proposed (Gomez et al., 2017; BID5 BID2 .

Most notably, in Jacobsen et al. (2018) a bijective network, up to its last layer, was trained successfully on ImageNet which does not exhibit any invariance.

However, the network is very robust towards many directions in the input which is reflected in a strongly instable inverse.

Hence, even carefully designed network show at least one of the two effects (invariance and robustness) studied in this work.

Especially stability has seen growing interest due to adversarial examples BID16 ), yet stability is mostly studied with respect to the forward mapping, see e.g. Cisse et al. (2017) .Two main resources for our view of rectifier networks as piecewise linear models are BID8 and BID9 .

Closest to our approach is the work of on global statements of injectivity and stability of a single layer including ReLU and pooling.

The authors focus on global injectivity and stability bounds via combinatorial statements over all configurations attainable by ReLU and pooling.

These conditions are valid on the entire input space, while the restriction to parts of the input space may be far from these worst-case conditions.

Further works focus on applications like inverse problems with learned forward models (Jensen et al., 1999; Lu et al., 1999) and parameter estimation problems (Lähivaara et al., 2018) , which are often formulated as inverse problems and require the inversion of networks.

In this section, we briefly state our notation as a reference: DISPLAYFORM0 • Pre-activations: DISPLAYFORM1 • Activation: DISPLAYFORM2 where φ : R → R the pointwise applied activation function, if not specified differently g := ReLU.• Number of layers: L ∈ N • Entire network: DISPLAYFORM3 For matrices A ∈ R m×n and I ⊂ [m] := {1, . . .

, m}, A| I denotes the matrix consisting of the rows of A whose index is in set I -analogously for vectors.

Also A| y 0 describes the restriction to the index set {i : y i > 0} for y ∈ R m , analogously for ≺, =, , .

For vectors y ∈ R m , y 0 is the elementwise relation, analogously for ≺, =, , .

Furthermore, we define N (A) as the null space of a matrix A. The Euclidean inner product is denoted by ·, · .

For every matrix A ∈ R m×n with the rows DISPLAYFORM4 .

Vice versa, we associate every finite set in R n with a matrix (only possible up to permutation of the indices).

In this section, we analyze different kinds of preimages of a ReLU-layer and investigate under which conditions the inverse image of a given point is a singleton (a set containing exactly one element) or has finite/infinite volume.

These conditions will yield a simple algorithm able to distinguish between these different preimages, which is applied in Section 2.2.For the analysis of preimages of a given output one can study single layers separately or multiple layers at once.

However, since the concatenation of two injective functions is again injective while a non-injective function followed by an injective function is non-injective, studying single layers is crucial.

We therefore develop a theory for the case of single layers in this section.

Notice that in case of multiple layers one is also required to investigate the image space of the previous layer.

We will focus our study on the most common activation function, ReLU.

One of its key features is the non-injectivity, caused by the constant mapping on the negative half space.

It provides neural networks with an efficient way to deploy invariances.

Basically all other common activation functions are injective, which would lead to a straightforward analysis of the preimages.

However, injective activations like ELU (Clevert et al., 2016) and Leaky ReLU (Maas et al., 2013) only swap the invariance for robustness, which in turn leads to the problem of having instable inverses.

This question of stability will be analyzed in more detail in Section 3.We start by introducing one of our main tools -namely the omnidirectionality.

i) A ∈ R m×n is called omnidirectional if every linear open halfspace in R n contains a row of A, i.e. for every given x ∈ R n \ {0} there exists an index i ∈ [m], such that a i , x > 0. ii) A ∈ R m×n and b ∈ R m are called omnidirectional for the point p ∈ R n if A is omnidirectional and b = −Ap.

Thus, if A is omnidirectional, for every direction of a hyperplane through the origin forming two halfspaces, there is a vector from the rows of A inside each open halfspace, hence the term omnidirectional (see FIG1 for an illustration).

Note that the hyperplanes are due to ReLU as it maps the open halfspace to positive values and the closed halfspace to zero.

A straightforward way to construct an omnidirectional matrix is by taking a matrix whose rows form a spanning set F and use the vertical concatenation of F and −F. This idea is related to cReLU BID13 .The following Corollary gives several equivalent formulations of omnidirectionality, which will turn out to be useful for the proof of the subsequent Theorem 4 in this section.

The short proofs of the statements are provided in Appendix A1.Corollary 2 (Equivalences of omnidirectionality) The following statements are equivalent: DISPLAYFORM0 ii) Ax 0 implies x = 0, where x ∈ R n .iii) There exists a unique x ∈ R n , such that Ax 0.iv) There exist no x ∈ R n \ {0}, such that Ax 0.More importantly, omnidirectionality is directly related to the ReLU-layer preimages and will provide us with a method to characterize their volume (see Theorem 4) .

To analyze such inverse images, we consider y = ReLU(Ax + b) for a given output y ∈ R m with A ∈ R m×n , b ∈ R m and x ∈ R n .

If we know A, b and y, we can write the equation as the following mixed linear system: DISPLAYFORM1 where A| y 0 denotes the restriction of the matrix A to the rows, which are specified by the index set {i :

y i > 0} (see Section 1.2 for the used notation).Remark 3 It is possible to enrich the mixed system to include conditions/priors on x (e.g. x ∈ R n ≥0 ).The inequality system in equation 2 links its set of solutions and therefore the volume of the preimages of the ReLU-layer with the omnidirectionality of A and b. Defining A := AO T , where O ∈ R k×n denotes an orthonormal basis of N (A| y 0 ) with k := dim N (A| y 0 ) and b := b| y 0 + A| y 0 (P N (A|y 0) ⊥ x), where P V denotes the orthogonal projection into the closed space V, leads to the following main theorem of this section, which is proven in Appendix A1.Theorem 4 (Preimages of ReLU-layers) Let A, b and k = dim N (A| y 0 ) be as above.

The preimage of a point y under a ReLU-layer is i) for k = 0 a singleton.ii) for k > 0 a singleton, if and only if there exists an index set I for the rows of A and b, such that (A| I , b| I ) is omnidirectional for some point p ∈ R k .iii) for k > 0 a compact polytope with finite volume, if and only if A is omnidirectional.

Thus, omnidirectionality allows in theory to distinguish whether the inverse image of a ReLU-layer is a singleton, a compact polytope or has infinite volume.

However, obtaining a method to check if a given matrix is omnidirectional is crucial for later numerical investigations.

For this reason, we will go back to the geometrical perspective of omnidirectionality (see FIG1 ).

This will also help us to get a better intuition on the frequency of occurrence of the different preimages.

The following Theorem 5 gives another geometrical interpretation of omnidirectionality, whose short proof is given in Appendix A1.

DISPLAYFORM2 o is the interior of the convex hull spanned by the rows of A (see Definition 10 in Appendix A1).Therefore, the matrix must contain a simplex in order to be omnidirectional, as the convex hull of the matrix A ∈ R m×n has to have an interior.

Hence, we have the following: DISPLAYFORM3 By considering the geometric perspective, a tuple (A ∈ R m×n , b ∈ R m ) is omnidirectional for a point p ∈ R n , if and only if the m hyperplanes generated by the rows of A with bias b intersect at p and their normal vectors (rows of A) form an omnidirectional set.

We can use Corollary 6 to conclude that singleton preimages of ReLU-layers are very unlikely to happen in practice (if we do not design for it), since a necessary condition is that n + 1 hyperplanes have to intersect in one point in R n .

Therefore we conclude, that singleton preimages of ReLU layers in practice only and exclusively occurs, if the mixed linear system already has sufficient linear equalities.

Algorithm to check uniqueness: The above results can be used to derive an algorithm to check whether a preimage of a given output is finite, infinite or just a singleton.

A singleton inverse image is obtained as long as rank(A| y 0 ) = n holds true, which can be easily computed.

To distinguish preimages with finite and infinite volumes, it is enough to check if A is omnidirectional (see Theorem 4iii), which can be done numerically by using the definition of the convex hull, Theorem 5 and Corollary 6.

This leads to a linear programming problem, which is presented in Appendix A3 and was also used to create Figure 1.

In this section, we demonstrate for a simple model that the preimage of a layer can be a singleton, infinite or finite depending on the given point.

For this purpose, we trained a MLP with two hidden ReLU layers of size 3500 and 784 on MNIST (LeCun & Cortes, 2010) .

We chose the layer size of 3500, because the likelihood of having roughly 784 (input dimension of MNIST) positive outputs was high for this setting.

In Figure 4 , we plotted the number of samples in the test set that have infinite (red curve) or finite (blue curve) preimages over the number of positive outputs.

It can be assumed that all samples which have more or equal to 784 (the input dimension) positive outputs have a singleton preimage and are therefore finite.

In the dark gray region between 723 and 784, both effects occurred, which can be seen by the overlap of the red and blue curve.

To determine whether a preimage for less than 784 positive outputs was compact we used Theorem 4iii and the algorithm described in Appendix A3.

In this section we analyze the robustness of rectifier MLPs against large perturbations via studying the stability of the inverse mapping.

Concretely, we study the effect of ReLU on the singular values of the linearization of network F .

While the linearization of a network F at some point x only provides a first impression on its global stability properties, the linearization of ReLU networks is exact in some neighborhood due to its piecewise-linear nature BID9 .

In particular, the input space R d of a rectifier network F is partitioned into convex polytopes P F , corresponding to a different linear function on each region (see FIG0 .

Hence, for each polytope P in the set of all input polytopes P F , the network F can be simplified as DISPLAYFORM0 DISPLAYFORM1 In particular, each of the linearized matrices A P can be written via a chain of weight matrix multiplications that incorporates the effect of ReLU.

To this end, the following definition introduces admissible index sets that formalize all possible local behaviors and diagonal matrices to locally model the effect of ReLU, see BID19 :Definition 7 (Admissible index sets, ReLU as diagonal matrix) An index set I l for a layer l is admissible if DISPLAYFORM2 Further, let D I denote a diagonal matrix with (D I ) ii = 1 for i ∈ I and (D I ) ii = 0 for i ∈ I, where I is an admissible index set.

Using this notation, the mapping of pre-activation z ∈ R d under ReLU can be written as DISPLAYFORM3 Thus, the linearization A P of a network with L layers is a matrix chain DISPLAYFORM4 , where A l are the weight matrices of layer l and DISPLAYFORM5 Of special interest for a stability analysis is the range of possible effects by the application of the rectifier.

Since the effect by ReLU corresponds to the application of D I for admissible I, we now turn to studying the changes of the singular values of a general matrix A compared to D I A. For example, the matrix A could represent the chain of matrix products up to pre-activations in layer l.

Then, the effect of ReLU can be globally upper bounded:Lemma 8 (Global upper bound for largest and smallest singular value) Let σ l be the singular values of D I A. Then for all admissible index sets I, the smallest non-zero singular value is upper bounded by min{σ l : σ l > 0} ≤σ k , where k = N − |I| andσ 1 ≥ ... ≥σ N > 0 are the non-zero singular values of A. Furthermore, the largest singular value is upper bounded by max{σ l : σ l > 0} ≤σ 1 .Lemma 8 analyzes the best case scenario with respect to the highest value of the smallest singular value.

While this would yield a more stable inverse mapping, one needs to keep in mind that N (A P ) grows by the corresponding elimination of rows via D I .

Moreover, reaching this bound is very unlikely as it requires the singular vectors to perfectly align with the directions that collapse due to D i .

Thus, we now turn to study effects which could happen locally for some input polytopes P .An example of a drastic effect through the application of ReLU is depicted in Figure 3 .

Since one vector is only weakly correlated to the removed vector and the situation is overdetermined, removing this feature for some inputs x in the blue area leaves over the strongly correlated features.

While the two singular values of the 3-vectors-system were close to one, the singular vectors after the removal by ReLU are badly ill-conditioned.

As many modern deep networks increase the dimension in the first layers, redundant situations as in Figure 3 are common, which are inherently vulnerable to such phenomena.

For example, BID10 proposes a regularizer to avoid such strongly correlated features.

The following lemma formalizes the situation exemplified before:Lemma 9 (Removal of weakly correlated rows) Let A ∈ R m×n with rows a j and DISPLAYFORM6 with M = m − |I| and constant c > 0.

Then for the singular values σ l = 0 of D I A it holds DISPLAYFORM7 Note that I has to be admissible when considering the effect of ReLU.Lemma 9 provides an upper bound on the smallest singular value, given a condition on the correlation of all a j and a k .

However, the condition 3 depends on the number M of remaining rows a j .

Hence, in a highly redundant setting even after removal by ReLU (large N ), c needs to be large such that the correlation fulfills the condition.

Yet, in this case the upper bound on the smallest singular value, given by c, is high.

We discuss this effect further and provide quantitative results in the Appendix A5.Effect under multiple layers: For the effect of ReLU applied to multiple layers, we are particularly interested in following questions:• Can the application of another layer have a pre-conditioning effect yielding a stable inverse?• What happens when we only compose orthogonal matrices which have stable inverses?Note that a way to enforce an approximate orthogonality constraint was proposed for CNNs in Cisse et al. (2017) , however only for the filters of the convolution.

For both situations the answer is similar: the nonlinear nature of ReLU induces locally different effects.

Thus, if we choose a pre-conditioner A l for a specific matrix A l−1 P , it might not stabilize the matrix product for matrices A l−1 P * corresponding to different input polytopes P * .

For the case of composing only orthogonal matrices, consider a network up to layer l − 1, where the linearization A l−1 P has orthogonal columns (assume the network gets larger, thus A l−1 P has more rows than columns).

Then, the application of ReLU in form of DISPLAYFORM8 removes the orthogonality property of the rows of A l−1 P , if setting entries in the rows from I l to zero results in non-orthogonal columns (likely when considering dense matrices).

Hence, DISPLAYFORM9 is not orthogonal for some I l .

In this case, the matrix product DISPLAYFORM10 is not orthogonal, which results in decaying singular values.

This is why, even when especially designing the network by e.g. orthogonal matrices, stability issues with respect to the inverse arise.

To conclude this section, we remark that the presented results are rather of a qualitative nature showcasing effects of ReLU on the singular values.

Yet, the analysis does not require any assumptions and is thus valid for any MLP (including CNNs without pooling).To give an idea of quantitative effects we study numerical examples in the subsequent subsection.

In this section, we show how the previously discussed theoretical stability properties can be examined for a given network.

In particular, we conduct experiments on CIFAR10 (Krizhevsky & Hinton, 2009) using two baseline CNNs, see A4 for details on architectures and training setup.

Our CNNs use only strides instead of pooling and use no residual connections and normalization layers.

Thus, the architectures fit to the theoretical study as the strided discrete convolution can be written as a matrix-vector multiplication.

Singular values over multiple layers: Experimentally most interesting is the development of singular values over multiple layer as several effects are potentially at interplay.

Figure 6 shows how all singular values evolve in convolutional layers (layers 1-6, after application of ReLU).

While the shape of the curve is similar for layer 1-5, it can be seen that the largest singular value grows, while the small singular values decrease significantly.

Note that this growth of the largest singular values is in line with observations for adversarial examples, see BID16 .

While many defense strategies like Cisse et al. (2017) or Jia (2017) focus on the largest singular value, the behavior of the smaller singular values is often overlooked.

Additionally, we provide in Appendix A5 a numerical analysis of the condition from Lemma 9 to gain an understanding of possible effects of ReLU on the singular values.

Furthermore, we add results for a thinner CNN (ThinCIFAR) and for the MLP from Section 2.2 in Appendix A6.Relationship between stability and invariance: While invariance is characterized by zero singular values, the condition number only takes non-zero singular values into account, see e.g. layer 6 from Figure 6 .

This tight relationship is further investigated in Figure 5 which compares the output size, the condition number and the number of non-zero singular values vs. the layers for WideCIFAR and ThinCIFAR.

In combination with lower output dimension, ReLU has a different effect for ThinCIFAR.

The number of singular values decreases in layer 5, which cuts off the smallest singular values, resulting in a lower condition number.

Yet, there are more invariance directions within the corresponding linear region.

For a visual comparison see FIG0 .Computational costs and scaling analysis:

First, we remark that the linearization of a network F for an input point x 0 can be computed via backpropagation.

Based on this linearization the computation of the full SVD scales cubically.

Especially, early CNN-layers have high dimensional outputs which may cause memory issues when computing the entire SVD.

We thus choose a small CNN trained on CIFAR10 as these inputs are only of size 32 × 32 × 3.

To scale this analysis up to e.g. ImageNet with VGG-networks, a restriction to a window of the input image is necessary to reduce the complexity of the full SVD especially for early layers.

See BID2 , where the singular values restricted to input windows were used to estimate the stability of the entire i-RevNet trained on ImageNet.

Characterization of preimages over multiple layers: Theorem 4 yields a characterization (singleton, finite, infinite) of the preimage of a point y under a single ReLU-layer.

When considering the preimage of y under multiple layers l to l − i, two difficulties arise: 1) If the preimage of y under layer l is not a singleton, one needs to compute the intersection of the image of layer l − 1 and the preimage of y under layer l − 1.

2) For all points in the intersection, the conditions of Theorem 4 need to be checked, which requires the solution of a linear program, see Algorithm 1 in Appendix A3.

Hence, our analysis is currently restricted to a layer-by-layer approach.

However, this layer-wise and local study could enable to pin down the specific layers where information, which is expressed in non-singleton preimages.

Preimages for convolutional layers: In general, convolutional layers are affine transforms Ax + b, but have a sparse and shared structure compared to dense matrices used in MLPs.

Thus, the trivial case rank(A| y 0 ) = n (Algorithm 1, Appendix A3) needs to be explicitly checked.

For MLPs it was assumed in section 2.2 that rank(A| y 0 ) = |{i : y i > 0}| as dense matrix rows are almost surely linear independent in practice.

Inverse stability for convolutional networks: For inverse stability, we consider the linearization DISPLAYFORM0 for an input polytope P .

In convolutional networks, each A l implements a multi-channel discrete convolution.

While singular values of each A l can be efficiently computed by leveraging the convolutional structure, see BID12 , the shared structure is not preserved in the matrix chain A P due to the application of ReLU (expressed via D I l ).

Thus, a tighter analysis that leverages the convolutional structure in A l , compared to our general assumption that A l can be any linear mapping, is not straightforward with current tools but would certainly lead to further insights.

In our stability analysis, we employ a piecewiselinear viewpoint which allows to characterize stability via the singular values of the linearization which is exact within an input polytope.

However, when considering an ε-ball B ε (y) around a point y = A P x + b p to model e.g. reconstruction from noisy activations y, further questions arise: 1) Can all points in B ε (y) be reached by an x * from the polytope P ?

2) Are points from another polytope P mapping to points in B ε (y)?

In this case, the inverse stability needs to be augmented by nonlinear considerations to model movements between piecewise-linear regions.

While the focus of this work was an in-depth analysis of potential effects on crucial properties like invariance and robustness due to ReLU, we envision several practical implications of our approach: Network design and regularization: As both the concept of omnidirectionality and removal of rows due to ReLU showed, there is a breadth of potential effects.

In terms of network design, controlling such effects could be desirable.

In particular, a change to injective activation functions (tanh, leakyReLU, ELU etc.) remove the discussed preimages, but immediately transfer to an instable inverse due to saturation.

Furthermore, Lemma 9 draws a connection to regularizing correlation between feature maps as introduced in BID10 .

Hence, both omnidirectionality and correlation between rows can be thought of as geometrical properties which could partially be controlled by regularization or architecture design.

Furthermore, the analysis also shows the difficulty of controlling these properties in vanilla architectures.

However, by incorporating additional structure like dimension splitting in reversible networks BID2 or invertible residual connections BID2 , the preimage is by design a singleton.

Connection to information loss: Our analysis is tightly related to mutual information I(x l ; x) loss, which has gained growing interest due to the information bottleneck BID17 BID11 .

In particular, invariance in layer l may induce I(x l ; x) ≤ I(x l−1 ; x) due to the data processing inequality (Cover & Thomas, 2006) .

Similarly, an instable inverse can induce an information loss as activations x l are quantized due to finite precision on hardware.

Implications for adversarial examples: Despite being crucial for many discriminative tasks to contract the space along uninformative directions BID6 , invariance and robustness may induce severe vulnerabilities for adversarial examples BID16 .

For instance, a model Despite the semantically very different examples, the features are identical as the original image "3" and the two perturbed variants "6" and "4" are in the same preimage.

Further details in Appendix A7.would be flawed if perturbations that alter the semantics only have a minor impact on the features of the network.

Classically, adversarial examples are viewed as small perturbations which induce large changes in the network outputs BID16 ).

Yet, reversing this perspective leads to another failure case: if large changes in the input alter its semantics for a given task, but the networks output is robust or even invariant to such changes, the model might just be as flawed from this reverse point of view.

This change in perspective leads to a natural way of addressing invariance and robustness via invertibility: If F is invariant to perturbations ∆x, then x and x + ∆x lie in the preimage of the output z = F (x) i.e. F is not uniquely invertible.

Robustness towards large perturbations induces an instable inverse mapping as small changes in the output can be due to large changes in the input.

Finally FIG4 demonstrates such a striking failure, where perturbations alter the semantics drastically, yet the activations even after the first layer are identical.

To find these examples, we leveraged the developed theory about preimages and a linear programming formulation, see Appendix A7.

We presented the inverse as an approach to tackle the invariance and robustness properties of ReLU networks.

Particularly, we studied two main effects: 1) conditions under which the preimage of a ReLU layer is a point, finite or infinite and 2) how ReLU can effect the inverse stability of the linearization.

By deriving approaches to numerically examine these effects, we highlighted the broad range of possible effects.

Moreover, controlling such properties may be desirable as our experiment on adversarial examples showed.

Besides the open questions on how to control the structure of preimages and inverse stability via architecture design or regularization, we envision several theoretical directions based on our work.

Especially, incorporate nonlinear effects like moving between linear regions of rectifier networks could lift the analysis closer to practice.

Furthermore, studying similarities of omnidirectionality as a geometrical property and singular values could further strengthen the link between these two crucial properties.

Proof (Corollary 2, Equivalences of omnidirectionality) We show the equivalences by proving i) DISPLAYFORM0 Let A ∈ R m×n be omnidirectional, i.e. for every x = 0, it holds that Ax 0.

This is equivalent to DISPLAYFORM1 which is ii).

The implications from ii) to iii) and from iii) to iv) are obvious.

From iv), we have that DISPLAYFORM2 which is equivalent to i), the omnidirectionality of A. Altogether, this shows the equivalence of all four points.

Definition 10 (Convex hull) For A ∈ R m×n , the convex hull is defined as DISPLAYFORM3 where a i ∈ R n are the rows of A.Theorem 11 (Stiemke's theorem, see Dantzig (1963) ) Let A ∈ R m×n be a matrix, then the following two expressions are equivalent.• y : Ay 0 DISPLAYFORM4 Here z 0 means that 0 = z 0 .

Proof (Theorem 12, Singleton solutions of inequality systems)

"⇐" Let (A| I , b| I ) be omnirectional for x 0 .

Then it holds that A| I x + b| I = A| I (x − x 0 ) 0.

Due to the omnidirectionality of A| I , x 0 is the unique solution of the inequality system A| I x + b| I 0.

The existence of a solution for the whole system Ax + b 0 is guaranteed by assumption and therefore x 0 is the unique solution of Ax + b 0.

"⇒" Here we will prove " I : (A| I , b| I ) omnidirectional for some p ⇒ solution non-unique".We will start by doing the following logical transformations: This means that A| I is not omnidirectional, because otherwise A| I x 0 + b| I = 0 due to the definition of I, which would lead to the contradiction that (A| I , b| I ) is omnidirectional for x 0 .

But this means ∃x = 0 : A| I x 0 as a result of Corollary 2.

Since A| I c x 0 + b| I c ≺ 0, we also have ∀x ∃ > 0 : A| I c (x 0 + x) + b| I c ≺ 0.

This holds in particular for x , so we define accordingly x * := εx = 0.

Therefore, we have A| I c (x 0 + x * ) + b| I c ≺ 0 as well as DISPLAYFORM5 DISPLAYFORM6 Altogether it holds that A(x 0 + x * ) + b 0 with x * = 0, which means that x 0 is a non-unique solution for the inequality system Ax + b 0.Proof (Theorem 4, Preimages of ReLU-layers) We consider the ReLU-layer DISPLAYFORM7 given its output y ∈ R m with A ∈ R m×n , b ∈ R m and x ∈ R n .

Clearly, this equation can also be written as the mixed linear system DISPLAYFORM8 This allows us to consider the two cases N (A| y 0 ) = {0} and N (A| y 0 ) = {0}.In the first case, we have a linear system which allows us to calculate x uniquely, i.e. we can do retrieval.

This leads us to the second case, the interesting one.

In this case we can only recover x uniquely if and only if the system of inequalities "pins down" P N (A|y 0) x, where P V is the orthogonal projection into the closed space V .

Formally this requires DISPLAYFORM9 to have a unique solution for x ∈ R n and P N (A|y 0) ⊥ x fixed (given via the equality system).

By defining b := b| y 0 + A| y 0 (P N (A|y 0 ) ⊥ x) we have DISPLAYFORM10

Proof (Lemma 8, Global upper bound for largest and smallest singular value) The upper bound on the largest singular value is trivial, as ReLU is contractive or in other terms D I Ax 2 ≤ Ax 2 for all I and x ∈ R n .

To prove the upper bound for the smallest singular value, we assume DISPLAYFORM0 and aim to produce a contradiction.

Consider all singular vectorsṽ k * with k * ≥ k from matrix A. It holds for allṽ k * σ DISPLAYFORM1 as D I is a projection matrix and thus only contracting.

As DISPLAYFORM2 In this section, we formulate the algorithm to determine whether the preimage of y given by DISPLAYFORM3 is finite.

This requires to check whether A (see Theorem 4) is omnidirectional, which is equivalent to DISPLAYFORM4 see Theorem 5.

Since it is reasonable to assume that 0 will not lie on the boundary of the convex hull, we can formulate this as a linear programming problem.

The side-conditions incorporate the definition of convex hulls (Definition 10, Appendix A1).

The objective function is chosen arbitrary, as we are only interested in a solution.

return False {see Corollary 6} end if c ← (1; . . . ; 1) {arbitrary objective} return Does a solution for the linear program DISPLAYFORM0

Training details for MLP on MNIST:• Training using Adam optimizer (Kingma & Ba, 2015) • Epochs: 25• Batch size: 1000Training details for WideCIFAR and ThinCIFAR:• Training setup from Keras (Chollet et al., 2015) examples: cifar10_cnn• No data augmentation• RMSprop optimizer• Epochs: 100• Batch size: 32 In order to better understand the bound on the smallest singular value after ReLU, given by Lemma 9, we numerically proceed as follows: Curve showing how many rows a i satisfy condition 3 from Lemma 9 depending on values of constant c. The red line shows the total number of remaining rows after removal by ReLU, M = 5120.

Even for small constants c most a i fulfill condition 3, yet not all, which is required by the lemma to give an upper bound on the smallest singular value.

The example is from layer 4 of WideCIFAR, for only one sample from the test set.1.

We choose c ∈ [a, b], where a, b are suitable interval endpoints.2.

Given c, we compute for every a k with k ∈ I the value of c a k 2 √ M (M is the number of remaining rows, in the example M = 5120).3.

For every a k we count the number of a i satisfying DISPLAYFORM0 4.

We take the a k with the maximal number of a i satisfying the condition. (Note, that this ignores the requirement a k ∈ N (D I A) ⊥ .)5.

If we have an a k , where all a i satisfy the condition, the corresponding constant c gives the upper bound on the smallest singular value after ReLU.

FIG11 shows the number of a i satisfying the correlation condition given different choices of c. The red line is reached for c ≈ 6.

However, even the largest singular value after ReLU is smaller than 2.5 (shown in FIG9 ).

Thus, the bound given by Lemma 9 is far off.

This can be explained by the

This section briefly describes how the results in FIG4 from the introduction were obtained (copied in FIG0 for readability).

After training the network from 1 (in Appendix A4), we searched the MNIST test set for input images with yielded the fewest positive activations in the first layer, in the figure the digits "3" and "4".

After selecting the example input x * , we selected another input c belonging to a different class (e.g. a "6" and "4" in the first example).

Hence, we searched within the preimage of the features y * of the first layer for examples x which resemble images c from another class.

By doing this we observe, that the preimages of the MLP may have large volume.

In these cases, the network is invariant to some semantics changes which shows how the study of preimages can reveal previously unknown properties.

@highlight

We analyze the invertibility of deep neural networks by studying preimages of ReLU-layers and the stability of the inverse.

@highlight

This paper studies the volume of preimage of a ReLU network’s activation at a certain layer, and it builds on the piecewise linearity of a ReLU network’s forward function. 

@highlight

This paper presents an analysis of the inverse invariance of ReLU networks and provides upper bounds on singular values of a train network.