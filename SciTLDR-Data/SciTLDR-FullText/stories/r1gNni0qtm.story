Recurrent Neural Networks (RNNs) are very successful at solving challenging problems with sequential data.

However, this observed efficiency is not yet entirely explained by theory.

It is known that a certain class of multiplicative RNNs enjoys the property of depth efficiency --- a shallow network of exponentially large width is necessary to realize the same score function as computed by such an RNN.

Such networks, however, are not very often applied to real life tasks.

In this work, we attempt to reduce the gap between theory and practice by extending the theoretical analysis to RNNs which employ various nonlinearities, such as Rectified Linear Unit (ReLU), and show that they also benefit from properties of universality and depth efficiency.

Our theoretical results are verified by a series of extensive computational experiments.

Recurrent Neural Networks are firmly established to be one of the best deep learning techniques when the task at hand requires processing sequential data, such as text, audio, or video BID10 BID18 BID7 .

The ability of these neural networks to efficiently represent a rich class of functions with a relatively small number of parameters is often referred to as depth efficiency, and the theory behind this phenomenon is not yet fully understood.

A recent line of work BID12 BID5 focuses on comparing various deep learning architectures in terms of their expressive power.

It was shown in that ConvNets with product pooling are exponentially more expressive than shallow networks, that is there exist functions realized by ConvNets which require an exponentially large number of parameters in order to be realized by shallow nets.

A similar result also holds for RNNs with multiplicative recurrent cells BID12 .

We aim to extend this analysis to RNNs with rectifier nonlinearities which are often used in practice.

The main challenge of such analysis is that the tools used for analyzing multiplicative networks, namely, properties of standard tensor decompositions and ideas from algebraic geometry, can not be applied in this case, and thus some other approach is required.

Our objective is to apply the machinery of generalized tensor decompositions, and show universality and existence of depth efficiency in such RNNs.

Tensor methods have a rich history of successful application in machine learning.

BID21 , in their framework of TensorFaces, proposed to treat facial image data as multidimensional arrays and analyze them with tensor decompositions, which led to significant boost in face recognition accuracy.

BID0 ) employed higher-order co-occurence data and tensor factorization techniques to improve on word embeddings models.

Tensor methods also allow to produce more accurate and robust recommender systems by taking into account a multifaceted nature of real environments BID6 .In recent years a great deal of work was done in applications of tensor calculus to both theoretical and practical aspects of deep learning algorithms.

BID15 represented filters in a convolutional network with CP decomposition BID11 BID1 which allowed for much faster inference at the cost of a negligible drop in performance.

BID19 proposed to use Tensor Train (TT) decomposition BID20 to compress fully-connected layers of large neural networks while preserving their expressive power.

Later on, TT was exploited to reduce the number of parameters and improve the performance of recurrent networks in long-term forecasting BID24 and video classification BID23 problems.

In addition to the practical benefits, tensor decompositions were used to analyze theoretical aspects of deep neural nets. ) investigated a connection between various network architectures and tensor decompositions, which made possible to compare their expressive power.

Specifically, it was shown that CP and Hierarchial Tucker BID9 decompositions correspond to shallow networks and convolutional networks respectively.

Recently, this analysis was extended by BID12 who showed that TT decomposition can be represented as a recurrent network with multiplicative connections.

This specific form of RNNs was also empirically proved to provide a substantial performance boost over standard RNN models BID22 .First results on the connection between tensor decompositions and neural networks were obtained for rather simple architectures, however, later on, they were extended in order to analyze more practical deep neural nets.

It was shown that theoretical results can be generalized to a large class of CNNs with ReLU nonlinearities and dilated convolutions BID5 , providing valuable insights on how they can be improved.

However, there is a missing piece in the whole picture as theoretical properties of more complex nonlinear RNNs have yet to be analyzed.

In this paper, we elaborate on this problem and present new tools for conducting a theoretical analysis of such RNNs, specifically when rectifier nonlinearities are used.

Let us now recall the known results about the connection of tensor decompositions and multiplicative architectures, and then show how they are generalized in order to include networks with ReLU nonlinearities.

Suppose that we are given a dataset of objects with a sequential structure, i.e. every object in the dataset can be written as DISPLAYFORM0 We also introduce a parametric feature map f θ : R N → R M which essentially preprocesses the data before it is fed into the network.

Assumption 1 holds for many types of data, e.g. in the case of natural images we can cut them into rectangular patches which are then arranged into vectors x (t) .

A typical choice for the feature map f θ in this particular case is an affine map followed by a nonlinear activation: f θ (x) = σ(Ax + b).

To draw the connection between tensor decompositions and feature tensors we consider the following score functions (logits 1 ): DISPLAYFORM1 where W ∈ R M ×M ×···×M is a trainable T -way weight tensor and Φ(X) ∈ R M ×M ×···×M is a rank 1 feature tensor, defined as DISPLAYFORM2 where we have used the operation of outer product ⊗, which is important in tensor calculus.

For a tensor A of order N and a tensor B of order M their outer product C = A ⊗ B is a tensor of order N + M defined as: DISPLAYFORM3 It is known that equation 2 possesses the universal approximation property (it can approximate any function with any prescribed precision given sufficiently large M ) under mild assumptions on f θ BID8 ).

Working the entire weight tensor W in eq. FORMULA1 is impractical for large M and T , since it requires exponential in T number of parameters.

Thus, we compactly represent it using tensor decompositions, which will further lead to different neural network architectures, referred to as tensor networks BID2 .

The most basic decomposition is the so-called Canonical (CP) decomposition BID11 BID1 which is defined as follows DISPLAYFORM0 where v (t) r ∈ R M and minimal value of R such that decomposition equation 5 exists is called canonical rank of a tensor (CP-rank).

By substituting eq. (5) into eq. (2) we find that DISPLAYFORM1 In the equation above, outer products ⊗ are taken between scalars and coincide with the ordinary products between two numbers.

However, we would like to keep this notation as it will come in handy later, when we generalize tensor decompositions to include various nonlinearities.

TT-decomposition Another tensor decomposition is Tensor Train (TT) decomposition (Oseledets, 2011) which is defined as follows DISPLAYFORM2 where g (t)rt−1rt ∈ R M and r 0 = r T = 1 by definition.

If we gather vectors g (t)rt−1rt for all corresponding indices r t−1 ∈ {1, . . .

, R t−1 } and r t ∈ {1, . . .

, R t } we will obtain three-dimensional tensors G (t) ∈ R M ×Rt−1×Rt (for t = 1 and t = T we will get matrices DISPLAYFORM3 is called TT-cores and minimal values of {R t } T −1 t=1 such that decomposition equation 7 exists are called TT-ranks.

In the case of TT decomposition, the score function has the following form: DISPLAYFORM4

Now we want to show that the score function for Tensor Train decomposition exhibits particular recurrent structure similar to that of RNN.

We define the following hidden states: DISPLAYFORM0 r0r1 , DISPLAYFORM1 Such definition of hidden states allows for more compact form of the score function.

Lemma 3.1.

Under the notation introduced in eq. (9), the score function can be written as DISPLAYFORM2 Proof of Lemma 3.1 as well as the proofs of our main results from Section 5 were moved to Appendix A due to limited space.

Note that with a help of TT-cores we can rewrite eq. (9) in a more convenient index form: DISPLAYFORM3 where the operation of tensor contraction is used.

Combining all weights from G (t) and f θ (·) into a single variable Θ (t)G and denoting the composition of feature map, outer product, and contraction as g : DISPLAYFORM4 Rt we arrive at the following vector form: G depend on the time step.

However, if we set DISPLAYFORM5 DISPLAYFORM6 we will get simplified hidden state equation used in standard recurrent architectures: DISPLAYFORM7 Note that this equation is applicable to all hidden states except for the first DISPLAYFORM8 and for the last DISPLAYFORM9 , due to two-dimensional nature of the corresponding TT-cores.

However, we can always pad the input sequence with two auxiliary vectors x (0) and x (T +1) to get full compliance with the standard RNN structure.

In the previous section we showed that tensor decompositions correspond to neural networks of specific structure, which are simplified versions of those used in practice as they contain multiplicative nonlinearities only.

One possible way to introduce more practical nonlinearities is to replace outer product ⊗ in eq. FORMULA5 and eq. (10) with a generalized operator ⊗ ξ in analogy to kernel methods when scalar product is replaced by nonlinear kernel function.

Let ξ : R × R → R be an associative and commutative binary operator (∀x, y, z ∈ R : ξ(ξ(x, y), z) = ξ(x, ξ(y, z)) and ∀x, y ∈ R : ξ(x, y) = ξ(y, x)).

Note that this operator easily generalizes to the arbitrary number of operands due to associativity.

For a tensor A of order N and a tensor B of order M we define their generalized outer product C = A ⊗ ξ B as an (N + M ) order tensor with entries given by: DISPLAYFORM0 Now we can replace ⊗ in eqs. FORMULA5 and FORMULA0 with ⊗ ξ and get networks with various nonlinearities.

For example, if we take ξ(x, y) = max(x, y, 0) we will get an RNN with rectifier nonlinearities; if we take ξ(x, y) = ln(e x + e y ) we will get an RNN with softplus nonlinearities; if we take ξ(x, y) = xy we will get a simple RNN defined in the previous section.

Concretely, we will analyze the following networks.

• Score function: DISPLAYFORM0 • Parameters of the network: DISPLAYFORM1 Generalized RNN with ξ-nonlinearity• Score function: DISPLAYFORM2 (16) • Parameters of the network: DISPLAYFORM3 Note that in eq. FORMULA0 we have introduced the matrices C (t) acting on the input states.

The purpose of this modification is to obtain the plausible property of generalized shallow networks being able to be represented as generalized RNNs of width 1 (i.e., with all R i = 1) for an arbitrary nonlinearity ξ.

In the case of ξ(x, y) = xy, the matrices C (t) were not necessary, since they can be simply absorbed by G (t) via tensor contraction (see Appendix A for further clarification on these points).Initial hidden state Note that generalized RNNs require some choice of the initial hidden state h (0) .

We find that it is convenient both for theoretical analysis and in practice to initialize h (0) as unit of the operator ξ, i.e. such an element u that ξ(x, y, u) = ξ(x, y) ∀x, y ∈ R. Henceforth, we will assume that such an element exists (e.g., for ξ(x, y) = max(x, y, 0) we take u = 0, for ξ(x, y) = xy we take u = 1), and set h (0) = u. For example, in eq. FORMULA10 it was implicitly assumed that h (0) = 1.

Introduction of generalized outer product allows us to investigate RNNs with wide class of nonlinear activation functions, especially ReLU.

While this change looks appealing from the practical viewpoint, it complicates following theoretical analysis, as the transition from obtained networks back to tensors is not straightforward.

In the discussion above, every tensor network had corresponding weight tensor W and we could compare expressivity of associated score functions by comparing some properties of this tensors, such as ranks BID12 .

This method enabled comprehensive analysis of score functions, as it allows us to calculate and compare their values for all possible input sequences X = x (1) , . . .

, x (T ) .

Unfortunately, we can not apply it in case of generalized tensor networks, as the replacement of standard outer product ⊗ with its generalized version ⊗ ξ leads to the loss of conformity between tensor networks and weight tensors.

Specifically, not for every generalized tensor network with corresponding score function (X) now exists a weight tensor W such that (X) = W, Φ(X) .

Also, such properties as universality no longer hold automatically and we have to prove them separately.

Indeed as it was noticed in shallow networks with ξ(x, y) = max(x, 0) + max(y, 0) no longer have the universal approximation property.

In order to conduct proper theoretical analysis, we adopt the apparatus of so-called grid tensors, first introduced in .Given a set of fixed vectors X = x (1) , . . .

, x (M ) referred to as templates, the grid tensor of X is defined to be the tensor of order T and dimension M in each mode, with entries given by: DISPLAYFORM0 where each index i t can take values from {1, . . .

, M }, i.e. we evaluate the score function on every possible input assembled from the template vectors {x DISPLAYFORM1 .

To put it simply, we previously considered the equality of score functions represented by tensor decomposition and tensor network on set of all possible input sequences X = x(1) , . . .

, x (T ) , x (t) ∈ R N , and now we restricted this set to exponentially large but finite grid of sequences consisting of template vectors only.

Define the matrix F ∈ R M ×M which holds the values taken by the representation function f θ : R N → R M on the selected templates X: DISPLAYFORM2 Using the matrix F we note that the grid tensor of generalized shallow network has the following form (see Appendix A for derivation): DISPLAYFORM3 Construction of the grid tensor for generalized RNN is a bit more involved.

We find that its grid tensor Γ (X) can be computed recursively, similar to the hidden state in the case of a single input sequence.

The exact formulas turned out to be rather cumbersome and we moved them to Appendix A.

With grid tensors at hand we are ready to compare the expressive power of generalized RNNs and generalized shallow networks.

In the further analysis, we will assume that ξ(x, y) = max(x, y, 0), i.e., we analyze RNNs and shallow networks with rectifier nonlinearity.

However, we need to make two additional assumptions.

First of all, similarly to we fix some templates X such that values of the score function outside of the grid generated by X are irrelevant for classification and call them covering templates.

It was argued that for image data values of M of order 100 are sufficient (corresponding covering template vectors may represent Gabor filters).

Secondly, we assume that the feature matrix F is invertible, which is a reasonable assumption and in the case of f θ (x) = σ(Ax + b) for any distinct template vectors X the parameters A and b can be chosen in such a way that the matrix F is invertible.

As was discussed in section 4.2 we can no longer use standard algebraic techniques to verify universality of tensor based networks.

Thus, our first result states that generalized RNNs with ξ(x, y) = max(x, y, 0) are universal in a sense that any tensor of order T and size of each mode being m can be realized as a grid tensor of such RNN (and similarly of a generalized shallow network).Theorem 5.1 (Universality).

Let H ∈ R M ×M ×···×M be an arbitrary tensor of order T .

Then there exist a generalized shallow network and a generalized RNN with rectifier nonlinearity ξ(x, y) = max(x, y, 0) such that grid tensor of each of the networks coincides with H.Part of Theorem 5.1 which corresponds to generalized shallow networks readily follows from (Cohen & Shashua, 2016, Claim 4) .

In order to prove the statement for the RNNs the following two lemmas are used.

Lemma 5.1.

Given two generalized RNNs with grid tensors Γ A (X), Γ B (X), and arbitrary ξ-nonlinearity, there exists a generalized RNN with grid tensor Γ C (X) satisfying DISPLAYFORM0 This lemma essentially states that the collection of grid tensors of generalized RNNs with any nonlinearity is closed under taking arbitrary linear combinations.

Note that the same result clearly holds for generalized shallow networks because they are linear combinations of rank 1 shallow networks by definition.

Lemma 5.2.

Let E (j1j2...j T ) be an arbitrary one-hot tensor, defined as DISPLAYFORM1 Then there exists a generalized RNN with rectifier nonlinearities such that its grid tensor satisfies DISPLAYFORM2 This lemma states that in the special case of rectifier nonlinearity ξ(x, y) = max(x, y, 0) any basis tensor can be realized by some generalized RNN.Proof of Theorem 5.1.

By Lemma 5.2 for each one-hot tensor E (i1i2...i T ) there exists a generalized RNN with rectifier nonlinearities, such that its grid tensor coincides with this tensor.

Thus, by Lemma 5.1 we can construct an RNN with DISPLAYFORM3 For generalized shallow networks with rectifier nonlinearities see the proof of , Claim 4).The same result regarding networks with product nonlinearities considered in BID12 directly follows from the well-known properties of tensor decompositions (see Appendix A).We see that at least with such nonlinearities as ξ(x, y) = max(x, y, 0) and ξ(x, y) = xy all the networks under consideration are universal and can represent any possible grid tensor.

Now let us head to a discussion of expressivity of these networks.

As was discussed in the introduction, expressivity refers to the ability of some class of networks to represent the same functions as some other class much more compactly.

In our case the parameters defining size of networks are ranks of the decomposition, i.e. in the case of generalized RNNs ranks determine the size of the hidden state, and in the case of generalized shallow networks rank determines the width of a network.

It was proven in BID12 that ConvNets and RNNs with multiplicative nonlinearities are exponentially more expressive than the equivalent shallow networks: shallow networks of exponentially large width are required to realize the same score functions as computed by these deep architectures.

Similarly to the case of ConvNets , we find that expressivity of generalized RNNs with rectifier nonlinearity holds only partially, as discussed in the following two theorems.

For simplicity, we assume that T is even.

Theorem 5.2 (Expressivity 1).

For every value of R there exists a generalized RNN with ranks ≤ R and rectifier nonlinearity which is exponentially more efficient than shallow networks, i.e., the corresponding grid tensor may be realized only by a shallow network with rectifier nonlinearity of width at least DISPLAYFORM0 This result states that at least for some subset of generalized RNNs expressivity holds: exponentially wide shallow networks are required to realize the same grid tensor.

Proof of the theorem is rather straightforward: we explicitly construct an example of such RNN which satisfies the following description.

Given an arbitrary input sequence X = x (1) , . . .

x (T ) assembled from the templates, these net- DISPLAYFORM1 , and 1 in every other case, i.e. they measure pairwise similarity of the input vectors.

A precise proof is given in Appendix A. In the case of multiplicative RNNs BID12 almost every network possessed this property.

This is not the case, however, for generalized RNNs with rectifier nonlinearities.

In other words, for every rank R we can find a set of generalized RNNs of positive measure such that the property of expressivity does not hold.

In the numerical experiments in Section 6 and Appendix A we validate whether this can be observed in practice, and find that the probability of obtaining CP-ranks of polynomial size becomes negligible with large T and R. Proof of Theorem 5.3 is provided in Appendix A.Shared case Note that all the RNNs used in practice have shared weights, which allows them to process sequences of arbitrary length.

So far in the analysis we have not made such assumptions about RNNs (i.e., G (2) = · · · = G (T −1) ).

By imposing this constraint, we lose the property of universality; however, we believe that the statements of Theorems 5.2 and 5.3 still hold (without requiring that shallow networks also have shared weights).

Note that the example constructed in the proof of Theorem 5.3 already has this property, and for Theorem 5.2 we provide numerical evidence in Appendix A.

In this section, we study if our theoretical findings are supported by experimental data.

In particular, we investigate whether generalized tensor networks can be used in practical settings, especially in problems typically solved by RNNs (such as natural language processing problems).

Secondly, according to Theorem 5.3 for some subset of RNNs the equivalent shallow network may have a low rank.

To get a grasp of how strong this effect might be in practice we numerically compute an estimate for this rank in various settings.

Performance For the first experiment, we use two computer vision datasets MNIST BID16 and CIFAR-10 (Krizhevsky & Hinton, 2009) , and natural language processing dataset for sentiment analysis IMDB BID17 .

For the first two datasets, we cut natural images into rectangular patches which are then arranged into vectors x (t) (similar to BID12 ) and for IMDB dataset the input data already has the desired sequential structure.

Figure 2 depicts test accuracy on IMDB dataset for generalized shallow networks and RNNs with rectifier nonlinearity.

We see that generalized shallow network of much higher rank is required to get the level of performance close to that achievable by generalized RNN.

Due to limited space, we have moved the results of the experiments on the visual datasets to Appendix B. Expressivity For the second experiment we generate a number of generalized RNNs with different values of TT-rank r and calculate a lower bound on the rank of shallow network necessary to realize the same grid tensor (to estimate the rank we use the same technique as in the proof of Theorem 5.2).

FIG4 shows that for different values of R and generalized RNNs of the corresponding rank there exist shallow networks of rank 1 realizing the same grid tensor, which agrees well with Theorem 5.3.

This result looks discouraging, however, there is also a positive observation.

While increasing rank of generalized RNNs, more and more corresponding shallow networks will necessarily have exponentially higher rank.

In practice we usually deal with RNNs of R = 10 2 − 10 3 (dimension of hidden states), thus we may expect that effectively any function besides negligible set realized by generalized RNNs can be implemented only by exponentially wider shallow networks.

The numerical results for the case of shared cores and other nonlinearities are given in Appendix B.

In this paper, we sought a more complete picture of the connection between Recurrent Neural Networks and Tensor Train decomposition, one that involves various nonlinearities applied to hidden states.

We showed how these nonlinearities could be incorporated into network architectures and provided complete theoretical analysis on the particular case of rectifier nonlinearity, elaborating on points of generality and expressive power.

We believe our results will be useful to advance theoretical understanding of RNNs.

In future work, we would like to extend the theoretical analysis to most competitive in practice architectures for processing sequential data such as LSTMs and attention mechanisms.

A PROOFS Lemma 3.1.

Under the notation introduced in eq. (9), the score function can be written as DISPLAYFORM0 Proof.

DISPLAYFORM1 rt−1rt h(1) r1 DISPLAYFORM2 r1r2 h(1) r1 DISPLAYFORM3 = . . .

DISPLAYFORM4 Proposition A.1.

If we replace the generalized outer product ⊗ ξ in eq. (16) with the standard outer product ⊗, we can subsume matrices C (t) into tensors G (t) without loss of generality.

Proof.

Let us rewrite hidden state equation eq. (16) after transition from ⊗ ξ to ⊗: DISPLAYFORM5 We see that the obtained expression resembles those presented in eq. (10) with TT-cores G (t) replaced byG (t) and thus all the reasoning applied in the absence of matrices C (t) holds valid.

Proposition A.2.

Grid tensor of generalized shallow network has the following form (eq. (20)): DISPLAYFORM6 denote an arbitrary sequence of templates.

Corresponding element of the grid tensor defined in eq. FORMULA1 has the following form: DISPLAYFORM7 Proposition A.3.

Grid tensor of a generalized RNN has the following form: DISPLAYFORM8 Proof.

Proof is similar to that of Proposition A.2 and uses eq. FORMULA0 to compute the elements of the grid tensor.

Lemma 5.1.

Given two generalized RNNs with grid tensors Γ A (X), Γ B (X), and arbitrary ξ-nonlinearity, there exists a generalized RNN with grid tensor Γ C (X) satisfying DISPLAYFORM9 Proof.

Let these RNNs be defined by the weight parameters DISPLAYFORM10 and DISPLAYFORM11 We claim that the desired grid tensor is given by the RNN with the following weight settings.

DISPLAYFORM12 It is straightforward to verify that the network defined by these weights possesses the following property: DISPLAYFORM13 , 0 < t < T, and h DISPLAYFORM14 B , concluding the proof.

We also note that these formulas generalize the well-known formulas for addition of two tensors in the Tensor Train format (Oseledets, 2011).Proposition A.4.

For any associative and commutative binary operator ξ, an arbitrary generalized rank 1 shallow network with ξ-nonlinearity can be represented in a form of generalized RNN with unit ranks (R 1 = · · · = R T −1 = 1) and ξ-nonlinearity.

DISPLAYFORM15 be the parameters specifying the given generalized shallow network.

Then the following weight settings provide the equivalent generalized RNN (with h (0) being the unity of the operator ξ).

DISPLAYFORM16 Indeed, in the notation defined above, hidden states of generalized RNN have the following form:Theorem 5.3 (Expressivity 2).

For every value of R there exists an open set (which thus has positive measure) of generalized RNNs with rectifier nonlinearity ξ(x, y) = max(x, y, 0), such that for each RNN in this open set the corresponding grid tensor can be realized by a rank 1 shallow network with rectifier nonlinearity.

Proof.

As before, let us denote by I (p,q) a matrix of size p × q such that I (p,q) ij = δ ij , and by a (p1,p2,...p d ) we denote a tensor of size p 1 × · · · × p d with each entry being a (sometimes we will omit the dimensions when they can be inferred from the context).

Consider the following weight settings for a generalized RNN.

DISPLAYFORM17 The RNN defined by these weights has the property that Γ (X) is a constant tensor with each entry being 2(M R) T −1 , which can be trivially represented by a rank 1 generalized shallow network.

We will show that this property holds under a small perturbation of C (t) , G (t) and F. Let us denote each of these perturbation (and every tensor appearing size of which can be assumed indefinitely small) collectively by ε.

Applying eq. FORMULA0 we obtain (with ξ(x, y) = max(x, y, 0)).

where we have used a simple property connecting ⊗ ξ with ξ(x, y) = max(x, y, 0) and ordinary ⊗: if for tensors A and B each entry of A is greater than each entry of B, A ⊗ ξ B = A ⊗ 1.

The obtained grid tensors can be represented using rank 1 generalized shallow networks with the following weight settings.

λ = 1, DISPLAYFORM18 DISPLAYFORM19 ε (2(MR) T−1 + ε), t = 1, 0, t > 1, where F ε is the feature matrix of the corresponding perturbed network.

In this section we provide the results additional computational experiments, aimed to provide more thorough and complete analysis of generalized RNNs.

Different ξ-nonlinearities In this paper we presented theoretical analysis of rectifier nonlinearity which corresponds to ξ(x, y) = max(x, y, 0).

However, there is a number of other associative binary operators ξ which can be incorporated in generalized tensor networks.

Strictly speaking, every one of them has to be carefully explored theoretically in order to speak about their generality and expressive power, but for now we can compare them empirically.

Table 1 shows the performance (accuracy on test data) of different nonlinearities on MNIST, CIFAR-10, and IMDB datasets for classification.

Although these problems are not considered hard to solve, we see that the right choice of nonlinearity can lead to a significant boost in performance.

For the experiments on the visual datasets we used T = 16, m = 32, R = 64 and for the experiments on the IMDB dataset we had T = 100, m = 50, R = 50.

Parameters of all networks were optimized using Adam (learning rate α = 10 −4 ) and batch size 250.

Expressivity in the case of shared cores We repeat the expressivity experiments from Section 6 in the case of equal TT-cores (G (2) = · · · = G (T −1) ).

We observe that similar to the case of different cores, there always exist rank 1 generalized shallow networks which realize the same score function as generalized RNN of higher rank, however, this situation seems too unlikely for big values of R. Figure 4 : Distribution of lower bounds on the rank of generalized shallow networks equivalent to randomly generated generalized RNNs of ranks (M = 6, T = 6, ξ(x, y) = max(x, y, 0)).Figure 5: Distribution of lower bounds on the rank of generalized shallow networks equivalent to randomly generated generalized RNNs of ranks (M = 6, T = 6, ξ(x, y) = x 2 + y 2 ).

@highlight

Analysis of expressivity and generality of recurrent neural networks with ReLu nonlinearities using Tensor-Train decomposition.