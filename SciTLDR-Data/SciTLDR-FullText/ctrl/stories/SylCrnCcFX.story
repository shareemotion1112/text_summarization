Deep networks realize complex mappings that are often understood by their locally linear behavior at or around points of interest.

For example, we use the derivative of the mapping with respect to its inputs for sensitivity analysis, or to explain (obtain coordinate relevance for) a prediction.

One key challenge is that such derivatives are themselves inherently unstable.

In this paper, we propose a new learning problem to encourage deep networks to have stable derivatives over larger regions.

While the problem is challenging in general, we focus on networks with piecewise linear activation functions.

Our algorithm consists of an inference step that identifies a region around a point where linear approximation is provably stable, and an optimization step to expand such regions.

We propose a novel relaxation to scale the algorithm to realistic models.

We illustrate our method with residual and recurrent networks on image and sequence datasets.

Complex mappings are often characterized by their derivatives at points of interest.

Such derivatives with respect to the inputs play key roles across many learning problems, including sensitivity analysis.

The associated local linearization is frequently used to obtain explanations for model predictions BID3 BID24 BID28 BID26 ; explicit first-order local approximations BID22 BID17 BID31 Koh & Liang, 2017; BID1 ; or used to guide learning through regularization of functional classes controlled by derivatives BID19 BID5 Mroueh et al., 2018) .

We emphasize that the derivatives discussed in this paper are with respect to the input coordinates rather than parameters.

The key challenge lies in the fact that derivatives of functions parameterized by deep learning models are not stable in general BID14 .

State-of-the-art deep learning models (He et al., 2016; Huang et al., 2017) are typically over-parametrized BID37 , leading to unstable functions as a by-product.

The instability is reflected in both the function values BID17 as well as the derivatives BID14 BID0 .

Due to unstable derivatives, first-order approximations used for explanations therefore also lack robustness BID14 BID0 .We note that gradient stability is a notion different from adversarial examples.

A stable gradient can be large or small, so long as it remains approximately invariant within a local region.

Adversarial examples, on the other hand, are small perturbations of the input that change the predicted output BID17 .

A large local gradient, whether stable or not in our sense, is likely to contribute to finding an adversarial example.

Robust estimation techniques used to protect against adversarial examples (e.g., (Madry et al., 2018) ) focus on stable function values rather than stable gradients but can nevertheless indirectly impact (potentially help) gradient stability.

A direct extension of robust estimation to ensure gradient stability would involve finding maximally distorted derivatives and require access to approximate Hessians of deep networks.

In this paper, we focus on deep networks with piecewise linear activations to make the problem tractable.

The special structure of this class of networks (functional characteristics) allows us to infer lower bounds on the p margin -the maximum radius of p -norm balls around a point where derivatives are provably stable.

In particular, we investigate the special case of p = 2 since the lower bound has an analytical solution, and permits us to formulate a regularization problem to maximize it.

The resulting objective is, however, rigid and non-smooth, and we further relax the learning problem in a manner resembling (locally) support vector machines (SVM) BID29 BID8 .Both the inference and learning problems in our setting require evaluating the gradient of each neuron with respect to the inputs, which poses a significant computational challenge.

For piecewise linear networks, given D-dimensional data, we propose a novel perturbation algorithm that collects all the exact gradients by means of forward propagating O(D) carefully crafted samples in parallel without any back-propagation.

When the GPU memory cannot fit O(D) samples in one batch, we develop an unbiased approximation to the objective with a random subset of such samples.

Empirically, we examine our inference and learning algorithms with fully-connected (FC), residual (ResNet) (He et al., 2016) , and recurrent (RNN) networks on image and time-series datasets with quantitative and qualitative experiments.

The main contributions of this work are as follows:• Inference algorithms that identify input regions of neural networks, with piecewise linear activation functions, that are provably stable.• A novel learning criterion that effectively expand regions of provably stable derivatives.• Novel perturbation algorithms that scale computation to high dimensional data.• Empirical evaluation with several types of networks.

For tractability reasons, we focus in this paper on neural networks with piecewise linear activation functions, such as ReLU BID15 and its variants (Maas et al., 2013; He et al., 2015; BID2 .

Since the nonlinear behavior of deep models is mostly governed by the activation function, a neural network defined with affine transformations and piecewise linear activation functions is inherently piecewise linear (Montufar et al., 2014) .

For example, FC, convolutional neural networks (CNN) (LeCun et al., 1998) , RNN, and ResNet (He et al., 2016) are all plausible candidates under our consideration.

We will call this kind of networks piecewise linear networks throughout the paper.

The proposed approach is based on a mixed integer linear representation of piecewise linear networks, activation pattern BID20 , which encodes the active linear piece (integer) of the activation function for each neuron; once an activation pattern is fixed, the network degenerates to a linear model (linear).

Thus the feasible set corresponding to an activation pattern in the input space is a natural region where derivatives are provably stable (same linear function).

Note the possible degenerate case where neighboring regions (with different activation patterns) nevertheless have the same end-to-end linear coefficients BID23 .

We call the feasible set induced by an activation pattern BID23 a linear region, and a maximal connected subset of the input space subject to the same derivatives of the network (Montufar et al., 2014) a complete linear region.

Activation pattern has been studied in various contexts, such as visualizing neurons BID13 , reachability of a specific output value (Lomuscio & Maganti, 2017) , its connection to vector quantization BID4 , counting the number of linear regions of piecewise linear networks BID20 Montúfar, 2017; BID23 , and adversarial attacks BID7 BID13 BID32 or defense .

Note the distinction between locally linear regions of the functional mapping and decision regions defined by classes BID36 Mirman et al., 2018; BID9 ).Here we elaborate differences between our work and the two most relevant categories above.

In contrast to quantifying the number of linear regions as a measure of complexity, we focus on the local linear regions, and try to expand them via learning.

The notion of stability we consider differs from adversarial examples.

The methods themselves are also different.

Finding the exact adversarial example is in general NP-complete (Katz et al., 2017; BID25 , and mixed integer linear programs that compute the exact adversarial example do not scale BID7 BID13 .

Layer-wise relaxations of ReLU activations BID32 are more scalable but yield bounds instead exact solutions.

Empirically, even relying on relaxations, the defense (learning) methods are still intractable on ImageNet scale images BID10 .

In contrast, our inference algorithm certifies the exact 2 margin around a point subject to its activation pattern by forwarding O(D) samples in parallel.

In a high-dimensional setting, where it is computationally challenging to compute the learning objective, we develop an unbiased estimation by a simple sub-sampling procedure, which scales to ResNet (He et al., 2016) on 299 × 299 × 3 dimensional images in practice.

The proposed learning algorithm is based on the inference problem with 2 margins.

The derivation is reminiscent of the SVM objective BID29 BID8 , but differs in its purpose; while SVM training seeks to maximize the 2 margin between data points and a linear classifier, our approach instead maximizes the 2 margin of linear regions around each data point.

Since there is no label information to guide the learning algorithm for each linear region, the objective is unsupervised and more akin to transductive/semi-supervised SVM (TSVM) BID30 BID6 .

In the literature, the idea of margin is also extended to nonlinear classifiers in terms of decision boundaries BID12 .

Concurrently, BID9 also leverages the (raw) p margin on small networks for adversarial training.

In contrast, we develop a smooth relaxation of the p margin and novel perturbation algorithms, which scale the computation to realistic networks, for gradient stability.

The problem we tackle has implications for interpretability and transparency of complex models.

The gradient has been a building block for various explanation methods for deep models, including gradient saliency map BID24 and its variants BID27 BID28 BID26 , which apply a gradient-based attribution of the prediction to the input with nonlinear post-processings for visualization (e.g., normalizing and clipping by the 99 th percentile BID26 BID28 ).

While one of the motivations for this work is the instability of gradient-based explanations BID14 BID0 , we focus more generally on the fundamental problem of establishing robust derivatives.

To simplify the exposition, the approaches are developed under the notation of FC networks with ReLU activations, which naturally generalizes to other settings.

We first introduce notation, and then present our inference and learning algorithms.

All the proofs are provided in Appendix A.

We consider a neural network θ with M hidden layers and N i neurons in the i th layer, and the corresponding function f θ : R D → R L it represents.

We use z i ∈ R Ni and a i ∈ R Ni to denote the vector of (raw) neurons and activated neurons in the i th layer, respectively.

We will use x and a 0 interchangeably to represent an input instance from R D = R N0 .

With an FC architecture and ReLU activations, each a i and z i are computed with the transformation matrix W i ∈ R Ni×Ni−1 and bias DISPLAYFORM0 where [M ] denotes the set {1, . . .

, M }.

We use subscript to further denote a specific neuron.

To avoid confusion from other instancesx ∈ R D , we assert all the neurons z i j are functions of the specific instance denoted by x. The output of the network is a linear transformation of the last hidden layer DISPLAYFORM1 The output can be further processed by a nonlinearity such as softmax for classification problems.

However, we focus on the piecewise linear property of neural networks represented by f θ (x), and leverage a generic loss function L(f θ (x), y) to fold such nonlinear mechanism.

We use D to denote the set of training data (x, y), D x to denote the same set without labels y, and B ,p (x) := {x ∈ R D : x − x p ≤ } to denote the p -ball around x with radius .The activation pattern BID20 used in this paper is defined as: Definition 1. (Activation Pattern) An activation pattern is a set of indicators for neurons O = {o i ∈ {−1, 1} Ni |i ∈ [M ]} that specifies the following functional constraints: DISPLAYFORM2 i j is called an activation indicator.

Note that a point on the boundary of a linear region is feasible for multiple activation patterns.

The definition fits the property of the activation pattern discussed in §2.

We define ∇ x z i j to be the sub-gradient found by back-propagation using ∂a DISPLAYFORM3 DISPLAYFORM4 , and the feasible set of the activation pattern is equivalent to DISPLAYFORM5 Remark 3.

Lemma 2 characterizes each linear region of f θ as the feasible set S(x) with a set of linear constraints with respect to the input space R D , and thus S(x) is a convex polyhedron.

The aforementioned linear property of an activation pattern equipped with the input space constraints from Lemma 2 yield the definition ofˆ x,p , the p margin of x subject to its activation pattern: DISPLAYFORM6 where S(x) can be based on any feasible activation pattern O on x; 3 therefore, ∂a DISPLAYFORM7 } is ensured with respect to some feasible activation pattern O. Note thatˆ x,p is a lower bound of the p margin subject to a derivative specification (i.e., a complete linear region).3.2.1 DIRECTIONAL VERIFICATION, THE CASES p = 1 AND p = ∞ We first exploit the convexity of S(x) to check the feasibility of a directional perturbation.

Proposition 4. (Directional Feasibility) Given a point x, a feasible set S(x) and a unit vector ∆x, if ∃¯ ≥ 0 such that x +¯ ∆x ∈ S(x), then f θ is linear in {x + ∆x : 0 ≤ ≤¯ }.The feasibility of x +¯ ∆x ∈ S(x) can be computed by simply checking whether x +¯ ∆x satisfies the activation pattern O in S(x).

Proposition 4 can be applied to the feasibility problem on 1 -balls.

Proposition 5. ( 1 -ball Feasibility) Given a point x, a feasible set S(x), and an 1 -ball B ,1 (x) with extreme points x 1 , . . . , DISPLAYFORM8 Proposition 5 can be generalized for an ∞ -ball.

However, in high dimension D, the number of extreme points of an ∞ -ball is exponential to D, making it intractable.

Instead, the number of extreme points of an 1 -ball is only linear to D (+ and − for each dimension).

With the above methods to verify feasibility, we can do binary searches to find the certificates of the margins for directional perturbationsˆ x,∆x := max { ≥0:x+ ∆x∈S(x)} and 1 -ballsˆ x,1 .

The details are in Appendix B.

The feasibility onˆ x,1 is tractable due to convexity of S(x) and its certification is efficient by a binary search; by further exploiting the polyhedron structure of S(x),ˆ x,2 can be certified analytically.

Proposition 6. ( 2 -ball Certificate) Given a point x,ˆ x,2 is the minimum 2 distance between x and the union of hyperplanes DISPLAYFORM0 To compute the 2 distance between x and the hyperplane induced by a neuron z i j , we evaluate DISPLAYFORM1 where all the z i j can be computed by a single forward pass.

4 We will show in §4.1 that all the ∇ x z i j can also be computed efficiently by forward passes in parallel.

We refer readers to FIG3 to see a visualization of the certificates on 2 margins.

The sizes of linear regions are related to their overall number, especially if we consider a bounded input space.

Counting the number of linear regions in f θ is, however, intractable due to the combinatorial nature of the activation patterns BID23 .

We argue that counting the number of linear regions on the whole space does not capture the structure of data manifold, and we propose to certify the number of complete linear regions (#CLR) of f θ among the data points D x , which turns out to be efficient to compute given a mild condition.

Here we use #A to denote the cardinality of a set A, and we have Lemma 7. (Complete Linear Region Certificate) If every data point x ∈ D x has only one feasible activation pattern denoted as O(x), the number of complete linear regions of f θ among D x is upperbounded by the number of different activation patterns #{O(x)|x ∈ D x }, and lower-bounded by the number of different Jacobians #{J x f θ (x)|x ∈ D x }.

In this section, we focus on methods aimed at maximizing the 2 marginˆ x,2 , since it is (sub-)differentiable.

We first formulate a regularization problem in the objective to maximize the margin: DISPLAYFORM0 However, the objective itself is rather rigid due to the inner-minimization and the reciprocal of ∇ x z i j 2 .

Qualitatively, such rigid loss surface hinders optimization and may attend infinity.

To alleviate the problem, we do a hinge-based relaxation to the distance function similar to SVM.

FORMULA12 is also optimal for Eq. (4).

DISPLAYFORM1 If the condition in Lemma 8 does not hold, Eq. FORMULA12 is still a valid upper bound of Eq. (4) due to a smaller feasible set.

An upper bound of Eq. FORMULA12 can be obtained consequently due to the constraints: DISPLAYFORM2 We then derive a relaxation that solves a smoother problem by relaxing the squared root and reciprocal on the 2 norm as well as the hard constraint with a hinge loss to a soft regularization problem: DISPLAYFORM3 where C is a hyper-parameter.

The relaxed regularization problem can be regarded as a maximum aggregation of TSVM losses among all the neurons, where a TSVM loss with only unannotated data D x can be written as: min DISPLAYFORM4 4 Concurrently, BID9 find that the p marginˆ x,p can be similarly computed as which pursues a similar goal to maximize the 2 margin in a linear model scenario, where the margin is computed between a linear hyperplane (the classifier) and the training points.

DISPLAYFORM5 To visualize the effect of the proposed methods, we make a toy 2D binary classification dataset, and train a 4-layer fully connected network with 1) (vanilla) binary cross-entropy loss L(·, ·), 2) distance regularization as in Eq. (4) , and 3) relaxed regularization as in Eq. FORMULA14 .

Implementation details are in Appendix F.

The resulting piecewise linear regions and prediction heatmaps along with gradient ∇ x f θ (x) annotations are shown in FIG3 .

The distance regularization enlarges the linear regions around each training point, and the relaxed regularization further generalizes the property to the whole space; the relaxed regularization possesses a smoother prediction boundary, and has a special central region where the gradients are 0 to allow gradients to change directions smoothly.

Since a linear region is shaped by a set of neurons that are "close" to a given a point, a noticeable problem of Eq. FORMULA14 is that it only focuses on the "closest" neuron, making it hard to scale the effect to large networks.

Hence, we make a generalization to the relaxed loss in Eq. (7) with a set of neurons that incur high losses to the given point.

We denoteÎ(x, γ) as the set of neurons with top γ percent relaxed loss (TSVM loss) on x. The generalized loss is our final objective for learning RObust Local Linearity (ROLL) and is written as: DISPLAYFORM0 A special case of Eq. FORMULA17 is when γ = 100 (i.e.

Î(x, 100) = I), where the nonlinear sorting step effectively disappears.

Such simple additive structure without a nonlinear sorting step can stabilize the training process, is simple to parallelize computation, and allows for an approximate learning algorithm as will be developed in §4.2.

Besides, taking γ = 100 can induce a strong synergy effect, as all the gradient norms ∇ x z i j 2 2 in Eq. (9) between any two layers are highly correlated.

The 2 marginˆ x,2 and the ROLL loss in Eq. (9) demands heavy computation on gradient norms.

While calling back-propagation |I| times is intractable, we develop a parallel algorithm without calling a single back-propagation by exploiting the functional structure of f θ .Given an activation pattern, we know that each hidden neuron z i j is also a linear function of x ∈ S(x).

We can construct another linear network g θ that is identical to f θ in S(x) based on the same set of parameters but fixed linear activation functions constructed to mimic the behavior of f θ in S(x).

Due to the linearity of g θ , the derivatives of all the neurons to an input axis can be computed by forwarding two samples: subtracting the neurons with an one-hot input from the same neurons with a zero input.

The procedure can be amortized and parallelized to all the dimensions by feeding To analyze the complexity of the proposed approach, we assume that parallel computation does not incur any overhead and a batch matrix multiplication takes a unit operation.

To compute the gradients of all the neurons for a batch of inputs, our perturbation algorithm takes 2M operations, while back-propagation takes DISPLAYFORM0 The detailed analysis is also in Appendix C.

Despite the parallelizable computation of ∇ x z i j , it is still challenging to compute the loss for large networks in a high dimension setting, where even calling D + 1 forward passes in parallel as used in §4.1 is infeasible due to memory constraints.

Hence we propose an unbiased estimator of the ROLL loss in Eq. FORMULA17 whenÎ(x, γ) = I. Note that (i,j)∈I C max(0, 1 − |z i j |) is already computable in one single forward pass.

For the sum of gradient norms, we use the following equivalent decoupling: DISPLAYFORM0 where the summation inside the expectation in the last equation can be efficiently computed using the procedure in §4.1 and is in general storable within GPU memory.

In practice, we can uniformly sample D (1 ≤ D D) input axes to have an unbiased approximation to Eq. (10), where computing all the partial derivatives with respect to D axes only requires D + 1 times memory (one hot vectors and a zero vector) than a typical forward pass for x.

The proposed algorithms can be used on all the deep learning models with affine transformations and piecewise linear activation functions by enumerating every neuron that will be imposed an ReLU-like activation function as z i j .

They do not immediately generalize to the nonlinearity of maxout/max-pooling BID16 ) that also yields a piecewise linear function.

We provide an initial step towards doing so in the Appendix E, but we suggest to use an average-pooling or convolution with large strides instead, since they do not induce extra linear constraints as maxpooling and do not in general yield significant difference in performance BID27 .

In this section, we compare our approach ('ROLL') with a baseline model with the same training procedure except the regularization ('vanilla') in several scenarios.

All the reported quantities are computed on a testing set.

Experiments are run on single GPU with 12G memory.

Evaluation Measures: 1) accuracy (ACC), 2) number of complete linear regions (#CLR), and 3) p margins of linear regionsˆ x,p .

We compute the marginˆ x,p for each testing point x with p ∈ {1, 2}, and we evaluateˆ x,p on 4 different percentiles P 25 , P 50 , P 75 , P 100 among the testing data.

DISPLAYFORM0 Figure 2: Parameter analysis on MNIST dataset.

P 50 ofˆ x,2 is the median ofˆ x,2 in the testing data.

We use a 55, 000/5, 000/10, 000 split of MNIST dataset for training/validation/testing.

Experiments are conducted on a 4-layer FC model with ReLU activations.

The implementation details are in Appendix G. We report the two models with the largest medianˆ x,2 among validation data given the same and 1% less validation accuracy compared to the baseline model.

The results are shown in TAB1 .

The tuned models have γ = 100, λ = 2, and different C as shown in the table.

The condition in Lemma 7 for certifying #CLR is satisfied with tight upper bound and lower bound, so a single number is reported.

Given the same performance, the ROLL loss achieves about 10 times larger margins for most of the percentiles than the vanilla loss.

By tradingoff 1% accuracy, about 30 times larger margins can be achieved.

The Spearman's rank correlation betweenˆ x,1 andˆ x,2 among testing data is at least 0.98 for all the cases.

The lower #CLR in our approach than the baseline model reflects the existence of certain larger linear regions that span across different testing points.

All the points inside the same linear region in the ROLL model with ACC= 98% have the same label, while there are visually similar digits (e.g., 1 and 7) in the same linear region in the other ROLL model.

We do a parameter analysis in Figure 2 with the ACC and P 50 ofˆ x,2 under different C, λ and γ when the other hyper-parameters are fixed.

As expected, with increased C and λ, the accuracy decreases with an increased 2 margin.

Due to the smoothness of the curves, higher γ values reflect less sensitivity to hyper-parameters C and λ.

To validate the efficiency of the proposed method, we measure the running time for performing a complete mini-batch gradient descent step (starting from the forward pass) on average.

We compare 1) the vanilla loss, 2) the full ROLL loss (γ = 100) in Eq. (9) computed by back-propagation, 3) the same as 2) but computed by our perturbation algorithm, and 4) the approximate ROLL loss in Eq. (10) computed by perturbations.

The approximation is computed with 3 = D/256 samples.

The results are shown in TAB2 .

The accuracy and 2 margins of the approximate ROLL loss are comparable to the full loss.

Overall, our approach is only twice slower than the vanilla loss.

The approximate loss is about 9 times faster than the full loss.

Compared to back-propagation, our perturbation algorithm achieves about 12 times empirical speed-up.

In summary, the computational overhead of our method is minimal compared to the vanilla loss, which is achieved by the perturbation algorithm and the approximate loss.

Table 4 : ResNet on Caltech-256.

Here ∆(x, x , y) denotes 1 gradient distortion ∇ x f θ (x ) y − ∇ x f θ (x) y 1 (the smaller the better for each r percentile P r among the testing data).

DISPLAYFORM1

We train RNNs for speaker identification on a Japanese Vowel dataset from the UCI machine learning repository BID11 with the official training/testing split.

6 The dataset has variable sequence length between 7 and 29 with 12 channels and 9 classes.

We implement the network with the state-of-the-art scaled Cayley orthogonal RNN (scoRNN) (Helfrich et al., 2018) , which parameterizes the transition matrix in RNN using orthogonal matrices to prevent gradient vanishing/exploding, with LeakyReLU activation.

The implementation details are in Appendix H. The reported models are based on the same criterion as §5.1.The results are reported in TAB3 .

With the same/1% inferior ACC, our approach leads to a model with about 4/20 times larger margins among the percentiles on testing data, compared to the vanilla loss.

The Spearman's rank correlation betweenˆ x,1 andˆ x,2 among all the cases are 0.98.

We also conduct sensitivity analysis on the derivatives by findingˆ x,∆x along each coordinate ∆x ∈ ∪ i ∪ 12 j=1{−e i,j , e i,j } (e i,j k,l = 0, ∀k, l except e i,j i,j = 1), which identifies the stability bounds [ˆ x,−e i,j ,ˆ x,e i,j ] at each timestamp i and channel j that guarantees stable derivatives.

The visualization using the vanilla and our ROLL model with 98% ACC is in FIG4 .

Qualitatively, the stability bound of the ROLL regularization is consistently larger than the vanilla model.

We conduct experiments on Caltech-256 BID18 , which has 256 classes, each with at least 80 images.

We downsize the images to 299 × 299 × 3 and train a 18-layer ResNet (He et al., 2016) with initializing from parameters pre-trained on ImageNet BID10 ).

The approximate ROLL loss in Eq. (10) is used with 120 random samples on each channel.

We randomly select 5 and 15 samples in each class as the validation and testing set, respectively, and put the remaining data into the training set.

The implementation details are in Appendix I.Evaluation Measures: Due to high input dimensionality (D ≈ 270K), computing the certificateŝ x,1 ,ˆ x,2 is computationally challenging without a cluster of GPUs.

Hence, we turn to a samplebased approach to evaluate the stability of the gradients f θ (x) y for the ground-truth label in a local region with a goal to reveal the stability across different linear regions.

Note that evaluating the gradient of the prediction instead is problematic to compare different models in this case.

Given labeled data (x, y), we evaluate the stability of gradient ∇ x f θ (x) y in terms of expected 1 distortion (over a uniform distribution) and the maximum 1 distortion within the intersection B ,∞ (x) = B ,∞ (x) ∩ X of an ∞ -ball and the domain of images X = [0, 1] 299×299×3 .

The 1 gradient distortion is defined as ∆(x, x , y) := ∇ x f θ (x ) y − ∇ x f θ (x) y 1 .

For a fixed x, we refer to Figure 4 : Visualization of the examples in Caltech-256 that yield the P 50 (above) and P 75 (below) of the maximum 1 gradient distortions among the testing data on our ROLL model.

The adversarial gradient is found by maximizing the distortion ∆(x, x , y) over the ∞ -norm ball with radius 8/256.

the maximizer ∇ x f θ (x ) y as the adversarial gradient.

Computation of the maximum 1 distortion requires optimization, but gradient-based optimization is not applicable since the gradient of the loss involves the Hessian ∇ 2 x f θ (x ) y which is either 0 or ill-defined due to piecewise linearity.

Hence, we use a genetic algorithm BID33 for black-box optimization.

Implementation details are provided in Appendix J. We use 8000 samples to approximate the expected 1 distortion.

Due to computational limits, we only evaluate 1024 random images in the testing set for both maximum and expected 1 gradient distortions.

The ∞ -ball radius is set to 8/256.The results along with precision at 1 and 5 (P@1 and P@5) are presented in Table 4 .

The ROLL loss yields more stable gradients than the vanilla loss with marginally superior precisions.

Out of 1024 examined examples x, only 40 and 42 gradient-distorted images change prediction labels in the ROLL and vanilla model, respectively.

We visualize some examples in Figure 4 with the original and adversarial gradients for each loss.

Qualitatively, the ROLL loss yields stable shapes and intensities of gradients, while the vanilla loss does not.

More examples with integrated gradient attributions BID28 are provided in Appendix K.

This paper introduces a new learning problem to endow deep learning models with robust local linearity.

The central attempt is to construct locally transparent neural networks, where the derivatives faithfully approximate the underlying function and lends itself to be stable tools for further applications.

We focus on piecewise linear networks and solve the problem based on a margin principle similar to SVM.

Empirically, the proposed ROLL loss expands regions with provably stable derivatives, and further generalize the stable gradient property across linear regions.

DISPLAYFORM0 , and the feasible set of the activation pattern is equivalent to DISPLAYFORM1 Ifx is feasible to the fixed activation pattern o 1 j , it is equivalent to thatx satisfies the linear constraint DISPLAYFORM2 in the first layer.

Assumex has satisfied all the constraints before layer i > 1.

We know if all the previous layers follows the fixed activation indicators, it is equivalent to rewrite each DISPLAYFORM3 Then for j ∈ [N i ], it is clear that z DISPLAYFORM4 The proof follows by induction.

Proposition 4. (Directional Feasibility) Given a point x, a feasible set S(x) and a unit vector ∆x, if ∃¯ ≥ 0 such that x +¯ ∆x ∈ S(x), then f θ is linear in {x + ∆x : 0 ≤ ≤¯ }.Proof.

Since S(x) is a convex set and x, x +¯ ∆x ∈ S(x), {x + ∆x : 0 ≤ ≤¯ } ⊆ S(x).

Proposition 5.( 1 -ball Feasibility) Given a point x, a feasible set S(x), and an 1 -ball B ,1 (x) with extreme points DISPLAYFORM0 Proof.

S(x) is a convex set and DISPLAYFORM1 .

Hence, ∀x ∈ B ,1 (x), we know x is a convex combination of x 1 , . . .

, x 2D , which implies x ∈ S(x).

Proposition 6. ( 2 -ball Certificate) Given a point x,ˆ x,2 is the minimum 2 distance between x and the union of hyperplanes DISPLAYFORM0 Proof.

Since S(x) is a convex polyhedron and x ∈ S(x), B ,2 (x) ⊆ S(x) is equivalent to the statement: the hyperplanes induced from the linear constraints in S(x) are away from x for at least in 2 distance.

Accordingly, the minimizing 2 distance between x and the hyperplanes is the maximizing distance that satisfies B ,2 (x) ⊆ S(x).

FORMULA12 is also optimal for Eq. (4).

DISPLAYFORM1 Proof.

The proof is based on constructing a neural network feasible in Eq. (5) that has the same loss as the optimal model in Eq. (4).

Since the optimum in Eq. FORMULA12 DISPLAYFORM2 C PARALLEL COMPUTATION OF THE GRADIENTS BY LINEARITYWe denote the corresponding neurons z i j and a DISPLAYFORM3 givenx, highlighting its functional relationship with respect to a new inputx.

The network g θ is constructed with exactly the same weights and biases as f θ but with a well-crafted linear activation function o i j = max(0, o i j ) ∈ {0, 1}. Note that since o is given,ô is fixed.

Then each layer in g θ is represented as: DISPLAYFORM4 We note thatâ i (x),ô i , andẑ i (x) are also functions of x, which we omitted for simplicity.

Since the new activation functionô is fixed given x, effectively it applies the same linearity toẑ DISPLAYFORM5 We then do the following procedure to collect the partial derivatives with respect to an input axis k: 1) feed a zero vector 0 to g θ to getẑ i j (0) and 2) feed a unit vector e k on the axis to getẑ i j (e k ).

Then the derivative of each neuron z i j with respect to x k can be computed aŝ DISPLAYFORM6 where the first equality comes from the linearity ofẑ i j (x) with respect to anyx.

With the procedure, the derivative of all the neurons to an input dimension can be computed with 2 forward pass, which can be further scaled by computing all the gradients of z To analyze the complexity of the proposed approach, we assume that parallel computation does not incur any overhead and a batch matrix multiplication takes a unit operation.

In this setting, a typical forward pass up to the last hidden layer takes M operations.

To compute the gradients of all the neurons for a batch of inputs, our perturbation algorithm first takes a forward pass to obtain the activation patterns for the batch of inputs, and then takes another forward pass with perturbations to obtain the gradients.

Since both forward passes are done up to the last hidden layers, it takes 2M operations in total.

In contrast, back-propagation cannot be parallelized among neurons, so computing the gradients of all the neurons must be done sequentially.

For each neuron z i j , it takes 2i operations for backpropagation to compute its gradient (i operations for each of the forward and backward pass).

Hence, it takes M i=1 2iN i operations in total for back-propagations to compute the same thing.

We can exploit the chain-rule of Jacobian to do dynamic programming for computing all the gradients of z i j .

Note that all the gradients of z i j in the i th layer can be represented by the Jacobian J x z i .Then 1) For the first layer, the Jacobian is trivially J x z 1 = W 1 .

2) We then iterate higher layers with the Jacobian of previous layers by chain rules J x z i = W i J z i−1 a i−1 J x z i−1 , where J x z i−1 and W i are stored and J z i−1 a i−1 is simply the Jacobian of activation function (a diagonal matrix with 0/1 entries for ReLU activations).

The dynamic programming approach is efficient for fully connected networks, but is inefficient for convolutional layers, where explicitly representing the convolutional operation in the form of linear transformation (∈ R Ni+1×Ni ) is expensive.

Here we only make an introductory guide to the derivations for maxout/max-pooling nonlinearity.

The goal is to highlight that it is feasible to derive inference and learning methods upon a piecewise linear network with max-pooling nonlinearity, but we do not suggest to use it since a max-pooling neuron would induce new linear constraints; instead, we suggest to use convolution with large strides or average-pooling which do not incur any constraint.

For simplicity, we assume the target network has a single nonlinearity, which maps N neurons to 1 output by the maximum a DISPLAYFORM0 Then we can define the corresponding activation pattern o = o 1 1 ∈ [N ] as which input is selected: DISPLAYFORM1 It is clear to see once an activation pattern is fixed, the network again degenerates to a linear model, as the nonlinearity in the max-pooling effectively disappears.

Such activation pattern induces a feasible set in the input space where derivatives are guaranteed to be stable, but such representation may have a similar degenerate case where two activation patterns yield the same linear coefficients.

The feasible set S(x) of a feasible activation pattern O = {o 1 1 } at x can be derived as: DISPLAYFORM2 To check its correctness, we know that Eq. FORMULA0 is equivalent to DISPLAYFORM3 where the linear constraints are evident, and the feasible set is thus again a convex polyhedron.

As a result, all the inference and learning algorithms can be applied with the linear constraints.

Clearly, for each max-pooling neuron with N inputs, it will induce N − 1 linear constraints.

The FC model consists of M = 4 fully-connected hidden layers, where each hidden layer has 100 neurons.

The input dimension D is 2 and the output dimension L is 1.

The loss function L(f θ (x), y) is sigmoid cross entropy.

We train the model for 5000 epochs with Adam (Kingma & Ba, 2015) optimizer, and select the model among epochs based on the training loss.

We fix C = 5, and increase λ ∈ {10 −2 , . . . , 10 2 } for both the distance regularization and relaxed regularization problems until the resulting classifier is not perfect.

The tuned λ in both cases are 1.

The data are normalized with µ = 0.1307 and σ = 0.3081.

We first compute the marginˆ x,p in the normalized data, and report the scaled margin σˆ x,p in the table, which reflects the actual margin in the original data space since DISPLAYFORM0 so the reported margin should be perceived in the data space of X = [0, 1] 28×28 .We compute the exact ROLL loss during training (i.e., approximate learning is not used).

The FC model consists of M = 4 fully-connected hidden layers, where each hidden layer has 300 neurons.

The activation function is ReLU.

The loss function L(f θ (x), y) is a cross-entropy loss with soft-max performed on f θ (x).

The number of epochs is 20, and the model is chosen from the best validation loss from all the epochs.

We use stochastic gradient descent with Nesterov momentum.

The learning rate is 0.01, the momentum is 0.5, and the batch size is 64.Tuning: We do a grid search on λ, C, γ, with λ ∈ {2 −3 , . . .

, 2 2 }, C ∈ {2 −2 , . . . , 2 3 }, γ ∈ {max, 25, 50, 75, 100} (max refers to Eq. FORMULA14 ), and report the models with the largest validation 2,50 given the same and 1% less validation accuracy compared to the baseline model (the vanilla loss).

The data are not normalized.

We compute the exact ROLL loss during training (i.e., approximate learning is not used).

The representation is learned with a single layer scoRNN, where the state embedding from the last timestamp for each sequence is treated as the representation along with a fully-connected layer to produce a prediction as f θ (x).

We use LeakyReLU as the activation functions in scoRNN.

The dimension of hidden neurons in scoRNN is set to 512.

The loss function L(f θ (x), y) is a cross-entropy loss with soft-max performed on f θ (x).

We use AMSGrad optimizer BID21 .

The learning rate is 0.001, and the batch size is 32 (sequences).Tuning: We do a grid search on λ ∈ {2 −6 , . . .

, 2 3 }, C ∈ {2 −5 , . . . , 2 7 }, and set γ = 100.

The models with the largest testingˆ 2,50 given the same and 1% less testing accuracy compared to the baseline model (the vanilla loss) are reported.

along each channel.

We train models on the normalized images, and establish a bijective mapping between the normalized distance and the distance in the original space with the trick introduced in Appendix G. The bijection is applied to our sample-based approach to compute We download the pre-trained ResNet-18 (He et al., 2016) from PyTorch (Paszke et al., 2017), and we revise the model architecture as follows: 1) we replace the max-pooling after the first convolutional layer with average-pooling to reduce the number of linear constraints (because max-pooling induces additional linear constraints on activation pattern, while average-pooling does not), and 2) we enlarge the receptive field of the last pooling layer such that the output will be 512 dimension, since ResNet-18 is originally used for smaller images in ImageNet data (most implementations use 224 × 224 × 3 dimensional images for ImageNet while our data has even higher dimension 299 × 299 × 3).

DISPLAYFORM0 We train the model with stochastic gradient descent with Nesterov momentum for 20 epochs.

The initial learning rate is 0.005, which is adjusted to 0.0005 after the first 10 epochs.

The momentum is 0.5.

The batch size is 32.

The model achieving the best validation loss among the 20 epochs is selected.

Tuning: Since the training is computationally demanding, we first fix C = 8, use only 18 samples (6 per channel) for approximate learning, and tune λ ∈ {10 −6 , 10 −5 , . . . } until the model yields significantly inferior validation accuracy than the vanilla model.

Afterwards, we fix λ to the highest plausible value (λ = 0.001) and try to increase C ∈ {8, 80, . . . }, but we found that C = 8 is already the highest plausible value.

Finally, we train a model with 360 random samples (120 per channel) for approximate learning to improve the quality of approximation.

We implement a genetic algorithm (GA) BID33 with 4800 populations P and 30 epochs.

Initially, we first uniformly sample 4800 samples (called chromosome in GA literature) in the domain B ,∞ (x) ∩ X for P. In each epoch, 1.

∀c ∈ P, we evaluate the 1 distance of its gradient from that of the target x: DISPLAYFORM0 2. (Selection) we sort the samples based on the 1 distance and keep the top 25% samples in the population (denoted asP).3. (Crossover) we replace the remaining 75% samples with a random linear combination of a pair (c, c ) fromP as: DISPLAYFORM1 4. (Projection) For all the updated samples c ∈ P, we do an ∞ -projection to the domain B ,∞ (x) ∩ X to ensure the feasibility.

Finally, the sample in P that achieves the maximum 1 distance is returned.

We didn't implement mutation in our GA algorithm due to computational reasons.

For the readers who are not familiar with GA, we comment that the crossover operator is analogous to a gradient step where the direction is determined by other samples and the step size is determined randomly.

We visualize the following images:• Original image.• Original gradient: the gradient on the original image.• Adversarial gradient: the maximum 1 distorted gradient in B ,∞ (x) ∩ X .•

Image of adv.

gradient: the image that yields adversarial gradient.• Original int.

gradient: the integrated gradient attribution BID28 on the original image.• Adversarial int.

gradient: the integrated gradient attribution BID28 on the 'image of adv.

gradient'.

Note that we didn't perform optimization to find the image that yields the maximum distorted integrated gradient.

We follow a common implementation in the literature BID26 BID28 to visualize gradients and integrated gradients by the following procedure:1.

Aggregating derivatives in each channel by summation.2.

Taking absolute value of aggregated derivatives.3.

Normalizing the aggregated derivatives by the 99 th percentile 4.

Clipping all the values above 1.After this, the derivatives are in the range [0, 1] 299×299 , which can be visualized as a gray-scaled image.

The original integrated gradient paper visualizes the element-wise product between the grayscaled integrated gradient and the original image, but we only visualize the integrated gradient to highlight its difference in different settings since the underlying images (the inputs) are visually indistinguishable.

We visualize the examples in Caltech-256 dataset that yield the P 25 , P 50 , P 75 , P 100 (P r denotes the r th percentile) of the maximum 1 gradient distortions among the testing data on our ROLL model in Figure 5 and 6, where the captions show the exact values of the maximum 1 gradient distortion for each image.

Note that the exact values are slightly different from Table 4 , because each percentile in Table 4 is computed by an interpolation between the closest ranks (as in numpy.percentile), and the figures in Figure 5 and 6 are chosen from the images that are the closest to the percentiles.

Figure 5 : Visualization of the examples in Caltech-256 dataset that yield the P 25 (above) and P 50 (below) of the maximum 1 gradient distortions among the testing data on our ROLL model.

For the vanilla model, the maximum 1 gradient distortion ∆(x, x , y) is equal to 893.3 for 'Projector' in Figure 5g Figure 6 : Visualization of the examples in Caltech-256 dataset that yield the P 75 (above) and P 100 (below) of the maximum 1 gradient distortions among the testing data on our ROLL model.

For the vanilla model, the maximum 1 gradient distortion ∆(x, x , y) is equal to 1547.1 for 'Bear' in Figure 6g and 5473.5 for 'Rainbow' in Figure 6q .

For the ROLL model, the maximum 1 gradient distortion ∆(x, x , y) is equal to 1367.9 for 'Bear' in Figure 6e and 3882.8 for 'Rainbow' in Figure 6o .

<|TLDR|>

@highlight

A scalable algorithm to establish robust derivatives of deep networks w.r.t. the inputs.