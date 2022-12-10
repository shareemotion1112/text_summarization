This work tackles the problem of characterizing and understanding the decision boundaries of neural networks with piece-wise linear non-linearity activations.

We use tropical geometry, a new development in the area of algebraic geometry, to provide a characterization of the decision boundaries of a simple neural network of the form (Affine, ReLU, Affine).

Specifically, we show that the decision boundaries are a subset of a tropical hypersurface, which is intimately related to a polytope formed by the convex hull of two zonotopes.

The generators of the zonotopes are precise functions of the neural network parameters.

We utilize this geometric characterization to shed light and new perspective on three tasks.

In doing so, we propose a new tropical perspective for the lottery ticket hypothesis, where we see the effect of different initializations on the tropical geometric representation of the decision boundaries.

Also, we leverage this characterization as a new set of tropical regularizers, which  deal directly  with the decision boundaries of a network.

We investigate the use of these regularizers  in neural network pruning (removing network parameters that do not contribute to the tropical geometric representation of the decision boundaries) and in generating adversarial input attacks (with input perturbations explicitly perturbing the decision boundaries geometry to change the network prediction of the input).

.

In addition, and in attempt to understand some of the subtle behaviours DNNs exhibit, e.g. the sensitive reaction of DNNs to small input perturbations, several works directly investigated the decision boundaries induced by a DNN used for classification.

The work of SeyedMohsen Moosavi-Dezfooli (2019) showed that the smoothness of these decision boundaries and their curvature can play a vital role in network robustness.

Moreover, He et al. (2018a) studied the expressiveness of these decision boundaries at perturbed inputs and showed that these boundaries do not resemble the boundaries around benign inputs.

Li et al. (2018) showed that under certain assumptions, the decision boundaries of the last fully connected layer of DNNs will converge to a linear SVM.

Also, Beise et al. (2018) showed that the decision regions of DNNs with width smaller than the input dimension are unbounded.

More recently, and due to the popularity of the piecewise linear ReLU as an activation function, there has been a surge in the number of works that study this class of DNNs in particular.

x ⊕ y = max{x, y}, x y = x + y, ∀x, y ∈ T.

It can be readily shown that −∞ is the additive identity and 0 is the multiplicative identity.

Given the previous definition, a tropical power can be formulated as x a = x x · · · x = a.x, for x ∈ T, a ∈ N, where a.x is standard multiplication.

Moreover, the tropical quotient can be defined as: x y = x − y where x −

y is the standard subtraction.

For ease of notation, we write x a as x a .

Now, we are in a position to define tropical polynomials, their solution sets and tropical rationals.

f (x) = (c 1 x a1 ) ⊕ (c 2 x a2 ) ⊕ · · · ⊕ (c n x an ), ∀ a i = a j when i = j.

We use the more compact vector notation x a = x

In this section, we analyze the decision boundaries of a network in the form (Affine, ReLU, Affine) using tropical geometry.

For ease, we use ReLUs as the non-linear activation, but any other piecewise linear function can also be used.

The functional form of this network is: f (x) = Bmax (Ax + c 1 , 0) + c 2 , where max(.) is an element-wise operator.

The outputs of the network f are the logit scores.

Throughout this section, we assume 2 that A ∈ Z p×n , B ∈ Z 2×p , c 1 ∈ R p and c 2 ∈ R 2 .

For ease of notation, we only consider networks with two outputs, i.e. B 2×p , where the extension to a multi-class output follows naturally and it is discussed in the appendix.

Now, since f is a piecewise linear function, each output can be expressed as a tropical rational as per Theorem 1.

If f 1 and f 2 refer to the first and second outputs respectively, we have f 1 (x) = H 1 (x) Q 1 (x) and f 2 (x) = H 2 (x) Q 2 (x), where H 1 , H 2 , Q 1 and Q 2 are tropical polynomials.

In what follows and for ease of presentation, we present our main results where the network f has no biases, i.e. c 1 = 0 and c 2 = 0, and we leave the generalization to the appendix.

Theorem 2.

For a bias-free neural network in the form of f (x) : R n → R 2 where A ∈ Z p×n and B ∈ Z 2×p , let R(x) = H 1 (x) Q 2 (x) ⊕ H 2 (x) Q 1 (x) be a tropical polynomial.

Then:

• Let B = {x ∈ R n : f 1 (x) = f 2 (x)} defines the decision boundaries of f , then B ⊆ T (R(x)).

• δ (R(x)) = ConvHull (Z G1 , Z G2 ).

Z G1 is a zonotope in R n with line segments The proof for Theorem 2 is left for the appendix.

Digesting Theorem 2.

Theorem 2 can be broken into two major results.

The first, which is on the algebra side, i.e. finding the solution set to tropical polynomials, states that the decision boundaries linear pieces separating classes C1 and C2.

As per Theorem 2, the dual subdivision of this single hidden neural network is the convex hull between the zonotopes Z G 1 and Z G 2 .

The normals to the dual subdivison δ(R(x)) are in one-to-one correspondence to the tropical hypersurface T (R(x)), which is a superset to the decision boundaries B. Note that some of the normals to δ(R(x)) (in red) are parallel to the decision boundaries.

B is a subset of the tropical hypersurface of the tropical polynomial R(x), i.e. T (R(x)).

The second result, which is on the geometry side, of Theorem 2 relates the tropical polynomial R(x) to the geometric representation of the solution set to R(x), i.e. T (R(x)), referred to as the dual subdivision, i.e. δ(R(x)).

In particular, Theorem 2 states that the dual subdivision for a network f is the convex hull of two zonotopes denoted as Z G1 and Z G2 .

Note that this dual subdivision is a function of only the network parameters A and B.

Theorem 2 bridges the gap between the behaviour of the decision boundaries B, through the super-set T (R(x)), and the polytope δ (R(x)), which is the convex hull of two zonotopes.

It is worthwhile to mention that Zhang et al. (2018) discussed a special case of the first part of Theorem 2 for a neural network with a single output and a score function s(x) to classify the output.

To the best of our knowledge, this work is the first to propose a tropical geometric formulation of a super-set containing the decision boundaries of a multi-class classification neural network.

In particular, the first result of Theorem 2 states that one can alter the network, e.g. by pruning network parameters, while preserving the decision boundaries B, if one preserves the tropical hypersurface of R(x) or T (R(x)).

While preserving the tropical hypersurfaces can be equally difficult to preserving the decision boundaries directly, the second result of Theorem 2 comes in handy.

For a bias free network, π becomes an identity mapping with δ(R(x)) = ∆(R(x)), and thus the dual subdivision δ(R(x)), which is the Newton polytope ∆(R(x)) in this case, becomes a well structured geometric object that can be exploited to preserve decision boundaries.

Since Maclagan & Sturmfels (2015) (Proposition 3.1.6) showed that the tropical hypersurface is the skeleton of the dual to δ(R(x)), the normal lines to the edges of the polytope δ(R(x)) are in one-to-one correspondence with the tropical hypersurface T (R(x)).

Figure 1 details this intimate relation between the decision boundaries, tropical hypersurface T (R(x)), and normals to δ (R(x)).

Before any further discussion, we recap the definition of zonotopes.

Equivalently, the zonotope can be expressed with respect to the generator matrix U ∈ R p×n , where

Another common definition for zonotopes is the Minkowski sum (refer to appendix A for the definition of the Minkowski sum) of a set of line segments that start from the origin with end points u 1 , . . . , u p ∈ R n .

It is also well known that the number of vertices of a zonotope is polynomial in the number of line segments.

That is to say,

While Theorem 2 presents a strong relation between a polytope (convex hull of two zonotopes) and the decision boundaries, it remains unclear how such a polytope can be efficiently constructed.

Although the number of vertices of a zonotope is polynomial in the number of its generating line segments, fast algorithms for enumerating these vertices are still restricted to zonotopes with line segments starting at the origin (Stinson et al., 2016) .

Since the line segments generating the zonotopes in Theorem 2 have arbitrary end points, we present the next result that transforms these line segments into a generator matrix of line segments starting from the origin, as prescribed in Definition 6.

This result is essential for the efficient computation of the zonotopes in Theorem 2.

Proposition 1.

Consider p line segments in R n with two arbitrary end points as follows

.

The zonotope formed by these line segments is equivalent to the zonotope formed by the line segments {[u

training dataset, decision boundaries polytope of original network followed by the decision boundaries polytope during several iterations of pruning with different initializations.

The proof is left for the appendix.

As per Proposition 1, the generator matrices of zonotopes In what follows, we show several applications for Theorem 2.

We begin by leveraging the geometric structure to help in reaffirming the behaviour of the lottery ticket hypothesis.

The lottery ticket hypothesis was recently proposed by Frankle & Carbin (2019) , in which the authors surmise the existence of sparse trainable sub-networks of dense, randomly-initialized, feedforward networks that-when trained in isolation-perform as well as the original network in a similar number of iterations.

To find such sub-networks, Frankle & Carbin (2019) propose the following simple algorithm: perform standard network pruning, initialize the pruned network with the same initialization that was used in the original training setting, and train with the same number of epochs.

They hypothesize that this should result in a smaller network with a similar accuracy to the larger dense network.

In other words, a subnetwork can have similar decision boundaries to the original network.

While in this section we do not provide a theoretical reason for why this proposed pruning algorithm performs favorably, we utilize the geometric structure that arises from Theorem 2 to reaffirm such behaviour.

In particular, we show that the orientation of the decision boundaries polytope δ(R(x)), known to be a superset to the decision boundaries T (R(x)), is preserved after pruning with the proposed initialization algorithm of Frankle & Carbin (2019).

On the other hand, pruning routines with a different initialization at each pruning iteration will result in a severe variation in the orientation of the decision boundaries polytope.

This leads to a large change in the orientation of the decision boundaries, which tends to hinder accuracy.

To this end, we train a neural network with 2 inputs (n = 2), 2 outputs, and a single hidden layer with 40 nodes (p = 40).

We then prune the network by removing the smallest x% of the weights.

The pruned network is then trained using different initializations: (i) the same initialization as the original network (Frankle & Carbin, 2019), (ii) Xavier (Glorot & Bengio, 2010), (iii) standard Gaussian and (iv) zero mean Gaussian with variance of 0.1.

Figure 2 shows the evolution of the decision boundaries polytope, i.e. δ(R(x)), as we perform more pruning (increasing the x%) with different initializations.

It is to be observed that the orientation of the polytopes δ(R(x)) vary much more for all different initialization schemes as compared to the lottery ticket initialization.

This gives an indication that lottery ticket initialization indeed preserves the decision boundaries throughout the evolution of pruning.

Another approach to investigate the lottery ticket could be by observing the polytopes representing the functional form of the network directly, i.e. δ(H {1,2} (x)) and δ(Q {1,2} (x)), in lieu of the decision boundaries polytopes.

However, this does not provide conclusive answers to the lottery ticket, since there can exist multiple functional forms, and correspondingly multiple polytopes δ(H {1,2} (x)) and δ(Q {1,2} (x)), for networks with the same decision boundaries.

This is why we explicitly focus our analysis on δ(R(x)), which is directly related to the decision boundaries of the network.

Further discussions and experiments are left for the appendix.

Network pruning has been identified as an effective approach for reducing the computational cost and memory usage during network inference time.

While pruning dates back to the work of LeCun et al. (1990) and Hassibi & Stork (1993) , it has recently gained more attention.

This is due to the fact that most neural networks over-parameterize commonly used datasets.

In network pruning, the task is to find a smaller subset of the network parameters, such that the resulting smaller network has similar decision boundaries (and thus supposedly similar accuracy) to the original over-parameterized network.

In this section, we show a new geometric approach towards network pruning.

In particu- th node, or equivalently removing the two yellow vertices of zonotope ZG 2 does not affect the decision boundaries polytope which will not lead to any change in accuracy.

lar, as indicated by Theorem 2, preserving the polytope δ(R(x)) preserves a superset to the decision boundaries T (R(x)), and thus supposedly the decision boundaries themselves.

Motivational Insight.

For a single hidden layer neural network, the dual subdivision to the decision boundaries is the polytope that is the convex hull of two zonotopes, where each is formed by taking the Minkowski sum of line segments (Theorem 2).

Figure 3 shows an example where pruning a neuron in the neural network has no effect on the dual subdivision polytope and equivalently no effect on the accuracy, since the decision boundaries of both networks remain the same.

Problem Formulation.

Given the motivational insight, a natural question arises: Given an overparameterized binary neural network f (x) = B max (Ax, 0), can one construct a new neural network, parameterized by some sparser weight matricesÃ andB, such that this smaller network has a dual subdivision δ(R(x)) that preserves the decision boundaries of the original network?

In order to address this question, we propose the following general optimization problem

The function d(.) defines a distance between two geometric objects.

Since the generatorsG 1 and G 2 are functions ofÃ andB (as per Theorem 2), this optimization problem can be challenging to solve.

However, for pruning purposes, one can observe from Theorem 2 that if the generatorsG 1 andG 2 had fewer number of line segments (rows), this corresponds to a fewer number of rows in the weight matrixÃ (sparser weights).

To this end, we observe that ifG 1 ≈ G 1 andG 2 ≈ G 2 , thenδ(R(x)) ≈ δ(R(x)), and thus the decision boundaries tend to be preserved as a consequence.

Therefore, we propose the following optimization problem as a surrogate to Problem (1)

The matrix mixed norm for C ∈ R n×k is defined as C 2,1 = n i=1 C(i, :) 2 , which encourages the matrix C to be row sparse, i.e. complete rows of C are zero.

Note thatG 1 = Diag[ReLU(B(1, : ))+ReLU(−B(2, :))]

Ã,G 2 = Diag[ReLU(B(2, :))+ReLU(−B(1, :))]Ã, and Diag(v) rearranges the elements of vector v in a diagonal matrix.

We solve the aforementioned problem with alternating optimization over the variablesÃ andB, where each sub-problem is solved in closed form.

Details of the optimization and the extension to multi-class case are left for the appendix.

Extension to Deeper Networks.

For deeper networks, one can still apply the aforementioned optimization for consecutive blocks.

In particular, we prune each consecutive block of the form (Affine,ReLU,Affine) starting from the input and ending at the output of the network.

Experiments on Tropical Pruning.

Here, we evaluate the performance of the proposed pruning approach as compared to several classical approaches on several architectures and datasets.

In particular, we compare our tropical pruning approach against Class Blind (CB), Class Uniform (CU) and Class Distribution (CD) Han et al. (2015) ; See et al. (2016) .

In Class Blind, all the parameters across all nodes of a layer are sorted by magnitude where x% with smallest magnitudes are pruned.

Similar to Class Blind, Class Uniform prunes the parameters with smallest x% magnitudes per node in a layer as opposed to sorting all parameters in all nodes as in Class Blind.

Lastly, Class Distribution performs pruning of all parameters for each node in the layer, just as in Class Uniform, but the parameters are pruned based on the standard deviation σ c of the magnitude of the parameters per node.

Since fully connected layers in deep neural networks tend to have much higher memory complexity than convolutional layers, we restrict our focus to pruning fully connected layers.

We train AlexNet and VGG16 on SVHN , CIFAR10, and CIFAR 100 datasets.

We observe that we can prune more than 90% of the classifier parameters for both networks without affecting the accuracy.

Moreover, we can boost the pruning ratio using our method without affecting the accuracy by simply retraining the network biases only.

Setup.

We adapt the architectures of AlexNet and VGG16, since they were originally trained on ImageNet (Deng et al., 2009), to account for the discrepancy in the input resolution.

The fully connected layers of AlexNet and VGG16 have sizes of (256,512,10) and (512,512,10), respectively on SVHN and CIFAR100 with the last layer replaced to 100 for CIFAR100.

All networks were trained to baseline test accuracy of (92%,74%,43%) for AlexNet on SVHN, CIFAR10 and CIFAR100, respectively and (92%,92%,70%) for VGG16.

To evaluate the performance of pruning, following previous works (Han et al., 2015) , we report the area under the curve (AUC) of the pruning-accuracy plot.

The higher the AUC is, the better the trade-off is between pruning rate and accuracy.

For efficiency purposes, we run the optimization in Problem (2) for a single alternating iteration to identify the rows inÃ and elements ofB that will be pruned, since an exact pruning solution might not be necessary.

The algorithm and the parameters setup to solving (2) is left for the appendix.

Results.

Figure 4 shows the pruning comparison between our tropical approach and the three aforementioned popular pruning schemes on both AlexNet and VGG16 over the different datasets.

Our proposed approach can indeed prune out as much as 90% of the parameters of the classifier without sacrificing much of the accuracy.

For AlexNet, we achieve much better performance in pruning as compared to other methods.

In particular, we are better in AUC by 3%, 3%, and 2% over other pruning methods on SVHN, CIFAR10 and CIFAR100, respectively.

This indicates that the decision boundaries can indeed be preserved by preserving the dual subdivision polytope.

For VGG16, we perform similarly well on both SVHN and CIFAR10 and slightly worse on CIFAR100.

While the performance achieved here is comparable to the other pruning schemes, if not better, we emphasize that our contribution does not lie in outperforming state-of-the-art pruning methods, but rather in giving a new geometry based perspective to network pruning.

We conduct more experiments, where only the biases of the network or the biases of the classifier are fine tuned after pruning .

Retraining biases can be sufficient as they do not contribute to the orientation of the decision boundaries polytope, thereafter the decision boundaries, but only a translation.

Discussion on biases and more results are left for the appendix.

DNNs are notoriously known to be susceptible to adversarial attacks.

In fact, adding small imperceptible noise, referred to as adversarial attacks, at the input of these networks can hinder their performance.

Several works investigated the decision boundaries of neural networks in the presence of adversarial attacks.

For instance, Khoury & Hadfield-Menell (2018) analyzed high dimensional geometry of adversarial examples by the means of manifold reconstruction.

Also, He et al. (2018b) crafted adversarial attacks by estimating the distance to the decision boundaries using random search directions.

In this work, we provide a tropical geometric view to this problem.

where we show how Theorem 2 can be leveraged to construct a tropical geometric based targeted adversarial attack.

Dual View to Adversarial Attacks.

For a classifier f : R n → R k and input x 0 that is classified as c, a standard formulation for targeted adversarial attacks flips the classifier prediction to a particular class t and it is usually defined as follows

This objective aims at computing the lowest energy input noise η (measured by D) such that the the new sample (x 0 + η) crosses the decision boundaries of f to a new classification region.

Here, we present a dual view to adversarial attacks.

Instead of designing a sample noise η such that (x 0 + η) belongs to a new decision region, one can instead fix x 0 and perturb the network parameters to move the decision boundaries in a way that x 0 appears in a new classification region.

In particular, let A 1 be the first linear layer of f , such that f (x 0 ) = g(A 1 x 0 ).

One can now perturb A 1 to alter the decision boundaries and relate the perturbation to the input perturbation as follows

From this dual view, we observe that traditional adversarial attacks are intimately related to perturbing the parameters of the first linear layer through the linear system:

To this end, Theorem 2 provides explicit means to geometrically construct adversarial attacks by means of perturbing decision boundaries.

In particular, since the normals to the dual subdivision polytope δ(R(x)) of a given neural network represent the tropical hypersurface set T (R(x)) which is, as per Theorem 2, a superset to the decision boundaries set B, ξ A1 can be designed to result in a minimal perturbation to the dual subdivision that is sufficient to change the network prediction of x 0 to the targeted class t. Based on this observation, we formulate the problem as follows

The loss is the standard cross-entropy loss.

The first row of constraints ensures that the network prediction is the desired target class t when the input x 0 is perturbed by η, and equivalently by perturbing the first linear layer A 1 by ξ A1 .

This is identical to f 1 as proposed by Carlini & Wagner (2016).

Moreover, the third and fourth constraints guarantee that the perturbed input is feasible and that the perturbation is bounded, respectively.

The fifth constraint is to limit the maximum perturbation on the first linear layer, while the last constraint enforces the dual equivalence between input perturbation and parameter perturbation.

The function D 2 captures the perturbation of the dual subdivision polytope upon perturbing the first linear layer by ξ A1 .

For a single hidden layer neural network parameterized as (A 1 + ξ A1 ) ∈ R p×n and B ∈ R 2×p for the 1 st and 2 nd layers respectively, D 2 can capture the perturbations in each of the two zonotopes discussed in Theorem 2.

The derivation, discussion, and extension of (6) to multi-class neural networks is left for the appendix.

We solve Problem (5) with a penalty method on the linear equality constraints, Motivational Insight to the Dual View.

This intuition is presented in Figure 5 .

We train a single hidden layer neural network where the size of the input is 2 with 50 hidden nodes and 2 outputs on a simple dataset as shown in Figure 5 .

We then solve Problem 5 for a given x 0 shown in black.

We show the decision boundaries for the network with and without the perturbation at the first linear layer ξ A1 .

Figure 5 shows that indeed perturbing an edge of the dual subdivision polytope, by perturbing the first linear layer, corresponds to perturbing the decision boundaries and results in miss-classifying x 0 .

Interestingly and as expected, perturbing different decision boundaries corresponds to perturbing different edges of the dual subdivision.

In particular, one can see from Figure 5 that altering the decision boundaries, by altering the dual subdivision polytope through perturbations in the first linear layer, can result in miss-classifying a previously correctly classified input x 0 .

MNIST Experiment.

Here, we design perturbations to misclassify MNIST images.

Figure 7 shows several adversarial examples that change the network prediction for digits 8 and 9 to digits 7, 5, and 4, respectively.

In some cases, the perturbation η is as small as = 0.1, where x 0 ∈ [0, 1] n .

Several other adversarial results are left for the appendix.

We again emphasize that our approach is not meant to be compared with (or beat) state of the art adversarial attacks, but rather to provide a novel geometrically inspired perspective that can shed new light in this field.

In this paper, we leverage tropical geometry to characterize the decision boundaries of neural networks in the form (Affine, ReLU, Affine) and relate it to well-studied geometric objects such as zonotopes and polytopes.

We leaverage this representation in providing a tropical perspective to support the lottery ticket hypothesis, network pruning and designing adversarial attacks.

One natural extension for this work is a compact derivation for the characterization of the decision boundaries of convolutional neural networks (CNNs) and graphical convolutional networks (GCNs).

Diego Ardila, Atilla P. Kiraly, Sujeeth Bharadwaj, Bokyung Choi, Joshua J. Reicher, Lily Peng, Daniel Tse, Mozziyar Etemadi, Wenxing Ye, Greg Corrado, David P. Naidich, and Shravya Shetty.

End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography.

Nature Medicine, 2019.

A PRELIMINARIES AND DEFINITIONS.

Fact 1.

P+Q = {p + q, ∀p ∈ P and q ∈ Q} is the Minkowski sum between two sets P and Q. Fact 2.

Let f be a tropical polynomial and let a ∈ N.

Then

Let both f and g be tropical polynomials, Then

Note that V(P(f )) is the set of vertices of the polytope P(f ).

Theorem 3.

For a bias-free neural network in the form of f (x) : R n → R 2 where A ∈ Z p×n and B ∈ Z 2×p , and let

• If the decision boundaries of f is given by the set B = {x ∈ R n : f 1 (x) = f 2 (x)}, then we have B ⊆ T (R(x)).

Note that A + = max(A, 0) and A − = max(−A, 0) where the max(.) is element-wise.

The line segment (B(1, j)

is one that has the end points A(j, :) + and A(j, :) − in R n and scaled by the constant B(1, j)

Proof.

For the first part, recall from Theorem1 that both f 1 and f 2 are tropical rationals and hence,

Recall that the tropical hypersurface is defined as the set of x where the maximum is attained by two or more monomials.

Therefore, the tropical hypersurface of R(x) is the set of x where the maximum is attained by two or more monomials in (H 1 (x) Q 2 (x)), or attained by two or more monomials in (H 2 (x) Q 1 (x)), or attained by monomials in both of them in the same time, which is the decision boundaries.

Hence, we can rewrite that as

Therefore, we have that

Therefore note that

, thus we have that

The operator+ indicates a Minkowski sum between sets.

Note that ConvexHull A

) is the convexhull between two points which is a line segment in Z n with end points that are

is a Minkowski sum of line segments which is is a zonotope.

Moreover, note that

tropically is given as follows

.

Thus it is easy to see that δ(Q 2 (x)) is the Minkowski sum of the points {(B − (1, j)−B + (2, j))A − (j, :)}∀j in R n (which is a standard sum) resulting in a point.

Lastly, it is easy to see that δ(H 1 (x))+δ(Q 2 (x)) is a Minkowski sum between a zonotope and a single point which corresponds to a shifted zonotope.

A similar symmetric argument can be applied for the second part δ(H 2 (x))+δ(Q 1 (x)).

It is also worthy to mention that the extension to network with multi class output is trivial.

In that case all of the analysis can be exactly applied studying the decision boundary between any two classes (i, j) where B = {x ∈ R n : f i (x) = f j (x)} and the rest of the proof will be exactly the same.

In this section, we derive the statement of Theorem 2 for the neural network in the form of (Affine, ReLU, Affine) with the consideration of non-zero biases.

We show that the presence of biases does not affect the obtained results as they only increase the dimension of the space, where the polytopes live, without affecting their shape or edge-orientation.

Starting with the first linear layer for x ∈ R n , we have

with coordinates

, and ∆(Q 1i ) is a point in (n + 1) dimensions at (A − (i, :), 0), while under π projection, δ(H 1i ) is a point in n dimensions at (A + (i, : )), and δ(Q 1i ) is a point in n dimensions at (A − (i, :)) .

It can be seen that under projection π, the geometrical representation of the output of the first linear layer does not change after adding biases.

Looking to the output after adding the ReLU layer, we get

, and δ(Q 1i ) is the point (A − (i, :)).

Again, the biases does not affect the geometry of the output after the ReLU layer, since the line segments now are connecting points in (n + 1) dimensions, but after projecting them using π, they will be identical to the line segments of the network with zero biases.

Finally, looking to the output of the second linear layer, we obtain

Similar arguments can be given for ∆(Q 3i ) and δ(Q 3i ).

It can be seen that the first part in both expressions is a Minkowski sum of line segments, which will give a zonotope in (n + 1), and n dimensions in the first and second expressions respectively.

While the second part in both expressions is a Minkowski sum of bunch of points which gives a single point in (n + 1) and n dimensions for the first and second expression respectively.

Note that the last dimension of the aforementioned point in n + 1 dimensions is exactly the i th coordinate of the bias of the second linear layer which is dropped under the π projection.

Therefore, the shape of the geometrical representation of the decision boundaries with non-zero biases will not be affected under the projection π, and hence the presence of the biases will not affect any of the results of the paper.

Proposition 1.

Consider p line segments in R n with two arbitrary end points as follows

.

The zonotope formed by these line segments is equivalent to the zonotope formed be the line segments

Proof.

Let U j be a matrix with U j (:, i) = u i j , i = 1, . . .

, p, w be a column-vector with w(i) = w i , i = 1, . . .

, p and 1 p is a column-vector of ones of length p.

Then, the zonotope Z formed by the Minkowski sum of line segments with arbitrary end points can be defined as

Note that the Minkowski sum of any polytope with a point is a translation; thus, the result follows directly from Definition 6.

A ← arg miñ

, where c 1 = ReLU(B(1, :)) + ReLU(−B(2, :)) and c 2 = ReLU(B(2, :)) + ReLU(−B(1, :)).

Note that the problem is separable per-row ofÃ.

Therefore, the problem reduces to updating rows ofÃ independently and the problem exhibits a closed form solution.

.

UpdateB + (1, :).

Note that C 1 = G 1 − Diag B − (2, :) Ã and where Diag B − (2, :) Ã .

Note the problem is separable in the coordinates ofB + (1, :) and a projected gradient descent can be used to solve the problem in such a way as

A similar symmetric argument can be used to update the variablesB + (2, :),B + (1, :) andB − (2, :).

Note that Theorem 2 describes a superset to the decision boundaries of a binary classifier through the dual subdivision R(x), i.e. δ(R(x)).

For a neural network f with k classes, a natural extension for it is to analyze the pair-wise decision boundaries of of all k-classes.

Thus, let T (R ij (x)) be the superset to the decision boundaries separating classes i and j.

Therefore, a natural extension to the geometric loss in equation 1 is to preserve the polytopes among all pairwise follows

(10) The set S is all possible pairwise combinations of the k classes such that S = {[i, j], ∀i = j, i = 1, . . .

, k, j = 1, . . .

, k}. The generator Z (G (i,j) ) is the zonotope with the generator matrix G (i + ,j − ) = Diag ReLU(B(i, :)) + ReLU(−B(j, :)) Ã .

However, such an approach is generally computationally expensive, particularly, when k is very large.

To this end, we make the following observation thatG ( i + , j − ) can be equivalently written as a Minkowski sum between two sets zonotopes with the generators

That is to say, ZG

.

This follows from the associative property of Minkowski sums given as follows:

be the set of n line segments.

Then we have that S = S 1+ . .

.+S n = P+V where the sets P =+ j∈C1 S j and V =+ j∈C2 S j where C 1 and C 2 are any complementary partitions of the set

Hence,G (i + ,j − ) can be seen a concatenation betweenG ( i + ) andG ( j − ).

Thus, the objective in 10 can be expanded as follows

The approximation follows in a similar argument to the binary classifier case where approximating the generators.

The last equality follows from a counting argument.

We solve the objective for all multi-class networks in the experiments with alternating optimization in a similar fashion to the binary classifier case.

Similarly to the binary classification approach, we introduce the 2,1 to enforce sparsity constraints for pruning purposes.

Therefore the overall objective has the form

For completion, we derive the updates forÃ andB.

A = arg miñ

Similar to the binary classification, the problem is seprable in the rows ofÃ. and a closed form solution in terms of the proximal operator of 2 norm follows naturally for eachÃ(i, :).

UpdateB + (i, :).

Note that the problem is separable per coordinates of B + (i, :) and each subproblem is updated as:

A similar argument can be used to updateB − (i, :) ∀i.

Finally, the parameters of the pruned network will be constructed A ←Ã and B ←B + −B − .

Input :

In this section, we are going to derive an algorithm for solving the following problem.

The function D 2 (ξ A ) captures the perturbdation in the dual subdivision polytope such that the dual subdivion of the network with the first linear layer A 1 is similar to the dual subdivion of the network with the first linear layer A 1 + ξ A1 .

This can be generally formulated as an approximation to the following distance function

This can thereafter be extended to multi-class network with k classes as follows

.

Following Xu et al. (2018), we take

.

Therefore, we can write 11 as follows

To enforce the linear equality constraints A 1 η − ξ A1 x 0 = 0, we use a penalty method, where each iteration of the penalty method we solve the sub-problem with ADMM updates.

That is, we solve the following optimization problem with ADMM with increasing λ such that λ → ∞. For ease of notation, lets denote

where

The augmented Lagrangian is thus given as follows

Thereafter, ADMM updates are given as follows

Updating η:

Updating w:

It is easy to show that the update w is separable in coordinates as follows

Updating z:

Liu et al. (2019) showed that the linearized ADMM converges for some non-convex problems.

Therefore, by linearizing L and adding Bergman divergence term η

, we can then update z as follows

It is worthy to mention that the analysis until this step is inspired by Xu et al. (2018) with modifications to adapt our new formulation.

Updating ξ A :

The previous problem can be solved with proximal gradient method.

In this section, we are going to describe the settings and the values of the hyper-parameters that we used in the experiments.

Moreover, we will show more results since we have limited space in the main paper.

We begin by throwing the following question.

Why investigating the tropical geometrical perspective of the decision boundaries is more important than investigating the tropical geometrical representation of the functional form of the network ?

In this section, we show one more experiment that differentiate between these two views.

In the following, we can see that variations can happen to the tropical geometrical representation of the functional form (zonotopes in case of single hidden layer neural network), but the shape of the polytope of the decision boundaries is still unchanged and consequently, the decision boundaries.

For this purpose, we trained a single hidden layer neural network on a simple dataset like the one in Figure 2 , then we do several iteration of pruning, and visualise at each iteration both the polytope of the decision boundaries and the zonotopes of the functional representation of the neural network.

It can be easily seen that changes in the zonotopes may not change the shape of the decision boundaries polytope and consequently the decision boundaries of the neural network.

And thus it can be clearly seen that our formulation, which is looking at the decision boundaries polytope is more general, precise and indeed more meaningful.

Moreover, we conducted the same experiment explained in the main paper of this section on another dataset to have further demonstration on the favour that the lottery ticket initialization has over other initialization when pruning and retraining the pruned model.

It is clear that the lottery initializations is the one that preserves the shape of the decision boundaries polytope the most.

In the tropical pruning, we have control on two hyper-parameters only, namely the number of iterations and the regularizer coefficient λ which controls the pruning rate.

In all of the experiments, we ran the algorithm for 1 iteration only and we increase λ starting from 0.02 linearly with a factor of 0.01 to reach 100% pruning.

It is also worthy to mention that the output of the algorithm will be new sparse matricesÃ,B, but the new network parameters will be the elements in the original matrices A, B that have indices correspond to the indices of non-zero elements inÃ,B. By that, the algorithm removes the non-effective line segments that do not contribute to the decision boundaries polytope, without changing the non-deleted segments.

Above all, more results of pruning of AlexNet and VGG16 on various datasets are shown below.

For all of the experiments, { 2 , λ, η, ρ} had the values {1, 10 −3 , 2.5, 1} respectively.

the value of 1 was 0.1 when attacking the -fours-images, and 0.2 for the rest of the images.

Finally, we show extra results of attacking the decision boundaries of synthetic data in R 2 and MNIST images by tropical adversarial attacks.

We thank R3 for the time spent reviewing the paper.

It is though not clear to the authors the main reason behind the initial score of weak reject as R3 seems to have very generic questions about our work but not a particular criticism of the novelty/contribution that we can address in our rebuttal.

We hope that the following response addresses and clarifies some key elements.

Moreover, we want to bring to the attention of R3 that we have addressed the comments/typos/suggestions of all reviewers in the revised version and marked them in blue.

Q1: What benefit does introducing tropical geometry brings in terms of theoretical analysis?

Does using tropical geometry give us the theoretical results that traditional analysis can not give us?

If so, what is it?

I am trying to understand why the authors use this tool.

The authors should be explicit in their motivation so that the readers are clear about the contribution of this paper.

More specifically, from my perspective, tropical semiring, tropical polynomials and tropical rational functions all can be represented with the standard mathematical tools.

Here they are just redefining several concepts.

As discussed thoroughly in the introduction (last paragraph of page 1), tropical geometry is the younger twin to algebraic geometry on a particular semiring defined in a way to align with the study of piecewise linear functions.

The early definitions stated in the paper (1 to 5) are well known in the TG literature and were restated for the completion of this paper.

While it is true that the definitions can be represented with standard mathematical tools; however, this misses the fundamental powerful element TG promises.

TG transforms algebraic problems of piecewise linear nature to a combinatoric problem on general polytopes.

To that end, Zhang et.

al. 2018 (to the best of our knowledge the only work at the intersection between TG and DNNs) rederived classical results (upper bound on the number of linear pieces of DNNs) in a much simpler analysis by counting vertices on polytopes.

In this work, instead of studying the functional representation of piecewise linear DNNs, we study their decision boundaries using the lens of TG.

To wit, the geometric characterization of the decision boundaries of DNNs developed in Theorem 2 cannot be attained using standard mathematical tools.

More specifically, Theorem 2 represented a superset to the decision boundaries (the tropical hypersurface T (R(x)), with a geometric structure that is the convex hull between two zonotopes.

While this by itself opens doors for a family of new geometrically motivated regualrizers for training DNNs that are in direct correspondence with the behaviour of the decision boundaries, we do not dwell on training beyond this point and leave that for future work.

However, this new result allowed for re-affirmation to the lottery ticket hypothesis in a new fresh perspective.

Moreover, we propose new optimization problems that are geometrically motivated (based on Theorem 2) for several classical problems, i.e. network pruning and adversarial attacks that were not possible before and have provided several new insights and directions.

That is we show an intimate relation between network perturbations (through decision boundary polytope perturbations) and the construction of adversarial attacks (input perturbations).

In Experiments on Tropical Pruning, the authors mentioned we compare our tropical pruning approach against Class Blind (CB), Class Uniform (CU), and Class Distribution (CD) methods Han et al. (2015) .

What is Class Blind, Class Uniform and Class Distribution?

There seems to be an error here Figure 5 shows the pruning comparison between our tropical approach ..., i think Figure 5 should be Figure 4 .

We have added the definition of the pruning methods of Han et al. (2015) in the revised version of the paper for completion, and corrected the typo in the Figure reference.

Q3: In the adversarial attack part, is the authors proposing a new attack method?

If so, then the authors should report the test accuracy under attack.

Also, the experimental results should not be restricted to MNIST dataset.

I am also not sure about the attack settings here, the authors said Instead of designing a sample noise such that (x0 + η) belongs to a new decision region, one can instead fix x0 and perturb the network parameters to move the decision boundaries in a way that x0 appears in a new classification region..

Why use this setting?

Are there any intuitions?

Since this is different from traditional adversarial attack terminology, the authors should stop using adversarial attacks as in tropical adversarial attacks because it is really misleading.

As highlighted in the last sentence in section 6, we are not competing against other attacks, but we rather show how this new geometric view to the decision boundaries provided by the TG analysis in Theorem 2 can be leveraged for the construction of adversarial attacks.

We want to emphasize to R3 that the polytope representing the decision boundaries (convex hull of two zonotopes as per Theorem 2) is a function of the network parameters and not of the input space.

Thus, it is not initially clear how one can frame the adversarial attacks problem in this new fresh tropical setting since adversarial attacks is the task of perturbing the input space as opposed to the parameters space of the network resulting in a flip in the prediction.

In the tropical adversarial attacks section, we show that the problem of designing an adversarial attack x 0 +η that flips the network prediction is closely related to the problem of flipping the network prediction by perturbing the network parameters in the first layer A 1 + ζ A1 where both problems are related through a linear system.

That is to say, if one finds ζ A1 that perturbs the geometric structure (convex hull between two zonotopes, i.e. decision boundaries) sufficiently enough to flip the network prediction, one can find an equivalent pixel adversarial attack η by solving the linear system A 1 η = ζ A1 x 0 that flips the prediction of the original unperturbed network (see the end of page 7).

We thereafter propose Problem (5) incorporating the geometric information from Theorem 2 where the linear system is accounted for in the constraints set.

We propose an algorithm to solve the problem (a mix of penalty and ADMM) detailed in Algorithm 1 in the appendix.

The solution to problem 5 by applying Algorithm 1 results in the construction of adversarial attacks (η) that indeed flip the network prediction over all tested examples on the MNIST dataset.

We thank R2 for the time spent reviewing the paper.

We also thank R2 for acknowledging our technical and theoretical contributions.

Please note that we have addressed the comments/typos/suggestions of all reviewers in the revised version and marked them in blue.

Follows our response.

Q1: This paper needs to be placed properly among several important missing references on the decision boundary of deep neural networks [1] [2] .

In particular, using introduced tropical geometry perspective, how we can obtain the complexity of the decision boundary of a deep neural network?

The two works referenced by R2 are not directly related to the body of our work.

Below, we summarize both works and state how our work is vastly different from both.

The authors of [1] show that under certain assumptions, the decision boundaries of the last fully connected layer converges to an SVM classifier.

That is to say, the features learnt in deep neural networks are linearly separable with max margin type linear classifier.

On the other hand, the authors of [2] showed that the decision regions of neural networks with width smaller than the input dimension are unbounded.

In our work, we use a new type of analysis (tropical geometry) to represent the set of decision boundaries B through its superset T (R(x)) that is the solution set to the tropical polynomial R(x).

We then show that this solution set is related to a geometric structure referred to as the decision boundaries polytope (convex hull between two zonotopes), this is analogous to constructing newton polytopes for the solution sets to classical polynomials in algebraic geoemtry.

The normals to the edges of this polytope are parallel to the superset of the decision boundaries T (R(x)).

That is to say, if one processes the polytope in an way that preserves the direction of the normals, the decision boundaries of the network are preserved.

This is the base idea behind all later experiments.

In general, this new representation presents a new fresh revisit to the lottery ticket hypothesis and an utterly new view to network pruning and adversarial attacks.

We do believe this new representation can be of benefit to other applications and can open doors for a family of new geometrically inspired network regularizers as well.

Q3: The second part of Theorem 2 should be explained straightforwardly and clearly as it plays an important role in the subsequent results and applications.

We have added a "Digesting Theorem 2" paragraph in the revised version and rearranged the structure a bit around Theorem 2.

Most of the parameters (memory complexity) are the in fully connected layers.

For example, the convolutional part of VGG16 has 14,714,688 parameters, whereas the fully connected layers have 262,000,400 parameters in total which is 17 times larger.

Similarly, the convolutional part of AlexNet has 3,747,200 parameters while only the first fully connected layer has 37,752,832 parameters [5] .

However, efficiently extending the tropical pruning to convolutional layers is a nontrivial interesting direction.

Generally speaking, convolutional layers fit our framework naturally since a convolutional kernel can be represented with a structured topelitz/circulant matrix.

However, a question of efficiency still remains as one still needs to construct the underlying structured matrix representing the convolutional kernel.

Thereafter, a direction of interest is the tropical formulation of the network pruning problem as a function of the convolutional kernels surpassing the need for the construction of the dense representation of the kernel.

We keep this for future work.

Q5:

For the similarity measure.

Comparing the exact decision boundaries between two different architectures can be very difficult in the sense where decision boundaries for a two-class output network f are defined as {x ∈ R n : f 1 (x) = f 2 (x)}. Another approach to compare decision boundaries, which is proposed by our work, is by computing the distance between the the dual subdivision polytope (δ(R(x))) of the tropical polynomials R(x) representing two different architectures.

This is since the normals to the edges of the polytope (δ(R(x))) are parallel to a superset of the decision boundaries (see Figure 1) .

This is exactly the proposed objective (1) where d(.) is a distance function to compare the orientation between two general polytopes.

Since finding a good choice for d(.) is generally difficult we instead approximate it by comparing the generators constructing the (δ(R(x))) in Euclidean distance for ease (objective (2)).

Experimentally and following prior art in the pruning literature (Han et.

al. 2015) , to compare the effectiveness of the pruning scheme, we compare the test accuracies across architectures as a function of the pruning ratio.

Regardless, the reviewer is right about that similar test accuracies does not imply similar decision boundaries but rather only an indication.

In adversarial examples generation, typically for a pre-trained deep neural network model one is interested in generating examples that are misclassified by the model while they resemble real instances.

In this setting, we keep the model and thus its decision boundary intact.

In this paper, nevertheless, aiming at generating adversarial examples, the decision boundary and thus the (pre-trained) model is altered.

By chaining the decision boundary, however, the model's decisions for original real samples might change as well.

Therefore, it is not clear to the reviewer how the introduced method is comparable to the well-established adversarial example generation setting.

The new approach is definitely comparable to the well-establisehd adversarial example generation setting.

Let us explain.

The new analysis provided by Theorem 2, allows to present the decision boundaries geometrically as a convex hull between two zonotopes which is a function of only the network parameters and not the input space.

Thus, it not clear how one can frame the adversarial attacks problem in this new fresh tropical setting since adversarial attacks is the task of perturbing the input space as opposed to the parameters space of the network resulting in a flip in the prediction.

In the tropical adversarial attacks section, we show that the problem of designing an adversarial attack x 0 + η that flips the network prediction is closely related to the problem of flipping the network prediction by perturbing the network parameters in the first layer A 1 + ζ A1 where both problems are related through a linear system.

That is to say, if one finds ζ A1 (perturbations in the first linear layer) that perturbs the geometric structure (convex hull between two zonotopes, i.e. decision boundaries) sufficiently enough to flip the network prediction, one can find an equivalent pixel adversarial attack η by solving the linear system A 1 η = ζ A1 x 0 that flips the prediction of the original unperturbed network (see the end of page 7).

We incorporate this in an overall objective (5) where the linear system is incorporated as a constraint.

Upon solving (5) with the proposed Algorithm 1, we attack the original network (unperturbed) with the adversarial attack η.

Therefore, this is comparable to the classical adversarial attacks framework.

This approach indeed resulted into flipping the network prediction over all tested examples on the MNIST dataset.

As highlighted at the end of section 6, we do not aim in our approach to outperform the state of the art adversarial attacks, but rather to provide a novel geometrically inspired perspective that can shed new light in this field.

Q7: Two previous papers investigated the decision boundary of the deep neural networks in the presence of adversarial examples [3] [4] .

Please discuss how the introduced method in this paper is placed among these methods.

The work of [3] analyzed the geometry of adversarial examples by means of manifold reconstruction to study the trade off between robustness under different norms.

On the other hand, [4] crafted adversarial attacks by estimating the distance to the decision boundaries using random search directions.

Both of the papers made a local estimation of the decision boundary around the attacked point to construct the adversarial attack to the input image.

In our work, we geometrically characterized the decision boundaries in Theorem 2 where the polytope (convex hull of two zonotopes) is only a function of the network parameters and NOT the input space.

We presented a dual view to adversarial attacks in which one can construct adversarial examples by investigating network parameters perturbations that results in the largest perturbation to this polytope representing the decision boundaries.

The scope of our work, and unlike prior art [3, 4] , is focused towards a new geometric polytope representation of the decision boundary in the network parameter space (not the input space) through a new novel analysis.

We have added a discussion of both papers in the adversarial attacks section (Section 6).

We have addressed the concerns of R2 and left the changes in blue in the revised version.

[1] "On the decision boundary of deep neural networks".

SLi, Yu and Richtarik, Peter and Ding, Lizhong and Gao, Xin.

We thank R1 for the constructive detailed thorough review of the paper and for acknowledging our contributions and the new insights.

Follows our response to R1's concerns.

In regards to clarity, exposition and focus.

To improve the clarity and exposition, we have added several paragraphs in the revised version of the paper.

The revised edits are marked in blue.

As in regards to the focus, we found that it is challenging within a reasonable time to carry out this major change in the paper.

To that end, we have done our best to further elaborate on several key results in the paper.

For instance, we have merged the paragraph above the contributions with the contributions paragraph.

We have added another paragraph dissecting Theorem 2.

We have added some few relevant references that are essential for the context of motivating tropical adversarial attacks.

In regards to the suggestions.

• Adding the information that the semiring lacks the additive inverse.

This has been addressed in the revised version.

•Adding tropical quotient to definition 1.

This has been addressed in the revised version.

• Definition of π and the upper faces.

Indeed, π is a projection operator that drops the last coordinate.

As for upper faces, the formal definition is given as follows, for a polytope P , F is an upper face of P if x + te / ∈ P for any x ∈ F, t > 0 where e is a canonical vector.

That is the faces that can be seen from "above".

A good graphical example can be found in Figure 2 from Zhang et.

al. 2018 .

We dropped this definition from the paper as it may add some confusion while it does not play an important role in the later analysis.

• Theorem 2 lacks intuitive formulation.

This has been addressed in the revised version and we have rearranged the structure of text below Theorem 2.

• Regarding the issue of the tropical hypersurface.

Indeed, the superset is in terms of set theory.

That is to say, the set of decision boundaries B is a supset of the tropical hypersurface set T (R(x)).

• Regarding Figure 2 .

The color map represents the compression percentage.

We have added a legend to Figure 2 .

Note that the second figure in Figure 2 tilted "original polytope" represents the polytope of the dual subdivision (convex hull between two zonotopes).

While the polytope seems to have only 4 vertices, there are in fact many other overlapping vertices and thereafter many small edges between all the seemingly overlapping vertices.

It is to observe that the normals to only the 4 major edges in the "original polytope" are indeed parallel to the decision boundaries plotted by performing forward passes through the network in the first figure titled "Input Space".

Note that despite the fact that more compression is performed, the orientation of the overall polytope is preserved for the lottery ticket initialization and thereafter preserving the main orientation of the decision boundaries.

This is unlike the the other types of initialization where the orientation of the polytope is vastly different with different compression ratios resulting into a larger change in the orientation of the decision boundaries.

• I would suggest to place Figure 1 after stating Theorem 2, since it is only referenced later on.

Furthermore, the red structures are somewhat confusing.

According to Theorem 2, the decision boundary is a subset of the hypersurface, right?

What is the relation of the red structures in the convex hull visualisation?

The caption states that they are normals, but as far as I can tell, this has not been formalised anywhere in the paper (it is used later on, though).

Correct.

The decision boundaries are subsets of the tropical hypersurfaces.

However, it has been shown by Maclagan & Sturmfels (2015) (Propositon 3.1.6) that the tropical hypersurface T to any d variate tropical polynomial is the (d-1)-skeleton of the polyhedral complex dual to the dual subdivision δ().

This implies that the normals to the edges (faces in higher dimensions) are parallel to the tropical hypersurface.

If R1 is interested in learning more about this, an excellent starting point to build up this intuition is the work of Erwan Brugall and Kristin Shaw "A bit of tropical geometry".

Please refer to section 2.2 "Dual subdivisions" pages 6 and 7.

• The discussion about the functional form.

We elaborate on this in section F in the appendix.

Instead of investigating the decision boundaries polytope of the tropical polynomial representing the decision boundaries R(x), we analyze the 4 different tropical polynomials representing the 2 different classes.

Recall each output function of the network is a tropical rational of two tropical polynomials giving rise to 4 different polytopes.

Figure  7 shows that the decision boundary polytope δ(R(x)) with the lottery ticket is hardly changed while pruning the network (first column in Figure 7 ).

On the contrary, the zonotopes (dual subdivisions of the 4 different tropical polynomials) vary much more significantly (coloumns 2,3,4,5).

This demonstrates that there can exist different tropical polynomials representing the functional form of the network (i.e. H 1,2 and Q 1,2 ) while having the same structure for the decision boundary polytope the corresponding first figure in the same row of Figure 7 .

This is a mere observation and worth investigating in future work.

• In Section 4, how many experiments of the sort were performed?

I find this a highly instructive view so I would love to see more experiments of this sort.

Do these claims hold over multiple repetitions and for (slightly) larger architectures as well?

We already had one extra experiment (Figure 8 ) in the appendix for another data.

We have also based on the suggestion of R1 added two more experiments (Figure 9 ) on two other datasets.

Note that extracting the decision boundaries polytope for larger architectures is much more difficult for two different reasons.

First, for deeper networks beyond the structure Affine-ReLU-Affine, the decision boundaries polytope is generic and no longer enjoys the nice properties zonotopes exhibit.

Enumerating their vertices rapidly turns to a computationally intractable problem.

Secondly, one can perhaps visualize the polytope for networks with 3 dimensional input but it gets trickier beyond that.

• Regarding that the claim that orientations are preserved should be formalised.

That is an excellent question.

We have investigated several metrics that try to capture information about the orientation of a given polytope.

For instance, we have investigated the feature that is the histogram of the oriented normals.

That is a histogram of angles for all the normals to edges given a polytope.

We have also investigated the Hausdorff distance as a metric between polytopes.

However, we have decided to keep this for a future direction as this is by itself a entirely new line of work.

That is designing the distance functions d(.) between polytopes that captures orientation information that can be used for tropical pruning (objective 8) or perhaps other applications.

Another interesting direction is whether such a distance function can be learnt in a meta-learning fashion.

For now, we restrict the experiments to the approximation used in objective (2) which for now only captures the distance between the sets of generators of the zonotopes in Eucledian sense.

• Adding link to the definition of minkowski sum in the appendix.

This has been addressed in the revised version.

• Description to other pruning methods.

This has been addressed in the revised version where we have added the definition to all pruning competitors.

• About the plots in Figure 4 .

In the experiments of Figure 4 , there is no stochasticity.

All base networks (AlexNet and VGG16) are trained before hand to achieve the state-of-art baseline results on the respective datasets (SVHN, CIFAR10 and CIFAR100).

The networks are then fixed and several pruning schemes, including the tropical approach, are applied with a varying level of pruning ratio.

We do not re-train the networks after each pruning step.

Thus there exists no source of randomness.

However, this is still an excellent observation.

This is since we conduct experiments in the appendix where after each pruning ratio we fine tune only the biases of the classifier (Figure 10 ) or we fine tune the biases of the complete network (Figure 11 ).

In such experiments, due to the fine tuning step, we will consider in the final version to report the results averaged over multiple runs.

• In Section 6, I find the comment on normals generating a superset to the decision boundaries hard to understand.

Indeed, the wording of the sentence was sub optimal.

The statement was to reassure what has been established in earlier sections that the normals to the decision boundary polytope δ(R(x)) represent the tropical hypersurface set T (R(x)) which is a super set to the decision boundaries set B as per Theorem 2.

We have rephrased the sentence.

• Take-away message regarding perturbing the decision boundaries.

The proposed approach of perturbing the decision boundaries is a mere dual view for adversarial attacks.

That is to say, to flip a network prediction for a sample x 0 , one can either adversarially perturb the sample (add noise) to cross the decision boundary to a new classification region or one can perturb the decision boundaries to move closer to the sample x 0 to appear as if it is in a new classification region.

Figure 5 , demonstrates the later visually through perturbing the decision boundary polytope.

One can observe that perturbing the correct edge in the polytope in Figure 5 , corresponds to altering a specific decision boundary.

Figure 5 , indeed as correctly pointed out of R1, is a feasibility study showing that one can perturb decision boundary by perturbing the dual subdiviosn polytope.

However, the real take away message is that these two views (perturbing the input or the parameter space) are intimately related through a linear system as discussed in the subsection titled "Dual View to Adversarial Attacks".

We propose an objective function (Problem 5) and an algorithm (Algorithm 1) to tackle this problem.

The solution to problem (5) provides an INPUT perturbation that results in altering the network prediction.

That is to say, the new framework allows for incorporating geometrically motivated objective function towards constructing classical adversarial attacks.

• Regarding the future extension on CNNs and GCNs.

We are currently investigating efficient extension of the current results to convolutional layers.

Note that while convolutional layers can be represented with a large structured topelitz/circulant matrix, we are interested in extensions that allow for similar analysis but as a function of the convolutional kernel surpassing the need to constructing convolutional matrix.

The GCNs is definitely an exciting excellent future direction that we have not yet entertained.

• Minor style issues.

We have addressed all the style issues in the revised version.

@highlight

Tropical geometry can be leveraged to represent the decision boundaries of neural networks and bring to light interesting insights.