We introduce an analytic distance function for moderately sized point sets of known cardinality that is shown to have very desirable properties, both as a loss function as well as a regularizer for machine learning applications.

We compare our novel construction to other point set distance functions and show proof of concept experiments for training neural networks end-to-end on point set prediction tasks such as object detection.

Parametric machine learning models, like artificial neural networks, are routinely trained by empirical risk minimization.

If we aim to predict an output y ∈ Y from an input x ∈ X , we collect a training set D of (x, y) pairs and train a parametrized prediction function h θ : X → Y by minimizing DISPLAYFORM0 Here, L : Y × Y → R is a loss function that assigns a scalar loss value to the predictionŷ = h θ (x) for the true value y. Classical machine learning problems allow for standard choices for the loss function.

E.g., for regression problems, where y,ŷ ∈ R, it is common to use the squared loss L(ŷ, y) = (ŷ − y) 2 .Assume, however, that we want to train a machine learning system which-given some inputpredicts a set of points.

In particular, the ordering of the points is neither semantically meaningful nor in any way consistent across data instances.

As an example, one may consider an object detection task such as finding the positions of the black stones on an image of a real-world game of Nine Men's Morris.

What should the loss function be in such a case?Naively, we might just impose an arbitrary ordering to the sets and treat them as ordered tuples.

However, this conceals an important property of the task, the permutation invariance, from our machine learning system.

This property might in fact be crucial to learn such a task efficiently.

Instead, we would like our loss function to define a meaningful distance between two sets, which requires such a loss to be permutation-invariant.

We consider distance functions a for pair of finite sets of points from R d and assume the cardinality N to be the same for both sets and known in advance.

While we are concerned with unordered sets of points, representing them in a machine-learning context will require us to impose some ordering to its elements.

Hence, we will represent a set {z 1 , . . .

, z N } ⊂ R d of points as an ordered tuple Z = (z 1 , . . .

, z N ) and define the shorthand Z N := R d N for the space of such tuples.

If a loss function L(Z,Ẑ) operating on such ordered tuples is supposed to define a meaningful distance between the underlying sets, it will have to be invariant to their arbitrary ordering.

Hence, for Z = (z 1 , . . .

, z N ) ∈ Z N and a permutation σ ∈ S N , we define σ(Z) := (z σ(1) , . . .

, z σ(N ) ) and demand the following property.

Definition 1 (Permutation-invariant loss).

We call a function L : Z N × Z N → R permutationinvariant if and only if for anyẐ, Z ∈ Z N and permutations π, σ ∈ S N , we have DISPLAYFORM0 1.2 RELATED WORK While distance measures for sets have been studied in Mathematics and theoretical computer science, they have rarely been investigated in the context of machine learning.

A natural way to define a point set distance is to explicitly find a "best" matching between target and prediction points and take the distance between the re-ordered tuples, e.g., DISPLAYFORM1 This has been proposed under the name Hungarian loss by BID3 for object detection.

The name refers to the Hungarian algorithm (Kuhn, 1955) , which can be used to solve the assignment problem.

To our knowledge, that is the first work to propose a permutation-invariant loss function for training a machine learning model.

This loss function is intuitive and is a simple, well-conditioned quadratic function ofẐ in regions where the optimal permutation is constant.

But it also has its drawbacks.

Its gradient is undefined at "transition points" with more than one optimal permutation.

More importantly, having to solve the assignment problem (Eq. 3) for each single evaluation of the loss function is problematic in a machine learning context, where we nowadays rely on simple, highly-efficient computational modules in frameworks such as TensorFlow BID0 .

It can also be computationally slow, since the Hungarian algorithm is O(N 3 ) and does not lend itself to a highly-parallelized implementation on GPUs.

BID2 propose to use a neural network to jointly model the elements of a set as well as the permutation in which they appear.

This approach gives rise to an alternating optimization procedure, where the first step solves for an optimal permutation using the Hungarian algorithm and the second step updates the weights of the neural network.

We note that their probabilistic formulation provides an elegant framework to learn the cardinality of the set alongside its elements.

While this is certainly important in practice, in this paper we focus on investigating properties of some point set distance functions, assuming the cardinality to be known.

Future work could integrate these loss functions with the cardinality learning procedure of BID2 to replace the cumbersome inner-loop solution of an optimal assignment problem.

BID4 discuss neural network architectures that take sets as inputs.

They find a general characterization of permutation-invariant functions and use that insight to devise neural network architectures which operate on sets in a permutation-invariant fashion.

In this work, we consider some alternative permutation-invariant point set distance functions, all of which have running time O(N 2 ) and can be implemented with operations that are readily available in automatic differentiation frameworks like TensorFlow.

In particular, they do not rely on the solution of an assignment problem akin to Eq. (3).

We pay specific attention to properties relevant to machine learning, in particular, how amenable such distances are to gradient-based numerical optimization.

As one option, we propose a novel distance function for point sets, which we name "holographic loss", since it is a metric distance on fingerprints of point sets that "holographically" encodes their structure, i.e. moving any point in the set will collectively (and analytically) change all entries in the fingerprint-vector.

We show that it has favorable properties in addition to being analytic everywhere, such as having a diagonal Hessian at the minima, which furthermore all are global and correspond to exactly matched up point sets.

Also, this loss function has a natural generalization to multi-sets.

We present proof of concept experiments on a simple object detection task based on MNIST digits, where we train a convolutional neural network to directly predict the locations of digits in an image.

These experiments show that end-to-end training with a simple permutation-invariant loss function is a viable approach for problems that can be formulated as a point set prediction task.

Distance functions for sets have been studied at least since the days of Hausdorff's 1914 book on set theory Hausdorff (1914) .

Considering Hausdorff's construction in our context, i.e. as a (2nd order) loss function for finite point sets, the one-sided Hausdorff distance is given by DISPLAYFORM0 2 , which is the maximum squared distance of any target point to its closest prediction point.

The symmetric Hausdorff distance is DISPLAYFORM1 L H,2 is easily computed with O(N 2 ) effort, almost everywhere differentiable, and makes intuitive sense as a machine learning loss function.

However, it has some properties that might not be desirable in a machine learning context.

Since the Hausdorff distance is determined by a single "worstoffender" pair of target and prediction points, each gradient descent step will only adjust the position of a single prediction point, until there is an almost-tie between different point-pairs, and henceforth subsequent gradient steps will jump around and keep almost-tied point pairs almost-tied.

It stands to reason that this might, especially with some optimization strategies, be undesirable compared to loss functions whose gradient collectively moves all points.

A possible remedy to that problem is to sum the minimal distances-squared instead of taking their maximum.

The one-sided sum of minimal distances-squared is DISPLAYFORM2 We do not consider the one-sided variants of these two distance functions in our comparison.

The one-sided variants centered on the prediction points will not make sense as a machine learning loss function, since they become zero when matching all predicted points to any one of the target points, e.g.,ẑ j = z 1 for all j.

The one-sided variants centered on the target points, while more sensible, have simple failure cases as well, such as the one depicted in Figure 4 .

We thus restrict our attention to the symmetric loss functions L H and L SMD .

FIG1 show contour plots for these two functions on a simple problem with two one-dimensional target points, while 1c shows our construction, described in the next section.

A general strategy to devise permutation-invariant loss functions is to (differentiably) map tuples Z representing a point set (or multi-set) to a "fingerprint" vector f (Z) in some fingerprint-space F such that f (Ẑ) = f (Z) if and only ifẐ = σ(Z) for some permutation σ.

A permutation-invariant loss function can then be defined via (some) metric distance in F. Any such loss function will inherit properties of the distance in F, such as positive definiteness, symmetry, or the triangle inequality.

In one dimension, a suggestive choice for a fingerprint function on tuples of N points would be a vector of moments DISPLAYFORM0 The mathematical appendix of BID4 identifies the moments of points in a set as an universal permutation-invariant fingerprint, in the sense that every other such fingerprint can be defined in terms of these quantities.

This suggest using squared-euclidean-distance-of-moments-vectors as a loss function.

However, this has some fundamental problems, such as poor conditioning, and it is non-trivial to generalize to higher-dimensional points.

See Appendix B for a detailed discussion.

In this section, we want to focus our attention to two-dimensional point sets (or multi-sets), and consider one-dimensional sets as a special case with all points having a 2nd coordinate of zero.

We will subsequently consider generalizations to higher dimensions.

We start by identifying a two-dimensional point z = (x, y) ∈ R 2 with the complex number z = x + iy ∈ C. With this, we can map any tuple Z = (z 1 , . . .

, z N ) that represents a (multi-)set to a complex polynomial DISPLAYFORM0 Note that P Z = PẐ if and only ifẐ = σ(Z) for some permutation σ ∈ S N .

Furthermore, such a monic (i.e. having leading coefficient 1) polynomial of degree N is uniquely determined by its values at N distinct points.

This suggests to choose a set U = (u 1 , . . . , u N ) of N distinct evaluation points and to define a fingerprint as DISPLAYFORM1 Taking the real-valued L 2 distance in C N gives us the "holographic" loss function DISPLAYFORM2 While the underlying fingerprint is holomorphic, i.e. complex differentiable, the distance-squared function is not, due to anti-linearity of the underlying complex scalar product in the first argument.

The adjective 'holographic' here refers to this function depending collectively on the locations of all points, unlike Hausdorff distance, which for a generic configuration will not change when slightly changing the positions of any points other than the specific pair that determines its value.

Also, for U = Z, correctly matching up the candidate and target points means matching up both the complex amplitude as well as the complex phase at the evaluation-points.

However, the analogy with holography ends here, there is no deeper correspondence beyond that.

Proposition 1.

The Holographic loss trivially has the following properties: DISPLAYFORM3 • Symmetry: DISPLAYFORM4 • L Hol U (Ẑ, Z) 2 is quadratic in each individualẑ i 's real and also imaginary part when keeping all other point-coordinates fixed.

If the target points form a set rather than a multi-set (i.e. there are no duplicates), we have the option to choose the set U of evaluation points as identical to the target points Z. Then, since P Z (z i ) = 0 by construction, we get the simpler loss function DISPLAYFORM0 In the two-dimensional case, note that the squared absolute value of the complex number z i −ẑ j is identical to the squared Euclidean distance z i −ẑ j 2 of the two points in R 2 : DISPLAYFORM1 This target-centered version will be used in the practical applications presented here.

For each target point, we take the product of squared distances to all prediction points and then sum this quantity over all target points.

If a target point z i is matched closely by any of the prediction points, the product of squared distances will tend to be small and, consequently, z i will not contribute significantly to the loss.

Note that, unlike Eq. FORMULA9 , this loss function will no longer be symmetric.

Other than that, it inherits the properties of Proposition 1.

There is no higher-dimensional equivalent of the polynomial construction in Eq. (6), since there is no known generalization of the Fundamental theorem of Algebra to higher-dimensional real algebras beyond D = 2.

However, we can heuristically extend Eq. (10), which is a simple sum of products of squared Euclidean distances, to point sets of arbitrary dimension.

As we show in the next section, for D = 2, the correspondence between point sets and polynomials can be used to prove some favorable properties of the loss function.

We do not yet have a proof for these properties for D = 2.

The gradient of the Holographic loss with an arbitrary set of evaluation points (Eq. 8) can be computed using complex backpropagation of errors, see Appendix A. Here, we restrict our attention to the gradient of Eq. (10) with respect tox k andŷ k , which evaluates to DISPLAYFORM0 This gradient has a rather intuitive interpretation (also for D > 2).

The negative gradient is a weighted sum of the vectors (z i −ẑ k ), which pulls the predictionẑ k in the direction of target point z i .

The weight of this attractive force in the direction of z i is ∝ j =k z i −ẑ j 2 , which measures how well point z i is already matched by any prediction point other thanẑ k .

Unlike the Hausdorff distance, the gradient of the Holographic loss thus adjusts the position of all points simultaneously.

We can get some intuition for the dynamics induced by this gradient from example solutions of the gradient flow ODE, see Appendix C.Stationary Points Obviously, any permutation-invariant loss function will be non-convex by the mere fact that it has multiple global minima.

Beyond that, the Holographic loss will have additional stationary points.

For example, for Z = ((0, 0), (1, 1)), the gradient vanishes at the non-optimal configurationẐ = ((0.5, 0.5), (0.5, 0.5)).

The following Proposition shows that, for D = 2, these non-optimal stationary points can not be local minima.

The proof (see Appendix A) relies on complex polynomials and, thus, only holds for two-dimensional point sets.

While we conjecture that this property holds for the Holographic loss in Eq. (10) for points of any dimension, we currently do not have a proof for this.

The following proposition characterizes the behavior of the Holographic loss near optima up to second order.

The proof can be found in Appendix A.Proposition 3.

Let Z be fixed and define DISPLAYFORM0 2 , whereẐ is treated as a R D·N vector.

At any global optimum whereẐ = σ(Z) with σ ∈ S N , the Hessian matrix DISPLAYFORM1 each appearing D times.

The nature of the diagonal elements, which are also the eigenvalues of the Hessian, suggests that the problem can become ill-conditioned if the pairwise distances between points have vastly different scales, e.g., if there is a pair of points that is very close to each other relative to all the other points.

One might even be worried whether minimizing the Holographic loss to numerical accuracy correctly matches up point sets in such pathological situations.

In Appendix B, we show that this not a problem even for sets with about 40 randomly sampled points.

This behavior is to a large extent due to the property of the Hessian always being diagonal at minima.

Since the eigendirections of the Hessian coincide with the coordinate directions, floating point numerics can represent the gradients in these eigendirections with high relative precision despite their vastly different scales.

This concludes our description of the Holographic loss.

TAB0 gives an overview of different properties of the loss functions under consideration.

We apply the three point set loss functions-Hausdorff, sum of minimal distances-squared, and Holographic loss-to a simple object detection toy tasks.

The purpose of these experiments is to demonstrate the viablity of end-to-end training with simple permutation-invariant loss functions without explicitly modelling permutations or having to solve optimal assignment problems.

Object detection is one of the most prominent tasks in the field of computer vision.

Given an input image, the task is to predict the locations (e.g., center points or bounding boxes) of objects shown in this image.

Since there usually will not be any meaningful or consistent ordering of objects in an image, this is inherently a point set prediction task.

However, existing approaches to object detection avoid treating it as such.

Instead, state-of-the-art object detectors are carefully engineered systems combining multiple components.

R-CNN (Girshick et al., 2014; BID1 BID1 generates proposal regions hypothesizing object locations and subsequently assigns a score to each region independently, indicating how likely it is to actually contain an object.

YOLO (Redmon et al., 2016) divides an image into a grid and asks each grid cell to predict a pre-specified number of likely object bounding boxes, together with confidence scores.

For both R-CNN and YOLO, generating a final output set involves post processing steps like non-maximum suppression.

None of these systems learn the map from an image input to the object locations end-to-end.

We want to demonstrate that this is possible to do with any of the three point set loss functions discussed in this paper.

Dataset For this proof of concept experiment, we create a simple object detection data set, MNIST-DETECT, with a fixed number of objects (MNIST digits) in each image.

To generate an MNISTDE-TECT image, we sample four MNIST digits, crop them to a tight bounding box and rescale them by a factor chosen uniformly at random from {0.5, 0.6, 0.7, 0.8, 0.9}. We then place them in random locations of a 50 × 50 pixels image, allowing overlap of the bounding boxes, but not of the actual digits.

Finally, we add noise to the image.

FIG2 shows some examples.

Each example is annotated with the center location as well as the bounding box of each digit with no particular ordering.

Neural Architecture and Training We use a very simple convolutional neural network that takes such an MNISTDETECT image as input and applies three convolutional layers (with 64 filters of 3×3 pixels each) and two fully-connected layers (500 and 200 neurons, respectively), all with ReLU activation.

The output layer is of size 4D with no non-linearity and is reshaped to a 4 × D array, where each row is supposed to predict the location of one of the four digits in the image.

We performed separate experiments for predicting center locations (D = 2) and bounding boxes (D = 4).

The network is trained end-to-end using Hausdorff, SMD, and Holographic loss.

For each loss function, we train the network for a fixed number of epochs using the Adam optimizer (Kingma & Ba, 2014) and a fixed mini-batch size of 128.

We evaluate the loss on a validation set after each epoch of training and retain the weights that achieve minimal validation loss.

The step size is tuned for each loss function independently via a grid search, but turned out to be the same (0.001) for all loss functions.

Evaluation For each loss function, we evaluate the network that achieved minimal validation loss on a held-out test set.

As "impartial" comparison metrics, we report the Hungarian loss for center point prediction and detection rate for bounding box prediction.

In the latter case, we count a digit as successfully detected if its bounding box has an intersection over union (IoU) larger than 0.5 with any of the predicted bounding boxes.

TAB1 shows quantitative results and FIG2 depicts some qualitative results for bounding box prediction.

With all three loss functions, the neural network manages to learn the task reasonably well even though it is a simplistic CNN architecture that has not been tuned to the task at all.

We discussed a novel point set loss function, which is analytic everywhere, vis-a-vis two other simple alternatives with matching computational complexity for point set prediction tasks as alternatives to more involved approaches BID3 BID2 .

Proof of concept experiments showed that end-to-end training with such simple loss functions is a viable approach for point set prediction tasks, such as object detection.

We expect that simple constructions such as the "holographic" point set distance introduced here may turn out useful not only for point set predictions, but also as a regularizer, for example to encourage (unsupervised) clustering to align its clusters with the clusters found by an earlier version of the model.

We will need the following Lemma, which states that the roots of a complex polynomial continuously depend on its coefficients.

DISPLAYFORM0 k=0 c k u k be a monic complex polynomial and factor it as F (u) = (u − a 1 )(u − a 2 ) · · · (u − a N ) with a k ∈ C some ordering of the roots.

Then, for every ε > 0 there exists δ > 0 such that every polynomial DISPLAYFORM1 Proof.

This is a reformulation of the well-known continuity result, a proof of which can be found, for example, inĆurgus & Mascioni (2006) .

One notes that, to first order in a small shift in the coefficients of a polynomial, the shift of the zeros can be found by a single step of Newton-Raphson iteration, except at higher-degree zeros (where the derivative of the polynomial in the denominator also has a zero).We can now prove the Proposition.

DISPLAYFORM2 We already observed that L(Ẑ, Z) 2 = 0 if and only ifẐ = σ(Z) for some σ ∈ S N .

Now assume L(Ẑ, Z) 2 = 0.

To see thatẐ can not be a local minimum, we need to show that for any ε > 0 there DISPLAYFORM3 To construct such a Z , we define the polynomial Q λ : C → C, DISPLAYFORM4 which linearly interpolates between PẐ and P Z .

Note that Q λ is monic for λ ∈ [0, 1] and Q 0 = PẐ.

Let Z λ be some ordering of the roots of Q λ .

Then DISPLAYFORM5 Since we assumed L(Ẑ, Z) > 0, this means that L(Z λ , Z) < L(Ẑ, Z) for any λ > 0.

It remains to show that for any ε > 0 there is λ ε > 0 and a permutation σ such that d(Ẑ, σ(Z λε )) < ε.

However, since the coefficients of Q λ continuously depend on λ, this follows as a simple consequence of Lemma 1.

Proof of Proposition 3.

Denote by z k,l := [ẑ k ] l the l-th coordinate of the k-th prediction point (and likewise for target points).

We have previously found the first derivatives of the loss w.r.t.

the elements ofẐ to be DISPLAYFORM0 We can easily calculate the second derivatives to be DISPLAYFORM1 At any optimum, the latter becomes zero, since for each z i there is exactly one j ∈ [N ] such thatẑ j = z i .

This makes the Hessian matrix diagonal.

Furthermore, the diagonal elements simplify at an optimum: Letting σ be the specific permutation of this optimum, such thatẑ j = z σ(j) , we get DISPLAYFORM2 The product becomes zero for all but one i: the one where σ(j) = i for all j ∈ [N ]\{k}, which is exactly σ −1 (k).

The diagonal element becomes DISPLAYFORM3 This is independent of l and thus appears D times for l = 1, . . .

, D and the set of diagonal elements is the same for each optimum (different optima correspond to different permutations σ, which only changes the ordering).

In D = 2, the Holographic loss (Eq. 8) is polynomial in the 2N real coordinates of N points, so a gradient can be obtained via backpropagation in the usual way.

When using a set of evaluationpoints U that does not coincide with the target points Z, there is some additional structure available due to the loss being based on a complex polynomial (hence complex-differentiable) that can be exploited to structurally simplify the computation.

To apply Wirtinger calculus (a readable introduction can be found in Hunger FORMULA1 ), we rewrite our loss function (considering Z and U fixed) DISPLAYFORM0 Then,L 2 is a complex-differentiable function on C N × C N , and the usual reasoning for sensitivity backpropagation also applies in this complex case.

Informally, denote the sensitivites ofL 2 w.r.t.

v j and w j by σ vj and σ wj .

It turns out that σ wj = σ * vj .

Now, we ultimately want to know the sensitivities on the (x,ŷ)-coordinates of the candidate points, that is, the real and imaginary parts of theẑ j .

Thanks to complex differentiability, we know that changing v j → v j + C and keeping W fixed will changeL by C ·

σ vj -and correspondingly for w j .

Now, if we simultaneously change (v j , w j ) → (v j + R , w j + R ) with R real, which corresponds to changing ( DISPLAYFORM1 , and henceL will change by i R · σ vj − i R · σ wj = i R · σ vj − i R · σ * vj = −2 R Im σ vj .

This, then, tells us how to link up the real sensitivities (gradient components) with the complex sensitivities: DISPLAYFORM2 B DIFFERENCE OF MOMENTS

Restricting our attention to one-dimensional points, an obvious choice for a permutation-invariant fingerprint would be moments of point sets.

With N = |Z|, the k-th moment of DISPLAYFORM0 Knowing the values of these elementary-symmetric functions for k = 1, . . .

, N completely determines the point set, making DISPLAYFORM1 T ∈ R N a valid set fingerprint.

Indeed, as explained in Appendix A of BID4 , this is in some sense the fundamental permutation-invariant set fingerprint, since every symmetric function on sets of N real numbers can be written in the form ρ(m Z ) with some suitable function ρ.

As explained there, this nicely parallels the Kolmogorov-Arnold theorem (Kolmogorov, 1956; Arnold, 1957) for the symmetric case.

However, this insight is only of limited use for constructing a practical permutation-invariant loss function, since numerical conditioning aspects have to be taken into account.

Specifically, simply taking the L 2 -distance between these moments fingerprints DISPLAYFORM2 turns out to have fundamental problems matching up even moderately-sized real point sets, see Figure 3 for a 2 − d example.

Beyond ∼ 7 points, there are ways to collectively shift points such that the impact of the distortion on low-order moments is compensated, while the impact on highorder moments falls below the numerical accuracy threshold.

Typically, the end result of a failed attempt to match up N real points using L Mom,2 will have points near zero being considerably off the target locations.

In the same experiment, the Holographic loss reliably matches up point sets with more than 40 randomly sampled points.

If we castẐ → L Hol U (Ẑ, Z) 2 into the ρ(m Z ) language it takes on the form of a polynomial in all the point coordinates (in two dimensions: DISPLAYFORM3 with all exponents ξ i,k , η i,k being either 0, 1, or 2.

This largely avoids numerical problems due to the locations of zeroes of higher-degree polynomials being ill-conditioned w.r.t.

coefficients.

So, the "holographic" loss function can be considered as "maximally simple" in the sense of being a quadratic function when considering each input-coordinate separately.

For large point sets, it may be useful to generalize the "holographic" loss function by also allowing weights for the contributions coming from different evaluation-points, in order to minimize the discrepancies between diagonal entries of the Hessian.

It is not obvious how one would extend the moments-based loss to higher dimensions.

Using just the N · D moments for every coordinate, DISPLAYFORM0 will not give us a proper set fingerprint, since it does not discriminate between point sets obtained by shuffling the values of any one coordinate between points.

That is, it could not discriminate {(x 0 , y 0 ); (x 1 , y 1 )} from {(x 0 , y 1 ); (x 1 , y 0 )}.

A more appropriate choice are the moments DISPLAYFORM1 for j k j ≤ N .

However, this scales infavorably with N and one would naturally expect the same numerical problems we have seen in one dimension to persist in higher dimensions.

This is confirmed in the investigations described next, where we used as a fingerprint the vector of all moments up to total scaling dimension N in D = 2, i.e. x a y b with a + b ≤ N .

In Figure 3 we show experiments matching up sets of randomly sampled points by minimizing L Mom,2 and L Hol,2 to numerical accuracy.

We see that, using 64-bit floating point numerics, the Holographic loss can in principle be used to match up sets with more than 40 randomly sampled points. (25) Figure 5 shows multiple instances of matching up sets of five two-dimensional points.

Lines show how the prediction points move according to the ODE (25).

Along each line, markers are placed indicating exponentially spaced time-steps t n = 2 n · ∆t with every fifth marker being drawn in black.

(These do not correspond to iterations of gradient descent, but to the exact solution of the ODE.

In the limit of infinitely small step size, the gradient descent dynamics would approach these curves.)

Figure 5 : Gradient flows illustrating the behavior of the Holographic loss function.

Left column: If there are reasonably clear ways to match up the two point sets, the gradient flow tends to find that identification and moves each point simultaneously in a meaningful way.

This allows each entry of the gradient to forward meaningful information towards earlier layers in ML training, unlike with Hausdorff-distance.

Right column: In more complex cases (such as two points almost coinciding), multiple points may travel alongside one another until the degeneracy finally gets broken.

@highlight

Permutation-invariant loss function for point set prediction.

@highlight

Proposes a new loss for points registration (aligning two point sets) with preferable permutation invariant property. 

@highlight

This paper introduces a novel distance function between point sets, applies two other permutation distances in an end-to-end object detection task, and shows that in two dimensions all local minima of the holographic loss are global minima.

@highlight

Proposes permutation invariant loss functions which depend on the distance of sets.