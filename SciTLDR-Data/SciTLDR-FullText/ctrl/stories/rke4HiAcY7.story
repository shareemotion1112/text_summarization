Information bottleneck (IB) is a method for extracting information from one random variable X that is relevant for predicting another random variable Y. To do so, IB identifies an intermediate "bottleneck" variable T that has low mutual information I(X;T) and high mutual information I(Y;T).

The "IB curve" characterizes the set of bottleneck variables that achieve maximal I(Y;T) for a given I(X;T), and is typically explored by maximizing the "IB Lagrangian", I(Y;T) - βI(X;T).

In some cases, Y is a deterministic function of X, including many classification problems in supervised learning where the output class Y is a deterministic function of the input X. We demonstrate three caveats when using IB in any situation where Y is a deterministic function of X: (1) the IB curve cannot be recovered by maximizing the IB Lagrangian for different values of β; (2) there are "uninteresting" trivial solutions at all points of the IB curve; and (3) for multi-layer classifiers that achieve low prediction error, different layers cannot exhibit a strict trade-off between compression and prediction, contrary to a recent proposal.

We also show that when Y is a small perturbation away from being a deterministic function of X, these three caveats arise in an approximate way.

To address problem (1), we propose a functional that, unlike the IB Lagrangian, can recover the IB curve in all cases.

We demonstrate the three caveats on the MNIST dataset.

The information bottleneck (IB) method BID29 provides a principled way to extract information that is present in one variable that is relevant for predicting another variable.

Given two random variables X and Y , IB posits a "bottleneck" variable T that obeys the Markov condition Y − X − T .

By the data processing inequality (DPI) BID9 , this Markov condition implies that I(X; T ) ≥ I(Y ; T ), meaning that the bottleneck variable cannot contain more information about Y than it does about X. In fact, any particular choice of the bottleneck variable T can be quantified by two terms: the mutual information I(X; T ), which reflects how much T compresses X, and the mutual information I(Y ; T ), which reflects how well T predicts Y .

In IB, bottleneck variables are chosen to maximize prediction given a constraint on compression BID32 BID1 BID13 , where ∆ is the set of random variables T obeying the Markov condition Y − X − T .

The values of F (r) for different r specify the IB curve.

In order to explore the IB curve, one must find optimal T for different values of r. It is known that the IB curve is concave in r but may not be strictly concave.

This seemingly minor issue of non-strict concavity will play a central role in our analysis.

In practice, the IB curve is almost always explored not via the constrained optimization problem of Eq.(1), but rather by maximizing the so-called IB Lagrangian, Several recent papers have drawn connections between IB and supervised learning, in particular classification using neural networks.

In this context, X represents input vectors, Y represents the output classes, and T represents intermediate representations used by the network architecture, such as the activity of hidden layer(s) BID30 .

Some of these papers modify neural network training algorithms so as to optimize the IB Lagrangian BID2 BID7 , thereby permitting the use of IB with high-dimensional, continuousvalued random variables.

Some papers have also suggested that by controlling the amount of compression, one can tune desired characteristics of trained models such as generalization error BID23 BID30 BID31 , robustness to adversarial inputs BID2 , and detection of out-of-distribution data BID3 .

Other research (ShwartzZiv & Tishby, 2017) has suggested -somewhat controversially BID22 -that stochastic gradient descent (SGD) training dynamics may implicitly favor hidden layer mappings that balances compression and prediction, with earlier hidden layers favoring prediction over compression and latter hidden layers favoring compression over prediction.

Finally, there is the general notion that intermediate representations that are optimal in the IB sense correspond to "interesting" or "useful" compressions of input vectors BID4 .There are also numerous application domains of IB beyond supervised learning, including clustering BID25 , coding theory and quantization BID6 BID34 BID8 , and cognitive science BID33 .

In most of these applications, it is of central interest to explore solutions at different points on the IB curve, for example to control the number of detected clusters, or to adapt codes to available channel capacity.

In some scenarios, Y may be a deterministic function of X, i.e., Y = f (X) for some single-valued function f .

For example, in many classification problems, it is assumed that any given input belongs to a single class, which implies a deterministic relationship between X and Y .

In this paper, we demonstrate three caveats for IB that appear whenever Y is a deterministic function of X: 1.

There is no one-to-one mapping between different points on the IB curve and maximizers of the IB Lagrangian L β IB for different β, thus the IB curve cannot be explored by maximizing L β IB while varying β.

This occurs because when Y = f (X), the IB curve has a piecewise linear shape and is therefore not strictly concave.

The dependence of the IB Lagrangian on the strict concavity of F (r) has been previously noted BID13 BID24 , but the implications and pervasiveness of this problem (e.g., in many classification scenarios) has not been fully recognized.

We analyze this issue and propose a solution in the form of an alternative objective function, which can be used to explore the IB curve even when Y = f (X).

2.

All points on the IB curve contain uninteresting trivial solutions (in particular, stochastic mixtures of two very simple solutions).

This suggests that IB-optimality is not sufficient for an intermediate representation to be an interesting or useful compression of input data.

3.

For a neural network with several hidden layers that achieves a low probability of error, the hidden layers cannot display a strict trade-off between compression and prediction (in particular, different layers can only differ in the amount of compression, not prediction).

In Appendix B, we show that the above three caveats also apply to the recently proposed deterministic IB variant of IB BID26 , in which the compression term is quantified using the entropy H(T ) rather than the mutual information I(X; T ).

In that Appendix, we propose an alternative objective function that can be used to resolve the first problem for dIB.We also show, in Appendix C, that our results apply when Y is not exactly a deterministic function of X, but -close to one.

In this case: (1) it is hard to explore the IB curve by optimizing the IB Lagrangian, because all optimizers will fall within O(− log ) of a single "corner" point on the information plane; (2) along all points on the IB curve, there are "uninteresting" trivial solutions that are no more than O(− log ) away from being optimal; (3) different layers of a neural networks can trade-off at most O(− log ) amount of prediction.

A recent paper BID4 ) also discusses several difficulties in using IB to analyze intermediate representations in supervised learning.

That paper does not consider the particular 1 Note that optimizing L β IB is still a constrained problem in that p(t|x) must be a valid conditional probability.

However, this constraint is usually easier to handle, e.g., by using an appropriate parameterization.issues that arise when Y is a deterministic function of X, and its arguments are complementary to ours.

BID24 [Sec. 2.4 ] discuss another caveat for IB in deterministic settings, concerning the relationship between sufficient statistics and the complexity of the inputoutput mapping, which is orthogonal to the three caveats analyzed here.

Finally, BID22 and BID4 observed that when T is continuous-valued and a deterministic function of a continuous-valued X, I(X; T ) can be unbounded, making the application of the IB framework problematic.

We emphasize that the caveats discussed in this paper are unrelated to that problem.

Note that our results are based on analytically-provable properties of the IB curve, i.e., global optima of Eq. (1), and do not concern practical issues of optimization (which may be important in realworld scenarios).

Our theoretical results are also independent of the practical issue of estimating MI between neural network layers, an active area of recent research BID5 BID14 BID10 BID12 , though our empirical experiments in Section 7 rely on the estimator proposed in .

Finally, our results are also independent of issues related to the relationship between IB, finite data sampling, and generalization error BID23 BID30 BID31 .In the next section, we review some of the connections between supervised learning and IB.

In Section 3, we show that when Y = f (X), the IB curve has a piecewise linear (not strictly concave) shape.

In Sections 4, 5 and 6, we discuss the three caveats mentioned above.

In Section 7, we demonstrate the caveats using a neural-network implementation of IB on the MNIST dataset.

In supervised learning, one is given a training dataset {x i , y i } i=1..

N of inputs and outputs.

In this case, the random variables X and Y refer to the inputs and outputs respectively, andp(x, y) indicates their joint empirical distribution in the training data.

It is usually assumed that the x's and y's are sampled i.i.d.

from some "true" distribution w(y|x)w(x).

The high-level goal of supervised learning is to use the dataset to select a particular conditional distribution q θ (ŷ|x) of outputs given inputs parameterized by θ that is a good approximation of w(y|x).

For clarity, we useŶ (andŷ) to indicate a random variable (and its outcome) corresponding to the predicted outputs.

We use X to indicate the set of outcomes of X, and Y to indicate the set of outcomes of Y andŶ .

Supervised learning is called classification when Y takes a finite set of values, and regression when Y is continuousvalued.

Here we focus on classification, where the set of possible output values can be written as Y = {0, 1, . . .

, m}. We expect our results to also apply in some regression scenarios (with some care regarding evaluating MI measures, see Footnote 2), but leave this for future work.

In practice, many supervised learning architectures use some kind of intermediate representation to make predictions about the output, such as hidden layers in neural networks.

In this case, the random variable T represents the activity of some particular hidden layer in a neural network.

Let T be a (possibly stochastic) function of the inputs, as determined by some parameterized conditional distribution q θ (t|x), so that T obeys the Markov condition Y − X − T .

The mapping from inputs to hidden layer activity can be either deterministic, as traditionally done in neural networks, or stochastic, as used in some architectures BID2 BID0 2 .

The mapping from hidden layer activity to predicted outputs is represented by the parameterized conditional distribution q θ (ŷ|t), so the full Markov condition between true outputs, inputs, hidden layer activity, and predicted outputs is Y − X − T −Ŷ .

The overall mapping from inputs to predicted outputs implemented by a neural network with a hidden layer can be written as DISPLAYFORM0 Note that T does not have to represent a particular hidden layer, but could instead represent the activity of the neurons in a set of contiguous hidden layers, or in fact any arbitrary set of neurons that separate inputs from predicted outputs (in the sense of conditional independence, so that Eq. (3) holds).

More generally yet, T could also represent some intermediate representation in a non-neuralnetwork supervised learning architecture, again as long as Eq. FORMULA1 holds.

For classification problems, training often involves selecting θ to minimize cross-entropy loss, sometimes while stochastically sampling the activity of hidden layers.

When T is the last hidden layer, or when the mapping from T to the last hidden layer is deterministic, cross-entropy loss can be written DISPLAYFORM1 where E indicates expectation,p θ (y, t) := 1 N i δ(y, y i )q θ (t|x i ) is the empirical distribution of outputs and hidden layer activity, and δ is the Kronecker delta.

Eq. (4) can also be written as DISPLAYFORM2 DISPLAYFORM3 is a constant that doesn't depend on θ).We can now make explicit the relationship between supervised learning and IB.

In IB, one is given a joint distribution p(x, y), and then seeks a bottleneck variable T that obeys the Markov condition Y − X − T and minimizes I(X; T ) while maximizing I(Y ; T ).

In supervised learning, one is given an empirical distributionp(x, y) and defines an intermediate representation T that obeys the Markov condition Y − X − T ; during training, cross-entropy loss is minimized, which is equivalent to maximizing a lower bound on I(Y ; T ) (given assumptions of Eq. (4) hold).

To complete the analogy, one might choose θ so as to also minimize I(X; T ), i.e., choose hidden layer mappings that provide compressed representations of input data BID2 BID7 BID0 .

In fact, we use such an approach in our experiments on the MNIST dataset in Section 7.In many supervised classification problems, there is only one correct class label associated with each possible input.

Formally, this means that given the empirical training dataset distributionp(y|x), one can write Y = f (X) for some single-valued function f .

This relationship between X and Y may arise because it holds under the "true" distribution w(y|x) from which the training dataset is sampled, or simply because each input vector x occurs at most once in the training data (as happens in most real-world classification training datasets).

Note that we make no assumptions about the relationship between X andŶ under q θ (ŷ|x), the distribution of predicted outputs given inputs implemented by the supervised learning architecture, and our analysis holds even if q θ (ŷ|x) is nondeterministic (e.g., the softmax function).It is important to note that not all classification problems are deterministic.

The map from X to Y may be intrinsically noisy or non-deterministic (e.g., if human labelers cannot agree on a class for some x), or noise may be intentionally added as a regularization technique (e.g., as done in the label smoothing method BID28 ).

Moreover, one of the pioneering papers on the relationship between IB and supervised learning BID24 analyzed an artificially-constructed classification problem in which p(y|x) was explicitly defined to be noisy (see their Eq. 10).

However, we believe that many if not most real-world classification problems do in fact exhibit a deterministic relation between X and Y .

In addition, as we show in Appendix C, even if the relationship between X and Y is not perfectly deterministic but close to being so, then the three caveats discussed in this paper still apply in an approximate sense.

Consider the random variables X, Y where Y is discrete-valued and a deterministic function of X, i.e., Y = f (X).

I(Y ; T ) can be upper bounded as I(Y ; T ) ≤ I(X; T ) by the DPI, and I(Y ; T ) ≤ H(Y ) by basic properties of MI BID9 .

To visualize these bounds, as well as other results of our analysis, we use so-called "information plane" diagrams BID29 .

The information plane represents various possible bottleneck variables in terms of their compression (I(X; T ), horizontal axis) and prediction (I(Y ; T ), vertical axis).

The two bounds mentioned above are plotted on an information plane in FIG2 .

In this section, we show that when Y = f (X), the IB curve saturates both bounds, and is not strictly concave but rather piece-wise linear.

To show that the IB curve saturates the DPI bound I(Y ; T ) ≤ I(X; T ) for I(X; T ) ∈ [0, H(Y )], let B α be a Bernoulli random variable that is equal to 1 with probability α.

Then, define a manifold of bottleneck variables T α parameterized by α ∈ [0, 1], DISPLAYFORM0 Thus, T α is equal to Y with probability α, and equal to 0 with probability 1 − α.

From Eq. FORMULA5 DISPLAYFORM1 as α ranges from 0 to 1, while obeying I(X; T α ) = I(Y ; T α ).

Thus, the manifold of T α bottleneck variables achieves the DPI bound over DISPLAYFORM2 Given its definition in Eq. FORMULA0 , the IB curve is monotonically increasing.

Since DISPLAYFORM3 At the same time, it is always the case that I(Y ; T ) ≤ H(Y ), by basic properties of MI.

Thus, the IB curve is a flat line at DISPLAYFORM4 Thus, when Y = f (X), the IB curve is piecewise linear and therefore not strictly concave.

DISPLAYFORM5 where the first inequality uses the DPI, and the second DISPLAYFORM6 Now consider the bottleneck variable T copy := f (X) = Y (or, equivalently, any one-to-one transformation of T copy ), which corresponds to T α for α = 1 in Eq. FORMULA5 , and is the "corner point" of the IB curve in Fig To summarize, when Y = f (X), the IB curve is piecewise linear and cannot be explored by optimizing L β IB while varying β.

On the flip side, if Y = f (X) and one's goal is precisely to find the corner point H(Y ), H(Y ) (maximum compression at no prediction loss), then our results suggest that the IB Lagrangian provides a robust way to do so, being invariant to the particular choice of β.

Moreover, if one does want to recover solutions at different points on the IB curve, and f : X → Y is fully known, our results show how to do so in a closed-form way (via T α ), without any optimization.

Is there a single procedure that can be used to explore the IB curve in all cases, whether it is strictly convex or not?

In principle, this can be done by solving the constrained optimization of Eq. FORMULA0 for different values of r. In practice, however, solving this constrained optimization problem -even approximately -is far from a simple task, in part due to the non-linear constraint on I(X; T ), and many off-the-shelf optimization tools cannot handle this kind of problem.

It is desirable to have an unconstrained objective function that can be used to explore any IB curve.

For this purpose, we propose an alternative objective function, which we call the squared-IB functional, L DISPLAYFORM7 We show that maximizing L β sq-IB while varying β recovers the IB curve, whether it is strictly concave or not.

To do so, we first analyze why the IB Lagrangian fails when Y = f (X) (or, more generally, when the IB curve is piecewise linear).

Over the increasing region of the IB curve, the inequality constraint in Eq. (1) can be replaced by an equality constraint (Witsenhausen & Wyner, 1975, Thm. 2.5) , so maximizing L β IB can be written as max T I(Y ; T ) − βI(X; T ) = max r F (r) − βr (i.e., the Legendre transform of −F (r)).

The derivative of F (r) − βr is zero when F (r) = β, and any point on the IB curve that has F (r) = β will maximize L β IB for that β.

When Y = f (X), all points on the increasing part of the curve have F (r) = 1, and will all simultaneously maximize L β IB for β = 1.

More generally, all points on a linear segment of a piecewise linear IB curve will have the same F (r), and will all simultaneously maximize L β IB for the corresponding β.

Using a similar argument as above, we write max T L β sq-IB = max r F (r) − βr 2 .

The derivative of F (r) − βr 2 is 0 when F (r)/(2r) = β.

Since F is concave, F (r) is decreasing in r, though it is not strictly decreasing when F is not strictly concave.

F (r)/(2r), on the other hand, is strictly decreasing in r in all cases, thus there can be only one r such that F (r)/(2r) = β for a given β.

For this reason, for any IB curve, there can be only one point that maximizes L β sq-IB for a given β.

Note also that any r that satisfies F (r) = β also satisfies F (r)/(2r) = β/(2r).

Thus, any point that maximizes L β IB for a given β also maximizes L β sq-IB under the transformation β → β/(2 · I(X; T )), and vice versa.

Importantly, unlike for L β IB , there can be non-trivial maximizers of L β sq-IB for β > 1.

The effect of optimizing L β sq-IB is illustrated in FIG4 , which shows two different IB curves (not strictly concave and strictly concave), isolines of L β sq-IB for different β, and points on the curves that maximize L β sq-IB .

For both curves, the maximizers for different β are at different points of the curve.

5 For simplicity, we consider only the differentiable points on the IB curve (a more thorough analysis would look at the superderivatives of F ).

Since F is concave, however, it must be differentiable almost everywhere (Rockafellar, 2015, Thm 25.5).

Finally, note that the IB curve is the Pareto front of the multi-objective optimization problem, DISPLAYFORM0 IB that allows us to explore a nonstrictly-concave Pareto front.

However, it is not the only possible modification, and other alternatives may be considered.

For a full treatment of multi-objective optimization, see BID20 .

It is often implicitly or explicitly assumed that IB optimal variables provide "useful" or "interesting" representations of the relevant information in X about Y (see also discussion and proposed criteria in BID4 ).

While concepts like usefulness and interestingness are subjective, the intuition can be illustrated using the following example.

Consider the ImageNet task BID11 , which labels images into 1000 classes, such as border collie, golden retriever, coffeepot, teapot, etc.

It is natural to expect that as one explores the IB curve for the ImageNet dataset, one will identify useful compressed representations of the space of images.

Such useful representations might, for instance, specify hard-clusterings that merge together inputs belonging to perceptually similar classes, such as border collie and golden retriever (but not border collie and teapot).

Here we show that such intuitions will not generally hold when Y is a deterministic function of X.Recall the analysis in Section 3, where Eq. (6) defines the manifold of bottleneck variables T α parameterized by α ∈ [0, 1].

The manifold of T α , which spans the entire increasing portion of the IB curve, represents a mixture of two trivial solutions: a constant mapping to 0, and an exact copy of Y .

Although the bottleneck variables on this manifold are IB-optimal, they do not offer interesting or useful representations of the input data.

The compression offered by T α arises by "forgetting" the input with some probability 1 − α, rather than performing any kind of useful hard-clustering, while the prediction comes from full knowledge of the function mapping inputs to outputs, f : X → Y.To summarize, when Y is a deterministic function of X, the fact that a variable T is on the IB curve does not necessarily imply that T is an interesting compressed representation of X. At the same time, we do not claim that Eq. (6) provides unique solutions.

There may also be IB-optimal variables that do compress X in interesting and useful ways (however such notions may be formalized).

Nonetheless, when Y is a deterministic function of X, for any "interesting" T , there will also be an "uninteresting" T α that achieves the same compression I(X; T ) and prediction I(Y ; T ) values, and IB does not distinguish between the two.

Therefore, identifying useful compressed representations must generally require the use of quality functions other than just IB.

6 ISSUE 3: NO TRADE-OFF AMONG DIFFERENT NEURAL NETWORK LAYERS So far we've analyzed IB in terms of a single intermediate representation T (e.g., a single hidden layer in a neural network).

Recent research in machine learning, however, has focused on "deep" neural networks, with multiple successive hidden layers.

What is the relationship between the compression and prediction achieved by these different layers?

Recently, BID24 suggested that due to SGD dynamics, different layers will explore a strict trade-off between compression and prediction: early layers will sacrifice compression (high I(X; T )) for good prediction (high I(Y ; T )), while latter layers will sacrifice prediction (low I(Y ; T )) for good compression (low I(X; T )).

The authors demonstrated a strict trade-off using an artificial classification dataset, in which Y was defined to be a noisy function of X (their Eq. 10 and Fig. 6 ).

As we show, however, this outcome cannot generally hold when Y is a deterministic function of X.Recall that we consider IB in terms of the empirical distribution of inputs, hidden layers, and outputs given the training data (Section 2).

Suppose that a classifier achieves 0 probability of error (i.e., classifies every input correctly) on training data.

This event, which can only occur when Y is a deterministic function of X, is in fact commonly observed in real-world deep neural networks (Zhang et al., 2017) .

In such cases, we show that while latter layers may have better compression than earlier layers, they cannot have worse prediction than earlier layers.

Therefore, different layers can only demonstrate a weak trade-off between compression and prediction. (The same argument holds if the neural network achieves 0 probability of error on held-out testing data, as long as the information-theoretic measures are evaluated on the same held-out data distribution.)Consider a neural network with k hidden layers, where each successive layer is a (possibly stochastic) function of the preceding one, and let T 1 , T 2 , . . .

, T k indicate the activity of layer 1, 2, . . . , k. Given the predicted distribution of outputs, q θ (Ŷ =ŷ|T k = t k ), one can make a "point prediction"Ỹ , e.g., by choosing the class with the highest predicted probability,Ỹ := arg maxŷ q θ (ŷ|T k ).

Given that the true output Y is a function of X, the above architecture obeys the Markov condition DISPLAYFORM0 By the DPI, for any i < j we have the inequalities DISPLAYFORM1 DISPLAYFORM2 Applying Fano's inequality (Cover & Thomas, 2012, p. 39) to the chain Y − T k −Ỹ gives DISPLAYFORM3 where |Y| is the number of output classes, P e = Pr(Y =Ỹ ) is the probability of error, and H(x) := −x log x − (1 − x) log(1 − x) is the binary entropy function.

Given our assumption that P e = 0, Eq. (12) implies that H(Y |T k ) = 0 and thus DISPLAYFORM4 Combining with Eq. (10) and DISPLAYFORM5 The above argument neither proves nor disproves the proposal in BID24 ) that SGD dynamics favor hidden layer mappings that provide compressed representations of the input.

Instead, our point is that for classifiers that achieve 0 prediction error, a strict compression/prediction trade-off is not possible.

Even so, by Eq. (11) it is possible that latter layers have more compression than earlier layers while achieving the same prediction level (i.e., a weak trade-off).

In terms of FIG2 , this means that different layers will be on the flat part of the IB curve.

Note that some neural network architectures do not have a simple layer structure, in which each layer's activity is a stochastic function of the previous layer's, but rather allow information to flow in parallel across different channels BID27 or to skip across some layers BID15 .

The analysis in this section can still apply to such cases, as long as the different T 1 , . . .

, T k are chosen to correspond to groups of neurons that satisfy the Markov condition in Eq. (9).

We demonstrate the three caveats using the MNIST dataset of hand-written digits.

To do so, we use the "nonlinear IB" method , which uses gradient-descent-based training to minimize cross-entropy loss plus a differentiable non-parametric estimate of I(X; T ) .

We use this technique to maximize a lower bound on the IB Lagrangian, as explained in , as well as a lower bound on the squared-IB functional.

In our architecture, the bottleneck variable, T ∈ R 2 , corresponds to the activity of a hidden layer with two hidden units.

Using a two-dimensional bottleneck variable facilitates easy visual analysis of its activity.

The map from input X to variable T has the form T = a θ (X)+Z, where Z ∼ N (0, I) is noise, and a θ is a deterministic function implemented using three fully-connected layers: two layers with 800 ReLU units each, and a third layer with two linear units.

Note that the stochasticity in the mapping from X to T makes our mutual information term, I(X; T ), well-defined and finite.

The decoding map, q θ (ŷ|t), uses a fully-connected layer with 800 ReLU units, followed by an Like other practical general-purpose IB methods, "nonlinear IB" is not guaranteed to find globallyoptimal bottleneck variables due to factors like: (a) difficulty of optimizing the non-convex IB objective; (b) error in estimating I(X; T ); (c) limited model class of T (i.e., T must be expressible in the form of T = a θ (X) + Z); (d) mismatch between actual decoder q θ (Ŷ |T ) and optimal decoder p θ (Y |T ) (see Section 2); and (e) stochasticity of training due to SGD.

Nonetheless, in practice, the solutions discovered by nonlinear IB were very close to IB-optimal, and are sufficient to demonstrate the three caveats discussed in the previous sections.

We first demonstrate that the IB curve cannot be explored by maximizing the IB Lagrangian, but can be explored by maximizing the squared-IB functional, thus supporting the arguments in Section 4.

FIG7 shows the theoretical IB curve for the MNIST dataset, as well as the IB curve empirically recovered by maximizing these two functionals (see also FIG8 for more details).By optimizing the IB Lagrangian FIG7 , we are only able to find three IB-optimal points: (1) Maximum prediction accuracy and no compression, I(Y ; T ) = H(Y ), I(X; T ) ≈ 8 nats; (2) Maximum compression possible at maximum prediction, I(Y ; T ) = I(X; T ) = H(Y ); (3) Total compression and zero prediction, I(X; T ) = I(Y ; T ) = 0.Note that the identified solutions are all very close to the theoretically-predicted IB-curve.

However, the switch between the 2nd regime (maximal compression possible at maximum prediction) and 3rd regime (total compression and zero prediction) in practice happens at β ≈ 0.45.

This is different from the theoretical prediction, which states that this switch should occur at β = 1.0.

The deviation from the theoretical prediction likely arises due to various practical details of our optimization procedure, as mentioned above.

The switch from the 1st regime (no compression) to the 2nd regime happened as soon as β > 0, as predicted theoretically.

In contrast to the IB Lagrangian, by optimizing the squared-IB functional FIG7 , we discover solutions located along different points on the IB curve for different values of β.

Additional insight is provided by visualizing the bottleneck variable T ∈ R 2 (i.e., hidden layer activity) for both experiments.

This is shown for different β in the scatter plot insets in FIG7 .

As expected, the IB Lagrangian experiments displayed three types of bottleneck variables: noncompressed variables (regime 1), compressed variables where each of the 10 classes is represented by its own compact cluster (regime 2), and a trivial solution where all activity is collapsed to a single cluster (regime 3).

For the squared-IB functional, a different behavior was observed: As β increases, multiple classes become clustered together and the total number of clusters decreased.

Thus, nonlinear IB with the squared-IB functional learned to group X into a varying number of clusters, in this way exploring the full trade-off between compression and prediction.

Issue 2: All points on IB curve have "uninteresting" solutions By maximizing the squared-IB functional, we could find (nearly) IB-optimal solutions along different points of the IB curve.

Here we show that such solutions do not provide particularly useful representations of the input data, supporting the arguments made in Section 5.Note that "stochastic mixture"-type solutions, in particular the family T α (Eq. (6)), are not in our model class, since they cannot be expressed in the form T = a θ (X) + Z. Instead, our implementation favors "hard-clusterings" in which all inputs belonging to a given output class are mapped to a compact, well-separated cluster in the activation space of the hidden layer (note that inputs belonging to multiple output classes may be mapped to a single cluster).

For instance, FIG7 , shows solutions with 10 clusters (β = 0.05), 6 clusters (β = 0.2), 3 clusters (β = 0.5), and 1 cluster (β = 2.0).

Interestingly, such hard-clusterings are characteristic of optimal solutions for deterministic IB (dIB), as discussed in Appendix B. At the same time, in our results, the classes are not clustered in any particularly meaningful or useful way, and clusters contain different combinations of classes for different solutions.

For instance, the solution shown for squared-IB functional with β = 0.5 has 3 clusters, one of which contains the classes {0, 2, 3, 4, 5, 6, 8, 9}, another contains the class 1, and the last contains the class 7.

However, in other runs for the same β value, different clusterings of the classes arose.

Moreover, because the different classes appear with close-to-uniform frequency in the MNIST dataset, any solution that groups the 10 classes into 3 clusters of size {8, 1, 1} will achieve similar values of I(X; T ), I(Y ; T ) as the solution shown for β = 0.5.

For both types of experiments, runs with β = 0 minimize cross-entropy loss only, without any regularization term that favors compression.

(Such runs are examples of "vanilla" supervised learning, though with stochasticity in the mapping between the input and the 2-node hidden layer.)

These runs achieved nearly-perfect prediction, so I(Y ; T ) ≈ H(Y ).

However, as shown in the scatter plots in FIG7 , these hidden layer activations fell into spread-out clusters, rather than point-like clusters seen for β > 0.

This shows that hidden layer activity was not compressed, in that it retained information about X that was irrelevant for predicting the class Y , and fell onto the flat part of the IB curve.

Recall that our neural network architecture has three hidden layers before T , and one hidden layer after it.

Due to the DPI inequalities Eqs. (10) and (11), the earlier hidden layers must have less compression than T , while the latter hidden layer must have more compression than T .

At the same time, β = 0 runs achieve nearly 0 probability of error on the training dataset (results not shown), meaning that all layers must achieve I(Y ; T ) ≈ H(Y ), the maximum possible.

Thus, for β = 0 runs, as in regular supervised learning, the activity of the all hidden layers is located on the flat part of the IB curve, demonstrating a lack of a strict trade-off between prediction and compression.

The information bottleneck principle has attracted a great deal of attention in various fields, including information theory, cognitive science, and machine learning, particularly in the context of classification using neural networks.

In this work, we showed that in any scenario where Y is a deterministic function of X -which includes many classification problems -IB demonstrates behavior that is qualitatively different from when the mapping from X to Y is stochastic.

In particular, in such cases: (1) the IB curve cannot be recovered by maximizing the IB Lagrangian I(Y ; T ) − βI(X; T ) while varying β; (2) all points on the IB curve contain "uninteresting" representations of inputs; (3) multi-layer classifiers that achieve zero probability of error cannot have a strict trade-off between prediction and compression among successive layers, contrary to a recent proposal.

Our results should not be taken to mean that the application of IB to supervised learning is without merit.

First, they do not apply to various non-deterministic classification problems where the output is stochastic.

Second, even for deterministic scenarios, one may still wish to control the amount of compression during training, e.g., to improve generalization or robustness to adversarial inputs.

In this case, however, our work shows that to achieve varying rates of compression, one should use a different objective function than the IB Lagrangian.

In Section 7, we demonstrate our results on the MNIST dataset BID19 .

This dataset contains a training set of 60,000 images and a test set of 10,000 images, each labeled according to digit.

X ∈ R 784 is defined to be a vector of pixels for a single 28 × 28 image, and Y ∈ {0, 1, ..., 9} is defined to be the class label.

Our experiments were carried out using the "nonlinear IB" method .

I(X; T ) was computed using the kernel-based mutual information upper bound and I(Y ; T ) was computed using the lower bound FORMULA3 ).

DISPLAYFORM0 The neural network was trained using the Adam algorithm (Kingma & Ba, 2014) with a mini-batch size of 128 and a learning rate of 10 −4 .

Unlike the implementation in , the same mini-batch was used to estimate the gradients of both I θ (X; T ) and the cross-entropy term.

Training was run for 200 epochs.

At the beginning of each epoch, the order of training examples was randomized.

To eliminate the effect of the local minima, for each possible value of β, we carried out 20 runs and then selected the run that achieved the best value of the objective function.

TensorFlow code is available at https://github.com/artemyk/ibcurve .Results for the MNIST dataset are shown in FIG7 and FIG8 , computed for a range of β values.

FIG8 shows results for both training and testing datasets, though the main text focuses exclusively on training data.

It can be seen that while the solutions found by IB Lagrangian jump discontinuously from the "fully clustered" solution (I(X; T ) = I(Y ; T ) = H(Y )) to the trivial solution (I(X; T ) = I(Y ; T ) = 0), solutions found by the squared-IB functional explore the trade-off in a continuous manner.

See figure captions for details.

Here we show that our analysis also applies to a recently-proposed BID26 variant of IB called deterministic IB (dIB).

dIB replaces the standard IB compression cost, I(X; T ), with the entropy of the bottleneck variable, H(T ).

This can be interpreted as operationalizing compression costs via a source-coding, rather than a channel-coding, scenario.

Formally, in dIB one is given random variables X and Y .

One then identifies bottleneck variables T that obey the Markov condition Y − X − T and maximize the dIB Lagrangian, DISPLAYFORM0 for β ∈ [0, 1], which can be considered as a relaxation of the constrained optimization problem DISPLAYFORM1 To guarantee that the compression cost is well-defined, T is typically assumed to be discretevalued BID26 .

We call F dIB the dIB curve.

Before proceeding, we note that the inequality constraint in the definition of F dIB (r) can be replaced by an equality constraint, DISPLAYFORM2 We do so by showing that F dIB (r) is monotonically increasing in r. Consider any T which maximizes I(Y ; T ) subject to the constraint H(T ) = r, and obeys the Markov condition Y − X − T .

Now imagine some random variable D which obeys the Markov condition Y − X − T − D, and define a new bottleneck variable T := (T, D) (i.e., the joint outcome of T and D).

We have DISPLAYFORM3 where we've used the chain rule for mutual information.

At the same time, D can always be chosen so that H(T ) = H(T, D) = r for any r ≥ r. Thus, we have shown that there are always random variables T that achieve at least I(Y ; T ) = max T :H(T )=r I(Y ; T ) and have H(T ) > r, meaning that F dIB (r) is monotonically increasing in r. This means the inequality constraint in Eq. (A14) can be replaced with an equality constraint.

As we will see, unlike the standard IB curve, F dIB is not necessarily concave.

Since F dIB can be defined using equality constraints, one can rewrite maximization of L β dIB as max T I(Y ; T )−βH(T ) = max r F dIB (r) − βr, the Legendre-Fenchel transform of −F dIB (r).

By properties of the LegendreFenchel transform, the optimizers of L β dIB must lie on the concave envelope of F dIB , which we indicate as F * dIB .

As in standard IB, for a discrete-valued Y we have the inequality DISPLAYFORM0 However, instead of the standard IB inequality I(Y ; T ) ≤ I(X; T ), we now employ DISPLAYFORM1 which makes use of the assumption that T is discrete-valued.

The dIB curve will have the same bounds as those shown for the standard IB curve FIG2 , except that H(T ) replaces I(X; T ) on the horizontal axis.

Now consider the case where Y is a deterministic function of X, i.e., Y = f (X).

It is easy to check that T copy := f (X) = Y achieves equality for both Eqs. (A16) and (A17), and thus lies on the dIB curve.

Since F dIB is monotonically increasing, the dIB curve is flat and achieves DISPLAYFORM2 We now consider the increasing part of the curve, DISPLAYFORM3 We call any T which is a deterministic function of Y (that is, any DISPLAYFORM4

dIB curve for Y = f(X) Figure A6 : A schematic of the dIB curve.

Dashed line is the bound DISPLAYFORM0 When Y is a deterministic function of X, the dIB curve saturate the second bound always.

Furthermore, it also saturates the first bound for any T that is a deterministic function of Y .

The qualitative shape of the resulting dIB curve is shown as thick gray line.

At the same time, the dIB curve cannot be composed entirely of hard-clustering, since -under the assumption that Y is discrete-valued -there can only be a countable number of T 's that are hard-clusterings of Y .

Thus, the dIB curve must also contain bottleneck variables that are not deterministic functions of Y , thus have H(T |Y ) > 0 and I(Y ; T ) < H(T ), and do not achieve the bound of Eq. (A17).

Geometrically-speaking, when Y is a deterministic function of X, the dIB curve must have a "step-like" structure over H(T ) ∈ [0, H(Y )], rather than increasing smoothly like the standard IB curve.

These results are shown schematically in Fig. A6 , where blue dots indicate hard clusters of Y .As mentioned, optimizers of L β dIB must lie on the concave envelope of F dIB , indicated by F * dIB .

Clearly, the step-like dIB curve that occurs when Y is a deterministic function of X is not concave, and only hard-clusterings of Y lie on its concave envelope for DISPLAYFORM1 In the analysis below, we will generally concern ourselves with optimizers which are hard-clusterings.

We now briefly consider the three issues discussed in the main text, in the context of dIB.

As usual, we assume that Y is a deterministic function of X.Issue 1: dIB Curve cannot be explored using the dIB LagrangianIn analogy to the analysis done in Section 4, we use the inequalities Eqs. (A16) and (A17) to bound the dIB Lagrangian as DISPLAYFORM0 Now consider the bottleneck variable T copy , for which L β dIB (T copy ) = (1 − β)H(Y ).

Therefore, T copy (or any one-to-one transformation of T copy ) will maximize L β dIB for all β ∈ [0, 1].

It is also straightforward to show that when β = 0, all bottleneck variables residing on the flat part of the dIB curve will simultaneously optimize the dIB Lagrangian.

Similarly, one can show that all hard-clusterings of Y , which achieve the bound of Eq. (A17), will simultaneously optimize the dIB Lagrangian for β = 1.

As before, this means that there is no one-to-one map between points on the dIB curve and optimizers of the dIB Lagrangian for different β.

As in Section 4, we propose to resolve this problem by maximizing an alternative objective function, which we call the squared-dIB functional, DISPLAYFORM1 We first demonstrate that any optimizer of the dIB Lagrangian must also be an optimizer of the squared-dIB functional.

Consider that maximization of L β dIB can be written as max T I(Y ; T ) − βH(T ) = max r F * dIB (r)

− r.

Then, for the point r, F * dIB (r) on the dIB curve to maximize L β dIB , it must have β ∈ ∂ r F * dIB (r), where ∂ r indicates the superderivative with regard to r. At the same time, for the point r, F * dIB (r) to maximize L β sq-dIB , it must satisfy 0 ∈ ∂ r F * dIB (r) − βr 2 , or after rearranging, DISPLAYFORM2 It is easy to see that if β ∈ ∂ r F * dIB (r) is satisfied, then Eq. (A19) is also satisfied under the transformation β → We now show that different hard-clusterings of Y will optimize the squared-dIB functional for different values of β, meaning that we can explore the envelope of the dIB curve by optimizing L β sq-dIB while varying β.

Formally, we show that for any given β > 0, the point DISPLAYFORM3 on the dIB curve will be a unique maximizer of L β sq-dIB for the corresponding β.

Consider the value of L β sq-dIB for any T satisfying Eq. (A20), DISPLAYFORM4 Now consider the value of of L β sq-dIB for any other T on the dIB curve which has DISPLAYFORM5 Inequality (a) comes from the assumption that DISPLAYFORM6 , meaning that any point on the dIB curve satisfying Eq. (A20) for a given β, assuming it exists, will be the unique maximizer of L β sq-dIB .

The situation is diagrammed visually in FIG10 .

Issue 2: All points on dIB curve have "uninteresting" solutionsThe family of bottleneck variables T α defined in Eq. (6), i.e., the mixture of two trivial solutions, are no longer optimal from the point of view of dIB.

However, as mentioned above, any T that is a hard-clustering of Y achieves the bound of Eq. (A17), and is thus on the dIB curve.

However, given a hard-clustering of Y , there is no reason for the clusters to obey any intuitions about semantic or perceptual similarity between grouped-together classes.

To use the ImageNet example from Section 5, there is no reason for dIB to prefer a coarse-graining with "natural" groups like {border collie, golden retriever} and {teapot, coffeepot}, rather than a coarse-graining with groups like {border collie, teapot} and {golden retriever, coffeepot}, assuming those classes are of the same size.

Distinguishing between such different clusterings requires some similarity or distortion measure between classes, which is not provided by the standard information theoretic measures.

In Section 6, we showed that for a neural network with many hidden layers and zero probability of error, the activity of the different layers will lie along the flat part of the standard IB curve, where there is no strict trade-off between compression I(X; T ) and prediction I(Y ; T ).

Note that any bottleneck variable that lies along the flat part of a standard IB curve will have I(X; T ) ≥ H(Y ) and I(Y ; T ) = H(Y ).

Using the standard information-theoretic inequality H(T ) ≥ I(X; T ), the same bottleneck variable must therefore have H(T ) ≥ H(Y ) and I(Y ; T ) = H(Y ), thus also lying on the flat part of the dIB curve.

Thus, for a neural network with many hidden layers and zero probability of error, the activity of the different layers will also lie along the flat part of the dIB curve, and not demonstrate any strict trade-off between compression H(T ) and prediction I(Y ; T ).

In this Appendix, we show that when Y is a small perturbation away from being a deterministic function of X, the three caveats discussed in the main text persist in an approximate manner.

To derive our results, we first prove several useful theorems.

In our proofs, we make central use of Thm.

17.3.3 from Cover & BID9 , which states that, for two distributions a and b over the same finite set of outcomes Y, if the 1 distance is bounded as |a − b| 1 ≤ ≤ 1 2 , then |H(a) − H(b)| ≤ − log |Y| .

We will also use the weaker bound |H(a) − H(b)| ≤ log |Y|, which is based on the maximum and minimum entropy for any distribution over outcomes Y. Theorem 1.

Let Z be a random variable (continuous or discrete), and Y a random variable with a finite set of outcomes Y. Consider two joint distributions over Z and Y , p ZY andp ZY , which have the same marginal over Z, p(z) =p(z), and obey DISPLAYFORM0 Proof.

For this proof, we will assume that Z is a continuous-valued.

In case it is discrete-valued, the below proof applies after replacing all integrals over the outcomes of Z with summations.

Letˆ := |p ZY −p ZY | 1 = y |p ZY (z, y) −p ZY (z, y)| dz be the actual 1 distance between p ZY andp ZY , and note thatˆ ≤ by assumption.

Without loss of generality, we assume thatˆ > 0.

Then, define the following probability distribution, DISPLAYFORM1 Note that q(z) integrates to 1 and is absolutely continuous with respect to p(z) (p(z) = 0 implies q(z) = 0), and that DISPLAYFORM2 where the first case follows from (Cover & Thomas, 2012, Thm. 17.3.3) , and the second case by considering minimum vs. maximum possible entropy values.

We now bound the difference in conditional entropies, using [·] to indicate the Iverson bracket, DISPLAYFORM3 Published as a conference paper at ICLR 2019We upper bound the first integral in Eq. (A21) as DISPLAYFORM4 where in Eq. (A22) we dropped the Iverson bracket, in Eq. (A23) we used the definition of KL divergence, and then its non-negativity.

To upper bound the second integral in Eq. (A21), observe that DISPLAYFORM5 Combining Eqs. (A21), (A23) and (A24) gives DISPLAYFORM6 The theorem follows by noting that −ˆ logˆ is monotonically increasing forˆ ≤ In the following theorems, we use the notation I q (Z; Y ) to indicate mutual information between Z and Y evaluated under some joint distribution q over Z and Y .

Theorem 2.

Let Z be a random variable (continuous or discrete), and Y a random variable with a finite set of outcomes.

Consider two joint distribution over X × Y , p XY andp XY , which have the same marginal over Z, p(z) =p(z), and obey DISPLAYFORM7 Proof.

We first bound the 1 distance between p Y andp Y , Thomas, 2012, Thm.

17.3.3) , DISPLAYFORM8 DISPLAYFORM9 We now bound the magnitude of the difference of mutual informations as DISPLAYFORM10 where in the last line we've used Eq. (A25) and Theorem 1.

DISPLAYFORM11 Proof.

For any compression level r > 0, let T andT be optimal bottleneck variables for p XY and p XY respectively, DISPLAYFORM12 Let T be defined by the stochastic map q(t|x), and note that p T (t) = p(x)q(t|x) dx = p(x)q(t|x) dx =p T (t).

Consider two joint distributions over T and Y , DISPLAYFORM13 and observe that DISPLAYFORM14 DISPLAYFORM15 The same argument can be repeated while exchanging the roles of T andT (thus defining the joint distributions pT Y andpT Y , etc.).

This gives the inequality Ip(Y ;T ) + 2 log |Y| 2 ≤ I p (Y ; T ) .The theorem follows by combining Eqs. (A26) and (A27), and using the relations I p (Y ; T ) = F (r), Ip(Y ; T ) =F (r).It is important to emphasize that having a small 1 distance between the joint distributions p XY and p XY does not imply that the conditional distributions p Y |X=x andp Y |X=x have to be close for all x. In fact, the conditional distributions can be arbitrarily different for some x, as long as such x do not have much probability under p(x).We use these theorems to demonstrate how, if the joint distribution of X and Y is -close to having Y be a deterministic function of X, then the three caveats discussed in the main text all apply in an approximate manner, meaning that can only be avoided up to order O(− log ).

Issue 1: IB curve cannot be explored using the IB LagrangianIn the main text, we demonstrated that when Y is a deterministic function of X, the single point H(Y ), H(Y ) on the information plane optimizes the IB Lagrangian for all β ∈ [0, 1].

We now consider the case where the joint distribution of X and Y is -close to having Y be a deterministic function of X. The next theorem shows that in this case, optimizers of the IB Lagrangian must still be close to H(Y ), H(Y ) .

Formally, for any fixed β ∈ (0, 1), the maximum possible distance from H(Y ), H(Y ) scales as O(− log ), though the scaling constant increases as β approaches 0 or 1.

This implies that for small , it will difficult to find solutions substantially different from H(Y ), H(Y ) , as one would have to search using β very close to 0 or very close to 1.

Theorem 4.

Let X be a random variable (continuous or discrete), and Y a random variable with a finite set of outcomes Y. Letp XY be a joint distribution over X and Y under which Y = f (X).

Let p XY be a joint distribution over X and Y which has the same marginal over X asp XY , p(x) = p(x), and obeys |p XY −p XY | 1 ≤ ≤ 1 2 .

Then, for any β ∈ (0, 1) and T which maximizes the p XY IB Lagrangian I p (Y ; T ) − βI p (X; T ), DISPLAYFORM0 where γ := −3 log + 5 log |Y|.

Let F andF indicate the IB curves for p XY andp XY , respectively, defined as in Eq. (1).

Becausẽ p XY is deterministic, the IB curveF is the piecewise linear functionF (x) = min{x, H(p(Y ))}.

Given Eq. (A30), this means that DISPLAYFORM1 We now to bound F (H(p(Y ))) as DISPLAYFORM2 where in the first line we've used Theorem 3, and in the second line we used Eq. (A31).

Now consider T , the bottleneck variable that optimizes the p XY IB Lagrangian for some β ∈ (0, 1).

By concavity of the IB curve F , all points on the IB curve must fall below the line with slope β that passes through the point I p (X; T ), I p (Y ; T ) on the information plane, including the point H(p(Y )), F (H(p(Y )) .

Formally, we write this condition as DISPLAYFORM3 where the second inequality uses Eq. ( Issue 3: Prediction/compression trade-off can be most of order O(− log )In the main text, we consider a neural network consisting of k layers, and show that when Y is a deterministic function of X, there can be no strict prediction/compression trade-off between the different layers.

We now consider the case where the joint distribution of X and Y is -close to having Y be a deterministic function of X.As before, we let X be a (continuous or discrete) random variable, and Y a random variable with a finite set of outcomes Y. Letp XY be a joint distribution over X and Y under which Y = f (X).

Let p XY be a joint distribution over X and Y which has the same marginal over X asp XY , p(x) =p(x), and obeys |p XY −p XY | 1 ≤ ≤ 1 2 .

We assume that for each input X, our classifier outputs the predictionỸ = f (X) -that is, it predicts according to the deterministic input-output mapping f .

For small , this is an optimal prediction strategy in terms of P e , the probability of error.

We write P e for this classifier as We now restate Fano's inequality (Eq. FORMULA0 ) for the activity of the last layer, T k , as H(Y |T k ) ≤ H(P e ) + P e log(|Y| − 1) ≤ −2P e log 2P e |Y| = − log |Y| ,where the second inequality is found in Zhang (2007).As before, we note that latter layers may compress input information more than earlier ones, in that I(T k ; X) may be smaller than I(T 1 ; X).

Moreover, when Y is not exactly a deterministic function of X, then in principle a strict prediction/compression trade-off between different layers is possible (i.e., I(T i ; Y ) can decrease for latter layers).

However, if the joint distribution of X and Y is -close to having Y = f (X), then the magnitude this trade-off is limited in size, DISPLAYFORM4 where we use non-negativity of conditional entropy and Eq. (A40).

<|TLDR|>

@highlight

Information bottleneck behaves in surprising ways whenever the output is a deterministic function of the input.

@highlight

Argues that most real classification problems show such a deterministic relation between the class labels and the inputs X and explores several issues that result from such pathologies.

@highlight

Explores issues that arise when applying information bottlenext concepts to deterministic supervised learning models

@highlight

The authors clarify several counter-intuitive behaviors of the information bottleneck method for supervised learning of a deterministic rule.