In practice it is often found that large over-parameterized neural networks generalize better than their smaller counterparts, an observation that appears to conflict with classical notions of function complexity, which typically favor smaller models.

In this work, we investigate this tension between complexity and generalization through an extensive empirical exploration of two natural metrics of complexity related to sensitivity to input perturbations.

Our experiments survey thousands of models with different architectures, optimizers, and other hyper-parameters, as well as four different image classification datasets.



We find that trained neural networks are more robust to input perturbations in the vicinity of the training data manifold, as measured by the input-output Jacobian of the network, and that this correlates well with generalization.

We further establish that factors associated with poor generalization -- such as full-batch training or using random labels -- correspond to higher sensitivity, while factors associated with good generalization  -- such as data augmentation and ReLU non-linearities -- give rise to more robust functions.

Finally, we demonstrate how the input-output Jacobian norm can be predictive of generalization at the level of individual test points.

The empirical success of deep learning has thus far eluded interpretation through existing lenses of computational complexity BID2 , numerical optimization BID4 BID8 BID5 and classical statistical learning theory (Zhang et al., 2016) : neural networks are highly non-convex models with extreme capacity that train fast and generalize well.

In fact, not only do large networks demonstrate good test performance, but larger networks often generalize better, counter to what would be expected from classical measures, such as VC dimension.

This phenomenon has been observed in targeted experiments BID29 , historical trends of Deep Learning competitions BID3 , and in the course of this work ( Figure 1 ).This observation is at odds with Occam's razor, the principle of parsimony, as applied to the intuitive notion of function complexity (see §A.2 for extended discussion).

One resolution of the apparent contradiction is to examine complexity of functions in conjunction with the input domain.

f (x) = x 3 sin(x) may seem decisively more complex than g(x) = x. But restrained to a narrow input domain of [−0.01, 0 .01]

they appear differently: g remains a linear function of the input, while f (x) = O x 4 resembles a constant 0.

In this work we find that such intuition applies to neural networks, that behave very differently close to the data manifold than away from it ( §4.1).We therefore analyze the complexity of models through their capacity to distinguish different inputs in the neighborhood of datapoints, or, in other words, their sensitivity.

We study two simple metrics presented in §3 and find that one of them, the norm of the input-output Jacobian, correlates with generalization in a very wide variety of scenarios.

Train loss Figure 1 : 2160 networks trained to 100% training accuracy on CIFAR10 (see §A.5.5 for experimental details).

Left: while increasing capacity of the model allows for overfitting (top), very few models do, and a model with the maximum parameter count yields the best generalization (bottom right).

Right: train loss does not correlate well with generalization, and the best model (minimum along the y-axis) has training loss many orders of magnitude higher than models that generalize worse (left).

This observation rules out underfitting as the reason for poor generalization in low-capacity models.

See BID29 for similar findings in the case of achievable 0 training loss.

This work considers sensitivity only in the context of image classification tasks.

We interpret the observed correlation with generalization as an expression of a universal prior on (natural) image classification functions that favor robustness (see §A.2 for details).

While we expect a similar prior to exist in many other perceptual settings, care should be taken when extrapolating our findings to tasks where such a prior may not be justified (e.g. weather forecasting).

We first define sensitivity metrics for fully-connected neural networks in §3.

We then relate them to generalization through a sequence of experiments of increasing level of nuance:• In §4.1 we begin by comparing the sensitivity of trained neural networks on and off the training data manifold, i.e. in the regions of best and typical (over the whole input space) generalization.• In §4.2 we compare sensitivity of identical trained networks that differ in a single hyperparameter which is important for generalization.• Further, §4.3 associates sensitivity and generalization in an unrestricted manner, i.e. comparing networks of a wide variety of hyper-parameters such as width, depth, non-linearity, weight initialization, optimizer, learning rate and batch size.• Finally, §4.4 explores how predictive sensitivity (as measured by the Jacobian norm) is for individual test points.

The novelty of this work can be summarized as follows:• Study of the behavior of trained neural networks on and off the data manifold through sensitivity metrics ( §4.1).• Evaluation of sensitivity metrics on trained neural networks in a very large-scale experimental setting and finding that they correlate with generalization ( §4. 2, §4.3, §4.4) .

§2 puts our work in context of related research studying complexity, generalization, or sensitivity metrics similar to ours.

We analyze complexity of fully-connected neural networks for the purpose of model comparison through the following sensitivity measures (see §3 for details):• estimating the number of linear regions a network splits the input space into;• measuring the norm of the input-output Jacobian within such regions.

A few prior works have examined measures related to the ones we consider.

In particular, BID33 ; BID24 ; have investigated the expressive power of fully-connected neural networks built out of piecewise-linear activation functions.

Such functions are themselves piecewise-linear over their input domain, so that the number of linear regions into which input space is divided is one measure of how nonlinear the function is.

A function with many linear regions has the capacity to build complex, flexible decision boundaries.

It was argued in BID33 BID24 that an upper bound to the number of linear regions scales exponentially with depth but polynomially in width, and a specific construction was examined. derived a tight analytic bound and considered the number of linear regions for generic networks with random weights, as would be appropriate, for instance, at initialization.

However, the evolution of this measure after training has not been investigated before.

We examine a related measure, the number of hidden unit transitions along one-dimensional trajectories in input space, for trained networks.

Further motivation for this measure is discussed in §3.Another perspective on function complexity can be gained by studying their robustness to perturbations to the input.

Indeed, BID36 demonstrate on a toy problem how complexity as measured by the number of parameters may be of limited utility for model selection, while measuring the output variation allows the invocation of Occam's razor.

In this work we apply related ideas to a large-scale practical context of neural networks with up to a billion free parameters ( §4.2, §4.3) and discuss potential ways in which sensitivity permits the application of Occam's razor to neural networks ( §A.2).

Sokolic et al. (2017) provide theoretical support for the relevance of robustness, as measured by the input-output Jacobian, to generalization.

They derive bounds for the generalization gap in terms of the Jacobian norm within the framework of algorithmic robustness (Xu & Mannor, 2012) .

Our results provide empirical support for their conclusions through an extensive number of experiments.

Several other recent papers have also focused on deriving tight generalization bounds for neural networks BID1 BID6 BID31 .

We do not propose theoretical bounds in this paper but establish a correlation between our metrics and generalization in a substantially larger experimental setting than undertaken in prior works.

In the context of regularization, increasing robustness to perturbations is a widely-used strategy: data augmentation, noise injection BID15 , weight decay BID19 , and max-pooling all indirectly reduce sensitivity of the model to perturbations, while BID37 Sokolic et al. (2017) explicitly penalize the Frobenius norm of the Jacobian in the training objective.

In this work we relate several of the above mentioned regularizing techniques to sensitivity, demonstrating through extensive experiments that improved generalization is consistently coupled with better robustness as measured by a single metric, the input-output Jacobian norm ( §4.2).

While some of these findings confirm common-sense expectations (random labels increase sensitivity, Figure 4 , top row), others challenge our intuition of what makes a neural network robust (ReLU-networks, with unbounded activations, tend to be more robust than saturating HardSigmoid-networks, Figure 4 , third row).

One of our findings demonstrates an inductive bias towards robustness in stochastic mini-batch optimization compared to full-batch training ( Figure 4 , bottom row).

Interpreting this regularizing effect in terms of some measure of sensitivity, such as curvature, is not new BID13 BID16 ), yet we provide a novel perspective by relating it to reduced sensitivity to inputs instead of parameters.

The inductive bias of SGD ("implicit regularization") has been previously studied in BID29 , where it was shown through rigorous experiments how increasing the width of a singlehidden-layer network improves generalization, and an analogy with matrix factorization was drawn to motivate constraining the norm of the weights instead of their count.

BID30 further explored several weight-norm based measures of complexity that do not scale with the size of the model.

One of our measures, the Frobenius norm of the Jacobian is of similar nature (since the Jacobian matrix size is determined by the task and not by a particular network architecture).

However, this particular metric was not considered, and, to the best of our knowledge, we are the first to evaluate it in a large-scale setting (e.g. our networks are up to 65 layers deep and up to 2 16 units wide).

Sensitivity to inputs has attracted a lot of interest in the context of adversarial examples (Szegedy et al., 2013) .

Several attacks locate points of poor generalization in the directions of high sensitivity of the network BID32 BID25 , while certain defences regularize the model by penalizing sensitivity BID10 or employing decaying (hence more robust) non-linearities BID20 .

However, work on adversarial examples relates highly specific perturbations to a similarly specific kind of generalization (i.e. performance on a very small, adversarial subset of the data manifold), while this paper analyzes average-case sensitivity ( §3) and typical generalization.

We propose two simple measures of sensitivity for a fully-connected neural network (without biases) f : R d → R n with respect to its input x ∈ R d (the output being unnormalized logits of the n classes).

Assume f employs a piecewise-linear activation function, like ReLU.

Then f itself, as a composition of linear and piecewise-linear functions, is a piecewise-linear map, splitting the input space R d into disjoint regions, implementing a single affine mapping on each.

Then we can measure two aspects of sensitivity by answering 1.

How does the output of the network change as the input is perturbed within the linear region?2.

How likely is the linear region to change in response to change in the input?We quantify these qualities as follows:1.

For a local sensitivity measure we adopt the Frobenius norm of the class probabilities Jacobian DISPLAYFORM0 , where f σ = σ • f with σ being the softmax function BID39 .

Given points of interest x test , we estimate the sensitivity of the function in those regions with the average Jacobian norm: DISPLAYFORM1 that we will further refer to as simply "Jacobian norm".

Note that this does not require the labels for x test .Interpretation.

The Frobenius norm J(x) F = ij J ij (x) 2 estimates the averagecase sensitivity of f σ around x. Indeed, consider an infinitesimal Gaussian perturbation BID39 The norm of the Jacobian with respect to logits ∂f (x) /∂x T experimentally turned out less predictive of test performance (not shown).

See §A.3 for discussion of why the softmax Jacobian is related to generalization.

∆x ∼ N (0, I): the expected magnitude of the output change is then DISPLAYFORM2 2.

To detect a change in linear region (further called a "transition"), we need to be able to identify it.

We do this analogously to .

For a network with piecewiselinear activations, we can, given an input x, assign a code to each neuron in the network f , that identifies the linear region of the pre-activation of that neuron.

E.g. each ReLU unit will have 0 or 1 assigned to it if the pre-activation value is less or greater than 0 respectively.

Similarly, a ReLU6 unit (see definition in §A.4) will have a code of 0, 1, or 2 assigned, since it has 3 linear regions 2 .

Then, a concatenation of codes of all neurons in the network (denoted by c(x)) uniquely identifies the linear region of the input x (see §A.1.1 for discussion of edge cases).

Given this encoding scheme, we can detect a transition by detecting a change in the code.

We then sample k equidistant points z 0 , . . .

, z k−1 on a closed one-dimensional trajectory T (x) (generated from a data point x and lying close to the data manifold; see below for details) and count transitions t(x) along it to quantify the number of linear regions: DISPLAYFORM3 where the norm of the directional derivative ∂c(z)/∂ (dz) 1 amounts to a Dirac delta function at each transition (see §A.1.2 for further details).By sampling multiple such trajectories around different points, we estimate the sensitivity metric: DISPLAYFORM4 that we will further refer to as simply "transitions" or "number of transitions." To assure the sampling trajectory T (x test ) is close to the data manifold (since this is the region of interest), we construct it through horizontal translations of the image x test in pixel space (Figure App.7, right) .

We similarly augment our training data with horizontal and vertical translations in the corresponding experiments ( Figure 4 , second row).

As earlier, this metric does not require knowing the label of x test .

Interpretation.

We can draw a qualitative parallel between the number of transitions and curvature of the function.

One measure of curvature of a function f is the total norm of the directional derivative of its first derivative f along a path: DISPLAYFORM5 A piecewise-linear function f has a constant first derivative f everywhere except for the transition boundaries.

Therefore, for a sufficiently large k, curvature can be expressed as DISPLAYFORM6 where z 0 , . . .

, z k−1 are equidistant samples on T (x).

This sum is similar to t(x) as defined in Equation 1, but quantifies the amount of change in between two linear regions in a nonbinary way.

However, estimating it on a densely sampled trajectory is a computationallyintensive task, which is one reason we instead count transitions.

As such, on a qualitative level, the two metrics (Jacobian norm and number of transitions) track the first and second order terms of the Taylor expansion of the function.

Above we have defined two sensitivity metrics to describe the learned function around the data, on average.

In §4.1 we analyze these measures on and off the data manifold by simply measuring them along circular trajectories in input space that intersect the data manifold at certain points, but generally lie away from it (Figure 2 , left).

In the following subsections ( §4.2, §4.3) each study analyzes performance of a large number (usually thousands) of fully-connected neural networks having different hyper-parameters and optimization procedures.

Except where specified, we include only models which achieve 100% training accuracy.

This allows us to study generalization disentangled from properties like expressivity and trainability, which are outside the scope of this work.

In order to efficiently evaluate the compute-intensive metrics ( §3) in a very wide range of hyperparameters settings (see e.g. §A.5.5) we only consider fully-connected networks.

Extending the investigation to more complex architectures is left for future work.

We analyze the behavior of a trained neural network near and away from training data.

We do this by comparing sensitivity of the function along 3 types of trajectories:1.

A random ellipse.

This trajectory is extremely unlikely to pass anywhere near the real data, and indicates how the function behaves in random locations of the input space that it never encountered during training.2.

An ellipse passing through three training points of different class (Figure 2 , left).

This trajectory does pass through the three data points, but in between it traverses images that are linear combinations of different-class images, and are expected to lie outside of the natural image space.

Sensitivity of the function along this trajectory allows comparison of its behavior on and off the data manifold, as it approaches and moves away from the three anchor points.3.

An ellipse through three training points of the same class.

This trajectory is similar to the previous one, but, given the dataset used in the experiment (MNIST), is expected to traverse overall closer to the data manifold, since linear combinations of the same digit are more likely to resemble a realistic image.

Comparing transition density along this trajectory to the one through points of different classes allows further assessment of how sensitivity changes in response to approaching the data manifold.

We find that, according to both the Jacobian norm and transitions metrics, functions exhibit much more robust behavior around the training data ( Figure 2 , center and right).

We further visualize this effect in 2D in Figure 3 , where we plot the transition boundaries of the last (pre-logit) layer of a neural network before and after training.

After training we observe that training points lie in regions of low transition density.

The observed contrast between the neural network behavior near and away from data further strengthens the empirical connection we draw between sensitivity and generalization in §4.2, §4.3 and §4.4; it also confirms that, as mentioned in §1, if a certain quality of a function is to be used for model comparison, input domain should always be accounted for.

In §4.1 we established that neural networks implement more robust functions in the vicinity of the training data manifold than away from it.

We now consider the more practical context of model selection.

Given two perfectly trained neural networks, does the model with better generalization implement a less sensitive function?

Figure 2 : A 100%-accurate (on training data) MNIST network implements a function that is much more stable near training data than away from it.

Left: depiction of a hypothetical circular trajectory in input space passing through three digits of different classes, highlighting the training point locations (π/3, π, 5π/3).

Center: Jacobian norm as the input traverses an elliptical trajectory.

Sensitivity drops significantly in the vicinity of training data while remaining uniform along random ellipses.

Right: transition density behaves analogously.

According to both metrics, as the input moves between points of different classes, the function becomes less stable than when it moves between points of the same class.

This is consistent with the intuition that linear combinations of different digits lie further from the data manifold than those of same-class digits (which need not hold for more complex datasets).

See §A.5.2 for experimental details.

After Training Figure 3 : Transition boundaries of the last (pre-logits) layer over a 2-dimensional slice through the input space defined by 3 training points (indicated by inset squares).

Left: boundaries before training.

Right: after training, transition boundaries become highly non-isotropic, with training points lying in regions of lower transition density.

See §A.5.3 for experimental details.

We study approaches in the machine learning community that are commonly believed to influence generalization ( Figure 4 , top to bottom):• random labels;• data augmentation;• ReLUs;• full-batch training.

We find that in each case, the change in generalization is coupled with the respective change in sensitivity (i.e. lower sensitivity corresponds to smaller generalization gap) as measured by the Jacobian norm (and almost always for the transitions metric).

Figure 4 : Improvement in generalization (left column) due to using correct labels, data augmentation, ReLUs, mini-batch optimization (top to bottom) is consistently coupled with reduced sensitivity as measured by the Jacobian norm (center column).

Transitions (right column) correlate with generalization in all considered scenarios except for comparing optimizers (bottom right).

Each point on the plot corresponds to two neural networks that share all hyper-parameters and the same optimization procedure, but differ in a certain property as indicated by axes titles.

The coordinates along each axis reflect the values of the quantity in the title of the plot in the respective setting (i.e. with true or random labels).

All networks have reached 100% training accuracy on CIFAR10 in both settings (except for the data-augmentation study, second row; see §A.5.4 for details).

See §A.5.5 for experimental details ( §A.5.4 for the data-augmentation study) and §4.2.1 for plot interpretation.

In Figure 4 , for many possible hyper-parameter configurations, we train two models that share all parameters and optimization procedure, but differ in a single binary setting (i.e. trained on true or random labels; with or without data augmentation; etc).

Out of all such network pairs, we select only those where each network reached 100% training accuracy on the whole training set (apart from the data augmentation study).

The two generalization or sensitivity values are then used as the x and y coordinates of a point corresponding to this pair of networks (with the plot axes labels denoting the respective value of the binary parameter considered).

The position of the point with respect to the diagonal y = x visually demonstrates which configuration has smaller generalization gap / lower sensitivity.

We now perform a large-scale experiment to establish direct relationships between sensitivity and generalization in a realistic setting.

In contrast to §4.1, where we selected locations in the input space, and §4.2, where we varied a single binary parameter impacting generalization, we now sweep simultaneously over many different architectural and optimization choices ( §A.5.5).Our main result is presented in FIG3 , indicating a strong relationship between the Jacobian norm and generalization.

In contrast, Figure App .8 demonstrates that the number of transitions is not alone sufficient to compare networks of different sizes, as the number of neurons in the networks has a strong influence on transition count.

In §4.3 we established a correlation between sensitivity (as measured by the Jacobian norm) and generalization averaged over a large test set (10 4 points).

We now investigate whether the Jacobian norm can be predictive of generalization at individual points.

As demonstrated in FIG4 (top), Jacobian norm at a point is predictive of the cross-entropy loss, but the relationship is not a linear one, and not even bijective (see §A.3 for analytic expressions explaining it).

In particular, certain misclassified points (right sides of the plots) have a Jacobian norm many orders of magnitude smaller than that of the correctly classified points (left sides).

However, we do remark a consistent tendency for points having the highest values of the Jacobian norm to be mostly misclassified.

A similar yet noisier trend is observed in networks trained using 2 -loss as depicted in FIG4 (bottom).

These observations make the Jacobian norm a promising quantity to consider in the contexts of active learning and confidence estimation in future research.

We have investigated sensitivity of trained neural networks through the input-output Jacobian norm and linear regions counting in the context of image classification tasks.

We have presented extensive experimental evidence indicating that the local geometry of the trained function as captured by the input-output Jacobian can be predictive of generalization in many different contexts, and that it varies drastically depending on how close to the training data manifold the function is evaluated.

We further established a connection between the cross-entropy loss and the Jacobian norm, indicating that it can remain informative of generalization even at the level of individual test points.

Interesting directions for future work include extending our investigation to more complex architectures and other machine learning tasks.

The way of encoding a linear region c (z) of a point z described in §3 (2) guarantees that different regions obtain different codes, but different codes might be assigned to the same region if all the neurons in any layer of the network are saturated (or if weights leading from the transitioning unit to active units are exactly zero, or exactly cancel).

However, the probability of such an arrangement drops exponentially with width and hence is ignored in this work.

The equality between the discrete and continuous versions of t (x) in Equation 1 becomes exact with a high-enough sampling density k such that there are no narrow linear regions missed in between consecutive points (precisely, the encoding c (z) has to only change at most once on the line between two consecutive points z i and z i+1 ).For computational efficiency we also assume that no two neurons transitions simultaneously, which is extremely unlikely in the context of random initialization and stochastic optimization.

Figure App.7: Depiction of a trajectory in input space used to count transitions as defined in §3 (2).

An interpolation between 28 horizontal translations of a single digit results in a complex trajectory that constrains all points to lie close to the translation-augmented data, and allows for a tractable estimate of transition density around the data manifold.

This metric is used to compare models in §4.2 and §4.3.

Straight lines indicate boundaries between different linear regions (straight-line boundaries between linear regions is accurate for the case of a single-layer piecewise-linear network.

The partition into linear regions is more complex for deeper networks ).

FIG4 , each plot shows 5 random networks that fit the respective training set to a 100% with each network having a unique color.

See §A.5.6 for experimental details.

Here we briefly discuss the motivation of this work in the context of Occam's razor.

Occam's razor is a heuristic for model comparison based on their complexity.

Given a dataset D, Occam's razor gives preference to simpler models H. In the Bayesian interpretation of the heuristic BID14 simplicity is defined as evidence P [D|H] and is often computed using the Laplace approximation.

Under further assumptions BID22 , this evidence can be shown to be inversely proportional to the number of parameters in the model.

Therefore, given a uniform prior P [H] on two competing hypothesis classes, the class posterior P [H|D] ∼ P [D|H] P [H] is higher for a model with fewer parameters.

An alternative, qualitative justification of the heuristic is through considering the evidence as a normalized probability distribution over the whole dataset space: DISPLAYFORM0 and remarking that models with more parameters have to spread the probability mass more evenly across all the datasets by virtue of being able to fit more of them (Figure App.10, left) .

This similarly suggests (under a uniform prior on competing hypothesis classes) preferring models with fewer parameters, assuming that evidence is unimodal and peaks close to the dataset of interest.

Occam's razor for neural networks.

As seen in Figure 1 , the above reasoning does not apply to neural networks: the best achieved generalization is obtained by a model that has around 10 4 times as many parameters as the simplest model capable of fitting the dataset (within the evaluated search space).On one hand, BID26 ; Telgarsky (2015) demonstrate on concrete examples that a high number of free parameters in the model doesn't necessarily entail high complexity.

On the other hand, a large body of work on the expressivity of neural networks BID33 BID24 shows that their ability to compute complex functions increases rapidly with size, while Zhang et al. (2016) validates that they also easily fit complex (even random) functions with stochastic optimization.

Classical metrics like VC dimension or Rademacher complexity increase with size of the network as well.

This indicates that weights of a neural network may actually correspond to its usable capacity, and hence "smear" the evidence P [D|H] along a very large space of datasets D , making the dataset of interest D less likely.

Potential issues.

We conjecture the Laplace approximation of the evidence P [D|H] and the simplified estimation of the "Occam's factor" in terms of the accessible volume of the parameter space might not hold for neural networks in the context of stochastic optimization, and, in particular, do not account for the combinatorial growth of the accessible volume of parameter space as width increases BID23 .

Similarly, when comparing evidence as probability distributions over datasets, the difference between two neural networks may not be as drastic as in Figure App .10 (left), but more nuanced as depicted in Figure App .10 (right), with the evidence ratio being highly dependent on the particular dataset.

We interpret our work as defining hypothesis classes based on sensitivity of the hypothesis (which yielded promising results in BID36 on a toy task) and observing a strongly non-uniform prior on these classes that enables model comparison.

Indeed, at least in the context of natural images classification, putting a prior on the number of parameters or Kolmogorov complexity of the hypothesis is extremely difficult.

However, a statement that the true classification function is robust to small perturbations in the input is much easier to justify.

As such, a prior P [H] in favor of robustness over sensitivity might fare better than a prior on specific network hyper-parameters.

Above is one way to interpret the correlation between sensitivity and generalization that we observe in this work.

It does not explain why large networks tend to converge to less sensitive functions.

We conjecture large networks to have access to a larger space of robust solutions due to solving a highly-underdetermined system when fitting a dataset, while small models converge to more extreme weight values due to being overconstrained by the data.

However, further investigation is needed to confirm this hypothesis.

Reality?

might nonetheless concentrate the majority of probability mass on simple functions and the evidence curves might intersect at a small angle.

In this case, while a dataset D lying close to the intersection can be fit by both models, the Bayesian evidence ratio depends on its exact position with respect to the intersection.

DISPLAYFORM0

Here we analyze the relationship between the Jacobian norm and the cross-entropy loss at individual test points as studied in §4.4.Target class Jacobian.

We begin by relating the derivative of the target class probability J y(x) to per-point cross-entropy loss l(x) = − log [f σ (x)] y(x) (where y(x) is the correct integer class).We will denote f σ (x) by σ and drop the x argument to de-clutter notation (i.e. write f instead of f (x)).

Then the Jacobian can be expressed as DISPLAYFORM0 where is the Hadamard element-wise product.

Then indexing both sides of the equation at the correct class y yields DISPLAYFORM1 where e y is a vector of zeros everywhere except for e y = 1.

Taking the norm of both sides results in DISPLAYFORM2 We now assume that magnitudes of the individual logit derivatives vary little in between logits and over the input space 3 : DISPLAYFORM3 /n.

Since σ lies on the (n − 1)-simplex ∆ n−1 , under these assumptions we can bound:( DISPLAYFORM4 and finally DISPLAYFORM5 or, in terms of the cross-entropy loss l = − log σ y : DISPLAYFORM6 We validate these approximate bounds in Figure App .11 (top).Full Jacobian.

Equation 5 establishes a close relationship between J y and loss l = − log σ y , but of course, at test time we do not know the target class y.

This allows us to only bound the full Jacobian norm from below: DISPLAYFORM7 For the upper bound, we assume the maximum-entropy case of σ y : σ i ≈ (1 − σ y )/(n − 1), for i = y. The Jacobian norm is DISPLAYFORM8 where the first summand becomes: All reported values, when applicable, were evaluated on the whole training and test sets of sizes 50000 and 10000 respectively.

E.g. "generalization gap" is defined as the difference between train and test accuracies evaluated on the whole train and test sets.

DISPLAYFORM9 When applicable, all trajectories/surfaces in input space were sampled with 2 20 points.

All figures except for 6 and App.11 are plotted with (pale) error bars (when applicable).

The reported quantity was usually evaluated 8 times with random seeds from 1 to 8 4 , unless specified otherwise.

E.g. if a network is said to be 100%-accurate on the training set, it means that each of the 8 randomlyinitialized networks is 100%-accurate after training.

The error bar is centered at the mean value of the quantity and spans the standard error of the mean in each direction.

If the bar appears to not be visible, it may be smaller than the mean value marker.

Weight initialization, training set shuffling, data augmentation, picking anchor points of data-fitted trajectories, selecting axes of a zero-centered elliptic trajectory depend on the random seed.

A random zero-centered ellipse was obtained by generating two axis vectors with normallydistributed entries of zero mean and unit variance (as such making points on the trajectory have an expected norm equal to that of training data) and sampling points on the ellipse with given axes.

A random data-fitted ellipse was generated by projecting three arbitrary input points onto a plane where they fall into vertices of an equilateral triangle, and then projecting their circumcircle back into the input space.

Relevant figure 3.A 15-layer ReLU6-network of width 300 was trained on MNIST for 2 18 steps using SGD with momentum BID38 ; images were randomly translated with wrapping by up to 4 pixels in each direction, horizontally and vertically, as well as randomly flipped along each axis, and randomly rotated by 90 degrees clockwise and counter-clockwise.

The sampling grid in input space was obtain by projecting three arbitrary input points into a plane as described in §A.5.2 such that the resulting triangle was centered at 0 and it's vertices were at a distance 0.8 form the origin.

Then, a sampling grid of points in the [−1 ; 1] ×2 square was projected back into the input space.

Relevant figures: 4 (second row) and 5 (bottom).All networks were trained for 2 18 steps of batch size of 256 using SGD with momentum.

Learning rate was set to 0.005 and momentum term coefficient to 0.9.Data augmentation consisted of random translation of the input by up to 4 pixels in each direction with wrapping, horizontally and vertically.

The input was also flipped horizontally with probability 0.5.

When applying data augmentation (second row of Figure 4 ), the network is unlikely to encounter the canonical training data, hence few configurations achieved 100% training accuracy.

However, we verified that all networks trained with data augmentation reached a higher test accuracy than their analogues without, ensuring that the generalization gap shrinks not simply because of lower training accuracy.

For each dataset, networks of width {100, 200, 500, 1000, 2000, 3000}, depth {2, 3, 5, 10, 15, 20} and activation function {ReLU, ReLU6, HardTanh, HardSigmoid} were evaluated on 8 random seeds from 1 to 8.

Relevant figures: 1, 4 (except for the second row), 5 (top), App.8.

335671 networks were trained for 2 19 steps with random hyper-parameters; if training did not complete, a checkpoint at step 2 18 was used instead, if available.

When using L-BFGS, the maximum number of iterations was set to 2684.

The space of available hyper-parameters included 5 :

@highlight

We perform massive experimental studies characterizing the relationships between Jacobian norms, linear regions, and generalization.