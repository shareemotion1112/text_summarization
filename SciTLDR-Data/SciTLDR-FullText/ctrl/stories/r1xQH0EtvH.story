The power of neural networks lies in their ability to generalize to unseen data, yet the underlying reasons for this phenomenon remain elusive.

Numerous rigorous attempts have been made to explain generalization, but available bounds are still quite loose, and analysis does not always lead to true understanding.

The goal of this work is to make generalization more intuitive.

Using visualization methods, we discuss the mystery of generalization, the geometry of loss landscapes, and how the curse (or, rather, the blessing) of dimensionality causes optimizers to settle into minima that generalize well.

Neural networks are a powerful tool for solving classification problems.

The power of these models is due in part to their expressiveness; they have many parameters that can be efficiently optimized to fit nearly any finite training set.

However, the real power of neural network models comes from their ability to generalize; they often make accurate predictions on test data that were not seen during training, provided the test data is sampled from the same distribution as the training data.

The generalization ability of neural networks is seemingly at odds with their expressiveness.

Neural network training algorithms work by minimizing a loss function that measures model performance using only training data.

Because of their flexibility, it is possible to find parameter configurations Figure 1: A minefield of bad minima: we train a neural net classifier and plot the iterates of SGD after each tenth epoch (red dots).

We also plot locations of nearby "bad" minima with poor generalization (blue dots).

We visualize these using t-SNE embedding.

All blue dots achieve near perfect train accuracy, but with test accuracy below 53% (random chance is 50%).

The final iterate of SGD (yellow star) also achieves perfect train accuracy, but with 98.5% test accuracy.

Miraculously, SGD always finds its way through a landscape full of bad minima, and lands at a minimizer with excellent generalization.

for neural networks that perfectly fit the training data and minimize the loss function while making mostly incorrect predictions on test data.

Miraculously, commonly used optimizers reliably avoid such "bad" minima of the loss function, and succeed at finding "good" minima that generalize well.

Our goal here is to develop an intuitive understanding of neural network generalization using visualizations and experiments rather than analysis.

We begin with some experiments to understand why generalization is puzzling, and how over-parameterization impacts model behavior.

Then, we explore how the "flatness" of minima correlates with generalization, and in particular try to understand why this correlation exists.

We explore how the high dimensionality of parameter spaces biases optimizers towards landing in flat minima that generalize well.

Finally, we present some counterfactual experiments to validate the intuition we develop.

Code to reproduce experiments is available at https://github.com/genviz2019/genviz.

Neural networks define a highly expressive model class.

In fact, given enough parameters, a neural network can approximate virtually any function (Cybenko, 1989) .

But just because neural nets have the power to represent any function does not mean they have the power to learn any function from a finite amount of training data.

Neural network classifiers are trained by minimizing a loss function that measures model performance using only training data.

A standard classification loss has the form

where p θ (x, y) is the probability that data sample x lies in class y according to a neural net with parameters θ, and D t is the training dataset of size |D t |.

This loss is near zero when a model with parameters θ accurately classifies the training data.

Over-parameterized neural networks (i.e., those with more parameters than training data) can represent arbitrary, even random, labeling functions on large datasets (Zhang et al., 2016) .

As a result, an optimizer can reliably fit an over-parameterized network to training data and achieve near zero loss (Laurent and Brecht, 2018; Kawaguchi, 2016 We illustrate the difference between model fitting and generalization with an experiment.

The CIFAR-10 training dataset contains 50,000 small images.

We train two over-parameterized models on this dataset.

The first is a neural network (ResNet-18) with 269,722 parameters (nearly 6× the number of training images).

The second is a linear model with a feature set that includes pixel intensities as well as pair-wise products of pixels intensities.

1 This linear model has 298, 369 parameters, which is comparable to the neural network, and both are trained using SGD.

On the left of Figure 2 , we see that overparameterization causes both models to achieve perfect accuracy on training data.

But the linear model achieves only 49% test accuracy, while ResNet-18 achieves 92%.

The excellent performance of the neural network model raises the question of whether bad minima exist at all.

Maybe deep networks generalize because bad minima are rare and lie far away from the region of parameter space where initialization takes place?

We can confirm the existence of bad minima by incorporating a loss term that explicitly promotes poor generalization, by discouraging performance on unseen data drawn from the same distribution.

We do this by minimizing

where D t is the training set, and D d is a set of unseen examples sampled from the same distribution.

D d could be obtained via a GAN (Goodfellow et al., 2014) or additional data collection (note that it is not the test set).

Here, β parametrizes the amount of "anti-generalization" we wish to achieve.

The first term in (2) is the standard cross entropy loss (1) on the training set D t , and is minimized when the training data are classified correctly.

The second term is the reverse cross entropy loss on D d , and is minimized when D d is classified incorrectly.

With a sufficiently over-parameterized network, gradient descent on (2) drives both terms to zero, and we find a parameter vector that minimizes the original training set loss (1) while failing to generalize.

In other words, the minima found by (2) are stationary points with comparable true objective function values (Eq. (1)), indicating that it's quite possible to land in one of these "bad" minima in a normal training routine (1) if initialized within the loss basin.

Sec. 5 will show that the likelihood of this occurring is negligible.

When we use the anti-generalization loss to search for bad minima near the optimization trajectory, we see that bad minima are everywhere.

We visualize the distribution of bad minima in Figure 1 .

We run a standard SGD optimizer on the swissroll and trace out the path it takes from a random initialization to a minimizer.

We plot the iterate after every tenth epoch as a red dot with opacity proportional to its epoch number.

Starting from these iterates, we run the anti-generalization optimizer to find nearby bad minima.

We project the iterates and bad minima into a 2D plane for visualization using a t-SNE embedding 2 .

Our anti-generalization optimizer easily finds minima with poor generalization within close proximity to every SGD iterate.

Yet SGD avoids these bad minima, carving out a path towards a parameter configuration that generalizes well.

Figure 1 illustrates that neural network optimizers are inherently biased towards good minima, a behavior commonly known as "implicit regularization."

To see how the choice of optimizer affects generalization, we trained a simple neural network (VGG13) on 11 different gradient methods and 2 non-gradient methods in Figure 2 (right).

This includes LBFGS (a second-order method), and ProxProp (which chooses search directions by solving least-squares problems rather than using the gradient).

Interestingly, all of these methods generalize far better than the linear model.

While there are undeniably differences between the performance of different optimizers, the presence of implicit regularization for virtually any optimizer strongly indicates that implicit regularization may be caused in part by the geometry of the loss function, rather than the choice of optimizer alone.

Later on, we visually explore the relationship between the loss function's geometry and generalization, and how the high dimensionality of parameter space is one source of implicit regularization for optimizers.

Classical PAC learning theory balances model complexity (the expressiveness of a model class) against data volume.

When a model class is too expressive relative to the volume of training data, it has the ability to ace the training data while flunking the test data, and learning fails.

Classical theory fails to explain generalization in over-parameterized neural nets, as the complexity of networks is often large (exponential in depth (Sun et al., 2016; Neyshabur et al., 2015; Xie et al., 2015) or linear in the number of parameters (Shalev-Shwartz and Ben-David, 2014; Bartlett et al., 1998; Harvey et al., 2017) ).

Therefore classical bounds become too loose or even vacuous in the over-parameterized setting that we are interested in studying.

To explain this mismatch between empirical observation and classical theory, a number of recent works propose new metrics that characterize the capacity of neural networks.

Most of these appeal to the PAC framework to characterize the generalization ability of a model class Θ (e.g., neural nets of a shared architecture) through a high probability upper bound: with probability at least 1 − δ,

where R(θ) is generalization risk (true error) of a net with parameters θ ∈ Θ,R S (θ) denotes empirical risk (training error) with training sample S. We explain B under different metrics below.

Model space complexity.

This line of work takes B to be proportional to the complexity of the model class being trained, and efforts have been put into finding tight characterizations of this complexity. ; Bartlett et al. (2017) built on prior works (Bartlett and Mendelson, 2003; Neyshabur et al., 2015) to produce bounds where model class complexity depends on the spectral norm of the weight matrices without having an exponential dependence on the depth of the network.

Such bounds can improve the model class complexity provided that weight matrices adhere to some structural constraints (e.g. sparsity or eigenvalue concentration).

Stability and robustness.

This line of work considers B to be proportional to the stability of the model Kuzborskij and Lampert, 2018; Gonen and Shalev-Shwartz, 2017) , which is a measure of how much changing a data point in S changes the output of the model (Sokolic et al., 2017) .

However it is nontrivial to characterize the stability of a neural network.

Robustness, while producing insightful and effective generalization bounds, still suffers from the curse of the dimensionality on the priori-known fixed input manifold.

PAC-Bayes and margin theory.

PAC-Bayes bounds (McAllester, 1998; 1999; Neyshabur et al., 2017; Bartlett and Mendelson, 2003; Neyshabur et al., 2015; Golowich et al., 2018) , provide generalization guarantees for randomized predictors drawn from a learned distribution that depends on the training data, as opposed to a learned single predictor.

These bounds often yield sample complexity bounds worse than naive parameter counting, however Dziugaite and Roy (2017) show that PAC-Bayes theory does provide meaningful generalization bounds for "flat" minima.

Model compression.

Most recent theoretical work can be understood through the lens of "model compression" (Arora et al., 2018) .

Clearly, it is impossible to generalize when the model class is too big; in this case, many different parameter choices explain the data perfectly while having wildly different predictions on test data.

The idea of model compression is that neural network model classes are effectively much smaller than they seem to be because optimizers are only willing to settle into a very selective set of minima.

When we restrict ourselves to only the narrow set of models that are acceptable to an optimizer, we end up with a smaller model class on which learning is possible.

While our focus is on gaining insights through visualizations, the intuitive arguments below can certainly be linked back to theory.

The class of models representable by a network architecture has extremely high complexity, but experiments suggest that most of these models are effectively removed from consideration by the optimizer, which has an extremely strong bias towards "flat" minima, resulting in a reduced effective model complexity.

Over-parameterization is not specific to neural networks.

A traditional approach to coping with over-parameterization for linear models is to use regularization (aka "priors") to bias the optimizer towards good minima.

For linear classification, a common regularizer is the wide margin penalty (which appears in the form of an 2 regularizer on the parameters of a support vector machine).

When used with linear classifiers, wide margin priors choose the linear classifier that maximizes Euclidean distance to the class boundaries while still classifying data correctly.

Neural networks replace the classical wide margin regularization with an implicit regulation that promotes the closely related notion of "flatness."

In this section, we explain the relationship between flat minima and wide margin classifiers, and provide intuition for why flatness is a good prior.

Many have observed links between flatness and generalization.

Hochreiter and Schmidhuber (1997) first proposed that flat minima tend to generalize well.

This idea was reinvigorated by Keskar et al. (2017) , who showed that large batch sizes yield sharper minima, and that sharp minima generalize poorly.

This correlation was subsequently observed for a range of optimizers by Izmailov et al. Flatness is a measure of how sensitive network performance is to perturbations in parameters.

Consider a parameter vector that minimizes the loss (i.e., it correctly classifies most if not all training data).

If small perturbations to this parameter vector cause a lot of data misclassification, the minimizer is sharp; a small movement away from the optimal parameters causes a large increase in the loss function.

In contrast, flat minima have training accuracy that remains nearly constant under small parameter perturbations.

The stability of flat minima to parameter perturbations can be seen as a wide margin condition.

When we add random perturbations to network parameters, it causes the class boundaries to wiggle around in space.

If the minimizer is flat, then training data lies a safe distance from the class boundary, and perturbing the class boundaries does not change the classification of nearby data points.

In contrast, sharp minima have class boundaries that pass close to training data, putting those nearby points at risk of misclassification when the boundaries are perturbed.

We visualize the impact of sharpness on neural networks in Figure 3 .

We train a 6-layer fully connected neural network on the swiss roll dataset using regular SGD, and also using the anti-generalization loss to find a minimizer that does not generalize.

The "good" minimizer has a wide margin -the class boundary lies far away from the training data.

The "bad" minimizer has almost zero margin, and each data point lies near the edge of class boundaries, on small class label "islands" surrounded by a different class label, or at the tips of "peninsulas" that reach from one class into the other.

The class labels of most training points are unstable under perturbations to network parameters, and so we expect this minimizer to be sharp.

An animation of the decision boundary under perturbation is provided at https://www.youtube.com/watch?v=4VUJyQknf4s&t=.

We can visualize the sharpness of the minima in Figure 3 , but we need to take some care with our metrics of sharpness.

It is known that trivial definitions of sharpness can be manipulated simply by rescaling network parameters (Dinh et al., 2017) .

When parameters are small (say, 0.1), a perturbation of size 1 might cause a major performance degradation.

Conversely, when parameters are large (say, 100), a perturbation of size 1 might have little impact on performance.

However, rescalings of network parameters are irrelevant; commonly used batch normalization layers remove the effect of parameter scaling.

For this reason, it is important to define measures of sharpness that are invariant to trivial rescalings of network parameters.

One such measure is local entropy (Chaudhari et al., 2017) , which is invariant to rescalings, but is difficult to compute.

For our purposes, we use the filter-normalization scheme proposed in Li et al. (2018) , which simply rescales network filters to have unit norm before plotting.

The resulting sharpness/flatness measures have been observed to correlate well with generalization.

The bottom of Figure 3 visualizes loss function geometry around the two minima for the swiss roll.

These surface plots show the loss evaluated on a random 2D plane 3 sliced out of parameter space using the method described in Li et al. (2018) .

We see that the instability of class labels under parameter perturbations does indeed lead to dramatically sharper minima for the bad minimizer, while the wide margin of the good minimizer produces a wide basin.

To validate our observations on a more complex problem, we produce similar sharpness plots for the Street View House Number (SVHN) classification problem in Figure 4 using ResNet-18.

The SVHN dataset (Netzer et al., 2011 ) is ideal for this experiment because, in addition to train and test data, the creators collected a large (531k) set of extra data from the same distribution that can be used for D d in Eq. (2).

We minimize the SVHN loss function using standard training with and without penalizing for generalization (Eq. (2)).

The good, well-generalizing minimizer is flat and achieves 97.1% test accuracy, while the bad minimizer is much sharper and achieves 28.2% test accuracy.

Both achieve 100% train accuracy and use identical hyperparameters (other than the β factor), network architecture, and weight initialization.

We have seen that neural network loss functions are densely populated with both good and bad minima, and that good minima tend to have "flat" loss function geometry.

But what causes optimizers to find these good/flat minima and avoid the bad ones?

One possible explanation to the bias of stochastic optimizers towards good minima is the volume disparity between the basins around good and bad minima.

Flat minima that generalize well lie in wide basins that occupy a large volume of parameter space, while sharp minima lie in narrow basins that occupy a comparatively small volume of parameter space.

As a result, an optimizer using random initialization is more likely to land in the attraction basin for a good minimizer than a bad one.

The volume disparity between good and bad minima is magnified by the curse (or, rather, the blessing?) of dimensionality.

The differences in "width" between good and bad basins does not appear too dramatic in the visualizations in Figures 3 and 4 , or in sharpness visualizations for other datasets (Li et al., 2018) .

However, the probability of colliding with a region during a random initialization does not scale with its width, but rather its volume.

Network parameters live in very highdimensional spaces where small differences in sharpness between minima translate to exponentially large disparities in the volume of their surrounding basins.

It should be noted that the vanishing probability of finding sets of small width in high dimensions is well studied by probabilists, and is formalized by a variety of escape theorems (Gordon, 1988; Vershynin, 2018) .

To explore the effect of dimensionality on neural loss landscapes, we quantify the local volume within the low-lying basins surrounding different minima.

The volume (or "horizon") of a basin is not well-defined, especially for SGD with discrete time-steps.

For this experiment, we define the "basin" to be the set of points in a neighborhood of the minimizer that have loss value below a cutoff of 0.1 (Fig. 7) .

We chose this definition because the volume of this set can be efficiently computed.

We calculate the volume of these basins using a Monte-Carlo integration method.

Let r(φ) denote the radius of the basin (distance from minimizer to basin boundary) in the direction of the unit vector φ.

Then the n-dimensional volume of the basin is

Γ(1+n/2) is the volume of the unit n-ball, and Γ is Euler's gamma function.

We estimate this expectation by calculating r(φ) for 3k random directions, as illustrated in Figure 7 .

In Figure 5 , we visualize the combined relationship between generalization and volume for swissroll and SVHN.

By varying β, we control the generalizability of each minimizer.

As generalization accuracy decreases, we see the radii of the basins decrease as well, indicating that minima become sharper.

Figure 5 also contains scatter plots showing a severe correlation between generalization and (log) volume for various choices of the basin cutoff value.

For SVHN, the basins surrounding good minima have a volume at least 10,000 orders of magnitude larger than that of bad minima, rendering it nearly impossible to accidentally stumble upon bad minima.

Finally, we visualize the decision boundaries for several levels of generalization in Figure 6 .

All networks achieve above 99.5% training accuracy.

As the generalization gap increases, the area that belongs to the red class begins encroaching into the area that belongs to the blue class, and vice versa.

The margin between the decision boundary and training points also decreases until the training points, though correctly classified, sit on "islands" or "peninsulas" as discussed above.

Neural nets solve complex classification problems by finding "flat" minima with class boundaries that assign labels that are stable to parameter perturbations.

Using this intuition, can we formulate a problem that neural nets can't solve?

Consider the problem of separating the blue and red dots in Figure 8 .

When the distance between the inner rings is large, a neural network consistently finds a well-behaved circular boundary as in Fig. 8aa .

The wide margin of this classifier makes the minimizer "flat," and the resulting high volume makes it likely to be found by SGD.

We can remove the well-behaved minima from this problem by pinching the margin between the inner red and blue rings.

In this case, a network trained with random initialization is shown in Fig. 8b .

Now, SGD finds networks that cherry-pick red points, and arc away from the more numerous blue points to maintain a large margin.

In contrast, a simple circular decision boundary as in Fig. 8a would pass extremely close to all points on the inner rings, making such a small margin solution less stable under perturbations and unlikely to be found by SGD.

We explored the connection between generalization and loss function geometry using visualizations and experiments on classification margin and loss basin volumes, the latter of which does not appear in the literature.

While experiments can provide useful insights, they sometimes raise more questions than they answer.

We explored why the "large margin" properties of flat minima promote generalization.

But what is the precise metric for "margin" that neural networks respect?

Experiments suggest that the small volume of bad minima prevents optimizers from landing in them.

But what is a correct definition of "volume" in a space that is invariant to parameter re-scaling and other transforms, and how do we correctly identify the attraction basins for good minima?

Finally and most importantly: how do we connect these observations back to a rigorous PAC learning framework?

The goal of this study is to foster appreciation for the complex behaviors of neural networks, and to provide some intuitions for why neural networks generalize.

We hope that the experiments contained here will provide inspiration for theoretical progress that leads us to rigorous and definitive answers to the deep questions raised by generalization.

<|TLDR|>

@highlight

An intuitive empirical and visual exploration of the generalization properties of deep neural networks.