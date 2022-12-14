Adam-typed optimizers, as a class of adaptive moment estimation methods with the exponential moving average scheme, have been successfully used in many applications of deep learning.

Such methods are appealing for capability on large-scale sparse datasets.

On top of that, they are computationally efficient and insensitive to the hyper-parameter settings.

In this paper, we present a new framework for adapting Adam-typed methods, namely AdamT. Instead of applying a simple exponential weighted average, AdamT also includes the trend information when updating the parameters with the adaptive step size and gradients.

The newly added term is expected to efficiently capture the non-horizontal moving patterns on the cost surface, and thus converge more rapidly.

We show empirically the importance of the trend component, where AdamT outperforms the conventional Adam method constantly in both convex and non-convex settings.

Employing first order optimization methods, such as stochastic gradient descent (SGD), is a key of solving large-scale problems.

The classic gradient descent algorithm is widely used to update the model parameters, denoted by x, x t+1 = x t − η∇f (x t ),

where the gradient is denoted by ∇f (x t ) and the step size by η.

While the method has shown its efficiency for many contemporary tasks, the adaptive variants of SGD outperform the vanilla SGD methods on their rapid training time.

Specifically, the step size η is substituted by an adaptive step size η/ √ v t , and v t is generated from the squared gradient [∇f (x t )] 2 .

Several variants of the popular adaptive optimizers can be summarized into such common format.

These optimizers share gradients calculation and parameters updating functions, but specify different moving average schemes for calculating the parameter-wise adaptive learning rate v t .

For example, AdaGrad (Duchi et al., 2011) takes the arithmetic average of historical squared gradients [∇f (x t )] 2 .

Compared with the conventional momentum method, it adapts the learning rate to each parameter to suit the sparse data structure, and thus gains a rapid convergence speed (Ruder, 2016) .

Later, Tieleman & Hinton (2012) proposed RMSProp to reduce the aggressiveness of the decay rate in AdaGrad.

The method modifies v t to the exponentially decayed squared gradients.

Similar implementations could also be found in ADADELTA (Zeiler, 2012) .

Instead of the squared gradients, the method applies squared parameter updates to define the adaptive learning rate.

As a result, each update guarantees the same hypothetical units as the parameter.

Later, Adam (Kingma & Ba, 2015) modifies RMSProp with the idea from momentum methods (Qian, 1999) .

Except for the second moment moving average, the new rule also replaces the gradient ∇f (x t ) at the end of the Equation (1) to the first-moment estimation.

The method has practically shown its superiority regarding the converge speed and memory requirement.

While the aforementioned methods are the most famous frameworks, there are also many variants for each of them.

The examples include NAdam (Dozat, 2016) , AMSGrad (Reddi et al., 2018) and Adafom (Chen et al., 2019) .

So far, the adaptive methods with exponential moving average gradients have gained great attention with huge success in many deep learning tasks.

However, it remains unsolved whether the simple exponential smoothing results or the level information is sufficient in capturing the landscape of the cost surface.

When clear upward or downward pattern could be recognized within the moving routine, it is suggested to add a trend term on top of the single level information.

In this paper, we modify the Adam rule with trend-corrected exponential smoothing schemes, namely AdamT, to obtain the local minima with a faster speed.

To the best of our knowledge, our research is the first to apply the trend-corrected features on gradients scaling and parameters updating.

It shall be emphasized that our framework is universally implementable for all adaptive update methods that apply the exponential average term, including but not restricted to ADADELTA, RMSProp, AdaMAX and other well-recognized methods.

For the sake of conciseness, in this specific paper, we focus on Adam regarding rule modification and performance comparison.

Our contributions in this paper could be summarized in three-fold:

1.

We propose the notion of trend corrected exponential smoothing to modify the conventional application of exponential moving average in optimizers with adaptive gradients.

Our AdamT method collaborates the trend information into the update rule of Adam.

2.

We show the conditions for the method to converge in convex settings.

The regret bound is in consistent to Adam at O( √ T ).

3.

We demonstrate AdamT's convergence in both convex and non-convex settings.

The performance is compared with Adam, where AdamT shows clear superiority on both the training set and the test set, especially for non-convex problems.

For the remainder of the paper, we present the fundamental idea of Adam and Holt's linear methods in Section 2.

In Section 3 and 4, we detail the update rules and experimental analysis, respectively.

In addition, Section 5 reviews recent developments of Adam-typed optimizers.

While many of them focus more on non-convex optimizations, there is a potential to incorporate our methods with such frameworks and this extension is expected for future settings.

For adaptive gradient methods, we apply the update rule at step t + 1

where m t is the gradient updates, and conventionally it is defined as the last gradient value ∇f (x t ).

To prevent zero division, a smoothing term is included on the denominator.

In this paper, we focus our analysis and modifications on Adam.

The method was initially proposed by Kingma & Ba (2015) , and quickly becomes one of the most popular optimizers in the last few years.

The adaptive step size is accelerated from the previous square gradients

In terms of the gradient m t , Adam takes the exponentially weighted average of all previous gradients instead of solely relying on the last gradient ∇f t

While the two moment estimates from Equations (2) & (3) could potentially counteract towards zero, the seriesm t andv t are considered for bias-correction.

Formally, the rules are defined as:

The idea of extracting a smoothed new point from all the previous information is called the exponential weighted moving average.

Holt (2004) extended the method by including the trend behaviours within the series, namely trend corrected exponential smoothing or Holt's linear method.

Consider a time series {y t } for t = 1, 2, ....

Our target is to find the smoothing results at step t. We denote the smoothing result as y t+1|t .

Holt's linear method formulates the conditional forecasting by summing two smoothing equations up

For a new estimation, we first update the level term t with the weighted average of the last observation y t and its estimation y t|t−1 .

The trend term b t is updated simultaneously as the weighted average of the estimated trend t − t−1 and its previous estimation b t−1 .

The smoothing parameters for the level and the trend are denoted as α and β.

Both values could be selected between 0 and 1.

Gardner Jr & McKenzie (1985) also suggested adding a damping factor φ, so that

The damped method is identical to Holt's Linear method with φ = 1, and is the same as simple exponential moving average method with φ = 0.

When φ is positive, the parameter could be used to control the significance of the trend component.

The damped trend methods are considerably popular for forecasting tasks (Hyndman & Athanasopoulos, 2018 ).

Such methods inherent both level and trend information from historical series, while stay flexible enough to adjust the influence of the trend term via φ.

On top of that, involving the damped factor could to some extend reduce the volatility of the smoothed line.

We introduce our proposed algorithm AdamT, which is based on Adam (Kingma & Ba, 2015) with added Holt's linear trend information for both of the first moment estimate and the second raw moment estimate.

Specifically, we use trend-corrected exponential weighted moving averages in the final parameter update step instead of the level-only estimates used in Adam.

Consider the gradient of a stochastic objective function f (x) evaluated at T iterations as a time series {∇f (x t )} for t = 1, 2, . . .

, T .

According to the Holt's linear trend method illustrated in Section 2.2, we write two series { m t } and {b m t } as the exponential weighted moving averages which estimate the level and trend information of the first moment ∇f (x t ):

where β 1 , γ 1 and φ 1 have the same functionality as explained in Section 2.2 and these are regarded as hyperparameters in our algorithm.

Equation (3.1) combines the level and the trend information of first moment, which will be used for calculating the final update rule and the trend-corrected level estimates.

The procedures for the second raw moment ∇f (x t ) • ∇f (x t ) is analogous:

where the operation "•" denotes an element-wise multiplication.

The hyperparameters β 2 , γ 2 and φ 2 here share the same corresponding meanings as before.

The moving averages { v t } and {b v t } estimate the level and trend of the second raw moment respectively.

The term v t combines these two information, which will be used in the calculations of final update rule and trend-corrected level estimates of the second raw moment.

The series {m t } and {v t }, as a result, are also initialized as zero vectors.

As observed in Kingma & Ba (2015) , the exponential weighted moving averages could bias towards zero, especially during the early training stage.

We perform the bias correction for the two level estimates { m t } and { v t } by following Kingma & Ba (2015) .

For the two trend estimates {b m t } and {b v t }, we correct the bias in a different way by taking into account the effect of damping parameters (φ 1 , φ 2 ).

Thus, the bias-corrected version of the series {m t } and {v t } can be written as:

The justification for the two bias-corrected trend estimates is provided in Appendix A. Similar to Adam, we consider the adaptive update rule with the bias-corrected first moment estimate and the second raw moment estimate:

where is a positive tiny number added in the denominator to avoid zero-division case.

Please note that the series {m t } and {v t } in AdamT are different from that of Adam.

The two series are trend-corrected (also bias-corrected) estimates of both moments.

The direction of the effective step ∆ t = η ·m t / |v t | (with = 0) in the parameter space depends on the joint effect of the first moment level and trend estimates.

In the update rule (4), we only care about the magnitude ofv t by taking the absolute value and thus the ratiom t / |v t | can be seen as a signal-to-noise ratio.

Note that the effective step ∆ t in our algorithm is also invariant to the scale of the gradients.

Specifically, re-scaling the gradients ∇f (x t ) with a factor c will scale respectively, and finally cancel out in the parameter update rule (c ·m t )/( |c 2 ·v t |) =m t / |v t |.

Note that our proposed method AdamT has two extra computational steps, that is Equations (3.1) & (3.1).

However, the computational complexity of these two steps is almost linear in time.

Therefore, we can conclude that AdamT yields a superior performance compared with Adam (the results will be shown in the experiment section) with a minimal additional computational cost.

In our algorithm, we set the hyperparameters β 1 , γ 1 , β 2 , γ 2 according to the suggestion in Kingma & Ba (2015) .

The smoothing parameters for the first moment estimates are set to 0.9, that is β 1 = γ 1 = 0.9, while the smoothing parameters for the second raw moment estimates are set to 0.999, that is β 2 = γ 2 = 0.999.

We empirically find that the good default values of the two damping parameters can be set to φ 1 = φ 2 = 0.5.

The pseudo-code of our AdamT is provided in Algorithm 1.

We present the key results of convergence analysis for AdamT in this section and the details is presented in Appendix B. Theorem 3.1.

Assume that the objective function f t has bounded gradients, that is ∇f t (x) 2 ≤ G,

AdamT achieves the following guarantee for all T ≥ 1

This result implies that AdamT has O(

Tv T,i ≤ dCG ∞ √ T for some positive constant C. Hence, we can prove that the average regret of AdamT converges,

17:

Corollary 3.1.1.

By following the assumptions in Theorem 3.1, for all T ≥ 1 AdamT achieves

This result follows immediately from Theorem 3.1 and thus lim T →∞

We evaluate the proposed algorithm AdamT on both convex and non-convex real-world optimization problems with several popular types of machine learning models.

The models we considered in the experiments include logistic regression which has a well-known convex loss surface, and different neural network models, including feedforward neural networks, convolutional neural networks and variational autoencoder.

Neural Networks with non-linear activation function typically have an inherent non-convex loss surface which is more challenging for an optimization method.

We compare our method with Adam (Kingma & Ba, 2015) and demonstrate the effectiveness of the trend information of the gradients infused in AdamT. The experiment results show that our method has a converge faster to reach a better minimum point than Adam.

The observation evidences that the added trend information effectively helps AdamT to better capture the landscape of loss surface.

In the following experiments, we prepare the same set of initial values for the models, so that the initial model losses (the loss value at epoch = 0) are identical for all the optimization methods.

In terms of the hyperparameters, all the smoothing parameters (β 1 , β 2 in Adam and β 1 , β 2 , γ 1 , γ 2 in AdamT) are set at their corresponding default values which are provided in Algorithm 1.

The damping factors (φ 1 , φ 2 ) and the learning rate η are tuned through a dense grid search to produce the best results for both of the optimizers.

All the experiments and optimizers are written in PyTorch.

We first evaluate AdamT on the logistic regression for multi-class classification problem with Fashion-MNIST dataset (Xiao et al., 2017) which is a MNIST-like dataset of fashion products.

The dataset has 60, 000 training samples and 10, 000 testing samples.

Each of them has 28 × 28 pixels.

Each of the samples is classified into one of the 10 fashion products.

The cross-entropy loss function has a well-behaved convex surface.

The learning rate η for both Adam and AdamT is set to be constant during the training procedure.

We use minibatch training with size set to 128.

The results are reported in Figure 1 .

Since the superiority of our method over Adam is relatively small in this experiment, the plot of loss value against epoch cannot visualize the difference.

Instead, we plot the loss difference of the two optimizers, which is (Loss Adam − Loss AdamT ) against training epoch.

The difference above zero reflect the advantage of AdamT. Figure 1 indicates that AdamT converges faster at the early training stage and constantly outperforms Adam during the rest of the training phase, though the advantage is relatively small in this experiment.

The AdamT gives a similar (slightly better) performance on the test dataset as Adam.

The loss surface of logistic regression is convex and well-behaved so that the trend information of AdamT cannot further provide much useful information for optimization, which results in a small advantage in this experiment.

To investigate the performance on non-convex objective functions, we conduct the experiment with feedforward neural networks on The Street View House Numbers (SVHN) dataset (Netzer et al., 2011) for a digit classification problem.

We pre-process this RGB image dataset into grayscale for dimension reduction by taking the average across the channels for each pixel in the image.

The samples are 32 × 32 grayscale images.

The neural network used in this experiment has two fullyconnected hidden layers, each of which has 1, 400 hidden units and ReLU activation function is used for the two hidden layers.

We use softmax cross-entropy loss function for training the model.

To evaluate the performance of the optimizers in noisy settings, we apply a stochastic regularization method in the model for a separate experiment.

Specifically, we include two dropout layers (Srivastava et al., 2014) , where one is applied between the two hidden layers and the other one is used before the output layer.

The dropout probability is set to 0.5 for both of the two dropout layers.

In the experiments, we use a constant learning rate η and minibatch training with size set to 128.

We examine the training and test loss of the models with and without dropout layers for the two optimizers.

According to Figure 2 , we find that AdamT outperforms Adam significantly.

In terms of the training process, AdamT yields a faster convergence and reaches a better position than Adam for the models, both with and without dropout layers.

The superior performance of AdamT is also shown in the test phase, which demonstrates that AdamT also has a better generalization ability than Adam.

For the model without dropout, our method performs on a par with Adam on the test dataset.

Comparing to logistic regression, the loss surface in this experiment becomes complex and non-convex.

The trend estimates of the gradients from AdamT can provide more meaningful information of the landscape of the loss surface, and it encourages a better performance on AdamT.

Convolutional neural network (CNN) is the main workhorse for Computer Vision tasks.

We train a CNN model on standard CIFAR-10 dataset for a multi-class classification task.

The dataset contains 50, 000 training samples and 10, 000 test samples, and each sample is an RGB 32 × 32 image.

We pre-process the dataset by normalizing the pixel values to the range [−1, 1] for a more robust training.

The CNN model employed in this experiment is similar to the model used in Reddi et al. (2018) , which has the following architecture.

There are 2 stages of alternating convolution and max pooling layers.

Each convolution layer has 64 channels and kernel size 5 × 5 with stride 1.

Each max pooling layer is applied with a kernel size of 2 × 2.

After that, there is a fully-connected layer with 600 hidden units and a dropout probability 0.5 followed by the output layer with 10 units.

We use ReLU for the activation function and softmax cross-entropy for the loss function.

The model is trained with a tuned constant learning rate and minibatch size 128 same as the previous experiments.

The experiment results are reported in Figure 3 .

We can observe that the proposed AdamT clearly excels Adam on the training loss, and this superiority translates into a more significant advantage of AdamT on the test loss, which again demonstrates a better generalization ability.

Variational Autoencoder (VAE) (Kingma & Welling, 2014; Rezende et al., 2014 ) is one of the most popular deep generative models for density estimation and image generation.

In this experiment, we train a VAE model on the standard MNIST dataset which contains 60, 000 training samples and 10, 000 test samples.

Each sample is one 28 × 28 black-and-white image of the handwritten digit.

The VAE model used in this experiment exactly matches the architecture presented in Kingma & Welling (2014) : Gaussian encoder and Bernoulli decoder, both of which are implemented by feedforward neural networks with single hidden layer and there are 500 hidden units in each hidden layer.

We employ the hyperbolic tangent activation function for the model and set the dimensionality of the latent space as 20.

We use the constant learning rate and set the minibatch size to 150.

We examine the Evidence Lower Bound (ELBO) of the training and testing phases for the two optimizers' performance assessment.

See Figure 4 for the experiment results.

Due to the issue of different scales, we plot the difference between the ELBOs produced by the two optimizers.

Similar to the first experiment, we plot the difference value (ELBO Adam − ELBO AdamT ) against the epoch for training and testing.

We observe that our AdamT has a much faster convergence at the early stage of training than Adam and constantly excels Adam during the rest of the training phase.

The superior performance of AdamT in this experiment also translates into a clear advantage in the testing phase.

We consider the class of adaptive moment estimation methods with exponential moving average scheme as Adam-type learning algorithms.

The fundamental idea was proposed in Kingma & Ba (2015) and quickly extended to many variants.

Some examples include AdaMax (Kingma & Ba, 2015) , Nadam (Dozat, 2016) and AdamW (Loshchilov & Hutter, 2019) .

Despite the efficiency in practice, the conventional Adam-type methods fail to guarantee global convergences.

Reddi et al. (2018) discussed the problematic short-term memory of the gradients.

For the convex settings, they proposed AMSGrad that promises a global optimization with a comparable performance.

Except for some other recent studies for convex optimization (Xu et al., 2017; Levy et al., 2018) , several works developed optimization methods for non-convex problems.

Padam (Chen & Gu, 2018; Zhou et al., 2018) introduces a partial adaptive parameter p to interpolate between SGD with momentum and Adam so that adjacent learning rates could decrease smoothly.

AdaUSM (Zou & Shen, 2018) appends the idea of unified momentum for non-decreasing sequence of weights.

AdaFom (Chen et al., 2019) obtains first-order stationary by taking simple average on the second moment estimation.

More conditions for pursuing global convergence were summarized in Zou et al. (2019), basing on the currently successful variants.

In this work, we have modified the scheme to calculate the adaptive step size from exponential moving average to trend-corrected exponential smoothing.

Empirical results demonstrate that our method, AdamT, works well in practice and constantly beats the baseline method Adam.

We leave some potentials for future developments.

First, although we focused primarily on ADAM for theoretical and experimental analysis, we believe that similar ideas could also extend to other adaptive gradient methods, such as RMSProp (Tieleman & Hinton, 2012) and AMSGrad (Reddi et al., 2018) .

Also, this work, the same as the original ADAM method, relies on the theoretical assumption of convex problems settings.

We have demonstrated its computational ability on the non-convex settings, and it is possible to extend the theoretical framework to non-convex scenarios.

Some potential candidates in the latest research are listed in Section 5.

To find how the expectation of the trend estimates b m t relates to the expectation of the difference between the level estimates at successive timesteps ( m t − m t−1 ), we take the expectation for both sides of the above equation:

where ζ can be considered as a small constant, since the factor (γ 1 φ 1 ) t−i will be tiny if the associated expectation E[( )] is stationary, the constant ζ will be zero.

To further simplify the above equation, we apply the formula for the sum of geometric sequence:

This suggests that we can use the term

] to correct the bias and close the discrepancy between the above two expectations at the presence of the damping factor φ 1 .

We investigate the convergence of AdamT with regret minimization by following Zinkevich (2003) .

Formally, the regret is defined as the difference between total cost and that of best point in hindsight

where f (x t ) is the convex function with arguments x t at iteration t and x * denotes the optimal set of the parameters.

For the sake of clarity, we denote f (x t ) as f t .

For a vanishing average regret, Jensen's inequality implies that the average decision f (x T ) converges to f (x * ).

In other words,

Below we list essential theorems and lemmas.

Definition B.1 (Convexity).

Let K ⊆ R d be a bounded convex and compact set in Euclidean space.

A function f : K → R is convex if for any x, y ∈ K

We denote by D an upper bound on the diameter of K, then x, y ∈ K, x − y ≤ D Definition B.2 (Subgradient).

The set of all subgradients of a function f at x, denoted ∂f (x), is the set of all vectors u such that

Suppose f is differentiable, for any existing gradient ∇f (x), we have ∇f (x) ∈ ∂f (x) and ∀y ∈ K

We denote G ≥ 0 and upper bound on the norm of the subgradients of f over K, i.e., ∇f (x) ≤ G for all x ∈ K. In other words, the function f is Lipschitz continuous with parameter G, that is, for all

Below are the three core lemmas that applied in proving the regret bound.

Lemma B.3 (Kingma & Ba (2015)).

We define ∇f 1:t,i ∈ R t as the vector at the i th dimension of the gradients over all iterations till t, ∇f 1:

Lemma B.4.

For β 1 , γ 1 , β 2 , γ 2 ∈ [0, 1) and φ 1 , φ 2 ∈ [0, 1] the series {m t,i } and {v t,i } has the following summation form

Proof.

We start fromm t,i .

It can be shown that

Similarly forv t , we havê

Theorem B.5.

Assume that the objective function f t has bounded gradients, that is ∇f t (x) 2 ≤ G,

We further assume that the distance between any x t produced by AdamT is bounded, that is x n − x m 2 ≤ D, x m − x n ∞ ≤ D ∞ for any m, n ∈ {1, 2, . . .

, T }, and β 1 , γ 1 , β 2 , γ 2 ∈ [0, 1), φ 1 , φ 2 ∈ [0, 1].

The proposed AdamT achieves the following guarantee for all T ≥ 1

Proof.

According to Theorem B.2, for any convex function f t and x * ∈ K, we have

which implies the regret upper bound

From the update rules in Algorithm, (1 − γ 1 )(1 − (γ 1 φ 1 ) t ) .

Consider the i th dimension of the parameter vector at time t (1 − γ t 1 )

The regret bound can be derived by taking summation across all the dimensions for i = 1, 2, . . .

, d and the convex functions for t = 1, 2, . . .

, T .

@highlight

We present a new framework for adapting Adam-typed methods, namely AdamT, to include the trend information when updating the parameters with the adaptive step size and gradients.

@highlight

A new type of Adam variant that uses Holt's linear method to compute the smoothed first order and second order momentum instead of using exponential weighted average.