Ordinary stochastic neural networks mostly rely on the expected values of their weights to make predictions, whereas the induced noise is mostly used to capture the uncertainty, prevent overfitting and slightly boost the performance through test-time averaging.

In this paper, we introduce variance layers, a different kind of stochastic layers.

Each weight of a variance layer follows a zero-mean distribution and is only parameterized by its variance.

It means that each object is represented by a zero-mean distribution in the space of the activations.

We show that such layers can learn surprisingly well, can serve as an efficient exploration tool in reinforcement learning tasks and provide a decent defense against adversarial attacks.

We also show that a number of conventional Bayesian neural networks naturally converge to such zero-mean posteriors.

We observe that in these cases such zero-mean parameterization leads to a much better training objective than more flexible conventional parameterizations where the mean is being learned.

Modern deep neural networks are usually trained in a stochastic setting.

They use different stochastic layers BID8 ; BID12 ) and stochastic optimization techniques BID14 ; Kingma & Ba (2014) ).

Stochastic methods are used to reduce overfitting BID8 ; BID13 ; BID12 ), estimate uncertainty BID5 ; Malinin & Gales (2018) ) and to obtain more efficient exploration for reinforcement learning BID4 ; Plappert et al. (2017) ) algorithms.

Bayesian deep learning provides a principled approach to training stochastic models (Kingma & Welling (2013) ; Rezende et al. (2014) ).

Several existing stochastic training procedures have been reinterpreted as special cases of particular Bayesian models, including, but not limited to different versions of dropout BID5 ), drop-connect (Kingma et al. (2015) ), and even the stochastic gradient descent itself BID7 ).

One way to create a stochastic neural network from an existing deterministic architecture is to replace deterministic weights w ij with random weightsŵ ij ∼ q(ŵ ij | φ ij ) (Hinton & Van Camp (1993) ; BID1 ).

During training, a distribution over the weights is learned instead of a single point estimate.

Ideally one would want to average the predictions over different samples of such distribution, which is known as test-time averaging, model averaging or ensembling.

However, test-time averaging is impractical, so during inference the learned distribution is often discarded, and only the expected values of the weights are used instead.

This heuristic is known as mean propagation or the weight scaling rule BID8 ; Goodfellow et al. (2016) ), and is widely and successfully used in practice BID8 ; Kingma et al. (2015) ; Molchanov et al. (2017) ).In our work we study the an extreme case of stochastic neural network where all the weights in one or more layers have zero means and trainable variances, e.g. w ij ∼ N (0, σ 2 ij ).

Although no information get stored in the expected values of the weights, these models can learn surprisingly well and achieve competitive performance.

Our key results can be summarized as follows:1.

We introduce variance layers, a new kind of stochastic layers that store information only in the variances of its weights, keeping the means fixed at zero, and mapping the objects into zero-mean distributions over activations.

The variance layer is a simple example when the weight scaling rule BID8 ) fails.2.

We draw the connection between neural networks with variance layers (variance networks) and conventional Bayesian deep learning models.

We show that several popular Bayesian models (Kingma et al. (2015) ; Molchanov et al. (2017) ) converge to variance networks, and demonstrate a surprising effect -a less flexible posterior approximation may lead to much better values of the variational inference objective (ELBO).3.

Finally, we demonstrate that variance networks perform surprisingly well on a number of deep learning problems.

They achieve competitive classification accuracy, are more robust to adversarial attacks and provide good exploration in reinforcement learning problems.

A deep neural network is a function that outputs the predictive distribution p(t | x, W ) of targets t given an object x and weights W .

Recently, stochastic deep neural networks -models that exploit some kind of random noise -have become widely popular BID8 ; Kingma et al. (2015) ).

We consider a special case of stochastic deep neural networks where the parameters W are drawn from a parametric distribution q(W | φ).

During training the parameters φ are adjusted to the training data (X, T ) by minimizing the sum of the expected negative log-likelihood and an optional regularization term R(φ).

In practice this objective equation 1 is minimized using one-sample minibatch gradient estimation.

DISPLAYFORM0 This training procedure arises in many conventional techniques of training stochastic deep neural networks, such as binary dropout BID8 ), variational dropout (Kingma et al. (2015) ) and drop-connect BID12 ).

The exact predictive distribution E q(W | φ) p(t | x, W ) for such models is usually intractable.

However, it can be approximated using K independent samples of the weights equation 2.

This technique is known as test-time averaging.

Its complexity increases linearly in K. DISPLAYFORM1 In order to obtain a more computationally efficient estimation, it is common practice to replace the weights with their expected values equation 3.

This approximation is known as the weight scaling rule BID8 ).

DISPLAYFORM2 As underlined by Goodfellow et al. (2016) , while being mathematically incorrect, this rule still performs very well on practice.

The success of weight scaling rule implies that a lot of learned information is concentrated in the expected value of the weights.

In this paper we consider symmetric weight distributions q(W | φ) = q(−W | φ).

Such distributions cannot store any information about the training data in their means as EW = 0.

In the case of conventional layers with symmetric weight distribution, the predictive distribution p(t | x, EW = 0) does not depend on the object x. Thus, the weight scaling rule results in a random guess quality predictions.

We would refer to such layers as the variance layers, and will call neural networks that at least one variance layer the variance networks.

The visualization of objects activation samples from a variance layer with two variance neurons.

The network was learned on a toy four-class classification problem.

The two plots correspond to two different random initializations.

We demonstrate that a variance layer can learn two fundamentally different kinds of representations (a) two neurons repeat each other, the information about each class is encoded in variance of each neuron (b) two neurons encode an orthogonal information, both neurons are needed to identify the class of the object.

In this section, we consider a single fully-connected layer 1 with I input neurons and O output neurons, before a non-linearity.

We denote an input vector by a k ∈ R I and an output vector by b k ∈ R O , a weight matrix by W ∈ R I×O .

The output of the layer is computed as b k = a k W .

A standard normal distributed random variable is denoted by ∼ N (0, 1).Most stochastic layers mostly rely on the expected values of the weights to make predicitons.

We introduce a Gaussian variance layer that by design cannot store any information in mean values of the weights, as opposed to conventional stochastic layers.

In a Gaussian variance layer the weights follow a zero-mean Gaussian distribution w ij = σ ij · ij ∼ N (w ij | 0, σ 2 ij ), so the information can be stored only in the variances of the weights.

To get the intuition of how the variance layer can output any sensible values let's take a look at the activations of this layer.

A Gaussian distribution over the weights implies a Gaussian distribution over the activations BID13 ; Kingma et al. (2015) ).

This fact is used in the local reparameterization trick (Kingma et al. (2015) ), and we also rely on it in our experiments.

DISPLAYFORM0 Conventional layer DISPLAYFORM1 Variance layerIn Gaussian variance layer an expectation of b mj is exactly zero, so the first term in eq. equation 4 disappears.

During training, the layer can only adjust the variances DISPLAYFORM2 ij of the output.

It means that each object is encoded by a zero-centered fully-factorized multidimensional Gaussian rather than by a single point / a non-zero-mean Gaussian.

The job of the following layers is then to classify such zero-mean Gaussians with different variances.

It turns out that such encodings are surprisingly robust and can be easily discriminated by the following layers.

We illustrate the intuition on a toy classification problem.

Object of four classes were generated from Gaussian distributions with µ ∈ {(3, 3), (3, 10), (10, 3), (10, 10)} and identity covariance matrices.

A classification network consisted of six fully-connected layers with ReLU non-linearities, where the fifth layer is a bottleneck variance layer with two output neurons.

In FIG0 we plot the activations of the variance layer that were sampled similar to equation equation 4.

The exact expressions are presented in Appendix E. Different colors correspond to different classes.

We found that a variance bottleneck layer can learn two fundamentally different kinds of representations that leads to equal near perfect performance on this task.

In the first case the same information is stored in two available neurons FIG0 .

We see this effect as a kind of in-network ensembling: averaging over two samples from the same distribution results in a more robust prediction.

Note that in this case the information about four classes is robustly represented by essentially only one neuron.

In the second case the information stored in these two neurons is different.

Each neuron can be either activated (i.e. have a large variance) or deactivated (i.e. have a low variance).

Activation of the first neuron corresponds to either class 1 or class 2, and activation of the second neuron corresponds to either class 1 or class 3.

This is also enough to robustly encode all four classes.

Other combinations of these two cases are possible, but in principle, we see how the variances of the activations can be used to encode the same information as the means.

As shown in Section 6, although this representation is rather noisy, it robustly yields a relatively high accuracy even using only one sample, and test-time averaging allows to raise it to competitive levels.

We observe the same behaviour with real tasks.

See Appendix D for the visualization of the embeddings, learned by a variance LeNet-5 architecture on the MNIST dataset.

One could argue that the non-linearity breaks the symmetry of the distribution.

This could mean that the expected activations become non-zero and could be used for prediction.

However, this is a fallacy: we argue that the correct intuition is that the model learns to distinguish activations of different variance.

To prove this point, we train variance networks with antisymmetric non-linearities like (e.g. a hyperbolic tangent) without biases.

That would make the mean activation of a variance layer exactly zero even after a non-linearity.

See Appendix C for more details.

Other types of variance layers may exploit different kinds of multiplicative or additive symmetric zero-mean noise distributions .

These types of noise include, but not limited to: DISPLAYFORM0 In all these models the learned information is stored only in the variances of the weights.

Applied to these type of models, the weight scaling rule (eq. equation 3) will result in the random guess performance, as the mean of the weights is equal to zero.

Note that we cannot perform an exact local reparameterization trick for Bernoulli or Uniform noise.

We can however use moment-matching techniques similar to fast dropout BID13 ).

Under fast dropout approximation all three cases would be equivalent to a Gaussian variance layer.

We were able to train a LeNet-5 architecture BID3 ) on the MNIST dataset with the first dense layer being a Gaussian variance layer up to 99.3 accuracy, and up to up to 98.7 accuracy with a Bernoulli or a uniform variance layer.

Such gap in the performance is due to the lack of the local reparameterization trick for Bernoulli or uniform random weights.

The complete results for the Gaussian variance layer are presented in Section 6.

In this section we review several Gaussian dropout posterior models with different prior distributions over the weights.

We show that the Gaussian dropout layers may converge to variance layers in practice.

Doubly stochastic variational inference (DSVI) BID11 ) with the (local) reparameterization trick (Kingma & Welling (2013) ; Kingma et al. (2015) ) can be considered as a special case of training with noise, described by eq. equation 1.

Given a likelihood function p(t | x, W ) and a prior distribution p(W ), we would like to approximate the posterior distribution p(W | Xtrain, Ttrain) ≈ q(W | φ) over the weights W .

This is performed by maximization of the variational lower bound (ELBO) w.r.t.

the parameters φ of the posterior approximation q(W | φ) DISPLAYFORM0 The variational lower bound consists of two parts.

One is the expected log likelihood term E q(W | φ) log p(T | X, W ) that reflects the predictive performance on the training set.

The other is the KL divergence term KL(q(W | φ) p(W )) that acts as a regularizer and allows us to capture the prior knowledge p(W ).

We consider the Gaussian dropout approximate posterior that is a fully-factorized Gaussian et al. (2015) ) with the Gaussian "dropout rate" α shared among all weights of one layer.

We explore the following prior distributions: DISPLAYFORM0 Symmetric log-uniform distribution p(w ij ) ∝ 1 |wij | is the prior used in variational dropout (Kingma et al. (2015) ; Molchanov et al. (2017) ).

The KL-term for the Gaussian dropout posterior turns out to be a function of α and can be expressed as follows: DISPLAYFORM1 This KL-term can be estimated using one MC sample, or accurately approximated.

In our experiments, we use the approximation, proposed in Sparse Variational Dropout (Molchanov et al. FORMULA0 ).Student's t-distribution with ν degrees of freedom is a proper analog of the log-uniform prior, as the log-uniform prior is a special case of the Student's t-distribution with zero degrees of freedom.

As ν goes to zero, the KL-term for the Student's t-prior equation 10 approaches the KL-term for the log-uniform prior equation 7.

DISPLAYFORM2 As the use of the improper log-uniform prior in neural networks is questionable (Hron et al. FORMULA0 ), we argue that the Student's t-distribution with diminishing values of ν results in the same properties of the model, but leads to a proper posterior.

We use one sample to estimate the expectation equation 10 in our experiments, and use ν = 10 BID9 MacKay et al. (1994) ) has been previously applied to linear models with DSVI BID11 ), and can be applied to Bayesian neural networks without changes.

Following BID11 ), we can show that in the case of the Gaussian dropout posterior, the optimal prior variance λ 2 ij would be equal to (α + 1)µ 2 ij , and the KL-term KL(q(W | φ) p(W )) would then be calculated as follows: DISPLAYFORM3 DISPLAYFORM4 Note that in the case of the log-uniform and the ARD priors, the KL-divergence between a zerocentered Gaussian and the prior is constant.

For the ARD prior it is trivial, as the prior distribution p(w) is equal to the approximate posterior q(w) and the KL is zero.

For the log-uniform distribution the proof is presented in Appendix F. Note that a zero-centered posterior is optimal in terms of these KL divergences.

In the next section we will show that Gaussian dropout layers with these priors can indeed converge to variance layers.

As illustrated in Figure 2 , in all three cases the KL term decreases in α and pushes α to infinity.

We would expect the data term to limit the learned values of α, as otherwise the model would seem to be overregularized.

Surprisingly, we find that in practice for some layers α's may grow to essentially infinite values (e.g. α > 10 7 ).

When this happens, the approximate posterior N (µ ij , αµ 2 ij ) becomes indistinguishable from its zero-mean approximation N (0, αµ 2 ij ), as its standard deviation √ α|µ ij | becomes much larger than its mean µ ij .

We prove that as α goes to infinity, the Maximum Mean Discrepancy (Gretton et al. (2012) ) between the approximate posterior and its zero-mean approximation goes to zero.

Theorem 1.

Assume that αt −→ +∞ as t −→ +∞.

Then the Gaussian dropout posterior DISPLAYFORM0 in terms of Maximum Mean Discrepancy: DISPLAYFORM1 The proof of this fact is provided in Appendix A.

It is an important result, as MMD provides an upper bound equation 38 on the change in the predictions of the ensemble.

It means that we can replace the learned posterior N (µ ij , αµ 2 ij ) with its zero-centered approximation N (0, αµ 2 ij ) without affecting the predictions of the model.

In this sense we see that some layers in these models may converge to variance layers.

Note that although α may grow to essentially infinite values, the mean and the variance of the corresponding Gaussian distribution remain finite.

In practice, as α tends to infinity, the means µ ij tend to zero, and the variances σ 2 ij = αµ 2 ij converge to finite values.

During the beginning of training, the Gaussian dropout rates α are low, and the weights can be replaced with their expectations with no accuracy degradation.

After the end of training, the dropout rates are essentially infinite, and all information is stored in the variances of the weights.

In these two regimes the network behave very differently.

If we track the train or test accuracy during training we can clearly see a kind of "phase transition" between these two regimes.

See Figure 3 for details.

We observe the same results for all mentioned prior distributions.

The corresponding details are presented in Appendix B.

In this section we show how different parameterizations of the Gaussian dropout posterior influence the value of the variational lower bound and the properties of obtained solution.

We consider the same objective that is used in sparse variational dropout model (Molchanov et al. (2017) ), and consider the following parameterizations for the approximate posterior q(w ij ): DISPLAYFORM0 Note that the additive and weight-wise parameterizations are equivalent and that the layer-and the neuron-wise parameterizations are their less flexible special cases.

We would expect that a more flexible approximation would result in a better value of variational lower bound.

Surprisingly, in practice we observe exactly the opposite: the simpler the approximation is, the better ELBO we obtain.

The optimal value of the KL term is achieved when all α's are set to infinity, or, equivalently, the mean is set to zero, and the variance is nonzero.

In the weight-wise and additive parameterization α's for some weights get stuck in low values, whereas simpler parameterizations have all α's converged to effectively infinite values.

The KL term for such flexible parameterizations is orders of magnitude worse, resulting in a much lower ELBO.

See TAB0 for further details.

It means that a more flexible parameterization makes the optimization problem much more difficult.

It potentially introduces a large amount of poor local optima, e.g. sparse solutions, studied in sparse variational dropout (Molchanov et al. (2017) ).

Although such solutions have lower ELBO, they can still be very useful in practice.

We perform experimental evaluation of variance networks on classification and reinforcement learning problems.

Although all learned information is stored only in the variances, the models perform surprisingly well on a number of benchmark problems.

Also, we found that variance networks Here we show that a variance layer can be pruned up to 98% with almost no accuracy degradation.

We use magnitude-based pruning for σ's (replace all σ ij that are below the threshold with zeros), and report test-time-averaging accuracy.are more resistant to adversarial attacks than conventional ensembling techniques.

All stochastic models were optimized with only one noise/weight sample per step.

Experiments were implemented using PyTorch (Paszke et al. FORMULA0 ).

The code is available at https://github.com/ da-molchanov/variance-networks.

We consider three image classification tasks, the MNIST (LeCun et al. (1998) ), CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton FORMULA1 datasets.

We use the LeNet-5-Caffe architecture BID3 ) as a base model for the experiments on the MNIST dataset, and a VGGlike architecture BID17 ) on CIFAR-10/100.

As can be seen in TAB1 , variance networks provide the same level of test accuracy as conventional binary dropout.

In the variance LeNet-5 all 4 layers are variational dropout layers with layer-wise parameterization equation 14.

Only the first fully-connected layer converged to a variance layer.

In the variance VGG the first fully-connected layer and the last three convolutional layers are variational dropout layers with layer-wise parameterization equation 14.

All these layers converged to variance layers.

For the first dense layer in LeNet-5 the value of log α reached 6.9, and for the VGG-like architecture log α > 15 for convolutional layers and log α > 12 for the dense layer.

As shown in FIG2 , all samples from the variance network posterior robustly yields a relatively high classification accuracy.

In Appendix D we show that the intuition provided for the toy problem still holds for a convolutional network on the MNIST dataset.

Similar to conventional pruning techniques (Han et al. (2015) ; BID15 ), we can prune variance layer in LeNet5 by the value of weights variances.

Weights with low variances have small contributions into the variance of activation and can be ignored.

In FIG3 we show sparsity and accuracy of obtained model for different threshold values.

Accuracy of the model is evaluated by test time averaging over 20 random samples.

Up to 98% of the layer parameters can be zeroed out with no accuracy degradation.

Recent progress in reinforcement learning shows that parameter noise may provide efficient exploration for a number of reinforcement learning algorithms BID4 ; Plappert et al. FORMULA0 ).

These papers utilize different types of Gaussian noise on the parameters of the model.

However, increasing the level of noise while keeping expressive ability may lead to better exploration.

In this section, we provide a proof-of-concept result for exploration with variance network parameter noise on two simple gym BID2 ) environments with discrete action space: the CartPole-v0 BID0 ) and the Acrobot-v1 FIG0 ).

The approach we used is a policy gradient proposed by (Williams (1992); Sutton et al. FORMULA1 ).In all experiments the policy was approximated with a three layer fully-connected neural network containing 256 neurons on each hidden layer.

Parameter noise and variance network policies had the second hidden layer to be parameter noise BID4 ; Plappert et al. FORMULA0 ) and variance (Section 3) layer respectively.

For both methods we made a gradient update for each episode with individual samples of noise per episode.

Stochastic gradient learning is performed using Adam (Kingma & Ba FORMULA0 ).

Results were averaged over nine runs with different random seeds.

FIG4 shows the training curves.

Using the variance layer parameter noise the algorithm progresses slowly but tends to reach a better final result.

Deep neural networks suffer from adversarial attacks (Goodfellow et al. (2014) ) -the predictions are not robust to even slight deviations of the input images.

In this experiment we study the robustness of variance networks to targeted adversarial attacks.

The experiment was performed on CIFAR-10 (Krizhevsky & Hinton FORMULA1 ) dataset on a VGG-like architecture BID17 ).

We build target adversarial attacks using the iterative fast sign algorithm (Goodfellow et al. FORMULA0 ) with a fixed step length ε = 0.5, and report the successful attack rate FIG5 .

We compare our approach to the following baselines: a dropout network with test time averaging BID8 ), and deep ensembles (Lakshminarayanan et al. (2017) ) and a deterministic network.

We average over 10 samples in ensemble inference techniques.

Deep ensembles were constructed from five separate networks.

All methods were trained without adversarial training (Goodfellow et al. (2014) ).

Our experiments show that variance network has better resistance to adversarial attacks.

We also present the results with deep ensembles of variance networks (denoted variance ensembles) and show that these two techniques can be efficiently combined to improve the robustness of the network even further.

In this paper we introduce variance networks, surprisingly stable stochastic neural networks that learn only the variances of the weights, while keeping the means fixed at zero in one or several layers.

We show that such networks can still be trained well and match the performance of conventional models.

Variance networks are more stable against adversarial attacks than conventional ensembling techniques, and can lead to better exploration in reinforcement learning tasks.

The success of variance networks raises several counter-intuitive implications about the training of deep neural networks:• DNNs not only can withstand an extreme amount of noise during training, but can actually store information using only the variances of this noise.

The fact that all samples from such zero-centered posterior yield approximately the same accuracy also provides additional evidence that the landscape of the loss function is much more complicated than was considered earlier BID6 ).•

A popular trick, replacing some random variables in the network with their expected values, can lead to an arbitrarily large degradation of accuracy -up to a random guess quality prediction.• Previous works used the signal-to-noise ratio of the weights or the layer output to prune excessive units BID1 ; Molchanov et al. (2017); Neklyudov et al. (2017) ).

However, we show that in a similar model weights or even a whole layer with an exactly zero SNR (due to the zero mean output) can be crucial for prediction and can't be pruned by SNR only.• We show that a more flexible parameterization of the approximate posterior does not necessarily yield a better value of the variational lower bound, and consequently does not necessarily approximate the posterior distribution better.

We believe that variance networks may provide new insights on how neural networks learn from data as well as give new tools for building better deep models.

A PROOF OF THEOREM 1 DISPLAYFORM0 t,i ) in terms of Maximum Mean Discrepancy: DISPLAYFORM1 Proof.

By the definition of the Maximum Mean Discrepancy, we have DISPLAYFORM2 where the supremum is taken over the set of continuous functions, bounded by 1.

Let's reparameterize and join the expectations: DISPLAYFORM3 (18) Since linear transformations of the argument do not change neither the norm of the function, nor its continuity, we can hide the component-wise multiplication of ε by √ α t µ t inside the function f (ε).This would not change the supremum.

DISPLAYFORM4 There exists a rotation matrix R such that R( DISPLAYFORM5 αt , 0, . . . , 0) .

As ε comes from an isotropic Gaussian ε ∼ N (0, I D ), its rotation Rε would follow the same distribution Rε ∼ N (0, I D ).

Once again, we can incorporate this rotation into the function f without affecting the supremum.

DISPLAYFORM6 Let's consider the integration over ε 1 separately (φ(ε 1 ) denotes the density of the standard Gaussian distribution): DISPLAYFORM7 Next, we view f (ε 1 , . . . ) as a function of ε 1 and denote its antiderivative as F 1 (ε) = f (ε)dε 1 .

Note that as f is bounded by 1, hence F 1 is Lipschitz in ε 1 with a Lipschitz constant L = 1.

It would allow us to bound its deviation DISPLAYFORM8 Let's use integration by parts: DISPLAYFORM9 The first term is equal to zero, as DISPLAYFORM10 Finally, we can use the Lipschitz property of F 1 (ε) to bound this value: DISPLAYFORM11 Thus, we obtain the following bound on the MMD: DISPLAYFORM12 This bound goes to zero as α t goes to infinity.

As the output of a softmax network lies in the interval [0, 1], we obtain the following bound on the deviation of the prediction of the ensemble after applying the zero-mean approximation: Figure 8 : These are the learning curves for VGG-like architectures, trained on CIFAR-10 with layerwise parameterization and with different prior distributions.

These plots show that all three priors are equivalent in practice: all three models converge to variance networks.

The convergence for the Student's prior is slower, because in this case the KL-term is estimated using one-sample MC estimate.

This makes the stochastic gradient w.r.t.

log α very noisy when α is large.

DISPLAYFORM13

We have considered the following setting.

We used a LeNet-5 network on the MNIST dataset with only tanh non-linearities and with no biases.

Note that it works well even if the second-to-last layer is a variance layer.

It means that the zeromean variance-only encodings are robustly discriminated using a linear model.

Figure 9: Average distributions of activations of a variance layer for objects from different classes for four random neurons.

Each line corresponds to an average distribution, and the filled areas correspond to the standard deviations of these p.d.f.s.

Each neuron essentially has several "energy levels", one for each class / a group of classes.

On one "energy level" the samples have roughly the same average magnitude, and samples with different magnitudes can easily be told apart with successive layers of the neural network.

Here we provide the expressions for the forward pass through a fully-connected and a convolutional variance layer with different parameterizations.

Fully-connected layer, q(w ij ) = N (µ ij , α ij µ 2 ij ): DISPLAYFORM0 Fully-connected layers, q(w ij ) = N (0, σ 2 ij ): DISPLAYFORM1 For fully-connected layers ε j ∼ N (0, 1) and all variables mentioned above are scalars.

Convolutional layer, q(w ijhw ) = N (µ ijhw , α ijhw µ 2 ijhw ): DISPLAYFORM2 Convolutional layers, q(w ijhw ) = N (0, σ 2 ijhw ): DISPLAYFORM3 In the last two equations denotes the component-wise multiplication, denotes the convolution operation, and the square and square root operations are component-wise.

ε jhw ∼ N (0, 1).

All variables b j , µ i , A i , σ i are 3D tensors.

For all layers ε is sampled independently for each object in a mini-batch.

The optimization is performed w.r.t.

µ, log α or w.r.t.

log σ, depending on the parameterization.

We show below that the KL divergence D KL (N (0, σ 2 ) LogU) is constant w.r.t.

σ.

DISPLAYFORM0 ∝ − 1 2 log 2πeσ 2 − E w∼N (0,σ 2 ) log 1 |x| == − 1 2 log 2πeσ 2 + E ε∼N (0,1) log |σε| == − 1 2 log 2πe + E ε∼N (0,1)

log |ε|

@highlight

It is possible to learn a zero-centered Gaussian distribution over the weights of a neural network by learning only variances, and it works surprisingly well.

@highlight

This paper investigates the effects of mean of variational posterior and proposes variance layer, which only uses variance to store information

@highlight

Studies variance neural networks which approximate the posterior of Bayesian neural networks with zero-mean Gaussian distributions