Bayesian neural networks, which both use the negative log-likelihood loss function and average their predictions using a learned posterior over the parameters, have been used successfully across many scientific fields, partly due to their ability to `effortlessly' extract desired representations from many large-scale datasets.

However, generalization bounds for this setting is still missing.

In this paper, we present a new PAC-Bayesian generalization bound for the negative log-likelihood loss which utilizes the \emph{Herbst Argument} for the log-Sobolev inequality to bound the moment generating function of the learners risk.

Deep neural networks are ubiquitous across disciplines and often achieve state of the art results (e.g., Krizhevsky et al. (2012) ; Simonyan & Zisserman (2014) ; He et al. (2016) ).

Albeit neural networks are able to encode highly complex input-output relations, in practice, they do not tend to overfit (Zhang et al., 2016) .

This tendency to not overfit has been investigated in numerous works on generalization bounds (Langford & Shawe-Taylor, 2002; Langford & Caruana, 2002; Bartlett et al., 2017a; 2019; McAllester, 2003; Germain et al., 2016; Dziugaite & Roy, 2017) .

Indeed, many generalization bounds apply to neural networks.

However, most of these bounds assume that the loss function is bounded (Bartlett et al., 2017a; Neyshabur et al., 2017; Dziugaite & Roy, 2017) .

Unfortunately, this assumption excludes the popular negative log-likelihood (NLL) loss, which is instrumental to Bayesian neural networks that have been used extensively to calibrate model performance and provide uncertainty measures to the model prediction.

In this work we introduce a new PAC-Bayesian generalization bound for NLL loss of deep neural networks.

Our work utilizes the Herbst argument for the logarithmic-Sobolev inequality (Ledoux, 1999) in order to bound the moment-generating function of the model risk.

Broadly, our PACBayesian bound is comprised of two terms: The first term is dominated by the norm of the gradients with respect to the input and it describes the expressivity of the model over the prior distribution.

The second term is the KL-divergence between the learned posterior and the prior, and it measures the complexity of the learning process.

In contrast, bounds for linear models or bounded loss functions lack the term that corresponds to the expressivity of the model over the prior distribution and therefore are the same when applied to shallow and deep models.

We empirically show that our PAC-Bayesian bound is tightest when we learn the mean and variance of each parameter separately, as suggested by Blundell et al. (2015) in the context of Bayesian neural networks (BNNs).

We also show that the proposed bound holds different insights regarding model architecture, optimization and prior distribution selection.

We demonstrate that such optimization minimizes the gap between risk and the empirical risk compared to the standard Bernoulli dropout and other Bayesian inference approximation while being consistent with the theoretical findings.

Additionally, we explore in-distribution and out-of-distribution examples to show that such optimization produces better uncertainty estimates than the baseline.

PAC-Bayesian bounds for the NLL loss function are intimately related to learning Bayesian inference (Germain et al., 2016) .

Recently many works applied various posteriors in Bayesian neural networks.

Gal & Ghahramani (2015) ; Gal (2016) introduce a Bayesian inference approximation using Monte Carlo (MC) dropout, which approximates a Gaussian posterior using Bernoulli dropout.

Srivastava et al. (2014) introduced Gaussian dropout which effectively creates a Gaussian posterior that couples between the mean and the variance of the learned parameters.

Kingma et al. (2015) explored the relation of this posterior to log-uniform priors, while Blundell et al. (2015) suggests to take a full Bayesian perspective and learn separately the mean and the variance of each parameter.

Our work uses the bridge between PAC-Bayesian bounds and Bayesian inference, as described by Germain et al. (2016) , to find the optimal prior parameters in PAC-Bayesian setting and apply it in the Bayesian setting.

Most of the literature regarding Bayesian modeling involves around a two-step formalism (Bernardo & Smith, 2009) : (1) a prior is specified for the parameters of the deep net; (2) given the training data, the posterior distribution over the parameters is computed and used to quantify predictive uncertainty.

Since exact Bayesian inference is computationally intractable for neural networks, approximations are used, including MacKay (1992); Hernández-Lobato & Adams (2015); Hasenclever et al. (2017); Balan et al. (2015) ; Springenberg et al. (2016) .

In this study we follow this two-step formalism, particularly we follow a similar approach to Blundell et al. (2015) in which we learn the mean and standard deviation for each parameter of the model using variational Bayesian practice.

Our experimental validation emphasizes the importance of learning both the mean and the variance.

Generalization bounds provide statistical guarantees on learning algorithms.

They measure how the learned parameters w perform on test data given their performance on the training data S = {(x 1 , y 1 ), . . .

, (x m , y m )}, where x i is the data instance and y i is its corresponding label.

The performance of the learning algorithm is measured by a loss function (w, x, y).

The risk of a learner is its average loss, when the data instance and its label are sampled from their true but unknown distribution D. We denote the risk by L D (w) = E (x,y)∼D (w, x, y).

The empirical risk is the average training set loss L S (w) = 1 m m i=1 (w, x i , y i ).

PAC-Bayesian theory bounds the risk of a learner E w∼q L D (w) when the parameters are averaged over the learned posterior distribution q. The parameters of the posterior distribution are learned from the training data S. In our work we focus on the following PAC-Bayesian bound: Theorem 1 (Alquier et al. (2016) ).

Let KL(q||p) = q(w) log(q(w)/p(w))dw be the KLdivergence between two probability density functions p, q. For any λ > 0 and for any δ ∈ (0, 1] and for any prior distribution p, with probability at least 1 − δ over the draw of the training set S, the following holds simultaneously for any posterior distribution q:

PAC-Bayesian theory is intimately connected to Bayesian inference when considering the negative log-likelihood loss function (w, x, y) = − log p(y|x, w) and λ = m. Germain et al. (2016) proved that the optimal posterior in this setting is q(w) = p(w|S).

Bayesian inference considers the posterior p(y|x, S) = p(w|S)p(y|x, w)dw, at test time for a data instance x, which corresponds to the risk of the optimal posterior.

Unfortunately, the optimal posterior is rarely available, and PAC-Bayes relies on the approximated posterior q.

Coincidently, the approximated posterior and its KL-divergence from the prior distribution are instrumental to the evidence lower bound (ELBO), which is extensively used in Bayesian neural networks (BNNs) to bound the log-likelihood

While the right hand side of a PAC-Bayesian bound, with the negative log-likelihood loss and λ = m, is identical to the right hand side of the ELBO bound in term of learning, they serve different purposes.

One is used for bounding the risk while the other is used for bounding the marginal loglikelihood.

Nevertheless, the same algorithms can be used to optimize BNNs and PAC-Bayesian intuitions and components can influence the practice of Bayesian neural networks.

It is challenging to derive a PAC-Bayesian bound for the negative log-likelihood (NLL) loss as it requires a bound on the log-partition function log E w∼p,

In cases where the loss function is uniformly bounded by a constant, e.g., the zero-one loss, the log-partition function is bounded as well.

Unfortunately, the NLL loss is unbounded, even when y is discrete.

For instance, consider fully connected case, where the input vector of the (k)-th layer is a function of the parameters of all previous layers, i.e., x k (W 0 , . . .

, W k−1 ).

The entries of x k are computed from the response of its preceding layer, i.e., W k−1 x k , followed by a transfer function σ(·), i.e.,

, if the rows in W k consist of the vector rx k then the NLL loss increases with r, and is unbounded when r → ∞.

Our main theorem shows that for smooth loss functions, the log-partition function is bounded by the expansion of the loss function, i.e., the norm of its gradient with respect to the data x. This property is appealing since these gradients often decay rapidly for deep neural networks, as we demonstrate in our experimental evaluation.

Consequently deep networks enjoy tighter generalization bounds than shallow networks.

Our proof technique follows the Herbst Argument for bounding the log-partition function using the Log-Sobolev inequality for Gaussian distributions (Ledoux, 2001) .

Theorem 2.

Assume (x, y) ∼ D and x given y follows the Gaussian distribution.

Let (w, x, y) be a smooth loss function (e.g., the negative log-likelihood loss).

For any δ ∈ (0, 1] and for any real number λ > 0, with probability at least 1 − δ over the draw of the training set S the following holds simultaneously for any posterior probability density function:

The Gaussian assumption for the data generating distribution D can be relaxed to any log-concave distribution, using Gentil (2005) , Corollary 2.5.

We use the Gaussian assumption to avoid notational overhead.

Broadly, the proposed bound is comprised of two terms: The first term is the log-partition function which is dominated by the norm of the gradients with respect to the input, namely E (x,ŷ)∼D e α(− (w,x,ŷ)) dα , and it describes the expressivity of the model over the prior distribution.

The second term is the KL-divergence between the learned posterior and the prior, and it measures the complexity of the learning process.

The proof starts with Eq. (1) and uses the Herbst Argument and the Log-Sobolev inequality to bound the moment-generating function

Specifically, the proof consists of three steps.

First we use the statistical independence of the training samples to decompose the moment generating function

Then we use the Herbst argument to bound the function M (

]

and obtain the following bound:

Finally we use the log-Sobolev inequality for Guassian distributions,

The above theorem can be extended to settings for which x is sampled from any log-concave distribution, e.g., the Laplace distribution.

The log-concave setting modifies the gradient norm and the log-Sobolev constant 2 in Eq. (6) that corresponds to Gaussian distributions, cf.

Gentil (2005) .

We avoid this generalization to simplify our mathematical derivations.

A detailed description of the proof can be found on Section 8.1 in the Appendix.

The bound in Theorem 2 is favorable when applied to deep networks since their gradients w.r.t.

data often decay rapidly.

Nevertheless we can also apply our technique to shallow nets trained with NLL loss.

We obtain PAC-Bayesian bounds for multi-class logistic regression.

The NLL loss for multiclass logistic regression takes the form:

, where x ∈ R d is the data instance, y ∈ {1, . . .

, k} are the possible labels, and W ∈ R k×d is the matrix of parameters.

The bound in Theorem 2 takes the form:

given y follows the Gaussian distribution.

Let (w, x, y) = − log p(y|x, w) be the negative log-likelihood loss for k−class logistic regression.

For any δ ∈ (0, 1], for any λ > 0 and for any prior density function with variance σ 2 p ≤ m/16λ 2 , with probability at least 1 − δ over the draw of the training set S the following holds simultaneously for any posterior probability density function:

Full proof can be found on Section 8.2 in the Appendix, while we sketch the main steps of the proof below.

The above corollary shows that PAC-Bayesian bound for classification using the NLL loss can achieve rate of λ = m. This result augments the PAC-Bayesian for regression using the NLL loss for regression, i.e., the square loss, of Germain et al. (2016) .

The PAC-Bayesian bound for logistic regression is derived by applying Theorem 2.

We begin by realizing the gradient of log p(y|x, w) with respect to x. We denote by w y the y−th row of the parameter matrix W .

Thus ∇ x log p(y|w, x) = ŷ p(ŷ|x, w)(w y − wŷ), and the gradient norm is upper bounded as follows: ∇ x log p(y|w, x) 2 ≤ 2 y w y 2 .

Plugging this result into Eq. (18) we obtain the following bound:

Finally, whenever λσ p ≤ m/8 we derive the bound

A detailed description of the proof can be found on Section 8.2 in the Appendix.

In this section we study the derived bound empirically.

We start with an ablation study of the proposed bound using classification and regression models.

Next, we present our results for multiclass classification tasks using different datasets and different architectures.

We conclude the section, with an analysis of the models' uncertainty estimates using for in-distribution examples and outof-distribution examples.

All suggested models follows the a Bayesian Neural Networks (BNN) perspective, in which we learn the mean and standard deviation for each learnable parameter in the network where we define N (0, σ 2 p I) to be the prior over weights.

6.1 ABLATION Effect of σ p .

We start by exploring the effect of σ p on the models' performance and the proposed generalization bound.

For that, we trained several models using σ p ∈ {0.05, 0.1, 0.2, 0.3} using the MNIST (LeCun & Cortes, 2010) and Fashioin-MNIST Xiao et al. (2017) datasets.

All results were obtained using fully connected layers with ReLU as non-linear activation function.

We optimized the NLL loss function using Stochastic Gradient Descent (SGD) for 50 epochs with a learning rate of 0.01 and momentum of 0.9.

For each model we compute the average train and test loss and accuracy together with the absolute difference between the training loss and the test loss, denoted as Generalization Loss.

Moreover, we compute the generalization bound as stated in Eq. (18) for all settings.

Results are summarized in Table 1 .

Although σ p = 0.2 reaches slightly better generalization bound on MNIST dataset, σ p = 0.1 performs better over all calculated metrics, i.e., average loss and accuracy, both on MNIST and Fashion-MNIST.

Notice, for Fashion-MNIST we observed slightly better generalization gap while using σ p = 0.05, however, its loss and accuracy are worse comparing to σ p = 0.1.

Effect of λ.

Recall, we bound the moment generating function using the norm of the functions' gradient with respect to the data x (Eq. (18)).

To construct tighter generalization bounds, we would like to set λ → m. However, in Eq. (18) λ appears in both numerator and denominator.

It is hence not clear whether the bound will converge, which depends on the model architecture, which is represented by the norm of its gradient.

In other words, models with lower gradient norm could benefit from larger values of λ, hence tighter generalization bounds.

To further explore this property we trained five different models with different number of layers (1-5).

We look into both classification models while optimizing the NLL loss function, and regression tasks while optimizing the Mean Squared Error (MSE) loss function.

For classification we used MNIST and Fashion-MNIST datasets, while for regression we use the Boston Housing dataset (for the regression models, results were obtained using 5-fold cross validation).

Except for the linear models, we force all models to have roughly the same numbers of parameters (∼80K for MNIST, ∼800K for Fashion-MNIST, ∼1500 for regression).

For all models we set ReLU as non-linear activation functions.

We optimize all models for 50 epochs using SGD with learning rate of 0.01 and momentum of 0.9.

Based on results of the prior paragraph, in all reported settings we set σ p = 0.1.

Results are reported in Table 2 .

It can be seen that deeper models produce tighter generalization bounds on all three datasets.

When considering model performance on down-stream classification task we notice that in general, models with better generalization bounds perform slightly better in terms of loss and accuracy.

One possible explanation is that deeper models have smaller gradients w.r.t.

the input.

To validate that we further computed the average squared gradient norm w.r.t.

the input as a function of the model depth, for both MNIST and Fashion-MNIST datasets.

It can be seen from Figure 1a that indeed the gradients decay rapidly as we add more layers to the network.

Next, we present in Figure 1b the generalization bound as a function of λ for MNIST models.

We explored λ ∈ [ √ m, m] and stopped the plot once the bound can no longer be computed.

Experiments using Fashion-MNIST produce similar plot and can be found on Section 8.7 in the Appendix.

Weights visualization.

Since we consider Bayesian Neural Networks (BNNs) and optimize the KLdivergence between the prior and the posterior over the weights, we can visualize the average mean and standard deviation (STD) of the posterior as a function of the model depth.

Figure 2a presents this for MNIST and Fashion-MNIST models using four and five depth levels.

As expected, we can see that the average mean over the weights is zero for all layers while weights STD approaches 0.1.

For the MNIST models (Figure 2a top row) , we observed the standard deviation are ∼0.7 and not 0.1.

We suspect this behaviour is due to fast optimization, hence the models do not have much signal to push the model towards the prior distribution.

Notice, in all settings the average STD of the model weights decreases on the last layer.

We observed a similar behavior also for the other models.

6.2 CLASSIFICATION Next, we compare BNN models against two commonly used baselines.

The first baseline is a softmax model using the same architecture as the BNN while adding dropout layers.

The second base- line is a Bayesian approximation using Monte Carlo Dropout (Gal & Ghahramani, 2015) , denoted as MC-Dropout, using different dropout rates and weight decay value of 1e-5.

To evaluate these approaches we conducted multi-class classification experiments using three classification benchmarks: MNIST, Fashion-MNIST, and CIFAR-10 ( Krizhevsky & Hinton, 2009 ).

We report train set and test set loss and accuracy, together with their generalization gaps (e.g., the difference between the test and training loss and accuracy).

Notice, as oppose to Dziugaite & Roy (2017) our results are reported for multi-class classification and not for binary classification.

For completeness, we report binary classification results on Section 8.4 in the Appendix.

The premise beyond these type of experiments is to preset the benefit of learning the mean and STD separately for each of the models' parameters.

Results are reported in Table 7 .

For BNN and MC-Dropout models, we sample 20 times from the posterior distribution and average their outputs to produce the model output.

We also sampled more times, however, we did not see any significant differences.

We observe that BNN models achieve comparable results to both baselines but with lower loss and accuracy generalization gaps.

Throughout the experiments we use dropout value of 0.3 for Softmax and MC-Dropout models and σ p = 0.1 for BNN models.

We chose these values after grid search over different dropout values for all baseline models.

A detailed description of all implementation details together with results for more dropout rates can be found on Sections 8.4, 8.5, and 8.3 in the Appendix.

Lastly, we evaluated the uncertainty estimates of BNN models against softmax models and MCDropout models.

We experimented with both in-distribution and out-of-distribution examples.

The purpose of the following experiments is to demonstrate that following the Bayesian approach together with the carefully picked prior can lead to better uncertainty estimates.

In-Distribution Examples.

In the context of in-distribution examples we follow the suggestion of Guo et al. (2017) and calculate the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) for all three models.

Figure 2b provides visual representation of the results.

Results suggest that BNNs produce better calibrated outputs for all settings, with two exception of ECE for MNIST and MCE for CIFAR10.

Out-of-Distribution Examples.

Next, we evaluated the uncertainty estimates using OOD examples.

We apply a model trained using dataset A to OOD examples from dataset B. We trained models on MNIST, Fashion-MNIST and CIFAR-10 and assess prediction confidence using OOD examples from MNIST, Fashion-MNIST, NotMNIST (Cohen et al., 2017) , and SVHN (Netzer et al., 2011) .

Results are summarized in Table 8 .

More OOD experiments using different dropout and prior rates can be found on Sections 8.3, 8.6 in the Appendix.

All models performed at the chance level (∼ 10% for 10 classes) for both OOD train and test sets.

When considering the loss, we observe significantly higher values for the softmax and MC-Dropout models.

These two findings imply that the softmax and MC-Dropout models are overly confident and tend to output a high probability for the max label.

Hence, we measure the average entropy for all models.

We expect BNNs to have higher entropy, due to the fact that it produces better uncertainty estimates, i.e., its' predictions for OOD samples are closer to a uniform distribution.

Indeed, results reported in Table 8 confirm this intuition.

In the following study we present a new PAC-Bayesian generalization bound for learning a deep net using the NLL loss function.

The proof relies on bounding the log-partition function using the squared norm of the gradients with respect to the input.

Experimental validation shows that the resulting bound provides insight for better model optimization and prior distribution search.

We demonstrate that learning the mean and STD for all parameters together with optimize prior over the parameters leads to better uncertainty estimates over the baselines and makes it harder to overfit.

Proof.

We begin by using the statistical independence of the training samples to decompose the following function:

Next we represent the moment generating function M (

where the last equality follows from a change of integration variable and the integral limits.

K (α) refers to the derivative at α.

We then compute K (α) and K(0):

Concluding the Herbst argument we obtain the following equality:

Combining Eq. (12) with Eq. (15) we derive:

Finally we apply the log-Sobolev inequality for Gaussian distributions (cf.

Ledoux (2001) , Chapter 2), as described in Eq. (6).

To complete the proof we combine Eq. (6) with Eq. (17) to obtain:

Proof.

To apply Theorem 2 we start by realizing the gradient of log p(y|x, w) with respect to x. We denote by w y the y−th row of the parameter matrix W .

Thus ∇ x log p(y|w, x) = ŷ p(ŷ|x, w)(w y − wŷ).

Using the convexity of the norm function we upper bound the gradient norm:

Next we use the fact that the gradient norm upper bound is independent of x to simplify the moment generating function bound in Theorem 2.

Since (w, x, y) = − log p(ŷ|x, w), we use the bound in Eq. (20):

Thus we are able to simplify Theorem 2 as follows

Finally, we recall that p is the prior density function N (0, σ 2 p ).

Since the parameters are statistically independent, this expectation decomposes to its kd parameters:

And the result follows from the fact

for

√ 2 and the result follows.

The architechtures described in this sub-section are used for the multi-class, binary, and uncertainty estimates experiments.

We use multilayer perceptrons for the MNIST dataset, while we use convolutional neural networks (CNNs) for both Fashion-MNIST and CIFAR-10.

A detailed description of the architectures is available in Table 5 .

We optimize the NLL loss function using SGD with a learning rate of 0.01 and a momentum value of 0.9 in all settings.

We use mini-batches of size 128 and did not use any learning rate scheduling.

For the MC-Dropout models we experienced with different weight decay values, however found that 1e-5 provides the best validation loss, hence choose this value. (10) 8.4 BINARY CLASSIFICATION Experiments in this sub-section were conducted to show consistency with Dziugaite & Roy (2017) .

We follow the same setting in which we use the MNIST dataset, where we group digits [0, 1, 2, 3, 4] into label zero, and labels [5, 6, 7, 8, 9] into label one.

All experiments in this subsection were conducted using multilayer perceptrons with one hidden layer consisting of 300 hidden neurons.

We use the Rectified Linear Unit (ReLU) as our activation function (Glorot et al., 2011) .

We optimize the negative log-likelihood loss function using stochastic gradient descent (SGD) with a learning rate of 0.1 and a momentum value of 0.9.

We did not use any learning rate scheduling.

SGD is run in mini-batches of size 128.

Each model was trained for 20 epochs.

We compared BNN to softmax models with dropout (Srivastava et al., 2014) rates chosen from the set {0.0, 0.3, 0.5}. Hereby, a dropout with a rate of 0.0 means no dropout at all.

In addition to the training set and test set loss and accuracy, we measure the generalization loss, while setting L D (w) to be the average test set loss.

In the same manner, we measure the generalization accuracy, while using the zero-one loss instead of the negative log-likelihood loss.

Table 6 summarizes the results.

All models achieve comparable accuracy levels, however the softmax models suffer from larger generalization errors both in terms of loss and accuracy.

Notice, as expected, using higher Bernoulli-dropout rates mitigates the generalization gap.

Here we report results for multi-class classification for BNN and the baselines.

Table 7 summarizes the results.

The main purpose of these additional experiments is to explore more dropout and σ p values for different models.

Figure 3 : Analysis of the proposed bound as a function network depth.

We report the the generalization bound as a function of λ for different deep net depth levels using the Fashion-MNIST dataset.

@highlight

We derive a new PAC-Bayesian Bound for unbounded loss functions (e.g. Negative Log-Likelihood). 