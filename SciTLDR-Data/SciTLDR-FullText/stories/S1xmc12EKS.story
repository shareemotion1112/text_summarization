The neural linear model is a simple adaptive Bayesian linear regression method that has recently been used in a number of problems ranging from Bayesian optimization to reinforcement learning.

Despite its apparent successes in these settings, to the best of our knowledge there has been no systematic exploration of its capabilities on simple regression tasks.

In this work we characterize these on the UCI datasets, a popular benchmark for Bayesian regression models, as well as on the recently introduced ''gap'' datasets, which are better tests of out-of-distribution uncertainty.

We demonstrate that the neural linear model is a simple method that shows competitive performance on these tasks.

Despite the recent successes that neural networks have shown in an impressive range of tasks, they tend to be overconfident in their predictions (Guo et al., 2017) .

Bayesian neural networks (BNNs; Neal (1995) ) attempt to address this by providing a principled framework for uncertainty estimation in predictions.

However, inference in BNNs is intractable to compute, requiring approximate inference techniques.

Of these, Monte Carlo methods and variational methods, including Monte Carlo dropout (MCD) (Gal and Ghahramani, 2016) , are popular; however, the former are difficult to tune, and the latter are often limited in their expressiveness (Foong et al., 2019b; Yao et al., 2019; Foong et al., 2019a) .

The neural linear model represents a compromise between tractability and expressiveness for BNNs in regression settings: instead of attempting to perform approximate inference over the entire set of weights, it performs exact inference on only the last layer, where prediction can be done in closed form.

It has recently been used in active learning (Pinsler et al., 2019) , Bayesian optimization (Snoek et al., 2015) , reinforcement learning (Riquelme et al., 2018) , and AutoML (Zhou and Precioso, 2019), among others; however, to the best of our knowledge, there has been no systematic attempt to benchmark the model in the simple regression setting.

In this work we do so, first demonstrating the model on a toy example, followed by experiments on the popular UCI datasets (as in Hernández-Lobato and Adams (2015) ) and the recent UCI gap datasets from Foong et al. (2019b) , who identified (along with Yao et al. (2019) ) well-calibrated 'in-between' uncertainty as a desirable feature of BNNs.

In this section, we briefly describe the different models we train in this work, which are variations of the neural linear (NL) model, in which a neural network extracts features from the input to be used as basis functions for Bayesian linear regression.

The central issue in the neural linear model is how to train the network: in this work, we provide three different models, with a total of four different training methods.

For a more complete mathematical description of the models, refer to Appendix A; we summarize the models in Appendix C. Snoek et al. (2015) , we can first train the neural network using maximum a posteriori (MAP) estimation.

After this training phase, the outputs of the last hidden layer of the network are used as the features for Bayesian linear regression.

To reduce overfitting, the noise variance and prior variance (for the Bayesian linear regression) are subsequently marginalized out by slice sampling (Neal et al., 2003) according to the tractable marginal likelihood, using uniform priors.

We refer to this model as the maximum a posteriori neural linear model (which we abbreviate as MAP-L NL, where L is the number of hidden layers in the network).

We tune the hyperparameters for the MAP estimation via Bayesian optimization (Snoek et al., 2012) .

Regularized NL The MAP NL model's basis functions are learned independently of the final model's predictions.

This is an issue for uncertainty quantification, as MAP training has no incentive to learn features useful for providing uncertainty in out-of-distribution areas.

To address this issue, we propose to learn the features by optimizing the (tractable) marginal likelihood with respect to the network weights (previous to the output layer), treating them as hyperparameters of the model in an approach analogous to hyperparameter optimization in Gaussian process (GP) regression (Rasmussen and Williams, 2006) .

However, unlike in GP regression, the per-iteration computational cost of this method is linear in the size of the data.

We additionally regularize the weights to reduce overfitting, resulting in a model we call regularized neural linear (which we abbreviate as Reg-L NL).

As in the MAP NL model, we marginalize out the noise and prior variances via slice sampling.

We tune the regularization and other hyperparameters via Bayesian optimization.

Bayesian noise NL Instead of using slice sampling for the noise variance, we can place a normal-inverse-gamma (N-Γ −1 ) prior on the weights and noise variance.

This formulation is still tractable, and integrates the marginalization of the noise variance into the model itself, rather than having it implemented after the features are learned.

Additionally, the N-Γ −1 prior can act as a regularizer, meaning that we can avoid using Bayesian optimization to tune the prior parameters by jointly optimizing the marginal likelihood over all hyperparameters.

However, this risks overfitting.

Therefore, we consider training this model, which we call the Bayesian noise (BN) neural linear model, both by maximizing the marginal likelihood for all parameters (including prior parameters), and by tuning the prior parameters with Bayesian optimization.

We abbreviate the first as BN(ML)-L NL and the second as BN(BO)-L NL.

Finally, in both cases we slice sample the remaining (non-weight) hyperparameters.

We compare these models on a toy problem, the UCI datasets, and the UCI "gap" datasets (Foong et al., 2019b) .

In all experiments, we consider 1-and 2-layer ReLU fullyconnected networks with 50 hidden units in each layer (except for the toy problem, where we only consider 2-layer networks).

We also provide results for simple MAP inference as a baseline.

For experimental details, refer to Appendix B.

We provide additional experimental results, including detailed statistical comparisons of the models, in Appendix D.

Toy problem We construct a synthetic 1-D dataset comprising 100 train and 100 test pairs (x, y), where x is sampled i.i.d.

in the range [−4, −2] ∪ [2, 4] and y is generated as y = x 3 + , ∼ N (0, 9).

This follows the example from Hernández-Lobato and Adams (2015) , with the exception of the "gap" added in the range for x, which was motivated by Foong et al. (2019b) and Yao et al. (2019) .

We plot predictive distributions for each model in Figure 1 .

Somewhat surprisingly, the MAP-2 NL model seems to struggle more than MAP with uncertainty in the gap, while having better uncertainty quantification at the edges.

Of the marginal likelihood-based methods, the BN(BO)-2 NL model qualitatively seems to perform the best.

UCI datasets We next provide results on the UCI datasets in Hernández-Lobato and Adams (2015) (omitting the 'year' dataset due to its size), a popular benchmark for Bayesian regression models in recent years.

We report average test log likelihoods and RMSEs for all the models in Appendix D.1, for both 1-and 2-layer architectures.

We visualize average test log likelihoods for the models in Figure 2 ; we tabulate the log likelihoods and RMSEs in Tables 2 and 3 in Appendix D.1, respectively.

From the figure and tables, we see that the BN(ML)-2 NL and BN(BO)-2 NL models have the best performance on these metrics, with reasonable log likelihoods and RMSEs compared to those in the literature for other BNN-based methods (Hernández-Lobato and Adams, 2015; Gal and Ghahramani, 2016; Bui et al., 2016; Tomczak et al.) .

In fact, these neural linear methods tend to achieve state-of-the-art or near state-of-the-art neural network performance on the 'energy' and 'naval' datasets.

While the performance of the Reg-L NL model is decent, it performs worse than the BN-L NL models, showing the advantage of a Bayesian treatment of the noise variance.

UCI gap datasets Finally, we provide results on the UCI "gap" datasets proposed by Foong et al. (2019b) , which consists of training and testing splits that artificially contain gaps in the training set, ensuring that the model will only succeed if it can represent uncertainty in-between gaps in the data.

We again visualize test log likelihoods in Figure 3 while tabulating log likelihoods and RMSEs in Tables 6 and 7 in Appendix D.1.

Our results on the MAP-based models in Figure 3 echo those of Foong et al. (2019b) , showing catastrophic failure to express in-between uncertainty for some datasets (particularly 'energy' and 'naval').

Somewhat surprisingly, the Reg-L NL models perform the worst of all the models.

However, the BN NL models do not seem to fail catastrophically, with the BN(BO)-2 NL model having by far the best performance.

In all of these models we used some form of hyperparameter tuning (Bayesian optimization for all models except the BN(ML)-L NL models, where we used a grid search) to obtain the results shown.

However, for the practitioner, performing an oftentimes costly hyperparameter search is not desirable, particularly where one of the main motivations for using the model is its simplicity, as in this case.

We therefore investigate the effect of the hyperparameter tuning on the models' performance.

Figure 4 shows the difference in average test log likelihoods and test RMSEs between the tuned models and models whose hyperparameters were set to "reasonable" values that a practitioner might choose by intuition (see Appendix D.2 for details) for the UCI datasets.

We observe that for each of the two-layer models there exists at least one dataset where the performance in terms of test log likelihood is significantly worsened by omitting hyperparameter tuning.

The performance difference for RMSEs is not as drastic, although it still exists.

In Appendix D.2 we show that these results extend to the UCI gap datasets and that the difference in performance is statistically significant for nearly all models across both the UCI and UCI gap datasets, for both log likelihood and RMSE performance.

Finally, in Appendix D.2.1 we show that mean field variational inference (MFVI) (Graves, 2011; Hinton and Van Camp, 1993; Blundell et al., 2015) and MCD can still obtain reasonable, although not state-of-the-art, performance on the UCI datasets without hyperparameter tuning: in many cases the performance is even competitive with the tuned NL models.

However, these suffer from the pathologies identified in Foong et al. (2019b); Yao et al. (2019); Foong et al. (2019a) on the gap datasets.

We have shown benchmark results for different variants of the neural linear model in the regression setting.

Our results show that the successes these models have seen in other areas such as reinforcement and active learning are not unmerited, with the models achieving generally good performance despite their simplicity.

Furthermore, they are not as susceptible to the the inability to express gap uncertainty as MFVI or MCD.

However, we have shown that to obtain reasonable performance extensive hyperparameter tuning is often required, unlike MFVI or MCD.

Finally, our work suggests that exact inference on a subset of parameters can perform better than approximate inference on the entire set, at least for BNNs.

We believe this broader issue is worthy of further investigation.

The neural linear model uses a neural network to parameterize basis functions for Bayesian linear regression by treating the output weights and bias of the network probabilistically, while treating the rest of the network's parameters θ as hyperparameters.

This can be used as an approximation to full Bayesian inference of the neural network's parameters, with the main advantage being that this simplified case is tractable (assuming Gaussian prior and likelihood).

Given the fact that there are significant redundancies in the weight-space posterior for BNNs, this tradeoff may not be a completely unreasonable approximation.

We now describe the model mathematically.

, where (x n , y n ) ∈ R d × R, be the training data, and let

T represent the outputs (post-activations) of the last hidden layer of the neural network, which will be parameterized by all the weights and biases up to the last layer, θ.

We then define a weight vector w ∈ R M = R N L +1 (this includes a bias term, augmenting φ θ (x) with a 1).

If we define a design matrix Φ θ = [φ θ (x 1 ), . . .

, φ θ (x N )] T , we can then define our model as

where we treat Y as a column vector of the y n .

Given an appropriate θ, Bayesian inference of the weights w is straightforward: given a prior p(w) = N (w; 0, αI M ) on the weights, the posterior is given by

The posterior predictive for a test input x * is then given by

It now remains to be determined how to learn θ.

As described in Snoek et al. (2015), we can learn θ by simply setting it to the values of the corresponding weights and biases in a maximum a posteriori (MAP)-trained network, maximizing the objective

with respect to θ F ull and σ 2 , where θ F ull represents the parameters of the full network (which includes the output weights and bias), and γ is a regularization parameter.

As in Snoek et al. (2015), once we have obtained θ from θ F ull , we use can use Bayesian linear regression as outlined above.

However, the question of setting α still remains.

To address this, we marginalize α and σ 2 out by slice sampling them according to the log marginal likelihood of the data:

In order to learn a suitable value of γ, along with learning rates and number of epochs, we use Bayesian optimization.

For a complete description of the experimental details, see Appendix B.

One key disadvantage of this approach is that it separates the feature learning from prediction: in particular, there is no reason for the network to learn features relevant for out-of-distribution prediction, particularly when it comes to uncertainty estimates.

From a Bayesian perspective, the neural linear model can be interpreted as a Gaussian process model with a covariance kernel determined by a finite number of basis functions φ θ,i with hyperparameters θ.

Therefore, as in Gaussian process regression, we propose to maximize the log marginal likelihood of the data, L θ,α,σ 2 (D), with respect to θ and σ as the hyperparameters of the model for an empirical Bayes approach.

Note that the computational complexity of this expression is O(N + M 3 ), as opposed to the O(N 3 ) cost typically seen in GP regression.

This is because we are able to apply the Woodbury identity to obtain the determinant in terms of V N , which is M × M , due to the fact that there is a finite number of basis functions.

Since we typically have that N M , this results in significant computational savings.

One issue with this Type-2 maximum likelihood approach is that it will tend to overfit to the training data due to the large number of hyperparameters θ.

As a result, the noise variance σ 2 will tend to be pushed towards zero.

One way of addressing this is by introducing a regularization scheme.

There are many potential regularization schemes that could be introduced: we could regularize θ, α, or σ individually, or using any combination of the three.

We found empirically that of these, simply regularizing θ alone via L 2 regularization seemed the most promising approach.

This results in a Type-2 MAP approach wherein we maximizeL

where we have divided θ into weights θ W and biases θ b and introduced regularization hyperparameters γ W and γ b .

An alternative to regularization would be to treat the noise variance in a Bayesian manner by integrating it out.

Fortunately, for Bayesian linear regression this is still tractable with the use of a normal-inverse-gamma prior on the outputs weights and parameters

The posterior has the form

with posterior predictive

where T ( · ; µ, Σ, ν) is a Student's t-distribution with mean µ, scale Σ, and degrees of freedom ν.

As before, we train the network using empirical Bayes, where the marginal likelihood is given by

Note that by using the Woodbury identity it is possible to compute this in O(N + M 3 ) computational cost as before.

All neural networks tested were ReLU networks with one or two 50-unit hidden layers.

When using a validation set, we set its size to be one fifth of the size of the training set, except for the toy example, where we used half the training set.

We now describe the experimental setup for each model we used.

MAP For the MAP baseline, we select a batch size of 32.

We subsequently use Bayesian optimization (see section B.1 for a description of the Bayesian optimization algorithm we use) to optimize four hyperparameters using validation log likelihood: the regularization parameter γ, a learning rate for the weights, a learning rate for the noise variance, and the number of epochs.

The regularization parameter is allowed to vary within the range corresponding to a log prior variance between -5 and 5.

The learning rates are also optimized in log space in the range [log 1e-4, log 1e − 2].

Finally, the number of epochs is set to vary between zero and the number required to obtain at least 10000 gradient steps (the number of epochs will thus vary with the size of the dataset given a constant batch size).

We initialize the regularization parameter to 0.5, the learning rates at 1e-3, the noise variance at e −3 , and the number of epochs at the maximum value.

The network itself is optimized using ADAM (Kingma and Ba, 2014) .

MAP NL For the MAP neural linear model, we take the above optimal MAP network and obtain 200 slice samples of α W (the output weight prior variance), α b (the output bias prior variance), and σ 2 for Bayesian linear regression.

We initialize α W = 1/50 and α b = 1, to match the scaling used in Neal (1995) .

Regularized NL For the regularized NL model, there are five hyperparameters which we tune via Bayesian optimization: γ W , γ b , a learning rate for θ, a learning rate for σ 2 , and the number of epochs.

We allow γ W and γ b to vary within a range of log prior variances between -10 and 10, and the number of epochs to be in the range of [0, 5000] (since each epoch corresponds to one gradient step).

The ranges for the other parameters remain the same.

We initialize γ W and γ b to 1, and the remaining parameters the same way as in the MAP model.

We again initialize α W = 1/50 and α b = 1.

As before, we use 200 slice samples to marginalize out σ 2 , α W , and α b after the Bayesian optimization was completed.

Bayesian noise NL (ML) Here we optimize the parameters θ, a 0 , b 0 , α W , and α b directly and jointly via the log marginal likelihood.

We employ early stopping by tracking the validation log likelihood up to 5000 epochs, and also maximize the validation log likelihood over a grid of 10 learning rates ranging logarithmically from log 1e-4 to log 1e-2.

We also initialize a 0 = b 0 = 1 and α W = α b = 1.

Finally, we use slice sampling to obtain 200 samples to marginalize out these hyperparameters.

Bayesian noise NL (BO) Instead of optimizing over the hyperparameters jointly as in the BN(ML) model, we keep all except θ fixed over each iteration of Bayesian optimization.

We retain the same initializations, and allow the following ranges for the hyperparameters:

, 10], with the ranges for the learning rate and number of epochs being the same as before.

We retain the same initializations as before as well.

The slice sampling also remains the same.

Here we describe the Bayesian optimization algorithm that we used throughout.

In each case we attempt to maximize the validation log likelihood.

We largely follow the formulation set out in Snoek et al. (2012) .

We use a Gaussian process with a Matérn-5/2 kernel with the model hyperparameters as inputs and the validation log likelihoods as outputs (normalizing the inputs and outputs).

We first learn the kernel hyperparameters (including a noise variance) by maximizing the marginal likelihood of the GP, using 5000 iterations of ADAM (Kingma and Ba, 2014) with a learning rate of 1e-2.

We then obtain 20 slice samples of the GP hyperparameters, before using the expected improvement acquisition function to find the next set of network hyperparameters to test.

In total, we use 50 iterations of Bayesian optimization for each model, initialized with 10 iterations of random search.

In Table 1 , we provide a summary of the models we use, describing which parameters are optimized and how (we exclude learning rates and the number of epochs from this

Table 1: Summary of the models presented.

The first column lists the model; the second shows the optimization objective, while the third shows which parameters were optimized using this objective.

Meanwhile, the fourth lists the parameters that were tuned using Bayesian optimization, while the final lists the parameters that slice sampling was performed on.

In this appendix, we provide the full results from the main text, before briefly describing empirically the effect of slice sampling on the models.

On the next pages, we present tables of average test log likelihoods and test RMSEs for the UCI and UCI gap datasets for all models.

For the UCI datasets, we present the average test log likelihoods and test RMSEs in Tables 2 and 3 , as well as train log likelihoods and RMSEs in Tables 4 and 5 .

Following Bui et al. (2016) , we also compute average ranks of the models across all splits of the standard UCI datasets.

As in Bui et al. (2016) , we additionally follow the procedure for the Friedman test as described in Demšar (2006) , generating the plots shown in Figures 5 and 6 .

These plots show the average rank of each method across all splits, where the difference between models is not statistically significant (p < 0.05) if the models are connected by a dark line, which is determined by the critical difference (CD).

We make a few observations from these results.

First, the two-layer marginal likelihoodbased methods generally outperform the other methods, with the BN(BO)-2 NL model performing the best of all (although not significantly different from the BN(ML)-2 NL model according to the Friedman test).

These are generally followed by the single-layer marginal-likelihood based methods, with the MAP-based methods performing the worst of all.

This confirms our intuition that the more Bayesian versions of the models would yield better performance.

From the train log likelihoods and RMSEs, we observe that all the models exhibit overfitting for most of the datasets, with especially noticeable overfitting on 'boston', 'concrete', 'energy', and 'yacht'.

The overfitting is generally worse on the two-layer models than the single-layer models, as there are more hyperparamters θ that can lead to overfitting in these models.

Based off this trend, we expect that as the number of layers is increased further that the overfitting would worsen, thereby potentially limiting the use of neural linear models to smaller, shallower neural networks.

In Tables 6 and 7 we show the test log likelihoods for the UCI gap datasets.

As we are more concerned about whether the models can capture in-between uncertainty than the performance of the models on these datasets, we do not compute average ranks for these datasets.

Additionally, since the test set is not within the same distribution as the training set, we do not show results on the training sets as they cannot be compared to the test performance.

From these tables, we see that the MAP-L, MAP-L NL, and Reg-L NL models fail catastrophically on the 'naval' and 'energy' dataset.

Additionally, MAP-1 performs especially poorly on 'yacht', although it is not clear whether this can be termed 'catastrophic'.

While the BN NL models perform poorly on 'naval' and 'energy', by looking at the log likelihoods on the individual splits themselves we found that they were not actually failing catastrophically.

This yet again confirms that the more fully Bayesian models are better, although it is surprising just how poorly the Reg-L NL models perform.

Additionally, because the overfitting we observed before worsens as the number of layers is increased, the performance of the BN NL models worsens as more layers are added.

We describe the setup for our experiments on the effect of hyperparameter tuning as well as provide additional results not in the main text.

We first describe the "reasonable" hyperparameter values that we selected:

MAP For the MAP baseline, we select a batch size of 32.

We set γ = 0.5, corresponding to a unity prior variance.

We set the two learning rates to the ADAM default of 1e-3 (Kingma and Ba, 2014).

Finally, we allow for approximately 10000 gradient steps (we ensure that the last epoch is completed, so that there are at least 10000 gradient steps).

MAP NL For the MAP neural linear model, we take the above optimal MAP network and obtain 200 slice samples of α W (the output weight prior variance), α b (the output bias prior variance), and σ 2 for Bayesian linear regression.

We initialize α W = 1/50 and α b = 1, to match the scaling used in Neal (1995).

Regularized NL We set γ W = γ b = 0.5, the learning rates to 1e-3 and the number of epochs to 5000.

We initialize a 0 = b 0 = α W = α b = 1, the learning rate to 1e-3, and the number of epochs to 5000.

We use the same hyperparameter settings as above, although in this case a 0 , b 0 , α W , and α b will remain fixed.

The log likelihoods and RMSEs for each split are then compared to those obtained when hyperparameter tuning is allowed.

We show the average test log likelihoods and test RMSEs for the models without hyperparameter tuning in Tables 9 and 10 for the UCI datasets and  Tables 11 and 12 for the UCI gap datasets.

We also visualize the results for the gap datasets in Figure 7 .

These results show that in general the hyperparameter tuning is of essential importance, particularly for the two-layer cases.

As a whole, the results are significantly worse than with hyperparameter tuning, and in particular, for each of the two-layer methods, there is at least one dataset where the results are catastrophically bad compared to the models with hyperparameter tuning.

Somewhat by contrast, however, while the results for the gap datasets are worse for the methods that did not fail catastrophically, they are not catastrophically worse.

For the methods that were not able to represent in-between uncertainty, however, we find that they now fail catastrophically on even more datasets.

We now verify that the differences induced by the hyperparameter tuning are indeed statistically significant.

In order to do so, we use the Wilcoxon signed-rank test (Wilcoxon, 1992) as described in Demšar (2006) .

By comparing each tuned model to its non-tuned counterpart over all splits, we arrive at the table shown in Table 8 .

This shows that the difference is indeed statistically significant (p < 0.05) on both the standard and gap datasets for the vast majority of models, measured both by log likelihoods and RMSEs.

Model boston concrete energy kin8nm naval power protein wine yacht MFVI-1 -2.60 ± 0.06 -3.09 ± 0.03 -0.74 ± 0.02 1.11 ± 0.01 5.91 ± 0.04 -2.82 ± 0.01 -2.94 ± 0.00 -0.97 ± 0.01 -1.25 ± 0.13 MFVI-2 -2.82 ± 0.04 -3.10 ± 0.02 -0.77 ± 0.02 1.24 ± 0.01 5.99 ± 0.08 -2.81 ± 0.01 -2.87 ± 0.00 -0.98 ± 0.01 -1.15 ± 0.05 MCD-1 -2.71 ± 0.11 -3.33 ± 0.02 -1.89 ± 0.03 0.67 ± 0.01 3.31 ± 0.01 -2.98 ± 0.01 -3.01 ± 0.00 -0.97 ± 0.02 -2.48 ± 0.10 MCD-2 -2.70 ± 0.12 -3.17 ± 0.03 -1.34 ± 0.02 0.74 ± 0.01 3.91 ± 0.02 -2.92 ± 0.01 -2.95 ± 0.00 -1.16 ± 0.04 -2.88 ± 0.22 To ensure that this worse behavior is not because all models require hyperparameter tuning to perform reasonably well, we now compare these results to results for mean field variational inference (MFVI) and Monte Carlo dropout (MCD) without hyperparameter tuning.

We implement MFVI according to Blundell et al. (2015) using the local reparameterization trick (Kingma et al., 2015) .

We set a unity prior variance and use a step size of 1e-3, using ADAM (Kingma and Ba, 2014) with approximately 25000 gradient steps and a batch size of 32.

We allow the gradients to be estimated using 10 samples from the approximate posterior at each step.

For testing we use 100 samples from the approximate posterior.

For MCD, we follow the implementation in Gal and Ghahramani (2016) .

We set the dropout rate to p = 0.05 with weight decay corresponding to unity prior variance.

We again use a learning rate of 1e-3, using ADAM (Kingma and Ba, 2014) with approximately 25000 gradient steps using a batch size of 32.

For testing we use 100 samples generated by the neural network.

For the UCI datasets, we tabulate the test log likelihoods and test RMSEs for one-and two-layer architectures in Tables 13 and 14 .

These generally show reasonable values for each dataset despite the absence of hyperparameter tuning: there is no dataset for which either method can be said to do catastrophically badly.

In fact, the results for MFVI are largely competitive with the best results we obtained for the neural linear models using hyperparameter tuning.

This difference suggests that the reason hyperparameter tuning is important in the neural linear models is because it is necessary to carefully regularize the weights, whereas being approximately Bayesian over all of the weights is not as sensitive to the choice of hyperparameters.

Although the train log likelihoods and RMSEs are not reported here, they confirm this intuition: the neural linear models still suffer from substantial overfitting, whereas the overfitting we observed for MFVI and MCD is far less.

As with the results for the UCI datasets with tuned hyperparameters, we compute average ranks for the models across all splits and use the Friedman test as described in Demšar (2006) to determine whether the differences are statistically significant.

We plot the ranking using test log likelihoods in Figure 8 and using test RMSEs in Figure 9 .

The rankings show that MCD performs poorly on average in terms of test log likelihood, whereas MFVI performs reasonably well for both log likelihood and RMSE.

However, these rankings : Average ranks of the single-run models on the UCI datasets according to test RMSEs, generated as described in Demšar (2006) .

Model boston concrete energy kin8nm naval power protein wine yacht MFVI-1 3.78 ± 0.19 7.04 ± 0.33 4.30 ± 1.82 0.08 ± 0.00 0.03 ± 0.00 4.25 ± 0.12 5.02 ± 0.06 0.63 ± 0.01 1.30 ± 0.12 MFVI-2 3.70 ± 0.16 7.33 ± 0.25 2.58 ± 0.88 0.07 ± 0.00 0.03 ± 0.00 4.67 ± 0.23 5.05 ± 0.11 0.63 ± 0.01 1.26 ± 0.16 MCD-1 3.66 ± 0.12 8.00 ± 0.20 5.01 ± 1.72 0.12 ± 0.00 0.01 ± 0.00 4.87 ± 0.14 5.20 ± 0.05 0.65 ± 0.01 3.17 ± 0.55 MCD-2 3.58 ± 0.12 8.06 ± 0.24 5.18 ± 2.12 0.11 ± 0.00 0.02 ± 0.00 5.54 ± 0.64 5.18 ± 0.08 0.70 ± 0.01 3.75 ± 0.61 Table 16 : Test RMSEs on the UCI Gap Datasets for MFVI and Monte Carlo Dropout do not take into account that the neural linear models fail catastrophically on some, but not all, datasets since they only take the ordering of the methods into account and not how well they perform.

Therefore, we still argue that for the standard UCI datasets MFVI and MCD are better since they perform relatively well across all datasets without the need for hyperparameter tuning.

Finally, we consider the performance of MFVI and MCD on the UCI gap datasets.

We tabulate average test log likelihoods and test RMSEs in Tables 15 and 16 .

These echo the results in Foong et al. (2019b) , showing catastrophic failure of MFVI to express 'in-between' uncertainty for the 'energy' and 'naval' datasets.

They also show that MCD fails catastrophically on the 'energy' datasets, as suggested by theoretical results in Foong et al. (2019a) ; however, we believe these are the first results in the literature that show catastrophic failure to express 'in-between' uncertainty on a real dataset.

The gap results therefore show one crucial advantage of the neural linear models over MFVI and MCD: their ability to express 'in-between' uncertainty.

In this section, we briefly investigate the effect of slice sampling on the performance of the models.

We first make plots of the predictive posterior distribution for each model trained on the toy problem of Section 3.

These plots are visible in Figure 10 .

Note that the MAP-2 NL model simply becomes MAP inference.

The most visible difference between Figure 10 and Figure 1 can be seen in the BN(BO)-2 NL model, which seems to have gained certainty at the edges while perhaps becoming slightly more uncertain in the gap; however, the effect in the gap is almost negligible.

We observe the opposite effect in the MAP-2 NL model.

Additionally, the Reg-2 NL model becomes slightly smoother.

In general, however, it would seem that the effect of slice sampling for the toy problem is small.

We then plot the differences in the log likelihoods and RMSEs between the full models (with slice sampling) and the equivalent models without the final slice sampling step, to observe any quantitative differences.

These plots are shown in Figure 11 for the UCI datasets and Figure 12 for the UCI gap datasets.

These plots do not give a clear picture of whether slice samping improves or worsens the performance of these models: it seems to depend on both the model and the dataset.

To gain a clearer insight into whether slice sampling improves the performance of the neural linear models, we once again perform the Wilcoxon signed-rank test to compare the results obtained with slice sampling to those without.

The results of this analysis is shown in Table 17 .

This shows that the majority of models are in fact improved by slice sampling, particularly when the improvement is measured in terms of the log likelihoods.

However, in many cases, particularly when performance is measured in terms of RMSE, the effect of slice sampling is not statistically significant.

Furthermore, it seems that performance for the Reg-L NL and BN(BO)-2 NL models may be worsened by slice sampling.

In conclusion, in most cases performance will not be worsened by slice sampling.

In particular, the MAP-L NL models seem to benefit especially from slice sampling.

However, slice sampling is likely detrimental to the Reg-L NL models in all cases and potentially harmful for the BN(BO)-2 NL model when it comes to in-between uncertainty.

@highlight

We benchmark the neural linear model on the UCI and UCI "gap" datasets.