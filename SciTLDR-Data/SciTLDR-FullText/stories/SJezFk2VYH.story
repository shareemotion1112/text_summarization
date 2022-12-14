Variational Bayesian Inference is a popular methodology for approximating posterior distributions over Bayesian neural network weights.

Recent work developing this class of methods has explored ever richer parameterizations of the approximate posterior in the hope of improving performance.

In contrast, here we share a curious experimental finding that suggests instead restricting the variational distribution to a more compact parameterization.

For a variety of deep Bayesian neural networks trained using Gaussian mean-field variational inference, we find that the posterior standard deviations consistently exhibit strong low-rank structure after convergence.

This means that by decomposing these variational parameters into a low-rank factorization, we can make our variational approximation more compact without decreasing the models' performance.

Furthermore, we find that such factorized parameterizations improve the signal-to-noise ratio of stochastic gradient estimates of the variational lower bound, resulting in faster convergence.

Bayesian Neural Networks (MacKay, 1992; Neal, 1993) explicitly represent their parameteruncertainty by forming a posterior distribution over model parameters, instead of relying on a single point estimate for making predictions, as is done in traditional deep learning.

Besides offering improved predictive performance over single models, Bayesian neural networks are also more robust to hard examples (Raftery et al., 2005) , have better calibration of predictive uncertainty and thus can be used for out-of-domain detection or other risk-sensitive applications (Ovadia et al., 2019) .

Variational inference (Peterson, 1987; Hinton and Van Camp, 1993 ) is a popular class of methods for approximating the posterior distribution p(w|x, y), since the exact Bayes' rule is often intractable to compute for models of practical interest.

This class of methods specifies a distribution q θ (w) of given parametric or functional form as the posterior approximation, and optimizes the approximation by solving an optimization problem.

In particular, we minimize the negative Evidence Lower Bound (negative ELBO) approximated by samples from the posterior:

by differentiating with respect to the variational parameters θ (Salimans et al., 2013; Kingma and Welling, 2013) .

In Gaussian Mean Field Variational Inference (GMFVI) (Blei et al., 2017; Blundell et al., 2015) , we choose the variational approximation to be a fully factorized Gaussian distribution:

q(w ij ), with q(w ij ) = N (µ ij , σ

where W ∈ R m×n is a weight matrix of a single network layer and i and j are the row and column indices in this weight matrix.

In practice, we often represent the posterior standard deviation parameters σ ij in the form of a matrix A ∈ R m×n +

.

With this notation, we have the relationship Σ q = diag(vec(A 2 )) where the elementwise-squared A is vectorized by stacking its columns, and then expanded as a diagonal matrix into R mn×mn + .

While Gaussian Mean-Field posteriors are considered to be one of the simplest types of variational approximations, with some known limitations (Giordano et al., 2018) , they scale to comparatively large models and generally provide competitive performance (Ovadia et al., 2019) .

However, when compared to deterministic neural networks, GMFVI doubles the number of parameters and is often harder to train due to the increased noise in stochastic gradient estimates.

Beyond fully factorized mean-field, recent research in variational inference has explored richer parameterizations of the approximate posterior in order to improve the performance of Bayesian neural networks (see Appendix A and Figure 3 ).

For instance, various structures of Gaussian posteriors have been proposed, with per layer block-structured covariances (Louizos and Welling, 2016; Sun et al., 2017; Zhang et al., 2017) , full covariances (Barber and Bishop, 1998) with different parametrizations (Seeger, 2000) , up to more flexible approximate posteriors using normalizing flows (Rezende and Mohamed, 2015) and extensions thereof (Louizos and Welling, 2017 ).

In contrast, here we study a simpler, more compactly parameterized mean-field variational posterior which ties variational parameters in the already diagonal covariance matrix.

We show that such a posterior approximation can also work well for a variety of models.

In particular we find that:

• Converged posterior standard deviations under GMFVI consistently display strong low-rank structure.

This means that by decomposing these variational parameters into a low-rank factorization, we can make our variational approximation more compact without decreasing our model's performance.

• Factorized parameterizations of posterior standard deviations improve the signal-to-noise ratio of stochastic gradient estimates, and thus not only reduce the number of parameters compared to standard GMFVI, but also can lead to faster convergence.

We start by empirically studying the properties of the spectrum of posterior standard deviation matrices A, post training, in models already trained until convergence using standard fully-parameterized Gaussian mean-field variational distributions.

Interestingly, we observe that those matrices naturally exhibit a low-rank structure (see Figure 1 ), i.e,

for some U ∈ R m×k , V ∈ R n×k and k a small value (e.g., 2 or 3).

This observation motivates the introduction of the following variational family, which we name k-tied Normal:

where the squaring of the matrix UV T is applied elementwise.

Due to the tied parametrization of the diagonal covariance matrix, we emphasize that this variational family is smaller-i.e., included in-the standard Gaussian mean-field variational distribution family.

Interestingly, we find that despite its compactness, our posterior is able to match the fully parametrized GMFVI in terms of ELBO and predictive performance both in a post training approximation (see Figure 1 ) and when training the tied parameters U and V from a random initialization (see Figure 2) .

Furthermore, the total number of the standard deviation parameters in our method is k(m + n) from U and V, compared to mn for A in the standard GMFVI parametrization.

Given that in our experiments the k is very low (e.g k = 2) this reduces the number of standard deviation parameters from quadratic to linear in the dimensions of the layer, see Table 1 .

More importantly, such parameter sharing across the weights leads to higher signal-to-noise ratio during training and thus in some cases faster convergence, see Figure 2 .

Finally, the matrix variate Gaussian distribution (Gupta and Nagar, 2018), referred to as MN and already used for variational inference in the most closely related work of Louizos and Welling (2016) and Sun et al. (2017) , is similar to our k-tied Normal distribution when k = 1 (see also Figure 3 ).

Interestingly, we prove that for k ≥ 2, our k-tied Normal distribution cannot be represented by any MN distribution (see Appendix B).

We now provide a short description of the experimental setting and more detailed experimental results.

In our experiments we use three model types: a 3 layer Multilayer Perceptron (MLP) trained on the MNIST dataset (LeCun and Cortes, 2010), a LeNet-type Convolutional Neural Network (CNN) (LeCun et al., 1998 ) trained on the CIFAR-100 dataset (Krizhevsky et al., 2009) , and a vanilla LSTM model (Hochreiter and Schmidhuber, 1997) trained on the IMDB dataset (Maas et al., 2011) .

Appendix E provides more details about the experimental setting.

We highlight that our experiments focus primarily on the comparison across a broad range of model types rather than competing with the state-of-the-art results over the specifically used datasets.

Therefore, we use small to medium models that are known to train well using the standard GMFVI approach explored in this paper.

Scaling GMFVI to larger model sizes is still a challenging research problem (Osawa et al., 2019) .

Figure 1: Posterior standard deviations, in contrast to posterior means, of dense layers in LeNet CNN trained using standard GMFVI display strong low-rank structure and can be approximated without loss to predictive metrics.

Top: Explained variance 1 per singular value from SVD of matrices of converged posterior means and standard deviations.

Bottom: Impact of post training low-rank approximation of the posterior standard deviation matrices on model's performance.

We report mean and standard error of the mean (SEM) for each metric across 100 models samples.

Figure 1 shows that GMFVI applied to the LeNet CNN learns posterior standard deviation matrices of the CNN's dense layers that have most of their variance explained 1 by the first two components of their SVD decomposition.

Furthermore, we also see that these matrices can be approximated post training by their low-rank SVD decompositions with little ELBO and predictive performance loss.

In Appendix C we show that these results also hold for the analyzed MLP and LSTM models.

Figure 2 shows the results of exploiting the above observation by applying the k-tied Normal posterior during GMFVI training.

We see that for k ≥ 2, the k-tied Normal posterior is able to achieve the performance competitive with the standard GMFVI posterior parametrization, while reducing the total number of model parameters.

The benefits of using the k-tied Normal posterior are most visible for models where the dense layers with the k-tied Normal posterior constitute a significant portion of the total number of the model parameters (e.g. MLPs and CNNs with dense layers for classification).

Furthermore, we observe a significant increase in the signal-to-noise ratio 2 (SNR) of the gradients of parameters of the GMFVI posterior standard deviations when using the k-tied Normal posterior.

Importantly, we also see that the increase in the gradient SNR translates into faster convergence of the negative ELBO objective in some of the analyzed models.

In this work we have shown that Bayesian Neural Networks trained with standard Gaussian meanfield variational inference exhibit posterior standard deviation matrices that can be approximated with little information loss by a low-rank decomposition.

This suggests that richer parameterizations of the variational posterior may not always be needed, and that compact parameterizations can also work well.

We used this insight to propose a simple, yet effective variational posterior parametrization, which speeds up training and reduces the number of variational parameters without degrading predictive performance on three different model types.

In future work, we hope to scale up variational inference with compactly parameterized approximate posteriors to much larger models and more complex problems.

For mean-field variational inference to work well in that setting several challenges will likely need to be addressed (Osawa et al., 2019) ; improving the signal-to-noise ratio of ELBO gradients using our compact variational parameterizations may provide a piece of the puzzle.

1.

Explained variance for the rank k approximation is calculated as γ , where g b is the gradient value for a single parameter.

The expectation E and variance V ar of the gradient values g b are calculated over a window of last 10 batches.

The application of variational inference to neural networks dates back at least to Peterson (1987) ; Hinton and Van Camp (1993) .

Many developments 3 have followed those seminal research efforts, in particular regarding (1) the expressiveness of the variational posterior distribution and (2) the way the variational parameters themselves can be structured to lead to compact, easier-to-learn and scalable formulations.

We organize the discussion of this section around those two aspects, with a specific focus on the Gaussian case.

For a graphical overview of the related work see Figure 3 .

Full Gaussian posterior.

Because of their substantial memory and computational cost, Gaussian variational distributions with full covariance matrices have been primarily applied to (generalized) linear models and shallow neural networks (Jaakkola and Jordan, 1997; Barber and Bishop, 1998; Marlin et al., 2011; Titsias and Lázaro-Gredilla, 2014; Miller et al., 2017; Ong et al., 2018) .

To represent the dense covariance matrix efficiently in terms of variational parameters, several schemes have been proposed, including the sum of low-rank plus diagonal matrices (Barber and Bishop, 1998; Seeger, 2000; Miller et al., 2017; Zhang et al., 2017; Ong et al., 2018) , the Cholesky decomposition (Challis and Barber, 2011) or by operating instead on the precision matrix (Tan and Nott, 2018; Mishkin et al., 2018) .

Gaussian posterior with block-structured covariances.

In the context of Bayesian neural networks, the layers represent a natural structure to be exploited by the covariance matrix.

When assuming independence across layers, the resulting covariance matrix exhibits a block-diagonal structure that has been shown to be a well-performing simplification of the dense setting (Sun et al., 2017; Zhang et al., 2017) , with both memory and computational benefits.

Within each layer, the corresponding diagonal block of the covariance matrix can be represented by a Kronecker product of two smaller matrices (Louizos and Welling, 2016; Sun et al., 2017) , possibly with a parametrization based on rotation matrices (Sun et al., 2017) .

Finally, using similar techniques, Zhang et al. (2017) proposed to use a block tridiagonal structure that better approximates the behavior of a dense covariance.

Fully factorized mean-field Gaussian posterior.

A fully factorized Gaussian variational distribution constitutes the simplest option for variational inference.

The resulting covariance matrix is diagonal and all underlying parameters are assumed to be independent.

While the mean-field assumption is known to have some limitations-e.g., underestimated variance of the posterior distribution (Turner and Sahani, 2011) and robustness issues (Giordano et al., 2018) -it leads to scalable formulations, with already competitive performance, as for instance illustrated by the recent uncertainty quantification benchmark of Ovadia et al. (2019) .

Parameters (total) Multivariate Normal mn + Because of its simplicity and scalability, the fully-factorized Gaussian variational distribution has been widely used for Bayesian neural networks (Graves, 2011; Ranganath et al., 2014; Blundell et al., 2015; Hernández-Lobato and Adams, 2015; Zhang et al., 2017; Khan et al., 2018) .

Our approach can be seen as an attempt to further reduce the number of parameters of the (already) diagonal covariance matrix.

Closest to our approach is the work of Louizos and Welling (2016) .

Their matrix variate Gaussian distribution instantiated with the Kronecker product of the diagonal row-and column-covariance matrices leads to a rank-1 tying of the posterior variances.

In contrast, we explore tying strategies beyond the rank-1 case, which we show to lead to better performance (both in terms of ELBO and predictive metrics).

Importantly, we further prove that tying strategies with a rank greater than one cannot be represented in a matrix variate Gaussian distribution, thus clearly departing from (Louizos and Welling, 2016) (see Appendix B for details).

Our approach can be also interpreted as a particular case of hierarchical variational inference (Ranganath et al., 2016) where the prior on the variational parameters corresponds to a Dirac distribution, non-zero only when a pre-specified low-rank tying relationship holds.

We close this related work section by mentioning the existence of other strategies to produce more flexible approximate posteriors, e.g., normalizing flows (Rezende and Mohamed, 2015) and extensions thereof (Louizos and Welling, 2017) .

In this section of the appendix, we formally explain the connections between the k-tied Normal distribution and the matrix variate Gaussian distribution (Gupta and Nagar, 2018), referred to as MN .

Consider positive definite matrices Q ∈ R r×r and P ∈ R c×c and some arbitrary matrix M ∈ R r×c .

We have by definition that W ∈ R r×c ∼ MN (M, Q, P) if and only if vec(W) ∼ N (vec(M), P ⊗ Q), where vec(·) stacks the columns of a matrix and ⊗ is the Kronecker product

The MN has already been used for variational inference by Louizos and Welling (2016) and Sun et al. (2017) .

In particular, Louizos and Welling (2016) consider the case where both P and Q are restricted to be diagonal matrices.

In that case, the resulting distribution corresponds to our k-tied Normal distribution with k = 1 since

Importantly, we prove below that, in the case where k ≥ 2, the k-tied Normal distribution cannot be represented as a matrix variate Gaussian distribution.

Lemma.

[Rank-2 matrix and Kronecker product] Let B be a rank-2 matrix in R r×c + .

There do not exist matrices Q ∈ R r×r and P ∈ R c×c such that diag(vec(B)) = P ⊗ Q.

Proof Let us introduce the shorthand D = diag(vec(B)).

By construction, D is diagonal and has its diagonal terms strictly positive (it is assumed that B ∈ R r×c + , i.e., b ij > 0 for all i, j).

We proceed by contradiction.

Assume there exist Q ∈ R r×r and P ∈ R c×c such that D = P ⊗ Q.

This implies that all diagonal blocks of P ⊗ Q are themselves diagonal with strictly positive diagonal terms.

Thus, p jj Q is diagonal for all j ∈ {1, . . .

, c}, which implies in turn that Q is diagonal, with non-zero diagonal terms and p jj = 0.

Moreover, since the off-diagonal blocks p ij Q for i = j must be zero and Q = 0, we have p ij = 0 and P is also diagonal.

To summarize, if there exist Q ∈ R r×r and P ∈ R c×c such that D = P ⊗ Q, then it holds that D = diag(p) ⊗ diag(q) with p ∈ R c and q ∈ R r .

This last equality can be rewritten as b ij = p j q i for all i ∈ {1, . . .

, r} and j ∈ {1, . . .

, c}, or equivalently

This leads to a contradiction since qp has rank one while B is assumed to have rank two.

Appendix C. Low rank-structure in the GMFVI posterior standard deviations

We provide here more results from the post training analysis of the converged posterior standard deviations trained with the standard parameterization of the GMFVI.

In particular, while in the main paper we focused on the CNN model, here we provide also the results for the MLP and LSTM model.

Our main experimental observation is that the standard GMFVI learns posterior standard deviation matrices that have a low-rank structure across different model types.

To show this, we investigate the results of the SVD decomposition of posterior standard deviation matrices for three types of models trained until ELBO convergence using GMFVI.

Figure 5 shows per rank percentage of explained variance with respect to the rank k of the low-rank SVD approximation.

The percent of explained variance for the rank k approximation is calculated as 100 · γ 2 k / i γ 2 i , where γ i are singular values.

We observe that most of the variance in the posterior standard deviation parameters is captured in the rank-1 approximation.

However, a more fine-grained analysis shows that a rank-2 approximation can encompass nearly all of the remaining variance.

Finally, we note that we do not observe the same behaviour for the posterior mean parameters as we do for the posterior standard deviation parameters.

Figure 4 further supports this claim visually by comparing the heat maps of the full-rank posterior standard deviations matrix with its rank-1 and rank-2 approximations.

In particular, we observe that the rank-2 approximation results in the heat-map looking visually very similar to the full-rank matrix.

Motivated by the above observation, we show that it is possible to replace the full-rank posterior standard deviation matrix with its low-rank approximation without a decrease in performance.

Table  2 shows the comparison of performance of models with different ranks of approximation to their posterior standard deviation matrix.

The results show that the post training approximation with ranks higher than 1 achieves predictive performance very close to that of the full-rank matrix.

This observation itself could be used as a form of a post training network compression.

Moreover, it gives rise to further interesting exploration directions such as formulating posteriors that exploit such a low rank structure.

In this paper by explore this particular direction in the form of the k-tied Normal posterior.

Table 2 : Impact of post training low-rank approximation of the GMFVI-trained posterior standard deviation matrices on ELBO and predictive performance, for three types of models.

We report mean and SEM of each metric across 100 weights samples.

Appendix D. Impact of the k-tied Normal on the GMFVI convergence speed Figure 6 shows convergence plots of negative ELBO on respective validation data sets for different model types trained with GMFVI using the standard parametrization (full-rank) and the k-tied Normal posterior with different levels of tying k. We observe that the impact of the k-tied Normal posterior on the convergence depends on the model type.

For the MLP model the impact is strong and consistent with the k-tied Normal posterior increasing convergence speed compared to the standard GMFVI parametrization.

For the LSTM model we also observe a similar speed-up.

However, for the CNN model the impact of the k-Normal posterior on the ELBO convergence is much smaller.

We hypothesize that this is due to the fact that we use the k-tied Normal posterior for all the layers trained using GMFVI in the MLP and the LSTM models, while in the CNN model we use the k-tied Normal posterior only for some of the GMFVI trained layers.

More precisely, in the CNN model we use the k-tied Normal posterior only for the two dense layers, while the two convolutional layers are trained using the standard parametrization of the GMFVI.

Full-rank Rank-1 Rank-2 Rank-3 Figure 6 : Impact of the k-tied Normal posterior with different ranks k on the convergence of negative ELBO (lower is better) reported on validation datasets of the MLP (left), CNN (center), and LSTM (right) models.

Full-rank is the standard parametrization of the GMFVI without any tying.

Model architectures We analyze three types of GMFVI Bayesian neural network models:

• Multilayer Perceptron (MLP): a network of three dense layers and ReLu activations that we train on the MNIST dataset (LeCun and Cortes, 2010) .

We use the last 10,000 examples of the training set as a validation set.

The three layers have sizes of 400, 400 and 10 hidden units.

• Convolutional Neural Network (CNN): a LeNet architecture (LeCun et al., 1998) with two convolutional layers and two dense layers that we train on the CIFAR-100 dataset (Krizhevsky et al., 2009) .

We use the last 10,000 examples of the training set as a validation set.

The two convolutional layers have filters of sizes 32 and 64.

The two dense layers have sizes of 512 and 100 hidden units.

• Long Short-Term Memory (LSTM): a model that consists of an embedding and an LSTM cell (Hochreiter and Schmidhuber, 1997), followed by a single unit dense layer.

We train it on an IMBD dataset (Maas et al., 2011) , in which we use the last 5,000 examples of the training set as a validation set.

The LSTM cell consists of two dense weight matrices, namely kernel and recurrent kernel.

The embedding and the LSTM cell are each of size 128.

More concretely, we use the model architecture available in the Keras (Chollet et al., 2015) examples 4 , but without dropout.

In the MLP and the CNN models we approximate the posterior using GMFVI for all the weights (both kernel and bias weights).

In the LSTM model we approximate the posterior using GMFVI only for the kernel and recurrent kernel weights, while the posterior for the bias weights is approximated using a MAP solution.

In each of the three models we use a mean-field Normal posterior with the standard reparametrization trick (Kingma and Welling, 2013 ) and a Normal prior N (0, σ p ) with a single scalar standard deviation hyper-parameter σ p for all the layers.

We initialize the variational posterior means using the standard He initialization (He et al., 2015) and the posterior standard deviations using samples from N (0.01, 0.001).

We select the σ p for each of the models separately from a set of {0.2, 0.3} based on the performance on the validation data set.

For optimization we use an Adam optimizer (Kingma and Ba (2014)).

We pick the optimal learning rate for each model from the set of {0.0001, 0.0003, 0.001, 0.003} based on the performance on the validation data set.

We chose the batch size also based on the performance on the validation data set.

For the MLP and the CNN models we use the batch size of 1024 and for the LSTM model a batch size of 128.

Low-rank structure analysis To investigate the low-rank structure in the converged posterior standard deviation matrices, we generate low-rank approximations to these matrices.

It is possible that such low-rank approximations contain negative values.

In such cases, we threshold the minimum values of the resulting approximations at a very low positive constant to meet the constraint on the positive values of the standard deviations.

k-tied Normal posterior training When training the GMFVI models with the k-tied Normal variational posterior, we use the k-tied Normal variational posterior for all the dense layers of the three analyzed models.

More concretely, we use the k-tied Normal variational posterior for all the three layers of the MLP model, for the two dense layers of the CNN model and for the LSTM cell's kernel and recurrent kernel.

We initialize the parameters u ik and v jk of the k-tied Normal distribution so that after the outerproduct operation the respective standard deviations σ ij have the same mean values as we obtain when using the standard GMFVI posterior parametrization.

In other words, we initialize the parameters u ik and v jk so that after the outer-product operation the respective σ ij standard deviations have means at 0.01 before transforming to log-domain.

This means that in the log domain the parameters u ik and v jk are initialized as 0.5(log(0.01) − log(k)).

We also add white noise N (0, 0.1) to the values of u ik and v jk in the log domain to break symmetry.

We recommend using KL annealing for training the models with the k-tied Normal posterior.

With KL annealing, we linearly scale-up the contribution of the KL term from a fraction of its full value to its full contribution over the course of training.

We select the best linear coefficient for the KL annealing from {5 × 10 −5 , 5 × 10 −6 } per batch and increase the KL contribution every 100 batches.

For instance, we use KL annealing to obtain the results for the test performance in Figure 2 .

However, we do not use KL annealing for the runs for which we report the SNR and negative ELBO convergence results in the same Figure 2 .

In these two cases KL annealing would occlude the values, which show the clear impact of the k-tied Normal posterior.

@highlight

Mean field VB uses twice as many parameters; we tie variance parameters in mean field VB without any loss in ELBO, gaining speed and lower variance gradients.