Obtaining reliable uncertainty estimates of neural network predictions is a long standing challenge.

Bayesian neural networks have been proposed as a solution, but it remains open how to specify their prior.

In particular, the common practice of a standard normal prior in weight space imposes only weak regularities, causing the function posterior to possibly generalize in unforeseen ways on inputs outside of the training distribution.

We propose noise contrastive priors (NCPs) to obtain reliable uncertainty estimates.

The key idea is to train the model to output high uncertainty for data points outside of the training distribution.

NCPs do so using an input prior, which adds noise to the inputs of the current mini batch, and an output prior, which is a wide distribution given these inputs.

NCPs are compatible with any model that can output uncertainty estimates, are easy to scale, and yield reliable uncertainty estimates throughout training.

Empirically, we show that NCPs prevent overfitting outside of the training distribution and result in uncertainty estimates that are useful for active learning.

We demonstrate the scalability of our method on the flight delays data set, where we significantly improve upon previously published results.

Many successful applications of neural networks BID25 BID53 BID57 are in restricted settings where predictions are only made for inputs similar to the training distribution.

In real-world scenarios, neural networks can face truly novel data points during inference, and in these settings it can be valuable to have good estimates of the model's uncertainty.

For example, in healthcare, reliable uncertainty estimates can prevent overconfident decisions for rare or novel patient conditions BID49 .

Similarly, autonomous agents that actively explore their environment can use uncertainty estimates to decide what data points will be most informative.

Epistemic uncertainty describes the amount of missing knowledge about the data generating function.

Uncertainty can in principle be completely reduced by observing more data points at the right locations and training on them.

In contrast, the data generating function may also have inherent randomness, which we call aleatoric noise.

This noise can be captured by models outputting a distribution rather than a point prediction.

Obtaining more data points allows the noise estimate to move closer to the true value, which is usually different from zero.

For active learning, it is crucial to separate the two types of randomness: we want to acquire labels in regions of high uncertainty but low noise BID36 .Bayesian analysis provides a principled approach to modeling uncertainty in neural networks BID8 BID37 .

Namely, one places a prior over the network's weights and biases.

This effectively places a distribution over the functions that the network represents, capturing uncertainty about which function best fits the data.

Specifying this prior remains an open challenge.

Common practice is to use a standard normal prior in weight space, which imposes weak shrinkage regularities analogous to weight decay.

It is neither informative about the induced function class nor the data (e.g., it is sensitive to parameterization).

This can cause the induced function posterior to generalize in unforeseen ways on out-of-distribution (OOD) inputs, which are inputs outside of the distribution that generated the training data.

Motivated by these challenges, we introduce noise contrastive priors (NCPs), which encourage uncertainty outside of the training distribution through a loss in data space.

NCPs are compatible with any model that represents functional uncertainty as a random variable, are easy to scale, and yield reliable uncertainty estimates that show significantly improved active learning performance.

Specifying priors is intuitive for small probabilistic models, where each variable typically has a clear interpretation BID2 .

It is less intuitive for neural networks, where the parameters serve more as adaptive basis coefficients in a nonparametric function.

For example, neural network models are nonidentifiable due to weight symmetries that yield the same function BID43 .

This makes it difficult to express informative priors on the weights, such as expressing high uncertainty on unfamiliar examples.

Data priors Unlike a prior in weight space, a data prior lets one easily express informative assumptions about input-output relationships.

Here, we use the example of a prior over a labeled data set {x, y}, although the prior can also be on x and another variable in the model that represents uncertainty and has a clear interpretation.

The prior takes the form p prior (x, y) = p prior (x) p prior (y | x), where p prior (x) denotes the input prior and p prior (y | x) denotes the output prior.

To prevent overconfident predictions, a good input prior p prior (x) should include OOD examples so that it acts beyond the training distribution.

A good output prior p prior (y | x) should be a high-entropy distribution, representing high uncertainty about the model output given OOD inputs.

The out-of-distribution classifier model uses a binary auxiliary variable o to determine if a given input is out-of-distribution; given its value, the output is drawn from either a neural network prediction or a wide output prior.

Generating OOD inputs Exactly generating OOD data is difficult.

A priori, we must uniformly represent the input domain.

A posteriori, we must represent the complement of the training distribution.

Both distributions are typically uniform over infinite support, making them ill-defined.

To estimate OOD inputs, we develop an algorithm inspired by noise contrastive estimation BID15 BID42 , where a complement distribution is approximated using random noise.

A hypothesis of our work is that in practice it is enough to encourage high uncertainty output near the boundary of the training distribution, and that this effect will propagate to the entire OOD space.

This hypothesis is backed up by previous work BID30 as well as our experiments (see FIG0 ).

This means we no longer need to sample arbitrary OOD inputs.

It is enough to sample OOD points that lie close to the boundary of the training distribution, and to apply our desired prior at those points.

Loss function Noise contrastive priors are data priors that are enforced on both training inputs x and inputs x perturbed by noise.

For example, in binary and categorical input domains, we approximate OOD inputs by randomly flipping the features to different classes with a certain probability.

For continuous valued inputs x, we can use additive Gaussian noise to obtain noised up inputsx = x + .

This expresses the noise contrastive prior where inputs are distributed according to the convolved distribution, DISPLAYFORM0 The variances ?? 2 x and ?? 2 y are hyperparameters that tune how far from the boundary we sample, and how large we want the output uncertainty to be.

We choose ?? x = 0 to apply the prior equally in all directions from the data manifold.

The output mean ?? y determines the default prediction of the model outside of the training distribution, for example ?? y = 0.

We set ?? y = y which corresponds to data augmentation BID39 BID0 , where a model is trained to recover the true labels from perturbed inputs.

This way, NCP makes the model uncertain while still trying to generalize to OOD inputs.

For training, we minimize the loss function DISPLAYFORM1 The first term represents typical maximum likelihood, in which one minimizes the KL divergence to the empirical training distribution p train (y | x) over training inputs.

The second term is added by our method: it represents the analogous term on a data prior.

The hyperparameter ?? sets the relative trade-off between them.

Interpretation as function prior The noise contrastive prior can be interpreted as inducing a function prior.

This is formalized through the prior predictive distribution, DISPLAYFORM2 The distribution marginalizes over network parameters ?? as well as data fantasized from the data prior.

The distribution p(?? |x,???) represents the distribution of model parameters after fitting the prior data.

That is, the belief over weights is shaped to make p(y | x) highly variable.

This parameter belief causes uncertain predictions outside of the training distribution, which we could not specify in weight space directly.

Because network weights are constrained to fit the data prior, the prior acts as "pseudo-data."

This is similar to classical work on conjugate priors: a Beta(??, ??) prior on the probability of a Bernoulli likelihood implies a Beta posterior, and if the posterior mode is chosen as an optimal parameter setting, then the prior translates to ?? ??? 1 successes and ?? ??? 1 failures.

It is also similar to pseudo-data in sparse Gaussian processes BID46 .Data priors encourage learning parameters that not only capture the training data well but also the prior data.

In practice, we can combine NCP with other priors, for example the typical standard normal prior in weight space for Bayesian neural networks, although we did not find this necessary in our experiments.

Noise contrastive priors are applicable to any model that represents uncertainty in a random variable.

The NCP can then be added to that random variable to make the model uncertain on OOD inputs.

In this section, we apply NCP to a Bayesian neural network (BNN) trained via variational inference.

BID3 introduce such a model under the name Bayes by Backprop (BBB) that uses a standard normal prior in weight space.

We extend this model with a NCP on the mean predicted by the neural network.

Consider a regression task with data {x, y} that we model as p(y | x, ??) = Normal(??(x), ?? 2 (x)) with mean and variance predicted by a neural network from the inputs.

This model is heteroskedastic, meaning that it can predict a different aleatoric noise amount for every point in the input space.

We use a weight prior for only the output layer BID29 BID5 that predicts the mean, resulting in the model DISPLAYFORM0 We do not model uncertainty about the noise estimate, as this is not required for the approximation for the Gaussian expected information gain (MacKay, 1992a) that we use to acquire labels.

Therefore, the distribution of the mean induced by the weight prior, q(??(x)) = ??(x, ??)q ?? (??) d??, represents the model's epistemic uncertainty.

Note that this is different from the predictive distribution, which combines both uncertainty and noise.

We place an NCP on the distribution of the mean, resulting in the loss function DISPLAYFORM1 Here,x are the perturbed inputs and q ?? (??) forms an approximate posterior over weights.

1 Because we only use the weight belief for the linear output layer, we can compute the KL-divergence of the NCP loss analytically.

In other models, it could be estimated using samples.

The loss function applies weight regularization in order for network weights to regress to a standard normal prior; like other regularization techniques, this assists in improving the network's generalization in-distribution.

The NCP loss encourages the network's generalization OOD by matching the mean distribution to the output prior.

Minimizing the KL divergence to a wide output prior results in high uncertainty on OOD inputs, so the model will explore these data points during active learning.

In practice, we find that NCP is sufficient as a prior for the BNN and set ?? = 0.

The appendix (Appendix B includes an alternative interpretation explaining why NCP might be sufficient, which represents the weight space KL-divergence in data space after a change of variables.

Priors for neural networks Classic work has investigated entropic priors BID4 and hierarchical priors BID37 BID44 BID28 ).

More recently, BID9 introduce networks with latent variables in order to disentangle forms of uncertainty, and FlamShepherd et al. (2017) propose general-purpose weight priors based on approximating Gaussian processes.

Other works have analyzed priors for compression and model selection BID14 BID34 .

Instead of a prior in weight space (or latent inputs as in BID9 ), NCPs take the functional view by imposing explicit regularities in terms of the network's inputs and outputs.

BID38 propose prior networks to avoid an explicit belief over parameters for classification tasks.

Input and output regularization There is classic work on adding noise to inputs for improved generalization BID39 BID0 BID1 .

For example, denoising autoencoders BID58 encourage reconstructions given noisy encodings.

Output regularization is also a classic idea from the maximum entropy principle BID21 , where it has motivated label smoothing BID54 and entropy penalties BID45 .

Also related is virtual adversarial training BID41 , which includes examples that are close to the current input but cause a maximal change in the model output, and mixup BID60 , which includes examples under the vicinity of training data.

These methods are orthogonal to NCPs: they aim to improve generalization from finite data within the training distribution (interpolation), while we aim to improve uncertainty estimates outside of the training distribution (extrapolation).Classifying out-of-distribution inputs A simple approach for neural network uncertainty is to classify whether data points belong to the data distribution, or are OOD BID17 .

This is core to noise contrastive estimation BID16 , a training method for intractable probabilistic models.

More recently, BID30 introduce a GAN to generate OOD samples, and BID33 add perturbations to the input, applying an "OOD detector" to improve softmax scores on OOD samples by scaling the temperature.

Extending these directions of research, we connect to Bayesian principles and focus on uncertainty estimates that are useful for active data acquisition.

To demonstrate their usefulness, we evaluate NCPs on various tasks where uncertainty estimates are desired.

Our focus is on active learning for regression tasks, where only few targets are visible in the beginning, and additional targets are selected regularly based on an acquisition function.

We use two data sets: a toy example and a large flights data set.

We also evaluate how sensitive our method is to the choice of input noise.

Finally, we show that NCP scales to large data sets by training on the full flights data set in a passive learning setting.

Our implementation uses TensorFlow Probability BID10 BID56 and is open-sourced at https://<hidden-for-review>.We compare four neural network models, all using leaky ReLU activations BID35 and trained using Adam BID24 Figure 3 : Active learning on the 1-dimensional regression problem, mean and standard deviation over 20 seeds.

The test root mean squared error (RMSE) and negative log predictive density (NLPD) of the models trained with NCP decreases during the active learning run, while the baseline models select less informative data and overfit.

The deterministic network is barely visible in the plots as it overfits quickly.

FIG0 shows the predictive distributions of the models.??? Deterministic neural network (Det) A neural network that predicts the mean and variance of a normal distribution.

The name stands for deterministic, as there is no weight uncertainty.??? Bayes by Backprop (BBB) A Bayesian neural network trained via gradient-based variational inference with a standard normal prior in weight space BID3 BID26 .We use the same model as in Section 3 but without the NCP loss term.??? Bayes by Backprop with noise contrastive prior (BBB+NCP) Bayes by Backprop with NCP on the predicted mean distribution as described in Section 3.??? Out-of-distribution classifier with noise contrastive prior (OCD+NCP) An uncertainty classifier model described in Appendix A. It is a deterministic neural network combined with NCP which we use as a baseline alternative to Bayes by Backprop with NCP.For active learning, we select new data points {x, y} for which x maximizes the expected information gain DISPLAYFORM0 Intuitively, this objective function is higher where the model has high epistemic uncertainty and predicts low aleatoric noise.

We use an approximation from MacKay (1992a) for Gaussian posterior predictive distributions.

Moreover, we place a softmax distribution on the information gain for all available data points and acquire labels by sampling with a temperature of ?? = 0.5 to get diversity when selecting batches of labels, DISPLAYFORM1 where ?? 2 (x) is the estimated aleatoric noise and q(??(x)) is the epistemic uncertainty projected into output space.

Since our Bayesian neural networks only use a weight belief for the output layer, Var[q(??(x))] is Gaussian and can be computed in closed form.

In general, it the epistemic part of the predictive variance would be estimated by sampling.

In the classifier model, we use the OOD probability p(o = 1|x) for this.

For the deterministic neural network, we use Var[p(y | x)] as proxy since it does not output an estimate of epistemic uncertainty.

For visualization purposes, we start with experiments on a 1-dimensional regression task that consists of a sine function with a small slope and increasing variance for higher inputs.

Training data can be acquired within two bands, and the model is evaluated on all data points that are not visible to the model.

This structured split Test NLPD BBB+NCP ODC+NCP BBB Det Figure 4 : Active learning on the flights data set.

The models trained with NCP achieve significantly lower negative log predictive density (NLPD) on the test set, and Bayes by Backprop with NCP achieves the lowest root mean squared error (RMSE).

The test NLPD for the baseline models diverges as they overfit to the visible data points.

Plots show mean and std over 10 runs.between training and testing data causes a distributional shift at test time, requiring successful models to have reliable uncertainty estimates to avoid mispredictions for OOD inputs.

For this experiment, we use two layers of 200 hidden units, a batch size of 10, and a learning rate of 3 ?? 10 DISPLAYFORM0 for all models.

NCP models use noise ??? Normal(0, 0.5).

We start with 10 randomly selected initial targets, and select 1 additional target every 1000 epochs.

Figure 3 shows the root mean squared error (RMSE) and negative log predictive density (NLPD) throughout learning.

The two baseline models severely overfit to the training distribution early on when only few data points are visible.

Models with NCP outperform BBB, which in turn outperforms Det.

FIG0 visualizes the models' predictive distributions at the end of training, showing that NCP prevents overconfident generalization.

We consider the flight delay data set BID18 BID7 BID27 , a large scale regression benchmark with several published results.

The data set has 8 input variables describing a flight, and the target is the delay of the flight in minutes.

There are 700K training examples and 100K test examples.

The test set has a subtle distributional shift, since the 100K data points temporally follow after the training data.

We use two layers with 50 units each, a batch size of 10, and a learning rate of 10 ???4 .

For NCP models, ??? Normal(0, 0.1).

Starting from 10 labels, the models select a batch of 10 additional labels every 50 epochs.

The 700K data points of the training data set are available for acquisition, and we evaluate performance on the typical test split.

Figure 4 shows the performance for the visible data points and the test set respectively.

We note that BBB and BBB+NCP show similar NLPD on the visible data points, but the NCP models generalize better to unseen data.

Moreover, the Bayesian neural network with NCP achieves lower RMSE than the one without and the classifier based model achieves lower RMSE than the deterministic neural network.

All uncertainty-based models outperform the deterministic neural network.

The choice of input noise might seem like a critical hyper parameter for NCP.

In this experiment, we find that our method is robust to the choice of input noise.

The experimental setup is the same as for the active learning experiment described in Section 5.2, but with uniform or normal input noise with different variance (?? 2 x ??? {0.1, 0.2, ?? ?? ?? , 1.0}).

For uniform input noise, this means noise is drawn from the interval [???2?? x , 2?? x ].

We observe that BBB+NCP is robust to the size of the input noise.

NCP consistently improves RMSE for the tested noise sizes and yields the best NLPD for all noise sizes below 0.6.

For our ODC baseline, we

In addition to the active learning experiments, we perform a passive learning run on all 700K data points of the flights data set to explore the scalability of NCP.

We use networks of 3 layers with 1000 units and a learning rate of 10 ???4 .

Table 1 compares the performance of our models to previously published results.

We significantly improve state of the art performance on this data set.

Table 1 : Performance on all 700K data points of the flights data set.

While uncertainty estimates are not necessary when a large data set that is similar to the test data set is available, it shows that our method scales easily to large data sets.

gPoE BID7 8.1 -SAVIGP (Bonilla et al. 2016) 5.02 -SVI GP BID18 We develop noise contrastive priors (NCPs), a prior for neural networks in data space.

NCPs encourage network weights that not only explain the training data but also capture high uncertainty on OOD inputs.

We show that NCPs offer strong improvements over baselines and scale to large regression tasks.

We focused on active learning for regression tasks, where uncertainty is crucial for determining which data points to select next.

In future work it would be interesting to apply NCPs to alternative settings where uncertainty is important, such as image classification and learning with sparse or missing data.

In addition, NCPs are only one form of a data prior, designed to encourage uncertainty on OOD inputs.

Priors in data space can easily capture other properties such as periodicity or spatial invariance, and they may provide a scalable alternative to Gaussian process priors.

We showed how to apply NCP to a Bayesian neural network model that captures function uncertainty in a belief over parameters.

An alternative approach to capture uncertainty is to make explicit predictions about whether an input is OOD.

There is no belief over weights in this model.

FIG1 shows such a mixture model via a binary variable o, DISPLAYFORM0 where p(o = 1 | x) is the OOD probability of x. If o = 0 ("in distribution"), the model outputs the neural network prediction.

Otherwise, if o = 1 ("out of distribution"), the model uses a fixed output prior.

The neural network weights ?? are estimated using a point estimate, so we do not maintain a belief distribution over them.

The classifier prediction p(o | x, ??) captures uncertainty in this model.

We apply the NCP p(o |x, ??) = ??(o = 1|x, ??) to this variable, which assumes noised-up inputs to be OOD.

During training on the data set, {x, y} and o = 0 are observed, as training data are in-distribution by definition.

Following Equation 2, the loss function is DISPLAYFORM1 Analogously to the Bayesian neural network model in Section 3, we can either set ?? y , ?? 2 y manually or use the neural network prediction for potentially improved generalization.

In our experiments, we implement the OOD classifier model using a single neural network with two output layers that parameterize the Gaussian distribution and the binary distribution.

In Section 3, we derived the Bayes by Backprop model with NCP by adding a forward KL-divergence from the mean prior to the model mean to the loss.

An alternative derivation uses the fact that the KL-divergence is invariant to parameterization to replace the reverse KL-divergence in weight space by a KL-divergence in output space, E p(x,y) ln p(y | x) = E p(x,y) ln p(y | x, ??)p(??) q(??) q(??) d?? DISPLAYFORM0 where p(??(x)) = ??(x, ??)p(??) d?? and q(??(x)) = ??(x, ??)q(??) d?? are the distributions of the predicted mean induces by the weight beliefs.

As a result, instead of specifying a prior in weight space, we can specify a prior in output space.

Above, we reparameteterized the KL in weight space as a KL in output space; by the change of variables, this is equivalent if the mapping ??(??, ??) is continuous and 1-1 with respect to ??.

This assumption does not hold for neural nets as multiple parameter vectors can lead to the same predictive distribution, thus the approximation above.

A compact reparameterization of the neural network (equivalence class of parameteters) would make this an equality.

<|TLDR|>

@highlight

We train neural networks to be uncertain on noisy inputs to avoid overconfident predictions outside of the training distribution.

@highlight

Presents an approach to obtain uncertainty estimates for neural network predictions that has good performance when quantifying predictive uncertainty at points that are outside of the training distribution.

@highlight

The paper considers the problem of uncertainty estimation of neural networks and proposes to use Bayesian approach with noice contrastive prior