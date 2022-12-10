In one-class-learning tasks, only the normal case can be modeled with data, whereas the variation of all possible anomalies is too large to be described sufficiently by samples.

Thus, due to the lack of representative data, the wide-spread discriminative approaches cannot cover such learning tasks, and rather generative models, which attempt to learn the input density of the normal cases, are used.

However, generative models suffer from a large input dimensionality (as in images) and are typically inefficient learners.

We propose to learn the data distribution more efficiently with a multi-hypotheses autoencoder.

Moreover, the model is criticized by a discriminator, which prevents artificial data modes not supported by data, and which enforces diversity across hypotheses.

This consistency-based anomaly detection (ConAD) framework allows the reliable identification of outof- distribution samples.

For anomaly detection on CIFAR-10, it yields up to 3.9% points improvement over previously reported results.

On a real anomaly detection task, the approach reduces the error of the baseline models from 6.8% to 1.5%.

Anomaly detection classifies a sample as normal or abnormal.

In many applications, however, it must be treated as a one-class-learning problem, since the abnormal class cannot be defined sufficiently by samples.

Samples of the abnormal class can be extremely rare, or they do not cover the full space of possible anomalies.

For instance, in an autonomous driving system, we may have a test case with a bear or a kangaroo on the road.

For defect detection in manufacturing, new, unknown production anomalies due to critical changes in the production environment can appear.

In medical data analysis, there can be unknown deviations from the healthy state.

In all these cases, the well-studied discriminative models, where decision boundaries of classifiers are learned from training samples of all classes, cannot be applied.

The decision boundary learning of discriminative models will be dominated by the normal class, which will negatively influence the classification performance.

Anomaly detection as one-class learning is typically approached by generative, reconstruction-based methods BID30 .

They approximate the input distribution of the normal cases by parametric models, which allow them to reconstruct input samples from this distribution.

At test time, the data log-likelihood serves as an anomaly-score.

In the case of high-dimensional inputs, such as images, learning a representative distribution model of the normal class is hard and requires many samples.

Typically, an autoencoder-based approach such as the variational autoencoder BID21 BID13 ) is used.

Autoencoders tend to produce blurry reconstructions, since they regress the conditional mean, and cannot model multi-modal distributions; see FIG0 for an example on a Metal Anomaly dataset.

Due to multiple modes in the actual distribution, the approximation with the mean predicts high probabilities in areas not supported by samples.

The blurry reconstructions in FIG0 should have a low probability and be classified as anomalies, but they have the highest likelihood under the learned autoencoder.

Multiple-hypotheses networks could give the model more expressive power BID23 , BID5 , BID11 , BID2 .

In conjunction with autoencoders, the multiple hypotheses can be realized with a multi-headed decoder.

Concretely, each network head may predict a Gaussian density estimate.

gives the network more expressive power with a multi-headed decoder (also known as multiple-hypotheses networks).

The resulting anomaly scores are hence much clearer in our framework ConAD.

were not yet applied to anomaly detection due to several difficulties in training these networks to produce a multi-modal distribution consistent with the training distribution.

The loosely coupled hypotheses branches are typically learned with a winner-takes-all loss, where all learning signal is transferred to one single best branch.

Hence, bad hypotheses branches are not penalized and may support non-existing data regions.

The artificial data modes, therefore, cannot be distinguished from normal data.

This is an undesired property for anomaly detection and becomes more severe with an increasing number of hypotheses.

Furthermore, the majority of multiple-hypotheses-branches tend to concentrate on the most dominant data modes.

This hypotheses concentration leads to over-fitting in the neighborhood of dominant modes and under-fitting in underrepresented data regions.

This, too, has a negative effect on the estimated anomaly scores.

Alternatively, mixture density networks (MDNs) BID3 provide a strict coupling of hypotheses branches.

These models learn a conditional Gaussian mixture distribution.

Hence, the hypotheses are coupled via mixing coefficients into a single likelihood function.

Anomaly scores for new points can be estimated using the data likelihood, as formally defined in Appendix A. FIG1 illustrates the different strategies.

A single-mode autoencoder (b) fails in case of multi-modal distributions.

MDNs (c) in principle can be used for abnormality detection even for multimodal distributions.

However, global, multi-modal distribution estimation is a hard learning problem that does not work as perfectly in practice as shown in this illustration.

For instance, MDNs tend to suffer from mode collapse in high-dimensional data spaces, i.e., the relevant data modes needed to distinguish rare but normal data from anomalies will be missed.

Contrary, Local-outlier-factor operates in images-space directly without training which (1) fails in very high-dimensional spaces (2) is slow at test time.

In this work, we adopt multiple-hypotheses networks for anomaly detection to provide a more finegrained description of the data distribution than a single-headed network.

Hypotheses are meant to form clusters in the data space and can capture model uncertainty not encoded by the latent code.

We reduce the problem of artificial data modes by combining multiple-hypotheses learning with a discriminator D as a critic.

The discriminator ensures the consistency of estimated data modes w.r.t.

the real data distribution.

Moreover, we propose to focus on the local neighborhood and to estimate the fit of a sample to the distribution model based on the distance to the closest cluster.

This avoids issues with global distribution estimation methods, such as mode collapse.

Hypotheses rather act as local, single mode density estimates and are easier and more sample-efficient to learn than a full multi-modal distribution.

Fig. 3c shows our framework applied to a variational autoencoder.

We evaluate anomaly detection performance of our approach on CIFAR-10 and a real anomaly image dataset, the "Metal Anomaly dataset" with images showing a structured metal surface, where anomalies in the form of scratches, dents or texture differences are to be detected.

We show that anomaly detection performance with multiple-hypotheses networks is significantly better compared The mixture density network (c) in principle can do so, but recognition of the sample as a normal case is very brittle and will fail in case of mode collapse.

In contrast, Local-Outlier-Factor (d) and our model (e) consider only the local neighborhood for anomaly score estimation and more reliably classify the point.

In our model, we encourage multiple hypotheses to cover different modes.

In each hypothesis branch, the probability mass is distributed only within the cluster and not beyond.to single-hypotheses networks.

On CIFAR-10, our proposed ConAD framework (consistency-based anomaly detection) improves on previously published results.

Furthermore, we show a large performance gap between ConAD and Mixture Density networks (MDNs).

This indicates that anomaly score estimation based on the global neighborhood (or data likelihood) is inferior to local neighborhood consideration.

Traditional one-class learning techniques BID27 BID29 BID17 BID4 often fail in high-dimensional input domain and require careful features selection BID30 .

To cope with high-dimensional domains, typically a reconstructionbased approach is used.

This paradigm comprises two steps: (1) during training, learn the normal data distribution and (2) at test time, use the negative likelihood for contaminated data as their anomaly score.

Recently, advances in generative modeling such as Generative Adversarial Network (GAN) BID10 and Variational Autoencoder (VAE) BID21 BID13 are used for anomaly detection BID30 BID26 .

However, GAN and VAE approaches have limitations in anomaly detection tasks.

The GAN tends to assign less probability mass to real samples while VAE typically regress to the conditional means, which can be seen from the blurry reconstructions.

The mean regression in VAE express the model uncertainty and falsify the reconstruction-errors for unseen images.

One simple way to address model uncertainty in VAE is giving the decoder additional expressive power with multi-headed decoders.

The idea is to approximate multiple conditional modes (dense data regions) by using multiple headed networks.

This idea leads to training of multiple networks in Multi-Choice-learning BID7 BID15 , the estimation of conditional Gaussian Mixture model in Mixture Density Network (MDN) BID3 and multiple-hypotheses predictions (MHP) BID11 BID5 BID2 BID23 .

In MDN, the mixtures are strictly coupled via mixture coefficients while mixtures in MHPs act as loosely coupled local density estimators.

In MHP, only the best hypothesis branch will receive a learning signal, that is, if it makes the closest guess to the training sample.

For anomaly detection, our model uses MHP-training with VAE to address the model uncertainty directly.

In MDN, the anomaly score is proportional to weighted distances to all data modes and in MHP only to closest data mode.

To highlight the change in paradigm, we refer to this learning in MHP as consistency-based learning.

Samples have a small effect on the loss as long they are close to one single data mode.

The learning dynamic in MHP is also different and more efficient than in MDN: the number of samples with a high loss is lower.

In this context, we relax the learning objective from density-based to consistency-based learning.

In Local Outlier Factor (LOF) BID4 , the outlier-score only depends on the local neighborhood.

The outlier score proportional to the mean density of neighboring points divided by the local point density.

Hence, samples further away do not influence the outlier-score.

Motivated by this heuristic, our model employs learning of many loosely decoupled local density estimates with MHP-learning.

Our model FORMULA0 concentrates only on the closest data mode instead of considering the data likelihood for outlier detection FORMULA1 and enables easier learning due to consistency-based learning instead of full density estimation.

LOF computes the outlierness only on test-time and in input spaces directly.

Contrary, our model first approximate the data manifold and subsequently performs anomaly detection in the input space under the learned model.

The MHP-technique has been used for uncertain tasks such as future prediction BID24 or optical flow prediction BID11 .

In the simplest form, the multiple networks heads learn from a winner-takes-all (WTA) loss, whereby only the best branch receives the learning signal.

Previous works employ loss extension such as the use of a smoothing loss BID11 or distribution of learning signal to non-optimal branches BID24 to generate diverse and meaningful hypotheses.

Compared to our framework, previous MHP-approaches were not developed for distribution learning.

There is no explicit mechanism to avoid mode collapse among hypotheses.

Furthermore, generated hypotheses could support non-existing data regions, which can be fatal for anomaly detection tasks.

Contrary, our framework ConAD employs a discriminator D to assess the quality of the generated hypotheses and to avoid support of non-existent data modes.

To reduce hypotheses mode collapse, our model employs hypotheses discrimination.

In the spirit of minibatch discrimination BID25 , D additionally receives pair-wise distances across a batch of hypotheses.

Since a batch of real samples is typically diverse, D can detect a homogeneous batch of hypotheses as fake easily.

Typically in distribution learning, Autoencoder-approaches regress the means and produce blurry reconstructions.

Therefore, we propose to employ MHP as additional expressive power for the decoder (Fig 3 (a-b) ).

First, we discuss two possible shortcomings of multiple-hypotheses learning: support of artificial data mode and hypotheses mode collapse.

Subsequently, we show how to reduce these effects with discriminator training and hypotheses discrimination (Fig 3 c) .

Support of artificial data mode in one-to-many mapping tasks To understand the shortcomings of learning with multiple-hypotheses-proposals (MHP), first consider a simple one-to-many mapping task from x to y as given in FIG8 .

Unimodal models (i.e., single-headed networks) fail to capture to data distribution.

Figure 4: Flipped half-moon data-set: mapping from x to y is not unique, e.g. for x = 0, there are four different modes.

Left to right: with an increasing number of mixture components in a mixture density network, the data distribution can be modeled increasingly well.

Similar to Mixture Density networks, each hypothesis branch in MHP-networks represents a Gaussian density function with a mean and variance.

Typically, MHP-networks learns from the winnertakes-all (MHP-WTA) loss in Eq. 1: DISPLAYFORM0 Whereby θ j is the parameter set of hypothesis branch j, θ h the best hypothesis concerning data likelihood given a sample x i .

In other words, only the network head with the best-matching hypothesis concerning the training samples receives the learning signal.

The best hypothesis is the one with the highest sample likelihood (or minimal distance to sample if the variance is equal for all hypotheses).

Additionally, BID23 proposed a -smoothed loss.

With this loss, a small -ratio of the learning signal is distributed among non-optimal hypotheses branches.

We refer to this loss as learning with MHP-loss (Appendix 11).However, learning with MHP or MHP-WTA may result in support of artificial (non-existing) data modes.

FIG3 illustrates this problem, which we refer to as inconsistency concerning the underlying distribution.

In regions where the half-moon abruptly ends, the hypotheses (in MHP and MHP-WTA) continue and support non-existing data regions.

This inconsistency effect is fatal for anomaly detection.

More details can be found in the experiments on the toy dataset in the appendix B. Intuitively, in learning with the winner-takes-all loss, the non-optimal hypotheses are not penalized.

Therefore they can support artificial data regions without being informed via the learning signal.

A more formal discussion can be found in Appendix D.The learning signal distribution with -parameter attempts to reduce support of artificial regions.

However, finding good is crucial and difficult.

If = 0, the MHP loss corresponds to learning with MHP-WTA.

If = H−1 H , whereby H is the number of hypotheses branches, all hypotheses will regress to the same conditional mean.

A more formal discussion can be found in Appendix E. Additionally, is an additional hyper-parameter to be chosen.

Choosing proper hyper-parameters in one-class-learning is difficult since there is no anomaly available at training time.

Distribution learning with Autoencoder as a one-to-many mapping task Training Autoencoders with likelihood-metric often results in blurry reconstructions.

This blurriness is fatal for anomaly detection since it falsifies the reconstructions error.

This effect can be understood as a regression to the conditional mean.

That means, after training convergence, each point on the learned manifold still represents many different data points in the input space.

In other words, the mapping from latent code to input space is a one-to-many mapping.

Certainly, in the optimal training case, each point on the data manifold should represent one single input vector.

However, this optimality requires either significantly more data to reduce the model uncertainty or powerful encoder network and latent code or both.

Contrary, we propose to let the Autoencoder express the model uncertainty with the multiple-hypotheses directly.

Hence, the change to Autoencoder is very simple, and no more data is required than before.

Mode collapse across hypotheses Furthermore, with the MHP and MHP-WTA learning objective, the hypotheses are encouraged to cover the existing modes.

When there are more hypotheses available than data modes, most of the hypotheses will tend to concentrate on the most dominant data modes.

This mode collapse can be avoided by enforcing diversity across hypotheses, which is similar to maximizing inter-class variance across clusters defined by the hypotheses.

We propose multiple-hypotheses Variational Autoencoder (VAE) for learning the normal data distribution for anomaly detection tasks.

Each hypothesis branch can be seen as a cluster in the data conditional space.

Anomalies are detected using the distance to next local clusters, in contrast to distances to all clusters in Mixture Density networks (MDN) BID3 .

To avoid coverage of non-existing data regions by the hypotheses, we propose to use a discriminator as a critic.

Further, we employ hypothesis discrimination to encourage diversity among hypotheses.

This constraint is similar to the improvement of inter-class variance among clusters.

The details are explained in the following.

In this work, we consider distribution learning in an Autoencoder as a one-to-many-mapping.

We propose to let the network express the model uncertainty in the conditional input space with multiple hypotheses predictions (MHP).

The hypotheses can be seen as a set of local density estimates (or cluster).

In contrast to that, Mixture Density Network (MDN) predicts a Gaussian Mixture model in the conditional space.

We refer to this estimate as a global density estimate.

The learning of different hypotheses is performed based on a winner-takes-all-objective as given in Eq. 2.

DISPLAYFORM0 Whereby L W T A is the winner-takes-all energy function, 1 ≤ j ≤ H indicates the different hypotheses networks, z i the respective latent code.

To reduce free parameters, hypotheses networks with params θ j share all layers but the last output layer.

Intuitive, it means that only the best matching hypothesis receives all of the learning signals from the negative log-likelihood (NLL) loss during training.

An efficient variant to realize MHP in neural networks is by using multi-headed-networks.

In this variant, only the last layer is split to provide different hypotheses.

All other layers are shared as shown in Fig. 3c .

Our framework is based on the Variational Autoencoder BID13 BID21 which provides an effective manifold learning and an efficient inference stage with a parameterized encoder q φ .Discriminator D to avoid non-existent mode coverage and mode collapse of hypotheses Hypotheses generated by the MHP-networks could support artificial data regions not covered by real samples due to the WTA loss.

To alleviate this, we propose to match the density estimates with MHP to the real underlying density.

The auxiliary task is to learn from a symmetric variant of the Kullback-Leibler divergence (KLD).

In detail, we employ the Jensen-Shannon divergence (JSD)-metric by using discriminator D as a critic for generated hypotheses.

Fig. 3c illustrates a sample realization with VAE.More concretely, the D and G are in a mini-max game in Eq. 3.

DISPLAYFORM1 In this energy formulation, the standard GAN loss is extended to assure the quality of generated hypotheses.

FIG5 illustrates how samples are fed into the discriminator.

Samples labeled as fake are: randomly-sampled imagesx z∼N (0,1) , data reconstruction defined by individual hypothesesx z∼N (µ z|x ,Σ z|x ) , the best combination of hypotheses according to the Winner-takes-all-losŝ x best guess .Accordingly, the learning objective for the VAE generator becomes: To address the mode collapse problem of hypotheses, we propose to employ hypotheses discrimination (based on minibatch discrimination BID25 ).

In each batch, the discriminator receives the pair-wise features distance across generated hypotheses.

Since batches of real images have large pair-wise distances, the generator has to generate diverse outputs to avoid being detected too easily.

DISPLAYFORM2 In summary, our framework ConAD proposes multiple-hypotheses learning with a VAE, supported by a discriminator D to avoid support of non-existing data modes and foster mode coverage.

The local likelihood estimates given by the closest hypothesis are used for anomaly detection.

In this section, we focus on the evaluation of our approach compared to recent deep learning and non-deep learning techniques for one-class learning tasks.

In these tasks, anomalies are extremely rare and hence not available at training time.

The main effort comes from the collection of a large dataset to receive anomalies, not from the labeling activity.

The details of the proposed framework; consistency-based anomaly detection (ConAD) is explained in the following.

A Variational Autoencoder Kingma & Welling (2013) with Gaussian output distribution is employed as a baseline model.

The decoder is then extended to a multiple-head-network to support multiple-hypotheses.

Each hypothesis itself predicts a Gaussian density estimate.

The outputs from the Autoencoders are criticized by a discriminator D. The network architecture follows principles from BID20 and BID28 .

Fig. 3 c) shows such a network conceptually.

The framework can be easily extended to recent advances in deep generative modeling.

Quantitative evaluation is done on CIFAR-10 and the Metal Anomaly dataset.

The typical 10-way classification task in CIFAR-10 is transformed into 10 one vs. nine anomaly detection tasks.

Each class is used as the normal class once; all remaining classes are treated as anomalies.

Details can be found in Tab.

1.

During model training, only data from the normal data class is used, data from anomalous classes are abandoned.

At test time, anomaly detection performance is measured in Area-Under-Curve of Receiver Operating Curve (AUROC) based on normalized negative log likelihood scores given by the training objective.

In Tab BID27 ).

The performance of traditional methods suffers due to the curse of dimensionality BID30 .Furthermore, on the high-dimensional Metal anomaly dataset, we focus only on the evaluation of deep learning techniques.

The GAN-techniques proposed by previous work AdGAN & AnoGAN heavily suffer from instability due to pure GAN-training on a small dataset.

Hence, their training leads to random anomaly detection performance.

Therefore, we only evaluate MHP-based approaches against their uni-modal counterparts (VAE, VAEGAN).

Table 3 : Anomaly detection performance on Metal Anomaly dataset.

To reduce noisy residuals due to the high-dimensional input domain, only 10% of maximally abnormal pixels with the highest residuals are summed to form the total anomaly score.

AUROC is computed on an unseen test set, a combination of normal and anomaly data.

For more detailed results, refer to attachment H. Anomaly detection performance of plain MHP rapidly breaks down with increasing number of hypotheses.

Tab.

2 shows an extensive evaluation of different traditional and deep learning techniques.

Results are adapted from in which the training and testing scenarios were similar.

Refer to Appendix.

G for more results.

Traditional, non-deep-learning methods only succeed to capture classes with a dominant homogeneous background such as ships, planes, frogs (backgrounds are water, sky, green nature respectively).

This issue occurs due to preceding feature projection with PCA, which focuses on dominant axes with large variance. reported that even discriminative features from a pretrained AlexNet have no positive effect on anomaly detection performance.

In contrast to that, deep learning methods are performing significantly better, even without careful parameter tuning.

When the MHP-technique is applied to this task, a performance comparable to previously reported deep learning, but non-MHP results is achieved.

Note that having the multiple output distributions is not sufficient to meet high performance: MDNs are performing worse than the local density estimation provided by the MHP-technique.

Nevertheless, the best performance is achieved in our ConAD-framework, by utilizing the flexibility of multiple hypotheses more effectively, leading to significantly higher detection performance of up to 5.1% absolute improvement.

Tab.3 shows an evaluation of MHP-methods against density-learning methods such as VAE BID13 , MDN (Bishop, 1994) , VAEGAN BID9 BID14 .

Note that the VAE-GAN model corresponds to our ConAD with a single hypothesis.

The VAE corresponds to a single hypothesis variant of MHP, MHP-WTA, and MDN.The significant improvement of up to 4.2% AUROC-score comes from our relaxation of density estimation into local density estimation in the spirit of LOF BID4 , i.e., each dense data region (mode) receives at least one hypothesis to cover the local density.

In a high-dimensional domain such as images, anomaly detection with MDN is worse than with our approach MHP approaches.

Consider images with an extremely rare value in one pixel-dimension.

The Mixture Density models evaluate likelihood based on all data modes found for this pixel.

In contrast to that, MHP-models only considers which data mode is the closest and computes the local likelihood as the anomaly score.

The local neighborhood suppresses the over-estimation of anomaly degree compared to a global likelihood.

Using the MHP-technique, better performance is already achieved with two hypotheses.

However, without the discriminator D, an increasing number of hypotheses rapidly leads to performance breakdown, due to the inconsistency property of generated hypotheses as discussed earlier.

Intuitively, additional non-optimal hypotheses are not strongly penalized during training, if they support artificial data regions which are not consistent w.r.t.

the real underlying data distribution.

With our framework ConAD, anomaly detection performance remains competitive or better even with an increasing number of hypotheses available.

The discriminator D makes the framework adaptable to the new dataset and less sensitive to the number of hypotheses to be used.

When more hypotheses are used (8), the anomaly detection performance rapidly breaks down.

We suggest that the noise is then learned too easily.

Consider the extreme case when there are 255 hypotheses available.

The Winner-Takes-all-loss will encourage each hypothesis branch to predict a constant image with one value from [0, 255] .

The discriminator D as a regularizer will try to prevent this effect.

That might be a reason why our ConAD has less severe performance breakdown.

Our model ConAD is less sensitive to the choice of the hyper-parameter for the number of hypotheses.

It also enables better exploitation of the additional expressive power provided by the MHP-technique for new anomaly detection tasks.

In this work, we propose to employ multiple-hypotheses networks for learning data distributions for anomaly detection tasks.

Hypotheses are meant to form clusters in the data space and can easily capture model uncertainty not encoded by the latent code.

multiple-hypotheses networks can provide a more fine-grained description of the data distribution and therefore enable also a more fine-grained anomaly detection.

Furthermore, to reduce support of artificial data modes by hypotheses learning, we propose using a discriminator D as a critic.

The combination of multiple-hypotheses learning with D aims to retain the consistency of estimated data modes w.r.t.

the real data distribution.

Further, D encourage diversity across hypotheses with hypotheses discrimination.

Our framework allows the model to identify out-of-distribution samples reliably.

For the anomaly detection task on CIFAR-10, our proposed model results in up to 3.9% points improvement over previously reported results.

On a real anomaly detection task, the approach reduces the error of the baseline models from 6.8% to 1.5%.

The Mixture Density networks predict a data conditional Gaussian mixture model (GMM)in the data space.

Conditioning means that each latent vector, i.e., a point on the learned manifold is projected back to a GMM in the data space.

A GMM learns from the following energy function: DISPLAYFORM0 Whereby x is the input data, µ h and σ h parametrize the h − th Gaussian distribution in the mixture.

α h are the mixing coefficients across the individual mixtures.

Contrary, a Mixture Density network hat multiple output heads (multiple-hypotheses).

The framework extends the GMM-learning by the data conditioning as follows: DISPLAYFORM1 whereby q φ is a inference network shared by all individual mixtures.

z is the latent code.

The hypotheses are coupled into forming a likelihood function by the mixing coefficients α i .

This task is a one-to-many mapping from x to y with a discontinuity at the point x = 0 and x = 0.5.

When the local density function abruptly ends, MHP-techniques support artificial data regions since they are not penalized for artificial modes by the objective function as discussed before.

We refer to this property as an inconsistency concerning the true underlying distribution.

In contrast to that, Mixture Density Networks (MDN) and our ConADs approaches reduce the inconsistencies to the minimum.

Consider a simple toy problem with an observable x and hidden y which is to be predicted and expressed by the conditional distribution p true (y|x) such as in FIG8 .

Since the data conditional is multi-modal for some x, an uni-modal output distribution cannot fully capture the underlying distribution.

Instead, the bias-free solution for the Mean-Squared-Error-minimizer is the empirical mean y xi of p train (y|x i ) on the training set.

However, this learned conditional density does not comply with the underlying distribution: sampled data points fall into the low-likelihood regions under p true (y|x).

With increasing number of output hypotheses, the data modes could be gradually captured.

For this task, the energy to be minimized is given by the Negative-log-likelihood of the Mixture Density Network (MDN) App.

A under a Gaussian Mixture with hypotheses h in Eq. 9 : DISPLAYFORM0 D LEMMA 4.1Given a sufficient number of hypotheses H', an optimal solution Θ * for E W T A (Θ * ) is not unique (permutation is excluded).

There exists a Θ with E W T A (Θ * ) = E W T A (Θ ) which is not consistent w.r.t.

the underlying output distribution p train (y i |x i ).Proof. : Suppose c is the maximal modes count of the dataset sampled from the real underlying conditional output distribution p(y i |x i ).

Since |{(x i , y i )}| < ∞ → c < ∞. Suppose H = c, then a trivial optimal solution for E W T A (Θ H ) is found by centering each hypothesis µ ik at a different empirical data point k y ik ∼ (y i , x i ) and σ ik → 0.

In this case lim DISPLAYFORM1 . .

θ h } for some random Θ H+1...H .

Due to randomness and without loss of generality, one can assume that ∀(x i , y i ), ∀θ i ∈ Θ H+1...H , θ i is not the optimal hypothesis for any training point DISPLAYFORM2 In this case due to the winner-takes-all energy formulation we have: DISPLAYFORM3 So Θ H and Θ H with H > H are both solutions to the loss formulation and share the same energy level.

The extended hypotheses can support arbitrary artificial data regions without being penalized.

E LEMMA 4.2 DISPLAYFORM4 Whereby x i ,y i is corresponding input-output pairs from the training dataset, 1 ≤ h ≤ H is a hypothesis branch, which is generated by a parametrized neural network with the parameter set θ h .

Furthermore, is a hyperparameter used to distribute the learning signal to the non-optimal hypotheses.

Θ is the collection of all θ h .

Lemma E.1.

Similar to Lemma D, minimizing E M HP in Eq. 11 might also lead to an inconsistent approximation of the real underlying output distribution.

DISPLAYFORM5 ∀θ h and training data points (x i , y ik ) the optimal least-squares solution is the mean, therefore we have: DISPLAYFORM6 In this case, all hypotheses are optimized independently and converge to the same solution similar to a single-hypothesis approach.

The resulting distribution is inconsistent w.r.t the real output distribution (see FIG8 for an example).Now consider → 1: DISPLAYFORM7 In this case E M HP shares the same inconsistency property with E W T A .

Consequently, choosing ∈ [0, DISPLAYFORM8 H ] only smoothes the penalty on suboptimal hypotheses.

The risk remains that distributions induced by non-optimal hypotheses are beyond the real modes of the underlying distribution.

Network architecture The networks are following DCGAN BID19 but only scaled down to support low-resolution of CIFAR-10.

Concretely, the decoder (generator) only uses deconvolutional layers.

Throughout the network, leaky-relu units are employed.

The framework is implemented in Lasagne (Dieleman et al., 2015) /Theano BID1 BID0 .Hypotheses branches are represented as decoder networks heads.

Each hypothesis predicts one Gaussian distribution with diagonal co-variance Σ and mean .

The winner-takes-all loss operates on pixel-level,i.e., for each predicted pixel, there is a single winner across hypotheses.

The bestcombined-reconstructions is the combination of winning hypotheses on pixel-level.

Training We feed the fake images to the discriminator D, consisting of 4 batches:• real n real images• fake: n random hypotheses from image reconstructions hypotheses branches• fake: n best-combined (based on winner hypotheses) reconstructions• fake: n random sampled images from latent prior N (0, 1)The batch-size n was set to 64 each on CIFAR-10, 32 on Metal Anomaly.

The training was performed with Adam (Kingma & Ba, 2014 ) with a learning rate of 0.001.

Per discriminator training, the generator is trained at most five epochs to balance both players. . . .

.659 Table 4 : CIFAR-10 anomaly detection: AUROC-performance of different approaches.

The column indicates which class was used as in-class data for distribution learning.

Note that random performance is at 50% and higher scores are better.

Top-2-methods are marked.

Our ConAD approach outperforms traditional methods and vanilla MHP-approaches significantly and can benefit from an increasing number of hypotheses.

Furthermore, Mixture Density Networks perform similarly to uni-modal output distributions of VAEs.

H METAL ANOMALY DATASET Figure 9 : Metal Anomaly dataset.

Image reconstructions: reconstructions from uni-modal models are blurry at convergence.

Using our ConAD-approach (last two rows), the maximally consistent reconstruction is closer to the original image, capturing many more details needed to differentiate between normal data noise and real anomalies, such as black spots or scratches.

The likelihood maximizer in the hypotheses space is much closer to the original and also more realistic.

The residuals are significantly clearer for our ConAD-method.

Table 5 : Anomaly detection performance on the Metal Anomaly dataset, measured in AUROC, showing how different multiple hypothesis approaches perform with increasing number of hypotheses.

Vanilla single-hypothesis approaches such as VAE and VAE+GAN under-perform on this task.

Even with more sophisticated multi-modal output distribution capacity (MDN), the discriminability is not improved.

The integration of MDN into the GAN-framework only slightly improves the results.

On the other hand, all other MHP-approaches perform similarly well with > 99% AUROC (at 1% of most abnormal pixels considered), which indicates that the task has become easily solvable for these methods.

@highlight

We propose an anomaly-detection approach that combines modeling the foreground class via multiple local densities with adversarial training.

@highlight

The paper proposes a technique to make generative models more robust by making them consistent with the local density.