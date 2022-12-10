Uncertainty estimation and ensembling methods go hand-in-hand.

Uncertainty estimation is one of the main benchmarks for assessment of ensembling performance.

At the same time, deep learning ensembles have provided state-of-the-art results in uncertainty estimation.

In this work, we focus on in-domain uncertainty for image classification.

We explore the standards for its quantification and point out pitfalls of existing metrics.

Avoiding these pitfalls, we perform a broad study of different ensembling techniques.

To provide more insight in the broad comparison, we introduce the deep ensemble equivalent (DEE) and show that many sophisticated ensembling techniques are equivalent to an ensemble of very few independently trained networks in terms of the test log-likelihood.

Deep neural networks (DNNs) have become one of the most popular families of machine learning models.

The predictive performance of DNNs for classification is often measured in terms of accuracy.

However, DNNs have been shown to yield inaccurate and unreliable probability estimates, or predictive uncertainty (Guo et al., 2017) .

This has brought considerable attention to the problem of uncertainty estimation with deep neural networks.

There are many faces to uncertainty estimation.

Different desirable uncertainty estimation properties of a model require different settings and metrics to capture them.

Out-of-domain uncertainty of the model is measured on data that does not follow the same distribution as the training dataset (out-of-domain data).

Out-of-domain data can include images corrupted with rotations or blurring, adversarial attacks (Szegedy et al., 2013) or data points from a completely different dataset.

The model is expected to be resistant to data corruptions and to be more uncertain on out-of-domain data than on in-domain data.

This setting was explored in a recent study by (Ovadia et al., 2019) .

On the contrary, in-domain uncertainty of the model is measured on data taken from the training data distribution, i.e. data from the same domain.

In this case, a model is expected to provide correct probability estimates: it should not be overconfident in the wrong predictions, and should not be too uncertain about the correct predictions.

Ensembles of deep neural networks have become a de-facto standard for uncertainty estimation and improving the quality of deep learning models (Hansen & Salamon, 1990; Krizhevsky et al., 2009; Lakshminarayanan et al., 2017) .

There are two main directions in the field of training ensembles of DNNs: training stochastic computation graphs and obtaining separate snapshots of neural network weights.

Methods based on the paradigm of stochastic computation graphs introduce noise over weights or activations of deep learning models.

When the model is trained, each sample of the noise corresponds to a member of the ensemble.

During test time, the predictions are averaged across the noise samples.

These methods include (test-time) data augmentation, dropout (Srivastava et al., 2014; Gal & Ghahramani, 2016) , variational inference (Blundell et al., 2015; Kingma et al., 2015; Louizos & Welling, 2017) , batch normalization (Ioffe & Szegedy, 2015; Teye et al., 2018; Atanov et al., 2019) , Laplace approximation (Ritter et al., 2018) and many more.

Snapshot-based methods aim to obtain sets of weights for deep learning models and then to average the predictions across these weights.

The weights can be trained independently (e.g., deep ensembles (Lakshminarayanan et al., 2017) ), collected on different stages of a training trajectory (e.g., snapshot ensembles (Huang et al., 2017) and fast geometric ensembles (Garipov et al., 2018) ), or obtained from a sampling process (e.g., MCMC-based methods (Welling & Teh, 2011; Zhang et al., 2019) ).

These two paradigms can be combined.

Some works suggest construction of ensembles of stochastic computation graphs (Tomczak et al., 2018) , while others make use of the collected snapshots to construct a stochastic computation graph (Wang et al., 2018; Maddox et al., 2019) .

In this paper, we focus on assessing the quality of in-domain uncertainty estimation.

We show that many common metrics in the field are either not comparable across different models or fail to provide a reliable ranking, and then address some of stated pitfalls.

Following that, we perform a broad evaluation of modern DNN ensembles on CIFAR-10/100 and ImageNet datasets.

To aid interpretatability, we introduce the deep ensemble equivalent score that essentially measures the number of "independent" models in an ensemble of DNNs.

We draw a set of conclusions with regard to ensembling performance and metric reliability to guide future research practices.

For example, we find that methods specifically designed to traverse different "optima" of the loss function (snapshot ensembles and cyclical SGLD) come close to matching the performance of deep ensembles while methods that only explore the vicinity of a single "optimum" (Dropout, FGE, K-FAC Laplace and variational inference) fall far behind.

We use standard benchmark problems of image classification as it is a common setting in papers on learning ensembles of neural networks.

There are other practically relevant settings where the correctness of probabilistic estimates can be a priority.

These settings include, but are not limited to, regression, image segmentation, language modelling (Gal, 2016) , active learning (Settles, 2012) and reinforcement learning (Buckman et al., 2018; Chua et al., 2018) .

We focus on in-domain uncertainty, as opposed to out-of-domain uncertainty.

Out-of-domain uncertainty includes detection of inputs that come from a completely different domain or have been corrupted by noise or adversarial attacks.

This setting has been thoroughly explored by (Ovadia et al., 2019) .

We only consider methods that are trained on clean data with simple data augmentation.

Some other methods use out-of-domain data (Malinin & Gales, 2018) or more elaborate data augmentation e.g., mixup (Zhang et al., 2017) , adversarial training (Lakshminarayanan et al., 2017) to improve accuracy, robustness and uncertainty.

We use conventional training procedures.

We use the stochastic gradient descent (SGD) and use batch normalization (Ioffe & Szegedy, 2015) , both being the de-facto standards in modern deep learning.

We refrain from using more elaborate optimization techniques including works on superconvergence (Smith & Topin, 2019) and stochastic weight averaging (Izmailov et al., 2018) .

These techniques can be used to drastically accelerate training and improve the predictive performance.

Because of that, we do not not comment on the training time of different ensembling methods since the use of more efficient training techniques would render such a comparison obsolete.

A number of related works study ways of approximating and accelerating prediction in ensembles.

The distillation mechanism allows to approximate the prediction of an ensemble by a single neural network (Hinton et al., 2015; Balan et al., 2015) , whereas fast dropout (Wang & Manning, 2013 ) and deterministic variational inference (Wu et al., 2018) allow to approximate the predictive distribution of specific stochastic computation graphs.

We measure the raw power of ensembling techniques without these approximations.

All of the aforementioned alternative settings are orthogonal to the scope of this paper and are promising points of interest for further research.

No single metric measures all desirable properties of uncertainty estimates obtained with a model.

Because of this, the community has used different metrics that aim to measure the quality of uncertainty estimation, e.g. the Brier score (Brier, 1950 ), log-likelihood (Quinonero-Candela et al., 2005 , different calibration metrics (Guo et al., 2017; Nixon et al., 2019) , performance of misclassification detection (Malinin & Gales, 2018) , and threshold-accuracy curves (Lakshminarayanan et al., 2017) .

We consider a classification problem with a dataset that consists of N training and n testing pairs (x i , y * i ) ∼ p(x, y), where x i is an object and y * i ∈ {1, . . .

, C} is a discrete class label.

A probabilistic classifier maps an object x i into a predictive distributionp(y | x i ).

The predictive distributionp(y | x i ) of deep neural networks is usually defined as a softmax function p(y | x) = Softmax(z(x)/T ), where z(x) is a vector of logits and T is a scalar parameter standing for the temperature of the predictive distribution.

The maximum probability max cp (y = c | x i ) is called a confidence of a classifierp on object x i .

The indicator function is denoted by I[·] throughout the text.

The average test log-likelihood LL = 1 n n i=1 logp(y = y * i | x i ) is a popular metric for measuring the quality of in-domain uncertainty of deep learning models.

It directly penalizes high probability scores assigned to incorrect labels and low probability scores assigned to the correct labels y * i .

LL is sensitive to the temperature T .

The temperature that has been learned during training can be far from optimal for the test data.

However, a nearly optimal temperature can be found post-hoc by maximizing the log-likelihood on validation data.

This approach is called temperature scaling or calibration (Guo et al., 2017) .

Despite its simplicity, temperature scaling results in a marked improvement in the LL.

While ensembling techniques tend to have better temperature than single models, the default choice of T = 1 is still sub-optimal.

Comparing the LL with sub-optimal temperatures-that is often the case-can produce an arbitrary ranking of different methods.

Comparison of the log-likelihood should only be performed at the optimal temperature.

Empirically, we demonstrate that the overall ordering of methods and also the best ensembling method according to the LL can vary depending on temperature T .

While this applies to most ensembling techniques (see Appendix C), this effect is most noticeable on experiments with data augmentation on ImageNet (Figure 1 ).

We will call the log-likelihood at the optimal temperature the calibrated log-likelihood.

We show how to obtain an unbiased estimate of the calibrated loglikelihood without a held-out validation set in Section 3.5.

LL also demonstrates a high correlation with accuracy (ρ > 0.86), that in case of calibrated LL becomes even stronger (ρ > 0.95).

That suggest that while (calibrated) LL measures the uncertainty of the model, it still significantly depends on the accuracy and vice versa.

A model with higher accuracy would likely have a higher log-likelihood even if the quality of its uncertainty is lower in some respects.

See Appendix C for more details.

2 has been known for a long time as a metric for verification of predicted probabilities (Brier, 1950) .

Similarly to the log-likelihood, the Brier score penalizes low probabilities assigned to correct predictions and high probabilities assigned to wrong ones.

It is also sensitive to the temperature of the softmax distribution and behaves similarly to the log-likelihood.

While these metrics are not strictly equivalent, they show a high empirical correlation for a wide range of models on CIFAR-10, CIFAR-100 and ImageNet datasets (see Appendix A).

Detection of wrong predictions of the model, or misclassifications, is a popular downstream problem aiding in assessing the quality of in-domain uncertainty.

Since misclassification detection is essentially a binary classification problem, some papers measure its quality using conventional metrics for binary classification such as AUC-ROC and AUC-PR (Malinin & Gales, 2018; Cui et al., 2019; Możejko et al., 2018) .

These papers use an uncertainty criterion like confidence or predictive entropy H[p(y | x i )] as a prediction score.

While these metrics can be used to assess the misclassification detection performance of a single model, they cannot be used to directly compare misclassification performance across different models.

Correct and incorrect predictions are specific for every model, therefore, every model induces Figure 2 : Thresholded adaptive calibration error (TACE) is highly sensitive to the threshold and the number of bins and does not provide a stable ranking for different methods.

TACE is reported for VGG16BN model on CIFAR-100 dataset and is evaluated at the optimal temperature.

its own binary classification problem.

The induced problems can differ significantly from each other since different models produce different confidences and misclassify different objects.

AUCs for misclassification detection can not be directly compared between different models.

While comparing AUCs is incorrect in this setting, it is correct to compare these metrics in many out-of-domain data detection problems.

In that case, both objects and targets of induced binary classification problems remain fixed for all models.

Note however that this condition still usually breaks down in the problem of detection of adversarial attacks since different models generally have different inputs after an adversarial attack.

Accuracy-confidence curves are another way to measure the performance of misclassification detection.

These curves measure the accuracy on the set of objects with confidence max cp (y = c | x i ) above a certain threshold τ (Lakshminarayanan et al., 2017) and ignoring or rejecting the others.

The main problem with accuracy-confidence curves is that they rely too much on calibration and the actual values of confidence.

Models with different temperatures have different numbers of objects at each confidence level which does not allow for a meaningful comparison.

To overcome this problem, one can switch from thresholding by the confidence level to thresholding by the number of rejected objects.

The corresponding curves are then less sensitive to temperature scaling and allow to compare the rejection ability in a more meaningful way.

Such curves have been known as accuracyrejection curves (Nadeem et al., 2009) .

In order to obtain a scalar metric for easy comparisons, one can compute the area under this curve, resulting in AU-ARC (Nadeem et al., 2009 ).

Informally speaking, a probabilistic classifier is calibrated if any predicted class probability is equal to the true class probability according to the underlying data distribution (see (Vaicenavicius et al., 2019) for formal definitions).

Any deviation from perfect calibration is called miscalibration.

For brevity, we will usep i,c to denotep(y = c | x i ) in the current section.

Expected Calibration Error (ECE) (Naeini et al., 2015) is a metric that estimates model miscalibration by binning the assigned probability scores and comparing them to average accuracies inside these bins.

Assuming B m denotes the m-th bin and M is overall number of bins, the ECE is defined as follows:

where acc(B) = |B| A recent line of works on measuring calibration in deep learning (Nixon et al., 2019; Vaicenavicius et al., 2019) outline several problems of the ECE score.

Firstly, ECE is a biased estimate of the true calibration.

Secondly, ECE-like scores cannot be optimized directly since they are minimized by a model with constant uniform predictions, making the infinite temperature T = +∞ its global optimum.

Thirdly, ECE only estimates miscalibration in terms of the maximum assigned probability whereas practical applications may require the full predicted probability vector to be calibrated.

Finally, biases of ECE on different models may not be equal, rendering the miscalibration estimates incompatible.

Thresholded Adaptive Calibration Error (TACE) was proposed as a step towards solving some of these problems (Nixon et al., 2019) .

TACE disregards all predicted probabilities that are less than a certain threshold (hence thresholded), chooses the bin locations adaptively so that each bin has the same number of objects (hence adaptive), and estimates miscalibration of probabilties across all classes in the prediction (not just the top-1 predicted class as in ECE).

Assuming that B TA m denotes the m-th thresholded adaptive bin and M is the overall number of bins, TACE is defined as follows:

where objs(B TA , c) =

Although TACE does solve several problems of ECE and is useful for measuring calibration of a specific model, it still cannot be used as a reliable criterion for comparing different models.

Theory suggests that it is still a biased estimate of true calibration with different bias for each model.

In practice, TACE is sensitive to its two parameters, the number of bins and the threshold, and does not provide a consistent ranking of different models which is shown in Figure 2 .

There are two common ways to perform temperature scaling using a validation set when training on datasets that only feature public training and test sets (e.g. CIFARs).

The public training set might be divided into a smaller training set and validation set, or the public test set can be split into test and validation parts (Guo et al., 2017; Nixon et al., 2019) .

The problem with the first method is that the resulting models cannot be directly compared with all the other models that have been trained on the full training set.

The second approach, however, provides an unbiased estimate of metrics such as log-likelihood and Brier score but introduces more variance.

In order to reduce the variance of the second approach, we perform a "test-time cross-validation".

We randomly divide the test set into two equal parts, then compute metrics for each half of the test set using the temperature optimized on another half.

We repeat this procedure five times and average the results across different random partitions to reduce the variance of the computed metrics.

In this paper we consider the following ensembling techniques: deep ensembles (Lakshminarayanan et al., 2017) , snapshot ensembles (SSE by (Huang et al., 2017) ), fast geometric ensembling (FGE by (Garipov et al., 2018) ), SWA-Gaussian (SWAG by (Maddox et al., 2019) ), cyclical SGLD (cSGLD by (Zhang et al., 2019) ), variational inference (VI by (Blundell et al., 2015) ), dropout (Srivastava et al., 2014) and test-time data augmentation (Krizhevsky et al., 2009 ).

These techniques were chosen to cover a diverse set of approaches keeping their predictive performance in mind.

All these techniques can be summarized as distributions q m (ω) over some parameters ω of computation graphs z ω (x), where m stands for the technique.

During testing, one can average the predictions across parameters ω ∼ q m (ω) to approximate the predictive distribution

For example, a deep ensemble of S networks can be represented in this form as a mixture of S Dirac's deltas q DE (ω) = 1 S S s=1 δ(ω − ω s ), centered at independently trained snapshots ω s .

Similarly, a Bayesian neural network with a fully-factorized Gaussian approximate posterior distribution over the weight matrices and convolutional kernels ω is represented as q VI (ω) = N (ω | µ, diag(σ 2 )), µ and σ 2 being the optimal variational means and variances respectively.

If one considers data augmentation as a part of the computational graph, it can be parameterized by the coordinates of the random crop and the flag for whether to flip the image horizontally or not.

Sampling from the corresponding q aug (ω) would generate different ways to augment the data.

However, as data augmentation is present by default during the training of all othe mentioned ensembling techniques, it is suitable to study it in combination with these methods and not as a separate ensembling technique.

We perform such an evaluation in Section 4.3.

Typically, the approximation (equation 3) requires K independent forward passes through a neural network, making the test-time budget directly comparable across all methods.

Most ensembling techniques under consideration are either bounded to a single mode, or provide positively correlated samples.

Deep ensembles, on the other hand, is a simple technique that provides independent samples from different modes of the loss landscape, which can intuitively result in a better ensemble.

Therefore deep ensembles can be considered as a strong baseline for performance of other ensembling techniques given a fixed test-time budget.

Instead of comparing the values of uncertainty estimation metrics directly, we ask the following question aiming to introduce perspective and interpretability in our comparison:

What number of independently trained networks combined yields the same performance as a particular ensembling method?

Following insights from the previous sections, we use the calibrated log-likelihood (CLL) as the main measure of uncertainty estimation performance of the ensemble.

We define the Deep Ensemble Equivalent (DEE) for an ensembling method m and its upper and lower bounds as follows: (Russakovsky et al., 2015) .

We use PyTorch (Paszke et al., 2017) for implementation of these models, building upon available public implementations.

Our implementation closely matches the quality of methods that has been reported in original works.

Technical details on training, hyperparameters and implementations can be found in Appendix D. We plan to make all computed metrics, source code and trained models publicly available.

As one can see on Figure 3 , ensembling methods clearly fall into three categories.

SSE and cSGLD outperform all other techniques except deep ensembles and enjoy a near-linear scaling of DEE with the number of samples.

The investigation of weight-space trajectories of cSGLD and SSE (Huang et al., 2017; Zhang et al., 2019) suggests that these methods can efficiently explore different modes of the loss landscape.

In terms of deep ensemble equivalent, these methods do not saturate unlike other methods that are bound to a single mode.

More verbose results are presented in Appendix E.

In our experiments SSE typically outperforms cSGLD.

This is mostly due to the fact that SSE has a much larger training budget.

The cycle lengths and learning rates of SSE and cSGLD are comparable, however, SSE collects one snapshot per cycle while cSGLD collects three snapshots.

This makes samples from SSE less correlated with each other while increasing the training budget.

Both SSE and cSGLD can be adjusted to obtain a different trade-off between the training budget and the DEE-to-samples ratio.

We reused the schedules provided in the original papers (Huang et al., 2017; Zhang et al., 2019) .

Being more "local" methods, FGE and SWAG perform worse than SSE and cSGLD, but still significantly outperform "single-snapshot" methods like dropout, K-FAC Laplace approximation and variational inference.

We hypothesize that by covering a single mode with a set of snapshots, FGE and SWAG provide a better fit for the local geometry than methods based on stochastic computation graphs.

This implies that the performance of FGE and SWAG should be achievable by methods that approximate the geometry of a single mode.

However, one might need more elaborate posterior approximations and better inference techniques in order to match the performance of FGE and SWAG by training a stochastic computation graph end-to-end (as opposed to SWAG that constructs a stochastic computation graph post-hoc).

Data augmentation is a time-honored technique that is widely used in deep learning, and is a crucial component for training modern DNNs.

Test-time data augmentation have been used for a long time to improve the performance of convolutional networks.

For example, multi-crop evaluation has long been a standard procedure for the ImageNet challenge (Simonyan & Zisserman, 2014; Szegedy et al., 2015; He et al., 2016) .

It, however, is not very popular in the literature on ensembling techniques in deep learning.

In this section, we study the effect of test-time data augmentation on the aforementioned ensembling techniques.

We report the results on combination of ensembles and test-time data augmentation for CIFAR-10 in Interestingly, test-time data augmentation on ImageNet improves accuracy but decreases the (uncalibrated) log-likelihood of the deep ensembles ( Figure 1 , Table REF) .

It breaks the nearly optimal temperature of deep ensembles and requires temperature scaling to show the actual performance of the method, as discussed in Section 3.1.

We show that test-time data augmentation with temperature scaling significantly improves predictive uncertainty of ensembling methods and should be considered as a baseline for them.

It is a striking example that highlights the importance of temperature scaling.

Our experiments demonstrate that ensembles may be severely miscalibrated by default while still providing superior predictive performance after calibration.

We have explored the field of in-domain uncertainty estimation and performed an extensive evaluation of modern ensembling techniques.

Our main findings can be summarized as follows:

• Temperature scaling is a must even for ensembles.

While ensembles generally have better calibration out-of-the-box, they are not calibrated perfectly and can benefit from the procedure.

Comparison of log-likelihoods of different ensembling methods without temperature scaling might not provide a fair ranking especially if some models happen to be miscalibrated.

• Many common metrics for measuring in-domain uncertainty are either unreliable (ECE and analogues) or cannot be used to compare different methods (AUC-ROC, AUC-PR for misclassification detection; accuracy-confidence curves).

In order to perform a fair comparison of different methods, one needs to be cautious of these pitfalls.

• Many popular ensembling techniques require dozens of samples for test-time averaging, yet are essentially equivalent to a handful of independently trained models.

Deep ensembles dominate other methods given a fixed test-time budget.

The results indicate in particular that exploration of different modes in the loss landscape is crucial for good predictive performance.

• Methods that are stuck in a single mode are unable to compete with methods that are designed to explore different modes of the loss landscape.

Would more elaborate posterior approximations and better inference techniques shorten this gap?

• Test-time data augmentation is a surprisingly strong baseline for in-domain uncertainty estimation and can significantly improve other methods without increasing training time or model size since data augmentation is usually already present during training.

Our takeaways are aligned with the take-home messages of (Ovadia et al., 2019 ) that relate to indomain uncertainty estimation.

We also observe a stable ordering of different methods in our experiments, and observe that deep ensembles with few members outperform methods based on stochastic computation graphs.

A large number of unreliable metrics inhibits a fair comparison of different methods.

Because of this, we urge the community to aim for more reliable benchmarks in the numerous setups of uncertainty estimation.

Implied probabilistic model Conventional neural networks for classification are usually trained using the average cross-entropy loss function with weight decay regularization hidden inside an optimizer in a deep learning framework like PyTorch.

The actual underlying optimization problem can be written as follows:

where

is the training dataset of N objects x i with corresponding labels y * i , λ is the weight decay scale andp(y * i = j | x i , w) denotes the probability that a neural network with parameters w assigns to class j when evaluated on object x i .

The cross-entropy loss defines a likelihood function p(y * | x, w) and weight decay regularization, or L 2 regularization, corresponds to a certain Gaussian prior distribution p(w).

The whole optimization objective then corresponds to maximum a posteriori inference in the following probabilistic model:

log p(y

As many of the considered methods are probabilistic in nature, we use the same probabilistic model for all of them.

We use the SoftMax-based likelihood for all models, and use the fully-factorized zero-mean Gaussian prior distribution with variances σ 2 = (N λ) −1 , where the number of objects N and the weight decay scale λ are dictated by the particular datasets and neural architectures, as defined in the following paragraph.

In order to make the result comparable across all ensembling techniques, we use the same prababilistic model for all methods, choosing fixed weight decay parameters for each architecture.

Conventional networks On CIFAR-10/100 datasets all networks were trained by SGD optimizer with batch size of 128, momentum 0.9 and model-specific parameters i.e., initial learning rate (lr init ), weight decay (wd), and number of optimization epoch (epoch).

The specific hyperparameters are shown in Table 2 .

The models used a unified learning rate scheduler that is shown in equation 10.

All models have been trained using data augmentation that consists of horizontal flips, random crop of size 32 with padding 4.

The standard data normalization has also been applied.

Weight decays, initial learning rates, and the learning rate scheduler were taken from (Garipov et al., 2018) paper.

Compared with hyperparameters of (Garipov et al., 2018) , the number of optimization epochs has been increased since we found that all models were underfitted.

While original WideResNet28x10 includes number of dropout layers with p = 0.3 and 200 training epoch, in this setting we find that WideResNet28x10 underfits, and requires a longer training.

Thus, we used p = 0, effectively it does not affect the final performance of the model in our experiments, but reduces training time.

On ImageNet dataset we used ResNet50 examples with a default hyperparameters from PyTorch examples 5 .

Specifically SGD optimizer with momentum 0.9, batch size of 256, initial learning rate 0.1, and with decay 1e-4.

The training also includes data augmentation random crop of size 224 × 224, horizontal flips, and normalization, and learning rate scheduler lr = lr init · 0.1 epoch//30 , where // denotes integer division.

We only deviated from standard parameters by increasing the number of training epochs from 90 to 130.

Or models achived top-1 error of 23.81 ± 0.15 that closely matches accuracy of the ResNet50 probided by PyTorch which is 23.85

6 .

Training of one model on a single NVIDIA Tesla V100 GPU takes approximately 5.5 days.

Deep Ensembles Deep ensembles (Lakshminarayanan et al., 2017) average the predictions across networks trained independently starting from different initializations.

To obtain Deep Ensemble we repeat the procedure of training standard networks 128 times for all architectures on CIFAR-10 and CIFAR-100 datasets (1024 networks over all) and 50 times for ImageNet dataset.

Every single member of Deep Ensembles were actually trained with exactly the same hyperparameters as conventional models of the same arhitecture.

Dropout The binary dropout (or MC dropout) (Srivastava et al., 2014; Gal & Ghahramani, 2016)

is one of the most known ensembling techniques.

It puts a multiplicative Bernoulli noise with parameter p over activations of ether fully-connected or convolutional layer, averaging predictions of the network w.r.t.

the noise during test.

The dropout layers have been applied to VGG, and WideResNet networks on CIFAR-10 and CIFAR-100 datasets.

For VGG the dropout has been applied to fully-connected (fc) layers with p = 0.5, overall two dropout layers, one before the first fc-layer and one before the second one.

While original version of VGG for CIFARs (Zagoruyko, 2015) exploits more dropout layers, we observed that any additional dropout layer deteriorates the performance on the model in ether deterministic or stochastic mode.

For WideResNet network we applied dropout consistently with the original paper (Zagoruyko & Komodakis, 2016) with p = 0.3.

The dropout usually increases the time to convergence, thus, VGG and WideResNet networks with dropout was trained for 400 epoch instead of 300 epoch for deterministic case.

The all other hyperparameters was the same as in case of conventional models.

Variational Inference The VI approximates a true posterior distribution p(w | Data) with a tractable variational approximation q θ (w), by maximizing so-called variational lower bound L (eq. 11) w.r.t.

parameters of variational approximation θ.

We used fully-factorized Gaussian approximation q(w), and Gaussian prior distribution p(w).

In the case of such a prior p(w) the probabilistic model remains consistent with conventional training which corresponds to MAP inference in the same probabilistic model.

We used variational inference for both convolutional and fully-connected layers, where variances of the weights was parameterized by log σ.

For fully-connected layers we applied the LRT (Kingma et al., 2015) .

While variational inference provide a theoretical grounded way to approximate a true posterior, on practice, it tends to underfit deep learning models (Kingma et al., 2015) .

The following tricks are applied to deal with it: pre-training (Molchanov et al., 2017) Consistently with the practical tricks we use a pre-training, specifically, we initialize µ with a snapshot of the weights of pretrained conventional model, and initialize log σ with model-specific constant log σ init .

The KL-divergence -except the term that corresponds to a weight decay -was scaled on model specific parameter β.

The weigh decay term was implemented as a part of the optimizer.

We used a fact that KL-divergence between two Gaussian distributions can be rewritten as two terms one of which is equal to wd regularization.

On CIFAR-10 and CIFAR-100 we used β 1e-4 for VGG, ResNet100 and ResNet164 networks, and β 1e-5 for WideResNet.

The initialization of log-variance log σ init was set to −5 for all models.

Parameters µ were optimized with conventional SGD (with the same parameters as conventional networks, except initial learning rate lr init that was set to 1e-3).

We used a separate Adam optimizer with constant learning rate 1e-3 to optimize log-variances of the weights log σ.

The training was held for 100 epochs, that corresponds to 400 epochs of training (including pre-training).

On ImageNet we used β = 1e-3, lr init = 0.01, log σ init = −6, and held training for a 45 epoch form a per-trained model.

The Laplace approximation uses the curvature information of the appropriately scaled loss function to construct a Gaussian approximation to the posterior distribution.

Ideally, one would use the Hessian of the loss function as the covariance matrix and use the maximum a posteriori estimate w M AP as the mean of the Gaussian approximation:

log p(w | x, y * ) = log p(y * | x, w) + log p(w) + const (13)

In order to keep the method scalable, we use the Fisher Information Matrix as an approximation to the true Hessian (Martens & Grosse, 2015) .

For K-FAC Laplace, we use the whole dataset to construct an approximation to the empirical Fisher Information Matrix, and use the π correction to reduce the bias (Ritter et al., 2018; Martens & Grosse, 2015) .

Following (Ritter et al., 2018) , we find the optimal noise scale for K-FAC Laplace on a held-out validation set by averaging across five random initializations.

We then reuse this scale for networks trained without a hold-out validation set.

We report the optimal values of scales in Table 3 .

Note that the optimal scale is different depending on whether we use test-time data augmentation or not.

Since the data augmentation also introduces some amount of additional noise, the optimal noise scale for K-FAC Laplace with data augmentation is lower.

Snapshot Ensembles Snapshot Ensembles (SSE) (Huang et al., 2017 ) is a simple example of an array of methods which collect samples from a training trajectory of a network in weight space to construct an ensemble.

Samples are collected in a cyclical manner: each cycle learning rate goes from a large value to near-zero and weights snapshot is taken at the end of the cycle.

SSE uses SGD with a cosine learning schedule defined as follows:

where α 0 is the initial learning rate, T is the total number of training iterations and M is the number of cycles.

On CIFAR-10/100 parameters from the original paper are reused, length of cycle is 40 epochs, maximum learning rate is 0.2, batch size is 64.

On ResNet50 on ImageNet we used hyperparameters from the original paper which are 45 epoch per cycle, maximum learning rate 0.1, and cosine scheduler of learning rate (eq. 16).

All other parameters are equal to the ones as were used conventional networks.

Cyclical SGLD Cyclical Stochastic Gradient Langevin Dynamics (cSGLD) (Zhang et al., 2019) is a state-of-the-art ensembling method for deep neural networks pertaining to stochastic Markov Chain Monte Carlo family of methods.

It bears similarity to SSE, e.g. it employs SGD with a learning rate schedule described with the equation 16 and training is cyclic in the same manner.

Its main differences from SSE are introducing gradient noise and capturing several snapshots per cycle, both of which aid in sampling from posterior distribution over neural network weights efficiently.

Some parameters from the original paper are reused: length of cycle is 50 epochs, maximum learning rate is 0.5, batch size is 64.

Number of epochs with gradient noise per cycle is 3 epochs.

This was found to yield much higher predictive performance and better uncertainty estimation compared to the original paper choice of 10 epochs for CIFAR-10 and 3 epochs for CIFAR-100.

Finally, cyclical Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) which reportedly has marginally better performance compared with cyclical SGLD (Zhang et al., 2019) could not be reproduced with a wide range of values of SGD momentum term.

Because of this, we only include cyclical SGLD in our benchmark.

FGE Fast Geometric Ensembling (FGE) is an ensembling method that is similar to SSE in that it collects samples from a training trajectory of a network in weight space to construct an ensemble.

Table 3 : Optimal noise scale for K-FAC Laplace for different datasets and architectures.

For ResNet50 on ImageNet, the optimal scale found was 2.0 with test-time augmentation and 6.8 without test-time augmentation.

Its main differences from SSE are pretraining, a short cycle length and a piecewise-linear learning rate schedule

Original hyperparameters are reused.

Model pretraining is done with SGD for 160 epochs according to the standard learning rate schedule described in equation 10 with maximum learning rates from Table 2 .

After that, a desired number of FGE cycles is done with one snapshot per cycle collected.

Learning rate in a cycle is changed with parameters α 1 = 1e − 2, α 2 = 5e − 4, cycle length of 2 epochs for VGG and α 1 = 5e − 2, α 2 = 5e − 4, cycle length of 4 epochs for other networks.

Batch size is 128.

SWAG SWA-Gaussian (SWAG) (Maddox et al., 2019 ) is an ensembling method based on fitting a Gaussian distribution to model weights on the SGD training trajectory and sampling from this distribution to construct an ensemble.

Like FGE, SWAG has a pretraining stage which is done according to the standard learning rate schedule described in equation 10 with maximum learning rates from Table 2.

After that, training continues with a constant learning rate of 1e-2 for all models except for PreResNet110 and PreResNet164 on CIFAR-100 where it continues with a constant learning rate of 5e-2 in accordance with the original paper.

Rank of the empirical covariance matrix which is used for estimation of Gaussian distribution parameters is set to be 20.

Area between DEE lower and DEE upper is shaded.

Lines 2-4 correspond to DEE based on other metrics, defined similarly to the log-likelihoodbased DEE.

Note that while the actual scale of DEE varies from metric to metric, the ordering of different methods and the overall behaviour of the lines remain the same.

SSE outperforms deep ensembles on CIFAR-10 on the WideResNet architecture.

It possibly indicates that the cosine learning rate schedule of SSE is more suitable for this architecture than the piecewise-linear learning rate schedule used in deep ensembles.

We will change the learning rate schedule on WideResNets to a more suitable option in further revisions of the paper.

ResNet110 0.0037±0.00 0.0032±0.00 0.0041±0.00 0.0051±0.00 0.0054±0.00 0.0035±0.00 0.0043±0.00 0.0049±0.00 ResNet164 0.0035±0.00 0.0031±0.00 0.0039±0.00 0.0049±0.00 0.0053±0.00 0.0034±0.00 0.0038±0.00 0.0049±0.00 VGG16 0.0051±0.00 0.0046±0.00 0.0053±0.00 0.0076±0.00 0.0109±0.00 0.0045±0.00 0.0054±0.00 0.0076±0.00 0.0116±0.00 WideResNet 0.0031±0.00 0.0029±0.00 0.0031±0.00 0.0040±0.00 0.0043±0.00 0.0031±0.00 0.0037±0.00 0.0039±0.00 ResNet110 0.0421±0.00 0.0382±0.00 0.0456±0.00 0.0607±0.00 0.0648±0.00 0.0426±0.00 0.0496±0.00 0.0593±0.00 ResNet164 0.0388±0.00 0.0359±0.00 0.0429±0.00 0.0556±0.00 0.0600±0.00 0.0402±0.00 0.0455±0.00 0.0534±0.00 VGG16 0.0528±0.00 0.0508±0.00 0.0570±0.00 0.0779±0.00 0.0855±0.00 0.0518±0.00 0.0572±0.00 0.0741±0.00 0.0857±0.00 WideResNet 0.0343±0.00 0.0324±0.00 0.0364±0.00 0.0498±0.00 0.0499±0.00 0.0366±0.00 0.0439±0.00 0.0476±0.00 Table 7 : Results before and after data augmentation on CIFAR10.

Error (%) on CIFAR100 dataset (100 samples) Table 9 : Results before and after data augmentation on ImageNet.

@highlight

We highlight the problems with common metrics of in-domain uncertainty and perform a broad study of modern ensembling techniques.