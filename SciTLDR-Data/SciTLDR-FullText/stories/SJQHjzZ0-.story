Generative adversarial networks (GANs) have been extremely effective in approximating complex distributions of high-dimensional, input data samples, and substantial progress has been made in understanding and improving GAN performance in terms of both theory and application.

However, we currently lack quantitative methods for model assessment.

Because of this, while many GAN variants being proposed, we have relatively little understanding of their relative abilities.

In this paper, we evaluate the performance of various types of GANs using divergence and distance functions typically used only for training.

We observe consistency across the various proposed metrics and, interestingly, the test-time metrics do not favour networks that use the same training-time criterion.

We also compare the proposed metrics to human perceptual scores.

Generative adversarial networks (GANs) aim to approximate a data distribution P , using a parameterized model distribution Q. They achieve this by jointly optimizing generative and discriminative networks BID9 .

GANs are end-to-end differentiable, and samples from the generative network are propagated forward to a discriminative network, and error signals are then propagated backwards from the discriminative network to the generative network.

The discriminative network is often viewed as learned, adaptive loss function for the generative network.

GANs have achieved state-of-the-art results for a number of applications BID8 , producing more realistic, sharper samples than other popular generative models, such as variational autoencoders BID22 .

Because of their success, many GAN frameworks have been proposed.

However, it has been difficult to compare these algorithms and understand their strengths and weaknesses because we are currently lacking in quantitative methods for assessing the learned generators.

In this work, we propose new metrics for measuring how realistic samples generated from GANs are.

These criteria are based on a formulation of divergence between the distributions P and Q BID33 BID38 : DISPLAYFORM0 Here, different choices of µ, υ, and F can correspond to different f -divergences BID33 or different integral probability metrics (IPMs) BID38 .

Importantly, J(Q) can be estimated using samples from P and Q, and does not require us to be able to estimate P (x) or Q(x) for samples x. Instead, evaluating J(Q) involves finding the function f ∈ F that is maximally different with respect to P and Q.This measure of divergence between the distributions P and Q is related to the GAN criterion if we restrict the function class F to be neural network functions parameterized by the vector φ and the class of approximating distributions to correspond to neural network generators G θ parameterized by the vector θ, allowing formulation as a min-max problem: H is a Reproducing Kernel Hilbert Space (RKHS) and · L is the Lipschitz constant.

For the LS-DCGAN, we used b = 1 and a = 0 BID28 .

DISPLAYFORM1 Metric µ υ Function Class GAN (GC) log f − log(1 − f ) X → R + , ∃M ∈ R : |f (x)| ≤ M Least-Squares GAN (LS) −(f − b) DISPLAYFORM2 In this formulation, Q θ corresponds to the generator network's distribution and D φ corresponds to the discriminator network (see BID33 for details).We propose using J(θ) to evaluate the performance of the generator network G θ for various choices of µ and υ, corresponding to different f -divergences or IPMs between distributions P and Q θ , that have been successfully used for GAN training.

Our proposed metrics differ from most existing metrics in that they are adaptive, and involve finding the maximum over discriminative networks.

We compare four metrics, those corresponding to the original GAN (GC) BID8 , the Least-Squares GAN (LS) BID28 ,the Wasserstein GAN (IW) , and the Maximum Mean Discrepency GAN (MMD) criteria.

Choices for µ, υ, and F for these metrics are shown in TAB0 .

Our method can easily be extended to other f -divergences or IPMs.

To compare these and previous metrics for evaluating GANs, we performed many experiments, training and comparing multiple types of GANs with multiple architectures on multiple data sets.

We qualitatively and quantitatively compared these metrics to human perception, and found that our proposed metrics better reflected human perception.

We also show that rankings produced using our proposed metrics are consistent across metrics, thus are robust to the exact choices of the functions µ and υ in Equation 2.We used the proposed metrics to quantitatively analyze three different families of GANs: Deep Convolutional Generative Adversarial Networks (DCGAN) BID34 , Least-Squares GANs (LS-DCGAN), and Wasserstein GANs (W-DCGAN), each of which corresponded to a different proposed metric.

Interestingly, we found that the different proposed metrics still agreed on the best GAN framework for each dataset.

Thus, even though, e.g. for MNIST the W-DCGAN was trained with the IW criterion, LS-DCGAN still outperformed it for the IW criterion.

Our analysis also included carrying out a sensitivity analysis with respect to various factors, such as the architecture size, noise dimension, update ratio between discriminator and generator, and number of data points.

Our empirical results show that: i) the larger the GAN architecture, the better the results; ii) having a generator network larger than the discriminator network does not yield good results; iii) the best ratio between discriminator and generator updates depend on the data set; and iv) the W-DCGAN and LS-DCGAN performance increases much faster than DCGAN as the number of training examples grows.

These metrics thus allow us to tune the hyper-parameters and architectures of GANs based on our proposed method.

GANs can be evaluated using manual annotations, but this is time consuming and difficult to reproduce.

Several automatically computable metrics have been proposed for evaluating the performance of probabilistic general models and GANs in particular.

We review some of these here, and compare our proposed metrics to these in our experiments.

Many previous probabilistic generative models were evaluated based on the pointwise likelihood of the test data, the criterion also used during training.

While GANs can be used to generate samples from the approximate distribution, its likelihood on test samples cannot be evaluated without simplifying assumptions.

As discussed in BID41 , likelihood often does not provide good rankings of how realistic samples look, the main goal of GANs.

We evaluted the efficacy of the log-likelihood of the test data, as estimated using Annealed Importance Sampling (AIS) BID43 .

AIS has been to estimate the likelihood of a test sample x by considering many intermediate distributions that are defined by taking a weighted geometric mean between the prior (input) distribution, p(z), and an approximation of the joint distribution p σ (x, z) = p σ (x|z)p(z).

Here, p σ (x|z) is a Gaussian kernel with fixed standard deviation σ around mean G θ (z).

The final estimate depends critically on the accuracy of this approximation.

In Section 4, we demonstrate that the AIS estimate of p(x) is highly dependent on the choice of this hyperparameter.

The Generative Adversarial Metric BID18 measures the relative performance of two GANs by measuring the likelihood ratio of the two models.

Consider two GANs with their respective trained partners, M 1 = (D 1 , G 1 ) and M 2 = (D 2 , G 2 ), where G 1 and G 2 are the generators and D 1 and D 2 are the discriminators.

The hypothesis H 1 is that M 1 is better than M 2 if G 1 fools D 2 more than G 2 fools D 1 , and vice versa for the hypothesis H 0 .

The likelihood-ratio is defined as: DISPLAYFORM0 where M 1 and M 2 are the swapped pairs (D 1 , G 2 ) and (D 2 , G 1 ), and p(x|y = 1, M ) is the likelihood of x generated from the data distribution p(x) by model M and p(y = 1|x; D) indicates that discriminator D thinks x is a real sample.

To evaluate this, we measure the ratio of how frequently G 1 , the generator from model 1, fools D 2 , the discriminator from model 2, and vice-versa: DISPLAYFORM1 , where x 1 ∼ G 1 and x 2 ∼ G 2 .

There are two main caveats to the Generative Adversarial Metric.

First, the measurement only provides comparisons between pairs of models.

Second, the metric has a constraint where the two discriminators must have an approximately similar performance on a calibration dataset, which can be difficult to satisfy in practice.

The Inception Score BID36 (IS) measures the performance of a model using a third-party neural network trained on a supervised classification task, e.g. Imagenet.

The IS computes the expectation of divergence between the distribution of class predictions for samples from the GAN compared to the distribution of class to the distribution of class labels used to train the third-party network, DISPLAYFORM2 Here, the class prediction given a sample x is computed using the third-party neural network.

In BID36 ), Google's Inception Network BID40 trained on Imagenet was the third-party neural network.

IS is the most widely used metric to measure GAN performance.

However, summarizing samples as the class prediction from a network trained for a different task discards much of the important information in the sample.

In addition, it requires another neural network that is trained separately via supervised learning.

We demonstrate an example of a failure case of IS in the Experiments section.

The Fréchet Inception Distance (FID) BID14 extends upon IS.

Instead of using the final classification outputs from the third-party network as representations of samples, it uses a representation computed from a late layer of the third-party network.

It compares the mean m Q and covariance C Q of the Inception-based representation of samples generated by the GAN to the mean m P and covariance C P of the same representation for training samples: DISPLAYFORM3 This method relies on the Inception-based representation of the samples capturing all important information and the first two moments of the distributions being descriptive of the distribution.

Classifier Two-Sample Tests (C2ST) BID27 proposes training a classifier, similar to a discriminator, that can distinguish real samples from P from generated samples from Q, and using the error rate of this classifier as a measure of GAN performance.

In their work, they used single-layer and k-nearest neighbor (KNN) classifiers trained on a representation of the samples computed from a late layer of a third-party network (in this case, ResNet BID13 ).

C2ST is an IPM BID38 , like the MMD and Wasserstein metrics we propose, with µ(f ) = f and υ(f ) = f , but with a different function class F, corresponding to the family of classifiers chosen (in this case, single-layer networks or KNN, see see our detailed explanation in Appendix 5).

The accuracy of a classifier trained to distinguish samples from distributions P and Q is just one way to measure the distance between these distributions, and, in this work, we propose a general family.

Given a generator G θ with parameters θ which generates samples from the distribution Q θ , we propose to measure the quality of G θ by estimating divergence between the true data distribution P and Q θ for different choices of divergence measure.

We train both G θ and D ϕ on a training data set, and measure performance on a separate test set.

See Algorithm 1 for details.

We consider metrics from two widely studied divergence and distance measures, f -divergence BID32 and the Integral Probability Metric (IPM) BID31 .

In our experiments, we consider the following four metrics that are commonly used to train GANs.

Below, ϕ represents the parameters of the discriminator network and θ represents the parameters of the generator network.

Training a standard GAN corresponds to minimizing the following BID9 : DISPLAYFORM0 where p(z) is the prior distribution of the generative network and G θ (z) is a differentiable function from z to the data space represented by a neural network with parameter θ.

D ϕ is trained with a sigmoid activation function, thus its output is guaranteed to be positive.

A Least-Squares GAN corresponds to training with a Pearson χ 2 divergence BID28 : DISPLAYFORM0 Following BID28 , we set a = 0 and b = 1 when training D ϕ .

The maximum mean discrepancy metric considers the largest difference in the expectations over a unit ball of RKHS H, DISPLAYFORM0 where H is the RKHS with kernel k(·, ·) BID11 .

In this case, we do not need to train a discriminator D ϕ to evaluate our metric.

Improved Wasserstein Distance (IW) proposed the use of the dual representation of the Wasserstein distance BID42 ) for training GANs.

The Wasserstein distance is an IPM which considers the 1-Lipschitz function class ϕ : DISPLAYFORM1 Note that IW and MMD BID39 were recently proposed to evaluate GANs, but have not been compared before.

DISPLAYFORM2 Initialize critic network parameter ϕ.3: DISPLAYFORM3 Sample data points from X, {x m } ∼ X tr .

Sample points from generative model, {s m } ∼ G θ .

ϕ ← ϕ + η∇ ϕ J({x m }, {s m }; ϕ).

Sample points from generative model, {s m } ∼ G θ .

return J(ϕ, X te , {s m }).

The goals in our experiments are two-fold.

First, we wanted to evaluate the metrics we proposed for evaluating GANs.

Second, we wanted to use these metrics to evaluate GAN frameworks and architectures.

In particular, we evaluated how size of the discriminator and generator networks affected performance, and the sensitivity of each algorithm to training data set size.

GAN frameworks.

We conducted our experiments on three types of GANs: Deep Convolutional Generative Adversarial Networks (DCGAN), Least-Squares GANs (LS-DCGAN), and Wasserstein GANs (W-DCGAN).

Note that to not confuse the test metric names with the GAN frameworks we evaluated, we use different abbreviations.

GC is the original GAN criterion, which is used to train DCGANs.

The LS criterion is used to train the LS-DCGAN, and the IW is used to train the W-DCGAN.Evaluation criteria.

We evaluated these three families of GANs with six metrics.

We compared our four proposed metrics to the two most commonly used metrics for evaluating GANs, the IS and FID.Because the optimization of a discriminator is required both during training and test time, we will call the discriminator learned for evaluaton of our metrics the critic, in order to not confuse the two discriminators.

We also compared these metrics to human perception, and had three volunteers evaluate and compare sets of images, either from the training data set or generated from different GAN frameworks during training.

Data sets.

In our experiments, we considered the MNIST (LeCun et al., 1998), CIFAR10, LSUN Bedroom, and Fashion MNIST datasets.

MNIST consists of 60,000 training and 10,000 test images with a size of 28 × 28 pixels, containing handwritten digits from the classes 0 to 9.

From the 60,000 training examples, we set aside 10,000 as validation examples to tune various hyper-parameters.

Similarly, FashionMNIST consists exactly the same number of training and test examples.

Each example is a 28x28 grayscale image, associated with a label from 10 classes.

The CIFAR10 dataset 1 consists of images with a size of 32 × 32 × 3 pixels, with ten different classes of objects.

We used 45,000, 5,000, and 10,000 examples as training, validation, and test data, respectively.

The LSUN Bedroom dataset consists of images with a size of 64×64 pixels, depicting various bedrooms.

From the 3,033,342 images, we used 90,000 images as training data and 90,000 images as validation data.

The learning rate was selected from discrete ranges and chosen based on a held-out validation set.

Hyperparameters.

TAB0 in the Appendix shows the learning rates and the convolutional kernel sizes that were used for each experiment.

The architecture of each network is presented in the Appendix in Figure 10 .

Additionally, we used exponential-mean-square kernels with several different sigma values for MMD.

A pre-trained logistic regression and pre-trained residual network were used for IS and FID on the MNIST and CIFAR10 datasets, respectively.

For every experiment, we retrained 10 times with different random seeds, and report the mean and standard deviation.

The log-likelihood measurement is the most commonly used metric for generative models.

We measured the log-likelihood using AIS 2 on GANs is strange, as shown in Figure 1 .

We measured the log-likelihood of the DCGAN on MNIST with three different variances, σ 2 = 0.01, 0.025, and 0.05.

The figure illustrates that the log-likelihood curve over the training epochs varies substantially depending on the variance, which indicates that the fixed Gaussian observable model might not be the ideal assumption for GANs.

Moreover, we observe a high log-likelihood at the beginning of training, followed by a drop in likelihood, which then returns to the high value.

The IS and MMD metrics do not require training a critic.

It was easy to find samples for which IS and MMD scores did not match their visual quality.

For example, Figure 2 shows samples generated by a DCGAN when it failed to train properly.

Even though the failed DCGAN samples are much darker than the samples on the right, the IS for the left samples is higher/better than for the right samples.

As the Imagenet-trained network is likely trained to be somewhat invariant to overall intensity, this issue is to be expected.

A failure case for MMD is shown in FIG2 .

The samples on the right are dark, like the previous examples, but still textually recognizable, whereas the samples on the left are totally meaningless.

However, MMD gives lower/worse distances to the left samples.

The average intensity of the pixels of the left samples are closer to that for the training data, suggesting that MMD is overly sensitive to image intensity.

Thus, IS is under-sensitive to image intensity, while MMD if oversensitive to it.

In Section 4.2.1, we conduct more systematic experiments by measuring the correlation between these metrics to human perceptual scores.

To both compare the metrics as well as different GAN frameworks, we evaluated the six metrics on different GAN frameworks.

TAB1 , and 4 present the results on MNIST, CIFAR10, and LSUN respectively.

As each type of GAN was trained using one of our proposed metrics, we investigated whether the metric favors samples from the model trained using the same metric.

Interestingly, we do not see this behavior, and our proposed metrics agree on which GAN framework produces samples closest to the test data set.

Every metric, except for MMD, showed that LS-DCGAN performed best for MNIST and CIFAR10, while W-DCGAN performed best for LSUN.

As discussed below, we found DCGAN to be unstable to train, and thus excluded GC as a metric for experiments except for this first data set.

For Fashion-MNIST, FID's ranking disagreed with IW and LS.We observed similar results for a range of different critic CNN architectures (number of feature maps in each convolutional layer): [3, 64, 128, 256] , [3, 128, 256, 512] , [3, 256, 512, 1024] , and [3, 320, 640, 1280] (see Supp.

FIG0 ).We evaluated a larger variety of GAN frameworks using pre-trained GANs downloaded from (pyt).

In particular, we evaluated on EBGAN(Junbo Zhao, 2016), BEGAN BID4 , W-DCGAN GP BID12 , and DRAGAN BID23 .

TAB4 presents the evaluation results.

Critic architectures were selected to match those of these pre-trained GANs.

For both MNIST and FashionMNIST, the three metrics are consistent and they rank DRAGAN the highest, followed by LS-DCGAN and DCGAN.The standard deviations for the IW distance are higher than for LS divergence.

We computed the Wilcoxon rank sum in order to test that whether medians of the distributions of distances are the same for DCGAN, LS-DCGAN, and W-DCGAN.

We found that the different GAN frameworks have significantly different performance according to the LS-GAN criterion, but not according to the IW criterion (p < .05, Wilcoxon rank-sum test).

Thus LS is more sensitive than IW.We evaluated the consistency of the metrics with respect to the size of validation set.

We trained our three GAN frameworks for 100 epochs with training 90,000 examples from the LSUN Bedroom dataset.

We then trained LS and IW critics using both 300 and 90,000 validation examples.

We looked at how often the critic trained with 300 examples agreed with that trained with 90,000 examples.

The LS critics agreed 88% of the time, while the IW critics agreed only 55% of the time (slightly better than chance).

Thus, LS is more robust to validation data set size.

Another advantage is that measuring the LS distance is faster than measuring the IW distance, as estimating IW involves regularizing with a gradient penalty time BID12 .

Computing the gradient penalty term and tuning its regularization coefficient requires extra computational time.

As mentioned above, we found training a critic using the GC criterion (corresponding to a DCGAN) to be unstable.

It has previously been speculated that this is the case because the support of the data and model distributions possibly becoming disjoint , and the Hessian of the GAN objective being non-Hermitian BID29 .

LS-DCGAN and W-DCGAN are proposed to address this by providing non-saturating gradients.

We also found DCGAN to be difficult to train, and thus only report results using the corresponding criterion GC for MNIST.

Note that this is different than training a discriminator as part of standard GAN training because we are training from a random initialization, not from the previous version of the discriminator.

Our experience was that the LS-DCGAN was the simplest and most stable model to train.

We visualized the 2D subspace of the loss surface of the GANs in Supp.

Fig. 29 .

Here, we took the parameters of three trained models (corresponds to red vertices in the figure) and applied barycentric interpolation with respect to three parameters (see details from BID20 ).

DCGAN surfaces have much sharper slopes when compared to the LS-DCGAN and W-DCGAN, and LS-DCGAN has the most gentle surfaces.

In what follows, we show that this geometric view is consistent with our finding that LS-DCGAN is the easiest and the most stable to train.

We compared the LS, IW, MMD, and IS metrics to human perception for the CIFAR10 dataset.

To accomplish this, we asked five volunteers to choose which of two sets of 100 samples, each generated using a different generator, looked most realistic.

Before surveying, the volunteers were trained to choose between real samples from CIFAR10 and samples generated by a GAN.

Supp.

FIG1 displays the user interface for the participants, and Supp.

FIG2 shows the fraction of labels that the volunteers agreed upon.

TAB5 ) presents the fraction of pairs for which each metric agrees with humans (higher is better).

IW has a slight edge over LS, and both outperform IS and MMD.

In Figure 3 , we show examples in which all humans agree and metrics disagrees with human perception.

All such examples are shown in Supp.

Fig. 21 Figure 3 : Pairs of generated image sets for which human perception and metrics disagree.

Here, we selected one such example for each metric for which the difference in that metric's scores was high.

For each pair, humans perceived the set of images on the left to be more realistic than those on the right, while the metric predicted the opposite.

Below each pair of images, we indicate the metric's score for the left and right image sets.

Several works have demonstrated an improvement in performance by enlarging deep network architectures BID24 BID37 BID13 BID16 .

Here, we investigate performance changes with respect to the width and depth of the networks.

First, we trained three GANs with varying numbers of feature map sizes, as shown in TAB8 (a-d).Note that we double the number of feature maps in TAB8 for both the discriminators and generators.

(a) Samples from (e) in TAB8 , MMD= 0.03, IS= 5.11(b) Samples from (f) in TAB8 , MMD= 0.49, IS= 6.15 In FIG1 , the performance of the LS score increases logarithmically as the number of feature maps is doubled.

A similar behaviour is observed in other metrics as well (see S.M. FIG4 ).

We then analyzed the importance of size in the discriminative and generative networks.

We considered two extreme feature map sizes, where we choose a small and large number of feature maps for the generator and discriminator, and vice versa (see label (e) and (f) in TAB8 , and results are shown in TAB7 .

For LS-DCGAN, it can be seen that a large number of feature maps for the discriminator has a better score than a large number of feature maps for the generator.

This can also be qualitatively verified by looking at the samples from architectures (a), (e), (f), and (d) in FIG4 .

For W-DCGAN, we observe the agreement between the LS and IW metric and conflict with MMD and IS.

When we look at the samples from the W-DCGAN in FIG2 , it is clear that the model with a larger number of feature maps in the discriminator should get a better score; this is another example of false intuition propagated by MMD and IS.

One interesting observation is that when we compare the score and samples from architecture (a) and (e) from TAB8 , architecture (a) is much better than (e) (see FIG4 ).

This demonstrates that having a large generator and small discriminator is worse than having a small architecture for both networks.

Overall, we found that having a larger generator than discriminator does not give good results, and that it is more desirable to have a larger discriminator than generator.

Similar results were also observed for MNIST, as shown in S.M. Figure 20 .

This result somewhat supports the theoretical result from BID2 , where the generator capacity needs to be modulated in order for approximately pure equilibrium to exist for GANs.

Lastly, we experimented with how performance changes with respect to the dimension of the noise vectors.

The source of the sample starts by transforming a noise vector into a meaningful image.

It is unclear how the size of noise affects the ability of the generator to generate a meaningful image.

Che et al. FORMULA0 TAB8 )."Small" and "large" number of filters for discriminator and generator respectively (ref.(e) in TAB8 )."Large" and "small" number of filters for discriminator and generator respectively (ref.(f) in TAB8 )."Large" number of filters for both discriminator and generator (ref.

(d) in TAB8 ). for DCGAN.

Our experiments show that this depends on the model.

Given a fixed size architecture (d) from TAB8 , we observed the performance of LS-DCGAN and W-DCGAN by varying the size of noise vector z. TAB9 illustrates that LS-DCGAN gives the best score with a noise dimension of 50 and W-DCGAN gives best score with a noise dimension of 150 for both IW and LS.

The outcome of LS-DCGAN is consistent with the result in BID5 .

It is possible that this occurs because both models fall into the category of f -divergences, whereas the W-DCGAN behaves differently because its metric falls under a different category, the Integral Probability Metric.

In practice, we alternate between updating the discriminator and generator, and yet this is not guaranteed to give the same result as the solution to the min-max problem in Equation 2.

Hence, the update ratio can influence the performance of GANs.

We experimented with three different update ratios, 5 : 1, 1 : 1, and 1 : 5, with respect to the discriminator and generator update.

We applied these ratios to both the MNIST and CIFAR10 datasets on all models.

FIG5 presents the LS scores on both MNIST and CIFAR10 and this result is consistent with the IW metric as well (see S.M. FIG2 ).

However, we did not find that any one update ratio was superior over others between the two datasets.

For CIFAR10, the 1 : 1 update ratio worked best for all models, and for MNIST, different ratios worked better for different models.

Hence, we conclude that number of update ratios for each model needs to be dynamically tuned.

The corresponding samples from the models trained by different update ratios are shown in S.M. Figure In practice, DCGANs are known to be unstable, and the generator tends to suffer as the discriminator gets better due to disjoint support between the data and generator distributions BID9 Here, we explore the sensitivity of three different kinds of GANs with respect to the number of training examples.

We have trained GANs with 10,000, 20,000, 30,000, 40,000, and 45,000 examples on CIFAR10.

FIG6 shows that the LS score curve of DCGAN grows quite slowly when compared to W-DCGAN and LS-DCGAN.

The three GANs have a relatively similar loss when they are trained with 10,000 training examples.

However, the DCGAN only gained 0.0124 ± 0.00127 by increasing from 10,000 to 40,000 training examples, whereas the performance of W-DCGAN and LS-DCGAN improved by 0.03016 ± 0.00469 and 0.0444 ± 0.0033, respectively.

Thus, we empirically observe that W-DCGAN and LS-DCGAN have faster performance increases than a DCGAN as the number of training examples grows.

In this paper, we proposed to use four well-known distance functions as an evaluation metrics, and empirically investigated the DCGAN, W-DCGAN, and LS-DCGAN families under these metrics.

Previously, these models were compared based on visual assessment of sample quality and difficulty of training.

In our experiments, we showed that there are performance differences in terms of average experiments, but that some are not statistically significant.

Moreover, we thoroughly analyzed the performance of GANs under different hyper-parameter settings.

There are still several types of GANs that need to be evaluated, such as GRAN BID18 , IW-DCGAN BID12 , BEGAN BID4 , MMDGAN , and CramerGAN (Bellemare et al., 2017) .

We hope to evaluate all of these models under this framework and thoroughly analyze them in the future.

Moreover, there has been an investigation into taking ensemble approaches to GANs, such as Generative Adversarial Parallelization BID19 .

Ensemble approaches have been empirically shown to work well in many domains of research, so it would be interesting to find out whether ensembles can also help in min-max problems.

Alternatively, we can also try to evaluate other log-likelihood-based models like NVIL BID30 , VAE BID22 , DVAE BID17 , DRAW BID10 , RBMs BID15 BID35 , NICE Dinh et al. (2014) , etc.

Model evaluation is an important and complex topic.

Model selection, model design, and even research direction can change depending on the evaluation metric.

Thus, we need to continuously explore different metrics and rigorously evaluate new models.

In this paper, we considered four distance metrics that belong to two class of metrics, φ-divergence and IPMs.

BID38 have shown that the optimal risk function is associated with a binary classifier with P and Q distributions conditioned on a class when the discriminant function is restricted to certain F (Theorem 17 from BID38 ).Let the optimal risk function be: DISPLAYFORM0 where F is the set of discriminant functions (classifier), y ∈ −1, 1, and L is the loss function.

By following derivation, we can see that the optimal risk function becomes IPM: DISPLAYFORM1 where DISPLAYFORM2 The second equality is derived by separating the loss for class 1 and class 0.

The third equality is from the way how we chose L(1,f(x)) and L(0,f(x)).

The last equality is derived from that fact that F is symmetric around zero (f ∈ F => −f ∈ F ).

Hence, this shows that with appropriately choosing L, MMD and Wasserstein distance can be understood as the optimal L-risk associated with binary classifier with specific set of F functions.

For example, Wasserstein distance and MMD distances are equivalent to the optimal risk function with 1-Lipschitz classifiers and a RKHS classifier with an unit length.

We trained two critics on training data and validation data, respectively, and evaluated on test data from both critics.

We trained six GANs (GAN, LS-DCGAN, W-DCGAN GP, DRAGAN, BEGAN, EBGAN) on MNIST and FashionMNIST.

We trained these GANs with 50,000 training examples.

At test time, we used 10,000 training and 10,000 validation examples for training the critics, and evaluated on 10,000 test examples.

Here, we present the test scores from the critics trained on training and validation data.

The results are shown in Table ? ?.

Note that we also have the IW and FID evaluation on these models in the paper.

For FashionMNIST, we find that test scores with a critic trained on training and validation data are very close.

Hence, we do not see any indication of overfitting.

On the other hand, there are gaps between the scores for the MNIST dataset and the test scores from critics trained on the validation set.

which gives better performance than the ones that are trained on the training set.

FIG1 : The participants are trained by selecting between random samples generated by GANs versus samples from data distribution.

They get a positive reward if they selected the data samples and a negative reward if they select the samples from the model.

After enough training, they choose the better group of samples among two randomly select set of samples.

(a) Samples from (e) in TAB8 , MMD= 0.03, IS= 5.11(b) Samples from (f) in TAB8 , , MMD= 0.49, IS= 6.15 TAB8 ).(b)

"Small" and "large" number of filters for discriminator and generator respectively (ref.(e) in TAB8 ).(c) "Large" and "small" number of filters for discriminator and generator respectively (ref.(f) in TAB8 ).(d) "Large" number of filters for both discriminator and generator (ref.

(d) in TAB8 ).

TAB8 ).(b)

"Small" and "large" number of filters for discriminator and generator respectively (ref.(e) in TAB8 ).(c) "Large" and "small" number of filters for discriminator and generator respectively (ref.(f) in TAB8 ).(d) "Large" number of filters for both discriminator and generator (ref.

(d) in TAB8 ).

Figure 31: The training curve of critics to show that the training curve converges.

IW distance curves in (a) increase because we used linear output unit for the critic network (by design choice).

This can be simply bounded by adding a sigmoid at the output of the critic network.

@highlight

An empirical evaluation on generative adversarial networks