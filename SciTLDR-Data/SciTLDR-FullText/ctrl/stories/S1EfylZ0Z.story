Many anomaly detection methods exist that perform well on low-dimensional problems however there is a notable lack of effective methods for high-dimensional spaces, such as images.

Inspired by recent successes in deep learning we propose a novel approach to anomaly detection using generative adversarial networks.

Given a sample under consideration, our method is based on searching for a good representation of that sample in the latent space of the generator; if such a representation is not found, the sample is deemed anomalous.

We achieve state-of-the-art performance on standard image benchmark datasets and visual inspection of the most anomalous samples reveals that our method does indeed return anomalies.

Given a collection of data it is often desirable to automatically determine which instances of it are unusual.

Commonly referred to as anomaly detection, this is a fundamental machine learning task with numerous applications in fields such as astronomy BID40 BID10 , medicine BID4 BID47 BID42 , fault detection BID16 , and intrusion detection BID14 BID17 .

Traditional algorithms often focus on the low-dimensional regime and face difficulties when applied to high-dimensional data such as images or speech.

Second to that, they require the manual engineering of features.

Deep learning omits manual feature engineering and has become the de-facto approach for tackling many high-dimensional machine learning tasks.

The latter is largely a testament of its experimental performance: deep learning has helped to achieve impressive results in image classification BID22 , and is setting new standards in domains such as natural language processing BID23 BID46 and speech recognition BID2 .In this paper we present a novel deep learning based approach to anomaly detection which uses generative adversarial networks (GANs) BID15 .

GANs have achieved state-ofthe-art performance in high-dimensional generative modeling.

In a GAN, two neural networksthe discriminator and the generator -are pitted against each other.

In the process the generator learns to map random samples from a low-dimensional to a high-dimensional space, mimicking the target dataset.

If the generator has successfully learned a good approximation of the training data's distribution it is reasonable to assume that, for a sample drawn from the data distribution, there exists some point in the GAN's latent space which, after passing it through the generator network, should closely resembles this sample.

We use this correspondence to perform anomaly detection with GANs (ADGAN).In Section 2 we give an overview of previous work on anomaly detection and discuss the modeling assumptions of this paper.

Section 3 contains a description of our proposed algorithm.

In our experiments, see Section 4, we both validate our method against traditional methods and showcase ADGAN's ability to detect anomalies in high-dimensional data.

Here we briefly review previous work on anomaly detection, touch on generative models, and highlight the methodology of GANs.

Anomaly detection.

Research on anomaly detection has a long history with early work going back as far as Edgeworth FORMULA1 , and is concerned with finding unusual or anomalous samples in a corpus of data.

An extensive overview over traditional anomaly detection methods as well as open challenges can be found in BID5 .

For a recent empirical comparison of various existing approaches, see BID12 .Generative models yield a whole family of anomaly detectors through estimation of the data distribution p.

Given data, we estimatep ≈ p and declare those samples which are unlikely underp to be anomalous.

This guideline is roughly followed by traditional non-parametric methods such as kernel density estimation (KDE) BID37 , which were applied to intrusion detection in BID49 .

Other research targeted mixtures of Gaussians for active learning of anomalies BID39 , hidden Markov models for registering network attacks BID36 , and dynamic Bayesian networks for traffic incident detection BID44 .Deep generative models.

Recently, variational autoencoders (VAEs) BID20 have been proposed as a deep generative model.

By optimizing over a variational lower bound on the likelihood of the data, the parameters of a neural network are tuned in such a way that samples resembling the data may be generated from a Gaussian prior.

Another generative approach is to train a pair of deep convolutional neural networks in an autoencoder setup (DCAE) BID30 and producing samples by decoding random points on the compression manifold.

Unfortunately, none of these approaches yield a tractable way of estimating p. Our approach uses a deep generative model in the context of anomaly detection.

Deep learning for anomaly detection.

Non-parametric anomaly detection methods suffer from the curse of dimensionality and are thus inadequate tools for the interpretation and analysis of highdimensional data.

Deep neural networks have been found to obviate many problems that arise in this context.

As a hybrid between the two approaches, deep belief networks were coupled with one-class support vector machines to detect anomalies in BID13 .

We found that this technique did not work well for image datasets, and indeed the authors included no such experiments in their paper.

Similarly, one may employ a network that was pretrained on a different task (such as classification on ImageNet) and then use this network's intermediate features to extract relevant information from images.

We tested this an approach in our experimental section.

Recently GANs, which we discuss in greater depth in the next section, have garnered much attention with performance surpassing previous deep generative methods.

Concurrently to this work, BID42 developed an anomaly detection framework that uses GANs in a similar way as we do.

We discuss the differences between our work and theirs in Section 3.2.

GANs, which lie at the heart of ADGAN, have set a new state-of-the-art in generative image modeling.

They provide a framework to generate samples that are approximately distributed to p, the distribution of the training data {x i } n i=1X ⊆ R d .

To achieve this, GANs attempt to learn the parametrization of a neural network, the so-called generator g θ , that maps low-dimensional samples drawn from some simple noise prior p z (e.g. a multivariate Gaussian) to samples in the image space, thereby inducing a distribution q θ (the push-forward of p z with respect to g θ ) that approximates p. To achieve this a second neural network, the discriminator d ω , learns to classify the data from p and q θ .

Through an alternating training procedure the discriminator becomes better at separating DISPLAYFORM0 Figure 1: An illustration of ADGAN.

In this example, ones from MNIST are considered normal.

After an initial draw from p z , the loss between the first generation g θ0 (z 0 ) and the image x whose anomaly we are assessing is computed.

This information is used to generate a consecutive image g θ1 (z 1 ) more alike x. After k steps, samples are scored.

If x is similar to the training data (blue example), then a similar object should be contained in the image of g θ k .

For a dissimilar x (red example), no similar image is found, resulting in a large loss.samples from p and samples from q θ , while the generator adjusts θ to fool the discriminator, thereby approximating p more closely.

The objective function of the GAN framework is thus: DISPLAYFORM1 where z are vectors that reside in a latent space of dimensionality d d.1 A recent work showed that this minmax optimization (1) equates to an empirical lower bound of an f -divergence BID34 .2 GAN training is difficult in practice, which has been shown to be a consequence of vanishing gradients in high-dimensional spaces .

These instabilities can be countered by training on integral probability metrics (IPMs) BID32 BID45 , one instance of which is the 1-Wasserstein distance.3 This distance, informally defined, is the amount of work to pull one density onto another, and is the basis of the Wasserstein GAN (WGAN) .

The objective function for WGANs is DISPLAYFORM2 where the parametrization of the discriminator is restricted to allow only 1-Lipschitz functions, i.e. DISPLAYFORM3 .

When compared to classic GANs, we have observed that WGAN training is extremely stable and is thus used in our experiments, see Section 4.

Our proposed method (ADGAN, see Alg.

1) sets in after GAN training has converged.

If the generator has indeed captured the distribution of the training data then, given a new sample x ∼ p, there should exist a point z in the latent space, such that g θ (z) ≈ x. Additionally we expect points away from the support of p to have no representation in the latent space, or at least occupy a small portion of the probability mass in the latent distribution, since they are easily discerned by d ω as not coming from p. Thus, given a test sample x, if there exists no z such that g θ (z) ≈ x, or if such a z is difficult to find, then it can be inferred that x is not distributed according to p, i.e. it is anomalous.

Our algorithm hinges on this hypothesis, which we illustrate in Fig. 1 .Algorithm 1: Anomaly Detection using Generative Adversarial Networks (ADGAN).

Input: parameters (γ, γ θ , n seed , k), sample x, GAN generator g θ , prior p z , reconstruction loss .

DISPLAYFORM0 θ.2 for j = 1, . . .

, n seed do DISPLAYFORM1

To find z, we initialize from z 0 ∼ p z , where p z is the same noise prior also used during GAN training.

For l = 1, . . .

, k steps, we backpropagate the reconstruction loss between g θ (z l ) and x, making the subsequent generation g θ (z l+1 ) more like x. At each iteration, we also allow a small amount of flexibility to the parametrization of the generator, resulting in a series of mappings from the latent space g θ0 (z 0 ), . . .

, g θ k (z k ) that more and more closely resembles x. Adjusting θ gives the generator additional representative capacity, which we found to improve the algorithm's performance.

Note that these adjustments to θ are not part of the GAN training procedure and θ is reset back to its original trained value for each new testing point.

To limit the risk of seeding in unsuitable regions and address the non-convex nature of the underlying optimization problem, the search is initialized from n seed individual points.

The key idea underlying ADGAN is that if the generator was trained on the same distribution x was drawn from, then the average over the final set of reconstruction losses { (x, g θ j,k (z j,k ))} nseed j=1 will assume low values, and high values otherwise.

Our method may also be understood from the standpoint of approximate inversion of the generator.

In this sense, the above backpropagation finds latent vectors z that lie close to g −1 θ (x).

Inversion of the generator was previously studied in BID6 , where it was verified experimentally that this task can be carried out with high fidelity.

In addition BID27 showed that generated images can be successfully recovered by backpropagating through the latent space.

4 Jointly optimizing latent vectors and the generator parametrization via backpropagation of reconstruction losses was investigated in detail by BID3 .

The authors found that it is possible to train the generator entirely without a discriminator, still yielding a model that incorporates many of the desirable properties of GANs, such as smooth interpolations between samples.

Given that GAN training also gives us a discriminator for discerning between real and fake samples, one might reasonably consider directly applying the discriminator for detecting anomalies.

However, once converged, the discriminator exploits checkerboard-like artifacts on the pixel level, induced by the generator architecture BID35 BID28 .

While it perfectly separates real from forged data, it is not equipped to deal with samples which are completely unlike the training data.

This line of reasoning is verified in Section 4 experimentally.

Another approach we considered was to evaluate the likelihood of the final latent vectors {z j,k } nseed j=1under the noise prior p z .

This approach was tested experimentally in Section 4, and while it showed some promise, it was consistently outperformed by ADGAN.In BID42 , the authors propose a technique for anomaly detection (called AnoGAN) which uses GANs in a way somewhat similar to our proposed algorithm.

Their algorithm also begins by training a GAN.

In a manner similar to our own, given a test point x, their algorithm searches for a point z in the latent space such that g θ (z) ≈ x and computes the reconstruction loss.

Additionally they use an intermediate discriminator layer d ω and compute the loss between d ω (g θ (z)) and d ω (x).

They use a convex combination of these two quantities as their anomaly score.

In ADGAN we never use the discriminator, which is discarded after training.

This makes it easy to couple ADGAN with any GAN-based approach, e.g. LSGAN BID29 , but also any other differentiable generator network such as VAEs or moment matching networks BID25 .

In addition, we account for the non-convexity of the underlying optimization by seeding from multiple areas in the latent space.

Lastly, during inference we update not only the latent vectors z, but jointly update the parametrization θ of the generator.

Here we present experimental evidence of the efficacy of ADGAN.

We compare our algorithm to competing methods on a controlled, classification-type task and show anomalous samples from popular image datasets.

Our main findings are that ADGAN:• outperforms non-parametric as well as available deep learning approaches on two controlled experiments where ground truth information is available; • may be used on large, unsupervised data (such as LSUN bedrooms) to detect anomalous samples that coincide with what we as humans would deem unusual.

Our experiments are carried out on three benchmark datasets with varying complexity: (i.) MNIST (LeCun, 1998) which contains grayscale scans of handwritten digits.

We tested the performance of ADGAN against three traditional, non-parametric approaches commonly used for anomaly detection: (i.) KDE with a Gaussian kernel BID37 .

The bandwidth is determined from maximum likelihood estimation over ten-fold cross validation, with h ∈ {2 0 , 2 1/2 , . . .

, 2 4 }.

(ii.)

One-class support vector machine (OC-SVM) BID43 ) with a Gaussian kernel.

The inverse length scale is selected from estimating performance on a small holdout set of 1000 samples, and γ ∈ {2 −7 , 2 −6 , . . .

, 2 −1 }.

(iii.)

Isolation forest (IF), which was largely stable to changes in its parametrization.

(iv.)

Gaussian mixture model (GMM).

We allowed the number of components to vary over {2, 3, . . .

, 20} and selected suitable hyperparameters by evaluating the Bayesian information criterion.

For the methods above we reduced the feature dimensionality before performing anomaly detection.

This was done via PCA BID38 , varying the dimensionality over {20, 40, . . .

, 100}; we simply report the results for which best performance on a small holdout set was attained.

As an alternative to a linear projection, we evaluated the performance of both methods after applying a nonlinear transformation to the image data instead via an Alexnet BID22 , pretrained on Imagenet BID7 .

Just as on images, the anomaly detection is carried out on the representation in the final convolutional layer of Alexnet.

This representation is then projected down via PCA, as otherwise the runtime of KDE and OC-SVM becomes problematic.

We also report the performance of two end-to-end deep learning approaches: VAEs and DCAEs.

For the DCAE we scored according to reconstruction losses, interpreting a high loss as indicative of a new sample differing from samples seen during training.

In VAEs we scored by evaluating the evidence lower bound (ELBO).

We found this to perform much better than thresholding directly via the prior likelihood in latent space or other more exotic approaches, such as scoring from the variance of the inference network.

In both DCAEs and VAEs we use a convolutional architecture similar to that of DCGAN , with batch norm regularizations BID18 and ReLU activations in each layer.

We also report the performance of AnoGAN.

To put it on equal footing, we pair it with DCGAN , the same architecture also used for training in our approach.

ADGAN requires a trained generator.

For this purpose, we trained on the WGAN objective (2), as this was much more stable than using GANs.

The architecture was fixed to that of DCGAN .

Following BID31 we set the dimensionality of the latent space to d = 256.For ADGAN, the searches in the latent space were initialized from the same noise prior that the GAN was trained on (in our case a normal distribution).

To take into account the non-convexity of the problem, we seeded from n seed = 8 points.

For the optimization of latent vectors and the parameters of the generator we used the Adam optimizer BID19 .

5 When searching for a point in in the latent space to match a test point, we found that more optimization steps always improved the performance in our experiments.

We found k = 5 steps to be a good trade-off between execution time and accuracy and used this value in the results we report.

Unless otherwise noted, we measured reconstruction quality with a squared L 2 loss.

The first task is designed to quantify the performance of competing methods.

In it, we closely follow the original publication on OC-SVMs BID43 and begin by training each model on data from a single class from MNIST.

We then evaluate performance on 5000 items randomly selected from the test set, which contains samples from all classes.

In each trial, we label the classes unseen in training as anomalous.

Ideally, a method assigns images from anomalous classes (say, digits 1-9) a higher anomaly score than images belonging to the normal class (zeros).

Varying the decision threshold yields the receiver operating characteristic (ROC), shown in FIG1 .

In Table 1 and 2, we report the AUCs that resulted from leaving out each class.

The second experiment follows this guideline with the colored images from CIFAR-10.In these controlled experiments we highlight the ability of ADGAN to perform on-par with traditional methods at the task of inferring anomaly of low-dimensional samples such as those contained in MNIST.

On CIFAR-10 we see that all tested methods see a drop in performance.

For these experiments ADGAN performed best, needing eight seeds to achieve this result.

Using a non-linear transformation with a pretrained Alexnet did not improve the performance of either MNIST or CI-FAR10, see Table 1 .While neither table explicitly contains results from scoring the samples using the GAN discriminator, we did run these experiments for both datasets.

Performance was weak, with an average AUC of 0.625 for MNIST and 0.513 for CIFAR-10.

Scoring according to the prior likelihood p z of the final latent vectors worked slightly better, resulting in an average AUC of 0.721 for MNIST and 0.554 for CIFAR-10.

In the second task we showcase the use of ADGAN in a practical setting where no ground truth information is available.

For this we first trained a generator on LSUN scenes.

We then used ADGAN to find the most anomalous images within the corresponding validation sets containing 300 images.

6 The images associated with the highest and lowest anomaly scores are shown in FIG2 and FIG3 .

It should be noted that the training set sizes studied in this experiment prohibit the use of non-parametric methods such as KDE and OC-SVMs.

As can be seen from visually inspecting the LSUN scenes flagged as anomalous, our method has the ability to discern usual from unusual samples.

We infer that ADGAN is able to incorporate many properties of an image.

It does not merely look at colors, but also takes into account whether shown geometries are canonical, or whether an image contains a foreign object (like a caption).

Opposed to this, samples that are assigned a low anomaly score are in line with a classes' Ideal Form.

They show plain colors, are devoid of foreign objects, and were shot from conventional angles.

In the case of bedrooms, some of the least anomalous samples are literally just a bed in a room.

Additional images that were retrieved from applying our method to CIFAR-10 and additional LSUN scenes have been collected into the Appendix.

We showed that searching the latent space of the generator can be leveraged for use in anomaly detection tasks.

To that end, our proposed method: (i.) delivers state-of-the-art performance on standard image benchmark datasets; (ii.) can be used to scan large collections of unlabeled images for anomalous samples.

To the best of our knowledge we also reported the first results of using VAEs for anomaly detection.

We remain optimistic that boosting its performance is possible by additional tuning of the underlying neural network architecture or an informed substitution of the latent prior.

Accounting for unsuitable initializations by jointly optimizing latent vectors and generator parameterization are key ingredients to help ADGAN achieve strong experimental performance.

Nonetheless, we are confident that approaches such as initializing from an approximate inversion of the generator as in ALI BID8 , or substituting the reconstruction loss for a more elaborate variant, such as the Laplacian pyramid loss BID26 , can be used to improve our method further.

Shown are additional experiments in which we determine anomalous samples of different classes (e.g. birds, cats, dogs) contained in CIFAR-10.

ADGAN was applied exactly as described in Section 3, with the search carried out for k = 100 steps.

In FIG6 we report the highest and lowest reconstruction losses of images that were randomly selected from the test set, conditioned on the respective classes.

<|TLDR|>

@highlight

We propose a method for anomaly detection with GANs by searching the generator's latent space for good sample representations.

@highlight

The authors propose using GAN for anomaly detection, a gradient-descent based method to iteratively update latent representations, and a novel parameter update to the generators.

@highlight

A GAN based approach to doing anomaly detection for image data where the generator's latent space is explored to find a representation for a test image.