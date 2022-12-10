Deep Infomax~(DIM) is an unsupervised representation learning framework by maximizing the mutual information between the inputs and the outputs of an encoder, while probabilistic constraints are imposed on the outputs.

In this paper, we propose Supervised Deep InfoMax~(SDIM), which introduces supervised probabilistic constraints to the encoder outputs.

The supervised probabilistic constraints are equivalent to a generative classifier on high-level data representations, where class conditional log-likelihoods of samples can be evaluated.

Unlike other works building generative classifiers with conditional generative models, SDIMs scale on complex datasets, and can achieve comparable performance with discriminative counterparts.

With SDIM, we could perform \emph{classification with rejection}. Instead of always reporting a class label, SDIM only makes predictions when test samples' largest logits surpass some pre-chosen thresholds, otherwise they will be deemed as out of the data distributions, and be rejected.

Our experiments show that SDIM with rejection policy can effectively reject illegal inputs including out-of-distribution samples and adversarial examples.

Non-robustness of neural network models emerges as a pressing concern since they are observed to be vulnerable to adversarial examples (Szegedy et al., 2013; Goodfellow et al., 2014) .

Many attack methods have been developed to find imperceptible perturbations to fool the target classifiers (Moosavi-Dezfooli et al., 2016; Carlini & Wagner, 2017; Brendel et al., 2017) .

Meanwhile, many defense schemes have also been proposed to improve the robustnesses of the target models (Goodfellow et al., 2014; Tramèr et al., 2017; Madry et al., 2017; Samangouei et al., 2018 ).

An important fact about these works is that they focus on discriminative classifiers, which directly model the conditional probabilities of labels given samples.

Another promising direction, which is almost neglected so far, is to explore robustness of generative classifiers (Ng & Jordan, 2002) .

A generative classifier explicitly model conditional distributions of inputs given the class labels.

During inference, it evaluates all the class conditional likelihoods of the test input, and outputs the class label corresponding to the maximum.

Conditional generative models are powerful and natural choices to model the class conditional distributions, but they suffer from two big problems: (1) it is hard to scale generative classifiers on high-dimensional tasks, like natural images classification, with comparable performance to the discriminative counterparts.

Though generative classifiers have shown promising results of adversarial robustness, they hardly achieve acceptable classification performance even on CIFAR10 Schott et al., 2018; Fetaya et al., 2019) .

(2) The behaviors of likelihood-based generative models can be counter-intuitive and brittle.

They may assign surprisingly higher likelihoods to out-of-distribution (OoD) samples (Nalisnick et al., 2018; Choi & Jang, 2018) .

Fetaya et al. (2019) discuss the issues of likelihood as a metric for density modeling, which may be the reason of non-robust classification, e.g. OoD samples detection.

In this paper, we propose supervised deep infomax (SDIM) by introducing supervised statistical constraints into deep infomax (DIM, Hjelm et al. (2018) ), an unsupervised learning framework by maximizing the mutual information between representations and data.

SDIM is trained by optimizing two objectives: (1) maximizing the mutual information (MI) between the inputs and the high-level data representations from encoder; (2) ensuring that the representations satisfy the supervised statistical constraints.

The supervised statistical constraints can be interpreted as a generative classifier on high-level data representations giving up the full generative process.

Unlike full generative models making implicit manifold assumptions, the supervised statistical constraints of SDIM serve as explicit enforcement of manifold assumption: data representations (low-dimensional) are trained to form clusters corresponding to their class labels.

With SDIM, we could perform classification with rejection (Nalisnick et al., 2019; Geifman & El-Yaniv, 2017) .

SDIMs reject illegal inputs based on off-manifold conjecture (Samangouei et al., 2018; Gu & Rigazio, 2014) , where illegal inputs, e.g. adversarial examples, lie far away from the data manifold.

Samples whose class conditionals are smaller than the pre-chosen thresholds will be deemed as off-manifold, and prediction requests on them will be rejected.

The contributions of this paper are :

• We propose Supervised Deep Infomax (SDIM), an end-to-end framework whose probabilistic constraints are equivalent to a generative classifier.

SDIMs can achieve comparable classification performance with similar discrinimative counterparts at the cost of small over-parameterization.

• We propose a simple but novel rejection policy based on off-manifold conjecture: SDIM outputs a class label only if the test sample's largest class conditional surpasses the prechosen class threshold, otherwise outputs rejection.

The choice of thresholds relies only on training set, and takes no additional computations.

• Experiments show that SDIM with rejection policy can effectively reject illegal inputs, including OoD samples and adversarial examples generated by a comprehensive group of adversarial attacks.

Deep InfoMax (DIM, Hjelm et al. (2018) ) is an unsupervised representation learning framework by maximizing the mutual information (MI) of the inputs and outputs of an encoder.

The computation of MI takes only input-output pairs with the deep neural networks based esimator MINE (Belghazi et al., 2018) .

Let E φ be an encoder parameterized by φ, working on the training set

, and generating

.

DIM is trained to find the set of parameters φ such that: (1) the mutual information I(X, Y ) is maximized over sample sets X and Y. (2) the representations, depending on the potential downstream tasks, match some prior distribution.

Denote J and M the joint and product of marginals of random variables X, Y respectively.

MINE estimates a lower-bound of MI with Donsker-Varadhan (Donsker & Varadhan, 1983) representation of KL-divergence:

(1) where T ω (x, y) ∈ R is a family of functions with parameters ω represented by a neural network.

Since in representation learning we are more interested in maximizing MI, than its exact value, non-KL divergences are also favorable candidates.

We can get a family of variational lower-bounds using f -divergence representations (Nguyen et al., 2010) :

where f * is the Fenchel conjugate of a specific divergence f .

For KL-divergence, f * (t) = e (t−1) .

A full f * list is provided in Tab.

6 of Nowozin et al. (2016) .

Noise-Contrastive Estimation (Gutmann & Hyvärinen, 2010) can also be used as lower-bound of MI in "infoNCE" (Oord et al., 2018) .

All the components of SDIM framework are summurized in Fig. 1 .

The focus of Supervised Deep InfoMax (SDIM) is on introducing supervision to probabilistic constraints of DIM for (generative) classification.

We choose to maximize the local MI, which has shown to be more effective in classification tasks than maximizing global MI .

Equivalently, we minimize J MI :

where L φ (x) is a local M × M feature map of x extracted from some intermediate layer of encoder E, andĨ can be any possible MI lower-bounds.

By adopting a generative approach p(x, y) = p(y)p(x|y), we assume that the data follows the manifold assumption: the (high-dimensional) data lies on low-dimensional manifolds corresponding to their class labels.

Denotex the compact representation generated with encoder E φ (x).

In order to explicitly enforce the manifold assumption , we admit the existence of data manifold in the representation space.

Assume that y is a discrete random variable representing class labels, and p(x|y) is the real class conditional distribution of the data manifold given y. Let p θ (x|y) be the class conditionals we model parameterized with θ.

We approximate p(x|y) by minimizing the KL-divergence between p(x|y) and our model p θ (x|y), which is given by:

where the first item on RHS is a constant independent of the model parameters θ.

Eq. 4 equals to maximize the expectation Ex ,y∼p(x,y) [log p θ (x|y)].

In practice, we minimize the following loss J NLL , equivalent to empicically maximize the above expectation over

:

Besides the introduction of supervision, SDIM differs from DIM in its way of enforcing the statistical constraints: DIM use adversarial learning (Makhzani et al., 2015) to push the representations to the desired priors, while SDIM directly maximizes the parameterized class conditional probability.

Maximize Likelihood Margins Since a generative classifier, at inference, decides which class a test input x belongs to according to its class conditional probability.

On one hand, we maximize samples' true class conditional probabilities (classes they belong to) using J NLL ; On the other hand, we also hope that samples' false class conditional probabilities (classes they do not belong to) can be minimized.

This is assured by the following likelihood margin loss J LM :

where K is a positive constant to control the margin.

For each encoder outputx i , the C − 1 truefalse class conditional gaps are squared 1 , which quadratically increases the penalties when the gap becomes large, then are averaged.

Putting all these together, the complete loss function we minimize is:

Parameterization of Class Conditional Probability Each of the class conditional distribution is represented as an isotropic Gaussian.

So the generative classifier is simply a embedding layer with C entries, and each entry contains the trainable mean and variance of a Gaussian.

This minimized parameterization encourages the encoder to learn simple and stable low-dimensional representations that can be easily explained by even unimodal distributions.

Considering that we maximize the true class conditional probability, and minimize the false class conditional probability at the same time, we do not choose conditional normalizing flows, since the parameters are shared across class labels, and the training can be very difficult.

In Schott et al. (2018) , each class conditional probability is represented with a VAE, thus scaling to complex datasets with huge number of classes, e.g. ImageNet, is almost impossible.

A generative approach models the class-conditional distributions p(x|y), as well as the class priors p(y).

For classification, we compute the posterior probabilities p(y|x) through Bayes' rule:

The prior p(y) can be computed from the training set, or we simply use uniform class prior for all class labels by default.

Then the prediction of test sample x * from posteriors is:

The drawback of the above decision function is that it always gives a prediction even for illegal inputs.

Instead of simply outputting the class label that maximizes class conditional probability of x * , we set a threshold for each class conditional probability, and define our decision function with rejection to be:

The model gives a rejection when log p(x * |y * ) is smaller than the threshold δ y * .

Note that here we can use p(x * |y * ) and p(x * |y * ) interchangeably.

This is also known as selective classification (Geifman & El-Yaniv, 2017) or classification with reject option (Nalisnick et al., 2019) (See Supp.

A) 4 RELATED WORKS Robustness of Likelihood-based Generative Models Though likelihood-based generative models have achieved great success in samples synthesis, the behaviors of their likelihoods can be counter-intuitive.

Flow-based models (Kingma & Dhariwal, 2018) and as well as VAEs (Kingma & Welling, 2013) , surprisingly assign even higher likelihoods to out-of-distribution samples than the samples in the training set (Nalisnick et al., 2018; Choi & Jang, 2018) .

Pixel-level statistical analyses in Nalisnick et al. (2018) show that OoD dataset may "sit inside of" the in-distribution dataset (i.e. with roughly the same mean but smaller variance).

Off-Manifold Conjecture Grosse et al. (2017) observe that adversarial examples are outside the training distribution via statistical testing.

DefenseGAN (Samangouei et al., 2018) models real data distribution with the generator G of GAN.

At inference, instead of feeding the test input x to the target classifier directly, it searches for the "closest" sample G(z * ) from generator distribution to x as the final input to the classifier.

It ensures that the classifier only make predictions on the data manifold represented by the generator, ruling out the potential adversarial perturbations in x. PixelDefend (Song et al., 2017) takes a similar approach which uses likelihood-based generative model -PixelCNN to model the data distribution.

Both DefenseGAN and PixelDefend are additionally trained as peripheral defense schemes agnostic to the target classifiers.

Training generative models on complex datasets notoriously takes huge amount of computational resources (Brock et al., 2018) .

In contrast, the training of SDIM is computationally similar to its discriminative counterpart.

The verification of whether inputs are offmanifold is a built-in property of the SDIM generative classifier.

The class conditionals of SDIM are modeled on low-dimensional data representations with simple Gaussians, which is much easier, and incurs very small computations.

Datasets We evaluate the effectiveness of the rejection policy of SDIM on four image datasets: MNIST, FashionMNIST (both resized to 32×32 from 28×28); and CIFAR10, SVHN.

See App.

B.1 for details of data processing.

For out-of-distribution samples detection, we use the dataset pairs on which likelihood-based generative models fail (Nalisnick et al., 2018; Choi & Jang, 2018) : FashionMNIST (in)-MNIST (out) and CIFAR10 (in)-SVHN (out).

Adversarial examples detection are evaluated on MNIST and CIFAR10.

Choice of thresholds It is natural that choosing thresholds based on what the model knows, i.e. training set, and can reject what the model does not know, i.e. possible illegal inputs.

We set one threshold for each class conditional.

For each class conditional probability, we choose to evaluate on two different thresholds: 1st and 2nd percentiles of class conditional log-likelihoods of the correctly classified training samples.

Compared to the detection methods proposed in , our choice of thresholds is much simpler, and takes no additional computations.

Models A typical SDIM instance consists of three networks: an encoder, parameterized by φ, which outputs a d-dimensional representation; mutual information evaluation networks, i.e. T ω in Eqn.

(1) and Eqn.

(2); and C-way class conditional embedding layer, parameterized by θ, with each entry a 2d-dimensional vector.

We set d = 64 in all our experiments.

For encoder of SDIM, we use ResNet (He et al., 2016) on 32 × 32 with a stack of 8n + 2 layers, and 4 filter sizes {32, 64, 128, 256}. The architecture is summarized as:

The last layer of encoder is a d-way fully-connected layer.

To construct a discriminative counterpart, we simply set the output size of the encoder's last layer to C for classification.

We use ResNet10 (n = 1) on MNIST, FashionMNIST, and ResNet26 (n = 3) on CIFAR10, SVHN.

We report the classification accuracies (see Tab.

1) of SDIMs and the discriminative counterparts on clean test sets .

Results show that SDIMs achieve the same level of accuracy as the discriminative counterparts with slightly increased number of parameters (17% increase for ResNet10, and 5% increase for ResNet26).

We are aware of the existence of better results reported on these datasets using more complex models (Huang et al., 2017; Han et al., 2017) or automatically designed architectures (Cai et al., 2018) , but pushing the state-of-the-art is not the focus of this paper.

Schott et al. (2018) , both model class conditional probability with VAE (Kingma & Welling, 2013; Rezende et al., 2014) , and achieve acceptable accuracies (> 98%) on MNIST.

However, it is hard for fully conditional generative models to achieve satisfactory classification accuracies even on CIFAR10.

On CIFAR10, methods in achieve only < 50% accuracy.

They also point out that the classification accuracy of a conditional PixelCNN++ (Salimans et al., 2017 ) is only 72.4%.

The test accuracy of ABS in (Schott et al., 2018 ) is only 54%.

In contrast, SDIM could achieve almost the same performance with similar discriminative classifier by giving up the full generative process, and building generative classifier on high-level representations.

improves the accuracy to 92% by feeding the features learned by powerful discriminative classifier-VGG16 (Simonyan & Zisserman, 2014) to their generative classifiers, which also suggests that modeling likelihood on high-level representation (features) is more favorable for generative classification than pixel-level likelihood of fully generative classifiers.

For classification tasks, discovering discriminative features is much more important than reconstructing the all the image pixels.

Thus performing generative classification with full generative models may not be the right choice.

We also investigate the implications of the proposed decision function with rejection under different thresholds.

The results in Tab.

2 show that choosing a higher percentile as threshold will reject more prediction requests.

At the same time, the classification accuracies of SDIM on the left test sets become increasingly better.

This demonstrate that out rejection policy tend to reject the ones on which SDIMs make wrong predictions.

Table 2 : Classification performances of SDIMs using the proposed decision function with rejection.

We report the rejection rates of the test sets and the accuracies on the left test sets for each threshold.

Class-wise OoD detections are performed, and mean detection rates over all in-distribution classes are reported in Tab.

3.

For each in-distribution class c, we evaluate the log-likelihoods of the whole OoD dataset.

Samples whose log-likelihoods are lower the class threshold δ c will be detected as OoD samples.

Same evaluations are applied on conditional Glows with 10th percentile thresholds, but the results are not good.

The results are clear and confirm that SDIMs, generative classifiers on high-level representations, are more effective on classification tasks than fully conditional generative models on raw pixels.

Note that fully generative models including VAE used in ; Schott et al. (2018) fail on OoD detection.

The stark difference between SDIM and full generative models (flows or VAEs) is that SDIM models samples' likelihoods in the high-level representation spaces, while generative models evaluate directly on the raw pixels.

See Supp.

C for more results about the histograms of the class conditionals of in-out distributions.

Table 3 : Mean detection rates of SDIMs and Glows with different thresholds on OoD detection.

We comprehensively evaluate the robustness of SDIMs against various attacks:

• gradient-based attacks: one-step gradient attack FGSM (Goodfellow et al., 2014) , its iterative variant projected gradient descent (PGD, Kurakin et al. (2016) ; Madry et al. (2017) ), CW-L 2 attack (Carlini & Wagner, 2017) , deepfool (Moosavi-Dezfooli et al., 2016 ).

• score-based attacks: local search attack (Narodytska & Kasiviswanathan, 2016 ).

• decision-based attack: boundary attack (Brendel et al., 2017) .

Attacks Using Cross-Entropy We find that SDIMs are much more robust to gradient-based attacks using cross-entropy, e.g. FGSM and PGD, since the gradients numerically vanish as a side effect of the likelihood margin loss J LM of SDIM.

This phenomenon is similar to some defences that try to hinder generations of adversarial examples by masking the gradients on inputs.

While full generative classifiers in still suffer from these attacks.

See Supp.

D.1 for detailed results.

Conservative Adversarial Examples Adversarial attacks aim to find the minimal perturbations that sufficiently change the classification labels, i.e. flip other logits to be the largest one.

We show case examples on MNIST generated by untargeted attacks and their logits in Tab.

4 (See Supp.

D.2 for examples of CIFAR10).

Though these attacks successfully flip the logits, they are designed to be conservative to avoid more distortions to the original images.

As a result, the largest logits of adversarial examples are still much lower than the thresholds, so they can be detected by our rejection policy.

We find that our rejection policy performs perfectly on MNIST, but fails to detect all adversarial examples on CIFAR10 except for Boundary attack (See Tab.

5).

It seems to be a well-known observation that models trained on CIFAR10 are more vulnerable than one trained on MNIST.

Gilmer et al. (2018) connects this observation to the generalization of models.

They found that many test samples, though correctly classified, are close to the misclassfied samples, which implies the existence of adversarial examples.

If a model has higher error rate, it would take smaller perturbations to move correctly classified samples to misclassified areas.

Table 4 : Full logits of the adversarial examples generated with different attacks.

The original image is the first sample of class 0 of MNIST test set.

The first row gives the 1st percentile thresholds, and the second row shows the logits of the original image.

The largest logits are marked in bold.

Adversarial examples with more confidence Based on the observations above, a natural question we should ask is: can we generate adversarial examples with not only successfully flipped logits, Table 5 : Detection rates of our rejection policies.

We perform untargeted adversarial evaluation on the first 1000 images of test sets.

CW-L 2 is not involved here, but carefully investigated below.

but also the largest logit larger than some threshold value?

Unlike the conservativeness on paying more distortions of other attacks, CW attack allows us to control the gap between largest and second largest logits with some confidence value κ.

We perform targeted CW attacks with confidences κ = {0, 500, 1000} (Tab.

6).

We find that increasing the confidences help increasing the largest logits of adversarial examples to some extent, but may lead to failures of generation.

The sensitivity to confidence values is also different given different targets.

The success rates of generating adversarial examples monotonically decreases with the confidences increasing (Tab.

7).

Note that on discriminative counterparts, CW-L 2 with the same settings easily achieves 100% success rates.

This means that explicitly forcing data representations to form clusters with maximum margins between them help increase average distances between normal samples and the nearest misclassified areas, thus increase the hardness of finding minimal adversarial perturbations .

In this case, it takes a large enough adversarial perturbation to move a sample from its cluster to the other.

Meanwhile, detection rates remain satisfactory on MNIST, but obviously decline on CIFAR10.

For victim generative classifiers in under CW-L 2 attack, the detection rates of adversarial examples using the proposed detection methods can be > 95% on MNIST, but fall < 50% on even CIFAR10-binary (their models don't scale on CIFAR10, and CW-L 2 with non-zero confidences are also not evaluated).

Table 7 : Targeted adversarial evaluations results of our rejection policies on the first 1000 test samples.

We report the detection rates with different thresholds and success rates of generating adversarial examples.

Discussions on off-manifold conjecture Gilmer et al. (2018) challenges whether the off-manifold conjecture holds in general.

They experiment on synthetic dataset-two high-dimensional concentric spheres with theoretical analyses, showing that even for a trained classifier with close to zero test error, there may be a constant fraction of the data manifold misclassified, which indicates the existence of adversarial examples within the manifold.

But there are still several concerns to be addressed: First, as also pointed out by the authors, the manifolds in natural datasets can be quite complex than that of simple synthesized dataset.

Fetaya et al. (2019) draws similar conclusion from analyses on synthesized data with particular geometry.

So the big concern is whether the conclusions in Gilmer et al. (2018); Fetaya et al. (2019) still hold for the manifolds in natural datasets.

A practical obstacle to verify this conclusion is that works modeling the full generative processes are based on manifold assumption, but provide no explicit manifolds for analytical analyses like Gilmer et al. (2018); Fetaya et al. (2019) .

While SDIM enables explicit and customized manifolds on high-level data representations via probabilistic constraints, thus enables analytical analyses.

In this paper, samples of different classes are trained to form isotropic Gaussians corresponding to their classes in representation space (other choices are possible).

The relation between the adversarial robustness and the forms and dimensionalities of data manifolds is to be explored.

Second, in their experiments, all models evaluated are discriminative classifiers.

Considering the recent promising results of generative classifiers against adversarial examples, would using generative classifiers lead to different results?

One thing making us feel optimistic is that even though the existence of adversarial examples is inevitable, Gilmer et al. (2018) suggest that adversarial robustness can be improved by minimizing the test errors, which is also supported by our experimental differences on MNIST and CIFAR10.

We introduce supervised probabilistic constraints to DIM.

Giving up the full generative process, SDIMs are equivalent to generative classifiers on high-level data representations.

Unlike full conditional generative models which achieve poor classification performance even on CIFAR10, SDIMs attain comparable performance as the discriminative counterparts on complex datasets.

The training of SDIM is also computationally similar to discriminative classifiers, and does not require prohibitive computational resources.

Our proposed rejection policy based on off-manifold conjecture, a built-in property of SDIM, can effectively reject illegal inputs including OoD samples and adversarial examples.

We demonstrate that likelihoods modeled on high-level data representations, rather than raw pixel intensities, are more robust on downstream tasks without the requirement of generating real samples.

We make comparisons between SDIM and GBZ , which consistently performs best in Deep Bayes.

FGSM and PGD-L ∞ The results in Fig 4 and Fig 5 show that SDIM performs consistently better than the baseline.

We find that increasing the distortion factor of FGSM has no influences of SDIM's accuracy, and the adversarial examples keep the same.

Recall that the class conditionals are optimized to keep a considerable margin.

Before evaluating the cross entropy loss, softmax is applied on the class conditionals log p(x|c) to generate a even sharper distribution.

So for the samples that are correctly classified, their losses are numerically zeros, and the gradient on inputs ∇J x (x, y) are also numerically zeros.

The PGD-L ∞ we use here is the randomized version (Madry et al., 2017) 2 , which adds a small random perturbation before the iterative loop.

The randomness is originally introduced to generate different adversarial examples for adversarial training, but here it breaks the zero loss so that the gradient on inputs ∇J x (x, y) will not be zeros in the loop.

FGSM can also be randomized (Tramèr et al., 2017) , which can be seen as a one-step variant of randomized PGD.

This phenomena is similar to what some defenses using gradient obfuscation want to achieve.

Defensive distillation (Carlini & Wagner, 2016) masks the gradients of cross-entropy by increasing the temperature of softmax.

But for CW attacks, which do not use cross-entropy, and operate on logits directly, this could be ineffective.

4 -353.1 -471.6 -400.3 -342.7 -367.2 -486.4 -326.4 Boundary 213.9 -417.4 -458.3 -548.0 -587.4 -236.3 214.0 -1246.1 -171.2 -555.6 LocalSearch 165.2 -485.7 190.9 -325.6 -439.0 -379.0 -318.8 -327.5 -357.9 -272.3 Table 9 : Full logits of adversarial examples generated with different attacks.

The original image is the fist sample of class 0 of CIFAR10 test set.

The first row gives the 1st percentile thresholds, and the second row shows the logits of the original image.

The largest logits are marked in bold.

<|TLDR|>

@highlight

scale generative classifiers  on complex datasets, and evaluate their effectiveness to reject illegal inputs including out-of-distribution samples and adversarial examples.