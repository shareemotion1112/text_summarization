The vulnerabilities of deep neural networks against adversarial examples have become a significant concern for deploying these models in sensitive domains.

Devising a definitive defense against such attacks is proven to be challenging, and the methods relying on detecting adversarial samples are only valid when the attacker is oblivious to the detection mechanism.

In this paper, we consider the adversarial detection problem under the robust optimization framework.

We partition the input space into subspaces and train adversarial robust subspace detectors using asymmetrical adversarial training (AAT).

The integration of the classifier and detectors presents a detection mechanism that provides a performance guarantee to the adversary it considered.

We demonstrate that AAT promotes the learning of class-conditional distributions, which further gives rise to generative detection/classification approaches that are both robust and more interpretable.

We provide comprehensive evaluations of the above methods, and demonstrate their competitive performances and compelling properties on adversarial detection and robust classification problems.

Deep neural networks have become the staple of modern machine learning pipelines, achieving stateof-the-art performance on extremely difficult tasks in various applications such as computer vision (He et al., 2016) , speech recognition (Amodei et al., 2016) , machine translation (Vaswani et al., 2017) , robotics (Levine et al., 2016) , and biomedical image analysis (Shen et al., 2017) .

Despite their outstanding performance, these networks are shown to be vulnerable against various types of adversarial attacks, including evasion attacks (aka, inference or perturbation attacks) (Szegedy et al., 2013; Goodfellow et al., 2014b; Carlini & Wagner, 2017b; Su et al., 2019) and poisoning attacks (Liu et al., 2017; Shafahi et al., 2018) .

These vulnerabilities in deep neural networks hinder their deployment in sensitive domains including, but not limited to, health care, finances, autonomous driving, and defense-related applications and have become a major security concern.

Due to the mentioned vulnerabilities, there has been a recent surge toward designing defense mechanisms against adversarial attacks (Gu & Rigazio, 2014; Jin et al., 2015; Papernot et al., 2016b; Bastani et al., 2016; Madry et al., 2017; Sinha et al., 2018) , which has in turn motivated the design of stronger attacks that defeat the proposed defenses (Goodfellow et al., 2014b; Kurakin et al., 2016b; a; Carlini & Wagner, 2017b; Xiao et al., 2018; Athalye et al., 2018; Chen et al., 2018; He et al., 2018) .

Besides, the proposed defenses have been shown to be limited and often not effective and easy to overcome (Athalye et al., 2018) .

Alternatively, a large body of work has focused on detection of adversarial examples (Bhagoji et al., 2017; Feinman et al., 2017; Gong et al., 2017; Grosse et al., 2017; Metzen et al., 2017; Hendrycks & Gimpel, 2017; Li & Li, 2017; Xu et al., 2017; Pang et al., 2018; Roth et al., 2019; Bahat et al., 2019; Ma et al., 2018; Zheng & Hong, 2018; Tian et al., 2018) .

While training robust classifiers focuses on maintaining performance in presence of adversarial examples, adversarial detection only cares for detecting these examples.

The majority of the current detection mechanisms focus on non-adaptive threats, for which the attacks are not specifically tuned/tailored to bypass the detection mechanism, and the attacker is oblivious to the detection mechanism.

In fact, Carlini & Wagner (2017a) and Athalye et al. (2018) showed that the detection methods presented in (Bhagoji et al., 2017; Feinman et al., 2017; Gong et al., 2017; Grosse et al., 2017; Metzen et al., 2017; Hendrycks & Gimpel, 2017; Li & Li, 2017; Ma et al., 2018) , are significantly less effective than their claimed performances under adaptive attacks.

The current solutions are mostly heuristic approaches that cannot provide performance guarantees to the adversary they considered.

In this paper, we are interested in detection mechanisms for adversarial examples that can withstand adaptive attacks.

Unlike previous approaches that assume adversarial and natural samples coming from different distributions, thus rely on using a single classifier to distinguish between them, we instead partition the input space into subspaces based on the classification system's output and perform adversarial/natural sample classification in these subspaces.

Importantly, the mentioned partitions allow us to drop the adversarial constrain and employ a novel asymmetrical adversarial training (AAT) objective to train robust binary classifiers in the subspaces.

Figure 1 demonstrates our idea of space partitioning and robust detector training.

Our qualitative results show that AAT supports detectors to learn class-conditional distributions, which further motivates generative detection/classification solutions that are both robust and interpretable.

Our specific contributions are:

• We develop adversarial example detection techniques that provide performance guarantees to norm constrained adversaries.

Empirically, our best models improve previous state-ofthe-art mean L 2 distortion from 3.68 to 4.47 on the MNIST dataset, and from 1.1 to 1.5 on the CIFAR10 dataset.

• We study powerful and versatile generative classification models derived from our detection framework and demonstrate their competitive performances over discriminative robust classifiers.

While defense mechanisms based on ordinary adversarial training are vulnerable to unrecognizable inputs (e.g., rubbish examples), inputs that cause confident predictions of our models have human-understandable semantic meanings.

• We demonstrate that AAT not only induces robustness as ordinary adversarial training methods do, but also promotes the learning of class-conditional distributions.

Intuitively, the learning mechanism is similar to that of GANs, but the objective doesn't learn a fixed generator.

On 1D and 2D benchmarking datasets we show this flexibility allows us to precisely control the data generation process such that the detector could be pushed to a good approximation of the underlying density function.

(In case of GANs at the global optimum the discriminator converges to a degenerated uniform solution.)

Our image generation results on CIFAR10 and ImageNet rival that of state-of-the-art GANs.

Adversarial attacks.

Since the pioneering work of Szegedy et al. (2013) , a large body of work has focused on designing algorithms that achieve successful attacks on neural networks (Goodfellow et al., 2014b; Moosavi-Dezfooli et al., 2016; Kurakin et al., 2016b; Chen et al., 2018; Papernot et al., 2016a; Carlini & Wagner, 2017b) .

More recently, iterative projected gradient descent (PGD), initially proposed by Kurakin et al. (2016b) , has been empirically identified as the most effective approach for performing norm ball constrained attacks, and the attack reasonably approximates the optimal attack (Madry et al., 2017) .

Adversarial detection techniques.

The majority of the methods developed for detecting adversarial attacks are based on the following core idea: given a trained K-class classifier, f : R d → {1...K}, and its corresponding natural training samples,

, generate a set of adversarially attacked samples

, and devise a mechanism to discriminate D from D .

For instance, Gong et al. (2017) use this exact idea and learn a binary classifier to distinguish the natural and adversarially perturbed sets.

Similarly, Grosse et al. (2017) append a new "attacked" class to the classifier, f , and re-train a secured network that classifies natural images, x ∈ D, into the K classes and all attacked images, x ∈ D , to the (K + 1)-th class.

In contrast to Gong et al. (2017); Grosse et al. (2017) , which aim at detecting adversarial examples directly from the image content, Metzen et al. (2017) trained a binary classifier that receives as input the intermediate layer features extracted from the classifier network f , and distinguished D from D based on such input features.

More importantly, Metzen et al. (2017) considered the so-called case of adaptive/dynamic adversary and proposed to harden the detector against such attacks using a similar adversarial training approach as in Goodfellow et al. (2014b) .

Unfortunately, the mentioned detection methods are significantly less effective under an adaptive adversary equipped with a strong attack (Carlini & Wagner, 2017a; Athalye et al., 2018) .

..

K} be the classifier that is used to do classification on D. With the labels and predicted labels the dataset respectively forms the partition

, 1} be a set of binary classifiers (detectors), in which h k is trained to discriminate natural samples classified as k, from adversarial samples that fool the network, f (·), to be classified as k. Also, let D be a set of L p norm bounded adversarial examples crafted from D:

Consider the following procedure to determine whether a sample x in D ∪ D is an adversarial example:

First obtain the estimated class label k := f (x), then use the k-th detector to predict: if h k (x) = 1 then x a natural sample, otherwise it's an adversarial sample.

The detection accuracy of the algorithm is given by

where

Thus minimizing the algorithm's classification error is equivalent to minimizing classification error of individual detectors.

Employing empirical risk minimization, detector k, parameterized by θ k , is trained by

where L is a loss function that measures the distance between h k 's output and the supplied label (e.g., the binary cross-entropy loss).

In the case of adaptive attacks, when the adversary aims to fool both the classifier and detectors, the accuracy of a naively trained detector could be significantly reduced.

In order to be robust to adaptive attacks, inspired by the idea of robust optimization (Madry et al., 2017) , we incorporate the attack into the training objective:

where D f \k = {x : f (x) = k, y = k, x ∈ D}, and we assume that perturbation budget is large enough such that ∀x ∈ D f \k , ∃δ ∈ S , s.t.

f (x + δ) = k. Now by dropping the f (x + δ) = k constrain we could derive an upper bound for the first loss term: max

The detector could instead be trained by minimizing this upper bound using the following unconstrained objective,

Further, we use the fact that when D is used as the training set, f could overfit on D such that D \k = {x i : y i = k} and D k are respectively good approximations of D f \k and D f k .

This leads to our proposed asymmetrical adversarial training (AAT) objective:

(5) In a nutshell, each detector is trained using in-class natural samples and detector-adversarial examples crafted from out-of-class samples.

We use iterative PGD attack (Madry et al., 2017) to solve the inner maximization.

Because of the integrated adversary, objective 5 is no longer a straightforward discriminative objective.

Our investigations (Appendix A) showed that the objective promotes detectors to learn conditional data distributions.

Similar to GANs' objective (Goodfellow et al., 2014a) , the AAT objective presents a minimax problem, where the adversary tries to generate perturbed samples that look like the target class data, and the detector is trained by discriminating between target class data and perturbed data.

The key difference is that instead of learning a mapping from latent space distribution to the data distribution (a.k.a., the generator), AAT relies on gradient descent (a.k.a., PGD attack) to generate data samples.

This is crucial, as it allows us to perform fine-grained control over the generation process (especially by constraining on perturbation limit), so that the discriminator (detector) could retain density information (see Appendix A) and not converge to a degenerate uniform solution as in the case of GANs (Goodfellow et al., 2014a; Dai et al., 2017) .

Unfortunately, the detector does not define an explicit density function.

Under the energy-based learning framework (LeCun et al., 2006) , we could, however, obtain the joint probability of the input and a class category using the Gibbs distribution:

, where Z Θ is an unknown normalizing constant, and E θ k (x) = −z(h k (x)) (see Appendix A for justification).

We could then apply the Bayes classification rule to obtain a generative classifier: H(x) = arg max k p(x, k) = arg max k z(h k (x)).

In addition, we could base on p(x, k) to reject low probability inputs.

We implement the reject option by thresholdingk-th detector's logit output, wherek is the predicted class.

In the context of adversarial example detection, rejected samples are considered as adversarial examples.

We first test the robustness of individual detectors.

We show that, once we train a detector with an adequately configured PGD attack, its performance cannot be significantly reduced by an adversary with much stronger configurations (stronger in terms of steps and step-size).

Although the PGD attack (Madry et al., 2017) can reasonably solve the inner maximization in objective 5, it is not clear whether the optimization landscape of the asymmetrical objective is the same as its symmetrical counterparts.

For instance, we found that the step-size used by Madry et al. (2017) to train their CIFAR10 robust classifier would not induce robustness to our detectors (see Appendix D.2.2).

We also face a unique challenge when training with objective 5: the number of positive and negative samples are highly imbalanced.

Our solution is to use re-sampling to balance positive and negative classes.

Furthermore, we use adversarial finetuning on CIFAR10 and ImageNet to speed up the training of our detectors.

With the robustness test, we show that robust optimization also introduces robustness within this new training paradigm.

We use AUC (area under the ROC Curve) to measure detection performances.

The metric could be interpreted as the probability that the detector assigns a higher score to a random positive sample than to a random negative example.

While the true-positive and the false-positive rates are the commonly used metrics for measuring the detection performance, they require a detection threshold to be specified.

AUC, however, is an aggregated measurement of detection performance across a range of thresholds, and we found it to be a more stable and reliable metric.

For the k-th detector h k , its AUC is computed on the set {(x, 0) :

\k )} (refer to loss 4).

Having validated the robustness of individual detectors, we evaluate the overall performance of our integrated detection system.

Recalling our detection rule, we first obtain the estimated class label k := f (x), then use the k-th detector's logit output z(h k (x)) to predict: if z(h k (x)) ≥ T k , then x is a natural sample, otherwise it is an adversarially perturbed sample.

For the sake of this evaluation, we use a universal threshold for all the detectors: ∀k ∈ {1...K} T k = T , and report detection performance at a range of universal thresholds.

In practice, however, the optimal value of each detector's detection threshold T k should be determined by optimizing a utility function.

to denote the test set that contains natural samples, and

to denote the corresponding perturbed test set.

For a given threshold T , we compute the true positive rate (TPR) on D and false positive rate (FPR) on D .

These two metrics are respectively defined as

and

In the FPR definition we use f (x) = y to constrain that only true adversarial examples are counted as false positives.

This constraint is necessary, as we found that for the norm ball constraint we considered in the experiments, not all perturbed samples are adversarial examples that cause misclassification on f .

In order to craft the perturbed dataset D , we consider three attacking scenarios.

Classifier attack.

This attack corresponds to the scenario where the adversary is oblivious to the detection mechanism.

For a given natural sample x and its label y, the perturbed sample x is computed by minimizing the loss,

where z(f (x )) is the classifier's logit outputs.

This objective is derived from the CW attack (Carlini & Wagner, 2017b) and used in MadryLab (b) and MadryLab (a) to perform untargeted attacks.

Detectors attack.

In this scenario adversarial examples are produced by attacking only the detectors.

We construct a single detection function H by using the i-th detector's logit output as its i-th logit output:

.

H is then treated as a single network, and the perturbed sample x for a given input (x, y) is computed by minimizing the loss

Note that, according to our detection rule, a low value of the detector's logit output indicates detection of an adversarial example, thus by minimizing the negative of logit output we make the perturbed example harder to detect.

H could also be fed directly to the CW loss 8 or to cross-entropy loss, but we found the attack based on the loss in 9 to be significantly more effective.

Combined attack.

With the goal of fooling both the classifier and detectors, perturbed samples are produced by attacking the integrated detection system.

We consider two loss functions for realizing the combined attack.

The first is based on the combined loss function (Carlini & Wagner, 2017a) that has been shown to be effective against an array of detection methods.

Given a natural example x and its label y, same as the detectors-attack scenario, we first construct a single detection function H by aggregating the logit outputs of individual detectors: z(H(x)) i := z(h i (x)).

We then use the aggregated detector's largest logit output max k =y z(H(x)) k (low value of this quantity indicates detection of an adversarial example) and the classifier logit outputs z(f (x)) to construct a surrogate classifier g, with its logit outputs being

A perturbed example x is then computed by minimizing the loss function

In practice we observe that the optimization of this loss tends to stuck at the point where max i =y z(f (x )) i keeps changing signs while max j =y z(H(x)) j stays as a large negative number (which indicates detection).

To derive a more effective attack we consider a simple combination of loss 8 and loss 9:

The objective is straightforward: if x is not yet an adversarial example on f , optimize it for that goal; otherwise optimize it for fooling the aggregated detector.

We mention briefly here that we perform the same performance analysis of our generative detection method (as detailed in Section 3.2) by computing TPR on D and FPR on D .

We use the loss 9 to perform attacks against the generative detection method, but also provide results of attacks based on cross-entropy loss and CW loss 8.

Integrated classification.

In addition to the generative classifier proposed in Section 3.2, we introduce another classification scheme that provides a reject option.

The scheme is based an integration of the naive classifier f and the detectors: for a given input x and its prediction label

We respectively use loss 12 and 9 to attack the integrated classifier and the generative classifier.

Performance metric.

In the context of robust classification, the performance of a robust classifier is measured using standard accuracy and robust accuracy -accuracies respectively computed on the natural dataset and perturbed dataset.

We provide a similar performance analysis of the above classification models.

On the natural dataset

, we compute the accuracy as the fraction of samples that are correctly classified (f (x) = y) and at the same time not rejected (z(h k (x)) ≥ T ):

On the perturbed dataset

we compute the error as the fraction of samples that are misclassified (f (x) = y) and at the same time not rejected:

Note that in this case the error is no longer a complement of the accuracy.

For a classification system with a reject option, any perturbed samples that are rejected should be considered as properly handled, regardless of whether they are misclassified.

Thus on the perturbed dataset, the error, which is the fraction of misclassified and not rejected samples, is a more proper notion of such system's performance.

For a standard robust classifier, its perturbed set error is computed as the complement of its accuracy on the perturbed set.

Using different p-norm and maximum perturbation constrains we trained four detection systems (each has 10 base detectors), with training and validation adversarial examples optimized using PGD attacks of different steps and step-size (see Table 6 ).

At each step of PGD attack we use the Adam optimizer to perform gradient descent, both for L 2 and L ∞ constrained scenarios.

Appendix B.1 provides more training details.

Robustness results.

The robustness test results in Table 1 confirm that the base detectors trained with objective 5 are able to withstand much stronger PGD attacks, for both L 2 and L ∞ scenarios.

Normalized steepest descent is another popular choice for performing PGD attack (Madry et al., 2017; MadryLab, b; a) , for which we got similar robustness results (Table 8) .

Further results on the complete list of performances of the L ∞ = 0.3 and L ∞ = 0.5 trained detectors, cross-norm and cross-perturbation test results, and random restart test results are included in Appendix D.1.

Table 1 : AUC scores of the first two detectors (k = 0, 1) tested with different strengths of PGD attacks using Adam optimizer.

PGD attack steps, step-size Detection results.

Figure 2a shows that the combined attack is the most effective attack against integrated detection.

Generative detection (attacked using loss 9) outperforms integrated detection, especially when the detection threshold is low (the region where TPR is high).

In Figure 12 we confirm that loss 9 is more effective than CW loss and cross-entropy loss for attacking generative detection.

Notably, the red curve that overlaps the y-axis shows that integrated detection can perfectly detect adversarial examples crafted by attacking only the classifier (using objective 8).

In Table 2 we compare the performances of our generative detection method with the state-of-the-art detection method as identified by Carlini & Wagner (2017a) .

Our method using L ∞ = 0.5 trained base detectors is able to outperform the state-of-the-art method by a large margin.

Appendix C describes the procedure we used to compute the mean L 2 distortion of our method.

Classification results.

In Figure 2b , we compare the robust classification performance of our methods and a state-of-the-art robust classifier.

While the performance of the robust classifier is fixed, by using different rejection thresholds, our classification methods provide the option to balance standard accuracy and robust error.

The generative classifier outperforms the integrated classifier when the rejection threshold is low (i.e., when the perturbed set error is high).

We observe that a stronger attack ( = 0.4) breaks the robust classifier, while the generative classifier still exhibits robustness, even though both systems are trained with the same L ∞ = 0.3 constrain.

shows perturbed samples by performing targeted attacks against the generative classifier and robust classifier.

We observe that perturbed samples produced by attacking the generative classifier have distinguishably visible features of the target class, indicating that the base detectors, from which the generative classifier is built, have successfully learned the class-conditional distributions, and the perturbation has to change the semantics of the underlying sample for a successful attack.

In contrast, perturbations introduced by attacking the robust classifier are not interpretable, even though they could cause high logit output of the target classes (see Figure 13 for the logit outputs distribution).

Following this path and use a larger perturbation limit, it is straightforward to generate unrecognizable images that cause highly confident predictions of the robust classifier.

Perturbed samples (generative classifier) Perturbed samples (robust classifier)

Figure 3: Natural samples and corresponding perturbed samples produced by performing a targeted attack against the generative classifier and robust classifier (Madry et al., 2017) .

Targets from top row to bottom row are digit class from 0 to 9.

We perform the targeted attack by maximizing the logit output of the targeted class, using L ∞ = 0.4 constrained PGD attack of steps 100 and step-size 0.01.

We note that both classifiers use L ∞ = 0.3 for their training constraint.

On CIFAR10 we train the base detectors using L ∞ = 8 constrain PGD attack of steps 40 and step size 0.5.

Note that the scale of and step-size here is 0-255 (rather than 0-1 as in the case of MNIST).

The robust classifier (Madry et al., 2017 ) that we will compare with is trained with the same L ∞ = 8 constraint but with a different step-size (see Appendix D.2.2 for a discussion of step-sizes).

Appendix B.2 provides the training details.

Robustness results.

Table 3 shows that the base detector models can withstand attacks that are significantly stronger than the training attack.

In Appendix D.2.1 we present random restart test results, cross-norm and cross-perturbation test results, and robustness test result for L 2 based models.

Detection results.

Consistent with the MNIST results, in Figure 4a combined attack is the most effective method against integrated detection.

Similarly, the generative detection outperforms integrated detection when the detection threshold is low (i.e., where TPR is high).

In this figure we use loss 9 to attack generative detection, and in Figure 14 we show that it's more effective than attack based on cross-entropy loss and CW loss.

In Table 4 our method outperforms the state-of-the-art adversarial detection method.

Classification results.

In Figure 4b , we did not observe a dramatic decrease in the robust classifier's performance when we increase the perturbation limit to = 12.

Integrated classification can reach the standard accuracy of a regular classifier, but at the cost of significantly increasing error on the perturbed set.

Figure 5 shows some perturbed samples produced by attacking the generative classifier and robust classifier.

While these two classifiers have similar errors on the perturbed set, samples produced by attacking the generative classifier have more visible features of the targets, which indicates that the adversary has to change more semantic in order to cause the same error.

Figures 6 and 15 demonstrate that hard to recognize images are able to cause high logit outputs of the robust classifier.

Such examples highlight a major defect of the defense mechanisms based on ordinary adversary training: they could be easily fooled by unrecognizable inputs (Nguyen et al., 2015; Goodfellow et al., 2014b; Schott et al., 2018) .

In contrast, samples that cause high logit outputs of the generative classifier all have clear semantic meaning.

Since both classifiers are trained with L ∞ = 8 constrain, these results indicate that AAT improves robust and interpretable feature learning.

(See Appendix E for Gaussian noise attack results and a discussion about the interpretability of our approach.)

The visual similarity between generated samples in Figure 6 and real samples further suggests that detectors have successfully learned the conditional data distributions.

Similarly, on ImageNet, we show asymmetrical adversarial training induces detection robustness and supports the learning of class-conditional distributions.

Our experiment is based on Restricted ImageNet (Tsipras et al., 2018), a subset of ImageNet that has its samples reorganized into customized categories.

The dog category contains images of different dog breeds collected from ImageNet class between 151 and 268.

We trained a dog class detector by finetuning a pre-trained ResNet50 model.

The dog category covers a range of ImageNet classes, with each one having its logit output.

We use the subnetwork defined by the logit output of class 151 as the detector (in principle logit output of other classes in the range should also work).

Due to computational constraints, we only validated the robustness of a L ∞ = 0.02 trained detector (trained with PGD attack of steps 40 and step-size 0.001), and we present the result in Table 5 Generated by attacking generative classifier Generated by attacking robust classifier Figure 6 : Images generated from class-conditional Gaussian noise by performing targeted attack against the generative classifier and robust classier.

we use PGD attack of steps 60 and step-size 0.5 × 255 to perform L 2 = 30 × 255 constrained attack (same as Santurkar et al. (2019) .

The Gaussian noise inputs from which these two plots are generated are the same.

Samples not selected.

In this paper, we studied the problem of adversarial detection under the robust optimization framework and proposed a novel adversarial detection scheme based on input space partitioning.

Our formulation leads to a new generative modeling technique which we called asymmetrical adversarial training (AAT).

AAT's capability to learn class-conditional distributions further gives rise to generative detection/classification methods that show competitive performance and improved interpretability.

In particular, our generative classifier is more resistant to "rubbish examples", a significant threat to even the most successful defense mechanisms.

High computational cost (see Appendix F for more discussion) is a major drawback of our methods, and in the future, we will explore the idea of shared computation between detectors.

A A DEEPER LOOK AT ASYMMETRICAL ADVERSARIAL TRAINING Under the energy-based learning framework (LeCun et al., 2006) , the AAT objective could be understood as learning an energy function that has low energy outputs on target class data points, and high energy outputs anywhere else.

The energy function in this case could be defined using the logit output of the target detector:

.

Using the Gibbs distribution we could obtain a density function and it should be interpreted as the joint distribution of the data point and the corresponding class category p(x, k) =

, where Z Θ is a normalizing constant known as the partition function, and could be computed in our case as Z Θ = k exp(−E θ k (x))dx.

(We note a similar formulation is used by Anonymous (2020) .)

When x is in high dimensional space, Z Θ is intractable, but for generative classification, only knowing the unnormalized probability will suffice.

While ordinary discriminative training only learns a good discriminative boundary, AAT is able to learns the underlying density function.

In Figure 7 we provide a schematic illustration of this phenomenon (results on real 1D data is in Figure 8 ).

In Figure 9 we observe similar results on 2D benchmarking datasets (Srivastava et al., 2017; Metz et al., 2016; De Cao et al., 2019; Huang et al., 2018; Chen et al., 2019 ) that detectors are able to cover all models of the distributions.

We note that unlike GANs that at the global optimum the discriminator converge to uniform solution and thus could not retain density information (Goodfellow et al., 2014a; Dai et al., 2017) , by properly constraining the adversary (especially on the perturbation limit), AAT is able to stably push the detector to a good approximation of the underlying density function.

Being able to do reliable (implicit) density estimation justifies our energy-based generative classification formulation (we leave the theoretical analysis of the objective's convergence property to future work).

The compelling property of AAT generalizes to the case of high dimensional data.

In the following, we present CIFAR10 and ImageNet image generation results for our models and state-of-the-art GANs models (Figure 10 vs. Figure 11 for CIFAR10, and Figure 21 vs. Figure 23 for ImageNet).

While images generated by our models are not necessarily more "realistic", they clearly have captured the structures of the target objects.

In particular, the results in Figure 21 show that our model is only modeling the main structures of target objects, but not other irrelevant elements like backgrounds and other objects.

We note that apart from being able to produce state-of-the-art image generation results, our approach at the same time provides classification/detection performance guarantee against norm constrained adversarial examples, while other generative models cannot (Fetaya et al., 2019) .

The resulting energy function after training.

The blue data distribution has two modes, but the energy function is not able to capture this structure due to its minute effects on the discriminative objective.

(c) By solving the inner maximization in objective 5 red points are pushed towards to the low energy region, but (crucially) they are not collapsed to the lowest energy point due to the perturbation limit.

The gap between the two sets of blue points is now filled with red points; minimizing the loss

] causes the energy function to be pulled up on this region.

(d) The energy function after training is able to capture the internal structure of the blue data.

The positive class data (blue points) are sampled from a mixture of Gaussians (mean 0.4 with std 0.01, and mean 0.6 with std 0.005, each with 250 samples).

Both the blue and red data has 500 samples.

The estimated density function is computed using Gibbs distribution and network logit outputs.

PGD attack steps 20, step size 0.05, and perturbation limit = 0.3.

Figure 9 : 2D datasets (top row, blue points are class 1 data, and red points are class 0 data, both have 1000 data points) and sigmoid outputs of AAT trained models (bottom row).

The architecture of the MLP model for solving these tasks is 2-500-500-500-500-500-1.

PGD attack steps 10, step-size 0.05, and perturbation limit L ∞ = 0.5.

We use 50K samples from the original training set for training and the rest 10K samples for validation, and report test performances based on the epoch-saved checkpoint that gives the best validation performance.

All base detectors are trained using a network consisting of two max-pooled convolutional layers each with 32 and 64 filters, and a fully connected layer of size 1024, same as the one used in Madry et al. (2017) .

At each iteration we sample a batch of 320 samples, from which in-class samples are used as positive samples, and out-of-class samples are used as the source for adversarial examples that will be used as negative samples.

To balance positive and negative examples at each batch, we resample the out-of-class set to have same number of samples as in-class set.

All base detectors are trained for 100 epochs.

Table 6 : MNIST dataset PGD attack steps and step-sizes for base detector training and validation.

We train our CIFAR10 base detectors using the ResNet50 model (He et al., 2016; Madry et al., 2017) .

To speedup training, we take advantage of a natural trained classifier: the subnetwork of f that defines the output logit z(f (·)) k is essentially a "detector", that would output high values for samples of class k, and low values for others.

Our detector is then trained by finetuning the subnetwork using objective 5.

Our pretrained classifier has a test accuracy of 95.01% (fetched from the CIFAR10 adversarial challenge (MadryLab, a)).

At each iteration of training we sample a batch of 300 samples, from which in-class samples are used as positive samples, while an equal number of out-of-class samples are used as sources for adversarial examples.

Adversarial examples for training L 2 and L ∞ models are both optimized using PGD attack with normalized steepest descent (MadryLab, b) .

We report results based on the best performances on the CIFAR10 test set (thus don't claim generalization performance of the proposed method).

We first find the detection threshold T with which the detection system has 0.95 TPR.

We construct a new loss function by adding a weighted loss term that measures perturbation size to objective 9

We then use unconstrained PGD attack to optimize L(x ).

We use binary search to find the optimal c, where in each bsearch attempt if x is a false positive (max i z(H(x )) i = y and max i =y z(H(x )) i > T ) we consider the current c as effective and continue with a larger c. The configurations for performing binary search and PGD attack are detailed in Table 7 .

The c upper bound is established such that with this upper bound, no samples except those that are inherently misclassified by the generative classifier, could be perturbed as a false positive.

With these settings, our MNIST generative detector with L ∞ = 0.3 base detectors reached 0.9962 FPR, generative detector with L ∞ = 0.5 base detectors reached 0.9936 FPR, and CIFAR10 generative detector reached 0.9995 FPR.

We note that it's very difficult to find the optimal c in loss 15 using binary search, hence performance based on mean L 2 distortion is not precise, and we encourage future work to measure detection performances based on norm constrained attacks (as in Figure 2a ).

Table 8 : AUC scores of the first two base detectors under different strengths of PGD attacks using normalized steepest descent.

The gradient descent rules for L 2 and L ∞ constrained attacks are respectively x n+1 = x n − γ ∇f (xn) ∇f (xn) 2 and x n+1 = x n − γ · sign(∇f (x n )).

Table 9 : AUC scores of the first two base detectors under cross-norm and cross-perturbation attacks.

L ∞ based attacks use steps 200 and step-size 0.01, and L 2 based attacks uses steps 200 and step-size 0.1.

Table 10 : AUC scores of the first MNIST base detector under fixed start and multiple random restarts attacks.

These two tests use the same attack configuration: the L ∞ = 0.5 trained base detector is attacked using L ∞ = 0.5 constrained PGD attack of steps 200 and step-size 0.01, and the L 2 = 5.0 trained base detector is attacked using L 2 = 5.0 constrained PGD attack of steps 200 and step-size 0.1.

Table 11 : AUC scores of all L ∞ = 0.3 trained base detectors.

Tested with L ∞ = 0.3 constrained PGD attacks of steps 200 and step-size 0.01.

Table 12 : AUC scores of all L ∞ = 0.5 trained base detectors.

Tested with L ∞ = 0.5 constrained PGD attacks of steps 200 and step-size 0.01.

Table 13 : AUC scores of the first CIFAR10 base detector under fixed start and multiple random restarts attacks.

The L ∞ = 2.0 base detector is attacked using PGD attack of steps 10 and stepsize 0.5, and the L ∞ = 8.0 base detector is attacked using PGD attack of steps 40 and step-size 0.5.

Table 16 : AUC scores of L ∞ = 2.0 trained base detectors under L ∞ = 2.0 constrained PGD attack of steps 10 and step-size 0.5.

Table 17 : AUC scores of L ∞ = 8.0 trained base detectors under L ∞ = 8.0 constrained PGD attack of steps 40 and step-size 0.5.

We found training with adversarial examples optimized with a sufficiently small step-size to be essential for detection robustness.

In table 18 we tested two L ∞ = 2.0 base detectors respectively trained with 0.5 and 1.0 step-size.

The step-size 1.0 model is not robust when tested with a much smaller step-size.

We observe that when training the step-size 1.0 model, training set adv AUC reached 1.0 in less than one hundred iterations, but test set natural AUC plummeted to around 0.95 and couldn't recover thereafter.

(Please refer to Figure 16 for the definitions of adv AUC and nat AUC.)

This suggests that naturally occurring data samples, and adversarial examples produced using a large step-size, live in two quite different data spaces -training a classifier to separate these two kinds of data is easy, but the performance won't generalize to real attacks.

While Madry et al.

(2017) was able to train their CIFAR10 robust classifier using step-size 2, we found this step-size not working in our case.

To study the effects of perturbation limit on asymmetrical adversarial training, we compare one L ∞ = 2.0 trained and one L ∞ = 8.0 trained base detector.

In Figure 16 we show the training and testing history of these two models.

The = 2.0 model history show that by adversarial finetuning the model reach robustness in just a few thousands of iterations, and the performance on natural samples is preserved (test natural AUC begins at 0.9971, and ends at 0.9981).

Adversarial finetuning on the = 8.0 model didn't converge after an extended 20K iterations of training.

The gap between train adv AUC and test adv AUC of the = 8.0 model is more pronounced, and we observed a decrease of test natural AUC from 0.9971 to 0.9909.

The comparison shows that training with larger perturbation limit is more time and resource consuming, and could lead to performance decrease on natural samples.

The benefit is that the model learns more interpretable features.

In Figure 17 , perturbations generated by attacking the naturally trained classifier (corresponds to 0 perturbation limit) don't have clear semantic.

In contrast, perturbed samples of the L ∞ = 8 model are completely recognizable.

Figure 18: ImageNet 224×224×3 random samples generated from class-conditional Gaussian noise by attacking robust classifier and detector models trained with different constrains.

Note than large perturbation models are still under training and haven't reached robustness.

Please refer to Santurkar et al. (2019) for the detail about how the class-conditional Gaussian is estimated.

In this section we use Gaussian noise attack experiment to motivate a discussion about the interpretabilities of our generative classification model and discriminative robust classification model (Madry et al., 2017) .

Our approach is more interpretable in the sense that it provides a probabilitic view of the decision making process of the classification problem, and this probabilistic interpolation is further supported by the experimental results.

We first discuss how these two approaches determine the posterior class probabilities.

For the discriminative classifier, the posterior probabilities are computed from the logit outputs of the classifier using the softmax function p(k|x) = exp(z(f (x)) k ) K j=1 exp(z(f (x))j )

.

For our generative classifier, the posterior probabilities are computed in two steps: in the first, we train our base detectors, which is the process of solving the inference problem of determining the joint probability p(x, k) (see Appendix A), and in the second, we use Bayes rule to compute the posterior probability p(k|x) = .

Coincidentally, the formulas for computing the posterior probabilities take the same form.

But in our approach, the exponential of the logit output of a detector (i.e., exp(z(h k (x)))) has a clear probabilistic interpretation: it's the unnormalized joint probability of the input and the corresponding class category.

We use Gaussian noise attack to demonstrate that this probabilistic interpretation is consistent with our visual perception.

We start from a Gaussian noise image, and gradually perturb it to cause higher and higher logit outputs.

This is implemented by targeted PGD attack against logit outputs of these two classification models.

The resulting images in Figure 24 show that, for our model, the logit output increase direction is the semantic changing direction; while for the discriminative robust model, the perturbed image computed by increasing logit outputs are not as clearly interpretable.

In particular, the perturbed images that cause high logit outputs of the robust classifiers are not recognizable.

In this section we provide an analysis of the computational cost of our generative classification approach.

In terms of memory requirements, if we assume the softmax classifier (i.e., the discriminative robust classifier) and the detectors use the same architecture (i.e., only defer in the final layer) then the detector-based generative classifier is approximately K times more expensive than the Kclass softmax classifier.

This also means that the computational graph of the generative classifier is K times larger than the softmax classifier.

Indeed, in the CIFAR10 task, on our Quadro M6000 24GB GPU (TensorFlow 1.13.1), the inference speed of the generative classifier is roughly ten times slower than the softmax classifier.

We next benchmark the training speed of these two types of classifiers.

The generative classifier has K logit outputs, with each one defined by the logit output of a detector.

Same with the softmax classifier, except that the K outputs share the parameters in the convolutional part.

Now consider ordinary adversarial training on the softmax classifier and asymmetrical adversarial training on the generative classifier.

To train the softmax classifier, we use batches of N samples.

For the generative classifier, we train each detector with batches of 2 × M samples (M positive samples and M negative samples).

At each iteration, we need to respectively compute N and M × K adversarial examples for these two classifiers.

Now we test the speed of the following two scenarios: 1) compute the gradient w.r.t.

to N samples on a single computational graph, and 2) compute the gradient w.r.t to M × K samples on K computational graphs, with each graph working on M samples.

We assume in scenario 2 that all the computational graphs are loaded to GPUs, and their computations are in parallel.

In our CIFAR10 experiment, we used batches consisting of 30 positive samples and 30 negative samples to train each ResNet50 based detectors.

In Madry et al. (2017) , the softmax classifier was trained with batches of 128 samples.

In this case, K = 10, M = 30, and N = 128.

On our GPU, scenario 1 took 683 ms ± 6.76 ms per loop, while scenario 2 took 1.85 s ± 42.7 ms per loop.

In this case, we could expect asymmetrical adversarial training to be about 2.7 times slower than ordinary adversarial training, if not considering parameter gradient computation.

(If we choose to use a large batch size the computational cost will increase accordingly.)

<|TLDR|>

@highlight

A new generative modeling technique based on asymmetrical adversarial training, and its applications to adversarial example detection and robust classification