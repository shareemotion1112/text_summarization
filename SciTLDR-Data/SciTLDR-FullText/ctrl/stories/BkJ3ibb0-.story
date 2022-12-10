In recent years, deep neural network approaches have been widely adopted for machine learning tasks, including classification.

However, they were shown to be vulnerable to adversarial perturbations: carefully crafted small perturbations can cause misclassification of legitimate images.

We propose Defense-GAN, a new framework leveraging the expressive capability of generative models to defend deep neural networks against such attacks.

Defense-GAN is trained to model the distribution of unperturbed images.

At inference time, it finds a close output to a given image which does not contain the adversarial changes.

This output is then fed to the classifier.

Our proposed method can be used with any classification model and does not modify the classifier structure or training procedure.

It can also be used as a defense against any attack as it does not assume knowledge of the process for generating the adversarial examples.

We empirically show that Defense-GAN is consistently effective against different attack methods and improves on existing defense strategies.

Despite their outstanding performance on several machine learning tasks, deep neural networks have been shown to be susceptible to adversarial attacks BID20 BID4 .

These attacks come in the form of adversarial examples: carefully crafted perturbations added to a legitimate input sample.

In the context of classification, these perturbations cause the legitimate sample to be misclassified at inference time BID20 BID4 BID16 BID11 .

Such perturbations are often small in magnitude and do not affect human recognition but can drastically change the output of the classifier.

Recent literature has considered two types of threat models: black-box and white-box attacks.

Under the black-box attack model, the attacker does not have access to the classification model parameters; whereas in the white-box attack model, the attacker has complete access to the model architecture and parameters, including potential defense mechanisms BID21 BID2 .Various defenses have been proposed to mitigate the effect of adversarial attacks.

These defenses can be grouped under three different approaches: (1) modifying the training data to make the classifier more robust against attacks, e.g., adversarial training which augments the training data of the classifier with adversarial examples BID20 BID4 , (2) modifying the training procedure of the classifier to reduce the magnitude of gradients, e.g., defensive distillation BID18 , and (3) attempting to remove the adversarial noise from the input samples BID6 BID13 .

All of these approaches have limitations in the sense that they are effective against either white-box attacks or black-box attacks, but not both BID21 BID13 .

Furthermore, some of these defenses are devised with specific attack models in mind and are not effective against new attacks.

In this paper, we propose a novel defense mechanism which is effective against both white-box and black-box attacks.

We propose to leverage the representative power of Generative Adversarial Networks (GAN) to diminish the effect of the adversarial perturbation, by "projecting" input images onto the range of the GAN's generator prior to feeding them to the classifier.

In the GAN framework, two models are trained simultaneously in an adversarial setting: a generative model that emulates the data distribution, and a discriminative model that predicts whether a certain input came from real data or was artificially created.

The generative model learns a mapping G from a low-dimensional vector z ∈ R k to the high-dimensional input sample space R n .

During training of the GAN, G is encouraged to generate samples which resemble the training data.

It is, therefore, expected that legitimate samples will be close to some point in the range of G, whereas adversarial samples will be further away from the range of G. Furthermore, "projecting" the adversarial examples onto the range of the generator G can have the desirable effect of reducing the adversarial perturbation.

The projected output, computed using Gradient Descent (GD), is fed into the classifier instead of the original (potentially adversarially modified) image.

We empirically demonstrate that this is an effective defense against both black-box and white-box attacks on two benchmark image datasets.

The rest of the paper is organized as follows.

We introduce the necessary background regarding known attack models, defense mechanisms, and GANs in Section 2.

Our defense mechanism, which we call Defense-GAN, is formally motivated and introduced in Section 3.

Finally, experimental results, under different threat models, as well as comparisons to other defenses are presented in Section 4.

In this work, we propose to use GANs for the purpose of defending against adversarial attacks in classification problems.

Before detailing our approach in the next section, we explain related work in three parts.

First, we discuss different attack models employed in the literature.

We, then, go over related defense mechanisms against these attacks and discuss their strengths and shortcomings.

Lastly, we explain necessary background information regarding GANs.

Various attack models and algorithms have been used to target classifiers.

All attack models we consider aim to find a perturbation δ to be added to a (legitimate) input x ∈ R n , resulting in the adversarial examplex = x + δ.

The ∞ -norm of the perturbation is denoted by BID4 and is chosen to be small enough so as to remain undetectable.

We consider two threat levels: black-and white-box attacks.

White-box models assume that the attacker has complete knowledge of all the classifier parameters, i.e., network architecture and weights, as well as the details of any defense mechanism.

Given an input image x and its associated ground-truth label y, the attacker thus has access to the loss function J(x, y) used to train the network, and uses it to compute the adversarial perturbation δ.

Attacks can be targeted, in that they attempt to cause the perturbed image to be misclassified to a specific target class, or untargeted when no target class is specified.

In this work, we focus on untargeted white-box attacks computed using the Fast Gradient Sign Method (FGSM) BID4 , the Randomized Fast Gradient Sign Method (RAND+FGSM) BID21 , and the Carlini-Wagner (CW) attack BID2 .

Although other attack models exist, such as the Iterative FGSM , the Jacobian-based Saliency Map Attack (JSMA) BID16 , and Deepfool (MoosaviDezfooli et al., 2016) , we focus on these three models as they cover a good breadth of attack algorthims.

FGSM is a very simple and fast attack algorithm which makes it extremely amenable to real-time attack deployment.

On the other hand, RAND+FGSM, an equally simple attack, increases the power of FGSM for white-box attacks BID21 , and finally, the CW attack is one of the most powerful white-box attacks to-date BID2 .Fast Gradient Sign Method (FGSM) Given an image x and its corresponding true label y, the FGSM attack sets the perturbation δ to: DISPLAYFORM0 (1)FGSM BID4 was designed to be extremely fast rather than optimal.

It simply uses the sign of the gradient at every pixel to determine the direction with which to change the corresponding pixel value.

Randomized Fast Gradient Sign Method (RAND+FGSM) The RAND+FGSM BID21 attack is a simple yet effective method to increase the power of FGSM against models which were adversarially trained.

The idea is to first apply a small random perturbation before using FGSM.

More explicitly, for α < , random noise is first added to the legitimate image x: DISPLAYFORM1 Then, the FGSM attack is computed on x , resulting iñ DISPLAYFORM2 The Carlini-Wagner (CW) attack The CW attack is an effective optimization-based attack model BID2 .

In many cases, it can reduce the classifier accuracy to almost 0% BID2 BID13 .

The perturbation δ is found by solving an optimization problem of the form: DISPLAYFORM3 where f is an objective function that drives the example x to be misclassified, and c > 0 is a suitably chosen constant.

The 2 , 0 , and ∞ norms are considered.

We refer the reader to BID2 for details regarding the approach to solving (4) and setting the constant c.

For black-box attacks we consider untargeted FGSM attacks computed on a substitute model .

As previously mentioned, black-box adversaries have no access to the classifier or defense parameters.

It is further assumed that they do not have access to a large training dataset but can query the targeted DNN as a black-box, i.e., access labels produced by the classifier for specific query images.

The adversary trains a model, called substitute, which has a (potentially) different architecture than the targeted classifier, using a very small dataset augmented by synthetic images labeled by querying the classifier.

Adversarial examples are then found by applying any attack method on the substitute network.

It was found that such examples designed to fool the substitute often end up being misclassified by the targeted classifier BID20 .

In other words, black-box attacks are easily transferrable from one model to the other.

Various defense mechanisms have been employed to combat the threat from adversarial attacks.

In what follows, we describe one representative defense strategy from each of the three general groups of defenses.

A popular approach to defend against adversarial noise is to augment the training dataset with adversarial examples BID20 BID4 BID14 .

Adversarial examples are generated using one or more chosen attack models and added to the training set.

This often results in increased robustness when the attack model used to generate the augmented training set is the same as that used by the attacker.

However, adversarial training does not perform as well when a different attack strategy is used by the attacker.

Additionally, it tends to make the model more robust to white-box attacks than to black-box attacks due to gradient masking BID17 BID21 .

Defensive distillation BID18 trains the classifier in two rounds using a variant of the distillation BID7 method.

This has the desirable effect of learning a smoother network and reducing the amplitude of gradients around input points, making it difficult for attackers to generate adversarial examples BID18 .

It was, however, shown that, while defensive distillation is effective against white-box attacks, it fails to adequately protect against black-box attacks transferred from other networks BID2 .

Recently, BID13 introduced MagNet as an effective defense strategy.

It trains a reformer network (which is an auto-encoder or a collection of auto-encoders) to move adversarial examples closer to the manifold of legitimate, or natural, examples.

When using a collection of auto-encoders, one reformer network is chosen at random at test time, thus strengthening the defense.

It was shown to be an effective defense against gray-box attacks where the attacker knows everything about the network and defense, except the parameters.

MagNet is the closest defense to our approach, as it attempts to reform an adversarial sample using a learnt auto-encoder.

The main differences between MagNet and our approach are: (1) we use GANs instead of auto-encoders, and, most importantly, (2) we use GD minimization to find latent codes as opposed to a feedforward encoder network.

This makes Defense-GAN more robust, especially against white-box attacks.

GANs, originally introduced by , consist of two neural networks, G and D. G : R k → R n maps a low-dimensional latent space to the high dimensional sample space of x. D is a binary neural network classifier.

In the training phase, G and D are typically learned in an adversarial fashion using actual input data samples x and random vectors z. An isotropic Gaussian prior is usually assumed on z. While G learns to generate outputs G(z) that have a distribution similar to that of x, D learns to discriminate between "real" samples x and "fake" samples G(z).

D and G are trained in an alternating fashion to minimize the following min-max loss : DISPLAYFORM0 It was shown that the optimal GAN is obtained when the resulting generator distribution p g = p data .However, GANs turned out to be difficult to train in practice BID5 , and alternative formulations have been proposed.

introduced Wasserstein GANs (WGANs) which are a variant of GANs that use the Wasserstein distance, resulting in a loss function with more desirable properties: DISPLAYFORM1 In this work, we use WGANs as our generative model due to the stability of their training methods, especially using the approach in BID5 .3 PROPOSED DEFENSE-GAN Figure 1 : Overview of the Defense-GAN algorithm.

As mentioned in Section 2.3, the GAN min-max loss in (5) admits a global optimum when p g = p data .

It can be similarly shown that WGAN admits an optimum to its own minmax loss in (6), when the set {x | p g (x) = p data (x)} has zero Lebesgue-measure.

Formally, Lemma 1 A generator distribution p g is a global optimum for the WGAN min-max game defined in (6), if and only if p g (x) = p data (x) for all x ∈ R n , potentially except on a set of zero Lebesguemeasure.

A sketch of the proof can be found in Appendix A.Additionally, it was shown that, if G and D have enough capacity to represent the data, and if the training algorithm is such that p g converges to p data , then DISPLAYFORM0 where G t is the generator of a GAN or WGAN 1 after t steps of its training algorithm BID8 .

This serves to show that, under ideal conditions, the addition of the GAN reconstruction loss minimization step should not affect the performance of the classifier on natural, legitimate samples, as such samples should be almost exactly recovered.

Furthermore, we hypothesize that this step will help reduce the adversarial noise which follows a different distribution than that of the GAN training examples.

Defense-GAN is a defense strategy to combat both white-box and black-box adversarial attacks against classification networks.

At inference time, given a trained GAN generator G and an image x to be classified, z * is first found so as to minimize DISPLAYFORM0 G(z * ) is then given as the input to the classifier.

The algorithm is illustrated in Figure 1 .

As FORMULA7 is a highly non-convex minimization problem, we approximate it by doing a fixed number L of GD steps using R different random initializations of z (which we call random restarts), as shown in Figures 1 and 2.The GAN is trained on the available classifier training dataset in an unsupervised manner.

The classifier can be trained on the original training images, their reconstructions using the generator G, or a combination of the two.

As was discussed in Section 3.1, as long as the GAN is appropriately trained and has enough capacity to represent the data, original clean images and their reconstructions should not defer much.

Therefore, these two classifier training strategies should, at least theoretically, not differ in performance.

Compared to existing defense mechanisms, our approach is different in the following aspects: 1.

Defense-GAN can be used in conjunction with any classifier and does not modify the classifier structure itself.

It can be seen as an add-on or pre-processing step prior to classification.2.

If the GAN is representative enough, re-training the classifier should not be necessary and any drop in performance due to the addition of Defense-GAN should not be significant.3.

Defense-GAN can be used as a defense to any attack: it does not assume an attack model, but simply leverages the generative power of GANs to reconstruct adversarial examples.4.

Defense-GAN is highly non-linear and white-box gradient-based attacks will be difficult to perform due to the GD loop.

A detailed discussion about this can be found in Appendix B.

We assume three different attack threat levels:1.

Black-box attacks: the attacker does not have access to the details of the classifier and defense strategy.

It therefore trains a substitute network to find adversarial examples.2.

White-box attacks: the attacker knows all the details of the classifier and defense strategy.

It can compute gradients on the classifier and defense networks in order to find adversarial examples.3.

White-box attacks, revisited: in addition to the details of the architectures and parameters of the classifier and defense, the attacker has access to the random seed and random number generator.

In the case of Defense-GAN, this means that the attacker knows all the random initializations {z DISPLAYFORM0 We compare our method to adversarial training BID4 and MagNet BID13 under the FGSM, RAND+FGSM, and CW (with 2 norm) white-box attacks, as well as the FGSM black-box attack.

Details of all network architectures used in this paper can be found in Appendix C. When the classifier is trained using the reconstructed images (G(z * )), we refer to our method as Defense-GAN-Rec, and we use Defense-GAN-Orig when the original images (x) are used to train the classifier.

Our GAN follows the WGAN training procedure in BID5 , and details of the generator and discriminator network architectures are given in TAB4 .

The reformer network (encoder) for the MagNet baseline is provided in TAB5 .

Our implementation is based on TensorFlow BID0 and builds on open-source software: CleverHans by BID15 and improved WGAN training by BID5 .

We use machines equipped with NVIDIA GeForce GTX TITAN X GPUs.

In our experiments, we use two different image datasets: the MNIST handwritten digits dataset BID10 and the Fashion-MNIST (F-MNIST) clothing articles dataset BID22 .

Both datasets consist of 60, 000 training images and 10, 000 testing images.

We split the training images into a training set of 50, 000 images and hold-out a validation set containing 10, 000 images.

For white-box attacks, the testing set is kept the same (10, 000 samples).

For black-box attacks, the testing set is divided into a small hold-out set of 150 samples reserved for adversary substitute training, as was done in , and the remaining 9, 850 samples are used for testing the different methods.

In this section, we present experimental results on FGSM black-box attacks.

As previously mentioned, the attacker trains a substitute model, which could differ in architecture from the targeted model, using a limited dataset consisting of 150 legitimate images augmented with synthetic images labeled using the target classifier.

The classifier and substitute model architectures used and referred to throughout this section are described in TAB3 in the Appendix.

In TAB0 , we present our classification accuracy results and compare to other defense methods.

As can be seen, FGSM black-box attacks were successful at reducing the classifier accuracy by up to 70%.

All considered defense mechanisms are relatively successful at diminishing the effect of the attacks.

We note that, as expected, the performance of Defense-GAN-Rec and that of Defense-GAN-Orig are very close.

In addition, they both perform consistently well across different classifier and substitute model combinations.

MagNet also performs in a consistent manner, but achieves lower accuracy than Defense-GAN.

Two adversarial training defenses are presented: the first one obtains the adversarial examples assuming the same attack = 0.3, and the second assumes a different = 0.15.

With incorrect knowledge of , the performance of adversarial training generally decreases.

In addition, the classification performance of this defense method has very large variance across the different architectures.

It is worth noting that adversarial training defense is only fit against FGSM attacks, because the adversarially augmented data, even with a different , is generated using the same method as the black-box attack (FGSM).

In contrast, Defense-GAN and MagNet are general defense mechanisms which do not assume a specific attack model.

The performances of defenses on the F-MNIST dataset, shown in TAB1 , are noticeably lower than on MNIST.

This is due to the large = 0.3 in the FGSM attack.

Please see Appendix D for qualitative examples showing that = 0.3 represents very high noise, which makes F-MNIST images difficult to classify, even by a human.

In addition, the Defense-GAN parameters used in this experiment were kept the same for both Tables, in order to study the effect of dataset complexity, and can be further optimized as investigated in the next section.

FIG1 shows the effect of varying the number of GD iterations L as well as the random restarts R used to compute the GAN reconstructions of input images.

Across different L and R values, Defense-GAN-Rec and Defense-GAN-Orig have comparable performance.

Increasing L has the expected effect of improving performance when no attack is present.

Interestingly, with an FGSM attack, the classification performance decreases after a certain L value.

With too many GD iterations on the mean squared error (MSE) ||G(z) − (x + δ)|| 2 2 , some of the adversarial noise components are retained.

In the right Figure, the effect of varying R is shown to be extremely pronounced.

This is due to the non-convex nature of the MSE, and increasing R enables us to sample different local minima.

We now investigate the effect of changing the attack in Table 3 .

As expected, with higher , the FGSM attack is more successful, especially on the F-MNIST dataset where the noise norm seems to have a more pronounced effect with nearly 37% drop in performance between = 0.1 and 0.3.

Figure 7 in Appendix D shows adversarial samples as well as their reconstructions with Defense-GAN at different values of .

We can see that for large , the class is difficult to discern, even for the human eye.

Even though it seems that increasing is a desirable strategy for the attacker, this increases the likelihood that the adversarial noise is discernible and therefore the attack is detected.

It is trivial for the attacker to provide adversarial images at very high , and a good measure of an attack's strength is its ability to affect performance at low .

In fact, in the next section, we discuss how Defense-GAN can be used to not only diminish the effect of attacks, but to also detect them.

We intuitively expect that clean, unperturbed images will lie closer to the range of the Defense-GAN generator G than adversarial examples.

This is due to the fact that G was trained to produce images which resemble the legitimate data.

In light of this observation, we propose to use the MSE of an image with it is reconstruction from (8) as a "metric" to decide whether or not the image was Table 3 : Classification accuracy of Model F using Defense-GAN (L = 400, R = 10), under FGSM black-box attacks for various noise norms and substitute Model E. adversarially manipulated.

In order words, for a given threshold θ > 0, the hypothesis test is:

We compute the reconstruction MSEs for every image from the test dataset, and its adversarially manipulated version using FGSM.

We show the Receiver Operating Characteristic (ROC) curves as well as the Area Under the Curve (AUC) metric for different Defense-GAN parameters and values in FIG2 .

The results show that this attack detection strategy is effective especially when the number of GD iterations L and random restarts R are large.

From the left and middle Figures, we can conclude that the number of random restarts plays a very important role in the detection false positive and true positive rates as was discussed in Section 4.1.1.

Furthermore, when is very small, it becomes difficult to detect attacks at low false positive rates.

We now present results on white-box attacks using three different strategies: FGSM, RAND+FGSM, and CW.

We perform the CW attack for 100 iterations of projected GD, with learning rate 10.0, and use c = 100 in equation FORMULA3 .

Table 4 shows the classification performance of different classifier models across different attack and defense strategies.

We note that Defense-GAN significantly outperforms the two other baseline defenses.

We even give the adversarial attacker access to the random initializations of z. However, we noticed that the performance does not change much when the attacker does not know the initialization.

Adversarial training was done using FGSM to generate the adversarial samples.

It is interesting to mention that when CW attack is used, adversarial training performs extremely poorly.

As previously discussed, adversarial training does not generalize well against different attack methods.

Due to the loop of L steps of GD, Defense-GAN is resilient to GD-based white-box attacks, since the attacker needs to "un-roll" the GD loop and propagate the gradient of the loss all the way across L steps.

In fact, from Table 4 L = 25, the performance of the same network drops to 0.947 (more than 5% drop).

This shows that using a larger L significantly increases the robustness of Defense-GAN against GD-based whitebox attacks.

This comes at the expense of increased inference time complexity.

We present a more detailed discussion about the difficulty of GD-based white-box attacks in Appendix B and time complexity in Appendix G. Additional white-box experimental results on higher-dimensional images are reported in Appendix F. Table 4 : Classification accuracies of different classifier models using various defense strategies on the MNIST (top) and F-MNIST (bottom) datasets, under FGSM, RAND+FGSM, and CW white-box attacks.

Defense-GAN has L = 200 and R = 10.

Classifier

In this paper, we proposed Defense-GAN, a novel defense strategy utilizing GANs to enhance the robustness of classification models against black-box and white-box adversarial attacks.

Our method does not assume a particular attack model and was shown to be effective against most commonly considered attack strategies.

We empirically show that Defense-GAN consistently provides adequate defense on two benchmark computer vision datasets, whereas other methods had many shortcomings on at least one type of attack.

It is worth mentioning that, although Defense-GAN was shown to be a feasible defense mechanism against adversarial attacks, one might come across practical difficulties while implementing and deploying this method.

The success of Defense-GAN relies on the expressiveness and generative power of the GAN.

However, training GANs is still a challenging task and an active area of research, and if the GAN is not properly trained and tuned, the performance of Defense-GAN will suffer on both original and adversarial examples.

Moreover, the choice of hyper-parameters L and R is also critical to the effectiveness of the defense and it may be challenging to tune them without knowledge of the attack.

A OPTIMALITY OF p g = p DATA FOR WGANS Sketch of proof of Lemma 1: The WGAN min-max loss is given by: DISPLAYFORM0 DISPLAYFORM1 For a fixed G, the optimal discriminator D which maximizes V W (D, G) is such that: DISPLAYFORM2 Plugging D * G back into (12), we get: DISPLAYFORM3 }.

Clearly, to minimize (15), we need to set p data (x) = p g (x) for x ∈ X .

Then, since both pdfs should integrate to 1, DISPLAYFORM4 However, this is a contradiction since p g (x) < p data (x) for x ∈ X c , unless µ(X c ) = 0 where µ is the Lebesgue measure.

This concludes the proof.

In order to perform a GD-based white-box attack on models using Defense-GAN, an attacker needs to compute the gradient of the output of the classifier with respect to the input.

From Figure 1 , the generator and the classifier can be seen as one, combined, feedforward network, through which it is easy to propagate gradients.

The difficulty lies in the orange box of the GD optimization detailed in FIG0 .For the sake of simplicity, let's assume that R = 1.

Define L(x, z) = ||G(z) − x|| 2 2 .

Then z * = z L , which is computed recursively as follows: DISPLAYFORM0 and so on.

Therefore, computing the gradient of z * with respect to x involves a large number (L) of recursive chain rules and high-dimensional Jacobian tensors.

This computation gets increasingly prohibitive for large L.

We describe the neural network architectures used throughout the paper.

The detail of models A through F used for classifier and substitute networks can be found in TAB3 .

In TAB4 , the GAN architectures are described, and in TAB5 , the encoder architecture for the MagNet baseline is given.

In what follows:• Conv(m, k × k, s) refers to a convolutional layer with m feature maps, filter size k × k, and stride s• ConvT(m, k × k) refers to the transpose (gradient) of Conv (sometimes referred to as "deconvolution") with m feature maps, filter size k × k, and stride s• FC(m) refers to a fully-connected layer with m outputs• Dropout(p) refers to a dropout layer with probability p• ReLU refers to the Rectified Linear Unit activation• LeakyReLU(α) is the leaky version of the Rectified Linear Unit with parameter α Generator Discriminator DISPLAYFORM0

We report results on white-box attacks on the CelebFaces Attributes dataset (CelebA) BID12 in TAB0 .

The CelebA dataset is a large-scale face dataset consisting of more than 200, 000 face images, split into training, validation, and testing sets.

The RGB images were center-cropped and resized to 64 × 64.

We performed the task of gender classification on this dataset.

The GAN architecture is the same as that in TAB4 , except for an additional ConvT(128, 5 × 5, 1) layer in the generator network.

G TIME COMPLEXITYThe computational complexity of reconstructing an image using Defense-GAN is on the order of the number of GD iterations performed to estimate z * , multiplied by the time to compute gradients.

The number of random restarts R has less effect on the running time, since random restarts are independent and can run in parallel if enough resources are available.

TAB0 shows the average running time, in seconds, to find the reconstructions of MNIST and F-MNIST images on one NVIDIA GeForce GTX TITAN X GPU.

For most applications, these running times are not prohibitive.

We can see a tradeoff between running time and defense robustness as well as accuracy.

<|TLDR|>

@highlight

Defense-GAN uses a Generative Adversarial Network to defend against white-box and black-box attacks in classification models.