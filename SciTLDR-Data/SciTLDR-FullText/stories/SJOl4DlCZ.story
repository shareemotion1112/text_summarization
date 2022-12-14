Suppose a deep classification model is trained with samples that need to be kept private for privacy or confidentiality reasons.

In this setting, can an adversary obtain the private samples if the classification model is given to the adversary?

We call this reverse engineering against the classification model the Classifier-to-Generator (C2G) Attack.

This situation arises when the classification model is embedded into mobile devices for offline prediction (e.g., object recognition for the automatic driving car and face recognition for mobile phone authentication).

For C2G attack, we introduce a novel GAN, PreImageGAN.

In PreImageGAN, the generator is designed to estimate the the sample distribution conditioned by the preimage of classification model $f$, $P(X|f(X)=y)$, where $X$ is the random variable on the sample space and $y$ is the probability vector representing the target label arbitrary specified by the adversary.

In experiments, we demonstrate PreImageGAN works successfully with hand-written character recognition and face recognition.

In character recognition, we show that, given a recognition model of hand-written digits, PreImageGAN allows the adversary to extract alphabet letter images without knowing that the model is built for alphabet letter images.

In face recognition, we show that, when an adversary obtains a face recognition model for a set of individuals, PreImageGAN allows the adversary to extract face images of specific individuals contained in the set, even when the adversary has no knowledge of the face of the individuals.

Recent rapid advances in deep learning technologies are expected to promote the application of deep learning to online services with recognition of complex objects.

Let us consider the face recognition task as an example.

The probabilistic classification model f takes a face image x and the model predicts the probability of which the given face image is associated with an individual t, f (x) ≃ Pr[T = t|X = x].The following three scenarios pose situations that probabilistic classification models need to revealed in public for online services in real applications:Prediction with cloud environment: Suppose an enterprise provides an online prediction service with a cloud environment, in which the service takes input from a user and returns predictions to the user in an online manner.

The enterprise needs to deploy the model f into the cloud to achieve this.

Prediction with private information: Suppose an enterprise develops a prediction model f (e.g., disease risk prediction) and a user wishes to have a prediction of the model with private input (e.g., personal genetic information).

The most straightforward way to preserve the user's privacy entirely is to let the user download the entire model and perform prediction on the user side locally.

Offline prediction: Automatic driving cars or laptops with face authentication contain face/object recognition systems in the device.

Since these devices are for mobile use and need to work standalone, the full model f needs to be embedded in the device.

In such situations that classification model f is revealed, we consider a reverse-engineering problem of models with deep architectures.

Let D tr and d X,T be a set of training samples and its underlying distribution, respectively.

Let f be a model trained with D tr .

In this situation, is it possible for an adversary to obtain the training samples D tr (or its underlying distribution d X,T ) if the classification model is given to the adversary?.

If this is possible, this can cause serious problems, particularly when D tr or d X,T is private or confidential information.

Privacy violation by releasing face authentication: Let us consider the face authentication task as an example again.

Suppose an adversary is given the classification model f .

The adversary aims to estimate the data (face) distribution of a target individual t * , d X|T =t * .

If this kind of reverseengineering works successfully, serious privacy violation arises because individual faces are private information.

Furthermore, once d X|T =t * is revealed, the adversary can draw samples from d X|T =t * , which would cause another privacy violation (say, the adversary can draw an arbitrary number of the target's face images).Confidential information leakage by releasing object recognizer: Let us consider an object recognition system for automatic driving cars.

Suppose a model f takes as input images from car-mounted cameras and detect various objects such as traffic signs or traffic lights.

Given f , the reverse engineering reveals the sample distribution of the training samples, which might help adversaries having malicious intentions.

For example, generation of adversarial examples that make the recognition system confuse without being detected would be possible.

Also, this kind of attack allows exposure of hidden functionalities for privileged users or unexpected vulnerabilities of the system.

If this kind of attack is possible, it indicates that careful treatment is needed before releasing model f in public considering that publication of f might cause serious problems as listed above.

We name this type of reverse engineering classifier-to-generator (C2G) attack .

In principle, estimation of labeled sample distributions from a classification/recognition model of complex objects (e.g., face images) is a difficult task because of the following two reasons.

First, estimation of generative models of complex objects is believed to be a challenging problem itself.

Second, model f often does not contain sufficient information to estimate the generative model of samples.

In supervised classification, the label space is always much more abstract than the sample space.

The classification model thus makes use of only a limited amount of information in the sample space that is sufficient to classify objects into the abstract label space.

In this sense, it is difficult to estimate the sample distribution given only classification model f .To resolve the first difficulty, we employ Generative Adversarial Networks (GANs).

GANs are a neural network architecture for generative models which has developed dramatically in the field of deep learning.

Also, we exploit one remarkable property of GANs, the ability to interpolate latent variables of inputs.

With this interpolation, GANs can generate samples (say, images) that are not included in the training samples, but realistic samples 1 .Even with this powerful generation ability of GANs, it is difficult to resolve the second difficulty.

To overcome this for the C2G attack, we assume that the adversary can make use of unlabeled auxiliary samples D aux as background knowledge.

Suppose f be a face recognition model that recognizes Alice and Bob, and the adversary tries to extract Alice's face image from f .

It is natural to suppose that the adversary can use public face image samples that do not contain Alice's and Bob's face images as D aux .

PreImageGAN exploits unlabeled auxiliary samples to complement knowledge extracted from the model f .

The contribution of this study is summarized as follows.• We formulate the Classifier-to-Generator (C2G) Attack, which estimates the training sample distribution when a classification model and auxiliary samples are given(Section 3) • We propose PreImageGAN as an algorithm for the C2G attack.

The proposed method estimates the sample generation model using the interpolation ability of GANs even when the auxiliary samples used by the adversary is not drawn from the same distribution as the training sample distribution (Section 4)1 Radford et al. (2015) reported GANs could generate intermediate images between two different images.

Also, Radford et al. (2015) realizes the operation of latent vectors.

For example, by subtracting a latent vector of a man's face from a face image of a man wearing glasses, and then adding a latent vector of a female's face, then the GAN can generate the woman's face image wearing glasses.•

We demonstrate the performance of C2G attack with PreImageGAN using EMNIST (alphanumeric image dataset) and FaceScrub (face image dataset).

Experimental results show that the adversary can estimate the sample distribution even when the adversary has no samples associated with the target label at all (Section 5)

Generative Adversarial Networks (GANs) is a recently developed methodology for designing generative models proposed by BID6 .

Given a set of samples, GANs is an algorithm with deep architectures that estimates the sample-generating distribution.

One significant property of GANs is that it is expected to be able to accurately estimate the sample distribution even when the sample space is in the high dimensional space, and the target distribution is highly complex, such as face images or natural images.

In this section, we introduce the basic concept of GANs and its variants.

The learning algorithm of GANs is formulated by minimax games consisting of two players, generator and discriminator BID6 ).

Generator G generates a fake sample G(z) using a random number z ∼ d Z drawn from any distribution (say, uniform distribution).

Discriminator D is a supervised model and is trained so that it outputs 1 if the input is a real sample x ∼ d X drawn from the sample generating distribution d X ; it outputs 0 or −1 if the input is a fake sample G(z).The generator is trained so that the discriminator determines a fake sample as a real sample.

By training the generator under the setting above, we can expect that samples generated from G(z) for arbitrary z are indistinguishable from real samples x ∼ d X .

Letting Z be the random variable of d Z , G(Z) can be regarded as the distribution of samples generated by the generator.

Training of GANs is known to be reduced to optimization of G so that the distribution between G(Z) and the data generating distribution d X is minimized in a certain type of divergence BID6 ).Training of GAN proposed by BID6 (VanillaGAN) is shown to be reduced to minimization e of Jensen Shannon (JS) divergence of G(Z) and d X .

Minimization of JS-divergence often suffers gradient explosion and mode collapse BID6 , ).

To overcome these problems, Wasserstein-GAN (WGAN), GAN that minimizes Wasserstein distance between G(Z) and d X , was proposed , ).

As a method to stabilize convergence behavior of WGAN, a method to add a regularization term called Gradient Penalty (GP) to the loss function of the discriminator was introduced BID8 ).Given a set of labeled samples {(x, c), · · · } where c denotes the label, Auxiliary Classifier GAN (ACGAN) was proposed as a GAN to estimate d X|C=c , sample distribution conditioned by label c BID16 ).

Differently from VanillaGAN, the generator of ACGAN takes as input a random noise z and a label c. Also, the discriminator of ACGAN is trained to predict a label of sample in addition to estimation of real or fake samples.

In the learning process of ACGAN, generator is trained so that discriminator predicts correctly the label of generated sample in addition.

The generator of ACGAN can generate samples with a label specified arbitrarily.

For example, when x corresponds to face images and c corresponds to age or gender, ACGAN can generate images with specifying the age or gender BID13 , BID5 ).

In our proposed algorithm introduced in the latter sections, we employ WGAN and ACGAN as building blocks.3 PROBLEM FORMULATION

We consider a supervised learning setting.

Let T be the label set, and X ⊆ R d be the sample domain where d denotes the sample dimension.

Let ρ t be the distribution of samples in X with label t. In face recognition, x ∈ X and t ∈ T correspond to a (face) image and an individual, respectively.

ρ t thus denotes the distribution of face images of individual t.

We suppose the images contained in the training dataset are associated with a label subset T tr ⊂ T. Then, the training dataset is defined as D tr = {(x, t)|x ∈ X, t ∈ T tr }.

We denote the random Figure 1 : Outline of Classifier-to-Generator (C2G) Attack.

The publisher trains classifier f from training data D tr and publishes f to the adversary.

However, the publisher does not wish to leak training data D tr and sample generating distribution ρ t by publishing f .

The goal of the adversary is to learn the publisher's private distribution ρ t * for any t * ∈ T tr specified by the adversary provided model f , target label t * and (unlabeled) auxiliary samples D aux .variables associated with (x, t) by (X tr , T tr ) Then, the distribution of X tr is given by a mixture distribution DISPLAYFORM0 where ∑ t∈Ttr α t = 1, α t > 0 for all t ∈ T tr .

In the face recognition task example again, a training sample consists of a pair of an individual t and his/her face image x, (x, t) where x ∼ ρ t .Next, we define the probabilistic discrimination model we consider in our problem.

Let Y be a set of |T tr |-dimension probability vector, ∆ |Ttr| .

Given a training dataset D tr , a learning algorithm L gives a probabilistic discrimination model f : DISPLAYFORM1 Here the tth element of the output (f (x)) t of f corresponds to the probability with which x has label t. Letting T tr and X tr represents the random variable of T tr and d Xtr , f is the approximation of Pr[T tr |X tr ].

We define the Classifier-to-Generator Attack (C2G Attack) in this section.

We consider two stakeholders, publisher and adversary in this attack.

The publisher holds training dataset D tr drawn from d Xtr and a learning algorithm L. She trains model f = L(D tr ) and publishes f to the adversary.

We suppose training dataset D tr and data generating distribution ρ t for any t ∈ T tr is private or confidential information of the publisher, and the publisher does not wish to leak them by publishing f .Given f and T tr , the adversary aims to obtain ρ t * for any label t * ∈ T tr specified by the adversary.

We suppose the adversary can make use of an auxiliary dataset D aux drawn from underlying distribution d Xaux as background knowledge.

D aux is a set of samples associated with labels in T aux ⊂ T. We remark that D aux is defined as a set of samples associated with a specific set of labels, however, in our algorithm described in the following sections, we do not require that samples in D aux are labeled.

Then, the underlying distribution d Xaux is defined as follows: DISPLAYFORM0 where DISPLAYFORM1 The richness of the background knowledge can be determined by the relation between T tr and T aux .

When T tr = T aux , d Xtr = d Xaux holds.

That is, the adversary can make use of samples drawn from the distribution that is exactly same as that of the publisher.

In this sense, this setting is the most advantageous to the adversary.

If t * / ∈ T aux , the adversary cannot make use of samples with the target label t * ; this setting is more advantageous to the publisher.

As the overlap between T tr and T aux increases, the situation becomes more advantageous to the adversary.

Discussions on the background knowledge of the adversary are given in 3.4 in detail.

The goal of the adversary is to learn the publisher's private distribution ρ t * for any t * ∈ T tr specified by the adversary provided model f , target label t * and auxiliary (unlabeled) samples D aux .

Let A be the adversary's attack algorithm.

Then, the attack by the adversary can be formulated bŷ DISPLAYFORM2 where the output of A is a distribution over X. In the face recognition example of Alice and Bob again, when the target label of the adversary is t * =Alice, the objective of the adversary is to estimate the distribution of face images of Alice by A(f, D aux , t * ).

The objective of the C2G attack to estimate ρ t * , the private data generating distribution of the publisher.

In principle, the measure of the success of the C2G attack is evaluated with the quasi-distance between the underlying distribution ρ t * and the estimated generative model A(f, D aux , t * ).

If the two distributions are close, we can confirm that the adversary successfully estimates ρ t * .

However, ρ t * is unknown, and we cannot evaluate this quasi-distance directly.

Instead of evaluating the distance of the two distributions directly, we evaluate the attack algorithm empirically.

We first prepare a classifier f ′ that is trained with D tr using a learning algorithm different from f .

We then give samples drawn from A(f, D aux , t * ) to f ′ and evaluate the probability of which the label of the given samples are predicted as t * .

We expect that the classifier f ′ would label samples drawn from A(f, D aux , t * ) as t * with high probability if A(f, D aux , t * ) successfully estimates ρ t * .

Considering the possibility that A(f, D aux , t * ) overfits to f , we employ another classifier f ′ for this evaluation.

This evaluation criterion is the same as the inception accuracy introduced for ACGAN by BID16 .

In our setting, since our objective is to estimate the distribution concerning a specific label t * , we employ the following inception accuracy: DISPLAYFORM0 We remark that the generated model with a high inception accuracy is not always a reasonable estimation of ρ t * .

Discrimination models with a deep architecture are often fooled with artifacts.

For example, BID15 reported that images look like white noise for humans can be classified as a specific object with high probability.

For this reason, we cannot conclude that a model with a high inception accuracy always generates meaningful images.

To avoid this, the quality of generated images should be subjectively checked by humans.

The evaluation criterion of the C2G Attack we employed for this study is similar to those for GANs.

Since the objective of GANs and the C2G attack is to estimate unknown generative models, we cannot employ the pseudo distance between the underlying generating distribution and the estimated distribution.

The evaluation criterion of GANs is still an open problem, and subjective evaluation is needed for evaluation of GANs BID7 ).

In this study, we employ both the inception accuracy and subjective evaluation for performance evaluation of the C2G attack.

The richness of the background knowledge of the adversary affects the performance of the C2G attack significantly.

We consider the following three levels of the background knowledge.

In the following, let T aux be the set of labels of samples generated by the underlying distribution of the auxiliary data, d Xaux .

Also, let T tr be the set of labels of samples generated by the underlying distribution of the training data.• Exact same: T tr = T aux In this setting, we suppose T tr is exactly same as the T aux .

Since D aux follows d Xtr , D aux contains samples with the target label.

That is, the adversary can obtain samples labeled with the target label.

The background knowledge of the adversary in this setting is the most powerful among the three settings.• Partly same:t * / ∈ T aux , T aux ⊂ T tr In this setting, T aux and T tr are overlapping.

However, T aux does not contain the target label.

That is, the adversary cannot obtain samples labeled with the target label.

In this sense, the background knowledge of the adversary in this setting is not as precise as that in the former setting.• Mutually exclusive: T aux ∩ T tr = ∅ In this setting, we suppose T aux and T tr are mutually exclusive, and the adversary cannot obtain samples labeled with the target label.

In this setting, the adversary cannot obtain any samples with labels used for training of model f .

In this sense, the background knowledge of the adversary in this setting is the poorest among the three settings.

If the sample distribution of the auxiliary samples is close to that of the true underlying distribution, we can expect that estimation of d Xaux|Yaux can be used as an approximation of d Xtr|Ytr .

More specifically, we can obtain the generative model of the target label d Xaux|Yaux=y (t * ) by specifying the one-hot vector of the target label as the condition.

As we mentioned in Section 3.4, the sample generating distribution of the auxiliary samples is not necessarily equal or close to the true sample generating distribution.

In the "partly same" setting or "mutually exclusive" setting, D aux does not contain samples labeled with t * at all.

It is well known that GANs can generate samples with interpolating latent variables BID2 ), Radford et al. (2015 ).

We expect that PreImageGAN generates samples with the target label by interpolation of latent variables of given samples without having samples of the target label.

More specifically, if latent variables of given auxiliary samples are diverse enough and d Xaux|Yaux well approximates the true sample generating distribution, we expect that GAN can generate samples with the target label by interpolating obtained latent variables of auxiliary samples without having samples with the target label.

Generator G : (Z, Y) → X of PreImageGAN generates fake samples x fake = G(z, y) using random draws of y and z. After the learning process is completed, we expect generated fake samples x fake satisfy f (x fake ) = y. On the other hand, discriminator D : X → R takes as input a sample x and discriminates whether it is a generated fake sample x fake or a real sample x real ∈ D aux .

FIG0 : Inference on the space of y. Here, we suppose the adversary has auxiliary samples labeled with alphabets only and a probabilistic classification model that takes an image of a number and outputs the corresponding number.

The axis corresponds to an element of the probabilistic vector outputted by the classification model.

For example, y 9 = (f (x)) 9 denotes the probability that the model discriminates the input image as "9".

The green region in the figure describes the spanned by auxiliary samples in D aux .

D aux does not contain images of numbers classified so that y 9 = 1 or y 8 = 1 whereas PreImageGAN generates samples close to "9" by interpolating latent variables of images that are close to "9" such as "Q" and "g".With these requirements, the objective function of G and D is formulated as follows DISPLAYFORM0 where ∥ · ∥ L ≤ 1 denotes α-Lipschitz functions with α ≤ 1.By maximizing the first and second term concerning D, Wasserstein distance between the marginal of the generator ∫ G(Z, Y aux )dY aux and the generative distribution of auxiliary samples d Xaux is minimized.

By maximizing the similarity between y and f (G (z, y) ), G is trained so that samples generated from G(z, y) satisfy f (G(z, y)) = y. γ ≥ 0 works as a parameter adjusts the effect of this term.

For sample generation with PreImageGAN, G(Z, y (t * ) ) is utilized as the estimation of ρ t * .

We here remark that model f is regarded as a constant in the learning process of GAN and used as it is.

In this section, we show that the proposed method enables to perform the C2G attack with experiments.

We experimentally demonstrate that the adversary can successfully estimate ρ t * given classifier f and the set of unlabeled auxiliary samples D aux = {x|x ∈ X} even in the partly same setting and the mutually exclusive setting under some conditions.

For demonstration, we consider a hand-written character classification problem (EMNIST) and a face recognition problem (FaceScrub).

We used the Adam optimizer (α = 2 × 10 −4 , β 1 = 0.5, β 2 = 0.999) for the training of the generator and discriminator.

The batch size was set as 64.

We set the number of discriminator (critic) iterations per each generator iteration n cric = 5.

To enforce the 1-Lipschitz continuity of the discriminator, we add a gradient penalty (GP) term to the loss function of the discriminator BID8 ) and set the strength parameter of GP as λ = 10.

We used 128-dim uniform random distribution [−1, 1] 128 as d Z .

We estimated d Yaux empirically from {f (x)|x ∈ D aux } using kernel density estimation where the bandwidth is 0.01, and the Gaussian kernel was employed.

EMNIST consists of grayscale 28x28 pixel images from 62 alphanumeric characters (0-9A-Za-z).

We evaluate the C2G attack with changing the richness of the adversary's background knowledge as discussed in Section 3.4 (exact same, partly same, and mutually exclusive) to investigate how the richness of the auxiliary data affects the results.

Also, to investigate how the choice of the auxiliary data affects the results, we tested two different types of target labels as summarized in TAB1 (lower-case target) and TAB2 (numeric target).

In the former setting, an alphanumeric classification model is given to the adversary.

In the latter setting, a numeric classification model is given to the adversary.

In this setting, the target label t * was set as lower-case characters (t * ∈ {a, b, . . .

, z}) ( TAB1 ).

In the exact/partly same setting, an alphanumeric classifier (62 labels) is given to the adversary where the classifier is trained for ten epochs and achieved test accuracy 0.8443.

In the mutually exclusive setting, an alphanumeric classifier (36 labels) given to the adversary where the classifier is trained for ten epochs and achieved test accuracy 0.9202.

See TAB1 for the detailed settings.

In the training process of PreImageGAN, we trained the generator for 20k iterations.

We set the initial value of γ to 0, incremented gamma by 0.001 per generator iteration while γ is kept less than 10.

Fig. 3 represents the results of the C2G attack with targeting lower-case characters against given alphanumeric classification models.

Alphabets whose lower-case and upper-case letters are similar (e.g., C, K) are easy to attack.

So, we selected alphabets whose lower-case letter and upper-case letter shapes are dissimilar in Fig. 3 .In the exact same setting, we can confirm that the PreImageGAN works quite successfully.

In the partly same setting, some generated images are disfigured compared to the exact same setting (especially when t * = q) while most of the target labels are successfully reconstructed.

In the mutually exclusive setting, some samples are disfigured (especially when t * = h, i, q) while remaining targets are successfully reconstructed.

As an extreme case, we tested the case when the auxiliary data consists of images drawn from uniform random, and we observed that the C2G attack could generate no meaningful images (See Fig. 7 in Appendix A).

From these results, we can conclude that the C2G attack against alphanumeric classifiers works successfully except several although there is an exception in the mutually exclusive setting.

We also tested the C2G attack when the target label t * was set as numeric characters (t * ∈ {0, 1, . . .

, 9}).

In the exact/partly same setting, an alphanumeric classifier (62 labels, test accuracy 0.8443.) is given to the adversary.

In the mutually exclusive setting, a numeric classifier (10 labels, test accuracy 0.9911) given to the adversary where the classifier is trained for ten epochs and achieved test accuracy 0.9202.

See TAB2 for the detailed settings.

PreImageGAN was trained in the same setting as the previous subsection.

Fig. 4 represents the results of the C2G attack with targeting numeric characters against given classification models.

In the exact/partly same setting, the PreImageGAN works quite successfully as well with some exceptions; e.g., "3" and "7" are slightly disfigured in the partly same setting.

On the other hand, in the mutually exclusive setting, images targeting "0" and "1" look like the target numeric characters while remaining images are disfigured or look like other alphabets.

As shown from these results, in the mutually exclusive setting, the C2G attack against alphabets works well with while it fails when targeting numeric characters.

One of the reasons for this failure is in the incompleteness of the classifier given to the attacker.

More precisely, when the classifier recognizes images with non-target labels as a target labels falsely, C2G attack fails.

For example, in Fig 4, images of "T" are generated as images of "7" in the mutually exclusive setting.

This is because the given classifier recognizes images of "T" as "7" falsely, and the PreImageGAN thus generates images like "T" as "7".

See Table 8 in Appendix B; many alphabets are recognized as numeric characters falsely.

As long as the given classification model recognizes Figure 3: C2G attack against an alphanumeric classifier with changing the richness of the background knowledge of the adversary targeting lowercase letters.

The samples in the bottom row ("y: random") are generated when y is randomly drawn from empirically estimated d Yaux .non-target characters as target characters, the C2G attack cannot generate images of the target labels correctly.

In Fig 3, images of "h", "i", and "q" are disfigured in the mutually exclusive setting.

This disfiguring occurs for a different reason (see Table 6 in Appendix B; no significant false recognition can be found for these characters).

We consider this is because the image manifold that the classifier recognizes as the target character does not exactly fit the image manifold of the target character.

Since the images generated by the C2G attack for "h," "i", and "q" are recognized as "h," "i", and "q" by the classifier with a very high probability, the images work as adversarial examples.

Detailed analysis of the failure of the C2G attack would give us a hint to establish defense methods against the C2G attack.

Detailed consideration of this failure remains as future work.

Finally, we evaluate the quality of the C2G attack; we measured the inception accuracy using.

Here we employed a ResNet-based network architecture as f ′ (see Table E in detail).

As a baseline, we trained ACGAN BID16 ] with the same training samples D tr and evaluated the inception accuracy.

In the exact same setting, the inception accuracy is almost equal to ACGAN.

From the results, we can see that the inception scores drop as the background knowledge of the adversary becomes poorer.

This result indicates that the background knowledge of the adversary affects the result of the C2G attack significantly.

FaceScrub dataset consists of color face images (530 persons).

We resized images to 64x64 pixel images for experiments and evaluated the C2G attack in the mutually exclusive setting.

In detail, we picked up 100 people as T tr (see Appendix D for the list) and used remaining 430 persons as T aux (mutually exclusive setting).

If the adversary can generate face images of the 100 people in T tr by utilizing model f recognizing T tr and face images with labels in T aux , we can confirm that the C2G attack works successfully.

D tr consists of 12k images, and D aux consists of 53k images.

f is trained on D tr for ten epochs and achieved test accuracy 0.8395.

In training PreImageGAN, we train the generator for 130k iterations.

We set the initial value of γ to 0, incremented gamma by 0.0001 per generator iteration while γ is kept less than 10.

FIG3 represents the results of the C2G attack against the face recognition model.

Those samples are randomly generated without human selection.

The generated face images well capture the features of the face images in the training samples.

From the results, we can see that the C2G attack successfully reconstruct training samples from the face recognition model without having training samples in the mutually exclusive setting.

As byproducts, we can learn what kind of features the model used for face recognition with the results of the C2G attack.

For example, all generated face images of Keanu Reeves wear the mustache.

This result implies that f exploits his mustache to recognize Keanu Reeves.

One may concern that the PreImageGAN simply picks up images in the auxiliary samples that look like the target quite well, but labeled as some other person.

To show that the PreImageGAN does not simply pick up similar images to the target, but it generates images of the target by exploiting the features of face images extracted from the auxiliary images, we conducted two experiments.

First, we evaluated the probability with which classifier f recognizes images in the auxiliary dataset and images generated by the PreImageGAN as the target (Keanu Reeves and Marg Helgenberger) in FIG4 .

The probabilities are sorted, and the top 500 results are shown in the figure.

The orange lines denote the probability with which the images in the auxiliary dataset is recognized as the target.

A few images have a high probability (>0.80), but much less than the probabilities with which the target image in training data(blue lines) is recognized as the target (>0.80, 80 images).

This indicates that the auxiliary samples do not contain images that are quite similar to the targets.

The green lines denote the probability with which the images generated by the PreImageGAN are recognized as the target.

As seen from the results, the generated images are recognized as the target with extremely high probability (> 0.95).

This suggests that the PreImageGAN could generate images recognized as a target with high probability from samples not recognized as the target.

Figure 4 : C2G attack against a numeric classification model with changing the richness of the background knowledge of the adversary targeting numeric letters.

The samples in the bottom row ("y: random") are images generated when y is randomly drawn from empirically estimated d Yaux .

Table 4 : Inception Accuracy with changing background knowledge of the adversary (C2G attack against numeric classifier) target label: t * Setting 0 1 2 3 4 5 6 7 8 9 Baseline (ACGAN) 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 Exact same 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 1.00 1.

Second, to demonstrate that the PreImgeGAN can generate a wide variety of images of the target, we generated images that interpolate two targets (see Appendix C for the settings and the results).

As seen from the results, the PreImageGAN can generate face images by exploiting both classifier f and the features extracted from the auxiliary dataset.

Finally, we conducted our experiments Intel(R) Xeon(R) CPU E5-2623 v3 and a single GTX TITAN X (Maxwell), and it spends about 35 hours to complete the entire training process for the FaceScrub experiment.

The computational capability of our environment is almost the same as the p2 instance (p2.xlarge) of Amazon Web Service.

Usage of a p2.xlarge instance for 50 hours costs about $45.

This means that the C2G attack is a quite practical attack anyone can try with a regular computational resource at low cost.6 RELATED WORK BID4 proposed the model inversion attack against machine learning algorithms that extract private input attributes from published predicted values.

Through a case study of personalized adjustment of Warfaline dosage, BID4 showed that publishing predicted dosage amount can cause leakage of private input attributes (e.g., personal genetic information) in generalized linear regression.

BID3 presented a model inversion attack that reconstructs face images from a face recognition model.

The significant difference between the C2G attack and the model inversion attack is the goal of the adversary.

In the model inversion attack, the adversary tries to estimate a private input (or input attributes) x from predicted values y = f (x) using the predictor f .

Thus, the adversary's goal is the private input x itself in the model inversion attack.

By contrast, in the C2G attack, the adversary's goal is to obtain the training sample distribution.

Another difference is that the target network model.

The target model of BID3 was a shallow neural network model while ours is deep neural networks.

As the network architecture becomes deeper, it becomes more difficult to extract information about the input because the output of the model tends to be more abstract BID10 ).

BID10 discussed leakage of training samples in collaborative learning based on the model inversion attack using the IcGAN BID17 ).

In their setting, the adversary's goal is not to estimate training sample distribution but to extract training samples.

Also, their demonstration is limited to small-scale datasets, such as MNIST dataset (hand-written digit grayscale images, 10 labels) and AT&T dataset (400 face grayscale images with 40 labels).

By contrast, our experiments are demonstrated with larger datasets, such as EMNIST dataset (62 labels) and FaceScrub dataset (530 labels, 100,000+ color images).

The results of the C2G attack against face recognition in the mutually exclusive setting.

We trained a face recognition model of 100 people (including Brad Pitt, Keanu Reeves, Nicolas Cage and Marg Helgenberger), and evaluated the C2G attack where the classification model for the 100 people is given to the adversary while no face images of the 100 people are not given.

Generated samples are randomly selected, and we did not cherry-pick "good" samples.

We can recognize the generated face images as the target label.

This indicates that the C2G attack works successfully for the face recognition model.

BID9 discussed the membership inference attack against a generative model trained by BEGAN BID2 ) or DCGAN (Radford et al. (2015) ).

In the membership inference attack, the adversary's goal is to determine whether the sample is contained in the private training dataset; the problem and the goal are apparently different from ours.

Images in the auxiliary data set are not recognized as the target with high probability while images generated by the PreImageGAN are recognized as the target with very high probability.

Song et al. FORMULA0 discussed malicious regularizer to memorize private training dataset when the adversary can specify the learning algorithm and obtain the classifier trained on the private training data.

Their experiments showed that the adversary can estimate training data samples from the classifier when the classifier is trained with malicious regularizer.

Since our setting does not assume that the adversary can specify the learning algorithm, the problem setting is apparently different from ours.

BID11 and BID12 consider the understanding representation of deep neural networks through reconstruction of input images from intermediate features.

Their studies are related to ours in the sense that the algorithm exploits intermediate features to attain the goal.

To the best of our knowledge, no attack algorithm has been presented to estimate private training sample distribution as the C2G attack achieves.

As described in this paper, we formulated the Classifier-to-Generator (C2G) Attack, which estimates the training sample distribution ρ t * from given classification model f and auxiliary dataset D tr .

As an algorithm for C2G attack, we proposed PreImageGAN which is based on ACGAN and WGAN.

Fig. 7 represents the results of the C2G attack when the auxiliary data consists of noisy images which are drawn from the uniform distribution.

All generated images look like noise images, not numeric letters.

This result reveals that the C2G attack fails when the auxiliary dataset is not sufficiently informative.

More specifically, we can consider the C2G attack fails when the attacker does not have appropriate background knowledge of the training data distribution.(a) t * = 0 DISPLAYFORM0 Figure 7: Images generated by the C2G attack when the target label is set as t * = 0, 1, 2 and uniformly generated noise images are used as the auxiliary dataset.

We used an alphanumeric letter classifier (label num:62) described in Sec. 5.2 as f for this experiment.

Images generated by the C2G attack is significantly affected by the property of the classification model f given to the C2G attack.

To investigate this, we measured how often non-target characters are falsely recognized as a target character by classification model f with high probability (greater than 0.9).

The tables shown below in this subsection contain at most the top-five falsely-recognized characters for each target label.

If no more than five characters are falsely recognized with high probability, the fields remain blank.

We consider an alphanumeric classifier f trained in the exactly/partly same setting of TAB1 represents the characters falsely recognized as the target label with high probability.

Similarly, for an alphanumeric classifier f trained in the mutually exclusive setting of TAB1 represents the characters falsely recognized as the target label with high probability.

Table 6 : Top-five non-target characters falsely recognized as target characters (lower-case) with high probability by alphanumeric (lower-case, numeric) classifier. (X: 0.123) means that the classifier misclassified 12.3%(0.123) of images of X as the target character with high probability (>0.9).

As confirmed by the results of TAB5 , few letters are falsely recognized by alphanumeric classifier f in the exactly/partly same setting.

This results support the fact that the C2G attack works quite successfully in this setting (see Figure 3 ).

In Table 6 , "E" and "F" are falsely recognized as "e" and "f", respectively frequently, while the C2G attack could successfully generate "e" and "f" in Figure 3 .

This is because ("e", "E") and ("f", "F") have similar shapes and this false recognition of the model does not (fortunately) disfiguring in generation of the images.

In Figure 3 , images of "i" and "q" in the mutually exclusive setting are somewhat disfigured while no false recognition of these characters is found in Table 6 .

This disfiguring is supposed to occur because the classification model f does not necessarily exploit the entire structure of "i" and "q"; the image manifold that f recognizes as "i" and "q" does not exactly fits the image manifold of "i" and "q".

Next, we consider an alphanumeric classifier f trained in the exactly/partly same setting of TAB2 represents the characters falsely recognized as the target label with high probability.

Similarly, for a numeric classifier f trained in the mutually exclusive setting of TAB2 represents the characters falsely recognized as the target label with high probability.

In the exactly/partly same setting, we see in Table 7 that not many letters are falsely recognized as non-target characters.

This results support the fact that the C2G attack against numeric characters works successfully in the exactly/partly same setting (see Figure 4) .On the other hand, in the mutually exclusive setting, Table 8 reveals that many non-target characters are falsely recognized as the target characters.

In Figure 4 , generated images of "6", "7" and "8" look like "h", "T" and "e", respectively.

In Table 8 , images of "h", "T" and "e" is falsely recognized as "6", "7" and "8", respectively.

From this analysis, if the classifier falsely recognized non-target images as the target label, the C2G attack fails and PreImageGAN generates non-target images as target images.

In Figure 4 , images of 0,1,5 and 9 seem to be generated successfully while images for the other numeric characters are more or less disfigured.

In Table 8 , "O", "l", "s" and "q" are falsely recognized as "0", "1" ,"5" and "9", respectively.

This suggests that f does not necessarily contain appropriate features to recognize the target.

However, the C2G attack could generate images that are quite similar to the target characters using f , and eventually, the C2G attack generated images look like "0", "1", "5" and "9" successfully.

For the remaining characters, since f does not contain the necessary information to generate the images of the target, the resulting images are disfigured.

(a) Interpolation on both z and y. z and y is randomly chosen from dZ and dY aux , respectively.(b) Interpolation on z with fixing y. y is set to one-hot vector in which the element corresponds to "Blad Pitt" is activated.(c) Interpolation on y with fixing z.

@highlight

Estimation of training data distribution from trained classifier using GAN.