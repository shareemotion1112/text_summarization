We present CROSSGRAD , a method to use multi-domain training data to learn a classifier that generalizes to new domains.

CROSSGRAD does not need an adaptation phase via labeled or unlabeled data, or domain features in the new domain.

Most existing domain adaptation methods attempt to erase domain signals using techniques like domain adversarial training.

In contrast, CROSSGRAD is free to use domain signals for predicting labels, if it can prevent overfitting on training domains.

We conceptualize the task in a Bayesian setting, in which a sampling step is implemented as data augmentation, based on domain-guided perturbations of input instances.

CROSSGRAD jointly trains a label and a domain classifier on examples perturbed by loss gradients of each other’s objectives.

This enables us to directly perturb inputs, without separating and re-mixing domain signals while making various distributional assumptions.

Empirical evaluation on three different applications where this setting is natural establishes that  (1) domain-guided perturbation provides consistently better generalization to unseen domains, compared to generic instance perturbation methods, and  (2) data augmentation is a more stable and accurate method than domain adversarial training.

We investigate how to train a classification model using multi-domain training data, so as to generalize to labeling instances from unseen domains.

This problem arises in many applications, viz., handwriting recognition, speech recognition, sentiment analysis, and sensor data interpretation.

In these applications, domains may be defined by fonts, speakers, writers, etc.

Most existing work on handling a target domain not seen at training time requires either labeled or unlabeled data from the target domain at test time.

Often, a separate "adaptation" step is then run over the source and target domain instances, only after which target domain instances are labeled.

In contrast, we consider the situation where, during training, we have labeled instances from several domains which we can collectively exploit so that the trained system can handle new domains without the adaptation step.

Let D be a space of domains.

During training we get labeled data from a proper subset D ⊂ D of these domains.

Each labeled example during training is a triple (x, y, d) where x is the input, y ∈ Y is the true class label from a finite set of labels Y and d ∈ D is the domain from which this example is sampled.

We must train a classifier to predict the label y for examples sampled from all domains, including the subset D \ D not seen in the training set.

Our goal is high accuracy for both in-domain (i.e., in D) and out-of-domain (i.e., in D \ D) test instances.

One challenge in learning a classifier that generalizes to unseen domains is that Pr(y|x) is typically harder to learn than Pr(y|x, d).

While BID31 addressed a similar setting, they assumed a specific geometry characterizing the domain, and performed kernel regression in this space.

In contrast, in our setting, we wish to avoid any such explicit domain representation, appealing instead to the power of deep networks to discover implicit features.

Lacking any feature-space characterization of the domain, conventional training objectives (given a choice of hypotheses having sufficient capacity) will tend to settle to solutions that overfit on the set of domains seen during training.

A popular technique in the domain adaptation literature to generalize to new domains is domain adversarial training BID2 BID26 .

As the name suggests, here the goal is to learn a transformation of input x to a domain-independent representation, with the hope that amputating domain signals will make the system robust to new domains.

We show in this paper that such training does not necessarily safeguard against over-fitting of the network as a whole.

We also argue that even if such such overfitting could be avoided, we do not necessarily want to wipe out domain signals, if it helps in-domain test instances.

In a marked departure from domain adaptation via amputation of domain signals, we approach the problem using a form of data augmentation based on domain-guided perturbations of input instances.

If we could model exactly how domain signals for d manifest in x, we could simply replace these signals with those from suitably sampled other domains d to perform data augmentation.

We first conceptualize this in a Bayesian setting: discrete domain d 'causes' continuous multivariate g, which, in combination with y, 'causes' x. Given an instance x, if we can recover g, we can then perturb g to g , thus generating an augmented instance x .

Because such perfect domain perturbation is not possible in reality, we first design an (imperfect) domain classifier network to be trained with a suitable loss function.

Given an instance x, we use the loss gradient w.r.t.

x to perturb x in directions that change the domain classifier loss the most.

The training loss for the y-predictor network on original instances is combined with the training loss on the augmented instances.

We call this approach cross-gradient training, which is embodied in a system we describe here, called CROSSGRAD.

We carefully study the performance of CROSSGRAD on a variety of domain adaptive tasks: character recognition, handwriting recognition and spoken word recognition.

We demonstrate performance gains on new domains without any out-of-domain instances available at training time.

Domain adaptation has been studied under many different settings: two domains BID2 BID26 or multiple domains BID17 , with target domain data that is labeled BID10 BID11 BID21 or unlabeled BID7 BID5 BID2 , paired examples from source and target domain (KuanChuan et al., 2017) , or domain features attached with each domain BID32 .

Domain adaptation techniques have been applied to numerous tasks in speech, language processing and computer vision BID29 BID23 BID12 BID10 BID21 BID7 BID16 BID9 BID27 .

However, unlike in our setting, these approaches typically assume the availability of some target domain data which is either labeled or unlabeled.

For neural networks a recent popular technique is domain adversarial networks (DANs) BID25 BID2 .

The main idea of DANs is to learn a representation in the last hidden layer (of a multilayer network) that cannot discriminate among different domains in the input to the first layer.

A domain classifier is created with the last layer as input.

If the last layer encapsulates no domain information apart from what can be inferred from the label, the accuracy of the domain classifier is low.

The DAN approach makes sense when all domains are visible during training.

In this paper, our goal is to generalize to unseen domains.

Domain generalization is traditionally addressed by learning representations that encompass information from all the training domains.

BID20 learn a kernel-based representation that minimizes domain dissimilarity and retains the functional relationship with the label.

BID1 extends BID20 by exploiting attribute annotations of examples to learn new feature representations for the task of attribute detection.

In BID3 , features that are shared across several domains are estimated by jointly learning multiple data-reconstruction tasks.

Such representations are shown to be effective for domain generalization, but ignore any additional information that domain features can provide about labels.

Domain adversarial networks (DANs) BID2 can also be used for domain generalization in order to learn domain independent representations.

A limitation of DANs is that they can be misled by a representation layer that over-fits to the set of training domains.

In the extreme case, a representation that simply outputs label logits via a last linear layer (making the softmax layer irrelevant) can keep both the adversarial loss and label loss small, and yet not be able to generalize to new test domains.

In other words, not being able to infer the domain from the last layer does not imply that the classification is domain-robust.

Since we do not assume any extra information about the test domains, conventional approaches for regularization and generalizability are also relevant.

BID30 use exemplar-based SVM classifiers regularized by a low-rank constraint over predictions.

BID13 also deploy SVM based classifier and regularize the domain specific components of the learners.

The method most related to us is the adversarial training of BID24 BID6 BID18 where examples perturbed along the gradient of classifier loss are used to augment the training data.

perturbs examples.

Instead, our method attempts to model domain variation in a continuous space and perturbs examples along domain loss.

Our Bayesian model to capture the dependence among label, domains, and input is similar to FIG1 ), but the crucial difference is the way the dependence is modeled and estimated.

Our method attempts to model domain variation in a continuous space and project perturbation in that space to the instances.

We assume that input objects are characterized by two uncorrelated or weakly correlated tags: their label and their domain.

E.g. for a set of typeset characters, the label could be the corresponding character of the alphabet ('A', 'B' etc) and the domain could be the font used to draw the character.

In general, it should be possible to change any one of these, while holding the other fixed. obtained by a complicated, un-observed mixing 1 of y and g. In the training sample L, nodes y, d, x are observed but L spans only a proper subset D of the set of all domains D. During inference, only x is observed and we need to compute the posterior Pr(y|x).

As reflected in the network, y is not independent of d given x. However, since d is discrete and we observe only a subset of d's during training, we need to make additional assumptions to ensure that we can generalize to a new d during testing.

The assumption we make is that integrated over the training domains the distribution P (g) of the domain features is well-supported in L. More precisely, generalizability of a training set of domains D to the universe of domains D requires that DISPLAYFORM0 Under this assumption P (g) can be modeled during training, so that during inference we can infer y for a given x by estimating DISPLAYFORM1 whereĝ = argmax g Pr(g|x) is the inferred continuous representation of the domain of x.

This assumption is key to our being able to claim generalization to new domains even though most real-life domains are discrete.

For example, domains like fonts and speakers are discrete, but their variation can be captured via latent continuous features (e.g. slant, ligature size etc.

of fonts; speaking rate, pitch, intensity, etc. for speech).

The assumption states that as long as the training domains span the latent continuous features we can generalize to new fonts and speakers.

We next elaborate on how we estimate Pr(y|x,ĝ) andĝ using the domain labeled data L = {(x, y, d)}. The main challenge in this task is to ensure that the model for Pr(y|x,ĝ) is not overfitted on the inferred g's of the training domains.

In many applications, the per-domain Pr(y|x, d) is significantly easier to train.

So, an easy local minima is to choose a different g for each training d and generate separate classifiers for each distinct training domain.

We must encourage the network to stay away from such easy solutions.

We strive for generalization by moving along the continuous space g of domains to sample new training examples from hallucinated domains.

Ideally, for each training instance (x i , y i ) from a given domain d i , we wish to generate a new x by transforming its (inferred) domain g i to a random domain sampled from P (g), keeping its label y i unchanged.

Under the domain continuity assumption (A1), a model trained with such an ideally augmented dataset is expected to generalize to domains in D \ D.However, there are many challenges to achieving such ideal augmentation.

To avoid changing y i , it is convenient to draw a sample g by perturbing g i .

But g i may not be reliably inferred, leading to a distorted sample of g. For example, if the g i obtained from an imperfect extraction conceals label information, then big jumps in the approximate g space could change the label too.

We propose a more cautious data augmentation strategy that perturbs the input to make only small moves along the estimated domain features, while changing the label as little as possible.

We arrive at our method as follows.ĝ DISPLAYFORM2 Figure 2: CROSSGRAD network design.

Domain inference.

We create a model G(x) to extract domain features g from an input x. We supervise the training of G to predict the domain label d i as S(G(x i )) where S is a softmax transformation.

We use J d to denote the cross-entropy loss function of this classifier.

Specifically, DISPLAYFORM3 is the domain loss at the current instance.

Domain perturbation.

Given an example (x i , y i , d i ), we seek to sample a new example (x i , y i ) (i.e., with the same label y i ), whose domain is as "far" from d i as possible.

To this end, consider setting DISPLAYFORM4 Intuitively, this perturbs the input along the direction of greatest domain change 2 , for a given budget of ||x i − x i ||.

However, this approach presupposes that the direction of domain change in our domain classifier is not highly correlated with the direction of label change.

To enforce this in our model, we shall train the domain feature extractor G to avoid domain shifts when the data is perturbed to cause label shifts.

What is the consequent change of the continuous domain featuresĝ i ?

This turns out to be DISPLAYFORM5 where J is the Jacobian ofĝ w.r.t.

x. Geometrically, the JJ term is the (transpose of the) metric tensor matrix accounting for the distortion in mapping from the x-manifold to theĝ-manifold.

While this perturbation is not very intuitive in terms of the direct relation betweenĝ and d, we show in the Appendix that the input perturbation ∇ xi J d (x i , d i ) is also the first step of a gradient descent process to induce the "natural" domain perturbation DISPLAYFORM6 The above development leads to the network sketched in FIG2 , and an accompanying training algorithm, CROSSGRAD, shown in Algorithm 1.

Here X, Y, D correspond to a minibatch of instances.

Our proposed method integrates data augmentation and batch training as an alternating sequence of steps.

The domain classifier is simultaneously trained with the perturbations from the label classifier network so as to be robust to label changes.

Thus, we construct cross-objectives J l and J d , and update their respective parameter spaces.

We found this scheme of simultaneously training both networks to be empirically superior to independent training even though the two classifiers do not share parameters.

Algorithm 1 CROSSGRAD training pseudocode.

DISPLAYFORM7 , step sizes l , d , learning rate η, data augmentation weights α l , α d , number of training steps n. DISPLAYFORM8 DISPLAYFORM9 If y and d are completely correlated, CROSSGRAD reduces to traditional adversarial training.

If, on the other extreme, they are perfectly uncorrelated, removing domain signal should work well.

The interesting and realistic situation is where they are only partially correlated.

CROSSGRAD is designed to handle the whole spectrum of correlations.

In this section, we demonstrate that CROSSGRAD provides effective domain generalization on four different classification tasks under three different model architectures.

We provide evidence that our Bayesian characterization of domains as continuous features is responsible for such generalization.

We establish that CROSSGRAD's domain guided perturbations provide a more consistent generalization to new domains than label adversarial perturbation BID6 which we denote by LABELGRAD.

Also, we show that DANs, a popular domain adaptation method that suppresses domain signals, provides little improvement over the baseline BID2 BID26 ).We describe the four different datasets and present a summary in TAB1 .Character recognition across fonts.

We created this dataset from Google Fonts 3 .

The task is to identify the character across different fonts as the domain.

The label set consists of twenty-six letters of the alphabet and ten numbers.

The data comprises of 109 fonts which are partitioned as 65% train, 25% test and 10% validation folds.

For each font and label, two to eighteen images are generated by randomly rotating the image around the center and adding pixel-level random noise.

The neural network is LeNet (LeCun et al., 1998) Handwriting recognition across authors.

We used the LipiTk dataset that comprises of handwritten characters from the Devanagari script 4 .

Each writer is considered as a domain, and the task is to recognize the character.

The images are split on writers as 60% train, 25% test and 15% validation.

The neural network is the same as for the Fonts dataset above.

MNIST across synthetic domains.

This dataset derived from MNIST was introduced by BID3 .

Here, labels comprise the 10 digits and domains are created by rotating the images in multiples of 15 degrees: 0, 15, 30, 45, 60 and 75.

The domains are labeled with the angle by which they are rotated, e.g., M15, M30, M45.

We tested on domain M15 while training on the rest.

The network is the 2-layer convolutional one used by BID19 .Spoken word recognition across users.

We used the Google Speech Command Dataset 5 that consists of spoken word commands from several speakers in different acoustic settings.

Spoken words are labels and speakers are domains.

We used 20% of domains each for testing and validation.

The number of training domains was 100 for the experiments in TAB3 .

We also report performance with varying numbers of domains later in TAB6 .

We use the architecture of BID22 6 .For all experiments, the set of domains in the training, test, and validation sets were disjoint.

We selected hyper-parameters based on accuracy on the validation set as follows.

For LABELGRAD the parameter α was chosen from {0.1, 0.25, 0.75, 0.5, 0.9} and for CROSSGRAD we chose α l = α d from the same set of values.

We chose ranges so that L ∞ norm of the perturbations are of similar sizes in LABELGRAD and CROSSGRAD.

The multiples in the range came from {0.5, 1, 2, 2.5}. The optimizer for the first three datasets is RMS prop with a learning rate (η) of 0.02 whereas for the last Speech dataset it is SGD with η = 0.001 initially and 0.0001 after 15 iterations.

In CROSSGRAD networks, g is incorporated in the label classifier network by concatenating with the output from the last but two hidden layer.

In TAB3 we compare CROSSGRAD with domain adversarial networks (DAN), label adversarial perturbation (LABELGRAD), and a baseline that performs no special training.

For the MNIST dataset the baseline is CCSA BID19 and D-MTAE BID3 .

We observe that, for all four datasets, CROSSGRAD provides an accuracy improvement.

DAN, which is designed specifically for domain adaptation, is worse than LABELGRAD, which does not exploit domain signal in any way.

While the gap between LABELGRAD and CROSSGRAD is not dramatic, it is consistent as supported by this Changing model architecture.

In order to make sure that these observed trends hold across model architectures, we compare different methods with the model changed to a 2-block ResNet BID8 ) instead of LeNet (LeCun et al., 1998) for the Fonts and Handwriting dataset in TAB4 :

Accuracy with varying model architectures.

We present insights on the working of CROSSGRAD via experiments on the MNIST dataset where the domains corresponding to image rotations are easy to interpret.

In Figure 6a we show PCA projections of the g embeddings for images from three different domains, corresponding to rotations by 30, 45, 60 degrees in green, blue, and yellow respectively.

The g embeddings of domain 45 (blue) lies in between the g of domains 30 (green) and 60 (yellow) showing that the domain classifier has successfully extracted continuous representation of the domain even when the input domain labels are categorical.

Figure 6b shows the same pattern for domains 0, Finally, we observe in FIG4 that the embeddings are not correlated with labels.

For both domains 30 and 45 the colors corresponding to different labels are not clustered.

This is a consequence of CROSSGRAD's symmetric training of the domain classifier via label-loss perturbed images.

We next present a couple of experiments that provide insight into the settings in which CROSSGRAD is most effective.

First, we show the effect of increasing the number of training domains.

Intuitively, we expect CROSSGRAD to be most useful when training domains are scarce and do not directly cover the test domains.

We verified this on the speech dataset where the number of available domains is large.

We varied the number of training domains while keeping the test and validation data fixed.

TAB6 summarizes our results.

Note that CROSSGRAD outperforms the baseline and LABELGRAD most significantly when the number of training domains is small (40).

As the training data starts to cover more and more of the possible domain variations, the marginal improvement provided by CROSS-GRAD decreases.

In fact, when the models are trained on the full training data (consisting of more than 1000 domains), the baseline achieves an accuracy of 88.3%, and both CROSSGRAD and LA-BELGRAD provide no gains 7 beyond that.

DAN, like in other datasets, provides unstable gains and is difficult to tune.

LABELGRAD shows smaller relative gains than CROSSGRAD but follows the same trend of reducing gains with increasing number of domains.

In general, how CROSSGRAD handles multidimensional, non-linear involvement of g in determining x is difficult to diagnose.

To initiate a basic understanding of how data augmentation supplies CROSSGRAD with hallucinated domains, we considered a restricted situation where the discrete domain is secretly a continuous 1-d space, namely, the angle of rotation in MNIST.

In this setting, We conducted leave-one-domain-out experiments by picking one domain as the test domain, and providing the others as training domains.

In TAB7 we compare the accuracy of different methods.

We also compare against the numbers reported by the CCSA method of domain generalization BID19 as reported by the authors.

It becomes immediately obvious from TAB7 that CROSSGRAD is beaten in only two cases: M0 and M75, which are the two extreme rotation angles.

For angles in the middle, CROSSGRAD is able to interpolate the necessary domain representation g via 'hallucination' from other training domains.

Recall from Figures 6c and 6d that the perturbed g during training covers for the missing test domains.

In contrast, when M0 or M75 are in the test set, CROSSGRAD's domain loss gradient does not point in the direction of domains outside the training domains.

If and how this insight might generalize to more dimensions or truly categorical domains is left as an open question.

Domain d and label y interact in complicated ways to influence the observable input x. Most domain adaption strategies implicitly consider the domain signal to be extraneous and seek to remove its effect to train more robust label predictors.

We presented CROSSGRAD, which considers them in a more symmetric manner.

CROSSGRAD provides a new data augmentation scheme based on the y (respectively, d) predictor using the gradient of the d (respectively, y) predictor over the input space, to generate perturbations.

Experiments comparing CROSSGRAD with various recent adversarial paradigms show that CROSSGRAD can make better use of partially correlated y and d, without requiring explicit distributional assumptions about how they affect x. CROSSGRAD is at its best when training domains are scarce and do not directly cover test domains well.

Future work includes extending CROSSGRAD to exploit labeled or unlabeled data in the test domain, and integrating the best of LABELGRAD and CROSSGRAD into a single algorithm.

Hence, the initial gradient descent step to affect a change of ∆ĝ in the domain features would increment x by J ∆ĝ.

The Jacobian, which is a matrix of first partial derivatives, can be computed by back-propagation.

Thus we get DISPLAYFORM0 which, by the chain rule, gives DISPLAYFORM1

@highlight

Domain guided augmentation of data provides a robust and stable method of domain generalization

@highlight

This paper proposes a domain generalization approach by domain-dependent data augmentation

@highlight

The authors introduce the CrossGrad method, which trains both a label classification task and a domain classification task.