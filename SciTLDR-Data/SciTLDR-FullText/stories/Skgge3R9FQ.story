Convolutional Neural Networks (CNNs) significantly improve the state-of-the-art for many applications, especially in computer vision.

However, CNNs still suffer from a tendency to confidently classify out-distribution samples from unknown classes into pre-defined known classes.

Further, they are also vulnerable to adversarial examples.

We are relating these two issues through the tendency of CNNs to over-generalize for areas of the input space not covered well by the training set.

We show that a CNN augmented with an extra output class can act as a simple yet effective end-to-end model for controlling over-generalization.

As an appropriate training set for the extra class, we introduce two resources that are computationally efficient to obtain: a representative natural out-distribution set and interpolated in-distribution samples.

To help select a representative natural out-distribution set among available ones, we propose a simple measurement to assess an out-distribution set's fitness.

We also demonstrate that training such an augmented CNN with representative out-distribution natural datasets and some interpolated samples allows it to better handle a wide range of unseen out-distribution samples and black-box adversarial examples without training it on any adversaries.

Finally, we show that generation of white-box adversarial attacks using our proposed augmented CNN can become harder, as the attack algorithms have to get around the rejection regions when generating actual adversaries.

Convolutional Neural Networks (CNNs) have allowed for significant improvements over the stateof-the-art in the last few years for various applications, and in particular for computer vision.

Notwithstanding these successes, challenging issues remain with these models.

In the following work, we specifically look at two concerns.

First, CNNs are vulnerable to different types of adversarial examples BID26 BID13 BID3 .

These adversarial examples are created by deliberately modifying clean samples with imperceptible perturbations, with the aim of misleading CNNs into classifying them to a wrong class with high confidence.

Second, CNNs are not able to handle instances coming from outside the task domain on which they are trained -the so-called out-distribution samples BID18 BID15 .

In other words, although these examples are semantically and statistically different from the (in-distribution) samples relevant to a given task, the neural network trained on the task assigns such out-of-concept samples with high-confidence to the pre-defined in-distribution classes.

Due to the susceptibility of CNNs to both adversaries and out-distribution samples, deploying them for real-world applications, in particular for security-sensitive ones, is a serious concern.

These two issues have been treated separately in the past, with two distinct family of approaches.

For instance, on the one hand, to handle out-distribution samples, some researchers have proposed threshold-based post-processing approaches with the aim of firstly calibrating the predictive confidence scores provided by either a single pre-trained CNN BID18 BID10 BID17 or an ensemble of CNNs BID15 , and then detecting out-distribution samples according to an optimal threshold.

However, it is difficult to define an optimal and stable threshold for rejecting a wide range of out-distribution samples without increasing the false negative rate (i.e., rejecting in-distribution samples).

On the other hand, researchers regarded adversarial examples as a distinct issue from the out-distribution problem and attempted to either correctly classify all adversaries through adversarial training of CNNs BID27 BID7 or reject all of them by training a separate detector BID5 BID20 .

The performance of these approaches at properly handling adversarial instances mostly depends on having access to a diverse set of training adversaries, which is not only computationally expensive but also handling some possible future adversaries, which have not been discovered yet, most likely is difficult.

It is known that deep neural networks (e.g. CNNs) are prone to over-generalization in the input space by partitioning it entirely into a set of pre-defined classes for a given in-distribution set (task), regardless of the fact that in-distribution samples may only be relevant to a small portion of the input space BID18 BID25 BID0 .

In this paper, we highlight that the two aforementioned issues of CNNs can be alleviated simultaneously through control of over-generalization.

To this end, we propose that an augmented CNN, a regular (naive) CNN with an extra class dubbed as dustbin, can be a simple yet effective solution, if it is trained on appropriate training samples for the dustbin class.

Furthermore, we introduce here a computationally-efficient answer to the following key question: how to acquire such an appropriate set to effectively reduced the over-generalized regions induced by naive CNN.

We note that our motivation for employing an augmented CNN is different from the threshold-based post-processing approaches that attempt to calibrate the predictive confidence scores of a pre-trained naive CNN without impacting its feature space.

Our motivation in fact is to learn a more expressive feature space, where along with learning the sub-manifolds corresponding to in-distribution classes, a distinct extra sub-manifold for the dustbin class can be obtained such that the samples drawn from many over-generalized regions including a wide-range of out-distribution samples and various types of adversaries are mapped to this "dustbin" sub-manifold.

As a training source for the extra class (dustbin), one can consider using synthetically generated out-distribution samples BID17 BID11 or adversarial examples BID8 .

However, using such generated samples is not only computationally expensive but also barely able to effectively reduce over-generalization compared to naive CNNs (see Sec. 3).

Instead of such synthetic samples, there are plenty of cost-effective training sources available for the extra dustbin class, namely natural out-distribution datasets.

By natural out-distribution sets we mean the sets containing some realistic (not synthetically generated) samples that are semantically and statistically different from those in the in-distribution set.

A representative natural out-distribution set for a given in-distribution task should be able to adequately cover the over-generalized regions.

To recognize such a representative natural set, we propose a simple measurement to assess its fitness for a given in-distribution set.

In addition to the selected set, we generate some artificial out-distribution samples through a straightforward and computationally efficient procedure, namely by interpolating some pair of in-distribution samples.

We believe a properly trained augmented CNN can be utilized as a threshold-free baseline for identifying concurrently a broad range of unseen out-distribution samples and different types of strong adversarial attacks.

The main contributions of the paper are summarized as:• By limiting the over-generalization regions induced by naive CNNs, we are able to drastically reduce the risk of misclassifying both adversaries and samples from a broad range of (unseen) out-distribution sets.

To this end, we demonstrate that an augmented CNN can act as a simple yet effective solution.• We introduce a measurement to select a representative natural out-distribution set among those available for training effective augmented CNNs, instead of synthesizing some dustbin samples using hard-to-train generators.• Based on extensive experiments on a range of different image classification tasks, we demonstrate that properly trained augmented CNNs can significantly reduce the misclassification rates for both 1) unseen out-distribution sets, and 2) for various types of strong black-box adversarial examples, even though they are never trained on any specific types of adversaries.• For the generation of white-box adversaries using our proposed augmented CNN, the adversarial attack algorithms frequently encounter dustbin regions rather than regions from other classes when distorting a clean samples, making the adversaries generation process more difficult.

The key idea of this paper is to make use of a CNN augmented with a dustbin class, trained on a representative set of out-distribution samples, as a simple yet effective candidate solution to limit over-generalization.

A visual illustration of this is given in FIG0 , which provides a schematic explanation of the influence of training samples used to learn the dustbin class on the out-distribution area coverage.

This figure illustrates how the choice of training samples for the extra dustbin class plays a central role for achieving an effective augmented CNN.

With a naive MLP (no dustbin class), a decision boundary is separating the whole input space into two classes ( FIG0 ), working on the complete input space even in regions that are deemed irrelevant for the task at hand.

As for augmented MLPs, the second plot ( FIG0 ) shows results with dustbin training samples picked to be around the decision boundary, where many adversarial examples are designed to be located.

As it can be observed, such augmented MLP can only slightly reduce over-generalized regions.

However, it might be able to classify some of adversaries as dustbin, and make generation of new adversaries harder since the adversarial attack algorithm should avoid the dustbin regions that are now located in between the two in-distribution classes.

Thus, using solely such adversaries as training set of the dustbin class can not adequately cover the over-generalized regions.

In another variation ( FIG0 ), the dustbin samples come from a out-distribution set, quite compact and located around the in-distribution samples from one specific class.

Training an augmented MLP on this kind of out-distribution samples cannot reduce over-generalization effectively.

Accordingly, we argue that out-distribution training samples that are distributed uniformly w.r.t in-distribution classes can be regarded as a representative set for the extra dustbin class ( FIG0 ).

Indeed, an augmented MLP trained on a representative set is able to classify a wide-range of unseen out-distribution sets and some of adversaries as its extra class, being more effective at controlling over-generalization.

It is worth to note that coupling a representative out-distribution set with the samples drawn around the decision boundaries can further strengthen the augmented neural network against adversarial examples.

There are many possible ways of acquiring some training samples for the extra class of augmented CNNs, ranging from artificially generated samples BID11 BID17 to natural available out-distribution sets.

Instead of making use of a generator, which is computationally expensive and hard to train, we propose the use of two cost-effective resources for acquiring dustbin training samples in order to train effective augmented CNNs: i) a selected representative natural out-distribution set and ii) interpolated samples.

A possible rich and readily accessible source of dustbin examples for training augmented models lies in natural out-distribution datasets.

These sets contain natural samples that are statistically and semantically different compared to the samples of a given task.

For example, NotMNIST and Omniglot datasets can be regarded as natural out-distribution sets when trying to classify MNIST digits.

However, it is not clear how to select a sufficiently representative set from the (possibly large) corpus of available datasets in order to properly train augmented CNNs.

We shed light on the selection of a representative natural out-distribution set by introducing a simple visualization metric.

Specifically, we deem a natural out-distribution set as representative for a given in-distribution task if it is misclassified uniformly over the in-distribution classes.

That is, if roughly an equal number of out-distribution samples are classified confidently as belonging to each of the in-distribution classes by the naive neural network.

Accordingly, to assess the appropriateness of out-distribution sets for a given task (or in-distribution set), we visualize the number of out-distribution samples that are misclassified to each of the in-distribution classes by using a histogram.

In other words, a natural out-distribution set which has a more uniform misclassification distribution over the in-distribution classes appears better suited for training an effective augmented CNN.In Fig. 2 , the uniformity characteristics of SVHN vs CIFAR-100 † as out-distribution sets for CIFAR-10, and LSUN vs DS-ImageNet † (i.e. Down Scaled ImageNet) for CIFAR-100 are shown 1 .

According to Fig. 2(a) , most of SVHN samples are misclassified into a limited number of CIFAR-10 classes (5 classes out of 10 classes), while CIFAR-100 † exhibits a relatively more uniform misclassifcation on CIFAR-10 classes.

Therefore, compared with SVHN, we consider CIFAR-100 as a more representative natural out-distribution set for CIFAR-10.

A full comparison of these two out-distribution sets according to their ability to control over-generalization can be found in TAB3 of the Appendix.

Similar behaviour can also be observed for LSUN vs DS-ImageNet as two out-distribution resources for training an augmented Resnet164 on CIFAR-100 (as in-distribution).

In this case DS-ImageNet † has a more uniform distribution when compared with LSUN.1 Throughout the paper, † indicates a modified out-distribution set by discarding the classes that have exact or semantic overlaps with the classes of its corresponding in-distribution set.

For example, the super-classes of vehicle from CIFAR-100 is removed due to their semantic overlap with automobile and truck classes of CIFAR-10.

Refer to Appendix A for detail information.

Algorithms to generate adversarial examples tend to produce results near (on margin) decision boundaries separating two classes .

Adding a set of diverse types of adversaries to a representative natural out-distribution set may further improve the rate of adversary identification by the augmented CNN.

But generating such a diverse set of adversarial examples for large-scale datasets is computationally expensive.

Furthermore, using only adversaries as dustbin training samples (without including a representative natural out-distribution set) cannot lead to an effective reduction of over-generalization (see FIG0 (b) and results in Sec. 3).Instead of generating adversarial examples for training, we propose an inexpensive and straightforward procedure for acquiring some samples around the decision boundaries.

To this end, we interpolate some pairs of correctly classified in-distribution samples from different classes.

An interpolated sample created from two samples with different classes aims to cover such regions (margins around decision boundaries) between two classes in order to assign them to out-distribution (dustbin) regions.

Formally speaking, consider a pair of input images from two different classes of a K-classification problem, i.e. DISPLAYFORM0 .., K}, where x j is the nearest neighbor of x i in the feature space of a CNN (its last convolution layer).

An interpolated sample x ∈ R D is generated by making a linear combination of the given pair in the input space, x = α x i + (1 − α) x j .

For all our experiments, we set α = 0.5.

Some interpolated samples can be seen in FIG2 for MNIST and CIFAR-10.

The reasons for finding the nearest neighbors in the feature space are twofold: computationally less expensive yet more accurate BID1 ) when compared to doing so in high-dimensional input space.

As an augmented CNN is trained in a end-to-end fashion, it allows learning of an extra sub-manifold corresponding to the added extra class (dustbin).

Thus, if the augmented CNN is trained properly on a representative out-distribution set, it is able to map a large variety of out-distribution sets onto its extra sub-manifold, whether or not they have been seen during training.

This should allow to learn a feature space that untangles the in-distribution set from the out-distribution samples.

This is in contrast to the feature space of its naive counterpart, where the in-distribution and out-distribution samples are likely to be mixed or placed near each other.

Moreover, a proper trained augmented CNN is surprisingly able to map a large portion of black-box adversaries onto its extra manifold, even though it is never trained on any adversaries.

Meanwhile some of the adversarial instances are mapped to their corresponding true class' sub-manifold.

Therefore, this leads to a more engaging classifier for many practical situations (real-world applications) as some of adversaries are classified into dustbin (equivalent to the rejection option) while some of remaining ones are correctly classified as their true class (particularly non-transferable adversaries attacks, see Sec.3).In Fig. 4 , we exhibit the feature spaces achieved from a naive CNN and its augmented counterpart for CIFAR-10 as an in-distribution task.

Note CIFAR-100 is used as the training set for the extra class of the augmented CNN.

As it can be visualized in Fig. 4 , the two out-distribution sets, including CIFAR-100 (green triangles) and Fast Gradient Sign (FGS) adversaries (shown with yellow triangles) are separated from CIFAR-10 samples in the feature space of the augmented CNN while they are mixed in the feature space of its naive counterpart.

Naive CNN Augmented CNN Figure 4 : Visualization of data distribution in last convolution layer (i.e., feature space) of an augmented CNN trained on CIFAR-10 and CIFAR-100 as in-distribution and out-distribution sets, respectively.

For visualization purposes, these feature spaces are reduced to 3D using PCA.

We conduct several experiments on three benchmarks, namely MNIST, CIFAR-10, and CIFAR-100 datasets, using three neural network architectures LeNet (LeCun et al., 1998) , VGG-16 BID24 , and ResNet164 BID9 .

To assess robustness of the augmented versions of these CNNs , we consider five well-known strong attack algorithms: Fast Gradient Sign (FGS) , Iterative FGS (I-FGS) BID19 , Targeted FGS (T-FGS) BID12 , DeepFool (Moosavi Dezfooli et al., 2016) , and C&W BID3 ) (see Appendix A.6 to learn about their hyper-parameter configurations).

Note that we evaluate performance using three metrics: 1) accuracy (Acc.), which captures the rate or percentage of samples classified correctly as their true associated label; 2) rejection rate (Rej.), to measure the rate of samples correctly classified as dustbin (equivalent to rejection option); and 3) error rate (Err.), which captures the rate of samples that are neither correctly classified nor rejected.

It is widely known that many of adversarial examples generated from a learning model (e.g., CNN) can be transferred to attack other victim models BID26 BID2 ) -such attacks are called transferable black-box attacks.

To evaluate robustness of the augmented CNNs on the aforementioned types of attacks generated in black-box setting, we generate adversarial samples corresponding to correctly classified clean test samples using a naive CNN, trained with different initial weights compared to the one under evaluation.

Moreover, in order to demonstrate the influence of using different out-distribution sets for training the extra class on identifying adversaries, we employ four different sources for acquiring dustbin training samples: 1) adversarial samples generated by I-FGS; 2) only interpolated in-distribution data; 3) only a representative natural out-distribution set (selected according our proposed metric); and 4) both interpolated samples along with a representative natural out-distribution set (selected according our proposed metric).To evaluate the generalization performance of the augmented CNNs on the in-distribution tasks, the in-distribution test accuracy rates are presented in TAB5 .

Compared to the naive CNNs, we observe a slight drop in test accuracy rates of their augmented counterparts (except for that trained on I-FGS adversaries) while, interestingly, having also the error rates (i.e., the number of wrong decisions) reduced, leading to less error in decision making.

This property can be highly beneficial for some security-sensitive applications, where making less error in some critical situations is vital.

For the augmented CNNs, rejection rate (i.e., assignments to dustbin) is reported in addition to accuracy (i.e., correct classifications) and error rates (i.e., misclassifications) 2 .

Comparing the augmented CNNs in TAB5 TAB5 : Results for black-box adversaries attacks on three classification tasks.

Values with * denotes best accuracy while boldface denotes lowest misclassification rate for each given dataset and attack method.

CNNs trained on a set of I-FGS adversaries can reject (classifying as dustbin) almost all test variants of FGS adversaries (i.e., FGS, I-FGS and T-FGS), however they fail to reject non-FGS variants of adversaries (e.g., C&W and DeepFool), as well as the natural out-distribution sets (see TAB4 of Appendix A).

Accordingly, we emphasize that using the samples drawn from the vicinity of decision boundaries such as I-FGS adversaries as a single training source for the extra dustbin class of augmented CNN can not effectively control over-generalization.

Contrary to I-FGS augmented CNN, augmented CNNs trained on a representative natural out-distribution set (selected according to our proposed metric) along with some interpolated samples consistently outperform their naive counterparts and the other augmented CNNs by achieving a drastic drop in error (misclassification) rates on all variants of adversaries, even though these augmented CNNs are not trained on any specific type of adversaries.

This illustrates that if an augmented CNN is trained on a representative outdistribution set along with some interpolated samples, it can efficiently reduce over-generalization, resulting in generally well-performing model in the case of adversaries and various natural out-distribution samples.

Due to space limitation, we place some results on the augmented CNNs trained with non-representative natural out-distribution sets in the Appendix A in TAB3 for illustrating the deficiency of such sets in controlling over-generalization.

To visualize and compare the classification regions in input space of our augmented CNN and its naive counterpart, we plot several church-windows (cross-sections) BID28 in FIG4 .

The x-axis of each window is the adversary direction achieved by FGS or DeepFool using the naive network.

For each adversary direction, we plot four windows by taking four random directions that are perpendicular to the given adversary direction (x-axis).

As it can be observed, the fooling classification regions (spanned by the adversary direction and one of its orthogonal random directions) of the naive CNNs are occupied by dustbin regions (indicated by orange) in their augmented counterparts.

White-box adversarial examples are generated by using directly the model on which they are applied.

We further evaluate the robustness of our augmented CNNs on different types of white-box attacks, using the same parameter configurations as with the black-box experiments.

For this purpose, we compute the percentage of visiting fooling classes (i.e., the classes different from dustbin and the true class associated to the clean samples) and the dustbin class when moving in the direction given by an attack method for a set of clean samples.

Note that for generating some authentic white-box adversaries by the augmented CNNs, the attack algorithm should avoid dustbin regions to preclude generation of useless adversaries (those already recognizable as dustbin by the augmented CNN).

In addition, the percentage of visiting the dustbin class when moving in a "legitimate" direction is also reported.

By this we mean moving from a given sample x to its nearest neighbor x from the same class in the direction of their convex combination (1 − ) x + x .

Results for legitimate directions are computed with varying ∈ [0.1, 0.5].To generate white-box adversaries using both a naive CNN and its augmented counterpart, MNIST and CIFAR-10 test sets are utilized.

As seen in Fig. 6 , adversaries generated for the augmented CNNs (trained on a representative natural out-distribution set) encounter more often the dustbin class than a fooling class, indicating that generation of white-box adversaries using the augmented CNNs becomes harder.

An adversarial algorithm needs to skip over some regions assigned to dustbin class, leading to a possible increase in the number of steps or the amount of distortions required for generating adversaries.

Moreover, by moving in legitimate directions, the augmented CNNs appear to remain largely in the current true classes.

The behavior of augmented CNNs is evaluated on several out-distribution sets across different indistribution tasks.

For each in-distribution task, we consider several natural out-distribution datasets, both seen and unseen during the training of the augmented CNN.

For comparison purposes, the rejection rates of two recent threshold-based approaches, including ODIN BID18 , and Table 2 : Comparison of augmented CNNs and threshold-based approaches on a range of natural out-distribution sets.

The size of input images of the datasets indicated by * are scaled to be consistent with their corresponding in-distribution set.

CIFAR-10 (gc) means gray-scaled and cropped version of CIFAR-10.Calibrated CNN BID17 are considered 3 .

These approaches attempt to identify and reject out-distribution samples according to a specific threshold on the calibrated predictive confidence scores.

For a fair comparison, the rejection rates (i.e. True Negative Rate) of these approaches are reported at the same True Positive Rates (TPR) as ours, where TPRs are considered 99%, 91%, and 95% for MNIST, CIFAR-10, and CIFAR-100, respectively.

4 .Moreover, for all ODIN experiments, we consider T = 1000 and ∈ {0, 5 × 10 −6 , 5 × 10 −5 , 5 × 10 −4 , 5 × 10 −3 , 1 × 10 −3 } is tuned for each pair of in-distribution and out-distribution validation sets such that the highest possible TNR at the specified TPR can be achieved.

For "calibrated CNN" approach, its hyper-parameter (β > 0), which controls the effect of having the calibrated (uniform) predictions on the synthesized out-distribution samples, is tuned such that training of the calibrated CNN can be converged on the given in-distribution training set.

We observe while the larger beta for MNIST and CIFAR-10 lead to better calibrated CNN on out-distribution samples, such a large beta for CIFAR-100 does not allow its training to converge.

Considering this trade-off between calibration and convergence, in our experiments, the values of β are regarded 1 and 0.01 for CIFAR-10/MNIST and CIFAR100, respectively.

Table 2 compares the rejection rates (i.e TNR) of ours with ODIN and "calibrated CNN" as well as the error rates of naive and the augmented CNNs (trained on a representative natural out-distribution set and interpolated samples), where the error rate measures the number of the out-distribution samples classified with confidence higher than 50% as one of the in-distribution classes.

These error rates by naive CNN aim to show the fact that a significant portion of out-distribution samples are confidently (confidence> 50%) misclassified by naive CNN.

As it can be seen in Table 2 , the augmented CNNs, which is trained on one single but representative natural out-distribution set (as well as interpolated samples), almost outperforms "calibrated CNN", which is trained on a set of synthetic out-distribution samples, and ODIN, which its hyper-parameter is tuned for each pair of in-distribution and outdistribution validation set.

It can demonstrate how controlling effectively over-generalization can lead to developing more robust CNNs in the presence of novel unseen out-distribution sets.

In this paper we bridge two issues of CNNs that were previously thought of as unrelated: susceptibility of naive CNNs to various types of adversarial examples and incorrect high confidence prediction for out-distribution samples.

We argue these two issues are connected through over-generalization.

We propose augmented CNNs as a simple yet effective solution for controlling over-generalization, when they are trained on an appropriate set of dustbin samples.

Through empirical evidence, we define an indicator for selecting an "appropriate" natural out-distribution set as training samples for dustbin class from among those available and show such selection plays a vital role for training effective augmented CNNs.

Through extensive experiments on several augmented CNNs in different settings, we demonstrate that reducing over-generalization can significantly reduce the misclassification error rates of CNNs on adversaries and out-distribution samples, simultaneously, while their accuracy rates on in-distribution samples are maintained.

Indeed, reducing over-generalization by such an end-to-end learning model (e.g., augmented CNNs) leads to learning more expressive feature space where these two categories of hostile samples (i.e., adversaries and out-distribution samples) are disentangled from in-distribution samples.

MNIST with NotMNIST MNIST consists of gray scale images of hand-written digits (0-9) and is made of 60k and 10k samples for training and testing, respectively.

NotMNIST dataset 5 , which involves 18,724 letters (A-J) printed with different font styles, is used as a source of out-distribution samples for MNIST.

Images of both MNIST and NotMNIST datasets have the same size (28 × 28 pixels), with all pixels scaled in [0, 1].

LeNet, the CNN model we used comprised three convolution layers of 32, 32, and 64 filters (5 × 5), respectively, and one Fully Connected (FC) layer with softmax activation function 6 .

In addition, dropout with p = 0.5 is used on the FC layer for regularization.

The augmented version of LeNet is trained with the 50k samples of MNIST, 10K randomly selected samples from NotMNIST for out-distribution samples and 15K interpolated samples (see Section2) generated from MNIST training samples.

The remaining samples from NotMNIST (≈8K) are used together with MNIST test samples to evaluate the augmented CNN.CIFAR-10 with CIFAR-100 † CIFAR-10 and CIFAR-100 represents low-resolution RGB images (32×32) of objects.

CIFAR-10 contains 50k training and 10k testing instances over 10 classes.

CIFAR-100 has the same characteristics except it is organized into 100 classes.

For the experiments with CIFAR-10, out-distribution samples are taken from CIFAR-100 † .

To avoid the semantic overlaps between the labels of CIFAR-10 and CIFAR-100, super-classes of CIFAR-100 conceptually similar to those of CIFAR-10 are ignored (i.e., vehicle 1, vehicle 2, medium-sized mammals, small mammals, and large carnivores excluded from CIFAR-100).

Pixels are scaled in [0, 1], and then normalized by subtracting the mean of the image of the CIFAR-10 training set.

VGG-16 BID24 ) is used as CNN architecture for CIFAR-10, which has has 13 convolution layers of 3 × 3 filters and three FC layers.

To train the augmented VGG-16, 15k samples are selected from CIFAR-100 † along with 15k interpolated samples from CIFAR-10 training set (both labeled as dustbin) and are appended to the CIFAR-10 training set.

CIFAR-100 with DS-ImageNet † Similar to CIFAR-10, training and test sets of CIFAR-100 contain 50K and 10K RGB images (32×32 pixels each).

As out-distribution samples for CIFAR-100, we utilized down-scaled version of ImageNet dataset (called DS-ImageNet) BID4 (images are scaled to 32 × 32).

To choose proper out-distribution samples for CIFAR-100, we utilized the samples from 62 classes of DS-ImageNet that have less conceptual overlap with CIFAR-100 labels.

For creating the training set for out-distribution dustbin class, samples from those 62 classes are taken from training set of DS-ImageNet.

Therefore, this training set has 79,856 samples in total, but we randomly selected 15K of them along with 15K interpolated samples to train our augmented CNNs.

We utilized validation set of DS-ImageNet containing 50K images as the test out-distribution task.

We use ResNet-164 BID9 to train an augmented CNN on CIFAR-100 (as in-distribution task).

For a given in-distribution dataset, there are many possible candidate out-distribution datasets for training an augmented CNN.

We argued in section 2 that an non-representative natural out-distribution set, can not effectively handle over-generalization.

According to our measurement, a non-representative natural out-distribution set is the one that are only miscalssified as a limited-number of in-distribution classes by a nive CNN.

Recall that, according to Fig. 2(a) , most of SVHN samples are misclassified into a limited number of CIFAR-10 classes (5 classes out of 10 classes) by the naive CNN, while CIFAR-100 † dataset exhibits a relatively more uniform misclassifcation on CIFAR-10 classes.

Therefore, compared to SVHN, we consider CIFAR-100 † as a more representative natural out-distribution set for CIFAR-10.

Similarly, for CIFAR-100, DS-ImageNet † dataset is more uniformly misclassified when compared with LSUN and is thus considered a more appropriate out-distribution dataset.

In TAB3 , we compare two types of out-distribution sets, representative vs non-representative across two classification tasks, and showing choosing a representative out-distribution set is a key factor for effectively reducing over-generalization such that a wide-range of other unseen out-distribution samples and adversarial examples can be confidently classified as dustbin (equivalent to rejection).In comparison to the augmented VGG used SVHN, the augmented VGG-16 trained on CIFAR-100 † as outdistribution training samples performs significantly better at rejecting both adversaries and unseen out-distribution samples TAB3 .

Similarly, when comparing two augmented Resnets (for CIFAR-100 as in-distribution), the one trained with LSUN as the source of out-distribution samples is less effective in reducing over-generalization when comparison to the other Resnet trained with DS-ImageNet † as the source of out-distribution samples.

TAB6 .

For obtaining ODIN results, the naive pre-trained models that used in TAB5 (having the same architecture as our augmented CNNs except the number of outputs) are considered and the optimal value for the hyper-parameters of ODIN, i.e. ∈ {0, 5 × 10 −6 , 5 × 10 −5 , 5 × 10 −4 , 5 × 10 −3 , 1 × 10 −3 } and T = 1000 are utilized.

The optimal value for is chosen separately for each set of black-box adversaries. (TPR) 91% and 95% for CIFAR-10 and CIFAR-100, respectively.

As it can be seen from TAB6 , the error rates of our augmented CNN on most of the various types of black-box adversaries are lower than ODIN due to its higher adversaries rejection (classifying as dustbin) rates.

For example, our method on I-FGS adversaries of CIFAR-100, which are highly transferable (it can be perceived from naive ResNet-164's low accuracy on I-FGS (i.e. 22.20% from FIG5 , we find that the over-generalization reduction leads to a more expressive feature space where all natural out-distribution samples along with many black-box adversarial examples are separated from in-distribution samples to be classified as belonging to the dustbin class.

Further, some adversarial instances are even placed very close to their corresponding true class, leading the augmented CNNs to classify them correctly.

Generally, an adversarial generation method can either be targeted or untargeted.

In targeted attacks, an adversary aims to generate an adversarial sample that makes a victim CNN misclassify it to an adversary selected target class (i.e., arg max F (x + δ) = y , where y is the targeted class and = y * the actual class).

In an untargeted attack, an adversary aims to make the victim CNN to simply misclassify perturbed image to a class other than the true label (i.e., arg max F (x + δ) = y * , where y * is the true class).

Here, we briefly explain some well-known targeted and untargeted attack algorithms.

Targeted Fast Gradient Sign (T-FGS) BID12 : This targeted attack method tends to modify a clean image x so that the loss function is minimized for a given pair of (x, y ), where target class y is different from the input's true label (y = y * ).

To this end, it uses the sign of gradient of loss function as follows:x adv = x − .sign(∇J (F (x, θ) , y )),where J(F (x, θ), y ) is the loss function and as the hyper-parameter controls the amount of distortion.

The transferability of T-FGS samples increases by utilizing larger at the cost of adding more distortions to the image.

Moreover, the untargeted variant of this method called FGS ) is as follows:x adv = x + .sign(∇J (F (x, θ) , y * )).Iterative Fast Gradient Sign (I-FGS) : This method actually is an iterative variant of Fast Gradient Sign (also called Projected Gradient Descent (PGD) BID19 ), where iteratively small amount of FGs perturbation is added by using a small value for .

To keep the perturbed sample in α-neighborhood of x, the achieved adversary sample in each iteration should be clipped.

Compared to FGS, I-FGS generates more optimal distortions.

DeepFool : This algorithm is an iterative but fast approach for creation of untargeted attacks with very small amount of perturbations.

Indeed, DeepFool generates sub-optimal perturbation for each sample where the perturbation is designed to transfer the clean sample across its nearest decision boundary.

Carlini Attack (C&W) BID3 : Unlike previous proposed methods which find the adversarial examples over loss functions of CNN, this method defines a different objective function which tends to optimize misclassification as follows:f (x ) = max(max{Z(x ) y − Z(x )y * }, −κ)Here Z(x) is the output of last fully connected (before softmax) layer and x is perturbed image x. Also κ denotes confidence parameter.

A larger value for κ leads the CNN to misclassify the input more confidently, however it also makes finding adversarial examples satisfying the condition (having high misclassification confidence) difficult.

Hyper-parameters of Attack Algorithms:

Each adversarial generation algorithm has a few hyperparameters as previously seen.

We provide details on the hyper-parameters used in our experimental evaluation in TAB7 .

To generate targeted Carlini attack (called C&W) BID3 , we used the authors' github code.

Due to large time complexity of C&W, we considered 100 randomly selected images for each dataset.

For each selected image, as was done in previous work BID29 , two targeted adversarial samples are generated, where the target classes are the least likely and second most likely classes according to the predictions provided by the underlying CNN.

Thus, in total 200 C&W adversarial examples are generated per dataset.

To increase transferability of C&W, we utilized κ = 20 for MNIST and κ = 10 for CIFAR-10.

For CIFAR-100, we used the same setting used for CIFAR-10 except for C&W, we used higher value for κ(= 20).

@highlight

Properly training CNNs with dustbin class increase their robustness to adversarial attacks and their capacity to deal with out-distribution samples.

@highlight

This paper proposes adding an additional label for detecting OOD samples and adversarial examples in CNN models.

@highlight

The paper proposes an additional class that incorporates natural out-distribution images and interpolated images for adversarial and out-distribution samples in CNNs