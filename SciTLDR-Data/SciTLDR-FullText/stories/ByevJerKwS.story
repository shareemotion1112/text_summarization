This paper studies \emph{model inversion attacks}, in which the access to a model is abused to infer information about the training data.

Since its first introduction by~\citet{fredrikson2014privacy}, such attacks have raised serious concerns given that training data usually contain sensitive information.

Thus far, successful model inversion attacks have only been demonstrated on simple models, such as linear regression and logistic regression.

Previous attempts to invert neural networks, even the ones with simple architectures, have failed to produce convincing results.

We present a novel attack method, termed the \emph{generative model inversion attack}, which can invert deep neural networks with high success rates.

Rather than reconstructing private training data from scratch, we leverage partial public information, which can be very generic, to learn a distributional prior via generative adversarial networks (GANs) and use it to guide the inversion process.

Moreover, we theoretically prove that a model's predictive power and its vulnerability to inversion attacks are indeed two sides of the same coin---highly predictive models are able to establish a strong correlation between features and labels, which coincides exactly with what an adversary exploits to mount the attacks.

Our experiments demonstrate that the proposed attack improves identification accuracy over the existing work by about $75\%$ for reconstructing face images from a state-of-the-art face recognition classifier.

We also show that differential privacy, in its canonical form, is of little avail to protect against our attacks.

Deep neural networks (DNNs) have been adopted in a wide range of applications, including computer vision, speech recognition, healthcare, among others.

The fact that many compelling applications of DNNs involve processing sensitive and proprietary datasets raised great concerns about privacy.

In particular, when machine learning (ML) algorithms are applied to private training data, the resulting models may unintentionally leak information about training data through their output (i.e., black-box attack) or their parameters (i.e., white-box attack).

A concrete example of privacy attacks is model inversion (MI) attacks, which aim to reconstruct sensitive features of training data by taking advantage of their correlation with the model output.

Algorithmically, MI attacks are implemented as an optimization problem seeking for the sensitive feature value that achieves the maximum likelihood under the target model.

The first MI attack was proposed in the context of genomic privacy (Fredrikson et al., 2014) , where the authors showed that adversarial access to a linear regression model for personalized medicine can be abused to infer private genomic attributes about individuals in the training dataset.

Recent work (Fredrikson et al., 2015) extended MI attacks to other settings, e.g., recovering an image of a person from a face recognition model given just their name, and other target models, e.g., logistic regression and decision trees.

Thus far, effective MI attacks have only been demonstrated on the aforementioned simple models.

It remains an open question whether it is possible to launch the attacks against a DNN and reconstruct its private training data.

The challenges of inverting DNNs arise from the intractability and ill-posedness of the underlying attack optimization problem.

For neural networks, even the ones with one hidden layer, the corresponding attack optimization becomes a non-convex problem; solving it via gradient descent methods may easily stuck in local minima, which leads to poor attack performance.

Moreover, in the attack scenarios where the target model is a DNN (e.g., attacking face recognition models), the sensitive features (face images) to be recovered often lie in a high-dimensional, continuous data space.

Directly optimizing over the high-dimensional space without any constraints may generate unrealistic features lacking semantic information (See Figure 1) .

Figure 1 : Reconstruction of the individual on the left by attacking three face recognition models (logistic regression, one-hidden-layer and twohidden-layer neural network) using the existing attack algorithm in (Fredrikson et al., 2015) In this paper, we focus on image data and propose a simple yet effective attack method, termed the generative model inversion (GMI) attack, which can invert DNNs and synthesize private training data with high fidelity.

The key observation supporting our approach is that it is arguably easy to obtain information about the general data distribution, especially for the image case.

For example, against a face recognition classifier, the adversary could randomly crawl facial images from the Internet without knowing the private training data.

We find these datasets, although may not contain the target individuals, still provide rich knowledge about how a face image might be structured; extraction and proper formulation of such prior knowledge will help regularize the originally ill-posed inversion problem.

We also move beyond specific attack algorithms and explore the fundamental reasons for a model's susceptibility to inversion attacks.

We show that the vulnerability is unavoidable for highly predictive models, since these models are able to establish a strong correlation between features and labels, which coincides exactly with what an adversary exploits to mount MI attacks.

Our contributions can be summarized as follows: (1) We propose to use generative models to learn an informative prior from public datasets so as to regularize the ill-posed inversion problem.

(2) We propose an end-to-end GMI attack algorithm based on GANs, which can reveal private training data of DNNs with high fidelity.

(3) We present a theoretical result that uncovers the fundamental connection between a model's predictive power and its susceptibility to general MI attacks and empirically validate it.

(4) We conduct extensive experiments to demonstrate the performance of the proposed attack.

Experiment code is publicly available at https://tinyurl.com/yxbnjk4s.

Related Work Privacy attacks against ML models consist of methods that aim to reveal some aspects of training data.

Of particular interest are membership attacks and MI attacks.

Membership attacks aim to determine whether a given individual's data is used in training the model (Shokri et al., 2017) .

MI attacks, on the other hand, aim to reconstruct the features corresponding to specific target labels.

In parallel to the emergence of various privacy attack methods, there is a line work that formalizes the privacy notion and develops defenses with formal and provable privacy guarantees.

One dominate definition of privacy is differential privacy (DP), which carefully randomizes an algorithm so that its output does not to depend too much on any individuals' data (Dwork et al., 2014) .

In the context of ML algorithms, DP guarantees protect against attempts to infer whether a data record is included in the training set from the trained model (Abadi et al., 2016) .

By definition, DP limits the success rate of membership attacks.

However, it does not explicitly protect attribute privacy, which is the target of MI attacks (Fredrikson et al., 2014) .

The first MI attack was demonstrated in (Fredrikson et al., 2014) , where the authors presented an algorithm to recover genetic markers given the linear regression that uses them as input features, the response of the model, as well as other non-sensitive features of the input.

Hidano et al. (2017) proposed a algorithm that allows MI attacks to be carried out without the knowledge of non-sensitive features by poisoning training data properly.

Despite the generality of the algorithmic frameworks proposed in the above two papers, the evaluation of the attacks is only limited to linear models.

Fredrikson et al. (2015) discussed the application of MI attacks to more complex models including some shallow neural networks in the context of face recognition.

Although the attack can reconstruct face images with identification rates much higher than random guessing, the recovered faces are indeed blurry and hardly recognizable.

Moreover, the quality of reconstruction tends to degrade for more complex architectures.

Yang et al. (2019b) proposed to train a separate network that swaps the input and output of the target network to perform MI attacks.

The inversion model can be trained with black-box accesses to the target model.

However, their approach cannot directly be benefited from the white-box setting.

Moreover, several recent papers started to formalize MI attacks and study the factors that affect a model's vulnerability from a theoretical viewpoint.

For instance, Wu et al. (2016) characterized model invertibility for Boolean functions using the concept of influence from Boolean analysis; Yeom et al. (2018) formalized the risk that the model poses specifically to individuals in the training data and shows that the risk increases with the degree of overfitting of the model.

However, their theory assumed that the adversary has access to the join distribution of private feature and label, which is overly strong for many attack scenarios.

Our theory does not rely on this assumption and better supports the experimental findings.

An overview of our GMI attack is illustrated in Figure 2 .

In this section, we will first discuss the threat model and then present our attack method in details.

In traditional MI attacks, an adversary, given a model trained to predict specific labels, uses it to make predictions of sensitive features used during training.

Throughout the paper, we will refer to the model subject to attacks as the target network.

We will use face recognition classifiers as a running example for the target network.

Face recognition classifiers label an image containing a face with an identifier corresponding to the individual depicted in the image.

We assume that the adversary employs an inference technique to discover the face image x for some specific identity y output by the classifier f .

Following the canonical setup of MI attacks, we assume that the adversary has access to the target network f .

In addition to f , the adversary may also have access to some auxiliary knowledge that facilitates his inference.

Possible Auxiliary Knowledge Examples of auxiliary knowledge could be a blurred or corrupted image which only contains nonsenstive information, such as background pixels in a face image.

This auxiliary knowledge might be easy to obtain, as blurring and corruption are often applied to protect anonymity of individuals in public datasets (Carrell et al., 2012; Li et al., 2019) .

The setup of MI attacks on images resembles the widely studied image inpainting tasks in computer vision, which also try to fill missing pixels of an image.

The difference is, however, in the goal of the two.

MI attacks try to fill the sensitive features associated with a specific identity in the training set.

In contrast, image inpainting tasks only aim to synthesize visually realistic and semantically plausible pixels for the missing regions; whether the synthesized pixels are consistent with a specific identity is beyond the scope.

Despite the difference, our approach to MI attacks leverages some training strategies from the venerable line of work on image inpainting (Yeh et al., 2017; Iizuka et al., 2017; Yang et al., 2019a) and significantly improves the recognizability of the reconstructed images over the existing attack methods.

To realistically reconstruct missing sensitive regions in an image, our approach utilizes the generator G and the discriminator D, all of which are trained with public data.

After training, we aim to find the latent vector??? that achieves highest likelihood under the target network while being constrained to the data manifold learned by G. However, if not properly designed, the generator may not allow the target network to easily distinguish between different latent vectors.

For instance, in extreme cases, if the generated images of all latent vectors collapse to the same point in the feature space of the target network, then there is no hope to identify which one is more likely to appear in its private training set of the target network.

To address this issue, we present a simple yet effective loss term to promote the diversity of the data manifold learned by G when projected to the target network's feature space.

Specifically, our reconstruction process consists of two stages: (1) Public knowledge distillation, in which we train the generator and the discriminators on public datasets in order to encourage the generator to generate realistic-looking images.

The public datasets can be unlabeled and have no identity overlapping with the private dataset.

(2) Secret revelation, in which we make use of the generator obtained from the first stage and solve an optimization problem to recover the missing sensitive regions in an image.

For the first stage, we leverage the canonical Wasserstein-GAN (Arjovsky et al., 2017) training loss.

The loss function is adapted to the two discriminators for our case:

In addition, inspired by Yang et al. (2019a) , we introduce a diversity loss term that promotes the diversity of the images synthesized by G when projected to the target network's feature space.

Let F denote the feature extractor of the target network.

The diversity loss can thus be expressed as

As discussed above, larger diversity will facilitate the targeted network to discern the generated image that is most likely to appear in its private training set.

Our full objective for public knowledge distillation can be written as

In the secret revelation stage, we solve the following optimization to find the latent vector that generates an image achieving the maximum likelihood under the target network while remaining

where the prior loss L prior (z) penalizes unrealistic images and the identity loss L id (z) encourages the generated images to have likelihood under the targeted network.

They are defined, respectively, by

where C(G(z)) represent the probability of G(z) output by the target network.

For a fixed data point (x, y), we can measure the performance of a model f for predicting the label y of feature x using the log likelihood log p f (y|x).

It is known that maximizing the log likelihood is equivalent to minimizing the cross entropy loss-one of the most commonly used loss function for training DNNs.

Thus, throughout the following analysis, we will focus on the log likelihood as a model performance measure.

Now, suppose that (X, Y ) is drawn from an unknown data distribution p(X, Y ).

Moreover, X = (X s , X ns ), where X s and X ns denote the sensitive and non-sensitive part of the feature, respectively.

We can define the predictive power of the sensitive feature X s under the model f (or equivalently, the predictive power of model f using X s ) as the change of model performance when excluding it from the input, i.e.,

.

Similarly, we define the predictive power of the sensitive feature given a specific class y and nonsensitive feature x ns as

We now consider the measure for the MI attack performance.

Recall the goal of the adversary is to guess the value of x s given its corresponding label y, the model f , and some auxiliary knowledge x ns .

The best attack outcome is the recovery of the entire posterior distribution of the sensitive feature, i.e., p(X s |y, x ns ).

However, due to the incompleteness of the information available to the adversary, the best possible attack result that adversary can achieve under the attack model can be captured by p f (X s |y, x ns ) ??? p f (y|X s , x ns )p(X s |x ns ), assuming that the adversary can have a fairly good estimate of p(X s |x ns ).

Such estimate can be obtained by, for example, learning from public datasets using the method in Section 2.2.

Although MI attack algorithms often output a single feature vector as the attack result, these algorithms can be adapted to output a feature distribution instead of a single point by randomizing the starting guess of the feature.

Thus, it is natural to measure the MI attack performance in terms of the similarity between p(X s |y, x ns ) and p f (X s |y, x ns ).

The next theorem indicates that the vulnerability to MI attacks is unavoidable if the sensitive features are highly predictive under the model.

When stating the theorem, we use the negative KL-divergence S KL (??||??) to measure the similarity between two distributions.

Theorem 1.

Let f 1 and f 2 be two models such that for any fixed label

We omit the proof of the theorem to the supplementary material.

Intuitively, highly predictive models are able to build a strong correlation between features and labels, which coincides exactly with what an adversary exploits to launch MI attacks; hence, more predictive power inevitably leads to higher attack performance.

In Yeom et al. (2018) , it is argued that a model is more vulnerable to MI attacks if it overfits data to a greater degree.

Their result is seemingly contradictory with ours, because fixing the training performance, more overfitting implies that the model has less predictive power.

However, the assumption underlying their result is fundamentally different from ours, which leads to the disparities.

The result in Yeom et al. (2018) assumes that the adversary has access to the joint distribution p(X s , X ns , Y ) that the private training data is drawn from and their setup of the goal of the MI attack is to learn the sensitive feature associated with a given label in a specific training dataset.

By contrast, our formulation of MI attacks is to learn about private feature distribution p(X s |y, x ns ) for a given label y from the model parameters.

We do not assume that the adversary has the prior knowledge of p(X s , X ns , Y ), as it is a overly strong assumption for our formulation-the adversary can easily obtain p(X s |y, x ns ) for any labels and any values of non-sensitive features when having access to the joint distribution.

Dataset We evalaute our method using three datasets: (1) the MNIST handwritten digit data (MNIST), (2) the Chest X-ray Database ) (ChestX-ray8), and (3) the CelebFaces Attributes Dataset (CelebA) containing 202,599 face images of 10,177 identities with coarse alignment.

We crop the images at the center and resize them to 64??64 so as to remove most background.

Protocol We split each dataset into two disjoint parts: one part used as the private dataset to train the target network and the other as a public dataset for prior knowledge distillation.

The public data, throughout the experiments, do not have class intersection with the private training data of the target network.

Therefore, the public dataset in our experiment only helps the adversary to gain knowledge about features generic to all classes and does not provide information about private, class-specific features for training the target network.

This ensures the fairness of the comparison with the existing MI attack (Fredrikson et al., 2015) .

Models We implement several different target networks with varied complexities.

For all the adapted networks, we modify the FC-layer to fit in our task.

For digit classification on MNIST, our target network consists of 3 convolutional layers and 2 pooling layers.

For the disease prediction on ChestX-ray8, we use ResNet-18 adapted from (He et al., 2015) as our target network.

For the face recognition tasks on CelebA, we use the following networks: (1) VGG16 adapted from (Simonyan and Zisserman, 2014) ; (2)ResNet-152 adapted from (He et al., 2015) ; (3) face.eoLVe adapted from the state-of-the-art face recognition network (Cheng et al., 2017) .

Training We split the private dataset defined above into training set (90%) and test set (10%) and use the SGD optimizer with learning rate 10 ???2 , batch size 64, momentum 0.9 and weight decay 10 ???4 to train these networks.

To train the GAN in the first stage of our attack pipeline, we set ?? d = 0.5 and use the Adam optimizer with the learning rate 0.004, batch size 64, ?? 1 = 0.5, and ?? 2 = 0.999 (Kingma and Ba, 2014) .

In the second stage, we set ?? i = 100 and use the SGD optimizer to optimize the latent vector z with the learning rate 0.01, batch size 64 and momentum 0.9.

z is drawn from a zero-mean unit-variance Gaussian distribution.

We randomly initialize z for 5 times and optimize each round for 1500 iterations.

We choose the solution with the lowest identity loss as our final latent vector.

Evaluating the success of MI attacks requires to assess whether the recovered image exposes the private information about a target individual.

Previous works analyzed the attack performance mainly qualitatively by visual inspection.

Herein, we introduce four metrics which allow to quantitatively judge the MI attack efficacy and perform evaluation at a large scale.

Peak Signal-to-Noise Ratio (PSNR) PSNR is the ratio of an image's maximum squared pixel fluctuation over the mean squared error between the target image and the reconstructed image Hore and Ziou (2010) .

PSNR measures the pixel-wise similarity between two images.

The higher the PSNR, the better the quality of the reconstructed image.

However, oftentimes, the reconstructed image may still reveal identity information even though it is not close to the target image pixel-wise.

For instance, a recovered face with different translation, scale and rotation from the target image will still incur privacy loss.

This necessitates the need for the following metrics that can evaluate the similarity between the reconstructed and the target image at a semantic level.

Attack Accuracy (Attack Acc) We build an evaluation classifier that predicts the identity based on the input reconstructed image.

If the evaluation classifier achieves high accuracy, the reconstructed image is considered to expose private information about the target individual.

The evaluation classifier should be different from the target network because the reconstructed images may incorporate features that overfit the target network while being semantically meaningless.

Moreover, the evaluation classifier should be highly performant.

For the reasons above, we adopt the state-of-the-art architecture in each task as the evaluation classifier.

For MNIST, our evaluation network consists of 5 convolutional layers and 2 pooling layers.

For ChestX-ray8, we adapt VGG-19 from (Simonyan and Zisserman, 2014) as our evaluation network.

For CeleA, we use the model in (Cheng et al., 2017) for the evaluation classifier.

We first pretrain it on the MS-Celeb-1M (Guo et al., 2016) and then fine tune on the identities in the training set of the target network.

The resulting evaluation classifier can achieve 96% accuracy on these identities.

Feature Distance (Feat Dist) Feat Dist measures the l 2 feature distance between the reconstructed image and the centroid of the target class.

The feature space is taken to be the output of the penultimate layer of the evaluation network.

K-Nearest Neighbor Distance (KNN Dist) KNN Dist looks at the shortest distance from the reconstructed image to the target class.

We identify the closest data point to the reconstructed image in the training set and output their distance.

The distance is measured by the l 2 distance between the two points in the feature space of the evaluation classifier.

Figure 3: Qualitative comparison of the proposed GMI attack with the existing MI attack (EMI), the pure image inpainting method (PII).

The ground truth target image is shown in 1st col.

We compare our approach with two baselines: (1) Existing model inversion attack (EMI), which implements the algorithm in (Fredrikson et al., 2015) .

For this algorithm, the adversary only exploits the identity loss for image reconstruction and return the pixel values that minimize the the identity loss; (2) Pure image inpainting (PII), which minimizes the W-GAN loss and performs image recovery based on the information completely from the public dataset.

For CelebA, the private set comprises 21,152 images of 1000 identities and samples from the rest are used as a public dataset.

We evaluate the attack performance in the three settings: (1) the attacker does not have any auxiliary knowledge about the private image, in which case he will recover the image from scratch; (2) the attacker has access to a blurred version of the private image and his goal is to deblur the image; (3) the attacker has access to a corrupted version of the private image wherein the sensitive, identity-revealing features (e.g., nose, mouth, etc) are blocked.

Table 1 compares the performance of our proposed GMI attack against EMI for different network architectures.

We can see that the EMI works poorly on the deep nets and achieve around zero attack accuracy.

GMI is much more effective than EMI.

Particularly, our method improves the accuracy of the attack against the state-of-the-art face.evoLVe classifier over the existing MI attack by 75% in terms of Top-5 attack accuracy.

Also, note that models that are more sophisticated and have more predictive power are more susceptible to attacks.

We will examine this phenomenon in more details in Section 4.3.3.

We now discuss the case where the attacker has access to some auxilliary knowledge in terms of blurred or partially blocked images.

For the latter, we consider two types of masks-center and face "T", illustrated by the second column of Figure 3 (c) and (d), respectively.

The center mask blocks the central part of the face and hides most of the identity-revealing features, such as eyes and nose, while the face T mask is designed to obstruct all private features in a face image.

Table 2 shows that our method consistently outperforms the two baselines discussed above.

Since the existing MI attack does not exploit any prior information, the inversion optimization problem is extremely ill-posed and performing gradient descent ends up at some visually meaningless local minimum, as illustrated by Figure 3 .

Interestingly, despite having the meaningless patterns, these images can all be classified correctly into the target label by the target network.

Hence, the existing MI attack tends to generate "adversarial examples" that can fool the target network but does not exhibit any recognizable features of the private data.

Figure 3 also compares our results with PII, which is completely based on the information from the public dataset to recover the private image.

We can see that although PII leads to realistic recoveries, the reconstructed images do not present the same identity features as the target images.

This can be further corroborated by the quantitative results in Table 2 .

Note that the attacks are more effective for the center mask than the face T mask.

This is because the face T mask we designed completely hides the identity revealing features on the face while the center mask may still expose the mouth information.

We have seen that distilling prior knowledge and properly incorporating it into the attack algorithm are important to the success of MI attacks.

In our proposed method, the prior knowledge is gleaned from public datasets through GAN.

We now evaluate the impact of public datasets on the attack performance.

We first consider the case where the public data is from the same distribution as the private data and study how the size of the public data affects the attack performance.

We change the size ratio (1:1, 1:4, 1:6, 1:10) of the public over the private data by varying the number of identities in the public dataset (1000, 250, 160, 100).

As shown in Table 3 , the attack performance varies by less than 7% when shrinking the public data size by 10 times.

Moreover, we study the effect of the distribution shift between the public and private data on the attack performance.

We train the GAN on the PubFig83 dataset, which contains 13,600 images with 83 identities, and attack the target network trained on CelebA. There are more faces with sunglasses in PubFig83 than CelebA, which makes it harder to distill generic face information.

Without any pre-processing, the attack accuracy drops by more than 20% despite still outperforming the existing MI attack by a large margin.

To further improve the reconstruction quality, we detect landmarks in the face images, rotate the images such that the eyes lie on a horizontal line, and crop the faces to remove the background.

These pre-processing steps make the public datasets better present the face information, thus improving the attack accuracy significantly.

We perform experiments to validate the connection between predictive power and the vulnerability to MI attacks.

We measure the predictive power of sensitive feature under a model using the difference of model testing accuracy based on all features and just non-sensitive features.

We consider the following different ways to construct models with increasing feature predictive powers, namely, enlarging the training size per class, adding dropout regularization, and performing batch normalization.

For the sake of efficiency, we slightly modify the proposed method in Section 2.2 in order to avert re-training GANs for different architectures.

Specifically, we exclude the diversity loss from the attack pipeline so that multiple architectures can share the same GAN for prior knowledge distillation.

Figure 4 shows that, in general, the attack performance will be better for models with higher feature predictive powers.

Moreover, this trend is consistent across different architectures.

Figure 5: Visualization of the recovered input images by the GMI and the EMI attack.

For MNIST, we use all 34265 images with labels 5, 6, 7, 8, 9 as private set, and the rest of 35725 images with labels 0, 1, 2, 3, 4 as a public dataset.

Note that the labels in the private and public data have no overlaps.

We augment the public data by training an autoencoder and interpolating in the latent space.

Our GMI attack is compared with the baseline in Table 4 .

We omit the PII baseline because the public and private set defined in this experiment are rather disparate and the PII essentially produce results close to random guesses.

We can see from the table that the performance of GMI is significantly better than the EMI.

Examples of the recovered images with both attacks are compared in Figure 5 .

For ChestX-ray8, we use 10000 images of seven classes as the private data and the other 10000 with different labels as public data.

The GMI and EMI attack are compared in Table 5 .

Again, the GMI attack outperforms the EMI attack by a large margin.

We investigate the implications of DP for MI attacks. ( , ??)-DP is ensured by adding Gaussian noise to clipped gradients in each training iteration Abadi et al. (2016) .

We find it challenging to produce useful face recognition models with DP guarantees due to the complexity of the task.

Therefore, we turn to a simpler dataset, MNIST, which is commonly used in differential private ML studies.

We set ?? = 10 ???5 and vary the noise scale to obtain target networks with different .

The attack performance against these target networks and their utility are illustrated in Figure 4 (d) .

Since the attack accuracy of the GMI attack on differentially private models is higher than that of PII which fills missing regions completely based on the public data, it is clear that the GMI attack can expose private information from differentially private models, even with stringent privacy guarantees, like = 0.1.

Moreover, varying differential privacy budgets helps little to protect against the GMI attack; sometimes, more privacy budgets even improve the attack performance (e.g., changing from 1 to 0.1).

This is because DP, in its canonical form, only hides the presence of a single instance in the training set.

Limiting the learning of specific individuals may facilitate the learning of generic features of a class, which, in turn, helps to stage MI attacks.

In this paper, we present a generative approach to MI attacks, which can achieve the-state-of-the-art success rates for attacking the DNNs with high-dimensional input data.

The idea of our approach is to extract generic knowledge from public datasets via GAN and use it to regularize the inversion problem.

Our experimental results show that our proposed attack is highly performant even when the public datasets (1) do not include the identities that the adversary aims to recover, (2) are unlabeled, (3) have small sizes, (4) come from a different distribution from the private data.

We also provide theoretical analysis showing the fundamental connection between a model's predictive power and its vulnerability to inversion attacks.

For future work, we are interested in extending the attack to the black-box setting and studying effective defenses against MI attacks.

A PROOF OF THEOREM 1 Theorem 2.

Let f 1 and f 2 are two models such that for any fixed label y ??? Y, U f1 (x ns , y) ??? U f2 (x ns , y).

Then, S KL (p(X s |y, x ns )||p f1 (X s |y, x ns )) ??? S KL (p(X s |y, x ns )||p f2 (X s |y, x ns )).

Proof.

We can expand the KL divergence D KL (p(X s |y, x ns )||p f1 (X s |y, x ns ) as follows.

Thus,

B EXPERIMENTAL DETAILS B.1 NETWORK ARCHITECTURE

The detailed architectures for the two encoders, the decoder of the generator, the local discriminator, and the global discriminator are presented in Table 6, Table 7, Table 8 , Table 9 , and Table 10 , respectively.

(1) LeNet adapted from (Lecun et al., 1998) , which has three convolutional layers, two max pooling layers and one FC layer; (2) SimpleCNN, which has five convolutional layers, each followed by a batch normalization layer and a leaky ReLU layer; (3) SoftmaxNet, which has only one FC layer.

We split the MNIST dataset into the private set used for training target networks with digits 0 ??? 4 and the public set used for distilling prior knowledge with digits 5 ??? 9.

The target network is implemented as a Multilayer Perceptron with 2 hidden layers, which have 512 and 256 neurons, respectively.

The evaluation classifier is a convulutional neural network with three convolution layers, followed by two fully-connected layers.

It is trained on the entire MNIST training set and can achieve 99.2% accuracy on the MNIST test set.

Differential privacy of target networks is guaranteed by adding Gaussian noise to each stochastic gradient descent step.

We use the moment accounting technique to keep track of the privacy budget spent during training (Abadi et al., 2016) .

During the training of the target networks, we set the batch size to be 256.

We fix the number of epochs to be 40 and clip the L2 norm of per-sample gradient to be bounded by 1.5.

We set the ratio between the noise scale and the gradient clipping threshold to be 0, 0.694, 0.92, 3, 28, respectively, to obtain the target networks with ?? = ???, 9.89, 4.94, 0.98, 0.10 when ?? = 10 ???5 .

For model with ?? = 0.1, we use the SGD with a small learning rate 0.01 to ensure stable convergence; otherwise, we set the learning rate to be 0.1.

The architecture of the generator in Section B.1 is tailored to the MNIST dataset.

We reduce the number of input channels, change the size of kernels, and modify the layers of discriminators to be compatible with the shape of the MNIST data.

To train the GAN in the first stage of our GMI attack, we set the batch size to be 64 and use the Adam optimizer with the learning rate 0.004, ?? 1 = 0.5, and ?? 2 = 0.999 (Kingma and Ba, 2014) .

For the second stage, we set the batch size to be 64 and use the SGD with the Nesterov momentum that has the learning rate 0.01 and momentum 0.9.

The optimization is performed for 1500 iterations.

The center mask depicted in the main text is used to block the central part of digits.

We report the attack accuracy averaged across 640 randomly sampled images from the private set and 5 random initializations of the latent vector for each sampled image.

@highlight

We develop a privacy attack that can recover the sensitive input data of a deep net from its output