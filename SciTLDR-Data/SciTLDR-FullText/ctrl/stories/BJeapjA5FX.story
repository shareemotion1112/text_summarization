We introduce a novel geometric perspective and unsupervised model augmentation framework for transforming traditional deep (convolutional) neural networks into adversarially robust classifiers.

Class-conditional probability densities based on Bayesian nonparametric mixtures of factor analyzers (BNP-MFA) over the input space are used to design soft decision labels for feature to label isometry.

Classconditional distributions over features are also learned using BNP-MFA to develop plug-in maximum a posterior (MAP) classifiers to replace the traditional multinomial logistic softmax classification layers.

This novel unsupervised augmented framework, which we call geometrically robust networks (GRN), is applied to CIFAR-10, CIFAR-100, and to Radio-ML (a time series dataset for radio modulation recognition).

We demonstrate the robustness of GRN models to adversarial attacks from fast gradient sign method, Carlini-Wagner, and projected gradient descent.

DeepConvNets are already prevalent in speech, vision, self-driving cars, biometrics, and robotics.

However, they possess discontinuities that are easy targets for attacks as evidenced in dozens of papers (see BID7 BID15 and references therein).

Adversarial images can be made to be robust to translation, scale, and rotation BID0 .

Adversarial attacks have also been applied to deep reinforcement learning BID9 BID10 and speech recognition BID3 .

In this work we will also consider attacks on automatic modulation recognition using deep convolutional networks BID17 .

Previous work in creating adversarially robust deep neural network classifiers includes robust optimization with saddle point formulations BID13 , adversarial training (see e.g., BID11 ), ensemble adversarial training BID24 , defensive distillation BID19 , and use of detector-reformer networks BID14 .

Defensive distillation has been found to be an insufficient defense BID1 and MagNet of BID14 was also shown to be defeatable in BID4 .

A summary of the attacks and defenses from the NIPS 2017 competition on adversarial attack and defense can be found in .In this paper we propose a statistical geometric model augmentation approach to designing robust neural networks.

We argue that signal representations involving projections onto lower-dimensional subspaces lower mean square error distortion.

We implement a statistical union of subspaces learned using a mixture of factor analyzers to create the auxiliary signal space structural information that neural networks can use to improve robustness.

We use the geometry of the input space to create unsupervised soft probabilistic decision labels to replace traditional hard one-hot encoded label vectors.

We also use the geometry of the feature space (after soft-decision supervised training) to create accurate class-conditional probability density estimates for MAP classifiers (to replace neural network classification layers).

We call this unsupervised geometric augmentation framework geometrically robust networks (GRN).

The main contributions of this paper are:1.

Geometric analysis of problems with current neural networks.2.

A novel soft decision label coding framework using unsupervised statistical-geometric union of subspace learning.3.

Maximum a posteriori classification framework based on class-conditional feature vector density estimation.

The rest of this paper is organized as follows.

In Section 2 we analyze neural networks from a geometric vantage point and recommend solution pathways for overcoming adversarial brittleness.

In Section 3 we describe the full details of the proposed geometrically robust network design framework.

We give experimental results on two datasets and three attacks in Section 4 and conclude in Section 5.

A deep (convolutional) neural network is a nested nonlinear function approximator that we can write as DISPLAYFORM0 where in our notation ?? c denotes parameters (weights and biases) associated with the classification stages c = 1, 2, ..., C, ?? f denotes parameters associated with the feature extraction stages f = 1, 2, ..., F , and the parameters are nested unions as ?? l ??? ?? l???1 culminating in ??. In principle an ideal function g * which could be resilient to bounded adversarial noise is guaranteed to exist by the universality theorem for neural networks BID6 , so this drives an investigation into what is making current architectures brittle.

The cross entropy loss objective function typically used to train function approximator DISPLAYFORM0 with label vectors y(x i ) has the form: DISPLAYFORM1 For hard decision labels, y(x i ) is an indicator vector (i.e. one-hot encoding), and (2) collapses to argmax ?? n ???1 i???{1,...,n} log g t(xi),?? (x i ) where t(x i ) is the element in indicator vector y(x i ) that is equal to one for the i th sample x i .

Following the analysis in BID19 , this means the stochastic gradient descent training algorithm minimizing (2) will inherently constrain the weights ?? c in the final classification layer(s) of (1) to try to output zeros for elements not corresponding to the correct class.

This artificially contrains the network to be overconfident and introduces brittleness.

There is also a geometric argument against one-hot encoding.

The mapping from feature space to label space is a surjective map since the mapping to hard decisions reduces the set size of the codomain to be equal to the number of classes, and the number of unique features will generally be larger than the number of unique classes.

This prevents the formation of an injective map from feature space to estimated label space.

If we can enlarge the set of labels to be infinite (i.e. soft decisions), then we allow room for an injective map to be learned from feature to estimated label space.

The resulting bijection (albeit a nonlinear bijection) then opens the door for distance preserving maps (isometries) which can guarantee that small distances between points in feature space remain small distances in the label space.

This kind of isometry is exactly what we need to be able to have adversarially robust networks.

Letting ?? 2 denote L 2 -norm, P (??) a projection matrix, P ??? (??) the orthogonal complement, x a natural input, and x adv the adversarially perturbed input, we know that DISPLAYFORM0 2 by Pythagorean theorem.

If the data x is well approximated by an information-preserving projection into another subspace, then we can reduce distortion of the adversarial input in the projected latent space.

If a network can be made to exploit knowledge of latent spaces with distortion-reducing representations of the data, then the overall classification performance would be less sensitive to adversarial perturbations.

A density estimate built upon this geometrical structure would then implicitly capture projected data representations and ultimately minimize the The signal data (either input observations to a neural network or feature vectors learned from the network) are modeled as small error displacements from a low dimensional linear subspace.

These union of subspaces can be learned statistically using a mixture of factor analyzers.

Since the number of subspaces and dimensionality of each subspace are unknown a priori, we must use a Bayesian nonparametric model.

label space deviation.

The vast majority of current deep neural network models make no use of geometrical-statistical models of the data and are solely supervised learning on labeled inputs.

We must use unsupervised learning to learn the latent manifold or union of subspaces topology to assist the supervised learning piece.

The latent structure of the data can be captured in both the input space and feature space as we will do in this study.

Here we briefly introduce the union of subspaces (UoS) model for modeling inputs and features.

To illustrate this, we take a vectorized signal segment x as shown in FIG1 (b) as a D-dimensional point living close to a union of T linear subspaces DISPLAYFORM1 where DISPLAYFORM2 The matrix A t = [a 1,t , a 2,t , ..., a dt,t ] is the matrix of basis vectors centered at ?? t for subspace index t, w is the coordinate of x at t, and t ??? N (0, ?? 2 I) is the modeling error.

The subspace coordinates w i and the closest subspace index t(i) are the latent variables for observation x i .

In FIG1 , we show the signal vector x, subspace offset vector ?? t , local basis vectors a jt , and modeling error t .

The locus of all potential signal vectors of interest is assumed to lie on or near one of the local subspaces.

Since the observation is assumed to lie close to one of the T subspaces we can therefore write the i th observation as DISPLAYFORM3 3 GEOMETRICALLY ROBUST NETWORKSIn this section we provide the complete framework for inserting our statistical-geometric viewpoint from Section 2 into a robust design approach that we will call geometrically robust networks (GRN).

We propose to use unsupervised learning on the input space for label encoding and unsupervised learning on the feature space for density estimation in a MAP classifier.

In this study we target the classification layers in g ?? of (1) as the key layers for improvement assuming the supervised feature learning has adequately reached the information bottleneck limit BID21 .

We will improve upon the classification layers in two fundamental ways:1.

Use soft decision labels to train the neural network.

The Bayesian nonparametric mixture of factor analyzers (BNP-MFA) from BID5 which is our building block for estimating the union of subspaces.

The tunable hyperparameters are the Dirichlet process (DP) concentration parameter which influences the number of mixtures/subspaces and the Beta process (BP) parameters which influence the dimensionality of each subspace.

, ..., h ?? c 1 which generally implement a softmax multinomial logistic regression with a Bayesian maximum a posteriori (MAP) classifier using plug-in class-conditional density estimates.

In Subsection 3.1 we summarize the Bayesian nonparametric mixture of factor analyzers (BNP-MFA) model for union of subspace learning.

In Subsection 3.2 we describe label encoding and MAP classifition steps which directly follow from learning the BNP-MFA.

To estimate the geometric model described above in Section 2.2 we use the Bayesian nonparametric formulation of the mixture of factor analyzers (BNP-MFA introduced in BID5 ) which has several advantages for estimating our required statistical union of subspaces:1.

Accuracy: Mixture of factor analyzer models empirically show higher test log-likelihoods (model evidence) than Boltzmann machine based models BID23 .

2. Speed: Since the BNP-MFA is a conjugate-exponential model it can be learned in online variational Bayes form with stochastic variational inference BID8 giving it orders of magnitude speed improvements compared to Markov chain Monte Carlo methods.

3.

Scales with data: Since the model is Bayesian nonparametric, there is no model overfitting and no need for regularization.

4.

Hyperparameter Insensitive Only two hyperparameters need to be set are they are very insensitive to overall performance.

Under the BNP-MFA framework we infer the number of subspaces and subspace rank from the data using Bayesian nonparametrics.

A Dirichlet process mixture model is used to model the clusters, while a Beta process is used to estimate the local subspace in each cluster.

The conjugate-exponential directed graphical model shown in FIG3 (a) and hierarchical roll out in FIG3 (b) is taken from BID5 .

Here, the {x i } n i=1 are vector-valued observations in R N with component weights given by the vectors DISPLAYFORM0 are various global parameters for each of the T mixture components, ?? ??? R is a global parameter for the Dirichlet process, ?? = 1 n n i=1 x i is the (fixed) sample mean of the data and a-h and ?? 0 are fixed constants.

More details on this model and the motivation for its construction can be found in BID5 After the BNP-MFA model finishes training, we have the all the parameters (centroids, subspace spanning vectors, and cluster weights) that we need to estimate the class conditional probability density function (6) which we will use for both MAP classification and soft-decision label encoding as we show in section 3.

The idea of soft decision labels was used in defensive distillation BID19 .

Papernot et.

al. used the first pass through their target neural network g(x) with annealed softmax to learn to the soft decision labels y = g(x).

They then used those learned labels y in the second pass through the same network but with different softmax thermal parameters to create the distilled network.

As pointed out in BID19 , the distilled network g d (??) will converge toward the originally network g(??) under a cross entropy loss given enough training data.

Thus, the distilled network can still possess some of the brittle nature of the original network trained with hard decision labels.

This vulnerability was revealed to be the case in BID1 .We deviate from Papernot's defensive distillation approach here by using class conditional density estimatep ?? (x|k) on the class-partitioned input data with K total classes to create our labels.

Here, we use the fact that the BNP-MFA is a demarginalization of a Gaussian mixture model (GMM) to form density estimates.

The term demarginalization from Robert & Casella FORMULA2 is taken here to mean the formation of a latent variable probability density which is the integrand under a marginalization integral.

We learn the BNP-MFA model with parameters ?? from the original class-partitioned signals/images as input and then estimate the class-conditional pdf over the input space a?? DISPLAYFORM0 Then, we assign our label vector as the posterior DISPLAYFORM1 .., n and ???k = 1, 2, ..., Kwhere p(k) is the class prior.

The term ?? ki is a combined correction factor and normalization factor to scale the correct class label higher than incorrect classes for the cases where DISPLAYFORM2 and where k * is the correct class index.

The ?? ki term also enforces that k y ki = 1, ???i.

Soft decision label encoding based on class conditional likelihood has the advantage that it is independent of any deep architecture.

Once we learn the labels {y i } n i=1 from FORMULA12 , we apply those labels to learn the neural network g ?? from g ?? (x i ) = y i , ???i = 1, ..., n using traditional backprogation with SGD training on a cross entropy loss function.

After the model converges, we extract features z i = h ?? f (x i ), ???i = 1, ..., n and learn a new BNP-MFA with model parameters ?? over the feature space.

To learn the feature space class-conditional pdfs we simply swap out x with z and the ?? with ?? in (6) to obtain the approximate class-conditional likelihood functions over the feature vectors.

The approximate posterior pdf is then simplyp ?? (k|z) ???p ?? (z|k)p(k).

For the datasets we benchmark over the prior p(k) = 1/K is uniform, and the MAP classifier reduces to maximum likelihood (ML) classification.

However, in the real world class priors are almost never uniform and MAP classification gives a significant boost not only over multinomial logistic regression but ML classification as well.

We summarize the training and testing stage procedures of GRN in Algorithm 1.In Figure 3 we show plots of the negative squared Mahalanobis distance for each cluster of the class conditional input space pdf in (6) for a single image confuser sample from a Carlini-Wagner attack compared to the original unperturbed image.

We see that the adversarial attack has almost no Algorithm 1 Geometrically Robust Networks (GRN) Augmentation Framework 1: procedure TRAINING PHASE SUMMARY 2: DISPLAYFORM3 Label encode: DISPLAYFORM4 Learn base model {g ?? (x i ) = y i } n i=1 using SGD backprop on cross entropy loss to select feature extraction layer h ?? f (x i ) in FORMULA0 5:Extract features: DISPLAYFORM5 learnp ?? (z|k) using BNP-MFA 7: end procedure 8: procedure TESTING PHASE SUMMARY 9: DISPLAYFORM6 , nonlinear function h ?? f (??), class prior p(k), and pdf estimat?? p ?? (??|k), estimate class label ask i = argmax kp ?? (h ?? f (x i )|k)p(k) 10: end procedure Figure 3 : Matrix of ten plots (one for each of the ten CIFAR-10 classes) showing the negative squared Mahalanobis distance for each cluster of the class conditional input space pdf in (6) for a single image confuser sample from a Carlini-Wagner attack compared to the original unperturbed image.

We see that the adversarial attack has almost no influence on the pdf components.influence on the pdf in (6) and, therefore, practically no variation on the corresponding label in (7).

This demonstrates the concept depicted in FIG1 of how the projected data points have very little deviation in the latent space.

For our base neural network from which we build the GRN in step 4 of Algorithm 1 for CIFAR-10 and CIFAR-100 we use Springenberger's "All convolutional network" BID22 which uses only convolutional layers for entire stack.

Our black box network from which we craft adversarial samples for CIFAR-10 and CIFAR-100 is: [3x3 conv 32 LeakyReLU(0.

The Radio-ML dataset https://radioml.com/datasets is a relatively new time series dataset for benchmarking radio modulation recognition tasks.

It has 11 modulation schemes (3 analog, 8 digital) undergoing sample rate offset, center frequency offset, frequency flat fading, and AWGN.

We measure probability of correct classification as a function of signal-to-noise ratio (SNR) for that dataset.

For the Radio-ML dataset our base neural network from which we build the GRN in step 4 of Algorithm 1 is given at the top of Figure 4 (b).

The black box attack network for Radio-ML is a version of LeNet-5 CNN used in (O'Shea et al., 2016) and is shown at the bottom of Figure 4 (b).For both datasets the data was scaled to lie between zero and one with respect to the adversarial parameter settings.

Using cleverhans (Nicolas Papernot, 2017), we craft adversarial samples from fast gradient sign method (FGSM), Carlini Wagner (CW) BID2 , and projected gradient descent (PGD) BID13 on the CIFAR-10 and CIFAR-100 dataset.

With FGSM we use eps=.005.

With CW method we use initial tradeoff-constant = 10, batch size = 200, 10 binary search steps, and 100 max iterations.

With PGD we use eps = 1, number of iterations = 7, and a step size for each attack iteration = 2.

These attack parameter settings were more than enough to confuse the base classifier while keeping the mean structural similarity index BID25 relatively constant between natural and adversarial images.

For the Radio-ML dataset we only experiment with FGSM (eps=.03).

For the CIFAR-100 dataset we used the following data augmentation parameters:(1) 10 degree random rotations, (2) zoom range form .8 to 1.2, (3) width shift range percentatage = 0.2, (4) height shift range percentage = 0.2, and (5) random horizontal flips.

As shown in TAB0 the proposed GRN model for CIFAR-10 performed remarkably well in the face of all three attacks suffering only about 10-20 percent accuracy compared to base network with no attack (natural test samples only).

(At the time of this submission we are running the CW and PGD attacks on CIFAR-100 and plan to include results on next iteration of paper.)

The CIFAR-100 results under no attack did not match the reported results in BID22 likely because we did not use as extensive data augmentation.

In Figure 4 (a) we plot the accuracy versus SNR for the four cases using the Radio-ML dataset: 1) base model no attack, 2) base model FGSM attack, 3) GRN model no attack, and 4) GRN model FGSM attack.

Again, we see that the GRN is relatively unaffected by the adversarial attack.

We also experimented with hyperparameter settings for the Dirichlet process concentration parameter ?? and the ratio of the Beta process parameters a b .

We observed that up to one order of magnitude in change in ?? and a b there was demonstrable change in the ultimate classification performance.

We have demonstrated that geometrical statistically augmented neural network models can achieve state-of-the-art robustness on CIFAR-10 under three different adversarial attack methods.

We hope that this work will be the start of further investigation into the idea of using geometrically centered unsupervised learning methods to assist in making deep learning models robust, not only to adversarial noise but to all types of noise.

There is more work that could be done to understand the best way to engineer soft decision labels given auxiliary data models.

We need to also understand if the training algorithms themselves can be directly manipulated to incorporate outside structural data models.

A main selling point of Bayesian nonparametrics has been that the complexity of the model can grow as more data is observed.

However, the current training algorithm for the BNP-MFA model is Gibbs sampling, which fails to scale to massive data sets.

Stochastic variational inference BID8 has been introduced as one such way to perform variational inference for massive or streaming data sets.

We are currently working to cast the BNP-MFA into a stochastic variational framework so that the GRN model can be extended to very large (or even streaming) datasets.

Figure 4: Network specification and performance results for proposed geometrically robust networks applied to the Radio-ML dataset (modulation recognition over 11 modulation formats).

<|TLDR|>

@highlight

We develop a statistical-geometric unsupervised learning augmentation framework for deep neural networks to make them robust to adversarial attacks.

@highlight

Transfroms traditional deep neural networks into adversarial robust calssifiers using GRNs

@highlight

Proposes a defense based on class-conditional feature distributions to turn deep neural netwroks into robust classifiers