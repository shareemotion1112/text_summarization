We propose a novel algorithm, Difference-Seeking Generative Adversarial Network (DSGAN), developed from traditional GAN.

DSGAN considers the scenario that the training samples of target distribution, $p_{t}$, are difficult to collect.



Suppose there are two distributions  $p_{\bar{d}}$ and $p_{d}$ such that the density of the target distribution can be the differences between the densities of $p_{\bar{d}}$ and $p_{d}$.

We show how to learn the target distribution $p_{t}$ only via samples from $p_{d}$ and $p_{\bar{d}}$ (relatively easy to obtain).



DSGAN has the flexibility to produce samples from various target distributions (e.g. the out-of-distribution).

Two key applications, semi-supervised learning and adversarial training, are taken as examples to validate the effectiveness of DSGAN.

We also provide theoretical analyses about the convergence of DSGAN.

In machine learning, how to learn a probability distribution is usually conducted in a unsupervised learning manner.

Generative approaches are developed for learning data distribution from its samples and thereafter produce novel and high-dimensional samples from learned distributions, such as image and speech synthesis BID18 ).

The state-of-the-art approaches is so-called Generative Adversarial Networks (GAN) BID6 ).

GAN produces sharp images based on a game-theoretic framework, but can be tricky and unstable to train due to multiple interacting losses.

Specifically, GAN consists of two functions: generator and discriminator.

Both functions are represented as parameterized neural networks.

The discriminator network is trained to classify whether or not inputs belong to real data or fake data created by the generator.

The generator learns to map a sample from a latent space to some distribution to increase the classification errors of discriminator.

GAN corresponds to a minimax two-player game, which ends if the generator actually learns the real data distribution.

The generator is of main interest because the discriminator will be unable to differentiate between both distributions once the generator has been trained well.

In reality, it is difficult to collect training samples from unseen classes, which none of their samples involves the training phase, but their samples could be encountered in the testing phase.

How to reject or recognize unseen data as "abnormal" (not belonging to the training data) is an important issue known as one-class classification BID17 ).

Due to the absence of unseen data, most of algorithms are unsupervised BID21 ).

Based on the assumption that there do exist very few unseen examples, some approaches BID24 ) focus on supervised learning using unbalanced data.

In addition to one-class classification, BID4 theoretically show that complementary data, which is also considered as unseen data, can improve semi-supervised learning.

Another related issues is adversarial attack, where classifiers may be vulnerable to adversarial examples, which are unseen during the training phase.

FORMULA18 in Appendix A illustrates the applications regarding unseen data.

Apparently, if we can generate unseen data via GAN, it is helpful for those applications.

But, traditional GAN requires preparing plenty of training samples of unseen classes for training, leading to the contradiction with the prerequisite.

This fact motivates us to design the proposed DSGAN, which can generate unseen data by taking seen data as training samples shown in FIG10 .

The nuclear idea is to consider the distribution of unseen data as the difference of two distributions, which both are relatively easy to obtain.

For example, out-of-distribution examples in the MNIST dataset, from another point of view, are found to belong to the difference of the set of examples in MNIST and the universal set.

Examples in both sets are relatively easy to obtain.

It should be noted that the target distribution is equal to the training data distribution in traditional GANs; nevertheless, both distributions are considered different in DSGAN.

FIG10 : The illustration of difference between traditional GAN and DSGAN.

In this paper, we make the following contributions:(1) We propose a novel algorithm, Difference-Seeking Generative Adversarial Network (DS-GAN), where the density of target distribution p t is the difference between those of any two distributions, pd and p d .

Nevertheless, the differences of two densities being negative are not well-defined.

Thus, instead of learning the target distribution p t directly, the generator distribution approximates p t by minimizing the statistical distance between the mixture distribution of p d and the generator distribution, and pd.(2) Theoretical results based on traditional GAN are extended to the case of mixture distribution considered in this paper.

With enough capacity of the generator and the discriminator, we show DSGAN can learn the generator distribution p g under mild condition where the support set of p g is the difference of support sets of pd and p d .(3) We show that DSGAN possesses the flexibility to learn different target distributions in two key applications: semi-supervised learning and adversarial training.

Samples from target distribution in semi-supervised learning must satisfy two conditions: i) are linear combination of any label data and unlabel data; ii) do not belong to neither label data nor unlabel data.

For adversarial training, samples from the target distribution are assigned as out-of-distribution examples with bounded distortion.

Experiments validate that DSGAN can learn these two kinds of distributions well in various datasets.

The paper is structured as follows.

In Sec. 2, we introduce DSGAN, including the algorithm in the training procedure.

Theoretical results are described in Sec. 3.

In Sec. 4, two applications are taken as examples to show the effectiveness of DSGAN.

Finally, we present the experimental results in Sec. 5 and conclude by pointing out some promising directions for future work in Sec. 7.2 PROPOSED METHOD-DSGAN 2.1 THE SCHEME OF DSGANWe denote the generator distribution as p g and training data distribution as p d , both in a Ndimensional space.

Let pd be the distribution decided by user.

For example, pd can be the convolution of p d and normal distribution.

Let p t be the target distribution which the user is interested in, and it can be expressed as DISPLAYFORM0 where ?? ??? [0, 1].

Our method, DSGAN, aims to learn p g such that p g = p t .

Intuitively, our method DISPLAYFORM1 In other words, the generator is going to output samples located in high-density areas of pd and low-density areas of p d .At first, we formulate the generator and discriminator in GANs.

The inputs z of the generator are drawn from p z (z) in an M -dimensional space.

The generator function G(z; ?? g ) : DISPLAYFORM2 represents a mapping to data space, where G is a differentiable function with parameters ?? g .

The discriminator is defined as D (x; ?? d ) : R N ??? [0, 1] that outputs a single scalar.

D (x) can be considered as the probability that x belongs to a class of real data.

Similar to traditional GAN, we train D to distinguish the real data and the fake data sampled from G. Meanwhile, G is trained to produce realistic data as possible to mislead D. But, in DSGAN, the definitions of "real data" and "fake data" are different from those in traditional GAN.

The samples from pd are considered as real but those from the mixture distribution between p d and p g are considered as fake.

The objective function is defined as follows: DISPLAYFORM3 (2) During the training procedure, an iterative approach like traditional GAN is to alternate between k steps of training D and one step of training G. In practice, minibatch stochastic gradient descent via back propagation is used to update ?? d and ?? g .

In other words, for each of p g , p d and pd, m sample are required for computing gradients, where m is the number of samples in a minibatch.

Algorithm 1 illustrates the training procedure in detail.

DSGAN suffers from the same drawbacks with traditional GAN (e.g., mode collapse, overfitting, and strong discriminator such that the generator gradient vanishes).

There are literatures BID19 , Miyato et al. (2018) ) focusing on improving the above problems, and their ideas can be combined into DSGAN.

and BID16 proposed the similar objective function like (2).

Their goal is to learn the conditional distribution of training data.

Nevertheless, we aim to learn the target distribution p t in Eq. BID29 , not the training data distribution.

The training procedure of DSGAN using minibatch stochastic gradient descent.

k is the number of steps to apply to the discriminator.

?? is the ratio between p g and p d in the mixture distribution.

We used k = 1 and ?? = 0.8 in experiments.01.

for number of training iterations do 02.

for k steps do 03.Sample minibatch of m noise samples z BID29 , ..., z (m) from p g (z).

Sample minibatch of m samples x DISPLAYFORM0 Sample minibatch of m samples x DISPLAYFORM1 Update the discriminator by ascending its stochastic gradient: DISPLAYFORM2 Sample minibatch of m noise samples z BID29 , ..., z (m) from p g (z).

Update the generator by descending its stochastic gradient: The illustration about generating boundary data around training data.

First, the convolution of p d and normal distribution makes the density on boundary data be no longer zero.

Second, we seek p g such that Eq. FORMULA0 holds, where the support set of p g is approximated by the difference of those between pd and of p d .

DISPLAYFORM0 Low-density samples generation FIG1 illustrates that DSGAN is able to generate boundary samples on the 2D swissroll.

Given the density function of the swissroll as p d , we assign pd as the convolution of p d and the normal distribution.

Then, by applying DSGAN, we achieve our goal to generate boundary samples.

The intuition of our idea is also illustrated by a 1D example in FIG3 .

In general, our idea will lead DSGAN to generate low-density samples like another example in FIG2 .

In this case, the density is low in the center of the circle, and our generator can not only create the boundary samples but also the samples located in low-density area.

Difference-set generation We also validate DSGAN on high dimensional dataset such as MNIST.In this example, we define p d be the distribution of digit "1" and pd be the distribution contains both digits "1" and "7".

Since the density p d (x) is high when x is digit "1", the generator is prone to output digit "7" with high probability.

From the above results, we can observe two properties of generator distribution p g : i) the higher density of p d (x), the lower density of p g (x); ii) p g prefers to output samples from high-density areas of pd(x)-p d (x).

In those case studies, ?? = 0.8 in Eq. FORMULA0 is used.

In the next section, we will show that the objective function is equivalent to minimizing the JensenShannon divergence between the mixture distribution (p d and p g ) and pd as G and D are given enough capacity.

Furthermore, we provide a trick (see Appendix C in details) by reformulating the objective function (2) such that it is more stable to train DSGAN.

There are two assumptions for subsequent proofs.

First, in a nonparametric setting, we assume both generator and discriminator have infinite capacity.

Second, p g is defined as the distribution of the samples drawn from G(z) under z ??? p z .

We will first show the optimal discriminator given G and then show that minimizing V (G, D) via G given the optimal discriminator is equivalent to minimizing the Jensen-Shannon divergence between (1 ??? ??)p g + ??p d and pd.

Proposition 1.

For G being fixed, the optimal discriminator D is DISPLAYFORM0 .Proof.

See Appendix B.1 in details.

Moreover, D can be considered to discriminate between samples from pd and those from DISPLAYFORM1 .

By replacing the optimal discriminator into V (G, D), we obtain DISPLAYFORM2 where the third equality holds because of the linearity of expectation.

Actually, the previous results show the optimal solution of D given G being fixed in (3).

Now, the next step is to find the optimal G with D * G being fixed.

DISPLAYFORM3 for all x's.

At that point, C(G) achieves the value ??? log 4.

The assumption, ??p d (x) ??? pd(x) for all x's, in Theorem 1 may be impractical in real applications.

We discuss that DSGAN still works well even though the assumption does not hold.

There are two facts: i) given D, V (G, D) is a convex function in p g and ii) Due to x p g (x)dx = 1, the set collecting all feasible solutions of p g is a convex set.

In other words, there always exists a global minimum of V (G, D) given D, but it may not be ??? log(4).

In this following, we show that the support set of p g is contained within the difference of support sets of pd and p d while achieving the global minimum such that we can generate the desired p g by designing appropriate pd.

Proposition 2.

Suppose ??p d (x) ??? pd for x ??? Supp(p d ) and all density functions p d (x), pd(x) and p g (x) are continuous.

If the global minimum of the virtual training criterion C(G) is achieved, then DISPLAYFORM0 Proof.

See Appendix B.3 in details.

In sum, the generator is prone to output samples located in high-density areas of pd and low-density areas of p d .Another concern is the convergence of Algorithm 1.

Proposition 3.

The discriminator reaches its optimal value given G in Algorithm 1, and p g is updated by minimizing DISPLAYFORM1 Proof.

See Appendix B.4 in details.

DSGAN can be applied to two applications: semi-supervised learning and adversarial training.

In semi-supervised learning, DSGAN acts as a "bad generator", which creates complement samples in the feature space of real data.

As for adversarial training, DSGAN generates adversarial examples located in the low-density areas of training data.

Semi-supervised learning (SSL) is a kind of learning model with the use of a small number of labeled data and a large amount of unlabeled data.

The existing SSL works based on generative model (e.g., VAE BID9 ) and GAN BID19 )) obtain good empirical results.

BID4 theoretically show that good semi-supervised learning requires a bad GAN with the objective function: DISPLAYFORM0 where (x, y) denotes a pair of data and its corresponding label, {1, 2, . . .

, K} denotes the label space for classification, and L = {(x, y)} is the label dataset.

Moreover, in the semi-supervised settings, the p d in FORMULA14 is the distribution of the unlabeled data.

Note that the discriminator D in GAN also plays the role of classifier.

If the generator distribution exactly matches the real data distribution (i.e., p g = p d ), then the classifier trained by the objective function (4) with the unlabeled data cannot have better performance than that trained by supervised learning with the objective function: DISPLAYFORM1 On the contrary, the generator is preferred to generate complement samples, which lie on lowdensity area of p d .

Under some mild assumptions, those complement samples help D to learn correct decision boundaries in low-density area because the probabilities of the true classes are forced to be low on out-of-distribution areas.

The complement samples in BID4 are complicate to produce.

We will demonstrate that DSGAN is easy to generate complement samples in Sec. 5.

Deep neural networks have impacted on our daily life.

Neural networks, however, are vulnerable to adversarial examples, as evidenced in recent studies (Papernot et al. (2016) ) BID3 ).

Thus, there has been significant interest in how to enhance the robustness of neural networks.

Unfortunately, if the adversary has full access to the network, namely white-box attack, a complete defense strategy has not yet been found.

BID2 surveyed the state-of-the-art defense strategies and showed that adversarial training BID15 ) is more robust than other strategies.

Given a trained classifier C parameterized by ?? and a loss function (x; y; C ?? ), adversarial training solves a min-max game, where the first step is to find adversarial examples within -ball for maximizing the loss, and the second step is to train the model for minimizing the loss, given adversarial examples.

Specifically, the objective BID15 ) is DISPLAYFORM0 The authors used projected gradient descent (PGD) to find adversarial examples by maximizing the inner optimization.

Instead of relying on PGD, our DSGAN generates adversarial examples directly, which are combined into real training data to fine-tune C ?? .

-ball in terms of 2 or inf can be intuitively incorporated into the generation of adversarial examples.

In this section, we demonstrate the empirical results about semi-supervised learning and adversarial training in Sec. 5.1 and Sec. 5.2, respectively.

Note that, the training procedure of DSGAN can be improved by other extensions of GANs such as WGAN ), WGAN-GP BID7 ), EBGAN (Zhao et al. (2017) ), LSGAN (Mao et al. FORMULA0 ) and etc.

We use the idea of WGAN-GP in our method such that DSGAN is stable in training and suffers less mode collapse.

Following the previous works, we apply the proposed DSGAN in semi-supervised learning on three benchmark datasets, including MNIST (LeCun et al. FORMULA0 We first introduce how DSGAN generates complement samples in the feature space.

Specifically, BID4 proved that if complement samples generated by G can satisfy the following two assumptions in (7) and (8): DISPLAYFORM0 where f is the feature extractor and w i is the linear classifier for the i th class and DISPLAYFORM1 then all unlabeled data will be classified correctly via the objective function (4).The assumption in (8) implies the complement samples have to be at the space created by linear combination of labeled and unlabeled data.

Besides, they cannot fall into the real data distribution p d due to the assumption (7).

In order to have DSGAN generate such samples, we let the samples of pd be the linear combination of those from L and DISPLAYFORM2 tend to match pd while the term ?????p d ensures that samples from p g do not belong to p d .

Thus, p g satisfies both assumptions in FORMULA17 and FORMULA18 .In practice, we parameterized f and all the w together as a neural network.

The details of the experiments, including the network models, can be found in Appendix D.

For evaluating the semi-supervised learning task, we used 60000/ 73257/ 50000 samples and 10000/ 26032/ 10000 samples from the MNIST/ SVHN/ CIFAR-10 dataset for training and testing, respectively.

Due to the semi-supervised setting, we randomly chose 100/ 1000/ 4000 samples from the training samples as the MNIST/ SVHN/ CIFAR-10 labeled dataset, and the amount of labeled data for all classes are equal.

Our criterion to determine the hyperparameters are in Appendix D.1.

We perform testing with 10/ 5/ 5 runs on MNIST/ SVHN/ CIFAR-10 based on the selected hyperparameters and randomly selected labeled dataset.

Following BID4 , the results are recorded as the mean and standard deviation of number of errors from each run.

First, the hyperparameters we chose is depicted in TAB3 in Appendix D.1.

Second, the results obtained from our DSGAN and the state-of-the-art methods on three benchmark datasets are depicted in TAB1 .It can be observed that our results can compete with state-of-the-art methods on the three datasets.

Moreover, in comparison with BID4 , our methods don't need to rely on an additional density estimation network PixelCNN++ BID20 ).

Although PixelCNN++ is one of the best density estimation network, it cannot estimate the density in the feature space, which is dynamic during training.

This drawback make the models in BID4 cannot completely fulfill the assumptions in their paper.

Our results in TAB1 is slightly inferior to the best record of badGAN BID4 ) but outperforms other approaches.

In comparison with them, there is a probable reason to explain the slightly inferior performance in the following.

Since the patterns of images are complicated, the generator without enough capacity is not able to learn our desired distribution, which is the distribution meets the conditions in (7) and (8).

However, this problem will be attenuated with the improvements of GAN, and our models benefit from them.

In badGAN, they rely on the feature matching in their objective function.

No matter how they change the divergence criterion on two distributions, feature matching still let them learn the distribution matching the first-order statistics, so they cannot totally get the advantage of the progress of GANs.

Our proposed DSGAN is capable to be used to improve the robustness of the classifier against adversarial examples.

In the experiments, we mainly validate DSGAN on CIFAR-10, which is widely used in adversarial training.

Recall that the objective function (6) requires finding adversarial examples to maximize the classification error (??).

Adversarial examples usually locate on the low-density area of p d and are generated from labeled data via gradient descent.

Instead of using gradient descent, we aim to generate adversarial examples via GAN.

By assigning pd as the convolution of p d and uniform distribution, samples from p g will locate on the low-density area of p d .

Furthermore, the distortion is directly related to the range of uniform distribution.

It, however, may be impractical for training the generator for each class.

Thus, we propose a novel semi-supervised adversarial learning approach here.

Three stages are required to train our model: First, we train a baseline classifier on all the training data.

All the training data are labeled and represent samples from L in (9).

Second, we train a generator to generate adversarial examples and treat these adversarial examples as additional unlabeled training data (x ??? p g in FORMULA20 ).

Third, we fine-tune the classifier C ?? with all training data and the data produced by the generator via minimizing the following objective: DISPLAYFORM0 where the first term is a typical supervised loss such as cross-entropy loss and the second term is the entropy loss H of generated unlabeled samples corresponding to the classifier, meaning that we would like the classifier to confidently classify the generated samples.

In other words, if an adversarial example x g is the closest to one of labeled data x, it should be classified into the class of x. Thus, the additional entropy loss will prevent our model from the attack by adversarial examples.

Furthermore, in (9), one can view the weight w is the trade-off between the importance of labeled data in high-density area and unlabeled data in low-density area.

If w is 0, the model might be prone to classify correctly only on the labeled data.

When increasing w, the model will place more emphasis on unlabeled data.

Since the unlabeled data acts as adversarial examples, therefore, the classifier is more robustness.

We evaluate the trained models against a range of adversaries, where the distortion is evaluated by 2 -norm or inf -norm.

The adversaries include:??? White-box attacks with Fast Gradient Sign Method (FGSM) BID5 ) using inf -norm.??? White-box attacks with PGD BID11 ) using inf -norm.??? White-box attacks with Deepfool (Moosavi-Dezfooli et al. FORMULA0 ) using 2 -norm and infnorm.

According to different adversaries, we generate 10000 adversarial examples from testing data and calculate the accuracy of the model after attacking.

The accuracy is record as the probability that adversarial examples fail to attack when the distortion created by attacking algorithm cannot exceed a maximum value.

We also train our models with different ranges of uniform distribution.

The experimental detail can be found in Appendix E.To validate our method, we propose two kinds of baseline networks.

One is a baseline classifier we train in the first stage, which is a typical classifier trained by all data.

The other one is the model with noisy inputs.

Adding noise to the input is a prevalent strategy to train a classifier and it is also able to protect the neighborhood of the training data.

For fare comparison to our method, uniform noise is used in the second baseline model.

Fig. 7 demonstrates that our models exhibit stronger robustness among all the adversaries.

w is set to 10 in this figure, other results with different w are displayed in the Appendix E and we claim that our method can outperform other baselines in a wide range of values of w. We notice that the model benefits from controlling the weight w. When we increase the w from 1 to 3, and then from 3 to 10, the robustness keeps becoming stronger.

Our second baseline models have the similar intuition with our method, they propagate the label information to the neighborhood of each data point by introducing the noise to inputs.

This strategy can improve the accuracy and the robustness of the model.

Nevertheless, the training data distribution after applying noise can be viewed as a smoother version of original distribution.

Most of samples still are locate on high-density area of original distribution.

Due to this reason, the second baseline models cannot emphasize low-density samples via w like the proposed model, leading to the inferior robustness.

Our method relies on a generator to produce low-density data.

The generated samples help our model to put decision boundary outside low-density area.

Thus, the model can resist adversarial attacks with larger distortion theoretically.

It's worth mentioning that our method is able to combine with the idea of second baseline to the supervised term in (9) and the performance might be improved.

We introduce related works about generating unseen data.

BID25 proposed a method to generate samples of unseen classes in the unsupervised manner via an adversarial learning strategy.

But, it requires solving an optimization problem for each sample, which may lead to high computation cost.

On the contrary, DSGAN has the capability to create infinite diverse unseen samples.

BID8 presented a new GAN architecture that can learn both distributions of unseen data from part of seen data and unlabeled data.

But, unlabeled data must be a mixture of seen and unseen samples; DSGAN does not require any unseen data instead.

Both BID4 aim to generate complementary samples (or out-of-distribution sample) but assume that in-distribution can be estimated by a pretrained model such as PixelCNN++, which might be difficult and expensive to train.

BID13 use a simple classifier to replace the role of PixelCNN++ in BID4 such that the training is much easier and more suitable.

Nevertheless, their method only focuses on generates unseen data surrounding low-density area of seen data, but DSGAN is more flexible to generate different kinds of unseen data (e.g., the linear combination of seen data shown in Sec.5.1).

Besides, their method needs the label information of data while ours is fully unsupervised.

In this paper, we propose DSGAN that can produce samples from the target distribution based on the assumption that the density of target distribution can be the difference between the densities of any two distributions.

DSGAN is useful in the environment when the samples from the target distribution are more difficult to collect than those from the two known distributions.

We demonstrate that DSGAN is really applicable to, for example, semi-supervised learning and adversarial training.

Empirical and theoretical results are provided to validate the effectiveness of DSGAN.

Finally, because DSGAN is developed based on traditional GAN, it is easy to extend any improvements of traditional GAN to DSGAN.

We complete this proof.

Proof.

We start from DISPLAYFORM0 where KL is the Kullback-Leibler divergence and JSD is the Jensen-Shannon divergence.

The JSD returns the minimal value, which is 0, iff both distributions are the same, namely pd = (1 ??? ??)p g + ??p d .

Because p g (x)'s are always non-negative, it should be noted both distributions are the same only if ??p d (x) ??? pd(x) for all x's.

We complete this proof.

Proof.

Recall DISPLAYFORM0 is to simplify the notations inside the integral.

For any x, S(p g ; x) in p g (x) is nonincreasing and S(p g ; x) ??? 0 always holds.

Specifically, S(p g ; x) is decreasing along the increase of p g (x) if pd(x) > 0; S(p g ; x) attains the maximum value, zero, for any p g (x) if pd(x) = 0.

Since DSGAN aims to minimize C(G) with the constraint DISPLAYFORM1 is increasing on p g (x) and converges to 0.

When DISPLAYFORM2 DISPLAYFORM3 the last inequality implies that there exists a solution such that DISPLAYFORM4 We complete this proof.

as a function of p g .

By the proof idea of Proposition 2 in BID6 , if f (x) = sup ?????A f ?? (x) and f ?? (x) is convex in x for every ??, then DISPLAYFORM0 includes the derivative of the function at the point, where the maximum is attained, implying the convergence with sufficiently small updates of p g .

We complete this proof.

We provide a trick to stabilize the training procedure by reformulating the objective function.

Specifically, V (G, D) in (2) is reformulated as: DISPLAYFORM0 Instead of sampling a mini-batch of m samples from p z and p d in Algorithm 1, (1 ??? ??)m and ??m samples from both distributions are required, respectively.

The computation cost in training can be reduced due to fewer samples.

Furthermore, although (10) is equivalent to (2) in theory, we find that the training using (10) achieves better performance than using (2) via empirical validation in TAB2 .

We conjecture that the equivalence between (10) and (2) is based on the linearity of expectation, but mini-batch stochastic gradient descent in practical training may lead to the different outcomes.

The hyperparameters were chosen to make our generated samples consistent with the assumptions in (7) and (8).

However, in practice, if we make all the samples produced by the generator following the assumption in (8), then the generated distribution is not close to the true distribution, even a large margin between them existing in most of the time, which is not what we desire.

So, in our experiments, we make a concession that the percentage of generated samples, which accords with the assumption, is around 90%.

To meet this objective, we tune the hyperparameters.

TAB3 shows our setting of hyperparameters, where ?? is defined in (8).

In order to fairly compare with other methods, our generators and classifiers for MNIST, SVHN, and CIFAR-10 are same as in BID19 and BID4 .

However, different from previous works that have only a generator and a discriminator, we design an additional discriminator in the feature space, and it's architecture is similar across all datasets with only the difference in the input dimensions.

Following BID4 , we also define the feature space as the input space of the output layer of discriminators.

Compared to SVHN and CIFAR-10, MNIST is a simple dataset as it is only composed of fully connected layers.

Batch normalization (BN) or weight normalization (WN) is used to every layer to stable training.

Moreover, Gaussian noise is added before each layer in the classifier, as proposed in Rasmus et al. (2015) .

We find that the added Gaussian noise exhibits positive effect for semisupervised learning and keep to use it.

The architecture is shown in TAB4 .

TAB6 are models for SVHN and CIFAR-10, respectively, and these models are almost the same except for some implicit differences, e.g., the number of convolutional filters and types of dropout.

In these tables, given a dropping rate, "Dropout" is a normal dropout in that the elements of input tensor are randomly set to zero while Dropout2d is a dropout only applied on the channels to randomly zero all the elements.

Furthermore, the training procedure alternates between k steps of optimizing D and one step of optimizing G. We find that k in Algorithm 1 is a key role to the problem of mode collapse for different applications.

For semi-supervised learning, we set k = 1 for all datasets.

In our experiments, as for the second stage, we train DSGAN for 50 epochs in Algorithm 1 to generate our adversarial examples.

In the third stage, we finetune the baseline classifier for 50 epochs.

In the experiments for adversarial training on CIFAR-10, the generator and discriminator are the same as those in semi-supervised learning.

The architecture is described in Table 5 and the classifier is modified from the one shown in TAB6 .

First, we get rid of all the dropouts and Gaussian noise so that we can compare among different models with less randomness.

Moreover, we decrease the number of layers in the original model, simply intending to accelerate training.

The number of layers following the feature space is increased to 3.

Because we apply our method in the feature space, the sub-network after feature space should be non-linear so that it can correctly classify generated data.

The architecture is described in TAB7 .

Furthermore, k is assigned to 5 in all experiments.

We show more results in FIG12 with w = 1 and 10 with w = 3.

@highlight

We proposed "Difference-Seeking Generative Adversarial Network" (DSGAN) model to learn the target distribution which is hard to collect training data.

@highlight

This paper presents DS-GAN, which aims to learn the difference between any two distributions whose samples are difficult or impossible to collect, and shows its effectiveness on semi-supervised learning and adversarial training tasks.

@highlight

This paper considers the problem of learning a GAN to capture a target distribution with only very few training samples from that distribution available.