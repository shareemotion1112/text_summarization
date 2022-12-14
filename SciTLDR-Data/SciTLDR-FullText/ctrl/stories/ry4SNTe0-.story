Improved generative adversarial network (Improved GAN) is a successful method of using generative adversarial models to solve the problem of semi-supervised learning.

However, it suffers from the problem of unstable training.

In this paper, we found that the instability is mostly due to the vanishing gradients on the generator.

To remedy this issue, we propose a new method to use collaborative training to improve the stability of semi-supervised GAN with the combination of Wasserstein GAN.

The experiments have shown that our proposed method is more stable than the original Improved GAN and achieves comparable classification accuracy on different data sets.

Generative adversarial networks (GANs) BID3 have been recently studied intensively and achieved great success in deep learning domain BID14 BID9 BID15 .

A typical GAN simulates a two-player minimax game, where one aims to fool the other and the overall system is finally able to achieve equilibrium.

Specifically speaking, we have a generator G to generate fake data G(z) from a random variable z whose distribution density is p(z), and also we have a discriminator D(x) to discriminate the real x from the generated data G(z), where x ∼ p r (x) and p r is the distribution density of real data.

We optimize the two players G(z) and D(x) by solving the following minimax problem: DISPLAYFORM0 This method is so called as the original GAN BID3 .

After this, many different types of GANs have been proposed, e.g., least-squared GAN BID9 , cat-GAN BID15 , W-GAN , Improved GAN BID14 , so on and so forth, focusing on improving the performance of GANs and extending the GAN idea to other application scenarios.

For instance, the original GAN is trained in a completely unsupervised learning way BID3 , along with many variants, such as LS-GAN and cat-GAN.

It was later extended to semi-supervised learning.

In BID14 , Salimans et al. proposed the Improved GAN to enable generation and classification of data simultaneously.

In BID7 , Li et al. extended this method to consider conditional data generation.

Another issue regarding the unsupervised learning of GANs is the lack of training stability in the original GANs, mostly because of dimension mismatch .

A lot of efforts have been dedicated to solve this issue.

For instance, in , the authors theoretically found that the instability problem and dimension mismatch of the unsupervised learning GAN was due to the maxing out of Jensen-Shannon divergence between the true and fake distribution and therefore proposed using the Wasserstein distance to train GAN.

However, to calculate the Wasserstein distance, the network functions are required to be 1-Lipschitz, which was simply implemented by clipping the weights of the networks in .

Later, Gulrajani et.

al. improved it by using gradient penalty BID4 .

Besides them, the same issue was also addressed from different perspectives.

In BID13 , Roth et al. used gradient norm-based regularization to smooth the f-divergence objective function so as to reduce dimension mismatch.

However, the method could not directly work on f-divergence, which was intractable to solve, but they instead optimized its variational lower bound.

Its converging rate is still an open question and its computational complexity may be high.

On the other hand, there were also some efforts to solve the issue of mode collapse, so as to try to stabilize the training of GANs from another perspective, including the unrolled method in BID10 , mode regularization with VAEGAN (Che et al., 2016) , and variance regularization with bi-modal Gaussian distributions BID5 .

However, all these methods were investigated in the context of unsupervised learning.

Instability issue for semi-supervised GAN is still open.

In this work, we focus on investigating the training stability issue for semi-supervised GAN.

To the authors' best knowledge, it is the first work to investigate the training instability for semi-supervised GANs, though some were done for unsupervised GANs as aforementioned.

The instability issue of the semi-supervised GAN BID14 is first identified and analyzed from a theoretical perspective.

We prove that this issue is in fact caused by the vanishing gradients theorem on the generator.

We thus propose to solve this issue by using collaborative training to improve its training stability.

We theoretically show that the proposed method does not have vanishing gradients on the generator, such that its training stability is improved.

Besides the theoretical contribution, we also show by experiments that the proposed method can indeed improve the training stability of the Improved GAN, and at the same time achieve comparable classification accuracy.

It is also worth to note that BID7 proposed the Triple GAN that also possessed two discriminators.

However, its purpose is focused on using conditional probability training (the original GAN uses unconditional probability) based on data labels to improve the training of GAN, but not on solving the instability issue.

Therefore, the question of instability for the Triple GAN is still unclear.

More importantly, the method, collaborative training, proposed for exploring the data labels with only unconditional probability in this paper , can also be applied to the Triple GAN to improve its training stability, in the case of conditional probability case.

The rest of the paper is organized as follows: in Section 2, we present the generator vanishing gradient theorem of the Improved GAN.

In Section 3, we propose a new method, collaborative training Wasserstein GAN (CTW-GAN) and prove its nonvanishing gradient theorem.

In Section 4, we present our experimental results and finally give our conclusion in Section 5.

The improved GAN BID14 combines supervised and unsupervised learning to solve the semi-supervised classification problem by simulating a two-player minmax game with adversarial training.

The adversarial training is split into two steps.

In the first step, it minimizes the following objective function for discriminator D for data x and labels y: DISPLAYFORM0 In the second step, it minimizes the distance of feature matching to optimize the generator G: DISPLAYFORM1 where DISPLAYFORM2 and D (−3) (x) are the outputs from the (n − 3)-th layer for a net with n layers.

In this subsection, we prove the theorem of vanishing gradients on the generator for Improved GAN.

This explains why the Improved GAN lacks training stability, as showed on some datasets, such as MNIST (cf.

Section 4).Theorem 2.1 (Vanishing gradients on the generator for Improved GAN) Let g θ : Z → X be a differentiable function that induces a distribution P g .

Let P r be the real data distribution.

Let D be a differentiable discriminator bounded by T , i.e., D(x) 2 ≤ T .

If the discriminator is trained to converge, i.e., D − D * 2 < , and DISPLAYFORM0 Proof 2.1 See Appendix A.1.This theorem implies that the generator gradients vanish when the discriminator is trained to converge.

In this case, the generator training saturates, which explains the training instability phenomenon of the Improved GAN.

The way to solve this problem is our next question.

In this section, we propose a new method to solve the instability issue of the Improved GAN by using collaborative training between two GANs.

These two GANs contribute to the adversarial training from two different perspectives, which may help avoid the drawbacks of each one.

This is the basic idea behind the proposed method.

The detailed procedure of CTW-GAN can be summarized as a minimax game carried out in two steps:At the first step, the discriminators D c and D w are optimized simultaneously: DISPLAYFORM0 where DISPLAYFORM1 At the second step, the generator G is then optimized by applying the optimized two discriminators D c , D w to G: DISPLAYFORM2 where DISPLAYFORM3 The overall architecture for CTW-GAN is described by Figure 1 , where x r and x u stand for labeled and unlabeled data respectively:Figure 1: Architecture for CTW-GAN

Bearing in mind the generator vanishing gradients theorem for Improved GAN, we may ask if a similar problem exists for our proposed CTW-GAN.

In the following, we prove that our proposed method does not have the vanishing gradients issue on the generator, which therefore improves the training stability of the original Improved GAN.Theorem 3.1 Let P r be any distribution.

Let P θ be the distribution of g θ (z) with z being a random variable with a density p and g θ a continuous function with respect to θ.

Then there is a set of solutions D c , D w to the problem DISPLAYFORM0 and we have DISPLAYFORM1 where the last term is the gradients of the Wasserstein distance W (P r , P θ ), i.e., DISPLAYFORM2 when the term L(D c , D w ) is well-defined.

Proof 3.1 see Appendix A.2.Remark: the above D w L ≤ 1 is required to be 1-Lipschitz.

The constraint can be realized by weight clipping or gradient penalty BID4 .The proposed algorithm is described as follows:Algorithm 1 CTW-GAN with gradient penalty: Require: Gradient penalty λ p = 10, generator weightλ g , and Adam hyperparameter α = 0.0001 Require: Initial parameter θ w for D w , θ c for D c and θ g for G.while θ g does not converge do for i = 1 · · · m do Sample a real data x ∼ p r and a noisy data DISPLAYFORM3 Sample a real data x ∼ p r , a noisy data z ∼ p(z) and a random variable ∼ U (0, 1) DISPLAYFORM4

In this section, we shall present the experiments to evaluate the proposed method.

Our evaluation goals are twofold.

On one hand, we evaluate the stability of CTW-GAN in comparison to that of the original Improved GAN to see whether our proposed method improves the training stability or not.

On the other hand, we evaluate whether the proposed method achieves comparable classification performance to the original Improved GAN.

To this end, we run experiments on two datasets: MNIST and CIFAR-10.

MNIST includes 50, 000 training images, 10, 000 validation images and 10, 000 testing images, which contain handwritten digits with size 28 × 28.Following BID14 , we randomly select a small set of labeled data from the 60, 000 training and validation set to perform semi-supervised learning with the selection size of 20, 50, 100, and 200 labeled examples.

We run our experiments 9 times by giving the program different seeds.

We use the seeds from 1 − 9.

For each seed, the labeled data is selected so as to have a balanced number of examples from each class.

The rest of the training images are used as unlabeled data.

In our method, we use three networks whose architectures are described in FIG0 .

We use batch normalization and add Gaussian noise to the output of each layer of the two discriminators as the original Improved GAN does BID14 .

We only tune the parameter λ = 0.1, 0.5 from two values on the MNIST dataset.

We do not tune any other parameters, such as learning rate, step size, etc.

: we keep these as in the original Improved GAN.

The results shown in TAB1 are reported with λ = 0.1, the threshold for gradient penalty is 10 and n critic = 5: DISPLAYFORM0 From the results, we can easily see that the original improved GAN has one or two out of nine runs for training failure (unexpected high error rates and poor generate image quality).

However, for our proposed method, no training failure occurs.

This shows that our method improves the training stability indeed.

On the other hand, besides making the training process more stable, our proposed method does not reduce the classification accuracy at all, which is beyond our original purpose of avoiding training instability of the Improved GAN.

Reasoning it, it may imply that the information explored by the two discriminators may be very different, thus reflecting a distinct Method n=50 n=100 n=200 DGN BID6 3.33(±0.14) Virtual Adversarial BID11 2.12 Cat-GAN BID15 1.91 (±0.10) Skip Deep Generative Model BID8 1.32 (±0.07) Ladder network BID12 1.06 (±0.37) Auxiliary Deep Generative Model BID8 0 GAN (Springenberg, 2015) 20.40 (±0.47) Ladder network BID12 19.58 (±0.46) Improved-GAN 0.1726 (±0.0032) Ours 0.1713 (±0.0014) There is no failure case found in three runs for the original GAN on CIFAR-10.

We use 4000 labeled samples.source of information for data representation.

Utilizing those different information sources may help to improve classification accuracy, as long as the source of information is meaningful to some extent, or at least not noise.

In our method, we use a very simple network for D w with only two layers.

It may be possible to further improve classification performance if a network with more layers is used.

We leave it for future work.

In this section, we test our proposed method on the data set of CIFAR-10.

CIFAR-10 consists of colored images belonging to 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

There are 50,000 training and 10,000 testing samples with the size of 32 × 32.

We split 5,000 training data of CIFAR-10 for validation if needed.

Following BID14 , we use a 9 layer deep convolutional network with dropout and weight normalization for the discriminator D c .

The generator G is a 4 layer deep CNN with batch normalization.

We use a very simple network with three layers for the discriminator D w , due to the limiting GPU resource.

The network architectures are given in FIG1 .

TAB2 summarizes our results on the semi-supervised learning task.

On CIFAR-10 dataset, it is interesting to see that there is no failure case found for the Improved GAN in three runs at the moment.

From the theoretical viewpoint, this may be due to the abundant richness of the image features in color being much harder to be modeled by the neural nets than that of MNIST in grayscale.

Thus, the discriminator D c trained on CIFAR-10 does not as easily converge as the one trained on MNIST, such that the gradients on the generator do not vanish.

However, it does not mean that this possibility is avoided.

In certain cases, as long as the discriminator is trained to converge., e.g., running more iterations than the generator, the gradients on the generator will surely vanish, as theoretically guaranteed by Theorem 2.1.

On the other hand, our proposed method is still able to achieve comparable results to the original Improved GAN, besides providing a theoretical guarantee to the training stability.

Due to the limiting GPU resource, we use a very simple network for D w .

In this sense, the characteristics captured by this network may not be rich enough.

However, the results showed that even with the very simple network, the classification performance obtained is roughly comparable to that of the Improved GAN.

We expect that it would be possibly improved further if we have more GPU resources and are able to train a deeper network for D w .

In the paper, we study the training instability issue of semi-supervised improved GAN.

We have found that the training instability is mainly due to the vanishing gradients on the generator of the Improved GAN.

In order to make the training of the Improved GAN more stable, we propose a collaborative training method to combine Wasserstein GAN with the semi-supervised improved GAN.

Both theoretical analysis and experimental results on MNIST and CIFAR-10 have shown the effectiveness of the proposed method to improve training stability of the Improved GAN.

In addition, it also achieves the classification accuracy comparable to the original Improved GAN.

We would like to thank National Natural Science Foundation of China FORMULA0 for previously supporting the authors to prepare for the knowledge and skills demanded by this work.

<|TLDR|>

@highlight

Improve Training Stability of Semi-supervised Generative Adversarial Networks with Collaborative Training