Generative adversarial nets (GANs) are widely used to learn the data sampling process and their performance may heavily depend on the loss functions, given a limited computational budget.

This study revisits MMD-GAN that uses the maximum mean discrepancy (MMD) as the loss function for GAN and makes two contributions.

First, we argue that the existing MMD loss function may discourage the learning of fine details in data as it attempts to contract the discriminator outputs of real data.

To address this issue, we propose a repulsive loss function to actively learn the difference among the real data by simply rearranging the terms in MMD.

Second, inspired by the hinge loss, we propose a bounded Gaussian kernel to stabilize the training of MMD-GAN with the repulsive loss function.

The proposed methods are applied to the unsupervised image generation tasks on CIFAR-10, STL-10, CelebA, and LSUN bedroom datasets.

Results show that the repulsive loss function significantly improves over the MMD loss at no additional computational cost and outperforms other representative loss functions.

The proposed methods achieve an FID score of 16.21 on the CIFAR-10 dataset using a single DCGAN network and spectral normalization.

Generative adversarial nets (GANs) BID7 ) are a branch of generative models that learns to mimic the real data generating process.

GANs have been intensively studied in recent years, with a variety of successful applications (Karras et al. (2018) ; Li et al. (2017b) ; Lai et al. (2017) ; Zhu et al. (2017) ; BID13 ).

The idea of GANs is to jointly train a generator network that attempts to produce artificial samples, and a discriminator network or critic that distinguishes the generated samples from the real ones.

Compared to maximum likelihood based methods, GANs tend to produce samples with sharper and more vivid details but require more efforts to train.

Recent studies on improving GAN training have mainly focused on designing loss functions, network architectures and training procedures.

The loss function, or simply loss, defines quantitatively the difference of discriminator outputs between real and generated samples.

The gradients of loss functions are used to train the generator and discriminator.

This study focuses on a loss function called maximum mean discrepancy (MMD), which is well known as the distance metric between two probability distributions and widely applied in kernel two-sample test BID8 ).

Theoretically, MMD reaches its global minimum zero if and only if the two distributions are equal.

Thus, MMD has been applied to compare the generated samples to real ones directly (Li et al. (2015) ; BID5 ) and extended as the loss function to the GAN framework recently (Unterthiner et al. (2018) ; Li et al. (2017a) ; ).In this paper, we interpret the optimization of MMD loss by the discriminator as a combination of attraction and repulsion processes, similar to that of linear discriminant analysis.

We argue that the existing MMD loss may discourage the learning of fine details in data, as the discriminator attempts to minimize the within-group variance of its outputs for the real data.

To address this issue, we propose a repulsive loss for the discriminator that explicitly explores the differences among real data.

The proposed loss achieved significant improvements over the MMD loss on image generation tasks of four benchmark datasets, without incurring any additional computational cost.

Furthermore, a bounded Gaussian kernel is proposed to stabilize the training of discriminator.

As such, using a single kernel in MMD-GAN is sufficient, in contrast to a linear combination of kernels used in Li et al. (2017a) and .

By using a single kernel, the computational cost of the MMD loss can potentially be reduced in a variety of applications.

The paper is organized as follows.

Section 2 reviews the GANs trained using the MMD loss (MMD-GAN) .

We propose the repulsive loss for discriminator in Section 3, introduce two practical techniques to stabilize the training process in Section 4, and present the results of extensive experiments in Section 5.

In the last section, we discuss the connections between our model and existing work.

In this section, we introduce the GAN model and MMD loss.

Consider a random variable X ∈ X with an empirical data distribution P X to be learned.

A typical GAN model consists of two neural networks: a generator G and a discriminator D. The generator G maps a latent code z with a fixed distribution P Z (e.g., Gaussian) to the data space X : y = G(z) ∈ X , where y represents the generated samples with distribution P G .

The discriminator D evaluates the scores D(a) ∈ R d of a real or generated sample a. This study focuses on image generation tasks using convolutional neural networks (CNN) for both G and D.Several loss functions have been proposed to quantify the difference of the scores between real and generated samples: {D(x)} and {D(y)}, including the minimax loss and non-saturating loss BID7 ), hinge loss (Tran et al. (2017) ), Wasserstein loss ; BID10 ) and maximum mean discrepancy (MMD) (Li et al. (2017a) ; ) (see Appendix B.1 for more details).

Among them, MMD uses kernel embedding φ(a) = k(·, a) associated with a characteristic kernel k such that φ is infinite-dimensional and φ(a), φ(b) H = k(a, b).

The squared MMD distance between two distributions P and Q is DISPLAYFORM0 The kernel k(a, b) measures the similarity between two samples a and b. BID8 proved that, using a characteristic kernel k, M ,P G ) reaches its minimum if and only if P X = P G (Li et al. (2017a) ).

Thus, the objective functions for G and D could be (Li et al. (2017a) ; ): min DISPLAYFORM1 ) MMD-GAN has been shown to be more effective than the model that directly uses MMD as the loss function for the generator G (Li et al. (2017a)).

Liu et al. (2017) showed that MMD and Wasserstein metric are weaker objective functions for GAN than the Jensen-Shannon (JS) divergence (related to minimax loss) and total variation (TV) distance (related to hinge loss).

The reason is that convergence of P G to P X in JS-divergence and TV distance also implies convergence in MMD and Wasserstein metric.

Weak metrics are desirable as they provide more information on adjusting the model to fit the data distribution (Liu et al. (2017) ).

Nagarajan & Kolter (2017) proved that the GAN trained using the minimax loss and gradient updates on model parameters is locally exponentially stable near equilibrium, while the GAN using Wasserstein loss is not.

In Appendix A, we demonstrate that the MMD-GAN trained by gradient descent is locally exponentially stable near equilibrium.

In this section, we interpret the training of MMD-GAN (using L First, consider a linear discriminant analysis (LDA) model as the discriminator.

The task is to find a projection w to maximize the between-group variance w T µ x − w T µ y and minimize the withingroup variance w T (Σ x + Σ y )w, where µ and Σ are group mean and covariance.

In MMD-GAN, the neural-network discriminator works in a similar way as LDA.

By minimizing L att D , the discriminator D tackles two tasks: DISPLAYFORM0 , causes the two groups {D(x)} and {D(y)} to repel each other (see FIG1 , or maximize betweengroup variance; and 2) D increases DISPLAYFORM1 e. contracts {D(x)} and {D(y)} within each group (see FIG1 , or minimize the within-group variance.

We refer to loss functions that contract real data scores as attractive losses.

We argue that the attractive loss L att D (Eq. 3) has two issues that may slow down the GAN training:1.

The discriminator D may focus more on the similarities among real samples (in order to contract {D(x)}) than the fine details that separate them.

Initially, G produces low-quality samples and it may be adequate for D to learn the common features of {x} in order to distinguish between {x} and {y}. Only when {D(y)} is sufficiently close to {D(x)} will D learn the fine details of {x} to be able to separate {D(x)} from {D(y)}. Consequently, D may leave out some fine details in real samples, thus G has no access to them during training.2.

As shown in FIG1 , the gradients on D(y) from the attraction (blue arrows) and repulsion (orange arrows) terms in L att D (and thus L mmd G ) may have opposite directions during training.

Their summation may be small in magnitude even when D(y) is far away from D(x), which may cause G to stagnate locally.

Therefore, we propose a repulsive loss for D to encourage repulsion of the real data scores {D(x)}: DISPLAYFORM2 The generator G uses the same MMD loss L mmd G as before (see Eq. 2).

Thus, the adversary lies in the fact that D contracts {D(y)} via maximizing FIG1 ) while G expands {D(y)} (see FIG1 ).

Additionally, D also learns to separate the real data by minimizing DISPLAYFORM3 DISPLAYFORM4 , which actively explores the fine details in real samples and may result in more meaningful gradients for G. Note that in Eq. 4, D does not explicitly push the average score of {D(y)} away from that of {D(x)} because it may have no effect on the pair-wise sample distances.

But G aims to match the average scores of both groups.

Thus, we believe, compared to the model using DISPLAYFORM5 and L rep D is less likely to yield opposite gradients when {D(y)} and {D(x)} are distinct (see FIG1 ).

In Appendix A, we demonstrate that GAN trained using gradient descent and the repulsive MMD loss (L At last, we identify a general form of loss function for the discriminator D: DISPLAYFORM6

In this section, we propose two approaches to stabilize the training of MMD-GAN: 1) a bounded kernel to avoid the saturation issue caused by an over-confident discriminator; and 2) a generalized power iteration method to estimate the spectral norm of a convolutional kernel, which was used in spectral normalization on the discriminator in all experiments in this study unless specified otherwise.

For MMD-GAN, the following two kernels have been used:• Gaussian radial basis function (RBF), or Gaussian kernel (Li et al. (2017a) DISPLAYFORM0 2 ) where σ > 0 is the kernel scale or bandwidth.• Rational quadratic kernel DISPLAYFORM1 , where the kernel scale α > 0 corresponds to a mixture of Gaussian kernels with a Gamma(α, 1) prior on the inverse kernel scales σ −1 .It is interesting that both studies used a linear combination of kernels with five different kernel scales, i.e., DISPLAYFORM2 , where σ i ∈ {1, 2, 4, 8, 16}, α i ∈ {0.2, 0.5, 1, 2, 5} (see FIG0 and 2c for illustration).

We suspect the reason is that a single kernel k(a, b) is saturated when the distance a − b is either too large or too small compared to the kernel scale (see FIG0 and 2d), which may cause diminishing gradients during training.

Both Li et al. (2017a) and applied penalties on the discriminator parameters but not to the MMD loss itself.

Thus the saturation issue may still exist.

Using a linear combination of kernels with different kernel scales may alleviate this issue but not eradicate it.

Inspired by the hinge loss (see Appendix B.1), we propose a bounded RBF (RBF-B) kernel for the discriminator.

The idea is to prevent D from pushing {D(x)} too far away from {D(y)} and causing saturation.

For L att D in Eq. 3, the RBF-B kernel is: DISPLAYFORM3 For L rep D in Eq. 4, the RBF-B kernel is: DISPLAYFORM4 where b l and b u are the lower and upper bounds.

As such, a single kernel is sufficient and we set σ = 1, b l = 0.25 and b u = 4 in all experiments for simplicity and leave their tuning for future work.

It should be noted that, like the case of hinge loss, the RBF-B kernel is used only for the discriminator to prevent it from being over-confident.

The generator is always trained using the original RBF kernel, thus we retain the interpretation of MMD loss L mmd G as a metric.

RBF-B kernel is among many methods to address the saturation issue and stabilize MMD-GAN training.

We found random sampling kernel scale, instance noise (Sønderby et al. (2017) ) and label smoothing (Szegedy et al. (2016); Salimans et al. (2016) ) may also improve the model performance and stability.

However, the computational cost of RBF-B kernel is relatively low.

Without any Lipschitz constraints, the discriminator D may simply increase the magnitude of its outputs to minimize the discriminator loss, causing unstable training 3 .

Spectral normalization divides the weight matrix of each layer by its spectral norm, which imposes an upper bound on the magnitudes of outputs and gradients at each layer of D (Miyato et al. (2018) ).

However, to estimate the spectral norm of a convolution kernel, Miyato et al. (2018) reshaped the kernel into a matrix.

We propose a generalized power iteration method to directly estimate the spectral norm of a convolution kernel (see Appendix C for details) and applied spectral normalization to the discriminator in all experiments.

In Appendix D.1, we explore using gradient penalty to impose the Lipschitz constraint BID10 ; ; ) for the proposed repulsive loss.

In this section, we empirically evaluate the proposed 1) repulsive loss L (Li et al. (2017a) ) and rational quadratic kernel (MMD-rq) ), as well as non-saturating loss BID7 ) and hinge loss (Tran et al. (2017) ).

To show the efficacy of RBF-B kernel, we applied it to both L (2018) ).

DISPLAYFORM0

Dataset: The loss functions were evaluated on four datasets: 1) CIFAR-10 (50K images, 32 × 32 pixels) (Krizhevsky & Hinton (2009)) ; 2) STL-10 (100K images, 48 × 48 pixels) BID3 ); 3) CelebA (about 203K images, 64 × 64 pixels) (Liu et al. FORMULA8 ); and 4) LSUN bedrooms (around 3 million images, 64×64 pixels) (Yu et al. FORMULA8 ).

The images were scaled to range [−1, 1] to avoid numeric issues.

FORMULA8 ) was used in the generator, and spectral normalization with the generalized power iteration (see Appendix C) in the discriminator.

For MMD related losses, the dimension of discriminator output layer was set to 16; for non-saturating loss and hinge loss, it was 1.

In Appendix D.2, we investigate the impact of discriminator output dimension on the performance of repulsive loss. ) and thus omitted.3 On LSUN-bedroom, MMD-rbf and MMD-rq did not achieve reasonable results and thus are omitted.

Hyper-parameters: We used Adam optimizer (Kingma & Ba FORMULA8 ) with momentum parameters β 1 = 0.5, β 2 = 0.999; two-timescale update rule (TTUR) BID12 ) with two learning rates (ρ D , ρ G ) chosen from {1e-4, 2e-4, 5e-4, 1e-3} (16 combinations in total); and batch size 64.

Fine-tuning on learning rates may improve the model performance, but constant learning rates were used for simplicity.

All models were trained for 100K iterations on CIFAR-10, STL-10, CelebA and LSUN bedroom datasets, with n dis = 1, i.e., one discriminator update per generator update 4 .

For MMD-rbf, the kernel scales σ i ∈ {1, √ 2, 2, 2 √ 2, 4} were used due to a better performance than the original values used in Li et al. (2017a) .

For MMD-rq, α i ∈ {0.2, 0.5, 1, 2, 5}. For MMD-rbf-b, MMD-rep, MMD-rep-b, a single Gaussian kernel with σ = 1 was used.

Evaluation metrics: Inception score (IS) (Salimans et al. (2016) ), Fréchet Inception distance (FID) BID12 ) and multi-scale structural similarity (MS-SSIM) (Wang et al. (2003) ) were used for quantitative evaluation.

Both IS and FID are calculated using a pre-trained Inception model (Szegedy et al. (2016) ).

Higher IS and lower FID scores indicate better image quality.

MS-SSIM calculates the pair-wise image similarity and is used to detect mode collapses among images of the same class (Odena et al. (2017) ).

Lower MS-SSIM values indicate perceptually more diverse images.

For each model, 50K randomly generated samples and 50K real samples were used to calculate IS, FID and MS-SSIM.

TAB0 shows the Inception score, FID and MS-SSIM of applying different loss functions on the benchmark datasets with the optimal learning rate combinations tested experimentally.

Note that the same training setup (i.e., DCGAN + BN + SN + TTUR) was applied for each loss function.

We observed that: 1) MMD-rep and MMD-rep-b performed significantly better than MMD-rbf and MMD-rbf-b respectively, showing the proposed repulsive loss L rep D (Eq. 4) greatly improved over the attractive loss L att D (Eq. 3); 2) Using a single kernel, MMD-rbf-b performed better than MMD-rbf and MMD-rq which used a linear combination of five kernels, indicating that the kernel saturation may be an issue that slows down MMD-GAN training; 3) MMD-rep-b performed comparable or better than MMD-rep on benchmark datasets where we found the RBF-B kernel managed to stabilize MMD-GAN training using repulsive loss.

4) MMD-rep and MMD-rep-b performed significantly better than the non-saturating and hinge losses, showing the efficacy of the proposed repulsive loss.

Additionally, we trained MMD-GAN using the general loss L D,λ (Eq. 5) for discriminator and L mmd G (Eq. 2) for generator on the CIFAR-10 dataset.

Each color bar represents the FID score using a learning rate combination (ρ D , ρ G ), in the order of (1e-4, 1e-4), (1e-4, 2e-4),...,(1e-3, 1e-3).

The discriminator was trained using L D,λ (Eq. 5) with λ ∈ {-1, -0.5, 0, 0.5, 1, 2}, and generator using L mmd G (Eq. 2).

We use the FID> 30 to indicate that the model diverged or produced poor results.of MMD-GAN with RBF and RBF-B kernel 5 .

Note that when λ = −1, the models are essentially MMD-rbf (with a single Gaussian kernel) and MMD-rbf-b when RBF and RBF-B kernel are used respectively.

We observed that: 1) the model performed well using repulsive loss (i.e., λ ≥ 0), with λ = 0.5, 1 slightly better than λ = −0.5, 0, 2; 2) the MMD-rbf model can be significantly improved by simply increasing λ from −1 to −0.5, which reduces the attraction of discriminator on real sample scores; 3) larger λ may lead to more diverged models, possibly because the discriminator focuses more on expanding the real sample scores over adversarial learning; note when λ 1, the model would simply learn to expand all real sample scores and pull the generated sample scores to real samples', which is a divergent process; 4) the RBF-B kernel managed to stabilize MMD-rep for most diverged cases but may occasionally cause the FID score to rise up.

The proposed methods were further evaluated in Appendix A, C and D. In Appendix A.2, we used a simulation study to show the local stability of MMD-rep trained by gradient descent, while its global stability is not guaranteed as bad initialization may lead to trivial solutions.

The problem may be alleviated by adjusting the learning rate for generator.

In Appendix C.3, we showed the proposed generalized power iteration (Section 4.2) imposes a stronger Lipschitz constraint than the method in Miyato et al. (2018) , and benefited MMD-GAN training using the repulsive loss.

Moreover, the RBF-B kernel managed to stabilize the MMD-GAN training for various configurations of the spectral normalization method.

In Appendix D.1, we showed the gradient penalty can also be used with the repulsive loss.

In Appendix D.2, we showed that it was better to use more than one neuron at the discriminator output layer for the repulsive loss.

The discriminator outputs may be interpreted as a learned representation of the input samples.

FIG8 visualizes the discriminator outputs learned by the MMD-rbf and proposed MMD-rep methods on CIFAR-10 dataset using t-SNE (van der Maaten FORMULA4 ).

MMD-rbf ignored the class structure in data (see FIG8 ) while MMD-rep learned to concentrate the data from the same class and separate different classes to some extent FIG8 .

This is because the discriminator D has to actively learn the data structure in order to expands the real sample scores {D(x)}. Thus, we speculate that techniques reinforcing the learning of cluster structures in data may further improve the training of MMD-GAN.In addition, the performance gain of proposed repulsive loss (Eq. 4) over the attractive loss (Eq. 3) comes at no additional computational cost.

In fact, by using a single kernel rather than a linear combination of kernels, MMD-rep and MMD-rep-b are simpler than MMD-rbf and MMD-rq.

Besides, given a typically small batch size and a small number of discriminator output neurons (64 and 16 in our experiments), the cost of MMD over the non-saturating and hinge loss is marginal compared to the convolution operations.

In Appendix D.3, we provide some random samples generated by the methods in our study.

This study extends the previous work on MMD-GAN (Li et al. (2017a) ) with two contributions.

First, we interpreted the optimization of MMD loss as a combination of attraction and repulsion processes, and proposed a repulsive loss for the discriminator that actively learns the difference among real data.

Second, we proposed a bounded Gaussian RBF (RBF-B) kernel to address the saturation issue.

Empirically, we observed that the repulsive loss may result in unstable training, due to factors including initialization (Appendix A.2), learning rate ( FIG7 and Lipschitz constraints on the discriminator (Appendix C.3).

The RBF-B kernel managed to stabilize the MMD-GAN training in many cases.

Tuning the hyper-parameters in RBF-B kernel or using other regularization methods may further improve our results.

The theoretical advantages of MMD-GAN require the discriminator to be injective.

The proposed repulsive loss (Eq. 4) attempts to realize this by explicitly maximizing the pair-wise distances among the real samples.

Li et al. (2017a) achieved the injection property by using the discriminator as the encoder and an auxiliary network as the decoder to reconstruct the real and generated samples, which is more computationally extensive than our proposed approach.

On the other hand, ; imposed a Lipschitz constraint on the discriminator in MMD-GAN via gradient penalty, which may not necessarily promote an injective discriminator.

The idea of repulsion on real sample scores is in line with existing studies.

It has been widely accepted that the quality of generated samples can be significantly improved by integrating labels (Odena et al. (2017); Miyato & Koyama (2018) ; Zhou et al. (2018) ) or even pseudo-labels generated by k-means method BID9 ) in the training of discriminator.

The reason may be that the labels help concentrate the data from the same class and separate those from different classes.

Using a pre-trained classifier may also help produce vivid image samples BID14 ) as the learned representations of the real samples in the hidden layers of the classifier tend to be well separated/organized and may produce more meaningful gradients to the generator.

At last, we note that the proposed repulsive loss is orthogonal to the GAN studies on designing network structures and training procedures, and thus may be combined with a variety of novel techniques.

For example, the ResNet architecture BID11 ) has been reported to outperform the plain DCGAN used in our experiments on image generation tasks (Miyato et al. (2018) ; BID10 ) and self-attention module may further improve the results (Zhang et al. (2018) ).

On the other hand, Karras et al. (2018) proposed to progressively grows the size of both discriminator and generator and achieved the state-of-the-art performance on unsupervised training of GANs on the CIFAR-10 dataset.

Future work may explore these directions.

This section demonstrates that, under mild assumptions, MMD-GAN trained by gradient descent is locally exponentially stable at equilibrium.

It is organized as follows.

The main assumption and proposition are presented in Section A.1, followed by simulation study in Section A.2 and proof in Section A.3.

We discuss the indications of assumptions on the discriminator of GAN in Section A.4.

We consider GAN trained using the MMD loss L DISPLAYFORM0 where Thus in contrast to Assumption 1, we assume Assumption 2.

For GANs using MMD loss in Eq. S1, and random initialization on parameters, at equilibrium, DISPLAYFORM1 DISPLAYFORM2 is not constant almost everywhere.

We use a simulation study in Section A.2 to show that D θ * D (x) = 0 does not hold in general for MMD loss.

Based on Assumption 2, we propose the following proposition and prove it in Appendix A.3: Proposition 1.

If there exists θ * G ∈ Θ G such that P θ * G = P X , then GANs with MMD loss in Eq. S1 has equilibria (θ * G , θ D ) for any θ D ∈ Θ D .

Moreover, the model trained using gradient descent methods is locally exponentially stable at (θ * DISPLAYFORM3 There may exist non-realizable cases where the mapping between P Z and P X cannot be represented by any generator G θ G with θ G ∈ Θ G .

In Section A.2, we use a simulation study to show that both the attractive MMD loss L att D (Eq. S1b) and the proposed repulsive loss L rep D (Eq. S1c) may be locally stable and leave the proof for future work.

In this section, we reused the example from Nagarajan & Kolter (2017) to show that GAN trained using the MMD loss in Eq. S1 is locally stable.

Consider a two-parameter MMD-GAN with uniform latent distribution P Z over [−1, 1], generator G(z) = w 1 z, discriminator D(x) = w 2 x 2 , and Gaussian kernel k (a) the data distribution P X is the same as P Z , i.e., uniform over [−1, 1], thus P X is realizable; DISPLAYFORM0 Figure S1: Streamline plots of MMD-GAN using the MMD-rbf and the MMD-rep model on distributions: P Z = U(−1, 1), P X = U(−1, 1) or P X = N (0, 1).

In (a) and (b), the equilibria satisfying P G = P X lie on the line w 1 = 1.

In (c), the equilibrium lies around point (1.55, 0.74); in (d), it is around (1.55, 0.32).(b) P X is standard Gaussian, thus non-realizable for any w 1 ∈ R. FIG1 shows that MMD-GAN are locally stable in both cases and D θ * D (x) = 0 does not hold in general for MMD loss.

However, MMD-rep may not be globally stable for the tested cases: initialization of (w 1 , w 2 ) in some regions may lead to the trivial solution w 2 = 0 (see FIG1 and S1d).

We note that by decreasing the learning rate for G, the area of such regions decreased.

At last, it is interesting to note that both MMD-rbf and MMD-rep had the same nontrivial solution w 1 ≈ 1.55 for generator in the non-realizable cases (see FIG1 and S1d).

This section divides the proof for Proposition 1 into two parts.

First, we show that GAN with the MMD loss in Eq. S1 has equilibria for any parameter configuration of discriminator D; second, we prove the model is locally exponentially stable.

For convenience, we consider the general form of discriminator loss in Eq. 5: DISPLAYFORM0 which has L att D and L rep D as the special cases when λ equals −1 and 1 respectively.

Consider real data X r ∼ P X , latent variable Z ∼ P Z and generated variable Y g = G θ G (Z).

Let x r , z, y g be their samples.

Denote ∇ .

DISPLAYFORM1 where L D and L G are the losses for D and G respectively.

Assume an isotropic stationary kernel k(a, b) = k I ( a − b ) BID6 ) is used in MMD.

We first show:Proposition 1 (Part 1).

If there exists θ * G ∈ Θ G such that P θ * G = P X , the GAN with the MMD loss in Eq. S1a and Eq. S2 has equilibria (θ * DISPLAYFORM2 where k is the kernel of MMD.

The gradients of MMD loss are DISPLAYFORM3 ∼ P G , an unbiased estimator of the squared MMD is BID8 ) We proceed to prove the model stability.

First, following Theorem 5 in BID8 and Theorem 4 in Li et al. (2017a) , it is straightforward to see: FORMULA13 ).

Consider a non-linear system of parameters (θ, γ): θ = h 1 (θ, γ),γ = h 2 (θ, γ) with an equilibrium point at (0, 0).

Let there exist such that ∀γ ∈ DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 is a Hurwitz matrix, the non-linear system is exponentially stable.

Proposition 1 (Part 2).

At equilibrium P θ * G = P X , the GAN trained using MMD loss and gradient descent methods is locally exponentially stable at (θ DISPLAYFORM0 ∂b∂c .

Based on Eq. S3, we have DISPLAYFORM1 where ⊗ is the kronecker product.

At equilibrium, consider a sequence of N samples DISPLAYFORM2 DISPLAYFORM3 Given Lemma A.1 and fact that J GG is the Hessian matrix of M DISPLAYFORM4 is locally constant along some directions in the parameter space of G. As a result, null(J GG ) ⊆ null(J DG ) because varying θ * G along these directions has no effect on D. Following Lemma C.3 of Nagarajan & Kolter (2017), we consider eigenvalue decomposition DISPLAYFORM5 .

Thus, the projections γ G = T G θ G are orthogonal to null(J GG ).

Then, the Jacobian corresponding to the projected system has the form DISPLAYFORM6 , where J GG is negative definite.

Moreover, on all directions exclude those described by J GG , the system is surrounded by a neighborhood of equilibia at least locally.

According to Lemma A.2, the system is exponentially stable.

This section shows that constant discriminator output DISPLAYFORM0 may have no discrimination power.

First, we make the following assumptions:Assumption 3.

1.

D is a multilayer perceptron where each layer l can be factorized into an affine transform and an element-wise activation function f l .

2.

Each activation function f l ∈ C 0 ; furthermore, f l has a finite number of discontinuities and f l ∈ C 06 .

3.

Input data to D is continuous and its support S is compact in R d with non-zero measure in each dimension and d > 1 7 .Based on Assumption 3, we have the following proposition:Proposition 2.

If ∀x ∈ S, D(x) = c, where c is constant, then there always exists distortion δx such that x + δx ∈ S and D(x + δx) = c.

are model weights and biases, f is an activation function satisfying Assumption 3.

For x ∈ S, since D(x) = c, we have h(x) ∈ null(W 2 ).

Furthermore: DISPLAYFORM0 has unique solution for any k ∈ R as long as k · h(x) is within the output range of f .

DISPLAYFORM1 and n is the nullity of W 2 .

Let the projected support beŜ. Thus, DISPLAYFORM2 T with z c = 0.Consider the Jacobian: DISPLAYFORM3 where DISPLAYFORM4 is the input to activation, or pre-activations.

SinceŜ is continuous and compact, it has infinite number of boundary points {x b } for d > 1.

Consider one boundary pointx b and its normal line δx b .

Let > 0 be a small scalar such thatx b − δx b ∈ S andx b + δx b ∈Ŝ.• For linear activation, ∇Σ = I and J is constant.

Then z c remains 0 DISPLAYFORM5 there exists z such that h(x + δx) ∈ null(W 2 ).•

For nonlinear activations, assume f has N discontinuities.

Since U x T 0 T T + b 1 = c has unique solution for any vector c, the boundary points {x b } cannot yield pre-activations {a b } that all lie on the discontinuities in any of the d h directions.

Though we might need to sample d N +1 h points in the worst case to find an exception, there are infinite number of exceptions.

Letx b be a sample where {a b } does not lie on the discontinuities in any direction.

Because f is continuous, z c remains 0 forx b + δx b , i.e., there exists z such that h(x + δx) ∈ null(W 2 ).In conclusion, we can always find δx such that x + δx / ∈ S and D(x + δx) = c.

cannot discriminate against fake samples with distortions to the original data.

In contrast, Assumption 2 and Lemma A.1 guarantee that, at equilibrium, the discriminator trained using MMD loss function is effective against such fake samples given a large number of i.i.d.

test samples BID8 ).

Several loss functions have been proposed to quantify the difference between real and generated sample scores, including: (assume linear activation is used at the last layer of D)• The Minimax loss BID7 ): Softplus(D(G(z) ))] and L G = −L D , which can be derived from the Jensen-Shannon (JS) divergence between P X and the model distribution P G .

DISPLAYFORM0 • The non-saturating loss BID7 ), which is a variant of the minimax loss with the same L D and DISPLAYFORM1 • The Hinge loss (Tran et al. (2017) ): DISPLAYFORM2 , which is notably known for usage in support vector machines and is related to the total variation (TV) distance (Nguyen et al. (2009) ).•

The Wasserstein loss ; BID10 ), which is derived from the Wasserstein distance between P X and P G : DISPLAYFORM3 , where D is subject to some Lipschitz constraint.•

The maximum mean discrepancy (MMD) (Li et al. (2017a); ), as described in Section 2.

For unsupervised image generation tasks on CIFAR-10 and STL-10 datasets, the DCGAN architecture from Miyato et al. (2018) was used.

For CelebA and LSUN bedroom datasets, we added more layers to the generator and discriminator accordingly.

See TAB0 and S2 for details.

TAB0 : DCGAN models for image generation on CIFAR-10 (h = w = 4, H = W = 32) and STL-10 (h = w = 6, H = W = 48) datasets.

For non-saturating loss and hinge loss, s = 1; for MMD-rand, MMD-rbf, MMD-rq, s = 16.

For a weight matrix W , the spectral norm is defined as σ(W ) = max v 2 ≤1 W v 2 .

The PIM is used to estimate σ(W ) (Miyato et al. (2018) ), which iterates between two steps: DISPLAYFORM0 The convolutional kernel W c is a tensor of shape h × w × c in × c out with h, w the receptive field size and c in , c out the number of input/output channels.

To estimate σ(W c ), Miyato et al. (2018) reshaped it into a matrix W rs of shape (hwc in ) × c out and estimated σ(W rs ).We propose a simple method to calculate W c directly based on the fact that convolution operation is linear.

For any linear map T : R m → R n , there exists matrix W L ∈ R n×m such that y = T (x) can be represented as y = W L x. Thus, we may simply substitute W L = ∂y ∂x in the PIM method to estimate the spectral norm of any linear operation.

In the case of convolution operation * , there exist doubly block circulant matrix DISPLAYFORM1 T u which is essentially the transpose convolution of W c on u BID4 ).

Thus, similar to PIM, PICO iterates between the following two steps: DISPLAYFORM2 2.

Do transpose convolution of W c on u to getv; update v =v/ v 2 .Similar approaches have been proposed in Tsuzuku et al. (2018) and Virmaux & Scaman (2018) from different angles, which we were not aware during this study.

In addition, Sedghi et al. (2019) proposes to compute the exact singular values of convolution kernels using FFT and SVD.

In spectral normalization, only the first singular value is concerned, making the power iteration methods PIM and PICO more efficient than FFT and thus preferred in our study.

However, we believe the exact method FFT+SVD (Sedghi et al. (2019) ) may eventually inspire more rigorous regularization methods for GAN.The proposed PICO method estimates the real spectral norm of a convolution kernel at each layer, thus enforces an upper bound on the Lipschitz constant of the discriminator D. Denote the upper bound as LIP PICO .

In this study, Leaky ReLU (LReLU) was used at each layer of D, thus LIP PICO ≈ 1 (Virmaux & Scaman (2018) ).

In practice, however, PICO would often cause the norm of the signal passing through D to decrease to zero, because at each layer,• the signal hardly coincides with the first singular-vector of the convolution kernel; and• the activation function LReLU often reduces the norm of the signal.

Consequently, the discriminator outputs tend to be similar for all the inputs.

To compensate the loss of norm at each layer, the signal is multiplied by a constant C after each spectral normalization.

This essentially enlarges LIP PICO by C K where K is the number of layers in the DCGAN discriminator.

For all experiments in Section 5, we fixed C = 1 0.55 ≈ 1.82 as all loss functions performed relatively well empirically.

In Appendix Section C.3, we tested the effects of coefficient C K on the performance of several loss functions.

PIM (Miyato et al. (2018) ) also enforces an upper bound LIP PIM on the Lipschitz constant of the discriminator D. Consider a convolution kernel W c with receptive field size h × w and stride s. Let σ PICO and σ PIM be the spectral norm estimated by PICO and PIM respectively.

We empirically

In this section, we empirically evaluate the effects of coefficient C K on the performance of PICO and compare PICO against PIM using several loss functions.

We used a similar setup as Section 5.1 with the following adjustments.

Four loss functions were tested: hinge, MMD-rbf, MMD-rep and MMD-rep-b.

Either PICO or PIM was used at each layer of the discriminator.

For PICO, five coefficients C K were tested: 16, 32, 64, 128 and 256 (note this is the overall coefficient for K layers; K = 8 for CIFAR-10 and STL-10; K = 10 for CelebA and LSUN-bedroom; see Appendix B.2).

FID was used to evaluate the performance of each combination of loss function and power iteration method, e.g., hinge + PICO with C K = 16.Results: For each combination of loss function and power iteration method, the distribution of FID scores over 16 learning rate combinations is shown in FIG0 .

We separated well-performed learning rate combinations from diverged or poorly-performed ones using a threshold τ as the diverged cases often had non-meaningful FID scores.

The boxplot shows the distribution of FID scores for goodperformed cases while the number of diverged or poorly-performed cases was shown above each box if it is non-zero.

1) When PICO was used, the hinge, MMD-rbf and MMD-rep methods were sensitive to the choices of C K while MMD-rep-b was robust.

For hinge and MMD-rbf, higher C K may result in better FID scores and less diverged cases over 16 learning rate combinations.

For MMD-rep, higher C K may cause more diverged cases; however, the best FID scores were often achieved with C K = 64 or 128.2) For CIFAR-10, STL-10 and CelebA datasets, PIM performed comparable to PICO with C K = 128 or 256 on four loss functions.

For LSUN bedroom dataset, it is likely that the performance of PIM corresponded to that of PICO with C K > 256.

This implies that PIM may result in a relatively loose Lipschitz constraint on deep convolutional networks.3) MMD-rep-b performed generally better than hinge and MMD-rbf with tested power iteration methods and hyper-parameter configurations.

Using PICO, MMD-rep also achieved generally better FID scores than hinge and MMD-rbf.

This implies that, given a limited computational budget, the proposed repulsive loss may be a better choice than the hinge and MMD loss for the discriminator.

TAB2 shows the best FID scores obtained by PICO and PIM where C K was fixed at 128 for hinge and MMD-rbf, and 64 for MMD-rep and MMD-rep-b.

For hinge and MMD-rbf, PICO performed significantly better than PIM on the LSUN-bedroom dataset and comparably on the rest datasets.

For MMD-rep and MMD-rep-b, PICO achieved consistently better FID scores than PIM.However, compared to PIM, PICO has a higher computational cost which roughly equals the additional cost incurred by increasing the batch size by two (Tsuzuku et al. (2018) ).

This may be problematic when a small batch has to be used due to memory constraints, e.g., when handling high resolution images on a single GPU.

Thus, we recommend using PICO when the computational cost is less of a concern.

D SUPPLEMENTARY EXPERIMENTS D.1 LIPSCHITZ CONSTRAINT VIA GRADIENT PENALTY Gradient penalty has been widely used to impose the Lipschitz constraint on the discriminator arguably since Wasserstein GAN BID10 ).

This section explores whether the proposed repulsive loss can be applied with gradient penalty.

Several gradient penalty methods have been proposed for MMD-GAN.

penalized the gradient norm of witness function y) ] w.r.t.

the interpolated sample z = ux + (1 − u)y to one, where u ∼ U(0, 1) 9 .

More recently, proposed to impose the Lipschitz constraint on the mapping φ • D directly and derived the Scaled MMD (SMMD) as SM k (P, Q) = σ µ,k,λ M k (P, Q), where the scale σ µ,k,λ incorporates gradient and smooth penalties.

Using the Gaussian kernel and measure µ = P X leads to the discriminator loss: DISPLAYFORM0 DISPLAYFORM1 We apply the same formation of gradient penalty to the repulsive loss: DISPLAYFORM2 where the numerator L rep D − 1 ≤ 0 so that the discriminator will always attempt to minimize both L rep D and the Frobenius norm of gradients ∇D(x) w.r.t.

real samples.

Meanwhile, the generator is trained using the MMD loss L mmd G (Eq. 2).Experiment setup: The gradient-penalized repulsive loss L rep-gp D (Eq. S8, referred to as MMD-repgp) was evaluated on the CIFAR-10 dataset.

We found λ = 10 in too restrictive 9 Empirically, we found this gradient penalty did not work with the repulsive loss.

The reason may be the attractive loss L att D (Eq. 3) is symmetric in the sense that swapping P X and PG results in the same loss; while the repulsive loss is asymmetric and naturally results in varying gradient norms in data space.

and used λ = 0.1 instead.

Same as , the output dimension of discriminator was set to one.

Since we entrusted the Lipschitz constraint to the gradient penalty, spectral normalization was not used.

The rest experiment setup can be found in Section 5.1.

TAB3 shows that the proposed repulsive loss can be used with gradient penalty to achieve reasonable results on CIFAR-10 dataset.

For comparison, we cited the Inception score and FID for Scaled MMD-GAN (SMMDGAN) and Scaled MMD-GAN with spectral normalization (SN-SMMDGAN) from .

Note that SMMDGAN and SN-SMMDGAN used the same DCGAN architecture as MMD-rep-gp, but were trained for 150k generator updates and 750k discriminator updates, much more than that of MMD-rep-gp (100k for both G and D).

Thus, the repulsive loss significantly improved over the attractive MMD loss for discriminator.

In this section, we investigate the impact of the output dimension of discriminator on the performance of repulsive loss.

Experiment setup: We used a similar setup as Section 5.1 with the following adjustments.

The repulsive loss was tested on the CIFAR-10 dataset with a variety of discriminator output dimensions: d ∈ {1, 4, 16, 64, 256}. Spectral normalization was applied to discriminator with the proposed PICO method (see Appendix C) and the coefficients C K selected from {16, 32, 64, 128, 256}.Results: TAB4 shows that using more than one output neuron in the discriminator D significantly improved the performance of repulsive loss over the one-neuron case on CIFAR-10 dataset.

The reason may be that using insufficient output neurons makes it harder for the discriminator to learn an injective and discriminative representation of the data (see FIG8 ).

However, the performance gain diminished when more neurons were used, perhaps because it becomes easier for D to surpass the generator G and trap it around saddle solutions.

The computation cost also slightly increased due to more output neurons.

Generated samples on CelebA dataset are given in FIG7 and LSUN bedrooms in FIG8 .

Spectral normalization was applied to discriminator with two power iteration methods: PICO and PIM.

For PICO, five coefficients C K were tested: 16, 32, 64, 128, and 256.

A learning rate combination was considered diverged or poorly-performed if the FID score exceeded a threshold τ , which is 50, 80, 50, 90 for CIFAR-10, STL-10, CelebA and LSUN-bedroom respectively.

The box quartiles were plotted based on the cases with FID < τ while the number of diverged or poorly-performed cases (out of 16 learning rate combinations) was shown above each box if it is non-zero.

We introduced τ because the diverged cases often had arbitrarily large and non-meaningful FID scores.

DISPLAYFORM0

@highlight

Rearranging the terms in maximum mean discrepancy yields a much better loss function for the discriminator of generative adversarial nets