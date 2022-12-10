Probabilistic modelling is a principled framework to perform model aggregation, which has been a primary mechanism to combat mode collapse in the context of Generative Adversarial Networks (GAN).

In this paper, we propose a novel probabilistic framework for GANs, ProbGAN, which iteratively learns a distribution over generators with a carefully crafted prior.

Learning is efficiently triggered by a tailored stochastic gradient Hamiltonian Monte Carlo with a novel gradient approximation to perform Bayesian inference.

Our theoretical analysis further reveals that our treatment is the first probabilistic framework that yields an equilibrium where generator distributions are faithful to the data distribution.

Empirical evidence on synthetic high-dimensional multi-modal data and image databases (CIFAR-10, STL-10, and ImageNet) demonstrates the superiority of our method over both start-of-the-art multi-generator GANs and other probabilistic treatment for GANs.

Generative Adversarial Networks (GAN) BID9 is notoriously hard to train and suffers from mode collapse.

There has been a series of works attempting to address these issues.

One noticeable thread focuses on objective design, which improves the original JensenShannon divergence with more stable pseudo-metrics such as f -divergence (Nowozin et al., 2016) , χ 2 -divergence (Mao et al., 2017) , and Wasserstein distance BID0 .

However, such treatment is inherently limited when a single generator does not include enough model capacity to capture the granularity in data distribution in practice.

Clearly, such a generator can hardly produce accurate samples regardless of the choice of objectives.

An alternative remedy is to learn multiple generators instead of a single one.

This type of methods (Hoang et al., 2018; Tolstikhin et al., 2017; Wang et al., 2016b ) is motivated by a straightforward intuition that multiple generators can better model multi-modal distributions since each generator only needs to capture a subset of the modes.

To entail model aggregation, probabilistic modelling is a natural and principled framework to articulate the aggregation process.

Recently, Saatci & Wilson (2017) propose Bayesian GAN, a probabilistic framework for GAN under Bayesian inference.

It shows that modelling the distribution of generator helps alleviate mode collapse and motivates the interpretability of the learned generators.

This probabilistic framework is built upon Bayesian models for generator and discriminator, whose maximum likelihood estimation can be realized as a metaphor of typical GAN objectives.

While empirical study on semi-supervised image classification tasks shows the effectiveness of Bayesian GAN, a critical theoretical question on this framework remains unanswered: Does it really converge to the generator distribution that produces the real data distribution?

Indeed, our theoretical analysis and experimental results on a simple toy dataset reveal that the current Bayesian GAN falls short of convergence guarantee.

With this observation, we follow the prior work to exploit probabilistic modelling as a principled way to realize model aggregation, but approach this problem from a theoretical perspective.

We analyze the developed treatment, including the choice of priors, approximate inference, as well as its convergence property, and simultaneously propose a new probabilistic framework with the desirable convergence guarantee and consequently superior empirical performance.

Our main contributions are:• We theoretically establish, to our best knowledge, the first probabilistic treatment of GANs such that any generator distribution faithful to the data distribution is an equilibrium.• We prove the previous Bayesian method (Saatci & Wilson, 2017) for any minimax GAN objective induces incompatibility of its defined conditional distributions.• We propose two special Monte Carlo inference algorithms for our probabilistic model which efficiently approximate the gradient of a non-differentiable criterion.• Empirical studies on synthetic high-dimensional multi-modal data and benchmark image datasets, CIFAR-10, STL-10, and ImageNet, demonstrate the superiority of the proposed framework over the state-of-the-art GAN methods.

Generative Adversarial Networks is a powerful class of methods to learn a generative model for any complex target data distribution.

There is a game between a generator and a discriminator.

Both of them adapt their strategies to maximize their own objective function involving the other: DISPLAYFORM0 Eqn.

1 gives a general mathematical form where p data is real data distribution and p gen (·; θ g ) are generated data distribution with generator parameter θ g .

The objective functions φ 1 , φ 2 , φ 3 (termed as GAN objective in this paper) are elaborately chosen such that at the equilibrium, the generator generates the target data distribution.

TAB0 summarizes several widely used GAN objectives, including the original min-max version, non-saturating version of original GAN (Goodfellow (2016)), LSGAN (Mao et al. (2017) ), and WGAN BID0 ).

As reported in TAB0 , some GAN objectives, satisfying φ 3 (·) = −φ 2 (·), actually represent a min-max game, i.e. DISPLAYFORM1 Training GAN with multiple generators is considered in several recent works to mitigate the mode collapse problem.

In the spirit of boosting algorithm, Wang et al. (2016b) propose to progressively train new generator using a subset of training data that are not well captured by previous generators, while Tolstikhin et al. (2017) further propose a more robust mechanism to reweight samples in the training set for a new generator.

From the perspective of game theory, MIX-GAN BID2 extends the game between a single generator and discriminator to the multiple-player setting.

Other works resort to third-party classifiers to help multiple generators and discriminators achieve better equilibrium, such as MAD-GAN BID7 and the recent state-of-art method, MGAN (Hoang et al., 2018) .Bayesian GAN proposed by Saatci & Wilson (2017) adopts a different approach which models generator and discriminator distributions by defining the conditional posteriors (Eqn. 8).

The likelihood model is specially designed such that maximizing it exactly corresponds to optimizing GAN objectives.

The authors argue that compare to point mass ML estimation, learning the generator distribution which is multi-modal itself offers better ability to fit a multi-modal data distribution.

To facilitate discussion, we categorize GAN frameworks into the following taxonomy: optimizationbased methods and probabilistic methods.

Optimization-based methods set up an explicit mini-max game between the generator and discriminator, where an ideal equilibrium typically characterize a generator faithful to data distribution.

In probabilistic methods, generators and discriminators evolve as particles of underlying distributions, where an equilibrium is searched from a stochastic exploration in the distribution space (of the generators and discriminators).

We first summarize the notations.

Second, we elaborate ProbGAN, our probabilistic modelling for GAN, and introduce its Bayesian interpretation by developing constituent prior and likelihood formulations.

Finally, we develop inference algorithms for ProbGAN.

A detailed discussion of the motivation of our modelling and the comparison with Bayesian GAN is included in Section 4.

p data (x) over a sample space X is the target data distribution we want to learn.

Our generator and discriminator are parameterized by θ g ∈ Θ g and θ d ∈ Θ d .

A generator with parameter θ g defines a mapping from a random noise vector z ∼ p z to a random vector G(z; θ g ).

The induced probability density of G(z; θ g ) is denoted as p gen (x; θ g ).

A discriminator is a function that maps data to a real-valued score, i.e. D(x; θ d ) : DISPLAYFORM0 to denote the distribution over generators and discriminators respectively.

The total data distribution generated by generator following the density q g (θ g ) is naturally a mixture of data distribution given by every single generator, DISPLAYFORM1 Our goal is to find a generator distribution q * g (θ g ) such that the total generated data distribution matches our target, i.e. DISPLAYFORM2 ) denote objective functions of generator and discriminator as introduced in Eqn.

1.

The common choices 1 are listed in TAB0 .

With a slight abuse of the notation, we extend the notation DISPLAYFORM3 represents discriminator objective given a virtual generator that generates data with density p gen (·).

Like Bayesian GAN, ProbGAN learns distributions of the generator and the discriminator.

During training, the target data distribution is given (by samples) and treated as an fixed environment.

While for the generator/discriminator, they observe each other and adapt their own parameters based on the observation.

To facilitate the comparison to Bayesian GAN, we state ProbGAN in a Bayesian formulation.

Every generator/discriminator distribution update can be viewed as the following posterior inference process.

Posterior.

ProbGAN updates generator/discriminator distributions based on their distributions in previous time step and the target data distribution as shown in Eqn.

2.

We we denote q (t) g and q DISPLAYFORM0 To understand it is a posterior modelling, we further interpret the terms in Eqn.

2 as likelihood term and prior term seperately.

Likelihood.

We call the exponential terms in Eqn.

2 as likelihood terms.

DISPLAYFORM1 where DISPLAYFORM2 is the mixed data distribution under the current generator distribution q DISPLAYFORM3 is the averaged discriminating score function under DISPLAYFORM4 These likelihoods indicate a preference for generators and discriminators, given current distributions of generator and discriminator.

More specifically, likelihoods in Eqn.

3 encode the information that distributionally reflect the objective of generators J g and discriminators J d .

Such quantities evaluate the fitness between the generator and the discriminator.

We emphasize, although sharing the same spirit of reflecting the GAN objectives in likelihood, there is a crucial difference between our likelihood model and that of Bayesian GAN.

We will revisit it in the later theory section.

Prior.

Unlike Bayesian GAN using normal distributions for both generator and discriminator, ProbGAN has less standard priors.

As Eqn.

2 suggests, we set different priors for the two players.

For the generator, we use the generator distribution in the previous time step as a prior.

The intuition is following.

When the generated data distribution is increasingly close to the real data distribution, there will be less information for discriminator to distinguish between them; consequently, the discriminator tends to assign equal scores to all data samples, resulting in equal likelihoods for all generators.

At that stage, a good strategy is to keep the generator distribution the same as the previous time step, since it already generates the desired data distribution perfectly.

Hence, we use the generator distribution in the previous time step as a prior for the next Such dynamically evolving prior for generator turns out to be crucial.

In Section 4.2, we show the Bayesian GAN suffers from bad convergence due to its fixed and weakly informative prior.

In contrast, we set a uniform improper prior on the discriminator to pursuit unrestricted adaptability to evolving generators.

DISPLAYFORM5 are used to approximate the generator distribution q DISPLAYFORM6 DISPLAYFORM7 Empowered by the adapted SGHMC (Algorithm 1 in the appendix), we are able to sample from q (t+1) g and q DISPLAYFORM8 based on gradients in Eqn.

4 and Eqn.

5.

The gradients come from two sides: the GAN objective J g , J d and the prior q (t) g .

Getting GAN objective's gradient is easy while computing the prior's gradient, ∇ θg log q (t) g (θ g ), is actually non-trivial since we have no exact analytic form of q (t) g (θ g ).

To address this challenge, we propose the following two methods to approximate ∇ θg log q (t) g (θ g ), leading to two practical inference algorithms.

Gaussian Mixture Approximation (GMA).

Although the analytic form of the distribution q DISPLAYFORM9 which enables us to directly approximate the distribution as a Mixture of Gaussian in the left side of Eqn.

6, where σ is a hyper-parameter and C is the normalization constant.

Then we derive the prior gradient approximation as shown in the right side of Eqn.

6.

DISPLAYFORM10 Partial Summation Approximation (PSA).

From Eqn.

5, actually we can make an interesting observation that the prior gradient can be recursively unfolded as a summation over all historical GAN objective gradients, shown as: DISPLAYFORM11 Figure 1: An example of data distributions produced by converged models in the toy experiment on categorical distribution.

We examined four possible combinations of likelihoods and priors.

L our , L bgan stand for the likelihoods of our ProbGAN model and BGAN model.

P our and P bgan stand for the priors.

Only our model (Figure 1(d) ) learns the target data distribution (Figure 1(e) ).Therefore if we store all historical discriminator samples {θ DISPLAYFORM12 , the prior gradient can be computed accurately via simple summation.

Practically, computing gradients with all discriminator samples costs huge amount of storage and computational time, which is unaffordable.

Hence we propose to maintain a subset of discriminators by subsampling the whole sequence of discriminators.

In this section, we first present the good convergence property of ProbGAN.

Second, we theoretically analyze the distribution evolution of Bayesian GAN (which will be referred to as BGAN in the rest of the paper) and compare BGAN with ProbGAN.

All proofs are included in the appendix (Section A).

We say a generator distribution is ideal if the generator following this distribution produces the target data distribution.

Theorem 1 shows that any ideal generator distribution is an equilibrium of the dynamics defined in Eqn.

2.

Although its mathematical proof involves more elaboration, the idea behind is quite simple.

When the generator distribution is ideal, the discriminator is not able to distinguish the synthetic data from real.

Thus the averaged discriminator function will degenerate to a constant function.

Afterwards, the generator distribution will remain unchanged since the discriminator essentially puts no preference over generators.

Here, we note that the discriminator is only involved in the likelihood.

The prior still needs to be carefully designed so that the model can converge to an equilibrium where the generator is ideal.

Simply choosing a weakly informative prior as Bayesian GAN did will not give Theorem 1.

Theorem 1.

Assume the GAN objective and the discriminator space are symmetry.

For any ideal generator distribution q * DISPLAYFORM0 d is an equilibrium of the dynamic defined in Eqn.

2.

This section presents analyses of the BGAN algorithm, where we find a theoretical issue in its convergence and highlight the importance of our renovation of the prior and likelihood.

Corollary 1 states our derivation of posterior modelling in BGAN, where DISPLAYFORM0 are the predefined priors.

In practice, BGAN use a fixed Gaussian prior.

Corollary 1.

The Bayesian GAN algorithm actually performs distribution dynamics in Eqn.

8.

DISPLAYFORM1 Difference in Likelihood.

The subtle adjustment of our likelihood term lies in the order of taking expectation.

As shown in Eqn.

3, our choice of likelihood yields a concrete physical meaning.

Our discriminator likelihood explicitly evaluates the discriminator ability of distinguishing real data distribution and total data distribution generated by all generators.

Hence, our approach matches the target data distribution with mixed data distribution produced by generators, while BGAN may not.

The choice of prior also plays an important role in convergence.

A weakly/noninformative prior, as adopted in BGAN, prevents the generators from convergence even if they already produce the data distribution faithfully.

The phenomenon arises from the fact that the information provided by discriminator vanishes when the generators are ideal and the resulting generator posterior will degenerate to the prior.

To remedy this issue, our solution is to take generator distribution at previous time step as the prior.

Then whenever the discriminator degenerates to a constant, the generator distribution will stay unchanged because the prior is itself.

An Analytical Case Study.

We demonstrate the superior convergence property of our model on a categorical distribution, where analytic posterior computation of various choices of likelihood and prior are feasible; we compute the exact equilibrium of GAN models under all the four combinations of the priors and likelihoods(ProbGAN's choices v.s. BGAN's choices).

The experiment details are in the appendix (section D).

Figure 1 is an example of the data distributions generated by each model after it converges.

Among all the combinations, our model is the only formulation that yields proper convergence, which validates our theoretical analysis.

Compatibility Issue.

We further show BGAN's choice of likelihood and prior leads to theoretical issues.

Specifically, BGAN is not suitable for any minimax-style GAN objective due to the incompatibility of its conditional posteriors.

This problem may limit the usage of BGAN since many widely used GAN objective is in min-max fashion, such as the original GAN and WGAN.Consider a simple case where we use only one Monte Carlo sample for the distributions q (t) g and q DISPLAYFORM0 Then the distribution evolution in Eqn.

8 will degenerate to a Gibbs sampling process.

DISPLAYFORM1 However, our theoretical analysis shows that such a presumed joint distribution does not exist when DISPLAYFORM2 .

Specifically, Lemma 1 shows the existence of a joint distribution satisfying the conditionals in Eqn.

9 requires the GAN objective to be decomposable, i.e. DISPLAYFORM3 Apparently, no valid GAN objective is decomposable.

Therefore, conditionals in Eqn.

9 are actually incompatible.

Sampling with incompatible conditional distribution is problematic and leads to unpredictable behavior BID1 .

Lemma 1.

Consider a joint distribution p(x, y) of variable X and Y .

Its conditional distributions can be represented in the forms of p(x|y) ∝ exp{L(x, y)}q x (x) and p(y|x) ∝ exp{−L(x, y)}q y (y) only if X and Y are independent, i.e., p(x, y) = p(x)p(y) and L(x, y) is decomposable, i.e., DISPLAYFORM4

In this section, we evaluate our model with two inference algorithms proposed in Section 3.3 (denoted as ProbGAN-GMA and ProbGAN-PSA).

We compare with three baselines: 1) GAN (or DCGAN 2 ): naively trained multiple generators in the vanilla GAN framework; 2) MGAN: Mixture GAN (Hoang et al., 2018) which is the start-of-art method to train GAN with multiple generators; 3) BGAN: DISPLAYFORM0 For each model, we conduct thorough experiments with the four different GAN objectives introduced in TAB0 , which are referred to as GAN-MM, GAN-NS, WGAN and LSGAN here.

For a fair comparison, each model has the same number of generators with the same architecture.

Discriminator architectures are also the same except for that of MGAN which has an additional branch of the classifier.

To facilitate reproducibility, we report implementation and experiment details in Section B of the appendix.

Dataset.

Consider learning a data distribution in a high dimensional space X = R D , which is a uniform mixture of n modes.

Each mode lies on a d-dimensional sub-space of X .

We call this d-dimensional sub-space as mode-space of the i-th mode.

Specifically, the data of the i-th mode is generated by the following process, DISPLAYFORM0 In our experiment, n, D, and d are set to 10, 100, and 2.

Hyper-parameters for A and b are set to be σ A = σ b = 5.

Each model train ten generators.

Metric.

We define projection distance p for generated data sample x as the minimum of Euclidean distance from x to any of mode-spaces i.e. DISPLAYFORM1 Then we set a threshold 3 η to test the belonging of x, i.e. the data samples whose Euclidean distance to the mode-space is below η are considered as belonging to that mode.

The trained models are evaluated by the samples {x k } K k=1 ∼ p model it generates.

We define hit set H i {x k | i (x k ) < η} to indicate the samples belong to each mode.

We further define projected hit set, DISPLAYFORM2 } by projecting data in each hit set back to the canonical low dimensional space.

Now we introduce three evaluation metrics: hit ratio, hit distance, and cover error.

Hit ratio H r n i=1 |H i | /K is the percentage of generated data belonging to any of the modes of real data.

DISPLAYFORM3 |H i | is the averaged projection distance over all data in the hit set.

Lastly, cover error C e evaluates how well the generated data covers each mode.

Essentially it computes the KL-divergence between the estimated distribution of samples in PH i and the uniform distribution over [−1, 1] d .

Formally, it is defined as the averaged KL-divergence on n modes i.e. C e DISPLAYFORM4 The intuition is that if data generated is close to the ground truth distribution, they should be uniformly distributed in the square area of each mode.

Optimization-Based v.s. Probabilistic.

The left part of TAB2 summarizes the results in terms of hit ratio and hit distance.

Probabilistic methods including our algorithms and BGAN always achieve a hit ratio of 1, which means every data point generated from these models is very close to one mode of the target distribution.

On the other hand, optimization based methods, both GAN and MGAN, consistently have a significantly larger hit error, and sometimes may even generate data samples that do not belong to any mode.

Moreover, the data distribution generated by the optimization-based methods fits the target uniform distribution much worse than its probabilistic counterparts, which is quantitatively reflected by the cover error showed in the right side of TAB2 and visually demonstrated by the projected hit sets in FIG1 .

According to the visualization, data generated by GAN or MGAN tend to be under dispersed and hardly cover the whole square region of the true mode, while data generated by probabilistic methods align much better with the ground truth distribution.

We attribute this superiority to stronger exploration power in the generator space coming from the randomness in probabilistic methods.

Bayesian GAN v.s. ProbGAN.

The incompatibility issue of BGAN with minimax-style GAN objectives theoretically derived in Section 4.2 is empirically verified in our experiments.

As visualized in FIG1 , with the GAN-MM objective, BGAN is trapped in a local equilibrium and fails in capturing one mode of the true data.

Besides, as shown in TAB2 , BGAN with the WGAN objective achieves much poorer coverage than with other GAN objectives, while our model is much more robust to the choice of GAN objectives (consistently lower cover errors).

A qualitative comparison is made in Figure 9 (in the appendix) which shows the data distribution generated by BGAN trained with WGAN objective tends to shrink.

More visual illustrations under different GAN objectives are placed in Section E of the appendix.

Datasets.

We evaluate our method on 3 widely-adopted datasets: CIFAR-10 (Krizhevsky et al., 2010), STL-10 (Coates et al., 2011) and ImageNet BID6 ).

CIFAR-10 has 50k training and 10k test 32x32 RGB images from 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

STL-10, containing 100k 96x96 RGB images, is a more diverse dataset than CIFAR-10.

ImageNet has over 1.2 million images from 1,000 classes and presents the most diverse dataset.

For a fair comparison with baselines, we use the same settings as MGAN.

We resize the STL-10 and ImageNet images down to 48x48 and 32x32 respectively.

Evaluation Protocols.

We employ two common image generation metrics: Inception Score (Salimans et al. FORMULA0 ) and Fréchet Inception Distance BID10 FORMULA0 ).

More implementation details such as hyper-parameters of network layers are reported in the appendix (Section B).Quantitative Results.

TAB3 summarize the Inception scores and FIDs on the three benchmark datasets for our model and baselines.

For each model, we denote the epoch delivering highest 'IS−0.1FID' as the best epoch and report the scores of the model at this epoch.

We design this 'IS−0.1FID' principle to neutralize the discrepancy between Inception score and FID.

Because Inception score and FID may not be consistent with each other, it is possible that the Inception score improves but the FID gets worse when we evaluate the model at a different checkpoint.

Hence, we need a rule to pick one point of the Pareto frontier of the model performance.

Overall, our proposed ProbGAN outperforms the baselines on all three benchmark image datasets.

Furthermore, according to results on CIFAR-10, our model achieves the best performance under all GAN objectives.

Generally, probabilistic methods achieve better scores than optimization based methods, which indicates that injecting stochasticity into GAN training helps generate more multi-modal images.

Note that the performance gap increases as the dataset gets more diverse.

Specifically, when using GAN-NS objective, our model improves FID by 4.00 on CIFAR-10 TAB3 in comparison to MGAN.

While the FID improvements are 4.82 and 18.06 on STL-10 and ImageNet TAB5 , respectively.

Besides, we note that Bayesian GAN has a significant performance drop when accompanied by min-max style GAN objectives, which provides another empirical evidence for our theory analysis in Section 4.2.

By contrast, ProbGAN fits any GAN objectives and constantly performs well.

Qualitative Results.

FIG2 displays the samples randomly generated by the baselines and our ProbGAN.

Each row in the figure contains samples of one learned generator.

In FIG2 and FIG12 (in the appendix), all the three baselines noticeably suffer from mode collapse.

Indeed, almost in every training trial, one or two generators of the baseline models degenerate during the training.

Hoang et al. (2018) already notice that mode collapse in one of the generators could happen after a long training procedure (around 250 epochs).

Our experiment shows Bayesian GAN also have this issue.

However, ProbGAN is robust to mode collapse.

Visual results for the entire ablation study on CIFAR-10 are included in Section F of the appendix.

Each row in these two figures contains generated images from one generator.

The results show that our method is robust to 'single generator mode collapse' in both STL-10 and ImageNet.

We also include cherry-picked results on STL-10 to exhibit the capability of our model to generate visually appealing images with complex details while improving the robustness against mode collapse.

In this paper, we propose ProbGAN, a novel probabilistic modelling framework for GAN.

From the perspective of Bayesian Modelling, it contributes a novel likelihood function establishing a connection to existing GAN models and a novel prior stabilizing the inference process.

We also design scalable and asymptotically correct inference algorithms for ProbGAN.

In the future work, we plan to extend the proposed framework to non-parametric Bayesian modelling and investigate more theoretical properties of GANs in the probabilistic modelling context.

Developing Bayesian generalization for deep learning models is not a recent idea and happens in many fields other than generative models such as adversarial training (Ye & Zhu, 2018) and Bayesian neural networks (Wang et al., 2016a) .

By this work, we emphasize the importance of going beyond the intuition and understanding the theoretical behavior of the Bayesian model.

We hope that our work helps inspire continued exploration into Bayesian deep learning (Wang & Yeung, 2016 ) from a more rigorous perspective.

A OMITTED PROOFS Theorem 1.

This theorem is general and holds when the GAN objective and the discriminator space have symmetry.

The symmetry of GAN objective means its functions φ 1 and φ 2 satisfy that ∃c ∈ R, ∀x ∈ R, φ 1 (x) ≡ φ 2 (c − x).

While the symmetry of discriminator space DISPLAYFORM0 Note that the symmetry condition are very weak, first it holds for all the common choices of GAN objectives such as those listed in TAB0 .

Second, it holds for neural network which is the most common parameterization for discriminator in practice.

DISPLAYFORM1 DISPLAYFORM2 Eqn.

11 and Eqn.

12 prove that q * DISPLAYFORM3 Thus the generator distribution will not change based on the dynamics in Eqn.

2 since q * DISPLAYFORM4 ).

Proof.

DISPLAYFORM0 log q DISPLAYFORM1 Algorithm 1 from the original paper (Saatci & Wilson, 2017) implies that Eqn.

14 where θ Be definition p(θ g |z DISPLAYFORM2 .

Hence, the total summation is a Monte Carlo approximation of the expectation in the right side of Eqn.

15.

The same derivation can be done to q DISPLAYFORM3 .

Together, we get Corollary 1.

Proof.

Remark on Inception score and FID.

BID3 point out that Inception score is sensitive to the inception model used and the number of data splits in the computation.

This is also true for Fréchet Inception Distance (FID).

We find that the FID computed by a PyTorch Inception model 6 is much lower than the FID given by a Tensorflow model 7 .

DISPLAYFORM0 In our experiments, to facilitate a fair comparison with prior work, we compute Inception score and FID using the Tensorflow Inception model.

We adopt the official Tensorflow implementation for FID to compute both Inception score and FID.

We will release our evaluation code soon.

Model architecture: In our experiments, each model is trained with 10 generators.

As for discriminator, DCGAN and MGAN have one discriminator while probabilistic models (BGAN and ours) have 4 discriminators (i.e. 4 Monte Carlo samples from discriminator distribution).The neural network structures are the same as MGAN.

As reported in TAB5 ,5,6 in Section C.2 of the original MGAN paper.

Briefly, the structures are the following.

Generator architecture has four (or five) deconvolution layers (kernel size 4, stride 2) with the following input, hidden feature-maps, output size: 100x1x1 → 512x4x4 → 256x8x8 → 128x16x16 → 3x32x32.

Every deconvolution layer is followed by batch-normalization layer and Relu activation except for the last deconvolution layer who is followed by Tanh activation.

Discriminator architecture has four (or five) convolution layers (kernel size 5, stride 2) with the following input, hidden feature-maps, output size: 3x32x32 → 128x16x16 → 256x8x8 → 512x4x4 → 1x1x1.

Batch-normalization is applied to each layer except the last one.

Activations are leaky-ReLU.Training hyperparameters: All models are optimized by Adam(Kingma & Ba, 2014 ) with a learning rate of 2 × 10 4 .

For probabilistic methods, the SGHMC noise factor is set as 3 × 10 2 .

Following the configuration in MGAN, the batch size of generators and discriminators are 120 and 64.

Note that, since probabilistic model has 10 generator Monte Carlo samples and 4 discriminators, indeed batch size for every generator and discriminator is 12 and 16 respectively.

Model Architecture:

Each generator or discriminator is a three layer perceptron.

For the generator, the dimensions of input, hidden layer and output are 10, 1000, and 100 respectively.

For the discriminator, the dimensions of input, hidden, output layers are 100, 1000, and 1.

All activation functions are leaky ReLU (Maas et al., 2013) . , learning rate η, SGHMC noise factor α, number of updates in SGHMC procedure L, number of updating iterations T .

DISPLAYFORM0 g,m ← θ g,m end for end for Algorithm 1: Our Adapted SGHMC Inference Algorithm Setup.

In this toy example, we consider the case where X , Θ g , and Θ d are all finite sets, specifically DISPLAYFORM1 DISPLAYFORM2 The target data distribution is a categorical distribution Cat(λ 1:N ) where λ i = p data (x i ) is the probability of generating data x i .

Generator G i generates data following the categorical distribution p data (x; θ Further, the probability distributions of generator and discriminator are categorical distributions q g (θ g ) = Cat(β 1:Ng ) and DISPLAYFORM3 In practice, we set N = 10, N g = 20, N d = 100.

In each experiment trial, target data distribution λ 1:N and data distributions for each generator α For the discriminators, their function values are randomly generated from a uniform distribution, i.e. TAB7 , we compare four models with different pairs of likelihood and prior.

There are two choices of likelihood, expectation of objective value and objective value of expectation.

Mathematically, expectation of objective value likelihood has the following formula:

While the objective value of expectation likelihood is different as follows: DISPLAYFORM0 Thus model A indeed is the design of Bayesian GAN, while model D is our proposed model.

Model B and C are introduced to conduct the ablation study.

Metric.

We employ l 1 distance for evaluation which can be directly computed on categorical distributions as follows.

DISPLAYFORM1 Evaluation.

In the categorical distribution settings, all the likelihood, prior and posterior computing can be done analytically.

For each model, we update the generator and discriminator distributions iteratively and monitor the distance between the target data distribution and the data distribution generated by the model.

We experiment with two different choices of GAN objectives, GAN-MM and GAN-NS.

Note GAN-MM is a minimax-style objective.

Result.

FIG8 shows how l 1 distance changes as the number of updating iterations increases.

As we can see, model A and model B are easily and quickly trapped into bad local minima, indicating the convergence issue caused by non-informative prior.

Interestingly, model C presents bifurcate results when accompanied by different GAN objectives.

Its abnormal behavior in the setting of using GAN-MM objective indicates that the expectation of objective value likelihood used in Bayesian GAN does not fit the minimax-style GAN objectives.

Finally, our ProbGAN converges the fastest (towards the global optima) and is robust to all GAN objectives.

In this section, we shows projected hit sets for all models under different GAN objectives.

Three probabilistic models performs perfectly in this case, while both the two optimization-based methods miss one mode of the true distribution.

This experiment illustrates that although MGAN employs an additional classifier to force the data generated by different generators to be disjoint, it still suffers from mode collapsing problem.

This is because in MGAN, generators may still generate disjoint data samples in the same mode and fail in capturing other modes.

ProbGAN-GMA + GAN-NS ProbGAN-PSA + GAN-NS Figure 9 : Visualization of the projected hit sets of different models trained with the GAN-NS objective.

All the models succeed in fitting each mode of true distribution with one of their generator.

Specifically, three probabilistic models generate data almost perfectly covering the ground-truth 'squares' while the optimization-based methods have difficulty covering the whole 'squares' and tend to yield under-dispersed data distributions.

Note that since the GAN-NS objective is not in a min-max style, the success of BGAN is expected.

In this section, we shows images generated by all models trained on CIFAR-10 under different GAN objectives.

@highlight

A novel probabilistic treatment for GAN with theoretical guarantee.

@highlight

This paper proposes a Bayesian GAN that has theoretical guarantees of convergence to the real distribution and put likelihoods over the generator and discriminator with logarithms proportional to the traditional GAN objective functions.