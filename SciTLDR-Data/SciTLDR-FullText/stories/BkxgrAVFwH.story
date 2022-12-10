In the field of Generative Adversarial Networks (GANs), how to design a stable training strategy remains an open problem.

Wasserstein GANs have largely promoted the stability over the original GANs by introducing Wasserstein distance, but still remain unstable and are prone to a variety of failure modes.

In this paper, we present a general framework named Wasserstein-Bounded GAN (WBGAN), which improves a large family of WGAN-based approaches by simply adding an upper-bound constraint to the Wasserstein term.

Furthermore, we show that WBGAN can reasonably measure the difference of distributions which almost have no intersection.

Experiments demonstrate that WBGAN can stabilize as well as accelerate convergence in the training processes of a series of WGAN-based variants.

Over the past few years, Generative Adversarial Networks (GANs) have shown impressive results in many generative tasks.

They are inspired by the game theory, that two models compete with each other: a generator which seeks to produce samples from the same distribution as the data, and a discriminator whose job is to distinguish between real and generated data.

Both models are forced stronger simultaneously during the training process.

GANs are capable of producing plausible synthetic data across a wide diversity of data modalities, including natural images (Karras et al., 2017; Brock et al., 2018; Lucic et al., 2019) , natural language (Press et al., 2017; Lin et al., 2017; Rajeswar et al., 2017) , music Mogren, 2016; Dong et al., 2017; Dong & Yang, 2018) , etc.

Despite their success, it is often difficult to train a GAN model in a fast and stable way, and researchers are facing issues like vanishing gradients, training instability, mode collapse, etc.

This has led to a proliferation of works that focus on improving the quality of GANs by stabilizing the training procedure (Radford et al., 2015; Salimans et al., 2016; Zhao et al., 2016; Nowozin et al., 2016; Qi, 2017; Deshpande et al., 2018) .

In particular, introduced a variant of GANs based on the Wasserstein distance, and releases the problem of gradient disappearance to some extent.

However, WGANs limit the weight within a range to enforce the continuity of Lipschitz, which can easily cause over-simplified critic functions (Gulrajani et al., 2017) .

To solve this issue, Gulrajani et al. (2017) proposed a gradient penalty method termed WGAN-GP, which replaces the weight clipping in WGANs with a gradient penalty term.

As such, WGAN-GP provides a more stable training procedure and succeeds in a variety of generating tasks.

Based on WGAN-GP, more works (Wei et al., 2018; Petzka et al., 2017; Wu et al., 2018; Mescheder et al., 2018; Thanh-Tung et al., 2019; Kodali et al., 2017; adopt different forms of gradient penalty terms to further improve training stability.

However, it is often observed that such gradient penalty strategy sometimes generate samples with unsatisfying quality, or even do not always converge to the equilibrium point (Mescheder et al., 2018) .

In this paper, we propose a general framework named Wasserstein-Bounded GAN (WBGAN), which improve the stability of WGAN training by bounding the Wasserstein term.

The highlight is that the instability of WGANs also resides in the dramatic changes of the estimated Wasserstein distance during the initial iterations.

Many previous works just focused on improving the gradient penalty term for stable training, while they ignored the bottleneck of the Wasserstein term.

The proposed training strategy is able to adaptively enforce the Wasserstein term within a certain value, so as to balance the Wasserstein loss and gradient penalty loss dynamically and make the training process more stable.

WBGANs are generalized, which can be instantiated using different kinds of bound estimations, and incorporated into any variant of WGANs to improve the training stability and accelerate the convergence.

Specifically, with Sinkhorn distance (Cuturi, 2013; Genevay et al., 2017) for bound estimation, we test three representative variants of WGANs (WGAN-GP (Gulrajani et al., 2017) , WGANdiv (Wu et al., 2018) , and WGAN-GPReal (Mescheder et al., 2018) ) on the CelebA dataset (Liu et al., 2015) .

As shown in Fig. 1

Wasserstein GANs (WGANs).

WGANs were primarily motivated by unstable training caused by the gradient vanishing problem of the original GANs (Goodfellow et al., 2014) .

They proposed to use 1-Wasserstein distance W 1 (P r , P g ) to measure the difference between P r and P g , the real and generated distributions, given that W 1 (P r , P g ) is continuous everywhere and differentiable almost everywhere under mild assumptions.

The objective of WGAN is formulated using the KantorovichRubinstein duality (Villani, 2008) :

where L 1 is the function space of all D satisfying the 1-Lipschitz constraint D L ≤ 1.

D is a critic and G is the generator, both of which are parameterized by a neural network.

Under an optimal critic, minimizing the objective with respect to G is to minimize W 1 (P r , P g ).

To enforce the 1-Lipschitz constraint on the critic, WGAN used a weight clipping on the critic to constrain the weights within a compact range, [−c, c] , which guarantees the set of critic functions is a subset of the k-Lipschitz functions for some k. With weight clipping, the critic tends to learn over-simplified functions (Gulrajani et al., 2017) , which may lead to unsatisfying results.

Gulrajani et al. (2017) ; Wei et al. (2018); Petzka et al. (2017) ; Wu et al. (2018) proposed different forms of gradient penalty as a regularization term, so that a generalized loss function with respect to the critic can be written as:

where

] stands for the Wasserstein term, and GP for the gradient penalty term.

L D is actually posing a tradeoff between these two objectives.

Wasserstein Distance between Empirical Distributions.

In practice, we approximate W 1 (P r , P g ) using W 1 P r ,P g , whereP r andP g denote the empirical version of P r and P g with N samples,

Here, y i is randomly sampled from the real image dataset, and δ yi is the Dirac delta function at location y i .

Computing W 1 P r ,P g is a typical problem named discrete optimal transport.

We denote B as the set of probabilistic couplings between two empirical distributions defined as:

where 1 N is a N -dimensional all-one vector.

Then we have W 1 (P r ,P g ) = min γ∈B Γ, C F , where ·, · F is the Frobenius dot-product and C is the cost matrix, with each element C i,j = c(G(z i ), y j ) denoting the cost to move a probability mass from G(z i ) to y j .

The optimal coupling is the solution of this minimization problem: Γ 0 = arg min Γ∈B Γ, C F .

The Sinkhorn Algorithm.

Despite Wasserstein distance has appealing theoretical properties in measuring the difference between distributions, its computational costs for linear programming are often high in particular when the problem size becomes large.

To alleviate this burden, Sinkhorn distance (Cuturi, 2013) was proposed to approximate Wasserstein distance:

where U α (P r ,P g ) is a subset of B defined in Eq. 3:

where H(·) is the entropy defined as H(Γ) = − N i,j=1 Γ i,j log Γ i,j and H(P r ) = − N n=1p n logp n wherep n is the probability of the n-th sample.

Compared to Wasserstein distance, Sinkhorn distance restricts the search space of joint probabilities to those with sufficient smoothness.

To compute Sinkhorn distance, a Lagrange multiplier was used:

can be computed with a much cheaper cost than the original Wasserstein distance using matrix scaling algorithms.

For λ > 0, the solution Γ λ is unique and has the form Γ λ = diag(u)K diag(v), where K is the element-wise exponential of −λC. u and v are two non-negative vectors uniquely defined up to a multiplicative factor (Cuturi, 2013

.

W is often referred to as the Wasserstein term, which is unbounded during the training process.

In a wide range of WGAN's variants such as WGAN-GP (Gulrajani et al., 2017) , the critic defined by L D is to maximize the Wasserstein term W while satisfying the gradient penalty GP.

However, in practice, we find that W often rises rapidly to a tremendous value which is far from rational during the initial training procedure.

A possible reason may lie in that the critic function does not satisfy the Lipschitz constraint during the initial training stage.

As shown in Fig. 2 , this leads to dramatic instability in optimization and finally results in unsatisfying performance in image generation.

Our idea is thus straightforward, i.e., setting an upper-bound for W .

The modified critic loss function is written as: Our formulation brings a benefit to the numerical stability of the Wasserstein term.

In practice, it remains comparable to the other term, λ GP · GP, so that both W and GP can be optimized in a 'mild' manner, i.e., without any one of them dominating or being ignored during training.

Note that the Ď W term cannot be chosen arbitrarily.

Setting it too small, Ď W will limit the capacity of the critic function, resulting in a poor generation.

Setting it too large, there will be no effect of bounding the W term.

The proposed bounded strategy is a general framework.

We name it general in two folds: First, WBGAN can be applied to almost all gradient penalty based WGANs, such as WGAN-GP (Gulrajani et al., 2017) , WGAN-GPReal (Mescheder et al., 2018) , etc.

Moreover, there are different ways to estimate the value of Ď W .

For example, the linear programming was applied successfully to some existing WGANs like WGAN-TS .

In what follows, we present an example which uses Sinkhorn distance to estimate Ď W , while we believe other ways of estimation are also possible.

In this section, we give an instantiation, Sikhorn distance (Cuturi, 2013) , to effectively compute the bounded term Ď W .

The motivation of using Sinkhorn distance lies in that in theory, the Wasserstein term of WGAN will eventually converge to the 1-Wasserstein distance between the real distribution P r and the generated distribution P g Gulrajani et al., 2017) .

Therefore, we can use the 1-Wasserstein distance between the empirical distributions,P r andP g , as the upper-bound Ď W .

Since the computation of Wasserstein distance involves a large linear programming which Algorithm 1 WBGAN with Sinkhorn distance Require: learning rate α, batch size M , the number of iterations of the critic per generator iteration N critic , weight of gradient penalty λ GP , weight of Sinkhorn distance λ s , initial parameters θ and φ 0 , other hyper-parameters; 1: while φ t has not converged do 2:

Sample a batch {z (m) } M m=1 ∼ P z of prior samples;

5:

7:

end for 10:

Sample a batch {z (m) } M m=1 ∼ P z of prior samples;

11:

12:

φ t+1 ← Adam(L φt , φ t , α, β 1 , β 2 ); 14: end while Ensure: trained parameters θ and φ T (converged).

suffers heavy computational costs, we replace it by Sinkhorn distance instead -the Sinkhorn distance betweenP r andP g can be computed using Sinkhorn's matrix scaling algorithm (Cuturi, 2013) , which is orders of magnitude faster than the linear programming solvers.

Mathematically, consider a generator function G φ (z) that produces samples by transforming noise input z drawn from a simple distribution P z , e.g., Gaussian distribution.

D θ stands for a critic function parameterized by θ.

The objective of the critic is:

where d λ (P r ,P g ) is the Sinkhorn distance defined in Eq. 6.

On the other hand, given a fixed critic function D θ , considering that Sinkhorn distance allows gradient back-propagation (Genevay et al., 2017) , we can find the optimal generator G φ by solving:

where λ s is a balancing hyper-parameter, which we set λ s = 0.5 in this paper.

In Algorithm 1, we summarize the flowchart of training WBGAN with Sinkhorn distance.

) be a separable metric space.

P(X) denotes the set of Borel probability measures.

P p (X) denotes the set of all µ ∈ P(X) such that X d(x, y) p dµ(x) < +∞ for some y ∈ X. We can suppose real data distribution P r , generated data distribution P g and their empirical distributionP r andP g all in P p (X).

Proposition 1.

Let P r and P g be real data distribution and generated data distribution.

Suppose that P r andP g are empirical measures of P r and P g .

Then we have 0

Proof.

Please refer to Appendix A.

Proposition 1 tells us that as E[W 1 (P r ,P g )] → 0, W 1 (P r , P g ) is forced to 0.

Cuturi (2013) has pointed out that if λ is chosen large enough, d λ (P r ,P g ) coincides with W 1 (P r ,P g ).

So, it is reasonable to use d λ (P r ,P g ) to constrain the Wasserstein term.

Most GANs measure the distance between distributions based on probability divergence.

We will prove that the Eq. 8 is indeed a valid divergence.

First, we have the following definition.

Definition 1.

Given probability measures p and q, D is a functional of p and q. If D satisfies the following properties:

then we say D is a probability divergence between p and q.

Remark 1.

The following W (P r , P g ) satisfies the Definition 1 and is therefore a probability divergence.

where L 1 is the 1-Lipschitz constraint.

Please see the proof and detailed discussion in Su (2018) .

This is the objective of critic used by WGAN .

Remark 2.

Equation 8 satisfies the Definition 1 and is a probability divergence.

Proof.

The proof is given in Appendix B.

Remark 3.

Consider two distributions P r (x) = δ(x − α), P g (x) = δ(x − β) that have no intersection (α = β).

δ is the Dirac delta function.

In such an extreme case, Eq. 8 can still be optimized by gradient descent.

Proof.

The proof is in Appendix C Remark 2 tells us that Eq. 8 is a valid divergence.

Since the real data distribution is supported by lowdimensional manifolds, the supports of generated distribution and real data distribution are unlikely to have a non-negligible intersection.

Remark 3 shows that compared to the standard GAN (Goodfellow et al., 2014) , WBGAN can continuously measure the difference between two distributions, even if there is almost no intersection between the distributions.

To verify that WBGAN is a generalized approach, we select three variants of WGAN, namely, WGAN-GP (Gulrajani et al., 2017) , WGAN-div (Wu et al., 2018) and WGAN-GPReal (gradient penalty on real data only) (Mescheder et al., 2018) as our baselines.

By adding bound constraints to these WGAN variants, we obtain the counterparts WBGAN-GP, WBGAN-div, and WBGANGPReal, respectively.

Two different network architectures are used, i.e., DCGAN (Radford et al., 2015) and BigGAN (Brock et al., 2018) .

For DCGAN, we directly output the activation before the sigmoid layer.

BigGAN is a conditional GAN (Mirza & Osindero, 2014) architecture, in which class conditioning is passed to generator by supplying it with class-conditional gains and biases in the batch normalization layer (Ioffe & Szegedy, 2015; de Vries et al., 2017; .

In addition, the discriminator is conditioned by using the cosine similarity between its features and a set of learned class embedding.

We use the spectral norm in BigGAN, but for the sake of simplicity, we do not use the self-attention module (Wang et al., 2017; .

Other hyper-parameters and the network architecture of BigGAN simply follow the original paper.

We choose the Fréchet Inception Distance (FID) (Heusel et al., 2017) for quantitative evaluation, which has been proven to be more consistent with individual assessment in evaluating the fidelity and variation of the generated image samples.

We first investigate mid-resolution image generation on the CelebA dataset (Liu et al., 2015) , a large-scale face image dataset with more than 200K face images.

During training, we crop 108 × 108 face from the original images and then resize them to 64 × 64.

FID Stability.

We first use DCGAN to build our generator and discriminator.

Training curves are shown in Fig. 2 , and quantitative results are summarized in Table 1 .

Each approach is executed for 5 times and the average is reported.

All FID curves are obtained from generators directly without using the moving average strategy (Karras et al., 2017; Mescheder et al., 2018; Brock et al., 2018; Yazıcıet al., 2018) to avoid over-smoothing the FID curves, such that we can diagnose the underlying oscillating properties of different methods during training.

One can see that WBGAN-based counterparts improve the stability during training, and achieve superior performance over the WGAN-based baselines.

We emphasize that the converged FID values reported by WBGAN-div and WBGANGPReal are lower than those reported by WGAN-div and WGAN-GPReal.

In particular, WGAN-div suffers several FID fluctuation unexpectedly, and WGAN-GPReal has not ever achieved FID convergence during the entire training process.

Regarding WGAN-GP, although the final FID is slightly better than that of WBGAN-GP (6.76 vs. 7.32), we observe a much slower convergence rate in Fig. 2(a) .

For the generated face images by different approaches, please refer to We also investigate a stronger backbone by replacing the network with BigGAN, a conditional GAN architecture that uses spectral normalization on both generator and discriminator.

We set the number of labels to be 1 since the CelebA dataset only contains face images.

Training curves are shown in Fig. 3 and quantitative results are summarized in Table 1 .

Among three WGAN-based methods, only WGAN-GP achieves convergence, but its convergence speed and the FID value are inferior to those reported by WBGAN-GP.

In opposite, both WGAN-div and WGAN-GPReal fails to converge while the counterparts equipped with WB-GAN perform well.

For the generated face images by different approaches, please refer to Fig. 11 and Fig. 12 in Appendix F for details.

Epoch Wasserstein Loss and Generator Loss Stability.

Next, we evaluate the stability of WBGAN in terms of the Wasserstein term and generator loss.

In Fig. 4 , we evaluate the impact on WGAN-GP (DCGAN on CelebA).

One can see that, after the bound is applied, the Wasserstein term W is stablized especially during the start of training.

Due to space limit, more results using BigGAN on CelebA are provided in Appendix E. In addition, we compute a new term named the generator loss, which is defined as Fig. 6 shows the curves of this statistics during the starting iterations.

Compared to WGAN-based approaches, WBGAN-based approaches produce more stable G loss terms, which verifies that the training process of GAN becomes more stable.

Ablation Study.

Before continuing to high-resolution experiments, we conduct an ablation study to investigate the contribution made by different components of WBGAN.

The backbone network is DCGAN, and the dataset is CelebA. We compare four configurations, i.e., WGAN-GP, with the original loss term used in WGAN-GP; WGAN-GP+D-bound, which adds a bound (Sinkhorn distance) to the Wasserstein term of the critic D of WGAN-GP; WGAN-GP+G-Sinkhorn, which adds Sinkhorn distance to the loss function of the generator G in WGAN-GP; and WGAN-GP+D-bound+G-Sinkhorn, which is equivalent to the final WBGAN-GP, with Sinkhorn distance added to both critic D and generator G. Fig. 5 plots the FID curves of all four settings.

One can see that, although the FID curves of WGAN-GP and WGAN-GP+G-Sinkhorn descend quickly in the first 10 epochs, they begin to fluctuate between 20 to 40 epochs.

On the other hand, when WGAN-GP is combined with D-bound, FID is able to descend smoothly (without fluctuation), showing that it is the bounded constraint that stablizes the training process.

Finally, by integrating both D-bound and G-Sinkhorn into WGAN-GP, the FID curve descends not only smoothly but also fast, which is what we desire in real-world applications.

In this section, we evaluate our approach on higher-resolution (128 × 128) images.

We use the CelebA-HQ dataset (Karras et al., 2017) , and use BigGAN (Brock et al., 2018) as the backbone.

As the target become larger (128 × 128), the number of images we can feed into a single batch becomes smaller (64).

Since we are using an empirical way of estimating Sinkhorn distance, it becomes less accurate in the scenario of small batch size and large image size.

In other words, it is no longer the best choice to use Sinkhorn distance to estimate the upper-bound Ď W .

Returning to our generalized formulation, Eq. 7, we note that other forms of bound to constrain the critic.

Here we consider a very simple bound, which is also based on empirical study.

Note that the baseline methods, though not converging very well, can finally arrive at a stablized W value.

Heuristically, we use this constant value (there is no need to be accurate) as the bound, which is 10 for WGAN-GP, 5 for WGAN-div and 3 for WGAN-GPReal, respectively.

In Appendix D, we provide the curves of the Wasserstein term for these baselines, which lead to our estimation.

FID curves and quantitative results using these constant bounds are shown in Fig. 7 and Table 2 , respectively.

We find that WBGAN-GP produces a similar convergence rate with WGAN-GP, WBGAN-div is slightly better than WGAN-div, and WBGAN-GPReal outperforms WGAN-GPReal and produces the best results.

For the generated face images by different approaches, please refer to Fig. 13 and Fig. 14 in Appendix F for details.

Discussions.

From the above experiments, we can see that Sinkhorn distance is just one way of upper-bound estimation.

In case that it becomes less accurate, we can freely replace it with other types of estimation.

Besides the constant bound used above, there also exist other examples, such as the two-step computation of the exact Wasserstein distance .

However, it is still a challenge to estimate the Wasserstein distance between high-resolution (1024 × 1024) image distributions efficiently.

Nevertheless, the most important deliveries of our work are that a bounded Wasserstein term can bring benefits on training stability, and that we can use it to a wide range of frameworks based on WGAN.

This paper introduced a general framework called WBGANs, which can be applied to a variety of WGAN variants to stabilize the training process and improve the performance.

We clarify that WBGANs can stabilize the Wasserstein term at the beginning of the iterations, which is beneficial for smoother convergence of WGAN-based methods.

We present an instantiated bound estimation method via Sinkhorn distance and give a theoretical analysis on it.

It remains an open topic on how to set a better bound for higher resolution image generation tasks.

Proof.

Suppose µ, υ 1 , υ 2 ∈ P p (X), t 1 , t 2 ≥ 0, t 1 + t 2 = 1, then there exist γ 1 (x, y) and γ 2 (x, y) with marginals (µ, υ 1 ) and (µ, υ 2 ) satisfying:

Let υ = t 1 υ 1 + t 2 υ 2 , γ(x, y) = t 1 γ 1 (x, y) + t 2 γ 2 (x, y), then γ(x, y) has marginals (µ, υ).

We can derive:

This conclusion can be extended to a general form:

where

are the independent empirical measures drawn fromP g .

From Eq. 15, we can get

According to the strong law of large numbers, we can derive that with probability 1, 1 n n i=1P gi → P g as n → ∞ (assuming P g has finite first moments).

Since W 1 is continuous in P p (X), we can derive that with probability 1, W 1 (P r , 1 n n i=1P gi ) →

W 1 (P r , P g ) as n → ∞.

By the law of large numbers again, with probability 1, 1 n n i=1 W 1 (P r ,P gi ) → E[W 1 (P r ,P g )] as n → ∞. Thus we can deduce that:

Similarly, supposeP ri (1 ≤ i ≤ n) are the independent empirical measures drawn fromP r .

Since the symmetry of Wasserstein distance, we can deduce that:

Therefore, combining Eq. 17 and Eq. 18, we can get

B PROOF OF REMARK 2

where d λ (P r ,P g ) ≥ 0 is the Sinkhorn distance defined in Eq. 6.

Next, if P r = P g , then we have L θ (P r , P g ) = 0.

So we only need to show L θ (P r , P g ) > 0 if

Applying this into Eq. 8 leads to L θ (P r , P g ) = max

Since P r = P g , we know that d λ (P r ,P g ) > 0.

Therefore, we have L θ (P r , P g ) > 0 while P r = P g .

We finish the proof.

Proof.

Let P r (x) = δ(x − α), P g (x) = δ(x − β) and α = β, then we have

We know that Wasserstein distance W (P r , P g ) = max D θ ∈L1 D θ (α) − D θ (β).

Since P r , P g are Dirac distributions, then we have W (P r , P g ) = d λ (P r ,P g ).

Combining this into Eq. 22 leads to L θ (P r , P g ) = d

λ (P r ,P g ).

Considering that Sinkhorn distance d λ (P r ,P g ) (Cuturi, 2013)

@highlight

Propose an improved framework for WGANs and demonstrate its better performance in theory and practice.