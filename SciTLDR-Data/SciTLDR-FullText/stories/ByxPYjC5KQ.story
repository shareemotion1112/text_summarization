Generative Adversarial Networks (GANs) are one of the most popular tools for learning complex high dimensional distributions.

However, generalization properties of GANs have not been well understood.

In this paper, we analyze the generalization of GANs in practical settings.

We show that discriminators trained on discrete datasets with the original GAN loss have poor generalization capability and do not approximate the theoretically optimal discriminator.

We propose a zero-centered gradient penalty for improving the generalization of the discriminator by pushing it toward the optimal discriminator.

The penalty guarantees the generalization and convergence of GANs.

Experiments on synthetic and large scale datasets verify our theoretical analysis.

GANs BID6 are one of the most popular tools for modeling high dimensional data.

The original GAN is, however, highly unstable and often suffers from mode collapse.

Much of recent researches has focused on improving the stability of GANs BID21 BID8 BID14 BID10 .

On the theoretical aspect, BID17 proved that gradient based training of the original GAN is locally stable.

BID8 further proved that GANs trained with Two Timescale Update Rule (TTUR) converge to local equilibria.

However, the generalization of GANs at local equilibria is not discussed in depth in these papers.

BID2 showed that the generator can win by remembering a polynomial number of training examples.

The result implies that a low capacity discriminator cannot detect the lack of diversity.

Therefore, it cannot teach the generator to approximate the target distribution.

In section 4, we discuss the generalization capability of high capacity discriminators.

We show that high capacity discriminators trained with the original GAN loss tends to overfit to the mislabeled samples in training dataset, guiding the generator toward collapsed equilibria (i.e. equilibria where the generator has mode collapse).

BID3 proposed to measure the generalization capability of GAN by estimating the number of modes in the model distribution using the birthday paradox.

Experiments on several datasets showed that the number of modes in the model distribution is several times greater than the number of training examples.

The author concluded that although GANs might not be able to learn distributions, they do exhibit some level of generalization.

Our analysis shows that poor generalization comes from the mismatch between discriminators trained on discrete finite datasets and the theoretically optimal discriminator.

We propose a zero-centered gradient penalty for improving the generalization capability of (high capacity) discriminators.

Our zero-centered gradient penalty pushes the discriminator toward the optimal one, making GAN to converge to equilibrium with good generalization capability.

Our contributions are as follow:1.

We show that discriminators trained with the original GAN loss have poor generalization capability.

Poor generalization in the discriminator prevents the generator from learning the target distribution.

TAB0 compares the key properties of our 0-GP with one centered GP (1-GP) BID7 and zero centered GP on real/fake samples only (0-GP-sample) BID13 .

p r the target distribution p g the model distribution p z the noise distribution d x the dimensionality of a data sample (real or fake) d z the dimensionality of a noise sample supp(p) the support of distribution p x ∼ p r a real sample z ∼ p z a noise vector drawn from the noise distribution p z y = G(z) a generated sample D r = {x 1 , ..., x n } the set of n real samples D (t) g = y (t) 1 , ..., y (t) m the set of m generated samples at step t DISPLAYFORM0 the training dataset at step t

Gradient penalties are widely used in GANs literature.

There are a plethora of works on using gradient penalty to improve the stability of GANs BID13 BID7 BID19 BID22 BID20 .

However, these works mostly focused on making the training of GANs stable and convergent.

Our work aims to improve the generalization capability of GANs via gradient regularization.

BID3 showed that the number of modes in the model distribution grows linearly with the size of the discriminator.

The result implies that higher capacity discriminators are needed for better approximation of the target distribution.

BID3 studied the tradeoff between generalization and discrimination in GANs.

The authors showed that generalization is guaranteed if the discriminator set is small enough.

In practice, rich discriminators are usually used for better discriminative power.

Our GP makes rich discriminators generalizable while remaining discriminative.

Although less mode collapse is not exactly the same as generalization, the ability to produce more diverse samples implies better generalization.

There are a large number of papers on preventing mode collapse in GANs.

BID21 ; BID23 introduced a number of empirical tricks to help stabilizing GANs.

showed the importance of divergences in GAN training, leading to the introduction of Wasserstein GAN .

The use of weak divergence is further explored by BID15 ; BID16 .

BID12 advocated the use of mixed-batches, mini-batches of real and fake data, GP Formula Improve generalization

Convergence guarantee to smooth out the loss surface.

The method exploits the distributional information in a mini-batch to prevent mode collapse.

VEEGAN BID24 uses an inverse of the generator to map the data to the prior distribution.

The mismatch between the inverse mapping and the prior is used to detect mode collapse.

If the generator can remember the entire training set, then the inverse mapping can be arbitrarily close the the prior distribution.

It suggests that VEEGAN might not be able to help GAN to generalize outside of the training dataset.

Our method helps GANs to discover unseen regions of the target distribution, significantly improve the diversity of generated samples.

DISPLAYFORM0

In the original GAN, the discriminator D maximizes the following objective BID6 showed that if the density functions p g and p r are known, then for a fixed generator G the optimal discriminator is DISPLAYFORM0 DISPLAYFORM1 In the beginning of the training, p g is very different from p r so we have p r (x) p g (x), for x ∈ D r and p g (y)p r (y), for y ∈ D g .

Therefore, in the beginning of the training D * (x) ≈ 1, for x ∈ D r and D * (y) ≈ 0, for y ∈ D g .

As the training progresses, the generator will bring p g closer to p r .

The game reaches the global equilibrium when p r = p g .

At the global equilibrium, DISPLAYFORM2 One important result of the original paper is that, if the discriminator is optimal at every step of the GAN algorithm, then p g converges to p r .In practice, density functions are not known and the optimal discriminator is approximated by optimizing the classification performance of a parametric discriminator DISPLAYFORM3 We call a discriminator trained on a discrete finite dataset an empirical discriminator.

The empirically optimal discriminator is denoted byD * .

BID2 DISPLAYFORM4 A discriminator D defines a divergence between two distributions.

The performance of a discriminator with good generalization capability on the training dataset should be similar to that on the entire data space.

In practice, generalization capability of D can be estimated by measuring the difference between its performance on the training dataset and a held-out dataset.

It has been observed that if the discriminator is too good at discriminating real and fake samples, the generator cannot learn effectively BID6 .

The phenomenon suggests thatD * does not well approximate D * , and does not guarantee the convergence of p g to p r .

In the following, we clarify the mismatch betweenD * and D * , and its implications.

g are disjoint with probability 1 even when p g and p r are exactly the same.

D * perfectly classifies the real and the fake datasets, andD DISPLAYFORM0 g .

The value ofD * on D (t) does not depend on the distance between the two distributions and does not reflect the learning progress.

The value ofD * on the training dataset approximates that of D * in the beginning of the learning process but not when the two distributions are close.

When trained using gradient descent on a discrete finite dataset with the loss in Eqn.

1, the discriminator D is pushed towardD * , not D * .

This behavior does not depend on the size of training set (see FIG0 , 1b), implying that the original GAN is not guaranteed to converge to the target distribution even when given enough data.

When the generator gets better, generated samples are more similar to samples from the target distribution.

However, regardless of their quality, generated samples are still labeled as fake in Eqn.

1.

The training dataset D is a bad dataset as it contains many mislabeled examples.

A discriminator trained on such dataset will overfit to the mislabeled examples and has poor generalization capability.

It will misclassify unseen samples and cannot teach the generator to generate these samples.

FIG0 and 1b demonstrate the problem on a synthetic dataset consisting of samples from two Gaussian distributions.

The discriminator in FIG0 overfits to the small dataset and does not generalize to new samples in FIG0 .

Although the discriminator in FIG0 was trained on a larger dataset which is sufficient to characterize the two distributions, it still overfits to the data and its value surface is very different from that of the theoretically optimal discriminator in FIG0 .An overfitted discriminator does not guide the model distribution toward target distribution but toward the real samples in the dataset.

This explains why the original GAN usually exhibits mode collapse behavior.

Finding the empirically optimal discriminator using gradient descent usually requires many iterations.

Heuristically, overfitting can be alleviated by limiting the number of discriminator updates per generator update.

BID6 recommended to update the discriminator once every generator update.

In the next subsection, we show that limiting the number of discriminator updates per generator update prevents the discriminator from overfitting.

* is costly to find and maintain.

We consider here a weaker notion of optimality which can be achieved in practical settings.

Definition 1 ( -optimal discriminator).

Given two disjoint datasets D r and D g , and a number > 0, a discriminator D is -optimal if DISPLAYFORM0 As observed in BID6 ,D * does not generate usable gradient for the generator.

Goodfellow et al. proposed the non-saturating loss for the generator to circumvent this vanishing gradient problem.

For an -optimal discriminator, if is relatively small, then the gradient of the discriminator w.r.t.

fake datapoints might not vanish and can be used to guide the model distribution toward the target distribution.

Proposition 2.

Given two disjoint datasets D r and D g , and a number > 0, an -optimal discriminator D exists and can be constructed as a one hidden layer MLP with O(d x (m + n)) parameters.

Proof.

See appendix B.Because deep networks are more powerful than shallow ones, the size of a deep -optimal discriminator can be much smaller than O(d x (m + n)).

From the formula, the size of a shallow -optimal discriminator for real world datasets ranges from a few to hundreds of millions parameters.

That is comparable to the size of discriminators used in practice. showed that even when the generator can generate realistic samples, a discriminator that can perfectly classify real and fake samples can be found easily using gradient descent.

The experiment verified that -optimal discriminator can be found using gradient descent in practical settings.

We observe that the norm of the gradient w.r.t.

the discriminator's parameters decreases as fakes samples approach real samples.

If the discriminator's learning rate is fixed, then the number of gradient descent steps that the discriminator has to take to reach -optimal state should increase.

Proposition 3.

Alternating gradient descent with the same learning rate for discriminator and generator, and fixed number of discriminator updates per generator update (Fixed-Alt-GD) cannot maintain the (empirical) optimality of the discriminator.

Fixed-Alt-GD decreases the discriminative power of the discriminator to improve its generalization capability.

The proof for linear case is given in appendix C.In GANs trained with Two Timescale Update Rule (TTUR) BID8 , the ratio between the learning rate of the discriminator and that of the generator goes to infinity as the iteration number goes to infinity.

Therefore, the discriminator can learn much faster than the generator and might be able to maintain its optimality throughout the learning process.

Let's consider a simplified scenario where the real and the fake datasets each contains a single datapoint: DISPLAYFORM0 .

Updating the generator according to the gradient from the discriminator will push y (t) toward x. The absolute value of directional derivative of DISPLAYFORM1 DISPLAYFORM2 The directional derivate of the -optimal discriminator explodes as the fake datapoint approaches the real datapoint.

Directional derivative exploding implies gradient exploding at datapoints on the line segment connecting x and y (t) .

If in the next iteration, the generator produces a sample in a region where the gradient explodes, then the gradient w.r.t.

the generator's parameters explodes.

Let's consider the following line integral DISPLAYFORM3 where C is the line segment from y (t) to x. As the model distribution gets closer to the target distribution, the length of C should be non increasing.

Therefore, maximizing D(x) − D(y (t) ), or the discriminative power of D, leads to the maximization of the directional derivative of D in the direction ds.

The original GAN loss makes D to maximize its discriminative power, encouraging gradient exploding to occur.

Gradient exploding happens in the discriminator trained with TTUR in FIG2 .

Because TTUR can help the discriminator to maintain its optimality, gradient exploding happens and persists throughout the training process.

Without TTUR, the discriminator cannot maintain its optimality so gradient exploding can happen sometimes during the training but does not persist ( FIG2 .

Because of the saturated regions in the sigmoid function used in neural network based discriminators, the gradient w.r.t.

datapoints in the training set could vanishes.

However, gradient exploding must happen at some datapoints on the path between a pair of samples, where the sigmoid function does not saturate.

In FIG0 , gradient exploding happens near the decision boundary.

In practice, D r and D g contain many datapoints and the generator is updated using the average of gradients of the discriminator w.r.t.

fake datapoints in the mini-batch.

If a fake datapoint y 0 is very close to a real datapoint x 0 , the gradient (∇D) y0 might explode.

When the average gradient is computed over the mini-batch, (∇D) y0 outweighs other gradients.

The generator updated with this average gradient will move many fake datapoints in the direction of (∇D) y0 , toward x 0 , making mode collapse visible.

Although the theoretically optimal discriminator D * is generalizable, the original GAN loss does not push empirical discriminators toward D * .

We aim to improve the generalization capability of empirical discriminators by pushing them toward D * .

For any input v ∈ supp(p r ) ∪ supp(p g ), the value of D * (v) goes to 1 2 and the gradient (∇D) v goes to 0 as p g approaches p r .

Consider again the line integral in Eqn.

4.

As D * (x) and D * (y) approach 1 2 for all x ∈ supp(p r ) and y ∈ supp(p g ), we have DISPLAYFORM0 for all pairs of x and y and all paths C from y to x.

That means, the discriminative power of D * must decrease as the two distributions become more similar.

To push an empirical discriminator D toward D * , we force D to satisfy two requirements: DISPLAYFORM1

The first requirement can be implemented by sampling some datapoints v ∈ supp(p r ) ∪ supp(p g ) and force (∇D) v to be 0.

The second requirement can be implemented by sampling pairs of real and fake datapoints (x, y) and force D(x) − D(y) to be 0.

The two requirements can be added to the discriminator's objective as followŝ DISPLAYFORM0 where L is the objective in Eqn.

1.

However, as discussed in section 4.2.2, an -optimal discriminator can have zero gradient on the training dataset and have gradient exploding outside of the training dataset.

The gradient norm could go to infinity even when D(x) − D(y) is small.

Regulating the difference between D(x) and D(y) is not an efficient way to prevent gradient exploding.

We want to prevent gradient exploding on every path in supp(p r ) ∪ supp(p g ).

Because (∇D * ) v → 0 for all v ∈ supp(p r ) ∪ supp(p g ) as p g approach p r , we could push the gradient w.r.t.

every datapoint on every path C ∈ supp(p r ) ∪ supp(p g ) toward 0.

We note that, if (∇D) v → 0, ∀ v ∈ C then C (∇D) v · ds → 0.

Therefore, the two requirements can be enforced by a single zero-centered gradient penalty of the form DISPLAYFORM1 The remaining problem is how to find the path C from a fake to a real sample which lies inside supp(p r ) ∪ supp(p g ).

Because we do not have access to the full supports of p r and p g , and the supports of two distributions could be disjoint in the beginning of the training process, finding a path which lies completely inside the support is infeasible.

In the current implementation, we approximate C with the straight line connecting a pair of samples, although there is no guarantee that all datapoints on that straight line are in supp(p r ) ∪ supp(p g ).That results in the following objective DISPLAYFORM2 wherex = αx + (1 − α)y, x ∼ p r , y ∼ p g , and α ∼ U(0, 1) 1 .

We describe a more sophisticated way of finding a better path in appendix F.The larger λ is, the stronger (∇D)x is pushed toward 0.

If λ is 0, then the discriminator will only focus on maximizing its discriminative power.

If λ approaches infinity, then the discriminator has maximum generalization capability and no discriminative power.

λ controls the tradeoff between discrimination and generalization in the discriminator.

BID13 proposed to force the gradient w.r.t.

datapoints in the real and/or fake dataset(s) to be 0 to make the training of GANs convergent.

In section 4, we showed that for discrete training dataset, an empirically optimal discriminatorD * always exists and could be found by gradient descent.

Although (∇D * ) v = 0, ∀ v ∈ D,D * does not satisfy the requirement in Eqn.

5 and have gradient exploding when some fake datapoints approach a real datapoint.

The discriminators in FIG0 , 2c and 2d have vanishingly small gradients on datapoints in the training dataset and very large gradients outside.

They have poor generalization capability and cannot teach the generator to generate unseen real datapoints.

Therefore, zero-centered gradient penalty on samples from p r and p g only cannot help improving the generalization of the discriminator.

Non-zero centered GPs do not push an empirical discriminator toward D * because the gradient does not converge to 0.

A commonly used non-zero centered GP is the one-centered GP (1-GP) BID7 which has the following form DISPLAYFORM0 wherex = αx + (1 − α)y, x ∼ p r , y ∼ p g , and α ∼ U(0, 1).

Although the initial goal of 1-GP was to enforce Lipschitz constraint on the discriminator 2 , BID5 found that 1-GP prevents gradient exploding, making the original GAN more stable.

1-GP forces the norm of gradients w.r.t.

datapoints on the line segment connecting x and y to be 1.

If all gradients on the line segment have norm 1, then the line integral in Eqn.

4 could be as large as x − y .

Because the distance between random samples grows with the dimensionality, in high dimensional space x − y is greater than 1 with high probability.

The discriminator could maximize the value of the line integral without violating the Lipschitz constraint.

The discriminator trained with 1-GP, therefore, can overfit to the training data and have poor generalization capability.

BID13 showed that zero-centered GP on real and/or fake samples (0-GP-sample) makes GANs convergent.

The penalty is based on the convergence analysis for the Dirac GAN, an 1-dimensional linear GAN which learns the Dirac distribution.

The intuition is that when p g is the same as p r , the gradient of the discriminator w.r.t.

the fake datapoints (which are also real datapoints) should be 0 so that generator will not move away when being updated using this gradient.

If the gradient from the discriminator is not 0, then the generator will oscillate around the equilibrium.

Our GP forces the gradient w.r.t.

all datapoints on the line segment between a pair of samples (including the two endpoints) to be 0.

As a result, our GP also prevents the generator from oscillating.

Therefore, our GP has the same convergence guarantee as the 0-GP-sample.

Discriminators trained with the original GAN loss tends to focus on the region of the where fake samples are close to real samples, ignoring other regions.

The phenomenon can be seen in FIG2 , 2c, 2d, 2h and 2i.

Gradients in the region where fake samples are concentrated are large while gradients in other regions, including regions where real samples are located, are very small.

The generator cannot discover and generate real datapoints in regions where the gradient vanishes.

When trained with the objective in Eqn.

6, the discriminator will have to balance between maximizing L and minimizing the GP.

For finite λ, the GP term will not be exactly 0.

Let DISPLAYFORM0 .

Among discriminators with the same value of γ, gradient descent will find the discriminator that maximizes L. As discussed in section 4.2.2, maximizing L leads to the maximization of norms of gradients on the path from y to x. The discriminator should maximize the value η = Ex[ (∇D)x ].

If γ is fixed then η is maximized when ∇Dx(i) = ∇Dx(j) , ∀ i, j (Cauchy-Schwarz inequality).

Therefore, our zero-centered GP encourages the gradients at different regions of the real data space to have the same norm.

The capacity of D is distributed more equally between regions of the real data space, effectively reduce mode collapse.

The effect can be seen in FIG2 1-GP encourages | ∇Dx(i) − 1| = | ∇Dx(j) − 1|, ∀ i, j. That allows gradient norms to be smaller than 1 in some regions and larger than 1 in some other regions.

The problem can be seen in FIG2 .

The code is made available at https://github.com/htt210/ GeneralizationAndStabilityInGANs.

To test the effectiveness of gradient penalties in preventing overfitting, we designed a dataset with real and fake samples coming from two Gaussian distributions and trained a MLP based discriminator on that dataset.

The result is shown in FIG0 .

As predicted in section 5.3, 0-GP-sample does not help to improve generalization.

1-GP helps to improve generalization.

The value surface in FIG0 is smoother than that in FIG0 .

However, as discussed in section 5.3, 1-GP cannot help much in higher dimensional space where the pair-wise distances are large.

The discriminator trained with our 0-GP has the best generalization capability, with a value surface which is the most similar to that of the theoretically optimal one.

We increased the number of discriminator updates per generator update to 5 to see the effect of GPs in preventing overfitting.

On the MNIST dataset, GAN without GP and with other GPs cannot learn anything after 10,000 iterations.

GAN with our 0-GP can still learn normally and start produce recognizable digits after only 1,000 iterations.

The result confirms that our GP is effective in preventing overfitting in the discriminator.

OF GANS SYNTHETIC DATAWe tested different gradient penalties on a number of synthetic datasets to compare their effectiveness.

The first dataset is a mixture of 8 Gaussians.

The dataset is scaled up by a factor of 10 to simulate the situation in high dimensional space where random samples are far from each other.

The result is shown in FIG2 .

GANs with other gradient penalties all fail to learn the distribution and exhibit mode collapse problem to different extents.

GAN with our 0-GP (GAN-0-GP) can successfully learn the distribution.

Furthermore, GAN-0-GP can generate datapoints on the circle, demonstrating good generalization capability.

The original GAN collapses to some disconnected modes and cannot perform smooth interpolation between modes: small change in the input result in large, unpredictable change in the output.

GAN with zero-centered GP on real/fake samples only also exhibits the same "mode jumping" behavior.

The behavior suggests that these GANs tend to remember the training dataset and have poor generalization capability.

Fig. 9 in appendix D demonstrates the problem on MNIST dataset.

We observe that GAN-0-GP behaves similar to Wasserstein GAN as it first learns the overall structure of the distribution and then focuses on the modes.

An evolution sequence of GAN-0-GP is shown in FIG4 in appendix D. Results on other synthetic datasets are shown in appendix D.

The result on MNIST dataset is shown in Fig. 3 .

After 1,000 iterations, all other GANs exhibit mode collapse or cannot learn anything.

GAN-0-GP is robust to changes in hyper parameters such BID23 on ImageNet of GAN-0-GP, GAN-0-GP-sample, and WGAN-GP.

The code for this experiment is adapted from BID13 .

We used λ = 10 for all GANs as recommended by Mescheder et al. The critic in WGAN-GP was updated 5 times per generator update.

To improve convergence, we used TTUR with learning rates of 0.0001 and 0.0003 for the generator and discriminator, respectively.

as learning rate and optimizers.

When Adam is initialized with large β 1 , e.g. 0.9, GANs with other GPs cannot learn anything after many iterations.

More samples are given in appendix D. DISPLAYFORM0 We observe that higher value of λ improves the diversity of generated samples.

For λ = 50, we observe some similar looking samples in the generated data.

This is consistent with our conjecture that larger λ leads to better generalization.

When trained on ImangeNet BID4 ), GAN-0-GP can produce high quality samples from all 1,000 classes.

We compared our method with GAN with 0-GP-sample and WGAN-GP.

GAN-0-GP-sample is able to produce samples of state of the art quality without using progressive growing trick BID10 .

The result in Fig. 4 shows that our method consistently outperforms GAN-0-GP-sample.

GAN-0-GP and GAN-0-GP-sample outperform WGAN-GP by a large margin.

Image samples are given in appendix D.

In this paper, we clarify the reason behind the poor generalization capability of GAN.

We show that the original GAN loss does not guide the discriminator and the generator toward a generalizable equilibrium.

We propose a zero-centered gradient penalty which pushes empirical discriminators toward the optimal discriminator with good generalization capability.

Our gradient penalty provides better generalization and convergence guarantee than other gradient penalties.

Experiments on diverse datasets verify that our method significantly improves the generalization and stability of GANs.

Pengchuan Zhang, Qiang Liu, Dengyong Zhou, Tao Xu, and Xiaodong He.

On the discriminationgeneralization tradeoff in GANs.

In International Conference on Learning Representations, 2018.A PROOF FOR PROPOSITION 1For continuous random variable V , P(V = v) = 0 for any v. The probability of finding a noise vector z such that G(z) is exactly equal to a real datapoint x ∈ D r via random sampling is 0.

Therefore, the probability of a real datapoint x i being in the fake dataset D g is 0.

Similarly, the probability of any fake datapoint being in the real dataset is 0.

DISPLAYFORM0 Furthermore, due to the curse of dimensionality, the probability of sampling a datapoint which is close to another datapoint in high dimensional space also decrease exponentially.

The distances between datapoints are larger in higher dimensional space.

That suggests that it is easier to separate D r and D (t) g in higher dimensional space.

To make the construction process simpler, let's assume that samples are normalized: DISPLAYFORM0 Let's use the following new notations for real and fake samples: DISPLAYFORM1 We construct the -optimal discriminator D as a MLP with 1 hidden layer.

Let W 1 ∈ R (m+n)×dx and W 2 ∈ R m+n be the weight matrices of D. The total number of parameters in DISPLAYFORM2 We set the value of W 1 as DISPLAYFORM3 and W 2 as DISPLAYFORM4 Given an input v ∈ D, the output is computed as: DISPLAYFORM5 where σ is the softmax function.

Let a = W 1 v, we have DISPLAYFORM6 As k → ∞, σ(W 1 v i ) becomes a one-hot vector with the i-th element being 1, all other elements being 0.

Thus, for large enough k, for any v j ∈ D, the output of the network is DISPLAYFORM7 C FIXED-ALT-GD CANNOT MAINTAIN THE OPTIMALITY OF -DISCRIMINATORS Let's consider the case where the real and the fake dataset each contain a single datapoint D r = {x}, D DISPLAYFORM8 , and the discriminator and the generator are linear: DISPLAYFORM9 and the objective is also linear (Wasserstein GAN's objective): DISPLAYFORM10 The same learning rate α is used for D and G.At step t, the discriminator is -optimal DISPLAYFORM11 The gradients w.r.t.

θ D and θ G are DISPLAYFORM12 DISPLAYFORM13 If the learning rate α is small enough, x − y (t) should decrease as t increases.

As the empirical fake distribution converges to the empirical real distribution, x − y (t) → 0.

The norm of gradient w.r.t.

θ D , therefore, decreases as t increases and vanishes when the two empirical distributions are the same.

From Eqn.

10, we see that, in order to maintain D's -optimality when x − y (t) decreases, θ D has to increase.

From Eqn.

10 and 12, we see that the gradient w.r.t.

θ G grows as the two empirical distributions are more similar.

As x − y (t) → 0, DISPLAYFORM14 Because the same learning rate α is used for both G and D, G will learn much faster than D. Furthermore, because x − y (t) decreases as t increases, the difference DISPLAYFORM15 increases with t. The number of gradient steps that D has to take to reach the next -optimal state increases, and goes to infinity as x − y (t) → 0.

Therefore, gradient descent with fixed number of updates to θ D cannot maintain the optimality of D.The derivation for the objective in Eqn.

1 is similar. : GANs trained with different gradient penalty on swissroll dataset.

Although GAN-1-GP is able to learn the distribution, the gradient field has bad pattern.

GAN-1-GP is more sensitive to change in hyper parameters and optimizers.

GAN-1-GP fails to learn the scaled up version of the distribution.

The entire ImageNet dataset with all 1000 classes was used in the experiment.

Because of our hardware limits, we used images of size 64 × 64.

We used the code from BID13 , available at https://github.com/LMescheder/GAN stability, for our experiment.

Generator and Discriminator are ResNets, each contains 5 residual blocks.

All GANs in our experiment have the same architectures and hyper parameters.

The configuration for WGAN-GP5 is as follows.

@highlight

We propose a zero-centered gradient penalty for improving generalization and stability of GANs