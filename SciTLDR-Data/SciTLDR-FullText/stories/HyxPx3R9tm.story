Adversarial learning methods have been proposed for a wide range of applications, but the training of adversarial models can be notoriously unstable.

Effectively balancing the performance of the generator and discriminator is critical, since a discriminator that achieves very high accuracy will produce relatively uninformative gradients.

In this work, we propose a simple and general technique to constrain information flow in the discriminator by means of an information bottleneck.

By enforcing a constraint on the mutual information between the observations and the discriminator's internal representation, we can effectively modulate the discriminator's accuracy and maintain useful and informative gradients.

We demonstrate that our proposed variational discriminator bottleneck (VDB) leads to significant improvements across three distinct application areas for adversarial learning algorithms.

Our primary evaluation studies the applicability of the VDB to imitation learning of dynamic continuous control skills, such as running.

We show that our method can learn such skills directly from raw video demonstrations, substantially outperforming prior adversarial imitation learning methods.

The VDB can also be combined with adversarial inverse reinforcement learning to learn parsimonious reward functions that can be transferred and re-optimized in new settings.

Finally, we demonstrate that VDB can train GANs more effectively for image generation, improving upon a number of prior stabilization methods.

Adversarial learning methods provide a promising approach to modeling distributions over highdimensional data with complex internal correlation structures.

These methods generally use a discriminator to supervise the training of a generator in order to produce samples that are indistinguishable from the data.

A particular instantiation is generative adversarial networks, which can be used for high-fidelity generation of images BID21 and other highdimensional data BID45 BID46 BID9 .

Adversarial methods can also be used to learn reward functions in the framework of inverse reinforcement learning BID10 BID12 , or to directly imitate demonstrations BID19 .

However, they suffer from major optimization challenges, one of which is balancing the performance of the generator and discriminator.

A discriminator that achieves very high accuracy can produce relatively uninformative gradients, but a weak discriminator can also hamper the generator's ability to learn.

These challenges have led to widespread interest in a variety of stabilization methods for adversarial learning algorithms BID24 BID4 .In this work, we propose a simple regularization technique for adversarial learning, which constrains the information flow from the inputs to the discriminator using a variational approximation to the information bottleneck.

By enforcing a constraint on the mutual information between the input observations and the discriminator's internal representation, we can encourage the discriminator to learn a representation that has heavy overlap between the data and the generator's distribution, thereby effectively modulating the discriminator's accuracy and maintaining useful and informative gradients for the generator.

Our approach to stabilizing adversarial learning can be viewed as an adaptive variant of instance noise BID39 .

However, we show that the adaptive nature of this method is critical.

Constraining the mutual information between the discriminator's internal representation and the input allows the regularizer to directly limit the discriminator's accuracy, which automates the choice of noise magnitude and applies this noise to a compressed representation of the input that is specifically optimized to model the most discerning differences between the generator and data distributions.

The main contribution of this work is the variational discriminator bottleneck (VDB), an adaptive stochastic regularization method for adversarial learning that substantially improves performance across a range of different application domains, examples of which are available in FIG0 .

Our method can be easily applied to a variety of tasks and architectures.

First, we evaluate our method on a suite of challenging imitation tasks, including learning highly acrobatic skills from mocap data with a simulated humanoid character.

Our method also enables characters to learn dynamic continuous control skills directly from raw video demonstrations, and drastically improves upon previous work that uses adversarial imitation learning.

We further evaluate the effectiveness of the technique for inverse reinforcement learning, which recovers a reward function from demonstrations in order to train future policies.

Finally, we apply our framework to image generation using generative adversarial networks, where employing VDB improves the performance in many cases.

Recent years have seen an explosion of adversarial learning techniques, spurred by the success of generative adversarial networks (GANs) .

A GAN framework is commonly composed of a discriminator and a generator, where the discriminator's objective is to classify samples as real or fake, while the generator's objective is to produce samples that fool the discriminator.

Similar frameworks have also been proposed for inverse reinforcement learning (IRL) BID11 and imitation learning BID19 .

The training of adversarial models can be extremely unstable, with one of the most prevalent challenges being balancing the interplay between the discriminator and the generator BID4 .

The discriminator can often overpower the generator, easily differentiating between real and fake samples, thus providing the generator with uninformative gradients for improvement BID7 .

Alternative loss functions have been proposed to mitigate this problem BID31 BID47 BID49 .

Regularizers have been incorporated to improve stability and convergence, such as gradient penalties BID24 BID14 , reconstruction loss BID7 , and a myriad of other heuristics BID39 BID4 .

Task-specific architectural designs can also substantially improve performance BID37 BID21 .

Similarly, our method also aims to regularize the discriminator in order to improve the feedback provided to the generator.

But instead of explicit regularization of gradients or architecture-specific constraints, we apply a general information bottleneck to the discriminator, which previous works have shown to encourage networks to ignore irrelevant cues BID0 ).

We hypothesize that this then allows the generator to focus on improving the most discerning differences between real and fake samples.

Adversarial techniques have also been applied to inverse reinforcement learning BID12 , where a reward function is recovered from demonstrations, which can then be used to train policies to reproduce a desired skill.

BID10 showed an equivalence between maximum entropy IRL and GANs.

Similar techniques have been developed for adversarial imitation learning BID19 BID32 , where agents learn to imitate demonstrations without explicitly recovering a reward function.

One advantage of adversarial methods is that by leveraging a discriminator in place of a reward function, they can be applied to imitate skills where reward functions can be difficult to engineer.

However, the performance of policies trained through adversarial methods still falls short of those produced by manually designed reward functions, when such reward functions are available BID38 BID36 .

We show that our method can significantly improve upon previous works that use adversarial techniques, and produces results of comparable quality to those from state-of-the-art approaches that utilize manually engineered reward functions.

Our variational discriminator bottleneck is based on the information bottleneck BID44 , a technique for regularizing internal representations to minimize the mutual information with the input.

Intuitively, a compressed representation can improve generalization by ignoring irrelevant distractors present in the original input.

The information bottleneck can be instantiated in practical deep models by leveraging a variational bound and the reparameterization trick, inspired by a similar approach in variational autoencoders (VAE) BID23 .

The resulting variational information bottleneck approximates this compression effect in deep networks BID1 BID0 .

A similar bottleneck has also been applied to learn disentangled representations BID17 .

Building on the success of VAEs and GANs, a number of efforts have been made to combine the two.

BID30 used adversarial discriminators during the training of VAEs to encourage the marginal distribution of the latent encoding to be similar to the prior distribution, similar techniques include and BID8 .

Conversely, Larsen et al. (2016) modeled the generator of a GAN using a VAE.

BID47 used an autoencoder instead of a VAE to model the discriminator, but does not enforce an information bottleneck on the encoding.

While instance noise is widely used in modern architectures BID39 , we show that explicitly enforcing an information bottleneck leads to improved performance over simply adding noise for a variety of applications.

In this section, we provide a review of the variational information bottleneck proposed by BID1 in the context of supervised learning.

Our variational discriminator bottleneck is based on the same principle, and can be instantiated in the context of GANs, inverse RL, and imitation learning.

Given a dataset {x i , y i }, with features x i and labels y i , the standard maximum likelihood estimate q(y i |x i ) can be determined according to DISPLAYFORM0 Unfortunately, this estimate is prone to overfitting, and the resulting model can often exploit idiosyncrasies in the data BID26 BID43 .

BID1 proposed regularizing the model using an information bottleneck to encourage the model to focus only on the most discriminative features.

The bottleneck can be incorporated by first introducing an encoder E(z|x) that maps the features x to a latent distribution over Z, and then enforcing an upper bound I c on the mutual information between the encoding and the original features I(X, Z).

This results in the following regularized objective J(q, E) DISPLAYFORM1 Note that the model q(y|z) now maps samples from the latent distribution z to the label y. The mutual information is defined according to DISPLAYFORM2 where p(x) is the distribution given by the dataset.

Computing the marginal distribution p(z) = E(z|x) p(x) dx can be challenging.

Instead, a variational lower bound can be obtained by using an approximation r(z) of the marginal.

Since KL [p(z)||r(z)] ≥ 0, p(z) log p(z) dz ≥ p(z) log r(z) dz, an upper bound on I(X, Z) can be obtained via the KL divergence, This provides an upper bound on the regularized objectiveJ(q, E) ≥ J(q, E), DISPLAYFORM3 DISPLAYFORM4 To solve this problem, the constraint can be subsumed into the objective with a coefficient β BID1 evaluated the method on supervised learning tasks, and showed that models trained with a VIB can be less prone to overfitting and more robust to adversarial examples.

DISPLAYFORM5

To outline our method, we first consider a standard GAN framework consisting of a discriminator D and a generator G, where the goal of the discriminator is to distinguish between samples from the target distribution p * (x) and samples from the generator G(x), DISPLAYFORM0 We incorporate a variational information bottleneck by introducing an encoder E into the discriminator that maps a sample x to a stochastic encoding z ∼ E(z|x), and then apply a constraint I c on the mutual information I(X, Z) between the original features and the encoding.

D is then trained to classify samples drawn from the encoder distribution.

A schematic illustration of the framework is available in FIG1 .

The regularized objective J(D, E) for the discriminator is given by DISPLAYFORM1 DISPLAYFORM2 G being a mixture of the target distribution and the generator.

We refer to this regularizer as the variational discriminator bottleneck (VDB).

To optimize this objective, we can introduce a Lagrange multiplier β, DISPLAYFORM3 (8) As we will discuss in Section 4.1 and demonstrate in our experiments, enforcing a specific mutual information budget between x and z is critical for good performance.

We therefore adaptively update β via dual gradient descent to enforce a specific constraint I c on the mutual information, DISPLAYFORM4 where DISPLAYFORM5 and α β is the stepsize for the dual variable in dual gradient descent BID5 .

In practice, we perform only one gradient step on D and E, followed by an update to β.

We refer to a GAN that incorporates a VDB as a variational generative adversarial network (VGAN).In our experiments, the prior r(z) = N (0, I) is modeled with a standard Gaussian.

The encoder E(z|x) = N (µ E (x), Σ E (x)) models a Gaussian distribution in the latent variables Z, with mean µ E (x) and diagonal covariance matrix Σ E (x).

When computing the KL loss, each batch of data contains an equal number of samples from p * (x) and G(x).

We use a simplified objective for the generator, max DISPLAYFORM6 (11) where the KL penalty is excluded from the generator's objective.

Instead of computing the expectation over Z, we found that approximating the expectation by evaluating D at the mean µ E (x) of the encoder's distribution was sufficient for our tasks.

The discriminator is modeled with a single linear unit followed by a sigmoid DISPLAYFORM7

To interpret the effects of the VDB, we consider the results presented by , which show that for two distributions with disjoint support, the optimal discriminator can perfectly classify all samples and its gradients will be zero almost everywhere.

Thus, as the discriminator converges to the optimum, the gradients for the generator vanishes accordingly.

To address this issue, proposed applying continuous noise to the discriminator inputs, thereby ensuring that the distributions have continuous support everywhere.

In practice, if the original distributions are sufficiently distant from each other, the added noise will have negligible effects.

As shown by , the optimal choice for the variance of the noise to ensure convergence can be quite delicate.

In our method, by first using a learned encoder to map the inputs to an embedding and then applying an information bottleneck on the embedding, we can dynamically adjust the variance of the noise such that the distributions not only share support in the embedding space, but also have significant overlap.

Since the minimum amount of information required for binary classification is 1 bit, by selecting an information constraint I c < 1, the discriminator is prevented from from perfectly differentiating between the distributions.

To illustrate the effects of the VDB, we consider a simple task of training a discriminator to differentiate between two Gaussian distributions.

FIG1 visualizes the decision boundaries learned with different bounds I c on the mutual information.

Without a VDB, the discriminator learns a sharp decision boundary, resulting in vanishing gradients for much of the space.

But as I c decreases and the bound tightens, the decision boundary is smoothed, providing more informative gradients that can be leveraged by the generator.

Taking this analysis further, we can extend Theorem 3.2 from to analyze the VDB, and show that the gradient of the generator will be non-degenerate for a small enough constraint I c , under some additional simplifying assumptions.

The result in states that the gradient consists of vectors that point toward samples on the data manifold, multiplied by coefficients that depend on the noise.

However, these coefficients may be arbitrarily small if the generated samples are far from real samples, and the noise is not large enough.

This can still cause the generator gradient to vanish.

In the case of the VDB, the constraint ensures that these coefficients are always bounded below.

Due to space constraints, this result is presented in Appendix A.

To extend the VDB to imitation learning, we start with the generative adversarial imitation learning (GAIL) framework BID19 , where the discriminator's objective is to differentiate between the state distribution induced by a target policy π * (s) and the state distribution of the agent's policy π(s), max DISPLAYFORM0 Figure 3: Simulated humanoid performing various skills.

VAIL is able to closely imitate a broad range of skills from mocap data.

The discriminator is trained to maximize the likelihood assigned to states from the target policy, while minimizing the likelihood assigned to states from the agent's policy.

The discriminator also serves as the reward function for the agent, which encourages the policy to visit states that, to the discriminator, appear indistinguishable from the demonstrations.

Similar to the GAN framework, we can incorporate a VDB into the discriminator, DISPLAYFORM1 π represents a mixture of the target policy and the agent's policy.

The reward for π is then specified by the discriminator r t = −log (1 − D(µ E (s))).

We refer to this method as variational adversarial imitation learning (VAIL).

The VDB can also be applied to adversarial inverse reinforcement learning BID12 to yield a new algorithm which we call variational adversarial inverse reinforcement learning (VAIRL).

AIRL operates in a similar manner to GAIL, but with a discriminator of the form DISPLAYFORM0 where f (s, a, s ) = g(s, a) + γh(s ) − h(s), with g and h being learned functions.

Under certain restrictions on the environment, Fu et al. show that if g(s, a) is defined to depend only on the current state s, the optimal g(s) recovers the expert's true reward function r * (s) up to a constant g * (s) = r * (s) + const.

In this case, the learned reward can be re-used to train policies in environments with different dynamics, and will yield the same policy as if the policy was trained under the expert's true reward.

In contrast, GAIL's discriminator typically cannot be re-optimized in this way BID12 .

In VAIRL, we introduce stochastic encoders E g (z g |s), E h (z h |s), and g(z g ), h(z h ) are modified to be functions of the encoding.

We can reformulate Equation 13 as DISPLAYFORM1 , DISPLAYFORM2 We then obtain a modified objective of the form DISPLAYFORM3 where π(s, s ) denotes the joint distribution of successive states from a policy, and E(z|s, DISPLAYFORM4 Figure 4: Learning curves comparing VAIL to other methods for motion imitation.

Performance is measured using the average joint rotation error between the simulated character and the reference motion.

Each method is evaluated with 3 random seeds.

We evaluate our method on adversarial learning problems in imitation learning, inverse reinforcement learning, and image generation.

In the case of imitation learning, we show that the VDB enables agents to learn complex motion skills from a single demonstration, including visual demonstrations provided in the form of video clips.

We also show that the VDB improves the performance of inverse RL methods.

Inverse RL aims to reconstruct a reward function from a set demonstrations, which can then used to perform the task in new environments, in contrast to imitation learning, which aims to recover a policy directly.

Our method is also not limited to control tasks, and we demonstrate its effectiveness for unconditional image generation.

The goal of the motion imitation tasks is to train a simulated character to mimic demonstrations provided by mocap clips recorded from human actors.

Each mocap clip provides a sequence of target states {s * 0 , s * 1 , ..., s * T } that the character should track at each timestep.

We use a similar experimental setup as BID36 , with a 34 degrees-of-freedom humanoid character.

We found that the discriminator architecture can greatly affect the performance on complex skills.

The particular architecture we employ differs substantially from those used in prior work BID32 , details of which are available in Appendix C. The encoding Z is 128D and an information constraint of I c = 0.5 is applied for all skills, with a dual stepsize of α β = 10 −5 .

All policies are trained using PPO .The motions learned by the policies are best seen in the supplementary video.

Snapshots of the character's motions are shown in Figure 3 .

Each skill is learned from a single demonstration.

VAIL is able to closely reproduce a variety of skills, including those that involve highly dynamics flips and complex contacts.

We compare VAIL to a number of other techniques, including state-only GAIL BID19 , GAIL with instance noise applied to the discriminator inputs (GAIL -noise), GAIL with instance noise applied to the last hidden layer (GAIL -noise z), and GAIL with a gradient penalty applied to the discriminator (GAIL -GP) .

Since the VDB helps to prevent vanishing gradients, while GP mitigates exploding gradients, the two techniques can be seen as being complementary.

Therefore, we also train a model that combines both VAIL and GP (VAIL - GP).

Implementation details for combining the VDB and GP are available in Appendix B. Learning curves for the various methods are shown in FIG0 and Table 1 summarizes the performance of the final policies.

Performance is measured in terms of the average joint rotation error between the simulated character and the reference motion.

We also include a reimplementation of the method described by BID32 .

For the purpose of our experiments, GAIL denotes policies trained using our particular architecture but without a VDB, and BID32 denotes policies trained using an architecture that closely mirror those from previous work.

Furthermore, we include comparisons to policies trained using the handcrafted reward from BID36 , as well as policies trained via behavioral cloning (BC).

Since mocap data does not provide expert actions, we use the policies from BID36 as oracles to provide state-action demonstrations, which are then used to train the BC policies via supervised learning.

Each BC policy is trained with 10k samples from the oracle policies, while all other policies are trained from just a single demonstration, the equivalent of approximately 100 samples.

VAIL consistently outperforms previous adversarial methods, and VAIL -GP achieves the best performance overall.

Simply adding instance noise to the inputs BID39 or hidden layer without the KL constraint leads to worse performance, since the network can learn a latent representation that renders the effects of the noise negligible.

Though training with the handcrafted reward still outperforms the adversarial methods, VAIL demonstrates comparable performance to the handcrafted reward without manual reward or feature engineering, and produces motions that closely resemble the original demonstrations.

The method from BID32 was able to imitate simple skills such as running, but was unable to reproduce more acrobatic skills such as the backflip and spinkick.

In the case of running, our implementation produces more natural gaits than the results reported in BID32 .

Behavioral cloning is unable to reproduce any of the skills, despite being provided with substantially more demonstration data than the other methods.

Video Imitation: While our method achieves substantially better results on motion imitation when compared to prior work, previous methods can still produce reasonable behaviors.

However, if the demonstrations are provided in terms of the raw pixels from video clips, instead of mocap data, the imitation task becomes substantially harder.

The goal of the agent is therefore to directly im- Figure 7 : Left: C-Maze and S-Maze.

When trained on the training maze on the left, AIRL learns a reward that overfits to the training task, and which cannot be transferred to the mirrored maze on the right.

In contrast, VAIRL learns a smoother reward function that enables more-reliable transfer.

Right: Performance on flipped test versions of our two training mazes.

We report mean return (± std.

dev.) over five runs, and the mean return for the expert used to generate demonstrations.itate the skill depicted in the video.

This is also a setting where manually engineering rewards is impractical, since simple losses like pixel distance do not provide a semantically meaningful measure of similarity.

FIG4 compares learning curves of policies trained with VAIL, GAIL, and policies trained using a reward function defined by the average pixel-wise difference between the frame M * t from the video demonstration and a rendered image M t of the agent at each timestep t, DISPLAYFORM0 Each frame is represented by a 64 × 64 RGB image.

Both GAIL and the pixel-loss are unable to learn the running gait.

VAIL is the only method that successfully learns to imitate the skill from the video demonstration.

Snapshots of the video demonstration and the simulated motion is available in FIG3 .

To further investigate the effects of the VDB, we visualize the gradient of the discriminator with respect to images from the video demonstration and simulation.

Saliency maps for discriminators trained with VAIL and GAIL are available in FIG3 .

The VAIL discriminator learns to attend to spatially coherent image patches around the character, while the GAIL discriminator exhibits less structure.

The magnitude of the gradients from VAIL also tend to be significantly larger than those from GAIL, which may suggests that VAIL is able to mitigate the problem of vanishing gradients present in GAIL.Adaptive Constraint: To evaluate the effects of the adaptive β updates, we compare policies trained with different fixed values of β and policies where β is updated adaptively to enforce a desired information constraint I c = 0.5.

FIG4 illustrates the learning curves and the KL loss over the course of training.

When β is too small, performance reverts to that achieved by GAIL.

Large values of β help to smooth the discriminator landscape and improve learning speed during the early stages of training, but converges to a worse performance.

Policies trained using dual gradient descent to adaptively update β consistently achieves the best performance overall.

Next, we use VAIRL to recover reward functions from demonstrations.

Unlike the discriminator learned by VAIL, the reward function recovered by VAIRL can be re-optimized to train new policies from scratch in the same environment.

In some cases, it can also be used to transfer similar behaviour to different environments.

In Figure 7 , we show the results of applying VAIRL to the C-maze from BID12 , and a more complex S-maze; the simple 2D observation spaces of these tasks make it easy to interpret the recovered reward functions.

In both mazes, the expert is trained to navigate from a start position at the bottom of the maze to a fixed target position at the top.

We use each method to obtain an imitation policy and to approximate the expert's reward on the original maze.

The recovered reward is then used to train a new policy to solve a left-right flipped version of the training maze.

On the C-maze, we found that plain AIRL-without a gradient penaltywould sometimes overfit and fail to transfer to the new environment, as evidenced by the reward visualization in Figure 7 (left) and the higher return variance in Figure 7 (right).

In contrast, by incorporating a VDB into AIRL, VAIRL learns a substantially smoother reward function that is more suitable for transfer.

Furthermore, we found that in the S-maze with two internal walls, AIRL was too unstable to acquire a meaningful reward function.

This was true even with the use of a gradient penalty.

In contrast, VAIRL was able to learn a reasonable reward in most cases without a gradient penalty, and its performance improved even further with the addition of a gradient penalty.

To evaluate the effects of the VDB, we observe that the performance of VAIRL drops on both tasks when the KL constraint is disabled (β = 0), suggesting that the improvements from the VDB cannot be attributed entirely to the noise introduced by the sampling process for z. Further details of these experiments and illustrations of the recovered reward functions are available in Appendix D.

Finally, we apply the VDB to image generation with generative adversarial networks, which we refer to as VGAN.

Experiment are conducted on CIFAR-10 (Krizhevsky et al.), CelebA BID28 ), and CelebAHQ BID22 datasets.

We compare our approach to recent stabilization techniques: WGAN-GP , instance noise , spectral normalization (SN) , and gradient penalty (GP) , as well as the original GAN on CIFAR-10.

To measure performance, we report the Fréchet Inception Distance (FID) BID16 , which has been shown to be more consistent with human evaluation.

All methods are implemented using the same base model, built on the resnet architecture of .

Aside from tuning the KL constraint I c for VGAN, no additional hyperparameter optimization was performed to modify the settings provided by .

The performance of the various methods on CIFAR-10 are shown in FIG5 .

While vanilla GAN and instance noise are prone to diverging as training progresses, VGAN remains stable.

Note that instance noise can be seen as a non-adaptive version of VGAN without constraints on I c .

This experiment again highlights that there is a significant improvement from imposing the information bottleneck over simply adding instance noise.

Combining both VDB and gradient penalty (VGAN -GP) achieves the best performance overall with an FID of 18.1.

We also experimented with combining the VDB with SN, but this combination is prone to diverging.

See FIG6 for samples of images generated with our approach.

Please refer to Appendix E for experimental details and more results.

We present the variational discriminator bottleneck, a general regularization technique for adversarial learning.

Our experiments show that the VDB is broadly applicable to a variety of domains, and yields significant improvements over previous techniques on a number of challenging tasks.

While our experiments have produced promising results for video imitation, the results have been primarily with videos of synthetic scenes.

We believe that extending the technique to imitating realworld videos is an exciting direction.

Another exciting direction for future work is a more in-depth theoretical analysis of the method, to derive convergence and stability results or conditions.

In this appendix, we show that the gradient of the generator when the discriminator is augmented with the VDB is non-degenerate, under some mild additional assumptions.

First, we assume a pointwise constraint of the form KL[E(z|x) r(z)]

≤ I c for all x. In reality, we use an average KL constraint, since we found it to be more convenient to optimize, though a pointwise constraint is also possible to enforce by using the largest constraint violation to increment β.

We could likely also extend the analysis to the average constraint, though we leave this to future work.

The main theorem can then be stated as follows:Theorem A.1.

Let g(u) denote the generator's mapping from a noise vector u ∼ p(u) to a point in X. Given the generator distribution G(x) and data distribution p * (x), a VDB with an encoder E(z|x) = N (µ E (x), Σ), and KL[E(z|x) r(z)]

≤ I c , the gradient passed to the generator has the form DISPLAYFORM0 where D * (z) is the optimal discriminator, a(x) and b(x) are positive functions, and we always have E(µ E (g(u))|x) > C(I c ), where C(I c ) is a continuous monotonic function, and C(I c ) → δ > 0 as I c → 0.Analysis for an encoder with an input-dependent variance Σ(x) is also possible, but more involved.

We'll further assume below for notational simplicity that Σ is diagonal with diagonal values σ 2 .

This assumption is not required, but substantially simplifies the linear algebra.

Analogously to Theorem 3.2 from , this theorem states that the gradient of the generator points in the direction of points in the data distribution, and away from points in the generator distribution.

However, going beyond the theorem in , this result states that the coefficients on these vectors, given by E(µ E (g(u))|x), are always bounded below by a value that approaches a positive constant δ as we decrease I c , meaning that the gradient does not vanish.

The proof of the first part of this theorem is essentially identical to the proof presented by , but accounting for the fact that the noise is now injected into the latent space of the VDB, rather than being added directly to x. This result assumes that E(z|x) has a learned but input-independent variance Σ = σ 2 I, though the proof can be repeated for an input-dependent or non-diagonal Σ:Proof.

Overloading p * (x) and G(x), let p * (z) and G(z) be the distribution of embeddings z under the real data and generator respectively.

p * (z) is then given by DISPLAYFORM1 and similarly for G(z) DISPLAYFORM2 From , the optimal discriminator between p * (z) and G(z) is DISPLAYFORM3 The gradient passed to the generator then has the form DISPLAYFORM4 We then have DISPLAYFORM5 )|x)dp DISPLAYFORM6 Similar to the result from , the gradient of the generator drives the generator's samples in the embedding space µ E (g(u)) towards embeddings of the points from the dataset µ E (x) weighted by their likelihood E(µ E (g(u))|x) under the real data.

For an arbitrary encoder E, real and fake samples in the embedding may be far apart.

As such, the coefficients E(µ E (g(u))|x) can be arbitrarily small, thereby resulting in vanishing gradients for the generator.

The second part of the theorem states that C(I c ) is a continuous monotonic function, and C(I c ) → δ > 0 as I c → 0.

This is the main result, and relies on the fact that KL[E(z|x)||r(z)]

≤ I c .

The intuition behind this result is that, for any two inputs x and y, their encoded distributions E(z|x) and E(z|y) have means that cannot be more than some distance apart, and that distance shrinks with I c .

This allows us to bound E(µ E (y))|x) below by C(I c ), which ensures that the coefficients on the vectors in the theorem above are always at least as large as C(I c ).Proof.

Let r(z) = N (0, I) be the prior distribution and suppose the KL divergence for all x in the dataset and all g(u) generated by the generator are bounded by I c DISPLAYFORM7 From the definition of the KL-divergence we can bound the length of all embedding vectors, DISPLAYFORM8 and similarly for ||µ E (g(u))|| 2 , with K denoting the dimension of Z. A lower bound on E(µ E (g(u))|x), where u ∼ p(u) and x ∼ p * (x), can then be determined by DISPLAYFORM9 Since DISPLAYFORM10 and it follows that DISPLAYFORM11 The likelihood is therefore bounded below by DISPLAYFORM12 From the KL constraint, we can derive a lower bound (I c ) and an upper bound U(I c ) on σ 2 .

DISPLAYFORM13 For the upper bound, since DISPLAYFORM14 Substituting (I c ) and U(I c ) into Equation 14, we arrive at the following lower bound DISPLAYFORM15

To combine VDB with gradient penalty, we use the reparameterization trick to backprop through the encoder when computing the gradient of the discriminator with respect to the inputs.

DISPLAYFORM0 The coefficient w GP weights the gradient penalty in the objective, w GP = 10 for the image generation, w GP = 1 for motion imitation, and w GP = 0.1 (C-maze) or w GP = 0.01 (S-maze) for the IRL tasks.

The gradient penalty is applied only to real samples p * (x).

We have experimented with apply the penalty to both real and fake samples, but found that performance was worse than penalizing only gradients from real samples.

This is consistent with the GP implementation from .

Experimental Setup: The goal of the motion imitation tasks is to train a simulated agent to mimic a demonstration provided in the form of a mocap clip recorded from a human actor.

We use a similar experimental setup as BID36 , with a 34 degrees-of-freedom humanoid character.

The state s consists of features that represent the configuration of the character's body (link positions and velocities).

We also include a phase variable φ ∈ [0, 1] among the state features, which records the character's progress along the motion and helps to synchronize the character with the reference motion.

With 0 and 1 denoting the start and end of the motion respectively.

The action a sampled from the policy π(a|s) specifies target poses for PD controller positioned at each joint.

Given a state, the policy specifies a Gaussian distribution over the action space π(a|s) = N (µ(s), Σ), with a state-dependent mean µ(s) and fixed diagonal covariance matrix Σ. µ(s) is modeled using a 3-layered fully-connected network with 1024 and 512 hidden units, followed by a linear output layer that specifies the mean of the Gaussian.

ReLU activations are used for all hidden layers.

The value function is modeled with a similar architecture but with a single linear output unit.

The policy is queried at 30Hz.

Physics simulation is performed at 1.2kHz using the Bullet physics engine Bullet (2015) .Given the rewards from the discriminator, PPO ) is used to train the policy, with a stepsize of 2.5 × 10 −6 for the policy, a stepsize of 0.01 for the value function, and a stepsize of 10 −5 for the discirminator.

Gradient descent with momentum 0.9 is used for all models.

The PPO clipping threshold is set to 0.2.

When evaluating the performance of the policies, each episode is simulated for a maximum horizon of 20s.

Early termination is triggered whenever the character's torso contacts the ground, leaving the policy is a maximum error of π radians for all remaining timesteps.

Phase-Functioned Discriminator: Unlike the policy and value function, which are modeled with standard fully-connected networks, the discriminator is modeled by a phase-functioned neural network (PFNN) to explicitly model the time-dependency of the reference motion BID20 .

While the parameters of a network are generally fixed, the parameters of a PFNN are functions of the phase variable φ.

The parameters θ of the network for a given φ is determined by a weighted combination of a set of fixed parameters {θ 0 , θ 1 , ..., θ k }, DISPLAYFORM0 where w i (φ) is a phase-dependent weight for θ i .

In our implementation, we use k = 5 sets of parameters and w i (φ) is designed to linearly interpolate between two adjacent sets of parameters for each phase φ, where each set of parameters θ i corresponds to a discrete phase value φ i spaced FIG0 : Learning curves comparing VAIL to other methods for motion imitation.

Performance is measured using the average joint rotation error between the simulated character and the reference motion.

Each method is evaluated with 3 random seeds.

Figure 11: Learning curves comparing VAIL with a discriminator modeled by a phase-functioned neural network (PFNN), to modeling the discriminator with a fully-conneted network that receives the phase-variable φ as part of the input (no PFNN), and a discriminator modeled with a fullyconnected network but does not receive φ as an input (no phase).uniformly between [0, 1].

For a given value of φ, the parameters of the discriminator are determined according to DISPLAYFORM1 where θ i and θ i+1 correspond to the phase values φ i ≤ φ < φ i+1 that form the endpoints of the phase interval that contains φ.

A PFNN is used for all motion imitation experiments, both VAIL and GAIL, except for those that use the approach proposed by BID32 , which use standard fully-connected networks for the discriminator.

FIG0 compares the performance of VAIL when the discriminator is modeled with a phase-functioned neural network (with PFNN) to discriminators modeled with standard fully-connected networks.

We increased the size of the layers of the fully-connected nets to have a similar number of parameters as a PFNN.

We evaluate the performance of fully-connected nets that receive the phase variable φ as part of the input (no PFNN), and fully-connected nets that do not receive φ as an input.

The phase-functioned discriminator leads to significant performance improvements across all tasks evaluated.

Policies trained without a phase variable performs worst overall, suggesting that phase information is critical for performance.

All methods perform well on simpler skills, such as running, but the additional phase structure introduced by the PFNN proved to be vital for successful imitation of more complex skills, such as the dance and backflip.

Next we compare the accuracy of discriminators trained using different methods.

FIG0 illustrates accuracy of the discriminators over the course of training.

Discriminators trained via GAIL quickly overpowers the policy, and learns to accurately differentiate between samples, even when instance noise is applied to the inputs.

VAIL without the KL constraint slows the discriminator's progress, but nonetheless reaches near perfect accuracy with a larger number of samples.

Once the KL constraint is enforced, the information bottleneck constrains the performance of the discriminator, converging to approximately 80% accuracy.

FIG0 also visualizes the value of β over the course of training for motion imitation tasks, along with the loss of the KL term in the objective.

The dual gradient descent update effectively enforces the VDB constraint I c .Video Imitation:

In the video imitation tasks, we use a simplified 2D biped character in order to avoid issues that may arise due to depth ambiguity from monocular videos.

The biped character has a total of 12 degrees-of-freedom, with similar state and action parameters as the humanoid.

The video demonstrations are generated by rendering a reference motion into a sequence of video frames, which are then provided to the agent as a demonstration.

The goal of the agent is to imitate the motion depicted in the video, without access to the original reference motion, and the reference motion is used only to evaluate performance.

Environments We evaluate on two maze tasks, as illustrated in FIG0 .

The C-maze is taken from BID12 : in this maze, the agent starts at a random point within a small fixed distance of the mean start position.

The agent has a continuous, 2D action space which allows it to accelerate in the x or y directions, and is able to observe its x and y position, but not its velocity.

The ground truth reward is r t = −d t − 10 −3 a t 2 , where d t is the agent's distance to the goal, and a t is its action (this action penalty is assumed to be zero in FIG0 ).

Episodes terminate after 100 steps; for evaluation, we report the undiscounted mean sum of rewards over each episode The S-maze is larger variant of the same environment with an extra wall between the agent and its goal.

To make the S-maze easier to solve for the expert, we added further reward shaping to encourage the agent to pass between the gaps between walls.

We also increased the maximum control forces relative to the C-maze to enable more rapid exploration.

Environments will be released along with the rest of our VAIRL implementations.

Hyperparameters Policy networks for all methods were two-layer ReLU MLPs with 32 hidden units per layer.

Reward and discriminator networks were similar, but with 32-unit mean and standard deviation layers inserted before the final layer for VDB methods.

To generate expert demonstrations, we trained a TRPO BID40 agent on the ground truth reward for the training environment for 200 iterations, and saved 107 trajectories from each of the policies corresponding to the five final iterations.

TRPO used a batch size of 10,000, a step size of 0.01, and entropy bonus with a coefficient of 0.1 to increase diversity.

After generating demonstrations, we trained the IRL and imitation methods on a training maze for 200 iterations; again, our policy optimizer was TRPO with the same hyperparameters used to generate demonstrations.

Between each policy update, we did 100 discriminator updates using Adam with a learning rate of 5 × 10 −5 and batch size of 32.

For the C-maze our VAIRL runs used a target KL of I C = 0.5, while for the more complex S-maze we FIG0 : Left: The C-maze used for training and its mirror version used for testing.

Colour contours show the ground truth reward function that we use to train the expert and evaluate transfer quality, while the red and green dots show the initial and goal positions, respectively.

Right: The analogous diagram for the S-maze.

use a tighter target of I C = 0.05.

For the test C-maze, we trained new policies against the recovered reward using TRPO with the hyperparameters described above; for the test S-maze, we modified these parameters to use a batch size of 50,000 and learning rate of 0.001 for 400 iterations.

FIG0 and 15 show the reward functions recovered by each IRL baseline on the C-maze and S-maze, respectively, along with sample trajectories for policies trained to optimize those rewards.

Notice that VAIRL tends to recover smoother reward functions that match the ground truth reward more closely than the baselines.

Addition of a gradient penalty enhances this effect for both AIRL and VAIRL.

This is especially true in S-maze, where combining a gradient penalty with a variational discriminator bottleneck leads to a smooth reward that gradually increases as the agent nears its goal position at the top of the maze.

We provide further experiment on image generation and details of the experimental setup.

We use the non-saturating objective of for all models except WGAN-GP.

Following BID29 , we compute FID on samples of size 10000 2 .

We base our implementation on , where we do not use any batch normalization for both the generator and the discriminator.

We use RMSprop (Hinton et al.) and a fixed learning rate for all experiments.

For convolutional GAN, variational discriminative bottleneck is implemented as a 1x1 convolution on the final embedding space that outputs a Gaussian distribution over Z parametrized with a mean and a diagonal covariance matrix.

For all image experiments, we preserve the dimensionality of the latent space.

All experiments use adaptive β update with a dual stepsize of α β = 10 −5 .

We will make our code public.

Similarly to VGAN, instance noise ; is added to the final embedding space of the discriminator right before applying the classifier.

Instance noise can be interpreted as a non-adaptive VGAN without a information constraint.

Architecture: For CIFAR-10, we use a resnet-based architecture adapted from detailed in Tables 2, 3 , and 4.

For CelebA and CelebAHQ, we use the same architecture used in .

BID22 1024 × 1024 resolution at 300k iterations.

Models are trained from scratch at full resolution, without the progressive scheme proposed by BID21 .

@highlight

Regularizing adversarial learning with an information bottleneck, applied to imitation learning, inverse reinforcement learning, and generative adversarial networks.