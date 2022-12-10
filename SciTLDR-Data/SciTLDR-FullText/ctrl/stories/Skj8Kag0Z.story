Adversarial neural networks solve many important problems in data science, but are notoriously difficult to train.

These difficulties come from the fact that optimal weights for adversarial nets correspond to saddle points, and not minimizers, of the loss function.

The alternating stochastic gradient methods typically used for such problems do not reliably converge to saddle points, and when convergence does happen it is often highly sensitive to learning rates.

We propose a simple modification of stochastic gradient descent that stabilizes adversarial networks.

We show, both in theory and practice, that the proposed method reliably converges to saddle points.

This makes adversarial networks less likely to "collapse," and enables faster training with larger learning rates.

Adversarial networks play an important role in a variety of applications, including image generation (Zhang et al., 2017; Wang & Gupta, 2016) , style transfer BID2 Taigman et al., 2017; Wang & Gupta, 2016; BID17 , domain adaptation (Taigman et al., 2017; Tzeng et al., 2017; BID11 , imitation learning BID15 , privacy BID9 BID0 , fair representation (Mathieu et al., 2016; BID9 , etc.

One particularly motivating application of adversarial nets is their ability to form generative models, as opposed to the classical discriminative models BID13 Radford et al., 2016; BID7 Mirza & Osindero, 2014) .While adversarial networks have the power to attack a wide range of previously unsolved problems, they suffer from a major flaw: they are difficult to train.

This is because adversarial nets try to accomplish two objectives simultaneously; weights are adjusted to maximize performance on one task while minimizing performance on another.

Mathematically, this corresponds to finding a saddle point of a loss function -a point that is minimal with respect to one set of weights, and maximal with respect to another.

Conventional neural networks are trained by marching down a loss function until a minimizer is reached ( FIG0 ).

In contrast, adversarial training methods search for saddle points rather than a minimizer, which introduces the possibility that the training path "slides off" the objective functions and the loss goes to −∞ FIG0 ), resulting in "collapse" of the adversarial network.

As a result, many authors suggest using early stopping, gradients/weight clipping , or specialized objective functions BID13 Zhao et al., 2017; to maintain stability.

In this paper, we present a simple "prediction" step that is easily added to many training algorithms for adversarial nets.

We present theoretical analysis showing that the proposed prediction method is asymptotically stable for a class of saddle point problems.

Finally, we use a wide range of experiments to show that prediction enables faster training of adversarial networks using large learning rates without the instability problems that plague conventional training schemes.

If minimization (or, conversely, maximization) is more powerful, the solution path "slides off" the loss surface and the algorithm becomes unstable, resulting in a sudden "collapse" of the network.

Saddle-point optimization problems have the general form DISPLAYFORM0 for some loss function L and variables u and v. Most authors use the alternating stochastic gradient method to solve saddle-point problems involving neural networks.

This method alternates between updating u with a stochastic gradient descent step, and then updating v with a stochastic gradient ascent step.

When simple/classical SGD updates are used, the steps of this method can be written DISPLAYFORM1 Here, {α k } and {β k } are learning rate schedules for the minimization and maximization steps, respectively.

The vectors L u (u, v) and L v (u, v) denote (possibly stochastic) gradients of L with respect to u and v. In practice, the gradient updates are often performed by an automated solver, such as the Adam optimizer BID19 , and include momentum updates.

We propose to stabilize the training of adversarial networks by adding a prediction step.

Rather than calculating v k+1 using u k+1 , we first make a prediction,ū k+1 , about where the u iterates will be in the future, and use this predicted value to obtain v k+1 .

The Prediction step (3) tries to estimate where u is going to be in the future by assuming its trajectory remains the same as in the current iteration.

We now discuss a few common adversarial network problems and their saddle-point formulations.

Generative Adversarial Networks (GANs) fit a generative model to a dataset using a game in which a generative model competes against a discriminator BID13 .

The generator, G(z; θ g ), takes random noise vectors z as inputs, and maps them onto points in the target data distribution.

The discriminator, D(x; θ d ), accepts a candidate point x and tries to determine whether it is really drawn from the empirical distribution (in which case it outputs 1), or fabricated by the generator (output 0).

During a training iteration, noise vectors from a Gaussian distribution G are pushed through the generator network G to form a batch of generated data samples denoted by D f ake .

A batch of empirical samples, D real , is also prepared.

One then tries to adjust the weights of each network to solve a saddle point problem, which is popularly formulated as, DISPLAYFORM0 Here f (.) is any monotonically increasing function.

Initially, BID13 proposed using f (x) = log(x).Domain Adversarial Networks (DANs) (Makhzani et al., 2016; BID11 BID9 ) take data collected from a "source" domain, and extract a feature representation that can be used to train models that generalize to another "target" domain.

For example, in the domain adversarial neural network (DANN BID11 ), a set of feature layers maps data points into an embedded feature space, and a classifier is trained on these embedded features.

Meanwhile, the adversarial discriminator tries to determine, using only the embedded features, whether the data points belong to the source or target domain.

A good embedding yields a better task-specific objective on the target domain while fooling the discriminator, and is found by solving DISPLAYFORM1 Here L d is any adversarial discriminator loss function and L y k denotes the task specific loss.

θ f , θ d , and θ y k are network parameter of feature mapping, discriminator, and classification layers.

It is well known that alternating stochastic gradient methods are unstable when using simple logarithmic losses.

This led researchers to explore multiple directions for stabilizing GANs; either by adding regularization terms BID12 BID4 Zhao et al., 2017) , a myriad of training "hacks" (Salimans et al., 2016; BID14 , re-engineering network architectures (Zhao et al., 2017) , and designing different solvers (Metz et al., 2017) .

Specifically, the Wasserstein GAN (WGAN) approach modifies the original objective by replacing f (x) = log(x) with f (x) = x. This led to a training scheme in which the discriminator weights are "clipped."

However, as discussed in , the WGAN training is unstable at high learning rates, or when used with popular momentum based solvers such as Adam.

Currently, it is known to work well only with RMSProp .The unrolled GAN (Metz et al., 2017 ) is a new solver that can stabilize training at the cost of more expensive gradient computations.

Each generator update requires the computation of multiple extra discriminator updates, which are then discarded when the generator update is complete.

While avoiding GAN collapse, this method requires increased computation and memory.

In the convex optimization literature, saddle point problems are more well studied.

One popular solver is the primal-dual hybrid gradient (PDHG) method (Zhu & Chan, 2008; BID10 , which has been popularized by BID3 , and has been successfully applied to a range of machine learning and statistical estimation problems BID12 .

PDHG relates closely to the method proposed here -it achieves stability using the same prediction step, although it uses a different type of gradient update and is only applicable to bi-linear problems.

Stochastic methods for convex saddle-point problems can be roughly divided into two categories: stochastic coordinate descent BID6 Lan & Zhou, 2015; Zhang & Lin, 2015; Zhu & Storkey, 2015; Wang & Xiao, 2017; Shibagaki & Takeuchi, 2017) and stochastic gradient descent BID5 Qiao et al., 2016) .

Similar optimization algorithms have been studied for reinforcement learning (Wang & Chen, 2016; BID8 .

Recently, a "doubly" stochastic method that randomizes both primal and dual updates was proposed for strongly convex bilinear saddle point problems (Yu et al., 2015) .

For general saddle point problems, "doubly" stochastic gradient descent methods are discussed in Nemirovski et al. (2009 ),Palaniappan & Bach (2016 , in DISPLAYFORM0 Figure 2: A schematic depiction of the prediction method.

When the minimization step is powerful and moves the iterates a long distance, the prediction step (dotted black arrow) causes the maximization update to be calculated further down the loss surface, resulting in a more dramatic maximization update.

In this way, prediction methods prevent the maximization step from getting overpowered by the minimization update.which primal and dual variables are updated simultaneously based on the previous iterates and the current gradients.

We present three ways to explain the effect of prediction: an intuitive, non-mathematical perspective, a more analytical viewpoint involving dynamical systems, and finally a rigorous proof-based approach.

The standard alternating SGD switches between minimization and maximization steps.

In this algorithm, there is a risk that the minimization step can overpower the maximization step, in which case the iterates will "slide off" the edge of saddle, leading to instability FIG0 .

Conversely, an overpowering maximization step will dominate the minimization step, and drive the iterates to extreme values as well.

The effect of prediction is visualized in Figure 2 .

Suppose that a maximization step takes place starting at the red dot.

Without prediction, the maximization step has no knowledge of the algorithm history, and will be the same regardless of whether the previous minimization update was weak ( Figure 2a ) or strong ( Figure 2b ).

Prediction allows the maximization step to exploit information about the minimization step.

If the previous minimizations step was weak (Figure 2a ), the prediction step (dotted black arrow) stays close to the red dot, resulting in a weak predictive maximization step (white arrow).

But if we arrived at the red dot using a strong minimization step (Figure 2b ), the prediction moves a long way down the loss surface, resulting in a stronger maximization step (white arrows) to compensate.

To get stronger intuition about prediction methods, let's look at the behavior of Algorithm (3) on a simple bi-linear saddle of the form DISPLAYFORM0 where K is a matrix.

When exact (non-stochastic) gradient updates are used, the iterates follow the path of a simple dynamical system with closed-form solutions.

We give here a sketch of this argument: a detailed derivation is provided in the Supplementary Material.

When the (non-predictive) gradient method (2) is applied to the linear problem (6), the resulting iterations can be written DISPLAYFORM1 When the stepsize α gets small, this behaves like a discretization of the system of differential equationṡ DISPLAYFORM2 whereu andv denote the derivatives of u and v with respect to time.

These equations describe a simple harmonic oscillator, and the closed form solution for u is DISPLAYFORM3 where Σ is a diagonal matrix, and the matrix C and vector φ depend on the initialization.

We can see that, for small values of α and β, the non-predictive algorithm (2) approximates an undamped harmonic motion, and the solutions orbit around the saddle without converging.

The prediction step (3) improves convergence because it produces damped harmonic motion that sinks into the saddle point.

When applied to the linearized problem (6), we get the dynamical systeṁ DISPLAYFORM4 which has solution DISPLAYFORM5 From this analysis, we see that the damping caused by the prediction step causes the orbits to converge into the saddle point, and the error decays exponentially fast.

While the arguments above are intuitive, they are also informal and do not address issues like stochastic gradients, non-constant stepsize sequences, and more complex loss functions.

We now provide a rigorous convergence analysis that handles these issues.

We assume that the function L(u, v) is convex in u and concave in v. We can then measure convergence using the "primal-dual" gap, DISPLAYFORM0 is a saddle.

Using these definitions, we formulate the following convergence result.

The proof is in the supplementary material.

Theorem 1.

Suppose the function L(u, v) is convex in u, concave in v, and that the partial gradient DISPLAYFORM1

, then the SGD method with prediction converges in expectation, and we have the error bound DISPLAYFORM0

We present a wide range of experiments to demonstrate the benefits of the proposed prediction step for adversarial nets.

We consider a saddle point problem on a toy dataset constructed using MNIST images, and then move on to consider state-of-the-art models for three tasks: GANs, domain adaptation, and learning of fair classifiers.

Additional results, and additional experiments involving mixtures of Gaussians, are presented in the Appendix.

The code is available at https: //github.com/jaiabhayk/stableGAN.

We consider the task of classifying MNIST digits as being even or odd.

To make the problem interesting, we corrupt 70% of odd digits with salt-and-pepper noise, while we corrupt only 30% of even digits.

When we train a LeNet network (LeCun et al., 1998) on this problem, we find that the network encodes and uses information about the noise; when a noise vs no-noise classifier is trained on the deep features generated by LeNet, it gets 100% accuracy.

The goal of this task is to force LeNet to ignore the noise when making decisions.

We create an adversarial model of the form (5) in which L y is a softmax loss for the even vs odd classifier.

We make L d a softmax loss for the task of discriminating whether the input sample is noisy or not.

The classifier and discriminator were both pre-trained using the default LeNet implementation in Caffe BID18 .

Then the combined adversarial net was jointly trained both with and without prediction.

For implementation details, see the Supplementary Material.

Figure 3 summarizes our findings.

In this experiment, we considered applying prediction to both the classifier and discriminator.

We note that our task is to retain good classification accuracy while preventing the discriminator from doing better than the trivial strategy of classifying odd digits as noisy and even digits as non-noisy.

This means that the discriminator accuracy should ideally be ∼ 0.7.

As shown in FIG1 , the prediction step hardly makes any difference when evaluated at the small learning rate of 10 −4.

However, when evaluated at higher rates, FIG1 show that the prediction solvers are very stable while one without prediction collapses (blue solid line is flat) very early.

FIG1 shows that the default learning rate (10 DISPLAYFORM0 ) of the Adam solver is unstable unless prediction is used.

Next, we test the efficacy and stability of our proposed predictive step on generative adversarial networks (GAN), which are formulated as saddle point problems (4) and are popularly solved using a heuristic approach BID13 .

We consider an image modeling task using CIFAR-10 (Krizhevsky, 2009) on the recently popular convolutional GAN architecture, DCGAN (Radford et al., 2016) .

We compare our predictive method with that of DCGAN and the unrolled GAN (Metz et al., 2017) using the training protocol described in Radford et al. (2016) .

Note that we compared against the unrolled GAN with stop gradient switch 1 and K = 5 unrolling steps.

All the approaches were trained for five random seeds and 100 epochs each.

We start with comparing all three methods using the default solver for DCGAN (the Adam optimizer) with learning rate=0.0002 and β 1 =0.5.

Figure 4 compares the generated sample images (at the 100 th epoch) and the training loss curve for all approaches.

The discriminator and generator loss curves in Figure 4e show that without prediction, the DCGAN collapses at the 45 th and 57 th epochs.

Similarly, Figure 4f shows that the training for unrolled GAN collapses in at least three instances.

The training procedure using predictive steps never collapsed during any epochs.

Qualitatively, the images generated using prediction are more diverse than the DCGAN and unrolled GAN images.

Figure 5 compares all approaches when trained with 5× higher learning rate (0.001) (the default for the Adam solver).

As observed in Radford et al. (2016) , the standard and unrolled solvers are very unstable and collapse at this higher rate.

However, as shown in Figure 5d , & 5a, training remains stable when a predictive step is used, and generates images of reasonable quality.

The training procedure for both DCGAN and unrolled GAN collapsed on all five random seeds.

The results on various additional intermediate learning rates as well as on high resolution Imagenet dataset are in the Supplementary Material.

In the Supplementary Material, we present one additional comparison showing results on a higher momentum of β 1 =0.9 (learning rate=0.0002).

We observe that all the training approaches are stable.

However, the quality of images generated using DCGAN is inferior to that of the predictive and unrolled methods.

Overall, of the 25 training settings we ran on (each of five learning rates for five random seeds), the DCGAN training procedure collapsed in 20 such instances while unrolled GAN collapsed in 14 experiments (not counting the multiple collapse in each training setting).

On the contrary, we find that our simple predictive step method collapsed only once.

Note that prediction adds trivial cost to the training algorithm.

Using a single TitanX Pascal, a training epoch of DCGAN takes 35 secs. With prediction, an epoch takes 38 secs. The unrolled GAN method, which requires extra gradient steps, takes 139 secs/epoch.

Finally, we draw quantitative comparisons based on the inception score (Salimans et al., 2016) , which is a widely used metric for visual quality of the generated images.

For this purpose, we consider the current state-of-the-art Stacked GAN BID16 architecture.

TAB0 lists the inception scores computed on the generated samples from Stacked GAN trained (200 epochs) with and without prediction at different learning rates.

The joint training of Stacked GAN collapses when trained at the default learning rate of adam solver (i.e., 0.001).

However, reasonably good samples are generated if the same is trained with prediction on both the generator networks.

The right end of TAB0 also list the inception score measured at fewer number of epochs for higher learning rates.

It suggest that the model trained with prediction methods are not only stable but also allows faster convergence using higher learning rates.

For reference the inception score on real images of CIFAR-10 dataset is 11.51 ± 0.17.

We consider the domain adaptation task (Saenko et al., 2010; BID11 Tzeng et al., 2017) wherein the representation learned using the source domain samples is altered so that it can also generalize to samples from the target distribution.

We use the problem setup and hyper-parameters as described in BID11 using the OFFICE dataset (Saenko et al., 2010) (experimental details are shared in the Supplementary Material).

In TAB1 , comparisons are drawn with respect to target domain accuracy on six pairs of source-target domain tasks.

We observe that the prediction step has mild benefits on the "easy" adaptation tasks with very similar source and target domain samples.

However, on the transfer learning tasks of AMAZON-to-WEBCAM, WEBCAM-to-AMAZON, and DSLR-to-AMAZON which has noticeably distinct data samples, an extra prediction step gives an absolute improvement of 1.3 − 6.9% in predicting target domain labels.

Finally, we consider a task of learning fair feature representations (Mathieu et al., 2016; BID9 Louizos et al., 2016) such that the final learned classifier does not discriminate with respect to a sensitive variable.

As proposed in BID9 one way to measure fairness is using discrimination, DISPLAYFORM0 Here s i is a binary sensitive variable for the i th data sample and N k denotes the total number of samples belonging to the k th sensitive class.

Similar to the domain adaptation task, the learning of each classifier can be formulated as a minimax problem in (5) BID9 Mathieu et al., 2016) .

Unlike the previous example though, this task has a model selection component.

From a pool of hundreds of randomly generated adversarial deep nets, for each value of t, one selects the model that maximizes the difference y t,Delta = y acc − t * y disc .The "Adult" dataset from the UCI machine learning repository is used.

The task (y acc ) is to classify whether a person earns ≥ $50k/year.

The person's gender is chosen to be the sensitive variable.

Details are in the supplementary.

To demonstrate the advantage of using prediction for model selection, we follow the protocol developed in BID9 .

In this work, the search space is restricted to a class of models that consist of a fully connected autoencoder, one task specific discriminator, and one adversarial discriminator.

The encoder output from the autoencoder acts as input to both the discriminators.

In our experiment, 100 models are randomly selected.

During the training of each adversarial model, L d is a cross-entropy loss while L y is a linear combination of reconstruction and cross-entropy loss.

Once all the models are trained, the best model for each value of t is selected by evaluating (9) on the validation set.

FIG3 plots the results on the test set for the AFLR approach with and without prediction steps in their default Adam solver.

For each value of t, FIG3 , 6c also compares the number of layers in the selected encoder and discriminator networks.

When using prediction for training, relatively stronger encoder models are produced and selected during validation, and hence the prediction results generalize better on the test set.

We present a simple modification to the alternating SGD method, called a prediction step, that improves the stability of adversarial networks.

We present theoretical results showing that the prediction step is asymptotically stable for solving saddle point problems.

We show, using a variety of test problems, that prediction steps prevent network collapse and enable training with a wider range of learning rates than plain SGD methods.

Here, we provide a detailed derivation of the harmonic oscillator behavior of Algorithm (3) on the simple bi-linear saddle of the form L(x, y) = y T Kx where K is a matrix.

Note that, within a small neighborhood of a saddle, all smooth weakly convex objective functions behave like (6).To see why, consider a smooth objective function L with a saddle point at x * = 0, y * = 0.

Within a small neighborhood of the saddle, we can approximate the function L to high accuracy using its Taylor approximation DISPLAYFORM0 where L xy denotes the matrix of mixed-partial derivatives with respect to x and y. Note that the first-order terms have vanished from this Taylor approximation because the gradients are zero at a saddle point.

The O( x 2 ) and O( y 2 ) terms vanish as well because the problem is assumed to be weakly convex around the saddle.

Up to third-order error (which vanishes quickly near the saddle), this Taylor expansion has the form (6).

For this reason, stability on saddles of the form (6) is a necessary condition for convergence of FORMULA2 , and the analysis here describes the asymptotic behavior of the prediction method on any smooth problem for which the method converges.

We will show that, as the learning rate gets small, the iterates of the non-prediction method (2) rotate in orbits around the saddle without converging.

In contrast, the iterates of the prediction method fall into the saddle and converge.

When the conventional gradient method (2) is applied to the linear problem (6), the resulting iterations can be written DISPLAYFORM1 When the stepsize α gets small, this behaves like a discretization of the differential equatioṅ DISPLAYFORM2 y = β/αKx (11) whereẋ andẏ denote the derivatives of x and y with respect to time.

The differential equations (10,11) describe a harmonic oscillator.

To see why, differentiate (10) and plug (11) into the result to get a differential equation in x alonë DISPLAYFORM3 We can decompose this into a system of independent single-variable problems by considering the eigenvalue decomposition β/αK T K = U ΣU T .

We now multiply both sides of (12) by U T , and make the change of variables z ← U T x to geẗ z = −Σz.

where Σ is diagonal.

This is the standard equation for undamped harmonic motion, and its solution is z = A cos(Σ 1/2 t + φ), where cos acts entry-wise, and the diagonal matrix A and vector φ are constants that depend only on the initialization.

Changing back into the variable x, we get the solution DISPLAYFORM4 We can see that, for small values of α and β, the non-predictive algorithm (2) approximates an undamped harmonic motion, and the solutions orbit around the saddle without converging.

The prediction step (3) improves convergence because it produces damped harmonic motion that sinks into the saddle point.

When applied to the linearized problem (6), the iterates of the predictive method (3) satisfy DISPLAYFORM5 For small α, this approximates the dynamical systeṁ DISPLAYFORM6 Like before, we differentiate (13) and use FORMULA0 to obtain DISPLAYFORM7 Finally, multiply both sides by U T and perform the change of variables z ← U T x to geẗ DISPLAYFORM8 This equation describes a damped harmonic motion.

The solutions have the form DISPLAYFORM9 .

Changing back to the variable x, we see that the iterates of the original method satisfy DISPLAYFORM10 where A and φ depend on the initialization.

From this analysis, we see that for small constant α the orbits of the lookahead method converge into the saddle point, and the error decays exponentially fast.

A PROOF OF THEOREM 1 DISPLAYFORM11 In the following proofs, we use g u (u, v), g v (u, v) to represent the stochastic approximation of gradients, where DISPLAYFORM12 We show the convergence of the proposed stochastic primal-dual gradients for the primal-dual gap DISPLAYFORM13 We prove the O(1/ √ k) convergence rate in Theorem 1 by using Lemma 1 and Lemma 2, which present the contraction of primal and dual updates, respectively.

DISPLAYFORM14 Proof.

Use primal update in (3), we have DISPLAYFORM15 Take expectation on both side of the equation, substitute with DISPLAYFORM16 Since L(u, v) is convex in u, we have DISPLAYFORM17 (16) is proved by combining FORMULA0 and FORMULA1 .Lemma 2.

Suppose L(u, v) is concave in v and has Lipschitz gradients, DISPLAYFORM18 Proof.

From the dual update in (3), we have DISPLAYFORM19 (23) Take expectation on both sides of the equation, substitute u, v) , and apply DISPLAYFORM20 DISPLAYFORM21 Reorganize FORMULA1 to get DISPLAYFORM22 The right hand side of (25) can be represented as DISPLAYFORM23 where DISPLAYFORM24 DISPLAYFORM25 DISPLAYFORM26 Lipschitz smoothness is used for (31); the prediction step in (3) is used for (32); the primal update in (3) is used for (33); bounded assumptions are used for (35).

DISPLAYFORM27 Combine equations (25, 28, 35 to get36) DISPLAYFORM28 Rearrange the order of (37) to achieve (21).We now present the proof of Theorem 1.Proof.

Combining FORMULA0 and FORMULA0 in the Lemmas, the primal-dual gap DISPLAYFORM29 Accumulate (38) from k = 1, . . .

, l to obtain DISPLAYFORM30 Our finding is summarized in FIG1 .

In addition, FIG5 provides head-to-head comparison of two popular solvers Adam and SGD using the predictive step.

Not surprisingly, the Adam solver shows relatively better performance and convergence even with an additional predictive step.

This also suggests that the default hyper-parameter for the Adam solver can be retained and utilized for training this networks without resorting to any further hyper-parameter tuning (as it is currently in practice).

Experimental details: To evaluate a domain adaptation task, we consider the OFFICE dataset (Saenko et al., 2010) .

OFFICE is a small scale dataset consisting of images collected from three distinct domains: AMAZON, DSLR and WEBCAM.

For such a small scale dataset, it is non-trivial to learn features from images of a single domain.

For instance, consider the largest subset AMAZON, which contains only 2,817 labeled images spread across 31 different categories.

However, one can leverage the power of domain adaptation to improve cross domain accuracy.

We follow the protocol listed in BID11 and the same network architecture is used.

Caffe BID18 is used for implementation.

The training procedure from BID11 is kept intact except for the additional prediction step.

In TAB1 comparisons are drawn with respect to target domain accuracy on three pairs of source-target domain tasks.

The test accuracy is reported at the end of 50,000 training iterations.

Experimental details: The "Adult" dataset from the UCI machine learning repository is used, which consists of census data from ∼ 45, 000 people.

The task is to classify whether a person earns ≥ $50k/year.

The person's gender is chosen to be the sensitive variable.

We binarize all the category attributes, giving us a total of 102 input features per sample.

We randomly split data into 35,000 samples for training, 5000 for validation and 5000 for testing.

The result reported here is an average over five such random splits.

Toy Dataset: To illustrate the advantage of the prediction method, we experiment on a simple GAN architecture with fully connected layers using the toy dataset.

The constructed toy example and its architecture is inspired by the one presented in Metz et al. (2017) .

The two dimensional data is sampled from the mixture of eight Gaussians with their means equally spaced around the unit circle centered at (0, 0).

The standard deviation of each Gaussian is set at 0.01.

The two dimensional latent vector z is sampled from the multivariate Gaussian distribution.

The generator and discriminator networks consist of two fully connected hidden layers, each with 128 hidden units and tanh activations.

The final layer of the generator has linear activation while that of discriminator has sigmoid activation.

The solver optimizes both the discriminator and the generator network using the objective in (4).

We use adam solver with its default parameters (i.e., learning rate = 0.001, β 1 = 0.9, β 2 = 0.999) and with input batch size of 512.

The generated two dimensional samples are plotted in the figure (8) .The straightforward utilization of the adam solver fails to construct all the modes of the underlying dataset while both unrolled GAN and our method are able to produce all the modes.

We further investigate the performance of GAN training algorithms on data sampled from a mixture of a large number of Gaussians.

We use 100 Gaussian modes which are equally spaced around a circle of radius 24 centered at (0, 0).

We retain the same experimental settings as described above and train GAN with two different input batch sizes, a small (64) and a large batch (6144) setting.

The Figure (9 ) plots the generated sample output of GAN trained (for fixed number of epochs) under the above setting using different training algorithms.

Note that for small batch size input, the default as well as the unrolled training for GAN fails to construct actual modes of the underlying dataset.

We hypothesize that this is perhaps due to the batch size, 64, being smaller than the number of input modes (100).

When trained with small batch the GAN observe samples only from few input modes at every iteration.

This causes instability leading to the failure of training algorithms.

This scenario is pertinent to real datasets wherein the number of modes are relatively high compared to input batch size.

In this section we demonstrate the advantage of prediction methods for generating higher resolution images of size 128 x 128.

For this purpose, the state-of-the-art AC-GAN (Odena et al., 2017) architecture is considered and conditionally learned using images of all 1000 classes from Imagenet dataset.

We have used the publicly available code for AC-GAN and all the parameter were set to it default as in Odena et al. (2017) .

The figure 14 plots the inception score measured at every training epoch of AC-GAN model with and without prediction.

The score is averaged over five independent runs.

From the figure, it is clear that even at higher resolution with large number of classes the prediction method is stable and aids in speeding up the training.

<|TLDR|>

@highlight

We present a simple modification to the alternating SGD method, called a prediction step, that improves the stability of adversarial networks.