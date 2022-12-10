Deep neural networks have been shown to perform well in many classical machine learning problems, especially in image classification tasks.

However, researchers have found that neural networks can be easily fooled, and they are surprisingly sensitive to small perturbations imperceptible to humans.

Carefully crafted input images (adversarial examples) can force a well-trained neural network to provide arbitrary outputs.

Including adversarial examples during training is a popular defense mechanism against adversarial attacks.

In this paper we propose a new defensive mechanism under the generative adversarial network~(GAN) framework.

We model the adversarial noise using a generative network, trained jointly with a classification discriminative network as a minimax game.

We show empirically that our adversarial network approach works well against black box attacks, with performance on par with state-of-art methods such as ensemble adversarial training and adversarial training with projected gradient descent.

Deep neural networks have been successfully applied to a variety of tasks, including image classification BID12 , speech recognition BID8 , and human-level playing of video games through deep reinforcement learning BID20 .

However, BID29 showed that convolutional neural networks (CNN) are extremely sensitive to carefully crafted small perturbations added to the input images.

Since then, many adversarial examples generating methods have been proposed, including Jacobian based saliency map attack (JSMA) BID23 , projected gradient descent (PGD) attack , and C&W's attack BID3 .

In general, there are two types of attack models: white box attack and black box attack.

Attackers in white box attack model have complete knowledge of the target network, including network's architecture and parameters.

Whereas in black box attacks, attackers only have partial or no information on the target network BID25 .Various defensive methods have been proposed to mitigate the effect of the adversarial examples.

Adversarial training which augments the training set with adversarial examples shows good defensive performance in terms of white box attacks BID13 .

Apart from adversarial training, there are many other defensive approaches including defensive distillation BID24 , using randomization at inference time BID33 , and thermometer encoding BID2 , etc.

In this paper, we propose a defensive method based on generative adversarial network (GAN) BID6 .

Instead of using the generative network to generate samples that can fool the discriminative network as real data, we train the generative network to generate (additive) adversarial noise that can fool the discriminative network into misclassifying the input image.

This allows flexible modeling of the adversarial noise by the generative network, which can take in the original image or a random vector or even the class label to create different types of noise.

The discriminative networks used in our approach are just the usual neural networks designed for their specific classification tasks.

The purpose of the discriminative network is to classify both clean and adversarial example with correct label, while the generative network aims to generate powerful perturbations to fool the discriminative network.

This approach is simple and it directly uses the minimax game concept employed by GAN.

Our main contributions include:• We show that our adversarial network approach can produce neural networks that are robust towards black box attacks.

In the experiments they show similar, and in some cases better, performance when compared to state-of-art defense methods such as ensemble adversarial training BID30 and adversarial training with projected gradient descent .

To our best knowledge we are also the first to study the joint training of a generative attack network and a discriminative network.• We study the effectiveness of different generative networks in attacking a trained discriminative network, and show that a variety of generative networks, including those taking in random noise or labels as inputs, can be effective in attacks.

We also show that training against these generative networks can provide robustness against different attacks.

The rest of the paper is organized as follows.

In Section 2, related works including multiple attack and defense methods are discussed.

Section 3 presents our defensive method in details.

Experimental results are shown in Section 4, with conclusions of the paper in Section 5.

In this section, we briefly review the attack and defense methods in neural network training.

Given a neural network model D θ parameterized by θ trained for classification, an input image x ∈ R d and its label y, we want to find a small adversarial perturbation ∆x such that x + ∆x is not classified as y. The minimum norm solution ∆x can be described as: DISPLAYFORM0 where arg max D θ (x) gives the predicted class for input x. BID29 introduced the first method to generate adversarial examples by considering the following optimization problem, DISPLAYFORM1 where L is a distance function measuring the closeness of the output D θ (x+z) with some targetŷ = y. The objective is minimized using box-constrained L-BFGS.

BID7 introduced the fast gradient sign method (FGS) to generate adversarial examples in one step, which can be represented as ∆x = · sign (∇ x l(D θ (x), y)), where l is the cross-entropy loss used in neural networks training.

argues with strong evidence that projected gradient descent (PGD), which can be viewed as an iterative version of the fast gradient sign method, is the strongest attack using only first-order gradient information.

BID25 presented a Jacobian-based saliency-map attack (J-BSMA) model to generate adversarial examples by changing a small number of pixels.

BID21 showed that there exist a single/universal small image perturbation that fools all natural images.

BID25 introduced the first demonstration of black-box attacks against neural network classifiers.

The adversary has no information about the architecture and parameters of the neural networks, and does not have access to large training dataset.

In order to mitigate the effect of the generated adversarial examples, various defensive methods have been proposed.

BID24 introduced distillation as a defense to adversarial examples.

BID16 introduced a foveation-based mechanism to alleviate adversarial examples.

The idea of adversarial training was first proposed by BID29 .

The effect of adversarial examples can be reduced through explicitly training the model with both original and perturbed adversarial images.

Adversarial training can be viewed as a minimax game, DISPLAYFORM0 Published as a conference paper at ICLR 2019 The inner maximization requires a separate oracle for generating the perturbations ∆x.

FGS is a common method for generating the adversarial perturbations ∆x due to its speed. advocates the use of PGD in generating adversarial examples.

Moreover, a cascade adversarial training is presented in BID22 , which injects adversarial examples from an already defended network added with adversarial images from the network being trained.

There are a few recent works on using GANs for generating and defending against adversarial examples.

BID27 and BID10 use GAN for defense by learning the manifold of input distribution with GAN, and then project any input examples onto this learned manifold before classification to filter out any potential adversarial noise.

Our approach is more direct because we do not learn the input distribution and no input denoising is involved.

Both BID1 and BID32 train neural networks to generate adversarial examples by maximizing the loss over a fixed pre-trained discriminative network.

They show that they can train neural networks to effectively attack undefended discriminative networks while ensuring the generated adversarial examples look similar to the original samples.

Our work is different from these because instead of having a fixed discriminative network, we co-train the discriminative network together with the adversarial generative network in a minimax game.

BID32 also train a second discriminative network as in typical GANs, but their discriminative network is used for ensuring the generated images look like the original samples, and not for classification.

BID14 also considered the use of GAN to train robust discriminative networks.

However, the inputs to their generative network is the gradient of the discriminative network with respect to the input image x, not just the image x as in our current work.

This causes complex dependence of the gradient of the generative network parameters to the discriminative network parameters, and makes the parameter updates for the generative network more complicated.

Also there is no single minimax objective that they are solving for in their work; the update rules for the discriminative and generative networks optimize related but different objectives.

In generative adversarial networks (GAN) BID6 , the goal is to learn a generative neural network that can model a distribution of unlabeled training examples.

The generative network transforms a random input vector into an output that is similar to the training examples, and there is a separate discriminative network that tries to distinguish the real training examples against samples generated by the generative network.

The generative and discriminative networks are trained jointly with gradient descent, and at equilibrium we want the samples from the generative network to be indistinguishable from the real training data by the discriminative network, i.e., the discriminative network does no better than doing a random coin flip.

We adopt the GAN approach in generating adversarial noise for a discriminative model to train against.

This approach has already been hinted at in BID30 , but they decided to train against a static set of adversarial models instead of training against a generative noise network that can dynamically adapt in a truly GAN fashion.

In this work we show that this idea can be carried out fruitfully to train robust discriminative neural networks.

Given an input x with correct label y, from the viewpoint of the adversary we want to find additive noise ∆x such that x + ∆x will be incorrectly classified by the discriminative neural network to some other labelsŷ = y. We model this additive noise as G(x), where G is a generative neural network that generates instance specific noise based on the input x and is the scaling factor that controls the size of the noise.

Notice that unlike white box attack methods such as FGS or PGD, once trained G does not need to know the parameters of the discriminative network that it is attacking.

G can also take in other inputs to generate adversarial noise, e.g., Gaussian random vector z ∈ R d as in typical GAN, or even the class label y.

For simplicity we assume G takes in x as input in the descriptions below.

Suppose we have a training set {(x 1 , y 1 ), . . .

, (x n , y n )} of image-label pairs.

Let D θ be the discriminator network (for classification) parameterized by θ, and G φ be the generator network parameterized by φ.

We want to solve the following minimax game between D θ and G φ : DISPLAYFORM0 where l is the cross-entropy loss, λ is the trade-off parameter between minimizing the loss on normal examples versus minimizing the loss on the adversarial examples, and is the magnitude of the noise.

See FIG0 for an illustration of the model.

In this work we focus on perturbations based on ∞ norm.

This can be achieved easily by adding a tanh layer as the final layer of the generator network G φ , which normalizes the output to the range of [−1, 1].

Perturbations based on 1 or 2 norms can be accommodated by having the appropriate normalization layers in the final layer of G φ .We now explain the intuition of our approach.

Ideally, we would like to find a solution θ that has small risk on clean examples DISPLAYFORM1 and also small risk on the adversarial examples under maximum perturbation of size DISPLAYFORM2 However, except for simple datasets like MNIST, there are usually fairly large differences between the solutions of R(θ) and solutions of R adv (θ) under the same model class D θ .

Optimizing for the risk under white box attacks R adv (θ) involves tradeoff on the risk on clean data R(θ).

Note that R adv (θ) represent the risk under white box attacks, since we are free to choose the perturbation ∆x with knowledge of θ.

This can be approximated using the powerful PGD attack.

Instead of allowing the perturbations ∆x to be completely free, we model the adversary as a neural network G φ with finite capacity DISPLAYFORM3 Here the adversarial noise G φ (x i ) is not allowed to directly depend on the discriminative network parameters θ.

Also, the generative network parameter φ is shared across all examples, not computed per example like ∆x.

We believe this is closer to the situation of defending against black box attacks, when the adversary does not know the discriminator network parameters.

However, we still want G φ to be expressive enough to represent powerful attacks, so that D θ has a good adversary to train against.

Previous work BID32 BID1 show that there are powerful classes of G φ that can attack trained classifiers D θ effectively.

In traditional GANs we are most interested in the distributions learned by the generative network.

The discriminative network is a helper that drives the training, but can be discarded afterwards.

In our setting we are interested in both the discriminative network and the generative network.

The generative network in our formulation can give us a powerful adversary for attacking, while the discriminative network can give us a robust classifier that can defend against adversarial noise.

The stability and convergence of GAN training is still an area of active research BID19 .

In this paper we adopt gradient regularization BID18 to stabilize the gradient descent/ascent training.

Denote the minimax objective in Eq. 4 as F (θ, φ).

With the generative network parameter fixed at φ k , instead of minimizing the usual objective F (θ, φ k ) to update θ for the discriminator network, we instead try to minimize the regularized objective DISPLAYFORM0 where γ is the regularization parameter for gradient regularization.

Minimizing the gradient norm ∇ φ F (θ, φ k ) 2 jointly makes sure that the norm of the gradient for φ at φ k does not grow when we update θ to reduce the objective F (θ, φ k ).

This is important because if the gradient norm DISPLAYFORM1 2 becomes large after an update of θ, it is easy to update φ to make the objective large again, leading to zigzagging behaviour and slow convergence.

Note that the gradient norm term is zero at a saddle point according to the first-order optimality conditions, so the regularizer does not change the set of solutions.

With these we update θ using SGD with step size η D : DISPLAYFORM2 can be computed with double backpropagation provided by packages like Tensorflow/PyTorch, but we find it faster to compute it with finite difference approximation.

Recall that for a function f (x) with gradient g(x) and Hessian H(x), the Hessian-vector product H(x)v can be approximated by (g(x + hv) − g(x))/h for small h (Pearlmutter, 1994).

Therefore we approximate: DISPLAYFORM3 where v = ∇ φ F (θ l , φ k ).

Note that φ k + hv is exactly a gradient step for generative network G φ .

Setting h to be too small can lead to numerical instability.

We therefore correlate h with the gradient step size and set h = η G /10 to capture the curvature at the scale of the gradient ascent algorithm.

We update the generative network parameters φ with using (stochastic) gradient ascent.

With the discriminative network parameters fixed at θ l and step size η G , we update: DISPLAYFORM4 We do not add a gradient regularization term for φ, since empirically we find that adding gradient regularization to θ is sufficient to stabilize the training.

In the experiments we train both the discriminative network and generative network from scratch with random weight initializations.

We do not need to pre-train the discriminative network with clean examples, or the generative network against some fixed discriminative networks, to arrive at good saddle point solutions.

In our experiments we find that the discriminative networks D θ we use tend to overpower the generative network G φ if we just perform simultaneous parameter updates to both networks.

This can lead to saddle point solutions where it seems G φ cannot be improved locally against D θ , but in reality can be made more powerful by just running more gradient steps on φ.

In other words we want the region around the saddle point solution to be relatively flat for G φ .

To make the generative network more powerful so that the discriminative network has a good adversary to train against, we adopt the following strategy.

For each update of θ for D θ , we perform multiple gradient steps on φ using the same mini-batch.

This allows the generative network to learn to map the inputs in the mini-batch to adversarial noises with high loss directly, compared to running multiple gradient steps on different mini-batches.

In the experiments we run 5 gradient steps on each mini-batch.

We fix the tradeoff parameter λ (Eq. 4) over loss on clean examples and adversarial loss at 1.

We also fix the gradient regularization parameter γ (Eq. 8) at 0.01, which works well for different datasets.

We implemented our adversarial network approach using Tensorflow BID0 , with the experiments run on several machines each with 4 GTX1080 Ti GPUs.

In addition to our adversarial networks, we also train standard undefended models and models trained with adversarial training using PGD for comparison.

For attacks we focus on the commonly used fast gradient sign (FGS) method, and the more powerful projected gradient descent (PGD) method.

For the fast gradient sign (FGS) attack, we compute the adversarial image bŷ DISPLAYFORM0 where Proj X projects onto the feasible range of rescaled pixel values X (e.g., [-1,1] ).For the projected gradient descent (PGD) attack, we iterate the fast gradient sign attack multiple times with projection, with random initialization near the starting point neighbourhood.

DISPLAYFORM1 where u ∈ R d is a uniform random vector in [−1, 1] d , δ is the step size, and B ∞ (x i ) is an ∞ ball centered around the input x i with radius .

In the experiments we set δ to be a quarter of the perturbation , i.e., /4, and the number of PGD steps k to be 10.

We adopt exactly the same PGD attack procedure when generating adversarial examples in PGD adversarial training.

Our implementation is available at https://github.com/whxbergkamp/RobustDL_GAN.

For MNIST the inputs are black and white images of digits of size 28x28 with pixel values scaled between 0 and 1.

We rescale the inputs to the range of [-1,1] .

Following previous work (Kannan et al., 2018), we study perturbations of = 0.3 (in the original scale of [0,1]).

We use a simple convolutional neural network similar to LeNet5 as our discriminator networks for all training methods.

For our adversarial approach we use an encoder-decoder network for the generator.

See Model D1 and Model G0 in the Appendix for the details of these networks.

We use SGD with learning rate of η D = 0.01 and momentum 0.9, batch size of 64, and run for 200k iterations for all the discriminative networks.

The learning rates are decreased by a factor of 10 after 100k iterations.

We use SGD with a fixed learning rate η G = 0.01 with momentum 0.9 for the generative network.

We use weight decay of 1E-4 for standard and adversarial PGD training, and 1E-5 for our adversarial network approach (for both D θ and G φ ).

For this dataset we find that we can improve the robustness of D θ by running more updates on G φ , so we run 5 updates on G φ (each update contains 5 gradient steps described in Section 3.2 ) for each update on D θ .Table 1(left) shows the white box attack accuracies of different models, under perturbations of = 0.3 for input pixel values between 0 and 1.

Adversarial training with PGD performs best under white box attacks.

Its accuracies stay above 90% under FGS and PGD attacks.

Our adversarial network model performs much better than the undefended standard training model, but there is still a gap in accuracies compared to the PGD model.

However, the PGD model has a small but noticeable drop in accuracy on clean examples compared to the standard model and adversarial network model.

Table 1(right) shows the black box attack accuracies of different models.

We generate the black box attack images by running the FGS and PGD attacks on surrogate models A', B' and C'.

These surrogate models are trained in the same way as their counterparts (standard -A, PGD -B, adversarial network -C) with the same network architecture, but using a different random seed.

We notice that the black box attacks tend to be the most effective on models trained with the same method (A' on A, B' on B, and C' on C).

Although adversarial PGD beats our adversarial network approach on white box attacks, they have comparable performance on these black box attacks.

Interestingly, the adversarial examples from adversarial PGD (B') and adversarial networks (C') do not transfer well to the undefended standard model.

The undefended model still have accuracies between 85-95%.

For the Street View House Number(SVHN) data, we use the original training set, augmented with 80k randomly sampled images from the extra set as our training data.

The test set remains the same and we do not perform any preprocessing on the images apart from scaling it to the range of [-1,1] .

We study perturbations of size = 0.05 (in the range of [0, 1] Table 2 : Classification accuracies under white box and black box attacks on SVHN ( = 0.05) et al., 2016) adapted to 32x32 images as our discriminative networks.

For the generator in our adversarial network we use an encoder-decoder network based on residual blocks from ResNet.

See Model D2 and Model G1 in the Appendix for details.

For the discriminative networks we use SGD with learning rate of η D = 0.01 and momentum 0.9, batch size of 64, weight decay of 1E-4 and run for 100k iterations, and then decrease the learning rate to 0.001 and run for another 100k iterations.

For the generative network we use SGD with a fixed learning rate of η G = 0.01 and momentum 0.9, and use weight decay of 1E-4.Table 2(left) shows the white box attack accuracies of the models.

Adversarial PGD performs best against PGD attacks, but has lower accuracies on clean data and against FGS attacks, since it is difficult to optimize over all three objectives with finite network capacity.

Our adversarial network approach has the best accuracies on clean data and against FGS attacks, and also improved accuracies against PGD over standard training.

Table 2(right) shows the black box attack accuracies of the models.

As before A', B', C' are networks trained in the same ways as their counterparts, but with a different random seed.

We can see that the adversarial network approach performs best across most attacks, except the PGD attack from its own copy C'.

It is also interesting to note that for this dataset, adversarial examples generated from the adversarial PGD model B' have the strongest attack power across all models.

In the other two datasets, adversarial examples generated from a model are usually most effective against their counterparts that are trained in the same way.

For CIFAR10 we scale the 32x32 inputs to the range of [-1,1] .

We also perform data augmentation by randomly padding and cropping the images by at most 4 pixels, and randomly flipping the images left to right.

In this experiment we use the same discriminative and generative networks as in SVHN.

We study perturbations of size = 8/256.

We train the discriminative networks with batch size of 64, and learning rate of η D = 0.1 for 100k iterations, and decrease learning rate to 0.01 for another 100k iterations.

We use Adam with learning rate η G = 0.002, β 1 = 0.5, β 2 = 0.999 for the generative network.

We use weight decay 1E-4 for standard training, and 1E-5 for adversarial PGD and our adversarial networks.

TAB1 (left) shows the white box accuracies of different models under attack with = 8/256.

The PGD model has the best accuracy under PGD attack, but suffer a considerably lower accuracy on clean data and FGS attack.

One reason for this is that it is difficult to balance between the objective of getting good accuracies on clean examples and good accuracies on very hard PGD attack adversarial examples with a discriminative network of limited capacity.

Our adversarial model is able to keep up with the standard model in terms of accuracies on clean examples, and improve upon it on accuracies against FGS and PGD attacks.

Table 4 : Classification accuracies under white box and black box attacks on ensemble adversarial training and adversarial networks on different datasets B', and offers the smallest drop in accuracies in general.

But its overall results are not the best since it suffers from the disadvantage of having a lower baseline accuracy on clean examples.

We have also performed experiments on CIFAR10 using a wider version of ResNet BID34 by multiplying the number of filters by 10 in each of the convolutional layers.

These wider version of ResNets have higher accuracies, but the relative strengths of the methods are similar to those presented here.

In addition we have experiments on CIFAR100, and the results are qualitatively similar to CIFAR10.

All these results are presented in the Appendix due to space restrictions.

We also compare against a version of ensemble adversarial training BID30 on the above 3 datasets.

Ensemble adversarial training works by including adversarial examples generated from static pre-trained models to enlarge the training set, and then train a new model on top of it.

The quality of solutions depends on the type of adversarial examples included.

Here we construct adversarial examples by running FGS (Eq. 10) and PGD (Eq. 11) on an undefended model, i.e., FGS(A) and PGD(A) in the previous tables.

Here for FGS we substitute the target label y with the most likely class arg max D θ (x) to avoid the problem of label leakage.

Following BID30 we also include another attack using the least likely class: DISPLAYFORM0 where y LL = arg min D θ (x i ) is the least likely class.

We include all these adversarial examples together with the original clean data for training.

We use the same perturbations as in the respective experiments above.

Table 4 shows the results comparing ensemble adversarial training (EAT) with our adversarial networks approach.

On MNIST, adversarial networks is better on white box attacks and also better on all black box attacks using models trained with standard training(A'), adversarial PGD(B'), and our adversarial networks approach(C') with different random seeds.

On SVHN and CIFAR10 adversarial networks is better on white box attacks, and both methods have wins and losses on the black box attacks, depending on the attacks used.

In general adversarial networks seem to have better white box attack accuracies since they are trained dynamically with a varying adversary.

The black box accuracies depend a lot on the dataset and the type of attacks used.

There is no definitive conclusion on whether training against a static set of adversaries as in EAT or training against a dynamically adjusting adversary as in adversarial networks is a better approach against black box attacks.

This is an interesting question requiring further research.

Table 6 : Classification accuracies under white box and black box attacks on CIFAR10 for adversarial networks trained with different generative adversaries ( = 8/256)

We also did a more in-depth study on the generative network with CIFAR10.

We want to understand how the capacity of the generative network affects the quality of saddle point solution, and also the power of the generative networks themselves as adversarial attack methods.

First we study the ability of the generative networks to learn to attack a fixed undefended discriminative network.

The architectures of the generative networks (G1, G2, G3) are described in the Appendix.

Here we study a narrow (G1, k = 8) and a wide version (G1, k = 64) of autoencoder networks using the input images as inputs, and also decoder networks G(z) using random Gaussian vectors z ∈ R d (G2) or networks G(y) using the labels y (G3) as inputs.

We run SGD for 200k iterations with step size 0.01 and momentum of 0.9, and use weight decay of 1E-5.

We report test accuracies on the original discriminator after attacks.

From TAB4 the wide autoencoder is more powerful than the narrow autoencoder in attacking the undefended discriminator network across different models.

As a white-box attack method, the wide autoencoder is close to PGD in terms of attack power (6.08% vs 1.32% in TAB1 (left)) on the undefended model.

As a black-box attack method on the undefended model A', it works even better than PGD (14.74% vs 22.88% in TAB1 (right)).

However, on defended models trained with PGD and our adversarial network approach the trained generator networks do not have much effect.

PGD is especially robust with very small drops in accuracies against these attacks.

It is interesting that generator network G(z) with random Gaussian z as inputs and G(y) with label as input works well against undefended models A and A', reducing the accuracies by more than 30%, even though they are not as effective as using the image as input.

G(z) is essentially a distribution of random adversarial noise that we add to the image without knowing the image or label.

G(y) is a generator network with many parameters, but after training it is essentially a set of 10 class conditional 32x32x3 filters.

We have also performed similar experiments on attacking models trained with adversarial PGD and our adversarial networks using the above generative networks.

The results are included in the Appendix due to space restrictions.

We also co-train these different generative networks with our discriminative network (D2) on CI-FAR10.

The results are shown in Table 6 .

It is slightly surprising that they all produce very similar performance in terms of white box and black box attacks, even as they have different attack powers against undefended networks.

The generative networks do have very similar decoder portions, and this could be a reason why they all converge to saddle points of similar quality.

In the experiments above we see that adversarial PGD training usually works best on white box attacks, but there is a tradeoff between accuracies on clean data against accuracies on adversarial examples due to finite model capacity.

We can try to use models with larger capacity, but there is always a tradeoff between the two, especially for larger perturbations .

There are some recent works that indicate training for standard accuracy and training for adversarial accuracy (e.g., with PGD) are two fairly different problems .

Examples generated from PGD are particularly difficult to train against.

This makes adversarial PGD training disadvantaged in many black box attack situations, when compared with models trained with weaker adversaries, e.g., ensemble adversarial training and our adversarial networks method.

We have also observed in the experiments that for black box attacks, the most effective adversarial examples are usually those constructed from models trained using the same method but with different random seed.

This suggests hiding the knowledge of the training method from the attacker could be an important factor in defending against black box attacks.

Defending against black box attacks is closely related to the question of the transferability of adversarial examples.

Although there are some previous works exploring this question BID15 , the underlying factors affecting transferability are still not well understood.

In our experimentation with the architectures of the discriminative and generative networks, the choice of architectures of G φ does not seem to have a big effect on the quality of solution.

The dynamics of training, such as the step size used and the number of iterations to run for each network during gradient descent/ascent, seem to have a bigger effect on the saddle point solution quality than the network architecture.

It would be interesting to find classes of generative network architectures that lead to substantially different saddle points when trained against a particular discriminative network architecture.

Also, recent works have shown that there are connected flat regions in the minima of neural network loss landscapes BID5 BID4 .

We believe that the same might hold true for GANs, and it would be interesting to explore how the training dynamics can lead to different GAN solutions that might have different robustness properties.

Our approach can be extended with multiple discriminative networks playing against multiple generative networks.

It can also be combined with ensemble adversarial training, where some adversarial examples come from static pre-trained models, while some other come from dynamically adjusting generative networks.

We have proposed an adversarial network approach to learning discriminative neural networks that are robust to adversarial noise, especially under black box attacks.

For future work we are interested in extending the experiments to ImageNet, and exploring the choice of architectures of the discriminative and generative networks and their interaction.

generative networks used in this paper.

G0 and G1 are encoder-decoder networks, while G2 and G3 are decoder networks using a random vector and a one-hot encoding of the label respectively.

The generative networks are parameterized by a factor k determining the number of filters used (width of network).

As default we use k = 64, and k = 16 for networks using labels as inputs.

EXTRA RESULTS ON CIFAR100 AND WIDE RESNET ON CIFAR10The discriminative and generative networks in our CIFAR100 experiment have the same network architecture as the CIFAR10 experiment, except that the output layer dimension of the D network is 100 other than 10 in CIFAR10.

We use learning rate of 0.1 for the first 100k iterations, and 0.01 for another 100k iterations.

The batch size is 64 and weight decay is 1E-5.

TAB8 gives the results on CIFAR10 using a wider version of Resnet (Model D2), by multiplying the number of filters in each convolutional layer by a factor of 10.

Some of the previous works in the literature use models of larger capacity for training adversarially robust models, so we perform experiments on these large capacity models here.

First the accuracies increase across the board with larger capacity models.

The accuracy gap on clean data between adversarial PGD and standard training still exists, but now there is also a small accuracy gap between our adversarial network approach and standard training.

For the rest of the white box and black accuracies the story is similar, the models are weakest against attacks trained with the same method but with a different random seed.

Our adversarial network approach has very good performance across different attacks, even as it is not always the winner for each individual attack.

TAB10 gives the results of Wide ResNet on CIFAR100, and the results are qualitatively similar.

Following Section 4.5, we run extra experiments on using different generative networks to attack networks trained with adversarial PGD and our adversarial networks approach, in addition to the white box black box training method\attack No Noise FGS PGD FGS(A') PGD(A') FGS(B') PGD(B') FGS(C') PGD(C') standard(A) TAB0 : Attack performance of various generator networks against our adversarial network in terms of test accuracies.

First column is the accuracy on the discriminative model D θ that the generative attacker G φ is trained on (similar to white box attacks).

The next three columns are the attack accuracies on other models by the learned G φ (similar to black box attacks) undefended network in Section 4.5.

TAB0 shows the results of various generative networks in attacking a network trained with adversarial PGD.

The adversarial PGD network is very robust, and the generative networks can at most reduce the accuracy by 5%.

Interestingly, the strongest attack come from the more restrictive generative network using only the label as input.

It is also the most successful in transferring to other networks.

However, since the adversarial PGD network is so robust, none of the generative networks can learn much from it in generating adversarial examples.

TAB0 shows the results of various generative networks in attacking our adversarial network.

Our adversarial network is not as robust as adversarial PGD under white box attack, and the autoencoder(64 filters) network can reduce its accuracy from over 90% to 53%.

Nonetheless, it is still much more robust than the undefended network.

Interestingly, in addition to transferring well to the adversarial network trained with a different random seed (C'), the autoencoder(64 filters) network also transfers well to the undefended network, reducing its accuracy to 46%.

@highlight

Jointly train an adversarial noise generating network with a classification network to provide better robustness to adversarial attacks.

@highlight

A GAN solution for deep models of classification, faced to white and black box attacks, that produces robust models. 

@highlight

The paper proposes a defensive mechanism against adversarial attacks using GANs with generated perturbations used as adversarial examples and a discriminator used to distinguish between them