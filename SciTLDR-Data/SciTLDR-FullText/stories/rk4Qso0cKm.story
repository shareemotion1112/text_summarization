We present a new algorithm to train a robust neural network against adversarial attacks.

Our algorithm is motivated by the following two ideas.

First, although recent work has demonstrated that fusing randomness can improve the robustness of neural networks (Liu 2017), we noticed that adding noise blindly to all the layers is not the optimal way to incorporate randomness.

Instead, we model randomness under the framework of Bayesian Neural Network (BNN) to formally learn the posterior distribution of models in a scalable way.

Second, we formulate the mini-max problem in BNN to learn the best model distribution under adversarial attacks, leading to an adversarial-trained Bayesian neural net.

Experiment results demonstrate that the proposed algorithm achieves state-of-the-art performance under strong attacks.

On CIFAR-10 with VGG network, our model leads to 14% accuracy improvement compared with adversarial training (Madry 2017) and random self-ensemble (Liu, 2017) under PGD attack with 0.035 distortion, and the gap becomes even larger on a subset of ImageNet.

Deep neural networks have demonstrated state-of-the-art performances on many difficult machine learning tasks.

Despite the fundamental breakthroughs in various tasks, deep neural networks have been shown to be utterly vulnerable to adversarial attacks BID32 BID11 .

Carefully crafted perturbations can be added to the inputs of the targeted model to drive the performances of deep neural networks to chance-level.

In the context of image classification, these perturbations are imperceptible to human eyes but can change the prediction of the classification model to the wrong class.

Algorithms seek to find such perturbations are denoted as adversarial attacks BID5 BID4 BID28 , and some attacks are still effective in the physical world BID17 BID9 .

The inherent weakness of lacking robustness to adversarial examples for deep neural networks brings out security concerns, especially for security-sensitive applications which require strong reliability.

To defend from adversarial examples and improve the robustness of neural networks, many algorithms have been recently proposed BID27 BID37 BID17 BID12 .

Among them, there are two lines of work showing effective results on medium-sized data (e.g., CIFAR-10).

The first line of work uses adversarial training to improve robustness, and the recent algorithm proposed in BID25 has been recognized as one of the most successful defenses, as shown in .

The second line of work adds stochastic components in the neural network to hide gradient information from attackers.

In the black-box setting, stochastic outputs can significantly increase query counts for attacks using finite-difference techniques BID5 , and even in the white-box setting the recent Random Self-Ensemble (RSE) approach proposed by BID23 achieves similar performance to Madry's adversarial training algorithm.

In this paper, we propose a new defense algorithm called Adv-BNN.

The idea is to combine adversarial training and Bayesian network, although trying BNNs in adversarial attacks is not new (e.g. BID20 BID10 BID30 ), and very recently BID36 also tried to combine Bayesian learning with adversarial training, this is the first time we scale the problem to complex data and our approach achieves better robustness than previous defense methods.

The contributions of this paper can be summarized below:• Instead of adding randomness to the input of each layer (as what has been done in RSE), we directly assume all the weights in the network are stochastic and conduct training with techniques commonly used in Bayesian Neural Network (BNN).• We propose a new mini-max formulation to combine adversarial training with BNN, and show the problem can be solved by alternating between projected gradient descent and SGD.• We test the proposed Adv-BNN approach on CIFAR10, STL10 and ImageNet143 datasets, and show significant improvement over previous approaches including RSE and adversarial training.

Notations A neural network parameterized by weights w ∈ R d is denoted by f (x; w), where x ∈ R p is an input example and y is the corresponding label, the training/testing dataset is D tr/te with size N tr/te respectively.

When necessary, we abuse D tr/te to define the empirical distribu- DISPLAYFORM0 δ(x i )δ(y i ), where δ(·) is the Dirac delta function.

x o represents the original input and x adv denotes the adversarial example.

The loss function is represented as f (x i ; w), y i , where i is the index of the data point.

Our approach works for any loss but we consider the cross-entropy loss in all the experiments.

The adversarial perturbation is denoted as ξ ∈ R p , and adversarial example is generated by x adv = x o + ξ.

In this paper, we focus on the attack under norm constraint BID25 , so that ξ ≤ γ.

In order to align with the previous works, in the experiments we set the norm to · ∞ .

The Hadamard product is denoted as .

In this section, we summarize related works on adversarial attack and defense.

Attack: Most algorithms generate adversarial examples based on the gradient of loss function with respect to the inputs.

For example, FGSM BID11 perturbs an example by the sign of gradient, and use a step size to control the ∞ norm of perturbation.

BID17 proposes to run multiple iterations of FGSM.

More recently, C&W attack BID3 formally poses attack as an optimization problem, and applies a gradient-based iterative solver to get an adversarial example.

Both C&W attack and PGD attack BID25 have been frequently used to benchmark the defense algorithms due to their effectiveness .

Throughout, we take the PGD attack as an example, largely following BID25 .The goal of PGD attack is to find adversarial examples in a γ-ball, which can be naturally formulated as the following objective function: DISPLAYFORM0 Starting from x 0 = x o , PGD attack conducts projected gradient descent iteratively to update the adversarial example: DISPLAYFORM1 where Π γ is the projection to the set {x| x−x o ∞ ≤ γ}. Although multi-step PGD iterations may not necessarily return the optimal adversarial examples, we decided to apply it in our experiments, following the previous work of BID25 ).

An advantage of PGD attack over C&W attack is that it gives us a direct control of distortion by changing γ, while in C&W attack we can only do this indirectly via tuning the regularizer.

Since we are dealing with networks with random weights, we elaborate more on which strategy should attackers take to increase their success rate, and the details can be found in .

In random neural networks, an attacker seeks a universal distortion ξ that cheats a majority of realizations of the random weights.

This can be achieved by maximizing the loss expectation DISPLAYFORM2 Here the model weights w are considered as random vector following certain distributions.

In fact, solving (3) to a saddle point can be done easily by performing multi-step (projected) SGD updates.

This is done inherently in some iterative attacks such as C&W or PGD discussed above, where the only difference is that we sample new weights w at each iteration.

Defense: There are a large variety of defense methods proposed in recent years, e.g. denoiser based HGD BID21 and randomized image preprocessing BID34 .

Readers can find more from BID18 .

Below we select two representative ones that turn out to be effective to white box attacks.

They are the major baselines in our experiments.

The first example is the adversarial training BID32 BID11 .

It is essentially a data augmentation method, which trains the deep neural networks on adversarial examples until the loss converges.

Instead of searching for adversarial examples and adding them into the training data, BID25 proposed to incorporate the adversarial search inside the training process, by solving the following robust optimization problem: DISPLAYFORM3 where D tr is the training data distribution.

The above problem is approximately solved by generating adversarial examples using PGD attack and then minimizing the classification loss of the adversarial example.

In this paper, we propose to incorporate adversarial training in Bayesian neural network to achieve better robustness.

The other example is RSE BID23 , in this algorithm the authors proposed a "noise layer", which fuses input features with Gaussian noise.

They show empirically that an ensemble of models can increase the robustness of deep neural networks.

Besides, their method can generate an infinite number of models on-the-fly without any additional memory cost.

The noise layer is applied in both training and testing phases, so the prediction accuracy will not be largely affected.

Our algorithm is different from RSE in two folds: 1) We add noise to each weight instead of input or hidden feature, and formally model it as a BNN.

2) We incorporate adversarial training to further improve the performance.

The idea of BNN is illustrated in FIG0 .

Given the observable random variables (x, y), we aim to estimate the distributions of hidden variables w. In our case, the observable random variables correspond to the features x and labels y, and we are interested in the posterior over the weights p(w|x, y) given the prior p(w).

However, the exact solution of posterior is often intractable: notice that p(w|x, y) =

-but the denominator involves a high dimensional integral BID1 , hence the conditional probabilities are hard to compute.

To speedup inference, we generally have two approaches-we can either sample w ∼ p(w|x, y) efficiently without knowing the closed-form formula through, for example, Stochastic Gradient Langevin Dynamics (SGLD) BID33 ), or we can approximate the true posterior p(w|x, y) by a parametric distribution q θ (w), where the unknown parameter θ is estimated by minimizing KL q θ (w) p(w|x, y) over θ.

For neural network, the exact form of KL-divergence can be unobtainable, but we can easily find an unbiased gradient estimator of it using backward propagation, namely Bayes by Backprop BID2 .Despite that both methods are widely used and analyzed in-depth, they have some obvious shortcomings, making high dimensional Bayesian inference remain to be an open problem.

For SGLD and its extension (e.g. BID19 ), since the algorithms are essentially SGD updates with extra Gaussian noise, they are very easy to implement.

However, they can only get one sample w ∼ p(w|x, y) in each minibatch iteration at the cost of one forward-backward propagation, thus not efficient enough for fast inference.

In addition, as the step size η t in SGLD decreases, the samples become more and more correlated so that one needs to generate many samples in order to control the variance.

Conversely, the variational inference method is efficient to generate samples since we know the approximated posterior q θ (w) once we minimized the KL-divergence.

The problem is that for simplicity we often assume the approximation q θ to be a fully factorized Gaussian distribution: DISPLAYFORM0 Although our assumption (5) has a simple form, it inherits the main drawback from mean-field approximation.

When the ground truth posterior has significant correlation between variables, the approximation in (5) will have a large deviation from true posterior p(w|x, y).

This is especially true for convolutional neural networks, where the values in the same convolutional kernel seem to be highly correlated.

However, we still choose this family of distribution in our design as the simplicity and efficiency are mostly concerned.

In fact, there are many techniques in deep learning area borrowing the idea of Bayesian inference without mentioning explicitly.

For example, Dropout BID31 ) is regarded as a powerful regularization tool for deep neural networks, which applies an element-wise product of the feature maps and i.i.d.

Bernoulli or Gaussian r.v.

B(1, α) (or N (1, α)).

If we allow each dimension to have an independent dropout rate and take them as model parameters to be learned, then we can extend it to the variational dropout method BID16 .

Notably, learning the optimal dropout rates for data relieves us from manually tuning hyper-parameter on hold-out data.

Similar idea is also used in RSE BID23 , except that it was used to improve the robustness under adversarial attacks.

As we discussed in the previous section, RSE incorporates Gaussian noise ∼ N (0, σ 2 ) in an additive manner, where the variance σ 2 is user predefined in order to maximize the performance.

Different from RSE, our Adv-BNN has two degrees of freedom (mean and variance) and the network is trained on adversarial examples.

In our method, we combine the idea of adversarial training BID25 with Bayesian neural network, hoping that the randomness in the weights w provides stronger protection for our model.

To build our Bayesian neural network, we assume the joint distribution q µ,s (w) is fully factorizable (see FORMULA6 ), and each posterior q µi,si (w i ) follows normal distribution with mean µ i and standard deviation exp(s i ) > 0.

The prior distribution is simply isometric Gaussian N (0 d , s 2 0 I d×d ).

We choose the Gaussian prior and posterior for its simplicity and closed-form KL-divergence, that is, for any two Gaussian distributions s and t, DISPLAYFORM0 Note that it is also possible to choose more complex priors such as "spike-and-slab" BID14 or Gaussian mixture, although in these cases the KL-divergence of prior and posterior is hard to compute and practically we replace it with the Monte-Carlo estimator, which has higher variance, resulting in slower convergence rate BID15 .Following the recipe of variational inference, we adapt the robust optimization to the evidence lower bound (ELBO) w.r.t.

the variational parameters during training.

First of all, recall the ELBO on the original dataset (the unperturbed data) can be written as DISPLAYFORM1 rather than directly maximizing the ELBO in FORMULA8 , we consider the following alternative objective, DISPLAYFORM2 This is essentially finding the minima for each data point (x i , y i ) ∈ D tr inside the γ-norm ball, we can also interpret (8) as an even looser lower bound of evidence.

So the robust optimization procedure is to maximize (8), i.e. DISPLAYFORM3 To make the objective more specific, we combine (8) with FORMULA10 and get arg max DISPLAYFORM4 In our case, p(y|x DISPLAYFORM5 ] is the network output on the adversarial sample (x adv i , y i ).

More generally, we can reformulate our model as y = f (x; w)+ζ and assume the residual ζ follows either Logistic(0, 1) or Gaussian distribution depending on the specific problem, so that our framework includes both classification and regression tasks.

We can see that the only difference between our Adv-BNN and the standard BNN training is that the expectation is now taken over the adversarial examples (x adv , y), rather than natural examples (x, y).

Therefore, at each iteration we first apply a randomized PGD attack (as introduced in eq (3)) for T iterations to find x adv , and then fix the x adv to update µ, s.

When updating µ and s, the KL term in (8) can be calculated exactly by (6), whereas the second term is very complex (for neural networks) and can only be approximated by sampling.

Besides, in order to fit into the back-propagation framework, we adopt the Bayes by Backprop algorithm BID2 .

Notice that we can reparameterize w = µ + exp(s) , where ∼ N (0 d , I d×d ) is a parameter free random vector, then for any differentiable function h(w, µ, s), we can show that DISPLAYFORM6 Now the randomness is decoupled from model parameters, and thus we can generate multiple to form a unbiased gradient estimator.

To integrate into deep learning framework more easily, we also designed a new layer called RandLayer, which is summarized in appendix.

It is worth noting that once we assume the simple form of variational distribution (5), we can also adopt the local reparameterization trick BID16 .

That is, rather than sampling the weights w, we directly sample the activations and enjoy the lower variance during the sampling process.

Although in our experiments we find the simple Bayes by Backprop method efficient enough.

For ease of doing SGD iterations, we rewrite (9) into a finite sum problem by dividing both sides by the number of training samples N tr µ * , s * = arg min DISPLAYFORM7 here we define g(µ, s) KL(q µ,s (w) p(w)) by the closed form solution (6), so there is no randomness in it.

We sample new weights by w = µ + exp (s) in each forward propagation, so that the stochastic gradient is unbiased.

In practice, however, we need a weaker regularization for small dataset or large model, since the original regularization in (12) can be too large.

We fix this problem by adding a factor 0 < α ≤ 1 to the regularization term, so the new loss becomes DISPLAYFORM8 In our experiments, we found little to no performance degradation compared with the same network without randomness, if we choose a suitable hyper-parameter α, as well as the prior distribution N (0, s 2 0 I).

The overall training algorithm is shown in Alg.

1.

To sum up, our Adv-BNN method trains an arbitrary Bayesian neural network with the min-max robust optimization, which is similar to BID25 .

As we mentioned earlier, even though our model contains noise and eventually the gradient information is also noisy, by doing multiple forward-backward iterations, the noise will be cancelled out due to the law of large numbers.

This is also the suggested way to bypass some stochastic defenses in .Algorithm 1 Code snippet for training Adv-BNN 1: procedure pgd attack(x, y, w) 2:Perform the PGD-attack (2), omitted for brevity 3: procedure train(data, w) 4:Input: dataset and network weights w 5:for (x, y) in data do 6:x adv ← pgd attack(x, y, w) Generate adversarial images 7: DISPLAYFORM9 loss ce ← cross entropy(ŷ, y) Cross-entropy loss 10:loss kl ← kl divergence(w) KL-divergence following FORMULA7 Will it be beneficial to have randomness in adversarial training?

After all, both randomized network and adversarial training can be viewed as different ways for controlling local Lipschitz constants of the loss surface around the image manifold, and thus it is non-trivial to see whether combining those two techniques can lead to better robustness.

The connection between randomized network (in particular, RSE) and local Lipschitz regularization has been derived in BID23 .

Adversarial training can also be connected to local Lipschitz regularization with the following arguments.

Recall that the loss function given data (x i , y i ) is denoted as f (x i ; w), y i , and similarly the loss on perturbed data (x i + ξ, y i ) is f (x i + ξ; w), y i ).

Then if we expand the loss to the first order DISPLAYFORM10 we can see that the robustness of a deep model is closely related to the gradient of the loss over the input, i.e. ∇ xi f (x i ), y i .

If ∇ xi f (x i ), y i is large, then we can find a suitable ξ such that ∆ is large.

Under such condition, the perturbed image x i + ξ is very likely to be an adversarial example.

It turns out that adversarial training (4) directly controls the local Lipschitz value on the training set, this can be seen if we combine FORMULA1 with (4) DISPLAYFORM11 Moreover, if we ignore the higher order term O( ξ 2 ) then (15) becomes DISPLAYFORM12 In other words, the adversarial training can be simplified to Lipschitz regularization, and if the model generalizes, the local Lipschitz value will also be small on the test set.

Yet, as BID22 indicates, for complex dataset like CIFAR-10, the local Lipschitz is still very large on test set, even though it is controlled on training set.

The drawback of adversarial training motivates us to combine the randomness model with adversarial training, and we observe a significant improvement over adversarial training or RSE alone (see the experiment section below).

In this section, we test the performance of our robust Bayesian neural networks (Adv-BNN) with strong baselines on a wide variety of datasets.

In essence, our method is inspired by adversarial training BID25 and BNN BID2 , so these two methods are natural baselines.

If we see a significant improvement in adversarial robustness, then it means that randomness and robust optimization have independent contributions to defense.

Additionally, we would like to compare our method with RSE BID23 , another strong defense algorithm relying on randomization.

Lastly, we include the models without any defense as references.

For ease of reproduction, we list the hyper-parameters in the appendix.

Readers can also refer to the source code on github.

It is known that adversarial training becomes increasingly hard for high dimensional data BID29 .

In addition to standard low dimensional dataset such as CIFAR-10, we also did experiments on two more challenging datasets: 1) STL-10 ( BID6 , which has 5,000 training images and 8,000 testing images.

Both of them are 96 × 96 pixels; 2) ImageNet-143, which is a subset of ImageNet BID7 , and widely used in conditional GAN training BID26 .

The dataset has 18,073 training and 7,105 testing images, and all images are 64×64 pixels.

It is a good benchmark because it has much more classes than CIFAR-10, but is still manageable for adversarial training.

In the first experiment, we compare the accuracy under the white box ∞ -PGD attack.

We set the maximum ∞ distortion to γ ∈ [0:0.07:0.005] and report the accuracy on test set.

The results are shown in Fig. 2 .

Note that when attacking models with stochastic components, we adjust PGD accordingly as mentioned in Section 2.1.

To demonstrate the relative performance more clearly, we show some numerical results in Tab.

1.

No defense BNN Adv.

training AdvBNN RSE Figure 2 : Accuracy under ∞ -PGD attack on three different datasets: CIFAR-10, STL-10 and ImageNet-143.

In particular, we adopt a smaller network for STL-10 namely "Model A" 1 , while the other two datasets are trained on VGG.From Fig. 2 and Tab.

1 we can observe that although BNN itself does not increase the robustness of the model, when combined with the adversarial training method, it dramatically increase the testing accuracy for ∼10% on a variety of datasets.

Moreover, the overhead of Adv-BNN over adversarial training is small: it will only double the parameter space (for storing mean and variance), and the total training time does not increase much.

Finally, similar to RSE, modifying existing network Table 1 : Comparing the testing accuracy under different levels of PGD attacks.

We include our method, Adv-BNN, and the state of the art defense method, the multi-step adversarial training proposed in BID25 .

The better accuracy is marked in bold.

Notice that although our Adv-BNN incurs larger accuracy drop in the original test set (where ξ ∞ = 0), we can choose a smaller α in (13) so that the regularization effect is weakened, in order to match the accuracy.architectures into BNN is fairly simple, we only need to replace Conv/BatchNorm/Linear layers by their variational version.

Hence we can easily build robust models based on existing ones.

Is our Adv-BNN model susceptible to transfer attack?

we answer this question by studying the affinity between models, because if two models are similar (e.g. in loss landscape) then we can easily attack one model using the adversarial examples crafted through the other.

In this section, we measure the adversarial sample transferability between different models namely None (no defense), BNN, Adv.

Train, RSE and Adv-BNN.

This is done by the method called "transfer attack" BID24 .

Initially it was proposed as a black box attack algorithm: when the attacker has no access to the target model, one can instead train a similar model from scratch (called source model), and then generate adversarial samples with source model.

As we can imagine, the success rate of transfer attack is directly linked with how similar the source/target models are.

In this experiment, we are interested in the following question: how easily can we transfer the adversarial examples between these five models?

We study the affinity between those models, where the affinity is defined by DISPLAYFORM0 where ρ However, ρ A →B = ρ B →A is not necessarily true, so the affinity matrix is not likely to be symmetric.

We illustrate the result in FIG3 .We can observe that {None, BNN} are similar models, their affinity is strong (ρ ≈ 0.85) for both direction: ρ BNN →None and ρ None →BNN .

Likewise, {RSE, Adv-BNN, Adv.

Train} constitute the other group, yet the affinity is not very strong (ρ ≈ 0.5∼0.6), meaning these three methods are all robust to the black box attack to some extent.

Following experiments are not crucial in showing the success of our method, however, we still include them to help clarifying some doubts of careful readers.

The first question is about sample efficiency, recall in prediction stage we sample weights from the approximated posterior and generate the label bŷ DISPLAYFORM0 In practice, we do not want to average over lots of forward propagation to control the variance, which will be much slower than other models during the prediction stage.

Here we take ImageNet-143 data + VGG network as an example, to show that only 10∼20 forward operations are sufficient for robust and accurate prediction.

Furthermore, the number seems to be independent on the adversarial distortion, as we can see in Fig. 4 (left).

So our algorithm is especially suitable to large scale scenario.

One might also be concerned about whether 20 steps of PGD iterations are sufficient to find adversarial examples.

It has been known that for certain adversarial defense method, the effectiveness appears to be worse than claimed , if we increase the PGD-steps from 20 to 100.

In Fig. 4 (right), we show that even if we increase the number of iteration to 1000, the accuracy does not change very much.

This means that even the adversary invests more resources to attack our model, its marginal benefit is negligible.

Figure 4: Left: we tried different number of forward propagation and averaged the results to make prediction (18).

We see that for different scales of perturbation γ ∈ {0, 0.01, 0.02}, choosing number of ensemble n = 10∼20 is good enough.

Right: testing accuracy stabilizes quickly as #PGD-steps goes greater than 20, so there is no necessity to further increase the number of PGD steps.

To conclude, we find that although the Bayesian neural network has no defense functionality, when combined with adversarial training, its robustness against adversarial attack increases significantly.

So this method can be regarded as a non-trivial combination of BNN and the adversarial training: robust classification relies on the controlled local Lipschitz value, while adversarial training does not generalize this property well enough to the test set; if we train the BNN with adversarial examples, the robustness increases by a large margin.

Admittedly, our method is still far from the ideal case, and it is still an open problem on what the optimal defense solution will be.

We largely follow the guidelines of attacking networks with "obfuscated gradients" in .

Specifically, we derive the algorithm for white box attack to random networks denoted as f (w; ), where w is the (fixed) network parameters and is the random vector.

Many random neural networks can be reparameterized to this form, where each forward propagation returns different results.

In particular, this framework includes our Adv-BNN model by setting w = (µ, s).

Recall the prediction is made through "majority voting": DISPLAYFORM0 So the optimal white-box attack should maximize the loss (19) on the ground truth label y * .

That is, ξ * = arg max ξ E f (x + ξ; w, ), y * ,and then x adv x + ξ * .

To do that we apply SGD optimizer and sampling at each iteration, DISPLAYFORM1 one can see the iteration (21) approximately solves (20).

It is very easy to implement the forward & backward propagation in BNN.

Here we introduce the RandLayer that can seamlessly integrate into major deep learning frameworks.

We take PyTorch as an example, the code snippet is shown in Alg.

1.

@highlight

We design an adversarial training method to Bayesian neural networks, showing a much stronger defense to white-box adversarial attacks