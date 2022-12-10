In this paper, we are interested in two seemingly different concepts: \textit{adversarial training} and \textit{generative adversarial networks (GANs)}. Particularly, how these techniques work to improve each other.

To this end, we analyze the limitation of adversarial training as a defense method, starting from questioning how well the robustness of a model can generalize.

Then, we successfully improve the generalizability via data augmentation by the ``fake'' images sampled from generative adversarial network.

After that, we are surprised to see that the resulting robust classifier leads to a better generator, for free.

We intuitively explain this interesting phenomenon and leave the theoretical analysis for future work.

Motivated by these observations, we propose a system that combines generator, discriminator, and adversarial attacker together in a single network.

After end-to-end training and fine tuning, our method can simultaneously improve the robustness of classifiers, measured by accuracy under strong adversarial attacks, and the quality of generators, evaluated both aesthetically and quantitatively.

In terms of the classifier, we achieve better robustness than the state-of-the-art adversarial training algorithm proposed in (Madry \textit{et al.}, 2017), while our generator achieves competitive performance compared with SN-GAN (Miyato and Koyama, 2018).

Deep neural networks have been very successful in modeling images, texts, and audios.

Nonetheless, their characters have not yet been fully understood BID36 , leaving a big hole for malicious attack algorithms.

In this paper, we start from adversarial attacks and defense but try to find the connection with Generative Adversarial Network (GAN) BID10 .

Superficially, the difference between them is that the adversarial attack is the algorithm that finds a highly resembled image to cheat the classifier, whereas the GAN algorithm at its core is a generative model where the generator learns to convert white noise to images that look like authentic to the discriminator.

We show in this paper that they are indeed closely related and can be used to strengthen each other: to accelerate and stabilize the GAN training cycle, the discriminator is expected to stay robust to adversarial examples; at the same time, a well trained generator provides a continuous support in probability space and thus improves the generalization ability of discriminator, even under adversarial attacks.

That is the starting point of our idea to associate generative networks with robust classifiers.

We find a novel way to make a connection between GAN and adversarial training.

More importantly, we develop a system called AdvGAN to combine generator, discriminator, and adversarial attacker in the same network.

Through the proposed "co-training" and "fine-tunning" steps, we are able to simultaneously improve the quality of generated images and the accuracy of discriminator under strong adversarial attacks.

For example, when applying state-of-the-art adversarial training technique BID25 , the accuracy of ResNet18(+CIFAR10) drops from 81.5% to 29.6%; whereas the accuracy of our discriminator network drops from 81.1% to 36.4% (keeping all the hyperparameters and network structure unchanged).

For the generator side, we are able to match or even beat the inception score of state-of-the-art method on medium scale datasets (see Sec. 4 for details), with significantly fewer iterations.

Lastly, we modify the loss of AC-GAN and our experiments confirm the superiority over the original one.

Notations Throughout this paper, we denote the (image, label) pair as (x i , y i ), i is the index of data point; The classifier parameterized by weights w is f (x; w), this function includes the final Softmax layer so the output is probabilities.

We also define D(x) and G(z) as the discriminator and generator networks respectively.

The adversarial example x adv is crafted by perturbing the original input, i.e. x adv = x + δ, where δ ≤ δ max .

For convenience, we consider ∞ -norm in our experiments.

The real and fake images are denoted as x real/fake , readers should differentiate the "fake" images with "adversarial" images 1 .

The training set is denoted as P real , this is the empirical distribution.

Given the training set P real , we define empirical loss function DISPLAYFORM0

Generative adversarial network.

This is a kind of algorithm that learns to model distribution either with or without supervision BID10 , which is often considered as a hard task especially for high dimensional data (images, texts, audios, etc.) .

In recent years, GANs keep to be intensively studied, toghther with other competitive generative models such as variational autoencoder or VAE, which learns the latent representation of data via prior knowledge BID20 , and auto-regressive model that models the conditional distribution given previous states (e.g. PixelCNN (van den Oord et al., 2016) ).

One advantage of GANs over other methods is that they are able to generate high quality images directly from certain distributions, whereas the other methods are either slow in generation, or yield blurry images.

A GAN has two competing networks with different objectives: in the training phase, the generator G(z) and the discriminator D(x) are evolved in a minimax game, which can be denoted as a unified loss: min DISPLAYFORM0 Unlike traditional machine learning problems where we typically minimize the loss, (1) is hard to optimize and that is the focus of recent literature.

Among them, a guideline for the architectures of G and D is summarized in BID32 .

Other training techniques, including feature matching (similar to MMD-GAN BID23 BID1 ) and mini-batch discrimination are proposed in BID12 to improve the stability and quality of networks.

For high resolution and photo-realistic image generation, currently the standard way is to first learn to generate low resolution images as the intermediate products, and then learn to refine them progressively BID8 BID19 , this turns out to be more stable than directly generate high resolution images through a gigantic network.

To reach the equilibrium efficiently, alternative loss metrics BID0 BID4 BID13 BID38 are applied and proven to be effective.

Among them, BID0 theoretically explains why training the DCGAN is highly unstable -since the image manifold is highly concentrated towards a low dimensional manifold, and if two distributions P real and P fake are supported on two low dimensional manifolds that do not perfectly align, then there exists an "optimal discriminator D(x)" that tells apart two distributions with probability one.

Moreover, under that situation, the gradient of discriminator ∇D(x) closes to zero and thus the training process is halted.

Closely following that theorem, proposes to use Wasserstein-1 distance to measure the distance between real and fake data distribution.

The resulting network, namely "Wasserstein-GAN", largely improves the stability of GAN training.

Another noteworthy work inspired by WGAN/WGAN-GP is spectral normalization , the idea is to estimate the operator norm σ max (W ) of weights W inside layers (convolution, linear, etc.) , and then normalize these weights to have 1-operator norm through dividing weight tensors by operator norm: W = W/σ max (W ).

Because ReLU non-linearity is already 1-Lipschitz, if we stack these layers together the network as a whole would still be 1-Lipschitz, that is exactly the prerequisite to apply Kantorovich-Rubinstein duality to estimate Wasserstein distance.

Despite the success of aforementioned works, we want to address one missing part of these models: to the best of our knowledge, none of them consider the robustness of discrimination network D(x).

This overlooked aspect can be problematic especially for high resolution images and large networks, this will be one of the central points of this paper.

Adversarial attacks and defenses: Apart from GAN, another key ingredient of our method is adversarial examples, originated in BID36 and further studied in BID11 .

They found that machine learning models can be easily "fooled" by slightly modified images if we design a tiny perturbation according to some "attack" algorithms.

In this paper we apply a simple yet efficient algorithm, namely PGD-attack BID25 , to generate adversarial examples.

Given an example x with ground truth label y, PGD computes adversarial perturbation δ by solving the following optimization with Projected Gradient Descent: DISPLAYFORM1 where f (·; w) is the network parameterized by weights w, (·, ·) is the loss function and for convenience we choose · to be the ∞ -norm in accordance with BID25 , but note that other norms are also applicable.

Intuitively, the idea of FORMULA2 is to find the point x adv := x + δ within an ∞ -ball such that the loss value of x adv is maximized, so that point is most likely to be an adversarial example.

In fact, most optimization based attacking algorithms (e.g. FGSM BID11 , C&W BID6 ) shares the same idea as PGD attack.

Opposite to the adversarial attacks, the adversarial defenses are techniques that make models resistant to adversarial examples.

It is worth noting that defense is a much harder task compared with attacks, especially for high dimensional data combined with complex models.

Despite that huge amount of defense methods are proposed BID31 BID25 BID5 BID24 BID14 BID9 BID40 BID33 , many of them rely on gradient masking or obfuscation, which provide an "illusion" of safety .

They claimed that the most effective defense algorithm is adversarial training BID25 , formulated as DISPLAYFORM2 where (x, y) ∼ P real is the (image, label) joint distribution of real data, f (x; w) is the network parameterized by w, f (x; w), y is the loss function of network (such as the cross-entropy loss).

We remark that the data distribution P real is often not available in practice, which will be replaced by the empirical distribution.3 PROPOSED APPROACH

In Sec. 2 we listed some of the published works on adversarial defense, and pointed out that adversarial training is the most effective method to date.

However, until now this method has only been tested on small dataset like MNIST and CIFAR10 and it is an open problem as to whether it scales to large dataset such as ImageNet.

To our knowledge, there are two significant drawbacks of this method that restrict its application.

First and most obviously, the overhead to find adversarial examples in each iteration is about 10x of the normal process (this can be inferred by #Iterations in each PGD attack FORMULA1 is obtained by adversarial training on CIFAR-10, we set δ max = 0.03125 in (3).

The horizontal axis is the attack strength δ which is equivalent to δ max in (2).

Note that δ max in FORMULA2 and FORMULA3 have different meaningsone is for attack and the other is for defense.

Notice the increasing accuracy gap when δ < 0.03125.

Right: The local Lipschitz value (LLV) measured by gradient norm ∂ ∂xi f (x i ; w), y i 2 , data pairs (x i , y i ) are chosen from the training and testing set respectively.

During the training process, LLV on the training set stabilizes at a low level, whereas LLV on the test set keeps growing.

methods such as .

In essence, restricting the LLV can be formulated as a composite loss minimization problem: DISPLAYFORM0 Notice that (5) can be regarded as the "one-step approximation" of (3).

In practice we need to change the expectation over P real to empirical distribution of finite data, DISPLAYFORM1 where DISPLAYFORM2 are feature-label pairs constitute the training set.

Ideally, if we have enough data and model size is moderate then the objective function in (6) still converges to (5).

However in practice when taking adversarial examples into account, we have one more problem to worry about: Does small LLV in training set imply small LLV in test set?

The enlarged accuracy gap shown in FIG0 (Left) tends to give a negative answer.

To verify this phenomenon directly, we calculate the LLV on images sampled from training and testing set respectively ( FIG0 ), we observe that in parallel with accuracy gap, the LLV gap between training and testing set is equally significant.

Thus we conclude that although adversarial training controls LLV around training set effectively, this property does not generalize to test set.

Notice that our empirical finding does not contradict the certified robustness of adversarial training using generalization theory (e.g. BID34 ), which only explains weak attack situation.

The generalization gap can be potentially reduced if we have a better understanding of P real instead of approximating it by training set.

This leads to our first motivation: can we use GAN to learn P real and plug it into adversarial training algorithm to improve robustness on test set?

We will give a possible solution in Sec. 3.3.

GANs are notoriously hard to train.

To our knowledge, there are two major symptoms of a failure trial -gradient vanishing and mode collapse.

The theoretical explanation of gradient vanishing problem is discussed in BID0 by assuming the images lie in a low dimensional manifold.

Following this idea, BID12 propose to use 1-Wasserstein distance in place of the KL-divergence.

The central character of WGAN and improved WGAN is that they require the set of discriminators {D(x; w)|∀w ∈ R d } equals to the set of all 1-Lipschitz functions w.r.t input x. Practically, we can either clip the weight of discriminator w , or add a gradient norm regularizer BID12 .

Recently, another regularization technique called "spectral normalization" ) is proposed to enforce 1-Lipschitz discriminator and for the first time, GAN learns to generate high quality images from full ImageNet data with only one generator-discriminator pair.

In contrast, AC-GAN BID30 -the supervised version of DCGAN -divides 1000 classes into 100 groups so each network-pair only learns 10 classes.

Despite the success along this line of research, we wonder if a weaker assumption to the discriminator is possible.

Concretely, instead of a strict one-Lipschitz function, we require a small local Lipschitz value on image manifold.

Indeed, we find a connection between robustness of discriminator and the learning efficiency of generator, as illustrated in Fig. 2 .

Fake images

Figure 2: Comparing robust and non-robust discriminators, for simplicity, we put them together into one graph.

Conceptually, the non-robust discriminator tends to make all images close to the decision boundary, so even a tiny distortion δ can make a fake image x 0 to be classified as a real image x adv = x 0 + δ.

In contrast, such δ is expected to be much larger for robust discriminators.

As one can see in Fig. 2 , if a discriminator D(x) has small LLV (or |D (x)|), then we know DISPLAYFORM0 for a "reasonably" large δ.

In other words, for robust discriminator, the perturbed fake image x adv = x 0 + δ is unlikely to be mistakenly classified as real image, unless δ is large.

Compared with adversarial attacks (2), the attacker is now a generator G(z; w) parameterized by w ∈ R d instead of the gradient ascend algorithm.

For making x 0 "looks like" a real image (x adv ), we must update generator G(z; w) to G(z; w ) and by assuming the Lipschitz continuity of G, DISPLAYFORM1 This indicates the movement of generator weights w − w is lower bounded by the distance of a fake image x 0 to the decision boundary, specifically we have w − w ≥ δ /L G .

Furthermore, recall that a robust discriminator D(x) implies a larger δ , putting them together we know that improving the robustness of discriminator will lead to larger updates of the generator.

In Sec. 4 we experimentally show that adversarial training not only speeds up the convergence to the equilibrium, but also obtains an excellent generator.

But we leave the rigorous analysis for future works.

Motivated by Sec. 3.1 and 3.2, we propose a system that combines generator, discriminator, and adversarial attacker into a single network.

Our system consists of two stages, the first stage is an end-to-end GAN training: the generator feeds fake images to the discriminator; meanwhile real images sampled from training set are processed by PGD attacking algorithm before sending to the discriminator.

After that the discriminator is learned to minimize both discrimination loss and classification loss (introduced below).

In the next stage, the discriminator is refined by combining the fake and real images.

The network structure is illustrated in Fig. 3 .

In what follows, we give more details about each component: Discriminator: The discriminator could have the standard architecture like AC-GAN.

In each iteration, it discriminates real and fake images.

When the ground truth labels are available, it also predicts the classes.

In this paper, we only consider the label-conditioning GANs BID27 BID30 , whose architectural differences are briefly overviewed in FIG3 .

Among them we simply choose AC-GAN, despite that SN-GAN (a combination of spectral normalization and projection discriminator ) performs much better in their paper.

The reason we choose the AC-GAN is that Step 1.

Co-trainingStep 2.

Fine-tuning Figure 3 : Illustration of the training process.

Step-1 is the standard GAN training, i.e. alternatively updating the G and D networks.

The only difference is that whenever feeding the real images to the D network, we first run 5 steps of PGD attack, so the discriminator is trained with adversarial examples.

Step-2 is a refining technique, aiming at improving prediction accuracy on the test set.

SN-GAN discriminator relies on the ground truth labels and their adversarial loss is not designed to encourage high classification accuracy.

But surprisingly, even though AC-GAN is beaten by SN-GAN by a large margin, after inserting the adversarial training module, the performance of AC-GAN matches or even surpasses the SN-GAN, due to the reason discussed in Sec. 3.2.

We also changed the loss objective of AC-GAN.

Recall that the original loss in BID30 defined by discrimination likelihood L S and classification likelihood L C : DISPLAYFORM0 where X real/fake are any real/fake images, S is the discriminator output, C is the classifier output.

Based on (8), the goal of discriminator is to maximize L S + L C while generator aims at maximizing L C − L S .

According to this definition, both G and D are optimized to increase L C : even if G(z) produces unrecognizable images, D(x) has to struggle to classify them (with high loss), in such case the corresponding gradient term ∇L C can contribute uninformative direction to the discriminator.

To resolve this issue, we split L C as follows, DISPLAYFORM1 then discriminator maximizes L S + L C1 and generator maximizes L C2 − L S .

The new objective functions ensure that discriminator only focuses on classifying the real images and discriminating real/fake images.

Generator: Similar to the traditional GAN training, the generator is updated on a regular basis to mimic the distribution of real data.

This is the key ingredient to improve the robustness of discriminators: as shown in Sec. 3.1, adversarial training performs well on training set but is vulnerable on test set.

Intuitively, this is because during adversarial training, the network only "sees" adversarial examples residing in the δ max -ball of all training samples, whereas the rest images in the data manifold are undefended.

Data augmentation is a natural way to resolve this issue, but traditional techniques BID21 BID15 BID37 BID41 BID17 rely largely on combinations of geometric transforms to the training images, in our case the support of the probability density function is still very small.

Instead, our system uses images sampled from generator to provide a continuously supported p.d.f.

for the adversarial training.

Unlike traditional augmentation methods, if the equilibrium in (1) is reached, then we can show that one desirable solution of (1) would be P fake (z) dist.= P real , and therefore the robust classifier can be trained on the learned distribution.

Fine-tuning the classifier: This step aims at improving the classification accuracy, based on the auxiliary classifier in the pretrained discriminator.

This is crucial because in the GAN training stage (step 1 in Fig. 3) , the discriminator is not trained to minimize the classification error, but a weighted loss of both discrimination and classification.

But in step 2, we want to focus on the robust classification task DISPLAYFORM2 where x adv = arg min DISPLAYFORM3 Here the function f (x; w) is just the classifier part of network D(x), recall that we are dealing with conditional GAN.

As we can see, throughout the fine-tuning stage, we force the discriminator to focus on the classification task rather than the discrimination task.

It turns out that the fine-tuning step boosts the accuracy by a large margin.

Adversarial attacker is omitted in Fig. 3 due to width limit.

We experiment on both CIFAR10 and a subset of ImageNet data.

Specifically, we extract classes y i such that y i ∈ np.arange(151, 294, 1) from the original ImageNet data: recall in total there are 1000 classes in ImageNet data and we sampled 294 − 151 = 143 classes from them.

We choose these datasets because 1) the current state-of-the-art GAN, SN-GAN , also worked on these datasets, and 2) the current state-of-the-art adversarial training method BID25 only scales to CIFAR dataset.

For fair comparison, we copy all the network architectures for generators and discriminators from SN-GAN, other important factors, such as learning rate, optimization algorithms, #discriminator updates in each cycle, etc. are also kept the same.

The only modification is that we discarded the feature projection layer and applied the auxiliary classifier (see FIG3 ).

Please refer to the appendix or source code for more implementation details.

In what follows, we check whether fine-tuning helps improving test set accuracy.

To this end, we design a experiment that compares two set of models: in the first set, we directly extract the auxiliary classifiers from discriminators to classify images; in the next set, we apply fine-tuning strategy to the pretrained model as Fig. 3 illustrated.

The results can be found in FIG4 , which supports our argument that fine-tuning is indeed useful for better prediction accuracy.

Robustness of discriminator: comparing robustness with/ without data augmentation In this experiment, we would like to compare the robustness of discriminator networks with or without data augmentation technique discussed in Sec. 3.3.

The robustness is measured by the prediction accuracy under adversarial attack.

For networks without data augmentation, that would be equal to the state-of-the-art Madry's algorithm BID25 .

For attacking algorithm, we choose the widely used ∞ PGD attack BID25 , but other gradient based attacks are expected to Table 1 : Accuracy of our model under ∞ PGD-attack.

Inside the parenthesis is the improvement over standard adversarial training defense BID25 .yield the same results.

We set the ∞ perturbation to σ max ∈ np.arange(0, 0.1, 0.01) as defined in (2).

Another minor detail is that we scale the images to [−1, 1] rather than usual [0, 1] .

This is because generators always have a tanh() output layer, so we need to do some adaptations accordingly.

We exhibit the results in Tab.

1, showing our method can improve the robustness of state-of-the-art defensive algorithm.

Effect of split classification loss Here we show the effect of split classification loss described in (9), recall that if we apply the loss in (8) then the resulting model is AC-GAN.

It is known that AC-GAN can easily lose modes in practice, i.e. the generator simply ignores the noise input z and produces fixed images according to the label y. This defect is observed in many previous works BID16 BID26 BID18 .

In this ablation experiment, we compare the generated images trained by two loss functions, the result is shown in FIG5 .

Quality of generator and convergence speed In the last experiment, we compare the quality of generators trained in three datasets: CIFAR10, ImageNet subset (64px) and ImageNet subset (128px).

Our baseline model is the SN-GAN, considering that, as far as we know, SN-GAN is the best GAN model capable of learning hundreds of classes.

Note that SN-GAN can also learn the conditional distribution of the entire ImageNet data (1000 classes), unfortunately, we are not able to match this experiment due to time and hardware limit.

To show that the adversarial training technique indeed accelerates the convergence speed, we also tried to exclude adversarial training -this is basically an AC-GAN, except that an improved loss function discussed in (9) is applied to discriminator D(x).

The results are exhibited in FIG6 , which shows that adversarial training can improve the performance of GAN, and our generator achieves better inception score than SNGAN.

Another finding is that our new loss proposed in FORMULA10 We compare the inception scores between our model and the SN-GAN.

Clearly our method learns a high quality generator in a short time, specifically, in both datasets, AC-GAN with adversarial training surpasses SN-GAN in just 25 epochs (64px) or 50 epochs (128px).

Another observation is that with adversarial training, the convergence is greatly accelerated.whether adversarial training with fake data augmentation really shrinks the generalization gap.

To this end, we draw the same figure as FIG0 , except that now the classification model is the discriminator after fine-tuning step (shown in Fig. 3) .

We compare the accuracy gap in FIG7 .

Clearly the model trained with the adversarial real+fake augmentation strategy works extremely well: it improves the testing accuracy under PGD-attack and so the generalization gap between training/testing set does not increase that much.

In this paper, we draw a connection between adversarial training BID25 and generative adversarial network BID10 .

Our primary goal is to improve the generalization ability of adversarial training and this is achieved by data augmentation by the unlimited fake images.

Independently, we see an improvement of both robustness and convergence speed in GAN training.

While the theoretical principle in behind is still unclear to us, we gave an intuitive explanation.

Apart from that, a minor contribution of our paper is the improved loss function of AC-GAN, showing a better result in image quality.

<|TLDR|>

@highlight

We found adversarial training not only speeds up the GAN training but also increases the image quality