Despite the remarkable performance of deep neural networks (DNNs) on various tasks, they are susceptible to adversarial perturbations which makes it difficult to deploy them in real-world safety-critical applications.

In this paper, we aim to obtain robust networks by sparsifying DNN's latent features sensitive to adversarial perturbation.

Specifically, we define vulnerability at the latent feature space and then propose a Bayesian framework to prioritize/prune features based on their contribution to both the original and adversarial loss.

We also suggest regularizing the features' vulnerability during training to improve robustness further.

While such network sparsification has been primarily studied in the literature for computational efficiency and regularization effect of DNNs, we confirm that it is also useful to design a defense mechanism through quantitative evaluation and qualitative analysis.

We validate our method, \emph{Adversarial Neural Pruning (ANP)} on multiple benchmark datasets, which results in an improvement in test accuracy and leads to state-of-the-art robustness.

ANP also tackles the practical problem of obtaining sparse and robust networks at the same time, which could be crucial to ensure adversarial robustness on lightweight networks deployed to computation and memory-limited devices.

In the last many years, deep neural networks (DNNs) have achieved impressive results on various artificial intelligence tasks, e.g., image classification , face and object recognition (He et al., 2015; Deng et al., 2018) , semantic segmetnation (Badrinarayanan et al., 2015; He et al., 2017) and playing games (Silver et al., 2016; .

The groundbreaking success of DNNs has motivated their use in a broader range of domains, including more safety-critical environments such as medical imaging (Esteva et al., 2017; Rajpurkar et al., 2017) and autonomous driving (Bojarski et al., 2016; Li et al., 2017) .

However, DNNs are shown to be extremely brittle to carefully crafted small adversarial perturbations added to the input (Szegedy et al., 2013; Goodfellow et al., 2014) .

These perturbations are imperceptible to human eyes but have been intentionally optimized to cause miss-classification.

While the field has primarily focused on the development of new attacks and defenses, a 'cat-andmouse' game between attacker and defender has arisen.

There has been a long list of proposed defenses to mitigate the effect of adversarial examples defenses (Papernot et al., 2015b; Xu et al., 2017b; Buckman et al., 2018; Dhillon et al., 2018; Xie et al., 2018; Tramèr et al., 2018; Liu et al., 2019) , followed by round of successful attacks (Carlini & Wagner, 2016; Uesato et al., 2018; designed in light of the new defense.

Since it shows that any defense mechanism that once looks successful could be circumvented with the invention of new attacks, we try to tackle the problem by identifying a more fundamental cause of the adversarial vulnerability of deep neural networks.

What makes deep neural networks vulnerable to adversarial attacks?

We conjecture that the adversarial vulnerability of deep neural networks is mostly due to the distortion in the latent feature space.

If any perturbation at the input level is successfully suppressed in the latent feature space at any layer of the neural network, such that clean and adversarial samples cannot be distinguished in the latent feature space, then it will not lead to misclassification.

However, not all latent features will contribute equally to the distortion in the latent feature space; some latent features may have larger distortion, by amplifying the perturbations at the input level while others will remain relatively static.

We consider a novel problem of distortion in latent features of a network in the presence of adversarial perturbation, where the model observes different degrees of distortion for different features (brighter red indicates higher level of distortion).

To solve this problem, our proposed method learns a bayesian pruning mask to suppress the higher distorted features in order to maximize it's robustness on adversarial perturbations.

In this paper, based on the motivation that adversarial vulnerability comes from distortion in the latent feature space, we first formally define the vulnerability of the latent features and propose to minimize the feature-level vulnerability to achieve adversarial robustness with DNNs.

One way to suppress the vulnerability in the feature space is by adding a regularization that minimizes it.

However, a more effective and irreversible means is to set the vulnerability to zero, by completely dropping the latent features with high vulnerability.

This is shown in Figure 2 (a), where sparse networks are shown to have a much smaller degree of vulnerability (average perturbation of the latent feature across all layers).

However, naive sparsification approaches will prune both the robust and vulnerable features, which will limit its effectiveness as a defense mechanism.

Moreover, when the sparsity is pushed further, it will prune out robust features which will hurt the model robustness.

To overcome this limitation, we propose the so-called adversarial neural pruning (ANP) method that adversarially learns the pruning mask, such that we can prune out vulnerable features while preserving robust ones.

Our method requires little or no modification of the existing network architectures, can be applied to any pre-trained networks and it effectively suppresses the distortion in the latent feature space (See Figure 1) and thus obtains a model that is more robust to adversarial perturbations.

We validate our model on multiple heterogeneous datasets including MNIST, CIFAR-10, and CIFAR-100 for its adversarial robustness.

Our experimental results show that ANP achieves significantly improved adversarial robustness, with significantly less memory and computational requirements.

In summary, the contribution of this paper is as follows:

• We consider the vulnerability of latent features as the main cause of DNN's susceptibility to adversarial attacks, and formally describe the concepts of vulnerable and robust latent features, based on the expectation of the distortion with respect to input perturbations.

• We show that while sparsity improves the robustness of DNNs by zeroing out distortion at the pruned features, it is still orthogonal to robustness and even degenerates robustness at a high degree, via experimental results and visualization of the loss landscape.

• Motivated by the above findings, we propose the ANP method that prunes out vulnerable features while preserving robust ones, by adversarially learning the pruning mask in a Bayesian framework.

During training, we also regularize the vulnerability of the latent features to improve robustness further.

•

The proposed ANP framework achieves state-of-the-art robustness on CIFAR-10 and CIFAR-100 datasets, along with a large reduction in memory and computation.

While our major focus is on achieving robustness with DNNs, we found that ANP also achieves higher accuracy for clean/non-adversarial inputs, compared to the baseline scheme of adversarial training (see the results of CIFAR datasets in Table 1 ).

This is due to the fact that sparsification helps to regularize models and is also an important benefit of ANP as it has been well known that adversarial training schemes tend to hurt the accuracy of the DNNs on non-adversarial samples Tsipras et al., 2019; Zhang et al., 2019) .

Moreover, our method enables to obtain a robust and lightweight network, which is useful when working with resource-limited devices.

Before presenting adversarial neural pruning, we first briefly introduce some notation and the concept of robust and vulnerable features in the deep latent representation space.

Let a L-layer neural network be represented by a function f : X → Y with dataset denoted by D = {X, Y} such that

for any x ∈ X. We use f θ to represent the neural network classifier, where

MNIST CIFAR10 CIFAR100

Figure 2: a) Mean distortion (average perturbation in latent features across all layers) for various networks.

We use Lenet-5-Caffe for MNIST and VGG-16 for CIFAR-10 and CIFAR-100 dataset.

Our proposed method has minimum distortion compared to all the other networks.

b) -d) Visualization of the mean distortion for the input layer for Lenet-5-caffe on MNIST for various methods.

The standard network has the maximum distortion which is comparatively reduced in adversarial training and further suppressed by our proposed method.

Let z l denote the feature vector for the l-th layer with rectified linear unit (ReLU) as the activation function, then f l (·) can be defined as

where W l denotes the weight parameter matrix and b l denotes the bias vector.

Let x and x adv denote clean and adversarial data points, respectively (x adv = x + δ) for any x ∈ X with l p -ball B(x, ε) around x with radius ε, z l and z adv l as their corresponding feature vectors for the l-th layer.

Vulnerability of a feature.

The vulnerability of a k-th feature for l-th layer (z lk ) can be measured by the distortion in that feature in the presence of an adversary.

The vulnerability of a latent feature could then be defined as the expectation of the absolute difference between the feature value for a clean example and its adversarial perturbation.

This could be formally defined as follows:

Definition 1 A feature z: X → R for a given strength of adversary ε is said to be (ε, δ)-robust to adversarial perturbation for a distribution D with respect to the vulnerability metric v, if there exists z adv such that v(z, z adv ) < δ.

Formally:

Definition 2 A feature z: X → R for a given strength of adversary ε is said to be (ε, δ)-vulnerable to adversarial perturbation for a distribution D with respect to the vulnerability metric v, if there exists z adv such that v(z, z adv ) ≥ δ.

Formally:

To measure the vulnerability of an entire network f θ (X), we simply need to compute the sum of the vulnerability of all the latent features vectors of the network before the logit layer, then V (f θ (X), f θ (X adv )) can be defined as:

where v l represents the vulnerability of the layer l with a feature vector composed of N l features.

Figure 2 (a) shows the vulnerability of different networks across various datasets.

It can be clearly observed that although adversarial training suppresses the vulnerability at the input level, the latent feature space is still vulnerable to adversarial perturbation, and that our proposed method achieves the minimum distortion across all the datasets.

Adversarial training.

Adversarial training (Goodfellow et al., 2014; Kurakin et al., 2016) was proposed as a data augmentation method to train the network on the mixture of clean and adversarial examples until the loss converges.

Instead of using it as a data augmentation technique, the adversarial search was incorporated inside the training process by solving the following non-convex outer minimization problem and a non-concave inner maximization problem :

In order to minimize the vulnerability of the network to further improve the robustness of the model we regularize the adversarial training loss with vulnerability suppression loss (VS):

where λ is the hyper-parameter determining the strength of the vulnerability suppression loss.

The vulnerability suppression loss directly aims to minimize the distortion of the latent features in the presence of adversarial perturbations.

Adding VS loss to the adversarial learning objective will make it minimize the adversarial loss by suppressing the distortions.

We empirically found that it also has an effect of pushing the decision boundary and increasing the smoothness of the model's output and its loss surface (See Figure 5 ).

In this section, we propose a new method, coined Adversarial Neural Pruning (ANP), to further reduce the vulnerability in the latent space.

ANP combines the idea of adversarial training with the Bayesian pruning methods.

used weight pruning and activation pruning to show that sparsifying networks leads to more robust networks.

The actual reason behind the robustness is obvious by our definitions: sparsity suppresses vulnerability to 0 and thus reduces the vulnerability of the network.

Yet, the network still does not take into account the robustness of a feature.

The basic idea of adversarial neural pruning is to achieve robustness while suppressing the distortion, by explicitly pruning out the latent features with high distortion.

Let L(θ M, x, y) be the loss function at data point x with class y for any x ∈ X for the model with parameters θ and mask parameters M, we can use Projected Gradient Descent (PGD) , a variant of IFGSM (Kurakin et al., 2016) to generate the adversarial examples:

where α is the step size and sgn(·) returns the sign of the vector.

In this work, we consider the l ∞ -bounded perturbations where δ is the added perturbation from the l ∞ norm-ball B(x, ε) around x with radius ε for each example.

We then use the following objectives to train the weight and mask parameters for our model:

vulnerability suppression loss

where β is the coefficient determining the strength of the adversarial classification loss and λ is the coefficient determining the strength of the vulnerability suppression loss.

We now introduce our proposed method Adversarial neural pruning for Beta Bernoulli dropout ) based on our complete Algorithm 1.

We emphasize that our proposed method can be extended to any existing or new sparsification method in a similar way.

Adversarial beta bernoulli dropout.

Beta Bernoulli Dropout learns to set the dropout rate by generating the dropout mask from sparsity-inducing beta-Bernoulli prior for each neuron.

Let W be a parameter of neural network layer with K channels and Z = {z 1 , . . .

, z n } be the mask sampled from the finite-dimensional beta-Bernoulli prior to be applied for the n-th observation x n .

The goal is to compute the posterior distribution p(W, Z, π|D) and we approximate this posterior using an approximate variational distribution q(W, Z, π|X) of known parametric form.

We conduct computationally efficient point-estimate for W to get the single value W, with the weight decay regularization from the zero-mean Gaussian prior.

For π, we use the Kumaraswamy distribution (Kumaraswamy, 1980) with parameters a and b. Using the Stochastic Gradient Variational

for number of training iterations do 3:

Generate adversarial examples {x Optimize the weights θ and the mask parameters M for the network in Equation 6 using gradient descent.

6: end for 7: end for 8: Return the pruned network.

Bayes (SGVB) framework , we can then get the final loss L M as:

where γ is Euler-Mascheroni constant, Ψ(.) is digamma function.

The first term in the loss measures the log-likelihood of the adversarial samples w.r.t.

q(Z; π) and the second term regularizes q(Z; π) so it doesn't deviate too much from the prior distribution.

We use ANP to refer to Adversarial Beta Bernoulli dropout for the rest of our paper.

The results for Adversarial Variational information bottleneck (Dai et al., 2018) and detailed derivation of Equation 7 are deferred to the appendix.

Datasets.

1) MNIST.

This dataset (LeCun, 1998) contains 60,000 28 × 28 grey scale images of handwritten digits for training example images, where there are 5,000 training instances and 1,000 test instances per class.

As for the base network, we use LeNet 5-Caffe 1 for this dataset.

2) CIFAR-10.

This dataset (Krizhevsky, 2012) consists of 60,000 images sized 32 × 32, from ten animal and vehicle classes.

For each class, there are 5,000 images for training and 1,000 images for test.

We use VGG-16 (Simonyan & Zisserman, 2015) for this dataset with 13 convolutional and two fully connected layers with pre-activation batch normalization and Binary Dropout.

3) CIFAR-100.

This dataset (Krizhevsky, 2012 ) also consists of 60,000 images of 32 × 32 pixels as in CIFAR-10 but has 100 generic object classes instead of 10.

Each class has 500 images for training and 100 images for test.

We use VGG-16 (Simonyan & Zisserman, 2015) similar to CIFAR-10 as the base network for this dataset.

Baselines and our model.

1) Standard.

The base convolution neural network.

2) Bayesian Pruning (BP).

The base network with beta-bernoulli dropout .

3) Adversarial Training (AT).

The adversarial trained network .

4) Adversarial Neural Pruning (ANP).

Adversarial neural pruning with beta-bernoulli dropout.

4) Adversarial Training with vulnerability suppression (AT-VS).

The adversarial trained network regularized with vulnerability suppression loss.

5) Adversarial Neural Pruning with vulnerability suppression (ANP-VS).

The adversarial neural pruning network regularized with vulnerability suppression loss.

Evaluation setup.

We report the clean accuracy, vulnerability metric (Equation 2) and accuracy on adversarial examples generated from l ∞ white box and black box attack using an adversarial trained full network for ANP and the standard base network for standard bayesian compression method.

All models and algorithms are implemented using the Tensorflow library (Abadi et al., 2016) .

For the reproduction of results, we list the hyper-parameters in the appendix.

Table 2 : Adversarial accuracy of CIFAR-10 and CIFAR-100 for VGG-16 architecture under l ∞ -PGD white box attack for different epsilon values (ε) with perturbation per step of (0.007), 40 total attack steps and different PGD iterations with ε = 0.03 and perturbation per step of (0.007).

Evaluation on MNIST.

We evaluate our Lenet 5-Caffe defense model for MNIST with similar attack parameters as in : total adversarial perturbation of 76.5/255 (0.3), perturbation per step of 2.55/255 (0.01) and 40 total steps with random restarts in Table 1 .

First observe that AT-VS achieves significantly improved robustness compared to the original AT model.

The standard Bayesian pruning method achieves the best generalization and marginally improve robustness in comparison to the standard-base model but they are not able to defend against the adversarial perturbation.

ANP-VS outperforms all the baselines, achieving 68% reduction in vulnerability with 2% improvement in adversarial accuracy under both white box and black box attacks.

Evaluation on CIFAR-10 and CIFAR-100.

Compared with MNIST, CIFAR-10 and CIFAR-100 are much more difficult tasks for classification and adversarial robustness.

Our goal here is not just to achieve state-of-the-art robustness but to also compare the the generalization capabilities of various training methods.

We use total adversarial perturbation of (0.03), perturbation per step of (0.007), 10 total attack steps for training and 40 total steps with random restarts for evaluating the CIFAR-10 and CIFAR-100 datasets.

The results for both CIFAR-10 and CIFAR-100 are summarized in Table 1 .

AT-VS improves the robustness of AT by 5% and vulnerability by 52% approximately.

However, ANP without VS largely outperforms AT-VS showing the effectiveness of adversarial pruning over regularization on the distortion, as it can drop features with high vulnerability.

ANP-VS achieves state of the art robustness for CIFAR-10 and CIFAR-100 for both white-box and black-box attack with 14.78% and 21.92% improvement in adversarial accuracy along with 65.96% and 58.33% reduction in vulnerability as compared to standard adversarial training for CIFAR-10 and CIFAR-100, respectively with significant improved clean accuracy.

We also consider various numbers of PGD steps and different epsilon values.

It has been shown that for certain defenses the robustness decreases as the number of PGD steps are increased (Engstrom et al., 2018) .

Table 2 shows the results for different l ∞ epsilon values and PGD iterations up to 1000.

It can be observed that compared to AT, all our methods achieve better robustness across all the l ∞ -epsilon values and PGD iterations, with ANP-VS outperforming all the competing methods on most adversaries.

This confirms that even if the attacker uses greater resources to attack our model, the effect is negligible.

Our adversarial neural pruning method could be useful when we want to obtain a lightweight yet robust network, for its deployment to computation and memory-limited devices.

To show that we can achieve both goals at once, we evaluate the defense performance of ANP at various sparsity levels in Figure 3 .

We experiment with different scaling coefficient by scaling the KL term in ELBO in Equation 7 to obtain architectures with different levels of sparsity, whose details can be found in the appendix.

ANP leads to 88% reduction in memory footprint while maintaining similar level of robustness.

We observe that ANP outperforms adversarial training up to a sparsity level of 80% for both CIFAR-10 and CIFAR-100 after which there is a decrease in the robustness and the clean accuracy.

The results are not surprising, as it is an overall outcome of the model capacity reduction and the removal of the robust features.

We further compare ANP with another baseline PAT where we first perform bayesian pruning, freeze the dropout mask and then perform adversarial training.

It can be observed that PAT slightly improves the adversarial robustness but loses on clean accuracy.

This proves the fact just naive approach of pruning over adversarial training can hurt performance.

This result confirms the effectiveness of our method as a defense mechanism.

The distributions of neurons and memory efficiency for various datasets can be seen in the appendix.

Analysis of individual components.

We first individually dissect the effectiveness of different components in adversarial neural pruning: Pretrained model for adversarial robustness, bayesian component in bayesian pruning, we conduct a series of ablation experiments with adversatrial training on a pretrained model (Pretrained AT, Hendrycks et al. 2019 ) and adversarial bayesian neural network (AT BNN, Liu et al. 2019) in Table 3 .

The results illustrate that both the methods individually improve the robustness of the model.

By combining pretraining and bayesian component, our final algorithm exhibits significant improvement across both the individual components.

Vulnerability analysis.

We next visualize the vulnerability of the latent-feature space.

Figure 4 shows the vulnerability for each image for various set of datasets and the vulnerability distribution for all the features of the input layer for CIFAR-10.

The results clearly show that the latent-features of the standard model are the most vulnerable, and the vulnerability decreases with the adversarial training and further suppressed by half with adversarial neural pruning.

Further, the latent features of our proposed method align much better with the human perception which also results in the interpretable gradients as observed in the previous work (Tsipras et al., 2019; .

The bottom row of Figure 4 shows the histogram of the feature vulnerability defined in Equation 1 for various methods.

We consistently see that standard Bayesian pruning zeros out some of the distortions, and adversarial training reduces the distortion level of all the features.

On the other hand, adversarial neural pruning does both, with the largest number of features with zero distortion and low distortion level in general.

Loss landscape visualization.

We further visualize the loss surface of the baseline models and network obtained using our adversarial pruning technique in Figure 5 .

We vary the input along a linear space defined by the sign of gradient where x and y-axes represent the perturbation added in each direction and the z-axis represents the loss.

The loss is highly curved in the vicinity of the data point x for the standard networks which reflects that the gradient poorly models the global landscape.

On the other hand, we observe that both sparsity and adversarial training make the loss surface smooth, with our model obtaining the most smooth surface.

Adversarial robustness.

Since the literature on adversarial robustness of neural networks is vast, we only discuss some of the most relevant studies.

Large number of defenses (Papernot et al., 2015a; Xu et al., 2017a; Buckman et al., 2018; Dhillon et al., 2018; Song et al., 2018; Wong & Kolter, 2018) have been proposed and consequently broken by more sophisticated attack methods (Carlini & Wagner, 2016; Uesato et al., 2018; .

One of the most successful defense is adversarial training , in which the neural network is trained to optimize the maximum loss obtainable using projected gradient descent over the region of allowable perturbations.

There has also been previous work which considered robust and vulnerable features at the input level.

Garg et al. (2018) establish a relation between adversarially robust features and the spectral property of the geometry of the dataset and Gao et al. (2017) proposed to remove unnecessary features in order to get robustness.

Our work is different from these existing work in that we consider and define the vulnerability at the latent feature level, which is more directly related to model prediction.

Sparsification methods.

Sparsification of neural networks is becoming increasingly important with the increased deployments of deep network models to resource-limited devices.

Simple heuristicsbased pruning methods based on removing weights with small magnitude (Ström, 1997; Collins & Kohli, 2014; Han et al., 2015) have demonstrated high compression with minimal accuracy loss.

However elementwise sparsity does not yield practical speed-ups and Wen et al. (2016) proposed to use group sparsity to drop a neuron or a filter, that will reduce the actual network size.

Recently, Bayesian and Information-theoretic approaches (Molchanov et al., 2017; Neklyudov et al., 2017; Dai et al., 2018; have shown to yield high compression rates and accuracy while providing theoretical motivation and connections to classical sparsification and regularization techniques.

Robustness and sparsity.

The sparsity and robustness have been explored and modelled together in various recent works.

and Ye et al. (2018) analyze sparsity and robustness from a theoretical and experimental perspective and demonstrate that appropriately higher sparsity leads to a more robust model.

In contrary, derived opposite conclusions showing that robustness decreases with increase in sparsity.

On the contrary, we sparsify networks while explicitly targeting for robustness, as we learn the pruning (dropout) mask to minimize loss on adversarial examples.

We propose a novel adversarial neural pruning and vulnerability suppression loss, as a defense mechanism to achieve adversarial robustness as well as a means of achieving a memory and computationefficient deep neural networks.

We observe that the latent features of deep networks have a varying degree of distortion/robustness to the adversarial perturbations to the input and formally defined the vulnerability and robustness of a latent feature.

This observation suggests that we can increase the robustness of the model by pruning out vulnerable latent features and by minimizing the vulnerability of the latent features, we show that sparsification thus leads to certain degree of robustness over the base network for this obvious reason.

We further propose a Bayesian formulation that trains the pruning mask in an adversarial training, such that the obtained neurons are beneficial both for the accuracy of the clean and adversarial inputs.

Experimental results on a range of architectures with multiple datasets demonstrate that our adversarial pruning is effective in improving the model robustness.

Further qualitative analysis shows that our method obtains more interpretable latent features compared to standard counterparts, suppresses feature-level distortions in general while zeroing out perturbations at many of them, and obtains smooth loss surface.

In this section, we prove the Equation 7 for beta bernoulli dropout.

Let W be the parameters for the neural network and let z n ∈ {0, 1} K be the mask sampled from the finite-dimensional beta-Bernoulli prior to be applied for the n-th observation x n .

The generative process of the bayesian neural network can be modelled as:

The goal is to compute the posterior distribution p(W, Z, π|D) and we approximate this posterior using an approximate variational distribution q(W, Z, π|X) of known parametric form.

We conduct computationally efficient point-estimate for W to get the single value W, with the weight decay regularization arising from the zero-mean Gaussian prior.

For π, we use the Kumaraswamy distribution (Kumaraswamy, 1980) with parameters a and b. and z k is sampled by reparametrization with continuous relaxation following :

where τ is a temperature continuous relaxation, u ∼ unif [0, 1], and sgm(x) = 1 1+e −x .

The KLdivergence between the prior and the variational distribution in closed form can then be defined as follows Nalisnick & Smyth (2016) ; :

where γ is Euler-Mascheroni constant and Ψ(.) is the digamma function.

We can use Stochastic Gradient Variational Bayes (SGVB) to minimize the KL divergence between the variational distribution q(Z; π) of known parametric form and the true posterior p(Z|π).

As we know from the SGVB framework minimizing the KL is equal to maximizing the evidence lower bound which can be done as follows:

where the first term measures the log-likelihood of the adversarial samples w.r.t.

q(Z; π) and the second term regularizes q(Z; π) so it doesn't deviate too much from the prior distribution.

We further extend the idea of Adversarial Neural Pruning to variational information bottleneck.

Variational information bottleneck Dai et al. (2018) uses information theoretic bound to reduce the redundancy between adjacent layers.

Consider a distribution D of N i.i.d samples (x, y) input to a neural network with L layers with the network hidden layer activations as

where h i ∈ R ri .

Let p(h i |h i−1 ) define the conditional probability and I(h i ; h i−1 ) define the mutual information between h i and h i−1 for every hidden layer in the network.

For every hidden layer h i , we would like to minimize the information bottleneck Tishby et al. (2000) I(h i ; h i−1 ) to remove interlayer redundancy, while simultaneously maximizing the mutual information I(h i ; y) between h i and the output y to encourage accurate predictions of adversarial examples.

The layer-wise energy L i can be written as:

where γ ≥ 0 is a coefficient that determines the strength of the bottleneck that can be defined as the degree to which we value compression over robustness.

The output layer approximates the true distribution p(y|h L ) via some tractable alternative q(y|h L ).

Let x adv be the adversarial example for l ∞ -bounded perturbations.

i.e. = {x + δ = ||δ|| ∞ ≤ } where δ is the added perturbation from the set of allowed perturbations for each example.

Using variational bounds, we can invoke the upper bound as: Equation 13 is composed of two terms, the first is the KL divergence between p(h i |h i−1 ) and q(h i ), which approximates information extracted by h i from h i−1 and the second term represents constancy with respect to the adversarial data distribution.

In order to optimize Equation 13, we can define the parametric form for the distributions p(h i |h i−1 ) and q(h i ) as follow:

where ξ i is an unknown vector of variances that can be learned from data.

The gaussian assumptions help us to get an interpretable, closed-form approximation for the KL term from Equation 13, which allows us to directly optimize ξ i out of the model.

The final variational information bottleneck can thus be obtained using Equation 15:

In this section, we describe our experimental settings for all the experiments.

We follow the two-step pruning procedure where we pretrain all the networks using the standard-training procedure followed by network sparsification using various sparsification methods.

We train each model with 200 epochs with a fixed batch size of 64 and show the results averaged over five runs.

Our pretrained standard Lenet 5-Caffe baseline model reaches over 99.29% accuracy on MNIST and VGG-16 reaches 92.65% and 67.1% on CIFAR-10 and CIFAR-100 respectively after 200 epochs.

We use Adam Kingma & Ba (2015) with the learning rate for the weights to be 0.1 times smaller than those for the variational parameters as in Neklyudov et al. (2017) ; .

For Beta-Bernoulli Dropout, we set α/K = 10 −4 for all the layers and prune the neurons/filters whose expected drop probability are smaller than a fixed threshold 10 −3 as originally proposed in the paper.

Due to the length limit of our paper, some results are illustrated here.

We validate all the models with respect to three metrics for compression ratio and model complexity: i) Run-time memory footprint (Memory) -The ratio of space for storing hidden feature maps during run-time in pruned-network versus original model.

ii) Floating point operations (xFLOPs) -The ratio of the number of floating point operations for original model versus pruned-network.

iii) Model size (Sparsity) -The ratio of the number of zero units in original model versus compressed network.

All the values are measured by computing mean and standard deviation across 3 trials upon randomly chosen seeds, respectively.

We also report the clean accuracy, vulnerability metric (Equation 2) and accuracy on adversarial examples generated from l ∞ white box and black box attack .

Table Appendix-1 also shows the results of Adversarial Neural Pruning with Variational Information Bottleneck.

We can observe that both ANP (BBD) and ANP (VIB) outperform the base adversarial training for robustness of adversarial examples while also achieving memory and computation efficiency.

We emphasize that ANP can similarly be extended to any existing or future sparsification method to improve performance.

Table Appendix -2 shows the number of units for all the baselines and our proposed method.

Figure D. 2 shows the histogram of the feature vulnerability for various datasets.

We can consistently observe that standard Bayesian pruning zeros out some of the distortions, adversarial training reduces the distortion level of all the features and adversarial neural pruning does both, with the largest number of features with zero distortion and low distortion level in general which confirms that our adversarial neural pruning works successfully as a defense against adversarial attacks.

of robust and vulnerable features in the latent-feature space from our paper.

Figure D .1 shows the visualization of robust and vulnerable features in the latent space for adversarial training.

It is important to observe that adversarial training also contains features with high vulnerability (vulnerable feature) and features with less vulnerability (robust feature) which align with our observation that the latent features have a varying degree of susceptibility to adversarial perturbations to the input.

As future work, we plan to explore more effective ways to suppress perturbation at the intermediate latent features of deep networks.

<|TLDR|>

@highlight

We propose a novel method for suppressing the vulnerability of latent feature space to achieve robust and compact networks.

@highlight

This paper proposes "adversarial neural pruning" method of training a pruning mask and a new vulnerability suppression loss to improve accuracy and adversarial robustness.