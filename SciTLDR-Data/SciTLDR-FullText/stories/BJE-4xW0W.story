We introduce causal implicit generative models (CiGMs): models that allow sampling from not only the true observational but also the true interventional distributions.

We show that adversarial training can be used to learn a CiGM, if the generator architecture is structured based on a given causal graph.

We consider the application of conditional and interventional sampling of face images with binary feature labels, such as mustache, young.

We preserve the dependency structure between the labels with a given causal graph.

We devise a two-stage procedure for learning a CiGM over the labels and the image.

First we train a CiGM over the binary labels using a  Wasserstein GAN where the generator neural network is consistent with the causal graph between the labels.

Later, we combine this with a conditional GAN to generate images conditioned on the binary labels.

We propose two new conditional GAN architectures: CausalGAN and CausalBEGAN.

We show that the optimal generator of the CausalGAN, given the labels, samples from the image distributions conditioned on these labels.

The conditional GAN combined with a trained CiGM for the labels is then a CiGM over the labels and the generated image.

We show that the proposed architectures can be used to sample from observational and interventional image distributions, even for interventions which do not naturally occur in the dataset.

An implicit generative model BID7 ) is a mechanism that can sample from a probability distribution without an explicit parameterization of the likelihood.

Generative adversarial networks (GANs) arguably provide one of the most successful ways to train implicit generative models.

GANs are neural generative models that can be trained using backpropagation to sample from very high dimensional nonparametric distributions (Goodfellow et al. (2014) ).

A generator network models the sampling process through feedforward computation given a noise vector.

The generator output is constrained and refined through feedback by a competitive adversary network, called the discriminator, that attempts to distinguish between the generated and real samples.

The objective of the generator is to maximize the loss of the discriminator (convince the discriminator that it outputs samples from the real data distribution).

GANs have shown tremendous success in generating samples from distributions such as image and video BID20 ).An extension of GANs is to enable sampling from the class conditional data distributions by feeding class labels to the generator alongside the noise vectors.

Various neural network architectures have been proposed for solving this problem BID6 ; BID10 ; Antipov et al. Figure 1: Observational and interventional samples from CausalBEGAN.

Our architecture can be used to sample not only from the joint distribution (conditioned on a label) but also from the interventional distribution, e.g., under the intervention do(M ustache = 1).

The two distributions are clearly different since P(M ale = 1|M ustache = 1) = 1 and P(Bald = 1|M ale = 0) = 0 in the data distribution P.(2017)).

However, these architectures do not capture the dependence between the labels.

Therefore, they do not have a mechanism to sample images given a subset of the labels, since they cannot sample the remaining labels.

In this paper, we are interested in extending the previous work on conditional image generation by i) capturing the dependence between labels and ii) capturing the causal effect between labels.

We can think of conditional image generation as a causal process: Labels determine the image distribution.

The generator is a non-deterministic mapping from labels to images.

This is consistent with the causal graph "Labels cause the Image", denoted by L → I, where L is the random vector for labels and I is the image random variable.

Using a finer model, we can also include the causal graph between the labels, if available.

As an example, consider the causal graph between Gender (G) and Mustache (M ) labels.

The causal relation is clearly Gender causes Mustache, denoted by the graph G → M .

Conditioning on Gender = male, we expect to see males with or without mustaches, based on the fraction of males with mustaches in the population.

When we condition on Mustache = 1, we expect to sample from males only since the population does not contain females with mustaches.

In addition to sampling from conditional distributions, causal models allow us to sample from various different distributions called interventional distributions.

An intervention is an experiment that fixes the value of a variable in a causal graph.

This affects the distributions of the descendants of the intervened variable in the graph.

But unlike conditioning, it does not affect the distribution of its ancestors.

For the same causal graph, intervening on Mustache = 1 would not change the distribution of Gender.

Accordingly, the label combination (Gender = female, Mustache = 1) would appear as often as Gender = female after the intervention.

Please see Figure 1 for some of our conditional and interventional samples, which illustrate this concept on the Bald and Mustache variables.

In this work we propose causal implicit generative models (CiGM): mechanisms that can sample not only from the correct joint probability distributions but also from the correct conditional and interventional probability distributions.

Our objective is not to learn the causal graph: we assume that the true causal graph is given to us.

We show that when the generator structure inherits its neural connections from the causal graph, GANs can be used to train causal implicit generative models.

We use Wasserstein GAN (WGAN) (Arjovsky et al. (2017) ) to train a CiGM for binary image labels, as the first step of a two-step procedure for training a CiGM for the images and image labels.

For the second step, we propose two novel conditional GANs called CausalGAN and CausalBEGAN.

We show that the optimal generator of CausalGAN can sample from the true conditional distributions (see Theorem 1).We show that combining CausalGAN with a CiGM on the labels yields a CiGM on the labels and the image, which is formalized in Corollary 1 in Section 5.

Our contributions are as follows:• We observe that adversarial training can be used after structuring the generator architecture based on the causal graph to train a CiGM.

We empirically show that WGAN can be used to learn a CiGM that outputs essentially discrete 1 labels, creating a CiGM for binary labels.• We consider the problem of conditional and interventional sampling of images given a causal graph over binary labels.

We propose a two-stage procedure to train a CiGM over the binary labels and the image.

As part of this procedure, we propose a novel conditional GAN architecture and loss function.

We show that the global optimal generator provably samples from the class conditional distributions.• We propose a natural but nontrivial extension of BEGAN to accept labels: using the same motivations for margins as in BEGAN (Berthelot et al. (2017) ), we arrive at a "margin of margins" term.

We show empirically that this model, which we call CausalBEGAN, produces high quality images that capture the image labels.• We evaluate our CiGM training framework on the labeled CelebA data BID2 ).We empirically show that CausalGAN and CausalBEGAN can produce label-consistent images even for label combinations realized under interventions that never occur during training, e.g., "woman with mustache" 2 .

Using a GAN conditioned on the image labels has been proposed before: In BID6 , authors propose conditional GAN (CGAN): They extend generative adversarial networks to the setting where there is extra information, such as labels.

Image labels are given to both the generator and the discriminator.

In BID10 , authors propose ACGAN: Instead of receiving the labels as input, the discriminator is now tasked with estimating the label.

In , the authors compare the performance of CGAN and ACGAN and propose an extension to the semi-supervised setting.

In BID15 , authors propose a new architecture called InfoGAN, which attempts to maximize a variational lower bound of mutual information between the inputs given to the generator and the image.

To the best of our knowledge, the existing conditional GANs do not allow sampling from label combinations that do not appear in the dataset BID18 ).BiGAN (Donahue et al. (2017b) ) and ALI (Dumoulin et al. (2017) ) extend the standard GAN framework by also learning a mapping from the image space to a latent space.

In CoGAN BID1 ) the authors learn a joint distribution over an image and its binary label by enforcing weight sharing between generators and discriminators.

SD-GAN (Donahue et al. (2017a) ) is a similar architecture which splits the latent space into "Identity" and "Observation" portions.

To generate faces of the same person, one can then fix the identity portion of the latent code.

If we consider the "Identity" and "Observation" codes to be the labels then SD-GAN can be seen as an extension of BEGAN to labels.

This is, to the best of our knowledge, the only extension of BEGAN to accept labels before CausalBEGAN.

It is not trivial to extend CoGAN and SD-GAN to more than two labels.

Authors in BID0 use CGAN of BID6 with a one-hot encoded vector that encodes the age interval.

A generator conditioned on this one-hot vector can then be used for changing the age attribute of a face image.

Another application of generative models is in compressed sensing: Authors in Bora et al. (2017) give compressed sensing guarantees for recovering a vector, if the data lies close to the output of a trained generative model.

Using causal principles for deep learning and using deep learning techniques for causal inference has been recently gaining attention.

In BID3 , the authors observe the connection between GAN layers, and structural equation models.

Based on this observation, they use CGAN BID6 ) to learn the causal direction between two variables from a dataset.

In BID5 , the authors propose using a neural network in order to discover the causal relation between image class labels based on static images.

In Bahadori et al. (2017) , authors propose a new regularization for training a neural network, which they call causal regularization, in order to assure that the model is predictive in a causal sense.

In a very recent work Besserve et al. (2017) , authors point out the connection of GANs to causal generative models.

However they see image as a cause of the neural net weights, and do not use labels.

In an independent parallel work, authors in Goudet et al. (2017) propose using neural networks for learning causal graphs.

Similar to us, they also use neural connections to mimic structural equations, but for learning the causal graph.

In this section, we give a brief introduction to causality.

Specifically, we use Pearl's framework BID11 ), i.e., structural causal models (SCMs), which uses structural equations and directed acyclic graphs between random variables to represent a causal model.

Consider two random variables X, Y .

Within the SCM framework and under the causal sufficiency assumption 3 , X causes Y means that there exists a function f and some unobserved random variable E, independent from X, such that the value of Y is determined based on the values of X and E through the function f , i.e., Y = f (X, E).

Unobserved variables are also called exogenous.

The causal graph that represents this relation is X → Y .

In general, a causal graph is a directed acyclic graph implied by the structural equations: The parents of a node X i in the causal graph, shown by P a i , represent the causes of that variable.

The causal graph can be constructed from the structural equations as follows: The parents of a variable are those that appear in the structural equation that determines the value of that variable.

Formally, a structural causal model is a tuple M = (V, E, F, P E (.)) that contains a set of functions F = {f 1 , f 2 , . . .

, f n }, a set of random variables V = {X 1 , X 2 , . . .

, X n }, a set of exogenous random variables E = {E 1 , E 2 , . . .

, E n }, and a product probability distribution over the exogenous variables P E .

The set of observable variables V has a joint distribution implied by the distribution of E, and the functional relations F. The causal graph D is then the directed acyclic graph on the nodes V, such that a node X j is a parent of node X i if and only if X j is in the domain of f i , i.e., X i = f i (X j , S, E i ), for some S ⊂ V .

See the Appendix for more details.

An intervention is an operation that changes the underlying causal mechanism, hence the corresponding causal graph.

An intervention on X i is denoted as do(X i = x i ).

It is different from conditioning on X i in the following way: An intervention removes the connections of node X i to its parents, whereas conditioning does not change the causal graph from which data is sampled.

The interpretation is that, for example, if we set the value of X i to 1, then it is no longer determined through the function f i (P a i , E i ).

An intervention on a set of nodes is defined similarly.

The joint distribution over the variables after an intervention (post-interventional distribution) can be calculated as follows: Since D is a Bayesian network for the joint distribution, the observational distribution can be factorized as P(x 1 , x 2 , . . .

x n ) = i∈[n] P(x i |P a i ), where the nodes in P a i are assigned to the corresponding values in {x i } i∈ [n] .

After an intervention on a set of nodes X S := {X i } i∈S , i.e., do(X S = s), the post-interventional distribution is given by i∈[n]\S P(x i |P a S i ), where P a S i represents the following assignment: DISPLAYFORM0 In general it is not possible to identify the true causal graph for a set of variables without performing experiments or making additional assumptions.

This is because there are multiple causal graphs that allow the same joint probability distribution even for two variables BID17 ).

This paper does not address the problem of learning the causal graph: We assume that the causal graph is given to us, and we learn a causal model, i.e., the functions comprising the structural equations for some choice of exogenous variables 5 .

There is significant prior work on learning causal graphs that could be used before our method BID17 Heckerman (1995) BID13 ; Kocaoglu et al. (2017b; a) ).

When the true causal graph is unknown using a Bayesian network that respects the conditional independences in the data allows us to sample from the correct observational distributions.

We explore the effect of the used Bayesian network in Section 8.10, 8.11.

Implicit generative models can sample from the data distribution.

However they do not provide the functionality to sample from interventional distributions.

We propose causal implicit generative models, which provide a way to sample from both observational and interventional distributions.

We show that generative adversarial networks can also be used for training causal implicit generative models.

Consider the simple causal graph X → Z ← Y .

Under the causal sufficiency assumption, this model can be written as DISPLAYFORM0 DISPLAYFORM1 is useful: In the GAN training framework, generator neural network connections can be arranged to reflect the causal graph structure.

Please see FIG3 for this architecture.

The feedforward neural networks can be used to represent the functions f X , f Y , f Z .

The noise terms (N X , N Y , N Z ) can be chosen as independent, complying with the condition that (E X , E Y , E Z ) are jointly independent.

Note that although we do not know the distributions of the exogenous variables, for a rich enough function class, we can use Gaussian distributed variables BID8 ) N X , N Y , N Z .

Hence this feedforward neural network can be used to represents the causal models with graph DISPLAYFORM2 The following proposition is well known in the causality literature.

It shows that given the true causal graph, two causal models that have the same observational distribution have the same interventional distributions for any intervention.

P V and Q V stands for the distributions induced on the set of variables in V by P N1 and Q N2 , respectively.

DISPLAYFORM3 be two causal models, where P N1 (.), Q N2 (.) are strictly positive densities.

If DISPLAYFORM4 We have the following definition, which ties a feedforward neural network with a causal graph: Definition 1.

Let Z = {Z 1 , Z 2 , . . .

, Z m } be a set of mutually independent random variables.

A feedforward neural network G that outputs the vector DISPLAYFORM5 where P a i are the set of parents of i in D, and Z Si := {Z j : j ∈ S i } are collections of subsets of Z such that DISPLAYFORM6 Based on the definition, we can define causal implicit generative models as follows: Definition 2 (CiGM).

A feedforward neural network G with output DISPLAYFORM7 is called a causal implicit generative model for the causal model DISPLAYFORM8 We propose using adversarial training where the generator neural network is consistent with the causal graph according to Definition 1, which is explained in the next section.

CiGMs can be trained with samples from a joint distribution given the causal graph between the variables.

However, for the application of image generation with binary labels, we found it difficult to simultaneously learn the joint label and image distribution 6 .

For this application, we focus on DISPLAYFORM0

Figure 3: CausalGAN architecture: Causal controller is a pretrained causal implicit generative model for the image labels.

Labeler is trained on the real data, Anti-Labeler is trained on generated data.

Generator minimizes Labeler loss and maximizes Anti-Labeler loss.dividing the task of learning a CiGM into two subtasks: First, we train a generative model over the labels, then train a generative model for the images conditioned on the labels.

For this training to be consistent with the causal structure, we assume that the image node is always the sink node of the causal graph for image generation problems (Please see FIG6 in Appendix).

As we show next, our new architecture and loss function (CausalGAN) assures that the optimum generator outputs the label conditioned image distributions, under the assumption that the joint probability distribution over the labels is strictly positive 7 .

Then for a strictly positive joint distribution between labels and the image, combining CiGM for only the labels with a label-conditioned image generator gives a CiGM for images and labels (see Corollary 1).

First we describe the adversarial training of a CiGM for binary labels.

This generative model, which we call the Causal Controller, will be used for controlling which distribution the images will be sampled from when intervened or conditioned on a set of labels.

As in Section 4, we structure the Causal Controller network to sequentially produce labels according to the causal graph.

Since our theoretical results hold for binary labels, we prefer a generator which can sample from an essentially discrete label distribution 8 .

However, the standard GAN training is not suited for learning a discrete distribution, since Jensen-Shannon divergence requires the support to be the same for giving meaningful gradients, which is harder with discrete data distributions.

To be able to sample from a discrete distribution, we employ WGAN (Arjovsky et al. FORMULA8 ).

We used the model of Gulrajani et al. (2017) , where the Lipschitz constraint on the gradient is replaced by a penalty term in the loss.

As part of the two-step process proposed in Section 4 for learning a CiGM over the labels and the image variables, we design a new conditional GAN architecture to generate the images based on the labels of the Causal Controller.

Unlike previous work, our new architecture and loss function assures that the optimum generator outputs the label conditioned image distributions.

We use a pretrained Causal Controller which is not further updated.

Labeler and Anti-Labeler: We have two separate labeler neural networks.

The Labeler is trained to estimate the labels of images in the dataset.

The Anti-Labeler is trained to estimate the labels of the images sampled from the generator, where image labels are those produced by the Causal Controller.

Generator: The objective of the generator is 3-fold: producing realistic images by competing with the discriminator, producing images consistent with the labels by minimizing the Labeler loss and avoiding unrealistic image distributions that are easy to label by maximizing the Anti-Labeler loss.

The most important distinction of CausalGAN with the existing conditional GAN architectures is that it uses an Anti-Labeler network in addition to a Labeler network.

Notice that the theoretical guarantee we develop in Section 5.2.3 does not hold without the Anti-Labeler.

Intuitively, the Anti-Labeler loss discourages the generator network to output only few typical faces for a fixed label combination.

This is a phenomenon that we call label-conditioned mode collapse.

Minibatch-features are one of the most popular techniques used to avoid mode-collapse BID15 ).

However, the diversity within a batch of images due to different label combinations can make this approach ineffective for combating label-conditioned mode collapse.

This effect is most prominent for rare label combinations.

In general, using Anti-Labeler helps with faster convergence.

Please see Section 9.4 in the Appendix for results.

We present the results for a single binary label l. The results can be extended to more labels.

For a single binary label l and the image x, we use P r (l, x) for the data distribution between the image and the labels.

Similarly P g (l, x) denotes the joint distribution between the labels given to the generator and the generated images.

In our analysis we assume a perfect Causal Controller 9 and use the shorthand DISPLAYFORM0 , and D LG (.) are the mappings due to generator, discriminator, Labeler, and Anti-Labeler respectively.

The generator loss function of CausalGAN contains label loss terms, the GAN loss in Goodfellow et al. FORMULA8 , and an added loss term due to the discriminator.

With the addition of this term to the generator loss, we are able to prove that the optimal generator outputs the class conditional image distribution.

This result is also true for multiple binary labels, which is shown in the Appendix.

For a fixed generator, Anti-Labeler solves the following optimization problem: DISPLAYFORM1 The Labeler solves the following optimization problem: DISPLAYFORM2 For a fixed generator, the discriminator solves the following optimization problem: DISPLAYFORM3 For a fixed discriminator, Labeler and Anti-Labeler, the generator solves the following problem: DISPLAYFORM4

We show that the best CausalGAN generator for the given loss function samples from the class conditional image distribution when Causal Controller samples from the true label distribution and the discriminator and labeler networks always operate at their optimum.

We show this result for the case of a single binary label l ∈ {0, 1}. The proof can be extended to multiple binary variables, which is given in the Appendix.

As far as we are aware of, this is the only conditional generative adversarial network architecture with this guarantee after CGAN 10 .First, we find the optimal discriminator for a fixed generator.

Note that in (4), the terms that the discriminator can optimize are the same as the GAN loss in Goodfellow et al. (2014) .

Hence the optimal discriminator behaves the same.

To characterize the optimum discriminator, labeler and anti-labeler, we have Proposition 2, Lemma 1 and Lemma 2 given in the appendix.

Let C(G) be the generator loss for when the discriminator, Labeler and Anti-Labeler are at the optimum.

Then the generator that minimizes C(G) samples from the class conditional distributions: DISPLAYFORM0 Then the global minimum of the virtual training criterion C(G) is achieved if and only if P g (l, x) = P r (l, x), i.e., if and only if given a label l, generator output G(z, l) has the same distribution as the class conditional image distribution P r (x|l).Now we can show that our two stage procedure can be used to train a causal implicit generative model for any causal graph where the Image variable is a sink node, captured by the following corollary.

L, I, Z 1 , Z 2 represent the space of labels, images, and noise variables, respectively.

Corollary 1.

Suppose C : Z 1 → L is a causal implicit generative model for the causal graph D = (V, E) where V is the set of image labels and the observational joint distribution over these labels are strictly positive.

Let G : L × Z 2 → I be a generator that can sample from the image distribution conditioned on the given label combination L ∈ L. Then DISPLAYFORM1 In Theorem 1 we show that the optimum generator samples from the class conditional distributions given a single binary label.

Our objective is to extend this result to the case with d binary labels.

First we show that if the Labeler and Anti-Labeler are trained to output 2 d scalars, each interpreted as the posterior probability of a particular label combination given the image, then the minimizer of C(G) samples from the class conditional distributions given d labels.

This result is shown in Theorem 2 in the appendix.

However, when d is large, this architecture may be hard to implement.

To resolve this, we propose an alternative architecture, which we implement for our experiments: We extend the single binary label setup and use cross entropy loss terms for each label.

This requires Labeler and Anti-Labeler to have only d outputs.

However, although we need the generator to capture the joint label posterior given the image, this only assures that the generator captures each label's posterior distribution, i.e., P r (l i |x) = P g (l i |x) (Proposition 3).

This, in general, does not guarantee that the class conditional distributions will be true to the data distribution.

However, for many joint distributions of practical interest, where the set of labels are completely determined by the image 11 , we show that this guarantee implies that the joint label posterior will be true to the data distribution, implying that the optimum generator samples from the class conditional distributions.

Please see Section 8.7 for the formal results and more details.

Remark: Note that the trained causal implicit generative models can also be used to sample from the counterfactual distributions if the exogenous noise terms are known.

Counterfactual sampling require conditioning on an event and sampling from the push-forward of the posterior distributions of the exogenous noise terms under the interventional causal graph due to a possible intervention.

This can be done through rejection sampling to observe the evidence, holding the exogenous noise terms consistent with the observed evidence and interventional sampling afterwards.

In this section, we sketch a simple, but non-trivial extension of BEGAN where we feed image labels to the generator, leaving the details to the Appendix (Section 8.8).

To accommodate interventional sampling, we again use the Causal Controller to produce labels.

In terms of architecture modifications, we use a Labeler network with a dual purpose: to label real images well and generated images poorly.

This network can be seen as both analogous to the original two-roled BEGAN discriminator and analogous to the CausalGAN Labeler and Anti-Labeler.

In terms of margin modifications, we are motivated by the following observations: (1) Just as a better trained BEGAN discriminator creates more useful gradients for image quality, (2) a better trained Labeler is a prerequisite for meaningful gradients for label quality.

Finally, (3) label gradients are most informative when the image quality is high.

Each observation suggests a margin term; the final observation suggests a (necessary) margin of margins term comparing the first two margins.

In this section, we train CausalGAN and CausalBEGAN on the CelebA Causal Graph given in FIG6 .

For this, we first trained the Causal Controller (See Section 8.11 for Causal Controller results.) on 11 The dataset we are using arguably satisfies this condition.

Since M ale → M ustache in CelebA Causal Graph, we do not expect do(M ustache = 1) to affect the probability of M ale = 1, i.e., P(M ale = 1|do(M ustache = 1)) = P(M ale = 1) = 0.42.

Accordingly, the top row shows both males and females with mustaches, even though the generator never sees the label combination {M ale = 0, M ustache = 1} during training.

The bottom row of images sampled from the conditional distribution P(.|M ustache = 1) shows only male images.

Since M ale → M ustache in CelebA Causal Graph, we do not expect do(M ustache = 1) to affect the probability of M ale = 1, i.e., P(M ale = 1|do(M ustache = 1)) = P(M ale = 1) = 0.42.

Accordingly, the top row shows both males and females with mustaches, even though the generator never sees the label combination {M ale = 0, M ustache = 1} during training.

The bottom row of images sampled from the conditional distribution P(.|M ustache = 1) shows only male images.

We proposed a novel generative model with label inputs.

In addition to being able to create samples conditioned on labels, our generative model can also sample from the interventional distributions.

Our theoretical analysis provides provable guarantees about correct sampling under such interventions.

Top: Intervene Narrow Eyes=1, Bottom: Condition Narrow Eyes=1Figure 7: Intervening/Conditioning on Narrow Eyes label in CelebA Causal Graph with CausalBEGAN.

Since Smiling → Narrow Eyes in CelebA Causal Graph, we do not expect do(Narrow Eyes = 1) to affect the probability of Smiling = 1, i.e., P(Smiling = 1|do(Narrow Eyes = 1)) = P(Smiling = 1) = 0.48.

However on the bottom row, conditioning on Narrow Eyes = 1 increases the proportion of smiling images (From 0.48 to 0.59 in the dataset), although 10 images may not be enough to show this difference statistically.

As a rare artifact, in the dark image in the third column the generator appears to rule out the possibility of Narrow Eyes = 0 instead of demonstrating Narrow Eyes = 1.Causality leads to generative models that are more creative since they can produce samples that are different from their training samples in multiple ways.

We have illustrated this point for two models (CausalGAN and CausalBEGAN).

We thank Ajil Jalal for the helpful discussions.

This research has been supported by NSF Grants CCF, 1407278, 1422549, 1618689, 1564167, DMS 1723052, ARO YIP W911NF-14-1-0258, NVIDIA Corporation and ONR N000141512009.

Formally, a structural causal model is a tuple M = (V, E, F, P E (.)) that contains a set of functions F = {f 1 , f 2 , . . .

, f n }, a set of random variables V = {X 1 , X 2 , . . .

, X n }, a set of exogenous random variables E = {E 1 , E 2 , . . .

, E n }, and a probability distribution over the exogenous variables P E 12 .

The set of observable variables V has a joint distribution implied by the distributions of E, and the functional relations F. This distribution is the projection of P E onto the set of variables V and is shown by P V .

The causal graph D is then the directed acyclic graph on the nodes V, such that a node X j is a parent of node X i if and only if X j is in the domain of f i , i.e., X i = f i (X j , S, E i ), for some S ⊂ V .

The set of parents of variable X i is shown by P a i .

D is then a Bayesian network for the induced joint probability distribution over the observable variables V. We assume causal sufficiency: Every exogenous variable is a direct parent of at most one observable variable.

Note that D 1 and D 2 are the same causal Bayesian networks BID11 .

Under the causal sufficiency assumption, interventional distributions for causal Bayesian networks can be directly calculated from the conditional probabilities and the causal graph.

Thus, M 1 and M 2 have the same interventional distributions.

In this section we use P r (l, x) for the joint data distribution over a single binary label l and the image x.

We use P g (l, x) for the joint distribution over the binary label l fed to the generator and the image x produced by the generator.

Later in Theorem 2, l is generalized to be a vector.

The following restates Proposition 1 from Goodfellow et al. (2014) as it applies to our discriminator: Proposition 2 (Goodfellow et al. FORMULA8 ).

For fixed G, the optimal discriminator D is given by DISPLAYFORM0 .Second, we identify the optimal Labeler and Anti-Labeler.

We have the following lemma: Lemma 1.

The optimum Labeler has D LR (x) = P r (l = 1|x).Proof.

The proof follows the same lines as in the proof for the optimal discriminator.

Consider the objective DISPLAYFORM1 Since 0 < D LR < 1, D LR that maximizes (3) is given by DISPLAYFORM2 ρP r (x|l = 1) DISPLAYFORM3 Similarly, we have the corresponding lemma for Anti-Labeler: Lemma 2.

For a fixed generator with x ∼ P g (x), the optimum Anti-Labeler has D LG (x) = P g (l = 1|x).Proof.

Proof is the same as the proof of Lemma 1.

12 The definition provided here assumes causal sufficiency, i.e., there are no exogenous variables that affect more than one observable variable.

Under causal sufficiency, Pearl's model assumes that the distribution over the exogenous variables is a product distribution, i.e., exogenous variables are mutually independent. .

We also add edges (see Appendix Section 8.10) to form the complete graph "cG1".

We also make use of the graph rcG1, which is obtained by reversing the direction of every edge in cG1.

Theorem 1.

Define C(G) as the generator loss for when discriminator, Labeler and Anti-Labeler are at their optimum.

Assume P g (l) = P r (l), i.e., the Causal Controller samples from the true label distribution.

Then the global minimum of the virtual training criterion C(G) is achieved if and only if P g (l, x) = P r (l, x), i.e., if and only if given a label l, generator output G(z, l) has the same distribution as the class conditional image distribution P r (x|l).Proof.

For a fixed generator, the optimum Labeler D * LR , Anti-Labeler D *LG , and discriminator D * obey the following relations by Prop 2, Lemma 1, and Lemma 2: DISPLAYFORM0 LG (x) = P g (l = 1|x).Then substitution into the generator objective in (5) yields DISPLAYFORM1 where KL is the Kullback-Leibler divergence, which is minimized if and only if P g = P d jointly over labels and images.

FORMULA8 is due to the fact that P r (l = 1) = P g (l = 1) = ρ.

Corollary 1.

Suppose C : Z 1 → L is a causal implicit generative model for the causal graph D = (V, E) where V is the set of image labels and the observational joint distribu-tion over these labels are strictly positive.

Let G : L × Z 2 → I be a generator that can sample from the image distribution conditioned on the given label combination L ∈ L. Then DISPLAYFORM0 Proof.

Since C is a causal implicit generative model for the causal graph D, by definition it is consistent with the causal graph D. Since in a conditional GAN, generator G is given the noise terms and the labels, it is easy to see that the concatenated generator neural network G(C(Z 1 ), Z 2 ) is consistent with the causal graph D , where D = (V ∪ {Image}, E ∪ {(V 1 , Image), (V 2 , Image), . . . (V n , Image)}).

Assume that C and G are perfect, i.e., they sample from the true label joint distribution and conditional image distribution.

Then the joint distribution over the generated labels and image is the true distribution since P(Image, Label) = P(Image|Label)P(Label).

By Proposition 1, the concatenated model can sample from the true observational and interventional distributions.

Hence, the concatenated model is a causal implicit generative model for graph D .

In this section, we explain the modifications required to extend the proof to the case with multiple binary labels.

The central difficulty with generalizing to a vector of labels l = (l j ) 1≤j≤d is that each labeler can only hope to learn about the posterior P(l j |x) for each j.

This is in general insufficient to characterize P r (l|x) and therefore the generator can not hope to learn the correct joint distribution.

We show two solutions to this problem.

(1) From a theoretical (but perhaps impractical) perspective each labeler can be made to estimate the probability of each of the 2 d label combinations instead of each label.

We do not adopt this in practice.

(2) If in fact the label vector is a deterministic function of the image (which seems likely for the present application), then using Labelers to estimate the probabilities of each of the d labels is sufficient to assure P g (l 1 , l 2 , . . .

, l d , x) = P r (l 1 , l 2 , . . .

, l d , x) at the minimizer of C(G).

In this section, we present the extension in (1) and present the results of FORMULA12 DISPLAYFORM0 where ρ j = P r (l = j).

We have the following Lemma: DISPLAYFORM1 , where x ∼ P r (x, l).

Then the optimum Labeler with respect to the loss in (12) has D * LR (x)[j] = P r (l = j|x).Proof.

Suppose P r (l = j|x) = 0 for a set of (label, image) combinations.

Then P r (x, l = j) = 0, hence these label combinations do not contribute to the expectation.

Thus, without loss of generality, we can consider only the combinations with strictly positive probability.

We can also restrict our attention to the functions D LR that are strictly positive on these (label,image) combinations; otherwise, loss becomes infinite, and as we will show we can achieve a finite loss.

Consider the vector D LR (x) with coordinates DISPLAYFORM2 .

The Labeler loss can be written as DISPLAYFORM3 where L x is the discrete random variable such that P(L x = j) = P r (l = j|x).

H(L x ) is the Shannon entropy of L x , and it only depends on the data.

Since KL divergence is greater than zero and p(x) is always non-negative, the loss is lower bounded by −H(L x ).

Notice that this minimum can be achieved by satisfying P(Z x = j) = P r (l = j|x).

Since KL divergence is minimized if and only if the two random variables have the same distribution, this is the unique optimum, i.e., DISPLAYFORM4 The lemma above simply states that the optimum Labeler network will give the posterior probability of a particular label combination, given the observed image.

In practice, the constraint that the coordinates sum to 1 could be satisfied by using a softmax function in the implementation.

Next, we have the corresponding loss function and lemma for the Anti-Labeler network.

The Anti-Labeler solves the following optimization problem DISPLAYFORM5 where P g (x|l = j) := P(G(z, l) = x|l = j) and ρ j = P(l = j).

We have the following Lemma:Lemma 4.

The optimum Anti-Labeler has D *LG (x)[j] = P g (l = j|x).Proof.

The proof is the same as the proof of Lemma 3, since Anti-Labeler does not have control over the joint distribution between the generated image and the labels given to the generator, and cannot optimize the conditional entropy of labels given the image under this distribution.

For a fixed discriminator, Labeler and Anti-Labeler, the generator solves the following optimization problem: DISPLAYFORM6 We then have the following theorem along the same lines as Theorem 1 showing that the optimal generator samples from the class conditional image distributions given a particular label combination:Theorem 2 (Theorem 1 formal for multiple binary labels).

Define C(G) as the generator loss as in Eqn.

16 when discriminator, Labeler and Anti-Labeler are at their optimum.

Assume P g (l) = P r (l), i.e., the Causal Controller samples from the true joint label distribution.

The global minimum of the virtual training criterion C(G) is achieved if and only if P g (l, x) = P r (l, x) for the vector of labels DISPLAYFORM7 Proof.

For a fixed generator, the optimum Labeler D * LR , Anti-Labeler D *LG , and discriminator D * obey the following relations by Prop 2, Lemma 3, and Lemma 4: DISPLAYFORM8 Then substitution into the generator objective C(G) yields Published as a conference paper at ICLR 2018 DISPLAYFORM9 where KL is the Kullback-Leibler divergence, which is minimized if and only if P g = P d jointly over labels and images.

While the previous section showed how to ensure P g (l, x) = P r (l, x) by relabeling combinations of a d binary labels as a 2 d label, this may be difficult in practice for a large number of labels and we do not adopt this approach in practice.

Instead, in this section, we provide the theoretical guarantees for the implemented CausalGAN architecture with d labels under the assumption that the relationship between the image and its labels is deterministic in the dataset, i.e., there is a deterministic function that maps an image to the corresponding label vector.

Later we show that this assumption is sufficient to gaurantee that the global optimal generator samples from the class conditional distributions.

DISPLAYFORM0 where P r (x|l j = 0) := P(X = x|l j = 0), P r (x|l j = 0) := P(X = x|l j = 0) and ρ j = P(l j = 1).

For a fixed generator, the Anti-Labeler solves the following optimization problem: DISPLAYFORM1 where P g (x|l j = 0) := P g (x|l j = 0), P g (x|l j = 0) := P g (x|l j = 0).

For a fixed discriminator, Labeler and Anti-Labeler, the generator solves the following optimization problem: DISPLAYFORM2 We have the following proposition, which characterizes the optimum generator for optimum Labeler, Anti-Labeler and Discriminator: Proposition 3.

Define C(G) as the generator loss for when discriminator, Labeler and Anti-Labeler are at their optimum obtained from (21).

The global minimum of the virtual training criterion C(G) is achieved if and only if P g (x|l i ) = P r (x|l i )∀i ∈ [d] and P g (x) = P r (x).Proof.

Proof follows the same lines as in the proof of Theorem 1 and Theorem 2 and is omitted.

Thus we have DISPLAYFORM3 However, this does not in general imply P r (x, l 1 , l 2 , . . .

, l d ) = P g (x, l 1 , l 2 , . . .

, l d ), which is equivalent to saying the generated distribution samples from the class conditional image distributions.

To guarantee the correct conditional sampling given all labels, we introduce the following assumption:We assume that the image x determines all the labels.

This assumption is very relevant in practice.

For example, in the CelebA dataset, which we use, the label vector, e.g., whether the person is a male or female, with or without a mustache, can be thought of as a deterministic function of the image.

When this is true, we can say that P r (l 1 , l 2 , . . .

, l n |x) = P r (l 1 |x)P r (l 2 |x) . . .

P r (l n |x).We need the following lemma, where kronecker delta function refers to the functions that take the value of 1 only on a single point, and 0 everywhere else: Lemma 5.

Any discrete joint probability distribution, where all the marginal probability distributions are kronecker delta functions is the product of these marginals.

Proof.

Let δ {x−u} be the kronecker delta function which is 1 if x = u and is 0 otherwise.

Consider a joint distribution p(X 1 , X 2 , . . .

, X n ), where p(X i ) = δ {Xi−ui} , ∀i ∈ [n], for some set of elements {u i } i∈ [n] .

We will show by contradiction that the joint probability distribution is zero everywhere except at (u 1 , u 2 , . . . , u n ).

Then, for the sake of contradiction, suppose for some DISPLAYFORM4 Then we can marginalize the joint distribution as DISPLAYFORM5 where the inequality is due to the fact that the particular configuration (v 1 , v 2 , . . .

, v n ) must have contributed to the summation.

However this contradicts with the fact that p(X j ) = 0, ∀X j = u j .

Hence, p(.) is zero everywhere except at (u 1 , u 2 , . . . , u n ), where it should be 1.We can now simply apply the above lemma on the conditional distribution P g (l 1 , l 2 , . . .

, l d |x).

Proposition 3 shows that the image distributions and the marginals P g (l i |x) are true to the data distribution due to Bayes' rule.

Since the vector (l 1 , . . . , l n ) is a deterministic function of x by assumption, P r (l i |x) are kronecker delta functions, and so are P g (l i |x) by Proposition 3.

Thus, since the joint P g (x, l 1 , l 2 , . . . , l d ) satisfies the condition that every marginal distribution p(l i |x) is a kronecker delta function, then it must be a product distribution by Lemma 5.

Thus we can write DISPLAYFORM6 Then we have the following chain of equalities.

DISPLAYFORM7 Thus, we also have P r (x|l 1 , l 2 , . . .

, l n ) = P g (x|l 1 , l 2 , . . .

, l n ) since P r (l 1 , l 2 , . . .

, l n ) = P g (l 1 , l 2 , . . .

, l n ), concluding the proof that the optimum generator samples from the class conditional image distributions.

In this section, we propose a simple, but non-trivial extension of BEGAN where we feed image labels to the generator.

One of the central contributions of BEGAN (Berthelot et al. (2017) ) is a control theory-inspired boundary equilibrium approach that encourages generator training only when the discriminator is near optimum and its gradients are the most informative.

The following observation helps us carry the same idea to the case with labels: Label gradients are most informative when the image quality is high.

Here, we introduce a new loss and a set of margins that reflect this intuition.

Formally, let L(x) be the average L 1 pixel-wise autoencoder loss for an image x, as in BEGAN.

Let L sq (u, v) be the squared loss term, i.e., u − v 2 2 .

Let (x, l x ) be a sample from the data distribution, where x is the image and l x is its corresponding label.

Similarly, G(z, l g ) is an image sample from the generator, where l g is the label used to generate this image.

Denoting the space of images by I, let G :

R n × {0, 1} m →

I be the generator.

As a naive attempt to extend the original BEGAN loss formulation to include the labels, we can write the following loss functions: Labeler(G(z, l g ) DISPLAYFORM0 DISPLAYFORM1 However, this naive formulation does not address the use of margins, which is extremely critical in the BEGAN formulation.

Just as a better trained BEGAN discriminator creates more useful gradients for image generation, a better trained Labeler is a prerequisite for meaningful gradients.

This motivates an additional margin-coefficient tuple (b 2 , c 2 ), as shown in FORMULA12 ).The generator tries to jointly minimize the two loss terms in the formulation in (24).

We empirically observe that occasionally the image quality will suffer because the images that best exploit the Labeler network are often not obliged to be realistic, and can be noisy or misshapen.

Based on this, label loss seems unlikely to provide useful gradients unless the image quality remains good.

Therefore we encourage the generator to incorporate label loss only when the image quality margin b 1 is large compared to the label margin b 2 .

To achieve this, we introduce a new margin of margins term, b 3 .

As a result, the margin equations and update rules are summarized as follows, where λ 1 , λ 2 , λ 3 are learning rates for the coefficients.

DISPLAYFORM2 One of the advantages of BEGAN is the existence of a monotonically decreasing scalar which can track the convergence of the gradient descent optimization.

Our extension preserves this property as we can define DISPLAYFORM3 and show that M complete decreases progressively during our optimizations.

See FIG9 .

In Section 4 we showed how a GAN could be used to train a causal implicit generative model by incorporating the causal graph into the generator structure.

Here we investigate the behavior and convergence of causal implicit generative models when the true data distribution arises from another (possibly distinct) causal graph.

We consider causal implicit generative model convergence on synthetic data whose three features {X, Y, Z} arise from one of three causal graphs: "line" X → Y → Z , "collider" X → Y ← Z, and "complete" X → Y → Z, X → Z.

For each node a (randomly sampled once) cubic polynomial in n + 1 variables computes the value of that node given its n parents and 1 uniform exogenous variable.

We then repeat, creating a new synthetic dataset in this way for each causal model and report the averaged results of 20 runs for each model.

For each of these data generating graphs, we compare the convergence of the joint distribution to the true joint in terms of the total variation distance, when the generator is structured according to a line, collider, or complete graph.

For completeness, we also include generators with no knowledge of causal structure: {f c3, f c5, f c10} are fully connected neural networks that map uniform random noise to 3 output variables using either 3,5, or 10 layers respectively.

The results are given in FIG9 .

Data is generated from line causal graph X → Y → Z (left panel), collider causal graph X → Y ← (middle panel), and complete causal graph X → Y → Z, X → Z (right panel).

Each curve shows the convergence behavior of the generator distribution, when generator is structured based on each one of these causal graphs.

We expect convergence when the causal graph used to structure the generator is capable of generating the joint distribution due to the true causal graph: as long as we use the correct Bayesian network, we should be able to fit to the true joint.

For example, complete graph can encode all joint distributions.

Hence, we expect complete graph to work well with all data generation models.

Standard fully connected layers correspond to the causal graph with a latent variable causing all the observable variables.

Ideally, this model should be able to fit to any causal generative model.

However, the convergence behavior of adversarial training across these models is unclear, which is what we are exploring with FIG9 .For the line graph data X → Y → Z, we see that the best convergence behavior is when line graph is used in the generator architecture.

As expected, complete graph also converges well, with slight delay.

Similarly, fully connected network with 3 layers show good performance, although surprisingly fully connected with 5 and 10 layers perform much worse.

It seems that although fully connected can encode the joint distribution in theory, in practice with adversarial training, the number of layers should be tuned to achieve the same performance as using the true causal graph.

Using the wrong Bayesian network, the collider, also yields worse performance.

For the collider graph, surprisingly using a fully connected generator with 3 and 5 layers shows the best performance.

However, consistent with the previous observation, the number of layers is important, and using 10 layers gives the worst convergence behavior.

Using complete and collider graphs achieves the same decent performance, whereas line graph, a wrong Bayesian network, performs worse than the two.

For the complete graph, fully connected 3 performs the best, followed by fully connected 5, 10 and the complete graph.

As we expect, line and collider graphs, which cannot encode all the distributions due to a complete graph, performs the worst and does not actually show any convergence behavior.

First, we evaluate the effect of using the wrong causal graph on an artificially generated dataset.

FIG10 shows the scatter plot for the two coordinates of a three dimensional distribution.

As we Data is generated using the causal graph X 1 → X 2 → X 3 .

(b) Generated distribution when generator causal graph is X 1 → X 2 → X 3 .

(c) Generated distribution when generator causal graph is .

Note that G1 treats Male and Young labels as independent, but does not completely fail to generate a reasonable (product of marginals) approximation.

Also note that when an edge is added Y oung → M ale, the learned distribution is nearly exact.

Note that both graphs contain the edge M ale → M ustache and so are able to learn that women have no mustaches.

DISPLAYFORM0 observe, using the correct graph gives the closest scatter plot to the original data, whereas using the wrong Bayesian network, collider graph, results in a very different distribution.

Second, we expand on the causal graphs used for experiments for the CelebA dataset.

We use a causal graph on a subset of the image labels of CelebA dataset, which we call CelebA Causal Graph (G1), illustrated in FIG6 .

The graph cG1, which is a completed version of G1, is the complete graph associated with the ordering: Young, Male, Eyeglasses, Bald, Mustache, Smiling, Wearing Lipstick, Mouth Slightly Open, Narrow Eyes.

For example, in cG1 Male causes Smiling because Male comes before Smiling in the ordering.

The graph rcG1 is formed by reversing every edge in cG1.Next, we check the effect of using the incorrect Bayesian network for the data.

The causal graph G1 generates Male and Young independently, which is incorrect in the data.

Comparison of pairwise distributions in TAB0 demonstrate that for G1 a reasonable approximation to the true distribution is still learned for {Male, Young} jointly.

For cG1 a nearly perfect distributional approximation is learned.

Furthermore we show that despite this inaccuracy, both graphs G1 and cG1 lead to Causal Controllers that never output the label combination {Female,Mustache}, which will be important later.

Wasserstein GAN in its original form (with Lipshitz discriminator) assures convergence in distribution of the Causal Controller output to the discretely supported distribution of labels.

We use a slightly modified version of Wasserstein GAN with a penalized gradient (Gulrajani et al. (2017) ).

We first demonstrate that learned outputs actually have "approximately discrete" support.

In FIG12 , we sample the joint label distribution 1000 times, and make a histogram of the (all) scalar outputs corresponding to any label.

Although FIG12 demonstrates conclusively good convergence for both graphs, TVD is not always intuitive.

For example, "how much can each marginal be off if there are 9 labels and the TVD is 0.14?".

To expand upon FIG3 where we showed that the causal controller learns the correct distribution for a pairwise subset of nodes, here we also show that both CelebA Causal Graph (G1) and the completion we define (cG1) allow training of very reasonable marginal distributions for all labels ( TAB0 ) that are not off by more than 0.03 for the worst label.

P D (L = 1) is the probability that the label is 1 in the dataset, and P G (L = 1) is the probability that the generated label is (around a small neighborhood of ) 1.

DISPLAYFORM1

We test the performance of our Wasserstein Causal Controller on a subset of the binary labels of CelebA datset.

We use the causal graph given in FIG6 .For causal graph training, first we verify that our Wasserstein training allows the generator to learn a mapping from continuous uniform noise to a discrete distribution.

FIG12 shows where the samples, averaged over all the labels in CelebA Causal Graph, from this generator appears on the real line.

The result emphasizes that the proposed Causal Controller outputs an almost discrete distribution: 96% of the samples appear in 0.05−neighborhood of 0 or 1.

Outputs shown are unrounded generator outputs.

A stronger measure of convergence is the total variational distance (TVD).

For CelebA Causal Graph (G1), our defined completion (cG1), and cG1 with arrows reversed (rcG1), we show convergence of TVD with training ( FIG12 ).

Both cG1 and rcG1 have TVD decreasing to 0, and TVD for G1 assymptotes to around 0.14 which corresponds to the incorrect conditional independence assumptions that G1 makes.

This suggests that any given complete causal graph will lead to a nearly perfect implicit causal generator over labels and that bayesian partially incorrect causal graphs can still give reasonable convergence.

In this section, we present additional CausalGAN results in FIG3 , 13.Intervening vs Conditioning on Wearing Lipstick, Top: Intervene Wearing Lipstick=1, Bottom: Condition Wearing Lipstick=1Figure 12: Intervening/Conditioning on Wearing Lipstick label in CelebA Causal Graph.

Since M ale → W earingLipstick in CelebA Causal Graph, we do not expect do(Wearing Lipstick = 1) to affect the probability of M ale = 1, i.e., P(M ale = 1|do(Wearing Lipstick = 1)) = P(M ale = 1) = 0.42.

Accordingly, the top row shows both males and females who are wearing lipstick.

However, the bottom row of images sampled from the conditional distribution P(.|Wearing Lipstick = 1) shows only female images because in the dataset P(M ale = 0|Wearing Lipstick = 1) ≈ 1.Intervening vs Conditioning on Narrow Eyes, Top: Intervene Narrow Eyes=1, Bottom: Condition Narrow Eyes=1Figure 13: Intervening/Conditioning on Narrow Eyes label in CelebA Causal Graph.

Since Smiling → Narrow Eyes in CelebA Causal Graph, we do not expect do(Narrow Eyes = 1) to affect the probability of Smiling = 1, i.e., P(Smiling = 1|do(Narrow Eyes = 1)) = P(Smiling = 1) = 0.48.

However on the bottom row, conditioning on Narrow Eyes = 1 increases the proportion of smiling images (From 0.48 to 0.59 in the dataset), although 10 images may not be enough to show this difference statistically.

In this section, we train CausalBEGAN on CelebA dataset using CelebA Causal Graph.

The Causal Controller is pretrained with a Wasserstein loss and used for training the CausalBEGAN.To first empirically justify the need for the margin of margins we introduced in (27) (c 3 and b 3 ), we train the same CausalBEGAN model setting c 3 = 1, removing the effect of this margin.

We show that the image quality for rare labels deteriorates.

Please see FIG6 in the appendix.

Then for the labels Bald, and Mouth Slightly Open, we illustrate the difference between interventional and conditional sampling when the label is 1.

FIG4 ).Intervening vs Conditioning on Bald, Top: Intervene Bald=1, Bottom: Condition Bald=1 FIG4 : Intervening/Conditioning on Bald label in CelebA Causal Graph.

Since M ale → Bald in CelebA Causal Graph, we do not expect do(Bald = 1) to affect the probability of M ale = 1, i.e., P(M ale = 1|do(Bald = 1)) = P(M ale = 1) = 0.42.

Accordingly, the top row shows both bald males and bald females.

The bottom row of images sampled from the conditional distribution P(.|Bald = 1) shows only male images because in the dataset P(M ale = 1|Bald = 1) ≈ 1.

Since Smiling → M outhSlightlyOpen in CelebA Causal Graph, we do not expect do(Mouth Slightly Open = 1) to affect the probability of Smiling = 1, i.e., P(Smiling = 1|do(Mouth Slightly Open = 1)) = P(Smiling = 1) = 0.48.

However on the bottom row, conditioning on Mouth Slightly Open = 1 increases the proportion of smiling images (From 0.48 to 0.76 in the dataset), although 10 images may not be enough to show this difference statistically.

In this section, we provide additional simulations for CausalGAN.

In Figures 16a-16d , we show the conditional image generation properties of CausalGAN by sweeping a single label from 0 to 1 while keeping all other inputs/labels fixed.

In Figure 17 , to examine the degree of mode collapse and show the image diversity, we show 256 randomly sampled images.

In this section, we provide additional simulation results for CausalBEGAN.

First we show that although our third margin term b 3 introduces complications, it can not be ignored.

FIG6 demonstrates that omitting the third margin on the image quality of rare labels.

Furthermore just as the setup in BEGAN permitted the definiton of a scalar "M", which was monotonically decreasing during training, our definition permits an obvious extension M complete (defined in 28) that preserves these properties.

See FIG9 to observe M complete decreaing monotonically during training.

We also show the conditional image generation properties of CausalBEGAN by using "label sweeps" that move a single label input from 0 to 1 while keeping all other inputs fixed FIG3 .

It is interesting to note that while generators are often implicitly thought of as continuous functions, the generator in this CausalBEGAN architecture learns a discrete function with respect to its label input parameters. (Initially there is label interpolation, and later in the optimization label interpolation becomes more step function like (not shown)).

Finally, to examine the degree of mode collapse and show the image diversity, we show a random sampling of 256 images FIG3 ).

In this section, we present the result of attempting to jointly train an implicit causal generative model for labels and the image.

This approach treats the image as part of the causal graph.

It is not clear how exactly to feed both labels and image to discriminator, but one way is to simply encode the label as a constant image in an additional channel.

We tried this for CelebA Causal Graph and observed that the image generation is not learned FIG3 ).

One hypothesis is that the discriminator focuses on labels without providing useful gradients to the image generation.

In this section, we explain the differences between implementation and theory, along with other implementation details for both CausalGAN and CausalBEGAN.

In this section, we explain the implementation details of the Wasserstein Causal Controller for generating face labels.

We used the total variation distance (TVD) between the distribution of generator and data distribution as a metric to decide the success of the models.

The gradient term used as a penalty is estimated by evaluating the gradient at points interpolated between the real and fake batches.

Interestingly, this Wasserstein approach gives us the opportunity to train the Causal Controller to output (almost) discrete labels (See FIG12 .

In practice though, we still found benefit in rounding them before passing them to the generator.

The generator architecture is structured in accordance with Section 4 based on the causal graph in FIG6 , using uniform noise as exogenous variables and 6 layer neural networks as functions mapping parents to children.

For the training, we used 25 Wasserstein discriminator (critic) updates per generator update, with a learning rate of 0.0008.

In practice, we use stochastic gradient descent to train our model.

We use DCGAN Radford et al. (2015) , a convolutional neural net-based implementation of generative adversarial networks, and extend it into our Causal GAN framework.

We have expanded it by adding our Labeler networks, training a Causal Controller network and modifying the loss functions appropriately.

Compared to DCGAN an important distinction is that we make 6 generator updates for each discriminator update on average.

The discriminator and labeler networks are concurrently updated in a single iteration.

Notice that the loss terms defined in Section 5.2.1 contain a single binary label.

In practice we feed a d-dimensional label vector and need a corresponding loss function.

We extend the Labeler and Anti-Labeler loss terms by simply averaging the loss terms for every label.

The i th coordinates of the d-dimensional vectors given by the labelers determine the loss terms for label i. Note that this is different than the architecture given in Section 8.6, where the discriminator outputs a length-2 d vector and estimates the probabilities of all label combinations given the image.

Therefore this approach does not have the guarantee to sample from the class conditional distributions, if the data distribution is not restricted.

However, for the type of labeled image dataset we use in this work, where labels seem to be completely determined given an image, this architecture is sufficient to have the same guarantees.

For the details, please see Section 8.7 in the supplementary material.

Compared to the theory we have, another difference in the implementation is that we have swapped the order of the terms in the cross entropy expressions for labeler losses.

This has provided sharper images at the end of the training.

The labels input to CausalBEGAN are taken from the Causal Controller.

We use very few parameter tunings.

We use the same learning rate (0.00008) for both the generator and discriminator and do 1 update of each simultaneously (calculating the for each before applying either).

We simply use γ 1 = γ 2 = γ 3 = 0.5.

We do not expect the model to be very sensitive to these parameter values, as we achieve good performance without hyperparameter tweaking.

We do use customized margin learning rates λ 1 = 0.001, λ 2 = 0.00008, λ 3 = 0.01, which reflect the asymmetry in how quickly the generator can respond to each margin.

For example c 2 can have much more "spiky", fast responding behavior compared to others even when paired with a smaller learning rate, although we have not explored this parameter space in depth.

In these margin behaviors, we observe that the best performing models have all three margins "active": near 0 while frequently taking small positive values.

In this section, we show results that compare the CausalGAN behavior with and without Anti-Labeler network.

In general, using Anti-Labeler allows for faster convergence.

For very rare labels, the model with Anti-Labeler provides more diverse images.

See FIG3 , 25.

@highlight

We introduce causal implicit generative models, which can sample from conditional and interventional distributions and also propose two new conditional GANs which we use for training them.

@highlight

A method of combining a casual graph, describing the dependency structure of labels with two conditional GAN architechtures that generate images conditioning on the binary label

@highlight

The authors address the issue of learning a causal model between image variables and the image itself from observational data, when given a causal structure between image labels.