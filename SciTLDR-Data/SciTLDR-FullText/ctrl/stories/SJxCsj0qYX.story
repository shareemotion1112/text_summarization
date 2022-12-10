We study the problem of alleviating the instability issue in the GAN training procedure via new architecture design.

The discrepancy between the minimax and maximin objective values could serve as a proxy for the difficulties that the alternating gradient descent encounters in the optimization of GANs.

In this work, we give new results on the benefits of multi-generator architecture of GANs.

We show that the minimax gap shrinks to \epsilon as the number of generators increases with rate O(1/\epsilon).

This improves over the best-known result of O(1/\epsilon^2).

At the core of our techniques is a novel application of Shapley-Folkman lemma to the generic minimax problem, where in the literature the technique was only known to work when the objective function is restricted to the Lagrangian function of a constraint optimization problem.

Our proposed Stackelberg GAN performs well experimentally in both synthetic and real-world datasets, improving Frechet Inception Distance by 14.61% over the previous multi-generator GANs on the benchmark datasets.

Generative Adversarial Nets (GANs) are emerging objects of study in machine learning, computer vision, natural language processing, and many other domains.

In machine learning, study of such a framework has led to significant advances in adversarial defenses BID25 BID22 and machine security BID3 BID22 .

In computer vision and natural language processing, GANs have resulted in improved performance over standard generative models for images and texts BID11 , such as variational autoencoder BID14 and deep Boltzmann machine BID20 .

A main technique to achieve this goal is to play a minimax two-player game between generator and discriminator under the design that the generator tries to confuse the discriminator with its generated contents and the discriminator tries to distinguish real images/texts from what the generator creates.

Despite a large amount of variants of GANs, many fundamental questions remain unresolved.

One of the long-standing challenges is designing universal, easy-to-implement architectures that alleviate the instability issue of GANs training.

Ideally, GANs are supposed to solve the minimax optimization problem BID11 , but in practice alternating gradient descent methods do not clearly privilege minimax over maximin or vice versa (page 35, Goodfellow (2016) ), which may lead to instability in training if there exists a large discrepancy between the minimax and maximin objective values.

The focus of this work is on improving the stability of such minimax game in the training process of GANs.

To alleviate the issues caused by the large minimax gap, our study is motivated by the so-called Stackelberg competition in the domain of game theory.

In the Stackelberg leadership model, the players of this game are one leader and multiple followers, where the leader firm moves first and then the follower firms move sequentially.

It is known that the Stackelberg model can be solved to find a subgame perfect Nash equilibrium.

We apply this idea of Stackelberg leadership model to the architecture design of GANs.

That is, we design an improved GAN architecture with multiple generators (followers) which team up to play against the discriminator (leader).

We therefore name our model Stackelberg GAN.

Our theoretical and experimental results establish that: GANs with multi-generator architecture have smaller minimax gap, and enjoy more stable training performances.

Our Contributions.

This paper tackles the problem of instability during the GAN training procedure with both theoretical and experimental results.

We study this problem by new architecture design.

Figure 1: Stackelberg GAN stabilizes the training procedure on a toy 2D mixture of 8 Gaussians.

Top Row: Standard GAN training.

It shows that several modes are dropped.

Bottom Row: Stackelberg GAN training with 8 generator ensembles, each of which is denoted by one color.

We can see that each generator exactly learns one mode of the distribution without any mode being dropped.

Under review as a conference paper at ICLR 2019 (a) Step 0 Standard GAN training.

It shows that several modes are dropped.

Bottom Row: Stackelberg GAN training with 8 generator ensembles, each of which is denoted by one color.

We can see that each generator exactly learns one mode of the distribution without any mode being dropped.•

We propose Stackelberg GAN framework of having multiple generators in the GAN architecture.

Our framework is general that can be applied to all variants of GANs, e.g., vanilla GAN, Wasserstein GAN, etc.

It is built upon the idea of jointly optimizing an ensemble of GAN losses w.r.t.

all pairs of discriminator and generator.

Differences with prior work.

Although the idea of having multiple generators in the GAN architecture is not totally new, e.g., MIX+GAN BID1 and MGAN BID13 , there are key differences between Stackelberg GAN and prior work.

a) In MGAN BID13 , various generators are combined as a mixture of probabilistic models with assumption that the generators and discriminator have enough capacity.

In contrast, in the Stackelberg GAN model we uniformly ensemble the losses of various standard GAN without any assumption on the model capacity.

b) In MIX+GAN BID1 , the losses are ensembled with learned weights and an extra regularization term, which discourages the weights being too far away from uniform.

We find it slightly unnecessary because the expressive power of each generator already allows implicit scaling of each generator.

To the contrary, in the Stackelberg GAN we apply equal weights for all generators.• We prove that the minimax duality gap shrinks as the number of generators increases (see Theorem 1 and Corollary 2).

Unlike the previous work, our result has no assumption on the expressive power of generators and discriminator, but instead depends on their non-convexity.

With extra condition on the expressive power of generators, we show that Stackelberg GAN is able to achieve ✏-approximate equilibrium with e O(1/✏) generators (see Theorem 3).

This Stackelberg GAN training with 10 generator ensembles on real images without cherry pick, where each row corresponds to one generator.

We can see that each generator exactly learns one mode of the distribution without any mode being dropped.[Pengtao: It is kind of abrupt that you say "Stackelberg GAN stabilizes the training procedure" in the beginning sentence, then the rest talks about losing mode.

In the introduction, a convincing tie between instability and mode collapse is still missing.]• We propose Stackelberg GAN framework of having multiple generators in the GAN architecture.

Our framework is general that can be applied to all variants of GANs, e.g., vanilla GAN, Wasserstein GAN, etc.

It is built upon the idea of jointly optimizing an ensemble of GAN losses w.r.t.

all pairs of discriminator and generator.

Differences with prior work.

Although the idea of having multiple generators in the GAN architecture is not totally new, e.g., MIX+GAN BID1 and MGAN BID13 , there are key differences between Stackelberg GAN and prior work.

a) In MGAN BID13 , various generators are combined as a mixture of probabilistic models with assumption that the generators and discriminator have enough capacity.

In contrast, in the Stackelberg GAN model we uniformly ensemble the losses of various standard GAN without any assumption on the model capacity.

b) In MIX+GAN BID1 , the losses are ensembled with learned weights and an extra regularization term, which discourages the weights being too far away from uniform.

We find it slightly unnecessary because the expressive power of each generator already allows implicit scaling of each generator.

To the contrary, in the Stackelberg GAN we apply equal weights for all generators.• We prove that the minimax duality gap shrinks as the number of generators increases (see Theorem 1 and Corollary 2).

Unlike the previous work, our result has no assumption on the • We propose the Stackelberg GAN framework of multiple generators in the GAN architecture.

Our framework is general since it can be applied to all variants of GANs, e.g., vanilla GAN, Wasserstein GAN, etc.

It is built upon the idea of jointly optimizing an ensemble of GAN losses w.r.t.

all pairs of discriminator and generator.

Differences from prior work.

Although the idea of having multiple generators in the GAN architecture is not totally new, e.g., MIX+GAN BID1 , MGAN BID13 , MAD-GAN BID9 and GMAN BID8 , there are key differences between Stackelberg GAN and prior work.

a) In MGAN BID13 and MAD-GAN BID9 , various generators are combined as a mixture of probabilistic models with assumption that the generators and discriminator have infinite capacity.

Also, they require that the generators share common network parameters.

In contrast, in the Stackelberg GAN model we allow various sampling schemes beyond the mixture model, e.g., each generator samples a fixed but unequal number of data points independently.

Furthermore, each generator has free parameters.

We also make no assumption on the model capacity in our analysis.

This is an important research question as raised by BID2 .

b) In MIX+GAN BID1 , the losses are ensembled with learned weights and an extra regularization term, which discourages the weights being too far away from uniform.

We find it slightly unnecessary because the expressive power of each generator already allows implicit scaling of each generator.

In the Stackelberg GAN, we apply equal weights for all generators and obtain improved guarantees.

c) In GMAN BID8 , there are multiple discriminators while it is unclear in theory why multi-discriminator architecture works well.

In this paper, we provide formal guarantees for our model.

• We prove that the minimax duality gap shrinks as the number of generators increases (see Theorem 1 and Corollary 2).

Unlike the previous work, our result has no assumption on the expressive power of generators and discriminator, but instead depends on their non-convexity.

With extra condition on the expressive power of generators, we show that Stackelberg GAN is able to achieve -approximate equilibrium with O(1/ ) generators (see Theorem 3).

This improves over the best-known result in BID1 which requires generators as many as O(1/ 2 ).

At the core of our techniques is a novel application of the ShapleyFolkman lemma to the generic minimax problem, where in the literature the technique was only known to work when the objective function is restricted to the Lagrangian function of a constrained optimization problem .

This results in tighter bounds than that of the covering number argument as in BID1 .

We also note that MIX+GAN is a heuristic model which does not exactly match the theoretical analysis in BID1 , while this paper provides formal guarantees for the exact model of Stackelberg GAN.• We empirically study the performance of Stackelberg GAN for various synthetic and real datasets.

We observe that without any human assignment, surprisingly, each generator automatically learns balanced number of modes without any mode being dropped (see FIG2 ).

Compared with other multi-generator GANs with the same network capacity, our experiments show that Stackelberg GAN enjoys 26.76 Fréchet Inception Distance on CIFAR-10 dataset while prior results achieve 31.34 (smaller is better), achieving an improvement of 14.61%.

Before proceeding, we define some notations and formalize our model setup in this section.

Notations.

We will use bold lower-case letter to represent vector and lower-case letter to represent scalar.

Specifically, we denote by θ ∈ R t the parameter vector of discriminator and γ ∈ R g the parameter vector of generator.

Let D θ (x) be the output probability of discriminator given input x, and let G γ (z) represent the generated vector given random input z. For any function f (u), we denote by f DISPLAYFORM0 } the conjugate function of f .

Letclf be the convex closure of f , which is defined as the function whose epigraph is the convex closed hull of that of function f .

We define clf := −cl(−f ).

We will use I to represent the number of generators.

Preliminaries.

The key ingredient in the standard GAN is to play a zero-sum two-player game between a discriminator and a generator -which are often parametrized by deep neural networks in practice -such that the goal of the generator is to map random noise z to some plausible images/texts G γ (z) and the discriminator D θ (·) aims at distinguishing the real images/texts from what the generator creates.

For every parameter implementations γ and θ of generator and discriminator, respectively, denote by the payoff value DISPLAYFORM0 where f (·) is some concave, increasing function.

Hereby, P d is the distribution of true images/texts and P z is a noise distribution such as Gaussian or uniform distribution.

The standard GAN thus solves the following saddle point problems: DISPLAYFORM1 For different choices of function f , problem (1) leads to various variants of GAN.

For example, when f (t) = log t, problem (1) is the classic GAN; when f (t) = t, it reduces to the Wasserstein GAN.

We refer interested readers to the paper of BID18 for more variants of GANs.

Stackelberg GAN.

Our model of Stackelberg GAN is inspired from the Stackelberg competition in the domain of game theory.

Instead of playing a two-player game as in the standard GAN, in Stackelberg GAN there are I + 1 players with two firms -one discriminator and I generators.

One can make an analogy between the discriminator (generators) in the Stackelberg GAN and the leader (followers) in the Stackelberg competition.

Stackelberg GAN is a general framework which can be built on top of all variants of standard GANs.

The objective function is simply an ensemble of losses w.r.t.

all possible pairs of generators and discriminator: DISPLAYFORM2 Thus it is very easy to implement.

The Stackelberg GAN therefore solves the following saddle point problems: DISPLAYFORM3 We term w * − q * the minimax (duality) gap.

We note that there are key differences between the naïve ensembling model and ours.

In the naïve ensembling model, one trains multiple GAN models independently and averages their outputs.

In contrast, our Stackelberg GAN shares a unique discriminator for various generators, thus requires jointly training.

FIG3 shows the architecture of our Stackelberg GAN.How to generate samples from Stackelberg GAN?

In the Stackelberg GAN, we expect that each generator learns only a few modes.

In order to generate a sample that may come from all modes, we use a mixed model.

In particular, we generate a uniformly random value i from 1 to I and use the i-th generator to obtain a new sample.

Note that this procedure in independent of the training procedure.

In this section, we develop our theoretical contributions and compare our results with the prior work.

We begin with studying the minimax gap of Stackelberg GAN.

Our main results show that the minimax gap shrinks as the number of generators increases.

To proceed, denote by DISPLAYFORM0 , where the conjugate operation is w.r.t.

the second argument of φ(γ i ; ·).

We clarify here that the subscript i in h i indicates that the function h i is derived from the i-th generator.

The argument of h i should depend on i, so we denote it by u i .

Intuitively, h i serves as an approximate convexification of −φ(γ i , ·) w.r.t the second argument due to the conjugate operation.

Denote byclh i the convex closure of h i : DISPLAYFORM1 clh i represents the convex relaxation of h i because the epigraph ofclh i is exactly the convex hull of epigraph of h i by the definition ofclh i .

Let DISPLAYFORM2 measures the non-convexity of objective function w.r.t.

argument θ.

For example, it is equal to 0 if and only if φ(γ i ; θ) is concave and closed w.r.t.

discriminator parameter θ.

We have the following guarantees on the minimax gap of Stackelberg GAN.

DISPLAYFORM3 Denote by t the number of parameters of discriminator, i.e., θ ∈ R t .

Suppose that h i (·) is continuous and domh i is compact and convex.

Then the duality gap can be bounded by DISPLAYFORM4 provided that the number of generators I > t+1 ∆ worst γ .

Remark 1.

Theorem 1 makes mild assumption on the continuity of loss and no assumption on the model capacity of discriminator and generators.

The analysis instead depends on their nonconvexity as being parametrized by deep neural networks.

In particular, ∆ , we have 0 ≤ w * − q * ≤ .

The results of Theorem 1 and Corollary 2 are independent of model capacity of generators and discriminator.

When we make assumptions on the expressive power of generator as in BID1 , we have the following guarantee (2) on the existence of -approximate equilibrium.

Theorem 3.

Under the settings of Theorem 1, suppose that for any ξ > 0, there exists a generator G such that E x∼P d ,z∼Pz G(z) − x 2 ≤ ξ.

Let the discriminator and the generators be L-Lipschitz w.r.t.

inputs and parameters, respectively.

Then for any > 0, DISPLAYFORM0 and a discriminator D θ * such that for some value V ∈ R, DISPLAYFORM1 Related Work.

While many efforts have been devoted to empirically investigating the performance of multi-generator GAN, little is known about how many generators are needed so as to achieve certain equilibrium guarantees.

Probably the most relevant prior work to Theorem 3 is that of BID1 .

In particular, BID1 showed that there exist I = 100t 2 ∆ 2 generators and one discriminator such that -approximate equilibrium can be achieved, provided that for all x and any ξ > 0, there exists a generator G such that E z∼Pz G(z) − x 2 ≤ ξ.

Hereby, ∆ is a global upper bound of function |f |, i.e., f ∈ [−∆, ∆].

In comparison, Theorem 3 improves over this result in two aspects: a) the assumption on the expressive power of generators in BID1 .

Therefore, Theorem 3 requires much fewer generators than that of BID1 .

DISPLAYFORM2

In this section, we empirically investigate the effect of network architecture and capacity on the mode collapse/dropping issues for various multi-generator architecture designs.

Hereby, the mode dropping refers to the phenomenon that generative models simply ignore some hard-to-represent modes of real distributions, and the mode collapse means that some modes of real distributions are "averaged" by generative models.

For GAN, it is widely believed that the two issues are caused by the large gap between the minimax and maximin objective function values (see page 35, BID10 ).

Our experiments verify that network capacity (change of width and depth) is not very crucial for resolving the mode collapse issue, though it can alleviate the mode dropping in certain senses.

Instead, the choice of architecture of generators plays a key role.

To visualize this discovery, we test the performance of varying architectures of GANs on a synthetic mixture of Gaussians dataset with 8 modes and 0.01 standard deviation.

We observe the following phenomena: Naïvely increasing capacity of one-generator architecture does not alleviate mode collapse.

It shows that the multi-generator architecture in the Stackelberg GAN effectively alleviates the mode collapse issue.

Though naïvely increasing capacity of one-generator architecture alleviates mode dropping issue, for more challenging mode collapse issue, the effect is not obvious (see FIG9 .

(b) show that increasing the model capacity can alleviate the mode dropping issue, though it does not alleviate the mode collapse issue.

(c) Multi-generator architecture with even small capacity resolves the mode collapse issue.

Stackelberg GAN outperforms multi-branch models.

We compare performance of multi-branch GAN and Stackelberg GAN with objective functions: DISPLAYFORM0 Hereby, the multi-branch GAN has made use of extra information that the real distribution is Gaussian mixture model with probability distribution function DISPLAYFORM1 , so that each γ i tries to fit one component.

However, even this we observe that with same model capacity, Stackelberg GAN significantly outperforms multi-branch GAN (see FIG11 (a)(c)) even without access to the extra information.

The performance of Stackelberg GAN is also better than multi-branch GAN of much larger capacity (see FIG11 ).

Generators tend to learn balanced number of modes when they have same capacity.

We observe that for varying number of generators, each generator in the Stackelberg GAN tends to learn equal number of modes when the modes are symmetric and every generator has same capacity (see FIG13 ).

In this section, we verify our theoretical contributions by the experimental validation.

We first show that Stackelberg GAN generates more diverse images on the MNIST dataset (LeCun et al., 1998) than classic GAN.

We follow the standard preprocessing step that each pixel is normalized via subtracting it by 0.5 and dividing it by 0.5.

The detailed network setups of discriminator and generators are in TAB4 .

Figure 6 shows the diversity of generated digits by Stackelberg GAN with varying number of generators.

When there is only one generator, the digits are not very diverse with many repeated "1"'s and much fewer "2"'s.

As the number of generators increases, the generated images tend to be more diverse.

In particular, for 10-generator Stackelberg GAN, each generator is associated with one or two digits without any digit being missed.

We also observe better performance by the Stackelberg GAN on the Fashion-MNIST dataset.

Fashion-MNIST is a dataset which consists of 60,000 examples.

Each example is a 28 × 28 grayscale image associating with a label from 10 classes.

We follow the standard preprocessing step that each pixel is normalized via subtracting it by 0.5 and dividing it by 0.5.

We specify the detailed network setups of discriminator and generators in TAB4 .

Figure 7 shows the diversity of generated fashions by Stackelberg GAN with varying number of generators.

When there is only one generator, the generated images are not very diverse without Figure 6 : Standard GAN vs. Stackelberg GAN on the MNIST dataset without cherry pick.

Left Figure: Digits generated by the standard GAN.

It shows that the standard GAN generates many "1"'s which are not very diverse.

Middle Figure: Digits generated by the Stackelberg GAN with 5 generators, where every two rows correspond to one generator.

Right Figure: Digits generated by the Stackelberg GAN with 10 generators, where each row corresponds to one generator.

Figure 7: Generated samples by Stackelberg GAN on CIFAR-10 dataset without cherry pick.

Left Figure: Examples generated by the standard GAN.

It shows that the standard GAN fails to generate bags.

Middle Figure: Examples generated by the Stackelberg GAN with 5 generators, where every two rows correspond to one generator.

Right Figure: Examples generated by the Stackelberg GAN with 10 generators, where each row corresponds to one generator.

any bags being found.

As the number of generators increases, the generated images tend to be more diverse.

In particular, for 10-generator Stackelberg GAN, each generator is associated with one class without any class being missed.

We then implement Stackelberg GAN on the CIFAR-10 dataset.

CIFAR-10 includes 60,000 32×32 training images, which fall into 10 classes BID15 ).

The architecture of generators and discriminator follows the design of DCGAN in BID19 .

We train models with 5, 10, and 20 fixed-size generators.

The results show that the model with 10 generators performs the best.

We also train 10-generator models where each generator has 2, 3 and 4 convolution layers.

We find that the generator with 2 convolution layers, which is the most shallow one, performs the best.

So we report the results obtained from the model with 10 generators containing 2 convolution layers.

FIG15 shows the samples produced by different generators.

The samples are randomly drawn instead of being cherry-picked to demonstrate the quality of images generated by our model.

For quantitative evaluation, we use Inception score and Fréchet Inception Distance (FID) to measure the difference between images generated by models and real images.

Results of Inception Score.

The Inception score measures the quality of a generated image and is correlated well with human's judgment BID21 .

We report the Inception score obtained by our Stackelberg GAN and other baseline methods in TAB1 .

For fair comparison, we only consider the baseline models which are completely unsupervised model and do not need any label information.

Instead of directly using the reported Inception scores by original papers, we replicate the experiment of MGAN using the code, architectures and parameters reported by their original papers, and evaluate the scores based on the new experimental results.

TAB1 shows that our model achieves a score of 7.62 in CIFAR-10 dataset, which outperforms the state-of-the-art models.

For fairness, we configure our Stackelberg GAN with the same capacity as MGAN, that is, the two models have comparative number of total parameters.

When the capacity of our Stackelberg GAN is as small as DCGAN, our model improves over DCGAN significantly.

Results of Fréchet Inception Distance.

We then evaluate the performance of models on CIFAR-10 dataset using the Fréchet Inception Distance (FID), which better captures the similarity between generated images and real ones BID12 .

As TAB1 shows, under the same capacity as DCGAN, our model reduces the FID by 20.74%.

Meanwhile, under the same capacity as MGAN, our model reduces the FID by 14.61%.

This improvement further indicates that our Stackelberg GAN with multiple light-weight generators help improve the quality of the generated images.

Real data 11.24 ± 0.16 -WGAN 3.82 ± 0.06 -MIX+WGAN BID1 4.04 ± 0.07 -Improved-GAN BID21 4.36 ± 0.04 -ALI BID7 5.34 ± 0.05 -BEGAN BID4 5.62 -MAGAN BID24 5.67 -GMAN BID8 6.00 ± 0.19 -DCGAN BID19 6.40 ± 0.05 37.7 Ours (capacity as DCGAN)7.02 ± 0.07 29.88 D2GAN BID17 7.15 ± 0.07 -MAD-GAN (our run) BID9 6.67 ± 0.07 34.10 MGAN (our run) BID13 7.52 ± 0.1 31.34 Ours ( FIG2 7.62 ± 0.07 26.76

We also evaluate the performance of Stackelberg GAN on the Tiny ImageNet dataset.

The Tiny ImageNet is a large image dataset, where each image is labelled to indicate the class of the object inside the image.

We resize the figures down to 32 × 32 following the procedure described in BID6 .

FIG15 shows the randomly picked samples generated by 10-generator Stackelberg GAN.

Each row has samples generated from one generator.

Since the types of some images in the Tiny ImageNet are also included in the CIFAR-10, we order the rows in the similar way as FIG15 .

In this work, we tackle the problem of instability during GAN training procedure, which is caused by the huge gap between minimax and maximin objective values.

The core of our techniques is a multi-generator architecture.

We show that the minimax gap shrinks to as the number of generators increases with rate O(1/ ), when the maximization problem w.r.t.

the discriminator is concave.

This improves over the best-known results of O(1/ 2 ).

Experiments verify the effectiveness of our proposed methods.

TAB5 is by the weak duality.

Thus it suffices to prove the other side of the inequality.

All notations in this section are defined in Section 3.1.

We first show that DISPLAYFORM0 Denote by DISPLAYFORM1 We have the following lemma.

Lemma 4.

We have DISPLAYFORM2 Proof.

By the definition of p(0), we have p(0) = inf γ1,...,γ I ∈R g sup θ∈R t Φ(γ 1 , ..., γ I ; θ).

Since (clp)(·) is the convex closure of function p(·) (a.k.a.

weak duality theorem), we have (clp)(0) ≤ p(0).

We now show that sup DISPLAYFORM3 Note that p(u) = inf γ1,...,γ I ∈R g p γ1,...

,γ I (u), where p γ1,...,γ I (u) = sup θ∈R t { Φ(γ 1 , ..., γ I ; θ) −

u T θ} = (− Φ(γ 1 , ..., γ I ; ·)) * (−u), and that .

We have the following lemma.

DISPLAYFORM4 Lemma 5.

Under the assumption in Theorem 1, DISPLAYFORM5 Proof.

We note that DISPLAYFORM6 where u 1 , ..., u I , u ∈ R t .

Therefore, DISPLAYFORM7 Consider the subset of R t+1 : DISPLAYFORM8 Define the vector summation DISPLAYFORM9 is continuous and domh i is compact, the set DISPLAYFORM10 DISPLAYFORM11 We apply Lemma 6 to prove Lemma 5 with m = t + 1.

Let (r, w) ∈ conv(Y) be such that r = 0, and w =clp(0).

DISPLAYFORM12 i ∈I DISPLAYFORM13 Representing elements of the convex hull of DISPLAYFORM14 by Carathéodory theorem, we have that for each i ∈ I, there are vectors {u DISPLAYFORM15 Recall that we definȇ DISPLAYFORM16 and DISPLAYFORM17 We have for i ∈ I, DISPLAYFORM18 Thus, by Eqns.

FORMULA27 and FORMULA30 , we have DISPLAYFORM19 Therefore, we have DISPLAYFORM20 (by Eqns.

FORMULA28 and FORMULA33 ) DISPLAYFORM21 , (by Lemma 6) as desired.

By Lemmas 4 and 5, we have proved that DISPLAYFORM22 To prove Theorem 1, we note that DISPLAYFORM23 When φ(γ i ; θ) is concave and closed w.r.t.

discriminator parameter θ, we have clφ = φ.

Thus, ∆ minimax θ = ∆ maximin θ = 0 and 0 ≤ w * − q * ≤ .

We first show that the equilibrium value V is 2f (1/2).

For the discriminator D θ which only outputs 1/2, it has payoff 2f (1/2) for all possible implementations of generators G γ1 , ..., G γ I .

Therefore, we have V ≥ 2f (1/2).

We now show that V ≤ 2f (1/2).

We note that by assumption, for any ξ > 0, there exists a closed neighbour of implementation of generator G ξ such that E x∼P d ,z∼Pz G ξ (z) − x 2 ≤ ξ for all G ξ in the neighbour.

Such a neighbour exists because the generator is Lipschitz w.r.t.

its parameters.

Let the parameter implementation of such neighbour of G ξ be Γ. The Wasserstein distance between G ξ and P d is ξ.

Since the function f and the discriminator are L f -Lipschitz and L-Lipschitz, respectively, we have DISPLAYFORM0 .

Thus, for any fixed γ, we have DISPLAYFORM1 which implies that sup θ∈R t Φ(γ 1 , ..., γ I ; θ) = 2f (1/2) for all γ 1 , ..., γ I ∈ Γ.

So we have V = 2f (1/2).

This means that the discriminator cannot do much better than a random guess.

The above analysis implies that the equilibrium is achieved when D θ * only outputs 1/2.

Denote by Θ the small closed neighbour of such θ * such that Φ(γ 1 , ..., γ I ; θ) is concave w.r.t.

θ ∈ Θ for any fixed γ 1 , ..., γ I ∈ Γ. We thus focus on the loss on Θ ⊆ R t and Γ ⊆ R g : DISPLAYFORM2 Since Φ(γ 1 , ..., γ I ; θ) is concave w.r.t.

θ ∈ Θ for all γ 1 , ..., γ I ∈ Γ, by Corollary 2, we have DISPLAYFORM3 The optimal implementations of γ 1 , ..., γ I is achieved by argmin γ1,...,γ I ∈Γ sup θ∈Θ DISPLAYFORM4 Proof.

We define DISPLAYFORM5 Clearly, the vanilla GAN optimization can be understood as projecting under L: DISPLAYFORM6 In the Stackelberg GAN setting, we are projecting under a different distanceL which is defined as DISPLAYFORM7 We note that f is strictly concave and the discriminator has capacity large enough implies the followings: L(P 1 , P 2 ), as a function of P 2 , achieves the global minimum if and only if P 2 = P 1 .

The theorem then follows from this fact and (11).E NETWORK SETUP Adam(β 1 = 0.5, β 2 = 0.999) Weight, bias initialization N (µ = 0, σ = 0.01), 0

<|TLDR|>

@highlight

We study the problem of alleviating the instability issue in the GAN training procedure via new architecture design, with theoretical guarantees.