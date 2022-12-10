Do GANS (Generative Adversarial Nets) actually learn the target distribution?

The foundational paper of Goodfellow et al. (2014) suggested they do, if they were given sufficiently large deep nets, sample size, and computation time.

A recent theoretical analysis in Arora et al. (2017) raised doubts whether the same holds when discriminator has bounded size.

It showed that the training objective can approach its optimum value even if the generated distribution has very low support.

In other words, the training objective is unable to prevent mode collapse.

The current paper makes two contributions.

(1) It proposes a novel test for estimating support size using the birthday paradox of discrete probability.

Using this  evidence is presented that well-known GANs approaches do learn distributions of fairly low support.

(2) It theoretically studies encoder-decoder GANs architectures (e.g., BiGAN/ALI), which were proposed to learn more meaningful features via GANs, and consequently to also solve the mode-collapse issue.

Our result shows that such encoder-decoder training objectives also cannot guarantee learning of the full distribution because they cannot prevent serious mode collapse.

More seriously, they cannot prevent learning meaningless codes for data, contrary to usual intuition.

From the earliest papers on Generative Adversarial Networks the question has been raised whether or not they actually come close to learning the distribution they are trained with (henceforth refered to as the target distribution)?

These methods train a generator deep net that converts a random seed into a realistic-looking image.

Concurrently they train a discriminator deep net to discriminate between its output and real images, which in turn is used to produce gradient feedback to improve the generator net.

In practice the generator deep net starts producing realistic outputs by the end, and the objective approaches its optimal value.

But does this mean the deep net has learnt the target distribution of real images?

Standard analysis introduced in BID3 shows that given "sufficiently large" generator and discriminator, sample size, and computation time the training does succeed in learning the underlying distribution arbitrarily closely (measured in JensenShannon divergence).

But this does not settle the question of what happens with realistic sample and net sizes.

Note that GANs differ from many previous methods for learning distributions in that they do not provide an estimate of a measure of distributional fit -e.g., perplexity score.

Therefore researchers have probed their performance using surrogate qualitative tests, which were usually designed to rule out the most obvious failure mode of the training, namely, that the GAN has simply memorized the training data.

One test checks the similarity of each generated image to the nearest images in the training set.

Another takes two random seeds s 1 , s 2 that produced realistic images and checks the images produced using seeds lying on the line joining s 1 , s 2 .

If such "interpolating" images are reasonable and original as well, then this may be taken as evidence that the generated distribution has many novel images.

Yet other tests check for existence of semantically meaningful directions in the latent space, meaning that varying the seed along these directions leads to predictable changes e.g., (in case of images of human faces) changes in facial hair, or pose.

A recent test proposed by BID12 checks the log-likelihoods of GANs using Annealed Importance Sampling, whose results indicate the mismatch between generator's distribution and the target distribution.

BID8 proposed a method to trade-off between sample quality and sample diversity but they don't provide a clear definition or a quantitative metric of sample diversity.

Recently a new theoretical analysis of GANs with finite sample sizes and finite discriminator size revealed the possibility that training objective can sometimes approach optimality even if the generator is far from having actually learnt the distribution.

Specifically, if the discriminator has size p, then the training objective could be close to optimal even though the output distribution is supported on only O(p log p/ 2 ) images.

By contrast one imagines that the target distribution usually must have very large support.

For example, the set of all possible images of human faces (a frequent setting in GANs work) must involve all combinations of hair color/style, facial features, complexion, expression, pose, lighting, race, etc., and thus the possible set of images of faces approaches infinity.

The above paper raises the possibility that the discriminator may be unable to meaningfully distinguish such a diverse target distribution from a trained distribution with fairly small support.

Furthermore, the paper notes that this failure mode is different from the one usually feared, namely the generator memorizing training samples.

The analysis of raises the possibility that the trained distribution has small support, and yet all its samples could be completely disjoint from the training samples.

However, the above analysis was only a theoretical one, exhibiting a particular near-equilibrium solution that can happen from certain hyper-parameter combinations.

It left open the possibility that real-life GANs training avoids such solutions thanks to some not-as-yet-understood property of SGD, or hyper-parameter choices.

Thus further experimental investigation seems necessary.

And yet it seems difficult at first sight to do such an empirical evaluation of the support size of a distribution: it is not humanly possible to go through hundreds of thousands of images, whereas automated tests of image similarity can be thrown off by small changes in lighting, pose etc.

The current paper makes two important contributions.

On the empirical side it introduces a new test for the support size of the trained distribution, and uses it to find that unfortunately these mode collapse problems do arise in many well-regarded GAN training methods.

On the theoretical side we prove the limitations of encoder-decoder frameworks like BiGAN BID1 and Adversarially Learned Inference or ALI BID2 , which, inspired by autoencoder models, require the setup to learn an inference mechanism as well as a generative mechanism.

The result of applies only to standard GAN training objectives (including JS and Wasserstein), but not to encoder-decoder setups.

The clear hope in defining encoder-decoder setups is that the encoding mechanism "inverts" the generator and thus forces the generator to learn meaningful featurizations of data that are useful in downstream applications.

In fact it has often been proposed that this need to learn meaningful featurizations will also solve the mode collapse problem: BID2 provide experiments on 2-dimensional mixtures of Gaussians suggesting this phenomenon.

Our analysis shows not only that encoder-decoder training objectives cannot avoid mode collapse, but that they also cannot enforce learning of meaningful codes/features.

Let's consider a simple test that estimates the support size of a discrete distribution.

Suppose a distribution has support N .

The famous birthday paradox 1 says that a batch of about √ N samples would be likely to have a duplicate.

Thus our proposed birthday paradox test for GANs is as follows.

If this test reveals that batches of size s have duplicate images with good probability, then suspect that the distribution has support size about s 2 .

Note that the test is not definitive, because the distribution could assign a probability 10% to a single image, and be uniform on a huge number of other images.

Then the test would be likely to find a duplicate even with 20 samples, though the true support size is huge.

But such nonuniformity (a lot of probability being assigned to a few images) is the only failure mode of the birthday paradox test calculation, and such nonuniformity would itself be considered a failure mode of GANs training.

This is captured in the following theorems: Theorem 1.

Given a discrete probability distribution P on a set Ω, if there exists a subset S ⊆ Ω of size N such that s∈S P (s) ≥ ρ, then the probability of encountering at least one collision among

)

Theorem 2.

Given a discrete probability distribution P on a set Ω, if the probability of encountering at least one collision among M i.i.d.

samples from P is γ, then for DISPLAYFORM0 , under realistic assumptions on parameters.

The proofs of these theorems are included in Appendix A.

It is important to note that Theorem 1 and 2 do not assume that the tested distribution is uniform.

In fact, Theorem 2 clarifies that if one can consistently see collisions in batches, then the distribution has a major component distribution that has limited support size but is almost indistinguishable from the full distribution via sampling a small number of samples.

Thus the distribution effectively has small support size, which is what one should care about when sampling from it.

Furthermore, without any further assumption, to accurately estimate the support size of an arbitrary distribution that has n modes, Ω(n/ log n) samples need to be seen, rendering it practically infeasible for a human examiner BID10 .

In the GAN setting, the distribution is continuous, not discrete.

When support size is infinite then in a finite sample, we should not expect exact duplicate images where every pixel is identical.

Thus a priori one imagines the birthday paradox test to completely not work.

But surprisingly, it still works if we look for near-duplicates.

Given a finite sample, we select the 20 closest pairs according to some heuristic metric, thus obtaining a candidate pool of potential near-duplicates inspect.

Then we visually identify if any of them would be considered duplicates by humans.

Our test were done using two datasets, CelebA (faces) BID7 and CIFAR-10 (Krizhevsky, 2009) .

Note that CelebA reasonably balanced, since the constructors intentionally made it unbiased (it contains ten thousand identities, each of which has twenty images).

Also, we report in Appendix B results on the Bedroom dataset from the LSUN BID13 .For faces, we find Euclidean distance in pixel space works well as a heuristic similarity measure, probably because the samples are centered and aligned.

For CIFAR-10, we pre-train a discriminative CNN for the full classification problem, and use the top layer representation as an embedding.

Heuristic similarity is then measured as the Euclidean distance in the embedding space.

These metrics can be crude, but note that improving them can only lower our estimate of the support size, since a better similarity measure can only increase the number of duplicates found.

Thus our reported estimates should be considered as upper bounds on the support size of the distribution.

Note: Some GANs (and also older methods such as variational autoencoders) implicitly or explicitly apply noise to the training and generated images.

This seems useful if the goal is to compute a perplexity score, which involves the model being able to assign a nonzero probability to every image.

Such noised images are usually very blurry and the birthday paradox test does not work well for them, primarily because the automated measure of similarity no longer works well.

Even visually judging similarity of noised images is difficult.

Thus our experiments work best with GANs that generate sharper, realistic images.

We test the following methods, doing the birthday paradox test with Euclidean distance in pixel space as the heuristic similarity measure For judging whether two images are the same, we set the criterion that the two faces are not exactly identical but look like doppelgangers. (Of course, in real life the only doppelgangers we know are usually twins.)• DCGAN -unconditional, with JSD objective as described in BID3 and BID9 .• MIX+ GAN protocol introduced in , specifically, MIX+DCGAN with 3 mixture components.• ALI (Dumoulin et al., 2017) (or equivalently BiGANs BID1 ).

For fair comparison, we set the discriminator of ALI (or BiGANs) to be roughly the same in size as that of the DCGAN model, since the results of Section 2.1.1 below suggests that the discriminator size has a strong effect on diversity of the learnt distribution.

We find that with probability ≥ 50%, a batch of ∼ 800 samples contains at least one pair of duplicates for both DCGAN and MIX+DCGAN.

FIG1 displays duplicates and their nearest neighbors in training set.

These results suggest that the support size of the distribution is less than 800 2 ≈ 640000, being at the same order of the size of the training set, but this distribution is not just memorizing the training set (see the dashed boxes).ALI (or BiGANs) appears to be more diverse, in that collisions appear with 50% probability only with a batch size of 1200, implying a support size of a million.

This is 6x the training set, but still much smaller than the diversity one would expect among human faces 3 .

Nevertheless, these tests do support the suggestion in BID2 and BID1 that the bidirectional structure prevents some of the mode collapse observed in usual GANs.

The analysis of Arora et al suggested that the support size could be as low as near-linear in the capacity of the discriminator; in other words, there is a near-equilibrium in which a distribution of such a small support could suffice to fool the best discriminator.

So it is worth investigating whether training in real life allows generator nets to exploit this "loophole" in the training that we now know is in principle available to them.

While a comprehensive test is beyond the scope of this paper, we do a first test with a simplistic version of discriminator size (i.e., capacity).

We build DCGANs with increasingly larger discriminators while fixing the other hyper-parameters.

The discriminator used here is a 5-layer Convolutional Neural Network such that the number of output channels of each layer is 1×, 2×, 4×, 8×dim where dim is chosen to be 16, 24, . . .

, 120, 128.

Thus the discriminator size should be proportional to dim 2 .

FIG2 suggests that in this simple setup the diversity of the learnt distribution does indeed grow near-linearly with the discriminator size.

On CIFAR-10, Euclidean distance in pixel space is not informative.

So we adopt a classifying CNN with 3 convolutional layers, 2 fully-connected layer and a 10-class soft-max output pretrained with a multi-class classification objective, and use its top layer features as embeddings for similarity test using Euclidean distance.

Firstly, we find that the result of the test is affected by the quality of samples.

If the training uses noised samples (with noise being added either explicitly or implicitly in the objective) then the generated samples are also quite noisy.

Then the most similar samples in a batch tend to be blurry blobs of low quality.

Indeed, when we test a DCGAN (even the best variant with 7.16 Inception Score reported in BID4 ), the pairs returned are mostly blobs.

To get meaningful test results, we turn to a Stacked GAN which is the state-of-the-art generative model on CIFAR-10 (Inception Score 8.59 BID4 ).

It also generates the most reallooking images.

Since this model is trained by conditioning on class label, we measure its diversity within each class separately.

The batch sizes needed for duplicates are shown in Table 1 .

Duplicate samples shown in FIG4 .

Truck 500 50 500 100 500 300 50 200 500 100 Table 1 : Class specific batch size needed to encounter duplicate samples with > 50% probability, from a Stacked GAN trained on CIFAR-10 We check whether the detected duplicates are close to any of the training images, by looking for the nearest neighbor in the training set using our heuristic similarity measure and visually inspecting the closest suspects.

We find that the closest image is quite different from the duplicates, suggesting the issue with GANs is indeed lack of diversity (low support size) instead of memorizing training set.

We recall the Adversarial Feature Learning (BiGAN) setup from BID1 .

The "generative" player consists of two parts: a generator G and an encoder E. The generator takes as input a latent variable z and produces a sample G(z); the encoder takes as input a data sample x and produces a guess for the latent variable E(x).

This produces two joint distributions over pairs of latent variables and data samples: (z, G(z)) and (E(x), x).

The goal of the "generative" player is to convince the discriminator that these two distributions are the same, whereas the discriminator is being trained to distinguish between them.

In the ideal case, the hope is that the "generative" player converges to (z, G(z)) and (E(x), x) both being jointly distributed as p(z, x) where p is the joint distribution of the latent variables and data -i.e.

G(z) is distributed as p(x|z): the true generator distribution; and E(x) is distributed as p(z|x): the true encoder distribution.

Using usual min-max formalism for adversarial training, the BiGAN objective is written as: DISPLAYFORM0 whereμ is the empirical distribution over data samples x;ν is a distribution over random "seeds" for the latent variables: typically sampled from a simple distribution like a standard Gaussian; and φ is a concave "measuring" function.

(The standard choice is log, though other options have been proposed in the literature.)

For our purposes, we will assume that φ outputs values in the range [−∆, ∆], ∆ ≥ 1, and is L φ -Lipschitz.

For ease of exposition we will refer to µ as the image distribution.

The proof is more elegant if we assume that µ consists of images that have been noised -concretely, think of replacing every 100th pixel by Gaussian noise.

Such noised images would of course look fine to our eyes, and we would expect the learnt encoder/decoder to not be affected by this noise.

For concreteness, we will take the seed/code distribution ν to be a spherical zero-mean Gaussian (in an arbitrary dimension and with an arbitrary variance).

4 Furthermore, we will assume that Domain(µ) = R d , Domain(ν) = Rd withd < d (we think of d d, which is certainly the case in practice).

As in Arora et al. FORMULA2 we assume that discriminators are L-lipschitz with respect to their trainable parameters, and the support size of the generator's distribution will depend upon this L and the capacity p (= number of parameters) of the discriminator.

There exists a generator G of support DISPLAYFORM0 and an encoder E with at mostd non-zero weights, s.t.

for all discriminators D that are L-Lipschitz and have capacity less than p, it holds that DISPLAYFORM1 The interpretation of the above theorem is as stated before: the encoder E has very small complexity (we will subsequently specify it precisely and show it simply extracts noise from the input x); the generator G is a small-support distribution (so presumably far from the true data distribution).

Nevertheless, the value of the BiGAN objective is small.

The argument of seems unable to apply to this setting.

It is a simple concentration/epsilon-net argument showing that the discriminator of capacity p cannot distinguish between a generator that samples from µ versus one that memorizes a subset of p log p 2 random images in µ and outputs one randomly from this subset.

By contrast, in the current setting we need to say what happens with the encoder.

The precise noise model: Denoting byμ the distribution of unnnoised images, and ν the distribution of seeds/codes, we define the distribution of noised images µ as the following distribution: to produce a sample in µ take a samplex fromμ and z from ν independently and output x =x z, which is defined as DISPLAYFORM2 In other words, set every The main idea is to show the existence of the generator/encoder pair via a probabilistic construction that is shown to succeed with high probability.• Encoder E: The encoder just extracts the noise from the noised image (by selecting the relevantd coordinates).

Namely, E(x z) = z. (So the code is just gaussian noise and has no meaningful content.)

It's easy to see this can be captured using a ReLU network with d weights: we can simply connect the i-th output to the (i DISPLAYFORM3 )-th input using an edge of weight 1.• Generator G: This is designed to produce a distribution of support size m := p∆ 2 log 2 (p∆LL φ / )

.

We first define a partition of Domain(ν) = Rd into m equal-measure blocks under ν.

Next, we sample m samples x * 1 , x * 2 , . . .

, x * m from the image distribution.

Finally, for a sample z, we define ind(z) to be the index of the block in the partition in which z lies, and define the generator as G(z) = x * ind(z) z. Since the set of samples x * i : i ∈ [m] is random, this specifies a distribution over generators.

We prove that with high probability, one of these generators satisfies the statement of Theorem 3.

Moreover, we show that such a generator can be easily implemented using a ReLU network of complexity O(md) in Theorem 5.The basic intuition of the proof is as follows.

We will call a set T of samples from ν non-colliding if no two lie in the same block.

Let T nc be the distribution over non-colliding sets {z 1 , z 2 , . . .

, z m }, s.t.

each z i is sampled independently from the conditional distribution of ν inside the i-th block.

First, we notice that under the distribution for G we defined, it holds that DISPLAYFORM0 In other words, the "expected" encoder correctly matches the expectation of φ(D(x, E(x))), so that the discriminator is fooled.

We want to show that E G E z∼ν φ(G(z), z) concentrates enough around this expectation, as a function of the randomness in G, so that we can say with high probability over the choice of DISPLAYFORM1 We handle the concentration argument in two steps:First, we note that we can calculate the expectation of φ(D(G(z), z)) when z ∼ ν by calculating the empirical expectation over m-sized non-colliding sets T sampled according to T nc .

Namely, as we show in Lemma D.1: DISPLAYFORM2 Thus, we have reduced our task to arguing about the concentration of E T ∼Tnc E z∼T φ(D(G(z), z)) (viewed as a random variable in G).

Towards this, we consider the random variable E z∼T φ(D (G(z), z) ) as a function of the randomness in G and T both.

Since T is a non-colliding set of samples, we can write DISPLAYFORM3 for some function f , where the random variables x * i , z i are all mutually independent -thus use McDiarmid's inequality to argue about the concentration of f in terms of both T and G.From this, we can use Markov's inequality to argue that all but an exponentially small (in p) fraction of encoders G satisfy that: for all but an exponentially small (in p) fraction of non-colliding sets (G(z), z) )| is small.

Note that this has to hold for all discriminators D -so we need to additionally build an epsilon-net, and union bound over all discriminators, similarly as in .

Then, it's easy to extrapolate that for such G, DISPLAYFORM4 DISPLAYFORM5 is small, as we want.

The details are in Lemma D.2 in the Appendix.

The paper reveals gaps in current thinking about GANs, and hopes to stimulate further theoretical and empirical study.

GANs research has always struggled with the issue of mode collapse, and recent theoretical analysis of shows that the GANs training objective is not capable of preventing mode collapse.

This exhibits the existence of bad solutions in the optimization landscape.

This in itself is not definitive, since existence of bad solutions is also known for the more traditional classification tasks , where heldout sets can nevertheless prove that a good solution has been reached.

The difference in case of GANs is lack of an obvious way to establish that training succeeded.

Our new Birthday Paradox test gives a new benchmark for testing the support size (i.e., diversity of images) in a distribution.

Though it may appear weak, experiments using this test suggest that current GANs approaches, specifically, the ones that produce images of higher visual quality, do suffer mode collapse.

Our rough experiments also suggest -again in line with the previous theoretical analysis-that the size of the distribution's support scales near-linearly with discriminator capacity.

Researchers have raised the possibility that the best use of GANs is not distribution learning but feature learning.

Encoder-decoder GAN architectures seem promising since they try to force the generator to use "meaningful" encodings of the image.

While such architectures do exhibit slightly better diversity in our experiments, our theoretical result suggest that the the encoder-decoder objective is also unable to avoid mode collapse, furthermore, also fails to guarantee meaningful codes.

In this section, we prove the statemts of Theorems 1 and 2.Proof of Theorem 1.Pr[there is at least a collision among M samples]

≥ 1 − Pr[there is no collision within set S among M samples]

DISPLAYFORM0 We use the fact that the worst case is when the ρ probability mass is uniformly distributed on S.Proof of Theorem 2.

Suppose X 1 , X 2 , . . .

are i.i.d.

samples from the discrete distribution P .

We define T = inf{t ≥ 2, X t ∈ {X 1 , X 2 , . . .

, X t−1 }} to be the collision time and also we use DISPLAYFORM1 X∈Ω P (X) 2 as a surrogate for the uniformity of P .

According Theorem 3 in Wiener FORMULA13 , P r[T ≥ M ] can be upper-bounded using β.

Specifically, with β > 1000 and M ≤ 2 √ β ln β, which is usually true when P is the distribution of a generative model of images, DISPLAYFORM2 To estimate β, we notice that DISPLAYFORM3 which immediately implies DISPLAYFORM4 This gives us a upper-bound of the uniformity of distribution P , which we can utilize.

Let S ⊆ Ω be the smallest set with probability mass ≥ ρ and suppose it size is N .

To estimate the largest possible N such that the previous inequality holds, we let DISPLAYFORM5

We test DCGANs trained on the trained on 64 × 64 center cropped Bedroom dataset(LSUN) using Euclidean distance to extract collision candidates since it is impossible to train a CNN classifier on such single-category (bedroom) dataset.

We notice that the most similar pairs are likely to be the corrupted samples with the same noise pattern (top-5 collision candidates all contain such patterns).

When ignoring the noisy pairs, the most similar "clean" pairs are not even similar according to human eyes.

This implies that the distribution puts significant probability on noise patterns, which can be seen as a form of under-fitting (also reported in the DCGAN paper).

We manually counted the number of samples with a fixed noise pattern from a batch of 900 i.i.d samples.

We find 43 such corrupted samples among the 900 generated images, which implies 43/900 ≈ 5% probability.

Given these findings, it is natural to wonder about the diversity of distributions learned using earlier methods such as Variational Auto-Encoders Kingma & Welling (2014) (VAEs).

Instead of using feedback from the discriminator, these methods train the generator net using feedback from an approximate perplexity calculation.

Thus the analysis of does not apply as is to such methods and it is conceivable they exhibit higher diversity.

However, we find the birthday paradox test difficult to run since samples from a VAE trained on CelebA are not realistic or sharp enough for a human to definitively conclude whether or not two images are almost the same.

Fig 5 shows examples of collision candidates found in batches of 400 samples; clearly some indicative parts (hair, eyes, mouth, etc.) are quite blurry in VAE samples.

We recall the basic notation from the main part: the image distribution will be denoted as µ, and the code/seed distribution as ν, which we assume is a spherical Gaussian.

For concreteness, we assumed the domain ofμ is R d and the domain of ν is Rd withd ≤ d. (As we said, we are thinking ofd d.)We also introduced the quantity m := DISPLAYFORM0 Before proving Theorem 3, let's note that the claim can easily be made into a finite-sample version.

Namely: Corollary D.1 (Main, finite sample version).

There exists a generator G of support m, s.t.

ifμ is the uniform distribution over a training set S of size at least m, andν is the uniform distribution over a sample T from ν of size at least m, for all discriminators D that are L-Lipschitz and have less than p parameters, with probability 1 − exp(−Ω(p)) over the choice of training set S,T we have: DISPLAYFORM1 As is noted in Theorem B.2 in , we can build a LL φ -net for the discriminators with a size bounded by e p log(LL φ p/ ) .

By Chernoff and union bounding over the points in the LL φ -net, with probability at least 1 − exp(−Ω(p)) over the choice of a training set S, we have DISPLAYFORM2 for all discriminators D with capacity at most p. Similarly, with probability at least 1 − exp(−Ω(p)) over the choice of a noise set T , Let us recall we call a set T of samples from ν non-colliding if no two lie in the same block and we denoted T nc to be the distribution over non-colliding sets {z 1 , z 2 , . . .

, z m }, s.t.

each z i is sampled independently from the conditional distribution of ν inside the i-th block of the partition.

First, notice the following Lemma: Lemma D.1 (Reducing to expectations over non-colliding sets).

Let G be a fixed generator, and D a fixed discriminator.

Then, i , z i are all mutually independent.

Note that the arguments that f is a function of are all independent variables, so we can apply McDiarmid's inequality.

Towards that, let's denote by z −i the vector of all inputs to f , except for z i .

Notice that Building a LL φ -net for the discriminators and union bounding over the points in the net, we get that Pr T,G (∃D, |R D,T,G | ≥ /2) ≤ exp(−Ω(p log(∆/ ))).

On the other hand, we also have DISPLAYFORM3 DISPLAYFORM4

<|TLDR|>

@highlight

We propose a support size estimator of GANs's learned distribution to show they indeed suffer from mode collapse, and we prove that encoder-decoder GANs do not avoid the issue as well.

@highlight

The paper attempts to estimate the size of the support for solutions produced by typical GANs experimentally. 

@highlight

This paper proposes a clever new test based on the birthday paradox for measuring diversity in generated sample, with experiment results interpreted to mean that mode collapse is strong in a number of state-of-the-art generative models.

@highlight

The paper uses birthday paradox to show that some GAN architectures generate distributions with fairly low support.