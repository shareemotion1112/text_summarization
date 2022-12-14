Generative adversarial networks (GANs) have been shown to provide an effective way to model complex distributions and have obtained impressive results on various challenging tasks.

However, typical GANs require fully-observed data during training.

In this paper, we present a GAN-based framework for learning from complex, high-dimensional incomplete data.

The proposed framework learns a complete data generator along with a mask generator that models the missing data distribution.

We further demonstrate how to impute missing data by equipping our framework with an adversarially trained imputer.

We evaluate the proposed framework using a series of experiments with several types of missing data processes under the missing completely at random assumption.

Generative adversarial networks (GANs) BID0 provide a powerful modeling framework for learning complex high-dimensional distributions.

Unlike likelihood-based methods, GANs are referred to as implicit probabilistic models BID8 .

They represent a probability distribution through a generator that learns to directly produce samples from the desired distribution.

The generator is trained adversarially by optimizing a minimax objective together with a discriminator.

In practice, GANs have been shown to be very successful in a range of applications including generating photorealistic images BID3 .

Other than generating samples, many downstream tasks require a good generative model, such as image inpainting BID9 BID15 .Training GANs normally requires access to a large collection of fully-observed data.

However, it is not always possible to obtain a large amount of fully-observed data.

Missing data is well-known to be prevalent in many real-world application domains where different data cases might have different missing entries.

This arbitrary missingness poses a significant challenge to many existing machine learning models.

Following BID6 , the generative process for incompletely observed data can be described as shown below where x ∈ R n is a complete data vector and m ∈ {0, 1} n is a binary mask 2 that determines which entries in x to reveal: DISPLAYFORM0 Let x obs denote the observed elements of x, and x mis denote the missing elements according to the mask m. In addition, let θ denote the unknown parameters of the data distribution, and φ denote the unknown parameters for the mask distribution, which are usually assumed to be independent of θ.

In the standard maximum likelihood setting, the unknown parameters are estimated by maximizing the 1 Our implementation is available at https://github.com/steveli/misgan 2 The complementm is usually referred to as the missing data indicator in the literature.following marginal likelihood, integrating over the unknown missing data values:p(x obs , m) = p θ (x obs , x mis )p φ (m|x obs , x mis )dx mis .Little & Rubin (2014) characterize the missing data mechanism p φ (m|x obs , x mis ) in terms of independence relations between the complete data x = [x obs , x mis ] and the masks m:• Missing completely at random (MCAR): p φ (m|x) = p φ (m),• Missing at random (MAR): p φ (m|x) = p φ (m|x obs ),• Not missing at random (NMAR): m depends on x mis and possibly also x obs .Most work on incomplete data assumes MCAR or MAR since under these assumptions p(x obs , m) can be factorized into p θ (x obs )p φ (m|x obs ).

With such decoupling, the missing data mechanism can be ignored when learning the data generating model while yielding correct estimates for θ.

When p θ (x) does not admit efficient marginalization over x mis , estimation of θ is usually performed by maximizing a variational lower bound, as shown below, using the EM algorithm or a more general approach BID6 Ghahramani & Jordan, 1994) :log p θ (x obs ) ≥ E q(xmis|xobs) [log p θ (x obs , x mis ) − log q(x mis |x obs )] .The primary contribution of this paper is the development of a GAN-based framework for learning high-dimensional data distributions in the presence of incomplete observations.

Our framework introduces an auxiliary GAN for learning a mask distribution to model the missingness.

The masks are used to "mask" generated complete data by filling the indicated missing entries with a constant value.

The complete data generator is trained so that the resulting masked data are indistinguishable from real incomplete data that are masked similarly.

Our framework builds on the ideas of AmbientGAN (Bora et al., 2018) .

AmbientGAN modifies the discriminator of a GAN to distinguish corrupted real samples from corrupted generated samples under a range of corruption processes (or measurement processes).

For images, examples of the measurement processes include random dropout, blur, block-patch, and so on.

Missing data can be seen as a special type of corruption, except that we have access to the missing pattern in addition to the corrupted measurements.

Moreover, AmbientGAN assumes the measurement process is known or parameterized only by a few parameters, which is not the case in general missing data problems.

We provide empirical evidence that the proposed framework is able to effectively learn complex, highdimensional data distributions from highly incomplete data when the GAN generator incorporates suitable priors on the data generating process.

We further show how the architecture can be used to generate high-quality imputations.

In the missing data problem, we know exactly which entries in each data examples are missing.

Therefore, we can represent an incomplete data case as a pair of a partially-observed data vector x ∈ R n and a corresponding mask m ∈ {0, 1} n that indicates which entries in x are observed: x d is observed if m d = 1 otherwise x d is missing and might contain an arbitrary value that we should ignore.

With this representation, an incomplete dataset is denoted D = {(x i , m i )} i=1,...,N (we assume instances are i.i.d.

samples).

We choose this representation instead of x obs because it leads to a cleaner description of the proposed MisGAN framework.

It also suggests how MisGAN can be implemented efficiently in practice as both x and m are fixed-length vectors.

We begin by defining a masking operator f τ that fills in missing entries with a constant value τ : DISPLAYFORM0 wherem denotes the complement of m and denotes element-wise multiplication.

Two key ideas underlie the MisGAN framework.

First, in addition to the complete data generator, we explicitly model the missing data process using a mask generator.

Since the masks in the incomplete dataset are fully observed, we can estimate their distribution using a standard GAN.

Second, we train the complete data generator adversarially by masking its outputs using generated masks and f τ and comparing to real incomplete data that are similarly masked by f τ .Specifically, we use two generator-discriminator pairs (G m , D m ) and (G x , D x ) for the masks and data respectively.

In this paper, we focus on the missing completely at random (MCAR) case, where the two generators are independent of each other and have their own noise distributions p z and p ε .

We define the following two loss functions, one for the masks and the other for the data: DISPLAYFORM1 The losses above follow the Wasserstein GAN formulation BID1 , although the proposed framework is compatible with many GAN variations BID0 Berthelot et al., 2017; BID1 .

We optimize the generators and the discriminators according to the following objectives: DISPLAYFORM2 DISPLAYFORM3 where F x , F m are defined such that D x , D m are both 1-Lipschitz for Wasserstein GANs BID1 .

Practically, we follow the common practice of alternating between a few steps of optimizing the discriminators and one step of optimizing the generators BID0 BID1 BID1 .

The coefficient α is introduced when optimizing the mask generator G m with the aim of minimizing a combination of L m and L x .

Although in theory we could choose α = 0 to train G m and D m without using the data, we find that choosing a small value such as α = 0.2 improves performance.

This encourages the generated masks to match the distribution of the real masks and the masked generated complete samples to match masked real data.

The overall structure of MisGAN is illustrated in FIG0 .Note that the data discriminator D x takes as input the masked samples as if the data are fullyobserved.

This allows us to use any existing architecture designed for complete data to construct the data discriminator.

There is no need to develop customized neural network modules for dealing with missing data.

For example, D x can be a standard convolutional network for image applications.

Note that the masks are binary-valued.

Since discrete data generating processes have zero gradient almost everywhere, to carry out gradient-based training for GANs, we relax the output of the mask generator G m from {0, 1} n to [0, 1] n .

We employ a sigmoid activation σ λ (x) = 1/(1 + exp(−x/λ)) with a low temperature 0 < λ < 1 to encourage saturation and make the output closer to zero or one.

Finally, we note that the discriminator D x in MisGAN is unaware of which entries are missing in the masked input samples, and does not even need to know which value τ is used for masking.

In the next section, we present a theoretical analysis providing support for the idea that this type of masking process does not necessarily make it more difficult to recover the complete data distribution.

The experiments provide compelling empirical evidence for the effectiveness of the proposed framework.

In Section 2 we described how the discriminator D x in MisGAN takes as input the masked samples using (3) without knowing what value τ is used or which entries in the input vector are missing.

In this section, we discuss the following two important questions: i) Does the choice of the filled-in value τ affect the ability to recover the data distribution?

ii) Does information about the location of missing values affect the ability to recover the data distribution?We address these questions in a simplified scenario where each dimension of the data vector takes values from a finite set P. For n-dimensional data, let M = {0, 1}n be the set of all possible masks and I = P n be the set of all possible data vectors.

Also let D M and D I be the set of all possible probability distributions on M and I respectively, whose elements are non-negative and sum to one.

We first discuss the case where the filled-in value τ is chosen from P.Given τ ∈ P and q ∈ D M , we can construct a left transition matrix T q,τ ∈ R I×I defined below where the (t, s)-th entry specifies the transition probability from a data vector s ∈ I to an outcome t ∈ I masked by f τ , which involves all possible masks under which s is converted into t by filling in the indicated missing entries with τ : DISPLAYFORM0 Let p * x ∈ D I be the unknown true data distribution we want to estimate.

In the presence of missing data specified by q, the masked samples then follow the distribution p y = T q,τ p * x .

Without imposing extra application-specific constraints, MisGAN with a fixed mask generator can be viewed as solving the linear system p y = T q,τ p x , where p x ∈ D I is the unknown data distribution to solve for.

Here we assume that p y and T q,τ are given, as those can be estimated separately from a collection of fully-observed masks and masked samples.

Note that a transition matrix preserves the sum of the vectors it is applied to since 1 T q,τ = 1 .

For p x to be a valid distribution vector, we only need the non-negativity constraint because any solution p x automatically sums to one.

That is, estimating the data generating process in the presence of missing data based on the masking scheme used in MisGAN is equivalent to solving the linear system DISPLAYFORM1 In Theorem 1, we state a key property of the transition matrix T q,τ that leads to the answer to our questions.

The proof of Theorem 1 is in Appendix A. Theorem 1.

Given q ∈ D M , all transition matrices T q,τ with τ ∈ P have the same null space.

Theorem 1 implies that if the solution to the constrained linear system FORMULA7 is not unique for a given τ 0 ∈ P, that is, there exists some non-negative DISPLAYFORM2 In other words, we have the following corollary: Corollary 1.

Whether the true data distribution is uniquely recoverable is independent of the choice of the filled-in value τ .Here we only discuss the case when the probability of observing all features q(1) is zero, where q(1) denotes the scalar entry of q indexed by 1 ∈ M. Otherwise, the linear system is uniquely solvable as the transition matrix T q,τ0 has full rank.

With the non-negativity constraint, it is possible that the solution for the linear system FORMULA7 is unique when the true data distribution p * x is sparse.

Specifically, if there exists two indices s 1 , s 2 ∈ I such that p * x (s 1 ) = p * x (s 2 ) = 0 and also v(s 1 ) > 0 and v(s 2 ) < 0 for all v ∈ Null(T q,τ ) \ {0}, then the solution to (8) is unique.

Sparsity of the data distribution is a reasonable assumption in many situations.

For example, natural images are typically considered to lie on a low dimensional manifold, which means most of the instances in I should have almost zero probability.

On the other hand, when the missing rate is high, that is, if the masks in M that have many zeros are more probable, the null space of T q,τ will be larger and therefore it is more likely that the non-negative solution is not unique.

Bruckstein et al. (2008) proposed a sufficient condition on the sparsity of the non-negative solutions to a general underdetermined linear system that guarantees unique optimality.

Next we note that in the case of τ ∈ P, an entry with value τ in a masked sample t ∈ I may come either from an observed entry with value τ in the unmasked sample or from an unobserved entry through the masking operation in (3).

One might wonder if this prevents an algorithm from recovering the true distribution when it is otherwise possible to do so.

In other words, if we take the location of the missing values into account, would that make the missing data problem less ill-posed?

However, this is not the case, as we state in Corollary 2.

The proof is in Appendix B where we discuss the case of τ / ∈ P. Corollary 2.

If the linear system T q,τ p x = T q,τ p * x does not have a unique non-negative solution, then for this missing data problem, we cannot uniquely recover the true data distribution even if we take the location of the missing values into account.

Note that the analysis in this section characterizes how difficult the missing data problem is, which is independent of the choice of the algorithm that solves it.

In practice, it is useful to incorporate application-specific prior knowledge into the model to regularize the problem when it is ill-posed.

For example, for modeling natural images, convolutional networks are commonly used to exploit the local structure of the data.

In addition, decoder-based deep generative models such as GANs implicitly enforce some sparsity constraints due to the use of low dimensional latent codes in the generator, which also helps to regularize the problem.

Finally, the following theorem justifies the training objective (6) of MisGAN for the missing data problem (see Appendix A for details).

Theorem 2.

Given a mask distribution p φ (m), two distributions p θ (x) and p θ (x) induce the same distribution for f τ (x, m) if and only if they have the same marginals p θ (x obs |m) = p θ (x obs |m) for all masks m with p φ (m) > 0.

Missing data imputation is an important task when dealing with incomplete data.

In this section, we show how to impute missing data according to p(x mis |x obs ) by equipping MisGAN with an imputer G i accompanied by a corresponding discriminator D i .

The imputer is a function of the incomplete example (x, m) and a random vector ω drawn from a noise distribution p ω .

It outputs the completed sample with the observed part in x kept intact.

To train the imputer-equipped MisGAN, we define the loss for the imputer in addition to (4) and (5): DISPLAYFORM3 We jointly learn the data generating process and the imputer according to the following objectives: DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where we use β = 0.1 in the experiments when optimizing G x .

This encourages the generated complete data to match the distribution of the imputed real data in addition to having the masked generated data match the masked real data.

The overall structure for MisGAN imputation is illustrated in Figure 2 .We can also train a stand-alone imputer using only (9) with a pre-trained data generator G x .

The architecture is as shown in Figure 2 with the faded parts removed.

Moreover, it is also possible to train the imputer to target a different missing distribution p m with a pre-trained data generator G x alone without access to the original (incomplete) training data: DISPLAYFORM7 We construct the imputer G i (x, m, ω) as follows: DISPLAYFORM8 Figure 2: Architecture for MisGAN imputation.

The complete data generator G x and the imputer G i can be trained jointly with all the components.

We can also independently train the imputer G i without the faded parts if the data generator G x has been pre-trained.where G i generates the imputed result with the same dimensionality as its input, x m + ω m, which could be implemented by a deep neural network.

The masking outside of G i ensures that the observed part of x stays the same in the output of the imputer G i .

The similar masking on the input of G i , x m + ω m, ensures that the amount of noise injected to G i scales with the number of missing dimensions.

This is intuitive in the sense that when a data case is almost fully-observed, we expect less variety in p(x mis |x obs ) and vice versa.

Note that the noise ω needs to have the same dimensionality as x.

In this section, we first assess various properties of MisGAN on the MNIST dataset: we demonstrate qualitatively how MisGAN behaves under different missing patterns and different architectures.

We then conduct an ablation study to justify the construction of MisGAN.

Finally, we compare MisGAN with various baseline methods on the missing data imputation task over three datasets under a series of missingness settings.

Data We evaluate MisGAN on three datasets: MNIST, CIFAR-10 and CelebA. MNIST is a dataset of handwritten digits images of size 28×28 BID5 .

We use the provided 60,000 training examples for the experiments.

CIFAR-10 is a dataset of 32×32 color images from 10 classes BID4 .

Similarly, we use 50,000 training examples for the experiments.

CelebA is a large-scale face attributes dataset BID7 that contains 202,599 face images, where we use the provided aligned and cropped images and resize them to 64×64.

For all three datasets, the range of pixel values of each image is rescaled to [0, 1].Missing data distributions We consider three types of missing data distribution: i) Square observation: all pixels are missing except for a square occurring at a random location on the image.

ii) Dropout: each pixel is independently missing according to a Bernoulli distribution.

iii) Variablesize rectangular observation: all pixels are missing except for a rectangular observed region.

The width and height of the rectangle are independently drawn from 25% to 75% of the image length uniformly at random, which results in a 75% missing rate on average.

In this missing data distribution, each example may have a different number of missing pixels.

The highest per-example missing data rate under this mechanism is 93.75%.Evaluation metric We use the Fréchet Inception Distance (FID) BID2 to evaluate the quality of the learned generative model.

For MNIST, instead of the Inception network trained on ImageNet BID13 , we use a basic LeNet model 4 trained on the complete MNIST training set, and then take the 50-dimensional output from the second-to-last fully-connected layer as the features to compute the FID.

For CIFAR-10 and CelebA, we follow the procedure described in BID2 to compute the FID using the pretrained Inception-v3 model.

When evaluating generative models using the FID, we use the same number of generated samples as the size of the training set.

In this section, we study various properties of MisGAN using the MNIST dataset.

Architectures We consider two kinds of architecture for MisGAN: convolutional networks and fully connected networks.

We follow the DCGAN architecture BID11 for (de)convolutional generators and discriminators to exploit the local structures of images.

We call this model ConvMisGAN.To demonstrate the performance of MisGAN in the absence of the implicit structural regularization provided by the use of a convolutional network, we construct another MisGAN with only fullyconnected layers for both the generators and the discriminators, which we call FC-MisGAN.In the experiments, both Conv-MisGAN and FC-MisGAN are trained using the improved procedure for the Wasserstein GAN with gradient penalty BID1 ).

Throughout we use τ = 0 for the masking operator and the temperature λ = 0.66 for the mask activation σ λ (x) described in Section 2.Baseline We compare MisGAN to a baseline model that is capable of learning from large-scale incomplete data: the generative convolutional arithmetic circuit (ConvAC) BID14 .

ConvAC is an expressive mixture model similar to sum-product networks BID10 with a compositional structure similar to deep convolutional networks.

Most importantly, ConvAC admits tractable marginalization due to the product form of the base distributions for the mixtures, which makes it readily capable of learning with missing data.

Results Figures 3 and 4 show the generated data samples as well as the learned mask samples produced by Conv-MisGAN and FC-MisGAN under the square observation and independent dropout missing mechanisms.

From these results, we can see that Conv-MisGAN produces visually better samples than FC-MisGAN on this problem.

On the other hand, under the same missing rate, independent dropout leads to worse samples than square observations.

Samples generated by ConvAC are shown in FIG0 in Appendix G.We quantitatively evaluate Conv-MisGAN, FC-MisGAN and ConvAC under two missing patterns with missing rates from 10% to 90% with a step of 10%.

Figure 5 shows that MisGAN in general outperforms ConvAC as ConvAC tends to generate samples with aliasing artifacts as shown in FIG0 .

It also shows that in the square observation case, Conv-MisGAN and FC-MisGAN have similar performance in terms of their FIDs.

However, under independent dropout, the performance of FC-MisGAN degrades significantly as the missing rate increases compared to Conv-MisGAN.

This is because independent dropout with high missing rate makes the problem more challenging as it induces less overlapping co-occurrence among pixels, which degrades the signal for understanding the overall structure.

This is illustrated in Figure 6 where the observed pattern comes from one of four equally probable 14×14 square quadrants with no overlap.

Clearly this missing data problem is ill-posed and we could never uniquely determine the correlation between pixels across different quadrants without additional assumptions.

The samples generated by the FC-MisGAN produce obvious discontinuity across the boundary of the quadrants as it does not rely on any prior knowledge about how pixels are correlated.

The discontinuity artifact is less severe with Conv-MisGAN since the convolutional layers encourage local smoothness.

This shows the importance of incorporating prior knowledge into the model when the problem is highly ill-posed.

Ablation study We point out that the mask discriminator in MisGAN is important for learning the correct distribution robustly.

Figure 7 shows two common failure scenarios that frequently happen with an AmbientGAN, which is essentially equivalent to a MisGAN without the mask discriminator.

Figure 7 (left) shows a case where AmbientGAN produces perfectly consistent masked outputs, but the learned mask distribution is completely wrong.

Since we use f τ =0 (x, m) = x m, it makes the role of x and m interchangeable when considering only the masked outputs.

Even if we rescale the range of pixel values from [0, 1] to [−1, 1] to avoid this situation, AmbientGAN still fails often as shown in Figure 7 (right).

In contrast, MisGAN avoids learning such degenerate solutions due to explicitly modeling the mask distribution.

Missing data imputation We construct the imputer network G i defined in (12) using a three-layer fully-connected network with 500 hidden units in the middle layers.

imputation results on different examples applying novel masks randomly drawn according to the same distribution.

FIG5 (right) shows the imputation results where each row corresponds to the same incomplete input.

It demonstrates that the imputer can produce a variety of different imputed results due to the random noise input to the imputer.

We also note that if we modify (11) to train the imputer together with the data generator from scratch without the mask generator/discriminator, it fails most of the time for a similar reason to why AmbientGAN fails.

The learning problem is highly ill-posed without the agreement on the mask distribution.

In this section, we quantitatively evaluate the performance of MisGAN on three datasets: MNIST, CIFAR-10, and CelebA. We focus on evaluating MisGAN on the missing data imputation task as it is widely studied and many baseline methods are available.

Baselines We compare the MisGAN imputer to a range of baseline methods including the basic zero/mean imputation, matrix factorization, and the recently proposed Generative Adversarial Imputation Network (GAIN) BID16 .

GAIN is an imputation model that employs an imputer network to complete the missing data.

It is trained adversarially with a discriminator that determines which entries in the completed data were actually observed and which were imputed.

It has shown to outperform many state-of-the-art imputation methods.

We impute all of the incomplete examples in the training set and use the FID between the imputed data and the original fully-observed data as the evaluation metric.

Architecture We use convolutional generators and discriminators for MisGAN for all experiments in this section.

For MNIST, we use the same fully-connected imputer network as described in the previous section; for CIFAR-10 and CelebA, we use a five-layer U-Net architecture BID12 for the imputer network G i in MisGAN.Results We compare all the methods under two missing patterns, square observation and independent dropout, with missing rates from 10% to 90%.

FIG6 shows that MisGAN consistently outperforms other methods in all cases, especially under high missing rates.

In our experiments, we found GAIN training to be quite unstable for the block missingness.

We also observed that there is a "sweet spot" for the number of training epochs when training GAIN.

If trained longer, the imputation behavior will gradually become similar to constant imputation (see Appendix H for details).

On the other hand, we find that training MisGAN is more stable than training GAIN across all scenarios in the experiments.

The imputation results of MisGAN and GAIN are shown in Appendix E, F, and H.

This work presents and evaluates a highly flexible framework for learning standard GAN data generators in the presence of missing data.

Although we only focus on the MCAR case in this work, MisGAN can be easily extended to cases where the output of the data generator is provided to the mask generator.

These modifications can capture both MAR and NMAR mechanisms.

The question of learnability requires further investigation as the analysis in Section 3 no longer holds due to dependence between the transition matrix and the data distribution under MAR and NMAR.

We have tried this modified architecture in our experiments and it showed similar results as to the original MisGAN.

This suggests that the extra dependencies may not adversely affect learnability.

We leave the formal evaluation of this modified framework for future work.

A PROOF OF THEOREM 1 AND THEOREM 2Let P be the finite set of feature values.

For the n-dimensional case, let M = {0, 1} n be the set of masks and I = P n be the set of all possible feature vectors.

Also let D M be the set of probability distributions on M, which implies m 0 and v∈I m(v) = 1 for all m ∈ M, where m(v) denotes the entry of m indexed by v.

Given τ ∈ P and q ∈ D M , define the transformation DISPLAYFORM0 where is the entry-wise multiplication and 1{·} is the indicator function.

Given m ∈ M, define an equivalent relation ∼ m on I by v ∼ m u iff v m = u m, and denote by [v] m the equivalence class containing v.

Given q ∈ D M , let S q ⊂ M be the support of q, that is, DISPLAYFORM1 Given τ ∈ P and v ∈ I, let M τ,v denote the set of masks consistent with v in the sense that q(m) > 0 and v m = τm, that is, DISPLAYFORM2 Proof.

This is clear from the following equation DISPLAYFORM3 which can be obtained from (13) as follows, DISPLAYFORM4 Proposition 2.

For any τ ∈ P, q ∈ D M and x ∈ R I , the vector T q,τ x determines the collection of marginals {x ([v] DISPLAYFORM5 Proof.

Fix τ ∈ P, q ∈ D M and x ∈ R I .

Since v m + τm ∈ [v] m , it suffices to show that we can solve for x ([v] m ) in terms of T q,τ x for m ∈ M τ,v = ∅. We use induction on the size of M τ,v .First consider the base case |M τ,v | = 1.

Consider v 0 ∈ I with M τ,v0 = {m 0 }.

By FORMULA0 , DISPLAYFORM6 , which proves the base case.

Now assume we can solve for x ([v] m ) in terms of T q,τ x for m ∈ S q and v ∈ I with |M τ,v | ≤ k. Consider v 0 ∈ I with |M τ,v0 | = k + 1; if no such v 0 exists, the conclusion holds trivially.

Let M τ,v0 = {m 0 , m 1 , . . .

, m k }.

We need to show that T q,τ x determines x([v 0 ] m ) for = 0, 1, . . .

, k. By (14) again, DISPLAYFORM7 Let m = k =0 m , which may or may not belong to S q .

Note that DISPLAYFORM8 and hence DISPLAYFORM9 Plugging FORMULA0 into FORMULA0 yields DISPLAYFORM10 Note that DISPLAYFORM11 It follows from FORMULA0 and FORMULA0 Theorem 1 is a direct consequence of Proposition 1 and Proposition 2 as the collection of marginals {x ([v] m ) : v ∈ I, m ∈ S q } is independent of τ .

Therefore, if x 1 , x 2 ∈ R I satisfy T q,τ0 x 1 = T q,τ0 x 2 for some τ 0 ∈ P, then T q,τ x 1 = T q,τ x 2 for all τ ∈ P. Theorem 1 is a special case when x 1 = 0.Moreover, Proposition 2 also shows that MisGAN overall learns the distribution p(x obs , m), as x([v] m ) is equivalent to p(x obs |m) and T q,τ x is essentially the distribution of f τ (x, m) under the optimally learned missingness q = p(m).

Theorem 2 basically restates Proposition 1 and Proposition 2.

This is also true when τ / ∈ P according to Appendix B.

Corollary 2 can be shown by augmenting the set of feature values by P = P ∪ {ψ} with a novel symbol ψ / ∈ P. If we choose τ = ψ for the masking operator, whenever we spot a ψ in a masked sample, we know that it corresponds to a missing entry.

We can also construct the corresponding transition matrix T q,ψ ∈ R I ×I where I = (P ) n given the mask distribution q ∈ D M before.

In this setting, the generative model for missing data is equivalent to solving the linear system T q,ψ p x = T q,ψ p * x so that p x ∈ R I is non-negative and p x (s) = 0 for all s ∈

I \

I, where the true distribution p * x is given by p * x (s) = p * x (s) for all s ∈ I and zeros elsewhere.

Theorem 1 implies that if the solution to original problem FORMULA7 is not unique, the non-negative solution to the augmented linear system with the extra constraint on I \ I with τ = ψ is not unique either.

Root mean square error (RMSE) is a commonly used metric for evaluating the performance of missing data imputation, which computes the RMSE of the imputed missing values against the ground truth.

However, in a complex system, the conditional distribution p(x mis |x obs ) is likely to be highly multimodal.

It's not guaranteed that the ground truth of the missing values in the incomplete dataset created under the missing completely at random (MCAR) assumption correspond to the global mode of p(x mis |x obs ).

A good imputation model might produce samples from p(x mis |x obs ) associated with a higher density than the ground truth (or from other modes that are similarly probable).

In this case, it will lead to a large error in terms of metrics like RMSE as multiple modes might be far away from each other in a complex distribution.

Therefore, we instead compute the FID between the distribution of the completed data and the distribution of the originally fully-observed data as our evaluation metric.

This provides a practical way to assess how close a model imputes according to p(x mis |x obs ) by comparing two groups of samples collectively.

As a concrete example, FIG0 compares the two evaluation metrics on MNIST, our distributionbased FID and the ground truth-based RMSE.

It shows that the rankings on most of the missing rates are not consistent across the two metrics.

In particular, under 90% missing rate, MisGAN is worse than GAIN and matrix factorization in terms of RMSE, but significantly better in terms of FID.

FIG0 plots the imputation results of the three methods mentioned above.

We can clearly see that MisGAN produces the best completion even though its RMSE is much higher than the other two.

It's not surprising as the mean of p(x mis |x obs ) minimizes the squared error in expectation, even if the mean might have low density.

This probably explains why the blurry completion results produced by matrix factorization achieve the lowest RMSE.

All of the generators and discriminators in Conv-MisGAN follow the architecture used by the DCGAN model BID11 with 128-dimensional latent code.

As For the imputer network for MisGAN trained on CIFAR-10 and CelebA, we follow the U-Net implementation of the CycleGAN and pix2pix work 6 .

In the experiments, we use 5-layer U-Nets for both CIFAR-10 and CelebA.For training Wasserstein GAN with gradient penalty, We use all the default hyperparameters reported in BID1 .

For all the datasets, MisGAN is trained for 300 epochs.

We train MisGAN imputer for 1000 epochs for MNIST and CIFAR-10 as the networks are smaller and 600 epochs for CelebA.For ConvAC, we use the same architecture described in BID14 .

We train ConvAC for 1000 epochs using Adam optimizer with learning rate 10 G RESULTS OF CONVAC FIG0 shows the samples generated by ConvAC trained with the square observation missing pattern on MNIST.H MISSING DATA IMPUTATION WITH GAIN FIG0 shows the imputation results of GAIN on different epochs during training with the 20×20 square observation missingnss.

We found that this is a common phenomenon for the square observation missing pattern.

To obtain better results for GAIN, we analyze the FIDs during the course of training and use the model that achieves the best FID to favorably compare with MisGAN for the square observation case.

For CIFAR-10, we use the results from the 500th epoch; for CelebA, we use the results from the 50th epoch.

Otherwise, we train GAIN for 1000 epochs for CIFAR-10 and 300 epochs for CelebA. Our implementation is adapted from the code released by the authors of GAIN.

@highlight

This paper presents a GAN-based framework for learning the distribution from high-dimensional incomplete data.