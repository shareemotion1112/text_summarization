This paper presents the Variation Network (VarNet), a  generative model providing means to manipulate the high-level attributes of a given input.

The originality of our approach is that VarNet is not only capable of handling pre-defined attributes but can also learn the relevant attributes of the dataset by itself.

These two settings can be easily combined  which makes VarNet applicable for a wide variety of tasks.

Further, VarNet has a sound probabilistic interpretation which grants us with  a novel way to navigate in the latent spaces as well as means to control how the  attributes are learned.

We demonstrate  experimentally that this model is capable of performing interesting input manipulation  and that the learned attributes are relevant and interpretable.

We focus on the problem of generating variations of a given input in an intended way.

This means that given some input element x, which can be considered as a template, we want to generate transformed versions of x with different high-level attributes.

Such a mechanism is of great use in many domains such as image edition since it allows to edit images on a more abstract level and is of crucial importance for creative uses since it allows to generate new content.

More precisely, given a dataset D = {(x (1) , m (1) ), . . .

, (x (N ) , m (N ) )} of N labeled elements (x, m) ∈ X × M, where X stands for the input space and M for the metadata space, we would like to obtain a model capable of learning a relevant attribute space Ψ ⊂ R d for some integer d > 0 and meaningful attribute functions φ : X × M → Ψ that we can then use to control generation.

In a great majority of the recent proposed methods BID13 ; BID16 , these attributes are assumed to be given.

We identify two shortcomings: labeled data is not always available and this approach de facto excludes attributes that can be hard to formulate in an absolute way.

The novelty of our approach is that these attributes can be either learned by the model (we name them free attributes) or imposed (fixed attributes).

This problem is an ill-posed one on many aspects.

Firstly, in the case of fixed attribute functions φ, there is no ground truth for variations since there is no x with two different attributes.

Secondly, it can be hard to determine if a learned free attribute is relevant.

However, we provide empirical evidence that our general approach is capable of learning such relevant attributes and that they can be used for generating meaningful variations.

In this paper, we introduce the Variation Network (VarNet), a probabilistic neural network which provides means to manipulate an input by changing its high-level attributes.

Our model has a sound probabilistic interpretation which makes the variations obtained by changing the attributes statistically meaningful.

As a consequence, this probabilistic framework provides us with a novel mechanism to "control" or "shape" the learned free attributes which then gives interpretable controls over the variations.

This architecture is general and provides a wide range of choices for the design of the attribute function φ: we can combine both free and fixed attributes and the fixed attributes can be either continuous or discrete.

Our contributions are the following:• A widely applicable encoder-decoder architecture which generalizes existing approaches BID11 ; BID14 ; BID13 The input x,x are in X , the input space and the metadata m is in M, the metadata space.

The latent template code z * lies in Z * , the template space, while the latent variable z lies in Z the latent space.

The variable u is sampled from a zero-mean unitvariance normal distribution.

Finally, the features φ(x, m) are in Ψ, the attribute space.

The Neural Autoregressive Flows (NAF) BID10 are represented using two arrows, one pointing to the center of the other one; this denotes the fact that the actual parameters of first neural network are obtained by feeding meta-parameters into a second neural network.

The discriminator D acts on Z * × Ψ.• An easy-to-use framework: any encoder-decoder architecture can be easily transformed into a VarNet in order to provide it with controlled input manipulation capabilities,• A novel and statistically sound approach to navigate in the latent space,• Ways to control the behavior of the free learned attributes.

The plan of this paper is the following: Sect.

2 presents the VarNet architecture together with its training algorithm.

For better clarity, we introduce separately all the components featured in our model and postpone the discussion about their interplay and the motivation behind our modeling choices in Sect.

3 and Sect.

4 discusses about the related works.

In particular, we show that VarNet provides an interesting solution to many constrained generation problems already considered in the literature.

Finally, we illustrate in Appendix A the possibilities offered by our proposed model and show that its faculty to generate variations in an intended way is of particular interest.

We now introduce our novel encoder-decoder architecture which we name Variation Network.

Our architecture borrows principles from the traditional Variational AutoEncoder (VAE) architecture BID11 and from the Wasserstein AutoEncoder (WAE) architecture BID15 ; BID14 .

It uses an adversarially learned regularization BID5 ; BID13 , introduces a separate latent space for templates BID0 and decomposes the attributes on an adaptive basis BID17 .

It can be seen as a VAE with a particular decoder network or as a WAE with a particular encoder network.

Our architecture is shown in FIG0 and our training algorithm is presented in Alg.

1.We detail in the following sections the different parts involved in our model.

In Sect.

2.1, we focus on the encoder-decoder part of VarNet and explain Eq. (3), (4) and (5).

In Sect.

2.2, we introduce the adversarially-learned regularization whose aim is to disentangle attributes from templates (Eq. FORMULA3 and FORMULA7 ).

Section 2.3 discusses the special parametrization that we adopted for the attribute space Ψ.

Require: DISPLAYFORM0 , reconstruction cost c, reproducing kernel k, batch size n 1: for Fixed number of iterations do 2:Sample x := (x 1 , . . .

, x n ) and m := (m 1 , . . .

, m n ) where DISPLAYFORM1 Compute z := {z 1 , . . .

, z n } where DISPLAYFORM2 Samplex := {x 1 , . . .

,x n } wherex i ∼

p(·|z i ),

Sample random features {ψ i } i=1..

n from feature space Ψ using ν (see Sect.

2.3) 7:Letz := {z 1 , . . .

,z n } wherez i ∼ p(·)

Discriminator training phase DISPLAYFORM0 10:Gradient ascent step on the discriminator parameters using ∇L Disc

Encoder-decoder training phase 12: DISPLAYFORM0 where DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 13:Gradient ascent step on all parameters except the discriminator parameters (encoder and decoder parameters, feature function parameters, features vectors and NAF f ) using ∇L EncDec 14: end for

Similar to the VAE architectures, we suppose that our data x ∈ X depends on some latent variable z ∈ Z through some decoder p(x|z) parametrized by a neural network.

We introduce a prior p(z) over this latent space so that the joint probability distribution is expressed as p(x, z) = p(x|z)p(z).

Since the posterior distribution p(z|x) is usually intractable, an approximate posterior distribution q(z|x) parametrized by a neural network is usually introduced.

The novelty of our approach is on how we write this encoder network.

Firstly, we introduce an attribute space Ψ ⊂ R d , where d is the dimension of the attribute space, on which we condition the encoder which we now denote as q(·|x, ψ ∈ Ψ).

More details about the attribute space Ψ are given in Sect.

2.3.

For the moment, we can consider it to be a subspace of R d from which we can sample from.

The objective in doing so is that decoding z ∼ q(·|x, ψ) using p(x|z) will result in a samplex that is a variation of x but with features ψ.

Secondly, in order to correctly reconstruct x, introduce an attribute function φ : X × M → Ψ computed from x and its metadata m with values in the attribute space Ψ. This attribute function is a deterministic neural network that will be learned during training and whose aim is to compute attributes of x.

For an input (x, m) ∈ D, we want to decouple a template obtained from x from its attributes φ(x, m) computed from x and (possibly) from its metadata m. This is done by introducing another latent space Z * that we term template space together with a approximated posterior distribution q * (z * |x)parametrized by a neural network and a fixed prior p * (z * ).

The idea is then to compute z from z * by applying a transformation parametrized only by the feature space Ψ. In practice, this is done by using a Neural Autoregressive Flow (NAF) BID10 f ψ : Z * → Z parametrized by ψ ∈ Ψ. Neural autoregressive flows are universal density estimation models which are capable of sampling any random variable Y by applying a learned transformation over a base random variable X (Thm. 1 in BID10 ).Given a reconstruction loss c on X , we have the following mean reconstruction loss: DISPLAYFORM0 We regularize the latent spaces Z * and Z by adding the usual KL term appearing in the VAE Evidence Lower Bound (ELBO) on Z * : DISPLAYFORM1 and an MMD-based regularization on Z similar the one used in WAEs (see Alg.

2 in BID15 ): DISPLAYFORM2 where k : Z ×Z → R is an positive-definite reproducing kernel and H k the associated Reproducing Kernel Hilbert Space (RKHS) BID1 .The equations FORMULA5 , FORMULA6 and (5) of Alg.

1 are estimators on a mini-batch of size n of equations FORMULA8 , FORMULA9 and FORMULA10 respectively, (5) being the unbiased U-statistic estimator of (9) BID7 .

Our encoder q(z|x, ψ) thus depends exclusively on x and on the feature space Ψ. However, there is no reason, for a random attribute ψ ∈ Ψ = φ(x, m), that p(x|z) where z ∼ q(z|x, φ) generates variations of the original x with features φ.

Indeed, all needed information for reconstructing x is potentially already contained in z.

We propose to add an adversarially-learned cost on the latent variable z * to force the encoder q * to discard information about the attributes of x: Specifically, we train a discriminator neural network D : Z * ×Ψ → [0, 1] whose role is to evaluate the probability D(z * , ψ) that there exists a (x, m) ∈ D such that ψ = φ(x, m) and z * ∼ q * (·|x).

In other words, the aim of the discriminator is to determine if the attributes ψ and the template code z * originate from the same (x, m) ∈ D or if the features ψ are randomly generated.

We postpone the explanation on how we sample random features ψ ∈ Ψ in Sect.

2.3 and suppose for the moment that we have access to a distribution ν(ψ) over Ψ from which we can sample.

The encoder-decoder architecture presented in Sect.

2.1 is trained to fool the discriminator: this means that for a given (x, m) ∈ D it tries to produce a template code z * ∼ q * (·|x) which contains no information about the features φ(x, m).In an optimal setting, i.e. when the discriminator is unable to match any z * ∈ Z * with a particular feature ψ ∈ Ψ, the space of template codes and the space of attributes are decorrelated.

All the missing information needed to reconstruct x given z * ∼ q * (·|x) lies in the transformation f φ(x,m) .

Since these transformations between the template space Z * and the latent space Z only depend on the feature space Ψ, they tend to be applicable over all template codes z * and generalize well.

During generation time, it is then possible to change the attributes of a sample without changing its template.

The discriminator is trained to maximize DISPLAYFORM0 while the encoder-decoder architecture is trained to minimize DISPLAYFORM1 Estimators of Eq. FORMULA3 and FORMULA3 are given by Eq. FORMULA3 and FORMULA7 respectively.

We adopt a particular parametrization of our attribute function φ : X × M so that we are able to sample fake attributes without the need to rely on an existing (x, m) ∈ D pair.

In the following, we make a distinction between two different cases: the case of continuous free attributes and the case of fixed continuous or discrete attributes.

In order to handle free attributes, which denote attributes that are not specified a priori but learned.

For this, we introduce d Ψ attribute vectors v i of dimension d together with an attention module α : X × M → [0, 1] dΨ , where d Ψ is the intrinsic dimension of the attribute space Ψ. By denoting α i the coordinates of α, we then write our attribute function φ as DISPLAYFORM0 This approach is similar to the style tokens approach presented in BID17 .

The v i 's are global and do not depend on a particular instance (x, m).

By varying the values of the α i 's between [0, 1], we can then span a d Ψ -dimensional hypercube in R d which stands for our attribute space Ψ. It is worth noting that the v i 's are also learned and thus constitute an adaptive basis of the attribute space.

In order to define a probability distribution ν over Ψ (note that this subspace also varies during training), we are free to choose any distribution ν α over [0, 1] dΨ .

We then sample random attributes from ν by In the continuous case, we write our attribute function DISPLAYFORM1 DISPLAYFORM2 while in the discrete case, we just consider DISPLAYFORM3 where e m is a d Ψ -dimensional embedding of the symbol m. It is important to note that even if the attributes are fixed, the v i 's or the embeddings e m are learned during training.

These two equations define a natural probability distribution ν over Ψ: DISPLAYFORM4

We now detail our objective (2) and notably explain our particular choice concerning the regularizations on the latent spaces Z * and Z. In Sect.

3.1, we will see that these insights suggest an additional way to "control" the influence of the learned free attributes.

In Sect.

3.2, we further discuss about the multiple possibilities that we have concerning the implementation of the attribute function.

We list, in Sect.

3.3, the different sampling schemes of VarNet.

Finally, Sect.

3.4 is dedicated to implementation details.

We discuss our choice concerning the regularizations of the latent spaces and specifically why we chose a KL regularization on Z * and an MMD loss on Z.We found that using a MMD-based regularization on the template space Z * resulted in approximated posterior distributions q * (·|x) with very small variances (almost deterministic mappings).

One explanation of this behavior is that the MMD regularization tries to enforce that the aggregated posterior DISPLAYFORM0 ) matches the prior p * : it does not act on the individual conditional probability distributions q * (·|x).

This degenerate behavior is a side-effect of our adversarial regularization since stochastic encoders have been successfully used in WAEs BID14 .

When using the the Kullback-Leibler regularization on Z * , this effect disappear which makes the KL regularization that we considered more suited for VarNet since it helps to keep our model out of a degenerate regime.

For some applications, it can still be of interest to have a control over the variance of the conditional probability distributions q * (·|x).

Similar to the approach of BID9 ; BID2 , we propose to multiply the KL term by a scalar parameter β > 0.

For β = 1, we retrieve the original formulation.

For β ∈]0, 1[, decreasing the value of β from one to zero decreases the variance of the q * (·|x).

We found no gain in considering values of β greater than 1.

Examples where this tuning provides an interesting application are given in Sect.

A.2.We now consider the regularization over Z. This regularization is in fact superfluous and could be removed.

However, we noticed that adding this MMD regularization helped obtaining better reconstruction losses.

In this section, we focus on the parametrization of the attribute function φ : X × Z → R d and propose some useful use cases.

The formulation of Sect.

2.3 is in fact too restrictive and considered only one attribute function.

It is in fact possible to mix different attributes functions by simply concatenating the resulting vectors.

By doing so, we can then combine free and fixed attributes in a natural way but also consider different attention modules α.

We can indeed use neural networks with different properties similarly to what is done in BID4 but also consider different distributions over the attention vectors α i .It is important to note that the free attributes presented in Sect.

2.3.1 can only capture global attributes, which are attributes that are relevant for all elements of the dataset D. In the presence of discrete labels m, it can be interesting to consider label-dependent free attributes, which are attributes specific to a subset of the dataset.

In this case, the attribute function φ can be written as DISPLAYFORM0 where e m,i designates the i th attribute vector of the label m. With all these possibilities at hand, it is possible to devise numerous applications in which the notions of template and attribute of an input x may have diverse interpretations.

Our choice of using a discriminator over Ψ instead of, for instance, over the values of α themselves allow to encompass within the same framework discrete and continuous fixed attributes.

This makes the combinations of such attributes functions natural.

We quickly review the different sampling schemes of VarNet.

We believe that this wide range of usages makes VarNet a promising model for a wide range of applications.

We can for instance:• generate random samplesx from the estimated dataset distribution: DISPLAYFORM0 • samplex with given attributes ψ: DISPLAYFORM1 • generate a variations of an input x with attributes ψ: DISPLAYFORM2 • generate random variations of an input x: x ∼ p(·|z) with z = f ψ (z * ) where z * ∼ q * (·|x) and ψ ∼ ν(·).Note that for sampling generate random samplesx, we do that by sampling z * ∼ p * (·) from the prior, ψ ∼ ν(·) from the distribution of the attributes and then decoding z = f ψ (z * ) decoding it using the decoder p(·|z) instead of just decoding a z * ∼ p * (·) sampled from the prior.

This is due to the fact that, as already mentioned, this MMD regularization is not an essential element of the VarNet architecture: its role is more about fixing the "scale" of the Z space rather than enforcing that the aggregated posterior distribution exactly matches the prior.

In the case of continuous attributes of the form Eq. FORMULA3 or FORMULA3 , VarNet also provides a new way to navigate in the latent space Z. Indeed, for a given template latent code z * , it is possible to move continuously in the latent space Z by simply changing continuously the values of the α i and then DISPLAYFORM3 The image by the above transformation in the Z space of the d Ψ dimensional hypercube [0, 1] d ψ constitutes the space of variations of the template z * .

Since our feature space bears a measure ν, this space of variations has a probabilistic interpretation.

To the best of our knowledge, we think that it is the first time that a meaningful probabilistic interpretation about the displacement in the latent space in terms of attributes is given: We'll see in Appendix A.3 that two similar variations applied on different templates can induce radically different displacements in the latent space Z. We hope that this new technique will be useful in many applications and help go beyond the traditional (but unjustified) linear or spherical interpolations BID18 .

Our architecture is general and any decoder and encoder networks can be used.

We chose to use a NAF 1 for our encoder network.

This choice has the advantage of using a more expressive posterior distribution compared to the often-used diagonal Gaussian posterior distributions.

Our priors p * and p are zero-mean unit-variance Gaussian distributions.

For the MMD regularization, we used the parameters used in BID15 (λ = 10 and k(x, y) = C/(C+ x−y 2 2 ) the inverse multiquadratics kernel with C = 2dim(Z)).

For the scalar coefficient γ, we found that a value of 10 worked well on all our experiments.

For the sampling of the α values in the free attributes case, we considered ν α to be a uniform distribution over [0, 1] d ψ .

In the fixed attribute case, we simply obtain a random sample {ψ i } n i=1 by shuffling the already computed batches of {φ(x i , m i )} n i=1 (lines 4 and 6 in Alg.1).

The Variation Network generalizes many existing models used for controlled input manipulation by providing a unified probabilistic framework for this task.

We now review the related literature and discuss the connections with VarNet.

The problem of controlled input manipulation has been considered in the Fader networks paper BID13 , where the authors are able to modify in a continuous manner the attributes of an input image.

Similar to us, this approach uses an encoder-decoder architecture together with an adversarial loss used to decouple templates and attributes.

The major difference with VarNet is that this model has a deterministic encoder which limits the sampling possibilities as discussed in Sect.

A.2.

Also, this approach can only deal with fixed attributes while VarNet is able to also learn meaningful free attributes.

In fact, VAEs BID11 , WAEs Tolstikhin et al. (2017) ; BID14 and Fader networks can be seen as special cases of VarNet.

Recently, the Style Tokens paper BID17 proposed a solution to learn relevant free attributes in the context of text-to-speech.

The similarities with our approach is that the authors condition an encoder model on an adaptive basis of style tokens (what we called attribute space in this work).

VarNet borrows this idea but cast it in a probabilistic framework, where a distribution over the attribute space is imposed and where the encoder is stochastic.

Our approach also allows to take into account fixed attributes, which we saw can help shaping the free attributes.

Traditional ways to explore the latent space of VAEs is by doing linear (or spherical BID18 ) interpolations between two points.

However, there are two major caveats in this approach: the requirement of always needing two points in order to explore the latent space is cumbersome and the interpolation scheme is arbitrary and bears no probabilistic interpretation.

Concerning the first point, a common approach is to find, a posteriori, directions in the latent space that accounts for a particular change of the (fixed) attributes BID16 .

These directions are then used to move in the latent space.

Similarly, BID8 proposes a model where these directions of interest are given a priori.

Concerning the second point, BID12 proposes to compute interpolation paths minimizing some energy functional which result in interpolation curves rather than interpolation straight lines.

However, this interpolation scheme is computationally demanding since an optimization problem must be solved for each point of the interpolation path.

Another trend in controlled input manipulation is to make a posteriori analysis on a trained generative model BID6 ; BID0 ; BID16 BID3 using different means.

One possible advantage of these methods compared to ours is that different attribute manipulations can be devised after the training of the generative model.

But, these procedures are still costly and so provide any real-time applications where a user could provide on-the-fly the attributes they would like to modify.

One of these approaches BID3 consists in using the trained decoder to obtained a mapping Z → X and then performing gradient descent on an objective which accounts for the constraints or change of the attributes.

Another related approach proposed in BID6 consists in training a Generative Adversarial Network which learns to move in the vicinity of a given point in the latent space so that the decoded output enforces some constraints.

The major difference of these two approaches with our work is that these movements are done in a unique latent space, while in our case we consider separate latent spaces.

But more importantly, these approaches implicitly consider that the variation of interest lies in a neighborhood of the provided input.

In BID0 the authors introduce an additional latent space called interpretable lens used to interpret the latent space of a generative model.

This space shares similarity with our latent space Z * and they also propose a joint optimization for their model, where the encoder-decoder architecture and the interpretable lens are learned jointly.

The difference with our approach is that the authors optimize an "interpretability" loss which requires labels and still need to perform a posteriori analysis to find relevant directions in the latent space.

We presented the Variation Network, a generative model able to vary attributes of a given input.

The novelty is that these attributes can be fixed or learned and have a sound probabilistic interpretation.

Many sampling schemes have been presented together with a detailed discussion and examples.

We hope that the flexibility in the design of the attribute function and the simplicity, from an implementation point of view, in transforming existing encoder-decoder architectures (it suffices to provide the encoder and decoder networks) will be of interest in many applications.

For future work, we would like to extend our approach in two different ways: being able to deal with partially-given fixed attributes and handling discrete free attributes.

We also want to investigate the of use stochastic attribute functions φ.

Indeed, it appeared to us that using deterministic attribute functions was crucial and we would like to go deeper in the understanding of the interplay between all VarNet components.

We now apply VarNet on MNIST in order to illustrate the different sampling schemes presented in Sect.

A.In all these experiments, we choose to use a simple MLP with one hidden layer of size 400 for the encoder and decoder networks.

We present and comment results for different attribute functions and different sampling schemes.

The different attribute functions we considered are• 1Free: one-dimensional free attribute space (Eq. FORMULA3

We display in Figure 2 samples obtained with the sampling procedures Eq. FORMULA3

From Fig. 2b , we see that the fixed label attribute have clearly been taken into account, but it can be hard to grasp which high-level attribute the free attribute function has captured.

In order to visualize this, we plot in Fig. 3 a visualization of the space of variations spanned by a given template latent code z * .

From these plots, it appears that the attribute vector encodes a notion of rotation meaningful for this digit dataset and it is interesting to note how different templates produce different "writing styles".

Free attributes can thus be particularly interesting for capturing high-level features, such like rotation, that cannot be described in an absolute way or which are ill-defined.

By observing carefully Fig. 3 , we note that the variations generated by varying the free attribute applies to all digit classes, irrespective of their label.

In such a case, it is impossible to obtain different "writing conventions" for the same digit (like cursive/printscript style for the digit "2") by only modifying the attributes.

We show in FIG3 that, by considering free label-dependent attributes, we are able to smoothly go from one "writing convention" to the other one.

We can gain further insight about the notion of template and attribute using the sampling scheme of Eq. (21).

This sampling exploits the stochasticity of the encoder q * (·|x) in order to generate variations of a given input x using a fixed attribute ψ.

An example of such variations is given in FIG4 .

The underlying idea is that, even for a given attribute ψ, there are multiple ways to generate variations of x with attributes ψ.

We believe that this stochasticity is essential since, in many applications, there should not exist only one way to make variations.

The parametrization of the attribute function has a crucial effect on the high-level features that they will able to capture.

For instance, if we do not provide any label information, the information present in the template and the information contained in the attribute function can differ drastically.

FIG5 show different space of variations where no label information is provided.

The concepts captured in these cases are then related to thinness/roundness.

Our intuition is that the free attributes capture the most general attributes of the dataset.

For some applications, variation spaces such as the one displayed in FIG5 , 6b or 6d are not desirable because they may tend to move too "far away" from the original input.

As discussed in Sect.

3.1, it is possible to reduce how "spread" the spaces of variation are by modifying the β parameter multiplying the KL term in the objective Eq. (2).

An example of such a variation space is displayed in FIG5 .From all examples above, we see that our architecture is indeed capable of decoupling templates from learned attributes and that we have two ways of controlling the free attributes that are learned: by modifying the KL term in the objective Eq. (2) and by carefully devising the attribute function.

Indeed, the learned free attributes can capture different high-level features depending on the other fixed attributes they are coupled with.

FIG5 and 6c display the space of variations using the 2Free attribute function for two different input.

FIG5 display the space of variations using the 1Free attribute function.

FIG5 was generated using a model trained with a low KL penalty (β = 0.1)

VarNet proposes a novel solution to explore the latent spaces.

Usual techniques to navigate in the space of VAEs such as interpolations or the use of attribute vectors (distinct from what we called attribute vectors in this work) are mostly intrinsically-based on moving using straight lines.

This assumes that the underlying geometry is euclidean, which is not the case, and forgets about the probabilistic framework.

Also, computing attribute vectors requires data with binary labels which are not always available.

On the contrary, our approach grants a sound probabilistic interpretation of the attributes and the variations they generate.

Indeed, when the discriminator is fooled by the encoder-decoder architecture, the attributes are distributed according to ν which has a simple interpretation (it is the push-forward of the ν α distribution which is considered to be a uniform distribution in all these examples).

Also, thinking about variations as a subspace of smaller dimension than the whole latent space makes much sense for us.

Figure 7 shows a visualization in the latent space Z of the variation spaces spanned by moving with constant steps in the attribute space Ψ. Two key elements appear: constant steps in the attribute space do not induce constant steps in the Z space and variation spaces are extremely diverse (they are not translated versions of a unique variation space).

For us, this advocates for the fact that displacements in the latent spaces using straight lines have a priori no meaningful interpretation: the same change of attributes for two different inputs can lead to radically different displacements in the latent space.

More generally, our proposition of parametrizing attribute-related displacements in a latent space using flows conditioned on a simpler space is appealing from a conceptual point of view since we do not mix, in the same latent space, its probabilistic interpretation given by the prior and its ability to grant meaningful ways to vary attributes.

@highlight

The Variation Network is a generative model able to learn high-level attributes without supervision that can then be used for controlled input manipulation.