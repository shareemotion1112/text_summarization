Deep generative models have achieved impressive success in recent years.

Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), as powerful frameworks for deep generative model learning, have largely been considered as two distinct paradigms and received extensive independent studies respectively.

This paper aims to establish formal connections between GANs and VAEs through a new formulation of them.

We interpret sample generation in GANs as performing posterior inference, and show that GANs and VAEs involve minimizing KL divergences of respective posterior and inference distributions with opposite directions, extending the two learning phases of classic wake-sleep algorithm, respectively.

The unified view provides a powerful tool to analyze a diverse set of existing model variants, and enables to transfer techniques across research lines in a principled way.

For example, we apply the importance weighting method in VAE literatures for improved GAN learning, and enhance VAEs with an adversarial mechanism that leverages generated samples.

Experiments show generality and effectiveness of the transfered techniques.

Deep generative models define distributions over a set of variables organized in multiple layers.

Early forms of such models dated back to works on hierarchical Bayesian models BID30 and neural network models such as Helmholtz machines , originally studied in the context of unsupervised learning, latent space modeling, etc.

Such models are usually trained via an EM style framework, using either a variational inference (Jordan et al., 1999) or a data augmentation BID35 algorithm.

Of particular relevance to this paper is the classic wake-sleep algorithm dates by for training Helmholtz machines, as it explored an idea of minimizing a pair of KL divergences in opposite directions of the posterior and its approximation.

In recent years there has been a resurgence of interests in deep generative modeling.

The emerging approaches, including Variational Autoencoders (VAEs) BID18 , Generative Adversarial Networks (GANs) BID12 , Generative Moment Matching Networks (GMMNs) BID24 BID10 , auto-regressive neural networks BID21 BID37 , and so forth, have led to impressive results in a myriad of applications, such as image and text generation (Radford et al., 2015; Hu et al., 2017; BID37 , disentangled representation learning BID7 BID20 , and semi-supervised learning (Salimans et al., 2016; BID19 .

The deep generative model literature has largely viewed these approaches as distinct model training paradigms.

For instance, GANs aim to achieve an equilibrium between a generator and a discriminator; while VAEs are devoted to maximizing a variational lower bound of the data log-likelihood.

A rich array of theoretical analyses and model extensions have been developed independently for GANs BID0 BID1 Salimans et al., 2016; BID31 and VAEs BID4 BID8 Hu et al., 2017) , respectively.

A few works attempt to combine the two objectives in a single model for improved inference and sample generation BID26 BID22 BID25 BID34 .

Despite the significant progress specific to each method, it remains unclear how these apparently divergent approaches connect to each other in a principled way.

In this paper, we present a new formulation of GANs and VAEs that connects them under a unified view, and links them back to the classic wake-sleep algorithm.

We show that GANs and VAEs involve minimizing opposite KL divergences of respective posterior and inference distributions, and extending the sleep and wake phases, respectively, for generative model learning.

More specifically, we develop a reformulation of GANs that interprets generation of samples as performing posterior inference, leading to an objective that resembles variational inference as in VAEs.

As a counterpart, VAEs in our interpretation contain a degenerated adversarial mechanism that blocks out generated samples and only allows real examples for model training.

The proposed interpretation provides a useful tool to analyze the broad class of recent GAN-and VAEbased algorithms, enabling perhaps a more principled and unified view of the landscape of generative modeling.

For instance, one can easily extend our formulation to subsume InfoGAN BID7 that additionally infers hidden representations of examples, VAE/GAN joint models BID22 BID5 ) that offer improved generation and reduced mode missing, and adversarial domain adaptation (ADA) BID11 Purushotham et al., 2017) that is traditionally framed in the discriminative setting.

The close parallelisms between GANs and VAEs further ease transferring techniques that were originally developed for improving each individual class of models, to in turn benefit the other class.

We provide two examples in such spirit: 1) Drawn inspiration from importance weighted VAE (IWAE) BID4 , we straightforwardly derive importance weighted GAN (IWGAN) that maximizes a tighter lower bound on the marginal likelihood compared to the vanilla GAN.

2) Motivated by the GAN adversarial game we activate the originally degenerated discriminator in VAEs, resulting in a full-fledged model that adaptively leverages both real and fake examples for learning.

Empirical results show that the techniques imported from the other class are generally applicable to the base model and its variants, yielding consistently better performance.

There has been a surge of research interest in deep generative models in recent years, with remarkable progress made in understanding several class of algorithms.

The wake-sleep algorithm is one of the earliest general approaches for learning deep generative models.

The algorithm incorporates a separate inference model for posterior approximation, and aims at maximizing a variational lower bound of the data log-likelihood, or equivalently, minimizing the KL divergence of the approximate posterior and true posterior.

However, besides the wake phase that minimizes the KL divergence w.r.t the generative model, the sleep phase is introduced for tractability that minimizes instead the reversed KL divergence w.r.t the inference model.

Recent approaches such as NVIL BID28 and VAEs BID18 are developed to maximize the variational lower bound w.r.t both the generative and inference models jointly.

To reduce the variance of stochastic gradient estimates, VAEs leverage reparametrized gradients.

Many works have been done along the line of improving VAEs.

BID4 develop importance weighted VAEs to obtain a tighter lower bound.

As VAEs do not involve a sleep phase-like procedure, the model cannot leverage samples from the generative model for model training.

Hu et al. (2017) combine VAEs with an extended sleep procedure that exploits generated samples for learning.

Another emerging family of deep generative models is the Generative Adversarial Networks (GANs) BID12 , in which a discriminator is trained to distinguish between real and generated samples and the generator to confuse the discriminator.

The adversarial approach can be alternatively motivated in the perspectives of approximate Bayesian computation BID13 and density ratio estimation BID29 .

The original objective of the generator is to minimize the log probability of the discriminator correctly recognizing a generated sample as fake.

This is equivalent to minimizing a lower bound on the Jensen-Shannon divergence (JSD) of the generator and data distributions BID12 BID31 Huszar, 2016; BID23 .

Besides, the objective suffers from vanishing gradient with strong discriminator.

Thus in practice people have used another objective which maximizes the log probability of the discriminator recognizing a generated sample as real BID12 BID0 .

The second objective has the same optimal solution as with the original one.

We base our analysis of GANs on the second objective as it is widely used in practice yet few theoretic analysis has been done on it.

Numerous extensions of GANs have been developed, including combination with VAEs for improved generation BID22 BID25 BID5 , and generalization of the objectives to minimize other f-divergence criteria beyond JSD BID31 BID34 .

The adversarial principle has gone beyond the generation setting and been applied to other contexts such as domain adaptation BID11 Purushotham et al., 2017) , and Bayesian inference BID26 BID36 BID34 Rosca et al., 2017) which uses implicit variational distributions in VAEs and leverage the adversarial approach for optimization.

This paper starts from the basic models of GANs and VAEs, and develops a general formulation that reveals underlying connections of different classes of approaches including many of the above variants, yielding a unified view of the broad set of deep generative modeling.

The structures of GANs and VAEs are at the first glance quite different from each other.

VAEs are based on the variational inference approach, and include an explicit inference model that reverses the generative process defined by the generative model.

On the contrary, in traditional view GANs lack an inference model, but instead have a discriminator that judges generated samples.

In this paper, a key idea to bridge the gap is to interpret the generation of samples in GANs as performing inference, and the discrimination as a generative process that produces real/fake labels.

The resulting new formulation reveals the connections of GANs to traditional variational inference.

The reversed generation-inference interpretations between GANs and VAEs also expose their correspondence to the two learning phases in the classic wake-sleep algorithm.

For ease of presentation and to establish a systematic notation for the paper, we start with a new interpretation of Adversarial Domain Adaptation (ADA) BID11 , the application of adversarial approach in the domain adaptation context.

We then show GANs are a special case of ADA, followed with a series of analysis linking GANs, VAEs, and their variants in our formulation.

ADA aims to transfer prediction knowledge learned from a source domain to a target domain, by learning domain-invariant features BID11 .

That is, it learns a feature extractor whose output cannot be distinguished by a discriminator between the source and target domains.

We first review the conventional formulation of ADA.

FIG0 illustrates the computation flow.

Let z be a data example either in the source or target domain, and y ∈ {0, 1} the domain indicator with y = 0 indicating the target domain and y = 1 the source domain.

The data distributions conditioning on the domain are then denoted as p(z|y).

The feature extractor G θ parameterized with θ maps z to feature x = G θ (z).

To enforce domain invariance of feature x, a discriminator D φ is learned.

Specifically, D φ (x) outputs the probability that x comes from the source domain, and the discriminator is trained to maximize the binary classification accuracy of recognizing the domains: DISPLAYFORM0 The feature extractor G θ is then trained to fool the discriminator: DISPLAYFORM1 Please see the supplementary materials for more details of ADA.With the background of conventional formulation, we now frame our new interpretation of ADA.

The data distribution p(z|y) and deterministic transformation G θ together form an implicit distribution over x, denoted as p θ (x|y), which is intractable to evaluate likelihood but easy to sample from.

Let p(y) be the distribution of the domain indicator y, e.g., a uniform distribution as in Eqs.(1)-(2).

The discriminator defines a conditional distribution q φ (y|x) = D φ (x).

Let q r φ (y|x) = q φ (1 − y|x) be the reversed distribution over domains.

The objectives of ADA are therefore rewritten as (omitting the constant scale factor 2): DISPLAYFORM2 Note that z is encapsulated in the implicit distribution p θ (x|y).

The only difference of the objectives of θ from φ is the replacement of q(y|x) with q r (y|x).

This is where the adversarial mechanism comes about.

We defer deeper interpretation of the new objectives in the next subsection.

Arrows with solid lines denote generative process; arrows with dashed lines denote inference; hollow arrows denote deterministic transformation leading to implicit distributions; and blue arrows denote adversarial mechanism that involves respective conditional distribution q and its reverse q r , e.g., q(y|x) and q r (y|x) (denoted as q (r) (y|x) for short).

Note that in GANs we have interpreted x as latent variable and (z, y) as visible.

(d) InfoGAN (Eq.9), which, compared to GANs, adds conditional generation of code z with distribution qη(z|x, y).

(e) VAEs (Eq.12), which is obtained by swapping the generation and inference processes of InfoGAN, i.e., in terms of the schematic graphical model, swapping solid-line arrows (generative process) and dashed-line arrows (inference) of (d).

GANs BID12 can be seen as a special case of ADA.

Taking image generation for example, intuitively, we want to transfer the properties of real image (source domain) to generated image (target domain), making them indistinguishable to the discriminator.

FIG0 shows the conventional view of GANs.

Formally, x now denotes a real example or a generated sample, z is the respective latent code.

For the generated sample domain (y = 0), the implicit distribution p θ (x|y = 0) is defined by the prior of z and the generator G θ (z), which is also denoted as p g θ (x) in the literature.

For the real example domain (y = 1), the code space and generator are degenerated, and we are directly presented with a fixed distribution p(x|y = 1), which is just the real data distribution p data (x).

Note that p data (x) is also an implicit distribution and allows efficient empirical sampling.

In summary, the conditional distribution over x is constructed as DISPLAYFORM0 Here, free parameters θ are only associated with p g θ (x) of the generated sample domain, while p data (x) is constant.

As in ADA, discriminator D φ is simultaneously trained to infer the probability that x comes from the real data domain.

That is, DISPLAYFORM1 With the established correspondence between GANs and ADA, we can see that the objectives of GANs are precisely expressed as Eq.(3).

To make this clearer, we recover the classical form by unfolding over y and plugging in conventional notations.

For instance, the objective of the generative parameters θ in Eq. FORMULA2 is translated into DISPLAYFORM2 where p(y) is uniform and results in the constant scale factor 1/2.

As noted in sec.2, we focus on the unsaturated objective for the generator BID12 , as it is commonly used in practice yet still lacks systematic analysis.

New Interpretation Let us take a closer look into the form of Eq.(3).

It closely resembles the data reconstruction term of a variational lower bound by treating y as visible variable while x as latent (as in ADA).

That is, we are essentially reconstructing the real/fake indicator y (or its reverse 1 − y) with the "generative distribution" q φ (y|x) and conditioning on x from the "inference distribution" p θ (x|y).

FIG0 (c) shows a schematic graphical model that illustrates such generative and inference processes.

(Sec.

D in the supplementary materials gives an example of translating a given schematic graphical model into mathematical formula.)

We go a step further to reformulate the objectives and reveal more insights to the problem.

In particular, for each optimization step of p θ (x|y) at point (θ 0 , φ 0 ) in the parameter space, we have: DISPLAYFORM3 Figure 2: One optimization step of the parameter θ through Eq. FORMULA8 at point θ0.

The posterior q r (x|y) is a mixture of p θ 0 (x|y = 0) (blue) and p θ 0 (x|y = 1) (red in the left panel) with the mixing weights induced from q r φ 0 (y|x).

Minimizing the KLD drives p θ (x|y = 0) towards the respective mixture q r (x|y = 0) (green), resulting in a new state where p θ new (x|y = 0) = pg θ new (x) (red in the right panel) gets closer to p θ 0 (x|y = 1) = p data (x).

Due to the asymmetry of KLD, pg θ new (x) missed the smaller mode of the mixture q r (x|y = 0) which is a mode of p data (x).

DISPLAYFORM4 , and q r (x|y) ∝ q r φ0 (y|x)p θ0 (x).

Therefore, the updates of θ at θ 0 have DISPLAYFORM5 where KL(· ·) and JSD(· ·) are the KL and Jensen-Shannon Divergences, respectively.

Proofs are in the supplements (sec.

B).

Eq. FORMULA8 offers several insights into the GAN generator learning:• Resemblance to variational inference.

As above, we see x as latent and p θ (x|y) as the inference distribution.

The p θ0 (x) is fixed to the starting state of the current update step, and can naturally be seen as the prior over x. By definition q r (x|y) that combines the prior p θ0 (x) and the generative distribution q r φ0 (y|x) thus serves as the posterior.

Therefore, optimizing the generator G θ is equivalent to minimizing the KL divergence between the inference distribution and the posterior (a standard from of variational inference), minus a JSD between the distributions p g θ (x) and p data (x).

The interpretation further reveals the connections to VAEs, as discussed later.• Training dynamics.

By definition, p θ0 (x) = (p g θ 0 (x)+p data (x))/2 is a mixture of p g θ 0 (x) and p data (x) with uniform mixing weights, so the posterior q r (x|y) ∝ q r φ0 (y|x)p θ0 (x) is also a mixture of p g θ 0 (x) and p data (x) with mixing weights induced from the discriminator q r φ0 (y|x).

For the KL divergence to minimize, the component with y = 1 is KL (p θ (x|y = 1) q r (x|y = 1)) = KL (p data (x) q r (x|y = 1)) which is a constant.

The active component for optimization is with y = 0, i.e., KL (p θ (x|y = 0) q r (x|y = 0)) = KL (p g θ (x) q r (x|y = 0)).

Thus, minimizing the KL divergence in effect drives p g θ (x) to a mixture of p g θ 0 (x) and p data (x).

Since p data (x) is fixed, p g θ (x) gets closer to p data (x).

Figure 2 illustrates the training dynamics schematically.• The JSD term.

The negative JSD term is due to the introduction of the prior p θ0 (x).

This term pushes p g θ (x) away from p data (x), which acts oppositely from the KLD term.

However, we show that the JSD term is upper bounded by the KLD term (sec.

C).

Thus, if the KLD term is sufficiently minimized, the magnitude of the JSD also decreases.

Note that we do not mean the JSD is insignificant or negligible.

Instead conclusions drawn from Eq.(6) should take the JSD term into account.• Explanation of missing mode issue.

JSD is a symmetric divergence measure while KLD is non-symmetric.

The missing mode behavior widely observed in GANs BID27 BID5 is thus explained by the asymmetry of the KLD which tends to concentrate p θ (x|y) to large modes of q r (x|y) and ignore smaller ones.

See Figure 2 for the illustration.

Concentration to few large modes also facilitates GANs to generate sharp and realistic samples.• Optimality assumption of the discriminator.

Previous theoretical works have typically assumed (near) optimal discriminator BID12 BID0 : DISPLAYFORM6 which can be unwarranted in practice due to limited expressiveness of the discriminator BID1 .

In contrast, our result does not rely on the optimality assumptions.

Indeed, our result is a generalization of the previous theorem in BID0 , which is recovered by Published as a conference paper at ICLR 2018 plugging Eq. FORMULA9 into Eq.(6): DISPLAYFORM7 which gives simplified explanations of the training dynamics and the missing mode issue only when the discriminator meets certain optimality criteria.

Our generalized result enables understanding of broader situations.

For instance, when the discriminator distribution q φ0 (y|x) gives uniform guesses, or when p g θ = p data that is indistinguishable by the discriminator, the gradients of the KL and JSD terms in Eq.(6) cancel out, which stops the generator learning.

InfoGAN BID7 developed InfoGAN which additionally recovers (part of) the latent code z given sample x. This can straightforwardly be formulated in our framework by introducing an extra conditional q η (z|x, y) parameterized by η.

As discussed above, GANs assume a degenerated code space for real examples, thus q η (z|x, y = 1) is fixed without free parameters to learn, and η is only associated to y = 0.

The InfoGAN is then recovered by combining q η (z|x, y) with q φ (y|x) in Eq.(3) to perform full reconstruction of both z and y: DISPLAYFORM8 Again, note that z is encapsulated in the implicit distribution p θ (x|y).

The model is expressed as the schematic graphical model in FIG0 DISPLAYFORM9 (y|x)p θ0 (x) be the augmented "posterior", the result in the form of Lemma.1 still holds by adding z-related conditionals: DISPLAYFORM10 The new formulation is also generally applicable to other GAN-related variants, such as Adversarial Autoencoder BID25 , Predictability Minimization BID33 , and cycleGAN BID38 .

In the supplements we provide interpretations of the above models.

We next explore the second family of deep generative modeling.

The resemblance of GAN generator learning to variational inference (Lemma.1) suggests strong relations between VAEs (Kingma & Welling, 2013) and GANs.

We build correspondence between them, and show that VAEs involve minimizing a KLD in an opposite direction, with a degenerated adversarial discriminator.

The conventional definition of VAEs is written as: DISPLAYFORM0 wherep θ (x|z) is the generator,q η (z|x) the inference model, andp(z) the prior.

The parameters to learn are intentionally denoted with the notations of corresponding modules in GANs.

VAEs appear to differ from GANs greatly as they use only real examples and lack adversarial mechanism.

To connect to GANs, we assume a perfect discriminator q * (y|x) which always predicts y = 1 with probability 1 given real examples, and y = 0 given generated samples.

Again, for notational simplicity, let q r * (y|x) = q * (1 − y|x) be the reversed distribution.

Lemma 2.

Let p θ (z, y|x) ∝ p θ (x|z, y)p(z|y)p(y).

The VAE objective L vae θ,η in Eq.(11) is equivalent to (omitting the constant scale factor 2): DISPLAYFORM1 Here most of the components have exact correspondences (and the same definitions) in GANs and InfoGAN (see TAB1 ), except that the generation distribution p θ (x|z, y) differs slightly from its indicates the respective component is involved in the generative process within our interpretation, while "[I]" indicates inference process.

This is also expressed in the schematic graphical models in FIG0 .counterpart p θ (x|y) in Eq.(4) to additionally account for the uncertainty of generating x given z: DISPLAYFORM2 We provide the proof of Lemma 2 in the supplementary materials.

FIG0 shows the schematic graphical model of the new interpretation of VAEs, where the only difference from InfoGAN ( FIG0 ) is swapping the solid-line arrows (generative process) and dashed-line arrows (inference).As in GANs and InfoGAN, for the real example domain with y = 1, both q η (z|x, y = 1) and p θ (x|z, y = 1) are constant distributions.

Since given a fake sample x from p θ0 (x), the reversed perfect discriminator q r * (y|x) always predicts y = 1 with probability 1, the loss on fake samples is therefore degenerated to a constant, which blocks out fake samples from contributing to learning.

TAB1 summarizes the correspondence between the approaches.

Lemma.1 and Lemma.2 have revealed that both GANs and VAEs involve minimizing a KLD of respective inference and posterior distributions.

In particular, GANs involve minimizing the KL p θ (x|y) q r (x|y) while VAEs the KL q η (z|x, y)q r * (y|x) p θ (z, y|x) .

This exposes several new connections between the two model classes, each of which in turn leads to a set of existing research, or can inspire new research directions: 1) As discussed in Lemma.1, GANs now also relate to the variational inference algorithm as with VAEs, revealing a unified statistical view of the two classes.

Moreover, the new perspective naturally enables many of the extensions of VAEs and vanilla variational inference algorithm to be transferred to GANs.

We show an example in the next section.

2) The generator parameters θ are placed in the opposite directions in the two KLDs.

The asymmetry of KLD leads to distinct model behaviors.

For instance, as discussed in Lemma.1, GANs are able to generate sharp images but tend to collapse to one or few modes of the data (i.e., mode missing).

In contrast, the KLD of VAEs tends to drive generator to cover all modes of the data distribution but also small-density regions (i.e., mode covering), which usually results in blurred, implausible samples.

This naturally inspires combination of the two KLD objectives to remedy the asymmetry.

Previous works have explored such combinations, though motivated in different perspectives BID22 BID5 Pu et al., 2017) .

We discuss more details in the supplements.

3) VAEs within our formulation also include an adversarial mechanism as in GANs.

The discriminator is perfect and degenerated, disabling generated samples to help with learning.

This inspires activating the adversary to allow learning from samples.

We present a simple possible way in the next section.

4) GANs and VAEs have inverted latent-visible treatments of (z, y) and x, since we interpret sample generation in GANs as posterior inference.

Such inverted treatments strongly relates to the symmetry of the sleep and wake phases in the wake-sleep algorithm, as presented shortly.

In sec.6, we provide a more general discussion on a symmetric view of generation and inference.

Wake-sleep algorithm was proposed for learning deep generative models such as Helmholtz machines .

WS consists of wake phase and sleep phase, which optimize the generative model and inference model, respectively.

We follow the above notations, and introduce new notations h to denote general latent variables and λ to denote general parameters.

The wake sleep algorithm is thus written as: DISPLAYFORM0 Briefly, the wake phase updates the generator parameters θ by fitting p θ (x|h) to the real data and hidden code inferred by the inference model q λ (h|x).

On the other hand, the sleep phase updates the parameters λ based on the generated samples from the generator.

The relations between WS and VAEs are clear in previous discussions BID3 BID18 .

Indeed, WS was originally proposed to minimize the variational lower bound as in VAEs (Eq.11) with the sleep phase approximation .

Alternatively, VAEs can be seen as extending the wake phase.

Specifically, if we let h be z and λ be η, the wake phase objective recovers VAEs (Eq.11) in terms of generator optimization (i.e., optimizing θ).

Therefore, we can see VAEs as generalizing the wake phase by also optimizing the inference model q η , with additional prior regularization on code z.

On the other hand, GANs closely resemble the sleep phase.

To make this clearer, let h be y and λ be φ.

This results in a sleep phase objective identical to that of optimizing the discriminator q φ in Eq.(3), which is to reconstruct y given sample x. We thus can view GANs as generalizing the sleep phase by also optimizing the generative model p θ to reconstruct reversed y. InfoGAN (Eq.9) further extends the correspondence to reconstruction of latents z.

The new interpretation not only reveals the connections underlying the broad set of existing approaches, but also facilitates to exchange ideas and transfer techniques across the two classes of algorithms.

For instance, existing enhancements on VAEs can straightforwardly be applied to improve GANs, and vice versa.

This section gives two examples.

Here we only outline the main intuitions and resulting models, while providing the details in the supplement materials.

BID4 proposed importance weighted autoencoder (IWAE) that maximizes a tighter lower bound on the marginal likelihood.

Within our framework it is straightforward to develop importance weighted GANs by copying the derivations of IWAE side by side, with little adaptations.

Specifically, the variational inference interpretation in Lemma.1 suggests GANs can be viewed as maximizing a lower bound of the marginal likelihood on y (putting aside the negative JSD term):

Following BID4 , we can derive a tighter lower bound through a k-sample importance weighting estimate of the marginal likelihood.

With necessary approximations for tractability, optimizing the tighter lower bound results in the following update rule for the generator learning: DISPLAYFORM0 wi∇ θ log q r φ 0 (y|x(zi, θ)) .As in GANs, only y = 0 (i.e., generated samples) is effective for learning parameters θ.

Compared to the vanilla GAN update (Eq.(6)), the only difference here is the additional importance weight w i which is the normalization of w i = q r φ 0 (y|xi) q φ 0 (y|xi) over k samples.

Intuitively, the algorithm assigns higher weights to samples that are more realistic and fool the discriminator better, which is consistent to IWAE that emphasizes more on code states providing better reconstructions.

Hjelm et al. FORMULA0 ; Che et al. (2017b) developed a similar sample weighting scheme for generator training, while their generator of discrete data depends on explicit conditional likelihood.

In practice, the k samples correspond to sample minibatch in standard GAN update.

Thus the only computational cost added by the importance weighting method is by evaluating the weight for each sample, and is negligible.

The discriminator is trained in the same way as in standard GANs.

Table 3 : Variational lower bounds on MNIST test set, trained on 1%, 10%, and 100% training data, respectively.

In the semi-supervised VAE (SVAE) setting, remaining training data are used for unsupervised training.

By Lemma.2, VAEs include a degenerated discriminator which blocks out generated samples from contributing to model learning.

We enable adaptive incorporation of fake samples by activating the adversarial mechanism.

Specifically, we replace the perfect discriminator q * (y|x) in VAEs with a discriminator network q φ (y|x) parameterized with φ, resulting in an adapted objective of Eq. FORMULA0 : DISPLAYFORM0 .

FORMULA0 As detailed in the supplementary material, the discriminator is trained in the same way as in GANs.

The activated discriminator enables an effective data selection mechanism.

First, AAVAE uses not only real examples, but also generated samples for training.

Each sample is weighted by the inverted discriminator q r φ (y|x), so that only those samples that resemble real data and successfully fool the discriminator will be incorporated for training.

This is consistent with the importance weighting strategy in IWGAN.

Second, real examples are also weighted by q r φ (y|x).

An example receiving large weight indicates it is easily recognized by the discriminator, which means the example is hard to be simulated from the generator.

That is, AAVAE emphasizes more on harder examples.

We conduct preliminary experiments to demonstrate the generality and effectiveness of the importance weighting (IW) and adversarial activating (AA) techniques.

In this paper we do not aim at achieving state-of-the-art performance, but leave it for future work.

In particular, we show the IW and AA extensions improve the standard GANs and VAEs, as well as several of their variants, respectively.

We present the results here, and provide details of experimental setups in the supplements.

We extend both vanilla GANs and class-conditional GANs (CGAN) with the IW method.

The base GAN model is implemented with the DCGAN architecture and hyperparameter setting (Radford et al., 2015) .

Hyperparameters are not tuned for the IW extensions.

We use MNIST, SVHN, and CIFAR10 for evaluation.

For vanilla GANs and its IW extension, we measure inception scores (Salimans et al., 2016) on the generated samples.

For CGANs we evaluate the accuracy of conditional generation (Hu et al., 2017 ) with a pre-trained classifier.

Please see the supplements for more details.

TAB3 , left panel, shows the inception scores of GANs and IW-GAN, and the middle panel gives the classification accuracy of CGAN and and its IW extension.

We report the averaged results ± one standard deviation over 5 runs.

The IW strategy gives consistent improvements over the base models.

We apply the AA method on vanilla VAEs, class-conditional VAEs (CVAE), and semi-supervised VAEs (SVAE) BID19 , respectively.

We evaluate on the MNIST data.

We measure the variational lower bound on the test set, with varying number of real training examples.

For each batch of real examples, AA extended models generate equal number of fake samples for training.

There is little difference of the two processes in terms of formulation: with implicit distribution modeling, both processes only need to perform simulation through black-box neural transformations between the latent and visible spaces.

Table 3 shows the results of activating the adversarial mechanism in VAEs.

Generally, larger improvement is obtained with smaller set of real training data.

TAB3 , right panel, shows the improved accuracy of AA-SVAE over the base semi-supervised VAE.

Our new interpretations of GANs and VAEs have revealed strong connections between them, and linked the emerging new approaches to the classic wake-sleep algorithm.

The generality of the proposed formulation offers a unified statistical insight of the broad landscape of deep generative modeling, and encourages mutual exchange of techniques across research lines.

One of the key ideas in our formulation is to interpret sample generation in GANs as performing posterior inference.

This section provides a more general discussion of this point.

Traditional modeling approaches usually distinguish between latent and visible variables clearly and treat them in very different ways.

One of the key thoughts in our formulation is that it is not necessary to make clear boundary between the two types of variables (and between generation and inference), but instead, treating them as a symmetric pair helps with modeling and understanding.

For instance, we treat the generation space x in GANs as latent, which immediately reveals the connection between GANs and adversarial domain adaptation, and provides a variational inference interpretation of the generation.

A second example is the classic wake-sleep algorithm, where the wake phase reconstructs visibles conditioned on latents, while the sleep phase reconstructs latents conditioned on visibles (i.e., generated samples).

Hence, visible and latent variables are treated in a completely symmetric manner.• Empirical data distributions are usually implicit, i.e., easy to sample from but intractable for evaluating likelihood.

In contrast, priors are usually defined as explicit distributions, amiable for likelihood evaluation.• The complexity of the two distributions are different.

Visible space is usually complex while latent space tends (or is designed) to be simpler.

However, the adversarial approach in GANs and other techniques such as density ratio estimation (Mohamed & Lakshminarayanan, 2016) and approximate Bayesian computation BID2 have provided useful tools to bridge the gap in the first point.

For instance, implicit generative models such as GANs require only simulation of the generative process without explicit likelihood evaluation, hence the prior distributions over latent variables are used in the same way as the empirical data distributions, namely, generating samples from the distributions.

For explicit likelihood-based models, adversarial autoencoder (AAE) leverages the adversarial approach to allow implicit prior distributions over latent space.

Besides, a few most recent work BID26 BID36 BID34 Rosca et al., 2017) extends VAEs by using implicit variational distributions as the inference model.

Indeed, the reparameterization trick in VAEs already resembles construction of implicit variational distributions (as also seen in the derivations of IWGANs in Eq.37).

In these algorithms, adversarial approach is used to replace intractable minimization of the KL divergence between implicit variational distributions and priors.

The second difference in terms of space complexity guides us to choose appropriate tools (e.g., adversarial approach v.s. reconstruction optimization, etc) to minimize the distance between distributions to learn and their targets.

However, the tools chosen do not affect the underlying modeling mechanism.

For instance, VAEs and adversarial autoencoder both regularize the model by minimizing the distance between the variational posterior and certain prior, though VAEs choose KL divergence loss while AAE selects adversarial loss.

We can further extend the symmetric treatment of visible/latent x/z pair to data/label x/t pair, leading to a unified view of the generative and discriminative paradigms for unsupervised and semi-supervised learning.

Specifically, conditional generative models create (data, label) pairs by generating data x given label t. These pairs can be used for classifier training (Hu et al., 2017; BID32 .

In parallel, discriminative approaches such as knowledge distillation BID14 BID17 create (data, label) pairs by generating label t conditioned on data x. With the symmetric view of x and t spaces, and neural network based black-box mappings across spaces, we can see the two approaches are essentially the same.

A ADVERSARIAL DOMAIN ADAPTATION (ADA)ADA aims to transfer prediction knowledge learned from a source domain with labeled data to a target domain without labels, by learning domain-invariant features.

Let D φ (x) = q φ (y|x) be the domain discriminator.

The conventional formulation of ADA is as following: DISPLAYFORM0 Further add the supervision objective of predicting label t(z) of data z in the source domain, with a classifier f ω (t|x) parameterized with π: DISPLAYFORM1 We then obtain the conventional formulation of adversarial domain adaptation used or similar in BID11 Purushotham et al., 2017)

.B PROOF OF LEMMA 1Proof.

DISPLAYFORM2 where DISPLAYFORM3 Note that p θ (x|y = 0) = p g θ (x), and p θ (x|y = 1) = p data (x).

DISPLAYFORM4 .

Eq.(21) can be simplified as: DISPLAYFORM5 On the other hand, DISPLAYFORM6 Note that DISPLAYFORM7 Taking derivatives of Eq. FORMULA1 w.r.t θ at θ 0 we get DISPLAYFORM8 Taking derivatives of the both sides of Eq. FORMULA1 at w.r.t θ at θ 0 and plugging the last equation of Eq. FORMULA1 , we obtain the desired results.

We show that, in Lemma.1 (Eq.6), the JSD term is upper bounded by the KL term, i.e., DISPLAYFORM9 DISPLAYFORM10 Proof.

From Eq. FORMULA1 , we have DISPLAYFORM11 From Eq. FORMULA1 and Eq. FORMULA1 , we have DISPLAYFORM12 Eq. FORMULA1 and Eq. FORMULA1

Adversarial Autoencoder (AAE) BID25 can be obtained by swapping code variable z and data variable x of InfoGAN in the graphical model, as shown in FIG2 .

To see this, we directly write down the objectives represented by the graphical model in the right panel, and show they are precisely the original AAE objectives proposed in BID25 .

We present detailed derivations, which also serve as an example for how one can translate a graphical model representation to the mathematical formulations.

Readers can do similarly on the schematic graphical models of GANs, InfoGANs, VAEs, and many other relevant variants and write down the respective objectives conveniently.

We stick to the notational convention in the paper that parameter θ is associated with the distribution over x, parameter η with the distribution over z, and parameter φ with the distribution over y. Besides, we use p to denote the distributions over x, and q the distributions over z and y.

From the graphical model, the inference process (dashed-line arrows) involves implicit distribution q η (z|y) (where x is encapsulated).

As in the formulations of GANs (Eq.4 in the paper) and VAEs (Eq.13 in the paper), y = 1 indicates the real distribution we want to approximate and y = 0 indicates the approximate distribution with parameters to learn.

So we have DISPLAYFORM0 where, as z is the hidden code, q(z) is the prior distribution over z 1 , and the space of x is degenerated.

Here q η (z|y = 0) is the implicit distribution such that DISPLAYFORM1 where E η (x) is a deterministic transformation parameterized with η that maps data x to code z. Note that as x is a visible variable, the pre-fixed distribution of x is the empirical data distribution.

On the other hand, the generative process (solid-line arrows) involves p θ (x|z, y)q DISPLAYFORM2 means we will swap between q r and q).

As the space of x is degenerated given y = 1, thus p θ (x|z, y) is fixed without parameters to learn, and θ is only associated to y = 0.With the above components, we maximize the log likelihood of the generative distributions log p θ (x|z, y)q (r) φ (y|z) conditioning on the variable z inferred by q η (z|y).

Adding the prior distributions, the objectives are then written as DISPLAYFORM3 Again, the only difference between the objectives of φ and {θ, η} is swapping between q φ (y|z) and its reverse q r φ (y|z).

To make it clearer that Eq.(31) is indeed the original AAE proposed in BID25 , we transform L φ as DISPLAYFORM4 That is, the discriminator with parameters φ is trained to maximize the accuracy of distinguishing the hidden code either sampled from the true prior p(z) or inferred from observed data example x. The objective L θ,η optimizes θ and η to minimize the reconstruction loss of observed data x and at the same time to generate code z that fools the discriminator.

We thus get the conventional view of the AAE model.

Predictability Minimization (PM) BID33 is the early form of adversarial approach which aims at learning code z from data such that each unit of the code is hard to predict by the accompanying code predictor based on remaining code units.

AAE closely resembles PM by seeing the discriminator as a special form of the code predictors.

CycleGAN BID38 is the model that learns to translate examples of one domain (e.g., images of horse) to another domain (e.g., images of zebra) and vice versa based on unpaired data.

Let x and z be the variables of the two domains, then the objectives of AAE (Eq.31) is precisely the objectives that train the model to translate x into z. The reversed translation is trained with the objectives of InfoGAN (Eq.9 in the paper), the symmetric counterpart of AAE.E PROOF OF LEMME 2Proof.

For the reconstruction term: DISPLAYFORM5 where y = 0 ∼ q r * (y|x) means q r * (y|x) predicts y = 0 with probability 1.

Note that both q η (z|x, y = 1) and p θ (x|z, y = 1) are constant distributions without free parameters to learn; q η (z|x, y = 0) = q η (z|x), and p θ (x|z, y = 0) =p θ (x|z).For the KL prior regularization term: DISPLAYFORM6 Combining Eq.(33) and Eq.(34) we recover the conventional VAE objective in Eq. FORMULA9 in the paper.

Previous works have explored combination of VAEs and GANs.

This can be naturally motivated by the asymmetric behaviors of the KL divergences that the two algorithms aim to optimize respectively.

Specifically, the VAE/GAN joint models BID22 Pu et al., 2017) that improve the sharpness of VAE generated images can be alternatively motivated by remedying the mode covering behavior of the KLD in VAEs.

That is, the KLD tends to drive the generative model to cover all modes of the data distribution as well as regions with small values of p data , resulting in blurred, implausible samples.

Incorporation of GAN objectives alleviates the issue as the inverted KL enforces the generator to focus on meaningful data modes.

From the other perspective, augmenting GANs with VAE objectives helps addressing the mode missing problem, which justifies the intuition of (Che et al., 2017a).

From Eq.(6) in the paper, we can view GANs as maximizing a lower bound of the "marginal log-likelihood" on y:log q(y) = log p θ (x|y) q r (y|x)p θ 0 (x) p θ (x|y) dx ≥ p θ (x|y) log q r (y|x)p θ 0 (x) p θ (x|y) dx = −KL(p θ (x|y) q r (x|y)) + const.

We can apply the same importance weighting method as in IWAE BID4 to derive a tighter bound.

DISPLAYFORM0 where we have denoted w i = q r (y|xi)p θ 0 (xi) p θ (xi|y), which is the unnormalized importance weight.

We recover the lower bound of Eq.(35) when setting k = 1.To maximize the importance weighted lower bound L k (y), we take the derivative w.r.t θ and apply the reparameterization trick on samples x: DISPLAYFORM1 w(y, x(zi, θ)) DISPLAYFORM2 wi∇ θ log w(y, x(zi, θ)) ,

We extend both vanilla GANs and class-conditional GANs (CGAN) with the importance weighting method.

The base GAN model is implemented with the DCGAN architecture and hyperparameter setting (Radford et al., 2015) .

We do not tune the hyperparameters for the importance weighted extensions.

We use MNIST, SVHN, and CIFAR10 for evaluation.

For vanilla GANs and its IW extension, we measure inception scores (Salimans et al., 2016) on the generated samples.

We train deep residual networks provided in the tensorflow library as evaluation networks, which achieve inception scores of 9.09, 6.55, and 8.77 on the test sets of MNIST, SVHN, and CIFAR10, respectively.

For conditional GANs we evaluate the accuracy of conditional generation (Hu et al., 2017) .

That is, we generate samples given class labels, and then use the pre-trained classifier to predict class labels of the generated samples.

The accuracy is calculated as the percentage of the predictions that match the conditional labels.

The evaluation networks achieve accuracy of 0.990 and 0.902 on the test sets of MNIST and SVHN, respectively.

We apply the adversary activating method on vanilla VAEs, class-conditional VAEs (CVAE), and semi-supervised VAEs (SVAE) BID19 .

We evaluate on the MNIST data.

The generator networks have the same architecture as the generators in GANs in the above experiments, with sigmoid activation functions on the last layer to compute the means of Bernoulli distributions over pixels.

The inference networks, discriminators, and the classifier in SVAE share the same architecture as the discriminators in the GAN experiments.

We evaluate the lower bound value on the test set, with varying number of real training examples.

For each minibatch of real examples we generate equal number of fake samples for training.

In the experiments we found it is generally helpful to smooth the discriminator distributions by setting the temperature of the output sigmoid function larger than 1.

This basically encourages the use of fake data for learning.

We select the best temperature from {1, 1.5, 3, 5} through cross-validation.

We do not tune other hyperparameters for the adversary activated extensions.

TAB5 : Classification accuracy of semi-supervised VAEs and the adversary activated extension on the MNIST test set, with varying size of real labeled training examples.

@highlight

A unified statistical view of the broad class of deep generative models 

@highlight

The paper develops a framework interpreting GAN algorithms as performing a form of variational inference on a generative model reconstructing an indicator variable of whether a sample is from the true of generative data distributions.