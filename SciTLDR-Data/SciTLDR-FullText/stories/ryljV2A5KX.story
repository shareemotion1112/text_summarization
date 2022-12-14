We present a novel architecture of GAN for a disentangled representation learning.

The new model architecture is inspired by Information Bottleneck (IB) theory thereby named IB-GAN.

IB-GAN objective is similar to that of InfoGAN but has a crucial difference; a capacity regularization for mutual information is adopted, thanks to which the generator of IB-GAN can harness a latent representation in disentangled and interpretable manner.

To facilitate the optimization of IB-GAN in practice, a new variational upper-bound is derived.

With experiments on CelebA, 3DChairs, and dSprites datasets, we demonstrate that the visual quality of samples generated by IB-GAN is often better than those by β-VAEs.

Moreover, IB-GAN achieves much higher disentanglement metrics score than β-VAEs or InfoGAN on the dSprites dataset.

Learning good representations for data is one of the essential topics in machine learning community.

Although any strict definition for it may not exist, the consensus about the useful properties of good representations has been discussed throughout many studies BID9 Lake et al., 2017; BID10 .

A disentanglement, one of those useful properties of representation, is often described as a statistical independence or factorization; each independent factor is expected to be semantically well aligned with the human intuition on the data generative factor (e.g. a chair-type from azimuth on Chairs dataset BID6 , or age from azimuth on CelebA dataset (Liu et al., 2015) ).

The learned representation distilling each important factors of data into a single independent direction is hard to be done but highly valuable for many other downstream tasks (Ridgeway, 2016; Higgins et al., 2017b; .Many models have been proposed for disentangled representation learning (Hinton et al., 2011; Kingma et al., 2014; Reed et al., 2014; Narayanaswamy et al., 2017; BID13 .

Despite their impressive results, they either require knowledge of ground-truth generative factors or weak-supervision (e.g. domain knowledge or partial labels).

In contrast, among many unsupervised approaches BID14 Kingma & Welling, 2013; Rezende et al., 2014; Springenberg, 2015; BID15 ), yet the two most successful approaches for the independent factor learning are β-VAE BID20 and InfoGAN BID12 .

BID20 demonstrate that encouraging the KL-divergence term of Variational autoencoder (VAE) objective (Kingma & Welling, 2013; Rezende et al., 2014) by multiplying a constant β > 1 induces a high-quality disentanglement of latent factors.

As follow-up research, BID10 provide a theoretical justification of the disentangling effect of β-VAE in the context of Information Bottleneck theory BID25 BID24 BID12 propose another fully unsupervised approach based on Generative Adversarial Network (GAN) BID18 .

He achieves the goal by enforcing the generator to learn disentangled representations through increasing the mutual information (MI) between the generated samples and the latent representations.

Although InfoGAN can learn to disentangle representations for relatively simple datasets (e.g. MNIST, 3D Chairs), it struggles to do so on more complicated datasets such as CelebA. Moreover, the disentangling performance of the learned representations from InfoGAN is known as not good as the performance of the β-VAE and its variant models BID20 Kim & Mnih, 2018; BID11 .Stimulated by the success of β-VAE models BID10 BID20 Kim & Mnih, 2018; BID11 BID17 with the Information Bottleneck theory BID5 BID0 ) in disentangled representations learning task, we hypothesize that the weakness of InfoGAN in the representation learning may originate from that it can only maximize the mutual information but lacks any constraining mechanisms.

In other words, InfoGAN misses the term upper-bounding the mutual information from the perspective of IB theory.

We present a novel unsupervised model named IB-GAN (Information Bottleneck GAN) for learning disentangled representations based on IB theory.

We propose a new architecture of GANs from IB theory so that the training objective involves an information capacity constraint that InfoGAN lacks but β-VAE has.

We also derive a new variational approximation algorithm to optimize IB-GAN objective in practice.

Thanks to the information regularizer, the generator can use the latent representations in a manner that is both more interpretable and disentangled than InfoGANThe contributions of this work are summarized as follows:1.

IB-GAN is a new GAN-based model for fully unsupervised learning of disentangled representations.

To the best of our knowledge, there is no other unsupervised GAN-based model for this sake except the InfoGAN's variants BID20 Kim & Mnih, 2018) .2.

Our work is the first attempt to utilize the IB theory into the GAN-based deep generative model.

IB-GAN can be seen as an extension to the InfoGAN, supplementing an information constraining regularizer that InfoGAN misses.3.

IB-GAN surpasses state-of-the-art disentanglement scores of BID20 BID16 on dSprites dataset (Matthey et al., 2017) .

The quality of generated samples by IB-GAN on 3D Chairs BID6 and CelebA (Liu et al., 2015) is also much realistic compared to that of the existing β-VAE variants of the same task.

We remind some backgrounds: IB principle in section 2.1 and the connection between β-VAE and IB theory BID10 in section 2.2.

Lastly, InfoGAN BID12 ) is briefly reviewed in section 2.3.

Let the input variable X and the target variable Y distributed according to some joint data distribution p(x, y).

The goal of the IB BID25 BID5 is to obtain a compressive representation Z from the input variable X, while maintaining the predictive information about the target variable Y as much as possible.

The objective for the IB is DISPLAYFORM0 where I(·, ·) denotes MI and β ≥ 0 is a Lagrange multiplier.

The goal is to obtain the optimal representation encoder q φ (z|x) that balances the trade-off between the maximization and minimization of both MI terms.

Hence, the IB objective in Eq.(1) provides a natural means for good representations by enforcing the representation Z to ignore irrelevant information from the input and simultaneously to be predictive about the target, which can act as a minimal sufficient statistic of X for predicting Y BID25 BID5 .A growing body of studies BID5 BID1 BID13 supports that the learned representations adapting the IB objective tend to be highly efficient and distilled in terms of its code length Shannon et al., 1951) .

As a consequence, the learned representation is more generalizable and robust to adversarial attack BID5 , disentangled BID10 and invariant to nuance factors BID0 .

Moreover, the IB framework prevents weight over-fitting BID1 BID27 , and can be used to visualize high dimensional embedding in a low dimensional latent space (Rabinowitz et al., 2018) .

β-VAE BID20 ) is one of the state-of-the-art unsupervised disentangled representation learning models.

The key idea of β-VAE is to multiply a constant β ≥ 1 to the KL-divergence term of the original VAE's objective (Kingma & Welling, 2013; Rezende et al., 2014) : DISPLAYFORM0 where the encoder q φ (z|x) is generally known as the variational approximation to the intractable p(z|x), p(z) is a prior for the latent representation and p θ (x|z) is the decoder in the VAE context.

Recently, a notable connection between β-VAE and the IB theory has been discovered in BID5 .

Eq.(2) can be derived from the variational approximation to the IB objective Eq.(1).

To clarify this connection, see the variational upper and lower bound of the MI: DISPLAYFORM1 The MI in Eq.(3) subscribed with q: DISPLAYFORM2 , is called as the representational MI 1 .

Given that computing marginal q φ (z) is intractable, we can use any prior p(z) to substitute for q φ (z), forming the variational upper-bound in Eq. FORMULA2 2 .

Likewise, we can use any decoder model p θ (x|z) to approximate q φ (x|z) = q φ (z|x)p(x)/q φ (z) of the MI, forming the variational lower-bound in Eq.(3).

If the target variable Y in Eq. FORMULA0 is replaced with X, the task is to reconstruct (auto-encode) data from the representation Z. The variational lower-bound of Eq. FORMULA0 obtained by leveraging the upper and lower bound of MI in Eq.(3) corresponds to Eq. FORMULA1 3 .The disentanglement-promoting behavior of the β-VAE based on IB theory is discussed in BID10 .

Constraining the MI (or minimizing KL-divergence in practice) forces the encoder to learn representation containing only strongly relevant information to the data reconstruction, while ignoring other unnecessary (or less-necessary) features.

The encoder becomes reluctant to use more channels (or dimensions) of the latent vector to lower the MI constraining cost.

Hence, the most distinctive and principle features of data are grouped and aligned along with each independent dimension of the representation space.

Generative Adversarial Networks (GAN) BID18 ) establish a min-max adversarial game between two neural networks, a generator G and a discriminator D. The discriminator D aims to distinguish well between real sample x ∼ p(x) and synthetic sample created by the G(z) with a random noise z ∼ p(z), while the generator G is trained to produce a realistic sample that is indistinguishable from the true sample.

The adversarial game is formulated as follow: DISPLAYFORM0 Under an optimal discriminator D * , Eq.(4) theoretically involves with the Jensen-Shannon divergence between the synthetic and the true sample distribution: JS(G(z)||p(x)).

However, Eq.(4) does not have any specific guidance on how G utilizes a mapping from z to x. That is, the variation of z in any independent dimension often yields entangled effects on a generated sample x.

On the other hand, InfoGAN BID12 ) is capable of learning disentangled representations.

InfoGAN introduces an additional latent code c and encourages it to describe the semantic features of the data.

To do so, the training objective of InfoGAN accommodates a mutual information maximization term between the latent code c and the generated sample x = G(z, c): DISPLAYFORM1 where I(·, ·) denote MI and λ is a weight coefficient.

To optimize Eq.(5), the variational lower bound of MI is also exploited similar to that of the IM algorithm BID8 .

1 We distinguish it from the generative in the next section.

2 The variational inference relies on the positivity of the KL divergence: BID25 BID28 .

DISPLAYFORM2 3 A constant data entropy term DISPLAYFORM3 is ignored for brevity.

We introduce IB-GAN for disentangled representation learning approach in section 3.1, and propose a practical variational approximation for IB-GAN model in section 3.2.

Finally, we discuss some distinctive characteristics of the IB-GAN in-depth in section 3.3.

Although InfoGAN BID12 ) is a fully unsupervised GAN-based approach for learning disentangled representations, its disentanglement performance is, constantly reported, lower than β-VAE and its variants BID20 Kim & Mnih, 2018; BID11 .

we hypothesis the weakness of InfoGAN in independent factor learning may originate from the absence of information constraint or any compression mechanism for the representation.

Hence, our motivation is straightforward; we adopt the IB principle to the objective of InfoGAN, presenting Information Bottleneck GAN (IB-GAN).

IB-GAN not only maximizes the MI term as the original InfoGAN does, but also constrains the maximization of MI simultaneously as DISPLAYFORM0 DISPLAYFORM1 where I L (·, ·) and I U (·, ·) denote the lower and upper bound of generative MI 4 respectively.

The parameters λ and β are the weight coefficients of the GAN loss and the upper-bound of MI, respectively.

More details on these parameters are discussed in section 3.3.

One important change 5 in Eq.(6) compared to the InfoGAN objective is regularizing the upper bound of MI with β, analogously to that of β-VAE and IB theory.

For the optimization of IB-GAN, we here define the tractable variational lower and upper bound of the MI in Eq.(6) using the similar derivation in BID12 BID2 .

For notational consistency, we use p θ (x|z) to denote the generator G(z).

Then, the variational lowerbound I L (z, G(z)) of the generative MI in Eq.(6) becomes DISPLAYFORM0 Since the generator marginal p θ (x) is difficult to calculate, a reconstructor model q φ (z|x) is introduced to approximate the quantity p θ (z|x) = p θ (x|z)p(z)/p θ (x) in Eq. (7) .

The lower-bound holds thanks to positivity of KL-divergence.

Intuitively, by improving the reconstruction of an input code z from a generated sample x = G(z), we can maximize the lower-bound of MI between the generator and the code z.

In contrast to the lower-bound, obtaining a practical variational upper-bound of the generative MI is not trivial.

If we follow the same approach in BID5 , the upper-bound I U (z, G(z)) of the generative MI becomes DISPLAYFORM1 where d(x) is a variational approximation to the generator marginal p θ (x) = z p(x|z)p(z).

However, one critical problem of this approach is, in practice, it is difficult to choose or correctly identify the proper approximation model for d(x).

Algorithm 1 IB-GAN training algorithm Input: batch size B, hyperparameters λ, β, and learning rates DISPLAYFORM2 DISPLAYFORM3 In theory, we can choose any model for d(x) (e.g. Gaussian), yet any improper choice of d(x) may severely downgrade the quality of synthesized samples from the generator p θ (x|z) since the upperbound I U (z, G(z)) in Eq. FORMULA11 is eventually identical to the KL(p θ (x|z)||d(x)).

Moreover, although we express G(z) as p θ (x|z) for notional convenience, the probabilistic modeling of generator G will lose the merit of GAN: the likelihood-free (or implicit) modeling assumption.

For this reason, we develop another formulation of the variational upper-bound on the MI term, based on the studies of deep-learning architecture and IB theory BID24 BID0 .

We define an additional stochastic model e ψ (r|z) that takes a noise input vector z and produces an intermediate stochastic representation r. In other words, we let x = G(r(z)) instead of x = G(z), then we can express the generator as p θ (x|z) = r p θ (x|r)e ψ (r|z).

Consequently, a practical upper-bound I U (z, R(z)) of the generative MI can be obtained as: DISPLAYFORM4 DISPLAYFORM5 The first inequality in Eq.(9) holds thanks to the Markov property BID24 : if any generative process follows Z → R → X, then I(Z, X) ≤ I(Z, R).

The inequality in Eq. FORMULA0 holds from the positivity of KL divergence.

Thus, any prior m(r) can be utilized for substituting the marginal e ψ (r) without affecting the generated samples directly; therefore, this can bypass the difficulty of choosing the prior d(x) in Eq.(8).Finally, from the variational lower-bound of the MI in Eq. (7) and the newly introduced upper-bound in Eq.(10), the lower-bound of IB-GAN objective in Eq.(6) can be written as: max DISPLAYFORM6 DISPLAYFORM7 In other words, the intermediate representation r and the KL(e ψ (r|z)||m(r)) with β in Eq. FORMULA0 are leveraged to constrain the amount of shared information between the generator G(z) and input z. Eq. FORMULA0 is optimized by alternatively maximizing the generator G = p θ (x|r), the representation encoder e ψ (r|z), the variational reconstructor q φ (z|x) and the discriminator D. The IB-GAN architecture is presented in FIG0 (a), and overall training procedure is described in Algorithm 1.

Connection to rate-distortion theory.

Information Bottleneck theory is a generalization of the rate-distortion theory BID25 authors, 2019) , in which the rate R is the code length per data sample to be transmitted through a noisy channel, and the distortion D represents the approximation error of reconstructing the input from the source code authors, 2019; Shannon et al., 1951) .

The goal of RD-theory is minimizing D without exceeding a certain level of rate R, can be formulated as min R,D D + βR, where β ∈ [0, ∞] decides a theoretical achievable optimal frontier in the auto-encoding limit .Likewise, z and r in IB-GAN can be treated as an input and the encoding of the input, respectively.

The distortion D is minimized by optimizing the variational reconstructor q φ (z|x(r)) to predict the input z from its encoding r, that is equivalent to maximizing I L (z, G(z)).

The minimization of rate R is related minimizing the KL(e ψ (r|z)||m(r)) which measures the in-efficiency (or excess rate) of the representation encoder e ψ (r|z) in terms of how much it deviates from the prior m(r).Disentanglement-promoting behavior.

The disentanglement-promoting behavior of β-VAE is encouraged by the variational upper-bound of MI term (i.e. KL(q(z|x)||p(z))).

Since p(z) is often a factored Gaussian distribution, the KL-divergence term is decomposed into the form containing a total correlation term (Hoffman & Johnson, 2016; Kim & Mnih, 2018; BID11 BID17 BID10 , which essentially enforces the encoder to output statistically factored representations (Kim & Mnih, 2018; BID11 .

Nevertheless, in IB-GAN, a noise input z is fed into the representation encoder e ψ (r|z) instead of the image x. Therefore, the disentangling mechanism of IB-GAN must be different from those of β-VAEs.

From the formulation of the Eq.(11), we could obtain another important insight: the GAN loss in IB-GAN can be seen as the secondary capacity regularizer over the noisy channel since the discriminator of GAN is the JS-divergence (or the reverse KL-divergence) between the generator and the empirical data distribution p(x) in its optimal BID18 BID22 .

Hence, λ controls the information compression level of z in the its encoding x = G(r(z)) 6 .

In other words, the GAN loss in IB-GAN is a second rate constraint in addition to the first rate constraint KL(e ψ (r|z)||m(r)) in the context of the rate-distortion theorem.

Therefore, we describe the disentanglement-promoting behavior of IB-GAN regarding the ratedistortion theorem.

Here, the goal is to deliver the input source z through the noisy channel using the coding r and x. We want to use compact encoding schemes for r and x. (1) The efficient encoding scheme for r is defined by minimizing KL(e ψ (r|z)||m(r)) with the factored Gaussian prior m(r), which promotes statistical independence of the r. (2) The efficient encoding scheme for x is defined by minimizing the divergence between G(z) and the data distribution p(x) via the discriminator; this promote the encoding x to be the realistic image.

(3) Maximizing I L (z, G(z)) in IB-GAN indirectly maximize I(r, G(r)) since I(z, G(z)) ≤ I(r, G(r)).

In other words, maximizing the lower-bound of MI will increases the statistical dependency between the coding r and G(r), while these encoding need to be efficient in terms of their rate.

Therefore, a single independent changes in r must be coordinated with the variations of a independent image factor.

How to choose hyperparameters.

Although setting any positive values for λ and β is possible , we set β ∈ [0, 1] and fix λ = 1.

We observe that, in the most of the cases, IU (r, R(z)) collapses to 0 when β > 0.75 in the experiments with dSprites.

Although λ is another interesting hyperparameter that can control the rate of x (i.e. the divergence of the G(z) from p(x)), we aims to support the usefulness of IB-GAN in the disentangled representation learning tasks, and thus we focus on the effect of β ∈ [0, 1.2] on the I U (r, R(z)) while fixing λ = 1.

More discussion on the hyperparameter setting will be discussed in Appendix. (Kim & Mnih, 2018; BID16 .

Our model's scores are obtained from 32 random seeds, with a peak score of (0.91, 0.78).

The baseline scores except InfoGAN are referred to BID17 .

We use DCGAN (Radford et al., 2016) with batch normalization (Ioffe & Szegedy, 2015) as our base model for the generator and the discriminator.

We let the reconstructor share the same frontend feature with the discriminator for efficient use of parameters as in the InfoGAN BID12 .

Also, the MLP-based representation encoder is used before the generator.

We train the model using RMSProp BID23 optimizer with momentum of 0.9.

The minibatch size is 64 in all experiments.

Lastly, we constrain true and synthetic images to be normalized as [−1, 1].

Almost identical architectural configurations for the generator, discriminator, reconstructor, and representation encoder are used in all experiments except that the numbers of parameters are changed depending on the datasets.

We defer more details on the models and experimental settings to Appendix.

Although it is not easy to evaluate the disentanglement of representations, some quantitative metrics BID20 Kim & Mnih, 2018; BID11 BID16 have been proposed based on the synthetic datasets providing ground-truth generative factors such as dSprites (Matthey et al., 2017) or teapots BID16 .

We verified our approach with the two different metrics (Kim & Mnih, 2018; BID16 on the dSprites dataset since this setting is tested with many other state-of-the-art baselines in BID17 including standard VAE (Kingma & Welling, 2013; Rezende et al., 2014) , β-VAE BID20 , TC-VAE BID11 and HFVAE BID17 .In experiments, we adopt the instance noises technique BID21 since the dSprites images are too simple for the generator of GAN to learn.

That is, the intensity distribution of synthetic images is unnaturally narrow (i.e. [0, 1]), making the overlapping probability with generated images very low, where the generator may barely learn from the discriminator.

Hence, by adding instance noises ∼ N (0, σ instance * I) to both true and generated inputs, we can significantly improve the training stability of GAN models.

This may be the reason for the inconsistency between the previous experiments (Kim & Mnih, 2018; BID11 on InfoGAN and our results.

For the technical detail, we anneal σ instance linearly from 1 to 0 during training for InfoGAN and IB-GAN.

FIG0 shows the variations of KL(e(r i |z)||m(r i )) for 10-dimensional r (i.e. i = 1, . . . , 10) over training iterations on dSprites dataset (Matthey et al., 2017) .

The sum of these values is the upper-bound of MI.

We observe that all factors of variations are capped by different values.

Similar behavior is exhibited in β-VAE BID10 .

During training, the encoder e ψ (r|z) is slowly adapted to capture the independent factors of dSprites dataset as the lower-bound of MI increases.

We present the visual inspection of the latent traversal BID20 with the learned IB-GAN model in FIG2 .

The IB-GAN successfully learns 5 out of 5 ground truth factors from dSprites dataset, including positions of Y and X, scales, rotations, and shapes, which aligns with the caps on KL scores in FIG0 .

More results and discussion about the convergence and the effects of β will is in Appendix B. TAB0 shows the quantitative results in terms of the two disentanglement metric scores (Kim & Mnih, 2018; BID16 .

IB-GAN outperforms other baselines (Kingma & Welling, 2013; BID20 BID11 BID17 .

Interestingly, in our experiments, InfoGAN attains comparable scores to those of other VAE-based models.

On the Eastwood's randomforest metric, InfoGAN slightly outperforms other baselines as well, which is consistent with the result of BID16 .

Following BID12 BID20 BID11 Kim & Mnih, 2018) , we evaluate the qualitative results of IB-GAN by inspecting latent traversals.

As shown in FIG4 , the IB-GAN discovers various human attributes such as azimuth, hair color and smiling face expression.

In addition, generated images of the IB-GAN are sharp and realistic than the result of β-VAE and its variants BID20 Kim & Mnih, 2018; BID11 .

We also show our qualitative results on 3D Chairs dataset in FIG4 .

IB-GAN successfully disentangles scales, leg types and azimuth of chairs.

These attributes are hardly captured in the original InfoGAN BID12 BID20 Kim & Mnih, 2018; BID11 , demonstrating the effectiveness of our approach.

The proposed IB-GAN is a novel unsupervised GAN-based model for learning disentangled representation.

We made a crucial modification on the InfoGAN's objective inspired by the IB theory and β-VAE; specifically, we developed an information capacity constraining term between the generator and the latent representation.

We also derived a new variational approximation technique for optimizing IB-GAN.

Our experimental results showed that IB-GAN achieved the state-of-the-art performance on disentangled representation learning.

The qualitatively generated samples of IB-GAN often had better quality than those of β-VAE on CelebA and 3D Chairs.

IB-GAN attained higher quantitative scores than β-VAE and InfoGAN with disentanglement metrics on dSprites dataset.

There are many possible directions for future work.

First, our model can be naturally extended to adapt a discrete latent representation, as discussed in section 3.3.

Second, many extensions of β-VAE have been actively proposed such as BID10 Kim & Mnih, 2018; BID11 BID17 , most of which are complementary for the IB-GAN objective.

Further exploration toward this direction could be another interesting next topic.

Reconstruction of input noise z. The resulting architecture of IB-GAN is partly analogous to that of β-VAE since both are derived from the IB theory.

However, β-VAE often generates blurry output images due to the large β > 1 ( Kim & Mnih, 2018; BID11 BID17 since setting β > 1 typically increases the distortion .

Recently, demonstrates the possibility of achieving small distortion with the minimum rate by adopting a complex auto-regressive decoder in β-VAE and by setting β < 1.

However, their experiment is performed on relatively small dataset (e.g. MNIST, Omniglot).In contrast, IB-GAN may not suffer from this shortcoming since the generator in IB-GAN learns to generate image by minimizing the rate.

Moreover, it does not rely on any probabilistic modeling assumption of the decoder unlike VAEs and can inherit all merits of InfoGANs (e.g. producing images of good quality by an implicit decoder, and an adaptation of categorical distribution).

One downside of our model would be the introduction of additional capacity control parameter λ.

Although, we fixed λ = 1 in all of our experiment, which could also affect the convergence or the generalization ability of the generator.

Further investigation on this subject could be an interesting future work.

Behaviors of IB-GAN according to β.

If β is too large such that the KL-divergence term is almost zero, then there would be no difference between the samples from the representation encoder e ψ (r|z) and the distortion prior m(r).

Then, both representation r and generated data x contain no information about z at all, resulting in that the signal from the reconstructor is meaningless to the generator.

In this case, the IB-GAN reduces to a vanilla GAN with an input r ∼ p(r).Maximization of variational lower-bound.

Maximizing the variational lower-bound of generative MI has been employed in IM algorithm BID2 and InfoGAN BID12 .

Recently, offer the lower-bound of MI, named GILBO, as a data independent measure for the complexity of the learned representations for trained generative models.

They discover the optimal lower-bound of the generative MI correlates well with the common image quality metrics of generative models (e.g. INCEPTION Salimans et al. (2016) or FID Heusel et al. (2017) ).

In this work, we discover a new way of upper-bounding the generative MI based on the causal relationship of deep learning architecture, and show the effectiveness of the upper-bound by measures the disentanglement of learned representation.

Implementation of IB-GAN.

Since the representation encoder e ψ (r|z) is stochastic, reparametrization trick (Kingma & Welling, 2013 ) is needed to backpropagate gradient signals for training the encoder model.

The representation r can be embedded along with an extra discrete code c ∼ p(c) before getting into the generator (i.e. G(r, c)), and accordingly the reconstructor network becomes q(r, c|x) to predict the discrete code c as well.

In this way, it is straightforward to introduce a discrete representation into IB-GAN, which is not an easy task in β-VAE based models.

Theoretically, we can choose the any number for the dimension of r and z. However, The disentangled representation of IB-GAN is learned via the representation encoder e ψ (r|z).

To obtain the representation r back from the real data x, we first sample z using the learned reconstructor q φ (z|x), and input it to the representation encoder e ψ (r|z).

Therefore, we typically choose a smaller r dimension than that of z. For more details on the architecture of IB-GAN, please refer Appendix.

E.Related Work.

Many extensions of β-VAE BID20 have been proposed.

BID10 modify β-VAE's objective such that the KL term is minimized to a specific target constant C instead of scaling the term using β.

Kim & Mnih (2018) and BID11 demonstrate using the ELBO surgery (Hoffman & Johnson, 2016; Makhzani & Frey, 2017 ) that minimizing the KL-divergence enforces factorization of the marginal encoder, and thus promotes the independence of learned representation.

However, a high value of β can decrease the MI term too much, and thus often leads to worse reconstruction fidelity compared to the standard VAE.

Hence, they introduce a total correlation BID26 based regularization to overcome the reconstruction and disentanglement trade-off.

These approaches could be complementary to IB-GAN, since the objective of IB-GAN also involves with the KL term.

This exploration could be an interesting future work.

One of the most important hyperparameter in the IB-GAN objective of Eq. FORMULA0 is β that controls the ratio of the lower-bound I L (z, R(z)) and the upper-bound I U (z, R(z)).

Hence, the optimal balance point between the lower and upper bound term is affected by the β .

Each panel in FIG6 shows the variational lower and upper-bound of MI along with independent KL(e(r i |z)||m(r i )) for each r i (i = 1, · · · , 10) over the 150K training iterations.

As shown in FIG6 , if β = 0, the upper-bound of MI in Eq. FORMULA0 is ignored and the constraining effect on the representation r disappears.

Hence, the lower-bound of MI can quickly increases up to its natural upper-bound 7 similar to the MI lower-bound in InfoGAN.

With β = 1 of FIG6 , the upper-bound of MI drops down to almost zero and so does the lower-bound.

Hence, the representation r is independent of z (i.e. dose not contain any information about z) and IB-GAN reduces to vanilla GAN.When β is set properly as in FIG6 , both lower and upper-bound of MI increase smoothly and the representation encoder e ψ (r|z) is slowly adapted to capture the distinctive factors of the dataset, where independent KL-divergence KL(e(r i |z)||m(r i )) increases one by one by capturing each disentangled attribute.

Note that the sum of individual KL scores is the upper-bound of MI I U (z, R(z)) = i KL(e(r i |z)||m(r i )).

We observe that all factors of variations are capped by different values; this behavior is reported as a key element of the disentangled representation learning in β-VAE BID10 .

FORMULA0 ) vs β.

FIG7 and 5b illustrates the expected converged value of upper and lower MI bounds over the different β.

Overall, the upper MI bound tends to decrease exponentially as β increases, consequently the lower MI bound decreases as well.

DISPLAYFORM0 Specifically, β = 0, the upper-bound MI term disappears in the IB-GAN Eq.(1).

Hence, the representation encoding r can diverge from the prior distribution m(r) without any restrictions, resulting in a high value of upper MI bound.

Interestingly, the gap between the upper and lower bound is also reduced as the β parameter increases as we can see in Figure.

5b.

Lastly, FIG7 shows the effect of β on the disentanglement scores.

The optimal disentanglement score was achieved when the β is around in a range of [0.1, 0.35], and optimal disentanglement score 0.91 is obtained when β = 0.212 supporting the fact that IB-GAN could control the disentanglement of the learned representation with the upper-bound of generative MI and the varying β.

Following BID12 BID20 BID11 Kim & Mnih, 2018) , we evaluate the qualitative results of IB-GAN by inspecting latent traversals.

As shown in Figure7, the IB-GAN discovers various human attributes such as (a) azimuth, (b) background color, (c) hair color, (d) skin color, (e) smile, and (f) gender.

All of the features in Figure7 and Figure3(a) are captured by one best model with the parameter of β = 0.2838, λ = 1.

These attributes are hardly captured in the original InfoGAN BID12 BID20 Kim & Mnih, 2018; BID11 , demonstrating the usefulness of the upper-bound of generative MI in IB-GAN.

In addition, generated images of the IB-GAN are often sharp and realistic than the results of β-VAE and its variants BID20 Kim & Mnih, 2018; BID11 .

BID20 737,280 binary 64 × 64 images of 2D shapes with 5 ground truth factors.

Ground truth factors consist of 3 shapes, 6 scales, 40 orientations, and 32 positions of X and Y .

3D Chairs BID6 86,366 gray-scale 64 × 64 images of 1,393 chair CAD models with 31 azimuth angles and 2 elevation angles.

CelebA (Liu et al., 2015) 202,599 RGB 64 × 64 × 3 images of celebrity faces consisting of 10,177 identities, 5 landmark locations, and 40 binary attributes.

We use the cropped version of the dataset.

TAB2 describes the details of hyperparameter settings used in our experiments.

LR G/E/Q 5e-5, D 1e-6 iterations 1.5e5

RMSProp(momentum=0.9), nc=1, ngf=32, ndf=16, z dim=64, r dim=10, λ=1, β=0.2Instance Noise: σ instance is annealed linearly from 1.0 to 0 for 1.3e5 iterations.

LR G/E/Q 5e-5, D 5e-6 iterations 2e5CelebA RMSProp(momentum=0.9), nc=3, ngf=64, ndf=64, z dim=500, r dim=15, λ=1, β=0.2838LR G/E/Q 5e-5, D 5e-7 iterations 2.5e5We summarize some implementation details of the models in our experiments on dSprites dataset, 3D Chairs, and CelebA datasets.

TAB3 shows the base architectures of IB-GAN for the generator, discriminator, and encoder, while TAB4 shows those of InfoGAN.

TAB2 also presents the hyperparameter settings that we use for the models in all experiments.

TAB2 for hyperparameter setting.

Generator ( DISPLAYFORM0

@highlight

Inspired by Information Bottleneck theory,  we propose a new architecture of GAN for a disentangled representation learning