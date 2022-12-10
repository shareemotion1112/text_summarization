Although variational autoencoders (VAEs) represent a widely influential deep generative model, many aspects of the underlying energy function remain poorly understood.

In particular, it is commonly believed that Gaussian encoder/decoder assumptions reduce the effectiveness of VAEs in generating realistic samples.

In this regard, we rigorously analyze the VAE objective, differentiating situations where this belief is and is not actually true.

We then leverage the corresponding insights to develop a simple VAE enhancement that requires no additional hyperparameters or sensitive tuning.

Quantitatively, this proposal produces crisp samples and stable FID scores that are actually competitive with a variety of GAN models, all while retaining desirable attributes of the original VAE architecture.

The code for our model is available at \url{https://github.com/daib13/TwoStageVAE}.

Our starting point is the desire to learn a probabilistic generative model of observable variables x ∈ χ, where χ is a r-dimensional manifold embedded in R d .

Note that if r = d, then this assumption places no restriction on the distribution of x ∈ R d whatsoever; however, the added formalism is introduced to handle the frequently encountered case where x possesses low-dimensional structure relative to a high-dimensional ambient space, i.e., r d. In fact, the very utility of generative models of continuous data, and their attendant low-dimensional representations, often hinges on this assumption BID1 .

It therefore behooves us to explicitly account for this situation.

Beyond this, we assume that χ is a simple Riemannian manifold, which means there exists a diffeomorphism ϕ between χ and R r , or more explicitly, the mapping ϕ : χ → R r is invertible and differentiable.

Denote a ground-truth probability measure on χ as µ gt such that the probability mass of an infinitesimal dx on the manifold is µ gt (dx) and χ µ gt (dx) = 1.The variational autoencoder (VAE) BID17 BID28 attempts to approximate this ground-truth measure using a parameterized density p θ (x) defined across all of R d since any underlying generative manifold is unknown in advance.

This density is further assumed to admit the latent decomposition p θ (x) = p θ (x|z)p(z)dz, where z ∈ R κ serves as a lowdimensional representation, with κ ≈ r and prior p(z) = N (z|0, I).Ideally we might like to minimize the negative log-likelihood − log p θ (x) averaged across the ground-truth measure µ gt , i.e., solve min θ χ − log p θ (x)µ gt (dx).

Unfortunately though, the required marginalization over z is generally infeasible.

Instead the VAE model relies on tractable encoder q φ (z|x) and decoder p θ (x|z) distributions, where φ represents additional trainable parameters.

The canonical VAE cost is a bound on the average negative log-likelihood given by L(θ, φ) χ {− log p θ (x) + KL [q φ (z|x)||p θ (z|x)]} µ gt (dx) ≥ χ − log p θ (x)µ gt (dx),where the inequality follows directly from the non-negativity of the KL-divergence.

Here φ can be viewed as tuning the tightness of bound, while θ dictates the actual estimation of µ gt .

Using a few standard manipulations, this bound can also be expressed as DISPLAYFORM0 which explicitly involves the encoder/decoder distributions and is conveniently amenable to SGD optimization of {θ, φ} via a reparameterization trick BID17 BID28 .

The first term in (2) can be viewed as a reconstruction cost (or a stochastic analog of a traditional autoencoder), while the second penalizes posterior deviations from the prior p(z).

Additionally, for any realizable implementation via SGD, the integration over χ must be approximated via a finite sum across training samples {x (i) } n i=1 drawn from µ gt .

Nonetheless, examining the true objective L(θ, φ) can lead to important, practically-relevant insights.

At least in principle, q φ (z|x) and p θ (x|z) can be arbitrary distributions, in which case we could simply enforce q φ (z|x) = p θ (z|x) ∝ p θ (x|z)p(z) such that the bound from (1) is tight.

Unfortunately though, this is essentially always an intractable undertaking.

Consequently, largely to facilitate practical implementation, a commonly adopted distributional assumption for continuous data is that both q φ (z|x) and p θ (x|z) are Gaussian.

This design choice has previously been cited as a key limitation of VAEs BID5 BID18 , and existing quantitative tests of generative modeling quality thus far dramatically favor contemporary alternatives such as generative adversarial networks (GAN) BID13 .

Regardless, because the VAE possesses certain desirable properties relative to GAN models (e.g., stable training BID29 , interpretable encoder/inference network BID4 , outlier-robustness BID9 , etc.), it remains a highly influential paradigm worthy of examination and enhancement.

In Section 2 we closely investigate the implications of VAE Gaussian assumptions leading to a number of interesting diagnostic conclusions.

In particular, we differentiate the situation where r = d, in which case we prove that recovering the ground-truth distribution is actually possible iff the VAE global optimum is reached, and r < d, in which case the VAE global optimum can be reached by solutions that reflect the ground-truth distribution almost everywhere, but not necessarily uniquely so.

In other words, there could exist alternative solutions that both reach the global optimum and yet do not assign the same probability measure as µ gt .Section 3 then further probes this non-uniqueness issue by inspecting necessary conditions of global optima when r < d. This analysis reveals that an optimal VAE parameterization will provide an encoder/decoder pair capable of perfectly reconstructing all x ∈ χ using any z drawn from q φ (z|x).

Moreover, we demonstrate that the VAE accomplishes this using a degenerate latent code whereby only r dimensions are effectively active.

Collectively, these results indicate that the VAE global optimum can in fact uniquely learn a mapping to the correct ground-truth manifold when r < d, but not necessarily the correct probability measure within this manifold, a critical distinction.

Next we leverage these analytical results in Section 4 to motivate an almost trivially-simple, twostage VAE enhancement for addressing typical regimes when r < d. In brief, the first stage just learns the manifold per the allowances from Section 3, and in doing so, provides a mapping to a lower dimensional intermediate representation with no degenerate dimensions that mirrors the r = d regime.

The second (much smaller) stage then only needs to learn the correct probability measure on this intermediate representation, which is possible per the analysis from Section 2.

Experiments from Sections 5 and 6 empirically corroborate motivational theory and reveal that the proposed two-stage procedure can generate high-quality samples, reducing the blurriness often attributed to VAE models in the past BID11 BID21 .

And to the best of our knowledge, this is the first demonstration of a VAE pipeline that can produce stable FID scores, an influential recent metric for evaluating generated sample quality BID16 , that are comparable to GAN models under neutral testing conditions.

Moreover, this is accomplished without additional penalties, cost function modifications, or sensitive tuning parameters.

Finally, an extended version of this work can be found in BID8 ).

There we include additional results, consideration of disentangled representations, as well as a comparative discussion of broader VAE modeling paradigms such as those involving normalizing flows or parameterized families for p(z).

Conventional wisdom suggests that VAE Gaussian assumptions will introduce a gap between L(θ, φ) and the ideal negative log-likelihood χ − log p θ (x)µ gt (dx), compromising efforts to learn the ground-truth measure.

However, we will now argue that this pessimism is in some sense premature.

In fact, we will demonstrate that, even with the stated Gaussian distributions, there exist parameters φ and θ that can simultaneously: (i) Globally optimize the VAE objective and, (ii) Recover the ground-truth probability measure in a certain sense described below.

This is possible because, at least for some coordinated values of φ and θ, q φ (z|x) and p θ (z|x) can indeed become arbitrarily close.

Before presenting the details, we first formalize a κ-simple VAE, which is merely a VAE model with explicit Gaussian assumptions and parameterizations:Definition 1 A κ-simple VAE is defined as a VAE model with dim[z] = κ latent dimensions, the Gaussian encoder q φ (z|x) = N (z|µ z , Σ z ), and the Gaussian decoder p θ (x|z) = N (x|µ x , Σ x ).

Moreover, the encoder moments are defined as µ z = f µz (x; φ) and Σ z = S z S z with S z = f Sz (x; φ).

Likewise, the decoder moments are µ x = f µx (z; θ) and Σ x = γI. Here γ > 0 is a tunable scalar, while f µz , f Sz and f µx specify parameterized differentiable functional forms that can be arbitrarily complex, e.g., a deep neural network.

Equipped with these definitions, we will now demonstrate that a κ-simple VAE, with κ ≥ r, can achieve the optimality criteria (i) and (ii) from above.

In doing so, we first consider the simpler case where r = d, followed by the extended scenario with r < d. The distinction between these two cases turns out to be significant, with practical implications to be explored in Section 4.

We first analyze the specialized situation where DISPLAYFORM0 represents the ground-truth probability density with respect to the standard Lebesgue measure in Euclidean space.

Given these considerations, the minimal possible value of (1) will necessarily occur if DISPLAYFORM1 This follows because by VAE design it must be that L(θ, φ) ≥ − p gt (x) log p gt (x)dx, and in the present context, this lower bound is achievable iff the conditions from (3) hold.

Collectively, this implies that the approximate posterior produced by the encoder q φ (z|x) is in fact perfectly matched to the actual posterior p θ (z|x), while the corresponding marginalized data distribution p θ (x) is perfectly matched the ground-truth density p gt (x) as desired.

Perhaps surprisingly, a κ-simple VAE can actually achieve such a solution:Theorem 1 Suppose that r = d and there exists a density p gt (x) associated with the ground-truth measure µ gt that is nonzero everywhere on R d .

1 .

Then for any κ ≥ r, there is a sequence of κ-simple VAE model parameters {θ * DISPLAYFORM2 All the proofs can be found in BID8 .

So at least when r = d, the VAE Gaussian assumptions need not actually prevent the optimal ground-truth probability measure from being recovered, as long as the latent dimension is sufficiently large (i.e., κ ≥ r).

And contrary to popular notions, a richer class of distributions is not required to achieve this.

Of course Theorem 1 only applies to a restricted case that excludes d > r; however, later we will demonstrate that a key consequence of this result can nonetheless be leveraged to dramatically enhance VAE performance.

When r < d, additional subtleties are introduced that will be unpacked both here and in the sequel.

To begin, if both q φ (z|x) and p θ (x|z) are arbitrary/unconstrained (i.e., not necessarily Gaussian), then inf φ,θ L(θ, φ) = −∞. To achieve this global optimum, we need only choose φ such that q φ (z|x) = p θ (z|x) (minimizing the KL term from (1)) while selecting θ such that all probability mass collapses to the correct manifold χ.

In this scenario the density p θ (x) will become unbounded on χ and zero elsewhere, such that χ − log p θ (x)µ gt (dx) will approach negative infinity.

But of course the stated Gaussian assumptions from the κ-simple VAE model could ostensibly prevent this from occurring by causing the KL term to blow up, counteracting the negative loglikelihood factor.

We will now analyze this case to demonstrate that this need not happen.

Before proceeding to this result, we first define a manifold densityp gt (x) as the probability density (assuming it exists) of µ gt with respect to the volume measure of the manifold χ.

If d = r then this volume measure reduces to the standard Lebesgue measure in R d andp gt (x) = p gt (x); however, when d > r a density p gt (x) defined in R d will not technically exist, whilep gt (x) is still perfectly well-defined.

We then have the following: Theorem 2 Assume r < d and that there exists a manifold densityp gt (x) associated with the ground-truth measure µ gt that is nonzero everywhere on χ.

Then for any κ ≥ r, there is a sequence of κ-simple VAE model parameters {θ * t , φ * t } such that DISPLAYFORM0 DISPLAYFORM1 for all measurable sets A ⊆ R d with µ gt (∂A ∩ χ) = 0, where ∂A is the boundary of A.Technical details notwithstanding, Theorem 2 admits a very intuitive interpretation.

First, (5) directly implies that the VAE Gaussian assumptions do not prevent minimization of L(θ, φ) from converging to minus infinity, which can be trivially viewed as a globally optimum solution.

Furthermore, based on (6), this solution can be achieved with a limiting density estimate that will assign a probability mass to most all measurable subsets of R d that is indistinguishable from the groundtruth measure (which confines all mass to χ).

Hence this solution is more-or-less an arbitrarily-good approximation to µ gt for all practical purposes.

Regardless, there is an absolutely crucial distinction between Theorem 2 and the simpler case quantified by Theorem 1.

Although both describe conditions whereby the κ-simple VAE can achieve the minimal possible objective, in the r = d case achieving the lower bound (whether the specific parameterization for doing so is unique or not) necessitates that the ground-truth probability measure has been recovered almost everywhere.

But the r < d situation is quite different because we have not ruled out the possibility that a different set of parameters {θ, φ} could push L(θ, φ) to −∞ and yet not achieve (6).

In other words, the VAE could reach the lower bound but fail to closely approximate µ gt .

And we stress that this uniqueness issue is not a consequence of the VAE Gaussian assumptions per se; even if q φ (z|x) were unconstrained the same lack of uniqueness can persist.

Rather, the intrinsic difficulty is that, because the VAE model does not have access to the groundtruth low-dimensional manifold, it must implicitly rely on a density p θ (x) defined across all of R d as mentioned previously.

Moreover, if this density converges towards infinity on the manifold during training without increasing the KL term at the same rate, the VAE cost can be unbounded from below, even in cases where (6) is not satisfied, meaning incorrect assignment of probability mass.

To conclude, the key take-home message from this section is that, at least in principle, VAE Gaussian assumptions need not actually be the root cause of any failure to recover ground-truth distributions.

Instead we expose a structural deficiency that lies elsewhere, namely, the non-uniqueness of solutions that can optimize the VAE objective without necessarily learning a close approximation to µ gt .

But to probe this issue further and motivate possible workarounds, it is critical to further disambiguate these optimal solutions and their relationship with ground-truth manifolds.

This will be the task of Section 3, where we will explicitly differentiate the problem of locating the correct groundtruth manifold, from the task of learning the correct probability measure within the manifold.

Note that the only comparable prior work we are aware of related to the results in this section comes from BID10 , where the implications of adopting Gaussian encoder/decoder pairs in the specialized case of r = d = 1 are briefly considered.

Moreover, the analysis there requires additional much stronger assumptions than ours, namely, that p gt (x) should be nonzero and infinitely differentiable everywhere in the requisite 1D ambient space.

These requirements of course exclude essentially all practical usage regimes where d = r > 1 or d > r, or when ground-truth densities are not sufficiently smooth.

We will now more closely examine the properties of optimal κ-simple VAE solutions, and in particular, the degree to which we might expect them to at least reflect the true χ, even if perhaps not the correct probability measure µ gt defined within χ.

To do so, we must first consider some necessary conditions for VAE optima:Theorem 3 Let {θ * γ , φ * γ } denote an optimal κ-simple VAE solution (with κ ≥ r) where the decoder variance γ is fixed (i.e., it is the sole unoptimized parameter).

Moreover, we assume that µ gt is not a Gaussian distribution when d = r.3 Then for any γ > 0, there exists a γ < γ such that DISPLAYFORM0 This result implies that we can always reduce the VAE cost by choosing a smaller value of γ, and hence, if γ is not constrained, it must be that γ → 0 if we wish to minimize (2).

Despite this necessary optimality condition, in existing practical VAE applications, it is standard to fix γ ≈ 1 during training.

This is equivalent to simply adopting a non-adaptive squared-error loss for the decoder and, at least in part, likely contributes to unrealistic/blurry VAE-generated samples BID3 .

Regardless, there are more significant consequences of this intrinsic favoritism for γ → 0, in particular as related to reconstructing data drawn from the ground-truth manifold χ:Theorem 4 Applying the same conditions and definitions as in Theorem 3, then for all x drawn from µ gt , we also have that DISPLAYFORM1 By design any random draw z ∼ q φ * γ (z|x) can be expressed as f µz (x; φ * γ ) + f Sz (x; φ * γ )ε for some ε ∼ N (ε|0, I).

From this vantage point then, (7) effectively indicates that any x ∈ χ will be perfectly reconstructed by the VAE encoder/decoder pair at globally optimal solutions, achieving this necessary condition despite any possible stochastic corrupting factor f Sz (x; φ * γ )ε.

But still further insights can be obtained when we more closely inspect the VAE objective function behavior at arbitrarily small but explicitly nonzero values of γ.

In particular, when κ = r (meaning z has no superfluous capacity), Theorem 4 and attendant analyses in BID8 ultimately imply that the squared eigenvalues of f Sz (x; φ * γ ) will become arbitrarily small at a rate proportional to γ, meaning 1 √ γ f Sz (x; φ * γ ) ≈ O(1) under mild conditions.

It then follows that the VAE data term integrand from (2), in the neighborhood around optimal solutions, behaves as DISPLAYFORM2 This expression can be derived by excluding the higher-order terms of a Taylor series approximation of f µx f µz (x; φ * γ ) + f Sz (x; φ * γ )ε; θ * γ around the point f µz (x; φ * γ ), which will be relatively tight under the stated conditions.

But because 2E q φ * γ (z|x) 1 DISPLAYFORM3 .

So in this sense (8) cannot be significantly lowered further.

This observation is significant when we consider the inclusion of addition latent dimensions by allowing κ > r. Clearly based on the analysis above, adding dimensions to z cannot improve the value of the VAE data term in any meaningful way.

However, it can have a detrimental impact on the the KL regularization factor in the γ → 0 regime, where DISPLAYFORM4 Herer denotes the number of eigenvalues {λ j (γ)} κ j=1 of f Sz (x; φ * γ ) (or equivalently Σ z ) that satisfy λ j (γ) → 0 if γ → 0.r can be viewed as an estimate of how many low-noise latent dimensions the VAE model is preserving to reconstruct x. Based on (9), there is obvious pressure to maker as small as possible, at least without disrupting the data fit.

The smallest possible value isr = r, since it is not difficult to show that any value below this will contribute consequential reconstruction errors, causing 2E q φ * γ (z|x) 1 DISPLAYFORM5 to grow at a rate of Ω 1 γ , pushing the entire cost function towards infinity.

4 Therefore, in the neighborhood of optimal solutions the VAE will naturally seek to produce perfect reconstructions using the fewest number of clean, low-noise latent dimensions, meaning dimensions whereby q φ (z|x) has negligible variance.

For superfluous dimensions that are unnecessary for representing x, the associated encoder variance in these directions can be pushed to one.

This will optimize KL [q φ (z|x)||p(z)] along these directions, and the decoder can selectively block the residual randomness to avoid influencing the reconstructions per Theorem 4.

So in this sense the VAE is capable of learning a minimal representation of the ground-truth manifold χ when r < κ.

But we must emphasize that the VAE can learn χ independently of the actual distribution µ gt within χ.

Addressing the latter is a completely separate issue from achieving the perfect reconstruction error defined by Theorem 4.

This fact can be understood within the context of a traditional PCAlike model, which is perfectly capable of learning a low-dimensional subspace containing some training data without actually learning the distribution of the data within this subspace.

The central issue is that there exists an intrinsic bias associated with the VAE objective such that fitting the distribution within the manifold will be completely neglected whenever there exists the chance for even an infinitesimally better approximation of the manifold itself.

Stated differently, if VAE model parameters have learned a near optimal, parsimonious latent mapping onto χ using γ ≈ 0, then the VAE cost will scale as (d − r) log γ regardless of µ gt .

Hence there remains a huge incentive to reduce the reconstruction error still further, allowing γ to push even closer to zero and the cost closer to −∞.

And if we constrain γ to be sufficiently large so as to prevent this from happening, then we risk degrading/blurring the reconstructions and widening the gap between q φ (z|x) and p θ (z|x), which can also compromise estimation of µ gt .

Fortunately though, as will be discussed next there is a convenient way around this dilemma by exploiting the fact that this dominanting (d − r) log γ factor goes away when d = r.

Sections 2 and 3 have exposed a collection of VAE properties with useful diagnostic value in and of themselves.

But the practical utility of these results, beyond the underappreciated benefit of learning γ, warrant further exploration.

In this regard, suppose we wish to develop a generative model of high-dimensional data x ∈ χ where unknown low-dimensional structure is significant (i.e., the r < d case with r unknown).

The results from Section 3 indicate that the VAE can partially handle this situation by learning a parsimonious representation of low-dimensional manifolds, but not necessarily the correct probability measure µ gt within such a manifold.

In quantitative terms, this means that a decoder p θ (x|z) will map all samples from an encoder q φ (z|x) to the correct manifold such that the reconstruction error is negligible for any x ∈ χ.

But if the measure µ gt on χ has not been accurately estimated, then DISPLAYFORM0 where q φ (z) is sometimes referred to as the aggregated posterior BID25 .

In other words, the distribution of the latent samples drawn from the encoder distribution, when averaged across the training data, will have lingering latent structure that is errantly incongruous with the original isotropic Gaussian prior.

This then disrupts the pivotal ancestral sampling capability of the VAE, implying that samples drawn from N (z|0, I) and then passed through the decoder p θ (x|z) will not closely approximate µ gt .

Fortunately, our analysis suggests the following two-stage remedy: DISPLAYFORM1 , train a κ-simple VAE, with κ ≥ r, to estimate the unknown r-dimensional ground-truth manifold χ embedded in R d using a minimal number of active latent dimensions.

Generate latent samples {z DISPLAYFORM2 .

By design, these samples will be distributed as q φ (z), but likely not N (z|0, I).

C γ + log γ = ∞ for any C > 0.2.

Train a second κ-simple VAE, with independent parameters {θ , φ } and latent representation u, to learn the unknown distribution q φ (z), i.e., treat q φ (z) as a new ground-truth distribution and use samples {z DISPLAYFORM0 to learn it.

3.

Samples approximating the original ground-truth µ gt can then be formed via the extended ancestral process u ∼ N (u|0, I), z ∼ p θ (z|u), and finally x ∼ p θ (x|z).The efficacy of the second-stage VAE from above is based on the following.

If the first stage was successful, then even though they will not generally resemble N (z|0, I), samples from q φ (z) will nonetheless have nonzero measure across the full ambient space R κ .

If κ = r, this occurs because the entire latent space is needed to represent an r-dimensional manifold, and if κ > r, then the extra latent dimensions will be naturally filled in via randomness introduced along dimensions associated with nonzero eigenvalues of the decoder covariance Σ z per the analysis in Section 3.Consequently, as long as we set κ ≥ r, the operational regime of the second-stage VAE is effectively equivalent to the situation described in Section 2.1 where the manifold dimension is equal to the ambient dimension.5 And as we have already shown there via Theorem 1, the VAE can readily handle this situation, since in the narrow context of the second-stage VAE, d = r = κ, the troublesome (d − r) log γ factor becomes zero, and any globally minimizing solution is uniquely matched to the new ground-truth distribution q φ (z).

Consequently, the revised aggregated posterior q φ (u) produced by the second-stage VAE should now closely resemble N (u|0, I).

And importantly, because we generally assume that d κ ≥ r, we have found that the second-stage VAE can be quite small.

It should also be emphasized that concatenating the two VAE stages and jointly training does not generally improve the performance.

If trained jointly the few extra second-stage parameters can simply be hijacked by the dominant influence of the first stage reconstruction term and forced to work on an incrementally better fit of the manifold rather than addressing the critical mismatch between q φ (z) and N (u|0, I).

This observation can be empirically tested, which we have done in multiple ways.

For example, we have tried fusing the respective encoders and decoders from the first and second stages to train what amounts to a slightly more complex single VAE model.

We have also tried merging the two stages including the associated penalty terms.

In both cases, joint training does not help at all as expected, with average performance no better than the first stage VAE (which contains the vast majority of parameters).

Consequently, although perhaps counterintuitive, separate training of these two VAE stages is actually critical to achieving high quality results as will be demonstrated next.

We first present quantitative evaluation of novel generated samples using the large-scale testing protocol of GAN models from BID24 .

In this regard, GANs are well-known to dramatically outperform existing VAE approaches in terms of the Fréchet Inception Distance (FID) score BID16 and related quantitative metrics.

For fair comparison, BID24 ) adopted a common neutral architecture for all models, with generator and discriminator networks based on InfoGAN BID6 ; the point here is standardized comparisons, not tuning arbitrarily-large networks to achieve the lowest possible absolute FID values.

We applied the same architecture to our first-stage VAE decoder and encoder networks respectively for direct comparison.

For the lowdimensional second-stage VAE we used small, 3-layer networks contributing negligible additional parameters beyond the first stage (see BID8 for further design details).We evaluated our proposed VAE pipeline, henceforth denoted as 2-Stage VAE, against three baseline VAE models differing only in the decoder output layer: a Gaussian layer with fixed γ, a Gaussian layer with a learned γ, and a cross-entropy layer as has been adopted in several previous applications involving images BID7 .

We also tested the Gaussian decoder VAE model (with learned γ) combined with an encoder augmented with normalizing flows BID27 , as well as the recently proposed Wasserstein autoencoder (WAE) BID29 which maintains a VAE-like structure.

All of these models were adapted to use the same neutral architecture from BID24 .

Note also that the WAE includes two variants, referred to as WAE-MMD and WAE-GAN because different Maximum Mean Discrepancy (MMD) and GAN regularization factors are involved.

We conduct experiments using the former because it does not involve potentially-unstable adversarial training, consistent with the other VAE baselines.

6 Additionally, we present results from BID24 involving numerous competing GAN models, including MM GAN BID13 , WGAN , WGAN-GP (Gulrajani et al., 2017) , NS GAN BID12 , DRAGAN BID19 , LS GAN BID26 and BEGAN BID2 .

Testing is conducted across four significantly different datasets: MNIST (LeCun et al., 1998) , Fashion MNIST BID30 , CIFAR-10 (Krizhevsky & Hinton, 2009 ) and CelebA BID23 .For each dataset we executed 10 independent trials and report the mean and standard deviation of the FID scores in Table 1 .

7 No effort was made to tune VAE training hyperparameters (e.g., learning rates, etc.); rather a single generic setting was first agnostically selected and then applied to all VAE-like models (including the WAE-MMD).

As an analogous baseline, we also report the value of the best GAN model for each dataset when trained using suggested settings from the authors; no single model was optimal across all datasets, so these values represent the best performance from different, dataset-dependent GANs.

Even so, our single 2-Stage VAE is still better on two of four datasets, and in aggregate, better than any individual GAN model.

For example, when averaged across datasets, the mean FID score for any individual GAN trained with suggested settings was always approximately 45 or higher (see BID24 [ Figure 4 ]), while our analogous 2-Stage VAE maintained a mean below 40.

The other VAE baselines were not competitive.

Note also that the relatively poor performance of the WAE-MMD on MNIST and Fashion MNIST data can be attributed to the sensitivity of this approach to the value of κ, which for consistency with other models was fixed at κ = 64 for all experiments.

This value is likely much larger than actually needed for these simpler data types (meaning r 64), and the WAE-MMD model can potentially be more reliant on having some κ ≈ r.

For head-to-head empirical tests of robustness to κ, please see BID8 .

Table 1 also displays FID scores from GAN models evaluated using hyperparameters obtained from a large-scale search executed independently across each dataset to achieve the best results; 100 settings per model per dataset, plus an optimal, data-dependent stopping criteria as described in BID24 .

Within this broader paradigm, cases of severe mode collapse were omitted when computing final GAN FID averages.

Despite these considerable GAN-specific advantages, the FID performance of the default 2-Stage VAE is well within the range of the heavily-optimized GAN models for each dataset unlike the other VAE baselines.

Overall then, these results represent the first demonstration of a VAE pipeline capable of competing with GANs in the arena of generated sample quality.

Additionally, representative samples produced using our 2-Stage VAE model can be found in BID8 .Beyond the neutral testing platform from BID24 , we also consider additional comparisons using the architecture and experimental setup from BID29 explicitly designed for applying WAE models to CelebA data.

In particular, we adopt the exact same encoder-decoder networks as the WAE models, and train using the same number of epochs.

We do not tune any hyperparameters whatsoever, and apply the same small second-stage VAE as used in previous experiments.

As before, the second-stage size is a small fraction of the first stage, so any benefit is not simply the consequence of a larger network structure.

Results are reported in Table 2 , where the 2-Stage VAE even outperforms the WAE-GAN model, which has the advantage of adversarial training tuned for this combination of data and network architecture.

The true test of any theoretical contribution is the degree to which it leads to useful, empiricallytestable predictions about behavior in real-world settings.

In the present context, although our theory from Sections 2 and 3 involves some unavoidable simplifying assumptions, it nonetheless makes predictions that can be tested under practically-relevant conditions where these assumptions may MNIST Fashion CIFAR-10 CelebA MM GAN 9.8 ± 0.9 29.6 ± 1.6 72.7 ± 3.6 65.6 ± 4.2 NS GAN 6.8 ± 0.5 26.5 ± 1.6 58.5 ± 1.9 55.0 ± 3.3 optimized, LSGAN 7.8 ± 0.6 30.7 ± 2.2 87.1 ± 47.5 53.9 ± 2.8 data-dependent WGAN 6.7 ± 0.4 21.5 ± 1.6 55.2 ± 2.3 41.3 ± 2.0 settings WGAN GP 20.3 ± 5.0 24.5 ± 2.1 55.8 ± 0.9 30.3 ± 1.0 DRAGAN 7.6 ± 0.4 27.7 ± 1.2 69.8 ± 2.0 42.3 ± 3.0 BEGAN 13.1 ± 1.0 22.9 ± 0.9 71.4 ± 1.6 38.9 ± 0.9 Best GAN ∼ 10 ∼ 32 ∼ 70 ∼ 49 VAE (cross-entr.)16.6 ± 0.4 43.6 ± 0.7 106.0 ± 1.0 53.3 ± 0.6 default VAE (fixed γ) 52.0 ± 0.6 84.6 ± 0.9 160.5 ± 1.1 55.9 ± 0.6 settings VAE (learned γ) 54.5 ± 1.0 60.0 ± 1.1 76.7 ± 0.8 60.5 ± 0.6 VAE + Flow 54.8 ± 2.8 62.1 ± 1.6 81.2 ± 2.0 65.7 ± 2.8 WAE-MMD 115.0 ± 1.1 101.7 ± 0.8 80.9 ± 0.4 62.9 ± 0.8 2-Stage VAE (ours) 12.6 ± 1.5 29.3 ± 1.0 72.9 ± 0.9 44.4 ± 0.7 Table 1 : FID score comparisons using neutral architecture.

For all GAN-based models listed in the top section of the table, reported values represent the optimal FID obtained across a large-scale hyperparameter search conducted separately for each dataset BID24 .

Outlier cases (e.g., severe mode collapse) were omitted, which would have otherwise increased these GAN FID scores.

In the lower section of the table, the label Best GAN indicates the lowest FID produced across all GAN approaches for each dataset when trained using settings suggested by original authors; these approximate values were extracted from BID24 [ Figure 4 ].

For the VAE results (including WAE), only a single default setting was adopted across all datasets and models (no tuning whatsoever), and no cases of mode collapse were removed.

Note that specialized architectures and/or random seed optimization can potentially improve the FID for all models reported here.

VAE WAE-MMD WAE-GAN 2-Stage VAE (ours) CelebA FID 63 55 42 34 Table 2 : FID scores on CelebA data obtained using the network structure and training protocol from BID29 .

For the 2-Stage VAE, we apply the exact same architecture and training epochs without any tuning of hyperparameters.not strictly hold.

We now present the results of such tests, which provide strong confirmation of our previous analysis.

In particular, after providing validation of Theorems 3 and 4, we explicitly demonstrate that the second stage of our 2-Stage VAE model can reduce the gap between q(z) and p(z).

This theorem implies that γ will converge to zero at any global minimum of the stated VAE objective under consideration.

Figure 1a presents empirical support for this result, where indeed the decoder variance γ does tend towards zero during training (red line).

This then allows for tighter image reconstructions (dark blue curve) with lower average squared error, i.e., a better manifold fit as expected.

Figure 1b bolsters this theorem, and the attendant analysis which follows in Section 3, by showcasing the dissimilar impact of noise factors applied to different directions in the latent space before passage through the decoder mean network f µx .

In a direction where an eigenvalue λ j of Σ z is large (i.e., a superfluous dimension), a random perturbation is completely muted by the decoder as predicted.

In contrast, in directions where such eigenvalues are small (i.e., needed for representing the manifold), varying the input causes large changes in the image space reflecting reasonable movement along the correct manifold.

Reduced Mismatch between q φ (z) and p(z): Although the VAE with a learnable γ can achieve high-quality reconstructions, the associated aggregated posterior is still likely not close to a standard Gaussian distribution as implied by (10).

This mismatch then disrupts the critical ancestral sampling Figure 1: (a) The red line shows the evolution of log γ, converging close to 0 during training as expected.

The two blue curves compare the associated pixel-wise reconstruction errors with γ fixed at 1 and with a learnable γ respectively.

(b) The j-th eigenvalue of Σ z , denoted λ j , should be very close to either 0 or 1 as argued in Section 3.

When λ j is close to 0, injecting noise along the corresponding direction will cause a large variance in the reconstructed image, meaning this direction is an informative one needed for representing the manifold.

In contrast, if λ j is close to 1, the addition of noise does not make any appreciable difference in the reconstructed image, indicating that the corresponding dimension is a superfluous one that has been ignored/blocked by the decoder.

Table 3 : Maximum mean discrepancy between N (0, I) and q φ (z) (first stage); likewise for q φ (u) (second stage).process.

As we have previously argued, the proposed 2-Stage VAE has the ability to overcome this issue and achieve a standard Gaussian aggregated posterior, or at least nearly so.

As empirical evidence for this claim, FIG2 displays the singular value spectrum of latent sample matrices Z = {z (i) } n i=1 drawn from q φ (z) (first stage), and U = {u (i) } n i=1 drawn from q φ (u) (enhanced second stage).

As expected, the latter is much closer to the spectrum from an analogous i.i.d.

N (0, I) matrix.

We also used these same sample matrices to estimate the MMD metric BID14 between N (0, I) and the aggregated posterior distributions from the first and second stages in Table 3 .

Clearly the second stage has dramatically reduced the difference from N (0, I) as quantified by the MMD.

Overall, these results indicate a superior latent representation, providing high-level support for our 2-Stage VAE proposal.

It is often assumed that there exists an unavoidable trade-off between the stable training, valuable attendant encoder network, and resistance to mode collapse of VAEs, versus the impressive visual quality of images produced by GANs.

While we certainly are not claiming that our two-stage VAE model is superior to the latest and greatest GAN-based architecture in terms of the realism of generated samples, we do strongly believe that this work at least narrows that gap substantially such that VAEs are worth considering in a broader range of applications.

For further results and discussion, including consideration of broader VAE modeling paradigms and the identifiability of disentangled representations, please see BID8 .

@highlight

We closely analyze the VAE objective function and draw novel conclusions that lead to simple enhancements.

@highlight

Proposes a two-stage VAE method to generate high-quality samples and avoid blurriness.

@highlight

This paper analyzes the Gaussian VAEs.

@highlight

The paper provides a number of theoretical results on "vanilla" Gaussian Variational Auto-Encoders, which are then used to build a new algorithm called "2 stage VAEs".